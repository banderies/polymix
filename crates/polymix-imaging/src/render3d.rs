//! The 3D view of a volume: GPU ray casting (`shaders/volume3d.wgsl`) drawn into egui's render
//! pass. Modes: maximum intensity projection, shaded volume rendering (opacity follows the
//! window), or the first surface above a threshold; label surfaces (segments) are drawn in
//! front, coloured like the slice overlays.
//!
//! Mouse: left drag rotates, right drag pans, middle drag or the wheel zooms, left + right
//! drag sets the window and level (as in the slice views, with the histogram shown while
//! dragging), double-click resets the camera; with `tool_drags`, ⌘ / Ctrl + left drag is the
//! tool's (`pick` finds the voxel under the pointer). The volume is uploaded once (as
//! normalised 16-bit floats, every n-th voxel if it exceeds the GPU's limits, with the range of
//! values in each brick of 8³ texels so rays skip empty space); labels again whenever their
//! version changes while they are shown. A frame is ray cast into a texture only when what it
//! shows changes (at half size while the view moves) and shown from there in other frames.

use crate::slice_view::{LabelOverlay, ViewState};
use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, wgpu};
use egui::{Color32, Pos2, Rect, Response, Sense, Vec2};
use glam::{DMat4, DVec3, Vec3};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// Most texels uploaded per volume (16-bit each).
const MAX_TEXELS: usize = 160_000_000;
/// Edge (in texels) of the bricks whose value ranges and occupancy let rays skip empty space.
const BRICK: usize = 8;
/// Samples per texel along the ray.
const RATE: f32 = 1.5;
/// Samples per texel of a MIP while the view moves.
const MOVING_MIP_RATE: f32 = 1.0;
/// Pixels per texel from which a moving view is drawn at half size: below it, the rays of a
/// half-size frame would lie more than a texel apart and read the volume so scattered that
/// they cost more than the pixels saved (a four-up MIP with segments: 26 ms at half size,
/// 16 ms in full).
const HALF_SIZE_MAGNIFICATION: f32 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode3d {
    /// Maximum intensity projection.
    Mip,
    /// Shaded volume rendering: opacity rises across the window.
    Volume,
    /// The first surface above the threshold.
    Surface,
}

impl Mode3d {
    pub const ALL: [Mode3d; 3] = [Mode3d::Mip, Mode3d::Volume, Mode3d::Surface];

    pub fn label(self) -> &'static str {
        match self {
            Mode3d::Mip => "MIP",
            Mode3d::Volume => "Volume",
            Mode3d::Surface => "Surface",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    inv_view_proj: [[f32; 4]; 4],
    world_to_tex: [[f32; 4]; 4],
    grad_to_world: [[f32; 4]; 4],
    dims: [f32; 4],
    window: [f32; 4],
    labels: [f32; 4],
    background: [f32; 4],
    light: [f32; 4],
    bricks: [f32; 4],
    palette: [[f32; 4]; 256],
}

/// The ray-casting pipeline that draws `u`: one per mode, with segments shown or not.
fn pipeline_slot(u: &Uniforms) -> usize {
    (u.window[3] as usize).min(2) * 2 + (u.labels[0] > 0.5) as usize
}

/// Where a frame drawn into a view's cache texture goes ([`Draw::paint`]).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
struct BlitUniforms {
    origin: [f32; 2],
    size: [f32; 2],
    drawn: [f32; 2],
    cache_size: [f32; 2],
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct VolumeView3d {
    pub mode: Mode3d,
    /// Surface threshold in the volume's units (`None`: the middle of the window).
    pub threshold: Option<f64>,
    /// Opacity of volume rendering (1 = default).
    pub opacity: f32,
    pub show_labels: bool,
    /// Smoothing of the segments' surfaces: passes of a 1-2-1 blur over their occupancy
    /// (0 shows the voxels' steps).
    pub label_smoothing: u8,
    /// A tool takes ⌘ / Ctrl + left drags (they do not turn the camera).
    pub tool_drags: bool,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pan: Vec3,
    /// A left + right drag (window / level) is in progress, until both buttons are up.
    windowing: bool,
    /// The camera when the buttons went down (put back if the drag turns into window / level).
    grab: Option<(f32, f32, Vec3)>,
    id: u64,
    /// Kept alive with the view; the GPU side drops its textures once it is gone.
    token: Arc<()>,
    /// What the last frames drew, to tell whether the view is moving.
    seen: Option<Box<Seen>>,
}

/// The uniforms of the last frame drawn and of the frame before it.
struct Seen {
    frame: u64,
    this: Uniforms,
    before: Option<Uniforms>,
}

impl Default for VolumeView3d {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU side of one view.
struct ViewGpu {
    alive: Weak<()>,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
    volume: wgpu::Texture,
    labels: wgpu::Texture,
    occupancy: wgpu::Texture,
    value_bricks: wgpu::Texture,
    occupancy_bricks: wgpu::Texture,
    /// The volume uploaded, held weakly: its allocation stays, so no other volume can come to
    /// have its address while this lives.
    source: Weak<LoadedVolume>,
    /// The labels' version and the smoothing they were uploaded with.
    label_key: Option<(u64, u8)>,
    /// Texels per axis and the voxel stride they were sampled with.
    tex_dims: [u32; 3],
    stride: usize,
    /// Counts changes of the textures (and of the bind group), which the cached frame was
    /// drawn from.
    gen: u64,
    cache: Option<Cache>,
}

/// A view's last frame, drawn again only when its key changes.
struct Cache {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    blit: wgpu::Buffer,
    bind: wgpu::BindGroup,
    /// What the texture holds: drawn with these uniforms, at this texture generation, place,
    /// size and scale.
    uniforms: Box<Uniforms>,
    key: Option<(u64, [u32; 2], [u32; 2], bool)>,
    /// The blit uniforms last written.
    placed: Option<BlitUniforms>,
    /// Whether this frame's Draw shows the texture (and at half size): set by prepare().
    shown: Option<bool>,
}

/// Shared by every 3D view, kept in egui-wgpu's callback resources.
struct Gpu {
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    /// Ray casting for each [`pipeline_slot`], made when first needed.
    pipelines: [Option<wgpu::RenderPipeline>; 6],
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Showing cached frames; `None` on an sRGB target (whose frames are drawn directly, as a
    /// round trip through an sRGB texture could change them).
    blitter: Option<Blitter>,
    views: HashMap<u64, ViewGpu>,
}

struct Blitter {
    layout: wgpu::BindGroupLayout,
    /// 1:1 and scaled up from half size.
    exact: wgpu::RenderPipeline,
    scaled: wgpu::RenderPipeline,
}

/// What [`VolumeView3d::sync`] put on the GPU that the uniforms depend on.
#[derive(Clone, Copy)]
struct Uploaded {
    tex_dims: [u32; 3],
    stride: usize,
    /// Frames can be drawn into a texture first (cached, or at half size while moving).
    offscreen: bool,
}

impl Gpu {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("volume3d.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/volume3d.wgsl").into()) });
        let tex = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
            count: None,
        };
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume3d"),
            entries: &[uniform(0), tex(1, float), sampler_entry(2), tex(3, float), tex(4, wgpu::TextureSampleType::Uint), tex(5, float), tex(6, float)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("volume3d"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume3d"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let blitter = (!format.is_srgb()).then(|| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("volume3d.blit"),
                entries: &[
                    uniform(10),
                    wgpu::BindGroupLayoutEntry { binding: 11, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: float, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                    sampler_entry(12),
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("volume3d.blit"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipeline = |entry: &str| render_pipeline(device, &shader, &pl, format, entry, &[]);
            Blitter { exact: pipeline("blit"), scaled: pipeline("blit_scaled"), layout }
        });
        Self { shader, pipeline_layout, format, pipelines: Default::default(), layout, sampler, blitter, views: HashMap::new() }
    }

    /// Make the ray-casting pipeline for `slot` if it is not there yet.
    fn ensure_pipeline(&mut self, device: &wgpu::Device, slot: usize) {
        let Self { shader, pipeline_layout, format, pipelines, .. } = self;
        pipelines[slot].get_or_insert_with(|| ray_pipeline(device, shader, pipeline_layout, *format, slot, true));
    }
}

/// The ray caster specialised for [`pipeline_slot`] `slot`; `skip` off walks every sample (to
/// check that skipping changes nothing).
fn ray_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, format: wgpu::TextureFormat, slot: usize, skip: bool) -> wgpu::RenderPipeline {
    let constants = [("MODE", (slot / 2) as f64), ("LABELS", (slot % 2) as f64), ("SKIP", skip as u8 as f64)];
    render_pipeline(device, shader, layout, format, "fs", &constants)
}

/// A full-screen triangle with fragment entry point `entry`.
fn render_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, format: wgpu::TextureFormat, entry: &str, constants: &[(&str, f64)]) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("volume3d"),
        layout: Some(layout),
        vertex: wgpu::VertexState { module: shader, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn texture_3d(device: &wgpu::Device, label: &str, dims: [u32; 3], format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Upload `data` (`bpp` bytes per texel, x fastest) in slabs of whole z slices.
fn write_3d(queue: &wgpu::Queue, t: &wgpu::Texture, dims: [u32; 3], bpp: u32, data: &[u8]) {
    let slice = (dims[0] * dims[1] * bpp) as usize;
    let per_slab = ((64 << 20) / slice.max(1)).max(1) as u32;
    let mut z = 0;
    while z < dims[2] {
        let n = per_slab.min(dims[2] - z);
        let bytes = &data[z as usize * slice..(z + n) as usize * slice];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: t, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z }, aspect: wgpu::TextureAspect::All },
            bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(dims[0] * bpp), rows_per_image: Some(dims[1]) },
            wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: n },
        );
        z += n;
    }
}

/// IEEE half-float bits of `x` in [0, 1] (no infinities or NaNs needed).
fn half(x: f32) -> u16 {
    let x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
    let b = x.to_bits();
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return 0;
    }
    let mant = (b >> 13) & 0x3ff;
    // Round to nearest.
    let round = (b >> 12) & 1;
    (((exp as u32) << 10 | mant) + round) as u16
}

/// Voxel stride so the texture fits the GPU's 3D size limit and [`MAX_TEXELS`].
fn stride_for(dims: [usize; 3], max_dim: u32) -> usize {
    let mut s = 1;
    loop {
        let t: Vec<usize> = dims.iter().map(|d| d.div_ceil(s)).collect();
        if t.iter().all(|&d| d <= max_dim as usize) && t.iter().product::<usize>() <= MAX_TEXELS {
            return s;
        }
        s += 1;
    }
}

/// The texels of `vol`'s texture: its values normalised to [0, 1] as half floats, every
/// `stride`-th voxel.
fn volume_halves(vol: &LoadedVolume, td: [u32; 3], stride: usize) -> Vec<u16> {
    let (lo, hi) = vol.range;
    let scale = if hi > lo { 1.0 / (hi - lo) } else { 0.0 };
    if stride == 1 {
        vol.values[..vol.voxel_count()].par_iter().map(|&x| half((x - lo) * scale)).collect()
    } else {
        sampled(vol, td, stride, |i| half((vol.values[i] - lo) * scale))
    }
}

/// Bricks per axis over a texture of `td` texels.
fn brick_dims(td: [u32; 3]) -> [u32; 3] {
    td.map(|d| d.div_ceil(BRICK as u32))
}

/// `join` over each brick of [`BRICK`]³ texels of `v` (x fastest, `td` texels) and one more
/// texel on each side, which is all that trilinear samples inside the brick read; one result
/// per brick, x fastest. Reduced along z (texel by texel over whole planes), then y, then x.
fn brick_reduce<T: Copy + Send + Sync>(v: &[T], td: [u32; 3], join: impl Fn(T, T) -> T + Sync) -> Vec<T> {
    let [nx, ny, nz] = td.map(|d| d as usize);
    let [bx, by, bz] = brick_dims(td).map(|d| d as usize);
    let plane = nx * ny;
    // The texels brick `b` reaches along an axis of `n` (never empty).
    let reach = |b: usize, n: usize| (b * BRICK).saturating_sub(1)..((b + 1) * BRICK + 1).min(n);
    let fold_into = |acc: &mut [T], src: &[T]| {
        for (a, &s) in acc.iter_mut().zip(src) {
            *a = join(*a, s);
        }
    };
    let mut out = vec![v[0]; bx * by * bz];
    out.par_chunks_mut(bx * by).enumerate().for_each(|(b, out)| {
        let r = reach(b, nz);
        let mut planes = v[r.start * plane..][..plane].to_vec();
        for z in r.start + 1..r.end {
            fold_into(&mut planes, &v[z * plane..][..plane]);
        }
        let mut rows = planes[..nx].to_vec();
        for (b, out) in out.chunks_exact_mut(bx).enumerate() {
            let r = reach(b, ny);
            rows.copy_from_slice(&planes[r.start * nx..][..nx]);
            for y in r.start + 1..r.end {
                fold_into(&mut rows, &planes[y * nx..][..nx]);
            }
            for (b, o) in out.iter_mut().enumerate() {
                let r = reach(b, nx);
                *o = rows[r.start + 1..r.end].iter().fold(rows[r.start], |a, &s| join(a, s));
            }
        }
    });
    out
}

/// The smallest and largest half-float value about each brick (non-negative halves order like
/// their bits), for an Rg16Float texture.
fn value_ranges(halves: &[u16], td: [u32; 3]) -> Vec<[u16; 2]> {
    let (lo, hi) = rayon::join(|| brick_reduce(halves, td, u16::min), || brick_reduce(halves, td, u16::max));
    lo.into_iter().zip(hi).map(|(lo, hi)| [lo, hi]).collect()
}

/// The largest occupancy about each brick.
fn occupancy_maxima(occ: &[u8], td: [u32; 3]) -> Vec<u8> {
    brick_reduce(occ, td, u8::max)
}

impl VolumeView3d {
    pub fn new() -> Self {
        Self { mode: Mode3d::Volume, threshold: None, opacity: 1.0, show_labels: true, label_smoothing: 2, tool_drags: false, yaw: 0.0, pitch: 0.15, zoom: 1.0, pan: Vec3::ZERO, windowing: false, grab: None, id: NEXT_ID.fetch_add(1, Ordering::Relaxed), token: Arc::new(()), seen: None }
    }

    pub fn reset_camera(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.15;
        self.zoom = 1.0;
        self.pan = Vec3::ZERO;
    }

    /// World-space bounds of the volume: centre and radius.
    fn bounds(vol: &LoadedVolume) -> (DVec3, f64) {
        let g = &vol.geometry;
        let d = vol.dims.map(|n| n as f64 - 1.0);
        let corners = (0..8).map(|c| g.world(DVec3::new(if c & 1 > 0 { d[0] } else { 0.0 }, if c & 2 > 0 { d[1] } else { 0.0 }, if c & 4 > 0 { d[2] } else { 0.0 })));
        let pts: Vec<DVec3> = corners.collect();
        let lo = pts.iter().fold(DVec3::splat(f64::INFINITY), |a, p| a.min(*p));
        let hi = pts.iter().fold(DVec3::splat(f64::NEG_INFINITY), |a, p| a.max(*p));
        ((lo + hi) * 0.5, ((hi - lo).length() * 0.5).max(1.0))
    }

    /// Camera: (eye, target, up, vertical field of view).
    fn camera(&self, vol: &LoadedVolume) -> (Vec3, Vec3, Vec3, f32) {
        let (c, r) = Self::bounds(vol);
        let fov = 30f32.to_radians();
        let dist = (r as f32 / (fov * 0.5).sin()) * 1.05 / self.zoom;
        // Yaw 0 looks at the patient's face (from anterior), superior up.
        let dir = Vec3::new(self.yaw.sin() * self.pitch.cos(), self.yaw.cos() * self.pitch.cos(), self.pitch.sin());
        let target = c.as_vec3() + self.pan;
        (target + dir * dist, target, Vec3::Z, fov)
    }

    /// Draw the view in the space left in `ui`, with its controls in the top left corner. A
    /// left + right drag sets the window in `state`, as in the slice views.
    pub fn ui(&mut self, ui: &mut egui::Ui, render_state: Option<&egui_wgpu::RenderState>, vol: &Arc<LoadedVolume>, state: &mut ViewState, labels: Option<&LabelOverlay>) -> Response {
        let rect = ui.available_rect_before_wrap();
        let r = ui.allocate_rect(rect, Sense::click_and_drag());
        // Dark whatever the theme, as medical viewers have it.
        let bg = Color32::from_rgb(12, 14, 18);
        if let Some(d) = self.navigate(ui, &r, vol) {
            state.window = crate::nav::drag_window(state.window, d);
        }
        let Some(rs) = render_state else {
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "The 3D view needs the GPU renderer", egui::FontId::proportional(13.0), ui.visuals().weak_text_color());
            return r;
        };
        let up = match self.sync(rs, vol, labels) {
            Ok(up) => up,
            Err(e) => {
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, e, egui::FontId::proportional(13.0), ui.visuals().warn_fg_color);
                return r;
            }
        };
        let mut uniforms = self.uniforms(vol, state, labels, rect, bg, &up);
        // While the view moves (turned, panned, zoomed, windowed), a MIP takes fewer samples and
        // a magnified view is drawn at half size; the first frame after it stops is drawn in
        // full.
        let moving = self.moving(ui.ctx().cumulative_frame_nr(), &uniforms);
        let half = moving && up.offscreen && self.pixels_per_texel(vol, rect.height() * ui.ctx().pixels_per_point(), up.stride) >= HALF_SIZE_MAGNIFICATION;
        if moving {
            if self.mode == Mode3d::Mip {
                uniforms.dims[3] = MOVING_MIP_RATE;
            }
            ui.ctx().request_repaint();
        }
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, Draw { id: self.id, rect, uniforms, half }));
        self.controls(ui, rect, vol, labels.is_some());
        let key = (vol.values.as_ptr() as usize, vol.values.len(), vol.range.0.to_bits(), vol.range.1.to_bits());
        crate::histogram::show(ui, r.id, rect, rect.bottom() - 24.0, key, &vol.values, state.window, vol.invert, self.windowing);
        r
    }

    /// Pixels per texel (along the volume's finest axis) at the camera's target, in a view
    /// `height_px` pixels tall.
    fn pixels_per_texel(&self, vol: &LoadedVolume, height_px: f32, stride: usize) -> f32 {
        let (eye, target, _, fov) = self.camera(vol);
        let m = vol.geometry.matrix();
        let texel = (0..3).map(|a| m.col(a).truncate().length()).fold(f64::INFINITY, f64::min) * stride as f64;
        height_px / (2.0 * (eye - target).length() * (fov * 0.5).tan()) * texel as f32
    }

    /// Whether `u` differs from what the previous frame drew (in a second pass of the same
    /// frame, from the frame before it too).
    fn moving(&mut self, frame: u64, u: &Uniforms) -> bool {
        let seen = self.seen.get_or_insert_with(|| Box::new(Seen { frame, this: *u, before: None }));
        if seen.frame != frame {
            seen.before = Some(seen.this);
            seen.frame = frame;
        }
        seen.this = *u;
        seen.before.is_some_and(|b| bytemuck::bytes_of(&b) != bytemuck::bytes_of(u))
    }

    /// Turn, pan and zoom the camera; a left + right drag is window / level instead (its
    /// movement returned: across widens the window, down lowers the level).
    fn navigate(&mut self, ui: &egui::Ui, r: &Response, vol: &LoadedVolume) -> Option<Vec2> {
        let (primary, secondary, any, delta) = ui.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down(), i.pointer.any_down(), i.pointer.delta()));
        let active = r.is_pointer_button_down_on() || r.dragged();
        if !any {
            self.windowing = false;
            self.grab = None;
        } else if active && self.grab.is_none() {
            self.grab = Some((self.yaw, self.pitch, self.pan));
        }
        if active && primary && secondary {
            if !self.windowing {
                // Undo the turn or pan made before the second button went down.
                if let Some((yaw, pitch, pan)) = self.grab {
                    (self.yaw, self.pitch, self.pan) = (yaw, pitch, pan);
                }
                self.windowing = true;
            }
            return (delta != Vec2::ZERO).then_some(delta);
        }
        if self.windowing {
            // Until both buttons are up: the one released last does not turn or pan.
            return None;
        }
        // Include the movement before the drag threshold, so a grabbed point follows exactly.
        let d = match (r.drag_started(), ui.input(|i| (i.pointer.press_origin(), i.pointer.latest_pos()))) {
            (true, (Some(a), Some(b))) => b - a,
            _ => r.drag_delta(),
        };
        if r.dragged_by(egui::PointerButton::Primary) && !(self.tool_drags && ui.input(|i| i.modifiers.command)) {
            // As if the front of the volume were grabbed: dragging right turns the face right,
            // dragging down tips the top towards the viewer (the camera rises).
            self.yaw += d.x * 0.01;
            self.pitch = (self.pitch + d.y * 0.01).clamp(-1.55, 1.55);
        } else if r.dragged_by(egui::PointerButton::Secondary) {
            // The point at the target's depth follows the cursor.
            let (eye, target, up, fov) = self.camera(vol);
            let per_point = 2.0 * (eye - target).length() * (fov * 0.5).tan() / r.rect.height().max(1.0);
            let fwd = (target - eye).normalize();
            let right = fwd.cross(up).normalize();
            let up2 = right.cross(fwd);
            self.pan += (-right * d.x + up2 * d.y) * per_point;
        } else if r.dragged_by(egui::PointerButton::Middle) {
            self.zoom = (self.zoom * (-d.y * 0.01).exp()).clamp(0.1, 40.0);
        }
        if r.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let f = (scroll * 0.003).exp() * pinch;
            if f != 1.0 {
                self.zoom = (self.zoom * f).clamp(0.1, 40.0);
            }
            ui.ctx().input_mut(|i| i.smooth_scroll_delta = Vec2::ZERO);
        }
        None
    }

    /// Where voxel coordinates `v` appear in the view drawn in `rect` (`None` behind the
    /// camera), to draw over the view.
    pub fn project(&self, vol: &LoadedVolume, rect: Rect, v: DVec3) -> Option<Pos2> {
        self.projector(vol, rect)(v)
    }

    /// [`Self::project`] as a function, for many points (its camera worked out once).
    pub fn projector(&self, vol: &LoadedVolume, rect: Rect) -> impl Fn(DVec3) -> Option<Pos2> + Sync + Send + use<> {
        let (eye, target, up, fov) = self.camera(vol);
        let (_, r) = Self::bounds(vol);
        let dist = (eye - target).length();
        let (near, far) = ((dist - r as f32 * 1.2).max(dist * 0.01), dist + r as f32 * 1.2);
        let proj = glam::camera::rh::proj::directx::perspective(fov, (rect.width() / rect.height().max(1.0)).max(1e-3), near, far);
        let view = glam::camera::rh::view::look_at_mat4(eye, target, up);
        let m = (proj * view).as_dmat4() * vol.geometry.matrix();
        move |v: DVec3| {
            let clip = m * v.extend(1.0);
            if clip.w <= 1e-9 {
                return None;
            }
            let (x, y) = (clip.x / clip.w, clip.y / clip.w);
            Some(Pos2::new(rect.min.x + ((x + 1.0) * 0.5) as f32 * rect.width(), rect.min.y + ((1.0 - y) * 0.5) as f32 * rect.height()))
        }
    }

    /// The voxel (coordinates) under `p` in the view drawn in `rect`: the first voxel of a
    /// shown segment along the ray when segments are shown, else the volume as the mode shows
    /// it — Surface the first value at the threshold, Volume the first at the window's centre,
    /// MIP the brightest.
    pub fn pick(&self, vol: &LoadedVolume, state: &ViewState, labels: Option<&LabelOverlay>, rect: Rect, p: Pos2) -> Option<DVec3> {
        let (eye, target, up, fov) = self.camera(vol);
        let (_, r) = Self::bounds(vol);
        let dist = (eye - target).length();
        let (near, far) = ((dist - r as f32 * 1.2).max(dist * 0.01), dist + r as f32 * 1.2);
        let proj = glam::camera::rh::proj::directx::perspective(fov, (rect.width() / rect.height().max(1.0)).max(1e-3), near, far);
        let view = glam::camera::rh::view::look_at_mat4(eye, target, up);
        let inv = (proj * view).as_dmat4().inverse();
        let ndc = DVec3::new(((p.x - rect.min.x) / rect.width().max(1.0) * 2.0 - 1.0) as f64, (1.0 - (p.y - rect.min.y) / rect.height().max(1.0) * 2.0) as f64, 0.0);
        let (a, b) = (inv.project_point3(ndc), inv.project_point3(DVec3::new(ndc.x, ndc.y, 1.0)));
        // Into voxel coordinates, clipped to the volume's box.
        let to_voxel = vol.geometry.matrix().inverse();
        let (a, b) = (to_voxel.transform_point3(a), to_voxel.transform_point3(b));
        let d = b - a;
        let (mut t0, mut t1) = (0f64, 1f64);
        for k in 0..3 {
            let (lo, hi) = (-0.5, vol.dims[k] as f64 - 0.5);
            if d[k].abs() < 1e-12 {
                if a[k] < lo || a[k] > hi {
                    return None;
                }
                continue;
            }
            let (u, v) = ((lo - a[k]) / d[k], (hi - a[k]) / d[k]);
            t0 = t0.max(u.min(v));
            t1 = t1.min(u.max(v));
        }
        if t0 > t1 {
            return None;
        }
        let steps = ((t1 - t0) * d.length() * 2.0).ceil().max(1.0) as usize;
        let (c, _) = state.window;
        let level = match self.mode {
            Mode3d::Surface => self.threshold.unwrap_or(c),
            _ => c,
        } as f32;
        let shown = |l: u8| l != 0 && labels.is_some_and(|o| o.colours.get(l as usize).is_some_and(|c| c.a() > 0));
        let mut brightest: Option<(f32, DVec3)> = None;
        for s in 0..=steps {
            let x = a + d * (t0 + (t1 - t0) * s as f64 / steps as f64);
            let [i, j, k] = [0, 1, 2].map(|n| x[n].round().clamp(0.0, vol.dims[n] as f64 - 1.0) as usize);
            let idx = vol.index(i, j, k);
            if self.show_labels && labels.is_some_and(|o| o.labels.len() == vol.values.len() && shown(o.labels[idx])) {
                return Some(x);
            }
            let v = vol.values[idx];
            let v = if vol.invert { -v } else { v };
            let level = if vol.invert { -level } else { level };
            match self.mode {
                Mode3d::Mip => {
                    if v.is_finite() && brightest.is_none_or(|(m, _)| v > m) {
                        brightest = Some((v, x));
                    }
                }
                _ => {
                    if v >= level {
                        return Some(x);
                    }
                }
            }
        }
        brightest.map(|(_, x)| x)
    }

    /// Create or update this view's textures.
    fn sync(&self, rs: &egui_wgpu::RenderState, vol: &Arc<LoadedVolume>, labels: Option<&LabelOverlay>) -> Result<Uploaded, String> {
        let mut renderer = rs.renderer.write();
        let max_dim = rs.device.limits().max_texture_dimension_3d;
        Ok(self.sync_with(&rs.device, &rs.queue, rs.target_format, &mut renderer.callback_resources, vol, labels, max_dim))
    }

    /// [`Self::sync`] with the GPU's parts and its largest 3D texture size given.
    #[allow(clippy::too_many_arguments)]
    fn sync_with(&self, device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, res: &mut egui_wgpu::CallbackResources, vol: &Arc<LoadedVolume>, labels: Option<&LabelOverlay>, max_dim: u32) -> Uploaded {
        if !res.contains::<Gpu>() {
            res.insert(Gpu::new(device, format));
        }
        let gpu = res.get_mut::<Gpu>().unwrap();
        let offscreen = gpu.blitter.is_some();
        let Gpu { layout, sampler, views, .. } = gpu;
        views.retain(|_, v| v.alive.strong_count() > 0);
        if views.get(&self.id).is_none_or(|v| v.source.as_ptr() != Arc::as_ptr(vol)) {
            let stride = stride_for(vol.dims, max_dim);
            let td = vol.dims.map(|d| d.div_ceil(stride) as u32);
            let halves = volume_halves(vol, td, stride);
            let volume = texture_3d(device, "volume3d.values", td, wgpu::TextureFormat::R16Float);
            write_3d(queue, &volume, td, 2, bytemuck::cast_slice(&halves));
            let nb = brick_dims(td);
            let value_bricks = texture_3d(device, "volume3d.value_bricks", nb, wgpu::TextureFormat::Rg16Float);
            write_3d(queue, &value_bricks, nb, 4, bytemuck::cast_slice(&value_ranges(&halves, td)));
            drop(halves);
            // No segments yet: nothing is occupied.
            let occupancy_bricks = texture_3d(device, "volume3d.occupancy_bricks", nb, wgpu::TextureFormat::R8Unorm);
            write_3d(queue, &occupancy_bricks, nb, 1, &vec![0; nb.iter().map(|&n| n as usize).product()]);
            let labels_t = texture_3d(device, "volume3d.labels", [1, 1, 1], wgpu::TextureFormat::R8Uint);
            let occupancy = texture_3d(device, "volume3d.occupancy", [1, 1, 1], wgpu::TextureFormat::R8Unorm);
            write_3d(queue, &labels_t, [1, 1, 1], 1, &[0]);
            write_3d(queue, &occupancy, [1, 1, 1], 1, &[0]);
            let uniforms = device.create_buffer(&wgpu::BufferDescriptor { label: Some("volume3d.uniforms"), size: std::mem::size_of::<Uniforms>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let bind = bind_group(device, layout, sampler, &uniforms, [&volume, &occupancy, &labels_t, &value_bricks, &occupancy_bricks]);
            let v = ViewGpu { alive: Arc::downgrade(&self.token), uniforms, bind, volume, labels: labels_t, occupancy, value_bricks, occupancy_bricks, source: Arc::downgrade(vol), label_key: None, tex_dims: td, stride, gen: 0, cache: None };
            views.insert(self.id, v);
        }
        let want = labels.filter(|l| l.labels.len() == vol.voxel_count()).map(|l| (l.version, self.label_smoothing));
        let v = views.get_mut(&self.id).unwrap();
        // Hidden segments are not uploaded (nor their key changed, so showing them uploads them).
        if let (Some(l), true) = (labels, self.show_labels && want.is_some() && want != v.label_key) {
            let (td, stride) = (v.tex_dims, v.stride);
            let shown: [u8; 256] = std::array::from_fn(|id| if id != 0 && l.colours.get(id).is_some_and(|c| c.a() > 0) { 255 } else { 0 });
            let mut occ: Vec<u8> = if stride == 1 { l.labels.par_iter().map(|&id| shown[id as usize]).collect() } else { sampled(vol, td, stride, |i| shown[l.labels[i] as usize]) };
            // Soften the voxel steps so label surfaces come out smooth.
            blur(&mut occ, self.label_smoothing, td);
            let resized = v.labels.size() != wgpu::Extent3d { width: td[0], height: td[1], depth_or_array_layers: td[2] };
            if resized {
                v.labels = texture_3d(device, "volume3d.labels", td, wgpu::TextureFormat::R8Uint);
                v.occupancy = texture_3d(device, "volume3d.occupancy", td, wgpu::TextureFormat::R8Unorm);
                v.bind = bind_group(device, layout, sampler, &v.uniforms, [&v.volume, &v.occupancy, &v.labels, &v.value_bricks, &v.occupancy_bricks]);
            }
            // With only the smoothing changed, the labels on the GPU are still right.
            if resized || v.label_key.map(|k| k.0) != want.map(|k| k.0) {
                if stride == 1 {
                    write_3d(queue, &v.labels, td, 1, l.labels);
                } else {
                    write_3d(queue, &v.labels, td, 1, &sampled(vol, td, stride, |i| l.labels[i]));
                }
            }
            write_3d(queue, &v.occupancy, td, 1, &occ);
            write_3d(queue, &v.occupancy_bricks, brick_dims(td), 1, &occupancy_maxima(&occ, td));
            v.label_key = want;
            v.gen += 1;
        }
        Uploaded { tex_dims: v.tex_dims, stride: v.stride, offscreen }
    }

    fn uniforms(&self, vol: &LoadedVolume, state: &ViewState, labels: Option<&LabelOverlay>, rect: Rect, bg: Color32, up: &Uploaded) -> Uniforms {
        let (eye, target, up_dir, fov) = self.camera(vol);
        let (_, r) = Self::bounds(vol);
        let dist = (eye - target).length();
        let near = (dist - r as f32 * 1.2).max(dist * 0.01);
        let far = dist + r as f32 * 1.2;
        let proj = glam::camera::rh::proj::directx::perspective(fov, (rect.width() / rect.height().max(1.0)).max(1e-3), near, far);
        let view = glam::camera::rh::view::look_at_mat4(eye, target, up_dir);
        let inv_view_proj = (proj * view).inverse();
        // World → texture coordinates: texel k holds voxel k · stride, its centre at
        // (k + 0.5) / size.
        let inv = vol.geometry.matrix().inverse();
        let (td, s) = (up.tex_dims.map(|d| d as f64), up.stride as f64);
        let world_to_tex = DMat4::from_scale(DVec3::new(1.0 / (td[0] * s), 1.0 / (td[1] * s), 1.0 / (td[2] * s))) * DMat4::from_translation(DVec3::splat(0.5 * s)) * inv;
        let m3 = glam::DMat3::from_mat4(world_to_tex).transpose();
        let grad_to_world = DMat4::from_mat3(m3);
        let (lo, hi) = (vol.range.0 as f64, vol.range.1 as f64);
        let norm = |x: f64| if hi > lo { ((x - lo) / (hi - lo)) as f32 } else { 0.0 };
        let (c, w) = state.window;
        let threshold = self.threshold.unwrap_or(c);
        let mut palette = [[0f32; 4]; 256];
        if let Some(l) = labels {
            for (k, col) in l.colours.iter().take(256).enumerate() {
                // egui renders to a non-sRGB target: colours stay in gamma space.
                palette[k] = col.to_normalized_gamma_f32();
            }
        }
        let bgc = bg.to_normalized_gamma_f32();
        let light = (eye - target).normalize();
        let nb = brick_dims(up.tex_dims);
        Uniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            world_to_tex: world_to_tex.as_mat4().to_cols_array_2d(),
            grad_to_world: grad_to_world.as_mat4().to_cols_array_2d(),
            dims: [up.tex_dims[0] as f32, up.tex_dims[1] as f32, up.tex_dims[2] as f32, RATE],
            window: [norm(c - w * 0.5), norm(c + w * 0.5), norm(threshold), match self.mode { Mode3d::Mip => 0.0, Mode3d::Volume => 1.0, Mode3d::Surface => 2.0 }],
            // Opacity per sample, which stands for `stride` voxels along the ray.
            labels: [if self.show_labels && labels.is_some() { 1.0 } else { 0.0 }, labels.map(|l| l.opacity.max(0.5)).unwrap_or(1.0), 0.12 * self.opacity * up.stride as f32, if vol.invert { 1.0 } else { 0.0 }],
            background: [bgc[0], bgc[1], bgc[2], 1.0],
            light: [light.x, light.y, light.z, 0.0],
            bricks: [nb[0] as f32, nb[1] as f32, nb[2] as f32, BRICK as f32],
            palette,
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui, rect: Rect, vol: &LoadedVolume, has_labels: bool) {
        let bar = Rect::from_min_size(rect.min + egui::vec2(6.0, 6.0), egui::vec2(rect.width() - 12.0, 22.0));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(bar).layout(egui::Layout::left_to_right(egui::Align::Center)).id_salt(("volume3d.controls", self.id)));
        egui::ComboBox::from_id_salt(("volume3d.mode", self.id)).selected_text(self.mode.label()).width(80.0).show_ui(&mut child, |ui| {
            for m in Mode3d::ALL {
                ui.selectable_value(&mut self.mode, m, m.label());
            }
        });
        match self.mode {
            Mode3d::Surface => {
                let (lo, hi) = (vol.range.0 as f64, vol.range.1 as f64);
                let mut t = self.threshold.unwrap_or((lo + hi) * 0.5);
                let speed = ((hi - lo) / 500.0).max(1e-6);
                if child.add(egui::DragValue::new(&mut t).range(lo..=hi).speed(speed).prefix("threshold ")).changed() {
                    self.threshold = Some(t);
                }
            }
            Mode3d::Volume => {
                child.add(egui::Slider::new(&mut self.opacity, 0.1..=5.0).logarithmic(true).text("opacity"));
            }
            Mode3d::Mip => {}
        }
        if has_labels {
            child.checkbox(&mut self.show_labels, "Segments");
            if self.show_labels {
                child.add(egui::Slider::new(&mut self.label_smoothing, 0..=8).text("smoothing")).on_hover_text("Smoothing of the segments' surfaces (0: the voxels' steps)");
            }
        }
    }
}

/// Every `stride`-th voxel of `vol` (clamped at the far edges), mapped by `f` from its index.
fn sampled<T: Send + Copy + Default>(vol: &LoadedVolume, td: [u32; 3], stride: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let plane = (td[0] * td[1]) as usize;
    let mut out = vec![T::default(); plane * td[2] as usize];
    out.par_chunks_mut(plane).enumerate().for_each(|(z, o)| {
        let k = (z * stride).min(vol.dims[2] - 1);
        for y in 0..td[1] as usize {
            let j = (y * stride).min(vol.dims[1] - 1);
            for x in 0..td[0] as usize {
                let i = (x * stride).min(vol.dims[0] - 1);
                o[y * td[0] as usize + x] = f(vol.index(i, j, k));
            }
        }
    });
    out
}

/// `passes` passes of [`blur_121`] over `v` (`td` texels).
fn blur(v: &mut Vec<u8>, passes: u8, td: [u32; 3]) {
    if passes == 0 {
        return;
    }
    let mut scratch = vec![0u8; v.len()];
    for _ in 0..passes {
        blur_121(v, &mut scratch, td);
    }
}

/// The [1 2 1] / 4 filter of three bytes, rounded.
#[inline(always)]
fn f121(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
}

/// One pass of the [1 2 1] / 4 filter along each axis (a small Gaussian), rounded to bytes
/// after each axis: x then y within each plane (through `scratch` and back), then z into
/// `scratch`, which then swaps places with `v`.
fn blur_121(v: &mut Vec<u8>, scratch: &mut Vec<u8>, td: [u32; 3]) {
    let [nx, ny, nz] = td.map(|d| d as usize);
    let plane = nx * ny;
    // Along a line: `out[i] = f(a[i], b[i], c[i])` with the neighbours' lines.
    let lines = |out: &mut [u8], a: &[u8], b: &[u8], c: &[u8]| {
        for (((o, &a), &b), &c) in out.iter_mut().zip(a).zip(b).zip(c) {
            *o = f121(a, b, c);
        }
    };
    v.par_chunks_mut(plane).zip(scratch.par_chunks_mut(plane)).for_each(|(p, s)| {
        for (row, out) in p.chunks_exact(nx).zip(s.chunks_exact_mut(nx)) {
            // Edges repeat their texel.
            if nx == 1 {
                out[0] = row[0];
                continue;
            }
            out[0] = f121(row[0], row[0], row[1]);
            lines(&mut out[1..nx - 1], &row[..nx - 2], &row[1..nx - 1], &row[2..]);
            out[nx - 1] = f121(row[nx - 2], row[nx - 1], row[nx - 1]);
        }
        for y in 0..ny {
            let (a, c) = (y.saturating_sub(1), (y + 1).min(ny - 1));
            lines(&mut p[y * nx..(y + 1) * nx], &s[a * nx..(a + 1) * nx], &s[y * nx..(y + 1) * nx], &s[c * nx..(c + 1) * nx]);
        }
    });
    let src: &[u8] = v;
    scratch.par_chunks_mut(plane).enumerate().for_each(|(z, out)| {
        let p = |k: usize| &src[k * plane..(k + 1) * plane];
        lines(out, p(z.saturating_sub(1)), p(z), p((z + 1).min(nz - 1)));
    });
    std::mem::swap(v, scratch);
}

/// The ray caster's bind group: the uniforms, then the volume, occupancy, labels, value bricks
/// and occupancy bricks textures.
fn bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, uniforms: &wgpu::Buffer, textures: [&wgpu::Texture; 5]) -> wgpu::BindGroup {
    let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
    let [volume, occupancy, labels, value_bricks, occupancy_bricks] = textures.map(view);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("volume3d"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&volume) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&occupancy) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&labels) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&value_bricks) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&occupancy_bricks) },
        ],
    })
}

impl Cache {
    /// A cache for views reaching `size` pixels from the target's top left (rounded up, so that
    /// resizing the view seldom makes a new one).
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, blitter: &Blitter, sampler: &wgpu::Sampler, size: [u32; 2]) -> Self {
        let [w, h] = size.map(|n| n.div_ceil(256) * 256);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume3d.cache"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blit = device.create_buffer(&wgpu::BufferDescriptor { label: Some("volume3d.blit"), size: std::mem::size_of::<BlitUniforms>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume3d.blit"),
            layout: &blitter.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 10, resource: blit.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });
        Self { texture, view, blit, bind, uniforms: Box::new(Uniforms::zeroed()), key: None, placed: None, shown: None }
    }

    /// Whether views reaching `size` fit, without the texture being far bigger than they need.
    fn fits(&self, size: [u32; 2]) -> bool {
        let (w, h) = (self.texture.width(), self.texture.height());
        w >= size[0] && h >= size[1] && (w as u64 * h as u64) <= 4 * (size[0] as u64 * size[1] as u64).max(256 * 256)
    }
}

/// The paint callback of one 3D view for one frame.
struct Draw {
    id: u64,
    rect: Rect,
    uniforms: Uniforms,
    /// Drawn at half size (the view is moving).
    half: bool,
}

impl egui_wgpu::CallbackTrait for Draw {
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue, screen: &egui_wgpu::ScreenDescriptor, _encoder: &mut wgpu::CommandEncoder, res: &mut egui_wgpu::CallbackResources) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = res.get_mut::<Gpu>() else { return Vec::new() };
        let slot = pipeline_slot(&self.uniforms);
        gpu.ensure_pipeline(device, slot);
        let Gpu { format, pipelines, sampler, blitter, views, .. } = gpu;
        let Some(v) = views.get_mut(&self.id) else { return Vec::new() };
        if let Some(c) = &mut v.cache {
            c.shown = None;
        }
        // The view's pixels as egui will set its viewport (rounded, clamped to the screen).
        let px = egui::PaintCallbackInfo { viewport: self.rect, clip_rect: self.rect, pixels_per_point: screen.pixels_per_point, screen_size_px: screen.size_in_pixels }.viewport_in_pixels();
        let (Some(blitter), Some(pipeline), true) = (blitter.as_ref(), pipelines[slot].as_ref(), px.width_px > 0 && px.height_px > 0) else {
            // Drawn directly in paint() (if at all).
            queue.write_buffer(&v.uniforms, 0, bytemuck::bytes_of(&self.uniforms));
            return Vec::new();
        };
        let (origin, size) = ([px.left_px as u32, px.top_px as u32], [px.width_px as u32, px.height_px as u32]);
        // A full frame goes where the view is on screen, so that its rays start exactly where
        // they would drawn directly (the rasterizer interpolates in screen coordinates); a
        // half-size one at the top left.
        let (at, drawn) = if self.half { ([0, 0], size.map(|n| n.div_ceil(2))) } else { (origin, size) };
        let extent = [origin[0] + size[0], origin[1] + size[1]];
        if v.cache.as_ref().is_none_or(|c| !c.fits(extent)) {
            v.cache = Some(Cache::new(device, *format, blitter, sampler, extent));
        }
        let c = v.cache.as_mut().unwrap();
        // Ray cast again only when something it depends on changed: the uniforms (camera,
        // window, mode, colours…), the textures, or the place and size.
        let key = (v.gen, at, drawn, self.half);
        if c.key != Some(key) || bytemuck::bytes_of(&*c.uniforms) != bytemuck::bytes_of(&self.uniforms) {
            queue.write_buffer(&v.uniforms, 0, bytemuck::bytes_of(&self.uniforms));
            // Submitted at once rather than with egui's frame, which is dropped unsubmitted when
            // there is no surface texture to draw to (an occluded window): the cache would then
            // hold nothing while its key said otherwise.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("volume3d.cache") });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volume3d.cache"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &c.view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(at[0] as f32, at[1] as f32, drawn[0] as f32, drawn[1] as f32, 0.0, 1.0);
            pass.set_scissor_rect(at[0], at[1], drawn[0], drawn[1]);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &v.bind, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            queue.submit([encoder.finish()]);
            *c.uniforms = self.uniforms;
            c.key = Some(key);
        }
        let placed = BlitUniforms {
            origin: [px.left_px as f32, px.top_px as f32],
            size: size.map(|n| n as f32),
            drawn: drawn.map(|n| n as f32),
            cache_size: [c.texture.width() as f32, c.texture.height() as f32],
        };
        if c.placed != Some(placed) {
            queue.write_buffer(&c.blit, 0, bytemuck::bytes_of(&placed));
            c.placed = Some(placed);
        }
        c.shown = Some(self.half);
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let Some(gpu) = res.get::<Gpu>() else { return };
        let Some(v) = gpu.views.get(&self.id) else { return };
        match (&gpu.blitter, v.cache.as_ref().and_then(|c| Some((c, c.shown?)))) {
            (Some(b), Some((c, half))) => {
                pass.set_pipeline(if half { &b.scaled } else { &b.exact });
                pass.set_bind_group(0, &c.bind, &[]);
            }
            _ => {
                let Some(pipeline) = &gpu.pipelines[pipeline_slot(&self.uniforms)] else { return };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &v.bind, &[]);
            }
        }
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use egui::{Event, PointerButton, Pos2};

    /// One frame of `view` filling a 400 × 300 window, with `events`.
    fn frame(ctx: &egui::Context, view: &mut VolumeView3d, vol: &Arc<LoadedVolume>, state: &mut ViewState, events: Vec<Event>) {
        let input = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 300.0))), events, ..Default::default() };
        let _ = ctx.run_ui(input, |ui| {
            view.ui(ui, None, vol, state, None);
        });
    }

    #[test]
    fn left_and_right_drag_sets_the_window_not_the_camera() {
        let g = mm_volume::geometry::Geometry::from_axes([4, 4, 4, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = Arc::new(LoadedVolume::from_values("v", [4, 4, 4], (0..64).map(|i| i as f32).collect(), g));
        let (ctx, mut view) = (egui::Context::default(), VolumeView3d::new());
        let mut state = ViewState::new(&vol);
        let (w0, yaw0) = (state.window, view.yaw);
        let button = |x: f32, b: PointerButton, pressed: bool| Event::PointerButton { pos: Pos2::new(x, 150.0), button: b, pressed, modifiers: Default::default() };
        let to = |x: f32| Event::PointerMoved(Pos2::new(x, 150.0));
        for events in [vec![to(200.0)], vec![button(200.0, PointerButton::Primary, true)], vec![to(215.0)], vec![to(230.0)]] {
            frame(&ctx, &mut view, &vol, &mut state, events);
        }
        // Left drag alone turns the camera.
        assert!(view.yaw != yaw0);
        frame(&ctx, &mut view, &vol, &mut state, vec![button(230.0, PointerButton::Secondary, true)]);
        for x in [250.0, 270.0, 290.0] {
            frame(&ctx, &mut view, &vol, &mut state, vec![to(x)]);
        }
        // Both buttons: the window widens (dragged across), the turn is undone.
        assert!(state.window.1 > w0.1 * 1.3, "{:?} vs {w0:?}", state.window);
        assert_eq!(view.yaw, yaw0);
        // Letting one button go first does not turn the camera.
        frame(&ctx, &mut view, &vol, &mut state, vec![button(290.0, PointerButton::Secondary, false)]);
        frame(&ctx, &mut view, &vol, &mut state, vec![to(320.0)]);
        assert_eq!(view.yaw, yaw0);
        frame(&ctx, &mut view, &vol, &mut state, vec![button(320.0, PointerButton::Primary, false)]);
    }

    use super::*;

    #[test]
    fn picking_finds_the_surface_or_segment_under_the_pointer() {
        // A bright cube (voxels 8…11) in a 20³ volume; the camera looks from anterior (+y).
        let g = mm_volume::geometry::Geometry::from_axes([20, 20, 20, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let inside = |n: usize| [n % 20, (n / 20) % 20, n / 400].iter().all(|c| (8..12).contains(c));
        let vol = LoadedVolume::from_values("t", [20, 20, 20], (0..8000).map(|n| if inside(n) { 100.0 } else { 0.0 }).collect(), g);
        let mut state = ViewState::new(&vol);
        state.window = (50.0, 100.0);
        let mut view = VolumeView3d::new();
        view.pitch = 0.0;
        let rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(400.0, 300.0));
        for mode in [Mode3d::Surface, Mode3d::Volume, Mode3d::Mip] {
            view.mode = mode;
            let p = view.pick(&vol, &state, None, rect, rect.center()).expect("hits the cube");
            assert!((p.x - 9.5).abs() < 0.6 && (p.z - 9.5).abs() < 0.6, "{mode:?} {p}");
            if mode != Mode3d::Mip {
                assert!((10.5..=12.0).contains(&p.y), "the front face: {mode:?} {p}");
            }
        }
        // What is picked under the centre projects back onto it.
        view.mode = Mode3d::Surface;
        let p = view.pick(&vol, &state, None, rect, rect.center()).unwrap();
        assert!(view.project(&vol, rect, p).unwrap().distance(rect.center()) < 0.5);
        // Off the volume: nothing.
        assert!(view.pick(&vol, &state, None, rect, Pos2::new(1.0, 1.0)).is_none());
        // A segment in front of the cube is found first; hidden segments are passed through.
        let labels: Vec<u8> = (0..8000).map(|n| ([n % 20, n / 400].iter().all(|c| (8..12).contains(c)) && (14..16).contains(&((n / 20) % 20))) as u8).collect();
        let mut colours = vec![Color32::TRANSPARENT, Color32::RED];
        let overlay = LabelOverlay { labels: &labels, colours: &colours, opacity: 1.0, version: 1 };
        view.mode = Mode3d::Surface;
        let p = view.pick(&vol, &state, Some(&overlay), rect, rect.center()).unwrap();
        assert!((13.5..=16.0).contains(&p.y), "{p}");
        colours[1] = Color32::TRANSPARENT;
        let overlay = LabelOverlay { labels: &labels, colours: &colours, opacity: 1.0, version: 1 };
        let p = view.pick(&vol, &state, Some(&overlay), rect, rect.center()).unwrap();
        assert!(p.y < 12.0, "{p}");
    }

    #[test]
    fn half_floats() {
        assert_eq!(half(0.0), 0);
        assert_eq!(half(1.0), 0x3c00);
        assert_eq!(half(0.5), 0x3800);
        assert_eq!(half(2.0), 0x3c00);
        // 0.1 ≈ 0x2e66.
        assert_eq!(half(0.1), 0x2e66);
    }

    #[test]
    fn strides_fit_the_limits() {
        assert_eq!(stride_for([512, 512, 300], 2048), 1);
        assert_eq!(stride_for([4000, 100, 100], 2048), 2);
        assert!(stride_for([1024, 1024, 1024], 2048) >= 2);
    }

    /// A deterministic pseudo-random number for test data.
    fn hash(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    /// The blur as it was first written (three copies a pass), for comparison.
    fn blur_121_reference(v: &mut [u8], td: [u32; 3]) {
        let [nx, ny, nz] = td.map(|d| d as usize);
        let f = |a: u8, b: u8, c: u8| ((a as u32 + 2 * b as u32 + c as u32 + 2) / 4) as u8;
        let plane = nx * ny;
        let src = v.to_vec();
        v.par_chunks_mut(nx).enumerate().for_each(|(r, row)| {
            let s = &src[r * nx..(r + 1) * nx];
            for x in 0..nx {
                row[x] = f(s[x.saturating_sub(1)], s[x], s[(x + 1).min(nx - 1)]);
            }
        });
        let src = v.to_vec();
        v.par_chunks_mut(plane).enumerate().for_each(|(z, out)| {
            let s = &src[z * plane..(z + 1) * plane];
            for y in 0..ny {
                let (a, c) = (y.saturating_sub(1), (y + 1).min(ny - 1));
                for x in 0..nx {
                    out[y * nx + x] = f(s[a * nx + x], s[y * nx + x], s[c * nx + x]);
                }
            }
        });
        let src = v.to_vec();
        v.par_chunks_mut(plane).enumerate().for_each(|(z, out)| {
            let p = |k: usize| &src[k * plane..(k + 1) * plane];
            let (a, b, c) = (p(z.saturating_sub(1)), p(z), p((z + 1).min(nz - 1)));
            for i in 0..plane {
                out[i] = f(a[i], b[i], c[i]);
            }
        });
    }

    #[test]
    fn the_blur_matches_the_reference_byte_for_byte() {
        for td in [[1, 1, 1], [1, 5, 3], [7, 1, 4], [2, 2, 2], [13, 11, 9], [64, 48, 20]] {
            let n = td.iter().map(|&d| d as usize).product::<usize>();
            // Blobs of 255 on 0 (as occupancy is), and noise.
            for data in [(0..n).map(|i| if hash(i as u64 / 7).is_multiple_of(3) { 255 } else { 0 }).collect::<Vec<u8>>(), (0..n).map(|i| hash(i as u64) as u8).collect()] {
                for passes in 0..=8u8 {
                    let mut expected = data.clone();
                    for _ in 0..passes {
                        blur_121_reference(&mut expected, td);
                    }
                    let mut got = data.clone();
                    blur(&mut got, passes, td);
                    assert!(got == expected, "{td:?}, {passes} passes");
                }
            }
        }
    }

    #[test]
    fn bricks_hold_the_range_of_what_they_reach() {
        for td in [[1, 1, 1], [8, 8, 8], [9, 17, 3], [30, 21, 16]] {
            let [nx, ny, nz] = td.map(|d| d as usize);
            let halves: Vec<u16> = (0..nx * ny * nz).map(|i| (hash(i as u64) % 0x3c01) as u16).collect();
            let occ: Vec<u8> = (0..nx * ny * nz).map(|i| hash(i as u64 + 99) as u8).collect();
            let (values, occupied) = (value_ranges(&halves, td), occupancy_maxima(&occ, td));
            let nb = brick_dims(td).map(|d| d as usize);
            assert_eq!(values.len(), nb.iter().product::<usize>());
            for (b, (range, o)) in values.iter().zip(&occupied).enumerate() {
                let at = [b % nb[0], (b / nb[0]) % nb[1], b / (nb[0] * nb[1])];
                let reach = |a: usize, n: usize| (at[a] * BRICK).saturating_sub(1)..((at[a] + 1) * BRICK + 1).min(n);
                let (mut lo, mut hi, mut om) = (u16::MAX, 0, 0);
                for z in reach(2, nz) {
                    for y in reach(1, ny) {
                        for x in reach(0, nx) {
                            let i = (z * ny + y) * nx + x;
                            (lo, hi, om) = (lo.min(halves[i]), hi.max(halves[i]), om.max(occ[i]));
                        }
                    }
                }
                assert_eq!((*range, *o), ([lo, hi], om), "{td:?} brick {at:?}");
            }
        }
    }

    #[test]
    fn volume_texels_are_as_sampled() {
        let g = mm_volume::geometry::Geometry::from_axes([9, 7, 5, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("v", [9, 7, 5], (0..315).map(|i| (hash(i) % 2000) as f32 - 1000.0).collect(), g);
        let td = [9, 7, 5];
        let halves = volume_halves(&vol, td, 1);
        let (lo, hi) = vol.range;
        let old: Vec<u8> = sampled(&vol, td, 1, |i| half((vol.values[i] - lo) * (1.0 / (hi - lo)))).into_iter().flat_map(u16::to_le_bytes).collect();
        assert_eq!(bytemuck::cast_slice::<u16, u8>(&halves), old);
        // Every other voxel, clamped at the far edges.
        let halves = volume_halves(&vol, [5, 4, 3], 2);
        assert_eq!(halves.len(), 60);
        assert_eq!(halves[59], half((vol.get(8, 6, 4) - lo) * (1.0 / (hi - lo))));
    }

    #[test]
    fn strided_textures_map_voxels_to_their_texels() {
        let g = mm_volume::geometry::Geometry::from_axes([10, 9, 7, 1], [0.5, 0.7, 1.2], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::new(3.0, -2.0, 1.0));
        let vol = LoadedVolume::from_values("v", [10, 9, 7], vec![0.0; 630], g);
        let view = VolumeView3d::new();
        let state = ViewState::new(&vol);
        let rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(100.0, 100.0));
        for stride in [1, 2, 3] {
            let td = vol.dims.map(|d| d.div_ceil(stride) as u32);
            let u = view.uniforms(&vol, &state, None, rect, Color32::BLACK, &Uploaded { tex_dims: td, stride, offscreen: true });
            assert_eq!(u.dims[..3], td.map(|d| d as f32));
            let m = glam::Mat4::from_cols_array_2d(&u.world_to_tex).as_dmat4();
            // Voxel (k · stride) is texel k, whose centre is at (k + 0.5) / size.
            let k = [1usize, 2, 1];
            let t = m.transform_point3(vol.geometry.world(DVec3::new((k[0] * stride) as f64, (k[1] * stride) as f64, (k[2] * stride) as f64)));
            for a in 0..3 {
                assert!((t[a] - (k[a] as f64 + 0.5) / td[a] as f64).abs() < 1e-5, "stride {stride}: {t}");
            }
            assert_eq!(u.labels[2], 0.12 * stride as f32);
        }
    }

    #[test]
    fn magnification_follows_zoom_spacing_and_stride() {
        // Voxels of 1 × 1.5 × 1 mm: the finest axis sets the texel (1 mm).
        let g =mm_volume::geometry::Geometry::from_axes([58, 58, 58, 1], [1.0, 1.5, 1.0], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("v", [58, 58, 58], vec![0.0; 58 * 58 * 58], g);
        let mut view = VolumeView3d::new();
        let base = view.pixels_per_texel(&vol, 1000.0, 1);
        let (_, r) = VolumeView3d::bounds(&vol);
        // The view's height spans 2 · dist · tan(15°) at the target, dist = r / sin(15°) · 1.05.
        let span = 2.0 * r as f32 / 15f32.to_radians().sin() * 1.05 * 15f32.to_radians().tan();
        assert!((base - 1000.0 / span).abs() < 1e-3 * base, "{base} vs {}", 1000.0 / span);
        assert!((view.pixels_per_texel(&vol, 1000.0, 2) - 2.0 * base).abs() < 1e-3 * base);
        view.zoom = 2.5;
        assert!((view.pixels_per_texel(&vol, 1000.0, 1) - 2.5 * base).abs() < 1e-3 * base);
    }

    #[test]
    fn moving_is_a_change_from_the_frame_before() {
        let mut view = VolumeView3d::new();
        let (a, mut b) = (Uniforms::zeroed(), Uniforms::zeroed());
        b.window[0] = 0.25;
        assert!(!view.moving(1, &a));
        assert!(!view.moving(2, &a));
        assert!(view.moving(3, &b));
        // A second pass of the same frame still compares with the frame before.
        assert!(view.moving(3, &b));
        assert!(!view.moving(4, &b));
        assert!(view.moving(5, &a));
    }

    #[test]
    fn the_shader_validates_for_every_pipeline() {
        use wgpu::naga;
        let module = naga::front::wgsl::parse_str(include_str!("shaders/volume3d.wgsl")).expect("parses");
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).expect("validates");
        for slot in 0..6 {
            for skip in [true, false] {
                let mut constants = naga::back::PipelineConstants::default();
                constants.insert("MODE".into(), (slot / 2) as f64);
                constants.insert("LABELS".into(), (slot % 2) as f64);
                constants.insert("SKIP".into(), skip as u8 as f64);
                naga::back::pipeline_constants::process_overrides(&module, &info, Some((naga::ShaderStage::Fragment, "fs")), &constants).unwrap_or_else(|e| panic!("slot {slot}, skip {skip}: {e}"));
            }
        }
        let none = naga::back::PipelineConstants::default();
        for (stage, entry) in [(naga::ShaderStage::Vertex, "vs"), (naga::ShaderStage::Fragment, "blit"), (naga::ShaderStage::Fragment, "blit_scaled")] {
            naga::back::pipeline_constants::process_overrides(&module, &info, Some((stage, entry)), &none).unwrap_or_else(|e| panic!("{entry}: {e}"));
        }
    }

    /// Wait for a future that wgpu's native backends have ready at once.
    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::yield_now();
        }
    }

    /// A headless GPU, or `None` without one (the GPU tests then pass without checking).
    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// `draw` as egui draws it into a `screen`-sized target at `ppp` pixels per point:
    /// prepare(), then paint() in a pass with the view's viewport; the target's RGBA pixels.
    fn render(device: &wgpu::Device, queue: &wgpu::Queue, res: &mut egui_wgpu::CallbackResources, draw: &Draw, screen: [u32; 2], ppp: f32) -> Vec<u8> {
        use egui_wgpu::CallbackTrait;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: screen[0], height: screen[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: screen, pixels_per_point: ppp };
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut buffers = draw.prepare(device, queue, &sd, &mut encoder, res);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            let info = egui::PaintCallbackInfo { viewport: draw.rect, clip_rect: draw.rect, pixels_per_point: ppp, screen_size_px: screen };
            let px = info.viewport_in_pixels();
            pass.set_viewport(px.left_px as f32, px.top_px as f32, px.width_px as f32, px.height_px as f32, 0.0, 1.0);
            draw.paint(info, &mut pass, res);
        }
        let row = (screen[0] * 4).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * screen[1]) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(screen[1]) } },
            wgpu::Extent3d { width: screen[0], height: screen[1], depth_or_array_layers: 1 },
        );
        buffers.push(encoder.finish());
        queue.submit(buffers);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        (0..screen[1] as usize).flat_map(|y| mapped[y * row as usize..][..screen[0] as usize * 4].to_vec()).collect()
    }

    /// A noisy ball with a bright shell (and a bright rod through it) in a box of air, a
    /// tube segment (1), a blob segment (2) and a hidden one (3).
    fn phantom(dims: [usize; 3], seed: u64) -> (Arc<LoadedVolume>, Vec<u8>) {
        let g = mm_volume::geometry::Geometry::from_axes([dims[0], dims[1], dims[2], 1], [0.8, 0.9, 1.1], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::new(-5.0, 2.0, 7.0));
        let c = dims.map(|d| d as f64 * 0.5);
        let n = dims.iter().product::<usize>();
        let (mut values, mut labels) = (vec![0f32; n], vec![0u8; n]);
        for (i, (v, l)) in values.iter_mut().zip(&mut labels).enumerate() {
            let p = [i % dims[0], (i / dims[0]) % dims[1], i / (dims[0] * dims[1])].map(|x| x as f64);
            let r = ((p[0] - c[0]).powi(2) / c[0].powi(2) + (p[1] - c[1]).powi(2) / c[1].powi(2) + (p[2] - c[2]).powi(2) / c[2].powi(2)).sqrt() * 1.25;
            let noise = (hash(i as u64 ^ seed) % 100) as f32;
            *v = if r < 0.8 { 300.0 + noise } else if r < 1.0 { 900.0 + noise } else { noise - 1000.0 };
            if ((p[0] - c[0]).powi(2) + (p[2] - c[2] * 0.7).powi(2)).sqrt() < 3.0 {
                *v = 1200.0;
            }
            if ((p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt() < 2.5 && r < 0.9 {
                *l = 1;
            } else if (p[0] - c[0] * 0.6).abs() < 4.0 && (p[1] - c[1] * 1.3).abs() < 3.0 && (p[2] - c[2] * 1.2).abs() < 5.0 {
                *l = 2;
            } else if r < 0.5 && p[2] < c[2] {
                *l = 3;
            }
        }
        (Arc::new(LoadedVolume::from_values("phantom", dims, values, g)), labels)
    }

    /// Pixels that differ between two images, and the largest difference.
    fn diff(a: &[u8], b: &[u8]) -> (usize, u8) {
        a.chunks(4).zip(b.chunks(4)).fold((0, 0), |(n, m), (p, q)| {
            let d = p.iter().zip(q).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
            (n + (d > 0) as usize, m.max(d))
        })
    }

    #[test]
    fn cached_skipping_and_half_size_frames_match_the_full_ray_cast() {
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let (vol, labels) = phantom([60, 52, 44], 1);
        let colours = [Color32::TRANSPARENT, Color32::from_rgb(220, 60, 50), Color32::from_rgb(70, 160, 230), Color32::TRANSPARENT];
        let overlay = LabelOverlay { labels: &labels, colours: &colours, opacity: 1.0, version: 1 };
        let mut res = egui_wgpu::CallbackResources::default();
        let mut view = VolumeView3d::new();
        let mut state = ViewState::new(&vol);
        // The same volume and segments in a view whose bricks are never empty: its rays walk
        // every sample, in the same pipelines.
        let walker = VolumeView3d::new();
        walker.sync_with(&device, &queue, TARGET, &mut res, &vol, Some(&overlay), 2048);
        let w = &res.get::<Gpu>().unwrap().views[&walker.id];
        let n = brick_dims(w.tex_dims).iter().map(|&n| n as usize).product::<usize>();
        write_3d(&queue, &w.value_bricks, brick_dims(w.tex_dims), 4, bytemuck::cast_slice(&vec![[0u16, 0x3c00]; n]));
        write_3d(&queue, &w.occupancy_bricks, brick_dims(w.tex_dims), 1, &vec![255; n]);
        // A view at a fractional place on a 1.5× screen, as egui may put it.
        let (screen, ppp) = ([230, 170], 1.5);
        let rect = Rect::from_min_size(Pos2::new(9.4, 6.7), egui::vec2(131.3, 97.9));
        let bg = Color32::from_rgb(12, 14, 18);
        let mut checked = 0;
        for (mode, window, threshold) in [(Mode3d::Mip, (500.0, 1000.0), None), (Mode3d::Mip, (400.0, 300.0), None), (Mode3d::Volume, (400.0, 1600.0), None), (Mode3d::Volume, (900.0, 300.0), None), (Mode3d::Surface, (0.0, 100.0), Some(600.0)), (Mode3d::Surface, (0.0, 100.0), Some(1100.0))] {
            for shown in [false, true] {
                for (yaw, pitch, zoom) in [(0.3, 0.15, 1.0), (2.2, -0.6, 2.5)] {
                    (view.mode, view.threshold, view.show_labels) = (mode, threshold, shown);
                    (view.yaw, view.pitch, view.zoom) = (yaw, pitch, zoom);
                    state.window = window;
                    let up = view.sync_with(&device, &queue, TARGET, &mut res, &vol, Some(&overlay), 2048);
                    let uniforms = view.uniforms(&vol, &state, Some(&overlay), rect, bg, &up);
                    let draw = |id, half| Draw { id, rect, uniforms, half };
                    let what = format!("{mode:?} window {window:?} segments {shown} yaw {yaw}");
                    // Drawn directly (as on an sRGB target): skipping bricks changes nothing.
                    let blitter = res.get_mut::<Gpu>().unwrap().blitter.take();
                    let direct = render(&device, &queue, &mut res, &draw(view.id, false), screen, ppp);
                    let walked = render(&device, &queue, &mut res, &draw(walker.id, false), screen, ppp);
                    assert_eq!(diff(&walked, &direct), (0, 0), "skipping bricks changed pixels: {what}");
                    if mode == Mode3d::Mip {
                        // Nor does stopping a saturated MIP (checked against a pipeline compiled
                        // without it, which the compiler may round differently in a pixel).
                        let gpu = res.get_mut::<Gpu>().unwrap();
                        let slot = pipeline_slot(&uniforms);
                        let skipping = gpu.pipelines[slot].replace(ray_pipeline(&device, &gpu.shader, &gpu.pipeline_layout, TARGET, slot, false));
                        let full = render(&device, &queue, &mut res, &draw(view.id, false), screen, ppp);
                        res.get_mut::<Gpu>().unwrap().pipelines[slot] = skipping;
                        let (n, max) = diff(&full, &direct);
                        assert!(max <= 1 && n * 1000 <= full.len() / 4, "stopping saturated rays changed {n} pixels by up to {max}: {what}");
                    }
                    res.get_mut::<Gpu>().unwrap().blitter = blitter;
                    // At half size while moving, then cached in full, then shown from the cache.
                    let moving = render(&device, &queue, &mut res, &draw(view.id, true), screen, ppp);
                    assert!(diff(&moving, &direct).0 > 0, "{what}");
                    for _ in 0..2 {
                        let cached = render(&device, &queue, &mut res, &draw(view.id, false), screen, ppp);
                        assert_eq!(diff(&cached, &direct), (0, 0), "cached frame: {what}");
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 24);
        // The view moved on screen (its frame the same): drawn again where it is now.
        let moved = rect.translate(egui::vec2(21.0, 13.3));
        let up = view.sync_with(&device, &queue, TARGET, &mut res, &vol, Some(&overlay), 2048);
        let draw = Draw { id: view.id, rect: moved, uniforms: view.uniforms(&vol, &state, Some(&overlay), moved, bg, &up), half: false };
        let cached = render(&device, &queue, &mut res, &draw, screen, ppp);
        let blitter = res.get_mut::<Gpu>().unwrap().blitter.take();
        let direct = render(&device, &queue, &mut res, &draw, screen, ppp);
        res.get_mut::<Gpu>().unwrap().blitter = blitter;
        assert_eq!(diff(&cached, &direct), (0, 0), "moved view");
        // A frame that egui prepares but drops unsubmitted (no surface texture): the next one
        // still shows the view.
        view.yaw += 0.4;
        let draw = Draw { id: view.id, rect: moved, uniforms: view.uniforms(&vol, &state, Some(&overlay), moved, bg, &up), half: false };
        let mut dropped = device.create_command_encoder(&Default::default());
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: screen, pixels_per_point: ppp };
        drop(egui_wgpu::CallbackTrait::prepare(&draw, &device, &queue, &sd, &mut dropped, &mut res));
        drop(dropped);
        let cached = render(&device, &queue, &mut res, &draw, screen, ppp);
        let blitter = res.get_mut::<Gpu>().unwrap().blitter.take();
        let direct = render(&device, &queue, &mut res, &draw, screen, ppp);
        res.get_mut::<Gpu>().unwrap().blitter = blitter;
        assert_eq!(diff(&cached, &direct), (0, 0), "after a dropped frame");
        // Smoothing changed only: the same as uploading everything again.
        view.label_smoothing = 3;
        let up = view.sync_with(&device, &queue, TARGET, &mut res, &vol, Some(&overlay), 2048);
        let uniforms = view.uniforms(&vol, &state, Some(&overlay), rect, bg, &up);
        let draw = Draw { id: view.id, rect, uniforms, half: false };
        let smoothed = render(&device, &queue, &mut res, &draw, screen, ppp);
        res.get_mut::<Gpu>().unwrap().views.get_mut(&view.id).unwrap().label_key = None;
        view.sync_with(&device, &queue, TARGET, &mut res, &vol, Some(&overlay), 2048);
        assert_eq!(diff(&render(&device, &queue, &mut res, &draw, screen, ppp), &smoothed), (0, 0));
        // Another volume on the same grid is uploaded, never taken for the first.
        let (other, _) = phantom([60, 52, 44], 2);
        view.sync_with(&device, &queue, TARGET, &mut res, &other, Some(&overlay), 2048);
        let source = &res.get::<Gpu>().unwrap().views[&view.id].source;
        assert_eq!(source.as_ptr(), Arc::as_ptr(&other));
    }

    #[test]
    fn segments_of_strided_volumes_keep_their_colours() {
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        // Labels only (no image shown): a red slab across the middle.
        let dims = [40, 36, 30];
        let g = mm_volume::geometry::Geometry::from_axes([40, 36, 30, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = Arc::new(LoadedVolume::from_values("v", dims, vec![0.0; 43200], g));
        let labels: Vec<u8> = (0..43200).map(|i| ((i / 40) % 36 >= 14 && (i / 40) % 36 < 22) as u8).collect();
        let colours = [Color32::TRANSPARENT, Color32::from_rgb(230, 30, 30)];
        let overlay = LabelOverlay { labels: &labels, colours: &colours, opacity: 1.0, version: 1 };
        let mut view = VolumeView3d::new();
        view.pitch = 0.0;
        let state = ViewState::new(&vol);
        let rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(64.0, 64.0));
        for max_dim in [2048, 20] {
            let mut res = egui_wgpu::CallbackResources::default();
            let up = view.sync_with(&device, &queue, TARGET, &mut res, &vol, Some(&overlay), max_dim);
            assert_eq!(up.stride, if max_dim == 20 { 2 } else { 1 });
            let uniforms = view.uniforms(&vol, &state, Some(&overlay), rect, Color32::BLACK, &up);
            let px = render(&device, &queue, &mut res, &Draw { id: view.id, rect, uniforms, half: false }, [64, 64], 1.0);
            // Shaded red (with a highlight), not the black of no label.
            let centre = &px[(32 * 64 + 32) * 4..][..3];
            assert!(centre[0] > 150 && centre[0] > 2 * centre[1] && centre[0] > 2 * centre[2], "stride {}: {centre:?}", up.stride);
        }
    }
}
