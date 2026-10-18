//! The 3D view of a volume: GPU ray casting (`shaders/volume3d.wgsl`) drawn straight into
//! egui's render pass. Modes: maximum intensity projection, shaded volume rendering (opacity
//! follows the window), or the first surface above a threshold; label surfaces (segments) are
//! drawn in front, coloured like the slice overlays.
//!
//! Mouse: left drag rotates, right drag pans, middle drag or the wheel zooms, double-click
//! resets the camera. The volume is uploaded once (as normalised 16-bit floats, every n-th
//! voxel if it exceeds the GPU's limits); labels again whenever their version changes.

use crate::slice_view::{LabelOverlay, ViewState};
use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, wgpu};
use egui::{Color32, Rect, Response, Sense, Vec2};
use glam::{DMat4, DVec3, Vec3};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// Most texels uploaded per volume (16-bit each).
const MAX_TEXELS: usize = 160_000_000;

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
    palette: [[f32; 4]; 256],
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct VolumeView3d {
    pub mode: Mode3d,
    /// Surface threshold in the volume's units (`None`: the middle of the window).
    pub threshold: Option<f64>,
    /// Opacity of volume rendering (1 = default).
    pub opacity: f32,
    pub show_labels: bool,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pan: Vec3,
    id: u64,
    /// Kept alive with the view; the GPU side drops its textures once it is gone.
    token: Arc<()>,
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
    volume_key: usize,
    label_key: Option<u64>,
    /// Texels per axis and the voxel stride they were sampled with.
    tex_dims: [u32; 3],
    stride: usize,
}

/// Shared by every 3D view, kept in egui-wgpu's callback resources.
struct Gpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    views: HashMap<u64, ViewGpu>,
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume3d"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(1, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                tex(3, wgpu::TextureSampleType::Float { filterable: true }),
                tex(4, wgpu::TextureSampleType::Uint),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("volume3d"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume3d"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume3d"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self { pipeline, layout, sampler, views: HashMap::new() }
    }
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

impl VolumeView3d {
    pub fn new() -> Self {
        Self { mode: Mode3d::Volume, threshold: None, opacity: 1.0, show_labels: true, yaw: 0.0, pitch: 0.15, zoom: 1.0, pan: Vec3::ZERO, id: NEXT_ID.fetch_add(1, Ordering::Relaxed), token: Arc::new(()) }
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

    /// Draw the view in the space left in `ui`, with its controls in the top left corner.
    pub fn ui(&mut self, ui: &mut egui::Ui, render_state: Option<&egui_wgpu::RenderState>, vol: &Arc<LoadedVolume>, state: &ViewState, labels: Option<&LabelOverlay>) -> Response {
        let rect = ui.available_rect_before_wrap();
        let r = ui.allocate_rect(rect, Sense::click_and_drag());
        // Dark whatever the theme, as medical viewers have it.
        let bg = Color32::from_rgb(12, 14, 18);
        let Some(rs) = render_state else {
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "The 3D view needs the GPU renderer", egui::FontId::proportional(13.0), ui.visuals().weak_text_color());
            return r;
        };
        self.navigate(ui, &r, vol);
        if let Err(e) = self.sync(rs, vol, labels) {
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, e, egui::FontId::proportional(13.0), ui.visuals().warn_fg_color);
            return r;
        }
        let uniforms = self.uniforms(vol, state, labels, rect, bg);
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, Draw { id: self.id, uniforms }));
        self.controls(ui, rect, vol, labels.is_some());
        r
    }

    fn navigate(&mut self, ui: &egui::Ui, r: &Response, vol: &LoadedVolume) {
        // Include the movement before the drag threshold, so a grabbed point follows exactly.
        let d = match (r.drag_started(), ui.input(|i| (i.pointer.press_origin(), i.pointer.latest_pos()))) {
            (true, (Some(a), Some(b))) => b - a,
            _ => r.drag_delta(),
        };
        if r.dragged_by(egui::PointerButton::Primary) {
            self.yaw -= d.x * 0.01;
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
        if r.double_clicked() {
            self.reset_camera();
        }
        if r.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let f = (scroll * 0.003).exp() * pinch;
            if f != 1.0 {
                self.zoom = (self.zoom * f).clamp(0.1, 40.0);
            }
            ui.ctx().input_mut(|i| i.smooth_scroll_delta = Vec2::ZERO);
        }
    }

    /// Create or update this view's textures.
    fn sync(&self, rs: &egui_wgpu::RenderState, vol: &Arc<LoadedVolume>, labels: Option<&LabelOverlay>) -> Result<(), String> {
        let mut renderer = rs.renderer.write();
        let res = &mut renderer.callback_resources;
        if !res.contains::<Gpu>() {
            res.insert(Gpu::new(&rs.device, rs.target_format));
        }
        let Gpu { layout, sampler, views, .. } = res.get_mut::<Gpu>().unwrap();
        views.retain(|_, v| v.alive.strong_count() > 0);
        let key = Arc::as_ptr(vol) as usize;
        let (device, queue) = (&rs.device, &rs.queue);
        if views.get(&self.id).is_none_or(|v| v.volume_key != key) {
            let stride = stride_for(vol.dims, device.limits().max_texture_dimension_3d);
            let td = vol.dims.map(|d| d.div_ceil(stride) as u32);
            let (lo, hi) = vol.range;
            let scale = if hi > lo { 1.0 / (hi - lo) } else { 0.0 };
            let bytes: Vec<u8> = sampled(vol, td, stride, |i| half((vol.values[i] - lo) * scale)).into_iter().flat_map(u16::to_le_bytes).collect();
            let volume = texture_3d(device, "volume3d.values", td, wgpu::TextureFormat::R16Float);
            write_3d(queue, &volume, td, 2, &bytes);
            let labels_t = texture_3d(device, "volume3d.labels", [1, 1, 1], wgpu::TextureFormat::R8Uint);
            let occupancy = texture_3d(device, "volume3d.occupancy", [1, 1, 1], wgpu::TextureFormat::R8Unorm);
            write_3d(queue, &labels_t, [1, 1, 1], 1, &[0]);
            write_3d(queue, &occupancy, [1, 1, 1], 1, &[0]);
            let uniforms = device.create_buffer(&wgpu::BufferDescriptor { label: Some("volume3d.uniforms"), size: std::mem::size_of::<Uniforms>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let bind = bind_group(device, layout, sampler, &uniforms, &volume, &occupancy, &labels_t);
            views.insert(self.id, ViewGpu { alive: Arc::downgrade(&self.token), uniforms, bind, volume, labels: labels_t, occupancy, volume_key: key, label_key: None, tex_dims: td, stride });
        }
        let want = labels.filter(|l| l.labels.len() == vol.voxel_count()).map(|l| l.version);
        let v = views.get_mut(&self.id).unwrap();
        if let (Some(l), true) = (labels, want.is_some() && want != v.label_key) {
            let (td, stride) = (v.tex_dims, v.stride);
            let ids = sampled(vol, td, stride, |i| l.labels[i]);
            let mut occ: Vec<u8> = ids.par_iter().map(|&id| if id != 0 && l.colours.get(id as usize).is_some_and(|c| c.a() > 0) { 255 } else { 0 }).collect();
            // Soften the voxel steps so label surfaces come out smooth.
            blur_121(&mut occ, td);
            blur_121(&mut occ, td);
            if v.labels.size().width != td[0] || v.labels.size().depth_or_array_layers != td[2] {
                v.labels = texture_3d(device, "volume3d.labels", td, wgpu::TextureFormat::R8Uint);
                v.occupancy = texture_3d(device, "volume3d.occupancy", td, wgpu::TextureFormat::R8Unorm);
                v.bind = bind_group(device, layout, sampler, &v.uniforms, &v.volume, &v.occupancy, &v.labels);
            }
            write_3d(queue, &v.labels, td, 1, &ids);
            write_3d(queue, &v.occupancy, td, 1, &occ);
            v.label_key = want;
        }
        Ok(())
    }

    fn uniforms(&self, vol: &LoadedVolume, state: &ViewState, labels: Option<&LabelOverlay>, rect: Rect, bg: Color32) -> Uniforms {
        let (eye, target, up, fov) = self.camera(vol);
        let (_, r) = Self::bounds(vol);
        let dist = (eye - target).length();
        let near = (dist - r as f32 * 1.2).max(dist * 0.01);
        let far = dist + r as f32 * 1.2;
        let proj = glam::camera::rh::proj::directx::perspective(fov, (rect.width() / rect.height().max(1.0)).max(1e-3), near, far);
        let view = glam::camera::rh::view::look_at_mat4(eye, target, up);
        let inv_view_proj = (proj * view).inverse();
        // World → texture coordinates: voxel centres at (index + 0.5) / size.
        let inv = vol.geometry.matrix().inverse();
        let world_to_tex = DMat4::from_scale(DVec3::new(1.0 / vol.dims[0] as f64, 1.0 / vol.dims[1] as f64, 1.0 / vol.dims[2] as f64)) * DMat4::from_translation(DVec3::splat(0.5)) * inv;
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
        Uniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            world_to_tex: world_to_tex.as_mat4().to_cols_array_2d(),
            grad_to_world: grad_to_world.as_mat4().to_cols_array_2d(),
            dims: [vol.dims[0] as f32, vol.dims[1] as f32, vol.dims[2] as f32, 1.5],
            window: [norm(c - w * 0.5), norm(c + w * 0.5), norm(threshold), match self.mode { Mode3d::Mip => 0.0, Mode3d::Volume => 1.0, Mode3d::Surface => 2.0 }],
            labels: [if self.show_labels && labels.is_some() { 1.0 } else { 0.0 }, labels.map(|l| l.opacity.max(0.5)).unwrap_or(1.0), 0.12 * self.opacity, if vol.invert { 1.0 } else { 0.0 }],
            background: [bgc[0], bgc[1], bgc[2], 1.0],
            light: [light.x, light.y, light.z, 0.0],
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

/// One pass of the [1 2 1] / 4 filter along each axis (a small Gaussian), in place.
fn blur_121(v: &mut [u8], td: [u32; 3]) {
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

fn bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, uniforms: &wgpu::Buffer, volume: &wgpu::Texture, occupancy: &wgpu::Texture, labels: &wgpu::Texture) -> wgpu::BindGroup {
    let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("volume3d"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view(volume)) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view(occupancy)) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view(labels)) },
        ],
    })
}

/// The paint callback of one 3D view for one frame.
struct Draw {
    id: u64,
    uniforms: Uniforms,
}

impl egui_wgpu::CallbackTrait for Draw {
    fn prepare(&self, _device: &wgpu::Device, queue: &wgpu::Queue, _screen: &egui_wgpu::ScreenDescriptor, _encoder: &mut wgpu::CommandEncoder, res: &mut egui_wgpu::CallbackResources) -> Vec<wgpu::CommandBuffer> {
        if let Some(v) = res.get::<Gpu>().and_then(|g| g.views.get(&self.id)) {
            queue.write_buffer(&v.uniforms, 0, bytemuck::bytes_of(&self.uniforms));
        }
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let Some(gpu) = res.get::<Gpu>() else { return };
        let Some(v) = gpu.views.get(&self.id) else { return };
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &v.bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
