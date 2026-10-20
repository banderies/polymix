//! Image slices in the Mesh 3D view: the axial, sagittal and coronal planes of a volume
//! through a voxel, drawn as textured quads among the meshes (depth-tested, seen from both
//! sides) so a segmented surface can be checked against the image it came from.
//!
//! As in the 2D views the planes follow the voxel grid (no resampling): each is the voxel
//! slice through the cursor, grey through the window, and its quad's corners are the outer
//! corners (±0.5 voxel) of the slice's first and last voxels, in LPS millimetres like meshes
//! made from segments (or RAS). Textures are re-made only when the volume, the slice or the
//! window changes; slices wider than [`MAX_TEXTURE`] are averaged over blocks of voxels.

// The Mesh view's controls (and camera fitting) use this API once they are wired up.

use crate::renderer::Globals;
use bytemuck::{Pod, Zeroable};
use eframe::wgpu;
use glam::{DVec3, Mat4, Vec3};
use mm_core::Aabb;
use mm_volume::load::LoadedVolume;
use mm_volume::slice::{plane_axes, Plane, PlaneAxes};
use polymix_imaging::plane_colour;
use rayon::prelude::*;
use std::sync::{Arc, Weak};

/// The planes in the order of [`PlaneSettings::show`] (Slicer's red, yellow, green views).
pub const PLANES: [Plane; 3] = [Plane::Axial, Plane::Sagittal, Plane::Coronal];

/// Most pixels along either side of a plane's texture.
pub const MAX_TEXTURE: usize = 2048;

/// How the planes are shown.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneSettings {
    /// Voxel coordinates (i, j, k) the planes go through (rounded to the voxel).
    pub cursor: [f64; 3],
    /// Window (centre, width).
    pub window: (f64, f64),
    /// Axial, sagittal, coronal.
    pub show: [bool; 3],
    /// 0 (only the outlines) to 1 (opaque).
    pub opacity: f32,
    /// World positions in LPS (meshes made from segments, Slicer models); off: RAS.
    pub lps: bool,
    /// Outline each plane in its view colour (axial red, sagittal yellow, coronal green).
    pub outline: bool,
}

impl PlaneSettings {
    /// All three planes through the middle voxel, opaque and outlined, with the automatic
    /// window, in LPS.
    pub fn new(vol: &LoadedVolume) -> Self {
        let mid = |a: usize| (vol.dims[a].saturating_sub(1) / 2) as f64;
        Self { cursor: [mid(0), mid(1), mid(2)], window: vol.auto_window, show: [true; 3], opacity: 1.0, lps: true, outline: true }
    }
}

/// One plane ready to upload: RGBA pixels row by row from the top left, and where they go.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneImage {
    /// Pixels across and down.
    pub size: [usize; 2],
    pub rgba: Vec<u8>,
    /// World positions (mm) of the slice's corners at the image's top left, top right,
    /// bottom right and bottom left.
    pub corners: [Vec3; 4],
    /// Texture coordinates of the bottom-right corner: below 1 when the last pixels of a
    /// strided image average fewer voxels (the image runs past the slice).
    pub uv_max: [f32; 2],
}

/// What the renderer draws: a volume and how its planes are shown.
#[derive(Clone)]
pub struct PlanesFrame {
    pub volume: Arc<LoadedVolume>,
    pub settings: PlaneSettings,
}

/// Voxel index nearest coordinate `x` on an axis of `n` voxels.
fn nearest(x: f64, n: usize) -> usize {
    x.round().clamp(0.0, n.saturating_sub(1) as f64) as usize
}

/// How `plane` cuts `vol` (the same voxel axes as the 2D views) and its slice through `cursor`.
fn cut(vol: &LoadedVolume, plane: Plane, cursor: [f64; 3]) -> (PlaneAxes, usize) {
    let axes = plane_axes(&vol.geometry, plane);
    (axes, nearest(cursor[axes.slice], vol.dims[axes.slice]))
}

/// World corners (mm) of `slice`: the outer corners of its voxels, in the image's order (top
/// left, top right, bottom right, bottom left; the image runs along the voxel axes).
fn corners(vol: &LoadedVolume, axes: PlaneAxes, slice: usize, lps: bool) -> [DVec3; 4] {
    let (w, h) = (vol.dims[axes.across] as f64, vol.dims[axes.down] as f64);
    [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(a, d)| {
        let mut v = DVec3::ZERO;
        v[axes.across] = a - 0.5;
        v[axes.down] = d - 0.5;
        v[axes.slice] = slice as f64;
        let p = vol.geometry.world(v);
        // RAS → LPS: x and y change sign.
        if lps { DVec3::new(-p.x, -p.y, p.z) } else { p }
    })
}

fn usable(vol: &LoadedVolume) -> bool {
    vol.voxel_count() > 0 && vol.values.len() == vol.voxel_count()
}

/// The image of `plane` as `s` shows it (`None` when it is hidden).
pub fn plane_image(vol: &LoadedVolume, plane: Plane, s: &PlaneSettings) -> Option<PlaneImage> {
    let shown = PLANES.iter().position(|&p| p == plane).is_some_and(|i| s.show[i]);
    if !shown || !usable(vol) {
        return None;
    }
    let (axes, slice) = cut(vol, plane, s.cursor);
    let (w, h) = (vol.dims[axes.across], vol.dims[axes.down]);
    // Voxels per pixel along both sides.
    let step = w.max(h).div_ceil(MAX_TEXTURE);
    let size = [w.div_ceil(step), h.div_ceil(step)];
    let stride = [1, vol.dims[0], vol.dims[0] * vol.dims[1]];
    let (sa, sd, base) = (stride[axes.across], stride[axes.down], slice * stride[axes.slice]);
    // Grey levels as the 2D views map them.
    let (c, ww) = s.window;
    let lo = c - ww * 0.5;
    let scale = 255.0 / ww.max(f64::MIN_POSITIVE);
    let grey = |x: f64| {
        let g = ((x - lo) * scale).round().clamp(0.0, 255.0) as u8;
        if vol.invert { 255 - g } else { g }
    };
    let mut rgba = vec![255u8; size[0] * size[1] * 4];
    rgba.par_chunks_mut(size[0] * 4).enumerate().for_each(|(y, row)| {
        let rows = y * step..((y + 1) * step).min(h);
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let value = if step == 1 {
                vol.values[base + y * sd + x * sa] as f64
            } else {
                let cols = x * step..((x + 1) * step).min(w);
                let n = rows.len() * cols.len();
                rows.clone().flat_map(|d| cols.clone().map(move |a| base + d * sd + a * sa)).map(|k| vol.values[k] as f64).sum::<f64>() / n as f64
            };
            px[..3].fill(grey(value));
        }
    });
    Some(PlaneImage {
        size,
        rgba,
        corners: corners(vol, axes, slice, s.lps).map(|p| p.as_vec3()),
        uv_max: [w as f32 / (size[0] * step) as f32, h as f32 / (size[1] * step) as f32],
    })
}

/// Axial, sagittal and coronal images (`None` for hidden planes).
#[cfg(test)]
pub fn planes(vol: &LoadedVolume, s: &PlaneSettings) -> [Option<PlaneImage>; 3] {
    PLANES.map(|p| plane_image(vol, p, s))
}

/// World bounds of the shown planes (`None` when none is), for fitting the camera.
pub fn bounds(vol: &LoadedVolume, s: &PlaneSettings) -> Option<Aabb> {
    if !usable(vol) {
        return None;
    }
    let shown = PLANES.iter().zip(s.show).filter(|&(_, on)| on);
    let b = Aabb::from_points(shown.flat_map(|(&p, _)| {
        let (axes, slice) = cut(vol, p, s.cursor);
        corners(vol, axes, slice, s.lps)
    }));
    (!b.is_empty()).then_some(b)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaneUniform {
    /// Top-left corner, and the edges to the top-right and bottom-left corners.
    origin: [f32; 4],
    across: [f32; 4],
    down: [f32; 4],
    /// Texture coordinates of the bottom-right corner (xy), where the other planes cut this
    /// one (zw, 0..1 across and down).
    tex: [f32; 4],
    /// Outline colour (linear RGB) and 1 with an outline.
    outline: [f32; 4],
    /// Opacity, -, -, -.
    misc: [f32; 4],
}

/// What a plane's texture shows.
struct Shown {
    volume: Weak<LoadedVolume>,
    slice: usize,
    window: [u64; 2],
}

impl Shown {
    fn same(&self, o: &Shown) -> bool {
        // A weak pointer keeps its allocation, so no other volume can take its address.
        Weak::ptr_eq(&self.volume, &o.volume) && self.slice == o.slice && self.window == o.window
    }
}

/// GPU side of one plane.
struct PlaneGpu {
    texture: wgpu::Texture,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    shown: Shown,
    uv_max: [f32; 2],
}

/// Draws the planes of a [`PlanesFrame`] into the Mesh renderer's pass (its colour, depth
/// and MSAA formats, reversed Z, its globals as bind group 0).
pub struct ImagePlanesGpu {
    opaque: wgpu::RenderPipeline,
    see_through: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// What the next frame shows (see [`Self::set`]).
    frame: Option<PlanesFrame>,
    planes: [Option<PlaneGpu>; 3],
    /// Quarters (plane, quarter) to draw this frame, back to front.
    pieces: Vec<(usize, u32)>,
    blended: bool,
}

impl ImagePlanesGpu {
    pub fn new(device: &wgpu::Device, globals: &wgpu::BindGroupLayout, color: wgpu::TextureFormat, depth: wgpu::TextureFormat, samples: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image_planes.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/image_planes.wgsl").into()),
        });
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image plane"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: fragment,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: fragment, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("image planes"), bind_group_layouts: &[Some(globals), Some(&layout)], immediate_size: 0 });
        let pipeline = |see_through: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if see_through { "image planes (see-through)" } else { "image planes" }),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_plane"), compilation_options: Default::default(), buffers: &[] },
                // Seen from both sides.
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                // Reversed Z as for the meshes; see-through planes leave the depth alone.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth,
                    depth_write_enabled: Some(!see_through),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_plane"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: color, blend: see_through.then_some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("image plane"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self { opaque: pipeline(false), see_through: pipeline(true), layout, sampler, frame: None, planes: Default::default(), pieces: Vec::new(), blended: false }
    }

    /// The planes to show in the next frame only (`None`: none; their textures are freed).
    pub fn set(&mut self, frame: Option<PlanesFrame>) {
        self.frame = frame;
    }

    /// Upload what this frame needs (textures only when a plane's volume, slice or window has
    /// changed) and order the pieces to draw back to front for `globals`' view.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, globals: &Globals) {
        self.pieces.clear();
        let Some(PlanesFrame { volume: vol, settings: s }) = self.frame.take() else {
            self.planes = Default::default();
            return;
        };
        let opacity = s.opacity.clamp(0.0, 1.0);
        if opacity == 0.0 && !s.outline {
            return;
        }
        self.blended = opacity < 1.0;
        let view = Mat4::from_cols_array_2d(&globals.view);
        let mut pieces: Vec<(f32, usize, u32)> = Vec::new();
        for (i, &plane) in PLANES.iter().enumerate() {
            if !s.show[i] {
                continue;
            }
            let (axes, slice) = cut(&vol, plane, s.cursor);
            let shown = Shown { volume: Arc::downgrade(&vol), slice, window: [s.window.0.to_bits(), s.window.1.to_bits()] };
            if !self.planes[i].as_ref().is_some_and(|p| p.shown.same(&shown)) {
                let Some(img) = plane_image(&vol, plane, &s) else { continue };
                self.upload(device, queue, i, &img, shown);
            }
            let Some(gpu) = &self.planes[i] else { continue };
            let [o, a, _, d] = corners(&vol, axes, slice, s.lps).map(|p| p.as_vec3());
            let (across, down) = (a - o, d - o);
            // The other planes cut this one through the centres of the cursor's voxels.
            let split = |axis: usize| (nearest(s.cursor[axis], vol.dims[axis]) as f32 + 0.5) / vol.dims[axis] as f32;
            let (sx, sy) = (split(axes.across), split(axes.down));
            let c = egui::Rgba::from(plane_colour(plane));
            let u = PlaneUniform {
                origin: o.extend(1.0).to_array(),
                across: across.extend(0.0).to_array(),
                down: down.extend(0.0).to_array(),
                tex: [gpu.uv_max[0], gpu.uv_max[1], sx, sy],
                outline: [c.r(), c.g(), c.b(), s.outline as u32 as f32],
                misc: [opacity, 0.0, 0.0, 0.0],
            };
            queue.write_buffer(&gpu.uniform, 0, bytemuck::bytes_of(&u));
            for q in 0..4u32 {
                let mid = |upper: bool, at: f32| if upper { (at + 1.0) * 0.5 } else { at * 0.5 };
                let p = o + across * mid(q & 1 != 0, sx) + down * mid(q & 2 != 0, sy);
                pieces.push((view.transform_point3(p).z, i, q));
            }
        }
        // The view looks down -z: the farthest pieces have the lowest z.
        pieces.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.pieces = pieces.into_iter().map(|(_, i, q)| (i, q)).collect();
    }

    /// Put `img` in plane `i`'s texture (re-made when its size changes).
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, i: usize, img: &PlaneImage, shown: Shown) {
        let size = wgpu::Extent3d { width: img.size[0] as u32, height: img.size[1] as u32, depth_or_array_layers: 1 };
        let mut gpu = match self.planes[i].take().filter(|p| p.texture.size() == size) {
            Some(p) => p,
            None => {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("image plane"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // Grey levels are display values, as in the 2D views.
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("image plane"),
                    size: std::mem::size_of::<PlaneUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let view = texture.create_view(&Default::default());
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("image plane"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ],
                });
                PlaneGpu { texture, uniform, bind, shown: Shown { volume: Weak::new(), slice: 0, window: [0; 2] }, uv_max: [1.0; 2] }
            }
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &gpu.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &img.rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size.width * 4), rows_per_image: Some(size.height) },
            size,
        );
        gpu.shown = shown;
        gpu.uv_max = img.uv_max;
        self.planes[i] = Some(gpu);
    }

    /// Draw the planes prepared for this frame. Call with `see_through` false after the
    /// opaque meshes and true after the lines (before see-through meshes): only the call
    /// matching the planes' opacity draws.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, see_through: bool) {
        if self.pieces.is_empty() || see_through != self.blended {
            return;
        }
        pass.set_pipeline(if see_through { &self.see_through } else { &self.opaque });
        let mut bound = None;
        for &(i, q) in &self.pieces {
            let Some(p) = &self.planes[i] else { continue };
            if bound != Some(i) {
                pass.set_bind_group(1, &p.bind, &[]);
                bound = Some(i);
            }
            pass.draw(0..6, q..q + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::geometry::Geometry;

    /// A volume whose values are the voxels' indices (grey levels themselves with the
    /// window (127.5, 255)), stored along `axes` from `origin`.
    fn volume(dims: [usize; 3], spacing: [f64; 3], axes: [DVec3; 3], origin: DVec3) -> LoadedVolume {
        let g = Geometry::from_axes([dims[0], dims[1], dims[2], 1], spacing, axes, origin);
        let mut values = Vec::new();
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    values.push((i + dims[0] * (j + dims[1] * k)) as f32);
                }
            }
        }
        LoadedVolume::from_values("test", dims, values, g)
    }

    /// i toward L, j toward P, k toward I (a flipped axis), from (10, -20, 30) RAS mm.
    fn lpi() -> LoadedVolume {
        volume([4, 5, 6], [0.5, 0.8, 1.5], [-DVec3::X, -DVec3::Y, -DVec3::Z], DVec3::new(10.0, -20.0, 30.0))
    }

    fn settings(vol: &LoadedVolume) -> PlaneSettings {
        PlaneSettings { cursor: [1.2, 2.7, 3.4], window: (127.5, 255.0), ..PlaneSettings::new(vol) }
    }

    fn close(a: Vec3, b: DVec3) -> bool {
        (a.as_dvec3() - b).length() < 1e-4
    }

    #[test]
    fn axial_corners_sit_on_the_voxel_corners() {
        let vol = lpi();
        // RAS position of voxel (i, j, k), worked out by hand.
        let ras = |i: f64, j: f64, k: f64| DVec3::new(10.0 - 0.5 * i, -20.0 - 0.8 * j, 30.0 - 1.5 * k);
        // The cursor's slice is k = 3; the image runs along i (4 voxels) and j (5).
        let expected = [ras(-0.5, -0.5, 3.0), ras(3.5, -0.5, 3.0), ras(3.5, 4.5, 3.0), ras(-0.5, 4.5, 3.0)];
        for lps in [false, true] {
            let s = PlaneSettings { lps, ..settings(&vol) };
            let img = plane_image(&vol, Plane::Axial, &s).unwrap();
            assert_eq!(img.size, [4, 5]);
            assert_eq!(img.uv_max, [1.0, 1.0]);
            for (c, e) in img.corners.iter().zip(expected) {
                let e = if lps { DVec3::new(-e.x, -e.y, e.z) } else { e };
                assert!(close(*c, e), "lps {lps}: {c} != {e}");
            }
        }
    }

    #[test]
    fn every_pixel_lies_on_its_voxel() {
        // Pixel centres, placed by the corners, land on the voxels whose values they show:
        // in every plane, for an LPI grid and a sagittal (PIR) acquisition.
        let pir = volume([5, 6, 4], [0.8, 1.5, 0.5], [-DVec3::Y, -DVec3::Z, DVec3::X], DVec3::new(-3.0, 7.0, 2.0));
        for vol in [lpi(), pir] {
            let s = PlaneSettings { lps: false, ..settings(&vol) };
            for (n, plane) in PLANES.into_iter().enumerate() {
                let img = plane_image(&vol, plane, &s).unwrap();
                let [o, a, _, d] = img.corners;
                let (w, h) = (img.size[0], img.size[1]);
                for y in 0..h {
                    for x in 0..w {
                        let v = img.rgba[(y * w + x) * 4] as usize;
                        let [d0, d1, _] = vol.dims;
                        let voxel = DVec3::new((v % d0) as f64, (v / d0 % d1) as f64, (v / (d0 * d1)) as f64);
                        let st = Vec3::new((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32, 0.0);
                        let p = o + (a - o) * st.x + (d - o) * st.y;
                        assert!(close(p, vol.geometry.world(voxel)), "{plane:?} pixel ({x}, {y}): {p} vs voxel {voxel}");
                    }
                }
                // Each plane is flat along its own world axis (S, R, A).
                let axis = [2, 0, 1][n];
                assert!(img.corners.iter().all(|c| (c[axis] - o[axis]).abs() < 1e-4), "{plane:?} is not flat");
            }
        }
    }

    #[test]
    fn grey_follows_the_window_and_invert() {
        let mut vol = lpi();
        // Axial slice k = 3 holds 60 + i + 4j: 60..=79.
        let s = PlaneSettings { window: (70.0, 10.0), ..settings(&vol) };
        let grey = |vol: &LoadedVolume, x: usize, y: usize| plane_image(vol, Plane::Axial, &s).unwrap().rgba[(y * 4 + x) * 4];
        assert_eq!(grey(&vol, 0, 0), 0); // 60, below the window
        assert_eq!(grey(&vol, 1, 2), 102); // 69: (69 - 65) * 255 / 10
        assert_eq!(grey(&vol, 3, 4), 255); // 79, above it
        vol.invert = true;
        assert_eq!(grey(&vol, 0, 0), 255);
        assert_eq!(grey(&vol, 1, 2), 153);
        assert_eq!(grey(&vol, 3, 4), 0);
        let img = plane_image(&vol, Plane::Axial, &s).unwrap();
        assert!(img.rgba.chunks(4).all(|p| p[0] == p[1] && p[1] == p[2] && p[3] == 255));
    }

    #[test]
    fn hidden_planes_are_none() {
        let vol = lpi();
        let s = PlaneSettings { show: [false, true, false], ..settings(&vol) };
        let [axial, sagittal, coronal] = planes(&vol, &s);
        assert!(axial.is_none() && coronal.is_none());
        // Sagittal: slice i = 1, across j (5), down k (6).
        assert_eq!(sagittal.unwrap().size, [5, 6]);
        assert!(bounds(&vol, &PlaneSettings { show: [false; 3], ..s }).is_none());
    }

    #[test]
    fn bounds_cover_the_shown_planes() {
        let vol = lpi();
        let s = settings(&vol);
        let b = bounds(&vol, &s).unwrap();
        // All three planes together span the whole grid (LPS).
        let ras_min = DVec3::new(10.0 - 0.5 * 3.5, -20.0 - 0.8 * 4.5, 30.0 - 1.5 * 5.5);
        let ras_max = DVec3::new(10.25, -19.6, 30.75);
        assert!((b.min - DVec3::new(-ras_max.x, -ras_max.y, ras_min.z)).length() < 1e-9);
        assert!((b.max - DVec3::new(-ras_min.x, -ras_min.y, ras_max.z)).length() < 1e-9);
        // The axial plane alone is flat at its slice.
        let b = bounds(&vol, &PlaneSettings { show: [true, false, false], ..s }).unwrap();
        assert!((b.min.z - 25.5).abs() < 1e-9 && (b.max.z - 25.5).abs() < 1e-9);
    }

    #[test]
    fn big_slices_are_strided() {
        // 5000 voxels across: 3 per pixel, values 0 1 2 repeating, so every block averages 1
        // (the last one, two voxels, 0.5 → 1 too).
        let dims = [5000, 3, 2];
        let g = Geometry::from_axes([dims[0], dims[1], dims[2], 1], [0.1, 1.0, 1.0], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let values = (0..dims.iter().product::<usize>()).map(|n| (n % dims[0] % 3) as f32).collect();
        let vol = LoadedVolume::from_values("big", dims, values, g);
        let s = PlaneSettings { cursor: [0.0; 3], window: (127.5, 255.0), lps: false, ..PlaneSettings::new(&vol) };
        let img = plane_image(&vol, Plane::Axial, &s).unwrap();
        assert_eq!(img.size, [1667, 1]);
        assert!(img.size.iter().all(|&n| n <= MAX_TEXTURE));
        assert!(img.rgba.chunks(4).all(|p| p[0] == 1), "blocks are averaged");
        // The quad still covers exactly the slice; the texture runs a voxel past it.
        assert!(close(img.corners[2], DVec3::new(499.95, 2.5, 0.0)));
        assert_eq!(img.uv_max, [5000.0 / 5001.0, 1.0]);
        // Sagittal: 3 × 2, untouched.
        assert_eq!(plane_image(&vol, Plane::Sagittal, &s).unwrap().size, [3, 2]);
    }

    #[test]
    fn shader_compiles() {
        use eframe::wgpu::naga;
        let module = naga::front::wgsl::parse_str(include_str!("shaders/image_planes.wgsl")).unwrap();
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap();
    }
}
