//! wgpu renderer: draws the scene into an offscreen MSAA target that egui displays as an
//! image. Meshes are uploaded as triangle soups so per-triangle state (selection, face
//! group) can live in a compact storage buffer indexed by `vertex_index / 3`.
//!
//! Uploads are self-contained: each one writes its data and, when it needs copies, submits
//! them at once, so syncing and rendering may interleave in any order. A frame whose inputs
//! all equal the last one's (see [`SceneRenderer::render`]) is not drawn again.

use crate::image_planes::{ImagePlanesGpu, PlanesFrame};
use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu;
use eframe::wgpu;
use glam::{Mat3, Mat4, Vec3, Vec4};
use mm_core::hash::FastMap;
use mm_core::shading::ShadingNormals;
use mm_core::{Mesh, TriSelection};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const SAMPLE_COUNT: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub eye: [f32; 4],
    pub brush: [f32; 4],
    pub misc: [f32; 4],
    pub cut_plane: [f32; 4],
    pub cut_region: [f32; 4],
    /// Background colour at the top of the view (rgb; the bottom is a darker shade).
    pub background: [f32; 4],
}

/// Default background colour (top of the view).
pub const DEFAULT_BACKGROUND: [f32; 3] = [0.20, 0.21, 0.24];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectUniform {
    color: [f32; 4],
    flags: [u32; 4],
    /// Material: (specular, shininess, rim, metal), (wrap, -, -, -).
    material: [f32; 4],
    material2: [f32; 4],
    /// Transform of a rigid preview (with `flags[3]` bit 3) and its normal matrix (columns
    /// padded to four floats, as WGSL lays out a `mat3x3`).
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 3],
}

/// `ObjectUniform::flags[3]` bits beyond dim (1), cut tint (2) and X-ray (4).
const MOVED: u32 = 8;
const MIRRORED: u32 = 16;
const NO_SELECTION: u32 = 32;

const IDENTITY: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
const IDENTITY3: [[f32; 4]; 3] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];

impl ObjectUniform {
    /// Wireframe and colour modes 4-7 need the full pipeline; the rest draw with the lean one.
    fn needs_full_pipeline(&self) -> bool {
        self.flags[0] != 0 || self.flags[2] >= 4
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
    /// Vertex colour as 0xRRGGBB (used in the vertex-colour mode).
    color: u32,
    /// Texture coordinate of this corner (UV checker mode).
    uv: [f32; 2],
}

/// Bytes of one soup triangle.
const TRI_BYTES: usize = 3 * std::mem::size_of::<MeshVertex>();
/// Triangles built per parallel task when filling upload memory.
const FILL_CHUNK: usize = 1024;
/// Flags compared per piece when a selection changes (16 KiB): only changed pieces upload.
const FLAG_CHUNK: usize = 4096;
/// Largest packed-upload buffer a mesh keeps between strokes; bigger uploads use a
/// temporary one.
const MAX_KEPT_UPLOAD: u64 = 32 << 20;
/// Shorter inputs are handled on the calling thread: waking the thread pool costs more than
/// it saves on them (a brush stroke's few thousand triangles).
const PARALLEL_MIN: usize = 1 << 14;
/// Meshes up to this many triangles keep the corner normals of UV-edited triangles (live UV
/// editing works on meshes of this size).
const UV_NORMALS_MAX_TRIS: usize = 1 << 19;

/// The ids of `ids` (all below `bound`) without repeats, in the order they first come.
fn unique(ids: impl IntoIterator<Item = u32>, bound: usize) -> Vec<u32> {
    let mut seen = vec![0u64; bound.div_ceil(64)];
    ids.into_iter()
        .filter(|&i| {
            let (word, bit) = ((i / 64) as usize, 1u64 << (i % 64));
            let new = seen[word] & bit == 0;
            seen[word] |= bit;
            new
        })
        .collect()
}

/// UV of corner `j` of triangle `t` (zero without UVs).
fn corner_uv(uvs: Option<&[[[f32; 2]; 3]]>, t: u32, j: usize) -> [f32; 2] {
    uvs.and_then(|u| u.get(t as usize)).map_or([0.0; 2], |c| c[j])
}

/// Packed vertex colour of `v` (grey when the mesh has none).
fn packed_color(mesh: &Mesh, v: u32, on: bool) -> u32 {
    if !on {
        return 0;
    }
    let c = mesh.vertex_color(v).unwrap_or(mm_core::mesh::DEFAULT_VERTEX_COLOR);
    let b = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u32;
    (b(c[0]) << 16) | (b(c[1]) << 8) | b(c[2])
}

/// The soup corners of triangle `t`: every upload builds them here, so partial and full
/// uploads agree to the bit.
fn triangle_vertices(mesh: &Mesh, normals: &ShadingNormals, t: u32, colors: bool, uvs: Option<&[[[f32; 2]; 3]]>) -> [MeshVertex; 3] {
    soup_triangle(mesh, t, colors, uvs, corner_normals(normals, t))
}

/// The shading normals of `t`'s corners as the soup stores them.
fn corner_normals(normals: &ShadingNormals, t: u32) -> [[f32; 3]; 3] {
    std::array::from_fn(|j| normals.corner(t, j).as_vec3().to_array())
}

/// The soup corners of triangle `t` with these corner normals.
fn soup_triangle(mesh: &Mesh, t: u32, colors: bool, uvs: Option<&[[[f32; 2]; 3]]>, normals: [[f32; 3]; 3]) -> [MeshVertex; 3] {
    let tv = mesh.triangle(t);
    std::array::from_fn(|j| MeshVertex {
        position: mesh.vertex(tv[j]).as_vec3().to_array(),
        normal: normals[j],
        color: packed_color(mesh, tv[j], colors),
        uv: corner_uv(uvs, t, j),
    })
}

/// What corner normals depend on: shape and topology versions, crease angle (bits) and group
/// normals (with the group version).
type NormalsKey = (u64, u64, Option<u32>, Option<u64>);

/// Corner normals of triangles, as the soup stores them, by triangle id.
type KnownNormals = FastMap<u32, [[f32; 3]; 3]>;

/// Write the soup triangles `tris` (built by `build`, in order) into `buf` from byte
/// `offset`, filled in parallel straight into the queue's staging memory.
fn write_triangles(queue: &wgpu::Queue, buf: &wgpu::Buffer, offset: u64, tris: &[u32], build: impl Fn(u32) -> [MeshVertex; 3] + Sync) {
    let Some(size) = std::num::NonZeroU64::new((tris.len() * TRI_BYTES) as u64) else { return };
    // `None` only after a validation error, which wgpu has already reported.
    let Some(mut view) = queue.write_buffer_with(buf, offset, size) else { return };
    fill_triangles(view.slice(..), tris, build);
}

/// Fill `out` (mapped memory of `tris.len()` soup triangles) with the triangles `tris`
/// built by `build`, in order; in parallel for long inputs.
fn fill_triangles(out: wgpu::WriteOnly<'_, [u8]>, tris: &[u32], build: impl Fn(u32) -> [MeshVertex; 3] + Sync) {
    let fill = |out: wgpu::WriteOnly<'_, [u8]>, tris: &[u32]| {
        for (slot, &t) in out.into_chunks::<TRI_BYTES>().0.into_iter().zip(tris) {
            slot.write(bytemuck::cast(build(t)));
        }
    };
    if tris.len() < PARALLEL_MIN {
        fill(out, tris);
        return;
    }
    // Whole chunks go to the thread pool (a sized piece of the view can move between
    // threads), the remainder is filled here.
    let (chunks, rest) = out.into_chunks::<{ FILL_CHUNK * TRI_BYTES }>();
    let chunks: Vec<_> = chunks.into_iter().collect();
    let whole = chunks.len() * FILL_CHUNK;
    chunks.into_par_iter().zip(tris[..whole].par_chunks(FILL_CHUNK)).for_each(|(out, tris)| fill(out.into(), tris));
    fill(rest, &tris[whole..]);
}

/// Most soup triangles the device's buffers hold (very large meshes are truncated with a
/// warning rather than crashing).
fn max_soup_tris(device: &wgpu::Device) -> usize {
    (device.limits().max_buffer_size / TRI_BYTES as u64).min(device.limits().max_storage_buffer_binding_size / 4) as usize
}

/// What a soup is built from: everything `sync_mesh` compares before building it again.
#[derive(Clone, Copy, PartialEq, Debug)]
struct SoupKeys {
    shape_version: u64,
    topo_version: u64,
    crease_deg: Option<f32>,
    /// The mesh's group version, with group normals.
    group_normals: Option<u64>,
    /// The mesh's colour version, when vertex colours are shown.
    colors_key: Option<u64>,
    /// The UVs' stamp, when UVs are shown.
    uv_key: Option<u64>,
}

impl SoupKeys {
    /// The keys of `mesh` drawn with `style`, and the UV corners it shows (from `uvs`, its
    /// UVs' stamp and corners when it has any).
    fn of<'a>(mesh: &Mesh, style: &ObjectStyle, uvs: Option<(u64, &'a [[[f32; 2]; 3]])>) -> (Self, Option<&'a [[[f32; 2]; 3]]>) {
        let show_colors = style.vertex_colors && mesh.has_vertex_colors();
        let uvs = uvs.filter(|_| style.uv_checker || style.texture_mode != 0);
        let keys = SoupKeys {
            shape_version: mesh.shape_version(),
            topo_version: mesh.topology_version(),
            crease_deg: style.crease_deg,
            group_normals: style.group_normals.then_some(mesh.group_version()),
            colors_key: show_colors.then_some(mesh.color_version()),
            uv_key: uvs.map(|u| u.0),
        };
        (keys, uvs.map(|u| u.1))
    }
}

/// A mesh's soup built away from the UI thread (say a tool's preview result, by the job that
/// made it) with [`SceneRenderer::prepare_soup`], straight into a GPU buffer. Hand it to
/// [`SceneRenderer::sync_mesh_with`] (see [`SyncExtras::prepared`]): when the mesh and style
/// are still those it was built for, it is used as it is instead of building the soup again.
/// It holds a GPU buffer of the whole soup (36 bytes per triangle corner) until then.
pub struct PreparedSoup {
    keys: SoupKeys,
    vertex_buf: wgpu::Buffer,
    capacity_tris: usize,
    tri_ids: Vec<u32>,
}

/// A new value for [`GpuMesh::generation`]: unique across all meshes.
fn next_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ColorVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl ColorVertex {
    pub fn new(p: Vec3, c: Vec4) -> Self {
        Self { position: p.to_array(), color: c.to_array() }
    }
}

/// Display options for one object.
#[derive(Clone, Copy, Debug)]
pub struct ObjectStyle {
    pub color: Vec4,
    pub wireframe: bool,
    pub flat: bool,
    pub group_colors: bool,
    pub dim: bool,
    /// Show the cut-plane preview tint on this object.
    pub cut_tint: bool,
    /// Keep edges sharper than this (degrees) crisp in smooth shading; `None` = smooth.
    pub crease_deg: Option<f32>,
    /// Keep face-group borders crisp too (Group Normals).
    pub group_normals: bool,
    /// Show the mesh's vertex colours (when it has any).
    pub vertex_colors: bool,
    /// Show a checker texture through the UVs (when there are any).
    pub uv_checker: bool,
    /// Show the object's texture through its UVs: 0 no, 1 lit, 2 unlit (its colours as they
    /// are).
    pub texture_mode: u8,
    /// See-through, the parts seen edge-on most opaque (X-ray).
    pub xray: bool,
    pub material: crate::materials::Material,
}

/// Per-triangle display values (indexed by triangle id) shown instead of the object colour.
#[derive(Clone, Copy)]
pub enum TriField<'a> {
    /// Values in `[0, 1]` shown with a diverging colormap.
    Scalar(&'a [f32]),
    /// Colours as `0xRRGGBB`.
    Rgb(&'a [u32]),
}

/// A [`TriField`] with a stamp that must change whenever the values do.
pub type ScalarField<'a> = Option<(u64, TriField<'a>)>;

/// Optional inputs of [`SceneRenderer::sync_mesh_with`]. Build it with
/// `..Default::default()`, so that inputs added later need no changes.
#[derive(Default)]
pub struct SyncExtras<'a> {
    /// Triangles whose UVs alone changed (an island drag), with the UV stamp those changes
    /// were made on top of. When the GPU copy holds exactly that stamp and nothing else
    /// changed, only these triangles are uploaded again. The list is cleared after the sync,
    /// like the moved vertices.
    pub uv_moved: Option<(&'a mut Vec<u32>, u64)>,
    /// The soup of this mesh as drawn with this style, built beforehand: used instead of
    /// building it here when the whole soup must be uploaded and the mesh, its colours and UVs
    /// and the style's normals are still those it was built for (otherwise it is dropped).
    pub prepared: Option<PreparedSoup>,
}

/// GPU copy of one mesh.
pub struct GpuMesh {
    vertex_buf: wgpu::Buffer,
    flags_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity_tris: usize,
    tri_count: u32,
    /// Soup triangle index -> mesh triangle id.
    tri_ids: Vec<u32>,
    shape_version: u64,
    topo_version: u64,
    selection_stamp: u64,
    scalar_stamp: Option<u64>,
    /// Crease angle the uploaded normals were computed with.
    crease_deg: Option<f32>,
    /// Group-normal setting and mesh group version the normals were built with.
    group_normals: Option<u64>,
    /// Mesh colour version the vertex colours were uploaded from (when shown).
    colors_key: Option<u64>,
    uv_key: Option<u64>,
    /// Mesh triangle id -> soup triangle index (u32::MAX if absent).
    soup_index: Vec<u32>,
    /// The object's texture on the GPU, with the image's stamp.
    texture: Option<(u64, wgpu::Texture, wgpu::TextureView)>,
    /// The per-triangle flags as uploaded (soup order), so that a selection change uploads
    /// only the pieces it touches.
    flags: Vec<u32>,
    /// The object uniform as uploaded.
    uniform: Option<ObjectUniform>,
    /// Drawn with the full pipeline (wireframe, colour modes 4-7) rather than the lean one.
    full: bool,
    /// Packed partial uploads land here before being copied run by run into the soup.
    upload: Option<wgpu::Buffer>,
    /// Corner normals of the triangles whose UVs were uploaded again, kept while they hold
    /// (an island drag moves the same triangles every frame and leaves the normals alone).
    uv_normals: Option<(NormalsKey, KnownNormals)>,
    /// Changes with anything this mesh draws (buffers, bind group, uniform), so that an
    /// unchanged frame can be recognised.
    generation: u64,
}

/// An object's texture for [`SceneRenderer::sync_mesh`]: stamp, width, height, RGBA rows.
pub type TextureData<'a> = (u64, u32, u32, &'a [u8]);

pub struct SceneRenderer {
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    object_bgl: wgpu::BindGroupLayout,
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_pipeline_transparent: wgpu::RenderPipeline,
    /// Colour modes 0-3 without the wireframe: no texture, vertex colour, UV or
    /// barycentric varyings (see `shaders/mesh.wgsl`).
    lean_pipeline: wgpu::RenderPipeline,
    lean_pipeline_transparent: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    tri_overlay_pipeline: wgpu::RenderPipeline,
    background_pipeline: wgpu::RenderPipeline,
    lines_buf: Option<(wgpu::Buffer, usize)>,
    overlay_tris_buf: Option<(wgpu::Buffer, usize)>,
    /// Image slices drawn among the meshes ([`Self::set_image_planes`]).
    image_planes: ImagePlanesGpu,
    /// What objects without a texture sample, and how textures are sampled.
    white: wgpu::TextureView,
    sampler: wgpu::Sampler,
    target: Option<Target>,
    /// The inputs of the frame in the target, to skip drawing the same frame again.
    last_frame: Option<FrameKey>,
    /// Frames drawn (not counting those skipped as unchanged).
    #[cfg(test)]
    frames_drawn: u64,
    /// egui's renderer, where the target is registered as a texture (freed on drop, e.g.
    /// when the Mesh workspace restarts after a crash).
    egui_renderer: std::sync::Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
}

impl Drop for SceneRenderer {
    fn drop(&mut self) {
        if let Some(t) = &self.target {
            self.egui_renderer.write().free_texture(&t.texture_id);
        }
    }
}

struct Target {
    size: [u32; 2],
    msaa_view: wgpu::TextureView,
    resolve_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    texture_id: egui::TextureId,
    #[cfg(test)]
    resolve: wgpu::Texture,
}

/// Everything a drawn frame depends on: the target size, the globals, each object's
/// generation (in order, with how it is blended), the overlay geometry and the image
/// planes' stamp.
#[derive(Default)]
struct FrameKey {
    size: [u32; 2],
    globals: Vec<u8>,
    objects: Vec<(u64, bool)>,
    lines: Vec<ColorVertex>,
    overlay_tris: Vec<ColorVertex>,
    planes: u64,
}

impl FrameKey {
    fn matches(&self, size: [u32; 2], frame: &FrameInput, planes: u64) -> bool {
        self.size == size
            && self.planes == planes
            && self.globals == bytemuck::bytes_of(&frame.globals)
            && self.objects.len() == frame.objects.len()
            && self.objects.iter().zip(&frame.objects).all(|(&(g, t), (m, transparent))| g == m.generation && t == *transparent)
            && bytemuck::cast_slice::<_, u8>(&self.lines) == bytemuck::cast_slice::<_, u8>(frame.lines)
            && bytemuck::cast_slice::<_, u8>(&self.overlay_tris) == bytemuck::cast_slice::<_, u8>(frame.overlay_tris)
    }

    fn set(&mut self, size: [u32; 2], frame: &FrameInput, planes: u64) {
        self.size = size;
        self.planes = planes;
        self.globals.clear();
        self.globals.extend_from_slice(bytemuck::bytes_of(&frame.globals));
        self.objects.clear();
        self.objects.extend(frame.objects.iter().map(|(m, t)| (m.generation, *t)));
        self.lines.clear();
        self.lines.extend_from_slice(frame.lines);
        self.overlay_tris.clear();
        self.overlay_tris.extend_from_slice(frame.overlay_tris);
    }
}

/// Everything needed to draw one frame.
pub struct FrameInput<'a> {
    pub globals: Globals,
    pub objects: Vec<(&'a GpuMesh, bool /*transparent*/)>,
    pub lines: &'a [ColorVertex],
    pub overlay_tris: &'a [ColorVertex],
}

impl SceneRenderer {
    pub fn new(rs: &egui_wgpu::RenderState) -> Self {
        let device = &rs.device;
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
        });
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
        });

        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        // Objects without a texture sample a white pixel.
        let white = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("white"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        rs.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &white, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &[255; 4],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white = white.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("object texture"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });

        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh"),
            bind_group_layouts: &[Some(&globals_bgl), Some(&object_bgl)],
            immediate_size: 0,
        });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay"),
            bind_group_layouts: &[Some(&globals_bgl)],
            immediate_size: 0,
        });

        let mesh_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32, 3 => Float32x2],
        };
        // The lean pipeline reads only the position and the normal of the same soup.
        let lean_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };
        let color_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ColorVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
        };
        let multisample = wgpu::MultisampleState { count: SAMPLE_COUNT, mask: !0, alpha_to_coverage_enabled: false };
        let depth = |write: bool, compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        };
        let blend_target = |blend: Option<wgpu::BlendState>| {
            [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend, write_mask: wgpu::ColorWrites::ALL })]
        };

        let make_mesh_pipeline = |transparent: bool, lean: bool| {
            let targets = blend_target(if transparent { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None });
            let label = match (lean, transparent) {
                (false, false) => "mesh",
                (false, true) => "mesh-transparent",
                (true, false) => "mesh-lean",
                (true, true) => "mesh-lean-transparent",
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&mesh_layout),
                vertex: wgpu::VertexState {
                    module: &mesh_shader,
                    entry_point: Some(if lean { "vs_lean" } else { "vs_main" }),
                    compilation_options: Default::default(),
                    buffers: &[Some(if lean { lean_vertex_layout.clone() } else { mesh_vertex_layout.clone() })],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: depth(!transparent, wgpu::CompareFunction::GreaterEqual),
                multisample,
                fragment: Some(wgpu::FragmentState {
                    module: &mesh_shader,
                    entry_point: Some(if lean { "fs_lean" } else { "fs_main" }),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let mesh_pipeline = make_mesh_pipeline(false, false);
        let mesh_pipeline_transparent = make_mesh_pipeline(true, false);
        let lean_pipeline = make_mesh_pipeline(false, true);
        let lean_pipeline_transparent = make_mesh_pipeline(true, true);

        let alpha_targets = blend_target(Some(wgpu::BlendState::ALPHA_BLENDING));
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_lines"),
                compilation_options: Default::default(),
                buffers: &[Some(color_vertex_layout.clone())],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: depth(false, wgpu::CompareFunction::GreaterEqual),
            multisample,
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_color"),
                compilation_options: Default::default(),
                targets: &alpha_targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        let tri_overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay-tris"),
            layout: Some(&overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_tris"),
                compilation_options: Default::default(),
                buffers: &[Some(color_vertex_layout)],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: depth(false, wgpu::CompareFunction::GreaterEqual),
            multisample,
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_color"),
                compilation_options: Default::default(),
                targets: &alpha_targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        let opaque_targets = blend_target(None);
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background"),
            layout: Some(&overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_background"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth(false, wgpu::CompareFunction::Always),
            multisample,
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_background"),
                compilation_options: Default::default(),
                targets: &opaque_targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        let image_planes = ImagePlanesGpu::new(device, &globals_bgl, COLOR_FORMAT, DEPTH_FORMAT, SAMPLE_COUNT);

        Self {
            globals_buf,
            globals_bg,
            object_bgl,
            mesh_pipeline,
            mesh_pipeline_transparent,
            lean_pipeline,
            lean_pipeline_transparent,
            line_pipeline,
            tri_overlay_pipeline,
            background_pipeline,
            lines_buf: None,
            overlay_tris_buf: None,
            image_planes,
            white,
            sampler,
            target: None,
            last_frame: None,
            #[cfg(test)]
            frames_drawn: 0,
            egui_renderer: rs.renderer.clone(),
        }
    }

    /// Image slices to draw among the meshes in the next [`Self::render`] only (set them
    /// before every frame that shows them, like [`FrameInput`]).
    pub fn set_image_planes(&mut self, planes: Option<PlanesFrame>) {
        self.image_planes.set(planes);
    }

    /// Create an empty GPU mesh; call [`Self::sync_mesh`] to fill it.
    pub fn create_mesh(&self, device: &wgpu::Device) -> GpuMesh {
        let (vertex_buf, flags_buf) = Self::alloc_mesh_buffers(device, 1);
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object uniform"),
            size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.make_object_bg(device, &uniform_buf, &flags_buf, &self.white);
        GpuMesh {
            vertex_buf,
            flags_buf,
            uniform_buf,
            bind_group,
            capacity_tris: 1,
            tri_count: 0,
            tri_ids: Vec::new(),
            shape_version: u64::MAX,
            topo_version: u64::MAX,
            selection_stamp: u64::MAX,
            scalar_stamp: None,
            crease_deg: None,
            group_normals: None,
            colors_key: None,
            uv_key: None,
            soup_index: Vec::new(),
            texture: None,
            flags: Vec::new(),
            uniform: None,
            full: false,
            upload: None,
            uv_normals: None,
            generation: next_generation(),
        }
    }

    fn alloc_mesh_buffers(device: &wgpu::Device, tris: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        (Self::alloc_vertex_buffer(device, tris, false), Self::alloc_flags_buffer(device, tris))
    }

    fn alloc_vertex_buffer(device: &wgpu::Device, tris: usize, mapped: bool) -> wgpu::Buffer {
        // COPY_SRC: tests read it back.
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh vertices"),
            size: (tris.max(1) * TRI_BYTES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: mapped,
        })
    }

    fn alloc_flags_buffer(device: &wgpu::Device, tris: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tri flags"),
            size: (tris.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Build the soup [`Self::sync_mesh`] would upload for `mesh` drawn with `style` (and
    /// `uvs`, as passed to it), on any thread: the work of a full upload, written straight
    /// into a new GPU buffer. See [`PreparedSoup`].
    #[allow(dead_code)] // For the jobs of tools with live previews, which are wired up next.
    pub fn prepare_soup(device: &wgpu::Device, mesh: &Mesh, style: &ObjectStyle, uvs: Option<(u64, &[[[f32; 2]; 3]])>) -> PreparedSoup {
        let (keys, uvs) = SoupKeys::of(mesh, style, uvs);
        let mut tri_ids: Vec<u32> = mesh.triangle_ids().collect();
        let max_tris = max_soup_tris(device);
        tri_ids.truncate(max_tris);
        let capacity_tris = ((tri_ids.len() as f64 * 1.25) as usize).min(max_tris).max(1);
        let vertex_buf = Self::alloc_vertex_buffer(device, capacity_tris, true);
        if !tri_ids.is_empty() {
            let normals = ShadingNormals::new(mesh, style.crease_deg.map(f64::from), style.group_normals);
            let colors = keys.colors_key.is_some();
            // A buffer made mapped is mapped whole.
            let mut view = vertex_buf.slice(..(tri_ids.len() * TRI_BYTES) as u64).get_mapped_range_mut().expect("mapped at creation");
            fill_triangles(view.slice(..), &tri_ids, |t| triangle_vertices(mesh, &normals, t, colors, uvs));
        }
        vertex_buf.unmap();
        PreparedSoup { keys, vertex_buf, capacity_tris, tri_ids }
    }

    fn make_object_bg(&self, device: &wgpu::Device, ub: &wgpu::Buffer, fb: &wgpu::Buffer, texture: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object"),
            layout: &self.object_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(texture) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        })
    }

    /// Upload `texture` for `gpu` when it changed (or drop it), rebuilding its bind group.
    fn sync_texture(&self, device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuMesh, texture: Option<TextureData>) {
        if gpu.texture.as_ref().map(|t| t.0) == texture.map(|t| t.0) {
            return;
        }
        gpu.texture = texture.filter(|t| t.1 > 0 && t.2 > 0 && t.3.len() >= (t.1 * t.2 * 4) as usize).map(|(stamp, w, h, rgba)| {
            let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("object texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &rgba[..(w * h * 4) as usize],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                size,
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (stamp, tex, view)
        });
        let view = gpu.texture.as_ref().map_or(&self.white, |t| &t.2);
        gpu.bind_group = self.make_object_bg(device, &gpu.uniform_buf, &gpu.flags_buf, view);
        gpu.generation = next_generation();
    }

    /// Upload mesh geometry and per-triangle flags if they changed.
    /// `selection_stamp` must change whenever the selection or face groups change.
    #[allow(clippy::too_many_arguments)]
    pub fn sync_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu: &mut GpuMesh,
        mesh: &Mesh,
        selection: Option<&TriSelection>,
        selection_stamp: u64,
        style: &ObjectStyle,
        moved: (&mut Vec<u32>, u64),
        scalars: ScalarField,
        uvs: Option<(u64, &[[[f32; 2]; 3]])>,
        texture: Option<TextureData>,
    ) {
        self.sync_mesh_with(device, queue, gpu, mesh, selection, selection_stamp, style, moved, scalars, uvs, texture, SyncExtras::default());
    }

    /// [`Self::sync_mesh`] with the optional inputs of [`SyncExtras`].
    #[allow(clippy::too_many_arguments)]
    pub fn sync_mesh_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu: &mut GpuMesh,
        mesh: &Mesh,
        selection: Option<&TriSelection>,
        selection_stamp: u64,
        style: &ObjectStyle,
        moved: (&mut Vec<u32>, u64),
        scalars: ScalarField,
        uvs: Option<(u64, &[[[f32; 2]; 3]])>,
        texture: Option<TextureData>,
        extras: SyncExtras,
    ) {
        let texture = texture.filter(|_| style.texture_mode != 0 && uvs.is_some());
        self.sync_texture(device, queue, gpu, texture);
        let (moved_vertices, moved_from) = moved;
        let (keys, uv_slice) = SoupKeys::of(mesh, style, uvs);
        let (groups_key, colors_key, uv_key) = (keys.group_normals, keys.colors_key, keys.uv_key);
        let show_colors = colors_key.is_some();
        let geometry_changed = gpu.shape_version != mesh.shape_version()
            || gpu.crease_deg != style.crease_deg
            || gpu.group_normals != groups_key
            || gpu.colors_key != colors_key
            || gpu.uv_key != uv_key;
        let crease = style.crease_deg.map(f64::from);
        let groups = style.group_normals;
        // Partial uploads are only valid on top of exactly the geometry the moves started from.
        // Colours may differ (vertex painting reports its vertices as moved).
        let partial = geometry_changed
            && gpu.crease_deg == style.crease_deg
            && gpu.group_normals == groups_key
            && gpu.colors_key.is_some() == colors_key.is_some()
            && gpu.uv_key == uv_key
            && gpu.topo_version == mesh.topology_version()
            && gpu.shape_version == moved_from
            && !moved_vertices.is_empty();
        // Likewise for UVs alone: everything else as uploaded, the UVs those changes started from.
        let (uv_moved, uv_moved_from) = match extras.uv_moved {
            Some((tris, from)) => (Some(tris), Some(from)),
            None => (None, None),
        };
        let uv_partial = !partial
            && geometry_changed
            && gpu.shape_version == mesh.shape_version()
            && gpu.topo_version == mesh.topology_version()
            && gpu.crease_deg == style.crease_deg
            && gpu.group_normals == groups_key
            && gpu.colors_key == colors_key
            && uv_key.is_some()
            && gpu.uv_key.is_some()
            && gpu.uv_key == uv_moved_from
            && uv_moved.as_ref().is_some_and(|t| !t.is_empty());
        if partial {
            self.partial_update(device, queue, gpu, mesh, moved_vertices, crease, groups, show_colors, uv_slice);
            gpu.shape_version = mesh.shape_version();
            gpu.colors_key = colors_key;
        } else if uv_partial {
            if let Some(tris) = uv_moved.as_deref() {
                self.uv_update(device, queue, gpu, mesh, tris, crease, groups, show_colors, uv_slice);
            }
            gpu.uv_key = uv_key;
        } else if geometry_changed {
            match extras.prepared.filter(|p| p.keys == keys) {
                Some(p) => self.install_soup(device, gpu, mesh, p),
                None => self.full_update(device, queue, gpu, mesh, crease, groups, show_colors, uv_slice),
            }
            gpu.shape_version = mesh.shape_version();
            gpu.topo_version = mesh.topology_version();
            gpu.crease_deg = style.crease_deg;
            gpu.group_normals = groups_key;
            gpu.colors_key = colors_key;
            gpu.uv_key = uv_key;
        }
        moved_vertices.clear();
        if let Some(tris) = uv_moved {
            tris.clear();
        }
        let scalar_stamp = scalars.map(|(s, _)| s);
        let rebuilt = geometry_changed && !partial && !uv_partial;
        if rebuilt || gpu.selection_stamp != selection_stamp || gpu.scalar_stamp != scalar_stamp {
            Self::sync_flags(queue, gpu, mesh, selection, scalars, rebuilt);
            gpu.selection_stamp = selection_stamp;
            gpu.scalar_stamp = scalar_stamp;
        }
        let u = ObjectUniform {
            color: style.color.to_array(),
            flags: [
                style.wireframe as u32,
                style.flat as u32,
                match scalars {
                    Some((_, TriField::Scalar(_))) => 2,
                    Some((_, TriField::Rgb(_))) => 3,
                    None if style.group_colors => 1,
                    None if uv_key.is_some() && gpu.texture.is_some() => 5 + style.texture_mode.min(2) as u32,
                    None if uv_key.is_some() && style.uv_checker => 5,
                    None if show_colors => 4,
                    None => 0,
                },
                style.dim as u32 | ((style.cut_tint as u32) << 1) | ((style.xray as u32) << 2),
            ],
            material: style.material.uniform().0,
            material2: style.material.uniform().1,
            model: IDENTITY,
            normal_matrix: IDENTITY3,
        };
        Self::write_uniform(queue, gpu, u);
    }

    /// Show `gpu`, an object's own GPU copy as [`Self::sync_mesh`] left it, as the live
    /// preview of that object moved rigidly by `model` (rotation, uniform scale, mirroring,
    /// translation), drawn the way a preview mesh is: no selection highlight, no texture, UV
    /// checker or per-triangle field, group or vertex colours as `style` asks. Draw it
    /// opaque. The next [`Self::sync_mesh`] shows the object as it is again. Returns false
    /// when this copy cannot show the preview (not a similarity transform, never synced,
    /// or holding a per-triangle field where group colours are asked for): upload the moved
    /// mesh instead.
    #[allow(dead_code)] // For the Transform tool's rigid previews, which are wired up next.
    pub fn sync_rigid_preview(&self, queue: &wgpu::Queue, gpu: &mut GpuMesh, style: &ObjectStyle, model: Mat4) -> bool {
        let m = Mat3::from_mat4(model);
        let scale = m.x_axis.length();
        let similar = scale > 0.0
            && [m.y_axis, m.z_axis].iter().all(|a| (a.length() - scale).abs() <= 1e-4 * scale)
            && [m.x_axis.dot(m.y_axis), m.y_axis.dot(m.z_axis), m.z_axis.dot(m.x_axis)].iter().all(|d| d.abs() <= 1e-4 * scale * scale);
        let groups_shown = style.group_colors && gpu.scalar_stamp.is_none();
        if !similar || !model.is_finite() || gpu.shape_version == u64::MAX || (style.group_colors && !groups_shown) {
            return false;
        }
        let mode = if groups_shown {
            1
        } else if style.vertex_colors && gpu.colors_key.is_some() {
            4
        } else {
            0
        };
        let normal = m.inverse().transpose();
        let mirrored = m.determinant() < 0.0;
        let u = ObjectUniform {
            color: style.color.to_array(),
            flags: [
                style.wireframe as u32,
                style.flat as u32,
                mode,
                style.dim as u32 | ((style.cut_tint as u32) << 1) | ((style.xray as u32) << 2) | MOVED | NO_SELECTION | if mirrored { MIRRORED } else { 0 },
            ],
            material: style.material.uniform().0,
            material2: style.material.uniform().1,
            model: model.to_cols_array_2d(),
            normal_matrix: [normal.x_axis.extend(0.0).to_array(), normal.y_axis.extend(0.0).to_array(), normal.z_axis.extend(0.0).to_array()],
        };
        Self::write_uniform(queue, gpu, u);
        true
    }

    /// Upload the object uniform unless it is unchanged.
    fn write_uniform(queue: &wgpu::Queue, gpu: &mut GpuMesh, u: ObjectUniform) {
        gpu.full = u.needs_full_pipeline();
        if gpu.uniform.as_ref().is_some_and(|old| bytemuck::bytes_of(old) == bytemuck::bytes_of(&u)) {
            return;
        }
        queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::bytes_of(&u));
        gpu.uniform = Some(u);
        gpu.generation = next_generation();
    }

    /// Rebuild the per-triangle flags (selection bit, and group, scalar or colour payload)
    /// and upload them: all of them after `rebuilt` geometry (new soup order, maybe a new
    /// buffer), otherwise only the pieces that changed.
    fn sync_flags(queue: &wgpu::Queue, gpu: &mut GpuMesh, mesh: &Mesh, selection: Option<&TriSelection>, scalars: ScalarField, rebuilt: bool) {
        let flag = |t: u32| {
            let sel = selection.map(|s| s.contains(t)).unwrap_or(false) as u32;
            let payload = match scalars {
                Some((_, TriField::Scalar(vals))) => (vals.get(t as usize).copied().unwrap_or(0.5).clamp(0.0, 1.0) * 65535.0) as u32,
                Some((_, TriField::Rgb(cols))) => cols.get(t as usize).copied().unwrap_or(0x808080) & 0x00ff_ffff,
                None => (mesh.tri_group(t) as u32) & 0x00ff_ffff,
            };
            sel | (payload << 8)
        };
        let everything = rebuilt || gpu.flags.len() != gpu.tri_ids.len();
        if everything {
            gpu.flags.clear();
            gpu.flags.resize(gpu.tri_ids.len(), 0);
        }
        // New values in place, noting which pieces differ from what the GPU holds.
        let changed: Vec<bool> = gpu
            .flags
            .par_chunks_mut(FLAG_CHUNK)
            .zip(gpu.tri_ids.par_chunks(FLAG_CHUNK))
            .map(|(old, tris)| {
                let mut differs = false;
                for (f, &t) in old.iter_mut().zip(tris) {
                    let new = flag(t);
                    differs |= *f != new;
                    *f = new;
                }
                differs
            })
            .collect();
        let mut wrote = false;
        let mut k = 0;
        while k < changed.len() {
            if !everything && !changed[k] {
                k += 1;
                continue;
            }
            let start = k;
            while k < changed.len() && (everything || changed[k]) {
                k += 1;
            }
            let range = start * FLAG_CHUNK..(k * FLAG_CHUNK).min(gpu.flags.len());
            queue.write_buffer(&gpu.flags_buf, (range.start * 4) as u64, bytemuck::cast_slice(&gpu.flags[range]));
            wrote = true;
        }
        if wrote {
            gpu.generation = next_generation();
        }
    }

    /// Upload the whole soup (and the triangle order it follows).
    #[allow(clippy::too_many_arguments)]
    fn full_update(&self, device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuMesh, mesh: &Mesh, crease: Option<f64>, groups: bool, colors: bool, uvs: Option<&[[[f32; 2]; 3]]>) {
        let mut tri_ids = std::mem::take(&mut gpu.tri_ids);
        tri_ids.clear();
        tri_ids.extend(mesh.triangle_ids());
        // Never exceed the device's buffer limits (very large meshes are truncated with a
        // warning rather than crashing).
        let max_tris = max_soup_tris(device);
        if tri_ids.len() > max_tris {
            log::warn!("mesh has {} triangles; only {} fit in GPU buffers", tri_ids.len(), max_tris);
            tri_ids.truncate(max_tris);
        }
        if tri_ids.len() > gpu.capacity_tris {
            let cap = ((tri_ids.len() as f64 * 1.25) as usize).min(max_tris);
            let (vb, fb) = Self::alloc_mesh_buffers(device, cap);
            gpu.vertex_buf = vb;
            gpu.flags_buf = fb;
            gpu.capacity_tris = cap;
            let view = gpu.texture.as_ref().map_or(&self.white, |t| &t.2);
            gpu.bind_group = self.make_object_bg(device, &gpu.uniform_buf, &gpu.flags_buf, view);
            // The new flags buffer holds nothing yet.
            gpu.flags.clear();
        }
        let normals = ShadingNormals::new(mesh, crease, groups);
        write_triangles(queue, &gpu.vertex_buf, 0, &tri_ids, |t| triangle_vertices(mesh, &normals, t, colors, uvs));
        gpu.uv_normals = None;
        gpu.tri_count = tri_ids.len() as u32;
        gpu.soup_index.clear();
        gpu.soup_index.resize(mesh.max_tid(), u32::MAX);
        for (i, &t) in tri_ids.iter().enumerate() {
            gpu.soup_index[t as usize] = i as u32;
        }
        gpu.tri_ids = tri_ids;
        gpu.generation = next_generation();
    }

    /// Take `p`'s soup as the GPU copy of `mesh` (whose keys it was built for).
    fn install_soup(&self, device: &wgpu::Device, gpu: &mut GpuMesh, mesh: &Mesh, p: PreparedSoup) {
        let n = p.tri_ids.len();
        gpu.vertex_buf = p.vertex_buf;
        if n > gpu.capacity_tris {
            gpu.flags_buf = Self::alloc_flags_buffer(device, p.capacity_tris);
            let view = gpu.texture.as_ref().map_or(&self.white, |t| &t.2);
            gpu.bind_group = self.make_object_bg(device, &gpu.uniform_buf, &gpu.flags_buf, view);
            gpu.capacity_tris = p.capacity_tris;
        } else {
            // Room for this many in both buffers (the next full upload reallocates beyond it).
            gpu.capacity_tris = gpu.capacity_tris.min(p.capacity_tris);
        }
        // All flags are uploaded again, into a buffer that may be new.
        gpu.flags.clear();
        gpu.tri_count = n as u32;
        gpu.soup_index.clear();
        gpu.soup_index.resize(mesh.max_tid(), u32::MAX);
        for (i, &t) in p.tri_ids.iter().enumerate() {
            gpu.soup_index[t as usize] = i as u32;
        }
        gpu.tri_ids = p.tri_ids;
        gpu.uv_normals = None;
        gpu.generation = next_generation();
    }

    /// Re-upload only the triangles around moved vertices (positions and normals).
    #[allow(clippy::too_many_arguments)]
    fn partial_update(&self, device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuMesh, mesh: &Mesh, moved: &[u32], crease: Option<f64>, groups: bool, colors: bool, uvs: Option<&[[[f32; 2]; 3]]>) {
        // Strokes report vertices once per dab: each counts once.
        let moved = unique(moved.iter().copied().filter(|&v| mesh.is_vertex(v)), mesh.max_vid());
        // Normals change on the one-ring of moved vertices, so refresh their triangles too.
        let verts = unique(moved.iter().flat_map(|&v| std::iter::once(v).chain(mesh.vertex_neighbors(v))), mesh.max_vid());
        let tris = unique(verts.iter().flat_map(|&v| mesh.vertex_triangles(v)), mesh.max_tid());
        let normals = ShadingNormals::for_triangles(mesh, &tris, crease, groups);
        self.upload_triangles(device, queue, gpu, &tris, |t| triangle_vertices(mesh, &normals, t, colors, uvs));
        gpu.uv_normals = None;
    }

    /// Re-upload the triangles `tris` whose UVs alone changed.
    #[allow(clippy::too_many_arguments)]
    fn uv_update(&self, device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuMesh, mesh: &Mesh, tris: &[u32], crease: Option<f64>, groups: bool, colors: bool, uvs: Option<&[[[f32; 2]; 3]]>) {
        let tris = unique(tris.iter().copied().filter(|&t| mesh.is_triangle(t)), mesh.max_tid());
        // The normals of these triangles are those uploaded with them: computed once while
        // the shape and the normals' settings stay.
        let key = (gpu.shape_version, gpu.topo_version, gpu.crease_deg.map(f32::to_bits), gpu.group_normals);
        let mut known = match gpu.uv_normals.take() {
            Some((k, known)) if k == key => known,
            _ => KnownNormals::default(),
        };
        let missing: Vec<u32> = tris.iter().copied().filter(|t| !known.contains_key(t)).collect();
        if !missing.is_empty() {
            let normals = ShadingNormals::for_triangles(mesh, &missing, crease, groups);
            let corners = |&t: &u32| (t, corner_normals(&normals, t));
            if missing.len() < PARALLEL_MIN {
                known.extend(missing.iter().map(corners));
            } else {
                known.extend(missing.par_iter().map(corners).collect::<Vec<_>>());
            }
        }
        self.upload_triangles(device, queue, gpu, &tris, |t| soup_triangle(mesh, t, colors, uvs, known[&t]));
        if mesh.triangle_count() <= UV_NORMALS_MAX_TRIS {
            gpu.uv_normals = Some((key, known));
        }
    }

    /// Upload the soup triangles of mesh triangles `tris` (built by `build`) to their slots:
    /// written packed in one go, then copied run by run, submitted at once.
    fn upload_triangles(&self, device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuMesh, tris: &[u32], build: impl Fn(u32) -> [MeshVertex; 3] + Sync) {
        let mut slots: Vec<(u32, u32)> = tris.iter().filter_map(|&t| gpu.soup_index.get(t as usize).copied().filter(|&i| i != u32::MAX).map(|i| (i, t))).collect();
        // Triangles come once each, so do their slots.
        if slots.len() < PARALLEL_MIN {
            slots.sort_unstable();
        } else {
            slots.par_sort_unstable();
        }
        if slots.is_empty() {
            return;
        }
        // Contiguous runs of soup slots: (first slot, index in `slots`, length).
        let mut runs: Vec<(u32, usize, usize)> = Vec::new();
        for (k, &(slot, _)) in slots.iter().enumerate() {
            match runs.last_mut() {
                Some((start, _, len)) if *start + *len as u32 == slot => *len += 1,
                _ => runs.push((slot, k, 1)),
            }
        }
        let order: Vec<u32> = slots.iter().map(|s| s.1).collect();
        let stride = TRI_BYTES as u64;
        if let [(start, _, _)] = runs[..] {
            write_triangles(queue, &gpu.vertex_buf, start as u64 * stride, &order, build);
        } else {
            let bytes = (order.len() * TRI_BYTES) as u64;
            let mut temporary = None;
            if gpu.upload.as_ref().is_none_or(|b| b.size() < bytes) {
                let size = bytes.next_power_of_two().max(64 << 10);
                let b = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mesh upload"),
                    size,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if size <= MAX_KEPT_UPLOAD {
                    gpu.upload = Some(b);
                } else {
                    temporary = Some(b);
                }
            }
            let Some(upload) = temporary.as_ref().or(gpu.upload.as_ref()) else { return };
            write_triangles(queue, upload, 0, &order, build);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mesh upload") });
            for &(slot, k, len) in &runs {
                encoder.copy_buffer_to_buffer(upload, k as u64 * stride, &gpu.vertex_buf, slot as u64 * stride, len as u64 * stride);
            }
            queue.submit(Some(encoder.finish()));
        }
        gpu.generation = next_generation();
    }

    /// The target for a frame of `size`, and whether it was just made (it holds no frame).
    fn ensure_target(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2]) -> (egui::TextureId, bool) {
        let size = [size[0].max(1), size[1].max(1)];
        if let Some(t) = &self.target {
            if t.size == size {
                return (t.texture_id, false);
            }
        }
        let device = &rs.device;
        let mk = |label, format, samples, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        // The multisampled colour and depth only live within the frame's one render pass
        // (cleared on load, discarded on store), so they can stay in tile memory. They must
        // keep that single-pass Clear/Discard use; only the resolve target is kept.
        let transient = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT_ATTACHMENT;
        let msaa = mk("scene msaa", COLOR_FORMAT, SAMPLE_COUNT, transient);
        let resolve = mk(
            "scene resolve",
            COLOR_FORMAT,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = mk("scene depth", DEPTH_FORMAT, SAMPLE_COUNT, transient);
        let resolve_view = resolve.create_view(&Default::default());
        let mut renderer = rs.renderer.write();
        let texture_id = if let Some(t) = &self.target {
            renderer.update_egui_texture_from_wgpu_texture(device, &resolve_view, wgpu::FilterMode::Linear, t.texture_id);
            t.texture_id
        } else {
            renderer.register_native_texture(device, &resolve_view, wgpu::FilterMode::Linear)
        };
        self.target = Some(Target {
            size,
            msaa_view: msaa.create_view(&Default::default()),
            resolve_view,
            depth_view: depth.create_view(&Default::default()),
            texture_id,
            #[cfg(test)]
            resolve,
        });
        (texture_id, true)
    }

    fn upload_dynamic(device: &wgpu::Device, queue: &wgpu::Queue, slot: &mut Option<(wgpu::Buffer, usize)>, data: &[ColorVertex]) {
        if data.is_empty() {
            return;
        }
        let need = data.len();
        let realloc = match slot {
            Some((_, cap)) => *cap < need,
            None => true,
        };
        if realloc {
            let cap = need.next_power_of_two().max(64);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay"),
                size: (cap * std::mem::size_of::<ColorVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *slot = Some((buf, cap));
        }
        queue.write_buffer(&slot.as_ref().unwrap().0, 0, bytemuck::cast_slice(data));
    }

    /// Render the frame into the offscreen target and return its egui texture id. A frame
    /// with the same inputs as the one already in the target (same size, globals, objects
    /// with unchanged GPU copies in the same order, overlays and image planes) is not drawn
    /// again: the target still shows it.
    pub fn render(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2], frame: &FrameInput) -> egui::TextureId {
        let (tex, fresh) = self.ensure_target(rs, size);
        let device = &rs.device;
        let queue = &rs.queue;
        self.image_planes.prepare(device, queue, &frame.globals);
        let planes = self.image_planes.stamp();
        if !fresh && self.last_frame.as_ref().is_some_and(|k| k.matches(size, frame, planes)) {
            return tex;
        }
        self.last_frame.get_or_insert_with(FrameKey::default).set(size, frame, planes);
        #[cfg(test)]
        {
            self.frames_drawn += 1;
        }
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&frame.globals));
        Self::upload_dynamic(device, queue, &mut self.lines_buf, frame.lines);
        Self::upload_dynamic(device, queue, &mut self.overlay_tris_buf, frame.overlay_tris);

        let target = self.target.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scene") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&target.resolve_view),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_pipeline(&self.background_pipeline);
            pass.draw(0..3, 0..1);

            // Objects in the order given (which wins ties in depth), switching between the
            // lean and the full pipeline only where it changes.
            let mut bound = None;
            for (m, transparent) in &frame.objects {
                if *transparent || m.tri_count == 0 {
                    continue;
                }
                if bound != Some(m.full) {
                    pass.set_pipeline(if m.full { &self.mesh_pipeline } else { &self.lean_pipeline });
                    bound = Some(m.full);
                }
                pass.set_bind_group(1, &m.bind_group, &[]);
                pass.set_vertex_buffer(0, m.vertex_buf.slice(..));
                pass.draw(0..m.tri_count * 3, 0..1);
            }
            self.image_planes.draw(&mut pass, false);
            if let Some((buf, _)) = &self.lines_buf {
                if !frame.lines.is_empty() {
                    pass.set_pipeline(&self.line_pipeline);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..frame.lines.len() as u32, 0..1);
                }
            }
            self.image_planes.draw(&mut pass, true);
            let mut bound = None;
            for (m, transparent) in &frame.objects {
                if !*transparent || m.tri_count == 0 {
                    continue;
                }
                if bound != Some(m.full) {
                    pass.set_pipeline(if m.full { &self.mesh_pipeline_transparent } else { &self.lean_pipeline_transparent });
                    bound = Some(m.full);
                }
                pass.set_bind_group(1, &m.bind_group, &[]);
                pass.set_vertex_buffer(0, m.vertex_buf.slice(..));
                pass.draw(0..m.tri_count * 3, 0..1);
            }
            if let Some((buf, _)) = &self.overlay_tris_buf {
                if !frame.overlay_tris.is_empty() {
                    pass.set_pipeline(&self.tri_overlay_pipeline);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..frame.overlay_tris.len() as u32, 0..1);
                }
            }
        }
        queue.submit(Some(encoder.finish()));
        tex
    }
}

pub fn globals(view_proj: Mat4, view: Mat4, eye: Vec3, brush: Option<(Vec3, f32)>, viewport_h: f32) -> Globals {
    let brush = brush.map(|(c, r)| [c.x, c.y, c.z, r]).unwrap_or([0.0; 4]);
    Globals {
        view_proj: view_proj.to_cols_array_2d(),
        view: view.to_cols_array_2d(),
        eye: [eye.x, eye.y, eye.z, 1.0],
        brush,
        misc: [0.0, viewport_h, 0.0, 0.0],
        cut_plane: [0.0; 4],
        cut_region: [0.0; 4],
        background: [DEFAULT_BACKGROUND[0], DEFAULT_BACKGROUND[1], DEFAULT_BACKGROUND[2], 1.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_core::primitives;

    /// wgpu's native futures are ready when first polled.
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

    /// A headless device with an egui renderer, as eframe would hand over (`None` without a
    /// GPU, and the GPU tests pass vacuously).
    fn render_state() -> Option<egui_wgpu::RenderState> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        let renderer = egui_wgpu::Renderer::new(&device, COLOR_FORMAT, egui_wgpu::RendererOptions::default());
        Some(egui_wgpu::RenderState {
            adapter,
            available_adapters: Vec::new(),
            instance,
            device,
            queue,
            target_format: COLOR_FORMAT,
            renderer: std::sync::Arc::new(egui::mutex::RwLock::new(renderer)),
        })
    }

    fn read_buffer(rs: &egui_wgpu::RenderState, buf: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let staging = rs.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: len.max(4), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut encoder = rs.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, len);
        rs.queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        rs.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = staging.slice(..len).get_mapped_range().unwrap().to_vec();
        bytes
    }

    /// The soup and the flags as the GPU holds them.
    fn gpu_contents(rs: &egui_wgpu::RenderState, gpu: &GpuMesh) -> (Vec<u8>, Vec<u8>) {
        let n = gpu.tri_count as u64;
        (read_buffer(rs, &gpu.vertex_buf, n * TRI_BYTES as u64), read_buffer(rs, &gpu.flags_buf, n * 4))
    }

    /// The last frame's pixels, row by row.
    fn read_target(rs: &egui_wgpu::RenderState, r: &SceneRenderer) -> Vec<u8> {
        let t = r.target.as_ref().unwrap();
        let [w, h] = t.size;
        let row = (w * 4).div_ceil(256) * 256;
        let staging = rs.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut encoder = rs.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &t.resolve, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        rs.queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        rs.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = staging.slice(..).get_mapped_range().unwrap();
        (0..h as usize).flat_map(|y| data[y * row as usize..y * row as usize + w as usize * 4].to_vec()).collect()
    }

    fn style() -> ObjectStyle {
        ObjectStyle {
            color: Vec4::new(0.8, 0.7, 0.6, 1.0),
            wireframe: false,
            flat: false,
            group_colors: false,
            dim: false,
            cut_tint: false,
            crease_deg: Some(40.0),
            group_normals: true,
            vertex_colors: false,
            uv_checker: false,
            texture_mode: 0,
            xray: false,
            material: crate::materials::MATERIALS[0],
        }
    }

    /// `m`'s triangles in a scrambled order, so that neighbouring triangles land far apart
    /// in the soup (as after remeshing) and partial uploads have many runs.
    fn scrambled(m: &Mesh) -> Mesh {
        let (pos, mut tris, mut groups) = m.to_arrays();
        let n = tris.len();
        for i in (1..n).rev() {
            let j = (i * 7919 + 13) % (i + 1);
            tris.swap(i, j);
            groups.swap(i, j);
        }
        Mesh::from_triangles(&pos, &tris, Some(&groups)).0
    }

    /// Per-corner UVs from the corner positions (and a shift).
    fn uvs_of(m: &Mesh, shift: f32) -> Vec<[[f32; 2]; 3]> {
        (0..m.max_tid() as u32)
            .map(|t| if m.is_triangle(t) { m.tri_positions(t).map(|p| [p.x as f32 * 0.5 + shift, p.y as f32 * 0.5]) } else { [[0.0; 2]; 3] })
            .collect()
    }

    /// A mesh with groups, vertex colours and every kind of corner (sharp and smooth).
    fn test_mesh() -> Mesh {
        let mut m = scrambled(&primitives::cylinder(1.0, 2.0, 40, 12, true));
        m.enable_vertex_colors();
        for t in m.triangle_ids().collect::<Vec<_>>() {
            let c = m.tri_centroid(t);
            m.set_tri_group(t, (c.x > 0.2) as i32 + 2 * (c.z > 1.5) as i32);
        }
        for v in m.vertex_ids().collect::<Vec<_>>() {
            let p = m.vertex(v);
            m.set_vertex_color(v, [p.x.abs() as f32, p.y.abs() as f32, (p.z * 0.5) as f32]);
        }
        m
    }

    #[test]
    fn shader_compiles() {
        use eframe::wgpu::naga;
        let module = naga::front::wgsl::parse_str(include_str!("shaders/mesh.wgsl")).unwrap();
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap();
        for entry in ["vs_main", "fs_main", "vs_lean", "fs_lean"] {
            assert!(module.entry_points.iter().any(|e| e.name == entry), "{entry}");
        }
        // The uniform as the shader lays it out.
        assert_eq!(std::mem::size_of::<ObjectUniform>(), 176);
    }

    #[test]
    fn partial_updates_match_a_full_rebuild() {
        let Some(rs) = render_state() else { return };
        let r = SceneRenderer::new(&rs);
        let mut m = test_mesh();
        let uvs = uvs_of(&m, 0.0);
        let s = ObjectStyle { vertex_colors: true, uv_checker: true, ..style() };
        let mut gpu = r.create_mesh(&rs.device);
        let mut moved = Vec::new();
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, None, 0, &s, (&mut moved, 0), None, Some((1, &uvs)), None);
        let slots = gpu.tri_ids.clone();
        for stroke in 0..6u32 {
            let from = m.shape_version();
            // Overlapping dabs report vertices more than once; a vertex painted too.
            let centre = DVec3::new((stroke as f64).cos(), (stroke as f64).sin(), 0.3 * stroke as f64);
            for dab in 0..3 {
                for v in m.vertex_ids().collect::<Vec<_>>() {
                    let p = m.vertex(v);
                    if p.distance(centre) < 0.5 + 0.1 * dab as f64 {
                        m.set_vertex(v, p + (centre - p) * 0.05);
                        if dab == 1 {
                            m.set_vertex_color(v, [0.2, 0.9, 0.4]);
                        }
                        moved.push(v);
                    }
                }
            }
            assert!(!moved.is_empty());
            r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, None, 0, &s, (&mut moved, from), None, Some((1, &uvs)), None);
            assert!(moved.is_empty());
            assert_eq!(gpu.tri_ids, slots, "a partial upload keeps the soup order");
        }
        assert!(gpu.upload.is_some(), "scattered triangles were uploaded packed");
        let mut fresh = r.create_mesh(&rs.device);
        r.sync_mesh(&rs.device, &rs.queue, &mut fresh, &m, None, 0, &s, (&mut moved, 0), None, Some((1, &uvs)), None);
        let (a, b) = (gpu_contents(&rs, &gpu), gpu_contents(&rs, &fresh));
        assert!(a.0 == b.0, "soup after partial uploads differs from a full rebuild");
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn uv_updates_match_a_full_rebuild() {
        let Some(rs) = render_state() else { return };
        let r = SceneRenderer::new(&rs);
        let m = test_mesh();
        let mut uvs = uvs_of(&m, 0.0);
        let s = ObjectStyle { uv_checker: true, ..style() };
        let mut gpu = r.create_mesh(&rs.device);
        let mut none = Vec::new();
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, None, 0, &s, (&mut none, 0), None, Some((1, &uvs)), None);
        // An island moves twice: only its triangles are written again.
        let island: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).x > 0.5).collect();
        for (stamp, shift) in [(2, 0.25), (3, 0.5)] {
            for &t in &island {
                for c in &mut uvs[t as usize] {
                    c[0] += shift;
                }
            }
            let generation = gpu.generation;
            let mut tris = island.clone();
            r.sync_mesh_with(&rs.device, &rs.queue, &mut gpu, &m, None, 0, &s, (&mut none, 0), None, Some((stamp, &uvs)), None, SyncExtras { uv_moved: Some((&mut tris, stamp - 1)), ..Default::default() });
            assert!(tris.is_empty());
            assert_eq!(gpu.uv_key, Some(stamp));
            assert_ne!(gpu.generation, generation);
            // The island's normals are kept for its next move.
            assert_eq!(gpu.uv_normals.as_ref().map(|(_, n)| n.len()), Some(island.len()));
        }
        let mut fresh = r.create_mesh(&rs.device);
        r.sync_mesh(&rs.device, &rs.queue, &mut fresh, &m, None, 0, &s, (&mut none, 0), None, Some((3, &uvs)), None);
        assert!(gpu_contents(&rs, &gpu) == gpu_contents(&rs, &fresh), "UV updates differ from a full rebuild");
        // Changes on top of UVs the GPU does not hold take the full path, with the same result.
        for c in &mut uvs[island[0] as usize] {
            c[1] += 1.0;
        }
        let mut tris = vec![island[0]];
        r.sync_mesh_with(&rs.device, &rs.queue, &mut gpu, &m, None, 0, &s, (&mut none, 0), None, Some((5, &uvs)), None, SyncExtras { uv_moved: Some((&mut tris, 4)), ..Default::default() });
        r.sync_mesh(&rs.device, &rs.queue, &mut fresh, &m, None, 0, &s, (&mut none, 0), None, Some((5, &uvs)), None);
        assert!(gpu_contents(&rs, &gpu) == gpu_contents(&rs, &fresh));
        assert!(gpu.uv_normals.is_none(), "a full upload drops the kept normals");
        // A stroke moves vertices, then the island moves again: its normals are new.
        let mut moved_mesh = m.clone();
        let from = moved_mesh.shape_version();
        let v = moved_mesh.triangle(island[0])[0];
        moved_mesh.set_vertex(v, moved_mesh.vertex(v) * 1.1);
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &moved_mesh, None, 0, &s, (&mut vec![v], from), None, Some((5, &uvs)), None);
        for &t in &island {
            for c in &mut uvs[t as usize] {
                c[0] -= 0.125;
            }
        }
        let mut tris = island.clone();
        r.sync_mesh_with(&rs.device, &rs.queue, &mut gpu, &moved_mesh, None, 0, &s, (&mut none, 0), None, Some((6, &uvs)), None, SyncExtras { uv_moved: Some((&mut tris, 5)), ..Default::default() });
        assert_eq!(gpu.uv_key, Some(6));
        r.sync_mesh(&rs.device, &rs.queue, &mut fresh, &moved_mesh, None, 0, &s, (&mut none, 0), None, Some((6, &uvs)), None);
        assert!(gpu_contents(&rs, &gpu) == gpu_contents(&rs, &fresh), "UV update after a stroke");
    }

    #[test]
    fn flags_follow_selections_growth_and_undo() {
        let Some(rs) = render_state() else { return };
        let r = SceneRenderer::new(&rs);
        let s = ObjectStyle { group_colors: true, ..style() };
        let mut m = primitives::icosphere(1.0, 5);
        let before = m.clone();
        let mut gpu = r.create_mesh(&rs.device);
        let mut none = Vec::new();
        let check = |gpu: &GpuMesh, m: &Mesh, sel: &TriSelection, scalars: ScalarField, label: &str| {
            let mut fresh = r.create_mesh(&rs.device);
            r.sync_mesh(&rs.device, &rs.queue, &mut fresh, m, Some(sel), 0, &s, (&mut Vec::new(), 0), scalars, None, None);
            assert!(gpu_contents(&rs, gpu).1 == gpu_contents(&rs, &fresh).1, "{label}: flags differ");
            assert_eq!(gpu.flags, fresh.flags, "{label}");
        };
        let mut sel = TriSelection::new(&m);
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 1, &s, (&mut none, 0), None, None, None);
        assert!(gpu.tri_count as usize > 3 * FLAG_CHUNK, "several flag pieces");
        // A brush stroke selecting a few triangles in one piece, then in another.
        for (stamp, range) in [(2, 100..140), (3, 9000..9010)] {
            for t in range {
                sel.insert(t);
            }
            let generation = gpu.generation;
            r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), stamp, &s, (&mut none, 0), None, None, None);
            assert_ne!(gpu.generation, generation);
            check(&gpu, &m, &sel, None, "selection");
        }
        // The same selection again: nothing to write.
        let generation = gpu.generation;
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 4, &s, (&mut none, 0), None, None, None);
        assert_eq!(gpu.generation, generation);
        // A scalar field, then back to groups.
        let vals: Vec<f32> = (0..m.max_tid()).map(|t| (t % 97) as f32 / 96.0).collect();
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 4, &s, (&mut none, 0), Some((7, TriField::Scalar(&vals))), None, None);
        check(&gpu, &m, &sel, Some((7, TriField::Scalar(&vals))), "scalars");
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 4, &s, (&mut none, 0), None, None, None);
        check(&gpu, &m, &sel, None, "groups again");
        // Growth past the buffers' capacity (new buffers), then an undo to fewer triangles.
        let cap = gpu.capacity_tris;
        let mut k = 0;
        while m.triangle_count() <= cap {
            let e = m.edge_ids().nth(k % m.edge_count()).unwrap();
            let _ = m.split_edge(e);
            k += 7;
        }
        sel.sync(&m);
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 5, &s, (&mut none, 0), None, None, None);
        assert!(gpu.capacity_tris > cap);
        check(&gpu, &m, &sel, None, "after growth");
        let mut sel_before = TriSelection::new(&before);
        sel_before.insert(5);
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &before, Some(&sel_before), 6, &s, (&mut none, 0), None, None, None);
        check(&gpu, &before, &sel_before, None, "after undo");
    }

    /// A camera looking at the origin from `eye`.
    fn view(eye: Vec3, size: [u32; 2]) -> Globals {
        let v = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Z);
        let p = glam::camera::rh::proj::directx::perspective_infinite_reverse(40f32.to_radians(), size[0] as f32 / size[1] as f32, 0.01);
        globals(p * v, v, eye, None, size[1] as f32)
    }

    #[test]
    fn lean_pipeline_draws_what_the_full_one_does() {
        let Some(rs) = render_state() else { return };
        let mut r = SceneRenderer::new(&rs);
        let size = [192, 144];
        let m = test_mesh();
        let mut sel = TriSelection::new(&m);
        for t in m.triangle_ids().step_by(5) {
            sel.insert(t);
        }
        let vals: Vec<f32> = (0..m.max_tid()).map(|t| (t % 31) as f32 / 30.0).collect();
        let rgb: Vec<u32> = (0..m.max_tid() as u32).map(|t| t.wrapping_mul(2654435761) & 0xff_ffff).collect();
        let mut base = view(Vec3::new(3.0, -4.0, 2.5), size);
        base.brush = [0.5, -0.5, 1.0, 0.6];
        base.cut_plane = [0.0, 0.0, 1.0, 1.2];
        let cases: Vec<(&str, ObjectStyle, ScalarField, bool)> = vec![
            ("object colour", style(), None, false),
            ("groups", ObjectStyle { group_colors: true, ..style() }, None, false),
            ("scalar field", style(), Some((1, TriField::Scalar(&vals))), false),
            ("per-triangle colours", style(), Some((2, TriField::Rgb(&rgb))), false),
            ("flat, dim, cut tint", ObjectStyle { flat: true, dim: true, cut_tint: true, ..style() }, None, false),
            ("metal", ObjectStyle { material: crate::materials::MATERIALS[3], ..style() }, None, false),
            ("x-ray", ObjectStyle { xray: true, group_colors: true, ..style() }, None, true),
            ("see-through", ObjectStyle { color: Vec4::new(0.5, 0.8, 0.9, 0.4), ..style() }, None, true),
        ];
        let mut none = Vec::new();
        for (label, s, scalars, transparent) in cases {
            let mut gpu = r.create_mesh(&rs.device);
            r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 1, &s, (&mut none, 0), scalars, None, None);
            assert!(!gpu.full, "{label} draws lean");
            let mut pixels = Vec::new();
            for full in [false, true] {
                gpu.full = full;
                gpu.generation = next_generation();
                r.render(&rs, size, &FrameInput { globals: base, objects: vec![(&gpu, transparent)], lines: &[], overlay_tris: &[] });
                pixels.push(read_target(&rs, &r));
            }
            let differ = pixels[0].iter().zip(&pixels[1]).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{label}: lean and full pipelines differ");
            assert!(pixels[0].chunks(4).any(|p| p[..3] != pixels[0][..3]), "{label}: something is drawn");
        }
        // Wireframe and the colour modes that need the full pipeline choose it.
        let uvs = uvs_of(&m, 0.0);
        for s in [ObjectStyle { wireframe: true, ..style() }, ObjectStyle { vertex_colors: true, ..style() }, ObjectStyle { uv_checker: true, ..style() }] {
            let mut gpu = r.create_mesh(&rs.device);
            r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, None, 1, &s, (&mut none, 0), None, Some((1, &uvs)), None);
            assert!(gpu.full);
        }
    }

    #[test]
    fn unchanged_frames_are_not_drawn_again() {
        let Some(rs) = render_state() else { return };
        let mut r = SceneRenderer::new(&rs);
        let m = test_mesh();
        let mut gpu = r.create_mesh(&rs.device);
        let mut other = r.create_mesh(&rs.device);
        let mut none = Vec::new();
        let mut sel = TriSelection::new(&m);
        let s = style();
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 1, &s, (&mut none, 0), None, None, None);
        r.sync_mesh(&rs.device, &rs.queue, &mut other, &primitives::cube(1.0, 2), None, 0, &s, (&mut none, 0), None, None, None);
        let mut size = [128, 96];
        let mut g = view(Vec3::new(3.0, -4.0, 2.5), size);
        let line = [ColorVertex::new(Vec3::ZERO, Vec4::ONE), ColorVertex::new(Vec3::X, Vec4::ONE)];
        let mut lines: Vec<ColorVertex> = line.to_vec();
        let mut drawn = r.frames_drawn;
        let mut frame = |r: &mut SceneRenderer, objects: Vec<(&GpuMesh, bool)>, g: Globals, size: [u32; 2], lines: &[ColorVertex], expect: bool, label: &str| {
            r.render(&rs, size, &FrameInput { globals: g, objects, lines, overlay_tris: &[] });
            assert_eq!(r.frames_drawn > drawn, expect, "{label}");
            drawn = r.frames_drawn;
        };
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, true, "first frame");
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, false, "same frame");
        // Syncing what has not changed writes nothing.
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 1, &s, (&mut none, 0), None, None, None);
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, false, "unchanged sync");
        g.brush = [0.0, 0.0, 1.0, 0.3];
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, true, "brush moved");
        let wire = ObjectStyle { wireframe: true, ..s };
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 1, &wire, (&mut none, 0), None, None, None);
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, true, "display option");
        sel.insert(3);
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m, Some(&sel), 2, &wire, (&mut none, 0), None, None, None);
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, true, "selection");
        let mut m2 = m.clone();
        let from = m2.shape_version();
        m2.set_vertex(0, m2.vertex(0) * 1.1);
        let mut moved = vec![0];
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &m2, Some(&sel), 2, &wire, (&mut moved, from), None, None, None);
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, true, "partial upload");
        lines.push(line[0]);
        lines.push(ColorVertex::new(Vec3::Y, Vec4::ONE));
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, true, "overlay lines");
        frame(&mut r, vec![(&gpu, false), (&other, true)], g, size, &lines, false, "same again");
        frame(&mut r, vec![(&other, true), (&gpu, false)], g, size, &lines, true, "objects reordered");
        frame(&mut r, vec![(&other, false), (&gpu, false)], g, size, &lines, true, "blending changed");
        size = [130, 96];
        frame(&mut r, vec![(&other, false), (&gpu, false)], g, size, &lines, true, "resized");
        // Image planes: shown, unchanged, a new window, gone.
        let dims = [6, 5, 4];
        let geometry = mm_volume::geometry::Geometry::from_axes([dims[0], dims[1], dims[2], 1], [0.2; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = std::sync::Arc::new(mm_volume::load::LoadedVolume::from_values("v", dims, (0..120).map(|i| i as f32).collect(), geometry));
        let mut settings = crate::image_planes::PlaneSettings::new(&vol);
        for (window, expect, label) in [(None, true, "planes shown"), (None, false, "planes unchanged"), (Some((40.0, 30.0)), true, "new window"), (Some((40.0, 30.0)), false, "same window")] {
            if let Some(w) = window {
                settings.window = w;
            }
            r.set_image_planes(Some(PlanesFrame { volume: vol.clone(), settings }));
            frame(&mut r, vec![(&other, false), (&gpu, false)], g, size, &lines, expect, label);
        }
        r.set_image_planes(None);
        frame(&mut r, vec![(&other, false), (&gpu, false)], g, size, &lines, true, "planes gone");
        frame(&mut r, vec![(&other, false), (&gpu, false)], g, size, &lines, false, "still none");
    }

    #[test]
    fn rigid_preview_draws_the_moved_object() {
        let Some(rs) = render_state() else { return };
        let mut r = SceneRenderer::new(&rs);
        let size = [192, 144];
        let g = view(Vec3::new(3.5, -5.0, 3.0), size);
        let source = test_mesh();
        let mut sel = TriSelection::new(&source);
        for t in source.triangle_ids().step_by(3) {
            sel.insert(t);
        }
        let s = ObjectStyle { group_colors: true, ..style() };
        let mut none = Vec::new();
        let mut gpu = r.create_mesh(&rs.device);
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &source, Some(&sel), 1, &s, (&mut none, 0), None, None, None);
        let rotation = glam::DQuat::from_axis_angle(DVec3::new(1.0, 2.0, 0.5).normalize(), 0.7);
        let mirror = glam::DMat4::from_scale(DVec3::new(-1.0, 1.0, 1.0));
        for (label, model) in [
            ("rotate, scale, move", glam::DMat4::from_scale_rotation_translation(DVec3::splat(0.8), rotation, DVec3::new(0.2, -0.1, -0.6))),
            ("mirror", glam::DMat4::from_translation(DVec3::new(0.0, 0.0, -0.8)) * mirror),
        ] {
            assert!(r.sync_rigid_preview(&rs.queue, &mut gpu, &s, model.as_mat4()), "{label}");
            // The preview mesh as tools make it: moved, and turned inside out again after a
            // mirror so that it faces outward; no selection.
            let mut moved = source.clone();
            moved.transform(|p| model.transform_point3(p));
            if model.determinant() < 0.0 {
                moved.reverse_orientation();
            }
            let mut reference = r.create_mesh(&rs.device);
            r.sync_mesh(&rs.device, &rs.queue, &mut reference, &moved, None, 0, &s, (&mut none, 0), None, None, None);
            let mut pixels = Vec::new();
            for m in [&gpu, &reference] {
                r.render(&rs, size, &FrameInput { globals: g, objects: vec![(m, false)], lines: &[], overlay_tris: &[] });
                pixels.push(read_target(&rs, &r));
            }
            // Positions and normals moved in f32 on the GPU rather than in f64: the same
            // picture up to rounding at silhouettes and a little shading.
            let far = pixels[0].chunks(4).zip(pixels[1].chunks(4)).filter(|(a, b)| a.iter().zip(*b).any(|(x, y)| x.abs_diff(*y) > 6)).count();
            assert!(far * 200 < size[0] as usize * size[1] as usize, "{label}: {far} pixels differ");
        }
        // Back to the object as it is: the regular uniform again.
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &source, Some(&sel), 1, &s, (&mut none, 0), None, None, None);
        assert_eq!(gpu.uniform.unwrap().flags[3] & MOVED, 0);
        // Not similarities, or group colours over a scalar field: the caller uploads instead.
        assert!(!r.sync_rigid_preview(&rs.queue, &mut gpu, &s, Mat4::from_scale(Vec3::new(1.0, 2.0, 1.0))));
        let vals = vec![0.5f32; source.max_tid()];
        r.sync_mesh(&rs.device, &rs.queue, &mut gpu, &source, Some(&sel), 1, &s, (&mut none, 0), Some((1, TriField::Scalar(&vals))), None, None);
        assert!(!r.sync_rigid_preview(&rs.queue, &mut gpu, &s, Mat4::IDENTITY));
    }

    #[test]
    fn prepared_soups_match_a_full_upload() {
        let Some(rs) = render_state() else { return };
        let r = SceneRenderer::new(&rs);
        let small = test_mesh();
        let big = scrambled(&primitives::icosphere(1.0, 5));
        let uvs = uvs_of(&small, 0.0);
        let s = ObjectStyle { vertex_colors: true, uv_checker: true, ..style() };
        let mut sel = TriSelection::new(&small);
        sel.insert(7);
        let mut none = Vec::new();
        // Preview results arriving one after another (bigger, then smaller), each prepared on
        // a job thread while the GPU copy still shows the previous one.
        let mut gpu = r.create_mesh(&rs.device);
        for (m, uv) in [(&small, Some((1, uvs.as_slice()))), (&big, None), (&small, Some((1, uvs.as_slice())))] {
            let p = std::thread::scope(|scope| scope.spawn(|| SceneRenderer::prepare_soup(&rs.device, m, &s, uv)).join().unwrap());
            let generation = gpu.generation;
            r.sync_mesh_with(&rs.device, &rs.queue, &mut gpu, m, Some(&sel), 1, &s, (&mut none, 0), None, uv, None, SyncExtras { prepared: Some(p), ..Default::default() });
            assert_ne!(gpu.generation, generation);
            let mut fresh = r.create_mesh(&rs.device);
            r.sync_mesh(&rs.device, &rs.queue, &mut fresh, m, Some(&sel), 1, &s, (&mut none, 0), None, uv, None);
            assert_eq!(gpu.tri_ids, fresh.tri_ids);
            assert!(gpu_contents(&rs, &gpu) == gpu_contents(&rs, &fresh), "a prepared soup differs from a full upload");
        }
        // Prepared for another mesh version or style: not used, the soup is built as usual.
        let p = SceneRenderer::prepare_soup(&rs.device, &big, &s, None);
        let mut moved = small.clone();
        moved.set_vertex(3, moved.vertex(3) * 1.2);
        let flat = ObjectStyle { crease_deg: None, ..s };
        let p2 = SceneRenderer::prepare_soup(&rs.device, &moved, &flat, None);
        for (prepared, m) in [(p, &moved), (p2, &moved)] {
            r.sync_mesh_with(&rs.device, &rs.queue, &mut gpu, m, None, 2, &s, (&mut none, 0), None, None, None, SyncExtras { prepared: Some(prepared), ..Default::default() });
            let mut fresh = r.create_mesh(&rs.device);
            r.sync_mesh(&rs.device, &rs.queue, &mut fresh, m, None, 2, &s, (&mut none, 0), None, None, None);
            assert!(gpu_contents(&rs, &gpu) == gpu_contents(&rs, &fresh));
            // Forget what was uploaded so that the next one is a full upload again.
            gpu.shape_version = u64::MAX;
        }
    }
}
