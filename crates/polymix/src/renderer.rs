//! wgpu renderer: draws the scene into an offscreen MSAA target that egui displays as an
//! image. Meshes are uploaded as triangle soups so per-triangle state (selection, face
//! group) can live in a compact storage buffer indexed by `vertex_index / 3`.

use crate::image_planes::{ImagePlanesGpu, PlanesFrame};
use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu;
use eframe::wgpu;
use glam::{Mat4, Vec3, Vec4};
use mm_core::shading::{corner_normal, ShadingNormals};
use mm_core::{Mesh, TriSelection};
use rayon::prelude::*;

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
}

/// An object's texture for [`SceneRenderer::sync_mesh`]: stamp, width, height, RGBA rows.
pub type TextureData<'a> = (u64, u32, u32, &'a [u8]);

pub struct SceneRenderer {
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    object_bgl: wgpu::BindGroupLayout,
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_pipeline_transparent: wgpu::RenderPipeline,
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

        let make_mesh_pipeline = |transparent: bool| {
            let targets = blend_target(if transparent { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if transparent { "mesh-transparent" } else { "mesh" }),
                layout: Some(&mesh_layout),
                vertex: wgpu::VertexState {
                    module: &mesh_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(mesh_vertex_layout.clone())],
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
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let mesh_pipeline = make_mesh_pipeline(false);
        let mesh_pipeline_transparent = make_mesh_pipeline(true);

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
            line_pipeline,
            tri_overlay_pipeline,
            background_pipeline,
            lines_buf: None,
            overlay_tris_buf: None,
            image_planes,
            white,
            sampler,
            target: None,
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
        }
    }

    fn alloc_mesh_buffers(device: &wgpu::Device, tris: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh vertices"),
            size: (tris.max(1) * 3 * std::mem::size_of::<MeshVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tri flags"),
            size: (tris.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (vb, fb)
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
        let texture = texture.filter(|_| style.texture_mode != 0 && uvs.is_some());
        self.sync_texture(device, queue, gpu, texture);
        let (moved_vertices, moved_from) = moved;
        let groups_key = style.group_normals.then_some(mesh.group_version());
        let show_colors = style.vertex_colors && mesh.has_vertex_colors();
        let colors_key = show_colors.then_some(mesh.color_version());
        let uvs = uvs.filter(|_| style.uv_checker || style.texture_mode != 0);
        let uv_key = uvs.map(|u| u.0);
        let uv_slice = uvs.map(|u| u.1);
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
        if partial {
            Self::partial_update(queue, gpu, mesh, moved_vertices, crease, groups, show_colors, uv_slice);
            gpu.shape_version = mesh.shape_version();
            gpu.colors_key = colors_key;
        } else if geometry_changed {
            let tri_ids: Vec<u32> = mesh.triangle_ids().collect();
            let normals = ShadingNormals::new(mesh, crease, groups);
            let verts: Vec<MeshVertex> = tri_ids
                .par_iter()
                .flat_map_iter(|&t| {
                    let normals = &normals;
                    mesh.triangle(t).into_iter().enumerate().map(move |(j, v)| MeshVertex {
                        position: mesh.vertex(v).as_vec3().to_array(),
                        normal: normals.corner(t, j).as_vec3().to_array(),
                        color: packed_color(mesh, v, show_colors),
                        uv: corner_uv(uv_slice, t, j),
                    })
                })
                .collect();
            // Never exceed the device's buffer limits (very large meshes are truncated with a
            // warning rather than crashing).
            let max_tris = (device.limits().max_buffer_size / (3 * std::mem::size_of::<MeshVertex>()) as u64)
                .min(device.limits().max_storage_buffer_binding_size / 4) as usize;
            let mut tri_ids = tri_ids;
            let mut verts = verts;
            if tri_ids.len() > max_tris {
                log::warn!("mesh has {} triangles; only {} fit in GPU buffers", tri_ids.len(), max_tris);
                tri_ids.truncate(max_tris);
                verts.truncate(max_tris * 3);
            }
            if tri_ids.len() > gpu.capacity_tris {
                let cap = ((tri_ids.len() as f64 * 1.25) as usize).min(max_tris);
                let (vb, fb) = Self::alloc_mesh_buffers(device, cap);
                gpu.vertex_buf = vb;
                gpu.flags_buf = fb;
                gpu.capacity_tris = cap;
                let view = gpu.texture.as_ref().map_or(&self.white, |t| &t.2);
                gpu.bind_group = self.make_object_bg(device, &gpu.uniform_buf, &gpu.flags_buf, view);
            }
            if !verts.is_empty() {
                queue.write_buffer(&gpu.vertex_buf, 0, bytemuck::cast_slice(&verts));
            }
            gpu.tri_count = tri_ids.len() as u32;
            gpu.soup_index = vec![u32::MAX; mesh.max_tid()];
            for (i, &t) in tri_ids.iter().enumerate() {
                gpu.soup_index[t as usize] = i as u32;
            }
            gpu.tri_ids = tri_ids;
            gpu.shape_version = mesh.shape_version();
            gpu.topo_version = mesh.topology_version();
            gpu.crease_deg = style.crease_deg;
            gpu.group_normals = groups_key;
            gpu.colors_key = colors_key;
            gpu.uv_key = uv_key;
        }
        moved_vertices.clear();
        let scalar_stamp = scalars.map(|(s, _)| s);
        if (geometry_changed && !partial) || gpu.selection_stamp != selection_stamp || gpu.scalar_stamp != scalar_stamp {
            let flags: Vec<u32> = gpu
                .tri_ids
                .par_iter()
                .map(|&t| {
                    let sel = selection.map(|s| s.contains(t)).unwrap_or(false) as u32;
                    let payload = match scalars {
                        Some((_, TriField::Scalar(vals))) => (vals.get(t as usize).copied().unwrap_or(0.5).clamp(0.0, 1.0) * 65535.0) as u32,
                        Some((_, TriField::Rgb(cols))) => cols.get(t as usize).copied().unwrap_or(0x808080) & 0x00ff_ffff,
                        None => (mesh.tri_group(t) as u32) & 0x00ff_ffff,
                    };
                    sel | (payload << 8)
                })
                .collect();
            if !flags.is_empty() {
                queue.write_buffer(&gpu.flags_buf, 0, bytemuck::cast_slice(&flags));
            }
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
        };
        queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    /// Re-upload only the triangles around moved vertices (positions and normals).
    #[allow(clippy::too_many_arguments)]
    fn partial_update(queue: &wgpu::Queue, gpu: &GpuMesh, mesh: &Mesh, moved: &[u32], crease: Option<f64>, groups: bool, colors: bool, uvs: Option<&[[[f32; 2]; 3]]>) {
        // Normals change on the one-ring of moved vertices, so refresh their triangles too.
        let mut verts: Vec<u32> = Vec::with_capacity(moved.len() * 7);
        for &v in moved {
            if mesh.is_vertex(v) {
                verts.push(v);
                verts.extend(mesh.vertex_neighbors(v));
            }
        }
        verts.sort_unstable();
        verts.dedup();
        let mut tris: Vec<u32> = verts.iter().flat_map(|&v| mesh.vertex_triangles(v)).collect();
        tris.sort_unstable();
        tris.dedup();
        let mut slots: Vec<(u32, u32)> = tris
            .iter()
            .filter_map(|&t| gpu.soup_index.get(t as usize).copied().filter(|&i| i != u32::MAX).map(|i| (i, t)))
            .collect();
        slots.sort_unstable();
        let stride = 3 * std::mem::size_of::<MeshVertex>() as u64;
        // Upload contiguous runs of soup triangles in one write each.
        let mut k = 0;
        while k < slots.len() {
            let start = slots[k].0;
            let mut run: Vec<MeshVertex> = Vec::new();
            let mut j = k;
            while j < slots.len() && slots[j].0 == start + (j - k) as u32 {
                let t = slots[j].1;
                for (j, v) in mesh.triangle(t).into_iter().enumerate() {
                    run.push(MeshVertex {
                        position: mesh.vertex(v).as_vec3().to_array(),
                        normal: corner_normal(mesh, t, j, crease, groups).as_vec3().to_array(),
                        color: packed_color(mesh, v, colors),
                        uv: corner_uv(uvs, t, j),
                    });
                }
                j += 1;
            }
            queue.write_buffer(&gpu.vertex_buf, start as u64 * stride, bytemuck::cast_slice(&run));
            k = j;
        }
    }

    fn ensure_target(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2]) -> egui::TextureId {
        let size = [size[0].max(1), size[1].max(1)];
        if let Some(t) = &self.target {
            if t.size == size {
                return t.texture_id;
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
        let msaa = mk("scene msaa", COLOR_FORMAT, SAMPLE_COUNT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolve = mk(
            "scene resolve",
            COLOR_FORMAT,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = mk("scene depth", DEPTH_FORMAT, SAMPLE_COUNT, wgpu::TextureUsages::RENDER_ATTACHMENT);
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
        });
        texture_id
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

    /// Render the frame into the offscreen target and return its egui texture id.
    pub fn render(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2], frame: &FrameInput) -> egui::TextureId {
        let tex = self.ensure_target(rs, size);
        let device = &rs.device;
        let queue = &rs.queue;
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&frame.globals));
        Self::upload_dynamic(device, queue, &mut self.lines_buf, frame.lines);
        Self::upload_dynamic(device, queue, &mut self.overlay_tris_buf, frame.overlay_tris);
        self.image_planes.prepare(device, queue, &frame.globals);

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

            pass.set_pipeline(&self.mesh_pipeline);
            for (m, transparent) in &frame.objects {
                if *transparent || m.tri_count == 0 {
                    continue;
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
            pass.set_pipeline(&self.mesh_pipeline_transparent);
            for (m, transparent) in &frame.objects {
                if !*transparent || m.tri_count == 0 {
                    continue;
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
