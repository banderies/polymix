// Colored lines and translucent triangles drawn over the scene (boundaries, gizmos),
// plus the background gradient.

struct Globals {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    eye: vec4<f32>,
    brush: vec4<f32>,
    misc: vec4<f32>,
    cut_plane: vec4<f32>,
    cut_region: vec4<f32>,
    // Background colour at the top of the view.
    background: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_lines(in: VsIn) -> VsOut {
    var out: VsOut;
    // Pull lines slightly toward the camera (proportional to their distance) so they win the
    // depth test against the surface they lie on without showing through the model.
    let to_eye = globals.eye.xyz - in.position;
    let p = in.position + to_eye * 0.002;
    out.clip = globals.view_proj * vec4<f32>(p, 1.0);
    out.color = in.color;
    return out;
}

@vertex
fn vs_tris(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_color(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}

struct BgOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) t: f32,
};

@vertex
fn vs_background(@builtin(vertex_index) vid: u32) -> BgOut {
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    var out: BgOut;
    // Reversed Z: the background sits at depth 0 (infinitely far).
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.t = y * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_background(in: BgOut) -> @location(0) vec4<f32> {
    let top = globals.background.rgb;
    let bottom = top * 0.28;
    return vec4<f32>(mix(bottom, top, in.t), 1.0);
}
