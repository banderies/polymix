// Image slices among the meshes: one textured quad per plane, unlit, seen from both sides,
// with an optional outline in the plane's colour.

struct Globals {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    eye: vec4<f32>,
    brush: vec4<f32>,
    misc: vec4<f32>,
    cut_plane: vec4<f32>,
    cut_region: vec4<f32>,
    background: vec4<f32>,
};

struct Plane {
    // World position of the image's top-left corner, and the edges to the top-right and
    // bottom-left corners.
    origin: vec4<f32>,
    across: vec4<f32>,
    down: vec4<f32>,
    // xy: texture coordinates of the bottom-right corner (below 1 when a strided image runs
    // past the slice); zw: where the other two planes cut this one (0..1 across, down).
    tex: vec4<f32>,
    // rgb: outline colour (linear); a: 1 with an outline, 0 without.
    outline: vec4<f32>,
    // x: opacity.
    misc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> plane: Plane;
@group(1) @binding(1) var image: texture_2d<f32>;
@group(1) @binding(2) var image_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // Position on the whole quad, 0..1 across and down.
    @location(0) st: vec2<f32>,
};

// Six vertices per quarter (instance) of the quad: the quarters meet where the other planes
// cut it, so see-through planes can be drawn back to front piece by piece.
@vertex
fn vs_plane(@builtin(vertex_index) vid: u32, @builtin(instance_index) quarter: u32) -> VsOut {
    var corner = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let upper = vec2<bool>((quarter & 1u) != 0u, (quarter & 2u) != 0u);
    let lo = select(vec2<f32>(0.0), plane.tex.zw, upper);
    let hi = select(plane.tex.zw, vec2<f32>(1.0), upper);
    let st = mix(lo, hi, corner[vid]);
    let p = plane.origin.xyz + plane.across.xyz * st.x + plane.down.xyz * st.y;
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(p, 1.0);
    out.st = st;
    return out;
}

@fragment
fn fs_plane(in: VsOut) -> @location(0) vec4<f32> {
    let grey = textureSample(image, image_sampler, in.st * plane.tex.xy).rgb;
    // Outline about 2 pixels wide along the edges, whatever the zoom.
    let px = max(fwidth(in.st), vec2<f32>(1e-6));
    let e = min(in.st, 1.0 - in.st) / px;
    let line = (1.0 - smoothstep(1.0, 2.0, min(e.x, e.y))) * plane.outline.a;
    return mix(vec4<f32>(grey, plane.misc.x), vec4<f32>(plane.outline.rgb, 1.0), line);
}
