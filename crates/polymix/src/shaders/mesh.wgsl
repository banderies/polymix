// Mesh shading: clay-like "matcap" lighting, selection / face-group tint,
// wireframe overlay from barycentrics, and a surface brush ring.

struct Globals {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    eye: vec4<f32>,
    // xyz = brush centre, w = radius (<= 0 disables the ring)
    brush: vec4<f32>,
    // x = time, y = viewport height px
    misc: vec4<f32>,
    // Cut-plane preview: xyz = normal, w = offset; tint fragments with dot(n, p) > w.
    // Disabled when the normal is zero.
    cut_plane: vec4<f32>,
    // Optional sphere limiting the cut preview (xyz centre, w radius; w <= 0 = unlimited).
    cut_region: vec4<f32>,
    // Background colour at the top of the view.
    background: vec4<f32>,
};

struct Object {
    color: vec4<f32>,
    // x: wireframe, y: flat shading, z: color mode (0 object, 1 face groups, 2 scalar field,
    // 3 per-triangle RGB, 4 vertex colours),
    // w: bit 0 dim (non-active object), bit 1 show cut-plane tint
    flags: vec4<u32>,
    // Material: x specular, y shininess, z rim, w metal; material2.x wrap.
    material: vec4<f32>,
    material2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: Object;
@group(1) @binding(1) var<storage, read> tri_flags: array<u32>;

struct VsIn {
    @builtin(vertex_index) vid: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) bary: vec3<f32>,
    @location(3) @interpolate(flat) tflags: u32,
    @location(4) vcolor: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(in.position, 1.0);
    out.world = in.position;
    out.normal = in.normal;
    let c = in.vid % 3u;
    out.bary = vec3<f32>(f32(c == 0u), f32(c == 1u), f32(c == 2u));
    out.tflags = tri_flags[in.vid / 3u];
    out.vcolor = vec3<f32>(f32((in.color >> 16u) & 0xffu), f32((in.color >> 8u) & 0xffu), f32(in.color & 0xffu)) / 255.0;
    return out;
}

// Diverging blue-white-red colormap for t in [0, 1] (0.5 = zero deviation).
fn diverging(t: f32) -> vec3<f32> {
    let blue = vec3<f32>(0.18, 0.36, 0.85);
    let white = vec3<f32>(0.93, 0.93, 0.93);
    let red = vec3<f32>(0.85, 0.2, 0.15);
    if (t < 0.5) {
        return mix(blue, white, t * 2.0);
    }
    return mix(white, red, (t - 0.5) * 2.0);
}

fn hash_color(g: u32) -> vec3<f32> {
    var x = g * 2654435761u;
    x = x ^ (x >> 15u);
    let h = f32(x % 360u) / 360.0;
    // HSV -> RGB with fixed s, v
    let s = 0.45;
    let v = 0.9;
    let k = vec3<f32>(5.0, 3.0, 1.0);
    let p = abs(fract(vec3<f32>(h) + k / 6.0) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n: vec3<f32>;
    if (object.flags.y != 0u) {
        // Screen-space derivative normal; framebuffer y points down, so negate to face the viewer.
        n = -normalize(cross(dpdx(in.world), dpdy(in.world)));
    } else {
        n = normalize(in.normal);
        if (!front) {
            n = -n;
        }
    }
    let nv = normalize((globals.view * vec4<f32>(n, 0.0)).xyz);
    let v = normalize(globals.eye.xyz - in.world);
    let vv = normalize((globals.view * vec4<f32>(v, 0.0)).xyz);

    var base = object.color.rgb;
    let selected = (in.tflags & 1u) != 0u;
    if (object.flags.z == 1u) {
        base = hash_color(in.tflags >> 8u);
    } else if (object.flags.z == 2u) {
        base = diverging(f32((in.tflags >> 8u) & 0xffffu) / 65535.0);
    } else if (object.flags.z == 3u) {
        let rgb = in.tflags >> 8u;
        base = vec3<f32>(f32((rgb >> 16u) & 0xffu), f32((rgb >> 8u) & 0xffu), f32(rgb & 0xffu)) / 255.0;
    } else if (object.flags.z == 4u) {
        base = in.vcolor;
    }
    if (selected) {
        base = vec3<f32>(1.0, 0.45, 0.08);
    }
    if (!front) {
        // Back faces (holes, flipped normals) show as a muted blue-grey.
        base = vec3<f32>(0.32, 0.36, 0.48);
    }

    // Clay-style lighting in view space, shaped by the material.
    let spec_k = object.material.x;
    let shininess = max(object.material.y, 1.0);
    let rim_k = object.material.z;
    let metal = object.material.w;
    let wrap = object.material2.x;
    let key = normalize(vec3<f32>(-0.45, 0.65, 0.62));
    let fill = normalize(vec3<f32>(0.7, -0.2, 0.5));
    let diff = max((dot(nv, key) + wrap) / (1.0 + wrap), 0.0);
    let fdiff = max((dot(nv, fill) + wrap) / (1.0 + wrap), 0.0);
    let hv = normalize(key + vv);
    let spec = pow(max(dot(nv, hv), 0.0), shininess);
    let rim = pow(1.0 - max(dot(nv, vv), 0.0), 3.0);
    // Metals: tinted highlights, little diffuse light, and a studio reflection (bright sky,
    // dark floor, a crisp horizon) that gives them their contrast.
    let spec_col = mix(vec3<f32>(1.0), base, metal);
    let spec2 = pow(max(dot(nv, normalize(fill + vv)), 0.0), shininess) * 0.5;
    let refl = reflect(-vv, nv);
    let horizon = smoothstep(-0.05, 0.2, refl.y);
    let env = mix(vec3<f32>(0.06), mix(vec3<f32>(0.55), vec3<f32>(1.0), clamp(refl.y, 0.0, 1.0)), horizon);
    var col = base * (0.22 + 0.68 * diff + 0.22 * fdiff) * (1.0 - 0.85 * metal)
        + spec_col * (spec_k * (spec + spec2 * metal))
        + base * env * (0.9 * metal)
        + base * rim * rim_k;

    // Wireframe overlay.
    if (object.flags.x != 0u) {
        let d = fwidth(in.bary);
        let a = smoothstep(vec3<f32>(0.0), d * 1.2, in.bary);
        let edge = 1.0 - min(min(a.x, a.y), a.z);
        // Fade the wireframe out when triangles get only a few pixels across.
        let density = smoothstep(0.35, 0.12, max(max(d.x, d.y), d.z));
        col = mix(col, vec3<f32>(0.08, 0.08, 0.1), edge * 0.85 * density);
    }

    // Brush ring.
    if (globals.brush.w > 0.0) {
        let dist = distance(in.world, globals.brush.xyz);
        let r = globals.brush.w;
        let px = max(fwidth(dist), 1e-6);
        let ring = 1.0 - smoothstep(0.0, 1.8 * px, abs(dist - r));
        let inside = select(0.0, 0.08, dist < r);
        col = mix(col, vec3<f32>(1.0, 0.95, 0.4), ring * 0.9);
        col = col + vec3<f32>(inside);
    }

    // Cut preview: the half that will be discarded is tinted red (as in Meshmixer).
    if ((object.flags.w & 2u) != 0u && dot(globals.cut_plane.xyz, globals.cut_plane.xyz) > 0.0) {
        let side = dot(globals.cut_plane.xyz, in.world) - globals.cut_plane.w;
        var inside_region = true;
        if (globals.cut_region.w > 0.0) {
            inside_region = distance(in.world, globals.cut_region.xyz) < globals.cut_region.w;
        }
        if (side > 0.0 && inside_region) {
            col = mix(col, vec3<f32>(0.85, 0.15, 0.12), 0.55);
        }
    }

    if ((object.flags.w & 1u) != 0u) {
        col = mix(col, vec3<f32>(0.5), 0.35);
    }
    return vec4<f32>(col, object.color.a);
}
