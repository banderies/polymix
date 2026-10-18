// Volume ray casting for the 3D view: maximum intensity projection, shaded volume rendering,
// or the first surface above a threshold, with label (segment) surfaces in front.
//
// The volume is a normalised R16Float texture (0 = smallest value, 1 = largest), labels an
// R8Uint texture with an R8Unorm occupancy texture (label > 0) for smooth label surfaces.

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    world_to_tex: mat4x4<f32>,
    // Texture-space gradient → world normal (transpose of world_to_tex's 3×3).
    grad_to_world: mat4x4<f32>,
    // xyz: texture size in texels; w: samples per texel along the ray.
    dims: vec4<f32>,
    // Window low and high, surface threshold (normalised), mode (0 MIP, 1 volume, 2 surface).
    window: vec4<f32>,
    // Labels shown (0 / 1), label opacity, image opacity scale, inverted grey (0 / 1).
    labels: vec4<f32>,
    background: vec4<f32>,
    // World direction towards the light (the camera's).
    light: vec4<f32>,
    palette: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var occ: texture_3d<f32>;
@group(0) @binding(4) var lab: texture_3d<u32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VOut;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    o.ndc = p[i];
    return o;
}

fn value(t: vec3<f32>) -> f32 {
    return textureSampleLevel(vol, samp, t, 0.0).r;
}

fn occupancy(t: vec3<f32>) -> f32 {
    return textureSampleLevel(occ, samp, t, 0.0).r;
}

fn grad_value(t: vec3<f32>) -> vec3<f32> {
    let h = 1.0 / u.dims.xyz;
    return vec3<f32>(
        value(t + vec3<f32>(h.x, 0.0, 0.0)) - value(t - vec3<f32>(h.x, 0.0, 0.0)),
        value(t + vec3<f32>(0.0, h.y, 0.0)) - value(t - vec3<f32>(0.0, h.y, 0.0)),
        value(t + vec3<f32>(0.0, 0.0, h.z)) - value(t - vec3<f32>(0.0, 0.0, h.z)),
    ) * u.dims.xyz * 0.5;
}

fn grad_occupancy(t: vec3<f32>) -> vec3<f32> {
    let h = 1.0 / u.dims.xyz;
    return vec3<f32>(
        occupancy(t + vec3<f32>(h.x, 0.0, 0.0)) - occupancy(t - vec3<f32>(h.x, 0.0, 0.0)),
        occupancy(t + vec3<f32>(0.0, h.y, 0.0)) - occupancy(t - vec3<f32>(0.0, h.y, 0.0)),
        occupancy(t + vec3<f32>(0.0, 0.0, h.z)) - occupancy(t - vec3<f32>(0.0, 0.0, h.z)),
    ) * u.dims.xyz * 0.5;
}

// Outward world normal from a texture-space gradient pointing inside.
fn normal_of(g: vec3<f32>) -> vec3<f32> {
    let n = (u.grad_to_world * vec4<f32>(-g, 0.0)).xyz;
    let l = length(n);
    if l < 1e-8 {
        return -u.light.xyz;
    }
    return n / l;
}

fn shade(base: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let l = u.light.xyz;
    // Two-sided, so surfaces seen from inside are lit too.
    let d = abs(dot(n, l));
    // Light at the camera: the half vector is the light direction itself.
    let s = pow(d, 40.0) * 0.15;
    return base * (0.25 + 0.75 * d) + vec3<f32>(s);
}

fn grey(v: f32) -> f32 {
    let g = clamp((v - u.window.x) / max(u.window.y - u.window.x, 1e-6), 0.0, 1.0);
    return select(g, 1.0 - g, u.labels.w > 0.5);
}

// The label at texture position `t` (the voxel there, or a labelled neighbour on the edge).
fn label_at(t: vec3<f32>) -> u32 {
    let size = vec3<i32>(u.dims.xyz);
    let c = clamp(vec3<i32>(floor(t * u.dims.xyz)), vec3<i32>(0), size - vec3<i32>(1));
    let l = textureLoad(lab, c, 0).r;
    if l != 0u {
        return l;
    }
    for (var dz = -1; dz <= 1; dz += 1) {
        for (var dy = -1; dy <= 1; dy += 1) {
            for (var dx = -1; dx <= 1; dx += 1) {
                let q = clamp(c + vec3<i32>(dx, dy, dz), vec3<i32>(0), size - vec3<i32>(1));
                let m = textureLoad(lab, q, 0).r;
                if m != 0u {
                    return m;
                }
            }
        }
    }
    return 0u;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let bg = vec4<f32>(u.background.xyz, 1.0);
    let near = u.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = u.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let o = (u.world_to_tex * vec4<f32>(near.xyz / near.w, 1.0)).xyz;
    let e = (u.world_to_tex * vec4<f32>(far.xyz / far.w, 1.0)).xyz;
    let d = e - o;
    let safe = select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
    let ta = (vec3<f32>(0.0) - o) / safe;
    let tb = (vec3<f32>(1.0) - o) / safe;
    let tmin = min(ta, tb);
    let tmax = max(ta, tb);
    let t0 = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    let t1 = min(min(tmax.x, tmax.y), min(tmax.z, 1.0));
    if t1 <= t0 {
        return bg;
    }
    let span_vox = length(d * u.dims.xyz) * (t1 - t0);
    let n = clamp(i32(span_vox * u.dims.w), 1, 2048);
    let dt = (t1 - t0) / f32(n);
    let mode = i32(u.window.w + 0.5);
    let show_labels = u.labels.x > 0.5;

    var mip = 0.0;
    var acc = vec3<f32>(0.0);
    var alpha = 0.0;
    var prev = 0.0;
    var prev_occ = 0.0;
    var oc = 0.0;
    for (var s = 0; s < n; s += 1) {
        let t = t0 + (f32(s) + 0.5) * dt;
        let p = o + d * t;
        if show_labels {
            oc = occupancy(p);
            if oc >= 0.5 {
                // Refine the crossing between the previous sample and this one.
                var lo = t - dt;
                var hi = t;
                if s > 0 && prev_occ < 0.5 {
                    for (var r = 0; r < 5; r += 1) {
                        let m = 0.5 * (lo + hi);
                        if occupancy(o + d * m) >= 0.5 { hi = m; } else { lo = m; }
                    }
                }
                let q = o + d * hi;
                let id = label_at(q);
                let c = u.palette[id];
                let lit = shade(c.xyz, normal_of(grad_occupancy(q)));
                let a = u.labels.y;
                if mode == 0 {
                    // Labels in front of the projection of what lies before them.
                    let g = grey(mip);
                    let behind = mix(u.background.xyz, vec3<f32>(g), clamp(g * 4.0, 0.0, 1.0));
                    return vec4<f32>(mix(behind, lit, a), 1.0);
                }
                acc += (1.0 - alpha) * a * lit;
                alpha += (1.0 - alpha) * a;
                if alpha > 0.98 {
                    break;
                }
            }
            prev_occ = oc;
        }
        let v = value(p);
        if mode == 0 {
            mip = max(mip, v);
        } else if mode == 1 {
            let g = grey(v);
            let a = clamp(g * g * u.labels.z / u.dims.w, 0.0, 1.0);
            if a > 0.002 {
                let lit = shade(vec3<f32>(g), normal_of(grad_value(p)));
                acc += (1.0 - alpha) * a * lit;
                alpha += (1.0 - alpha) * a;
                if alpha > 0.98 {
                    break;
                }
            }
        } else {
            // Next to a segment, its surface wins over the image's.
            if v >= u.window.z && !(show_labels && oc > 0.02) {
                var lo = t - dt;
                var hi = t;
                if s > 0 && prev < u.window.z {
                    for (var r = 0; r < 5; r += 1) {
                        let m = 0.5 * (lo + hi);
                        if value(o + d * m) >= u.window.z { hi = m; } else { lo = m; }
                    }
                }
                let q = o + d * hi;
                let lit = shade(vec3<f32>(0.93, 0.89, 0.80), normal_of(grad_value(q)));
                acc += (1.0 - alpha) * lit;
                alpha = 1.0;
                break;
            }
            prev = v;
        }
    }
    if mode == 0 {
        let g = grey(mip);
        return vec4<f32>(mix(u.background.xyz, vec3<f32>(g), clamp(g * 4.0, 0.0, 1.0)), 1.0);
    }
    return vec4<f32>(acc + (1.0 - alpha) * u.background.xyz, 1.0);
}
