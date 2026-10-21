// Volume ray casting for the 3D view: maximum intensity projection, shaded volume rendering,
// or the first surface above a threshold, with label (segment) surfaces in front.
//
// The volume is a normalised R16Float texture (0 = smallest value, 1 = largest), labels an
// R8Uint texture with an R8Unorm occupancy texture (label > 0) for smooth label surfaces.
// Each pipeline is made for one mode and for segments shown or not (the overrides below).
// Bricks of 8³ texels hold the range of values and the largest occupancy around them, so the
// ray can jump over bricks where nothing would be drawn.

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
    // xyz: bricks per axis; w: brick edge in texels.
    bricks: vec4<f32>,
    palette: array<vec4<f32>, 256>,
};

// The mode (0 MIP, 1 volume, 2 surface) and whether segments are shown, as in the uniforms.
override MODE: i32 = 1;
override LABELS: bool = true;
// Skip what cannot change the pixel: empty bricks, and the rest of a MIP ray once the window
// is saturated (off only to check that skipping changes nothing).
override SKIP: bool = true;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var occ: texture_3d<f32>;
@group(0) @binding(4) var lab: texture_3d<u32>;
// Per brick, over its texels and one more on each side (all that trilinear samples inside it
// read): the smallest and largest value (rg), and the largest occupancy.
@group(0) @binding(5) var value_bricks: texture_3d<f32>;
@group(0) @binding(6) var occupancy_bricks: texture_3d<f32>;

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

// Whether no sample inside brick `b` can draw anything (volume and surface modes). The value
// range is widened a little against rounding in the filtering; occupancy maxima of at most
// 127/255 stay below 0.5 whatever the filter does.
fn brick_empty(b: vec3<i32>) -> bool {
    if LABELS && textureLoad(occupancy_bricks, b, 0).r >= 0.5 {
        return false;
    }
    let range = textureLoad(value_bricks, b, 0).rg;
    if MODE == 1 {
        // Opacity grows with the grey level, which is monotonic in the value.
        let g = max(grey(range.x - 1e-3), grey(range.y + 1e-3));
        return clamp(g * g * u.labels.z / u.dims.w, 0.0, 1.0) <= 0.002;
    }
    return range.y + 1e-3 < u.window.z;
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
    let mode = MODE;
    let show_labels = LABELS;
    // MIP without segments: past this value the grey level is 1 (or 0 inverted) for good.
    let saturated = u.window.x + max(u.window.y - u.window.x, 1e-6);
    let nb = vec3<i32>(u.bricks.xyz);
    var brick = vec3<i32>(-1);

    var mip = 0.0;
    var acc = vec3<f32>(0.0);
    var alpha = 0.0;
    var prev = 0.0;
    var prev_occ = 0.0;
    var oc = 0.0;
    var s = 0;
    loop {
        if s >= n {
            break;
        }
        // Fused multiply-adds, as the compiler made them before the shader was specialised: left
        // to its fast math, a specialised MIP rounds differently (1/255 in a few pixels).
        let t = fma(f32(s) + 0.5, dt, t0);
        let p = fma(d, vec3<f32>(t), o);
        if SKIP && mode != 0 {
            let b = clamp(vec3<i32>(floor(p * u.dims.xyz / u.bricks.w)), vec3<i32>(0), nb - vec3<i32>(1));
            if any(b != brick) {
                brick = b;
                if brick_empty(b) {
                    // On to the first sample past the brick. The samples skipped would have
                    // left only these behind: below the threshold, outside segments.
                    let lo = vec3<f32>(b) * u.bricks.w / u.dims.xyz;
                    let hi = (vec3<f32>(b) + vec3<f32>(1.0)) * u.bricks.w / u.dims.xyz;
                    let tx = max((lo - o) / safe, (hi - o) / safe);
                    let exit = min(min(tx.x, tx.y), tx.z);
                    s = max(s + 1, i32(ceil(min((exit - t0) / dt - 0.5, f32(n)))));
                    prev = u.window.z - 1.0;
                    prev_occ = 0.0;
                    continue;
                }
            }
        }
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
            // Nothing further along can change a saturated grey level (checked as grey() has
            // it, so the pixel is exactly the one the whole ray gives).
            if SKIP && !show_labels && mip >= saturated && clamp((mip - u.window.x) / max(u.window.y - u.window.x, 1e-6), 0.0, 1.0) >= 1.0 {
                break;
            }
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
        s += 1;
    }
    if mode == 0 {
        let g = grey(mip);
        return vec4<f32>(mix(u.background.xyz, vec3<f32>(g), clamp(g * 4.0, 0.0, 1.0)), 1.0);
    }
    return vec4<f32>(acc + (1.0 - alpha) * u.background.xyz, 1.0);
}

// Showing a frame drawn earlier into the view's cache texture (drawn again only when what it
// shows changes): 1:1 from where the view is on screen (where it was drawn, as rays start
// from positions interpolated in screen coordinates), or scaled up from a half-size frame at
// the cache's top left while the view moves.
struct Blit {
    // The view's top left corner in the target and its size, in pixels.
    origin: vec2<f32>,
    size: vec2<f32>,
    // The size of the half-size frame, and the cache's size.
    drawn: vec2<f32>,
    cache_size: vec2<f32>,
};

@group(0) @binding(10) var<uniform> blit_u: Blit;
@group(0) @binding(11) var cache: texture_2d<f32>;
@group(0) @binding(12) var cache_samp: sampler;

@fragment
fn blit(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Pixel centres are at .5: the pixel drawn at the same place.
    return textureLoad(cache, vec2<i32>(pos.xy), 0);
}

@fragment
fn blit_scaled(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = (pos.xy - blit_u.origin) / blit_u.size * blit_u.drawn;
    // Not past the frame's edge pixels (the rest of the cache is stale).
    let at = clamp(px, vec2<f32>(0.5), blit_u.drawn - vec2<f32>(0.5));
    return textureSampleLevel(cache, cache_samp, at / blit_u.cache_size, 0.0);
}
