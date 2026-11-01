// SPDX-License-Identifier: AGPL-3.0-only
//! From a click to the whole surface. A patch round the seed gives the robust estimate (see
//! `ransac`), retried on a larger patch when it is too small or too flat; the region then
//! grows from the seed, the model refitted every round, and a vertex joins only inside the
//! noise band and with a normal that agrees with the model's — the normal test is what stops
//! it at an edge rather than leaking onto the next face. The rings of a scan's rounded edge
//! that still get in are peeled off the rim afterwards. A plain flood bounded by the turn
//! between neighbouring normals sits beside it, fitting nothing.
//!
//! Ported from ScanRuler (`src/core/fit/regionGrow.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use std::cmp::Ordering;
use std::collections::HashSet;

use glam::DVec3;

use super::cone::{cone_normal_align, cone_normal_signed, cone_residual, fit_cone_clipped};
use super::cylinder::{cylinder_residual, fit_cylinder_clipped};
use super::guide::{in_box, FitGuide, Side};
use super::linalg3::{js_max, js_min};
use super::plane::{fit_plane_clipped, plane_residual};
use super::sphere::fit_sphere_clipped;
use super::torus::{fit_torus_clipped, torus_normal_align, torus_residual};
use super::types::{Cone, Cylinder, Plane, Sphere, Torus};
use crate::fit2d::FitError;
use crate::mesh_graph::{MeshGraph, Stamps};

/// Vertices whose fit uses a strided subsample beyond this size; the region itself is still
/// complete, only the per-round solve is capped.
const FIT_SUBSAMPLE: usize = 60_000;

fn cos_deg(deg: f64) -> f64 {
    (deg * std::f64::consts::PI / 180.0).cos()
}

/// Smallest |cos| between a vertex normal and the model's that still counts as the same
/// surface, per kind: 38° for a sphere, 32° for a cylinder, cone and torus, 25° for a plane.
fn cos_sphere_max() -> f64 {
    cos_deg(38.0)
}
fn cos_cylinder_max() -> f64 {
    cos_deg(32.0)
}
fn cos_cone_max() -> f64 {
    cos_deg(32.0)
}
fn cos_plane_max() -> f64 {
    cos_deg(25.0)
}

/// A rim ring is peeled while its normals turn away from the model by at least this multiple
/// of the surface's own noise tilt (the median over the region) — twice the noise is a
/// surface starting to curve away, not noise.
const PEEL_TILT_FACTOR: f64 = 2.0;
/// …and by at least this much in absolute terms (half a degree), so a perfectly tessellated
/// mesh (noise tilt zero) does not peel on rounding error.
const PEEL_TILT_FLOOR: f64 = (0.5 * std::f64::consts::PI) / 180.0;
/// Rings peeled at most — the rounding of a scanned edge is a couple of vertex rings wide;
/// anything beyond is not an edge.
const MAX_PEEL_RINGS: usize = 6;

/// Collect up to `limit` vertices connected to the seeds — no membership criterion, plain
/// surface breadth-first walk. Used to gather the robust estimate's patch. Every seed is
/// taken, even past the limit; a seed that is no vertex of the graph is skipped.
pub fn collect_patch(g: &MeshGraph, seeds: &[u32], limit: usize) -> Vec<u32> {
    let mut stamps = Stamps::new(g.vertex_count);
    let mut queue: Vec<u32> = Vec::new();
    for &s in seeds {
        if (s as usize) < g.vertex_count && !stamps.seen(s) {
            stamps.mark(s);
            queue.push(s);
        }
    }
    let mut head = 0;
    while head < queue.len() && queue.len() < limit {
        let v = queue[head];
        head += 1;
        for &nb in g.neighbours(v) {
            if stamps.seen(nb) {
                continue;
            }
            stamps.mark(nb);
            queue.push(nb);
            if queue.len() >= limit {
                break;
            }
        }
    }
    queue
}

/// The vertex ids of `ids` that name vertices of the graph, in their order. ScanRuler reads an
/// id past the mesh as undefined coordinates that fail every test further on; here such an
/// id is left out before anything indexes with it.
pub(crate) fn on_graph(g: &MeshGraph, ids: &[u32]) -> Vec<u32> {
    ids.iter().copied().filter(|&v| (v as usize) < g.vertex_count).collect()
}

/// The patch sizes a fit from a click tries in turn — the next when the one before was too
/// small or too flat to pin the surface down.
pub const SEED_PATCHES: [usize; 3] = [1500, 6000, 24000];

/// The seeds that grew nothing, of a region grown from them: a click on another surface than
/// the one the estimate found, among several clicks — its vertices are in the region only as
/// the places the growing started from, standing alone. Seeds joined to one another (a
/// click's triangle) go together, and stay while one of them meets a vertex of the region
/// that is no seed.
pub fn stranded_seeds(g: &MeshGraph, region: &[u32], seeds: &[u32]) -> HashSet<u32> {
    let mut stamps = Stamps::new(g.vertex_count);
    for &v in region {
        if (v as usize) < g.vertex_count {
            stamps.mark(v);
        }
    }
    let mut is_seed: Vec<u32> = Vec::new();
    let mut is_seed_set: HashSet<u32> = HashSet::new();
    for &s in seeds {
        if (s as usize) < g.vertex_count && stamps.seen(s) && is_seed_set.insert(s) {
            is_seed.push(s);
        }
    }
    let mut stranded = HashSet::new();
    let mut seen: HashSet<u32> = HashSet::new();
    for &s in &is_seed {
        if !seen.insert(s) {
            continue;
        }
        let mut joined = vec![s];
        let mut grew = false;
        let mut k = 0;
        while k < joined.len() {
            let v = joined[k];
            k += 1;
            for &nb in g.neighbours(v) {
                if !stamps.seen(nb) {
                    continue;
                }
                if !is_seed_set.contains(&nb) {
                    grew = true;
                } else if seen.insert(nb) {
                    joined.push(nb);
                }
            }
        }
        if !grew {
            stranded.extend(joined);
        }
    }
    stranded
}

/// A region grown from a seed with the model refitted on it.
#[derive(Debug, Clone, PartialEq)]
pub struct GrowResult<T> {
    pub region: Vec<u32>,
    pub model: T,
    pub sigma: f64,
}

/// A fit from a click, the steps every kind shares: a patch round the seeds, a robust
/// estimate on it, the region that estimate grows to — 60 points at least, seeds that grew
/// nothing left out of it (`stranded_seeds`) — and the kind's own best fit and checks. A step
/// that comes back `None` moves on to the next patch size; `None` when none is left. `finish`
/// may refuse outright with an error, which ends the search.
///
/// `estimate` is handed the patch and the seed of its sampling (the first click plus the
/// patch size).
pub fn seeded_fit<C, T, R>(
    g: &MeshGraph,
    seeds: &[u32],
    mut estimate: impl FnMut(&[u32], u32) -> Option<C>,
    mut grow: impl FnMut(C) -> Option<GrowResult<T>>,
    mut finish: impl FnMut(GrowResult<T>) -> Result<Option<R>, FitError>,
) -> Result<Option<R>, FitError> {
    for patch_size in SEED_PATCHES {
        let patch = collect_patch(g, seeds, patch_size);
        if patch.len() < 30 {
            break;
        }
        let seed = seeds.first().copied().unwrap_or(1).wrapping_add(patch_size as u32);
        let Some(cand) = estimate(&patch, seed) else { continue };
        let Some(mut grown) = grow(cand) else { continue };
        if grown.region.len() < 60 {
            continue;
        }
        // Clicks on two surfaces grow the one the estimate found; the other's seeds would
        // stand in its region alone, in its box, extents and fit.
        if seeds.len() > 1 {
            let stranded = stranded_seeds(g, &grown.region, seeds);
            if !stranded.is_empty() {
                grown.region.retain(|v| !stranded.contains(v));
            }
        }
        if grown.region.len() < 60 {
            continue;
        }
        if let Some(fit) = finish(grown)? {
            return Ok(Some(fit));
        }
    }
    Ok(None)
}

/// Flood from a seed over every edge whose two vertex normals turn by less than
/// `max_angle_deg` — QuickSurface's click-and-grow, with the angle as the slider. Unlike the
/// model-guided growing below, nothing is fitted and nothing peeled: the region is whatever
/// the surface reaches without a crease, which on a tessellated part is exactly one face and
/// on a scan is a patch the angle bounds. `limit` caps the flood so a degree set wide cannot
/// take the whole part before the hand can react (ScanRuler's default: the vertex count).
/// Empty when the seed is no vertex of the graph.
pub fn flood_by_normal(g: &MeshGraph, seed: u32, max_angle_deg: f64, limit: Option<usize>) -> Vec<u32> {
    if seed as usize >= g.vertex_count {
        return Vec::new();
    }
    let limit = limit.unwrap_or(g.vertex_count);
    let cos_min = (js_max(0.0, max_angle_deg) * std::f64::consts::PI / 180.0).cos();
    let n = &g.normals;
    let mut stamps = Stamps::new(g.vertex_count);
    let mut queue = vec![seed];
    stamps.mark(seed);
    let mut head = 0;
    while head < queue.len() && queue.len() < limit {
        let v = queue[head];
        head += 1;
        let nv = n[v as usize];
        for &nb in g.neighbours(v) {
            if stamps.seen(nb) {
                continue;
            }
            let nn = n[nb as usize];
            if nv.x * nn.x + nv.y * nn.y + nv.z * nn.z < cos_min {
                continue;
            }
            stamps.mark(nb);
            queue.push(nb);
            if queue.len() >= limit {
                break;
            }
        }
    }
    queue
}

/// A test of a vertex normal against the model at a point.
type NormalTest<'a, T> = &'a dyn Fn(&T, DVec3, DVec3) -> f64;
/// A clipped refit from the model on a set of vertices: the new model and its σ.
type Refit<'a, T> = &'a dyn Fn(&[DVec3], &[u32], &T) -> Option<(T, f64)>;

/// What one geometry needs to contribute to the shared growing loop.
struct GrowSpec<'a, T> {
    /// Smallest |cos| between a vertex normal and the model's surface normal that still
    /// counts as the same surface.
    cos_max: f64,
    /// Half-width of the acceptance band around the surface, from the model and its σ.
    band: &'a dyn Fn(&T, f64) -> f64,
    /// Distance of a vertex from the model surface; NaN where undefined.
    residual: &'a dyn Fn(&T, DVec3) -> f64,
    /// |cos| between the vertex normal and the model's surface normal there.
    align: NormalTest<'a, T>,
    /// The same cosine signed: positive where the vertex normal points away from the axis or
    /// centre (a convex surface), for a guide's side.
    signed: Option<NormalTest<'a, T>>,
    /// The clipped refit on the region (or a subsample of it), starting from the model:
    /// the new model and its σ.
    refit: Refit<'a, T>,
}

/// JavaScript's numeric sort of a `Float32Array`: ascending, a negative zero before a
/// positive one, NaN last.
fn sort_f32_like_js(values: &mut [f32]) {
    values.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.total_cmp(b),
    });
}

/// Peel the transition off the rim of a grown region: a scan rounds every edge over a few
/// vertex rings, and the growing test lets the first of them in — their distance is still
/// inside the band and their normals still within the coarse angle that has to tolerate scan
/// noise. Those rings sit systematically off the surface (into the part on a rounded edge,
/// out of it at a burr or a bore's flared mouth) and their normals turn away from the model
/// far more than the noise on the surface proper does. So the rim ring goes while its median
/// tilt exceeds the region's median tilt by `PEEL_TILT_FACTOR`, ring by ring, until a ring
/// looks like the interior. Never below half the grown region, so a small face is not eaten
/// from the outside in. `stamp[v] == gen` marks membership on the way in and is cleared for
/// what is peeled; returns the vertices that stay.
fn peel_transition(g: &MeshGraph, stamp: &mut [i32], gen: i32, queue: Vec<u32>, align_at: impl Fn(u32) -> f64) -> Vec<u32> {
    let n = queue.len();
    if n < 60 {
        return queue;
    }
    let mut all: Vec<f32> = queue.iter().map(|&v| align_at(v) as f32).collect();
    sort_f32_like_js(&mut all);
    // The largest |cos| is the smallest tilt: the median of one is the median of the other.
    let noise_tilt = js_min(1.0, f64::from(all[n >> 1])).acos();
    let limit = js_max(PEEL_TILT_FACTOR * noise_tilt, noise_tilt + PEEL_TILT_FLOOR).cos();
    let keep_at_least = 60usize.max(n >> 1);

    let mut ring: Vec<u32> = Vec::new();
    for &v in &queue {
        if g.neighbours(v).iter().any(|&w| stamp[w as usize] != gen) {
            ring.push(v);
        }
    }

    let mut remaining = n;
    let mut k = 0;
    while k < MAX_PEEL_RINGS && !ring.is_empty() {
        if (remaining as i64 - ring.len() as i64) < keep_at_least as i64 {
            break;
        }
        let mut ring_align: Vec<f32> = ring.iter().map(|&v| align_at(v) as f32).collect();
        sort_f32_like_js(&mut ring_align);
        if f64::from(ring_align[ring.len() >> 1]) >= limit {
            break;
        }

        // gen is positive, so neither 0 nor −gen ever reads as a member.
        for &v in &ring {
            stamp[v as usize] = 0;
        }
        remaining -= ring.len();
        let mut next: Vec<u32> = Vec::new();
        for &v in &ring {
            for &nb in g.neighbours(v) {
                if stamp[nb as usize] == gen {
                    stamp[nb as usize] = -gen;
                    next.push(nb);
                }
            }
        }
        for &v in &next {
            stamp[v as usize] = gen;
        }
        ring = next;
        k += 1;
    }

    if remaining == n {
        return queue;
    }
    queue.into_iter().filter(|&v| stamp[v as usize] == gen).collect()
}

/// Grow the element's surface region from the seed while the model is refit every round.
/// Membership needs BOTH a distance inside the current noise band AND a surface normal
/// agreeing with the model's — the normal test is what stops the region from leaking across
/// an edge onto a neighbouring face, or onto the connecting rod at a sphere's neck. The walk
/// restarts from the seed each round with the improved model, capped at 3× the previous
/// region size so a bad early model cannot flood the whole mesh before the fit corrects it.
/// What the finished region holds of the edges' rounding is peeled off its rim at the end
/// (see `peel_transition`) — after the growing, not inside it, so the band that decides
/// where a surface ends stays the one tuned against GOM's selections and is not tightened by
/// its own trimming.
fn grow_region<T: Clone>(g: &MeshGraph, seeds: &[u32], init: T, init_sigma: f64, init_count: usize, spec: &GrowSpec<'_, T>, guide: Option<&FitGuide>) -> Option<GrowResult<T>> {
    let (positions, normals) = (&g.positions, &g.normals);
    // A side asks for the signed test: a vertex whose normal faces the wrong way — the
    // outside of a cup grown from its inside — does not join.
    let side_sign = match (guide.and_then(|g| g.side), spec.signed) {
        (Some(Side::Outside), Some(_)) => 1.0,
        (Some(Side::Inside), Some(_)) => -1.0,
        _ => 0.0,
    };
    let in_box_needed = guide.is_some_and(|g| g.within.is_some());
    let mut model = init;
    let mut sigma = js_max(init_sigma, 1e-6);
    let mut prev = init_count.max(50);
    let mut stamps = Stamps::new(g.vertex_count);
    // The last round's region as grown, with the generation that marks it and whether it hit
    // the cap.
    let mut last: Option<(Vec<u32>, i32, bool)> = None;

    for round in 0..30 {
        let band = (spec.band)(&model, sigma);
        let cap = 6000usize.max(prev * 3);
        let gen = stamps.next_gen();
        let mut queue: Vec<u32> = Vec::new();
        let mut hit_cap = false;

        for &seed in seeds {
            if (seed as usize) < g.vertex_count && !stamps.seen(seed) {
                stamps.mark(seed);
                queue.push(seed);
            }
        }

        let mut head = 0;
        'outer: while head < queue.len() {
            let v = queue[head];
            head += 1;
            for &nb in g.neighbours(v) {
                if stamps.seen(nb) {
                    continue;
                }
                let p = positions[nb as usize];
                let res = (spec.residual)(&model, p);
                // NaN (undefined residual) fails this test, as it should.
                if !(res.abs() <= band) {
                    continue;
                }
                let n = normals[nb as usize];
                let a = match spec.signed {
                    Some(signed) if side_sign != 0.0 => side_sign * signed(&model, p, n),
                    _ => (spec.align)(&model, p, n),
                };
                if a < spec.cos_max {
                    continue;
                }
                if in_box_needed && !guide.is_some_and(|g| in_box(g, p)) {
                    continue;
                }
                stamps.mark(nb);
                queue.push(nb);
                if queue.len() >= cap {
                    hit_cap = true;
                    break 'outer;
                }
            }
        }

        if queue.len() < 10 {
            return None;
        }

        let fit_idx: Vec<u32>;
        let idx: &[u32] = if queue.len() > FIT_SUBSAMPLE {
            let stride = queue.len() as f64 / FIT_SUBSAMPLE as f64;
            fit_idx = (0..FIT_SUBSAMPLE).map(|i| queue[(i as f64 * stride).floor() as usize]).collect();
            &fit_idx
        } else {
            &queue
        };
        let (fit_model, fit_sigma) = (spec.refit)(positions, idx, &model)?;

        let growth = queue.len().abs_diff(prev);
        model = fit_model;
        sigma = js_max(fit_sigma, 1e-6);
        let region_len = queue.len();
        last = Some((queue, gen, hit_cap));
        if !hit_cap && round >= 3 && (growth as f64) <= js_max(3.0, 0.002 * prev as f64) {
            break;
        }
        prev = region_len;
    }

    let (queue, gen, hit_cap) = last?;
    // A capped region stops mid-surface; its rim is the walk's front, not an edge, so there is
    // nothing to peel.
    let kept = if hit_cap {
        queue
    } else {
        let m = &model;
        peel_transition(g, &mut stamps.stamp, gen, queue, |v| (spec.align)(m, positions[v as usize], normals[v as usize]))
    };
    Some(GrowResult { region: kept, model, sigma })
}

/// The radial cosine at a point of a cylinder: the vertex normal against the unit direction
/// away from the axis; 0 on the axis.
fn cylinder_radial(c: &Cylinder, p: DVec3, n: DVec3) -> f64 {
    let (qx, qy, qz) = (p.x - c.point.x, p.y - c.point.y, p.z - c.point.z);
    let t = qx * c.axis.x + qy * c.axis.y + qz * c.axis.z;
    let (wx, wy, wz) = (qx - t * c.axis.x, qy - t * c.axis.y, qz - t * c.axis.z);
    let rho = (wx * wx + wy * wy + wz * wz).sqrt();
    if !(rho > 1e-12) {
        return 0.0;
    }
    (wx * n.x + wy * n.y + wz * n.z) / rho
}

/// Grow the spherical surface around the seed.
pub fn grow_sphere_region(g: &MeshGraph, seeds: &[u32], init: Sphere, init_sigma: f64, init_count: usize, guide: Option<&FitGuide>) -> Option<GrowResult<Sphere>> {
    let band = |s: &Sphere, sigma: f64| js_max(3.5 * sigma, 0.004 * s.r);
    let residual = |s: &Sphere, p: DVec3| {
        let d = ((p.x - s.center.x).powi(2) + (p.y - s.center.y).powi(2) + (p.z - s.center.z).powi(2)).sqrt();
        if d < 1e-12 {
            f64::NAN
        } else {
            d - s.r
        }
    };
    let align = |s: &Sphere, p: DVec3, n: DVec3| {
        let (dx, dy, dz) = (p.x - s.center.x, p.y - s.center.y, p.z - s.center.z);
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        ((dx * n.x + dy * n.y + dz * n.z) / d).abs()
    };
    let signed = |s: &Sphere, p: DVec3, n: DVec3| {
        let (dx, dy, dz) = (p.x - s.center.x, p.y - s.center.y, p.z - s.center.z);
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d > 1e-12 {
            (dx * n.x + dy * n.y + dz * n.z) / d
        } else {
            0.0
        }
    };
    let refit = |positions: &[DVec3], idx: &[u32], _: &Sphere| fit_sphere_clipped(positions, idx, 3.0, None).map(|f| (f.sphere, f.sigma));
    let spec = GrowSpec { cos_max: cos_sphere_max(), band: &band, residual: &residual, align: &align, signed: Some(&signed), refit: &refit };
    grow_region(g, seeds, init, init_sigma, init_count, &spec, guide)
}

/// Grow the cylindrical surface around the seed. The region is free to run the full length
/// of the cylinder — it is the radial band and the radial normal check that stop it at the
/// end faces.
pub fn grow_cylinder_region(g: &MeshGraph, seeds: &[u32], init: Cylinder, init_sigma: f64, init_count: usize, guide: Option<&FitGuide>) -> Option<GrowResult<Cylinder>> {
    let band = |c: &Cylinder, sigma: f64| js_max(3.5 * sigma, 0.004 * c.r);
    let residual = |c: &Cylinder, p: DVec3| cylinder_residual(c, p);
    let align = |c: &Cylinder, p: DVec3, n: DVec3| cylinder_radial(c, p, n).abs();
    let signed = |c: &Cylinder, p: DVec3, n: DVec3| cylinder_radial(c, p, n);
    let refit = |positions: &[DVec3], idx: &[u32], c: &Cylinder| fit_cylinder_clipped(positions, idx, *c, 3.0).map(|f| (f.cylinder, f.sigma));
    let spec = GrowSpec { cos_max: cos_cylinder_max(), band: &band, residual: &residual, align: &align, signed: Some(&signed), refit: &refit };
    grow_region(g, seeds, init, init_sigma, init_count, &spec, guide)
}

/// Grow the toroidal surface around the seed — a fillet's round, a bend. The band floor
/// hangs off the tube radius, the smaller of the two, since that is the curvature the scan
/// resolves the surface against.
pub fn grow_torus_region(g: &MeshGraph, seeds: &[u32], init: Torus, init_sigma: f64, init_count: usize, guide: Option<&FitGuide>) -> Option<GrowResult<Torus>> {
    let band = |t: &Torus, sigma: f64| js_max(3.5 * sigma, 0.004 * t.minor);
    let residual = |t: &Torus, p: DVec3| torus_residual(t, p);
    let align = |t: &Torus, p: DVec3, n: DVec3| torus_normal_align(t, p, n);
    let refit = |positions: &[DVec3], idx: &[u32], t: &Torus| fit_torus_clipped(positions, idx, 3.0, *t).map(|f| (f.torus, f.sigma));
    let spec = GrowSpec { cos_max: cos_cylinder_max(), band: &band, residual: &residual, align: &align, signed: None, refit: &refit };
    grow_region(g, seeds, init, init_sigma, init_count, &spec, guide)
}

/// Grow the conical surface around the seed. Same shape of spec as the cylinder's — the band
/// floor hangs off the anchor radius, which sits at the clicked patch and so tracks the size
/// of the surface actually under the region.
pub fn grow_cone_region(g: &MeshGraph, seeds: &[u32], init: Cone, init_sigma: f64, init_count: usize, guide: Option<&FitGuide>) -> Option<GrowResult<Cone>> {
    let band = |c: &Cone, sigma: f64| js_max(3.5 * sigma, 0.004 * c.r);
    let residual = |c: &Cone, p: DVec3| cone_residual(c, p);
    let align = |c: &Cone, p: DVec3, n: DVec3| cone_normal_align(c, p, n);
    let signed = |c: &Cone, p: DVec3, n: DVec3| cone_normal_signed(c, p, n);
    let refit = |positions: &[DVec3], idx: &[u32], c: &Cone| fit_cone_clipped(positions, idx, *c, 3.0).map(|f| (f.cone, f.sigma));
    let spec = GrowSpec { cos_max: cos_cone_max(), band: &band, residual: &residual, align: &align, signed: Some(&signed), refit: &refit };
    grow_region(g, seeds, init, init_sigma, init_count, &spec, guide)
}

/// Grow the flat surface around the seed. A plane has no size of its own, so the band floor
/// is tied to the model's overall scale rather than a radius.
pub fn grow_plane_region(g: &MeshGraph, seeds: &[u32], init: Plane, init_sigma: f64, init_count: usize, guide: Option<&FitGuide>) -> Option<GrowResult<Plane>> {
    let floor = 2e-4 * g.bbox_diag;
    let band = move |_: &Plane, sigma: f64| js_max(3.5 * sigma, floor);
    let residual = |p: &Plane, x: DVec3| plane_residual(p, x);
    let align = |p: &Plane, _: DVec3, n: DVec3| (p.normal.x * n.x + p.normal.y * n.y + p.normal.z * n.z).abs();
    let refit = |positions: &[DVec3], idx: &[u32], _: &Plane| fit_plane_clipped(positions, idx, 3.0).map(|f| (f.plane, f.sigma));
    let spec = GrowSpec { cos_max: cos_plane_max(), band: &band, residual: &residual, align: &align, signed: None, refit: &refit };
    grow_region(g, seeds, init, init_sigma, init_count, &spec, guide)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::test_meshes::{sample_part, SAMPLE_DEPTH, SAMPLE_HEIGHT, SAMPLE_LEG, SAMPLE_WIDTH};
    use crate::fit3d::testkit::graph_indexed;

    /// The sample L, flat in its own frame, as a graph with vertex normals.
    fn sample_graph() -> MeshGraph {
        graph_indexed(sample_part())
    }

    fn vertex_at(g: &MeshGraph, x: f64, y: f64, z: f64) -> u32 {
        g.positions.iter().position(|p| p.x == x && p.y == y && p.z == z).expect("a vertex there") as u32
    }

    #[test]
    fn takes_one_flat_face_of_a_tessellated_part_at_a_small_angle_with_its_edge_rows() {
        let g = sample_graph();
        // The middle of the L's end face: interior vertices share its normal exactly; the
        // vertices on its rim carry a normal averaged with the side faces' and stop the flood.
        let seed = vertex_at(&g, 6.0, 6.0, SAMPLE_DEPTH);
        let region = flood_by_normal(&g, seed, 1.0, None);
        let face = SAMPLE_WIDTH * SAMPLE_LEG + SAMPLE_LEG * (SAMPLE_HEIGHT - SAMPLE_LEG);
        // Every interior vertex of the face — the rim rows are the ones whose normal leans
        // toward a side, and they lie one step past the interior.
        assert!(region.len() as f64 > face * 0.7);
        for &v in &region {
            assert_eq!(g.positions[v as usize].z, SAMPLE_DEPTH);
        }
    }

    #[test]
    fn crosses_a_right_angle_only_when_the_angle_allows_it() {
        let g = sample_graph();
        let seed = vertex_at(&g, 6.0, 6.0, SAMPLE_DEPTH);
        let wide = flood_by_normal(&g, seed, 60.0, None);
        // Rim normals lean 45° between two faces (or more at a corner), so 60° lets the flood
        // over every edge: the whole part.
        assert_eq!(wide.len(), g.vertex_count);
        let capped = flood_by_normal(&g, seed, 60.0, Some(100));
        assert_eq!(capped.len(), 100);
        // ScanRuler's seed −1, a vertex that is not there: an id past the mesh.
        assert_eq!(flood_by_normal(&g, u32::MAX, 60.0, None).len(), 0);
    }

    #[test]
    fn a_seed_standing_alone_on_another_surface_is_stranded() {
        let g = sample_graph();
        // A region of the end face's interior, plus a lone seed on the far end face.
        let seed = vertex_at(&g, 6.0, 6.0, SAMPLE_DEPTH);
        let mut region = flood_by_normal(&g, seed, 1.0, None);
        let far = vertex_at(&g, 6.0, 6.0, 0.0);
        region.push(far);
        let stranded = stranded_seeds(&g, &region, &[seed, far]);
        assert_eq!(stranded.into_iter().collect::<Vec<_>>(), vec![far]);
        let patch = collect_patch(&g, &[seed], 50);
        assert_eq!(patch.len(), 50);
        assert_eq!(patch[0], seed);
    }
}
