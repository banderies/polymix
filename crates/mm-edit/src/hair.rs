//! Grow Hair (Meshmixer's Select > Edit > Grow Hair): strands rooted on the selected faces,
//! grown along the surface normal, bent by gravity according to their stiffness (which
//! varies from root to tip), curled into a helix, optionally kept out of the object, and
//! finally turned into tapered tubes.
//!
//! Three ways to root them ([`HairMode`]): a fixed number at random points, an even spread
//! at a density per unit area (Poisson-disc: no two roots on the same side of the surface
//! closer than [`fill_spacing`]), or
//! hair patches: a few guide strands grown as above, the rest following the guides around
//! them and gathered toward the nearest one's tip in clumps.

use crate::tube_mesh;
use glam::DVec3;
use mm_core::hash::FastMap;
use mm_core::{Bvh, Mesh};
use rayon::prelude::*;

/// How strands are rooted on the selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HairMode {
    /// `count` strands at random points of the selection.
    #[default]
    Count,
    /// Strands spread evenly over the selection, `density` per unit area (at most
    /// [`MAX_HAIRS`]); no two roots closer than [`fill_spacing`] (unless they face apart,
    /// on the two sides of a thin part).
    FillDensity,
    /// `clumps` evenly spread guide strands; the other `count - clumps` strands follow the
    /// guides around them and are drawn toward the nearest guide's tip by `clumping`.
    HairPatch,
}

/// Most strands Fill Density grows.
pub const MAX_HAIRS: usize = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct HairParams {
    /// How strands are rooted.
    pub mode: HairMode,
    /// Strands (Count and Hair Patch modes; in Hair Patch, guides included).
    pub count: usize,
    /// Fill Density: strands per unit area (per mm² for models in millimetres).
    pub density: f64,
    /// Hair Patch: guide strands, one per clump.
    pub clumps: usize,
    /// Hair Patch, 0..1: how far each strand is drawn toward its guide, growing from the
    /// root to the tip (1 = the tips meet the guide's tip).
    pub clumping: f64,
    /// Points along each strand.
    pub points: usize,
    pub length: f64,
    /// Helix radius (0 = straight).
    pub curl_radius: f64,
    /// Length of one helix turn.
    pub curl_pitch: f64,
    /// 0..1 at the root and at the tip: how much a strand resists gravity.
    pub hair_stiffness: f64,
    pub tip_stiffness: f64,
    /// 0..1.
    pub gravity: f64,
    /// Direction gravity pulls.
    #[serde(skip)]
    #[schemars(skip)]
    pub down: DVec3,
    /// Push strands out of the object.
    pub collisions: bool,
    pub base_radius: f64,
    pub tip_radius: f64,
    pub seed: u64,
}

impl Default for HairParams {
    fn default() -> Self {
        Self {
            mode: HairMode::Count,
            count: 200,
            density: 1.0,
            clumps: 12,
            clumping: 0.7,
            points: 12,
            length: 10.0,
            curl_radius: 0.0,
            curl_pitch: 5.0,
            hair_stiffness: 0.8,
            tip_stiffness: 0.2,
            gravity: 0.3,
            down: -DVec3::Y,
            collisions: true,
            base_radius: 0.3,
            tip_radius: 0.1,
            seed: 7,
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Smallest distance between Fill Density roots: 0.6 / √density, a little over half the
/// mean spacing, which an even (Poisson-disc) spread reaches easily.
pub fn fill_spacing(density: f64) -> f64 {
    0.6 / density.max(1e-300).sqrt()
}

/// Strands Fill Density grows on `area` (at most [`MAX_HAIRS`]).
pub fn fill_count(area: f64, density: f64) -> usize {
    let n = (area * density).round();
    if n.is_finite() && n > 0.0 {
        (n as usize).min(MAX_HAIRS)
    } else {
        0
    }
}

/// Total area of the (live) triangles `tris`.
pub fn selection_area(mesh: &Mesh, tris: &[u32]) -> f64 {
    tris.iter().filter(|&&t| mesh.is_triangle(t)).map(|&t| mesh.tri_area(t)).sum()
}

/// A point of the surface and its smoothed normal.
#[derive(Clone, Copy)]
struct Root {
    p: DVec3,
    n: DVec3,
}

/// Uniformly random points on a set of triangles.
struct AreaSampler {
    tris: Vec<u32>,
    cum: Vec<f64>,
    total: f64,
}

impl AreaSampler {
    fn new(mesh: &Mesh, tris: &[u32]) -> Self {
        let tris: Vec<u32> = tris.iter().copied().filter(|&t| mesh.is_triangle(t) && mesh.tri_area(t) > 0.0).collect();
        let mut cum = Vec::with_capacity(tris.len());
        let mut total = 0.0;
        for &t in &tris {
            total += mesh.tri_area(t);
            cum.push(total);
        }
        Self { tris, cum, total }
    }

    fn sample(&self, mesh: &Mesh, rng: &mut Rng) -> Root {
        let t = self.tris[self.cum.partition_point(|&c| c < rng.next() * self.total).min(self.tris.len() - 1)];
        let (mut s1, mut s2) = (rng.next(), rng.next());
        if s1 + s2 > 1.0 {
            s1 = 1.0 - s1;
            s2 = 1.0 - s2;
        }
        let b = DVec3::new(1.0 - s1 - s2, s1, s2);
        let [a, c, d] = mesh.tri_positions(t);
        Root { p: a * b.x + c * b.y + d * b.z, n: normal_at(mesh, t, b) }
    }
}

/// Up to `n` random points of the surface, no two facing the same way closer than `r` (dart
/// throwing against a hash grid; gives up after 40 misses per point on average).
fn poisson_roots(mesh: &Mesh, area: &AreaSampler, n: usize, r: f64, rng: &mut Rng) -> Vec<Root> {
    let mut out: Vec<Root> = Vec::with_capacity(n);
    // Looked up by cell only, never walked, so the hasher cannot change the result.
    let mut grid: FastMap<[i64; 3], Vec<usize>> = FastMap::default();
    let cell = |p: DVec3| (p / r).floor().to_array().map(|x| x as i64);
    let mut tries = 0;
    while out.len() < n && tries < 40 * n + 100 {
        tries += 1;
        let q = area.sample(mesh, rng);
        let [x, y, z] = cell(q.p);
        let mut free = true;
        'scan: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &k in grid.get(&[x + dx, y + dy, z + dz]).into_iter().flatten() {
                        // Roots facing apart (the two sides of a thin part) do not crowd
                        // each other.
                        if out[k].p.distance_squared(q.p) < r * r && out[k].n.dot(q.n) > 0.0 {
                            free = false;
                            break 'scan;
                        }
                    }
                }
            }
        }
        if free {
            grid.entry([x, y, z]).or_default().push(out.len());
            out.push(q);
        }
    }
    out
}

/// Smoothly interpolated normal at a point of triangle `t` (barycentric `b`).
fn normal_at(m: &Mesh, t: u32, b: DVec3) -> DVec3 {
    let tv = m.triangle(t);
    (m.vertex_normal(tv[0]) * b.x + m.vertex_normal(tv[1]) * b.y + m.vertex_normal(tv[2]) * b.z).normalize_or(m.tri_normal(t))
}

/// Push a strand point out of the object to `clearance` above the surface; returns whether
/// it moved.
fn keep_out(mesh: &Mesh, bvh: &Bvh, q: &mut DVec3, clearance: f64, reach: f64) -> bool {
    if let Some(h) = bvh.nearest(mesh, *q, reach) {
        let nn = normal_at(mesh, h.tid, h.bary);
        if (*q - h.point).dot(nn) < clearance {
            *q = h.point + nn * clearance;
            return true;
        }
    }
    false
}

/// Strands grown together by one parallel task: a strand without collisions costs only
/// microseconds, too little to hand out one at a time.
const STRANDS_PER_TASK: usize = 16;

/// A strand's random start angle for its curl: the one draw it makes.
fn phase(rng: &mut Rng) -> f64 {
    rng.next() * std::f64::consts::TAU
}

/// Strands from `roots`, each with its curl's start angle. The random numbers are all drawn
/// beforehand, in the order the strands used to draw them one after another, so the strands
/// grow in parallel and come out the very same, in the same order.
fn grow_all(mesh: &Mesh, bvh: &Bvh, roots: &[(Root, f64)], p: &HairParams) -> Vec<Vec<DVec3>> {
    roots.par_iter().with_min_len(STRANDS_PER_TASK).map(|(r, phase)| grow(mesh, bvh, r, p, *phase)).collect()
}

/// The roots with a start angle each, drawn in their order.
fn with_phases(roots: &[Root], rng: &mut Rng) -> Vec<(Root, f64)> {
    roots.iter().map(|r| (*r, phase(rng))).collect()
}

/// One strand from `root`, its curl starting at angle `phase`.
fn grow(mesh: &Mesh, bvh: &Bvh, root: &Root, p: &HairParams, phase: f64) -> Vec<DVec3> {
    let down = p.down.normalize_or(-DVec3::Y);
    let seg = p.length / (p.points - 1) as f64;
    let clearance = p.base_radius.max(p.tip_radius) * 1.5;
    let n = root.n;
    // Curl frame around the root normal, with a random start angle.
    let (u, v) = mm_core::geom::orthonormal_basis(n);
    let mut dir = n;
    let mut spine = root.p;
    let mut pts = vec![root.p];
    for k in 1..p.points {
        let s = k as f64 / (p.points - 1) as f64;
        let stiff = p.hair_stiffness + (p.tip_stiffness - p.hair_stiffness) * s;
        // Gravity turns the strand toward `down` at every step, more where it is soft
        // (at full gravity and no stiffness, up to a full turn over the strand).
        let max_turn = std::f64::consts::TAU / (p.points - 1) as f64;
        let turn = (p.gravity.clamp(0.0, 1.0) * (1.0 - stiff.clamp(0.0, 1.0)) * max_turn).min(dir.angle_between(down));
        if turn > 0.0 {
            let axis = dir.cross(down);
            let axis = if axis.length_squared() > 1e-18 { axis.normalize() } else { mm_core::geom::orthonormal_basis(dir).0 };
            dir = (glam::DQuat::from_axis_angle(axis, turn) * dir).normalize();
        }
        spine += dir * seg;
        let mut q = spine;
        if p.curl_radius > 0.0 && p.curl_pitch > 0.0 {
            let a = phase + std::f64::consts::TAU * (s * p.length) / p.curl_pitch;
            // The helix grows in from the root so the strand starts on the surface.
            q += (u * a.cos() + v * a.sin() - u * phase.cos() - v * phase.sin()) * (p.curl_radius * s.min(1.0));
        }
        if p.collisions && keep_out(mesh, bvh, &mut q, clearance, clearance * 4.0 + p.length) {
            spine = q;
        }
        pts.push(q);
    }
    pts
}

/// Strand polylines rooted on `tris` of `mesh`. In Hair Patch mode the guide strands come
/// first (`clumps` of them, fewer if `count` is smaller or the selection too small to
/// spread them).
pub fn hair_curves(mesh: &Mesh, bvh: &Bvh, tris: &[u32], p: &HairParams) -> Vec<Vec<DVec3>> {
    let area = AreaSampler::new(mesh, tris);
    if area.tris.is_empty() || p.points < 2 || p.length.is_nan() || p.length <= 0.0 {
        return Vec::new();
    }
    let mut rng = Rng(p.seed.max(1));
    match p.mode {
        HairMode::Count => {
            // Each strand's root, then its start angle.
            let roots: Vec<(Root, f64)> = (0..p.count)
                .map(|_| {
                    let r = area.sample(mesh, &mut rng);
                    (r, phase(&mut rng))
                })
                .collect();
            grow_all(mesh, bvh, &roots, p)
        }
        HairMode::FillDensity => {
            let n = fill_count(area.total, p.density);
            let roots = poisson_roots(mesh, &area, n, fill_spacing(p.density), &mut rng);
            grow_all(mesh, bvh, &with_phases(&roots, &mut rng), p)
        }
        HairMode::HairPatch => patches(mesh, bvh, &area, p, &mut rng),
    }
}

/// Hair Patch strands: the guides, then the strands following them.
fn patches(mesh: &Mesh, bvh: &Bvh, area: &AreaSampler, p: &HairParams, rng: &mut Rng) -> Vec<Vec<DVec3>> {
    if p.count == 0 {
        return Vec::new();
    }
    let n_guides = p.clumps.clamp(1, p.count);
    // Guides spread evenly: no two closer than half their mean spacing.
    let spacing = 0.5 * (area.total / n_guides as f64).sqrt();
    let roots = poisson_roots(mesh, area, n_guides, spacing, rng);
    let guides = grow_all(mesh, bvh, &with_phases(&roots, rng), p);
    // The followers' roots are drawn in order; their strands draw nothing, so they are made in
    // parallel.
    let followers: Vec<Root> = (guides.len()..p.count).map(|_| area.sample(mesh, rng)).collect();
    let strands: Vec<Vec<DVec3>> = followers.par_iter().with_min_len(STRANDS_PER_TASK).map(|r| follow(mesh, bvh, &roots, &guides, p, r)).collect();
    let mut out = guides;
    out.extend(strands);
    out
}

/// A Hair Patch strand rooted at `r`, following the guides (grown from `roots`) around it.
fn follow(mesh: &Mesh, bvh: &Bvh, roots: &[Root], guides: &[Vec<DVec3>], p: &HairParams, r: &Root) -> Vec<DVec3> {
    let clearance = p.base_radius.max(p.tip_radius) * 1.5;
    let pull = p.clumping.clamp(0.0, 1.0);
    let root = r.p;
    // The strand's shape: the guides around its root, blended by inverse squared
    // distance, so the hair flows on smoothly from one patch to the next. Guides on the
    // far side of a thin part (facing the other way) are not "around" it.
    let facing = |i: &usize| roots[*i].n.dot(r.n) > 0.0;
    let pool: Vec<usize> = if (0..guides.len()).any(|i| facing(&i)) { (0..guides.len()).filter(facing).collect() } else { (0..guides.len()).collect() };
    let mut near: Vec<(f64, usize)> = pool.into_iter().map(|i| (guides[i][0].distance_squared(root), i)).collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    near.truncate(3);
    let w: Vec<f64> = near.iter().map(|&(d2, _)| 1.0 / (d2 + 1e-12 * p.length * p.length)).collect();
    let wsum: f64 = w.iter().sum();
    let own = &guides[near[0].1];
    (0..p.points)
        .map(|k| {
            let s = k as f64 / (p.points - 1) as f64;
            let shape: DVec3 = near.iter().zip(&w).map(|(&(_, i), &wi)| (guides[i][k] - guides[i][0]) * (wi / wsum)).sum();
            // Clumping: drawn toward its own guide, the more the nearer the tip.
            let mut q = (root + shape).lerp(own[k], pull * s);
            if p.collisions && k > 0 {
                keep_out(mesh, bvh, &mut q, clearance, clearance * 4.0 + p.length);
            }
            q
        })
        .collect()
}

/// Tapered tubes along the strands, as one mesh.
pub fn hair_mesh(curves: &[Vec<DVec3>], base_radius: f64, tip_radius: f64, slices: usize) -> Mesh {
    let mut m = Mesh::new();
    let tip = tip_radius.max(base_radius * 0.05).max(1e-9);
    for c in curves.iter().filter(|c| c.len() >= 2) {
        m.append_mesh(&tube_mesh(c, base_radius.max(1e-9), tip, slices), 0);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn hair_grows_out_of_the_selection_and_droops() {
        let ball = primitives::icosphere(10.0, 3);
        let bvh = Bvh::build(&ball);
        let top: Vec<u32> = ball.triangle_ids().filter(|&t| ball.tri_centroid(t).z > 8.0).collect();
        let p = HairParams { count: 50, length: 6.0, gravity: 0.0, down: -DVec3::Z, ..Default::default() };
        let straight = hair_curves(&ball, &bvh, &top, &p);
        assert_eq!(straight.len(), 50);
        for c in &straight {
            assert_eq!(c.len(), p.points);
            assert!(c[0].z > 7.5, "rooted on the selection");
            // Without gravity or curl the strand runs along the normal: tip ≈ 16 from the centre.
            assert!((c[c.len() - 1].length() - 16.0).abs() < 0.5, "{}", c[c.len() - 1].length());
        }
        // Gravity (toward -z, sideways for strands near the top) lowers the tips.
        let droop = hair_curves(&ball, &bvh, &top, &HairParams { gravity: 1.0, ..p });
        let mean_z = |cs: &[Vec<DVec3>]| cs.iter().map(|c| c[c.len() - 1].z).sum::<f64>() / cs.len() as f64;
        assert!(mean_z(&droop) < mean_z(&straight) - 1.0);
        // Collisions keep every point (past the root) outside the ball's faceted surface,
        // which sags about 0.01 inside the radius.
        let side: Vec<u32> = ball.triangle_ids().filter(|&t| ball.tri_centroid(t).x > 8.0).collect();
        // Gravity pointing back into the ball.
        let soft = HairParams { gravity: 1.0, hair_stiffness: 0.0, tip_stiffness: 0.0, length: 25.0, down: -DVec3::X, ..p };
        let hang = hair_curves(&ball, &bvh, &side, &HairParams { collisions: true, ..soft });
        assert!(hang.iter().flat_map(|c| &c[1..]).all(|q| q.length() > 9.98), "{:?}", hang.iter().flat_map(|c| &c[1..]).map(|q| q.length()).fold(f64::MAX, f64::min));
        let through = hair_curves(&ball, &bvh, &side, &HairParams { collisions: false, ..soft });
        assert!(through.iter().flatten().any(|q| q.length() < 9.0), "without collisions strands sag into the ball");
        // Tubes.
        let m = hair_mesh(&straight, 0.3, 0.1, 6);
        m.check_validity().unwrap();
        assert!(m.is_closed());
        assert_eq!(mm_core::topology::connected_components(&m).len(), 50);
    }

    /// The strands made one after another, each drawing its start angle when it starts
    /// growing and each Hair Patch follower drawing its root when it is made: the order the
    /// random numbers were drawn in before the strands grew in parallel.
    fn one_by_one(mesh: &Mesh, bvh: &Bvh, tris: &[u32], p: &HairParams) -> Vec<Vec<DVec3>> {
        let area = AreaSampler::new(mesh, tris);
        if area.tris.is_empty() || p.points < 2 || p.length.is_nan() || p.length <= 0.0 {
            return Vec::new();
        }
        let mut rng = Rng(p.seed.max(1));
        let grow_now = |r: &Root, rng: &mut Rng| {
            let a = phase(rng);
            grow(mesh, bvh, r, p, a)
        };
        match p.mode {
            HairMode::Count => (0..p.count)
                .map(|_| {
                    let r = area.sample(mesh, &mut rng);
                    grow_now(&r, &mut rng)
                })
                .collect(),
            HairMode::FillDensity => {
                let roots = poisson_roots(mesh, &area, fill_count(area.total, p.density), fill_spacing(p.density), &mut rng);
                roots.iter().map(|r| grow_now(r, &mut rng)).collect()
            }
            HairMode::HairPatch => {
                if p.count == 0 {
                    return Vec::new();
                }
                let n_guides = p.clumps.clamp(1, p.count);
                let roots = poisson_roots(mesh, &area, n_guides, 0.5 * (area.total / n_guides as f64).sqrt(), &mut rng);
                let guides: Vec<Vec<DVec3>> = roots.iter().map(|r| grow_now(r, &mut rng)).collect();
                let mut out = guides.clone();
                for _ in guides.len()..p.count {
                    let r = area.sample(mesh, &mut rng);
                    out.push(follow(mesh, bvh, &roots, &guides, p, &r));
                }
                out
            }
        }
    }

    #[test]
    fn strands_grown_in_parallel_are_the_ones_grown_one_by_one() {
        let ball = primitives::icosphere(10.0, 3);
        let bvh = Bvh::build(&ball);
        let cap: Vec<u32> = ball.triangle_ids().filter(|&t| ball.tri_centroid(t).z > 2.0).collect();
        let base = HairParams { count: 300, length: 8.0, curl_radius: 0.6, curl_pitch: 3.0, gravity: 0.6, down: -DVec3::X, density: 0.5, clumps: 9, seed: 99, ..Default::default() };
        for mode in [HairMode::Count, HairMode::FillDensity, HairMode::HairPatch] {
            for collisions in [true, false] {
                let p = HairParams { mode, collisions, ..base };
                let (a, b) = (hair_curves(&ball, &bvh, &cap, &p), one_by_one(&ball, &bvh, &cap, &p));
                assert!(a.len() > 50, "{mode:?}");
                let bits = |c: &[Vec<DVec3>]| c.iter().flatten().flat_map(|q| q.to_array().map(f64::to_bits)).collect::<Vec<u64>>();
                assert_eq!(bits(&a), bits(&b), "{mode:?}, collisions {collisions}");
            }
        }
    }

    #[test]
    fn curls_wind_around_the_strand() {
        let plane = primitives::plane_grid(10.0, 10.0, 4, 4);
        let bvh = Bvh::build(&plane);
        let all: Vec<u32> = plane.triangle_ids().collect();
        let p = HairParams { count: 5, points: 60, length: 20.0, curl_radius: 1.0, curl_pitch: 5.0, gravity: 0.0, collisions: false, ..Default::default() };
        for c in hair_curves(&plane, &bvh, &all, &p) {
            // Four turns: the horizontal offset from the root reaches about twice the radius.
            let reach = c.iter().map(|q| (*q - c[0]).truncate().length()).fold(0.0, f64::max);
            assert!(reach > 1.5 && reach < 2.2, "{reach}");
            assert!((c[c.len() - 1].z - 20.0).abs() < 1e-6);
        }
    }

    #[test]
    fn fill_density_spreads_hair_evenly() {
        // On a plane and on a curved cap.
        let plane = primitives::plane_grid(20.0, 20.0, 10, 10);
        let ball = primitives::icosphere(10.0, 4);
        for (mesh, density) in [(&plane, 2.0), (&ball, 0.7)] {
            let bvh = Bvh::build(mesh);
            let tris: Vec<u32> = mesh.triangle_ids().filter(|&t| mesh.tri_centroid(t).z > -2.0).collect();
            let area = selection_area(mesh, &tris);
            let p = HairParams { mode: HairMode::FillDensity, density, gravity: 0.0, collisions: false, ..Default::default() };
            let curves = hair_curves(mesh, &bvh, &tris, &p);
            let want = area * density;
            assert!((curves.len() as f64 - want).abs() <= 0.02 * want, "{} strands for {want}", curves.len());
            // No two roots closer than the spacing.
            let r = fill_spacing(density);
            let roots: Vec<DVec3> = curves.iter().map(|c| c[0]).collect();
            let closest = (0..roots.len()).flat_map(|i| (i + 1..roots.len()).map(move |j| (i, j))).map(|(i, j)| roots[i].distance(roots[j])).fold(f64::MAX, f64::min);
            assert!(closest >= r, "{closest} < {r}");
            // Rooted on the selection.
            assert!(roots.iter().all(|q| q.z > -2.5));
        }
        // The count follows the density.
        let bvh = Bvh::build(&plane);
        let all: Vec<u32> = plane.triangle_ids().collect();
        let n = |density: f64| hair_curves(&plane, &bvh, &all, &HairParams { mode: HairMode::FillDensity, density, collisions: false, ..Default::default() }).len();
        assert_eq!(n(0.25), 100);
        assert_eq!(n(1000.0), MAX_HAIRS);
    }

    #[test]
    fn hair_patches_gather_into_clumps() {
        let plane = primitives::plane_grid(20.0, 20.0, 10, 10);
        let bvh = Bvh::build(&plane);
        let all: Vec<u32> = plane.triangle_ids().collect();
        let base = HairParams { mode: HairMode::HairPatch, count: 400, clumps: 10, length: 6.0, gravity: 0.5, collisions: false, ..Default::default() };
        let tip = |c: &Vec<DVec3>| c[c.len() - 1];
        // Each strand's guide is the one whose root is nearest its own root.
        let spread = |clumping: f64| -> (f64, f64, usize) {
            let curves = hair_curves(&plane, &bvh, &all, &HairParams { clumping, ..base });
            assert_eq!(curves.len(), 400);
            let guides = &curves[..10];
            let (mut at_root, mut at_tip) = (0.0, 0.0);
            let mut tips = Vec::new();
            for c in &curves[10..] {
                assert!(c[0].z.abs() < 1e-12, "rooted on the plane");
                let g = guides.iter().min_by(|a, b| a[0].distance(c[0]).total_cmp(&b[0].distance(c[0]))).unwrap();
                at_root += g[0].distance(c[0]);
                at_tip += tip(g).distance(tip(c));
                let t = tip(c);
                if !tips.iter().any(|q: &DVec3| q.distance(t) < 1e-6) {
                    tips.push(t);
                }
            }
            (at_root / 390.0, at_tip / 390.0, tips.len())
        };
        // Unclumped, the strands keep their spacing up to the tips; fully clumped, every tip
        // meets its guide's, so there are as many tufts as guides.
        let (root0, tip0, _) = spread(0.0);
        assert!((tip0 - root0).abs() < 0.05 * root0, "{tip0} vs {root0}");
        let (root1, tip1, tufts) = spread(1.0);
        assert!(tip1 < 1e-9 * root1, "{tip1}");
        assert_eq!(tufts, 10);
        let (root_half, tip_half, _) = spread(0.6);
        assert!((tip_half / root_half - 0.4).abs() < 0.05, "{}", tip_half / root_half);
        // Guides are spread out, not bunched.
        let curves = hair_curves(&plane, &bvh, &all, &base);
        let g: Vec<DVec3> = curves[..10].iter().map(|c| c[0]).collect();
        let closest = (0..10).flat_map(|i| (i + 1..10).map(move |j| (i, j))).map(|(i, j)| g[i].distance(g[j])).fold(f64::MAX, f64::min);
        assert!(closest >= 0.5 * (400.0f64 / 10.0).sqrt(), "{closest}");
    }

    #[test]
    fn the_two_sides_of_a_thin_part_are_haired_apart() {
        // A plate 0.1 thick, hair on its top and bottom.
        let mut slab = primitives::cube(10.0, 8);
        slab.transform(|q| DVec3::new(q.x, q.y, q.z * 0.01));
        let bvh = Bvh::build(&slab);
        let faces: Vec<u32> = slab.triangle_ids().filter(|&t| slab.tri_normal(t).z.abs() > 0.9).collect();
        let area = selection_area(&slab, &faces);
        let p = HairParams { mode: HairMode::FillDensity, density: 4.0, gravity: 0.0, collisions: false, ..Default::default() };
        let n = hair_curves(&slab, &bvh, &faces, &p).len() as f64;
        assert!((n - area * 4.0).abs() <= 0.02 * area * 4.0, "{n} for {}", area * 4.0);
        // Strands follow guides on their own side.
        let patch = HairParams { mode: HairMode::HairPatch, count: 300, clumps: 8, clumping: 0.5, ..p };
        for c in hair_curves(&slab, &bvh, &faces, &patch) {
            let up = c[0].z > 0.0;
            assert_eq!(c[c.len() - 1].z > c[0].z, up, "{:?}", c[0]);
        }
    }
}
