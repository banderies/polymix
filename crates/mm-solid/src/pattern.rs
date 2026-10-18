//! Make Pattern (Meshmixer's Edit > Make Pattern): build a pattern of tubes and balls from
//! the object and mesh it as one solid.
//!
//! The elements — capsules (tubes) and balls — form an implicit field: the smooth minimum
//! of their signed distances, so joints get a small fillet. Edge patterns use the field on
//! its own (optionally clipped to the object); tiled, lattice and random patterns combine it
//! with the object's signed distance, subtracting the elements (perforating the object) or
//! intersecting with them. The result is extracted from the sparse grid like Make Solid.

use crate::field::{BoolField, BoolOp, MeshSdf, DEFAULT_BAND_CELLS};
use crate::grid::{build_field_with, FieldSource, GridSpec};
use crate::progress::Prog;
use crate::solid::{compact, extract_and_clean};
use crate::{Progress, Result, SolidError};
use glam::DVec3;
use mm_core::{Aabb, Mesh, INVALID};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatternType {
    /// A tube along every mesh edge.
    Edges,
    /// Tubes joining the centres of neighbouring triangles.
    DualEdges,
    /// [`PatternType::Edges`] after Delaunay edge flips (better proportioned cells).
    DelaunayEdges,
    /// [`PatternType::DualEdges`] after Delaunay edge flips.
    DelaunayDualEdges,
    /// Parallel tubes through the object, on a 2D tiling across its thinnest direction.
    TiledTubes,
    /// Balls on a 3D tiling.
    TiledSpheres,
    /// Tubes along x, y and z on a cubic grid.
    Lattice,
    /// Balls at random points on the surface.
    RandomSurface,
    /// Balls at random points inside the object.
    RandomVolume,
    /// Tubes along face-group borders and open boundaries.
    GroupBorders,
}

impl PatternType {
    /// Patterns made from the object's own edges (the rest fill its volume).
    pub fn from_edges(self) -> bool {
        matches!(self, PatternType::Edges | PatternType::DualEdges | PatternType::DelaunayEdges | PatternType::DelaunayDualEdges | PatternType::GroupBorders)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tiling {
    Regular,
    Hex,
}

/// How volume patterns combine with the object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Composition {
    /// Object minus the elements.
    Subtract,
    /// The elements, kept inside the object.
    Intersect,
}

/// Element size varying across the object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gradient {
    None,
    /// Along the object's longest axis, `start` at its minimum end.
    Line,
    /// From the object's centre outward.
    Sphere,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatternParams {
    pub kind: PatternType,
    pub tiling: Tiling,
    /// Tube / ball diameter.
    pub dimension: f64,
    /// Distance between neighbouring elements (tiled, lattice and random patterns).
    pub spacing: f64,
    pub composition: Composition,
    /// Edge patterns: keep only the part inside the object.
    pub clip_to_surface: bool,
    /// Fillet radius at joints, as a fraction of the element radius (0 = sharp).
    pub smoothing: f64,
    /// Grid cells across an element (higher = finer result).
    pub resolution: f64,
    pub gradient: Gradient,
    pub start_scale: f64,
    pub end_scale: f64,
    pub seed: u64,
}

impl Default for PatternParams {
    fn default() -> Self {
        Self {
            kind: PatternType::TiledTubes,
            tiling: Tiling::Hex,
            dimension: 1.0,
            spacing: 1.5,
            composition: Composition::Subtract,
            clip_to_surface: true,
            smoothing: 0.25,
            resolution: 4.0,
            gradient: Gradient::None,
            start_scale: 0.5,
            end_scale: 2.0,
            seed: 1,
        }
    }
}

/// A capsule (`a == b` for a ball).
#[derive(Clone, Copy, Debug)]
struct Element {
    a: DVec3,
    b: DVec3,
    r: f64,
}

impl Element {
    fn distance(&self, p: DVec3) -> f64 {
        let ab = self.b - self.a;
        let l2 = ab.length_squared();
        let t = if l2 > 0.0 { ((p - self.a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
        p.distance(self.a + ab * t) - self.r
    }
    fn bounds(&self, pad: f64) -> (DVec3, DVec3) {
        let e = DVec3::splat(self.r + pad);
        (self.a.min(self.b) - e, self.a.max(self.b) + e)
    }
}

/// Smooth minimum (polynomial, blend width `k`).
fn smin(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0);
    a.min(b) - h * h / (4.0 * k)
}

/// The elements' field, with a bucket grid so each sample only looks at nearby elements.
struct ElementField {
    elements: Vec<Element>,
    buckets: HashMap<[i64; 3], Vec<u32>>,
    bucket: f64,
    far: f64,
    blend: f64,
}

impl ElementField {
    fn new(elements: Vec<Element>, far: f64, blend: f64) -> Self {
        let rmax = elements.iter().map(|e| e.r).fold(0.0, f64::max);
        let bucket = (4.0 * (rmax + far + blend)).max(1e-9);
        let mut buckets: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        let cell = |p: DVec3| [(p.x / bucket).floor() as i64, (p.y / bucket).floor() as i64, (p.z / bucket).floor() as i64];
        for (i, e) in elements.iter().enumerate() {
            let (lo, hi) = e.bounds(far + blend);
            let (lo, hi) = (cell(lo), cell(hi));
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        buckets.entry([x, y, z]).or_default().push(i as u32);
                    }
                }
            }
        }
        Self { elements, buckets, bucket, far, blend }
    }

    fn key(&self, p: DVec3) -> [i64; 3] {
        [(p.x / self.bucket).floor() as i64, (p.y / self.bucket).floor() as i64, (p.z / self.bucket).floor() as i64]
    }
}

impl FieldSource for ElementField {
    fn eval(&self, p: DVec3) -> f64 {
        let Some(list) = self.buckets.get(&self.key(p)) else { return self.far };
        let mut v = self.far + self.blend;
        for &i in list {
            v = smin(v, self.elements[i as usize].distance(p), self.blend);
        }
        v.clamp(-self.far, self.far)
    }
    fn far(&self) -> f64 {
        self.far
    }
    fn seed_bricks(&self, grid: &GridSpec) -> Vec<[u32; 3]> {
        let mut out = Vec::new();
        let pad = self.far + self.blend + grid.cell_size;
        for e in &self.elements {
            let (lo, hi) = e.bounds(pad);
            let (lo, hi) = (grid.brick_of(lo), grid.brick_of(hi));
            for x in lo[0].max(0)..=hi[0].min(grid.bricks[0] as i64 - 1) {
                for y in lo[1].max(0)..=hi[1].min(grid.bricks[1] as i64 - 1) {
                    for z in lo[2].max(0)..=hi[2].min(grid.bricks[2] as i64 - 1) {
                        out.push([x as u32, y as u32, z as u32]);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
    fn uniform_sign(&self, c: DVec3, r: f64) -> Option<bool> {
        // Outside when no element can reach the ball.
        let (lo, hi) = (self.key(c - DVec3::splat(r)), self.key(c + DVec3::splat(r)));
        let n: i64 = (0..3).map(|a| hi[a] - lo[a] + 1).product();
        if n > 4096 {
            return None;
        }
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    if self.buckets.contains_key(&[x, y, z]) {
                        return None;
                    }
                }
            }
        }
        Some(false)
    }
}

/// Element size factor at `p` for the gradient.
fn gradient_scale(p: &PatternParams, b: &Aabb, at: DVec3) -> f64 {
    let t = match p.gradient {
        Gradient::None => return 1.0,
        Gradient::Line => {
            let e = b.extents();
            let a = if e.x >= e.y && e.x >= e.z { 0 } else if e.y >= e.z { 1 } else { 2 };
            ((at[a] - b.min[a]) / e[a].max(1e-300)).clamp(0.0, 1.0)
        }
        Gradient::Sphere => (at.distance(b.center()) / (b.diagonal() * 0.5).max(1e-300)).clamp(0.0, 1.0),
    };
    p.start_scale + (p.end_scale - p.start_scale) * t
}

/// Points of a regular or hex tiling covering `[lo, hi]` in the plane of axes `u`, `v`
/// (indices), with spacing `s`.
fn tiling_2d(lo: DVec3, hi: DVec3, u: usize, v: usize, s: f64, hex: bool) -> Vec<(f64, f64)> {
    let dv = if hex { s * 3f64.sqrt() * 0.5 } else { s };
    let mut out = Vec::new();
    let nv = ((hi[v] - lo[v]) / dv).ceil() as i64 + 1;
    let nu = ((hi[u] - lo[u]) / s).ceil() as i64 + 1;
    for j in 0..nv {
        let shift = if hex && j % 2 == 1 { s * 0.5 } else { 0.0 };
        for i in 0..nu {
            out.push((lo[u] + i as f64 * s + shift, lo[v] + j as f64 * dv));
        }
    }
    out
}

/// Tiny deterministic generator for the random patterns.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Keep candidate points at least `d` apart (in order).
fn thin_out(points: impl Iterator<Item = DVec3>, d: f64, limit: usize) -> Vec<DVec3> {
    let mut grid: HashMap<[i64; 3], Vec<DVec3>> = HashMap::new();
    let key = |p: DVec3| [(p.x / d).floor() as i64, (p.y / d).floor() as i64, (p.z / d).floor() as i64];
    let mut out = Vec::new();
    for p in points {
        let k = key(p);
        let near = (-1..=1).any(|x| (-1..=1).any(|y| (-1..=1).any(|z| grid.get(&[k[0] + x, k[1] + y, k[2] + z]).is_some_and(|l| l.iter().any(|q| q.distance(p) < d)))));
        if !near {
            grid.entry(k).or_default().push(p);
            out.push(p);
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn elements(mesh: &Mesh, p: &PatternParams, sdf: Option<&MeshSdf>) -> Vec<Element> {
    let b = mesh.bounds();
    let r0 = p.dimension * 0.5;
    let radius = |at: DVec3| r0 * gradient_scale(p, &b, at);
    let tube = |a: DVec3, c: DVec3| Element { a, b: c, r: radius((a + c) * 0.5) };
    let ball = |a: DVec3| Element { a, b: a, r: radius(a) };
    let flipped;
    let m = if matches!(p.kind, PatternType::DelaunayEdges | PatternType::DelaunayDualEdges) {
        let mut c = mesh.clone();
        crate::cleanup::flip_to_improve(&mut c, 8);
        flipped = c;
        &flipped
    } else {
        mesh
    };
    let pad = DVec3::splat(p.spacing);
    match p.kind {
        PatternType::Edges | PatternType::DelaunayEdges => m.edge_ids().map(|e| {
            let [a, c] = m.edge_vertices(e);
            tube(m.vertex(a), m.vertex(c))
        }).collect(),
        PatternType::GroupBorders => m
            .edge_ids()
            .filter(|&e| {
                let [t0, t1] = m.edge_triangles(e);
                t1 == INVALID || m.tri_group(t0) != m.tri_group(t1)
            })
            .map(|e| {
                let [a, c] = m.edge_vertices(e);
                tube(m.vertex(a), m.vertex(c))
            })
            .collect(),
        PatternType::DualEdges | PatternType::DelaunayDualEdges => m
            .edge_ids()
            .map(|e| {
                let [t0, t1] = m.edge_triangles(e);
                let end = if t1 == INVALID { m.edge_midpoint(e) } else { m.tri_centroid(t1) };
                tube(m.tri_centroid(t0), end)
            })
            .collect(),
        PatternType::TiledTubes => {
            // Through the thinnest direction, so the tubes pierce the object.
            let e = b.extents();
            let ax = if e.x <= e.y && e.x <= e.z { 0 } else if e.y <= e.z { 1 } else { 2 };
            let (u, v) = ((ax + 1) % 3, (ax + 2) % 3);
            let (lo, hi) = (b.min - pad, b.max + pad);
            tiling_2d(lo, hi, u, v, p.spacing, p.tiling == Tiling::Hex)
                .into_iter()
                .map(|(x, y)| {
                    let mut a = DVec3::ZERO;
                    a[u] = x;
                    a[v] = y;
                    let mut c = a;
                    a[ax] = lo[ax];
                    c[ax] = hi[ax];
                    tube(a, c)
                })
                .collect()
        }
        PatternType::TiledSpheres => {
            let (lo, hi) = (b.min - pad, b.max + pad);
            let hex = p.tiling == Tiling::Hex;
            // Hex: layers of hex grids, each shifted (close packing).
            let dz = if hex { p.spacing * (2.0f64 / 3.0).sqrt() } else { p.spacing };
            let nz = ((hi.z - lo.z) / dz).ceil() as i64 + 1;
            let mut out = Vec::new();
            for k in 0..nz {
                let shift = if hex && k % 2 == 1 { DVec3::new(p.spacing * 0.5, p.spacing * 3f64.sqrt() / 6.0, 0.0) } else { DVec3::ZERO };
                for (x, y) in tiling_2d(lo, hi, 0, 1, p.spacing, hex) {
                    out.push(ball(DVec3::new(x, y, lo.z + k as f64 * dz) + shift));
                }
            }
            out
        }
        PatternType::Lattice => {
            let (lo, hi) = (b.min - pad, b.max + pad);
            let mut out = Vec::new();
            for ax in 0..3 {
                let (u, v) = ((ax + 1) % 3, (ax + 2) % 3);
                for (x, y) in tiling_2d(lo, hi, u, v, p.spacing, false) {
                    let mut a = DVec3::ZERO;
                    a[u] = x;
                    a[v] = y;
                    let mut c = a;
                    a[ax] = lo[ax];
                    c[ax] = hi[ax];
                    out.push(tube(a, c));
                }
            }
            out
        }
        PatternType::RandomSurface => {
            // Area-weighted samples, thinned to the spacing.
            let tris: Vec<u32> = m.triangle_ids().collect();
            let mut cum = Vec::with_capacity(tris.len());
            let mut total = 0.0;
            for &t in &tris {
                total += m.tri_area(t);
                cum.push(total);
            }
            let mut rng = Rng(p.seed.max(1));
            let n = ((total / (p.spacing * p.spacing)) * 4.0).clamp(16.0, 200_000.0) as usize;
            let pts = (0..n).map(|_| {
                let x = rng.next() * total;
                let t = tris[cum.partition_point(|&c| c < x).min(tris.len() - 1)];
                let (mut s1, mut s2) = (rng.next(), rng.next());
                if s1 + s2 > 1.0 {
                    s1 = 1.0 - s1;
                    s2 = 1.0 - s2;
                }
                let [a, c, d] = m.tri_positions(t);
                a + (c - a) * s1 + (d - a) * s2
            });
            thin_out(pts, p.spacing, 50_000).into_iter().map(ball).collect()
        }
        PatternType::RandomVolume => {
            let mut rng = Rng(p.seed.max(1));
            let e = b.extents();
            let n = ((e.x * e.y * e.z) / p.spacing.powi(3) * 4.0).clamp(16.0, 400_000.0) as usize;
            let pts: Vec<DVec3> = (0..n).map(|_| b.min + DVec3::new(rng.next() * e.x, rng.next() * e.y, rng.next() * e.z)).collect();
            let inside = pts.into_iter().filter(|&q| sdf.is_none_or(|s| s.eval(q) < 0.0));
            thin_out(inside, p.spacing, 50_000).into_iter().map(ball).collect()
        }
    }
}

/// Build the pattern of `mesh` (see the module docs).
pub fn make_pattern(mesh: &Mesh, p: &PatternParams, progress: Progress) -> Result<Mesh> {
    let prog = Prog::new(progress);
    if mesh.triangle_count() == 0 {
        return Err(SolidError::EmptyInput);
    }
    if !(p.dimension > 0.0 && p.dimension.is_finite()) || !(p.resolution > 0.0) {
        return Err(SolidError::InvalidParameter("element dimension and resolution must be positive".into()));
    }
    if !p.kind.from_edges() && !(p.spacing > 0.0) {
        return Err(SolidError::InvalidParameter("element spacing must be positive".into()));
    }
    let min_scale = match p.gradient {
        Gradient::None => 1.0,
        _ => p.start_scale.min(p.end_scale).max(0.05),
    };
    let h = p.dimension * min_scale / p.resolution;
    let far = DEFAULT_BAND_CELLS * h;
    let blend = p.smoothing.max(0.0) * p.dimension * 0.5;
    let uses_object = !p.kind.from_edges() || p.clip_to_surface;
    let sdf = uses_object.then(|| MeshSdf::new(mesh, h, 0.0, 0.5));
    let els = elements(mesh, p, sdf.as_ref());
    if els.is_empty() {
        return Err(SolidError::InvalidParameter(if p.kind == PatternType::GroupBorders {
            "the object has no face-group borders or open edges".into()
        } else {
            "the pattern has no elements (spacing too large?)".into()
        }));
    }
    prog.report(0.05)?;
    let field = ElementField::new(els, far, blend);
    let rmax = field.elements.iter().map(|e| e.r).fold(0.0, f64::max);
    let b = mesh.bounds();
    let pad = rmax + blend + 2.0 * far + 2.0 * h;
    let out = match (sdf, p.kind.from_edges()) {
        (None, _) => {
            let grid = GridSpec::covering(&b, h, pad)?;
            let f = build_field_with(&field, grid, &prog.sub(0.05, 0.8))?;
            extract_and_clean(&f, false, 0.3 * h, true, 0.0, &prog.sub(0.8, 1.0))?
        }
        (Some(sdf), edges) => {
            let op = if edges || p.composition == Composition::Intersect { BoolOp::Intersection } else { BoolOp::Difference };
            let src = BoolField { a: sdf, b: field, op };
            let grid = GridSpec::covering(&b, h, 2.0 * far + 2.0 * h)?;
            let f = build_field_with(&src, grid, &prog.sub(0.05, 0.8))?;
            extract_and_clean(&f, false, 0.3 * h, true, 0.0, &prog.sub(0.8, 1.0))?
        }
    };
    Ok(compact(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn vol(m: &Mesh) -> f64 {
        mm_core::measure::volume(m)
    }

    #[test]
    fn edge_pattern_is_a_closed_frame() {
        let cube = primitives::cube(10.0, 1);
        let p = PatternParams { kind: PatternType::Edges, dimension: 1.0, clip_to_surface: false, resolution: 3.0, ..Default::default() };
        let m = make_pattern(&cube, &p, None).unwrap();
        m.check_validity().unwrap();
        assert!(m.is_closed());
        // 18 edges (12 sides + 6 diagonals) of radius 0.5, minus overlaps at the corners.
        let tubes = 12.0 * 10.0 + 6.0 * 10.0 * 2f64.sqrt();
        let v = vol(&m);
        let want = tubes * std::f64::consts::PI * 0.25;
        assert!(v > 0.7 * want && v < 1.3 * want, "{v} vs {want}");
        // One connected frame.
        assert_eq!(mm_core::topology::connected_components(&m).len(), 1);
    }

    #[test]
    fn tiled_tubes_perforate_and_intersect() {
        let slab = {
            let mut m = primitives::cube(1.0, 1);
            m.transform(|p| p * DVec3::new(20.0, 20.0, 4.0));
            m
        };
        let base = PatternParams { kind: PatternType::TiledTubes, dimension: 2.0, spacing: 4.0, resolution: 3.0, ..Default::default() };
        let holes = make_pattern(&slab, &base, None).unwrap();
        holes.check_validity().unwrap();
        assert!(holes.is_closed());
        let v0 = vol(&slab);
        let v_holes = vol(&holes);
        assert!(v_holes < 0.9 * v0 && v_holes > 0.5 * v0, "{v_holes} of {v0}");
        // Holes go through: genus > 0 (Euler characteristic below 2).
        let chi = holes.vertex_count() as i64 - holes.edge_count() as i64 + holes.triangle_count() as i64;
        assert!(chi < 0, "{chi}");
        // Intersect keeps the complement inside the slab.
        let rods = make_pattern(&slab, &PatternParams { composition: Composition::Intersect, ..base }, None).unwrap();
        assert!(((vol(&rods) + v_holes) - v0).abs() < 0.08 * v0, "{} + {v_holes} vs {v0}", vol(&rods));
    }

    #[test]
    fn spheres_lattice_and_random_patterns_build() {
        let mut ball = primitives::icosphere(10.0, 3);
        for t in ball.triangle_ids().collect::<Vec<_>>() {
            if ball.tri_centroid(t).z > 0.0 {
                ball.set_tri_group(t, 1);
            }
        }
        for kind in [PatternType::TiledSpheres, PatternType::Lattice, PatternType::RandomSurface, PatternType::RandomVolume, PatternType::DualEdges, PatternType::GroupBorders] {
            let p = PatternParams { kind, dimension: 2.0, spacing: 5.0, resolution: 2.5, clip_to_surface: true, composition: Composition::Subtract, ..Default::default() };
            let m = make_pattern(&ball, &p, None).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            m.check_validity().unwrap();
            assert!(m.is_closed() && m.triangle_count() > 100, "{kind:?}");
            assert!(vol(&m) < vol(&ball) * 1.01, "{kind:?}");
        }
    }

    #[test]
    fn gradient_grows_elements() {
        let p = PatternParams { gradient: Gradient::Line, start_scale: 0.5, end_scale: 2.0, ..Default::default() };
        let b = Aabb { min: DVec3::ZERO, max: DVec3::new(10.0, 1.0, 1.0) };
        assert!((gradient_scale(&p, &b, DVec3::ZERO) - 0.5).abs() < 1e-12);
        assert!((gradient_scale(&p, &b, DVec3::new(10.0, 0.0, 0.0)) - 2.0).abs() < 1e-12);
        assert!((smin(1.0, 3.0, 0.0) - 1.0).abs() < 1e-12 && smin(1.0, 1.0, 0.5) < 1.0);
    }
}
