//! Stamp (Meshmixer's Stamp tool): embed a planar polygon — a stencil — into the surface
//! along its projection, so the enclosed area becomes its own face group (or a hole).
//!
//! The stencil lives in a plane over the surface. Each stencil edge is walked across the
//! triangles as seen in that plane: mesh edges it crosses are split where it crosses them
//! and its corners are poked into the triangles under them, so the outline ends up as a
//! loop of mesh edges lying on the surface. Crossings very close to an existing vertex go
//! through that vertex instead, which keeps slivers away at the cost of a slightly bent
//! outline.

use crate::{EditError, EditResult};
use glam::{DVec2, DVec3};
use mm_core::hash::FastSet;
use mm_core::Mesh;

/// Crossings this close to an end of a mesh edge (as a fraction of it) snap to the vertex;
/// corners this close to an edge or vertex (barycentric) land on it.
const SNAP: f64 = 0.06;

/// Stencil plane: `origin`, orthonormal in-plane axes `u` and `v`, and normal `n = u × v`
/// pointing out of the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StampFrame {
    pub origin: DVec3,
    pub u: DVec3,
    pub v: DVec3,
    pub n: DVec3,
}

impl StampFrame {
    /// Frame at `origin` facing `normal`, with `up` (projected into the plane) as `v`.
    pub fn new(origin: DVec3, normal: DVec3, up: DVec3) -> Self {
        let n = normal.normalize_or_zero();
        let mut v = (up - n * up.dot(n)).normalize_or_zero();
        if v == DVec3::ZERO {
            v = mm_core::geom::orthonormal_basis(n).1;
        }
        Self { origin, u: v.cross(n), v, n }
    }
    pub fn project(&self, p: DVec3) -> DVec2 {
        let d = p - self.origin;
        DVec2::new(d.dot(self.u), d.dot(self.v))
    }
    pub fn lift(&self, q: DVec2) -> DVec3 {
        self.origin + self.u * q.x + self.v * q.y
    }
}

/// Built-in stencils.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stencil {
    Circle,
    Square,
    Triangle,
    Hexagon,
    Star,
    Arrow,
    Heart,
}

impl Stencil {
    pub const ALL: [Stencil; 7] = [Stencil::Circle, Stencil::Square, Stencil::Triangle, Stencil::Hexagon, Stencil::Star, Stencil::Arrow, Stencil::Heart];

    pub fn label(self) -> &'static str {
        match self {
            Stencil::Circle => "Circle",
            Stencil::Square => "Square",
            Stencil::Triangle => "Triangle",
            Stencil::Hexagon => "Hexagon",
            Stencil::Star => "Star",
            Stencil::Arrow => "Arrow",
            Stencil::Heart => "Heart",
        }
    }

    /// Counter-clockwise outline fitting the unit circle.
    pub fn outline(self) -> Vec<DVec2> {
        let ring = |n: usize, start_deg: f64, r: &dyn Fn(usize) -> f64| -> Vec<DVec2> {
            (0..n)
                .map(|i| {
                    let a = (start_deg + 360.0 * i as f64 / n as f64).to_radians();
                    DVec2::new(a.cos(), a.sin()) * r(i)
                })
                .collect()
        };
        let pts = match self {
            Stencil::Circle => ring(48, 0.0, &|_| 1.0),
            Stencil::Square => ring(4, 45.0, &|_| 1.0),
            Stencil::Triangle => ring(3, 90.0, &|_| 1.0),
            Stencil::Hexagon => ring(6, 0.0, &|_| 1.0),
            Stencil::Star => ring(10, 90.0, &|i| if i % 2 == 0 { 1.0 } else { 0.42 }),
            Stencil::Arrow => [(0.0, 1.0), (-0.65, 0.25), (-0.25, 0.25), (-0.25, -1.0), (0.25, -1.0), (0.25, 0.25), (0.65, 0.25)].map(|(x, y)| DVec2::new(x, y)).to_vec(),
            Stencil::Heart => {
                let raw: Vec<DVec2> = (0..48)
                    .map(|i| {
                        let t = std::f64::consts::TAU * i as f64 / 48.0;
                        DVec2::new(16.0 * t.sin().powi(3), 13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos())
                    })
                    .collect();
                let c = raw.iter().copied().sum::<DVec2>() / raw.len() as f64;
                let r = raw.iter().map(|p| (*p - c).length()).fold(0.0, f64::max);
                raw.iter().map(|p| (*p - c) / r).collect()
            }
        };
        let r = pts.iter().map(|p| p.length()).fold(0.0, f64::max);
        ccw(pts.into_iter().map(|p| p / r).collect())
    }
}

fn signed_area(p: &[DVec2]) -> f64 {
    (0..p.len()).map(|i| p[i].perp_dot(p[(i + 1) % p.len()])).sum::<f64>() * 0.5
}

fn ccw(mut p: Vec<DVec2>) -> Vec<DVec2> {
    if signed_area(&p) < 0.0 {
        p.reverse();
    }
    p
}

/// What [`stamp_polygon`] produced.
#[derive(Clone, Debug, Default)]
pub struct StampResult {
    /// Triangles enclosed by the outline.
    pub inside: Vec<u32>,
    /// The outline as a closed loop of vertices (counter-clockwise seen from `n`).
    pub outline: Vec<u32>,
}

/// Embed `poly` (stencil-plane coordinates, any orientation) into the surface around the
/// triangle `seed`. Only surface facing the stencil (normals within about 85° of `frame.n`)
/// and connected to `seed` is cut, so the back of a thin part is left alone.
pub fn stamp_polygon(mesh: &mut Mesh, seed: u32, frame: &StampFrame, poly: &[DVec2]) -> EditResult<StampResult> {
    if poly.len() < 3 || signed_area(poly).abs() < 1e-18 {
        return Err(EditError::InvalidParameter("the stencil needs an outline with an area".into()));
    }
    if !mesh.is_triangle(seed) {
        return Err(EditError::InvalidParameter("no surface under the stamp".into()));
    }
    let poly = ccw(poly.to_vec());
    let reach = poly.iter().map(|p| p.length()).fold(0.0, f64::max);
    let region = vec![false; mesh.max_tid()];
    let mut w = Walker { mesh, frame: *frame, region, grid: Grid::default() };
    w.grow_region(seed, reach);

    // Corners first (each lands on a vertex), then the edges between them.
    let mut corners = Vec::with_capacity(poly.len());
    for &p in &poly {
        corners.push(w.insert_point(p)?);
    }
    let mut outline: Vec<u32> = Vec::new();
    for i in 0..poly.len() {
        let (a, b) = (corners[i], corners[(i + 1) % poly.len()]);
        let path = w.walk(a, b)?;
        outline.extend(&path[..path.len() - 1]);
    }
    outline.dedup();
    if outline.len() > 1 && outline.first() == outline.last() {
        outline.pop();
    }
    if outline.len() < 3 {
        return Err(EditError::Failed("the stamp is too small for the mesh here".into()));
    }
    let inside = w.fill_inside(&outline)?;
    Ok(StampResult { inside, outline })
}

/// Triangles bucketed by the cells of a uniform grid over the stencil plane that their boxes
/// there overlap (each box a little widened). A point lies only in triangles of its own
/// cell, so the triangle under a stencil corner is looked for among a few, not the whole
/// region.
#[derive(Default)]
struct Grid {
    min: DVec2,
    cell: f64,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<u32>>,
    /// Widening of every box beyond its own share, for rounding in the stencil plane.
    pad: f64,
}

impl Grid {
    /// A grid over `boxes` (triangle, box corners), with a few triangles to a cell.
    fn new(boxes: &[(u32, DVec2, DVec2)]) -> Self {
        let (mut lo, mut hi, mut size) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY), 0.0);
        for (_, a, b) in boxes {
            lo = lo.min(*a);
            hi = hi.max(*b);
            size += (*b - *a).max_element();
        }
        let ext = hi - lo;
        if !(ext.is_finite() && ext.max_element() > 0.0) {
            // Nothing to spread out: one cell holds every triangle.
            let mut g = Self { min: DVec2::ZERO, cell: 1.0, nx: 1, ny: 1, cells: vec![Vec::new()], pad: 0.0 };
            for &(t, a, b) in boxes {
                g.insert(t, a, b);
            }
            return g;
        }
        // Cells the size of an average triangle, but no more of them than half the triangles.
        let n = boxes.len().max(1) as f64;
        let mut cell = (size / n).max(ext.max_element() / 4096.0);
        while (ext.x / cell + 1.0) * (ext.y / cell + 1.0) > 0.5 * n + 4.0 {
            cell *= 1.25;
        }
        let (nx, ny) = ((ext.x / cell) as usize + 1, (ext.y / cell) as usize + 1);
        let mut g = Self { min: lo, cell, nx, ny, cells: vec![Vec::new(); nx * ny], pad: ext.max_element() * 1e-9 };
        for &(t, a, b) in boxes {
            g.insert(t, a, b);
        }
        g
    }

    /// The cell holding `p` (points beyond the grid go to the cells at its border).
    fn cell_of(&self, p: DVec2) -> (usize, usize) {
        let q = (p - self.min) / self.cell;
        ((q.x.floor().max(0.0) as usize).min(self.nx - 1), (q.y.floor().max(0.0) as usize).min(self.ny - 1))
    }

    /// File triangle `t` under every cell its box `a`..`b` (widened by about a millionth of
    /// its size) overlaps.
    fn insert(&mut self, t: u32, a: DVec2, b: DVec2) {
        let pad = DVec2::splat((b - a).max_element() * 1e-6 + self.pad);
        let ((x0, y0), (x1, y1)) = (self.cell_of(a - pad), self.cell_of(b + pad));
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.cells[y * self.nx + x].push(t);
            }
        }
    }

    /// The triangles filed where `p` is.
    fn at(&self, p: DVec2) -> &[u32] {
        let (x, y) = self.cell_of(p);
        &self.cells[y * self.nx + x]
    }
}

struct Walker<'a> {
    mesh: &'a mut Mesh,
    frame: StampFrame,
    /// Triangles the stamp may cut (front-facing, connected to the seed), by triangle id.
    region: Vec<bool>,
    /// The region's triangles by where they lie in the stencil plane.
    grid: Grid,
}

impl Walker<'_> {
    fn p2(&self, v: u32) -> DVec2 {
        self.frame.project(self.mesh.vertex(v))
    }

    fn in_region(&self, t: u32) -> bool {
        self.region.get(t as usize).copied().unwrap_or(false)
    }

    /// Triangle `t`'s box in the stencil plane.
    fn tri_box(&self, t: u32) -> (DVec2, DVec2) {
        let [a, b, c] = self.mesh.triangle(t).map(|v| self.p2(v));
        (a.min(b).min(c), a.max(b).max(c))
    }

    /// Add a triangle made by a split or poke of a region triangle to the region.
    fn add(&mut self, t: u32) {
        let i = t as usize;
        if i >= self.region.len() {
            self.region.resize(i + 1, false);
        }
        self.region[i] = true;
        let (a, b) = self.tri_box(t);
        self.grid.insert(t, a, b);
    }

    fn grow_region(&mut self, seed: u32, reach: f64) {
        let m = &*self.mesh;
        let limit = reach * 1.5 + m.triangle(seed).iter().map(|&v| m.vertex(v).distance(m.tri_centroid(seed))).fold(0.0, f64::max) * 2.0;
        let mut stack = vec![seed];
        self.region[seed as usize] = true;
        while let Some(t) = stack.pop() {
            for e in m.tri_edges(t) {
                let o = m.edge_other_triangle(e, t);
                if o == mm_core::INVALID || self.region[o as usize] {
                    continue;
                }
                let near = m.triangle(o).iter().any(|&v| self.frame.project(m.vertex(v)).length() < limit);
                if near && m.tri_normal(o).dot(self.frame.n) > 0.08 {
                    self.region[o as usize] = true;
                    stack.push(o);
                }
            }
        }
        let boxes: Vec<(u32, DVec2, DVec2)> = (0..self.region.len() as u32)
            .filter(|&t| self.region[t as usize])
            .map(|t| {
                let (a, b) = self.tri_box(t);
                (t, a, b)
            })
            .collect();
        self.grid = Grid::new(&boxes);
    }

    /// Track triangles created by splitting region triangles.
    fn adopt(&mut self, orig: [u32; 2], new: [u32; 2]) {
        for k in 0..2 {
            if orig[k] != mm_core::INVALID && self.in_region(orig[k]) && new[k] != mm_core::INVALID {
                self.add(new[k]);
            }
        }
    }

    /// Split the edge between `a` and `b` at the fraction `s` from `a`.
    fn split(&mut self, a: u32, b: u32, s: f64) -> EditResult<u32> {
        let e = self.mesh.find_edge(a, b).ok_or_else(|| EditError::Failed("stamp outline lost its edge".into()))?;
        let [ea, _] = self.mesh.edge_vertices(e);
        let info = self.mesh.split_edge_at(e, if ea == a { s } else { 1.0 - s })?;
        self.adopt(info.t_orig, info.t_new);
        Ok(info.v_new)
    }

    /// Barycentric coordinates of `p` in a triangle's projection.
    fn bary(&self, t: u32, p: DVec2) -> Option<DVec3> {
        let [a, b, c] = self.mesh.triangle(t).map(|v| self.p2(v));
        let d = (b - a).perp_dot(c - a);
        if d.abs() < 1e-300 {
            return None;
        }
        let wb = (p - a).perp_dot(c - a) / d;
        let wc = (b - a).perp_dot(p - a) / d;
        Some(DVec3::new(1.0 - wb - wc, wb, wc))
    }

    /// The triangle of `tris` (ascending ids) whose projection holds `p` best: the largest
    /// smallest barycentric coordinate, the last one on a tie. None if `p` is in none of them.
    fn holding(&self, tris: impl Iterator<Item = u32>, p: DVec2) -> Option<(u32, DVec3)> {
        let found = tris.filter_map(|t| self.bary(t, p).map(|b| (t, b))).max_by(|x, y| x.1.min_element().total_cmp(&y.1.min_element()));
        found.filter(|(_, b)| b.min_element() > -1e-9)
    }

    /// Put a vertex at stencil point `p`: an existing vertex, a split edge, or a poked triangle.
    fn insert_point(&mut self, p: DVec2) -> EditResult<u32> {
        // The triangles filed where `p` is; any region triangle holding it is among them,
        // unless it is too thin to say where it is, so then the whole region is asked.
        let mut near = self.grid.at(p).to_vec();
        near.sort_unstable();
        near.dedup();
        let found = self.holding(near.into_iter(), p).or_else(|| self.holding((0..self.region.len() as u32).filter(|&t| self.region[t as usize]), p));
        let Some((t, b)) = found else {
            return Err(EditError::Failed("the stamp reaches past the surface facing it".into()));
        };
        let tv = self.mesh.triangle(t);
        let small: Vec<usize> = (0..3).filter(|&i| b[i] < SNAP).collect();
        match small.len() {
            0 => {
                let info = self.mesh.poke_triangle(t, b)?;
                for n in info.t_new {
                    self.add(n);
                }
                Ok(info.v_new)
            }
            1 => {
                // On the edge opposite the small coordinate.
                let i = small[0];
                let (a, c) = (tv[(i + 1) % 3], tv[(i + 2) % 3]);
                let (wa, wc) = (b[(i + 1) % 3], b[(i + 2) % 3]);
                self.split(a, c, wc / (wa + wc))
            }
            _ => Ok(tv[(0..3).max_by(|&x, &y| b[x].total_cmp(&b[y])).unwrap()]),
        }
    }

    /// Walk in the stencil plane from vertex `from` straight to vertex `end`, splitting the
    /// edges crossed. Returns the vertex path, `from` .. `end`.
    fn walk(&mut self, from: u32, end: u32) -> EditResult<Vec<u32>> {
        let to = self.p2(end);
        let mut path = vec![from];
        let mut cur = from;
        for _ in 0..100_000 {
            if cur == end {
                return Ok(path);
            }
            let pc = self.p2(cur);
            let dir = to - pc;
            // The triangle around `cur` whose wedge holds the direction.
            let mut step = None;
            for t in self.mesh.vertex_triangles(cur) {
                if !self.in_region(t) {
                    continue;
                }
                let tv = self.mesh.triangle(t);
                let k = tv.iter().position(|&v| v == cur).unwrap();
                let (x, y) = (tv[(k + 1) % 3], tv[(k + 2) % 3]);
                let (px, py) = (self.p2(x), self.p2(y));
                if (px - pc).perp_dot(dir) >= -1e-12 * dir.length() * (px - pc).length() && dir.perp_dot(py - pc) >= -1e-12 * dir.length() * (py - pc).length() {
                    step = Some((x, y, px, py));
                    break;
                }
            }
            let Some((x, y, px, py)) = step else {
                return Err(EditError::Failed("the stamp outline runs off the surface facing it".into()));
            };
            // Does the segment end inside this triangle? Then `end` is one of its corners
            // (vertices never sit inside another triangle unless the surface folds over).
            let edge = py - px;
            if edge.perp_dot(to - px) > 0.0 {
                if end == x || end == y {
                    path.push(end);
                    return Ok(path);
                }
                return Err(EditError::Failed("the surface folds over under the stamp".into()));
            }
            // Crossing point along x → y.
            let denom = dir.perp_dot(edge);
            let s = if denom.abs() < 1e-300 { 0.5 } else { (dir.perp_dot(pc - px) / denom).clamp(0.0, 1.0) };
            let next = if s < SNAP {
                x
            } else if s > 1.0 - SNAP {
                y
            } else {
                self.split(x, y, s)?
            };
            path.push(next);
            cur = next;
        }
        Err(EditError::Failed("stamp outline did not converge".into()))
    }

    /// Triangles on the left of the (counter-clockwise) outline.
    fn fill_inside(&self, outline: &[u32]) -> EditResult<Vec<u32>> {
        let m = &*self.mesh;
        // Both sets are only asked, never walked: the result is sorted below.
        let mut wall = FastSet::default();
        for i in 0..outline.len() {
            let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
            let e = m.find_edge(a, b).ok_or_else(|| EditError::Failed("stamp outline is not a closed loop of edges".into()))?;
            wall.insert(e);
        }
        let (a, b) = (outline[0], outline[1]);
        let e = m.find_edge(a, b).unwrap();
        let start = m.edge_triangles(e).into_iter().find(|&t| {
            t != mm_core::INVALID && {
                let tv = m.triangle(t);
                (0..3).any(|k| tv[k] == a && tv[(k + 1) % 3] == b)
            }
        });
        let Some(start) = start else { return Err(EditError::Failed("stamp outline has no inside".into())) };
        let mut seen: FastSet<u32> = FastSet::default();
        seen.insert(start);
        let mut stack = vec![start];
        while let Some(t) = stack.pop() {
            if !self.in_region(t) {
                return Err(EditError::Failed("the stamp outline does not close on the surface".into()));
            }
            for e in m.tri_edges(t) {
                let o = m.edge_other_triangle(e, t);
                if !wall.contains(&e) && o != mm_core::INVALID && seen.insert(o) {
                    stack.push(o);
                }
            }
        }
        let mut inside: Vec<u32> = seen.into_iter().collect();
        inside.sort_unstable();
        Ok(inside)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;
    use std::collections::HashSet;

    fn area(m: &Mesh, tris: &[u32]) -> f64 {
        tris.iter().map(|&t| m.tri_area(t)).sum()
    }

    #[test]
    fn stencils_are_counter_clockwise_and_fit_the_unit_circle() {
        for s in Stencil::ALL {
            let p = s.outline();
            assert!(signed_area(&p) > 0.1, "{s:?}");
            assert!(p.iter().all(|q| q.length() <= 1.0 + 1e-9), "{s:?}");
        }
    }

    #[test]
    fn square_on_a_plane_encloses_its_area() {
        let mut m = primitives::plane_grid(4.0, 4.0, 17, 17);
        let n0 = m.triangle_count();
        let frame = StampFrame::new(DVec3::new(0.03, -0.02, 0.0), DVec3::Z, DVec3::Y);
        let seed = m.triangle_ids().find(|&t| m.tri_centroid(t).truncate().length() < 0.2).unwrap();
        let poly: Vec<DVec2> = Stencil::Square.outline().iter().map(|p| *p * 1.3).collect();
        let r = stamp_polygon(&mut m, seed, &frame, &poly).unwrap();
        m.check_validity().unwrap();
        assert!(m.triangle_count() > n0);
        // Square of side 1.3·√2: area 3.38; snapping bends the outline a little.
        let a = area(&m, &r.inside);
        assert!((a - 3.38).abs() < 0.12, "{a}");
        // The outline is on the square's border.
        for &v in &r.outline {
            let q = frame.project(m.vertex(v));
            let d = (q.x.abs().max(q.y.abs()) - 1.3 / 2f64.sqrt()).abs();
            assert!(d < 0.25 * 4.0 / 17.0, "{q:?}");
        }
        // Inside triangles are exactly those whose centroid is inside the square.
        let inside: HashSet<u32> = r.inside.iter().copied().collect();
        for t in m.triangle_ids() {
            let c = frame.project(m.tri_centroid(t));
            let s = 1.3 / 2f64.sqrt();
            if c.x.abs() < s - 0.2 && c.y.abs() < s - 0.2 {
                assert!(inside.contains(&t));
            }
            if c.x.abs() > s + 0.2 || c.y.abs() > s + 0.2 {
                assert!(!inside.contains(&t));
            }
        }
    }

    #[test]
    fn every_stencil_stamps_a_sphere() {
        for s in Stencil::ALL {
            let mut m = primitives::icosphere(1.0, 4);
            let frame = StampFrame::new(DVec3::Z, DVec3::Z, DVec3::Y);
            let seed = m.triangle_ids().max_by(|&a, &b| m.tri_centroid(a).z.total_cmp(&m.tri_centroid(b).z)).unwrap();
            let poly: Vec<DVec2> = s.outline().iter().map(|p| *p * 0.45).collect();
            let r = stamp_polygon(&mut m, seed, &frame, &poly).unwrap_or_else(|e| panic!("{s:?}: {e}"));
            m.check_validity().unwrap();
            assert!(m.is_closed());
            // Projected area of the inside ≈ the stencil's area.
            let proj: f64 = r.inside.iter().map(|&t| m.tri_normal(t).dot(DVec3::Z) * m.tri_area(t)).sum();
            let want = signed_area(&poly);
            assert!((proj - want).abs() < 0.15 * want + 0.01, "{s:?}: {proj} vs {want}");
        }
    }

    #[test]
    fn random_stamps_keep_the_mesh_valid() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut m = primitives::icosphere(1.0, 4);
        let mut ok = 0;
        for i in 0..40 {
            let dir = DVec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
            let bvh = mm_core::Bvh::build(&m);
            let h = bvh.nearest(&m, dir * 1.2, 2.0).unwrap();
            let up = DVec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
            let frame = StampFrame::new(h.point, m.tri_normal(h.tid), up);
            let stencil = Stencil::ALL[i % Stencil::ALL.len()];
            let size = 0.08 + 0.3 * rnd();
            let poly: Vec<DVec2> = stencil.outline().iter().map(|p| *p * size).collect();
            if let Ok(r) = stamp_polygon(&mut m, h.tid, &frame, &poly) {
                assert!(!r.inside.is_empty());
                for &t in &r.inside {
                    m.set_tri_group(t, i as i32 + 1);
                }
                ok += 1;
            }
            m.check_validity().unwrap();
            assert!(m.is_closed());
        }
        assert!(ok >= 36, "{ok} of 40 stamps went through");
    }

    #[test]
    fn the_grid_finds_the_triangle_a_scan_of_the_whole_region_finds() {
        let mut seed = 99u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut m = primitives::icosphere(1.0, 4);
        let frame = StampFrame::new(DVec3::Z, DVec3::new(0.1, 0.05, 1.0), DVec3::Y);
        let top = m.triangle_ids().max_by(|&a, &b| m.tri_centroid(a).z.total_cmp(&m.tri_centroid(b).z)).unwrap();
        let region = vec![false; m.max_tid()];
        let mut w = Walker { mesh: &mut m, frame, region, grid: Grid::default() };
        w.grow_region(top, 0.5);
        // Random points (some off the region), the region's vertices and edge midpoints (on
        // several triangles at once); every few points one is put in, which splits or pokes
        // triangles that the grid then holds too.
        let mut points: Vec<DVec2> = (0..300).map(|_| DVec2::new(rnd() * 1.8 - 0.9, rnd() * 1.8 - 0.9)).collect();
        for t in (0..w.region.len() as u32).filter(|&t| w.region[t as usize]).step_by(7) {
            let [a, b, _] = w.mesh.triangle(t).map(|v| w.p2(v));
            points.extend([a, (a + b) * 0.5]);
        }
        let (mut found, mut inserted) = (0, 0);
        for (i, p) in points.into_iter().enumerate() {
            let mut near = w.grid.at(p).to_vec();
            near.sort_unstable();
            near.dedup();
            let by_grid = w.holding(near.into_iter(), p);
            let by_scan = w.holding((0..w.region.len() as u32).filter(|&t| w.region[t as usize]), p);
            assert_eq!(by_grid, by_scan, "{p:?}");
            found += by_scan.is_some() as usize;
            if by_scan.is_some() && i % 4 == 0 && w.insert_point(p).is_ok() {
                inserted += 1;
            }
        }
        assert!(found > 150 && inserted > 30, "{found} found, {inserted} put in");
        w.mesh.check_validity().unwrap();
    }

    #[test]
    fn stamps_past_the_facing_surface_are_rejected() {
        let mut m = primitives::icosphere(1.0, 3);
        let frame = StampFrame::new(DVec3::Z, DVec3::Z, DVec3::Y);
        let seed = m.triangle_ids().max_by(|&a, &b| m.tri_centroid(a).z.total_cmp(&m.tri_centroid(b).z)).unwrap();
        let poly: Vec<DVec2> = Stencil::Circle.outline().iter().map(|p| *p * 1.5).collect();
        assert!(stamp_polygon(&mut m, seed, &frame, &poly).is_err());
    }
}
