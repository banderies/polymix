//! Ear-clipping triangulation of a planar polygon with holes.
//!
//! Holes are merged into the outer ring through bridge edges found with a leftward ray from
//! each hole's leftmost vertex (Eberly, "Triangulation by Ear Clipping"), then ears are
//! clipped. Unlike general-purpose triangulators, *every* input vertex is kept (the points
//! are mesh vertices of a boundary loop that must be closed), so collinear and coincident
//! points are handled by staged relaxation instead of being filtered out:
//!
//! 1. strict ears (positive area, no other vertex inside or on the ear),
//! 2. relaxed ears (zero-area ears allowed, only strictly interior vertices block),
//! 3. forced clip of the best remaining convex vertex (only reached on self-intersecting
//!    input; guarantees termination with a topologically valid result).
//!
//! Point-in-ear queries use a uniform grid so large loops (thousands of vertices) stay fast.
//! Output triangles are counter-clockwise and follow the input ring orientation.

use mm_core::DVec2;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
struct Node {
    p: u32,
    prev: u32,
    next: u32,
    alive: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Triangulation {
    /// Triangles as point indices, counter-clockwise.
    pub triangles: Vec<[usize; 3]>,
    /// Holes that could not be bridged to the outer ring (left untriangulated).
    pub unbridged_holes: Vec<usize>,
    /// Number of forced (possibly overlapping) clips.
    pub forced: usize,
}

#[inline]
fn cross(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    (b - a).perp_dot(c - a)
}

#[inline]
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Orientation-agnostic inclusive point-in-triangle test.
fn point_in_triangle_any(a: DVec2, b: DVec2, c: DVec2, p: DVec2) -> bool {
    let d1 = cross(a, b, p);
    let d2 = cross(b, c, p);
    let d3 = cross(c, a, p);
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

struct Grid {
    min: DVec2,
    inv: DVec2,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<u32>>,
}

impl Grid {
    fn new(min: DVec2, max: DVec2, n: usize) -> Self {
        let ext = (max - min).max(DVec2::splat(1e-300));
        let aspect = (ext.x / ext.y).clamp(1e-3, 1e3);
        let n = n.max(1) as f64;
        let nx = ((n * aspect).sqrt().ceil() as usize).clamp(1, 2048);
        let ny = ((n / aspect).sqrt().ceil() as usize).clamp(1, 2048);
        Grid { min, inv: DVec2::new(nx as f64 / ext.x, ny as f64 / ext.y), nx, ny, cells: vec![Vec::new(); nx * ny] }
    }
    #[inline]
    fn cell(&self, p: DVec2) -> (usize, usize) {
        let q = (p - self.min) * self.inv;
        let x = if q.x.is_finite() { (q.x.max(0.0) as usize).min(self.nx - 1) } else { 0 };
        let y = if q.y.is_finite() { (q.y.max(0.0) as usize).min(self.ny - 1) } else { 0 };
        (x, y)
    }
    fn insert(&mut self, id: u32, p: DVec2) {
        let (x, y) = self.cell(p);
        self.cells[y * self.nx + x].push(id);
    }
}

struct Clipper<'a> {
    pts: &'a [DVec2],
    nodes: Vec<Node>,
    used: HashSet<(u32, u32)>,
    eps: f64,
    grid: Option<Grid>,
}

impl<'a> Clipper<'a> {
    #[inline]
    fn pt(&self, n: u32) -> DVec2 {
        self.pts[self.nodes[n as usize].p as usize]
    }
    #[inline]
    fn pid(&self, n: u32) -> u32 {
        self.nodes[n as usize].p
    }
    #[inline]
    fn next(&self, n: u32) -> u32 {
        self.nodes[n as usize].next
    }
    #[inline]
    fn prev(&self, n: u32) -> u32 {
        self.nodes[n as usize].prev
    }

    fn ring(&mut self, idx: &[usize]) -> u32 {
        let first = self.nodes.len() as u32;
        let n = idx.len() as u32;
        for (k, &i) in idx.iter().enumerate() {
            let k = k as u32;
            self.nodes.push(Node { p: i as u32, prev: first + (k + n - 1) % n, next: first + (k + 1) % n, alive: true });
        }
        for k in 0..idx.len() {
            self.used.insert(key(idx[k] as u32, idx[(k + 1) % idx.len()] as u32));
        }
        first
    }

    fn unlink(&mut self, n: u32) {
        let (p, q) = (self.prev(n), self.next(n));
        self.nodes[p as usize].next = q;
        self.nodes[q as usize].prev = p;
        self.nodes[n as usize].alive = false;
    }

    /// Link node `a` (outer) to node `b` (hole) with a doubled bridge edge.
    fn split(&mut self, a: u32, b: u32) {
        let a2 = self.nodes.len() as u32;
        self.nodes.push(Node { p: self.pid(a), prev: 0, next: 0, alive: true });
        let b2 = self.nodes.len() as u32;
        self.nodes.push(Node { p: self.pid(b), prev: 0, next: 0, alive: true });
        let an = self.next(a);
        let bp = self.prev(b);
        self.nodes[a as usize].next = b;
        self.nodes[b as usize].prev = a;
        self.nodes[a2 as usize].next = an;
        self.nodes[an as usize].prev = a2;
        self.nodes[b2 as usize].next = a2;
        self.nodes[a2 as usize].prev = b2;
        self.nodes[bp as usize].next = b2;
        self.nodes[b2 as usize].prev = bp;
        let k = key(self.pid(a), self.pid(b));
        self.used.insert(k);
    }

    fn locally_inside(&self, a: u32, b: u32) -> bool {
        let (pa, pb) = (self.pt(a), self.pt(b));
        let (pp, pn) = (self.pt(self.prev(a)), self.pt(self.next(a)));
        if cross(pp, pa, pn) > 0.0 {
            cross(pa, pn, pb) >= 0.0 && cross(pa, pb, pp) >= 0.0
        } else {
            cross(pa, pn, pb) > 0.0 || cross(pa, pb, pp) > 0.0
        }
    }

    /// Find an outer-ring node visible from hole node `h` (leftward ray cast).
    fn find_bridge(&self, h: u32, start: u32) -> Option<u32> {
        let hp = self.pt(h);
        let (hx, hy) = (hp.x, hp.y);
        let mut qx = f64::NEG_INFINITY;
        let mut m: Option<u32> = None;
        let mut p = start;
        loop {
            let a = self.pt(p);
            let nx = self.next(p);
            let b = self.pt(nx);
            if hy <= a.y && hy >= b.y && b.y != a.y {
                let x = a.x + (hy - a.y) * (b.x - a.x) / (b.y - a.y);
                if x <= hx && x > qx {
                    qx = x;
                    m = Some(if a.x < b.x { p } else { nx });
                    if x == hx {
                        return m;
                    }
                }
            }
            p = nx;
            if p == start {
                break;
            }
        }
        let mut m = m?;
        // Vertices inside triangle (hole point, ray hit, candidate) may block the bridge;
        // pick the one making the smallest angle with the ray.
        let stop = m;
        let mp = self.pt(m);
        let mut tan_min = f64::INFINITY;
        let ta = DVec2::new(hx, hy);
        let tb = DVec2::new(qx, hy);
        p = m;
        loop {
            let pp = self.pt(p);
            if hx >= pp.x && pp.x >= mp.x && hx != pp.x && point_in_triangle_any(ta, mp, tb, pp) {
                let tan = (hy - pp.y).abs() / (hx - pp.x);
                let cur = self.pt(m);
                if self.locally_inside(p, h) && (tan < tan_min || (tan == tan_min && pp.x > cur.x)) {
                    m = p;
                    tan_min = tan;
                }
            }
            p = self.next(p);
            if p == stop {
                break;
            }
        }
        Some(m)
    }

    fn diag_allowed(&self, a: u32, c: u32, diag_ok: &mut dyn FnMut(usize, usize) -> bool) -> bool {
        a != c && !self.used.contains(&key(a, c)) && diag_ok(a as usize, c as usize)
    }

    fn is_ear(&self, prev: u32, ear: u32, next: u32, relaxed: bool) -> bool {
        let (a, b, c) = (self.pt(prev), self.pt(ear), self.pt(next));
        let ar = cross(a, b, c);
        let eps = self.eps;
        if !relaxed {
            if ar <= eps {
                return false;
            }
        } else if ar < -eps || (ar <= eps && (b - a).dot(c - b) <= 0.0) {
            return false;
        }
        let ids = [self.pid(prev), self.pid(ear), self.pid(next)];
        let grid = self.grid.as_ref().expect("grid built");
        let bmin = a.min(b).min(c);
        let bmax = a.max(b).max(c);
        let (x0, y0) = grid.cell(bmin);
        let (x1, y1) = grid.cell(bmax);
        for y in y0..=y1 {
            for x in x0..=x1 {
                for &n in &grid.cells[y * grid.nx + x] {
                    let node = self.nodes[n as usize];
                    if !node.alive || ids.contains(&node.p) {
                        continue;
                    }
                    let q = self.pts[node.p as usize];
                    if q.x < bmin.x || q.x > bmax.x || q.y < bmin.y || q.y > bmax.y {
                        continue;
                    }
                    let (d1, d2, d3) = (cross(a, b, q), cross(b, c, q), cross(c, a, q));
                    let blocked =
                        if relaxed { d1 > eps && d2 > eps && d3 > eps } else { d1 >= -eps && d2 >= -eps && d3 >= -eps };
                    if blocked {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn emit(&mut self, prev: u32, ear: u32, next: u32, out: &mut Vec<[usize; 3]>) {
        let (a, b, c) = (self.pid(prev), self.pid(ear), self.pid(next));
        out.push([a as usize, b as usize, c as usize]);
        self.used.insert(key(a, c));
    }

    /// Choose a vertex to clip when no ear exists.
    fn pick_forced(&self, start: u32, diag_ok: &mut dyn FnMut(usize, usize) -> bool) -> u32 {
        let mut best: Option<(u32, f64)> = None;
        let mut fallback_diag: Option<u32> = None;
        let mut fallback_any: Option<u32> = None;
        let mut n = start;
        loop {
            let (p, q) = (self.prev(n), self.next(n));
            let (ip, i, iq) = (self.pid(p), self.pid(n), self.pid(q));
            if ip != i && i != iq && ip != iq {
                fallback_any.get_or_insert(n);
                if self.diag_allowed(ip, iq, diag_ok) {
                    fallback_diag.get_or_insert(n);
                    let ar = cross(self.pt(p), self.pt(n), self.pt(q));
                    if ar > self.eps && best.is_none_or(|(_, b)| ar > b) {
                        best = Some((n, ar));
                    }
                }
            }
            n = q;
            if n == start {
                break;
            }
        }
        best.map(|b| b.0).or(fallback_diag).or(fallback_any).unwrap_or(start)
    }

    fn clip(
        &mut self,
        start: u32,
        mut remaining: usize,
        diag_ok: &mut dyn FnMut(usize, usize) -> bool,
        res: &mut Triangulation,
    ) {
        let mut ear = start;
        let mut stop = ear;
        let mut relaxed = false;
        while remaining > 3 {
            let prev = self.prev(ear);
            let next = self.next(ear);
            let (ip, ie, inx) = (self.pid(prev), self.pid(ear), self.pid(next));
            // Zero-length edges and spikes (bridge artefacts): drop without a triangle.
            if ie == inx {
                self.unlink(next);
                remaining -= 1;
                stop = ear;
                continue;
            }
            if ie == ip || ip == inx {
                self.unlink(ear);
                remaining -= 1;
                ear = prev;
                stop = ear;
                continue;
            }
            if self.is_ear(prev, ear, next, relaxed) && self.diag_allowed(ip, inx, diag_ok) {
                self.emit(prev, ear, next, &mut res.triangles);
                self.unlink(ear);
                remaining -= 1;
                ear = self.next(next);
                stop = ear;
                relaxed = false;
                continue;
            }
            ear = next;
            if ear == stop {
                if !relaxed {
                    relaxed = true;
                } else {
                    let f = self.pick_forced(ear, diag_ok);
                    let (p, q) = (self.prev(f), self.next(f));
                    if self.pid(p) != self.pid(f) && self.pid(f) != self.pid(q) && self.pid(p) != self.pid(q) {
                        self.emit(p, f, q, &mut res.triangles);
                    }
                    self.unlink(f);
                    remaining -= 1;
                    res.forced += 1;
                    ear = q;
                    stop = ear;
                    relaxed = false;
                }
            }
        }
        if remaining == 3 {
            let (p, q) = (self.prev(ear), self.next(ear));
            let (a, b, c) = (self.pid(p), self.pid(ear), self.pid(q));
            if a != b && b != c && a != c {
                res.triangles.push([a as usize, b as usize, c as usize]);
            }
        }
    }
}

/// Triangulate the polygon `outer` (counter-clockwise) with `holes` (clockwise), given as
/// indices into `pts`. Equal indices denote the same vertex. `diag_ok(i, j)` may veto a
/// new diagonal between points `i` and `j` (e.g. because that mesh edge already exists).
pub(crate) fn triangulate(
    pts: &[DVec2],
    outer: &[usize],
    holes: &[Vec<usize>],
    diag_ok: &mut dyn FnMut(usize, usize) -> bool,
) -> Triangulation {
    let mut res = Triangulation::default();
    if outer.len() < 3 {
        return res;
    }
    let mut bmin = DVec2::splat(f64::INFINITY);
    let mut bmax = DVec2::splat(f64::NEG_INFINITY);
    for &i in outer.iter().chain(holes.iter().flatten()) {
        bmin = bmin.min(pts[i]);
        bmax = bmax.max(pts[i]);
    }
    let scale = (bmax - bmin).max_element().max(1e-300);
    let mut c = Clipper { pts, nodes: Vec::new(), used: HashSet::new(), eps: 1e-13 * scale * scale, grid: None };
    let start = c.ring(outer);

    let mut hs: Vec<(usize, u32)> = Vec::new();
    for (hi, h) in holes.iter().enumerate() {
        if h.len() < 3 {
            res.unbridged_holes.push(hi);
            continue;
        }
        let first = c.ring(h);
        let mut left = first;
        let mut n = first;
        loop {
            let (p, l) = (c.pt(n), c.pt(left));
            if p.x < l.x || (p.x == l.x && p.y < l.y) {
                left = n;
            }
            n = c.next(n);
            if n == first {
                break;
            }
        }
        hs.push((hi, left));
    }
    hs.sort_by(|a, b| {
        let (pa, pb) = (c.pt(a.1), c.pt(b.1));
        pa.x.partial_cmp(&pb.x).unwrap_or(std::cmp::Ordering::Equal).then(pa.y.partial_cmp(&pb.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    for (hi, h) in hs {
        match c.find_bridge(h, start) {
            Some(m) => c.split(m, h),
            None => {
                res.unbridged_holes.push(hi);
                let mut n = h;
                loop {
                    c.nodes[n as usize].alive = false;
                    n = c.next(n);
                    if n == h {
                        break;
                    }
                }
            }
        }
    }

    let mut ring_nodes = Vec::new();
    let mut n = start;
    loop {
        ring_nodes.push(n);
        n = c.next(n);
        if n == start || ring_nodes.len() > c.nodes.len() {
            break;
        }
    }
    let mut grid = Grid::new(bmin, bmax, ring_nodes.len());
    for &n in &ring_nodes {
        grid.insert(n, c.pt(n));
    }
    c.grid = Some(grid);
    let remaining = ring_nodes.len();
    c.clip(start, remaining, diag_ok, &mut res);
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(pts: &[DVec2], t: &[usize; 3]) -> f64 {
        0.5 * cross(pts[t[0]], pts[t[1]], pts[t[2]])
    }

    #[test]
    fn square_with_hole() {
        let pts = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(4.0, 0.0),
            DVec2::new(4.0, 4.0),
            DVec2::new(0.0, 4.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(1.0, 3.0),
            DVec2::new(3.0, 3.0),
            DVec2::new(3.0, 1.0),
        ];
        let r = triangulate(&pts, &[0, 1, 2, 3], &[vec![4, 5, 6, 7]], &mut |_, _| true);
        assert_eq!(r.triangles.len(), 8);
        assert_eq!(r.forced, 0);
        let total: f64 = r.triangles.iter().map(|t| area(&pts, t)).sum();
        assert!((total - 12.0).abs() < 1e-12);
        assert!(r.triangles.iter().all(|t| area(&pts, t) > 0.0));
    }

    #[test]
    fn collinear_points_are_kept() {
        // Square with extra collinear points on every side.
        let mut pts = Vec::new();
        for i in 0..4 {
            pts.push(DVec2::new(i as f64, 0.0));
        }
        for i in 0..4 {
            pts.push(DVec2::new(4.0, i as f64));
        }
        for i in 0..4 {
            pts.push(DVec2::new(4.0 - i as f64, 4.0));
        }
        for i in 0..4 {
            pts.push(DVec2::new(0.0, 4.0 - i as f64));
        }
        let outer: Vec<usize> = (0..pts.len()).collect();
        let r = triangulate(&pts, &outer, &[], &mut |_, _| true);
        assert_eq!(r.triangles.len(), pts.len() - 2);
        let total: f64 = r.triangles.iter().map(|t| area(&pts, t)).sum();
        assert!((total - 16.0).abs() < 1e-12);
        assert!(r.triangles.iter().all(|t| area(&pts, t) > 0.0), "no degenerate ears");
    }

    #[test]
    fn many_holes_large_ring() {
        let n = 2000;
        let mut pts = Vec::new();
        for i in 0..n {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            pts.push(DVec2::new(10.0 * a.cos(), 10.0 * a.sin()));
        }
        let outer: Vec<usize> = (0..n).collect();
        let mut holes = Vec::new();
        for (cx, cy) in [(-4.0, 0.0), (4.0, 0.0), (0.0, 4.0), (0.0, -4.0)] {
            let start = pts.len();
            for i in 0..32 {
                let a = -std::f64::consts::TAU * i as f64 / 32.0;
                pts.push(DVec2::new(cx + a.cos(), cy + a.sin()));
            }
            holes.push((start..start + 32).collect::<Vec<_>>());
        }
        let r = triangulate(&pts, &outer, &holes, &mut |_, _| true);
        assert!(r.unbridged_holes.is_empty());
        assert_eq!(r.forced, 0);
        assert_eq!(r.triangles.len(), n + 4 * 32 + 2 * 4 - 2);
        let total: f64 = r.triangles.iter().map(|t| area(&pts, t)).sum();
        let outer_area = 0.5 * n as f64 * 100.0 * (std::f64::consts::TAU / n as f64).sin();
        let hole_area = 4.0 * 0.5 * 32.0 * (std::f64::consts::TAU / 32.0).sin();
        assert!((total - (outer_area - hole_area)).abs() < 1e-8);
    }
}
