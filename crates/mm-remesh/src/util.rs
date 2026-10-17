//! Small internal helpers.

use mm_core::DVec3;
use std::cmp::Ordering;

/// Wraps the optional user progress callback and remembers cancellation.
pub(crate) struct Progress<'a> {
    f: Option<&'a dyn Fn(f32) -> bool>,
    last: f32,
    pub cancelled: bool,
}

impl<'a> Progress<'a> {
    pub fn new(f: Option<&'a dyn Fn(f32) -> bool>) -> Self {
        Self { f, last: 0.0, cancelled: false }
    }
    /// Report progress (clamped to `[0, 1]` and never decreasing); returns `false` when
    /// the operation should stop.
    pub fn report(&mut self, frac: f32) -> bool {
        if self.cancelled {
            return false;
        }
        if let Some(f) = self.f {
            let x = if frac.is_finite() { frac.clamp(0.0, 1.0) } else { 0.0 };
            self.last = self.last.max(x);
            if !f(self.last) {
                self.cancelled = true;
            }
        }
        !self.cancelled
    }
    pub fn ok(&self) -> bool {
        !self.cancelled
    }
}

/// Totally ordered f64 for heaps (NaN sorts last via `total_cmp`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OrdF64(pub f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Twice the area vector of triangle (a, b, c).
#[inline]
pub(crate) fn cross3(a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    (b - a).cross(c - a)
}

/// Shape quality in `[0, 1]`: `4·√3·area / Σ edge²` (1 = equilateral, 0 = degenerate).
#[inline]
pub(crate) fn quality(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let s = (b - a).length_squared() + (c - b).length_squared() + (a - c).length_squared();
    if s <= 0.0 {
        return 0.0;
    }
    2.0 * 3f64.sqrt() * cross3(a, b, c).length() / s
}

/// Reusable "visited" marks over a growing id range, cleared in O(1) by bumping a stamp.
#[derive(Default)]
pub(crate) struct Stamp {
    marks: Vec<u32>,
    cur: u32,
}

impl Stamp {
    pub fn begin(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            self.marks.iter_mut().for_each(|m| *m = 0);
            self.cur = 1;
        }
    }
    /// Mark `i`; returns true if it was not yet marked in this round.
    #[inline]
    pub fn mark(&mut self, i: u32) -> bool {
        let i = i as usize;
        if i >= self.marks.len() {
            self.marks.resize(i + 1, 0);
        }
        if self.marks[i] == self.cur {
            false
        } else {
            self.marks[i] = self.cur;
            true
        }
    }
}

/// Triangles below this quality are treated as degenerate (normal undefined).
pub(crate) const DEGENERATE_Q: f64 = 1e-6;

/// Shape test for collapsing `rem` into `keep` with `keep` moved to `pos` (`t0`, `t1`
/// are the triangles of the collapsed edge). Rejects degenerate results, normals rotating
/// below `min_dot` (degenerate old triangles are compared with the fan normal instead),
/// and new minimum quality below `min_q` when that is worse than before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collapse_shape_ok(
    mesh: &mm_core::Mesh,
    keep: u32,
    rem: u32,
    pos: DVec3,
    t0: u32,
    t1: u32,
    min_dot: f64,
    min_q: f64,
) -> bool {
    let moved_keep = mesh.vertex(keep) != pos;
    let mut tris: smallvec::SmallVec<[u32; 32]> = smallvec::SmallVec::new();
    for t in mesh.vertex_triangles(rem) {
        if t != t0 && t != t1 {
            tris.push(t);
        }
    }
    if moved_keep {
        for t in mesh.vertex_triangles(keep) {
            if t != t0 && t != t1 {
                tris.push(t);
            }
        }
    }
    if tris.is_empty() {
        return false;
    }
    let fan_n = tris.iter().map(|&t| mesh.tri_cross(t)).sum::<DVec3>().normalize_or_zero();
    let mut old_min_q = f64::INFINITY;
    let mut new_min_q = f64::INFINITY;
    for &t in &tris {
        let tv = mesh.triangle(t);
        let o = [mesh.vertex(tv[0]), mesh.vertex(tv[1]), mesh.vertex(tv[2])];
        let mut n = o;
        for k in 0..3 {
            if tv[k] == rem || tv[k] == keep {
                n[k] = pos;
            }
        }
        let qn = quality(n[0], n[1], n[2]);
        let qo = quality(o[0], o[1], o[2]);
        if qn < DEGENERATE_Q {
            if qo >= DEGENERATE_Q {
                return false;
            }
            // Degenerate before and after (coincident-vertex clusters): not made worse in
            // any meaningful way, and collapsing is what eventually removes it.
            continue;
        }
        old_min_q = old_min_q.min(qo);
        new_min_q = new_min_q.min(qn);
        let nn = cross3(n[0], n[1], n[2]).normalize_or_zero();
        let rn = if qo > 1e-3 { cross3(o[0], o[1], o[2]).normalize_or_zero() } else { fan_n };
        if rn != DVec3::ZERO && nn.dot(rn) < min_dot {
            return false;
        }
    }
    !(new_min_q < min_q && new_min_q < old_min_q)
}
