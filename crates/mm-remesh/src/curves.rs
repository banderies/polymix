//! Nearest-point queries on the original feature curves (one small AABB tree per curve).

use mm_core::{DVec3, INVALID};

/// A segment of an original feature curve, with the sizing-field value at its ends.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Seg {
    pub a: DVec3,
    pub b: DVec3,
    pub sa: f64,
    pub sb: f64,
    pub curve: u32,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    min: DVec3,
    max: DVec3,
    /// Leaf: start into `order`; interior: index of the left child (right = left + 1).
    start: u32,
    /// Leaf size, 0 for interior nodes.
    count: u32,
}

const LEAF: usize = 4;

#[derive(Default)]
pub(crate) struct CurveIndex {
    segs: Vec<Seg>,
    order: Vec<u32>,
    nodes: Vec<Node>,
    roots: Vec<u32>,
}

#[inline]
fn seg_closest(p: DVec3, s: &Seg) -> (DVec3, f64) {
    let d = s.b - s.a;
    let l2 = d.length_squared();
    let t = if l2 > 0.0 { ((p - s.a).dot(d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (s.a + d * t, t)
}

#[inline]
fn box_dist2(min: DVec3, max: DVec3, p: DVec3) -> f64 {
    (min - p).max(p - max).max(DVec3::ZERO).length_squared()
}

impl CurveIndex {
    pub fn new(segs: Vec<Seg>, curve_count: u32) -> Self {
        let mut idx = CurveIndex { segs, order: Vec::new(), nodes: Vec::new(), roots: vec![INVALID; curve_count as usize] };
        let mut by_curve: Vec<Vec<u32>> = vec![Vec::new(); curve_count as usize];
        for (i, s) in idx.segs.iter().enumerate() {
            if (s.curve as usize) < by_curve.len() {
                by_curve[s.curve as usize].push(i as u32);
            }
        }
        for (c, items) in by_curve.into_iter().enumerate() {
            if !items.is_empty() {
                idx.roots[c] = idx.build(items);
            }
        }
        idx
    }

    fn build(&mut self, items: Vec<u32>) -> u32 {
        let base = self.order.len();
        let n = items.len();
        self.order.extend(items);
        let root = self.nodes.len() as u32;
        self.nodes.push(Node { min: DVec3::ZERO, max: DVec3::ZERO, start: 0, count: 0 });
        let mut stack = vec![(root as usize, base, base + n)];
        while let Some((ni, lo, hi)) = stack.pop() {
            let mut min = DVec3::splat(f64::INFINITY);
            let mut max = DVec3::splat(f64::NEG_INFINITY);
            let mut cmin = min;
            let mut cmax = max;
            for &s in &self.order[lo..hi] {
                let sg = &self.segs[s as usize];
                min = min.min(sg.a).min(sg.b);
                max = max.max(sg.a).max(sg.b);
                let c = (sg.a + sg.b) * 0.5;
                cmin = cmin.min(c);
                cmax = cmax.max(c);
            }
            if hi - lo <= LEAF {
                self.nodes[ni] = Node { min, max, start: lo as u32, count: (hi - lo) as u32 };
                continue;
            }
            let ext = cmax - cmin;
            let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
            let mid = (hi - lo) / 2;
            let segs = &self.segs;
            self.order[lo..hi].select_nth_unstable_by(mid, |&a, &b| {
                let ca = (segs[a as usize].a + segs[a as usize].b)[axis];
                let cb = (segs[b as usize].a + segs[b as usize].b)[axis];
                ca.total_cmp(&cb)
            });
            let left = self.nodes.len();
            self.nodes.push(Node { min: DVec3::ZERO, max: DVec3::ZERO, start: 0, count: 0 });
            self.nodes.push(Node { min: DVec3::ZERO, max: DVec3::ZERO, start: 0, count: 0 });
            self.nodes[ni] = Node { min, max, start: left as u32, count: 0 };
            stack.push((left, lo, lo + mid));
            stack.push((left + 1, lo + mid, hi));
        }
        root
    }

    /// Nearest point on `curve` to `p`, with the interpolated sizing value there.
    pub fn nearest(&self, curve: u32, p: DVec3) -> Option<(DVec3, f64)> {
        let root = *self.roots.get(curve as usize)?;
        if root == INVALID {
            return None;
        }
        let mut best = f64::INFINITY;
        let mut out = None;
        let mut stack: smallvec::SmallVec<[u32; 32]> = smallvec::SmallVec::new();
        stack.push(root);
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if box_dist2(n.min, n.max, p) > best {
                continue;
            }
            if n.count > 0 {
                for &s in &self.order[n.start as usize..(n.start + n.count) as usize] {
                    let sg = &self.segs[s as usize];
                    let (q, t) = seg_closest(p, sg);
                    let d2 = q.distance_squared(p);
                    if d2 < best {
                        best = d2;
                        out = Some((q, sg.sa + (sg.sb - sg.sa) * t));
                    }
                }
            } else {
                let l = &self.nodes[n.start as usize];
                let r = &self.nodes[n.start as usize + 1];
                let dl = box_dist2(l.min, l.max, p);
                let dr = box_dist2(r.min, r.max, p);
                if dl < dr {
                    stack.push(n.start + 1);
                    stack.push(n.start);
                } else {
                    stack.push(n.start);
                    stack.push(n.start + 1);
                }
            }
        }
        out
    }
}
