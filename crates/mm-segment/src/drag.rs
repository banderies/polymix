//! Growing a region from one voxel through the voxels a test allows, nearest first by path
//! length in mm, as far as a drag reaches (Threshold grow, Select part). The search goes only
//! as far as asked and continues from there when asked for more, so a press costs nothing on
//! a big volume and each frame of a drag searches just the new shell.
//!
//! Paths step to any of the 26 neighbours, but diagonally only where the voxels the step
//! passes also hold a path through faces: distances come out close to straight lines, while
//! the region stays connected through faces, as flood fill's is, and never leaks between
//! structures that only touch at an edge or a corner.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A voxel and its tentative path length; the heap pops the nearest first.
#[derive(PartialEq)]
struct Item(f32, usize);

impl Eq for Item {}

impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0)
    }
}

pub struct DragGrow {
    dims: [usize; 3],
    spacing: [f64; 3],
    /// Best path length found so far plus one, per voxel; 0 where the search has not been.
    /// Zeroed memory costs nothing until written, so only what the search reaches is paid for.
    best: Vec<f32>,
    heap: BinaryHeap<Item>,
    /// Voxels whose path length is final, nearest first.
    settled: Vec<(f32, usize)>,
}

impl DragGrow {
    /// A search from voxel `seed` (an index into a grid of `dims`, voxel size `spacing` mm),
    /// which the test given to [`DragGrow::reach`] must allow.
    pub fn new(dims: [usize; 3], spacing: [f64; 3], seed: usize) -> Self {
        let mut best = vec![0.0f32; dims.iter().product()];
        let mut heap = BinaryHeap::new();
        if seed < best.len() {
            best[seed] = 1.0;
            heap.push(Item(0.0, seed));
        }
        Self { dims, spacing, best, heap, settled: Vec::new() }
    }

    /// Search until every voxel within `reach` mm of the seed (along allowed voxels) is
    /// settled; `allowed` must give the same answers on every call. Returns how many voxels
    /// that is: the first so many of [`DragGrow::settled`].
    pub fn reach(&mut self, reach: f64, allowed: impl Fn(usize) -> bool) -> usize {
        let reach = reach as f32;
        let [nx, ny, nz] = self.dims;
        let plane = nx * ny;
        let s = self.spacing.map(|x| x as f32);
        let size = [nx, ny, nz];
        let stride = [1isize, nx as isize, plane as isize];
        // Every step: its offset along each axis (−1, 0 or 1), the axes it moves along, and its
        // length.
        let steps: Vec<([isize; 3], Vec<usize>, f32)> = (0..27)
            .map(|k| [k % 3 - 1, (k / 3) % 3 - 1, k / 9 - 1])
            .filter(|o| *o != [0; 3])
            .map(|o| (o, (0..3).filter(|&a| o[a] != 0).collect(), (0..3).map(|a| (o[a] as f32 * s[a]).powi(2)).sum::<f32>().sqrt()))
            .collect();
        while self.heap.peek().is_some_and(|i| i.0 <= reach) {
            let Item(d, v) = self.heap.pop().unwrap();
            if d + 1.0 > self.best[v] {
                continue;
            }
            self.settled.push((d, v));
            let c = [v % nx, (v / nx) % ny, v / plane];
            let offset = |o: [isize; 3]| (v as isize + o[0] * stride[0] + o[1] * stride[1] + o[2] * stride[2]) as usize;
            // A path through faces from `v` to `v + o` within the cube they span: the axes
            // taken one at a time in some order, every voxel on the way allowed.
            let through_faces = |o: [isize; 3], axes: &[usize]| -> bool {
                let along = |taken: &[usize]| {
                    let mut p = [0isize; 3];
                    for &a in taken {
                        p[a] = o[a];
                    }
                    allowed(offset(p))
                };
                match *axes {
                    [_] => true,
                    [a, b] => along(&[a]) || along(&[b]),
                    [a, b, c] => [[a, b], [a, c], [b, a], [b, c], [c, a], [c, b]].iter().any(|order| along(&order[..1]) && along(order)),
                    _ => false,
                }
            };
            for (o, axes, step) in &steps {
                let (o, step) = (*o, *step);
                let inside = (0..3).all(|a| {
                    let x = c[a] as isize + o[a];
                    x >= 0 && (x as usize) < size[a]
                });
                if !inside {
                    continue;
                }
                let n = offset(o);
                let nd = d + step;
                if (self.best[n] == 0.0 || nd + 1.0 < self.best[n]) && allowed(n) && through_faces(o, axes) {
                    self.best[n] = nd + 1.0;
                    self.heap.push(Item(nd, n));
                }
            }
        }
        self.settled.partition_point(|&(d, _)| d <= reach)
    }

    /// The voxels settled so far with their path lengths (mm), nearest first.
    pub fn settled(&self) -> &[(f32, usize)] {
        &self.settled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path lengths from `seed` to every voxel `allowed` reaches, searched all at once.
    fn full(dims: [usize; 3], spacing: [f64; 3], seed: usize, allowed: &dyn Fn(usize) -> bool) -> Vec<f32> {
        let mut g = DragGrow::new(dims, spacing, seed);
        g.reach(f64::INFINITY, allowed);
        let mut d = vec![f32::INFINITY; dims.iter().product()];
        for &(x, v) in g.settled() {
            d[v] = x;
        }
        d
    }

    #[test]
    fn grows_nearest_first_and_only_as_far_as_asked() {
        let dims = [40, 30, 20];
        let n: usize = dims.iter().product();
        let seed = 20 + 15 * 40 + 10 * 1200;
        let all = |_: usize| true;
        let mut g = DragGrow::new(dims, [1.0, 1.0, 2.0], seed);
        assert_eq!(g.reach(0.0, all), 1, "the seed alone");
        let k = g.reach(3.0, all);
        assert!(k > 1 && g.settled().len() < n / 20, "a short reach searches a little: {} of {n}", g.settled().len());
        assert!(g.settled()[..k].windows(2).all(|w| w[0].0 <= w[1].0), "nearest first");
        // Close to straight-line distances: exact along the steps' directions, at most about
        // 12% longer between them on this grid of voxels twice as deep as they are wide.
        let at = |x: usize, y: usize, z: usize| x + y * 40 + z * 1200;
        let d = full(dims, [1.0, 1.0, 2.0], seed, &all);
        for (v, straight, exact) in [
            (at(30, 15, 10), 10.0, true),
            (at(27, 22, 10), 98f32.sqrt(), true),
            (at(20, 15, 14), 8.0, true),
            (at(22, 17, 12), 24f32.sqrt(), true),
            (at(24, 19, 12), 48f32.sqrt(), false),
            (at(26, 21, 13), 108f32.sqrt(), false),
        ] {
            let most = if exact { straight * 1.0001 } else { straight * 1.12 };
            assert!(d[v] >= straight * 0.9999 && d[v] <= most, "{} vs {straight}", d[v]);
        }
        // Asking in steps, out and back, gives what one search to the end does.
        let mut g = DragGrow::new(dims, [1.0, 1.0, 2.0], seed);
        for r in [2.0, 7.5, 1.0, 12.0, 4.0] {
            let k = g.reach(r, all);
            let mut got: Vec<usize> = g.settled()[..k].iter().map(|s| s.1).collect();
            got.sort_unstable();
            let want: Vec<usize> = (0..n).filter(|&v| d[v] as f64 <= r).collect();
            assert_eq!(got, want, "reach {r}");
        }
    }

    #[test]
    fn stays_in_the_allowed_voxels_and_never_cuts_corners() {
        // Two bars that touch only along an edge: a diagonal step would cross between them.
        let dims = [12, 12, 3];
        let at = |x: usize, y: usize, z: usize| x + y * 12 + z * 144;
        let bar_a = |v: usize| v % 12 <= 5 && (v / 12) % 12 <= 5;
        let bar_b = |v: usize| v % 12 >= 6 && (v / 12) % 12 >= 6;
        let allowed = |v: usize| bar_a(v) || bar_b(v);
        let mut g = DragGrow::new(dims, [1.0; 3], at(1, 1, 1));
        let k = g.reach(1e9, allowed);
        assert!(g.settled()[..k].iter().all(|&(_, v)| bar_a(v)), "only the first bar");
        assert_eq!(k, 6 * 6 * 3);
        // A gap in the allowed voxels is a wall: the path goes around it.
        let wall = |v: usize| !(v % 12 == 5 && (v / 12) % 12 <= 8);
        let d = full(dims, [1.0; 3], at(2, 2, 1), &wall);
        assert!(d[at(8, 2, 1)] > 12.0, "around the wall, not through it: {}", d[at(8, 2, 1)]);
        assert!(d[at(5, 4, 1)].is_infinite(), "the wall itself is never reached");
    }
}
