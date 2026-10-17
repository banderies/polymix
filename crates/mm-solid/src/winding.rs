//! Fast generalized winding numbers for triangle soups.
//!
//! Implements the hierarchical evaluation of Barill, Dickson, Schmidt, Levin and Jacobson,
//! "Fast Winding Numbers for Soups and Clouds" (SIGGRAPH 2018). The winding number
//!
//! ```text
//! w(q) = 1/(4π) Σ_t Ω_t(q) = 1/(4π) ∫_S (x − q)·n(x) / |x − q|³ dA(x)
//! ```
//!
//! is ≈1 inside a closed outward-oriented surface, ≈0 outside, and degrades gracefully
//! (smoothly, harmonically) across holes, overlaps and self-intersections.
//!
//! A bounding-volume tree is built over the triangles. Each node stores a Taylor expansion
//! of its triangles' contribution about the node's area-weighted centroid `p̃`: the dipole
//! term `Σ aₜnₜ`, the first-order tensor `D = Σ ∫(x−p̃)⊗n`, and the second-order tensor
//! `C = Σ n ⊗ ∫(x−p̃)⊗(x−p̃)` (all integrals exact for flat triangles). During evaluation a
//! node whose expansion centre is farther than `β·r` from the query (`r` = node radius)
//! uses the expansion; otherwise its children are visited, and leaves sum exact
//! Van Oosterom–Strackee solid angles.

use crate::hash::FastMap;
use glam::DVec3;
use mm_core::geom::triangle_solid_angle;
use mm_core::Mesh;
use rayon::prelude::*;
use std::f64::consts::PI;

const LEAF_SIZE: usize = 16;
const PAR_THRESHOLD: usize = 32 * 1024;
const INV_4PI: f64 = 1.0 / (4.0 * PI);

/// Packed tree node (expansion coefficients are stored relative to `center`).
#[derive(Clone, Copy, Debug, Default)]
struct WNode {
    center: DVec3,
    /// Squared radius of the smallest centre-anchored ball containing the node's triangles.
    r2: f64,
    /// Order 0: Σ aₜ nₜ.
    n: DVec3,
    /// Leaf: first triangle; interior: index of the right child (the left child is `self + 1`).
    start: u32,
    /// Triangle count for leaves, 0 for interior nodes.
    count: u32,
    /// Order 1: trace of D and coefficients of the quadratic form yᵀDy
    /// (xx, yy, zz, xy, xz, yz).
    d_tr: f32,
    d_q: [f32; 6],
    /// Order 2: `6v + 3u` with v_k = Σᵢ C_iik, u_i = Σⱼ C_ijj.
    s: [f32; 3],
    /// Order 2: coefficients of the cubic form Σ C_ijk yᵢyⱼy_k
    /// (x³, y³, z³, x²y, x²z, xy², y²z, xz², yz², xyz).
    p: [f32; 10],
}

impl WNode {
    /// Far-field contribution (in solid-angle units) at the query `q`, given `y = p̃ − q`.
    #[inline(always)]
    fn far_field(&self, y: DVec3, y2: f64) -> f64 {
        let ir2 = 1.0 / y2;
        let ir = ir2.sqrt();
        let ir3 = ir * ir2;
        let ir5 = ir3 * ir2;
        let ir7 = ir5 * ir2;
        let (x, yy, z) = (y.x, y.y, y.z);
        let q = &self.d_q;
        let quad = q[0] as f64 * x * x
            + q[1] as f64 * yy * yy
            + q[2] as f64 * z * z
            + q[3] as f64 * x * yy
            + q[4] as f64 * x * z
            + q[5] as f64 * yy * z;
        let p = &self.p;
        let (x2, y2_, z2) = (x * x, yy * yy, z * z);
        let cubic = p[0] as f64 * x2 * x
            + p[1] as f64 * y2_ * yy
            + p[2] as f64 * z2 * z
            + p[3] as f64 * x2 * yy
            + p[4] as f64 * x2 * z
            + p[5] as f64 * x * y2_
            + p[6] as f64 * y2_ * z
            + p[7] as f64 * x * z2
            + p[8] as f64 * yy * z2
            + p[9] as f64 * x * yy * z;
        let s_dot = self.s[0] as f64 * x + self.s[1] as f64 * yy + self.s[2] as f64 * z;
        self.n.dot(y) * ir3 + self.d_tr as f64 * ir3 - 3.0 * quad * ir5 + 0.5 * (15.0 * cubic * ir7 - s_dot * ir5)
    }
}

/// Expansion moments in full precision, used while building.
#[derive(Clone, Copy)]
struct Moments {
    area: f64,
    center: DVec3,
    n: DVec3,
    /// `d[j][i] = Σ ∫ (x−p̃)_j n_i dA`.
    d: [[f64; 3]; 3],
    /// `c[i][j][k] = Σ n_i ∫ (x−p̃)_j (x−p̃)_k dA`.
    c: [[[f64; 3]; 3]; 3],
}

impl Moments {
    fn zero(center: DVec3) -> Self {
        Moments { area: 0.0, center, n: DVec3::ZERO, d: [[0.0; 3]; 3], c: [[[0.0; 3]; 3]; 3] }
    }

    /// Exact moments of a set of triangles about their area-weighted centroid.
    fn of_triangles(items: &[Item]) -> Self {
        let mut area = 0.0;
        let mut acc = DVec3::ZERO;
        let mut mean = DVec3::ZERO;
        for it in items {
            let a = 0.5 * (it.tri[1] - it.tri[0]).cross(it.tri[2] - it.tri[0]).length();
            area += a;
            acc += it.centroid * a;
            mean += it.centroid;
        }
        let center = if area > 0.0 { acc / area } else { mean / items.len().max(1) as f64 };
        let mut m = Moments::zero(center);
        m.area = area;
        for it in items {
            let [v0, v1, v2] = it.tri;
            // a·n̂ = ½ (v1−v0)×(v2−v0): no normalisation needed, robust for slivers.
            let an = 0.5 * (v1 - v0).cross(v2 - v0);
            m.n += an;
            let e = [v0 - center, v1 - center, v2 - center];
            let dc = (e[0] + e[1] + e[2]) / 3.0;
            let s = e[0] + e[1] + e[2];
            // ∫ d dᵀ dA = (A/12)(Σ eₐeₐᵀ + s sᵀ); multiplied by n̂ᵢ → (an_i/12)(...).
            let mut second = [[0.0; 3]; 3];
            for j in 0..3 {
                for k in 0..3 {
                    second[j][k] = e[0][j] * e[0][k] + e[1][j] * e[1][k] + e[2][j] * e[2][k] + s[j] * s[k];
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    m.d[j][i] += dc[j] * an[i];
                }
                let w = an[i] / 12.0;
                for j in 0..3 {
                    for k in 0..3 {
                        m.c[i][j][k] += w * second[j][k];
                    }
                }
            }
        }
        m
    }

    /// Combine two child expansions about the parent's area-weighted centroid.
    fn combine(a: &Moments, b: &Moments) -> Moments {
        let area = a.area + b.area;
        let center = if area > 0.0 {
            (a.center * a.area + b.center * b.area) / area
        } else {
            (a.center + b.center) * 0.5
        };
        let mut m = Moments::zero(center);
        m.area = area;
        for ch in [a, b] {
            let dl = ch.center - center;
            m.n += ch.n;
            for j in 0..3 {
                for i in 0..3 {
                    m.d[j][i] += ch.d[j][i] + dl[j] * ch.n[i];
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        m.c[i][j][k] += ch.c[i][j][k]
                            + dl[j] * ch.d[k][i]
                            + dl[k] * ch.d[j][i]
                            + dl[j] * dl[k] * ch.n[i];
                    }
                }
            }
        }
        m
    }

    fn pack(&self, r2: f64, start: u32, count: u32) -> WNode {
        let d = &self.d;
        let c = &self.c;
        let mut v = [0.0; 3];
        let mut u = [0.0; 3];
        for k in 0..3 {
            v[k] = c[0][0][k] + c[1][1][k] + c[2][2][k];
        }
        for i in 0..3 {
            u[i] = c[i][0][0] + c[i][1][1] + c[i][2][2];
        }
        WNode {
            center: self.center,
            r2,
            n: self.n,
            start,
            count,
            d_tr: (d[0][0] + d[1][1] + d[2][2]) as f32,
            d_q: [
                d[0][0] as f32,
                d[1][1] as f32,
                d[2][2] as f32,
                (d[0][1] + d[1][0]) as f32,
                (d[0][2] + d[2][0]) as f32,
                (d[1][2] + d[2][1]) as f32,
            ],
            s: [(6.0 * v[0] + 3.0 * u[0]) as f32, (6.0 * v[1] + 3.0 * u[1]) as f32, (6.0 * v[2] + 3.0 * u[2]) as f32],
            p: [
                c[0][0][0] as f32,
                c[1][1][1] as f32,
                c[2][2][2] as f32,
                (2.0 * c[0][0][1] + c[1][0][0]) as f32,
                (2.0 * c[0][0][2] + c[2][0][0]) as f32,
                (2.0 * c[1][0][1] + c[0][1][1]) as f32,
                (2.0 * c[1][1][2] + c[2][1][1]) as f32,
                (2.0 * c[2][0][2] + c[0][2][2]) as f32,
                (2.0 * c[2][1][2] + c[1][2][2]) as f32,
                (2.0 * (c[0][1][2] + c[1][0][2] + c[2][0][1])) as f32,
            ],
        }
    }
}

#[derive(Clone, Copy)]
struct Item {
    tri: [DVec3; 3],
    centroid: DVec3,
}

/// Hierarchical fast winding number evaluator for a triangle soup.
///
/// Build once with [`WindingNumber::new`] (or [`WindingNumber::from_triangles`] for raw
/// soups), then query with [`WindingNumber::eval`] or the parallel
/// [`WindingNumber::eval_batch`]. The value is ≈1 inside and ≈0 outside a closed
/// outward-oriented surface (−1 inside an inverted one, 2 inside doubly covered regions);
/// for open surfaces it varies smoothly across the holes.
#[derive(Clone, Debug, Default)]
pub struct WindingNumber {
    nodes: Vec<WNode>,
    tris: Vec<[DVec3; 3]>,
    beta: f64,
    beta2: f64,
}

impl WindingNumber {
    /// Default Taylor-accuracy parameter (Barill et al. recommend β = 2).
    pub const DEFAULT_BETA: f64 = 2.0;

    /// Build over all triangles of `mesh` with β = 2.
    pub fn new(mesh: &Mesh) -> Self {
        Self::with_beta(mesh, Self::DEFAULT_BETA)
    }

    /// Build over all triangles of `mesh` with accuracy parameter `beta` (larger is more
    /// accurate and slower; the far field is used when the distance exceeds `beta · r`).
    pub fn with_beta(mesh: &Mesh, beta: f64) -> Self {
        let tris: Vec<[DVec3; 3]> = mesh.triangle_ids().map(|t| mesh.tri_positions(t)).collect();
        Self::from_triangles(tris, beta)
    }

    /// Build over an indexed triangle soup.
    pub fn from_soup(positions: &[DVec3], triangles: &[[u32; 3]], beta: f64) -> Self {
        let tris = triangles
            .iter()
            .map(|t| [positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]])
            .collect();
        Self::from_triangles(tris, beta)
    }

    /// Build over explicit triangle corner positions (orientation matters: outward normals
    /// are counter-clockwise when seen from outside).
    pub fn from_triangles(tris: Vec<[DVec3; 3]>, beta: f64) -> Self {
        let beta = if beta.is_finite() && beta > 0.0 { beta } else { Self::DEFAULT_BETA };
        let mut items: Vec<Item> = tris
            .into_iter()
            .filter(|t| t.iter().all(|p| p.is_finite()))
            .map(|tri| Item { tri, centroid: (tri[0] + tri[1] + tri[2]) / 3.0 })
            .collect();
        let mut wn = WindingNumber { nodes: Vec::new(), tris: Vec::new(), beta, beta2: beta * beta };
        if items.is_empty() {
            return wn;
        }
        let mut memo = FastMap::default();
        let total = node_count(items.len(), &mut memo);
        let mut nodes = vec![WNode::default(); total];
        build_rec(&mut items, &mut nodes, 0, 0, &memo);
        wn.nodes = nodes;
        wn.tris = items.into_iter().map(|it| it.tri).collect();
        wn
    }

    /// The accuracy parameter β.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Change the accuracy parameter without rebuilding.
    pub fn set_beta(&mut self, beta: f64) {
        if beta.is_finite() && beta > 0.0 {
            self.beta = beta;
            self.beta2 = beta * beta;
        }
    }

    /// Number of triangles in the tree.
    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    /// Approximate heap memory used, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<WNode>() + self.tris.len() * std::mem::size_of::<[DVec3; 3]>()
    }

    /// Fast approximate winding number at `q`.
    pub fn eval(&self, q: DVec3) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut stack = [0u32; 128];
        let mut sp = 1usize;
        let mut sum = 0.0;
        let beta2 = self.beta2;
        while sp > 0 {
            sp -= 1;
            let ni = stack[sp] as usize;
            let node = &self.nodes[ni];
            let y = node.center - q;
            let y2 = y.length_squared();
            if y2 > beta2 * node.r2 {
                sum += node.far_field(y, y2);
            } else if node.count > 0 {
                let s = node.start as usize;
                for t in &self.tris[s..s + node.count as usize] {
                    sum += triangle_solid_angle(q, t[0], t[1], t[2]);
                }
            } else {
                stack[sp] = node.start;
                stack[sp + 1] = ni as u32 + 1;
                sp += 2;
            }
        }
        sum * INV_4PI
    }

    /// Exact winding number (brute-force sum of solid angles over all triangles).
    pub fn eval_exact(&self, q: DVec3) -> f64 {
        self.tris.iter().map(|t| triangle_solid_angle(q, t[0], t[1], t[2])).sum::<f64>() * INV_4PI
    }

    /// Evaluate at many points in parallel.
    pub fn eval_batch(&self, points: &[DVec3]) -> Vec<f64> {
        points.par_iter().map(|&p| self.eval(p)).collect()
    }
}

/// Exact winding number of `mesh` at `q` by brute-force summation of triangle solid
/// angles (O(n); used as ground truth in tests).
pub fn winding_number_exact(mesh: &Mesh, q: DVec3) -> f64 {
    mesh.triangle_ids()
        .map(|t| {
            let [a, b, c] = mesh.tri_positions(t);
            triangle_solid_angle(q, a, b, c)
        })
        .sum::<f64>()
        * INV_4PI
}

fn node_count(n: usize, memo: &mut FastMap<usize, usize>) -> usize {
    if n <= LEAF_SIZE {
        return 1;
    }
    if let Some(&c) = memo.get(&n) {
        return c;
    }
    let c = 1 + node_count(n / 2, memo) + node_count(n - n / 2, memo);
    memo.insert(n, c);
    c
}

#[inline]
fn lookup_count(n: usize, memo: &FastMap<usize, usize>) -> usize {
    if n <= LEAF_SIZE {
        1
    } else {
        memo[&n]
    }
}

/// Build the subtree for `items` into `nodes` (pre-order: `nodes[0]` is this node, then the
/// left subtree, then the right subtree). Returns the subtree's moments.
fn build_rec(items: &mut [Item], nodes: &mut [WNode], node_index: u32, first_tri: u32, memo: &FastMap<usize, usize>) -> Moments {
    let n = items.len();
    if n <= LEAF_SIZE {
        let m = Moments::of_triangles(items);
        let r2 = max_dist2(items, m.center);
        nodes[0] = m.pack(r2, first_tri, n as u32);
        return m;
    }
    // Median split along the longest axis of the centroid bounds.
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for it in items.iter() {
        lo = lo.min(it.centroid);
        hi = hi.max(it.centroid);
    }
    let ext = hi - lo;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    let mid = n / 2;
    items.select_nth_unstable_by(mid, |a, b| a.centroid[axis].total_cmp(&b.centroid[axis]));
    let nl = lookup_count(mid, memo);
    let right_index = node_index + 1 + nl as u32;
    let (head, rest) = nodes.split_at_mut(1);
    let (lnodes, rnodes) = rest.split_at_mut(nl);
    let (litems, ritems) = items.split_at_mut(mid);
    let (ml, mr) = if n > PAR_THRESHOLD {
        rayon::join(
            || build_rec(litems, lnodes, node_index + 1, first_tri, memo),
            || build_rec(ritems, rnodes, right_index, first_tri + mid as u32, memo),
        )
    } else {
        (
            build_rec(litems, lnodes, node_index + 1, first_tri, memo),
            build_rec(ritems, rnodes, right_index, first_tri + mid as u32, memo),
        )
    };
    let m = Moments::combine(&ml, &mr);
    let r2 = if n > PAR_THRESHOLD {
        items.par_chunks(4096).map(|c| max_dist2(c, m.center)).reduce(|| 0.0, f64::max)
    } else {
        max_dist2(items, m.center)
    };
    head[0] = m.pack(r2, right_index, 0);
    m
}

fn max_dist2(items: &[Item], c: DVec3) -> f64 {
    let mut r2: f64 = 0.0;
    for it in items {
        for p in it.tri {
            r2 = r2.max(p.distance_squared(c));
        }
    }
    r2
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn v(&mut self, s: f64) -> DVec3 {
            DVec3::new(self.f() * 2.0 - 1.0, self.f() * 2.0 - 1.0, self.f() * 2.0 - 1.0) * s
        }
    }

    fn check_against_exact(m: &Mesh, scale: f64, tol: f64, n: usize, seed: u64) {
        let wn = WindingNumber::new(m);
        let mut rng = Rng(seed);
        let mut max_err: f64 = 0.0;
        for _ in 0..n {
            let p = rng.v(scale);
            let a = wn.eval(p);
            let b = winding_number_exact(m, p);
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < tol, "max error {max_err}");
    }

    #[test]
    fn sphere_inside_outside() {
        let m = primitives::icosphere(1.0, 4);
        let wn = WindingNumber::new(&m);
        // β = 2 gives ~1e-3 absolute accuracy (third-order convergence in 1/β).
        assert!((wn.eval(DVec3::ZERO) - 1.0).abs() < 3e-3);
        assert!((wn.eval(DVec3::new(0.3, -0.5, 0.2)) - 1.0).abs() < 3e-3);
        assert!(wn.eval(DVec3::new(3.0, 0.0, 0.0)).abs() < 1e-3);
        assert!(wn.eval(DVec3::new(1.05, 0.0, 0.1)).abs() < 3e-3);
        check_against_exact(&m, 2.0, 5e-3, 400, 7);
        // Single far triangle: the expansion itself is accurate to ~1e-8.
        let tri = vec![[DVec3::ZERO, DVec3::new(0.1, 0.0, 0.0), DVec3::new(0.0, 0.1, 0.0)]];
        let one = WindingNumber::from_triangles(tri, 2.0);
        for p in [DVec3::new(0.0, 0.0, 1.0), DVec3::new(0.5, 0.3, -1.0), DVec3::new(1.0, 1.0, 0.2)] {
            assert!((one.eval(p) - one.eval_exact(p)).abs() < 1e-8);
        }
    }

    #[test]
    fn open_and_soupy_meshes_match_brute_force() {
        // Open hemisphere-ish surface: cut away half the triangles of a sphere.
        let s = primitives::uv_sphere(1.0, 48, 24);
        let keep: Vec<u32> = s.triangle_ids().filter(|&t| s.tri_centroid(t).z > -0.3).collect();
        let (open, _) = s.submesh(&keep);
        check_against_exact(&open, 1.6, 6e-3, 400, 11);
        // Thin tube with open ends.
        let path: Vec<DVec3> = (0..40).map(|i| DVec3::new((i as f64 * 0.3).sin(), i as f64 * 0.1, 0.0)).collect();
        let radii = vec![0.08; path.len()];
        let tube = primitives::tube_along(&path, &radii, 16);
        check_against_exact(&tube, 2.5, 6e-3, 400, 13);
        // Cube with very large triangles (stresses the far-field radius bound).
        let c = primitives::cube(2.0, 1);
        check_against_exact(&c, 3.0, 6e-3, 400, 17);
    }

    #[test]
    fn accuracy_improves_with_beta() {
        let m = primitives::icosphere(1.0, 3);
        let mut wn = WindingNumber::new(&m);
        let mut rng = Rng(5);
        let pts: Vec<DVec3> = (0..300).map(|_| rng.v(1.5)).collect();
        let err = |wn: &WindingNumber| pts.iter().map(|&p| (wn.eval(p) - wn.eval_exact(p)).abs()).fold(0.0, f64::max);
        wn.set_beta(1.2);
        let e_lo = err(&wn);
        wn.set_beta(4.0);
        let e_hi = err(&wn);
        assert!(e_hi <= e_lo + 1e-12, "beta 4 error {e_hi} vs beta 1.2 error {e_lo}");
        assert!(e_hi < 5e-4, "beta 4 error {e_hi}");
        let batch = wn.eval_batch(&pts);
        for (p, b) in pts.iter().zip(batch) {
            assert_eq!(wn.eval(*p), b);
        }
    }

    #[test]
    fn inverted_and_doubled() {
        let mut m = primitives::icosphere(1.0, 2);
        m.reverse_orientation();
        let wn = WindingNumber::new(&m);
        assert!((wn.eval(DVec3::ZERO) + 1.0).abs() < 5e-3);
        let mut d = primitives::icosphere(1.0, 2);
        let d2 = primitives::icosphere(1.0, 2);
        d.append_mesh(&d2, 0);
        let wn = WindingNumber::new(&d);
        assert!((wn.eval(DVec3::new(0.1, 0.2, 0.0)) - 2.0).abs() < 1e-2);
    }
}
