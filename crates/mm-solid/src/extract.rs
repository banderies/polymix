//! Manifold iso-surface extraction by marching tetrahedra on the Kuhn triangulation.
//!
//! Every lattice cube is split into the six tetrahedra that share its main diagonal
//! (corner 0 → corner 7). Because each cube face is split along the diagonal through its
//! lowest and highest corners, neighbouring cubes agree on the face triangulation and the
//! tetrahedra form a conforming simplicial complex (the Freudenthal triangulation). Its
//! edges are the lattice edges in the 7 directions with components in {0, 1}.
//!
//! The zero set of the piecewise-linear interpolant of nonzero vertex values on a
//! simplicial complex is a 2-manifold without boundary (inside the domain), so welding
//! one vertex per crossing edge yields:
//! * every mesh edge lies in a tetrahedron face (shared by exactly 2 tets) or inside a
//!   tetrahedron (quad diagonal) → exactly two incident triangles;
//! * the triangles around a vertex are those of the tets around its lattice edge, which
//!   form a single cycle → no bowties;
//! * orientation is derived combinatorially from positively oriented tetrahedra, so normals
//!   always point from negative (inside) to positive (outside) values.
//!
//! Stored values are never zero (see `grid::quantize`) and the outermost sample layer is
//! positive, so the result is closed.
//!
//! The extraction runs in two parallel passes over bricks: (1) each active brick creates
//! the vertices of crossing edges whose lower endpoint it owns (a 3584-bit table with
//! prefix counts gives each crossing edge a rank); (2) every brick whose cells may contain
//! crossings emits triangles, resolving edge ids through the owning brick's table.

use crate::grid::{lidx, SparseField, Slot};
use crate::progress::Prog;
use crate::{Result, SolidError};
use glam::DVec3;
use mm_core::Mesh;
use rayon::prelude::*;

const EDGE_WORDS: usize = 512 * 7 / 64; // 56

/// The 6 Kuhn tetrahedra (cube-corner bitmasks x=1, y=2, z=4), positively oriented.
fn kuhn_tets() -> [[u8; 4]; 6] {
    let raw: [[u8; 4]; 6] = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];
    let mut out = raw;
    for t in out.iter_mut() {
        let p = |c: u8| DVec3::new((c & 1) as f64, ((c >> 1) & 1) as f64, ((c >> 2) & 1) as f64);
        let det = (p(t[1]) - p(t[0])).dot((p(t[2]) - p(t[0])).cross(p(t[3]) - p(t[0])));
        if det < 0.0 {
            t.swap(2, 3);
        }
    }
    out
}

/// Triangulation of one tetrahedron case, as cube-corner edge pairs `(lo, hi)` with
/// `lo ⊂ hi` (bitwise).
#[derive(Clone, Copy, Debug)]
enum TetCase {
    Empty,
    Tri([(u8, u8); 3]),
    /// Quad cycle; split along the shorter diagonal at runtime.
    Quad([(u8, u8); 4]),
}

fn parity4(p: [usize; 4]) -> bool {
    let mut inv = 0;
    for i in 0..4 {
        for j in i + 1..4 {
            if p[i] > p[j] {
                inv += 1;
            }
        }
    }
    inv % 2 == 0
}

fn ordered(a: u8, b: u8) -> (u8, u8) {
    if a & b == a {
        (a, b)
    } else {
        debug_assert_eq!(a & b, b);
        (b, a)
    }
}

/// Case table for all 6 tets × 16 inside-masks.
fn case_table() -> [[TetCase; 16]; 6] {
    let tets = kuhn_tets();
    let mut table = [[TetCase::Empty; 16]; 6];
    for (ti, tet) in tets.iter().enumerate() {
        for mask in 0..16usize {
            let inside: Vec<usize> = (0..4).filter(|&i| mask >> i & 1 == 1).collect();
            let outside: Vec<usize> = (0..4).filter(|&i| mask >> i & 1 == 0).collect();
            // Even permutation with the distinguished vertices first.
            let perm = |first: &[usize], rest: &[usize]| -> [usize; 4] {
                let mut p = [0usize; 4];
                for (k, &v) in first.iter().chain(rest).enumerate() {
                    p[k] = v;
                }
                if !parity4(p) {
                    p.swap(2, 3);
                }
                p
            };
            let e = |i: usize, j: usize| ordered(tet[i], tet[j]);
            table[ti][mask] = match inside.len() {
                1 => {
                    let [a, b, c, d] = perm(&inside, &outside);
                    TetCase::Tri([e(a, b), e(a, c), e(a, d)])
                }
                3 => {
                    let [a, b, c, d] = perm(&outside, &inside);
                    TetCase::Tri([e(a, b), e(a, d), e(a, c)])
                }
                2 => {
                    let [a, b, c, d] = perm(&inside, &outside);
                    TetCase::Quad([e(a, c), e(a, d), e(b, d), e(b, c)])
                }
                _ => TetCase::Empty,
            };
        }
    }
    table
}

/// Per-brick crossing-edge table from pass 1.
struct EdgeTable {
    bits: [u64; EDGE_WORDS],
    prefix: [u32; EDGE_WORDS],
    positions: Vec<DVec3>,
}

impl EdgeTable {
    #[inline]
    fn rank(&self, code: usize) -> Option<u32> {
        let w = code >> 6;
        let b = code & 63;
        if (self.bits[w] >> b) & 1 == 0 {
            return None;
        }
        Some(self.prefix[w] + (self.bits[w] & ((1u64 << b) - 1)).count_ones())
    }
}

#[inline]
fn corner_off(c: u8) -> [usize; 3] {
    [(c & 1) as usize, ((c >> 1) & 1) as usize, ((c >> 2) & 1) as usize]
}

#[inline]
fn g9(x: usize, y: usize, z: usize) -> usize {
    x + 9 * y + 81 * z
}

/// Extract the zero set of `field` as an indexed triangle list (outward oriented,
/// closed, edge-manifold, bowtie-free).
pub fn marching_tetrahedra(field: &SparseField) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    marching_tetrahedra_with(field, &Prog::none()).expect("cannot be cancelled without a callback")
}

pub(crate) fn marching_tetrahedra_with(field: &SparseField, prog: &Prog) -> Result<(Vec<DVec3>, Vec<[u32; 3]>)> {
    let grid = *field.grid();
    let h = grid.cell_size;
    let nbricks = field.active_bricks();
    let far = field.far();

    // Pass 1: vertices on crossing edges owned by each active brick.
    let tables: Vec<EdgeTable> = (0..nbricks as u32)
        .into_par_iter()
        .map(|bi| {
            let b = field.brick_coord(bi);
            let mut v = [0i16; 729];
            field.gather9(b, &mut v);
            let mut t = EdgeTable { bits: [0; EDGE_WORDS], prefix: [0; EDGE_WORDS], positions: Vec::new() };
            let base = [b[0] * 8, b[1] * 8, b[2] * 8];
            for z in 0..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        let fp = v[g9(x, y, z)];
                        let lp = lidx(x, y, z);
                        for dir in 0..7usize {
                            let m = dir + 1;
                            let (dx, dy, dz) = (m & 1, (m >> 1) & 1, (m >> 2) & 1);
                            let fq = v[g9(x + dx, y + dy, z + dz)];
                            if (fp < 0) != (fq < 0) {
                                let code = lp * 7 + dir;
                                t.bits[code >> 6] |= 1 << (code & 63);
                                let pp = grid.sample_point([base[0] + x as u32, base[1] + y as u32, base[2] + z as u32]);
                                let s = fp as f64 / (fp as f64 - fq as f64);
                                let d = DVec3::new(dx as f64, dy as f64, dz as f64) * h;
                                t.positions.push(pp + d * s);
                            }
                        }
                    }
                }
            }
            let mut acc = 0u32;
            for w in 0..EDGE_WORDS {
                t.prefix[w] = acc;
                acc += t.bits[w].count_ones();
            }
            t
        })
        .collect();
    let _ = far;
    prog.report(0.4)?;
    let mut base = Vec::with_capacity(nbricks);
    let mut total = 0u32;
    for t in &tables {
        base.push(total);
        total += t.positions.len() as u32;
    }

    // Bricks whose cells may contain crossings: active bricks and their −{0,1}³ neighbours.
    let mut cell_bricks: Vec<u64> = field
        .active_brick_coords()
        .par_iter()
        .flat_map_iter(|&b| {
            (0..8u32).filter_map(move |o| {
                let c = [b[0] as i64 - (o & 1) as i64, b[1] as i64 - ((o >> 1) & 1) as i64, b[2] as i64 - ((o >> 2) & 1) as i64];
                (c.iter().all(|&x| x >= 0)).then(|| crate::grid::key([c[0] as u32, c[1] as u32, c[2] as u32]))
            })
        })
        .collect();
    cell_bricks.par_sort_unstable();
    cell_bricks.dedup();
    prog.report(0.5)?;

    let table = case_table();
    let tets = kuhn_tets();
    let failures = std::sync::atomic::AtomicUsize::new(0);

    // Pass 2: triangles.
    let tri_lists: Vec<Vec<[u32; 3]>> = cell_bricks
        .par_iter()
        .map(|&k| {
            let b = crate::grid::unkey(k);
            let mut v = [0i16; 729];
            field.gather9(b, &mut v);
            let first = v[0] < 0;
            if v.iter().all(|&q| (q < 0) == first) {
                return Vec::new();
            }
            // Edge tables of the 8 potential owners.
            let mut owners: [Option<(u32, &EdgeTable)>; 8] = [None; 8];
            for o in 0..8u32 {
                let nb = [b[0] as i64 + (o & 1) as i64, b[1] as i64 + ((o >> 1) & 1) as i64, b[2] as i64 + ((o >> 2) & 1) as i64];
                if let Slot::Active(i) = field.slot(nb) {
                    owners[o as usize] = Some((base[i as usize], &tables[i as usize]));
                }
            }
            let vid = |p: [usize; 3], dir: usize| -> Option<(u32, DVec3)> {
                let o = (p[0] >> 3) | ((p[1] >> 3) << 1) | ((p[2] >> 3) << 2);
                let (bs, t) = owners[o]?;
                let code = lidx(p[0] & 7, p[1] & 7, p[2] & 7) * 7 + dir;
                let r = t.rank(code)?;
                Some((bs + r, t.positions[r as usize]))
            };
            let mut out = Vec::new();
            for z in 0..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        let mut cmask = 0u8;
                        for c in 0..8u8 {
                            let o = corner_off(c);
                            if v[g9(x + o[0], y + o[1], z + o[2])] < 0 {
                                cmask |= 1 << c;
                            }
                        }
                        if cmask == 0 || cmask == 255 {
                            continue;
                        }
                        for (ti, tet) in tets.iter().enumerate() {
                            let mut tm = 0usize;
                            for (i, &c) in tet.iter().enumerate() {
                                if cmask >> c & 1 == 1 {
                                    tm |= 1 << i;
                                }
                            }
                            let edge = |(lo, hi): (u8, u8)| {
                                let o = corner_off(lo);
                                vid([x + o[0], y + o[1], z + o[2]], (lo ^ hi) as usize - 1)
                            };
                            match table[ti][tm] {
                                TetCase::Empty => {}
                                TetCase::Tri(es) => match (edge(es[0]), edge(es[1]), edge(es[2])) {
                                    (Some(a), Some(b), Some(c)) => out.push([a.0, b.0, c.0]),
                                    _ => {
                                        failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                },
                                TetCase::Quad(es) => match (edge(es[0]), edge(es[1]), edge(es[2]), edge(es[3])) {
                                    (Some(ac), Some(ad), Some(bd), Some(bc)) => {
                                        if ac.1.distance_squared(bd.1) <= ad.1.distance_squared(bc.1) {
                                            out.push([ac.0, ad.0, bd.0]);
                                            out.push([ac.0, bd.0, bc.0]);
                                        } else {
                                            out.push([ad.0, bd.0, bc.0]);
                                            out.push([ad.0, bc.0, ac.0]);
                                        }
                                    }
                                    _ => {
                                        failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                },
                            }
                        }
                    }
                }
            }
            out
        })
        .collect();
    prog.report(0.9)?;
    let nf = failures.into_inner();
    if nf > 0 {
        return Err(SolidError::Internal(format!("{nf} crossing edges without an owning active brick")));
    }
    let mut positions = Vec::with_capacity(total as usize);
    for t in tables {
        positions.extend(t.positions);
    }
    let ntri: usize = tri_lists.iter().map(|l| l.len()).sum();
    let mut tris = Vec::with_capacity(ntri);
    for l in tri_lists {
        tris.extend(l);
    }
    prog.report(1.0)?;
    Ok((positions, tris))
}

/// Build an mm-core [`Mesh`] from extraction output, verifying that no edge needed
/// non-manifold splitting.
pub(crate) fn build_mesh(positions: &[DVec3], tris: &[[u32; 3]]) -> Result<Mesh> {
    let (m, rep) = Mesh::from_triangles(positions, tris, None);
    if rep.nonmanifold_split > 0 || rep.duplicate > 0 || rep.degenerate > 0 {
        return Err(SolidError::Internal(format!("extraction produced a non-manifold mesh: {rep:?}")));
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kuhn_tets_are_positive_and_fill_the_cube() {
        let tets = kuhn_tets();
        let mut vol = 0.0;
        for t in tets {
            let p = |c: u8| DVec3::new((c & 1) as f64, ((c >> 1) & 1) as f64, ((c >> 2) & 1) as f64);
            let det = (p(t[1]) - p(t[0])).dot((p(t[2]) - p(t[0])).cross(p(t[3]) - p(t[0])));
            assert!(det > 0.0);
            vol += det / 6.0;
            // All tet edges are lattice edges with nonnegative direction.
            for i in 0..4 {
                for j in i + 1..4 {
                    let (lo, hi) = ordered(t[i], t[j]);
                    assert_eq!(lo & hi, lo);
                }
            }
        }
        assert!((vol - 1.0).abs() < 1e-12);
    }

    #[test]
    fn case_table_orientation() {
        // For each single-inside case the triangle normal must point away from the inside vertex.
        let tets = kuhn_tets();
        let table = case_table();
        let p = |c: u8| DVec3::new((c & 1) as f64, ((c >> 1) & 1) as f64, ((c >> 2) & 1) as f64);
        for (ti, tet) in tets.iter().enumerate() {
            for mask in 1..15usize {
                let mid = |e: (u8, u8)| (p(e.0) + p(e.1)) * 0.5;
                let inside_c: DVec3 =
                    (0..4).filter(|&i| mask >> i & 1 == 1).map(|i| p(tet[i])).sum::<DVec3>() / (mask.count_ones() as f64);
                let outside_c: DVec3 =
                    (0..4).filter(|&i| mask >> i & 1 == 0).map(|i| p(tet[i])).sum::<DVec3>() / ((4 - mask.count_ones()) as f64);
                let dir = outside_c - inside_c;
                let n = match table[ti][mask] {
                    TetCase::Tri(es) => (mid(es[1]) - mid(es[0])).cross(mid(es[2]) - mid(es[0])),
                    TetCase::Quad(es) => (mid(es[1]) - mid(es[0])).cross(mid(es[2]) - mid(es[0])),
                    TetCase::Empty => panic!("non-trivial case without triangles"),
                };
                assert!(n.dot(dir) > 0.0, "tet {ti} mask {mask}");
            }
        }
    }
}
