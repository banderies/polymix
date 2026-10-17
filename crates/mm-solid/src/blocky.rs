//! Blocky surface: the boundary faces of the voxels whose centres are inside.
//!
//! A raw voxel boundary is non-manifold wherever two voxels touch only along an edge or
//! at a corner. Before emitting faces the voxel set is made *well-composed* (Latecki):
//! every critical 2×2 configuration (diagonal pair along an edge) and critical 2×2×2
//! configuration (opposite corners only, in either colour) is removed by filling a voxel.
//! Filling only grows the solid, so the process terminates, and the boundary of a
//! well-composed set is a closed 2-manifold.

use crate::grid::{key, lidx, unkey, SparseField, Slot};
use crate::hash::FastMap;
use crate::progress::Prog;
use crate::{Result, SolidError};
use glam::DVec3;
use rayon::prelude::*;

struct Occupancy<'a> {
    field: &'a SparseField,
    bricks: FastMap<u64, [u64; 8]>,
    samples: [i64; 3],
}

impl<'a> Occupancy<'a> {
    fn new(field: &'a SparseField) -> Self {
        let mut bricks = FastMap::default();
        bricks.reserve(field.active_bricks());
        for i in 0..field.active_bricks() as u32 {
            let v = field.brick_values(i);
            let mut bits = [0u64; 8];
            for (l, &q) in v.iter().enumerate() {
                if q < 0 {
                    bits[l >> 6] |= 1 << (l & 63);
                }
            }
            bricks.insert(key(field.brick_coord(i)), bits);
        }
        let s = field.grid().samples();
        Self { field, bricks, samples: [s[0] as i64, s[1] as i64, s[2] as i64] }
    }

    #[inline]
    fn in_domain(&self, p: [i64; 3]) -> bool {
        (0..3).all(|a| p[a] >= 0 && p[a] < self.samples[a])
    }

    #[inline]
    fn inside(&self, p: [i64; 3]) -> bool {
        if !self.in_domain(p) {
            return false;
        }
        let b = [(p[0] >> 3) as u32, (p[1] >> 3) as u32, (p[2] >> 3) as u32];
        let l = lidx((p[0] & 7) as usize, (p[1] & 7) as usize, (p[2] & 7) as usize);
        match self.bricks.get(&key(b)) {
            Some(bits) => (bits[l >> 6] >> (l & 63)) & 1 == 1,
            None => matches!(self.field.slot([b[0] as i64, b[1] as i64, b[2] as i64]), Slot::Uniform(true)),
        }
    }

    #[inline]
    fn materialized(&self, p: [i64; 3]) -> bool {
        self.in_domain(p) && self.bricks.contains_key(&key([(p[0] >> 3) as u32, (p[1] >> 3) as u32, (p[2] >> 3) as u32]))
    }

    fn fill(&mut self, p: [i64; 3]) {
        let b = [(p[0] >> 3) as u32, (p[1] >> 3) as u32, (p[2] >> 3) as u32];
        let field = self.field;
        let bits = self.bricks.entry(key(b)).or_insert_with(|| {
            let inside = matches!(field.slot([b[0] as i64, b[1] as i64, b[2] as i64]), Slot::Uniform(true));
            [if inside { u64::MAX } else { 0 }; 8]
        });
        let l = lidx((p[0] & 7) as usize, (p[1] & 7) as usize, (p[2] & 7) as usize);
        bits[l >> 6] |= 1 << (l & 63);
    }
}

#[inline]
fn off(c: usize) -> [i64; 3] {
    [(c & 1) as i64, ((c >> 1) & 1) as i64, ((c >> 2) & 1) as i64]
}

/// For the 2×2×2 block with voxel states `v` (index = corner bitmask), return the corner
/// to fill if the block contains a critical configuration.
fn critical_fix(v: [bool; 8]) -> Option<usize> {
    // 2×2 squares: the 4 voxels with bit `a` equal to `l`, in cyclic order.
    for a in 0..3 {
        let (u, w) = ((a + 1) % 3, (a + 2) % 3);
        for l in 0..2 {
            let base = l << a;
            let cyc = [base, base | (1 << u), base | (1 << u) | (1 << w), base | (1 << w)];
            let s = cyc.map(|c| v[c]);
            if s[0] == s[2] && s[1] == s[3] && s[0] != s[1] {
                return cyc.into_iter().find(|&c| !v[c]);
            }
        }
    }
    let n = v.iter().filter(|&&x| x).count();
    if n == 2 {
        let c = (0..8).find(|&c| v[c]).unwrap();
        if v[c ^ 7] {
            return Some(c ^ 1);
        }
    } else if n == 6 {
        let c = (0..8).find(|&c| !v[c]).unwrap();
        if !v[c ^ 7] {
            return Some(c);
        }
    }
    None
}

/// Extract the blocky voxel-face surface of `field` (closed, 2-manifold, outward).
pub fn voxel_faces(field: &SparseField) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    voxel_faces_with(field, &Prog::none()).expect("cannot be cancelled without a callback")
}

pub(crate) fn voxel_faces_with(field: &SparseField, prog: &Prog) -> Result<(Vec<DVec3>, Vec<[u32; 3]>)> {
    let grid = *field.grid();
    let h = grid.cell_size;
    let mut occ = Occupancy::new(field);

    // Blocks whose min corner lies in an active brick or a −{0,1}³ neighbour.
    let mut cand: Vec<u64> = field
        .active_brick_coords()
        .iter()
        .flat_map(|&b| {
            (0..8usize).filter_map(move |o| {
                let d = off(o);
                let c = [b[0] as i64 - d[0], b[1] as i64 - d[1], b[2] as i64 - d[2]];
                c.iter().all(|&x| x >= 0).then(|| key([c[0] as u32, c[1] as u32, c[2] as u32]))
            })
        })
        .collect();
    cand.par_sort_unstable();
    cand.dedup();
    let occ_ref = &occ;
    let mut work: Vec<[i64; 3]> = cand
        .par_iter()
        .flat_map_iter(|&k| {
            let b = unkey(k);
            let mut out = Vec::new();
            for z in 0..8i64 {
                for y in 0..8i64 {
                    for x in 0..8i64 {
                        let p = [b[0] as i64 * 8 + x, b[1] as i64 * 8 + y, b[2] as i64 * 8 + z];
                        let v: [bool; 8] = std::array::from_fn(|c| {
                            let d = off(c);
                            occ_ref.inside([p[0] + d[0], p[1] + d[1], p[2] + d[2]])
                        });
                        if v.iter().any(|&s| s != v[0]) && critical_fix(v).is_some() {
                            out.push(p);
                        }
                    }
                }
            }
            out.into_iter()
        })
        .collect();
    prog.report(0.3)?;

    let budget = 64 * (field.active_bricks() * 512).max(1024);
    let mut fills = 0usize;
    while let Some(p) = work.pop() {
        let v: [bool; 8] = std::array::from_fn(|c| {
            let d = off(c);
            occ.inside([p[0] + d[0], p[1] + d[1], p[2] + d[2]])
        });
        let Some(c) = critical_fix(v) else { continue };
        let d = off(c);
        let q = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
        if !occ.in_domain(q) {
            continue;
        }
        occ.fill(q);
        fills += 1;
        if fills > budget {
            return Err(SolidError::Internal("voxel well-composedness repair did not converge".into()));
        }
        for o in 0..8 {
            let d = off(o);
            work.push([q[0] - d[0], q[1] - d[1], q[2] - d[2]]);
        }
    }
    prog.report(0.6)?;

    // Faces.
    let keys: Vec<u64> = occ.bricks.keys().copied().collect();
    let occ_ref = &occ;
    let faces: Vec<([i64; 3], usize, bool)> = keys
        .par_iter()
        .flat_map_iter(|&k| {
            let b = unkey(k);
            let mut out = Vec::new();
            for z in 0..8i64 {
                for y in 0..8i64 {
                    for x in 0..8i64 {
                        let g = [b[0] as i64 * 8 + x, b[1] as i64 * 8 + y, b[2] as i64 * 8 + z];
                        let gin = occ_ref.inside(g);
                        for a in 0..3 {
                            for s in [-1i64, 1] {
                                let mut q = g;
                                q[a] += s;
                                let qin = occ_ref.inside(q);
                                if gin && !qin {
                                    out.push((g, a, s > 0));
                                } else if !gin && qin && !occ_ref.materialized(q) {
                                    out.push((q, a, s < 0));
                                }
                            }
                        }
                    }
                }
            }
            out.into_iter()
        })
        .collect();
    prog.report(0.8)?;

    let mut weld: FastMap<u64, u32> = FastMap::default();
    let mut positions = Vec::new();
    let mut tris = Vec::with_capacity(faces.len() * 2);
    let mut vid = |c: [i64; 3], positions: &mut Vec<DVec3>| -> u32 {
        let k = (c[0] + 1) as u64 | (((c[1] + 1) as u64) << 21) | (((c[2] + 1) as u64) << 42);
        *weld.entry(k).or_insert_with(|| {
            positions.push(grid.origin + (DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64) - DVec3::splat(0.5)) * h);
            (positions.len() - 1) as u32
        })
    };
    for (g, a, positive) in faces {
        let (u, w) = ((a + 1) % 3, (a + 2) % 3);
        let mut c0 = g;
        if positive {
            c0[a] += 1;
        }
        let mut c1 = c0;
        c1[u] += 1;
        let mut c2 = c1;
        c2[w] += 1;
        let mut c3 = c0;
        c3[w] += 1;
        let q = [vid(c0, &mut positions), vid(c1, &mut positions), vid(c2, &mut positions), vid(c3, &mut positions)];
        if positive {
            tris.push([q[0], q[1], q[2]]);
            tris.push([q[0], q[2], q[3]]);
        } else {
            tris.push([q[0], q[2], q[1]]);
            tris.push([q[0], q[3], q[2]]);
        }
    }
    prog.report(1.0)?;
    Ok((positions, tris))
}
