// SPDX-License-Identifier: AGPL-3.0-only
//! A cell of a patch network mapped onto the unit square (or, for a cell to be split from its
//! centre, onto a regular polygon) so that each of its scan points has a (u, v) on the patch
//! fitted through it.
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkMap.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! The cell's boundary goes onto the polygon's sides, corner to corner, by arc length; each
//! interior vertex is placed at a weighted mean of its neighbours, the weights Floater's
//! mean-value coordinates: positive for any triangle, shape-preserving on a flat one.
//! Floater's theorem (Tutte's for the plain mean) says a map with positive weights onto a
//! convex boundary is one-to-one, however the cell lies in space, so a cell that folds back
//! round a cylinder maps as well as a flat one, which is what Fit Surface's height field
//! cannot do. The weights are not symmetric (w_ij ≠ w_ji), so the system is solved by
//! BiCGSTAB rather than conjugate gradient, u and v the same matrix. Their mean would be
//! symmetric, and positive, and one-to-one still; but it loses the mean-value weights'
//! linear precision (a flat mesh is mapped linearly onto a flat square only by them) and a
//! strip of a cylinder mapped with it was 0.12 mm off its patch at the start of the fit,
//! which parameter correction took twelve rounds to bring to a third of that.
//!
//! A scan hole wholly inside the cell is closed for the map by a fan from the hole's middle;
//! the fan's vertex carries no data. Any triangle that comes out flipped (a cell whose
//! boundary is not a disk's, a crumb of the scan folded on itself) refuses the cell, by where
//! it is.

use std::collections::{HashMap, HashSet};

use glam::DVec3;

use crate::network_cells::fixed1;
use crate::scan_path::ScanPaths;
use crate::sparse::{biconjugate_gradient_stabilized, CgOptions, Triplets};

/// The solver's tolerance. The map is where each point starts from (the fit takes it onto
/// the Coons surface of the patch's sides and on from there), so it need not be exact, only
/// one-to-one: at 1e-7 no triangle flipped that 1e-9 left unflipped, on the cylinder's
/// strip, the saddle or the sphere's cells, in a third of the time.
const MAP_TOLERANCE: f64 = 1e-7;

/// A cell's mesh as the map and the fit read it: local vertices, each the scan vertex it is
/// or -1 for a fan's middle, and the triangles.
#[derive(Debug, Clone, PartialEq)]
pub struct CellMesh {
    /// x, y, z per local vertex.
    pub positions: Vec<f64>,
    pub scan_vertex: Vec<i32>,
    /// Three local vertices a triangle.
    pub triangles: Vec<u32>,
    /// The boundary loop, local ids in order with the cell on the left.
    pub boundary: Vec<usize>,
    /// Where each corner is in `boundary`.
    pub corner_at: Vec<usize>,
}

impl CellMesh {
    /// Local vertex v's position.
    pub fn position(&self, v: usize) -> DVec3 {
        DVec3::new(self.positions[v * 3], self.positions[v * 3 + 1], self.positions[v * 3 + 2])
    }
}

/// A cell mapped onto its polygon.
#[derive(Debug, Clone, PartialEq)]
pub struct CellMap {
    pub mesh: CellMesh,
    /// (u, v) per local vertex.
    pub uv: Vec<f64>,
    /// The solver's iterations, u and v together.
    pub iterations: usize,
}

/// A point as a map's refusal names it.
fn named(p: DVec3) -> String {
    format!("({}, {}, {})", fixed1(p.x), fixed1(p.y), fixed1(p.z))
}

/// The cell's local mesh from its scan triangles, with a fan in each hole, its boundary
/// loop, and the corners found on it; or why there is none: a cell with two boundaries is a
/// ring, not a disk.
pub fn cell_mesh(scan: &ScanPaths, triangles: &[u32], holes: &[Vec<u32>], corners: &[DVec3]) -> Result<CellMesh, String> {
    let mut local: HashMap<u32, usize> = HashMap::new();
    let mut positions: Vec<f64> = Vec::new();
    let mut scan_vertex: Vec<i32> = Vec::new();
    let mut tri: Vec<usize> = Vec::new();
    let mut take = |v: u32, positions: &mut Vec<f64>, scan_vertex: &mut Vec<i32>| -> usize {
        *local.entry(v).or_insert_with(|| {
            let l = scan_vertex.len();
            scan_vertex.push(v as i32);
            let p = scan.vertex(v);
            positions.extend_from_slice(&[p.x, p.y, p.z]);
            l
        })
    };
    for &t in triangles {
        if t as usize >= scan.triangle_count() {
            continue;
        }
        for v in scan.triangle(t) {
            let l = take(v, &mut positions, &mut scan_vertex);
            tri.push(l);
        }
    }
    // A hole's rim runs with the cell on its left; the fan on the other side of each rim
    // edge runs it backwards.
    for hole in holes {
        if hole.is_empty() {
            continue;
        }
        let mut c = DVec3::ZERO;
        for &v in hole {
            c += scan.vertex(v) / hole.len() as f64;
        }
        let middle = scan_vertex.len();
        scan_vertex.push(-1);
        positions.extend_from_slice(&[c.x, c.y, c.z]);
        for i in 0..hole.len() {
            let a = take(hole[(i + 1) % hole.len()], &mut positions, &mut scan_vertex);
            let b = take(hole[i], &mut positions, &mut scan_vertex);
            tri.extend_from_slice(&[a, b, middle]);
        }
    }
    // The boundary: the edges of one triangle only, in its winding.
    let count = tri.len() / 3;
    let key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let mut edge_use: HashMap<(usize, usize), u32> = HashMap::new();
    for t in 0..count {
        for e in 0..3 {
            *edge_use.entry(key(tri[t * 3 + e], tri[t * 3 + (e + 1) % 3])).or_default() += 1;
        }
    }
    let mut next_of: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut next_order: Vec<usize> = Vec::new();
    for t in 0..count {
        for e in 0..3 {
            let a = tri[t * 3 + e];
            let b = tri[t * 3 + (e + 1) % 3];
            if edge_use.get(&key(a, b)) != Some(&1) {
                continue;
            }
            next_of
                .entry(a)
                .or_insert_with(|| {
                    next_order.push(a);
                    Vec::new()
                })
                .push(b);
        }
    }
    let mut used: HashSet<(usize, usize)> = HashSet::new();
    let mut loops: Vec<Vec<usize>> = Vec::new();
    for &a0 in &next_order {
        let Some(list) = next_of.get(&a0) else { continue };
        for &b0 in list {
            if used.contains(&(a0, b0)) {
                continue;
            }
            let mut lp = vec![a0];
            let mut b = b0;
            let mut a = a0;
            for _guard in 0..=scan_vertex.len() {
                used.insert((a, b));
                if b == a0 {
                    break;
                }
                lp.push(b);
                let Some(option) = next_of.get(&b).and_then(|l| l.iter().copied().find(|&x| !used.contains(&(b, x)))) else {
                    break;
                };
                a = b;
                b = option;
            }
            loops.push(lp);
        }
    }
    if loops.is_empty() {
        return Err("it has no boundary — the scan closes round it".to_string());
    }
    let pos = |v: usize| DVec3::new(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
    if loops.len() > 1 {
        loops.sort_by(|x, y| y.len().cmp(&x.len()));
        let v = loops[1][0];
        return Err(format!("it is a ring, not a disk — another boundary inside it at {}", named(pos(v))));
    }
    let boundary = loops.swap_remove(0);
    // Each corner at the loop's vertex nearest it, in the corners' order.
    let corner_at: Vec<usize> = corners
        .iter()
        .map(|&c| {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (i, &v) in boundary.iter().enumerate() {
                let d = (pos(v) - c).length();
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            best
        })
        .collect();
    // Round the loop from the first corner, the others come in order.
    let l = boundary.len();
    let rel: Vec<usize> = corner_at.iter().map(|&i| (i + l - corner_at[0]) % l).collect();
    for k in 1..rel.len() {
        if rel[k] <= rel[k - 1] {
            return Err("its corners do not come round its boundary in order — a side doubles back on itself".to_string());
        }
    }
    Ok(CellMesh { positions, scan_vertex, triangles: tri.iter().map(|&v| v as u32).collect(), boundary, corner_at })
}

/// The polygon a cell is mapped onto: the unit square for four corners (side 0 along v = 0
/// from (0, 0), then round counter-clockwise) or a regular polygon of n corners in the unit
/// disk, its first corner at angle 0.
pub fn polygon_corners(n: usize) -> Vec<[f64; 2]> {
    if n == 4 {
        return vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    }
    (0..n)
        .map(|k| {
            let a = (2.0 * std::f64::consts::PI * k as f64) / n as f64;
            [0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]
        })
        .collect()
}

/// The cell mapped onto its polygon (see the module documentation), or why not. `start`, by
/// local vertex, is where the solver starts the interior from.
pub fn map_cell(mesh: CellMesh, start: Option<&[f64]>) -> Result<CellMap, String> {
    let nv = mesh.scan_vertex.len();
    let corners = polygon_corners(mesh.corner_at.len());
    let mut uv = vec![0.0f64; nv * 2];
    let mut fixed = vec![false; nv];
    // The boundary, side by side, by arc length.
    let lp = &mesh.boundary;
    let l = lp.len();
    let nc = mesh.corner_at.len();
    for k in 0..nc {
        let from = mesh.corner_at[k];
        let to = mesh.corner_at[(k + 1) % nc];
        let steps = match (to + l - from) % l {
            0 => l,
            s => s,
        };
        let mut along = vec![0.0f64; steps + 1];
        for s in 1..=steps {
            let a = lp[(from + s - 1) % l];
            let b = lp[(from + s) % l];
            along[s] = along[s - 1] + (mesh.position(b) - mesh.position(a)).length();
        }
        let total = if along[steps] != 0.0 && !along[steps].is_nan() { along[steps] } else { 1.0 };
        let [x0, y0] = corners[k];
        let [x1, y1] = corners[(k + 1) % corners.len()];
        for s in 0..steps {
            let v = lp[(from + s) % l];
            let f = along[s] / total;
            uv[v * 2] = x0 + f * (x1 - x0);
            uv[v * 2 + 1] = y0 + f * (y1 - y0);
            fixed[v] = true;
        }
    }
    // The interior's unknowns.
    let mut unknown = vec![-1i64; nv];
    let mut m = 0usize;
    for v in 0..nv {
        if !fixed[v] {
            unknown[v] = m as i64;
            m += 1;
        }
    }
    // Mean-value weights, per triangle corner: tan(θ/2) / |edge| toward each of the corner's
    // two neighbours, summed into w_ij; row i of the system Σ_j w_ij (u_i − u_j) = 0
    // straight into the triplets, whose duplicates the CSR sums, the boundary's known u moved
    // right.
    let t = &mesh.triangles;
    let mut a = Triplets::new(m, t.len() * 4);
    let mut bu = vec![0.0f64; m];
    let mut bv = vec![0.0f64; m];
    let mut add = |i: usize, j: usize, wij: f64| {
        let ui = unknown[i];
        if ui < 0 || !(wij > 0.0) {
            return;
        }
        let ui = ui as usize;
        a.add(ui, ui, wij);
        let uj = unknown[j];
        if uj >= 0 {
            a.add(ui, uj as usize, -wij);
        } else {
            bu[ui] += wij * uv[j * 2];
            bv[ui] += wij * uv[j * 2 + 1];
        }
    };
    for f in 0..t.len() / 3 {
        for c in 0..3 {
            let i = t[f * 3 + c] as usize;
            let j = t[f * 3 + (c + 1) % 3] as usize;
            let k = t[f * 3 + (c + 2) % 3] as usize;
            let e = mesh.position(j) - mesh.position(i);
            let g = mesh.position(k) - mesh.position(i);
            let le = e.length();
            let lg = g.length();
            if !(le > 0.0 && lg > 0.0) {
                continue;
            }
            let cos = (e.dot(g) / (le * lg)).clamp(-1.0, 1.0);
            // tan(θ/2) = sin θ / (1 + cos θ).
            let half = (1.0 - cos * cos).max(0.0).sqrt() / (1.0 + cos + 1e-300);
            add(i, j, half / le);
            add(i, k, half / lg);
        }
    }
    let mut iterations = 0;
    if m > 0 {
        let csr = a.to_csr();
        let (mut x0u, mut x0v) = (None, None);
        if let Some(start) = start {
            let mut su = vec![0.0; m];
            let mut sv = vec![0.0; m];
            for v in 0..nv {
                if unknown[v] < 0 || v * 2 + 1 >= start.len() {
                    continue;
                }
                su[unknown[v] as usize] = start[v * 2];
                sv[unknown[v] as usize] = start[v * 2 + 1];
            }
            x0u = Some(su);
            x0v = Some(sv);
        }
        let su = biconjugate_gradient_stabilized(&csr, &bu, &CgOptions { tolerance: MAP_TOLERANCE, x0: x0u, ..Default::default() });
        let sv = biconjugate_gradient_stabilized(&csr, &bv, &CgOptions { tolerance: MAP_TOLERANCE, x0: x0v, ..Default::default() });
        iterations = su.iterations + sv.iterations;
        for v in 0..nv {
            if unknown[v] < 0 {
                continue;
            }
            uv[v * 2] = su.x[unknown[v] as usize];
            uv[v * 2 + 1] = sv.x[unknown[v] as usize];
        }
    }
    // Every triangle the same way round as the polygon.
    let faces = t.len() / 3;
    let mut area = 0.0;
    let mut signed = vec![0.0f64; faces];
    for f in 0..faces {
        let (p, q, r) = (t[f * 3] as usize, t[f * 3 + 1] as usize, t[f * 3 + 2] as usize);
        signed[f] = (uv[q * 2] - uv[p * 2]) * (uv[r * 2 + 1] - uv[p * 2 + 1]) - (uv[r * 2] - uv[p * 2]) * (uv[q * 2 + 1] - uv[p * 2 + 1]);
        area += signed[f].abs();
    }
    let slack = 1e-9 * (area / faces.max(1) as f64);
    for f in 0..faces {
        if signed[f] < -slack {
            let p = t[f * 3] as usize;
            return Err(format!("the map folds over at {} — the cell's boundary is not a simple loop there", named(mesh.position(p))));
        }
    }
    Ok(CellMap { mesh, uv, iterations })
}
