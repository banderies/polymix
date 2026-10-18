//! Complexes and convex pieces (Meshmixer's Edit > Generate / Split Complex, Convex
//! Decomposition).
//!
//! * [`generate_complex`]: the cells of overlapping solids — every region covered by a
//!   particular set of them (for two: A − B, A ∩ B, B − A) — by exact booleans.
//! * [`convex_decomposition`]: split a solid by planes until every piece is nearly convex
//!   (its hull adds little volume) and return the pieces' hulls. Each split tries planes
//!   across the piece's principal axes and the world axes, placed at the piece's own vertex
//!   positions (so cuts can run exactly along concave edges of CAD-like parts), and keeps the
//!   one whose halves are the most convex.

use crate::cut::{plane_cut, CutFill, KeepSide, PlaneCutParams};
use crate::{convex_hull, mesh_boolean, BooleanOp, EditError, EditResult};
use glam::DVec3;
use mm_core::{measure::volume, topology, Mesh, Plane};

/// Cells of up to six overlapping closed solids, largest first. Empty cells are left out.
pub fn generate_complex(solids: &[Mesh]) -> EditResult<Vec<Mesh>> {
    if !(2..=6).contains(&solids.len()) {
        return Err(EditError::InvalidParameter("Generate Complex needs 2 to 6 closed objects".into()));
    }
    let n = solids.len();
    let mut cells = Vec::new();
    for mask in 1u32..(1 << n) {
        let inside: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let mut cell = solids[inside[0]].clone();
        let mut empty = false;
        for &i in &inside[1..] {
            cell = mesh_boolean(&cell, &solids[i], BooleanOp::Intersection)?.mesh;
            if cell.triangle_count() == 0 {
                empty = true;
                break;
            }
        }
        if !empty {
            for j in (0..n).filter(|j| mask & (1 << j) == 0) {
                cell = mesh_boolean(&cell, &solids[j], BooleanOp::Difference)?.mesh;
                if cell.triangle_count() == 0 {
                    break;
                }
            }
        }
        if cell.triangle_count() > 0 && volume(&cell) > 0.0 {
            cells.push(cell);
        }
    }
    cells.sort_by(|a, b| volume(b).total_cmp(&volume(a)));
    Ok(cells)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvexParams {
    /// Stop at this many pieces.
    pub max_pieces: usize,
    /// Stop when no piece's hull adds more than this fraction of the whole hull's volume.
    pub concavity: f64,
}

impl Default for ConvexParams {
    fn default() -> Self {
        Self { max_pieces: 16, concavity: 0.02 }
    }
}

struct Piece {
    mesh: Mesh,
    hull: Mesh,
    /// Hull volume minus piece volume.
    concavity: f64,
}

fn piece(mesh: Mesh) -> Option<Piece> {
    let pts: Vec<DVec3> = mesh.vertex_ids().map(|v| mesh.vertex(v)).collect();
    let hull = convex_hull(&pts)?;
    let concavity = (volume(&hull) - volume(&mesh)).max(0.0);
    Some(Piece { mesh, hull, concavity })
}

/// Cut `m` by `plane` into its closed pieces.
fn halves(m: &Mesh, plane: Plane) -> Option<Vec<Mesh>> {
    let mut c = m.clone();
    let r = plane_cut(&mut c, plane, &PlaneCutParams { keep: KeepSide::Both, fill: CutFill::Flat, ..Default::default() }).ok()?;
    if r.cut_loops.is_empty() || r.loop_capped.iter().any(|&k| !k) {
        return None;
    }
    let parts: Vec<Mesh> = topology::connected_components(&c).into_iter().map(|t| c.submesh(&t).0).filter(|p| volume(p) > 0.0).collect();
    (parts.len() >= 2).then_some(parts)
}

/// Best split of a piece, as its new pieces.
fn split(p: &Piece) -> Option<Vec<Piece>> {
    let m = &p.mesh;
    let pts: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).collect();
    let c = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
    let mut cov = [[0.0; 3]; 3];
    for q in &pts {
        mm_core::linalg::add_outer(&mut cov, *q - c, 1.0);
    }
    let (_, axes) = mm_core::linalg::sym_eigen3(cov);
    let mut best: Option<(f64, Vec<Piece>)> = None;
    for n in axes.into_iter().chain([DVec3::X, DVec3::Y, DVec3::Z]) {
        // Distinct vertex positions along the axis (inner ones), thinned to a dozen.
        let mut d: Vec<f64> = pts.iter().map(|q| (*q - c).dot(n)).collect();
        d.sort_by(|a, b| a.total_cmp(b));
        let (lo, hi) = (d[0], d[d.len() - 1]);
        let eps = (hi - lo) * 1e-6;
        d.dedup_by(|a, b| (*a - *b).abs() <= eps);
        let inner: Vec<f64> = d.into_iter().filter(|&x| x > lo + (hi - lo) * 0.05 && x < hi - (hi - lo) * 0.05).collect();
        let step = inner.len().div_ceil(12).max(1);
        for off in inner.into_iter().step_by(step) {
            let Some(parts) = halves(m, Plane::from_point_normal(c + n * off, n)) else { continue };
            let pieces: Vec<Piece> = parts.into_iter().filter_map(piece).collect();
            let score: f64 = pieces.iter().map(|q| q.concavity).sum();
            if pieces.len() >= 2 && best.as_ref().is_none_or(|b| score < b.0) {
                best = Some((score, pieces));
            }
        }
    }
    best.map(|b| b.1)
}

/// Convex hulls of the pieces of `mesh` (see the module docs).
pub fn convex_decomposition(mesh: &Mesh, p: &ConvexParams) -> EditResult<Vec<Mesh>> {
    if !mesh.is_closed() || mesh.triangle_count() == 0 {
        return Err(EditError::InvalidParameter("Convex Decomposition needs a closed object (Make Solid first)".into()));
    }
    let first = piece(mesh.clone()).ok_or_else(|| EditError::Failed("the object is flat or degenerate".into()))?;
    let total = volume(&first.hull).max(1e-300);
    let mut pieces = vec![first];
    let mut stuck = vec![false];
    while pieces.len() < p.max_pieces.max(1) {
        let Some(i) = (0..pieces.len()).filter(|&i| !stuck[i]).max_by(|&a, &b| pieces[a].concavity.total_cmp(&pieces[b].concavity)) else { break };
        if pieces[i].concavity / total <= p.concavity {
            break;
        }
        match split(&pieces[i]) {
            Some(new) => {
                pieces.swap_remove(i);
                stuck.swap_remove(i);
                stuck.extend(new.iter().map(|_| false));
                pieces.extend(new);
            }
            None => stuck[i] = true,
        }
    }
    Ok(pieces.into_iter().map(|q| q.hull).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn two_overlapping_cubes_make_three_cells() {
        let a = primitives::cube(2.0, 1);
        let mut b = primitives::cube(2.0, 1);
        b.transform(|p| p + DVec3::new(1.0, 0.3, 0.2));
        let cells = generate_complex(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(cells.len(), 3);
        for c in &cells {
            c.check_validity().unwrap();
            assert!(c.is_closed());
        }
        // The cells tile the union.
        let sum: f64 = cells.iter().map(volume).sum();
        let union = volume(&mesh_boolean(&a, &b, BooleanOp::Union).unwrap().mesh);
        assert!((sum - union).abs() < 1e-9 * union.max(1.0), "{sum} vs {union}");
        // Disjoint solids: each is its own cell, no intersection.
        let mut far = primitives::cube(1.0, 1);
        far.transform(|p| p + DVec3::X * 10.0);
        assert_eq!(generate_complex(&[a, far]).unwrap().len(), 2);
    }

    #[test]
    fn an_l_shape_decomposes_into_two_boxes() {
        // L-shaped prism: union of two boxes.
        let mut a = primitives::cube(1.0, 2);
        a.transform(|p| p * DVec3::new(4.0, 1.0, 1.0));
        let mut b = primitives::cube(1.0, 2);
        b.transform(|p| p * DVec3::new(1.0, 4.0, 1.0) + DVec3::new(-1.5, 1.5, 0.0));
        let l = mesh_boolean(&a, &b, BooleanOp::Union).unwrap().mesh;
        let hulls = convex_decomposition(&l, &ConvexParams { max_pieces: 8, concavity: 0.01 }).unwrap();
        assert!((2..=4).contains(&hulls.len()), "{}", hulls.len());
        let total: f64 = hulls.iter().map(volume).sum();
        // Convex pieces cover the shape with little extra volume.
        assert!(total >= volume(&l) * 0.999 && total < volume(&l) * 1.05, "{total} vs {}", volume(&l));
        for h in &hulls {
            assert!(h.is_closed());
        }
        // A convex object stays one piece.
        let ball = primitives::icosphere(1.0, 3);
        assert_eq!(convex_decomposition(&ball, &ConvexParams::default()).unwrap().len(), 1);
    }
}
