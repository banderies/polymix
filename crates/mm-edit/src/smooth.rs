//! Smooth Boundary: relax a ragged boundary curve within the surface's tangent planes.

use crate::util::orient_boundary_loop;
use crate::EditResult;
use mm_core::{DVec3, Mesh};
use std::collections::HashSet;

/// Smooth the boundary loop `loop_vertices` (either orientation) with `iterations` passes
/// of Taubin λ/μ smoothing along the curve. Each displacement is projected onto the
/// surface's tangent plane at the vertex, so the curve straightens without leaving the
/// surface or shrinking; the first ring of interior vertices is relaxed tangentially
/// afterwards to avoid fold-overs.
pub fn smooth_boundary(mesh: &mut Mesh, loop_vertices: &[u32], iterations: usize) -> EditResult<()> {
    let lp = orient_boundary_loop(mesh, loop_vertices)?;
    let n = lp.len();
    let on_loop: HashSet<u32> = lp.iter().copied().collect();
    let mut ring: Vec<u32> = lp
        .iter()
        .flat_map(|&v| mesh.vertex_neighbors(v).collect::<Vec<_>>())
        .filter(|w| !on_loop.contains(w) && !mesh.is_boundary_vertex(*w))
        .collect();
    ring.sort_unstable();
    ring.dedup();
    let tangent = |m: &Mesh, v: u32, d: DVec3| {
        let nv = m.vertex_normal(v);
        d - nv * d.dot(nv)
    };
    for _ in 0..iterations {
        for &w in &[0.5, -0.53] {
            let pts: Vec<DVec3> = lp.iter().map(|&v| mesh.vertex(v)).collect();
            let moves: Vec<DVec3> = (0..n)
                .map(|i| {
                    let lap = (pts[(i + n - 1) % n] + pts[(i + 1) % n]) * 0.5 - pts[i];
                    pts[i] + tangent(mesh, lp[i], lap * w)
                })
                .collect();
            for (i, p) in moves.into_iter().enumerate() {
                mesh.set_vertex(lp[i], p);
            }
        }
        let moves: Vec<(u32, DVec3)> = ring
            .iter()
            .map(|&v| {
                let nb: Vec<u32> = mesh.vertex_neighbors(v).collect();
                let avg = nb.iter().map(|&x| mesh.vertex(x)).sum::<DVec3>() / nb.len().max(1) as f64;
                (v, mesh.vertex(v) + tangent(mesh, v, (avg - mesh.vertex(v)) * 0.5))
            })
            .collect();
        for (v, p) in moves {
            mesh.set_vertex(v, p);
        }
    }
    Ok(())
}
