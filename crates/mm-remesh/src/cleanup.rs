//! Small edge-length cleanup helpers built on the remesher's local operations.

use crate::constraints::{BoundaryRule, ConstraintOptions};
use crate::remesh::{Remesher, Setup};
use crate::util::Progress;
use crate::{ProgressFn, RemeshError};
use mm_core::Mesh;

/// Result of [`refine_uniform`] / [`collapse_short_edges`].
#[derive(Clone, Debug, Default)]
pub struct CleanupReport {
    pub splits: usize,
    pub collapses: usize,
    /// New ids of the region triangles (only when a region was given).
    pub region_triangles: Option<Vec<u32>>,
    pub cancelled: bool,
}

fn finish(r: &mut Remesher, prog: &Progress) -> CleanupReport {
    CleanupReport {
        splits: r.splits,
        collapses: r.collapses,
        region_triangles: if r.region.enabled() { Some(r.region.triangles(r.mesh)) } else { None },
        cancelled: prog.cancelled,
    }
}

/// Split (at midpoints, longest first) every edge longer than `max_edge_length`.
/// Vertex positions never move, so the surface is unchanged. With a region, border
/// edges are not split and outside triangles are untouched; next to border edges longer
/// than `max_edge_length` a transition band (narrower than one border edge) keeps longer
/// edges, because splitting there could only produce slivers.
pub fn refine_uniform(
    mesh: &mut Mesh,
    max_edge_length: f64,
    region: Option<&[u32]>,
    progress: Option<ProgressFn>,
) -> Result<CleanupReport, RemeshError> {
    if !(max_edge_length.is_finite() && max_edge_length > 0.0) {
        return Err(RemeshError::InvalidParameter("max edge length must be positive".into()));
    }
    let setup = Setup {
        region,
        constraints: ConstraintOptions { boundary: BoundaryRule::Free, sharp_deg: None, groups: false, corner_deg: None },
        base: max_edge_length,
        adaptive: false,
        project: false,
        smoothing: 0.0,
        free_boundary: false,
    };
    let mut prog = Progress::new(progress);
    let mut r = Remesher::new(mesh, setup);
    // Each pass terminates on its own (see `split_pass`); repeat to pick up edges whose
    // split was deferred, with a cap because fixed long edges can force endless splits.
    for pass in 0..8 {
        let before = r.splits;
        r.split_pass(1.0, &mut prog, pass as f32 / 8.0);
        if !prog.ok() || r.splits == before {
            break;
        }
    }
    if prog.ok() {
        prog.report(1.0);
    }
    Ok(finish(&mut r, &prog))
}

/// Collapse edges shorter than `min_edge_length` (repeated sweeps, shortest first).
///
/// Boundary vertices are only removed along the boundary and interior edges touching
/// the boundary collapse onto the boundary vertex; face-group borders are treated the
/// same way. Collapses that would flip normals or create degenerate triangles are
/// skipped, so some short edges may remain.
pub fn collapse_short_edges(
    mesh: &mut Mesh,
    min_edge_length: f64,
    region: Option<&[u32]>,
    progress: Option<ProgressFn>,
) -> Result<CleanupReport, RemeshError> {
    if !(min_edge_length.is_finite() && min_edge_length > 0.0) {
        return Err(RemeshError::InvalidParameter("min edge length must be positive".into()));
    }
    let setup = Setup {
        region,
        constraints: ConstraintOptions { boundary: BoundaryRule::Curve, sharp_deg: None, groups: true, corner_deg: None },
        base: min_edge_length,
        adaptive: false,
        project: false,
        smoothing: 0.0,
        free_boundary: false,
    };
    let mut prog = Progress::new(progress);
    let mut r = Remesher::new(mesh, setup);
    r.check_collapse_len = false;
    r.collapse_min_quality = 0.0;
    for pass in 0..8 {
        if !prog.report(pass as f32 / 8.0) {
            break;
        }
        if r.collapse_pass(1.0, &mut prog, pass as f32 / 8.0) == 0 {
            break;
        }
    }
    if prog.ok() {
        prog.report(1.0);
    }
    Ok(finish(&mut r, &prog))
}
