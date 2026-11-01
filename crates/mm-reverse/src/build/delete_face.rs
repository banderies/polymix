// SPDX-License-Identifier: AGPL-3.0-only
//! The Delete face feature's build: faces of a body taken off and the faces round them run on
//! to close the gap — a fillet, a chamfer, a hole, a boss taken out of a part. Ported from
//! ScanRuler (`plugins/reverse/core/resolve.ts` the deleteFace step, `build.ts` applyToBody,
//! `kernel.ts` deleteFaces), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only, on the kernel's own
//! `delete_faces`. The faces that ran on over the gap are found again each on its own surface,
//! and stay what they were.

use solidcraft_kernel as kernel;

use super::{diagonal_of, kerr, origins_after, read_topology, volume, Builder};
use crate::model::{DeleteFaceFeature, Feature, Part};
use crate::topo::{face_distance, MatchTolerance};

pub(crate) fn build(b: &mut Builder, _part: &Part, _f: &Feature, x: &DeleteFaceFeature) -> Result<(), String> {
    if x.faces.is_empty() {
        return Err("Pick the faces to delete.".into());
    }
    let (id, state) = b.body_for(x.body)?;
    let found = b.find_faces(id, &x.faces, &x.faces_where).map_err(|lost| format!("A face this feature deletes is gone (was: {lost})."))?;
    let t = b.topology_of(id).unwrap_or_else(|| state.topology());
    let made = kernel::delete_faces(&state.shape, &found).map_err(|e| format!("Deleting faces: {}", kerr(e)))?;
    // A face taken off that is still there, as it was, or a solid left with no inside: the
    // faces round the gap did not run on over it.
    let after = read_topology(&made, &vec![-1; made.face_count()]).face_prints();
    // As it was: in place, the same size, to a millionth of the part.
    let tol = MatchTolerance { position: 1e-6 * diagonal_of(&made).max(1.0), angle: 1e-6, size: 1e-6 };
    let still = found.iter().filter_map(|i| t.faces.get(*i)).any(|f| after.iter().any(|a| face_distance(&f.fingerprint, a, &tol).is_some()));
    if still || (!state.surface && !(volume(&made).unwrap_or(0.0) > 0.0)) {
        return Err("The faces could not be taken off — the faces round them do not run on to close the gap. Pick the whole of a round, a hole or a boss: every face it is made of.".into());
    }
    let origins = origins_after(&made, &[(&state.shape, &state.origins)], b.step);
    let ids = b.settle(id, made, origins, false)?;
    b.touched(ids);
    Ok(())
}

#[cfg(test)]
mod tests;
