// SPDX-License-Identifier: AGPL-3.0-only
//! The Mirror feature's build: a body mirrored about a plane — any plane reference (a
//! coordinate plane at its offset, a section's, a measured plane, a radial plane, the scan's
//! own symmetry plane) or a flat face of the body found again by its fingerprint — the copy
//! joined to the body, or a body of its own beside it. Ported from ScanRuler
//! (`plugins/reverse/core/resolve.ts` the mirror step, `build.ts` applyToBody and the mirror
//! not joined), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use solidcraft_kernel::{self as kernel, BoolOp};

use super::cut_plane::SHEET_NOT_SOLID;
use super::{describe_plane_ref, kerr, mirror_matrix, origins_after, resolve_plane, Builder};
use crate::model::{Feature, MirrorFeature, MirrorPlane, Part};
use crate::topo::FaceKind;

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, m: &MirrorFeature) -> Result<(), String> {
    let (id, state) = b.body_for(m.body)?;
    // A mirror joined works on the body as a solid; one not joined is a copy beside it, a
    // sheet's too.
    if state.surface && m.join {
        return Err(SHEET_NOT_SOLID.into());
    }
    let (origin, normal) = match &m.plane {
        MirrorPlane::Plane(p) if !super::cut_plane::plane_chosen(p) => return Err("No plane chosen — click a flat face of the body, or choose a plane.".into()),
        MirrorPlane::Plane(p) => resolve_plane(p, part).ok_or_else(|| format!("The plane this mirror is about is gone ({}).", describe_plane_ref(p, part)))?,
        MirrorPlane::Face { face } => {
            let found = b.find_faces(id, std::slice::from_ref(face), &[]).map_err(|lost| format!("The face this mirror is about is gone (was: {lost})."))?;
            let fp = found.first().and_then(|i| b.topology_of(id).unwrap_or_else(|| state.topology()).faces.get(*i).map(|x| x.fingerprint)).ok_or("The face this mirror is about is gone.")?;
            if fp.kind != FaceKind::Plane {
                return Err("The face this mirror is about is not a plane.".into());
            }
            (fp.centroid, fp.dir)
        }
    };
    let copy = kernel::transform_matrix(&state.shape, mirror_matrix(origin, normal)).map_err(kerr)?;
    if m.join {
        // The mirrored half's faces keep the makers of the faces they mirror, carried through
        // the join.
        let copy_origins = origins_after(&copy, &[(&state.shape, &state.origins)], b.step);
        let joined = kernel::boolean(&state.shape, &copy, BoolOp::Union).map_err(kerr)?.ok_or("The mirror left no closed solid.")?;
        let origins = origins_after(&joined, &[(&state.shape, &state.origins), (&copy, &copy_origins)], b.step);
        let ids = b.settle(id, joined, origins, false)?;
        b.touched(ids);
        return Ok(());
    }
    // A mirror not joined is a body of its own beside its source, as an offset that keeps its
    // body is: the source stands as it was, and a Combine between the two finds both.
    let fresh = b.fresh_id()?;
    let own = b.own_origins(&copy);
    // A sheet's copy is a sheet, one body however many pieces it is in.
    let ids = if state.surface { b.settle_sheet(fresh, copy, own) } else { b.settle(fresh, copy, own, false)? };
    b.touched(ids);
    Ok(())
}
