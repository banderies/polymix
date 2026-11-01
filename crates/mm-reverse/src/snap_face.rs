// SPDX-License-Identifier: AGPL-3.0-only
//! A face of a body snapped onto the scan: the part is modelled roughly, a face is picked, the
//! scan is clicked where that face should lie, and the element fitted there — a plane for a flat
//! face, a cylinder for a round one — says how far the face has to move. This module is that
//! reading: the offset a Move face feature is made with, and how well the scan's face agrees
//! with the body's, for the box to say so. Where a flat face of the scan leans to the body's by
//! more than the fit's noise can explain, the reading also holds the draft that would lay it on
//! — a moulded wall modelled square. Plain geometry, no kernel: the fit is made elsewhere, the
//! kernel moves.
//!
//! Beside the reading: the Draft feature a snapped face is followed by (ScanRuler's
//! `snapDraftOf`, `state/reDrafts.ts`), and the ghost of faces moved where they will lie
//! (`movedFaceMesh`, `core/move.ts`).
//!
//! Ported from ScanRuler (`plugins/reverse/core/snapFace.ts`, `core/move.ts` movedFaceMesh,
//! `state/reDrafts.ts` snapDraftOf), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::DVec3;

use crate::build::BodyMesh;
use crate::fit3d::types::{FitData, PlaneFit};
use crate::frame::SectionFrame;
use crate::js_number::to_fixed;
use crate::model::{DraftFeature, Exprs, MirrorPlane, PlaneRef};
use crate::model::BodyId;
use crate::topo::{FaceFingerprint, FaceKind, Topology};

/// The draft that lays a moved flat face on the scan's leaning plane: a Draft feature's neutral
/// plane and angle, made after the move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapDraft {
    /// Degrees the face leans, as a Draft feature reads it: positive leans the face in as it
    /// goes along `pull`.
    pub angle_deg: f64,
    /// A point of the neutral plane: the moved face's middle, where the scan's plane crosses
    /// it. The line the face hinges on runs through it, along the face and across the tilt.
    pub origin: DVec3,
    /// The neutral plane's normal: in the face, the way the scan's plane leans into the body.
    pub pull: DVec3,
    /// The tilt the fit's own scatter could account for, degrees — what the offered angle had
    /// to clear.
    pub noise_deg: f64,
}

/// What the scan says about a face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapReading {
    /// Millimetres the face moves: along its outward normal for a plane, the change of radius
    /// for a cylinder.
    pub offset: f64,
    /// The angle between the body's face and the scan's — between the normals of two planes,
    /// the axes of two cylinders — degrees. A move cannot take it out; the box says how big it
    /// is.
    pub tilt_deg: f64,
    /// How far the scan's cylinder axis stands off the body's, at the face's middle,
    /// millimetres; absent for a plane. The move keeps the axis.
    pub axis_shift: Option<f64>,
    /// The scatter of the scan's points about the fitted element.
    pub sigma: f64,
    /// For a flat face whose tilt is real — above the noise floor — the draft that would take
    /// it out after the move; absent where the move alone lays the face on the scan, or for a
    /// round face.
    pub draft: Option<SnapDraft>,
}

/// The element fitted to the scan for a face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapFit {
    Plane,
    Cylinder,
}

impl SnapFit {
    pub fn name(self) -> &'static str {
        match self {
            SnapFit::Plane => "plane",
            SnapFit::Cylinder => "cylinder",
        }
    }
}

/// The element the scan should be fitted with for a face, or `None` for a face that cannot be
/// moved onto the scan.
pub fn snap_fit_kind(face: &FaceFingerprint) -> Option<SnapFit> {
    match face.kind {
        FaceKind::Plane => Some(SnapFit::Plane),
        FaceKind::Cylinder => Some(SnapFit::Cylinder),
        _ => None,
    }
}

/// Past this the scan's face is not the body's face turned a little but another face
/// altogether — the click landed on a neighbour.
const SNAP_MAX_TILT_DEG: f64 = 20.0;

/// Below this a tilt is not a draft, whatever the fit says: a tenth of a degree over a 50 mm
/// wall is 0.09 mm, the scan's own scatter.
const SNAP_DRAFT_MIN_DEG: f64 = 0.2;

/// How many times the fit's own angular noise a tilt must clear to be read as a draft rather
/// than as the patch's scatter.
const DRAFT_NOISE_FACTOR: f64 = 4.0;

/// The tilt of a fitted plane that its scatter alone could produce, degrees: the slope of a
/// line fitted through `n` points spread over a length `span` with scatter σ is uncertain by
/// σ·√(12/n)/span.
fn plane_noise_deg(fit: &PlaneFit, along: DVec3) -> f64 {
    let half = along.dot(fit.basis_u).abs() * fit.extent_u + along.dot(fit.basis_v).abs() * fit.extent_v;
    let span = 2.0 * half;
    if !(span > 1e-6) || !(fit.base.sigma > 0.0) {
        return 0.0;
    }
    let n = (fit.base.used_points as f64).max(3.0);
    (fit.base.sigma * (12.0 / n).sqrt() / span).atan().to_degrees()
}

/// The draft that lays a flat face, once moved onto the scan's plane at its middle, along that
/// plane: `n` the face's outward normal, `m` the fitted normal turned to n's side, `tilt_deg`
/// between them. A face leaning in as it goes along the pull is a positive draft, so the pull
/// is the fitted normal's part in the face — the way the scan's plane dips into the body.
/// `None` where the tilt is within what the fit's noise explains.
fn plane_draft(face: &FaceFingerprint, fit: &PlaneFit, n: DVec3, m: DVec3, offset: f64, tilt_deg: f64) -> Option<SnapDraft> {
    let pull = (m - n * n.dot(m)).try_normalize()?;
    let noise_deg = plane_noise_deg(fit, pull);
    if tilt_deg < SNAP_DRAFT_MIN_DEG.max(DRAFT_NOISE_FACTOR * noise_deg) {
        return None;
    }
    Some(SnapDraft { angle_deg: crate::round::round_to(tilt_deg, 2), origin: face.centroid + n * offset, pull, noise_deg })
}

/// The move that lays a face on the element fitted to the scan, or the reason there is none. A
/// plane is met along the face's own normal through its middle, so a slightly tilted scan face
/// puts the body's face through its centre rather than at its nearest corner.
pub fn snap_reading(face: &FaceFingerprint, fit: &FitData) -> Result<SnapReading, String> {
    match face.kind {
        FaceKind::Plane => {
            let FitData::Plane(fit) = fit else { return Err("A flat face snaps onto a plane of the scan.".into()) };
            let (Some(n), Some(m)) = (face.dir.try_normalize(), fit.normal.try_normalize()) else { return Err("The face has no direction.".into()) };
            let along = n.dot(m);
            let tilt_deg = along.abs().min(1.0).acos().to_degrees();
            if tilt_deg > SNAP_MAX_TILT_DEG {
                return Err(format!("The scan's face there is turned {}° to this one — click the face of the scan that matches it.", to_fixed(tilt_deg, 0)));
            }
            let offset = (fit.center - face.centroid).dot(m) / along;
            let draft = plane_draft(face, fit, n, if along < 0.0 { -m } else { m }, offset, tilt_deg);
            Ok(SnapReading { offset, tilt_deg, axis_shift: None, sigma: fit.base.sigma, draft })
        }
        FaceKind::Cylinder => {
            let FitData::Cylinder(fit) = fit else { return Err("A round face snaps onto a cylinder of the scan.".into()) };
            let (Some(radius), Some(center)) = (face.radius, face.center) else { return Err("The round face has no axis to keep.".into()) };
            let (Some(a), Some(b)) = (face.dir.try_normalize(), fit.axis.try_normalize()) else { return Err("The face has no axis.".into()) };
            let tilt_deg = a.dot(b).abs().min(1.0).acos().to_degrees();
            if tilt_deg > SNAP_MAX_TILT_DEG {
                return Err(format!("The scan's cylinder there is turned {}° to this one — click the round face of the scan that matches it.", to_fixed(tilt_deg, 0)));
            }
            let between = fit.center - center;
            let axis_shift = (between - a * between.dot(a)).length();
            Ok(SnapReading { offset: fit.radius - radius, tilt_deg, axis_shift: Some(axis_shift), sigma: fit.base.sigma, draft: None })
        }
        _ => Err("Only a flat or a cylindrical face can be snapped onto the scan.".into()),
    }
}

/// Whether two faces lie on one surface: one plane (facing one way), or one cylinder, cone or
/// sphere. Halves of one face meet as well (`halves_of`): a pick of either is a pick of the
/// face, which a Move face moves once and a Draft leans once.
pub fn one_surface(a: &FaceFingerprint, b: &FaceFingerprint) -> bool {
    match a.kind {
        FaceKind::Plane => b.kind == FaceKind::Plane && a.dir.dot(b.dir) > 1.0 - 1e-6 && (b.centroid - a.centroid).dot(a.dir).abs() <= 1e-3,
        _ => crate::topo::on_one_surface(a, b),
    }
}

/// A face of a body and the faces it is split from on its own surface — the two halves the
/// kernel keeps a whole bore or boss as, joined along their seams — by index in `t.faces`. Two
/// faces on one surface that do not meet (two walls in line either side of a notch, two bores
/// on one axis) are faces of their own, as they are to the kernel ScanRuler builds on.
pub fn halves_of(t: &Topology, face: usize) -> Vec<usize> {
    let mut out = vec![face];
    let mut k = 0;
    while let Some(at) = out.get(k).copied() {
        k += 1;
        let Some(here) = t.faces.get(at).map(|f| f.fingerprint) else { continue };
        for e in &t.edges {
            if !e.faces.iter().any(|f| *f as usize == at) {
                continue;
            }
            for other in e.faces.iter().map(|f| *f as usize) {
                if out.contains(&other) {
                    continue;
                }
                if t.faces.get(other).is_some_and(|o| one_surface(&here, &o.fingerprint)) {
                    out.push(other);
                }
            }
        }
    }
    out
}

/// Whether two faces of a body are one face to pick: the same, or halves of one (`halves_of`).
pub fn same_face(t: &Topology, a: usize, b: usize) -> bool {
    a == b || halves_of(t, a).contains(&b)
}

/// What the neutral plane of the Draft a snapped face is followed by is called.
pub const SNAP_DRAFT_NEUTRAL: &str = "the line across the face’s middle where the scan crosses it";

/// The Draft feature a snapped face is followed by — the scan's flat face leaning to the body's
/// by more than the fit's noise explains — or `None` where the face is not flat, or the move
/// has no length. The face is named where the move leaves it, and the neutral plane stands
/// square to the face through the line across its middle where the scan's plane crosses it, so
/// the face turns about that line onto the scan.
pub fn snap_draft_feature(body: BodyId, face: &FaceFingerprint, offset: f64, lean: &SnapDraft) -> Option<DraftFeature> {
    if face.kind != FaceKind::Plane || !(offset.abs() > 1e-6) {
        return None;
    }
    let u = face.dir.try_normalize().unwrap_or(DVec3::Z);
    let frame = SectionFrame { origin: lean.origin, normal: lean.pull, u, v: lean.pull.cross(u) };
    Some(DraftFeature {
        body,
        neutral: MirrorPlane::Plane(PlaneRef::Fitted { frame, label: Some(SNAP_DRAFT_NEUTRAL.to_string()) }),
        faces: vec![FaceFingerprint { centroid: face.centroid + face.dir * offset, ..*face }],
        faces_where: Vec::new(),
        angle_deg: lean.angle_deg,
        groups: Vec::new(),
        exprs: Exprs::new(),
    })
}

/// A face of a body moved, for the ghost: its index in the body's mesh and its fingerprint.
pub type MovedFace = (u32, FaceFingerprint);

/// Faces of a body's mesh, moved: a flat face along its normal, a round one out from its axis —
/// where faces being moved will lie, drawn as a ghost when the move is an edit of the history
/// and there is no one step for the kernel to preview. A triangle soup (three corners a
/// triangle) with a normal at each corner; `None` when the mesh names none of the faces. A lean
/// is the draft that follows a single snapped face.
pub fn moved_face_mesh(mesh: &BodyMesh, faces: &[MovedFace], offset: f64, lean: Option<&SnapDraft>) -> Option<(Vec<DVec3>, Vec<DVec3>)> {
    // The draft that follows the move turns the face about the line through the lean's origin,
    // along the face and square to the pull, so that the normal tips towards the pull:
    // Rodrigues about the axis dir × pull.
    let turn = match (lean, faces) {
        (Some(l), [only]) => only.1.dir.cross(l.pull).try_normalize().map(|k| (k, l)),
        _ => None,
    };
    let rotate = |v: DVec3| -> DVec3 {
        let Some((k, l)) = turn else { return v };
        let t = l.angle_deg.to_radians();
        let (s, c) = t.sin_cos();
        v * c + k.cross(v) * s + k * (k.dot(v) * (1.0 - c))
    };
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        let Some(owner) = mesh.tri_face.get(t) else { continue };
        let Some((_, face)) = faces.iter().find(|(h, _)| h == owner) else { continue };
        for v in tri {
            let Some(p) = mesh.positions.get(*v as usize).copied() else { continue };
            let n = mesh.normals.get(*v as usize).copied().unwrap_or(face.dir);
            let along = match (face.kind, face.center) {
                (FaceKind::Cylinder, Some(c)) => {
                    let d = p - c;
                    (d - face.dir * d.dot(face.dir)).try_normalize().unwrap_or(face.dir)
                }
                _ => face.dir,
            };
            let moved = p + along * offset;
            let q = match turn {
                Some((_, l)) => l.origin + rotate(moved - l.origin),
                None => moved,
            };
            positions.push(q);
            normals.push(rotate(n));
        }
    }
    (!positions.is_empty()).then_some((positions, normals))
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reSnapFace.test.ts` and the plain parts of `reSnapDraft.test.ts`.
    use super::*;
    use crate::fit3d::types::{CylinderFit, FitBase};

    const STATS: FitBase = FitBase { sigma: 0.02, used_points: 500, region_size: 500, form_error: None, residual_min: None, residual_max: None };

    fn top() -> FaceFingerprint {
        FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(20.0, 10.0, 10.0), dir: DVec3::Z, area: 800.0, radius: None, center: None, origin: 2 }
    }

    fn plane(center: [f64; 3], normal: [f64; 3]) -> FitData {
        FitData::Plane(PlaneFit { base: STATS, center: DVec3::from(center), normal: DVec3::from(normal), basis_u: DVec3::X, basis_v: DVec3::Y, extent_u: 5.0, extent_v: 5.0 })
    }

    fn bore() -> FaceFingerprint {
        FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(10.0, 10.0, 5.0), dir: DVec3::Z, area: 188.0, radius: Some(3.0), center: Some(DVec3::new(10.0, 10.0, 5.0)), origin: 5 }
    }

    fn reading(face: &FaceFingerprint, fit: &FitData) -> SnapReading {
        snap_reading(face, fit).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn reads_how_far_a_flat_face_moves_out_or_in_whichever_way_the_fitted_normal_points() {
        let out = reading(&top(), &plane([3.0, 4.0, 11.5], [0.0, 0.0, 1.0]));
        assert!((out.offset - 1.5).abs() < 1e-12 && out.tilt_deg == 0.0 && out.sigma == 0.02);
        // A plane fitted facing the other way is the same plane.
        assert!((reading(&top(), &plane([3.0, 4.0, 11.5], [0.0, 0.0, -1.0])).offset - 1.5).abs() < 1e-12);
        let inward = reading(&top(), &plane([30.0, 2.0, 9.2], [0.0, 0.0, 1.0]));
        assert!((inward.offset + 0.8).abs() < 1e-9);
    }

    #[test]
    fn meets_a_tilted_scan_face_through_the_middle_of_the_bodys_face_and_says_how_tilted() {
        // A plane through (20, 10, 12) — above the face's middle — tilted 2° about Y, fitted
        // on a patch away from the middle.
        let t = 2f64.to_radians();
        let n = [t.sin(), 0.0, t.cos()];
        let at = 10.0;
        let center = [20.0 + at, 10.0, 12.0 - at * n[0] / n[2]];
        let r = reading(&top(), &plane(center, n));
        assert!((r.offset - 2.0).abs() < 1e-6);
        assert!((r.tilt_deg - 2.0).abs() < 1e-6);
    }

    #[test]
    fn refuses_a_click_on_a_neighbouring_face_and_the_wrong_kind_of_element() {
        let e = snap_reading(&top(), &plane([40.0, 10.0, 5.0], [1.0, 0.0, 0.0])).unwrap_err();
        assert!(e.contains("turned 90°"), "{e}");
        let cyl = FitData::Cylinder(CylinderFit { base: STATS, center: DVec3::ZERO, axis: DVec3::Z, radius: 3.0, length: 10.0, coverage: 360.0 });
        assert!(snap_reading(&top(), &cyl).unwrap_err().contains("plane"));
    }

    #[test]
    fn reads_a_round_faces_change_of_radius_and_how_far_the_scans_axis_stands_off() {
        let fit = FitData::Cylinder(CylinderFit { base: STATS, center: DVec3::new(10.3, 10.4, 2.0), axis: DVec3::new(0.0, 0.0, -1.0), radius: 3.4, length: 10.0, coverage: 360.0 });
        let r = reading(&bore(), &fit);
        assert!((r.offset - 0.4).abs() < 1e-9);
        assert!(r.tilt_deg.abs() < 1e-6);
        assert!((r.axis_shift.unwrap_or(f64::NAN) - 0.5).abs() < 1e-9);
        assert_eq!(snap_fit_kind(&bore()), Some(SnapFit::Cylinder));
        assert_eq!(snap_fit_kind(&FaceFingerprint { kind: FaceKind::Cone, ..bore() }), None);
    }

    /// A plane through (20, 10, 12), 2 mm above the face's middle, dipping 3° as it goes along
    /// +X — a moulded wall's top scanned with its draft.
    fn dipping(flip: f64) -> FitData {
        let t = 3f64.to_radians();
        plane([20.0, 10.0, 12.0], [flip * t.sin(), 0.0, flip * t.cos()])
    }

    #[test]
    fn a_draft_is_offered_for_a_tilt_the_fits_scatter_cannot_explain_hinged_through_the_moved_middle() {
        let r = reading(&top(), &dipping(1.0));
        assert!((r.offset - 2.0).abs() < 1e-6);
        let d = r.draft.expect("a draft");
        assert!((d.angle_deg - 3.0).abs() < 0.005);
        assert!((d.origin.x - 20.0).abs() < 1e-9 && (d.origin.z - 12.0).abs() < 1e-9);
        assert!((d.pull.x - 1.0).abs() < 1e-9 && d.pull.z.abs() < 1e-9);
        assert!(d.noise_deg < 0.1);
        // The same plane fitted facing down is the same offer.
        let flipped = reading(&top(), &dipping(-1.0)).draft.expect("a draft");
        assert!((flipped.pull.x - 1.0).abs() < 1e-9);
        assert!((flipped.angle_deg - 3.0).abs() < 0.005);
    }

    #[test]
    fn no_draft_under_the_floor_nor_within_a_small_rough_patchs_noise_nor_for_a_round_face() {
        let tiny = 0.05f64.to_radians();
        assert!(reading(&top(), &plane([20.0, 10.0, 12.0], [tiny.sin(), 0.0, tiny.cos()])).draft.is_none());
        // Half a degree read off 20 rough points 4 mm across: the slope of such a fit is
        // uncertain by more than that.
        let t2 = 0.5f64.to_radians();
        let rough = FitData::Plane(PlaneFit { base: FitBase { sigma: 0.5, used_points: 20, region_size: 20, ..STATS }, center: DVec3::new(20.0, 10.0, 12.0), normal: DVec3::new(t2.sin(), 0.0, t2.cos()), basis_u: DVec3::X, basis_v: DVec3::Y, extent_u: 2.0, extent_v: 2.0 });
        let noisy = reading(&top(), &rough);
        assert!((noisy.tilt_deg - 0.5).abs() < 1e-6);
        assert!(noisy.draft.is_none());
        let t = 3f64.to_radians();
        let leaning = FitData::Cylinder(CylinderFit { base: STATS, center: DVec3::new(10.0, 10.0, 2.0), axis: DVec3::new(t.sin(), 0.0, t.cos()), radius: 3.4, length: 10.0, coverage: 360.0 });
        let r = reading(&bore(), &leaning);
        assert!((r.tilt_deg - 3.0).abs() < 1e-6);
        assert!(r.draft.is_none());
    }

    #[test]
    fn the_draft_after_a_snap_names_the_face_where_the_move_leaves_it_about_a_fitted_neutral_plane() {
        let r = reading(&top(), &dipping(1.0));
        let lean = r.draft.expect("a draft");
        let d = snap_draft_feature(BodyId::new(2), &top(), crate::round::round3(r.offset), &lean).expect("a draft feature");
        assert_eq!(d.faces[0].centroid, DVec3::new(20.0, 10.0, 12.0));
        assert!(matches!(d.neutral, MirrorPlane::Plane(PlaneRef::Fitted { .. })));
        assert!((d.angle_deg - 3.0).abs() < 0.005);
        // No length to the move, no draft.
        assert!(snap_draft_feature(BodyId::new(2), &top(), 0.0, &lean).is_none());
    }

    #[test]
    fn the_ghost_moves_a_flat_face_along_its_normal_and_a_round_one_out_from_its_axis() {
        let mesh = BodyMesh {
            positions: vec![DVec3::new(0.0, 0.0, 10.0), DVec3::new(1.0, 0.0, 10.0), DVec3::new(0.0, 1.0, 10.0), DVec3::new(13.0, 10.0, 0.0), DVec3::new(13.0, 10.0, 1.0), DVec3::new(10.0, 13.0, 0.0)],
            normals: vec![DVec3::Z; 6],
            triangles: vec![[0, 1, 2], [3, 4, 5]],
            tri_face: vec![0, 1],
            ..Default::default()
        };
        let (p, _) = moved_face_mesh(&mesh, &[(0, top())], 2.0, None).expect("a ghost");
        assert_eq!(p.len(), 3);
        assert!(p.iter().all(|q| (q.z - 12.0).abs() < 1e-12));
        let (p, _) = moved_face_mesh(&mesh, &[(1, bore())], 0.5, None).expect("a ghost");
        assert!((p[0] - DVec3::new(13.5, 10.0, 0.0)).length() < 1e-12);
        assert!(moved_face_mesh(&mesh, &[(7, top())], 2.0, None).is_none());
    }
}
