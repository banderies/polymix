// SPDX-License-Identifier: AGPL-3.0-only
//! The Move face feature's build: faces of a body moved by one offset — a flat face along its
//! outward normal, a full cylinder's radius changed by it, the axis kept. Several faces picked
//! together are a whole made larger or smaller: a cylinder's wall and its ends, every face of a
//! block. What a face snapped onto the scan becomes, too.
//!
//! A flat face is moved as ScanRuler moves it: the face's own outline swept by the offset and
//! joined to the body (out) or cut from it (in), so the walls round the face run on as they
//! stand — walls square to the face carry on, a wall that leans does not follow its lean, and a
//! round on the face's rim stays behind as a pad or a pocket. A round face is resized by a
//! tube between the two radii over the face's length, joined or cut (a bore bored out is cut
//! with the whole cylinder: nothing is in it to mind). The faces are moved one after the other,
//! each found again on the body the one before it left, by the surface it lies on.
//!
//! Ported from ScanRuler (`plugins/reverse/core/kernelFaces.ts` moveFace, offsetFacesTracked;
//! `core/kernelOps.ts` readRound; `core/kernelMath.ts` sameSurface; `core/resolve.ts` and
//! `core/build.ts` the moveFace step), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only, on the
//! SolidCraft kernel: the face swept by `face_prism`, joined or cut by its boolean.

use solidcraft_geom::{Loop2, Plane, Region2, Vec2 as SVec2};
use solidcraft_kernel::{self as kernel, BoolOp, Body, FaceSurface};

use super::{dv3, kerr, origins_after, read_topology, sv3, Builder};
use crate::model::{Feature, MoveFaceFeature, Part};
use crate::snap_face::{halves_of, same_face};
use crate::topo::{FaceFingerprint, FaceKind, Topology};

/// Whether Move face takes a face: a flat one, or a round one with a radius to change (that it
/// goes the whole way round is the kernel's to tell).
pub fn movable_face(face: &FaceFingerprint) -> bool {
    matches!(face.kind, FaceKind::Plane | FaceKind::Cylinder)
}

pub(crate) fn build(b: &mut Builder, _part: &Part, _f: &Feature, x: &MoveFaceFeature) -> Result<(), String> {
    if x.faces.is_empty() {
        return Err("No face is picked to move.".into());
    }
    if x.faces.iter().any(|f| !movable_face(f)) {
        return Err("Only a flat or a cylindrical face can be moved.".into());
    }
    if !(x.offset.abs() > 1e-9) {
        return Err("The face has nowhere to move — the offset is zero.".into());
    }
    let (id, state) = b.body_for(x.body)?;
    let found = b.find_faces(id, &x.faces, &x.faces_where).map_err(|lost| format!("A face this feature moves is gone (was: {lost})."))?;
    let t = b.topology_of(id).unwrap_or_else(|| state.topology());
    let (made, origins) = move_faces(&state.shape, &state.origins, &t, &found, x.offset, b.step).map_err(|e| format!("Moving the faces: {e}"))?;
    if !made.validity().is_empty() || made.face_count() == 0 {
        return Err(format!("The {} left no closed solid.", if x.faces.len() == 1 { "moved face" } else { "moved faces" }));
    }
    let ids = b.settle(id, made, origins, false)?;
    b.touched(ids);
    Ok(())
}

/// The face of `candidates` on the surface `was` lay on, the nearest where several are
/// (ScanRuler's `sameSurface`).
pub(crate) fn same_surface(was: &FaceFingerprint, candidates: &[FaceFingerprint]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, f) in candidates.iter().enumerate() {
        if f.kind != was.kind {
            continue;
        }
        if was.kind == FaceKind::Plane {
            if f.dir.dot(was.dir) < 1.0 - 1e-6 || (f.centroid - was.centroid).dot(was.dir).abs() > 1e-4 {
                continue;
            }
        } else {
            if f.dir.dot(was.dir).abs() < 1.0 - 1e-6 {
                continue;
            }
            if let Some(r) = was.radius {
                if f.radius.is_none_or(|x| (x - r).abs() > 1e-4) {
                    continue;
                }
            }
            if let (Some(a), Some(c)) = (was.center, f.center) {
                let off = c - a;
                let along = off.dot(was.dir);
                let across = off - was.dir * along;
                // A sphere's centre is a point; an axis may be read anywhere along it.
                if across.length() > 1e-4 || (was.kind == FaceKind::Sphere && along.abs() > 1e-4) {
                    continue;
                }
            }
        }
        let d = (f.centroid - was.centroid).length();
        if best.is_none_or(|(_, x)| d < x) {
            best = Some((i, d));
        }
    }
    best.map(|b| b.0)
}

/// Faces of a solid moved by one offset (ScanRuler's `offsetFacesTracked`): one at a time, each
/// found again by its surface on the body the one before it left. The halves the kernel keeps a
/// whole bore as are one face to move; two faces on one surface apart (two tops in line either
/// side of a slot) are two. The body and its faces' makers.
pub(crate) fn move_faces(body: &Body, origins: &[i64], t: &Topology, faces: &[usize], offset: f64, step: u32) -> Result<(Body, Vec<i64>), String> {
    if faces.is_empty() {
        return Err("no face given.".into());
    }
    let mut taken: Vec<usize> = Vec::new();
    let mut wanted: Vec<FaceFingerprint> = Vec::new();
    for i in faces {
        let f = t.faces.get(*i).ok_or("the face is not on the body.")?;
        if !taken.iter().any(|k| same_face(t, *k, *i)) {
            taken.push(*i);
            wanted.push(f.fingerprint);
        }
    }
    let mut shape = body.clone();
    let mut made_by: Vec<i64> = origins.to_vec();
    for (k, w) in wanted.iter().enumerate() {
        let (at, topo) = if k == 0 {
            (taken[0], None)
        } else {
            let now = read_topology(&shape, &made_by);
            let prints: Vec<FaceFingerprint> = now.faces.iter().map(|f| f.fingerprint).collect();
            let i = same_surface(w, &prints).ok_or("a face to move was lost after the one before it was moved.")?;
            (i, Some(now))
        };
        let full = match &topo {
            Some(now) => now.faces.get(at).and_then(|f| f.full_turn),
            None => t.faces.get(at).and_then(|f| f.full_turn),
        };
        let (next, next_origins) = move_face(&shape, &made_by, at, full, offset, step)?;
        shape = next;
        made_by = next_origins;
    }
    Ok((shape, made_by))
}

/// One face of a solid moved (ScanRuler's `moveFace`): a flat face along its outward normal by
/// `offset` millimetres — its own outline swept and joined (out) or cut away (in) — or a round
/// face's radius changed by `offset`, the axis kept, over the length the face has. `full` says
/// whether a round face goes the whole way round, where the body's reading knows.
fn move_face(body: &Body, origins: &[i64], face: usize, full: Option<bool>, offset: f64, step: u32) -> Result<(Body, Vec<i64>), String> {
    if !(offset.abs() > 1e-9) {
        return Err("the face has nowhere to move — the offset is zero.".into());
    }
    let own = |b: &Body| vec![step as i64; b.face_count()];
    match kernel::face_surface(body, face) {
        Some(FaceSurface::Plane { normal, .. }) => {
            let n = dv3(normal.into());
            let whole = kernel::WholeFace { face, normal: sv3(n), height: 0.0 };
            let prism = kernel::face_prism(body, whole, offset.abs(), offset > 0.0, body).map_err(kerr)?;
            let op = if offset > 0.0 { BoolOp::Union } else { BoolOp::Cut };
            let made = kernel::boolean(body, &prism, op).map_err(kerr)?.ok_or("the move leaves nothing of the body.")?;
            let o = origins_after(&made, &[(body, &origins.to_vec()), (&prism, &own(&prism))], step);
            Ok((made, o))
        }
        Some(FaceSurface::Cylinder { radius: r0, .. }) => {
            if full == Some(false) {
                return Err("the round face does not go the whole way round — only a full cylinder can be resized.".into());
            }
            let mesh = body.tessellate((body.size() * 1e-3).max(1e-3)).map_err(kerr)?;
            let c = kernel::cylinder_face(body, &mesh, face).ok_or("the round face does not go the whole way round — only a full cylinder can be resized.")?;
            // The round as a whole: its halves, and the pieces of its wall an end moved out
            // before it left on its surface — one face to the kernel ScanRuler builds on, so
            // resized over the length of them all.
            let (base0, axis0) = (dv3(c.axis_point), dv3(c.axis.into()));
            let (mut lo, mut hi) = (c.start.min(c.end), c.start.max(c.end));
            for other in halves_of(&read_topology(body, origins), face) {
                let Some(o) = kernel::cylinder_face(body, &mesh, other).filter(|_| other != face) else { continue };
                for s in [o.start, o.end] {
                    let at = (dv3(o.axis_point) + dv3(o.axis.into()) * s - base0).dot(axis0);
                    lo = lo.min(at);
                    hi = hi.max(at);
                }
            }
            if !(hi - lo > 1e-9) {
                return Err("the round face has no length.".into());
            }
            let radius = r0 + offset;
            if !(radius > 1e-6) {
                return Err("the radius would not be positive.".into());
            }
            let axis = dv3(c.axis.into());
            let base = dv3(c.axis_point) + axis * lo;
            let plane = Plane::from_normal(sv3(base), sv3(axis)).ok_or("the round face has no axis.")?;
            // A hole bored out is cut with the whole cylinder — nothing is in it to mind; every
            // other case is a tube between the two radii, so a boss with a bore keeps its bore.
            let grow = offset > 0.0;
            let convex = !c.internal;
            let circle = |r: f64| Loop2::circle(SVec2::new(0.0, 0.0), r);
            let profile = if !convex && grow { Region2 { outer: circle(radius), holes: Vec::new() } } else { Region2 { outer: circle(radius.max(r0)), holes: vec![circle(radius.min(r0))] } };
            let tool = kernel::extrude(&plane, &[profile], 0.0, hi - lo).map_err(kerr)?.pop().ok_or("the tube to resize the face with could not be made.")?;
            let op = if convex == grow { BoolOp::Union } else { BoolOp::Cut };
            let made = kernel::boolean(body, &tool, op).map_err(kerr)?.ok_or("the move leaves nothing of the body.")?;
            let o = origins_after(&made, &[(body, &origins.to_vec()), (&tool, &own(&tool))], step);
            Ok((made, o))
        }
        _ => Err("only a flat or a cylindrical face can be moved.".into()),
    }
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reMoveFaces.test.ts`: several faces of a body moved by one offset — a
    //! cylinder made longer and wider at once, and a box grown and shrunk on every side.
    use super::*;
    use crate::snap_face::one_surface;
    use glam::DVec3;
    use solidcraft_geom::Vec3 as SVec3;

    fn vol(b: &Body) -> f64 {
        super::super::volume(b).unwrap_or(f64::NAN)
    }

    fn moved(body: &Body, pick: impl Fn(&FaceFingerprint) -> bool, offset: f64) -> Result<Body, String> {
        let origins = vec![1; body.face_count()];
        let t = read_topology(body, &origins);
        let faces: Vec<usize> = t.faces.iter().enumerate().filter(|(_, f)| pick(&f.fingerprint)).map(|(i, _)| i).collect();
        move_faces(body, &origins, &t, &faces, offset, 9).map(|(b, _)| b)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-3 * b.abs().max(1.0)
    }

    fn cylinder(r: f64, length: f64) -> Body {
        kernel::cylinder(SVec3::new(0.0, 0.0, 0.0), SVec3::new(0.0, 0.0, 1.0), r, length).unwrap_or_else(|e| panic!("{e}"))
    }

    fn block(w: f64, d: f64, h: f64) -> Body {
        kernel::box_solid(SVec3::new(0.0, 0.0, 0.0), SVec3::new(w, d, h)).unwrap_or_else(|e| panic!("{e}"))
    }

    fn surfaces(b: &Body) -> usize {
        let t = read_topology(b, &vec![1; b.face_count()]);
        let mut kept: Vec<FaceFingerprint> = Vec::new();
        for f in &t.faces {
            if !kept.iter().any(|k| one_surface(k, &f.fingerprint)) {
                kept.push(f.fingerprint);
            }
        }
        kept.len()
    }

    #[test]
    fn makes_a_cylinder_longer_and_wider_at_once_wall_and_both_ends_together() {
        let pi = std::f64::consts::PI;
        let c = cylinder(4.0, 10.0);
        let base = vol(&c);
        // The kernel's volume of a round body reads a hair under the exact one.
        let k = base / (pi * 16.0 * 10.0);
        let grown = moved(&c, |_| true, 0.5).unwrap_or_else(|e| panic!("{e}"));
        assert!(grown.validity().is_empty());
        assert!(close(vol(&grown), k * pi * 4.5 * 4.5 * 11.0), "{}", vol(&grown));
        assert_eq!(surfaces(&grown), 3);
        let bb = grown.outline_bounds();
        assert!((bb.min.z + 0.5).abs() < 1e-6 && (bb.max.z - 10.5).abs() < 1e-6, "{bb:?}");
        let shrunk = moved(&c, |_| true, -1.0).unwrap_or_else(|e| panic!("{e}"));
        assert!(close(vol(&shrunk), k * pi * 9.0 * 8.0), "{}", vol(&shrunk));
        assert_eq!(surfaces(&shrunk), 3);
    }

    #[test]
    fn lengthens_a_cylinder_by_one_end_alone_and_grows_a_box_on_every_side() {
        let pi = std::f64::consts::PI;
        let c = cylinder(4.0, 10.0);
        let k = vol(&c) / (pi * 16.0 * 10.0);
        let longer = moved(&c, |f| f.kind == FaceKind::Plane && f.dir.z > 0.9, 15.0).unwrap_or_else(|e| panic!("{e}"));
        assert!(close(vol(&longer), k * pi * 16.0 * 25.0), "{}", vol(&longer));
        let cube = block(40.0, 20.0, 10.0);
        let big = moved(&cube, |_| true, 1.0).unwrap_or_else(|e| panic!("{e}"));
        assert!(close(vol(&big), 42.0 * 22.0 * 12.0), "{}", vol(&big));
        assert_eq!(big.face_count(), 6);
        let small = moved(&cube, |_| true, -2.0).unwrap_or_else(|e| panic!("{e}"));
        assert!(close(vol(&small), 36.0 * 16.0 * 6.0), "{}", vol(&small));
        assert_eq!(small.face_count(), 6);
    }

    #[test]
    fn bores_out_two_holes_of_a_plate_together_and_refuses_a_round_face_that_is_not_a_whole_cylinder() {
        let pi = std::f64::consts::PI;
        let mut plate = block(40.0, 20.0, 10.0);
        for x in [10.0, 30.0] {
            let pin = kernel::cylinder(SVec3::new(x, 10.0, -1.0), SVec3::new(0.0, 0.0, 1.0), 3.0, 12.0).unwrap_or_else(|e| panic!("{e}"));
            plate = kernel::boolean(&plate, &pin, BoolOp::Cut).ok().flatten().unwrap_or_else(|| panic!("the hole"));
        }
        let holed = vol(&plate);
        assert!(close(holed, 8000.0 - 2.0 * pi * 9.0 * 10.0), "{holed}");
        let bored = moved(&plate, |f| f.kind == FaceKind::Cylinder, 0.5).unwrap_or_else(|e| panic!("{e}"));
        assert!(close(vol(&bored), 8000.0 - 2.0 * pi * 3.5 * 3.5 * 10.0), "{}", vol(&bored));
        // A round on an edge is a quarter of a cylinder: it has no radius of its own to change.
        let rounded = kernel::fillet(&plate, &[SVec3::new(20.0, 0.0, 10.0)], 2.0).unwrap_or_else(|e| panic!("{e}"));
        let e = moved(&rounded, |f| (f.kind == FaceKind::Cylinder && (f.radius.unwrap_or(0.0) - 2.0).abs() < 1e-6) || (f.kind == FaceKind::Plane && f.dir.z > 0.9), 0.5).unwrap_err();
        assert!(e.contains("whole way round"), "{e}");
    }

    #[test]
    fn keeps_a_tilted_cylinder_three_faces_through_move_after_move() {
        // A fitted cylinder's axis is never a coordinate axis: its wall pushed out twice, an end
        // pushed out, in again — no ring left on an end, no seam round the wall.
        let pi = std::f64::consts::PI;
        let n = DVec3::new(0.3007679386167831, 0.20051195907785532, 0.9323806097120273);
        let base = DVec3::new(1.0, 2.0, 3.0) + n * -2.0;
        let mut body = kernel::cylinder(sv3(base), sv3(n), 5.7, 11.0).unwrap_or_else(|e| panic!("{e}"));
        let k = vol(&body) / (pi * 5.7 * 5.7 * 11.0);
        let (mut r, mut length) = (5.7, 11.0);
        for (which, offset) in [("wall", 1.3), ("wall", 0.7), ("top", 0.6), ("bottom", 0.9), ("wall", -0.4), ("bottom", -0.3)] {
            let pick = |f: &FaceFingerprint| match which {
                "wall" => f.kind == FaceKind::Cylinder,
                "top" => f.kind == FaceKind::Plane && f.dir.dot(n) > 0.0,
                _ => f.kind == FaceKind::Plane && f.dir.dot(n) < 0.0,
            };
            body = moved(&body, pick, offset).unwrap_or_else(|e| panic!("{which} {offset}: {e}"));
            if which == "wall" {
                r += offset;
            } else {
                length += offset;
            }
            assert_eq!(surfaces(&body), 3, "{which} {offset}");
            assert!((vol(&body) - k * pi * r * r * length).abs() / vol(&body) < 2e-3, "{which} {offset}: {} vs {}", vol(&body), pi * r * r * length);
        }
    }

    #[test]
    fn takes_the_faces_in_any_order_the_ends_first_the_wall_last() {
        let pi = std::f64::consts::PI;
        let c = cylinder(4.0, 10.0);
        let k = vol(&c) / (pi * 16.0 * 10.0);
        let origins = vec![1; c.face_count()];
        let t = read_topology(&c, &origins);
        let mut order: Vec<usize> = (0..t.faces.len()).filter(|i| t.faces[*i].fingerprint.kind == FaceKind::Plane).collect();
        order.extend((0..t.faces.len()).filter(|i| t.faces[*i].fingerprint.kind == FaceKind::Cylinder));
        let (grown, _) = move_faces(&c, &origins, &t, &order, 0.5, 9).unwrap_or_else(|e| panic!("{e}"));
        assert!(close(vol(&grown), k * pi * 4.5 * 4.5 * 11.0), "{}", vol(&grown));
        assert_eq!(surfaces(&grown), 3);
    }

    #[test]
    fn moves_two_tops_in_line_either_side_of_a_slot_each_and_a_bore_s_two_halves_once() {
        // A 40 × 20 × 10 block with a slot 10 wide and 5 deep across its middle: its top is two
        // faces on one plane, apart. Both picked, both rise.
        let slot = kernel::box_solid(SVec3::new(15.0, -1.0, 5.0), SVec3::new(25.0, 21.0, 11.0)).unwrap_or_else(|e| panic!("{e}"));
        let body = kernel::boolean(&block(40.0, 20.0, 10.0), &slot, BoolOp::Cut).ok().flatten().unwrap_or_else(|| panic!("the slot"));
        assert!(close(vol(&body), 7000.0), "{}", vol(&body));
        let tops = |f: &FaceFingerprint| f.kind == FaceKind::Plane && f.dir.z > 0.9 && (f.centroid.z - 10.0).abs() < 1e-6;
        let t = read_topology(&body, &vec![1; body.face_count()]);
        assert_eq!(t.faces.iter().filter(|f| tops(&f.fingerprint)).count(), 2);
        let raised = moved(&body, tops, 2.0).unwrap_or_else(|e| panic!("{e}"));
        assert!(close(vol(&raised), 7000.0 + 2.0 * 15.0 * 20.0 * 2.0), "{}", vol(&raised));
        // A bore is kept as two halves on one surface, joined: picked both, it is bored once.
        let pi = std::f64::consts::PI;
        let pin = kernel::cylinder(SVec3::new(10.0, 10.0, -1.0), SVec3::new(0.0, 0.0, 1.0), 3.0, 12.0).unwrap_or_else(|e| panic!("{e}"));
        let plate = kernel::boolean(&block(40.0, 20.0, 10.0), &pin, BoolOp::Cut).ok().flatten().unwrap_or_else(|| panic!("the hole"));
        let halves = read_topology(&plate, &vec![1; plate.face_count()]).faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Cylinder).count();
        assert_eq!(halves, 2);
        let bored = moved(&plate, |f| f.kind == FaceKind::Cylinder, 0.5).unwrap_or_else(|e| panic!("{e}"));
        assert!(close(vol(&bored), 8000.0 - pi * 3.5 * 3.5 * 10.0), "{}", vol(&bored));
    }

    #[test]
    fn reads_a_move_face_saved_with_its_one_face() {
        // ScanRuler's projectSection.ts: a Move face saved before several faces could be picked.
        let json = r#"{"body":2,"face":{"kind":"plane","centroid":[20,10,10],"dir":[0,0,1],"area":800,"origin":2},"offset":1.5}"#;
        let x: MoveFaceFeature = serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(x.faces.len(), 1);
        assert_eq!(x.faces[0].centroid, DVec3::new(20.0, 10.0, 10.0));
        assert_eq!(x.offset, 1.5);
        // Written again, it holds the list.
        let again = serde_json::to_string(&x).unwrap_or_default();
        assert!(again.contains("\"faces\"") && !again.contains("\"face\""), "{again}");
        let back: MoveFaceFeature = serde_json::from_str(&again).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, x);
    }

    #[test]
    fn says_why_a_face_cannot_move() {
        let cube = block(40.0, 20.0, 10.0);
        assert!(moved(&cube, |f| f.dir.z > 0.9, 0.0).unwrap_err().contains("nowhere to move"));
        let c = cylinder(4.0, 10.0);
        assert!(moved(&c, |f| f.kind == FaceKind::Cylinder, -5.0).unwrap_err().contains("radius would not be positive"));
    }
}
