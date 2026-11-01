// SPDX-License-Identifier: AGPL-3.0-only
//! Where a Move puts the bodies it takes. Ported from ScanRuler
//! (`plugins/reverse/core/move.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The bodies are turned about a pivot — about X, then about Y, then about Z, each through the
//! pivot — and then carried along X, Y and Z, the six numbers its box shows. The pivot is the
//! middle of the bodies as they were picked, kept with the feature, so an earlier feature that
//! grows a body does not swing it somewhere else.
//!
//! The gizmo drags the same six numbers: an arrow along each axis for a distance, a ring for
//! each angle. Three angles turned one after another are a gimbal — the X ring rides on what
//! the Y and Z turns did, the Y ring on what the Z turn did — and the rings are drawn so: each
//! about the axis its own angle turns about, so the turn under the hand is the turn the body
//! makes and a drag changes one number only.
//!
//! Last, where a face move puts the faces it moves, on a body's mesh: the ghost drawn while
//! that move is an edit of the history.

use std::collections::HashMap;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::topo::{FaceFingerprint, FaceInfo, FaceKind};

/// A Move's numbers: what it turns about, how far it turns about each axis (degrees,
/// right-handed) and how far it carries the bodies after (mm).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub pivot: DVec3,
    pub translate: DVec3,
    pub rotate: DVec3,
}

/// A rigid move as the kernel takes it: row-major 3 × 4, the turn in the first three columns
/// of each row, the carry in the fourth.
pub type Affine = [f64; 12];

/// A row-major 3 × 3 matrix.
pub type Mat3 = [f64; 9];

const RAD: f64 = std::f64::consts::PI / 180.0;

fn rot_x(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]
}

fn rot_y(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
}

fn rot_z(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
}

/// Two row-major 3 × 3 matrices multiplied, a then b applied: a · b.
fn mul3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
    out
}

fn apply3(m: &Mat3, v: DVec3) -> DVec3 {
    DVec3::new(m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z, m[6] * v.x + m[7] * v.y + m[8] * v.z)
}

/// The turn of three angles in degrees — about X first, then Y, then Z, each about the part's
/// own axis: Rz · Ry · Rx — row-major 3 × 3.
pub fn rotation_of(rotate: DVec3) -> Mat3 {
    mul3(&rot_z(rotate.z * RAD), &mul3(&rot_y(rotate.y * RAD), &rot_x(rotate.x * RAD)))
}

/// The whole move, the kernel's way round: p′ = R (p − pivot) + pivot + translate.
pub fn move_matrix(p: &Placement) -> Affine {
    let r = rotation_of(p.rotate);
    let turned = apply3(&r, p.pivot);
    let t = DVec3::new(p.pivot.x + p.translate.x - turned.x, p.pivot.y + p.translate.y - turned.y, p.pivot.z + p.translate.z - turned.z);
    [r[0], r[1], r[2], t.x, r[3], r[4], r[5], t.y, r[6], r[7], r[8], t.z]
}

/// A 3 × 4 move as the viewport takes it: column-major 4 × 4.
pub fn matrix4_of(m: &Affine) -> [f64; 16] {
    [m[0], m[4], m[8], 0.0, m[1], m[5], m[9], 0.0, m[2], m[6], m[10], 0.0, m[3], m[7], m[11], 1.0]
}

/// A point carried by a move.
pub fn move_point(m: &Affine, p: DVec3) -> DVec3 {
    DVec3::new(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7], m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11])
}

/// A direction turned by a move — the carry leaves it alone.
pub fn move_dir(m: &Affine, d: DVec3) -> DVec3 {
    DVec3::new(m[0] * d.x + m[1] * d.y + m[2] * d.z, m[4] * d.x + m[5] * d.y + m[6] * d.z, m[8] * d.x + m[9] * d.y + m[10] * d.z)
}

/// Whether a placement's carry and turn leave everything where it was.
pub fn moves_nothing(translate: DVec3, rotate: DVec3) -> bool {
    translate.to_array().iter().all(|v| v.abs() < 1e-9) && rotate.to_array().iter().all(|v| (v % 360.0).abs() < 1e-9)
}

/// The axes the three rings turn about, X's, Y's and Z's: Z the part's own, Y as the Z turn
/// left it, X as the Z and Y turns left it — the axis each angle's own change turns the bodies
/// about (Rz·Ry·Rx·Rx(δ) is a turn by δ about Rz·Ry·X, and so on).
pub fn ring_axes(rotate: DVec3) -> [DVec3; 3] {
    let z = rot_z(rotate.z * RAD);
    let zy = mul3(&z, &rot_y(rotate.y * RAD));
    [apply3(&zy, DVec3::X), apply3(&z, DVec3::Y), DVec3::Z]
}

/// Where the gizmo stands: the pivot, carried with the bodies.
pub fn moved_pivot(p: &Placement) -> DVec3 {
    p.pivot + p.translate
}

/// A body's mesh as a move reads it: its points and normals, its triangles, and — on a body's
/// mesh — the face each triangle lies on, by the face's hash (`topo::FaceInfo::hash`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MoveMesh {
    pub positions: Vec<DVec3>,
    pub normals: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
    pub face: Option<Vec<u32>>,
}

/// Meshes as one, in their order: what a ghost of several bodies is drawn as. A face's hash is
/// kept where every mesh has them.
pub fn merge_meshes(meshes: &[MoveMesh]) -> MoveMesh {
    if meshes.len() == 1 {
        return meshes[0].clone();
    }
    let hashed = meshes.iter().all(|m| m.face.is_some());
    let mut out = MoveMesh { face: hashed.then(Vec::new), ..Default::default() };
    for m in meshes {
        let first = out.positions.len() as u32;
        out.positions.extend_from_slice(&m.positions);
        out.normals.extend_from_slice(&m.normals);
        out.triangles.extend(m.triangles.iter().map(|t| [t[0] + first, t[1] + first, t[2] + first]));
        if let (Some(all), Some(own)) = (out.face.as_mut(), m.face.as_ref()) {
            all.extend_from_slice(own);
        }
    }
    out
}

/// A mesh carried by a move: a copy, its points moved and its normals turned.
pub fn moved_mesh(mesh: &MoveMesh, m: &Affine) -> MoveMesh {
    MoveMesh {
        positions: mesh.positions.iter().map(|p| move_point(m, *p)).collect(),
        normals: mesh.normals.iter().map(|n| move_dir(m, *n)).collect(),
        triangles: mesh.triangles.clone(),
        face: mesh.face.clone(),
    }
}

/// `Math.round` as JavaScript has it: halves round up, towards +∞.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// The middle of the boxes (min, max) round some bodies — what a Move turns them about — or
/// the origin with none.
pub fn pivot_of(boxes: &[(DVec3, DVec3)]) -> DVec3 {
    if boxes.is_empty() {
        return DVec3::ZERO;
    }
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for (lo, hi) in boxes {
        min = min.min(*lo);
        max = max.max(*hi);
    }
    // Rounded to the micron: the pivot is saved and read back, and a number with sixteen
    // digits says nothing the micron does not.
    let tidy = |v: f64| js_round(v * 1000.0) / 1000.0;
    DVec3::new(tidy((min.x + max.x) / 2.0), tidy((min.y + max.y) / 2.0), tidy((min.z + max.z) / 2.0))
}

/// The draft that follows a single snapped face as it moves: the face turned by `angle_deg`
/// about the line through `origin`, along the face and square to `pull`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lean {
    pub origin: DVec3,
    pub pull: DVec3,
    pub angle_deg: f64,
}

/// A vector made unit, or `None` for one shorter than 1e-12.
fn normalize(a: DVec3) -> Option<DVec3> {
    let l = a.length();
    (l >= 1e-12).then(|| a / l)
}

/// Faces of a body's mesh, moved: a flat face along its normal, a round one out from its axis —
/// where faces being moved will lie, drawn as a ghost when the move is an edit of the history
/// and there is no one step for the kernel to preview. `None` when the mesh names none of the
/// faces. A lean is the draft that follows a single snapped face.
pub fn moved_face_mesh(mesh: &MoveMesh, faces: &[FaceInfo], offset: f64, lean: Option<Lean>) -> Option<MoveMesh> {
    let owner = mesh.face.as_ref()?;
    let by_hash: HashMap<u32, &FaceFingerprint> = faces.iter().map(|f| (f.hash, &f.fingerprint)).collect();
    // The draft that follows the move turns the face about the line through the lean's origin,
    // along the face and square to the pull, so that the normal tips towards the pull:
    // Rodrigues about the axis dir × pull.
    let turn = match (lean, faces) {
        (Some(l), [only]) => normalize(only.fingerprint.dir.cross(l.pull)).map(|axis| (axis, l)),
        _ => None,
    };
    let rotate = |v: DVec3| -> DVec3 {
        let Some((axis, l)) = turn else { return v };
        let t = (l.angle_deg * std::f64::consts::PI) / 180.0;
        let (s, c) = t.sin_cos();
        let k = axis.cross(v);
        let d = axis.dot(v) * (1.0 - c);
        DVec3::new(v.x * c + k.x * s + axis.x * d, v.y * c + k.y * s + axis.y * d, v.z * c + k.z * s + axis.z * d)
    };
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for (t, hash) in owner.iter().enumerate() {
        let Some(face) = by_hash.get(hash) else { continue };
        let Some(tri) = mesh.triangles.get(t) else { continue };
        // A triangle naming a point the mesh does not have is left out whole.
        if tri.iter().any(|&v| v as usize >= mesh.positions.len() || v as usize >= mesh.normals.len()) {
            continue;
        }
        for &v in tri {
            let (p, n) = (mesh.positions[v as usize], mesh.normals[v as usize]);
            let mut along = face.dir;
            if let (FaceKind::Cylinder, Some(center)) = (face.kind, face.center) {
                let d = p - center;
                let h = d.dot(face.dir);
                along = normalize(d - face.dir * h).unwrap_or(face.dir);
            }
            let moved = p + along * offset;
            let q = match turn {
                Some((_, l)) => l.origin + rotate(moved - l.origin),
                None => moved,
            };
            positions.push(q);
            normals.push(rotate(n));
        }
    }
    if positions.is_empty() {
        return None;
    }
    let triangles = (0..positions.len() as u32 / 3).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect();
    Some(MoveMesh { positions, normals, triangles, face: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expect_vec(a: &[f64], b: &[f64], digits: i32) {
        assert_eq!(a.len(), b.len());
        let tol = 10f64.powi(-digits) / 2.0;
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < tol, "{a:?} against {b:?}");
        }
    }

    #[test]
    fn turns_about_the_pivot_x_then_y_then_z_and_carries_after() {
        let turn = move_matrix(&Placement { pivot: DVec3::new(10.0, 0.0, 0.0), translate: DVec3::ZERO, rotate: DVec3::new(0.0, 0.0, 90.0) });
        expect_vec(&move_point(&turn, DVec3::new(20.0, 0.0, 0.0)).to_array(), &[10.0, 10.0, 0.0], 6);
        expect_vec(&move_point(&turn, DVec3::new(10.0, 0.0, 0.0)).to_array(), &[10.0, 0.0, 0.0], 6);
        let both = move_matrix(&Placement { pivot: DVec3::ZERO, translate: DVec3::new(5.0, 0.0, 0.0), rotate: DVec3::new(90.0, 0.0, 90.0) });
        // (0, 1, 0) about X by 90° is (0, 0, 1); about Z it stays; then +5 in X.
        expect_vec(&move_point(&both, DVec3::new(0.0, 1.0, 0.0)).to_array(), &[5.0, 0.0, 1.0], 6);
        // (1, 0, 0) stays under X, goes to (0, 1, 0) under Z.
        expect_vec(&move_point(&both, DVec3::new(1.0, 0.0, 0.0)).to_array(), &[5.0, 1.0, 0.0], 6);
        assert!(moves_nothing(DVec3::ZERO, DVec3::new(0.0, 360.0, 0.0)));
        assert!(!moves_nothing(DVec3::new(0.0, 0.0, 1e-3), DVec3::ZERO));
    }

    #[test]
    fn stands_each_ring_about_the_axis_its_own_angle_turns_about() {
        let rotate = DVec3::new(20.0, -35.0, 50.0);
        let [ax, ay, az] = ring_axes(rotate);
        let turn_about = |axis: DVec3, deg: f64| -> Mat3 {
            // Rodrigues: a turn by deg about a unit axis, row-major 3 × 3.
            let t = (deg * std::f64::consts::PI) / 180.0;
            let (c, s) = (t.cos(), t.sin());
            let k = 1.0 - c;
            let (x, y, z) = (axis.x, axis.y, axis.z);
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s, y * x * k + z * s, c + y * y * k, y * z * k - x * s, z * x * k - y * s, z * y * k + x * s, c + z * z * k]
        };
        let base = rotation_of(rotate);
        // A few degrees more of one angle is that turn about its ring's axis, after everything
        // else.
        expect_vec(&rotation_of(DVec3::new(27.0, -35.0, 50.0)), &mul3(&turn_about(ax, 7.0), &base), 9);
        expect_vec(&rotation_of(DVec3::new(20.0, -28.0, 50.0)), &mul3(&turn_about(ay, 7.0), &base), 9);
        expect_vec(&rotation_of(DVec3::new(20.0, -35.0, 57.0)), &mul3(&turn_about(az, 7.0), &base), 9);
        // With nothing turned the rings stand about X, Y and Z themselves.
        let flat: Vec<f64> = ring_axes(DVec3::ZERO).iter().flat_map(|v| v.to_array()).collect();
        expect_vec(&flat, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 6);
    }

    #[test]
    fn carries_a_mesh_and_puts_meshes_together_and_finds_the_middle_of_boxes() {
        let tri = MoveMesh {
            positions: vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0)],
            normals: vec![DVec3::Z; 3],
            triangles: vec![[0, 1, 2]],
            face: None,
        };
        let moved = moved_mesh(&tri, &move_matrix(&Placement { pivot: DVec3::ZERO, translate: DVec3::new(0.0, 0.0, 5.0), rotate: DVec3::new(90.0, 0.0, 0.0) }));
        let flat: Vec<f64> = moved.positions[1..3].iter().flat_map(|v| v.to_array()).collect();
        expect_vec(&flat, &[1.0, 0.0, 5.0, 0.0, 0.0, 6.0], 6);
        expect_vec(&moved.normals[0].to_array(), &[0.0, -1.0, 0.0], 6);
        let both = merge_meshes(&[tri, moved]);
        assert_eq!(both.triangles, vec![[0, 1, 2], [3, 4, 5]]);
        assert_eq!(matrix4_of(&move_matrix(&Placement { pivot: DVec3::ZERO, translate: DVec3::new(1.0, 2.0, 3.0), rotate: DVec3::ZERO }))[12..], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(pivot_of(&[(DVec3::ZERO, DVec3::splat(2.0)), (DVec3::new(10.0, -2.0, 0.0), DVec3::new(12.0, 0.0, 4.0))]), DVec3::new(6.0, 0.0, 2.0));
    }

    /// Not a ScanRuler test: a face of a mesh moved along its normal, the rest left out.
    #[test]
    fn moves_the_named_face_along_its_normal() {
        let mesh = MoveMesh {
            positions: vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::new(0.0, 0.0, 1.0), DVec3::new(1.0, 0.0, 1.0), DVec3::new(0.0, 1.0, 1.0)],
            normals: vec![-DVec3::Z, -DVec3::Z, -DVec3::Z, DVec3::Z, DVec3::Z, DVec3::Z],
            triangles: vec![[0, 2, 1], [3, 4, 5]],
            face: Some(vec![0, 1]),
        };
        let top = FaceInfo {
            hash: 1,
            fingerprint: FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(0.3, 0.3, 1.0), dir: DVec3::Z, area: 0.5, radius: None, center: None, origin: 1 },
            full_turn: None,
            point: DVec3::new(0.3, 0.3, 1.0),
        };
        let ghost = moved_face_mesh(&mesh, &[top], 2.0, None).expect("a ghost");
        assert_eq!(ghost.positions, vec![DVec3::new(0.0, 0.0, 3.0), DVec3::new(1.0, 0.0, 3.0), DVec3::new(0.0, 1.0, 3.0)]);
        assert_eq!(ghost.triangles, vec![[0, 1, 2]]);
        assert!(moved_face_mesh(&MoveMesh { face: None, ..mesh }, &[], 2.0, None).is_none());
    }
}
