// SPDX-License-Identifier: AGPL-3.0-only
//! A body's triangles with every face there: a mesher can leave a face bare or drawn in small
//! part now and then, which only the face's area against its triangles' shows, and such a face
//! is drawn again on its own. Ported from ScanRuler (`plugins/reverse/core/kernelMesh.ts`
//! meshEveryFace, meshFaultOut), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A mesh with a face missing is not closed: an STL with a hole in it, a comparison that sees
//! through it, a volume that reads material all the way up over it. A face under
//! `MESHED_SHARE` of its area is meshed again — at twice the chord and on, then the reference
//! way — and its triangles put in the place of what it had; the rest of the mesh keeps the
//! chord asked. Its rim is its own, so it meets the faces beside it to within the chord, not
//! vertex to vertex.
//!
//! A face's area is the larger of what the display mesh and the kernel's reference mesh give
//! it (SolidCraft reads a face's area off its triangles); a face both leave bare is drawn by
//! whichever try covers it most.

use glam::DVec3;
use solidcraft_geom::{Mesh, Vec3 as SVec3};
use solidcraft_kernel::Body;

use super::dv3;
use crate::js_number::{js_round, js_string, to_precision_number};

/// A face whose triangles cover less than this share of its area has not been drawn.
const MESHED_SHARE: f64 = 0.75;
/// A face's own triangles are refused over this share of its area: they stand off it.
const OVERDRAWN_SHARE: f64 = 1.5;
/// Under this a face has no area to draw, mm².
const NO_FACE_MM2: f64 = 1e-6;
/// Over this a mesh's triangles do not sum to nothing, mm²: it is not closed.
const OPEN_MM2: f64 = 1e-3;
/// How many times a face is meshed again at twice the chord, and at what angles after that.
const CHORD_DOUBLINGS: u32 = 7;
const COARSER_ANGLES: [f64; 2] = [0.5, 1.0];

/// A face the mesher did not draw whole with the others.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshFault {
    /// The face, by its index in the kernel's order.
    pub face: usize,
    /// The middle of the face, mm.
    pub at: DVec3,
    /// Its area and what its triangles cover, mm².
    pub area: f64,
    pub drawn: f64,
    /// The chord it was drawn to on its own, where one did it whole.
    pub chord: Option<f64>,
}

fn tri_area(m: &Mesh, t: &[u32; 3]) -> (f64, DVec3, DVec3) {
    let p = |i: u32| m.positions.get(i as usize).copied().map(dv3).unwrap_or_default();
    let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
    let n = (b - a).cross(c - a);
    (n.length() * 0.5, (a + b + c) / 3.0, n * 0.5)
}

/// Each face's area as a mesh draws it, and the middle of what it draws.
fn covered(m: &Mesh, faces: usize) -> (Vec<f64>, Vec<DVec3>) {
    let mut area = vec![0.0; faces];
    let mut mid = vec![DVec3::ZERO; faces];
    for (t, f) in m.triangles.iter().zip(&m.tri_face) {
        let f = *f as usize;
        if f >= faces {
            continue;
        }
        let (a, c, _) = tri_area(m, t);
        area[f] += a;
        mid[f] += c * a;
    }
    for f in 0..faces {
        if area[f] > 0.0 {
            mid[f] /= area[f];
        }
    }
    (area, mid)
}

/// Whether a mesh's triangles sum to nothing: a closed mesh's do.
fn closes(m: &Mesh) -> bool {
    let sum = m.triangles.iter().fold(DVec3::ZERO, |s, t| s + tri_area(m, t).2);
    sum.length() <= OPEN_MM2
}

/// `mesh` with face `face`'s triangles taken from `piece` in place of its own.
fn with_face(mesh: &Mesh, face: usize, piece: &Mesh) -> Mesh {
    let mut out = mesh.clone();
    let kept: Vec<usize> = (0..mesh.triangles.len()).filter(|i| mesh.tri_face.get(*i).copied() != Some(face as u32)).collect();
    out.triangles = kept.iter().map(|i| mesh.triangles[*i]).collect();
    out.tri_face = kept.iter().map(|i| mesh.tri_face[*i]).collect();
    let mut index: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (t, f) in piece.triangles.iter().zip(&piece.tri_face) {
        if *f as usize != face {
            continue;
        }
        let mut tri = [0u32; 3];
        for (k, v) in t.iter().enumerate() {
            tri[k] = *index.entry(*v).or_insert_with(|| {
                out.positions.push(piece.positions.get(*v as usize).copied().unwrap_or(SVec3::new(0.0, 0.0, 0.0)));
                out.normals.push(piece.normals.get(*v as usize).copied().unwrap_or(SVec3::new(0.0, 0.0, 1.0)));
                (out.positions.len() - 1) as u32
            });
        }
        out.triangles.push(tri);
        out.tri_face.push(face as u32);
    }
    out
}

/// The body's display mesh with a triangle on every face it can have (ScanRuler's
/// `meshEveryFace`): `first` as it came, unless a face is bare or drawn in small part — then
/// such a face is meshed again on its own and put in. The faces that were, each said.
pub fn mesh_every_face(body: &Body, first: Mesh, chord: f64) -> (Mesh, Vec<MeshFault>) {
    let faces = body.face_count();
    let (drawn, mid) = covered(&first, faces);
    if drawn.iter().all(|a| *a > 0.0) && closes(&first) {
        return (first, Vec::new());
    }
    // What the reference mesher gives each face: the face's area where the display mesh fell
    // short of it.
    let reference = body.tessellate(chord).ok();
    let (ref_area, ref_mid) = reference.as_ref().map_or((vec![0.0; faces], vec![DVec3::ZERO; faces]), |m| covered(m, faces));
    let mut tries: Vec<(Option<f64>, Option<Mesh>)> = Vec::new();
    let mut tried = false;
    let mut mesh = first;
    let mut faults = Vec::new();
    for f in 0..faces {
        let area = drawn[f].max(ref_area[f]);
        let had = drawn[f];
        if !(area > NO_FACE_MM2) || had >= MESHED_SHARE * area {
            continue;
        }
        if !tried {
            tried = true;
            for k in 1..=CHORD_DOUBLINGS {
                let c = chord * f64::from(2u32.pow(k));
                tries.push((Some(c), body.display_mesh(c).ok()));
            }
            tries.push((None, reference.clone()));
            for angle in COARSER_ANGLES {
                tries.push((Some(chord), body.display_mesh_angular(chord, angle).ok()));
            }
        }
        let mut best: Option<&Mesh> = None;
        let mut cover = had;
        let mut at_chord = None;
        for (c, m) in &tries {
            let Some(m) = m else { continue };
            let a: f64 = m.triangles.iter().zip(&m.tri_face).filter(|(_, x)| **x as usize == f).map(|(t, _)| tri_area(m, t).0).sum();
            if a > cover && a <= OVERDRAWN_SHARE * area {
                best = Some(m);
                cover = a;
            }
            if cover >= MESHED_SHARE * area {
                at_chord = Some(c.unwrap_or(chord));
                break;
            }
        }
        if let Some(piece) = best {
            mesh = with_face(&mesh, f, piece);
        }
        let at = if had > 0.0 { mid[f] } else { ref_mid[f] };
        faults.push(MeshFault { face: f, at, area, drawn: cover, chord: at_chord });
    }
    (mesh, faults)
}

/// A face the mesher did not draw whole, as an answer names it: which it is and where, and how
/// it was drawn in the end (ScanRuler's `meshFaultOut` words).
pub fn mesh_fault_words(body: &str, f: &MeshFault, made_by: Option<&str>) -> String {
    let r2 = |v: f64| js_string(js_round(v * 100.0) / 100.0 + 0.0);
    let which = format!("face {} of {body}{} at ({}, {}, {})", f.face, made_by.map_or(String::new(), |m| format!(", made by {m},")), r2(f.at.x), r2(f.at.y), r2(f.at.z));
    let how = match f.chord {
        Some(c) => format!("drawn on its own at a {} mm chord", js_string(to_precision_number(c, 2))),
        None => format!("a hole in the mesh, its triangles covering {} of {} mm²", r2(f.drawn), r2(f.area)),
    };
    format!("The mesher failed on {which}: {how}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn says_a_face_the_mesher_failed_on() {
        let f = MeshFault { face: 7, at: DVec3::new(1.234, 2.0, -0.001), area: 3.5, drawn: 0.1, chord: Some(0.08) };
        assert_eq!(mesh_fault_words("Solid 1", &f, Some("Fillet 2")), "The mesher failed on face 7 of Solid 1, made by Fillet 2, at (1.23, 2, 0): drawn on its own at a 0.08 mm chord.");
        let g = MeshFault { chord: None, ..f };
        assert_eq!(mesh_fault_words("Solid 1", &g, None), "The mesher failed on face 7 of Solid 1 at (1.23, 2, 0): a hole in the mesh, its triangles covering 0.1 of 3.5 mm².");
    }

    #[test]
    fn leaves_a_mesh_with_every_face_drawn_as_it_was() {
        let block = solidcraft_kernel::box_solid(SVec3::new(0.0, 0.0, 0.0), SVec3::new(10.0, 10.0, 10.0));
        let Ok(block) = block else { return };
        let Ok(m) = block.display_mesh(0.01) else { return };
        let n = m.triangles.len();
        let (out, faults) = mesh_every_face(&block, m, 0.01);
        assert!(faults.is_empty());
        assert_eq!(out.triangles.len(), n);
        // A face taken off the mesh is put back.
        let mut bare = out.clone();
        let keep: Vec<usize> = (0..bare.triangles.len()).filter(|i| bare.tri_face[*i] != 2).collect();
        bare.triangles = keep.iter().map(|i| out.triangles[*i]).collect();
        bare.tri_face = keep.iter().map(|i| out.tri_face[*i]).collect();
        let (mended, faults) = mesh_every_face(&block, bare, 0.01);
        assert_eq!(faults.len(), 1);
        assert_eq!(faults[0].face, 2);
        assert!((faults[0].area - 100.0).abs() < 1e-6, "{}", faults[0].area);
        assert!(mended.tri_face.contains(&2));
        assert!(closes(&mended));
    }
}
