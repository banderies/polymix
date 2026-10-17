//! Display normals with sharp edges.
//!
//! Averaging face normals per vertex makes every corner look rounded: on CAD models (long
//! thin triangles meeting at hard corners) the flat faces pick up gradients from their
//! neighbours. Here each triangle *corner* gets its own normal: the angle-weighted
//! (Thürmer & Wüthrich) average of the faces around the corner's vertex that can be reached
//! from the triangle without crossing a sharp edge, i.e. an edge whose dihedral angle
//! exceeds the crease angle. Gentle curvature stays smooth, hard edges stay crisp.

use crate::mesh::{Mesh, INVALID};
use glam::DVec3;
use rayon::prelude::*;

/// Unit normal and interior corner angles of a triangle (zero normal if degenerate).
fn face_data(m: &Mesh, t: u32) -> (DVec3, [f64; 3]) {
    let p = m.tri_positions(t);
    let angle = |j: usize| {
        let a = p[(j + 1) % 3] - p[j];
        let b = p[(j + 2) % 3] - p[j];
        a.cross(b).length().atan2(a.dot(b))
    };
    (m.tri_normal(t), [angle(0), angle(1), angle(2)])
}

/// Cosine threshold for a crease angle in degrees (`None` = never sharp).
fn crease_cos(crease_degrees: Option<f64>) -> f64 {
    crease_degrees.map(|d| d.to_radians().cos()).unwrap_or(-2.0)
}

/// Walk the fan around corner `j` of `t` in both directions, stopping at boundaries, sharp
/// edges and (with `groups`) face-group borders, and sum the angle-weighted face normals of
/// the wedge.
fn wedge_normal(m: &Mesh, t: u32, j: usize, cos_crease: f64, groups: bool, face: impl Fn(u32) -> (DVec3, [f64; 3])) -> DVec3 {
    let v = m.triangle(t)[j];
    let (n0, a0) = face(t);
    let mut sum = n0 * a0[j];
    let te = m.tri_edges(t);
    let limit = m.vertex_valence(v) + 1;
    for start in [te[j], te[(j + 2) % 3]] {
        let (mut cur, mut e, mut reference) = (t, start, n0);
        let mut closed = false;
        for _ in 0..limit {
            let n = m.edge_other_triangle(e, cur);
            if n == INVALID {
                break;
            }
            if n == t {
                closed = true;
                break;
            }
            if groups && m.tri_group(n) != m.tri_group(t) {
                break;
            }
            let (nn, na) = face(n);
            // Degenerate faces have no normal; compare against the last real one.
            if nn != DVec3::ZERO && reference != DVec3::ZERO && reference.dot(nn) < cos_crease {
                break;
            }
            let k = m.triangle(n).iter().position(|&x| x == v).unwrap();
            sum += nn * na[k];
            if nn != DVec3::ZERO {
                reference = nn;
            }
            // Continue through the other edge of `n` at `v`.
            let ne = m.tri_edges(n);
            e = if ne[k] == e { ne[(k + 2) % 3] } else { ne[k] };
            cur = n;
        }
        if closed {
            // Went all the way round; the other direction would count the faces twice.
            break;
        }
    }
    sum.try_normalize().unwrap_or_else(|| m.vertex_normal(v))
}

/// Shading normal of corner `j` (0..3) of triangle `t`. `crease_degrees`: edges whose
/// dihedral angle exceeds it are kept sharp (`None` = smooth everywhere); `groups`: face-group
/// borders are kept sharp too (Meshmixer's "Group Normals").
pub fn corner_normal(m: &Mesh, t: u32, j: usize, crease_degrees: Option<f64>, groups: bool) -> DVec3 {
    wedge_normal(m, t, j, crease_cos(crease_degrees), groups, |f| face_data(m, f))
}

/// Shading normals for a whole mesh, with face data computed once up front.
pub struct ShadingNormals<'a> {
    mesh: &'a Mesh,
    faces: Vec<(DVec3, [f64; 3])>,
    cos_crease: f64,
    groups: bool,
}

impl<'a> ShadingNormals<'a> {
    pub fn new(mesh: &'a Mesh, crease_degrees: Option<f64>, groups: bool) -> Self {
        let faces = (0..mesh.max_tid() as u32)
            .into_par_iter()
            .map(|t| if mesh.is_triangle(t) { face_data(mesh, t) } else { (DVec3::ZERO, [0.0; 3]) })
            .collect();
        Self { mesh, faces, cos_crease: crease_cos(crease_degrees), groups }
    }

    /// Same result as [`corner_normal`].
    pub fn corner(&self, t: u32, j: usize) -> DVec3 {
        wedge_normal(self.mesh, t, j, self.cos_crease, self.groups, |f| self.faces[f as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn cube_faces_stay_flat() {
        let m = primitives::cube(2.0, 1);
        let sn = ShadingNormals::new(&m, Some(30.0), false);
        for t in m.triangle_ids() {
            let n = m.tri_normal(t);
            for j in 0..3 {
                assert!(sn.corner(t, j).dot(n) > 1.0 - 1e-9, "corner normal must equal the face normal");
                assert!((corner_normal(&m, t, j, Some(30.0), false) - sn.corner(t, j)).length() < 1e-12);
            }
        }
        // Without a crease the corner normals point diagonally out of the cube.
        let t = m.triangle_ids().next().unwrap();
        let n = corner_normal(&m, t, 0, None, false);
        assert!(n.dot(m.tri_normal(t)) < 0.9);
        let p = m.vertex(m.triangle(t)[0]);
        assert!(n.dot(p.normalize()) > 0.99);
    }

    #[test]
    fn smooth_sphere_matches_vertex_normals() {
        let m = primitives::icosphere(1.0, 3);
        let sn = ShadingNormals::new(&m, Some(40.0), false);
        for t in m.triangle_ids() {
            for (j, v) in m.triangle(t).into_iter().enumerate() {
                assert!(sn.corner(t, j).dot(m.vertex(v).normalize()) > 0.999);
            }
        }
    }

    #[test]
    fn cylinder_side_is_smooth_but_caps_are_sharp() {
        let m = primitives::cylinder(1.0, 2.0, 32, 2, true);
        let sn = ShadingNormals::new(&m, Some(40.0), false);
        for t in m.triangle_ids() {
            let fnrm = m.tri_normal(t);
            for (j, v) in m.triangle(t).into_iter().enumerate() {
                let n = sn.corner(t, j);
                if fnrm.z.abs() > 0.99 {
                    assert!(n.dot(fnrm) > 1.0 - 1e-9, "cap corners keep the cap normal");
                } else {
                    // Side corners are radial (no z component), even at the rim.
                    let p = m.vertex(v);
                    let radial = DVec3::new(p.x, p.y, 0.0).normalize();
                    assert!(n.dot(radial) > 0.999);
                }
            }
        }
    }

    #[test]
    fn group_borders_split_normals() {
        // A sphere split into two groups: smooth within each, sharp along the border.
        let mut m = primitives::icosphere(1.0, 3);
        for t in m.triangle_ids().collect::<Vec<_>>() {
            let g = (m.tri_centroid(t).z > 0.0) as i32;
            m.set_tri_group(t, g);
        }
        let sn = ShadingNormals::new(&m, None, true);
        let smooth = ShadingNormals::new(&m, None, false);
        let mut split = 0;
        for t in m.triangle_ids() {
            for (j, &v) in m.triangle(t).iter().enumerate() {
                let border = m.vertex_triangles(v).iter().any(|&o| m.tri_group(o) != m.tri_group(t));
                let differs = (sn.corner(t, j) - smooth.corner(t, j)).length() > 1e-9;
                assert_eq!(border, differs, "only corners on the group border change");
                split += border as usize;
            }
        }
        assert!(split > 0);
    }
}
