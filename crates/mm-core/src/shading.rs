//! Display normals with sharp edges.
//!
//! Averaging face normals per vertex makes every corner look rounded: on CAD models (long
//! thin triangles meeting at hard corners) the flat faces pick up gradients from their
//! neighbours. Here each triangle *corner* gets its own normal: the angle-weighted
//! (Thürmer & Wüthrich) average of the faces around the corner's vertex that can be reached
//! from the triangle without crossing a sharp edge, i.e. an edge whose dihedral angle
//! exceeds the crease angle. Gentle curvature stays smooth, hard edges stay crisp.
//!
//! Most vertices sit in one closed fan with no sharp edge: there every corner reaches all
//! the faces around the vertex, so they share one normal, summed once per vertex. Only the
//! other corners (borders, sharp edges, group borders, degenerate faces, bowties) walk their
//! fan one by one.

use crate::hash::{fast_map_with_capacity, FastMap};
use crate::mesh::{Mesh, INVALID};
use glam::DVec3;
use rayon::prelude::*;

/// A face's unit normal (zero if degenerate) and interior corner angles.
type Face = (DVec3, [f64; 3]);

/// Shorter inputs are handled on the calling thread: waking the thread pool costs more than
/// it saves on them (a brush stroke's few thousand triangles).
const PARALLEL_MIN: usize = 1 << 14;

/// `f` over `items`, in order; in parallel for long inputs.
fn map_ordered<T: Sync, U: Send>(items: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
    if items.len() < PARALLEL_MIN {
        items.iter().map(f).collect()
    } else {
        items.par_iter().map(f).collect()
    }
}

/// The ids of `ids` (all below `bound`) without repeats, in the order they first come.
fn unique(ids: impl IntoIterator<Item = u32>, bound: usize) -> Vec<u32> {
    let mut seen = vec![0u64; bound.div_ceil(64)];
    ids.into_iter()
        .filter(|&i| {
            let (word, bit) = ((i / 64) as usize, 1u64 << (i % 64));
            let new = seen[word] & bit == 0;
            seen[word] |= bit;
            new
        })
        .collect()
}

/// Unit normal and interior corner angles of a triangle (zero normal if degenerate).
fn face_data(m: &Mesh, t: u32) -> Face {
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
fn wedge_normal(m: &Mesh, t: u32, j: usize, cos_crease: f64, groups: bool, face: impl Fn(u32) -> Face) -> DVec3 {
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

/// The normal every corner at `v` shares, when its triangles form one closed fan with no
/// sharp edge, no face-group border (with `groups`) and no degenerate face: then the walk
/// from any corner goes all the way round and sums the same faces, only starting elsewhere.
/// The sum here starts at the first triangle of `v`'s first edge, so it is that corner's
/// walk exactly; the other corners' walks add the same terms in another order and differ
/// from it by rounding alone (in practice only components that should be zero come out as
/// other specks of 1e-18; at most one unit in the last place of the f32 normals the GPU
/// gets). `None` where the corners must be
/// walked one by one (open or several fans, a sharp or group edge, a degenerate face, or
/// faces that nearly cancel so that the order of the sum could matter).
fn smooth_vertex_normal(m: &Mesh, v: u32, cos_crease: f64, groups: bool, face: impl Fn(u32) -> Face) -> Option<DVec3> {
    let edges = m.vertex_edges(v);
    let valence = edges.len();
    if valence < 3 {
        return None;
    }
    let t = match m.edge_triangles(edges[0]) {
        [INVALID, INVALID] => return None,
        [INVALID, t] | [t, _] => t,
    };
    let j = m.triangle(t).iter().position(|&x| x == v)?;
    let (n0, a0) = face(t);
    if n0 == DVec3::ZERO {
        return None;
    }
    let group = m.tri_group(t);
    let mut sum = n0 * a0[j];
    let mut weight = a0[j];
    let (mut cur, mut e, mut previous) = (t, m.tri_edges(t)[j], n0);
    // The same steps as `wedge_normal`'s first direction; a single closed fan returns to `t`
    // after exactly one face per edge at `v`.
    for step in 1..=valence {
        let n = m.edge_other_triangle(e, cur);
        if n == INVALID || (groups && m.tri_group(n) != group) {
            return None;
        }
        let (nn, na) = face(n);
        // Every pair of neighbouring faces is checked, the closing one too, so that no
        // corner's walk would stop early.
        if nn == DVec3::ZERO || previous.dot(nn) < cos_crease {
            return None;
        }
        if n == t {
            let normal = sum.try_normalize()?;
            return (step == valence && sum.length() > 1e-6 * weight).then_some(normal);
        }
        let k = m.triangle(n).iter().position(|&x| x == v)?;
        sum += nn * na[k];
        weight += na[k];
        previous = nn;
        let ne = m.tri_edges(n);
        e = if ne[k] == e { ne[(k + 2) % 3] } else { ne[k] };
        cur = n;
    }
    None
}

/// Shading normal of corner `j` of `t` with the face data from `face`.
fn corner_with(m: &Mesh, t: u32, j: usize, cos_crease: f64, groups: bool, smooth: Option<DVec3>, face: impl Fn(u32) -> Face + Copy) -> DVec3 {
    smooth.unwrap_or_else(|| wedge_normal(m, t, j, cos_crease, groups, face))
}

/// Shading normal of corner `j` (0..3) of triangle `t`. `crease_degrees`: edges whose
/// dihedral angle exceeds it are kept sharp (`None` = smooth everywhere); `groups`: face-group
/// borders are kept sharp too (Meshmixer's "Group Normals").
pub fn corner_normal(m: &Mesh, t: u32, j: usize, crease_degrees: Option<f64>, groups: bool) -> DVec3 {
    let cos_crease = crease_cos(crease_degrees);
    let face = |f| face_data(m, f);
    let smooth = smooth_vertex_normal(m, m.triangle(t)[j], cos_crease, groups, face);
    corner_with(m, t, j, cos_crease, groups, smooth, face)
}

/// Face data: for every triangle of the mesh, or for those around some corners.
enum Faces {
    /// Indexed by triangle id.
    All(Vec<Face>),
    /// By triangle id; others are computed when asked for.
    Near(FastMap<u32, Face>),
}

/// Per vertex, the normal its corners share ([`smooth_vertex_normal`]; zero: walk them).
enum Smooth {
    /// Indexed by vertex id.
    All(Vec<DVec3>),
    /// By vertex id; others are computed when asked for.
    Near(FastMap<u32, DVec3>),
}

/// Shading normals for a whole mesh, or for some of its triangles, with the face data and
/// the shared vertex normals computed once up front.
pub struct ShadingNormals<'a> {
    mesh: &'a Mesh,
    faces: Faces,
    smooth: Smooth,
    cos_crease: f64,
    groups: bool,
}

impl<'a> ShadingNormals<'a> {
    pub fn new(mesh: &'a Mesh, crease_degrees: Option<f64>, groups: bool) -> Self {
        let cos_crease = crease_cos(crease_degrees);
        let faces: Vec<Face> = (0..mesh.max_tid() as u32)
            .into_par_iter()
            .map(|t| if mesh.is_triangle(t) { face_data(mesh, t) } else { (DVec3::ZERO, [0.0; 3]) })
            .collect();
        let smooth = (0..mesh.max_vid() as u32)
            .into_par_iter()
            .map(|v| {
                let shared = mesh.is_vertex(v).then(|| smooth_vertex_normal(mesh, v, cos_crease, groups, |f| faces[f as usize]));
                shared.flatten().unwrap_or(DVec3::ZERO)
            })
            .collect();
        Self { mesh, faces: Faces::All(faces), smooth: Smooth::All(smooth), cos_crease, groups }
    }

    /// Normals for the corners of `tris` only (any order, repeats allowed): the face data of
    /// the triangles their walks can reach, and their vertices' shared normals. Corners of
    /// other triangles still come out right, only slower. The same results as [`Self::new`].
    pub fn for_triangles(mesh: &'a Mesh, tris: &[u32], crease_degrees: Option<f64>, groups: bool) -> Self {
        let cos_crease = crease_cos(crease_degrees);
        let verts = unique(tris.iter().flat_map(|&t| mesh.triangle(t)), mesh.max_vid());
        // Every walk stays among the triangles of its own vertex.
        let near = unique(verts.iter().flat_map(|&v| mesh.vertex_triangles(v)), mesh.max_tid());
        let data = map_ordered(&near, |&t| face_data(mesh, t));
        let mut faces = fast_map_with_capacity(near.len());
        faces.extend(near.into_iter().zip(data));
        let face = |f: u32| faces.get(&f).copied().unwrap_or_else(|| face_data(mesh, f));
        let shared = map_ordered(&verts, |&v| smooth_vertex_normal(mesh, v, cos_crease, groups, face).unwrap_or(DVec3::ZERO));
        let mut smooth = fast_map_with_capacity(verts.len());
        smooth.extend(verts.into_iter().zip(shared));
        Self { mesh, faces: Faces::Near(faces), smooth: Smooth::Near(smooth), cos_crease, groups }
    }

    fn face(&self, t: u32) -> Face {
        match &self.faces {
            Faces::All(f) => f[t as usize],
            Faces::Near(f) => f.get(&t).copied().unwrap_or_else(|| face_data(self.mesh, t)),
        }
    }

    /// Same result as [`corner_normal`].
    pub fn corner(&self, t: u32, j: usize) -> DVec3 {
        let v = self.mesh.triangle(t)[j];
        let shared = match &self.smooth {
            Smooth::All(s) => s[v as usize],
            Smooth::Near(s) => match s.get(&v) {
                Some(&n) => n,
                None => smooth_vertex_normal(self.mesh, v, self.cos_crease, self.groups, |f| self.face(f)).unwrap_or(DVec3::ZERO),
            },
        };
        let smooth = (shared != DVec3::ZERO).then_some(shared);
        corner_with(self.mesh, t, j, self.cos_crease, self.groups, smooth, |f| self.face(f))
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

    /// Meshes that exercise every kind of corner: smooth closed fans, sharp edges, group
    /// borders, open borders, degenerate faces, a bowtie vertex and inconsistent orientation.
    fn assorted() -> Vec<(&'static str, Mesh)> {
        let mut noisy = primitives::icosphere(1.0, 3);
        for v in noisy.vertex_ids().collect::<Vec<_>>() {
            let p = noisy.vertex(v);
            let h = ((v as f64 * 12.9898).sin() * 43758.5453).fract();
            noisy.set_vertex(v, p * (1.0 + 0.15 * h));
        }
        let mut grouped = primitives::cylinder(1.0, 2.0, 24, 6, true);
        for t in grouped.triangle_ids().collect::<Vec<_>>() {
            let g = (grouped.tri_centroid(t).x > 0.3) as i32;
            grouped.set_tri_group(t, g);
        }
        // A degenerate (zero-area) triangle inside a strip, and a flipped one.
        let p = [DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.1), DVec3::new(1.0, 1.0, 0.0), DVec3::new(2.0, 1.0, 0.2), DVec3::new(1.0, -1.0, 0.0)];
        let (degenerate, _) = Mesh::from_triangles(&p, &[[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [0, 2, 1], [0, 6, 2]], None);
        let (flipped, _) = Mesh::from_triangles(&p, &[[0, 1, 4], [0, 3, 4], [1, 2, 5], [1, 5, 4], [0, 6, 1], [1, 6, 2]], None);
        // Two closed pyramids touching at their apex: a bowtie vertex with two closed fans.
        let base = |z: f64| [DVec3::new(1.0, 0.0, z), DVec3::new(0.0, 1.0, z), DVec3::new(-1.0, 0.0, z), DVec3::new(0.0, -1.0, z)];
        let mut bp = vec![DVec3::ZERO];
        bp.extend(base(1.0));
        bp.extend(base(-1.0));
        let mut bt = Vec::new();
        for i in 0..4u32 {
            let (a, b) = (1 + i, 1 + (i + 1) % 4);
            bt.push([0, a, b]);
            bt.push([0, b + 4, a + 4]);
        }
        bt.extend([[1, 3, 2], [1, 4, 3], [5, 6, 7], [5, 7, 8]]);
        let (bowtie, _) = Mesh::from_triangles(&bp, &bt, None);
        vec![
            ("cube", primitives::cube(2.0, 3)),
            ("noisy sphere", noisy),
            ("capped cylinder in two groups", grouped),
            ("open grid", primitives::plane_grid(2.0, 1.0, 6, 4)),
            ("degenerate face", degenerate),
            ("flipped face", flipped),
            ("bowtie", bowtie),
        ]
    }

    #[test]
    fn shared_vertex_normals_match_the_walk() {
        for (name, m) in assorted() {
            for (crease, groups) in [(None, false), (Some(40.0), false), (Some(40.0), true), (Some(5.0), true)] {
                let cos = crease_cos(crease);
                let face = |f| face_data(&m, f);
                let (mut shared, mut walked) = (0, 0);
                for t in m.triangle_ids() {
                    for (j, v) in m.triangle(t).into_iter().enumerate() {
                        let walk = wedge_normal(&m, t, j, cos, groups, face);
                        let n = corner_normal(&m, t, j, crease, groups);
                        match smooth_vertex_normal(&m, v, cos, groups, face) {
                            Some(s) => {
                                // The same faces summed from another start: rounding only.
                                assert_eq!(n, s);
                                assert!((n - walk).length() < 1e-12, "{name} {crease:?} {groups}: corner ({t}, {j})");
                                shared += 1;
                            }
                            None => {
                                assert_eq!(n, walk, "{name}: corners left to the walk are unchanged");
                                walked += 1;
                            }
                        }
                    }
                }
                if name == "noisy sphere" && crease.is_none() {
                    assert_eq!(walked, 0, "a smooth closed sphere shares every vertex normal");
                }
                if ["open grid", "degenerate face", "flipped face", "bowtie"].contains(&name) {
                    assert!(walked > 0, "{name}: some corners must be walked");
                }
                assert!(shared + walked == 3 * m.triangle_count());
            }
        }
    }

    #[test]
    fn bowtie_fans_keep_their_own_normals() {
        let (_, m) = assorted().into_iter().find(|(n, _)| *n == "bowtie").unwrap();
        assert!(m.is_bowtie_vertex(0));
        let sn = ShadingNormals::new(&m, None, false);
        for t in m.triangle_ids() {
            if let Some(j) = m.triangle(t).iter().position(|&v| v == 0) {
                // Both pyramids face inward: the apex corners of the upper one point up into
                // it, those of the lower one down.
                let up = m.triangle(t).iter().any(|&v| (1..=4).contains(&v));
                let n = sn.corner(t, j);
                assert!(if up { n.z > 0.99 } else { n.z < -0.99 }, "corner ({t}, {j}): {n}");
            }
        }
    }

    #[test]
    fn partial_tables_give_the_same_normals() {
        for (name, m) in assorted() {
            for (crease, groups) in [(None, false), (Some(40.0), true)] {
                let all = ShadingNormals::new(&m, crease, groups);
                let tris: Vec<u32> = m.triangle_ids().filter(|t| t % 3 != 1).collect();
                let near = ShadingNormals::for_triangles(&m, &tris, crease, groups);
                // Including corners of triangles outside the list.
                for t in m.triangle_ids() {
                    for j in 0..3 {
                        let n = all.corner(t, j);
                        assert_eq!(n, near.corner(t, j), "{name}: ({t}, {j})");
                        assert_eq!(n, corner_normal(&m, t, j, crease, groups), "{name}: ({t}, {j})");
                    }
                }
            }
        }
    }
}
