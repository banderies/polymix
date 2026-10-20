//! SetUV: texture coordinates by projection, without unwrapping (Polymix's reading; Meshmixer
//! does not document its SetUV).
//!
//! The projection has a frame: a centre `c`, an axis `w` (X, Y, Z or the view direction) and
//! two directions `u`, `v` across it (turned about `w` by `rotation`), with `u × v = w`. For a
//! point `p`, `d = p − c`:
//!
//! * **Planar**: `U = (d·u) / size`, `V = (d·v) / (size · aspect)` — a slide projector along
//!   `w` (the back gets the mirror image).
//! * **Cylindrical**: `U` is the angle about `w` (one turn across the sheet; the seam is on
//!   the `−u` side), `V = (d·w) / (size · aspect)`.
//! * **Spherical**: `U` as cylindrical, `V` is the latitude (south pole 0, north pole 1).
//! * **Box**: each triangle is projected flat along the frame axis closest to its normal, seen
//!   from outside (so no side is mirrored).
//!
//! `size` is the model length across the sheet's width and `aspect` the sheet's height over its
//! width, so a projection keeps true proportions on a `width × height` sheet, as Unwrap does.
//! Then `U ← U · scale.u + 0.5 + offset.u` and `V ← V · scale.v + 0.5 + offset.v` (the centre
//! lands in the middle of the sheet). UVs are per triangle corner: a triangle across the seam of
//! a cylindrical or spherical projection gets corners on the same side (U runs past 1 or below
//! 0, which repeats the texture) instead of spanning the whole sheet, and a corner on the axis
//! (a pole) takes its triangle's angle.

use crate::{EditError, EditResult};
use glam::{DVec2, DVec3};
use mm_core::Mesh;
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum Projection {
    Planar,
    Cylindrical,
    Spherical,
    Box,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ProjectionAxis {
    X,
    Y,
    Z,
    /// Along the view (Planar: as seen on screen).
    View,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SetUvParams {
    pub projection: Projection,
    /// Axis the projection looks along (Planar, Box) or turns about (Cylindrical, Spherical).
    pub axis: ProjectionAxis,
    /// Centre of the projection in mm (null: the middle of the projected triangles).
    pub center: Option<[f64; 3]>,
    /// Model length (mm) across the sheet's width (null: fit the triangles on the sheet).
    /// Cylindrical uses it for V only; Spherical not at all.
    pub size: Option<f64>,
    /// Extra U and V scale (1 = none; −1 mirrors).
    pub scale: [f64; 2],
    /// Added to U and V.
    pub offset: [f64; 2],
    /// Turn of the projection about its axis, in degrees.
    pub rotation: f64,
}

impl Default for SetUvParams {
    fn default() -> Self {
        Self { projection: Projection::Planar, axis: ProjectionAxis::Z, center: None, size: None, scale: [1.0, 1.0], offset: [0.0, 0.0], rotation: 0.0 }
    }
}

/// The frame of a projection: `u`, `v` across the axis `w` (`u × v = w`), at `center`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionFrame {
    pub center: DVec3,
    pub u: DVec3,
    pub v: DVec3,
    pub w: DVec3,
}

/// The frame of `p` at `center`; `view` is the camera's (right, up) for [`ProjectionAxis::View`].
pub fn projection_frame(p: &SetUvParams, center: DVec3, view: Option<(DVec3, DVec3)>) -> ProjectionFrame {
    let (u, v) = match p.axis {
        // Seen from the + side of the axis: right and up on screen.
        ProjectionAxis::X => (-DVec3::Z, DVec3::Y),
        ProjectionAxis::Y => (DVec3::X, -DVec3::Z),
        ProjectionAxis::Z => (DVec3::X, DVec3::Y),
        ProjectionAxis::View => {
            let (r, up) = view.unwrap_or((DVec3::X, DVec3::Y));
            let r = r.normalize_or(DVec3::X);
            let up = (up - r * up.dot(r)).normalize_or(r.any_orthonormal_vector());
            (r, up)
        }
    };
    let w = u.cross(v);
    let (s, c) = p.rotation.to_radians().sin_cos();
    ProjectionFrame { center, u: u * c + v * s, v: v * c - u * s, w }
}

/// The middle (bounding-box centre) of the triangles' corners.
pub fn projection_center(mesh: &Mesh, tris: &[u32]) -> DVec3 {
    let b = mm_core::Aabb::from_points(tris.iter().flat_map(|&t| mesh.tri_positions(t)));
    if b.is_empty() {
        DVec3::ZERO
    } else {
        b.center()
    }
}

/// Distinct vertices of `tris`.
fn vertices(mesh: &Mesh, tris: &[u32]) -> Vec<u32> {
    let mut vs: Vec<u32> = tris.iter().flat_map(|&t| mesh.triangle(t)).collect();
    vs.sort_unstable();
    vs.dedup();
    vs
}

/// The `size` that fits the projected triangles on the sheet (see the module docs).
pub fn fit_size(mesh: &Mesh, tris: &[u32], p: &SetUvParams, f: &ProjectionFrame, aspect: f64) -> f64 {
    let aspect = aspect.max(1e-9);
    let ds: Vec<DVec3> = vertices(mesh, tris).into_iter().map(|v| mesh.vertex(v) - f.center).collect();
    let max = |g: &dyn Fn(DVec3) -> f64| ds.iter().map(|&d| g(d)).fold(0.0f64, f64::max);
    let size = match p.projection {
        Projection::Planar => max(&|d| 2.0 * d.dot(f.u).abs()).max(max(&|d| 2.0 * d.dot(f.v).abs() / aspect)),
        Projection::Cylindrical => {
            // True proportions (one turn = the circumference) if the height fits.
            let r = ds.iter().map(|d| (*d - f.w * d.dot(f.w)).length()).sum::<f64>() / ds.len().max(1) as f64;
            (TAU * r).max(max(&|d| 2.0 * d.dot(f.w).abs() / aspect))
        }
        Projection::Spherical => 1.0,
        Projection::Box => 2.0 * max(&|d| d.dot(f.u).abs().max(d.dot(f.v).abs()).max(d.dot(f.w).abs())) * (1.0 / aspect).max(1.0),
    };
    if size > 0.0 && size.is_finite() {
        size
    } else {
        1.0
    }
}

/// Angle of `d` about the frame axis, or `None` on the axis.
fn angle(d: DVec3, f: &ProjectionFrame, tiny: f64) -> Option<f64> {
    let (x, y) = (d.dot(f.u), d.dot(f.v));
    (x.hypot(y) > tiny).then(|| y.atan2(x))
}

/// UVs of `tris` (in that order) by the projection `p` (see the module docs). `view` is the
/// camera's (right, up) for [`ProjectionAxis::View`]; `aspect` the sheet's height over width.
pub fn project_uvs(mesh: &Mesh, tris: &[u32], p: &SetUvParams, view: Option<(DVec3, DVec3)>, aspect: f64) -> EditResult<Vec<[DVec2; 3]>> {
    if tris.is_empty() {
        return Err(EditError::EmptySelection);
    }
    let positive = |x: f64| x > 0.0 && x.is_finite();
    if !positive(aspect) || p.size.is_some_and(|s| !positive(s)) || p.scale.iter().chain(&p.offset).any(|x| !x.is_finite()) || !p.rotation.is_finite() {
        return Err(EditError::InvalidParameter("SetUV needs a positive size and sheet, and finite scale, offset and rotation".into()));
    }
    let center = p.center.map(DVec3::from).unwrap_or_else(|| projection_center(mesh, tris));
    let f = projection_frame(p, center, view);
    let size = p.size.unwrap_or_else(|| fit_size(mesh, tris, p, &f, aspect));
    let tiny = 1e-12 * size.max(mesh.bounds().diagonal());
    let finish = |q: DVec2| DVec2::new(q.x * p.scale[0] + 0.5 + p.offset[0], q.y * p.scale[1] + 0.5 + p.offset[1]);
    let flat = |d: DVec3, right: DVec3, up: DVec3| finish(DVec2::new(d.dot(right) / size, d.dot(up) / (size * aspect)));
    let out = tris
        .iter()
        .map(|&t| {
            let ds = mesh.tri_positions(t).map(|q| q - f.center);
            match p.projection {
                Projection::Planar => ds.map(|d| flat(d, f.u, f.v)),
                Projection::Box => {
                    let n = mesh.tri_normal(t);
                    let n = if n.length_squared() > 0.0 { n } else { f.w };
                    let (a, b, c) = (n.dot(f.u), n.dot(f.v), n.dot(f.w));
                    // (right, up) seen from outside each side: right × up = the side's normal.
                    let (right, up) = if c.abs() >= a.abs() && c.abs() >= b.abs() {
                        if c >= 0.0 { (f.u, f.v) } else { (-f.u, f.v) }
                    } else if a.abs() >= b.abs() {
                        if a >= 0.0 { (-f.w, f.v) } else { (f.w, f.v) }
                    } else if b >= 0.0 {
                        (f.u, -f.w)
                    } else {
                        (f.u, f.w)
                    };
                    ds.map(|d| flat(d, right, up))
                }
                Projection::Cylindrical | Projection::Spherical => {
                    // The triangle's own angle; corners join it on the same side of the seam.
                    let mid = (ds[0] + ds[1] + ds[2]) / 3.0;
                    let own = angle(mid, &f, tiny).or_else(|| ds.iter().find_map(|&d| angle(d, &f, tiny))).unwrap_or(0.0);
                    ds.map(|d| {
                        let a = angle(d, &f, tiny).map_or(own, |a| a + TAU * ((own - a) / TAU).round());
                        let v = if p.projection == Projection::Cylindrical {
                            d.dot(f.w) / (size * aspect)
                        } else {
                            let l = d.length();
                            if l > tiny {
                                (d.dot(f.w) / l).clamp(-1.0, 1.0).asin() / PI
                            } else {
                                0.0
                            }
                        };
                        finish(DVec2::new(a / TAU, v))
                    })
                }
            }
        })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    /// A single triangle with the given corners.
    fn tri(ps: [DVec3; 3]) -> Mesh {
        Mesh::from_triangles(&ps, &[[0, 1, 2]], None).0
    }

    fn uv1(m: &Mesh, p: &SetUvParams, aspect: f64) -> [DVec2; 3] {
        project_uvs(m, &[0], p, None, aspect).unwrap()[0]
    }

    fn near(a: DVec2, b: [f64; 2]) -> bool {
        (a - DVec2::from(b)).length() < 1e-9
    }

    #[test]
    fn planar_maps_known_points() {
        let m = tri([DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 3.0), DVec3::new(-1.0, -1.0, 5.0)]);
        let p = SetUvParams { center: Some([0.0; 3]), size: Some(2.0), ..Default::default() };
        let q = uv1(&m, &p, 1.0);
        assert!(near(q[0], [1.0, 0.5]) && near(q[1], [0.5, 1.0]) && near(q[2], [0.0, 0.0]), "{q:?}");
        // On a sheet twice as tall as wide, V covers twice the length.
        let q = uv1(&m, &p, 2.0);
        assert!(near(q[1], [0.5, 0.75]), "{q:?}");
        // Turned 90° about the axis, scaled and offset.
        let q = uv1(&m, &SetUvParams { rotation: 90.0, scale: [2.0, 1.0], offset: [0.25, -0.5], ..p }, 1.0);
        assert!(near(q[0], [0.75, -0.5]), "{q:?}");
        // Along X (seen from +X: right is −Z, up is Y).
        let q = uv1(&m, &SetUvParams { axis: ProjectionAxis::X, ..p }, 1.0);
        assert!(near(q[1], [0.5 - 1.5, 1.0]), "{q:?}");
        // From the view: right and up on screen.
        let r = project_uvs(&m, &[0], &SetUvParams { axis: ProjectionAxis::View, ..p }, Some((DVec3::Y, DVec3::Z)), 1.0).unwrap()[0];
        assert!(near(r[1], [1.0, 2.0]), "{r:?}");
        // Automatic: the whole triangle fits the sheet, centred.
        let q = uv1(&m, &SetUvParams::default(), 1.0);
        let lo = q.iter().fold(DVec2::splat(9.0), |a, b| a.min(*b));
        let hi = q.iter().fold(DVec2::splat(-9.0), |a, b| a.max(*b));
        assert!(lo.min_element() >= -1e-12 && hi.max_element() <= 1.0 + 1e-12 && ((lo + hi) * 0.5 - DVec2::splat(0.5)).length() < 1e-12, "{lo} {hi}");
    }

    #[test]
    fn cylindrical_and_spherical_map_known_points() {
        let m = tri([DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 2.0), DVec3::new(0.0, -1.0, -2.0)]);
        let p = SetUvParams { projection: Projection::Cylindrical, center: Some([0.0; 3]), size: Some(8.0), ..Default::default() };
        let q = uv1(&m, &p, 1.0);
        assert!(near(q[0], [0.5, 0.5]) && near(q[1], [0.75, 0.75]) && near(q[2], [0.25, 0.25]), "{q:?}");
        let s = SetUvParams { projection: Projection::Spherical, ..p };
        let m = tri([DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 1.0), DVec3::new(0.0, 0.0, -3.0)]);
        let q = uv1(&m, &s, 1.0);
        // Equator at V ½, 45° north at ¾; the south pole takes the triangle's angle.
        assert!(near(q[0], [0.5, 0.5]) && near(q[1], [0.75, 0.75]), "{q:?}");
        assert!((q[2].y - 0.0).abs() < 1e-12, "{q:?}");
    }

    #[test]
    fn box_projects_each_side_from_outside() {
        let m = primitives::cube(2.0, 1);
        let tris: Vec<u32> = m.triangle_ids().collect();
        let p = SetUvParams { projection: Projection::Box, center: Some([0.0; 3]), size: Some(2.0), ..Default::default() };
        let uv = project_uvs(&m, &tris, &p, None, 1.0).unwrap();
        for (k, &t) in tris.iter().enumerate() {
            // Every side fills the sheet, none mirrored (counter-clockwise like the mesh).
            let q = uv[k];
            assert!(q.iter().all(|c| c.min_element() >= -1e-12 && c.max_element() <= 1.0 + 1e-12), "{q:?}");
            assert!((q[1] - q[0]).perp_dot(q[2] - q[0]) > 0.0, "side of normal {} is mirrored", m.tri_normal(t));
        }
        // The +Z side is seen as the plain planar projection.
        let top = tris.iter().position(|&t| m.tri_normal(t).z > 0.9).unwrap();
        let planar = project_uvs(&m, &[tris[top]], &SetUvParams { projection: Projection::Planar, ..p }, None, 1.0).unwrap();
        assert_eq!(uv[top], planar[0]);
    }

    #[test]
    fn cylindrical_seam_does_not_span_the_sheet() {
        // A tube around Z with 24 sides: no triangle may span more than a side in U, and the
        // triangles at the seam (−X) run just past the sheet's edge.
        let m = primitives::cylinder(5.0, 30.0, 24, 4, false);
        let tris: Vec<u32> = m.triangle_ids().collect();
        let uv = project_uvs(&m, &tris, &SetUvParams { projection: Projection::Cylindrical, ..Default::default() }, None, 1.0).unwrap();
        let mut across = 0;
        for q in &uv {
            let (lo, hi) = (q.iter().map(|c| c.x).fold(9.0, f64::min), q.iter().map(|c| c.x).fold(-9.0, f64::max));
            assert!(hi - lo <= 1.0 / 24.0 + 1e-9, "{q:?}");
            assert!(lo >= -1.0 / 24.0 - 1e-9 && hi <= 1.0 + 1.0 / 24.0 + 1e-9, "{q:?}");
            if lo < 0.0 - 1e-9 || hi > 1.0 + 1e-9 {
                across += 1;
            }
        }
        // The seam runs through vertices at −X: the triangles there touch it from either side.
        assert!(across == 0 || across <= 8, "{across}");
        // Unrolled, the tube keeps its proportions: one turn (31.4 mm) is the sheet's width.
        let v_span = uv.iter().flatten().map(|c| c.y).fold(-9.0, f64::max) - uv.iter().flatten().map(|c| c.y).fold(9.0, f64::min);
        assert!((v_span - 30.0 / (std::f64::consts::TAU * 5.0)).abs() < 1e-6, "{v_span}");
        // Turned so the seam cuts through the middle of a side, the side's triangles get
        // corners on both sides of U = 0 / 1 and still stay narrow.
        let p = SetUvParams { projection: Projection::Cylindrical, rotation: 7.5, ..Default::default() };
        let uv = project_uvs(&m, &tris, &p, None, 1.0).unwrap();
        let crossing = uv.iter().filter(|q| q.iter().any(|c| c.x > 1.0) || q.iter().any(|c| c.x < 0.0)).count();
        assert!(crossing >= 8, "{crossing}");
        assert!(uv.iter().all(|q| q.iter().map(|c| c.x).fold(-9.0, f64::max) - q.iter().map(|c| c.x).fold(9.0, f64::min) <= 1.0 / 24.0 + 1e-9));
    }
}
