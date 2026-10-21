//! Parts (Meshmixer's Meshmix library): shapes that are dropped onto a surface.
//!
//! * A **solid part** is a closed mesh; it is placed with its +z axis along the surface
//!   normal and then combined with the object (boolean, append or new object — the caller's
//!   choice).
//! * An **open part** has one boundary loop, its rim. It is stored relative to the plane of
//!   its rim, scaled so the rim fits the unit circle, and protrudes toward +z (the side its
//!   normals face). Dropped, it can follow the target surface: every vertex keeps its height
//!   above the rim plane but measured from the surface beneath it. [`drop_open_part`] then
//!   cuts a slightly larger hole of the rim's shape into the surface (see
//!   [`crate::stamp`]) and bridges the part's rim to it.
//!
//! The built-in parts are generated here (surfaces of revolution and primitives).

use crate::stamp::{stamp_polygon, StampFrame};
use crate::{join_loops, BridgeParams, EditError, EditResult};
use glam::{DVec2, DVec3};
use mm_core::{primitives, Bvh, Mesh};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartKind {
    Solid,
    Open,
}

/// A part in its normalized frame.
#[derive(Clone)]
pub struct Part {
    pub name: String,
    pub kind: PartKind,
    pub mesh: Mesh,
    /// Open parts: the rim loop, counter-clockwise around +z.
    pub rim: Vec<u32>,
}

impl Part {
    /// Normalize a mesh with one boundary loop into an open part.
    pub fn open(name: impl Into<String>, mut mesh: Mesh) -> EditResult<Self> {
        let loops = mm_core::topology::boundary_loops(&mesh);
        if loops.len() != 1 {
            return Err(EditError::InvalidParameter(format!("an open part needs exactly one boundary loop (this has {})", loops.len())));
        }
        let rim = loops[0].vertices.clone();
        let pts: Vec<DVec3> = rim.iter().map(|&v| mesh.vertex(v)).collect();
        let c = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
        // Rim plane: least-squares normal, oriented so the rim runs counter-clockwise.
        let mut cov = [[0.0; 3]; 3];
        for p in &pts {
            mm_core::linalg::add_outer(&mut cov, *p - c, 1.0);
        }
        let (_, vecs) = mm_core::linalg::sym_eigen3(cov);
        let mut n = vecs[0];
        let (mut u, _) = mm_core::geom::orthonormal_basis(n);
        let area = |n: DVec3, u: DVec3| {
            let v = n.cross(u);
            let q: Vec<DVec2> = pts.iter().map(|p| DVec2::new((*p - c).dot(u), (*p - c).dot(v))).collect();
            (0..q.len()).map(|i| q[i].perp_dot(q[(i + 1) % q.len()])).sum::<f64>()
        };
        if area(n, u) < 0.0 {
            n = -n;
            u = -u;
        }
        let v = n.cross(u);
        let r = pts.iter().map(|p| {
            let d = *p - c;
            (d - n * d.dot(n)).length()
        });
        let r = r.fold(0.0, f64::max);
        if r <= 0.0 {
            return Err(EditError::InvalidParameter("the part's rim has no extent".into()));
        }
        mesh.transform(|p| {
            let d = (p - c) / r;
            DVec3::new(d.dot(u), d.dot(v), d.dot(n))
        });
        Ok(Self { name: name.into(), kind: PartKind::Open, mesh, rim })
    }

    /// Normalize a closed mesh into a solid part: centred, unit bounding radius.
    pub fn solid(name: impl Into<String>, mut mesh: Mesh) -> EditResult<Self> {
        if !mesh.is_closed() || mesh.triangle_count() == 0 {
            return Err(EditError::InvalidParameter("a solid part must be closed (select a whole shell)".into()));
        }
        let c = mesh.bounds().center();
        let r = mesh.vertex_ids().map(|v| mesh.vertex(v).distance(c)).fold(0.0, f64::max);
        mesh.transform(|p| (p - c) / r.max(1e-300));
        Ok(Self { name: name.into(), kind: PartKind::Solid, mesh, rim: Vec::new() })
    }

    /// The rim in the part plane (open parts).
    pub fn rim_outline(&self) -> Vec<DVec2> {
        self.rim.iter().map(|&v| self.mesh.vertex(v).truncate()).collect()
    }
}

/// Where and how big a part is dropped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// Drop point, surface normal (+z of the part) and in-plane orientation.
    pub frame: StampFrame,
    /// Rim radius (open parts) or bounding radius (solid parts).
    pub scale: f64,
    /// Solid parts: centre offset along the normal, in part radii (0 = centred on the surface).
    pub offset: f64,
    /// Open parts: heights follow the surface beneath (otherwise the part is placed rigidly).
    pub follow: bool,
}

/// Surface point and interpolated normal nearest to `p`.
fn surface_at(mesh: &Mesh, bvh: &Bvh, p: DVec3, reach: f64) -> Option<(DVec3, DVec3)> {
    let h = bvh.nearest(mesh, p, reach)?;
    let tv = mesh.triangle(h.tid);
    let n = (mesh.vertex_normal(tv[0]) * h.bary.x + mesh.vertex_normal(tv[1]) * h.bary.y + mesh.vertex_normal(tv[2]) * h.bary.z).normalize_or(mesh.tri_normal(h.tid));
    Some((h.point, n))
}

/// The part's mesh moved into place (no welding). `target` is sampled when an open part
/// follows the surface.
pub fn place_part(part: &Part, at: &Placement, target: Option<(&Mesh, &Bvh)>) -> Mesh {
    let f = &at.frame;
    let s = at.scale;
    let mut m = part.mesh.clone();
    let rigid = |p: DVec3| f.origin + (f.u * p.x + f.v * p.y + f.n * p.z) * s;
    match (part.kind, target) {
        (PartKind::Open, Some((tm, bvh))) if at.follow => {
            let rim: std::collections::HashSet<u32> = part.rim.iter().copied().collect();
            let ids: Vec<u32> = m.vertex_ids().collect();
            for v in ids {
                let p = m.vertex(v);
                let base = f.lift(p.truncate() * s);
                let h = if rim.contains(&v) { 0.0 } else { p.z * s };
                let q = surface_at(tm, bvh, base, s * 4.0).map_or(rigid(p), |(sp, sn)| sp + sn * h);
                m.set_vertex(v, q);
            }
        }
        (PartKind::Solid, _) => m.transform(|p| rigid(p) + f.n * (at.offset * s)),
        _ => m.transform(rigid),
    }
    m
}

/// What [`drop_open_part`] added.
#[derive(Clone, Debug, Default)]
pub struct DropResult {
    /// The part's triangles (in the target).
    pub part_triangles: Vec<u32>,
    /// The band joining the part's rim to the surface.
    pub band_triangles: Vec<u32>,
}

/// Weld an open part into `target` around the triangle `seed`: cut a hole of the rim's shape,
/// `gap` (fraction of the rim radius) wider than the rim, and bridge the rim to it. The part
/// gets a new face group; the band takes the group of the surface it replaced.
pub fn drop_open_part(target: &mut Mesh, seed: u32, part: &Part, at: &Placement, gap: f64) -> EditResult<DropResult> {
    check_drop(part, gap)?;
    let bvh = Bvh::build(target);
    drop_open_part_with(target, &bvh, seed, part, at, gap)
}

/// [`drop_open_part`] with a tree over all of `target` as it is now, such as the one the
/// part was placed with for its preview, so that no tree is built per drop. The part follows
/// the surface through the tree's nearest points; a tree of another layout (one refitted
/// rather than built) can differ only where two triangles are exactly as near, and then only
/// in which one's normal the part's vertex takes.
pub fn drop_open_part_with(target: &mut Mesh, bvh: &Bvh, seed: u32, part: &Part, at: &Placement, gap: f64) -> EditResult<DropResult> {
    check_drop(part, gap)?;
    if !bvh.is_current(target) || bvh.tris.len() != target.triangle_count() {
        return Err(EditError::InvalidParameter("the tree is not over the surface as it is".into()));
    }
    let placed = place_part(part, at, Some((target, bvh)));
    let group = target.tri_group(seed);
    let poly: Vec<DVec2> = part.rim_outline().iter().map(|q| *q * at.scale * (1.0 + gap)).collect();
    let cut = stamp_polygon(target, seed, &at.frame, &poly)?;
    target.remove_triangles(cut.inside.iter().copied());
    let first_part_tri = target.max_tid() as u32;
    let vmap = target.append_mesh(&placed, target.max_group_id() + 1);
    let rim: Vec<u32> = part.rim.iter().map(|&v| vmap[v as usize]).collect();
    let part_triangles: Vec<u32> = (first_part_tri..target.max_tid() as u32).filter(|&t| target.is_triangle(t)).collect();
    let band = join_loops(target, &cut.outline, &rim, &BridgeParams { segments: 0, smooth: false })?;
    for &t in &band.triangles {
        target.set_tri_group(t, group);
    }
    Ok(DropResult { part_triangles, band_triangles: band.triangles })
}

fn check_drop(part: &Part, gap: f64) -> EditResult<()> {
    if part.kind != PartKind::Open {
        return Err(EditError::InvalidParameter("only open parts are welded into the surface".into()));
    }
    if gap.is_nan() || gap <= 0.0 {
        return Err(EditError::InvalidParameter("the gap must be positive".into()));
    }
    Ok(())
}

/// Surface of revolution around +z through `profile` (r, z) points; points with r = 0
/// become poles. `closed` joins the last profile point back to the first (a torus).
/// Following the profile, the surface faces to its right in the (r, z) plane.
pub fn revolve(profile: &[(f64, f64)], slices: usize, closed: bool) -> Mesh {
    let mut pos = Vec::new();
    let mut rings: Vec<Vec<u32>> = Vec::new();
    for &(r, z) in profile {
        if r.abs() < 1e-12 {
            rings.push(vec![pos.len() as u32; slices]);
            pos.push(DVec3::new(0.0, 0.0, z));
        } else {
            let base = pos.len() as u32;
            rings.push((0..slices as u32).map(|j| base + j).collect());
            for j in 0..slices {
                let a = std::f64::consts::TAU * j as f64 / slices as f64;
                pos.push(DVec3::new(r * a.cos(), r * a.sin(), z));
            }
        }
    }
    let mut tris = Vec::new();
    let n = rings.len();
    for i in 0..if closed { n } else { n - 1 } {
        let (lo, hi) = (&rings[i], &rings[(i + 1) % n]);
        for j in 0..slices {
            let k = (j + 1) % slices;
            let (a, b, c, d) = (lo[j], lo[k], hi[k], hi[j]);
            if a != b {
                tris.push([a, b, c]);
            }
            if c != d {
                tris.push([a, c, d]);
            }
        }
    }
    Mesh::from_triangles(&pos, &tris, None).0
}

fn arc(center: (f64, f64), radius: f64, from_deg: f64, to_deg: f64, steps: usize) -> Vec<(f64, f64)> {
    (0..=steps)
        .map(|i| {
            let a = (from_deg + (to_deg - from_deg) * i as f64 / steps as f64).to_radians();
            (center.0 + radius * a.cos(), center.1 + radius * a.sin())
        })
        .collect()
}

fn line(a: (f64, f64), b: (f64, f64), steps: usize) -> Vec<(f64, f64)> {
    (0..=steps).map(|i| (a.0 + (b.0 - a.0) * i as f64 / steps as f64, a.1 + (b.1 - a.1) * i as f64 / steps as f64)).collect()
}

/// Concatenate profile pieces, dropping the duplicated joints.
fn chain(pieces: &[Vec<(f64, f64)>]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    for p in pieces {
        for &q in p {
            if out.last().is_none_or(|l| (l.0 - q.0).abs() + (l.1 - q.1).abs() > 1e-12) {
                out.push(q);
            }
        }
    }
    out
}

/// Polymix's built-in parts (our own shapes, generated).
pub fn builtin_parts() -> Vec<Part> {
    let solid = |name: &str, m: Mesh| Part::solid(name, m).expect("built-in solid part");
    let open = |name: &str, m: Mesh| Part::open(name, m).expect("built-in open part");
    let s = 40;
    vec![
        solid("Sphere", primitives::icosphere(1.0, 3)),
        solid("Cube", primitives::cube(1.4, 6)),
        solid("Cylinder", revolve(&chain(&[line((0.0, -1.0), (1.0, -1.0), 5), line((1.0, -1.0), (1.0, 1.0), 10), line((1.0, 1.0), (0.0, 1.0), 5)]), s, false)),
        solid("Cone", revolve(&chain(&[line((0.0, -1.0), (1.0, -1.0), 5), line((1.0, -1.0), (0.0, 1.0), 12)]), s, false)),
        solid("Capsule", revolve(&chain(&[arc((0.0, -0.6), 0.6, -90.0, 0.0, 8), line((0.6, -0.6), (0.6, 0.6), 6), arc((0.0, 0.6), 0.6, 0.0, 90.0, 8)]), s, false)),
        solid("Torus", revolve(&arc((0.7, 0.0), 0.3, 0.0, 360.0 * 23.0 / 24.0, 23), s, true)),
        open("Disc", revolve(&line((1.0, 0.0), (0.0, 0.0), 8), s, false)),
        open("Dome", revolve(&arc((0.0, 0.0), 1.0, 0.0, 90.0, 12), s, false)),
        open("Boss", revolve(&chain(&[line((1.0, 0.0), (1.0, 0.8), 6), line((1.0, 0.8), (0.0, 0.8), 6)]), s, false)),
        open("Spike", revolve(&line((1.0, 0.0), (0.0, 1.6), 14), s, false)),
        open("Ring", revolve(&chain(&[line((1.0, 0.0), (1.0, 0.3), 3), arc((0.85, 0.3), 0.15, 0.0, 180.0, 8), line((0.7, 0.3), (0.7, 0.0), 3), line((0.7, 0.0), (0.0, 0.0), 5)]), s, false)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_parts_are_valid() {
        for p in builtin_parts() {
            p.mesh.check_validity().unwrap();
            match p.kind {
                PartKind::Solid => {
                    assert!(p.mesh.is_closed(), "{}", p.name);
                    assert!(mm_core::measure::volume(&p.mesh) > 0.1, "{} faces outward", p.name);
                    let r = p.mesh.vertex_ids().map(|v| p.mesh.vertex(v).length()).fold(0.0, f64::max);
                    assert!(r <= 1.0 + 1e-9, "{}", p.name);
                }
                PartKind::Open => {
                    // Rim on z = 0 in the unit circle, counter-clockwise; protrudes up.
                    let q = p.rim_outline();
                    assert!(q.iter().all(|x| x.length() <= 1.0 + 1e-9), "{}", p.name);
                    let area: f64 = (0..q.len()).map(|i| q[i].perp_dot(q[(i + 1) % q.len()])).sum();
                    assert!(area > 0.0, "{}", p.name);
                    assert!(p.rim.iter().all(|&v| p.mesh.vertex(v).z.abs() < 1e-9), "{}", p.name);
                    let up: f64 = p.mesh.triangle_ids().map(|t| p.mesh.tri_normal(t).z * p.mesh.tri_area(t)).sum();
                    assert!(up > 0.0, "{}", p.name);
                }
            }
        }
    }

    #[test]
    fn open_part_normalization_is_independent_of_pose() {
        // A dome tilted and moved: normalizing brings it back to the canonical frame.
        let dome = revolve(&arc((0.0, 0.0), 1.0, 0.0, 90.0, 10), 24, false);
        let mut posed = dome.clone();
        let q = glam::DQuat::from_axis_angle(DVec3::new(1.0, 2.0, 0.5).normalize(), 1.1);
        posed.transform(|p| q * (p * 3.0) + DVec3::new(5.0, -2.0, 7.0));
        let a = Part::open("a", dome).unwrap();
        let b = Part::open("b", posed).unwrap();
        let top = |m: &Mesh| m.vertex_ids().map(|v| m.vertex(v).z).fold(f64::MIN, f64::max);
        assert!((top(&a.mesh) - 1.0).abs() < 1e-9 && (top(&b.mesh) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn dropped_dome_welds_into_a_sphere() {
        let mut target = primitives::icosphere(10.0, 4);
        let v0 = mm_core::measure::volume(&target);
        let parts = builtin_parts();
        let dome = parts.iter().find(|p| p.name == "Dome").unwrap();
        let seed = target.triangle_ids().max_by(|&a, &b| target.tri_centroid(a).z.total_cmp(&target.tri_centroid(b).z)).unwrap();
        let top = target.tri_centroid(seed);
        let at = Placement { frame: StampFrame::new(top, DVec3::Z, DVec3::Y), scale: 3.0, offset: 0.0, follow: true };
        let r = drop_open_part(&mut target, seed, dome, &at, 0.15).unwrap();
        target.check_validity().unwrap();
        assert!(target.is_closed());
        assert!(!r.part_triangles.is_empty() && !r.band_triangles.is_empty());
        // A dome of radius 3 on the sphere adds about half a ball of volume.
        let added = mm_core::measure::volume(&target) - v0;
        let half_ball = 2.0 / 3.0 * std::f64::consts::PI * 27.0;
        assert!((added - half_ball).abs() < 0.25 * half_ball, "{added} vs {half_ball}");
        // The dome follows the sphere: its top is about 3 above the surface.
        let peak = target.vertex_ids().map(|v| target.vertex(v).length()).fold(0.0, f64::max);
        assert!((peak - 13.0).abs() < 0.3, "{peak}");
    }

    #[test]
    fn dropping_with_the_objects_tree_gives_the_same_mesh() {
        let mut target = primitives::icosphere(10.0, 4);
        target.transform(|p| DVec3::new(p.x, p.y * 0.8, p.z));
        let parts = builtin_parts();
        let dome = parts.iter().find(|p| p.name == "Dome").unwrap();
        let seed = target.triangle_ids().max_by(|&a, &b| target.tri_centroid(a).z.total_cmp(&target.tri_centroid(b).z)).unwrap();
        let at = Placement { frame: StampFrame::new(target.tri_centroid(seed), DVec3::Z, DVec3::Y), scale: 3.0, offset: 0.0, follow: true };
        let mut a = target.clone();
        let ra = drop_open_part(&mut a, seed, dome, &at, 0.15).unwrap();
        let mut b = target.clone();
        let rb = drop_open_part_with(&mut b, &Bvh::build(&target), seed, dome, &at, 0.15).unwrap();
        assert_eq!(a.to_arrays(), b.to_arrays());
        assert_eq!((ra.part_triangles, ra.band_triangles), (rb.part_triangles, rb.band_triangles));
        // A tree of another shape is turned away.
        let mut c = target.clone();
        let stale = Bvh::build(&primitives::icosphere(10.0, 4));
        assert!(drop_open_part_with(&mut c, &stale, seed, dome, &at, 0.15).is_err());
        assert_eq!(c.to_arrays(), target.to_arrays());
    }

    #[test]
    fn solid_part_placement() {
        let parts = builtin_parts();
        let capsule = parts.iter().find(|p| p.name == "Capsule").unwrap();
        let at = Placement { frame: StampFrame::new(DVec3::new(1.0, 2.0, 3.0), DVec3::X, DVec3::Z), scale: 2.0, offset: 0.5, follow: false };
        let m = place_part(capsule, &at, None);
        let c = m.bounds().center();
        assert!((c - DVec3::new(2.0, 2.0, 3.0)).length() < 1e-6, "{c:?}");
        // The capsule's long axis (+z of the part) lies along the normal (+x).
        let e = m.bounds().extents();
        assert!(e.x > e.y && e.x > e.z);
    }
}
