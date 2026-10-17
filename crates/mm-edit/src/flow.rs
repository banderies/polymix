//! Flow extensions: extrude a (cut) boundary loop straight out of the mesh, optionally
//! morphing the cross-section into a circle of equal area, and cap the end. This is the
//! usual CFD pre-processing step that gives inlets/outlets a straight, developed section.

use crate::cap::cap_oriented_loops;
use crate::cut::CutFill;
use crate::util::*;
use crate::{EditError, EditResult};
use mm_core::{DVec2, DVec3, Mesh};
use std::f64::consts::{PI, TAU};

/// Extension length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExtensionLength {
    /// Absolute length in mesh units.
    Absolute(f64),
    /// Multiples of the loop's equivalent diameter `2·sqrt(area/π)`.
    Diameters(f64),
}

/// Alias matching the Meshmixer-style naming used in the UI.
pub type Length = ExtensionLength;

/// Parameters for [`extrude_boundary_loop`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowExtensionParams {
    /// Extension length (absolute or in equivalent diameters).
    pub length: ExtensionLength,
    /// Extrusion direction; default = loop-plane normal pointing away from the mesh.
    pub direction: Option<DVec3>,
    /// Number of rings along the extension; 0 = automatic (≈ isotropic triangles).
    pub segments: usize,
    /// Blend the cross-section into a circle of the same area along the extension.
    pub make_circular: bool,
    /// Fraction of the length over which the circular blend happens (0..=1, default 0.5);
    /// the rest of the extension is a straight circular cylinder.
    pub transition: f64,
    /// Close the end with a planar cap.
    pub cap: bool,
    /// Remesh the end cap to the ring's edge length (otherwise only ring vertices are used).
    pub cap_remesh: bool,
}

impl Default for FlowExtensionParams {
    fn default() -> Self {
        Self {
            length: ExtensionLength::Diameters(5.0),
            direction: None,
            segments: 0,
            make_circular: false,
            transition: 0.5,
            cap: false,
            cap_remesh: true,
        }
    }
}

/// Output of [`extrude_boundary_loop`].
#[derive(Clone, Debug, Default)]
pub struct FlowExtensionResult {
    /// New rings, from the original loop outwards (each in boundary-loop order).
    pub rings: Vec<Vec<u32>>,
    /// The end ring (boundary orientation); open unless capped.
    pub end_loop: Vec<u32>,
    /// Side-wall triangles of the extension.
    pub wall_triangles: Vec<u32>,
    /// End-cap triangles (empty unless capped).
    pub cap_triangles: Vec<u32>,
    /// Whether the end was capped.
    pub capped: bool,
    /// Face group of the wall.
    pub wall_group: i32,
    /// Face group of the cap.
    pub cap_group: Option<i32>,
    /// Unit extrusion direction.
    pub direction: DVec3,
    /// Extension length in mesh units.
    pub length: f64,
    /// Equivalent diameter of the original loop (projected perpendicular to `direction`).
    pub equivalent_diameter: f64,
    /// Area centroid of the original loop.
    pub start_center: DVec3,
    /// Centre of the end ring.
    pub end_center: DVec3,
}

/// Extrude the boundary loop `loop_vertices` (either orientation; e.g.
/// `&edge_loop.vertices` from `mm_core::topology::boundary_loops`, or a loop from a plane
/// cut) into a straight flow extension. The new wall shares the loop's edges, so the result
/// is manifold and consistently oriented; the end ring is planar and perpendicular to the
/// direction (non-planar loops are flattened progressively along the extension).
pub fn extrude_boundary_loop(
    mesh: &mut Mesh,
    loop_vertices: &[u32],
    params: &FlowExtensionParams,
) -> EditResult<FlowExtensionResult> {
    let lp = orient_boundary_loop(mesh, loop_vertices)?;
    let n = lp.len();
    let pts = positions(mesh, &lp);
    let dir = match params.direction {
        Some(d) => {
            if !(d.length() > 1e-300 && d.is_finite()) {
                return Err(EditError::InvalidParameter("direction must be non-zero".into()));
            }
            d.normalize()
        }
        None => loop_outward_direction(mesh, &lp)
            .ok_or_else(|| EditError::InvalidLoop("loop has zero area; give a direction".into()))?,
    };
    let (e1, e2) = frame(dir);
    let m = mean(&pts);
    let p2: Vec<DVec2> = pts
        .iter()
        .map(|p| {
            let d = *p - m;
            DVec2::new(d.dot(e1), d.dot(e2))
        })
        .collect();
    let (a2, c2) = polygon_area_centroid_2d(&p2);
    let area = a2.abs();
    if area <= 0.0 {
        return Err(EditError::InvalidLoop("loop projects to zero area along the direction".into()));
    }
    let c = m + e1 * c2.x + e2 * c2.y;
    let d_eq = 2.0 * (area / PI).sqrt();
    let length = match params.length {
        ExtensionLength::Absolute(l) => l,
        ExtensionLength::Diameters(k) => k * d_eq,
    };
    if !(length.is_finite() && length > 0.0) {
        return Err(EditError::InvalidParameter("extension length must be positive".into()));
    }
    let mut arc = vec![0.0; n + 1];
    for j in 0..n {
        arc[j + 1] = arc[j] + pts[j].distance(pts[(j + 1) % n]);
    }
    let perimeter = arc[n];
    let mean_edge = perimeter / n as f64;
    let segs = if params.segments == 0 { ((length / mean_edge).round() as usize).clamp(1, 10_000) } else { params.segments };

    // Circle of equal area, parametrised by arc length (keeps the ordering monotonic) and
    // rotated to best match the loop's own angles.
    let sigma = if a2 >= 0.0 { 1.0 } else { -1.0 };
    let radius = (area / PI).sqrt();
    let (mut sx, mut sy) = (0.0, 0.0);
    for j in 0..n {
        let rel = p2[j] - c2;
        let th = rel.y.atan2(rel.x);
        let ph = th - sigma * TAU * arc[j] / perimeter;
        sx += ph.cos();
        sy += ph.sin();
    }
    let phi0 = sy.atan2(sx);
    let circle: Vec<DVec3> = (0..n)
        .map(|j| {
            let ph = phi0 + sigma * TAU * arc[j] / perimeter;
            (e1 * ph.cos() + e2 * ph.sin()) * radius
        })
        .collect();
    let inplane: Vec<DVec3> = pts
        .iter()
        .map(|p| {
            let d = *p - c;
            d - dir * d.dot(dir)
        })
        .collect();
    let offset: Vec<f64> = pts.iter().map(|p| (*p - c).dot(dir)).collect();
    let transition = params.transition.clamp(1e-6, 1.0);

    let wall_group = new_group(mesh);
    let mut res = FlowExtensionResult {
        wall_group,
        direction: dir,
        length,
        equivalent_diameter: d_eq,
        start_center: c,
        end_center: c + dir * length,
        ..Default::default()
    };
    let mut prev = lp.clone();
    for k in 1..=segs {
        let t = k as f64 / segs as f64;
        let b = if params.make_circular { smoothstep(t / transition) } else { 0.0 };
        let ring: Vec<u32> = (0..n)
            .map(|j| {
                let p = c + dir * (length * t + offset[j] * (1.0 - t)) + inplane[j] * (1.0 - b) + circle[j] * b;
                mesh.append_vertex(p)
            })
            .collect();
        if let Err(e) = stitch_band(mesh, &prev, &ring, wall_group, &mut res.wall_triangles) {
            rollback(mesh, &res.wall_triangles);
            mesh.remove_isolated_vertices();
            return Err(e);
        }
        res.rings.push(ring.clone());
        prev = ring;
    }
    res.end_loop = prev.clone();
    if params.cap {
        let ring_edge = (0..n).map(|j| mesh.vertex(prev[j]).distance(mesh.vertex(prev[(j + 1) % n]))).sum::<f64>()
            / n as f64;
        let fill = if params.cap_remesh {
            CutFill::FlatRemeshed { target_edge_length: ring_edge.max(1e-12) }
        } else {
            CutFill::Flat
        };
        let cr = cap_oriented_loops(mesh, &[prev], Some(dir), fill);
        res.capped = cr.failed_loops.is_empty() && !cr.triangles.is_empty();
        res.cap_group = cr.groups.first().copied();
        res.cap_triangles = cr.triangles;
    }
    Ok(res)
}
