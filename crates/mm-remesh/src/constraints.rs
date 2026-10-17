//! Vertex and edge constraints shared by the remesher, the reducer and the cleanup tools.
//!
//! Feature edges (sharp creases, face-group borders, mesh boundaries in "constrained"
//! mode) are chained into curves. A vertex inside a chain is a *curve* vertex: it may
//! only slide along its curve and may only be removed by collapsing one of its own
//! curve edges. Chain endpoints (corners: feature valence != 2), vertices touching a
//! fixed edge, region-border vertices and bowties are *fixed*.

use crate::region::Region;
use mm_core::{Mesh, INVALID};
use rayon::prelude::*;

pub(crate) const V_FREE: u8 = 0;
pub(crate) const V_CURVE: u8 = 1;
pub(crate) const V_FIXED: u8 = 2;

pub(crate) const E_NONE: u8 = 0;
pub(crate) const E_FEATURE: u8 = 1;
pub(crate) const E_FIXED: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VCons {
    pub kind: u8,
    pub curve: u32,
}

impl VCons {
    pub const FREE: VCons = VCons { kind: V_FREE, curve: INVALID };
    pub fn on_curve(curve: u32) -> Self {
        VCons { kind: V_CURVE, curve }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ECons {
    pub kind: u8,
    pub curve: u32,
}

impl ECons {
    pub const NONE: ECons = ECons { kind: E_NONE, curve: INVALID };
}

/// Can vertex `v` be removed by collapsing an edge with constraint `e`?
#[inline]
pub(crate) fn removable(v: VCons, e: ECons) -> bool {
    match v.kind {
        V_FREE => e.kind == E_NONE,
        V_CURVE => e.kind == E_FEATURE && e.curve == v.curve,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundaryRule {
    Fixed,
    Curve,
    Free,
}

pub(crate) struct ConstraintOptions {
    pub boundary: BoundaryRule,
    /// Dihedral angle (degrees) above which an edge is a sharp feature.
    pub sharp_deg: Option<f64>,
    /// Face-group borders are feature curves.
    pub groups: bool,
    /// Curve vertices whose curve turns by more than this angle become corners.
    pub corner_deg: Option<f64>,
    /// Flag group-border curves for smoothing: `Some(None)` = every group border,
    /// `Some(Some(split))` = borders between groups `< split` and `>= split`.
    pub smooth_groups: Option<Option<i32>>,
}

pub(crate) struct Constraints {
    pub v: Vec<VCons>,
    pub e: Vec<ECons>,
    pub curve_count: u32,
    /// Per curve: flagged for smoothing (see `ConstraintOptions::smooth_groups`).
    pub smooth: Vec<bool>,
}

/// Classify `edges` and `verts` (the edges and vertices of the region triangles).
pub(crate) fn build_constraints(
    mesh: &Mesh,
    region: &Region,
    edges: &[u32],
    verts: &[u32],
    opts: &ConstraintOptions,
) -> Constraints {
    let mut ec = vec![ECons::NONE; mesh.max_eid()];
    let mut vc = vec![VCons::FREE; mesh.max_vid()];
    let cos_sharp = opts.sharp_deg.map(|d| d.to_radians().cos());

    let kinds: Vec<u8> = edges
        .par_iter()
        .map(|&e| {
            let [t0, t1] = mesh.edge_triangles(e);
            if t1 == INVALID {
                return match opts.boundary {
                    BoundaryRule::Fixed => E_FIXED,
                    BoundaryRule::Curve => E_FEATURE,
                    BoundaryRule::Free => E_NONE,
                };
            }
            if region.contains(t0) != region.contains(t1) {
                return E_FIXED;
            }
            if opts.groups && mesh.tri_group(t0) != mesh.tri_group(t1) {
                return E_FEATURE;
            }
            if let Some(c) = cos_sharp {
                let n0 = mesh.tri_normal(t0);
                let n1 = mesh.tri_normal(t1);
                if n0 != mm_core::DVec3::ZERO && n1 != mm_core::DVec3::ZERO && n0.dot(n1) < c {
                    return E_FEATURE;
                }
            }
            E_NONE
        })
        .collect();
    for (&e, &k) in edges.iter().zip(&kinds) {
        ec[e as usize].kind = k;
    }

    let cos_corner = opts.corner_deg.map(|d| d.to_radians().cos());
    let ecr = &ec;
    let vkinds: Vec<u8> = verts
        .par_iter()
        .map(|&v| {
            let mut fixed = false;
            let mut feat = [INVALID; 2];
            let mut nfeat = 0;
            for &e in mesh.vertex_edges(v) {
                match ecr[e as usize].kind {
                    E_FIXED => fixed = true,
                    E_FEATURE => {
                        if nfeat < 2 {
                            feat[nfeat] = e;
                        }
                        nfeat += 1;
                    }
                    _ => {}
                }
            }
            if !fixed && region.enabled() {
                fixed = mesh.vertex_triangles(v).iter().any(|&t| !region.contains(t));
            }
            if fixed || mesh.vertex_edges(v).is_empty() || mesh.is_bowtie_vertex(v) {
                return V_FIXED;
            }
            match nfeat {
                0 => V_FREE,
                2 => {
                    if let Some(cc) = cos_corner {
                        let p = mesh.vertex(v);
                        let p1 = mesh.vertex(mesh.edge_other_vertex(feat[0], v));
                        let p2 = mesh.vertex(mesh.edge_other_vertex(feat[1], v));
                        let d1 = (p - p1).normalize_or_zero();
                        let d2 = (p2 - p).normalize_or_zero();
                        if d1.dot(d2) < cc {
                            return V_FIXED;
                        }
                    }
                    V_CURVE
                }
                _ => V_FIXED,
            }
        })
        .collect();
    for (&v, &k) in verts.iter().zip(&vkinds) {
        vc[v as usize].kind = k;
    }

    // Chain feature edges into curves, walking through curve vertices.
    let mut curve_count = 0u32;
    let mut smooth = Vec::new();
    for &e in edges {
        if ec[e as usize].kind != E_FEATURE || ec[e as usize].curve != INVALID {
            continue;
        }
        let id = curve_count;
        curve_count += 1;
        smooth.push(opts.smooth_groups.is_some_and(|split| smoothable_group_border(mesh, e, split, cos_sharp)));
        ec[e as usize].curve = id;
        for start in mesh.edge_vertices(e) {
            let mut v = start;
            let mut prev = e;
            loop {
                if vc[v as usize].kind != V_CURVE {
                    break;
                }
                vc[v as usize].curve = id;
                let next = mesh
                    .vertex_edges(v)
                    .iter()
                    .copied()
                    .find(|&x| x != prev && ec[x as usize].kind == E_FEATURE);
                let Some(ne) = next else { break };
                if ec[ne as usize].curve != INVALID {
                    break;
                }
                ec[ne as usize].curve = id;
                v = mesh.edge_other_vertex(ne, v);
                prev = ne;
            }
        }
    }
    Constraints { v: vc, e: ec, curve_count, smooth }
}

/// Is feature edge `e` a face-group border selected for smoothing (and not a sharp crease)?
fn smoothable_group_border(mesh: &Mesh, e: u32, split: Option<i32>, cos_sharp: Option<f64>) -> bool {
    let [t0, t1] = mesh.edge_triangles(e);
    if t1 == INVALID {
        return false;
    }
    let (g0, g1) = (mesh.tri_group(t0), mesh.tri_group(t1));
    if g0 == g1 || split.is_some_and(|s| (g0 < s) == (g1 < s)) {
        return false;
    }
    match cos_sharp {
        Some(c) => mesh.tri_normal(t0).dot(mesh.tri_normal(t1)) >= c,
        None => true,
    }
}
