// SPDX-License-Identifier: AGPL-3.0-only
//! How the faces either side of a body's edge meet, read off the mesh. Ported from ScanRuler
//! (`plugins/reverse/core/edgeLines.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Not every outline of a face is an edge of the part's shape. A round face that goes the
//! whole way round meets itself along a seam; a face a boolean cut in two where it lay on one
//! surface is two faces with an edge between them where the part is one smooth surface.
//! Neither is drawn, as no CAD system draws them. A tangent edge — a round running into a flat
//! — is a real edge of the shape, drawn, dimmed or left out as the operator sets it. How the
//! faces meet is read off the mesh: the faces either side of an outline side each have their
//! own vertices there, carrying each face's own surface normal, so a side whose partner is of
//! the same face is a seam and one whose normals agree with its partner's is tangent.

use std::collections::HashMap;

use glam::DVec3;

use crate::build::BodyMesh;

/// How the faces either side of an edge meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeJoin {
    Sharp,
    Tangent,
    Split,
    Seam,
}

/// How the tangent edges are drawn: as the sharp ones, fainter, or not at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TangentEdges {
    #[default]
    Visible,
    Dimmed,
    Removed,
}

/// Two faces whose normals agree this closely at both ends of a side meet tangent there.
fn tangent_cos() -> f64 {
    0.25f64.to_radians().cos()
}

/// Every side of the triangles that one triangle alone has, as (a, b, triangle).
fn outline_sides(positions: &[DVec3], triangles: &[[u32; 3]]) -> Vec<(u32, u32, usize)> {
    let mut owner: HashMap<(u32, u32), Option<usize>> = HashMap::new();
    let mut order: Vec<(u32, u32)> = Vec::new();
    for (t, tri) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            match owner.get_mut(&key) {
                Some(o) => *o = None,
                None => {
                    owner.insert(key, Some(t));
                    order.push(key);
                }
            }
        }
    }
    order
        .into_iter()
        .filter_map(|k| {
            let t = owner.get(&k).copied().flatten()?;
            (positions[k.0 as usize] != positions[k.1 as usize]).then_some((k.0, k.1, t))
        })
        .collect()
}

fn place_key(positions: &[DVec3], a: u32, b: u32) -> ([i64; 3], [i64; 3]) {
    let at = |v: u32| {
        let p = positions[v as usize];
        [(p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64, (p.z * 1e5).round() as i64]
    };
    let (ka, kb) = (at(a), at(b));
    if ka < kb {
        (ka, kb)
    } else {
        (kb, ka)
    }
}

/// The outline sides, each with how the faces meet along it.
fn read_sides(mesh: &BodyMesh, one_surface: Option<&dyn Fn(u32, u32) -> bool>) -> (Vec<(u32, u32, usize)>, Vec<EdgeJoin>) {
    let sides = outline_sides(&mesh.positions, &mesh.triangles);
    let mut places: HashMap<([i64; 3], [i64; 3]), Vec<usize>> = HashMap::new();
    for (s, (a, b, _)) in sides.iter().enumerate() {
        places.entry(place_key(&mesh.positions, *a, *b)).or_default().push(s);
    }
    let mut joins = vec![EdgeJoin::Sharp; sides.len()];
    let has_normals = mesh.normals.len() == mesh.positions.len();
    for list in places.values() {
        if list.len() != 2 {
            continue;
        }
        let (s, t) = (list[0], list[1]);
        let fs = mesh.tri_face.get(sides[s].2).copied();
        let ft = mesh.tri_face.get(sides[t].2).copied();
        let mut join = EdgeJoin::Sharp;
        if fs.is_some() && fs == ft {
            join = EdgeJoin::Seam;
        } else if has_normals {
            let (a, b) = (sides[s].0, sides[s].1);
            let (mut ta, mut tb) = (sides[t].0, sides[t].1);
            let d = |x: u32, y: u32| mesh.positions[x as usize].distance_squared(mesh.positions[y as usize]);
            if d(a, ta) > d(a, tb) {
                std::mem::swap(&mut ta, &mut tb);
            }
            let n = |v: u32| mesh.normals[v as usize];
            if n(a).dot(n(ta)) >= tangent_cos() && n(b).dot(n(tb)) >= tangent_cos() {
                join = match (one_surface, fs, ft) {
                    (Some(f), Some(x), Some(y)) if f(x, y) => EdgeJoin::Split,
                    _ => EdgeJoin::Tangent,
                };
            }
        }
        joins[s] = join;
        joins[t] = join;
    }
    (sides, joins)
}

fn polyline_distance2(p: &[DVec3], q: DVec3, limit: f64) -> f64 {
    let mut best = limit;
    for w in p.windows(2) {
        let (a, d) = (w[0], w[1] - w[0]);
        let ll = d.length_squared();
        let t = if ll > 0.0 { ((q - a).dot(d) / ll).clamp(0.0, 1.0) } else { 0.0 };
        best = best.min((a + d * t - q).length_squared());
    }
    best
}

/// An edge as the drawing needs it: its hash, the faces beside it, its polyline.
pub struct OutlinedEdge<'a> {
    pub hash: u32,
    pub faces: &'a [u32],
    pub polyline: &'a [DVec3],
}

/// Each edge's share of the outlines and how its faces meet, by the edge's hash (ScanRuler's
/// `edgeShares`): an outline side goes to the edge of its face whose polyline passes nearest
/// the side's middle; what most of an edge's share says is how its faces meet. An edge given
/// no side is taken for sharp.
pub fn edge_joins(mesh: &BodyMesh, edges: &[OutlinedEdge], one_surface: Option<&dyn Fn(u32, u32) -> bool>) -> HashMap<u32, EdgeJoin> {
    struct Candidate<'a> {
        hash: u32,
        p: &'a [DVec3],
        lo: DVec3,
        hi: DVec3,
    }
    let mut round: HashMap<u32, Vec<Candidate>> = HashMap::new();
    for e in edges {
        if e.polyline.len() < 2 {
            continue;
        }
        let lo = e.polyline.iter().fold(DVec3::splat(f64::INFINITY), |a, b| a.min(*b));
        let hi = e.polyline.iter().fold(DVec3::splat(f64::NEG_INFINITY), |a, b| a.max(*b));
        let mut seen = Vec::new();
        for f in e.faces {
            if seen.contains(f) {
                continue;
            }
            seen.push(*f);
            round.entry(*f).or_default().push(Candidate { hash: e.hash, p: e.polyline, lo, hi });
        }
    }
    let (sides, joins) = read_sides(mesh, one_surface);
    let mut votes: HashMap<u32, [usize; 4]> = HashMap::new();
    for (s, (a, b, t)) in sides.iter().enumerate() {
        let Some(face) = mesh.tri_face.get(*t) else { continue };
        let Some(candidates) = round.get(face) else { continue };
        let m = (mesh.positions[*a as usize] + mesh.positions[*b as usize]) / 2.0;
        let mut best = f64::INFINITY;
        let mut which = None;
        for c in candidates {
            let bx = (c.lo - m).max(DVec3::ZERO).max(m - c.hi);
            if bx.length_squared() >= best {
                continue;
            }
            let d = polyline_distance2(c.p, m, best);
            if d < best {
                best = d;
                which = Some(c.hash);
            }
        }
        if let Some(h) = which {
            let k = match joins[s] {
                EdgeJoin::Sharp => 0,
                EdgeJoin::Tangent => 1,
                EdgeJoin::Split => 2,
                EdgeJoin::Seam => 3,
            };
            votes.entry(h).or_insert([0; 4])[k] += 1;
        }
    }
    edges
        .iter()
        .map(|e| {
            let join = match votes.get(&e.hash) {
                Some(c) => {
                    let max = *c.iter().max().unwrap_or(&0);
                    // A tie goes to the join drawn the plainest: sharp before tangent.
                    [EdgeJoin::Sharp, EdgeJoin::Tangent, EdgeJoin::Split, EdgeJoin::Seam][c.iter().position(|x| *x == max).unwrap_or(0)]
                }
                None => EdgeJoin::Sharp,
            };
            (e.hash, join)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{body_mesh, build};
    use crate::frame::WorldAxis;
    use crate::model::{BodyOp, ExtrudeExtent, ExtrudeFeature, FeatureKind, Part, SketchFeature};
    use crate::sketch::{Shape, Sketch};
    use glam::DVec2;

    fn joins_of(sketch: Sketch) -> Vec<EdgeJoin> {
        let mut part = Part::default();
        let regions = crate::build::default_regions(&crate::regions::find_regions(&sketch));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, sketch)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::Join)));
        let built = build(&part);
        let b = &built.bodies[0];
        let mesh = body_mesh(&b.body).unwrap();
        let edges: Vec<OutlinedEdge> = b.topology.edges.iter().map(|e| OutlinedEdge { hash: e.hash, faces: &e.faces, polyline: &e.polyline }).collect();
        let one = |x: u32, y: u32| match (b.topology.face(x), b.topology.face(y)) {
            (Some(a), Some(c)) => crate::topo::on_one_surface(&a.fingerprint, &c.fingerprint),
            _ => false,
        };
        let joins = edge_joins(&mesh, &edges, Some(&one));
        b.topology.edges.iter().map(|e| joins[&e.hash]).collect()
    }

    // Ported from ScanRuler's tests/reEdgeLines.test.ts: a box's edges are all sharp.
    #[test]
    fn a_box_has_twelve_sharp_edges() {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::ZERO, DVec2::new(10.0, 10.0));
        let joins = joins_of(s);
        assert_eq!(joins.len(), 12);
        assert!(joins.iter().all(|j| *j == EdgeJoin::Sharp), "{joins:?}");
    }

    #[test]
    fn a_slot_has_tangent_edges_where_its_rounds_run_into_its_flats() {
        // A stadium: two lines and two half circles, tangent at four places.
        let mut s = Sketch::default();
        s.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(10.0, 0.0) });
        s.add_shape(Shape::Arc { c: DVec2::new(10.0, 5.0), r: 5.0, start: -std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::PI });
        s.add_shape(Shape::Line { a: DVec2::new(10.0, 10.0), b: DVec2::new(0.0, 10.0) });
        s.add_shape(Shape::Arc { c: DVec2::new(0.0, 5.0), r: 5.0, start: std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::PI });
        let joins = joins_of(s);
        let tangent = joins.iter().filter(|j| **j == EdgeJoin::Tangent).count();
        assert_eq!(tangent, 4, "{joins:?}");
    }
}
