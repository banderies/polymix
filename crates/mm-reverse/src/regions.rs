// SPDX-License-Identifier: AGPL-3.0-only
//! The closed regions of a sketch — what Extrude and Revolve take. Where ScanRuler walks its
//! own planar graph (`plugins/reverse/core/regions.ts`), this port hands the sketch's outline
//! to SolidCraft's profile finder (`solidcraft_sketch::find_profiles`, MIT OR Apache-2.0),
//! which splits crossing pieces, nests holes and gives back exact boundary segments the kernel
//! builds faces from.
//!
//! A region is known by its *key*: the sorted ids of the pieces round its outer loop. The key
//! survives a solve and an edit that leaves the outline's pieces as they were, so a feature
//! keeps the regions it was made with.

use glam::DVec2;
use solidcraft_geom::{Region2, Vec2 as SVec2};

use crate::sketch::{EntityId, Shape, Sketch};

/// A closed region of the sketch.
#[derive(Debug, Clone)]
pub struct Region {
    pub key: String,
    /// The pieces round the outer loop, and round each hole.
    pub outer: Vec<EntityId>,
    pub holes: Vec<Vec<EntityId>>,
    /// The exact profile, in sheet millimetres.
    pub region: Region2,
    pub area: f64,
    pub centroid: DVec2,
}

impl Region {
    /// Whether a sheet point lies inside the region (in the outer loop, out of every hole).
    pub fn contains(&self, p: DVec2) -> bool {
        let inside = |l: &solidcraft_geom::Loop2| point_in_polygon(&l.polyline(0.01).iter().map(|v| DVec2::new(v.x, v.y)).collect::<Vec<_>>(), p);
        inside(&self.region.outer) && !self.region.holes.iter().any(inside)
    }

    /// The outer loop as a polyline, for drawing.
    pub fn outline(&self, tol: f64) -> Vec<Vec<DVec2>> {
        self.region.loops().map(|l| l.polyline(tol).iter().map(|v| DVec2::new(v.x, v.y)).collect()).collect()
    }
}

pub fn point_in_polygon(poly: &[DVec2], p: DVec2) -> bool {
    let mut inside = false;
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn sv(p: DVec2) -> SVec2 {
    SVec2::new(p.x, p.y)
}

fn curve_name(id: EntityId) -> String {
    format!("e{id}")
}

fn entity_of(name: &str) -> Option<EntityId> {
    name.strip_prefix('e')?.parse().ok()
}

/// The key a region with these outer pieces is known by.
pub fn region_key(outer: &[EntityId]) -> String {
    let mut ids = outer.to_vec();
    ids.sort_unstable();
    ids.dedup();
    ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
}

/// The sketch's outline in SolidCraft's sketch model: the same pieces, ends that coincide
/// sharing one point.
fn to_solidcraft(sketch: &Sketch) -> solidcraft_sketch::Sketch {
    let mut sk = solidcraft_sketch::Sketch::new();
    let mut points: Vec<(DVec2, usize)> = Vec::new();
    let mut point = |sk: &mut solidcraft_sketch::Sketch, p: DVec2| -> Option<usize> {
        if let Some((_, i)) = points.iter().find(|(q, _)| (*q - p).length() < 1e-6) {
            return Some(*i);
        }
        let i = sk.add_point(sv(p), None).ok()?;
        points.push((p, i));
        Some(i)
    };
    for e in &sketch.entities {
        if e.construction {
            continue;
        }
        let name = curve_name(e.id);
        match e.shape {
            Shape::Line { a, b } => {
                if (b - a).length() < 1e-9 {
                    continue;
                }
                if let (Some(pa), Some(pb)) = (point(&mut sk, a), point(&mut sk, b)) {
                    if pa != pb {
                        let _ = sk.add_line_pts(pa, pb, Some(&name));
                    }
                }
            }
            Shape::Circle { c, r } => {
                let _ = sk.add_circle(sv(c), r, None, Some(&name));
            }
            Shape::Arc { c, r, start, sweep } => {
                if sweep >= std::f64::consts::TAU - 1e-9 {
                    let _ = sk.add_circle(sv(c), r, None, Some(&name));
                    continue;
                }
                let a = crate::sketch::arc_point(c, r, start);
                let b = crate::sketch::arc_point(c, r, start + sweep);
                let ends = [None, point(&mut sk, a), point(&mut sk, b)];
                let _ = sk.add_arc(sv(c), sv(a), sv(b), ends, Some(&name));
            }
            // A spline bounds no region yet (SKT pass 2 lays it here as Béziers).
            Shape::Point { .. } | Shape::Spline(_) => {}
        }
    }
    sk
}

/// The sketch's closed regions, largest first.
pub fn find_regions(sketch: &Sketch) -> Vec<Region> {
    let sk = to_solidcraft(sketch);
    let mut out: Vec<Region> = solidcraft_sketch::find_profiles(&sk)
        .into_iter()
        .map(|p| {
            let outer: Vec<EntityId> = p.outer_curves.iter().filter_map(|n| entity_of(n)).collect();
            let holes = p.hole_curves.iter().map(|h| h.iter().filter_map(|n| entity_of(n)).collect()).collect();
            Region { key: region_key(&outer), outer, holes, area: p.area, centroid: DVec2::new(p.centroid.x, p.centroid.y), region: p.region }
        })
        .collect();
    out.sort_by(|a, b| b.area.total_cmp(&a.area));
    out
}

/// The region a sheet point falls in: the smallest that contains it.
pub fn region_at(regions: &[Region], p: DVec2) -> Option<&Region> {
    regions.iter().filter(|r| r.contains(p)).min_by(|a, b| a.area.total_cmp(&b.area))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_rectangle_with_a_hole_in_it() {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(40.0, 20.0));
        s.add_shape(Shape::Circle { c: DVec2::new(20.0, 10.0), r: 4.0 });
        let regions = find_regions(&s);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].holes.len(), 1);
        assert!((regions[0].area - (800.0 - std::f64::consts::PI * 16.0)).abs() < 0.5, "{}", regions[0].area);
        assert!(regions[1].holes.is_empty());
        assert_eq!(region_at(&regions, DVec2::new(20.0, 10.0)).unwrap().key, regions[1].key);
        assert_eq!(region_at(&regions, DVec2::new(2.0, 2.0)).unwrap().key, regions[0].key);
    }
}
