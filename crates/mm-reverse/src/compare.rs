// SPDX-License-Identifier: AGPL-3.0-only
//! The deviation map: how far the scan lies from the modelled bodies. Ported from ScanRuler
//! (`plugins/reverse/core/liveMap.ts`, `src/core/deviation/deviation.ts`,
//! `src/core/deviation/surface.ts`, `src/core/field/colormap.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.
//!
//! Every scan vertex near the bodies gets its signed distance from their surface — positive
//! outside the bodies (red on the ramp), negative inside (blue), green within the range. A
//! vertex further than the reach from any body is left as it is. A vertex lying *deeper*
//! inside the bodies than the reach is the one error a ramp about the surface never shows: the
//! scan is the part's skin, so skin deep inside the model is a hole, a pocket or a slot the
//! bodies lack — it is counted apart (`deep`).
//!
//! The map is sparse on purpose. A ghost is one feature of a part — a boss, a pocket, one wall
//! — and the scan around it says nothing about the feature; so only the vertices near the
//! ghost are read, and only those within reach of its surface are painted. The rest of the
//! scan keeps whatever colour it wears.
//!
//! The distance is signed by the angle-weighted pseudonormal of the feature the nearest point
//! lands on (a face, an edge or a vertex of the mesh), as ScanRuler's `NominalSurface` signs
//! it; a sheet has no inside and is signed by its front. The scan may be read by the way it
//! faces ([`ScanSide`]): a vertex whose nearest surface faces away from it — the far skin of a
//! thin wall — is read against the nearest one within reach that faces its way.

use glam::DVec3;
use mm_core::bvh::{Bvh, NearestHit};
use mm_core::Mesh;
use rayon::prelude::*;

use crate::build::BodyMesh;
pub use crate::mesh_graph::curvature::FieldScale;

// ---- the colour map (src/core/field/colormap.ts) ----

/// A colour, three bytes.
pub type Rgb = [u8; 3];

/// The unmeasured scan surface: a machined-aluminium grey.
pub const UNMEASURED_RGB: Rgb = [126, 131, 138];

/// A ramp, and the two colours that stand beyond its ends.
#[derive(Debug, Clone, Copy)]
pub struct Colormap {
    pub id: &'static str,
    pub label: &'static str,
    /// Evenly spaced colours from the low end of the ramp to the high, three bytes each.
    pub stops: &'static [u8],
    /// Caps for a reading past the low end of the ramp, and past the high.
    pub under: Rgb,
    pub over: Rgb,
}

/// Jet, pinned so that the middle of the scale is green — the map ScanRuler opens with.
/// Polymix has no colour-map setting, so it is the one map painted here.
pub const JET: Colormap = Colormap { id: "jet", label: "Jet", stops: &[0, 0, 255, 0, 255, 255, 0, 200, 0, 255, 255, 0, 255, 0, 0], under: [0, 0, 110], over: [130, 0, 0] };

/// Colour of a position `t` along a map's ramp, `t` clamped to 0…1.
pub fn ramp_color(map: &Colormap, t: f64) -> Rgb {
    let s = map.stops;
    let last = s.len() / 3 - 1;
    let u = (if t <= 0.0 { 0.0 } else if t >= 1.0 { 1.0 } else { t }) * last as f64;
    let i = (last - 1).min(u.floor().max(0.0) as usize);
    let f = u - i as f64;
    let a = i * 3;
    let lerp = |k: usize| (s[a + k] as f64 + (s[a + 3 + k] as f64 - s[a + k] as f64) * f).round().clamp(0.0, 255.0) as u8;
    [lerp(0), lerp(1), lerp(2)]
}

/// Snap a ramp position to the middle of one of `bands` equal steps.
pub fn quantize(t: f64, bands: u32) -> f64 {
    if bands < 2 {
        return t;
    }
    let u = t.clamp(0.0, 1.0);
    ((bands as f64 - 1.0).min((u * bands as f64).floor()) + 0.5) / bands as f64
}

/// The colours drawn for a measured value below the scale's `low` and above its `high`.
pub fn scale_caps(scale: &FieldScale, map: &Colormap) -> (Rgb, Rgb) {
    if scale.reversed {
        (map.over, map.under)
    } else {
        (map.under, map.over)
    }
}

/// One value painted as ScanRuler's `paintField` paints it: the ramp within the scale, the
/// map's cap past either end, the unmeasured grey outside the valid window.
pub fn paint_value(v: f64, scale: &FieldScale, map: &Colormap) -> Rgb {
    let (low_cap, high_cap) = scale_caps(scale, map);
    if !(v >= scale.valid_min && v <= scale.valid_max) {
        return UNMEASURED_RGB;
    }
    if v > scale.high {
        return high_cap;
    }
    if v < scale.low {
        return low_cap;
    }
    let span = scale.high - scale.low;
    let mut t = if span > 0.0 { (v - scale.low) / span } else { 0.0 };
    if scale.reversed {
        t = 1.0 - t;
    }
    if let Some(b) = scale.bands {
        t = quantize(t, b);
    }
    ramp_color(map, t)
}

/// How a deviation field is read as colour: symmetric about zero, everything past the search
/// distance left bare.
pub fn deviation_scale(range: f64, max_distance: f64, bands: Option<u32>) -> FieldScale {
    FieldScale { low: -range, high: range, bands, valid_min: -max_distance, valid_max: max_distance, reversed: false }
}

// ---- the live map's numbers (liveMap.ts) ----

/// The range the ramp opens at, ± millimetres.
pub const LIVE_MAP_RANGE_MM: f64 = 0.1;
/// How far past the ramp's range a vertex is still read, as a multiple of the range.
pub const LIVE_MAP_REACH: f64 = 10.0;
/// The smallest range the field takes, millimetres.
pub const LIVE_MAP_MIN_RANGE_MM: f64 = 0.005;
/// The colour a vertex deep inside the bodies wears on the live map.
pub const FAR_INSIDE_RGB: Rgb = [186, 50, 196];
/// The facing limit the scan is read under unless told otherwise, degrees: the Deviation
/// map's — surface facing away, and nothing short of that.
pub const DEFAULT_FACING_DEG: f64 = 90.0;

pub fn live_map_reach(range: f64) -> f64 {
    range.max(LIVE_MAP_MIN_RANGE_MM) * LIVE_MAP_REACH
}

/// How the map is read: the Deviation ramp about zero at ± the range, a cap past either end
/// out to the reach.
pub fn live_map_scale(range: f64) -> FieldScale {
    deviation_scale(range.max(LIVE_MAP_MIN_RANGE_MM), live_map_reach(range), None)
}

// ---- the surface (src/core/deviation/surface.ts) ----

/// A closest point on the surface.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    /// Positive outside, by the pseudonormal of what the point lands on.
    pub signed: f64,
    pub distance: f64,
    pub point: DVec3,
    /// The outward normal there (the pseudonormal).
    pub normal: DVec3,
    /// The triangle, by its index in the meshes as given (concatenated).
    pub tri: usize,
}

/// One body as the surface is made of it: its mesh, a tag the hits name it by, and whether it
/// is a sheet (no inside: read on its front).
pub struct SurfaceBody<'a> {
    pub tag: u64,
    pub mesh: &'a BodyMesh,
    pub sheet: bool,
}

/// The bodies' surface, ready for distance queries — several bodies read as one part.
pub struct Surface {
    mesh: Mesh,
    bvh: Bvh,
    /// Per source triangle: the body's tag, the B-rep face, whether a sheet's.
    pub body_of: Vec<u64>,
    pub face_of: Vec<i64>,
    pub sheet_of: Vec<bool>,
    pub bbox_min: DVec3,
    pub bbox_max: DVec3,
}

impl Surface {
    /// The surface of solid bodies (tagged by their order).
    pub fn new(bodies: &[&BodyMesh]) -> Option<Surface> {
        let list: Vec<SurfaceBody> = bodies.iter().enumerate().map(|(i, m)| SurfaceBody { tag: i as u64, mesh: m, sheet: false }).collect();
        Surface::of_bodies(&list)
    }

    /// The surface of several bodies, each triangle remembering its body and face. The meshes
    /// are welded so a point landing on an edge between two faces is signed by both.
    pub fn of_bodies(bodies: &[SurfaceBody]) -> Option<Surface> {
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        let mut body_of = Vec::new();
        let mut face_of = Vec::new();
        let mut sheet_of = Vec::new();
        for b in bodies {
            let base = positions.len() as u32;
            positions.extend_from_slice(&b.mesh.positions);
            for (i, t) in b.mesh.triangles.iter().enumerate() {
                tris.push([t[0] + base, t[1] + base, t[2] + base]);
                body_of.push(b.tag);
                face_of.push(b.mesh.tri_face.get(i).map_or(-1, |f| *f as i64));
                sheet_of.push(b.sheet);
            }
        }
        if tris.is_empty() {
            return None;
        }
        let (welded, wtris) = weld(&positions, &tris);
        let groups: Vec<i32> = (0..wtris.len() as i32).collect();
        let (mesh, _) = Mesh::from_triangles(&welded, &wtris, Some(&groups));
        if mesh.triangle_count() == 0 {
            return None;
        }
        let bvh = Bvh::build(&mesh);
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in &positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        Some(Surface { mesh, bvh, body_of, face_of, sheet_of, bbox_min: lo, bbox_max: hi })
    }

    /// The diagonal of the surface's box.
    pub fn diagonal(&self) -> f64 {
        (self.bbox_max - self.bbox_min).length()
    }

    fn src(&self, tid: u32) -> usize {
        self.mesh.tri_group(tid).max(0) as usize
    }

    /// The angle-weighted pseudonormal of what a closest point lands on: the face itself, an
    /// edge (its two faces), or a vertex (its faces, by their angle there).
    fn pseudo_normal(&self, tid: u32, bary: DVec3) -> DVec3 {
        let own = self.mesh.tri_normal(tid);
        const EPS: f64 = 1e-6;
        let zero = [bary.x.abs() < EPS, bary.y.abs() < EPS, bary.z.abs() < EPS];
        let nz = zero.iter().filter(|z| **z).count();
        let tv = self.mesh.triangle(tid);
        if nz >= 2 {
            let k = (0..3).find(|k| !zero[*k]).unwrap_or(0);
            let v = tv[k];
            let p = self.mesh.vertex(v);
            let mut n = DVec3::ZERO;
            for t in self.mesh.vertex_triangles(v) {
                let tri = self.mesh.triangle(t);
                let j = (0..3).find(|j| tri[*j] == v).unwrap_or(0);
                let a = self.mesh.vertex(tri[(j + 1) % 3]) - p;
                let b = self.mesh.vertex(tri[(j + 2) % 3]) - p;
                let angle = a.angle_between(b);
                if angle.is_finite() {
                    n += self.mesh.tri_normal(t) * angle;
                }
            }
            return n.try_normalize().unwrap_or(own);
        }
        if nz == 1 {
            // The edge across from the corner with no weight.
            let k = (0..3).find(|k| zero[*k]).unwrap_or(0);
            let e = self.mesh.tri_edges(tid)[(k + 1) % 3];
            let mut n = DVec3::ZERO;
            for t in self.mesh.edge_triangles(e) {
                if t != mm_core::INVALID && self.mesh.is_triangle(t) {
                    n += self.mesh.tri_normal(t);
                }
            }
            return n.try_normalize().unwrap_or(own);
        }
        own
    }

    fn hit_of(&self, p: DVec3, h: NearestHit) -> Hit {
        let d = h.dist_sq.sqrt();
        let src = self.src(h.tid);
        let sheet = self.sheet_of.get(src).copied().unwrap_or(false);
        let tri_n = self.mesh.tri_normal(h.tid);
        let pn = if sheet { tri_n } else { self.pseudo_normal(h.tid, h.bary) };
        let outside = (p - h.point).dot(pn) >= 0.0;
        let signed = if outside { d } else { -d };
        let normal = if d > 1e-9 && !sheet { ((p - h.point) / d) * if outside { 1.0 } else { -1.0 } } else { pn };
        Hit { signed, distance: d, point: h.point, normal, tri: src }
    }

    /// The closest point within `max_distance` (unbounded with infinity).
    pub fn closest(&self, p: DVec3, max_distance: f64) -> Option<Hit> {
        let h = self.bvh.nearest(&self.mesh, p, max_distance)?;
        Some(self.hit_of(p, h))
    }

    /// The closest point of a triangle whose own normal turns from `s` by no more than
    /// `min_facing` allows (a cosine), within `reach`.
    pub fn closest_facing(&self, p: DVec3, s: DVec3, min_facing: f64, reach: f64) -> Option<Hit> {
        let plain = self.bvh.nearest(&self.mesh, p, reach)?;
        let mut r = (plain.dist_sq.sqrt() * 1.5).max(reach * 1e-3).max(1e-6).min(reach);
        loop {
            let mut best: Option<NearestHit> = None;
            self.bvh.query_sphere(&self.mesh, p, r, |t| {
                if self.mesh.tri_normal(t).dot(s) < min_facing {
                    return;
                }
                let [a, b, c] = self.mesh.tri_positions(t);
                let (q, bary) = mm_core::geom::closest_point_on_triangle(p, a, b, c);
                let d2 = q.distance_squared(p);
                if best.is_none_or(|x| d2 < x.dist_sq) {
                    best = Some(NearestHit { tid: t, point: q, dist_sq: d2, bary });
                }
            });
            if let Some(h) = best {
                if h.dist_sq.sqrt() <= r {
                    return Some(self.hit_of(p, h));
                }
            }
            if r >= reach {
                return None;
            }
            r = (r * 2.0).min(reach);
        }
    }

    /// The signed distance of a point from the surface — positive outside — or `None` beyond
    /// `reach`.
    pub fn signed_distance(&self, p: DVec3, reach: f64) -> Option<f64> {
        self.closest(p, reach).map(|h| h.signed)
    }

    /// Whether a point lies inside the bodies.
    pub fn contains(&self, p: DVec3) -> bool {
        self.closest(p, f64::INFINITY).is_some_and(|h| h.signed < 0.0)
    }
}

/// Weld coincident points (exactly the same coordinates), so faces meshed apart share their
/// edge's points.
fn weld(positions: &[DVec3], tris: &[[u32; 3]]) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let mut map: std::collections::HashMap<[u64; 3], u32> = std::collections::HashMap::new();
    let mut out = Vec::new();
    let mut remap = Vec::with_capacity(positions.len());
    for p in positions {
        let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        let id = *map.entry(key).or_insert_with(|| {
            out.push(*p);
            (out.len() - 1) as u32
        });
        remap.push(id);
    }
    let tris = tris.iter().map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]]).collect();
    (out, tris)
}

// ---- reading the scan by its side (liveMap.ts ScanSide, sideReader) ----

/// A vertex no surface within reach faces: read against the nearest all the same, or left
/// out and counted apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Unfacing {
    #[default]
    Keep,
    Drop,
}

/// Which side of the model the scan is read on.
#[derive(Debug, Clone, Copy)]
pub struct ScanSide<'a> {
    /// The scan's vertex normals, one per point (any winding: it is read off the scan).
    pub normals: &'a [DVec3],
    /// The facing limit, degrees; `None` for the nearest whatever it faces.
    pub facing_deg: Option<f64>,
    pub unfacing: Unfacing,
    /// Only the scan whose normal faces out of the model.
    pub skin_outer: bool,
}

/// What reading the scan by its side came to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SideCounts {
    pub refaced: usize,
    pub unfacing: usize,
    pub back: usize,
    pub inward: usize,
}

impl SideCounts {
    pub fn add(&mut self, o: SideCounts) {
        self.refaced += o.refaced;
        self.unfacing += o.unfacing;
        self.back += o.back;
        self.inward += o.inward;
    }
}

/// What one read came to.
pub enum Read {
    Read(Hit),
    None,
    Out,
}

/// The scan read against a surface by its side.
pub struct SideReader<'a> {
    surface: &'a Surface,
    side: Option<ScanSide<'a>>,
    outward: f64,
    min_facing: Option<f64>,
    sheet_facing: f64,
}

/// Which way the scan's normals point: 1 out of the material, -1 into it — the sign of their
/// summed agreement with the surface they lie nearest, each vote weighed down by its distance
/// past `near`.
pub fn scan_normals_sign(surface: &Surface, points: &[DVec3], normals: &[DVec3], vertices: &[u32], near: Option<f64>) -> f64 {
    let n = vertices.len();
    let step = (n / 2000).max(1);
    let mut agreement = 0.0;
    let mut i = 0;
    while i < n {
        let v = vertices[i] as usize;
        if let (Some(p), Some(sn)) = (points.get(v), normals.get(v)) {
            if let Some(h) = surface.closest(*p, f64::INFINITY) {
                let weight = near.map_or(1.0, |near| 1.0 / (1.0 + (h.distance / near).powi(2)));
                agreement += weight * sn.dot(h.normal);
            }
        }
        i += step;
    }
    if agreement < 0.0 {
        -1.0
    } else {
        1.0
    }
}

impl<'a> SideReader<'a> {
    pub fn new(surface: &'a Surface, points: &[DVec3], side: Option<ScanSide<'a>>, voters: &[u32], near: f64) -> SideReader<'a> {
        let min_facing = side.and_then(|s| s.facing_deg).map(|d| d.to_radians().cos());
        let sheet_facing = min_facing.unwrap_or(-1.0).max(DEFAULT_FACING_DEG.to_radians().cos());
        let outward = side.map_or(1.0, |s| scan_normals_sign(surface, points, s.normals, voters, Some(near)));
        SideReader { surface, side, outward, min_facing, sheet_facing }
    }

    fn normal_of(&self, v: usize) -> DVec3 {
        self.side.and_then(|s| s.normals.get(v).copied()).unwrap_or(DVec3::ZERO) * self.outward
    }

    /// Read vertex `v` at `p`: within `max_distance`; a facing surface looked for within
    /// `facing_reach`.
    pub fn read(&self, v: usize, p: DVec3, max_distance: f64, facing_reach: f64, counts: &mut SideCounts) -> Read {
        let Some(hit) = self.surface.closest(p, max_distance) else { return Read::None };
        let Some(side) = self.side else { return Read::Read(hit) };
        let sn = self.normal_of(v);
        let facing = sn.dot(hit.normal);
        let on_sheet = self.surface.sheet_of.get(hit.tri).copied().unwrap_or(false);
        if side.skin_outer && !on_sheet && facing < 0.0 {
            counts.inward += 1;
            return Read::Out;
        }
        let limit = if on_sheet { Some(self.sheet_facing) } else { self.min_facing };
        let Some(limit) = limit else { return Read::Read(hit) };
        if facing >= limit {
            return Read::Read(hit);
        }
        if let Some(h) = self.surface.closest_facing(p, sn, limit, facing_reach.min(max_distance)) {
            if sn.dot(h.normal) >= limit {
                counts.refaced += 1;
                return Read::Read(h);
            }
        }
        if on_sheet {
            counts.back += 1;
            return Read::Out;
        }
        counts.unfacing += 1;
        if side.unfacing == Unfacing::Drop {
            Read::Out
        } else {
            Read::Read(hit)
        }
    }

    /// Whether a reading of vertex `v` past the reach belongs to the side asked for.
    pub fn keeps_far(&self, v: usize, hit: &Hit) -> bool {
        let Some(side) = self.side else { return true };
        if self.normal_of(v).dot(hit.normal) >= 0.0 {
            return true;
        }
        let on_sheet = self.surface.sheet_of.get(hit.tri).copied().unwrap_or(false);
        !(side.skin_outer || on_sheet)
    }
}

/// The scan vertices whose position lies inside a box grown by `margin` on every side.
pub fn vertices_near_box(points: &[DVec3], min: DVec3, max: DVec3, margin: f64) -> Vec<u32> {
    let (lo, hi) = (min - DVec3::splat(margin), max + DVec3::splat(margin));
    (0..points.len() as u32)
        .filter(|v| {
            let p = points[*v as usize];
            !(p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y || p.z < lo.z || p.z > hi.z)
        })
        .collect()
}

/// A reading at some of the scan's vertices.
#[derive(Debug, Clone, Default)]
pub struct SparseField {
    /// Scan vertex indices, ascending.
    pub subset: Vec<u32>,
    /// Signed millimetres per subset entry, positive outside the surface.
    pub values: Vec<f32>,
    /// Vertices deeper inside than the reach (read against the bodies only).
    pub far_inside: Option<Vec<u32>>,
}

/// The whole map for one range: the vertices within `LIVE_MAP_REACH` ranges of the surface and
/// their signed distances — and, with `far_inside`, the vertices deep inside it.
pub fn live_map(surface: &Surface, points: &[DVec3], range: f64, far_inside: bool, side: Option<ScanSide>) -> SparseField {
    let reach = live_map_reach(range);
    let near = vertices_near_box(points, surface.bbox_min, surface.bbox_max, reach);
    let reader = SideReader::new(surface, points, side, &near, range.max(LIVE_MAP_MIN_RANGE_MM));
    let read: Vec<(u32, Option<f32>, bool)> = near
        .par_iter()
        .map(|v| {
            let mut c = SideCounts::default();
            match reader.read(*v as usize, points[*v as usize], f64::INFINITY, reach, &mut c) {
                Read::Read(h) if h.signed.abs() <= reach => (*v, Some(h.signed as f32), false),
                Read::Read(h) => (*v, None, far_inside && h.signed < -reach),
                _ => (*v, None, false),
            }
        })
        .collect();
    let mut field = SparseField::default();
    let mut deep = Vec::new();
    for (v, value, d) in read {
        if let Some(x) = value {
            field.subset.push(v);
            field.values.push(x);
        } else if d {
            deep.push(v);
        }
    }
    if far_inside {
        field.far_inside = Some(deep);
    }
    field
}

/// What the box says about the map: how many vertices it read, how many lie within the range,
/// and how many lie deep inside the bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MapSummary {
    pub points: usize,
    pub within: usize,
    pub far_inside: usize,
}

pub fn sparse_summary(field: &SparseField, range: f64) -> MapSummary {
    let within = field.values.iter().filter(|v| (v.abs() as f64) <= range).count();
    MapSummary { points: field.values.len(), within, far_inside: field.far_inside.as_ref().map_or(0, |f| f.len()) }
}

/// The field's vertices and colours as the viewport paints them: the ramp over the read
/// vertices, the far-inside colour over the deep ones.
pub fn painted_sparse(field: &SparseField, scale: &FieldScale, map: &Colormap) -> (Vec<u32>, Vec<Rgb>) {
    let mut subset = field.subset.clone();
    let mut rgb: Vec<Rgb> = field.values.iter().map(|v| paint_value(*v as f64, scale, map)).collect();
    if let Some(deep) = &field.far_inside {
        subset.extend_from_slice(deep);
        rgb.extend(std::iter::repeat_n(FAR_INSIDE_RGB, deep.len()));
    }
    (subset, rgb)
}

/// Area-weighted vertex normals of an indexed mesh (any winding).
pub fn vertex_normals(positions: &[DVec3], triangles: &[[u32; 3]]) -> Vec<DVec3> {
    let mut n = vec![DVec3::ZERO; positions.len()];
    for t in triangles {
        let (Some(a), Some(b), Some(c)) = (positions.get(t[0] as usize), positions.get(t[1] as usize), positions.get(t[2] as usize)) else { continue };
        let x = (*b - *a).cross(*c - *a);
        for v in t {
            n[*v as usize] += x;
        }
    }
    n.iter().map(|v| v.normalize_or_zero()).collect()
}

// ---- the older whole-scan form, kept for its callers ----

/// A deviation map over the scan's vertices.
#[derive(Debug, Clone, Default)]
pub struct DeviationMap {
    /// Per vertex: the signed distance, NaN where the vertex is beyond the reach.
    pub values: Vec<f32>,
    /// Per vertex: lying deeper inside the bodies than the reach (skin the bodies lack).
    pub deep: Vec<bool>,
    /// Vertices within ± the range, and vertices measured at all.
    pub within: usize,
    pub measured: usize,
    pub deep_count: usize,
    pub range: f64,
}

/// The scan's vertices against the bodies: distances within `10 × range` are measured, and
/// (with `find_deep`) vertices further in than that are flagged.
pub fn deviation_map(surface: &Surface, points: &[DVec3], range: f64, find_deep: bool) -> DeviationMap {
    deviation_map_side(surface, points, range, find_deep, None)
}

/// As [`deviation_map`], the scan read by its side.
pub fn deviation_map_side(surface: &Surface, points: &[DVec3], range: f64, find_deep: bool, side: Option<ScanSide>) -> DeviationMap {
    let field = live_map(surface, points, range, find_deep, side);
    let mut map = DeviationMap { range, values: vec![f32::NAN; points.len()], deep: vec![false; points.len()], ..Default::default() };
    for (v, x) in field.subset.iter().zip(&field.values) {
        map.values[*v as usize] = *x;
    }
    for v in field.far_inside.iter().flatten() {
        map.deep[*v as usize] = true;
    }
    let s = sparse_summary(&field, range);
    map.measured = s.points;
    map.within = s.within;
    map.deep_count = s.far_inside;
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{body_mesh, build};
    use crate::frame::WorldAxis;
    use crate::model::{BodyOp, ExtrudeExtent, ExtrudeFeature, FeatureKind, Part, SketchFeature};
    use crate::sketch::Sketch;
    use glam::DVec2;

    #[test]
    fn a_box_reads_its_own_surface_as_zero_and_the_far_side_as_outside() {
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(10.0, 10.0));
        let regions = crate::build::default_regions(&crate::regions::find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::Join)));
        let built = build(&part);
        let mesh = body_mesh(&built.bodies[0].body).unwrap();
        let surface = Surface::new(&[&mesh]).unwrap();
        let points = [DVec3::new(5.0, 5.0, 10.0), DVec3::new(5.0, 5.0, 10.3), DVec3::new(5.0, 5.0, 9.8), DVec3::new(5.0, 5.0, 5.0), DVec3::new(50.0, 5.0, 5.0)];
        let map = deviation_map(&surface, &points, 0.1, true);
        assert!(map.values[0].abs() < 1e-6);
        assert!((map.values[1] - 0.3).abs() < 1e-6);
        assert!((map.values[2] + 0.2).abs() < 1e-6);
        assert!(map.values[3].is_nan() && map.deep[3], "the box's middle is deep inside");
        assert!(map.values[4].is_nan() && !map.deep[4]);
        assert_eq!(map.within, 1);
        assert_eq!(map.deep_count, 1);
    }

    /// A closed box `size` across about `at`, four cells a side, as a body mesh (welded).
    pub(crate) fn box_mesh(size: f64, at: DVec3) -> BodyMesh {
        let m = mm_core::primitives::cube(size, 4);
        let positions: Vec<DVec3> = m.positions().iter().map(|p| *p + at).collect();
        let ids: Vec<u32> = m.triangle_ids().collect();
        let triangles: Vec<[u32; 3]> = ids.iter().map(|t| m.triangle(*t)).collect();
        let n = triangles.len();
        BodyMesh { normals: vertex_normals(&positions, &triangles), positions, triangles, tri_face: vec![0; n], ..Default::default() }
    }

    const SIZE: f64 = 20.0;

    fn scan() -> Vec<(DVec3, Option<f64>)> {
        vec![
            (DVec3::new(0.0, 0.0, 10.0), Some(0.0)),
            (DVec3::new(3.0, -2.0, 10.05), Some(0.05)),
            (DVec3::new(-4.0, 1.0, 9.93), Some(-0.07)),
            (DVec3::new(10.5, 0.0, 0.0), Some(0.5)),
            (DVec3::new(0.0, 0.0, 8.0), Some(-2.0)),
            (DVec3::new(0.0, 0.0, 40.0), None),
            (DVec3::new(-30.0, 30.0, 0.0), None),
        ]
    }

    fn points() -> Vec<DVec3> {
        scan().iter().map(|s| s.0).collect()
    }

    // Ported from ScanRuler's tests/reLiveMap.test.ts.
    #[test]
    fn reads_the_vertices_within_reach_and_no_others_signed_outward() {
        let m = box_mesh(SIZE, DVec3::ZERO);
        let surface = Surface::new(&[&m]).unwrap();
        let range = 0.1;
        let field = live_map(&surface, &points(), range, false, None);
        let reach = live_map_reach(range);
        assert!((reach - LIVE_MAP_REACH * range).abs() < 1e-9);
        for (v, (at, expect)) in scan().iter().enumerate() {
            let read = field.subset.iter().position(|s| *s as usize == v).map(|i| field.values[i] as f64);
            match expect {
                Some(e) if e.abs() <= reach => assert!((read.unwrap() - e).abs() < 1e-5, "vertex {v} at {at}"),
                _ => assert!(read.is_none(), "vertex {v} at {at}"),
            }
        }
        assert!(field.subset.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn widens_its_reach_with_the_range() {
        let m = box_mesh(SIZE, DVec3::ZERO);
        let surface = Surface::new(&[&m]).unwrap();
        let wide = live_map(&surface, &points(), 0.5, false, None);
        let i = wide.subset.iter().position(|v| *v == 4).expect("the vertex 2 mm inside is read at 0.5");
        assert!((wide.values[i] + 2.0).abs() < 1e-5);
    }

    #[test]
    fn sieves_by_the_box_before_asking_the_tree() {
        assert_eq!(vertices_near_box(&points(), DVec3::splat(-10.0), DVec3::splat(10.0), 1.0), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn reads_several_bodies_as_one_part() {
        let a = box_mesh(SIZE, DVec3::ZERO);
        let b = box_mesh(SIZE, DVec3::new(40.0, 0.0, 0.0));
        let two = Surface::new(&[&a, &b]).unwrap();
        let pts = [DVec3::new(0.0, 0.0, 10.02), DVec3::new(40.0, 0.0, 9.98), DVec3::new(20.0, 0.0, 0.0)];
        let field = live_map(&two, &pts, 0.1, false, None);
        assert_eq!(field.subset, vec![0, 1]);
        assert!((field.values[0] - 0.02).abs() < 1e-5);
        assert!((field.values[1] + 0.02).abs() < 1e-5);
        assert!(Surface::new(&[]).is_none());
    }

    #[test]
    fn paints_the_ramp_within_the_range_the_caps_past_it() {
        let m = box_mesh(SIZE, DVec3::ZERO);
        let surface = Surface::new(&[&m]).unwrap();
        let range = 0.1;
        let field = live_map(&surface, &points(), range, false, None);
        let scale = live_map_scale(range);
        let colour = |v: u32| {
            let i = field.subset.iter().position(|s| *s == v).unwrap();
            paint_value(field.values[i] as f64, &scale, &JET)
        };
        assert_eq!(colour(0), ramp_color(&JET, 0.5));
        assert_eq!(colour(3), JET.over);
        let inside = colour(2);
        assert_ne!(inside, JET.under);
        assert!(inside[2] > inside[0]);
        assert_eq!(sparse_summary(&field, range), MapSummary { points: 4, within: 3, far_inside: 0 });
    }

    #[test]
    fn names_the_vertices_deep_inside_and_paints_them_apart() {
        let m = box_mesh(SIZE, DVec3::ZERO);
        let surface = Surface::new(&[&m]).unwrap();
        let range = 0.1;
        assert!(live_map(&surface, &points(), range, false, None).far_inside.is_none());
        let field = live_map(&surface, &points(), range, true, None);
        assert_eq!(field.far_inside.clone().unwrap(), vec![4]);
        assert!(!field.subset.contains(&4));
        assert_eq!(sparse_summary(&field, range), MapSummary { points: 4, within: 3, far_inside: 1 });
        let (subset, rgb) = painted_sparse(&field, &live_map_scale(range), &JET);
        assert_eq!(subset.len(), 5);
        let i = subset.iter().position(|v| *v == 4).unwrap();
        assert_eq!(rgb[i], FAR_INSIDE_RGB);
    }

    #[test]
    fn jet_is_green_in_the_middle_and_the_caps_stand_apart() {
        assert_eq!(ramp_color(&JET, 0.5), [0, 200, 0]);
        assert_eq!(ramp_color(&JET, 0.0), [0, 0, 255]);
        assert_eq!(ramp_color(&JET, 1.0), [255, 0, 0]);
        let s = deviation_scale(1.0, 2.0, None);
        assert_eq!(paint_value(1.5, &s, &JET), JET.over);
        assert_eq!(paint_value(-1.5, &s, &JET), JET.under);
        assert_eq!(paint_value(3.0, &s, &JET), UNMEASURED_RGB);
        assert_eq!(paint_value(f64::NAN, &s, &JET), UNMEASURED_RGB);
        assert!((quantize(0.51, 4) - 0.625).abs() < 1e-12);
    }

    // Ported from ScanRuler's tests/reCompareSide.test.ts: a thin wall with only the outer
    // skin modelled; the inner skin's vertices face back at it.
    #[test]
    fn a_vertex_whose_nearest_surface_faces_away_is_read_against_one_that_faces_it() {
        // Two boxes standing 1 mm apart: a vertex 0.4 mm off the first's +X face, facing -X
        // (it came off the second's -X face), is refaced onto the second.
        let a = box_mesh(SIZE, DVec3::ZERO);
        let b = box_mesh(SIZE, DVec3::new(SIZE + 1.0, 0.0, 0.0));
        let surface = Surface::new(&[&a, &b]).unwrap();
        let pts = vec![DVec3::new(10.4, 0.0, 0.0), DVec3::new(0.0, 0.0, 10.0)];
        let normals = vec![DVec3::new(-1.0, 0.0, 0.0), DVec3::Z];
        let side = ScanSide { normals: &normals, facing_deg: Some(DEFAULT_FACING_DEG), unfacing: Unfacing::Keep, skin_outer: false };
        let field = live_map(&surface, &pts, 1.0, false, Some(side));
        let plain = live_map(&surface, &pts, 1.0, false, None);
        assert!((plain.values[0] - 0.4).abs() < 1e-5, "nearest: 0.4 out of the first box");
        assert!((field.values[0] - 0.6).abs() < 1e-5, "faced: 0.6 out of the second box, {}", field.values[0]);
    }
}
