// SPDX-License-Identifier: AGPL-3.0-only
//! Crease-aware shading normals: sharp edges drawn sharp.
//!
//! A viewer that shades with one normal per welded vertex, the area-weighted sum of the faces
//! that meet there, is right on a scan: the surface is dense and smooth, and one normal per
//! vertex is what a smooth surface has. On a low-poly mesh out of a CAD system it is wrong at
//! every edge: a cube's corner gets a normal pointing out of the corner, and each of its flat
//! faces shades as a gradient from that corner inward, so a box looks like a pillow and a
//! hole looks like a dent.
//!
//! The fix is the one every modelling tool ships as "auto smooth": a vertex on a sharp edge
//! is drawn as two vertices, one per side, each with the normal of its own side. Around each
//! vertex the incident triangles are grouped by walking from triangle to triangle across the
//! edges whose dihedral is under the crease angle; each group beyond the first becomes a copy
//! of the vertex, appended after the mesh's own vertices with the group's normal. Nothing
//! else changes: the triangles keep their count and order, so a face index from a raycast
//! still names the same triangle, and the first `vertex_count` entries of every array are
//! still the mesh's own vertices in their own order, which is what every region, marking and
//! field is keyed by. Only the index buffer names copies, and `copy_of` says which vertex
//! each copy is.
//!
//! A scan is split too, by default. Noise tips its triangles against each other a few
//! degrees, seldom tens: a real scan of a few hundred thousand vertices has a few hundred
//! split from 30°, at its edges — the rim of a bore, a step — and its odd spike, and its
//! noise stays smooth. Where noise does pass the angle, splitting it speckles the surface and
//! multiplies the vertex list, so the operator says from what angle an edge is sharp — 50° or
//! 60° leaves a rough scan's noise smooth — or asks for Auto, which splits only a mesh that
//! looks like a tessellation (see `looks_tessellated`), or for none at all. And whatever they
//! say, a split that would add more than the budget is not made at all.
//!
//! Ported from ScanRuler (`src/core/geometry/crease.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::normals::compute_vertex_normals;
use super::union_find::find_root;
use super::wire_slots::wire_slots;
use super::MeshGraph;

/// What the operator asked for the scan: split always — the default — only when it looks
/// like CAD, or never. The reference is always split — it is CAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CreaseMode {
    Auto,
    #[default]
    On,
    Off,
}

/// The dihedral angle from which an edge counts as sharp, in degrees. The default of every
/// modelling tool's auto-smooth; a 12° step between the facets of a coarse cylinder stays
/// smooth, a 45° chamfer breaks.
pub const CREASE_ANGLE_DEG: f64 = 30.0;

/// The angles the operator may set it to, in degrees, and the step.
pub const CREASE_ANGLE_MIN: f64 = 5.0;
pub const CREASE_ANGLE_MAX: f64 = 120.0;
pub const CREASE_ANGLE_STEP: f64 = 5.0;

/// What the operator asked for the scan's sharp edges: when they are split, and from what
/// angle on an edge is sharp.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreaseSetting {
    pub mode: CreaseMode,
    pub angle_deg: f64,
}

/// The setting as the store holds it (its crease mode and angle), in one piece for a load.
pub fn crease_setting(crease_mode: CreaseMode, crease_angle: f64) -> CreaseSetting {
    CreaseSetting { mode: crease_mode, angle_deg: crease_angle }
}

/// A stored or typed angle as one the setting takes: in range, or the default when it is no
/// angle at all. A number is taken as it is, a string as JavaScript's `Number` reads it;
/// anything else, or a string that reads as no finite number, gives the default.
pub fn crease_angle_of(value: &serde_json::Value) -> f64 {
    let v = match value {
        serde_json::Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        serde_json::Value::String(s) if !s.trim().is_empty() => js_number_of(s),
        _ => f64::NAN,
    };
    if !v.is_finite() {
        return CREASE_ANGLE_DEG;
    }
    v.clamp(CREASE_ANGLE_MIN, CREASE_ANGLE_MAX)
}

/// A string read as JavaScript's `Number` reads it: trimmed, decimal (with an exponent, a
/// sign and `Infinity`), or an unsigned `0x`, `0o` or `0b` integer; NaN when it is none of
/// these.
fn js_number_of(s: &str) -> f64 {
    let t = s.trim();
    let radix = |digits: &str, base: u32| -> f64 {
        if digits.is_empty() {
            return f64::NAN;
        }
        digits.chars().try_fold(0.0f64, |acc, c| c.to_digit(base).map(|d| acc * f64::from(base) + f64::from(d))).unwrap_or(f64::NAN)
    };
    for (prefix, base) in [("0x", 16), ("0X", 16), ("0o", 8), ("0O", 8), ("0b", 2), ("0B", 2)] {
        if let Some(rest) = t.strip_prefix(prefix) {
            return radix(rest, base);
        }
    }
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    if body == "Infinity" {
        return if t.starts_with('-') { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    // Rust also reads "inf" and "nan", which JavaScript does not: only digits, a point and an
    // exponent are let through.
    if body.is_empty() || !body.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-')) {
        return f64::NAN;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

/// The share of a mesh's edges that must be dead flat for it to count as a tessellation —
/// see `looks_tessellated`.
const TESSELLATED_SHARE: f64 = 0.1;

/// An edge is flat when the faces either side of it agree to within this, in degrees.
/// Single-precision coordinates round a planar face's triangles a few millionths of a degree
/// apart; scanner noise tips adjacent triangles a tenth of a degree or more, even on the
/// cleanest scan of a flat.
const FLAT_DEG: f64 = 0.01;

/// How many copies a split may append before it is abandoned: never more than the mesh's own
/// vertex count, and any small mesh fits whole. A copy costs what a vertex costs in the
/// viewer — position, normal, colour, tint, paint, corner slot — so this bounds the split at
/// doubling the mesh.
pub fn crease_budget(vertex_count: usize) -> usize {
    vertex_count.max(250_000)
}

/// A mesh with its sharp edges split.
#[derive(Debug, Clone, PartialEq)]
pub struct CreaseSplit {
    /// The mesh's vertices followed by the copies.
    pub positions: Vec<DVec3>,
    /// Per vertex and copy, the normal of the side it is on.
    pub normals: Vec<DVec3>,
    /// The same triangles, with the corners on a crease pointing at copies.
    pub indices: Vec<u32>,
    /// The corner slots, each copy wearing its vertex's. A copy shares every triangle corner
    /// its vertex would have, so the colouring still holds.
    pub wire_slots: Vec<u8>,
    /// The vertex each copy stands in for: copy `vertex_count + k` is vertex `copy_of[k]`.
    /// Empty when nothing was split.
    pub copy_of: Vec<u32>,
}

/// Why no split was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CreaseSkip {
    /// The operator said never.
    Off,
    /// The mesh looked like a scan under Auto.
    Scan,
    /// The split would have gone past the budget.
    Budget,
}

/// What became of the split, for the status line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreaseReport {
    /// Copies appended to the vertex list; 0 when nothing needed splitting.
    pub added: usize,
    /// Why no split was made, or `None` when one was (`added` may still be 0).
    pub skipped: Option<CreaseSkip>,
}

/// Per-triangle normals, unnormalised: the cross product of two edges, whose length is twice
/// the area. In doubles — the flatness test looks for agreement to a hundredth of a degree,
/// closer than a float holds.
fn face_normals(positions: &[DVec3], indices: &[u32]) -> Vec<DVec3> {
    indices
        .as_chunks::<3>().0.iter()
        .map(|t| {
            let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
            let (abx, aby, abz) = (b.x - a.x, b.y - a.y, b.z - a.z);
            let (acx, acy, acz) = (c.x - a.x, c.y - a.y, c.z - a.z);
            DVec3::new(aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx)
        })
        .collect()
}

/// Vertex-to-triangle incidence in CSR form.
struct Fan {
    offsets: Vec<u32>,
    list: Vec<u32>,
}

fn vertex_triangles(indices: &[u32], vertex_count: usize) -> Fan {
    let mut offsets = vec![0u32; vertex_count + 1];
    for &i in indices {
        offsets[i as usize + 1] += 1;
    }
    for v in 0..vertex_count {
        offsets[v + 1] += offsets[v];
    }
    let mut list = vec![0u32; indices.len()];
    let mut cursor: Vec<u32> = offsets[..vertex_count].to_vec();
    for (i, &v) in indices.iter().enumerate() {
        let c = &mut cursor[v as usize];
        list[*c as usize] = (i / 3) as u32;
        *c += 1;
    }
    Fan { offsets, list }
}

/// The cosine of the angle between two face normals, or 1 when either face is a degenerate
/// sliver with no direction — a sliver joins whatever it touches rather than cutting a crease
/// through a flat.
fn cos_between(fnormals: &[DVec3], ti: u32, tj: u32) -> f64 {
    let (i, j) = (fnormals[ti as usize], fnormals[tj as usize]);
    let li = i.x * i.x + i.y * i.y + i.z * i.z;
    let lj = j.x * j.x + j.y * j.y + j.z * j.z;
    if li < 1e-30 || lj < 1e-30 {
        return 1.0;
    }
    (i.x * j.x + i.y * j.y + i.z * j.z) / (li * lj).sqrt()
}

/// Walk the edges around every vertex, calling `edge` once per pair of triangles that share
/// an edge with it — each interior edge is therefore seen twice, once from either end, which
/// nothing here minds. `stamp` and `first` are scratch over the vertices: a corner seen for
/// the first time around this vertex is remembered, and the next triangle around the same
/// vertex that reaches it shares the edge. `edge` gets the vertex and the two triangles' slots
/// in its fan.
fn for_each_fan_edge(indices: &[u32], fan: &Fan, vertex_count: usize, mut edge: impl FnMut(usize, usize, usize)) {
    let mut stamp = vec![0u32; vertex_count];
    let mut first = vec![0u32; vertex_count];
    for v in 0..vertex_count {
        let (begin, end) = (fan.offsets[v] as usize, fan.offsets[v + 1] as usize);
        if end - begin < 2 {
            continue;
        }
        let mark = v as u32 + 1;
        for k in begin..end {
            let t = fan.list[k] as usize * 3;
            for c in 0..3 {
                let w = indices[t + c] as usize;
                if w == v {
                    continue;
                }
                if stamp[w] == mark {
                    edge(v, first[w] as usize, k - begin);
                } else {
                    stamp[w] = mark;
                    first[w] = (k - begin) as u32;
                }
            }
        }
    }
}

/// The share of the mesh's edges whose two faces are dead flat against each other. A
/// tessellated CAD part is full of them — every planar face is a fan of coplanar triangles —
/// while a scan has practically none, because noise puts every edge at some small angle.
/// That is the one signal that tells the two apart without looking at the file name or the
/// triangle count, and it is what Auto goes on.
pub fn coplanar_share(positions: &[DVec3], indices: &[u32]) -> f64 {
    let vertex_count = positions.len();
    let fnormals = face_normals(positions, indices);
    let fan = vertex_triangles(indices, vertex_count);
    let flat = (FLAT_DEG * std::f64::consts::PI / 180.0).cos();
    let mut edges = 0usize;
    let mut coplanar = 0usize;
    for_each_fan_edge(indices, &fan, vertex_count, |v, fi, fj| {
        let begin = fan.offsets[v] as usize;
        edges += 1;
        if cos_between(&fnormals, fan.list[begin + fi], fan.list[begin + fj]) >= flat {
            coplanar += 1;
        }
    });
    if edges == 0 {
        0.0
    } else {
        coplanar as f64 / edges as f64
    }
}

/// Whether a mesh reads as a tessellation of exact surfaces rather than a scan — the call
/// Auto makes.
pub fn looks_tessellated(positions: &[DVec3], indices: &[u32]) -> bool {
    coplanar_share(positions, indices) >= TESSELLATED_SHARE
}

/// The slot of vertex `v` among the three corners of the triangle starting at `t`.
fn corner_of(indices: &[u32], t: usize, v: usize) -> usize {
    if indices[t] as usize == v {
        0
    } else if indices[t + 1] as usize == v {
        1
    } else {
        2
    }
}

/// Split the vertices on sharp edges, one copy per smooth side beyond the first. `None` when
/// the split would append more than `budget` copies — the mesh is then shown as it was,
/// smooth everywhere, rather than doubled. `angle_deg` is the dihedral from which an edge is
/// sharp (ScanRuler's defaults: `CREASE_ANGLE_DEG` and `crease_budget` of the vertex count).
pub fn split_creases(positions: &[DVec3], indices: &[u32], wire_slots: &[u8], angle_deg: f64, budget: usize) -> Option<CreaseSplit> {
    let vertex_count = positions.len();
    let fnormals = face_normals(positions, indices);
    let fan = vertex_triangles(indices, vertex_count);
    let smooth = (angle_deg * std::f64::consts::PI / 180.0).cos();
    let (offsets, list) = (&fan.offsets, &fan.list);

    // Union-find over the triangles around one vertex, scratch sized to the widest fan. Pairs
    // sharing an edge are joined when the edge is smooth.
    let mut parent = vec![0u32; 64];
    // Pass one: group the fans. The group of a corner is written into `group_of_corner` (per
    // corner, the root's fan slot) so the copies can be dealt out after the walk without
    // repeating it.
    let mut stamp = vec![0u32; vertex_count];
    let mut first = vec![0u32; vertex_count];
    let mut group_of_corner = vec![0u32; indices.len()];
    for v in 0..vertex_count {
        let (begin, end) = (offsets[v] as usize, offsets[v + 1] as usize);
        let k = end - begin;
        if k < 2 {
            continue;
        }
        if parent.len() < k {
            parent = vec![0u32; k * 2];
        }
        for (i, p) in parent.iter_mut().enumerate().take(k) {
            *p = i as u32;
        }
        let mark = v as u32 + 1;
        for i in 0..k {
            let t = list[begin + i] as usize * 3;
            for c in 0..3 {
                let w = indices[t + c] as usize;
                if w == v {
                    continue;
                }
                if stamp[w] == mark {
                    let j = first[w] as usize;
                    if cos_between(&fnormals, list[begin + i], list[begin + j]) >= smooth {
                        let ri = find_root(&mut parent, i);
                        let rj = find_root(&mut parent, j);
                        if ri != rj {
                            parent[ri] = rj as u32;
                        }
                    }
                } else {
                    stamp[w] = mark;
                    first[w] = i as u32;
                }
            }
        }
        for i in 0..k {
            let t = list[begin + i] as usize * 3;
            let c = corner_of(indices, t, v);
            group_of_corner[t + c] = find_root(&mut parent, i) as u32;
        }
    }

    // Pass two: deal out the copies. Around each vertex the group holding its first triangle
    // keeps the vertex; every other group gets a copy, in the order the groups are first met.
    let mut copies: Vec<u32> = Vec::new();
    let mut new_indices = indices.to_vec();
    let mut slots = vec![-1i64; 64];
    for v in 0..vertex_count {
        let (begin, end) = (offsets[v] as usize, offsets[v + 1] as usize);
        let k = end - begin;
        if k < 2 {
            continue;
        }
        if slots.len() < k {
            slots = vec![-1i64; k * 2];
        }
        slots[..k].fill(-1);
        let mut keep: i64 = -1;
        for i in 0..k {
            let t = list[begin + i] as usize * 3;
            let c = corner_of(indices, t, v);
            let g = group_of_corner[t + c] as usize;
            if keep < 0 {
                keep = g as i64;
            }
            if g as i64 == keep {
                continue;
            }
            if slots[g] < 0 {
                if copies.len() >= budget {
                    return None;
                }
                slots[g] = (vertex_count + copies.len()) as i64;
                copies.push(v as u32);
            }
            new_indices[t + c] = slots[g] as u32;
        }
    }

    let added = copies.len();
    let mut new_positions = Vec::with_capacity(vertex_count + added);
    new_positions.extend_from_slice(positions);
    let mut new_slots = Vec::with_capacity(vertex_count + added);
    new_slots.extend_from_slice(&wire_slots[..vertex_count.min(wire_slots.len())]);
    new_slots.resize(vertex_count, 0);
    for &v in &copies {
        new_positions.push(positions[v as usize]);
        new_slots.push(new_slots[v as usize]);
    }
    // With the sides parted, the plain per-vertex sum is each side's own normal — and the
    // mesh's winding was put outward when it was built, so the sums point the right way.
    let normals = compute_vertex_normals(&new_positions, &new_indices);
    Some(CreaseSplit { positions: new_positions, normals, indices: new_indices, wire_slots: new_slots, copy_of: copies })
}

/// The split a load makes, given what the operator asked for: `On` splits, `Off` does not,
/// and `Auto` splits only a mesh that looks tessellated — at edges from `angle_deg` on. `None`
/// with the reason when no split was made.
pub fn crease_for(mode: CreaseMode, positions: &[DVec3], indices: &[u32], wire_slots: &[u8], angle_deg: f64) -> (Option<CreaseSplit>, CreaseReport) {
    if mode == CreaseMode::Off {
        return (None, CreaseReport { added: 0, skipped: Some(CreaseSkip::Off) });
    }
    if mode == CreaseMode::Auto && !looks_tessellated(positions, indices) {
        return (None, CreaseReport { added: 0, skipped: Some(CreaseSkip::Scan) });
    }
    match split_creases(positions, indices, wire_slots, angle_deg, crease_budget(positions.len())) {
        None => (None, CreaseReport { added: 0, skipped: Some(CreaseSkip::Budget) }),
        Some(split) => {
            let added = split.copy_of.len();
            (Some(split), CreaseReport { added, skipped: None })
        }
    }
}

/// A scan's render arrays, laid out for the viewport: its own copies of the positions,
/// normals and indices, the corners the mesh mode draws its edges from, and whatever copies a
/// split for sharp edges appended.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanRender {
    pub positions: Vec<DVec3>,
    pub indices: Vec<u32>,
    pub normals: Vec<DVec3>,
    pub wire_slots: Vec<u8>,
    pub copy_of: Vec<u32>,
    pub crease: CreaseReport,
}

/// The render arrays of a welded mesh, its sharp edges split as `crease` asks — for a scan
/// when it loads and whenever the setting changes, and for any other mesh shown in a scan's
/// place.
pub fn scan_render(g: &MeshGraph, crease: CreaseSetting) -> ScanRender {
    let slots_own = wire_slots(&g.adj_offsets, &g.adj_list, g.vertex_count);
    let (split, report) = crease_for(crease.mode, &g.positions, &g.indices, &slots_own, crease.angle_deg);
    match split {
        Some(s) => ScanRender { positions: s.positions, indices: s.indices, normals: s.normals, wire_slots: s.wire_slots, copy_of: s.copy_of, crease: report },
        None => ScanRender {
            positions: g.positions.clone(),
            indices: g.indices.clone(),
            normals: g.normals.clone(),
            wire_slots: slots_own,
            copy_of: Vec::new(),
            crease: report,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::{box_mesh, cylinder_mesh, graph_indexed, graph_soup, icosphere};
    use crate::mesh_graph::wire_slots::wire_conflicts;
    use serde_json::json;

    /// A welded graph and its corner slots, as the worker hands them on.
    fn welded(soup: Vec<DVec3>) -> (MeshGraph, Vec<u8>) {
        let graph = graph_soup(soup);
        let slots = wire_slots(&graph.adj_offsets, &graph.adj_list, graph.vertex_count);
        (graph, slots)
    }

    fn split(g: &(MeshGraph, Vec<u8>)) -> CreaseSplit {
        split_creases(&g.0.positions, &g.0.indices, &g.1, CREASE_ANGLE_DEG, crease_budget(g.0.vertex_count)).expect("split refused")
    }

    /// The unit normal of a triangle as wound.
    fn face_normal(p: &[DVec3], a: u32, b: u32, c: u32) -> DVec3 {
        let (pa, pb, pc) = (p[a as usize], p[b as usize], p[c as usize]);
        let n = (pb - pa).cross(pc - pa);
        n / n.length()
    }

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn parts_a_cube_at_its_corners_and_edges_each_side_with_its_own_normal() {
        let g = welded(box_mesh(10.0, 4, 0.0, 13));
        let s = split(&g);
        let n = g.0.vertex_count;
        // Corners meet three faces (two copies each), edge vertices two (one copy), face
        // interiors stay whole: 8·2 + 12·3·1.
        assert_eq!(s.copy_of.len(), 8 * 2 + 12 * 3);
        assert_eq!(s.positions.len(), n + s.copy_of.len());
        assert_eq!(s.indices.len(), g.0.indices.len());
        // Every corner of every triangle now wears exactly that triangle's normal.
        for t in s.indices.as_chunks::<3>().0 {
            let fnm = face_normal(&s.positions, t[0], t[1], t[2]);
            for &v in t {
                assert!(close(s.normals[v as usize].dot(fnm), 1.0, 6));
            }
        }
        // A copy sits exactly where its vertex does and wears its corner slot.
        for (k, &v) in s.copy_of.iter().enumerate() {
            let d = n + k;
            assert!((v as usize) < n);
            assert_eq!(s.positions[d], s.positions[v as usize]);
            assert_eq!(s.wire_slots[d], g.1[v as usize]);
        }
        // The mesh's own vertices are untouched, in their order.
        assert_eq!(&s.positions[..n], &g.0.positions[..]);
        // And the mesh mode still draws every edge.
        assert_eq!(wire_conflicts(&s.indices, &s.wire_slots), 0);
    }

    #[test]
    fn leaves_a_smooth_surface_alone() {
        let graph = graph_indexed(icosphere(4, 10.0, 0.0, 7));
        let slots = wire_slots(&graph.adj_offsets, &graph.adj_list, graph.vertex_count);
        let g = (graph, slots);
        let s = split(&g);
        assert_eq!(s.copy_of.len(), 0);
        assert_eq!(s.normals, g.0.normals);
        assert_eq!(s.indices, g.0.indices);
    }

    #[test]
    fn parts_a_cylinder_only_along_its_rims_keeping_the_wall_smooth() {
        let radial = 48;
        let g = welded(cylinder_mesh(5.0, 20.0, radial, 6, 0.0, 11).0);
        let s = split(&g);
        // Each rim vertex is on the wall and on a cap: one copy each, 2 rims.
        assert_eq!(s.copy_of.len(), 2 * radial);
        let n = g.0.vertex_count;
        // Every copy is a cap normal or a wall normal — never a blend.
        for k in 0..s.copy_of.len() {
            let nv = s.normals[n + k];
            let axial = nv.z.abs();
            let radial_n = nv.x.hypot(nv.y);
            assert!(close(axial.max(radial_n), 1.0, 3));
        }
        // Wall triangles keep smooth shading: each corner's normal is radial, and
        // neighbouring corners around the wall differ by one facet.
        for t in s.indices.as_chunks::<3>().0 {
            let fnm = face_normal(&s.positions, t[0], t[1], t[2]);
            if fnm.z.abs() > 0.5 {
                continue; // a cap
            }
            for &v in t {
                let nv = s.normals[v as usize];
                assert!(nv.z.abs() < 1e-6);
                assert!(nv.dot(fnm) > (2.0 * std::f64::consts::PI / radial as f64).cos());
            }
        }
    }

    #[test]
    fn refuses_a_split_past_the_budget() {
        let g = welded(box_mesh(10.0, 4, 0.0, 13));
        assert!(split_creases(&g.0.positions, &g.0.indices, &g.1, 30.0, 10).is_none());
        assert!(split_creases(&g.0.positions, &g.0.indices, &g.1, 30.0, 52).is_some());
    }

    #[test]
    fn bounds_the_budget_at_doubling_the_mesh_and_lets_a_small_one_through_whole() {
        assert_eq!(crease_budget(100), 250_000);
        assert_eq!(crease_budget(2_000_000), 2_000_000);
    }

    #[test]
    fn sees_the_coplanar_triangles_of_a_cad_box() {
        let g = welded(box_mesh(10.0, 4, 0.0, 13));
        // Inside each face every edge is flat; only the twelve cube edges are not.
        assert!(coplanar_share(&g.0.positions, &g.0.indices) > 0.8);
        assert!(looks_tessellated(&g.0.positions, &g.0.indices));
    }

    #[test]
    fn sees_no_flats_on_a_noisy_scan_of_the_same_box_nor_on_a_ball() {
        let g = welded(box_mesh(10.0, 20, 0.02, 13));
        assert!(coplanar_share(&g.0.positions, &g.0.indices) < 0.05);
        assert!(!looks_tessellated(&g.0.positions, &g.0.indices));
        let (positions, faces) = icosphere(4, 10.0, 0.0, 7);
        let indices: Vec<u32> = faces.iter().flatten().copied().collect();
        assert_eq!(coplanar_share(&positions, &indices), 0.0);
    }

    #[test]
    fn under_auto_splits_the_box_and_leaves_the_scan_alone_on_and_off_do_as_told() {
        let cad = welded(box_mesh(10.0, 4, 0.0, 13));
        let scan = welded(box_mesh(10.0, 20, 0.05, 13));
        let auto = crease_for(CreaseMode::Auto, &cad.0.positions, &cad.0.indices, &cad.1, CREASE_ANGLE_DEG);
        assert!(auto.0.is_some());
        assert_eq!(auto.1, CreaseReport { added: 8 * 2 + 12 * 3, skipped: None });
        let kept = crease_for(CreaseMode::Auto, &scan.0.positions, &scan.0.indices, &scan.1, CREASE_ANGLE_DEG);
        assert!(kept.0.is_none());
        assert_eq!(kept.1, CreaseReport { added: 0, skipped: Some(CreaseSkip::Scan) });
        let forced = crease_for(CreaseMode::On, &scan.0.positions, &scan.0.indices, &scan.1, CREASE_ANGLE_DEG);
        assert!(forced.0.is_some());
        assert_eq!(forced.1.skipped, None);
        let off = crease_for(CreaseMode::Off, &cad.0.positions, &cad.0.indices, &cad.1, CREASE_ANGLE_DEG);
        assert!(off.0.is_none());
        assert_eq!(off.1, CreaseReport { added: 0, skipped: Some(CreaseSkip::Off) });
    }

    // A scan of a box: noise tips its triangles some degrees against each other everywhere,
    // its own edges stand at 90°. Twelve edges of 19 inner vertices, each parted once, and
    // eight corners, each into three.
    const EDGES: usize = 12 * 19 + 8 * 2;

    #[test]
    fn splits_a_scans_own_edges_and_leaves_its_noise_smooth_from_an_angle_the_noise_does_not_reach() {
        let scan = welded(box_mesh(10.0, 20, 0.05, 13));
        let at = |deg: f64| crease_for(CreaseMode::On, &scan.0.positions, &scan.0.indices, &scan.1, deg).1.added;
        assert_eq!(at(60.0), EDGES);
        assert!(at(30.0) >= EDGES);
        // Well past the box's own edges nothing is sharp.
        assert_eq!(at(120.0), 0);
    }

    #[test]
    fn lays_the_scan_out_at_the_angle_asked_for() {
        let scan = welded(box_mesh(10.0, 20, 0.05, 13));
        let r = scan_render(&scan.0, crease_setting(CreaseMode::On, 60.0));
        assert_eq!(r.copy_of.len(), EDGES);
        assert_eq!(r.positions.len(), scan.0.vertex_count + EDGES);
        assert_eq!(r.crease, CreaseReport { added: EDGES, skipped: None });
        let off = scan_render(&scan.0, CreaseSetting { mode: CreaseMode::Off, angle_deg: 60.0 });
        assert_eq!(off.copy_of.len(), 0);
        assert_eq!(off.indices, scan.0.indices);
    }

    #[test]
    fn takes_a_stored_angle_in_range_and_the_default_for_anything_that_is_not_one() {
        assert_eq!(crease_angle_of(&json!(null)), 30.0);
        assert_eq!(crease_angle_of(&json!("")), 30.0);
        assert_eq!(crease_angle_of(&json!("abc")), 30.0);
        assert_eq!(crease_angle_of(&json!("45")), 45.0);
        assert_eq!(crease_angle_of(&json!(60)), 60.0);
        assert_eq!(crease_angle_of(&json!(1)), 5.0);
        assert_eq!(crease_angle_of(&json!(500)), 120.0);
    }
}
