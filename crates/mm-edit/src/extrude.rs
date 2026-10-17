//! Selection extrude / offset (Meshmixer *Edit > Extrude*, *Offset*) and a naive
//! whole-surface offset.

use crate::util::*;
use crate::{EditError, EditResult};
use mm_core::{DVec3, Mesh, INVALID};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Direction in which a selection is extruded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExtrudeDirection {
    /// Per-vertex average of the selected faces' normals.
    Normal,
    /// A fixed direction (normalised; the magnitude is given by `distance`).
    Fixed(DVec3),
}

/// Parameters for [`extrude_selection`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtrudeParams {
    /// Offset distance (negative extrudes inwards).
    pub distance: f64,
    /// Offset direction.
    pub direction: ExtrudeDirection,
    /// Rows of side-wall quads (≥ 1).
    pub side_segments: usize,
    /// Give the side walls their own face group (a hard crease for group-aware tools).
    pub harden_sides: bool,
}

impl Default for ExtrudeParams {
    fn default() -> Self {
        Self { distance: 1.0, direction: ExtrudeDirection::Normal, side_segments: 1, harden_sides: true }
    }
}

/// Output of [`extrude_selection`].
#[derive(Clone, Debug, Default)]
pub struct ExtrudeResult {
    /// The offset region (input triangles; some ids may have changed).
    pub new_region_tris: Vec<u32>,
    /// Side-wall triangles joining the old and new region borders.
    pub side_tris: Vec<u32>,
    /// (old border vertex, its moved copy).
    pub border_vertex_map: Vec<(u32, u32)>,
    /// Face group of the side walls.
    pub side_group: i32,
}

/// Extrude the selected faces: the region is offset and side walls are stitched between
/// its old border (still attached to the unselected surface) and its new border. Works for
/// regions with several border loops and for regions touching the mesh boundary (there the
/// region's open edges simply move along and the wall strips end at the boundary).
pub fn extrude_selection(mesh: &mut Mesh, tris: &[u32], params: &ExtrudeParams) -> EditResult<ExtrudeResult> {
    let region = valid_unique_tris(mesh, tris);
    if region.is_empty() {
        return Err(EditError::EmptySelection);
    }
    if !(params.distance.is_finite() && params.distance.abs() > 1e-300) {
        return Err(EditError::InvalidParameter("extrude distance must be non-zero".into()));
    }
    let fixed = match params.direction {
        ExtrudeDirection::Fixed(d) => {
            if !(d.length() > 1e-300 && d.is_finite()) {
                return Err(EditError::InvalidParameter("extrude direction must be non-zero".into()));
            }
            Some(d.normalize())
        }
        ExtrudeDirection::Normal => None,
    };
    let segs = params.side_segments.max(1);
    let region_group = mesh.tri_group(region[0]);
    let mut in_r = vec![false; mesh.max_tid()];
    for &t in &region {
        in_r[t as usize] = true;
    }

    // Border edges between region and the rest, oriented as in the region triangle.
    let mut border: Vec<(u32, u32)> = Vec::new();
    for &t in &region {
        let tv = mesh.triangle(t);
        let te = mesh.tri_edges(t);
        for j in 0..3 {
            let o = mesh.edge_other_triangle(te[j], t);
            if o != INVALID && !in_r[o as usize] {
                border.push((tv[j], tv[(j + 1) % 3]));
            }
        }
    }

    // Displacement per region vertex.
    let mut disp: HashMap<u32, DVec3> = HashMap::new();
    for &t in &region {
        let c = mesh.tri_cross(t);
        for v in mesh.triangle(t) {
            *disp.entry(v).or_insert(DVec3::ZERO) += c;
        }
    }
    for d in disp.values_mut() {
        let dir = fixed.unwrap_or_else(|| d.normalize_or_zero());
        *d = dir * params.distance;
    }

    let mut is_border: HashMap<u32, u32> = HashMap::new();
    for &(a, b) in &border {
        is_border.insert(a, INVALID);
        is_border.insert(b, INVALID);
    }
    let mut verts: Vec<u32> = disp.keys().copied().collect();
    verts.sort_unstable();
    for &v in &verts {
        if !is_border.contains_key(&v) {
            let p = mesh.vertex(v) + disp[&v];
            mesh.set_vertex(v, p);
        }
    }

    // Re-attach region triangles at border vertices to moved copies.
    let mut border_verts: Vec<u32> = is_border.keys().copied().collect();
    border_verts.sort_unstable();
    for &v in &border_verts {
        let nv = mesh.append_vertex(mesh.vertex(v) + disp[&v]);
        is_border.insert(v, nv);
    }
    let mut affected: Vec<u32> =
        region.iter().copied().filter(|&t| mesh.triangle(t).iter().any(|v| is_border.contains_key(v))).collect();
    affected.sort_unstable();
    let recs: Vec<([u32; 3], i32)> = affected.iter().map(|&t| (mesh.triangle(t), mesh.tri_group(t))).collect();
    for &t in &affected {
        mesh.remove_triangle(t, false)?;
    }
    let mut new_region: Vec<u32> = region.iter().copied().filter(|t| affected.binary_search(t).is_err()).collect();
    for (tv, g) in recs {
        let m = tv.map(|v| is_border.get(&v).copied().unwrap_or(v));
        new_region.push(mesh.append_triangle(m, g)?);
    }

    // Side walls with `segs` rows.
    let side_group = if params.harden_sides { new_group(mesh) } else { region_group };
    let mut chain: HashMap<u32, Vec<u32>> = HashMap::new();
    for &v in &border_verts {
        let top = is_border[&v];
        let (p0, p1) = (mesh.vertex(v), mesh.vertex(top));
        let mut c = vec![v];
        for k in 1..segs {
            c.push(mesh.append_vertex(p0.lerp(p1, k as f64 / segs as f64)));
        }
        c.push(top);
        chain.insert(v, c);
    }
    let mut side = Vec::with_capacity(border.len() * 2 * segs);
    for &(a, b) in &border {
        let (ca, cb) = (&chain[&a], &chain[&b]);
        for k in 0..segs {
            let (a0, b0, a1, b1) = (ca[k], cb[k], ca[k + 1], cb[k + 1]);
            side.push(mesh.append_triangle([a0, b0, b1], side_group)?);
            side.push(mesh.append_triangle([a0, b1, a1], side_group)?);
        }
    }
    Ok(ExtrudeResult {
        new_region_tris: new_region,
        side_tris: side,
        border_vertex_map: border_verts.iter().map(|&v| (v, is_border[&v])).collect(),
        side_group,
    })
}

#[derive(PartialEq)]
struct HeapItem(f64, u32);
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Move the selected region's vertices along their normals by `distance`, keeping the
/// selection border fixed. With `falloff_width > 0` the offset ramps smoothly from 0 at
/// the border to full at that geodesic distance; otherwise every non-border vertex moves
/// the full distance. A selection without border (a whole closed component) moves
/// uniformly. Returns the number of vertices moved.
pub fn offset_selection(mesh: &mut Mesh, tris: &[u32], distance: f64, falloff_width: f64) -> EditResult<usize> {
    let region = valid_unique_tris(mesh, tris);
    if region.is_empty() {
        return Err(EditError::EmptySelection);
    }
    let mut in_r = vec![false; mesh.max_tid()];
    for &t in &region {
        in_r[t as usize] = true;
    }
    let mut verts: Vec<u32> = region.iter().flat_map(|&t| mesh.triangle(t)).collect();
    verts.sort_unstable();
    verts.dedup();
    let border: Vec<u32> = verts
        .iter()
        .copied()
        .filter(|&v| mesh.vertex_triangles(v).iter().any(|&t| !in_r[t as usize]))
        .collect();
    let mut weight: HashMap<u32, f64> = verts.iter().map(|&v| (v, 1.0)).collect();
    if !border.is_empty() {
        if falloff_width > 0.0 {
            let mut dist: HashMap<u32, f64> = HashMap::new();
            let mut heap = BinaryHeap::new();
            for &b in &border {
                dist.insert(b, 0.0);
                heap.push(HeapItem(0.0, b));
            }
            while let Some(HeapItem(d, v)) = heap.pop() {
                if d > dist[&v] || d > falloff_width {
                    continue;
                }
                for &e in mesh.vertex_edges(v) {
                    let [t0, t1] = mesh.edge_triangles(e);
                    if !(in_r[t0 as usize] || (t1 != INVALID && in_r[t1 as usize])) {
                        continue;
                    }
                    let w = mesh.edge_other_vertex(e, v);
                    let nd = d + mesh.edge_length(e);
                    if dist.get(&w).is_none_or(|&x| nd < x) {
                        dist.insert(w, nd);
                        heap.push(HeapItem(nd, w));
                    }
                }
            }
            for &v in &verts {
                let d = dist.get(&v).copied().unwrap_or(f64::INFINITY);
                weight.insert(v, smoothstep(d / falloff_width));
            }
        } else {
            for &b in &border {
                weight.insert(b, 0.0);
            }
        }
    }
    let moves: Vec<(u32, DVec3)> = verts
        .iter()
        .filter(|&&v| weight[&v] > 0.0)
        .map(|&v| (v, mesh.vertex(v) + mesh.vertex_normal(v) * (distance * weight[&v])))
        .collect();
    for &(v, p) in &moves {
        mesh.set_vertex(v, p);
    }
    Ok(moves.len())
}

/// Move every vertex along its (area-weighted) normal by `distance`. Naive: may
/// self-intersect where |distance| exceeds the local feature size.
pub fn offset_surface(mesh: &mut Mesh, distance: f64) {
    let normals = mesh.compute_vertex_normals();
    let ids: Vec<u32> = mesh.vertex_ids().collect();
    let pos = mesh.positions_mut();
    for v in ids {
        pos[v as usize] += normals[v as usize] * distance;
    }
    mesh.bump_shape_version();
}
