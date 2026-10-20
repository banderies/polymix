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
    /// One direction for the whole region: its average normal (Meshmixer "Constant").
    Average,
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
    /// Flatten the moved region onto one plane across the extrusion direction (Meshmixer
    /// EndType "Flat"); otherwise it keeps its shape ("Offset").
    pub flat_end: bool,
    /// Give the moved region a new face group instead of keeping its groups.
    pub new_region_group: bool,
}

impl Default for ExtrudeParams {
    fn default() -> Self {
        Self { distance: 1.0, direction: ExtrudeDirection::Normal, side_segments: 1, harden_sides: true, flat_end: false, new_region_group: false }
    }
}

/// Unit direction per region vertex for `dir` (`None`: invalid fixed direction).
fn region_directions(mesh: &Mesh, region: &[u32], dir: ExtrudeDirection) -> Option<HashMap<u32, DVec3>> {
    let mut acc: HashMap<u32, DVec3> = HashMap::new();
    let mut total = DVec3::ZERO;
    for &t in region {
        let c = mesh.tri_cross(t);
        total += c;
        for v in mesh.triangle(t) {
            *acc.entry(v).or_insert(DVec3::ZERO) += c;
        }
    }
    let fixed = match dir {
        ExtrudeDirection::Fixed(d) => {
            if !(d.length() > 1e-300 && d.is_finite()) {
                return None;
            }
            Some(d.normalize())
        }
        ExtrudeDirection::Average => Some(total.normalize_or_zero()),
        ExtrudeDirection::Normal => None,
    };
    for d in acc.values_mut() {
        *d = fixed.unwrap_or_else(|| d.normalize_or_zero());
    }
    Some(acc)
}

/// Flatten `pts` onto the plane through their mean, perpendicular to `n`.
fn flatten(pts: &mut [(u32, DVec3)], n: DVec3) {
    let n = n.normalize_or_zero();
    if pts.is_empty() || n == DVec3::ZERO {
        return;
    }
    let c = pts.iter().map(|p| p.1).sum::<DVec3>() / pts.len() as f64;
    for p in pts.iter_mut() {
        p.1 -= n * (p.1 - c).dot(n);
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

    // Displacement per region vertex (flattened end: onto one plane across the mean
    // direction).
    let dirs = region_directions(mesh, &region, params.direction).ok_or_else(|| EditError::InvalidParameter("extrude direction must be non-zero".into()))?;
    let mut disp: HashMap<u32, DVec3> = dirs.iter().map(|(&v, &d)| (v, d * params.distance)).collect();
    if params.flat_end {
        let mean_dir: DVec3 = dirs.values().copied().sum();
        let mut moved: Vec<(u32, DVec3)> = disp.iter().map(|(&v, &d)| (v, mesh.vertex(v) + d)).collect();
        // Plane at the farthest moved point, so the end sits at the full extrusion distance.
        let n = mean_dir.normalize_or_zero();
        let far = moved.iter().map(|p| p.1.dot(n) * params.distance.signum()).fold(f64::MIN, f64::max) * params.distance.signum();
        flatten(&mut moved, n);
        for (v, p) in moved {
            let p = p + n * (far - p.dot(n));
            disp.insert(v, p - mesh.vertex(v));
        }
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
    if params.new_region_group {
        let g = new_group(mesh);
        for &t in &new_region {
            mesh.set_tri_group(t, g);
        }
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

/// Extract (Meshmixer *Edit > Extract*, Shift+D): a new closed solid made from the selected
/// faces thickened by `distance` (top = the faces moved along `direction`, bottom = the
/// original faces, side walls along the patch border). The input mesh is not changed.
pub fn extract_solid(mesh: &Mesh, tris: &[u32], distance: f64, direction: ExtrudeDirection, flat_end: bool) -> EditResult<Mesh> {
    let region = valid_unique_tris(mesh, tris);
    if region.is_empty() {
        return Err(EditError::EmptySelection);
    }
    if !(distance.is_finite() && distance.abs() > 1e-300) {
        return Err(EditError::InvalidParameter("extract distance must be non-zero".into()));
    }
    let (patch, _) = mesh.submesh(&region);
    let all: Vec<u32> = patch.triangle_ids().collect();
    let dirs = region_directions(&patch, &all, direction).ok_or_else(|| EditError::InvalidParameter("extract direction must be non-zero".into()))?;
    let mut top: Vec<(u32, DVec3)> = dirs.iter().map(|(&v, &d)| (v, patch.vertex(v) + d * distance)).collect();
    if flat_end {
        let n: DVec3 = dirs.values().copied().sum::<DVec3>().normalize_or_zero();
        let far = top.iter().map(|p| p.1.dot(n) * distance.signum()).fold(f64::MIN, f64::max) * distance.signum();
        flatten(&mut top, n);
        for p in top.iter_mut() {
            p.1 += n * (far - p.1.dot(n));
        }
    }
    let top: HashMap<u32, DVec3> = top.into_iter().collect();
    let mut out = Mesh::new();
    let mut bottom_id: HashMap<u32, u32> = HashMap::new();
    let mut top_id: HashMap<u32, u32> = HashMap::new();
    for v in patch.vertex_ids() {
        bottom_id.insert(v, out.append_vertex(patch.vertex(v)));
        top_id.insert(v, out.append_vertex(top.get(&v).copied().unwrap_or(patch.vertex(v))));
    }
    for t in patch.triangle_ids() {
        let [a, b, c] = patch.triangle(t);
        let g = patch.tri_group(t);
        out.append_triangle([top_id[&a], top_id[&b], top_id[&c]], g)?;
        out.append_triangle([bottom_id[&a], bottom_id[&c], bottom_id[&b]], g)?;
    }
    // Walls: for a patch-boundary edge a→b (as in its triangle), (a, b, b') and (a, b', a').
    for e in patch.boundary_edge_ids() {
        let [a, b] = patch.oriented_edge_vertices(e);
        let (a0, b0, a1, b1) = (bottom_id[&a], bottom_id[&b], top_id[&a], top_id[&b]);
        out.append_triangle([a0, b0, b1], 0)?;
        out.append_triangle([a0, b1, a1], 0)?;
    }
    // Inward extracts come out inside-out; orient to positive volume.
    mm_core::topology::orient_consistently(&mut out);
    Ok(out)
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
    offset_selection_with(mesh, tris, distance, falloff_width, false)
}

/// [`offset_selection`]; with `keep_mesh_boundary`, open-boundary vertices of the region stay
/// put too (and the soft transition ramps away from them).
pub fn offset_selection_with(mesh: &mut Mesh, tris: &[u32], distance: f64, falloff_width: f64, keep_mesh_boundary: bool) -> EditResult<usize> {
    let (verts, weight) = offset_weights(mesh, tris, falloff_width, keep_mesh_boundary)?;
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

/// [`offset_selection_with`], then each moved vertex corrected `iterations` times towards its
/// intended distance (`distance` × its transition weight) from the region as it was: moving
/// along the normals alone crowds the surface in concave places and thins it over sharp convex
/// ones, where a true offset keeps the distance everywhere.
pub fn offset_selection_exact(mesh: &mut Mesh, tris: &[u32], distance: f64, falloff_width: f64, keep_mesh_boundary: bool, iterations: usize) -> EditResult<usize> {
    let (verts, weight) = offset_weights(mesh, tris, falloff_width, keep_mesh_boundary)?;
    let original = mesh.clone();
    let region = valid_unique_tris(mesh, tris);
    let n = offset_selection_with(mesh, tris, distance, falloff_width, keep_mesh_boundary)?;
    if iterations == 0 || distance == 0.0 {
        return Ok(n);
    }
    let bvh = mm_core::bvh::Bvh::build_subset(&original, &region);
    let moved: Vec<u32> = verts.into_iter().filter(|v| weight[v] > 0.0).collect();
    for _ in 0..iterations {
        let mut change = 0f64;
        for &v in &moved {
            let target = distance * weight[&v];
            let p = mesh.vertex(v);
            let Some(hit) = bvh.nearest(&original, p, f64::INFINITY) else { continue };
            let side = original.tri_normal(hit.tid) * target.signum();
            let d = p - hit.point;
            // Away from the nearest point, on the side the offset goes (the normal there if the
            // vertex has crossed to the other side or sits on the surface).
            let dir = if d.length() > 1e-12 * (1.0 + target.abs()) && d.dot(side) > 0.0 { d.normalize() } else { side };
            let q = hit.point + dir * target.abs();
            change = change.max(q.distance(p));
            mesh.set_vertex(v, q);
        }
        if change < 1e-6 * distance.abs() {
            break;
        }
    }
    Ok(n)
}

/// The region's vertices and how far each moves (0 on the fixed border, ramping to 1 over
/// `falloff_width`).
fn offset_weights(mesh: &Mesh, tris: &[u32], falloff_width: f64, keep_mesh_boundary: bool) -> EditResult<(Vec<u32>, HashMap<u32, f64>)> {
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
        .filter(|&v| mesh.vertex_triangles(v).iter().any(|&t| !in_r[t as usize]) || (keep_mesh_boundary && mesh.is_boundary_vertex(v)))
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
    Ok((verts, weight))
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
