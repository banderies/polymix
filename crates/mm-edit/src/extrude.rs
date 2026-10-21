//! Selection extrude / offset (Meshmixer *Edit > Extrude*, *Offset*) and a naive
//! whole-surface offset.

use crate::util::*;
use crate::{EditError, EditResult};
use mm_core::{DVec3, Mesh, INVALID};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

/// A region's vertices (sorted, unique) and each one's place among them, by vertex id
/// (`u32::MAX` for vertices outside the region): arrays parallel to the vertices then stand in
/// for maps keyed by vertex.
struct RegionVerts {
    verts: Vec<u32>,
    slot: Vec<u32>,
}

impl RegionVerts {
    fn new(mesh: &Mesh, region: &[u32]) -> Self {
        let mut verts: Vec<u32> = region.iter().flat_map(|&t| mesh.triangle(t)).collect();
        verts.par_sort_unstable();
        verts.dedup();
        let mut slot = vec![u32::MAX; mesh.max_vid()];
        for (i, &v) in verts.iter().enumerate() {
            slot[v as usize] = i as u32;
        }
        Self { verts, slot }
    }

    /// The place of region vertex `v`.
    #[inline]
    fn at(&self, v: u32) -> usize {
        self.slot[v as usize] as usize
    }
}

/// Unit direction per region vertex for `dir`, parallel to the region's vertices (`None`:
/// invalid fixed direction).
fn region_directions(mesh: &Mesh, region: &[u32], dir: ExtrudeDirection) -> Option<(RegionVerts, Vec<DVec3>)> {
    let rv = RegionVerts::new(mesh, region);
    let mut acc = vec![DVec3::ZERO; rv.verts.len()];
    let mut total = DVec3::ZERO;
    for &t in region {
        let c = mesh.tri_cross(t);
        total += c;
        for v in mesh.triangle(t) {
            acc[rv.at(v)] += c;
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
    for d in acc.iter_mut() {
        *d = fixed.unwrap_or_else(|| d.normalize_or_zero());
    }
    Some((rv, acc))
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

/// Move each vertex to its new place, all with one new shape version (nothing changes when
/// there are no moves).
fn place(mesh: &mut Mesh, moves: impl ExactSizeIterator<Item = (u32, DVec3)>) {
    if moves.len() == 0 {
        return;
    }
    let pos = mesh.positions_mut();
    for (v, p) in moves {
        pos[v as usize] = p;
    }
    mesh.bump_shape_version();
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
    // direction), parallel to the region's vertices.
    let (rv, dirs) = region_directions(mesh, &region, params.direction).ok_or_else(|| EditError::InvalidParameter("extrude direction must be non-zero".into()))?;
    let verts = &rv.verts;
    let mut disp: Vec<DVec3> = dirs.iter().map(|&d| d * params.distance).collect();
    if params.flat_end {
        // (Summed in vertex order: the same every run.)
        let mean_dir: DVec3 = dirs.iter().copied().sum();
        let mut moved: Vec<(u32, DVec3)> = verts.iter().zip(&disp).map(|(&v, &d)| (v, mesh.vertex(v) + d)).collect();
        // Plane at the farthest moved point, so the end sits at the full extrusion distance.
        let n = mean_dir.normalize_or_zero();
        let far = moved.iter().map(|p| p.1.dot(n) * params.distance.signum()).fold(f64::MIN, f64::max) * params.distance.signum();
        flatten(&mut moved, n);
        for (i, (v, p)) in moved.into_iter().enumerate() {
            let p = p + n * (far - p.dot(n));
            disp[i] = p - mesh.vertex(v);
        }
    }

    let mut on_border = vec![false; verts.len()];
    for &(a, b) in &border {
        on_border[rv.at(a)] = true;
        on_border[rv.at(b)] = true;
    }
    let inner: Vec<(u32, DVec3)> = verts.iter().enumerate().filter(|&(i, _)| !on_border[i]).map(|(i, &v)| (v, mesh.vertex(v) + disp[i])).collect();
    place(mesh, inner.into_iter());

    // Re-attach region triangles at border vertices to moved copies.
    let border_verts: Vec<u32> = verts.iter().enumerate().filter(|&(i, _)| on_border[i]).map(|(_, &v)| v).collect();
    let mut copy = vec![INVALID; verts.len()];
    for &v in &border_verts {
        let i = rv.at(v);
        copy[i] = mesh.append_vertex(mesh.vertex(v) + disp[i]);
    }
    let moved_to = |v: u32| {
        let i = rv.at(v);
        if on_border[i] { copy[i] } else { v }
    };
    let mut affected: Vec<u32> = region.iter().copied().filter(|&t| mesh.triangle(t).iter().any(|&v| on_border[rv.at(v)])).collect();
    affected.sort_unstable();
    let recs: Vec<([u32; 3], i32)> = affected.iter().map(|&t| (mesh.triangle(t), mesh.tri_group(t))).collect();
    for &t in &affected {
        mesh.remove_triangle(t, false)?;
    }
    let mut new_region: Vec<u32> = region.iter().copied().filter(|t| affected.binary_search(t).is_err()).collect();
    for (tv, g) in recs {
        new_region.push(mesh.append_triangle(tv.map(moved_to), g)?);
    }
    if params.new_region_group {
        let g = new_group(mesh);
        for &t in &new_region {
            mesh.set_tri_group(t, g);
        }
    }

    // Side walls with `segs` rows: a chain of vertices up from each border vertex.
    let side_group = if params.harden_sides { new_group(mesh) } else { region_group };
    let mut chain_of = vec![usize::MAX; verts.len()];
    let mut chains: Vec<Vec<u32>> = Vec::with_capacity(border_verts.len());
    for &v in &border_verts {
        let i = rv.at(v);
        let top = copy[i];
        let (p0, p1) = (mesh.vertex(v), mesh.vertex(top));
        let mut c = vec![v];
        for k in 1..segs {
            c.push(mesh.append_vertex(p0.lerp(p1, k as f64 / segs as f64)));
        }
        c.push(top);
        chain_of[i] = chains.len();
        chains.push(c);
    }
    let mut side = Vec::with_capacity(border.len() * 2 * segs);
    for &(a, b) in &border {
        let (ca, cb) = (&chains[chain_of[rv.at(a)]], &chains[chain_of[rv.at(b)]]);
        for k in 0..segs {
            let (a0, b0, a1, b1) = (ca[k], cb[k], ca[k + 1], cb[k + 1]);
            side.push(mesh.append_triangle([a0, b0, b1], side_group)?);
            side.push(mesh.append_triangle([a0, b1, a1], side_group)?);
        }
    }
    Ok(ExtrudeResult {
        new_region_tris: new_region,
        side_tris: side,
        border_vertex_map: border_verts.iter().map(|&v| (v, copy[rv.at(v)])).collect(),
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
    let (rv, dirs) = region_directions(&patch, &all, direction).ok_or_else(|| EditError::InvalidParameter("extract direction must be non-zero".into()))?;
    let mut top: Vec<(u32, DVec3)> = rv.verts.iter().zip(&dirs).map(|(&v, &d)| (v, patch.vertex(v) + d * distance)).collect();
    if flat_end {
        // (Summed in vertex order: the same every run.)
        let n: DVec3 = dirs.iter().copied().sum::<DVec3>().normalize_or_zero();
        let far = top.iter().map(|p| p.1.dot(n) * distance.signum()).fold(f64::MIN, f64::max) * distance.signum();
        flatten(&mut top, n);
        for p in top.iter_mut() {
            p.1 += n * (far - p.1.dot(n));
        }
    }
    let mut out = Mesh::new();
    let mut bottom_id = vec![INVALID; patch.max_vid()];
    let mut top_id = vec![INVALID; patch.max_vid()];
    for v in patch.vertex_ids() {
        bottom_id[v as usize] = out.append_vertex(patch.vertex(v));
        let raised = match rv.slot[v as usize] {
            u32::MAX => patch.vertex(v),
            i => top[i as usize].1,
        };
        top_id[v as usize] = out.append_vertex(raised);
    }
    for t in patch.triangle_ids() {
        let [a, b, c] = patch.triangle(t);
        let g = patch.tri_group(t);
        out.append_triangle([top_id[a as usize], top_id[b as usize], top_id[c as usize]], g)?;
        out.append_triangle([bottom_id[a as usize], bottom_id[c as usize], bottom_id[b as usize]], g)?;
    }
    // Walls: for a patch-boundary edge a→b (as in its triangle), (a, b, b') and (a, b', a').
    for e in patch.boundary_edge_ids() {
        let [a, b] = patch.oriented_edge_vertices(e);
        let (a0, b0, a1, b1) = (bottom_id[a as usize], bottom_id[b as usize], top_id[a as usize], top_id[b as usize]);
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
    let (rv, weight) = offset_weights(mesh, tris, falloff_width, keep_mesh_boundary)?;
    let moves = offset_moves(mesh, &rv.verts, &weight, distance);
    let n = moves.len();
    place(mesh, moves.into_iter().map(|(v, p, _)| (v, p)));
    Ok(n)
}

/// Each vertex with a weight over 0 and where the offset along its normal takes it (all
/// from the mesh as it is), with its weight.
fn offset_moves(mesh: &Mesh, verts: &[u32], weight: &[f64], distance: f64) -> Vec<(u32, DVec3, f64)> {
    verts.par_iter().zip(weight).filter(|&(_, &w)| w > 0.0).map(|(&v, &w)| (v, mesh.vertex(v) + mesh.vertex_normal(v) * (distance * w), w)).collect()
}

/// [`offset_selection_with`], then each moved vertex corrected `iterations` times towards its
/// intended distance (`distance` × its transition weight) from the region as it was: moving
/// along the normals alone crowds the surface in concave places and thins it over sharp convex
/// ones, where a true offset keeps the distance everywhere.
pub fn offset_selection_exact(mesh: &mut Mesh, tris: &[u32], distance: f64, falloff_width: f64, keep_mesh_boundary: bool, iterations: usize) -> EditResult<usize> {
    offset_selection_exact_with(mesh, tris, distance, falloff_width, keep_mesh_boundary, iterations, None)
}

/// [`offset_selection_exact`], reporting progress (0..1) to `progress`, which returns `false`
/// to cancel: the correction then stops before its next pass with the error "cancelled", and
/// `mesh` is left as it was.
///
/// The moved vertices are kept apart from the mesh until the end, so the mesh itself is the
/// region as it was (no copy of it is needed), and each pass corrects every vertex at once
/// (in parallel): a vertex's correction depends only on its own place and the region as it
/// was, so this is what correcting them one by one gave.
pub fn offset_selection_exact_with(
    mesh: &mut Mesh,
    tris: &[u32],
    distance: f64,
    falloff_width: f64,
    keep_mesh_boundary: bool,
    iterations: usize,
    progress: Option<&dyn Fn(f32) -> bool>,
) -> EditResult<usize> {
    let going = |x: f32| progress.is_none_or(|p| p(x));
    let cancelled = || EditError::Failed("cancelled".into());
    let (rv, weight) = offset_weights(mesh, tris, falloff_width, keep_mesh_boundary)?;
    if !going(0.0) {
        return Err(cancelled());
    }
    let moves = offset_moves(mesh, &rv.verts, &weight, distance);
    let n = moves.len();
    if iterations == 0 || distance == 0.0 {
        place(mesh, moves.into_iter().map(|(v, p, _)| (v, p)));
        return Ok(n);
    }
    let region = valid_unique_tris(mesh, tris);
    let bvh = mm_core::bvh::Bvh::build_subset(mesh, &region);
    let mut at: Vec<DVec3> = moves.iter().map(|m| m.1).collect();
    for it in 0..iterations {
        if !going(it as f32 / iterations as f32) {
            return Err(cancelled());
        }
        let next: Vec<Option<DVec3>> = moves
            .par_iter()
            .zip(&at)
            .map(|(&(_, _, w), &p)| {
                let target = distance * w;
                let hit = bvh.nearest(mesh, p, f64::INFINITY)?;
                let side = mesh.tri_normal(hit.tid) * target.signum();
                let d = p - hit.point;
                // Away from the nearest point, on the side the offset goes (the normal there if the
                // vertex has crossed to the other side or sits on the surface).
                let dir = if d.length() > 1e-12 * (1.0 + target.abs()) && d.dot(side) > 0.0 { d.normalize() } else { side };
                Some(hit.point + dir * target.abs())
            })
            .collect();
        let mut change = 0f64;
        for (p, q) in at.iter_mut().zip(next) {
            if let Some(q) = q {
                change = change.max(q.distance(*p));
                *p = q;
            }
        }
        if change < 1e-6 * distance.abs() {
            break;
        }
    }
    place(mesh, moves.iter().zip(at).map(|(m, p)| (m.0, p)));
    Ok(n)
}

/// The region's vertices and how far each moves (0 on the fixed border, ramping to 1 over
/// `falloff_width`), parallel to them.
fn offset_weights(mesh: &Mesh, tris: &[u32], falloff_width: f64, keep_mesh_boundary: bool) -> EditResult<(RegionVerts, Vec<f64>)> {
    let region = valid_unique_tris(mesh, tris);
    if region.is_empty() {
        return Err(EditError::EmptySelection);
    }
    let mut in_r = vec![false; mesh.max_tid()];
    for &t in &region {
        in_r[t as usize] = true;
    }
    let rv = RegionVerts::new(mesh, &region);
    let on_border: Vec<bool> = rv
        .verts
        .par_iter()
        .map(|&v| mesh.vertex_triangles(v).iter().any(|&t| !in_r[t as usize]) || (keep_mesh_boundary && mesh.is_boundary_vertex(v)))
        .collect();
    let border: Vec<u32> = rv.verts.iter().zip(&on_border).filter(|&(_, &b)| b).map(|(&v, _)| v).collect();
    let mut weight = vec![1.0; rv.verts.len()];
    if !border.is_empty() {
        if falloff_width > 0.0 {
            // Geodesic distance from the border over the region (by the region's vertices).
            let mut dist: Vec<Option<f64>> = vec![None; rv.verts.len()];
            let mut heap = BinaryHeap::new();
            for &b in &border {
                dist[rv.at(b)] = Some(0.0);
                heap.push(HeapItem(0.0, b));
            }
            while let Some(HeapItem(d, v)) = heap.pop() {
                if dist[rv.at(v)].is_some_and(|x| d > x) || d > falloff_width {
                    continue;
                }
                for &e in mesh.vertex_edges(v) {
                    let [t0, t1] = mesh.edge_triangles(e);
                    if !(in_r[t0 as usize] || (t1 != INVALID && in_r[t1 as usize])) {
                        continue;
                    }
                    // (An edge of a region triangle: its other end is a region vertex too.)
                    let w = mesh.edge_other_vertex(e, v);
                    let nd = d + mesh.edge_length(e);
                    let i = rv.at(w);
                    if dist[i].is_none_or(|x| nd < x) {
                        dist[i] = Some(nd);
                        heap.push(HeapItem(nd, w));
                    }
                }
            }
            for (w, d) in weight.iter_mut().zip(&dist) {
                *w = smoothstep(d.unwrap_or(f64::INFINITY) / falloff_width);
            }
        } else {
            for (w, &b) in weight.iter_mut().zip(&on_border) {
                if b {
                    *w = 0.0;
                }
            }
        }
    }
    Ok((rv, weight))
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

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;
    use std::collections::HashMap;

    /// The weights as they were found before, in maps.
    fn weights_in_maps(mesh: &Mesh, tris: &[u32], falloff_width: f64, keep_mesh_boundary: bool) -> (Vec<u32>, HashMap<u32, f64>) {
        let region = valid_unique_tris(mesh, tris);
        let mut in_r = vec![false; mesh.max_tid()];
        for &t in &region {
            in_r[t as usize] = true;
        }
        let mut verts: Vec<u32> = region.iter().flat_map(|&t| mesh.triangle(t)).collect();
        verts.sort_unstable();
        verts.dedup();
        let border: Vec<u32> = verts.iter().copied().filter(|&v| mesh.vertex_triangles(v).iter().any(|&t| !in_r[t as usize]) || (keep_mesh_boundary && mesh.is_boundary_vertex(v))).collect();
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
        (verts, weight)
    }

    /// The offset as it was done before: weights in maps, the mesh copied, each vertex moved
    /// and then corrected one after another.
    fn offset_one_by_one(mesh: &mut Mesh, tris: &[u32], distance: f64, falloff_width: f64, keep_mesh_boundary: bool, iterations: usize) -> usize {
        let (verts, weight) = weights_in_maps(mesh, tris, falloff_width, keep_mesh_boundary);
        let original = mesh.clone();
        let moves: Vec<(u32, DVec3)> = verts.iter().filter(|&&v| weight[&v] > 0.0).map(|&v| (v, mesh.vertex(v) + mesh.vertex_normal(v) * (distance * weight[&v]))).collect();
        for &(v, p) in &moves {
            mesh.set_vertex(v, p);
        }
        if iterations == 0 || distance == 0.0 {
            return moves.len();
        }
        let region = valid_unique_tris(mesh, tris);
        let bvh = mm_core::bvh::Bvh::build_subset(&original, &region);
        for _ in 0..iterations {
            let mut change = 0f64;
            for &(v, _) in &moves {
                let target = distance * weight[&v];
                let p = mesh.vertex(v);
                let Some(hit) = bvh.nearest(&original, p, f64::INFINITY) else { continue };
                let side = original.tri_normal(hit.tid) * target.signum();
                let d = p - hit.point;
                let dir = if d.length() > 1e-12 * (1.0 + target.abs()) && d.dot(side) > 0.0 { d.normalize() } else { side };
                let q = hit.point + dir * target.abs();
                change = change.max(q.distance(p));
                mesh.set_vertex(v, q);
            }
            if change < 1e-6 * distance.abs() {
                break;
            }
        }
        moves.len()
    }

    fn bits(m: &Mesh) -> Vec<[u64; 3]> {
        m.vertex_ids().map(|v| m.vertex(v).to_array().map(f64::to_bits)).collect()
    }

    #[test]
    fn exact_offset_is_the_one_by_one_offset() {
        // A bumpy sphere (concave and convex places), all of it and a cap with a soft border,
        // out and in, with and without the open boundary kept.
        let mut base = primitives::icosphere(5.0, 4);
        base.transform(|p| p * (1.0 + 0.15 * (3.0 * p.x).sin() * (2.0 * p.y).cos()));
        let mut open = primitives::plane_grid(4.0, 4.0, 30, 30);
        open.transform(|p| DVec3::new(p.x, p.y, 0.2 * (2.0 * p.x).sin() * p.y.cos()));
        for m0 in [&base, &open] {
            let all: Vec<u32> = m0.triangle_ids().collect();
            let cap: Vec<u32> = all.iter().copied().filter(|&t| m0.tri_centroid(t).x > 0.3).collect();
            for (tris, d, fall, keep, it) in [(&all, 0.6, 0.0, false, 5), (&cap, -0.4, 1.5, true, 3), (&cap, 0.8, 0.0, false, 0), (&all, 0.3, 0.7, true, 8)] {
                let (mut a, mut b) = (m0.clone(), m0.clone());
                let na = offset_selection_exact(&mut a, tris, d, fall, keep, it).unwrap();
                let nb = offset_one_by_one(&mut b, tris, d, fall, keep, it);
                assert_eq!(na, nb);
                assert!(bits(&a) == bits(&b), "distance {d}, falloff {fall}, {it} iterations: positions differ");
            }
        }
    }

    /// Extrude as it was done before, with maps keyed by vertex (not flat ends, whose sums
    /// followed the maps' random order).
    fn extrude_with_maps(mesh: &mut Mesh, tris: &[u32], params: &ExtrudeParams) -> ExtrudeResult {
        assert!(!params.flat_end);
        let region = valid_unique_tris(mesh, tris);
        let segs = params.side_segments.max(1);
        let region_group = mesh.tri_group(region[0]);
        let mut in_r = vec![false; mesh.max_tid()];
        for &t in &region {
            in_r[t as usize] = true;
        }
        let mut border: Vec<(u32, u32)> = Vec::new();
        for &t in &region {
            let (tv, te) = (mesh.triangle(t), mesh.tri_edges(t));
            for j in 0..3 {
                let o = mesh.edge_other_triangle(te[j], t);
                if o != INVALID && !in_r[o as usize] {
                    border.push((tv[j], tv[(j + 1) % 3]));
                }
            }
        }
        let mut acc: HashMap<u32, DVec3> = HashMap::new();
        let mut total = DVec3::ZERO;
        for &t in &region {
            let c = mesh.tri_cross(t);
            total += c;
            for v in mesh.triangle(t) {
                *acc.entry(v).or_insert(DVec3::ZERO) += c;
            }
        }
        let fixed = match params.direction {
            ExtrudeDirection::Fixed(d) => Some(d.normalize()),
            ExtrudeDirection::Average => Some(total.normalize_or_zero()),
            ExtrudeDirection::Normal => None,
        };
        let disp: HashMap<u32, DVec3> = acc.iter().map(|(&v, &d)| (v, fixed.unwrap_or_else(|| d.normalize_or_zero()) * params.distance)).collect();
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
        let mut border_verts: Vec<u32> = is_border.keys().copied().collect();
        border_verts.sort_unstable();
        for &v in &border_verts {
            let nv = mesh.append_vertex(mesh.vertex(v) + disp[&v]);
            is_border.insert(v, nv);
        }
        let mut affected: Vec<u32> = region.iter().copied().filter(|&t| mesh.triangle(t).iter().any(|v| is_border.contains_key(v))).collect();
        affected.sort_unstable();
        let recs: Vec<([u32; 3], i32)> = affected.iter().map(|&t| (mesh.triangle(t), mesh.tri_group(t))).collect();
        for &t in &affected {
            mesh.remove_triangle(t, false).unwrap();
        }
        let mut new_region: Vec<u32> = region.iter().copied().filter(|t| affected.binary_search(t).is_err()).collect();
        for (tv, g) in recs {
            new_region.push(mesh.append_triangle(tv.map(|v| is_border.get(&v).copied().unwrap_or(v)), g).unwrap());
        }
        if params.new_region_group {
            let g = new_group(mesh);
            for &t in &new_region {
                mesh.set_tri_group(t, g);
            }
        }
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
        let mut side = Vec::new();
        for &(a, b) in &border {
            let (ca, cb) = (&chain[&a], &chain[&b]);
            for k in 0..segs {
                side.push(mesh.append_triangle([ca[k], cb[k], cb[k + 1]], side_group).unwrap());
                side.push(mesh.append_triangle([ca[k], cb[k + 1], ca[k + 1]], side_group).unwrap());
            }
        }
        ExtrudeResult { new_region_tris: new_region, side_tris: side, border_vertex_map: border_verts.iter().map(|&v| (v, is_border[&v])).collect(), side_group }
    }

    #[test]
    fn extrude_is_the_extrude_with_maps() {
        let base = primitives::icosphere(3.0, 4);
        let cap: Vec<u32> = base.triangle_ids().filter(|&t| base.tri_centroid(t).z > 1.0).collect();
        let band: Vec<u32> = base.triangle_ids().filter(|&t| base.tri_centroid(t).z.abs() < 0.8).collect();
        let mut grid = primitives::plane_grid(4.0, 4.0, 20, 20);
        grid.transform(|p| DVec3::new(p.x, p.y, 0.1 * p.x * p.y));
        let corner: Vec<u32> = grid.triangle_ids().filter(|&t| grid.tri_centroid(t).x > 0.5).collect();
        for (m0, tris) in [(&base, &cap), (&base, &band), (&grid, &corner)] {
            for (direction, segs, new_group, harden) in [(ExtrudeDirection::Normal, 1, false, true), (ExtrudeDirection::Average, 3, true, false), (ExtrudeDirection::Fixed(DVec3::new(0.3, -0.2, 1.0)), 2, true, true)] {
                let params = ExtrudeParams { distance: -0.35, direction, side_segments: segs, harden_sides: harden, flat_end: false, new_region_group: new_group };
                let (mut a, mut b) = (m0.clone(), m0.clone());
                let ra = extrude_selection(&mut a, tris, &params).unwrap();
                let rb = extrude_with_maps(&mut b, tris, &params);
                assert!(bits(&a) == bits(&b), "{direction:?}: positions differ");
                assert!(a.vertex_ids().eq(b.vertex_ids()) && a.triangle_ids().eq(b.triangle_ids()));
                assert!(a.triangle_ids().all(|t| a.triangle(t) == b.triangle(t) && a.tri_group(t) == b.tri_group(t)));
                assert_eq!((ra.new_region_tris, ra.side_tris, ra.border_vertex_map, ra.side_group), (rb.new_region_tris, rb.side_tris, rb.border_vertex_map, rb.side_group));
            }
        }
    }

    #[test]
    fn a_flat_end_is_the_same_every_time() {
        let base = primitives::icosphere(3.0, 4);
        let cap: Vec<u32> = base.triangle_ids().filter(|&t| base.tri_centroid(t).z > 1.0).collect();
        let params = ExtrudeParams { distance: 0.5, flat_end: true, ..Default::default() };
        let (mut a, mut b) = (base.clone(), base.clone());
        extrude_selection(&mut a, &cap, &params).unwrap();
        extrude_selection(&mut b, &cap, &params).unwrap();
        assert!(bits(&a) == bits(&b));
        let (x, y) = (extract_solid(&base, &cap, 0.4, ExtrudeDirection::Normal, true).unwrap(), extract_solid(&base, &cap, 0.4, ExtrudeDirection::Normal, true).unwrap());
        assert!(bits(&x) == bits(&y));
    }

    #[test]
    fn a_cancelled_exact_offset_leaves_the_mesh() {
        let m0 = primitives::icosphere(2.0, 3);
        let all: Vec<u32> = m0.triangle_ids().collect();
        let mut m = m0.clone();
        let calls = std::cell::Cell::new(0);
        // (Stopped before the first correcting pass: a sphere's offset settles after one.)
        let stop = |_: f32| {
            calls.set(calls.get() + 1);
            calls.get() < 2
        };
        let r = offset_selection_exact_with(&mut m, &all, 0.5, 0.0, false, 5, Some(&stop));
        assert!(matches!(r, Err(EditError::Failed(ref s)) if s == "cancelled"), "{r:?}");
        assert!(bits(&m) == bits(&m0));
        // Reported per pass, and the same result as without a callback.
        let seen = std::cell::RefCell::new(Vec::new());
        let watch = |x: f32| {
            seen.borrow_mut().push(x);
            true
        };
        let (mut a, mut b) = (m0.clone(), m0.clone());
        offset_selection_exact_with(&mut a, &all, 0.5, 0.0, false, 5, Some(&watch)).unwrap();
        offset_selection_exact(&mut b, &all, 0.5, 0.0, false, 5).unwrap();
        assert!(bits(&a) == bits(&b));
        assert!(seen.borrow().len() >= 2);
    }
}
