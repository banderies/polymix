// SPDX-License-Identifier: AGPL-3.0-only
//! Vertex normals for a freshly read mesh, and the one correction a file's winding may need:
//! a part exported inside-out is turned the right way out, normals and triangles together,
//! because alignment and wall thickness both go by which way the surface faces.
//!
//! Ported from ScanRuler (`src/core/geometry/normals.ts`, with `tetraVolume6` from
//! `centroid.ts` and `extentOf` from `bounds.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. ScanRuler sums and keeps the normals in single precision
//! (`Float32Array`); here they are summed and kept in double precision, as Polymix keeps its
//! geometry.

use glam::DVec3;

use crate::fit3d::linalg3::Extent;

/// Area-weighted per-vertex normals (unnormalised cross products summed, then normalised).
/// Orientation follows the triangle winding — see `orient_normals_outward` for how an
/// inside-out mesh is put right. A vertex no triangle with an area touches keeps whatever
/// tiny sum it has in x and y and gets 1 in z, as ScanRuler gives it.
pub fn compute_vertex_normals(positions: &[DVec3], indices: &[u32]) -> Vec<DVec3> {
    let mut normals = vec![DVec3::ZERO; positions.len()];
    for tri in indices.as_chunks::<3>().0 {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (pa, pb, pc) = (positions[a], positions[b], positions[c]);
        let (abx, aby, abz) = (pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
        let (acx, acy, acz) = (pc.x - pa.x, pc.y - pa.y, pc.z - pa.z);
        let n = DVec3::new(aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx);
        normals[a] += n;
        normals[b] += n;
        normals[c] += n;
    }
    for n in &mut normals {
        let len = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
        if len > 1e-20 {
            *n = DVec3::new(n.x / len, n.y / len, n.z / len);
        } else {
            n.z = 1.0;
        }
    }
    normals
}

/// The least and greatest coordinate on each axis over all the positions — the box inside
/// out (±infinity) when there are none.
pub fn extent_of(positions: &[DVec3]) -> Extent {
    extent_where(positions, |_| true)
}

/// The extent over the positions `keep` takes.
pub(crate) fn extent_where(positions: &[DVec3], keep: impl Fn(usize) -> bool) -> Extent {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for (v, p) in positions.iter().enumerate() {
        if !keep(v) {
            continue;
        }
        let p = p.to_array();
        for k in 0..3 {
            if p[k] < min[k] {
                min[k] = p[k];
            }
            if p[k] > max[k] {
                max[k] = p[k];
            }
        }
    }
    Extent { min: DVec3::from_array(min), max: DVec3::from_array(max) }
}

/// Six times the signed volume of the tetrahedron triangle `a b c` spans with `o` — summed
/// over a closed mesh, six times what it encloses (the divergence theorem), positive when its
/// triangles wind outward.
pub fn tetra_volume6(pa: DVec3, pb: DVec3, pc: DVec3, o: DVec3) -> f64 {
    let (ax, ay, az) = (pa.x - o.x, pa.y - o.y, pa.z - o.z);
    let (bx, by, bz) = (pb.x - o.x, pb.y - o.y, pb.z - o.z);
    let (cx, cy, cz) = (pc.x - o.x, pc.y - o.y, pc.z - o.z);
    ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
}

/// Put an inside-out mesh right, in place: flip every normal and reverse every triangle's
/// winding when the winding decisively encloses its volume inside-out. The alignment code
/// matches surfaces by the *signed* dot of scan and nominal normals, and wall thickness casts
/// into the material along them, so a file exported with reversed winding would silently fail
/// both.
///
/// The winding must turn with the normals: a viewer that draws the scan double-sided takes
/// the lit side from the winding and negates the vertex normal on back faces, so corrected
/// normals over uncorrected winding would be negated right back on every visible fragment,
/// leaving the whole part lit from behind.
///
/// The test is the signed volume about the centroid: origin-independent for a closed mesh,
/// and anchoring at the centroid keeps an open scan's spurious contribution small. A scan too
/// open to give a decisive signal (a single sheet has none) is left exactly as wound. It is
/// still a guess: a strip open on every side that curves round its own inside can give a
/// decisive signal the wrong way, so a file that says its winding is settled is not put
/// through it. Returns whether the mesh was turned.
pub fn orient_normals_outward(positions: &[DVec3], indices: &mut [u32], normals: &mut [DVec3]) -> bool {
    orient_over(positions, |_| true, indices, normals)
}

/// `orient_normals_outward` with the centroid and the box taken over the vertices `live`
/// takes — for a mesh whose vertex slots include removed vertices.
pub(crate) fn orient_over(positions: &[DVec3], live: impl Fn(usize) -> bool, indices: &mut [u32], normals: &mut [DVec3]) -> bool {
    let mut count = 0usize;
    let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
    for (v, p) in positions.iter().enumerate() {
        if !live(v) {
            continue;
        }
        cx += p.x;
        cy += p.y;
        cz += p.z;
        count += 1;
    }
    if count == 0 {
        return false;
    }
    let Extent { min, max } = extent_where(positions, &live);
    let n = count as f64;
    let centre = DVec3::new(cx / n, cy / n, cz / n);

    let mut vol6 = 0.0;
    for tri in indices.as_chunks::<3>().0 {
        vol6 += tetra_volume6(positions[tri[0] as usize], positions[tri[1] as usize], positions[tri[2] as usize], centre);
    }

    let bbox_vol = (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
    // Decisive means the enclosed volume is a real fraction of the bounding box; below that
    // the mesh is a sheet and its orientation is not knowable.
    if !(bbox_vol > 0.0) || vol6 >= -6e-3 * bbox_vol {
        return false;
    }
    for n in normals.iter_mut() {
        *n = -*n;
    }
    for tri in indices.as_chunks_mut::<3>().0 {
        tri.swap(1, 2);
    }
    true
}
