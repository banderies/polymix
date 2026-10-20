//! UV layout editing (the Unwrap tool's layout editor): find the UV islands of a layout, pick
//! them, move / turn / scale / mirror them, keep them on the sheet and re-pack the rest.
//!
//! * An **island** is a set of triangles joined across edges whose two ends have the same UVs
//!   on both sides — a chart of [`crate::unwrap`], or one side of a box projection.
//! * Edits work in **sheet units** (millimetres on the `width × height` sheet of
//!   [`crate::UnwrapParams`]: `u · width`, `v · height`), so turning an island never shears
//!   it when the sheet is not square. UVs stay the sheet position divided by the sheet size.
//! * Editing only moves corners of whole islands, so the islands of a layout stay the same
//!   while it is edited (an island moved onto another one does not merge with it).

use glam::{DAffine2, DVec2};
use mm_core::Mesh;

/// UVs of one triangle's three corners (as stored with the object and written to OBJ).
pub type Corners = [[f32; 2]; 3];

/// Corners closer than this (in UV) are the same.
const SAME: f32 = 1e-6;

/// The UV islands of a layout.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UvIslands {
    /// Island of each triangle id (`u32::MAX` for triangles that do not exist).
    pub of_tri: Vec<u32>,
    /// Triangles of each island.
    pub islands: Vec<Vec<u32>>,
}

impl UvIslands {
    pub fn len(&self) -> usize {
        self.islands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.islands.is_empty()
    }

    /// The island of triangle `t`.
    pub fn island_of(&self, t: u32) -> Option<usize> {
        self.of_tri.get(t as usize).filter(|&&i| i != u32::MAX).map(|&i| i as usize)
    }

    /// The triangles of the given islands.
    pub fn triangles<'a>(&'a self, which: &'a [usize]) -> impl Iterator<Item = u32> + 'a {
        which.iter().filter_map(|&i| self.islands.get(i)).flatten().copied()
    }
}

fn same(a: [f32; 2], b: [f32; 2]) -> bool {
    (a[0] - b[0]).abs() <= SAME && (a[1] - b[1]).abs() <= SAME
}

fn corner_of(tv: [u32; 3], v: u32) -> usize {
    tv.iter().position(|&x| x == v).unwrap_or(0)
}

/// The islands of the layout `uv` (corner UVs by triangle id) on `mesh`.
pub fn uv_islands(mesh: &Mesh, uv: &[Corners]) -> UvIslands {
    let mut of_tri = vec![u32::MAX; mesh.max_tid()];
    let mut islands = Vec::new();
    for t0 in mesh.triangle_ids() {
        if of_tri[t0 as usize] != u32::MAX || t0 as usize >= uv.len() {
            continue;
        }
        let id = islands.len() as u32;
        of_tri[t0 as usize] = id;
        let mut island = vec![t0];
        let mut i = 0;
        while i < island.len() {
            let t = island[i];
            i += 1;
            let tv = mesh.triangle(t);
            for e in mesh.tri_edges(t) {
                let o = mesh.edge_other_triangle(e, t);
                if o == mm_core::INVALID || of_tri[o as usize] != u32::MAX || o as usize >= uv.len() {
                    continue;
                }
                let ov = mesh.triangle(o);
                let joined = mesh.edge_vertices(e).iter().all(|&v| same(uv[t as usize][corner_of(tv, v)], uv[o as usize][corner_of(ov, v)]));
                if joined {
                    of_tri[o as usize] = id;
                    island.push(o);
                }
            }
        }
        islands.push(island);
    }
    UvIslands { of_tri, islands }
}

/// UV → sheet units.
pub fn to_sheet(q: [f32; 2], sheet: DVec2) -> DVec2 {
    DVec2::new(q[0] as f64, q[1] as f64) * sheet
}

/// Sheet units → UV.
pub fn from_sheet(p: DVec2, sheet: DVec2) -> [f32; 2] {
    let q = p / sheet;
    [q.x as f32, q.y as f32]
}

/// Bounding box (sheet units) of the given islands, or `None` if they have no triangles.
pub fn islands_bounds(uv: &[Corners], islands: &UvIslands, which: &[usize], sheet: DVec2) -> Option<(DVec2, DVec2)> {
    let mut lo = DVec2::splat(f64::MAX);
    let mut hi = DVec2::splat(f64::MIN);
    for t in islands.triangles(which) {
        for q in uv.get(t as usize).into_iter().flatten() {
            let p = to_sheet(*q, sheet);
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    (lo.x <= hi.x).then_some((lo, hi))
}

/// Apply `xf` (in sheet units) to every corner of the given islands.
pub fn transform_islands(uv: &mut [Corners], islands: &UvIslands, which: &[usize], xf: DAffine2, sheet: DVec2) {
    for &i in which {
        for &t in islands.islands.get(i).into_iter().flatten() {
            if let Some(c) = uv.get_mut(t as usize) {
                for q in c.iter_mut() {
                    *q = from_sheet(xf.transform_point2(to_sheet(*q, sheet)), sheet);
                }
            }
        }
    }
}

/// `xf` about the point `pivot`.
pub fn about(pivot: DVec2, xf: DAffine2) -> DAffine2 {
    DAffine2::from_translation(pivot) * xf * DAffine2::from_translation(-pivot)
}

fn centre(uv: &[Corners], islands: &UvIslands, which: &[usize], sheet: DVec2) -> Option<DVec2> {
    islands_bounds(uv, islands, which, sheet).map(|(lo, hi)| (lo + hi) * 0.5)
}

/// Move the given islands by `delta` (sheet units).
pub fn move_islands(uv: &mut [Corners], islands: &UvIslands, which: &[usize], delta: DVec2, sheet: DVec2) {
    transform_islands(uv, islands, which, DAffine2::from_translation(delta), sheet);
}

/// Turn the given islands about the centre of their bounding box (degrees, counter-clockwise).
pub fn rotate_islands(uv: &mut [Corners], islands: &UvIslands, which: &[usize], degrees: f64, sheet: DVec2) {
    if let Some(c) = centre(uv, islands, which, sheet) {
        transform_islands(uv, islands, which, about(c, DAffine2::from_angle(degrees.to_radians())), sheet);
    }
}

/// Scale the given islands about the centre of their bounding box.
pub fn scale_islands(uv: &mut [Corners], islands: &UvIslands, which: &[usize], factor: DVec2, sheet: DVec2) {
    if let Some(c) = centre(uv, islands, which, sheet) {
        transform_islands(uv, islands, which, about(c, DAffine2::from_scale(factor)), sheet);
    }
}

/// Mirror the given islands in their bounding box: `horizontal` swaps left and right (U),
/// otherwise top and bottom (V).
pub fn flip_islands(uv: &mut [Corners], islands: &UvIslands, which: &[usize], horizontal: bool, sheet: DVec2) {
    let f = if horizontal { DVec2::new(-1.0, 1.0) } else { DVec2::new(1.0, -1.0) };
    scale_islands(uv, islands, which, f, sheet);
}

fn in_triangle(p: DVec2, a: DVec2, b: DVec2, c: DVec2) -> bool {
    let d1 = (b - a).perp_dot(p - a);
    let d2 = (c - b).perp_dot(p - b);
    let d3 = (a - c).perp_dot(p - c);
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

/// The island with a triangle under the UV point `p` (the last one, drawn on top, when
/// islands overlap).
pub fn island_at(uv: &[Corners], islands: &UvIslands, p: DVec2) -> Option<usize> {
    let q = |c: [f32; 2]| DVec2::new(c[0] as f64, c[1] as f64);
    (0..islands.len()).rev().find(|&i| {
        islands.islands[i].iter().any(|&t| uv.get(t as usize).is_some_and(|c| in_triangle(p, q(c[0]), q(c[1]), q(c[2]))))
    })
}

/// Islands with a corner inside the UV box spanned by `a` and `b`.
pub fn islands_in_box(uv: &[Corners], islands: &UvIslands, a: DVec2, b: DVec2) -> Vec<usize> {
    let (lo, hi) = (a.min(b), a.max(b));
    (0..islands.len())
        .filter(|&i| {
            islands.islands[i].iter().any(|&t| {
                uv.get(t as usize).is_some_and(|c| c.iter().any(|q| (lo.x..=hi.x).contains(&(q[0] as f64)) && (lo.y..=hi.y).contains(&(q[1] as f64))))
            })
        })
        .collect()
}

/// Shift (sheet units) that snaps a box `lo..hi` to the sheet: its sides to the sheet's sides
/// or its centre to the sheet's centre, each axis on its own, when they are within
/// `threshold`.
pub fn snap_to_sheet(lo: DVec2, hi: DVec2, sheet: DVec2, threshold: f64) -> DVec2 {
    let axis = |lo: f64, hi: f64, size: f64| {
        [0.0 - lo, size - hi, size * 0.5 - (lo + hi) * 0.5]
            .into_iter()
            .filter(|d| d.abs() <= threshold)
            .min_by(|a, b| a.abs().total_cmp(&b.abs()))
            .unwrap_or(0.0)
    };
    DVec2::new(axis(lo.x, hi.x, sheet.x), axis(lo.y, hi.y, sheet.y))
}

/// Bring the given islands (as a whole) onto the sheet, `border` from its edges: moved in,
/// and shrunk first if they are bigger than the sheet.
pub fn fit_into_sheet(uv: &mut [Corners], islands: &UvIslands, which: &[usize], sheet: DVec2, border: f64) {
    let Some((lo, hi)) = islands_bounds(uv, islands, which, sheet) else { return };
    let room = (sheet - DVec2::splat(2.0 * border)).max(DVec2::splat(1e-9));
    let size = (hi - lo).max(DVec2::splat(1e-12));
    let s = (room / size).min_element().min(1.0);
    let c = (lo + hi) * 0.5;
    let half = size * s * 0.5;
    // Centre after scaling, clamped so the box lies inside.
    let lo_ok = DVec2::splat(border) + half;
    let hi_ok = sheet - DVec2::splat(border) - half;
    let nc = c.clamp(lo_ok.min(hi_ok), hi_ok.max(lo_ok));
    let xf = DAffine2::from_translation(nc) * DAffine2::from_scale(DVec2::splat(s)) * DAffine2::from_translation(-c);
    transform_islands(uv, islands, which, xf, sheet);
}

/// Bottom-left positions of boxes of `sizes` packed in shelves (tallest first) on the sheet,
/// `border` apart and from the sheet's edges, around the `obstacles` (boxes as lo, hi). `None`
/// if they do not all fit.
fn shelves(sizes: &[DVec2], obstacles: &[(DVec2, DVec2)], sheet: DVec2, border: f64) -> Option<Vec<DVec2>> {
    let eps = 1e-9 * sheet.max_element();
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| sizes[b].y.total_cmp(&sizes[a].y).then(sizes[b].x.total_cmp(&sizes[a].x)));
    let mut pos = vec![DVec2::ZERO; sizes.len()];
    let (mut x, mut y, mut shelf) = (border, border, 0.0f64);
    for i in order {
        let s = sizes[i];
        if s.x > sheet.x - 2.0 * border + eps || s.y > sheet.y - 2.0 * border + eps {
            return None;
        }
        loop {
            if y + s.y > sheet.y - border + eps {
                return None;
            }
            if x + s.x > sheet.x - border + eps {
                // Next shelf; an empty one (blocked by obstacles) climbs over the lowest of them.
                let step = if shelf > 0.0 {
                    shelf + border
                } else {
                    obstacles
                        .iter()
                        .filter(|(lo, hi)| lo.y - border < y + s.y && hi.y + border > y)
                        .map(|(_, hi)| hi.y + border - y)
                        .fold(f64::MAX, f64::min)
                };
                if step == f64::MAX {
                    return None;
                }
                y += step.max(eps);
                x = border;
                shelf = 0.0;
                continue;
            }
            let blocking = obstacles
                .iter()
                .filter(|(lo, hi)| x < hi.x + border - eps && x + s.x > lo.x - border + eps && y < hi.y + border - eps && y + s.y > lo.y - border + eps)
                .map(|(_, hi)| hi.x)
                .fold(f64::MIN, f64::max);
            if blocking > f64::MIN {
                x = blocking + border;
                continue;
            }
            pos[i] = DVec2::new(x, y);
            x += s.x + border;
            shelf = shelf.max(s.y);
            break;
        }
    }
    Some(pos)
}

/// Re-pack the islands that are not `fixed`: placed in shelves on the free part of the sheet,
/// `border` apart and clear of the fixed islands, keeping their orientation and shrinking them
/// all by the same factor if they do not fit as they are. Returns that factor (1 = the same
/// size).
pub fn repack_islands(uv: &mut [Corners], islands: &UvIslands, fixed: &[bool], sheet: DVec2, border: f64) -> Result<f64, String> {
    let is_fixed = |i: usize| fixed.get(i).copied().unwrap_or(false);
    let obstacles: Vec<(DVec2, DVec2)> = (0..islands.len()).filter(|&i| is_fixed(i)).filter_map(|i| islands_bounds(uv, islands, &[i], sheet)).collect();
    let moving: Vec<(usize, (DVec2, DVec2))> = (0..islands.len()).filter(|&i| !is_fixed(i)).filter_map(|i| Some((i, islands_bounds(uv, islands, &[i], sheet)?))).collect();
    if moving.is_empty() {
        return Ok(1.0);
    }
    let sizes = |s: f64| moving.iter().map(|(_, (lo, hi))| (*hi - *lo) * s).collect::<Vec<_>>();
    let mut s = 1.0;
    let mut placed = shelves(&sizes(1.0), &obstacles, sheet, border);
    if placed.is_none() {
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if shelves(&sizes(mid), &obstacles, sheet, border).is_some() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s = lo;
        placed = (s > 0.0).then(|| shelves(&sizes(s), &obstacles, sheet, border)).flatten();
    }
    let placed = placed.ok_or("the islands do not fit next to the fixed ones (border too wide?)")?;
    for ((i, (lo, _)), p) in moving.iter().zip(placed) {
        let xf = DAffine2::from_translation(p) * DAffine2::from_scale(DVec2::splat(s)) * DAffine2::from_translation(-*lo);
        transform_islands(uv, islands, &[*i], xf, sheet);
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{unwrap, UnwrapElements, UnwrapParams, UnwrapType};
    use glam::DVec3;
    use mm_core::primitives;

    const SHEET: DVec2 = DVec2::new(200.0, 250.0);

    /// Three separate flat squares, unwrapped (one island each).
    fn layout() -> (Mesh, Vec<Corners>) {
        let mut m = Mesh::new();
        for i in 0..3 {
            let mut q = primitives::plane_grid(20.0 + i as f64 * 5.0, 10.0, 4, 2);
            q.transform(|p| p + DVec3::new(i as f64 * 100.0, 0.0, 0.0));
            m.append_mesh(&q, 0);
        }
        let l = unwrap(&m, &UnwrapParams { kind: UnwrapType::Conformal, elements: UnwrapElements::Components, ..Default::default() }).unwrap();
        let uv = l.uv.iter().map(|c| c.map(|q| q.as_vec2().to_array())).collect();
        (m, uv)
    }

    fn signed_area(c: &Corners) -> f64 {
        let p = c.map(|q| to_sheet(q, SHEET));
        (p[1] - p[0]).perp_dot(p[2] - p[0]) * 0.5
    }

    fn island_area(uv: &[Corners], isl: &UvIslands, i: usize) -> f64 {
        isl.islands[i].iter().map(|&t| signed_area(&uv[t as usize])).sum()
    }

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5
    }

    #[test]
    fn islands_follow_the_charts() {
        let (m, uv) = layout();
        let isl = uv_islands(&m, &uv);
        assert_eq!(isl.len(), 3);
        for t in m.triangle_ids() {
            let i = isl.island_of(t).unwrap();
            // Each square is its own component, unwrapped as one chart.
            assert!(isl.islands[i].contains(&t));
        }
        // Picking: the centre of a triangle finds its island; a far point finds nothing.
        let t = isl.islands[1][0];
        let c = uv[t as usize];
        let mid = DVec2::new((c[0][0] + c[1][0] + c[2][0]) as f64 / 3.0, (c[0][1] + c[1][1] + c[2][1]) as f64 / 3.0);
        assert_eq!(island_at(&uv, &isl, mid), Some(1));
        assert_eq!(island_at(&uv, &isl, DVec2::new(5.0, 5.0)), None);
        // A box around that point selects that island only.
        let d = DVec2::splat(1e-4);
        assert_eq!(islands_in_box(&uv, &isl, mid - d, mid + d), Vec::<usize>::new(), "no corner inside a tiny box");
        let (lo, hi) = islands_bounds(&uv, &isl, &[1], SHEET).unwrap();
        assert_eq!(islands_in_box(&uv, &isl, lo / SHEET, hi / SHEET), vec![1]);
    }

    #[test]
    fn moving_turning_and_scaling_an_island_keeps_the_others() {
        let (m, uv0) = layout();
        let isl = uv_islands(&m, &uv0);
        let others = |uv: &[Corners]| isl.triangles(&[0, 2]).all(|t| uv[t as usize] == uv0[t as usize]);
        // Move by (10, 5) mm.
        let mut uv = uv0.clone();
        move_islands(&mut uv, &isl, &[1], DVec2::new(10.0, 5.0), SHEET);
        assert!(others(&uv));
        for t in isl.triangles(&[1]) {
            for k in 0..3 {
                let d = to_sheet(uv[t as usize][k], SHEET) - to_sheet(uv0[t as usize][k], SHEET);
                assert!((d - DVec2::new(10.0, 5.0)).length() < 1e-3, "{d}");
            }
        }
        // Turn by 90°: width and height swap (in mm, so no shear on the 200 × 250 sheet), the
        // area and the orientation stay, the centre stays.
        let mut uv = uv0.clone();
        let (lo, hi) = islands_bounds(&uv, &isl, &[1], SHEET).unwrap();
        rotate_islands(&mut uv, &isl, &[1], 90.0, SHEET);
        assert!(others(&uv));
        let (lo2, hi2) = islands_bounds(&uv, &isl, &[1], SHEET).unwrap();
        assert!(((hi2 - lo2) - DVec2::new((hi - lo).y, (hi - lo).x)).length() < 1e-3);
        assert!((((lo2 + hi2) - (lo + hi)) * 0.5).length() < 1e-3);
        assert!((island_area(&uv, &isl, 1) - island_area(&uv0, &isl, 1)).abs() < 1e-3);
        // A full turn in four steps comes back.
        for _ in 0..3 {
            rotate_islands(&mut uv, &isl, &[1], 90.0, SHEET);
        }
        assert!(isl.triangles(&[1]).all(|t| (0..3).all(|k| close(uv[t as usize][k], uv0[t as usize][k]))));
        // Scale ×2 about the centre: four times the area.
        let mut uv = uv0.clone();
        scale_islands(&mut uv, &isl, &[1], DVec2::splat(2.0), SHEET);
        assert!(others(&uv));
        let r = island_area(&uv, &isl, 1) / island_area(&uv0, &isl, 1);
        assert!((r - 4.0).abs() < 1e-3, "{r}");
    }

    #[test]
    fn flips_mirror_in_place() {
        let (m, uv0) = layout();
        let isl = uv_islands(&m, &uv0);
        for horizontal in [true, false] {
            let mut uv = uv0.clone();
            flip_islands(&mut uv, &isl, &[0, 2], horizontal, SHEET);
            assert!(isl.triangles(&[1]).all(|t| uv[t as usize] == uv0[t as usize]), "the others stay");
            // The two islands are mirrored as a whole, in their common box.
            let (a, b) = (islands_bounds(&uv0, &isl, &[0, 2], SHEET).unwrap(), islands_bounds(&uv, &isl, &[0, 2], SHEET).unwrap());
            assert!((a.0 - b.0).length() < 1e-3 && (a.1 - b.1).length() < 1e-3);
            for i in [0, 2] {
                let (x, y) = (islands_bounds(&uv0, &isl, &[i], SHEET).unwrap(), islands_bounds(&uv, &isl, &[i], SHEET).unwrap());
                let mirrored = if horizontal { (a.0.x + a.1.x - x.1.x - y.0.x).abs() } else { (a.0.y + a.1.y - x.1.y - y.0.y).abs() };
                assert!(mirrored < 1e-3, "island {i}");
                // Mirrored triangles: the orientation reverses, the area stays.
                assert!((island_area(&uv, &isl, i) + island_area(&uv0, &isl, i)).abs() < 1e-3);
                assert!(isl.islands[i].iter().all(|&t| signed_area(&uv[t as usize]) * signed_area(&uv0[t as usize]) <= 0.0));
            }
            flip_islands(&mut uv, &isl, &[0, 2], horizontal, SHEET);
            assert!(m.triangle_ids().all(|t| (0..3).all(|k| close(uv[t as usize][k], uv0[t as usize][k]))), "twice is the identity");
        }
    }

    #[test]
    fn snapping_and_fitting_keep_islands_on_the_sheet() {
        assert_eq!(snap_to_sheet(DVec2::new(2.0, 50.0), DVec2::new(30.0, 60.0), SHEET, 3.0), DVec2::new(-2.0, 0.0));
        let d = snap_to_sheet(DVec2::new(85.0, 240.0), DVec2::new(114.0, 248.0), SHEET, 3.0);
        assert!((d - DVec2::new(0.5, 2.0)).length() < 1e-12, "centre in x, top in y: {d}");
        let (m, uv0) = layout();
        let isl = uv_islands(&m, &uv0);
        let mut uv = uv0.clone();
        move_islands(&mut uv, &isl, &[2], DVec2::new(500.0, -40.0), SHEET);
        scale_islands(&mut uv, &isl, &[2], DVec2::splat(30.0), SHEET);
        fit_into_sheet(&mut uv, &isl, &[2], SHEET, 1.0);
        let (lo, hi) = islands_bounds(&uv, &isl, &[2], SHEET).unwrap();
        assert!(lo.x >= 1.0 - 1e-6 && lo.y >= 1.0 - 1e-6 && hi.x <= 199.0 + 1e-3 && hi.y <= 249.0 + 1e-3, "{lo} {hi}");
        assert!(isl.triangles(&[0, 1]).all(|t| uv[t as usize] == uv0[t as usize]));
    }

    #[test]
    fn repacking_the_rest_keeps_the_fixed_island_and_avoids_it() {
        let (m, uv0) = layout();
        let isl = uv_islands(&m, &uv0);
        let mut uv = uv0.clone();
        // Put island 1 in the middle of a small sheet, then re-pack the others around it.
        let sheet = DVec2::new(60.0, 60.0);
        let to_small = |uv: &mut Vec<Corners>| {
            for c in uv.iter_mut() {
                for q in c.iter_mut() {
                    *q = from_sheet(to_sheet(*q, SHEET), sheet);
                }
            }
        };
        to_small(&mut uv);
        fit_into_sheet(&mut uv, &isl, &[1], sheet, 0.0);
        let (lo, hi) = islands_bounds(&uv, &isl, &[1], sheet).unwrap();
        move_islands(&mut uv, &isl, &[1], DVec2::splat(30.0) - (lo + hi) * 0.5, sheet);
        let fixed_before: Vec<Corners> = isl.triangles(&[1]).map(|t| uv[t as usize]).collect();
        let fixed = [false, true, false];
        let s = repack_islands(&mut uv, &isl, &fixed, sheet, 1.0).unwrap();
        assert!(s > 0.0 && s <= 1.0);
        assert_eq!(isl.triangles(&[1]).map(|t| uv[t as usize]).collect::<Vec<_>>(), fixed_before, "the fixed island stays");
        let boxes: Vec<(DVec2, DVec2)> = (0..3).map(|i| islands_bounds(&uv, &isl, &[i], sheet).unwrap()).collect();
        for (i, (lo, hi)) in boxes.iter().enumerate() {
            assert!(lo.x >= 1.0 - 1e-6 && lo.y >= 1.0 - 1e-6 && hi.x <= 59.0 + 1e-6 && hi.y <= 59.0 + 1e-6, "island {i} on the sheet: {lo} {hi}");
        }
        for i in 0..3 {
            for j in i + 1..3 {
                let (a, b) = (boxes[i], boxes[j]);
                let overlap = a.0.x < b.1.x - 1e-6 && b.0.x < a.1.x - 1e-6 && a.0.y < b.1.y - 1e-6 && b.0.y < a.1.y - 1e-6;
                assert!(!overlap, "{i} {j}");
            }
        }
        // Shapes survive (moved and uniformly scaled only).
        for i in [0, 2] {
            // (Both areas are read on the big sheet; `uv` is on the small one.)
            let r = island_area(&uv, &isl, i) / island_area(&uv0, &isl, i) * (sheet.x * sheet.y) / (SHEET.x * SHEET.y);
            assert!((r - s * s).abs() < 1e-3, "{r} vs {}", s * s);
        }
    }
}
