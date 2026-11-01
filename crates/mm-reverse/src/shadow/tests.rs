// SPDX-License-Identifier: AGPL-3.0-only
//! Ported from ScanRuler's `tests/sectionShadow.test.ts`, © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only: the shadow of a mesh on a plane, its widest outline seen along the normal,
//! the exact silhouette refined whatever the raster's cell, and the contour the section keeps
//! read as lines and arcs that never come inside the part. The DXF, SVG and STEP writers, the
//! section store and its keys are the app's, and are tested there; of those tests the pieces
//! the exports are made from (the flat fits, the regions a block stands on) are kept here.

use super::*;
use crate::frame::CutAxis;
use crate::slice::slice_mesh;

use std::collections::HashMap;

/// The plane z = 0 seen from +Z: u = X, v = Y.
fn xy() -> SectionFrame {
    SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
}

fn p(x: f64, y: f64) -> DVec2 {
    DVec2::new(x, y)
}

/// A triangle soup welded where its corners are the same point, as a scan's file is read.
fn welded(soup: &[f32]) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let mut index: HashMap<[u32; 3], u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut ids = Vec::new();
    for c in soup.chunks_exact(3) {
        let key = [c[0].to_bits(), c[1].to_bits(), c[2].to_bits()];
        let id = *index.entry(key).or_insert_with(|| {
            positions.push(DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64));
            positions.len() as u32 - 1
        });
        ids.push(id);
    }
    let triangles = ids.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]).collect();
    (positions, triangles)
}

fn mesh_of(positions: &[f32], indices: &[u32]) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    (positions.chunks_exact(3).map(|c| DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64)).collect(), indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect())
}

/// A closed mesh of filled voxels, each face wound outward — boxes, plates with holes, blocks
/// with notches.
fn voxels(dims: [i64; 3], h: f64, filled: impl Fn(i64, i64, i64) -> bool, at: [f64; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let on = |i: i64, j: i64, k: i64| i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz && filled(i, j, k);
    let mut tris: Vec<f32> = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if !on(i, j, k) {
                    continue;
                }
                let cell = [i, j, k];
                for a in 0..3 {
                    for s in [1i64, -1] {
                        let mut nb = cell;
                        nb[a] += s;
                        if on(nb[0], nb[1], nb[2]) {
                            continue;
                        }
                        let b = (a + 1) % 3;
                        let c = (a + 2) % 3;
                        let corner = |db: i64, dc: i64| -> [f32; 3] {
                            let mut q = cell;
                            q[a] += if s > 0 { 1 } else { 0 };
                            q[b] += db;
                            q[c] += dc;
                            [(at[0] + q[0] as f64 * h) as f32, (at[1] + q[1] as f64 * h) as f32, (at[2] + q[2] as f64 * h) as f32]
                        };
                        let quad = if s > 0 { [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)] } else { [corner(0, 0), corner(0, 1), corner(1, 1), corner(1, 0)] };
                        for v in [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]] {
                            tris.extend(v);
                        }
                    }
                }
            }
        }
    }
    tris
}

/// A rigid turn of every vertex: about Y by a, then about X by b (degrees).
fn turned(positions: &[f32], a: f64, b: f64) -> Vec<f32> {
    let (sa, ca) = a.to_radians().sin_cos();
    let (sb, cb) = b.to_radians().sin_cos();
    positions
        .chunks_exact(3)
        .flat_map(|q| {
            let (x, y, z) = (q[0] as f64, q[1] as f64, q[2] as f64);
            let x1 = ca * x + sa * z;
            let z1 = -sa * x + ca * z;
            [x1 as f32, (cb * y - sb * z1) as f32, (sb * y + cb * z1) as f32]
        })
        .collect()
}

/// The convex hull of points, counter-clockwise.
fn hull(pts: &[DVec2]) -> Vec<DVec2> {
    let mut sorted = pts.to_vec();
    sorted.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let cross = |o: DVec2, a: DVec2, b: DVec2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let mut lower: Vec<DVec2> = Vec::new();
    for q in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], *q) <= 1e-12 {
            lower.pop();
        }
        lower.push(*q);
    }
    let mut upper: Vec<DVec2> = Vec::new();
    for q in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], *q) <= 1e-12 {
            upper.pop();
        }
        upper.push(*q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn seg_dist(q: DVec2, a: DVec2, b: DVec2) -> f64 {
    let d = b - a;
    let l2 = d.dot(d);
    let t = if l2 > 0.0 { ((q - a).dot(d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (q.x - a.x - t * d.x).hypot(q.y - a.y - t * d.y)
}

fn poly_dist(q: DVec2, poly: &[DVec2]) -> f64 {
    (0..poly.len()).map(|i| seg_dist(q, poly[i], poly[(i + 1) % poly.len()])).fold(f64::INFINITY, f64::min)
}

fn inside_poly(q: DVec2, poly: &[DVec2]) -> bool {
    in_polygon(q, poly)
}

/// The kept contour as points along every piece, densely.
fn along_shape(sh: &ShadowShape, step: f64) -> Vec<DVec2> {
    along_loops(&sh.loops, step)
}

fn along_loops(loops: &[ShadowLoop], step: f64) -> Vec<DVec2> {
    let mut out = Vec::new();
    for l in loops {
        for pc in &l.pieces {
            match *pc {
                ShadowPiece::Line { from, to } => {
                    let k = ((to - from).length() / step).ceil().max(1.0) as usize;
                    for j in 0..=k {
                        out.push(from + (to - from) * (j as f64 / k as f64));
                    }
                }
                ShadowPiece::Arc { from, center, radius, sweep, .. } => {
                    let a0 = (from.y - center.y).atan2(from.x - center.x);
                    let k = ((sweep.abs() * radius / step).ceil() as usize).max(2);
                    for j in 0..=k {
                        let t = a0 + sweep * j as f64 / k as f64;
                        out.push(center + DVec2::new(t.cos(), t.sin()) * radius);
                    }
                }
                ShadowPiece::Circle { center, radius } => {
                    let k = ((TAU * radius / step).ceil() as usize).max(2);
                    for j in 0..=k {
                        let t = TAU * j as f64 / k as f64;
                        out.push(center + DVec2::new(t.cos(), t.sin()) * radius);
                    }
                }
            }
        }
    }
    out
}

/// Every chain of the cut on the XY sheet, as [u, v].
fn chains_uv(cut: &SectionCut) -> Vec<Vec<DVec2>> {
    cut.chains.iter().map(|c| c.iter().map(|q| p(q.x, q.y)).collect()).collect()
}

fn area(poly: &[DVec2]) -> f64 {
    let n = poly.len();
    let mut s = 0.0;
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        s += poly[j].x * poly[i].y - poly[i].x * poly[j].y;
        j = i;
    }
    s / 2.0
}

fn quad(tris: &mut Vec<f32>, a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) {
    for v in [a, b, c, a, c, d] {
        tris.extend(v.map(|x| x as f32));
    }
}

/// The box of ScanRuler's `tests/helpers.ts`, without its noise: a cube `size` across about the
/// origin, each face a `grid` × `grid` lattice of quads wound outward.
fn box_mesh(size: f64, grid: usize) -> Vec<f32> {
    let h = size / 2.0;
    let faces: [([f64; 3], [f64; 3], [f64; 3]); 6] = [
        ([-h, -h, h], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([-h, -h, -h], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([h, -h, -h], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-h, -h, -h], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([-h, h, -h], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([-h, -h, -h], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    let mut tris = Vec::new();
    for (o, eu, ev) in faces {
        let vert = |i: usize, j: usize| {
            let (a, b) = ((i as f64 / grid as f64) * size, (j as f64 / grid as f64) * size);
            [o[0] + a * eu[0] + b * ev[0], o[1] + a * eu[1] + b * ev[1], o[2] + a * eu[2] + b * ev[2]]
        };
        for i in 0..grid {
            for j in 0..grid {
                quad(&mut tris, vert(i, j), vert(i + 1, j), vert(i + 1, j + 1), vert(i, j + 1));
            }
        }
    }
    tris
}

/// The closed cylinder of ScanRuler's `tests/helpers.ts`, without its noise: about +Z, centred,
/// a wall of `radial` × `axial` quads and flat end caps fanned from the axis.
fn cylinder_mesh(radius: f64, length: f64, radial: usize, axial: usize) -> Vec<f32> {
    let z0 = -length / 2.0;
    let grid: Vec<Vec<[f64; 3]>> = (0..=axial)
        .map(|j| {
            let z = z0 + length * j as f64 / axial as f64;
            (0..radial).map(|k| {
                let a = (k as f64 / radial as f64) * TAU;
                [a.cos() * radius, a.sin() * radius, z]
            }).collect()
        })
        .collect();
    let mut tris = Vec::new();
    for j in 0..axial {
        for k in 0..radial {
            let k2 = (k + 1) % radial;
            quad(&mut tris, grid[j][k], grid[j][k2], grid[j + 1][k2], grid[j + 1][k]);
        }
    }
    for j in [0, axial] {
        let hub = [0.0, 0.0, grid[j][0][2]];
        for k in 0..radial {
            let k2 = (k + 1) % radial;
            let tri = if j == 0 { [hub, grid[j][k2], grid[j][k]] } else { [hub, grid[j][k], grid[j][k2]] };
            for v in tri {
                tris.extend(v.map(|x| x as f32));
            }
        }
    }
    tris
}

fn opts(tolerance: f64) -> ShadowOptions {
    ShadowOptions { tolerance: Some(tolerance), ..Default::default() }
}

/// A cube turned off every axis: seen along Z its outline is the hexagon round the projected
/// corners.
fn turned_cube() -> ((Vec<DVec3>, Vec<[u32; 3]>), Vec<DVec2>) {
    let cube = turned(&box_mesh(20.0, 4), 30.0, 20.0);
    let mesh = welded(&cube);
    let mut corners = Vec::new();
    for sx in [-10.0f32, 10.0] {
        for sy in [-10.0f32, 10.0] {
            for sz in [-10.0f32, 10.0] {
                let t = turned(&[sx, sy, sz], 30.0, 20.0);
                corners.push(p(t[0] as f64, t[1] as f64));
            }
        }
    }
    (mesh, hull(&corners))
}

#[test]
fn is_the_hexagon_of_a_turned_box_refined_onto_its_exact_sides() {
    let ((pos, tris), hexagon) = turned_cube();
    assert_eq!(hexagon.len(), 6);
    let out = shadow_outline(&pos, &tris, &xy(), &opts(0.1));
    assert_eq!(out.exact.len(), 1);
    assert!(!out.exact[0].hole);
    for q in &out.exact[0].points {
        assert!(poly_dist(*q, &hexagon) < 1e-4, "{q:?} is {} off", poly_dist(*q, &hexagon));
    }
    // Every corner of the hexagon is reached, to the cell.
    for c in &hexagon {
        let near = out.exact[0].points.iter().map(|q| (*q - *c).length()).fold(f64::INFINITY, f64::min);
        assert!(near < 0.1, "{near}");
    }
    // The contour kept: six lines meeting at corners just outside the true ones.
    let lp = &out.shape.loops[0];
    assert!(lp.pieces.iter().all(|pc| pc.is_line()), "{:?}", lp.pieces);
    assert_eq!(lp.pieces.len(), 6);
    for c in &hexagon {
        let near = lp.pieces.iter().map(|pc| if let ShadowPiece::Line { from, .. } = pc { (*from - *c).length() } else { f64::INFINITY }).fold(f64::INFINITY, f64::min);
        assert!(near < 0.25, "{near}");
    }
}

#[test]
fn is_never_inside_the_part_between_play_and_play_plus_twice_the_tolerance_outside() {
    let ((pos, tris), hexagon) = turned_cube();
    for (play, tolerance) in [(0.0, 0.1), (0.5, 0.1), (0.05, 0.02), (-0.3, 0.1)] {
        let out = shadow_outline(&pos, &tris, &xy(), &ShadowOptions { tolerance: Some(tolerance), play: Some(play), ..Default::default() });
        let sh = &out.shape;
        assert!(sh.margin.min >= play - 1e-6, "{play} {tolerance}: {:?}", sh.margin);
        // Checked here independently of the kernel: against the true hexagon.
        for q in along_shape(sh, 0.01) {
            let d = poly_dist(q, &hexagon) * if inside_poly(q, &hexagon) { -1.0 } else { 1.0 };
            assert!(d >= play - 1e-4, "{play} {tolerance}: {d}");
            assert!(d <= play + 2.0 * tolerance + 1e-4, "{play} {tolerance}: {d}");
        }
    }
}

#[test]
fn gives_a_cylinder_seen_along_its_axis_one_round_loop_and_across_it_a_rectangle() {
    let (pos, tris) = welded(&cylinder_mesh(10.0, 30.0, 720, 4));
    let out = shadow_outline(&pos, &tris, &xy(), &opts(0.1));
    assert_eq!(out.exact.len(), 1);
    // The 720-gon's flats lie within r(1 − cos(π/720)) of the circle.
    for q in &out.exact[0].points {
        let r = q.length();
        assert!(r < 10.0 + 1e-4);
        assert!(r > 10.0 * (PI / 720.0).cos() - 1e-4);
    }
    let lp = &out.shape.loops[0];
    assert_eq!(lp.pieces.len(), 1);
    let ShadowPiece::Circle { center, radius } = lp.pieces[0] else { panic!("{:?}", lp.pieces[0]) };
    assert!(center.length() < 1e-3);
    assert!((10.0..=10.2).contains(&radius), "{radius}");
    // Across: the cylinder seen from the side, onto the XZ plane.
    let side = SectionFrame { origin: DVec3::ZERO, normal: DVec3::new(0.0, -1.0, 0.0), u: DVec3::X, v: DVec3::Z };
    let across = shadow_outline(&pos, &tris, &side, &opts(0.1));
    assert_eq!(across.exact.len(), 1);
    let rect = [p(-10.0, -15.0), p(10.0, -15.0), p(10.0, 15.0), p(-10.0, 15.0)];
    for q in &across.exact[0].points {
        assert!(poly_dist(*q, &rect) < 1e-4);
    }
    assert_eq!(across.shape.loops[0].pieces.iter().filter(|pc| pc.is_line()).count(), 4);
    assert!(across.shape.area > 600.0);
    assert!(across.shape.area < 20.4 * 30.4);
}

fn holed_plate() -> (Vec<DVec3>, Vec<[u32; 3]>) {
    // 20 × 20 × 2 with a 6 × 6 hole in the middle.
    welded(&voxels([20, 20, 2], 1.0, |i, j, _| !((7..13).contains(&i) && (7..13).contains(&j)), [0.0; 3]))
}

#[test]
fn marks_the_hole_of_a_plate_with_a_through_hole_and_runs_it_the_other_way() {
    let (pos, tris) = holed_plate();
    let sh = shadow_mesh(&pos, &tris, &xy(), &opts(0.1)).shadow;
    assert_eq!(sh.loops.len(), 2);
    let outer = sh.loops.iter().find(|l| !l.hole).expect("an outer loop");
    let hole = sh.loops.iter().find(|l| l.hole).expect("a hole");
    let signed = |pieces: &[ShadowPiece]| area(&along_loops(&[ShadowLoop { hole: false, pieces: pieces.to_vec() }], 0.01));
    assert!(signed(&outer.pieces) > 0.0);
    assert!(signed(&hole.pieces) < 0.0);
    // 400 less 36, give or take the band.
    assert!(sh.area > 400.0 - 36.0);
    assert!(sh.area < 20.4 * 20.4 - 5.6 * 5.6);
}

#[test]
fn gives_two_boxes_apart_two_loops() {
    let (pos, tris) = welded(&voxels([20, 5, 5], 1.0, |i, _, _| !(6..14).contains(&i), [0.0; 3]));
    let sh = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions::default()).shadow;
    assert_eq!(sh.loops.len(), 2);
    assert!(sh.loops.iter().all(|l| !l.hole));
}

#[test]
fn gives_an_open_sheet_seen_edge_on_nothing() {
    // A square in the XZ plane, seen along Z.
    let (pos, tris) = mesh_of(&[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 10.0], &[0, 1, 2, 0, 2, 3]);
    let cut = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions::default());
    assert!(cut.cut.chains.is_empty());
    assert!(cut.shadow.loops.is_empty());
}

#[test]
fn is_the_full_extent_of_a_block_notched_top_front_and_bottom_where_every_cut_is_smaller() {
    // 40 × 30 × 20 mm, voxels of 1 mm: a groove along X on top, a pocket in the front face
    // (y = 0) between z 5 and 15, a groove along Y underneath.
    let notched = |i: i64, j: i64, k: i64| {
        if k >= 16 && (10..20).contains(&j) {
            return false;
        }
        if j < 4 && (5..15).contains(&k) && (10..30).contains(&i) {
            return false;
        }
        if k < 4 && (15..25).contains(&i) {
            return false;
        }
        true
    };
    let (pos, tris) = welded(&voxels([40, 30, 20], 1.0, notched, [0.0; 3]));
    let sh = shadow_mesh(&pos, &tris, &xy(), &opts(0.1)).shadow;
    assert_eq!(sh.loops.len(), 1);
    assert!(sh.area >= 1200.0);
    assert!(sh.area < 40.4 * 30.4);
    // The part stands from z 0 to 20, whatever plane the shadow lies on.
    assert!(sh.extent.min.abs() < 5e-6);
    assert!((sh.extent.max - 20.0).abs() < 5e-6);
    for z in [2.0, 6.0, 10.0, 14.0, 18.0] {
        let cut = slice_mesh(&pos, &tris, DVec3::new(0.0, 0.0, z), DVec3::Z, 0.0);
        let cut_area: f64 = chains_uv(&cut).iter().map(|c| area(c)).sum();
        assert!(cut_area.abs() < 1200.0 - 1.0, "{z}: {cut_area}");
    }
}

#[test]
fn draws_closed_loops_with_the_outward_direction_at_every_point() {
    let (pos, tris) = holed_plate();
    let cut = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions::default());
    let chains = chains_uv(&cut.cut);
    assert_eq!(chains.len(), 2);
    for (c, ch) in chains.iter().enumerate() {
        assert_eq!(ch[0], ch[ch.len() - 1]);
        for k in 0..ch.len() - 1 {
            let nq = cut.cut.normals[c][k];
            let n = p(nq.x, nq.y);
            assert!((n.length() - 1.0).abs() < 5e-6);
            // A step along the normal leaves the material: out of the plate, or into the hole.
            let out = ch[k] + n * 0.3;
            let in_plate = out.x > 0.0 && out.x < 20.0 && out.y > 0.0 && out.y < 20.0 && !(out.x > 7.0 && out.x < 13.0 && out.y > 7.0 && out.y < 13.0);
            assert!(!in_plate, "{:?} along {:?}", ch[k], n);
        }
    }
}

#[test]
fn refines_onto_the_true_silhouette_however_coarse_the_cell() {
    let ((pos, tris), hexagon) = turned_cube();
    for tolerance in [0.04, 0.1, 0.5, 1.0] {
        let out = shadow_outline(&pos, &tris, &xy(), &opts(tolerance));
        for q in &out.exact[0].points {
            assert!(poly_dist(*q, &hexagon) < 1e-4, "{tolerance}");
        }
    }
    // A cell widened for a big part refines just as well.
    let wide = shadow_outline(&pos, &tris, &xy(), &ShadowOptions { tolerance: Some(0.1), max_cells: Some(20_000), ..Default::default() });
    assert!(wide.shape.widened);
    assert!(wide.shape.cell > 0.05);
    for q in &wide.exact[0].points {
        assert!(poly_dist(*q, &hexagon) < 1e-4);
    }
}

#[test]
fn slides_along_its_normal_without_being_taken_again() {
    let ((pos, tris), _) = turned_cube();
    let at = |offset: f64| SectionFrame::along(CutAxis { origin: DVec3::ZERO, dir: DVec3::Z, u: DVec3::X }, offset, None).expect("a frame");
    let cut = shadow_mesh(&pos, &tris, &at(0.0), &ShadowOptions::default());
    let moved = shift_shadow(&cut, &at(7.0)).expect("slid");
    assert!((moved.cut.chains[0][0].z - (cut.cut.chains[0][0].z + 7.0)).abs() < 5e-6);
    assert!((moved.shadow.extent.min - (cut.shadow.extent.min - 7.0)).abs() < 5e-10);
    // Turned, it has to be taken again.
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let tilted = SectionFrame { origin: DVec3::ZERO, normal: DVec3::new(0.0, r, r), u: DVec3::X, v: DVec3::new(0.0, r, -r) };
    assert!(shift_shadow(&cut, &tilted).is_none());
}

#[test]
fn goes_out_to_the_drawing_exports_as_lines_and_arcs_not_a_polyline() {
    // ScanRuler writes these fits to DXF and SVG; the writers are the app's.
    let (pos, tris) = welded(&voxels([20, 30, 4], 1.0, |_, _, _| true, [0.0; 3]));
    let side = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions { tolerance: Some(0.1), play: Some(0.5), ..Default::default() });
    let fits = shadow_flat_fits(&side.shadow);
    assert_eq!(fits.iter().filter(|f| matches!(f, ShadowFlatFit::Line(_))).count(), 4);
    assert_eq!(fits.iter().filter(|f| matches!(f, ShadowFlatFit::Arc(_))).count(), 4);
    // Each arc a quarter turn about a corner, at the offset's radius.
    for f in &fits {
        if let ShadowFlatFit::Arc(a) = f {
            assert!((a.sweep - PI / 2.0).abs() < 0.005, "{}", a.sweep);
            assert!(a.radius > 0.5 && a.radius < 0.75, "{}", a.radius);
        }
    }
    // The readout's words: the band to hundredths of a millimetre, thousandths of an inch.
    let sh = &side.shadow;
    let words = describe_shadow(sh, DisplayUnits::Mm);
    assert_eq!(words.loops, "1 loop");
    assert_eq!(words.area, format!("{} mm²", crate::js_fmt::grouped(sh.area)));
    assert_eq!(words.band, format!("{} to {} mm outside the part", to_fixed(sh.margin.min, 2), to_fixed(sh.margin.max, 2)));
    let inches = describe_shadow(sh, DisplayUnits::In);
    assert_eq!(inches.band, format!("{} to {} in outside the part", to_fixed(sh.margin.min / 25.4, 3), to_fixed(sh.margin.max / 25.4, 3)));
    assert_eq!(inches.area, format!("{} in²", to_fixed(sh.area / (25.4 * 25.4), 3)));
}

#[test]
fn stands_on_a_region_per_outer_loop_with_its_holes() {
    // A plate with a hole beside a lone box: two loops and a hole. (ScanRuler's export stands
    // a block on each of these regions through the part's thickness.)
    let (pos, tris) = welded(&voxels(
        [30, 20, 4],
        1.0,
        |i, j, k| if i < 20 { !((7..13).contains(&i) && (7..13).contains(&j)) } else { i >= 24 && j < 6 && k < 2 },
        [0.0; 3],
    ));
    let cut = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions { tolerance: Some(0.1), play: Some(0.3), ..Default::default() });
    assert_eq!(cut.shadow.loops.len(), 3);
    let regions = shadow_regions(&cut.shadow);
    assert_eq!(regions.len(), 2);
    assert_eq!(regions.iter().map(|r| r.holes.len()).sum::<usize>(), 1);
    assert!(cut.shadow.extent.min.abs() < 5e-6);
    assert!((cut.shadow.extent.max - 4.0).abs() < 5e-6);
}

/// A prism 5 mm high on a wavy outline about the origin — r = 20 + 1.5·sin(12θ), teeth and
/// roots a lines-and-arcs reading finds hard — fanned from the centre at each end.
fn gear_prism(n: usize) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let mut pos: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 5.0];
    for i in 0..n {
        let t = (i as f64 / n as f64) * TAU;
        let r = 20.0 + 1.5 * (12.0 * t).sin();
        pos.extend([(r * t.cos()) as f32, (r * t.sin()) as f32, 0.0, (r * t.cos()) as f32, (r * t.sin()) as f32, 5.0]);
    }
    let mut idx: Vec<u32> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let (b0, b1) = (2 + i as u32 * 2, 2 + j as u32 * 2);
        idx.extend([0, b1, b0, 1, b0 + 1, b1 + 1, b0, b1, b1 + 1, b0, b1 + 1, b0 + 1]);
    }
    mesh_of(&pos, &idx)
}

#[test]
fn keeps_every_piece_between_play_and_play_plus_twice_the_tolerance_outside_whatever_the_reading_made_of_it() {
    let n = 1440;
    let (pos, tris) = gear_prism(n);
    let outline: Vec<DVec2> = (0..n)
        .map(|i| {
            let t = (i as f64 / n as f64) * TAU;
            let r = 20.0 + 1.5 * (12.0 * t).sin();
            p(r * t.cos(), r * t.sin())
        })
        .collect();
    for (play, tolerance) in [(0.0, 0.1), (0.5, 0.1), (0.2, 0.05)] {
        let sh = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions { tolerance: Some(tolerance), play: Some(play), ..Default::default() }).shadow;
        assert_eq!(sh.loops.len(), 1);
        assert!(sh.margin.min >= play - 1e-6, "{play} {tolerance}: {:?}", sh.margin);
        assert!(sh.margin.max <= play + 2.0 * tolerance + 1e-6, "{play} {tolerance}: {:?}", sh.margin);
        for q in along_shape(&sh, 0.02) {
            let d = poly_dist(q, &outline) * if inside_poly(q, &outline) { -1.0 } else { 1.0 };
            assert!(d >= play - 1e-3, "{play} {tolerance}: {d}");
            assert!(d <= play + 2.0 * tolerance + 1e-3, "{play} {tolerance}: {d}");
        }
    }
}

/// The tests' own generator: a 32-bit LCG, as ScanRuler's test writes it.
fn rng(seed: u32) -> impl FnMut() -> f64 {
    let mut s = seed as u64;
    move || {
        s = (s * 1_664_525 + 1_013_904_223) % (1u64 << 32);
        s as f64 / 4_294_967_296.0
    }
}

/// A convex part like a scanned case: corners turning 20–75°, rounded by 1.5 mm, its sides
/// waving 0.4 mm along a few millimetres, stacked as four rings a little apart, each jittered.
/// (6.28 is the test's own figure for a turn, kept so the same parts come out.)
#[allow(clippy::approx_constant)]
fn scanned_case(seed: u32) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let mut rnd = rng(seed * 101);
    let mut turns: Vec<f64> = Vec::new();
    let mut sum = 0.0;
    while sum < 285.0 {
        turns.push(20.0 + rnd() * 55.0);
        sum += turns[turns.len() - 1];
    }
    turns.push(360.0 - sum);
    let mut poly: Vec<DVec2> = Vec::new();
    let (mut x, mut y, mut a) = (0.0, 0.0, rnd() * 6.28);
    for t in &turns {
        let len = 20.0 + rnd() * 60.0;
        x += a.cos() * len;
        y += a.sin() * len;
        poly.push(p(x, y));
        a += t * PI / 180.0;
    }
    let r = 1.5;
    let n = poly.len();
    let (ph, wl) = (rnd() * 6.28, 3.0 + rnd() * 10.0);
    let mut base: Vec<DVec2> = Vec::new();
    let mut s = 0.0;
    let push = |base: &mut Vec<DVec2>, s: f64, q: DVec2, nrm: DVec2| {
        let e = 0.4 * (ph + (TAU * s) / wl).sin() * (0.5 + 0.5 * (s / 17.0).sin());
        base.push(q + nrm * e);
    };
    let corner: Vec<(DVec2, DVec2, DVec2, DVec2)> = (0..n)
        .map(|i| {
            let (p0, p1, p2) = (poly[(i + n - 1) % n], poly[i], poly[(i + 1) % n]);
            let (l1, l2) = ((p1 - p0).length(), (p2 - p1).length());
            let (u1, u2) = ((p1 - p0) / l1, (p2 - p1) / l2);
            let back = r / ((-(u1.dot(u2))).clamp(-1.0, 1.0).acos() / 2.0).tan();
            (p1 - u1 * back, p1 + u2 * back, u1, u2)
        })
        .collect();
    for i in 0..n {
        let (q0, q, u1, u2) = corner[i];
        let c = p(q0.x - u1.y * r, q0.y + u1.x * r);
        let a0 = (q0.y - c.y).atan2(q0.x - c.x);
        let mut a1 = (q.y - c.y).atan2(q.x - c.x);
        while a1 < a0 {
            a1 += TAU;
        }
        let k = (((a1 - a0) * r) / 0.1).ceil().max(1.0) as usize;
        for j in 0..=k {
            let t = a0 + (a1 - a0) * j as f64 / k as f64;
            s += ((a1 - a0) * r) / k as f64;
            push(&mut base, s, c + p(t.cos(), t.sin()) * r, p(t.cos(), t.sin()));
        }
        let e = corner[(i + 1) % n].0;
        let len = (e - q).length();
        let m = (len / 0.25).ceil() as usize;
        for j in 1..m {
            s += len / m as f64;
            push(&mut base, s, q + (e - q) * (j as f64 / m as f64), p(u2.y, -u2.x));
        }
    }
    let mut rnd2 = rng(seed);
    let nb = base.len();
    let cx = base.iter().map(|q| q.x).sum::<f64>() / nb as f64;
    let cy = base.iter().map(|q| q.y).sum::<f64>() / nb as f64;
    let mut pos: Vec<f32> = Vec::new();
    for k in 0..4 {
        for q in &base {
            let (dx, dy) = (q.x - cx, q.y - cy);
            let l = dx.hypot(dy);
            let e = (rnd2() * 2.0 - 1.0) * 0.05;
            pos.extend([(q.x + (dx / l) * e) as f32, (q.y + (dy / l) * e) as f32, (k * 2) as f32]);
        }
    }
    pos.extend([cx as f32, cy as f32, 0.0, cx as f32, cy as f32, 6.0]);
    let nb32 = nb as u32;
    let c0 = 4 * nb32;
    let mut idx: Vec<u32> = Vec::new();
    for k in 0..3 {
        for i in 0..nb32 {
            let (pp, qq) = (k * nb32 + i, k * nb32 + (i + 1) % nb32);
            idx.extend([pp, qq, qq + nb32, pp, qq + nb32, pp + nb32]);
        }
    }
    for i in 0..nb32 {
        idx.extend([c0, (i + 1) % nb32, i, c0 + 1, 3 * nb32 + i, 3 * nb32 + (i + 1) % nb32]);
    }
    mesh_of(&pos, &idx)
}

/// Short lines between two lines that turn one way into them and the other way out: a stub
/// sticking out of a corner.
fn teeth(pieces: &[ShadowPiece], short: f64) -> usize {
    let n = pieces.len();
    let cross = |a: DVec2, b: DVec2| a.x * b.y - a.y * b.x;
    (0..n)
        .filter(|&i| {
            let (ShadowPiece::Line { from: af, to: at }, ShadowPiece::Line { from: bf, to: bt }, ShadowPiece::Line { from: cf, to: ct }) = (pieces[(i + n - 1) % n], pieces[i], pieces[(i + 1) % n]) else { return false };
            if (bt - bf).length() > short {
                return false;
            }
            cross(at - af, bt - bf) * cross(bt - bf, ct - cf) < 0.0
        })
        .count()
}

#[test]
fn reads_a_rounded_corner_at_a_coarse_tolerance_and_no_play_as_one_corner_not_two_with_a_stub_between() {
    // Parts whose corners each came out with a stub 0.3–0.6 mm long.
    for seed in [2, 26, 38, 41] {
        let (pos, tris) = scanned_case(seed);
        let o = shadow_outline(&pos, &tris, &xy(), &ShadowOptions { play: Some(0.0), tolerance: Some(0.5), ..Default::default() });
        assert_eq!(o.shape.loops.len(), 1, "seed {seed}");
        assert_eq!(teeth(&o.shape.loops[0].pieces, 1.0), 0, "seed {seed}");
        assert!(o.shape.margin.min >= -1e-6, "seed {seed}: {:?}", o.shape.margin);
        assert!(o.shape.margin.max <= 1.0 + 1e-6, "seed {seed}: {:?}", o.shape.margin);
        let exact = &o.exact[0].points;
        for q in along_shape(&o.shape, 0.05) {
            let d = poly_dist(q, exact) * if inside_poly(q, exact) { -1.0 } else { 1.0 };
            assert!(d >= -1e-3, "seed {seed}: {d}");
            assert!(d <= 1.0 + 1e-3, "seed {seed}: {d}");
        }
    }
}

/// A prism 5 mm high on a regular n-gon of circumradius r.
fn polygon_prism(n: usize, r: f64) -> ((Vec<DVec3>, Vec<[u32; 3]>), Vec<DVec2>) {
    let outline: Vec<DVec2> = (0..n).map(|i| p(r * (i as f64 / n as f64 * TAU).cos(), r * (i as f64 / n as f64 * TAU).sin())).collect();
    let mut pos: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 5.0];
    for q in &outline {
        pos.extend([q.x as f32, q.y as f32, 0.0, q.x as f32, q.y as f32, 5.0]);
    }
    let mut idx: Vec<u32> = Vec::new();
    for i in 0..n {
        let (b0, b1) = (2 + i as u32 * 2, 2 + ((i + 1) % n) as u32 * 2);
        idx.extend([0, b1, b0, 1, b0 + 1, b1 + 1, b0, b1, b1 + 1, b0, b1 + 1, b0 + 1]);
    }
    (mesh_of(&pos, &idx), outline)
}

/// The direction a piece runs in at its start, or at its end.
fn tangent(pc: &ShadowPiece, end: bool) -> DVec2 {
    match *pc {
        ShadowPiece::Line { from, to } => (to - from) / (to - from).length(),
        ShadowPiece::Circle { .. } => DVec2::ZERO,
        ShadowPiece::Arc { from, to, center, radius, sweep } => {
            let q = if end { to } else { from };
            let s = sweep.signum();
            p((-(q.y - center.y) / radius) * s, ((q.x - center.x) / radius) * s)
        }
    }
}

/// How far each loop turns where one piece meets the next, degrees.
fn kinks(pieces: &[ShadowPiece]) -> Vec<f64> {
    (0..pieces.len())
        .map(|k| {
            let a = tangent(&pieces[k], true);
            let b = tangent(&pieces[(k + 1) % pieces.len()], false);
            a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
        })
        .collect()
}

#[test]
fn rounds_a_faceted_curve_smooth_below_the_angle_and_keeps_it_faceted_at_0() {
    let ((pos, tris), outline) = polygon_prism(24, 20.0);
    let sharp = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions { tolerance: Some(0.1), smooth_deg: Some(0.0), ..Default::default() }).shadow;
    assert_eq!(sharp.smooth_deg, 0.0);
    assert!(sharp.loops[0].pieces.iter().all(|pc| pc.is_line()));
    assert!(kinks(&sharp.loops[0].pieces).into_iter().fold(f64::NEG_INFINITY, f64::max) > 14.0);

    let smooth = shadow_mesh(&pos, &tris, &xy(), &opts(0.1)).shadow;
    assert_eq!(smooth.smooth_deg, 30.0);
    let pieces = &smooth.loops[0].pieces;
    assert_eq!(pieces.iter().filter(|pc| pc.is_arc()).count(), 24);
    // Every line meets the next piece without a kink.
    assert!(kinks(pieces).into_iter().fold(f64::NEG_INFINITY, f64::max) < 0.01);
    // And the outline still lies in the band outside the part.
    assert!(smooth.margin.min >= -1e-6);
    assert!(smooth.margin.max <= 0.2 + 1e-6);
    for q in along_shape(&smooth, 0.02) {
        let d = poly_dist(q, &outline) * if inside_poly(q, &outline) { -1.0 } else { 1.0 };
        assert!(d >= -1e-3, "{d}");
        assert!(d <= 0.2 + 1e-3, "{d}");
    }
}

#[test]
fn keeps_a_corner_turning_more_than_the_angle_sharp() {
    // An octagon turns 45° at each corner.
    let ((pos, tris), _) = polygon_prism(8, 20.0);
    let at30 = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions { tolerance: Some(0.1), smooth_deg: Some(30.0), ..Default::default() }).shadow;
    assert!(at30.loops[0].pieces.iter().all(|pc| pc.is_line()));
    assert_eq!(at30.loops[0].pieces.len(), 8);
    let at50 = shadow_mesh(&pos, &tris, &xy(), &ShadowOptions { tolerance: Some(0.1), smooth_deg: Some(50.0), ..Default::default() }).shadow;
    assert_eq!(at50.loops[0].pieces.iter().filter(|pc| pc.is_arc()).count(), 8);
    assert!(kinks(&at50.loops[0].pieces).into_iter().fold(f64::NEG_INFINITY, f64::max) < 0.01);
    assert!(at50.margin.min >= -1e-6);
    assert!(at50.margin.max <= 0.2 + 1e-6);
}
