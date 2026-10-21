//! Unwrap (Meshmixer's Edit > Unwrap): flatten the surface into UV charts and pack them on a
//! sheet.
//!
//! * Charts are the face groups or the connected components. A chart that is not a topological
//!   disc (closed, or with several borders) is cut in two by a plane through its centre, again
//!   and again, until every piece is one.
//! * **Conformal** flattens each chart by least-squares conformal maps (angle preserving, free
//!   border, two pinned vertices); **Rigid** then relaxes it toward an as-rigid-as-possible
//!   map (local best rotation per triangle, global cotangent-Laplacian solve), which keeps
//!   lengths better.
//! * Charts are scaled to their true size, turned so their long side is horizontal and packed
//!   (stacked, in a line, or in shelves) on a `width × height` sheet with a `border`, shrinking
//!   everything if needed. UVs are the sheet position divided by the sheet size.

use crate::{EditError, EditResult};
use glam::{DVec2, DVec3};
use mm_core::{topology, Mesh};
use mm_repair::sparse::{solve_pcg, CsrMatrix, PcgParams};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum UnwrapElements {
    Components,
    Groups,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum UnwrapType {
    Conformal,
    Rigid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum Packing {
    /// Every chart at the sheet origin.
    Stacked,
    /// Side by side.
    Line,
    /// Rows of charts filling the sheet.
    Optimized,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct UnwrapParams {
    pub elements: UnwrapElements,
    pub kind: UnwrapType,
    pub packing: Packing,
    pub width: f64,
    pub height: f64,
    pub border: f64,
}

impl Default for UnwrapParams {
    fn default() -> Self {
        Self { elements: UnwrapElements::Groups, kind: UnwrapType::Rigid, packing: Packing::Optimized, width: 200.0, height: 250.0, border: 1.0 }
    }
}

/// Result of [`unwrap`].
#[derive(Clone, Debug, Default)]
pub struct UvLayout {
    /// UVs of each triangle's three corners, indexed by triangle id.
    pub uv: Vec<[DVec2; 3]>,
    /// Charts (after cutting into discs).
    pub charts: usize,
    /// Sheet scale relative to true size (1 = millimetre for millimetre).
    pub scale: f64,
    /// Mean relative difference between UV and 3D triangle areas (after scaling).
    pub area_distortion: f64,
}

/// Charts by face group (split into connected pieces) or by connected component.
fn initial_charts(mesh: &Mesh, by: UnwrapElements) -> Vec<Vec<u32>> {
    let mut seen = vec![false; mesh.max_tid()];
    let mut out = Vec::new();
    for t0 in mesh.triangle_ids() {
        if seen[t0 as usize] {
            continue;
        }
        let g = mesh.tri_group(t0);
        let mut chart = vec![t0];
        seen[t0 as usize] = true;
        let mut i = 0;
        while i < chart.len() {
            let t = chart[i];
            i += 1;
            for e in mesh.tri_edges(t) {
                let o = mesh.edge_other_triangle(e, t);
                if o != mm_core::INVALID && !seen[o as usize] && (by == UnwrapElements::Components || mesh.tri_group(o) == g) {
                    seen[o as usize] = true;
                    chart.push(o);
                }
            }
        }
        out.push(chart);
    }
    out
}

/// Whether the triangles form a topological disc (one border, Euler characteristic 1).
fn is_disc(mesh: &Mesh, tris: &[u32]) -> bool {
    let (sub, _) = mesh.submesh(tris);
    let chi = sub.vertex_count() as i64 - sub.edge_count() as i64 + sub.triangle_count() as i64;
    chi == 1 && topology::boundary_loops(&sub).len() == 1
}

/// Cut `tris` into discs (see the module docs).
fn discs(mesh: &Mesh, tris: Vec<u32>, depth: usize, out: &mut Vec<Vec<u32>>) {
    // Split into connected pieces first.
    let (sub, _) = mesh.submesh(&tris);
    let comps = topology::connected_components(&sub);
    for comp in comps {
        let piece: Vec<u32> = comp.iter().map(|&s| tris[s as usize]).collect();
        if piece.len() < 2 || depth >= 8 || is_disc(mesh, &piece) {
            out.push(piece);
            continue;
        }
        // Principal axes of the triangle centroids, smallest spread first.
        let cs: Vec<DVec3> = piece.iter().map(|&t| mesh.tri_centroid(t)).collect();
        let c = cs.iter().copied().sum::<DVec3>() / cs.len() as f64;
        let mut cov = [[0.0; 3]; 3];
        for p in &cs {
            mm_core::linalg::add_outer(&mut cov, *p - c, 1.0);
        }
        let (_, axes) = mm_core::linalg::sym_eigen3(cov);
        let mut split = None;
        for n in axes {
            let (a, b): (Vec<u32>, Vec<u32>) = piece.iter().partition(|&&t| (mesh.tri_centroid(t) - c).dot(n) < 0.0);
            if !a.is_empty() && !b.is_empty() {
                split = Some((a, b));
                break;
            }
        }
        match split {
            Some((a, b)) => {
                discs(mesh, a, depth + 1, out);
                discs(mesh, b, depth + 1, out);
            }
            None => out.push(piece),
        }
    }
}

/// Triangle corners in its own plane: (0,0), (|e1|,0), (x, y > 0).
fn local_coords(m: &Mesh, t: u32) -> [DVec2; 3] {
    let [p0, p1, p2] = m.tri_positions(t);
    let e1 = p1 - p0;
    let l = e1.length();
    let x = if l > 0.0 { e1 / l } else { DVec3::X };
    let y = m.tri_normal(t).cross(x);
    [DVec2::ZERO, DVec2::new(l, 0.0), DVec2::new((p2 - p0).dot(x), (p2 - p0).dot(y))]
}

/// Least-squares conformal map of a (compact) disc mesh.
fn lscm(m: &Mesh) -> Option<Vec<DVec2>> {
    let n = m.max_vid();
    let verts: Vec<u32> = m.vertex_ids().collect();
    if verts.len() < 3 {
        return None;
    }
    let border: Vec<u32> = verts.iter().copied().filter(|&v| m.is_boundary_vertex(v)).collect();
    let pool = if border.len() >= 2 { &border } else { &verts };
    let far = |from: u32| *pool.iter().max_by(|&&a, &&b| m.vertex(a).distance(m.vertex(from)).total_cmp(&m.vertex(b).distance(m.vertex(from)))).unwrap();
    let a = far(pool[0]);
    let b = far(a);
    if a == b {
        return None;
    }
    let (pa, pb) = (m.vertex(a), m.vertex(b));
    let pins = [(a, DVec2::ZERO), (b, DVec2::new(pa.distance(pb), 0.0))];
    let mut index = vec![usize::MAX; n];
    let mut free = Vec::new();
    for &v in &verts {
        if v != a && v != b {
            index[v as usize] = free.len();
            free.push(v);
        }
    }
    let k = free.len();
    // Normal equations of the 2 rows per triangle.
    let mut trip: Vec<(u32, u32, f64)> = Vec::new();
    let mut rhs = vec![0.0; 2 * k];
    for t in m.triangle_ids() {
        let area = m.tri_area(t);
        if area <= 0.0 {
            continue;
        }
        let p = local_coords(m, t);
        let tv = m.triangle(t);
        let s = 1.0 / (2.0 * area).sqrt();
        let w = [p[2] - p[1], p[0] - p[2], p[1] - p[0]].map(|d| d * s);
        for row in 0..2 {
            let mut cols: Vec<(usize, f64)> = Vec::with_capacity(6);
            let mut constant = 0.0;
            for j in 0..3 {
                // re: Wr·u − Wi·v ; im: Wi·u + Wr·v
                let (cu, cv) = if row == 0 { (w[j].x, -w[j].y) } else { (w[j].y, w[j].x) };
                let v = tv[j];
                if let Some(&(_, q)) = pins.iter().find(|x| x.0 == v) {
                    constant += cu * q.x + cv * q.y;
                } else {
                    let i = index[v as usize];
                    cols.push((i, cu));
                    cols.push((k + i, cv));
                }
            }
            for &(i, ci) in &cols {
                rhs[i] -= ci * constant;
                for &(j, cj) in &cols {
                    trip.push((i as u32, j as u32, ci * cj));
                }
            }
        }
    }
    let nmat = CsrMatrix::from_triplets(2 * k, 2 * k, &trip);
    // Warm start: projection onto the best-fit plane, aligned with the pins.
    let c = verts.iter().map(|&v| m.vertex(v)).sum::<DVec3>() / verts.len() as f64;
    let mut cov = [[0.0; 3]; 3];
    for &v in &verts {
        mm_core::linalg::add_outer(&mut cov, m.vertex(v) - c, 1.0);
    }
    let normal = mm_core::linalg::sym_eigen3(cov).1[0];
    let mut xa = pb - pa;
    xa = (xa - normal * xa.dot(normal)).normalize_or(DVec3::X);
    let ya = normal.cross(xa);
    let mut x = vec![0.0; 2 * k];
    for (i, &v) in free.iter().enumerate() {
        let d = m.vertex(v) - pa;
        x[i] = d.dot(xa);
        x[k + i] = d.dot(ya);
    }
    if k > 0 {
        solve_pcg(&nmat, &rhs, &mut x, &PcgParams { tolerance: 1e-10, max_iterations: 20_000 });
    }
    let mut uv = vec![DVec2::ZERO; n];
    for &(v, q) in &pins {
        uv[v as usize] = q;
    }
    for (i, &v) in free.iter().enumerate() {
        uv[v as usize] = DVec2::new(x[i], x[k + i]);
    }
    uv.iter().all(|p| p.is_finite()).then_some(uv)
}

/// As-rigid-as-possible relaxation of a parameterization (`iterations` local/global steps).
fn arap(m: &Mesh, uv: &mut [DVec2], iterations: usize) {
    let verts: Vec<u32> = m.vertex_ids().collect();
    if verts.len() < 3 {
        return;
    }
    let pin = verts[0];
    let n = m.max_vid();
    let mut index = vec![usize::MAX; n];
    let mut free = Vec::new();
    for &v in &verts {
        if v != pin {
            index[v as usize] = free.len();
            free.push(v);
        }
    }
    let k = free.len();
    let tris: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_area(t) > 0.0).collect();
    // Per triangle: local coords and the (clamped) cotangent of each corner.
    let data: Vec<([DVec2; 3], [f64; 3])> = tris
        .iter()
        .map(|&t| {
            let p = local_coords(m, t);
            let cot = |k: usize| {
                let (a, b) = (p[(k + 1) % 3] - p[k], p[(k + 2) % 3] - p[k]);
                (a.dot(b) / a.perp_dot(b).abs().max(1e-300)).clamp(1e-3, 1e3)
            };
            (p, [cot(0), cot(1), cot(2)])
        })
        .collect();
    let mut trip: Vec<(u32, u32, f64)> = Vec::new();
    for (&t, (_, cot)) in tris.iter().zip(&data) {
        let tv = m.triangle(t);
        for c in 0..3 {
            // Edge opposite corner c.
            let (i, j, w) = (tv[(c + 1) % 3], tv[(c + 2) % 3], cot[c]);
            for (x, y) in [(i, j), (j, i)] {
                if let Some(&ix) = index.get(x as usize).filter(|&&q| q != usize::MAX) {
                    trip.push((ix as u32, ix as u32, w));
                    if let Some(&iy) = index.get(y as usize).filter(|&&q| q != usize::MAX) {
                        trip.push((ix as u32, iy as u32, -w));
                    }
                }
            }
        }
    }
    let lap = CsrMatrix::from_triplets(k, k, &trip);
    let mut x: Vec<DVec3> = free.iter().map(|&v| uv[v as usize].extend(0.0)).collect();
    for _ in 0..iterations {
        let mut rhs = vec![DVec3::ZERO; k];
        for (&t, (p, cot)) in tris.iter().zip(&data) {
            let tv = m.triangle(t);
            // Best rotation taking the triangle's own shape to its current UVs.
            let (mut s00, mut s01, mut s10, mut s11) = (0.0, 0.0, 0.0, 0.0);
            for (c, &w) in cot.iter().enumerate() {
                let (a, b) = ((c + 1) % 3, (c + 2) % 3);
                let du = uv[tv[a] as usize] - uv[tv[b] as usize];
                let dx = p[a] - p[b];
                s00 += w * du.x * dx.x;
                s01 += w * du.x * dx.y;
                s10 += w * du.y * dx.x;
                s11 += w * du.y * dx.y;
            }
            let rot = DVec2::from_angle((s10 - s01).atan2(s00 + s11));
            for (c, &w) in cot.iter().enumerate() {
                let (a, b) = ((c + 1) % 3, (c + 2) % 3);
                let r = rot.rotate(p[a] - p[b]) * w;
                for (v, other, sign) in [(tv[a], tv[b], 1.0), (tv[b], tv[a], -1.0)] {
                    if let Some(&iv) = index.get(v as usize).filter(|&&q| q != usize::MAX) {
                        rhs[iv] += (r * sign).extend(0.0);
                        if other == pin {
                            rhs[iv] += (uv[pin as usize] * w).extend(0.0);
                        }
                    }
                }
            }
        }
        solve_pcg(&lap, &rhs, &mut x, &PcgParams { tolerance: 1e-9, max_iterations: 5000 });
        for (i, &v) in free.iter().enumerate() {
            uv[v as usize] = x[i].truncate();
        }
    }
}

struct Chart {
    tris: Vec<u32>,
    /// Corner UVs of each triangle (chart space).
    uv: Vec<[DVec2; 3]>,
    size: DVec2,
}

/// Flatten one disc of `mesh`.
fn flatten(mesh: &Mesh, tris: Vec<u32>, kind: UnwrapType) -> Chart {
    let (sub, _) = mesh.submesh(&tris);
    let mut uv = lscm(&sub).unwrap_or_else(|| {
        // Degenerate pieces: project onto the first triangle's plane.
        let p = local_coords(&sub, 0);
        let mut out = vec![DVec2::ZERO; sub.max_vid()];
        for (k, v) in sub.triangle(0).into_iter().enumerate() {
            out[v as usize] = p[k];
        }
        out
    });
    if kind == UnwrapType::Rigid {
        arap(&sub, &mut uv, 10);
    }
    let mut corners: Vec<[DVec2; 3]> = (0..tris.len() as u32).map(|t| sub.triangle(t).map(|v| uv[v as usize])).collect();
    // True size: match the total area.
    let a3: f64 = (0..tris.len() as u32).map(|t| sub.tri_area(t)).sum();
    let auv: f64 = corners.iter().map(|c| ((c[1] - c[0]).perp_dot(c[2] - c[0]) * 0.5).abs()).sum();
    let s = if auv > 0.0 { (a3 / auv).sqrt() } else { 1.0 };
    // Long side horizontal.
    let pts: Vec<DVec2> = corners.iter().flatten().map(|p| *p * s).collect();
    let c = pts.iter().copied().sum::<DVec2>() / pts.len().max(1) as f64;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in &pts {
        let d = *p - c;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
    }
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let rot = DVec2::from_angle(-angle);
    for cs in &mut corners {
        for p in cs.iter_mut() {
            *p = rot.rotate((*p * s) - c);
        }
    }
    let lo = corners.iter().flatten().fold(DVec2::splat(f64::MAX), |a, p| a.min(*p));
    let hi = corners.iter().flatten().fold(DVec2::splat(f64::MIN), |a, p| a.max(*p));
    for cs in &mut corners {
        for p in cs.iter_mut() {
            *p -= lo;
        }
    }
    Chart { tris, uv: corners, size: (hi - lo).max(DVec2::ZERO) }
}

/// The discs of the charts (see [`discs`]), in chart order.
fn pieces(mesh: &Mesh, by: UnwrapElements) -> Vec<Vec<u32>> {
    // Charts are cut on their own, so they are spread over the threads; collecting keeps
    // their order.
    initial_charts(mesh, by)
        .into_par_iter()
        .flat_map_iter(|chart| {
            let mut out = Vec::new();
            discs(mesh, chart, 0, &mut out);
            out
        })
        .collect()
}

/// Every disc flattened (see [`flatten`]), in the order of [`pieces`]. Each disc is flattened
/// on its own (a small one wholly on one thread), so they are spread over the threads.
fn flattened(mesh: &Mesh, p: &UnwrapParams) -> Vec<Chart> {
    pieces(mesh, p.elements).into_par_iter().map(|t| flatten(mesh, t, p.kind)).collect()
}

/// Chart offsets for `packing` at scale `s`; returns the offsets and the used extent.
fn place(charts: &[Chart], packing: Packing, s: f64, width: f64, border: f64) -> (Vec<DVec2>, DVec2) {
    let mut off = vec![DVec2::ZERO; charts.len()];
    let mut used = DVec2::ZERO;
    match packing {
        Packing::Stacked => {
            for (i, c) in charts.iter().enumerate() {
                off[i] = DVec2::splat(border);
                used = used.max(c.size * s + DVec2::splat(2.0 * border));
            }
        }
        Packing::Line => {
            let mut x = border;
            for (i, c) in charts.iter().enumerate() {
                off[i] = DVec2::new(x, border);
                x += c.size.x * s + border;
                used = used.max(DVec2::new(x, c.size.y * s + 2.0 * border));
            }
        }
        Packing::Optimized => {
            // Shelves, tallest charts first.
            let mut order: Vec<usize> = (0..charts.len()).collect();
            order.sort_by(|&a, &b| charts[b].size.y.total_cmp(&charts[a].size.y));
            let (mut x, mut y, mut shelf) = (border, border, 0.0f64);
            for i in order {
                let sz = charts[i].size * s;
                if x > border && x + sz.x + border > width {
                    x = border;
                    y += shelf + border;
                    shelf = 0.0;
                }
                off[i] = DVec2::new(x, y);
                x += sz.x + border;
                shelf = shelf.max(sz.y);
                used = used.max(DVec2::new(x, y + sz.y + border));
            }
        }
    }
    (off, used)
}

/// Unwrap `mesh` (see the module docs).
pub fn unwrap(mesh: &Mesh, p: &UnwrapParams) -> EditResult<UvLayout> {
    if mesh.triangle_count() == 0 {
        return Err(EditError::EmptySelection);
    }
    if !(p.width > 0.0 && p.height > 0.0 && p.border >= 0.0) {
        return Err(EditError::InvalidParameter("the sheet needs a positive width and height".into()));
    }
    let charts = flattened(mesh, p);
    // Largest scale (≤ 1) at which the layout fits the sheet.
    let fits = |s: f64| {
        let (_, used) = place(&charts, p.packing, s, p.width, p.border);
        used.x <= p.width + 1e-9 && used.y <= p.height + 1e-9
    };
    let mut s = 1.0;
    if !fits(1.0) {
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s = lo;
    }
    if s <= 0.0 {
        return Err(EditError::InvalidParameter("the charts do not fit the sheet (border too wide?)".into()));
    }
    let (off, _) = place(&charts, p.packing, s, p.width, p.border);
    let sheet = DVec2::new(p.width, p.height);
    let mut uv = vec![[DVec2::ZERO; 3]; mesh.max_tid()];
    let mut dist = 0.0;
    for (c, o) in charts.iter().zip(&off) {
        for (k, &t) in c.tris.iter().enumerate() {
            let q = c.uv[k].map(|x| (x * s + *o) / sheet);
            uv[t as usize] = q;
            let a3 = mesh.tri_area(t) * s * s;
            let auv = ((c.uv[k][1] - c.uv[k][0]).perp_dot(c.uv[k][2] - c.uv[k][0]) * 0.5).abs() * s * s;
            if a3 > 0.0 {
                dist += (auv - a3).abs() / a3;
            }
        }
    }
    Ok(UvLayout { uv, charts: charts.len(), scale: s, area_distortion: dist / mesh.triangle_count() as f64 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn signed(u: &[DVec2; 3]) -> f64 {
        (u[1] - u[0]).perp_dot(u[2] - u[0])
    }

    #[test]
    fn flat_patch_unwraps_isometrically() {
        let mut m = primitives::plane_grid(20.0, 10.0, 12, 6);
        m.transform(|p| glam::DQuat::from_rotation_x(0.6) * p);
        for kind in [UnwrapType::Conformal, UnwrapType::Rigid] {
            let l = unwrap(&m, &UnwrapParams { kind, ..Default::default() }).unwrap();
            assert_eq!(l.charts, 1);
            assert!((l.scale - 1.0).abs() < 1e-12);
            assert!(l.area_distortion < 1e-6, "{kind:?}: {}", l.area_distortion);
            // Lengths survive: an edge of the grid is 20/12 long on the 200 mm sheet.
            let t = m.triangle_ids().next().unwrap();
            let u = l.uv[t as usize];
            let [a, b, _] = m.tri_positions(t);
            let d = ((u[1] - u[0]) * DVec2::new(200.0, 250.0)).length();
            assert!((d - a.distance(b)).abs() < 1e-6, "{d}");
            assert!(m.triangle_ids().all(|t| signed(&l.uv[t as usize]) > 0.0), "no flips");
        }
    }

    #[test]
    fn closed_and_tubular_charts_are_cut_into_discs() {
        let sphere = primitives::icosphere(10.0, 3);
        let tube = primitives::cylinder(5.0, 30.0, 24, 12, false);
        for (m, name) in [(&sphere, "sphere"), (&tube, "tube")] {
            let mut distortion = Vec::new();
            for kind in [UnwrapType::Conformal, UnwrapType::Rigid] {
                let l = unwrap(m, &UnwrapParams { kind, elements: UnwrapElements::Components, ..Default::default() }).unwrap();
                assert!(l.charts >= 2, "{name}");
                let flipped = m.triangle_ids().filter(|&t| signed(&l.uv[t as usize]) <= 0.0).count();
                assert!(flipped * 100 < m.triangle_count(), "{name} {kind:?}: {flipped} flipped");
                assert!(l.area_distortion < 0.6, "{name} {kind:?}: {}", l.area_distortion);
                // Inside the sheet.
                assert!(l.uv.iter().flatten().all(|p| (-1e-9..=1.0 + 1e-9).contains(&p.x) && (-1e-9..=1.0 + 1e-9).contains(&p.y)));
                distortion.push(l.area_distortion);
            }
            // Rigid keeps areas better than conformal (a half tube flattens almost exactly).
            assert!(distortion[1] < distortion[0] + 1e-6, "{name}: {distortion:?}");
        }
    }

    #[test]
    fn charts_flattened_together_are_the_same_as_one_by_one() {
        // Separate patches, a closed sphere and a tube (cut into several discs each), in face
        // groups: spread over threads, every chart comes out bit for bit as on its own, in the
        // same order. (Charts this small solve on one thread, so each is deterministic.)
        let mut m = Mesh::new();
        for i in 0..12 {
            let mut q = primitives::plane_grid(10.0 + i as f64, 8.0, 6, 5);
            q.transform(|p| glam::DQuat::from_rotation_x(0.2 * i as f64) * p + DVec3::new(i as f64 * 40.0, 0.0, 0.0));
            m.append_mesh(&q, i % 3);
        }
        let mut sphere = primitives::icosphere(6.0, 2);
        sphere.transform(|p| p + DVec3::new(0.0, 60.0, 0.0));
        m.append_mesh(&sphere, 7);
        let mut tube = primitives::cylinder(4.0, 20.0, 16, 6, false);
        tube.transform(|p| p + DVec3::new(0.0, -60.0, 0.0));
        m.append_mesh(&tube, 8);
        for elements in [UnwrapElements::Groups, UnwrapElements::Components] {
            for kind in [UnwrapType::Conformal, UnwrapType::Rigid] {
                let mut one_by_one = Vec::new();
                for chart in initial_charts(&m, elements) {
                    discs(&m, chart, 0, &mut one_by_one);
                }
                assert_eq!(pieces(&m, elements), one_by_one);
                let want: Vec<Chart> = one_by_one.into_iter().map(|t| flatten(&m, t, kind)).collect();
                let got = flattened(&m, &UnwrapParams { elements, kind, ..Default::default() });
                assert_eq!(got.len(), want.len());
                let bits = |c: &Chart| (c.tris.clone(), c.uv.iter().flatten().flat_map(|p| [p.x.to_bits(), p.y.to_bits()]).collect::<Vec<_>>(), [c.size.x.to_bits(), c.size.y.to_bits()]);
                assert!(got.iter().zip(&want).all(|(a, b)| bits(a) == bits(b)), "{elements:?} {kind:?}");
            }
        }
    }

    #[test]
    fn packing_keeps_charts_apart_and_shrinks_to_fit() {
        // Six separate squares of 50 mm on a 120 × 120 sheet: they must shrink to fit.
        let mut m = Mesh::new();
        for i in 0..6 {
            let mut q = primitives::plane_grid(50.0, 50.0, 2, 2);
            q.transform(|p| p + DVec3::new(i as f64 * 100.0, 0.0, 0.0));
            m.append_mesh(&q, i);
        }
        let l = unwrap(&m, &UnwrapParams { kind: UnwrapType::Conformal, width: 120.0, height: 120.0, border: 2.0, ..Default::default() }).unwrap();
        assert_eq!(l.charts, 6);
        assert!(l.scale < 1.0 && l.scale > 0.5, "{}", l.scale);
        // Bounding boxes of the charts do not overlap.
        let boxes: Vec<(DVec2, DVec2)> = (0..6)
            .map(|g| {
                let pts: Vec<DVec2> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).flat_map(|t| l.uv[t as usize]).collect();
                (pts.iter().fold(DVec2::splat(9.0), |a, p| a.min(*p)), pts.iter().fold(DVec2::splat(-9.0), |a, p| a.max(*p)))
            })
            .collect();
        for i in 0..6 {
            for j in i + 1..6 {
                let (a, b) = (boxes[i], boxes[j]);
                let overlap = a.0.x < b.1.x - 1e-9 && b.0.x < a.1.x - 1e-9 && a.0.y < b.1.y - 1e-9 && b.0.y < a.1.y - 1e-9;
                assert!(!overlap, "{i} {j}");
            }
        }
    }
}
