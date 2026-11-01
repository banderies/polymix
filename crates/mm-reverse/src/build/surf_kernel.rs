// SPDX-License-Identifier: AGPL-3.0-only
//! What the surface features ask of the kernel beyond SolidCraft's own calls, made on its
//! public API: a face through a fitted surface's control net, a face through a regular grid of
//! points (ScanRuler's `surfaceThrough`, Open CASCADE's PointsToBSplineSurface), a body's
//! surface area, its open edges, one point on each face, and a body's display mesh made a
//! search structure — the nearest point of its faces, which face that is, whether a point lies
//! inside. The calls ported from ScanRuler (`plugins/reverse/core/kernel.ts` surfaceThrough,
//! surfaceArea, `kernelMath.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only; the kernel is
//! SolidCraft's (`solidcraft_kernel`, MIT OR Apache-2.0).

use glam::DVec3;
use mm_core::bvh::Bvh;
use mm_core::Mesh;
use solidcraft_geom::bspline::{CurveN, NurbsSurface};
use solidcraft_geom::tspline::TSpline;
use solidcraft_geom::Vec3 as SVec3;
use solidcraft_kernel::{self as kernel, Body};

use super::{diagonal_of, dv3, kerr, mesh_deflection};
use crate::surface_fit::{greville, Grid, SurfaceFitResult};

/// A face through a tensor B-spline net: `points` with u running fastest (`nu` × `nv`), on
/// clamped knots of odd degree. One face, a surface body.
pub(crate) fn net_face(degree: usize, knots_u: Vec<f64>, knots_v: Vec<f64>, points: &[DVec3]) -> Result<Body, String> {
    let pts: Vec<SVec3> = points.iter().map(|p| SVec3::new(p.x, p.y, p.z)).collect();
    let weights = vec![1.0; pts.len()];
    let surface = NurbsSurface::new([degree, degree], [knots_u, knots_v], &pts, &weights).map_err(|e| format!("The surface's net is no B-spline surface: {e}."))?;
    let ts = TSpline::from_nurbs(&surface).map_err(|e| format!("The surface's net is no B-spline surface: {e}."))?;
    kernel::tspline_sheet(&ts).map_err(kerr)
}

/// The kernel's face of a fitted freeform surface: its own control net, exactly — each control
/// point standing over the Greville abscissa of its knots at its height, which the height
/// field's B-spline reproduces over the plane. ScanRuler's kernel takes no net and is handed a
/// grid sampled off the fit (`gridOf`, `surfaceThrough`); SolidCraft's takes the net, so the face
/// is the fit itself. A fit of even degree, which a T-spline sheet does not take, goes through
/// a grid of it.
pub(crate) fn fit_face(fit: &SurfaceFitResult) -> Result<Body, String> {
    let p = fit.degree;
    if p % 2 == 0 || fit.nu <= p || fit.nv <= p || fit.heights.len() < fit.nu * fit.nv || fit.knots_u.len() != fit.nu + p + 1 || fit.knots_v.len() != fit.nv + p + 1 {
        let grid = crate::surface_fit::grid_of(fit, crate::surface_fit::GRID_PER_SPAN);
        return surface_through(&grid);
    }
    let gu = greville(&fit.knots_u, fit.nu, p);
    let gv = greville(&fit.knots_v, fit.nv, p);
    let mut points = Vec::with_capacity(fit.nu * fit.nv);
    for (j, v) in gv.iter().enumerate() {
        for (i, u) in gu.iter().enumerate() {
            let h = fit.heights[i * fit.nv + j];
            points.push(fit.origin + fit.axis_u * *u + fit.axis_v * *v + fit.normal * h);
        }
    }
    net_face(p, fit.knots_u.clone(), fit.knots_v.clone(), &points)
}

/// A cubic face through a regular grid of points (ScanRuler's kernel `surfaceThrough`): `rows`
/// along u, `cols` along v, row-major — each row interpolated, then each column of the rows'
/// control points (global interpolation, The NURBS Book A9.4), the parameters spaced evenly as
/// the grid is. Through the points exactly; between them as a C2 surface runs.
pub(crate) fn surface_through(grid: &Grid) -> Result<Body, String> {
    let (rows, cols) = (grid.rows, grid.cols);
    if rows < 4 || cols < 4 || grid.points.len() != rows * cols {
        return Err("The surface grid is malformed.".into());
    }
    let even = |n: usize| -> Vec<f64> { (0..n).map(|i| i as f64 / (n - 1) as f64).collect() };
    let (tu, tv) = (even(rows), even(cols));
    let mut knots_v = Vec::new();
    // Each row's control points along v.
    let mut by_row: Vec<Vec<[f64; 3]>> = Vec::with_capacity(rows);
    for r in 0..rows {
        let row: Vec<[f64; 3]> = grid.points[r * cols..(r + 1) * cols].iter().map(|p| [p.x, p.y, p.z]).collect();
        let c = CurveN::<3>::interpolate(&tv, &row, 3).map_err(|e| format!("The surface could not be made through the grid: {e}."))?;
        knots_v = c.knots;
        by_row.push(c.ctrl);
    }
    let mut knots_u = Vec::new();
    // Each column of those along u: ctrl[c][r].
    let mut by_col: Vec<Vec<[f64; 3]>> = Vec::with_capacity(cols);
    for c in 0..cols {
        let col: Vec<[f64; 3]> = by_row.iter().map(|row| row[c]).collect();
        let k = CurveN::<3>::interpolate(&tu, &col, 3).map_err(|e| format!("The surface could not be made through the grid: {e}."))?;
        knots_u = k.knots;
        by_col.push(k.ctrl);
    }
    let mut points = Vec::with_capacity(rows * cols);
    for col in &by_col {
        for p in col {
            points.push(DVec3::new(p[0], p[1], p[2]));
        }
    }
    net_face(3, knots_u, knots_v, &points)
}

/// A fitted surface carried on past its range to span `u` and `v`: past its edge not on as its
/// polynomial but flat, along the slope it has there (`carried_height`) — ScanRuler's surface
/// through `reachingGridOf`'s grid. Here a cubic height field over the fit's own frame on the
/// fit's own knots, a span more each way it is carried on (a cubic span runs flat), fitted by
/// least squares to the carried heights sampled four a span — which it holds exactly: a net
/// hardly larger than the fit's, which the kernel cuts with quickly, where a surface through
/// every sample of the grid has a control point for each.
pub(crate) fn carried_fit(fit: &SurfaceFitResult, u: [f64; 2], v: [f64; 2]) -> Result<SurfaceFitResult, String> {
    let p = 3;
    // The fit's own knots inside, and its range's ends twice where it is carried on past them:
    // the carried height runs on from the fit with its slope but not its bend, so the spline
    // there is C1 — and holds the fit exactly inside, and the flat run outside.
    let knots_of = |own: &[f64], n: usize, reach: [f64; 2]| -> Vec<f64> {
        let inner = own.get(p..=n).unwrap_or(&[]);
        let (first, last) = (inner.first().copied(), inner.last().copied());
        let mut k = vec![reach[0]; p + 1];
        for x in inner {
            if !(*x > reach[0] + 1e-9 && *x < reach[1] - 1e-9) || k.last().is_some_and(|l| *x <= *l) {
                continue;
            }
            k.push(*x);
            if Some(*x) == first || Some(*x) == last {
                k.push(*x);
            }
        }
        k.extend(std::iter::repeat_n(reach[1], p + 1));
        k
    };
    let ku = knots_of(&fit.knots_u, fit.nu, u);
    let kv = knots_of(&fit.knots_v, fit.nv, v);
    let (nu, nv) = (ku.len() - p - 1, kv.len() - p - 1);
    if nu <= p || nv <= p {
        return Err("The surface could not be carried on.".into());
    }
    // Four samples a span, at the spans' own spacing.
    let samples = |k: &[f64]| -> Vec<f64> {
        let mut out = Vec::new();
        for w in k.windows(2).filter(|w| w[1] > w[0]) {
            for s in 0..4 {
                out.push(w[0] + (w[1] - w[0]) * s as f64 / 4.0);
            }
        }
        out.push(*k.last().unwrap_or(&0.0));
        out
    };
    let us = samples(&ku);
    let vs = samples(&kv);
    let mv = vs.len();
    // Each sample's basis row: the span and its p + 1 values.
    let rows = |knots: &[f64], n: usize, ts: &[f64]| -> Vec<Vec<f64>> {
        ts.iter()
            .map(|t| {
                let span = crate::surface_fit::find_span(knots, n, p, *t);
                let mut b = vec![0.0; p + 1];
                crate::surface_fit::basis_functions(knots, span, p, *t, &mut b);
                let mut row = vec![0.0; n];
                for (k, x) in b.iter().enumerate() {
                    if let Some(slot) = row.get_mut(span - p + k) {
                        *slot = *x;
                    }
                }
                row
            })
            .collect()
    };
    let (bu, bv) = (rows(&ku, nu, &us), rows(&kv, nv, &vs));
    let normal = |b: &[Vec<f64>], n: usize| -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for row in b {
            for r in 0..n {
                if row[r] == 0.0 {
                    continue;
                }
                for c in 0..n {
                    a[r * n + c] += row[r] * row[c];
                }
            }
        }
        a
    };
    let (au, av) = (normal(&bu, nu), normal(&bv, nv));
    // Along u for each column of samples, then along v for each row of those.
    let mut y = vec![0.0; nu * mv];
    for (j, vv) in vs.iter().enumerate() {
        let mut rhs = vec![0.0; nu];
        for (i, uu) in us.iter().enumerate() {
            let h = crate::surface_fit::carried_height(fit, *uu, *vv);
            for r in 0..nu {
                rhs[r] += bu[i][r] * h;
            }
        }
        let x = crate::linalg::cholesky_solve(nu, &au, &rhs, 1e-14).ok_or("The surface could not be carried on.")?;
        for r in 0..nu {
            y[r * mv + j] = x[r];
        }
    }
    let mut heights = vec![0.0; nu * nv];
    for r in 0..nu {
        let mut rhs = vec![0.0; nv];
        for (j, row) in bv.iter().enumerate() {
            for c in 0..nv {
                rhs[c] += row[c] * y[r * mv + j];
            }
        }
        let x = crate::linalg::cholesky_solve(nv, &av, &rhs, 1e-14).ok_or("The surface could not be carried on.")?;
        heights[r * nv..(r + 1) * nv].copy_from_slice(&x);
    }
    Ok(SurfaceFitResult { u_range: u, v_range: v, knots_u: ku, knots_v: kv, nu, nv, heights, degree: p, footprint: None, ..fit.clone() })
}

/// A solid's volume off its display mesh, mm³ — quick, to tell pieces apart by: the kernel's
/// own measure tessellates a trimmed freeform face far finer, and takes seconds where this
/// takes milliseconds.
pub(crate) fn mesh_volume(body: &Body) -> f64 {
    let Ok(m) = body.display_mesh(mesh_deflection(diagonal_of(body))) else { return 0.0 };
    let p = |i: u32| m.positions.get(i as usize).map(|v| dv3(*v)).unwrap_or_default();
    m.triangles.iter().map(|t| p(t[0]).dot(p(t[1]).cross(p(t[2])))).sum::<f64>() / 6.0
}

/// A shape's surface area, mm² (ScanRuler's `surfaceArea`).
pub(crate) fn surface_area(body: &Body) -> f64 {
    kernel::measure(body).map(|m| m.area).unwrap_or(0.0)
}

/// A body's display mesh at the chord its size reads at, as plain points and triangles with
/// each triangle's face and the edges with their faces — `scale` times coarser where asked.
pub(crate) struct ShapeMesh {
    pub positions: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
    pub tri_face: Vec<u32>,
    pub edges: Vec<Vec<DVec3>>,
    pub edge_faces: Vec<Vec<u32>>,
    pub seams: Vec<bool>,
}

impl ShapeMesh {
    pub(crate) fn of(body: &Body, scale: f64) -> Result<ShapeMesh, String> {
        let tol = mesh_deflection(diagonal_of(body)) * scale.max(1.0);
        let m = body.display_mesh(tol).map_err(kerr)?;
        Ok(ShapeMesh {
            positions: m.positions.iter().map(|p| dv3(*p)).collect(),
            triangles: m.triangles.clone(),
            tri_face: m.tri_face.clone(),
            edges: m.edges.iter().map(|e| e.iter().map(|p| dv3(*p)).collect()).collect(),
            edge_faces: m.edge_faces.clone(),
            seams: m.seams.clone(),
        })
    }

    /// The edges only one face meets: a sheet's open edges (seams aside).
    pub(crate) fn open_edges(&self) -> Vec<Vec<DVec3>> {
        self.edges
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.seams.get(*i).copied().unwrap_or(false))
            .filter(|(i, _)| {
                let mut f: Vec<u32> = self.edge_faces.get(*i).cloned().unwrap_or_default();
                f.dedup();
                f.len() == 1
            })
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// One point on each face: the middle of its first triangle (a round face's centroid is off
    /// it). By face index; `None` for a face with no triangle.
    pub(crate) fn face_points(&self, faces: usize) -> Vec<Option<DVec3>> {
        let mut out = vec![None; faces];
        for (t, f) in self.triangles.iter().zip(&self.tri_face) {
            if let Some(slot) = out.get_mut(*f as usize) {
                if slot.is_none() {
                    let [a, b, c] = t.map(|v| self.positions.get(v as usize).copied().unwrap_or_default());
                    *slot = Some((a + b + c) / 3.0);
                }
            }
        }
        out
    }

    /// The triangles of one face.
    pub(crate) fn face_triangles(&self, face: u32) -> Vec<[u32; 3]> {
        self.triangles.iter().zip(&self.tri_face).filter(|(_, f)| **f == face).map(|(t, _)| *t).collect()
    }
}

/// A mesh made a search structure: the nearest point of it to a point and the triangle that is
/// on, and whether a point lies inside it (by rays, for a closed one).
pub(crate) struct MeshIndex {
    mesh: Mesh,
    bvh: Bvh,
    tri_face: Vec<u32>,
}

impl MeshIndex {
    pub(crate) fn new(positions: &[DVec3], triangles: &[[u32; 3]], tri_face: &[u32]) -> Option<MeshIndex> {
        if triangles.is_empty() {
            return None;
        }
        let (mesh, _) = Mesh::from_triangles(positions, triangles, None);
        let bvh = Bvh::build(&mesh);
        Some(MeshIndex { mesh, bvh, tri_face: tri_face.to_vec() })
    }

    pub(crate) fn of(m: &ShapeMesh) -> Option<MeshIndex> {
        MeshIndex::new(&m.positions, &m.triangles, &m.tri_face)
    }

    /// The nearest point within `reach`, its distance, the face it lies on and that triangle's
    /// unit normal (the way the face looks).
    pub(crate) fn nearest(&self, p: DVec3, reach: f64) -> Option<(DVec3, f64, u32, DVec3)> {
        let h = self.bvh.nearest(&self.mesh, p, reach)?;
        let face = self.tri_face.get(h.tid as usize).copied().unwrap_or(0);
        Some((h.point, h.dist_sq.sqrt(), face, self.mesh.tri_normal(h.tid)))
    }

    /// Whether a point lies inside the mesh, by rays.
    pub(crate) fn inside(&self, p: DVec3) -> bool {
        self.bvh.is_inside(&self.mesh, p)
    }

    /// Whether a point lies inside the mesh deeper than `depth`.
    pub(crate) fn deep_inside(&self, p: DVec3, depth: f64) -> bool {
        self.inside(p) && self.nearest(p, depth).is_none()
    }
}

/// Which feature made each face of a shape an operation left with no history of its own — a
/// split, a sheet trimmed or sewn (ScanRuler's `tracedHistory`): each face traced by a point on
/// it to the face of the bodies before (`before`, each with its faces' makers) it lies on, and
/// keeping that face's maker; a face lying on none is the step's own.
pub(crate) fn traced_origins(result: &Body, before: &[(&Body, &[i64])], step: i64) -> Vec<i64> {
    let n = result.face_count();
    let mut out = vec![step; n];
    let Ok(m) = ShapeMesh::of(result, 1.0) else { return out };
    let points = m.face_points(n);
    for (b, origins) in before {
        let Ok(bm) = ShapeMesh::of(b, 1.0) else { continue };
        let Some(index) = MeshIndex::of(&bm) else { continue };
        let tol = super::surf_split::on_tolerance(diagonal_of(result), diagonal_of(b));
        for (face, p) in points.iter().enumerate() {
            let Some(p) = p else { continue };
            if out[face] != step {
                continue;
            }
            if let Some((_, _, on, _)) = index.nearest(*p, tol) {
                if let Some(o) = origins.get(on as usize).copied().filter(|o| *o >= 0) {
                    out[face] = o;
                }
            }
        }
    }
    out
}

/// Points spread along polylines, `count` in all by length (at least one a polyline), each at
/// the middle of its share.
pub(crate) fn along_polylines(lines: &[Vec<DVec3>], count: usize) -> Vec<DVec3> {
    let lengths: Vec<f64> = lines.iter().map(|l| l.windows(2).map(|w| (w[1] - w[0]).length()).sum()).collect();
    let total: f64 = lengths.iter().sum();
    let mut out = Vec::new();
    if !(total > 0.0) {
        return out;
    }
    for (l, len) in lines.iter().zip(&lengths) {
        let n = ((count as f64 * len / total).round() as usize).max(1);
        for j in 0..n {
            let at = len * (j as f64 + 0.5) / n as f64;
            out.push(point_along(l, at));
        }
    }
    out
}

/// The point a distance along a polyline.
pub(crate) fn point_along(line: &[DVec3], at: f64) -> DVec3 {
    let mut acc = 0.0;
    for w in line.windows(2) {
        let l = (w[1] - w[0]).length();
        if acc + l >= at && l > 0.0 {
            return w[0] + (w[1] - w[0]) * ((at - acc) / l);
        }
        acc += l;
    }
    line.last().copied().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_fit::{fit_surface, grid_area, grid_of, SurfaceFitParams, PULL};

    #[test]
    fn makes_the_fits_own_face_and_one_through_a_grid_of_it() {
        // A saddle sampled over ±15 mm and fitted with a 5 × 5 net.
        let mut pts = Vec::new();
        for i in 0..21 {
            for j in 0..21 {
                let (x, y) = (-15.0 + 1.5 * i as f64, -15.0 + 1.5 * j as f64);
                pts.push(DVec3::new(x, y, 3.0 + 0.03 * x * x - 0.01 * y * y));
            }
        }
        let params = SurfaceFitParams { net: Some(crate::surface_fit::Net { u: 5, v: 5 }), expand_pct: 5.0, ..SurfaceFitParams::default() };
        let fit = fit_surface(&pts, &params, None, PULL).unwrap();
        let face = fit_face(&fit).unwrap();
        assert!(face.is_surface());
        assert_eq!(face.face_count(), 1);
        let grid = grid_of(&fit, 3);
        let area = surface_area(&face);
        assert!((area / grid_area(&grid) - 1.0).abs() < 0.01, "{area} against {}", grid_area(&grid));
        let through = surface_through(&grid).unwrap();
        assert!((surface_area(&through) / area - 1.0).abs() < 0.005);
        // The face lies on the fit: every grid point within a micron of its mesh.
        let m = ShapeMesh::of(&face, 1.0).unwrap();
        let idx = MeshIndex::of(&m).unwrap();
        for p in grid.points.iter().step_by(7) {
            let (_, d, _, _) = idx.nearest(*p, 1.0).unwrap();
            assert!(d < 0.02, "{d}");
        }
        assert_eq!(m.open_edges().len(), 4);
    }

    #[test]
    fn spreads_points_along_polylines_by_length() {
        let lines = vec![vec![DVec3::ZERO, DVec3::new(30.0, 0.0, 0.0)], vec![DVec3::ZERO, DVec3::new(0.0, 10.0, 0.0)]];
        let pts = along_polylines(&lines, 4);
        assert_eq!(pts.len(), 4);
        assert!((pts[0] - DVec3::new(5.0, 0.0, 0.0)).length() < 1e-12);
        assert!((pts[3] - DVec3::new(0.0, 5.0, 0.0)).length() < 1e-12);
    }

    #[test]
    fn carries_a_fit_on_flat_along_its_slope_at_the_edge() {
        // A bowl fitted over ±3 mm, carried on to ±23: past its edge the carried surface runs
        // on along the slope it has there.
        let mut pts = Vec::new();
        for i in 0..25 {
            for j in 0..25 {
                let (x, y) = (-3.0 + 0.25 * i as f64, -3.0 + 0.25 * j as f64);
                pts.push(DVec3::new(x, y, 5.0 + 0.01 * (x * x + y * y)));
            }
        }
        let params = SurfaceFitParams { net: Some(crate::surface_fit::Net { u: 4, v: 4 }), expand_pct: 10.0, ..SurfaceFitParams::default() };
        let fit = fit_surface(&pts, &params, None, PULL).unwrap();
        let reach = |range: [f64; 2]| [range[0] - 20.0, range[1] + 20.0];
        let carried = carried_fit(&fit, reach(fit.u_range), reach(fit.v_range)).unwrap();
        for (u, v) in [(0.0, 0.0), (fit.u_range[1] + 15.0, 0.0), (fit.u_range[0] - 15.0, 3.0), (2.0, fit.v_range[1] + 18.0)] {
            let want = crate::surface_fit::carried_height(&fit, u, v);
            let got = crate::surface_fit::eval_surface(&carried, u, v);
            let got_h = (got - carried.origin).dot(carried.normal);
            // The carried height takes its slope at the edge by a difference; the cubic holds
            // it to a hair.
            assert!((got_h - want).abs() < 1e-4, "at ({u}, {v}): {got_h} against {want}");
        }
    }
}
