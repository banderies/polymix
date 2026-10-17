//! Fit Primitive: replace a selected region by an ideal shape fitted to it (plane patches,
//! cylinder, spheres, disc, linear sweep, convex hull, height field).
//!
//! Planes come from a principal-component fit of the region (area weighted); a cylinder's
//! axis is the direction the surface normals are perpendicular to, its radius a circle fit
//! of the cross-section; spheres come from an algebraic fit refined by Gauss–Newton. The
//! convex hull is built incrementally with exact orientation tests.

use crate::{EditError, EditResult};
use mm_core::linalg::{add_outer, sym_eigen3};
use mm_core::{primitives, Bvh, DMat3, DVec2, DVec3, Mesh, Ray};
use std::collections::HashMap;

/// Shape to fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    Square,
    Rectangle,
    Cylinder,
    SphereFromPlane,
    SphereBestFit,
    DiscFromPlane,
    LinearSweep,
    ConvexHull,
    HeightField,
}

impl PrimitiveType {
    pub const ALL: [PrimitiveType; 9] = [
        PrimitiveType::Square,
        PrimitiveType::Rectangle,
        PrimitiveType::Cylinder,
        PrimitiveType::SphereFromPlane,
        PrimitiveType::SphereBestFit,
        PrimitiveType::DiscFromPlane,
        PrimitiveType::LinearSweep,
        PrimitiveType::ConvexHull,
        PrimitiveType::HeightField,
    ];
    pub fn label(self) -> &'static str {
        match self {
            PrimitiveType::Square => "Square",
            PrimitiveType::Rectangle => "Rectangle",
            PrimitiveType::Cylinder => "Cylinder",
            PrimitiveType::SphereFromPlane => "Sphere from plane",
            PrimitiveType::SphereBestFit => "Sphere best-fit",
            PrimitiveType::DiscFromPlane => "Disc from plane",
            PrimitiveType::LinearSweep => "Linear Sweep",
            PrimitiveType::ConvexHull => "Convex Hull",
            PrimitiveType::HeightField => "Height Field",
        }
    }
}

/// A fitted primitive.
#[derive(Clone, Debug)]
pub struct FitResult {
    pub mesh: Mesh,
    /// Human-readable fitted dimensions (e.g. "cylinder r 1.52, length 12.3").
    pub description: String,
}

/// Points of a region with area weights and normals.
struct Region {
    pts: Vec<DVec3>,
    /// Triangle centroids, areas and unit normals.
    tris: Vec<(DVec3, f64, DVec3)>,
}

impl Region {
    fn new(mesh: &Mesh, tris: &[u32]) -> Self {
        let mut vs: Vec<u32> = tris.iter().flat_map(|&t| mesh.triangle(t)).collect();
        vs.sort_unstable();
        vs.dedup();
        Self {
            pts: vs.iter().map(|&v| mesh.vertex(v)).collect(),
            tris: tris.iter().map(|&t| (mesh.tri_centroid(t), mesh.tri_area(t), mesh.tri_normal(t))).collect(),
        }
    }
    fn centroid(&self) -> DVec3 {
        let w: f64 = self.tris.iter().map(|t| t.1).sum();
        if w > 0.0 {
            self.tris.iter().map(|t| t.0 * t.1).sum::<DVec3>() / w
        } else {
            self.pts.iter().sum::<DVec3>() / self.pts.len().max(1) as f64
        }
    }
    fn mean_normal(&self) -> DVec3 {
        self.tris.iter().map(|t| t.2 * t.1).sum::<DVec3>()
    }
    /// Principal axes of the points (eigenvalues ascending).
    fn pca(&self, c: DVec3) -> ([f64; 3], [DVec3; 3]) {
        let mut a = [[0.0; 3]; 3];
        for &p in &self.pts {
            add_outer(&mut a, p - c, 1.0);
        }
        sym_eigen3(a)
    }
}

/// Plane fit: centre, unit normal (facing like the region) and in-plane axes `u` (longest
/// spread) and `v`, right-handed with the normal.
fn plane_frame(r: &Region) -> (DVec3, DVec3, DVec3, DVec3) {
    let c = r.centroid();
    let (_, vecs) = r.pca(c);
    let mut n = vecs[0];
    if n.dot(r.mean_normal()) < 0.0 {
        n = -n;
    }
    let u = vecs[2];
    (c, n, u, n.cross(u))
}

/// Coordinates of the points in the plane frame.
fn project(r: &Region, c: DVec3, u: DVec3, v: DVec3) -> Vec<DVec2> {
    r.pts.iter().map(|&p| DVec2::new((p - c).dot(u), (p - c).dot(v))).collect()
}

fn bounds2(pts: &[DVec2]) -> (DVec2, DVec2) {
    pts.iter().fold((DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)), |(lo, hi), &p| (lo.min(p), hi.max(p)))
}

/// Place a mesh built in a local frame (x = u, y = v, z = n) at `c`.
fn place(mut m: Mesh, c: DVec3, u: DVec3, v: DVec3, n: DVec3) -> Mesh {
    let r = DMat3::from_cols(u, v, n);
    m.transform(|p| c + r * p);
    if r.determinant() < 0.0 {
        m.reverse_orientation();
    }
    m
}

/// Least-squares circle through 2D points: algebraic fit, then geometric refinement.
fn fit_circle(pts: &[DVec2]) -> (DVec2, f64) {
    // x² + y² + D x + E y + F = 0.
    let mut ata = [[0.0; 3]; 3];
    let mut atb = DVec3::ZERO;
    for p in pts {
        let row = DVec3::new(p.x, p.y, 1.0);
        add_outer(&mut ata, row, 1.0);
        atb += row * -(p.length_squared());
    }
    let m = DMat3::from_cols_array_2d(&ata).transpose();
    let sol = if m.determinant().abs() > 1e-300 { m.inverse() * atb } else { DVec3::ZERO };
    let mut c = DVec2::new(-sol.x / 2.0, -sol.y / 2.0);
    let mut r = (c.length_squared() - sol.z).max(0.0).sqrt();
    // Gauss–Newton on Σ (|p − c| − r)².
    for _ in 0..20 {
        let mut jtj = [[0.0; 3]; 3];
        let mut jtr = DVec3::ZERO;
        for &p in pts {
            let d = p - c;
            let l = d.length().max(1e-300);
            let res = l - r;
            let j = DVec3::new(-d.x / l, -d.y / l, -1.0);
            add_outer(&mut jtj, j, 1.0);
            jtr += j * res;
        }
        let a = DMat3::from_cols_array_2d(&jtj).transpose();
        if a.determinant().abs() < 1e-300 {
            break;
        }
        let step = a.inverse() * jtr;
        c -= DVec2::new(step.x, step.y);
        r -= step.z;
        if step.length() < 1e-12 * (1.0 + r.abs()) {
            break;
        }
    }
    (c, r.abs())
}

/// Least-squares sphere: algebraic fit, then geometric refinement.
fn fit_sphere(pts: &[DVec3]) -> (DVec3, f64) {
    // x² + y² + z² + D x + E y + F z + G = 0, solved by normal equations (4×4).
    let mut ata = [[0.0f64; 4]; 4];
    let mut atb = [0.0f64; 4];
    for p in pts {
        let row = [p.x, p.y, p.z, 1.0];
        for i in 0..4 {
            for j in 0..4 {
                ata[i][j] += row[i] * row[j];
            }
            atb[i] -= row[i] * p.length_squared();
        }
    }
    let sol = solve4(ata, atb).unwrap_or([0.0; 4]);
    let mut c = DVec3::new(-sol[0] / 2.0, -sol[1] / 2.0, -sol[2] / 2.0);
    let mut r = (c.length_squared() - sol[3]).max(0.0).sqrt();
    for _ in 0..20 {
        let mut jtj = [[0.0f64; 4]; 4];
        let mut jtr = [0.0f64; 4];
        for &p in pts {
            let d = p - c;
            let l = d.length().max(1e-300);
            let res = l - r;
            let j = [-d.x / l, -d.y / l, -d.z / l, -1.0];
            for a in 0..4 {
                for b in 0..4 {
                    jtj[a][b] += j[a] * j[b];
                }
                jtr[a] += j[a] * res;
            }
        }
        let Some(step) = solve4(jtj, jtr) else { break };
        c -= DVec3::new(step[0], step[1], step[2]);
        r -= step[3];
        if step.iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-12 * (1.0 + r.abs()) {
            break;
        }
    }
    (c, r.abs())
}

/// Solve a 4×4 system by Gaussian elimination with partial pivoting.
fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let piv = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..4 {
            let f = a[row][col] / a[col][col];
            let pivot = a[col];
            for (x, p) in a[row].iter_mut().zip(pivot).skip(col) {
                *x -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 4];
    for row in (0..4).rev() {
        let s: f64 = (row + 1..4).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    Some(x)
}

/// 2D convex hull (counter-clockwise, monotone chain).
fn hull2(pts: &[DVec2]) -> Vec<DVec2> {
    let mut p: Vec<DVec2> = pts.to_vec();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: DVec2, a: DVec2, b: DVec2| (a - o).perp_dot(b - o);
    let mut lower: Vec<DVec2> = Vec::new();
    for &q in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 0.0 {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<DVec2> = Vec::new();
    for &q in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 0.0 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Closed prism: the polygon `ring` (CCW) extruded from z = 0 to z = `len`.
fn prism(ring: &[DVec2], len: f64) -> Mesh {
    let n = ring.len();
    let mut pos: Vec<DVec3> = Vec::new();
    for z in [0.0, len] {
        pos.extend(ring.iter().map(|p| DVec3::new(p.x, p.y, z)));
    }
    let c = ring.iter().sum::<DVec2>() / n as f64;
    pos.push(DVec3::new(c.x, c.y, 0.0));
    pos.push(DVec3::new(c.x, c.y, len));
    let (b0, b1) = (2 * n as u32, 2 * n as u32 + 1);
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for i in 0..n as u32 {
        let j = (i + 1) % n as u32;
        let (a, b, c2, d) = (i, j, n as u32 + i, n as u32 + j);
        tris.push([a, b, d]);
        tris.push([a, d, c2]);
        tris.push([b0, j, i]);
        tris.push([b1, c2, d]);
    }
    Mesh::from_triangles(&pos, &tris, None).0
}

/// Flat disc of `radius` in the XY plane facing +Z, `rings` rings deep.
fn disc(radius: f64, slices: usize, rings: usize) -> Mesh {
    let (slices, rings) = (slices.max(3), rings.max(1));
    let mut pos = vec![DVec3::ZERO];
    for k in 1..=rings {
        let r = radius * k as f64 / rings as f64;
        for i in 0..slices {
            let a = i as f64 / slices as f64 * std::f64::consts::TAU;
            pos.push(DVec3::new(r * a.cos(), r * a.sin(), 0.0));
        }
    }
    let at = |k: usize, i: usize| (1 + (k - 1) * slices + i % slices) as u32;
    let mut tris = Vec::new();
    for i in 0..slices {
        tris.push([0, at(1, i), at(1, i + 1)]);
    }
    for k in 1..rings {
        for i in 0..slices {
            let (a, b, c, d) = (at(k, i), at(k, i + 1), at(k + 1, i), at(k + 1, i + 1));
            tris.push([a, c, d]);
            tris.push([a, d, b]);
        }
    }
    Mesh::from_triangles(&pos, &tris, None).0
}

/// 3D convex hull of `pts` (closed, outward). Incremental: each point outside the hull
/// replaces the faces it sees by a fan over the horizon. Orientation tests are exact.
pub fn convex_hull(pts: &[DVec3]) -> Option<Mesh> {
    let mut p: Vec<DVec3> = pts.to_vec();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)).then(a.z.total_cmp(&b.z)));
    p.dedup();
    let orient = |a: DVec3, b: DVec3, c: DVec3, d: DVec3| {
        let c3 = |q: DVec3| robust::Coord3D { x: q.x, y: q.y, z: q.z };
        // > 0 when d is above the plane of (a, b, c) taken counter-clockwise.
        -robust::orient3d(c3(a), c3(b), c3(c), c3(d))
    };
    // Initial tetrahedron from extreme points.
    let i0 = 0;
    let i1 = (1..p.len()).max_by(|&a, &b| p[a].distance_squared(p[i0]).total_cmp(&p[b].distance_squared(p[i0])))?;
    let line = p[i1] - p[i0];
    let i2 = (0..p.len()).max_by(|&a, &b| line.cross(p[a] - p[i0]).length_squared().total_cmp(&line.cross(p[b] - p[i0]).length_squared()))?;
    let i3 = (0..p.len()).max_by(|&a, &b| orient(p[i0], p[i1], p[i2], p[a]).abs().total_cmp(&orient(p[i0], p[i1], p[i2], p[b]).abs()))?;
    if orient(p[i0], p[i1], p[i2], p[i3]) == 0.0 {
        return None;
    }
    let mut faces: Vec<[usize; 3]> =
        if orient(p[i0], p[i1], p[i2], p[i3]) < 0.0 { vec![[i0, i1, i2], [i0, i3, i1], [i1, i3, i2], [i2, i3, i0]] } else { vec![[i0, i2, i1], [i0, i1, i3], [i1, i2, i3], [i2, i0, i3]] };
    for (k, &q) in p.iter().enumerate() {
        if [i0, i1, i2, i3].contains(&k) {
            continue;
        }
        let visible: Vec<bool> = faces.iter().map(|f| orient(p[f[0]], p[f[1]], p[f[2]], q) > 0.0).collect();
        if !visible.contains(&true) {
            continue;
        }
        // Horizon: directed edges of visible faces whose twin belongs to a hidden face.
        let mut seen: HashMap<(usize, usize), bool> = HashMap::new();
        for (f, &vis) in faces.iter().zip(&visible) {
            for j in 0..3 {
                seen.insert((f[j], f[(j + 1) % 3]), vis);
            }
        }
        let mut next: Vec<[usize; 3]> = Vec::new();
        for (f, &vis) in faces.iter().zip(&visible) {
            if !vis {
                next.push(*f);
                continue;
            }
            for j in 0..3 {
                let (a, b) = (f[j], f[(j + 1) % 3]);
                if seen.get(&(b, a)) == Some(&false) {
                    next.push([a, b, k]);
                }
            }
        }
        faces = next;
    }
    let tris: Vec<[u32; 3]> = faces.iter().map(|f| [f[0] as u32, f[1] as u32, f[2] as u32]).collect();
    let mut m = Mesh::from_triangles(&p, &tris, None).0;
    m = m.compacted().0;
    Some(m)
}

/// Fit `kind` to the triangles `tris` of `mesh`; `density` sets the resolution of the
/// generated surface (Meshmixer's Mesh Density, 32 by default).
pub fn fit_primitive(mesh: &Mesh, tris: &[u32], kind: PrimitiveType, density: usize) -> EditResult<FitResult> {
    let tris: Vec<u32> = tris.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
    if tris.is_empty() {
        return Err(EditError::EmptySelection);
    }
    let r = Region::new(mesh, &tris);
    if r.pts.len() < 3 {
        return Err(EditError::Failed("too few points to fit".into()));
    }
    let n = density.clamp(2, 512);
    let res = match kind {
        PrimitiveType::Square | PrimitiveType::Rectangle => {
            let (c, nrm, u, v) = plane_frame(&r);
            let (lo, hi) = bounds2(&project(&r, c, u, v));
            let mid = (lo + hi) * 0.5;
            let mut size = hi - lo;
            if kind == PrimitiveType::Square {
                size = DVec2::splat(size.max_element());
            }
            let grid = primitives::plane_grid(size.x, size.y, n, n);
            let centre = c + u * mid.x + v * mid.y;
            FitResult { mesh: place(grid, centre, u, v, nrm), description: format!("{} {:.4} × {:.4}", kind.label().to_lowercase(), size.x, size.y) }
        }
        PrimitiveType::DiscFromPlane => {
            let (c, nrm, u, v) = plane_frame(&r);
            let p2 = project(&r, c, u, v);
            let (cc, _) = fit_circle(&p2);
            let radius = p2.iter().map(|p| p.distance(cc)).fold(0.0, f64::max);
            let centre = c + u * cc.x + v * cc.y;
            FitResult { mesh: place(disc(radius, 4 * n, n / 2), centre, u, v, nrm), description: format!("disc r {radius:.4}") }
        }
        PrimitiveType::SphereFromPlane => {
            let (c, nrm, u, v) = plane_frame(&r);
            let (cc, radius) = fit_circle(&project(&r, c, u, v));
            let centre = c + u * cc.x + v * cc.y;
            let s = primitives::uv_sphere(radius, 2 * n, n);
            FitResult { mesh: place(s, centre, u, v, nrm), description: format!("sphere r {radius:.4}") }
        }
        PrimitiveType::SphereBestFit => {
            let (centre, radius) = fit_sphere(&r.pts);
            if !(radius.is_finite() && radius > 0.0) {
                return Err(EditError::Failed("no sphere fits the selection".into()));
            }
            let mut s = primitives::uv_sphere(radius, 2 * n, n);
            s.transform(|p| p + centre);
            FitResult { mesh: s, description: format!("sphere r {radius:.4}") }
        }
        PrimitiveType::Cylinder => {
            // Axis: the direction the normals are (least) spread along.
            let mut a = [[0.0; 3]; 3];
            for &(_, w, nn) in &r.tris {
                add_outer(&mut a, nn, w);
            }
            let (_, vecs) = sym_eigen3(a);
            let axis = vecs[0];
            let (u, v) = mm_core::geom::orthonormal_basis(axis);
            let c = r.centroid();
            let p2: Vec<DVec2> = r.pts.iter().map(|&p| DVec2::new((p - c).dot(u), (p - c).dot(v))).collect();
            let (cc, radius) = fit_circle(&p2);
            let along: Vec<f64> = r.pts.iter().map(|&p| (p - c).dot(axis)).collect();
            let (z0, z1) = along.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &z| (lo.min(z), hi.max(z)));
            let len = (z1 - z0).max(1e-9);
            let cyl = primitives::cylinder(radius, len, 2 * n, (n / 4).max(1), true);
            let base = c + u * cc.x + v * cc.y + axis * z0;
            FitResult { mesh: place(cyl, base, u, v, axis), description: format!("cylinder r {radius:.4}, length {len:.4}") }
        }
        PrimitiveType::LinearSweep => {
            let c = r.centroid();
            let (_, vecs) = r.pca(c);
            let axis = vecs[2];
            let (u, v) = mm_core::geom::orthonormal_basis(axis);
            let p2: Vec<DVec2> = r.pts.iter().map(|&p| DVec2::new((p - c).dot(u), (p - c).dot(v))).collect();
            let ring = hull2(&p2);
            if ring.len() < 3 {
                return Err(EditError::Failed("the selection has no cross-section to sweep".into()));
            }
            let along: Vec<f64> = r.pts.iter().map(|&p| (p - c).dot(axis)).collect();
            let (z0, z1) = along.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &z| (lo.min(z), hi.max(z)));
            FitResult { mesh: place(prism(&ring, (z1 - z0).max(1e-9)), c + axis * z0, u, v, axis), description: format!("sweep length {:.4}", z1 - z0) }
        }
        PrimitiveType::ConvexHull => {
            let m = convex_hull(&r.pts).ok_or_else(|| EditError::Failed("the selection is flat; no hull".into()))?;
            let d = format!("convex hull, {} triangles", m.triangle_count());
            FitResult { mesh: m, description: d }
        }
        PrimitiveType::HeightField => {
            let (c, nrm, u, v) = plane_frame(&r);
            let (lo, hi) = bounds2(&project(&r, c, u, v));
            let (sub, _) = mesh.submesh(&tris);
            let bvh = Bvh::build(&sub);
            let top = r.pts.iter().map(|&p| (p - c).dot(nrm)).fold(f64::NEG_INFINITY, f64::max) + 1.0;
            let span = (hi - lo).max(DVec2::splat(1e-9));
            let mut pos = Vec::new();
            let mut id: HashMap<(usize, usize), u32> = HashMap::new();
            for j in 0..=n {
                for i in 0..=n {
                    // Inset by a hair: rays exactly on the region's outline can miss it.
                    let f = |k: usize| 1e-7 + (1.0 - 2e-7) * k as f64 / n as f64;
                    let q = lo + span * DVec2::new(f(i), f(j));
                    let from = c + u * q.x + v * q.y + nrm * top;
                    if let Some(h) = bvh.raycast(&sub, &Ray::new(from, -nrm)) {
                        id.insert((i, j), pos.len() as u32);
                        pos.push(h.point);
                    }
                }
            }
            let mut out = Vec::new();
            for j in 0..n {
                for i in 0..n {
                    let q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)].map(|k| id.get(&k).copied());
                    if let [Some(a), Some(b), Some(c2), Some(d)] = q {
                        out.push([a, b, c2]);
                        out.push([a, c2, d]);
                    }
                }
            }
            if out.is_empty() {
                return Err(EditError::Failed("the height field missed the selection".into()));
            }
            let mut m = Mesh::from_triangles(&pos, &out, None).0;
            // Grid triangles are counter-clockwise around +normal only if (u, v, nrm) is
            // right-handed, which plane_frame guarantees.
            m = m.compacted().0;
            FitResult { mesh: m, description: format!("height field {n} × {n}") }
        }
    };
    Ok(res)
}
