// SPDX-License-Identifier: AGPL-3.0-only
//! Synthetic scans for the fillet readers' tests. Ported from ScanRuler
//! (`plugins/reverse/tests/sweptScan.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A profile (w out from an outline, z up) swept round a closed outline in XY — a circle for a
//! boss, a hole or a bowl, a rounded rectangle for a plate — with noise along the normals. The
//! scans are indexed meshes, their positions rounded to single precision as ScanRuler's
//! `Float32Array`s hold them.

use std::f64::consts::PI;

use glam::{DVec2, DVec3};

use crate::fillet_num::js_round;
use crate::fit2d::Mulberry32;
use crate::slice::{slice_mesh, SectionCut};

/// A profile point (w out from the outline, z up) and its outward normal.
#[derive(Debug, Clone, Copy)]
pub struct Prof {
    pub p: DVec2,
    pub n: DVec2,
}

/// An outline point in XY and its outward normal.
pub type Outline = Vec<Prof>;

/// An indexed mesh.
#[derive(Debug, Clone, Default)]
pub struct Scan {
    pub positions: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
}

impl Scan {
    /// The scan's cut by a plane, as the mesh worker's section gives it.
    pub fn section(&self) -> impl FnMut(DVec3, DVec3, f64) -> SectionCut + '_ {
        move |origin, normal, min_length| slice_mesh(&self.positions, &self.triangles, origin, normal, min_length)
    }

    /// Two scans as one, as a triangle soup of both is welded.
    pub fn and(mut self, other: &Scan) -> Scan {
        let base = self.positions.len() as u32;
        self.positions.extend(other.positions.iter().copied());
        self.triangles.extend(other.triangles.iter().map(|t| t.map(|v| v + base)));
        self
    }

    /// The radius of the scan's bounding sphere: the part's size, as ScanRuler takes it.
    pub fn size(&self) -> f64 {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in &self.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (hi - lo).length() / 2.0
    }
}

/// A double held as a single, as a `Float32Array` holds it.
pub fn f32r(v: DVec3) -> DVec3 {
    DVec3::new(v.x as f32 as f64, v.y as f32 as f64, v.z as f32 as f64)
}

/// The tests' Gaussian draw, as ScanRuler's tests write it.
pub fn gauss(rand: &mut Mulberry32) -> f64 {
    let u = rand.next_f64().max(1e-12);
    (-2.0 * u.ln()).sqrt() * (2.0 * PI * rand.next_f64()).cos()
}

/// A straight run of a profile, sampled every `step`.
pub fn line(a: [f64; 2], b: [f64; 2], n: [f64; 2], step: f64) -> Vec<Prof> {
    let (a, b, n) = (DVec2::from(a), DVec2::from(b), DVec2::from(n));
    let k = js_round((b - a).length() / step).max(1.0) as usize;
    (0..k).map(|i| Prof { p: DVec2::new(a.x + (b.x - a.x) * i as f64 / k as f64, a.y + (b.y - a.y) * i as f64 / k as f64), n }).collect()
}

/// An arc about c of radius r from angle a0 to a1; `inward` for a concave one, whose outward
/// normal points at its centre.
pub fn arc(c: [f64; 2], r: f64, a0: f64, a1: f64, inward: bool, step: f64) -> Vec<Prof> {
    let c = DVec2::from(c);
    let k = js_round((a1 - a0).abs() * r / step).max(2.0) as usize;
    (0..k)
        .map(|i| {
            let a = a0 + (a1 - a0) * i as f64 / k as f64;
            let u = DVec2::new(a.cos(), a.sin());
            Prof { p: DVec2::new(c.x + r * u.x, c.y + r * u.y), n: if inward { -u } else { u } }
        })
        .collect()
}

/// A circle outline of radius `r` about `c`, `k` points.
pub fn circle_outline(r: f64, k: usize, c: [f64; 2]) -> Outline {
    (0..k)
        .map(|i| {
            let a = 2.0 * PI * i as f64 / k as f64;
            Prof { p: DVec2::new(c[0] + r * a.cos(), c[1] + r * a.sin()), n: DVec2::new(a.cos(), a.sin()) }
        })
        .collect()
}

/// A rectangle 2a × 2b with corners rounded rc, every `step`.
pub fn rounded_rect(a: f64, b: f64, rc: f64, step: f64) -> Outline {
    let mut out: Outline = Vec::new();
    let corners = [DVec2::new(a - rc, b - rc), DVec2::new(-(a - rc), b - rc), DVec2::new(-(a - rc), -(b - rc)), DVec2::new(a - rc, -(b - rc))];
    for (i, c) in corners.iter().enumerate() {
        let a0 = i as f64 * PI / 2.0;
        let mut s = 0.0;
        while s < (PI / 2.0) * rc {
            let t = a0 + s / rc;
            out.push(Prof { p: DVec2::new(c.x + rc * t.cos(), c.y + rc * t.sin()), n: DVec2::new(t.cos(), t.sin()) });
            s += step;
        }
        let next = corners[(i + 1) % 4];
        let e = DVec2::new((a0 + PI / 2.0).cos(), (a0 + PI / 2.0).sin());
        let from = DVec2::new(c.x + rc * e.x, c.y + rc * e.y);
        let to = DVec2::new(next.x + rc * e.x, next.y + rc * e.y);
        let l = (to - from).length();
        let mut s = 0.0;
        while s < l {
            out.push(Prof { p: DVec2::new(from.x + (to.x - from.x) * s / l, from.y + (to.y - from.y) * s / l), n: e });
            s += step;
        }
    }
    out
}

/// The profile swept round the outline: an indexed sheet, each vertex moved along its normal
/// by Gaussian noise.
pub fn sweep(outline: &Outline, profile: &[Prof], noise: f64, seed: u32, closed: bool) -> Scan {
    let mut rand = Mulberry32::new(seed);
    let mut positions = Vec::with_capacity(outline.len() * profile.len());
    for o in outline {
        for Prof { p, n } in profile {
            let d = gauss(&mut rand) * noise;
            let nn = DVec3::new(n.x * o.n.x, n.x * o.n.y, n.y);
            positions.push(f32r(DVec3::new(o.p.x + p.x * o.n.x + d * nn.x, o.p.y + p.x * o.n.y + d * nn.y, p.y + d * nn.z)));
        }
    }
    let pn = profile.len() as u32;
    let mut triangles = Vec::new();
    let rows = if closed { outline.len() } else { outline.len().saturating_sub(1) };
    for i in 0..rows {
        let j = ((i + 1) % outline.len()) as u32;
        let i = i as u32;
        for k in 0..pn.saturating_sub(1) {
            triangles.push([i * pn + k, j * pn + k, j * pn + k + 1]);
            triangles.push([i * pn + k, j * pn + k + 1, i * pn + k + 1]);
        }
    }
    Scan { positions, triangles }
}

/// `sweep` with the defaults ScanRuler's helper has: seed 3, closed.
pub fn sweep3(outline: &Outline, profile: &[Prof], noise: f64) -> Scan {
    sweep(outline, profile, noise, 3, true)
}
