//! Vessel enhancement: multi-scale Hessian filters that respond to tubes, as a volume to
//! threshold or segment. Frangi et al. 1998, Sato et al. 1998 and Jerman et al. 2016 (IEEE
//! TMI 35(9)), each on the eigenvalues of the scale-normalised Hessian (σ² H) of the image
//! smoothed at scale σ; the result is the largest response over the scales, in [0, 1].
//! Frangi suppresses blobs and plates; Sato suppresses plates; Jerman's response is close to
//! uniform across vessel sizes and, by design, also covers blob-like aneurysms.
//!
//! Scales: a vessel of diameter d responds most at σ = d / (2√2) (= radius / √2, the peak of
//! σ² ∂² of a solid cylinder's blurred profile), so the diameter range gives the σ range,
//! spread geometrically.

use crate::filter::gaussian;
use crate::image::{check, Progress};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VesselMethod {
    Frangi,
    Sato,
    Jerman,
}

impl VesselMethod {
    pub const ALL: [VesselMethod; 3] = [VesselMethod::Frangi, VesselMethod::Sato, VesselMethod::Jerman];

    pub fn label(self) -> &'static str {
        match self {
            VesselMethod::Frangi => "Frangi",
            VesselMethod::Sato => "Sato",
            VesselMethod::Jerman => "Jerman",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VesselParams {
    pub method: VesselMethod,
    pub min_diameter_mm: f64,
    pub max_diameter_mm: f64,
    pub scales: usize,
    /// Bright vessels on a dark background (TOF, CTA) or dark vessels (e.g. SWI, black-blood).
    pub bright: bool,
    /// Frangi α, β, c (c ≤ 0 → automatic: half the maximum Hessian norm, as Frangi suggests); Jerman τ.
    pub alpha: f64,
    pub beta: f64,
    pub c: f64,
    pub tau: f64,
}

impl Default for VesselParams {
    fn default() -> Self {
        Self { method: VesselMethod::Frangi, min_diameter_mm: 0.5, max_diameter_mm: 4.0, scales: 4, bright: true, alpha: 0.5, beta: 0.5, c: 0.0, tau: 0.75 }
    }
}

/// Scale σ (mm) at which a vessel of diameter `d` (mm) responds most.
pub fn sigma_for_diameter(d: f64) -> f64 {
    d / (2.0 * std::f64::consts::SQRT_2)
}

/// The scales (σ, mm) for the diameter range, spread geometrically.
pub fn scales(p: &VesselParams) -> Vec<f64> {
    let (a, b) = (sigma_for_diameter(p.min_diameter_mm.max(1e-3)), sigma_for_diameter(p.max_diameter_mm.max(p.min_diameter_mm).max(1e-3)));
    let n = p.scales.max(1);
    if n == 1 || (b - a).abs() < 1e-9 {
        return vec![(a * b).sqrt()];
    }
    (0..n).map(|k| a * (b / a).powf(k as f64 / (n - 1) as f64)).collect()
}

/// Eigenvalues of a symmetric 3 × 3 matrix (analytic, Smith 1961), sorted by magnitude.
pub fn eigenvalues(a00: f64, a01: f64, a02: f64, a11: f64, a12: f64, a22: f64) -> [f64; 3] {
    let p1 = a01 * a01 + a02 * a02 + a12 * a12;
    let mut e = if p1 <= 1e-30 * (a00 * a00 + a11 * a11 + a22 * a22).max(1e-300) {
        [a00, a11, a22]
    } else {
        let q = (a00 + a11 + a22) / 3.0;
        let p2 = (a00 - q).powi(2) + (a11 - q).powi(2) + (a22 - q).powi(2) + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        let (b00, b11, b22) = ((a00 - q) / p, (a11 - q) / p, (a22 - q) / p);
        let (b01, b02, b12) = (a01 / p, a02 / p, a12 / p);
        let det = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
        let phi = (det / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
        let e1 = q + 2.0 * p * phi.cos();
        let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
        [e1, 3.0 * q - e1 - e3, e3]
    };
    e.sort_by(|x, y| x.abs().total_cmp(&y.abs()));
    e
}

/// The scale-normalised Hessian eigenvalues at voxel `q` of the smoothed image `g` (sorted by
/// magnitude), with the sign flipped for dark vessels so vessels always give λ2, λ3 < 0.
fn hessian_eigen(g: &[f32], n: [usize; 3], h: [f64; 3], ijk: [usize; 3], scale2: f64, bright: bool) -> [f64; 3] {
    let at = |di: i64, dj: i64, dk: i64| {
        let i = (ijk[0] as i64 + di).clamp(0, n[0] as i64 - 1) as usize;
        let j = (ijk[1] as i64 + dj).clamp(0, n[1] as i64 - 1) as usize;
        let k = (ijk[2] as i64 + dk).clamp(0, n[2] as i64 - 1) as usize;
        g[(k * n[1] + j) * n[0] + i] as f64
    };
    let c = at(0, 0, 0);
    let dxx = (at(1, 0, 0) - 2.0 * c + at(-1, 0, 0)) / (h[0] * h[0]);
    let dyy = (at(0, 1, 0) - 2.0 * c + at(0, -1, 0)) / (h[1] * h[1]);
    let dzz = (at(0, 0, 1) - 2.0 * c + at(0, 0, -1)) / (h[2] * h[2]);
    let dxy = (at(1, 1, 0) - at(1, -1, 0) - at(-1, 1, 0) + at(-1, -1, 0)) / (4.0 * h[0] * h[1]);
    let dxz = (at(1, 0, 1) - at(1, 0, -1) - at(-1, 0, 1) + at(-1, 0, -1)) / (4.0 * h[0] * h[2]);
    let dyz = (at(0, 1, 1) - at(0, 1, -1) - at(0, -1, 1) + at(0, -1, -1)) / (4.0 * h[1] * h[2]);
    let s = if bright { scale2 } else { -scale2 };
    eigenvalues(dxx * s, dxy * s, dxz * s, dyy * s, dyz * s, dzz * s)
}

/// Run `f` on the eigenvalues of every voxel (in parallel, plane by plane).
fn per_voxel<T: Send>(g: &[f32], n: [usize; 3], h: [f64; 3], scale2: f64, bright: bool, f: impl Fn(usize, [f64; 3]) -> T + Sync) -> Vec<Vec<T>> {
    (0..n[2])
        .into_par_iter()
        .map(|k| {
            let mut out = Vec::with_capacity(n[0] * n[1]);
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let q = (k * n[1] + j) * n[0] + i;
                    out.push(f(q, hessian_eigen(g, n, h, [i, j, k], scale2, bright)));
                }
            }
            out
        })
        .collect()
}

/// Frangi's vesselness (λ1 ≈ 0 along the vessel, λ2, λ3 < 0 across it).
fn frangi(l: [f64; 3], alpha: f64, beta: f64, c: f64) -> f64 {
    let [l1, l2, l3] = l;
    if l2 >= 0.0 || l3 >= 0.0 {
        return 0.0;
    }
    let ra = l2.abs() / l3.abs();
    let rb = l1.abs() / (l2 * l3).abs().sqrt();
    let s2 = l1 * l1 + l2 * l2 + l3 * l3;
    (1.0 - (-ra * ra / (2.0 * alpha * alpha)).exp()) * (-rb * rb / (2.0 * beta * beta)).exp() * (1.0 - (-s2 / (2.0 * c * c)).exp())
}

/// Sato's line measure (γ12 = γ23 = 0.5, α = 0.25 as in the paper), unnormalised.
fn sato(l: [f64; 3]) -> f64 {
    // Sato orders λ1 ≥ λ2 ≥ λ3 (signed): here λ1 is the smallest in magnitude (along the
    // line), and λ2, λ3 must be negative with |λ3| ≥ |λ2|.
    let [l1, l2, l3] = l;
    if l2 >= 0.0 || l3 >= 0.0 {
        return 0.0;
    }
    let base = l3.abs() * (l2 / l3).sqrt();
    if l1 <= 0.0 {
        base * (1.0 + l1 / l2.abs()).max(0.0).sqrt()
    } else if l1 < l2.abs() / 0.25 {
        base * (1.0 - 0.25 * l1 / l2.abs()).max(0.0).sqrt()
    } else {
        0.0
    }
}

/// Jerman's vesselness with the regularised λ3 (`lrho`), in [0, 1].
fn jerman(l: [f64; 3], lrho: f64) -> f64 {
    let (l2, l3) = (-l[1], -l[2]);
    let _ = l3;
    if l2 <= 0.0 || lrho <= 0.0 {
        return 0.0;
    }
    if l2 >= lrho / 2.0 {
        return 1.0;
    }
    l2 * l2 * (lrho - l2) * 27.0 / (l2 + lrho).powi(3)
}

/// Vesselness in [0, 1] per voxel: the largest response over the scales.
pub fn vesselness(vol: &LoadedVolume, p: &VesselParams, progress: Progress) -> anyhow::Result<Vec<f32>> {
    let n = vol.dims;
    let h = vol.geometry.spacing;
    let sigmas = scales(p);
    let smooth = |sigma: f64, f0: f32, f1: f32| -> anyhow::Result<Vec<f32>> {
        let sub = |x: f32| progress(f0 + (f1 - f0) * x);
        gaussian(&vol.values, n, h.map(|sp| sigma / sp), &sub)
    };
    let steps = (sigmas.len() * if p.method == VesselMethod::Frangi && p.c <= 0.0 { 2 } else { 1 }) as f32;
    let mut done = 0.0f32;
    let mut out = vec![0.0f32; vol.values.len()];
    match p.method {
        VesselMethod::Frangi => {
            // c: half the largest Hessian norm over all scales, unless given.
            let c = if p.c > 0.0 {
                p.c
            } else {
                let mut most = 0.0f64;
                for &s in &sigmas {
                    let g = smooth(s, done / steps, (done + 0.7) / steps)?;
                    let norms = per_voxel(&g, n, h, s * s, p.bright, |_, l| l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
                    most = most.max(norms.iter().flatten().fold(0.0f64, |a, &b| a.max(b)).sqrt());
                    done += 1.0;
                    check(progress, done / steps)?;
                }
                (0.5 * most).max(1e-12)
            };
            for &s in &sigmas {
                let g = smooth(s, done / steps, (done + 0.7) / steps)?;
                let v = per_voxel(&g, n, h, s * s, p.bright, |_, l| frangi(l, p.alpha, p.beta, c) as f32);
                for (o, x) in out.iter_mut().zip(v.into_iter().flatten()) {
                    *o = o.max(x);
                }
                done += 1.0;
                check(progress, done / steps)?;
            }
        }
        VesselMethod::Sato => {
            for &s in &sigmas {
                let g = smooth(s, done / steps, (done + 0.7) / steps)?;
                let v = per_voxel(&g, n, h, s * s, p.bright, |_, l| sato(l) as f32);
                for (o, x) in out.iter_mut().zip(v.into_iter().flatten()) {
                    *o = o.max(x);
                }
                done += 1.0;
                check(progress, done / steps)?;
            }
            // Sato's measure has the image's units: scale the strongest response to 1.
            let top = out.iter().fold(0.0f32, |a, &b| a.max(b));
            if top > 0.0 {
                out.par_iter_mut().for_each(|x| *x /= top);
            }
        }
        VesselMethod::Jerman => {
            for &s in &sigmas {
                let g = smooth(s, done / steps, (done + 0.7) / steps)?;
                // λρ: λ3 regularised by τ times its largest value at this scale.
                let l3 = per_voxel(&g, n, h, s * s, p.bright, |_, l| -l[2]);
                let top = l3.iter().flatten().fold(0.0f64, |a, &b| a.max(b));
                let floor = p.tau * top;
                let v = per_voxel(&g, n, h, s * s, p.bright, |_, l| {
                    let l3 = -l[2];
                    let lrho = if l3 > floor { l3 } else if l3 > 0.0 { floor } else { 0.0 };
                    jerman(l, lrho) as f32
                });
                for (o, x) in out.iter_mut().zip(v.into_iter().flatten()) {
                    *o = o.max(x);
                }
                done += 1.0;
                check(progress, done / steps)?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::no_progress;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    fn grid(dims: [usize; 3], f: impl Fn(DVec3) -> f32) -> LoadedVolume {
        let mut v = Vec::new();
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    v.push(f(DVec3::new(i as f64, j as f64, k as f64)));
                }
            }
        }
        let g = Geometry::from_axes([dims[0], dims[1], dims[2], 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        LoadedVolume::from_values("t", dims, v, g)
    }

    /// A tube of radius `r` along z through (16, 16), on a 33 × 33 × 24 grid.
    fn tube(r: f64, bright: bool) -> LoadedVolume {
        grid([33, 33, 24], |p| {
            let inside = ((p.x - 16.0).powi(2) + (p.y - 16.0).powi(2)).sqrt() <= r;
            match (inside, bright) {
                (true, true) | (false, false) => 100.0,
                _ => 10.0,
            }
        })
    }

    #[test]
    fn eigenvalues_of_known_matrices() {
        let e = eigenvalues(2.0, 0.0, 0.0, -5.0, 0.0, 1.0);
        assert_eq!(e, [1.0, 2.0, -5.0]);
        // [[2,1,0],[1,2,0],[0,0,-4]]: 1, 3, -4.
        let e = eigenvalues(2.0, 1.0, 0.0, 2.0, 0.0, -4.0);
        assert!((e[0] - 1.0).abs() < 1e-9 && (e[1] - 3.0).abs() < 1e-9 && (e[2] + 4.0).abs() < 1e-9, "{e:?}");
    }

    #[test]
    fn a_tube_responds_most_at_its_scale() {
        let r = 3.0;
        let vol = tube(r, true);
        let centre = vol.index(16, 16, 12);
        let response = |s: f64| {
            let g = gaussian(&vol.values, vol.dims, [s; 3], no_progress()).unwrap();
            let [l1, l2, l3] = hessian_eigen(&g, vol.dims, [1.0; 3], [16, 16, 12], s * s, true);
            let _ = centre;
            (l1, l2, l3)
        };
        let best = sigma_for_diameter(2.0 * r);
        let at = |s: f64| response(s).2.abs();
        for s in [best * 0.5, best * 0.75, best * 1.33, best * 2.0] {
            assert!(at(best) >= at(s), "σ {s:.2}: {:.2} vs best σ {best:.2}: {:.2}", at(s), at(best));
        }
        let (l1, l2, l3) = response(best);
        assert!(l1.abs() < 0.05 * l3.abs() && l2 < 0.0 && l3 < 0.0, "{l1} {l2} {l3}");
    }

    #[test]
    fn tubes_light_up_spheres_and_plates_do_not() {
        let p = |method| VesselParams { method, min_diameter_mm: 2.0, max_diameter_mm: 8.0, scales: 4, ..Default::default() };
        let ball = grid([33, 33, 33], |q| if (q - DVec3::splat(16.0)).length() <= 5.0 { 100.0 } else { 10.0 });
        let plate = grid([33, 33, 33], |q| if (q.z - 16.0).abs() <= 2.0 { 100.0 } else { 10.0 });
        for method in VesselMethod::ALL {
            let pr = p(method);
            for bright in [true, false] {
                let vol = tube(3.0, bright);
                let v = vesselness(&vol, &VesselParams { bright, ..pr.clone() }, no_progress()).unwrap();
                assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)), "{method:?} out of range");
                let on = v[vol.index(16, 16, 12)];
                let off = v[vol.index(3, 3, 12)];
                eprintln!("{} {}: centreline {on:.2}, background {off:.2}", method.label(), if bright { "bright" } else { "dark" });
                assert!(on > 0.5, "{method:?} bright={bright}: centreline {on}");
                assert!(off < 0.05, "{method:?}: background {off}");
                // The wrong polarity gives nothing on the centreline.
                let wrong = vesselness(&vol, &VesselParams { bright: !bright, ..pr.clone() }, no_progress()).unwrap();
                assert!(wrong[vol.index(16, 16, 12)] < 0.1, "{method:?}: wrong polarity {}", wrong[vol.index(16, 16, 12)]);
            }
            // Sato scales its strongest response to 1 (so its plate and ball are not compared
            // absolutely); Jerman's filter answers blobs (aneurysms) as strongly as vessels on
            // purpose, so only Frangi must ignore the ball.
            let vb = vesselness(&ball, &pr, no_progress()).unwrap();
            let vp = vesselness(&plate, &pr, no_progress()).unwrap();
            let (b, pl) = (vb[ball.index(16, 16, 16)], vp[plate.index(16, 16, 16)]);
            eprintln!("{}: ball {b:.2}, plate {pl:.2}", method.label());
            match method {
                VesselMethod::Frangi => assert!(b < 0.2 && pl < 0.2, "Frangi: ball {b}, plate {pl}"),
                VesselMethod::Jerman => assert!(b > 0.5 && pl < 0.2, "Jerman: ball {b}, plate {pl}"),
                VesselMethod::Sato => assert!(pl < 0.2, "Sato: plate {pl}"),
            }
        }
        assert!(vesselness(&tube(3.0, true), &p(VesselMethod::Frangi), &|_| false).is_err());
    }
}
