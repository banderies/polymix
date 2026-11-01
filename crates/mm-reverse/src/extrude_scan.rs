// SPDX-License-Identifier: AGPL-3.0-only
//! An extrude's end read off the scan ("To scan"): the face of the scan capping the extruded
//! region, and its offset along the extrusion — the farthest such face for a body made (the top
//! of a boss), the nearest for a cut (the floor of a pocket), or the one nearest a height given.
//! The feature keeps the number as a distance, as one typed; a project holds nothing new. Ported
//! from ScanRuler (`plugins/reverse/app/reFit.ts` fitExtrudeEnd), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::frame::SectionFrame;
use crate::intent::{round4, rounding};
use crate::nominal::NominalKind;
use crate::regions::point_in_polygon;

/// How far inside a region's outline a scan point has to lie to be read for the end — clear of
/// the walls' rounds.
pub const END_INSET_MM: f64 = 0.5;
/// A gap along the extrusion that splits the points into two faces, mm.
pub const END_GAP_MM: f64 = 0.3;
/// How far a cap's normal may turn from the extrusion's, degrees.
pub const END_TURN_DEG: f64 = 15.0;

/// A region to read the end over: its outer loop and its holes, on the sketch's sheet.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EndRegion {
    pub outer: Vec<DVec2>,
    pub holes: Vec<Vec<DVec2>>,
}

/// What is asked: where the extrusion starts and which way it looks, what it does, and about
/// where the end is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndAsk {
    pub from: f64,
    /// Against the normal.
    pub reverse: bool,
    /// A cut ends on the floor of what it takes out, which faces back.
    pub cut: bool,
    /// About where the end is, mm along the normal: the scan face nearest this.
    pub near: Option<f64>,
}

/// An extrude's end as the scan gives it: the offset, its sigma, the nominal and how many
/// sigmas off, the points it rests on, and each face met where there were several.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndReading {
    pub to: f64,
    pub value: f64,
    pub sigma: Option<f64>,
    pub nominal: f64,
    pub off: f64,
    pub sigmas: Option<f64>,
    pub points: usize,
    pub rms: f64,
    /// Each face met along the way, by its median offset, where there were several.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub readings: Vec<f64>,
}

fn edge_distance(poly: &[DVec2], p: DVec2) -> f64 {
    let mut d = f64::INFINITY;
    let n = poly.len();
    if n == 0 {
        return d;
    }
    let mut j = n - 1;
    for i in 0..n {
        let a = poly[j];
        let b = poly[i];
        let e = b - a;
        let ll = e.length_squared();
        let t = if ll > 0.0 { ((p - a).dot(e) / ll).clamp(0.0, 1.0) } else { 0.0 };
        d = d.min((p - (a + e * t)).length());
        j = i;
    }
    d
}

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let m = s.len() >> 1;
    if s.is_empty() {
        f64::NAN
    } else if s.len() % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}

/// The box the regions' prism sweeps over `size` either way of the start along `dir`: what of
/// the scan is worth reading.
pub fn end_box(frame: &SectionFrame, dir: DVec3, regions: &[EndRegion], from: f64, size: f64) -> (DVec3, DVec3) {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for r in regions {
        for p in &r.outer {
            for t in [from - size, from + size] {
                let q = frame.origin + frame.u * p.x + frame.v * p.y + dir * t;
                lo = lo.min(q);
                hi = hi.max(q);
            }
        }
    }
    (lo, hi)
}

/// The end of an extrude read off the scan's points and their unit normals (ScanRuler's
/// `fitExtrudeEnd`): every point whose normal faces along the extrusion (back, for a cut) within
/// [`END_TURN_DEG`], past the start, inside a region by [`END_INSET_MM`] — gathered into faces by
/// gaps of [`END_GAP_MM`] along the way, the face chosen as asked, its median the end. Refused,
/// saying so, where no face of the scan caps the region.
pub fn fit_extrude_end(frame: &SectionFrame, dir: DVec3, regions: &[EndRegion], points: &[DVec3], normals: &[DVec3], ask: EndAsk) -> Result<EndReading, String> {
    let dir = dir.try_normalize().unwrap_or(frame.normal);
    let sign = if ask.reverse { -1.0 } else { 1.0 };
    let cap_dir = if ask.cut { -sign } else { sign };
    let cos_turn = END_TURN_DEG.to_radians().cos();
    let n_dot_dir = frame.normal.dot(dir);
    let mut ts: Vec<f64> = Vec::new();
    for (p, m) in points.iter().zip(normals) {
        if m.dot(dir) * cap_dir < cos_turn {
            continue;
        }
        let d = *p - frame.origin;
        let t = d.dot(frame.normal) / n_dot_dir;
        if (t - ask.from) * sign <= 0.05 {
            continue;
        }
        let q = d - dir * t;
        let uv = DVec2::new(q.dot(frame.u), q.dot(frame.v));
        let hit = regions.iter().any(|r| point_in_polygon(&r.outer, uv) && edge_distance(&r.outer, uv) >= END_INSET_MM && !r.holes.iter().any(|h| point_in_polygon(h, uv) || edge_distance(h, uv) < END_INSET_MM));
        if hit {
            ts.push(t);
        }
    }
    // The faces met along the way, by gaps along the extrusion.
    ts.sort_by(|a, b| a.total_cmp(b));
    let mut clusters: Vec<Vec<f64>> = Vec::new();
    for t in &ts {
        match clusters.last_mut() {
            Some(last) if *t - last[last.len() - 1] <= END_GAP_MM => last.push(*t),
            _ => clusters.push(vec![*t]),
        }
    }
    let most = clusters.iter().map(Vec::len).max().unwrap_or(0);
    let floor = 20f64.max(0.1 * most as f64);
    let faces: Vec<&Vec<f64>> = clusters.iter().filter(|c| c.len() as f64 >= floor).collect();
    if faces.is_empty() {
        let n = ts.len();
        return Err(format!("No face of the scan caps the region along the extrusion{}: check the direction (reverse for the other way) and the op.", if n > 0 { format!(" — {n} points, none gathered into a face") } else { String::new() }));
    }
    let along = |c: &[f64]| median(c) * sign;
    let chosen = match ask.near {
        Some(near) => faces.iter().copied().reduce(|a, b| if (median(b) - near).abs() < (median(a) - near).abs() { b } else { a }),
        None if ask.cut => faces.iter().copied().reduce(|a, b| if along(b) < along(a) { b } else { a }),
        None => faces.iter().copied().reduce(|a, b| if along(b) > along(a) { b } else { a }),
    };
    let Some(chosen) = chosen else { return Err("No face of the scan caps the region along the extrusion.".into()) };
    let to = median(chosen);
    let mad = median(&chosen.iter().map(|t| (t - to).abs()).collect::<Vec<_>>());
    let rms = (chosen.iter().map(|t| (t - to).powi(2)).sum::<f64>() / chosen.len() as f64).sqrt();
    // A scan's noise runs together across a face — a stripe, a pass — so its thousands of points
    // are not thousands of readings: no better than a hundred independent ones.
    let s = ((1.4826 * mad) / (chosen.len() as f64).sqrt()).max(rms / 10.0);
    let sigma = (s > 0.0).then_some(s);
    let nominal = rounding(to, NominalKind::Length, sigma);
    Ok(EndReading {
        to,
        value: round4(to),
        sigma: sigma.map(round4),
        nominal: nominal.nominal,
        off: nominal.off,
        sigmas: nominal.sigmas,
        points: chosen.len(),
        rms: round4(rms),
        readings: if faces.len() > 1 { faces.iter().map(|c| round4(median(c))).collect() } else { Vec::new() },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 20 × 20 boss's scan: its top at 12.03 up, a floor at 0, and the walls between, the
    /// points on a grid with their normals.
    fn boss() -> (Vec<DVec3>, Vec<DVec3>) {
        let mut p = Vec::new();
        let mut n = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                let (x, y) = (i as f64 * 0.5, j as f64 * 0.5);
                p.push(DVec3::new(x, y, 12.03 + if (i + j) % 2 == 0 { 0.01 } else { -0.01 }));
                n.push(DVec3::Z);
                p.push(DVec3::new(x - 10.0, y - 10.0, 0.0));
                n.push(DVec3::Z);
                p.push(DVec3::new(0.0, y, x * 0.6));
                n.push(DVec3::new(-1.0, 0.0, 0.0));
            }
        }
        (p, n)
    }

    fn square() -> Vec<EndRegion> {
        vec![EndRegion { outer: vec![DVec2::new(0.0, 0.0), DVec2::new(20.0, 0.0), DVec2::new(20.0, 20.0), DVec2::new(0.0, 20.0)], holes: Vec::new() }]
    }

    fn xy() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }

    #[test]
    fn reads_the_top_of_a_boss_as_the_end_with_its_nominal() {
        let (p, n) = boss();
        let r = fit_extrude_end(&xy(), DVec3::Z, &square(), &p, &n, EndAsk { from: 0.0, reverse: false, cut: false, near: None }).expect("an end");
        assert!((r.to - 12.03).abs() < 0.011, "{r:?}");
        assert_eq!(r.nominal, 12.0);
        assert!(r.points > 1000);
        assert!(r.sigma.is_some());
    }

    #[test]
    fn refuses_where_no_face_caps_the_region_saying_what_to_check() {
        let (p, n) = boss();
        let e = fit_extrude_end(&xy(), DVec3::Z, &square(), &p, &n, EndAsk { from: 0.0, reverse: true, cut: false, near: None });
        assert!(e.is_err_and(|m| m.starts_with("No face of the scan caps the region along the extrusion") && m.ends_with("check the direction (reverse for the other way) and the op.")));
    }
}
