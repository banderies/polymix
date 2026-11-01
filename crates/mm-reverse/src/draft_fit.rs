// SPDX-License-Identifier: AGPL-3.0-only
//! A draft's angle read off the scan ("fit" in place of a Draft feature's angle): the scan
//! inside each face's outline fitted with a plane, then fitted again to the points about that
//! plane, each face's lean against the neutral plane's pull read off it, and the median of the
//! faces taken — with how much of the face's height along the pull the scan spans said, and a
//! warning where it spans under half of it. A spread of the faces' readings past two sigmas is
//! refused, every reading listed, for the faces to be split into features of their own.
//!
//! The scan comes in through a function that hands back its points and their normals in a box
//! (ScanRuler's worker query), so this is plain geometry and testable without the app.
//!
//! Ported from ScanRuler (`plugins/reverse/app/reFit.ts`: `combined`, `footprintOf`,
//! `planeThrough`, `fitDraftFace`, `fitDraftFaces`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::{DVec2, DVec3};

use crate::fit3d::plane::fit_plane_tls;
use crate::intent::{rounding, Rounding};
use crate::nominal::NominalKind;
use crate::round::{number_text, round4};
use crate::topo::{FaceFingerprint, FaceKind};

/// How far off the plane fitted a wall's scan point still counts, mm, at most — the band
/// narrows to three times the points' scatter, not under `WALL_BAND_MIN_MM` — and how far its
/// normal may turn, degrees.
const WALL_BAND_MM: f64 = 1.5;
const WALL_BAND_MIN_MM: f64 = 0.3;
const WALL_TURN_DEG: f64 = 25.0;
/// How far a scan wall may lean off the face picked, degrees, for the first reading: the
/// points are looked for over the face's footprint up to that far off it, and the band besides.
const WALL_LEAN_MAX_DEG: f64 = 15.0;
/// How many times the plane is fitted again to the points about it.
const WALL_PASSES: usize = 3;
/// Under this share of a face's height along the pull, a draft read off the scan is said to
/// rest on a strip of it.
pub const WALL_COVERAGE_WARN: f64 = 0.5;
/// How many scan points a face's reading asks for at most.
pub const SCAN_LIMIT: usize = 40_000;

/// A face a draft reads, as picked, and its triangles off the body's mesh where they are at
/// hand — its outline; `None` where they are not, and a square of its area stands in.
#[derive(Debug, Clone)]
pub struct DraftFaceIn {
    pub fingerprint: FaceFingerprint,
    pub triangles: Option<Vec<[DVec3; 3]>>,
}

/// The scan's points in a box (`lo`, `hi`) with their unit normals, at most `limit` of them.
pub type ScanIn<'a> = &'a dyn Fn(DVec3, DVec3, usize) -> (Vec<DVec3>, Vec<DVec3>);

/// One reading of a number: a face's lean, or why there is none.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub value: Option<f64>,
    pub sigma: Option<f64>,
    pub why: Option<String>,
    /// How much of the face's height along the pull the scan points read span, 0–1.
    pub coverage: Option<f64>,
    pub points: usize,
}

impl Reading {
    fn none(why: String) -> Reading {
        Reading { value: None, sigma: None, why: Some(why), coverage: None, points: 0 }
    }
}

/// A number fitted to the scan, as an answer gives it.
#[derive(Debug, Clone, PartialEq)]
pub struct FitReading {
    pub value: f64,
    pub sigma: f64,
    pub nominal: f64,
    pub off: f64,
    pub sigmas: Option<f64>,
    pub points: usize,
    /// Each face's own reading — the value is their median.
    pub readings: Vec<Reading>,
    /// The least share of a face's height along the pull the scan spans, 0–1.
    pub coverage: Option<f64>,
    /// What makes the reading doubtful: a draft read off a strip of its faces.
    pub warning: Option<String>,
}

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let m = s.len() >> 1;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}

/// Several readings of one number made one: the median, its sigma (the readings' combined, a
/// quarter more for the median), and a refusal when one stands more than two sigmas off the
/// rest — each reading listed, to pick the faces again.
pub fn combined(what: &str, unit: &str, readings: &[Reading], kind: NominalKind) -> Result<FitReading, String> {
    let ok: Vec<(f64, f64)> = readings.iter().filter_map(|r| Some((r.value?, r.sigma?))).collect();
    let listed: Vec<String> = readings
        .iter()
        .enumerate()
        .map(|(i, r)| match r.value {
            None => format!("#{} {}", i + 1, r.why.clone().unwrap_or_else(|| "no reading".into())),
            Some(v) => format!("#{} {} ± {} {unit}", i + 1, number_text(round4(v)), number_text(round4(r.sigma.unwrap_or(0.0)))),
        })
        .collect();
    let listed = listed.join("; ");
    if ok.len() < readings.len() || ok.is_empty() {
        return Err(format!("The {what} cannot be fitted: not every one picked reads — {listed}. Pick only those that do, or give a number."));
    }
    let value = median(&ok.iter().map(|r| r.0).collect::<Vec<_>>());
    let pooled = 1.0 / ok.iter().map(|r| 1.0 / r.1.max(1e-9).powi(2)).sum::<f64>().sqrt();
    let sigma = if ok.len() > 1 { 1.2533 * pooled } else { pooled };
    for (v, s) in &ok {
        if (v - value).abs() > 2.0 * s.hypot(sigma) {
            return Err(format!("The {what} cannot be one number: the readings spread past two sigmas — {listed}. Split them into features of their own, or give a number."));
        }
    }
    let Rounding { nominal, off, sigmas, .. } = rounding(value, kind, Some(sigma));
    let readings = readings.iter().map(|r| Reading { value: r.value.map(round4), sigma: r.sigma.map(round4), ..r.clone() }).collect();
    Ok(FitReading { value: round4(value), sigma: round4(sigma), nominal, off, sigmas, points: 0, readings, coverage: None, warning: None })
}

/// A face's footprint in its own plane — its triangles, or the square of its area about its
/// middle — as a point-in test in (u, w); the box round it; its height along `up`.
struct Footprint {
    uv: Vec<DVec2>,
    boxes: Vec<[f64; 4]>,
    square: Option<f64>,
    c: DVec3,
    u: DVec3,
    w: DVec3,
    lo: DVec3,
    hi: DVec3,
    height: f64,
}

impl Footprint {
    fn of(face: &DraftFaceIn, c: DVec3, u: DVec3, w: DVec3, up: DVec3) -> Footprint {
        let tris = face.triangles.as_ref().filter(|t| !t.is_empty());
        let Some(tris) = tris else {
            let half = face.fingerprint.area.sqrt() / 2.0;
            let corners: Vec<DVec3> = [-1.0, 1.0].iter().flat_map(|a| [-1.0, 1.0].map(|b| c + u * (a * half) + w * (b * half))).collect();
            let along: Vec<f64> = corners.iter().map(|p| p.dot(up)).collect();
            let lo = corners.iter().fold(DVec3::splat(f64::INFINITY), |m, p| m.min(*p));
            let hi = corners.iter().fold(DVec3::splat(f64::NEG_INFINITY), |m, p| m.max(*p));
            let height = along.iter().copied().fold(f64::NEG_INFINITY, f64::max) - along.iter().copied().fold(f64::INFINITY, f64::min);
            return Footprint { uv: Vec::new(), boxes: Vec::new(), square: Some(half), c, u, w, lo, hi, height };
        };
        let mut uv = Vec::with_capacity(tris.len() * 3);
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        let (mut up_lo, mut up_hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for t in tris {
            for p in t {
                let d = *p - c;
                uv.push(DVec2::new(d.dot(u), d.dot(w)));
                lo = lo.min(*p);
                hi = hi.max(*p);
                let a = p.dot(up);
                up_lo = up_lo.min(a);
                up_hi = up_hi.max(a);
            }
        }
        // Each triangle's box in (u, w), to pass over most of them at once.
        let boxes = uv.chunks(3).map(|t| [t[0].x.min(t[1].x).min(t[2].x), t[0].y.min(t[1].y).min(t[2].y), t[0].x.max(t[1].x).max(t[2].x), t[0].y.max(t[1].y).max(t[2].y)]).collect();
        Footprint { uv, boxes, square: None, c, u, w, lo, hi, height: up_hi - up_lo }
    }

    fn inside(&self, p: DVec3) -> bool {
        let d = p - self.c;
        let (x, y) = (d.dot(self.u), d.dot(self.w));
        if let Some(half) = self.square {
            return x.abs() <= half && y.abs() <= half;
        }
        const EPS: f64 = 1e-6;
        for (t, b) in self.boxes.iter().enumerate() {
            if x < b[0] - EPS || y < b[1] - EPS || x > b[2] + EPS || y > b[3] + EPS {
                continue;
            }
            let (a, bb, cc) = (self.uv[t * 3], self.uv[t * 3 + 1], self.uv[t * 3 + 2]);
            let d1 = (x - bb.x) * (a.y - bb.y) - (a.x - bb.x) * (y - bb.y);
            let d2 = (x - cc.x) * (bb.y - cc.y) - (bb.x - cc.x) * (y - cc.y);
            let d3 = (x - a.x) * (cc.y - a.y) - (cc.x - a.x) * (y - a.y);
            if !((d1 < -EPS || d2 < -EPS || d3 < -EPS) && (d1 > EPS || d2 > EPS || d3 > EPS)) {
                return true;
            }
        }
        false
    }
}

/// A plane fitted through the points `idx`, its normal turned to `n`'s side, with the points'
/// scatter about it.
fn plane_through(positions: &[DVec3], idx: &[u32], n: DVec3) -> Option<(DVec3, f64, f64)> {
    if idx.len() < 20 {
        return None;
    }
    let plane = fit_plane_tls(positions, idx)?;
    let (mut normal, mut d) = (plane.normal, plane.d);
    if normal.dot(n) < 0.0 {
        normal = -normal;
        d = -d;
    }
    let sum: f64 = idx.iter().map(|i| (positions[*i as usize].dot(normal) - d).powi(2)).sum();
    Some((normal, d, (sum / idx.len() as f64).sqrt()))
}

/// One face's lean off the scan. The points over its footprint — its outline in its own plane —
/// facing as it does within `WALL_TURN_DEG`, up to `WALL_LEAN_MAX_DEG` of lean off it; a plane
/// through their middle, facing as they do on average; then a plane fitted to the points within
/// the band of it, and again to those within the band of that, the band narrowed to the
/// scatter, `WALL_PASSES` times. Not the band about the face picked: a wall leaned off it
/// crosses that band in a strip, and the strip's noise pulls the fit toward no lean. Answers
/// how much of the face's height along the pull the points span.
pub fn fit_draft_face(face: &DraftFaceIn, pull: (DVec3, DVec3), scan: ScanIn) -> Reading {
    let fp = &face.fingerprint;
    let c = fp.centroid;
    let at = format!("[{}, {}, {}]", number_text(round4(c.x)), number_text(round4(c.y)), number_text(round4(c.z)));
    if fp.kind != FaceKind::Plane {
        return Reading::none(format!("a {} face — only flat faces are drafted", fp.kind.name()));
    }
    let n = fp.dir.try_normalize().unwrap_or(DVec3::Z);
    // Away from the neutral plane, on the face's side of it.
    let s = (c - pull.0).dot(pull.1);
    let side = if s > 0.0 {
        1.0
    } else if s < 0.0 {
        -1.0
    } else {
        1.0
    };
    let away = pull.1 * side;
    // Up the face: the pull as it runs across the face; along it, square.
    let Some(w) = (away - n * away.dot(n)).try_normalize() else { return Reading::none(format!("the face about {at} is square to the pull")) };
    let u = n.cross(w);
    let foot = Footprint::of(face, c, u, w, away);
    let reach = WALL_BAND_MM + foot.height * WALL_LEAN_MAX_DEG.to_radians().tan();
    let (positions, normals) = scan(foot.lo - DVec3::splat(reach), foot.hi + DVec3::splat(reach), SCAN_LIMIT);
    let cos_turn = WALL_TURN_DEG.to_radians().cos();
    let over: Vec<u32> = (0..positions.len().min(normals.len()))
        .filter(|i| {
            let p = positions[*i];
            (p - c).dot(n).abs() <= reach && normals[*i].dot(n) >= cos_turn && foot.inside(p)
        })
        .map(|i| i as u32)
        .collect();
    let too_few = || Reading::none(format!("too few scan points on the face about {at}"));
    if over.len() < 20 {
        return too_few();
    }
    let sum = over.iter().fold(DVec3::ZERO, |a, i| a + normals[*i as usize]);
    let mut normal = sum.try_normalize().unwrap_or(n);
    let mut offsets: Vec<f64> = over.iter().map(|i| positions[*i as usize].dot(normal)).collect();
    offsets.sort_by(|a, b| a.total_cmp(b));
    let mut d = offsets[offsets.len() >> 1];
    let mut band = WALL_BAND_MM;
    let mut idx: Vec<u32> = Vec::new();
    let mut rms = 0.0;
    for _ in 0..=WALL_PASSES {
        let near: Vec<u32> = over.iter().copied().filter(|i| (positions[*i as usize].dot(normal) - d).abs() <= band && normals[*i as usize].dot(normal) >= cos_turn).collect();
        let Some((nn, dd, r)) = plane_through(&positions, &near, n) else { break };
        idx = near;
        normal = nn;
        d = dd;
        rms = r;
        band = WALL_BAND_MM.min(WALL_BAND_MIN_MM.max(3.0 * rms));
    }
    if idx.len() < 20 {
        return too_few();
    }
    let angle = normal.dot(away).clamp(-1.0, 1.0).asin().to_degrees();
    // The angle's sigma: the points' scatter about the plane over their spread along the pull,
    // the direction the lean is read in.
    let mut ups: Vec<f64> = idx.iter().map(|i| positions[*i as usize].dot(away)).collect();
    ups.sort_by(|a, b| a.total_cmp(b));
    let middle = ups.iter().sum::<f64>() / ups.len() as f64;
    let spread: f64 = ups.iter().map(|x| (x - middle).powi(2)).sum();
    let sigma = (spread > 0.0).then(|| rms / spread.sqrt() * (180.0 / std::f64::consts::PI));
    // The span from the 2nd to the 98th percentile, so a stray point does not stretch it.
    let q = |f: f64| ups[crate::round::js_round(f * (ups.len() - 1) as f64) as usize];
    let coverage = if foot.height > 1e-9 { ((q(0.98) - q(0.02)) / foot.height).min(1.0) } else { 1.0 };
    Reading { value: Some(angle), sigma, why: None, coverage: Some(coverage), points: idx.len() }
}

/// A draft's angle off the scan for faces: each face's lean, the median — and the least share
/// of a face's height the scan spans, with a warning naming each face under half. `what` names
/// the number in a refusal ("draft’s angle", "angle of group 1").
pub fn fit_draft_faces(faces: &[DraftFaceIn], pull: (DVec3, DVec3), what: &str, scan: ScanIn) -> Result<FitReading, String> {
    if faces.is_empty() {
        return Err("A fitted draft needs the faces it leans.".into());
    }
    let read: Vec<Reading> = faces.iter().map(|f| fit_draft_face(f, pull, scan)).collect();
    let mut r = combined(what, "°", &read, NominalKind::Draft)?;
    let pct = |x: f64| format!("{} %", crate::round::js_round(x * 100.0));
    let thin: Vec<String> = read
        .iter()
        .enumerate()
        .filter_map(|(i, x)| x.coverage.filter(|c| *c < WALL_COVERAGE_WARN).map(|c| if faces.len() == 1 { pct(c) } else { format!("#{} {}", i + 1, pct(c)) }))
        .collect();
    r.points = read.iter().map(|x| x.points).sum();
    r.readings = r.readings.into_iter().zip(&read).map(|(x, o)| Reading { coverage: o.coverage.map(round4), ..x }).collect();
    r.coverage = read.iter().filter_map(|x| x.coverage).reduce(f64::min).map(round4);
    if !thin.is_empty() {
        r.warning = Some(format!("The scan spans under half the face’s height along the pull ({}): the angle rests on a strip of it — check it on a section across the wall, or give a number.", thin.join(", ")));
    }
    Ok(r)
}

/// The default name of the number a draft's fit reads.
pub const DRAFT_ANGLE: &str = "draft’s angle";

#[cfg(test)]
mod tests {
    //! ScanRuler's `reDraftFit.test.ts`: a draft's angle read where the scan's wall leans well
    //! off the face picked — a block whose +Y wall leans in 9° and whose −Y wall leans in 4.5°
    //! as they rise, against a box whose walls stand upright a millimetre outside the scan's
    //! foot. The scan is built here as points with normals, as the worker hands them over.
    use super::*;
    use crate::fit2d::Mulberry32;

    const PLUS_Y_DEG: f64 = 9.0;
    const MINUS_Y_DEG: f64 = 4.5;

    /// The block's profile in YZ — its foot ±20 at z = −20, its top at z = 20, the +Y wall
    /// leaning in 9° and the −Y wall 4.5° — sampled every 0.4 with noise, run along X ±20 in
    /// 80 strips: the points of its walls with their outward normals.
    fn scan() -> (Vec<DVec3>, Vec<DVec3>) {
        let mut rand = Mulberry32::new(11);
        let mut gauss = move || {
            let a = rand.next_f64().max(1e-12);
            let b = rand.next_f64();
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        };
        let tan = |deg: f64| deg.to_radians().tan();
        let corners = [DVec2::new(-20.0, -20.0), DVec2::new(20.0, -20.0), DVec2::new(20.0 - 40.0 * tan(PLUS_Y_DEG), 20.0), DVec2::new(-20.0 + 40.0 * tan(MINUS_Y_DEG), 20.0)];
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        for c in 0..4 {
            let (a, b) = (corners[c], corners[(c + 1) % 4]);
            let len = (b - a).length();
            let n = DVec2::new((b.y - a.y) / len, -(b.x - a.x) / len);
            let k = (len / 0.4).round().max(1.0) as usize;
            for i in 0..k {
                let d = gauss() * 0.02;
                let p = a + (b - a) * (i as f64 / k as f64) + n * d;
                for s in 0..=80 {
                    positions.push(DVec3::new(-20.0 + 40.0 * s as f64 / 80.0, p.x, p.y));
                    normals.push(DVec3::new(0.0, n.x, n.y));
                }
            }
        }
        (positions, normals)
    }

    fn query<'a>(pts: &'a (Vec<DVec3>, Vec<DVec3>)) -> impl Fn(DVec3, DVec3, usize) -> (Vec<DVec3>, Vec<DVec3>) + 'a {
        move |lo: DVec3, hi: DVec3, limit: usize| {
            let inside: Vec<usize> = (0..pts.0.len()).filter(|i| pts.0[*i].cmpge(lo).all() && pts.0[*i].cmple(hi).all()).collect();
            let step = inside.len().div_ceil(limit.max(1)).max(1);
            let kept: Vec<usize> = inside.into_iter().step_by(step).collect();
            (kept.iter().map(|i| pts.0[*i]).collect(), kept.iter().map(|i| pts.1[*i]).collect())
        }
    }

    /// A box face upright a millimetre outside the scan's foot along Y, `height` tall from the
    /// scan's foot, as a square of its area (no triangles at hand).
    fn wall(y: f64, height: f64) -> DraftFaceIn {
        DraftFaceIn {
            fingerprint: FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(0.0, y, -20.0 + height / 2.0), dir: DVec3::new(0.0, y.signum(), 0.0), area: 40.0 * height, radius: None, center: None, origin: 2 },
            triangles: Some(vec![
                [DVec3::new(-20.0, y, -20.0), DVec3::new(20.0, y, -20.0), DVec3::new(20.0, y, -20.0 + height)],
                [DVec3::new(-20.0, y, -20.0), DVec3::new(20.0, y, -20.0 + height), DVec3::new(-20.0, y, -20.0 + height)],
            ]),
        }
    }

    const BASE: (DVec3, DVec3) = (DVec3::new(0.0, 0.0, -20.0), DVec3::Z);

    #[test]
    fn reads_9_degrees_off_a_face_standing_upright_a_millimetre_outside_the_walls_foot() {
        let pts = scan();
        let q = query(&pts);
        let a = fit_draft_faces(&[wall(21.0, 40.0)], BASE, DRAFT_ANGLE, &q).unwrap_or_else(|e| panic!("{e}"));
        assert!((a.value - PLUS_Y_DEG).abs() < 0.2, "{}", a.value);
        assert_eq!(a.nominal, PLUS_Y_DEG);
        assert!(a.coverage.unwrap_or(0.0) > 0.9, "{:?}", a.coverage);
        assert!(a.warning.is_none());
        assert!(a.points > 1000, "{}", a.points);
    }

    #[test]
    fn warns_where_the_scan_spans_under_half_the_faces_height() {
        // 100 tall where the scan stands 40: the wall covers 40 % of the face.
        let pts = scan();
        let q = query(&pts);
        let a = fit_draft_faces(&[wall(21.0, 100.0)], BASE, DRAFT_ANGLE, &q).unwrap_or_else(|e| panic!("{e}"));
        assert!((a.value - PLUS_Y_DEG).abs() < 0.2, "{}", a.value);
        let c = a.coverage.unwrap_or(0.0);
        assert!(c > 0.3 && c < 0.5, "{c}");
        let w = a.warning.unwrap_or_default();
        assert!(w.contains("under half the face’s height along the pull (3"), "{w}");
    }

    #[test]
    fn reads_each_walls_own_angle_and_refuses_one_number_for_two_that_differ() {
        let pts = scan();
        let q = query(&pts);
        let minus = fit_draft_faces(&[wall(-21.0, 40.0)], BASE, "angle of group 1", &q).unwrap_or_else(|e| panic!("{e}"));
        assert!((minus.value - MINUS_Y_DEG).abs() < 0.2, "{}", minus.value);
        let both = fit_draft_faces(&[wall(21.0, 40.0), wall(-21.0, 40.0)], BASE, DRAFT_ANGLE, &q).unwrap_err();
        assert!(both.starts_with("The draft’s angle cannot be one number: the readings spread past two sigmas — #1 "), "{both}");
        // A face with no scan over it does not read.
        let far = DraftFaceIn { fingerprint: FaceFingerprint { centroid: DVec3::new(200.0, 21.0, 0.0), ..wall(21.0, 40.0).fingerprint }, triangles: None };
        let e = fit_draft_faces(&[far], BASE, DRAFT_ANGLE, &q).unwrap_err();
        assert!(e.contains("not every one picked reads — #1 too few scan points on the face about [200, 21, 0]"), "{e}");
    }
}
