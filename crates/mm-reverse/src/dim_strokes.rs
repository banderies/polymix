// SPDX-License-Identifier: AGPL-3.0-only
//! A dimension on a 2D sheet as a drawing carries it. Ported from ScanRuler
//! (`src/core/flat/dimStrokes.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Extension lines off the two measured points, a dimension line between them with a filled
//! arrow at each end, the number written along the line in a gap left for it; a radius as a
//! leader with its arrow on the rim; a diameter as a line through the centre with an arrow on
//! the rim at each end; an angle as an arc between the two lines.
//!
//! Pure: a `DimShape` says what is measured and where the number sits, in sheet millimetres;
//! `dim_strokes` turns it into lines and arrowheads for one zoom — the arrows, the gaps and
//! the overshoot are sized in screen pixels, so the strokes are made again when the zoom
//! changes and the shape is not.

use std::f64::consts::PI;

use glam::DVec2;
use serde::{Deserialize, Serialize};

/// An arc's own start and sweep, for a radius leader that lands beside it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArcSpan {
    pub start: f64,
    pub sweep: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DimShape {
    /// A length or a distance: measured from `p` to `q`, the dimension line `off` along `n`
    /// from them, the number `along` `d` from its middle.
    Linear { p: DVec2, q: DVec2, d: DVec2, n: DVec2, off: f64, along: f64 },
    /// A radius: the leader at `angle` about `c`, the number `reach` from the centre. `span` is
    /// the arc's own, for a leader that lands beside it.
    Radius { c: DVec2, r: f64, angle: f64, reach: f64, span: Option<ArcSpan> },
    /// A diameter: the line through `c` at `angle`, rim to rim, the number `reach` from the
    /// centre the way `angle` points — inside the circle in a gap of the line, or outside at
    /// the end of the line carried on.
    Diameter { c: DVec2, r: f64, angle: f64, reach: f64 },
    /// An angle: the arc of `radius` about `vertex` from `a0` counter-clockwise through
    /// `sweep`, the number at the angle `at`; the two lines' own stretches, for the extension
    /// an arc end beside its line needs.
    Angle { vertex: DVec2, a0: f64, sweep: f64, radius: f64, at: f64, lines: [[DVec2; 2]; 2] },
}

/// A filled arrowhead: the tip, and the way it points.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arrow {
    pub tip: DVec2,
    pub dir: DVec2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimStrokes {
    pub lines: Vec<Vec<DVec2>>,
    pub arrows: Vec<Arrow>,
    /// The middle of the number, and the angle it is written at — along the dimension line,
    /// turned to read from below or from the right.
    pub text: DVec2,
    pub text_angle: f64,
}

/// Screen pixels.
const ARROW_PX: f64 = 11.0;
const EXT_GAP_PX: f64 = 3.0;
const EXT_OVER_PX: f64 = 7.0;
const TEXT_PAD_PX: f64 = 5.0;
const TAIL_PX: f64 = 18.0;

/// The point `r` from `c` at the angle `a`.
pub fn polar(c: DVec2, r: f64, a: f64) -> DVec2 {
    DVec2::new(c.x + r * a.cos(), c.y + r * a.sin())
}

/// An angle into (-π, π].
fn wrap(a: f64) -> f64 {
    let mut x = a % (2.0 * PI);
    if x > PI {
        x -= 2.0 * PI;
    }
    if x <= -PI {
        x += 2.0 * PI;
    }
    x
}

/// Text is never written upside down: the angle into (-π/2, π/2].
fn readable(a: f64) -> f64 {
    let mut x = wrap(a);
    if x > PI / 2.0 {
        x -= PI;
    }
    if x <= -PI / 2.0 {
        x += PI;
    }
    x
}

/// A stretch [lo, hi] of a line or an arc with the number's gap cut out.
fn with_gap(lo: f64, hi: f64, gap_lo: f64, gap_hi: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    if gap_lo > lo {
        out.push((lo, hi.min(gap_lo)));
    }
    if gap_hi < hi {
        out.push((lo.max(gap_hi), hi));
    }
    out.retain(|(a, b)| b - a > 1e-12);
    out
}

fn arc_points(c: DVec2, r: f64, from: f64, to: f64) -> Vec<DVec2> {
    let steps = (((to - from).abs() / 0.08).ceil() as usize).max(2);
    (0..=steps).map(|i| polar(c, r, from + ((to - from) * i as f64) / steps as f64)).collect()
}

/// `Math.sign`: −1, 0 or 1 (a zero keeps its sign), and NaN for NaN — not `f64::signum`,
/// which reads a zero as 1.
fn js_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// A unit vector, or `None` under 1e-12.
fn normalize2(v: DVec2) -> Option<DVec2> {
    let l = v.x.hypot(v.y);
    (l > 1e-12).then(|| v / l)
}

/// The strokes of a dimension at one zoom. `px` is sheet millimetres per screen pixel,
/// `text_px` the width of the number as written.
pub fn dim_strokes(shape: &DimShape, px: f64, text_px: f64) -> DimStrokes {
    let arrow = ARROW_PX * px;
    let tail = TAIL_PX * px;
    let half = (text_px / 2.0 + TEXT_PAD_PX) * px;
    let mut lines: Vec<Vec<DVec2>> = Vec::new();
    let mut arrows: Vec<Arrow> = Vec::new();

    match *shape {
        DimShape::Linear { p, q, d, n, off, along } => {
            let l = (q - p).x.hypot((q - p).y);
            let p1 = p + n * off;
            let mid = (p + q) * 0.5 + n * off;
            if off.abs() > (EXT_GAP_PX + 1.0) * px {
                let s = js_sign(off);
                for from in [p, q] {
                    lines.push(vec![from + n * (s * EXT_GAP_PX * px), from + n * (off + s * EXT_OVER_PX * px)]);
                }
            }
            // The arrows stand inside the stretch while they and the number fit there; on a
            // short one they point in at it from outside.
            let text_inside = along.abs() < l / 2.0;
            let outside = l < 2.4 * arrow || (text_inside && l < 2.0 * half + 2.2 * arrow);
            let mut lo = -l / 2.0;
            let mut hi = l / 2.0;
            if outside {
                lo -= tail;
                hi += tail;
            }
            lo = lo.min(along - half);
            hi = hi.max(along + half);
            for (a, b) in with_gap(lo, hi, along - half, along + half) {
                lines.push(vec![mid + d * a, mid + d * b]);
            }
            let back = d * -1.0;
            arrows.push(Arrow { tip: p1, dir: if outside { d } else { back } });
            arrows.push(Arrow { tip: p1 + d * l, dir: if outside { back } else { d } });
            DimStrokes { lines, arrows, text: mid + d * along, text_angle: readable(d.y.atan2(d.x)) }
        }
        DimShape::Radius { c, r, angle, reach, span } => {
            let u = DVec2::new(angle.cos(), angle.sin());
            let rim = polar(c, r, angle);
            if reach > r {
                // The number outside: the leader comes in to the rim.
                if reach - half > r {
                    lines.push(vec![rim, polar(c, reach - half, angle)]);
                }
                arrows.push(Arrow { tip: rim, dir: u * -1.0 });
            } else {
                for (a, b) in with_gap(0.0, r, reach - half, reach + half) {
                    lines.push(vec![polar(c, a, angle), polar(c, b, angle)]);
                }
                arrows.push(Arrow { tip: rim, dir: u });
            }
            if let Some(span) = span {
                // A leader landing beside the arc meets the arc carried on to it.
                let rel = wrap(angle - (span.start + span.sweep / 2.0));
                let over = rel.abs() - span.sweep / 2.0;
                if over > 0.0 {
                    let end = if rel > 0.0 { span.start + span.sweep } else { span.start };
                    lines.push(arc_points(c, r, end, end + js_sign(rel) * (over + (EXT_OVER_PX * px) / r.max(1e-9))));
                }
            }
            DimStrokes { lines, arrows, text: polar(c, reach, angle), text_angle: readable(angle) }
        }
        DimShape::Diameter { c, r, angle, reach } => {
            let u = DVec2::new(angle.cos(), angle.sin());
            let near = polar(c, r, angle);
            let far = polar(c, -r, angle);
            // On a circle too small for two arrows inside, they point in at the rim from
            // outside, as a short length's do.
            let outside = 2.0 * r < 2.4 * arrow;
            let mut lo = if outside { -r - tail } else { -r };
            let mut hi = if outside { r + tail } else { r };
            lo = lo.min(reach - half);
            hi = hi.max(reach + half);
            for (a, b) in with_gap(lo, hi, reach - half, reach + half) {
                lines.push(vec![polar(c, a, angle), polar(c, b, angle)]);
            }
            arrows.push(Arrow { tip: near, dir: if outside { u * -1.0 } else { u } });
            arrows.push(Arrow { tip: far, dir: if outside { u } else { u * -1.0 } });
            DimStrokes { lines, arrows, text: polar(c, reach, angle), text_angle: readable(angle) }
        }
        DimShape::Angle { vertex, a0, sweep, radius: big_r, at: shape_at, lines: measured } => {
            let mut a0 = a0;
            // The number dragged into the sector across the vertex takes the arc with it: the
            // same angle, read between the lines' other halves.
            let mut rel = wrap(shape_at - (a0 + sweep / 2.0));
            let across = wrap(rel - PI);
            if across.abs() < rel.abs() {
                a0 += PI;
                rel = across;
            }
            let at = sweep / 2.0 + rel;
            let half_a = half / big_r;
            let arrow_a = arrow / big_r;
            let text_inside = at > 0.0 && at < sweep;
            let outside = sweep < 2.4 * arrow_a || (text_inside && sweep < 2.0 * half_a + 2.2 * arrow_a);
            let mut lo = 0.0;
            let mut hi = sweep;
            if outside {
                lo -= tail / big_r;
                hi += tail / big_r;
            }
            lo = f64::min(lo, at - half_a);
            hi = hi.max(at + half_a);
            for (a, b) in with_gap(lo, hi, at - half_a, at + half_a) {
                lines.push(arc_points(vertex, big_r, a0 + a, a0 + b));
            }
            // An arrowhead is straight and the arc is not: it is laid along the chord of the
            // stretch of arc it covers, so its tail sits on the arc too.
            let mut head = |end: f64, inward: f64| {
                let chord = end + (inward * arrow_a) / 2.0;
                let t = DVec2::new(-(a0 + chord).sin(), (a0 + chord).cos());
                arrows.push(Arrow { tip: polar(vertex, big_r, a0 + end), dir: t * -inward });
            };
            head(0.0, if outside { -1.0 } else { 1.0 });
            head(sweep, if outside { 1.0 } else { -1.0 });
            // An arc end that lies off its line is met by the line carried on to it.
            let ends = [(a0, measured[0]), (a0 + sweep, measured[1])];
            for (angle, [la, lb]) in ends {
                let e = polar(vertex, big_r, angle);
                let Some(dir) = normalize2(lb - la) else { continue };
                let t = (e - la).dot(dir);
                let length = (lb - la).x.hypot((lb - la).y);
                if t >= 0.0 && t <= length {
                    continue;
                }
                let from = if t < 0.0 { la } else { lb };
                if let Some(out) = normalize2(e - from) {
                    lines.push(vec![from + out * (EXT_GAP_PX * px), e + out * (EXT_OVER_PX * px)]);
                }
            }
            DimStrokes { lines, arrows, text: polar(vertex, big_r, a0 + at), text_angle: readable(a0 + at + PI / 2.0) }
        }
    }
}
