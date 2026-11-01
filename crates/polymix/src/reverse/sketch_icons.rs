// SPDX-License-Identifier: AGPL-3.0-only
//! The pictures on the sketch box's keys, drawn from ScanRuler's icon set (`src/ui/icons.tsx`
//! and `plugins/reverse/ui/reIcons.tsx`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only: a
//! 24-unit grid with a 1.5-unit stroke, ink for what is there, the accent for what the key
//! does, dashed for what is only referred to. The pictures are kept as the SVG they were
//! drawn in — paths, circles, rectangles — and painted here by a small reader of that SVG.

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke};

/// The look of an element (the icon set's classes): `a` the accent stroke, `f` the accent on
/// a soft fill, `g` ink with a faint fill, `gf` the faint fill alone, `dt`/`adt` ink or accent
/// dots, `th` a thin stroke, `ft` faint, `ds` dashed, `ax` a centre line.
#[derive(Debug, Clone, Copy, Default)]
struct Class {
    a: bool,
    f: bool,
    g: bool,
    gf: bool,
    dt: bool,
    adt: bool,
    th: bool,
    ft: bool,
    ds: bool,
    ax: bool,
}

fn class(names: &str) -> Class {
    let mut c = Class::default();
    for n in names.split_whitespace() {
        match n {
            "i-a" => c.a = true,
            "i-f" => c.f = true,
            "i-g" => c.g = true,
            "i-gf" => c.gf = true,
            "i-dt" => c.dt = true,
            "i-adt" => c.adt = true,
            "i-th" => c.th = true,
            "i-ft" => c.ft = true,
            "i-ds" => c.ds = true,
            "i-ax" => c.ax = true,
            _ => {}
        }
    }
    c
}

/// One element of a picture.
enum El {
    Path(&'static str, &'static str),
    Circle(&'static str, f32, f32, f32),
    Rect(&'static str, f32, f32, f32, f32, f32),
}

use El::{Circle as C, Path as P, Rect as R};

/// The colours a picture is drawn in: the key's text colour, the accent and its soft fill.
#[derive(Debug, Clone, Copy)]
pub struct Ink {
    pub ink: Color32,
    pub accent: Color32,
    pub soft: Color32,
    /// On a primary key the drawing goes the key's text colour, the made part told by its
    /// fill alone.
    pub primary: bool,
}

/// The pictures the sketch box uses, by their name in the icon set.
fn elements(name: &str) -> &'static [El] {
    match name {
        "select" => &[P("i-g", "M6 3v15l4-3.5 2.8 6 2.2-1-2.7-5.9 5.2-.4Z")],
        "line" => &[P("", "M5 18.5 19 5.5"), C("i-adt", 5.0, 18.5, 1.8), C("i-adt", 19.0, 5.5, 1.8)],
        "rectangle" => &[R("", 4.0, 6.0, 16.0, 12.0, 0.0), C("i-adt", 4.0, 6.0, 1.8), C("i-adt", 20.0, 18.0, 1.8)],
        "rectangleCenter" => &[R("", 4.0, 6.0, 16.0, 12.0, 0.0), P("i-th i-ds", "m4 6 16 12M20 6 4 18"), C("i-adt", 12.0, 12.0, 1.8), C("i-adt", 20.0, 18.0, 1.8)],
        "rectangleThreePoint" => &[P("", "M3.7 12.6 14.9 4.2l5.4 7.2-11.2 8.4Z"), C("i-adt", 3.7, 12.6, 1.8), C("i-adt", 14.9, 4.2, 1.8), C("i-adt", 14.7, 15.6, 1.8)],
        "polygon" => &[C("i-th i-ds", 12.0, 12.0, 8.5), P("", "M20.5 12 16.25 19.36h-8.5L3.5 12l4.25-7.36h8.5Z"), C("i-adt", 12.0, 12.0, 1.6), C("i-adt", 20.5, 12.0, 1.8)],
        "polygonOutside" => &[P("", "M19.5 7.67v8.66L12 20.66l-7.5-4.33V7.67L12 3.34Z"), C("i-th i-ds", 12.0, 12.0, 7.5), C("i-adt", 12.0, 12.0, 1.6), C("i-adt", 19.5, 12.0, 1.8)],
        "circle" => &[C("", 12.0, 12.0, 8.0), P("i-a i-th i-ds", "M12 12 17.66 6.34"), C("i-adt", 12.0, 12.0, 1.6), C("i-adt", 17.66, 6.34, 1.8)],
        "circleTwoPoint" => &[C("", 12.0, 12.0, 8.0), P("i-a i-th i-ds", "M6.34 17.66 17.66 6.34"), C("i-adt", 6.34, 17.66, 1.8), C("i-adt", 17.66, 6.34, 1.8)],
        "circleThreePoint" => &[C("", 12.0, 12.0, 8.0), C("i-adt", 5.07, 8.0, 1.8), C("i-adt", 18.93, 8.0, 1.8), C("i-adt", 12.0, 20.0, 1.8)],
        "arc" => &[P("", "M4 17a8 8 0 0 1 16 0"), C("i-adt", 4.0, 17.0, 1.8), C("i-adt", 20.0, 17.0, 1.8), C("i-adt", 12.0, 9.0, 1.8)],
        "arcCenter" => &[P("", "M18.5 18.5a13 13 0 0 0-13-13"), P("i-a i-th i-ds", "M5.5 18.5h13"), C("i-adt", 5.5, 18.5, 1.6), C("i-adt", 18.5, 18.5, 1.8), C("i-adt", 5.5, 5.5, 1.8)],
        "spline" => &[P("", "M3 17C7 5 10 5 12 12s5 7 9-5"), C("i-adt", 3.0, 17.0, 1.7), C("i-adt", 7.7, 8.3, 1.7), C("i-adt", 12.0, 12.0, 1.7), C("i-adt", 16.3, 15.7, 1.7), C("i-adt", 21.0, 7.0, 1.7)],
        "filletCorner" => &[P("", "M4 20v-8M12 4h8"), P("i-th i-ft i-ds", "M4 12V4h8"), P("i-a", "M4 12a8 8 0 0 1 8-8")],
        "chamferCorner" => &[P("", "M4 20v-8M12 4h8"), P("i-th i-ft i-ds", "M4 12V4h8"), P("i-a", "M4 12l8-8")],
        "project" => &[P("i-g", "M3 20h18"), P("i-th i-ft i-ds", "M6 5v15M18 8.5V20"), P("", "m6 5 12 3.5"), P("i-a", "M6 20h12")],
        "gridSnap" => &[P("i-th i-ft", "M3 8h18M3 16h18M8 3v18M16 3v18"), C("i-a", 16.0, 8.0, 3.4), C("i-adt", 16.0, 8.0, 1.5)],
        "eye" => &[P("", "M2.5 12C5.5 6.5 9 5 12 5s6.5 1.5 9.5 7c-3 5.5-6.5 7-9.5 7s-6.5-1.5-9.5-7Z"), C("i-f", 12.0, 12.0, 3.0)],
        "eyeOff" => &[P("i-ft", "M2.5 12C5.5 6.5 9 5 12 5s6.5 1.5 9.5 7c-3 5.5-6.5 7-9.5 7s-6.5-1.5-9.5-7Z"), P("", "M4.5 19.5 19.5 4.5")],
        "fitAll" => &[
            P("i-a", "M3 8h9a7 7 0 0 1 7 7v6"),
            C("i-dt", 4.0, 7.2, 0.95),
            C("i-dt", 7.0, 8.8, 0.95),
            C("i-dt", 10.0, 7.3, 0.95),
            C("i-dt", 13.2, 8.9, 0.95),
            C("i-dt", 16.0, 8.6, 0.95),
            C("i-dt", 17.5, 12.0, 0.95),
            C("i-dt", 19.7, 14.6, 0.95),
            C("i-dt", 18.3, 17.8, 0.95),
            C("i-dt", 19.6, 20.6, 0.95),
        ],
        "inferCorners" => &[P("", "M4 19.5 10.07 8.5M20 19.5 13.93 8.5"), P("i-a i-ds", "M10.07 8.5 12 5l1.93 3.5"), C("i-adt", 12.0, 5.0, 1.8)],
        "closeChain" => &[P("", "M9 5H5v14h14V5h-4"), P("i-a", "M9 5h6"), C("i-adt", 9.0, 5.0, 1.7), C("i-adt", 15.0, 5.0, 1.7)],
        "dimDistance" => &[P("", "M4 4.5v15M20 4.5v15"), P("i-a", "M4.8 12h14.4M7.6 9.2 4.8 12l2.8 2.8M16.4 9.2l2.8 2.8-2.8 2.8")],
        "coincident" => &[P("", "M4 5l8 7.5M12 12.5l8-4"), C("i-a", 12.0, 12.5, 3.6), C("i-adt", 12.0, 12.5, 1.6)],
        "horizontal" => &[P("", "M6 8h15"), C("i-dt", 6.0, 8.0, 1.5), C("i-dt", 21.0, 8.0, 1.5), P("i-th i-ft", "M4 20.5v-7"), P("i-a", "M4 20.5h9M10.8 18.3l2.2 2.2-2.2 2.2")],
        "vertical" => &[P("", "M16 3v15"), C("i-dt", 16.0, 3.0, 1.5), C("i-dt", 16.0, 18.0, 1.5), P("i-th i-ft", "M4 20.5h7"), P("i-a", "M4 20.5v-9M1.8 13.7 4 11.5l2.2 2.2")],
        "parallel" => &[P("", "M4 17 13 4"), P("i-a", "M11 20l9-13")],
        "perpendicular" => &[P("", "M3 19h18"), P("i-a", "M12 19V4"), P("i-a i-th", "M12 15.5h3.5V19")],
        "tangent" => &[C("", 11.0, 15.0, 6.5), P("i-a", "M7.32 5.37 21.18 13.37"), C("i-adt", 14.25, 9.37, 1.6)],
        "concentric" => &[C("", 12.0, 12.0, 8.5), C("i-a", 12.0, 12.0, 4.5), C("i-adt", 12.0, 12.0, 1.4)],
        "equal" => &[P("", "M3 6.5h9M12 17.5h9"), P("i-a", "M9.5 10.5h5M9.5 13.5h5")],
        "symmetric" => &[P("i-ax i-ft", "M12 2v20"), C("i-g", 5.5, 12.0, 3.0), C("i-f", 18.5, 12.0, 3.0), P("i-a i-th", "M9.5 12h5M10.9 10.6 9.5 12l1.4 1.4M13.1 10.6l1.4 1.4-1.4 1.4")],
        "del" => &[P("", "M4.5 7h15M9 7V4.5h6V7M6.5 7l1 13.5h9L17.5 7M10.3 10.5v6.5M13.7 10.5v6.5")],
        "sharpCorner" => &[P("", "M4 20v-8M12 4h8"), P("i-th i-ft i-ds", "M4 12a8 8 0 0 1 8-8"), P("i-a", "M4 12V4h8")],
        "fix" => &[R("i-g", 5.5, 11.0, 13.0, 9.5, 1.5), P("", "M8.5 11V8a3.5 3.5 0 0 1 7 0v3"), C("i-adt", 12.0, 15.7, 1.6)],
        "construction" => &[P("i-a i-ax", "M5 18.5 19 5.5"), C("i-dt", 5.0, 18.5, 1.6), C("i-dt", 19.0, 5.5, 1.6)],
        "sketchPattern" => &[P("i-g", "M3.5 3.5h6v6h-6Z"), P("i-a", "M14.5 3.5h6v6h-6ZM3.5 14.5h6v6h-6ZM14.5 14.5h6v6h-6Z")],
        "sketchPatternRound" => &[C("i-g", 12.0, 4.5, 2.5), C("i-a", 18.5, 15.75, 2.5), C("i-a", 5.5, 15.75, 2.5), C("i-th i-ds", 12.0, 12.0, 7.5), C("i-adt", 12.0, 12.0, 1.4)],
        "alignSheet" => &[P("i-th i-ft i-ds", "M3 19 19 7"), P("", "M3 19h18"), P("i-a", "M14.2 10.6A14 14 0 0 1 16.86 17.05M15.22 15.29l1.64 1.76 1.1-2.14"), C("i-dt", 3.0, 19.0, 1.5)],
        "sketchImage" => &[P("i-g", "M3 4.5h18v15H3Z"), P("i-th", "m3 16.5 5-5 4 4 3-3 6 6"), C("i-th", 15.5, 8.5, 1.5), P("i-a", "M5 21.5h14"), C("i-adt", 5.0, 21.5, 1.4), C("i-adt", 19.0, 21.5, 1.4)],
        "extrude" => &[P("i-f", "M4 12h11v8H4Z"), P("i-f", "m15 12 4-4v8l-4 4Z"), P("i-f", "m4 12 4-4h11l-4 4Z"), P("", "M4 20h11l4-4"), P("i-a", "M11.5 10V1.5M9.3 3.7l2.2-2.2 2.2 2.2")],
        "revolve" => &[P("i-f", "M12 8.5H6.5v4H9v9h3Z"), P("i-g", "M12 8.5h5.5v4H15v9h-3Z"), P("i-ax i-ft", "M12 1v22"), P("i-a", "M17 3.6A5 1.8 0 0 1 7.59 4.45M8.41 6.6l-.82-2.15 2.3-.03")],
        _ => &[],
    }
}

// ---- reading the SVG path ----

/// The numbers and commands of a path's `d`.
fn tokens(d: &str) -> Vec<Result<char, f32>> {
    let mut out = Vec::new();
    let b: Vec<char> = d.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_alphabetic() {
            out.push(Ok(c));
            i += 1;
        } else if c == '-' || c == '.' || c.is_ascii_digit() {
            let start = i;
            let mut seen_dot = false;
            if c == '-' {
                i += 1;
            }
            while i < b.len() {
                let x = b[i];
                if x.is_ascii_digit() {
                    i += 1;
                } else if x == '.' && !seen_dot {
                    seen_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            let s: String = b[start..i].iter().collect();
            if let Ok(v) = s.parse::<f32>() {
                out.push(Err(v));
            }
        } else {
            i += 1;
        }
    }
    out
}

/// The point at `t` of a cubic Bézier.
fn cubic(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
    let u = 1.0 - t;
    let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
    Pos2::new(w[0] * p0.x + w[1] * p1.x + w[2] * p2.x + w[3] * p3.x, w[0] * p0.y + w[1] * p1.y + w[2] * p2.y + w[3] * p3.y)
}

/// An SVG arc from `p0` to `p1`, sampled (the endpoint-to-centre conversion of the SVG spec).
#[allow(clippy::too_many_arguments)]
fn arc_points(p0: Pos2, rx: f32, ry: f32, rot_deg: f32, large: bool, sweep: bool, p1: Pos2, out: &mut Vec<Pos2>) {
    let (mut rx, mut ry) = (rx.abs(), ry.abs());
    if rx < 1e-6 || ry < 1e-6 {
        out.push(p1);
        return;
    }
    let phi = rot_deg.to_radians();
    let (s, c) = phi.sin_cos();
    let dx = (p0.x - p1.x) / 2.0;
    let dy = (p0.y - p1.y) / 2.0;
    let x1 = c * dx + s * dy;
    let y1 = -s * dx + c * dy;
    let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry);
    if lambda > 1.0 {
        rx *= lambda.sqrt();
        ry *= lambda.sqrt();
    }
    let num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1;
    let den = rx * rx * y1 * y1 + ry * ry * x1 * x1;
    let mut k = if den > 0.0 { (num / den).max(0.0).sqrt() } else { 0.0 };
    if large == sweep {
        k = -k;
    }
    let cx1 = k * rx * y1 / ry;
    let cy1 = -k * ry * x1 / rx;
    let cx = c * cx1 - s * cy1 + (p0.x + p1.x) / 2.0;
    let cy = s * cx1 + c * cy1 + (p0.y + p1.y) / 2.0;
    let angle = |ux: f32, uy: f32, vx: f32, vy: f32| (ux * vy - uy * vx).atan2(ux * vx + uy * vy);
    let t1 = angle(1.0, 0.0, (x1 - cx1) / rx, (y1 - cy1) / ry);
    let mut dt = angle((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry);
    if !sweep && dt > 0.0 {
        dt -= std::f32::consts::TAU;
    } else if sweep && dt < 0.0 {
        dt += std::f32::consts::TAU;
    }
    let n = ((dt.abs() / 0.2).ceil() as usize).clamp(4, 64);
    for i in 1..=n {
        let t = t1 + dt * i as f32 / n as f32;
        let (st, ct) = t.sin_cos();
        out.push(Pos2::new(cx + c * rx * ct - s * ry * st, cy + s * rx * ct + c * ry * st));
    }
}

/// A path's subpaths as polylines on the 24-unit grid, and whether each is closed.
fn subpaths(d: &str) -> Vec<(Vec<Pos2>, bool)> {
    let toks = tokens(d);
    let mut out: Vec<(Vec<Pos2>, bool)> = Vec::new();
    let mut cur: Vec<Pos2> = Vec::new();
    let mut pen = Pos2::ZERO;
    let mut start = Pos2::ZERO;
    let mut last_ctrl: Option<Pos2> = None;
    let mut cmd = 'M';
    let mut i = 0;
    let num = |i: &mut usize| -> Option<f32> {
        match toks.get(*i) {
            Some(Err(v)) => {
                *i += 1;
                Some(*v)
            }
            _ => None,
        }
    };
    while i < toks.len() {
        if let Ok(c) = toks[i] {
            cmd = c;
            i += 1;
            if c == 'Z' || c == 'z' {
                if !cur.is_empty() {
                    out.push((std::mem::take(&mut cur), true));
                }
                pen = start;
                last_ctrl = None;
                continue;
            }
        }
        let rel = cmd.is_ascii_lowercase();
        let base = if rel { pen } else { Pos2::ZERO };
        let at = |x: f32, y: f32| Pos2::new(base.x + x, base.y + y);
        match cmd.to_ascii_uppercase() {
            'M' => {
                let (Some(x), Some(y)) = (num(&mut i), num(&mut i)) else { break };
                if cur.len() > 1 {
                    out.push((std::mem::take(&mut cur), false));
                }
                cur.clear();
                pen = at(x, y);
                start = pen;
                cur.push(pen);
                // Further pairs after a move are lines.
                cmd = if rel { 'l' } else { 'L' };
                last_ctrl = None;
            }
            'L' => {
                let (Some(x), Some(y)) = (num(&mut i), num(&mut i)) else { break };
                pen = at(x, y);
                cur.push(pen);
                last_ctrl = None;
            }
            'H' => {
                let Some(x) = num(&mut i) else { break };
                pen = Pos2::new(if rel { pen.x + x } else { x }, pen.y);
                cur.push(pen);
                last_ctrl = None;
            }
            'V' => {
                let Some(y) = num(&mut i) else { break };
                pen = Pos2::new(pen.x, if rel { pen.y + y } else { y });
                cur.push(pen);
                last_ctrl = None;
            }
            'C' | 'S' => {
                let p1 = if cmd.to_ascii_uppercase() == 'C' {
                    let (Some(x), Some(y)) = (num(&mut i), num(&mut i)) else { break };
                    at(x, y)
                } else {
                    last_ctrl.map_or(pen, |c| Pos2::new(2.0 * pen.x - c.x, 2.0 * pen.y - c.y))
                };
                let (Some(x2), Some(y2), Some(x3), Some(y3)) = (num(&mut i), num(&mut i), num(&mut i), num(&mut i)) else { break };
                let (p2, p3) = (at(x2, y2), at(x3, y3));
                for k in 1..=12 {
                    cur.push(cubic(pen, p1, p2, p3, k as f32 / 12.0));
                }
                last_ctrl = Some(p2);
                pen = p3;
            }
            'A' => {
                let (Some(rx), Some(ry), Some(rot), Some(large), Some(sweep), Some(x), Some(y)) = (num(&mut i), num(&mut i), num(&mut i), num(&mut i), num(&mut i), num(&mut i), num(&mut i)) else { break };
                let to = at(x, y);
                arc_points(pen, rx, ry, rot, large != 0.0, sweep != 0.0, to, &mut cur);
                pen = to;
                last_ctrl = None;
            }
            _ => {
                i += 1;
            }
        }
    }
    if cur.len() > 1 {
        out.push((cur, false));
    }
    out
}

fn circle_points(cx: f32, cy: f32, r: f32) -> Vec<Pos2> {
    let n = 32;
    (0..n).map(|i| {
        let t = std::f32::consts::TAU * i as f32 / n as f32;
        Pos2::new(cx + r * t.cos(), cy + r * t.sin())
    }).collect()
}

fn rect_points(x: f32, y: f32, w: f32, h: f32, rx: f32) -> Vec<Pos2> {
    if rx <= 0.0 {
        return vec![Pos2::new(x, y), Pos2::new(x + w, y), Pos2::new(x + w, y + h), Pos2::new(x, y + h)];
    }
    let mut out = Vec::new();
    let corners = [(x + w - rx, y + rx, -90.0f32), (x + w - rx, y + h - rx, 0.0), (x + rx, y + h - rx, 90.0), (x + rx, y + rx, 180.0)];
    for (cx, cy, a0) in corners {
        for k in 0..=4 {
            let t = (a0 + 22.5 * k as f32).to_radians();
            out.push(Pos2::new(cx + rx * t.cos(), cy + rx * t.sin()));
        }
    }
    out
}

/// A simple polygon's triangles, by ear clipping (the faint fills; a few corners each).
fn triangulate(pts: &[Pos2]) -> Vec<[Pos2; 3]> {
    let mut idx: Vec<usize> = (0..pts.len()).collect();
    let area: f32 = (0..pts.len()).map(|i| {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        a.x * b.y - b.x * a.y
    }).sum();
    if area < 0.0 {
        idx.reverse();
    }
    let cross = |a: Pos2, b: Pos2, c: Pos2| (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    let mut out = Vec::new();
    let mut guard = 0;
    while idx.len() > 3 && guard < 1000 {
        guard += 1;
        let n = idx.len();
        let mut clipped = false;
        for k in 0..n {
            let (ia, ib, ic) = (idx[(k + n - 1) % n], idx[k], idx[(k + 1) % n]);
            let (a, b, c) = (pts[ia], pts[ib], pts[ic]);
            if cross(a, b, c) <= 0.0 {
                continue;
            }
            let inside = idx.iter().any(|&j| j != ia && j != ib && j != ic && cross(a, b, pts[j]) > 0.0 && cross(b, c, pts[j]) > 0.0 && cross(c, a, pts[j]) > 0.0);
            if inside {
                continue;
            }
            out.push([a, b, c]);
            idx.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        out.push([pts[idx[0]], pts[idx[1]], pts[idx[2]]]);
    }
    out
}

/// Paint the picture `name` into `rect` (its best square).
pub fn paint(painter: &Painter, rect: Rect, name: &str, ink: Ink) {
    let size = rect.width().min(rect.height());
    let k = size / 24.0;
    let o = rect.center() - egui::vec2(size, size) * 0.5;
    let map = |p: Pos2| Pos2::new(o.x + p.x * k, o.y + p.y * k);
    // At 16 points and under a 1.5-unit stroke is less than a pixel.
    let small = size <= 16.0;
    for el in elements(name) {
        let (cls, outline, filled_shape) = match *el {
            El::Path(c, d) => (class(c), subpaths(d), false),
            El::Circle(c, cx, cy, r) => (class(c), vec![(circle_points(cx, cy, r), true)], true),
            El::Rect(c, x, y, w, h, rx) => (class(c), vec![(rect_points(x, y, w, h, rx), true)], true),
        };
        let accent_stroke = if ink.primary { ink.ink } else { ink.accent };
        let mut stroke_color = if cls.a || cls.f { accent_stroke } else { ink.ink };
        let mut width = if cls.th || cls.ax {
            if small {
                1.4
            } else {
                1.1
            }
        } else if small {
            1.9
        } else {
            1.5
        };
        if let El::Path(c, _) = el {
            if c.contains("i-a i-ax") {
                width = 1.5;
            }
        }
        let faint = |c: Color32| if cls.ft { c.gamma_multiply(0.5) } else { c };
        stroke_color = faint(stroke_color);
        let fill = if cls.dt {
            Some(faint(ink.ink))
        } else if cls.adt {
            Some(faint(if ink.primary { ink.ink } else { ink.accent }))
        } else if cls.f {
            Some(faint(if ink.primary { Color32::from_white_alpha(71) } else { ink.soft }))
        } else if cls.g || cls.gf {
            Some(faint(ink.ink.gamma_multiply(0.13)))
        } else {
            None
        };
        let stroked = !(cls.dt || cls.adt || cls.gf);
        for (pts, shut) in outline {
            let screen: Vec<Pos2> = pts.iter().map(|p| map(*p)).collect();
            if let Some(fill) = fill {
                if filled_shape || shut {
                    let mut mesh = egui::Mesh::default();
                    for t in triangulate(&screen) {
                        let base = mesh.vertices.len() as u32;
                        for p in t {
                            mesh.colored_vertex(p, fill);
                        }
                        mesh.add_triangle(base, base + 1, base + 2);
                    }
                    painter.add(Shape::mesh(mesh));
                }
            }
            if !stroked {
                continue;
            }
            let mut line = screen.clone();
            if shut {
                if let Some(first) = line.first().copied() {
                    line.push(first);
                }
            }
            let stroke = Stroke::new(width * k, stroke_color);
            if cls.ax {
                painter.extend(Shape::dashed_line(&line, stroke, 3.2 * k, 2.2 * k));
            } else if cls.ds {
                painter.extend(Shape::dashed_line(&line, stroke, 2.0 * k, 2.4 * k));
            } else {
                painter.add(Shape::line(line, stroke));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_reads_its_moves_lines_and_closes() {
        let paths = subpaths("M3.5 3.5h6v6h-6Z");
        assert_eq!(paths.len(), 1);
        let (pts, closed) = &paths[0];
        assert!(*closed);
        assert_eq!(pts.len(), 4);
        assert_eq!(pts[2], Pos2::new(9.5, 9.5));
        // Numbers run together: "2.2-2.2" is two, ".82-2.15" too.
        assert_eq!(tokens("l2.2-2.2").len(), 3);
        assert_eq!(tokens("l-.82-2.15").len(), 3);
    }

    #[test]
    fn an_arc_ends_where_it_is_told() {
        let paths = subpaths("M4 17a8 8 0 0 1 16 0");
        let end = *paths[0].0.last().expect("points");
        assert!((end.x - 20.0).abs() < 1e-3 && (end.y - 17.0).abs() < 1e-3);
        // It bulges up (the sweep flag), through the top of the circle.
        assert!(paths[0].0.iter().any(|p| (p.y - 9.0).abs() < 0.1));
    }

    #[test]
    fn every_picture_the_box_names_is_drawn() {
        for name in ["select", "line", "rectangle", "rectangleCenter", "rectangleThreePoint", "polygon", "polygonOutside", "circle", "circleTwoPoint", "circleThreePoint", "arc", "arcCenter", "filletCorner", "chamferCorner", "project", "gridSnap", "eye", "eyeOff", "fitAll", "inferCorners", "closeChain", "dimDistance", "coincident", "horizontal", "vertical", "parallel", "perpendicular", "tangent", "concentric", "equal", "symmetric", "del", "sharpCorner", "fix", "construction", "sketchPattern", "sketchPatternRound", "alignSheet", "sketchImage", "extrude", "revolve"] {
            assert!(!elements(name).is_empty(), "{name}");
        }
    }
}
