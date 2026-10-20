//! Polymix's label font: a plain monoline sans drawn as round-capped strokes of one width
//! (capital letters, digits and `- . + / :`), outlined with [`Sketch`]. Letters are one unit
//! tall (baseline at 0, capitals' centre lines up to 1); strokes add half their width
//! around that.

use super::{arc, Sketch};
use glam::DVec2;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Stroke width, in cap heights.
const STROKE: f64 = 0.16;
/// Space between the outlines of neighbouring letters.
const GAP: f64 = 0.14;
/// Advance of a space.
const SPACE: f64 = 0.45;
/// Grid used to outline the strokes.
const STEP: f64 = 0.008;

/// Every character the font draws (lower case letters are drawn as capitals).
pub const FONT_CHARS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.+/: ";

fn l(p: &[(f64, f64)]) -> Vec<DVec2> {
    p.iter().map(|&(x, y)| DVec2::new(x, y)).collect()
}

fn e(cx: f64, cy: f64, rx: f64, ry: f64, a0: f64, a1: f64) -> Vec<DVec2> {
    arc(DVec2::new(cx, cy), rx, ry, a0, a1)
}

/// Pieces joined into one stroke (straight from the end of one to the start of the next).
fn join(parts: &[Vec<DVec2>]) -> Vec<DVec2> {
    let mut out: Vec<DVec2> = Vec::new();
    for p in parts {
        for q in p {
            if out.last().is_none_or(|l| l.distance(*q) > 1e-9) {
                out.push(*q);
            }
        }
    }
    out
}

/// Centre lines of a character's strokes and the width they span.
fn strokes(c: char) -> Option<(f64, Vec<Vec<DVec2>>)> {
    let p_bowl = || vec![l(&[(0.0, 0.0), (0.0, 1.0)]), join(&[l(&[(0.0, 1.0), (0.31, 1.0)]), e(0.31, 0.745, 0.25, 0.255, 90.0, -90.0), l(&[(0.31, 0.49), (0.0, 0.49)])])];
    let six = || vec![e(0.28, 0.3, 0.27, 0.3, 0.0, 360.0), e(0.52, 0.3, 0.51, 0.7, 180.0, 95.0)];
    let s = match c {
        'A' => (0.62, vec![l(&[(0.0, 0.0), (0.31, 1.0), (0.62, 0.0)]), l(&[(0.099, 0.32), (0.521, 0.32)])]),
        'B' => (
            0.575,
            vec![
                l(&[(0.0, 0.0), (0.0, 1.0)]),
                join(&[l(&[(0.0, 1.0), (0.3, 1.0)]), e(0.3, 0.755, 0.245, 0.245, 90.0, -90.0), l(&[(0.3, 0.51), (0.0, 0.51)])]),
                join(&[l(&[(0.0, 0.51), (0.32, 0.51)]), e(0.32, 0.255, 0.255, 0.255, 90.0, -90.0), l(&[(0.32, 0.0), (0.0, 0.0)])]),
            ],
        ),
        'C' => (0.6, vec![e(0.3, 0.5, 0.3, 0.5, 45.0, 315.0)]),
        'D' => (0.6, vec![l(&[(0.0, 0.0), (0.0, 1.0)]), join(&[l(&[(0.0, 1.0), (0.18, 1.0)]), e(0.18, 0.5, 0.42, 0.5, 90.0, -90.0), l(&[(0.18, 0.0), (0.0, 0.0)])])]),
        'E' => (0.52, vec![l(&[(0.52, 1.0), (0.0, 1.0), (0.0, 0.0), (0.52, 0.0)]), l(&[(0.0, 0.5), (0.44, 0.5)])]),
        'F' => (0.52, vec![l(&[(0.52, 1.0), (0.0, 1.0), (0.0, 0.0)]), l(&[(0.0, 0.5), (0.44, 0.5)])]),
        'G' => (0.6, vec![join(&[e(0.3, 0.5, 0.3, 0.5, 45.0, 360.0), l(&[(0.6, 0.5), (0.34, 0.5)])])]),
        'H' => (0.58, vec![l(&[(0.0, 0.0), (0.0, 1.0)]), l(&[(0.58, 0.0), (0.58, 1.0)]), l(&[(0.0, 0.5), (0.58, 0.5)])]),
        'I' => (0.0, vec![l(&[(0.0, 0.0), (0.0, 1.0)])]),
        'J' => (0.5, vec![join(&[l(&[(0.5, 1.0), (0.5, 0.27)]), e(0.25, 0.27, 0.25, 0.27, 0.0, -180.0)])]),
        'K' => (0.58, vec![l(&[(0.0, 0.0), (0.0, 1.0)]), l(&[(0.56, 1.0), (0.0, 0.36)]), l(&[(0.17, 0.555), (0.58, 0.0)])]),
        'L' => (0.5, vec![l(&[(0.0, 1.0), (0.0, 0.0), (0.5, 0.0)])]),
        'M' => (0.72, vec![l(&[(0.0, 0.0), (0.0, 1.0), (0.36, 0.32), (0.72, 1.0), (0.72, 0.0)])]),
        'N' => (0.58, vec![l(&[(0.0, 0.0), (0.0, 1.0), (0.58, 0.0), (0.58, 1.0)])]),
        'O' => (0.66, vec![e(0.33, 0.5, 0.33, 0.5, 0.0, 360.0)]),
        'P' => (0.56, p_bowl()),
        'Q' => (0.7, vec![e(0.33, 0.5, 0.33, 0.5, 0.0, 360.0), l(&[(0.42, 0.22), (0.7, -0.06)])]),
        'R' => (0.58, [p_bowl(), vec![l(&[(0.28, 0.49), (0.58, 0.0)])]].concat()),
        'S' => (0.56, vec![join(&[e(0.28, 0.75, 0.27, 0.25, 15.0, 270.0), e(0.28, 0.25, 0.28, 0.25, 90.0, -165.0)])]),
        'T' => (0.58, vec![l(&[(0.0, 1.0), (0.58, 1.0)]), l(&[(0.29, 1.0), (0.29, 0.0)])]),
        'U' => (0.58, vec![join(&[l(&[(0.0, 1.0), (0.0, 0.3)]), e(0.29, 0.3, 0.29, 0.3, 180.0, 360.0), l(&[(0.58, 0.3), (0.58, 1.0)])])]),
        'V' => (0.6, vec![l(&[(0.0, 1.0), (0.3, 0.0), (0.6, 1.0)])]),
        'W' => (0.84, vec![l(&[(0.0, 1.0), (0.2, 0.0), (0.42, 0.72), (0.64, 0.0), (0.84, 1.0)])]),
        'X' => (0.58, vec![l(&[(0.0, 0.0), (0.58, 1.0)]), l(&[(0.0, 1.0), (0.58, 0.0)])]),
        'Y' => (0.6, vec![l(&[(0.0, 1.0), (0.3, 0.48), (0.6, 1.0)]), l(&[(0.3, 0.48), (0.3, 0.0)])]),
        'Z' => (0.56, vec![l(&[(0.0, 1.0), (0.56, 1.0), (0.0, 0.0), (0.56, 0.0)])]),
        '0' => (0.54, vec![e(0.27, 0.5, 0.27, 0.5, 0.0, 360.0)]),
        '1' => (0.56, vec![l(&[(0.06, 0.78), (0.3, 1.0), (0.3, 0.0)]), l(&[(0.04, 0.0), (0.56, 0.0)])]),
        '2' => (0.56, vec![join(&[e(0.27, 0.72, 0.27, 0.28, 165.0, -35.0), l(&[(0.0, 0.0), (0.56, 0.0)])])]),
        '3' => (0.55, vec![e(0.27, 0.75, 0.25, 0.25, 155.0, -90.0), e(0.27, 0.255, 0.28, 0.255, 90.0, -155.0), l(&[(0.14, 0.505), (0.27, 0.505)])]),
        '4' => (0.58, vec![l(&[(0.42, 0.0), (0.42, 1.0), (0.0, 0.3), (0.58, 0.3)])]),
        '5' => (0.55, vec![join(&[l(&[(0.5, 1.0), (0.08, 1.0)]), e(0.27, 0.31, 0.28, 0.31, 140.0, -150.0)])]),
        '6' => (0.55, six()),
        '7' => (0.56, vec![l(&[(0.0, 1.0), (0.56, 1.0), (0.2, 0.0)])]),
        '8' => (0.55, vec![e(0.28, 0.765, 0.23, 0.235, 0.0, 360.0), e(0.28, 0.265, 0.27, 0.265, 0.0, 360.0)]),
        '9' => (0.56, six().into_iter().map(|s| s.into_iter().map(|p| DVec2::new(0.56 - p.x, 1.0 - p.y)).collect()).collect()),
        '-' => (0.36, vec![l(&[(0.0, 0.45), (0.36, 0.45)])]),
        '.' => (0.0, vec![l(&[(0.0, 0.0)])]),
        '+' => (0.5, vec![l(&[(0.0, 0.45), (0.5, 0.45)]), l(&[(0.25, 0.2), (0.25, 0.7)])]),
        '/' => (0.42, vec![l(&[(0.0, 0.0), (0.42, 1.0)])]),
        ':' => (0.0, vec![l(&[(0.0, 0.05)]), l(&[(0.0, 0.6)])]),
        _ => return None,
    };
    Some(s)
}

struct Glyph {
    loops: Vec<Vec<DVec2>>,
    width: f64,
}

fn glyph(c: char) -> Option<Arc<Glyph>> {
    static CACHE: OnceLock<Mutex<HashMap<char, Arc<Glyph>>>> = OnceLock::new();
    let c = c.to_ascii_uppercase();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(g) = cache.lock().ok()?.get(&c) {
        return Some(g.clone());
    }
    let (width, lines) = strokes(c)?;
    let sketch = lines.into_iter().fold(Sketch::default(), |s, line| s.stroke(line, STROKE));
    let g = Arc::new(Glyph { loops: sketch.outline(STEP), width });
    cache.lock().ok()?.insert(c, g.clone());
    Some(g)
}

/// A character's outline in font units (its strokes' left edge at x = 0, baseline at y = 0,
/// capitals one unit tall plus the stroke width), or None if the font has no such character.
pub fn glyph_loops(c: char) -> Option<Vec<Vec<DVec2>>> {
    let g = glyph(c)?;
    let shift = DVec2::new(STROKE * 0.5, 0.0);
    Some(g.loops.iter().map(|l| l.iter().map(|p| *p + shift).collect()).collect())
}

/// A line of text in font units, left to right from x = 0 (lower case is drawn as capitals).
pub fn text_loops(text: &str) -> Result<Vec<Vec<DVec2>>, String> {
    let text = text.trim();
    let missing: String = text.chars().filter(|&c| c != ' ' && strokes(c.to_ascii_uppercase()).is_none()).collect();
    if !missing.is_empty() {
        return Err(format!("the stamp font has no {missing} (it has A–Z, 0–9 and - . + / :)"));
    }
    if text.is_empty() {
        return Err("type the text to stamp".into());
    }
    let mut out = Vec::new();
    let mut x = 0.0;
    for c in text.chars() {
        if c == ' ' {
            x += SPACE;
            continue;
        }
        let g = glyph(c).ok_or_else(|| format!("no glyph for {c}"))?;
        let shift = DVec2::new(x + STROKE * 0.5, 0.0);
        out.extend(g.loops.iter().map(|l| l.iter().map(|p| *p + shift).collect::<Vec<_>>()));
        x += g.width + STROKE + GAP;
    }
    Ok(out)
}
