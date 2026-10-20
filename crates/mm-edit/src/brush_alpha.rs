//! Brush alphas, stencils and dab spacing for the sculpt brushes.
//!
//! * [`AlphaImage`] — a grey image (0..1) sampled bilinearly; [`Pattern`] draws the built-in
//!   ones (our own procedural designs).
//! * [`AlphaMap`] — how an alpha sits in a dab: scale, turn, strength, tiling, invert. A
//!   vertex's brush weight is multiplied by the alpha where the vertex falls in the dab's
//!   tangent frame ([`DabFrame`]), whose first axis follows the stroke (AlignToStroke) or the
//!   view, so the footprint moves and turns with the dab.
//! * [`ScreenStencil`] — an alpha fixed to the view: a surface point gets the stencil's value
//!   where it projects on screen ([`ScreenMap`]).
//! * [`Spacer`] — dab centres a fixed distance apart along a stroke, however fast it moves.

use glam::{DMat4, DVec2, DVec3};

/// A grey image, values 0..1, rows from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl AlphaImage {
    /// `data` row by row from the top left (clamped to 0..1).
    pub fn new(width: usize, height: usize, mut data: Vec<f32>) -> Self {
        data.resize(width * height, 0.0);
        for x in &mut data {
            *x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
        }
        Self { width, height, data }
    }

    /// `f(u, v)` at the pixel centres (u right, v down, both 0..1).
    pub fn from_fn(width: usize, height: usize, f: impl Fn(f64, f64) -> f64) -> Self {
        let data = (0..width * height).map(|i| f(((i % width) as f64 + 0.5) / width as f64, ((i / width) as f64 + 0.5) / height as f64) as f32).collect();
        Self::new(width, height, data)
    }

    /// Bilinear value at `uv` (0..1 across the image, v down). Outside the image: 0, or the
    /// image repeated with `tile`.
    pub fn sample(&self, uv: DVec2, tile: bool) -> f64 {
        if self.width == 0 || self.height == 0 || !uv.is_finite() || (!tile && !in_unit(uv)) {
            return 0.0;
        }
        let (w, h) = (self.width as i64, self.height as i64);
        let (x, y) = (uv.x * w as f64 - 0.5, uv.y * h as f64 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |i: i64, j: i64| {
            let (i, j) = if tile { (i.rem_euclid(w), j.rem_euclid(h)) } else { (i.clamp(0, w - 1), j.clamp(0, h - 1)) };
            self.data[(j * w + i) as usize] as f64
        };
        let (i, j) = (x0 as i64, y0 as i64);
        let top = at(i, j) + (at(i + 1, j) - at(i, j)) * fx;
        let bottom = at(i, j + 1) + (at(i + 1, j + 1) - at(i, j + 1)) * fx;
        top + (bottom - top) * fy
    }

    /// Shrunk by a whole factor (box filter) until neither side exceeds `max`.
    pub fn fit(self, max: usize) -> Self {
        let k = self.width.max(self.height).div_ceil(max.max(1));
        if k <= 1 {
            return self;
        }
        let (w, h) = (self.width.div_ceil(k), self.height.div_ceil(k));
        let mut data = vec![0.0f32; w * h];
        for (j, row) in data.chunks_mut(w).enumerate() {
            for (i, out) in row.iter_mut().enumerate() {
                let (mut sum, mut n) = (0.0, 0.0);
                for y in j * k..((j + 1) * k).min(self.height) {
                    for x in i * k..((i + 1) * k).min(self.width) {
                        sum += self.data[y * self.width + x];
                        n += 1.0;
                    }
                }
                *out = if n > 0.0 { sum / n } else { 0.0 };
            }
        }
        Self { width: w, height: h, data }
    }
}

fn in_unit(uv: DVec2) -> bool {
    (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y)
}

/// Built-in alphas. All but Disc and Rings repeat seamlessly (for tiling).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pattern {
    Disc,
    Noise,
    Rings,
    Stripes,
    Dots,
    Crosshatch,
    Cells,
}

impl Pattern {
    pub const ALL: [Pattern; 7] = [Pattern::Disc, Pattern::Noise, Pattern::Rings, Pattern::Stripes, Pattern::Dots, Pattern::Crosshatch, Pattern::Cells];

    /// Name in settings and scripts.
    pub fn key(self) -> &'static str {
        match self {
            Pattern::Disc => "disc",
            Pattern::Noise => "noise",
            Pattern::Rings => "rings",
            Pattern::Stripes => "stripes",
            Pattern::Dots => "dots",
            Pattern::Crosshatch => "crosshatch",
            Pattern::Cells => "cells",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Pattern::Disc => "Disc",
            Pattern::Noise => "Noise",
            Pattern::Rings => "Rings",
            Pattern::Stripes => "Stripes",
            Pattern::Dots => "Dots",
            Pattern::Crosshatch => "Crosshatch",
            Pattern::Cells => "Cells",
        }
    }

    pub fn about(self) -> &'static str {
        match self {
            Pattern::Disc => "A flat-topped round footprint with a soft rim",
            Pattern::Noise => "Irregular grain (several scales of smooth noise)",
            Pattern::Rings => "Concentric rings around the dab centre",
            Pattern::Stripes => "Parallel rounded ridges",
            Pattern::Dots => "A grid of round domes (rivets)",
            Pattern::Crosshatch => "Two sets of thin diagonal ridges",
            Pattern::Cells => "Cobblestones: raised cells with grooves between them",
        }
    }

    pub fn from_key(key: &str) -> Option<Pattern> {
        Pattern::ALL.into_iter().find(|p| p.key().eq_ignore_ascii_case(key))
    }

    /// Value at `(u, v)` (0..1 across the image, v down).
    pub fn value(self, u: f64, v: f64) -> f64 {
        use std::f64::consts::TAU;
        let r = 2.0 * DVec2::new(u - 0.5, v - 0.5).length();
        match self {
            Pattern::Disc => 1.0 - smoothstep(0.86, 1.0, r),
            Pattern::Rings => 0.5 + 0.5 * (TAU * 4.0 * r).cos(),
            Pattern::Stripes => 0.5 + 0.5 * (TAU * 5.0 * u).cos(),
            Pattern::Crosshatch => {
                let ridge = |t: f64| (0.5 + 0.5 * (TAU * 6.0 * t).cos()).powi(6);
                ridge(u + v).max(ridge(u - v))
            }
            Pattern::Dots => {
                let d = DVec2::new((u * 4.0).fract() - 0.5, (v * 4.0).fract() - 0.5).length();
                (1.0 - (d / 0.36).powi(2)).max(0.0).sqrt()
            }
            Pattern::Noise => {
                let (mut sum, mut amp, mut total) = (0.0, 1.0, 0.0);
                for octave in 0..5 {
                    sum += amp * value_noise(u, v, 4 << octave, octave as u32);
                    total += amp;
                    amp *= 0.5;
                }
                ((sum / total - 0.5) * 1.8 + 0.5).clamp(0.0, 1.0)
            }
            Pattern::Cells => {
                // Distance to the nearest and second-nearest jittered cell point, on a grid
                // that wraps so the image tiles.
                const N: i64 = 5;
                let p = DVec2::new(u, v) * N as f64;
                let (ci, cj) = (p.x.floor() as i64, p.y.floor() as i64);
                let (mut f1, mut f2) = (f64::INFINITY, f64::INFINITY);
                for dj in -1..=1 {
                    for di in -1..=1 {
                        let (i, j) = (ci + di, cj + dj);
                        let (hi, hj) = (i.rem_euclid(N), j.rem_euclid(N));
                        let q = DVec2::new(i as f64 + 0.15 + 0.7 * hash(hi, hj, 11), j as f64 + 0.15 + 0.7 * hash(hi, hj, 12));
                        let d = p.distance(q);
                        if d < f1 {
                            f2 = f1;
                            f1 = d;
                        } else if d < f2 {
                            f2 = d;
                        }
                    }
                }
                smoothstep(0.0, 0.35, f2 - f1)
            }
        }
    }

    /// The pattern drawn at `size` × `size` pixels.
    pub fn image(self, size: usize) -> AlphaImage {
        AlphaImage::from_fn(size, size, |u, v| self.value(u, v))
    }
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Pseudo-random 0..1 for lattice point (x, y).
fn hash(x: i64, y: i64, seed: u32) -> f64 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_add(1).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    h as f64 / u32::MAX as f64
}

/// Smooth value noise with `period` lattice cells across the unit square (wrapping).
fn value_noise(u: f64, v: f64, period: i64, seed: u32) -> f64 {
    let (x, y) = (u * period as f64, v * period as f64);
    let (x0, y0) = (x.floor(), y.floor());
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (fx, fy) = (s(x - x0), s(y - y0));
    let g = |i: i64, j: i64| hash(i.rem_euclid(period), j.rem_euclid(period), seed);
    let (i, j) = (x0 as i64, y0 as i64);
    let top = g(i, j) + (g(i + 1, j) - g(i, j)) * fx;
    let bottom = g(i, j + 1) + (g(i + 1, j + 1) - g(i, j + 1)) * fx;
    top + (bottom - top) * fy
}

/// How an alpha sits in its frame (the dab, or the viewport for a stencil).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaMap {
    /// Image side over the brush diameter (a stencil: over the viewport height).
    pub scale: f64,
    /// Turns the image counter-clockwise (degrees, as seen from the front).
    pub rotation_deg: f64,
    /// 0..1: 0 leaves the brush alone, 1 multiplies it by the alpha.
    pub strength: f64,
    /// Repeat the image beyond its square (otherwise the brush does nothing there).
    pub tile: bool,
    /// Use 1 − the image's value.
    pub invert: bool,
}

impl Default for AlphaMap {
    fn default() -> Self {
        Self { scale: 1.0, rotation_deg: 0.0, strength: 1.0, tile: false, invert: false }
    }
}

impl AlphaMap {
    /// Image coordinates of `local` (x across, y up the frame, in units where the frame's
    /// half-size is 1: brush radii, or half the viewport height).
    pub fn uv(&self, local: DVec2) -> DVec2 {
        let (s, c) = self.rotation_deg.to_radians().sin_cos();
        let span = 2.0 * self.scale.max(1e-9);
        let (x, y) = (c * local.x + s * local.y, -s * local.x + c * local.y);
        DVec2::new(0.5 + x / span, 0.5 - y / span)
    }

    /// The inverse of [`AlphaMap::uv`].
    pub fn local(&self, uv: DVec2) -> DVec2 {
        let (s, c) = self.rotation_deg.to_radians().sin_cos();
        let span = 2.0 * self.scale.max(1e-9);
        let (x, y) = ((uv.x - 0.5) * span, (0.5 - uv.y) * span);
        DVec2::new(c * x - s * y, s * x + c * y)
    }

    /// What the brush weight at `local` is multiplied by.
    pub fn factor(&self, img: &AlphaImage, local: DVec2) -> f64 {
        let uv = self.uv(local);
        let a = if self.tile || in_unit(uv) {
            let a = img.sample(uv, self.tile);
            if self.invert { 1.0 - a } else { a }
        } else {
            0.0
        };
        1.0 - self.strength + self.strength * a
    }
}

/// A dab's tangent frame: `across` and `up` span the plane facing the surface at `center`;
/// lengths in brush radii `r`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DabFrame {
    pub center: DVec3,
    pub across: DVec3,
    pub up: DVec3,
    pub r: f64,
}

impl DabFrame {
    /// Frame facing `normal`, `across` along `dir` laid into the tangent plane (the stroke
    /// direction for AlignToStroke, else the view's right), `up = normal × across`.
    pub fn new(center: DVec3, normal: DVec3, dir: DVec3, r: f64) -> Self {
        let n = normal.normalize_or(DVec3::Z);
        let mut across = (dir - n * dir.dot(n)).normalize_or_zero();
        if across == DVec3::ZERO {
            across = mm_core::geom::orthonormal_basis(n).0;
        }
        Self { center, across, up: n.cross(across), r: r.max(1e-12) }
    }

    /// `p` in the frame, in brush radii (x across, y up).
    pub fn local(&self, p: DVec3) -> DVec2 {
        let d = p - self.center;
        DVec2::new(d.dot(self.across), d.dot(self.up)) / self.r
    }

    /// The point at `local` on the frame's plane.
    pub fn point(&self, local: DVec2) -> DVec3 {
        self.center + (self.across * local.x + self.up * local.y) * self.r
    }
}

/// How the view maps world points to the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenMap {
    pub view_proj: DMat4,
    /// Viewport size in pixels.
    pub size: DVec2,
}

impl ScreenMap {
    /// Pixel position of `p` from the viewport's top left; `None` behind the camera.
    pub fn to_screen(&self, p: DVec3) -> Option<DVec2> {
        let clip = self.view_proj * p.extend(1.0);
        if clip.w <= 1e-12 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(DVec2::new((ndc.x + 1.0) * 0.5 * self.size.x, (1.0 - ndc.y) * 0.5 * self.size.y))
    }
}

/// An alpha fixed to the view (Meshmixer's stencils): a square image over the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenStencil {
    /// Centre, as fractions of the viewport's width and height from its top left.
    pub center: DVec2,
    /// `scale` is the image side over the viewport height.
    pub map: AlphaMap,
}

impl Default for ScreenStencil {
    fn default() -> Self {
        Self { center: DVec2::splat(0.5), map: AlphaMap { scale: 0.6, ..Default::default() } }
    }
}

impl ScreenStencil {
    /// Pixel `px` (from the viewport's top left) in the stencil's frame: x right, y up, in
    /// half viewport heights from the centre.
    pub fn local(&self, px: DVec2, viewport: DVec2) -> DVec2 {
        let c = self.center * viewport;
        DVec2::new(px.x - c.x, c.y - px.y) / (0.5 * viewport.y.max(1e-9))
    }

    /// Image coordinates under pixel `px`.
    pub fn uv(&self, px: DVec2, viewport: DVec2) -> DVec2 {
        self.map.uv(self.local(px, viewport))
    }

    /// The pixel showing image coordinates `uv` (for drawing the stencil).
    pub fn screen_of(&self, uv: DVec2, viewport: DVec2) -> DVec2 {
        let l = self.map.local(uv) * (0.5 * viewport.y);
        self.center * viewport + DVec2::new(l.x, -l.y)
    }

    /// What the brush weight at surface point `p` is multiplied by.
    pub fn factor(&self, img: &AlphaImage, p: DVec3, screen: &ScreenMap) -> f64 {
        match screen.to_screen(p) {
            Some(px) => self.map.factor(img, self.local(px, screen.size)),
            None => 1.0 - self.map.strength,
        }
    }
}

/// Dab centres along a stroke, `step` apart, the remainder carried from one pointer move
/// to the next so the spacing does not depend on how fast the stroke goes.
#[derive(Clone, Debug)]
pub struct Spacer {
    step: f64,
    last: Option<DVec3>,
    /// Path length since the last dab.
    carried: f64,
}

impl Spacer {
    pub fn new(step: f64) -> Self {
        Self { step: step.max(1e-12), last: None, carried: 0.0 }
    }

    /// Where the stroke is now (the last point it was moved to).
    pub fn last(&self) -> Option<DVec3> {
        self.last
    }

    /// Move the stroke on to `p`: the dab centres passed on the way (the stroke's first point
    /// is one). At most `max`: a longer jump spreads them evenly up to `p`.
    pub fn advance(&mut self, p: DVec3, max: usize) -> Vec<DVec3> {
        let Some(a) = self.last.replace(p) else {
            self.carried = 0.0;
            return vec![p];
        };
        let len = a.distance(p);
        let mut first = self.step - self.carried;
        // Rounding must not drop a dab that lands on the pointer.
        let eps = 1e-9 * self.step;
        if len <= 0.0 || first > len + eps {
            self.carried += len;
            return Vec::new();
        }
        let mut step = self.step;
        let mut n = ((len - first + eps) / step).floor() as usize + 1;
        if n > max.max(1) {
            n = max.max(1);
            step = len / n as f64;
            first = step;
        }
        self.carried = (len - (first + (n - 1) as f64 * step)).max(0.0);
        (0..n).map(|k| a.lerp(p, (first + k as f64 * step) / len)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacer_places_dabs_a_step_apart_at_any_speed() {
        // One slow and one fast stroke over the same line give the same dabs.
        let along = |xs: &[f64]| {
            let mut s = Spacer::new(1.0);
            xs.iter().flat_map(|&x| s.advance(DVec3::X * x, 100)).map(|p| p.x).collect::<Vec<f64>>()
        };
        let slow: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
        let want: Vec<f64> = (0..=10).map(f64::from).collect();
        for got in [along(&slow), along(&[0.0, 0.4, 2.7, 2.7, 10.0]), along(&[0.0, 10.0])] {
            assert_eq!(got.len(), want.len(), "{got:?}");
            assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-9), "{got:?}");
        }
        // Resting adds nothing; the distance carried across moves counts along the path,
        // round a corner too.
        let mut s = Spacer::new(1.0);
        assert_eq!(s.advance(DVec3::ZERO, 10), vec![DVec3::ZERO]);
        assert!(s.advance(DVec3::ZERO, 10).is_empty());
        assert!(s.advance(DVec3::X * 0.75, 10).is_empty());
        let d = s.advance(DVec3::new(0.75, 1.0, 0.0), 10);
        assert_eq!(d.len(), 1);
        assert!((d[0] - DVec3::new(0.75, 0.25, 0.0)).length() < 1e-12);
        assert_eq!(s.last(), Some(DVec3::new(0.75, 1.0, 0.0)));
        // A long jump is capped, the dabs spread evenly up to the pointer.
        let mut s = Spacer::new(1.0);
        s.advance(DVec3::ZERO, 8);
        let d = s.advance(DVec3::X * 100.0, 8);
        assert_eq!(d.len(), 8);
        assert!((d[7].x - 100.0).abs() < 1e-9 && (d[0].x - 12.5).abs() < 1e-9);
    }

    #[test]
    fn sampling_is_bilinear_and_tiles() {
        let img = AlphaImage::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        // Pixel centres, and halfway between them.
        assert!((img.sample(DVec2::new(0.25, 0.25), false)).abs() < 1e-12);
        assert!((img.sample(DVec2::new(0.75, 0.25), false) - 1.0).abs() < 1e-12);
        assert!((img.sample(DVec2::new(0.5, 0.25), false) - 0.5).abs() < 1e-12);
        assert!((img.sample(DVec2::new(0.5, 0.5), false) - 0.5).abs() < 1e-12);
        // Outside: nothing, or the image again.
        assert_eq!(img.sample(DVec2::new(1.25, 0.25), false), 0.0);
        assert!((img.sample(DVec2::new(1.75, 0.25), true) - 1.0).abs() < 1e-12);
        assert!((img.sample(DVec2::new(-0.25, 1.25), true) - 1.0).abs() < 1e-12);
        // Shrinking averages blocks.
        let big = AlphaImage::from_fn(8, 4, |u, _| if u < 0.5 { 1.0 } else { 0.0 }).fit(4);
        assert_eq!((big.width, big.height), (4, 2));
        assert_eq!(big.data, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        // Built-ins stay in 0..1 and the tiling ones repeat seamlessly.
        for p in Pattern::ALL {
            let img = p.image(64);
            assert!(img.data.iter().all(|x| (0.0..=1.0).contains(x)), "{p:?}");
            assert!(img.data.iter().any(|&x| x > 0.6) && img.data.iter().any(|&x| x < 0.4), "{p:?} has contrast");
            if !matches!(p, Pattern::Disc | Pattern::Rings) {
                for t in [0.1, 0.37, 0.8] {
                    assert!((p.value(0.0, t) - p.value(1.0, t)).abs() < 1e-6 && (p.value(t, 0.0) - p.value(t, 1.0)).abs() < 1e-6, "{p:?} tiles");
                }
            }
            assert_eq!(Pattern::from_key(p.key()), Some(p));
        }
    }

    /// An image black except a bright spot at (0.75, 0.5): half a radius "ahead" in the frame.
    fn spot() -> AlphaImage {
        AlphaImage::from_fn(64, 64, |u, v| (1.0 - (DVec2::new(u - 0.75, v - 0.5).length() - 0.06) / 0.06).clamp(0.0, 1.0))
    }

    #[test]
    fn alpha_turns_with_the_stroke_and_its_rotation() {
        let img = spot();
        let map = AlphaMap::default();
        let at = |frame: &DabFrame, map: &AlphaMap, p: DVec3| map.factor(&img, frame.local(p));
        let (c, r) = (DVec3::new(1.0, 2.0, 3.0), 2.0);
        // On a surface facing +z, stroking along +x: the spot is half a radius along +x.
        let fx = DabFrame::new(c, DVec3::Z, DVec3::X, r);
        assert!(at(&fx, &map, c + DVec3::X) > 0.99);
        assert!(at(&fx, &map, c + DVec3::Y) < 0.01);
        // AlignToStroke: the stroke turns to +y, so does the footprint.
        let fy = DabFrame::new(c, DVec3::Z, DVec3::Y, r);
        assert!(at(&fy, &map, c + DVec3::Y) > 0.99);
        assert!(at(&fy, &map, c + DVec3::X) < 0.01);
        // A tilted direction is laid into the tangent plane.
        let ft = DabFrame::new(c, DVec3::Z, DVec3::new(0.0, 1.0, 5.0), r);
        assert!((ft.across - DVec3::Y).length() < 1e-12 && (ft.up + DVec3::X).length() < 1e-12);
        // Rotation turns the image counter-clockwise seen from the front: +x ahead → +y.
        let turned = AlphaMap { rotation_deg: 90.0, ..map };
        assert!(at(&fx, &turned, c + DVec3::Y) > 0.99);
        assert!(at(&fx, &turned, c + DVec3::X) < 0.01);
        // Scale: half size puts the spot a quarter radius out; outside the image nothing,
        // unless it tiles; strength blends toward 1; invert.
        let half = AlphaMap { scale: 0.5, ..map };
        assert!(at(&fx, &half, c + DVec3::X * 0.5) > 0.99);
        assert_eq!(at(&fx, &half, c - DVec3::X * 1.2), 0.0);
        assert!(at(&fx, &AlphaMap { tile: true, ..half }, c + DVec3::X * 2.5) > 0.99);
        assert!((at(&fx, &AlphaMap { strength: 0.25, ..map }, c + DVec3::Y) - 0.75).abs() < 1e-9);
        assert!(at(&fx, &AlphaMap { invert: true, ..map }, c + DVec3::Y) > 0.99);
        // uv and local are inverses.
        let m = AlphaMap { scale: 0.7, rotation_deg: 33.0, ..map };
        let l = DVec2::new(0.3, -0.45);
        assert!((m.local(m.uv(l)) - l).length() < 1e-12);
    }

    #[test]
    fn stencil_takes_the_value_where_a_point_projects() {
        // Orthographic view down −z of the box x ∈ [−2, 2], y ∈ [−1, 1], 200 × 100 pixels.
        let screen = ScreenMap { view_proj: DMat4::from_scale(DVec3::new(0.5, 1.0, 0.1)), size: DVec2::new(200.0, 100.0) };
        assert!((screen.to_screen(DVec3::ZERO).unwrap() - DVec2::new(100.0, 50.0)).length() < 1e-9);
        assert!((screen.to_screen(DVec3::new(2.0, 1.0, 0.0)).unwrap() - DVec2::new(200.0, 0.0)).length() < 1e-9);
        // A stencil half the viewport high (50 px) in the middle.
        let st = ScreenStencil { center: DVec2::splat(0.5), map: AlphaMap { scale: 0.5, ..Default::default() } };
        let uv_of = |p: DVec3| st.uv(screen.to_screen(p).unwrap(), screen.size);
        assert!((uv_of(DVec3::new(0.0, 0.0, -3.0)) - DVec2::splat(0.5)).length() < 1e-9, "depth doesn't matter");
        // 10 px right and 10 px up of the centre: 0.2 of the image right and up.
        assert!((uv_of(DVec3::new(0.2, 0.2, 0.0)) - DVec2::new(0.7, 0.3)).length() < 1e-9);
        // The spot (0.75, 0.5) shows 12.5 px right of the centre: a point there gets it.
        let img = spot();
        assert!(st.factor(&img, DVec3::new(0.25, 0.0, 0.0), &screen) > 0.99);
        assert!(st.factor(&img, DVec3::new(-0.25, 0.0, 0.0), &screen) < 0.01);
        assert_eq!(st.factor(&img, DVec3::new(1.5, 0.0, 0.0), &screen), 0.0, "outside the stencil");
        // Turned 90°: the image's bottom points right; moved: it follows its centre.
        let turned = ScreenStencil { map: AlphaMap { rotation_deg: 90.0, ..st.map }, ..st };
        assert!((turned.uv(DVec2::new(110.0, 50.0), screen.size) - DVec2::new(0.5, 0.7)).length() < 1e-9);
        let moved = ScreenStencil { center: DVec2::new(0.25, 0.5), ..st };
        assert!((moved.uv(DVec2::new(50.0, 50.0), screen.size) - DVec2::splat(0.5)).length() < 1e-9);
        // Drawing corners round-trip.
        for uv in [DVec2::ZERO, DVec2::X, DVec2::ONE, DVec2::new(0.3, 0.8)] {
            assert!((turned.uv(turned.screen_of(uv, screen.size), screen.size) - uv).length() < 1e-9);
        }
        assert!((st.screen_of(DVec2::ZERO, screen.size) - DVec2::new(75.0, 25.0)).length() < 1e-9);
    }
}
