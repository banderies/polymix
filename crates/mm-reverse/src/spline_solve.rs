// SPDX-License-Identifier: AGPL-3.0-only
//! The sketch solver's spline terms, as functions `solver::solve_sketch` calls. Ported from
//! ScanRuler (the spline parts of `plugins/reverse/core/solver.ts`: `paramsOf` / `withParams`
//! for a spline, `meanChord`, the stay-put scaling of its end directions, `splineEndDir`,
//! `withHeldEnds`, the soft-data block with its refreshed feet and the refit rounds, the
//! tangent at a joint, a point held on a spline, and a spline's share of the solve's length
//! scale), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A spline's parameters are its fit points, then the end directions a tangent relation
//! holds: an end held tangent gets a direction to solve for (`with_held_ends`), taken from the
//! curve as it runs there now, and an end nothing holds has none and is natural again. The
//! stay-put term reads those angles as the chord they swing (`spline_stay_units`).
//!
//! A fitted spline's slice points are soft data measured both ways from the place on the
//! curve each lay nearest when the solve began — not along the normal alone, where a fit
//! point could slide along the curve for nothing and an open curve's tip run off past its
//! points (`SplineSoftData`). The feet are then found again where the curve has got to and
//! the curve settled again, up to `REFIT_ROUNDS` times, until no foot moves by
//! `REFIT_SETTLED` of a segment: after the first full descent the solver runs
//!
//! ```text
//! for round in 0..REFIT_ROUNDS (not on a drag's quick step):
//!     moved = max over spline soft blocks of block.refresh(x)
//!     if moved < REFIT_SETTLED: stop
//!     descend over every block, MAX_ITERATIONS, 1e-9, 1e-12
//! ```
//!
//! before the hand lets go and before the pass that closes relations the slice held open.

use std::cell::RefCell;

use glam::DVec2;

use crate::sketch::{AnchorAt, Constraint, EntityId, Rule};
use crate::sketch_spline::{is_closed, nearest_on_spline, spline_at, spline_curve, spline_dir_at, spline_end, spline_joint, SplineJoint, SplineShape};

/// The most times a fitted spline's points find their feet on the curve again within one
/// solve.
pub const REFIT_ROUNDS: usize = 24;
/// The largest move of a foot, in the curve's own parameter, under which the feet have
/// settled and the refit rounds stop.
pub const REFIT_SETTLED: f64 = 1e-7;

/// A direction, or `None` for one too short to have any.
fn normalize2(a: DVec2) -> Option<DVec2> {
    let l = a.x.hypot(a.y);
    if !(l > 1e-12) {
        return None;
    }
    Some(DVec2::new(a.x / l, a.y / l))
}

/// How many parameters a spline has in the solve: two a fit point, and one for each end
/// direction a tangent relation holds.
pub fn spline_param_count(e: &SplineShape) -> usize {
    2 * e.pts.len() + usize::from(e.start_dir.is_some()) + usize::from(e.end_dir.is_some())
}

/// A spline's parameters: the fit points, then the end directions a tangent relation holds.
pub fn spline_params(e: &SplineShape) -> Vec<f64> {
    let mut out = Vec::with_capacity(spline_param_count(e));
    for p in &e.pts {
        out.push(p.x);
        out.push(p.y);
    }
    out.extend(e.start_dir);
    out.extend(e.end_dir);
    out
}

/// A spline read back from the solve's parameters at `at`, laid out as `spline_params` lays
/// out `e`. A parameter past the end of `x` reads `NaN`, as ScanRuler's `undefined` does.
pub fn spline_with_params(e: &SplineShape, x: &[f64], at: usize) -> SplineShape {
    let n = e.pts.len();
    let get = |i: usize| x.get(i).copied().unwrap_or(f64::NAN);
    let pts = (0..n).map(|i| DVec2::new(get(at + 2 * i), get(at + 2 * i + 1))).collect();
    let mut k = at + 2 * n;
    let mut out = SplineShape { pts, closed: e.closed, start_dir: None, end_dir: None };
    if e.start_dir.is_some() {
        out.start_dir = Some(get(k));
        k += 1;
    }
    if e.end_dir.is_some() {
        out.end_dir = Some(get(k));
    }
    out
}

/// The mean length of a spline's chords — what an angle of its is read in millimetres by.
pub fn mean_chord(e: &SplineShape) -> f64 {
    let n = e.pts.len();
    let segments = if is_closed(e) { n } else { n.saturating_sub(1) };
    let mut sum = 0.0;
    for i in 0..segments {
        let d = e.pts[(i + 1) % n] - e.pts[i];
        sum += d.x.hypot(d.y);
    }
    if segments > 0 {
        sum / segments as f64
    } else {
        1.0
    }
}

/// What each of a spline's parameters is scaled by in its stay-put term, so the term reads in
/// millimetres: 1 for a fit point's coordinates, the mean chord (a micron at least) for an
/// end direction — the chord it swings.
pub fn spline_stay_units(e: &SplineShape) -> Vec<f64> {
    let fit_coords = 2 * e.pts.len();
    let chord = mean_chord(e).max(1e-3);
    (0..spline_param_count(e)).map(|i| if i >= fit_coords { chord } else { 1.0 }).collect()
}

/// A spline's share of the solve's typical size — what angle equations are weighed by to
/// read like length equations: the furthest two of its fit points lie apart.
pub fn spline_size(e: &SplineShape) -> f64 {
    let mut s = 0.0f64;
    for p in &e.pts {
        for q in &e.pts {
            let d = *p - *q;
            s = s.max(d.x.hypot(d.y));
        }
    }
    s
}

/// The unit direction a spline runs at one of its ends (`A` its start, anything else its
/// end): the one the solver holds where a tangent relation holds the end, the curve's own
/// elsewhere.
pub fn spline_end_dir(s: &SplineShape, end: AnchorAt) -> DVec2 {
    let angle = if end == AnchorAt::A { s.start_dir } else { s.end_dir };
    if let Some(angle) = angle {
        return DVec2::new(angle.cos(), angle.sin());
    }
    let fit = spline_curve(s);
    spline_dir_at(&fit, if end == AnchorAt::A { 0.0 } else { spline_end(&fit) })
}

/// Which ends of a spline a tangent relation holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeldEnds {
    pub a: bool,
    pub b: bool,
}

/// The ends of spline `spline` (shaped `e`) a tangent relation holds: those a tangent to
/// another piece meets it at, through a coincidence (`spline_joint`). A closed spline has no
/// ends to hold.
pub fn held_ends(constraints: &[Constraint], spline: EntityId, e: &SplineShape) -> HeldEnds {
    let mut held = HeldEnds::default();
    if is_closed(e) {
        return held;
    }
    for c in constraints {
        let Rule::Tangent { a, b } = c.rule else { continue };
        if a != spline && b != spline {
            continue;
        }
        let other = if a == spline { b } else { a };
        if let Some(joint) = spline_joint(constraints, spline, other) {
            if joint.end == AnchorAt::A {
                held.a = true;
            } else {
                held.b = true;
            }
        }
    }
    held
}

/// A spline with an end direction to solve for at each end a tangent relation holds — taken
/// from the curve as it runs there now — and none at an end nothing holds, which is natural
/// again. Run on every spline before the solve lays out its parameters.
pub fn with_held_ends(s: &SplineShape, held: HeldEnds) -> SplineShape {
    if held.a == s.start_dir.is_some() && held.b == s.end_dir.is_some() {
        return s.clone();
    }
    let fit = spline_curve(s);
    let dir = |t: f64| {
        let d = spline_dir_at(&fit, t);
        d.y.atan2(d.x)
    };
    let mut out = SplineShape { start_dir: None, end_dir: None, ..s.clone() };
    if held.a {
        out.start_dir = Some(s.start_dir.unwrap_or_else(|| dir(0.0)));
    }
    if held.b {
        out.end_dir = Some(s.end_dir.unwrap_or_else(|| dir(spline_end(&fit))));
    }
    out
}

/// A fitted spline's soft-data block. Each point is held against the place on the curve it
/// lay nearest when the solve began — both ways, not only off the curve: measured along the
/// normal alone, a fit point could slide along the curve for nothing, and an open curve's tip
/// runs off past its points. `refresh` finds the feet again where the curve has got to and
/// says how far they went, in the curve's own parameter; the solver then settles the curve
/// again, until they stay where they are.
///
/// The block's parameters are the spline's (`spline_param_count` of them from `offset`), its
/// weight 1/σ of the piece's scatter (a micron at least), and its residuals two per point.
/// The points handed in are the ones the solver takes — already thinned to its cap.
#[derive(Debug)]
pub struct SplineSoftData {
    template: SplineShape,
    offset: usize,
    px: Vec<f64>,
    py: Vec<f64>,
    feet: RefCell<Vec<f64>>,
}

impl SplineSoftData {
    /// The block for spline `e` — as laid out in the solve, its held ends given — whose
    /// parameters start at `offset`.
    pub fn new(e: &SplineShape, offset: usize, points: &[DVec2]) -> Self {
        let fit0 = spline_curve(e);
        let feet = points.iter().map(|p| nearest_on_spline(&fit0, *p, 8).t).collect();
        Self { template: e.clone(), offset, px: points.iter().map(|p| p.x).collect(), py: points.iter().map(|p| p.y).collect(), feet: RefCell::new(feet) }
    }

    /// The parameters the block depends on.
    pub fn params(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + spline_param_count(&self.template)
    }

    /// How many residuals the block gives: two a point.
    pub fn residual_count(&self) -> usize {
        2 * self.px.len()
    }

    /// The points' feet on the curve as they stand.
    pub fn feet(&self) -> Vec<f64> {
        self.feet.borrow().clone()
    }

    fn read(&self, xs: &[f64]) -> SplineShape {
        spline_with_params(&self.template, xs, self.offset)
    }

    /// The residuals, unweighted: each point less the curve at its foot, x then y.
    pub fn eval(&self, xs: &[f64]) -> Vec<f64> {
        let fit = spline_curve(&self.read(xs));
        let feet = self.feet.borrow();
        let mut out = vec![0.0; 2 * self.px.len()];
        for i in 0..self.px.len() {
            let c = spline_at(&fit, feet[i]);
            out[2 * i] = self.px[i] - c.x;
            out[2 * i + 1] = self.py[i] - c.y;
        }
        out
    }

    /// Takes the block's linearisation again where the solve has got to — the points finding
    /// their feet on the curve as it moved — and says how far the feet went.
    pub fn refresh(&self, xs: &[f64]) -> f64 {
        let fit = spline_curve(&self.read(xs));
        let mut feet = self.feet.borrow_mut();
        let mut moved = 0.0f64;
        for i in 0..self.px.len() {
            let t = nearest_on_spline(&fit, DVec2::new(self.px[i], self.py[i]), 8).t;
            moved = moved.max((t - feet[i]).abs());
            feet[i] = t;
        }
        moved
    }
}

/// The other piece at a spline's joint, as the solve reads it.
#[derive(Debug, Clone, Copy)]
pub enum JointPiece<'a> {
    Line { a: DVec2, b: DVec2 },
    Arc { start: f64, sweep: f64 },
    Spline(&'a SplineShape),
    /// A piece that has no one direction at the joint: the relation holds nothing.
    Other,
}

/// A tangent relation between a spline and another piece, held where the spline's end meets
/// it (`spline_joint`): the direction the spline runs there along the line, square to the
/// arc's radius, along the other spline's end — times `scale`, the solve's typical size, so
/// the equation weighs like a length. Off the joint a spline has no one direction to hold, and
/// the solver adds no equation at all.
pub fn spline_tangent_residual(s: &SplineShape, joint: SplineJoint, other: JointPiece, scale: f64) -> f64 {
    let u = spline_end_dir(s, joint.end);
    let at_end = matches!(joint.at, AnchorAt::A | AnchorAt::B);
    match other {
        JointPiece::Line { a, b } => {
            let d = normalize2(b - a).unwrap_or(DVec2::X);
            (u.x * d.y - u.y * d.x) * scale
        }
        JointPiece::Arc { start, sweep } if at_end => {
            let t = if joint.at == AnchorAt::A { start } else { start + sweep };
            let radial = DVec2::new(t.cos(), t.sin());
            (radial.x * u.x + radial.y * u.y) * scale
        }
        JointPiece::Spline(o) if at_end => {
            let v = spline_end_dir(o, joint.at);
            (u.x * v.y - u.y * v.x) * scale
        }
        _ => 0.0,
    }
}

/// A point held on a spline: its distance to the curve, signed by the side it lies on — zero
/// on it, and never on the line a tip runs on along past its end.
pub fn point_on_spline_residual(e: &SplineShape, p: DVec2) -> f64 {
    let fit = spline_curve(e);
    let near = nearest_on_spline(&fit, p, 8);
    let d = spline_dir_at(&fit, near.t);
    let side = (p.y - near.point.y) * d.x - (p.x - near.point.x) * d.y;
    if side < 0.0 {
        -near.dist
    } else {
        near.dist
    }
}

#[cfg(test)]
mod tests {
    //! The spline cases of ScanRuler's `reSpline.test.ts` ("the solver and a spline", "a chain
    //! fitted as one spline"). The Rust sketch solver has no spline piece yet, so they run on
    //! a harness that follows `solveSketch` step for step — held ends, stay-put and soft
    //! blocks, the hard equations these cases use, Levenberg–Marquardt with the refit rounds
    //! and the closing pass — and calls the functions above exactly where the solver will.

    use std::collections::HashMap;
    use std::rc::Rc;

    use super::*;
    use crate::chain::view_chain;
    use crate::linalg::{cholesky_solve, solve_linear};
    use crate::sketch::{anchor_point, Anchor, OnTarget, Shape};
    use crate::sketch_spline::{fit_points_along, spline_anchor_order, spline_anchor_point, spline_fit_point_index, spline_point_index, spline_scatter};

    const HARD_WEIGHT: f64 = 1e4;
    const FREE_WEIGHT: f64 = 1e-1;
    const RELEASE_ITERATIONS: usize = 6;
    const UNMET_TOLERANCE: f64 = 1e-3;
    const CLOSE_FROM: f64 = 1e-7;
    const MAX_POINTS: usize = 240;
    const MAX_ITERATIONS: usize = 60;

    #[derive(Debug, Clone)]
    enum Geo {
        Shape(Shape),
        Spline(SplineShape),
    }

    #[derive(Debug, Clone)]
    struct Piece {
        id: EntityId,
        geo: Geo,
        points: Vec<DVec2>,
        scatter: f64,
        fixed: bool,
    }

    /// A sketch with splines in it: the pieces, the relations as the sketch stores them, and
    /// coincidences at a spline's fit point (`p{i}`), which the stored anchors cannot name
    /// yet.
    #[derive(Debug, Default)]
    struct Model {
        pieces: Vec<Piece>,
        constraints: Vec<Constraint>,
        fit_pins: Vec<(EntityId, usize, Anchor)>,
        next: u32,
    }

    impl Model {
        fn add(&mut self, geo: Geo) -> EntityId {
            self.next += 1;
            self.pieces.push(Piece { id: self.next, geo, points: Vec::new(), scatter: 0.0, fixed: false });
            self.next
        }
        fn add_fixed(&mut self, geo: Geo) -> EntityId {
            let id = self.add(geo);
            self.pieces.last_mut().unwrap().fixed = true;
            id
        }
        fn constrain(&mut self, rule: Rule) -> u32 {
            self.next += 1;
            self.constraints.push(Constraint { id: self.next, rule, inferred: false, expr: None, place: None });
            self.next
        }
        fn spline(&self, id: EntityId) -> SplineShape {
            match &self.pieces.iter().find(|p| p.id == id).unwrap().geo {
                Geo::Spline(s) => s.clone(),
                Geo::Shape(_) => panic!("not a spline"),
            }
        }
        fn shape(&self, id: EntityId) -> Shape {
            match &self.pieces.iter().find(|p| p.id == id).unwrap().geo {
                Geo::Shape(s) => s.clone(),
                Geo::Spline(_) => panic!("not a shape"),
            }
        }
    }

    fn params_of(g: &Geo) -> Vec<f64> {
        match g {
            Geo::Shape(Shape::Line { a, b }) => vec![a.x, a.y, b.x, b.y],
            Geo::Shape(Shape::Arc { c, r, start, sweep }) => vec![c.x, c.y, *r, *start, *sweep],
            Geo::Shape(Shape::Circle { c, r }) => vec![c.x, c.y, *r],
            Geo::Shape(Shape::Point { p }) => vec![p.x, p.y],
            Geo::Shape(Shape::Spline(s)) | Geo::Spline(s) => spline_params(s),
        }
    }

    fn with_params(g: &Geo, x: &[f64], at: usize) -> Geo {
        match g {
            Geo::Shape(Shape::Line { .. }) => Geo::Shape(Shape::Line { a: DVec2::new(x[at], x[at + 1]), b: DVec2::new(x[at + 2], x[at + 3]) }),
            Geo::Shape(Shape::Arc { .. }) => Geo::Shape(Shape::Arc { c: DVec2::new(x[at], x[at + 1]), r: x[at + 2], start: x[at + 3], sweep: x[at + 4] }),
            Geo::Shape(Shape::Circle { .. }) => Geo::Shape(Shape::Circle { c: DVec2::new(x[at], x[at + 1]), r: x[at + 2] }),
            Geo::Shape(Shape::Point { .. }) => Geo::Shape(Shape::Point { p: DVec2::new(x[at], x[at + 1]) }),
            Geo::Shape(Shape::Spline(s)) | Geo::Spline(s) => Geo::Spline(spline_with_params(s, x, at)),
        }
    }

    struct Slots {
        map: HashMap<EntityId, (usize, usize, Geo)>,
    }

    impl Slots {
        fn read(&self, id: EntityId, xs: &[f64]) -> Option<Geo> {
            self.map.get(&id).map(|(at, _, g)| with_params(g, xs, *at))
        }
        fn params_for(&self, ids: &[EntityId]) -> Vec<usize> {
            let mut out = Vec::new();
            let mut seen = Vec::new();
            for id in ids {
                if seen.contains(id) {
                    continue;
                }
                seen.push(*id);
                if let Some((at, count, _)) = self.map.get(id) {
                    out.extend(*at..*at + *count);
                }
            }
            out
        }
        fn anchor(&self, xs: &[f64], a: Anchor) -> DVec2 {
            match self.read(a.entity, xs) {
                Some(Geo::Shape(s)) => anchor_point(&s, a.at),
                Some(Geo::Spline(s)) => spline_anchor_point(&s, spline_point_index(&s, a.at)),
                None => DVec2::ZERO,
            }
        }
        fn fit_point(&self, xs: &[f64], spline: EntityId, i: usize) -> DVec2 {
            match self.read(spline, xs) {
                Some(Geo::Spline(s)) => spline_anchor_point(&s, spline_fit_point_index(&s, i)),
                _ => DVec2::ZERO,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Stay,
        Soft,
        Hard,
    }

    type Eval<'a> = Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>;

    struct Block<'a> {
        params: Vec<usize>,
        weight: f64,
        kind: Kind,
        eval: Eval<'a>,
        refresh: Option<Rc<SplineSoftData>>,
    }

    fn thin(points: &[DVec2]) -> Vec<DVec2> {
        if points.len() <= MAX_POINTS {
            return points.to_vec();
        }
        let step = points.len() as f64 / MAX_POINTS as f64;
        (0..MAX_POINTS).map(|i| points[(i as f64 * step) as usize]).collect()
    }

    fn cost(rs: &[Vec<f64>]) -> f64 {
        rs.iter().flat_map(|r| r.iter()).map(|v| v * v).sum()
    }

    #[allow(clippy::too_many_arguments)]
    fn descend(active: &[&Block], x: &mut [f64], free: &[bool], free_index: &[usize], m: usize, max_iterations: usize, step_tolerance: f64, gain_tolerance: f64) {
        let residuals = |xs: &[f64]| -> Vec<Vec<f64>> { active.iter().map(|b| (b.eval)(xs).into_iter().map(|r| r * b.weight).collect()).collect() };
        let n = x.len();
        let mut lambda = 1e-3;
        let mut rs = residuals(x);
        let mut current = cost(&rs);
        let mut jtj = vec![0.0; m * m];
        let mut jtr = vec![0.0; m];
        let mut xt = x.to_vec();
        let mut converged = false;
        let mut it = 0;
        while it < max_iterations && !converged {
            it += 1;
            jtj.iter_mut().for_each(|v| *v = 0.0);
            jtr.iter_mut().for_each(|v| *v = 0.0);
            xt.copy_from_slice(x);
            for (bi, b) in active.iter().enumerate() {
                let r0 = &rs[bi];
                let cols: Vec<usize> = b.params.iter().copied().filter(|&p| free[p]).collect();
                if cols.is_empty() {
                    continue;
                }
                let mut jac: Vec<Vec<f64>> = Vec::with_capacity(cols.len());
                for &p in &cols {
                    let h = 1e-6 * x[p].abs().max(1.0);
                    xt[p] = x[p] + h;
                    let rp = (b.eval)(&xt);
                    xt[p] = x[p] - h;
                    let rm = (b.eval)(&xt);
                    xt[p] = x[p];
                    let w = b.weight / (2.0 * h);
                    jac.push(rp.iter().zip(&rm).map(|(a, b)| (a - b) * w).collect());
                }
                for i in 0..cols.len() {
                    let ci = free_index[cols[i]];
                    let g: f64 = jac[i].iter().zip(r0).map(|(j, r)| j * r).sum();
                    jtr[ci] += g;
                    for j in 0..cols.len() {
                        let cj = free_index[cols[j]];
                        let s: f64 = jac[i].iter().zip(&jac[j]).map(|(a, b)| a * b).sum();
                        jtj[ci * m + cj] += s;
                    }
                }
            }
            let grad_norm = jtr.iter().fold(0.0f64, |g, v| g.max(v.abs()));
            if grad_norm < 1e-9 {
                break;
            }
            let mut improved = false;
            for _ in 0..12 {
                let s: Vec<f64> = (0..m).map(|i| 1.0 / (jtj[i * m + i] * (1.0 + lambda) + 1e-30).sqrt()).collect();
                let mut a = vec![0.0; m * m];
                for i in 0..m {
                    for j in 0..m {
                        a[i * m + j] = jtj[i * m + j] * s[i] * s[j];
                    }
                    a[i * m + i] += lambda * jtj[i * m + i] * s[i] * s[i];
                }
                let rhs: Vec<f64> = (0..m).map(|i| -jtr[i] * s[i]).collect();
                let scaled = cholesky_solve(m, &a, &rhs, 1e-12).or_else(|| solve_linear(m, &mut a.clone(), &rhs));
                let Some(scaled) = scaled else {
                    lambda *= 10.0;
                    continue;
                };
                xt.copy_from_slice(x);
                let mut step_norm = 0.0f64;
                for i in 0..n {
                    if free[i] {
                        let d = scaled[free_index[i]] * s[free_index[i]];
                        xt[i] += d;
                        step_norm = step_norm.max(d.abs());
                    }
                }
                let rt = residuals(&xt);
                let ct = cost(&rt);
                if ct < current {
                    x.copy_from_slice(&xt);
                    rs = rt;
                    let gain = current - ct;
                    current = ct;
                    lambda = (lambda / 3.0).max(1e-12);
                    improved = true;
                    if step_norm < step_tolerance || gain < gain_tolerance * current.max(1.0) {
                        converged = true;
                    }
                    break;
                }
                lambda *= 4.0;
            }
            if !improved {
                break;
            }
        }
    }

    /// A hard equation's block.
    fn hard<'a>(slots: &Slots, ids: &[EntityId], eval: Eval<'a>, blocks: &mut Vec<Block<'a>>) {
        blocks.push(Block { params: slots.params_for(ids), weight: HARD_WEIGHT, kind: Kind::Hard, eval, refresh: None });
    }

    /// `solveSketch` for these cases; whether every hard equation was met.
    fn solve(model: &mut Model) -> bool {
        // Every spline with an end direction to solve for at each end a tangent holds.
        for i in 0..model.pieces.len() {
            if let Geo::Spline(s) = &model.pieces[i].geo {
                let held = held_ends(&model.constraints, model.pieces[i].id, s);
                model.pieces[i].geo = Geo::Spline(with_held_ends(s, held));
            }
        }
        let mut map = HashMap::new();
        let mut x0: Vec<f64> = Vec::new();
        for p in &model.pieces {
            let ps = params_of(&p.geo);
            map.insert(p.id, (x0.len(), ps.len(), p.geo.clone()));
            x0.extend(ps);
        }
        let n = x0.len();
        let mut x = x0.clone();
        let slots = Slots { map };
        let slots = &slots;
        let mut free = vec![true; n];
        for p in &model.pieces {
            if p.fixed {
                let (at, count, _) = &slots.map[&p.id];
                free[*at..*at + *count].iter_mut().for_each(|f| *f = false);
            }
        }

        let mut blocks: Vec<Block> = Vec::new();
        let mut soft_count = 0;
        let mut rests: Vec<(usize, Rc<RefCell<Vec<f64>>>)> = Vec::new();
        for p in &model.pieces {
            if p.fixed {
                continue;
            }
            let (at, count, geo) = slots.map[&p.id].clone();
            let init = Rc::new(RefCell::new(x0[at..at + count].to_vec()));
            rests.push((at, init.clone()));
            let unit: Vec<f64> = match &geo {
                Geo::Spline(s) => spline_stay_units(s),
                Geo::Shape(Shape::Arc { r, .. }) => (0..count).map(|i| if i >= 3 { r.abs().max(1e-3) } else { 1.0 }).collect(),
                Geo::Shape(_) => vec![1.0; count],
            };
            blocks.push(Block {
                params: (at..at + count).collect(),
                weight: FREE_WEIGHT,
                kind: Kind::Stay,
                eval: Box::new(move |xs| init.borrow().iter().enumerate().map(|(i, v)| (xs[at + i] - v) * unit[i]).collect()),
                refresh: None,
            });
            if p.points.is_empty() {
                continue;
            }
            let points = thin(&p.points);
            let scatter = p.scatter.max(1e-3);
            soft_count += points.len();
            if let Geo::Spline(s) = &geo {
                let soft = Rc::new(SplineSoftData::new(s, at, &points));
                assert_eq!(soft.params(), at..at + count);
                let evaluated = soft.clone();
                blocks.push(Block { params: soft.params().collect(), weight: 1.0 / scatter, kind: Kind::Soft, eval: Box::new(move |xs| evaluated.eval(xs)), refresh: Some(soft) });
            }
        }

        // A typical size, so angle equations weigh like length equations.
        let mut scale = 0.0f64;
        for p in &model.pieces {
            match &p.geo {
                Geo::Shape(Shape::Line { a, b }) => scale = scale.max((*b - *a).length()),
                Geo::Shape(Shape::Arc { r, .. }) | Geo::Shape(Shape::Circle { r, .. }) => scale = scale.max(2.0 * r),
                Geo::Shape(Shape::Point { .. }) => {}
                Geo::Shape(Shape::Spline(s)) | Geo::Spline(s) => scale = scale.max(spline_size(s)),
            }
        }
        let l_scale = scale.max(1.0);

        let is_spline = |id: EntityId| matches!(slots.map.get(&id), Some((_, _, Geo::Spline(_))));
        let constraints = model.constraints.clone();
        for c in &constraints {
            match c.rule.clone() {
                Rule::Coincident { p, q } => hard(
                    slots,
                    &[p.entity, q.entity],
                    Box::new(move |xs| {
                        let (a, b) = (slots.anchor(xs, p), slots.anchor(xs, q));
                        vec![a.x - b.x, a.y - b.y]
                    }),
                    &mut blocks,
                ),
                Rule::Tangent { a, b } if is_spline(a) || is_spline(b) => {
                    let spline = if is_spline(a) { a } else { b };
                    let other = if spline == a { b } else { a };
                    let Some(joint) = spline_joint(&constraints, spline, other) else { continue };
                    hard(
                        slots,
                        &[a, b],
                        Box::new(move |xs| {
                            let Some(Geo::Spline(s)) = slots.read(spline, xs) else { return vec![0.0] };
                            let o = slots.read(other, xs);
                            let piece = match &o {
                                Some(Geo::Shape(Shape::Line { a, b })) => JointPiece::Line { a: *a, b: *b },
                                Some(Geo::Shape(Shape::Arc { start, sweep, .. })) => JointPiece::Arc { start: *start, sweep: *sweep },
                                Some(Geo::Spline(o)) => JointPiece::Spline(o),
                                _ => JointPiece::Other,
                            };
                            vec![spline_tangent_residual(&s, joint, piece, l_scale)]
                        }),
                        &mut blocks,
                    );
                }
                Rule::PointOn { p, on: OnTarget::Curve(cv), .. } => hard(
                    slots,
                    &[p.entity, cv],
                    Box::new(move |xs| {
                        let pp = slots.anchor(xs, p);
                        match slots.read(cv, xs) {
                            Some(Geo::Spline(s)) => vec![point_on_spline_residual(&s, pp)],
                            Some(Geo::Shape(Shape::Arc { c, r, .. })) | Some(Geo::Shape(Shape::Circle { c, r })) => vec![(pp - c).length() - r],
                            _ => vec![0.0],
                        }
                    }),
                    &mut blocks,
                ),
                other => panic!("the harness does not hold {other:?}"),
            }
        }
        for &(spline, i, q) in &model.fit_pins {
            hard(
                slots,
                &[spline, q.entity],
                Box::new(move |xs| {
                    let (a, b) = (slots.fit_point(xs, spline, i), slots.anchor(xs, q));
                    vec![a.x - b.x, a.y - b.y]
                }),
                &mut blocks,
            );
        }

        let mut free_index = vec![usize::MAX; n];
        let mut m = 0;
        for i in 0..n {
            if free[i] {
                free_index[i] = m;
                m += 1;
            }
        }
        if m > 0 {
            let all: Vec<&Block> = blocks.iter().collect();
            descend(&all, &mut x, &free, &free_index, m, MAX_ITERATIONS, 1e-9, 1e-12);
            // A fitted spline's points found their feet on the curve where it began; where it
            // has moved they are found again and the curve settled to them, until the feet
            // stay put.
            let refreshed: Vec<&Rc<SplineSoftData>> = blocks.iter().filter_map(|b| b.refresh.as_ref()).collect();
            for _ in 0..REFIT_ROUNDS {
                if refreshed.is_empty() {
                    break;
                }
                let moved = refreshed.iter().fold(0.0f64, |w, b| w.max(b.refresh(&x)));
                if moved < REFIT_SETTLED {
                    break;
                }
                descend(&all, &mut x, &free, &free_index, m, MAX_ITERATIONS, 1e-9, 1e-12);
            }
            let worst = blocks.iter().filter(|b| b.kind == Kind::Hard).flat_map(|b| (b.eval)(&x)).fold(0.0f64, |w, v| w.max(v.abs()));
            if soft_count > 0 && worst > CLOSE_FROM {
                for (at, init) in &rests {
                    let mut v = init.borrow_mut();
                    let len = v.len();
                    v.copy_from_slice(&x[*at..*at + len]);
                }
                let active: Vec<&Block> = blocks.iter().filter(|b| b.kind == Kind::Hard || b.kind == Kind::Stay).collect();
                descend(&active, &mut x, &free, &free_index, m, RELEASE_ITERATIONS, 1e-9, 1e-12);
            }
        }
        let ok = blocks.iter().filter(|b| b.kind == Kind::Hard).all(|b| (b.eval)(&x).iter().all(|v| v.abs() <= UNMET_TOLERANCE));
        // Written back as `finish` writes it: a round piece's radius positive, an arc's sweep
        // positive and within a turn, its start within a turn.
        for p in &mut model.pieces {
            let (at, _, geo) = &slots.map[&p.id];
            p.geo = match with_params(geo, &x, *at) {
                Geo::Shape(Shape::Circle { c, r }) => Geo::Shape(Shape::Circle { c, r: r.abs() }),
                Geo::Shape(Shape::Arc { c, r, mut start, mut sweep }) => {
                    if sweep < 0.0 {
                        start += sweep;
                        sweep = -sweep;
                    }
                    let two = 2.0 * std::f64::consts::PI;
                    let wrapped = ((sweep % two) + two) % two;
                    sweep = if wrapped < 1e-9 && sweep > 1e-9 { two } else { wrapped };
                    start = ((start % two) + two) % two;
                    Geo::Shape(Shape::Arc { c, r: r.abs(), start, sweep })
                }
                g => g,
            };
        }
        ok
    }

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn near(a: DVec2, b: DVec2) -> f64 {
        (a.x - b.x).hypot(a.y - b.y)
    }

    #[test]
    fn holds_a_spline_tangent_to_a_line_where_an_end_of_it_is_joined_and_lets_go_when_the_relation_goes() {
        let mut s = Model::default();
        let line = s.add_fixed(Geo::Shape(Shape::Line { a: v(0.0, 0.0), b: v(20.0, 0.0) }));
        let sp = s.add(Geo::Spline(SplineShape::open(vec![v(20.0, 0.0), v(30.0, 8.0), v(45.0, 10.0)])));
        s.constrain(Rule::Coincident { p: Anchor::new(line, AnchorAt::B), q: Anchor::new(sp, AnchorAt::A) });
        let tangent = s.constrain(Rule::Tangent { a: line, b: sp });
        assert!(solve(&mut s));
        let e = s.spline(sp);
        assert!(e.start_dir.is_some());
        let d = spline_dir_at(&spline_curve(&e), 0.0);
        assert!(d.y.abs() < 1e-6);
        assert!(near(e.pts[0], v(20.0, 0.0)) < 1e-6);
        s.constraints.retain(|c| c.id != tangent);
        solve(&mut s);
        assert!(s.spline(sp).start_dir.is_none());
    }

    #[test]
    fn holds_two_splines_tangent_end_to_end_and_a_spline_to_an_arc() {
        let mut s = Model::default();
        let a = s.add(Geo::Spline(SplineShape::open(vec![v(0.0, 0.0), v(10.0, 6.0), v(20.0, 0.0)])));
        let b = s.add(Geo::Spline(SplineShape::open(vec![v(20.0, 0.0), v(30.0, 5.0), v(40.0, 2.0)])));
        s.constrain(Rule::Coincident { p: Anchor::new(a, AnchorAt::B), q: Anchor::new(b, AnchorAt::A) });
        s.constrain(Rule::Tangent { a, b });
        let arc = s.add(Geo::Shape(Shape::Arc { c: v(40.0, 12.0), r: 10.0, start: -std::f64::consts::PI / 2.0, sweep: std::f64::consts::PI / 2.0 }));
        s.constrain(Rule::Coincident { p: Anchor::new(b, AnchorAt::B), q: Anchor::new(arc, AnchorAt::A) });
        s.constrain(Rule::Tangent { a: b, b: arc });
        assert!(solve(&mut s));
        let ea = spline_curve(&s.spline(a));
        let eb = spline_curve(&s.spline(b));
        let u = spline_dir_at(&ea, spline_end(&ea));
        let w0 = spline_dir_at(&eb, 0.0);
        assert!((u.x * w0.y - u.y * w0.x).abs() < 1e-6);
        let Shape::Arc { start, .. } = s.shape(arc) else { panic!("no arc") };
        let radial = v(start.cos(), start.sin());
        let w = spline_dir_at(&eb, spline_end(&eb));
        assert!((radial.x * w.x + radial.y * w.y).abs() < 1e-6);
    }

    #[test]
    fn pulls_a_fitted_spline_onto_its_slice_points_closer_than_the_curve_through_points_on_them() {
        let points: Vec<DVec2> = (0..201).map(|i| v(i as f64 * 0.2, 10.0 * ((i as f64 * 0.2) / 10.0).sin())).collect();
        let xs = [0.0, 40.0 / 6.0, 80.0 / 6.0, 20.0, 160.0 / 6.0, 200.0 / 6.0, 40.0];
        // Through seven points on the sine itself, the natural ends miss it by some
        // hundredths; the fit, free to move its points, misses it by less.
        let through = spline_scatter(&SplineShape::open(xs.iter().map(|x| v(*x, 10.0 * (x / 10.0).sin())).collect()), &points);
        let mut s = Model::default();
        let sp = s.add(Geo::Spline(SplineShape::open(xs.iter().map(|x| v(*x, 10.0 * (x / 10.0).sin() + 0.7)).collect())));
        s.pieces[0].points = points.clone();
        s.pieces[0].scatter = 0.01;
        assert!(spline_scatter(&s.spline(sp), &points) > 0.5);
        assert!(solve(&mut s));
        let after = spline_scatter(&s.spline(sp), &points);
        assert!(after < through, "{after} against {through}");
        assert!(after < 0.05);
    }

    #[test]
    fn holds_a_point_on_a_spline() {
        let mut s = Model::default();
        let sp = s.add_fixed(Geo::Spline(SplineShape::open(vec![v(0.0, 0.0), v(15.0, 12.0), v(30.0, 0.0)])));
        // A hair off the curve, as a point snapped onto it is left.
        let on = spline_at(&spline_curve(&s.spline(sp)), 0.9);
        let pt = s.add(Geo::Shape(Shape::Point { p: v(on.x, on.y + 0.01) }));
        s.constrain(Rule::PointOn { p: Anchor::new(pt, AnchorAt::Center), on: OnTarget::Curve(sp), bounded: false });
        assert!(solve(&mut s));
        let Shape::Point { p } = s.shape(pt) else { panic!("no point") };
        assert!(nearest_on_spline(&spline_curve(&s.spline(sp)), p, 16).dist < 1e-5);
    }

    #[test]
    fn a_fit_point_is_a_point_to_drag_and_to_join_the_curve_bends_through_it() {
        let mut s = Model::default();
        let sp = s.add(Geo::Spline(SplineShape::open(vec![v(0.0, 0.0), v(10.0, 5.0), v(20.0, 0.0)])));
        // The anchors a, b, p1: the ends first, then the point between.
        assert_eq!(spline_anchor_order(&s.spline(sp)), vec![0, 2, 1]);
        // Dragged, a fit point goes where the hand is (drag.ts moveAnchor's spline case).
        let mut moved = s.spline(sp);
        let i = spline_fit_point_index(&moved, 1).unwrap();
        moved.pts[i] = v(10.0, 9.0);
        assert_eq!(spline_anchor_point(&moved, spline_fit_point_index(&moved, 1)), v(10.0, 9.0));
        let pt = s.add_fixed(Geo::Shape(Shape::Point { p: v(11.0, 12.0) }));
        s.fit_pins.push((sp, 1, Anchor::new(pt, AnchorAt::Center)));
        assert!(solve(&mut s));
        assert!(near(s.spline(sp).pts[1], v(11.0, 12.0)) < 1e-6);
    }

    #[test]
    fn lays_a_closed_spline_round_a_closed_chain_and_settles_it_onto_the_points() {
        // An ellipse 30 by 18, sampled every degree or so.
        let chain: Vec<DVec2> = (0..360).map(|i| v(30.0 * (2.0 * std::f64::consts::PI * i as f64 / 360.0).cos(), 18.0 * (2.0 * std::f64::consts::PI * i as f64 / 360.0).sin())).collect();
        // What fitChain does with the spline option at 8 and no tolerance.
        let view = view_chain(&chain);
        let count = 8usize.max(if view.closed { 3 } else { 2 });
        assert!(view.points.len() >= 2 * count);
        let pts = fit_points_along(&view.points, count, view.closed).unwrap();
        let mut s = Model::default();
        let id = s.add(Geo::Spline(SplineShape { pts, closed: view.closed, ..SplineShape::default() }));
        s.pieces[0].points = view.points.clone();
        s.pieces[0].scatter = view.scatter.max(1e-3);
        let e = s.spline(id);
        assert!(e.closed);
        assert_eq!(e.pts.len(), 8);
        // Closer than the curve through eight points on the ellipse itself.
        let through = spline_scatter(
            &SplineShape::closed((0..8).map(|i| v(30.0 * (2.0 * std::f64::consts::PI * i as f64 / 8.0).cos(), 18.0 * (2.0 * std::f64::consts::PI * i as f64 / 8.0).sin())).collect()),
            &chain,
        );
        solve(&mut s);
        assert!(spline_scatter(&s.spline(id), &chain) < through);
    }

    // Beyond the ScanRuler tests: the pieces the solver lays out and reads back.

    #[test]
    fn a_splines_parameters_round_trip_with_its_held_ends() {
        let e = SplineShape { pts: vec![v(0.0, 0.0), v(10.0, 5.0), v(20.0, 0.0)], closed: false, start_dir: Some(0.25), end_dir: None };
        let x = spline_params(&e);
        assert_eq!(x, vec![0.0, 0.0, 10.0, 5.0, 20.0, 0.0, 0.25]);
        let mut padded = vec![7.0, 7.0];
        padded.extend(&x);
        assert_eq!(spline_with_params(&e, &padded, 2), e);
        let units = spline_stay_units(&e);
        assert_eq!(units.len(), 7);
        assert!((units[6] - mean_chord(&e)).abs() < 1e-15);
        assert!((spline_size(&e) - 20.0).abs() < 1e-12);
        let held = with_held_ends(&e, HeldEnds { a: true, b: true });
        assert_eq!(held.start_dir, Some(0.25));
        let end = spline_dir_at(&spline_curve(&e), 2.0);
        assert!((held.end_dir.unwrap() - end.y.atan2(end.x)).abs() < 1e-15);
        assert_eq!(with_held_ends(&held, HeldEnds::default()).start_dir, None);
        let free = SplineShape::open(e.pts.clone());
        assert!(point_on_spline_residual(&free, v(10.0, 5.0)).abs() < 1e-9);
        assert!(point_on_spline_residual(&free, v(10.0, 7.0)) > 1.9);
        assert!(point_on_spline_residual(&free, v(10.0, 3.0)) < -1.9);
    }
}
