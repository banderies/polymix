// SPDX-License-Identifier: AGPL-3.0-only
//! One solver for the sketch: constrained least squares. Ported from ScanRuler
//! (`plugins/reverse/core/solver.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A fitted piece keeps the slice points it was segmented from, and they are soft data — the
//! residual of every point off its line or arc, weighted by the chain's scatter, is what the
//! solver minimises. A relation or a typed dimension is a hard equation. Rather than eliminate
//! the hard equations, they go into the same least squares at a weight so far above the soft
//! data that they are met to a fraction of a micron while the soft data settles the rest;
//! afterwards their residuals are read back, and one that could not be met is reported as
//! such rather than split down the middle. Free geometry, with no points behind it, gets a
//! weak wish to stay where it was drawn.
//!
//! Levenberg–Marquardt on the stacked residual, with the Jacobian taken by finite differences
//! block by block: every residual block names the few parameters it depends on. The normal
//! equations are solved directly — a sketch has a few hundred parameters at most.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::f64::consts::{PI, TAU};
use std::rc::Rc;

use glam::DVec2;

use crate::datums::{trace_of, DatumTrace};
use crate::linalg::{cholesky_solve, solve_linear, symmetric_eigen_n};
use crate::sketch::{anchor_point, Anchor, AnchorAt, ConstraintId, DistanceTo, EntityId, MirrorAxis, OnTarget, Rule, Shape, ShapeKind, Sketch};

/// How the hard equations outweigh the soft data, whose weight is 1/σ.
const HARD_WEIGHT: f64 = 1e4;
/// The stay-put weight of free geometry, against soft data of weight 1/σ.
const FREE_WEIGHT: f64 = 1e-1;
/// The pull of a dragged point towards the cursor: above the stay-put and the soft data,
/// under the hard equations, so a point dragged off its horizontal slides along it.
const PULL_WEIGHT: f64 = 1e2;
/// The iterations the let-go pass is allowed.
const RELEASE_ITERATIONS: usize = 6;
/// A hard residual over this, millimetres, after convergence is a constraint not met.
const UNMET_TOLERANCE: f64 = 1e-3;
/// A relation held open by more than this against fitted pieces is closed by a pass without
/// the slice points.
const CLOSE_FROM: f64 = 1e-7;
/// Slice points per piece the residuals are taken over.
const MAX_POINTS: usize = 240;
const MAX_ITERATIONS: usize = 60;
/// The iterations a quick solve (one step of a drag) is allowed.
const QUICK_ITERATIONS: usize = 4;

/// A point being dragged: the anchor and where the hand is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pull {
    pub anchor: Anchor,
    pub to: DVec2,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SolveOptions<'a> {
    /// Points being dragged, pulled to the cursor while everything else follows.
    pub pull: &'a [Pull],
    /// A step of a drag: a few iterations and a looser stop.
    pub quick: bool,
    /// The part's coordinate planes as lines in this sketch's sheet.
    pub datums: &'a [DatumTrace],
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolveReport {
    /// Whether every hard equation was met.
    pub ok: bool,
    pub iterations: usize,
    /// Constraints that were not met, with how far off they ended.
    pub unmet: Vec<(ConstraintId, f64)>,
    /// How many hard equations say nothing new. Zero on a healthy sketch.
    pub redundant: usize,
    /// How many hard equations there are, the redundant ones counted.
    pub equations: usize,
    /// The RMS of the soft residuals — how well the pieces sit on their slice points, mm.
    pub soft_rms: f64,
    /// The pieces that can still move: a parameter no relation or dimension pins down. Fixed,
    /// projected and fitted pieces count as held.
    pub under_constrained: Vec<EntityId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Stay,
    Soft,
    Pull,
    Hard,
}

type Eval<'a> = Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>;

struct Block<'a> {
    params: Vec<usize>,
    weight: f64,
    kind: BlockKind,
    constraint: Option<ConstraintId>,
    /// A bound rather than an equation: met or not, but no row of the constraint Jacobian.
    bound: bool,
    eval: Eval<'a>,
}

fn params_of(s: &Shape) -> Vec<f64> {
    if let Shape::Spline(e) = s {
        return crate::spline_solve::spline_params(e);
    }
    match *s {
        Shape::Spline(_) => Vec::new(),
        Shape::Line { a, b } => vec![a.x, a.y, b.x, b.y],
        Shape::Arc { c, r, start, sweep } => vec![c.x, c.y, r, start, sweep],
        Shape::Circle { c, r } => vec![c.x, c.y, r],
        Shape::Point { p } => vec![p.x, p.y],
    }
}

fn with_params(kind: ShapeKind, x: &[f64], at: usize) -> Shape {
    match kind {
        ShapeKind::Line => Shape::Line { a: DVec2::new(x[at], x[at + 1]), b: DVec2::new(x[at + 2], x[at + 3]) },
        ShapeKind::Arc => Shape::Arc { c: DVec2::new(x[at], x[at + 1]), r: x[at + 2], start: x[at + 3], sweep: x[at + 4] },
        ShapeKind::Circle => Shape::Circle { c: DVec2::new(x[at], x[at + 1]), r: x[at + 2] },
        ShapeKind::Point => Shape::Point { p: DVec2::new(x[at], x[at + 1]) },
        // A spline is read through its template (`Slots::read`); this is never reached.
        ShapeKind::Spline => Shape::Point { p: DVec2::new(x[at], x[at + 1]) },
    }
}

/// Where each entity's parameters live in x.
struct Slots {
    map: HashMap<EntityId, (usize, usize, ShapeKind)>,
    /// Each spline as laid out — its point count, its closure and its held ends — which its
    /// parameters are read back through.
    splines: HashMap<EntityId, crate::sketch_spline::SplineShape>,
}

impl Slots {
    fn read(&self, id: EntityId, xs: &[f64]) -> Option<Shape> {
        let &(at, _, kind) = self.map.get(&id)?;
        if kind == ShapeKind::Spline {
            return self.splines.get(&id).map(|t| Shape::Spline(crate::spline_solve::spline_with_params(t, xs, at)));
        }
        Some(with_params(kind, xs, at))
    }

    fn range(&self, id: EntityId) -> Vec<usize> {
        self.map.get(&id).map(|&(at, count, _)| (at..at + count).collect()).unwrap_or_default()
    }

    fn params_for(&self, ids: &[EntityId]) -> Vec<usize> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for id in ids {
            if seen.insert(*id) {
                out.extend(self.range(*id));
            }
        }
        out
    }

    fn anchor(&self, xs: &[f64], a: Anchor) -> DVec2 {
        self.read(a.entity, xs).map_or(DVec2::ZERO, |s| anchor_point(&s, a.at))
    }

    fn line(&self, xs: &[f64], id: EntityId) -> Option<(DVec2, DVec2)> {
        match self.read(id, xs)? {
            Shape::Line { a, b } => Some((a, b)),
            _ => None,
        }
    }

    fn dir(&self, xs: &[f64], id: EntityId) -> DVec2 {
        self.line(xs, id).and_then(|(a, b)| (b - a).try_normalize()).unwrap_or(DVec2::X)
    }
}

/// Subsample a piece's points evenly to the cap.
fn thin(points: &[DVec2]) -> Vec<DVec2> {
    if points.len() <= MAX_POINTS {
        return points.to_vec();
    }
    let step = points.len() as f64 / MAX_POINTS as f64;
    (0..MAX_POINTS).map(|i| points[(i as f64 * step) as usize]).collect()
}

fn cross(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}

/// The unit radius of an arc at one of its ends.
fn radial_at(start: f64, sweep: f64, at: AnchorAt) -> DVec2 {
    let t = if at == AnchorAt::A { start } else { start + sweep };
    DVec2::new(t.cos(), t.sin())
}

/// Solve the sketch in place: every entity's parameters are moved to the least-squares
/// optimum under its constraints. Fixed entities keep their parameters; entities with points
/// are pulled to them; free ones stay near where they were.
pub fn solve_sketch(sketch: &mut Sketch, options: SolveOptions) -> SolveReport {
    // Every spline with an end direction to solve for where a tangent relation holds it, and
    // none where nothing does (ScanRuler's `withHeldEnds`), before its parameters are laid out.
    let held: Vec<(usize, crate::sketch_spline::SplineShape)> = sketch
        .entities
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.shape.spline().map(|sp| (i, crate::spline_solve::with_held_ends(sp, crate::spline_solve::held_ends(&sketch.constraints, e.id, sp)))))
        .collect();
    for (i, sp) in held {
        sketch.entities[i].shape = Shape::Spline(sp);
    }
    let mut map = HashMap::new();
    let mut splines = HashMap::new();
    let mut x0: Vec<f64> = Vec::new();
    for e in &sketch.entities {
        let p = params_of(&e.shape);
        map.insert(e.id, (x0.len(), p.len(), e.shape.kind()));
        if let Shape::Spline(sp) = &e.shape {
            splines.insert(e.id, sp.clone());
        }
        x0.extend(p);
    }
    let n = x0.len();
    let mut x = x0.clone();
    if n == 0 {
        return SolveReport { ok: true, ..Default::default() };
    }
    let slots = Slots { map, splines };
    let slots = &slots;

    // Which parameters may move: a fixed entity's are left out of the unknowns.
    let mut free = vec![true; n];
    for e in &sketch.entities {
        if e.fixed || e.projected {
            let (at, count, _) = slots.map[&e.id];
            free[at..at + count].iter_mut().for_each(|f| *f = false);
        }
    }

    let mut blocks: Vec<Block> = Vec::new();
    // ---- soft data and stay-put terms ----
    // Every piece keeps a weak wish to stay where it is: on free geometry it is all that
    // holds the piece, on a fitted one it settles what the points cannot. Angles are scaled
    // by the radius so the term reads in millimetres.
    let mut soft_count = 0usize;
    let mut rests: Vec<(usize, Rc<RefCell<Vec<f64>>>)> = Vec::new();
    let mut refits: Vec<Rc<crate::spline_solve::SplineSoftData>> = Vec::new();
    for e in &sketch.entities {
        if e.fixed || e.projected {
            continue;
        }
        let (at, count, _) = slots.map[&e.id];
        let init = Rc::new(RefCell::new(x0[at..at + count].to_vec()));
        rests.push((at, init.clone()));
        let unit: Vec<f64> = match &e.shape {
            Shape::Spline(sp) => crate::spline_solve::spline_stay_units(sp),
            shape => (0..count)
                .map(|i| match *shape {
                    Shape::Arc { r, .. } if i >= 3 => r.abs().max(1e-3),
                    _ => 1.0,
                })
                .collect(),
        };
        blocks.push(Block {
            params: (at..at + count).collect(),
            weight: FREE_WEIGHT,
            kind: BlockKind::Stay,
            constraint: None,
            bound: false,
            eval: Box::new(move |xs| {
                let init = init.borrow();
                init.iter().enumerate().map(|(i, v)| (xs[at + i] - v) * unit[i]).collect()
            }),
        });
        if e.points.is_empty() || e.kind() == ShapeKind::Point {
            continue;
        }
        let points = thin(&e.points);
        let scatter = e.scatter.max(1e-3);
        soft_count += points.len();
        let o = at;
        if let Shape::Spline(sp) = &e.shape {
            // A fitted spline: each point held against its foot on the curve, both ways; the
            // feet are found again in the refit rounds after the first descend.
            let data = Rc::new(crate::spline_solve::SplineSoftData::new(sp, at, &points));
            refits.push(data.clone());
            blocks.push(Block { params: data.params().collect(), weight: 1.0 / scatter, kind: BlockKind::Soft, constraint: None, bound: false, eval: Box::new(move |xs| data.eval(xs)) });
            continue;
        }
        let eval: Eval = if e.kind() == ShapeKind::Line {
            Box::new(move |xs| {
                let (ax, ay) = (xs[o], xs[o + 1]);
                let (mut dx, mut dy) = (xs[o + 2] - ax, xs[o + 3] - ay);
                let l = dx.hypot(dy);
                if l > 0.0 {
                    dx /= l;
                    dy /= l;
                } else {
                    dx = 1.0;
                    dy = 0.0;
                }
                // Signed distance off the carrier: the point against the left normal.
                points.iter().map(|p| (p.y - ay) * dx - (p.x - ax) * dy).collect()
            })
        } else {
            Box::new(move |xs| {
                let (cx, cy, r) = (xs[o], xs[o + 1], xs[o + 2]);
                points.iter().map(|p| (p.x - cx).hypot(p.y - cy) - r).collect()
            })
        };
        blocks.push(Block { params: (at..at + count).collect(), weight: 1.0 / scatter, kind: BlockKind::Soft, constraint: None, bound: false, eval });
    }

    // ---- the hand ----
    for pull in options.pull {
        if !slots.map.contains_key(&pull.anchor.entity) {
            continue;
        }
        let (anchor, to) = (pull.anchor, pull.to);
        blocks.push(Block {
            params: slots.range(anchor.entity),
            weight: PULL_WEIGHT,
            kind: BlockKind::Pull,
            constraint: None,
            bound: false,
            eval: Box::new(move |xs| {
                let p = slots.anchor(xs, anchor);
                vec![p.x - to.x, p.y - to.y]
            }),
        });
    }

    // A typical size, so angle equations weigh like length equations.
    let mut scale = 0.0f64;
    for e in &sketch.entities {
        match e.shape {
            Shape::Line { a, b } => scale = scale.max((b - a).length()),
            Shape::Arc { r, .. } | Shape::Circle { r, .. } => scale = scale.max(2.0 * r),
            Shape::Point { .. } => {}
            Shape::Spline(ref sp) => scale = scale.max(crate::spline_solve::spline_size(sp)),
        }
    }
    let l_scale = scale.max(1.0);

    // Where two pieces are held end to end: the ends of `a` and `b` a coincidence joins.
    let joint_of = |a: EntityId, b: EntityId| -> Option<(AnchorAt, AnchorAt)> {
        for k in &sketch.constraints {
            let Rule::Coincident { p, q } = k.rule else { continue };
            if p.at == AnchorAt::Center || q.at == AnchorAt::Center {
                continue;
            }
            if p.entity == a && q.entity == b {
                return Some((p.at, q.at));
            }
            if p.entity == b && q.entity == a {
                return Some((q.at, p.at));
            }
        }
        None
    };
    let datums = options.datums;
    let kind_of = |id: EntityId| slots.map.get(&id).map(|s| s.2);

    for c in &sketch.constraints {
        let cid = c.id;
        match c.rule.clone() {
            Rule::Coincident { p, q } => hard(
                cid,
                &[p.entity, q.entity],
                Box::new(move |xs| {
                    let (a, b) = (slots.anchor(xs, p), slots.anchor(xs, q));
                    vec![a.x - b.x, a.y - b.y]
                }),
                &mut blocks, slots,
            ),
            Rule::Horizontal { entity } => hard(cid, &[entity], Box::new(move |xs| vec![slots.line(xs, entity).map_or(0.0, |(a, b)| b.y - a.y)]), &mut blocks, slots),
            Rule::Vertical { entity } => hard(cid, &[entity], Box::new(move |xs| vec![slots.line(xs, entity).map_or(0.0, |(a, b)| b.x - a.x)]), &mut blocks, slots),
            Rule::Parallel { a, b } => hard(cid, &[a, b], Box::new(move |xs| vec![cross(slots.dir(xs, a), slots.dir(xs, b)) * l_scale]), &mut blocks, slots),
            Rule::Perpendicular { a, b } => hard(cid, &[a, b], Box::new(move |xs| vec![slots.dir(xs, a).dot(slots.dir(xs, b)) * l_scale]), &mut blocks, slots),
            Rule::Angle { a, b, value } => {
                let target = value.to_radians();
                hard(
                    cid,
                    &[a, b],
                    Box::new(move |xs| {
                        let (da, db) = (slots.dir(xs, a), slots.dir(xs, b));
                        let angle = cross(da, db).atan2(da.dot(db));
                        // Directions have a sign the angle should not: fold to 0–π.
                        let folded = angle.rem_euclid(PI);
                        let mut diff = folded - target;
                        if diff > PI / 2.0 {
                            diff -= PI;
                        }
                        if diff < -PI / 2.0 {
                            diff += PI;
                        }
                        vec![diff * l_scale]
                    }),
                    &mut blocks, slots,
                );
            }
            Rule::Tangent { a, b } if kind_of(a) == Some(ShapeKind::Spline) || kind_of(b) == Some(ShapeKind::Spline) => {
                // A spline is held tangent where one of its ends meets the other piece, and
                // nowhere else: with no joint the relation holds nothing.
                let (spline, other) = if kind_of(a) == Some(ShapeKind::Spline) { (a, b) } else { (b, a) };
                let Some(joint) = crate::sketch_spline::spline_joint(&sketch.constraints, spline, other) else { continue };
                hard(
                    cid,
                    &[a, b],
                    Box::new(move |xs| {
                        let Some(Shape::Spline(sp)) = slots.read(spline, xs) else { return vec![0.0] };
                        let o = slots.read(other, xs);
                        let piece = match &o {
                            Some(Shape::Line { a, b }) => crate::spline_solve::JointPiece::Line { a: *a, b: *b },
                            Some(Shape::Arc { start, sweep, .. }) => crate::spline_solve::JointPiece::Arc { start: *start, sweep: *sweep },
                            Some(Shape::Spline(other)) => crate::spline_solve::JointPiece::Spline(other),
                            _ => crate::spline_solve::JointPiece::Other,
                        };
                        vec![crate::spline_solve::spline_tangent_residual(&sp, joint, piece, l_scale)]
                    }),
                    &mut blocks, slots,
                );
            }
            Rule::Tangent { a, b } => {
                let kinds = [kind_of(a), kind_of(b)];
                let has_arc = kinds.contains(&Some(ShapeKind::Arc));
                let has_line = kinds.contains(&Some(ShapeKind::Line));
                let both_arcs = kinds[0] == Some(ShapeKind::Arc) && kinds[1] == Some(ShapeKind::Arc);
                // At a joint, a tangency is held as the radius there square to the line (or
                // along the other's radius) — an equation in the angle — rather than as the
                // line's distance from the centre, which a joint sliding off the tangent point
                // changes only by the square of the slide.
                if let Some((ja, jb)) = joint_of(a, b).filter(|_| has_arc && (has_line || both_arcs)) {
                    hard(
                        cid,
                        &[a, b],
                        Box::new(move |xs| {
                            let (Some(ea), Some(eb)) = (slots.read(a, xs), slots.read(b, xs)) else { return vec![0.0] };
                            match (ea, eb) {
                                (Shape::Arc { r: ra, start: sa, sweep: wa, .. }, Shape::Arc { r: rb, start: sb, sweep: wb, .. }) => {
                                    let (ua, ub) = (radial_at(sa, wa, ja), radial_at(sb, wb, jb));
                                    vec![cross(ua, ub) * ra.abs().min(rb.abs())]
                                }
                                (Shape::Arc { r, start, sweep, .. }, Shape::Line { a: la, b: lb }) => {
                                    let d = (lb - la).try_normalize().unwrap_or(DVec2::X);
                                    vec![radial_at(start, sweep, ja).dot(d) * r.abs()]
                                }
                                (Shape::Line { a: la, b: lb }, Shape::Arc { r, start, sweep, .. }) => {
                                    let d = (lb - la).try_normalize().unwrap_or(DVec2::X);
                                    vec![radial_at(start, sweep, jb).dot(d) * r.abs()]
                                }
                                _ => vec![0.0],
                            }
                        }),
                        &mut blocks, slots,
                    );
                    continue;
                }
                hard(
                    cid,
                    &[a, b],
                    Box::new(move |xs| {
                        let (Some(ea), Some(eb)) = (slots.read(a, xs), slots.read(b, xs)) else { return vec![0.0] };
                        let (line, round) = match (ea.clone(), eb.clone()) {
                            (Shape::Line { a, b }, r) | (r, Shape::Line { a, b }) => (Some((a, b)), r),
                            (r, _) => (None, r),
                        };
                        let (Some(rc), Some(rr)) = (round.centre(), round.radius()) else { return vec![0.0] };
                        if let Some((la, lb)) = line {
                            let d = (lb - la).try_normalize().unwrap_or(DVec2::X);
                            let off = (rc - la).dot(d.perp());
                            return vec![off.abs() - rr];
                        }
                        // Two round pieces: externally or internally tangent, whichever the
                        // geometry is nearer to.
                        let (Some(ca), Some(ra), Some(cb), Some(rb)) = (ea.centre(), ea.radius(), eb.centre(), eb.radius()) else { return vec![0.0] };
                        let dist = (ca - cb).length();
                        let external = (dist - (ra + rb)).abs();
                        let internal = (dist - (ra - rb).abs()).abs();
                        vec![if external <= internal { dist - (ra + rb) } else { dist - (ra - rb).abs() }]
                    }),
                    &mut blocks, slots,
                );
            }
            Rule::Equal { a, b } => hard(
                cid,
                &[a, b],
                Box::new(move |xs| {
                    let size = |s: Option<Shape>| match s {
                        Some(Shape::Line { a, b }) => (b - a).length(),
                        Some(Shape::Arc { r, .. }) | Some(Shape::Circle { r, .. }) => r,
                        _ => 0.0,
                    };
                    vec![size(slots.read(a, xs)) - size(slots.read(b, xs))]
                }),
                &mut blocks, slots,
            ),
            Rule::Concentric { a, b } => hard(
                cid,
                &[a, b],
                Box::new(move |xs| {
                    let centre = |s: Option<Shape>| match s {
                        Some(Shape::Line { a, .. }) => a,
                        Some(Shape::Spline(sp)) => sp.pts.first().copied().unwrap_or(DVec2::ZERO),
                        Some(s) => s.centre().unwrap_or(DVec2::ZERO),
                        None => DVec2::ZERO,
                    };
                    let d = centre(slots.read(a, xs)) - centre(slots.read(b, xs));
                    vec![d.x, d.y]
                }),
                &mut blocks, slots,
            ),
            Rule::Symmetric { p, q, about } => {
                let mut ids = vec![p.entity, q.entity];
                if let MirrorAxis::Line(l) = about {
                    ids.push(l);
                }
                // A coordinate plane's line is read off the frame; with no trace given the
                // relation holds nothing.
                let axis = match about {
                    MirrorAxis::Datum(plane) => match trace_of(datums, plane) {
                        Some(t) => Some(t.line()),
                        None => continue,
                    },
                    _ => None,
                };
                hard(
                    cid,
                    &ids,
                    Box::new(move |xs| {
                        let (pp, qq) = (slots.anchor(xs, p), slots.anchor(xs, q));
                        let (a, d) = match about {
                            MirrorAxis::X => return vec![pp.y + qq.y, pp.x - qq.x],
                            MirrorAxis::Y => return vec![pp.x + qq.x, pp.y - qq.y],
                            MirrorAxis::Line(l) => match slots.line(xs, l) {
                                Some((a, b)) => (a, (b - a).try_normalize().unwrap_or(DVec2::X)),
                                None => return vec![0.0, 0.0],
                            },
                            MirrorAxis::Datum(_) => axis.unwrap_or((DVec2::ZERO, DVec2::X)),
                        };
                        let mid = (pp + qq) / 2.0;
                        // The midpoint on the line, the join square to it.
                        vec![(mid - a).dot(d.perp()), (qq - pp).dot(d)]
                    }),
                    &mut blocks, slots,
                );
            }
            Rule::Length { entity, value } => hard(cid, &[entity], Box::new(move |xs| vec![slots.line(xs, entity).map_or(0.0, |(a, b)| (b - a).length() - value)]), &mut blocks, slots),
            Rule::Radius { entity, value } | Rule::Diameter { entity, value } => {
                let want = if matches!(c.rule, Rule::Radius { .. }) { value } else { value / 2.0 };
                hard(cid, &[entity], Box::new(move |xs| vec![slots.read(entity, xs).and_then(|s| s.radius()).map_or(0.0, |r| r - want)]), &mut blocks, slots);
            }
            Rule::Distance { p, q, value } => match q {
                DistanceTo::Line(l) => hard(
                    cid,
                    &[p.entity, l],
                    Box::new(move |xs| {
                        let pp = slots.anchor(xs, p);
                        let Some((a, b)) = slots.line(xs, l) else { return vec![0.0] };
                        let d = (b - a).try_normalize().unwrap_or(DVec2::X);
                        vec![(pp - a).dot(d.perp()).abs() - value]
                    }),
                    &mut blocks, slots,
                ),
                DistanceTo::Datum(plane) => {
                    if let Some(t) = trace_of(datums, plane) {
                        hard(cid, &[p.entity], Box::new(move |xs| vec![t.offset(slots.anchor(xs, p)).abs() - value]), &mut blocks, slots);
                    }
                }
                DistanceTo::Anchor(q) => hard(cid, &[p.entity, q.entity], Box::new(move |xs| vec![(slots.anchor(xs, p) - slots.anchor(xs, q)).length() - value]), &mut blocks, slots),
            },
            Rule::PointOn { p, on, bounded } => {
                match on {
                    OnTarget::Line(l) => hard(
                        cid,
                        &[p.entity, l],
                        Box::new(move |xs| {
                            let Some((a, b)) = slots.line(xs, l) else { return vec![0.0] };
                            let d = (b - a).try_normalize().unwrap_or(DVec2::X);
                            vec![(slots.anchor(xs, p) - a).dot(d.perp())]
                        }),
                        &mut blocks, slots,
                    ),
                    OnTarget::Curve(cv) => hard(
                        cid,
                        &[p.entity, cv],
                        Box::new(move |xs| {
                            let pp = slots.anchor(xs, p);
                            match slots.read(cv, xs) {
                                Some(Shape::Arc { c, r, .. }) | Some(Shape::Circle { c, r }) => vec![(pp - c).length() - r],
                                Some(Shape::Spline(sp)) => vec![crate::spline_solve::point_on_spline_residual(&sp, pp)],
                                _ => vec![0.0],
                            }
                        }),
                        &mut blocks, slots,
                    ),
                    OnTarget::Datum(plane) => {
                        if let Some(t) = trace_of(datums, plane) {
                            hard(cid, &[p.entity], Box::new(move |xs| vec![t.offset(slots.anchor(xs, p))]), &mut blocks, slots);
                        }
                    }
                }
                if bounded {
                    let target = match on {
                        OnTarget::Line(l) | OnTarget::Curve(l) => Some(l),
                        OnTarget::Datum(_) => None,
                    };
                    if let Some(id) = target {
                        blocks.push(Block {
                            params: slots.params_for(&[p.entity, id]),
                            weight: HARD_WEIGHT,
                            kind: BlockKind::Hard,
                            constraint: Some(cid),
                            bound: true,
                            eval: Box::new(move |xs| {
                                let pp = slots.anchor(xs, p);
                                match slots.read(id, xs) {
                                    Some(Shape::Line { a, b }) => {
                                        let d = b - a;
                                        let l = d.length();
                                        if !(l > 0.0) {
                                            return vec![0.0];
                                        }
                                        let s = (pp - a).dot(d) / l;
                                        vec![if s < 0.0 {
                                            s
                                        } else if s > l {
                                            s - l
                                        } else {
                                            0.0
                                        }]
                                    }
                                    Some(Shape::Arc { c, r, start, sweep }) => {
                                        let start = if sweep < 0.0 { start + sweep } else { start };
                                        let sweep = sweep.abs();
                                        let along = ((pp.y - c.y).atan2(pp.x - c.x) - start).rem_euclid(TAU);
                                        if sweep >= TAU || along <= sweep {
                                            return vec![0.0];
                                        }
                                        let r = r.abs();
                                        vec![if along - sweep <= TAU - along { (along - sweep) * r } else { -(TAU - along) * r }]
                                    }
                                    _ => vec![0.0],
                                }
                            }),
                        });
                    }
                }
            }
            Rule::Collinear { line, on } => match on {
                OnTarget::Line(m) | OnTarget::Curve(m) => hard(
                    cid,
                    &[line, m],
                    Box::new(move |xs| {
                        let (Some((la, lb)), Some((ma, mb))) = (slots.line(xs, line), slots.line(xs, m)) else { return vec![0.0, 0.0] };
                        let nrm = (mb - ma).try_normalize().unwrap_or(DVec2::X).perp();
                        vec![(la - ma).dot(nrm), (lb - ma).dot(nrm)]
                    }),
                    &mut blocks, slots,
                ),
                OnTarget::Datum(plane) => {
                    if let Some(t) = trace_of(datums, plane) {
                        hard(cid, &[line], Box::new(move |xs| slots.line(xs, line).map_or(vec![0.0, 0.0], |(a, b)| vec![t.offset(a), t.offset(b)])), &mut blocks, slots);
                    }
                }
            },
        }
    }

    // ---- Levenberg–Marquardt ----
    let mut free_index = vec![usize::MAX; n];
    let mut m = 0;
    for i in 0..n {
        if free[i] {
            free_index[i] = m;
            m += 1;
        }
    }
    let mut iterations = 0;
    let quick = options.quick;
    if m > 0 {
        let all: Vec<&Block> = blocks.iter().collect();
        descend(&all, &mut x, &free, &free_index, m, if quick { QUICK_ITERATIONS } else { MAX_ITERATIONS }, if quick { 1e-6 } else { 1e-9 }, if quick { 1e-8 } else { 1e-12 }, &mut iterations);
        if !quick && !refits.is_empty() {
            // A fitted spline's points find their feet on the curve again where it has got
            // to, and the sketch settles again, until the feet stay where they are.
            for _ in 0..crate::spline_solve::REFIT_ROUNDS {
                let moved = refits.iter().map(|r| r.refresh(&x)).fold(0.0f64, f64::max);
                if moved < crate::spline_solve::REFIT_SETTLED {
                    break;
                }
                descend(&all, &mut x, &free, &free_index, m, MAX_ITERATIONS, 1e-9, 1e-12, &mut iterations);
            }
        }
        if !options.pull.is_empty() {
            // The hand let go: every piece now wishes to stay where the pull left it, and the
            // relations close the last microns the pull held open.
            for (at, init) in &rests {
                let mut v = init.borrow_mut();
                let len = v.len();
                v.copy_from_slice(&x[*at..*at + len]);
            }
            let active: Vec<&Block> = blocks.iter().filter(|b| b.kind != BlockKind::Pull).collect();
            descend(&active, &mut x, &free, &free_index, m, RELEASE_ITERATIONS, 1e-9, 1e-12, &mut iterations);
        }
        if !quick && soft_count > 0 && worst_hard(&blocks, &x) > CLOSE_FROM {
            // Slice points a few microns apart weigh as much as the relations and hold them a
            // micron or two open: the pieces settled where the data put them now stay there
            // and let the relations close.
            for (at, init) in &rests {
                let mut v = init.borrow_mut();
                let len = v.len();
                v.copy_from_slice(&x[*at..*at + len]);
            }
            let active: Vec<&Block> = blocks.iter().filter(|b| b.kind == BlockKind::Hard || b.kind == BlockKind::Stay).collect();
            descend(&active, &mut x, &free, &free_index, m, RELEASE_ITERATIONS, 1e-9, 1e-12, &mut iterations);
        }
    }
    let report = finish(&blocks, &x, iterations, soft_count, quick, sketch, slots);
    // Write the parameters back.
    for e in &mut sketch.entities {
        let Some(mut s) = slots.read(e.id, &x) else { continue };
        match &mut s {
            Shape::Circle { r, .. } => *r = r.abs(),
            Shape::Arc { r, start, sweep, .. } => {
                *r = r.abs();
                if *sweep < 0.0 {
                    *start += *sweep;
                    *sweep = -*sweep;
                }
                // An arc's end is periodic in its sweep, so the solver may have walked the
                // sweep round by whole turns; the arc is the one within a turn. A sweep that
                // lands on a whole turn is a whole turn, not nothing.
                let wrapped = sweep.rem_euclid(TAU);
                *sweep = if wrapped < 1e-9 && *sweep > 1e-9 { TAU } else { wrapped };
                *start = start.rem_euclid(TAU);
            }
            _ => {}
        }
        e.shape = s;
    }
    report
}

/// A hard equation's block.
fn hard<'a>(id: ConstraintId, ids: &[EntityId], eval: Eval<'a>, blocks: &mut Vec<Block<'a>>, slots: &Slots) {
    blocks.push(Block { params: slots.params_for(ids), weight: HARD_WEIGHT, kind: BlockKind::Hard, constraint: Some(id), bound: false, eval });
}

fn cost(rs: &[Vec<f64>]) -> f64 {
    rs.iter().flat_map(|r| r.iter()).map(|v| v * v).sum()
}

/// Descend over `active` from where x stands, for at most `max_iterations` or until a step
/// or its gain is under the tolerances.
#[allow(clippy::too_many_arguments)]
fn descend(active: &[&Block], x: &mut [f64], free: &[bool], free_index: &[usize], m: usize, max_iterations: usize, step_tolerance: f64, gain_tolerance: f64, iterations: &mut usize) {
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
        *iterations += 1;
        jtj.iter_mut().for_each(|v| *v = 0.0);
        jtr.iter_mut().for_each(|v| *v = 0.0);
        xt.copy_from_slice(x);
        for (bi, b) in active.iter().enumerate() {
            let r0 = &rs[bi];
            let cols: Vec<usize> = b.params.iter().copied().filter(|&p| free[p]).collect();
            if cols.is_empty() {
                continue;
            }
            // Central differences, one parameter perturbed at a time.
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
        for _attempt in 0..12 {
            // Damped normal equations, scaled to unit diagonal: the hard rows put entries a
            // million times the stay-put terms' on the diagonal.
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

/// The largest residual of any relation, unweighted, in millimetres.
fn worst_hard(blocks: &[Block], x: &[f64]) -> f64 {
    blocks.iter().filter(|b| b.constraint.is_some()).flat_map(|b| (b.eval)(x)).fold(0.0, |w, v| w.max(v.abs()))
}

fn finish(blocks: &[Block], x: &[f64], iterations: usize, soft_count: usize, quick: bool, sketch: &Sketch, slots: &Slots) -> SolveReport {
    let mut unmet: Vec<(ConstraintId, f64)> = Vec::new();
    let mut soft_sq = 0.0;
    let mut hard_rows: Vec<Vec<f64>> = Vec::new();
    let mut hard_cols: Vec<Vec<usize>> = Vec::new();
    for b in blocks {
        let r = (b.eval)(x);
        let Some(id) = b.constraint else {
            if b.kind == BlockKind::Soft {
                soft_sq += r.iter().map(|v| v * v).sum::<f64>();
            }
            continue;
        };
        let worst = r.iter().fold(0.0f64, |w, v| w.max(v.abs()));
        if worst > UNMET_TOLERANCE {
            match unmet.iter_mut().find(|u| u.0 == id) {
                Some(had) => had.1 = had.1.max(worst),
                None => unmet.push((id, worst)),
            }
        }
        if quick || b.bound {
            continue;
        }
        // Rows of the constraint Jacobian, unweighted, for the rank.
        let mut xt = x.to_vec();
        let cols = &b.params;
        let mut rows = vec![vec![0.0; cols.len()]; r.len()];
        for (ci, &p) in cols.iter().enumerate() {
            let h = 1e-6 * x[p].abs().max(1.0);
            xt.copy_from_slice(x);
            xt[p] += h;
            let rp = (b.eval)(&xt);
            xt[p] -= 2.0 * h;
            let rm = (b.eval)(&xt);
            for k in 0..r.len() {
                rows[k][ci] = (rp[k] - rm[k]) / (2.0 * h);
            }
        }
        for row in rows {
            hard_rows.push(row);
            hard_cols.push(cols.clone());
        }
    }
    let n = x.len();
    let redundant = if quick { 0 } else { redundant_rows(&hard_rows, &hard_cols, n, sketch, slots) };
    SolveReport {
        ok: unmet.is_empty(),
        iterations,
        unmet,
        redundant,
        equations: hard_rows.len(),
        soft_rms: if soft_count > 0 { (soft_sq / soft_count as f64).sqrt() } else { 0.0 },
        under_constrained: if quick { Vec::new() } else { loose_entities(&hard_rows, &hard_cols, n, sketch, slots) },
    }
}

/// The pieces a relation or dimension does not pin down: in reduced row echelon form of the
/// constraint Jacobian, a parameter is loose when its column has no pivot, or its pivot's row
/// reaches a column that has none. Pieces held by themselves (fixed, projected, fitted) are
/// left out.
fn loose_entities(rows: &[Vec<f64>], cols: &[Vec<usize>], n: usize, sketch: &Sketch, slots: &Slots) -> Vec<EntityId> {
    let mut held = vec![false; n];
    let mut owner = vec![0; n];
    for e in &sketch.entities {
        let (at, count, _) = slots.map[&e.id];
        for i in at..at + count {
            owner[i] = e.id;
            held[i] = e.fixed || e.projected || e.is_fitted();
        }
    }
    let mut column = vec![usize::MAX; n];
    let mut param = Vec::new();
    for p in 0..n {
        if !held[p] {
            column[p] = param.len();
            param.push(p);
        }
    }
    let m = param.len();
    if m == 0 {
        return Vec::new();
    }
    let mut dense: Vec<Vec<f64>> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let mut row = vec![0.0; m];
        let mut norm = 0.0;
        for (ci, &p) in cols[i].iter().enumerate() {
            if column[p] != usize::MAX {
                row[column[p]] = r[ci];
                norm += r[ci] * r[ci];
            }
        }
        let norm = f64::sqrt(norm);
        if norm > 1e-9 {
            row.iter_mut().for_each(|v| *v /= norm);
            dense.push(row);
        }
    }
    const EPS: f64 = 1e-6;
    let mut pivot_of = vec![usize::MAX; m];
    let mut rank = 0;
    for j in 0..m {
        if rank >= dense.len() {
            break;
        }
        let mut best = rank;
        for i in rank + 1..dense.len() {
            if dense[i][j].abs() > dense[best][j].abs() {
                best = i;
            }
        }
        if dense[best][j].abs() < EPS {
            continue;
        }
        dense.swap(rank, best);
        let scale = 1.0 / dense[rank][j];
        dense[rank].iter_mut().for_each(|v| *v *= scale);
        let pivot = dense[rank].clone();
        for (i, row) in dense.iter_mut().enumerate() {
            if i == rank {
                continue;
            }
            let f = row[j];
            if f.abs() < EPS {
                continue;
            }
            for k in 0..m {
                row[k] -= f * pivot[k];
            }
        }
        pivot_of[j] = rank;
        rank += 1;
    }
    let mut loose: Vec<EntityId> = Vec::new();
    for j in 0..m {
        let id = owner[param[j]];
        let is_loose = if pivot_of[j] == usize::MAX {
            true
        } else {
            let row = &dense[pivot_of[j]];
            (0..m).any(|k| pivot_of[k] == usize::MAX && row[k].abs() > EPS)
        };
        if is_loose && !loose.contains(&id) {
            loose.push(id);
        }
    }
    loose
}

/// How many of the constraint equations are linearly dependent on the others at the
/// solution: the Gram matrix of the rows has one zero eigenvalue per dependency.
fn redundant_rows(rows: &[Vec<f64>], cols: &[Vec<usize>], n: usize, sketch: &Sketch, slots: &Slots) -> usize {
    let k = rows.len();
    if k == 0 {
        return 0;
    }
    let mut fixed = vec![false; n];
    for e in &sketch.entities {
        if e.fixed || e.projected {
            let (at, count, _) = slots.map[&e.id];
            fixed[at..at + count].iter_mut().for_each(|f| *f = true);
        }
    }
    // Sparse rows brought to unit length, then the Gram matrix.
    let sparse: Vec<Vec<(usize, f64)>> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut entries: Vec<(usize, f64)> = Vec::new();
            for (ci, &p) in cols[i].iter().enumerate() {
                if !fixed[p] {
                    match entries.iter_mut().find(|e| e.0 == p) {
                        Some(e) => e.1 = r[ci],
                        None => entries.push((p, r[ci])),
                    }
                }
            }
            let norm = entries.iter().map(|e| e.1 * e.1).sum::<f64>().sqrt();
            if norm > 0.0 {
                entries.iter_mut().for_each(|e| e.1 /= norm);
            }
            entries.sort_by_key(|e| e.0);
            entries
        })
        .collect();
    let dot = |a: &[(usize, f64)], b: &[(usize, f64)]| -> f64 {
        let (mut i, mut j, mut s) = (0, 0, 0.0);
        while i < a.len() && j < b.len() {
            match a[i].0.cmp(&b[j].0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    s += a[i].1 * b[j].1;
                    i += 1;
                    j += 1;
                }
            }
        }
        s
    };
    let mut gram = vec![0.0; k * k];
    let mut largest = 0.0f64;
    for i in 0..k {
        for j in i..k {
            let s = dot(&sparse[i], &sparse[j]);
            gram[i * k + j] = s;
            gram[j * k + i] = s;
            if i == j {
                largest = largest.max(s);
            }
        }
    }
    if largest == 0.0 {
        return k;
    }
    let (values, _) = symmetric_eigen_n(k, &gram);
    // The rows are finite differences good to some eight digits; a dependency is an
    // eigenvalue that is zero but for that noise.
    values.iter().filter(|v| **v < 1e-6 * largest).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::{Entity, Shape};

    fn line(s: &Sketch, id: EntityId) -> (DVec2, DVec2) {
        match s.entity(id).unwrap().shape {
            Shape::Line { a, b } => (a, b),
            _ => panic!("not a line"),
        }
    }

    #[test]
    fn meets_typed_lengths_on_a_hand_drawn_rectangle_and_keeps_it_square() {
        let mut s = Sketch::default();
        let ids = s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(38.0, 21.0));
        s.constrain(Rule::Length { entity: ids[0], value: 40.0 });
        s.constrain(Rule::Length { entity: ids[1], value: 20.0 });
        let rep = solve_sketch(&mut s, SolveOptions::default());
        assert!(rep.ok, "{rep:?}");
        let (a, b) = line(&s, ids[0]);
        assert!(((b - a).length() - 40.0).abs() < 1e-6);
        assert!((b.y - a.y).abs() < 1e-6);
        let (c, d) = line(&s, ids[1]);
        assert!(((d - c).length() - 20.0).abs() < 1e-6);
        assert_eq!(rep.redundant, 0);
        // Position is still free: the rectangle's corner is loose.
        assert!(!rep.under_constrained.is_empty());
    }

    #[test]
    fn reports_a_dimension_that_fights_another() {
        let mut s = Sketch::default();
        let l = s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 0.0) });
        s.constrain(Rule::Length { entity: l, value: 10.0 });
        let second = s.constrain(Rule::Length { entity: l, value: 12.0 });
        let rep = solve_sketch(&mut s, SolveOptions::default());
        assert!(!rep.ok);
        assert!(rep.unmet.iter().any(|u| u.0 == second) || rep.unmet.len() == 2);
    }

    #[test]
    fn a_fitted_line_settles_on_its_points_under_a_horizontal() {
        let mut s = Sketch::default();
        let pts: Vec<DVec2> = (0..50).map(|i| DVec2::new(i as f64 * 0.5, 2.0 + 0.01 * (i % 3) as f64)).collect();
        let id = s.add(Entity::new(Shape::Line { a: DVec2::new(0.0, 1.5), b: DVec2::new(24.5, 2.6) }).fitted(pts, 0.01));
        s.constrain(Rule::Horizontal { entity: id });
        let rep = solve_sketch(&mut s, SolveOptions::default());
        assert!(rep.ok);
        let (a, b) = line(&s, id);
        assert!((a.y - b.y).abs() < 1e-6);
        assert!((a.y - 2.01).abs() < 0.01, "settled at {}", a.y);
        assert!(rep.soft_rms < 0.02);
    }

    #[test]
    fn a_dragged_end_slides_along_its_horizontal() {
        let mut s = Sketch::default();
        let l = s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 0.0) });
        s.infer(Rule::Horizontal { entity: l });
        let pull = [Pull { anchor: Anchor::new(l, AnchorAt::B), to: DVec2::new(15.0, 3.0) }];
        let rep = solve_sketch(&mut s, SolveOptions { pull: &pull, ..Default::default() });
        assert!(rep.ok);
        let (a, b) = line(&s, l);
        assert!((a.y - b.y).abs() < 1e-6);
        assert!((b.x - 15.0).abs() < 0.2, "{b}");
    }

    #[test]
    fn tangent_line_and_arc_at_a_joint() {
        let mut s = Sketch::default();
        let l = s.add_shape(Shape::Line { a: DVec2::new(-10.0, 0.1), b: DVec2::new(0.0, 0.0) });
        let arc = s.add_shape(Shape::Arc { c: DVec2::new(0.2, 5.0), r: 5.0, start: -PI / 2.0, sweep: PI / 2.0 });
        s.constrain(Rule::Coincident { p: Anchor::new(l, AnchorAt::B), q: Anchor::new(arc, AnchorAt::A) });
        s.constrain(Rule::Tangent { a: l, b: arc });
        let rep = solve_sketch(&mut s, SolveOptions::default());
        assert!(rep.ok, "{rep:?}");
        let (a, b) = line(&s, l);
        let Shape::Arc { start, .. } = s.entity(arc).unwrap().shape else { panic!() };
        let radial = DVec2::new(start.cos(), start.sin());
        assert!(radial.dot((b - a).normalize()).abs() < 1e-6);
    }
}
