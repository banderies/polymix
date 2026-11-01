// SPDX-License-Identifier: AGPL-3.0-only
//! Ported from ScanRuler's `plugins/reverse/tests/reStations.test.ts`, `reStations15.test.ts`
//! and `reStationsOpenRun.test.ts`, © Stefan Hermann (CNC Kitchen), AGPL-3.0-only: a chain cut
//! into another station's pieces, a station read as a profile, fitted like another, checked
//! piece by piece, moved in by a wall, and refused on an open run with the numbers it decided
//! on. The moulded wing lofted end to end through the commands and the kernel is the app's to
//! test.

use super::*;
use crate::sketch::{Constraint, Entity};

fn p(x: f64, y: f64) -> DVec2 {
    DVec2::new(x, y)
}

fn deg() -> f64 {
    PI / 180.0
}

/// What a ScanRuler test's regular expression asks of a message, matched here by hand: the
/// pattern is anchored at the start and read literally, but for `{digit}` — `\d` —, `{num}` —
/// `\d+(\.\d+)?` —, `{dec1}` — `\d+\.\d` —, `{any}` — `.*` — and a closing `{end}` — `$`;
/// without it the message may run on.
fn like(said: &str, pattern: &str) -> bool {
    #[derive(Clone, Copy)]
    enum Tok {
        Lit(char),
        Digit,
        Num,
        Dec1,
        Any,
        End,
    }
    let mut toks = Vec::new();
    let mut rest = pattern;
    while let Some(c) = rest.chars().next() {
        let special = [("{digit}", Tok::Digit), ("{num}", Tok::Num), ("{dec1}", Tok::Dec1), ("{any}", Tok::Any), ("{end}", Tok::End)].into_iter().find(|(k, _)| rest.starts_with(k));
        if let Some((k, t)) = special {
            toks.push(t);
            rest = &rest[k.len()..];
        } else {
            toks.push(Tok::Lit(c));
            rest = &rest[c.len_utf8()..];
        }
    }
    fn digits(s: &[char]) -> usize {
        s.iter().take_while(|c| c.is_ascii_digit()).count()
    }
    fn go(toks: &[Tok], s: &[char]) -> bool {
        let Some((&t, more)) = toks.split_first() else { return true };
        match t {
            Tok::End => s.is_empty(),
            Tok::Lit(c) => s.first() == Some(&c) && go(more, &s[1..]),
            Tok::Digit => s.first().is_some_and(|c| c.is_ascii_digit()) && go(more, &s[1..]),
            Tok::Any => (0..=s.len()).any(|k| go(more, &s[k..])),
            Tok::Num => {
                let i = digits(s);
                if i == 0 {
                    return false;
                }
                if (1..=i).any(|k| go(more, &s[k..])) {
                    return true;
                }
                let j = if s.get(i) == Some(&'.') { digits(&s[i + 1..]) } else { 0 };
                (1..=j).any(|k| go(more, &s[i + 1 + k..]))
            }
            Tok::Dec1 => {
                let i = digits(s);
                i > 0 && s.get(i) == Some(&'.') && s.get(i + 1).is_some_and(|c| c.is_ascii_digit()) && go(more, &s[i + 2..])
            }
        }
    }
    go(&toks, &said.chars().collect::<Vec<_>>())
}

#[test]
fn reads_a_pattern_as_the_tests_regular_expressions_do() {
    assert!(like("lands 12.5 mm", "lands {num} mm{end}"));
    assert!(like("lands 12 mm and on", "lands {num} mm"));
    assert!(!like("lands 12. mm", "lands {num} mm"));
    assert!(like("to 200.4°, more", "to 200.{digit}°, more"));
    assert!(!like("to 200.45°, more", "to 200.{digit}°, more"));
    assert!(like("to 200.4°, more", "to {dec1}°, more"));
    assert!(!like("to 200°, more", "to {dec1}°, more"));
    assert!(like("a b c. Point it.", "a {any}Point it.{end}"));
    assert!(!like("a b c. Point it. ", "a {any}Point it.{end}"));
}

fn loop_of(s: &Sketch) -> StationLoop {
    match station_loop(&s.entities, "Station", false) {
        Ok(l) => l,
        Err(e) => panic!("{e}"),
    }
}

fn line_of(s: &Sketch, id: EntityId) -> (DVec2, DVec2) {
    match s.entity(id).map(|e| e.shape.clone()) {
        Some(Shape::Line { a, b }) => (a, b),
        other => panic!("piece {id} is {other:?}"),
    }
}

fn arc_of(s: &Sketch, id: EntityId) -> (DVec2, f64, f64, f64) {
    match s.entity(id).map(|e| e.shape.clone()) {
        Some(Shape::Arc { c, r, start, sweep }) => (c, r, start, sweep),
        other => panic!("piece {id} is {other:?}"),
    }
}

// ---- the stations of round 15 ------------------------------------------------

/// A piece of a channel station: a line (drawn or fitted) or a fitted arc, in degrees.
enum Piece15 {
    Line(EntityId, [f64; 2], [f64; 2], bool),
    Arc(EntityId, [f64; 2], f64, f64, f64),
}

/// A station of six pieces, ids 7–11 fitted and 40 drawn across the top, as a channel station
/// is fitted like its template. A fitted piece's slice points: any will do for a station's
/// reading.
fn station(pieces: &[Piece15]) -> Sketch {
    let fitted = vec![p(0.0, 0.0)];
    let mut s = Sketch { entities: vec![], constraints: vec![], next_id: 41 };
    for piece in pieces {
        let e = match *piece {
            Piece15::Line(id, a, b, drawn) => {
                let e = Entity { id, ..Entity::new(Shape::Line { a: p(a[0], a[1]), b: p(b[0], b[1]) }) };
                if drawn {
                    e
                } else {
                    e.fitted(fitted.clone(), 0.0)
                }
            }
            Piece15::Arc(id, c, r, start, sweep) => Entity { id, ..Entity::new(Shape::Arc { c: p(c[0], c[1]), r, start: start * deg(), sweep: sweep * deg() }) }.fitted(fitted.clone(), 0.0),
        };
        s.entities.push(e);
    }
    s
}

/// A sound channel station: the floor runs on from the first round into the second, every
/// corner tangent.
fn sound() -> Sketch {
    station(&[
        Piece15::Line(7, [29.5081, 2.0], [34.1744, -11.6253], false),
        Piece15::Arc(8, [36.7455, -10.7447], 2.7178, 198.905, 71.2241),
        Piece15::Line(9, [36.7517, -13.4625], [37.7577, -13.4602], false),
        Piece15::Arc(10, [37.7476, -8.9843], 4.4759, 270.129, 88.26),
        Piece15::Line(11, [42.2217, -9.1101], [42.5342, 2.0], false),
        Piece15::Line(40, [29.5081, 2.0], [42.5342, 2.0], true),
    ])
}

/// The same station folded: its floor runs 1.05 mm back between the rounds — two regions, one
/// of 0.019 mm².
fn backwards() -> Sketch {
    station(&[
        Piece15::Line(7, [36.1393, 2.0], [35.5425, -9.9852], false),
        Piece15::Arc(8, [37.3741, -10.0764], 1.8339, 177.1493, 96.5961),
        Piece15::Line(9, [37.4939, -11.9064], [36.4499, -11.9747], false),
        Piece15::Arc(10, [36.2013, -8.1773], 3.8055, 273.7454, 85.8917),
        Piece15::Line(11, [40.0068, -8.2014], [40.0714, 2.0], false),
        Piece15::Line(40, [36.1393, 2.0], [40.0714, 2.0], true),
    ])
}

/// An outlet station whose first round is R1.12 over 3.3°, entered against the wall coming
/// down into it.
fn sliver_round() -> Sketch {
    station(&[
        Piece15::Line(7, [33.0088, 2.0], [33.0088, -10.3777], false),
        Piece15::Arc(8, [31.887, -10.3777], 1.1219, 0.0, 3.2907),
        Piece15::Line(9, [33.007, -10.3133], [33.3512, -16.3008], false),
        Piece15::Arc(10, [42.3415, -15.7839], 9.0051, 183.2907, 174.8001),
        Piece15::Line(11, [51.3415, -16.0839], [51.9443, 2.0], false),
        Piece15::Line(40, [33.0088, 2.0], [51.9443, 2.0], true),
    ])
}

fn pull_floor(s: &mut Sketch) {
    if let Some(e) = s.entity_mut(9) {
        if let Shape::Line { a, b } = &mut e.shape {
            *b = p(a.x + 0.0005, a.y);
        }
    }
}

#[test]
fn names_a_floor_run_backwards_one_pulled_to_nothing_and_a_round_under_5_degrees_and_passes_a_sound_one() {
    let template = sound();
    let lp = loop_of(&template);
    assert!(station_faults(&template, &lp, &sound().entities, None).is_empty());
    assert_eq!(station_faults(&template, &lp, &backwards().entities, None), ["piece 9 (a line) runs 1.046 mm backwards against the template’s piece 9"]);
    // The floor pulled to half a micron by the solve.
    let mut pulled = sound();
    pull_floor(&mut pulled);
    assert_eq!(station_faults(&template, &lp, &pulled.entities, None), ["piece 9 (a line) is 0.0005 mm long — under 0.05 mm"]);
    assert_eq!(station_faults(&template, &lp, &sliver_round().entities, None), ["piece 8 (an arc) sweeps 3.3° — under 5°"]);
}

#[test]
fn names_a_piece_shorter_than_touch_instead_of_reading_three_pieces_meeting() {
    let mut s = sound();
    pull_floor(&mut s);
    let said = station_loop(&s.entities, "Passage start station 2", false).expect_err("a sliver");
    assert!(said.0.starts_with("Passage start station 2's piece 9 (a line) is 0.0005 mm long — shorter than the 0.002 mm two ends meet within"), "{said}");
    assert!(!said.0.contains("branches"));
}

/// A template: a wall down, a floor down and out, a wall up — fitted — closed by a line across
/// the top.
fn template15() -> Sketch {
    let mut s = Sketch::default();
    let fitted = vec![p(0.0, 0.0)];
    s.add(Entity::new(Shape::Line { a: p(0.0, 10.0), b: p(0.0, 0.0) }).fitted(fitted.clone(), 0.0));
    s.add(Entity::new(Shape::Line { a: p(0.0, 0.0), b: p(10.0, -10.0) }).fitted(fitted.clone(), 0.0));
    s.add(Entity::new(Shape::Line { a: p(10.0, -10.0), b: p(20.0, 10.0) }).fitted(fitted, 0.0));
    s.add(Entity::new(Shape::Line { a: p(20.0, 10.0), b: p(0.0, 10.0) }));
    s
}

/// Points along a stretch, `n` of them, the last left off.
fn along(a: DVec2, b: DVec2, n: usize) -> Vec<DVec2> {
    (0..n).map(|i| a + (b - a) * (i as f64 / n as f64)).collect()
}

#[test]
fn refuses_a_corner_met_far_past_the_run_and_fits_the_same_run_with_its_wall_up() {
    let t = template15();
    let lp = loop_of(&t);
    // The last wall lies nearly flat — the run took in a boss beside the station — and meets
    // the line across the top 380 mm out.
    let mut flat = along(p(0.0, 10.0), p(0.0, 0.0), 20);
    flat.extend(along(p(0.0, 0.0), p(10.0, -10.0), 20));
    flat.extend(along(p(10.0, -10.0), p(30.0, -9.0), 40));
    flat.push(p(30.0, -9.0));
    let said = like_station(&t, &lp, &flat, None).expect_err("a corner far off");
    assert!(like(&said.0, "the corner of pieces 3 and 4 lands {num} mm past the run’s ends — over 2× the run’s extent ({num} mm){any}Point it with within or facing.{end}"), "{said}");
    let mut up = along(p(0.0, 10.0), p(0.0, 0.0), 20);
    up.extend(along(p(0.0, 0.0), p(10.0, -10.0), 20));
    up.extend(along(p(10.0, -10.0), p(19.0, 8.0), 40));
    up.push(p(19.0, 8.0));
    let made = like_station(&t, &lp, &up, None).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(made.sketch.entities.iter().map(|e| e.id).collect::<Vec<_>>(), [1, 2, 3, 4]);
}

#[test]
fn refuses_the_folded_stations_inset_naming_the_round_and_the_fold_and_an_outlet_round_swollen_from_3_degrees() {
    let folded = backwards();
    let lp = loop_of(&folded);
    let keep: HashSet<EntityId> = inset_keeps(&lp, None).into_iter().collect();
    let said = inset_sketch(&folded, &lp, -3.0, &keep, &[]).expect_err("folded");
    assert!(
        like(&said.0, "the arc 8 would turn inside out at -3 mm — its sweep goes from 96.6° to 200.{digit}°, more than the corners it meets its neighbours at let it (15°): they cross past it. The source folds back at the corner of pieces 8 and 9 (it turns 180° there) — fit that station again"),
        "{said}"
    );
    let outlet = sliver_round();
    let lp = loop_of(&outlet);
    let keep: HashSet<EntityId> = inset_keeps(&lp, None).into_iter().collect();
    let said = inset_sketch(&outlet, &lp, -3.0, &keep, &[]).expect_err("swollen");
    assert!(like(&said.0, "the arc 8 would turn inside out at -3 mm — its sweep goes from 3.3° to {dec1}°{any}folds back at the corner of pieces 7 and 8"), "{said}");
}

#[test]
fn moves_a_sound_station_out_by_the_wall_its_rounds_sweeping_as_they_did() {
    let s = sound();
    let lp = loop_of(&s);
    let keep: HashSet<EntityId> = inset_keeps(&lp, None).into_iter().collect();
    let out = inset_sketch(&s, &lp, -3.0, &keep, &[]).unwrap_or_else(|e| panic!("{e}"));
    // Within the tangencies' rounding in the numbers above.
    let sweeps = |k: &Sketch| k.entities.iter().filter_map(|e| if let Shape::Arc { sweep, .. } = e.shape { Some(sweep / deg()) } else { None }).collect::<Vec<_>>();
    assert_eq!(sweeps(&out.sketch).len(), sweeps(&s).len());
    for (now, was) in sweeps(&out.sketch).iter().zip(sweeps(&s)) {
        assert!((now - was).abs() < 0.5, "{now} against {was}");
    }
    assert!(out.dropped.is_empty());
}

// ---- the wing's stations -------------------------------------------------------

/// The wing's section at z: an arch of radius R about (0, rise) between walls flaring out at
/// `flare` down to the base, `t` thick.
struct Wing {
    r: f64,
    rise: f64,
    flare: f64,
    base: f64,
}

fn wing_at(z: f64) -> Wing {
    Wing { r: 24.0 + 0.15 * z, rise: 0.05 * z, flare: (8.0 + 0.1 * z) * deg(), base: -15.0 }
}

/// Where a wall tangent to the arch at angle `theta` (radius r) reaches the base: down along
/// the tangent, away from the crown.
fn wall_foot(r: f64, rise: f64, theta: f64, base: f64, side: f64) -> (DVec2, DVec2) {
    let top = p(r * theta.cos(), rise + r * theta.sin());
    // Down the wall: clockwise of the arch on the right, counter-clockwise on the left.
    let d = if side > 0.0 { p(theta.sin(), -theta.cos()) } else { p(-theta.sin(), theta.cos()) };
    let k = (base - top.y) / d.y;
    (top, p(top.x + d.x * k, base))
}

/// The tests' own generator: a 32-bit LCG giving −1 to 1, as ScanRuler's test writes it.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = (s * 1_664_525 + 1_013_904_223) % (1u64 << 32);
        (s as f64 / 4_294_967_296.0) * 2.0 - 1.0
    }
}

/// The outer skin at z as the stations see it: the right wall up from v = from, the arch, the
/// left wall down — a noisy run of points.
fn outer_run(z: f64, from: f64, noise: f64) -> Vec<DVec2> {
    let w = wing_at(z);
    let (rtop, rfoot) = wall_foot(w.r, w.rise, w.flare, w.base, 1.0);
    let (ltop, lfoot) = wall_foot(w.r, w.rise, PI - w.flare, w.base, -1.0);
    let k = |y: f64, a: DVec2, b: DVec2| {
        let f = (y - a.y) / (b.y - a.y);
        p(a.x + (b.x - a.x) * f, y)
    };
    let mut pts = along(k(from, rfoot, rtop), rtop, 60);
    for i in 0..200 {
        let a = w.flare + ((PI - 2.0 * w.flare) * i as f64) / 200.0;
        pts.push(p(w.r * a.cos(), w.rise + w.r * a.sin()));
    }
    pts.extend(along(ltop, k(from, lfoot, ltop), 60));
    pts.push(k(from, lfoot, ltop));
    let mut rand = lcg(11);
    pts.into_iter().map(|q| p(q.x + rand() * noise, q.y + rand() * noise)).collect()
}

/// A station drawn as the template is: wall, arch, wall fitted (they carry slice points),
/// joined tangent, closed by a line across the base.
fn template_at(z: f64) -> Sketch {
    let w = wing_at(z);
    let (rtop, rfoot) = wall_foot(w.r, w.rise, w.flare, -15.0, 1.0);
    let (ltop, lfoot) = wall_foot(w.r, w.rise, PI - w.flare, -15.0, -1.0);
    let at = |a: DVec2, b: DVec2, y: f64| p(a.x + ((b.x - a.x) * (y - a.y)) / (b.y - a.y), y);
    let r0 = at(rfoot, rtop, -13.0);
    let l0 = at(lfoot, ltop, -13.0);
    let run = outer_run(z, -13.0, 0.02);
    let mut s = Sketch::default();
    let wall_r = s.add(Entity::new(Shape::Line { a: r0, b: rtop }).fitted(run[..60].to_vec(), 0.02));
    let arch = s.add(Entity::new(Shape::Arc { c: p(0.0, w.rise), r: w.r, start: w.flare, sweep: PI - 2.0 * w.flare }).fitted(run[60..260].to_vec(), 0.02));
    let wall_l = s.add(Entity::new(Shape::Line { a: ltop, b: l0 }).fitted(run[260..].to_vec(), 0.02));
    let base = s.add_shape(Shape::Line { a: l0, b: r0 });
    let co = |s: &mut Sketch, a: EntityId, x: AnchorAt, b: EntityId, y: AnchorAt| s.constrain(Rule::Coincident { p: Anchor::new(a, x), q: Anchor::new(b, y) });
    co(&mut s, wall_r, AnchorAt::B, arch, AnchorAt::A);
    co(&mut s, arch, AnchorAt::B, wall_l, AnchorAt::A);
    co(&mut s, wall_l, AnchorAt::B, base, AnchorAt::A);
    co(&mut s, base, AnchorAt::B, wall_r, AnchorAt::A);
    s.constrain(Rule::Tangent { a: wall_r, b: arch });
    s.constrain(Rule::Tangent { a: arch, b: wall_l });
    s.constrain(Rule::Horizontal { entity: base });
    s.constrain(Rule::Radius { entity: arch, value: w.r });
    s
}

/// A line's distance from a parallel one.
fn line_gap(a: (DVec2, DVec2), b: (DVec2, DVec2)) -> f64 {
    let d = b.1 - b.0;
    ((a.0 - b.0).perp_dot(d) / d.length()).abs()
}

#[test]
fn breaks_a_wall_arch_wall_run_where_its_pieces_meet_whatever_the_cut_would_do() {
    let r = wing_at(20.0).r;
    let (cut, _) = kinds_of_chain(&outer_run(20.0, -13.0, 0.02), &[PieceKind::Line, PieceKind::Arc, PieceKind::Line], None, false).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(cut.segments.iter().map(|s| s.piece.kind()).collect::<Vec<_>>(), [PieceKind::Line, PieceKind::Arc, PieceKind::Line]);
    let crate::segment::Piece::Arc(arc) = cut.segments[1].piece else { panic!("not an arc") };
    assert!((arc.radius - r).abs() < 0.5, "{}", arc.radius);
    // The breaks within a couple of points of the tangent points (60, 260).
    assert!((cut.segments[0].hi - 60).abs() < 6, "{}", cut.segments[0].hi);
    assert!((cut.segments[2].lo - 260).abs() < 6, "{}", cut.segments[2].lo);
    let few = kinds_of_chain(&outer_run(20.0, -13.0, 0.02)[..8], &[PieceKind::Line, PieceKind::Arc, PieceKind::Arc, PieceKind::Line], None, false);
    assert!(few.is_err_and(|e| e.0.contains("too few")));
}

#[test]
fn starts_a_closed_profile_where_its_fitted_run_starts_and_refuses_a_branch_or_a_circle() {
    let lp = loop_of(&template_at(0.0));
    assert!(lp.closed);
    assert_eq!(lp.pieces.iter().map(|q| (q.kind, q.fitted)).collect::<Vec<_>>(), [(ShapeKind::Line, true), (ShapeKind::Arc, true), (ShapeKind::Line, true), (ShapeKind::Line, false)]);
    let mut branch = template_at(0.0);
    let (_, b0) = line_of(&branch, 1);
    branch.add_shape(Shape::Line { a: b0, b: p(0.0, 0.0) });
    assert!(station_loop(&branch.entities, "Wing 1", false).is_err_and(|e| e.0.contains("branches")));
    // ScanRuler's test draws a spline here; this sketch has none, and a circle is refused the
    // same way.
    let mut round = Sketch::default();
    round.add_shape(Shape::Circle { c: p(0.0, 0.0), r: 3.0 });
    assert!(station_loop(&round.entities, "Wing 1", false).is_err_and(|e| e.0.contains("lines and arcs")));
}

#[test]
fn gives_the_run_the_templates_pieces_in_its_order_under_its_ids_the_base_line_copied_and_met() {
    let template = template_at(0.0);
    let lp = loop_of(&template);
    let forward = outer_run(30.0, -13.0, 0.02);
    let mut reversed = forward.clone();
    reversed.reverse();
    for run in [forward, reversed] {
        let made = like_station(&template, &lp, &run, None).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(made.sketch.entities.iter().map(|e| (e.id, e.kind())).collect::<Vec<_>>(), template.entities.iter().map(|e| (e.id, e.kind())).collect::<Vec<_>>());
        let (_, r, _, _) = arc_of(&made.sketch, 2);
        assert!((r - wing_at(30.0).r).abs() < 0.5, "{r}");
        // The base line lies where the template's does, its ends on the walls.
        let (ba, bb) = line_of(&made.sketch, 4);
        assert!((ba.y + 13.0).abs() < 5e-7 && (bb.y + 13.0).abs() < 5e-7, "{ba:?} {bb:?}");
        let (wa, wb) = line_of(&made.sketch, 1);
        assert!([wa, wb].iter().any(|q| (q.y + 13.0).abs() < 1e-6));
        // The relations copied: four joins, two tangencies, the level base; the radius is the
        // scan's, not the template's.
        let mut kinds: Vec<&str> = made.sketch.constraints.iter().map(|c| rule_kind(&c.rule)).collect();
        kinds.sort();
        assert_eq!(kinds, ["coincident", "coincident", "coincident", "coincident", "horizontal", "tangent", "tangent"]);
    }
}

#[test]
fn moves_the_walls_in_along_their_normals_and_the_arch_down_its_radius_the_base_line_kept() {
    let template = template_at(0.0);
    let lp = loop_of(&template);
    assert_eq!(inset_keeps(&lp, None), [4]);
    let keep: HashSet<EntityId> = inset_keeps(&lp, None).into_iter().collect();
    let out = inset_sketch(&template, &lp, 2.0, &keep, &[]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out.moved, [1, 2, 3]);
    let (_, was_r, _, _) = arc_of(&template, 2);
    let (_, now_r, _, _) = arc_of(&out.sketch, 2);
    assert!((now_r - (was_r - 2.0)).abs() < 5e-10);
    assert!((line_gap(line_of(&out.sketch, 1), line_of(&template, 1)) - 2.0).abs() < 5e-10);
    assert!((line_gap(line_of(&out.sketch, 3), line_of(&template, 3)) - 2.0).abs() < 5e-10);
    assert!(out.sketch.entity(1).is_some_and(|e| e.points.is_empty()));
    // The walls still run tangent into the arch, and reach the base line.
    let wall_r = out.sketch.entity(1).map(|e| e.shape.clone()).expect("wall");
    let arch = out.sketch.entity(2).map(|e| e.shape.clone()).expect("arch");
    assert!((anchor_point(&wall_r, AnchorAt::B) - anchor_point(&arch, AnchorAt::A)).length() < 1e-6);
    let (ba, bb) = line_of(&out.sketch, 4);
    assert!((ba.y + 13.0).abs() < 5e-10);
    let (ta, tb) = line_of(&template, 4);
    assert!((bb.x - ba.x).abs() < (tb.x - ta.x).abs());
    // The radius dimension follows the arc; nothing else had to go.
    let radius = out.sketch.constraints.iter().find_map(|c: &Constraint| if let Rule::Radius { value, .. } = c.rule { Some(value) } else { None }).expect("a radius");
    assert!((radius - (was_r - 2.0)).abs() < 5e-10);
    assert!(out.dropped.is_empty());
}

#[test]
fn refuses_an_open_profile_and_an_arc_tighter_than_the_inset() {
    let mut open = template_at(0.0);
    open.entities.pop();
    let lp = loop_of(&open);
    assert!(inset_sketch(&open, &lp, 2.0, &HashSet::new(), &[]).is_err_and(|e| e.0.contains("open")));
    let template = template_at(0.0);
    let lp = loop_of(&template);
    let said = inset_sketch(&template, &lp, 30.0, &[4].into_iter().collect(), &[]).expect_err("too deep");
    assert!(said.0.contains("close up") || said.0.contains("inside out"), "{said}");
}

#[test]
fn moves_every_closed_chain_into_its_own_inside_and_names_the_open_ones() {
    // A 40 × 20 rectangle with a circle inside it and an open stroke beside them: the
    // rectangle in by 1, the circle (inside it) out of its own inside — smaller — by 1, the
    // stroke left.
    let mut s = Sketch::default();
    s.add_rectangle(p(0.0, 0.0), p(40.0, 20.0));
    let circle = s.add_shape(Shape::Circle { c: p(20.0, 10.0), r: 4.0 });
    s.add_shape(Shape::Line { a: p(50.0, 0.0), b: p(60.0, 5.0) });
    s.add_shape(Shape::Line { a: p(60.0, 5.0), b: p(70.0, 0.0) });
    let out = inset_chains(&s, &InsetChainsAsk::new(1.0, "Plate")).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out.open.len(), 1);
    assert_eq!(out.chains.len(), 2);
    let Some(Shape::Circle { r, .. }) = out.sketch.entity(circle).map(|e| e.shape.clone()) else { panic!("no circle") };
    assert!((r - 3.0).abs() < 1e-12);
    let (a, b) = line_of(&out.sketch, 1);
    assert!((a.y - 1.0).abs() < 1e-9 && (b.y - 1.0).abs() < 1e-9, "{a:?} {b:?}");
    // The relations still hold: the rectangle's corners met, level and upright.
    assert!(out.dropped.is_empty(), "{:?}", out.dropped);
    // Too deep a move closes the circle, and moves nothing.
    let said = inset_chains(&s, &InsetChainsAsk::new(5.0, "Plate")).expect_err("closes up");
    assert!(said.0.starts_with("1 of the 2 chains do not move: "), "{said}");
    assert!(said.0.contains(&format!("pieces {circle}: the circle {circle} (R4) is tighter than 5 mm — it would close up.")), "{said}");
    assert!(said.0.ends_with(" — nothing was moved."), "{said}");
}

// ---- a station on an open run ------------------------------------------------------

/// The square 40 × 40, walked from (0, 0) along the bottom, up the right, back along the top
/// and down the left: the point s mm along it.
fn round_square(s: f64) -> DVec2 {
    let t = ((s % 160.0) + 160.0) % 160.0;
    if t <= 40.0 {
        p(t, 0.0)
    } else if t <= 80.0 {
        p(40.0, t - 40.0)
    } else if t <= 120.0 {
        p(120.0 - t, 40.0)
    } else {
        p(0.0, 160.0 - t)
    }
}

/// The walk from s0 to s1, a point every half millimetre and both ends.
fn walk(s0: f64, s1: f64) -> Vec<DVec2> {
    let mut out = Vec::new();
    let mut s = s0;
    while s < s1 {
        out.push(round_square(s));
        s += 0.5;
    }
    out.push(round_square(s1));
    out
}

/// The square cut with gaps along its straight sides: 3 mm from 10 on the bottom, 8 mm from 50
/// up the right, 19 mm from 90 along the top.
fn broken() -> Vec<Vec<DVec2>> {
    vec![walk(13.0, 50.0), walk(58.0, 90.0), walk(109.0, 170.0)]
}

/// A closed square template of four fitted lines, run counter-clockwise.
fn square_template() -> Sketch {
    let fitted = vec![p(0.0, 0.0)];
    let line = |id: EntityId, a: DVec2, b: DVec2| Entity { id, ..Entity::new(Shape::Line { a, b }) }.fitted(fitted.clone(), 0.0);
    Sketch { entities: vec![line(1, p(0.0, 0.0), p(40.0, 0.0)), line(2, p(40.0, 0.0), p(40.0, 40.0)), line(3, p(40.0, 40.0), p(0.0, 40.0)), line(4, p(0.0, 40.0), p(0.0, 0.0))], constraints: vec![], next_id: 5 }
}

/// A chain after the gaps were bridged, as ScanRuler's `joinChainEnds`
/// (`plugins/reverse/core/chainOutline.ts`, outside this package) gives it: its points and the
/// chains it was made of.
struct Joined {
    chain: Vec<DVec2>,
    from: Vec<usize>,
}

/// ScanRuler's `joinChainEnds`, for these tests: the open chains' ends joined wherever two
/// stand within `gap` of each other, nearest pairs first, each end joined once, a bridge filled
/// with points at the chains' spacing.
fn join_chain_ends(chains: &[Vec<DVec2>], gap: f64) -> Vec<Joined> {
    let n = chains.len();
    let open: Vec<bool> = chains.iter().map(|c| c.len() >= 2 && !closed_chain(c)).collect();
    let end = |e: usize| if e & 1 == 1 { chains[e >> 1][chains[e >> 1].len() - 1] } else { chains[e >> 1][0] };
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        if !open[i] {
            continue;
        }
        if chains[i].len() >= 3 {
            let d = (end(2 * i) - end(2 * i + 1)).length();
            if d <= gap {
                pairs.push((2 * i, 2 * i + 1, d));
            }
        }
        for j in i + 1..n {
            if !open[j] {
                continue;
            }
            for e in [2 * i, 2 * i + 1] {
                for f in [2 * j, 2 * j + 1] {
                    let d = (end(e) - end(f)).length();
                    if d <= gap {
                        pairs.push((e, f, d));
                    }
                }
            }
        }
    }
    if pairs.is_empty() {
        return chains.iter().enumerate().map(|(i, c)| Joined { chain: c.clone(), from: vec![i] }).collect();
    }
    pairs.sort_by(|a, b| a.2.total_cmp(&b.2));
    let mut mate: Vec<Option<usize>> = vec![None; 2 * n];
    for &(e, f, _) in &pairs {
        if mate[e].is_some() || mate[f].is_some() {
            continue;
        }
        mate[e] = Some(f);
        mate[f] = Some(e);
    }
    let mut spacings: Vec<f64> = chains.iter().filter(|c| c.len() >= 2).map(|c| view_chain(c).spacing).collect();
    spacings.sort_by(|a, b| a.total_cmp(b));
    let step = 1e-3f64.max(spacings.get(spacings.len() >> 1).copied().unwrap_or(f64::INFINITY));
    let bridge = |out: &mut Vec<DVec2>, a: DVec2, b: DVec2| {
        let k = ((b - a).length() / step).floor() as usize;
        for t in 1..k {
            out.push(a + (b - a) * (t as f64 / k as f64));
        }
    };
    let mut done = vec![false; n];
    let mut out: Vec<Joined> = Vec::new();
    for i in 0..n {
        if done[i] {
            continue;
        }
        if !open[i] || (mate[2 * i].is_none() && mate[2 * i + 1].is_none()) {
            done[i] = true;
            out.push(Joined { chain: chains[i].clone(), from: vec![i] });
            continue;
        }
        // Back to where the run starts — its free end, or round a loop to this chain again.
        let mut entry = 2 * i;
        for _ in 0..=n {
            let Some(m) = mate[entry] else { break };
            if (m ^ 1) == 2 * i {
                break;
            }
            entry = m ^ 1;
        }
        let start = entry;
        let mut pts: Vec<DVec2> = Vec::new();
        let mut from: Vec<usize> = Vec::new();
        loop {
            let c = &chains[entry >> 1];
            done[entry >> 1] = true;
            from.push(entry >> 1);
            let mut w = c.clone();
            if entry & 1 == 1 {
                w.reverse();
            }
            if let Some(&last) = pts.last() {
                bridge(&mut pts, last, w[0]);
            }
            pts.extend(w);
            let Some(m) = mate[entry ^ 1] else { break };
            if m == start {
                let (last, first) = (pts[pts.len() - 1], pts[0]);
                bridge(&mut pts, last, first);
                pts.push(first);
                break;
            }
            entry = m;
        }
        from.sort_unstable();
        out.push(Joined { chain: pts, from });
    }
    out.sort_by_key(|j| j.from[0]);
    out
}

#[test]
fn finds_the_three_gaps_of_the_square_those_close_gaps_bridged_and_those_it_left_open_widest_first() {
    let g = gaps_round(&broken(), &[0], 0.0);
    assert!(g.bridged.is_empty());
    assert_eq!(g.open.len(), 3);
    for (got, want) in g.open.iter().zip([19.0, 8.0, 3.0]) {
        assert!((got - want).abs() < 5e-7, "{:?}", g.open);
    }
    let joined = join_chain_ends(&broken(), 3.5);
    let run = joined.iter().find(|c| c.from.contains(&0)).expect("the run");
    assert_eq!(run.from, [0, 2]);
    let g = gaps_round(&broken(), &run.from, 3.5);
    assert_eq!(g.bridged.len(), 1);
    assert!((g.bridged[0] - 3.0).abs() < 5e-7);
    assert_eq!(g.open.len(), 2);
    assert!((g.open[0] - 19.0).abs() < 5e-7 && (g.open[1] - 8.0).abs() < 5e-7);
}

#[test]
fn names_the_runs_ends_the_close_gaps_given_the_gap_bridged_and_those_still_open() {
    let tmpl = square_template();
    let lp = station_loop(&tmpl.entities, "Square", false).unwrap_or_else(|e| panic!("{e}"));
    let chains = broken();
    let joined = join_chain_ends(&chains, 3.5);
    let w = joined.iter().fold(&joined[0], |a, b| if b.chain.len() > a.chain.len() { b } else { a });
    let made = like_station(&tmpl, &lp, &w.chain, None).expect_err("an open run");
    assert!(made.0.contains("closed outline fitted whole, and this run is open"), "{made}");
    let facts = RunFacts { run: &w.chain, joined_chain: &w.chain, joined_from: &w.from, chains: &chains, runs: 1, fit: RunFit { close_gaps: Some(3.5), ..Default::default() } };
    let words = run_facts_words(&lp, &facts);
    // The longest run: from 109 round past 0 to 50, the 3 mm gap bridged — its ends, at 50 and
    // 109, stand 8 and 19 mm of gaps and the corner between them apart.
    let ends = (round_square(50.0) - round_square(109.0)).length();
    assert!(words.contains(&format!("The run’s ends stand {} mm apart", num(js_round(ends * 1000.0) / 1000.0))), "{words}");
    assert!(words.contains("closeGaps 3.5 bridged 1 gap (3 mm); still open: 19, 8 mm"), "{words}");
    assert!(words.contains("closeGaps 19.95 bridges them where they are holes in the scan; where they are the part’s own (a gap between keys), close the template with a drawn piece"), "{words}");
    // Without closeGaps, all three are open.
    let plain = run_facts_words(&lp, &RunFacts { run: &chains[2], joined_chain: &chains[2], joined_from: &[2], chains: &chains, runs: 1, fit: RunFit::default() });
    assert!(plain.contains("no closeGaps given, none bridged; still open: 19, 8, 3 mm"), "{plain}");
}

#[test]
fn says_within_split_a_chain_that_had_closed() {
    let tmpl = square_template();
    let lp = station_loop(&tmpl.entities, "Square", false).unwrap_or_else(|e| panic!("{e}"));
    let whole = walk(0.0, 160.0);
    let run = walk(5.0, 150.0);
    let words = run_facts_words(&lp, &RunFacts { run: &run, joined_chain: &whole, joined_from: &[0], chains: std::slice::from_ref(&whole), runs: 2, fit: RunFit { within: true, ..Default::default() } });
    assert!(
        like(&words, " The run’s ends stand {num} mm apart: its chain closed (no closeGaps given, none bridged), and within split it into 2 runs, the longest taken — fit the closed chain without it"),
        "{words}"
    );
}

#[test]
fn names_the_bridges_on_any_other_refusal_of_a_bridged_run_and_adds_nothing_to_one_with_none() {
    let tmpl = square_template();
    let lp = station_loop(&tmpl.entities, "Square", false).unwrap_or_else(|e| panic!("{e}"));
    let open = StationLoop { pieces: lp.pieces.iter().enumerate().map(|(i, q)| StationPiece { fitted: i < 3, ..*q }).collect(), closed: true };
    let chains = broken();
    let joined = join_chain_ends(&chains, 3.5);
    let w = joined.iter().find(|c| c.from.contains(&0)).expect("the run");
    assert_eq!(
        run_facts_words(&open, &RunFacts { run: &w.chain, joined_chain: &w.chain, joined_from: &w.from, chains: &chains, runs: 1, fit: RunFit { close_gaps: Some(3.5), ..Default::default() } }),
        " (closeGaps 3.5 bridged 1 gap (3 mm) — a bridge is a straight run across a hole, and the station’s pieces may not fall on it)"
    );
    assert_eq!(run_facts_words(&open, &RunFacts { run: &chains[1], joined_chain: &chains[1], joined_from: &[1], chains: &chains, runs: 1, fit: RunFit::default() }), "");
}
