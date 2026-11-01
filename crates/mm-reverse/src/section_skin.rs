// SPDX-License-Identifier: AGPL-3.0-only
//! Which side of the material a section's points lie on: one skin of a thin wall told from
//! the other, and the points that face a way. Ported from ScanRuler
//! (`src/core/section/skin.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A cut through a moulded shell crosses each wall twice — once through the skin the part
//! shows, once through the skin inside — and ribs and bosses on the inside add pieces of
//! their own. A form fitted to all of it is pulled off both. Every point of a cut carries the
//! direction out of the material there (from the scan's triangle winding), and with it the cut
//! can be read for one side alone:
//!
//! - outer skin: a point whose outward direction faces away from the middle of the cut (the
//!   mean of its points by length) and whose ray along that direction leaves the cut without
//!   crossing it — the part as seen from outside. A rib's face looks at the next rib or the
//!   wall it stands on, and the inner skin of a closed shell at the wall across, so neither is
//!   outer.
//! - inner skin: a point whose outward direction faces the middle of the cut and whose ray
//!   back through the material first comes out on the outer skin, no farther than about the
//!   wall's thickness (one and a half times the median of those distances over the cut) — the
//!   other side of the same wall. A rib's face looks through the rib onto its other face, not
//!   the outer skin, and a rib's tip through the rib and the wall, much farther, so neither is
//!   a skin; a solid's far side faces away from the middle, so a part that is not hollow has no
//!   inner skin.
//! - facing: a point whose outward direction lies within an angle of a direction given on the
//!   sheet.
//!
//! Each point is judged on its outward direction smoothed over its neighbours, and the verdict
//! is cleaned along the chain: a gap in a kept stretch, or a kept stretch, shorter than a few
//! point spacings takes its neighbours' verdict — the scan's noise, not the part. What is kept
//! comes out as runs, each a chain of its own (`runs_where`), the way a box on the sheet keeps
//! a run of a chain.
//!
//! "Middle" is a judgement, not a measurement: a cut through an open shell (an arch, a cover)
//! has it under the arch, which is what tells the arch's inside from its outside. Where that
//! fails — a part whose cut has no clear inside — facing names the side by direction instead.

use glam::DVec2;
use serde::{Deserialize, Serialize};

/// One skin of a wall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Skin {
    Outer,
    Inner,
}

/// Points whose outward direction lies within `within_deg` of `dir`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Facing {
    pub dir: DVec2,
    pub within_deg: f64,
}

/// Which points to keep: a skin, a facing, or both (a point must pass both).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SideRule {
    pub skin: Option<Skin>,
    pub facing: Option<Facing>,
}

/// A cut laid flat as chains of points, each point with its outward direction on the sheet.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SidedChains {
    pub chains: Vec<Vec<DVec2>>,
    pub normals: Vec<Vec<DVec2>>,
}

/// How far off its direction facing takes a point's outward direction unless told, degrees.
pub const FACING_DEG: f64 = 60.0;

/// How many neighbours on each side a point's direction is smoothed over.
const SMOOTH: i64 = 2;

/// A gap in a kept stretch, or a kept stretch, shorter than this many point spacings takes
/// its neighbours' verdict.
const CLEAN_SPACINGS: f64 = 4.0;

/// How much thicker than the wall (the median over the cut) the material behind a point of
/// the inner skin may be.
const WALL_SHARE: f64 = 1.5;

/// How many points on each side of a point a ray from it ignores on its own chain — the
/// surface it starts from.
const RAY_SKIP: i64 = 2;

/// Segments are boxed in groups of this many for the ray tests.
const CHUNK: usize = 16;

/// A cut's points and their outward directions as chains — the two laid flat point by point,
/// `offsets` giving where each chain starts in them (and, last, where the final one ends).
/// A direction missing from `normals` is read as none.
pub fn sided_chains(points: &[DVec2], offsets: &[usize], normals: &[DVec2]) -> SidedChains {
    let mut chains = Vec::new();
    let mut out = Vec::new();
    for c in 0..offsets.len().saturating_sub(1) {
        let mut chain = Vec::new();
        let mut n = Vec::new();
        for i in offsets[c]..offsets[c + 1] {
            let Some(&p) = points.get(i) else { break };
            chain.push(p);
            n.push(normals.get(i).copied().unwrap_or(DVec2::ZERO));
        }
        chains.push(chain);
        out.push(n);
    }
    SidedChains { chains, normals: out }
}

/// Whether a chain comes back onto its own first point, as a section's loops do.
fn is_closed(c: &[DVec2]) -> bool {
    c.len() > 2 && c[0].x == c[c.len() - 1].x && c[0].y == c[c.len() - 1].y
}

/// Per chain, per point, whether the point is on the side the rule asks for. The chains and
/// their directions are a whole cut (`sided_chains`): the skins are judged against all of it,
/// so pass every chain, not the few wanted.
pub fn side_mask(chains: &[Vec<DVec2>], normals: &[Vec<DVec2>], rule: &SideRule) -> Vec<Vec<bool>> {
    let closed: Vec<bool> = chains.iter().map(|c| is_closed(c)).collect();
    let lens: Vec<usize> = chains.iter().zip(&closed).map(|(c, &k)| if k { c.len() - 1 } else { c.len() }).collect();
    let normal_at = |c: usize, j: usize| normals.get(c).and_then(|n| n.get(j)).copied().unwrap_or(DVec2::ZERO);
    // Directions smoothed along each chain, round a loop.
    let dirs: Vec<Vec<DVec2>> = (0..chains.len())
        .map(|c| {
            let n = lens[c] as i64;
            (0..n)
                .map(|i| {
                    let (mut u, mut v) = (0.0, 0.0);
                    for k in -SMOOTH..=SMOOTH {
                        let mut j = i + k;
                        if closed[c] {
                            j = j.rem_euclid(n);
                        } else if j < 0 || j >= n {
                            continue;
                        }
                        let q = normal_at(c, j as usize);
                        u += q.x;
                        v += q.y;
                    }
                    let l = u.hypot(v);
                    if l > 1e-9 {
                        DVec2::new(u / l, v / l)
                    } else {
                        DVec2::ZERO
                    }
                })
                .collect()
        })
        .collect();
    // The spacing of the points, and the middle of the cut by length.
    let mut steps: Vec<f64> = Vec::new();
    let (mut mu, mut mv, mut total) = (0.0, 0.0, 0.0);
    for chain in chains {
        for i in 1..chain.len() {
            let l = (chain[i].x - chain[i - 1].x).hypot(chain[i].y - chain[i - 1].y);
            if l <= 0.0 {
                continue;
            }
            steps.push(l);
            mu += ((chain[i].x + chain[i - 1].x) / 2.0) * l;
            mv += ((chain[i].y + chain[i - 1].y) / 2.0) * l;
            total += l;
        }
    }
    steps.sort_by(|a, b| a.total_cmp(b));
    let spacing = steps.get(steps.len() >> 1).copied().unwrap_or(0.0);
    let middle = if total > 0.0 { DVec2::new(mu / total, mv / total) } else { DVec2::ZERO };
    let away_from_middle = |c: usize, i: usize| {
        let d = dirs[c][i];
        d.x * (chains[c][i].x - middle.x) + d.y * (chains[c][i].y - middle.y)
    };
    let live = |c: usize, i: usize| dirs[c][i].x != 0.0 || dirs[c][i].y != 0.0;
    let clean = |mask: &[Vec<bool>]| -> Vec<Vec<bool>> { mask.iter().enumerate().map(|(c, m)| clean_along(&chains[c], m, closed[c], CLEAN_SPACINGS * spacing)).collect() };
    let rays = rule.skin.map(|_| RayCaster::new(chains, &closed));

    let mut outer: Option<Vec<Vec<bool>>> = None;
    if let Some(rays) = rays.as_ref() {
        let raw: Vec<Vec<bool>> = (0..chains.len()).map(|c| (0..lens[c]).map(|i| live(c, i) && away_from_middle(c, i) > 0.0 && rays.first(c, i, dirs[c][i]).is_none()).collect()).collect();
        outer = Some(clean(&raw));
    }
    let mut side: Vec<Vec<bool>> = (0..chains.len()).map(|c| vec![true; lens[c]]).collect();
    match (rule.skin, outer.as_ref(), rays.as_ref()) {
        (Some(Skin::Outer), Some(o), _) => side = o.clone(),
        (Some(Skin::Inner), Some(o), Some(rays)) => {
            // How far through the material each point facing the middle comes out on the
            // outer skin; NaN where it does not.
            let through: Vec<Vec<f64>> = (0..chains.len())
                .map(|c| {
                    (0..lens[c])
                        .map(|i| {
                            if !live(c, i) || away_from_middle(c, i) >= 0.0 {
                                return f64::NAN;
                            }
                            let d = dirs[c][i];
                            let Some(hit) = rays.first(c, i, -d) else { return f64::NAN };
                            let ends = &o[hit.chain];
                            let n = lens[hit.chain];
                            let at = |k: usize| ends.get(k).copied().unwrap_or(false);
                            if n > 0 && (at(hit.seg) || at((hit.seg + 1) % n)) {
                                hit.t
                            } else {
                                f64::NAN
                            }
                        })
                        .collect()
                })
                .collect();
            // The wall's thickness, as most of those points have it: a rib's tip looks through
            // the rib and the wall onto the outer skin too, but from much farther.
            let mut all: Vec<f64> = through.iter().flatten().copied().filter(|t| !t.is_nan()).collect();
            all.sort_by(|a, b| a.total_cmp(b));
            let wall = all.get(all.len() >> 1).copied().unwrap_or(0.0);
            let reach = WALL_SHARE * wall + 2.0 * spacing;
            side = through.iter().map(|m| m.iter().map(|&t| t <= reach).collect()).collect();
        }
        _ => {}
    }
    if let Some(f) = rule.facing {
        let fl = f.dir.x.hypot(f.dir.y);
        let cos = ((f.within_deg * std::f64::consts::PI) / 180.0).cos();
        side = side
            .iter()
            .enumerate()
            .map(|(c, m)| m.iter().enumerate().map(|(i, &k)| k && fl > 0.0 && live(c, i) && (dirs[c][i].x * f.dir.x + dirs[c][i].y * f.dir.y) / fl >= cos).collect())
            .collect();
    }
    if rule.skin == Some(Skin::Inner) || rule.facing.is_some() {
        side = clean(&side);
    }
    // A loop's last point is its first again.
    side.into_iter()
        .enumerate()
        .map(|(c, mut m)| {
            if closed[c] {
                if let Some(&first) = m.first() {
                    m.push(first);
                }
            }
            m
        })
        .collect()
}

/// A verdict along a chain cleaned of stretches shorter than `least` mm: first the gaps
/// between kept stretches filled, then the kept stretches left that short dropped. A gap at an
/// open chain's end is left.
fn clean_along(chain: &[DVec2], mask: &[bool], closed: bool, least: f64) -> Vec<bool> {
    let n = mask.len();
    if n == 0 || !(least > 0.0) {
        return mask.to_vec();
    }
    let mut out = mask.to_vec();
    let step = |i: usize| {
        let j = if i + 1 < n { i + 1 } else { 0 };
        match (chain.get(i), chain.get(j)) {
            (Some(a), Some(b)) => (b.x - a.x).hypot(b.y - a.y),
            _ => 0.0,
        }
    };
    for value in [false, true] {
        // Where to start walking: a loop from a change, so no stretch is split.
        let mut start = 0;
        if closed {
            match (0..n).position(|i| out[i] != out[(i + n - 1) % n]) {
                Some(s) => start = s,
                None => continue,
            }
        }
        let mut i = 0;
        while i < n {
            let at = (start + i) % n;
            if out[at] != value {
                i += 1;
                continue;
            }
            // A stretch of `value` from here.
            let mut j = i;
            let mut length = 0.0;
            while j < n && out[(start + j) % n] == value {
                if j + 1 < n {
                    length += step((start + j) % n);
                }
                j += 1;
            }
            let bounded = closed || (i > 0 && j < n);
            // A gap is filled only between kept stretches; a kept stretch goes wherever it
            // lies.
            if length < least && (value || bounded) {
                for k in i..j {
                    out[(start + k) % n] = !value;
                }
            }
            i = j;
        }
    }
    out
}

/// The runs of a chain where `keep` holds of its points, in order along it — each a chain of
/// its own, with at least `min_points` points. A loop's run that runs through its first point
/// comes out as one; a loop kept whole comes out whole, closed.
pub fn runs_where(chain: &[DVec2], keep: impl Fn(usize) -> bool, min_points: usize) -> Vec<Vec<DVec2>> {
    run_indices(chain.len(), is_closed(chain), keep, min_points).into_iter().map(|run| run.into_iter().map(|i| chain[i]).collect()).collect()
}

/// `runs_where` by index: the runs of a chain of `count` points where `keep` holds, as the
/// indices of their points. A closed chain's last point is its first again; a loop kept whole
/// ends on it.
pub fn run_indices(count: usize, closed: bool, keep: impl Fn(usize) -> bool, min_points: usize) -> Vec<Vec<usize>> {
    let n = if closed { count.saturating_sub(1) } else { count };
    let flags: Vec<bool> = (0..n).map(&keep).collect();
    if closed && flags.iter().all(|&x| x) {
        return if count >= min_points { vec![(0..count).collect()] } else { Vec::new() };
    }
    // A loop is walked from a point it leaves out, so its runs come whole.
    let start = if closed { flags.iter().position(|&x| !x).unwrap_or(0) } else { 0 };
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut run: Vec<usize> = Vec::new();
    for k in 0..n {
        let i = (start + k) % n;
        if flags[i] {
            run.push(i);
        } else if !run.is_empty() {
            runs.push(std::mem::take(&mut run));
        }
    }
    if !run.is_empty() {
        runs.push(run);
    }
    runs.into_iter().filter(|r| r.len() >= min_points).collect()
}

/// A run of one chain of a cut, kept for a side.
#[derive(Debug, Clone, PartialEq)]
pub struct SideRun {
    pub chain: usize,
    pub points: Vec<DVec2>,
}

/// The runs of every chain on the side a rule asks for — what a section read for one skin is
/// fitted on.
pub fn side_runs(sided: &SidedChains, rule: &SideRule, min_points: usize) -> Vec<SideRun> {
    let mask = side_mask(&sided.chains, &sided.normals, rule);
    sided
        .chains
        .iter()
        .enumerate()
        .flat_map(|(c, chain)| {
            let m = &mask[c];
            runs_where(chain, |i| m.get(i).copied().unwrap_or(false), min_points).into_iter().map(move |points| SideRun { chain: c, points })
        })
        .collect()
}

/// Where a ray first crosses the cut: segment `seg` (from point `seg` to point `seg + 1`) of
/// chain `chain`, `t` along the ray.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Hit {
    chain: usize,
    seg: usize,
    t: f64,
}

/// A short run of a chain's segments and the box round them.
struct SegBox {
    chain: usize,
    from: usize,
    to: usize,
    lo: DVec2,
    hi: DVec2,
}

/// Rays cast across a cut's segments, nearest crossing first; the segments boxed in short runs
/// so most are passed over at once.
struct RayCaster<'a> {
    chains: &'a [Vec<DVec2>],
    closed: &'a [bool],
    boxes: Vec<SegBox>,
}

impl<'a> RayCaster<'a> {
    fn new(chains: &'a [Vec<DVec2>], closed: &'a [bool]) -> Self {
        let mut boxes = Vec::new();
        for (c, chain) in chains.iter().enumerate() {
            let mut from = 0;
            while from + 1 < chain.len() {
                let to = (chain.len() - 1).min(from + CHUNK);
                let mut lo = DVec2::splat(f64::INFINITY);
                let mut hi = DVec2::splat(f64::NEG_INFINITY);
                for p in &chain[from..=to] {
                    lo.x = lo.x.min(p.x);
                    lo.y = lo.y.min(p.y);
                    hi.x = hi.x.max(p.x);
                    hi.y = hi.y.max(p.y);
                }
                boxes.push(SegBox { chain: c, from, to, lo, hi });
                from += CHUNK;
            }
        }
        Self { chains, closed, boxes }
    }

    /// The nearest segment the ray from point `i` of chain `c` along `d` crosses, the segments
    /// near the point on its own chain aside; `None` when it leaves the cut.
    fn first(&self, c: usize, i: usize, d: DVec2) -> Option<Hit> {
        let o = self.chains[c][i];
        let closed = self.closed.get(c).copied().unwrap_or(false);
        let n = if closed { self.chains[c].len() as i64 - 1 } else { self.chains[c].len() as i64 };
        let ii = i as i64;
        let near = |s: usize| {
            let s = s as i64;
            if !closed {
                return s >= ii - RAY_SKIP - 1 && s <= ii + RAY_SKIP;
            }
            let k = (s - ii).rem_euclid(n.max(1));
            k <= RAY_SKIP || k >= n - RAY_SKIP - 1
        };
        let mut best: Option<Hit> = None;
        for b in &self.boxes {
            if !ray_meets_box(o, d, b.lo, b.hi, best.map_or(f64::INFINITY, |h| h.t)) {
                continue;
            }
            let chain = &self.chains[b.chain];
            for s in b.from..b.to {
                if b.chain == c && near(s) {
                    continue;
                }
                if let Some(t) = ray_segment(o, d, chain[s], chain[s + 1]) {
                    if best.is_none_or(|h| t < h.t) {
                        best = Some(Hit { chain: b.chain, seg: s, t });
                    }
                }
            }
        }
        best
    }
}

/// Whether the ray from `o` along `d` meets the box before `far`.
fn ray_meets_box(o: DVec2, d: DVec2, lo: DVec2, hi: DVec2, far: f64) -> bool {
    let mut t0: f64 = 0.0;
    let mut t1 = far;
    for k in 0..2 {
        if d[k].abs() < 1e-15 {
            if o[k] < lo[k] || o[k] > hi[k] {
                return false;
            }
            continue;
        }
        let mut a = (lo[k] - o[k]) / d[k];
        let mut b = (hi[k] - o[k]) / d[k];
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        t0 = t0.max(a);
        t1 = t1.min(b);
        if t0 > t1 {
            return false;
        }
    }
    true
}

/// How far along the ray from `o` along `d` it crosses the segment a–b, or `None` where it does
/// not (or only at its start).
fn ray_segment(o: DVec2, d: DVec2, a: DVec2, b: DVec2) -> Option<f64> {
    let (ex, ey) = (b.x - a.x, b.y - a.y);
    let den = d.x * ey - d.y * ex;
    if den.abs() < 1e-15 {
        return None;
    }
    let (wx, wy) = (a.x - o.x, a.y - o.y);
    let t = (wx * ey - wy * ex) / den;
    let s = (wx * d.y - wy * d.x) / den;
    (t > 1e-9 && (0.0..=1.0).contains(&s)).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit circle's loop of 20 points, closed on its first.
    fn unit_loop() -> Vec<DVec2> {
        let mut l: Vec<DVec2> = (0..21).map(|i| DVec2::from_angle((i as f64 / 20.0) * 2.0 * std::f64::consts::PI)).collect();
        l[20] = l[0];
        l
    }

    #[test]
    fn keeps_a_loops_run_through_its_first_point_as_one() {
        let lp = unit_loop();
        let runs = runs_where(&lp, |i| !(5..=15).contains(&i), 4);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].len(), 9);
        assert_eq!(runs[0][0], lp[16]);
        assert_eq!(runs[0][8], lp[4]);
    }

    #[test]
    fn keeps_a_loop_kept_whole_closed_and_drops_runs_too_short_to_fit() {
        let lp = unit_loop();
        assert_eq!(runs_where(&lp, |_| true, 4), vec![lp.clone()]);
        let open = &lp[..20];
        assert_eq!(runs_where(open, |i| i != 3 && i != 10, 4).iter().map(|x| x.len()).collect::<Vec<_>>(), vec![6, 9]);
    }

    /// `densify` of ScanRuler's shell tests: points `step` apart round a polygon, its corners
    /// among them — with each point's direction out of the material, the side's own (a
    /// polygon wound counter-clockwise round the material has the material on its left).
    fn sided_polygon(poly: &[[f64; 2]], step: f64) -> (Vec<DVec2>, Vec<DVec2>) {
        let (mut pts, mut out) = (Vec::new(), Vec::new());
        for i in 0..poly.len() {
            let (a, b) = (DVec2::from(poly[i]), DVec2::from(poly[(i + 1) % poly.len()]));
            let k = ((b - a).length() / step).ceil().max(1.0) as usize;
            let n = (b - a).normalize();
            for t in 0..k {
                pts.push(a + (b - a) * (t as f64 / k as f64));
                out.push(DVec2::new(n.y, -n.x));
            }
        }
        pts.push(pts[0]);
        out.push(out[0]);
        (pts, out)
    }

    /// ScanRuler's sliceSkin case, the cut laid flat by hand rather than sliced from a mesh:
    /// a solid has an outer skin all round and no inner skin.
    #[test]
    fn a_solid_has_an_outer_skin_all_round_and_no_inner_skin() {
        let (chain, normals) = sided_polygon(&[[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0]], 0.5);
        let sided = SidedChains { chains: vec![chain.clone()], normals: vec![normals] };
        let outer = side_runs(&sided, &SideRule { skin: Some(Skin::Outer), facing: None }, 4);
        assert_eq!(outer.len(), 1);
        assert_eq!(outer[0].points.len(), chain.len());
        assert_eq!(side_runs(&sided, &SideRule { skin: Some(Skin::Inner), facing: None }, 4), Vec::new());
    }

    /// ScanRuler's sliceSkin case, the cut laid flat by hand: a square tube 2 mm thick, cut
    /// across — two loops, the inside one wound the other way round — has its inside as the
    /// inner skin.
    #[test]
    fn a_hollow_box_has_its_inside_as_the_inner_skin() {
        let (big, big_n) = sided_polygon(&[[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]], 0.5);
        let (small, small_n) = sided_polygon(&[[2.0, 2.0], [2.0, 28.0], [38.0, 28.0], [38.0, 2.0]], 0.5);
        let chains = vec![small, big];
        let normals = vec![small_n, big_n];
        let inner = side_mask(&chains, &normals, &SideRule { skin: Some(Skin::Inner), facing: None });
        assert!(!inner[1].iter().any(|&k| k));
        assert!(inner[0].iter().all(|&k| k));
        let outer = side_mask(&chains, &normals, &SideRule { skin: Some(Skin::Outer), facing: None });
        assert!(outer[1].iter().all(|&k| k));
        assert!(!outer[0].iter().any(|&k| k));
    }

    /// Not a ScanRuler test: the runs of a loop come whole through its first point, and a loop
    /// kept whole stays closed.
    #[test]
    fn a_loops_run_through_its_first_point_comes_whole() {
        let square: Vec<DVec2> = (0..8).map(|i| DVec2::new(i as f64, 0.0)).chain(std::iter::once(DVec2::ZERO)).collect();
        let runs = run_indices(square.len(), true, |i| i != 4, 2);
        assert_eq!(runs, vec![vec![5, 6, 7, 0, 1, 2, 3]]);
        assert_eq!(run_indices(square.len(), true, |_| true, 4), vec![(0..9).collect::<Vec<_>>()]);
        assert_eq!(runs_where(&square, |i| i < 3, 4).len(), 0);
    }

    /// Not a ScanRuler test: a ring of two walls, the outer facing out and the inner facing in;
    /// the outer skin is the outer wall, the inner skin the inner one.
    #[test]
    fn tells_the_outer_skin_of_a_ring_from_the_inner() {
        let ring = |r: f64, out: f64| -> (Vec<DVec2>, Vec<DVec2>) {
            let n = 72;
            let mut p: Vec<DVec2> = (0..n).map(|i| DVec2::from_angle(std::f64::consts::TAU * i as f64 / n as f64) * r).collect();
            let mut q: Vec<DVec2> = (0..n).map(|i| DVec2::from_angle(std::f64::consts::TAU * i as f64 / n as f64) * out).collect();
            p.push(p[0]);
            q.push(q[0]);
            (p, q)
        };
        let (outer, on) = ring(20.0, 1.0);
        let (inner, inn) = ring(18.0, -1.0);
        let chains = vec![outer, inner];
        let normals = vec![on, inn];
        let o = side_mask(&chains, &normals, &SideRule { skin: Some(Skin::Outer), facing: None });
        assert!(o[0].iter().all(|&k| k));
        assert!(o[1].iter().all(|&k| !k));
        let i = side_mask(&chains, &normals, &SideRule { skin: Some(Skin::Inner), facing: None });
        assert!(i[0].iter().all(|&k| !k));
        assert!(i[1].iter().all(|&k| k));
    }
}
