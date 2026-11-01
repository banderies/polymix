// SPDX-License-Identifier: AGPL-3.0-only
//! A slice's chains made ready for a sketch. Ported from ScanRuler
//! (`plugins/reverse/core/chainOutline.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A scan full of small holes cuts into chains that stop wherever the surface has a gap, and
//! nothing joins them again: so the ends that stand closer than a gap the caller allows are
//! joined across it, nearest first — two chains into one, a chain's own two ends into a loop.
//! Chains are told apart by their size and by a point one closes round; and a chain is thinned
//! to the few corners a polyline needs (Ramer–Douglas–Peucker, `simplify`), the sides that run
//! nearly along the sheet's axes laid exactly along them where asked.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use glam::{DVec2, DVec3};

use crate::chain::view_chain;
use crate::frame::SectionFrame;
use crate::section_skin::{runs_where, side_mask, SideRule, SidedChains};
use crate::simplify::simplify_run;

/// Whether a chain comes back onto its own first point, as a section's loops do.
pub fn chain_closed(c: &[DVec2]) -> bool {
    c.len() > 2 && (c[0].x - c[c.len() - 1].x).hypot(c[0].y - c[c.len() - 1].y) < 1e-9
}

/// A chain's length along its points, mm.
pub fn chain_length(c: &[DVec2]) -> f64 {
    let mut l = 0.0;
    for i in 1..c.len() {
        l += (c[i].x - c[i - 1].x).hypot(c[i].y - c[i - 1].y);
    }
    l
}

/// The area a closed chain encloses; zero for an open one.
pub fn chain_area(c: &[DVec2]) -> f64 {
    if !chain_closed(c) {
        return 0.0;
    }
    let mut a2 = 0.0;
    for i in 1..c.len() {
        a2 += c[i - 1].x * c[i].y - c[i].x * c[i - 1].y;
    }
    a2.abs() / 2.0
}

/// Whether a point lies inside a closed chain (even–odd).
fn inside(p: DVec2, c: &[DVec2]) -> bool {
    let mut odd = false;
    for i in 1..c.len() {
        let (a, b) = (c[i - 1], c[i]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + ((p.y - a.y) * (b.x - a.x)) / (b.y - a.y) {
            odd = !odd;
        }
    }
    odd
}

/// How far a point lies from a segment.
fn to_segment(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 1e-18 { (((p.x - a.x) * dx + (p.y - a.y) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p.x - a.x - t * dx).hypot(p.y - a.y - t * dy)
}

/// How far a point lies from a chain.
fn to_chain(p: DVec2, c: &[DVec2]) -> f64 {
    if c.len() == 1 {
        return (p.x - c[0].x).hypot(p.y - c[0].y);
    }
    let mut best = f64::INFINITY;
    for i in 1..c.len() {
        best = best.min(to_segment(p, c[i - 1], c[i]));
    }
    best
}

/// `Math.round` as JavaScript has it: halves round up, towards +∞.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// A chain after the gaps were bridged: its points, the chains it was made of (by their index
/// before, ascending — one where nothing joined it), and the gaps bridged in it with the
/// widest.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinedChain {
    pub chain: Vec<DVec2>,
    pub from: Vec<usize>,
    pub gaps: usize,
    pub largest: f64,
}

/// The chains as they are, nothing joined.
pub fn unjoined(chains: &[Vec<DVec2>]) -> Vec<JoinedChain> {
    chains.iter().enumerate().map(|(i, chain)| JoinedChain { chain: chain.clone(), from: vec![i], gaps: 0, largest: 0.0 }).collect()
}

/// The open chains' ends joined wherever two stand within `gap` mm of each other — the ends of
/// two chains, or a chain's own two ends — nearest pairs first, each end joined once. A bridge
/// is filled with points at the chains' spacing, so a fit sees a straight run across the hole;
/// a chain that comes round onto itself ends on its own first point, as a section's loops do.
/// The chains come back in the order of the lowest index each was made of.
pub fn join_chain_ends(chains: &[Vec<DVec2>], gap: f64) -> Vec<JoinedChain> {
    let n = chains.len();
    let open: Vec<bool> = chains.iter().map(|c| c.len() >= 2 && !chain_closed(c)).collect();
    // End 2i is chain i's first point, 2i + 1 its last.
    let end = |e: usize| -> DVec2 {
        let c = &chains[e >> 1];
        if e & 1 == 1 {
            c[c.len() - 1]
        } else {
            c[0]
        }
    };
    let dist = |a: DVec2, b: DVec2| (a.x - b.x).hypot(a.y - b.y);
    struct Pair {
        e: usize,
        f: usize,
        d: f64,
    }
    let mut pairs: Vec<Pair> = Vec::new();
    for i in 0..n {
        if !open[i] {
            continue;
        }
        if chains[i].len() >= 3 {
            let d = dist(end(2 * i), end(2 * i + 1));
            if d <= gap {
                pairs.push(Pair { e: 2 * i, f: 2 * i + 1, d });
            }
        }
        for j in i + 1..n {
            if !open[j] {
                continue;
            }
            for e in [2 * i, 2 * i + 1] {
                for f in [2 * j, 2 * j + 1] {
                    let d = dist(end(e), end(f));
                    if d <= gap {
                        pairs.push(Pair { e, f, d });
                    }
                }
            }
        }
    }
    if pairs.is_empty() {
        return unjoined(chains);
    }
    pairs.sort_by(|a, b| a.d.total_cmp(&b.d));
    let mut mate: Vec<Option<usize>> = vec![None; 2 * n];
    let mut width = vec![0.0; 2 * n];
    for p in &pairs {
        if mate[p.e].is_some() || mate[p.f].is_some() {
            continue;
        }
        mate[p.e] = Some(p.f);
        mate[p.f] = Some(p.e);
        width[p.e] = p.d;
        width[p.f] = p.d;
    }
    // The points a bridge is filled with, a step apart: the chains' own.
    let mut spacings: Vec<f64> = chains.iter().filter(|c| c.len() >= 2).map(|c| view_chain(c).spacing).collect();
    spacings.sort_by(|a, b| a.total_cmp(b));
    let step = spacings.get(spacings.len() >> 1).copied().unwrap_or(f64::INFINITY).max(1e-3);
    let bridge = |out: &mut Vec<DVec2>, a: DVec2, b: DVec2| {
        let k = ((b.x - a.x).hypot(b.y - a.y) / step).floor();
        let k = if k.is_finite() && k > 0.0 { k as usize } else { 0 };
        for t in 1..k {
            out.push(DVec2::new(a.x + ((b.x - a.x) * t as f64) / k as f64, a.y + ((b.y - a.y) * t as f64) / k as f64));
        }
    };
    let mut done = vec![false; n];
    let mut out: Vec<JoinedChain> = Vec::new();
    for i in 0..n {
        if done[i] {
            continue;
        }
        if !open[i] || (mate[2 * i].is_none() && mate[2 * i + 1].is_none()) {
            done[i] = true;
            out.push(JoinedChain { chain: chains[i].clone(), from: vec![i], gaps: 0, largest: 0.0 });
            continue;
        }
        // Back to where the run starts — its free end, or round a loop to this chain again.
        let mut entry = 2 * i;
        for _guard in 0..=n {
            match mate[entry] {
                Some(m) if (m ^ 1) != 2 * i => entry = m ^ 1,
                _ => break,
            }
        }
        let start = entry;
        let mut pts: Vec<DVec2> = Vec::new();
        let mut from: Vec<usize> = Vec::new();
        let (mut gaps, mut largest) = (0usize, 0.0f64);
        // Each chain is walked once at most; the guard only stops a walk that could not end.
        for _guard in 0..=n {
            let c = &chains[entry >> 1];
            done[entry >> 1] = true;
            from.push(entry >> 1);
            let walk: Vec<DVec2> = if entry & 1 == 1 { c.iter().rev().copied().collect() } else { c.clone() };
            if let (Some(&last), Some(&first)) = (pts.last(), walk.first()) {
                bridge(&mut pts, last, first);
            }
            pts.extend(walk);
            let exit = entry ^ 1;
            let Some(m) = mate[exit] else { break };
            gaps += 1;
            largest = largest.max(width[exit]);
            if m == start {
                if let (Some(&last), Some(&first)) = (pts.last(), pts.first()) {
                    bridge(&mut pts, last, first);
                    pts.push(first);
                }
                break;
            }
            entry = m;
        }
        from.sort_unstable();
        out.push(JoinedChain { chain: pts, from, gaps, largest });
    }
    out.sort_by_key(|c| c.from.first().copied().unwrap_or(usize::MAX));
    out
}

/// The gaps bridged over a set of joined chains, and the widest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bridged {
    pub gaps: usize,
    pub largest: f64,
}

/// The gaps bridged over a set of joined chains, and the widest.
pub fn bridged_of(chains: &[JoinedChain]) -> Bridged {
    let (mut gaps, mut largest) = (0, 0.0f64);
    for c in chains {
        gaps += c.gaps;
        largest = largest.max(c.largest);
    }
    Bridged { gaps, largest }
}

/// A box on the sheet, [u, v] mm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheetBox {
    pub min: DVec2,
    pub max: DVec2,
}

impl SheetBox {
    fn holds(&self, p: DVec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
}

/// A chain's runs that lie in a box on the sheet — each long enough to fit, in order along the
/// chain.
pub fn runs_within(chain: &[DVec2], sheet_box: SheetBox) -> Vec<Vec<DVec2>> {
    runs_kept(chain, &KeepOptions { sheet_box: Some(sheet_box), side: None, test: None })
}

/// A point's verdict for a side, kept by the point — by its coordinates, here.
pub type SideVerdicts = HashMap<(u64, u64), bool>;

/// The key a point's verdict is kept under: its coordinates' bits, a negative zero read as
/// zero.
pub fn point_key(p: DVec2) -> (u64, u64) {
    ((p.x + 0.0).to_bits(), (p.y + 0.0).to_bits())
}

/// A cut's points on the side a rule asks for (`section_skin`), each point's verdict kept by
/// the point itself: joining chains across gaps keeps the points it joins, so the verdict
/// outlives it. The chains are a whole cut, laid flat with their outward directions.
pub fn side_of(sided: &SidedChains, rule: &SideRule) -> SideVerdicts {
    let mask = side_mask(&sided.chains, &sided.normals, rule);
    let mut out = SideVerdicts::new();
    for (c, chain) in sided.chains.iter().enumerate() {
        for (i, p) in chain.iter().enumerate() {
            out.insert(point_key(*p), mask[c].get(i).copied().unwrap_or(false));
        }
    }
    out
}

/// What `runs_kept` keeps a point by: a box on the sheet, a side, a test of the point — each
/// where given.
#[derive(Clone, Copy, Default)]
pub struct KeepOptions<'a> {
    pub sheet_box: Option<SheetBox>,
    pub side: Option<&'a SideVerdicts>,
    pub test: Option<&'a dyn Fn(DVec2) -> bool>,
}

/// A chain's runs that lie in a box on the sheet, on a side (`side_of`), where a test of the
/// point holds, or all of them — each long enough to fit (four points), in order along it; a
/// loop's run through its first point is one run, a loop kept whole stays closed. A point a
/// bridge across a gap added takes the verdict of the points either side of the gap, kept
/// where both are.
pub fn runs_kept(chain: &[DVec2], opts: &KeepOptions) -> Vec<Vec<DVec2>> {
    let mut on_side: Option<Vec<bool>> = None;
    if let Some(side) = opts.side {
        let n = chain.len();
        let known: Vec<Option<bool>> = chain.iter().map(|p| side.get(&point_key(*p)).copied()).collect();
        // The nearest verdict before and after each point — round a loop, whose last bridge
        // closes it onto its first point.
        let lp = chain_closed(chain);
        let mut before: Vec<Option<bool>> = vec![None; n];
        let mut after: Vec<Option<bool>> = vec![None; n];
        let mut last = if lp { known.iter().rev().find_map(|k| *k) } else { None };
        for i in 0..n {
            last = known[i].or(last);
            before[i] = last;
        }
        let mut next = if lp { known.iter().find_map(|k| *k) } else { None };
        for i in (0..n).rev() {
            next = known[i].or(next);
            after[i] = next;
        }
        on_side = Some(known.iter().enumerate().map(|(i, k)| k.unwrap_or(before[i] == Some(true) && after[i] == Some(true))).collect());
    }
    runs_where(
        chain,
        |i| {
            let p = chain[i];
            opts.sheet_box.is_none_or(|b| b.holds(p)) && opts.test.is_none_or(|t| t(p)) && on_side.as_ref().is_none_or(|s| s[i])
        },
        4,
    )
}

/// A box a `within` takes: on the command's own sheet, [u, v] mm — or in the world, [x, y, z]
/// mm, whatever sheet the command reads on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WithinBox {
    Sheet(SheetBox),
    World { min: DVec3, max: DVec3 },
}

/// How far a point of the cut may stand outside a world box and still be in it, mm: a box
/// drawn flat on the plane the cut lies in.
const WORLD_BOX_SLACK_MM: f64 = 1e-6;

/// A world box as a test of points on a sheet, lifted into the world through the sheet's frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldBoxTest {
    pub min: DVec3,
    pub max: DVec3,
    pub frame: SectionFrame,
}

impl WorldBoxTest {
    pub fn holds(&self, p: DVec2) -> bool {
        let q = self.frame.to_world(p);
        (0..3).all(|k| q[k] >= self.min[k].min(self.max[k]) - WORLD_BOX_SLACK_MM && q[k] <= self.min[k].max(self.max[k]) + WORLD_BOX_SLACK_MM)
    }
}

/// What `runs_kept` takes for a within: a box on the sheet as it is, or a world box as a test
/// of each point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnSheet {
    pub sheet_box: Option<SheetBox>,
    pub world: Option<WorldBoxTest>,
}

/// What `runs_kept` takes for a within: a box on the sheet as it is; a world box as a test of
/// each point, lifted into the world.
pub fn within_on_sheet(w: &WithinBox, frame: &SectionFrame) -> OnSheet {
    match *w {
        WithinBox::Sheet(b) => OnSheet { sheet_box: Some(b), world: None },
        WithinBox::World { min, max } => OnSheet { sheet_box: None, world: Some(WorldBoxTest { min, max, frame: *frame }) },
    }
}

/// The box chains span on their sheet; `None` for no points.
pub fn chains_span(chains: &[Vec<DVec2>]) -> Option<(DVec2, DVec2)> {
    let mut min = DVec2::splat(f64::INFINITY);
    let mut max = DVec2::splat(f64::NEG_INFINITY);
    for p in chains.iter().flatten() {
        min = DVec2::new(min.x.min(p.x), min.y.min(p.y));
        max = DVec2::new(max.x.max(p.x), max.y.max(p.y));
    }
    min.x.is_finite().then_some((min, max))
}

/// The box chains span in the world, given their sheet's frame; `None` for no points.
pub fn chains_span_world(chains: &[Vec<DVec2>], frame: &SectionFrame) -> Option<(DVec3, DVec3)> {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for p in chains.iter().flatten() {
        let q = frame.to_world(*p);
        min = min.min(q);
        max = max.max(q);
    }
    min.x.is_finite().then_some((min, max))
}

/// What to say of a within that catches nothing of the chains: the box, and where the cut
/// lies — on `sheet` (the command's own frame, named in words), and in the world for a world
/// box.
pub fn box_missed(w: &WithinBox, chains: &[Vec<DVec2>], frame: Option<&SectionFrame>, sheet: &str) -> String {
    let r = |x: f64| js_round(x * 1e4) / 1e4 + 0.0;
    let list = |v: &[f64]| v.iter().map(|x| r(*x).to_string()).collect::<Vec<_>>().join(", ");
    let on_sheet = match chains_span(chains) {
        Some((min, max)) => format!("u {}…{}, v {}…{}", r(min.x), r(max.x), r(min.y), r(max.y)),
        None => "nothing".to_string(),
    };
    match *w {
        WithinBox::World { min, max } => {
            let in_world = match frame.and_then(|f| chains_span_world(chains, f)) {
                Some((a, b)) => format!("x {}…{}, y {}…{}, z {}…{}", r(a.x), r(b.x), r(a.y), r(b.y), r(a.z), r(b.z)),
                None => "nothing".to_string(),
            };
            format!("Nothing in the world box [{}]…[{}] (within.world); the cut spans {in_world} in the world — {on_sheet} on {sheet}.", list(&min.to_array()), list(&max.to_array()))
        }
        WithinBox::Sheet(b) => format!(
            "Nothing in the box [{}]…[{}] — [u, v] on {sheet}; the cut spans {on_sheet}. A box in world mm goes as within: {{ world: {{ min: [x, y, z], max: [x, y, z] }} }}.",
            list(&b.min.to_array()),
            list(&b.max.to_array())
        ),
    }
}

/// How near a point has to lie to a chain, in the chain's own scatters (`view_chain`), to name
/// that chain rather than the loop round it.
pub const ON_CHAIN_SCATTERS: f64 = 3.0;

/// Which chain a pick names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChainPick {
    /// The chain made of this one, by its index before joining.
    Index(usize),
    /// The closed one of the most area, else the longest.
    Largest,
    /// The chain a point names (`chain_at`).
    At(DVec2),
}

/// Which chain a pick names: by index (the chain made of it), the largest (the closed one of
/// the most area, else the longest), or by a point (`chain_at`). Its place in the list, or
/// `None` for an index no chain was made of.
pub fn pick_chain(chains: &[JoinedChain], pick: ChainPick) -> Option<usize> {
    match pick {
        ChainPick::Index(i) => chains.iter().position(|c| c.from.contains(&i)),
        _ if chains.is_empty() => None,
        ChainPick::At(at) => chain_at(chains, at).index,
        ChainPick::Largest => {
            let mut best = None;
            let mut best_key = f64::NEG_INFINITY;
            for (i, c) in chains.iter().enumerate() {
                let a = chain_area(&c.chain);
                let key = if a > 0.0 { 1e12 + a } else { chain_length(&c.chain) };
                if key > best_key {
                    best_key = key;
                    best = Some(i);
                }
            }
            best
        }
    }
}

/// By which rule `chain_at` named a chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainBy {
    On,
    Round,
    Nearest,
}

/// The chain a point names, the rule that named it, its area and how many closed chains lie
/// round the point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainAt {
    pub index: Option<usize>,
    pub by: ChainBy,
    pub area: f64,
    pub enclosing: usize,
}

/// The chain a point names, and by which rule: the chain it lies on, within
/// `ON_CHAIN_SCATTERS` of its scatter (the nearest such) — on; else the closed one round it
/// (the smallest) — round; else the nearest — nearest. A point on an opening's rim is the
/// opening, not the outline round it. With the chain's area (0 for an open one) and how many
/// closed chains lie round the point (enclosing), whichever rule named it. Index `None` for no
/// chains.
pub fn chain_at(chains: &[JoinedChain], at: DVec2) -> ChainAt {
    let round: Vec<usize> = chains.iter().enumerate().filter(|(_, c)| chain_closed(&c.chain) && inside(at, &c.chain)).map(|(i, _)| i).collect();
    let out = |index: Option<usize>, by: ChainBy| ChainAt { index, by, area: index.map_or(0.0, |i| chain_area(&chains[i].chain)), enclosing: round.len() };
    let mut best = None;
    let mut near = f64::INFINITY;
    for (i, c) in chains.iter().enumerate() {
        let d = to_chain(at, &c.chain);
        if d < near && d <= ON_CHAIN_SCATTERS * view_chain(&c.chain).scatter {
            near = d;
            best = Some(i);
        }
    }
    if best.is_some() {
        return out(best, ChainBy::On);
    }
    if let Some(&first) = round.first() {
        let smallest = round.iter().copied().fold(first, |a, b| if chain_area(&chains[b].chain) < chain_area(&chains[a].chain) { b } else { a });
        return out(Some(smallest), ChainBy::Round);
    }
    near = f64::INFINITY;
    for (i, c) in chains.iter().enumerate() {
        let d = to_chain(at, &c.chain);
        if d < near {
            near = d;
            best = Some(i);
        }
    }
    out(best, ChainBy::Nearest)
}

/// What `filter_chains` keeps.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChainFilter {
    pub min_length: Option<f64>,
    pub max_length: Option<f64>,
    pub closed_only: bool,
    pub max_chains: Option<usize>,
}

/// How many chains each rule of a filter left out — `None` where it left none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LeftOut {
    pub short: Option<usize>,
    pub long: Option<usize>,
    pub open: Option<usize>,
    pub beyond: Option<usize>,
}

/// The chains a filter kept, by their places in the list, and what it left out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filtered {
    pub kept: Vec<usize>,
    pub left: LeftOut,
}

/// Which chains a filter keeps: none shorter than `min_length`, none longer than
/// `max_length`, none open where `closed_only`, and of the rest the `max_chains` longest —
/// returned as their places in the list, in order — with how many each rule left out.
pub fn filter_chains(chains: &[Vec<DVec2>], opts: &ChainFilter) -> Filtered {
    let mut left = LeftOut::default();
    let lengths: Vec<f64> = chains.iter().map(|c| chain_length(c)).collect();
    let mut kept: Vec<usize> = (0..chains.len()).collect();
    let drop = |kept: &mut Vec<usize>, slot: &mut Option<usize>, keep: &dyn Fn(usize) -> bool| {
        let before = kept.len();
        kept.retain(|&i| keep(i));
        if kept.len() < before {
            *slot = Some(before - kept.len());
        }
    };
    if let Some(min) = opts.min_length {
        drop(&mut kept, &mut left.short, &|i| lengths[i] >= min);
    }
    if let Some(max) = opts.max_length {
        drop(&mut kept, &mut left.long, &|i| lengths[i] <= max);
    }
    if opts.closed_only {
        drop(&mut kept, &mut left.open, &|i| chain_closed(&chains[i]));
    }
    if let Some(max_chains) = opts.max_chains {
        if kept.len() > max_chains {
            left.beyond = Some(kept.len() - max_chains);
            let mut by_length = kept.clone();
            by_length.sort_by(|&a, &b| lengths[b].total_cmp(&lengths[a]));
            let longest: HashSet<usize> = by_length.into_iter().take(max_chains).collect();
            kept.retain(|i| longest.contains(i));
        }
    }
    Filtered { kept, left }
}

/// A chain thinned to a polyline: its corners, whether it closes, the tolerance it was thinned
/// at, how far the chain strays from it at most, and how many sides were laid along an axis.
#[derive(Debug, Clone, PartialEq)]
pub struct Outline {
    pub points: Vec<DVec2>,
    pub closed: bool,
    pub tolerance: f64,
    pub max_deviation: f64,
    pub snapped: usize,
    /// How many corners `simplify_deg` took out.
    pub merged: usize,
}

/// How `outline_of` thins a chain.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OutlineOptions {
    pub tolerance: f64,
    pub snap_deg: Option<f64>,
    pub max_points: Option<usize>,
    pub simplify_deg: Option<f64>,
}

/// The chain's points kept at a tolerance, as places along the walk — a loop's last point (its
/// first again) left out.
fn thinned(walk: &[DVec2], closed: bool, tolerance: f64) -> Vec<usize> {
    let mut kept = simplify_run(walk, 0, walk.len(), tolerance);
    if !closed {
        return kept;
    }
    kept.pop();
    // The walk's start is kept whatever it is; where it lies along a side, the side is one
    // piece without it.
    if kept.len() > 3 && walk.len() > 1 {
        let (prev, next) = (kept[kept.len() - 1], kept[1]);
        let lp = walk.len() - 1;
        let mut fits = true;
        let mut i = prev;
        while i < lp + next && fits {
            fits = to_segment(walk[i % lp], walk[prev], walk[next]) <= tolerance;
            i += 1;
        }
        if fits {
            kept.remove(0);
        }
    }
    kept
}

/// A chain as the polyline that keeps within `tolerance` of every point — the tolerance grown
/// until it takes at most `max_points` corners — and, with `snap_deg`, every side within that
/// many degrees of the sheet's u or v laid exactly along it, at the mean of the points it
/// stands for: a run of such sides is one side, and where a level side meets an upright one the
/// corner is square. With `simplify_deg`, a corner the polyline turns at by less than that is
/// taken out, so two sides that nearly run on are one.
pub fn outline_of(chain: &[DVec2], opts: &OutlineOptions) -> Outline {
    let closed = chain_closed(chain);
    let mut walk: Vec<DVec2> = chain.to_vec();
    if closed {
        // A loop starts where it most likely turns: the point farthest from its middle.
        let body = &chain[..chain.len() - 1];
        let len = body.len() as f64;
        let mid = body.iter().fold(DVec2::ZERO, |m, p| DVec2::new(m.x + p.x / len, m.y + p.y / len));
        let d = |p: DVec2| (p.x - mid.x).hypot(p.y - mid.y);
        let mut far = 0;
        for (i, p) in body.iter().enumerate() {
            if d(*p) > d(body[far]) {
                far = i;
            }
        }
        walk = body[far..].iter().chain(body[..far].iter()).copied().chain(std::iter::once(body[far])).collect();
    }
    let mut tolerance = opts.tolerance;
    let mut at = thinned(&walk, closed, tolerance);
    if let Some(max_points) = opts.max_points {
        let mut k = 0;
        while at.len() > max_points && k < 40 {
            tolerance *= 1.5;
            at = thinned(&walk, closed, tolerance);
            k += 1;
        }
    }
    let mut pts: Vec<DVec2> = at.iter().map(|&i| walk[i]).collect();
    let m = pts.len();
    let sides = if closed { m } else { m.saturating_sub(1) };
    // The points a side stands for, from its corner to the next.
    let span = |k: usize| -> (usize, usize) { (at[k], if k + 1 < m { at[k + 1] } else { walk.len() - 1 }) };
    let mut snapped = 0;
    let mut gone: HashSet<usize> = HashSet::new();
    if let Some(snap_deg) = opts.snap_deg.filter(|d| *d > 0.0) {
        if sides >= 1 {
            let lean = (snap_deg * PI) / 180.0;
            // 1: a level side (v held), 0: an upright one (u held), -1: neither.
            let axis_of = |k: usize| -> i8 {
                let (a, b) = (pts[k], pts[(k + 1) % m]);
                let ang = (b.y - a.y).abs().atan2((b.x - a.x).abs());
                if ang <= lean {
                    1
                } else if PI / 2.0 - ang <= lean {
                    0
                } else {
                    -1
                }
            };
            let kinds: Vec<i8> = (0..sides).map(axis_of).collect();
            // Runs of sides of one kind, a loop's taken round from a change.
            let first = if closed { (0..sides).position(|k| kinds[k] != kinds[(k + sides - 1) % sides]).unwrap_or(0) } else { 0 };
            let mut runs: Vec<(usize, Vec<usize>)> = Vec::new();
            for s in 0..sides {
                let k = (first + s) % sides;
                let kind = kinds[k];
                if kind < 0 {
                    continue;
                }
                let kind = kind as usize;
                let prev_k = (k + sides - 1) % sides;
                match runs.last_mut() {
                    Some((last_kind, last_sides)) if *last_kind == kind && last_sides.last() == Some(&prev_k) && s > 0 => last_sides.push(k),
                    _ => runs.push((kind, vec![k])),
                }
            }
            for (kind, run_sides) in &runs {
                if run_sides.len() == sides {
                    continue;
                }
                let (mut sum, mut count) = (0.0, 0usize);
                for &k in run_sides {
                    let (a, b) = span(k);
                    for i in a..=b {
                        sum += walk[i % walk.len()][*kind];
                        count += 1;
                    }
                }
                let level = sum / count as f64;
                for &k in run_sides {
                    pts[k][*kind] = level;
                    pts[(k + 1) % m][*kind] = level;
                }
                for &k in run_sides.iter().skip(1) {
                    gone.insert(k);
                }
                snapped += 1;
            }
        }
    }
    // The corners left, with no two on one point.
    let alive: Vec<usize> = (0..m).filter(|i| !gone.contains(i)).collect();
    let mut keep: Vec<usize> = alive
        .iter()
        .enumerate()
        .filter(|&(j, &i)| {
            let next = alive[(j + 1) % alive.len()];
            !(next != i && (closed || j + 1 < alive.len()) && (pts[next].x - pts[i].x).hypot(pts[next].y - pts[i].y) < 1e-6)
        })
        .map(|(_, &i)| i)
        .collect();
    // The corners turned at by less than simplify_deg go, the gentlest first, its neighbours
    // then read with the turn it leaves them: a long shallow curve keeps the corners its
    // turning adds up to, a crease between two sides that nearly run on does not.
    let mut merged = 0;
    if let Some(simplify_deg) = opts.simplify_deg.filter(|d| *d > 0.0) {
        let limit = (simplify_deg * PI) / 180.0;
        let turn_at = |keep: &[usize], j: usize| -> f64 {
            let n = keep.len();
            if !closed && (j == 0 || j == n - 1) {
                return f64::INFINITY;
            }
            let (a, b, c) = (pts[keep[(j + n - 1) % n]], pts[keep[j]], pts[keep[(j + 1) % n]]);
            let d = (c.y - b.y).atan2(c.x - b.x) - (b.y - a.y).atan2(b.x - a.x);
            d.sin().atan2(d.cos()).abs()
        };
        while keep.len() > if closed { 3 } else { 2 } {
            let mut best = None;
            let mut least = limit;
            for j in 0..keep.len() {
                let t = turn_at(&keep, j);
                if t < least {
                    least = t;
                    best = Some(j);
                }
            }
            let Some(best) = best else { break };
            keep.remove(best);
            merged += 1;
        }
    }
    // How far the chain strays from the polyline, side by side.
    let mut max_deviation: f64 = 0.0;
    let lp = walk.len().saturating_sub(1);
    let side_count = if closed { keep.len() } else { keep.len().saturating_sub(1) };
    for j in 0..side_count {
        let (a, b) = (keep[j], keep[(j + 1) % keep.len()]);
        let to = if b > a { at[b] } else { at[b] + lp };
        for i in at[a]..=to {
            let p = walk[if closed && lp > 0 { i % lp } else { i }];
            max_deviation = max_deviation.max(to_segment(p, pts[a], pts[b]));
        }
    }
    Outline { points: keep.iter().map(|&i| pts[i]).collect(), closed, tolerance, max_deviation, snapped, merged }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit2d::Mulberry32;

    /// Points from a to b, `step` apart, a's included and b's not.
    fn run(a: [f64; 2], b: [f64; 2], step: f64) -> Vec<DVec2> {
        let n = js_round((b[0] - a[0]).hypot(b[1] - a[1]) / step).max(1.0) as usize;
        (0..n).map(|i| DVec2::new(a[0] + ((b[0] - a[0]) * i as f64) / n as f64, a[1] + ((b[1] - a[1]) * i as f64) / n as f64)).collect()
    }

    /// A rectangle's outline from `lo` to `hi`, closed on its first point, with Gaussian noise
    /// of `noise` mm on every point.
    fn rectangle(lo: [f64; 2], hi: [f64; 2], noise: f64, seed: u32) -> Vec<DVec2> {
        let mut rand = Mulberry32::new(seed);
        let c = [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]];
        let pts: Vec<DVec2> = (0..4).flat_map(|i| run(c[i], c[(i + 1) % 4], 0.1)).collect();
        let mut noisy: Vec<DVec2> = pts
            .iter()
            .map(|p| {
                let x = p.x + rand.gaussian() * noise;
                let y = p.y + rand.gaussian() * noise;
                DVec2::new(x, y)
            })
            .collect();
        noisy.push(noisy[0]);
        noisy
    }

    fn halves() -> (Vec<DVec2>, Vec<DVec2>) {
        // Two halves of a 20 × 10 rectangle, each stopping 0.3 mm short of the other at both
        // ends.
        let mut left = run([9.85, 10.0], [0.0, 10.0], 0.1);
        left.extend(run([0.0, 10.0], [0.0, 0.0], 0.1));
        left.extend(run([0.0, 0.0], [9.85, 0.0], 0.1));
        left.push(DVec2::new(9.85, 0.0));
        let mut right = run([10.15, 0.0], [20.0, 0.0], 0.1);
        right.extend(run([20.0, 0.0], [20.0, 10.0], 0.1));
        right.extend(run([20.0, 10.0], [10.15, 10.0], 0.1));
        right.push(DVec2::new(10.15, 10.0));
        (left, right)
    }

    #[test]
    fn joins_two_open_chains_into_one_loop_where_the_gaps_are_within_reach() {
        let (left, right) = halves();
        let joined = join_chain_ends(&[left, right], 0.5);
        assert_eq!(joined.len(), 1);
        assert_eq!(joined[0].from, vec![0, 1]);
        assert_eq!(joined[0].gaps, 2);
        assert!((joined[0].largest - 0.3).abs() < 5e-7);
        assert!(chain_closed(&joined[0].chain));
        assert_eq!(bridged_of(&joined), Bridged { gaps: 2, largest: joined[0].largest });
    }

    #[test]
    fn leaves_them_apart_where_the_gaps_are_wider_than_allowed() {
        let (left, right) = halves();
        let joined = join_chain_ends(&[left, right], 0.2);
        assert_eq!(joined.len(), 2);
        assert_eq!(joined.iter().map(|j| j.from.clone()).collect::<Vec<_>>(), vec![vec![0], vec![1]]);
        assert_eq!(bridged_of(&joined).gaps, 0);
    }

    #[test]
    fn closes_a_chain_on_itself_and_leaves_a_closed_one_alone() {
        let full = rectangle([0.0, 0.0], [10.0, 10.0], 0.0, 3);
        let open = full[..full.len() - 3].to_vec();
        let closed = rectangle([30.0, 0.0], [40.0, 10.0], 0.0, 3);
        let joined = join_chain_ends(&[closed.clone(), open], 0.5);
        assert_eq!(joined.iter().map(|j| j.from.clone()).collect::<Vec<_>>(), vec![vec![0], vec![1]]);
        assert_eq!(joined[0].chain, closed);
        assert!(chain_closed(&joined[1].chain));
        assert_eq!(joined[1].gaps, 1);
    }

    fn told_apart() -> (Vec<DVec2>, Vec<DVec2>, Vec<DVec2>) {
        let big = rectangle([0.0, 0.0], [40.0, 20.0], 0.0, 3);
        let hole = rectangle([5.0, 5.0], [10.0, 10.0], 0.0, 3);
        let wall = run([50.0, 0.0], [50.0, 30.0], 0.1);
        (big, hole, wall)
    }

    #[test]
    fn picks_the_largest_the_one_round_a_point_and_the_nearest() {
        let (big, hole, wall) = told_apart();
        let chains = unjoined(&[hole, big, wall]);
        assert_eq!(pick_chain(&chains, ChainPick::Largest), Some(1));
        assert_eq!(pick_chain(&chains, ChainPick::At(DVec2::new(7.0, 7.0))), Some(0));
        assert_eq!(pick_chain(&chains, ChainPick::At(DVec2::new(30.0, 10.0))), Some(1));
        assert_eq!(pick_chain(&chains, ChainPick::At(DVec2::new(52.0, 10.0))), Some(2));
        assert_eq!(pick_chain(&chains, ChainPick::Index(2)), Some(2));
        assert_eq!(pick_chain(&chains, ChainPick::Index(5)), None);
    }

    #[test]
    fn takes_the_chain_a_point_lies_on_within_three_of_its_scatters_over_the_loop_round_it() {
        let (big, _, _) = told_apart();
        // A noisy opening inside the outline; the point on its rim, a hair outside it, lies
        // inside the outline only.
        let opening = rectangle([5.0, 5.0], [10.0, 10.0], 0.03, 7);
        let rimmed = unjoined(&[big, opening]);
        assert_eq!(pick_chain(&rimmed, ChainPick::At(DVec2::new(10.02, 7.0))), Some(1));
        assert_eq!(pick_chain(&rimmed, ChainPick::At(DVec2::new(9.98, 7.0))), Some(1));
        // A millimetre off the rim it is the loop round it again.
        assert_eq!(pick_chain(&rimmed, ChainPick::At(DVec2::new(11.0, 7.0))), Some(0));
        assert_eq!(pick_chain(&rimmed, ChainPick::At(DVec2::new(8.0, 7.0))), Some(1));
    }

    #[test]
    fn says_which_rule_named_the_chain_round_a_point_its_area_and_how_many_closed_chains_lie_round_it() {
        let (big, hole, wall) = told_apart();
        let chains = unjoined(&[hole, big, wall]);
        // Round 15: the fan wheel's blade — the agent could not tell on from round.
        let a = chain_at(&chains, DVec2::new(7.0, 7.0));
        assert_eq!((a.index, a.by, a.enclosing), (Some(0), ChainBy::Round, 2));
        assert!((a.area - 25.0).abs() < 5e-7);
        let b = chain_at(&chains, DVec2::new(30.0, 10.0));
        assert_eq!((b.index, b.by, b.enclosing), (Some(1), ChainBy::Round, 1));
        assert!((b.area - 800.0).abs() < 5e-7);
        let c = chain_at(&chains, DVec2::new(5.0, 7.0));
        assert_eq!((c.index, c.by, c.enclosing), (Some(0), ChainBy::On, 2));
        assert_eq!(chain_at(&chains, DVec2::new(50.01, 12.0)), ChainAt { index: Some(2), by: ChainBy::On, area: 0.0, enclosing: 0 });
        assert_eq!(chain_at(&chains, DVec2::new(55.0, 12.0)), ChainAt { index: Some(2), by: ChainBy::Nearest, area: 0.0, enclosing: 0 });
        assert_eq!(chain_at(&[], DVec2::ZERO), ChainAt { index: None, by: ChainBy::Nearest, area: 0.0, enclosing: 0 });
    }

    #[test]
    fn filters_by_length_by_closing_and_by_count() {
        let (big, hole, wall) = told_apart();
        let chains = [hole, big, wall];
        assert_eq!(filter_chains(&chains, &ChainFilter { min_length: Some(25.0), ..Default::default() }), Filtered { kept: vec![1, 2], left: LeftOut { short: Some(1), ..Default::default() } });
        assert_eq!(filter_chains(&chains, &ChainFilter { closed_only: true, ..Default::default() }), Filtered { kept: vec![0, 1], left: LeftOut { open: Some(1), ..Default::default() } });
        assert_eq!(filter_chains(&chains, &ChainFilter { max_chains: Some(2), ..Default::default() }), Filtered { kept: vec![1, 2], left: LeftOut { beyond: Some(1), ..Default::default() } });
        assert_eq!(filter_chains(&chains, &ChainFilter::default()), Filtered { kept: vec![0, 1, 2], left: LeftOut::default() });
    }

    #[test]
    fn takes_a_noisy_rectangle_to_its_four_corners_within_the_tolerance() {
        let o = outline_of(&rectangle([12.0, -3.0], [52.0, 17.0], 0.01, 3), &OutlineOptions { tolerance: 0.05, ..Default::default() });
        assert!(o.closed);
        assert!(o.points.len() >= 4 && o.points.len() <= 5);
        assert!(o.max_deviation <= 0.05 + 1e-9);
        for corner in [[12.0, -3.0], [52.0, -3.0], [52.0, 17.0], [12.0, 17.0]] {
            let near = o.points.iter().map(|p| (p.x - corner[0]).hypot(p.y - corner[1])).fold(f64::INFINITY, f64::min);
            assert!(near < 0.1);
        }
    }

    #[test]
    fn lays_the_sides_along_the_axes_the_corners_square() {
        let o = outline_of(&rectangle([12.0, -3.0], [52.0, 17.0], 0.01, 3), &OutlineOptions { tolerance: 0.05, snap_deg: Some(1.0), ..Default::default() });
        assert_eq!(o.points.len(), 4);
        assert_eq!(o.snapped, 4);
        for i in 0..4 {
            let (a, b) = (o.points[i], o.points[(i + 1) % 4]);
            assert!(a.x == b.x || a.y == b.y);
        }
        assert!(o.max_deviation < 0.06);
    }

    fn circle() -> Vec<DVec2> {
        let mut c: Vec<DVec2> = (0..721).map(|i| DVec2::new(10.0 * ((i as f64 * PI) / 360.0).cos(), 10.0 * ((i as f64 * PI) / 360.0).sin())).collect();
        c[720] = c[0];
        c
    }

    #[test]
    fn grows_the_tolerance_to_keep_to_max_points() {
        let circle = circle();
        let fine = outline_of(&circle, &OutlineOptions { tolerance: 0.01, ..Default::default() });
        let few = outline_of(&circle, &OutlineOptions { tolerance: 0.01, max_points: Some(12), ..Default::default() });
        assert!(fine.points.len() > 12);
        assert!(few.points.len() <= 12);
        assert!(few.tolerance > 0.01);
    }

    #[test]
    fn keeps_an_open_chain_open_its_ends_kept() {
        let mut chain = run([0.0, 0.0], [10.0, 0.0], 0.1);
        chain.extend(run([10.0, 0.0], [10.0, 5.0], 0.1));
        chain.push(DVec2::new(10.0, 5.0));
        let o = outline_of(&chain, &OutlineOptions { tolerance: 0.05, ..Default::default() });
        assert!(!o.closed);
        assert_eq!(o.points, vec![DVec2::new(0.0, 0.0), DVec2::new(10.0, 0.0), DVec2::new(10.0, 5.0)]);
    }

    #[test]
    fn takes_out_the_corners_it_barely_turns_at_with_simplify_deg_and_keeps_a_curves() {
        // A side, one that runs on 2° off it, and a corner square up.
        let lift = 10.0 * ((2.0 * PI) / 180.0).tan();
        let mut crease = run([0.0, 0.0], [10.0, 0.0], 0.1);
        crease.extend(run([10.0, 0.0], [20.0, lift], 0.1));
        crease.extend(run([20.0, lift], [20.0, 10.0], 0.1));
        crease.push(DVec2::new(20.0, 10.0));
        let sharp = outline_of(&crease, &OutlineOptions { tolerance: 0.05, ..Default::default() });
        assert_eq!(sharp.points.len(), 4);
        assert_eq!(sharp.merged, 0);
        let merged = outline_of(&crease, &OutlineOptions { tolerance: 0.05, simplify_deg: Some(3.0), ..Default::default() });
        assert_eq!(merged.points, vec![DVec2::new(0.0, 0.0), DVec2::new(20.0, lift), DVec2::new(20.0, 10.0)]);
        assert_eq!(merged.merged, 1);
        // The chain now strays from the long side by the crease's height there.
        assert!((merged.max_deviation - lift / 2.0).abs() < 5e-3);
        // A circle's corners turn by more than a degree each: they stay.
        let circle = circle();
        let round = outline_of(&circle, &OutlineOptions { tolerance: 0.01, max_points: Some(24), ..Default::default() });
        assert_eq!(outline_of(&circle, &OutlineOptions { tolerance: 0.01, max_points: Some(24), simplify_deg: Some(1.0), snap_deg: None }).points.len(), round.points.len());
    }

    /// Not a ScanRuler test: a world box misses the cut and says where the cut lies.
    #[test]
    fn says_where_the_cut_lies_when_a_box_misses_it() {
        let frame = SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let chains = vec![vec![DVec2::new(0.0, 0.0), DVec2::new(2.5, 1.0)]];
        let sheet = box_missed(&WithinBox::Sheet(SheetBox { min: DVec2::new(5.0, 5.0), max: DVec2::new(6.0, 6.0) }), &chains, Some(&frame), "the XY plane");
        assert_eq!(sheet, "Nothing in the box [5, 5]…[6, 6] — [u, v] on the XY plane; the cut spans u 0…2.5, v 0…1. A box in world mm goes as within: { world: { min: [x, y, z], max: [x, y, z] } }.");
        let world = WithinBox::World { min: DVec3::new(0.0, 0.0, 1.0), max: DVec3::new(1.0, 1.0, 2.0) };
        let on = within_on_sheet(&world, &frame);
        assert!(on.world.is_some_and(|w| !w.holds(DVec2::new(0.5, 0.5))));
        assert!(box_missed(&world, &chains, Some(&frame), "the XY plane").contains("x 0…2.5, y 0…1, z 0…0 in the world"));
    }
}
