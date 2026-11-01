// SPDX-License-Identifier: AGPL-3.0-only
//! A thread read off the scan. Ported from ScanRuler (`plugins/reverse/core/threadRead.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The scan's points in a thin shell about a bore's or a boss's axis, their radius laid out
//! along the axis, and in that profile the pitch — the spacing the pattern repeats at — the
//! crest and root diameters, which way the helix turns, and the standard size it is nearest.
//! What an agent did with thirty probes along the wall, as one reading.
//!
//! The pitch is the first strong peak of the profile's autocorrelation: a thread is the one
//! feature of a part that repeats along an axis, so the profile agrees with itself shifted by
//! a pitch and by nothing shorter. The profile is taken sector by sector round the axis —
//! along a line up the wall, as a probe would take it — because at one height the radius runs
//! through a whole tooth round the turn, and a profile over all angles at once would be flat.
//! Each sector's autocorrelation is added to the rest. It is read in windows a few turns long
//! along the axis, and the pitch is the one most windows agree on: a set screw's hole is
//! threaded half its length, a chamfer opens it, and over the whole length the plain half and
//! the rims drown the teeth. The hand is read on the helix itself: a point at angle θ about
//! the axis lies a pitch × θ / 2π further along it than its neighbour at θ = 0, one way for a
//! right-hand thread and the other for a left — the phase of every point under each
//! hypothesis is gathered, and the hand whose phases agree is the one.

use std::cmp::Ordering;
use std::f64::consts::PI;

use glam::DVec3;
use serde::Serialize;

use crate::round::{js_round, round3, round4};
use crate::thread::{minor_diameter, thread_depth, thread_designation, thread_sizes, ThreadStandard, THREAD_STANDARDS};

/// The points a thread is read in, and the shell about the axis they are taken from.
#[derive(Debug, Clone, Copy)]
pub struct ThreadReadInput<'a> {
    /// The points about the axis.
    pub points: &'a [DVec3],
    /// Their outward normals, one a point; any other count is taken for none.
    pub normals: &'a [DVec3],
    pub origin: DVec3,
    pub dir: DVec3,
    /// The nominal radius — the fitted cylinder's, the face's.
    pub radius: f64,
    /// How far either side of it a point is still the thread's, mm.
    pub band: f64,
    /// How far along the axis from the origin, each way, mm.
    pub half_length: f64,
}

/// The radius of the points read: its mean, and its 5th and 95th percentile.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ReadRadius {
    pub mean: f64,
    pub low: f64,
    pub high: f64,
}

/// A point of the radius along the axis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ProfilePoint {
    pub z: f64,
    pub r: f64,
}

/// A window along the axis the pitch is read in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ThreadWindow {
    pub from: f64,
    pub to: f64,
    pub pitch: Option<f64>,
    pub strength: f64,
    pub contrast: Option<f64>,
}

/// A thread as the scan has it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadReading {
    /// How many points were read, of those handed in.
    pub points: usize,
    /// How many more lay in the shell but faced along the axis — a chamfer, a rim — and were
    /// left out.
    pub off_wall: usize,
    /// Where along the axis they lie, mm from the origin.
    pub span: [f64; 2],
    /// The radius: its mean, and its 5th and 95th percentile — a nut's crests and root, a
    /// bolt's root and crests — each point's taken about a straight line through its
    /// sector's along the axis, so an axis a little off or aslant reads no deeper teeth.
    pub radius: ReadRadius,
    pub minor: f64,
    pub major: f64,
    /// How deep the pattern cuts, high − low, mm.
    pub depth: f64,
    /// A bore (the normals face the axis), a boss, or not clear.
    pub internal: Option<bool>,
    /// The pitch, mm — the one most windows along the axis agree on; `None` where the profile
    /// does not repeat.
    pub pitch: Option<f64>,
    /// How well the profile agrees with itself a pitch along, 0 to 1 — the agreeing windows'
    /// median.
    pub strength: f64,
    /// The tooth's height, mm: the profile folded at the pitch, each sector of the wall on its
    /// own, the folded profile's peak to peak — what repeats; the agreeing windows' median,
    /// `None` with no pitch.
    pub contrast: Option<f64>,
    /// The scan's scatter on the wall, mm: the points' radii about their sector's median at
    /// their height (1.48 × the median absolute deviation).
    pub noise: f64,
    pub left_handed: Option<bool>,
    /// How much better the hand found fits than the other, 0 to 1.
    pub hand_confidence: f64,
    /// Each window along the axis, mm from the origin, and what it read.
    pub windows: Vec<ThreadWindow>,
    /// How many of the windows read the pitch — 0 where the whole span read one no window
    /// could see, a pitch too coarse for them.
    pub agree: usize,
    /// Where along the axis the agreeing windows lie — where the thread is; the radius, minor
    /// and major are read there. `None` with no pitch.
    pub threaded: Option<[f64; 2]>,
    /// The radius along the axis, thinned, for a look.
    pub profile: Vec<ProfilePoint>,
}

/// A standard size the reading may be.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadCandidate {
    pub standard: ThreadStandard,
    pub size: String,
    pub designation: String,
    pub diameter: f64,
    pub pitch: f64,
    /// How far the reading is off this size: the pitch's share and the diameter's.
    pub pitch_off: f64,
    pub diameter_off: f64,
    pub score: f64,
}

/// The shortest pitch looked for and the longest, mm.
const PITCH_MIN: f64 = 0.15;
const PITCH_MAX: f64 = 8.0;
/// How many bins the profile is read in, at most, and the finest bin.
const PROFILE_BINS: f64 = 400.0;
const BIN_MIN: f64 = 0.01;
/// The longest stretch of a window a sector may leave empty, of its length.
const GAP_SHARE: f64 = 0.25;
/// How many sectors round the axis the profile is read in: each a line up the wall, thirty
/// degrees wide.
const SECTORS: usize = 12;
/// The least agreement a pitch is reported at.
const STRENGTH_MIN: f64 = 0.25;
/// How many times the scan's noise the teeth must stand to be a thread: below, what repeats
/// is the scan's own ripple — layer lines, the scanner's stripes — on a plain wall.
const CONTRAST_MIN: f64 = 3.0;
/// A peak within this share of the best counts as the pitch when it comes first — the
/// pattern repeats at its pitch before it repeats at two.
const FIRST_PEAK_SHARE: f64 = 0.75;
/// How far along a window's profile the lag is searched, of its length: past half, so a
/// window holding two turns sees their pitch, and short of where a single bump, a step or a
/// slanted end repeats as well as a thread does.
const LAG_SHARE: f64 = 0.6;
/// A point whose normal leans further along the axis than this (|n·axis|) faces a chamfer, a
/// rim or a floor, not the wall.
const WALL_FACING: f64 = 0.8;
/// The windows the pitch is read in along the axis: this long, or the radius when that is
/// more — a coarse thread goes with a wide one — a fifth of that apart, and this many at
/// most.
const WINDOW_MM: f64 = 2.5;
const WINDOW_STEP: f64 = 0.2;
const WINDOWS_MAX: f64 = 16.0;
/// How many points a sector needs for its own line along the axis.
const SECTOR_LINE_MIN: usize = 8;
/// Windows whose pitches lie within this share of one another agree.
const AGREE_SHARE: f64 = 0.06;
/// How many profile points an answer carries.
const PROFILE_OUT: f64 = 200.0;
/// How far off a size a reading may be and still be suggested.
const SUGGEST_PITCH: f64 = 0.12;
const SUGGEST_DIAMETER: f64 = 0.08;

/// TypeScript's `normalize`: the unit vector, or `None` for one of no length.
fn normalize(v: DVec3) -> Option<DVec3> {
    let l = v.length();
    if l < 1e-12 {
        None
    } else {
        Some(v / l)
    }
}

/// Ascending, as JavaScript's `sort((a, b) => a - b)` has it.
fn sort_numbers(xs: &mut [f64]) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
}

/// The median: the middle value, or the mean of the middle two. NaN for none.
fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    sort_numbers(&mut s);
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n >> 1]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// The thread in a shell of points about an axis; `None` for an axis of no direction or fewer
/// than twenty points in the shell.
pub fn read_thread(input: &ThreadReadInput<'_>) -> Option<ThreadReading> {
    let d = normalize(input.dir)?;
    let o = input.origin;
    let u0 = if d.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let u = normalize(d.cross(u0))?;
    let v = d.cross(u);
    let with_normals = input.normals.len() == input.points.len();
    let mut zs: Vec<f64> = Vec::new();
    let mut rs: Vec<f64> = Vec::new();
    let mut ths: Vec<f64> = Vec::new();
    let mut facing = 0.0;
    let mut off_wall = 0usize;
    for (i, &p) in input.points.iter().enumerate() {
        let q = p - o;
        let z = q.dot(d);
        if z.abs() > input.half_length {
            continue;
        }
        let radial = q - d * z;
        let r = radial.length();
        if (r - input.radius).abs() > input.band {
            continue;
        }
        // A chamfer, a rim, a floor: a point facing along the axis is no part of the wall the
        // teeth stand on.
        let normal = if with_normals { input.normals.get(i).copied() } else { None };
        if let Some(n) = normal {
            if n.dot(d).abs() > WALL_FACING {
                off_wall += 1;
                continue;
            }
        }
        zs.push(z);
        rs.push(r);
        ths.push(radial.dot(v).atan2(radial.dot(u)));
        if let Some(n) = normal {
            if r > 1e-9 {
                facing += n.dot(radial) / r;
            }
        }
    }
    let m = zs.len();
    if m < 20 {
        return None;
    }
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for &z in &zs {
        if z < z_min {
            z_min = z;
        }
        if z > z_max {
            z_max = z;
        }
    }
    let span = z_max - z_min;
    let facing_share = facing / m as f64;
    let internal = if facing_share < -0.3 {
        Some(true)
    } else if facing_share > 0.3 {
        Some(false)
    } else {
        None
    };
    let sectors: Vec<usize> = ths.iter().map(|th| (SECTORS - 1).min((((th + PI) / (2.0 * PI)) * SECTORS as f64).floor().max(0.0) as usize)).collect();

    // The pitch, window by window along the axis: a thread that runs only part of the way —
    // the far half of a set-screw hole plain, a chamfer at its mouth — repeats in the windows
    // on it, and the pitch is the one most of them agree on, not the strongest single one.
    let window_mm = WINDOW_MM.max(input.radius);
    let count = if span <= window_mm * 1.2 { 1 } else { WINDOWS_MAX.min(((span - window_mm) / (window_mm * WINDOW_STEP)).ceil() + 1.0) as usize };
    let window_length = if count == 1 { span } else { window_mm };
    let window_step = if count == 1 { 0.0 } else { (span - window_length) / (count - 1) as f64 };
    let mut reads: Vec<ThreadWindow> = Vec::with_capacity(count);
    for k in 0..count {
        let from = z_min + k as f64 * window_step;
        let to = if count == 1 { z_max } else { from + window_length };
        reads.push(window_read(&zs, &rs, &sectors, from, to));
    }
    let mut agreeing = agree_on(&reads);
    let agree = agreeing.len();
    // Windows too short to see a coarse pitch: the whole span's own read.
    if agree == 0 && count > 1 {
        let whole = window_read(&zs, &rs, &sectors, z_min, z_max);
        if whole.pitch.is_some() {
            agreeing = vec![whole];
        }
    }
    let pitch = if agreeing.is_empty() { None } else { Some(median(&agreeing.iter().filter_map(|w| w.pitch).collect::<Vec<_>>())) };
    let strength = if agreeing.is_empty() {
        reads.iter().map(|w| w.strength).fold(0.0, f64::max)
    } else {
        median(&agreeing.iter().map(|w| w.strength).collect::<Vec<_>>())
    };
    let contrast = if agreeing.is_empty() { None } else { Some(median(&agreeing.iter().map(|w| w.contrast.unwrap_or(0.0)).collect::<Vec<_>>())) };
    let threaded: Option<[f64; 2]> = if agreeing.is_empty() {
        None
    } else {
        Some([agreeing.iter().map(|w| w.from).fold(f64::INFINITY, f64::min), agreeing.iter().map(|w| w.to).fold(f64::NEG_INFINITY, f64::max)])
    };

    // The crests and the root where the thread is (over the whole span where there is none),
    // each point's radius against a straight line through its sector's along the axis: an
    // axis a little off the scan's — a hole modelled half a millimetre aside, a few degrees
    // over — moves each sector's radius as a whole, and pooled that reads as depth. The
    // radius is the sectors' median; a sector cut short (a cross bore, a gap in the scan)
    // keeps its level but takes no slope.
    let on: Vec<usize> = (0..m).filter(|&i| threaded.is_none_or(|t| zs[i] >= t[0] && zs[i] <= t[1])).collect();
    let reach = threaded.map_or(span, |t| t[1] - t[0]);
    let mut by_sector: Vec<Vec<usize>> = vec![Vec::new(); SECTORS];
    for &i in &on {
        by_sector[sectors[i]].push(i);
    }
    let mut lines = vec![f64::NAN; m];
    let mut levels: Vec<f64> = Vec::new();
    for list in &by_sector {
        if list.len() < SECTOR_LINE_MIN {
            continue;
        }
        let (mut sz, mut sr, mut lo, mut hi) = (0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY);
        for &i in list {
            sz += zs[i];
            sr += rs[i];
            lo = lo.min(zs[i]);
            hi = hi.max(zs[i]);
        }
        let zm = sz / list.len() as f64;
        let rm = sr / list.len() as f64;
        let (mut szz, mut szr) = (0.0, 0.0);
        for &i in list {
            szz += (zs[i] - zm).powi(2);
            szr += (zs[i] - zm) * (rs[i] - rm);
        }
        let slope = if hi - lo >= reach / 2.0 && szz > 1e-12 { szr / szz } else { 0.0 };
        for &i in list {
            lines[i] = rm + slope * (zs[i] - zm);
        }
        levels.push(rm);
    }
    let centre = if levels.is_empty() { median(&on.iter().map(|&i| rs[i]).collect::<Vec<_>>()) } else { median(&levels) };
    let mut dev = vec![0.0; m];
    for &i in &on {
        dev[i] = rs[i] - if lines[i].is_nan() { centre } else { lines[i] };
    }
    let mut sorted_dev: Vec<f64> = on.iter().map(|&i| dev[i]).collect();
    sort_numbers(&mut sorted_dev);
    let pct = |p: f64| -> f64 {
        if sorted_dev.is_empty() {
            return f64::NAN;
        }
        sorted_dev[(sorted_dev.len() - 1).min((p * sorted_dev.len() as f64).floor() as usize)]
    };
    let low = centre + pct(0.05);
    let high = centre + pct(0.95);
    let mean = centre;

    // The hand: under which the phases of the crests agree — the points at the smallest radii
    // and the largest, which lie on the helix's crest and its root; a point halfway up a flank
    // lies on either hand's helix.
    let mut left_handed: Option<bool> = None;
    let mut hand_confidence = 0.0;
    if let Some(pitch) = pitch {
        let low_cut = pct(0.15);
        let high_cut = pct(0.85);
        let agreement = |sign: f64| -> f64 {
            let (mut cx, mut cy, mut hx, mut hy, mut nl, mut nh) = (0.0, 0.0, 0.0, 0.0, 0usize, 0usize);
            for &i in &on {
                let r = dev[i];
                if r > low_cut && r < high_cut {
                    continue;
                }
                let phase = ((zs[i] - (sign * pitch * ths[i]) / (2.0 * PI)) / pitch) * 2.0 * PI;
                if r <= low_cut {
                    cx += phase.cos();
                    cy += phase.sin();
                    nl += 1;
                } else {
                    hx += phase.cos();
                    hy += phase.sin();
                    nh += 1;
                }
            }
            (if nl > 0 { cx.hypot(cy) / nl as f64 } else { 0.0 }) + (if nh > 0 { hx.hypot(hy) / nh as f64 } else { 0.0 })
        };
        let right = agreement(1.0);
        let left = agreement(-1.0);
        if right.max(left) > 0.05 {
            left_handed = Some(left > right);
            hand_confidence = (left - right).abs() / left.max(right);
        }
    }

    // The noise: each point against its sector's median at its height, over the whole span;
    // and the whole-turn profile, for a look.
    let bin = BIN_MIN.max(span / PROFILE_BINS);
    let bins = ((span / bin).ceil() as usize).max(1);
    let bin_of = |z: f64| (bins - 1).min(((z - z_min) / bin).floor().max(0.0) as usize);
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); bins];
    let mut sector_buckets: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); bins]; SECTORS];
    for i in 0..m {
        buckets[bin_of(zs[i])].push(rs[i]);
        sector_buckets[sectors[i]][bin_of(zs[i])].push(rs[i]);
    }
    for list in buckets.iter_mut().chain(sector_buckets.iter_mut().flatten()) {
        sort_numbers(list);
    }
    let mut deviations: Vec<f64> = Vec::new();
    for i in 0..m {
        let list = &sector_buckets[sectors[i]][bin_of(zs[i])];
        if list.len() < 2 {
            continue;
        }
        deviations.push((rs[i] - list[list.len() / 2]).abs());
    }
    sort_numbers(&mut deviations);
    let noise = if deviations.is_empty() { 0.0 } else { 1.4826 * deviations[deviations.len() >> 1] };

    let stride = ((bins as f64 / PROFILE_OUT).ceil() as usize).max(1);
    let mut out: Vec<ProfilePoint> = Vec::new();
    for b in (0..bins).step_by(stride) {
        let list = &buckets[b];
        if list.is_empty() {
            continue;
        }
        out.push(ProfilePoint { z: round3(z_min + (b as f64 + 0.5) * bin), r: round3(list[list.len() / 2]) });
    }
    Some(ThreadReading {
        points: m,
        off_wall,
        span: [round3(z_min), round3(z_max)],
        radius: ReadRadius { mean: round3(mean), low: round3(low), high: round3(high) },
        minor: round3(2.0 * low),
        major: round3(2.0 * high),
        depth: round3(high - low),
        internal,
        pitch: pitch.map(round4),
        strength: round3(strength),
        contrast: contrast.map(round4),
        noise: round4(noise),
        left_handed,
        hand_confidence: round3(hand_confidence),
        windows: reads
            .iter()
            .map(|w| ThreadWindow { from: round3(w.from), to: round3(w.to), pitch: w.pitch.map(round4), strength: round3(w.strength), contrast: w.contrast.map(round4) })
            .collect(),
        agree,
        threaded: threaded.map(|t| [round3(t[0]), round3(t[1])]),
        profile: out,
    })
}

/// The pitch in the points between `from` and `to` along the axis: in each sector round it,
/// the median radius in bins along it, the empty bins bridged, a straight trend taken out —
/// a taper is not a thread; the sectors' autocorrelations added, and the first strong peak
/// past the zero lag's own shoulder; the teeth's height folded at it.
fn window_read(zs: &[f64], rs: &[f64], sectors: &[usize], from: f64, to: f64) -> ThreadWindow {
    let none = ThreadWindow { from, to, pitch: None, strength: 0.0, contrast: None };
    let length = to - from;
    let bin = BIN_MIN.max(length / PROFILE_BINS);
    let bins = ((length / bin).ceil() as usize).max(1);
    let mut sector_buckets: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); bins]; SECTORS];
    for i in 0..zs.len() {
        if zs[i] < from || zs[i] > to {
            continue;
        }
        sector_buckets[sectors[i]][(bins - 1).min(((zs[i] - from) / bin).floor().max(0.0) as usize)].push(rs[i]);
    }
    let profile_of = |lists: &mut Vec<Vec<f64>>| -> Option<Vec<f64>> {
        let mut out = vec![f64::NAN; bins];
        let mut filled = 0usize;
        let mut gap = 0usize;
        let mut longest = 0usize;
        let mut last = f64::NAN;
        for b in 0..bins {
            let list = &mut lists[b];
            if !list.is_empty() {
                sort_numbers(list);
                last = list[list.len() / 2];
                filled += 1;
                gap = 0;
            } else {
                gap += 1;
                longest = longest.max(gap);
            }
            out[b] = last;
        }
        // A sector the wall leaves for a stretch — a cross bore, a gap in the scan — is no
        // line up the wall; one a coarse scan fills only row by row still is.
        if filled < SECTOR_LINE_MIN || longest as f64 > bins as f64 * GAP_SHARE {
            return None;
        }
        // Leading empty bins take the first value there is.
        let first = out.iter().copied().find(|x| !x.is_nan()).unwrap_or(f64::NAN);
        for x in out.iter_mut() {
            if !x.is_nan() {
                break;
            }
            *x = first;
        }
        detrend(&mut out);
        Some(out)
    };
    let profiles: Vec<Vec<f64>> = sector_buckets.iter_mut().filter_map(profile_of).collect();
    let lag_min = 2.0f64.max(js_round(PITCH_MIN / bin)) as usize;
    let lag_max = ((bins as f64 * LAG_SHARE).floor()).min(js_round(PITCH_MAX / bin)).max(0.0) as usize;
    if !(lag_max > lag_min + 2) || profiles.is_empty() {
        return none;
    }
    let mut ac = vec![0.0; lag_max + 1];
    let mut weighed = 0usize;
    for profile in &profiles {
        let norm: f64 = profile.iter().map(|x| x * x).sum();
        if !(norm > 0.0) {
            continue;
        }
        weighed += 1;
        for lag in lag_min..=lag_max {
            let mut s = 0.0;
            for i in 0..bins.saturating_sub(lag) {
                s += profile[i] * profile[i + lag];
            }
            ac[lag] += (s / norm) * (bins as f64 / (bins - lag) as f64);
        }
    }
    if weighed == 0 {
        return none;
    }
    for value in ac.iter_mut().take(lag_max + 1).skip(lag_min) {
        *value /= weighed as f64;
    }
    // Past the first minimum: a profile still agrees with itself a short way along — a 1 mm
    // thread correlates 0.5 or more at 0.15 mm — and that shoulder is no repeat.
    let mut start = lag_min;
    while start < lag_max && ac[start + 1] <= ac[start] {
        start += 1;
    }
    let best = ac[start..=lag_max].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weak = ThreadWindow { strength: best.clamp(0.0, 1.0), ..none };
    if !(best >= STRENGTH_MIN) {
        return weak;
    }
    // The first local maximum worth most of the best: the pitch, not twice it.
    let Some(at) = ((lag_min + 1).max(start)..lag_max).find(|&lag| ac[lag] >= ac[lag - 1] && ac[lag] >= ac[lag + 1] && ac[lag] >= FIRST_PEAK_SHARE * best) else {
        return weak;
    };
    // The peak's top by a parabola through its neighbours.
    let (y0, y1, y2) = (ac[at - 1], ac[at], ac[at + 1]);
    let denom = y0 - 2.0 * y1 + y2;
    let shift = if denom.abs() > 1e-12 { ((0.5 * (y0 - y2)) / denom).clamp(-1.0, 1.0) } else { 0.0 };
    let pitch = (at as f64 + shift) * bin;
    // The contrast: each sector's profile folded at the pitch — its own phase, the helix
    // moving it from one sector to the next — and the folded profile's peak to peak, the
    // median over the sectors.
    let phases = 4.0f64.max(24.0f64.min(js_round(pitch / bin))) as usize;
    let mut heights: Vec<f64> = Vec::new();
    for profile in &profiles {
        let mut sum = vec![0.0; phases];
        let mut count = vec![0.0; phases];
        for (b, &value) in profile.iter().enumerate() {
            let z = (b as f64 + 0.5) * bin;
            let k = (phases - 1).min((((z / pitch) % 1.0) * phases as f64).floor().max(0.0) as usize);
            sum[k] += value;
            count[k] += 1.0;
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for k in 0..phases {
            if count[k] == 0.0 {
                continue;
            }
            let value = sum[k] / count[k];
            lo = lo.min(value);
            hi = hi.max(value);
        }
        if hi > lo {
            heights.push(hi - lo);
        }
    }
    sort_numbers(&mut heights);
    ThreadWindow { from, to, pitch: Some(pitch), strength: y1.clamp(0.0, 1.0), contrast: Some(if heights.is_empty() { 0.0 } else { heights[heights.len() >> 1] }) }
}

/// The windows that read the pitch most of them agree on — within AGREE_SHARE of one another;
/// between as many, the stronger. None where no window reads a pitch.
fn agree_on(reads: &[ThreadWindow]) -> Vec<ThreadWindow> {
    let read: Vec<(ThreadWindow, f64)> = reads.iter().filter_map(|w| w.pitch.map(|p| (*w, p))).collect();
    let mut best: Vec<ThreadWindow> = Vec::new();
    let mut best_strength = f64::NEG_INFINITY;
    for &(_, wp) in &read {
        let near: Vec<ThreadWindow> = read.iter().filter(|(_, xp)| (xp - wp).abs() <= AGREE_SHARE * wp).map(|(x, _)| *x).collect();
        let strength: f64 = near.iter().map(|x| x.strength).sum();
        if near.len() > best.len() || (near.len() == best.len() && strength > best_strength) {
            best = near;
            best_strength = strength;
        }
    }
    best
}

/// Whether a reading is a thread: a pitch found, and teeth standing well clear of the scan's
/// noise.
pub fn is_thread(r: &ThreadReading) -> bool {
    matches!((r.pitch, r.contrast), (Some(_), Some(contrast)) if contrast >= CONTRAST_MIN * r.noise)
}

/// The standard sizes a reading is nearest, and the one near enough to suggest.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThreadMatch {
    /// Best first — three at most.
    pub candidates: Vec<ThreadCandidate>,
    pub suggested: Option<ThreadCandidate>,
}

/// The standard sizes a reading is nearest, best first — three at most — and the one near
/// enough to suggest, if any. A bore is matched on its crests, the minor diameter, which a
/// scan resolves; a boss on its own crests, the major. A boss read at an ISO coarse pitch is
/// as near the printing form's size at that pitch as the metric one — the same major diameter
/// — and the standard listed first takes the tie: metric, before print. A bore tells them
/// apart by the minor.
pub fn match_thread(reading: &ThreadReading, standards: &[ThreadStandard]) -> ThreadMatch {
    let Some(read_pitch) = reading.pitch else { return ThreadMatch { candidates: Vec::new(), suggested: None } };
    let mut all: Vec<ThreadCandidate> = Vec::new();
    for &standard in standards {
        for size in thread_sizes(standard) {
            for &pitch in &size.pitches {
                let pitch_off = (pitch - read_pitch).abs() / pitch;
                let off_major = (size.diameter - reading.major).abs() / size.diameter;
                let off_minor = (minor_diameter(size.diameter, pitch, standard) - reading.minor).abs() / size.diameter;
                let diameter_off = match reading.internal {
                    Some(true) => off_minor,
                    Some(false) => off_major,
                    None => off_major.min(off_minor),
                };
                all.push(ThreadCandidate {
                    standard,
                    size: size.label.clone(),
                    designation: thread_designation(standard, size.diameter, pitch).unwrap_or_else(|| size.label.clone()),
                    diameter: size.diameter,
                    pitch,
                    pitch_off: round3(pitch_off),
                    diameter_off: round3(diameter_off),
                    score: round3(2.0 * pitch_off + diameter_off),
                });
            }
        }
    }
    // A stable sort, as JavaScript's is: between equal scores the standard listed first.
    all.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));
    all.truncate(3);
    let suggested = all.first().filter(|best| best.pitch_off <= SUGGEST_PITCH && best.diameter_off <= SUGGEST_DIAMETER).cloned();
    ThreadMatch { candidates: all, suggested }
}

/// The thread's depth a standard's basic profile would cut at a pitch, for the answer to set
/// the reading against.
pub fn basic_depth(pitch: f64, standard: ThreadStandard) -> f64 {
    round3(thread_depth(pitch, standard))
}

/// Every standard, for a reading matched against any.
pub const ALL_STANDARDS: [ThreadStandard; 4] = THREAD_STANDARDS;

/// The least-squares line through the values taken out of them.
fn detrend(values: &mut [f64]) {
    let n = values.len();
    if n < 2 {
        return;
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let nf = n as f64;
    let denom = nf * sxx - sx * sx;
    let slope = if denom > 0.0 { (nf * sxy - sx * sy) / denom } else { 0.0 };
    let mean = sy / nf;
    let mid = sx / nf;
    for (i, y) in values.iter_mut().enumerate() {
        *y -= mean + slope * (i as f64 - mid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ThreadStandard::{Metric, Unc, Unf};

    /// The tests' own generator: a linear congruential one, uniform in [−0.5, 0.5).
    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(self.0) / 4_294_967_296.0 - 0.5
        }
    }

    /// A point as a Float32Array holds it.
    fn f32v(x: f64, y: f64, z: f64) -> DVec3 {
        DVec3::new(x as f32 as f64, y as f32 as f64, z as f32 as f64)
    }

    struct Wall {
        points: Vec<DVec3>,
        normals: Vec<DVec3>,
    }

    struct WallOpts {
        r_low: f64,
        r_high: f64,
        pitch: f64,
        length: f64,
        internal: bool,
        left_handed: bool,
        angles: usize,
        rows: usize,
        noise: f64,
    }

    impl Default for WallOpts {
        fn default() -> Self {
            WallOpts { r_low: 0.0, r_high: 0.0, pitch: 1.0, length: 1.0, internal: false, left_handed: false, angles: 90, rows: 400, noise: 0.0 }
        }
    }

    /// Points on a threaded wall about the z axis: the radius a triangle wave of the helix's
    /// phase between `r_low` and `r_high`, `angles` round by `rows` along, over `length` mm. A
    /// bore's normals face the axis.
    fn threaded_wall(o: WallOpts) -> Wall {
        let mut points = Vec::new();
        let mut normals = Vec::new();
        let mut rand = Lcg(7);
        let hand = if o.left_handed { -1.0 } else { 1.0 };
        for j in 0..o.rows {
            let z = -o.length / 2.0 + (o.length * j as f64) / (o.rows - 1) as f64;
            for k in 0..o.angles {
                let th = (k as f64 / o.angles as f64) * 2.0 * PI;
                let phase = (z - (hand * o.pitch * th) / (2.0 * PI)) / o.pitch;
                let t = phase - phase.floor();
                let tri = if t < 0.5 { t * 2.0 } else { 2.0 - t * 2.0 };
                let r = o.r_low + (o.r_high - o.r_low) * tri + o.noise * rand.next();
                points.push(f32v(r * th.cos(), r * th.sin(), z));
                let s = if o.internal { -1.0 } else { 1.0 };
                normals.push(f32v(s * th.cos(), s * th.sin(), 0.0));
            }
        }
        Wall { points, normals }
    }

    fn input<'a>(wall: &'a Wall, origin: DVec3, dir: DVec3, radius: f64, band: f64, half_length: f64) -> ThreadReadInput<'a> {
        ThreadReadInput { points: &wall.points, normals: &wall.normals, origin, dir, radius, band, half_length }
    }

    #[test]
    fn reads_an_m6x1_bore_pitch_crests_root_right_hand_and_the_size() {
        let wall = threaded_wall(WallOpts { r_low: 2.459, r_high: 3.0, pitch: 1.0, length: 8.0, internal: true, noise: 0.02, ..Default::default() });
        let r = read_thread(&input(&wall, DVec3::ZERO, DVec3::Z, 2.75, 0.6, 5.0)).expect("a reading");
        assert_eq!(r.points, wall.points.len());
        assert_eq!(r.internal, Some(true));
        let pitch = r.pitch.unwrap_or(f64::NAN);
        assert!((pitch - 1.0).abs() < 0.05);
        assert!((pitch - 1.0).abs() < 0.03);
        assert!(r.strength > 0.5);
        assert!((r.minor - 4.92).abs() < 0.5, "minor {}", r.minor);
        assert!((r.major - 6.0).abs() < 0.5, "major {}", r.major);
        assert_eq!(r.left_handed, Some(false));
        assert!(r.hand_confidence > 0.3);
        assert!(r.profile.len() > 50);
        let found = match_thread(&r, &[Metric]);
        assert_eq!(found.suggested.map(|s| s.designation).as_deref(), Some("M6×1"));
        assert_eq!(found.candidates[0].size, "M6");
    }

    #[test]
    fn reads_a_left_hand_quarter_20_boss_and_a_plain_cylinder_reads_no_pitch() {
        let wall = threaded_wall(WallOpts { r_low: 2.5, r_high: 3.175, pitch: 1.27, length: 12.0, internal: false, left_handed: true, noise: 0.01, ..Default::default() });
        let r = read_thread(&input(&wall, DVec3::ZERO, DVec3::Z, 2.9, 0.7, 6.0)).expect("a reading");
        assert_eq!(r.internal, Some(false));
        assert!((r.pitch.unwrap_or(f64::NAN) - 1.27).abs() < 0.04);
        assert_eq!(r.left_handed, Some(true));
        let suggested = match_thread(&r, &[Metric, Unc, Unf]).suggested.expect("a suggestion");
        assert_eq!(suggested.standard, Unc);
        assert_eq!(suggested.size, "1/4");
        let plain = threaded_wall(WallOpts { r_low: 3.0, r_high: 3.0, pitch: 1.0, length: 10.0, internal: true, noise: 0.01, ..Default::default() });
        let p = read_thread(&input(&plain, DVec3::ZERO, DVec3::Z, 3.0, 0.5, 5.0)).expect("a reading");
        assert_eq!(p.pitch, None);
        assert!(match_thread(&p, &[Metric]).candidates.is_empty());
    }

    #[test]
    fn a_plain_bore_whose_wall_ripples_within_the_scans_noise_repeats_but_is_no_thread() {
        // Rings 0.2 mm apart, 0.024 mm peak to peak — a print's layer lines, the scanner's
        // stripes — under noise of ±0.08 mm, densely scanned.
        let mut points = Vec::with_capacity(360 * 400);
        let mut normals = Vec::with_capacity(360 * 400);
        let mut rand = Lcg(11);
        for j in 0..400 {
            let z = -5.0 + (10.0 * j as f64) / 399.0;
            for k in 0..360 {
                let th = (k as f64 / 360.0) * 2.0 * PI;
                let r = 3.0 + 0.012 * ((2.0 * PI * z) / 0.2).sin() + 0.16 * rand.next();
                points.push(f32v(r * th.cos(), r * th.sin(), z));
                normals.push(f32v(-th.cos(), -th.sin(), 0.0));
            }
        }
        let rippled = Wall { points, normals };
        let plain = read_thread(&input(&rippled, DVec3::ZERO, DVec3::Z, 3.0, 0.5, 5.0)).expect("a reading");
        // The ripple repeats, which alone used to read as a thread.
        assert!(plain.pitch.is_some(), "the ripple repeats, which alone used to read as a thread");
        assert!(plain.contrast.unwrap_or(f64::NAN) < 3.0 * plain.noise);
        assert!(!is_thread(&plain));
        let m6 = threaded_wall(WallOpts { r_low: 2.459, r_high: 3.0, pitch: 1.0, length: 8.0, internal: true, noise: 0.08, ..Default::default() });
        let real = read_thread(&input(&m6, DVec3::ZERO, DVec3::Z, 2.75, 0.6, 5.0)).expect("a reading");
        let contrast = real.contrast.unwrap_or(f64::NAN);
        assert!(contrast > 0.3);
        assert!(contrast > 10.0 * real.noise);
        assert!(is_thread(&real));
    }

    #[test]
    fn reads_a_short_thread_whose_length_holds_a_chamfer_a_plain_half_and_a_floor() {
        // A set screw's M6×1 hole: a 45° chamfer at the mouth (z −0.6 to 0), threaded 3 mm,
        // then plain 2.5 mm to a flat floor — as the mirror holder's: over the whole length
        // the plain half and the rims drown the teeth.
        let mut points = Vec::new();
        let mut normals = Vec::new();
        let mut rand = Lcg(5);
        let s45 = std::f64::consts::FRAC_1_SQRT_2;
        let mut z = -0.6;
        while z <= 5.5 {
            for k in 0..90 {
                let th = (k as f64 / 90.0) * 2.0 * PI;
                let (c, s) = (th.cos(), th.sin());
                let mut r;
                let mut n = [-c, -s, 0.0];
                if z < 0.0 {
                    r = 3.0 - z;
                    n = [-c * s45, -s * s45, -s45];
                } else if z <= 3.0 {
                    let t = ((z - th / (2.0 * PI)) % 1.0 + 1.0) % 1.0;
                    r = 2.459 + 0.541 * if t < 0.5 { t * 2.0 } else { 2.0 - t * 2.0 };
                } else {
                    r = 2.459;
                }
                r += 0.01 * rand.next();
                points.push(f32v(r * c, r * s, z));
                normals.push(f32v(n[0], n[1], n[2]));
            }
            z += 0.02;
        }
        // The floor, facing up the hole.
        let mut r = 0.2;
        while r <= 2.459 {
            for k in 0..90 {
                let th = (k as f64 / 90.0) * 2.0 * PI;
                points.push(f32v(r * th.cos(), r * th.sin(), 5.5));
                normals.push(f32v(0.0, 0.0, -1.0));
            }
            r += 0.03;
        }
        let hole = Wall { points, normals };
        let read = read_thread(&input(&hole, DVec3::new(0.0, 0.0, 2.45), DVec3::Z, 2.75, 0.6, 3.1)).expect("a reading");
        assert!(read.off_wall > 0, "the floor faces along the axis");
        assert!(read.windows.len() > 2);
        assert!(read.agree >= 2);
        assert!((read.pitch.unwrap_or(f64::NAN) - 1.0).abs() < 0.05);
        assert!(is_thread(&read));
        // Where the agreeing windows lie: from the mouth over the thread (to 0.55 along the
        // axis), short of the plain half's end; the windows there read nothing.
        let threaded = read.threaded.expect("where the thread is");
        assert!(threaded[0] < -2.0);
        assert!(threaded[1] < 2.0);
        assert_eq!(read.windows.last().and_then(|w| w.pitch), None);
        assert_eq!(read.left_handed, Some(false));
        assert_eq!(match_thread(&read, &[Metric]).suggested.map(|s| s.designation).as_deref(), Some("M6×1"));
    }

    #[test]
    fn reads_along_a_tilted_axis_anywhere_in_space_and_refuses_too_few_points() {
        let wall = threaded_wall(WallOpts { r_low: 1.2, r_high: 1.5, pitch: 0.5, length: 6.0, internal: true, rows: 300, angles: 60, ..Default::default() });
        // Turned about x by 30° and moved.
        let (c, s) = ((PI / 6.0).cos(), (PI / 6.0).sin());
        let pts: Vec<DVec3> = wall.points.iter().map(|p| f32v(p.x + 10.0, c * p.y - s * p.z + 20.0, s * p.y + c * p.z + 30.0)).collect();
        let nrm: Vec<DVec3> = wall.normals.iter().map(|n| f32v(n.x, c * n.y - s * n.z, s * n.y + c * n.z)).collect();
        let turned = Wall { points: pts, normals: nrm };
        let origin = DVec3::new(10.0, 20.0, 30.0);
        let dir = DVec3::new(0.0, -s, c);
        let r = read_thread(&input(&turned, origin, dir, 1.35, 0.4, 4.0)).expect("a reading");
        assert!((r.pitch.unwrap_or(f64::NAN) - 0.5).abs() < 0.03);
        assert_eq!(match_thread(&r, &[Metric]).suggested.map(|s| s.designation).as_deref(), Some("M3×0.5"));
        let few = Wall { points: turned.points[..10].to_vec(), normals: turned.normals[..10].to_vec() };
        assert!(read_thread(&input(&few, origin, dir, 1.35, 0.4, 4.0)).is_none());
    }
}
