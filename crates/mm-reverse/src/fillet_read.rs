// SPDX-License-Identifier: AGPL-3.0-only
//! The radius of the round the scan has where a body has a sharp edge — read across the edge
//! at a few places along it, as the fillet box reads it for the first edge picked — so the
//! agent says "this edge is rounded" and ScanRuler says by how much. Ported from ScanRuler
//! (`plugins/reverse/app/reFilletRead.ts`, its pure parts, and `rounding` from
//! `plugins/reverse/core/intent.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! What is here: how each place along an edge is chosen, the scan cut square to the edge
//! there and handed to the reader that fits the edge's faces (`fillet_estimate` for a straight
//! edge between two flats, `curved_edge` for a curved one or one of any shape), the places'
//! readings made one, the reach widened where the round may be larger than it, the line
//! through the radii along the edge, and the command's answer over several edges. The scan's
//! cut is the caller's: a closure from a plane (origin, normal, the shortest chain kept) to
//! the scan's [`SectionCut`] — [`ScanSection`] makes one off an indexed mesh with
//! `slice_mesh`. What is not here is the command's plumbing: finding the body and its picked
//! edges, the store's scan and the mesh worker.

use glam::DVec3;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::curved_edge::{curved_stations, curved_unreadable, free_edge_of, free_stations, free_unreadable, is_curved_edge, read_curved_cut, read_free_cut, CurvedEdge, CurvedRead, EdgeSurface, FacedMesh, FreeEdge, Station, StationFaces};
use crate::fillet_estimate::{combine_readings, corner_unreadable, read_corner_cut, read_corner_setbacks, FilletReading, Miss, ReadShape};
use crate::fillet_num::{fixed_value, js_num, js_round, round4};
use crate::nominal::{nominal_of, nominal_step, NominalKind};
use crate::slice::{slice_mesh, SectionCut};
use crate::topo::{EdgeFingerprint, EdgeInfo, EdgeKind, Topology};

/// How many places along an edge the round is read at, by default, and where: evenly over the
/// middle of the edge, clear of its ends where the neighbouring faces' rounds run in.
pub const STATIONS: usize = 3;

/// The same on an edge of any other shape: its faces' traces change along it, and each place
/// reads them a little differently, so the spread of more places is what says how well the
/// round is known.
pub const FREE_STATIONS: usize = 7;

const MIDDLE_SHARE: f64 = 0.6;

/// The reaches a read is widened to, as multiples of the one asked, where the round reads at
/// the top of the search or not at all.
const WIDER: [f64; 2] = [2.0, 4.0];
/// How near the reach a radius counts as at the top.
const AT_TOP: f64 = 0.8;

/// The share of the points the other shape must rest on to count as fitting better.
const OTHER_POINTS: f64 = 0.75;

/// How many places must read for a line through them.
const LAW_PLACES: usize = 3;

/// A round's residual about its points: the RMS, mm, and how many points it rests on.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Residual {
    pub rms: f64,
    pub points: usize,
}

/// A countersink's figures' sigmas.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CountersinkSigma {
    pub angle_deg: f64,
    pub diameter: f64,
    pub depth: f64,
}

/// A chamfer on a hole's mouth as a countersink: the included angle, the diameter where it
/// meets the flat face, and its depth along the axis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeCountersink {
    pub angle_deg: f64,
    pub diameter: f64,
    pub depth: f64,
    pub sigma: CountersinkSigma,
}

/// A round read off the scan at one edge, for the answer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeRound {
    pub radius: f64,
    pub sigma: f64,
    /// The radius's nominal (to the half millimetre), how far off, and in how many sigmas.
    pub nominal: f64,
    pub off: f64,
    pub sigmas: Option<f64>,
    pub residual: Residual,
    /// A chamfer's setback on each of the edge's faces apart, mm, with their sigmas — an
    /// uneven bevel, a countersink.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setbacks: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setback_sigma: Option<[f64; 2]>,
    /// A chamfer on a hole's mouth as a countersink.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub countersink: Option<EdgeCountersink>,
}

/// An edge as the reading takes it: a fingerprint, with the ends and the faces' surfaces where
/// the body gives them — a circle edge needs both — and, for an edge of any other shape, its
/// polyline, its faces by hash and the body's mesh they are found on.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadEdge<'a> {
    /// The fingerprint, the ends and the faces' surfaces.
    pub edge: CurvedEdge,
    pub polyline: Option<Vec<DVec3>>,
    pub face_hashes: Option<Vec<u32>>,
    pub mesh: Option<&'a FacedMesh>,
}

impl ReadEdge<'_> {
    /// An edge known by its fingerprint alone, as a stored pick is.
    pub fn of_fingerprint(fingerprint: EdgeFingerprint) -> ReadEdge<'static> {
        ReadEdge { edge: CurvedEdge { fingerprint, ends: None, surfaces: None }, polyline: None, face_hashes: None, mesh: None }
    }

    pub fn fingerprint(&self) -> &EdgeFingerprint {
        &self.edge.fingerprint
    }
}

/// An edge with its faces' surfaces off the topology it was picked on, and the body's mesh
/// where it is at hand.
pub fn read_edge_of<'a>(edge: &EdgeInfo, topology: &Topology, mesh: Option<&'a FacedMesh>) -> ReadEdge<'a> {
    let surfaces = edge.faces.iter().map(|h| topology.face(*h).map(|f| EdgeSurface::from(&f.fingerprint))).collect();
    let curved = CurvedEdge { fingerprint: edge.fingerprint.clone(), ends: Some(edge.ends), surfaces: Some(surfaces) };
    match mesh.filter(|m| m.face_hash.is_some()) {
        Some(mesh) => ReadEdge { edge: curved, polyline: Some(edge.polyline.clone()), face_hashes: Some(edge.faces.clone()), mesh: Some(mesh) },
        None => ReadEdge { edge: curved, polyline: None, face_hashes: None, mesh: None },
    }
}

/// The edge as one of any shape, where the body's mesh came with it.
fn free_of(edge: &ReadEdge) -> Option<FreeEdge> {
    match (&edge.polyline, &edge.face_hashes, edge.mesh) {
        (Some(polyline), Some(faces), Some(mesh)) => free_edge_of(polyline, faces, mesh),
        _ => None,
    }
}

/// A shape read where the scan has the other one: a straight cut fits a round's points too,
/// loosely, and a big round a chamfer's — the other shape, read there as well, fits them
/// markedly closer, resting on nearly as many: a small chamfer across a round's middle fits
/// the few points it rests on closer and leaves the rest.
fn other_fits_better(read: &FilletReading, other: Option<&FilletReading>) -> bool {
    other.is_some_and(|o| o.rms < 0.7 * read.rms && o.points as f64 >= OTHER_POINTS * read.points as f64)
}

/// A number with its value, sigma, how many points and their RMS, as several stations read it.
struct Valued {
    value: f64,
    sigma: f64,
    points: usize,
    rms: f64,
}

/// Several stations' readings of one number made one, as `combine_readings` makes the radius.
fn combine_values(xs: &[Valued]) -> Option<FilletReading> {
    combine_readings(&xs.iter().map(|x| FilletReading { radius: x.value, sigma: x.sigma, points: x.points, rms: x.rms }).collect::<Vec<_>>())
}

/// What one place read: the round, or why none — "sharp", "no scan", "chamfer better",
/// "S-profile: 45 % on the arc", "another face 1.7 mm off within the reach — reach ≤ 1.5" …
pub type PlaceRead = Result<CurvedRead, String>;

/// A place's verdict as the answer gives it: "R0.8" (a chamfer "C0.5"), or why it read none.
fn verdict(x: &PlaceRead, shape: ReadShape) -> String {
    match x {
        Err(why) => why.clone(),
        Ok(read) => format!("{}{}", if shape == ReadShape::Chamfer { "C" } else { "R" }, js_num(fixed_value(read.reading.radius, 2))),
    }
}

/// A measured figure, its nominal (the value itself where it already is one), how far off it
/// lies and in how many sigmas — `rounding` of `intent.ts`.
struct Rounding {
    nominal: f64,
    off: f64,
    sigmas: Option<f64>,
}

fn rounding(measured: f64, kind: NominalKind, sigma: Option<f64>) -> Rounding {
    let step = nominal_step(measured, kind);
    let nominal = nominal_of(measured, kind).unwrap_or_else(|| js_round(measured / step) * step);
    let off = measured - nominal;
    let sigmas = match sigma {
        Some(s) if s > 0.0 => Some(round4(off.abs() / s)),
        _ => None,
    };
    Rounding { nominal: round4(nominal), off: round4(off), sigmas }
}

/// How a round is read along an edge: at how many places (the default for the edge's kind
/// where `None`), how far from the edge, what shape, and whether the reach may widen by itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadOptions {
    pub stations: Option<usize>,
    pub reach: f64,
    pub shape: ReadShape,
    pub widen: bool,
}

impl ReadOptions {
    /// A round read at `reach`, at the default places, the reach held.
    pub fn reach(reach: f64) -> ReadOptions {
        ReadOptions { stations: None, reach, shape: ReadShape::Round, widen: false }
    }
}

/// A place an edge's round was read at: where along the edge (t, 0 at its first end), the
/// point, and what it read there — `None` where none.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdgeStation {
    pub t: f64,
    pub at: DVec3,
    pub radius: Option<f64>,
    pub sigma: Option<f64>,
}

/// A round running evenly along an edge: the line through the places' radii over t, carried
/// out to the edge's ends — [first end, other] as a variable fillet takes it — its rms about
/// them, and theirs about their median.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinearLaw {
    pub radius: [f64; 2],
    pub rms: f64,
    pub median_rms: f64,
}

/// The line through the radii read along an edge, from three places that read at least;
/// `None` with fewer.
pub fn linear_law(stations: &[EdgeStation]) -> Option<LinearLaw> {
    let read: Vec<(f64, f64)> = stations.iter().filter_map(|s| s.radius.map(|r| (s.t, r))).collect();
    let n = read.len();
    if n < LAW_PLACES {
        return None;
    }
    let count = n as f64;
    let tm = read.iter().fold(0.0, |a, s| a + s.0) / count;
    let rm = read.iter().fold(0.0, |a, s| a + s.1) / count;
    let stt = read.iter().fold(0.0, |a, s| a + (s.0 - tm).powi(2));
    if !(stt > 1e-9) {
        return None;
    }
    let slope = read.iter().fold(0.0, |a, s| a + (s.0 - tm) * (s.1 - rm)) / stt;
    let at = |t: f64| rm + slope * (t - tm);
    let mut sorted: Vec<f64> = read.iter().map(|s| s.1).collect();
    sorted.sort_by(f64::total_cmp);
    let median = if n % 2 == 1 { sorted[n >> 1] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 };
    Some(LinearLaw {
        radius: [round4(at(0.0)), round4(at(1.0))],
        rms: round4((read.iter().fold(0.0, |a, s| a + (s.1 - at(s.0)).powi(2)) / count).sqrt()),
        median_rms: round4((read.iter().fold(0.0, |a, s| a + (s.1 - median).powi(2)) / count).sqrt()),
    })
}

/// The round across an edge as read off the scan: the round, or `None` with `why`; the radius
/// each place read (`None` where it read none); one verdict a place where they say more than
/// a sharp corner; the reach read at where it was widened; the places themselves and the line
/// through their radii.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EdgeRoundRead {
    pub round: Option<EdgeRound>,
    pub why: Option<String>,
    pub radii: Vec<Option<f64>>,
    pub places: Option<Vec<String>>,
    pub reach: Option<f64>,
    pub stations: Option<Vec<EdgeStation>>,
    pub linear: Option<LinearLaw>,
}

impl EdgeRoundRead {
    fn unread(why: impl Into<String>) -> EdgeRoundRead {
        EdgeRoundRead { why: Some(why.into()), ..EdgeRoundRead::default() }
    }
}

/// The scan's cut by a plane: from the plane's origin, its normal and the shortest chain kept,
/// mm, to the chains it cuts — the mesh worker's section in ScanRuler.
pub type Section<'s> = dyn FnMut(DVec3, DVec3, f64) -> SectionCut + 's;

/// The scan as `read_edge_round` cuts it: an indexed mesh, cut by `slice_mesh`.
#[derive(Debug, Clone, Default)]
pub struct ScanSection {
    pub positions: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
}

impl ScanSection {
    pub fn new(positions: Vec<DVec3>, triangles: Vec<[u32; 3]>) -> ScanSection {
        ScanSection { positions, triangles }
    }

    /// A mesh's live vertices and triangles, compacted.
    pub fn of_mesh(mesh: &mm_core::Mesh) -> ScanSection {
        let (positions, triangles, _) = mesh.to_arrays();
        ScanSection { positions, triangles }
    }

    /// The scan cut by the plane through `origin` square to `normal`, chains shorter than
    /// `min_length` dropped.
    pub fn cut(&self, origin: DVec3, normal: DVec3, min_length: f64) -> SectionCut {
        slice_mesh(&self.positions, &self.triangles, origin, normal, min_length)
    }

    /// The radius of the scan's bounding sphere — half its box's diagonal, a ten-thousandth
    /// at least — the part's size as ScanRuler takes it for the default reach (`round_reach`;
    /// ScanRuler's `boundsOf` over the scan's vertices).
    pub fn size(&self) -> f64 {
        if self.positions.is_empty() {
            return 1e-4;
        }
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in &self.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        ((hi - lo).length() / 2.0).max(1e-4)
    }
}

/// Where a place is read and how: a station on an edge of any shape (with its faces' cuts
/// kept between reads), a station on a curved edge, or a point on a straight edge between
/// two flats.
enum Spot {
    Free(Station, Box<StationFaces>),
    Curved(Station),
    Corner(DVec3),
}

/// One place along the edge: where, and the scan's cut there, taken once — the cut does not
/// depend on the reach, so a wider read cuts nothing again.
struct Place {
    spot: Spot,
    cut: SectionCut,
}

/// What the edge's places read at one reach.
#[derive(Clone)]
struct ReadAtReach {
    reach: f64,
    readings: Vec<PlaceRead>,
    found: Vec<CurvedRead>,
    reading: Option<FilletReading>,
}

/// The round across `edge` read off the scan at `stations` places, or why it cannot be:
/// `round: None` with `why` for an edge of the wrong kind (no radii), or where the scan shows
/// no round (radii, all or some `None`). A straight edge between two flats is read by
/// `fillet_estimate`, a curved one — a circle, an arc, an edge on a cylinder — by
/// `curved_edge` off its faces' surfaces, and any other edge picked on a body by
/// `curved_edge` off the body's mesh. `places` is one verdict a place — the round it read, or
/// why none — where the edge reads a round at some places only, or a place read none for more
/// than a sharp corner. Shared with the fillet's "fit" value.
pub fn read_edge_round(section: &mut Section, edge: &ReadEdge, opts: ReadOptions) -> EdgeRoundRead {
    let ReadOptions { stations: asked, reach, shape, widen } = opts;
    let fp = edge.fingerprint();
    let curved = is_curved_edge(fp);
    let unreadable = if curved { curved_unreadable(&edge.edge) } else { corner_unreadable(fp) };
    // What the faces' surfaces do not say, the body's mesh does: an edge of any shape, read
    // off its polyline and its faces' triangles.
    let free = if unreadable.is_some() { free_of(edge) } else { None };
    if let (Some(why), None) = (&unreadable, &free) {
        return EdgeRoundRead::unread(if fp.kind == EdgeKind::Other && edge.mesh.is_none() { "a curved edge known by its fingerprint only — pick it on the body".to_string() } else { why.clone() });
    }
    if let Some(why) = free.as_ref().and_then(free_unreadable) {
        return EdgeRoundRead::unread(why);
    }
    let stations = asked.unwrap_or(STATIONS);
    // Each place's cut, taken once, and where each place lies along the edge — and whether
    // the edge closes on itself (a full circle, a loop), which has no first end to run from.
    let mut places: Vec<Place> = Vec::new();
    let mut where_: Vec<(f64, DVec3)> = Vec::new();
    let closed: bool;
    if let Some(free) = &free {
        let Some(at) = free_stations(free, asked.unwrap_or(FREE_STATIONS), MIDDLE_SHARE) else {
            return EdgeRoundRead::unread("the edge’s geometry places no cut across it");
        };
        let poly = &free.polyline;
        closed = match (poly.first(), poly.last()) {
            (Some(a), Some(b)) => (*a - *b).length() < 1e-6,
            _ => false,
        };
        for st in at {
            where_.push((st.t.unwrap_or(0.5), st.at));
            let cut = section(st.at, st.normal, 0.0);
            places.push(Place { spot: Spot::Free(st, Box::default()), cut });
        }
    } else if curved {
        let Some(at) = curved_stations(&edge.edge, stations, MIDDLE_SHARE) else {
            return EdgeRoundRead::unread("the edge’s geometry places no cut across it");
        };
        closed = fp.kind == EdgeKind::Circle && fp.length / fp.radius.unwrap_or(f64::NAN) > 2.0 * std::f64::consts::PI - 1e-3;
        for st in at {
            where_.push((st.t.unwrap_or(0.5), st.at));
            let cut = section(st.at, st.normal, 0.0);
            places.push(Place { spot: Spot::Curved(st), cut });
        }
    } else {
        // A fingerprint alone (a stored pick) has its middle, direction and length: the ends
        // lie half the length either side.
        let half = fp.length / 2.0;
        let [a, b] = edge.edge.ends.unwrap_or([fp.mid - fp.dir * half, fp.mid + fp.dir * half]);
        let ts: Vec<f64> = if stations == 1 { vec![0.5] } else { (0..stations).map(|i| 0.5 - MIDDLE_SHARE / 2.0 + MIDDLE_SHARE * i as f64 / (stations - 1) as f64).collect() };
        closed = false;
        for t in ts {
            let p = a + (b - a) * t;
            where_.push((t, p));
            let cut = section(p, fp.dir, 0.0);
            places.push(Place { spot: Spot::Corner(p), cut });
        }
    }
    // The other shape fitting markedly closer is a verdict of its own.
    let better = format!("{} better", shape.other().word());
    let other = shape.other();
    let read_place = |place: &mut Place, reach: f64| -> PlaceRead {
        let cut = &place.cut;
        match &mut place.spot {
            Spot::Free(st, faces) => {
                let Some(free) = free.as_ref() else { return Err("sharp".to_string()) };
                let mut miss = Miss::default();
                let Some(read) = read_free_cut(free, st, faces, cut, reach, shape, Some(&mut miss)) else { return Err(miss.why.unwrap_or_else(|| "sharp".to_string())) };
                let again = read_free_cut(free, st, faces, cut, reach, other, None);
                if other_fits_better(&read.reading, again.as_ref().map(|r| &r.reading)) {
                    Err(better.clone())
                } else {
                    Ok(read)
                }
            }
            Spot::Curved(st) => {
                let mut miss = Miss::default();
                let Some(read) = read_curved_cut(&edge.edge, st, cut, reach, shape, Some(&mut miss)) else { return Err(miss.why.unwrap_or_else(|| "sharp".to_string())) };
                let again = read_curved_cut(&edge.edge, st, cut, reach, other, None);
                if other_fits_better(&read.reading, again.as_ref().map(|r| &r.reading)) {
                    Err(better.clone())
                } else {
                    Ok(read)
                }
            }
            Spot::Corner(p) => {
                let mut miss = Miss::default();
                let Some(reading) = read_corner_cut(fp, *p, cut, reach, shape, Some(&mut miss)) else { return Err(miss.why.unwrap_or_else(|| "sharp".to_string())) };
                if other_fits_better(&reading, read_corner_cut(fp, *p, cut, reach, other, None).as_ref()) {
                    return Err(better.clone());
                }
                let setbacks = if shape == ReadShape::Chamfer { read_corner_setbacks(fp, *p, cut, reach) } else { None };
                Ok(CurvedRead { reading, setbacks, countersink: None })
            }
        }
    };
    // A round at fewer than half the places read is no round along this edge: a dent, a
    // chip, a blob of the scan.
    let count = places.len();
    let enough = |n: usize| n > 0 && n * 2 >= count;
    let mut read_at = |r: f64| -> ReadAtReach {
        let readings: Vec<PlaceRead> = places.iter_mut().map(|place| read_place(place, r)).collect();
        let found: Vec<CurvedRead> = readings.iter().filter_map(|x| x.as_ref().ok().copied()).collect();
        let reading = if enough(found.len()) { combine_readings(&found.iter().map(|x| x.reading).collect::<Vec<_>>()) } else { None };
        ReadAtReach { reach: r, readings, found, reading }
    };
    let mut read = read_at(reach);
    // The search runs up to the reach: a round read at the top of it, or none, may be a
    // larger one than the reach takes in — read again at 2× and 4× the reach. A round at the
    // top that a wider read does not bear out stands.
    if widen {
        let mut most = read.clone();
        for k in WIDER {
            if read.reading.is_some_and(|r| r.radius < AT_TOP * read.reach) {
                break;
            }
            let wider = read_at(k * reach);
            if wider.reading.is_some() {
                read = wider;
            } else if read.reading.is_some() {
                break;
            } else if wider.found.len() >= most.found.len() {
                most = wider;
            }
        }
        if read.reading.is_none() {
            read = if !most.found.is_empty() { most } else { ReadAtReach { reach: WIDER[WIDER.len() - 1] * reach, ..read } };
        }
    }
    let ReadAtReach { reach: read_reach, readings, found, reading } = read;
    let radii: Vec<Option<f64>> = readings.iter().map(|r| r.as_ref().ok().map(|x| round4(x.reading.radius))).collect();
    let along: Vec<EdgeStation> = where_
        .iter()
        .enumerate()
        .map(|(i, (t, at))| {
            let r = readings.get(i).and_then(|r| r.as_ref().ok());
            EdgeStation { t: round4(*t), at: DVec3::new(round4(at.x), round4(at.y), round4(at.z)), radius: r.map(|x| x.reading.radius), sigma: r.map(|x| x.reading.sigma) }
        })
        .collect();
    let law = if closed { None } else { linear_law(&along) };
    let verdicts: Vec<String> = readings.iter().map(|r| verdict(r, shape)).collect();
    // The verdicts, where a place read none for more than a sharp corner: an edge sharp at
    // every place it missed says so once, as before.
    let said = verdicts.iter().zip(&readings).any(|(v, r)| r.is_err() && v != "sharp");
    let what = shape.word();
    let wide = if read_reach != reach { Some(round4(read_reach)) } else { None };
    // Every place missing for one reason, not a sharp corner: that reason, in place of the
    // list of what it might be.
    let one = match verdicts.first() {
        Some(first) if found.is_empty() && verdicts.iter().all(|v| v == first) && first != "sharp" => Some(first.clone()),
        _ => None,
    };
    let Some(reading) = reading else {
        let why = if found.is_empty() {
            let rest = one.clone().unwrap_or_else(|| {
                format!(
                    "the scan shows a sharp corner, {}, a gap or a {what} larger than the reach takes in here — give a larger reach for a larger {what}",
                    if shape == ReadShape::Chamfer { "a round" } else { "a chamfer" }
                )
            });
            format!("no {what} within {} mm (reach){}: {rest}", js_num(round4(read_reach)), if widen { ", the reach widened to 4× by itself" } else { "" })
        } else {
            format!("a {what} at {} of {} places only — not one along the edge", found.len(), readings.len())
        };
        let places = if !found.is_empty() {
            Some(verdicts)
        } else if one.is_some() || !said {
            None
        } else {
            Some(verdicts)
        };
        return EdgeRoundRead { round: None, why: Some(why), radii, places, reach: wide, stations: Some(along), linear: law };
    };
    let r = rounding(reading.radius, if shape == ReadShape::Chamfer { NominalKind::Length } else { NominalKind::Radius }, Some(reading.sigma));
    let mut round = EdgeRound {
        radius: round4(reading.radius),
        sigma: round4(reading.sigma),
        nominal: r.nominal,
        off: r.off,
        sigmas: r.sigmas,
        residual: Residual { rms: round4(reading.rms), points: reading.points },
        setbacks: None,
        setback_sigma: None,
        countersink: None,
    };
    let sb: Vec<_> = found.iter().filter_map(|x| x.setbacks).collect();
    if enough(sb.len()) {
        let side = |k: usize| combine_values(&sb.iter().map(|x| Valued { value: x.s[k], sigma: x.sigma[k], points: x.points, rms: x.rms }).collect::<Vec<_>>());
        if let (Some(s0), Some(s1)) = (side(0), side(1)) {
            round.setbacks = Some([round4(s0.radius), round4(s1.radius)]);
            round.setback_sigma = Some([round4(s0.sigma), round4(s1.sigma)]);
        }
    }
    let cs: Vec<_> = found.iter().filter_map(|x| x.countersink).collect();
    if enough(cs.len()) {
        let one = |v: &dyn Fn(&crate::curved_edge::Countersink) -> (f64, f64)| {
            combine_values(
                &cs.iter()
                    .map(|c| {
                        let (value, sigma) = v(c);
                        Valued { value, sigma, points: 1, rms: 0.0 }
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let angle = one(&|c| (c.angle_deg, c.angle_sigma));
        let dia = one(&|c| (c.diameter, c.diameter_sigma));
        let depth = one(&|c| (c.depth, c.depth_sigma));
        if let (Some(angle), Some(dia), Some(depth)) = (angle, dia, depth) {
            round.countersink = Some(EdgeCountersink {
                angle_deg: round4(angle.radius),
                diameter: round4(dia.radius),
                depth: round4(depth.radius),
                sigma: CountersinkSigma { angle_deg: round4(angle.sigma), diameter: round4(dia.sigma), depth: round4(depth.sigma) },
            });
        }
    }
    EdgeRoundRead { round: Some(round), why: None, radii, places: if said { Some(verdicts) } else { None }, reach: wide, stations: Some(along), linear: law }
}

/// How far from an edge the scan is read for its round: a few times the largest round
/// expected, from the part's size (`model_size`, the radius of the scan's bounding sphere).
pub fn round_reach(shape: ReadShape, model_size: f64) -> f64 {
    let least: f64 = if shape == ReadShape::Chamfer { 5.0 } else { 3.0 };
    least.max(model_size * 0.05)
}

/// The command's name (`reverse.body.fillet_read` in the reverse workspace), its title and its
/// description for the agent, as ScanRuler gives them.
pub const FILLET_READ_NAME: &str = "fillet_read";
pub const FILLET_READ_TITLE: &str = "Read rounds off the scan";
pub const FILLET_READ_DESCRIPTION: &str = "The round (or chamfer) the scan has along each of a body’s sharp edges — what a fillet or chamfer there should be — read in cuts of the scan square to the edge at several stations along it, the round fitted between the two faces’ traces (by their surfaces where they are flat, cylindrical, conical or spherical, else off the body’s own faces cut at each station). Edges are picked as a fillet picks them. Each edge answers round: true with radius (the median), sigma, nominal, off, sigmas, residual, radii, stations and linear (radius [first end, other], rms, medianRms — a variable fillet’s law where rms is well under medianRms); round: false with why where only some stations read one (places gives a verdict a station); round: null for an edge it cannot read. Edges where no station reads a round are gathered into sharp: { edges, why }. shape: \"chamfer\" answers chamfer, distance, setbacks, setbackSigma and on a hole’s mouth countersink. Then median and spread over the edges read. reach caps the radius read: left out, a round at the top of the search is read again at 2× and 4×; an empty read says \"no round within N mm\" — give a larger one. Needs a scan. Changes nothing.";

/// The most stations the command takes along an edge (and the fewest, 1).
pub const MAX_STATIONS: usize = 9;

/// The command's refusal where there is no scan, with the code `no_scan`.
pub const FILLET_READ_NO_SCAN: &str = "There is no scan to read a round off.";

/// The command's input fields as ScanRuler describes them to the agent: `edges` (one at
/// least), `shape` ("round" or "chamfer"), `stations` (an integer 1 to `MAX_STATIONS`) and
/// `reach` (above 0); `body` and `edges` are required.
pub const FILLET_READ_EDGES: &str = "The edges, as a fillet picks them.";
pub const FILLET_READ_SHAPE: &str = "What to read: a round (default) or a chamfer.";
pub const FILLET_READ_STATIONS: &str = "Stations along each edge; 3 by default, 7 on an edge read off the mesh.";
pub const FILLET_READ_REACH: &str = "How far from the edge the scan is read, mm; caps the radius. Left out: 5 % of the part’s size, 3 mm at least (5 for a chamfer).";

/// An edge picked for `fillet_read_answer`: its index in the body's edge list and the edge as
/// the reading takes it.
#[derive(Debug, Clone)]
pub struct PickedEdge<'a> {
    pub index: usize,
    pub edge: ReadEdge<'a>,
}

/// What `reverse.body.fillet_read` answers for a body's picked edges — the command's `run`
/// once the body, its picked edges and the scan's section are at hand: a row an edge (the
/// round it read, or why none), the edges no place reads a round on gathered into one row
/// with their why once, how many read, and the median and spread of their radii. `reach`
/// given caps the radius; left out, it is `round_reach` of the part's size, widened by
/// itself.
#[allow(clippy::too_many_arguments)]
pub fn fillet_read_answer(section: &mut Section, body_id: u64, body_name: &str, edges: &[PickedEdge], shape: ReadShape, stations: Option<usize>, reach: Option<f64>, model_size: f64) -> Value {
    let chamfer = shape == ReadShape::Chamfer;
    let mut out: Vec<Value> = Vec::new();
    let mut radii: Vec<f64> = Vec::new();
    // The edges no place reads a round on, gathered: one row and its why once, rather than a
    // row each with the same long why — a broad query comes back mostly as those. The why is
    // the same for all of them in one call (the reach given, or widened to its top alike); one
    // that differs keeps its own row.
    let mut sharp: Option<(Vec<usize>, String)> = None;
    for picked in edges {
        let index = picked.index;
        // Where the edge is, so a row is told by its place as well as its index.
        let m = picked.edge.fingerprint().mid;
        let mid = json!([round4(m.x), round4(m.y), round4(m.z)]);
        // The reach by default widens by itself; one given caps the radius.
        let opts = ReadOptions { stations, reach: reach.unwrap_or_else(|| round_reach(shape, model_size)), shape, widen: reach.is_none() };
        let read = read_edge_round(section, &picked.edge, opts);
        let found = if read.round.is_some() { json!(true) } else if !read.radii.is_empty() { json!(false) } else { Value::Null };
        let mut row = Map::new();
        row.insert("edge".into(), json!(index));
        row.insert("mid".into(), mid);
        // Where along the edge each place lies, for the radii beside them.
        let placed = read.stations.as_ref().map(|s| Value::Array(s.iter().map(|st| json!({ "t": st.t, "at": [st.at.x, st.at.y, st.at.z] })).collect()));
        if let Some(round) = &read.round {
            radii.push(round.radius);
            if chamfer {
                row.insert("chamfer".into(), json!(true));
                row.insert("distance".into(), json!(round.radius));
            } else {
                row.insert("round".into(), json!(true));
                row.insert("radius".into(), json!(round.radius));
            }
            if let Ok(Value::Object(rest)) = serde_json::to_value(round) {
                for (k, v) in rest {
                    if k != "radius" {
                        row.insert(k, v);
                    }
                }
            }
            row.insert("radii".into(), json!(read.radii));
            if let Some(p) = placed {
                row.insert("stations".into(), p);
            }
            if let Some(law) = &read.linear {
                row.insert("linear".into(), json!(law));
            }
            if let Some(places) = &read.places {
                row.insert("places".into(), json!(places));
            }
            if let Some(r) = read.reach {
                row.insert("reach".into(), json!(r));
            }
            out.push(Value::Object(row));
            continue;
        }
        let gathered = match (&read.why, &sharp) {
            (Some(why), s) => read.places.is_none() && !read.radii.is_empty() && read.radii.iter().all(|r| r.is_none()) && s.as_ref().is_none_or(|(_, w)| w == why),
            _ => false,
        };
        if gathered {
            let why = read.why.clone().unwrap_or_default();
            sharp.get_or_insert_with(|| (Vec::new(), why)).0.push(index);
            continue;
        }
        row.insert(if chamfer { "chamfer" } else { "round" }.into(), found);
        if let Some(why) = &read.why {
            row.insert("why".into(), json!(why));
        }
        if !read.radii.is_empty() {
            row.insert("radii".into(), json!(read.radii));
            if let Some(p) = placed {
                row.insert("stations".into(), p);
            }
        }
        if let Some(places) = &read.places {
            row.insert("places".into(), json!(places));
        }
        if let Some(r) = read.reach {
            row.insert("reach".into(), json!(r));
        }
        out.push(Value::Object(row));
    }
    let mut sorted = radii.clone();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n == 0 {
        None
    } else if n % 2 == 1 {
        Some(sorted[n >> 1])
    } else {
        Some((sorted[(n >> 1) - 1] + sorted[n >> 1]) / 2.0)
    };
    let mut answer = Map::new();
    answer.insert("body".into(), json!({ "id": body_id, "name": body_name }));
    answer.insert("edges".into(), Value::Array(out));
    if let Some((edges, why)) = sharp {
        answer.insert("sharp".into(), json!({ "edges": edges, "why": why }));
    }
    answer.insert(if chamfer { "chamfers" } else { "rounds" }.into(), json!(radii.len()));
    if let (Some(median), Some(lo), Some(hi)) = (median, sorted.first(), sorted.last()) {
        answer.insert("median".into(), json!(round4(median)));
        answer.insert("spread".into(), json!(round4(hi - lo)));
    }
    let mut answer = Value::Object(answer);
    js_json(&mut answer);
    answer
}

/// The answer's numbers as JavaScript writes them into JSON: a whole number without its
/// ".0" (12, not 12.0), −0 as 0.
fn js_json(v: &mut Value) {
    match v {
        Value::Number(n) => {
            if let Some(x) = n.as_f64().filter(|x| n.is_f64() && x.fract() == 0.0 && x.abs() < 9_007_199_254_740_992.0) {
                *v = Value::from(x as i64);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(js_json),
        Value::Object(map) => map.values_mut().for_each(js_json),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reFilletCurved.test.ts` (rounds and
    //! chamfers across circle edges and a straight edge on a cylinder, on swept scans) and
    //! `reRoundShortFace.test.ts` (rounds on short faces, a verdict for every place); the
    //! kernel-free parts of `reFilletRead.test.ts`, `reFilletCurvedBody.test.ts`,
    //! `reFilletLaw.test.ts`, `reFilletFree.test.ts` and `reRoundsHonest.test.ts`, the bodies
    //! the kernel built there given here as the edge fingerprints it would give.
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    use glam::DVec2;

    use super::*;
    use crate::fillet_estimate::{corner_window, cut_chains, estimate_fillet_radius, face_window};
    use crate::fillet_testkit::{arc, circle_outline, f32r, gauss, line, rounded_rect, sweep, sweep3, Outline, Prof, Scan};
    use crate::fit2d::Mulberry32;
    use crate::topo::{canonical_dir, EdgeSide, FaceKind};

    const Z: DVec3 = DVec3::Z;
    const STEP: f64 = 0.05;

    fn side(kind: FaceKind, dir: DVec3) -> EdgeSide {
        EdgeSide { kind, dir }
    }

    fn surface(kind: FaceKind, dir: DVec3, radius: Option<f64>, center: Option<DVec3>) -> Option<EdgeSurface> {
        Some(EdgeSurface { kind, dir, radius, center })
    }

    /// The other face of a circle edge, as a test names it.
    #[derive(Clone, Copy)]
    struct Other {
        kind: FaceKind,
        radius: Option<f64>,
        center: Option<DVec3>,
        dir: Option<DVec3>,
    }

    fn other(kind: FaceKind) -> Other {
        Other { kind, radius: None, center: None, dir: None }
    }

    /// A full circle edge about the Z axis at height z — its seam at −X, its middle at +X, as
    /// the kernel reads one.
    fn circle_edge(r: f64, z: f64, o: Other, centre: [f64; 2]) -> ReadEdge<'static> {
        let seam = DVec3::new(centre[0] - r, centre[1], z);
        let od = o.dir.unwrap_or(Z);
        let fingerprint = EdgeFingerprint {
            kind: EdgeKind::Circle,
            mid: DVec3::new(centre[0] + r, centre[1], z),
            dir: Z,
            length: 2.0 * PI * r,
            radius: Some(r),
            faces: vec![side(FaceKind::Plane, Z), side(o.kind, od)],
            origin: 1,
        };
        let surfaces = vec![surface(FaceKind::Plane, Z, None, None), surface(o.kind, od, o.radius, o.center)];
        ReadEdge { edge: CurvedEdge { fingerprint, ends: Some([seam, seam]), surfaces: Some(surfaces) }, polyline: None, face_hashes: None, mesh: None }
    }

    fn cylinder(radius: f64, center: [f64; 3]) -> Other {
        Other { kind: FaceKind::Cylinder, radius: Some(radius), center: Some(DVec3::from(center)), dir: None }
    }

    fn read(scan: &Scan, edge: &ReadEdge, opts: ReadOptions) -> EdgeRoundRead {
        read_edge_round(&mut scan.section(), edge, opts)
    }

    fn chamfer(reach: f64) -> ReadOptions {
        ReadOptions { shape: ReadShape::Chamfer, ..ReadOptions::reach(reach) }
    }

    fn concat(pieces: &[Vec<Prof>]) -> Vec<Prof> {
        pieces.concat()
    }

    /// The boss's foot rounded R1: the plate, the round onto the boss's side, the side up and
    /// the boss's top.
    fn boss(with_top: bool) -> Vec<Prof> {
        let r = 1.0;
        let mut p = vec![line([12.0, 0.0], [r, 0.0], [0.0, 1.0], STEP), arc([r, r], r, -PI / 2.0, -PI, true, STEP / 2.0), line([0.0, r], [0.0, 6.0], [1.0, 0.0], STEP)];
        if with_top {
            p.push(line([0.0, 6.0], [-4.5, 6.0], [0.0, 1.0], STEP));
        }
        concat(&p)
    }

    // ---- rounds across circle edges (reFilletCurved.test.ts) ----------------------------

    #[test]
    fn reads_the_r1_round_at_a_boss_s_foot_a_concave_circle_edge_between_the_plate_and_the_cylinder() {
        let scan = sweep3(&circle_outline(5.0, 360, [0.0, 0.0]), &boss(true), 0.01);
        let edge = circle_edge(5.0, 0.0, cylinder(5.0, [0.0, 0.0, 3.0]), [0.0, 0.0]);
        let got = read(&scan, &edge, ReadOptions::reach(3.0));
        assert_eq!(got.why, None);
        assert_eq!(got.radii.len(), 3);
        let round = got.round.expect("a round");
        assert!((round.radius - 1.0).abs() < 0.04, "{round:?}");
        assert_eq!(round.nominal, 1.0);
        assert!(round.sigma > 0.0 && round.sigma < 0.05, "{round:?}");
        // Not a chamfer.
        assert!(read(&scan, &edge, chamfer(3.0)).round.is_none());
    }

    /// A Ø6 hole's mouth countersunk 90° × Ø12, the plate 8 thick.
    fn countersunk() -> Scan {
        let profile = concat(&[
            line([12.0, 0.0], [3.0, 0.0], [0.0, 1.0], STEP),
            line([3.0, 0.0], [0.0, -3.0], [-FRAC_1_SQRT_2, FRAC_1_SQRT_2], STEP),
            line([0.0, -3.0], [0.0, -8.0], [-1.0, 0.0], STEP),
            line([0.0, -8.0], [12.0, -8.0], [0.0, -1.0], STEP),
        ]);
        sweep3(&circle_outline(3.0, 360, [0.0, 0.0]), &profile, 0.01)
    }

    #[test]
    fn reads_a_90_x_d12_countersink_on_a_d6_hole_s_mouth_and_no_round_there() {
        let scan = countersunk();
        let edge = circle_edge(3.0, 0.0, cylinder(3.0, [0.0, 0.0, -4.0]), [0.0, 0.0]);
        let got = read(&scan, &edge, chamfer(5.0));
        assert_eq!(got.why, None);
        let c = got.round.expect("a chamfer");
        assert!((c.radius - 3.0).abs() < 0.1, "{c:?}");
        let s = c.setbacks.expect("setbacks");
        assert!((s[0] - 3.0).abs() < 0.05 && (s[1] - 3.0).abs() < 0.05, "{s:?}");
        let cs = c.countersink.expect("a countersink");
        assert!((cs.angle_deg - 90.0).abs() < 2.0, "{cs:?}");
        assert!((cs.diameter - 12.0).abs() < 0.2, "{cs:?}");
        assert!((cs.depth - 3.0).abs() < 0.1, "{cs:?}");
        assert!(cs.sigma.angle_deg > 0.0 && cs.sigma.angle_deg < 1.0, "{cs:?}");
        assert!(read(&scan, &edge, ReadOptions::reach(5.0)).round.is_none());
    }

    #[test]
    fn reads_an_82_countersink_as_82_its_setbacks_apart() {
        let half = 41.0f64.to_radians();
        let depth = 2.0;
        let out = 3.0 + depth * half.tan();
        let n = [-half.cos(), half.sin()];
        let profile = concat(&[
            line([12.0, 0.0], [out - 3.0, 0.0], [0.0, 1.0], STEP),
            line([out - 3.0, 0.0], [0.0, -depth], n, STEP),
            line([0.0, -depth], [0.0, -8.0], [-1.0, 0.0], STEP),
            line([0.0, -8.0], [12.0, -8.0], [0.0, -1.0], STEP),
        ]);
        let scan = sweep3(&circle_outline(3.0, 360, [0.0, 0.0]), &profile, 0.01);
        let got = read(&scan, &circle_edge(3.0, 0.0, cylinder(3.0, [0.0, 0.0, -4.0]), [0.0, 0.0]), chamfer(5.0));
        let cs = got.round.and_then(|c| c.countersink).expect("a countersink");
        assert!((cs.angle_deg - 82.0).abs() < 2.0, "{cs:?}");
        assert!((cs.diameter - 2.0 * out).abs() < 0.2, "{cs:?}");
        assert!((cs.depth - depth).abs() < 0.1, "{cs:?}");
    }

    #[test]
    fn reads_r1_top_rounds_on_a_plate_outline_s_corner_arcs() {
        let r = 1.0;
        let profile = concat(&[line([-10.0, 0.0], [-r, 0.0], [0.0, 1.0], STEP), arc([-r, -r], r, PI / 2.0, 0.0, false, STEP / 2.0), line([0.0, -r], [0.0, -5.0], [1.0, 0.0], STEP)]);
        let scan = sweep3(&rounded_rect(20.0, 15.0, 5.0, 0.1), &profile, 0.01);
        // The arc at the (+X, +Y) corner: centre (15, 10), from (20, 10) to (15, 15).
        let mid = DVec3::new(15.0 + 5.0 * FRAC_1_SQRT_2, 10.0 + 5.0 * FRAC_1_SQRT_2, 0.0);
        let fingerprint = EdgeFingerprint {
            kind: EdgeKind::Circle,
            mid,
            dir: Z,
            length: PI / 2.0 * 5.0,
            radius: Some(5.0),
            faces: vec![side(FaceKind::Cylinder, Z), side(FaceKind::Plane, Z)],
            origin: 1,
        };
        let edge = ReadEdge {
            edge: CurvedEdge {
                fingerprint,
                ends: Some([DVec3::new(20.0, 10.0, 0.0), DVec3::new(15.0, 15.0, 0.0)]),
                surfaces: Some(vec![surface(FaceKind::Cylinder, Z, Some(5.0), Some(DVec3::new(15.0, 10.0, -2.5))), surface(FaceKind::Plane, Z, None, None)]),
            },
            polyline: None,
            face_hashes: None,
            mesh: None,
        };
        let got = read(&scan, &edge, ReadOptions::reach(3.0));
        assert_eq!(got.why, None);
        assert!((got.round.expect("a round").radius - 1.0).abs() < 0.04);
        assert!(got.radii.iter().all(|x| x.is_some_and(|x| (x - 1.0).abs() < 0.06)), "{:?}", got.radii);
    }

    #[test]
    fn reads_the_r1_5_round_on_a_spherical_bowl_s_rim_a_circle_edge_between_the_rim_and_the_sphere() {
        // The sphere R30 about (0, 0, 20) meets the rim z = 0 at Rc = √500.
        let rs = 30.0f64;
        let h = 20.0f64;
        let rc = (rs * rs - h * h).sqrt();
        let r = 1.5f64;
        // In (w, z) about the rim circle the sphere's centre is (−Rc, h); the round's centre r
        // under the rim and r outside the sphere.
        let s = DVec2::new(-rc, h);
        let c = DVec2::new(-rc + ((rs + r).powi(2) - (h + r).powi(2)).sqrt(), -r);
        let u = (c - s) / (rs + r);
        // The sphere from the round's tangent point down towards the bottom, its outward normal
        // into the bowl.
        let mut sphere: Vec<Prof> = Vec::new();
        let mut a = u.y.atan2(u.x);
        while a > -PI / 2.0 + 0.05 {
            let p = DVec2::new(s.x + rs * a.cos(), s.y + rs * a.sin());
            if p.x < -rc + 1.0 {
                break;
            }
            sphere.push(Prof { p, n: DVec2::new(-a.cos(), -a.sin()) });
            a -= STEP / rs;
        }
        let profile = concat(&[line([8.0, 0.0], [c.x, 0.0], [0.0, 1.0], STEP), arc([c.x, c.y], r, PI / 2.0, (-u.y).atan2(-u.x), false, STEP / 2.0), sphere]);
        let scan = sweep3(&circle_outline(rc, 720, [0.0, 0.0]), &profile, 0.01);
        let edge = circle_edge(rc, 0.0, Other { kind: FaceKind::Sphere, radius: Some(rs), center: None, dir: Some(Z) }, [0.0, 0.0]);
        let got = read(&scan, &edge, ReadOptions::reach(4.0));
        assert_eq!(got.why, None);
        let round = got.round.expect("a round");
        assert!((round.radius - r).abs() < 0.06, "{round:?}");
    }

    #[test]
    fn reads_the_boss_s_foot_through_three_times_the_noise() {
        let scan = sweep(&circle_outline(5.0, 360, [0.0, 0.0]), &boss(false), 0.03, 11, true);
        let got = read(&scan, &circle_edge(5.0, 0.0, cylinder(5.0, [0.0, 0.0, 3.0]), [0.0, 0.0]), ReadOptions::reach(3.0));
        let round = got.round.expect("a round");
        assert!((round.radius - 1.0).abs() < 0.05, "{round:?}");
        assert!(round.sigmas.is_some_and(|s| s < 3.0), "{round:?}");
    }

    #[test]
    fn reads_an_r1_round_on_a_straight_edge_between_a_flat_face_and_a_cylinder() {
        // Along Y: the top z = 0 meets a cylinder R5 about (x 0, z −3) at x = 4, rounded R1 —
        // the round's centre 1 under the top and 4 from the axis.
        let big_r = 5.0f64;
        let q = DVec2::new(0.0, -3.0);
        let r = 1.0f64;
        let c = DVec2::new(((big_r - r).powi(2) - (q.y + r).powi(2)).sqrt(), -r);
        let a2 = (c.y - q.y).atan2(c.x - q.x);
        let profile = concat(&[line([-6.0, 0.0], [c.x, 0.0], [0.0, 1.0], STEP), arc([c.x, c.y], r, PI / 2.0, a2, false, STEP / 2.0), arc([q.x, q.y], big_r, a2, -PI / 2.0, false, STEP)]);
        let outline: Outline = (0..201).map(|i| Prof { p: DVec2::new(0.0, -20.0 + 0.2 * i as f64), n: DVec2::X }).collect();
        let scan = sweep(&outline, &profile, 0.01, 5, false);
        let y = DVec3::Y;
        let edge = ReadEdge {
            edge: CurvedEdge {
                fingerprint: EdgeFingerprint { kind: EdgeKind::Line, mid: DVec3::new(4.0, 0.0, 0.0), dir: y, length: 40.0, radius: None, faces: vec![side(FaceKind::Plane, Z), side(FaceKind::Cylinder, y)], origin: 1 },
                ends: Some([DVec3::new(4.0, -20.0, 0.0), DVec3::new(4.0, 20.0, 0.0)]),
                surfaces: Some(vec![surface(FaceKind::Plane, Z, None, None), surface(FaceKind::Cylinder, y, Some(big_r), Some(DVec3::new(0.0, 0.0, -3.0)))]),
            },
            polyline: None,
            face_hashes: None,
            mesh: None,
        };
        let got = read(&scan, &edge, ReadOptions::reach(3.0));
        assert_eq!(got.why, None);
        let round = got.round.expect("a round");
        assert!((round.radius - 1.0).abs() < 0.04, "{round:?}");
    }

    /// A flat cut along an R40 drum about (x 0, z −30), running along Y, the edge between them
    /// chamfered C1: the scan, the edge's x, and the drum's axis in (x, z).
    fn drum_chamfer_scan() -> (Scan, f64, DVec2) {
        let big_r = 40.0f64;
        let q = DVec2::new(0.0, -30.0);
        let x = (big_r * big_r - q.y * q.y).sqrt();
        let corner_angle = (0.0 - q.y).atan2(x - q.x);
        let a = DVec2::new(x - 1.0, 0.0);
        let b_angle = corner_angle - 1.0 / big_r;
        let b = q + DVec2::new(b_angle.cos(), b_angle.sin()) * big_r;
        let cut = b - a;
        let n = DVec2::new(cut.y, -cut.x).normalize();
        let profile = concat(&[
            line([-6.0, 0.0], [a.x, a.y], [0.0, 1.0], STEP),
            line([a.x, a.y], [b.x, b.y], [n.x, n.y], STEP),
            arc([q.x, q.y], big_r, b_angle, b_angle - 8.0 / big_r, false, STEP),
        ]);
        let outline: Outline = (0..201).map(|i| Prof { p: DVec2::new(0.0, -20.0 + 0.2 * i as f64), n: DVec2::X }).collect();
        (sweep(&outline, &profile, 0.01, 5, false), x, q)
    }

    #[test]
    fn reads_a_c1_against_a_drum_s_tangent_on_a_straight_edge_between_a_flat_face_and_the_drum() {
        // Adapted from `reCurvedChamfer.test.ts`, whose drum (R40) and bore are the kernel's and
        // whose rim is read off the body's mesh (the test after this one): here a flat cut along
        // an R40 drum, the edge chamfered C1, read by the curved reader across the drum's trace
        // — a circle — against its tangent at the corner, over the reach short enough for the
        // drum to stay within 0.05 mm of it.
        let (scan, x, q) = drum_chamfer_scan();
        let y = DVec3::Y;
        let edge = ReadEdge {
            edge: CurvedEdge {
                fingerprint: EdgeFingerprint { kind: EdgeKind::Line, mid: DVec3::new(x, 0.0, 0.0), dir: y, length: 40.0, radius: None, faces: vec![side(FaceKind::Plane, Z), side(FaceKind::Cylinder, y)], origin: 1 },
                ends: Some([DVec3::new(x, -20.0, 0.0), DVec3::new(x, 20.0, 0.0)]),
                surfaces: Some(vec![surface(FaceKind::Plane, Z, None, None), surface(FaceKind::Cylinder, y, Some(40.0), Some(DVec3::new(0.0, 0.0, q.y)))]),
            },
            polyline: None,
            face_hashes: None,
            mesh: None,
        };
        let got = read(&scan, &edge, chamfer(5.0));
        assert_eq!(got.why, None);
        let c = got.round.expect("a chamfer");
        assert!((c.radius - 1.0).abs() <= 0.05, "{c:?}");
        assert_eq!(c.nominal, 1.0);
    }

    #[test]
    fn reads_a_c1_against_a_drum_s_tangent_on_an_edge_read_off_the_body_s_mesh() {
        // Adapted from `reCurvedChamfer.test.ts` along its own path: the rim there is no line
        // or circle, so it is read as an edge of any shape off the sharp body's mesh — the
        // drum's trace a circle off the mesh's normals, the chamfer read against its tangent
        // over the reach short enough for the drum to stay within 0.05 mm of it. Here the
        // drum's straight edge taken so, its faces the body's: the flat top meshed as two long
        // slivers, the drum as strips along Y, each carrying its normals.
        let (scan, x, q) = drum_chamfer_scan();
        let mut body = FacedMesh { face_hash: Some(Vec::new()), ..FacedMesh::default() };
        let push = |body: &mut FacedMesh, quad: [DVec3; 4], normals: [DVec3; 4], hash: u32| {
            let base = body.positions.len() as u32;
            body.positions.extend(quad);
            body.normals.extend(normals);
            body.triangles.push([base, base + 1, base + 2]);
            body.triangles.push([base, base + 2, base + 3]);
            if let Some(fh) = body.face_hash.as_mut() {
                fh.extend([hash, hash]);
            }
        };
        push(&mut body, [DVec3::new(-6.0, -20.0, 0.0), DVec3::new(x, -20.0, 0.0), DVec3::new(x, 20.0, 0.0), DVec3::new(-6.0, 20.0, 0.0)], [Z; 4], 1);
        let corner_angle = (0.0 - q.y).atan2(x - q.x);
        let at = |a: f64| (q + DVec2::new(a.cos(), a.sin()) * 40.0, DVec3::new(a.cos(), 0.0, a.sin()));
        for i in 0..40 {
            let (p0, n0) = at(corner_angle - 0.25 * i as f64 / 40.0);
            let (p1, n1) = at(corner_angle - 0.25 * (i + 1) as f64 / 40.0);
            push(&mut body, [DVec3::new(p0.x, -20.0, p0.y), DVec3::new(p0.x, 20.0, p0.y), DVec3::new(p1.x, 20.0, p1.y), DVec3::new(p1.x, -20.0, p1.y)], [n0, n0, n1, n1], 2);
        }
        let mut edge = ReadEdge::of_fingerprint(EdgeFingerprint {
            kind: EdgeKind::Other,
            mid: DVec3::new(x, 0.0, 0.0),
            dir: DVec3::Y,
            length: 40.0,
            radius: None,
            faces: vec![side(FaceKind::Plane, Z), side(FaceKind::Cylinder, DVec3::Y)],
            origin: 1,
        });
        edge.polyline = Some((0..=40).map(|i| DVec3::new(x, -20.0 + i as f64, 0.0)).collect());
        edge.face_hashes = Some(vec![1, 2]);
        edge.mesh = Some(&body);
        let got = read(&scan, &edge, chamfer(5.0));
        assert_eq!(got.why, None);
        assert_eq!(got.radii.len(), FREE_STATIONS);
        let c = got.round.expect("a chamfer");
        assert!((c.radius - 1.0).abs() <= 0.05, "{c:?}");
        assert_eq!(c.nominal, 1.0);
    }

    #[test]
    fn says_why_an_edge_is_not_read() {
        let scan = sweep3(&circle_outline(5.0, 360, [0.0, 0.0]), &line([12.0, 0.0], [0.0, 0.0], [0.0, 1.0], STEP), 0.01);
        let no_ends = ReadEdge::of_fingerprint(circle_edge(5.0, 0.0, other(FaceKind::Cylinder), [0.0, 0.0]).edge.fingerprint);
        let why = |edge: &ReadEdge| read(&scan, edge, ReadOptions::reach(3.0)).why.unwrap_or_default();
        assert!(why(&no_ends).contains("pick it on the body"), "{}", why(&no_ends));
        let torus = why(&circle_edge(5.0, 0.0, other(FaceKind::Torus), [0.0, 0.0]));
        assert!(torus.contains("torus"), "{torus}");
        let across = why(&circle_edge(5.0, 0.0, Other { dir: Some(DVec3::X), ..other(FaceKind::Cylinder) }, [0.0, 0.0]));
        assert!(across.contains("another axis"), "{across}");
    }

    // ---- rounds on short faces (reRoundShortFace.test.ts) -------------------------------

    /// The key's radius, and how high it stands above the housing.
    const KEY: f64 = 4.0;
    const HEIGHT: f64 = 1.7;

    /// The key's top edge: a full circle at z 0 between the top and the key's side, a
    /// cylinder about Z.
    fn key_edge(r: f64) -> ReadEdge<'static> {
        circle_edge(r, 0.0, cylinder(r, [0.0, 0.0, -1.0]), [0.0, 0.0])
    }

    /// A key's profile in (out from its side, up): the top, a round of `r` (none at 0) onto
    /// the side, the side down `HEIGHT`, and the housing running out from its foot.
    fn key(r: f64) -> Vec<Prof> {
        let mut p = vec![line([-3.5, 0.0], [-r, 0.0], [0.0, 1.0], STEP)];
        if r > 0.0 {
            p.push(arc([-r, -r], r, PI / 2.0, 0.0, false, STEP / 2.0));
        }
        p.push(line([0.0, -r], [0.0, -HEIGHT], [1.0, 0.0], STEP));
        p.push(line([0.0, -HEIGHT], [6.0, -HEIGHT], [0.0, 1.0], STEP));
        concat(&p)
    }

    /// The scan's chains at the one station on the key's edge, in the station's plane.
    fn key_chains(scan: &Scan) -> Vec<Vec<DVec2>> {
        let st = curved_stations(&key_edge(KEY).edge, 1, 0.6).expect("a station")[0];
        let cut = scan.section()(st.at, st.normal, 0.0);
        cut_chains(&cut, |p| {
            let d = p - st.at;
            DVec2::new(d.dot(st.x), d.dot(st.y))
        })
    }

    #[test]
    fn reads_the_r1_1_round_on_a_key_standing_1_7_mm_above_the_housing() {
        let scan = sweep3(&circle_outline(KEY, 360, [0.0, 0.0]), &key(1.1), 0.01);
        let got = read(&scan, &key_edge(KEY), ReadOptions::reach(3.0));
        assert_eq!(got.why, None);
        assert!((got.round.expect("a round").radius - 1.1).abs() < 0.05);
        assert!(got.radii.iter().all(|x| x.is_some_and(|x| (x - 1.1).abs() < 0.08)), "{:?}", got.radii);
    }

    #[test]
    fn ends_the_window_where_the_side_meets_the_housing_and_holds_the_reach_to_0_9_of_the_side() {
        let scan = sweep3(&circle_outline(KEY, 360, [0.0, 0.0]), &key(1.1), 0.01);
        let chains = key_chains(&scan);
        // In the station's plane: x out from the key's axis, y up. The side's outward normal is
        // +x, the top's +y; the side runs down from the corner, the top in towards the axis.
        let n = [DVec2::new(0.0, 1.0), DVec2::new(1.0, 0.0)];
        let w = face_window(&chains, 3.0, n, [DVec2::new(-1.0, 0.0), DVec2::new(0.0, -1.0)]);
        let end = w.end.clone().expect("a face end");
        assert_eq!(end.face, 1);
        assert_eq!(end.what, "another face 1.7 mm off");
        assert!((end.at - HEIGHT).abs() < 0.05, "{end:?}");
        assert!((w.reach - 0.9 * end.at).abs() < 0.5e-9);
        assert!(w.points.iter().all(|p| p.y > -HEIGHT + 0.05));
        // Over the whole reach the housing's points outweigh the round's: the plain window
        // reads no round; the face window reads it.
        assert!(estimate_fillet_radius(&corner_window(&chains, 3.0), n[0], n[1], 3.0, None).is_none());
        let reading = estimate_fillet_radius(&w.points, n[0], n[1], w.reach, None).expect("a round");
        assert!((reading.radius - 1.1).abs() < 0.05, "{reading:?}");
    }

    #[test]
    fn says_why_a_sharp_key_reads_none() {
        let scan = sweep3(&circle_outline(KEY, 360, [0.0, 0.0]), &key(0.0), 0.01);
        let got = read(&scan, &key_edge(KEY), ReadOptions::reach(3.0));
        assert!(got.round.is_none());
        assert_eq!(got.why.as_deref(), Some("no round within 3 mm (reach): another face 1.7 mm off within the reach — reach ≤ 1.5"));
    }

    #[test]
    fn leaves_a_face_that_runs_past_the_reach_as_it_was() {
        // The same round on a key standing 6 mm: no face ends, the window and the reach are the
        // plain ones.
        let tall = concat(&[line([-3.5, 0.0], [-1.1, 0.0], [0.0, 1.0], STEP), arc([-1.1, -1.1], 1.1, PI / 2.0, 0.0, false, STEP / 2.0), line([0.0, -1.1], [0.0, -6.0], [1.0, 0.0], STEP)]);
        let scan = sweep3(&circle_outline(KEY, 360, [0.0, 0.0]), &tall, 0.01);
        let chains = key_chains(&scan);
        let w = face_window(&chains, 3.0, [DVec2::new(0.0, 1.0), DVec2::new(1.0, 0.0)], [DVec2::new(-1.0, 0.0), DVec2::new(0.0, -1.0)]);
        assert!(w.end.is_none());
        assert_eq!(w.reach, 3.0);
        assert_eq!(w.points.len(), corner_window(&chains, 3.0).len());
    }

    #[test]
    fn gives_one_verdict_a_place_a_round_a_sharp_corner_a_step_no_scan() {
        let outline = circle_outline(5.0, 360, [0.0, 0.0]);
        let round = sweep3(&outline, &concat(&[line([-4.0, 0.0], [-0.8, 0.0], [0.0, 1.0], STEP), arc([-0.8, -0.8], 0.8, PI / 2.0, 0.0, false, STEP / 2.0), line([0.0, -0.8], [0.0, -5.0], [1.0, 0.0], STEP)]), 0.01);
        let sharp = sweep3(&outline, &concat(&[line([-4.0, 0.0], [0.0, 0.0], [0.0, 1.0], STEP), line([0.0, 0.0], [0.0, -5.0], [1.0, 0.0], STEP)]), 0.01);
        // Sharp, the side stepping out 0.8 mm a millimetre and a bit down.
        let step = sweep3(
            &outline,
            &concat(&[
                line([-4.0, 0.0], [0.0, 0.0], [0.0, 1.0], STEP),
                line([0.0, 0.0], [0.0, -1.2], [1.0, 0.0], STEP),
                line([0.0, -1.2], [0.8, -1.2], [0.0, -1.0], STEP),
                line([0.8, -1.2], [0.8, -5.0], [1.0, 0.0], STEP),
            ]),
            0.01,
        );
        let scans = [round, sharp, step];
        // Each place cut off its own scan, the last off none at all.
        let mut place = 0;
        let mut section = |origin: DVec3, normal: DVec3, min_length: f64| {
            let cut = scans.get(place).map(|s| slice_mesh(&s.positions, &s.triangles, origin, normal, min_length)).unwrap_or_default();
            place += 1;
            cut
        };
        let got = read_edge_round(&mut section, &key_edge(5.0), ReadOptions { stations: Some(4), ..ReadOptions::reach(3.0) });
        assert!(got.round.is_none());
        assert_eq!(got.why.as_deref(), Some("a round at 1 of 4 places only — not one along the edge"));
        assert_eq!(got.places, Some(vec!["R0.8".to_string(), "sharp".to_string(), "step 0.8 mm on face 2".to_string(), "no scan".to_string()]));
        assert_eq!(got.radii.iter().map(|r| r.is_some()).collect::<Vec<_>>(), vec![true, false, false, false]);
    }

    // ---- reverse.body.fillet_read on a box (reFilletRead.test.ts) ------------------------

    /// The profile of the scanned part in XZ: a 40 mm square about the origin, its top right
    /// corner (x = 20, z = 20) rounded `r`, sampled every `step` along the flats and in
    /// `arc_steps` along the round, each point moved off the surface by Gaussian noise.
    fn box_profile(r: f64, step: f64, arc_steps: usize, noise: f64) -> Vec<DVec2> {
        let mut rand = Mulberry32::new(7);
        let mut out: Vec<(DVec2, DVec2)> = Vec::new();
        let line = |out: &mut Vec<(DVec2, DVec2)>, a: [f64; 2], b: [f64; 2], n: [f64; 2]| {
            let (a, b) = (DVec2::from(a), DVec2::from(b));
            let k = js_round((b - a).length() / step).max(1.0) as usize;
            for i in 0..k {
                out.push((a + (b - a) * (i as f64 / k as f64), DVec2::from(n)));
            }
        };
        // Counter-clockwise in (x, z): bottom, right, the round, top, left.
        line(&mut out, [-20.0, -20.0], [20.0, -20.0], [0.0, -1.0]);
        line(&mut out, [20.0, -20.0], [20.0, 20.0 - r], [1.0, 0.0]);
        for i in 0..arc_steps {
            let a = PI / 2.0 * (i as f64 / arc_steps as f64);
            out.push((DVec2::new(20.0 - r + r * a.cos(), 20.0 - r + r * a.sin()), DVec2::new(a.cos(), a.sin())));
        }
        line(&mut out, [20.0 - r, 20.0], [-20.0, 20.0], [0.0, 1.0]);
        line(&mut out, [-20.0, 20.0], [-20.0, -20.0], [-1.0, 0.0]);
        out.into_iter()
            .map(|(p, n)| {
                let d = gauss(&mut rand) * noise;
                p + n * d
            })
            .collect()
    }

    /// Profiles in (x, z) at each y of `rows`, joined row to row and capped: a closed scan,
    /// welded as ScanRuler welds a loaded STL, its positions held as single floats.
    fn joined_rows(rows: &[f64], profiles: &[Vec<DVec2>]) -> Scan {
        let mut positions: Vec<DVec3> = Vec::new();
        let mut triangles: Vec<[u32; 3]> = Vec::new();
        let at = |p: DVec2, y: f64| f32r(DVec3::new(p.x, y, p.y));
        let base: Vec<u32> = profiles
            .iter()
            .zip(rows)
            .map(|(profile, y)| {
                let b = positions.len() as u32;
                positions.extend(profile.iter().map(|p| at(*p, *y)));
                b
            })
            .collect();
        let n = profiles[0].len() as u32;
        for j in 0..rows.len() - 1 {
            let (p, q) = (base[j], base[j + 1]);
            for i in 0..n {
                let k = (i + 1) % n;
                triangles.push([p + i, p + k, q + k]);
                triangles.push([p + i, q + k, q + i]);
            }
        }
        let first = positions.len() as u32;
        positions.push(f32r(DVec3::new(0.0, rows[0], 0.0)));
        let last = positions.len() as u32;
        positions.push(f32r(DVec3::new(0.0, rows[rows.len() - 1], 0.0)));
        let (b0, b1) = (base[0], base[rows.len() - 1]);
        for i in 0..n {
            let k = (i + 1) % n;
            triangles.push([first, b0 + k, b0 + i]);
            triangles.push([last, b1 + i, b1 + k]);
        }
        Scan { positions, triangles }
    }

    /// The box profile run along Y from −20 to 20, capped.
    fn rounded_box() -> Scan {
        let p = box_profile(2.5, 0.4, 24, 0.02);
        joined_rows(&[-20.0, 20.0], &[p.clone(), p])
    }

    /// A straight edge of the 40 mm box body between two of its faces (outward normals), as
    /// the kernel fingerprints it: along the line where the two faces meet, ±20 either way.
    fn box_edge(a: DVec3, b: DVec3) -> ReadEdge<'static> {
        let dir = canonical_dir(a.cross(b));
        let mid = (a + b) * 20.0;
        let fingerprint = EdgeFingerprint { kind: EdgeKind::Line, mid, dir, length: 40.0, radius: None, faces: vec![side(FaceKind::Plane, a), side(FaceKind::Plane, b)], origin: 1 };
        ReadEdge {
            edge: CurvedEdge { fingerprint, ends: Some([mid - dir * 20.0, mid + dir * 20.0]), surfaces: Some(vec![surface(FaceKind::Plane, a, None, None), surface(FaceKind::Plane, b, None, None)]) },
            polyline: None,
            face_hashes: None,
            mesh: None,
        }
    }

    fn picked(edges: Vec<ReadEdge<'_>>) -> Vec<PickedEdge<'_>> {
        edges.into_iter().enumerate().map(|(index, edge)| PickedEdge { index, edge }).collect()
    }

    #[test]
    fn reads_the_round_the_scan_has_along_a_sharp_edge_and_gathers_the_sharp_ones_into_one_row() {
        // Adapted: the sharp box body is the kernel's; its four top edges are given here as the
        // fingerprints it gives them.
        let scan = rounded_box();
        let edges = picked(vec![box_edge(Z, DVec3::X), box_edge(Z, DVec3::NEG_X), box_edge(Z, DVec3::Y), box_edge(Z, DVec3::NEG_Y)]);
        let answer = fillet_read_answer(&mut scan.section(), 1, "Solid 1", &edges, ReadShape::Round, None, None, scan.size());
        // The rounded edge a row; the three sharp ones one row, the why once.
        let rows = answer["edges"].as_array().expect("rows");
        assert_eq!(rows.len(), 1, "{answer:#}");
        let rounded = &rows[0];
        assert_eq!(rounded["round"], json!(true));
        let radius = rounded["radius"].as_f64().unwrap_or(f64::NAN);
        assert!((radius - 2.5).abs() < 0.05, "{answer:#}");
        let sigma = rounded["sigma"].as_f64().unwrap_or(f64::NAN);
        assert!(sigma > 0.0 && sigma < 0.1, "{answer:#}");
        assert_eq!(rounded["nominal"], json!(2.5));
        assert!(!rounded["sigmas"].is_null());
        assert!(rounded["residual"]["points"].as_u64().unwrap_or(0) > 10);
        assert_eq!(rounded["radii"].as_array().map(Vec::len), Some(3));
        let sharp = &answer["sharp"];
        let sharp_edges = sharp["edges"].as_array().expect("sharp edges");
        assert_eq!(sharp_edges.len(), 3);
        assert!(!sharp_edges.contains(&rounded["edge"]));
        let why = sharp["why"].as_str().unwrap_or_default();
        assert!(why.starts_with("no round"), "{answer:#}");
        // Widened by itself to 4× the default reach, and read sharp all along: what it might be.
        assert_eq!(
            why,
            "no round within 12 mm (reach), the reach widened to 4× by itself: the scan shows a sharp corner, a chamfer, a gap or a round larger than the reach takes in here — give a larger reach for a larger round"
        );
        assert_eq!(answer.to_string().matches("no round within").count(), 1);
        // Whole numbers written as JavaScript writes them: the box's edge middle at x 20.
        assert!(answer.to_string().contains(r#""mid":[20,0,20]"#), "{answer}");
        assert_eq!(answer["rounds"], json!(1));
        assert!((answer["median"].as_f64().unwrap_or(f64::NAN) - radius).abs() < 0.5e-6);
    }

    #[test]
    fn says_why_an_edge_is_not_read_a_line_running_tangent_off_a_round_and_an_arc_the_scan_has_sharp() {
        // Adapted: the box's bottom edge at y 20 rounded R3 by the kernel; the round's edges are
        // given as the fingerprints it gives them — a line where the round runs tangent off the
        // bottom, and the arc at its end on the face x 20, between that face and the round's
        // cylinder. The scan is sharp there.
        let scan = rounded_box();
        let axis = DVec3::X;
        let centre = DVec3::new(0.0, 17.0, -17.0);
        let cyl = |r: f64| surface(FaceKind::Cylinder, axis, Some(r), Some(centre));
        let down = DVec3::NEG_Z;
        let line_edge = ReadEdge {
            edge: CurvedEdge {
                fingerprint: EdgeFingerprint { kind: EdgeKind::Line, mid: DVec3::new(0.0, 17.0, -20.0), dir: axis, length: 40.0, radius: None, faces: vec![side(FaceKind::Plane, down), side(FaceKind::Cylinder, axis)], origin: 1 },
                ends: Some([DVec3::new(-20.0, 17.0, -20.0), DVec3::new(20.0, 17.0, -20.0)]),
                surfaces: Some(vec![surface(FaceKind::Plane, down, None, None), cyl(3.0)]),
            },
            polyline: None,
            face_hashes: None,
            mesh: None,
        };
        let s = 3.0 * FRAC_1_SQRT_2;
        let arc_edge = ReadEdge {
            edge: CurvedEdge {
                fingerprint: EdgeFingerprint {
                    kind: EdgeKind::Circle,
                    mid: DVec3::new(20.0, 17.0 + s, -17.0 - s),
                    dir: axis,
                    length: PI / 2.0 * 3.0,
                    radius: Some(3.0),
                    faces: vec![side(FaceKind::Plane, axis), side(FaceKind::Cylinder, axis)],
                    origin: 1,
                },
                ends: Some([DVec3::new(20.0, 17.0, -20.0), DVec3::new(20.0, 20.0, -17.0)]),
                surfaces: Some(vec![surface(FaceKind::Plane, axis, None, None), cyl(3.0)]),
            },
            polyline: None,
            face_hashes: None,
            mesh: None,
        };
        let edges = vec![PickedEdge { index: 7, edge: line_edge }, PickedEdge { index: 9, edge: arc_edge }];
        let answer = fillet_read_answer(&mut scan.section(), 1, "Solid 1", &edges, ReadShape::Round, None, None, scan.size());
        let rows = answer["edges"].as_array().expect("rows");
        // The line runs tangent off the round: no corner. The arc is a circle edge between the
        // end face and the round's cylinder — read, and the scan sharp there: where the cut does
        // not have the cylinder at all, its places say so.
        assert_eq!(rows.iter().map(|r| r["round"].clone()).collect::<Vec<_>>(), vec![Value::Null, json!(false)], "{answer:#}");
        assert!(rows[0]["why"].as_str().unwrap_or_default().contains("faces meet tangent"), "{answer:#}");
        assert_eq!(rows[1]["edge"], json!(9));
        assert!(rows[1]["why"].as_str().unwrap_or_default().starts_with("no round"), "{answer:#}");
        let places: Vec<&str> = rows[1]["places"].as_array().map(|p| p.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        assert!(places.contains(&"the faces not found in the cut"), "{answer:#}");
        assert!(places.iter().all(|p| ["sharp", "the faces not found in the cut"].contains(p)), "{answer:#}");
        assert!(answer.get("sharp").is_none());
        assert_eq!(answer["rounds"], json!(0));
        assert!(answer.get("median").is_none());
    }

    // ---- a plate's circle edges (reFilletCurvedBody.test.ts) ------------------------------

    #[test]
    fn reads_the_boss_s_foot_r1_its_top_sharp_and_the_hole_s_mouth_a_90_x_d12_countersink() {
        // Adapted: the plate, boss and hole are the kernel's; their circle edges are given as
        // the fingerprints it gives them.
        let hole = concat(&[
            line([12.0, 0.0], [3.0, 0.0], [0.0, 1.0], STEP),
            line([3.0, 0.0], [0.0, -3.0], [-FRAC_1_SQRT_2, FRAC_1_SQRT_2], STEP),
            line([0.0, -3.0], [0.0, -4.0], [-1.0, 0.0], STEP),
            line([0.0, -4.0], [12.0, -4.0], [0.0, -1.0], STEP),
        ]);
        let scan = sweep(&circle_outline(5.0, 360, [0.0, 0.0]), &boss(true), 0.01, 3, true).and(&sweep(&circle_outline(3.0, 360, [40.0, 0.0]), &hole, 0.01, 4, true));
        let foot = circle_edge(5.0, 0.0, cylinder(5.0, [0.0, 0.0, 3.0]), [0.0, 0.0]);
        let top = circle_edge(5.0, 6.0, cylinder(5.0, [0.0, 0.0, 3.0]), [0.0, 0.0]);
        let mouth = circle_edge(3.0, 0.0, cylinder(3.0, [40.0, 0.0, -2.0]), [40.0, 0.0]);
        let rounds = fillet_read_answer(&mut scan.section(), 2, "Plate", &[PickedEdge { index: 3, edge: foot }, PickedEdge { index: 4, edge: top }], ReadShape::Round, None, None, scan.size());
        let rows = rounds["edges"].as_array().expect("rows");
        assert_eq!(rows.len(), 1, "{rounds:#}");
        let foot = &rows[0];
        assert_eq!(foot["round"], json!(true), "{rounds:#}");
        assert!((foot["radius"].as_f64().unwrap_or(f64::NAN) - 1.0).abs() < 0.04, "{rounds:#}");
        // The top sharp: gathered into sharp, not a row of its own.
        assert_eq!(rounds["sharp"]["edges"], json!([4]));
        // A row says where its edge is: the foot's middle on the plate, on the boss's rim.
        let mid: Vec<f64> = foot["mid"].as_array().map(|m| m.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
        assert!(mid[2].abs() < 0.5e-6);
        assert!((mid[0].hypot(mid[1]) - 5.0).abs() < 0.5e-4);
        let sink = fillet_read_answer(&mut scan.section(), 2, "Plate", &[PickedEdge { index: 5, edge: mouth }], ReadShape::Chamfer, None, None, scan.size());
        let m = &sink["edges"][0];
        assert_eq!(m["chamfer"], json!(true), "{sink:#}");
        assert!((m["distance"].as_f64().unwrap_or(f64::NAN) - 3.0).abs() < 0.1, "{sink:#}");
        assert!((m["countersink"]["angleDeg"].as_f64().unwrap_or(f64::NAN) - 90.0).abs() < 2.0, "{sink:#}");
        assert!((m["countersink"]["diameter"].as_f64().unwrap_or(f64::NAN) - 12.0).abs() < 0.2, "{sink:#}");
        assert!((m["countersink"]["depth"].as_f64().unwrap_or(f64::NAN) - 3.0).abs() < 0.1, "{sink:#}");
        assert_eq!(sink["chamfers"], json!(1));
    }

    // ---- a round read along its edge (reFilletLaw.test.ts) -------------------------------

    /// The box's profile in XZ at y, counter-clockwise: the same number of points at every y,
    /// each moved off the surface by Gaussian noise. Its bottom right round R2.8 at both ends
    /// and R1.2 in the middle, its top right one from R1.2 at y −20 to R2.8 at y 20, its top
    /// left one R2 all along.
    fn law_profile(y: f64, rand: &mut Mulberry32) -> Vec<DVec2> {
        let rising = 1.2 + 1.6 * (y + 20.0) / 40.0;
        let vee = 1.2 + 1.6 * y.abs() / 20.0;
        let even = 2.0;
        let mut out: Vec<(DVec2, DVec2)> = Vec::new();
        let line = |out: &mut Vec<(DVec2, DVec2)>, a: [f64; 2], b: [f64; 2], n: [f64; 2]| {
            let (a, b) = (DVec2::from(a), DVec2::from(b));
            for i in 0..80 {
                out.push((a + (b - a) * (i as f64 / 80.0), DVec2::from(n)));
            }
        };
        let arc = |out: &mut Vec<(DVec2, DVec2)>, c: [f64; 2], r: f64, from: f64| {
            for i in 0..24 {
                let a = from + PI / 2.0 * (i as f64 / 24.0);
                out.push((DVec2::new(c[0] + r * a.cos(), c[1] + r * a.sin()), DVec2::new(a.cos(), a.sin())));
            }
        };
        let (br, tr, tl) = (vee, rising, even);
        line(&mut out, [-20.0, -20.0], [20.0 - br, -20.0], [0.0, -1.0]);
        arc(&mut out, [20.0 - br, -20.0 + br], br, -PI / 2.0);
        line(&mut out, [20.0, -20.0 + br], [20.0, 20.0 - tr], [1.0, 0.0]);
        arc(&mut out, [20.0 - tr, 20.0 - tr], tr, 0.0);
        line(&mut out, [20.0 - tr, 20.0], [-20.0 + tl, 20.0], [0.0, 1.0]);
        arc(&mut out, [-20.0 + tl, 20.0 - tl], tl, PI / 2.0);
        line(&mut out, [-20.0, 20.0 - tl], [-20.0, -20.0], [-1.0, 0.0]);
        out.into_iter()
            .map(|(p, n)| {
                let d = gauss(rand) * 0.01;
                p + n * d
            })
            .collect()
    }

    #[test]
    fn answers_where_each_place_lies_and_the_line_through_the_radii_carried_to_the_ends() {
        // Adapted: the box body is the kernel's; its top edges are given as the fingerprints
        // it gives them, the first end at y −20.
        let mut rand = Mulberry32::new(11);
        let rows: Vec<f64> = (0..81).map(|j| -20.0 + j as f64 * 0.5).collect();
        let profiles: Vec<Vec<DVec2>> = rows.iter().map(|y| law_profile(*y, &mut rand)).collect();
        let scan = joined_rows(&rows, &profiles);
        let mut top_right = box_edge(Z, DVec3::X);
        top_right.edge.ends = Some([DVec3::new(20.0, -20.0, 20.0), DVec3::new(20.0, 20.0, 20.0)]);
        let y0 = -20.0f64;
        let answer = fillet_read_answer(&mut scan.section(), 1, "Solid 1", &picked(vec![top_right, box_edge(Z, DVec3::NEG_X)]), ReadShape::Round, Some(7), Some(5.0), scan.size());
        let tr = &answer["edges"][0];
        let tl = &answer["edges"][1];
        let stations = tr["stations"].as_array().expect("stations");
        assert_eq!(stations.len(), 7, "{answer:#}");
        // t runs from the first end over the middle 60 %, and at lies there.
        assert!((stations[0]["t"].as_f64().unwrap_or(f64::NAN) - 0.2).abs() < 0.5e-6);
        assert!((stations[6]["t"].as_f64().unwrap_or(f64::NAN) - 0.8).abs() < 0.5e-6);
        for s in stations {
            let t = s["t"].as_f64().unwrap_or(f64::NAN);
            let y = s["at"][1].as_f64().unwrap_or(f64::NAN);
            assert!((y - (y0 + (-y0).signum() * 40.0 * t)).abs() < 0.5e-3, "{s}");
        }
        // The line through them: R1.2 to R2.8, the first end's first.
        let law = &tr["linear"];
        let r = |v: &Value, k: usize| v["radius"][k].as_f64().unwrap_or(f64::NAN);
        assert!((r(law, 0) - 1.2).abs() < 0.1, "{answer:#}");
        assert!((r(law, 1) - 2.8).abs() < 0.1, "{answer:#}");
        assert!(law["rms"].as_f64().unwrap_or(f64::NAN) < law["medianRms"].as_f64().unwrap_or(f64::NAN) / 4.0, "{answer:#}");
        // An even round: a flat line, its rms about the median as small.
        assert!((tl["radius"].as_f64().unwrap_or(f64::NAN) - 2.0).abs() < 0.05, "{answer:#}");
        assert!((r(&tl["linear"], 0) - r(&tl["linear"], 1)).abs() < 0.1, "{answer:#}");
    }

    #[test]
    fn a_line_through_three_places_or_more_and_none_through_fewer() {
        // Not in ScanRuler's tests: `linearLaw` worked through by hand.
        let st = |t: f64, radius: Option<f64>| EdgeStation { t, at: DVec3::ZERO, radius, sigma: None };
        let law = linear_law(&[st(0.2, Some(1.4)), st(0.5, Some(2.0)), st(0.8, Some(2.6)), st(0.9, None)]).expect("a line");
        assert_eq!(law.radius, [1.0, 3.0]);
        assert_eq!(law.rms, 0.0);
        assert_eq!(law.median_rms, round4((0.72f64 / 3.0).sqrt()));
        assert!(linear_law(&[st(0.2, Some(1.0)), st(0.8, Some(2.0)), st(0.5, None)]).is_none());
    }

    // ---- an edge of any shape (reFilletFree.test.ts) -------------------------------------

    #[test]
    fn says_to_pick_an_edge_of_any_shape_on_the_body_when_only_its_fingerprint_is_at_hand() {
        let fingerprint = EdgeFingerprint { kind: EdgeKind::Other, mid: DVec3::new(10.0, 10.0, 9.0), dir: DVec3::X, length: 20.0, radius: None, faces: vec![side(FaceKind::Cone, Z), side(FaceKind::Other, DVec3::Y)], origin: 4 };
        let mut section = |_: DVec3, _: DVec3, _: f64| SectionCut::default();
        let got = read_edge_round(&mut section, &ReadEdge::of_fingerprint(fingerprint), ReadOptions::reach(3.0));
        assert!(got.round.is_none());
        assert!(got.why.as_deref().unwrap_or_default().contains("pick it on the body"), "{got:?}");
    }

    /// The 40 mm box as a body's mesh: each face two triangles of its own, the face's normal
    /// at every corner, the face's hash 0 to 5 (+X, −X, +Y, −Y, +Z, −Z).
    fn box_mesh() -> FacedMesh {
        let mut mesh = FacedMesh { face_hash: Some(Vec::new()), ..FacedMesh::default() };
        let normals = [DVec3::X, DVec3::NEG_X, DVec3::Y, DVec3::NEG_Y, DVec3::Z, DVec3::NEG_Z];
        for (h, n) in normals.iter().enumerate() {
            let u = if n.x.abs() > 0.5 { DVec3::Y } else { DVec3::X };
            let v = n.cross(u);
            let base = mesh.positions.len() as u32;
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                mesh.positions.push(*n * 20.0 + u * (20.0 * a) + v * (20.0 * b));
                mesh.normals.push(*n);
            }
            mesh.triangles.push([base, base + 1, base + 2]);
            mesh.triangles.push([base, base + 2, base + 3]);
            if let Some(fh) = mesh.face_hash.as_mut() {
                fh.extend([h as u32, h as u32]);
            }
        }
        mesh
    }

    #[test]
    fn reads_a_round_on_an_edge_of_any_shape_off_the_body_s_mesh() {
        // Not in ScanRuler's tests, whose edges of any shape are the kernel's (a blade's foot on
        // a coned dish): the box's rounded edge taken as one, its faces' traces read off the
        // box body's own mesh, seven places along it.
        let scan = rounded_box();
        let mesh = box_mesh();
        let mut edge = box_edge(Z, DVec3::X);
        edge.edge.fingerprint.kind = EdgeKind::Other;
        edge.polyline = Some(vec![DVec3::new(20.0, -20.0, 20.0), DVec3::new(20.0, 0.0, 20.0), DVec3::new(20.0, 20.0, 20.0)]);
        edge.face_hashes = Some(vec![4, 0]);
        edge.mesh = Some(&mesh);
        let got = read(&scan, &edge, ReadOptions::reach(5.0));
        assert_eq!(got.why, None);
        assert_eq!(got.radii.len(), FREE_STATIONS);
        assert!(got.radii.iter().all(|r| r.is_some()), "{:?}", got.radii);
        let round = got.round.expect("a round");
        assert!((round.radius - 2.5).abs() < 0.05, "{round:?}");
        assert_eq!(round.nominal, 2.5);
        // A seam between two faces meeting flat says so.
        let mut seam = edge.clone();
        seam.face_hashes = Some(vec![4, 4]);
        assert!(read(&scan, &seam, ReadOptions::reach(5.0)).round.is_none());
    }

    // ---- a round larger than the default reach (reRoundsHonest.test.ts) ------------------

    #[test]
    fn reads_a_round_larger_than_the_reach_by_widening_it_and_held_to_a_reach_says_none_within_it() {
        // Not in ScanRuler's tests (whose own case, the R20 foot on a dome, is the test after
        // this one): the same through the curved reader — a swept disc whose top edge, between
        // a flat and a cylinder, is rounded R6, read at the 3 mm a part this size gets by
        // default.
        let r = 6.0;
        let profile = concat(&[line([-20.0, 0.0], [-r, 0.0], [0.0, 1.0], STEP), arc([-r, -r], r, PI / 2.0, 0.0, false, STEP / 2.0), line([0.0, -r], [0.0, -30.0], [1.0, 0.0], STEP)]);
        let scan = sweep3(&circle_outline(30.0, 720, [0.0, 0.0]), &profile, 0.015);
        let edge = circle_edge(30.0, 0.0, cylinder(30.0, [0.0, 0.0, -15.0]), [0.0, 0.0]);
        let reach = 3.0;
        // Held to the reach, as a reach given holds it: no round within it, said so — not "a
        // sharp corner". (Here every place says the one thing — the faces run on past the reach
        // — and says it in place of the list of what it might be.)
        let held = read(&scan, &edge, ReadOptions::reach(reach));
        assert!(held.round.is_none());
        assert_eq!(held.why.as_deref(), Some("no round within 3 mm (reach): the faces not found in the cut"));
        // Widened, as by default: R6 at every place, and the reach that read.
        let got = read(&scan, &edge, ReadOptions { widen: true, ..ReadOptions::reach(reach) });
        assert_eq!(got.why, None);
        assert!(got.radii.iter().all(|r| r.is_some()), "{:?}", got.radii);
        let round = got.round.expect("a round");
        assert!((round.radius - r).abs() < 0.1, "{round:?}");
        assert_eq!(round.nominal, r);
        assert!(got.reach.is_some_and(|x| x > reach), "{:?}", got.reach);
    }

    // ---- ScanRuler's own case: a boss's foot on a dome, read off the body's mesh ----------

    /// The dome R150 from r 80 at z 10 up to the boss r 25, as `reRoundsHonest.test.ts` draws
    /// it: the sphere's centre (0, ZC) in (r, z), the foot at z Z1.
    const DOME: f64 = 150.0;
    fn dome_zc() -> f64 {
        10.0 - (DOME * DOME - 80.0 * 80.0).sqrt()
    }
    fn dome_z1() -> f64 {
        dome_zc() + (DOME * DOME - 25.0 * 25.0).sqrt()
    }

    /// A face of a body of revolution about Z, as the kernel meshes one: a ring of `k` points
    /// at every profile point (r, z), the profile's normal swung round with it, the triangles
    /// between rings all on face `hash`.
    fn revolved_face(mesh: &mut FacedMesh, rings: &[(DVec2, DVec2)], k: usize, hash: u32) {
        let base = mesh.positions.len() as u32;
        for (p, n) in rings {
            for j in 0..k {
                let a = 2.0 * PI * j as f64 / k as f64;
                mesh.positions.push(DVec3::new(p.x * a.cos(), p.x * a.sin(), p.y));
                mesh.normals.push(DVec3::new(n.x * a.cos(), n.x * a.sin(), n.y));
            }
        }
        let k32 = k as u32;
        for i in 0..rings.len() as u32 - 1 {
            for j in 0..k32 {
                let jn = (j + 1) % k32;
                let (a, b, c, d) = (base + i * k32 + j, base + i * k32 + jn, base + (i + 1) * k32 + jn, base + (i + 1) * k32 + j);
                mesh.triangles.push([a, b, c]);
                mesh.triangles.push([a, c, d]);
            }
        }
        if let Some(fh) = mesh.face_hash.as_mut() {
            fh.extend(std::iter::repeat_n(hash, 2 * (rings.len() - 1) * k));
        }
    }

    #[test]
    fn reads_r20_at_a_boss_s_foot_on_a_dome_the_reach_widened_held_to_a_reach_says_none_within_it() {
        // ScanRuler's `reRoundsHonest.test.ts` case, the kernel's bodies drawn here: the boss's
        // foot a circle between a sphere and a cylinder, concave — no flat face, so it is read
        // as an edge of any shape off the sharp body's own mesh, its sphere's trace a circle.
        let (zc, z1) = (dome_zc(), dome_z1());
        let s = DVec2::new(0.0, zc);
        let r = 20.0;
        // The round's centre: r off the boss's side, r outside the sphere.
        let c = DVec2::new(25.0 + r, zc + ((DOME + r).powi(2) - (25.0 + r).powi(2)).sqrt());
        let a0 = (10.0 - zc).atan2(80.0);
        let a_t = (c.y - s.y).atan2(c.x - s.x);
        let a_s = (s.y - c.y).atan2(s.x - c.x);
        let profile = concat(&[arc([s.x, s.y], DOME, a0, a_t, false, 0.1), arc([c.x, c.y], r, a_s, -PI, true, STEP), line([25.0, c.y], [25.0, 60.0], [1.0, 0.0], 0.1), line([25.0, 60.0], [6.0, 60.0], [0.0, 1.0], 0.1)]);
        let scan = sweep(&circle_outline(0.0, 360, [0.0, 0.0]), &profile, 0.01, 11, true);

        // The sharp body near the foot: the dome's band (face 1) and the boss's side (face 2).
        let mut body = FacedMesh { face_hash: Some(Vec::new()), ..FacedMesh::default() };
        let a1 = (z1 - zc).atan2(25.0);
        let dome: Vec<(DVec2, DVec2)> = (0..=90)
            .map(|i| {
                let a = a1 + (a0 - a1) * i as f64 / 90.0;
                let u = DVec2::new(a.cos(), a.sin());
                (s + u * DOME, u)
            })
            .collect();
        let wall: Vec<(DVec2, DVec2)> = (0..=58).map(|i| (DVec2::new(25.0, z1 + (60.0 - z1) * i as f64 / 58.0), DVec2::X)).collect();
        revolved_face(&mut body, &dome, 360, 1);
        revolved_face(&mut body, &wall, 360, 2);
        // The foot's polyline, closed: its last point its first.
        let polyline: Vec<DVec3> = (0..=360)
            .map(|i| {
                let a = 2.0 * PI * (i % 360) as f64 / 360.0;
                DVec3::new(25.0 * a.cos(), 25.0 * a.sin(), z1)
            })
            .collect();
        let edge = ReadEdge {
            edge: CurvedEdge {
                fingerprint: EdgeFingerprint {
                    kind: EdgeKind::Circle,
                    mid: DVec3::new(-25.0, 0.0, z1),
                    dir: Z,
                    length: 2.0 * PI * 25.0,
                    radius: Some(25.0),
                    faces: vec![side(FaceKind::Sphere, DVec3::new(25.0, 0.0, z1 - zc) / DOME), side(FaceKind::Cylinder, Z)],
                    origin: 2,
                },
                ends: Some([DVec3::new(25.0, 0.0, z1); 2]),
                surfaces: Some(vec![surface(FaceKind::Sphere, Z, Some(DOME), Some(DVec3::new(0.0, 0.0, zc))), surface(FaceKind::Cylinder, Z, Some(25.0), Some(DVec3::new(0.0, 0.0, z1)))]),
            },
            polyline: Some(polyline),
            face_hashes: Some(vec![1, 2]),
            mesh: Some(&body),
        };
        // The default reach, 5 % of the part's size: well under the round.
        let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
        for p in &scan.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let reach = (0.05 * (hi - lo).length()).max(3.0);
        assert!(reach < r);
        // Held to the reach, as a reach given holds it: no round within it, said so — not "a
        // sharp corner".
        let held = read(&scan, &edge, ReadOptions::reach(reach));
        assert!(held.round.is_none());
        let why = held.why.clone().unwrap_or_default();
        let lead = why.strip_prefix("no round within ").and_then(|w| w.split_once(" mm (reach)")).map(|(n, _)| n.to_string()).unwrap_or_default();
        assert!(!lead.is_empty() && lead.chars().all(|ch| ch.is_ascii_digit() || ch == '.'), "{why}");
        assert!(why.contains("larger reach"), "{why}");
        // Widened, as by default: R20 at every place, and the reach that read.
        let got = read(&scan, &edge, ReadOptions { widen: true, ..ReadOptions::reach(reach) });
        assert_eq!(got.why, None);
        assert!(got.radii.iter().all(|x| x.is_some()), "{:?}", got.radii);
        let round = got.round.expect("a round");
        assert!((round.radius - r).abs() < 0.1, "{round:?}");
        assert_eq!(round.nominal, r);
        assert!(got.reach.is_some_and(|x| x > reach), "{:?}", got.reach);
    }

    #[test]
    fn the_command_s_words_and_shapes_are_scanruler_s() {
        // Not in ScanRuler's tests: the input's stations text follows the defaults, and a
        // shape reads and writes as the command's "round" / "chamfer".
        assert_eq!(FILLET_READ_STATIONS, format!("Stations along each edge; {STATIONS} by default, {FREE_STATIONS} on an edge read off the mesh."));
        assert_eq!(serde_json::to_value(ReadShape::Chamfer).ok(), Some(json!("chamfer")));
        assert_eq!(serde_json::from_value::<ReadShape>(json!("round")).ok(), Some(ReadShape::Round));
    }

    #[test]
    fn the_default_reach_is_a_twentieth_of_the_part_s_size_3_mm_at_least_5_for_a_chamfer() {
        assert_eq!(round_reach(ReadShape::Round, 20.0), 3.0);
        assert_eq!(round_reach(ReadShape::Chamfer, 20.0), 5.0);
        assert_eq!(round_reach(ReadShape::Round, 200.0), 10.0);
    }

    #[test]
    fn cuts_an_mm_core_mesh_as_the_scan() {
        let (mesh, _) = mm_core::Mesh::from_triangles(&rounded_box().positions, &rounded_box().triangles, None);
        let section = ScanSection::of_mesh(&mesh);
        let cut = section.cut(DVec3::ZERO, DVec3::Y, 0.0);
        assert_eq!(cut.chains.len(), 1);
        assert!((section.size() - 20.0 * 3f64.sqrt()).abs() < 0.1, "{}", section.size());
    }
}
