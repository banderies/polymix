// SPDX-License-Identifier: AGPL-3.0-only
//! The Extrude feature's build: a sketch's picked regions — or a flat face of a body, pulled as it
//! stands — swept into a solid, joined to the bodies it meets, cut out of them, or kept a body of
//! its own. Ported from ScanRuler (`plugins/reverse/core/resolve.ts` the extrude step,
//! `buildExtrude.ts`, `build.ts` extrude, extrudeToShape, extrudeUntil, nextPiece, pullFace),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Every extent ScanRuler has is built: a distance, a symmetric span, a plane to stop at or
//! planes on both sides — at any tilt, the prism run past them and trimmed by half-spaces — a
//! face of a body (a flat one as its plane, a curved one as itself), a body's surface, to next
//! and to last. A taper leans the sides from the sketch plane, or the end of the extent nearest
//! it, both ways for an extent reaching both ways, and is refused where it closes the profile,
//! saying where. A direction of its own sweeps an oblique prism whose ends stay parallel to the
//! sketch.
//!
//! Open CASCADE's calls are replaced by SolidCraft's: an oblique prism is the straight one
//! sheared along the sketch's normal (its ends stay in planes parallel to the sketch, its
//! sections translated copies of the profile, as OCCT's prism along a vector makes them), a
//! taper along a slant that prism with its sides drafted off the sweep about the start's
//! section, as OCCT's drafted prism leans them; a split by a body is
//! the prism less the body and the prism within it, and a split by a sheet or a curved face the
//! prism less the sheet thickened a hair beyond the side the extrude comes from.

use glam::{DVec2, DVec3};
use solidcraft_geom::{Loop2, Plane, Region2, Seg2, Vec2 as SVec2};
use solidcraft_kernel::{self as kernel, BoolOp, Body};

use super::{describe_plane_ref, dv3, islands, kernel_plane, kerr, mesh_deflection, plane_through, resolve_plane, sv3, volume, BodyState, Builder};
use crate::extrude_draft::{extent_lacks, NO_BODY};
use crate::frame::SectionFrame;
use crate::model::{BodyId, BodyOp, ExtrudeExtent, ExtrudeFeature, FaceProfile, Feature, Part, PlaneRef, MIN_SWEEP_RAD};
use crate::taper::{face_outline, taper_limit, OutlineEdge, TaperBreak, TaperLoop};
use crate::topo::FaceKind;

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, e: &ExtrudeFeature) -> Result<(), String> {
    if e.draft_deg.abs() >= 89.0 {
        return Err("The taper has to stay under 89° — the sides would lie down flat.".into());
    }
    // The profile: a face of a body pulled as it stands, or the sketch's picked regions.
    let (frame, regions) = match &e.face {
        Some(face) => pulled_face(b, face)?,
        None => b.sketch_profiles(part, e.sketch_id, &e.regions, e.region_pick.as_ref(), "extrude")?,
    };
    let n = frame.normal;
    // Swept another way: never along the plane, which sweeps nothing.
    let dir = match e.direction {
        None => n,
        Some(d) => {
            let Some(along) = d.try_normalize() else {
                return Err(format!("The direction runs nowhere — the extrusion would have no thickness; it has to leave the plane at {}° at least.", MIN_SWEEP_RAD.to_degrees()));
            };
            if along.dot(n).abs() < MIN_SWEEP_RAD.sin() {
                return Err(format!("The direction runs along the sketch plane — the extrusion would have no thickness; it has to leave the plane at {}° at least.", MIN_SWEEP_RAD.to_degrees()));
            }
            along
        }
    };
    if let Some(why) = extent_lacks(&e.extent) {
        return Err(why.to_string());
    }
    let sweep = Sweep { frame, plane: kernel_plane(&frame)?, regions, dir, draft_deg: e.draft_deg, op: e.op, reach_hint: sketch_reach(part, e) };
    let tools = sweep.solid(b, part, e)?;
    let made: f64 = tools.iter().filter_map(volume).sum();
    b.place(e.op, e.target, tools)?;
    // ---- EXT ----: the ghost's volume, for the box's preview line.
    b.out.ghost_volumes.insert(b.step, made);
    Ok(())
}

/// The plane a limit reaches to: a point on it and its unit normal.
#[derive(Debug, Clone, Copy)]
struct Limit {
    origin: DVec3,
    normal: DVec3,
}

/// The extrude's sweep: the profile's frame and regions, the way it is swept, its draft and what
/// it does with the bodies.
struct Sweep {
    frame: SectionFrame,
    plane: Plane,
    regions: Vec<Region2>,
    /// The way the profile is swept, a unit vector: the frame's normal, or a direction of its
    /// own. Offsets count along it.
    dir: DVec3,
    draft_deg: f64,
    op: BodyOp,
    /// The sketch's size, which a prism reaching to a plane is run past it by.
    reach_hint: f64,
}

impl Sweep {
    fn tapered(&self) -> bool {
        self.draft_deg.abs() > 1e-9
    }

    /// How steeply the sweep leaves the profile's plane: the cosine of its slant.
    fn steep(&self) -> f64 {
        self.dir.dot(self.frame.normal)
    }

    /// The solid the extrude makes before it is placed into the bodies, by its extent.
    fn solid(&self, b: &mut Builder, part: &Part, e: &ExtrudeFeature) -> Result<Vec<Body>, String> {
        let origin = self.frame.origin;
        let plane_of = |r: &PlaneRef| -> Result<Limit, String> {
            resolve_plane(r, part).map(|(o, n)| Limit { origin: o, normal: n }).ok_or_else(|| format!("The plane this extrusion reaches to is gone ({}).", describe_plane_ref(r, part)))
        };
        match &e.extent {
            ExtrudeExtent::ToBody { from, body, reverse } => {
                let (name, state) = body_before(b, part, *body, "The body this extrusion reaches to is gone, or failed to build.")?;
                self.to_shape(*from, &state, None, &name, *reverse)
            }
            ExtrudeExtent::ToNext { from, reverse } => self.until(b, e.target, *from, *reverse, false, e.op == BodyOp::Cut),
            ExtrudeExtent::ToLast { from, reverse } => self.until(b, e.target, *from, *reverse, true, e.op == BodyOp::Cut),
            ExtrudeExtent::ToFace { from, body, face } => {
                // Up to a flat face of a body: the face's plane, as to any plane. A curved one is
                // reached as the face it is.
                let found = if *body == NO_BODY { Err("the body is gone".to_string()) } else { b.find_faces(*body, &[*face], &[]) };
                let at = found.map_err(|lost| format!("The face this extrusion reaches to is gone (was: {lost})."))?.first().copied().ok_or("The face this extrusion reaches to is not on the body.")?;
                let (name, state) = body_before(b, part, *body, "The face this extrusion reaches to is not on the body.")?;
                let t = state.topology();
                let f = t.faces.iter().find(|x| x.hash as usize == at).ok_or("The face this extrusion reaches to is not on the body.")?;
                if f.fingerprint.kind != FaceKind::Plane {
                    return self.to_shape(*from, &state, Some(at), &name, false);
                }
                let to_face = Limit { origin: f.fingerprint.centroid, normal: f.fingerprint.dir };
                self.to_planes(Bound::At(*from), Bound::Face(to_face), origin)
            }
            ExtrudeExtent::Distance { from, to } => self.to_planes(Bound::At(*from), Bound::At(*to), origin),
            ExtrudeExtent::Symmetric { total } => self.to_planes(Bound::At(-total / 2.0), Bound::At(total / 2.0), origin),
            ExtrudeExtent::ToPlane { from, plane } => self.to_planes(Bound::At(*from), Bound::Plane(plane_of(plane)?), origin),
            ExtrudeExtent::BetweenPlanes { a, b: other } => self.to_planes(Bound::Plane(plane_of(a)?), Bound::Plane(plane_of(other)?), origin),
        }
    }

    // ---- prisms ----

    /// The prism of the regions from `start` to `end` along the sweep, leaning by the draft from
    /// the section at `start` — one solid a region.
    fn prism(&self, start: f64, end: f64) -> Result<Vec<Body>, String> {
        self.prism_of(&self.regions, start, end)
    }

    fn prism_of(&self, regions: &[Region2], start: f64, end: f64) -> Result<Vec<Body>, String> {
        if (end - start).abs() < 1e-9 {
            return Err("The extrusion has no length.".into());
        }
        let c = self.steep();
        let (h0, h1) = (start * c, end * c);
        let oblique = (c - 1.0).abs() >= 1e-12;
        if self.tapered() && !oblique {
            // The kernel's taper leans the sides out; the draft leans them in.
            let base = self.plane.offset(h0);
            let (len, sign) = ((h1 - h0).abs(), (h1 - h0).signum());
            return regions.iter().map(|r| kernel::extrude_tapered(&base, r, len, sign, (-self.draft_deg).to_radians()).map_err(kerr)).collect();
        }
        let (lo, hi) = (h0.min(h1), h0.max(h1));
        let made = if self.op == BodyOp::Cut || self.tapered() { kernel::extrude(&self.plane, regions, lo, hi).map_err(kerr)? } else { kernel::extrude_whole_sides(&self.plane, regions, lo, hi).map_err(kerr)? };
        if !oblique {
            return Ok(made);
        }
        // Aslant: the straight prism sheared along the normal — each section carried across by
        // its height, the ends staying parallel to the sketch.
        let m = shear_matrix(self.frame.origin, self.frame.normal, self.dir);
        let sheared: Vec<Body> = made.iter().map(|s| kernel::transform_matrix(s, m).map_err(kerr)).collect::<Result<_, _>>()?;
        if !self.tapered() {
            return Ok(sheared);
        }
        // Tapered aslant: each side leaned by the draft off the sweep, about where it leaves the
        // profile's section at the start.
        let at_start = self.frame.origin + self.dir * start;
        let neutral = Plane::new(sv3(at_start), sv3(self.frame.u), sv3(self.frame.v)).ok_or("The sketch plane has no direction.")?;
        let pull = self.dir * (end - start).signum();
        let mid = (start + end) / 2.0;
        sheared
            .iter()
            .zip(regions)
            .map(|(s, r)| {
                let sides: Vec<solidcraft_geom::Vec3> = r.loops().flat_map(|l| l.segs.iter()).map(|g| {
                    let m = g.mid();
                    sv3(self.frame.to_world(DVec2::new(m.x, m.y)) + self.dir * mid)
                }).collect();
                kernel::draft(s, &sides, &neutral, sv3(pull), self.draft_deg.to_radians()).map_err(|e| format!("The kernel cannot lean the sides of this profile swept aslant ({}) — sweep it along the sketch plane's normal, or take the taper off.", kerr(e)))
            })
            .collect()
    }

    /// The regions' outlines in the plane square to the sweep, edge by edge — what a taper is
    /// checked against.
    fn outlines(&self) -> Result<Vec<Outline>, String> {
        let dir = self.dir;
        let seed = if dir.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let u = dir.cross(seed).try_normalize().unwrap_or(DVec3::X);
        let v = dir.cross(u);
        let size = self.regions.iter().flat_map(|r| r.outer.polyline(0.05).into_vec()).fold((DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)), |(lo, hi), p| (lo.min(DVec2::new(p.x, p.y)), hi.max(DVec2::new(p.x, p.y))));
        let tol = mesh_deflection((size.1 - size.0).length().max(1.0)) / 20.0;
        self.regions
            .iter()
            .map(|r| {
                let edges: Vec<OutlineEdge> = r
                    .loops()
                    .flat_map(|l| l.segs.iter())
                    .map(|s| {
                        let polyline: Vec<DVec3> = s.polyline(tol).iter().map(|p| self.frame.to_world(DVec2::new(p.x, p.y))).collect();
                        let ends = [self.frame.to_world(DVec2::new(s.start().x, s.start().y)), self.frame.to_world(DVec2::new(s.end().x, s.end().y))];
                        OutlineEdge { polyline, ends }
                    })
                    .collect();
                let loops = face_outline(&edges, self.frame.origin, u, v).map_err(|e| e.0)?;
                Ok(Outline { loops: loops.into_iter().map(TaperLoop::Edges).collect(), origin: self.frame.origin, u, v })
            })
            .collect()
    }

    /// How far the outlines can run with the taper, and where the first to break broke.
    fn taper_reach(&self, outlines: &[Outline], length: f64) -> (f64, Option<(TaperBreak, Vec<DVec3>)>) {
        let lean_in = self.draft_deg.to_radians().tan();
        let mut out: (f64, Option<(TaperBreak, Vec<DVec3>)>) = (length, None);
        for o in outlines {
            let limit = taper_limit(&o.loops, lean_in, out.0);
            if limit.reach < out.0 {
                let broke = limit.broke.map(|b| {
                    let at = b.at.iter().map(|p| o.origin + o.u * p.x + o.v * p.y).collect();
                    (b, at)
                });
                out = (limit.reach, broke);
            }
        }
        out
    }

    /// A taper that would close the profile within `length` refused, saying where.
    fn taper_check(&self, outlines: &[Outline], length: f64) -> Result<(), String> {
        let (reach, broke) = self.taper_reach(outlines, length);
        if reach < length - 1e-9 {
            return Err(taper_refusal(self.draft_deg, reach, length, broke.as_ref()));
        }
        Ok(())
    }

    // ---- distances and planes ----

    /// An extrude from one bound to the other: offsets along the sweep, planes it reaches to
    /// (the prism run past them, then trimmed), or a flat face of a body as its plane.
    fn to_planes(&self, from: Bound, to: Bound, origin: DVec3) -> Result<Vec<Body>, String> {
        // Each limiting plane with the point whose side of it is kept: the start's, for a plane
        // or a face the extrude reaches to; the other plane's, for two it runs between.
        let start_at = |x: f64| origin + self.dir * x;
        let on_line = |p: Limit| {
            let c = self.dir.dot(p.normal);
            if c.abs() < 1e-9 {
                origin
            } else {
                origin + self.dir * ((p.origin - origin).dot(p.normal) / c)
            }
        };
        let mut planes: Vec<(Limit, DVec3)> = Vec::new();
        match (from, to) {
            (Bound::At(a), Bound::Plane(p) | Bound::Face(p)) => planes.push((p, start_at(a))),
            (Bound::Plane(a), Bound::Plane(b) | Bound::Face(b)) => {
                planes.push((a, on_line(b)));
                planes.push((b, on_line(a)));
            }
            (Bound::At(a), Bound::At(b)) => {
                if (b - a).abs() < 1e-9 {
                    return Err("The extrusion has no length.".into());
                }
            }
            _ => return Err("The extrusion has no length.".into()),
        }
        let solids = if self.tapered() {
            self.tapered_prism(from, to)?
        } else {
            let limits: Vec<Limit> = planes.iter().map(|(p, _)| *p).collect();
            let far = if limits.is_empty() { 0.0 } else { self.far_reach(&limits) };
            let lo = match from {
                Bound::At(x) => x,
                _ => -far,
            };
            let hi = match to {
                Bound::At(x) => x,
                // Towards the plane or the face, whichever side of the start it lies on.
                Bound::Plane(p) | Bound::Face(p) => {
                    if from_is_number(from) && (p.origin - origin).dot(self.dir) < lo {
                        -far
                    } else {
                        far
                    }
                }
            };
            self.prism(lo.min(hi), lo.max(hi))?
        };
        trim_by_planes(solids, &planes)
    }

    /// How far "far" is for an extrusion limited by a plane: past every plane and the sketch by
    /// a comfortable margin, so the trim always has material to trim (ScanRuler's `farReach`).
    fn far_reach(&self, planes: &[Limit]) -> f64 {
        let mut reach = self.reach_hint;
        for p in planes {
            reach = reach.max((p.origin - self.frame.origin).dot(self.dir).abs() * 2.0);
        }
        (reach * 4.0).max(10.0)
    }

    /// The prism of an extrude with a taper, before its limiting planes trim it. The profile is
    /// the section at the sketch plane — at the end of the extent nearest the plane, for one
    /// standing off it — and the sides lean from there as the prism goes away from it: both ways,
    /// for an extent reaching both ways, two prisms joined at the profile. An end that is a plane
    /// is reached by a prism just long enough to pass it everywhere — the profile's outline run
    /// out to it, and further where sides leaning out meet a plane tilted back from them. A taper
    /// that closes the profile, or a hole of it, before the end is refused, saying where.
    fn tapered_prism(&self, from: Bound, to: Bound) -> Result<Vec<Body>, String> {
        let dir = self.dir;
        let outlines = self.outlines()?;
        let corners: Vec<DVec3> = outlines.iter().flat_map(|o| o.loops.iter().flat_map(move |l| loop_points(l).into_iter().map(move |p| o.origin + o.u * p.x + o.v * p.y))).collect();
        let lean_out = (-self.draft_deg).max(0.0).to_radians().tan();
        // Where the outline meets a plane, as offsets along the sweep, and how much further a
        // side leaning out has to run to meet it.
        let hits = |plane: Limit| -> Result<(f64, f64, f64), String> {
            let c = dir.dot(plane.normal);
            if c.abs() < 1e-9 {
                return Err("A plane this extrusion reaches to runs along it, so it never ends it.".into());
            }
            let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
            for p in &corners {
                let t = (plane.origin - *p).dot(plane.normal) / c;
                min = min.min(t);
                max = max.max(t);
            }
            // A side leaning out meets a plane tilted back from it further on.
            let tilt = (1.0 - c * c).max(0.0).sqrt() / c.abs();
            if lean_out * tilt >= 1.0 - 1e-9 {
                return Err("The sides lean out further than the plane they reach to tilts away — they never meet it. Ease the taper.".into());
            }
            Ok((min, max, 1.0 / (1.0 - lean_out * tilt)))
        };
        // The two ends as offsets along the sweep, and whether each is found on a plane — the
        // prism then runs a hair past it, for the trim to cut.
        let (lo, hi, lo_on_plane, hi_on_plane, stretch) = match (from, to) {
            (Bound::At(a), Bound::At(b)) => (a, b, false, false, 1.0),
            (Bound::At(a), Bound::Plane(p) | Bound::Face(p)) => {
                // To a plane or a face: as far as the outline has to go to pass it, the way it
                // lies from the start.
                let (min, max, stretch) = hits(p)?;
                (a, if (min + max) / 2.0 >= a { max } else { min }, false, true, stretch)
            }
            (Bound::Plane(pa), Bound::Plane(pb) | Bound::Face(pb)) => {
                // Between two planes: over everything between them.
                let a = hits(pa)?;
                let b = hits(pb)?;
                (a.0.min(b.0), a.1.max(b.1), true, true, a.2.max(b.2))
            }
            _ => return Err("The extrusion has no length.".into()),
        };
        // The profile's own section: at the sketch plane, or at the end of the extent nearest it.
        let start = lo.min(hi).max(0.0).min(lo.max(hi));
        let past = |e: f64| start + (e - start) * stretch + (e - start).signum() * (1e-3 * (e - start).abs() + 1e-3);
        let ends: Vec<f64> = [if lo_on_plane { past(lo) } else { lo }, if hi_on_plane { past(hi) } else { hi }].into_iter().filter(|e| (e - start).abs() > 1e-9).collect();
        if ends.is_empty() {
            return Err("The extrusion has no length.".into());
        }
        for end in &ends {
            self.taper_check(&outlines, (end - start).abs())?;
        }
        // Each region's prisms either way made one at the profile.
        let mut out = Vec::new();
        for r in &self.regions {
            let mut parts = Vec::new();
            for end in &ends {
                parts.extend(self.prism_of(std::slice::from_ref(r), start, *end)?);
            }
            out.push(fuse_all(parts)?);
        }
        Ok(out)
    }

    // ---- to a face, a body ----

    /// An extrude up to a curved face of a body (`face`, by index), or up to the body itself — a
    /// solid's skin, a sheet: the way the face or body lies from the start along the sweep, or
    /// along it where it lies both ways — against it with `reverse`.
    fn to_shape(&self, from: f64, body: &BodyState, face: Option<usize>, body_name: &str, reverse: bool) -> Result<Vec<Body>, String> {
        let start = from;
        let (limit, what, sheet) = match face {
            Some(i) => {
                let t = body.topology();
                let f = t.faces.iter().find(|x| x.hash as usize == i).ok_or("The face this extrusion reaches to is not on the body.")?;
                // The whole of the face's surface the kernel keeps in pieces (a cylinder's two
                // halves), as Open CASCADE keeps it one face.
                let whole = kernel::with_same_surface(&body.shape, &[i]);
                let points: Vec<solidcraft_geom::Vec3> = whole.iter().filter_map(|k| t.faces.iter().find(|x| x.hash as usize == *k)).map(|x| sv3(x.point)).collect();
                let copy = kernel::copy_faces(&body.shape, &points).map_err(kerr)?;
                let kind = if f.fingerprint.kind == FaceKind::Other { "curved" } else { f.fingerprint.kind.name() };
                (copy, format!("the {kind} face of {body_name}"), true)
            }
            None => (body.shape.clone(), body_name.to_string(), body.surface),
        };
        let reach = reach_along(&[&limit], self.frame.origin, self.dir);
        let toward = if reach.1 <= start + 1e-6 {
            -1.0
        } else if reach.0 >= start - 1e-6 {
            1.0
        } else if reverse {
            -1.0
        } else {
            1.0
        };
        self.up_to(start, &limit, sheet, toward, &what)
    }

    /// An extrude up to a shape the kernel has no plane of — a curved face, a body's skin
    /// (`limit`, named `what` in a refusal): each region of the profile run from `start` right
    /// through the shape, `toward` the way it lies along the sweep, and split by it; of the
    /// pieces, those that start at the profile are kept — along every line of the profile, from
    /// the sketch to where it first meets the shape. A kept piece that still reaches the far end
    /// never met it there: the shape does not cover the whole profile, and that is refused,
    /// saying where, rather than left sticking out of the part. A taper leans from the start; one
    /// that closes the profile before the far end is refused as for any extrude.
    fn up_to(&self, start: f64, limit: &Body, sheet: bool, toward: f64, what: &str) -> Result<Vec<Body>, String> {
        let reach = reach_along(&[limit], self.frame.origin, self.dir);
        if !reach.0.is_finite() {
            return Err(format!("{} has nothing to reach to.", upper(what)));
        }
        if if toward > 0.0 { !(reach.1 > start + 1e-6) } else { !(reach.0 < start - 1e-6) } {
            return Err(format!("{} lies {} the sketch — nothing meets it that way.", upper(what), if toward > 0.0 { "behind" } else { "ahead of" }));
        }
        let margin = (0.01 * (reach.1 - reach.0)).max(1.0);
        let far = if toward > 0.0 { reach.1.max(start) + margin } else { reach.0.min(start) - margin };
        let length = (far - start).abs();
        if self.tapered() {
            self.taper_check(&self.outlines()?, length)?;
        }
        // How near the start a piece has to begin, and the far end to reach, to count as there.
        let hair = 1e-3 * length + 1e-4;
        // A sheet splits by itself thickened a hair on its far side, so what is kept ends on it.
        let cutter = if sheet { thickened_beyond(limit, self.dir * toward, hair.min(0.01))? } else { limit.clone() };
        let mut kept = Vec::new();
        for r in &self.regions {
            for prism in self.prism_of(std::slice::from_ref(r), start, far)? {
                let mut pieces: Vec<Body> = Vec::new();
                if let Some(outside) = kernel::boolean(&prism, &cutter, BoolOp::Cut).map_err(kerr)? {
                    pieces.extend(islands(&outside).into_iter().map(|(p, _)| p));
                }
                if !sheet {
                    if let Some(inside) = kernel::boolean(&prism, &cutter, BoolOp::Intersect).map_err(kerr)? {
                        pieces.extend(islands(&inside).into_iter().map(|(p, _)| p));
                    }
                }
                if pieces.len() < 2 {
                    return Err(if !prism.meets(&cutter).unwrap_or(true) {
                        format!("The profile never meets {what} that way — it passes beside it.")
                    } else {
                        format!("{} does not cover the whole profile: the extrude runs past its edge and never ends on it. Draw the profile within what the face covers, or extrude up to the body.{}", upper(what), short_of_profile(&prism, limit, self.dir))
                    });
                }
                for piece in pieces {
                    let r = reach_along(&[&piece], self.frame.origin, self.dir);
                    let begins = if toward > 0.0 { r.0 - start <= hair } else { start - r.1 <= hair };
                    if !begins {
                        continue;
                    }
                    if if toward > 0.0 { r.1 >= far - hair } else { r.0 <= far + hair } {
                        let at = kernel::measure(&piece).map(|m| dv3(m.centroid)).unwrap_or_default();
                        return Err(format!(
                            "{} does not cover the whole profile: about ({}, {}, {}) the extrude runs past its edge and never ends on it. Draw the profile within what the face covers, or extrude up to the body.{}",
                            upper(what),
                            fixed2(at.x),
                            fixed2(at.y),
                            fixed2(at.z),
                            short_of_profile(&prism, limit, self.dir)
                        ));
                    }
                    kept.push(piece);
                }
            }
        }
        if kept.is_empty() {
            return Err(format!("Nothing of the profile reaches {what}."));
        }
        Ok(kept)
    }

    // ---- to next, to last ----

    /// An extrude whose end is found on the bodies. **To last** runs to the far end of the
    /// bodies it is measured against — just past it for a cut, so the cut goes through. **To
    /// next** runs the profile right through them and keeps the first piece of it: of what lies
    /// outside the bodies for a boss (the piece between the sketch and the first face met), of
    /// what lies inside them for a cut (the first wall gone through). A piece that still reaches
    /// the far end was never stopped — the body does not cover the whole profile that way — and
    /// is refused rather than left sticking out of the part. Measured against the one body a
    /// feature saved with a target works on, every body there is otherwise.
    fn until(&self, b: &Builder, target: Option<BodyId>, from: f64, reverse: bool, last: bool, cutting: bool) -> Result<Vec<Body>, String> {
        let ids = match target {
            Some(t) => vec![b.resolve(t)],
            None => b.body_ids(),
        };
        let bodies: Vec<Body> = ids.iter().filter_map(|id| b.body(*id)).filter(|s| !s.surface).map(|s| s.shape.clone()).collect();
        if bodies.is_empty() {
            return Err("There is no body to extrude up to — make one first, or give a distance.".into());
        }
        let start = from;
        let sign = if reverse { -1.0 } else { 1.0 };
        let refs: Vec<&Body> = bodies.iter().collect();
        let reach = reach_along(&refs, self.frame.origin, self.dir);
        let margin = (0.01 * (reach.1 - reach.0)).max(1.0);
        let end = if reverse { reach.0 } else { reach.1 };
        if !(sign * (end - start) > 1e-6) {
            return Err("No body lies that way from the sketch — reverse the direction.".into());
        }
        // A taper leans from the start, the profile's own section, on to the end; how far it can
        // lean before it closes the profile is found first.
        let outlines = if self.tapered() { self.outlines()? } else { Vec::new() };
        let can_run = |length: f64| if self.tapered() { self.taper_reach(&outlines, length) } else { (length, None) };
        if last {
            let to = if cutting { end + sign * margin } else { end };
            let (fits, broke) = can_run((to - start).abs());
            if fits < (to - start).abs() - 1e-9 {
                return Err(taper_refusal(self.draft_deg, fits, (to - start).abs(), broke.as_ref()));
            }
            return self.prism(start, to);
        }
        // To next stops where the profile meets the body: a taper that closes the profile
        // further on does not matter, so the prism runs no further than it can — and if the body
        // is not met by then, the taper says so.
        let wanted = (end + sign * margin - start).abs();
        let (fits, broke) = can_run(wanted);
        let closes_first = fits < wanted - 1e-9;
        let refusal = || taper_refusal(self.draft_deg, fits, wanted, broke.as_ref());
        if closes_first && fits < 1e-6 {
            return Err(refusal());
        }
        let far = if closes_first { start + sign * fits * (1.0 - 1e-6) } else { end + sign * margin };
        let run = Run { start, sign, far, reverse, cutting };
        let one = |r: &Region2| -> Result<Body, String> {
            let whole = fuse_all(self.prism_of(std::slice::from_ref(r), start, far)?)?;
            self.next_piece(&bodies, whole, &run, if closes_first { Some(&refusal as &dyn Fn() -> String) } else { None })
        };
        // Each region of the profile runs to its own next face — three circles under a ceiling
        // are three bosses, not the nearest one alone — and what each keeps is joined.
        if self.regions.len() > 1 {
            let mut kept = Vec::new();
            for (i, r) in self.regions.iter().enumerate() {
                match one(r) {
                    Ok(k) => kept.push(k),
                    Err(e) => {
                        // Which region, by where it lies: the regions come in no order the
                        // sketch shows.
                        let c = r.outer.centroid();
                        let at = self.frame.to_world(DVec2::new(c.x, c.y));
                        let at = if at.is_finite() { format!(" about ({}, {}, {})", fixed2(at.x), fixed2(at.y), fixed2(at.z)) } else { format!(" {} of {}", i + 1, self.regions.len()) };
                        return Err(format!("The profile's region{at}: {e}"));
                    }
                }
            }
            return Ok(kept);
        }
        let r = self.regions.first().ok_or("No profile picked — click the sketch's regions in the viewport.")?;
        Ok(vec![one(r)?])
    }

    /// To next for one prism (`whole`): of the pieces it leaves outside the bodies (a boss) or
    /// inside them (a cut), the first along the way. `refusal` is the taper's, where it closes
    /// the profile before the far end.
    fn next_piece(&self, bodies: &[Body], whole: Body, run: &Run, refusal: Option<&dyn Fn() -> String>) -> Result<Body, String> {
        let mut pieces: Vec<Body> = Vec::new();
        if run.cutting {
            for b in bodies {
                if let Some(inside) = kernel::boolean(&whole, b, BoolOp::Intersect).map_err(kerr)? {
                    pieces.extend(islands(&inside).into_iter().map(|(p, _)| p));
                }
            }
        } else {
            let mut outside = Some(whole.clone());
            for b in bodies {
                let Some(o) = outside else { break };
                outside = if o.meets(b).unwrap_or(true) { kernel::boolean(&o, b, BoolOp::Cut).map_err(kerr)? } else { Some(o) };
            }
            if let Some(o) = outside {
                pieces.extend(islands(&o).into_iter().map(|(p, _)| p));
            }
        }
        if pieces.is_empty() {
            return Err(match refusal {
                Some(r) => r(),
                None if run.cutting => "The profile meets no body that way.".into(),
                None => "The profile starts inside the body — there is nothing between it and the next face.".into(),
            });
        }
        // The first piece along the way, and how far it runs.
        let mut first = 0;
        let mut nearest = f64::INFINITY;
        let mut runs_to = 0.0;
        for (i, p) in pieces.iter().enumerate() {
            let r = reach_along(&[p], self.frame.origin, self.dir);
            let near = if run.reverse { run.start - r.1 } else { r.0 - run.start };
            if near < nearest {
                nearest = near;
                first = i;
                runs_to = if run.reverse { run.start - r.0 } else { r.1 - run.start };
            }
        }
        if !run.cutting && runs_to >= run.sign * (run.far - run.start) - 1e-6 {
            return Err(match refusal {
                Some(r) => r(),
                None => "The body does not cover the whole profile that way, so nothing stops it — use To a face, or a distance.".into(),
            });
        }
        Ok(pieces.swap_remove(first))
    }
}

/// Where a run to the next face starts, which way it goes and how far, and whether it cuts.
struct Run {
    start: f64,
    sign: f64,
    far: f64,
    reverse: bool,
    cutting: bool,
}

/// One end of an extent: an offset along the sweep, a plane it reaches to, or a flat face of a
/// body as its plane.
#[derive(Debug, Clone, Copy)]
enum Bound {
    At(f64),
    Plane(Limit),
    Face(Limit),
}

fn from_is_number(b: Bound) -> bool {
    matches!(b, Bound::At(_))
}

/// A face's outline in the plane square to the sweep: its loops, and the plane's origin and
/// axes they are measured in.
struct Outline {
    loops: Vec<TaperLoop>,
    origin: DVec3,
    u: DVec3,
    v: DVec3,
}

fn loop_points(l: &TaperLoop) -> Vec<DVec2> {
    match l {
        TaperLoop::Plain(p) => p.clone(),
        TaperLoop::Edges(e) => e.points.clone(),
    }
}

/// A taper refused: how far it can run against how far the extrude does, and — when it is known
/// — what breaks there and where.
pub(crate) fn taper_refusal(draft_deg: f64, reach: f64, length: f64, broke: Option<&(TaperBreak, Vec<DVec3>)>) -> String {
    let at = |p: &DVec3| format!("({}, {}, {})", fixed2(p.x), fixed2(p.y), fixed2(p.z));
    let what = match broke {
        None => String::new(),
        Some((b, pts)) => match b.broke.why() {
            "crosses" if pts.len() >= 2 => format!(": the edges through {} and {} run into each other there", at(&pts[0]), at(&pts[1])),
            "vanishes" if !pts.is_empty() => format!(": the edge through {} shrinks to nothing there", at(&pts[0])),
            _ if !pts.is_empty() => format!(": the edge through {} folds over itself there", at(&pts[0])),
            _ => String::new(),
        },
    };
    format!(
        "The taper {} {} mm from where it starts, and the extrude runs {} mm that way{what} — ease the taper, or shorten the extrude.",
        if draft_deg > 0.0 { "closes the profile" } else { "closes a hole or a notch of the profile" },
        fixed2(reach),
        fixed2(length)
    )
}

fn fixed2(x: f64) -> String {
    crate::js_number::to_fixed(x, 2)
}

fn upper(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// The shear that carries a straight prism along `normal` onto one swept along `along`: each
/// point moved across by its height over the plane through `origin`, the plane itself kept — a
/// column-major 4 × 4.
fn shear_matrix(origin: DVec3, normal: DVec3, along: DVec3) -> [[f64; 4]; 4] {
    let n = normal;
    let w = along / along.dot(n) - n;
    let t = -w * n.dot(origin);
    let mut m = [[0.0; 4]; 4];
    for (c, col) in m.iter_mut().enumerate().take(3) {
        for r in 0..3 {
            col[r] = if r == c { 1.0 } else { 0.0 } + w[r] * n[c];
        }
    }
    m[3] = [t.x, t.y, t.z, 1.0];
    m
}

/// Solids made one, where they touch or overlap; apart, they stay pieces of one list.
fn fuse_all(parts: Vec<Body>) -> Result<Body, String> {
    let mut it = parts.into_iter();
    let mut acc = it.next().ok_or("The extrusion has no length.")?;
    for p in it {
        acc = kernel::boolean(&acc, &p, BoolOp::Union).map_err(kerr)?.ok_or("The join produced nothing.")?;
    }
    Ok(acc)
}

/// A limiting plane trims what reached past it: the half of space on the far side of each plane,
/// seen from the point given with it, is cut away (the kernel's half-space box on that side, cut
/// out of the solid, as ScanRuler cuts its half-space).
fn trim_by_planes(solids: Vec<Body>, planes: &[(Limit, DVec3)]) -> Result<Vec<Body>, String> {
    let mut out = solids;
    for (p, keep) in planes {
        let away = if (*keep - p.origin).dot(p.normal) >= 0.0 { -p.normal } else { p.normal };
        let plane = plane_through(p.origin, away)?;
        let mut next = Vec::new();
        for s in &out {
            match kernel::half_space(s, &plane, true).map_err(kerr)? {
                // Nothing of it lies beyond the plane: kept whole.
                (None, _) => next.push(s.clone()),
                (Some(_), tool) => {
                    if let Some(kept) = kernel::boolean(s, &tool, BoolOp::Cut).map_err(kerr)? {
                        next.extend(islands(&kept).into_iter().map(|(b, _)| b));
                    }
                }
            }
        }
        out = next;
    }
    if out.is_empty() {
        return Err("The extrusion lies wholly beyond the plane it reaches to — nothing of it is left.".into());
    }
    Ok(out)
}

/// How far shapes reach along a direction from a point, off their tessellation — a bounding box
/// is too long on a body that lies askew.
pub(crate) fn reach_along(shapes: &[&Body], origin: DVec3, dir: DVec3) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for s in shapes {
        let tol = mesh_deflection(super::diagonal_of(s)) * 4.0;
        let Ok(m) = s.tessellate(tol) else { continue };
        for p in &m.positions {
            let t = (dv3(*p) - origin).dot(dir);
            min = min.min(t);
            max = max.max(t);
        }
    }
    (min, max)
}

/// How far a face stops short of a prism, square to the extrude: the largest the prism's box
/// runs past the face's on the two axes most square to it, in mm and as a share of the face's
/// width there — what a fitted surface's margin has to grow by. Empty where the boxes do not
/// show it.
fn short_of_profile(prism: &Body, limit: &Body, dir: DVec3) -> String {
    let (p, f) = (prism.outline_bounds(), limit.outline_bounds());
    if p.is_empty() || f.is_empty() {
        return String::new();
    }
    let (pmin, pmax, fmin, fmax) = (dv3(p.min), dv3(p.max), dv3(f.min), dv3(f.max));
    let (mut mm, mut axis) = (0.0, 0usize);
    for i in 0..3 {
        if dir[i].abs() > 0.7 {
            continue;
        }
        let past = (fmin[i] - pmin[i]).max(pmax[i] - fmax[i]);
        if past > mm {
            mm = past;
            axis = i;
        }
    }
    if !(mm > 1e-3) {
        return String::new();
    }
    let width = fmax[axis] - fmin[axis];
    let share = if width > 0.0 { format!(" — about {} % of its width, which a fitted surface's expandPct adds", ((100.0 * mm) / width).ceil()) } else { String::new() };
    format!(" The face stops up to {} mm short along {}{share}.", fixed2(mm), ["x", "y", "z"][axis])
}

/// A sheet thickened by `hair` to the side `beyond` points to: a solid whose near skin is the
/// sheet itself, for a prism to be split on it exactly.
fn thickened_beyond(sheet: &Body, beyond: DVec3, hair: f64) -> Result<Body, String> {
    let hair = hair.max(1e-4);
    let thick = kernel::thicken(sheet, hair, false).map_err(|e| format!("The sheet this extrusion reaches to cannot be read as a skin to stop on ({}).", kerr(e)))?;
    // The side it grew to: a point a little beyond the sheet is inside it, or the other way.
    let probe = sheet_point(sheet);
    if let Some(p) = probe {
        if thick.contains(sv3(p + beyond.normalize_or_zero() * hair * 0.5)) {
            return Ok(thick);
        }
    }
    kernel::thicken(sheet, -hair, false).map_err(kerr)
}

/// A point on a sheet: the middle of its largest facet.
fn sheet_point(sheet: &Body) -> Option<DVec3> {
    let tol = mesh_deflection(super::diagonal_of(sheet)).max(1e-3);
    let m = sheet.tessellate(tol).ok()?;
    let mut best: Option<(f64, DVec3)> = None;
    for t in &m.triangles {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.positions.get(i as usize).copied().map(dv3).unwrap_or_default());
        let area = (b - a).cross(c - a).length();
        if best.is_none_or(|(x, _)| area > x) {
            best = Some((area, (a + b + c) / 3.0));
        }
    }
    best.map(|b| b.1)
}

/// A body as it stands before the step, with the name a refusal calls it by: the feature that
/// made it, as ScanRuler's builder names its bodies.
fn body_before(b: &Builder, part: &Part, id: BodyId, gone: &str) -> Result<(String, BodyState), String> {
    if id == NO_BODY {
        return Err(gone.into());
    }
    let at = b.resolve(id);
    let state = b.body(at).cloned().ok_or_else(|| gone.to_string())?;
    let name = part.feature(at.maker()).map_or_else(|| at.to_string(), |f| f.name.clone());
    Ok((name, state))
}

/// The sketch's size, which a prism reaching to a plane is run past: its regions' diagonal on
/// the sheet (10 mm for a face pulled, as ScanRuler's empty sketch).
fn sketch_reach(part: &Part, e: &ExtrudeFeature) -> f64 {
    if e.face.is_some() {
        return 10.0;
    }
    let Some(s) = part.sketch(e.sketch_id) else { return 10.0 };
    let mut lo = DVec2::splat(f64::INFINITY);
    let mut hi = DVec2::splat(f64::NEG_INFINITY);
    for x in &s.sketch.entities {
        for p in crate::sketch::polyline(&x.shape, 16) {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    if lo.x.is_finite() {
        (hi - lo).length()
    } else {
        10.0
    }
}

// ---- a face of a body, pulled as it stands ----

/// A face of a body pulled as it stands: the face is found again by its fingerprint on the body
/// as it stood before the step and its outline taken as the profile, in the face's own plane at
/// its centroid, swept along its outward normal — so the profile is whatever the face is by the
/// time the step runs, holes and rounds included, and an earlier feature that reshapes the face
/// reshapes the boss pulled from it. A face the fingerprint no longer finds fails the step by
/// name; one that is no longer flat fails it too.
fn pulled_face(b: &mut Builder, on: &FaceProfile) -> Result<(SectionFrame, Vec<Region2>), String> {
    let at = b.resolve(on.body);
    if b.body(at).is_none() {
        return Err("The body this extrude pulls a face of is gone.".into());
    }
    let index = b.find_faces(at, std::slice::from_ref(&on.face), &[]).map_err(|lost| format!("The face this extrude pulls is gone (was: {lost})."))?.first().copied().ok_or("The face this extrude pulls is gone.")?;
    let state = b.body(at).cloned().ok_or("The body this extrude pulls a face of is gone.")?;
    let t = state.topology();
    let face = t.faces.iter().find(|f| f.hash as usize == index).ok_or("The body this extrude pulls a face of is gone.")?;
    if face.fingerprint.kind != FaceKind::Plane {
        return Err("The face this extrude pulls is not flat any more.".into());
    }
    let n = face.fingerprint.dir.try_normalize().ok_or("The face this extrude pulls has no normal.")?;
    let frame = SectionFrame::sketch_on(face.fingerprint.centroid, n, None).ok_or("The face this extrude pulls has no normal.")?;
    let plane = kernel_plane(&frame)?;
    let edges: Vec<usize> = t.edges.iter().filter(|e| e.faces.contains(&(index as u32)) && !(e.faces.len() == 2 && e.faces[0] == e.faces[1])).map(|e| e.hash as usize).collect();
    let exact = state.shape.planar_edges(&plane, &edges);
    let mut segs: Vec<Seg2> = Vec::new();
    for (k, e) in edges.iter().enumerate() {
        match exact.get(k).cloned().flatten() {
            Some(s) => segs.extend(s),
            None => {
                // Read off the edge as drawn where the kernel cannot give it exactly.
                let Some(info) = t.edges.iter().find(|x| x.hash as usize == *e) else { continue };
                let pts: Vec<SVec2> = info.polyline.iter().map(|p| {
                    let q = frame.to_sheet(*p);
                    SVec2::new(q.x, q.y)
                }).collect();
                segs.extend(pts.windows(2).filter(|w| w[0].dist(w[1]) > 1e-9).map(|w| Seg2::Line { a: w[0], b: w[1] }));
            }
        }
    }
    let region = region_of_segments(segs, state.diagonal()).ok_or("The face this extrude pulls has no closed outline to pull.")?;
    Ok((frame, vec![region]))
}

/// Segments that close loops, chained end to end, as a region: the largest loop its outline,
/// counter-clockwise, the rest its holes, clockwise. `None` where they do not close.
fn region_of_segments(mut pool: Vec<Seg2>, size: f64) -> Option<Region2> {
    let tol = (size * 1e-6).max(1e-6) * 10.0;
    let mut loops: Vec<Loop2> = Vec::new();
    while let Some(first) = pool.pop() {
        let mut lp = vec![first];
        let start = first.start();
        let mut guard = 0usize;
        while lp.last().is_some_and(|s| s.end().dist(start) > tol) {
            guard += 1;
            if guard > 100_000 {
                return None;
            }
            let end = lp.last()?.end();
            let next = pool.iter().position(|s| s.start().dist(end) <= tol || s.end().dist(end) <= tol)?;
            let s = pool.swap_remove(next);
            lp.push(if s.start().dist(end) <= tol { s } else { s.reversed() });
        }
        loops.push(Loop2 { segs: lp });
    }
    if loops.is_empty() {
        return None;
    }
    loops.sort_by(|a, b| b.signed_area().abs().total_cmp(&a.signed_area().abs()));
    let outer = loops.remove(0).ccw();
    let holes = loops.into_iter().map(|l| l.ccw().reversed()).collect();
    Some(Region2 { outer, holes })
}

#[cfg(test)]
mod tests;
