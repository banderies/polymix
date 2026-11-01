// SPDX-License-Identifier: AGPL-3.0-only
//! The Loft feature's build: a solid through the profiles of two or more sketches, each its one
//! region or its largest, in the order given — straight between them (ruled) or one smooth skin
//! through them all. Ported from ScanRuler (`plugins/reverse/core/` `resolve.ts` the loft step,
//! `build.ts` loftPlaced and loftOf, `profiles.ts` loftProfile, loftProblem, liftRegions,
//! openEndsWords, `kernel.ts` loft, `loftFold.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only, on the SolidCraft kernel.
//!
//! The kernel's loft is used as it stands: through two sections it is straight, through three
//! or more one smooth surface for each piece of the outline. A ruled loft through three or more
//! is built here as a loft between each pair of neighbours, the spans joined into one solid —
//! what Open CASCADE's ruled loft is, a face per span. A smooth loft is read for a skin crossing
//! itself (`loft_fold`), as ScanRuler reads it: the kernel's own check passes it. Its faces are
//! read through `FoldFace` off one-face copies of each side face, the surface sampled over its
//! whole parameter box. Where the kernel could not make the smooth skin and fell back to
//! straight spans, the row says so.
//!
//! Also here, for the Sweep beside it: a sketch's profile as a loft or a sweep takes it, lifted
//! into the part, and the words for a profile that is not closed.

use std::collections::HashSet;

use glam::{DVec2, DVec3};
use solidcraft_geom::{Loop2, Plane, Seg2};
use solidcraft_kernel::{self as kernel, BoolOp, Body, FaceSurface};

use super::{absent_sketch, dv3, kernel_plane, kerr, sv3, Builder};
use crate::corners::open_ends;
use crate::frame::SectionFrame;
use crate::js_fmt::{num, round_to, to_precision_num};
use crate::loft_fold::{loft_fold, loft_fold_refusal, sections_normal, FoldFace, LoftFoldOptions, NamedSection, UvBounds};
use crate::loop_piece::{piece_points, LoopPiece, Profile, PIECE_STEP};
use crate::model::{Feature, FeatureId, LoftFeature, Part};
use crate::regions::Region;
use crate::sketch::{AnchorAt, Shape, Sketch};
use crate::stations::STATION_MIN_PIECE_MM;

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &LoftFeature) -> Result<(), String> {
    if x.sketches.len() < 2 {
        return Err("A loft needs two or more sketches.".into());
    }
    let mut sections: Vec<Section> = Vec::new();
    for id in &x.sketches {
        let Some(placed) = placed_sketch(b, part, *id) else {
            return Err(absent(part, *id, b.step, "loft", "A sketch this loft runs through is gone."));
        };
        let region = loft_profile(&placed.name, placed.sketch, &placed.regions, "the loft")?.clone();
        sections.push(Section { name: placed.name, frame: placed.frame, profile: lift_region(&placed.frame, &region), region, sketch: placed.sketch.clone() });
    }
    // Checked before the kernel tries it: a sliver piece it can stop on, a section run off.
    if let Some(why) = loft_problem(&sections) {
        return Err(why);
    }
    let solid = loft_solid(b, &sections, x.ruled)?;
    b.place(x.op, x.target, vec![solid])?;
    Ok(())
}

// ---- a sketch's profile ----

/// A sketch as a loft or a sweep reads it: its name, the frame the build placed it in, its
/// pieces and its closed regions.
pub(crate) struct PlacedSketch<'a> {
    pub(crate) name: String,
    pub(crate) frame: SectionFrame,
    pub(crate) sketch: &'a Sketch,
    pub(crate) regions: Vec<Region>,
}

/// A sketch of the part, as the build placed it before the feature — `None` where it is not
/// there (further down, suppressed, gone).
pub(crate) fn placed_sketch<'a>(b: &Builder, part: &'a Part, id: FeatureId) -> Option<PlacedSketch<'a>> {
    let built = b.out.sketches.get(&id)?;
    let sketch = part.sketch(id)?;
    let name = part.feature(id).map_or_else(|| format!("Sketch {id}"), |f| f.name.clone());
    Some(PlacedSketch { name, frame: built.frame, sketch: &sketch.sketch, regions: built.regions.clone() })
}

/// Why a sketch a feature names is not there: further down the list or suppressed, said so —
/// else `gone`.
pub(crate) fn absent(part: &Part, id: FeatureId, feature: FeatureId, what: &str, gone: &str) -> String {
    let said = absent_sketch(part, id, feature, what);
    if said.starts_with("The sketch this ") {
        gone.to_string()
    } else {
        said
    }
}

/// One section of a loft: the sketch's name, its frame, the region it gives and the region
/// lifted into the part.
struct Section {
    name: String,
    frame: SectionFrame,
    region: Region,
    profile: Profile,
    sketch: Sketch,
}

/// A length or an area as an answer gives it: to the micron, and to two figures under a
/// hundredth, where microns are all there is (ScanRuler's `mmWords`).
pub(crate) fn mm_words(x: f64) -> String {
    if x.abs() < 0.01 {
        num(to_precision_num(x, 2))
    } else {
        num(round_to(x, 3))
    }
}

/// How many open ends a "no longer closed" answer names.
const OPEN_ENDS_NAMED: usize = 4;

/// A sketch's open ends in words, for an answer whose profile is not closed — each end, the end
/// nearest it and the gap, the smallest first. Empty when every end meets another
/// (ScanRuler's `openEndsWords`).
pub(crate) fn open_ends_words(sketch: &Sketch) -> String {
    let all = open_ends(sketch);
    // Two ends that miss each other are one miss, told once.
    let open: Vec<_> = all
        .iter()
        .enumerate()
        .filter(|(i, o)| !all[..*i].iter().any(|p| o.nearest.is_some_and(|n| n.0 == p.piece && n.1 == p.end) && p.nearest.is_some_and(|n| n.0 == o.piece && n.1 == o.end)))
        .map(|(_, o)| *o)
        .collect();
    if open.is_empty() {
        return String::new();
    }
    let end = |a: AnchorAt| if a == AnchorAt::A { "a" } else { "b" };
    let named: Vec<String> = open
        .iter()
        .take(OPEN_ENDS_NAMED)
        .map(|o| match o.nearest {
            Some((piece, at, gap)) => format!("piece {} end {} is {} mm from piece {piece} end {}", o.piece, end(o.end), mm_words(gap), end(at)),
            None => format!("piece {} end {} meets nothing", o.piece, end(o.end)),
        })
        .collect();
    let more = if open.len() > named.len() { format!(" (and {} more)", open.len() - named.len()) } else { String::new() };
    format!(" Its open ends: {}{more} — put the ends together (a coincident) or redraw the piece to end exactly there.", named.join("; "))
}

/// The regions that share pieces with `region`, it among them, round their outer loops — one
/// chain closing them all, crossing itself or cut across.
fn regions_of_one_chain<'a>(region: &'a Region, regions: &'a [Region]) -> Vec<&'a Region> {
    let mut group: Vec<&Region> = vec![region];
    let mut ids: HashSet<u32> = region.outer.iter().copied().collect();
    let mut grew = true;
    while grew {
        grew = false;
        for r in regions {
            if group.iter().any(|g| std::ptr::eq(*g, r)) || !r.outer.iter().any(|p| ids.contains(p)) {
                continue;
            }
            group.push(r);
            ids.extend(r.outer.iter().copied());
            grew = true;
        }
    }
    group
}

/// The profile a loft (or a sweep, `what`) takes from a sketch: its largest region — or why
/// there is none, or why it would be the wrong one: the chain it is closed by closes others
/// too, crossing itself (a folded station), and the feature would take a fragment of the
/// profile without a word (ScanRuler's `loftProfile`).
pub(crate) fn loft_profile<'a>(name: &str, sketch: &Sketch, regions: &'a [Region], what: &str) -> Result<&'a Region, String> {
    let Some(region) = regions.iter().reduce(|a, b| if b.area > a.area { b } else { a }) else {
        return Err(format!("{name} has no closed region for {what}.{}", open_ends_words(sketch)));
    };
    let one = regions_of_one_chain(region, regions);
    if one.len() > 1 {
        let mut areas: Vec<f64> = one.iter().map(|r| r.area).collect();
        areas.sort_by(|a, b| b.total_cmp(a));
        let areas: Vec<String> = areas.into_iter().map(mm_words).collect();
        return Err(format!(
            "{name}'s pieces close {} regions (areas {} mm²) — the profile crosses itself or is cut across, and {what} would take the largest alone. Read its pieces (reverse.feature.list) and fit or draw it again so it closes one region.",
            one.len(),
            areas.join(", ")
        ));
    }
    Ok(region)
}

/// One loop of a region, lifted from its sheet into the part.
fn lift_loop(frame: &SectionFrame, l: &Loop2) -> Vec<LoopPiece> {
    let at = |p: solidcraft_geom::Vec2| frame.to_world(DVec2::new(p.x, p.y));
    let mut out = Vec::new();
    for s in &l.segs {
        match *s {
            Seg2::Line { a, b } => out.push(LoopPiece::Line { a: at(a), b: at(b) }),
            Seg2::Arc { center, radius, start, sweep } => {
                let on = |t: f64| frame.to_world(DVec2::new(center.x + radius * t.cos(), center.y + radius * t.sin()));
                out.push(LoopPiece::Arc { a: on(start), m: on(start + sweep / 2.0), b: on(start + sweep) });
            }
            ref other => {
                // No sketch piece makes one; read as the short lines it is drawn with.
                let pts = Loop2 { segs: vec![other.clone()] }.polyline(0.01);
                for w in pts.windows(2) {
                    out.push(LoopPiece::Line { a: at(w[0]), b: at(w[1]) });
                }
            }
        }
    }
    out
}

/// A region lifted into the part (ScanRuler's `liftRegions` for one region).
pub(crate) fn lift_region(frame: &SectionFrame, region: &Region) -> Profile {
    Profile { outer: lift_loop(frame, &region.region.outer), holes: region.region.holes.iter().map(|h| lift_loop(frame, h)).collect() }
}

/// The middle of a profile's outline: the mean of the points it is drawn through — which end of
/// a path a sweep starts from (ScanRuler's `profileMiddle`).
pub(crate) fn profile_middle(profile: &Profile) -> DVec3 {
    let points: Vec<DVec3> = profile.outer.iter().flat_map(|p| piece_points(p, PIECE_STEP)).collect();
    points.iter().fold(DVec3::ZERO, |s, p| s + *p) / points.len().max(1) as f64
}

/// A loft's profile spanning more than this many times its neighbours' (or, at an end, its
/// neighbour's times the loft's own growth there) has run off: two pieces met far out, nothing
/// of the part.
const LOFT_EXTENT_JUMP: f64 = 3.0;

/// A profile's extent: the diagonal of the box round its outline.
fn profile_extent(profile: &Profile) -> f64 {
    let pts: Vec<DVec3> = profile.outer.iter().flat_map(|p| piece_points(p, PIECE_STEP)).collect();
    let lo = pts.iter().fold(DVec3::splat(f64::INFINITY), |m, p| m.min(*p));
    let hi = pts.iter().fold(DVec3::splat(f64::NEG_INFINITY), |m, p| m.max(*p));
    (hi - lo).length()
}

/// What would make a loft through these sections fail in the kernel, or come out wrong without
/// a word, checked before it is tried — or `None`: a piece of a section under
/// STATION_MIN_PIECE_MM, or a section far larger than its neighbours (a corner met far off).
/// A section that crosses itself is `loft_profile`'s to refuse (ScanRuler's `loftProblem`).
fn loft_problem(sections: &[Section]) -> Option<String> {
    for s in sections {
        for e in &s.sketch.entities {
            if e.construction {
                continue;
            }
            let (length, kind) = match e.shape {
                Shape::Line { a, b } => ((b - a).length(), "a line"),
                Shape::Arc { r, sweep, .. } => (r * sweep.abs(), "an arc"),
                _ => continue,
            };
            if length < STATION_MIN_PIECE_MM {
                return Some(format!(
                    "{}'s piece {} ({kind}) is {} mm long — under {} mm: no piece of a section, and one the kernel can stop on or fold at. Take it out (reverse.sketch.edit remove) and join its neighbours, or fit that section again.",
                    s.name,
                    e.id,
                    mm_words(length),
                    num(STATION_MIN_PIECE_MM)
                ));
            }
        }
    }
    let extents: Vec<f64> = sections.iter().map(|s| profile_extent(&s.profile)).collect();
    let n = extents.len();
    let one = |x: f64| num(round_to(x, 1));
    if n >= 3 {
        for i in 0..n {
            let near: Vec<f64> = [i.checked_sub(1).and_then(|k| extents.get(k)), extents.get(i + 1)].into_iter().flatten().copied().collect();
            // At an end, a jump past the loft's own growth into it.
            let growth = if i == 0 {
                extents[1] / extents[2]
            } else if i == n - 1 {
                extents[n - 2] / extents[n - 3]
            } else {
                1.0
            };
            let most = near.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if extents[i] > LOFT_EXTENT_JUMP * growth.max(1.0) * most {
                let near: Vec<String> = near.into_iter().map(one).collect();
                return Some(format!(
                    "{} spans {} mm across, its neighbours {} mm — a piece of it runs far off, likely two nearly parallel pieces met far out. Read its pieces (reverse.feature.list) and fit it again, pointed with within or facing.",
                    sections[i].name,
                    one(extents[i]),
                    near.join(" and ")
                ));
            }
        }
    }
    None
}

// ---- the solid ----

/// The loft through the sections' outer loops, in order: ruled — straight between each pair of
/// neighbours — or smooth, then read for a skin that crosses itself.
fn loft_solid(b: &mut Builder, sections: &[Section], ruled: bool) -> Result<Body, String> {
    let planes: Vec<(Plane, Loop2)> = sections.iter().map(|s| kernel_plane(&s.frame).map(|p| (p, s.region.region.outer.clone()))).collect::<Result<_, _>>()?;
    if ruled && planes.len() > 2 {
        let mut acc: Option<Body> = None;
        for w in planes.windows(2) {
            let span = kernel::loft(w).map_err(kerr)?;
            acc = Some(match acc {
                None => span,
                Some(a) => kernel::boolean(&a, &span, BoolOp::Union).map_err(kerr)?.ok_or("The loft's spans could not be joined into one solid.")?,
            });
        }
        return acc.ok_or_else(|| "A loft needs two or more sketches.".to_string());
    }
    let solid = kernel::loft(&planes).map_err(kerr)?;
    if ruled {
        return Ok(solid);
    }
    let profiles: Vec<Profile> = sections.iter().map(|s| s.profile.clone()).collect();
    let normal = sections_normal(&profiles);
    let faces = fold_faces(&solid, sections, normal);
    if sections.len() >= 3 && !faces.is_empty() && !faces.iter().filter(|f| !f.plane).all(|f| f.whole) {
        b.warn("The kernel could not make one smooth skin through these sections: the loft is straight between them.");
        return Ok(solid);
    }
    if let Some(fold) = loft_fold(&faces, &LoftFoldOptions { normal, faces: None, trimmed: false }) {
        let named: Vec<NamedSection> = sections.iter().map(|s| NamedSection { name: s.name.clone(), profile: s.profile.clone() }).collect();
        return Err(loft_fold_refusal(&fold, &named));
    }
    Ok(solid)
}

// ---- the faces the fold check reads ----

/// How many points a side face is sampled at each way, over its whole parameter box: odd
/// multiples of 1/80 are the middles of the fold check's FOLD_CUTS cuts, multiples of 1/8 its
/// steps across a face, so both fall on samples.
const FOLD_GRID: usize = 81;

/// A face of a loft as the fold check reads it: its surface sampled over its parameter box, u
/// across the sections and v along the loft (turned round where the kernel's surface runs the
/// other way), both read 0 to 1.
pub(crate) struct GridFace {
    hash: u64,
    plane: bool,
    /// `grid[i][j]`: the point at u = i / (n − 1), v = j / (n − 1).
    grid: Vec<Vec<DVec3>>,
    /// The face runs from the first section to the last.
    whole: bool,
}

impl GridFace {
    fn at(&self, i: usize, j: usize) -> DVec3 {
        self.grid.get(i).and_then(|r| r.get(j)).copied().unwrap_or_default()
    }
}

impl FoldFace for GridFace {
    fn hash(&self) -> u64 {
        self.hash
    }

    fn is_plane(&self) -> bool {
        self.plane
    }

    fn uv_bounds(&self) -> UvBounds {
        UvBounds { u_min: 0.0, u_max: 1.0, v_min: 0.0, v_max: 1.0 }
    }

    fn point_on_surface(&self, u: f64, v: f64) -> DVec3 {
        let n = self.grid.len().max(2);
        let m = self.grid.first().map_or(2, |r| r.len().max(2));
        let fu = u.clamp(0.0, 1.0) * (n - 1) as f64;
        let fv = v.clamp(0.0, 1.0) * (m - 1) as f64;
        let (i, j) = ((fu.floor() as usize).min(n - 2), (fv.floor() as usize).min(m - 2));
        let (su, sv) = (fu - i as f64, fv - j as f64);
        let a = self.at(i, j).lerp(self.at(i + 1, j), su);
        let b = self.at(i, j + 1).lerp(self.at(i + 1, j + 1), su);
        a.lerp(b, sv)
    }

    fn is_out(&self, _u: f64, _v: f64) -> bool {
        false
    }
}

/// The faces of a loft the fold check reads: each face copied out on its own and its surface
/// sampled; the caps (planes) marked as such. A face whose copy or samples fail is left out.
fn fold_faces(solid: &Body, sections: &[Section], normal: Option<DVec3>) -> Vec<GridFace> {
    let middles: Vec<DVec3> = sections.iter().map(|s| profile_middle(&s.profile)).collect();
    let (Some(first), Some(last)) = (middles.first().copied(), middles.last().copied()) else { return Vec::new() };
    // The way along the loft: the sections' common normal, else first middle to last.
    let along = normal.or_else(|| (last - first).try_normalize()).unwrap_or(DVec3::Z);
    let span = middles.iter().map(|m| m.dot(along)).fold(f64::NEG_INFINITY, f64::max) - middles.iter().map(|m| m.dot(along)).fold(f64::INFINITY, f64::min);
    let size = super::diagonal_of(solid);
    let Ok(mesh) = solid.display_mesh(super::mesh_deflection(size)) else { return Vec::new() };
    // A point on each face: the middle of its largest triangle.
    let mut on: Vec<Option<(f64, DVec3)>> = vec![None; solid.face_count()];
    for (t, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| mesh.positions.get(i as usize).copied().map(dv3).unwrap_or_default());
        let area = (b - a).cross(c - a).length();
        if let Some(slot) = on.get_mut(*f as usize) {
            if slot.is_none_or(|(x, _)| area > x) {
                *slot = Some((area, (a + b + c) / 3.0));
            }
        }
    }
    let mut out = Vec::new();
    for (i, point) in on.iter().enumerate() {
        let Some((_, p)) = point else { continue };
        if matches!(kernel::face_surface(solid, i), Some(FaceSurface::Plane { .. })) {
            out.push(GridFace { hash: i as u64, plane: true, grid: Vec::new(), whole: true });
            continue;
        }
        let Ok(copy) = kernel::copy_faces(solid, &[sv3(*p)]) else { continue };
        if copy.face_count() != 1 {
            continue;
        }
        let Ok(rows) = kernel::sample_grid(&copy, FOLD_GRID, FOLD_GRID) else { continue };
        let mut grid: Vec<Vec<DVec3>> = rows.into_iter().map(|r| r.into_iter().map(dv3).collect()).collect();
        // A cap the kernel holds on a spline surface is a plane all the same: flat throughout.
        if flat(&grid, size) {
            out.push(GridFace { hash: i as u64, plane: true, grid: Vec::new(), whole: true });
            continue;
        }
        let plane = false;
        grid = oriented(grid, along);
        let length = spread(grid.first().map_or(&[][..], |r| r.as_slice()), along);
        out.push(GridFace { hash: i as u64, plane, grid, whole: span <= 0.0 || length >= 0.9 * span });
    }
    out
}

/// How far points spread along a way.
fn spread(pts: &[DVec3], along: DVec3) -> f64 {
    let h = pts.iter().map(|q| q.dot(along));
    h.clone().fold(f64::NEG_INFINITY, f64::max) - h.fold(f64::INFINITY, f64::min)
}

/// A side face's samples laid as the fold check reads them: v along the loft — the first row
/// across lies in a section, the first column runs the loft's length — and from the first
/// section's end to the last's, on every face alike, so a cut at one v is at the one place along
/// the loft on each face. Turned round where the surface has u and v the other way, and each
/// way reversed where it runs back.
fn oriented(mut grid: Vec<Vec<DVec3>>, along: DVec3) -> Vec<Vec<DVec3>> {
    let across: Vec<DVec3> = grid.iter().filter_map(|r| r.first().copied()).collect();
    let down: &[DVec3] = grid.first().map_or(&[], |r| r.as_slice());
    if spread(&across, along) > spread(down, along) {
        let n = grid.iter().map(Vec::len).min().unwrap_or(0);
        grid = (0..n).map(|j| grid.iter().map(|r| r[j]).collect()).collect();
    }
    let ends = grid.first().and_then(|r| Some((*r.first()?, *r.last()?)));
    if let Some((a, b)) = ends {
        if a.dot(along) > b.dot(along) {
            for r in &mut grid {
                r.reverse();
            }
        }
    }
    grid
}

/// Whether sampled points lie in one plane, to a millionth of `size`: through the first point,
/// square to the widest pair of directions the samples span from it.
fn flat(grid: &[Vec<DVec3>], size: f64) -> bool {
    let pts: Vec<DVec3> = grid.iter().flatten().copied().collect();
    let Some(&o) = pts.first() else { return true };
    let far = pts.iter().copied().max_by(|a, b| (*a - o).length_squared().total_cmp(&(*b - o).length_squared())).unwrap_or(o);
    let d1 = far - o;
    let Some(normal) = pts.iter().map(|p| d1.cross(*p - o)).max_by(|a, b| a.length_squared().total_cmp(&b.length_squared())).and_then(|n| n.try_normalize()) else { return true };
    let tol = (size * 1e-6).max(1e-9);
    pts.iter().all(|p| (*p - o).dot(normal).abs() <= tol)
}

#[cfg(test)]
mod tests {
    //! The parts of ScanRuler's `plugins/reverse/tests/reLoftChecks.test.ts` and
    //! `reLoftFold.test.ts` a loft's own build answers, on the SolidCraft kernel.
    use super::*;
    use crate::build::{build, volume};
    use crate::frame::WorldAxis;
    use crate::model::{BodyOp, FeatureKind, SketchFeature};
    use crate::regions::find_regions;

    fn sketch_at(part: &mut Part, name: &str, z: f64, draw: impl FnOnce(&mut Sketch)) -> FeatureId {
        let mut s = Sketch::default();
        draw(&mut s);
        part.add_named(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, z, s)), name.into())
    }

    fn square(part: &mut Part, name: &str, z: f64, side: f64) -> FeatureId {
        sketch_at(part, name, z, |s| {
            s.add_rectangle(DVec2::splat(-side / 2.0), DVec2::splat(side / 2.0));
        })
    }

    fn loft(part: &mut Part, sketches: Vec<FeatureId>, ruled: bool) -> FeatureId {
        part.add(FeatureKind::Loft(LoftFeature { sketches, ruled, op: BodyOp::New, target: None }))
    }

    fn error(part: &Part, id: FeatureId) -> Option<String> {
        build(part).errors.get(&id).cloned()
    }

    fn lines(s: &mut Sketch, corners: &[[f64; 2]]) {
        for i in 0..corners.len() {
            let (a, b) = (corners[i], corners[(i + 1) % corners.len()]);
            s.add_shape(Shape::Line { a: DVec2::from(a), b: DVec2::from(b) });
        }
    }

    #[test]
    fn refuses_a_section_that_crosses_itself_giving_the_areas_and_lofts_a_ring_as_one() {
        let mut bow = Sketch::default();
        lines(&mut bow, &[[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]]);
        let regions = find_regions(&bow);
        let said = loft_profile("Bow", &bow, &regions, "the loft").err().unwrap_or_default();
        assert!(said.starts_with("Bow's pieces close 2 regions (areas 25, 25 mm²) — the profile crosses itself or is cut across, and the loft would take the largest alone."), "{said}");
        let said = loft_profile("Bow", &bow, &regions, "the sweep").err().unwrap_or_default();
        assert!(said.contains("the sweep would take the largest alone"), "{said}");
        let mut ring = Sketch::default();
        ring.add_rectangle(DVec2::splat(-10.0), DVec2::splat(10.0));
        ring.add_shape(Shape::Circle { c: DVec2::ZERO, r: 4.0 });
        let regions = find_regions(&ring);
        let r = loft_profile("Ring", &ring, &regions, "the loft");
        assert_eq!(r.map(|r| r.holes.len()), Ok(1));
    }

    #[test]
    fn refuses_a_sliver_piece_and_a_section_run_far_off_and_lofts_a_steady_funnel() {
        // A side split 0.01 mm from its corner.
        let mut part = Part::default();
        let low = square(&mut part, "Low", 0.0, 10.0);
        let mid = sketch_at(&mut part, "Mid", 10.0, |s| lines(s, &[[-5.0, -5.0], [-4.99, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]));
        let high = square(&mut part, "High", 20.0, 10.0);
        let l = loft(&mut part, vec![low, mid, high], false);
        let said = error(&part, l).unwrap_or_default();
        assert!(said.starts_with("Mid's piece 1 (a line) is 0.01 mm long — under 0.05 mm: no piece of a section"), "{said}");
        // A middle section ten times its neighbours, and a last one past the loft's own growth.
        let spans = |sides: [f64; 3]| {
            let mut part = Part::default();
            let a = square(&mut part, "Low", 0.0, sides[0]);
            let b = square(&mut part, "Mid", 10.0, sides[1]);
            let c = square(&mut part, "High", 20.0, sides[2]);
            let l = loft(&mut part, vec![a, b, c], false);
            error(&part, l)
        };
        let said = spans([10.0, 100.0, 10.0]).unwrap_or_default();
        assert!(said.starts_with("Mid spans 141.4 mm across, its neighbours 14.1 and 14.1 mm — a piece of it runs far off"), "{said}");
        let said = spans([10.0, 11.0, 100.0]).unwrap_or_default();
        assert!(said.starts_with("High spans 141.4 mm across"), "{said}");
        assert_eq!(spans([10.0, 20.0, 40.0]), None);
        let mut part = Part::default();
        let a = square(&mut part, "Low", 0.0, 5.0);
        let c = square(&mut part, "High", 10.0, 100.0);
        let l = loft(&mut part, vec![a, c], false);
        assert_eq!(error(&part, l), None);
    }

    #[test]
    fn names_a_sketch_further_down_the_list_or_suppressed_rather_than_calling_it_gone() {
        let mut part = Part::default();
        let low = square(&mut part, "Low", 0.0, 10.0);
        let l = loft(&mut part, vec![low, 3], false);
        let high = square(&mut part, "Top", 20.0, 10.0);
        assert_eq!(high, 3);
        let said = error(&part, l).unwrap_or_default();
        assert!(said.starts_with("Sketch 3 (Top) stands after this loft — a feature takes only the sketches above it. Move it above"), "{said}");
        let mut part = Part::default();
        let low = square(&mut part, "Low", 0.0, 10.0);
        let high = square(&mut part, "Top", 20.0, 10.0);
        if let Some(f) = part.feature_mut(high) {
            f.flags.suppressed = true;
        }
        let l = loft(&mut part, vec![low, high], false);
        let said = error(&part, l).unwrap_or_default();
        assert!(said.starts_with("Sketch 2 (Top) is suppressed — build it again"), "{said}");
        let mut part = Part::default();
        let low = square(&mut part, "Low", 0.0, 10.0);
        let l = loft(&mut part, vec![low, 9], false);
        assert_eq!(error(&part, l).as_deref(), Some("A sketch this loft runs through is gone."));
    }

    #[test]
    fn a_profile_no_longer_closed_names_its_open_ends() {
        let mut part = Part::default();
        let low = square(&mut part, "Low", 0.0, 10.0);
        // One side's end moved 0.099 mm out from its corner.
        let open = sketch_at(&mut part, "Open", 10.0, |s| {
            s.add_rectangle(DVec2::splat(-5.0), DVec2::splat(5.0));
            if let Some(e) = s.entities.get_mut(1) {
                if let Shape::Line { b, .. } = &mut e.shape {
                    let out = DVec2::new((b.x).signum(), (b.y).signum());
                    *b += out * (0.099 / std::f64::consts::SQRT_2);
                }
            }
            s.constraints.clear();
        });
        let l = loft(&mut part, vec![low, open], false);
        let said = error(&part, l).unwrap_or_default();
        assert!(said.starts_with("Open has no closed region for the loft. Its open ends: piece "), "{said}");
        assert!(said.contains(" is 0.099 mm from piece "), "{said}");
        assert!(said.ends_with("— put the ends together (a coincident) or redraw the piece to end exactly there."), "{said}");
    }

    #[test]
    fn lofts_two_squares_into_a_frustum_and_three_ruled_or_smooth() {
        // Two squares 20 and 10 on a side, 10 apart: a frustum of a pyramid.
        let mut part = Part::default();
        let a = square(&mut part, "Low", 0.0, 20.0);
        let b = square(&mut part, "High", 10.0, 10.0);
        loft(&mut part, vec![a, b], true);
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let v = built.bodies.first().and_then(|x| volume(&x.body)).unwrap_or(0.0);
        let frustum = 10.0 / 3.0 * (400.0 + 100.0 + 200.0);
        assert!((v - frustum).abs() < 1e-3 * frustum, "{v}");
        // Three: ruled is two frusta joined, one solid; smooth is a solid too.
        for ruled in [true, false] {
            let mut part = Part::default();
            let a = square(&mut part, "Low", 0.0, 20.0);
            let b = square(&mut part, "Mid", 10.0, 10.0);
            let c = square(&mut part, "High", 20.0, 20.0);
            loft(&mut part, vec![a, b, c], ruled);
            let built = build(&part);
            assert!(built.errors.is_empty(), "{ruled}: {:?}", built.errors);
            assert_eq!(built.bodies.len(), 1);
            let v = built.bodies.first().and_then(|x| volume(&x.body)).unwrap_or(0.0);
            if ruled {
                assert!((v - 2.0 * frustum).abs() < 1e-3 * frustum, "{v}");
                assert_eq!(built.bodies[0].body.face_count(), 10, "two spans of four sides and two caps");
            } else {
                assert!(v > 1000.0 && v < 2.0 * frustum, "{v}");
                assert!(built.warnings.is_empty(), "{:?}", built.warnings);
            }
        }
    }

    /// The hump of ScanRuler's `reLoftFold.test.ts`: sections of a moulded boss, the one at
    /// z 24 three mm above its neighbour and far wider, so the smooth skin through all five
    /// overshoots between them.
    fn hump(part: &mut Part, with_24: bool) -> Vec<FeatureId> {
        const HUMP: [(f64, f64); 5] = [(8.0, 1.4), (21.0, 1.82), (24.0, 2.7), (33.0, 0.55), (38.0, 0.47)];
        let mut ids = Vec::new();
        for (z, w) in HUMP {
            if !with_24 && z == 24.0 {
                continue;
            }
            let id = sketch_at(part, &format!("Hump z{z}"), z, |s| {
                // A front edge of half-width w at y 12, slanting sides to an upright
                // rectangle's, an arc round the back.
                let corners = [[-10.0, 0.0], [-10.0, 6.0], [-w, 12.0], [w, 12.0], [10.0, 6.0], [10.0, 0.0]];
                for k in 0..corners.len() - 1 {
                    s.add_shape(Shape::Line { a: DVec2::from(corners[k]), b: DVec2::from(corners[k + 1]) });
                }
                s.add_shape(Shape::Arc { c: DVec2::ZERO, r: 10.0, start: std::f64::consts::PI, sweep: std::f64::consts::PI });
            });
            ids.push(id);
        }
        ids
    }

    #[test]
    fn a_smooth_loft_that_folds_is_refused_naming_the_close_pair_and_where_it_folds() {
        let mut part = Part::default();
        let ids = hump(&mut part, true);
        let l = loft(&mut part, ids, false);
        let built = build(&part);
        // Open CASCADE's skin folds at z 9–17.6 and 35–37; the kernel's through the same
        // sections folds over a stretch of its own between the same pair.
        let said = built.errors.get(&l).cloned().unwrap_or_default();
        assert!(said.starts_with("The smooth loft crosses itself at z "), "{said}");
        assert!(said.contains(", between Hump z8 and Hump z21 (folded over z "), "{said}");
        assert!(said.ends_with("): Hump z21 and Hump z24 are 3 mm apart and differ — drop or move one, or make the loft ruled."), "{said}");
    }

    #[test]
    fn lays_every_side_face_with_v_along_the_loft_from_the_first_section() {
        // A face sampled u along the loft, and one with v running back from the last section:
        // both read with v from the first section (z 0) to the last (z 10), u across.
        let at = |u: f64, z: f64| DVec3::new(u, 0.0, z);
        let n = 5;
        let steps = |k: usize| k as f64 / (n - 1) as f64;
        let u_along: Vec<Vec<DVec3>> = (0..n).map(|i| (0..n).map(|j| at(steps(j), 10.0 * steps(i))).collect()).collect();
        let v_back: Vec<Vec<DVec3>> = (0..n).map(|i| (0..n).map(|j| at(steps(i), 10.0 * (1.0 - steps(j)))).collect()).collect();
        for grid in [u_along, v_back] {
            let g = oriented(grid, DVec3::Z);
            for (i, row) in g.iter().enumerate() {
                for (j, p) in row.iter().enumerate() {
                    assert!((p.z - 10.0 * steps(j)).abs() < 1e-12 && (p.x - steps(i)).abs() < 1e-12, "{i} {j}: {p:?}");
                }
            }
        }
        // A face's v as the fold check reads it: a cut at one v is at one height on every face.
        let face = GridFace { hash: 0, plane: false, grid: oriented((0..n).map(|i| (0..n).map(|j| at(steps(i), 10.0 * (1.0 - steps(j)))).collect()).collect(), DVec3::Z), whole: true };
        assert!((face.point_on_surface(0.3, 0.25).z - 2.5).abs() < 1e-12);
    }

    #[test]
    fn builds_the_folding_hump_ruled_and_without_the_z_24_section() {
        for (what, with_24, ruled) in [("ruled", true, true), ("4 sections", false, false)] {
            let mut part = Part::default();
            let ids = hump(&mut part, with_24);
            let l = loft(&mut part, ids, ruled);
            let built = build(&part);
            assert!(!built.errors.contains_key(&l), "{what}: {:?}", built.errors);
            assert!(!built.warnings.contains_key(&l), "{what}: {:?}", built.warnings);
            let v = built.bodies.first().and_then(|b| volume(&b.body)).unwrap_or(0.0);
            assert!(v > 9000.0 && v < 12000.0, "{what}: {v}");
        }
    }
}
