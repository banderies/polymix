//! Mic pad: a flat-topped block standing off an object's surface, to hold a contact
//! microphone (with ultrasound gel between) over a stenosis of a printed vessel.
//!
//! The pad is laid out on a straight line through a point of the surface, the *centre*. Its
//! *base* is a rectangle `length` × `width` centred on that line, in the plane square to the
//! surface's mean normal around the centre (the pad's *axis*). From the base the pad rises
//! `height` along the axis to a flat top, its sides leaning out by the draft angle, so the top
//! is `length + 2·height·tan(draft)` by `width + 2·height·tan(draft)` ([`PadParams::top`]).
//!
//! The pad joins the surface over its *contact* ([`PadPlan::contact`]): the base, cut back to
//! where the surface faces the pad. It stops where the surface turns away from the pad more
//! steeply than `contact_deg` (past a vessel's crown, down its sides), so a pad wider than the
//! vessel does not wrap round it; from the contact's edge the pad's sides lean out to the
//! top's edge, a wedge. The contact's plane rests on the highest point of the surface under
//! it; below it the pad goes straight down to the surface and on into it (the *skirt*), so it
//! meets the surface all round the contact's outline.
//!
//! The skirt's underside follows the surface (a ray down the axis at every point of a grid
//! over the contact) and, as a Stand leg's root does ([`crate::stand`]), stops
//! [`PLUG_SHARE`] of the way to the next surface down the ray (out of the wall into a lumen,
//! …): united with a hollow vessel, the pad changes only its outer surface. [`plan_pad`] also
//! reports what would make the pad cut into the object instead: the object lying over the pad
//! ([`PadPlan::buried`]) or its sides reaching more than that share through the wall beside
//! the contact ([`PadPlan::cuts_in`]). Such pads should not be made.

use crate::stand::{root_axis, PLUG_SHARE};
use crate::{EditError, EditResult};
use mm_core::{measure, Bvh, DQuat, DVec3, Mesh, Ray};
use rayon::prelude::*;

/// A pad's size (model units, mm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadParams {
    /// Along the line.
    pub length: f64,
    /// Across the line.
    pub width: f64,
    /// From the base up to the top face.
    pub height: f64,
    /// Degrees each side leans out from the axis on the way up (negative: in).
    pub draft_deg: f64,
    /// The steepest the surface may turn away from the pad where the pad joins it (degrees,
    /// more than 0 and at most 90): the contact stops there ([`PadPlan::contact`]).
    pub contact_deg: f64,
}

impl Default for PadParams {
    fn default() -> Self {
        Self { length: 15.0, width: 5.0, height: 5.0, draft_deg: 30.0, contact_deg: 45.0 }
    }
}

impl PadParams {
    /// The steepest draft allowed either way (degrees).
    pub const MAX_DRAFT: f64 = 75.0;

    fn check(&self) -> EditResult<()> {
        for (v, what) in [(self.length, "length"), (self.width, "width"), (self.height, "height")] {
            if !(v > 0.0 && v.is_finite()) {
                return Err(EditError::InvalidParameter(format!("the pad's {what} must be positive (it is {v})")));
            }
        }
        if self.draft_deg.is_nan() || self.draft_deg.abs() > Self::MAX_DRAFT {
            return Err(EditError::InvalidParameter(format!("the pad's draft must be between −{0}° and {0}° (it is {1})", Self::MAX_DRAFT, self.draft_deg)));
        }
        if !(self.contact_deg > 0.0 && self.contact_deg <= 90.0) {
            return Err(EditError::InvalidParameter(format!("the pad's contact slope must be more than 0° and at most 90° (it is {})", self.contact_deg)));
        }
        let (l, w) = self.top();
        if l < 0.05 * self.length || w < 0.05 * self.width {
            return Err(EditError::InvalidParameter("the draft narrows the pad to nothing below its top: lower the height or the draft".into()));
        }
        Ok(())
    }

    /// The top face's size: (along the line, across it).
    pub fn top(&self) -> (f64, f64) {
        let grow = 2.0 * self.height * self.draft_deg.to_radians().tan();
        (self.length + grow, self.width + grow)
    }
}

/// A rectangle in the pad's plane about its centre: `x` along the line, `y` across it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadRect {
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
}

impl PadRect {
    fn about(length: f64, width: f64) -> Self {
        Self { x0: -0.5 * length, x1: 0.5 * length, y0: -0.5 * width, y1: 0.5 * width }
    }

    /// (Along the line, across it.)
    pub fn size(&self) -> (f64, f64) {
        (self.x1 - self.x0, self.y1 - self.y0)
    }

    fn short(&self) -> f64 {
        let (a, b) = self.size();
        a.min(b)
    }
}

/// The way the surface runs around `p`, square to `axis`: the direction least like the
/// normals of the surface within `radius` (the eigenvector of the least of their area-weighted
/// spread). Along a vessel that is its length. `None` where no one direction stands out (a
/// flat patch, a ball).
pub fn surface_run(mesh: &Mesh, bvh: &Bvh, p: DVec3, axis: DVec3, radius: f64) -> Option<DVec3> {
    let mut m = [[0.0f64; 3]; 3];
    let mut total = 0.0;
    bvh.query_sphere(mesh, p, radius, |t| {
        let c = mesh.tri_cross(t);
        let a = c.length();
        if a > 0.0 {
            let n = (c / a).to_array();
            for (i, row) in m.iter_mut().enumerate() {
                for (j, x) in row.iter_mut().enumerate() {
                    *x += a * n[i] * n[j];
                }
            }
            total += a;
        }
    });
    if total <= 0.0 {
        return None;
    }
    let (values, vectors) = mm_core::linalg::sym_eigen3(m);
    // The normals must spread one way (the surface curves round), much more than the other.
    if !(values[1] > 0.005 * total && values[0] < 0.5 * values[1]) {
        return None;
    }
    let d = vectors[0] - axis * vectors[0].dot(axis);
    (d.length() > 0.5).then(|| d.normalize())
}

/// The world axis most square to `axis`, turned square to it.
fn world_run(axis: DVec3) -> DVec3 {
    let w = [DVec3::X, DVec3::Y, DVec3::Z].into_iter().min_by(|a, b| a.dot(axis).abs().total_cmp(&b.dot(axis).abs())).unwrap_or(DVec3::X);
    (w - axis * w.dot(axis)).normalize()
}

/// What a pad will be, worked out from the object, its centre, its angle and its size
/// ([`plan_pad`]). Heights are along the axis above the base's plane.
#[derive(Clone, Debug)]
pub struct PadPlan {
    pub params: PadParams,
    /// The centre: on the object's surface.
    pub center: DVec3,
    /// The way the pad stands off the surface: the object's mean outward normal around the
    /// centre.
    pub axis: DVec3,
    /// The line's direction (the pad's length runs along it), square to the axis.
    pub along: DVec3,
    /// Across the line (the pad's width): `axis × along`.
    pub across: DVec3,
    /// The direction the line's angle is measured from: the way the surface runs at the centre
    /// (a vessel's length), or a world axis where it runs no way in particular.
    pub reference: DVec3,
    /// Whether `reference` was found from the surface.
    pub follows_surface: bool,
    /// Where the pad joins the surface: the base, cut back where the surface turns away from
    /// the pad more steeply than [`PadParams::contact_deg`] (or is another part of the object,
    /// or is not there). From its edge the pad's sides lean out to the top's.
    pub contact: PadRect,
    /// The contact's plane (the base's): its height above the centre (more than 0 where the
    /// surface rises under the pad: it rests on the highest point).
    pub lift: f64,
    /// How deep the pad's underside sits below the surface at the centre. All over the contact
    /// it stops short of the next surface down the axis (at most [`PLUG_SHARE`] of the way
    /// there).
    pub embed: f64,
    /// The object's thickness under the centre, along the axis (`None`: no far side near it).
    pub wall: Option<f64>,
    /// Share of the contact with no surface under it (0..1).
    pub overhang: f64,
    /// The object lies over the pad (another branch close by): it would cut into it.
    pub buried: bool,
    /// The pad's sides or top would reach more than [`PLUG_SHARE`] of the way through the
    /// object's wall beside the contact (it could open into a lumen there).
    pub cuts_in: bool,
    /// The least room between the pad's sides and the object beside the contact (negative
    /// where they overlap; infinite with no object under them).
    pub clearance: f64,
    /// The contact's grid lines, along the line and across it (one of each at 0, the centre).
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// At each point of the contact's grid (row by row across): how far below its plane the
    /// surface is (`None`: no surface there), and the pad's underside.
    surface: Vec<Option<f64>>,
    bottom: Vec<f64>,
    /// The surface's height above the centre along the whole line (`None`: no surface there).
    line_heights: Vec<(f64, Option<f64>)>,
}

/// A surface hit by a ray down the axis.
enum Hit {
    Surface {
        /// Height above the plane the ray is measured from.
        height: f64,
        /// To the next surface on down the ray.
        through: Option<f64>,
    },
    /// The ray started inside the object.
    Buried,
    Miss,
}

/// Cast a ray from `origin` down `-axis` (no further than `reach`) and on to the next surface
/// past the first one met (no further than `wall_reach`; not at all for 0). Heights are
/// relative to `origin`.
fn cast(mesh: &Mesh, bvh: &Bvh, origin: DVec3, axis: DVec3, reach: f64, wall_reach: f64) -> Hit {
    let ray = Ray::new(origin, -axis);
    let Some(hit) = bvh.raycast_filtered(mesh, &ray, reach, |_| true) else { return Hit::Miss };
    if mesh.tri_normal(hit.tid).dot(axis) <= 0.0 {
        return Hit::Buried;
    }
    if wall_reach <= 0.0 {
        return Hit::Surface { height: -hit.t, through: None };
    }
    // The hit triangle's neighbours are left out: one met at a shared corner is the same
    // surface.
    let corners = mesh.triangle(hit.tid);
    let on = Ray::new(hit.point, -axis);
    let next = bvh.raycast_range(mesh, &on, 1e-9 * reach.max(1.0), wall_reach, |t| !mesh.triangle(t).iter().any(|v| corners.contains(v)));
    Hit::Surface { height: -hit.t, through: next.map(|h| h.t) }
}

/// Grid lines from `lo` (below 0) to `hi` (above 0) about `step` apart, one of them at 0, and
/// that one's index (about `max` steps at most).
fn lines(lo: f64, hi: f64, step: f64, max: usize) -> (Vec<f64>, usize) {
    let step = step.max((hi - lo) / max as f64);
    let (a, b) = (((-lo / step).ceil() as usize).max(1), ((hi / step).ceil() as usize).max(1));
    let mut out: Vec<f64> = (0..a).map(|k| if k == 0 { lo } else { lo * (a - k) as f64 / a as f64 }).collect();
    out.extend((0..=b).map(|k| if k == b { hi } else { hi * k as f64 / b as f64 }));
    (out, a)
}

/// Where the pad joins the surface ([`PadPlan::contact`]): the largest rectangle about the
/// centre, within the base, over which the surface is there and is one surface (no sudden
/// rise, as to another branch), turning away from the pad by no more than `contact_deg` on the
/// way out from the centre. `height(x, y)` is the surface's height above the centre's plane
/// there (`None`: no surface, or something over it). Looked for on a grid of rays, then again
/// more finely about what was found while the grid is coarse beside it. Also the surface's
/// heights along the line, over the whole base's length.
fn find_contact(p: &PadParams, height: impl Fn(f64, f64) -> Option<f64> + Sync) -> (PadRect, Vec<(f64, Option<f64>)>) {
    let full = PadRect::about(p.length, p.width);
    let short = full.short();
    let tan = p.contact_deg.min(89.9).to_radians().tan();
    let (mut region, mut found, mut line) = (full, full, Vec::new());
    let mut step = short / 40.0;
    for round in 0..3 {
        let (xs, ic) = lines(region.x0, region.x1, step, 240);
        let (ys, jc) = lines(region.y0, region.y1, step, 240);
        let w = xs.len();
        let hs: Vec<Option<f64>> = (0..w * ys.len()).into_par_iter().map(|k| height(xs[k % w], ys[k / w])).collect();
        if round == 0 {
            line = xs.iter().enumerate().map(|(i, &x)| (x, hs[jc * w + i])).collect();
        }
        // A grid point the pad can join: on the surface, neither falling away too steeply nor
        // rising suddenly from its neighbours toward the centre.
        let ok = |i: usize, j: usize| {
            let Some(h) = hs[j * w + i] else { return false };
            if (i, j) == (ic, jc) {
                return h <= 0.05 * short;
            }
            let toward = [(i != ic).then(|| (if i > ic { i - 1 } else { i + 1 }, j)), (j != jc).then(|| (i, if j > jc { j - 1 } else { j + 1 }))];
            toward.into_iter().flatten().all(|(a, b)| {
                let d = (xs[a] - xs[i]).abs() + (ys[b] - ys[j]).abs();
                hs[b * w + a].is_some_and(|hn| hn - h <= tan * d && h - hn <= 3.0 * d)
            })
        };
        // The largest rectangle about the centre with every point joinable: for each run of
        // rows through the centre, the run of columns through it joinable over all of them.
        let h = ys.len();
        let can: Vec<bool> = (0..w * h).map(|k| ok(k % w, k / w)).collect();
        let mut best = (0.0, (ic, ic, jc, jc));
        let mut below = vec![true; w];
        for j0 in (0..=jc).rev() {
            for (i, b) in below.iter_mut().enumerate() {
                *b &= can[j0 * w + i];
            }
            if !below[ic] {
                break;
            }
            let mut cols = below.clone();
            for j1 in jc..h {
                if j1 > jc {
                    for (i, c) in cols.iter_mut().enumerate() {
                        *c &= can[j1 * w + i];
                    }
                }
                if !cols[ic] {
                    break;
                }
                let (mut i0, mut i1) = (ic, ic);
                while i0 > 0 && cols[i0 - 1] {
                    i0 -= 1;
                }
                while i1 + 1 < w && cols[i1 + 1] {
                    i1 += 1;
                }
                let area = (xs[i1] - xs[i0]) * (ys[j1] - ys[j0]);
                if area > best.0 {
                    best = (area, (i0, i1, j0, j1));
                }
            }
        }
        let (i0, i1, j0, j1) = best.1;
        found = PadRect { x0: xs[i0], x1: xs[i1], y0: ys[j0], y1: ys[j1] };
        let fine = found.short() / 40.0;
        if round == 2 || fine <= 0.0 || step <= 1.5 * fine {
            break;
        }
        // Again over what was found and a little past it, more finely.
        let widest = |v: &[f64]| v.windows(2).map(|s| s[1] - s[0]).fold(0.0, f64::max);
        let (gx, gy) = (2.0 * widest(&xs), 2.0 * widest(&ys));
        region = PadRect { x0: (found.x0 - gx).max(full.x0), x1: (found.x1 + gx).min(full.x1), y0: (found.y0 - gy).max(full.y0), y1: (found.y1 + gy).min(full.y1) };
        step = fine;
    }
    // Never less than a little about the centre (on a sharp ridge).
    let m = 0.05 * short;
    (PadRect { x0: found.x0.min(-m), x1: found.x1.max(m), y0: found.y0.min(-m), y1: found.y1.max(m) }, line)
}

/// An even count of about `size / step` (2 at least, `max` at most).
fn even_steps(size: f64, step: f64, max: usize) -> usize {
    let n = ((size / step).ceil() as usize).clamp(2, max);
    n + n % 2
}

/// Work out a pad centred on the surface point nearest `at`, its line turned `angle_deg`
/// about the axis from the way the surface runs there ([`PadPlan::reference`]).
pub fn plan_pad(mesh: &Mesh, bvh: &Bvh, at: DVec3, angle_deg: f64, p: &PadParams) -> EditResult<PadPlan> {
    p.check()?;
    if !angle_deg.is_finite() {
        return Err(EditError::InvalidParameter("the pad's angle must be a number".into()));
    }
    let near = bvh.nearest(mesh, at, f64::INFINITY).ok_or_else(|| EditError::Failed("the object has no surface".into()))?;
    let center = near.point;
    let (short, long) = (p.length.min(p.width), p.length.max(p.width));
    let axis = root_axis(mesh, bvh, center, mesh.tri_normal(near.tid), 0.5 * short);
    let found = surface_run(mesh, bvh, center, axis, (0.5 * p.length).max(p.width));
    let follows_surface = found.is_some();
    let mut reference = found.unwrap_or_else(|| world_run(axis));
    // A fixed sign (either way is the same line), so the plan does not flip about.
    let big = reference.abs().max_element();
    if reference.to_array().iter().find(|c| c.abs() == big).is_some_and(|c| *c < 0.0) {
        reference = -reference;
    }
    let along = DQuat::from_axis_angle(axis, angle_deg.to_radians()) * reference;
    let along = (along - axis * along.dot(axis)).normalize();
    let across = axis.cross(along);

    // Rays down the axis from above the top.
    let start = p.height + 0.5 * long;
    let reach = start + long;
    let wall_reach = 2.0 * long;
    let from = |x: f64, y: f64| center + along * x + across * y + axis * start;
    let (contact, line_heights) = find_contact(p, |x, y| match cast(mesh, bvh, from(x, y), axis, reach, 0.0) {
        Hit::Surface { height, .. } => Some(height + start),
        _ => None,
    });

    // The surface under the contact.
    let short = contact.short();
    let step = short / 10.0;
    let (xs, ic) = lines(contact.x0, contact.x1, step, 160);
    let (ys, jc) = lines(contact.y0, contact.y1, step, 160);
    let (nx, ny) = (xs.len() - 1, ys.len() - 1);
    let hits: Vec<Hit> = (0..(nx + 1) * (ny + 1))
        .into_par_iter()
        .map(|k| match cast(mesh, bvh, from(xs[k % (nx + 1)], ys[k / (nx + 1)]), axis, reach, wall_reach) {
            Hit::Surface { height, through } => Hit::Surface { height: height + start, through },
            h => h,
        })
        .collect();
    let centre = jc * (nx + 1) + ic;
    let cap = 0.15 * short;
    let plug_of = |through: Option<f64>| through.map_or(cap, |t| (PLUG_SHARE * t).clamp(0.05 * cap, cap));
    let heights: Vec<Option<f64>> = hits.iter().map(|h| if let Hit::Surface { height, .. } = h { Some(*height) } else { None }).collect();
    let (mut min_plug, mut buried) = (f64::INFINITY, false);
    for h in &hits {
        match *h {
            Hit::Surface { through, .. } => min_plug = min_plug.min(plug_of(through)),
            Hit::Buried => buried = true,
            Hit::Miss => {}
        }
    }
    if heights.iter().all(Option::is_none) {
        return Err(EditError::Failed("there is no surface under the pad".into()));
    }
    // The surface the centre is on: the grid points reached from it without a sudden rise
    // (any fall, or a rise of up to 3 in 1 between neighbours). A surface met past a sudden
    // rise is another part of the object (a branch close by).
    let mut joined = vec![false; hits.len()];
    match heights[centre] {
        // Something over the centre (the ray down to it met another surface first).
        Some(h) if h > 0.05 * short => buried = true,
        Some(_) => {
            joined[centre] = true;
            let mut todo = vec![centre];
            while let Some(k) = todo.pop() {
                let (i, j, h) = (k % (nx + 1), k / (nx + 1), heights[k].unwrap_or(0.0));
                let mut near = Vec::with_capacity(4);
                if i > 0 {
                    near.push((k - 1, xs[i] - xs[i - 1]));
                }
                if i < nx {
                    near.push((k + 1, xs[i + 1] - xs[i]));
                }
                if j > 0 {
                    near.push((k - nx - 1, ys[j] - ys[j - 1]));
                }
                if j < ny {
                    near.push((k + nx + 1, ys[j + 1] - ys[j]));
                }
                for (n, d) in near {
                    if !joined[n] && heights[n].is_some_and(|hn| hn - h <= 3.0 * d) {
                        joined[n] = true;
                        todo.push(n);
                    }
                }
            }
        }
        None => buried = true,
    }
    if buried {
        joined = heights.iter().map(Option::is_some).collect();
    }
    // The contact rests on the highest point of that surface; any other part of the object
    // rising over it would be cut into.
    let rest = heights.iter().zip(&joined).filter(|(_, j)| **j).filter_map(|(h, _)| *h).fold(f64::NEG_INFINITY, f64::max);
    buried |= heights.iter().zip(&joined).any(|(h, j)| !*j && h.is_some_and(|h| h > rest));
    // Its plane sits a little below the highest point, so the skirt crosses the surface there
    // rather than ending on it.
    let lift = rest - 0.5 * min_plug.min(cap);
    let lowest = hits.iter().filter_map(|h| if let Hit::Surface { height, .. } = h { Some(lift - height) } else { None }).fold(0.0, f64::max);
    let (mut surface, mut bottom, mut missed) = (Vec::with_capacity(hits.len()), Vec::with_capacity(hits.len()), 0usize);
    for h in &hits {
        let (s, b) = match *h {
            // (Never above the contact's plane, even under another part of the object: such a
            // pad is not to be made, but it is still a solid.)
            Hit::Surface { height, through } => (Some(lift - height), (lift - height + plug_of(through)).max(0.5 * min_plug)),
            Hit::Buried => (None, cap),
            Hit::Miss => {
                missed += 1;
                (None, lowest + cap)
            }
        };
        surface.push(s);
        bottom.push(b);
    }
    let (embed, wall) = match hits[centre] {
        Hit::Surface { through, .. } => (plug_of(through), through),
        _ => (0.0, None),
    };

    // Beside the contact: the object must not reach far into the pad's sides (rays down from
    // the top over the part of it outside the contact).
    let (mut clearance, mut cuts_in) = (f64::INFINITY, false);
    let (tl, tw) = p.top();
    let top_rect = PadRect::about(tl, tw);
    // How far up the side is where it passes over (x, y), as a share of the height: from
    // there up the pad is solid.
    let share = |v: f64, lo: f64, hi: f64, top_lo: f64, top_hi: f64| {
        if v > hi {
            (v - hi) / (top_hi - hi)
        } else if v < lo {
            (lo - v) / (lo - top_lo)
        } else {
            0.0
        }
    };
    let (mx, my) = (even_steps(tl, step, 200), even_steps(tw, step, 200));
    let side: Vec<(f64, f64, f64)> = (0..(mx + 1) * (my + 1))
        .filter_map(|k| {
            let (x, y) = (top_rect.x0 + tl * (k % (mx + 1)) as f64 / mx as f64, top_rect.y0 + tw * (k / (mx + 1)) as f64 / my as f64);
            let up = share(x, contact.x0, contact.x1, top_rect.x0, top_rect.x1).max(share(y, contact.y0, contact.y1, top_rect.y0, top_rect.y1));
            (up > 1e-9).then_some((x, y, p.height * up.min(1.0)))
        })
        .collect();
    let top = p.height + 1e-3 * short;
    let found: Vec<Hit> = side.par_iter().map(|&(x, y, low)| cast(mesh, bvh, center + along * x + across * y + axis * (lift + top), axis, top - low + long, wall_reach)).collect();
    for (&(_, _, low), h) in side.iter().zip(&found) {
        match *h {
            Hit::Surface { height, through } => {
                // How far into the object the side reaches here.
                let into = (top + height) - low;
                clearance = clearance.min(-into);
                cuts_in |= through.is_some_and(|t| into > PLUG_SHARE * t);
            }
            // The object over the side or the top.
            Hit::Buried => buried = true,
            Hit::Miss => {}
        }
    }
    Ok(PadPlan {
        params: *p,
        center,
        axis,
        along,
        across,
        reference,
        follows_surface,
        contact,
        lift,
        embed,
        wall,
        overhang: missed as f64 / hits.len() as f64,
        buried,
        cuts_in,
        clearance,
        xs,
        ys,
        surface,
        bottom,
        line_heights,
    })
}

impl PadPlan {
    /// The point at (along, across, height above the contact's plane).
    pub fn world(&self, x: f64, y: f64, z: f64) -> DVec3 {
        self.center + self.along * x + self.across * y + self.axis * (self.lift + z)
    }

    /// Grid point (i, j) of the contact on the surface (on its plane where there is none).
    fn on_surface(&self, i: usize, j: usize) -> DVec3 {
        self.world(self.xs[i], self.ys[j], -self.surface[j * self.xs.len() + i].unwrap_or(0.0))
    }

    /// The line the pad is laid out on, along the surface from end to end (its whole length,
    /// whatever the contact).
    pub fn line(&self) -> Vec<DVec3> {
        self.line_heights.iter().map(|&(x, h)| self.world(x, 0.0, h.map_or(0.0, |h| h - self.lift))).collect()
    }

    /// The line's ends, on the surface.
    pub fn ends(&self) -> (DVec3, DVec3) {
        let line = self.line();
        (line[0], line[line.len() - 1])
    }

    /// Whether the contact is smaller than the base (the surface turns away under it).
    pub fn cut_back(&self) -> bool {
        let (l, w) = self.contact.size();
        l < self.params.length - 1e-9 || w < self.params.width - 1e-9
    }

    /// The contact's outline on the surface (where the pad meets it), as a closed loop.
    pub fn footprint(&self) -> Vec<DVec3> {
        boundary(self.xs.len() - 1, self.ys.len() - 1).into_iter().map(|(i, j)| self.on_surface(i, j)).collect()
    }

    /// The contact's corners (in its plane) and the top's, in the same order round.
    pub fn corners(&self) -> ([DVec3; 4], [DVec3; 4]) {
        let (c, (tl, tw)) = (self.contact, self.params.top());
        let s = [(false, false), (true, false), (true, true), (false, true)];
        let base = s.map(|(a, b)| self.world(if a { c.x1 } else { c.x0 }, if b { c.y1 } else { c.y0 }, 0.0));
        let top = s.map(|(a, b)| self.world(if a { 0.5 } else { -0.5 } * tl, if b { 0.5 } else { -0.5 } * tw, self.params.height));
        (base, top)
    }

    /// The centre of the top face.
    pub fn top_center(&self) -> DVec3 {
        self.world(0.0, 0.0, self.params.height)
    }

    /// Whether the pad can be made as it is (see [`PadPlan::buried`], [`PadPlan::cuts_in`]).
    pub fn ok(&self) -> bool {
        !self.buried && !self.cuts_in
    }
}

/// The base grid's outer points as (i, j), counter-clockwise seen from the top (along, then
/// across, then back, then down the near side).
fn boundary(nx: usize, ny: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(2 * (nx + ny));
    out.extend((0..nx).map(|i| (i, 0)));
    out.extend((0..ny).map(|j| (nx, j)));
    out.extend((0..nx).map(|i| (nx - i, ny)));
    out.extend((0..ny).map(|j| (0, ny - j)));
    out
}

/// Build the pad `plan` describes: a closed, manifold mesh (one face group, 0) with a flat
/// top, flat sides and an underside following the surface over the contact, edges about a
/// tenth of the contact's shorter side long.
pub fn pad_mesh(plan: &PadPlan) -> EditResult<Mesh> {
    let p = &plan.params;
    let (nx, ny) = (plan.xs.len() - 1, plan.ys.len() - 1);
    let c = plan.contact;
    let (tl, tw) = p.top();
    // The top's point over the contact's (x, y): the contact stretched to the top.
    let to_top = |x: f64, y: f64| (-0.5 * tl + tl * (x - c.x0) / (c.x1 - c.x0), -0.5 * tw + tw * (y - c.y0) / (c.y1 - c.y0));
    let step = c.short() / 10.0;
    let deepest = plan.bottom.iter().copied().fold(0.0, f64::max);
    let ns = ((deepest / step).ceil() as usize).clamp(1, 64);
    let slant = [0.5 * tl - c.x1, c.x0 + 0.5 * tl, 0.5 * tw - c.y1, c.y0 + 0.5 * tw].into_iter().map(|d| d.hypot(p.height)).fold(0.0, f64::max);
    let nf = ((slant / step).ceil() as usize).clamp(1, 128);
    let g = (nx + 1) * (ny + 1);
    let mut pos: Vec<DVec3> = Vec::with_capacity(2 * g + 2 * (nx + ny) * (ns + nf));
    // The underside, then the top, grid point (i, j) at j·(nx + 1) + i.
    for j in 0..=ny {
        for i in 0..=nx {
            pos.push(plan.world(plan.xs[i], plan.ys[j], -plan.bottom[j * (nx + 1) + i]));
        }
    }
    for j in 0..=ny {
        for i in 0..=nx {
            let (x, y) = to_top(plan.xs[i], plan.ys[j]);
            pos.push(plan.world(x, y, p.height));
        }
    }
    // The sides: a column of points over each outer grid point, from the underside up the
    // skirt to the contact's plane (row ns), then up the side to the top (row ns + nf).
    let ring = boundary(nx, ny);
    let rows = ns + nf;
    let inner = rows - 1;
    let wall0 = pos.len();
    for &(i, j) in &ring {
        let (x, y) = (plan.xs[i], plan.ys[j]);
        let (tx, ty) = to_top(x, y);
        let b = plan.bottom[j * (nx + 1) + i];
        for r in 1..rows {
            pos.push(if r <= ns {
                plan.world(x, y, -b * (1.0 - r as f64 / ns as f64))
            } else {
                let u = (r - ns) as f64 / nf as f64;
                plan.world(x + (tx - x) * u, y + (ty - y) * u, p.height * u)
            });
        }
    }
    let under = |i: usize, j: usize| (j * (nx + 1) + i) as u32;
    let top = |i: usize, j: usize| (g + j * (nx + 1) + i) as u32;
    let at = |k: usize, r: usize| {
        let (i, j) = ring[k % ring.len()];
        match r {
            0 => under(i, j),
            r if r == rows => top(i, j),
            r => (wall0 + (k % ring.len()) * inner + r - 1) as u32,
        }
    };
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(4 * nx * ny + 4 * ring.len() * rows);
    // Quads (a, b, c, d) counter-clockwise seen from outside, split along the shorter diagonal.
    let mut quad = |a: u32, b: u32, c: u32, d: u32| {
        if pos[a as usize].distance_squared(pos[c as usize]) <= pos[b as usize].distance_squared(pos[d as usize]) {
            tris.extend([[a, b, c], [a, c, d]]);
        } else {
            tris.extend([[a, b, d], [b, c, d]]);
        }
    };
    for j in 0..ny {
        for i in 0..nx {
            quad(top(i, j), top(i + 1, j), top(i + 1, j + 1), top(i, j + 1));
            quad(under(i, j), under(i, j + 1), under(i + 1, j + 1), under(i + 1, j));
        }
    }
    for k in 0..ring.len() {
        for r in 0..rows {
            quad(at(k, r), at(k + 1, r), at(k + 1, r + 1), at(k, r + 1));
        }
    }
    let (mut mesh, _) = Mesh::from_triangles(&pos, &tris, None);
    if mesh.triangle_count() != tris.len() {
        return Err(EditError::Failed("the pad came out broken".into()));
    }
    if measure::volume(&mesh) < 0.0 {
        mesh.reverse_orientation();
    }
    Ok(mesh)
}
