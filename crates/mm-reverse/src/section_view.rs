// SPDX-License-Identifier: AGPL-3.0-only
//! The section view: the parts cut open at a plane, to look inside them, and the caps that
//! close the closed ones where the plane cuts. Ported from ScanRuler
//! (`src/state/sectionViewStore.ts`, the caps of `src/viewer/sliceCap.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! A way of looking, like the see-through switches beside it in the view bar — held for the
//! session, never saved, no step of the history. The plane is laid across a coordinate plane
//! or an element with a direction, slid along it, tilted by the gizmo's rings and flipped to
//! take away the other half. Like a section, it is frozen where it was laid.
//!
//! The caps: where a closed shell is cut, the loops the plane cuts through it are filled
//! (even-odd, so a hole in a wall stays a hole) and drawn flat in the part's cut colour; an
//! open scan stays a shell.

use glam::{DQuat, DVec2, DVec3};

use crate::frame::{world_cut_axis, CutAxis, SectionFrame, WorldAxis};
use crate::model::{Element, ElementGeometry, ElementId};
use crate::slice::slice_mesh;

/// The colours the section view fills a cut face with: a different one for each part, no red.
pub const CUT_COLORS: [[u8; 3]; 8] = [[0xbe, 0x7b, 0x4d], [0x5b, 0x8d, 0xb8], [0x7a, 0xa2, 0x5c], [0x55, 0x66, 0xb0], [0x8d, 0x6f, 0xc0], [0xc9, 0xa8, 0x3c], [0x4f, 0xb3, 0xa5], [0xb0, 0x69, 0x9e]];

/// Each cap's place: the scan's first, the reference's next, and a layer's parts from `LAYERS`.
pub const CAP_SLOT_SCAN: usize = 0;
pub const CAP_SLOT_LAYERS: usize = 2;

/// The cut colour of the part in a slot.
pub fn cut_color(slot: usize) -> [u8; 3] {
    CUT_COLORS[slot % CUT_COLORS.len()]
}

/// The cut face of the part under a sketch: a quiet slate, dark and light.
pub const SKETCH_CAP_DARK: [u8; 3] = [0x47, 0x53, 0x5f];
pub const SKETCH_CAP_LIGHT: [u8; 3] = [0xb9, 0xbf, 0xc6];

/// What the plane is laid across.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionRef {
    World(WorldAxis),
    Element(ElementId),
}

/// The section view's state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionView {
    pub on: bool,
    pub reference: SectionRef,
    /// The line the plane slides along.
    pub axis: Option<CutAxis>,
    /// The reference's own direction — what a tilt is read against.
    pub ref_dir: Option<DVec3>,
    pub offset: f64,
    pub flipped: bool,
    pub centre: DVec3,
}

impl Default for SectionView {
    fn default() -> Self {
        SectionView { on: false, reference: SectionRef::World(WorldAxis::Z), axis: None, ref_dir: None, offset: 0.0, flipped: false, centre: DVec3::ZERO }
    }
}

/// The line an element gives to cut across: a plane's normal, a cylinder's, a cone's or a
/// line's axis, a circle's normal, a section's normal — through its centre.
pub fn element_cut_axis(e: &Element) -> Option<CutAxis> {
    let (origin, dir) = match &e.geometry {
        ElementGeometry::Plane { center, normal, .. } => (*center, *normal),
        ElementGeometry::Cylinder { center, axis, .. } | ElementGeometry::Cone { center, axis, .. } => (*center, *axis),
        ElementGeometry::Line { center, dir, .. } => (*center, *dir),
        ElementGeometry::Circle { center, normal, .. } => (*center, *normal),
        ElementGeometry::Section { frame } => (frame.origin, frame.normal),
        _ => return None,
    };
    let dir = dir.try_normalize()?;
    let (u, _) = crate::linalg::ortho_basis(dir);
    Some(CutAxis { origin, dir, u })
}

/// Whether the half the arrow points into is away from the camera.
fn faces_away(dir: DVec3, view: DVec3) -> bool {
    dir.dot(view) > 0.0
}

impl SectionView {
    fn ref_axis(&self, r: SectionRef, elements: &[Element]) -> Option<CutAxis> {
        match r {
            SectionRef::World(a) => Some(world_cut_axis(a, self.centre)),
            SectionRef::Element(id) => elements.iter().find(|e| e.id == id).and_then(element_cut_axis),
        }
    }

    /// Cut the parts open across the coordinate plane most square to `view` — the way the
    /// camera looks — through `centre`, taking away the half toward the camera.
    pub fn open(&mut self, view: DVec3, centre: DVec3) {
        let a = view.abs();
        let axis = if a.x >= a.y && a.x >= a.z {
            WorldAxis::X
        } else if a.y >= a.z {
            WorldAxis::Y
        } else {
            WorldAxis::Z
        };
        let cut = world_cut_axis(axis, centre);
        *self = SectionView { on: true, reference: SectionRef::World(axis), axis: Some(cut), ref_dir: Some(cut.dir), offset: centre.dot(cut.dir), flipped: faces_away(cut.dir, view), centre };
    }

    pub fn close(&mut self) {
        self.on = false;
    }

    /// Lay the plane across a coordinate plane or an element. Chosen again, a reference keeps
    /// its offset: that is how a tilted plane is squared up.
    pub fn set_ref(&mut self, r: SectionRef, view: Option<DVec3>, elements: &[Element]) {
        let Some(axis) = self.ref_axis(r, elements) else { return };
        let offset = if self.reference == r && self.axis.is_some() {
            self.offset
        } else {
            match r {
                SectionRef::World(a) => self.centre.dot(a.dir()),
                SectionRef::Element(_) => 0.0,
            }
        };
        self.reference = r;
        self.axis = Some(axis);
        self.ref_dir = Some(axis.dir);
        self.offset = offset;
        if let Some(v) = view {
            self.flipped = faces_away(axis.dir, v);
        }
    }

    pub fn set_offset(&mut self, mm: f64) {
        if mm.is_finite() {
            self.offset = mm;
        }
    }

    pub fn flip(&mut self) {
        self.flipped = !self.flipped;
    }

    /// The line turned by a ring: about `about` (through the plane's centre) by `angle` radians.
    pub fn turn(&mut self, about: DVec3, angle: f64) {
        let (Some(axis), Some(gizmo)) = (self.axis, self.frames().map(|f| f.1)) else { return };
        let Some(about) = about.try_normalize() else { return };
        let q = DQuat::from_axis_angle(about, angle);
        let dir = (q * axis.dir).normalize();
        let u = (q * gizmo.u).normalize();
        // The plane keeps passing through where its centre was.
        let centre = gizmo.origin;
        let origin = centre - dir * centre.dot(dir);
        self.axis = Some(CutAxis { origin, dir, u });
        self.offset = centre.dot(dir) - origin.dot(dir);
    }

    /// How far the rings have tilted the plane off what it was laid across, degrees.
    pub fn tilt_deg(&self) -> f64 {
        match (self.ref_dir, self.axis) {
            (Some(r), Some(a)) => {
                let d = r.dot(a.dir).abs().clamp(0.0, 1.0);
                let t = d.acos().to_degrees();
                if t < 1e-6 {
                    0.0
                } else {
                    t
                }
            }
            _ => 0.0,
        }
    }

    /// Where the view cuts: `cut` takes away the half its normal points to, `gizmo` is the plane
    /// with its normal along the arrow. `None` while it is off.
    pub fn frames(&self) -> Option<(SectionFrame, SectionFrame)> {
        if !self.on {
            return None;
        }
        let gizmo = SectionFrame::along(self.axis?, self.offset, None)?;
        let cut = if self.flipped { SectionFrame { normal: -gizmo.normal, v: -gizmo.v, ..gizmo } } else { gizmo };
        Some((cut, gizmo))
    }
}

// ---- the caps ----

/// Triangles covering the loops by the even-odd rule, by slabs between the vertex heights:
/// inside each slab the loops' edges do not cross, so the crossings pair up into trapezoids.
pub fn fill_loops(loops: &[Vec<DVec2>]) -> Vec<[DVec2; 3]> {
    let mut edges: Vec<(DVec2, DVec2)> = Vec::new();
    for l in loops {
        let n = l.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let (a, b) = (l[i], l[(i + 1) % n]);
            if (a.y - b.y).abs() > 1e-12 {
                edges.push(if a.y < b.y { (a, b) } else { (b, a) });
            }
        }
    }
    let mut ys: Vec<f64> = edges.iter().flat_map(|(a, b)| [a.y, b.y]).collect();
    ys.sort_by(f64::total_cmp);
    ys.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    edges.sort_by(|a, b| a.0.y.total_cmp(&b.0.y));
    let x_at = |e: &(DVec2, DVec2), y: f64| e.0.x + (y - e.0.y) / (e.1.y - e.0.y) * (e.1.x - e.0.x);
    let mut next = 0;
    let mut span: Vec<(DVec2, DVec2)> = Vec::new();
    let mut out = Vec::new();
    for w in ys.windows(2) {
        let (y0, y1) = (w[0], w[1]);
        let ym = (y0 + y1) * 0.5;
        while let Some(e) = edges.get(next) {
            if e.0.y > ym {
                break;
            }
            span.push(*e);
            next += 1;
        }
        span.retain(|e| e.1.y >= ym);
        span.sort_by(|a, b| x_at(a, ym).total_cmp(&x_at(b, ym)));
        for pair in span.chunks(2) {
            let [l, r] = pair else { continue };
            let (l0, l1, r0, r1) = (DVec2::new(x_at(l, y0), y0), DVec2::new(x_at(l, y1), y1), DVec2::new(x_at(r, y0), y0), DVec2::new(x_at(r, y1), y1));
            if (r0.x - l0.x).abs() > 1e-12 {
                out.push([l0, r0, r1]);
            }
            if (r1.x - l1.x).abs() > 1e-12 {
                out.push([l0, r1, l1]);
            }
        }
    }
    out
}

/// The cap a plane closes a mesh with where it cuts it: the closed loops of the cut, filled,
/// stood up on the plane. `keep` says which triangles may be capped (a closed shell's).
pub fn mesh_cap(positions: &[DVec3], triangles: &[[u32; 3]], frame: &SectionFrame, keep: Option<&[bool]>) -> Vec<[DVec3; 3]> {
    let kept: Vec<[u32; 3]>;
    let tris = match keep {
        Some(k) => {
            kept = triangles.iter().zip(k).filter(|(_, k)| **k).map(|(t, _)| *t).collect();
            &kept[..]
        }
        None => triangles,
    };
    let cut = slice_mesh(positions, tris, frame.origin, frame.normal, 0.0);
    let loops: Vec<Vec<DVec2>> = cut
        .project(frame)
        .into_iter()
        .filter(|c| c.len() >= 4 && c.first().zip(c.last()).is_some_and(|(a, b)| a.distance(*b) < 1e-9))
        .map(|mut c| {
            c.pop();
            c
        })
        .collect();
    fill_loops(&loops).into_iter().map(|t| [frame.to_world(t[0]), frame.to_world(t[1]), frame.to_world(t[2])]).collect()
}

/// Which triangles of a mesh lie in closed shells — the only ones a cap can close: welded by
/// where their points lie, a piece none of whose sides is a single triangle's.
pub fn closed_triangles(positions: &[DVec3], triangles: &[[u32; 3]]) -> Vec<bool> {
    use std::collections::HashMap;
    let mut key: HashMap<[u64; 3], u32> = HashMap::new();
    let id: Vec<u32> = positions
        .iter()
        .map(|p| {
            let n = key.len() as u32;
            *key.entry([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).or_insert(n)
        })
        .collect();
    let mut parent: Vec<usize> = (0..triangles.len()).collect();
    fn find(p: &mut [usize], mut a: usize) -> usize {
        while p[a] != a {
            p[a] = p[p[a]];
            a = p[a];
        }
        a
    }
    let mut sides: HashMap<(u32, u32), (usize, usize)> = HashMap::new();
    for (t, tri) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (id[tri[k] as usize], id[tri[(k + 1) % 3] as usize]);
            let s = if a < b { (a, b) } else { (b, a) };
            let e = sides.entry(s).or_insert((t, 0));
            e.1 += 1;
            let other = e.0;
            let (ra, rb) = (find(&mut parent, t), find(&mut parent, other));
            parent[rb] = ra;
        }
    }
    let mut open = vec![false; triangles.len()];
    for (s, (t, n)) in &sides {
        if *n == 1 && s.0 != s.1 {
            let r = find(&mut parent, *t);
            open[r] = true;
        }
    }
    (0..triangles.len()).map(|t| !open[find(&mut parent, t)]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_across_the_plane_most_square_to_the_view_taking_away_the_near_half() {
        let mut s = SectionView::default();
        s.open(DVec3::new(0.2, -0.9, 0.1), DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(s.reference, SectionRef::World(WorldAxis::Y));
        assert!((s.offset - 2.0).abs() < 1e-12);
        assert!(!s.flipped, "the camera looks down -Y: the +Y half is toward it");
        let (cut, gizmo) = s.frames().unwrap();
        assert!((cut.normal - DVec3::Y).length() < 1e-12);
        assert!((gizmo.origin.y - 2.0).abs() < 1e-12);
        s.flip();
        assert!((s.frames().unwrap().0.normal + DVec3::Y).length() < 1e-12);
        s.close();
        assert!(s.frames().is_none());
    }

    #[test]
    fn a_tilt_is_read_off_the_reference_and_choosing_it_again_squares_it() {
        let mut s = SectionView::default();
        s.open(DVec3::Z, DVec3::ZERO);
        s.set_offset(5.0);
        s.turn(DVec3::X, 10f64.to_radians());
        assert!((s.tilt_deg() - 10.0).abs() < 1e-9);
        let centre = s.frames().unwrap().1.origin;
        assert!((centre.z - 5.0).abs() < 1e-9, "the plane turns about its centre");
        s.set_ref(SectionRef::World(WorldAxis::Z), None, &[]);
        assert_eq!(s.tilt_deg(), 0.0);
        assert!((s.offset - s.offset).abs() < 1e-12);
    }

    #[test]
    fn the_loops_are_filled_even_odd() {
        let sq = |h: f64| vec![DVec2::new(-h, -h), DVec2::new(h, -h), DVec2::new(h, h), DVec2::new(-h, h)];
        let area = |tris: &[[DVec2; 3]]| tris.iter().map(|t| ((t[1] - t[0]).perp_dot(t[2] - t[0])).abs() / 2.0).sum::<f64>();
        assert!((area(&fill_loops(&[sq(2.0)])) - 16.0).abs() < 1e-9);
        assert!((area(&fill_loops(&[sq(2.0), sq(1.0)])) - 12.0).abs() < 1e-9, "a hole stays a hole");
    }

    #[test]
    fn a_closed_cube_is_capped_and_an_open_one_is_not() {
        let m = mm_core::primitives::cube(10.0, 2);
        let pos: Vec<DVec3> = m.positions().to_vec();
        let tris: Vec<[u32; 3]> = m.triangle_ids().map(|t| m.triangle(t)).collect();
        let frame = SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let keep = closed_triangles(&pos, &tris);
        assert!(keep.iter().all(|k| *k));
        let cap = mesh_cap(&pos, &tris, &frame, Some(&keep));
        let area: f64 = cap.iter().map(|t| (t[1] - t[0]).cross(t[2] - t[0]).length() / 2.0).sum();
        assert!((area - 100.0).abs() < 1e-6, "{area}");
        // The top taken off: open, nothing capped.
        let open: Vec<[u32; 3]> = tris.iter().copied().filter(|t| t.iter().any(|v| pos[*v as usize].z < 4.999)).collect();
        let keep = closed_triangles(&pos, &open);
        assert!(keep.iter().all(|k| !*k));
        assert!(mesh_cap(&pos, &open, &frame, Some(&keep)).is_empty());
    }
}
