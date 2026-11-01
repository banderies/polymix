// SPDX-License-Identifier: AGPL-3.0-only
//! The ways of looking at the part, in the bottom-left corner of the stage, and the maps the
//! scan is coloured with. Ported from ScanRuler (`plugins/reverse/app/useReversePlugin.tsx`,
//! `src/ui/ViewBar.tsx`, `src/ui/PartKey.tsx`, `src/ui/SectionViewKeys.tsx`,
//! `src/ui/MapLegend.tsx`, the view parts of `app/useReverseWorkspace.ts`,
//! `viewer/reScene.ts` and `viewer/coordinatePlanes.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! One row, read from the left: a key for each part — the scan, the bodies, the elements —
//! that shows it or puts it away, with a menu over it for how it is drawn; at the side the
//! Deviation key; last the section view, whose own row stands above while it is open. While a
//! sketch is open the row says what of the part lies under the sketch instead. The colour
//! scale of the map on the scan stands down the right edge.
//!
//! The maps: the live map (the scan near the ghost of the feature being set up, or near the
//! committed bodies on the Deviation key) and the curvature map (the scan by how sharply it
//! bends) — one at a time, the ghost's taking the surface over while it is up.

use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Instant;

use egui::{Align2, Color32, FontId, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Vec2};
use glam::{DVec3, Vec4};
use mm_reverse::build::{body_mesh, build_upto, BodyMesh};
use mm_reverse::compare::{live_map, live_map_scale, paint_value, sparse_summary, vertex_normals, ScanSide, Surface, SurfaceBody, Unfacing, DEFAULT_FACING_DEG, FAR_INSIDE_RGB, JET};
use mm_reverse::edge_lines::{edge_joins, EdgeJoin, OutlinedEdge, TangentEdges};
use mm_reverse::frame::{SectionFrame, WorldAxis};
use mm_reverse::mesh_graph::curvature::{curvature_range, curvature_scale, mean_curvature, CURVATURE_SMOOTHING};
use mm_reverse::mesh_graph::MeshGraph;
use mm_reverse::model::{BodyId, BodyOp, FeatureKind};
use mm_reverse::section_view::{closed_triangles, cut_color, element_cut_axis, mesh_cap, SectionRef, SectionView, CAP_SLOT_LAYERS, CAP_SLOT_SCAN, SKETCH_CAP_DARK};
use mm_reverse::silhouette::{silhouette_segments, SilhouetteView};

use super::{Mode, ReverseWorkspace, ScanPart};
use crate::renderer::ColorVertex;
use crate::tools::Viewport;

/// The workspace's colour tokens (dark).
pub(crate) mod tok {
    use egui::Color32;
    pub const PANEL: Color32 = Color32::from_rgb(0x25, 0x28, 0x2c);
    pub const WELL: Color32 = Color32::from_rgb(0x1e, 0x21, 0x24);
    pub const LINE: Color32 = Color32::from_rgb(0x38, 0x3c, 0x41);
    pub const LINE_STRONG: Color32 = Color32::from_rgb(0x4b, 0x50, 0x57);
    pub const INK: Color32 = Color32::from_rgb(0xe4, 0xe6, 0xe8);
    pub const DIM: Color32 = Color32::from_rgb(0xa0, 0xa4, 0xa8);
    pub const FAINT: Color32 = Color32::from_rgb(0x76, 0x7a, 0x7f);
    pub const ACCENT: Color32 = Color32::from_rgb(0x3f, 0x93, 0xd6);
    pub const ACCENT_D: Color32 = Color32::from_rgb(0x62, 0xaa, 0xe4);
    pub const ACCENT_SOFT: Color32 = Color32::from_rgb(0x22, 0x3c, 0x55);
    pub const AMBER: Color32 = Color32::from_rgb(0xe2, 0xa4, 0x3b);
    /// The axis colours: X, Y, Z.
    pub const AXIS: [Color32; 3] = [Color32::from_rgb(0xe5, 0x53, 0x4b), Color32::from_rgb(0x2e, 0x7d, 0x46), Color32::from_rgb(0x18, 0x77, 0xc0)];
}

/// How long the pointer rests on a key before its menu opens, and how long it may be off the
/// key and the menu before the menu closes, seconds.
const OPEN_DELAY: f64 = 0.14;
const CLOSE_DELAY: f64 = 0.26;
/// How long a ghost stands still before its map is read again, seconds.
const MAP_DEBOUNCE: f64 = 0.12;
/// How far the camera turns before the round faces' outlines are found again, degrees.
const SILHOUETTE_TURN_DEG: f64 = 0.2;
/// The gizmo's arrow, its standing off and its rings, pixels.
const ARROW_PX: f32 = 44.0;
const RING_PX: f32 = 70.0;

/// The tooltips and labels, as ScanRuler writes them.
mod words {
    pub const SCAN: &str = "Show the scan, or put it away. How it is drawn is in the menu over the key";
    pub const SCAN_NONE: &str = "No scan loaded";
    pub const SCAN_NO_BODY: &str = "Build a body first — until then there is only the scan to look at";
    pub const SCAN_TRANSPARENT: &str = "Draw the scan transparent — what lies inside it, the fitted elements and the pinned readings show instead of hiding behind the surface in front of them";
    pub const SCAN_MESH: &str = "Draw the triangle edges on the scan, to see how fine it is and where it is patchy. The mesh comes up as you zoom in: from a distance the triangles are smaller than a pixel, and the edges fade out rather than turning the part black";
    pub const SCAN_BACKFACES: &str = "Colour the far side of every triangle — holes in the scan and inverted normals stop looking like solid part";
    pub const SCAN_BACKFACES_NOT_YET: &str = "Colour the far side of every triangle — not in this view yet: its renderer shades the far side of a triangle as the near one";
    pub const CURVATURE: &str = "Colour the scan by how sharply it bends — flats green, convex rounds towards red, hollows towards blue, sharp edges dark. Where a flat ends and a round or a blend begins is where a surface region should end. While a feature is being set up its ghost takes the map over. Puts the Deviation key's map out";
    pub const CURVATURE_NO_SCAN: &str = "Load a scan first — the map reads its surface";
    pub const BODIES: &str = "Show the bodies, or put them away as a whole and leave the scan to stand alone — with the deviation map still on it, read against the bodies as they are. Each body's own eye in the panel takes it out of the model and the map instead";
    pub const BODIES_NONE: &str = "Build a body first";
    pub const BODIES_TRANSPARENT: &str = "Draw the bodies transparent — the scan, the elements and the bodies inside them show through. With the scan solid as well, each part shows where it stands proud of the other";
    pub const OVERLAPS: &str = "Draw where two bodies run into each other in hatched red — what they share, over the faces of both that would otherwise fight for the same pixels. Off for bodies that overlap on purpose, a modifier volume for a slicer; saved with the project";
    pub const TANGENT: &str = "How the edges are drawn where a round runs into a flat. A round face's seam and the cut a join leaves across one surface are never drawn";
    pub const TANGENT_DRAWN: &str = "Tangent edges drawn — where a round runs into a flat — as the sharp ones are, the way CAD draws them";
    pub const TANGENT_DIMMED: &str = "Tangent edges dimmed";
    pub const TANGENT_OFF: &str = "Tangent edges left out, and not picked: a round shows by its outline against the background";
    pub const ELEMENTS: &str = "Show the elements and sections measured in the 3D Measure workspace on the part — the planes, cylinders and cuts the model is built against";
    pub const ELEMENTS_NONE: &str = "Nothing measured yet — fit elements or cut sections in the 3D Measure workspace and they show here";
    pub const DEVIATION: &str = "Colour the scan by how far it lies from the bodies, on the deviation colour map at the range set in the feature boxes — outside the bodies up the ramp, inside down it — and violet where the scan lies deep inside them: a hole, a pocket or a slot the bodies lack. While a feature is being set up its ghost takes the map over. Puts the curvature colouring out";
    pub const DEVIATION_NONE: &str = "Build a body first — the map reads the scan against it";
    pub const SECTION: &str = "Cut the parts open at a plane to look inside — the scan, the reference and everything measured on them. Closed parts are closed where they are cut; an open scan stays a shell. Slide the plane with the arrow, tilt it with the rings";
    pub const NO_PART: &str = "The sketch on the bare stage, the 3D view under it put away";
    pub const CUT_PART: &str = "The part under the sketch, facing the sketch plane and following the sheet's pan and zoom, cut open at the plane where the plane runs through it — the half toward you taken away, so the slice being drawn on is the face in view";
    pub const FULL_PART: &str = "The whole part under the sketch, facing the sketch plane and following the sheet's pan and zoom, nothing cut away";
    pub const ACROSS: &str = "What the plane is laid across: a coordinate plane through the part’s middle, or an element with a direction — a plane’s normal, the axis of a cylinder, cone or line, a circle’s normal. The half toward you is taken away.";
    pub const OFFSET: &str = "Where along its direction the plane lies — drag the arrow on the part, or type it here";
    pub const SQUARE: &str = "The rings have tilted the plane off what it was laid across — turn it square again";
    pub const FLIP: &str = "Take away the other half instead";
    pub const LIMIT: &str = "Click to set the range of the scale";
    pub const DEVIATION_MAP: &str = "Colours the scan near the ghost by how far it lies from the ghost's surface — green on it, red outside the body, blue inside — on the Deviation ramp at ± this range, and darker past it. The rest of the scan keeps its colour. Drag the grips and watch the feature settle onto the scan.";
}

/// The keys whose menus open over them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuKey {
    Scan = 0,
    Bodies = 1,
}

/// A key's menu: open, opened by its caret, and the timers of the pointer resting on it or
/// gone from it.
#[derive(Debug, Clone, Copy, Default)]
struct Menu {
    open: bool,
    pinned: bool,
    since: Option<f64>,
    left: Option<f64>,
    area: Option<Rect>,
}

/// What the live map is reading and what it came to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MapReport {
    pub ghost: bool,
    pub points: usize,
    pub within: usize,
    pub far_inside: usize,
    pub range: f64,
}

type MapKey = (u64, u64, u64, bool);

struct MapDone {
    key: MapKey,
    colors: Vec<(u64, Arc<Vec<u32>>)>,
    report: MapReport,
}

#[derive(Default)]
pub(crate) struct MapState {
    job: Option<Receiver<MapDone>>,
    key: Option<MapKey>,
    pending: Option<(MapKey, Instant)>,
    pub report: Option<MapReport>,
    colors: Vec<(u64, Arc<Vec<u32>>)>,
}

#[derive(Default)]
struct Curvature {
    key: Option<u64>,
    job: Option<Receiver<(u64, Vec<(u64, Arc<Vec<u32>>)>, f64)>>,
    colors: Vec<(u64, Arc<Vec<u32>>)>,
    range: Option<f64>,
}

#[derive(Default)]
struct Ghost {
    key: Option<u64>,
    job: Option<Receiver<(u64, Vec<Arc<BodyMesh>>)>>,
    meshes: Option<(u64, Vec<Arc<BodyMesh>>)>,
}

/// A drag of the section view's gizmo.
#[derive(Debug, Clone, Copy)]
enum Gizmo {
    Slide { last: Pos2 },
    Turn { about: DVec3, last: Pos2 },
}

pub(crate) struct ViewBar {
    /// The scan a see-through veil: `None` follows whether a body is shown.
    pub scan_see_through: Option<bool>,
    pub scan_mesh: bool,
    pub bodies_see_through: bool,
    pub tangent_edges: TangentEdges,
    // The Overlaps switch is the part's (`Part::overlaps_shown`, PICKLIST's `set_overlaps_shown`).
    pub elements_shown: bool,
    pub curvature_map: bool,
    pub sketch_model_shown: bool,
    pub sketch_sliced: bool,
    pub section: SectionView,
    /// The ghost's live map (remembered).
    pub live_map: bool,
    // The coordinate planes, their eyes and the one under the cursor are PICKLIST's
    // (`picks.planes_shown`, `picks.hover_plane`).
    menus: [Menu; 2],
    legend_edit: Option<(bool, String)>,
    offset_text: Option<String>,
    joins: HashMap<usize, Arc<HashMap<u32, EdgeJoin>>>,
    silhouettes: HashMap<BodyId, (usize, DVec3, Arc<Vec<DVec3>>)>,
    caps: Option<(String, Arc<Vec<ColorVertex>>)>,
    scan_closed: HashMap<u64, (u64, Arc<Vec<bool>>)>,
    pub(crate) map: MapState,
    curvature: Curvature,
    ghost: Ghost,
    gizmo: Option<Gizmo>,
    shown_stamp: u64,
    /// The bodies meshed for the Mesh workspace's Deviation tool (Compare in Deviation).
    pub to_compare: Option<(String, mm_core::Mesh)>,
}

impl ViewBar {
    pub fn new(storage: Option<&dyn eframe::Storage>) -> ViewBar {
        let live_map = storage.filter(|_| !polymix_workspace::scripted()).and_then(|s| s.get_string("reverse.live_map")).is_none_or(|v| v != "0");
        ViewBar {
            scan_see_through: None,
            scan_mesh: false,
            bodies_see_through: false,
            tangent_edges: TangentEdges::Visible,
            elements_shown: true,
            curvature_map: false,
            sketch_model_shown: true,
            sketch_sliced: true,
            section: SectionView::default(),
            live_map,
            menus: [Menu::default(); 2],
            legend_edit: None,
            offset_text: None,
            joins: HashMap::new(),
            silhouettes: HashMap::new(),
            caps: None,
            scan_closed: HashMap::new(),
            map: MapState::default(),
            curvature: Curvature::default(),
            ghost: Ghost::default(),
            gizmo: None,
            shown_stamp: 1 << 40,
            to_compare: None,
        }
    }

    pub fn save(&self, storage: &mut dyn eframe::Storage) {
        storage.set_string("reverse.live_map", if self.live_map { "1" } else { "0" }.into());
    }

    /// The scan, the bodies or the range changed: the live map is read again.
    pub fn map_changed(&mut self) {
        self.map.key = None;
    }

    /// Whether a job of the view bar's is under way (scripts wait for it).
    pub fn idle(&self) -> bool {
        self.map.job.is_none() && self.curvature.job.is_none() && self.ghost.job.is_none() && self.map.pending.is_none()
    }

    /// Whether an edge of a body is left out of the drawing and the picks: a tangent edge with
    /// the tangent edges off, a seam, a split.
    pub fn edge_off(&self, topology: &mm_reverse::topo::Topology, hash: u32) -> bool {
        let joins = self.joins.get(&(topology as *const _ as usize));
        self.edge_alpha(joins.map(|j| j.as_ref()), hash).is_none()
    }

    /// Whether the gizmo has the pointer: the view is not moved by it.
    pub fn grabs_pointer(&self) -> bool {
        self.gizmo.is_some()
    }

    /// How an edge of a body is drawn, by how its faces meet: `None` not at all.
    pub fn edge_alpha(&self, joins: Option<&HashMap<u32, EdgeJoin>>, hash: u32) -> Option<f32> {
        match joins.and_then(|j| j.get(&hash)).copied().unwrap_or(EdgeJoin::Sharp) {
            EdgeJoin::Sharp => Some(1.0),
            EdgeJoin::Tangent => match self.tangent_edges {
                TangentEdges::Visible => Some(1.0),
                TangentEdges::Dimmed => Some(0.3),
                TangentEdges::Removed => None,
            },
            EdgeJoin::Split | EdgeJoin::Seam => None,
        }
    }
}

fn rgb_u32(c: [u8; 3]) -> u32 {
    ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32
}

fn base_rgb(c: Vec4) -> [u8; 3] {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    [b(c.x), b(c.y), b(c.z)]
}

/// Per triangle id, the colour of a map read at vertices: the mean of the painted corners'
/// readings, the violet where two corners lie deep inside, the scan's own colour elsewhere.
fn triangle_colors(p: &ScanPart, values: &[f32], deep: &[bool], base: [u8; 3], paint: &dyn Fn(f64) -> [u8; 3]) -> Vec<u32> {
    let mut colors = vec![rgb_u32(base); p.max_tid];
    for (k, t) in p.tri_ids.iter().enumerate() {
        let tri = p.triangles[k];
        let d = tri.iter().filter(|v| deep.get(**v as usize).copied().unwrap_or(false)).count();
        if d >= 2 {
            colors[*t as usize] = rgb_u32(FAR_INSIDE_RGB);
            continue;
        }
        let vals: Vec<f64> = tri.iter().filter_map(|v| values.get(*v as usize)).filter(|x| x.is_finite()).map(|x| *x as f64).collect();
        if !vals.is_empty() {
            colors[*t as usize] = rgb_u32(paint(vals.iter().sum::<f64>() / vals.len() as f64));
        }
    }
    colors
}

/// The part of a segment on the kept side of a cut — `plane` as (normal, offset) in the scene,
/// the side its normal points to taken away — or `None` when all of it is taken away.
pub(crate) fn clip_segment(a: DVec3, b: DVec3, plane: Option<(DVec3, f64)>) -> Option<(DVec3, DVec3)> {
    let Some((n, d)) = plane else { return Some((a, b)) };
    let (da, db) = (n.dot(a) - d, n.dot(b) - d);
    match (da <= 0.0, db <= 0.0) {
        (true, true) => Some((a, b)),
        (false, false) => None,
        (true, false) => Some((a, a + (b - a) * (da / (da - db)))),
        (false, true) => Some((a + (b - a) * (da / (da - db)), b)),
    }
}

/// A kind with its op set to New, when it makes a tool body that joins or cuts: its ghost.
fn as_tool(kind: &FeatureKind) -> Option<FeatureKind> {
    let mut k = kind.clone();
    let op = match &mut k {
        FeatureKind::Extrude(x) => &mut x.op,
        FeatureKind::Revolve(x) => &mut x.op,
        FeatureKind::Extract(x) => &mut x.op,
        FeatureKind::Thicken(x) => &mut x.op,
        FeatureKind::Loft(x) => &mut x.op,
        FeatureKind::Sweep(x) => &mut x.op,
        _ => return None,
    };
    if *op == BodyOp::New {
        return None;
    }
    *op = BodyOp::New;
    Some(k)
}


impl ReverseWorkspace {
    // ---- what is shown ----

    /// Whether any body is shown.
    pub(crate) fn any_body_shown(&self) -> bool {
        self.show_bodies && self.bodies.iter().any(|b| !self.hidden_bodies.contains(&b.id))
    }

    /// The 3D view under a sketch put away (the sketch on the bare stage).
    pub(crate) fn vb_part_hidden(&self) -> bool {
        matches!(self.mode, Mode::Sketch(_)) && !self.vb.sketch_model_shown
    }

    /// Where the part is cut open, and whether by the sketch (else by the section view).
    pub(crate) fn vb_cut(&self) -> Option<(SectionFrame, bool)> {
        if let Mode::Sketch(e) = &self.mode {
            if self.vb.sketch_model_shown && self.vb.sketch_sliced {
                return Some((e.frame, true));
            }
        }
        self.vb.section.frames().map(|f| (f.0, false))
    }

    /// The cut plane in the scene, for the lines drawn over the part: what lies on the side
    /// its normal points to is taken away.
    pub(crate) fn vb_clip(&self) -> Option<(DVec3, f64)> {
        let (f, _) = self.vb_cut()?;
        let n = self.to_scene(f.normal);
        let o = self.to_scene(f.origin) + n * 1e-3;
        Some((n, n.dot(o)))
    }

    /// The flags the view draws by, from the view bar: the scan a veil over the bodies as the
    /// workspace opens, solid while no body is shown.
    pub(crate) fn sync_view_flags(&mut self) {
        let veil = self.vb.scan_see_through.unwrap_or(self.any_body_shown());
        self.scan_see_through = veil && self.show_scan;
    }

    /// The edge joins of the bodies shown, by body id, found once per body mesh.
    pub(crate) fn vb_edge_joins(&mut self) -> HashMap<BodyId, Arc<HashMap<u32, EdgeJoin>>> {
        let mut out = HashMap::new();
        let live: Vec<usize> = self.bodies.iter().map(|b| Arc::as_ptr(&b.topology) as usize).collect();
        self.vb.joins.retain(|k, _| live.contains(k));
        for b in &self.bodies {
            let key = Arc::as_ptr(&b.topology) as usize;
            let joins = self
                .vb
                .joins
                .entry(key)
                .or_insert_with(|| {
                    let edges: Vec<OutlinedEdge> = b.topology.edges.iter().map(|e| OutlinedEdge { hash: e.hash, faces: &e.faces, polyline: &e.polyline }).collect();
                    let t = &b.topology;
                    let one = |x: u32, y: u32| match (t.face(x), t.face(y)) {
                        (Some(a), Some(c)) => mm_reverse::topo::on_one_surface(&a.fingerprint, &c.fingerprint),
                        _ => false,
                    };
                    Arc::new(edge_joins(&b.part_mesh, &edges, Some(&one)))
                })
                .clone();
            out.insert(b.id, joins);
        }
        out
    }

    /// The round faces' outlines from where the camera stands, as line segments in the scene,
    /// found again when the camera turns.
    pub(crate) fn vb_silhouettes(&mut self, lines: &mut Vec<ColorVertex>, color: Vec4) {
        if !self.show_bodies || self.vb_part_hidden() {
            return;
        }
        let eye = self.to_part(self.view.camera.eye());
        let ortho = self.view.camera.orthographic;
        let look = self.to_part(self.view.camera.target - self.view.camera.eye()).normalize_or_zero();
        let ids: Vec<BodyId> = self.bodies.iter().map(|b| b.id).collect();
        self.vb.silhouettes.retain(|k, _| ids.contains(k));
        let cam_eye = self.view.camera.eye();
        let clip = self.vb_clip();
        for b in self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)) {
            let key = Arc::as_ptr(&b.part_mesh) as usize;
            let centre = b.part_mesh.positions.iter().fold(DVec3::ZERO, |a, p| a + *p) / b.part_mesh.positions.len().max(1) as f64;
            let dir = if ortho { look } else { (centre - eye).normalize_or_zero() };
            let fresh = match self.vb.silhouettes.get(&b.id) {
                Some((k, d, _)) => *k == key && d.angle_between(dir).to_degrees() <= SILHOUETTE_TURN_DEG,
                None => false,
            };
            if !fresh {
                let m = &b.part_mesh;
                let round: std::collections::HashSet<u32> = b.topology.faces.iter().filter(|f| f.fingerprint.kind != mm_reverse::topo::FaceKind::Plane).map(|f| f.hash).collect();
                let mut ranges = Vec::new();
                let mut t = 0;
                while t < m.tri_face.len() {
                    if !round.contains(&m.tri_face[t]) {
                        t += 1;
                        continue;
                    }
                    let start = t;
                    while t < m.tri_face.len() && round.contains(&m.tri_face[t]) {
                        t += 1;
                    }
                    ranges.push((start, t - start));
                }
                let pos: Vec<f32> = m.positions.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
                let nor: Vec<f32> = m.normals.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
                let idx: Vec<u32> = m.triangles.iter().flatten().copied().collect();
                let view = if ortho { SilhouetteView::Dir(look) } else { SilhouetteView::Eye(eye) };
                let segs = silhouette_segments(&pos, &nor, &idx, &ranges, view);
                let pts: Vec<DVec3> = segs.chunks(3).map(|c| self.to_scene(DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64))).collect();
                self.vb.silhouettes.insert(b.id, (key, dir, Arc::new(pts)));
            }
            if let Some((_, _, pts)) = self.vb.silhouettes.get(&b.id) {
                for w in pts.chunks(2) {
                    let [a, c] = w else { continue };
                    let Some((a, c)) = clip_segment(*a, *c, clip) else { continue };
                    for p in [a, c] {
                        let q = p + (cam_eye - p) * 2e-4;
                        lines.push(ColorVertex::new(q.as_vec3(), color));
                    }
                }
            }
        }
    }

    /// The flat triangles drawn over the part: the caps where it is cut. The coordinate
    /// planes' sheets and what two bodies share are PICKLIST's (`picks.rs`), drawn with these.
    pub(crate) fn vb_overlay(&mut self, _lines: &mut Vec<ColorVertex>) -> Vec<ColorVertex> {
        let mut tris = Vec::new();
        if !self.vb_part_hidden() {
            if let Some(caps) = self.vb_caps() {
                tris.extend_from_slice(&caps);
            }
        }
        tris
    }

    fn scan_closed(&mut self, p: &ScanPart) -> Arc<Vec<bool>> {
        if let Some((k, c)) = self.vb.scan_closed.get(&p.id) {
            if *k == self.scan_key {
                return c.clone();
            }
        }
        let c = Arc::new(closed_triangles(&p.positions, &p.triangles));
        self.vb.scan_closed.insert(p.id, (self.scan_key, c.clone()));
        c
    }

    /// The caps where the part is cut: the scan's closed shells and each solid body, filled.
    fn vb_caps(&mut self) -> Option<Arc<Vec<ColorVertex>>> {
        let (frame, sketch) = self.vb_cut()?;
        let shown: Vec<usize> = self.bodies.iter().enumerate().filter(|(_, b)| self.show_bodies && !self.hidden_bodies.contains(&b.id)).map(|(i, _)| i).collect();
        let key = format!("{:?}|{sketch}|{}|{}|{}|{:?}|{}", [frame.origin, frame.normal], self.built_generation, self.scan_key, self.show_scan, shown, self.bodies.len());
        if let Some((k, c)) = &self.vb.caps {
            if *k == key {
                return Some(c.clone());
            }
        }
        let mut out = Vec::new();
        let colour = |c: [u8; 3]| Vec4::new(c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0);
        if self.show_scan {
            let parts = self.scan_parts();
            let scan_colour = colour(if sketch { SKETCH_CAP_DARK } else { cut_color(CAP_SLOT_SCAN) });
            for p in parts.iter() {
                let keep = self.scan_closed(p);
                if !keep.iter().any(|k| *k) {
                    continue;
                }
                for t in mesh_cap(&p.positions, &p.triangles, &frame, Some(&keep)) {
                    for q in t {
                        out.push(ColorVertex::new(self.to_scene(q).as_vec3(), scan_colour));
                    }
                }
            }
        }
        for (slot, i) in shown.iter().enumerate() {
            let b = &self.bodies[*i];
            if self.built.body(b.id).is_some_and(|x| x.surface) {
                continue;
            }
            let c = colour(cut_color(CAP_SLOT_LAYERS + slot));
            let m = &b.part_mesh;
            let (pos, tris) = mm_reverse::build::welded(&m.positions, &m.triangles, 1e-7);
            for t in mesh_cap(&pos, &tris, &frame, None) {
                for q in t {
                    out.push(ColorVertex::new(self.to_scene(q).as_vec3(), c));
                }
            }
        }
        let caps = Arc::new(out);
        self.vb.caps = Some((key, caps.clone()));
        Some(caps)
    }

    /// Which coordinate planes are drawn: as their rows' eyes say (PICKLIST's `plane_shown`),
    /// else up on a blank stage and put away over a scan; none under a sketch.
    pub(crate) fn planes_drawn(&self) -> Vec<WorldAxis> {
        if matches!(self.mode, Mode::Sketch(_)) {
            return Vec::new();
        }
        WorldAxis::ALL.into_iter().filter(|a| self.plane_shown(*a)).collect()
    }

    // ---- the maps ----

    /// Read the live map and the curvature map again when what they read changed, on worker
    /// threads, and put the one shown on the scan.
    pub(crate) fn update_maps(&mut self, ctx: &egui::Context) {
        self.poll_ghost();
        if let Some(rx) = &self.vb.map.job {
            match rx.try_recv() {
                Ok(done) => {
                    self.vb.map.job = None;
                    self.vb.map.key = Some(done.key);
                    self.vb.map.colors = done.colors;
                    self.vb.map.report = Some(done.report);
                    self.vb.shown_stamp += 1;
                    ctx.request_repaint();
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(40)),
                Err(_) => self.vb.map.job = None,
            }
        }
        if let Some(rx) = &self.vb.curvature.job {
            match rx.try_recv() {
                Ok((key, colors, range)) => {
                    self.vb.curvature.job = None;
                    self.vb.curvature.key = Some(key);
                    self.vb.curvature.colors = colors;
                    self.vb.curvature.range = Some(range);
                    self.vb.shown_stamp += 1;
                    ctx.request_repaint();
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(60)),
                Err(_) => self.vb.curvature.job = None,
            }
        }
        // What the map reads: the ghost while a box has one up and the map is on; the bodies
        // on the Deviation key otherwise.
        let feature_open = matches!(self.mode, Mode::Feature(_));
        let preview_ok = feature_open && !self.building() && self.preview_feature.is_some_and(|f| !self.built.errors.contains_key(&f));
        let source: Option<bool> = if preview_ok && self.vb.live_map {
            Some(true)
        } else if feature_open && self.building() && self.vb.live_map && self.vb.map.report.is_some_and(|r| r.ghost) {
            // A ghost being rebuilt leaves the last map on the scan until the new one lands.
            return self.paint_maps();
        } else if !preview_ok && self.deviation.on && !self.bodies.is_empty() && !self.building() && !self.scan.is_empty() {
            Some(false)
        } else {
            None
        };
        match source {
            None => {
                if self.vb.map.report.is_some() || !self.vb.map.colors.is_empty() {
                    self.vb.map.report = None;
                    self.vb.map.colors.clear();
                    self.vb.map.key = None;
                    self.vb.shown_stamp += 1;
                }
                self.vb.map.pending = None;
            }
            Some(ghost) => self.start_map(ghost, ctx),
        }
        // The curvature, read once per scan while its key is on.
        if self.vb.curvature_map && !self.scan.is_empty() && self.vb.curvature.key != Some(self.scan_key) && self.vb.curvature.job.is_none() {
            self.start_curvature();
        }
        self.paint_maps();
    }

    /// Put the map shown on the scan: the live map, else the curvature map.
    fn paint_maps(&mut self) {
        let ghost_up = matches!(self.mode, Mode::Feature(_)) && self.vb.live_map && self.vb.map.report.is_some_and(|r| r.ghost);
        let shown: &Vec<(u64, Arc<Vec<u32>>)> = if self.vb.map.report.is_some() {
            &self.vb.map.colors
        } else if self.vb.curvature_map && !ghost_up && self.vb.curvature.key == Some(self.scan_key) {
            &self.vb.curvature.colors
        } else {
            &Vec::new()
        };
        let stamp = self.vb.shown_stamp;
        if self.deviation.stamp == stamp && self.deviation.colors.len() == shown.len() {
            return;
        }
        self.deviation.colors.clear();
        for (id, c) in shown {
            self.deviation.colors.insert(*id, (stamp, c.clone()));
        }
        self.deviation.stamp = stamp;
    }

    /// The ghost of the open box: the tool body a join or a cut adds or takes, else the bodies
    /// the feature leaves.
    fn ghost_meshes(&mut self) -> Option<(u64, Vec<Arc<BodyMesh>>)> {
        let Mode::Feature(edit) = &self.mode else { return None };
        let fid = self.preview_feature?;
        if as_tool(&edit.kind).is_some() {
            let key = self.built_generation;
            if self.vb.ghost.key != Some(key) && self.vb.ghost.job.is_none() {
                let (mut part, _) = self.shown_part();
                let upto = part.index_of(fid).map(|i| i + 1)?;
                if let Some(f) = part.feature_mut(fid) {
                    if let Some(k) = as_tool(&f.kind) {
                        f.kind = k;
                    }
                }
                let (tx, rx) = channel();
                self.vb.ghost.job = Some(rx);
                self.vb.ghost.key = Some(key);
                std::thread::spawn(move || {
                    let built = build_upto(&part, upto);
                    let meshes = built.bodies.iter().filter(|b| b.id.feature == fid).filter_map(|b| body_mesh(&b.body).ok().map(Arc::new)).collect();
                    let _ = tx.send((key, meshes));
                });
            }
            return self.vb.ghost.meshes.clone().filter(|m| Some(m.0) == self.vb.ghost.key);
        }
        let touched = self.built.touched.get(&fid);
        let meshes: Vec<Arc<BodyMesh>> = self.bodies.iter().filter(|b| b.id.feature == fid || touched.is_some_and(|t| t.contains(&b.id))).map(|b| b.part_mesh.clone()).collect();
        Some((self.built_generation, meshes))
    }

    fn poll_ghost(&mut self) {
        if let Some(rx) = &self.vb.ghost.job {
            if let Ok((key, meshes)) = rx.try_recv() {
                self.vb.ghost.job = None;
                self.vb.ghost.meshes = Some((key, meshes));
            }
        }
    }

    fn start_map(&mut self, ghost: bool, ctx: &egui::Context) {
        let range = self.deviation.range;
        let (data_key, meshes): (u64, Vec<(Arc<BodyMesh>, bool)>) = if ghost {
            match self.ghost_meshes() {
                Some((k, m)) => (k.wrapping_mul(31).wrapping_add(1), m.into_iter().map(|x| (x, false)).collect()),
                None => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(40));
                    return;
                }
            }
        } else {
            let shown: Vec<&super::BodyView> = self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)).collect();
            let key = shown.iter().fold(self.built_generation.wrapping_mul(7919), |h, b| h.wrapping_mul(1_000_003).wrapping_add(Arc::as_ptr(&b.part_mesh) as u64));
            (key, shown.iter().map(|b| (b.part_mesh.clone(), self.built.body(b.id).is_some_and(|x| x.surface))).collect())
        };
        let key: MapKey = (self.scan_key, data_key, range.to_bits(), ghost);
        if self.vb.map.key == Some(key) || self.vb.map.job.is_some() {
            return;
        }
        // A ghost's map waits for the ghost to stand still a moment.
        if ghost {
            match self.vb.map.pending {
                Some((k, t)) if k == key => {
                    if t.elapsed().as_secs_f64() < MAP_DEBOUNCE {
                        ctx.request_repaint_after(std::time::Duration::from_millis(30));
                        return;
                    }
                }
                _ => {
                    self.vb.map.pending = Some((key, Instant::now()));
                    ctx.request_repaint_after(std::time::Duration::from_millis(130));
                    return;
                }
            }
        }
        self.vb.map.pending = None;
        let parts = self.scan_parts();
        let bases: Vec<(u64, [u8; 3])> = self.scan.iter().map(|o| (o.id, base_rgb(o.color))).collect();
        let (tx, rx) = channel();
        self.vb.map.job = Some(rx);
        std::thread::spawn(move || {
            let list: Vec<SurfaceBody> = meshes.iter().enumerate().map(|(i, (m, sheet))| SurfaceBody { tag: i as u64, mesh: m, sheet: *sheet }).collect();
            let mut report = MapReport { ghost, points: 0, within: 0, far_inside: 0, range };
            let mut colors = Vec::new();
            if let Some(surface) = Surface::of_bodies(&list) {
                let scale = live_map_scale(range);
                for p in parts.iter() {
                    let normals = vertex_normals(&p.positions, &p.triangles);
                    let side = ScanSide { normals: &normals, facing_deg: Some(DEFAULT_FACING_DEG), unfacing: Unfacing::Keep, skin_outer: false };
                    let field = live_map(&surface, &p.positions, range, !ghost, Some(side));
                    let s = sparse_summary(&field, range);
                    report.points += s.points;
                    report.within += s.within;
                    report.far_inside += s.far_inside;
                    let mut values = vec![f32::NAN; p.positions.len()];
                    for (v, x) in field.subset.iter().zip(&field.values) {
                        values[*v as usize] = *x;
                    }
                    let mut deep = vec![false; p.positions.len()];
                    for v in field.far_inside.iter().flatten() {
                        deep[*v as usize] = true;
                    }
                    let base = bases.iter().find(|b| b.0 == p.id).map_or([192, 192, 192], |b| b.1);
                    colors.push((p.id, Arc::new(triangle_colors(p, &values, &deep, base, &|v| paint_value(v, &scale, &JET)))));
                }
            }
            let _ = tx.send(MapDone { key, colors, report });
        });
    }

    fn start_curvature(&mut self) {
        let key = self.scan_key;
        let parts = self.scan_parts();
        let objects: Vec<(u64, Arc<mm_core::Mesh>, [u8; 3])> = self.scan.iter().filter(|o| o.visible).map(|o| (o.id, o.mesh.clone(), base_rgb(o.color))).collect();
        let (tx, rx) = channel();
        self.vb.curvature.job = Some(rx);
        std::thread::spawn(move || {
            let mut all: Vec<(u64, Vec<f32>)> = Vec::new();
            let mut size = 0.0f64;
            for (id, mesh, _) in &objects {
                let b = mesh.bounds();
                if !b.is_empty() {
                    size = size.max(b.diagonal());
                }
                if let Ok(g) = MeshGraph::from_mesh(mesh, false) {
                    all.push((*id, mean_curvature(&g, CURVATURE_SMOOTHING)));
                }
            }
            let joined: Vec<f32> = all.iter().flat_map(|(_, v)| v.iter().copied()).collect();
            let range = curvature_range(&joined, size);
            let scale = curvature_scale(range);
            let colors = all
                .iter()
                .filter_map(|(id, values)| {
                    let p = parts.iter().find(|p| p.id == *id)?;
                    let base = objects.iter().find(|o| o.0 == *id).map_or([192, 192, 192], |o| o.2);
                    Some((*id, Arc::new(triangle_colors(p, values, &[], base, &|v| paint_value(v, &scale, &JET)))))
                })
                .collect();
            let _ = tx.send((key, colors, range));
        });
    }

    // ---- the section view's gizmo ----

    fn gizmo_geometry(&self, vp: Viewport) -> Option<(Pos2, Pos2, [Vec<Pos2>; 2], [DVec3; 2])> {
        let (_, g) = self.vb.section.frames()?;
        let c = self.project(vp, g.origin)?;
        // The arrow held at a size on screen.
        let tip3 = g.origin + g.normal * 1.0;
        let tip_dir = self.project(vp, tip3).map(|t| (t - c).normalized()).unwrap_or(Vec2::X);
        let tip = c + tip_dir * ARROW_PX;
        // The rings about the plane's u and v, at a radius held on screen.
        let mm_per_px = {
            let a = self.project(vp, g.origin + g.u);
            a.map_or(0.1, |a| 1.0 / (a - c).length().max(1e-3) as f64)
        };
        let r = RING_PX as f64 * mm_per_px;
        let ring = |axis: DVec3| -> Vec<Pos2> {
            let (a, b) = if axis == g.u { (g.normal, g.v) } else { (g.normal, g.u) };
            (0..=48).filter_map(|i| {
                let t = i as f64 / 48.0 * std::f64::consts::TAU;
                self.project(vp, g.origin + (a * t.cos() + b * t.sin()) * r)
            }).collect()
        };
        Some((c, tip, [ring(g.u), ring(g.v)], [g.u, g.v]))
    }

    fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
        let ab = b - a;
        let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
        (a + ab * t - p).length()
    }

    /// The gizmo takes a press near its arrow or a ring, and the drag that follows.
    pub(crate) fn section_gizmo_input(&mut self, ctx: &egui::Context, vp: Viewport, response: &Response) {
        let Some((c, tip, rings, axes)) = self.gizmo_geometry(vp) else {
            self.vb.gizmo = None;
            return;
        };
        let (pressed, down, pos) = ctx.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down(), i.pointer.hover_pos()));
        if pressed && response.hovered() {
            if let Some(p) = pos {
                if Self::seg_dist(p, c, tip) <= 8.0 {
                    self.vb.gizmo = Some(Gizmo::Slide { last: p });
                } else {
                    for (k, r) in rings.iter().enumerate() {
                        if r.windows(2).any(|w| Self::seg_dist(p, w[0], w[1]) <= 6.0) {
                            self.vb.gizmo = Some(Gizmo::Turn { about: axes[k], last: p });
                            break;
                        }
                    }
                }
            }
        }
        if !down {
            self.vb.gizmo = None;
            return;
        }
        let Some(p) = pos else { return };
        match self.vb.gizmo {
            Some(Gizmo::Slide { last }) => {
                // One millimetre along the normal, on screen.
                let one = self.vb.section.frames().and_then(|(_, g)| self.project(vp, g.origin + g.normal)).map_or(10.0, |q| (q - c).length().max(1e-3));
                let along = (p - last).dot((tip - c).normalized());
                let off = self.vb.section.offset + (along / one) as f64;
                self.vb.section.set_offset(off);
                self.vb.offset_text = None;
                self.vb.gizmo = Some(Gizmo::Slide { last: p });
            }
            Some(Gizmo::Turn { about, last }) => {
                let a0 = (last - c).angle();
                let a1 = (p - c).angle();
                let mut da = (a1 - a0) as f64;
                if da > std::f64::consts::PI {
                    da -= std::f64::consts::TAU;
                } else if da < -std::f64::consts::PI {
                    da += std::f64::consts::TAU;
                }
                // Screen y runs down: a turn on screen is the other way round about an axis
                // pointing at the eye.
                let toward = self.to_part(self.view.camera.eye() - self.view.camera.target).normalize_or_zero();
                let sign = if about.dot(toward) >= 0.0 { -1.0 } else { 1.0 };
                self.vb.section.turn(about, sign * da);
                self.vb.gizmo = Some(Gizmo::Turn { about, last: p });
            }
            None => {}
        }
    }

    fn paint_gizmo(&self, painter: &egui::Painter, vp: Viewport) {
        let Some((c, tip, rings, _)) = self.gizmo_geometry(vp) else { return };
        let col = tok::ACCENT;
        painter.line_segment([c, tip], Stroke::new(2.5, col));
        let d = (tip - c).normalized();
        let n = Vec2::new(-d.y, d.x);
        painter.add(egui::Shape::convex_polygon(vec![tip + d * 10.0, tip + n * 5.0, tip - n * 5.0], col, Stroke::NONE));
        for (k, r) in rings.iter().enumerate() {
            let rc = if k == 0 { tok::AXIS[0] } else { tok::AXIS[1] };
            painter.add(egui::Shape::line(r.clone(), Stroke::new(2.0, rc.gamma_multiply(0.85))));
        }
        painter.circle_filled(c, 3.5, col);
    }

    // ---- the overlays: view bar, legend, chips ----

    /// What the view bar and the stage's overlays draw over the viewport, every frame.
    pub(crate) fn viewcmp_overlay(&mut self, ctx: &egui::Context, vp: Viewport) {
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, egui::Id::NULL.with("re_vb_paint"))).with_clip_rect(vp.rect);
        if self.vb.section.on && !self.vb_part_hidden() {
            self.paint_gizmo(&painter, vp);
        }
        // The hovered coordinate plane's name, and the chips at the top of the stage (the
        // build's, the ready one), are PICKLIST's (`paint_picks`) and BUILD's (`build_chip`).
        let now = ctx.input(|i| i.time);
        self.legend(ctx, vp);
        self.view_bar(ctx, vp, now);
    }

    /// The scale beside the map, down the right edge.
    fn legend(&mut self, ctx: &egui::Context, vp: Viewport) {
        enum Kind {
            Live(MapReport),
            Curvature(f64),
        }
        let ghost_up = matches!(self.mode, Mode::Feature(_)) && self.vb.live_map && self.vb.map.report.is_some_and(|r| r.ghost);
        let kind = match (self.vb.map.report, self.vb.curvature.range) {
            (Some(r), _) => Kind::Live(r),
            (None, Some(range)) if self.vb.curvature_map && !ghost_up && self.vb.curvature.key == Some(self.scan_key) => Kind::Curvature(range),
            _ => {
                self.vb.legend_edit = None;
                return;
            }
        };
        let (head, high_words, low_words, scale, stats, decimals): (String, &str, &str, mm_reverse::compare::FieldScale, Vec<(String, String, bool)>, usize) = match &kind {
            Kind::Live(r) => {
                let mut stats = vec![("within".to_string(), format!("{} / {}", grouped(r.within), grouped(r.points)), true), ("against".to_string(), if r.ghost { "the ghost".into() } else { "the bodies".into() }, false)];
                if r.far_inside > 0 {
                    stats.push(("deep inside (violet)".into(), grouped(r.far_inside), true));
                }
                ("mm".into(), "scan outside the body", "scan inside the body", live_map_scale(r.range), stats, 3)
            }
            Kind::Curvature(range) => ("1/mm".into(), "convex — a round", "concave — a hollow", curvature_scale(*range), vec![("full colour at".into(), format!("R {} mm", mm_reverse::js_fmt::to_fixed(1.0 / (2.0 * range), 1)), true)], 3),
        };
        let editable = matches!(kind, Kind::Live(_));
        let fmt = |v: f64| -> String {
            match kind {
                Kind::Live(_) => format!("{}{}", if v >= 0.0 { "+" } else { "−" }, mm_reverse::js_fmt::to_fixed(v.abs(), decimals)),
                Kind::Curvature(_) => mm_reverse::js_fmt::to_fixed(v, 3),
            }
        };
        let width = 170.0;
        let top = vp.rect.top() + 14.0;
        // Clear of the orientation gizmo in the bottom-right corner.
        let bottom = vp.rect.bottom() - 90.0;
        let right = vp.rect.right() - 14.0;
        let mut commit: Option<f64> = None;
        let mut edit = self.vb.legend_edit.take();
        egui::Area::new(egui::Id::NULL.with("re_map_legend")).order(egui::Order::Middle).fixed_pos(Pos2::new(right - width, top)).interactable(true).show(ctx, |ui| {
            let full = Rect::from_min_max(Pos2::new(right - width, top), Pos2::new(right, bottom));
            ui.set_min_size(Vec2::new(width, 10.0));
            let p = ui.painter().clone();
            let small = FontId::proportional(11.0);
            let mut y = top;
            p.text(Pos2::new(right, y), Align2::RIGHT_TOP, &head, FontId::proportional(12.0), tok::INK);
            y += 18.0;
            p.text(Pos2::new(right, y), Align2::RIGHT_TOP, format!("+ {high_words}"), small.clone(), tok::INK);
            y += 16.0;
            let stats_h = 16.0 * stats.len() as f32 + 22.0;
            let ramp_top = y + 4.0;
            let ramp_bottom = (full.bottom() - stats_h - 20.0).max(ramp_top + 80.0);
            let cap = 10.0;
            let ramp_x = right - 14.0;
            let (low_cap, high_cap) = mm_reverse::compare::scale_caps(&scale, &JET);
            let c32 = |c: [u8; 3]| Color32::from_rgb(c[0], c[1], c[2]);
            let bar = Rect::from_min_max(Pos2::new(ramp_x, ramp_top + cap), Pos2::new(right, ramp_bottom - cap));
            p.rect_filled(Rect::from_min_max(Pos2::new(ramp_x, ramp_top), Pos2::new(right, ramp_top + cap - 1.0)), 0.0, c32(high_cap));
            p.rect_filled(Rect::from_min_max(Pos2::new(ramp_x, ramp_bottom - cap + 1.0), Pos2::new(right, ramp_bottom)), 0.0, c32(low_cap));
            let steps = 64;
            for i in 0..steps {
                let t = 1.0 - (i as f64 + 0.5) / steps as f64;
                let c = mm_reverse::compare::ramp_color(&JET, t);
                let cell = Rect::from_min_size(Pos2::new(bar.left(), bar.top() + bar.height() * i as f32 / steps as f32), Vec2::new(bar.width(), bar.height() / steps as f32 + 0.6));
                p.rect_filled(cell, 0.0, c32(c));
            }
            let cap_hover_top = ui.interact(Rect::from_min_max(Pos2::new(ramp_x, ramp_top), Pos2::new(right, ramp_top + cap)), egui::Id::NULL.with("re_legend_cap_hi"), Sense::hover());
            cap_hover_top.on_hover_text("Beyond the top of the scale");
            let cap_hover_bottom = ui.interact(Rect::from_min_max(Pos2::new(ramp_x, ramp_bottom - cap), Pos2::new(right, ramp_bottom)), egui::Id::NULL.with("re_legend_cap_lo"), Sense::hover());
            cap_hover_bottom.on_hover_text("Beyond the bottom of the scale");
            // Nine ticks with their figures; the end ones keys for the range.
            for i in 0..9 {
                let f = i as f32 / 8.0;
                let v = scale.low + (scale.high - scale.low) * f as f64;
                let ty = bar.bottom() - bar.height() * f;
                p.line_segment([Pos2::new(ramp_x - 4.0, ty), Pos2::new(ramp_x, ty)], Stroke::new(1.0, tok::DIM));
                let text = fmt(v);
                let at = Pos2::new(ramp_x - 7.0, ty);
                let end = if i == 8 { Some(true) } else if i == 0 { Some(false) } else { None };
                match (end, editable) {
                    (Some(high), true) => {
                        let editing = edit.as_ref().is_some_and(|e| e.0 == high);
                        let r = Rect::from_min_size(at - Vec2::new(64.0, 9.0), Vec2::new(64.0, 18.0));
                        if editing {
                            let mut text = edit.as_ref().map(|e| e.1.clone()).unwrap_or_default();
                            let resp = ui.put(r, egui::TextEdit::singleline(&mut text).font(small.clone()));
                            resp.request_focus();
                            let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                            let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
                            if esc {
                                edit = None;
                            } else if enter || resp.lost_focus() {
                                let shown = mm_reverse::js_fmt::num(mm_reverse::js_fmt::fixed_num(v.abs(), 6));
                                if text.trim() != shown {
                                    if let Ok(x) = text.trim().parse::<f64>() {
                                        if x.is_finite() && x.abs() > 0.0 && x.abs() != v.abs() {
                                            commit = Some(x.abs());
                                        }
                                    }
                                }
                                edit = None;
                            } else {
                                edit = Some((high, text));
                            }
                        } else {
                            let resp = ui.interact(r, egui::Id::NULL.with(("re_legend_end", high)), Sense::click()).on_hover_text(words::LIMIT).on_hover_cursor(egui::CursorIcon::PointingHand);
                            p.text(at, Align2::RIGHT_CENTER, &text, small.clone(), if resp.hovered() { tok::ACCENT_D } else { tok::INK });
                            if resp.clicked() {
                                edit = Some((high, mm_reverse::js_fmt::num(mm_reverse::js_fmt::fixed_num(v.abs(), 6))));
                            }
                        }
                    }
                    _ => {
                        p.text(at, Align2::RIGHT_CENTER, &text, small.clone(), if v.abs() < 1e-12 { Color32::WHITE } else { tok::INK });
                    }
                }
            }
            let mut y = ramp_bottom + 6.0;
            p.text(Pos2::new(right, y), Align2::RIGHT_TOP, format!("− {low_words}"), small.clone(), tok::INK);
            y += 20.0;
            for (label, value, _) in &stats {
                p.text(Pos2::new(right, y), Align2::RIGHT_TOP, value, FontId::proportional(11.5), tok::INK);
                let w = p.layout_no_wrap(value.clone(), FontId::proportional(11.5), tok::INK).size().x;
                p.text(Pos2::new(right - w - 8.0, y), Align2::RIGHT_TOP, label, small.clone(), tok::DIM.gamma_multiply(1.2));
                y += 16.0;
            }
            ui.allocate_rect(Rect::from_min_max(Pos2::new(right - width, top), Pos2::new(right, y)), Sense::hover());
        });
        self.vb.legend_edit = edit;
        if let Some(v) = commit {
            // Either end sets the symmetric range, within what the field takes.
            self.deviation.range = v.clamp(mm_reverse::compare::LIVE_MAP_MIN_RANGE_MM, 100.0);
        }
    }

    /// The view bar, bottom-left of the stage.
    fn view_bar(&mut self, ctx: &egui::Context, vp: Viewport, now: f64) {
        let mut x = vp.rect.left() + 12.0;
        if let Some(r) = self.view.panel_rect.filter(|r| r.bottom() > vp.rect.bottom() - 60.0 && r.left() <= x + 4.0) {
            x = r.right() + 12.0;
        }
        let at = Pos2::new(x, vp.rect.bottom() - 12.0);
        let sketch = matches!(self.mode, Mode::Sketch(_));
        let no_part = self.vb_part_hidden();
        let pointer = ctx.input(|i| i.pointer.hover_pos());
        let pressed = ctx.input(|i| i.pointer.any_pressed());
        let mut key_rects: [Option<Rect>; 2] = [None, None];
        let mut caret_clicked: [bool; 2] = [false, false];
        egui::Area::new(egui::Id::NULL.with("re_view_bar")).order(egui::Order::Foreground).pivot(Align2::LEFT_BOTTOM).fixed_pos(at).show(ctx, |ui| {
            ui.spacing_mut().item_spacing = Vec2::new(6.0, 6.0);
            ui.vertical(|ui| {
                if self.vb.section.on && !no_part {
                    self.section_row(ui);
                }
                ui.horizontal_top(|ui| {
                    if !no_part {
                        group(ui, |ui| {
                            let (r, c) = self.scan_key_ui(ui, sketch);
                            key_rects[0] = Some(r);
                            caret_clicked[0] = c;
                            if !sketch {
                                let (r, c) = self.bodies_key_ui(ui);
                                key_rects[1] = Some(r);
                                caret_clicked[1] = c;
                                self.elements_key_ui(ui);
                            }
                        });
                    }
                    if sketch {
                        group(ui, |ui| self.part_views_ui(ui));
                    } else {
                        group(ui, |ui| {
                            let can = !self.bodies.is_empty();
                            let r = key(ui, Glyph::Compare, "Deviation", self.deviation.on, can, false).on_hover_text(if can { words::DEVIATION } else { words::DEVIATION_NONE });
                            if r.clicked() && can {
                                self.deviation.on = !self.deviation.on;
                                if self.deviation.on {
                                    self.vb.curvature_map = false;
                                }
                                self.deviation.key = None;
                            }
                        });
                    }
                    if !no_part {
                        group(ui, |ui| {
                            let r = key(ui, Glyph::SectionView, "Section", self.vb.section.on, true, false).on_hover_text(words::SECTION);
                            if r.clicked() {
                                if self.vb.section.on {
                                    self.vb.section.close();
                                } else {
                                    let view = self.to_part(self.view.camera.target - self.view.camera.eye()).normalize_or_zero();
                                    let centre = self.stage_centre();
                                    self.vb.section.open(view, centre);
                                }
                                self.vb.offset_text = None;
                            }
                        });
                    }
                });
            });
        });
        // The menus over the Scan and Bodies keys.
        for (k, which) in [(0usize, MenuKey::Scan), (1usize, MenuKey::Bodies)] {
            let Some(rect) = key_rects[k] else {
                self.vb.menus[k] = Menu::default();
                continue;
            };
            let mut m = self.vb.menus[k];
            let over = pointer.is_some_and(|p| rect.expand(2.0).contains(p) || m.area.is_some_and(|a| a.expand(4.0).contains(p)));
            if caret_clicked[k] {
                if m.open && m.pinned {
                    m.open = false;
                    m.pinned = false;
                } else {
                    m.open = true;
                    m.pinned = true;
                }
                m.since = None;
            } else if over {
                m.left = None;
                if !m.open {
                    let since = *m.since.get_or_insert(now);
                    if now - since >= OPEN_DELAY {
                        m.open = true;
                        m.pinned = false;
                    } else {
                        ctx.request_repaint_after(std::time::Duration::from_millis(30));
                    }
                }
            } else {
                m.since = None;
                if m.open {
                    let left = *m.left.get_or_insert(now);
                    if now - left >= CLOSE_DELAY {
                        m.open = false;
                        m.pinned = false;
                        m.left = None;
                    } else {
                        ctx.request_repaint_after(std::time::Duration::from_millis(30));
                    }
                }
                if pressed && m.open {
                    m.open = false;
                    m.pinned = false;
                }
            }
            if m.open {
                let area = egui::Area::new(egui::Id::NULL.with(("re_part_menu", k))).fade_in(false).order(egui::Order::Foreground).pivot(Align2::LEFT_BOTTOM).fixed_pos(rect.left_top() - Vec2::new(0.0, 4.0)).show(ctx, |ui| {
                    egui::Frame::new().fill(tok::PANEL).stroke(Stroke::new(1.0, tok::LINE_STRONG)).corner_radius(3.0).inner_margin(5.0).show(ui, |ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(4.0, 4.0);
                        match which {
                            MenuKey::Scan => self.scan_menu(ui, sketch),
                            MenuKey::Bodies => self.bodies_menu(ui),
                        }
                    });
                });
                m.area = Some(area.response.rect);
            } else {
                m.area = None;
            }
            self.vb.menus[k] = m;
        }
    }

    /// The middle of what is on the stage, in the part's frame.
    fn stage_centre(&self) -> DVec3 {
        let mut b = mm_core::Aabb::EMPTY;
        for o in self.scan.iter().filter(|o| o.visible) {
            let ob = o.mesh.bounds();
            if !ob.is_empty() {
                b.include(self.to_part(ob.min));
                b.include(self.to_part(ob.max));
            }
        }
        for body in &self.bodies {
            for p in [body.part_mesh.positions.iter().fold(DVec3::splat(f64::INFINITY), |a, p| a.min(*p)), body.part_mesh.positions.iter().fold(DVec3::splat(f64::NEG_INFINITY), |a, p| a.max(*p))] {
                if p.is_finite() {
                    b.include(p);
                }
            }
        }
        if b.is_empty() {
            DVec3::ZERO
        } else {
            b.center()
        }
    }

    fn scan_key_ui(&mut self, ui: &mut egui::Ui, sketch: bool) -> (Rect, bool) {
        let disabled = if self.scan.is_empty() {
            Some(words::SCAN_NONE)
        } else if self.bodies.is_empty() {
            Some(words::SCAN_NO_BODY)
        } else {
            None
        };
        let any = self.any_body_shown();
        let changed = self.vb.scan_see_through.is_some_and(|v| v != any) || self.vb.scan_mesh || (!sketch && self.vb.curvature_map);
        let on = !self.scan.is_empty() && self.show_scan;
        let (r, caret) = part_key(ui, Glyph::Scan, "Scan", on, disabled.is_none(), changed, "How the scan is drawn");
        let r = r.on_hover_text(disabled.unwrap_or(words::SCAN));
        if r.clicked() && disabled.is_none() {
            self.show_scan = !self.show_scan;
        }
        (r.rect.union(caret.rect), caret.clicked())
    }

    fn scan_menu(&mut self, ui: &mut egui::Ui, sketch: bool) {
        let any = self.any_body_shown();
        let see = self.vb.scan_see_through.unwrap_or(any);
        if option(ui, Glyph::Transparent, "Transparent", see, true).on_hover_text(words::SCAN_TRANSPARENT).clicked() {
            self.vb.scan_see_through = Some(!see);
        }
        if option(ui, Glyph::Mesh, "Mesh", self.vb.scan_mesh, true).on_hover_text(words::SCAN_MESH).clicked() {
            self.vb.scan_mesh = !self.vb.scan_mesh;
        }
        option(ui, Glyph::Backfaces, "Backfaces", false, false).on_disabled_hover_text(words::SCAN_BACKFACES_NOT_YET).on_hover_text(words::SCAN_BACKFACES);
        if !sketch {
            let can = !self.scan.is_empty();
            let r = option(ui, Glyph::Curvature, "Curvature", self.vb.curvature_map, can);
            let r = if can { r.on_hover_text(words::CURVATURE) } else { r.on_disabled_hover_text(words::CURVATURE_NO_SCAN) };
            if r.clicked() && can {
                self.vb.curvature_map = !self.vb.curvature_map;
                if self.vb.curvature_map {
                    self.deviation.on = false;
                }
            }
        }
    }

    fn bodies_key_ui(&mut self, ui: &mut egui::Ui) -> (Rect, bool) {
        let has = !self.bodies.is_empty();
        let changed = self.vb.bodies_see_through || !self.part.overlaps_shown || self.vb.tangent_edges != TangentEdges::Visible;
        let (r, caret) = part_key(ui, Glyph::Combine, "Bodies", self.show_bodies && has, has, changed, "How the bodies are drawn");
        let r = r.on_hover_text(if has { words::BODIES } else { words::BODIES_NONE });
        if r.clicked() && has {
            self.show_bodies = !self.show_bodies;
        }
        (r.rect.union(caret.rect), caret.clicked())
    }

    fn bodies_menu(&mut self, ui: &mut egui::Ui) {
        if option(ui, Glyph::Transparent, "Transparent", self.vb.bodies_see_through, true).on_hover_text(words::BODIES_TRANSPARENT).clicked() {
            self.vb.bodies_see_through = !self.vb.bodies_see_through;
        }
        // ---- PICKLIST ---- the part's Overlaps switch, saved with it.
        if option(ui, Glyph::OpIntersect, "Overlaps", self.part.overlaps_shown, true).on_hover_text(words::OVERLAPS).clicked() {
            self.set_overlaps_shown(!self.part.overlaps_shown);
        }
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(Vec2::new(110.0, 24.0), Sense::hover());
            paint_glyph(ui.painter(), Rect::from_min_size(r.min + Vec2::new(4.0, 4.0), Vec2::splat(16.0)), Glyph::TangentEdges, tok::DIM);
            ui.painter().text(r.min + Vec2::new(26.0, 12.0), Align2::LEFT_CENTER, "Tangent edges", FontId::proportional(12.0), tok::DIM);
            r_hover(ui, r, words::TANGENT);
            for (mode, label, tip) in [(TangentEdges::Visible, "Drawn", words::TANGENT_DRAWN), (TangentEdges::Dimmed, "Dimmed", words::TANGENT_DIMMED), (TangentEdges::Removed, "Off", words::TANGENT_OFF)] {
                if small_key(ui, label, self.vb.tangent_edges == mode).on_hover_text(tip).clicked() {
                    self.vb.tangent_edges = mode;
                }
            }
        });
    }

    fn elements_key_ui(&mut self, ui: &mut egui::Ui) {
        let has = !self.part.elements.is_empty();
        let r = key(ui, Glyph::ElCylinder, "Elements", self.vb.elements_shown && has, has, false).on_hover_text(if has { words::ELEMENTS } else { words::ELEMENTS_NONE });
        if r.clicked() && has {
            self.vb.elements_shown = !self.vb.elements_shown;
        }
    }

    fn part_views_ui(&mut self, ui: &mut egui::Ui) {
        let now = if !self.vb.sketch_model_shown {
            0
        } else if self.vb.sketch_sliced {
            1
        } else {
            2
        };
        for (i, glyph, label, tip) in [(0, Glyph::EyeOff, "No part", words::NO_PART), (1, Glyph::Slice, "Section", words::CUT_PART), (2, Glyph::Model, "Full part", words::FULL_PART)] {
            let r = key(ui, glyph, label, now == i, true, false).on_hover_text(tip);
            if r.clicked() {
                self.vb.sketch_model_shown = i != 0;
                if i != 0 {
                    self.vb.sketch_sliced = i == 1;
                }
            }
        }
    }

    /// The section view's own row: what the plane is laid across, where along it, square, flip.
    fn section_row(&mut self, ui: &mut egui::Ui) {
        group(ui, |ui| {
            let elements: Vec<(u32, String)> = self.part.elements.iter().filter(|e| element_cut_axis(e).is_some()).map(|e| (e.id, e.name.clone())).collect();
            let reference = self.vb.section.reference;
            let orphan = matches!(reference, SectionRef::Element(id) if !elements.iter().any(|e| e.0 == id));
            let name = |r: SectionRef| match r {
                SectionRef::World(a) => a.plane_name().to_string(),
                SectionRef::Element(id) => elements.iter().find(|e| e.0 == id).map_or("(deleted element)".into(), |e| e.1.clone()),
            };
            ui.label(egui::RichText::new("Across").size(12.0).color(tok::DIM)).on_hover_text(words::ACROSS);
            let mut chosen = None;
            egui::ComboBox::from_id_salt("re_section_ref").width(120.0).selected_text(if orphan { "(deleted element)".to_string() } else { name(reference) }).show_ui(ui, |ui| {
                for a in [WorldAxis::Z, WorldAxis::Y, WorldAxis::X] {
                    if ui.selectable_label(reference == SectionRef::World(a), a.plane_name()).clicked() {
                        chosen = Some(SectionRef::World(a));
                    }
                }
                for (id, n) in &elements {
                    if ui.selectable_label(reference == SectionRef::Element(*id), n).clicked() {
                        chosen = Some(SectionRef::Element(*id));
                    }
                }
            });
            if let Some(r) = chosen {
                let view = self.to_part(self.view.camera.target - self.view.camera.eye()).normalize_or_zero();
                let elements = self.part.elements.clone();
                self.vb.section.set_ref(r, Some(view), &elements);
                self.vb.offset_text = None;
            }
            let tilt = self.vb.section.tilt_deg();
            let label = match (reference, tilt == 0.0) {
                (SectionRef::World(a), true) => match a {
                    WorldAxis::X => "X",
                    WorldAxis::Y => "Y",
                    WorldAxis::Z => "Z",
                },
                _ => "Offset",
            };
            ui.label(egui::RichText::new(label).size(12.0).color(tok::DIM)).on_hover_text(words::OFFSET);
            let shown = mm_reverse::js_fmt::num((self.vb.section.offset * 1000.0).round() / 1000.0);
            let mut text = self.vb.offset_text.clone().unwrap_or_else(|| shown.clone());
            let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(64.0)).on_hover_text(words::OFFSET);
            if r.changed() {
                self.vb.offset_text = Some(text.clone());
            }
            if r.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) && r.has_focus() {
                if text.trim() != shown {
                    if let Ok(v) = text.trim().parse::<f64>() {
                        self.vb.section.set_offset(v);
                    }
                }
                self.vb.offset_text = None;
            }
            ui.label(egui::RichText::new("mm").italics().size(12.0).color(tok::DIM));
            if tilt > 0.0 && !orphan && small_key(ui, &format!("Square {}°", mm_reverse::js_fmt::to_fixed(tilt, 1)), false).on_hover_text(words::SQUARE).clicked() {
                let elements = self.part.elements.clone();
                self.vb.section.set_ref(reference, None, &elements);
            }
            if key(ui, Glyph::Flip, "Flip", false, true, false).on_hover_text(words::FLIP).clicked() {
                self.vb.section.flip();
            }
        });
    }

    /// The live map's row in every feature box: the switch, its range and what it read.
    pub(crate) fn analysis_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.vb.live_map, "Deviation map").changed() {
                self.vb.map.key = None;
            }
            let r = ui.small_button("ⓘ");
            r.on_hover_text(format!("Deviation map\n\n{}", words::DEVIATION_MAP));
        });
        if !self.vb.live_map {
            return;
        }
        ui.horizontal(|ui| {
            ui.label("Range ±");
            let r = ui.add(egui::DragValue::new(&mut self.deviation.range).range(mm_reverse::compare::LIVE_MAP_MIN_RANGE_MM..=100.0).speed(0.005).max_decimals(3).suffix(" mm"));
            if r.changed() {
                self.vb.map.key = None;
            }
        });
        let preview = self.preview_feature.is_some() && !self.building();
        let hint = match self.vb.map.report {
            _ if !preview && self.vb.map.job.is_none() => "The map appears with the ghost.".to_string(),
            _ if self.vb.map.job.is_some() || self.vb.map.pending.is_some() => "Reading the scan against the ghost…".to_string(),
            Some(r) if r.ghost && r.points == 0 => "No scan near the ghost.".to_string(),
            Some(r) if r.ghost => format!("{} of {} scan points near the ghost lie within ±{} mm of it.", grouped(r.within), grouped(r.points), mm_reverse::js_fmt::num(r.range)),
            _ => "The map appears with the ghost.".to_string(),
        };
        ui.label(egui::RichText::new(hint).size(11.5).color(tok::DIM));
    }
}

/// A count as en-US writes it, grouped by thousands.
fn grouped(n: usize) -> String {
    mm_reverse::js_fmt::grouped(n as f64)
}

// ---- the keys ----

/// The view bar's pictures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Glyph {
    Scan,
    Combine,
    ElCylinder,
    Compare,
    SectionView,
    Transparent,
    Mesh,
    Backfaces,
    Curvature,
    OpIntersect,
    TangentEdges,
    EyeOff,
    Slice,
    Model,
    Flip,
}

fn paint_glyph(painter: &egui::Painter, rect: Rect, g: Glyph, color: Color32) {
    let s = rect.width().min(rect.height());
    let o = rect.center() - Vec2::splat(s / 2.0);
    let p = |x: f32, y: f32| Pos2::new(o.x + x * s, o.y + y * s);
    let st = Stroke::new(1.4, color);
    let line = |pts: Vec<Pos2>| {
        painter.add(egui::Shape::line(pts, st));
    };
    match g {
        Glyph::Scan => {
            line(vec![p(0.1, 0.7), p(0.3, 0.45), p(0.5, 0.6), p(0.7, 0.35), p(0.9, 0.5)]);
            line(vec![p(0.1, 0.85), p(0.3, 0.62), p(0.5, 0.76), p(0.7, 0.52), p(0.9, 0.66)]);
            for x in [0.3, 0.5, 0.7] {
                painter.circle_filled(p(x, 0.2), 1.1, color);
            }
        }
        Glyph::Combine => {
            painter.rect_stroke(Rect::from_min_max(p(0.1, 0.3), p(0.6, 0.85)), 1.0, st, StrokeKind::Middle);
            painter.rect_stroke(Rect::from_min_max(p(0.4, 0.12), p(0.9, 0.65)), 1.0, st, StrokeKind::Middle);
        }
        Glyph::ElCylinder => {
            painter.add(egui::Shape::ellipse_stroke(p(0.5, 0.22), Vec2::new(0.32 * s, 0.1 * s), st));
            line(vec![p(0.18, 0.22), p(0.18, 0.78)]);
            line(vec![p(0.82, 0.22), p(0.82, 0.78)]);
            line(vec![p(0.18, 0.78), p(0.35, 0.86), p(0.65, 0.86), p(0.82, 0.78)]);
        }
        Glyph::Compare => {
            line(vec![p(0.1, 0.75), p(0.35, 0.35), p(0.6, 0.6), p(0.9, 0.2)]);
            painter.add(egui::Shape::dashed_line(&[p(0.1, 0.6), p(0.35, 0.25), p(0.6, 0.48), p(0.9, 0.1)], Stroke::new(1.0, color), 2.0, 2.0));
            line(vec![p(0.1, 0.9), p(0.9, 0.9)]);
        }
        Glyph::SectionView => {
            painter.rect_stroke(Rect::from_min_max(p(0.15, 0.15), p(0.85, 0.85)), 1.0, st, StrokeKind::Middle);
            painter.add(egui::Shape::convex_polygon(vec![p(0.15, 0.85), p(0.85, 0.15), p(0.85, 0.85)], color.gamma_multiply(0.45), Stroke::NONE));
        }
        Glyph::Transparent => {
            painter.rect_stroke(Rect::from_min_max(p(0.1, 0.1), p(0.62, 0.62)), 1.0, st, StrokeKind::Middle);
            painter.rect_filled(Rect::from_min_max(p(0.38, 0.38), p(0.9, 0.9)), 1.0, color.gamma_multiply(0.35));
        }
        Glyph::Mesh => {
            line(vec![p(0.1, 0.85), p(0.5, 0.15), p(0.9, 0.85), p(0.1, 0.85)]);
            line(vec![p(0.3, 0.5), p(0.7, 0.5), p(0.5, 0.85), p(0.3, 0.5)]);
        }
        Glyph::Backfaces => {
            line(vec![p(0.1, 0.85), p(0.5, 0.15), p(0.9, 0.85), p(0.1, 0.85)]);
            painter.add(egui::Shape::convex_polygon(vec![p(0.5, 0.15), p(0.9, 0.85), p(0.5, 0.85)], color.gamma_multiply(0.45), Stroke::NONE));
        }
        Glyph::Curvature => {
            let pts: Vec<Pos2> = (0..=16).map(|i| {
                let t = std::f32::consts::PI * (1.0 + i as f32 / 16.0);
                p(0.5 + 0.38 * t.cos(), 0.8 + 0.55 * t.sin())
            }).collect();
            line(pts);
            line(vec![p(0.5, 0.25), p(0.5, 0.05)]);
        }
        Glyph::OpIntersect => {
            painter.circle_stroke(p(0.38, 0.5), 0.3 * s, st);
            painter.circle_stroke(p(0.62, 0.5), 0.3 * s, st);
            painter.add(egui::Shape::convex_polygon(vec![p(0.5, 0.26), p(0.6, 0.5), p(0.5, 0.74), p(0.4, 0.5)], color.gamma_multiply(0.5), Stroke::NONE));
        }
        Glyph::TangentEdges => {
            line(vec![p(0.1, 0.85), p(0.5, 0.85)]);
            let pts: Vec<Pos2> = (0..=10).map(|i| {
                let t = std::f32::consts::FRAC_PI_2 * (i as f32 / 10.0);
                p(0.5 + 0.4 * t.sin(), 0.45 + 0.4 * t.cos())
            }).collect();
            line(pts);
            line(vec![p(0.9, 0.45), p(0.9, 0.1)]);
            painter.circle_filled(p(0.5, 0.85), 1.6, color);
        }
        Glyph::EyeOff => {
            painter.add(egui::Shape::ellipse_stroke(p(0.5, 0.5), Vec2::new(0.4 * s, 0.22 * s), st));
            painter.circle_stroke(p(0.5, 0.5), 0.1 * s, st);
            line(vec![p(0.15, 0.85), p(0.85, 0.15)]);
        }
        Glyph::Slice => {
            painter.rect_stroke(Rect::from_min_max(p(0.2, 0.2), p(0.8, 0.8)), 1.0, Stroke::new(1.0, color.gamma_multiply(0.6)), StrokeKind::Middle);
            line(vec![p(0.05, 0.5), p(0.95, 0.5)]);
            painter.rect_filled(Rect::from_min_max(p(0.2, 0.5), p(0.8, 0.8)), 0.0, color.gamma_multiply(0.35));
        }
        Glyph::Model => {
            let f = [p(0.12, 0.35), p(0.62, 0.35), p(0.62, 0.88), p(0.12, 0.88)];
            line(vec![f[0], f[1], f[2], f[3], f[0]]);
            line(vec![f[0], p(0.38, 0.12), p(0.88, 0.12), f[1]]);
            line(vec![p(0.88, 0.12), p(0.88, 0.65), f[2]]);
        }
        Glyph::Flip => {
            line(vec![p(0.1, 0.35), p(0.85, 0.35)]);
            line(vec![p(0.7, 0.2), p(0.85, 0.35), p(0.7, 0.5)]);
            line(vec![p(0.9, 0.7), p(0.15, 0.7)]);
            line(vec![p(0.3, 0.55), p(0.15, 0.7), p(0.3, 0.85)]);
        }
    }
}

/// A block of keys.
fn group<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new()
        .fill(tok::PANEL.gamma_multiply(0.94))
        .stroke(Stroke::new(1.0, tok::LINE))
        .corner_radius(4.0)
        .inner_margin(4.0)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::new(3.0, 3.0);
            ui.horizontal(add).inner
        })
        .inner
}

fn r_hover(ui: &mut egui::Ui, r: Rect, tip: &str) {
    ui.interact(r, egui::Id::NULL.with(("re_hover", r.min.x as i32, r.min.y as i32)), Sense::hover()).on_hover_text(tip);
}

/// A key: icon and word, pressed while on; greyed when it cannot be clicked; a dot when
/// something in its menu stands otherwise than the workspace opens with.
fn key(ui: &mut egui::Ui, glyph: Glyph, label: &str, on: bool, enabled: bool, dot: bool) -> Response {
    let font = FontId::proportional(12.5);
    let galley = ui.painter().layout_no_wrap(label.to_string(), font, tok::INK);
    let size = Vec2::new(8.0 + 16.0 + 5.0 + galley.size().x + 9.0, 26.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let hovered = resp.hovered() && enabled;
    let (fill, stroke) = if on { (tok::ACCENT_SOFT, tok::ACCENT) } else if hovered { (tok::ACCENT_SOFT.gamma_multiply(0.6), tok::LINE_STRONG) } else { (tok::WELL, tok::LINE) };
    ui.painter().rect(rect, 3.0, fill, Stroke::new(1.0, stroke), StrokeKind::Inside);
    let fg = if !enabled { tok::FAINT } else if on { tok::ACCENT_D } else { tok::INK };
    paint_glyph(ui.painter(), Rect::from_min_size(rect.min + Vec2::new(8.0, 5.0), Vec2::splat(16.0)), glyph, fg);
    ui.painter().text(rect.min + Vec2::new(29.0, 13.0), Align2::LEFT_CENTER, label, FontId::proportional(12.5), fg);
    if dot {
        ui.painter().circle_filled(rect.right_top() + Vec2::new(-5.0, 5.0), 2.5, tok::AMBER);
    }
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

/// A part's key and the caret that opens its menu.
fn part_key(ui: &mut egui::Ui, glyph: Glyph, label: &str, on: bool, enabled: bool, dot: bool, caret_tip: &str) -> (Response, Response) {
    let r = key(ui, glyph, label, on, enabled, dot);
    let (rect, caret) = ui.allocate_exact_size(Vec2::new(14.0, 26.0), Sense::click());
    let fill = if on { tok::ACCENT_SOFT } else { tok::WELL };
    ui.painter().rect(rect, 3.0, fill, Stroke::new(1.0, if on { tok::ACCENT } else { tok::LINE }), StrokeKind::Inside);
    let c = rect.center();
    ui.painter().add(egui::Shape::line(vec![c + Vec2::new(-3.0, 1.5), c + Vec2::new(0.0, -1.5), c + Vec2::new(3.0, 1.5)], Stroke::new(1.5, if on { tok::ACCENT_D } else { tok::DIM })));
    let caret = caret.on_hover_text(caret_tip);
    (r, caret)
}

/// A switch in a part's menu, as a key of its own: pressed while on; greyed, with its reason
/// for a tip, when it cannot be switched.
fn option(ui: &mut egui::Ui, glyph: Glyph, label: &str, on: bool, enabled: bool) -> Response {
    ui.add_enabled_ui(enabled, |ui| {
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(170.0, 24.0), Sense::click());
        let hovered = resp.hovered() && enabled;
        let (fill, stroke) = if on { (tok::ACCENT_SOFT, tok::ACCENT) } else if hovered { (tok::ACCENT_SOFT.gamma_multiply(0.6), tok::LINE_STRONG) } else { (tok::WELL, tok::LINE) };
        ui.painter().rect(rect, 3.0, fill, Stroke::new(1.0, stroke), StrokeKind::Inside);
        let fg = if !enabled { tok::FAINT } else if on { tok::ACCENT_D } else { tok::INK };
        paint_glyph(ui.painter(), Rect::from_min_size(rect.min + Vec2::new(6.0, 4.0), Vec2::splat(16.0)), glyph, fg);
        ui.painter().text(rect.min + Vec2::new(28.0, 12.0), Align2::LEFT_CENTER, label, FontId::proportional(12.0), fg);
        resp
    })
    .inner
}

/// A small radio key in a choice row.
fn small_key(ui: &mut egui::Ui, label: &str, on: bool) -> Response {
    let galley = ui.painter().layout_no_wrap(label.to_string(), FontId::proportional(11.5), tok::INK);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(galley.size().x + 14.0, 24.0), Sense::click());
    let (fill, stroke) = if on { (tok::ACCENT_SOFT, tok::ACCENT) } else if resp.hovered() { (tok::ACCENT_SOFT.gamma_multiply(0.6), tok::LINE_STRONG) } else { (tok::WELL, tok::LINE) };
    ui.painter().rect(rect, 3.0, fill, Stroke::new(1.0, stroke), StrokeKind::Inside);
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, if on { tok::ACCENT_D } else { tok::INK });
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_reverse::model::{ExtrudeExtent, ExtrudeFeature};

    #[test]
    fn a_segment_is_cut_where_the_plane_takes_the_half_its_normal_points_to() {
        let plane = Some((DVec3::Z, 0.0));
        let (a, b) = clip_segment(DVec3::new(0.0, 0.0, -1.0), DVec3::new(0.0, 0.0, 1.0), plane).unwrap();
        assert_eq!(a, DVec3::new(0.0, 0.0, -1.0));
        assert!(b.z.abs() < 1e-12);
        assert!(clip_segment(DVec3::new(0.0, 0.0, 1.0), DVec3::new(1.0, 0.0, 2.0), plane).is_none());
        assert!(clip_segment(DVec3::new(0.0, 0.0, -1.0), DVec3::new(1.0, 0.0, -2.0), plane).is_some());
        assert!(clip_segment(DVec3::ZERO, DVec3::X, None).is_some());
    }

    #[test]
    fn a_join_or_a_cut_has_its_tool_for_a_ghost_a_new_body_none() {
        let e = |op| FeatureKind::Extrude(ExtrudeFeature::new(1, vec![], ExtrudeExtent::Distance { from: 0.0, to: 1.0 }, op));
        match as_tool(&e(BodyOp::Cut)) {
            Some(FeatureKind::Extrude(x)) => assert_eq!(x.op, BodyOp::New),
            _ => panic!("a cut's ghost is its tool"),
        }
        assert!(as_tool(&e(BodyOp::New)).is_none());
    }

    #[test]
    fn a_triangle_wears_the_mean_of_its_painted_corners_or_the_violet_or_its_own_colour() {
        let p = ScanPart { id: 1, positions: vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::ONE], triangles: vec![[0, 1, 2], [1, 3, 2]], tri_ids: vec![0, 1], max_tid: 2, live_vertices: vec![0, 1, 2, 3] };
        let values = [0.0f32, 0.0, f32::NAN, f32::NAN];
        let deep = [false, false, true, true];
        let colors = triangle_colors(&p, &values, &deep, [10, 20, 30], &|v| if v == 0.0 { [0, 200, 0] } else { [255, 0, 0] });
        assert_eq!(colors[0], rgb_u32([0, 200, 0]));
        assert_eq!(colors[1], rgb_u32(FAR_INSIDE_RGB));
        let none = triangle_colors(&p, &[f32::NAN; 4], &[], [10, 20, 30], &|_| [0, 0, 0]);
        assert_eq!(none[0], rgb_u32([10, 20, 30]));
    }
}
