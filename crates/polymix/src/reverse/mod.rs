// SPDX-License-Identifier: AGPL-3.0-only
//! The Reverse Engineer workspace: from a scan to a solid STEP body. A Rust port of the
//! Reverse engineer workspace of ScanRuler (<https://github.com/CNCKitchen/scanruler>), ©
//! Stefan Hermann (CNC Kitchen), AGPL-3.0-only; the scan-reading core is in `mm-reverse`, the
//! solid modelling runs on the SolidCraft kernel instead of Open CASCADE.
//!
//! It is built the way every reverse-engineering tool builds a prismatic part: sketch the
//! profile on a section through the scan, close it, extrude or revolve it, and check it
//! against the scan. **Fit all chains** reads the slice as lines and arcs whose corners are
//! joined, and the solver holds them to the slice points while every relation and dimension
//! is met exactly; a click on the scan while an extrude is open puts its end at that height;
//! the deviation map colours the scan by how far it lies from the bodies, and skin deep
//! inside the bodies (a pocket the model lacks) violet. No scan is needed to model: on a
//! blank stage it works as a small parametric CAD tool.
//!
//! The scan is the Mesh workspace's (every object it holds), read in place: the part is
//! modelled in the files' own axes (Z up), the view shows it as Mesh does. Bodies go back to
//! Mesh as meshes (**Send to Mesh**), or out as a STEP file.

mod build_service;
mod checks;
mod editors;
mod network_layer; // ---- NET ----
mod params;
mod sketch_box;
mod sketch_icons;
mod sketch_paint;
mod sketch_project;
mod sketcher;
mod view;
// ---- PICKLIST ----
mod bodies;
mod chips;
mod pick_steps;
mod picks;
mod viewbar;

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

use eframe::egui_wgpu;
use egui::{Color32, RichText};
use glam::{DVec2, DVec3, Vec4};
use mm_core::bvh::Bvh;
use mm_core::Mesh;
use mm_reverse::build::{default_regions, describe_plane_ref, export_step, resolve_plane_frame, BodyMesh, Built};
use mm_reverse::frame::{SectionFrame, WorldAxis};
use mm_reverse::model::{plane_offset, plane_takes_offset, with_kept_queries, with_plane_offset, AxisRef, BodyId, BodyOp, ExtrudeExtent, ExtrudeFeature, Feature, FeatureId, FeatureKind, Part, PlaneRef, RevolveFeature, SketchFeature};
use mm_reverse::regions::find_regions;
use mm_reverse::section::{section_chains, FitOptions};
use mm_reverse::sketch::Sketch;
use mm_reverse::topo::Topology;
use polymix_workspace::Workspace;

use crate::navigation::NavBindings;
use sketcher::{plural, SketchEdit};

/// One object of the Mesh workspace's scene, as the scan this workspace models over.
#[derive(Clone)]
pub struct ScanObject {
    pub id: u64,
    pub name: String,
    pub mesh: Arc<Mesh>,
    pub visible: bool,
    pub color: Vec4,
}

/// How the Mesh workspace's view is moved — mouse, wheel and 3D mouse (its settings and what
/// its buttons do, Preferences → 3D mouse) — so this one moves the same way.
#[derive(Clone)]
pub struct NavSettings {
    pub bindings: NavBindings,
    pub zoom_speed: f32,
    pub zoom_to_cursor: bool,
    pub background: [f32; 3],
    pub space: crate::spacemouse::Settings,
    pub space_buttons: Vec<(u32, crate::spacemouse::ButtonAction)>,
}

impl Default for NavSettings {
    fn default() -> Self {
        Self {
            bindings: NavBindings::FUSION360,
            zoom_speed: 1.0,
            zoom_to_cursor: true,
            background: crate::renderer::DEFAULT_BACKGROUND,
            space: crate::spacemouse::Settings::default(),
            space_buttons: crate::spacemouse::default_buttons(),
        }
    }
}

/// A body as the view shows it: its mesh in scene coordinates (each triangle's group the
/// B-rep face it lies on) and its edges.
#[derive(Clone)]
pub(crate) struct BodyView {
    pub id: BodyId,
    pub name: String,
    pub mesh: Arc<Mesh>,
    pub edges: Vec<Vec<DVec3>>,
    pub volume: f64,
    /// The tessellation in part coordinates (deviation, Send to Mesh).
    pub part_mesh: Arc<BodyMesh>,
    /// Every face and edge with its fingerprint, in part coordinates: what a pick stores.
    pub topology: Arc<Topology>,
}

/// The scan in part coordinates, per object (vertex ids kept), for slicing and the map.
pub(crate) struct ScanPart {
    pub id: u64,
    /// Indexed by vertex id (dead ids hold whatever their slot held).
    pub positions: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
    pub tri_ids: Vec<u32>,
    pub max_tid: usize,
    pub live_vertices: Vec<u32>,
}

/// What the workspace is doing.
pub(crate) enum Mode {
    Idle,
    Sketch(Box<SketchEdit>),
    Feature(Box<FeatureEdit>),
}

/// What a click in the viewport picks while a feature box is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PickWhat {
    /// A sketch's regions (Extrude, Revolve).
    Regions,
    /// A height on the scan or a body, for the extrude's end.
    Height,
    /// Edges of a body (Fillet, Chamfer).
    Edges,
    /// Faces to open (Shell).
    Faces,
    /// A body (Shell, Mirror; Combine's target and tool).
    #[allow(dead_code)]
    Body,
    #[allow(dead_code)]
    Tool,
    Nothing,
    /// Whatever an editor of its own picks, by a number it gives the meaning of.
    #[allow(dead_code)]
    Custom(u32),
}

/// A feature box: the feature as it will be, previewed by building the part with it.
pub(crate) struct FeatureEdit {
    /// The feature being edited, or `None` for a new one.
    pub editing: Option<FeatureId>,
    pub kind: FeatureKind,
    pub pick: PickWhat,
}

/// The deviation map's settings and its latest reading.
pub(crate) struct DeviationView {
    /// The Deviation key: the scan coloured against the committed bodies.
    pub on: bool,
    /// The live map's range, ± mm (remembered).
    pub range: f64,
    /// The scan, the bodies and the range the map was read for; `None` reads it again.
    pub key: Option<(u64, u64, u64)>,
    /// Per object, the triangle colours shown (stamp, colours by triangle id).
    pub colors: HashMap<u64, (u64, Arc<Vec<u32>>)>,
    pub stamp: u64,
}

pub struct ReverseWorkspace {
    pub(crate) part: Part,
    undo: Vec<Part>,
    redo: Vec<Part>,
    /// The part as last built, and its bodies as shown.
    pub(crate) built: Built,
    pub(crate) bodies: Vec<BodyView>,
    /// While a feature box is open, the bodies as they stood before its feature: what a click
    /// picks faces, edges and bodies on, as ScanRuler builds only the features before the one
    /// being edited (`featuresToBuild`) — a pick is the face or edge the feature will find
    /// again on the body as it stands before it, never one its own preview made. `None` until
    /// such a build comes in; see `ReverseWorkspace::pick_set`.
    pub(crate) pick_bodies: Option<Vec<BodyView>>,
    build_generation: u64,
    built_generation: u64,
    /// The part the last build request was for (to ask again only on a change).
    built_for: Option<String>,
    /// The build thread, what it was asked and when, and where the CAD kernel stands.
    pub(crate) builds: build_service::Builds,
    /// The feature the open feature box previews, in the part as shown.
    pub(crate) preview_feature: Option<FeatureId>,
    pub(crate) hidden_bodies: HashSet<BodyId>,
    pub(crate) mode: Mode,
    pub(crate) scan: Vec<ScanObject>,
    pub(crate) flip: bool,
    scan_key: u64,
    pub(crate) scan_parts: Option<(u64, Arc<Vec<ScanPart>>)>,
    pub(crate) scan_bvh: HashMap<u64, (u64, Arc<Bvh>)>,
    bvh_jobs: HashMap<u64, (u64, Receiver<Bvh>)>,
    pub(crate) nav: NavSettings,
    pub(crate) view: view::ViewState,
    pub(crate) show_scan: bool,
    pub(crate) scan_see_through: bool,
    pub(crate) show_bodies: bool,
    pub(crate) show_slice: bool,
    pub(crate) deviation: DeviationView,
    pub(crate) fit_options: FitOptions,
    pub(crate) sketch_plane: (WorldAxis, bool),
    /// ---- PLANE ---- the elements, the sections and the planes a sketch is offered (`elements.rs`).
    pub(crate) elements: elements::ElementsUi,
    pub(crate) status: String,
    to_mesh: Vec<(String, Mesh)>,
    script: VecDeque<String>,
    by_agent: bool,
    panel_width: f32,
    autosave_due: bool,
    pub(crate) renderer: Option<crate::renderer::SceneRenderer>,
    // ---- NET ----
    /// The patch network's box and layer, while its box is open.
    pub(crate) net: network_layer::NetState,
    /// ---- PICKLIST ---- the selection, its chips and the body list (`picks.rs`).
    pub(crate) picks: picks::PickState,
    /// The view bar, the maps and the section view (viewbar.rs).
    pub(crate) vb: viewbar::ViewBar,
    /// The parameters' panel and the display units (`params.rs`).
    pub(crate) params: params::ParamsState,
}

/// Where the part is kept between sessions.
fn part_file() -> Option<PathBuf> {
    polymix_workspace::data_dir().map(|d| d.join("reverse").join("part.json"))
}

impl ReverseWorkspace {
    pub fn new(_ctx: &egui::Context, rs: Option<&egui_wgpu::RenderState>, storage: Option<&dyn eframe::Storage>) -> Self {
        let scripted = polymix_workspace::scripted();
        // A part saved by an earlier model of the workspace is not read: a new part is started,
        // and the status says so.
        let saved = if scripted { None } else { part_file().and_then(|p| std::fs::read_to_string(p).ok()).map(|t| Part::from_json(&t)) };
        let refused = match &saved {
            Some(Err(e)) => Some(e.clone()),
            _ => None,
        };
        let part = saved.and_then(|r| r.ok());
        let mut fit_options = FitOptions::default();
        let mut deviation_range = 0.1;
        if let Some(st) = storage.filter(|_| !scripted) {
            if let Some(d) = st.get_string("reverse.fit_detail").and_then(|v| v.parse().ok()) {
                fit_options.detail = d;
            }
            fit_options.join_corners = st.get_string("reverse.fit_join").map_or(true, |v| v == "1");
            fit_options.tolerance = st.get_string("reverse.fit_tolerance").and_then(|v| v.parse().ok()).filter(|t: &f64| *t > 0.0);
            if let Some(r) = st.get_string("reverse.deviation_range").and_then(|v| v.parse().ok()).filter(|r: &f64| *r > 0.0) {
                deviation_range = r;
            }
        }
        let restored = part.is_some();
        sketcher::load_prefs(storage); // ---- SKETCH ----
        let mut w = ReverseWorkspace {
            part: part.unwrap_or_default(),
            undo: Vec::new(),
            redo: Vec::new(),
            built: Built::default(),
            bodies: Vec::new(),
            pick_bodies: None,
            build_generation: 0,
            built_generation: 0,
            built_for: None,
            builds: build_service::Builds::default(),
            preview_feature: None,
            hidden_bodies: HashSet::new(),
            mode: Mode::Idle,
            scan: Vec::new(),
            flip: true,
            scan_key: 0,
            scan_parts: None,
            scan_bvh: HashMap::new(),
            bvh_jobs: HashMap::new(),
            nav: NavSettings::default(),
            view: view::ViewState::default(),
            show_scan: true,
            scan_see_through: false,
            show_bodies: true,
            show_slice: true,
            deviation: DeviationView { on: false, range: deviation_range, key: None, colors: HashMap::new(), stamp: 0 },
            fit_options,
            sketch_plane: (WorldAxis::Z, false),
            elements: elements::ElementsUi::default(),
            status: match (&refused, restored) {
                (Some(e), _) => format!("{e} A new part was started."),
                (None, true) => "The part from the last session is open. New part starts again.".into(),
                (None, false) => "Choose a plane under Sketch on to sketch on — over the Mesh workspace's scan, or on a blank stage.".into(),
            },
            to_mesh: Vec::new(),
            script: VecDeque::new(),
            by_agent: false,
            panel_width: 300.0,
            autosave_due: false,
            renderer: rs.map(crate::renderer::SceneRenderer::new),
            net: Default::default(), // ---- NET ----
            picks: picks::PickState::default(),
            vb: viewbar::ViewBar::new(storage),
            params: params::ParamsState::new(storage.filter(|_| !scripted)),
        };
        w.request_build();
        w
    }

    /// The scan as the Mesh workspace holds it now (called by the shell while this tab is
    /// shown), with how its view moves.
    pub fn set_scan(&mut self, scan: Vec<ScanObject>, flip: bool, nav: Option<NavSettings>) {
        if let Some(n) = nav {
            self.nav = n;
        }
        let key = scan.iter().fold(flip as u64 + 1, |h, o| {
            let p = Arc::as_ptr(&o.mesh) as u64;
            h.wrapping_mul(1_000_003).wrapping_add(p ^ o.mesh.shape_version().rotate_left(17) ^ o.mesh.topology_version().rotate_left(31) ^ (o.visible as u64))
        });
        self.scan = scan;
        self.flip = flip;
        if key != self.scan_key {
            self.scan_key = key;
            self.deviation.key = None;
            if !self.view.framed {
                self.view.want_fit = true;
            }
        }
    }

    /// Meshes sent to the Mesh workspace (Send to Mesh), taken by the shell.
    pub fn take_for_mesh(&mut self) -> Vec<(String, Mesh)> {
        std::mem::take(&mut self.to_mesh)
    }

    // ---- coordinates ----

    /// A part point (Z up, as in the files) in the scene the view shows.
    pub(crate) fn to_scene(&self, p: DVec3) -> DVec3 {
        if self.flip {
            crate::app::flip_zy_in(p)
        } else {
            p
        }
    }

    /// A scene point in the part.
    pub(crate) fn to_part(&self, p: DVec3) -> DVec3 {
        if self.flip {
            crate::app::flip_zy_out(p)
        } else {
            p
        }
    }

    /// The scan in part coordinates, per visible object, built on first use for this scan.
    pub(crate) fn scan_parts(&mut self) -> Arc<Vec<ScanPart>> {
        if let Some((k, parts)) = &self.scan_parts {
            if *k == self.scan_key {
                return parts.clone();
            }
        }
        let flip = self.flip;
        let parts: Vec<ScanPart> = self
            .scan
            .iter()
            .filter(|o| o.visible)
            .map(|o| {
                let m = &o.mesh;
                let positions: Vec<DVec3> = m.positions().iter().map(|p| if flip { crate::app::flip_zy_out(*p) } else { *p }).collect();
                let tri_ids: Vec<u32> = m.triangle_ids().collect();
                let triangles = tri_ids.iter().map(|t| m.triangle(*t)).collect();
                ScanPart { id: o.id, positions, triangles, tri_ids, max_tid: m.max_tid(), live_vertices: m.vertex_ids().collect() }
            })
            .collect();
        let parts = Arc::new(parts);
        self.scan_parts = Some((self.scan_key, parts.clone()));
        parts
    }

    /// The slice a plane cuts through the scan, laid flat on its sheet.
    pub(crate) fn slice(&mut self, frame: &SectionFrame) -> Vec<Vec<DVec2>> {
        let parts = self.scan_parts();
        parts.iter().flat_map(|p| section_chains(&p.positions, &p.triangles, frame)).collect()
    }

    /// The centre of the scan's bounding box, in part coordinates.
    pub(crate) fn scan_centre(&self) -> Option<DVec3> {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for o in self.scan.iter().filter(|o| o.visible) {
            let b = o.mesh.bounds();
            if b.is_empty() {
                continue;
            }
            for c in [b.min, b.max] {
                let p = self.to_part(c);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        lo.x.is_finite().then(|| (lo + hi) / 2.0)
    }

    // ---- the part and its build ----

    /// Keep the part for Undo before a change.
    fn checkpoint(&mut self) {
        self.undo.push(self.part.clone());
        if self.undo.len() > 100 {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.autosave_due = true;
    }

    fn undo(&mut self) {
        if let Some(p) = self.undo.pop() {
            self.redo.push(std::mem::replace(&mut self.part, p));
            self.status = "Undone.".into();
            self.request_build();
            self.autosave_due = true;
        }
    }

    fn redo(&mut self) {
        if let Some(p) = self.redo.pop() {
            self.undo.push(std::mem::replace(&mut self.part, p));
            self.status = "Redone.".into();
            self.request_build();
            self.autosave_due = true;
        }
    }

    /// The part as the view shows it: with the open feature box's feature in it.
    pub(crate) fn shown_part(&self) -> (Part, Option<FeatureId>) {
        let mut part = self.part.clone();
        let Mode::Feature(edit) = &self.mode else { return (part, None) };
        // ---- DRAFT ----
        if editors::previews_by_ghost(self, edit) {
            return (part, None);
        }
        // ---- LOFT ---- a Loft, Sweep, Pattern or Move not ready builds nothing yet: ScanRuler
        // previews no draft its Create waits on. An Extrude or Revolve box not ready builds
        // nothing either (EXT).
        let waits = editors::loft_ready(&edit.kind) == Some(false) || editors::not_ready(&edit.kind).is_some();
        match edit.editing.and_then(|id| part.feature_mut(id).map(|f| (id, f))) {
            Some((id, f)) => {
                f.kind = edit.kind.clone();
                f.flags.suppressed = waits;
                (part, Some(id))
            }
            None => {
                let id = part.add(edit.kind.clone());
                if let (true, Some(f)) = (waits, part.feature_mut(id)) {
                    f.flags.suppressed = true;
                }
                (part, Some(id))
            }
        }
    }

    /// Build the part as shown, on a worker thread, if it changed since the last build.
    pub(crate) fn request_build(&mut self) {
        // The numbers that read parameters brought up to them first.
        self.follow_parameters_before_build();
        // ---- SURF ---- a region whose box a parameter moved is read again first.
        editors::surf_common::reread_stale_regions(self);
        let (part, preview) = self.shown_part();
        // A box closed, or another opened: its picks are no longer taken on the bodies as they
        // stood before the last one's feature. Until the build of the new one comes in, the
        // bodies shown are the part as it stands — before a new feature, all of it.
        if preview.is_none() || preview != self.preview_feature {
            self.pick_bodies = None;
        }
        self.preview_feature = preview;
        // ---- SURF ---- the scan the build reads, and the part's key with it.
        let scan = editors::surf_common::build_scan(self);
        let key = format!("{}#{}", part.to_json(), self.scan_key);
        if self.built_for.as_deref() == Some(key.as_str()) {
            return;
        }
        self.built_for = Some(key);
        self.build_generation += 1;
        let generation = self.build_generation;
        // To the build thread, which takes up from the builds it kept (with a feature box open,
        // the part as it stood before the feature too: what its picks are taken on); only the
        // newest build asked counts.
        let net = network_layer::build_context(self, &part, preview, generation); // ---- NET ----
        // ---- SURF ---- the scan goes with it: what a split, a trim, a fitted surface and a
        // shadow read.
        self.builds.ask(build_service::BuildRequest { generation, part, preview, flip: self.flip, net, scan });
    }

    /// Whether a build is under way: the newest asked for has not landed.
    pub(crate) fn building(&self) -> bool {
        self.builds.building()
    }

    fn poll_build(&mut self, ctx: &egui::Context) {
        let answers = match self.builds.poll() {
            Ok(a) => a,
            Err(why) => {
                self.status = why;
                ctx.request_repaint();
                return;
            }
        };
        for answer in answers {
            match answer {
                build_service::Answer::Done(done) if done.generation == self.build_generation => {
                    let done = *done;
                    self.builds.pending = None;
                    self.builds.last_steps = done.steps;
                    if let Some(said) = build_service::newly_failed(&self.part, &self.built, &done.built) {
                        self.status = said;
                    }
                    self.built = done.built;
                    self.bodies = done.bodies;
                    // ---- DRAFT ----
                    // A build asked for a box's preview can land after the box was created (its
                    // part the same, so no other build was asked): with no box open, the bodies
                    // before a feature are no pick set.
                    self.pick_bodies = if self.preview_feature.is_some() { done.before } else { None };
                    self.forget_unshown_meshes();
                    self.built_generation = done.generation;
                    self.deviation.key = None;
                    if self.view.want_fit_bodies {
                        self.view.want_fit = true;
                        self.view.want_fit_bodies = false;
                    }
                    ctx.request_repaint();
                }
                build_service::Answer::Failed(g, why) if g == self.build_generation => {
                    self.builds.pending = None;
                    self.status = why;
                    ctx.request_repaint();
                }
                build_service::Answer::Skipped(g) => log::debug!("reverse build {g} overtaken by a newer one, not played"),
                // An overtaken build's answer is for a list that no longer stands.
                _ => ctx.request_repaint(),
            }
        }
        if self.building() {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
    }

    /// Change the part (one Undo step) and build it again.
    fn edit_part(&mut self, f: impl FnOnce(&mut Part)) {
        self.checkpoint();
        f(&mut self.part);
        self.request_build();
    }

    // ---- the scan's BVHs (picking) ----

    /// The BVH of a scan object, built on a worker thread the first time it is asked for.
    pub(crate) fn scan_bvh(&mut self, o: &ScanObject) -> Option<Arc<Bvh>> {
        let version = o.mesh.shape_version() ^ o.mesh.topology_version().rotate_left(32);
        if let Some((v, b)) = self.scan_bvh.get(&o.id) {
            if *v == version && b.is_current(&o.mesh) {
                return Some(b.clone());
            }
        }
        if let Some((v, rx)) = self.bvh_jobs.get(&o.id) {
            if *v == version {
                if let Ok(b) = rx.try_recv() {
                    let b = Arc::new(b);
                    self.scan_bvh.insert(o.id, (version, b.clone()));
                    self.bvh_jobs.remove(&o.id);
                    return Some(b);
                }
                return None;
            }
        }
        let (tx, rx) = channel();
        let mesh = o.mesh.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Bvh::build(&mesh));
        });
        self.bvh_jobs.insert(o.id, (version, rx));
        None
    }

    // ---- sketches ----

    /// The plane the Sketch on list stands for now: a coordinate plane, through the origin or
    /// through the scan's centre (its offset the centre's coordinate, as ScanRuler offers it).
    fn chosen_plane(&self) -> PlaneRef {
        let (axis, through_centre) = self.sketch_plane;
        match (through_centre, self.scan_centre()) {
            (true, Some(c)) => PlaneRef::world(axis, c.dot(axis.dir())),
            _ => PlaneRef::world(axis, 0.0),
        }
    }

    /// The frame a plane stands for now, in this part.
    pub(crate) fn plane_frame(&self, plane: &PlaneRef) -> Option<SectionFrame> {
        resolve_plane_frame(plane, &self.part)
    }

    /// The frame a sketch lies in as last built (a sketch on a face follows the face), or as
    /// it was kept.
    pub(crate) fn sketch_frame(&self, id: FeatureId) -> Option<SectionFrame> {
        self.built.sketches.get(&id).map(|b| b.frame).or_else(|| self.part.sketch(id).map(|s| s.frame))
    }

    /// Open a new sketch on a plane.
    pub(crate) fn new_sketch(&mut self, plane: PlaneRef) {
        self.close_mode();
        let Some(frame) = self.plane_frame(&plane) else {
            self.status = format!("The plane is gone ({}).", describe_plane_ref(&plane, &self.part));
            return;
        };
        let chains = self.slice(&mm_reverse::measure_elements::slice_frame(&plane, &frame)); // ---- PLANE ----
        let n = chains.len();
        let edit = SketchEdit::new(None, plane, frame, Sketch::default(), chains, self.fit_options);
        self.view.face_plane(&edit.frame, self.flip);
        self.mode = Mode::Sketch(Box::new(edit));
        self.status = if self.scan.is_empty() {
            "Sketch on a blank stage: draw with the tools; Finish keeps it.".into()
        } else if n == 0 {
            "The plane cuts nothing of the scan: move it with Offset, or draw by hand.".into()
        } else {
            format!("The plane cuts {n} chain{} through the scan: Fit all chains reads them as lines and arcs.", plural(n))
        };
    }

    /// Open a sketch feature for editing.
    pub(crate) fn edit_sketch(&mut self, id: FeatureId) {
        let Some(s) = self.part.sketch(id).cloned() else { return };
        self.close_mode();
        let frame = self.sketch_frame(id).unwrap_or(s.frame);
        let chains = self.slice(&frame);
        let mut edit = SketchEdit::new(Some(id), s.plane, frame, s.sketch, chains, self.fit_options);
        edit.image = s.image; // ---- SKETCH ---- the picture laid under it.
        self.view.face_plane(&edit.frame, self.flip);
        self.mode = Mode::Sketch(Box::new(edit));
    }

    /// Finish the open sketch: kept as a feature (an empty new sketch makes nothing). Returns
    /// the sketch's feature id.
    pub(crate) fn finish_sketch(&mut self) -> Option<FeatureId> {
        let Mode::Sketch(edit) = std::mem::replace(&mut self.mode, Mode::Idle) else { return None };
        self.fit_options = edit.fit_options;
        self.view.restore_after_sketch();
        let feature_id = edit.feature;
        let edit = *edit;
        // What the box does not edit — expressions, a picture, an inset — stays as it was.
        let kept = feature_id.and_then(|id| self.part.sketch(id).cloned());
        let feature = match kept {
            Some(k) => SketchFeature { plane: edit.plane, frame: edit.frame, sketch: edit.sketch, image: edit.image, ..k },
            None => SketchFeature { plane: edit.plane, frame: edit.frame, sketch: edit.sketch, exprs: Default::default(), image: edit.image, inset: None },
        };
        let id = match feature_id {
            Some(id) => {
                self.edit_part(|p| {
                    if let Some(f) = p.feature_mut(id) {
                        f.kind = FeatureKind::Sketch(feature);
                    }
                });
                id
            }
            None if feature.sketch.entities.is_empty() && feature.image.is_none() => {
                self.status = "The sketch was empty: nothing was kept.".into();
                return None;
            }
            None => {
                let mut id = 0;
                self.edit_part(|p| id = p.add(FeatureKind::Sketch(feature)));
                id
            }
        };
        self.status = format!("{} kept.", self.part.feature(id).map_or("Sketch", |f| f.name.as_str()));
        Some(id)
    }

    /// Put a feature box or a sketch away without keeping it.
    pub(crate) fn close_mode(&mut self) {
        if let Mode::Sketch(edit) = &self.mode {
            self.fit_options = edit.fit_options;
            self.view.restore_after_sketch();
        }
        self.mode = Mode::Idle;
        self.request_build();
    }

    // ---- features ----

    /// The last sketch in the part (what a new Extrude or Revolve starts on).
    fn last_sketch(&self) -> Option<FeatureId> {
        self.part.features.iter().rev().find(|f| matches!(f.kind, FeatureKind::Sketch(_)) && !f.suppressed()).map(|f| f.id)
    }

    pub(crate) fn open_feature(&mut self, kind: FeatureKind, editing: Option<FeatureId>) {
        editors::box_opens();
        if let Mode::Sketch(_) = self.mode {
            self.finish_sketch();
        }
        let pick = editors::first_pick(&kind);
        // ---- RND ---- a box opened starts with nothing remembered from the last one.
        editors::rnd_forget();
        self.mode = Mode::Feature(Box::new(FeatureEdit { editing, kind, pick }));
        self.request_build();
    }

    pub(crate) fn new_extrude(&mut self, sketch: Option<FeatureId>, regions: Vec<String>) {
        let Some(sketch) = sketch.or_else(|| self.last_sketch()) else {
            self.status = "Extrude takes a sketch: make one first (New sketch).".into();
            return;
        };
        // Its one region where it has exactly one; the operator clicks those wanted otherwise (EXT).
        let regions = if regions.is_empty() { editors::auto_regions_of(self, sketch) } else { regions };
        self.open_feature(FeatureKind::Extrude(ExtrudeFeature::new(sketch, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::Join)), None);
        self.status = "Click the part to put the extrude's end at that height, or type the distances.".into();
    }

    pub(crate) fn new_revolve(&mut self, sketch: Option<FeatureId>, regions: Vec<String>) {
        let Some(sketch) = sketch.or_else(|| self.last_sketch()) else {
            self.status = "Revolve takes a sketch: make one first (New sketch).".into();
            return;
        };
        // No axis until one is chosen or clicked (EXT).
        let axis = mm_reverse::extrude_draft::no_axis();
        let regions = if regions.is_empty() { editors::auto_regions_of(self, sketch) } else { regions };
        self.open_feature(FeatureKind::Revolve(RevolveFeature::new(sketch, regions, axis, 360.0, BodyOp::Join)), None);
    }

    /// The regions a feature takes from a sketch when none is picked: every region that is no
    /// hole of another, read off the sketch as it stands.
    #[allow(dead_code)] // the boxes that pick a sketch's regions; Extrude and Revolve have their own (EXT)
    pub(crate) fn default_regions(&self, sketch: FeatureId) -> Vec<String> {
        match self.built.sketches.get(&sketch) {
            Some(b) => default_regions(&b.regions),
            None => self.part.sketch(sketch).map_or(Vec::new(), |s| default_regions(&find_regions(&s.sketch))),
        }
    }

    /// Create the open feature (or keep the edit of an existing one).
    pub(crate) fn commit_feature(&mut self) {
        // ---- DRAFT ----
        if editors::commit_instead(self) {
            self.request_build();
            return;
        }
        // ---- LOFT ---- a Loft, Sweep, Pattern or Move not ready makes nothing, as its Create waits.
        if let Mode::Feature(edit) = &self.mode {
            if editors::loft_ready(&edit.kind) == Some(false) {
                return;
            }
        }
        // An Extrude or Revolve box not ready makes nothing: it says what it lacks (EXT).
        if let Mode::Feature(e) = &self.mode {
            if let Some(why) = editors::not_ready(&e.kind) {
                self.status = why;
                return;
            }
        }
        // ---- BOOL ---- a feature not ready waits, as its Create key does.
        if let Mode::Feature(edit) = &self.mode {
            if let Err(why) = editors::ready_to_create(&edit.kind) {
                self.status = why;
                return;
            }
        }
        let Mode::Feature(edit) = std::mem::replace(&mut self.mode, Mode::Idle) else { return };
        let mut made = None;
        match edit.editing {
            Some(id) => self.edit_part(|p| {
                if let Some(f) = p.feature_mut(id) {
                    // The queries its picks were given by stay while what they found is picked.
                    let next = Feature { kind: edit.kind.clone(), ..f.clone() };
                    *f = with_kept_queries(f, next);
                    made = Some(id);
                }
            }),
            None => self.edit_part(|p| made = Some(editors::add_new(p, edit.kind.clone()))),
        }
        if let Some(id) = made {
            // ---- SURF ----
            editors::surf_common::after_commit(self, id);
            // Once a feature is made from it, its sketch is hidden.
            for sk in self.part.feature(id).map(|f| f.kind.source_sketches()).unwrap_or_default() {
                if let Some(f) = self.part.feature_mut(sk) {
                    f.flags.visible = false;
                }
            }
            let name = self.part.feature(id).map(|f| f.name.clone()).unwrap_or_default();
            self.status = match self.built.errors.get(&id) {
                Some(e) => format!("{name}: {e}"),
                None => format!("{name} made."),
            };
        }
        self.request_build();
    }

    /// Open a feature's box (or its sketch) for editing.
    pub(crate) fn edit_feature(&mut self, id: FeatureId) {
        let Some(f) = self.part.feature(id) else { return };
        match &f.kind {
            FeatureKind::Sketch(_) => self.edit_sketch(id),
            kind => {
                let mut kind = kind.clone();
                build_service::without_target(&mut kind);
                self.open_feature(kind, Some(id));
            }
        }
    }

    // ---- export ----

    /// The bodies shown, for export.
    fn export_bodies(&self) -> Vec<(String, mm_reverse::build::BuiltBody)> {
        self.built.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)).map(|b| (b.name.clone(), self.worn(b))).collect()
    }

    pub(crate) fn export_step_to(&mut self, path: &std::path::Path) -> Result<usize, String> {
        let bodies = self.export_bodies();
        if bodies.is_empty() {
            return Err("There is no body to export.".into());
        }
        // ---- SURF ---- a sheet is never exported on its own.
        if bodies.iter().all(|(_, b)| b.surface) {
            return Err("No visible solid to export — a sheet is never exported on its own.".into());
        }
        let list: Vec<(&str, &mm_reverse::build::KernelBody)> = bodies.iter().map(|(n, b)| (n.as_str(), &b.body)).collect();
        let text = export_step(&list)?;
        std::fs::write(path, text).map_err(|e| e.to_string())?;
        Ok(bodies.len())
    }

    fn export_step_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("STEP", &["step", "stp"]).set_file_name("part.step").save_file() else { return };
        self.status = match self.export_step_to(&path) {
            Ok(n) => format!("Exported {n} bod{} to {}.", if n == 1 { "y" } else { "ies" }, path.display()),
            Err(e) => format!("STEP export failed: {e}"),
        };
    }

    /// Send the bodies shown to the Mesh workspace as meshes (in its scene's axes).
    pub(crate) fn send_to_mesh(&mut self) {
        let meshes: Vec<(String, Mesh)> = self
            .bodies
            .iter()
            .filter(|b| !self.hidden_bodies.contains(&b.id))
            .map(|b| {
                let pm = &b.part_mesh;
                let scene: Vec<DVec3> = pm.positions.iter().map(|p| self.to_scene(*p)).collect();
                // Welded, so the body comes into Mesh as one closed surface.
                let (positions, triangles) = mm_reverse::build::welded(&scene, &pm.triangles, 1e-7);
                let (m, _) = Mesh::from_triangles(&positions, &triangles, None);
                (b.name.clone(), m)
            })
            .collect();
        if meshes.is_empty() {
            self.status = "There is no body to send.".into();
            return;
        }
        self.status = format!("Sent {} bod{} to the Mesh workspace.", meshes.len(), if meshes.len() == 1 { "y" } else { "ies" });
        self.to_mesh = meshes;
    }

    fn save_part_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Reverse Engineer part", &["rpart"]).set_file_name("part.rpart").save_file() else { return };
        self.status = match std::fs::write(&path, self.part.to_json()) {
            Ok(()) => format!("Part saved to {}.", path.display()),
            Err(e) => format!("Could not save the part: {e}"),
        };
    }

    fn open_part_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Reverse Engineer part", &["rpart", "json"]).pick_file() else { return };
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| Part::from_json(&t)) {
            Ok(p) => {
                self.close_mode();
                self.checkpoint();
                self.part = p;
                self.request_build();
                self.view.want_fit_bodies = true;
                self.status = format!("Opened {}.", path.display());
            }
            Err(e) => self.status = format!("Could not open the part: {e}"),
        }
    }

    fn autosave(&mut self) {
        if !self.autosave_due || polymix_workspace::scripted() {
            return;
        }
        self.autosave_due = false;
        if let Some(path) = part_file() {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, self.part.to_json()).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }

    // ---- the deviation map ----

    /// Read the maps again when the scan, the bodies or the range changed (viewbar.rs): a
    /// `key` let go reads the live map again.
    fn update_deviation(&mut self, ctx: &egui::Context) {
        let key = (self.scan_key, self.built_generation, self.deviation.range.to_bits());
        if self.deviation.key != Some(key) {
            self.deviation.key = Some(key);
            self.vb.map_changed();
        }
        self.update_maps(ctx);
    }

    // ---- the panel ----

    fn panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Reverse Engineer");
            if self.building() {
                polymix_workspace::spinner(ui);
            }
        });
        let tris: usize = self.scan.iter().filter(|o| o.visible).map(|o| o.mesh.triangle_count()).sum();
        if self.scan.is_empty() {
            ui.weak("No scan: modelling on a blank stage. A scan opened in the Mesh workspace joins the model here.");
        } else {
            let names: Vec<&str> = self.scan.iter().map(|o| o.name.as_str()).collect();
            ui.weak(format!("Scan: {} object{} from Mesh, {} triangles", self.scan.len(), plural(self.scan.len()), tris)).on_hover_text(names.join("\n"));
        }
        self.kernel_status(ui);
        ui.separator();
        self.parameters_group(ui);
        ui.separator();
        match std::mem::replace(&mut self.mode, Mode::Idle) {
            Mode::Sketch(mut edit) => {
                let label = describe_plane_ref(&edit.plane, &self.part);
                let title = sketch_box::title(&self.part, &edit);
                let action = sketch_box::sketch_box(ui, &mut edit, self.scan.is_empty(), &label, &title);
                self.mode = Mode::Sketch(edit);
                self.sketch_action(action);
            }
            Mode::Feature(mut edit) => {
                let action = self.feature_box(ui, &mut edit);
                self.mode = Mode::Feature(edit);
                match action {
                    FeatureAction::None => {}
                    FeatureAction::Changed => self.request_build(),
                    FeatureAction::Create => self.commit_feature(),
                    FeatureAction::Cancel => {
                        self.close_mode();
                        self.status = "Put away.".into();
                    }
                }
            }
            Mode::Idle => {
                self.idle_box(ui);
            }
        }
        // ---- PLANE ---- the elements and the sections.
        self.elements_group(ui);
        ui.separator();
        self.feature_list(ui);
        ui.separator();
        self.body_list(ui);
        ui.separator();
        self.view_box(ui);
    }

    fn sketch_action(&mut self, action: view::SketchAction) {
        let revolve = matches!(action, view::SketchAction::Revolve(_));
        match action {
            view::SketchAction::None => {}
            view::SketchAction::Finish => {
                self.finish_sketch();
            }
            view::SketchAction::Extrude(region) | view::SketchAction::Revolve(region) => {
                if let Some(id) = self.finish_sketch() {
                    // The sketch's own keys take every region of it (EXT).
                    let regions = match region {
                        Some(r) => vec![r],
                        None => editors::every_region_of(self, id),
                    };
                    if revolve {
                        self.new_revolve(Some(id), regions);
                    } else {
                        self.new_extrude(Some(id), regions);
                    }
                }
            }
            view::SketchAction::Level(v) => {
                // ---- SKETCH ---- the scan cut at another level, the chains of that level.
                let frame = match &mut self.mode {
                    Mode::Sketch(edit) => edit.set_level(v).then(|| edit.slice_frame()),
                    _ => None,
                };
                if let Some(frame) = frame {
                    let chains = self.slice(&frame);
                    if let Mode::Sketch(edit) = &mut self.mode {
                        edit.chains = chains;
                    }
                }
            }
            view::SketchAction::Offset(d) => {
                if let Mode::Sketch(edit) = &self.mode {
                    let (plane, frame) = with_plane_offset(&edit.plane, &edit.frame, d);
                    let chains = self.slice(&if edit.level != 0.0 { frame.offset(edit.level) } else { frame });
                    if let Mode::Sketch(edit) = &mut self.mode {
                        edit.set_plane(plane, frame, chains);
                        self.view.face_plane(&edit.frame, self.flip);
                    }
                }
            }
        }
    }

    fn idle_box(&mut self, ui: &mut egui::Ui) {
        // ---- PLANE ---- the planes a sketch is offered, the one under the pointer shown on the part.
        self.sketch_on_field(ui);
        ui.add_space(4.0);
        ui.label(RichText::new("Features").strong());
        let has_sketch = self.last_sketch().is_some();
        egui::Grid::new("re_features").num_columns(3).spacing([4.0, 4.0]).show(ui, |ui| {
            if ui.add_enabled(has_sketch, egui::Button::new("Extrude")).on_hover_text("Extrude a sketch's regions — pick them in the viewport").clicked() {
                self.new_extrude(None, Vec::new());
            }
            if ui.add_enabled(has_sketch, egui::Button::new("Revolve")).on_hover_text("Revolve a sketch's regions about an axis").clicked() {
                self.new_revolve(None, Vec::new());
            }
            let mut column = 2;
            for kind in editors::NEW_FROM_PANEL.into_iter().filter(|k| editors::ready(k)) {
                let made = editors::new_feature(kind, self);
                let (label, hint) = editors::button(kind);
                let why = made.as_ref().err().cloned().unwrap_or_default();
                if ui.add_enabled(made.is_ok(), egui::Button::new(label)).on_hover_text(hint).on_disabled_hover_text(format!("{hint} — {why}")).clicked() {
                    if let Ok(k) = made {
                        self.open_feature(k, None);
                    }
                }
                column += 1;
                if column % 3 == 0 {
                    ui.end_row();
                }
            }
            ui.end_row();
        });
        // The kinds whose editors are still being ported: the build knows them, the panel waits.
        egui::CollapsingHeader::new("Not built yet").id_salt("re_features_waiting").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for kind in editors::NEW_FROM_PANEL.into_iter().filter(|k| !editors::ready(k)) {
                    let (label, hint) = editors::button(kind);
                    ui.add_enabled(false, egui::Button::new(label)).on_disabled_hover_text(format!("{hint} — its editor is not built yet."));
                }
            });
        });
        self.sketch_start_hint(ui); // ---- PLANE ----
    }

    fn feature_list(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Features").strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add_enabled(!self.redo.is_empty(), egui::Button::new("↷").small()).on_hover_text("Redo (⌘⇧Z)").clicked() {
                    self.redo();
                }
                if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("↶").small()).on_hover_text("Undo (⌘Z)").clicked() {
                    self.undo();
                }
            });
        });
        if self.part.features.is_empty() {
            ui.weak("None yet.");
            return;
        }
        let mut edit = None;
        let mut delete = None;
        let mut toggle_visible = None;
        let mut toggle_suppress = None;
        let editing = match &self.mode {
            Mode::Sketch(e) => e.feature,
            Mode::Feature(e) => e.editing,
            Mode::Idle => None,
        };
        egui::ScrollArea::vertical().id_salt("re_feature_list").max_height(220.0).show(ui, |ui| {
            for f in &self.part.features {
                ui.horizontal(|ui| {
                    let error = self.built.errors.get(&f.id);
                    if matches!(f.kind, FeatureKind::Sketch(_)) {
                        let eye = if f.visible() { "👁" } else { "–" };
                        if ui.small_button(eye).on_hover_text("Show or hide the sketch on the part").clicked() {
                            toggle_visible = Some(f.id);
                        }
                    } else {
                        ui.add_space(22.0);
                    }
                    let mut text = RichText::new(&f.name);
                    if f.suppressed() {
                        text = text.strikethrough().weak();
                    }
                    if let Some(_) = error {
                        text = text.color(Color32::from_rgb(255, 110, 100));
                    }
                    if editing == Some(f.id) {
                        text = text.strong();
                    }
                    let r = ui.selectable_label(editing == Some(f.id), text);
                    let warnings = self.built.warnings.get(&f.id).map(|w| w.join("\n")).unwrap_or_default();
                    let touched: Vec<String> = self.built.touched.get(&f.id).map(|t| t.iter().map(|id| self.built.body(*id).map_or(id.to_string(), |b| b.name.clone())).collect()).unwrap_or_default();
                    let r = match (error, touched.is_empty()) {
                        (Some(e), _) => r.on_hover_text(format!("{e}\nClick to edit.")),
                        (None, false) => r.on_hover_text(format!("{}{}{}\nClick to edit.", touched.join(", "), if warnings.is_empty() { "" } else { "\n" }, warnings)),
                        (None, true) if !warnings.is_empty() => r.on_hover_text(format!("{warnings}\nClick to edit.")),
                        _ => r.on_hover_text("Click to edit"),
                    };
                    if r.clicked() {
                        edit = Some(f.id);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text("Delete (and what is made from it)").clicked() {
                            delete = Some(f.id);
                        }
                        let s = if f.suppressed() { "off" } else { "on" };
                        if ui.small_button(s).on_hover_text(if f.suppressed() { "Suppressed: click to build it again" } else { "Built: click to suppress (leave it out of the build)" }).clicked() {
                            toggle_suppress = Some(f.id);
                        }
                    });
                });
                build_service::row_notes(ui, &self.built, f.id);
            }
        });
        if let Some(id) = edit {
            self.edit_feature(id);
        }
        if let Some(id) = delete {
            self.close_mode();
            self.edit_part(|p| {
                p.remove(id);
            });
        }
        if let Some(id) = toggle_visible {
            self.edit_part(|p| {
                if let Some(f) = p.feature_mut(id) {
                    f.flags.visible = !f.flags.visible;
                }
            });
        }
        if let Some(id) = toggle_suppress {
            self.edit_part(|p| {
                if let Some(f) = p.feature_mut(id) {
                    f.flags.suppressed = !f.flags.suppressed;
                }
            });
        }
    }

    fn body_list(&mut self, ui: &mut egui::Ui) {
        // ---- PICKLIST ---- the Bodies group (`bodies.rs`).
        self.bodies_group(ui);
        if self.bodies.is_empty() {
            return;
        }
        ui.horizontal(|ui| {
            if ui.button("Export STEP…").on_hover_text("The bodies shown as a STEP file (AP242) CAD opens as solid parts").clicked() {
                self.export_step_dialog();
            }
            if ui.button("Send to Mesh").on_hover_text("The bodies shown as new objects in the Mesh workspace").clicked() {
                self.send_to_mesh();
            }
        });
        let can_compare = !self.scan.is_empty() && self.built.bodies.iter().any(|b| !b.surface && !self.hidden_bodies.contains(&b.id)) && !self.building();
        if ui.add_enabled(can_compare, egui::Button::new("Compare in Deviation")).on_hover_text("Measure the scan against the visible bodies in the Deviation workspace — its scale, histogram, pins and report. The bodies go in as the reference part; no best fit is run, they stand in the scan's frame already").clicked() {
            self.compare_in_deviation();
        }
    }

    fn view_box(&mut self, ui: &mut egui::Ui) {
        // The ways of looking at the part are on the view bar, bottom left of the stage; the
        // Overlaps switch (PICKLIST) is in the menu over its Bodies key.
        self.units_choice(ui);
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Save part…").clicked() {
                self.save_part_dialog();
            }
            if ui.button("Open part…").clicked() {
                self.open_part_dialog();
            }
            if ui.button("New part").on_hover_text("Start again with no features (Undo brings the part back)").clicked() {
                self.close_mode();
                self.edit_part(|p| *p = Part::default());
                self.hidden_bodies.clear();
            }
        });
    }

    // ---- script steps (tests) ----

    fn run_script_step(&mut self, step: &str) {
        let mut parts = step.split(':');
        let cmd = parts.next().unwrap_or("");
        let args: Vec<&str> = parts.collect();
        let num = |i: usize, default: f64| args.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(default);
        let axis = |s: Option<&&str>| match s.copied() {
            Some("x") | Some("yz") => WorldAxis::X,
            Some("y") | Some("xz") => WorldAxis::Y,
            _ => WorldAxis::Z,
        };
        match cmd {
            // ---- PLANE ---- the elements, the sections and the sketch planes (`elements.rs`).
            _ if self.plane_script_step(cmd, &args) => {}
            // sketch:<axis>[:<offset>[:centre]]
            "sketch" => {
                self.sketch_plane = (axis(args.first()), args.get(2) == Some(&"centre"));
                let plane = self.chosen_plane();
                let plane = mm_reverse::model::plane_at_offset(&plane, plane_offset(&plane) + num(1, 0.0));
                self.new_sketch(plane);
            }
            "fit" => {
                if let Mode::Sketch(e) = &mut self.mode {
                    e.fit_all();
                    self.status = e.message.clone();
                }
            }
            "close" => {
                if let Mode::Sketch(e) = &mut self.mode {
                    e.close();
                }
            }
            "rect" => {
                if let Mode::Sketch(e) = &mut self.mode {
                    e.checkpoint();
                    e.sketch.add_rectangle(DVec2::new(num(0, 0.0), num(1, 0.0)), DVec2::new(num(2, 10.0), num(3, 10.0)));
                    e.solve();
                }
            }
            "circle" => {
                if let Mode::Sketch(e) = &mut self.mode {
                    e.checkpoint();
                    e.sketch.add_shape(mm_reverse::sketch::Shape::Circle { c: DVec2::new(num(0, 0.0), num(1, 0.0)), r: num(2, 5.0) });
                    e.solve();
                }
            }
            "finish" => {
                self.finish_sketch();
            }
            // extrude:<to>[:<from>[:join|cut|new]]
            "extrude" => {
                // A script's extrude takes every region that is no hole of another, as before the
                // box took one region alone (EXT).
                let regions = self.last_sketch().map(|s| self.default_regions(s)).unwrap_or_default();
                self.new_extrude(None, regions);
                if let Mode::Feature(edit) = &mut self.mode {
                    if let FeatureKind::Extrude(e) = &mut edit.kind {
                        e.extent = ExtrudeExtent::Distance { from: num(1, 0.0), to: num(0, 10.0) };
                        e.op = match args.get(2).copied() {
                            Some("cut") => BodyOp::Cut,
                            Some("new") => BodyOp::New,
                            Some("join") | None => BodyOp::Join,
                            Some(other) => {
                                log::warn!("extrude step: {other} is no extrude's result (join, cut or new); joined");
                                BodyOp::Join
                            }
                        };
                    }
                }
                self.commit_feature();
            }
            // revolve:<axis>[:<angle>]
            "revolve" => {
                let regions = self.last_sketch().map(|s| self.default_regions(s)).unwrap_or_default();
                self.new_revolve(None, regions);
                if let Mode::Feature(edit) = &mut self.mode {
                    if let FeatureKind::Revolve(r) = &mut edit.kind {
                        r.axis = AxisRef::World { axis: axis(args.first()) };
                        r.angle_deg = num(1, 360.0);
                    }
                }
                self.commit_feature();
            }
            "deviation" => {
                self.deviation.on = true;
                self.deviation.range = num(0, self.deviation.range);
                self.deviation.key = None;
            }
            "export" => {
                let path = PathBuf::from(args.join(":"));
                self.status = match self.export_step_to(&path) {
                    Ok(n) => format!("exported {n}"),
                    Err(e) => e,
                };
                log::info!("reverse export: {}", self.status);
            }
            "send" => self.send_to_mesh(),
            // ---- LOFT ---- the Loft, Sweep, Pattern and Move body boxes (editors/loft_kit.rs).
            "loft" | "sweep" | "sweep-route" | "pattern" | "move" | "loft-open" | "sweep-open" | "sweep-route-open" | "pattern-open" | "move-open" | "tube" | "loft-stations" | "move-grip" | "grip-field" | "move-click" => editors::loft_script_step(self, cmd, &args),
            // ---- RND ---- rnd:<step>: the Fillet, Chamfer, Shell and Delete face boxes.
            "rnd" => editors::rnd_script(self, &args),
            // net:<step>[:…] — the patch network's box (see network_layer::script). ---- NET ----
            "net" => network_layer::script(self, &args),
            "fitview" => self.view.want_fit = true,
            "undo" => self.undo(),
            "report" => {
                // The part and its build, to the log (tests read it).
                let errors: Vec<String> = self.built.errors.iter().map(|(id, e)| format!("{id}: {e}")).collect();
                let vols: Vec<String> = self.bodies.iter().map(|b| format!("{} ({}) {:.3}", b.name, b.id, b.volume)).collect();
                log::info!("reverse report: {} features, bodies [{}], errors [{}], status: {}", self.part.features.len(), vols.join(", "), errors.join("; "), self.status);
                let st = &self.builds.last_steps;
                let warnings: Vec<String> = self.built.warnings.iter().map(|(id, w)| format!("{id}: {}", w.join(" "))).collect();
                log::info!("reverse build: {} of {} steps played, {} restored; kernel: {}; warnings [{}]", st.recomputed, st.of, st.restored, self.builds.kernel_words(), warnings.join("; "));
            }
            // buildstop — the Stop key of the build chip.
            "buildstop" => {
                self.stop_build();
                log::info!("reverse build stop: {}", self.status);
            }
            // slowbuild:<ms> — every step of the builds after it held this long (tests of a stop);
            // 0 for none.
            "slowbuild" => {
                self.builds.set_slow(std::time::Duration::from_millis(num(0, 0.0).max(0.0) as u64));
                self.built_for = None;
                self.request_build();
            }
            // kernel — where the CAD kernel stands, to the log.
            "kernel" => log::info!("reverse kernel: {}", self.builds.kernel_words()),
            // ---- DRAFT ----
            "draft" | "draftfit" | "moveface" | "snapface" | "snapat" | "draftchip" | "movechip" => editors::draft_script(self, cmd, &args),
            // ---- BOOL ----
            "bool" => editors::bool_script_step(self, &args),
            // ---- HOLE ----
            // hole:x:y:z[:d=6][:depth=8|:through][:insert=M3][:open] — or hole:element:<id>[:…]
            "hole" => {
                self.status = editors::hole_script_step(self, &args);
                log::info!("reverse hole: {}", self.status);
            }
            // thread:x:y:z[:std=metric|unc|unf|print45][:size=M10x1][:d=…][:p=…][:lh][:len=…][:start=…][:flip][:chamfer][:offset=…][:scan][:open]
            "thread" => {
                self.status = editors::thread_script_step(self, &args);
                log::info!("reverse thread: {}", self.status);
            }
            // threadread:x:y:z[:any] — the thread read off the scan along the round face there.
            "threadread" => {
                let answer = editors::thread_read_step(self, &args);
                log::info!("reverse threadread: {answer}");
            }
            // ---- PANEL ---- the parameters' steps (params.rs).
            c if c.starts_with("param") => self.param_script_step(c, &args),
            // ---- EXT ---- the Extrude and Revolve boxes' steps (editors/ext_script.rs).
            _ if editors::script_step(self, cmd, &args) => {}
            // ---- PICKLIST ----
            _ if self.picks_script_step(cmd, &args) => {}
            // The view bar's, the maps' and the checks' steps (checks.rs).
            c if self.viewcmp_step(c, &args) => {}
            // ---- SURF ----
            other if editors::surf_script(self, other, &args) => {}
            // ---- SKETCH ----
            _ if self.sketch_script_step(cmd, &args) => {}
            _ => log::warn!("unknown Reverse Engineer script step {step}"),
        }
    }
}

// ---- BUILD ----

impl ReverseWorkspace {
    /// The Model group's word on the CAD kernel (ScanRuler's kernel status hint): waiting,
    /// loading, ready with its version, or why not — with a key to start it again.
    fn kernel_status(&mut self, ui: &mut egui::Ui) {
        let words = self.builds.kernel_words();
        let mut retry = false;
        ui.horizontal_wrapped(|ui| match &self.builds.kernel {
            build_service::KernelStatus::Loading => {
                polymix_workspace::spinner(ui);
                ui.weak(words);
            }
            build_service::KernelStatus::Unavailable(_) => {
                ui.colored_label(Color32::from_rgb(226, 164, 59), words);
                retry = ui.small_button("Retry CAD kernel").clicked();
            }
            _ => {
                ui.weak(words);
            }
        });
        if retry {
            self.builds.restart();
            self.built_for = None;
            self.request_build();
        }
    }

    /// The edges and outlines the diagnoses of the failed features found, as segments in the
    /// scene: what their rows say is drawn red on the part.
    pub(crate) fn failure_lines(&self) -> Vec<(DVec3, DVec3)> {
        self.built
            .failed_at
            .values()
            .flatten()
            .flat_map(|line| line.windows(2).map(|w| (self.to_scene(w[0]), self.to_scene(w[1]))).collect::<Vec<_>>())
            .collect()
    }

    /// Stop the build the kernel is at (ScanRuler's session `stop`, asked): a kernel call cannot
    /// be interrupted, so the build thread is let go and a fresh one started; the feature whose
    /// step was in play is suppressed (one undo step), its picks and numbers kept, and the part
    /// built again without it. A box open on it is put away: built again it would hang the same
    /// way.
    pub(crate) fn stop_build(&mut self) {
        if !self.building() {
            self.status = "Nothing is building: the CAD kernel is at no build.".into();
            return;
        }
        let (in_play, at, played) = self.builds.in_play();
        let waited = self.builds.waited().unwrap_or_default();
        let (shown, _) = self.shown_part();
        let stopped = build_service::stop_words(&shown, in_play, at == self.build_generation, played, waited, |id| self.part.feature(id).is_some());
        self.builds.restart();
        self.built_for = None;
        let box_on_it = match (&self.mode, &stopped.feature) {
            (Mode::Feature(e), Some((id, _))) => e.editing == Some(*id),
            _ => false,
        };
        if stopped.put_away || box_on_it {
            self.mode = Mode::Idle;
        }
        match stopped.feature {
            Some((id, _)) => self.edit_part(|p| {
                if let Some(f) = p.feature_mut(id) {
                    f.flags.suppressed = true;
                }
            }),
            None => self.request_build(),
        }
        self.status = stopped.message;
    }
}

enum FeatureAction {
    None,
    Changed,
    Create,
    Cancel,
}

impl ReverseWorkspace {
    /// The open feature's box: its kind's own editor (`editors/<kind>.rs`), the build's word on
    /// it, and OK or Cancel.
    fn feature_box(&mut self, ui: &mut egui::Ui, edit: &mut FeatureEdit) -> FeatureAction {
        let title = match edit.editing.and_then(|id| self.part.feature(id)) {
            Some(f) => format!("Edit {}", f.name),
            None => edit.kind.kind_name().to_string(),
        };
        let title = editors::box_words(self, edit).map_or(title, |w| w.title);
        ui.label(RichText::new(title).strong().size(15.0));
        let changed = editors::ui(self, ui, edit);
        let words = editors::box_words(self, edit);
        self.analysis_row(ui);
        let id = edit.editing.or(self.preview_feature);
        if let Some(e) = id.and_then(|id| self.built.errors.get(&id)).filter(|_| !words.as_ref().is_some_and(|w| w.own_preview)) {
            ui.colored_label(Color32::from_rgb(255, 120, 100), e);
        }
        if let Some(w) = id.and_then(|id| self.built.warnings.get(&id)) {
            for line in w {
                ui.colored_label(Color32::from_rgb(240, 200, 110), line);
            }
        }
        ui.add_space(4.0);
        let mut action = if changed { FeatureAction::Changed } else { FeatureAction::None };
        let ready = editors::ready_to_create(&edit.kind);
        ui.horizontal(|ui| {
            let label = if edit.editing.is_some() { "OK" } else { "Create" };
            // ---- DRAFT ---- / ---- LOFT ---- a kind that words its Create its own way (Move the
            // face, Move the body), and a Loft, Sweep, Pattern or Move waits until it is ready.
            // ---- NET ---- a patch network waits for a curve and a filled cell.
            // ---- BOOL ---- a cut, mirror, combine or offset waits for what it needs.
            let label = editors::create_label(self, edit).unwrap_or_else(|| label.to_string());
            let can = editors::loft_ready(&edit.kind).unwrap_or(true) && editors::can_create(&edit.kind) && ready.is_ok();
            // ---- EXT ---- the Extrude and Revolve boxes word their Create, its tooltip and its
            // readiness their own way.
            let (label, tip, can) = words.as_ref().map_or((label, "Enter or a middle click", can), |w| (w.create.clone(), w.tip, w.ready));
            let mut key = ui.add_enabled(can, egui::Button::new(RichText::new(label).strong())).on_hover_text(tip);
            // ---- BOOL ---- the held-down key says why it waits.
            if let Err(why) = &ready {
                key = key.on_disabled_hover_text(why.as_str());
            }
            if key.clicked() {
                action = FeatureAction::Create;
            }
            if ui.button("Cancel").on_hover_text("Escape").clicked() {
                action = FeatureAction::Cancel;
            }
        });
        if words.is_some_and(|w| w.create_now) {
            action = FeatureAction::Create;
        }
        action
    }
}

// The scan's map is the view bar's now (viewbar.rs); these two colour the patch network's
// curvature map (network_layer.rs). ---- NET ----
fn rgb(c: [f32; 3]) -> u32 {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    (b(c[0]) << 16) | (b(c[1]) << 8) | b(c[2])
}

/// The deviation ramp at `t` ranges: blue at −1 through green at 0 to red at +1, a darker cap
/// past either end.
pub(crate) fn ramp(t: f64) -> u32 {
    if !t.is_finite() {
        return 0x808080;
    }
    if t > 1.0 {
        return 0x8c1010;
    }
    if t < -1.0 {
        return 0x10247a;
    }
    let stops: [(f64, [f32; 3]); 5] = [(-1.0, [0.1, 0.3, 1.0]), (-0.5, [0.1, 0.8, 0.95]), (0.0, [0.15, 0.85, 0.25]), (0.5, [0.95, 0.85, 0.1]), (1.0, [0.95, 0.15, 0.1])];
    for w in stops.windows(2) {
        let ((t0, c0), (t1, c1)) = (w[0], w[1]);
        if t <= t1 {
            let s = ((t - t0) / (t1 - t0)) as f32;
            return rgb([c0[0] + (c1[0] - c0[0]) * s, c0[1] + (c1[1] - c0[1]) * s, c0[2] + (c1[2] - c0[2]) * s]);
        }
    }
    rgb(stops[4].1)
}

impl Workspace for ReverseWorkspace {
    fn name(&self) -> &'static str {
        "Reverse Engineer"
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        egui::Panel::bottom("re_status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if self.building() {
                    polymix_workspace::spinner(ui);
                    ui.weak("Building…");
                }
                let text = match &self.mode {
                    Mode::Sketch(e) if !e.message.is_empty() => e.message.clone(),
                    _ => self.status.clone(),
                };
                ui.label(text);
            });
        });
        let vr = ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::NULL.with("re_vp_rect"))).unwrap_or(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)));
        let margin = 8.0;
        let width = self.panel_width;
        let r = egui::Area::new(egui::Id::NULL.with("re_panel")).fixed_pos(vr.min + egui::vec2(margin, margin)).order(egui::Order::Middle).show(&ctx, |ui| {
            egui::Frame::window(ui.style()).show(ui, |ui| {
                ui.set_width(width);
                let max_h = (vr.height() - 2.0 * margin - 20.0).max(80.0);
                ui.set_max_height(max_h);
                egui::ScrollArea::vertical().max_height(max_h).show(ui, |ui| self.panel(ui));
            });
        });
        let rect = r.response.rect;
        let edge = egui::Rect::from_min_max(egui::pos2(rect.right() - 4.0, rect.top()), egui::pos2(rect.right() + 4.0, rect.bottom()));
        let handle = egui::Area::new(egui::Id::NULL.with("re_panel_edge"))
            .fixed_pos(edge.min)
            .order(egui::Order::Middle)
            .show(&ctx, |ui| ui.allocate_response(edge.size(), egui::Sense::drag()).on_hover_cursor(egui::CursorIcon::ResizeHorizontal));
        if handle.inner.dragged() {
            self.panel_width = (self.panel_width + handle.inner.drag_delta().x).clamp(260.0, 480.0);
        }
        self.view.panel_rect = Some(rect);
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            ctx.data_mut(|d| d.insert_temp(egui::Id::NULL.with("re_vp_rect"), rect));
            self.viewport(ui, frame);
        });
        // The build chip at the top of the stage, with its Stop key.
        let stage = ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::NULL.with("re_vp_rect"))).unwrap_or(vr);
        if build_service::build_chip(&ctx, stage, &self.builds, &self.part, !self.part.features.is_empty()) {
            self.stop_build();
        }
        self.keys(&ctx);
        // ---- PICKLIST ---- the chips by the click.
        self.context_chips(&ctx);
        if let Some(step) = self.script.pop_front() {
            self.run_script_step(&step);
            ctx.request_repaint();
        }
    }

    fn tick(&mut self, ctx: &egui::Context, active: bool) {
        self.poll_build(ctx);
        if active {
            self.update_deviation(ctx);
            // ---- PICKLIST ----
            self.tick_picks(ctx);
        }
        self.autosave();
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if polymix_workspace::scripted() {
            return;
        }
        if let Mode::Sketch(e) = &self.mode {
            self.fit_options = e.fit_options;
        }
        storage.set_string("reverse.fit_detail", self.fit_options.detail.to_string());
        storage.set_string("reverse.fit_join", if self.fit_options.join_corners { "1" } else { "0" }.into());
        storage.set_string("reverse.fit_tolerance", self.fit_options.tolerance.map_or(String::new(), |t| t.to_string()));
        storage.set_string("reverse.deviation_range", self.deviation.range.to_string());
        self.vb.save(storage);
        sketcher::save_prefs(storage); // ---- SKETCH ----
        self.params.save(storage);
        self.autosave_due = true;
        self.autosave();
    }

    fn on_exit(&mut self) {
        self.autosave_due = true;
        self.autosave();
    }

    fn emergency_save(&mut self) {
        self.autosave_due = true;
        self.autosave();
    }

    fn title(&self) -> String {
        match &self.mode {
            Mode::Sketch(_) => "Polymix — Reverse Engineer — Sketch".into(),
            _ => "Polymix — Reverse Engineer".into(),
        }
    }

    fn script_push(&mut self, step: String) {
        self.script.push_back(step);
    }

    fn script_idle(&self) -> bool {
        // A build made slow on purpose (slowbuild) is not waited for: the test looks at it.
        self.script.is_empty() && (!self.building() || self.builds.slowed()) && self.vb.idle() && !editors::rnd_busy() && !self.picks_busy()
    }

    fn agent_acting(&mut self) {
        self.by_agent = true;
    }
}

/// The part's features, for code outside (tests).
#[allow(dead_code)]
pub(crate) fn feature_names(part: &Part) -> Vec<String> {
    part.features.iter().map(|f: &Feature| f.name.clone()).collect()
}

/// Whether a plane takes an offset the sketch box can change.
pub(crate) fn offsettable(plane: &PlaneRef) -> bool {
    plane_takes_offset(plane)
}

// ---- PLANE ----
mod elements;
