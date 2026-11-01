// SPDX-License-Identifier: AGPL-3.0-only
//! The chips that follow a click on the part when no box is open: what the thing under the
//! click can become. A sketch's region offers Extrude and Revolve, an element with an axis a
//! revolve about it, a plane and a flat body face a sketch on it, an edge a fillet. Object first,
//! verb second — the way Fusion's marking menu and Onshape's context toolbar work — so a beginner
//! never has to find the panel's feature keys to get from a closed profile to a body. A pick on
//! the bodies heads the chips with a card of readings: the part it is on (its name, the
//! components it is in, keys to hide it or every other part), what was picked, and — for two
//! picks — what lies between them.
//!
//! Ported from ScanRuler (`plugins/reverse/ui/ContextChips.tsx`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use egui::{Pos2, RichText};
use glam::DVec3;
use mm_reverse::body_folders::{common_folder, folder_of, folder_path_text, FolderMove};
use mm_reverse::body_move::pivot_of;
use mm_reverse::display_units::DisplayUnits;
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{
    body_face_plane, frame_on_face, movable_face, partial_round, AxisRef, BodyId, BodyOp, ChamferFeature, DeleteFaceFeature, DraftFeature, EdgeRef, ElementGeometry, ElementId, Exprs, ExtractFeature, ExtrudeExtent, ExtrudeFeature, FaceProfile, FaceRef, FeatureId, FeatureKind, FilletFeature, HoleAxis, HoleFeature, MirrorFeature, MirrorPlane, MoveBodyFeature, MoveFaceFeature, Part, PlaneRef, ShellFeature,
    ThreadFeature, ThreadStandard,
};
use mm_reverse::selection_info::{selection_info, PickedItem};
use mm_reverse::sketch::EntityId;
use mm_reverse::thread::{thread_face_reading, FaceMesh, RoundFacePick};
use mm_reverse::topo::{FaceInfo, FaceKind};

use super::picks::{Selected, Selection, SKETCH_STARTED};
use super::{Mode, ReverseWorkspace};

/// A wall's default thickness, mm (ScanRuler's `DEFAULT_WALL_MM`).
pub(crate) const DEFAULT_WALL_MM: f64 = 2.0;
/// The fillet radius when the scan shows no round at the edge (`DEFAULT_FILLET_MM`).
pub(crate) const DEFAULT_FILLET_MM: f64 = 1.0;
/// A draft's default lean, degrees (`DEFAULT_DRAFT_DEG`).
pub(crate) const DEFAULT_DRAFT_DEG: f64 = 2.0;

/// What a chip does once taken.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ChipAction {
    ExtrudeRegion(FeatureId, String),
    RevolveRegion(FeatureId, String),
    RevolveAboutElement(ElementId),
    HoleOnElement(ElementId, f64),
    SketchOn(PlaneRef),
    ExtractElement(ElementId),
    HidePlane(WorldAxis),
    RevolveAboutLine(FeatureId, EntityId),
    Edit(FeatureId),
    ExtrudeFace(BodyId, FaceRef),
    SketchHere(BodyId, FaceRef),
    MoveFaces(BodyId, Vec<FaceRef>),
    SnapToScan(BodyId, FaceRef),
    Shell(BodyId, Vec<FaceRef>),
    DeleteFaces(BodyId, Vec<FaceRef>),
    HoleOnFace(BodyId, FaceRef),
    Thread(BodyId, FaceInfo),
    Mirror(BodyId, FaceRef),
    Draft(BodyId, FaceRef),
    MoveBodies(Vec<BodyId>),
    Fillet(BodyId, Vec<EdgeRef>),
    Chamfer(BodyId, Vec<EdgeRef>),
    Clearance(BodyId, BodyId),
}

/// A chip: its key (what a script step names it by), its word, its tooltip and what it does.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Chip {
    pub key: &'static str,
    pub label: String,
    pub title: String,
    pub action: ChipAction,
}

fn chip(key: &'static str, label: impl Into<String>, title: impl Into<String>, action: ChipAction) -> Chip {
    Chip { key, label: label.into(), title: title.into(), action }
}

/// The chip that moves the bodies picked — or copies them — by the gizmo or by numbers.
fn move_chip(bodies: Vec<BodyId>) -> Chip {
    let one = bodies.len() == 1;
    chip(
        "move-body",
        if one { "Move body" } else { "Move bodies" },
        format!("Move or turn {} — drag the arrows and rings on the part, or type the distances and angles; tick Copy to move a copy instead", if one { "this body" } else { "these bodies" }),
        ChipAction::MoveBodies(bodies),
    )
}

/// Whether a face of a build is a round one known to stop short of a full turn.
fn partial(face: &FaceInfo) -> bool {
    partial_round(face.full_turn)
}

/// Whether Snap to scan takes a face: a plane or a cylinder is fitted to the scan there.
fn snappable(face: &FaceInfo) -> bool {
    matches!(face.fingerprint.kind, FaceKind::Plane | FaceKind::Cylinder)
}

/// What the chips say for each kind of thing picked.
pub(crate) fn chips_for(sel: &Selection, part: &Part, has_body: bool) -> Vec<Chip> {
    match sel {
        Selection::Region { sketch, key } => vec![
            chip("extrude", "Extrude", "Extrude this profile — the ghost and its arrows appear on the part", ChipAction::ExtrudeRegion(*sketch, key.clone())),
            chip("revolve", "Revolve", "Revolve this profile about an axis picked next", ChipAction::RevolveRegion(*sketch, key.clone())),
        ],
        Selection::Element { id, name } => {
            let mut chips = Vec::new();
            let Some(e) = part.element(*id) else { return chips };
            let g = &e.geometry;
            if matches!(g, ElementGeometry::Line { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Circle { .. }) {
                chips.push(chip("revolve-about", "Revolve about this", format!("Revolve the last sketch's profile about {name}"), ChipAction::RevolveAboutElement(*id)));
            }
            if has_body {
                if let ElementGeometry::Cylinder { radius, .. } | ElementGeometry::Circle { radius, .. } = g {
                    chips.push(chip("hole", "Hole", format!("Drill a hole through the body along {name}"), ChipAction::HoleOnElement(*id, *radius)));
                }
            }
            // A plane is sketched on as it is from the panel's Sketch on list — a section's too.
            match g {
                ElementGeometry::Plane { .. } => chips.push(chip("sketch-on", "Sketch", format!("Start a sketch on {name}"), ChipAction::SketchOn(PlaneRef::Plane { id: *id, offset: None, up: None }))),
                ElementGeometry::Section { .. } => chips.push(chip("sketch-on", "Sketch", format!("Start a sketch on {name}"), ChipAction::SketchOn(PlaneRef::Section { id: *id, offset: None, up: None }))),
                _ => {}
            }
            if matches!(g, ElementGeometry::Plane { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Sphere { .. } | ElementGeometry::Torus { .. }) {
                chips.push(chip("extract", "Extract as body", format!("{name} as a solid at its extents, extended as in 3D Measure"), ChipAction::ExtractElement(*id)));
            }
            chips
        }
        Selection::CoordinatePlane { axis } => {
            let name = axis.plane_name();
            vec![
                chip("sketch-on", "Sketch", format!("Start a sketch on the {name}"), ChipAction::SketchOn(PlaneRef::world(*axis, 0.0))),
                chip("hide-plane", "Hide", format!("Put the {name} away — its eye in the body list brings it back"), ChipAction::HidePlane(*axis)),
            ]
        }
        Selection::SketchLine { sketch, entity } => vec![chip("revolve-about", "Revolve about this", "Revolve the sketch about this line", ChipAction::RevolveAboutLine(*sketch, *entity))],
        Selection::Body(items) => match items.as_slice() {
            [PickedItem::Face { body, face, .. }] => face_chips(*body, face, part),
            [PickedItem::Edge { body, edge }] => vec![
                chip("fillet", "Fillet", "Round this edge — the radius is read off the scan", ChipAction::Fillet(*body, vec![edge.fingerprint.clone()])),
                chip("chamfer", "Chamfer", "Chamfer this edge", ChipAction::Chamfer(*body, vec![edge.fingerprint.clone()])),
                move_chip(vec![*body]),
            ],
            // A corner is read, not built on.
            [PickedItem::Vertex { .. }] | [] => Vec::new(),
            many => items_chips(many),
        },
    }
}

/// A face of a body: its maker opened, a flat one pulled or sketched on, a face moved, laid on
/// the scan, opened by a shell or taken off, a round one drilled or threaded, a flat one mirrored
/// about or drafted from, and its body moved.
fn face_chips(body: BodyId, face: &FaceInfo, part: &Part) -> Vec<Chip> {
    let fp = face.fingerprint;
    let flat = frame_on_face(&fp).is_some();
    let mut chips = Vec::new();
    // The feature that made the face, opened as its row in the list opens it.
    if fp.origin >= 0 {
        if let Some(maker) = part.feature(fp.origin as FeatureId) {
            chips.push(chip("edit", format!("Edit {}", maker.name), format!("Open {}, the feature that made this face, to change it", maker.name), ChipAction::Edit(maker.id)));
        }
    }
    // A round face short of a full turn — a fillet's — is neither resized nor threaded.
    let part_turn = partial(face);
    if flat {
        chips.push(chip("extrude-face", "Extrude", "Pull this face: drag the arrow out to add to the body, in to cut into it", ChipAction::ExtrudeFace(body, fp)));
        chips.push(chip("sketch-here", "Sketch here", "Start a sketch on this face of the body, its outline projected into the sketch", ChipAction::SketchHere(body, fp)));
    }
    if movable_face(&fp) && !part_turn {
        let title = if fp.kind == FaceKind::Cylinder {
            "Offset this round face: type how much its radius grows or shrinks — Ctrl+click its ends too to make the whole cylinder longer and wider"
        } else {
            "Offset this face along its normal by a typed distance — Ctrl+click more faces to move them together"
        };
        chips.push(chip("move-face", "Move face", title, ChipAction::MoveFaces(body, vec![fp])));
    }
    if snappable(face) && !part_turn {
        chips.push(chip("snap", "Snap to scan", "Move this face onto the scan: click the scan where the face should lie — a plane or a cylinder is fitted there and the face moved to it", ChipAction::SnapToScan(body, fp)));
    }
    chips.push(chip("shell", "Shell", "Hollow the body, this face taken away to open it", ChipAction::Shell(body, vec![fp])));
    chips.push(chip("delete-face", "Delete face", "Take this face off the body and heal the gap — with the rest of the round, the hole or the boss it is part of", ChipAction::DeleteFaces(body, vec![fp])));
    if fp.kind == FaceKind::Cylinder && fp.center.is_some() {
        chips.push(chip("hole", "Hole", "A hole along this face’s axis", ChipAction::HoleOnFace(body, fp)));
        if !part_turn {
            chips.push(chip("thread", "Thread", "A modeled screw thread on this round face — a bolt’s on a boss, a nut’s in a bore, the standard size for it to start", ChipAction::Thread(body, face.clone())));
        }
    }
    if flat {
        chips.push(chip("mirror", "Mirror", "Mirror the body about this face", ChipAction::Mirror(body, fp)));
        chips.push(chip("draft", "Draft", "Lean faces of the body about this face as the neutral plane — click the faces to lean next", ChipAction::Draft(body, fp)));
    }
    chips.push(move_chip(vec![body]));
    chips
}

/// Several picks: edges of one body rounded or chamfered together, faces of one body moved,
/// shelled or taken off together, picks on two bodies their clearance, on several the bodies
/// moved together; a mixed pick is there to be read.
fn items_chips(items: &[PickedItem]) -> Vec<Chip> {
    let first = items[0].body();
    let one_body = items.iter().all(|i| i.body() == first);
    let edges: Vec<EdgeRef> = items.iter().filter_map(|i| if let PickedItem::Edge { edge, .. } = i { Some(edge.fingerprint.clone()) } else { None }).collect();
    let faces: Vec<&FaceInfo> = items.iter().filter_map(|i| if let PickedItem::Face { face, .. } = i { Some(face) } else { None }).collect();
    if one_body && edges.len() == items.len() {
        let n = edges.len();
        return vec![
            chip("fillet", "Fillet", format!("Round these {n} edges"), ChipAction::Fillet(first, edges.clone())),
            chip("chamfer", "Chamfer", format!("Chamfer these {n} edges"), ChipAction::Chamfer(first, edges)),
        ];
    }
    let mut bodies: Vec<BodyId> = Vec::new();
    for i in items {
        if !bodies.contains(&i.body()) {
            bodies.push(i.body());
        }
    }
    if bodies.len() > 2 {
        return vec![move_chip(bodies)];
    }
    if bodies.len() == 2 {
        return vec![move_chip(bodies.clone()), chip("clearance", "Clearance", "How far apart the two bodies stand at the closest — or how much of them runs into each other", ChipAction::Clearance(bodies[0], bodies[1]))];
    }
    if one_body && faces.len() == items.len() {
        let n = faces.len();
        let prints: Vec<FaceRef> = faces.iter().map(|f| f.fingerprint).collect();
        let mut chips = vec![
            chip("shell", "Shell", format!("Hollow the body, open at these {n} faces"), ChipAction::Shell(first, prints.clone())),
            chip("delete-face", "Delete faces", format!("Take these {n} faces off the body and heal the gap"), ChipAction::DeleteFaces(first, prints.clone())),
        ];
        if faces.iter().all(|f| movable_face(&f.fingerprint) && !partial(f)) {
            chips.insert(0, chip("move-face", "Move faces", format!("Offset these {n} faces together by one distance — the flat ones along their normals, the round ones in radius"), ChipAction::MoveFaces(first, prints)));
        }
        return chips;
    }
    Vec::new()
}

impl ReverseWorkspace {
    /// The chips for what is selected now.
    pub(crate) fn current_chips(&self) -> Vec<Chip> {
        match &self.picks.selection {
            Some(s) => chips_for(&s.what, &self.part, !self.bodies.is_empty()),
            None => Vec::new(),
        }
    }

    /// The chip bar — or, with readings, the card — by the click, wholly on the screen.
    pub(crate) fn context_chips(&mut self, ctx: &egui::Context) {
        if !matches!(self.mode, Mode::Idle) {
            return;
        }
        let Some(Selected { what, at }) = self.picks.selection.clone() else { return };
        let chips = chips_for(&what, &self.part, !self.bodies.is_empty());
        let items: Vec<PickedItem> = self.picks.items().to_vec();
        let info = selection_info(&items, DisplayUnits::Mm);
        if chips.is_empty() && info.is_none() {
            return;
        }
        let id = egui::Id::NULL.with("re_context_chips");
        let size = ctx.memory(|m| m.area_rect(id)).map_or(egui::vec2(220.0, 40.0), |r| r.size());
        let screen = ctx.content_rect();
        let (left, top) = (at.x + 12.0, at.y + 12.0);
        let dx = (screen.right() - 8.0 - (left + size.x)).min(0.0);
        // Past the bottom it stands above the click instead.
        let dy = if top + size.y > screen.bottom() - 8.0 { -(size.y + 24.0) } else { 0.0 };
        let pos = Pos2::new((left + dx).max(8.0), (top + dy).max(8.0));
        let mut taken: Option<ChipAction> = None;
        let mut body_key: Option<BodyKey> = None;
        egui::Area::new(id).order(egui::Order::Foreground).fixed_pos(pos).fade_in(false).show(ctx, |ui| {
            egui::Frame::popup(ui.style()).inner_margin(4.0).corner_radius(3.0).show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                if let Some(info) = &info {
                    ui.set_max_width(300.0);
                    if !items.is_empty() {
                        body_key = self.body_line(ui, &items);
                    }
                    ui.label(RichText::new(&info.title).size(11.5).strong().color(ui.visuals().weak_text_color()));
                    egui::Grid::new("re_chip_info").num_columns(2).spacing([10.0, 1.0]).show(ui, |ui| {
                        for r in &info.rows {
                            ui.label(RichText::new(&r.label).size(11.5).color(ui.visuals().weak_text_color()));
                            ui.add(egui::Label::new(RichText::new(&r.value).size(11.5).monospace()).selectable(true));
                            ui.end_row();
                        }
                    });
                    if !matches!(&what, Selection::Body(i) if i.len() > 1) {
                        ui.label(RichText::new("Ctrl+click adds · Shift+drag a window").size(10.5).color(ui.visuals().weak_text_color().gamma_multiply(0.8)));
                    }
                    if !chips.is_empty() {
                        ui.separator();
                    }
                }
                let mut show = |ui: &mut egui::Ui| {
                    for c in &chips {
                        if ui.add(egui::Button::new(RichText::new(&c.label).size(12.0)).wrap_mode(egui::TextWrapMode::Extend)).on_hover_text(&c.title).clicked() {
                            taken = Some(c.action.clone());
                        }
                    }
                };
                if info.is_some() {
                    ui.horizontal_wrapped(|ui| show(ui));
                } else {
                    ui.horizontal(|ui| show(ui));
                }
            });
        });
        if let Some(k) = body_key {
            self.run_body_key(k, &items);
        }
        if let Some(a) = taken {
            // Every chip first lets the selection go, then runs.
            self.picks.selection = None;
            self.run_chip(a);
        }
    }

    /// The part — or parts — a pick on the bodies is on: named, with the components it is in,
    /// and its eye: hide it, or every other body — a key that shows them all again once they
    /// are — and, for picks on several, a component made of them.
    fn body_line(&self, ui: &mut egui::Ui, items: &[PickedItem]) -> Option<BodyKey> {
        let mut ids: Vec<BodyId> = Vec::new();
        for i in items {
            if !ids.contains(&i.body()) {
                ids.push(i.body());
            }
        }
        let these: Vec<&super::BodyView> = self.bodies.iter().filter(|b| ids.contains(&b.id)).collect();
        if these.is_empty() {
            return None;
        }
        let one = (these.len() == 1).then(|| these[0]);
        let folders = &self.part.body_folders;
        let path = one.map(|b| folder_path_text(folders, folder_of(folders, b.id).map(|f| f.id))).unwrap_or_default();
        let names = these.iter().map(|b| b.name.as_str()).collect::<Vec<_>>().join(", ");
        let alone = self.others_hidden(&ids);
        let mut key = None;
        ui.horizontal(|ui| {
            if let Some(b) = one {
                let (r, _) = ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
                ui.painter().circle_filled(r.center(), 4.5, self.body_color32(b.id));
            }
            ui.vertical(|ui| {
                let name = one.map_or_else(|| format!("{} bodies", these.len()), |b| b.name.clone());
                let hover = match one {
                    Some(b) if !path.is_empty() => format!("{}, in {path}", b.name),
                    Some(b) => b.name.clone(),
                    None => names.clone(),
                };
                ui.add(egui::Label::new(RichText::new(name).strong()).selectable(false)).on_hover_text(hover);
                if !path.is_empty() {
                    ui.label(RichText::new(format!("in {path}")).small().color(ui.visuals().weak_text_color()));
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let small = |t: &str| egui::Button::new(RichText::new(t).size(11.0)).small();
                if these.len() > 1 && ui.add(small("Make component")).on_hover_text(format!("Put {names} into a new component in the body list")).clicked() {
                    key = Some(BodyKey::MakeComponent(ids.clone()));
                }
                if self.bodies.len() > these.len() {
                    let (word, tip) = if alone { ("Show all", "Show every body again".to_string()) } else { ("Hide others", format!("Hide every body but {}", one.map_or("these", |b| b.name.as_str()))) };
                    if ui.add(small(word)).on_hover_text(tip).clicked() {
                        key = Some(if alone { BodyKey::ShowAll } else { BodyKey::HideOthers(ids.clone()) });
                    }
                }
                let tip = match one {
                    Some(b) => format!("Hide {} — its eye in the body list brings it back", b.name),
                    None => format!("Hide these bodies — {names}"),
                };
                if ui.add(small("Hide")).on_hover_text(tip).clicked() {
                    key = Some(BodyKey::Hide(ids.clone()));
                }
            });
        });
        ui.separator();
        key
    }

    fn run_body_key(&mut self, k: BodyKey, _items: &[PickedItem]) {
        match k {
            BodyKey::Hide(ids) => {
                self.picks.selection = None;
                self.set_bodies_visible(&ids, false);
            }
            BodyKey::HideOthers(ids) => self.isolate_bodies(&ids),
            BodyKey::ShowAll => self.set_all_bodies_visible(true),
            BodyKey::MakeComponent(ids) => {
                let what = FolderMove { bodies: ids, folders: Vec::new() };
                let parent = common_folder(&self.part.body_folders, &what);
                let id = self.add_body_folder(parent, Some(&what));
                let name = self.part.body_folders.iter().find(|f| f.id == id).map_or("A component".to_string(), |f| f.name.clone());
                self.status = format!("{name} holds them now — double-click its name in the body list to name it.");
            }
        }
    }

    // ---- what a chip does ----

    /// The feature a kind's own box would start, where its editor gives one of that kind.
    fn fresh(&self, kind: &str) -> Option<FeatureKind> {
        super::editors::new_feature(kind, self).ok()
    }

    /// A box opened on a feature a chip filled in; the rows picked in the list are let go.
    fn open_from_chip(&mut self, kind: FeatureKind) {
        self.picks.list_pick = Default::default();
        self.open_feature(kind, None);
    }

    /// The middle of the bodies' boxes — what a Move turns them about.
    fn bodies_pivot(&self, ids: &[BodyId]) -> DVec3 {
        let boxes: Vec<(DVec3, DVec3)> = ids
            .iter()
            .filter_map(|id| self.bodies.iter().find(|b| b.id == *id))
            .filter_map(|b| {
                let pts = &b.part_mesh.positions;
                let lo = pts.iter().copied().reduce(|a, c| a.min(c))?;
                let hi = pts.iter().copied().reduce(|a, c| a.max(c))?;
                Some((lo, hi))
            })
            .collect();
        pivot_of(&boxes)
    }

    /// Open a Move on bodies — the box's own pick of them, its pivot their middle.
    pub(crate) fn start_move_bodies(&mut self, ids: Vec<BodyId>) {
        let ids: Vec<BodyId> = ids.into_iter().filter(|id| self.bodies.iter().any(|b| b.id == *id)).collect();
        let pivot = self.bodies_pivot(&ids);
        let kind = match self.fresh("moveBody") {
            Some(FeatureKind::MoveBody(m)) => FeatureKind::MoveBody(MoveBodyFeature { bodies: ids, pivot, ..m }),
            _ => FeatureKind::MoveBody(MoveBodyFeature { bodies: ids, pivot, translate: DVec3::ZERO, rotate: DVec3::ZERO, copy: false, exprs: Exprs::new() }),
        };
        self.open_from_chip(kind);
    }

    /// A Move's bodies changed from the list: the pivot follows them to their middle, and stays
    /// where it was with none.
    pub(crate) fn set_move_bodies(&mut self, ids: Vec<BodyId>) {
        let mut known: Vec<BodyId> = Vec::new();
        for id in ids {
            if !known.contains(&id) && self.bodies.iter().any(|b| b.id == id) {
                known.push(id);
            }
        }
        let pivot = (!known.is_empty()).then(|| self.bodies_pivot(&known));
        if let Mode::Feature(edit) = &mut self.mode {
            if let FeatureKind::MoveBody(m) = &mut edit.kind {
                m.bodies = known;
                if let Some(p) = pivot {
                    m.pivot = p;
                }
            }
        }
        self.request_build();
    }

    pub(crate) fn run_chip(&mut self, action: ChipAction) {
        match action {
            ChipAction::ExtrudeRegion(sketch, key) => self.new_extrude(Some(sketch), vec![key]),
            ChipAction::RevolveRegion(sketch, key) => self.new_revolve(Some(sketch), vec![key]),
            ChipAction::RevolveAboutElement(id) => {
                self.new_revolve(None, Vec::new());
                if let Mode::Feature(edit) = &mut self.mode {
                    if let FeatureKind::Revolve(r) = &mut edit.kind {
                        r.axis = AxisRef::Element { id };
                    }
                }
                self.request_build();
            }
            ChipAction::RevolveAboutLine(sketch, entity) => {
                self.new_revolve(Some(sketch), Vec::new());
                if let Mode::Feature(edit) = &mut self.mode {
                    if let FeatureKind::Revolve(r) = &mut edit.kind {
                        r.axis = AxisRef::SketchLine { entity };
                    }
                }
                self.request_build();
            }
            ChipAction::HoleOnElement(id, radius) => {
                let body = self.last_solid();
                let kind = match (self.fresh("hole"), body) {
                    (Some(FeatureKind::Hole(h)), _) => FeatureKind::Hole(HoleFeature { axis: HoleAxis::Element { id }, radius, ..h }),
                    (_, Some(body)) => FeatureKind::Hole(HoleFeature { body, axis: HoleAxis::Element { id }, radius, depth: None, insert: None, exprs: Exprs::new() }),
                    _ => {
                        self.status = "A hole is drilled in a body: make one first.".into();
                        return;
                    }
                };
                self.open_from_chip(kind);
            }
            ChipAction::SketchOn(plane) => {
                if self.plane_frame(&plane).is_none() {
                    self.status = "That plane is not available any more.".into();
                    return;
                }
                self.new_sketch(plane);
                self.status = SKETCH_STARTED.into();
            }
            ChipAction::ExtractElement(id) => {
                let kind = match self.fresh("extract") {
                    Some(FeatureKind::Extract(x)) => FeatureKind::Extract(ExtractFeature { element: Some(id), section: None, ..x }),
                    _ => FeatureKind::Extract(ExtractFeature { element: Some(id), section: None, thickness: DEFAULT_WALL_MM, op: BodyOp::New, target: None, exprs: Exprs::new() }),
                };
                self.open_from_chip(kind);
            }
            ChipAction::HidePlane(axis) => self.set_plane_shown(axis, false),
            ChipAction::Edit(id) => self.edit_feature(id),
            ChipAction::ExtrudeFace(body, face) => {
                if face.kind != FaceKind::Plane {
                    self.status = "Only a flat face can be pulled.".into();
                    return;
                }
                let mut x = ExtrudeFeature::new(0, Vec::new(), ExtrudeExtent::Distance { from: 0.0, to: 5.0 }, BodyOp::Join);
                x.face = Some(FaceProfile { body, face });
                self.open_from_chip(FeatureKind::Extrude(x));
                self.status = "Drag the arrow out to add to the body, in to cut into it — or type the distance.".into();
            }
            ChipAction::SketchHere(body, face) => {
                let Some(plane) = body_face_plane(body, &face) else {
                    self.status = "Only a flat face takes a sketch.".into();
                    return;
                };
                self.new_sketch(plane);
                // The face's outline is not projected into the sketch yet: the projection of a
                // body's edges into a sketch is still to be ported.
                self.status = "Sketch on the face — its outline is not projected in yet (no projection of a body's edges into a sketch in this version). Draw on it.".into();
            }
            ChipAction::MoveFaces(body, faces) => self.open_move_face(body, faces),
            ChipAction::SnapToScan(body, face) => {
                self.open_move_face(body, vec![face]);
                self.picks.snap_request = true;
                self.status = "Now click the scan where this face should lie.".into();
            }
            ChipAction::Shell(body, faces) => {
                let kind = match self.fresh("shell") {
                    Some(FeatureKind::Shell(s)) => FeatureKind::Shell(ShellFeature { body, faces, faces_where: Vec::new(), ..s }),
                    _ => FeatureKind::Shell(ShellFeature { body, faces, faces_where: Vec::new(), thickness: DEFAULT_WALL_MM, exprs: Exprs::new() }),
                };
                self.open_from_chip(kind);
            }
            ChipAction::DeleteFaces(body, faces) => {
                // ScanRuler opens a Delete face on the whole round, hole or boss the face is
                // part of (its face groups); until those are ported, on the face alone.
                self.open_from_chip(FeatureKind::DeleteFace(DeleteFaceFeature { body, faces, faces_where: Vec::new() }));
            }
            ChipAction::HoleOnFace(body, face) => {
                let radius = face.radius.unwrap_or(3.0);
                let kind = match self.fresh("hole") {
                    Some(FeatureKind::Hole(h)) => FeatureKind::Hole(HoleFeature { body, axis: HoleAxis::Face { face }, radius, ..h }),
                    _ => FeatureKind::Hole(HoleFeature { body, axis: HoleAxis::Face { face }, radius, depth: None, insert: None, exprs: Exprs::new() }),
                };
                self.open_from_chip(kind);
            }
            ChipAction::Thread(body, face) => self.open_thread(body, face),
            ChipAction::Mirror(body, face) => {
                let kind = match self.fresh("mirror") {
                    Some(FeatureKind::Mirror(m)) => FeatureKind::Mirror(MirrorFeature { body, plane: MirrorPlane::Face { face }, ..m }),
                    _ => FeatureKind::Mirror(MirrorFeature { body, plane: MirrorPlane::Face { face }, join: true, exprs: Exprs::new() }),
                };
                self.open_from_chip(kind);
            }
            ChipAction::Draft(body, face) => {
                let kind = match self.fresh("draft") {
                    Some(FeatureKind::Draft(d)) => FeatureKind::Draft(DraftFeature { body, neutral: MirrorPlane::Face { face }, faces: Vec::new(), faces_where: Vec::new(), ..d }),
                    _ => FeatureKind::Draft(DraftFeature { body, neutral: MirrorPlane::Face { face }, faces: Vec::new(), faces_where: Vec::new(), angle_deg: DEFAULT_DRAFT_DEG, groups: Vec::new(), exprs: Exprs::new() }),
                };
                self.open_from_chip(kind);
            }
            ChipAction::MoveBodies(ids) => self.start_move_bodies(ids),
            ChipAction::Fillet(body, edges) => {
                let kind = match self.fresh("fillet") {
                    Some(FeatureKind::Fillet(f)) => FeatureKind::Fillet(FilletFeature { body, edges, edges_where: Vec::new(), ..f }),
                    _ => FeatureKind::Fillet(FilletFeature { body, edges, edges_where: Vec::new(), radius: DEFAULT_FILLET_MM, chain: true, skip_failing: false, variable: Vec::new(), exprs: Exprs::new() }),
                };
                self.open_from_chip(kind);
            }
            ChipAction::Chamfer(body, edges) => {
                let kind = match self.fresh("chamfer") {
                    Some(FeatureKind::Chamfer(c)) => FeatureKind::Chamfer(ChamferFeature { body, edges, edges_where: Vec::new(), ..c }),
                    _ => FeatureKind::Chamfer(ChamferFeature { body, edges, edges_where: Vec::new(), distance: DEFAULT_FILLET_MM, chain: true, skip_failing: false, mode: None, distance2: None, angle_deg: None, flip: false, exprs: Exprs::new() }),
                };
                self.open_from_chip(kind);
            }
            ChipAction::Clearance(a, b) => self.measure_clearance(a, b),
        }
    }

    fn open_move_face(&mut self, body: BodyId, faces: Vec<FaceRef>) {
        let kind = match self.fresh("moveFace") {
            Some(FeatureKind::MoveFace(m)) => FeatureKind::MoveFace(MoveFaceFeature { body, faces, faces_where: Vec::new(), ..m }),
            _ => FeatureKind::MoveFace(MoveFaceFeature { body, faces, faces_where: Vec::new(), offset: 0.0, exprs: Exprs::new() }),
        };
        self.open_from_chip(kind);
    }

    /// A thread on a round face, sized as the standard size for the face to start: a bolt's on a
    /// boss, a nut's in a bore, read off the body's mesh.
    fn open_thread(&mut self, body: BodyId, face: FaceInfo) {
        let fp = face.fingerprint;
        let reading = self.bodies.iter().find(|b| b.id == body).and_then(|b| {
            let m = &b.part_mesh;
            thread_face_reading(FaceMesh { positions: &m.positions, triangles: &m.triangles, tri_face: &m.tri_face }, &RoundFacePick { face: Some(face.hash), center: fp.center, radius: fp.radius, dir: fp.dir }, mm_reverse::thread::ThreadStandard::Metric, false)
        });
        let (diameter, pitch) = match reading {
            Some(r) => (r.diameter.unwrap_or(10.0), r.pitch.unwrap_or(1.5)),
            None => (10.0, 1.5),
        };
        let kind = match self.fresh("thread") {
            Some(FeatureKind::Thread(t)) => FeatureKind::Thread(ThreadFeature { body, face: fp, diameter, pitch, length: None, ..t }),
            _ => FeatureKind::Thread(ThreadFeature { body, face: fp, standard: ThreadStandard::Metric, diameter, pitch, left_handed: false, length: None, start: 0.0, flip: false, offset: 0.0, chamfer: None, exprs: Exprs::new() }),
        };
        self.open_from_chip(kind);
    }

    /// The last solid body — what a feature most likely works on.
    fn last_solid(&self) -> Option<BodyId> {
        self.bodies.iter().rev().find(|b| !self.body_is_surface(b.id)).or(self.bodies.last()).map(|b| b.id)
    }
}

/// A key on the chips' body line.
#[derive(Debug, Clone)]
enum BodyKey {
    Hide(Vec<BodyId>),
    HideOthers(Vec<BodyId>),
    ShowAll,
    MakeComponent(Vec<BodyId>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_reverse::model::{Element, ExtrudeFeature};
    use mm_reverse::topo::{EdgeFingerprint, EdgeInfo, EdgeKind, FaceFingerprint};

    fn face(kind: FaceKind, origin: i64, full_turn: Option<bool>) -> FaceInfo {
        let radius = (kind != FaceKind::Plane).then_some(3.0);
        let center = (kind == FaceKind::Cylinder).then_some(DVec3::ZERO);
        FaceInfo { hash: 1, fingerprint: FaceFingerprint { kind, centroid: DVec3::new(0.0, 0.0, 10.0), dir: DVec3::Z, area: 100.0, radius, center, origin }, full_turn, point: DVec3::ZERO }
    }

    fn edge(hash: u32) -> EdgeInfo {
        EdgeInfo { hash, fingerprint: EdgeFingerprint { kind: EdgeKind::Line, mid: DVec3::new(hash as f64, 0.0, 0.0), dir: DVec3::X, length: 10.0, radius: None, faces: Vec::new(), origin: 1 }, faces: Vec::new(), ends: [DVec3::ZERO, DVec3::X], end_tangents: [DVec3::X, DVec3::X], polyline: vec![DVec3::ZERO, DVec3::X] }
    }

    fn keys(chips: &[Chip]) -> Vec<&'static str> {
        chips.iter().map(|c| c.key).collect()
    }

    fn part_with_extrude() -> Part {
        let mut p = Part::default();
        p.add(FeatureKind::Extrude(ExtrudeFeature::new(0, Vec::new(), ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::Join)));
        p
    }

    #[test]
    fn a_region_offers_extrude_and_revolve() {
        let c = chips_for(&Selection::Region { sketch: 1, key: "r0".into() }, &Part::default(), false);
        assert_eq!(keys(&c), vec!["extrude", "revolve"]);
        assert_eq!(c[0].title, "Extrude this profile — the ghost and its arrows appear on the part");
    }

    #[test]
    fn a_flat_face_offers_everything_a_flat_face_can_become() {
        let p = part_with_extrude();
        let c = chips_for(&Selection::Body(vec![PickedItem::Face { body: BodyId::new(1), face: face(FaceKind::Plane, 1, None), edges: Vec::new() }]), &p, true);
        assert_eq!(keys(&c), vec!["edit", "extrude-face", "sketch-here", "move-face", "snap", "shell", "delete-face", "mirror", "draft", "move-body"]);
        assert_eq!(c[0].label, "Edit Extrude 1");
        assert_eq!(c[3].title, "Offset this face along its normal by a typed distance — Ctrl+click more faces to move them together");
    }

    #[test]
    fn a_round_face_is_drilled_and_threaded_a_part_turn_neither_resized_nor_threaded() {
        let full = chips_for(&Selection::Body(vec![PickedItem::Face { body: BodyId::new(1), face: face(FaceKind::Cylinder, 9, Some(true)), edges: Vec::new() }]), &Part::default(), true);
        assert_eq!(keys(&full), vec!["move-face", "snap", "shell", "delete-face", "hole", "thread", "move-body"]);
        let part = chips_for(&Selection::Body(vec![PickedItem::Face { body: BodyId::new(1), face: face(FaceKind::Cylinder, 9, Some(false)), edges: Vec::new() }]), &Part::default(), true);
        assert_eq!(keys(&part), vec!["shell", "delete-face", "hole", "move-body"]);
    }

    #[test]
    fn edges_corners_and_several_picks() {
        let b = BodyId::new(1);
        let one = chips_for(&Selection::Body(vec![PickedItem::Edge { body: b, edge: edge(1) }]), &Part::default(), true);
        assert_eq!(keys(&one), vec!["fillet", "chamfer", "move-body"]);
        assert!(chips_for(&Selection::Body(vec![PickedItem::Vertex { body: b, point: DVec3::ZERO }]), &Part::default(), true).is_empty());
        let two = chips_for(&Selection::Body(vec![PickedItem::Edge { body: b, edge: edge(1) }, PickedItem::Edge { body: b, edge: edge(2) }]), &Part::default(), true);
        assert_eq!(keys(&two), vec!["fillet", "chamfer"]);
        assert_eq!(two[0].title, "Round these 2 edges");
        let bodies = chips_for(&Selection::Body(vec![PickedItem::Edge { body: b, edge: edge(1) }, PickedItem::Vertex { body: BodyId::new(2), point: DVec3::ZERO }]), &Part::default(), true);
        assert_eq!(keys(&bodies), vec!["move-body", "clearance"]);
        assert_eq!(bodies[0].label, "Move bodies");
        let three = chips_for(&Selection::Body(vec![PickedItem::Vertex { body: b, point: DVec3::ZERO }, PickedItem::Vertex { body: BodyId::new(2), point: DVec3::ZERO }, PickedItem::Vertex { body: BodyId::new(3), point: DVec3::ZERO }]), &Part::default(), true);
        assert_eq!(keys(&three), vec!["move-body"]);
        let mut f2 = face(FaceKind::Plane, 1, None);
        f2.hash = 2;
        let faces = chips_for(&Selection::Body(vec![PickedItem::Face { body: b, face: face(FaceKind::Plane, 1, None), edges: Vec::new() }, PickedItem::Face { body: b, face: f2, edges: Vec::new() }]), &Part::default(), true);
        assert_eq!(keys(&faces), vec!["move-face", "shell", "delete-face"]);
        assert_eq!(faces[2].label, "Delete faces");
        let mixed = chips_for(&Selection::Body(vec![PickedItem::Face { body: b, face: face(FaceKind::Plane, 1, None), edges: Vec::new() }, PickedItem::Edge { body: b, edge: edge(1) }]), &Part::default(), true);
        assert!(mixed.is_empty());
    }

    #[test]
    fn a_coordinate_plane_a_sketch_line_and_elements() {
        let c = chips_for(&Selection::CoordinatePlane { axis: WorldAxis::Z }, &Part::default(), false);
        assert_eq!(keys(&c), vec!["sketch-on", "hide-plane"]);
        assert_eq!(c[0].title, "Start a sketch on the XY plane");
        assert_eq!(c[1].title, "Put the XY plane away — its eye in the body list brings it back");
        assert_eq!(keys(&chips_for(&Selection::SketchLine { sketch: 1, entity: 2 }, &Part::default(), false)), vec!["revolve-about"]);
        let mut p = Part::default();
        p.elements.push(Element { id: 1, name: "Cylinder 1".into(), geometry: ElementGeometry::Cylinder { center: DVec3::ZERO, axis: DVec3::Z, radius: 4.0, length: 10.0, coverage: 360.0 }, stats: None });
        p.elements.push(Element { id: 2, name: "Plane 1".into(), geometry: ElementGeometry::Plane { center: DVec3::ZERO, normal: DVec3::Z, basis_u: DVec3::X, basis_v: DVec3::Y, extent_u: 5.0, extent_v: 5.0 }, stats: None });
        let cyl = chips_for(&Selection::Element { id: 1, name: "Cylinder 1".into() }, &p, true);
        assert_eq!(keys(&cyl), vec!["revolve-about", "hole", "extract"]);
        assert_eq!(cyl[1].title, "Drill a hole through the body along Cylinder 1");
        assert_eq!(keys(&chips_for(&Selection::Element { id: 1, name: "Cylinder 1".into() }, &p, false)), vec!["revolve-about", "extract"]);
        assert_eq!(keys(&chips_for(&Selection::Element { id: 2, name: "Plane 1".into() }, &p, true)), vec!["sketch-on", "extract"]);
    }
}
