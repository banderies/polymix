// SPDX-License-Identifier: AGPL-3.0-only
//! The Bodies group of the Reverse Engineer panel — an assembly browser: the coordinate planes
//! at its head, then a row a body — its colour, its name, what it holds and its eye — sorted into
//! components, folders that keep an assembly's parts apart, one inside another. A click on a row
//! picks it, Ctrl+click one more, Shift+click every row up to it; what is picked is lit in the
//! view, dragged together, moved together by Move body, and has keys of its own under the list. A
//! body or a component dragged onto a component goes into it; dropped on a row, it goes where
//! that row is; on the strip under the list while something is dragged, to the head of the list.
//! A right click on a row has the same moves in a menu. The pointer on a row lights the body in
//! the view, and the body under the cursor in the view rings its row. While a Move is open, a
//! click on a row puts the body into the Move or takes it out.
//!
//! Ported from ScanRuler (`plugins/reverse/ui/BodyList.tsx`, the plane rows of
//! `ui/ReversePanel.tsx`, the store's body steps in `state/reStore.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use egui::{Color32, RichText, Sense, Stroke};
use mm_reverse::body_folders::{bodies_of_pick, bodies_within, common_folder, folder_of, folder_path_text, folders_within, move_into, next_folder_id, next_folder_name, outermost, without_folder, FolderMove, ListPick};
use mm_reverse::body_style::{body_color_of, rgb_of, BODY_SWATCHES};
use mm_reverse::display_units::{fmt_vol, DisplayUnits, FmtOptions};
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{BodyFolder, BodyId, BodyStyle, FeatureKind};

use super::picks::{plane_color, HoverFrom, PLANE_AXES};
use super::{Mode, ReverseWorkspace};

/// How far in a row stands for each component it is in.
const INDENT_PX: f32 = 14.0;
const ROW_H: f32 = 20.0;

/// A row by what it is: a body's, or a component's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowKey {
    Body(BodyId),
    Folder(u32),
}

fn keys_of(pick: &ListPick) -> Vec<RowKey> {
    pick.folders.iter().map(|f| RowKey::Folder(*f)).chain(pick.bodies.iter().map(|b| RowKey::Body(*b))).collect()
}

fn pick_of(keys: &[RowKey]) -> ListPick {
    let mut p = ListPick::default();
    for k in keys {
        match k {
            RowKey::Body(b) => p.bodies.push(*b),
            RowKey::Folder(f) => p.folders.push(*f),
        }
    }
    p
}

/// What is dragged in the list.
#[derive(Debug, Clone)]
struct DragRows {
    what: FolderMove,
    words: String,
    many: bool,
}

/// What a pick is, in words: "Shaft", "Drive", "3 bodies", "2 bodies and a component".
pub(crate) fn pick_words(pick: &ListPick, names: &dyn Fn(BodyId) -> Option<String>, folders: &[BodyFolder]) -> String {
    if pick.len() == 1 {
        return match (pick.bodies.first(), pick.folders.first()) {
            (Some(b), _) => names(*b).unwrap_or_else(|| "Body".into()),
            (_, Some(f)) => folders.iter().find(|x| x.id == *f).map_or_else(|| "Component".into(), |x| x.name.clone()),
            _ => String::new(),
        };
    }
    let n = |count: usize, one: &str, many: &str| match count {
        0 => String::new(),
        1 => format!("a {one}"),
        _ => format!("{count} {many}"),
    };
    let text = [n(pick.bodies.len(), "body", "bodies"), n(pick.folders.len(), "component", "components")].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" and ");
    let mut c = text.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// A count with the thousands set apart, as the list writes a triangle count.
fn grouped(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// What a row does once the list is drawn.
enum RowAct {
    Pick(RowKey, egui::Modifiers),
    Menu(RowKey),
    Drop(FolderMove, Option<u32>),
    Eye(Vec<BodyId>, bool),
    Fold(u32, bool),
    NameFolder(u32),
    NameBody(BodyId),
    KeepFolderName(u32, String),
    KeepBodyName(BodyId, String),
    Paint(BodyId),
    Color(BodyId, Option<String>),
    PlaneEye(WorldAxis, bool),
    MenuChosen(MenuAct),
}

/// An item of the rows' right-click menu.
enum MenuAct {
    ShowAll,
    HideOthers(Vec<BodyId>),
    MoveBodies(Vec<BodyId>),
    NewInside(u32),
    Rename(u32),
    NewWith(FolderMove),
    MoveTo(FolderMove, Option<u32>),
    Remove(u32),
}

/// A row's look: picked in the list, its body under the cursor in the view, a face or an edge of
/// it selected there, hidden.
#[derive(Default, Clone, Copy)]
struct Marks {
    selected: bool,
    lit: bool,
    picked: bool,
    ghost: bool,
    drop: bool,
}

impl ReverseWorkspace {
    // ---- the components and the styles, each an undo step ----

    /// A new component — of what is given, where it stands — at the head of `parent`'s list.
    /// Its id.
    pub(crate) fn add_body_folder(&mut self, parent: Option<u32>, what: Option<&FolderMove>) -> u32 {
        let folders = &self.part.body_folders;
        let id = next_folder_id(folders);
        let mut made = folders.clone();
        made.push(BodyFolder { id, name: next_folder_name(folders), parent: parent.filter(|p| folders.iter().any(|f| f.id == *p)), bodies: Vec::new() });
        let next = what.and_then(|w| move_into(&made, w, Some(id))).unwrap_or(made);
        self.checkpoint();
        self.part.body_folders = next;
        id
    }

    pub(crate) fn rename_body_folder(&mut self, id: u32, name: &str) {
        let text = name.trim();
        let Some(f) = self.part.body_folders.iter().find(|f| f.id == id) else { return };
        if text.is_empty() || text == f.name {
            return;
        }
        self.checkpoint();
        if let Some(f) = self.part.body_folders.iter_mut().find(|f| f.id == id) {
            f.name = text.to_string();
        }
    }

    pub(crate) fn remove_body_folder(&mut self, id: u32) {
        if !self.part.body_folders.iter().any(|f| f.id == id) {
            return;
        }
        self.checkpoint();
        self.part.body_folders = without_folder(&self.part.body_folders, id);
        self.picks.closed_folders.retain(|c| *c != id);
    }

    /// Bodies and components into a component, or to the head of the list. Whether it could be.
    pub(crate) fn move_to_body_folder(&mut self, what: &FolderMove, target: Option<u32>) -> bool {
        let Some(next) = move_into(&self.part.body_folders, what, target) else { return false };
        if next != self.part.body_folders {
            self.checkpoint();
            self.part.body_folders = next;
        }
        true
    }

    pub(crate) fn set_folder_open(&mut self, id: u32, open: bool) {
        let closed = &mut self.picks.closed_folders;
        if open {
            closed.retain(|c| *c != id);
        } else if !closed.contains(&id) {
            closed.push(id);
        }
    }

    /// A body named (cleared: Solid n again) or coloured (cleared: the colour its maker gives
    /// it). One undo step; the names are read again at once.
    pub(crate) fn set_body_style(&mut self, id: BodyId, name: Option<&str>, color: Option<Option<&str>>) {
        let was = self.part.body_styles.get(&id).cloned().unwrap_or_default();
        let mut next = was.clone();
        if let Some(n) = name {
            let t = n.trim();
            next.name = (!t.is_empty()).then(|| t.to_string());
        }
        if let Some(c) = color {
            next.color = c.and_then(body_color_of);
        }
        if next == was {
            return;
        }
        self.checkpoint();
        if next == BodyStyle::default() {
            self.part.body_styles.remove(&id);
        } else {
            self.part.body_styles.insert(id, next);
        }
        mm_reverse::build::name_bodies(&mut self.built.bodies, &self.part);
        for i in 0..self.bodies.len() {
            if let Some(n) = self.built.body(self.bodies[i].id).map(|b| b.name.clone()) {
                self.bodies[i].name = n;
            }
        }
    }

    /// A built body as it goes into a STEP file: in the colour it wears.
    pub(crate) fn worn(&self, b: &mm_reverse::build::BuiltBody) -> mm_reverse::build::BuiltBody {
        let c = self.body_color32(b.id);
        let mut out = b.clone();
        out.body = out.body.clone().with_color(Some([c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0]));
        out
    }

    /// The rows picked, of what is there: a component taken away or a body no longer built is
    /// not.
    fn live_pick(&self) -> ListPick {
        let p = &self.picks.list_pick;
        ListPick { bodies: p.bodies.iter().copied().filter(|id| self.bodies.iter().any(|b| b.id == *id)).collect(), folders: p.folders.iter().copied().filter(|id| self.part.body_folders.iter().any(|f| f.id == *id)).collect() }
    }

    /// The bodies of an open Move.
    fn moving(&self) -> Option<Vec<BodyId>> {
        match &self.mode {
            Mode::Feature(edit) => match &edit.kind {
                FeatureKind::MoveBody(m) => Some(m.bodies.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    fn words_of(&self, pick: &ListPick) -> String {
        let names = |id: BodyId| self.bodies.iter().find(|b| b.id == id).map(|b| b.name.clone());
        pick_words(pick, &names, &self.part.body_folders)
    }

    /// The built bodies in a component and those inside it.
    fn built_in(&self, folder: u32) -> Vec<BodyId> {
        let within = bodies_within(&self.part.body_folders, folder);
        self.bodies.iter().map(|b| b.id).filter(|id| within.contains(id)).collect()
    }

    // ---- the group ----

    /// The Bodies group: its label with the keys for a new component and for every eye, the
    /// coordinate planes, the components and the bodies, the keys for what is picked, and the
    /// rows' right-click menu.
    pub(crate) fn bodies_group(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let moving = self.moving();
        let editor_open = !matches!(self.mode, Mode::Idle);
        let pick = self.live_pick();
        let pick_count = pick.len();
        let picked_keys = keys_of(&pick);
        let mut acts: Vec<RowAct> = Vec::new();
        let mut menu_acts: Vec<MenuAct> = Vec::new();
        let mut new_folder = false;

        // A click on the list where there is no row lets the rows picked go.
        if let Some(r) = self.picks.list_rect {
            let bg = ui.interact(r, egui::Id::NULL.with("re_body_list_bg"), Sense::click());
            if bg.clicked() && pick_count > 0 && moving.is_none() {
                self.picks.list_pick = ListPick::default();
            }
        }
        let top = ui.cursor().min;

        // The label.
        ui.horizontal(|ui| {
            ui.label(RichText::new("Bodies").strong());
            if !self.bodies.is_empty() {
                let tip = if pick_count > 0 && moving.is_none() {
                    format!("A new component holding {} — a folder to sort bodies into", self.words_of(&pick).to_lowercase())
                } else {
                    "A new component — a folder to sort bodies into, an assembly's parts by what they make up: drag bodies onto it, or pick rows first to put them in".into()
                };
                if ui.add(egui::Button::new(RichText::new("▣ COMPONENT").size(10.0)).small()).on_hover_text(tip).clicked() {
                    new_folder = true;
                }
                let any = self.bodies.iter().any(|b| !self.hidden_bodies.contains(&b.id));
                let (word, tip) = if any { ("👁 Hide all", "Hide all bodies in the viewport") } else { ("👁 Show all", "Show all bodies") };
                if ui.add(egui::Button::new(RichText::new(word).size(11.0)).small()).on_hover_text(tip).clicked() {
                    self.set_all_bodies_visible(!any);
                }
            }
            ui.label(RichText::new(self.bodies.len().to_string()).strong());
        });

        // The coordinate planes, first: XY, XZ, YZ.
        for axis in PLANE_AXES {
            let shown = self.plane_shown(axis);
            let name = axis.plane_name();
            let at = match axis {
                WorldAxis::Z => "z = 0",
                WorldAxis::Y => "y = 0",
                WorldAxis::X => "x = 0",
            };
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_H), Sense::hover());
            resp.on_hover_text(format!("The {name} through the origin — click it in the viewport to sketch on it"));
            let fade = if shown { 1.0 } else { 0.45 };
            ui.scope_builder(egui::UiBuilder::new().max_rect(rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
                ui.add_space(4.0);
                let (d, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                let c = d.center();
                // A skewed square: a plane seen at a slant.
                let pts = vec![c + egui::vec2(-5.0, 3.5), c + egui::vec2(1.5, 3.5), c + egui::vec2(5.0, -3.5), c + egui::vec2(-1.5, -3.5)];
                ui.painter().add(egui::Shape::convex_polygon(pts, plane_color(axis).gamma_multiply(fade), Stroke::NONE));
                ui.add(egui::Label::new(RichText::new(name).color(ui.visuals().text_color().gamma_multiply(fade))).selectable(false));
                ui.add(egui::Label::new(RichText::new(at).strong().color(ui.visuals().text_color().gamma_multiply(fade))).selectable(false));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let tip = if shown { format!("Hide the {name}") } else { format!("Show the {name} at the origin") };
                    if ui.small_button(if shown { "👁" } else { "–" }).on_hover_text(tip).clicked() {
                        acts.push(RowAct::PlaneEye(axis, !shown));
                    }
                });
            });
        }

        // The rows, as they stand: each component, its insides if open, then its bodies.
        let mut order: Vec<RowKey> = Vec::new();
        let view_lit: Vec<BodyId> = match &self.picks.body_hover {
            Some((ids, HoverFrom::View)) => ids.clone(),
            _ => Vec::new(),
        };
        let picked = self.picks.picked_bodies();
        let dragging = egui::DragAndDrop::payload::<DragRows>(&ctx);
        let mut hovered_rows: Option<Vec<BodyId>> = None;
        let rows_top = ui.cursor().min.y;
        self.walk_rows(ui, None, 0, &mut order, &mut acts, &mut hovered_rows, &RowCtx { moving: moving.as_deref(), picked_keys: &picked_keys, view_lit: &view_lit, picked: &picked, dragging: dragging.as_deref() });

        // While something is dragged, the strip to take it out of its component.
        if let Some(d) = &dragging {
            if move_into(&self.part.body_folders, &d.what, None).is_some_and(|n| n != self.part.body_folders) {
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_H + 4.0), Sense::hover());
                let on = resp.dnd_hover_payload::<DragRows>().is_some();
                let accent = ui.visuals().selection.bg_fill;
                ui.painter().rect(rect.shrink(1.0), 3.0, if on { accent.gamma_multiply(0.25) } else { Color32::TRANSPARENT }, Stroke::new(1.0, accent), egui::StrokeKind::Inside);
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "Drop here to take it out of its component", egui::FontId::proportional(11.0), ui.visuals().text_color());
                if let Some(p) = resp.dnd_release_payload::<DragRows>() {
                    acts.push(RowAct::Drop(p.what.clone(), None));
                }
            }
            // Several rows go as one pill that says how many.
            if d.many {
                if let Some(p) = ctx.pointer_hover_pos() {
                    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::NULL.with("re_body_drag")));
                    let galley = painter.layout_no_wrap(d.words.clone(), egui::FontId::proportional(11.0), Color32::WHITE);
                    let r = egui::Rect::from_min_size(p + egui::vec2(14.0, 12.0), galley.size() + egui::vec2(12.0, 6.0));
                    painter.rect_filled(r, 9.0, ui.visuals().selection.bg_fill);
                    painter.galley(r.min + egui::vec2(6.0, 3.0), galley, Color32::WHITE);
                }
            }
        }
        if moving.is_some() && self.bodies.len() > 1 {
            ui.label(RichText::new("Click rows to put bodies into the Move or take them out — a component's row all of its bodies; Shift+click puts in every row up to it.").small().weak());
        }

        // The keys for what is picked.
        let picked_bodies: Vec<BodyId> = bodies_of_pick(&self.part.body_folders, &pick).into_iter().filter(|id| self.bodies.iter().any(|b| b.id == *id)).collect();
        if moving.is_none() && pick_count > 0 {
            egui::Frame::new().fill(ui.visuals().selection.bg_fill.gamma_multiply(0.18)).corner_radius(3.0).inner_margin(egui::Margin::symmetric(6, 5)).show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(self.words_of(&pick)).strong());
                    let tip = if editor_open { "Finish what is open first" } else { "Move and turn these bodies together — or copies of them — by the gizmo or by numbers" };
                    let r = ui.add_enabled(!picked_bodies.is_empty() && !editor_open, egui::Button::new("Move").small());
                    if r.on_hover_text(tip).on_disabled_hover_text(tip).clicked() {
                        menu_acts.push(MenuAct::MoveBodies(picked_bodies.clone()));
                    }
                    if ui.small_button("Component").on_hover_text("Put what is picked into a new component").clicked() {
                        new_folder = true;
                    }
                    if !picked_bodies.is_empty() && picked_bodies.len() < self.bodies.len() {
                        let alone = self.others_hidden(&picked_bodies);
                        if ui.small_button(if alone { "Show all" } else { "Hide others" }).on_hover_text(if alone { "Show every body again" } else { "Hide every body but these" }).clicked() {
                            menu_acts.push(if alone { MenuAct::ShowAll } else { MenuAct::HideOthers(picked_bodies.clone()) });
                        }
                    }
                    if ui.small_button("×").on_hover_text("Let the rows go (Escape)").clicked() {
                        self.picks.list_pick = ListPick::default();
                    }
                });
            });
        }
        if !self.bodies.is_empty() {
            ui.label(RichText::new("Drawn solid, the scan a see-through veil over them. The view bar's Opaque key turns that round; its Scan key puts the scan away, its Bodies key the bodies — the map staying on the scan. Ctrl+click or Shift+click rows to pick several, and drag them onto a component to put them in; right-click a row for more.").small().weak());
        } else {
            ui.weak(if self.building() { "Building…" } else { "None yet." });
        }
        let bottom = ui.cursor().min.y;
        self.picks.list_rect = Some(egui::Rect::from_min_max(egui::pos2(top.x, rows_top.min(bottom)), egui::pos2(top.x + ui.available_width(), bottom)));

        // The pointer on a row lights its bodies; off them, what the list lit goes out.
        match hovered_rows {
            Some(ids) if dragging.is_none() => self.picks.body_hover = Some((ids, HoverFrom::List)),
            _ => {
                if matches!(self.picks.body_hover, Some((_, HoverFrom::List))) {
                    self.picks.body_hover = None;
                }
            }
        }

        if new_folder {
            let what = (pick_count > 0 && moving.is_none()).then(|| outermost(&self.part.body_folders, &pick));
            let parent = what.as_ref().and_then(|w| common_folder(&self.part.body_folders, w));
            let id = self.add_body_folder(parent, what.as_ref());
            let name = self.part.body_folders.iter().find(|f| f.id == id).map(|f| f.name.clone()).unwrap_or_default();
            self.picks.naming_folder = Some((id, name));
        }
        for a in acts {
            self.row_act(a, &order, moving.as_deref());
        }
        for a in menu_acts {
            self.menu_act(a);
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
    }

    /// The rows under a component (or at the head of the list), and on into the open ones.
    #[allow(clippy::too_many_arguments)]
    fn walk_rows(&mut self, ui: &mut egui::Ui, parent: Option<u32>, depth: usize, order: &mut Vec<RowKey>, acts: &mut Vec<RowAct>, hovered: &mut Option<Vec<BodyId>>, rc: &RowCtx<'_>) {
        let folders: Vec<BodyFolder> = self.part.body_folders.iter().filter(|f| f.parent == parent).cloned().collect();
        for f in folders {
            let open = !self.picks.closed_folders.contains(&f.id);
            let inside = self.built_in(f.id);
            let key = RowKey::Folder(f.id);
            order.push(key);
            let visible = inside.iter().any(|id| !self.hidden_bodies.contains(id));
            let marks = Marks {
                selected: match rc.moving {
                    Some(m) => !inside.is_empty() && inside.iter().all(|b| m.contains(b)),
                    None => rc.picked_keys.contains(&key),
                },
                lit: !open && inside.iter().any(|b| rc.view_lit.contains(b)),
                picked: !open && inside.iter().any(|b| rc.picked.contains(b)),
                ghost: !inside.is_empty() && !visible,
                drop: false,
            };
            self.folder_row(ui, &f, depth, open, &inside, marks, acts, hovered, rc);
            if open {
                self.walk_rows(ui, Some(f.id), depth + 1, order, acts, hovered, rc);
            }
        }
        let here: Vec<BodyId> = self.bodies.iter().map(|b| b.id).filter(|id| folder_of(&self.part.body_folders, *id).map(|f| f.id) == parent).collect();
        for id in here {
            let key = RowKey::Body(id);
            order.push(key);
            let marks = Marks {
                selected: match rc.moving {
                    Some(m) => m.contains(&id),
                    None => rc.picked_keys.contains(&key),
                },
                lit: rc.view_lit.contains(&id),
                picked: rc.picked.contains(&id),
                ghost: self.hidden_bodies.contains(&id),
                drop: false,
            };
            self.body_row(ui, id, parent, depth, marks, acts, hovered, rc);
        }
    }

    /// A row's ground and marks, and its response: a click picks it, a drag carries it.
    fn row_frame(&self, ui: &mut egui::Ui, id: egui::Id, depth: usize, m: Marks) -> (egui::Rect, egui::Response) {
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_H), Sense::click_and_drag());
        let v = ui.visuals();
        let accent = v.selection.bg_fill;
        let p = ui.painter();
        if m.selected {
            p.rect_filled(rect, 3.0, accent.gamma_multiply(0.3));
        } else if resp.hovered() {
            p.rect_filled(rect, 3.0, v.widgets.hovered.weak_bg_fill.gamma_multiply(0.6));
        }
        if m.drop {
            p.rect_stroke(rect.shrink(0.5), 3.0, Stroke::new(1.0, accent), egui::StrokeKind::Inside);
            p.rect_filled(rect, 3.0, accent.gamma_multiply(0.15));
        }
        if m.lit {
            p.rect_stroke(rect.shrink(0.5), 3.0, Stroke::new(1.0, accent), egui::StrokeKind::Inside);
        }
        if m.picked {
            p.rect_filled(egui::Rect::from_min_size(rect.min + egui::vec2(depth as f32 * INDENT_PX, 0.0), egui::vec2(2.0, rect.height())), 0.0, accent);
        }
        let _ = id;
        (rect, resp)
    }

    /// A component's row: its fold, its picture, its name — renamed by a double click — how many
    /// bodies it holds, and the eye of all of them.
    #[allow(clippy::too_many_arguments)]
    fn folder_row(&mut self, ui: &mut egui::Ui, f: &BodyFolder, depth: usize, open: bool, inside: &[BodyId], mut marks: Marks, acts: &mut Vec<RowAct>, hovered: &mut Option<Vec<BodyId>>, rc: &RowCtx<'_>) {
        let key = RowKey::Folder(f.id);
        let accepts = rc.dragging.is_some_and(|d| move_into(&self.part.body_folders, &d.what, Some(f.id)).is_some_and(|n| n != self.part.body_folders));
        let naming = self.picks.naming_folder.as_ref().filter(|(id, _)| *id == f.id).map(|(_, t)| t.clone());
        let id = egui::Id::NULL.with(("re_folder_row", f.id));
        // A drop would go into it: the row shows it.
        let probe = ui.cursor().min;
        let would = accepts && ui.ctx().pointer_hover_pos().is_some_and(|p| egui::Rect::from_min_size(probe, egui::vec2(ui.available_width(), ROW_H)).contains(p));
        marks.drop = would;
        let (rect, resp) = self.row_frame(ui, id, depth, marks);
        let visible = inside.iter().any(|b| !self.hidden_bodies.contains(b));
        let n = inside.len();
        let resp = resp.on_hover_text(format!("{} — {n} bod{}. Drag bodies onto it to put them in; right-click for more", f.name, if n == 1 { "y" } else { "ies" }));
        let fade = if marks.ghost { 0.5 } else { 1.0 };
        ui.scope_builder(egui::UiBuilder::new().max_rect(rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
            ui.add_space(1.0 + depth as f32 * INDENT_PX);
            if ui.add(egui::Button::new(if open { "▾" } else { "▸" }).small().frame(false)).on_hover_text(if open { format!("Close {}", f.name) } else { format!("Open {}", f.name) }).clicked() {
                acts.push(RowAct::Fold(f.id, !open));
            }
            ui.add(egui::Label::new(RichText::new("▣").color(ui.visuals().weak_text_color())).selectable(false));
            match naming {
                Some(mut text) => {
                    let r = ui.add(egui::TextEdit::singleline(&mut text).hint_text("Name").desired_width(120.0));
                    if !r.has_focus() && !r.lost_focus() {
                        r.request_focus();
                    }
                    if r.changed() {
                        self.picks.naming_folder = Some((f.id, text.clone()));
                    }
                    if r.lost_focus() {
                        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            self.picks.naming_folder = None;
                        } else {
                            acts.push(RowAct::KeepFolderName(f.id, text));
                        }
                    }
                }
                None => {
                    ui.add(egui::Label::new(RichText::new(&f.name).strong().color(ui.visuals().text_color().gamma_multiply(fade))).selectable(false)).on_hover_text("Double-click to name the component");
                }
            }
            ui.label(RichText::new(n.to_string()).strong().color(ui.visuals().weak_text_color()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let tip = if visible { format!("Hide every body in {}", f.name) } else { format!("Show every body in {}", f.name) };
                if ui.add_enabled(n > 0, egui::Button::new(if visible { "👁" } else { "–" }).small()).on_hover_text(tip).clicked() {
                    acts.push(RowAct::Eye(inside.to_vec(), !visible));
                }
            });
        });
        self.row_interactions(&resp, key, FolderMove { bodies: Vec::new(), folders: vec![f.id] }, f.name.clone(), Some(f.id), inside, acts, hovered, rc);
        if resp.double_clicked() {
            acts.push(RowAct::NameFolder(f.id));
        }
    }

    /// A body's row: its colour — a click opens the swatches — its name, renamed by a double
    /// click, as the STEP file names the part; what it holds, and its eye.
    #[allow(clippy::too_many_arguments)]
    fn body_row(&mut self, ui: &mut egui::Ui, id: BodyId, parent: Option<u32>, depth: usize, marks: Marks, acts: &mut Vec<RowAct>, hovered: &mut Option<Vec<BodyId>>, rc: &RowCtx<'_>) {
        let Some((name, volume, faces, triangles)) = self.bodies.iter().find(|b| b.id == id).map(|b| (b.name.clone(), b.volume, b.topology.faces.len(), b.mesh.triangle_count())) else { return };
        let key = RowKey::Body(id);
        let color = self.body_color32(id);
        let surface = self.body_is_surface(id);
        let closed = self.body_is_closed(id);
        let naming = self.picks.naming_body.as_ref().filter(|(x, _)| *x == id).map(|(_, t)| t.clone());
        let row_id = egui::Id::NULL.with(("re_body_row", id.feature, id.island));
        let (rect, resp) = self.row_frame(ui, row_id, depth, marks);
        let reading = if surface {
            "surface".to_string()
        } else if closed {
            fmt_vol(volume, DisplayUnits::Mm, 1, FmtOptions::default())
        } else {
            "⚠ open".to_string()
        };
        let fade = if marks.ghost { 0.5 } else { 1.0 };
        let shown = !self.hidden_bodies.contains(&id);
        ui.scope_builder(egui::UiBuilder::new().max_rect(rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
            ui.add_space(1.0 + depth as f32 * INDENT_PX + 3.0);
            let (d, dr) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::click());
            ui.painter().circle_filled(d.center(), 5.5, color.gamma_multiply(fade));
            if dr.on_hover_text("Colour the body — the STEP file carries it").clicked() {
                acts.push(RowAct::Paint(id));
            }
            match naming {
                Some(mut text) => {
                    let r = ui.add(egui::TextEdit::singleline(&mut text).hint_text("Name").desired_width(120.0));
                    if !r.has_focus() && !r.lost_focus() {
                        r.request_focus();
                    }
                    if r.changed() {
                        self.picks.naming_body = Some((id, text.clone()));
                    }
                    if r.lost_focus() {
                        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            self.picks.naming_body = None;
                        } else {
                            acts.push(RowAct::KeepBodyName(id, text));
                        }
                    }
                }
                None => {
                    let mut t = RichText::new(&name).color(ui.visuals().text_color().gamma_multiply(fade));
                    if marks.picked {
                        t = t.strong();
                    }
                    ui.add(egui::Label::new(t).selectable(false)).on_hover_text("Double-click to name the body — the name it has in a STEP file; cleared, it is Solid n again");
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button(if shown { "👁" } else { "–" }).on_hover_text(if shown { "Hide the body" } else { "Show the body" }).clicked() {
                    acts.push(RowAct::Eye(vec![id], !shown));
                }
                ui.add(egui::Label::new(RichText::new(reading).strong().size(11.0).color(ui.visuals().text_color().gamma_multiply(fade))).selectable(false)).on_hover_text(format!("{faces} faces · {} triangles drawn", grouped(triangles)));
            });
        });
        self.row_interactions(&resp, key, FolderMove { bodies: vec![id], folders: Vec::new() }, name.clone(), parent, &[id], acts, hovered, rc);
        if resp.double_clicked() {
            acts.push(RowAct::NameBody(id));
        }
        // The swatches its dot opens.
        if self.picks.painting == Some(id) {
            let own = self.part.body_styles.get(&id).and_then(|s| s.color.clone());
            ui.horizontal_wrapped(|ui| {
                ui.add_space(18.0 + depth as f32 * INDENT_PX);
                ui.spacing_mut().item_spacing.x = 3.0;
                for (name, hex) in BODY_SWATCHES {
                    let [r, g, bl] = rgb_of(hex).unwrap_or([128, 128, 128]);
                    let (sq, sr) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), Sense::click());
                    ui.painter().rect_filled(sq, 4.0, Color32::from_rgb(r, g, bl));
                    if own.as_deref() == Some(hex) {
                        ui.painter().rect_stroke(sq.expand(1.5), 5.0, Stroke::new(2.0, ui.visuals().selection.bg_fill), egui::StrokeKind::Outside);
                    }
                    let mut c = name.chars();
                    let title = c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default();
                    if sr.on_hover_text(title).clicked() {
                        acts.push(RowAct::Color(id, Some(hex.to_string())));
                    }
                }
                if ui.selectable_label(own.is_none(), "Auto").on_hover_text("The colour the body had — by the feature that made it").clicked() {
                    acts.push(RowAct::Color(id, None));
                }
            });
        }
    }

    /// What every row does: hover lights its bodies, a click picks it, a drag carries it (or all
    /// that is picked, when it is), a drop on it moves what is dragged where it is, a right click
    /// opens the menu.
    #[allow(clippy::too_many_arguments)]
    fn row_interactions(&mut self, resp: &egui::Response, key: RowKey, own: FolderMove, label: String, drop_target: Option<u32>, lights: &[BodyId], acts: &mut Vec<RowAct>, hovered: &mut Option<Vec<BodyId>>, rc: &RowCtx<'_>) {
        if resp.hovered() && !lights.is_empty() {
            *hovered = Some(lights.to_vec());
        }
        if resp.clicked() {
            let mods = resp.ctx.input(|i| i.modifiers);
            acts.push(RowAct::Pick(key, mods));
        }
        if resp.secondary_clicked() {
            acts.push(RowAct::Menu(key));
        }
        if resp.drag_started() {
            let many = rc.moving.is_none() && rc.picked_keys.contains(&key) && rc.picked_keys.len() > 1;
            let pick = self.live_pick();
            let what = if many { outermost(&self.part.body_folders, &pick) } else { own };
            let words = if many { self.words_of(&pick) } else { label };
            resp.dnd_set_drag_payload(DragRows { what, words, many });
        }
        if let Some(p) = resp.dnd_release_payload::<DragRows>() {
            acts.push(RowAct::Drop(p.what.clone(), drop_target));
        }
        // The rows' menu, for the rows picked — or this one.
        let menu_pick = self.picks.menu_pick.clone();
        let mut chosen: Option<MenuAct> = None;
        resp.context_menu(|ui| {
            let pick = if menu_pick.is_empty() { pick_of(&[key]) } else { menu_pick.clone() };
            ui.label(RichText::new(self.words_of(&pick)).strong());
            ui.separator();
            for (act, label, title) in self.menu_items(&pick) {
                if ui.button(label).on_hover_text(title).clicked() {
                    chosen = Some(act);
                    ui.close();
                }
            }
        });
        if let Some(a) = chosen {
            acts.push(RowAct::MenuChosen(a));
        }
    }

    /// What a right click offers for the rows picked: Hide others, a Move of their bodies, a
    /// component of them, a move into another; for one component also a component inside it,
    /// its name and its removal.
    fn menu_items(&self, pick: &ListPick) -> Vec<(MenuAct, String, String)> {
        let folders = &self.part.body_folders;
        let mut items = Vec::new();
        let ids: Vec<BodyId> = bodies_of_pick(folders, pick).into_iter().filter(|id| self.bodies.iter().any(|b| b.id == *id)).collect();
        let words = self.words_of(pick);
        let one_folder = (pick.bodies.is_empty() && pick.folders.len() == 1).then(|| folders.iter().find(|f| f.id == pick.folders[0])).flatten();
        if !ids.is_empty() && ids.len() < self.bodies.len() {
            if self.others_hidden(&ids) {
                items.push((MenuAct::ShowAll, "Show all".into(), "Show every body again".into()));
            } else {
                let but = match one_folder {
                    Some(f) => format!("those in {}", f.name),
                    None if ids.len() == 1 => "this one".into(),
                    None => "these".into(),
                };
                items.push((MenuAct::HideOthers(ids.clone()), "Hide others".into(), format!("Hide every body but {but}")));
            }
        }
        if !ids.is_empty() && matches!(self.mode, Mode::Idle) {
            let label = if ids.len() == 1 { "Move body".to_string() } else { format!("Move {} bodies", ids.len()) };
            items.push((MenuAct::MoveBodies(ids.clone()), label, "Move and turn them together — or copies of them — by the gizmo or by numbers".into()));
        }
        let what = outermost(folders, pick);
        if let Some(f) = one_folder {
            items.push((MenuAct::NewInside(f.id), "New component inside".into(), format!("A component within {}", f.name)));
            items.push((MenuAct::Rename(f.id), "Rename".into(), "Name the component".into()));
        } else {
            let one = pick.bodies.len() == 1 && pick.folders.is_empty();
            items.push((MenuAct::NewWith(what.clone()), if one { "New component with this body".into() } else { "New component with these".into() }, format!("Put {} into a component of their own, where they stand", words.to_lowercase())));
        }
        // Into every component that would take all of it, and out to the head.
        let within: std::collections::HashSet<u32> = what.folders.iter().flat_map(|id| folders_within(folders, *id)).collect();
        let from = common_folder(folders, &what);
        let spread: Vec<Option<u32>> = what.bodies.iter().map(|b| folder_of(folders, *b).map(|f| f.id)).chain(what.folders.iter().map(|id| folders.iter().find(|f| f.id == *id).and_then(|f| f.parent))).collect();
        for f in folders {
            if within.contains(&f.id) || (Some(f.id) == from && spread.iter().all(|p| *p == Some(f.id))) {
                continue;
            }
            items.push((MenuAct::MoveTo(what.clone(), Some(f.id)), format!("Move to {}", folder_path_text(folders, Some(f.id))), format!("Put {} into {}", words.to_lowercase(), f.name)));
        }
        if spread.iter().any(|p| p.is_some()) {
            items.push((MenuAct::MoveTo(what.clone(), None), "Move to the top of the list".into(), "Take it out of its component".into()));
        }
        if let Some(f) = one_folder {
            items.push((MenuAct::Remove(f.id), "Remove component".into(), format!("Take {} away — what it holds moves up a level; no body is deleted", f.name)));
        }
        items
    }

    fn menu_act(&mut self, a: MenuAct) {
        match a {
            MenuAct::ShowAll => self.set_all_bodies_visible(true),
            MenuAct::HideOthers(ids) => self.isolate_bodies(&ids),
            MenuAct::MoveBodies(ids) => {
                if matches!(self.mode, Mode::Idle) {
                    self.start_move_bodies(ids);
                }
            }
            MenuAct::NewInside(id) => {
                self.set_folder_open(id, true);
                let made = self.add_body_folder(Some(id), None);
                self.start_naming_folder(made);
            }
            MenuAct::Rename(id) => self.start_naming_folder(id),
            MenuAct::NewWith(what) => {
                let parent = common_folder(&self.part.body_folders, &what);
                let made = self.add_body_folder(parent, Some(&what));
                self.start_naming_folder(made);
            }
            MenuAct::MoveTo(what, target) => {
                self.move_to_body_folder(&what, target);
            }
            MenuAct::Remove(id) => self.remove_body_folder(id),
        }
    }

    fn start_naming_folder(&mut self, id: u32) {
        let name = self.part.body_folders.iter().find(|f| f.id == id).map(|f| f.name.clone()).unwrap_or_default();
        self.picks.naming_folder = Some((id, name));
    }

    fn row_act(&mut self, a: RowAct, order: &[RowKey], moving: Option<&[BodyId]>) {
        match a {
            RowAct::Pick(key, mods) => self.click_row(key, mods, order, moving),
            RowAct::Menu(key) => {
                // On a row of what is picked, the menu is for all of it; on another, that row is
                // picked alone first. While a Move is open, the menu is the row's.
                let pick = self.live_pick();
                if moving.is_some() {
                    self.picks.menu_pick = pick_of(&[key]);
                } else if keys_of(&pick).contains(&key) {
                    self.picks.menu_pick = pick;
                } else {
                    self.picks.list_pick = pick_of(&[key]);
                    self.picks.list_anchor = Some(key);
                    self.picks.menu_pick = pick_of(&[key]);
                }
            }
            RowAct::Drop(what, target) => {
                if self.move_to_body_folder(&what, target) {
                    if let Some(t) = target {
                        self.set_folder_open(t, true);
                    }
                }
            }
            RowAct::Eye(ids, visible) => self.set_bodies_visible(&ids, visible),
            RowAct::Fold(id, open) => self.set_folder_open(id, open),
            RowAct::NameFolder(id) => self.start_naming_folder(id),
            RowAct::NameBody(id) => {
                let name = self.part.body_styles.get(&id).and_then(|s| s.name.clone()).unwrap_or_else(|| self.body_name_of(id));
                self.picks.naming_body = Some((id, name));
            }
            RowAct::KeepFolderName(id, text) => {
                self.rename_body_folder(id, &text);
                self.picks.naming_folder = None;
            }
            RowAct::KeepBodyName(id, text) => {
                self.set_body_style(id, Some(&text), None);
                self.picks.naming_body = None;
            }
            RowAct::Paint(id) => self.picks.painting = if self.picks.painting == Some(id) { None } else { Some(id) },
            RowAct::Color(id, hex) => {
                self.set_body_style(id, None, Some(hex.as_deref()));
                self.picks.painting = None;
            }
            RowAct::PlaneEye(axis, shown) => self.set_plane_shown(axis, shown),
            RowAct::MenuChosen(m) => self.menu_act(m),
        }
    }

    /// A click on a row: picks only it; Ctrl/Cmd toggles it; Shift spans from the anchor row
    /// (Ctrl+Shift adds the span). While a Move is open, a click puts its bodies in or takes them
    /// out, and Shift puts in every row up to it.
    pub(crate) fn click_row(&mut self, key: RowKey, mods: egui::Modifiers, order: &[RowKey], moving: Option<&[BodyId]>) {
        let from = match self.picks.list_anchor {
            Some(a) if mods.shift => order.iter().position(|k| *k == a),
            _ => None,
        };
        let at = order.iter().position(|k| *k == key).unwrap_or(0);
        let span: Vec<RowKey> = match from {
            Some(f) => order[f.min(at)..=f.max(at)].to_vec(),
            None => vec![key],
        };
        if let Some(m) = moving {
            let ids: Vec<BodyId> = bodies_of_pick(&self.part.body_folders, &pick_of(&span)).into_iter().filter(|id| self.bodies.iter().any(|b| b.id == *id)).collect();
            let all = !ids.is_empty() && ids.iter().all(|id| m.contains(id));
            let next: Vec<BodyId> = if from.is_none() && all { m.iter().copied().filter(|id| !ids.contains(id)).collect() } else { m.iter().copied().chain(ids).collect() };
            self.set_move_bodies(next);
            if from.is_none() {
                self.picks.list_anchor = Some(key);
            }
            return;
        }
        let now = keys_of(&self.live_pick());
        let next: Vec<RowKey> = if from.is_some() {
            if mods.command {
                let mut n = now.clone();
                for k in span {
                    if !n.contains(&k) {
                        n.push(k);
                    }
                }
                n
            } else {
                span
            }
        } else if mods.command {
            if now.contains(&key) {
                now.into_iter().filter(|k| *k != key).collect()
            } else {
                let mut n = now;
                n.push(key);
                n
            }
        } else {
            vec![key]
        };
        if from.is_none() {
            self.picks.list_anchor = Some(key);
        }
        self.picks.list_pick = pick_of(&next);
    }

    /// The rows in the order they stand, as drawn — what Shift+click spans.
    pub(crate) fn row_order(&self) -> Vec<RowKey> {
        let mut out = Vec::new();
        self.order_under(None, &mut out);
        out
    }

    fn order_under(&self, parent: Option<u32>, out: &mut Vec<RowKey>) {
        for f in self.part.body_folders.iter().filter(|f| f.parent == parent) {
            out.push(RowKey::Folder(f.id));
            if !self.picks.closed_folders.contains(&f.id) {
                self.order_under(Some(f.id), out);
            }
        }
        for b in self.bodies.iter().filter(|b| folder_of(&self.part.body_folders, b.id).map(|f| f.id) == parent) {
            out.push(RowKey::Body(b.id));
        }
    }

    /// The rows picked, as a script or a test names them: body numbers.
    #[allow(dead_code)]
    pub(crate) fn list_pick_words(&self) -> String {
        self.words_of(&self.live_pick())
    }
}

/// What every row of one drawing of the list reads.
struct RowCtx<'a> {
    moving: Option<&'a [BodyId]>,
    picked_keys: &'a [RowKey],
    view_lit: &'a [BodyId],
    picked: &'a [BodyId],
    dragging: Option<&'a DragRows>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pick_in_words() {
        let names = |id: BodyId| Some(format!("Solid {}", id.feature));
        let folders = vec![BodyFolder { id: 1, name: "Drive".into(), parent: None, bodies: Vec::new() }];
        assert_eq!(pick_words(&ListPick { bodies: vec![BodyId::new(2)], folders: vec![] }, &names, &folders), "Solid 2");
        assert_eq!(pick_words(&ListPick { bodies: vec![], folders: vec![1] }, &names, &folders), "Drive");
        assert_eq!(pick_words(&ListPick { bodies: vec![BodyId::new(1), BodyId::new(2), BodyId::new(3)], folders: vec![] }, &names, &folders), "3 bodies");
        assert_eq!(pick_words(&ListPick { bodies: vec![BodyId::new(1), BodyId::new(2)], folders: vec![1] }, &names, &folders), "2 bodies and a component");
    }

    #[test]
    fn a_triangle_count_with_its_thousands_apart() {
        assert_eq!(grouped(12), "12");
        assert_eq!(grouped(1234), "1,234");
        assert_eq!(grouped(1234567), "1,234,567");
    }
}
