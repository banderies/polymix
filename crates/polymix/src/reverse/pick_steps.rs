// SPDX-License-Identifier: AGPL-3.0-only
//! Script steps for the selection, its chips and the body list, so they run headless in tests
//! (`POLYMIX_SCRIPT`), and the Overlaps switch of the view box. Ported behaviour is ScanRuler's
//! (`plugins/reverse/ui/ContextChips.tsx`, `ui/BodyList.tsx`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only; the steps themselves are Polymix's.
//!
//! The steps:
//! * `pick:<face|edge|corner>:<n>[:<body n>][:add]` — the n-th face, edge or corner of a body
//!   (the first shown by default), picked as a click on it takes it (with Ctrl: `add`).
//! * `pickregion[:<n>]`, `pickline`, `pickplane:<x|y|z>`, `pickelement:<id>`, `clearpick`.
//! * `clickat:<x>:<y>[:ctrl]` — a click on the view; `boxpick:<x0>:<y0>:<x1>:<y1>` — a window.
//! * `chip:<key>` — take the chip of that key (`chips.rs`); `chips` logs the card and the chips.
//! * `listpick:<n>[:ctrl|shift]` — a click on the n-th body row; `listfolder:<n>` the n-th
//!   component's; `listhover:<n>` lights the n-th body as the pointer on its row does
//!   (`listhover:none` puts it out).
//! * `component` — the Component key; `hideothers`, `showall`, `hidebody:<n>`.
//! * `bodyname:<n>:<text>`, `bodycolor:<n>:<swatch or #hex or auto>`.
//! * `planeshow:<x|y|z>:<0|1>`, `overlaps:<0|1>`, `foldermove:<body n>:<folder n|top>`.
//! * `selreport` — the selection, the rows picked, the components and the styles, to the log.

use egui::Pos2;
use glam::DVec3;
use mm_reverse::body_folders::{common_folder, outermost, FolderMove};
use mm_reverse::display_units::DisplayUnits;
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{BodyId, FeatureKind};
use mm_reverse::selection_info::selection_info;

use super::picks::{Candidate, Selected, Selection};
use super::ReverseWorkspace;

fn axis_of(s: Option<&&str>) -> WorldAxis {
    match s.copied() {
        Some("x") | Some("yz") => WorldAxis::X,
        Some("y") | Some("xz") => WorldAxis::Y,
        _ => WorldAxis::Z,
    }
}

impl ReverseWorkspace {
    /// The n-th body shown.
    fn nth_body(&self, n: usize) -> Option<BodyId> {
        self.bodies.get(n).map(|b| b.id)
    }

    /// Where a part point stands on screen in the view as last drawn, or the middle of it.
    fn screen_at(&self, p: DVec3) -> Pos2 {
        match self.picks.vp {
            Some(vp) => self.project(vp, p).unwrap_or(vp.rect.center()),
            None => Pos2::new(600.0, 400.0),
        }
    }

    /// A script step of the selection, the chips or the body list; whether it was one.
    pub(crate) fn picks_script_step(&mut self, cmd: &str, args: &[&str]) -> bool {
        let num = |i: usize, default: usize| args.get(i).and_then(|v| v.parse::<usize>().ok()).unwrap_or(default);
        match cmd {
            "pick" => {
                let add = args.contains(&"add");
                let n = num(1, 0);
                let body = args.get(2).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
                let Some(id) = self.nth_body(body) else {
                    log::warn!("pick step: no body {body}");
                    return true;
                };
                let Some(view) = self.pick_view(id).cloned() else { return true };
                let candidate = match args.first().copied() {
                    Some("face") => view.topology.faces.get(n).map(|f| Candidate::Face { body: id, face: f.hash, distance: 0.0 }),
                    Some("edge") => (n < view.topology.edges.len()).then_some(Candidate::Edge { body: id, edge: n }),
                    Some("corner") | Some("vertex") => {
                        let mut corners: Vec<DVec3> = Vec::new();
                        for e in &view.topology.edges {
                            for p in e.ends {
                                if !corners.iter().any(|c| (*c - p).length() < 1e-3) {
                                    corners.push(p);
                                }
                            }
                        }
                        corners.get(n).map(|p| Candidate::Vertex { body: id, point: *p })
                    }
                    _ => None,
                };
                let Some(c) = candidate else {
                    log::warn!("pick step: nothing at {}", args.join(":"));
                    return true;
                };
                let at = match &c {
                    Candidate::Face { face, .. } => view.topology.face(*face).map(|f| f.point),
                    Candidate::Edge { edge, .. } => view.topology.edges.get(*edge).map(|e| e.fingerprint.mid),
                    Candidate::Vertex { point, .. } => Some(*point),
                };
                let at = self.screen_at(at.unwrap_or(DVec3::ZERO));
                if let Some(item) = self.picked_item(&c) {
                    self.pick_body_item(item, at, add);
                }
            }
            "pickregion" => {
                let sketch = self.part.features.iter().rev().find(|f| matches!(f.kind, FeatureKind::Sketch(_))).map(|f| f.id);
                if let Some(s) = sketch {
                    if let Some(r) = self.built.sketches.get(&s).and_then(|b| b.regions.get(num(0, 0))) {
                        self.picks.selection = Some(Selected { what: Selection::Region { sketch: s, key: r.key.clone() }, at: Pos2::new(600.0, 400.0) });
                    }
                }
            }
            "pickline" => {
                let line = self.part.features.iter().rev().find_map(|f| match &f.kind {
                    FeatureKind::Sketch(s) => s.sketch.entities.iter().find(|e| matches!(e.shape, mm_reverse::sketch::Shape::Line { .. })).map(|e| (f.id, e.id)),
                    _ => None,
                });
                if let Some((sketch, entity)) = line {
                    self.picks.selection = Some(Selected { what: Selection::SketchLine { sketch, entity }, at: Pos2::new(600.0, 400.0) });
                }
            }
            "pickplane" => {
                let axis = axis_of(args.first());
                self.picks.selection = Some(Selected { what: Selection::CoordinatePlane { axis }, at: Pos2::new(600.0, 400.0) });
            }
            "pickelement" => self.select_element(num(0, 1) as u32, Pos2::new(600.0, 400.0)),
            "clearpick" => self.picks.selection = None,
            // clickat:<x>:<y>[:ctrl] — a click on the view at a screen point, as the pointer
            // makes it; boxpick:<x0>:<y0>:<x1>:<y1> — a Shift+drag window.
            "clickat" | "boxpick" => {
                let f = |i: usize| args.get(i).and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
                let Some(vp) = self.picks.vp else { return true };
                if cmd == "clickat" {
                    let mods = if args.get(2) == Some(&"ctrl") { egui::Modifiers::COMMAND } else { egui::Modifiers::NONE };
                    self.idle_pointer(vp, Some(Pos2::new(f(0), f(1))), false, false, false, true, mods);
                } else {
                    let (a, b) = (Pos2::new(f(0), f(1)), Pos2::new(f(2), f(3)));
                    self.idle_pointer(vp, Some(a), true, true, false, false, egui::Modifiers::SHIFT);
                    self.idle_pointer(vp, Some(b), false, false, true, false, egui::Modifiers::SHIFT);
                }
            }
            "chip" => {
                let key = args.first().copied().unwrap_or("");
                match self.current_chips().into_iter().find(|c| c.key == key) {
                    Some(c) => {
                        self.picks.selection = None;
                        self.run_chip(c.action);
                    }
                    None => log::warn!("chip step: no chip {key} for the selection"),
                }
            }
            "chips" => {
                let chips: Vec<String> = self.current_chips().iter().map(|c| format!("{}={}", c.key, c.label)).collect();
                let info = selection_info(self.picks.items(), DisplayUnits::Mm).map(|i| format!("{}: {}", i.title, i.rows.iter().map(|r| format!("{} {}", r.label, r.value)).collect::<Vec<_>>().join("; "))).unwrap_or_default();
                log::info!("reverse chips: [{}] card [{}]", chips.join(", "), info);
            }
            "listpick" => {
                let Some(id) = self.nth_body(num(0, 0)) else { return true };
                let mods = match args.get(1).copied() {
                    Some("ctrl") => egui::Modifiers::COMMAND,
                    Some("shift") => egui::Modifiers::SHIFT,
                    _ => egui::Modifiers::NONE,
                };
                self.script_pick_row(super::bodies::RowKey::Body(id), mods);
            }
            "listfolder" => {
                if let Some(f) = self.part.body_folders.get(num(0, 0)).map(|f| f.id) {
                    self.script_pick_row(super::bodies::RowKey::Folder(f), egui::Modifiers::NONE);
                }
            }
            "listhover" => {
                self.picks.script_lit = if args.first() == Some(&"none") { Vec::new() } else { self.nth_body(num(0, 0)).into_iter().collect() };
            }
            "component" => {
                let pick = self.picks.list_pick.clone();
                let what = (!pick.is_empty()).then(|| outermost(&self.part.body_folders, &pick));
                let parent = what.as_ref().and_then(|w| common_folder(&self.part.body_folders, w));
                self.add_body_folder(parent, what.as_ref());
            }
            "foldermove" => {
                let Some(id) = self.nth_body(num(0, 0)) else { return true };
                let target = match args.get(1).copied() {
                    Some("top") | None => None,
                    Some(n) => n.parse::<usize>().ok().and_then(|n| self.part.body_folders.get(n)).map(|f| f.id),
                };
                self.move_to_body_folder(&FolderMove { bodies: vec![id], folders: Vec::new() }, target);
            }
            "hideothers" => {
                let ids = mm_reverse::body_folders::bodies_of_pick(&self.part.body_folders, &self.picks.list_pick);
                let ids = if ids.is_empty() { self.picks.picked_bodies() } else { ids };
                self.isolate_bodies(&ids);
            }
            "showall" => self.set_all_bodies_visible(true),
            "hidebody" => {
                if let Some(id) = self.nth_body(num(0, 0)) {
                    self.set_bodies_visible(&[id], false);
                }
            }
            "bodyname" => {
                if let Some(id) = self.nth_body(num(0, 0)) {
                    let text = args.get(1..).map(|a| a.join(":")).unwrap_or_default();
                    self.set_body_style(id, Some(&text), None);
                }
            }
            "bodycolor" => {
                if let Some(id) = self.nth_body(num(0, 0)) {
                    let c = args.get(1).copied().unwrap_or("auto");
                    self.set_body_style(id, None, Some((c != "auto").then_some(c)));
                }
            }
            "planeshow" => {
                let axis = axis_of(args.first());
                self.set_plane_shown(axis, args.get(1) != Some(&"0"));
            }
            "overlaps" => self.set_overlaps_shown(args.first() != Some(&"0")),
            "selreport" => {
                let sel = match &self.picks.selection {
                    Some(s) => match &s.what {
                        Selection::Body(items) => format!("{} on the bodies: {}", items.len(), items.iter().map(|i| i.word()).collect::<Vec<_>>().join(", ")),
                        other => format!("{other:?}"),
                    },
                    None => "none".into(),
                };
                let folders: Vec<String> = self.part.body_folders.iter().map(|f| format!("{} ({}) in {:?} holds [{}]", f.name, f.id, f.parent, f.bodies.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", "))).collect();
                let styles: Vec<String> = self.part.body_styles.iter().map(|(id, s)| format!("{id}: {:?} {:?}", s.name, s.color)).collect();
                let hidden: Vec<String> = self.hidden_bodies.iter().map(|b| b.to_string()).collect();
                let names: Vec<String> = self.bodies.iter().map(|b| b.name.clone()).collect();
                let mode = match &self.mode {
                    super::Mode::Idle => "idle".to_string(),
                    super::Mode::Sketch(_) => "sketch".to_string(),
                    super::Mode::Feature(e) => format!("feature {}", e.kind.kind()),
                };
                log::info!(
                    "reverse picks: selection {sel}; rows picked {}; components [{}]; styles [{}]; hidden [{}]; bodies [{}]; overlaps {} ({} shown); mode {mode}; status: {}",
                    self.list_pick_words(),
                    folders.join("; "),
                    styles.join("; "),
                    hidden.join(", "),
                    names.join(", "),
                    self.picks.overlaps.len(),
                    self.part.overlaps_shown,
                    self.status
                );
            }
            _ => return false,
        }
        true
    }

    fn script_pick_row(&mut self, key: super::bodies::RowKey, mods: egui::Modifiers) {
        let order = self.row_order();
        let moving = match &self.mode {
            super::Mode::Feature(e) => match &e.kind {
                FeatureKind::MoveBody(m) => Some(m.bodies.clone()),
                _ => None,
            },
            _ => None,
        };
        self.click_row(key, mods, &order, moving.as_deref());
    }

    /// The Overlaps switch: kept with the part, no step of the undo history.
    pub(crate) fn set_overlaps_shown(&mut self, on: bool) {
        self.part.overlaps_shown = on;
        for p in self.undo.iter_mut().chain(self.redo.iter_mut()) {
            p.overlaps_shown = on;
        }
        self.autosave_due = true;
    }
}
