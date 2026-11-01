// SPDX-License-Identifier: AGPL-3.0-only
//! The model's parameters and the number fields that read them, in the Reverse Engineer panel.
//! Ported from ScanRuler (`plugins/reverse/ui/ParameterList.tsx`, `ui/ExprInput.tsx`, the `Num`
//! field of `ui/featureFields.tsx`, `core/nominal.ts` nominalText, the shared `src/ui/InfoDot.tsx`,
//! the units setting of `src/ui/SettingsModal.tsx`, and the store's parameter actions in
//! `state/reStore.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A parameter is a named number — a length, an angle, a whole number or a plain number —
//! given as an expression, in one list for the whole part (`Part::parameters`). Every number
//! field of the workspace takes an expression too ([`expr_input`], [`num_field`]); a feature
//! keeps the expression beside the number it came to, under the field's slot path, and the
//! numbers are written from the expressions again whenever a parameter changes and before
//! every build (`mm_reverse::part_parameters`). The kernel only ever sees numbers.
//!
//! Lengths are held in millimetres and shown and typed in the display units — millimetres or
//! inches, a setting of this workspace (ScanRuler's is app-wide) kept between sessions.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use egui::text::{CCursorRange, LayoutJob};
use egui::{Color32, RichText};
use mm_reverse::display_units::{fmt_len, unit_word, DisplayUnits, FmtOptions, DISPLAY_UNITS};
use mm_reverse::expression::{evaluate_for, format_value, ParameterKind};
use mm_reverse::js_number::to_fixed;
use mm_reverse::length_fields::{field_text, parameter_expression, parameter_text, read_field};
use mm_reverse::model::{Exprs, FeatureId, FeatureKind, Parameter};
use mm_reverse::nominal::{nominal_of, NominalKind};
use mm_reverse::parameters::{evaluate_parameters, ExprErrors, ParameterId, ParameterPatch, ParameterTable, PARAMETER_KINDS};
use mm_reverse::part_parameters::{self, set_draft_expr, Drafts};

use super::{editors, Mode, ReverseWorkspace};

// ---- colours (the spec's tokens, light and dark) ----

/// The accent: what an expression-driven field's text and the nominal chip are drawn in.
pub(crate) fn accent(ui: &egui::Ui) -> Color32 {
    if ui.visuals().dark_mode {
        Color32::from_rgb(0x3f, 0x93, 0xd6)
    } else {
        Color32::from_rgb(0x18, 0x77, 0xc0)
    }
}

/// The warning hint's colour: an expression refused, a row refused.
pub(crate) fn warn(ui: &egui::Ui) -> Color32 {
    if ui.visuals().dark_mode {
        Color32::from_rgb(0xe2, 0x6b, 0x4f)
    } else {
        Color32::from_rgb(0xb3, 0x36, 0x1c)
    }
}

/// Labels and hints.
pub(crate) fn dim(ui: &egui::Ui) -> Color32 {
    if ui.visuals().dark_mode {
        Color32::from_rgb(0xa0, 0xa4, 0xa8)
    } else {
        Color32::from_rgb(0x75, 0x78, 0x7b)
    }
}

// ---- the workspace's state ----

/// What the parameter list and the number fields keep between frames.
pub(crate) struct ParamsState {
    /// The Parameters section is open; it starts folded.
    pub open: bool,
    /// The row just added, whose name takes the focus.
    fresh: Option<ParameterId>,
    /// Why the last edit of a row was refused, under the row, until an edit is taken.
    refusals: HashMap<ParameterId, String>,
    /// The rows whose note line is open.
    noting: HashSet<ParameterId>,
    /// What lengths are shown and typed in.
    pub units: DisplayUnits,
    /// The table of the parameters as last read, and the list it was read from.
    table: RefCell<Option<(Vec<Parameter>, Rc<ParameterTable>)>>,
    /// The expressions that did not evaluate when the part was last brought up to its
    /// parameters, by feature.
    pub errors: BTreeMap<FeatureId, ExprErrors>,
    /// Text a script typed into a field, by the field's name, taken by the field as typed and
    /// entered.
    typed: RefCell<Option<(String, String)>>,
}

impl ParamsState {
    /// The state as the last session left its settings.
    pub fn new(storage: Option<&dyn eframe::Storage>) -> Self {
        let units = storage.and_then(|s| s.get_string("reverse.units")).and_then(|u| mm_reverse::display_units::display_units_of(&u)).unwrap_or_default();
        Self { open: false, fresh: None, refusals: HashMap::new(), noting: HashSet::new(), units, table: RefCell::new(None), errors: BTreeMap::new(), typed: RefCell::new(None) }
    }

    /// Keep the settings for the next session.
    pub fn save(&self, storage: &mut dyn eframe::Storage) {
        storage.set_string("reverse.units", self.units.id().to_string());
    }

    /// The text a script typed into the field of this name, once.
    fn take_typed(&self, id: &str) -> Option<String> {
        let mut t = self.typed.borrow_mut();
        if t.as_ref().is_some_and(|(field, _)| field == id) {
            return t.take().map(|(_, text)| text);
        }
        None
    }
}

impl ReverseWorkspace {
    /// The parameters evaluated, as a scope for the number fields: read again when the list
    /// changed.
    pub(crate) fn param_table(&self) -> Rc<ParameterTable> {
        let mut cache = self.params.table.borrow_mut();
        if let Some((list, table)) = cache.as_ref() {
            if *list == self.part.parameters {
                return table.clone();
            }
        }
        let table = Rc::new(evaluate_parameters(&self.part.parameters));
        *cache = Some((self.part.parameters.clone(), table.clone()));
        table
    }

    /// What lengths are shown and typed in.
    pub(crate) fn units(&self) -> DisplayUnits {
        self.params.units
    }

    /// The part's numbers brought up to its parameters, before a build: a number left behind
    /// by an older save or an undo is written from its expression again (ScanRuler writes them
    /// as a parameter changes; this does it then and before every build as well).
    pub(crate) fn follow_parameters_before_build(&mut self) {
        let (changed, errors) = part_parameters::apply_to_part(&mut self.part);
        self.params.errors = errors;
        if changed {
            self.autosave_due = true;
        }
    }

    /// The drafts open in the boxes, as an edit of the parameters reaches them.
    fn drafts_of(mode: &Mode) -> Drafts<'_> {
        match mode {
            Mode::Feature(e) => Drafts { feature: Some(&e.kind), sketch: None },
            Mode::Sketch(e) => Drafts { feature: None, sketch: Some((&e.sketch, &e.frame)) },
            Mode::Idle => Drafts::default(),
        }
    }

    /// A parameter's row edited (ScanRuler's `setParameter`): refused with why — nothing
    /// changed — or taken as one step of Undo, the features, the open box and the sketch being
    /// drawn following it, and the part built again when a number moved.
    pub(crate) fn set_parameter(&mut self, id: ParameterId, patch: ParameterPatch) -> Result<(), String> {
        let before = self.part.clone();
        let r = part_parameters::set_parameter(&mut self.part, id, &patch, Self::drafts_of(&self.mode));
        let followed = match r {
            Err(why) => return Err(why),
            Ok(None) => return Ok(()),
            Ok(Some(f)) => f,
        };
        // The step to undo is the part as it stood.
        let after = std::mem::replace(&mut self.part, before);
        self.checkpoint();
        self.part = after;
        self.params.errors = followed.errors;
        match &mut self.mode {
            Mode::Feature(edit) => {
                if let Some(kind) = followed.feature {
                    edit.kind = kind;
                }
            }
            Mode::Sketch(edit) => {
                if let Some(sketch) = followed.sketch {
                    // A dimension moved is a step of the sketch's own history, as one typed
                    // is; a rename moves nothing.
                    if followed.sketch_moved {
                        edit.checkpoint();
                    }
                    edit.sketch = sketch;
                    edit.solve();
                }
            }
            Mode::Idle => {}
        }
        self.request_build();
        Ok(())
    }

    /// A new parameter at zero, named in turn (ScanRuler's `addParameter`): one step of Undo.
    pub(crate) fn add_parameter(&mut self, kind: ParameterKind) -> ParameterId {
        self.checkpoint();
        part_parameters::add_parameter(&mut self.part, kind)
    }

    /// A parameter removed (ScanRuler's `removeParameter`): refused, naming what reads it.
    pub(crate) fn remove_parameter(&mut self, id: ParameterId) -> Result<(), String> {
        let mut part = self.part.clone();
        if !part_parameters::remove_parameter(&mut part, id, Self::drafts_of(&self.mode))? {
            return Ok(());
        }
        self.checkpoint();
        self.part = part;
        self.params.refusals.remove(&id);
        self.params.noting.remove(&id);
        Ok(())
    }

    /// The expressions of a feature that did not evaluate at the last build, by slot.
    #[allow(dead_code)]
    pub(crate) fn expression_errors(&self, id: FeatureId) -> Option<&ExprErrors> {
        self.params.errors.get(&id)
    }

    // ---- the Parameters group ----

    /// The Parameters section of the panel: folded at first; a row a parameter — its name, its
    /// kind, the value it comes to, a note, and the expression it is given — and a key to add
    /// one. A row refused says why under it and keeps what was typed.
    pub(crate) fn parameters_group(&mut self, ui: &mut egui::Ui) {
        let n = self.part.parameters.len();
        let mut add = false;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 7.0;
            let open = self.params.open;
            if fold_key(ui, open, "Parameters").on_hover_text(if open { "Fold the parameters away" } else { "Show the parameters" }).clicked() {
                self.params.open = !open;
            }
            info_dot(ui, "re_parameters_info", "Parameters", &PARAMETERS_CARD);
            ui.label(RichText::new(n.to_string()).strong());
            if ui.add(egui::Button::new(RichText::new("+ Add").size(11.0))).on_hover_text("Add a parameter").clicked() {
                add = true;
            }
        });
        if add {
            let id = self.add_parameter(ParameterKind::Length);
            self.params.open = true;
            self.params.fresh = Some(id);
        }
        if !self.params.open {
            return;
        }
        if self.part.parameters.is_empty() {
            hint(ui, "No parameters yet. Add one, name it, and use the name in any number field.");
            return;
        }
        let table = self.param_table();
        let units = self.units();
        let names: Vec<String> = self.part.parameters.iter().map(|p| p.name.clone()).collect();
        let mut actions: Vec<(ParameterId, RowAction)> = Vec::new();
        for p in &self.part.parameters {
            let reading = table.reading(p.id);
            let refusal = self.params.refusals.get(&p.id).map(String::as_str);
            let problem = refusal.or_else(|| reading.and_then(|r| r.error.as_deref()));
            let row = ParameterRow { p, value: reading.and_then(|r| r.value), problem, noting: self.params.noting.contains(&p.id), focus: self.params.fresh == Some(p.id), units, names: &names };
            if let Some(a) = parameter_row(ui, &row) {
                actions.push((p.id, a));
            }
        }
        if self.params.fresh.is_some() {
            self.params.fresh = None;
        }
        for (id, action) in actions {
            let told = match action {
                RowAction::Note => {
                    if !self.params.noting.remove(&id) {
                        self.params.noting.insert(id);
                    }
                    continue;
                }
                RowAction::Remove => self.remove_parameter(id),
                RowAction::Name(name) => self.set_parameter(id, ParameterPatch { name: Some(name), ..Default::default() }),
                RowAction::Kind(kind) => self.set_parameter(id, ParameterPatch { kind: Some(kind), ..Default::default() }),
                RowAction::Expression(expression) => self.set_parameter(id, ParameterPatch { expression: Some(expression), ..Default::default() }),
                RowAction::Comment(comment) => self.set_parameter(id, ParameterPatch { comment: Some(comment), ..Default::default() }),
            };
            match told {
                Ok(()) => {
                    self.params.refusals.remove(&id);
                }
                Err(why) => {
                    self.params.refusals.insert(id, why);
                }
            }
        }
    }

    /// The display units' choice (ScanRuler's Settings → Units), with what it changes.
    pub(crate) fn units_choice(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Lengths in");
            for u in DISPLAY_UNITS {
                if ui.selectable_label(self.params.units == u, u.label()).clicked() && self.params.units != u {
                    self.params.units = u;
                }
            }
        });
        hint(ui, "What is shown and typed. Files, projects and agent answers stay in millimetres.");
    }

    // ---- script steps (tests) ----

    /// A script step of the parameters: `param:<name>:<expression>[:<kind>]` sets a parameter
    /// (added when there is none of that name), `param-rename:<from>:<to>`,
    /// `param-kind:<name>:<kind>`, `param-remove:<name>`, `param-open`, `param-fold`,
    /// `param-units:<mm|in>`, `param-box:<kind>` opens a feature box, `param-edit:<feature name>`
    /// opens one for editing, `param-type:<field>:<text>`
    /// types into a field of the panel and enters it, `param-create` / `param-cancel` close the
    /// box, and `param-report` writes the parameters and the features' expressions to the log.
    pub(crate) fn param_script_step(&mut self, cmd: &str, args: &[&str]) {
        let by_name = |ws: &ReverseWorkspace, name: &str| ws.part.parameters.iter().find(|p| p.name == name).map(|p| p.id);
        let kind_of = |s: Option<&&str>| match s.copied() {
            Some("angle") => ParameterKind::Angle,
            Some("integer") => ParameterKind::Integer,
            Some("number") => ParameterKind::Number,
            _ => ParameterKind::Length,
        };
        let mut target = None;
        let told = match cmd {
            "param" => {
                let (Some(name), Some(expression)) = (args.first(), args.get(1)) else {
                    log::warn!("param step: param:<name>:<expression>[:<kind>]");
                    return;
                };
                let kind = args.get(2).map(|_| kind_of(args.get(2)));
                target = by_name(self, name);
                match target {
                    Some(id) => self.set_parameter(id, ParameterPatch { expression: Some(expression.to_string()), kind, ..Default::default() }),
                    None => {
                        let id = self.add_parameter(kind.unwrap_or(ParameterKind::Length));
                        self.params.open = true;
                        target = Some(id);
                        self.set_parameter(id, ParameterPatch { name: Some(name.to_string()), expression: Some(expression.to_string()), ..Default::default() })
                    }
                }
            }
            "param-rename" | "param-kind" | "param-remove" => {
                target = args.first().and_then(|n| by_name(self, n));
                match target {
                    None => Err(format!("There is no parameter {}.", args.first().unwrap_or(&""))),
                    Some(id) => match cmd {
                        "param-rename" => self.set_parameter(id, ParameterPatch { name: args.get(1).map(|s| s.to_string()), ..Default::default() }),
                        "param-kind" => self.set_parameter(id, ParameterPatch { kind: Some(kind_of(args.get(1))), ..Default::default() }),
                        _ => self.remove_parameter(id),
                    },
                }
            }
            "param-open" | "param-fold" => {
                self.params.open = cmd == "param-open";
                Ok(())
            }
            "param-units" => {
                self.params.units = if args.first() == Some(&"in") { DisplayUnits::In } else { DisplayUnits::Mm };
                Ok(())
            }
            "param-box" => {
                match args.first().copied() {
                    Some("extrude") => self.new_extrude(None, Vec::new()),
                    Some("revolve") => self.new_revolve(None, Vec::new()),
                    Some(kind) => match editors::new_feature(kind, self) {
                        Ok(k) => self.open_feature(k, None),
                        Err(e) => self.status = e,
                    },
                    None => {}
                }
                Ok(())
            }
            "param-type" => {
                let (Some(field), Some(text)) = (args.first(), args.get(1)) else {
                    log::warn!("param-type step: param-type:<field>:<text>");
                    return;
                };
                *self.params.typed.borrow_mut() = Some((field.to_string(), args[1..].join(":").trim().to_string()));
                let _ = text;
                Ok(())
            }
            "param-edit" => {
                match args.first().and_then(|n| self.part.features.iter().find(|f| f.name == *n)).map(|f| f.id) {
                    Some(id) => self.edit_feature(id),
                    None => self.status = format!("There is no feature {}.", args.first().unwrap_or(&"")),
                }
                Ok(())
            }
            "param-create" => {
                self.commit_feature();
                Ok(())
            }
            "param-cancel" => {
                self.close_mode();
                Ok(())
            }
            "param-report" => {
                let table = self.param_table();
                let params: Vec<String> = table.readings.iter().map(|r| format!("{} = {}", r.name, r.value.map_or_else(|| format!("— ({})", r.error.clone().unwrap_or_default()), |v| format_value(v, r.kind, self.units())))).collect();
                let features: Vec<String> = self
                    .part
                    .features
                    .iter()
                    .filter_map(|f| {
                        let e = f.kind.exprs().filter(|e| !e.is_empty())?;
                        let shown: Vec<String> = e.iter().map(|(k, v)| format!("{k}={v}")).collect();
                        Some(format!("{}: {} [{}]", f.name, shown.join(", "), feature_numbers(&f.kind)))
                    })
                    .collect();
                log::info!("reverse params: [{}], features [{}], errors {:?}, status: {}", params.join("; "), features.join("; "), self.params.errors, self.status);
                Ok(())
            }
            other => {
                log::warn!("unknown parameter script step {other}");
                Ok(())
            }
        };
        match (told, target) {
            (Err(why), t) => {
                self.status = why.clone();
                log::info!("reverse param step refused: {why}");
                // Said under the row, as a refused edit typed there is.
                if let Some(id) = t {
                    self.params.refusals.insert(id, why);
                }
            }
            (Ok(()), Some(id)) => {
                self.params.refusals.remove(&id);
            }
            (Ok(()), None) => {}
        }
    }
}

/// The numbers of a feature as a report reads them.
fn feature_numbers(kind: &FeatureKind) -> String {
    match kind {
        FeatureKind::Extrude(e) => format!("{:?}, taper {}", e.extent, -e.draft_deg),
        FeatureKind::Revolve(r) => format!("angle {}", r.angle_deg),
        FeatureKind::Fillet(f) => format!("radius {}", f.radius),
        FeatureKind::Chamfer(c) => format!("distance {}", c.distance),
        FeatureKind::Shell(s) => format!("wall {}", s.thickness),
        _ => String::new(),
    }
}

/// The Parameters card (an ⓘ beside the section's head).
const PARAMETERS_CARD: [&str; 3] = [
    "A parameter is a named number — a length, an angle, a whole number or a plain number — given as an expression: `25`, `25.4/12`, `width/2 + 1`. Every number field in the workspace takes an expression too, and a parameter by name: an extrude `height + 5` deep, a sketch's width `width`. Change the parameter and every number that reads it follows, and the part is built again.",
    "Numbers add and multiply as in a calculator, with `sin cos tan asin acos atan atan2 sqrt abs min max floor ceil round pow`, `pi` and `e`; angles are in degrees. A number may carry a unit — `1 in`, `2 cm`, `30 deg` — and one without takes the field's. Kinds are kept apart: a length and an angle cannot be added, and a length times a length is an area no field wants.",
    "A plain number typed over a field that read a parameter lets the parameter go; so does dragging the field's grip. Renaming a parameter renames it in every expression; one that something reads cannot be removed.",
];

/// What a parameter's row asks for.
enum RowAction {
    Name(String),
    Kind(ParameterKind),
    Expression(String),
    Comment(String),
    Note,
    Remove,
}

struct ParameterRow<'a> {
    p: &'a Parameter,
    value: Option<f64>,
    problem: Option<&'a str>,
    noting: bool,
    focus: bool,
    units: DisplayUnits,
    names: &'a [String],
}

/// One parameter's row: its name, kind, value, note key and remove key; under them the
/// expression, the full width; the note when open; and why the last edit was refused.
fn parameter_row(ui: &mut egui::Ui, row: &ParameterRow) -> Option<RowAction> {
    let p = row.p;
    let mut action = None;
    let line = egui::Layout::right_to_left(egui::Align::Center);
    ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), 20.0), line, |ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        if ui.add(egui::Button::new("×").frame(false)).on_hover_text("Remove the parameter — refused while anything reads it").clicked() {
            action = Some(RowAction::Remove);
        }
        let comment = p.comment.as_deref().filter(|c| !c.is_empty());
        let note_tip = comment.map_or_else(|| "Add a note on this parameter".to_string(), |c| format!("Note: {c}"));
        let pen = RichText::new("✏").color(if row.noting { accent(ui) } else { ui.visuals().text_color() });
        if ui.add(egui::Button::new(pen).frame(false)).on_hover_text(note_tip).clicked() {
            action = Some(RowAction::Note);
        }
        let shown = row.value.map_or_else(|| "—".to_string(), |v| format_value(v, p.kind, row.units));
        let mut value = RichText::new(shown).strong();
        if row.problem.is_some() {
            value = value.color(warn(ui));
        }
        let r = ui.add(egui::Label::new(value));
        // At least 44 px wide, so the kinds line up down the list.
        if r.rect.width() < 44.0 {
            ui.add_space(44.0 - r.rect.width());
        }
        if let Some(problem) = row.problem {
            r.on_hover_text(problem);
        }
        let kind_tip = format!("What the parameter is: a length in {}, an angle in degrees, a whole number, or a plain number", if row.units == DisplayUnits::In { "inches" } else { "millimetres" });
        let mut kind = p.kind;
        let label = PARAMETER_KINDS.iter().find(|k| k.kind == kind).map_or("", |k| k.label);
        egui::ComboBox::from_id_salt(("re_parameter_kind", p.id))
            .width(72.0)
            .selected_text(RichText::new(label).size(11.0))
            .show_ui(ui, |ui| {
                for k in PARAMETER_KINDS {
                    ui.selectable_value(&mut kind, k.kind, k.label);
                }
            })
            .response
            .on_hover_text(kind_tip);
        if kind != p.kind {
            action = Some(RowAction::Kind(kind));
        }
        let width = ui.available_width();
        if let Some(name) = text_commit(ui, egui::Id::NULL.with(("re_parameter_name", p.id)), &p.name, "", "The parameter's name, as expressions use it", true, row.names, width, row.focus) {
            action = Some(RowAction::Name(name));
        }
    });
    // The expression as the row shows it; left as shown, the one kept stays as it was — a
    // figure shown rounded in inches is not taken back.
    let shown_expression = parameter_text(&p.expression, p.kind, row.units);
    let width = ui.available_width();
    if let Some(typed) = text_commit(ui, egui::Id::NULL.with(("re_parameter_expression", p.id)), &shown_expression, "", "The expression the parameter is given — a number, or arithmetic over numbers and other parameters", true, row.names, width, false) {
        let kept = if typed == shown_expression.trim() { p.expression.clone() } else { parameter_expression(&typed, p.kind, row.units) };
        action = Some(RowAction::Expression(kept));
    }
    if row.noting {
        let width = ui.available_width();
        if let Some(comment) = text_commit(ui, egui::Id::NULL.with(("re_parameter_comment", p.id)), p.comment.as_deref().unwrap_or(""), "A note: what the parameter is for", "A note on the parameter", false, &[], width, false) {
            action = Some(RowAction::Comment(comment));
        }
    }
    if let Some(problem) = row.problem {
        ui.label(RichText::new(problem).color(warn(ui)).size(11.5));
    }
    ui.add(egui::Separator::default().spacing(4.0));
    action
}

/// A folding section's head (ScanRuler's `fold` key): a small dim triangle — ▸ folded, ▾ open —
/// then the head's words; the whole is the key.
pub(crate) fn fold_key(ui: &mut egui::Ui, open: bool, text: &str) -> egui::Response {
    let r = ui.add(egui::Button::new(sec_head(&format!("    {text}"))).frame(false));
    let c = egui::pos2(r.rect.left() + 6.0, r.rect.center().y);
    let s = 3.5;
    let points = if open { vec![c + egui::vec2(-s, -s * 0.6), c + egui::vec2(s, -s * 0.6), c + egui::vec2(0.0, s * 0.8)] } else { vec![c + egui::vec2(-s * 0.6, -s), c + egui::vec2(-s * 0.6, s), c + egui::vec2(s * 0.8, 0.0)] };
    ui.painter().add(egui::Shape::convex_polygon(points, dim(ui), egui::Stroke::NONE));
    r
}

/// A section's head: condensed capitals, spaced.
pub(crate) fn sec_head(text: &str) -> RichText {
    RichText::new(text.to_uppercase()).strong().size(11.5).extra_letter_spacing(1.6)
}

/// A small dim paragraph.
pub(crate) fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).size(11.5).color(dim(ui)));
}

/// A paragraph with `code` and **bold** spans, wrapped to `width`.
fn paragraph(ui: &mut egui::Ui, text: &str, width: f32) {
    let mut job = LayoutJob::default();
    let (mut code, mut bold) = (false, false);
    let mut run = String::new();
    let flush = |job: &mut LayoutJob, run: &mut String, code: bool, bold: bool, ui: &egui::Ui| {
        if run.is_empty() {
            return;
        }
        let mut t = RichText::new(std::mem::take(run));
        if code {
            t = t.code();
        }
        if bold {
            t = t.strong();
        }
        t.append_to(job, ui.style(), egui::FontSelection::Default, egui::Align::Center);
    };
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '`' {
            flush(&mut job, &mut run, code, bold, ui);
            code = !code;
        } else if c == '*' && chars.peek() == Some(&'*') {
            chars.next();
            flush(&mut job, &mut run, code, bold, ui);
            bold = !bold;
        } else {
            run.push(c);
        }
    }
    flush(&mut job, &mut run, code, bold, ui);
    job.wrap.max_width = width;
    ui.label(job);
}

/// An ⓘ after a label: a click opens a card — a heading and paragraphs (`code` and **bold**
/// spans read as such) — outside the panel, level with the dot; a second click, a click
/// anywhere else, or Escape closes it (ScanRuler's `InfoDot`).
pub(crate) fn info_dot(ui: &mut egui::Ui, id_salt: &str, title: &str, paragraphs: &[&str]) {
    let id = egui::Id::NULL.with(("re_info_dot", id_salt));
    let mut open = ui.data(|d| d.get_temp::<bool>(id)).unwrap_or(false);
    let dot = ui.add(egui::Button::new(RichText::new("ⓘ").size(12.0).color(dim(ui))).frame(false));
    if dot.clicked() {
        open = !open;
    }
    if open {
        let x = ui.clip_rect().right().max(dot.rect.right()) + 16.0;
        let card = egui::Area::new(id.with("card")).order(egui::Order::Foreground).fixed_pos(egui::pos2(x, dot.rect.top() - 6.0)).show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_max_width(320.0);
                ui.label(RichText::new(title).strong());
                for p in paragraphs {
                    paragraph(ui, p, 320.0);
                }
            });
        });
        let pressed_elsewhere = ui.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| !card.response.rect.contains(p) && !dot.rect.contains(p)));
        let escape = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        if pressed_elsewhere || escape {
            open = false;
        }
    }
    ui.data_mut(|d| d.insert_temp(id, open));
}

// ---- text fields ----

/// What a text field keeps between frames: the text in it, the text it was last given to
/// show, why it was refused, whether it had the focus, and the pass it was last drawn in.
#[derive(Clone, Default)]
struct FieldMemory {
    text: String,
    seeded: String,
    error: Option<String>,
    focused: bool,
    pass: u64,
}

/// What a key did to a field this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Left {
    /// Still being typed in, or not touched.
    No,
    /// Left by a click elsewhere or Tab.
    Blur,
    Enter,
    Escape,
}

/// The parameters' names as the field offers them for the word being typed.
fn suggestions(names: &[String], word: &str) -> Vec<egui::Suggestion> {
    if word.is_empty() || word.starts_with(|c: char| c.is_ascii_digit()) {
        return Vec::new();
    }
    let lower = word.to_lowercase();
    names.iter().filter(|n| n.to_lowercase().starts_with(&lower) && n.as_str() != word).map(|n| egui::Suggestion::new(n.clone())).collect()
}

/// A single-line field over a field memory, offering the parameters' names as a word is typed:
/// what it was left by this frame.
#[allow(clippy::too_many_arguments)]
fn text_field(ui: &mut egui::Ui, id: egui::Id, mem: &mut FieldMemory, placeholder: &str, tooltip: Option<String>, mono: bool, color: Option<Color32>, names: &[String], width: f32, focus: bool, tail: f32) -> (Left, egui::Response) {
    let mut edit = egui::TextEdit::singleline(&mut mem.text).desired_width(width).hint_text(placeholder.to_string());
    if tail > 0.0 {
        // Room at the right, inside the field, for the unit's word or what the expression
        // comes to.
        edit = edit.margin(egui::Margin { left: 4, right: tail.min(120.0) as i8, top: 2, bottom: 2 });
    }
    if mono {
        edit = edit.font(egui::TextStyle::Monospace);
    }
    if let Some(c) = color {
        edit = edit.text_color(c);
    }
    let out = egui::CompletionPopup::new(id).word_boundary(|c| !(c.is_alphanumeric() || c == '_')).show(ui, edit, |q| suggestions(names, q.word));
    let mut response = out.text_edit.response.response;
    if focus {
        response.request_focus();
        let mut state = out.text_edit.state;
        state.cursor.set_char_range(Some(CCursorRange::select_all_str(&mem.text)));
        state.store(ui.ctx(), response.id);
    }
    if let Some(t) = tooltip {
        response = response.on_hover_text(t);
    }
    let focused = response.has_focus() || focus;
    let left = if mem.focused && !focused {
        let (enter, escape) = ui.input(|i| (i.key_pressed(egui::Key::Enter), i.key_pressed(egui::Key::Escape)));
        if escape {
            Left::Escape
        } else if enter {
            Left::Enter
        } else {
            Left::Blur
        }
    } else {
        Left::No
    };
    mem.focused = focused;
    (left, response)
}

/// The field memory of `id`, seeded with `shown`: given the text to show again when it is not
/// being typed in and what it shows changed, and started afresh when it was not drawn the pass
/// before (a box closed and opened again).
fn field_memory(ui: &egui::Ui, id: egui::Id, shown: &str) -> FieldMemory {
    let pass = ui.ctx().cumulative_pass_nr();
    let mut mem = ui.data(|d| d.get_temp::<FieldMemory>(id)).filter(|m| m.pass + 1 >= pass).unwrap_or_else(|| FieldMemory { text: shown.to_string(), seeded: shown.to_string(), ..Default::default() });
    let focused = ui.memory(|m| m.has_focus(id));
    if !focused && mem.seeded != shown {
        mem.text = shown.to_string();
        mem.seeded = shown.to_string();
    }
    mem.pass = pass;
    mem
}

/// Keep a key from going on to the workspace (Escape closing the box, Enter creating it).
fn swallow(ui: &egui::Ui, key: egui::Key) {
    ui.ctx().input_mut(|i| i.consume_key(egui::Modifiers::NONE, key));
}

/// A text field committed on blur or Enter, Escape putting the old text back (ScanRuler's
/// `TextCommit`): the trimmed text, handed on as it stands, changed or not.
#[allow(clippy::too_many_arguments)]
fn text_commit(ui: &mut egui::Ui, id: egui::Id, shown: &str, placeholder: &str, tooltip: &str, mono: bool, names: &[String], width: f32, focus: bool) -> Option<String> {
    let mut mem = field_memory(ui, id, shown);
    let (left, _) = text_field(ui, id, &mut mem, placeholder, Some(tooltip.to_string()), mono, None, names, width, focus, 0.0);
    let out = match left {
        Left::No => None,
        Left::Escape => {
            swallow(ui, egui::Key::Escape);
            mem.text = shown.to_string();
            None
        }
        Left::Enter | Left::Blur => {
            if left == Left::Enter {
                swallow(ui, egui::Key::Enter);
            }
            Some(mem.text.trim().to_string())
        }
    };
    ui.data_mut(|d| d.insert_temp(id, mem));
    out
}

// ---- the number field ----

/// A number field that takes an expression (ScanRuler's `ExprInput`): what it shows, and how.
pub(crate) struct ExprInput<'a> {
    /// The field's name: its identity between frames, and what a script types into.
    pub id: &'a str,
    /// The number it holds (a length in millimetres).
    pub value: f64,
    /// The expression behind the number, or none.
    pub expr: Option<&'a str>,
    pub kind: ParameterKind,
    /// The word after the number while no expression is set: "°", "turns"; a length writes
    /// the display unit's word whatever is given here, but for `Some("")`, which leaves the
    /// word to the label.
    pub unit: Option<&'a str>,
    /// The decimals the number is shown to.
    pub places: usize,
    /// While nothing is typed into it, Enter and Escape do what they do in the box — create
    /// it, close it — as if the field had not taken them.
    pub pass_keys: bool,
}

/// A number field that takes an expression — `25.4/12`, `height + 5`, `sin(30) * r` — read
/// against the parameters. A plain number is handed on as one; anything else is handed on with
/// the expression beside it, and the field keeps showing the expression, the value it comes to
/// written where the unit was. An expression that cannot be read stays in the field with why
/// under it until it is put right or Escape puts the old text back. The parameters' names are
/// offered as the field is typed into. Commits on blur or Enter. The number and the expression
/// typed (none for a plain number). For a number that stands in a row of its own rather than
/// as a labelled field of a box (a sketch's dimension, a plane's offset): see `num_field`.
#[allow(dead_code)]
pub(crate) fn expr_input(ui: &mut egui::Ui, ws: &ReverseWorkspace, f: &ExprInput) -> Option<(f64, Option<String>)> {
    let mut st = ExprState::begin(ui, ws, f);
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| st.row(ui, f));
    st.end(ui)
}

/// A number field being drawn: what it shows, what it keeps, and what was committed.
struct ExprState {
    id: egui::Id,
    mem: FieldMemory,
    shown: String,
    word: String,
    result: String,
    units: DisplayUnits,
    table: Rc<ParameterTable>,
    names: Vec<String>,
    committed: Option<(f64, Option<String>)>,
    /// Why the expression kept no longer evaluates, where it does not.
    stale: Option<String>,
}

impl ExprState {
    fn begin(ui: &egui::Ui, ws: &ReverseWorkspace, f: &ExprInput) -> Self {
        let units = ws.units();
        let id = egui::Id::NULL.with(("re_expr_field", f.id));
        let shown = f.expr.map_or_else(|| field_text(f.value, f.kind, f.places, units), str::to_string);
        let word = match f.unit {
            Some("") => String::new(),
            _ if f.kind == ParameterKind::Length => unit_word(units, 1),
            u => u.unwrap_or("").to_string(),
        };
        let mem = field_memory(ui, id, &shown);
        let table = ws.param_table();
        let stale = f.expr.and_then(|e| evaluate_for(e, f.kind, table.as_ref(), false).err()).map(|e| e.message);
        let names = ws.part.parameters.iter().map(|p| p.name.clone()).collect();
        let mut st = ExprState { id, mem, shown, word, result: format_value(f.value, f.kind, units), units, table, names, committed: None, stale };
        if let Some(typed) = ws.params.take_typed(f.id) {
            st.mem.text = typed;
            st.take(f);
        }
        st
    }

    /// What was typed, read: the number and the expression, or why it will not do. Whether it
    /// was taken.
    fn take(&mut self, f: &ExprInput) -> bool {
        let t = self.mem.text.trim().to_string();
        if t.is_empty() || t == self.shown {
            self.mem.error = None;
            if t.is_empty() {
                self.mem.text = self.shown.clone();
            }
            return true;
        }
        match read_field(&t, f.kind, self.table.as_ref(), self.units) {
            Err(e) => {
                self.mem.error = Some(e);
                false
            }
            Ok(r) => {
                self.mem.error = None;
                self.committed = Some((r.value, r.expr));
                // Shown again from what it comes to.
                self.mem.seeded.clear();
                true
            }
        }
    }

    /// The field and its tail, in a right-to-left row: the tail at the right — the unit's word,
    /// or what the expression comes to — and the field before it.
    fn row(&mut self, ui: &mut egui::Ui, f: &ExprInput) {
        ui.spacing_mut().item_spacing.x = 4.0;
        // The tail, inside the field at its right: the unit's word, or what the expression
        // comes to (at most 68 px of it, cut short with an ellipsis).
        let (tail, most) = match f.expr {
            Some(_) => (RichText::new(format!("= {}", self.result)).monospace().size(11.0).color(dim(ui)), 68.0),
            None => (RichText::new(self.word.clone()).italics().color(dim(ui)), 60.0),
        };
        let galley = egui::WidgetText::from(tail).into_galley(ui, Some(egui::TextWrapMode::Truncate), most, egui::TextStyle::Body);
        let tail_width = if galley.size().x > 0.0 { galley.size().x + 8.0 } else { 0.0 };
        let width = ui.available_width().clamp(60.0, 172.0);
        let tooltip = f.expr.map(|e| format!("{e} = {}", self.result));
        let color = f.expr.map(|_| accent(ui));
        let names = std::mem::take(&mut self.names);
        let (left, response) = text_field(ui, self.id, &mut self.mem, "", tooltip, false, color, &names, width, false, tail_width);
        self.names = names;
        if tail_width > 0.0 {
            let at = egui::pos2(response.rect.right() - tail_width + 3.0, response.rect.center().y - galley.size().y / 2.0);
            ui.painter().galley(at, galley, dim(ui));
        }
        // Nothing typed into it: the keys it has no use for go on to the box.
        let untouched = self.mem.text == self.shown && self.mem.error.is_none();
        let passed = f.pass_keys && untouched;
        match left {
            Left::No => {}
            Left::Escape => {
                if !passed {
                    // The field's own: the old text comes back, and the key goes no further.
                    swallow(ui, egui::Key::Escape);
                    self.mem.text = self.shown.clone();
                    self.mem.error = None;
                }
            }
            Left::Enter => {
                if !passed {
                    swallow(ui, egui::Key::Enter);
                    if !self.take(f) {
                        response.request_focus();
                        self.mem.focused = true;
                    }
                }
            }
            Left::Blur => {
                self.take(f);
            }
        }
    }

    /// Why the text will not do, under the field; the field kept for the next frame. What was
    /// committed.
    fn end(self, ui: &mut egui::Ui) -> Option<(f64, Option<String>)> {
        if let Some(e) = self.mem.error.as_ref().or(self.stale.as_ref()) {
            // A typed text refused — or the expression kept no longer evaluating (a project
            // from elsewhere): the number stands as it last came to, and why is said.
            ui.label(RichText::new(e).size(11.5).color(warn(ui)));
        }
        ui.data_mut(|d| d.insert_temp(self.id, self.mem));
        self.committed
    }
}

/// The chip's text for a nominal value: enough decimals to show a half; a length in inches to
/// the ten-thousandth that tells 6 mm from 6.5 mm (ScanRuler's `nominalText`).
pub(crate) fn nominal_text(value: f64, kind: NominalKind, units: DisplayUnits) -> String {
    let whole = value % 1.0 == 0.0;
    match kind {
        NominalKind::Angle => format!("{}°", to_fixed(value, 0)),
        NominalKind::Draft => format!("{}°", to_fixed(value, if whole { 0 } else { 1 })),
        NominalKind::Length | NominalKind::Radius => fmt_len(value, units, if units == DisplayUnits::Mm { usize::from(!whole) } else { 3 }, FmtOptions::default()),
    }
}

/// One labelled number of a box (ScanRuler's `Num`); see `editors::expr_field`, which every
/// feature box calls.
#[derive(Clone, Copy)]
pub(crate) struct Num<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub value: f64,
    pub slot: Option<&'a str>,
    pub kind: Option<ParameterKind>,
    pub unit: Option<&'a str>,
    pub nominal: Option<NominalKind>,
    pub hint: Option<&'a str>,
    pub places: usize,
    pub pass_keys: bool,
}

#[allow(dead_code)] // the feature boxes build their fields with it (editors::expr_field)
impl<'a> Num<'a> {
    /// A length field named `id` (its identity, and what a script types into), labelled, showing
    /// `value`.
    pub fn new(id: &'a str, label: &'a str, value: f64) -> Self {
        Self { id, label, value, slot: None, kind: None, unit: None, nominal: None, hint: None, places: 3, pass_keys: false }
    }

    /// The slot path the field's expression is kept under in the feature's expressions
    /// ("extent.to", "radius", "angleDeg"); without one the field takes an expression for
    /// its number but keeps none.
    pub fn slot(mut self, path: &'a str) -> Self {
        self.slot = Some(path);
        self
    }

    /// What the field wants, where the unit does not say it.
    #[allow(dead_code)]
    pub fn kind(mut self, kind: ParameterKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// The word after the number: "°" makes an angle field, any other a plain number.
    pub fn unit(mut self, unit: &'a str) -> Self {
        self.unit = Some(unit);
        self
    }

    /// The round number a dragged or measured value was probably meant as is offered beside
    /// the field, one click to take — not while an expression drives it.
    pub fn nominal(mut self, kind: NominalKind) -> Self {
        self.nominal = Some(kind);
        self
    }

    /// What the ⓘ after the label says.
    pub fn hint(mut self, hint: &'a str) -> Self {
        self.hint = Some(hint);
        self
    }

    /// The decimals shown (three unless given).
    #[allow(dead_code)]
    pub fn places(mut self, places: usize) -> Self {
        self.places = places;
        self
    }

    /// Enter and Escape on the untouched field do the box's own: create it, close it.
    pub fn pass_keys(mut self) -> Self {
        self.pass_keys = true;
        self
    }

    /// The kind of number the field wants: as given, or read off the unit — "°" an angle, no
    /// unit a length, any other unit a plain number.
    pub fn field_kind(&self) -> ParameterKind {
        self.kind.unwrap_or(if self.nominal == Some(NominalKind::Angle) || self.unit == Some("°") {
            ParameterKind::Angle
        } else if self.unit.is_none() {
            ParameterKind::Length
        } else {
            ParameterKind::Number
        })
    }
}

/// A labelled number of a box: the label (with its ⓘ), the nominal chip, and the expression
/// field. The number committed — typed, given by an expression, or taken off the chip — for
/// the box to write as it writes its own; the expression typed is kept in `exprs` under the
/// field's slot, and a plain number lets the slot's expression go.
pub(crate) fn num_field(ui: &mut egui::Ui, ws: &ReverseWorkspace, exprs: Option<&mut Exprs>, num: Num) -> Option<f64> {
    let expr: Option<String> = match (num.slot, exprs.as_deref()) {
        (Some(path), Some(e)) => e.get(path).map(str::to_string),
        _ => None,
    };
    let units = ws.units();
    let round = num.nominal.filter(|_| expr.is_none()).and_then(|k| nominal_of(num.value, k).map(|r| (r, k)));
    let field = ExprInput { id: num.id, value: num.value, expr: expr.as_deref(), kind: num.field_kind(), unit: num.unit, places: num.places, pass_keys: num.pass_keys };
    let mut st = ExprState::begin(ui, ws, &field);
    let mut out = None;
    ui.horizontal(|ui| {
        ui.label(RichText::new(num.label).color(dim(ui)));
        if let Some(h) = num.hint {
            info_dot(ui, num.id, num.label, &[h]);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            st.row(ui, &field);
            if let Some((r, k)) = round {
                if nominal_chip(ui, &format!("→ {}", nominal_text(r, k, units))) {
                    out = Some(r);
                }
            }
        });
    });
    if let Some((v, e)) = st.end(ui) {
        if let (Some(path), Some(exprs)) = (num.slot, exprs) {
            set_draft_expr(exprs, path, e.as_deref());
        }
        out = Some(v);
    }
    out
}

/// The nominal chip: a small round key in the accent, filled when the pointer is on it; never
/// takes Tab's focus. Whether it was clicked.
fn nominal_chip(ui: &mut egui::Ui, text: &str) -> bool {
    let id = ui.next_auto_id();
    let hovered = ui.ctx().read_response(id).is_some_and(|r| r.hovered());
    let a = accent(ui);
    let label = RichText::new(text).monospace().size(10.5).color(if hovered { Color32::WHITE } else { a });
    let chip = egui::Button::new(label).corner_radius(10.0).stroke(egui::Stroke::new(1.0, a)).fill(if hovered { a } else { Color32::TRANSPARENT });
    ui.add(chip).on_hover_text("The round number this was probably meant as — click to take it").clicked()
}

#[cfg(test)]
mod tests {
    //! The chip's text, ported from ScanRuler's `plugins/reverse/tests/reNominal.test.ts`
    //! ("nominalText"), and the field kinds `Num` reads off its unit (`featureFields.tsx`).
    use super::*;

    #[test]
    fn nominal_text_shows_enough_decimals_for_a_half() {
        assert_eq!(nominal_text(6.0, NominalKind::Length, DisplayUnits::Mm), "6 mm");
        assert_eq!(nominal_text(6.5, NominalKind::Length, DisplayUnits::Mm), "6.5 mm");
        assert_eq!(nominal_text(2.5, NominalKind::Radius, DisplayUnits::Mm), "2.5 mm");
        assert_eq!(nominal_text(45.0, NominalKind::Angle, DisplayUnits::Mm), "45°");
        assert_eq!(nominal_text(1.5, NominalKind::Draft, DisplayUnits::Mm), "1.5°");
        assert_eq!(nominal_text(2.0, NominalKind::Draft, DisplayUnits::Mm), "2°");
        // reCoreUnits.test.ts: in inches, a length's chip reads to the ten-thousandth.
        assert_eq!(nominal_text(6.0, NominalKind::Length, DisplayUnits::In), "0.2362 in");
        assert_eq!(nominal_text(2.5, NominalKind::Radius, DisplayUnits::In), "0.0984 in");
        assert_eq!(nominal_text(45.0, NominalKind::Angle, DisplayUnits::In), "45°");
        assert_eq!(nominal_text(1.5, NominalKind::Draft, DisplayUnits::In), "1.5°");
        assert_eq!(nominal_text(6.5, NominalKind::Radius, DisplayUnits::Mm), "6.5 mm");
    }

    /// One frame of a bare context with these events: what `f` returns, and whether a key
    /// pressed is still there for the workspace after it.
    fn frame<R>(ctx: &egui::Context, events: Vec<egui::Event>, mut f: impl FnMut(&mut egui::Ui) -> R) -> (R, bool, bool) {
        let mut out = None;
        let mut left = (false, false);
        let _ = ctx.run_ui(egui::RawInput { events, ..Default::default() }, |ui| {
            out = Some(f(ui));
            left = ui.input(|i| (i.key_pressed(egui::Key::Escape), i.key_pressed(egui::Key::Enter)));
        });
        match out {
            Some(r) => (r, left.0, left.1),
            None => panic!("the frame drew nothing"),
        }
    }
    fn key(key: egui::Key) -> Vec<egui::Event> {
        [true, false].map(|pressed| egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: Default::default() }).to_vec()
    }

    /// ScanRuler's `TextCommit`: Escape puts the old text back and goes no further; Enter
    /// commits what was typed and goes no further either.
    #[test]
    fn a_text_field_takes_escape_and_enter_for_itself() {
        let ctx = egui::Context::default();
        let id = egui::Id::NULL.with("test_field");
        let field = |focus: bool| move |ui: &mut egui::Ui| text_commit(ui, id, "abc", "", "tip", false, &[], 100.0, focus);
        frame(&ctx, Vec::new(), field(true));
        let typed = frame(&ctx, vec![egui::Event::Text("xyz".into())], field(false));
        assert_eq!(typed, (None, false, false));
        let (out, escape, _) = frame(&ctx, key(egui::Key::Escape), field(false));
        assert_eq!(out, None);
        assert!(!escape, "Escape went on to the workspace");
        assert_eq!(ctx.data(|d| d.get_temp::<FieldMemory>(id)).map(|m| m.text), Some("abc".to_string()));
        frame(&ctx, Vec::new(), field(true));
        frame(&ctx, vec![egui::Event::Text("width".into())], field(false));
        let (out, _, enter) = frame(&ctx, key(egui::Key::Enter), field(false));
        assert_eq!(out, Some("width".to_string()));
        assert!(!enter, "Enter went on to the workspace");
        // A field nobody is typing into lets the keys go by.
        let (_, escape, enter) = frame(&ctx, [key(egui::Key::Escape), key(egui::Key::Enter)].concat(), field(false));
        assert!(escape && enter);
    }

    #[test]
    fn a_num_field_reads_its_kind_off_its_unit() {
        assert_eq!(Num::new("a", "A", 1.0).field_kind(), ParameterKind::Length);
        assert_eq!(Num::new("a", "A", 1.0).unit("°").field_kind(), ParameterKind::Angle);
        assert_eq!(Num::new("a", "A", 1.0).nominal(NominalKind::Angle).field_kind(), ParameterKind::Angle);
        assert_eq!(Num::new("a", "A", 1.0).unit("turns").field_kind(), ParameterKind::Number);
        assert_eq!(Num::new("a", "A", 1.0).unit("turns").kind(ParameterKind::Integer).field_kind(), ParameterKind::Integer);
    }
}
