//! Agents read and work the shown panels through egui's accessibility tree (AccessKit), the way
//! a screen reader does: every widget with its role, label, value and state, and the actions a
//! screen reader takes on them (click, set a value, step a slider). That covers every panel and
//! dialog without code of their own; typed tools (`mesh_tool` settings) are steadier where they
//! exist.
//!
//! egui builds the tree only while AccessKit is on, so it is turned on when an agent asks and
//! off again after a quiet half minute. A plugin keeps the last frame's tree. Widget ids are
//! egui's, stable as long as the panel's layout is.

use egui::accesskit::{Action, ActionData, ActionRequest, Node, NodeId, Role, Toggled, TreeId, TreeUpdate};
use polymix_mcp::{Call, Reply};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Seconds AccessKit stays on after an agent last used it.
const KEEP_ON: f64 = 30.0;
/// Windows agents may read but never act in (agent access itself: no granting themselves
/// changes, no new tokens).
pub const PROTECTED_WINDOWS: [&str; 1] = ["Agent access"];

#[derive(Default)]
struct Captured {
    /// Frames finished since start.
    frame: u64,
    tree: Option<TreeUpdate>,
}

struct Capture(Arc<Mutex<Captured>>);

impl egui::Plugin for Capture {
    fn debug_name(&self) -> &'static str {
        "polymix agent ui"
    }

    fn output_hook(&mut self, _ctx: &egui::Context, output: &mut egui::FullOutput) {
        let mut c = self.0.lock().unwrap_or_else(|e| e.into_inner());
        c.frame += 1;
        c.tree = output.platform_output.accesskit_update.clone();
    }
}

/// A call waiting for a frame drawn after it was made.
struct Waiting {
    call: Call,
    /// Answer once this many frames have finished.
    after: u64,
    mask: Option<crate::agents::Mask>,
    done: Option<String>,
}

pub struct UiAccess {
    captured: Arc<Mutex<Captured>>,
    /// When AccessKit is turned off again.
    on_until: Option<std::time::Instant>,
    waiting: Vec<Waiting>,
    /// Actions for the next frame's input.
    actions: Vec<ActionRequest>,
}

impl UiAccess {
    pub fn new(ctx: &egui::Context) -> Self {
        let captured = Arc::new(Mutex::new(Captured::default()));
        ctx.add_plugin(Capture(captured.clone()));
        Self { captured, on_until: None, waiting: Vec::new(), actions: Vec::new() }
    }

    fn frame(&self) -> u64 {
        self.captured.lock().unwrap_or_else(|e| e.into_inner()).frame
    }

    fn keep_on(&mut self, ctx: &egui::Context) {
        if self.on_until.is_none() {
            ctx.enable_accesskit();
        }
        self.on_until = Some(std::time::Instant::now() + std::time::Duration::from_secs_f64(KEEP_ON));
        ctx.request_repaint();
    }

    /// `app_ui`: answered with the tree of a frame drawn after now.
    pub fn read(&mut self, call: Call, mask: Option<crate::agents::Mask>, ctx: &egui::Context) {
        self.keep_on(ctx);
        let after = self.frame() + 2;
        self.waiting.push(Waiting { call, after, mask, done: None });
    }

    /// `app_ui_action`: checked against the last tree, handed to egui with the next frame,
    /// answered once that frame is drawn.
    pub fn act(&mut self, call: Call, ctx: &egui::Context) {
        self.keep_on(ctx);
        let request = {
            let c = self.captured.lock().unwrap_or_else(|e| e.into_inner());
            match &c.tree {
                Some(tree) => action_request(tree, &call.args),
                None => Err("Read the panels with app_ui first.".into()),
            }
        };
        match request {
            Ok((req, what)) => {
                self.actions.push(req);
                let after = self.frame() + 2;
                self.waiting.push(Waiting { call, after, mask: None, done: Some(what) });
            }
            Err(e) => call.answer(Reply::Error(e)),
        }
    }

    /// Calls waiting for a drawn frame.
    pub fn is_waiting(&self) -> bool {
        !self.waiting.is_empty()
    }

    /// The window is not drawn: answer the waiting calls with `why`, and drop the actions so
    /// none lands later, after the agent was told it failed.
    pub fn fail_waiting(&mut self, why: &str) {
        self.actions.clear();
        for w in self.waiting.drain(..) {
            w.call.answer(Reply::Error(why.to_string()));
        }
    }

    /// The actions for this pass, handed to egui at its start (only a pass that is drawn
    /// gets them).
    pub fn take_actions(&mut self) -> Vec<egui::Event> {
        self.actions.drain(..).map(egui::Event::AccessKitActionRequest).collect()
    }

    /// Answer the calls whose frame has been drawn; turn AccessKit off after a quiet while.
    pub fn tick(&mut self, ctx: &egui::Context) {
        let (frame, tree) = {
            let c = self.captured.lock().unwrap_or_else(|e| e.into_inner());
            (c.frame, if self.waiting.iter().any(|w| w.done.is_none() && c.frame >= w.after) { c.tree.clone() } else { None })
        };
        let mut i = 0;
        while i < self.waiting.len() {
            if frame < self.waiting[i].after {
                i += 1;
                continue;
            }
            let w = self.waiting.remove(i);
            let reply = match w.done {
                Some(what) => Reply::Json(json!({"done": what, "next": "Read the panels again with app_ui to see the effect."})),
                None => match &tree {
                    Some(t) => {
                        let mut v = snapshot(t, &w.call.args);
                        if let Some(m) = &w.mask {
                            mask_value(&mut v, m.as_ref());
                        }
                        Reply::Json(v)
                    }
                    None => Reply::Error("The panels could not be read (no frame was drawn; is the window minimized?).".into()),
                },
            };
            w.call.answer(reply);
        }
        if !self.waiting.is_empty() || !self.actions.is_empty() {
            ctx.request_repaint();
        } else if let Some(until) = self.on_until {
            let now = std::time::Instant::now();
            if now > until {
                ctx.disable_accesskit();
                self.on_until = None;
                self.captured.lock().unwrap_or_else(|e| e.into_inner()).tree = None;
            } else {
                ctx.request_repaint_after(until - now);
            }
        }
    }
}

fn mask_value(v: &mut Value, mask: &(dyn Fn(&str) -> String + Send + Sync)) {
    match v {
        Value::String(s) => *s = mask(s),
        Value::Array(a) => a.iter_mut().for_each(|x| mask_value(x, mask)),
        Value::Object(o) => o.values_mut().for_each(|x| mask_value(x, mask)),
        _ => {}
    }
}

pub fn id_text(id: NodeId) -> String {
    format!("#{:x}", id.0)
}

fn parse_id(s: &str) -> Option<NodeId> {
    u64::from_str_radix(s.trim().trim_start_matches('#'), 16).ok().map(NodeId)
}

fn role_name(r: Role) -> String {
    match r {
        Role::Button | Role::DefaultButton => "button".into(),
        Role::CheckBox => "checkbox".into(),
        Role::RadioButton => "radio".into(),
        Role::Slider => "slider".into(),
        Role::SpinButton => "number".into(),
        Role::TextInput | Role::MultilineTextInput | Role::SearchInput => "textbox".into(),
        Role::ComboBox => "combobox".into(),
        Role::Label | Role::Paragraph => "text".into(),
        Role::Window | Role::Dialog => "window".into(),
        Role::Link => "link".into(),
        Role::Tab => "tab".into(),
        Role::MenuItem | Role::MenuItemCheckBox | Role::MenuItemRadio => "menuitem".into(),
        Role::Image => "image".into(),
        Role::ProgressIndicator => "progress".into(),
        other => {
            let s = format!("{other:?}");
            let mut out = String::new();
            for (k, c) in s.chars().enumerate() {
                if c.is_uppercase() && k > 0 {
                    out.push(' ');
                }
                out.extend(c.to_lowercase());
            }
            out
        }
    }
}

/// Containers that only group other widgets: not listed themselves.
fn is_plain_container(n: &Node) -> bool {
    matches!(n.role(), Role::GenericContainer | Role::Unknown | Role::ScrollView | Role::Group | Role::Pane | Role::Row | Role::Cell | Role::Grid | Role::Table) && n.label().is_none_or(str::is_empty)
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\n', " ").replace('"', "'"))
}

/// One widget's line: role, label, value and state, and its id (text has none: there is
/// nothing to do with it).
fn describe(id: NodeId, n: &Node) -> Option<String> {
    let label = n.label().map(str::trim).filter(|l| !l.is_empty());
    let value = n.value().map(str::trim).filter(|v| !v.is_empty() && Some(*v) != label);
    if matches!(n.role(), Role::Label | Role::Paragraph) {
        return label.or(value).map(|t| format!("text {}", quote(t)));
    }
    let mut line = role_name(n.role());
    if let Some(l) = label {
        line += &format!(" {}", quote(l));
    }
    if let Some(v) = n.numeric_value() {
        line += &format!(" = {}", polymix_workspace::agent::round(v, 6));
        if let (Some(lo), Some(hi)) = (n.min_numeric_value(), n.max_numeric_value()) {
            line += &format!(" ({}..{})", polymix_workspace::agent::round(lo, 6), polymix_workspace::agent::round(hi, 6));
        }
    } else if let Some(v) = value {
        line += &format!(" = {}", quote(v));
    }
    match n.toggled() {
        Some(Toggled::True) => line += " checked",
        Some(Toggled::False) => line += " unchecked",
        Some(Toggled::Mixed) => line += " mixed",
        None => {}
    }
    if n.is_selected() == Some(true) {
        line += " selected";
    }
    match n.is_expanded() {
        Some(true) => line += " expanded",
        Some(false) => line += " collapsed",
        None => {}
    }
    if n.is_disabled() {
        line += " disabled";
    }
    let actionable = [Action::Click, Action::SetValue, Action::Increment].iter().any(|a| n.supports_action(*a));
    if label.is_none() && value.is_none() && n.numeric_value().is_none() && !actionable {
        return None;
    }
    if actionable || n.numeric_value().is_some() {
        line += &format!(" {}", id_text(id));
    }
    Some(line)
}

/// `app_ui`'s answer: the shown widgets as indented lines.
pub fn snapshot(tree: &TreeUpdate, args: &Value) -> Value {
    let nodes: HashMap<NodeId, &Node> = tree.nodes.iter().map(|(id, n)| (*id, n)).collect();
    let root = tree.tree.as_ref().map(|t| t.root).or_else(|| tree.nodes.first().map(|(id, _)| *id));
    let contains = polymix_workspace::agent::arg_str(args, "contains").map(str::to_lowercase);
    let limit = polymix_workspace::agent::arg_usize(args, "limit", 400, 20, 3000);
    let mut lines: Vec<String> = Vec::new();
    let mut stack: Vec<(NodeId, usize)> = root.into_iter().map(|r| (r, 0)).collect();
    while let Some((id, depth)) = stack.pop() {
        let Some(n) = nodes.get(&id) else { continue };
        // A text's runs repeat it.
        if n.role() == Role::TextRun {
            continue;
        }
        let mut child_depth = depth;
        if !is_plain_container(n) {
            if let Some(line) = describe(id, n) {
                lines.push(format!("{}- {line}", "  ".repeat(depth)));
                child_depth = depth + 1;
            }
        }
        for c in n.children().iter().rev() {
            stack.push((*c, child_depth));
        }
    }
    let matching: Vec<String> = match &contains {
        Some(q) => lines.into_iter().filter(|l| l.to_lowercase().contains(q)).collect(),
        None => lines,
    };
    let truncated = matching.len() > limit;
    json!({
        "widgets": matching.iter().take(limit).cloned().collect::<Vec<_>>().join("\n"),
        "truncated": truncated,
        "how": "Each line: role \"label\" = value, state, #id. Act on a widget with app_ui_action and its #id.",
    })
}

/// The action an `app_ui_action` call asks for, checked against the last tree.
fn action_request(tree: &TreeUpdate, args: &Value) -> Result<(ActionRequest, String), String> {
    let id = polymix_workspace::agent::arg_str(args, "id").and_then(parse_id).ok_or("Give the widget's #id from app_ui.")?;
    let nodes: HashMap<NodeId, &Node> = tree.nodes.iter().map(|(id, n)| (*id, n)).collect();
    let n = nodes.get(&id).ok_or_else(|| format!("No widget {} on screen now: read app_ui again.", id_text(id)))?;
    if n.is_disabled() {
        return Err(format!("{} is disabled.", describe(id, n).unwrap_or_default()));
    }
    // Never inside the Agent access window.
    let parents: HashMap<NodeId, NodeId> = tree.nodes.iter().flat_map(|(p, n)| n.children().iter().map(move |c| (*c, *p))).collect();
    let mut up = Some(id);
    while let Some(a) = up {
        if let Some(node) = nodes.get(&a) {
            if matches!(node.role(), Role::Window | Role::Dialog) && node.label().is_some_and(|l| PROTECTED_WINDOWS.contains(&l)) {
                return Err("Agents cannot change agent access settings; only the user can.".into());
            }
        }
        up = parents.get(&a).copied();
    }
    let value = args.get("value");
    let (action, data) = match polymix_workspace::agent::arg_str(args, "action").unwrap_or("click") {
        "click" => (Action::Click, None),
        "set_value" => match value {
            Some(Value::Number(v)) => (Action::SetValue, Some(ActionData::NumericValue(v.as_f64().unwrap_or_default()))),
            Some(Value::String(s)) => (Action::SetValue, Some(ActionData::Value(s.clone().into_boxed_str()))),
            _ => return Err("set_value needs `value` (a number or text).".into()),
        },
        "increment" => (Action::Increment, None),
        "decrement" => (Action::Decrement, None),
        "focus" => (Action::Focus, None),
        "expand" => (Action::Expand, None),
        "collapse" => (Action::Collapse, None),
        "scroll_into_view" => (Action::ScrollIntoView, None),
        other => return Err(format!("Unknown action {other}")),
    };
    let what = format!("{action:?} on {}", describe(id, n).unwrap_or_else(|| id_text(id)));
    Ok((ActionRequest { action, target_tree: TreeId::ROOT, target_node: id, data }, what))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> TreeUpdate {
        let mut root = Node::new(Role::Window);
        root.set_children(vec![NodeId(2), NodeId(3), NodeId(4)]);
        let mut panel = Node::new(Role::Window);
        panel.set_label("Remesh");
        panel.set_children(vec![NodeId(5), NodeId(6), NodeId(8)]);
        let mut slider = Node::new(Role::Slider);
        slider.set_label("Density");
        slider.set_numeric_value(40.0);
        slider.set_min_numeric_value(-100.0);
        slider.set_max_numeric_value(100.0);
        slider.add_action(Action::SetValue);
        let mut accept = Node::new(Role::Button);
        accept.set_label("Accept");
        accept.add_action(Action::Click);
        let mut holes = Node::new(Role::Label);
        holes.set_value("Holes: 2");
        holes.add_action(Action::Click);
        holes.set_children(vec![NodeId(9)]);
        let mut run = Node::new(Role::TextRun);
        run.set_value("Holes: 2");
        let mut agents = Node::new(Role::Window);
        agents.set_label("Agent access");
        agents.set_children(vec![NodeId(7)]);
        let mut writes = Node::new(Role::CheckBox);
        writes.set_label("Let agents make changes");
        writes.set_toggled(Toggled::False);
        writes.add_action(Action::Click);
        let group = Node::new(Role::GenericContainer);
        TreeUpdate { nodes: vec![(NodeId(1), root), (NodeId(2), panel), (NodeId(3), agents), (NodeId(4), group), (NodeId(5), slider), (NodeId(6), accept), (NodeId(7), writes), (NodeId(8), holes), (NodeId(9), run)], tree: Some(egui::accesskit::TreeInfo::new(NodeId(1))), tree_id: TreeId::ROOT, focus: NodeId(1) }
    }

    #[test]
    fn lists_widgets_with_ids() {
        let s = snapshot(&tree(), &json!({}));
        let text = s["widgets"].as_str().unwrap();
        assert!(text.contains("- window \"Remesh\""), "{text}");
        assert!(text.contains("  - slider \"Density\" = 40 (-100..100) #5"), "{text}");
        assert!(text.contains("  - button \"Accept\" #6"), "{text}");
        assert!(text.contains("checkbox \"Let agents make changes\" unchecked #7"), "{text}");
        assert!(text.contains("text \"Holes: 2\"") && !text.contains("#8"), "{text}");
        let only = snapshot(&tree(), &json!({"contains": "accept"}));
        assert_eq!(only["widgets"].as_str().unwrap().lines().count(), 1);
    }

    #[test]
    fn actions_are_checked() {
        let (req, _) = action_request(&tree(), &json!({"id": "#5", "action": "set_value", "value": 25})).unwrap();
        assert_eq!(req.target_node, NodeId(5));
        assert!(matches!(req.data, Some(ActionData::NumericValue(v)) if v == 25.0));
        assert!(action_request(&tree(), &json!({"id": "#99"})).unwrap_err().contains("No widget"));
        assert!(action_request(&tree(), &json!({"id": "#7"})).unwrap_err().contains("only the user"));
        assert!(action_request(&tree(), &json!({"id": "#5", "action": "set_value"})).is_err());
    }
}
