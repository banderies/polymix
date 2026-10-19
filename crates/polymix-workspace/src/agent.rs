//! What a workspace offers agents: LLM clients that read the running app over MCP (the
//! `polymix-mcp` crate serves them; the shell routes their calls here).
//!
//! Each workspace lists its tools once ([`crate::Workspace::agent_tools`]) and answers calls on
//! the UI thread between frames ([`crate::Workspace::agent_call`]). An answer that takes more
//! than a moment (statistics of a multi-million-triangle mesh) is [`AgentReply::Later`]: the
//! workspace snapshots what it needs (meshes are `Arc`s, so that is cheap) and the work runs on
//! a worker thread, so the UI never waits on an agent.
//!
//! Most tools only read. Tools of kind [`ToolKind::Change`] or [`ToolKind::Destructive`] change
//! the session; the shell refuses them unless the user let agents make changes, and every
//! change they make is an undo step labelled "Agent: …".

use serde_json::{json, Value};
use std::borrow::Cow;

/// What a tool may do (in order: a tool that does several things is the most of them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolKind {
    /// Only reads.
    Read,
    /// Changes the session in ways that are easily undone (selection, view, opening a tool).
    Change,
    /// Changes meshes, the scene or files.
    Destructive,
}

/// One tool, as listed to agents.
#[derive(Clone, Debug)]
pub struct AgentTool {
    /// `<workspace>_<what>` (e.g. `mesh_object`), unique across the app.
    pub name: Cow<'static, str>,
    pub description: Cow<'static, str>,
    /// JSON Schema of the arguments (an object schema).
    pub input_schema: Value,
    pub kind: ToolKind,
}

impl AgentTool {
    /// A reading tool whose arguments are `properties` (a JSON Schema `properties` object),
    /// all optional.
    pub fn new(name: impl Into<Cow<'static, str>>, description: impl Into<Cow<'static, str>>, properties: Value) -> Self {
        Self { name: name.into(), description: description.into(), input_schema: json!({"type": "object", "properties": properties, "additionalProperties": false}), kind: ToolKind::Read }
    }

    pub fn kind(mut self, kind: ToolKind) -> Self {
        self.kind = kind;
        self
    }

    /// Arguments that must be given.
    pub fn required(mut self, names: &[&str]) -> Self {
        self.input_schema["required"] = json!(names);
        self
    }
}

/// A named operation agents can ask a tool or a workspace for, with its arguments described
/// as JSON Schema. The same description lists it to agents (in tool descriptions and state)
/// and checks the arguments of each call.
#[derive(Clone, Debug)]
pub struct Action {
    pub name: &'static str,
    pub about: &'static str,
    /// JSON Schema `properties` of the arguments.
    pub args: Value,
    pub required: Vec<&'static str>,
    /// What it may do (a workspace's `<workspace>_do` tool is the most of its actions').
    pub kind: ToolKind,
}

impl Action {
    pub fn new(name: &'static str, about: &'static str, args: Value) -> Self {
        Self { name, about, args, required: Vec::new(), kind: ToolKind::Change }
    }

    /// An action without arguments.
    pub fn plain(name: &'static str, about: &'static str) -> Self {
        Self::new(name, about, json!({}))
    }

    pub fn required(mut self, names: &[&'static str]) -> Self {
        self.required = names.to_vec();
        self
    }

    /// It changes data or files in ways an undo may not reach (removing, saving).
    pub fn destructive(mut self) -> Self {
        self.kind = ToolKind::Destructive;
        self
    }

    /// The JSON Schema of the arguments.
    pub fn schema(&self) -> Value {
        json!({"type": "object", "properties": self.args, "required": self.required, "additionalProperties": false})
    }

    /// The arguments, required ones first: (name, schema).
    fn arg_list(&self) -> Vec<(&String, &Value)> {
        let mut args: Vec<(&String, &Value)> = self.args.as_object().into_iter().flatten().collect();
        args.sort_by_key(|(k, _)| self.required.iter().position(|r| r == k).unwrap_or(usize::MAX));
        args
    }

    /// `name(arg: type, [optional: type])`, for listings.
    pub fn signature(&self) -> String {
        let args: Vec<String> = self
            .arg_list()
            .into_iter()
            .map(|(k, v)| {
                let t = type_name(v);
                if self.required.contains(&k.as_str()) { format!("{k}: {t}") } else { format!("[{k}: {t}]") }
            })
            .collect();
        format!("{}({})", self.name, args.join(", "))
    }

    /// The action as agents read it: name, purpose, and its arguments' schema.
    pub fn describe(&self) -> Value {
        json!({"name": self.name, "about": self.about, "signature": self.signature(), "args": self.args, "required": self.required})
    }

    /// Check a call's arguments against the schema (names, presence and types).
    pub fn check(&self, args: &Value) -> Result<(), String> {
        let given = match args {
            Value::Null => serde_json::Map::new(),
            Value::Object(o) => o.clone(),
            _ => return Err(format!("{}: the arguments must be an object", self.signature())),
        };
        let props = self.args.as_object().cloned().unwrap_or_default();
        for k in given.keys() {
            if !props.contains_key(k) {
                return Err(format!("{} has no argument `{k}` (it takes {})", self.name, self.signature()));
            }
        }
        for r in &self.required {
            if given.get(*r).is_none_or(Value::is_null) {
                return Err(format!("{} needs `{r}`: {}", self.name, self.signature()));
            }
        }
        for (k, v) in &given {
            if !v.is_null() && !fits(&props[k], v) {
                return Err(format!("{}: `{k}` must be {}", self.name, type_name(&props[k])));
            }
        }
        Ok(())
    }
}

/// A schema's type, shortly ("number", "[x, y, z]", "\"a\" | \"b\"", "integer | string").
fn type_name(schema: &Value) -> String {
    if let Some(e) = schema.get("enum").and_then(Value::as_array) {
        return e.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" | ");
    }
    let is_point = |s: &Value| s.get("minItems") == Some(&json!(3)) && s.get("maxItems") == Some(&json!(3));
    match schema.get("type") {
        Some(Value::String(t)) if t == "array" && is_point(schema) => "[x, y, z]".into(),
        Some(Value::String(t)) if t == "array" && schema.get("items").is_some_and(is_point) => "[[x, y, z], …]".into(),
        Some(Value::String(t)) => t.clone(),
        Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" | "),
        _ => "value".into(),
    }
}

/// The agent tool `<workspace>_do`, made from the workspace's actions: `action` names one and
/// `args` holds its arguments. Its description lists every action's signature, purpose and
/// arguments, so an action a workspace adds reaches agents without more code.
pub fn do_tool(workspace: &str, about: &str, actions: &[Action]) -> AgentTool {
    // Each argument's meaning, once: an argument as an earlier action described it is not repeated.
    let mut described = std::collections::HashSet::new();
    let list: String = actions
        .iter()
        .map(|a| {
            let args: String = a
                .arg_list()
                .into_iter()
                .filter_map(|(k, v)| {
                    let about = v.get("description")?.as_str()?;
                    described.insert((k.clone(), about.to_string())).then(|| format!("\n    {k}: {about}"))
                })
                .collect();
            format!("\n- {} — {}{args}", a.signature(), a.about)
        })
        .collect();
    let names: Vec<&str> = actions.iter().map(|a| a.name).collect();
    let description = format!("{about}\n\nActions (`action` names one, `args` holds its arguments; those in [brackets] are optional, and an argument means the same wherever it is described first):{list}");
    let kind = actions.iter().map(|a| a.kind).max().unwrap_or(ToolKind::Change);
    AgentTool::new(format!("{workspace}_do"), description, json!({"action": {"type": "string", "enum": names, "description": "The action"}, "args": {"type": "object", "description": "Its arguments, by name (see the list above)"}}))
        .required(&["action"])
        .kind(kind)
}

/// The action a `<workspace>_do` call names, and its arguments, checked.
pub fn find_action<'a>(actions: &'a [Action], call: &Value) -> Result<(&'a Action, Value), String> {
    let name = call.get("action").and_then(Value::as_str).ok_or("`action` is needed")?;
    let a = actions.iter().find(|a| a.name == name).ok_or_else(|| format!("no action {name}; the actions are {}", actions.iter().map(|a| a.name).collect::<Vec<_>>().join(", ")))?;
    let args = call.get("args").cloned().unwrap_or(Value::Null);
    a.check(&args)?;
    Ok((a, args))
}

/// Whether `v` is of the schema's type (shallow: arrays' items are not checked).
fn fits(schema: &Value, v: &Value) -> bool {
    if let Some(e) = schema.get("enum").and_then(Value::as_array) {
        return e.contains(v);
    }
    let ok = |t: &str| match t {
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64(),
        "boolean" => v.is_boolean(),
        "string" => v.is_string(),
        "object" => v.is_object(),
        "array" => v.as_array().is_some_and(|a| {
            let n = a.len() as u64;
            schema.get("minItems").and_then(Value::as_u64).is_none_or(|m| n >= m) && schema.get("maxItems").and_then(Value::as_u64).is_none_or(|m| n <= m)
        }),
        _ => true,
    };
    match schema.get("type") {
        Some(Value::String(t)) => ok(t),
        Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).any(ok),
        _ => true,
    }
}

/// Property schemas for action arguments.
pub mod arg {
    use serde_json::{json, Value};

    pub fn number(about: &str) -> Value {
        json!({"type": "number", "description": about})
    }
    pub fn integer(about: &str) -> Value {
        json!({"type": "integer", "description": about})
    }
    pub fn boolean(about: &str) -> Value {
        json!({"type": "boolean", "description": about})
    }
    pub fn text(about: &str) -> Value {
        json!({"type": "string", "description": about})
    }
    pub fn choice(values: &[&str], about: &str) -> Value {
        json!({"type": "string", "enum": values, "description": about})
    }
    /// A point (or vector) in mm: [x, y, z].
    pub fn point(about: &str) -> Value {
        json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": about})
    }
    /// Points: [[x, y, z], …].
    pub fn points(about: &str) -> Value {
        json!({"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "minItems": 1, "description": about})
    }
}

/// An action's arguments, read with errors that name the argument.
pub struct Args<'a>(pub &'a Value);

impl Args<'_> {
    fn get(&self, k: &str) -> Option<&Value> {
        self.0.get(k).filter(|v| !v.is_null())
    }
    pub fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    pub fn f64(&self, k: &str) -> Result<f64, String> {
        self.opt_f64(k).ok_or_else(|| format!("`{k}` (a number) is needed"))
    }
    pub fn opt_f64(&self, k: &str) -> Option<f64> {
        self.get(k).and_then(Value::as_f64)
    }
    pub fn usize(&self, k: &str) -> Result<usize, String> {
        self.opt_usize(k).ok_or_else(|| format!("`{k}` (a whole number) is needed"))
    }
    pub fn opt_usize(&self, k: &str) -> Option<usize> {
        self.get(k).and_then(Value::as_u64).map(|v| v as usize)
    }
    pub fn opt_bool(&self, k: &str) -> Option<bool> {
        self.get(k).and_then(Value::as_bool)
    }
    pub fn str(&self, k: &str) -> Result<&str, String> {
        self.opt_str(k).ok_or_else(|| format!("`{k}` (text) is needed"))
    }
    pub fn opt_str(&self, k: &str) -> Option<&str> {
        self.get(k).and_then(Value::as_str)
    }
    pub fn point(&self, k: &str) -> Result<[f64; 3], String> {
        self.opt_point(k).ok_or_else(|| format!("`{k}` must be a point [x, y, z] in mm"))
    }
    pub fn opt_point(&self, k: &str) -> Option<[f64; 3]> {
        as_point(self.get(k)?)
    }
    pub fn points(&self, k: &str) -> Result<Vec<[f64; 3]>, String> {
        self.get(k).and_then(Value::as_array).and_then(|a| a.iter().map(as_point).collect()).ok_or_else(|| format!("`{k}` must be a list of points [[x, y, z], …]"))
    }
}

fn as_point(v: &Value) -> Option<[f64; 3]> {
    match v.as_array()?.iter().map(Value::as_f64).collect::<Option<Vec<f64>>>()?.as_slice() {
        [x, y, z] => Some([*x, *y, *z]),
        _ => None,
    }
}

/// Data for the agent (a JSON object), or what went wrong in words it can act on.
pub type AgentResult = Result<Value, String>;

/// A workspace's answer to a call.
pub enum AgentReply {
    Now(AgentResult),
    /// Computed on a worker thread from data the workspace snapshotted.
    Later(Box<dyn FnOnce() -> AgentResult + Send>),
}

impl AgentReply {
    pub fn later(f: impl FnOnce() -> AgentResult + Send + 'static) -> Self {
        Self::Later(Box::new(f))
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self::Now(Err(msg.into()))
    }
}

/// An optional integer argument, clamped to `lo..=hi`.
pub fn arg_usize(args: &Value, key: &str, default: usize, lo: usize, hi: usize) -> usize {
    args.get(key).and_then(Value::as_u64).map_or(default, |v| v as usize).clamp(lo, hi)
}

/// An optional string argument.
pub fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// `[x, y, z]` with the numbers rounded to `digits` decimals (agents read every digit).
pub fn vec3(v: [f64; 3], digits: i32) -> Value {
    json!(v.map(|x| round(x, digits)))
}

pub fn round(x: f64, digits: i32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let k = 10f64.powi(digits);
    // + 0.0 turns -0.0 (an empty sum) into 0.
    (x * k).round() / k + 0.0
}

/// A date reduced to its year ("2024-01-02 10:20" → "2024"): exact dates of care identify
/// patients, and agents send what they read to their provider.
pub fn year_only(date: &str) -> String {
    let digits: String = date.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.len() >= 4 { digits[..4].to_string() } else if date.trim().is_empty() { String::new() } else { "•••".into() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(year_only("2024-01-02 10:20"), "2024");
        assert_eq!(year_only("20240102"), "2024");
        assert_eq!(year_only(""), "");
        assert_eq!(year_only("Jan 2"), "•••");
        assert_eq!(round(1.23456, 2), 1.23);
        assert!(round(-0.0, 2).is_sign_positive());
        let args = json!({"n": 500, "s": "abc", "e": ""});
        assert_eq!(arg_usize(&args, "n", 10, 1, 100), 100);
        assert_eq!(arg_usize(&args, "missing", 10, 1, 100), 10);
        assert_eq!(arg_str(&args, "s"), Some("abc"));
        assert_eq!(arg_str(&args, "e"), None);
    }
}

#[cfg(test)]
mod action_tests {
    use super::*;

    fn set_end() -> Action {
        Action::new("set_end", "Put an end at the wall point nearest `at`", json!({"at": arg::point("Point in mm"), "end": arg::choice(&["1", "2"], "Which end")})).required(&["at"])
    }

    #[test]
    fn signatures_and_checks() {
        let a = set_end();
        assert_eq!(a.signature(), "set_end(at: [x, y, z], [end: \"1\" | \"2\"])");
        assert!(a.check(&json!({"at": [1, 2, 3]})).is_ok());
        assert!(a.check(&json!({"at": [1, 2, 3], "end": "2"})).is_ok());
        assert!(a.check(&json!({})).unwrap_err().contains("needs `at`"));
        assert!(a.check(&json!({"at": [1, 2]})).unwrap_err().contains("[x, y, z]"));
        assert!(a.check(&json!({"at": [1, 2, 3], "end": "3"})).is_err());
        assert!(a.check(&json!({"at": [1, 2, 3], "colour": "red"})).unwrap_err().contains("no argument `colour`"));
        assert!(Action::plain("clear", "Remove both ends").check(&Value::Null).is_ok());
        let args = json!({"at": [1.0, 2.0, 3.5], "n": 4});
        let r = Args(&args);
        assert_eq!(r.point("at").unwrap(), [1.0, 2.0, 3.5]);
        assert_eq!(r.usize("n").unwrap(), 4);
        assert!(r.f64("missing").unwrap_err().contains("`missing`"));
    }

    #[test]
    fn workspace_do_tools_list_and_route_actions() {
        let actions = [set_end(), Action::plain("clear", "Remove both ends"), Action::new("save", "Save", json!({"path": arg::text("Where"), "path_ends": arg::points("Ends")})).destructive()];
        let t = do_tool("vessel", "Work the vessel.", &actions);
        assert_eq!(t.name, "vessel_do");
        assert_eq!(t.kind, ToolKind::Destructive);
        assert!(t.description.contains("- set_end(at: [x, y, z], [end: \"1\" | \"2\"]) — Put an end"), "{}", t.description);
        assert!(t.description.contains("save([path: string], [path_ends: [[x, y, z], …]])"), "{}", t.description);
        assert!(t.description.contains("\n    at: Point in mm\n    end: Which end\n"), "{}", t.description);
        // Required arguments first; an argument described once is not described again.
        let b = Action::new("b", "", json!({"a": arg::text("A"), "at": arg::point("Point in mm"), "z": arg::text("Z")})).required(&["z"]);
        assert_eq!(b.signature(), "b(z: string, [a: string], [at: [x, y, z]])");
        let once = do_tool("v", "", &[set_end(), b]).description;
        assert_eq!(once.matches("at: Point in mm").count(), 1, "{once}");
        assert_eq!(t.input_schema["properties"]["action"]["enum"], json!(["set_end", "clear", "save"]));
        let (a, args) = find_action(&actions, &json!({"action": "set_end", "args": {"at": [0, 0, 1]}})).unwrap();
        assert_eq!((a.name, args), ("set_end", json!({"at": [0, 0, 1]})));
        assert_eq!(find_action(&actions, &json!({"action": "clear"})).unwrap().0.name, "clear");
        assert!(find_action(&actions, &json!({"action": "nope"})).unwrap_err().contains("set_end, clear, save"));
        assert!(find_action(&actions, &json!({"action": "set_end", "args": {}})).unwrap_err().contains("needs `at`"));
        let pts = json!({"p": [[1, 2, 3], [4, 5, 6]], "bad": [[1, 2]]});
        assert_eq!(Args(&pts).points("p").unwrap(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert!(Args(&pts).points("bad").is_err());
        assert_eq!(do_tool("x", "", &[Action::plain("a", "")]).kind, ToolKind::Change);
    }
}
