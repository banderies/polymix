//! What a workspace offers agents: LLM clients that read the running app over MCP (the
//! `polymix-mcp` crate serves them; the shell routes their calls here).
//!
//! Each workspace lists its tools once ([`crate::Workspace::agent_tools`]) and answers calls on
//! the UI thread between frames ([`crate::Workspace::agent_call`]). An answer that takes more
//! than a moment (statistics of a multi-million-triangle mesh) is [`AgentReply::Later`]: the
//! workspace snapshots what it needs (meshes are `Arc`s, so that is cheap) and the work runs on
//! a worker thread, so the UI never waits on an agent.
//!
//! The tools only read: none changes a scene, a selection or the view.

use serde_json::{json, Value};

/// One tool, as listed to agents.
#[derive(Clone, Debug)]
pub struct AgentTool {
    /// `<workspace>_<what>` (e.g. `mesh_object`), unique across the app.
    pub name: &'static str,
    pub description: &'static str,
    /// JSON Schema of the arguments (an object schema).
    pub input_schema: Value,
}

impl AgentTool {
    /// A tool whose arguments are `properties` (a JSON Schema `properties` object), all optional.
    pub fn new(name: &'static str, description: &'static str, properties: Value) -> Self {
        Self { name, description, input_schema: json!({"type": "object", "properties": properties, "additionalProperties": false}) }
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
