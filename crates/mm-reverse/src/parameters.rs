// SPDX-License-Identifier: AGPL-3.0-only
//! The parameters of a model and the numbers that read them. Ported from ScanRuler
//! (`plugins/reverse/core/parameters.ts`, with the parameter list's edits from
//! `plugins/reverse/state/reStore.ts` and `followParameters` from
//! `plugins/reverse/state/reFeatureHelpers.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! A parameter is a named expression of a kind — a length, an angle, a whole number, a plain
//! number — in one list for the whole model, outside the feature list and the rollback bar.
//! A feature's numbers stay numbers, holding the value last evaluated; beside them the
//! feature keeps the expressions that drive them, keyed by *slot* — the field as the box
//! shows it (an extrude's To, a hole's Diameter), which is not always the field as the
//! feature stores it (a radius, a draft with the opposite sign). A sketch dimension keeps its
//! expression beside its value the same way. When a parameter changes, every expression is
//! evaluated again, the numbers written, every sketch with a changed dimension solved again,
//! and the part rebuilt from nothing but numbers — the kernel never sees an expression.
//!
//! The slots are generic here: a [`Slot`] reads and writes one number of any feature type
//! through closures, a type that keeps expressions says which slots it offers
//! ([`Slotted`]), and [`apply_exprs`] evaluates a slot-path → expression map into them.
//! `param_slots` holds ScanRuler's own table of slots for every feature kind, on its feature
//! fields, and the same mapping onto Polymix's `model::FeatureKind`.
//!
//! "The same object back when nothing changes" is a borrowed [`Cow`] here, and "changed" is
//! value inequality rather than JavaScript object identity: a number written over with the
//! value it already held changes nothing.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use glam::DVec2;
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::datums::{datum_traces, DatumTrace};
use crate::expression::{evaluate_for, is_reserved, references, rename_in, valid_name, ReadingDepth, Scope, ScopeEntry, Value};
use crate::frame::SectionFrame;
use crate::js_number::js_string;
use crate::model::FeatureId;
use crate::sketch::{Rule, Sketch};
use crate::solver::{solve_sketch, SolveOptions};

pub use crate::expression::ParameterKind;

pub type ParameterId = u32;

/// A named expression of a kind, in the model's one list of parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub id: ParameterId,
    /// The identifier expressions use — see `expression::valid_name`.
    pub name: String,
    pub kind: ParameterKind,
    /// As typed: "25", "25.4/12", "threadlength+10".
    pub expression: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

/// A kind as the parameter list offers it: its label and the unit its value is answered in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterKindInfo {
    pub kind: ParameterKind,
    pub label: &'static str,
    pub unit: &'static str,
}

pub const PARAMETER_KINDS: [ParameterKindInfo; 4] = [
    ParameterKindInfo { kind: ParameterKind::Length, label: "Length", unit: "mm" },
    ParameterKindInfo { kind: ParameterKind::Angle, label: "Angle", unit: "°" },
    ParameterKindInfo { kind: ParameterKind::Integer, label: "Integer", unit: "" },
    ParameterKindInfo { kind: ParameterKind::Number, label: "Number", unit: "" },
];

// ---- the expressions a feature keeps ------------------------------------------------

/// The expressions a feature or a draft keeps beside its numbers, by slot path. Kept in the
/// order they were first set, as a JavaScript object keeps its keys — the order they are
/// evaluated in — and compared as a map, whatever the order.
#[derive(Debug, Clone, Default)]
pub struct ExprMap(Vec<(String, String)>);

impl ExprMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, path: &str) -> Option<&str> {
        self.0.iter().find(|(k, _)| k == path).map(|(_, e)| e.as_str())
    }

    pub fn contains(&self, path: &str) -> bool {
        self.0.iter().any(|(k, _)| k == path)
    }

    /// Set the expression of a slot: in its place when it has one already, at the end when
    /// not.
    pub fn insert(&mut self, path: impl Into<String>, expr: impl Into<String>) {
        let (path, expr) = (path.into(), expr.into());
        match self.0.iter_mut().find(|(k, _)| *k == path) {
            Some(slot) => slot.1 = expr,
            None => self.0.push((path, expr)),
        }
    }

    pub fn remove(&mut self, path: &str) -> Option<String> {
        let at = self.0.iter().position(|(k, _)| k == path)?;
        Some(self.0.remove(at).1)
    }

    /// The slots and their expressions, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, e)| (k.as_str(), e.as_str()))
    }

    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }

    pub fn exprs(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(_, e)| e.as_str())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl PartialEq for ExprMap {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(k, e)| other.get(k) == Some(e))
    }
}

impl<K: Into<String>, V: Into<String>> FromIterator<(K, V)> for ExprMap {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut out = ExprMap::new();
        out.extend(iter);
        out
    }
}

/// Entries set in turn, as `{ ...a, ...b }` spreads `b` over `a`: a path already there takes
/// the new value in its place, a new one goes at the end.
impl<K: Into<String>, V: Into<String>> Extend<(K, V)> for ExprMap {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl IntoIterator for ExprMap {
    type Item = (String, String);
    type IntoIter = std::vec::IntoIter<(String, String)>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Serialize for ExprMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (k, e) in self.iter() {
            map.serialize_entry(k, e)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for ExprMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries;
        impl<'de> Visitor<'de> for Entries {
            type Value = ExprMap;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a map of slot paths to expressions")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<ExprMap, A::Error> {
                let mut out = ExprMap::new();
                while let Some((k, e)) = access.next_entry::<String, String>()? {
                    out.insert(k, e);
                }
                Ok(out)
            }
        }
        deserializer.deserialize_map(Entries)
    }
}

/// Whether text is a plain number with an optional sign: `^[+-]?\d*\.?\d+(e[+-]?\d+)?$`,
/// the exponent's `e` in either case.
fn signed_plain_number(t: &str) -> bool {
    let c: Vec<char> = t.chars().collect();
    let digit = |i: usize| c.get(i).is_some_and(|ch| ch.is_ascii_digit());
    let mut i = 0;
    if matches!(c.first(), Some('+') | Some('-')) {
        i = 1;
    }
    let start = i;
    while digit(i) {
        i += 1;
    }
    // `\d*\.?\d+`: digits, then a point and at least one digit; or digits alone.
    if c.get(i) == Some(&'.') {
        i += 1;
        if !digit(i) {
            return false;
        }
        while digit(i) {
            i += 1;
        }
    } else if i == start {
        return false;
    }
    if matches!(c.get(i), Some('e') | Some('E')) {
        i += 1;
        if matches!(c.get(i), Some('+') | Some('-')) {
            i += 1;
        }
        if !digit(i) {
            return false;
        }
        while digit(i) {
            i += 1;
        }
    }
    i == c.len()
}

/// An expression of the opposite sign: "-(e)", or e again for "-(e)".
pub fn negated_expr(e: &str) -> String {
    let t = e.trim();
    if signed_plain_number(t) {
        if let Ok(n) = t.parse::<f64>() {
            return js_string(-n);
        }
    }
    if let Some(inner) = t.strip_prefix('-').map(str::trim_start).and_then(|r| r.strip_prefix('(')).and_then(|r| r.strip_suffix(')')) {
        // Only when the outer brackets pair with each other: not "-(a) * (b)".
        let mut depth = 0i64;
        let mut pairs = true;
        for ch in inner.chars() {
            if ch == '(' {
                depth += 1;
            } else if ch == ')' {
                depth -= 1;
                if depth < 0 {
                    pairs = false;
                }
            }
        }
        if pairs && depth == 0 {
            return inner.trim().to_string();
        }
    }
    format!("-({t})")
}

/// An extrude keeps its lean as `draftDeg` (positive leans in) and the expression on the
/// panel's taper, the other way round. Commands take and give it as `draftDeg` too, so the
/// sign never has to be turned by hand: the expressions a command gave, with an extrude's
/// `draftDeg` turned into its `taper`.
pub fn exprs_from_command(kind: Option<&str>, given: serde_json::Map<String, serde_json::Value>) -> serde_json::Map<String, serde_json::Value> {
    if kind != Some("extrude") || !given.contains_key("draftDeg") {
        return given;
    }
    let mut rest = given;
    let draft = rest.remove("draftDeg").unwrap_or(serde_json::Value::Null);
    let taper = match &draft {
        serde_json::Value::String(s) if !s.trim().is_empty() => serde_json::Value::String(negated_expr(s)),
        _ => draft,
    };
    rest.insert("taper".to_string(), taper);
    rest
}

/// The expressions as a command answers them: an extrude's taper as `draftDeg`.
pub fn exprs_for_command(kind: Option<&str>, exprs: &ExprMap) -> ExprMap {
    let Some(taper) = exprs.get("taper").filter(|_| kind == Some("extrude")) else {
        return exprs.clone();
    };
    let mut rest: ExprMap = exprs.iter().filter(|(k, _)| *k != "taper").collect();
    rest.insert("draftDeg", negated_expr(taper));
    rest
}

// ---- the table ------------------------------------------------------------------

/// A parameter as the table reads it: its value, or why it has none.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterReading {
    pub id: ParameterId,
    pub name: String,
    pub kind: ParameterKind,
    pub value: Option<f64>,
    pub error: Option<String>,
}

/// The parameters evaluated: each one's reading, and — through [`Scope`] — the parameters as
/// a scope for the features' expressions, where a parameter with no value answers with why.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParameterTable {
    pub readings: Vec<ParameterReading>,
    /// The first reading of each name.
    by_name: HashMap<String, usize>,
}

impl ParameterTable {
    /// What a name stands for in the features' expressions.
    pub fn scope(&self, name: &str) -> Option<ScopeEntry> {
        // The first reading of the name: by the index kept, or — `readings` being anyone's to
        // change — found again when the index no longer points at it.
        let kept = self.by_name.get(name).and_then(|&i| self.readings.get(i)).filter(|r| r.name == name);
        let r = match kept {
            Some(r) => r,
            None => self.readings.iter().find(|r| r.name == name)?,
        };
        Some(match r.value {
            None => ScopeEntry::Error(format!("“{name}” has no value")),
            Some(value) => ScopeEntry::Value(Value { value, dim: r.kind.dim() }),
        })
    }

    /// The reading of the parameter with this id.
    pub fn reading(&self, id: ParameterId) -> Option<&ParameterReading> {
        self.readings.iter().find(|r| r.id == id)
    }
}

impl Scope for ParameterTable {
    fn lookup(&self, name: &str) -> Option<ScopeEntry> {
        self.scope(name)
    }
}

/// Why a name will not do, or none. `self_id` is the parameter being named, which may keep
/// its own name.
pub fn name_problem(name: &str, others: &[Parameter], self_id: Option<ParameterId>) -> Option<String> {
    if name.is_empty() {
        return Some("needs a name".to_string());
    }
    if is_reserved(name) {
        return Some(format!("“{name}” is a word the expressions keep"));
    }
    if !valid_name(name) {
        return Some("a name is a letter or an underscore, then letters, digits and underscores".to_string());
    }
    let lower = name.to_lowercase();
    let clash = others.iter().find(|p| Some(p.id) != self_id && p.name.to_lowercase() == lower)?;
    Some(if clash.name == name { format!("another parameter is named “{name}” already") } else { format!("too like “{}” — a name may not differ from another’s by its case alone", clash.name) })
}

/// The table being read: each parameter read once, after the ones it reads.
struct TableReader<'a> {
    parameters: &'a [Parameter],
    by_name: HashMap<&'a str, usize>,
    readings: RefCell<HashMap<ParameterId, ParameterReading>>,
    visiting: RefCell<Vec<usize>>,
    looped: RefCell<HashMap<ParameterId, String>>,
}

impl TableReader<'_> {
    fn read(&self, index: usize) -> ParameterReading {
        let p = &self.parameters[index];
        if let Some(done) = self.readings.borrow().get(&p.id) {
            return done.clone();
        }
        let at = self.visiting.borrow().iter().position(|&i| i == index);
        if let Some(at) = at {
            let ring: Vec<usize> = self.visiting.borrow()[at..].to_vec();
            let mut names: Vec<&str> = ring.iter().map(|&i| self.parameters[i].name.as_str()).collect();
            names.push(&p.name);
            let message = format!("goes round in a loop: {}", names.join(" → "));
            let mut looped = self.looped.borrow_mut();
            for &i in &ring {
                looped.insert(self.parameters[i].id, message.clone());
            }
            return ParameterReading { id: p.id, name: p.name.clone(), kind: p.kind, value: None, error: Some(message) };
        }
        self.visiting.borrow_mut().push(index);
        let mut reading = match name_problem(&p.name, self.parameters, Some(p.id)) {
            Some(problem) => ParameterReading { id: p.id, name: p.name.clone(), kind: p.kind, value: None, error: Some(problem) },
            None => {
                let scope = |name: &str| -> Option<ScopeEntry> {
                    let &q = self.by_name.get(name)?;
                    let r = self.read(q);
                    Some(match r.value {
                        None => ScopeEntry::Error(format!("“{name}” has no value: {}", r.error.unwrap_or_default())),
                        Some(value) => ScopeEntry::Value(Value { value, dim: r.kind.dim() }),
                    })
                };
                match evaluate_for(&p.expression, p.kind, &scope, true) {
                    Ok(v) => ParameterReading { id: p.id, name: p.name.clone(), kind: p.kind, value: Some(v.value), error: None },
                    Err(e) => ParameterReading { id: p.id, name: p.name.clone(), kind: p.kind, value: None, error: Some(e.message) },
                }
            }
        };
        self.visiting.borrow_mut().pop();
        if let Some(ring) = self.looped.borrow().get(&p.id) {
            reading = ParameterReading { value: None, error: Some(ring.clone()), ..reading };
        }
        self.readings.borrow_mut().insert(p.id, reading.clone());
        reading
    }
}

/// Evaluate the whole table, each parameter after the ones it reads: a parameter that cannot
/// be read — a name that is wrong, an expression that does not evaluate, a loop of
/// parameters reading each other — is reported with why, and whatever read it says so too.
///
/// Each parameter is read inside the reading of the one that reads it, as ScanRuler reads
/// them, and the whole table counts against the one stack budget of `expression`: a chain of
/// parameters each reading the last, far longer than any table holds, is cut where it runs
/// out — the parameter there "is nested too deeply to read" and those reading it have no
/// value — where ScanRuler's own stack would throw out of the table altogether.
pub fn evaluate_parameters(parameters: &[Parameter]) -> ParameterTable {
    let _depth = ReadingDepth::enter();
    let mut by_name = HashMap::new();
    for (i, p) in parameters.iter().enumerate() {
        by_name.entry(p.name.as_str()).or_insert(i);
    }
    let reader = TableReader { parameters, by_name, readings: RefCell::default(), visiting: RefCell::default(), looped: RefCell::default() };
    let readings: Vec<ParameterReading> = (0..parameters.len()).map(|i| reader.read(i)).collect();
    let mut by_name = HashMap::new();
    for (i, r) in readings.iter().enumerate() {
        by_name.entry(r.name.clone()).or_insert(i);
    }
    ParameterTable { readings, by_name }
}

// ---- slots ------------------------------------------------------------------------

/// How a slot reads its number off a value.
type SlotGet<T> = Box<dyn Fn(&T) -> f64>;
/// How a slot writes its number into a value, giving the value back with it.
type SlotSet<T> = Box<dyn Fn(&T, f64) -> T>;

/// A number of a feature as its box shows it, read and written through the feature's own
/// fields: its path (the key its expression is kept under), its label and its kind.
pub struct Slot<T> {
    pub path: String,
    pub label: String,
    pub kind: ParameterKind,
    get: SlotGet<T>,
    set: SlotSet<T>,
}

impl<T> Slot<T> {
    pub fn new(path: impl Into<String>, label: impl Into<String>, kind: ParameterKind, get: impl Fn(&T) -> f64 + 'static, set: impl Fn(&T, f64) -> T + 'static) -> Self {
        Self { path: path.into(), label: label.into(), kind, get: Box::new(get), set: Box::new(set) }
    }

    /// The number as the box shows it.
    pub fn get(&self, x: &T) -> f64 {
        (self.get)(x)
    }

    /// `x` with the number set as the box would set it — tidied as the slot tidies a typed
    /// number (its sign dropped, a count rounded, an angle kept in range).
    pub fn set(&self, x: &T, value: f64) -> T {
        (self.set)(x, value)
    }

    /// The same slot on a type that holds a `T`: `inner` finds it, `put` writes it back.
    pub fn lift<U>(self, inner: fn(&U) -> &T, put: fn(&U, T) -> U) -> Slot<U>
    where
        T: 'static,
        U: 'static,
    {
        let Slot { path, label, kind, get, set } = self;
        let set = move |u: &U, v: f64| put(u, set(inner(u), v));
        Slot { path, label, kind, get: Box::new(move |u: &U| get(inner(u))), set: Box::new(set) }
    }
}

impl<T> fmt::Debug for Slot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slot").field("path", &self.path).field("label", &self.label).field("kind", &self.kind).finish()
    }
}

/// A typed number with its sign dropped — what most sizes go through.
pub fn positive(v: f64) -> f64 {
    v.abs()
}

/// A feature or a draft that keeps expressions beside its numbers, and offers slots for them
/// to drive.
pub trait Slotted: Clone + PartialEq {
    /// The slots it offers as it stands, as its box shows them.
    fn slots(&self) -> Vec<Slot<Self>>;
    /// The expressions it keeps, by slot path; none when it keeps none.
    fn exprs(&self) -> Option<&ExprMap>;
    /// It with these expressions kept — or none kept.
    fn with_exprs(&self, exprs: Option<ExprMap>) -> Self;
}

/// Where an expression went wrong, by slot path (a sketch's by `dim:<id>`): path → why, kept —
/// as ScanRuler's object keeps them — in the order the expressions were read, a path written
/// again keeping its place. The same ordered map as [`ExprMap`], holding messages.
pub type ExprErrors = ExprMap;

/// The slot of a feature or a draft at a path, or none.
pub fn slot_at<T: Slotted>(x: &T, path: &str) -> Option<Slot<T>> {
    x.slots().into_iter().find(|s| s.path == path)
}

/// Expressions evaluated into the slots of a value.
#[derive(Debug)]
pub struct ExprsApplied<'a, T: Clone> {
    /// The value with the numbers written — borrowed when no number changed.
    pub next: Cow<'a, T>,
    /// The expressions less those whose slot the value no longer has; none when every one
    /// still has its slot.
    pub kept: Option<ExprMap>,
    pub errors: ExprErrors,
}

/// Evaluate a slot-path → expression map into the numbers of `x`, `slots` naming the slots
/// `x` offers as it stands (asked again after each number written, as a written number may
/// change them). An expression that does not evaluate leaves its number as it was and is
/// reported; one whose slot the value no longer has (an extent of another kind) is dropped
/// from `kept`. A count field rounds what comes out; it is not asked to be whole.
pub fn apply_exprs<'a, T: Clone + PartialEq>(x: &'a T, exprs: &ExprMap, scope: &dyn Scope, slots: &dyn Fn(&T) -> Vec<Slot<T>>) -> ExprsApplied<'a, T> {
    let mut errors = ExprErrors::new();
    let mut next: Cow<'a, T> = Cow::Borrowed(x);
    let mut kept: Option<ExprMap> = None;
    for (path, expr) in exprs.iter() {
        let Some(slot) = slots(&next).into_iter().find(|s| s.path == path) else {
            kept.get_or_insert_with(|| exprs.clone()).remove(path);
            continue;
        };
        match evaluate_for(expr, slot.kind, scope, false) {
            Err(e) => {
                errors.insert(path.to_string(), e.message);
            }
            Ok(r) => {
                if (slot.get(&next) - r.value).abs() > 1e-9 {
                    let written = slot.set(&next, r.value);
                    if written != *next {
                        next = Cow::Owned(written);
                    }
                }
            }
        }
    }
    ExprsApplied { next, kept, errors }
}

/// A feature or a draft with its expressions evaluated.
#[derive(Debug)]
pub struct SlotsApplied<'a, T: Clone> {
    /// Borrowed when nothing changes.
    pub next: Cow<'a, T>,
    pub errors: ExprErrors,
}

/// A feature or a draft with every expression it keeps evaluated and the numbers written —
/// the same value back when nothing changes. An expression that does not evaluate leaves
/// its number as it was and is reported; one whose slot the feature no longer has (an extent
/// of another kind) is dropped.
pub fn apply_slots<'a, T: Slotted>(x: &'a T, scope: &dyn Scope) -> SlotsApplied<'a, T> {
    apply_slots_with(x, scope, &|y: &T| y.slots())
}

/// [`apply_slots`] with the slots named by a closure — ScanRuler's `kind` argument, for a
/// draft that does not say which kind it is.
pub fn apply_slots_with<'a, T: Slotted>(x: &'a T, scope: &dyn Scope, slots: &dyn Fn(&T) -> Vec<Slot<T>>) -> SlotsApplied<'a, T> {
    let Some(exprs) = x.exprs() else {
        return SlotsApplied { next: Cow::Borrowed(x), errors: ExprErrors::new() };
    };
    let r = apply_exprs(x, exprs, scope, slots);
    let next = match r.kept {
        None => r.next,
        Some(kept) => Cow::Owned(r.next.with_exprs(if kept.is_empty() { None } else { Some(kept) })),
    };
    SlotsApplied { next, errors: r.errors }
}

/// A feature or a draft with one slot's number set — and its expression, or none: a plain
/// number typed over an expression unlinks it (an empty expression is none).
pub fn with_slot<T: Slotted>(x: &T, path: &str, value: f64, expr: Option<&str>) -> T {
    with_slot_with(x, path, value, expr, &|y: &T| y.slots())
}

/// [`with_slot`] with the slots named by a closure, for a draft that does not say which kind
/// it is.
pub fn with_slot_with<T: Slotted>(x: &T, path: &str, value: f64, expr: Option<&str>, slots: &dyn Fn(&T) -> Vec<Slot<T>>) -> T {
    let next = match slots(x).into_iter().find(|s| s.path == path) {
        Some(slot) => slot.set(x, value),
        None => x.clone(),
    };
    let mut exprs = x.exprs().cloned().unwrap_or_default();
    match expr {
        Some(e) if !e.is_empty() => exprs.insert(path, e),
        _ => {
            exprs.remove(path);
        }
    }
    next.with_exprs(if exprs.is_empty() { None } else { Some(exprs) })
}

/// A feature or a draft with a slot's expression let go — a grip dragged, a number typed.
pub fn unlink_slot<'a, T: Slotted>(x: &'a T, path: &str) -> Cow<'a, T> {
    let Some(exprs) = x.exprs().filter(|e| e.contains(path)) else {
        return Cow::Borrowed(x);
    };
    let mut exprs = exprs.clone();
    exprs.remove(path);
    Cow::Owned(x.with_exprs(if exprs.is_empty() { None } else { Some(exprs) }))
}

/// The kind of a sketch dimension's number.
pub fn dimension_kind(rule: &Rule) -> ParameterKind {
    match rule {
        Rule::Angle { .. } => ParameterKind::Angle,
        _ => ParameterKind::Length,
    }
}

// ---- writing the expressions through ------------------------------------------------

/// A sketch with its dimensions' expressions evaluated.
#[derive(Debug)]
pub struct SketchApplied<'a> {
    /// Borrowed when no value changes.
    pub next: Cow<'a, Sketch>,
    pub errors: ExprErrors,
}

/// The dimensions of a sketch with their expressions evaluated: the values written and the
/// sketch solved again on them — a copy, the given sketch untouched — or the same sketch
/// back when no value changes. `frame` is the plane the sketch lies on, whose coordinate
/// planes' traces its dimensions may be taken to.
pub fn apply_sketch_expressions<'a>(sketch: &'a Sketch, frame: &SectionFrame, scope: &dyn Scope) -> SketchApplied<'a> {
    apply_sketch_expressions_on(sketch, &datum_traces(frame), scope)
}

fn apply_sketch_expressions_on<'a>(sketch: &'a Sketch, datums: &[DatumTrace], scope: &dyn Scope) -> SketchApplied<'a> {
    let mut errors = ExprErrors::new();
    let mut values: HashMap<u32, f64> = HashMap::new();
    for c in &sketch.constraints {
        let (Some(value), Some(expr)) = (c.rule.value(), c.expr.as_deref().filter(|e| !e.is_empty())) else { continue };
        match evaluate_for(expr, dimension_kind(&c.rule), scope, false) {
            Err(e) => {
                errors.insert(format!("dim:{}", c.id), e.message);
            }
            Ok(r) => {
                if (value - r.value).abs() > 1e-9 {
                    values.insert(c.id, r.value);
                }
            }
        }
    }
    if values.is_empty() {
        return SketchApplied { next: Cow::Borrowed(sketch), errors };
    }
    let mut next = sketch.clone();
    for c in &mut next.constraints {
        if let Some(&v) = values.get(&c.id) {
            c.rule.set_value(v);
        }
    }
    // ScanRuler catches what the solver throws into `errors.sketch`; this solver reports
    // rather than throws, and a sketch it cannot meet is told by the build, as ScanRuler's
    // is.
    solve_sketch(&mut next, SolveOptions { datums, ..Default::default() });
    SketchApplied { next: Cow::Owned(next), errors }
}

/// Where a sketch's inset profile came from: the sketch, the wall, and the pieces kept where
/// they lie (ScanRuler's `SketchInset`, `features.ts`).
pub use crate::model::SketchInset;

/// A feature of the list as the parameters reach it. Every feature is [`Slotted`]; a sketch
/// also holds its sketch and the frame it lies in, and a feature made from a sketch names
/// it. The defaults are those of a feature that is no sketch.
pub trait ParametricFeature: Slotted {
    fn id(&self) -> FeatureId;
    fn name(&self) -> &str;
    /// The sketch of a sketch feature (or a sketch being drawn); none for any other.
    fn sketch(&self) -> Option<&Sketch> {
        None
    }
    /// The frame a sketch lies in, as its plane leaves it.
    fn sketch_frame(&self) -> Option<SectionFrame> {
        None
    }
    /// It with another sketch (a sketch feature only).
    fn with_sketch(&self, _sketch: Sketch) -> Self {
        self.clone()
    }
    /// The sketch an extrude or a revolve takes its profiles from.
    fn source_sketch(&self) -> Option<FeatureId> {
        None
    }
    /// A profile moved in from another sketch: where from and by what wall.
    fn inset(&self) -> Option<SketchInset> {
        None
    }
    /// It with this inset, or none.
    fn with_inset(&self, _inset: Option<SketchInset>) -> Self {
        self.clone()
    }
}

/// The feature list with another sketch in the place of the one with `id`: the features on
/// that sketch keep their profiles — each is handed to `rekey`, which matches every pick to
/// the region it names in the sketch as it is now ([`rekeyed`]) and gives back the feature
/// re-keyed, or none to keep it as it was. ScanRuler's `patch` (a new name with the sketch)
/// is the caller's to apply.
pub fn replace_sketch<F: ParametricFeature>(features: &[F], id: FeatureId, sketch: &Sketch, rekey: &mut dyn FnMut(&F) -> Option<F>) -> Vec<F> {
    features
        .iter()
        .map(|f| {
            if f.id() == id && f.sketch().is_some() {
                f.with_sketch(sketch.clone())
            } else if f.source_sketch() == Some(id) {
                rekey(f).unwrap_or_else(|| f.clone())
            } else {
                f.clone()
            }
        })
        .collect()
}

/// How a feature's profiles were picked, kept as given: every region, the largest, or each by
/// key or by a point inside it (ScanRuler's `RegionPick['given']`). In JSON as ScanRuler writes
/// it: `"all"`, `"largest"`, or a list of keys and `{ "at": [u, v] }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "PickGivenJson", into = "PickGivenJson")]
pub enum PickGiven {
    All,
    Largest,
    Picks(Vec<PickItem>),
}

/// One pick of a list: a region by its key, or the region under a point of the sheet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PickItem {
    Key(String),
    At { at: DVec2 },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum PickWord {
    All,
    Largest,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum PickGivenJson {
    Word(PickWord),
    Picks(Vec<PickItem>),
}

impl From<PickGivenJson> for PickGiven {
    fn from(j: PickGivenJson) -> Self {
        match j {
            PickGivenJson::Word(PickWord::All) => PickGiven::All,
            PickGivenJson::Word(PickWord::Largest) => PickGiven::Largest,
            PickGivenJson::Picks(items) => PickGiven::Picks(items),
        }
    }
}

impl From<PickGiven> for PickGivenJson {
    fn from(g: PickGiven) -> Self {
        match g {
            PickGiven::All => PickGivenJson::Word(PickWord::All),
            PickGiven::Largest => PickGivenJson::Word(PickWord::Largest),
            PickGiven::Picks(items) => PickGivenJson::Picks(items),
        }
    }
}

/// A pick as it was given, and the keys it came to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionPick {
    pub given: PickGiven,
    pub keys: Vec<String>,
}

/// The region picks of a feature on a sketch: its region keys, the pick they came from, and
/// those found by place after an edit remade their pieces.
pub trait RegionPicks: Clone {
    fn regions(&self) -> Vec<String>;
    fn region_pick(&self) -> Option<RegionPick>;
    fn regions_by_place(&self) -> Option<Vec<String>>;
    /// It with these picks. `pick` none leaves the pick it has as it stands (ScanRuler's
    /// `rekeyed` writes a pick over or keeps it, never takes one away); `by_place` none leaves
    /// it with no regions found by place.
    fn with_picks(&self, regions: Vec<String>, pick: Option<RegionPick>, by_place: Option<Vec<String>>) -> Self;
}

/// The regions of a sketch as it is now (by index), and the ones it had before the edit (by
/// key), with the matching ScanRuler's `regions.ts` does on them: `matchRegion`,
/// `regionsByPick`, `regionByPlace` and `splitParts`.
pub trait RegionMatch {
    /// The key of a region as it is now.
    fn key(&self, region: usize) -> &str;
    /// The region a key names now: the same key, or the region made of the same pieces.
    fn match_region(&self, key: &str) -> Option<usize>;
    /// The regions a pick kept as given takes now; none when there are none to take.
    fn regions_by_pick(&self, given: &PickGiven) -> Option<Vec<usize>>;
    /// Whether the sketch had a region of this key before the edit.
    fn was_region(&self, key: &str) -> bool;
    /// The region now covering half or more of the one of this key before, not one of
    /// `taken`.
    fn region_by_place(&self, was_key: &str, taken: &[usize]) -> Option<usize>;
    /// The regions now inside the one of this key before, made from its pieces.
    fn split_parts(&self, was_key: &str) -> Vec<usize>;
}

/// Whether two lists of region keys hold the same keys.
fn same_keys(a: &[String], b: &[String]) -> bool {
    a.len() == b.len() && a.iter().all(|k| b.contains(k))
}

/// A feature's picks written over with the keys of the regions they name now. A pick kept as
/// it was given — every region, the largest, points inside them (`RegionPick`) — is taken
/// again as given, while it stands. Otherwise each key is matched (`match_region`); one whose
/// pieces were all remade is found by place, the region now covering half of it or more
/// (`region_by_place`), and its row says so (`regions_by_place`). A picked region the edit
/// cut in two is picked in both parts — the feature makes what it made — and a pick that
/// names nothing now, or a region another pick names, is kept as it was: the build names it
/// lost on the feature's row rather than the profile leaving quietly.
pub fn rekeyed<'a, T: RegionPicks>(f: &'a T, regions: &dyn RegionMatch) -> Cow<'a, T> {
    let picked = f.regions();
    let was_by_place = f.regions_by_place();
    let pick = f.region_pick().filter(|p| same_keys(&p.keys, &picked));
    if let Some(pick) = pick {
        if let Some(asked) = regions.regions_by_pick(&pick.given) {
            let keys: Vec<String> = asked.iter().map(|&r| regions.key(r).to_string()).collect();
            if same_keys(&keys, &picked) && was_by_place.is_none() && keys.len() == picked.len() && keys.iter().zip(&picked).all(|(k, was)| k == was) {
                return Cow::Borrowed(f);
            }
            return Cow::Owned(f.with_picks(keys.clone(), Some(RegionPick { given: pick.given, keys }), None));
        }
    }
    let mut taken: Vec<usize> = Vec::new();
    let mut by_place: Vec<String> = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for key in &picked {
        let mut r = regions.match_region(key);
        if r.is_none() && regions.was_region(key) {
            r = regions.region_by_place(key, &taken);
            if let Some(found) = r {
                by_place.push(regions.key(found).to_string());
            }
        }
        match r {
            Some(found) if !taken.contains(&found) => {
                taken.push(found);
                out.push(regions.key(found).to_string());
            }
            _ => out.push(key.clone()),
        }
    }
    if taken.is_empty() {
        return Cow::Borrowed(f);
    }
    for key in &picked {
        if !regions.was_region(key) {
            continue;
        }
        for part in regions.split_parts(key) {
            if taken.contains(&part) {
                continue;
            }
            taken.push(part);
            out.push(regions.key(part).to_string());
        }
    }
    // Found by place before and still picked: still said.
    let mut placed: Vec<String> = Vec::new();
    for k in was_by_place.iter().flatten().filter(|k| out.contains(k)).chain(by_place.iter()) {
        if !placed.contains(k) {
            placed.push(k.clone());
        }
    }
    let same = out.len() == picked.len() && out.iter().zip(&picked).all(|(k, was)| k == was);
    if same && placed.len() == was_by_place.as_ref().map_or(0, Vec::len) {
        return Cow::Borrowed(f);
    }
    Cow::Owned(f.with_picks(out, f.region_pick(), if placed.is_empty() { None } else { Some(placed) }))
}

/// What puts a sketch in the place of the one with an id and re-keys the features on it — see
/// [`apply_parameters`].
pub type ReplaceSketch<'a, F> = dyn FnMut(&[F], FeatureId, Sketch) -> Vec<F> + 'a;

/// What moves a sketch's inset profile in again from the sketch it came from — see
/// [`apply_parameters`].
pub type InsetAgain<'a, F> = dyn FnMut(&F, &[F]) -> Result<Sketch, String> + 'a;

/// The move of an inset itself: the source sketch, the inset, the datums of the sketch being
/// moved into and the source's name, to the moved sketch or why it cannot be.
pub type InsetChains<'a> = dyn Fn(&Sketch, &SketchInset, &[DatumTrace], &str) -> Result<Sketch, String> + 'a;

/// Every feature with its expressions evaluated against the parameters.
#[derive(Debug)]
pub struct ParametersApplied<F> {
    pub features: Vec<F>,
    /// Whether any number moved — what asks for a rebuild.
    pub changed: bool,
    /// By feature id.
    pub errors: BTreeMap<FeatureId, ExprErrors>,
}

/// Every feature with its expressions evaluated against the parameters: the numbers written,
/// the sketches with a changed dimension solved again and the features on them re-keyed. A
/// feature that comes out the same is the same value, and a list where nothing moved is the
/// same list (`changed` false).
///
/// `replace_sketch` puts a sketch in the place of the one with an id and re-keys the features
/// on it — [`replace_sketch`] with a `rekey` that runs [`rekeyed`] against the sketch's
/// regions. `inset_again` moves an inset profile in again from the sketch it came from
/// ([`inset_again`]); it is asked only of a sketch whose inset's wall changed.
pub fn apply_parameters<F: ParametricFeature>(
    features: &[F],
    scope: &dyn Scope,
    replace_sketch: &mut ReplaceSketch<F>,
    inset_again: &mut InsetAgain<F>,
) -> ParametersApplied<F> {
    let mut errors: BTreeMap<FeatureId, ExprErrors> = BTreeMap::new();
    let mut out: Vec<F> = features.to_vec();
    let mut changed = false;
    // The list's length read again on every step, as ScanRuler's loop reads it: the list is
    // `replace_sketch`'s to hand back, and a step past its end is no step.
    let mut i = 0;
    while i < out.len() {
        let f = out[i].clone();
        if f.sketch().is_some() {
            // The plane's offset first: the dimensions are solved in the frame it leaves.
            let placed = apply_slots(&f, scope);
            let mut placed_errors = placed.errors;
            let mut next = placed.next.into_owned();
            // A profile moved in from another sketch, by a wall that came to another number:
            // moved in again from that sketch as it stands now.
            let mut moved_in: Option<Sketch> = None;
            if let (Some(now), Some(was)) = (next.inset(), f.inset()) {
                if now.by != was.by {
                    match inset_again(&next, &out) {
                        Err(why) => {
                            placed_errors.insert("inset".to_string(), why);
                            next = next.with_inset(Some(was));
                        }
                        Ok(again) => {
                            next = next.with_sketch(again.clone());
                            moved_in = Some(again);
                        }
                    }
                }
            }
            if next != f {
                out[i] = next.clone();
                changed = true;
            }
            // The features on it keep their profiles, as a dimension's change leaves them.
            if let Some(again) = moved_in {
                out = replace_sketch(&out, f.id(), again);
            }
            let mut wrong = placed_errors;
            if let Some(sketch) = next.sketch() {
                let datums = next.sketch_frame().map(|frame| datum_traces(&frame)).unwrap_or_default();
                let r = apply_sketch_expressions_on(sketch, &datums, scope);
                wrong.extend(r.errors);
                if let Cow::Owned(solved) = r.next {
                    out = replace_sketch(&out, f.id(), solved);
                    changed = true;
                }
            }
            if !wrong.is_empty() {
                errors.insert(f.id(), wrong);
            }
        } else {
            let r = apply_slots(&f, scope);
            if !r.errors.is_empty() {
                errors.insert(f.id(), r.errors);
            }
            if let Cow::Owned(next) = r.next {
                out[i] = next;
                changed = true;
            }
        }
        i += 1;
    }
    ParametersApplied { features: if changed { out } else { features.to_vec() }, changed, errors }
}

/// A sketch's profile moved in again from the sketch it was moved in from, by its inset's
/// wall, and solved — or why not. `inset_chains` is the move itself (ScanRuler's
/// `insetChains`, `stations.ts`): the source sketch, the inset, the datums of the sketch
/// being moved into and the source's name, to the moved sketch or why it cannot be.
pub fn inset_again<F: ParametricFeature>(f: &F, features: &[F], inset_chains: &InsetChains) -> Result<Sketch, String> {
    let Some(inset) = f.inset() else {
        return Err("the sketch was not moved in from another".to_string());
    };
    // The first sketch of that id, as ScanRuler's `find` asks for both at once.
    let Some((of, of_sketch)) = features.iter().find_map(|x| x.sketch().filter(|_| x.id() == inset.of).map(|s| (x, s))) else {
        return Err(format!("the sketch it was moved in from ({}) is gone", inset.of));
    };
    let datums = f.sketch_frame().map(|frame| datum_traces(&frame)).unwrap_or_default();
    let mut moved = inset_chains(of_sketch, &inset, &datums, of.name())?;
    solve_sketch(&mut moved, SolveOptions { datums: &datums, ..Default::default() });
    Ok(moved)
}

/// [`apply_parameters`] as a change to the parameter table reaches the features: a rename
/// written through every expression first, then every expression evaluated against the
/// table (ScanRuler's `followParameters`, for the feature list; the drafts open in its
/// store follow with [`rename_in_feature`] and [`apply_slots`] the same way).
pub fn follow_parameters<F: ParametricFeature>(
    features: &[F],
    table: &ParameterTable,
    renamed: Option<(&str, &str)>,
    replace_sketch: &mut ReplaceSketch<F>,
    inset_again: &mut InsetAgain<F>,
) -> ParametersApplied<F> {
    let features: Vec<F> = match renamed {
        Some((from, to)) => features.iter().map(|f| rename_in_feature(f, from, to).into_owned()).collect(),
        None => features.to_vec(),
    };
    apply_parameters(&features, table, replace_sketch, inset_again)
}

/// The expressions of a feature that do not evaluate now, by slot — for its row to say so.
pub fn feature_expression_errors<F: ParametricFeature>(f: &F, scope: &dyn Scope) -> ExprErrors {
    let mut errors = ExprErrors::new();
    if let Some(sketch) = f.sketch() {
        for (path, expr) in f.exprs().into_iter().flat_map(ExprMap::iter) {
            if let Err(e) = evaluate_for(expr, ParameterKind::Length, scope, false) {
                errors.insert(path.to_string(), e.message);
            }
        }
        for c in &sketch.constraints {
            let Some(expr) = c.expr.as_deref().filter(|e| !e.is_empty() && c.rule.is_dimension()) else { continue };
            if let Err(e) = evaluate_for(expr, dimension_kind(&c.rule), scope, false) {
                errors.insert(format!("dim:{}", c.id), e.message);
            }
        }
        return errors;
    }
    let slots = f.slots();
    for (path, expr) in f.exprs().into_iter().flat_map(ExprMap::iter) {
        let Some(slot) = slots.iter().find(|s| s.path == path) else { continue };
        if let Err(e) = evaluate_for(expr, slot.kind, scope, false) {
            errors.insert(path.to_string(), e.message);
        }
    }
    errors
}

// ---- who reads a parameter -------------------------------------------------------------

/// What reads a parameter: another parameter or a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserKind {
    Parameter,
    Feature,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterUser {
    pub kind: UserKind,
    pub id: u32,
    pub name: String,
}

/// The dimension expressions of a sketch, by constraint id.
fn dimension_exprs(sketch: &Sketch) -> impl Iterator<Item = (u32, &str)> {
    sketch.constraints.iter().filter(|c| c.rule.is_dimension()).filter_map(|c| c.expr.as_deref().filter(|e| !e.is_empty()).map(|e| (c.id, e)))
}

fn reads(expr: &str, name: &str) -> bool {
    references(expr).iter().any(|r| r == name)
}

/// Whether a feature's expressions — its slots', or its sketch's dimensions' — read the
/// name.
pub fn feature_reads<F: ParametricFeature>(f: &F, name: &str) -> bool {
    if f.exprs().is_some_and(|m| m.exprs().any(|e| reads(e, name))) {
        return true;
    }
    f.sketch().is_some_and(|s| dimension_exprs(s).any(|(_, e)| reads(e, name)))
}

/// Everything that reads a parameter by name: the parameters whose expressions use it, and
/// the features — a feature's slots, a sketch's dimensions.
pub fn users_of<F: ParametricFeature>(name: &str, parameters: &[Parameter], features: &[F]) -> Vec<ParameterUser> {
    let mut out = Vec::new();
    for p in parameters {
        if p.name != name && reads(&p.expression, name) {
            out.push(ParameterUser { kind: UserKind::Parameter, id: p.id, name: p.name.clone() });
        }
    }
    for f in features {
        if feature_reads(f, name) {
            out.push(ParameterUser { kind: UserKind::Feature, id: f.id(), name: f.name().to_string() });
        }
    }
    out
}

/// Where a feature reads a parameter: its slots by their paths ("extent.to"), a sketch's
/// dimensions as "dimension 14".
pub fn fields_reading<F: ParametricFeature>(f: &F, name: &str) -> Vec<String> {
    let mut out: Vec<String> = f.exprs().into_iter().flat_map(ExprMap::iter).filter(|(_, e)| reads(e, name)).map(|(path, _)| path.to_string()).collect();
    if let Some(sketch) = f.sketch() {
        out.extend(dimension_exprs(sketch).filter(|(_, e)| reads(e, name)).map(|(id, _)| format!("dimension {id}")));
    }
    out
}

/// The parameters that drive nothing: no feature reads them, nor reads a parameter that
/// does. One only another idle parameter reads is idle too: changing it moves no face.
pub fn idle_parameters<'a, F: ParametricFeature>(parameters: &'a [Parameter], features: &[F]) -> Vec<&'a Parameter> {
    let mut driving: HashSet<&str> = parameters.iter().filter(|p| features.iter().any(|f| feature_reads(f, &p.name))).map(|p| p.name.as_str()).collect();
    let mut grew = true;
    while grew {
        grew = false;
        for p in parameters {
            if !driving.contains(p.name.as_str()) {
                continue;
            }
            for name in references(&p.expression) {
                if driving.contains(name.as_str()) {
                    continue;
                }
                let Some(q) = parameters.iter().find(|q| q.name == name) else { continue };
                driving.insert(q.name.as_str());
                grew = true;
            }
        }
    }
    parameters.iter().filter(|p| !driving.contains(p.name.as_str())).collect()
}

/// A feature or a draft with a parameter renamed in every expression it keeps — the same
/// value back when it reads no such name.
pub fn rename_in_feature<'a, F: ParametricFeature>(f: &'a F, from: &str, to: &str) -> Cow<'a, F> {
    if !feature_reads(f, from) {
        return Cow::Borrowed(f);
    }
    let mut next = f.clone();
    if let Some(exprs) = f.exprs() {
        next = next.with_exprs(Some(exprs.iter().map(|(k, e)| (k, rename_in(e, from, to))).collect()));
    }
    if let Some(sketch) = f.sketch() {
        next = next.with_sketch(rename_in_sketch(sketch, from, to).into_owned());
    }
    Cow::Owned(next)
}

/// A sketch with a parameter renamed in its dimensions' expressions — the same sketch back
/// when none reads it.
pub fn rename_in_sketch<'a>(sketch: &'a Sketch, from: &str, to: &str) -> Cow<'a, Sketch> {
    if !dimension_exprs(sketch).any(|(_, e)| reads(e, from)) {
        return Cow::Borrowed(sketch);
    }
    let mut next = sketch.clone();
    for c in &mut next.constraints {
        if !c.rule.is_dimension() {
            continue;
        }
        if let Some(e) = c.expr.as_mut().filter(|e| !e.is_empty()) {
            *e = rename_in(e, from, to);
        }
    }
    Cow::Owned(next)
}

/// The parameters with one of them renamed, in its row and in every expression that reads
/// it.
pub fn rename_in_parameters(parameters: &[Parameter], id: ParameterId, to: &str) -> Vec<Parameter> {
    let Some(from) = parameters.iter().find(|p| p.id == id).map(|p| p.name.clone()).filter(|from| from != to) else {
        return parameters.to_vec();
    };
    parameters
        .iter()
        .map(|p| {
            if p.id == id {
                Parameter { name: to.to_string(), ..p.clone() }
            } else if reads(&p.expression, &from) {
                Parameter { expression: rename_in(&p.expression, &from, to), ..p.clone() }
            } else {
                p.clone()
            }
        })
        .collect()
}

// ---- the parameter list's edits (ScanRuler's store) -------------------------------------

/// A new parameter at zero, named in turn: the first of p1, p2, … not taken.
pub fn new_parameter(parameters: &[Parameter], id: ParameterId, kind: ParameterKind) -> Parameter {
    let mut n = 1;
    while parameters.iter().any(|p| p.name == format!("p{n}")) {
        n += 1;
    }
    Parameter { id, name: format!("p{n}"), kind, expression: "0".to_string(), comment: None }
}

/// What an edit of a parameter's row changes; a field left none is left as it is.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParameterPatch {
    pub name: Option<String>,
    pub kind: Option<ParameterKind>,
    pub expression: Option<String>,
    pub comment: Option<String>,
}

/// A parameter edit taken: the list and its table, and the rename it made (from, to) for the
/// features to follow ([`follow_parameters`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterEdit {
    pub parameters: Vec<Parameter>,
    pub table: ParameterTable,
    pub renamed: Option<(String, String)>,
}

/// A parameter's row edited — or refused, and why, with nothing changed: a name that will
/// not do, an expression that does not evaluate, or an edit that would leave another
/// parameter without a value (a kind changed under one that reads it). `before` is the table
/// of the list as it stands. A rename is written through every parameter that reads it.
pub fn edit_parameter(parameters: &[Parameter], before: &ParameterTable, id: ParameterId, patch: &ParameterPatch) -> Result<ParameterEdit, String> {
    let Some(p) = parameters.iter().find(|x| x.id == id) else {
        return Err("There is no such parameter.".to_string());
    };
    let mut list = parameters.to_vec();
    let renamed = patch.name.as_ref().filter(|to| **to != p.name).map(|to| (p.name.clone(), to.clone()));
    if let Some((_, to)) = &renamed {
        if let Some(problem) = name_problem(to, parameters, Some(id)) {
            return Err(problem);
        }
        list = rename_in_parameters(&list, id, to);
    }
    for x in list.iter_mut().filter(|x| x.id == id) {
        if let Some(kind) = patch.kind {
            x.kind = kind;
        }
        if let Some(expression) = &patch.expression {
            x.expression = expression.clone();
        }
        if let Some(comment) = &patch.comment {
            x.comment = Some(comment.clone());
        }
    }
    let table = evaluate_parameters(&list);
    if let Some(error) = table.reading(id).and_then(|r| r.error.clone()) {
        return Err(error);
    }
    // Nor may an edit leave another parameter without a value — a kind changed under one
    // that reads it.
    let was: HashMap<ParameterId, bool> = before.readings.iter().map(|r| (r.id, r.error.is_some())).collect();
    if let Some(broke) = table.readings.iter().find(|r| r.error.is_some() && !was.get(&r.id).copied().unwrap_or(false)) {
        return Err(format!("would leave “{}” with no value: {}", broke.name, broke.error.clone().unwrap_or_default()));
    }
    Ok(ParameterEdit { parameters: list, table, renamed })
}

/// How the store names a draft open on a parameter when it refuses to remove it.
pub const READ_BY_FEATURE_DRAFT: &str = "the feature being set up";
pub const READ_BY_SKETCH_DRAFT: &str = "the sketch being drawn";

/// Why a parameter may not be removed — what reads it — or none. `features` is every feature,
/// rolled back or not; `drafts_reading` names the drafts open on it that read it
/// ([`READ_BY_FEATURE_DRAFT`], [`READ_BY_SKETCH_DRAFT`]).
pub fn removal_problem<F: ParametricFeature>(parameters: &[Parameter], features: &[F], id: ParameterId, drafts_reading: &[&str]) -> Option<String> {
    let p = parameters.iter().find(|x| x.id == id)?;
    let mut users: Vec<String> = users_of(&p.name, parameters, features).into_iter().map(|u| u.name).collect();
    users.extend(drafts_reading.iter().map(|d| d.to_string()));
    if users.is_empty() {
        return None;
    }
    Some(format!("{} is read by {} — take it out of those first.", p.name, users.join(", ")))
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reParameters.test.ts` ("the table"),
    //! the parameter list of `reParameterStore.test.ts`, and the pieces of the store the
    //! port carries; the slots and the features follow in `param_slots`.
    use super::*;

    pub(crate) fn param(id: u32, name: &str, expression: &str, kind: ParameterKind) -> Parameter {
        Parameter { id, name: name.to_string(), kind, expression: expression.to_string(), comment: None }
    }
    fn length(id: u32, name: &str, expression: &str) -> Parameter {
        param(id, name, expression, ParameterKind::Length)
    }
    fn row(t: &ParameterTable) -> Vec<(String, Option<f64>, Option<String>)> {
        t.readings.iter().map(|r| (r.name.clone(), r.value, r.error.clone())).collect()
    }

    #[test]
    fn the_table_evaluates_each_parameter_after_the_ones_it_reads_in_any_order() {
        let t = evaluate_parameters(&[length(1, "total", "width + height"), length(2, "width", "40"), length(3, "height", "width / 2")]);
        assert_eq!(row(&t), vec![("total".to_string(), Some(60.0), None), ("width".to_string(), Some(40.0), None), ("height".to_string(), Some(20.0), None)]);
        assert_eq!(t.scope("total"), Some(ScopeEntry::Value(Value { value: 60.0, dim: crate::expression::Dim::Of { mm: 1.0, deg: 0.0 } })));
        assert_eq!(t.scope("nothing"), None);
    }

    #[test]
    fn the_table_reads_a_bare_number_in_the_parameters_kind_and_keeps_the_kinds_apart() {
        let t = evaluate_parameters(&[
            param(1, "turn", "30", ParameterKind::Angle),
            param(2, "count", "6", ParameterKind::Integer),
            param(3, "ratio", "0.5", ParameterKind::Number),
            param(4, "bad", "turn + 2 mm", ParameterKind::Angle),
            param(5, "worse", "count", ParameterKind::Length),
        ]);
        assert_eq!(t.readings[0].value, Some(30.0));
        assert_eq!(t.readings[1].value, Some(6.0));
        assert_eq!(t.readings[2].value, Some(0.5));
        assert_eq!((t.readings[3].value, t.readings[3].error.as_deref()), (None, Some("an angle and a length cannot be added")));
        assert_eq!(t.readings[4].value, None);
        assert!(t.readings[4].error.as_deref().is_some_and(|e| e.contains("comes out as a number; a length is wanted")));
    }

    #[test]
    fn the_table_asks_a_whole_number_of_an_integer_parameter() {
        let t = evaluate_parameters(&[param(1, "holes", "7.3", ParameterKind::Integer), param(2, "teeth", "round(7.3)", ParameterKind::Integer)]);
        assert!(t.readings[0].error.as_deref().is_some_and(|e| e.contains("not whole")));
        assert_eq!(t.readings[1].value, Some(7.0));
    }

    #[test]
    fn the_table_reports_a_loop_on_every_parameter_in_it_and_what_reads_the_loop() {
        let t = evaluate_parameters(&[length(1, "a", "b + 1"), length(2, "b", "c * 2"), length(3, "c", "a"), length(4, "d", "a + 5"), length(5, "f", "3")]);
        for r in &t.readings[..3] {
            assert_eq!(r.value, None);
            assert!(r.error.as_deref().is_some_and(|e| e.contains("goes round in a loop")), "{r:?}");
        }
        assert_eq!(t.readings[0].error.as_deref(), Some("goes round in a loop: a → b → c → a"));
        assert_eq!(t.readings[3].value, None);
        assert!(t.readings[3].error.as_deref().is_some_and(|e| e.contains("“a” has no value")));
        assert_eq!(t.readings[4].value, Some(3.0));
        assert_eq!(t.scope("a"), Some(ScopeEntry::Error("“a” has no value".to_string())));
    }

    #[test]
    fn the_table_refuses_a_name_that_will_not_do() {
        let others = [length(1, "width", "1"), length(2, "Height", "2")];
        assert_eq!(name_problem("", &others, None).as_deref(), Some("needs a name"));
        assert!(name_problem("sin", &others, None).is_some_and(|e| e.contains("a word the expressions keep")));
        assert!(name_problem("2wide", &others, None).is_some_and(|e| e.contains("a letter or an underscore")));
        assert!(name_problem("width", &others, None).is_some_and(|e| e.contains("already")));
        assert_eq!(name_problem("width", &others, Some(1)), None);
        assert!(name_problem("height", &others, None).is_some_and(|e| e.contains("case alone")));
        assert_eq!(name_problem("depth", &others, None), None);
        let t = evaluate_parameters(&[length(1, "w", "1"), length(2, "w", "2"), length(3, "mm", "3")]);
        assert!(t.readings[1].error.as_deref().is_some_and(|e| e.contains("already")));
        assert!(t.readings[2].error.as_deref().is_some_and(|e| e.contains("keep")));
    }

    /// Not in ScanRuler's tests: a chain of parameters each reading the last is read in full
    /// where any table could hold one, and one far longer — ScanRuler's own stack would throw
    /// out of the table — is cut where the stack budget runs out, never aborting. On a thread
    /// of 2 MiB, Rust's least, so an unoptimised build shows it too.
    #[test]
    fn the_table_reads_a_long_chain_and_cuts_one_too_long_to_follow() {
        let chain = |n: u32| -> Vec<Parameter> {
            // Listed last first, so the first reading goes all the way down.
            (0..n).rev().map(|i| length(i + 1, &format!("p{i}"), &if i == 0 { "1".to_string() } else { format!("p{} + 1", i - 1) })).collect()
        };
        let small = std::thread::Builder::new().stack_size(2 << 20).spawn(move || {
            let t = evaluate_parameters(&chain(40));
            assert!(t.readings.iter().all(|r| r.error.is_none()), "{:?}", t.readings.iter().find(|r| r.error.is_some()));
            assert_eq!(t.readings[0].value, Some(40.0));
            let t = evaluate_parameters(&chain(5_000));
            let cut = t.readings.iter().filter(|r| r.error.is_some()).count();
            assert!(cut > 0 && t.readings.iter().all(|r| r.value.is_some() || r.error.as_deref().is_some_and(|e| e.contains("nested too deeply"))));
            assert_eq!(t.readings.last().and_then(|r| r.value), Some(1.0));
        });
        assert!(small.is_ok_and(|h| h.join().is_ok()));
    }

    /// Not in ScanRuler's tests: the table's readings are anyone's to change, and its scope
    /// still answers by name — never past the end of the list.
    #[test]
    fn the_table_answers_by_name_after_its_readings_change() {
        let mut t = evaluate_parameters(&[length(1, "a", "1"), length(2, "b", "2")]);
        t.readings.remove(0);
        assert_eq!(t.scope("b"), Some(ScopeEntry::Value(Value { value: 2.0, dim: crate::expression::LENGTH })));
        assert_eq!(t.scope("a"), None);
        t.readings.clear();
        assert_eq!(t.scope("b"), None);
    }

    #[test]
    fn negated_expressions_and_the_commands_draft() {
        assert_eq!(negated_expr("5"), "-5");
        assert_eq!(negated_expr(" -2.5 "), "2.5");
        assert_eq!(negated_expr("0"), "0");
        assert_eq!(negated_expr("1e3"), "-1000");
        assert_eq!(negated_expr("a + b"), "-(a + b)");
        assert_eq!(negated_expr("-( a + b )"), "a + b");
        assert_eq!(negated_expr("-(a) * (b)"), "-(-(a) * (b))");
        let given: serde_json::Map<String, serde_json::Value> = serde_json::from_str(r#"{ "extent.to": "h", "draftDeg": "lean" }"#).unwrap_or_default();
        let taken = exprs_from_command(Some("extrude"), given.clone());
        assert_eq!(taken.get("taper"), Some(&serde_json::json!("-(lean)")));
        assert!(!taken.contains_key("draftDeg"));
        assert_eq!(exprs_from_command(Some("revolve"), given.clone()), given);
        let kept: ExprMap = [("extent.to", "h"), ("taper", "-(lean)")].into_iter().collect();
        assert_eq!(exprs_for_command(Some("extrude"), &kept), [("extent.to", "h"), ("draftDeg", "lean")].into_iter().collect());
    }

    #[test]
    fn an_expression_map_keeps_its_order_and_reads_back_as_json() {
        let mut m = ExprMap::new();
        m.insert("taper", "2");
        m.insert("extent.to", "h");
        m.insert("taper", "3");
        assert_eq!(m.paths().collect::<Vec<_>>(), ["taper", "extent.to"]);
        let text = serde_json::to_string(&m).unwrap_or_default();
        assert_eq!(text, r#"{"taper":"3","extent.to":"h"}"#);
        let back: ExprMap = serde_json::from_str(&text).unwrap_or_default();
        assert_eq!(back.paths().collect::<Vec<_>>(), ["taper", "extent.to"]);
        let other: ExprMap = [("extent.to", "h"), ("taper", "3")].into_iter().collect();
        assert_eq!(m, other);
    }

    /// `reParameterStore.test.ts`: the parameter list adds a parameter at zero, named in
    /// turn, and takes its name, kind and expression.
    #[test]
    fn the_parameter_list_adds_and_takes_its_name_kind_and_expression() {
        let mut list = vec![new_parameter(&[], 1, ParameterKind::Length)];
        assert_eq!(list, vec![param(1, "p1", "0", ParameterKind::Length)]);
        let mut table = evaluate_parameters(&list);
        assert_eq!((table.readings[0].value, table.readings[0].error.clone()), (Some(0.0), None));
        list.push(new_parameter(&list, 2, ParameterKind::Integer));
        assert_eq!((list[1].name.as_str(), list[1].kind), ("p2", ParameterKind::Integer));
        table = evaluate_parameters(&list);
        let edit = edit_parameter(&list, &table, 1, &ParameterPatch { name: Some("height".into()), expression: Some("25.4 / 2".into()), ..Default::default() });
        let Ok(edit) = edit else { panic!("{edit:?}") };
        assert_eq!((edit.parameters[0].name.as_str(), edit.parameters[0].expression.as_str()), ("height", "25.4 / 2"));
        assert!((edit.table.readings[0].value.unwrap_or(f64::NAN) - 12.7).abs() < 0.005);
        let patch = ParameterPatch { name: Some("count".into()), expression: Some("round(height / 2 mm)".into()), comment: Some("how many".into()), ..Default::default() };
        let edit = edit_parameter(&edit.parameters, &edit.table, 2, &patch);
        let Ok(edit) = edit else { panic!("{edit:?}") };
        assert_eq!((edit.table.readings[1].value, edit.table.readings[1].error.clone()), (Some(6.0), None));
        assert_eq!(edit.parameters[1].comment.as_deref(), Some("how many"));
    }

    #[test]
    fn the_parameter_list_refuses_what_will_not_do_and_changes_nothing() {
        let list = vec![length(1, "width", "40")];
        let table = evaluate_parameters(&list);
        let refuse = |list: &[Parameter], table: &ParameterTable, id: u32, patch: ParameterPatch| match edit_parameter(list, table, id, &patch) {
            Ok(_) => panic!("{patch:?} was taken"),
            Err(e) => e,
        };
        assert!(refuse(&list, &table, 1, ParameterPatch { name: Some("sin".into()), ..Default::default() }).contains("keep"));
        assert!(refuse(&list, &table, 1, ParameterPatch { name: Some("2w".into()), ..Default::default() }).contains("a letter or an underscore"));
        assert_eq!(refuse(&list, &table, 1, ParameterPatch { expression: Some("nothing + 1".into()), ..Default::default() }), "there is no parameter “nothing”");
        assert!(refuse(&list, &table, 1, ParameterPatch { expression: Some("1 /".into()), ..Default::default() }).contains("expected"));
        assert!(refuse(&list, &table, 1, ParameterPatch { kind: Some(ParameterKind::Integer), expression: Some("7.3".into()), ..Default::default() }).contains("not whole"));
        // An edit that would leave another parameter without a value is refused too.
        let mut list = list;
        list.push(new_parameter(&list, 2, ParameterKind::Length));
        let table = evaluate_parameters(&list);
        let half = edit_parameter(&list, &table, 2, &ParameterPatch { name: Some("half".into()), expression: Some("width / 2".into()), ..Default::default() });
        let Ok(half) = half else { panic!("{half:?}") };
        let why = refuse(&half.parameters, &half.table, 1, ParameterPatch { kind: Some(ParameterKind::Angle), ..Default::default() });
        assert!(why.contains("would leave “half” with no value"), "{why}");
        assert_eq!(edit_parameter(&list, &table, 9, &ParameterPatch::default()).err().as_deref(), Some("There is no such parameter."));
    }

    #[test]
    fn the_parameter_list_refuses_a_remove_while_anything_reads_the_parameter() {
        let list = vec![length(1, "width", "40"), length(2, "half", "width / 2")];
        let none: Vec<crate::param_slots::JsonFeature> = Vec::new();
        assert!(removal_problem(&list, &none, 1, &[]).is_some_and(|e| e.contains("half")));
        assert_eq!(removal_problem(&list, &none, 1, &[]).as_deref(), Some("width is read by half — take it out of those first."));
        assert_eq!(removal_problem(&list, &none, 2, &[]), None);
        assert_eq!(removal_problem(&list[..1], &none, 1, &[]), None);
        assert!(removal_problem(&list[..1], &none, 1, &[READ_BY_SKETCH_DRAFT]).is_some_and(|e| e.contains("the sketch being drawn")));
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Picked {
        regions: Vec<String>,
        pick: Option<RegionPick>,
        by_place: Option<Vec<String>>,
    }

    impl RegionPicks for Picked {
        fn regions(&self) -> Vec<String> {
            self.regions.clone()
        }
        fn region_pick(&self) -> Option<RegionPick> {
            self.pick.clone()
        }
        fn regions_by_place(&self) -> Option<Vec<String>> {
            self.by_place.clone()
        }
        fn with_picks(&self, regions: Vec<String>, pick: Option<RegionPick>, by_place: Option<Vec<String>>) -> Self {
            Picked { regions, pick, by_place }
        }
    }

    /// Regions now "r:1+2" (made of the pieces of the old "r:1") and "r:9", where before there
    /// were "r:1" and "r:5"; "r:5" was remade and is found by place in "r:9".
    struct Toy;

    impl RegionMatch for Toy {
        fn key(&self, region: usize) -> &str {
            ["r:1+2", "r:9", "r:7"][region]
        }
        fn match_region(&self, key: &str) -> Option<usize> {
            match key {
                "r:1+2" | "r:1" => Some(0),
                "r:9" => Some(1),
                _ => None,
            }
        }
        fn regions_by_pick(&self, given: &PickGiven) -> Option<Vec<usize>> {
            match given {
                PickGiven::All => Some(vec![0, 1, 2]),
                PickGiven::Largest => Some(vec![1]),
                PickGiven::Picks(items) => Some(items.iter().filter_map(|i| if let PickItem::Key(k) = i { self.match_region(k) } else { None }).collect()),
            }
        }
        fn was_region(&self, key: &str) -> bool {
            key == "r:1" || key == "r:5"
        }
        fn region_by_place(&self, was_key: &str, taken: &[usize]) -> Option<usize> {
            (was_key == "r:5" && !taken.contains(&1)).then_some(1)
        }
        fn split_parts(&self, was_key: &str) -> Vec<usize> {
            if was_key == "r:1" {
                vec![0, 2]
            } else {
                Vec::new()
            }
        }
    }

    /// `reParameters.test.ts`, "replace a sketch and rekey the features on it": a pick that
    /// matches nothing is kept as it was, for the feature to report — here against a toy set
    /// of regions, as the regions of a sketch are `regions.rs`'s to find.
    #[test]
    fn rekeying_keeps_a_pick_that_matches_nothing_and_follows_the_rest() {
        let lost = Picked { regions: vec!["r:nothing".into()], pick: None, by_place: None };
        assert!(matches!(rekeyed(&lost, &Toy), Cow::Borrowed(_)));
        // Matched by pieces, cut in two (both parts picked), and found by place.
        let f = Picked { regions: vec!["r:1".into(), "r:5".into()], pick: None, by_place: None };
        let r = rekeyed(&f, &Toy).into_owned();
        assert_eq!(r.regions, ["r:1+2", "r:9", "r:7"]);
        assert_eq!(r.by_place.as_deref(), Some(&["r:9".to_string()][..]));
        // Nothing moves a second time, and the region found by place is still said.
        assert!(matches!(rekeyed(&r, &Toy), Cow::Borrowed(_)));
        // A pick kept as given is taken again as given.
        let largest = Picked { regions: vec!["r:old".into()], pick: Some(RegionPick { given: PickGiven::Largest, keys: vec!["r:old".into()] }), by_place: None };
        let r = rekeyed(&largest, &Toy).into_owned();
        assert_eq!(r.regions, ["r:9"]);
        assert_eq!(r.pick, Some(RegionPick { given: PickGiven::Largest, keys: vec!["r:9".into()] }));
        assert!(matches!(rekeyed(&r, &Toy), Cow::Borrowed(_)));
    }
}
