// SPDX-License-Identifier: AGPL-3.0-only
//! The arithmetic a number field takes: `25.4/12`, `threadlength + 10`, `sin(30) * r`.
//! Ported from ScanRuler (`plugins/reverse/core/expression.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.
//!
//! A small language of its own — numbers with an optional unit, the parameters by name, the
//! four operations and a power, a few functions — parsed and evaluated here, and kind-checked
//! as it goes: a length and an angle cannot be added, a length times a length is an area and
//! no field wants one. A bare number has no kind of its own until it meets something that
//! has one, or lands in a field that has — `10` is ten millimetres in a length field and ten
//! degrees in an angle field, and `width / 2` is a length because `2` stays a plain scalar
//! beside a length.
//!
//! Every reading of an expression goes through the one tokenizer below — what evaluates it,
//! what finds the names it uses, what rewrites a name — so a parameter renamed is renamed
//! everywhere the same way.
//!
//! Positions (`at`) count characters of the source, which is what ScanRuler's UTF-16 indices
//! are for every character the language reads (`°`, `×`, `÷`, `−` and ASCII).

use std::cell::Cell;

use serde::{Deserialize, Serialize};

use crate::display_units::{length_decimals, to_display, unit_word, DisplayUnits};
use crate::js_number::{fixed_trimmed, js_max, js_min, js_pow, js_round, js_string};

/// What a parameter is, and what a number field wants: a length in millimetres, an angle in
/// degrees, a whole number, or a plain number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParameterKind {
    Length,
    Angle,
    Integer,
    Number,
}

impl ParameterKind {
    /// The kind a field of this kind wants.
    pub fn dim(self) -> Dim {
        dim_of_kind(self)
    }

    /// The kind's id as stored: "length", "angle", "integer", "number".
    pub fn id(self) -> &'static str {
        match self {
            ParameterKind::Length => "length",
            ParameterKind::Angle => "angle",
            ParameterKind::Integer => "integer",
            ParameterKind::Number => "number",
        }
    }
}

/// The kind of a value as exponents of millimetres and degrees — a length is mm¹, an area
/// mm², a ratio mm⁰ — or `Free`: a bare number that takes the kind of whatever it meets.
///
/// The exponents are kept as `f64`, as ScanRuler keeps them, so a power however large never
/// overflows them; they are always whole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dim {
    Of { mm: f64, deg: f64 },
    Free,
}

pub const UNITLESS: Dim = Dim::Of { mm: 0.0, deg: 0.0 };
pub const LENGTH: Dim = Dim::Of { mm: 1.0, deg: 0.0 };
pub const ANGLE: Dim = Dim::Of { mm: 0.0, deg: 1.0 };

/// The kind a field of this kind wants.
pub fn dim_of_kind(kind: ParameterKind) -> Dim {
    match kind {
        ParameterKind::Length => LENGTH,
        ParameterKind::Angle => ANGLE,
        ParameterKind::Integer | ParameterKind::Number => UNITLESS,
    }
}

/// A value as the evaluator hands it on: its number and its kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Value {
    pub value: f64,
    pub dim: Dim,
}

/// What a name in an expression stands for: its value, or why it has none (a parameter that
/// cannot be read).
#[derive(Debug, Clone, PartialEq)]
pub enum ScopeEntry {
    Value(Value),
    Error(String),
}

/// What the names in an expression stand for: an entry, or none for no such name. Any
/// `Fn(&str) -> Option<ScopeEntry>` is a scope, and so is the parameter table.
pub trait Scope {
    fn lookup(&self, name: &str) -> Option<ScopeEntry>;
}

impl<F: Fn(&str) -> Option<ScopeEntry>> Scope for F {
    fn lookup(&self, name: &str) -> Option<ScopeEntry> {
        self(name)
    }
}

/// The scope that knows no names.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoNames;

impl Scope for NoNames {
    fn lookup(&self, _name: &str) -> Option<ScopeEntry> {
        None
    }
}

/// Why an expression has no value, and where in it (character positions, end exclusive).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct ExpressionError {
    pub message: String,
    pub at: (usize, usize),
}

impl ExpressionError {
    fn new(message: impl Into<String>, at: (usize, usize)) -> Self {
        Self { message: message.into(), at }
    }
}

/// An expression evaluated: its value and kind, or why not and where.
pub type Evaluated = Result<Value, ExpressionError>;

/// The units a number may carry, and what one of each is in the workspace's own millimetres
/// and degrees.
fn unit(name: &str) -> Option<Value> {
    let (value, dim) = match name {
        "mm" => (1.0, LENGTH),
        "cm" => (10.0, LENGTH),
        "m" => (1000.0, LENGTH),
        "in" => (25.4, LENGTH),
        "deg" => (1.0, ANGLE),
        "rad" => (180.0 / std::f64::consts::PI, ANGLE),
        _ => return None,
    };
    Some(Value { value, dim })
}

const UNIT_NAMES: [&str; 6] = ["mm", "cm", "m", "in", "deg", "rad"];

fn constant(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        "e" => Some(std::f64::consts::E),
        _ => None,
    }
}

const CONSTANT_NAMES: [&str; 2] = ["pi", "e"];

const FUNCTIONS: [&str; 15] = ["sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sqrt", "abs", "min", "max", "floor", "ceil", "round", "pow"];

fn is_function(name: &str) -> bool {
    FUNCTIONS.contains(&name)
}

/// The names an expression already means something by, which no parameter may take: the
/// units, the constants and the functions.
pub fn reserved_names() -> impl Iterator<Item = &'static str> {
    UNIT_NAMES.into_iter().chain(CONSTANT_NAMES).chain(FUNCTIONS)
}

/// Whether the expressions keep this word for themselves.
pub fn is_reserved(name: &str) -> bool {
    unit(name).is_some() || constant(name).is_some() || is_function(name)
}

fn is_name_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Whether a name may be a parameter's: a letter or an underscore, then letters, digits and
/// underscores — and not a word the language keeps.
pub fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if is_name_start(c) => {}
        _ => return false,
    }
    chars.all(is_name_char) && !is_reserved(name)
}

// ---- tokens --------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Number(f64),
    Name(String),
    /// One of `+ - * / ^ ( ) ,`.
    Op(char),
    End,
}

#[derive(Debug, Clone, PartialEq)]
struct Token {
    tok: Tok,
    at: (usize, usize),
}

/// The length of the number at the start of `rest`, read as
/// `(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?`, or none.
fn number_len(rest: &[char]) -> Option<usize> {
    let digit = |i: usize| rest.get(i).is_some_and(|c| c.is_ascii_digit());
    let mut j = 0;
    if digit(0) {
        while digit(j) {
            j += 1;
        }
        if rest.get(j) == Some(&'.') {
            j += 1;
            while digit(j) {
                j += 1;
            }
        }
    } else if rest.first() == Some(&'.') && digit(1) {
        j = 1;
        while digit(j) {
            j += 1;
        }
    } else {
        return None;
    }
    if matches!(rest.get(j), Some('e') | Some('E')) {
        let mut k = j + 1;
        if matches!(rest.get(k), Some('+') | Some('-')) {
            k += 1;
        }
        if digit(k) {
            while digit(k) {
                k += 1;
            }
            j = k;
        }
    }
    Some(j)
}

/// The length of the name at the start of `rest`, or none.
fn name_len(rest: &[char]) -> Option<usize> {
    match rest.first() {
        Some(&c) if is_name_start(c) => Some(1 + rest[1..].iter().take_while(|&&c| is_name_char(c)).count()),
        _ => None,
    }
}

fn tokenize(src: &[char]) -> Result<Vec<Token>, ExpressionError> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < src.len() {
        let c = src[i];
        if matches!(c, ' ' | '\t' | '\n' | '\r') {
            i += 1;
            continue;
        }
        let rest = &src[i..];
        if let Some(n) = number_len(rest) {
            let text: String = rest[..n].iter().collect();
            let value = text.parse::<f64>().unwrap_or(f64::NAN);
            out.push(Token { tok: Tok::Number(value), at: (i, i + n) });
            i += n;
            continue;
        }
        if let Some(n) = name_len(rest) {
            out.push(Token { tok: Tok::Name(rest[..n].iter().collect()), at: (i, i + n) });
            i += n;
            continue;
        }
        let op = match c {
            '°' => {
                out.push(Token { tok: Tok::Name("deg".to_string()), at: (i, i + 1) });
                i += 1;
                continue;
            }
            '+' | '-' | '*' | '/' | '^' | '(' | ')' | ',' => c,
            '×' | '·' => '*',
            '÷' | ':' => '/',
            '−' => '-',
            _ => {
                let message = if c == ',' { "a comma separates a function’s arguments; the decimal point is a dot".to_string() } else { format!("“{c}” is not part of an expression") };
                return Err(ExpressionError::new(message, (i, i + 1)));
            }
        };
        out.push(Token { tok: Tok::Op(op), at: (i, i + 1) });
        i += 1;
    }
    out.push(Token { tok: Tok::End, at: (src.len(), src.len()) });
    Ok(out)
}

// ---- the tree ------------------------------------------------------------------

#[derive(Debug)]
enum NodeKind {
    Number { value: f64, unit: Option<String> },
    Name(String),
    Neg(Box<Node>),
    Binary { op: char, a: Box<Node>, b: Box<Node> },
    Call { name: String, args: Vec<Node> },
}

#[derive(Debug)]
struct Node {
    kind: NodeKind,
    at: (usize, usize),
}

impl Node {
    fn new(kind: NodeKind, at: (usize, usize)) -> Self {
        Node { kind, at }
    }
}

/// A tree let go of one node at a time: `1 + 2 + … + n` nests n deep down its left side, and
/// dropping it a level of recursion at a time would run out of stack where reading it does
/// not.
impl Drop for Node {
    fn drop(&mut self) {
        let mut below = Vec::new();
        take_children(&mut self.kind, &mut below);
        while let Some(mut n) = below.pop() {
            take_children(&mut n.kind, &mut below);
        }
    }
}

/// The children of a node moved out into `out`, the node left with none.
fn take_children(kind: &mut NodeKind, out: &mut Vec<Node>) {
    match std::mem::replace(kind, NodeKind::Name(String::new())) {
        NodeKind::Neg(arg) => out.push(*arg),
        NodeKind::Binary { a, b, .. } => {
            out.push(*a);
            out.push(*b);
        }
        NodeKind::Call { args, .. } => out.extend(args),
        leaf => *kind = leaf,
    }
}

fn binary_power(op: char) -> u32 {
    match op {
        '+' | '-' => 10,
        '*' | '/' => 20,
        _ => 30,
    }
}

/// Unary minus binds under a power, so −2² is −4, and over a product.
const NEGATE_POWER: u32 = 25;

// ---- how deep a reading may go ------------------------------------------------------------

/// The most stack the reading of expressions may take below where it was first entered on a
/// thread. That covers an expression's brackets, signs, arguments and powers one inside the
/// other, and the parameters a parameter reads, each inside the one reading it. ScanRuler has
/// no bound of its own: JavaScript's stack stops it some thousands deep, where it throws. Here
/// running out of stack aborts the process, so a pathological paste or parameter table is
/// refused first, as "nested too deeply to read". The budget is half of a thread of 2 MiB,
/// Rust's least. It is counted in bytes, not levels, so it holds unoptimised too, where frames
/// are larger and fewer levels fit. Either way it lies far beyond anything typed into a field.
/// A run of operations at one level (`1 + 2 + … + n`) takes no stack per operation and has no
/// bound, as in ScanRuler.
const STACK_BUDGET: usize = 1 << 20;

thread_local! {
    /// Where on this thread's stack the reading of expressions was first entered, while it is.
    static STACK_BASE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The reading of expressions entered on this thread. The outermost entry marks the base the
/// budget counts from and clears it again when it goes, unwinding included.
pub(crate) struct ReadingDepth {
    outermost: bool,
}

impl ReadingDepth {
    pub(crate) fn enter() -> Self {
        let here = stack_position();
        let outermost = STACK_BASE.with(|base| {
            let first = base.get().is_none();
            if first {
                base.set(Some(here));
            }
            first
        });
        ReadingDepth { outermost }
    }
}

impl Drop for ReadingDepth {
    fn drop(&mut self) {
        if self.outermost {
            STACK_BASE.with(|base| base.set(None));
        }
    }
}

/// About where the stack stands: the address of a local of a frame of its own.
#[inline(never)]
fn stack_position() -> usize {
    let marker = 0u8;
    std::hint::black_box(std::ptr::addr_of!(marker)) as usize
}

/// Whether the reading has taken more stack than it may since it was first entered.
pub(crate) fn too_deep() -> bool {
    let here = stack_position();
    STACK_BASE.with(|base| base.get().is_some_and(|b| b.abs_diff(here) > STACK_BUDGET))
}

const TOO_DEEP: &str = "is nested too deeply to read";

struct Parser {
    tokens: Vec<Token>,
    i: usize,
}

impl Parser {
    fn parse(mut self) -> Result<Node, ExpressionError> {
        if self.tokens.first().is_none_or(|t| t.tok == Tok::End) {
            return Err(ExpressionError::new("nothing to read", (0, 0)));
        }
        let node = self.expression(0)?;
        let t = self.peek().clone();
        if t.tok != Tok::End {
            let message = if t.tok == Tok::Op(')') { "a “)” with no “(” before it".to_string() } else { format!("“{}” where an operator was expected", text(&t)) };
            return Err(ExpressionError::new(message, t.at));
        }
        Ok(node)
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.i.min(self.tokens.len() - 1)]
    }

    fn next(&mut self) -> Token {
        let t = self.peek().clone();
        self.i += 1;
        t
    }

    fn expression(&mut self, min_power: u32) -> Result<Node, ExpressionError> {
        if too_deep() {
            return Err(ExpressionError::new(TOO_DEEP, self.peek().at));
        }
        self.operations(min_power)
    }

    fn operations(&mut self, min_power: u32) -> Result<Node, ExpressionError> {
        let mut left = self.prefix()?;
        loop {
            let t = self.peek().clone();
            let op = match t.tok {
                Tok::Op(op) if !matches!(op, '(' | ')' | ',') => op,
                _ => break,
            };
            let power = binary_power(op);
            if power < min_power {
                break;
            }
            self.next();
            // A power binds to the right (2^3^2 is 2^9); the rest to the left.
            let right = self.expression(if op == '^' { power } else { power + 1 })?;
            let at = (left.at.0, right.at.1);
            left = Node::new(NodeKind::Binary { op, a: Box::new(left), b: Box::new(right) }, at);
        }
        Ok(left)
    }

    fn prefix(&mut self) -> Result<Node, ExpressionError> {
        let t = self.next();
        match &t.tok {
            Tok::Number(value) => {
                let value = *value;
                let u = self.peek().clone();
                if let Tok::Name(name) = &u.tok {
                    if unit(name).is_some() {
                        self.next();
                        return Ok(Node::new(NodeKind::Number { value, unit: Some(name.clone()) }, (t.at.0, u.at.1)));
                    }
                }
                Ok(Node::new(NodeKind::Number { value, unit: None }, t.at))
            }
            Tok::Name(name) => {
                let open = self.peek().clone();
                if open.tok == Tok::Op('(') {
                    if !is_function(name) {
                        return Err(ExpressionError::new(format!("there is no function “{name}”"), t.at));
                    }
                    self.next();
                    let mut args = Vec::new();
                    if self.peek().tok != Tok::Op(')') {
                        loop {
                            args.push(self.expression(0)?);
                            if self.peek().tok == Tok::Op(',') {
                                self.next();
                                continue;
                            }
                            break;
                        }
                    }
                    let close = self.next();
                    if close.tok != Tok::Op(')') {
                        return Err(ExpressionError::new(format!("“{name}(” is not closed"), (t.at.0, close.at.1)));
                    }
                    return Ok(Node::new(NodeKind::Call { name: name.clone(), args }, (t.at.0, close.at.1)));
                }
                // A constant takes a unit as a number does: `pi rad` is half a turn.
                if let (Some(value), Tok::Name(u)) = (constant(name), &open.tok) {
                    if unit(u).is_some() {
                        self.next();
                        return Ok(Node::new(NodeKind::Number { value, unit: Some(u.clone()) }, (t.at.0, open.at.1)));
                    }
                }
                Ok(Node::new(NodeKind::Name(name.clone()), t.at))
            }
            Tok::Op('(') => {
                let inner = self.expression(0)?;
                let close = self.next();
                if close.tok != Tok::Op(')') {
                    return Err(ExpressionError::new("a “(” is not closed", (t.at.0, close.at.1)));
                }
                Ok(inner)
            }
            Tok::Op('-') => {
                let arg = self.expression(NEGATE_POWER)?;
                let at = (t.at.0, arg.at.1);
                Ok(Node::new(NodeKind::Neg(Box::new(arg)), at))
            }
            Tok::Op('+') => self.expression(NEGATE_POWER),
            Tok::End => Err(ExpressionError::new("the expression ends where a number was expected", t.at)),
            Tok::Op(_) => Err(ExpressionError::new(format!("“{}” where a number was expected", text(&t)), t.at)),
        }
    }
}

/// A token as a message quotes it.
fn text(t: &Token) -> String {
    match &t.tok {
        Tok::Number(v) => js_string(*v),
        Tok::Name(name) => name.clone(),
        Tok::Op(op) => op.to_string(),
        Tok::End => "the end".to_string(),
    }
}

// ---- kinds ----------------------------------------------------------------------

fn dim_equal(a: Dim, b: Dim) -> bool {
    match (a, b) {
        (Dim::Free, Dim::Free) => true,
        (Dim::Of { mm: am, deg: ad }, Dim::Of { mm: bm, deg: bd }) => am == bm && ad == bd,
        _ => false,
    }
}

/// An exponent as a superscript: none for one, "²" for two, "⁻¹" for minus one.
fn sup(n: f64) -> String {
    if n == 1.0 {
        return String::new();
    }
    js_string(n)
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            d @ '0'..='9' => ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'][usize::from(d as u8 - b'0')],
            other => other,
        })
        .collect()
}

/// A kind in words: "a length", "an angle", "a number", "an area (mm²)".
pub fn describe_dim(d: Dim) -> String {
    let (mm, deg) = match d {
        Dim::Free => return "a number".to_string(),
        Dim::Of { mm, deg } => (mm, deg),
    };
    if mm == 0.0 && deg == 0.0 {
        return "a number".to_string();
    }
    if mm == 1.0 && deg == 0.0 {
        return "a length".to_string();
    }
    if mm == 0.0 && deg == 1.0 {
        return "an angle".to_string();
    }
    if mm == 2.0 && deg == 0.0 {
        return "an area (mm²)".to_string();
    }
    if mm == 3.0 && deg == 0.0 {
        return "a volume (mm³)".to_string();
    }
    let mut parts = Vec::new();
    if mm != 0.0 {
        parts.push(format!("mm{}", sup(mm)));
    }
    if deg != 0.0 {
        parts.push(format!("°{}", sup(deg)));
    }
    format!("a {} quantity", parts.join("·"))
}

/// What a field of this kind wants, in words.
pub fn describe_kind(kind: ParameterKind) -> &'static str {
    match kind {
        ParameterKind::Length => "a length",
        ParameterKind::Angle => "an angle",
        ParameterKind::Integer => "a whole number",
        ParameterKind::Number => "a number",
    }
}

fn combine(a: Dim, b: Dim, f: impl Fn(f64, f64) -> f64) -> Dim {
    match (a, b) {
        (Dim::Free, Dim::Free) => Dim::Free,
        (Dim::Free, Dim::Of { mm, deg }) => Dim::Of { mm: f(0.0, mm), deg: f(0.0, deg) },
        (Dim::Of { mm, deg }, Dim::Free) => Dim::Of { mm: f(mm, 0.0), deg: f(deg, 0.0) },
        (Dim::Of { mm: am, deg: ad }, Dim::Of { mm: bm, deg: bd }) => Dim::Of { mm: f(am, bm), deg: f(ad, bd) },
    }
}

fn near_int(x: f64) -> bool {
    (x - js_round(x)).abs() < 1e-9
}

fn power(base: Dim, exp: f64, at: (usize, usize)) -> Result<Dim, ExpressionError> {
    let (bm, bd) = match base {
        Dim::Free => return Ok(Dim::Free),
        Dim::Of { mm, deg } => (mm, deg),
    };
    let mm = bm * exp;
    let deg = bd * exp;
    if !near_int(mm) || !near_int(deg) {
        return Err(ExpressionError::new(format!("{} cannot be raised to the power {}", describe_dim(base), js_string(exp)), at));
    }
    Ok(Dim::Of { mm: js_round(mm), deg: js_round(deg) })
}

fn same(a: Value, b: Value, what: &str, at: (usize, usize)) -> Result<Dim, ExpressionError> {
    if a.dim == Dim::Free {
        return Ok(b.dim);
    }
    if b.dim == Dim::Free {
        return Ok(a.dim);
    }
    if dim_equal(a.dim, b.dim) {
        return Ok(a.dim);
    }
    Err(ExpressionError::new(format!("{} and {} cannot {}", describe_dim(a.dim), describe_dim(b.dim), what), at))
}

// ---- evaluation -------------------------------------------------------------------

fn evaluate_node(n: &Node, scope: &dyn Scope) -> Evaluated {
    // Deeper here than the reading it came from — a parameter's expression, read for one that
    // reads it — and so counted against the same budget.
    if too_deep() {
        return Err(ExpressionError::new(TOO_DEEP, n.at));
    }
    match &n.kind {
        NodeKind::Number { value, unit: u } => match u.as_deref().and_then(unit) {
            None => Ok(Value { value: *value, dim: Dim::Free }),
            Some(u) => Ok(Value { value: value * u.value, dim: u.dim }),
        },
        NodeKind::Name(name) => {
            if let Some(c) = constant(name) {
                return Ok(Value { value: c, dim: Dim::Free });
            }
            if let Some(u) = unit(name) {
                return Ok(u);
            }
            match scope.lookup(name) {
                None => {
                    let message = if is_function(name) { format!("“{name}” is a function — it takes its argument in parentheses") } else { format!("there is no parameter “{name}”") };
                    Err(ExpressionError::new(message, n.at))
                }
                Some(ScopeEntry::Error(e)) => Err(ExpressionError::new(e, n.at)),
                Some(ScopeEntry::Value(v)) => Ok(v),
            }
        }
        NodeKind::Neg(arg) => {
            let v = evaluate_node(arg, scope)?;
            Ok(Value { value: -v.value, dim: v.dim })
        }
        NodeKind::Binary { .. } => evaluate_chain(n, scope),
        NodeKind::Call { name, args } => call(name, args, n.at, scope),
    }
}

/// A binary operation and every one down its left side — `1 + 2 + … + n` nests n deep there —
/// evaluated in a loop rather than a level of recursion each, in ScanRuler's order: each left
/// side before its right.
fn evaluate_chain(n: &Node, scope: &dyn Scope) -> Evaluated {
    let mut spine: Vec<(char, &Node, (usize, usize))> = Vec::new();
    let mut leftmost = n;
    while let NodeKind::Binary { op, a, b } = &leftmost.kind {
        spine.push((*op, b, leftmost.at));
        leftmost = a;
    }
    let mut a = evaluate_node(leftmost, scope)?;
    while let Some((op, nb, at)) = spine.pop() {
        let b = evaluate_node(nb, scope)?;
        a = binary(op, a, b, at, nb.at)?;
    }
    Ok(a)
}

/// One operation on its two sides' values; `at` is the operation's place, `b_at` its right
/// side's.
fn binary(op: char, a: Value, b: Value, at: (usize, usize), b_at: (usize, usize)) -> Evaluated {
    match op {
        '+' => Ok(Value { value: a.value + b.value, dim: same(a, b, "be added", at)? }),
        '-' => Ok(Value { value: a.value - b.value, dim: same(a, b, "be subtracted", at)? }),
        '*' => Ok(Value { value: a.value * b.value, dim: combine(a.dim, b.dim, |x, y| x + y) }),
        '/' => {
            if b.value == 0.0 {
                return Err(ExpressionError::new("divides by zero", at));
            }
            Ok(Value { value: a.value / b.value, dim: combine(a.dim, b.dim, |x, y| x - y) })
        }
        _ => {
            if b.dim != Dim::Free && !dim_equal(b.dim, UNITLESS) {
                return Err(ExpressionError::new(format!("a power must be a plain number, not {}", describe_dim(b.dim)), b_at));
            }
            Ok(Value { value: js_pow(a.value, b.value), dim: power(a.dim, b.value, at)? })
        }
    }
}

const RAD: f64 = std::f64::consts::PI / 180.0;

fn call(name: &str, nodes: &[Node], at: (usize, usize), scope: &dyn Scope) -> Evaluated {
    let args = nodes.iter().map(|a| evaluate_node(a, scope)).collect::<Result<Vec<_>, _>>()?;
    let want = |count: usize| -> Result<(), ExpressionError> {
        if args.len() != count {
            let takes = if count == 1 { "one argument".to_string() } else { format!("{count} arguments") };
            return Err(ExpressionError::new(format!("{name}() takes {takes}"), at));
        }
        Ok(())
    };
    let angle_in = |v: Value, at: (usize, usize)| -> Result<f64, ExpressionError> {
        if v.dim != Dim::Free && !dim_equal(v.dim, ANGLE) && !dim_equal(v.dim, UNITLESS) {
            return Err(ExpressionError::new(format!("{name}() takes an angle, not {}", describe_dim(v.dim)), at));
        }
        Ok(v.value * RAD)
    };
    let plain = |v: Value, at: (usize, usize)| -> Result<f64, ExpressionError> {
        if v.dim != Dim::Free && !dim_equal(v.dim, UNITLESS) {
            return Err(ExpressionError::new(format!("{name}() takes a plain number, not {}", describe_dim(v.dim)), at));
        }
        Ok(v.value)
    };
    let unitless = |value: f64| Ok(Value { value, dim: UNITLESS });
    let angle = |value: f64| Ok(Value { value, dim: ANGLE });
    match name {
        "sin" => {
            want(1)?;
            unitless(angle_in(args[0], nodes[0].at)?.sin())
        }
        "cos" => {
            want(1)?;
            unitless(angle_in(args[0], nodes[0].at)?.cos())
        }
        "tan" => {
            want(1)?;
            unitless(angle_in(args[0], nodes[0].at)?.tan())
        }
        "asin" => {
            want(1)?;
            angle(plain(args[0], nodes[0].at)?.asin() / RAD)
        }
        "acos" => {
            want(1)?;
            angle(plain(args[0], nodes[0].at)?.acos() / RAD)
        }
        "atan" => {
            want(1)?;
            angle(plain(args[0], nodes[0].at)?.atan() / RAD)
        }
        "atan2" => {
            want(2)?;
            same(args[0], args[1], "be compared in atan2()", at)?;
            angle(args[0].value.atan2(args[1].value) / RAD)
        }
        "sqrt" => {
            want(1)?;
            let v = args[0];
            if v.value < 0.0 {
                return Err(ExpressionError::new("sqrt() of a negative number", at));
            }
            let dim = match v.dim {
                Dim::Free => Dim::Free,
                Dim::Of { mm, deg } => {
                    if mm % 2.0 != 0.0 || deg % 2.0 != 0.0 {
                        return Err(ExpressionError::new(format!("sqrt() of {} is no kind of quantity", describe_dim(v.dim)), at));
                    }
                    Dim::Of { mm: mm / 2.0, deg: deg / 2.0 }
                }
            };
            Ok(Value { value: v.value.sqrt(), dim })
        }
        "abs" => {
            want(1)?;
            Ok(Value { value: args[0].value.abs(), dim: args[0].dim })
        }
        "floor" => {
            want(1)?;
            Ok(Value { value: args[0].value.floor(), dim: args[0].dim })
        }
        "ceil" => {
            want(1)?;
            Ok(Value { value: args[0].value.ceil(), dim: args[0].dim })
        }
        "round" => {
            want(1)?;
            Ok(Value { value: js_round(args[0].value), dim: args[0].dim })
        }
        "min" | "max" => {
            let Some(first) = args.first() else {
                return Err(ExpressionError::new(format!("{name}() takes at least one argument"), at));
            };
            let mut dim = first.dim;
            let mut value = first.value;
            for arg in &args[1..] {
                dim = same(Value { value, dim }, *arg, &format!("be compared in {name}()"), at)?;
                value = if name == "min" { js_min(value, arg.value) } else { js_max(value, arg.value) };
            }
            Ok(Value { value, dim })
        }
        "pow" => {
            want(2)?;
            if args[1].dim != Dim::Free && !dim_equal(args[1].dim, UNITLESS) {
                return Err(ExpressionError::new(format!("a power must be a plain number, not {}", describe_dim(args[1].dim)), nodes[1].at));
            }
            Ok(Value { value: js_pow(args[0].value, args[1].value), dim: power(args[0].dim, args[1].value, at)? })
        }
        _ => Err(ExpressionError::new(format!("there is no function “{name}”"), at)),
    }
}

/// Evaluate an expression against a scope, to whatever kind it comes out as.
pub fn evaluate(src: &str, scope: &dyn Scope) -> Evaluated {
    let _depth = ReadingDepth::enter();
    let chars: Vec<char> = src.chars().collect();
    let node = Parser { tokens: tokenize(&chars)?, i: 0 }.parse()?;
    let v = evaluate_node(&node, scope)?;
    if !v.value.is_finite() {
        return Err(ExpressionError::new("does not come out as a number", node.at));
    }
    Ok(v)
}

/// Evaluate an expression for a field of the given kind: a bare number is read in the
/// field's unit, anything of another kind is refused, and for a whole-number field `whole`
/// asks that the result be one (a parameter of that kind) rather than rounded (a count
/// field, which rounds).
pub fn evaluate_for(src: &str, kind: ParameterKind, scope: &dyn Scope, whole: bool) -> Evaluated {
    let r = evaluate(src, scope)?;
    let want = dim_of_kind(kind);
    let all = (0, src.chars().count());
    if r.dim != Dim::Free && !dim_equal(r.dim, want) {
        let hint = if matches!(kind, ParameterKind::Length | ParameterKind::Angle) && dim_equal(r.dim, UNITLESS) {
            format!(" — a number with a unit, {}, makes it one", if kind == ParameterKind::Length { "“10 mm”" } else { "“10 deg”" })
        } else {
            String::new()
        };
        return Err(ExpressionError::new(format!("comes out as {}; {} is wanted here{}", describe_dim(r.dim), describe_kind(kind), hint), all));
    }
    if kind == ParameterKind::Integer && whole && !near_int(r.value) {
        return Err(ExpressionError::new(format!("comes out {}, not whole — round(), floor() or ceil() make it so", format_number(r.value, 3)), all));
    }
    Ok(Value { value: r.value, dim: want })
}

// ---- reading an expression without evaluating it ------------------------------------

/// The tokens of an expression, or — where the tokenizer refuses it — every identifier in it
/// and every "(", so a rename or a count of users does not miss one.
fn tokens_for_reading(chars: &[char]) -> Vec<Token> {
    tokenize(chars).unwrap_or_else(|_| tokenize_loosely(chars))
}

/// The names of an expression the tokenizer refuses: every identifier in it, so a rename or
/// a count of users does not miss one.
fn tokenize_loosely(src: &[char]) -> Vec<Token> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < src.len() {
        if src[i] == '(' {
            out.push(Token { tok: Tok::Op('('), at: (i, i + 1) });
            i += 1;
        } else if let Some(n) = name_len(&src[i..]) {
            out.push(Token { tok: Tok::Name(src[i..i + n].iter().collect()), at: (i, i + n) });
            i += n;
        } else {
            i += 1;
        }
    }
    out.push(Token { tok: Tok::End, at: (src.len(), src.len()) });
    out
}

/// The parameter names an expression uses, each once, in order of first use — read off the
/// tokens, so a broken expression still tells.
pub fn references(src: &str) -> Vec<String> {
    let chars: Vec<char> = src.chars().collect();
    let tokens = tokens_for_reading(&chars);
    let mut out: Vec<String> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let Tok::Name(name) = &t.tok else { continue };
        if is_reserved(name) {
            continue;
        }
        if tokens.get(i + 1).is_some_and(|n| n.tok == Tok::Op('(')) {
            continue;
        }
        if !out.contains(name) {
            out.push(name.clone());
        }
    }
    out
}

/// The expression with every use of a parameter named `from` written as `to` — whole names
/// only, so `thread` inside `threadlength` is left be — and everything else kept exactly as
/// typed.
pub fn rename_in(src: &str, from: &str, to: &str) -> String {
    if from == to {
        return src.to_string();
    }
    let chars: Vec<char> = src.chars().collect();
    let tokens = tokens_for_reading(&chars);
    let mut out = String::new();
    let mut last = 0;
    for (i, t) in tokens.iter().enumerate() {
        match &t.tok {
            Tok::Name(name) if name == from => {}
            _ => continue,
        }
        if tokens.get(i + 1).is_some_and(|n| n.tok == Tok::Op('(')) {
            continue;
        }
        out.extend(&chars[last..t.at.0]);
        out.push_str(to);
        last = t.at.1;
    }
    out.extend(&chars[last..]);
    out
}

/// Whether an expression is a bare number and nothing more — what is stored as a number
/// alone, with no expression kept.
pub fn is_plain_number(src: &str) -> bool {
    plain_number(src).is_some()
}

/// The number a plain-number expression is, or none. What does not even read as tokens is
/// no plain number.
pub fn plain_number(src: &str) -> Option<f64> {
    let chars: Vec<char> = src.chars().collect();
    match tokenize(&chars).ok()?.as_slice() {
        [Token { tok: Tok::Number(v), .. }, _] => Some(*v),
        _ => None,
    }
}

// ---- words ---------------------------------------------------------------------

/// A number with no zeros carried behind it, to `places` places (ScanRuler's default three).
pub fn format_number(value: f64, places: usize) -> String {
    fixed_trimmed(value, places)
}

/// A value with its unit, as a row reads it: "35 mm", "30°", "6", "0.5" — a length in the
/// units shown, to a micron in millimetres.
pub fn format_value(value: f64, kind: ParameterKind, units: DisplayUnits) -> String {
    match kind {
        ParameterKind::Length => format!("{} {}", format_number(to_display(value, units), length_decimals(3, units)), unit_word(units, 1)),
        ParameterKind::Angle => format!("{}°", format_number(value, 2)),
        ParameterKind::Integer => js_string(js_round(value)),
        ParameterKind::Number => format_number(value, 4),
    }
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reExpression.test.ts`: what the
    //! language reads, what it refuses and how it says so, and the two readings that do not
    //! evaluate — the names an expression uses, and a name rewritten. The parameter's value
    //! test of `reCoreUnits.test.ts` closes it.
    use super::*;
    use std::collections::HashMap;

    fn scope(vars: &[(&str, Value)]) -> impl Fn(&str) -> Option<ScopeEntry> {
        let map: HashMap<String, Value> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        move |name: &str| map.get(name).map(|v| ScopeEntry::Value(*v))
    }
    fn mm(value: f64) -> Value {
        Value { value, dim: LENGTH }
    }
    fn deg(value: f64) -> Value {
        Value { value, dim: ANGLE }
    }
    fn num(value: f64) -> Value {
        Value { value, dim: UNITLESS }
    }

    fn value_in(src: &str, s: &dyn Scope) -> f64 {
        match evaluate(src, s) {
            Ok(v) => v.value,
            Err(e) => panic!("{src}: {}", e.message),
        }
    }
    fn value(src: &str) -> f64 {
        value_in(src, &NoNames)
    }
    fn error_in(src: &str, s: &dyn Scope) -> String {
        match evaluate(src, s) {
            Ok(v) => panic!("{src} evaluated to {}", v.value),
            Err(e) => e.message,
        }
    }
    fn error(src: &str) -> String {
        error_in(src, &NoNames)
    }
    fn close(a: f64, b: f64, digits: i32) -> bool {
        // vitest's toBeCloseTo: |a − b| < 10^−digits / 2.
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn reads_numbers_the_four_operations_and_a_power_in_the_usual_order() {
        assert_eq!(value("2 + 3 * 4"), 14.0);
        assert_eq!(value("(2 + 3) * 4"), 20.0);
        assert!(close(value("25.4 / 12"), 2.11667, 5));
        assert_eq!(value("10 - 2 - 3"), 5.0);
        assert_eq!(value("24 / 4 / 2"), 3.0);
        assert_eq!(value("2 ^ 3 ^ 2"), 512.0);
        assert_eq!(value(".5 + 1e1"), 10.5);
    }

    #[test]
    fn binds_unary_minus_under_a_power_and_over_a_product() {
        assert_eq!(value("-2 ^ 2"), -4.0);
        assert_eq!(value("(-2) ^ 2"), 4.0);
        assert_eq!(value("2 ^ -1"), 0.5);
        assert_eq!(value("-2 * -3"), 6.0);
        assert_eq!(value("+4"), 4.0);
        assert_eq!(value("--4"), 4.0);
    }

    // The figures are ScanRuler's, to the places its test reads them.
    #[allow(clippy::approx_constant)]
    #[test]
    fn knows_the_functions_in_degrees() {
        assert!(close(value("sin(30)"), 0.5, 2));
        assert!(close(value("cos(60 deg)"), 0.5, 2));
        assert!(close(value("tan(45)"), 1.0, 2));
        assert!(close(value("asin(0.5)"), 30.0, 2));
        assert!(close(value("acos(0.5)"), 60.0, 2));
        assert!(close(value("atan(1)"), 45.0, 2));
        assert!(close(value("atan2(1, 1)"), 45.0, 2));
        assert_eq!(value("sqrt(16)"), 4.0);
        assert_eq!(value("abs(-3)"), 3.0);
        assert_eq!(value("min(3, 1, 2)"), 1.0);
        assert_eq!(value("max(3, 1, 2)"), 3.0);
        assert_eq!(value("floor(2.7)"), 2.0);
        assert_eq!(value("ceil(2.2)"), 3.0);
        assert_eq!(value("round(2.5)"), 3.0);
        assert_eq!(value("pow(2, 10)"), 1024.0);
        assert!(close(value("2 * pi"), 6.2832, 4));
        assert!(close(value("e"), 2.7183, 4));
    }

    #[test]
    fn reads_units_into_millimetres_and_degrees() {
        assert!(close(value("1 in"), 25.4, 2));
        assert!(close(value("1in"), 25.4, 2));
        assert_eq!(value("2 cm"), 20.0);
        assert_eq!(value("0.5 m"), 500.0);
        assert_eq!(value("10 mm"), 10.0);
        assert!(close(value("pi rad"), 180.0, 2));
        assert_eq!(value("30°"), 30.0);
        assert_eq!(evaluate("1 in", &NoNames).map(|v| v.dim), Ok(LENGTH));
        assert_eq!(evaluate("1 rad", &NoNames).map(|v| v.dim), Ok(ANGLE));
    }

    #[test]
    fn reads_parameters_from_the_scope() {
        let s = scope(&[("width", mm(40.0)), ("count", num(4.0))]);
        assert_eq!(value_in("width / 2", &s), 20.0);
        assert_eq!(value_in("width + 10", &s), 50.0);
        assert_eq!(value_in("width / count", &s), 10.0);
        assert_eq!(value_in("threadlength+10", &scope(&[("threadlength", mm(25.0))])), 35.0);
    }

    #[test]
    fn refusals_say_what_is_wrong_and_where() {
        assert_eq!(evaluate("", &NoNames).map_err(|e| e.message), Err("nothing to read".to_string()));
        assert_eq!(evaluate("2 +", &NoNames).map_err(|e| e.at), Err((3, 3)));
        assert!(error("2 + * 3").contains("where a number was expected"));
        assert!(error("(2 + 3").contains("not closed"));
        assert!(error("2 + 3)").contains("“)”"));
        assert!(error("2 3").contains("where an operator was expected"));
        assert!(error("2 # 3").contains("“#”"));
        assert_eq!(evaluate("width", &NoNames), Err(ExpressionError { message: "there is no parameter “width”".to_string(), at: (0, 5) }));
        assert!(error("foo(2)").contains("no function “foo”"));
        assert!(error("sin").contains("is a function"));
        assert!(error("sin(1, 2)").contains("takes one argument"));
        assert_eq!(error("1 / 0"), "divides by zero");
        assert!(error("sqrt(-1)").contains("negative"));
        assert!(error("2,5").contains("operator was expected"));
    }

    #[test]
    fn refusals_keep_lengths_angles_and_numbers_apart() {
        let s = scope(&[("width", mm(40.0)), ("height", mm(20.0)), ("turn", deg(30.0)), ("count", num(4.0))]);
        let dim = |src: &str| evaluate(src, &s).map(|v| v.dim);
        assert_eq!(error_in("width + turn", &s), "a length and an angle cannot be added");
        assert_eq!(error_in("width - count", &s), "a length and a number cannot be subtracted");
        assert_eq!(dim("width * height"), Ok(Dim::Of { mm: 2.0, deg: 0.0 }));
        assert_eq!(dim("width / height"), Ok(UNITLESS));
        assert_eq!(dim("width * count"), Ok(LENGTH));
        assert_eq!(dim("sin(turn) * width"), Ok(LENGTH));
        assert_eq!(dim("atan(width / height)"), Ok(ANGLE));
        assert_eq!(dim("sqrt(width * height)"), Ok(LENGTH));
        assert!(error_in("sqrt(width)", &s).contains("no kind of quantity"));
        assert_eq!(error_in("sin(width)", &s), "sin() takes an angle, not a length");
        assert_eq!(error_in("asin(width)", &s), "asin() takes a plain number, not a length");
        assert!(error_in("width ^ 0.5", &s).contains("cannot be raised"));
        assert_eq!(dim("width ^ 2"), Ok(Dim::Of { mm: 2.0, deg: 0.0 }));
        assert!(error_in("2 ^ width", &s).contains("a power must be a plain number"));
        assert!(error_in("min(width, turn)", &s).contains("cannot be compared in min()"));
        assert!(error_in("atan2(width, count)", &s).contains("atan2()"));
        assert_eq!(evaluate("max(width, 10)", &s), Ok(Value { value: 40.0, dim: LENGTH }));
    }

    #[test]
    fn refusals_refuse_what_does_not_come_out_as_a_number() {
        assert!(error("10 ^ 400").contains("does not come out"));
    }

    fn field() -> impl Fn(&str) -> Option<ScopeEntry> {
        scope(&[("width", mm(40.0)), ("turn", deg(30.0)), ("count", num(4.0)), ("ratio", num(0.5))])
    }
    fn refused(r: Evaluated) -> String {
        match r {
            Ok(v) => panic!("evaluated to {}", v.value),
            Err(e) => e.message,
        }
    }

    #[test]
    fn evaluate_for_reads_a_bare_number_in_the_fields_unit() {
        let s = field();
        assert_eq!(evaluate_for("10", ParameterKind::Length, &s, false), Ok(Value { value: 10.0, dim: LENGTH }));
        assert_eq!(evaluate_for("10", ParameterKind::Angle, &s, false), Ok(Value { value: 10.0, dim: ANGLE }));
        assert_eq!(evaluate_for("25.4/12", ParameterKind::Length, &s, false).map(|v| v.dim), Ok(LENGTH));
        assert_eq!(evaluate_for("2*3", ParameterKind::Integer, &s, false).map(|v| v.value), Ok(6.0));
    }

    #[test]
    fn evaluate_for_takes_the_fields_own_kind_and_refuses_another() {
        let s = field();
        let v = |src: &str, kind| evaluate_for(src, kind, &s, false).map(|v| v.value);
        let e = |src: &str, kind| refused(evaluate_for(src, kind, &s, false));
        assert_eq!(v("width / 2", ParameterKind::Length), Ok(20.0));
        assert_eq!(v("width + 10", ParameterKind::Length), Ok(50.0));
        assert_eq!(v("turn * 2", ParameterKind::Angle), Ok(60.0));
        assert_eq!(e("width", ParameterKind::Angle), "comes out as a length; an angle is wanted here");
        assert_eq!(e("count", ParameterKind::Length), "comes out as a number; a length is wanted here — a number with a unit, “10 mm”, makes it one");
        assert_eq!(v("count * 2 mm", ParameterKind::Length), Ok(8.0));
        assert_eq!(e("width * width", ParameterKind::Length), "comes out as an area (mm²); a length is wanted here");
        assert_eq!(e("width", ParameterKind::Number), "comes out as a length; a number is wanted here");
        assert_eq!(v("width / 10 mm", ParameterKind::Integer), Ok(4.0));
        assert_eq!(v("ratio * 2", ParameterKind::Number), Ok(1.0));
    }

    #[test]
    fn evaluate_for_asks_a_whole_number_parameter_to_come_out_whole_and_lets_a_count_field_round() {
        let s = field();
        assert_eq!(refused(evaluate_for("7.3", ParameterKind::Integer, &s, true)), "comes out 7.3, not whole — round(), floor() or ceil() make it so");
        assert_eq!(evaluate_for("round(7.3)", ParameterKind::Integer, &s, true).map(|v| v.value), Ok(7.0));
        assert_eq!(evaluate_for("7.3", ParameterKind::Integer, &s, false).map(|v| v.value), Ok(7.3));
        assert!(evaluate_for("6.0000000001", ParameterKind::Integer, &s, true).is_ok());
    }

    #[test]
    fn finds_the_names_an_expression_uses_once_each_in_order() {
        assert_eq!(references("width + height * width"), ["width", "height"]);
        assert_eq!(references("sin(turn) * 2 mm + pi"), ["turn"]);
        assert!(references("25.4 / 12").is_empty());
        assert_eq!(references("width +"), ["width"]);
        assert_eq!(references("width # height"), ["width", "height"]);
    }

    #[test]
    fn renames_a_parameter_and_leaves_the_rest_as_typed() {
        assert_eq!(rename_in("thread + threadlength*2", "thread", "pitch"), "pitch + threadlength*2");
        assert_eq!(rename_in("width/2 + width", "width", "w"), "w/2 + w");
        assert_eq!(rename_in("sin(width)", "sin", "cos"), "sin(width)");
        assert_eq!(rename_in("width +", "width", "w"), "w +");
        assert_eq!(rename_in("a + b", "c", "d"), "a + b");
    }

    #[test]
    fn tells_a_plain_number_from_an_expression() {
        assert!(is_plain_number("25"));
        assert!(is_plain_number(" 2.5 "));
        assert!(!is_plain_number("25 mm"));
        assert!(!is_plain_number("-25"));
        assert!(!is_plain_number("25/2"));
        assert!(!is_plain_number("width"));
        assert_eq!(plain_number("12.5"), Some(12.5));
        assert_eq!(plain_number("12.5/2"), None);
    }

    #[test]
    fn judges_a_parameter_name() {
        assert!(valid_name("width"));
        assert!(valid_name("_w2"));
        assert!(!valid_name("2w"));
        assert!(!valid_name("thread length"));
        assert!(!valid_name("sin"));
        assert!(!valid_name("mm"));
        assert!(!valid_name("pi"));
        assert!(!valid_name(""));
    }

    #[test]
    fn writes_a_value_in_its_unit() {
        let mm = DisplayUnits::Mm;
        assert_eq!(format_value(35.0, ParameterKind::Length, mm), "35 mm");
        assert_eq!(format_value(2.116666, ParameterKind::Length, mm), "2.117 mm");
        assert_eq!(format_value(30.0, ParameterKind::Angle, mm), "30°");
        assert_eq!(format_value(6.9, ParameterKind::Integer, mm), "7");
        assert_eq!(format_value(0.5, ParameterKind::Number, mm), "0.5");
    }

    /// `reCoreUnits.test.ts`: a parameter's value — a length in inches, no zeros behind it;
    /// other kinds as they are.
    #[test]
    fn a_parameters_value_in_inches() {
        let inch = DisplayUnits::In;
        assert_eq!(format_value(25.4, ParameterKind::Length, inch), "1 in");
        assert_eq!(format_value(35.0, ParameterKind::Length, inch), "1.378 in");
        assert_eq!(format_value(30.0, ParameterKind::Angle, inch), "30°");
        assert_eq!(format_value(6.0, ParameterKind::Integer, inch), "6");
        assert_eq!(format_value(35.0, ParameterKind::Length, DisplayUnits::Mm), "35 mm");
    }

    /// Not in ScanRuler's tests: the guards a port adds or keeps — a stack budget in place of
    /// a blown stack (brackets, signs, calls and powers, one inside the other), a long run of
    /// operations read in full as ScanRuler reads it, NaN kept through min() as JavaScript
    /// keeps it, and the reserved words listed. Run on a thread of 2 MiB, Rust's least, so an
    /// unoptimised build shows the budget holds there too.
    #[test]
    fn port_guards() {
        let small = std::thread::Builder::new().stack_size(2 << 20).spawn(|| {
            for deep in [format!("{}1{}", "(".repeat(20_000), ")".repeat(20_000)), format!("{}1", "-".repeat(20_000)), format!("{}1{}", "abs(".repeat(20_000), ")".repeat(20_000)), vec!["2"; 20_000].join("^")] {
                assert_eq!(error(&deep), "is nested too deeply to read", "{}", &deep[..12]);
            }
            // Far deeper than anything typed is read, unoptimised too.
            let fits = format!("{}1{}", "abs(".repeat(50), ")".repeat(50));
            assert_eq!(value(&fits), 1.0);
            let long = vec!["1"; 20_000].join("+");
            assert_eq!(value(&long), 20_000.0);
            let mixed = format!("{} - 1", vec!["2 * 3"; 5000].join(" + "));
            assert_eq!(value(&mixed), 29_999.0);
        });
        assert!(small.is_ok_and(|h| h.join().is_ok()));
        // The budget is the reading's own: entered and left, it leaves nothing behind.
        assert_eq!(value("((((1))))"), 1.0);
        assert!(error("min(asin(2), 1)").contains("does not come out"));
        assert_eq!(reserved_names().count(), 23);
        assert!(reserved_names().all(is_reserved));
        assert_eq!(error("1.5.3"), "“0.3” where an operator was expected");
        assert_eq!(evaluate("2 × 3 ÷ 4 − 1", &NoNames).map(|v| v.value), Ok(0.5));
        assert_eq!(evaluate("1 mm ^ 4 / 1 mm ^ 4", &NoNames).map(|v| v.dim), Ok(UNITLESS));
        assert_eq!(error_in("width ^ 2 * turn ^ -1 + width", &scope(&[("width", mm(1.0)), ("turn", deg(1.0))])), "a mm²·°⁻¹ quantity and a length cannot be added");
    }
}
