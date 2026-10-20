//! Complexes and convex pieces (Meshmixer's Edit > Generate / Filter / Split Complex, Convex
//! Decomposition). Meshmixer documents none of the complex tools; this is Polymix's reading.
//!
//! * [`generate_complex_cells`]: the cells of overlapping solids — every region covered by a
//!   particular set of them (for two: A − B, A ∩ B, B − A) — by exact booleans. Each cell
//!   carries its *membership*: a bit mask of the inputs it is inside (A = bit 0, B = bit 1, …).
//!   Polymix keeps a complex as one object whose face groups are the cells, the group id being
//!   the membership mask (so A ∩ B is group 3), each cell a closed shell of its own. Cells
//!   share faces where they touch (the part of B's surface inside A bounds both A − B and
//!   A ∩ B), stored once in each cell. Inputs that touch (coplanar faces) are moved apart by
//!   10⁻⁷ of their size first, so the cells still tile the union exactly.
//! * [`filter_complex`]: keep the cells a test on their membership accepts — a checklist, or a
//!   [`CellFilter`] expression such as `A & B`, `A - B`, `n == 1` (inside exactly one input)
//!   — and join the kept cells into one solid. A face two kept cells share is inside the result
//!   and goes (from both cells); the kept cells' remaining faces are welded where they meet.
//!   Cells that only touch along a curve (A − B and B − A) stay separate shells, so the result
//!   stays manifold. Which cell lies across a face is found geometrically (the shared faces
//!   coincide, facing the other way), so a complex reloaded from a project filters the same.
//! * Split Complex (in the app) puts each cell in its own object.
//! * [`convex_decomposition`]: split a solid by planes until every piece is nearly convex
//!   (its hull adds little volume) and return the pieces' hulls. Each split tries planes
//!   across the piece's principal axes and the world axes, placed at the piece's own vertex
//!   positions (so cuts can run exactly along concave edges of CAD-like parts), and keeps the
//!   one whose halves are the most convex.

use crate::cut::{plane_cut, CutFill, KeepSide, PlaneCutParams};
use crate::{convex_hull, mesh_boolean, BooleanOp, EditError, EditResult};
use glam::DVec3;
use mm_core::{measure::volume, topology, Bvh, Mesh, Plane};
use rayon::prelude::*;

/// Most inputs of a complex (its cells are the non-empty ones of 2ⁿ − 1 combinations).
pub const MAX_COMPLEX_INPUTS: usize = 6;

/// Faces of two cells closer than this (relative to the complex's size) are the same face.
const COINCIDENT: f64 = 1e-8;

/// A cell of a complex: the region inside exactly the inputs whose bits are set in `mask`.
#[derive(Clone, Debug)]
pub struct ComplexCell {
    pub mask: u32,
    pub mesh: Mesh,
}

/// [`generate_complex_cells`] without the memberships.
pub fn generate_complex(solids: &[Mesh]) -> EditResult<Vec<Mesh>> {
    Ok(generate_complex_cells(solids)?.into_iter().map(|c| c.mesh).collect())
}

/// Cells of up to six overlapping closed solids, largest first. Empty cells are left out.
pub fn generate_complex_cells(solids: &[Mesh]) -> EditResult<Vec<ComplexCell>> {
    if !(2..=MAX_COMPLEX_INPUTS).contains(&solids.len()) {
        return Err(EditError::InvalidParameter("Generate Complex needs 2 to 6 closed objects".into()));
    }
    match cells_of(solids, false)? {
        (cells, false) => Ok(cells),
        // Touching inputs (coplanar faces, a vertex on another surface) make each boolean
        // nudge its second operand its own way, so the cells would overlap or leave gaps of
        // that size. Then every input but the first is moved once instead, by a different
        // 10⁻⁷ of the size, so all booleans see the same inputs in general position and the
        // cells tile exactly (where faces were coplanar, a cell gets a sheet that thin).
        (_, true) => Ok(cells_of(solids, true)?.0),
    }
}

/// The cells, largest first, and whether any boolean had to nudge its operand.
fn cells_of(solids: &[Mesh], spread: bool) -> EditResult<(Vec<ComplexCell>, bool)> {
    let n = solids.len();
    let size = solids.iter().fold(mm_core::Aabb::EMPTY, |b, m| b.union(&m.bounds())).diagonal().max(1e-300);
    let dirs = [DVec3::new(0.5773, 0.6181, 0.5331), DVec3::new(-0.4422, 0.7193, 0.5318), DVec3::new(0.6338, -0.3271, 0.7008)];
    let moved: Vec<Mesh> = solids
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut m = m.clone();
            if spread && i > 0 {
                let d = dirs[(i - 1) % 3].normalize() * (1e-7 * size * i as f64);
                m.transform(|p| p + d);
            }
            m
        })
        .collect();
    let mut nudged = false;
    let mut boolean = |a: &Mesh, b: &Mesh, op| -> EditResult<Mesh> {
        let r = mesh_boolean(a, b, op)?;
        nudged |= r.perturbed;
        Ok(r.mesh)
    };
    let mut cells = Vec::new();
    for mask in 1u32..(1 << n) {
        let inside: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let mut cell = moved[inside[0]].clone();
        let mut empty = false;
        for &i in &inside[1..] {
            cell = boolean(&cell, &moved[i], BooleanOp::Intersection)?;
            if cell.triangle_count() == 0 {
                empty = true;
                break;
            }
        }
        if !empty {
            for j in (0..n).filter(|j| mask & (1 << j) == 0) {
                cell = boolean(&cell, &moved[j], BooleanOp::Difference)?;
                if cell.triangle_count() == 0 {
                    break;
                }
            }
        }
        if cell.triangle_count() > 0 && volume(&cell) > 0.0 {
            cells.push(ComplexCell { mask, mesh: cell });
        }
    }
    cells.sort_by(|a, b| volume(&b.mesh).total_cmp(&volume(&a.mesh)));
    Ok((cells, nudged))
}

/// Letter of input `i` in cell labels and filter expressions (A, B, …).
pub fn input_letter(i: usize) -> char {
    (b'A' + i.min(25) as u8) as char
}

/// A cell's membership as text: the inputs it is inside, minus those it is outside, e.g.
/// `A ∩ B − C` (with `names` instead of the letters, if given).
pub fn cell_label(mask: u32, inputs: usize, names: Option<&[String]>) -> String {
    let name = |i: usize| names.and_then(|n| n.get(i)).cloned().unwrap_or_else(|| input_letter(i).to_string());
    let inside: Vec<String> = (0..inputs).filter(|i| mask & (1 << i) != 0).map(name).collect();
    let mut s = inside.join(" ∩ ");
    for o in (0..inputs).filter(|i| mask & (1 << i) == 0) {
        s += " − ";
        s += &name(o);
    }
    s
}

/// A cell of a complex object (see [`complex_cells`]).
#[derive(Clone, Debug)]
pub struct CellInfo {
    pub mask: u32,
    pub triangles: Vec<u32>,
    pub volume: f64,
}

/// The cells of a complex object.
#[derive(Clone, Debug)]
pub struct ComplexInfo {
    /// In membership order.
    pub cells: Vec<CellInfo>,
    /// Number of inputs (the highest input any cell is inside).
    pub inputs: usize,
}

/// The cells of `mesh` as a complex: one closed face group per cell, the group id being the
/// membership mask, as Generate Complex makes it.
pub fn complex_cells(mesh: &Mesh) -> EditResult<ComplexInfo> {
    let not = |why: String| EditError::InvalidParameter(format!("not a complex: {why} (Edit > Generate Complex makes one, a face group per cell)"));
    let mut by: std::collections::BTreeMap<i32, Vec<u32>> = Default::default();
    for t in mesh.triangle_ids() {
        by.entry(mesh.tri_group(t)).or_default().push(t);
    }
    let Some(&last) = by.keys().last() else { return Err(EditError::EmptySelection) };
    let max = (1i32 << MAX_COMPLEX_INPUTS) - 1;
    if let Some(&g) = by.keys().find(|&&g| !(1..=max).contains(&g)) {
        return Err(not(format!("face group {g} is not a cell")));
    }
    let inputs = 32 - (last as u32).leading_zeros() as usize;
    let mut cells = Vec::with_capacity(by.len());
    for (g, triangles) in by {
        let (sub, _) = mesh.submesh(&triangles);
        if !sub.is_closed() {
            return Err(not(format!("cell {} is not closed", cell_label(g as u32, inputs, None))));
        }
        cells.push(CellInfo { mask: g as u32, triangles, volume: volume(&sub) });
    }
    Ok(ComplexInfo { cells, inputs })
}

/// A test on a cell's membership, written as an expression (Filter Complex):
///
/// * `A` … `F` — inside that input; `n` (or `count`) — how many inputs the cell is inside;
///   `all` — inside every input; whole numbers; `true`, `false`.
/// * `!x` (`not`, `~`), `x & y` (`and`, `∩`, `*`), `x - y` (`−`: x and not y), `x ^ y`
///   (`xor`), `x | y` (`or`, `∪`, `+`), comparisons of numbers (`==` or `=`, `!=`, `<`, `<=`,
///   `>`, `>=`) and parentheses. Binding, tightest first: `!`, comparisons, `&`, `-`, `^`,
///   `|`.
///
/// Examples: `A & B` (where A and B overlap), `A - B` (A outside B), `n == 1` (inside exactly
/// one input), `n >= 2` (every overlap), `A ^ B` (inside one of A and B, not both).
#[derive(Clone, Debug, PartialEq)]
pub struct CellFilter {
    expr: Expr,
}

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Input(u32),
    Count,
    All,
    Num(i64),
    Bool(bool),
    Not(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    And,
    Or,
    Xor,
    Minus,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Term(Expr),
    Not,
    Op(Op),
    Open,
    Close,
}

fn tokens(s: &str) -> Result<Vec<Tok>, String> {
    let cs: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        let next = cs.get(i + 1).copied();
        i += 1;
        // Two-character operators.
        let mut pair = |second: char| {
            let hit = next == Some(second);
            if hit {
                i += 1;
            }
            hit
        };
        let tok = match c {
            _ if c.is_whitespace() => continue,
            '(' => Tok::Open,
            ')' => Tok::Close,
            '!' if pair('=') => Tok::Op(Op::Ne),
            '!' | '~' | '¬' => Tok::Not,
            '&' | '∩' | '*' | '·' | '∧' => {
                if c == '&' {
                    pair('&');
                }
                Tok::Op(Op::And)
            }
            '|' | '∪' | '+' | '∨' => {
                if c == '|' {
                    pair('|');
                }
                Tok::Op(Op::Or)
            }
            '^' | '⊕' => Tok::Op(Op::Xor),
            '-' | '−' | '\\' => Tok::Op(Op::Minus),
            '≠' => Tok::Op(Op::Ne),
            '≤' => Tok::Op(Op::Le),
            '≥' => Tok::Op(Op::Ge),
            '=' => {
                pair('=');
                Tok::Op(Op::Eq)
            }
            '<' => Tok::Op(if pair('=') { Op::Le } else { Op::Lt }),
            '>' => Tok::Op(if pair('=') { Op::Ge } else { Op::Gt }),
            _ if c.is_ascii_digit() => {
                let mut v = (c as u8 - b'0') as i64;
                while let Some(d) = cs.get(i).and_then(|d| d.to_digit(10)) {
                    v = v.saturating_mul(10).saturating_add(d as i64);
                    i += 1;
                }
                Tok::Term(Expr::Num(v))
            }
            _ if c.is_alphabetic() => {
                let mut w = c.to_string();
                while let Some(&d) = cs.get(i).filter(|d| d.is_alphanumeric() || **d == '_') {
                    w.push(d);
                    i += 1;
                }
                match w.to_lowercase().as_str() {
                    "and" => Tok::Op(Op::And),
                    "or" => Tok::Op(Op::Or),
                    "xor" => Tok::Op(Op::Xor),
                    "not" => Tok::Not,
                    "n" | "count" => Tok::Term(Expr::Count),
                    "all" => Tok::Term(Expr::All),
                    "true" => Tok::Term(Expr::Bool(true)),
                    "false" => Tok::Term(Expr::Bool(false)),
                    l @ ("a" | "b" | "c" | "d" | "e" | "f") => Tok::Term(Expr::Input((l.as_bytes()[0] - b'a') as u32)),
                    _ => return Err(format!("unknown name \"{w}\" (the inputs are A to F; n counts the inputs a cell is inside)")),
                }
            }
            _ => return Err(format!("unexpected \"{c}\"")),
        };
        out.push(tok);
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.at)
    }

    /// Binary operators of one binding level, left to right.
    fn level(&mut self, ops: &[Op], next: fn(&mut Self) -> Result<Expr, String>) -> Result<Expr, String> {
        let mut e = next(self)?;
        while let Some(&Tok::Op(op)) = self.peek() {
            if !ops.contains(&op) {
                break;
            }
            self.at += 1;
            e = Expr::Bin(op, Box::new(e), Box::new(next(self)?));
        }
        Ok(e)
    }

    fn or(&mut self) -> Result<Expr, String> {
        self.level(&[Op::Or], Self::xor)
    }

    fn xor(&mut self) -> Result<Expr, String> {
        self.level(&[Op::Xor], Self::minus)
    }

    fn minus(&mut self) -> Result<Expr, String> {
        self.level(&[Op::Minus], Self::and)
    }

    fn and(&mut self) -> Result<Expr, String> {
        self.level(&[Op::And], Self::unary)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.peek() == Some(&Tok::Not) {
            self.at += 1;
            return Ok(Expr::Not(Box::new(self.unary()?)));
        }
        self.level(&[Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge], Self::atom)
    }

    fn atom(&mut self) -> Result<Expr, String> {
        let t = self.peek().cloned();
        self.at += 1;
        match t {
            Some(Tok::Term(e)) => Ok(e),
            Some(Tok::Open) => {
                let e = self.or()?;
                if self.peek() != Some(&Tok::Close) {
                    return Err("missing \")\"".into());
                }
                self.at += 1;
                Ok(e)
            }
            Some(Tok::Close) => Err("unexpected \")\"".into()),
            Some(_) => Err("an operator needs something on each side".into()),
            None => Err("the expression ends too early".into()),
        }
    }
}

/// Whether `e` is a condition (true) or a number (false), checking its parts.
fn is_condition(e: &Expr) -> Result<bool, String> {
    let need = |e: &Expr, cond: bool| -> Result<(), String> {
        match is_condition(e)? {
            c if c == cond => Ok(()),
            true => Err("a comparison needs numbers on both sides (like n >= 2)".into()),
            false => Err("&, |, ^, - and ! need conditions (like A or n == 1), not numbers".into()),
        }
    };
    match e {
        Expr::Input(_) | Expr::All | Expr::Bool(_) => Ok(true),
        Expr::Count | Expr::Num(_) => Ok(false),
        Expr::Not(a) => need(a, true).map(|_| true),
        Expr::Bin(op, a, b) => {
            let cmp = matches!(op, Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge);
            need(a, !cmp)?;
            need(b, !cmp)?;
            Ok(true)
        }
    }
}

impl CellFilter {
    pub fn parse(text: &str) -> Result<Self, String> {
        let toks = tokens(text)?;
        if toks.is_empty() {
            return Err("empty expression".into());
        }
        let mut p = Parser { toks, at: 0 };
        let expr = p.or()?;
        if let Some(t) = p.peek() {
            return Err(match t {
                Tok::Close => "unexpected \")\"".into(),
                _ => "two terms without an operator between them (use &, |, -, ^)".into(),
            });
        }
        if !is_condition(&expr)? {
            return Err("the expression is a number; compare it (like n == 1)".into());
        }
        Ok(Self { expr })
    }

    /// Whether a cell inside the inputs of `mask` (out of `inputs`) passes.
    pub fn matches(&self, mask: u32, inputs: usize) -> bool {
        fn num(e: &Expr, mask: u32) -> i64 {
            match e {
                Expr::Count => mask.count_ones() as i64,
                Expr::Num(v) => *v,
                _ => 0,
            }
        }
        fn eval(e: &Expr, mask: u32, inputs: usize) -> bool {
            match e {
                Expr::Input(i) => mask & (1 << i) != 0,
                Expr::All => inputs > 0 && mask.count_ones() as usize >= inputs,
                Expr::Bool(b) => *b,
                Expr::Count | Expr::Num(_) => false,
                Expr::Not(a) => !eval(a, mask, inputs),
                Expr::Bin(op, a, b) => match op {
                    Op::And => eval(a, mask, inputs) && eval(b, mask, inputs),
                    Op::Or => eval(a, mask, inputs) || eval(b, mask, inputs),
                    Op::Xor => eval(a, mask, inputs) != eval(b, mask, inputs),
                    Op::Minus => eval(a, mask, inputs) && !eval(b, mask, inputs),
                    Op::Eq => num(a, mask) == num(b, mask),
                    Op::Ne => num(a, mask) != num(b, mask),
                    Op::Lt => num(a, mask) < num(b, mask),
                    Op::Le => num(a, mask) <= num(b, mask),
                    Op::Gt => num(a, mask) > num(b, mask),
                    Op::Ge => num(a, mask) >= num(b, mask),
                },
            }
        }
        eval(&self.expr, mask, inputs)
    }

    /// The inputs the expression names, as a bit mask.
    pub fn inputs_named(&self) -> u32 {
        fn walk(e: &Expr) -> u32 {
            match e {
                Expr::Input(i) => 1 << i,
                Expr::Not(a) => walk(a),
                Expr::Bin(_, a, b) => walk(a) | walk(b),
                _ => 0,
            }
        }
        walk(&self.expr)
    }
}

/// Face groups of a [`filter_complex`] result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum FilterGroups {
    /// The cell each face came from (its group in the complex: the membership mask).
    Cells,
    /// The input whose surface the face lies on (A = 1, B = 2, …).
    Inputs,
    /// One face group.
    Single,
}

/// Result of [`filter_complex`].
#[derive(Clone, Debug)]
pub struct FilterResult {
    pub mesh: Mesh,
    /// Cells kept.
    pub cells: usize,
    /// Open edges left (none for a clean complex).
    pub open_edges: usize,
}

/// Keep the cells of the complex `mesh` whose membership mask `keep` accepts, joined into one
/// object along the faces they share (see the module docs).
pub fn filter_complex(mesh: &Mesh, keep: &dyn Fn(u32) -> bool, groups: FilterGroups) -> EditResult<FilterResult> {
    let info = complex_cells(mesh)?;
    let cells = &info.cells;
    let kept: Vec<bool> = cells.iter().map(|c| keep(c.mask)).collect();
    if !kept.contains(&true) {
        return Err(EditError::InvalidParameter("no cell is kept".into()));
    }
    let across = cells_across(mesh, cells, &kept);
    // The kept faces as a soup, each cell with its own vertices.
    let mut soup = Soup::default();
    let mut local = vec![u32::MAX; mesh.max_vid()];
    let mut n_cells = 0;
    for (ci, c) in cells.iter().enumerate().filter(|(i, _)| kept[*i]) {
        n_cells += 1;
        for &t in &c.triangles {
            let other = across[t as usize];
            if other.is_some_and(|j| kept[j]) {
                continue;
            }
            let tv = mesh.triangle(t).map(|v| {
                if local[v as usize] == u32::MAX {
                    local[v as usize] = soup.pos.len() as u32;
                    soup.pos.push(mesh.vertex(v));
                }
                local[v as usize]
            });
            soup.tris.push(tv);
            soup.cell.push(ci as u32);
            soup.group.push(match groups {
                FilterGroups::Cells => c.mask as i32,
                FilterGroups::Inputs => {
                    let bits = c.mask ^ other.map_or(0, |j| cells[j].mask);
                    if bits.count_ones() == 1 {
                        bits.trailing_zeros() as i32 + 1
                    } else {
                        0
                    }
                }
                FilterGroups::Single => 0,
            });
        }
    }
    // Zip the kept cells together where the shared faces went.
    soup.zip(COINCIDENT * mesh.bounds().diagonal().max(1e-300));
    let (mut out, _) = Mesh::from_triangles(&soup.pos, &soup.tris, Some(&soup.group));
    out.split_bowtie_vertices();
    let (out, _) = out.compacted();
    let open_edges = out.boundary_edge_ids().count();
    Ok(FilterResult { mesh: out, cells: n_cells, open_edges })
}

/// The cell across each face of the `kept` cells (by triangle id; `None`: the outside).
///
/// A face shared with another cell coincides with one of that cell's faces, facing the other
/// way, to rounding ([`generate_complex_cells`] keeps the booleans from nudging). Slivers
/// too thin to tell a shared face from a neighbour across a seam take the answer of their
/// most parallel neighbour — first of coplanar ones (pieces of the same input triangle), then
/// of ever less parallel ones.
fn cells_across(mesh: &Mesh, cells: &[CellInfo], kept: &[bool]) -> Vec<Option<usize>> {
    let tol = COINCIDENT * mesh.bounds().diagonal().max(1e-300);
    let thin = 10.0 * tol;
    let height = |t: u32| {
        let [a, b, c] = mesh.tri_positions(t);
        2.0 * mesh.tri_area(t) / a.distance(b).max(b.distance(c)).max(c.distance(a)).max(1e-300)
    };
    let mut cell_of = vec![usize::MAX; mesh.max_tid()];
    for (i, c) in cells.iter().enumerate() {
        for &t in &c.triangles {
            cell_of[t as usize] = i;
        }
    }
    let bvhs: Vec<Bvh> = cells.iter().map(|c| Bvh::build_subset(mesh, &c.triangles)).collect();
    let tris: Vec<u32> = cells.iter().zip(kept).filter(|(_, &k)| k).flat_map(|(c, _)| c.triangles.iter().copied()).collect();
    // `Some(answer)` once decided.
    let decided: Vec<Option<Option<usize>>> = tris
        .par_iter()
        .map(|&t| {
            if height(t) <= thin {
                return None;
            }
            let (c, n, me) = (mesh.tri_centroid(t), mesh.tri_normal(t), cell_of[t as usize]);
            let mut best: Option<(f64, usize)> = None;
            for (j, b) in bvhs.iter().enumerate() {
                if j == me {
                    continue;
                }
                b.query_sphere(mesh, c, tol, |o| {
                    if mesh.tri_normal(o).dot(n) < -0.9 {
                        let [p, q, r] = mesh.tri_positions(o);
                        let d = mm_core::geom::closest_point_on_triangle(c, p, q, r).0.distance_squared(c);
                        if best.is_none_or(|b| d < b.0) {
                            best = Some((d, j));
                        }
                    }
                });
            }
            Some(best.map(|b| b.1))
        })
        .collect();
    let mut answer: Vec<Option<Option<usize>>> = vec![None; mesh.max_tid()];
    for (&t, d) in tris.iter().zip(decided) {
        answer[t as usize] = d;
    }
    for min_dot in [1.0 - 1e-9, 1.0 - 1e-6, 0.99, 0.9, 0.5, -1.0] {
        loop {
            let mut changed = false;
            for &t in &tris {
                if answer[t as usize].is_some() {
                    continue;
                }
                let n = mesh.tri_normal(t);
                let from = mesh
                    .tri_neighbors(t)
                    .into_iter()
                    .filter(|&o| o != mm_core::INVALID)
                    .filter_map(|o| Some((mesh.tri_normal(o).dot(n), answer[o as usize]?)))
                    .filter(|x| x.0 >= min_dot)
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|x| x.1);
                if from.is_some() {
                    answer[t as usize] = from;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
    answer.into_iter().map(Option::flatten).collect()
}

/// Faces of several closed cells with the shared ones taken out, to be zipped together.
#[derive(Default)]
struct Soup {
    pos: Vec<DVec3>,
    tris: Vec<[u32; 3]>,
    /// Cell of each triangle.
    cell: Vec<u32>,
    group: Vec<i32>,
}

type Grid = std::collections::HashMap<(i64, i64, i64), Vec<u32>>;

/// An edge to split (its two ends) and the vertices that go on it, with their parameters.
type EdgeSplits = (u32, u32, Vec<(f64, u32)>);

fn grid_key(p: DVec3, size: f64) -> (i64, i64, i64) {
    let q = (p / size).floor();
    (q.x as i64, q.y as i64, q.z as i64)
}

impl Soup {
    /// Directed edges without a twin, as (from, to, triangle).
    fn open_edges(&self) -> Vec<(u32, u32, usize)> {
        let all: std::collections::HashSet<(u32, u32)> = self.tris.iter().flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]).collect();
        let mut out = Vec::new();
        for (i, t) in self.tris.iter().enumerate() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if !all.contains(&(b, a)) {
                    out.push((a, b, i));
                }
            }
        }
        out
    }

    /// Join the open borders of different cells that lie on top of each other (within `tol`).
    /// A border is only ever joined to another cell's (exact booleans leave edges far shorter
    /// than `tol` along their seams, which must not collapse): vertices that are each other's
    /// nearest are welded, and where one border has a vertex the other lacks, the other's edge
    /// is split there first.
    fn zip(&mut self, tol: f64) {
        for _ in 0..16 {
            let open = self.open_edges();
            if open.is_empty() {
                return;
            }
            // The cells whose open borders run through each vertex.
            let mut cells: std::collections::HashMap<u32, smallvec::SmallVec<[u32; 2]>> = Default::default();
            for &(a, b, t) in &open {
                for v in [a, b] {
                    let cs = cells.entry(v).or_default();
                    if !cs.contains(&self.cell[t]) {
                        cs.push(self.cell[t]);
                    }
                }
            }
            let apart = |v: u32, w: u32| cells[&v].iter().all(|c| !cells[&w].contains(c));
            // 1. Weld open vertices of different cells that are each other's nearest.
            let mut grid = Grid::default();
            for &v in cells.keys() {
                grid.entry(grid_key(self.pos[v as usize], tol)).or_default().push(v);
            }
            let nearest = |v: u32| -> Option<u32> {
                let p = self.pos[v as usize];
                let (x, y, z) = grid_key(p, tol);
                let mut best: Option<(f64, u32)> = None;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            for &w in grid.get(&(x + dx, y + dy, z + dz)).into_iter().flatten() {
                                let d = self.pos[w as usize].distance(p);
                                if w != v && d <= tol && apart(v, w) && best.is_none_or(|b| (d, w) < b) {
                                    best = Some((d, w));
                                }
                            }
                        }
                    }
                }
                best.map(|b| b.1)
            };
            let near: std::collections::HashMap<u32, u32> = cells.keys().filter_map(|&v| Some((v, nearest(v)?))).collect();
            let mut rep: Vec<u32> = (0..self.pos.len() as u32).collect();
            let mut welded = 0;
            for (&v, &w) in &near {
                if v < w && near.get(&w) == Some(&v) {
                    rep[w as usize] = v;
                    welded += 1;
                }
            }
            if welded > 0 {
                for t in &mut self.tris {
                    *t = t.map(|x| rep[x as usize]);
                }
                let keep: Vec<bool> = self.tris.iter().map(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0]).collect();
                let mut k = keep.iter();
                self.tris.retain(|_| *k.next().unwrap());
                let mut k = keep.iter();
                self.cell.retain(|_| *k.next().unwrap());
                let mut k = keep.iter();
                self.group.retain(|_| *k.next().unwrap());
                continue;
            }
            // 2. Split the open edges of other cells where an open vertex without a partner
            //    lies on them (one edge per triangle per round).
            let mean = open.iter().map(|&(a, b, _)| self.pos[a as usize].distance(self.pos[b as usize])).sum::<f64>() / open.len() as f64;
            let size = mean.max(4.0 * tol);
            let mut edges = Grid::default();
            for (i, &(a, b, _)) in open.iter().enumerate() {
                let (pa, pb) = (self.pos[a as usize], self.pos[b as usize]);
                let (lo, hi) = (grid_key(pa.min(pb) - DVec3::splat(tol), size), grid_key(pa.max(pb) + DVec3::splat(tol), size));
                if (hi.0 - lo.0 + 1) * (hi.1 - lo.1 + 1) * (hi.2 - lo.2 + 1) > 4096 {
                    continue;
                }
                for x in lo.0..=hi.0 {
                    for y in lo.1..=hi.1 {
                        for z in lo.2..=hi.2 {
                            edges.entry((x, y, z)).or_default().push(i as u32);
                        }
                    }
                }
            }
            // By triangle: the split edge and the vertices on it, with their parameters.
            let mut splits: std::collections::HashMap<usize, EdgeSplits> = Default::default();
            for (&v, cs) in &cells {
                let p = self.pos[v as usize];
                let mut best: Option<(f64, usize, f64)> = None;
                for &i in edges.get(&grid_key(p, size)).into_iter().flatten() {
                    let (a, b, t) = open[i as usize];
                    if a == v || b == v || cs.contains(&self.cell[t]) {
                        continue;
                    }
                    let (pa, pb) = (self.pos[a as usize], self.pos[b as usize]);
                    let d = pb - pa;
                    let s = (p - pa).dot(d) / d.length_squared().max(1e-300);
                    if !(1e-9..=1.0 - 1e-9).contains(&s) {
                        continue;
                    }
                    let dist = (pa + d * s).distance(p);
                    if dist <= tol && best.is_none_or(|b| dist < b.0) {
                        best = Some((dist, i as usize, s));
                    }
                }
                if let Some((_, i, s)) = best {
                    let (a, b, t) = open[i];
                    let e = splits.entry(t).or_insert((a, b, Vec::new()));
                    if (e.0, e.1) == (a, b) {
                        e.2.push((s, v));
                    }
                }
            }
            if splits.is_empty() {
                return;
            }
            for (t, (a, b, mut at)) in splits {
                at.sort_by(|x, y| x.0.total_cmp(&y.0));
                at.dedup_by(|x, y| x.1 == y.1);
                let c = self.tris[t].into_iter().find(|&x| x != a && x != b).unwrap_or(a);
                // a → p1 → … → pk → b, fanned from the opposite corner; each new vertex sits
                // exactly on the vertex that asked for it, so the two weld next round.
                let mut chain = vec![a];
                for &(_, v) in &at {
                    chain.push(self.pos.len() as u32);
                    self.pos.push(self.pos[v as usize]);
                }
                chain.push(b);
                self.tris[t] = [chain[0], chain[1], c];
                for w in chain[1..].windows(2) {
                    self.tris.push([w[0], w[1], c]);
                    self.cell.push(self.cell[t]);
                    self.group.push(self.group[t]);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ConvexParams {
    /// Stop at this many pieces.
    pub max_pieces: usize,
    /// Stop when no piece's hull adds more than this fraction of the whole hull's volume.
    pub concavity: f64,
}

impl Default for ConvexParams {
    fn default() -> Self {
        Self { max_pieces: 16, concavity: 0.02 }
    }
}

struct Piece {
    mesh: Mesh,
    hull: Mesh,
    /// Hull volume minus piece volume.
    concavity: f64,
}

fn piece(mesh: Mesh) -> Option<Piece> {
    let pts: Vec<DVec3> = mesh.vertex_ids().map(|v| mesh.vertex(v)).collect();
    let hull = convex_hull(&pts)?;
    let concavity = (volume(&hull) - volume(&mesh)).max(0.0);
    Some(Piece { mesh, hull, concavity })
}

/// Cut `m` by `plane` into its closed pieces.
fn halves(m: &Mesh, plane: Plane) -> Option<Vec<Mesh>> {
    let mut c = m.clone();
    let r = plane_cut(&mut c, plane, &PlaneCutParams { keep: KeepSide::Both, fill: CutFill::Flat, ..Default::default() }).ok()?;
    if r.cut_loops.is_empty() || r.loop_capped.iter().any(|&k| !k) {
        return None;
    }
    let parts: Vec<Mesh> = topology::connected_components(&c).into_iter().map(|t| c.submesh(&t).0).filter(|p| volume(p) > 0.0).collect();
    (parts.len() >= 2).then_some(parts)
}

/// Best split of a piece, as its new pieces.
fn split(p: &Piece) -> Option<Vec<Piece>> {
    let m = &p.mesh;
    let pts: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).collect();
    let c = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
    let mut cov = [[0.0; 3]; 3];
    for q in &pts {
        mm_core::linalg::add_outer(&mut cov, *q - c, 1.0);
    }
    let (_, axes) = mm_core::linalg::sym_eigen3(cov);
    let mut best: Option<(f64, Vec<Piece>)> = None;
    for n in axes.into_iter().chain([DVec3::X, DVec3::Y, DVec3::Z]) {
        // Distinct vertex positions along the axis (inner ones), thinned to a dozen.
        let mut d: Vec<f64> = pts.iter().map(|q| (*q - c).dot(n)).collect();
        d.sort_by(|a, b| a.total_cmp(b));
        let (lo, hi) = (d[0], d[d.len() - 1]);
        let eps = (hi - lo) * 1e-6;
        d.dedup_by(|a, b| (*a - *b).abs() <= eps);
        let inner: Vec<f64> = d.into_iter().filter(|&x| x > lo + (hi - lo) * 0.05 && x < hi - (hi - lo) * 0.05).collect();
        let step = inner.len().div_ceil(12).max(1);
        for off in inner.into_iter().step_by(step) {
            let Some(parts) = halves(m, Plane::from_point_normal(c + n * off, n)) else { continue };
            let pieces: Vec<Piece> = parts.into_iter().filter_map(piece).collect();
            let score: f64 = pieces.iter().map(|q| q.concavity).sum();
            if pieces.len() >= 2 && best.as_ref().is_none_or(|b| score < b.0) {
                best = Some((score, pieces));
            }
        }
    }
    best.map(|b| b.1)
}

/// Convex hulls of the pieces of `mesh` (see the module docs).
pub fn convex_decomposition(mesh: &Mesh, p: &ConvexParams) -> EditResult<Vec<Mesh>> {
    if !mesh.is_closed() || mesh.triangle_count() == 0 {
        return Err(EditError::InvalidParameter("Convex Decomposition needs a closed object (Make Solid first)".into()));
    }
    let first = piece(mesh.clone()).ok_or_else(|| EditError::Failed("the object is flat or degenerate".into()))?;
    let total = volume(&first.hull).max(1e-300);
    let mut pieces = vec![first];
    let mut stuck = vec![false];
    while pieces.len() < p.max_pieces.max(1) {
        let Some(i) = (0..pieces.len()).filter(|&i| !stuck[i]).max_by(|&a, &b| pieces[a].concavity.total_cmp(&pieces[b].concavity)) else { break };
        if pieces[i].concavity / total <= p.concavity {
            break;
        }
        match split(&pieces[i]) {
            Some(new) => {
                pieces.swap_remove(i);
                stuck.swap_remove(i);
                stuck.extend(new.iter().map(|_| false));
                pieces.extend(new);
            }
            None => stuck[i] = true,
        }
    }
    Ok(pieces.into_iter().map(|q| q.hull).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn two_overlapping_cubes_make_three_cells() {
        let a = primitives::cube(2.0, 1);
        let mut b = primitives::cube(2.0, 1);
        b.transform(|p| p + DVec3::new(1.0, 0.3, 0.2));
        let cells = generate_complex(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(cells.len(), 3);
        for c in &cells {
            c.check_validity().unwrap();
            assert!(c.is_closed());
        }
        // The cells tile the union.
        let sum: f64 = cells.iter().map(volume).sum();
        let union = volume(&mesh_boolean(&a, &b, BooleanOp::Union).unwrap().mesh);
        assert!((sum - union).abs() < 1e-9 * union.max(1.0), "{sum} vs {union}");
        // Disjoint solids: each is its own cell, no intersection.
        let mut far = primitives::cube(1.0, 1);
        far.transform(|p| p + DVec3::X * 10.0);
        assert_eq!(generate_complex(&[a, far]).unwrap().len(), 2);
    }

    /// A complex object as the app keeps it: the cells' shells, one face group per cell (its
    /// membership mask).
    fn complex_object(solids: &[Mesh]) -> Mesh {
        let mut m = Mesh::new();
        for c in generate_complex_cells(solids).unwrap() {
            let start = m.max_tid() as u32;
            m.append_mesh(&c.mesh, 0);
            for t in start..m.max_tid() as u32 {
                if m.is_triangle(t) {
                    m.set_tri_group(t, c.mask as i32);
                }
            }
        }
        m
    }

    #[test]
    fn cell_filters_parse_and_match() {
        let pass = |e: &str, inputs: usize| -> Vec<u32> {
            let f = CellFilter::parse(e).unwrap_or_else(|err| panic!("{e}: {err}"));
            (1..1u32 << inputs).filter(|&m| f.matches(m, inputs)).collect()
        };
        assert_eq!(pass("A & B", 2), vec![3]);
        assert_eq!(pass("a and b", 2), vec![3]);
        assert_eq!(pass("A ∩ B", 2), vec![3]);
        assert_eq!(pass("A - B", 2), vec![1]);
        assert_eq!(pass("A − B", 2), vec![1]);
        assert_eq!(pass("A ^ B", 2), vec![1, 2]);
        assert_eq!(pass("n == 1", 2), vec![1, 2]);
        assert_eq!(pass("n=1", 3), vec![1, 2, 4]);
        assert_eq!(pass("count >= 2", 3), vec![3, 5, 6, 7]);
        assert_eq!(pass("all", 3), vec![7]);
        assert_eq!(pass("A | B", 3), vec![1, 2, 3, 5, 6, 7]);
        assert_eq!(pass("!(A | B)", 3), vec![4]);
        assert_eq!(pass("A & B - C", 3), vec![3]);
        // `-` binds looser than `&`, `^` looser than `-`, `|` loosest.
        assert_eq!(pass("A - B & C", 3), vec![1, 3, 5]);
        assert_eq!(pass("C | A & B", 3), vec![3, 4, 5, 6, 7]);
        assert_eq!(pass("not A and n != 0", 2), vec![2]);
        assert_eq!(CellFilter::parse("A & B - C").unwrap().inputs_named(), 0b111);
        for bad in ["", "A &", "A B", "n", "A & 2", "Z", "(A", "A)", "n >= A", "!3", "A $ B"] {
            assert!(CellFilter::parse(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn filter_complex_of_two_cubes() {
        let a = primitives::cube(2.0, 1);
        let mut b = primitives::cube(2.0, 1);
        b.transform(|p| p + DVec3::new(1.0, 0.3, 0.2));
        let m = complex_object(&[a, b]);
        let info = complex_cells(&m).unwrap();
        assert_eq!((info.inputs, info.cells.len()), (2, 3));
        let both = 1.0 * 1.7 * 1.8;
        let cases = [
            ("A & B", both, 1),
            ("A - B", 8.0 - both, 1),
            ("A ^ B", 2.0 * (8.0 - both), 2),
            ("n == 1", 2.0 * (8.0 - both), 2),
            ("A", 8.0, 1),
            ("A | B", 16.0 - both, 1),
        ];
        for (expr, expected, shells) in cases {
            let f = CellFilter::parse(expr).unwrap();
            let r = filter_complex(&m, &|mask| f.matches(mask, 2), FilterGroups::Cells).unwrap();
            let out = &r.mesh;
            out.check_validity().unwrap_or_else(|e| panic!("{expr}: {e}"));
            assert_eq!(r.open_edges, 0, "{expr}");
            assert!(out.is_closed() && out.is_consistently_oriented(), "{expr}");
            assert!(out.vertex_ids().all(|v| !out.is_bowtie_vertex(v)), "{expr}: manifold vertices");
            let v = volume(out);
            assert!((v - expected).abs() < 1e-9 * 16.0, "{expr}: volume {v} vs {expected}");
            assert_eq!(topology::connected_components(out).len(), shells, "{expr}");
        }
        // Joined along the shared faces: A rebuilt from A − B and A ∩ B has no face inside A.
        let r = filter_complex(&m, &|mask| mask & 1 != 0, FilterGroups::Cells).unwrap();
        for t in r.mesh.triangle_ids() {
            let c = r.mesh.tri_centroid(t);
            assert!((c.abs().max_element() - 1.0).abs() < 1e-9, "face inside A at {c}");
        }
        // Face groups by input surface: A ∩ B is bounded by both surfaces; one group otherwise.
        let r = filter_complex(&m, &|mask| mask == 3, FilterGroups::Inputs).unwrap();
        let groups: std::collections::BTreeSet<i32> = r.mesh.triangle_ids().map(|t| r.mesh.tri_group(t)).collect();
        assert_eq!(groups.into_iter().collect::<Vec<_>>(), vec![1, 2]);
        let r = filter_complex(&m, &|_| true, FilterGroups::Single).unwrap();
        assert!(r.mesh.triangle_ids().all(|t| r.mesh.tri_group(t) == 0));
        // Nothing kept, or not a complex.
        assert!(filter_complex(&m, &|_| false, FilterGroups::Cells).is_err());
        assert!(complex_cells(&primitives::cube(1.0, 1)).is_err(), "group 0 is not a cell");
    }

    #[test]
    fn filter_complex_of_three_spheres() {
        // Three overlapping balls: seven cells; the ones inside at least two make a closed solid.
        let mut balls = Vec::new();
        for (i, c) in [DVec3::ZERO, DVec3::new(0.93, 0.07, 0.11), DVec3::new(0.41, 0.77, 0.19)].into_iter().enumerate() {
            let mut s = primitives::icosphere(1.0, 2 + i % 2);
            s.transform(|p| p + c);
            balls.push(s);
        }
        let m = complex_object(&balls);
        let info = complex_cells(&m).unwrap();
        assert_eq!((info.inputs, info.cells.len()), (3, 7));
        let vol = |pred: &dyn Fn(u32) -> bool| info.cells.iter().filter(|c| pred(c.mask)).map(|c| c.volume).sum::<f64>();
        for expr in ["n >= 2", "A - B - C", "A | B | C", "A ^ B", "(A | B) - C"] {
            let f = CellFilter::parse(expr).unwrap();
            let r = filter_complex(&m, &|mask| f.matches(mask, 3), FilterGroups::Inputs).unwrap();
            r.mesh.check_validity().unwrap_or_else(|e| panic!("{expr}: {e}"));
            assert_eq!(r.open_edges, 0, "{expr}");
            let expected = vol(&|mask| f.matches(mask, 3));
            let v = volume(&r.mesh);
            assert!((v - expected).abs() < 1e-9 * expected.max(1.0), "{expr}: {v} vs {expected}");
        }
    }

    #[test]
    fn filter_complex_of_crossing_tubes() {
        // Three capped tubes crossing like vessels at a junction.
        let tube = |r: f64, c: DVec3, axis: usize| {
            let mut t = primitives::cylinder(r, 4.0, 32, 8, true);
            t.transform(|p| {
                let q = p - DVec3::new(0.0, 0.0, 2.0);
                let q = match axis {
                    0 => DVec3::new(q.z, q.x, q.y),
                    1 => DVec3::new(q.y, q.z, q.x),
                    _ => q,
                };
                q + c
            });
            t
        };
        let m = complex_object(&[tube(0.5, DVec3::ZERO, 0), tube(0.4, DVec3::new(0.3, 0.0, 0.05), 1), tube(0.45, DVec3::new(-0.2, 0.1, 0.0), 2)]);
        let info = complex_cells(&m).unwrap();
        assert_eq!((info.inputs, info.cells.len()), (3, 7));
        for expr in ["n >= 1", "n >= 2", "n == 1", "A", "A - B", "A ^ B", "all"] {
            let f = CellFilter::parse(expr).unwrap();
            let r = filter_complex(&m, &|mask| f.matches(mask, 3), FilterGroups::Cells).unwrap();
            r.mesh.check_validity().unwrap_or_else(|e| panic!("{expr}: {e}"));
            assert_eq!(r.open_edges, 0, "{expr}");
            assert!(r.mesh.vertex_ids().all(|v| !r.mesh.is_bowtie_vertex(v)), "{expr}");
            let expected: f64 = info.cells.iter().filter(|c| f.matches(c.mask, 3)).map(|c| c.volume).sum();
            assert!((volume(&r.mesh) - expected).abs() < 1e-9 * expected.max(1.0), "{expr}");
        }
    }

    #[test]
    fn touching_inputs_still_tile() {
        // Cubes with coplanar faces: the cells still tile the union exactly (no overlaps or
        // gaps from the booleans nudging each its own way), and single cells stay closed.
        let a = primitives::cube(2.0, 2);
        let mut b = primitives::cube(2.0, 2);
        b.transform(|p| p + DVec3::X);
        let cells = generate_complex_cells(&[a, b]).unwrap();
        assert_eq!(cells.len(), 3);
        let sum: f64 = cells.iter().map(|c| volume(&c.mesh)).sum();
        assert!((sum - 12.0).abs() < 1e-5, "{sum}");
        for c in &cells {
            assert!(c.mesh.is_closed());
            assert!((volume(&c.mesh) - 4.0).abs() < 1e-5, "cell {}: {}", c.mask, volume(&c.mesh));
        }
        let m = complex_object(&[primitives::cube(2.0, 2), {
            let mut b = primitives::cube(2.0, 2);
            b.transform(|p| p + DVec3::X);
            b
        }]);
        for mask in [1, 2, 3] {
            let r = filter_complex(&m, &|k| k == mask, FilterGroups::Cells).unwrap();
            assert_eq!(r.open_edges, 0);
        }
        let r = filter_complex(&m, &|k| k & 1 != 0, FilterGroups::Cells).unwrap();
        assert!((volume(&r.mesh) - 8.0).abs() < 1e-5);
    }

    #[test]
    fn an_l_shape_decomposes_into_two_boxes() {
        // L-shaped prism: union of two boxes.
        let mut a = primitives::cube(1.0, 2);
        a.transform(|p| p * DVec3::new(4.0, 1.0, 1.0));
        let mut b = primitives::cube(1.0, 2);
        b.transform(|p| p * DVec3::new(1.0, 4.0, 1.0) + DVec3::new(-1.5, 1.5, 0.0));
        let l = mesh_boolean(&a, &b, BooleanOp::Union).unwrap().mesh;
        let hulls = convex_decomposition(&l, &ConvexParams { max_pieces: 8, concavity: 0.01 }).unwrap();
        assert!((2..=4).contains(&hulls.len()), "{}", hulls.len());
        let total: f64 = hulls.iter().map(volume).sum();
        // Convex pieces cover the shape with little extra volume.
        assert!(total >= volume(&l) * 0.999 && total < volume(&l) * 1.05, "{total} vs {}", volume(&l));
        for h in &hulls {
            assert!(h.is_closed());
        }
        // A convex object stays one piece.
        let ball = primitives::icosphere(1.0, 3);
        assert_eq!(convex_decomposition(&ball, &ConvexParams::default()).unwrap().len(), 1);
    }
}
