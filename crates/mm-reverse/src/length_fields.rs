// SPDX-License-Identifier: AGPL-3.0-only
//! Lengths in the workspace's boxes, rows and number fields, in the display units. Ported from
//! ScanRuler (`plugins/reverse/ui/reLength.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! The model holds millimetres — features, drafts, parameters and the expressions kept beside
//! them — and keeps them so: what is shown converts on the way out and what is typed on the
//! way in, and a project saved with either unit on screen builds the same.
//!
//! Typed text in inches, for a length:
//! - a number, signed or not, is the figure in inches: it is held as its millimetres, with no
//!   expression kept (a parameter, whose expression is all it holds, keeps it as `0.5 in`,
//!   which the language reads as 12.7 mm);
//! - arithmetic of bare numbers and nothing else (`1/4`, `0.5 + 0.125`) is also a figure in
//!   inches: it is kept as `(1/4) * 1 in`, which reads the same in millimetres wherever it is
//!   evaluated again;
//! - anything that names a parameter or carries a unit (`width + 5`, `2 * 1 in`) is kept as
//!   typed and read as the language always reads it — a bare number beside a length is
//!   millimetres there, so `width + 0.5` adds half a millimetre and an inch is written
//!   `0.5 in`. Converting those literals would need the evaluator to say which of them became
//!   lengths, and an expression kept with inches baked into bare numbers would build otherwise
//!   when evaluated as millimetres.

use crate::display_units::{from_display, length_decimals, length_input_text, to_display, unit_word, DisplayUnits};
use crate::expression::{evaluate, evaluate_for, is_plain_number, Dim, NoNames, ParameterKind, Scope};
use crate::js_number::{js_round, js_string};

/// A number to `places` decimals, with no zeros carried behind it.
fn rounded(v: f64, places: usize) -> f64 {
    let scale = 10f64.powi(i32::try_from(places).unwrap_or(i32::MAX));
    js_round(v * scale) / scale
}

/// A length as a number in the units shown, without its unit: to `mm_places` decimals in
/// millimetres, one more (at least three) in inches, no zeros carried — "12.5", "0.492".
/// ScanRuler's default is three places.
pub fn len_bare(mm: f64, units: DisplayUnits, mm_places: usize) -> String {
    js_string(rounded(to_display(mm, units), length_decimals(mm_places, units)))
}

/// A length as a row reads it, with its unit: "12.5 mm", "0.492 in". ScanRuler's default is
/// two places.
pub fn len_short(mm: f64, units: DisplayUnits, mm_places: usize) -> String {
    format!("{} {}", len_bare(mm, units, mm_places), unit_word(units, 1))
}

/// The number a field shows for what it holds, while no expression is set: a length in the
/// units shown.
pub fn field_text(value: f64, kind: ParameterKind, places: usize, units: DisplayUnits) -> String {
    if kind != ParameterKind::Length {
        return js_string(rounded(value, places));
    }
    len_bare(value, units, places)
}

/// A bare number with a sign before it or none: "-5", "+0.25" —
/// `^[+-]?\s*(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$`.
fn signed_number(t: &str) -> bool {
    let c: Vec<char> = t.chars().collect();
    let digit = |i: usize| c.get(i).is_some_and(char::is_ascii_digit);
    let mut i = 0;
    if matches!(c.first(), Some('+') | Some('-')) {
        i = 1;
    }
    while c.get(i).is_some_and(|ch| ch.is_whitespace()) {
        i += 1;
    }
    if digit(i) {
        while digit(i) {
            i += 1;
        }
        if c.get(i) == Some(&'.') {
            i += 1;
            while digit(i) {
                i += 1;
            }
        }
    } else if c.get(i) == Some(&'.') && digit(i + 1) {
        i += 1;
        while digit(i) {
            i += 1;
        }
    } else {
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

/// Whether the text is arithmetic of bare numbers and nothing else — no parameter, no unit: a
/// figure the person means in the units shown.
fn bare_arithmetic(text: &str) -> bool {
    evaluate(text, &NoNames).is_ok_and(|r| r.dim == Dim::Free)
}

/// Bare arithmetic kept so that it reads in inches wherever it is evaluated again:
/// `(1/4) * 1 in`.
fn in_inches(text: &str) -> String {
    format!("({text}) * 1 in")
}

/// A field's text read: the value to hold, in millimetres for a length, and the expression to
/// keep beside it — none for a plain number.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldRead {
    pub value: f64,
    pub expr: Option<String>,
}

/// A field's text read for a field of `kind`, or why it will not do. In millimetres, and for
/// every field that is not a length, it is what the field always did; in inches see the head of
/// this module.
pub fn read_field(text: &str, kind: ParameterKind, scope: &dyn Scope, units: DisplayUnits) -> Result<FieldRead, String> {
    let t = text.trim();
    let r = evaluate_for(t, kind, scope, false).map_err(|e| e.message)?;
    if kind != ParameterKind::Length || units == DisplayUnits::Mm {
        return Ok(FieldRead { value: r.value, expr: (!is_plain_number(t)).then(|| t.to_string()) });
    }
    if signed_number(t) {
        return Ok(FieldRead { value: from_display(r.value, units), expr: None });
    }
    if bare_arithmetic(t) {
        return Ok(FieldRead { value: from_display(r.value, units), expr: Some(in_inches(t)) });
    }
    Ok(FieldRead { value: r.value, expr: Some(t.to_string()) })
}

/// A parameter's expression as its row shows it: a length given as a plain number is
/// millimetres, shown in the units shown; anything else as it was typed.
pub fn parameter_text(expression: &str, kind: ParameterKind, units: DisplayUnits) -> String {
    let t = expression.trim();
    if kind != ParameterKind::Length || units == DisplayUnits::Mm || !signed_number(t) {
        return expression.to_string();
    }
    let figure: String = t.chars().filter(|c| !c.is_whitespace()).collect();
    length_input_text(figure.parse::<f64>().unwrap_or(f64::NAN), units)
}

/// What a parameter's expression is kept as, from what was typed into its row — the same text
/// but where a length's figure was typed in inches.
pub fn parameter_expression(typed: &str, kind: ParameterKind, units: DisplayUnits) -> String {
    let t = typed.trim();
    if kind != ParameterKind::Length || units == DisplayUnits::Mm || t.is_empty() {
        return typed.to_string();
    }
    if signed_number(t) {
        return format!("{t} in");
    }
    if bare_arithmetic(t) {
        return in_inches(t);
    }
    typed.to_string()
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reUiUnits.test.ts`: the workspace's
    //! lengths in the display units — the text a row and a field show, and what typed text is
    //! kept as. In millimetres every one is what it always was.
    use super::*;
    use crate::expression::{ScopeEntry, Value, LENGTH};
    use crate::parameters::{evaluate_parameters, Parameter};

    const MM: DisplayUnits = DisplayUnits::Mm;
    const IN: DisplayUnits = DisplayUnits::In;
    const LEN: ParameterKind = ParameterKind::Length;

    fn width(name: &str) -> Option<ScopeEntry> {
        (name == "width").then_some(ScopeEntry::Value(Value { value: 20.0, dim: LENGTH }))
    }
    fn read(text: &str, kind: ParameterKind, scope: &dyn Scope, units: DisplayUnits) -> (f64, Option<String>) {
        match read_field(text, kind, scope, units) {
            Ok(r) => (r.value, r.expr),
            Err(e) => panic!("{text}: {e}"),
        }
    }
    fn some(s: &str) -> Option<String> {
        Some(s.to_string())
    }

    #[test]
    fn length_text_reads_as_before_in_millimetres() {
        assert_eq!(len_short(12.345, MM, 2), "12.35 mm");
        assert_eq!(len_bare(13.4349, MM, 3), "13.435");
        assert_eq!(field_text(10.0, LEN, 3, MM), "10");
        assert_eq!(field_text(30.123, ParameterKind::Angle, 2, MM), "30.12");
    }

    #[test]
    fn length_text_reads_in_inches_a_decimal_more() {
        assert_eq!(len_short(12.7, IN, 2), "0.5 in");
        assert_eq!(len_bare(6.35, IN, 3), "0.25");
        assert_eq!(field_text(25.4, LEN, 3, IN), "1");
        assert_eq!(field_text(1.0, LEN, 3, IN), "0.0394");
        // An angle is no length: the same in either unit.
        assert_eq!(field_text(30.123, ParameterKind::Angle, 2, IN), "30.12");
    }

    #[test]
    fn a_field_read_in_millimetres_as_it_always_was() {
        assert_eq!(read("12", LEN, &NoNames, MM), (12.0, None));
        assert_eq!(read("width + 5", LEN, &width, MM), (25.0, some("width + 5")));
        assert_eq!(read("-5", LEN, &NoNames, MM), (-5.0, some("-5")));
    }

    #[test]
    fn a_field_read_takes_a_plain_number_in_inches_and_holds_its_millimetres() {
        assert_eq!(read("0.5", LEN, &NoNames, IN), (12.7, None));
        assert_eq!(read("-2", LEN, &NoNames, IN), (-50.8, None));
    }

    #[test]
    fn a_field_read_keeps_bare_arithmetic_so_it_reads_in_inches_when_evaluated_again() {
        let (value, expr) = read("1/4", LEN, &NoNames, IN);
        assert_eq!((value, expr.as_deref()), (6.35, Some("(1/4) * 1 in")));
        // The model evaluates what is kept in millimetres, and gets the same.
        let again = evaluate_for(expr.as_deref().unwrap_or(""), LEN, &NoNames, false).map(|v| v.value);
        assert!(again.as_ref().is_ok_and(|v| (v - 6.35).abs() < 0.5e-12), "{again:?}");
    }

    #[test]
    fn a_field_read_keeps_an_expression_with_a_parameter_or_a_unit_as_typed() {
        assert_eq!(read("width + 5", LEN, &width, IN), (25.0, some("width + 5")));
        assert_eq!(read("2 in", LEN, &NoNames, IN), (50.8, some("2 in")));
    }

    #[test]
    fn a_field_read_leaves_a_field_that_is_no_length_alone() {
        assert_eq!(read("30", ParameterKind::Angle, &NoNames, IN), (30.0, None));
        assert_eq!(read("3", ParameterKind::Integer, &NoNames, IN), (3.0, None));
    }

    #[test]
    fn a_field_read_refuses_what_the_field_refuses() {
        assert!(read_field("width +", LEN, &width, IN).is_err());
    }

    fn p(expression: String) -> Parameter {
        Parameter { id: 1, name: "w".to_string(), kind: LEN, expression, comment: None }
    }

    #[test]
    fn a_parameter_row_shows_a_length_given_as_a_plain_number_in_inches() {
        assert_eq!(parameter_text("25.4", LEN, IN), "1");
        assert_eq!(parameter_text("25.4", LEN, MM), "25.4");
        assert_eq!(parameter_text("width / 2", LEN, IN), "width / 2");
        assert_eq!(parameter_text("30", ParameterKind::Angle, IN), "30");
    }

    #[test]
    fn a_parameter_row_keeps_a_figure_typed_in_inches_so_that_it_reads_as_millimetres() {
        assert_eq!(parameter_expression("1", LEN, IN), "1 in");
        assert_eq!(parameter_expression("0.5 + 0.25", LEN, IN), "(0.5 + 0.25) * 1 in");
        assert_eq!(parameter_expression("width + 1 in", LEN, IN), "width + 1 in");
        assert_eq!(parameter_expression("1", LEN, MM), "1");
        assert_eq!(parameter_expression("45", ParameterKind::Angle, IN), "45");
        let one = evaluate_parameters(&[p(parameter_expression("1", LEN, IN))]).readings[0].value;
        assert!(one.is_some_and(|v| (v - 25.4).abs() < 0.5e-12), "{one:?}");
        let sum = evaluate_parameters(&[p(parameter_expression("0.5 + 0.25", LEN, IN))]).readings[0].value;
        assert!(sum.is_some_and(|v| (v - 19.05).abs() < 0.5e-12), "{sum:?}");
    }

    /// Not in ScanRuler's tests: the signed number the fields read, as its pattern reads it.
    #[test]
    fn a_signed_number_is_what_the_pattern_reads() {
        for yes in ["5", "-5", "+ 0.25", "5.", ".5", "1e3", "-2.5E-2"] {
            assert!(signed_number(yes), "{yes}");
        }
        for no in ["", "-", ".", "5e", "1/4", "5 mm", "--5", "1.2.3"] {
            assert!(!signed_number(no), "{no}");
        }
        assert_eq!(parameter_text("- 25.4", LEN, IN), "-1");
    }
}
