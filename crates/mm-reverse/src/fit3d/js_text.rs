// SPDX-License-Identifier: AGPL-3.0-only
//! Numbers written into messages the way ScanRuler's JavaScript writes them, so a refusal
//! reads the same: `Number.prototype.toFixed` and the plain conversion of a number to text.
//!
//! Written for the port of ScanRuler's fits (© Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only), whose messages are built with template strings.

/// `x.toFixed(digits)`: the exact decimal value of `x` rounded to `digits` places, a tie
/// going to the larger magnitude; `NaN` and `±Infinity` as words; a magnitude of 1e21 or more
/// as the plain conversion. A negative zero is written as zero, and a negative number that
/// rounds to zero keeps its sign ("-0.00"), as in JavaScript.
pub fn to_fixed(x: f64, digits: usize) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.abs() >= 1e21 {
        return js_number(x);
    }
    let negative = x < 0.0;
    // Every finite double has at most 1074 decimal places, so this is its exact value.
    let exact = format!("{:.*}", digits.max(1100) + 1, x.abs());
    let (int_part, frac_part) = exact.split_once('.').unwrap_or((&exact, ""));
    let mut digits_kept: Vec<u8> = int_part.bytes().chain(frac_part.bytes().take(digits)).collect();
    let next = frac_part.as_bytes().get(digits).copied().unwrap_or(b'0');
    if next >= b'5' {
        // Round half up on the exact expansion: carry through the kept digits.
        let mut i = digits_kept.len();
        loop {
            if i == 0 {
                digits_kept.insert(0, b'1');
                break;
            }
            i -= 1;
            if digits_kept[i] == b'9' {
                digits_kept[i] = b'0';
            } else {
                digits_kept[i] += 1;
                break;
            }
        }
    }
    let split = digits_kept.len() - digits;
    let int_s = std::str::from_utf8(&digits_kept[..split]).unwrap_or("0");
    let frac_s = std::str::from_utf8(&digits_kept[split..]).unwrap_or("");
    let body = if digits == 0 { int_s.to_owned() } else { format!("{int_s}.{frac_s}") };
    if negative {
        format!("-{body}")
    } else {
        body
    }
}

/// A number as JavaScript converts it to text in a template string: the shortest digits that
/// read back to it, no trailing ".0", "NaN", "Infinity", a negative zero as "0", and — below
/// 1e-6 or from 1e21 in size — the exponent form JavaScript switches to ("1e-7", "1.5e+21").
pub fn js_number(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else if x == 0.0 {
        "0".into()
    } else if x.abs() < 1e-6 || x.abs() >= 1e21 {
        // Rust's `{:e}` writes the same shortest digits, without the exponent's plus sign.
        let s = format!("{x:e}");
        match s.split_once('e') {
            Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
            _ => s,
        }
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_numbers_as_javascript_does() {
        assert_eq!(to_fixed(1.005, 2), "1.00"); // 1.005 is a hair under in binary
        assert_eq!(to_fixed(0.125, 2), "0.13"); // an exact tie goes up
        assert_eq!(to_fixed(-0.125, 2), "-0.13");
        assert_eq!(to_fixed(-0.0001, 2), "-0.00");
        assert_eq!(to_fixed(-0.0, 2), "0.00");
        assert_eq!(to_fixed(9.999, 2), "10.00");
        assert_eq!(to_fixed(14.0, 3), "14.000");
        assert_eq!(to_fixed(2.5, 0), "3");
        assert_eq!(to_fixed(f64::NAN, 1), "NaN");
        assert_eq!(js_number(20.0), "20");
        assert_eq!(js_number(3.5), "3.5");
        assert_eq!(js_number(-0.0), "0");
        assert_eq!(js_number(0.894), "0.894");
        // String(x) and x.toFixed(2) in Node.
        assert_eq!(js_number(1e-7), "1e-7");
        assert_eq!(js_number(9.9e-7), "9.9e-7");
        assert_eq!(js_number(1e-6), "0.000001");
        assert_eq!(js_number(0.000001234), "0.000001234");
        assert_eq!(js_number(-2.5e-8), "-2.5e-8");
        assert_eq!(js_number(1.5e21), "1.5e+21");
        assert_eq!(js_number(1e21), "1e+21");
        assert_eq!(js_number(123456789012345680000.0), "123456789012345680000");
        assert_eq!(js_number(f64::MAX), "1.7976931348623157e+308");
        assert_eq!(js_number(5e-324), "5e-324");
        assert_eq!(to_fixed(1.5e21, 2), "1.5e+21");
        assert_eq!(to_fixed(123456789012345680000.0, 2), "123456789012345683968.00");
        assert_eq!(to_fixed(-2.5e-8, 2), "-0.00");
        assert_eq!(to_fixed(5e-324, 2), "0.00");
    }
}
