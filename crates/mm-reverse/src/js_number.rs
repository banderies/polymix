// SPDX-License-Identifier: AGPL-3.0-only
//! Numbers written and rounded the way ScanRuler's TypeScript writes and rounds them, so the
//! words a port shows — "comes out 7.3, not whole", "1.378 in", "-5" — are the words
//! ScanRuler shows. A support module for the port of ScanRuler
//! (`plugins/reverse/core/expression.ts`, `src/core/units.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.
//!
//! Rust's own formatting differs from JavaScript's in three places that matter here: `{:.N}`
//! rounds an exact tie to even where `toFixed` rounds it away from zero, `{}` never switches
//! to an exponent where `String(x)` does below 1e-6 and from 1e21, and `f64::round` rounds a
//! half away from zero where `Math.round` rounds it up. `f64::min`, `f64::max` and `powf`
//! also pass a NaN by where JavaScript keeps it.

/// The exact decimal expansion of a finite, non-negative double: its digits and how many of
/// them stand before the decimal point. Every double has a finite expansion of at most 1074
/// fractional digits, so the fixed formatting below is exact, not rounded.
fn exact_digits(x: f64) -> (Vec<u8>, usize) {
    let text = format!("{:.1080}", x);
    let (int, frac) = text.split_once('.').unwrap_or((text.as_str(), ""));
    let mut digits: Vec<u8> = int.bytes().map(|b| b - b'0').collect();
    let point = digits.len();
    digits.extend(frac.bytes().map(|b| b - b'0'));
    (digits, point)
}

/// Digits rounded half up at `keep` digits from the start (a digit at `keep` or beyond is
/// dropped; a carry may add a digit in front, which moves the point one along).
fn round_half_up(mut digits: Vec<u8>, mut point: usize, keep: usize) -> (Vec<u8>, usize) {
    if keep >= digits.len() {
        return (digits, point);
    }
    let up = digits[keep] >= 5;
    digits.truncate(keep);
    if up {
        let mut i = keep;
        loop {
            if i == 0 {
                digits.insert(0, 1);
                point += 1;
                break;
            }
            i -= 1;
            if digits[i] == 9 {
                digits[i] = 0;
            } else {
                digits[i] += 1;
                break;
            }
        }
    }
    (digits, point)
}

/// `x.toFixed(places)`: `places` decimals, an exact half rounded away from zero, a negative
/// sign kept even on a figure that rounds to zero ("-0.00"), and a figure of 1e21 or more
/// written as `String(x)` writes it.
pub fn to_fixed(x: f64, places: usize) -> String {
    if !x.is_finite() || x.abs() >= 1e21 {
        return js_string(x);
    }
    let negative = x < 0.0;
    let (digits, point) = exact_digits(x.abs());
    let (digits, point) = round_half_up(digits, point, point + places);
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    let int: String = digits[..point].iter().map(|d| char::from(b'0' + d)).collect();
    out.push_str(if int.is_empty() { "0" } else { &int });
    if places > 0 {
        out.push('.');
        let frac: String = (0..places).map(|i| char::from(b'0' + digits.get(point + i).copied().unwrap_or(0))).collect();
        out.push_str(&frac);
    }
    out
}

/// `Number(x.toPrecision(digits))`: `x` rounded to `digits` significant digits, an exact half
/// away from zero.
pub fn to_precision_number(x: f64, digits: usize) -> f64 {
    if !x.is_finite() || x == 0.0 || digits == 0 {
        return x;
    }
    let negative = x < 0.0;
    let (all, point) = exact_digits(x.abs());
    let first = all.iter().position(|&d| d != 0).unwrap_or(0);
    let (rounded, point) = round_half_up(all, point, first + digits);
    let mut text = String::new();
    if negative {
        text.push('-');
    }
    for (i, d) in rounded.iter().enumerate() {
        if i == point {
            text.push('.');
        }
        text.push(char::from(b'0' + d));
    }
    for _ in rounded.len()..point {
        text.push('0');
    }
    text.parse().unwrap_or(x)
}

/// `String(x)`: the shortest digits that read back as `x`, in plain decimals from 1e-6 up to
/// 1e21 and with an exponent outside that ("1e-7", "1e+21"); "0" for either zero.
pub fn js_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let a = x.abs();
    if (1e-6..1e21).contains(&a) {
        return format!("{}", x);
    }
    let text = format!("{:e}", x);
    match text.split_once('e') {
        Some((mantissa, exp)) if !exp.starts_with('-') => format!("{mantissa}e+{exp}"),
        _ => text,
    }
}

/// `Math.round(x)`: to the nearest whole number, a half up (towards +∞), so −2.5 is −2.
pub fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// `Math.pow(a, b)`: as `powf`, but NaN for a NaN power and for ±1 to an infinite one.
pub fn js_pow(a: f64, b: f64) -> f64 {
    if b.is_nan() || (a.abs() == 1.0 && b.is_infinite()) {
        return f64::NAN;
    }
    a.powf(b)
}

/// `Math.min(a, b)`: NaN when either is.
pub fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// `Math.max(a, b)`: NaN when either is.
pub fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `String(Number(x.toFixed(places)))`: a number to `places` decimals with no zeros carried
/// behind it.
pub fn fixed_trimmed(x: f64, places: usize) -> String {
    js_string(to_fixed(x, places).parse::<f64>().unwrap_or(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_fixed_rounds_an_exact_half_away_from_zero() {
        assert_eq!(to_fixed(0.125, 2), "0.13");
        assert_eq!(to_fixed(2.5, 0), "3");
        assert_eq!(to_fixed(-2.5, 0), "-3");
        // 1.005 is a hair under the half as a double.
        assert_eq!(to_fixed(1.005, 2), "1.00");
        assert_eq!(to_fixed(9.9996, 3), "10.000");
        assert_eq!(to_fixed(-0.0001, 2), "-0.00");
        assert_eq!(to_fixed(0.0, 3), "0.000");
        assert_eq!(to_fixed(35.0 / 25.4, 4), "1.3780");
        assert_eq!(to_fixed(1e21, 2), "1e+21");
    }

    #[test]
    fn strings_as_javascript_writes_them() {
        assert_eq!(js_string(35.0), "35");
        assert_eq!(js_string(-0.0), "0");
        assert_eq!(js_string(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(js_string(1e-7), "1e-7");
        assert_eq!(js_string(0.000001), "0.000001");
        assert_eq!(js_string(1.5e-7), "1.5e-7");
        assert_eq!(js_string(1e21), "1e+21");
        assert_eq!(js_string(f64::INFINITY), "Infinity");
        assert_eq!(fixed_trimmed(7.3, 3), "7.3");
        assert_eq!(fixed_trimmed(-0.0001, 3), "0");
    }

    #[test]
    fn rounds_and_powers_as_javascript_does() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(0.49999999999999994), 0.0);
        assert!(js_pow(1.0, f64::INFINITY).is_nan());
        assert!(js_pow(1.0, f64::NAN).is_nan());
        assert_eq!(js_pow(2.0, 10.0), 1024.0);
        assert!(js_min(1.0, f64::NAN).is_nan());
        assert_eq!(to_precision_number(0.25, 1), 0.3);
        assert_eq!(to_precision_number(0.0984, 1), 0.1);
        assert_eq!(to_precision_number(2.0 / 25.4, 1), 0.08);
        assert_eq!(to_precision_number(1234.0, 1), 1000.0);
    }
}
