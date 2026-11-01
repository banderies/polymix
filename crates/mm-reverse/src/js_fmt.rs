// SPDX-License-Identifier: AGPL-3.0-only
//! Numbers written the way ScanRuler's messages write them. Its refusals and answers are
//! JavaScript template strings, so a number in them reads as JavaScript prints it: `20` not
//! `20.0`, `0.5`, a rounding with `Math.round`, a figure to fixed decimals with `toFixed`, to
//! significant digits with `toPrecision`. The same rules here, so a message ported from
//! ScanRuler (© Stefan Hermann (CNC Kitchen), AGPL-3.0-only) says what it said there.

/// `Math.round`: to the nearest whole number, a half rounded up (towards +∞), so −2.5 is −2.
pub fn js_round(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = x.floor();
    let r = if x - f >= 0.5 { f + 1.0 } else { f };
    if r == 0.0 && x < 0.0 {
        -0.0
    } else {
        r
    }
}

/// `x || 0`: the figure, with −0 and NaN (both falsy in JavaScript) taken as 0 — how
/// ScanRuler's rounding helpers (`Math.round(x * 1000) / 1000 || 0`) keep "-0" and "NaN" out of
/// what they write.
pub fn or_zero(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() {
        0.0
    } else {
        x
    }
}

/// `Math.round(x * 10^d) / 10^d || 0`: rounded to `d` decimals the way ScanRuler's `r1`, `r3`
/// and the like do it.
pub fn round_to(x: f64, d: i32) -> f64 {
    let k = 10f64.powi(d);
    or_zero(js_round(x * k) / k)
}

/// A number as JavaScript's `String(x)` writes it: the shortest digits that read back as the
/// same number, no trailing `.0`, `0` for −0, and an exponent below 1e-6 or from 1e21 up.
pub fn num(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return "0".into();
    }
    let a = x.abs();
    if !(1e-6..1e21).contains(&a) {
        let s = format!("{x:e}");
        if let Some((m, e)) = s.split_once('e') {
            if !e.starts_with('-') {
                return format!("{m}e+{e}");
            }
        }
        return s;
    }
    format!("{x}")
}

/// The digits of `a` (not negative) with `d` after the point, rounded as JavaScript rounds a
/// figure for `toFixed`: to the nearer, and on a tie to the larger.
fn fixed_digits(a: f64, d: usize) -> String {
    let s = format!("{:.*}", d + 60, a);
    let (int, frac) = match s.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (s.clone(), String::new()),
    };
    let mut digits: Vec<u8> = int.bytes().chain(frac.bytes().take(d)).map(|b| b - b'0').collect();
    let up = frac.as_bytes().get(d).is_some_and(|&b| b >= b'5');
    if up {
        let mut k = digits.len();
        loop {
            if k == 0 {
                digits.insert(0, 1);
                break;
            }
            k -= 1;
            if digits[k] == 9 {
                digits[k] = 0;
            } else {
                digits[k] += 1;
                break;
            }
        }
    }
    let text: String = digits.iter().map(|d| (b'0' + d) as char).collect();
    if d == 0 {
        return text;
    }
    let split = text.len() - d;
    format!("{}.{}", &text[..split], &text[split..])
}

/// `x.toFixed(d)`: `d` decimals, a minus sign on any negative figure (even one that rounds to
/// zero, `-0.00`), none on −0.
pub fn to_fixed(x: f64, d: usize) -> String {
    if !x.is_finite() || x.abs() >= 1e21 {
        return num(x);
    }
    let body = fixed_digits(x.abs(), d);
    if x < 0.0 {
        format!("-{body}")
    } else {
        body
    }
}

/// `Number(x.toFixed(d))`: the figure rounded to `d` decimals, as a number again.
pub fn fixed_num(x: f64, d: usize) -> f64 {
    to_fixed(x, d).parse().unwrap_or(x)
}

/// `Number(x.toPrecision(p))`: the figure rounded to `p` significant digits.
pub fn to_precision_num(x: f64, p: usize) -> f64 {
    if !x.is_finite() || x == 0.0 || p == 0 {
        return x;
    }
    let s = format!("{:.*e}", p - 1 + 60, x.abs());
    let Some((mant, exp)) = s.split_once('e') else { return x };
    let Ok(mut e) = exp.parse::<i32>() else { return x };
    let ds: Vec<u8> = mant.bytes().filter(|b| b.is_ascii_digit()).map(|b| b - b'0').collect();
    let mut digits: Vec<u8> = ds.iter().copied().take(p).collect();
    if ds.get(p).is_some_and(|&d| d >= 5) {
        let mut k = digits.len();
        loop {
            if k == 0 {
                digits.insert(0, 1);
                digits.pop();
                e += 1;
                break;
            }
            k -= 1;
            if digits[k] == 9 {
                digits[k] = 0;
            } else {
                digits[k] += 1;
                break;
            }
        }
    }
    let text: String = digits.iter().map(|d| (b'0' + d) as char).collect();
    let back: f64 = format!("{}.{}e{}", &text[..1], &text[1..], e).parse().unwrap_or(x.abs());
    if x < 0.0 {
        -back
    } else {
        back
    }
}

/// A whole number as `toLocaleString('en-US')` writes it: thousands set off by commas.
pub fn grouped(x: f64) -> String {
    let n = js_round(x);
    let neg = n < 0.0;
    let digits = format!("{:.0}", n.abs());
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if neg {
        format!("-{out}")
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_numbers_as_javascript_does() {
        assert_eq!(num(20.0), "20");
        assert_eq!(num(-0.0), "0");
        assert_eq!(num(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(num(1.5e-7), "1.5e-7");
        assert_eq!(num(1e21), "1e+21");
        assert_eq!(num(0.002), "0.002");
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(to_fixed(0.125, 2), "0.13");
        assert_eq!(to_fixed(1.005, 2), "1.00");
        assert_eq!(to_fixed(-0.001, 2), "-0.00");
        assert_eq!(to_fixed(-0.0, 2), "0.00");
        assert_eq!(to_fixed(9.996, 2), "10.00");
        assert_eq!(to_fixed(12.0, 0), "12");
        assert_eq!(to_precision_num(0.000_523, 2), 0.000_52);
        assert_eq!(to_precision_num(0.0005, 2), 0.0005);
        assert_eq!(to_precision_num(0.00999, 2), 0.01);
        assert_eq!(grouped(11243.0), "11,243");
        assert_eq!(grouped(999.0), "999");
        assert_eq!(grouped(1_234_567.4), "1,234,567");
        assert!(or_zero(f64::NAN) == 0.0 && or_zero(-0.0).is_sign_positive());
        assert_eq!(round_to(-0.0004, 3), 0.0);
        assert!(round_to(-0.0004, 3).is_sign_positive());
        assert_eq!(round_to(f64::NAN, 1), 0.0);
        assert_eq!(round_to(1.23456, 3), 1.235);
    }
}
