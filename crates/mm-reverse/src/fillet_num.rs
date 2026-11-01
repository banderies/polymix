// SPDX-License-Identifier: AGPL-3.0-only
//! Numbers kept to a fixed count of decimals, and numbers written into words, as ScanRuler
//! writes them: what the readings hand back and what their reasons say. Ported from ScanRuler
//! (`src/core/round.ts`, and the JavaScript number rules the fillet readers' messages lean on:
//! `Math.round`, `toFixed`, `toPrecision` and a number turned into text), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! JavaScript rounds a half up (towards +∞) where Rust's `round` takes it away from zero, and
//! `toFixed` / `toPrecision` round the exact decimal value of a double half away from zero
//! where Rust's formatting rounds it half to even. The helpers here keep JavaScript's rules so
//! a reading's figures and its words come out as ScanRuler's do.

/// `Math.round`: the nearest integer, a half rounded up (towards +∞).
pub fn js_round(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = x.floor();
    // x − ⌊x⌋ is exact for any double this size, so the half is told exactly.
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// Four decimals — what an answer is read at. Halves round up, as `Math.round` has them.
pub fn round4(v: f64) -> f64 {
    js_round(v * 10_000.0) / 10_000.0
}

/// A number written as JavaScript writes it into text: the shortest digits that read back
/// the same, no trailing ".0", exponent form under a millionth and from 10²¹ on, and −0 as 0.
pub fn js_num(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() };
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let a = x.abs();
    if !(1e-6..1e21).contains(&a) {
        let s = format!("{x:e}");
        // JavaScript writes a positive exponent with its sign.
        return match s.split_once('e') {
            Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
            _ => s,
        };
    }
    format!("{x}")
}

/// The exact decimal digits of |x| as `d.ddd…` with `digits` digits after the point, rounded
/// half away from zero on the exact value, and its exponent. Rust formats the exact value too
/// but takes a tie to the even digit; the tie is told here from a long exact expansion.
fn exp_digits(a: f64, digits: usize) -> (Vec<u8>, i32) {
    // Long enough that a double whose expansion runs past it cannot stand on a tie.
    let exact = format!("{:.*e}", digits + 1100, a);
    let (mant, exp) = exact.split_once('e').unwrap_or((exact.as_str(), "0"));
    let mut e: i32 = exp.parse().unwrap_or(0);
    let all: Vec<u8> = mant.bytes().filter(|b| b.is_ascii_digit()).map(|b| b - b'0').collect();
    let keep = digits + 1;
    let mut out: Vec<u8> = all.iter().take(keep).copied().collect();
    let rest = &all[keep.min(all.len())..];
    let up = match rest.first() {
        Some(&d) if d > 5 => true,
        Some(&5) => true,
        _ => false,
    };
    // A first dropped digit of 5 rounds up whether or not anything follows it: a tie rounds
    // half up on the magnitude, and anything more is past the half.
    if up {
        let mut i = out.len();
        loop {
            if i == 0 {
                out.insert(0, 1);
                out.truncate(keep);
                e += 1;
                break;
            }
            i -= 1;
            if out[i] == 9 {
                out[i] = 0;
            } else {
                out[i] += 1;
                break;
            }
        }
    }
    (out, e)
}

/// `x.toPrecision(p)`: `p` significant digits, in fixed notation where the exponent lies in
/// −6 … p − 1, else in exponent form, as JavaScript writes it.
pub fn to_precision(x: f64, p: usize) -> String {
    if !x.is_finite() {
        return js_num(x);
    }
    let p = p.max(1);
    let sign = if x < 0.0 { "-" } else { "" };
    if x == 0.0 {
        return if p == 1 { "0".to_string() } else { format!("0.{}", "0".repeat(p - 1)) };
    }
    let (d, e) = exp_digits(x.abs(), p - 1);
    let digits: String = d.iter().map(|v| char::from(b'0' + v)).collect();
    if e < -6 || e >= p as i32 {
        let m = if p == 1 { digits } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        let es = if e >= 0 { format!("+{e}") } else { format!("{e}") };
        return format!("{sign}{m}e{es}");
    }
    if e >= 0 {
        let int = (e + 1) as usize;
        if int >= p {
            format!("{sign}{digits}")
        } else {
            format!("{sign}{}.{}", &digits[..int], &digits[int..])
        }
    } else {
        format!("{sign}0.{}{digits}", "0".repeat((-e - 1) as usize))
    }
}

/// `Number(x.toPrecision(p))`: the value at `p` significant digits.
pub fn precision_value(x: f64, p: usize) -> f64 {
    to_precision(x, p).parse().unwrap_or(x)
}

/// `Number(x.toFixed(f))`: the value at `f` decimals, a half rounded away from zero on the
/// exact value, as `toFixed` rounds it.
pub fn fixed_value(x: f64, f: usize) -> f64 {
    if !x.is_finite() || x.abs() >= 1e21 {
        return x;
    }
    let exact = format!("{:.*}", f + 1100, x.abs());
    let (int, frac) = exact.split_once('.').unwrap_or((exact.as_str(), ""));
    let mut digits: Vec<u8> = int.bytes().chain(frac.bytes().take(f)).map(|b| b - b'0').collect();
    let up = frac.as_bytes().get(f).is_some_and(|&b| b >= b'5');
    if up {
        let mut i = digits.len();
        loop {
            if i == 0 {
                digits.insert(0, 1);
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
    let n = digits.len() - f;
    let text: String = digits[..n].iter().map(|v| char::from(b'0' + v)).collect::<String>() + "." + &digits[n..].iter().map(|v| char::from(b'0' + v)).collect::<String>();
    let v: f64 = text.parse().unwrap_or(x.abs());
    if x < 0.0 {
        -v
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_and_writes_as_javascript_does() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(0.49999999999999994), 0.0);
        assert_eq!(round4(1.23456), 1.2346);
        assert_eq!(js_num(1.7), "1.7");
        assert_eq!(js_num(120.0), "120");
        assert_eq!(js_num(-0.0), "0");
        assert_eq!(js_num(1.2e-7), "1.2e-7");
        assert_eq!(to_precision(3.0, 3), "3.00");
        assert_eq!(to_precision(1234.0, 3), "1.23e+3");
        assert_eq!(to_precision(0.0123, 2), "0.012");
        assert_eq!(to_precision(9.96, 2), "10");
        assert_eq!(to_precision(0.125, 2), "0.13");
        assert_eq!(precision_value(1.7049, 2), 1.7);
        assert_eq!(fixed_value(0.8, 2), 0.8);
        assert_eq!(fixed_value(1.005, 2), 1.0); // 1.005 is 1.00499… in binary
        assert_eq!(fixed_value(0.125, 2), 0.13);
        assert_eq!(fixed_value(-0.125, 2), -0.13);
    }
}
