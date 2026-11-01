// SPDX-License-Identifier: AGPL-3.0-only
//! Numbers kept to a fixed count of decimals, for an answer to be read rather than computed
//! on: what the commands hand back, and the readouts beside them. Halves round up, as
//! JavaScript's `Math.round` has them. Ported from ScanRuler (`src/core/round.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Alongside, the two pieces of JavaScript's own number handling the messages ported from
//! ScanRuler lean on: `Math.round` (halves towards +∞, where Rust's `f64::round` takes them
//! away from zero) and `String(number)` (the shortest text that reads back as the number,
//! with no trailing `.0`).

/// JavaScript's `Math.round`: the nearest integer, halves rounded up (towards +∞), so
/// −2.5 rounds to −2 where `f64::round` gives −3.
pub fn js_round(v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    let f = v.floor();
    // `v - f` is exact for every double, so a value a hair under a half stays under it.
    let r = if v - f >= 0.5 { f + 1.0 } else { f };
    // Math.round keeps the sign of a zero it rounds to: −0.4 gives −0.
    if r == 0.0 && v.is_sign_negative() {
        -0.0
    } else {
        r
    }
}

/// `v` to `places` decimals.
pub fn round_to(v: f64, places: i32) -> f64 {
    let k = 10f64.powi(places);
    js_round(v * k) / k
}

/// Three decimals — a micrometre, for millimetres.
pub fn round3(v: f64) -> f64 {
    js_round(v * 1000.0) / 1000.0
}

/// Four decimals.
pub fn round4(v: f64) -> f64 {
    js_round(v * 10_000.0) / 10_000.0
}

/// A number as JavaScript's `String(number)` writes it: the shortest text that reads back
/// as the number, no trailing `.0` on a whole one, `0` for a negative zero, and exponent
/// notation from 10²¹ up and under 10⁻⁶, as `1e+21` and `1.5e-7`.
pub fn number_text(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() };
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if a >= 1e21 || a < 1e-6 {
        let text = format!("{v:e}");
        // Rust writes `1e21` and `1.5e-7`; JavaScript puts a sign on every exponent.
        return match text.split_once('e') {
            Some((mantissa, exp)) if !exp.starts_with('-') => format!("{mantissa}e+{exp}"),
            _ => text,
        };
    }
    format!("{v}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_halves_up_as_javascript_does() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(0.49999999999999994), 0.0);
        assert_eq!(round3(1.0005), 1.001);
        assert_eq!(round4(-0.00005), -0.0);
    }

    #[test]
    fn writes_numbers_as_javascript_does() {
        assert_eq!(number_text(16.0), "16");
        assert_eq!(number_text(0.35), "0.35");
        assert_eq!(number_text(-0.0), "0");
        assert_eq!(number_text(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(number_text(1e21), "1e+21");
        assert_eq!(number_text(1.5e-7), "1.5e-7");
        assert_eq!(number_text(0.000001), "0.000001");
    }
}
