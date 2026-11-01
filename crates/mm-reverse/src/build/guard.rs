// SPDX-License-Identifier: AGPL-3.0-only
//! What a boolean's volume must come to, whatever the kernel says: a join never loses volume
//! and never makes more than its parts had, a cut never gains and never goes below nothing, and
//! a Combine keeps the volume identity — what the two bodies share counted once. Ported from
//! ScanRuler (`plugins/reverse/core/booleanGuard.ts`, and `build.ts` guarded), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A boolean can come back closed and valid and still wrong: a face lost on a tangency, a piece
//! left out on a tolerance. ScanRuler plays such a step again the kernel's other ways; the
//! SolidCraft kernel's boolean already retries its own ways inside one call (shifted copies,
//! other tolerances, piece by piece) and offers no other to ask for, so a result that cannot
//! stand fails its feature here, saying why.

use crate::js_number::{js_string, to_precision_number};

/// How far a volume may stray from what it must be, of the larger: the kernel's tolerance on a
/// curved face, not a lost piece.
pub const VOLUME_SLACK: f64 = 0.002;

/// A boolean as the guard reads it, every volume in mm³. A join: its parts and the result. A
/// cut: the body and the result. A Combine — union, subtract, intersect — the body, the tool,
/// what they share (when the builder worked it out; without it a union and a subtract are held
/// to their bounds alone) and the result.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardAsk {
    Join { parts: Vec<f64>, result: f64 },
    Cut { body: f64, result: f64 },
    Union { body: f64, tool: f64, common: Option<f64>, result: f64 },
    Subtract { body: f64, tool: f64, common: Option<f64>, result: f64 },
    Intersect { body: f64, tool: f64, result: f64 },
}

fn mm3(v: f64) -> String {
    format!("{} mm³", js_string(to_precision_number(v, 6)))
}

/// Why a boolean's result cannot stand, or `None` when it can.
pub fn volume_refusal(ask: &GuardAsk) -> Option<String> {
    let off = |got: f64, want: f64, of: f64| (got - want).abs() > VOLUME_SLACK * of.max(1e-9);
    match ask {
        GuardAsk::Join { parts, result } => {
            let largest = parts.iter().fold(0.0f64, |m, v| m.max(*v));
            let sum: f64 = parts.iter().sum();
            if *result < largest * (1.0 - VOLUME_SLACK) {
                return Some(format!("the join lost volume — {}, less than the largest part alone, {}", mm3(*result), mm3(largest)));
            }
            if *result > sum * (1.0 + VOLUME_SLACK) {
                return Some(format!("the join made volume — {}, more than its parts together, {}", mm3(*result), mm3(sum)));
            }
            None
        }
        GuardAsk::Cut { body, result } => {
            if *result > body * (1.0 + VOLUME_SLACK) {
                return Some(format!("the cut gained volume — {} from {}", mm3(*result), mm3(*body)));
            }
            if *result < 0.0 {
                return Some("the cut left less than nothing".into());
            }
            None
        }
        GuardAsk::Union { body, tool, common, result } => {
            if *result < body.max(*tool) * (1.0 - VOLUME_SLACK) {
                return Some(format!("the join lost volume — {}, less than the larger body alone, {}", mm3(*result), mm3(body.max(*tool))));
            }
            if *result > (body + tool) * (1.0 + VOLUME_SLACK) {
                return Some(format!("the join made volume — {}, more than the two bodies together, {}", mm3(*result), mm3(body + tool)));
            }
            if let Some(c) = common {
                if off(*result, body + tool - c, body + tool) {
                    return Some(format!("the join is {} where the two bodies less what they share come to {}", mm3(*result), mm3(body + tool - c)));
                }
            }
            None
        }
        GuardAsk::Subtract { body, tool, common, result } => {
            if *result > body * (1.0 + VOLUME_SLACK) {
                return Some(format!("the cut gained volume — {} from {}", mm3(*result), mm3(*body)));
            }
            if *result < body - tool - VOLUME_SLACK * body {
                return Some(format!("the cut took more than the tool holds — {} left of {} by a tool of {}", mm3(*result), mm3(*body), mm3(*tool)));
            }
            if let Some(c) = common {
                if off(*result, body - c, *body) {
                    return Some(format!("the cut left {} where the body less what it shares with the tool comes to {}", mm3(*result), mm3(body - c)));
                }
            }
            None
        }
        GuardAsk::Intersect { body, tool, result } => {
            // The result is what they share: bounded by the smaller.
            if *result > body.min(*tool) * (1.0 + VOLUME_SLACK) {
                return Some(format!("what the bodies share came to {}, more than the smaller holds, {}", mm3(*result), mm3(body.min(*tool))));
            }
            None
        }
    }
}

/// A boolean held to its volume: its result's volume asked of the guard, and the failure said
/// as ScanRuler's `guarded` says it — "Joining cannot stand: the join lost volume — …".
pub fn guarded(what: &str, ask: &GuardAsk) -> Result<(), String> {
    match volume_refusal(ask) {
        Some(reason) => Err(format!("{what} cannot stand: {reason}.")),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn starts(r: Option<String>, with: &str) -> bool {
        r.is_some_and(|s| s.starts_with(with))
    }

    #[test]
    fn holds_a_booleans_volume_to_what_it_must_come_to() {
        assert_eq!(volume_refusal(&GuardAsk::Join { parts: vec![100.0, 50.0], result: 140.0 }), None);
        assert!(starts(volume_refusal(&GuardAsk::Join { parts: vec![100.0, 50.0], result: 90.0 }), "the join lost volume"));
        assert!(starts(volume_refusal(&GuardAsk::Join { parts: vec![100.0, 50.0], result: 151.0 }), "the join made volume"));
        assert_eq!(volume_refusal(&GuardAsk::Cut { body: 100.0, result: 99.0 }), None);
        assert!(starts(volume_refusal(&GuardAsk::Cut { body: 100.0, result: 101.0 }), "the cut gained volume"));
        // The identity: the two bodies less what they share, within 0.2 %.
        assert_eq!(volume_refusal(&GuardAsk::Union { body: 100.0, tool: 50.0, common: Some(20.0), result: 130.2 }), None);
        assert!(volume_refusal(&GuardAsk::Union { body: 100.0, tool: 50.0, common: Some(20.0), result: 125.0 }).is_some_and(|s| s.contains("less what they share")));
        assert_eq!(volume_refusal(&GuardAsk::Subtract { body: 100.0, tool: 50.0, common: Some(20.0), result: 80.0 }), None);
        assert!(starts(volume_refusal(&GuardAsk::Subtract { body: 100.0, tool: 50.0, common: Some(20.0), result: 70.0 }), "the cut left 70 mm³"));
        assert!(volume_refusal(&GuardAsk::Subtract { body: 100.0, tool: 10.0, common: None, result: 50.0 }).is_some_and(|s| s.contains("more than the tool holds")));
        assert!(volume_refusal(&GuardAsk::Intersect { body: 100.0, tool: 50.0, result: 60.0 }).is_some_and(|s| s.contains("more than the smaller holds")));
        assert_eq!(guarded("Joining", &GuardAsk::Join { parts: vec![100.0, 50.0], result: 90.0 }), Err("Joining cannot stand: the join lost volume — 90 mm³, less than the largest part alone, 100 mm³.".to_string()));
    }
}
