// SPDX-License-Identifier: AGPL-3.0-only
//! The colours bodies wear: the palette a body is drawn in by the feature that made it, the
//! swatches its dot in the body list offers, and a colour as given by name or by hex.
//!
//! Ported from ScanRuler (`plugins/reverse/state/reTypes.ts` `BODY_SWATCHES`, `bodyColorOf`;
//! `state/reDrafts.ts` `bodyColor`, `roundFaceLength`; `src/state/palette.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use crate::model::BodyId;
use crate::topo::FaceFingerprint;

/// The eight accent colours everything numbered on screen is drawn in, saturated enough to hold
/// against a grey stage and the tone of an unmeasured scan.
pub const PALETTE: [&str; 8] = ["#1877c0", "#e8590c", "#2e7d46", "#b5179e", "#c99a0a", "#0f9b9b", "#b3361c", "#6d4bbd"];

/// The colours a body is offered — a board's greens and blues, the metals of what is bought
/// in — by the names the commands take them by too.
pub const BODY_SWATCHES: [(&str, &str); 15] = [
    ("black", "#1e1f22"),
    ("white", "#f1f1ee"),
    ("grey", "#8b9096"),
    ("aluminium", "#c5c9cd"),
    ("steel", "#707880"),
    ("brass", "#c9a646"),
    ("copper", "#b87333"),
    ("pcb green", "#1f5e33"),
    ("pcb blue", "#1d3f7a"),
    ("red", "#c62f2f"),
    ("orange", "#e57a2e"),
    ("yellow", "#efc33a"),
    ("green", "#3f9b4a"),
    ("blue", "#2f73c4"),
    ("purple", "#7c52b3"),
];

/// The colour a body wears, by the feature that made it — and a further body a step left, a
/// piece apart, a colour further along.
pub fn body_color(id: BodyId) -> &'static str {
    PALETTE[((id.feature as u64 + 5 + 3 * id.island as u64) % PALETTE.len() as u64) as usize]
}

/// A colour as given — #rgb, #rrggbb, or a swatch's name, any case — as #rrggbb, or `None` for
/// none of those.
pub fn body_color_of(given: &str) -> Option<String> {
    let t = given.trim().to_lowercase();
    let squeezed: String = t.chars().filter(|c| !(c.is_whitespace() || *c == '_' || *c == '-')).collect();
    if let Some((_, c)) = BODY_SWATCHES.iter().find(|(name, _)| *name == t || name.replacen(' ', "", 1) == squeezed) {
        return Some((*c).to_string());
    }
    let hex = t.strip_prefix('#')?;
    if !(hex.len() == 3 || hex.len() == 6) || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(if hex.len() == 3 { format!("#{}", hex.chars().flat_map(|c| [c, c]).collect::<String>()) } else { t })
}

/// A #rrggbb colour as its three bytes, or `None` for one that is not.
pub fn rgb_of(hex: &str) -> Option<[u8; 3]> {
    let h = hex.strip_prefix('#')?;
    if h.len() != 6 {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
    Some([byte(0)?, byte(2)?, byte(4)?])
}

/// How long a round face runs along its axis, read off its fingerprint as a full cylinder's —
/// what a thread that does not run the whole face starts at.
pub fn round_face_length(face: &FaceFingerprint) -> f64 {
    let r = face.radius.unwrap_or(0.0);
    if r > 0.0 {
        (face.area / (2.0 * std::f64::consts::PI * r) * 1000.0).round() / 1000.0
    } else {
        10.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::FaceKind;
    use glam::DVec3;

    #[test]
    fn a_body_takes_the_palette_by_its_maker_and_its_piece() {
        assert_eq!(body_color(BodyId::new(1)), PALETTE[6]);
        assert_eq!(body_color(BodyId::new(3)), PALETTE[0]);
        assert_eq!(body_color(BodyId { feature: 3, island: 1 }), PALETTE[3]);
    }

    #[test]
    fn a_colour_by_name_or_by_hex() {
        assert_eq!(body_color_of("PCB green").as_deref(), Some("#1f5e33"));
        assert_eq!(body_color_of("pcb_blue").as_deref(), Some("#1d3f7a"));
        assert_eq!(body_color_of(" Brass ").as_deref(), Some("#c9a646"));
        assert_eq!(body_color_of("#ABC").as_deref(), Some("#aabbcc"));
        assert_eq!(body_color_of("#12ab9f").as_deref(), Some("#12ab9f"));
        assert_eq!(body_color_of("#12ab9"), None);
        assert_eq!(body_color_of("mauve"), None);
        assert_eq!(rgb_of("#1f5e33"), Some([0x1f, 0x5e, 0x33]));
        assert_eq!(rgb_of("1f5e33"), None);
    }

    #[test]
    fn a_round_face_runs_its_area_over_its_girth() {
        let f = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::ZERO, dir: DVec3::Z, area: 2.0 * std::f64::consts::PI * 5.0 * 12.0, radius: Some(5.0), center: Some(DVec3::ZERO), origin: 1 };
        assert_eq!(round_face_length(&f), 12.0);
        assert_eq!(round_face_length(&FaceFingerprint { radius: None, ..f }), 10.0);
    }
}
