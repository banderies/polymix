// SPDX-License-Identifier: AGPL-3.0-only
//! The slots of every kind of feature: the numbers a feature's box shows, read and written
//! through the feature's own fields. Ported from ScanRuler (`plugins/reverse/core/parameters.ts`
//! `slotsOf` and `slotAt`, with `planeAtOffset`, `planeOffset`, `planeTakesOffset`,
//! `withPlaneOffset` and `revolveMaxDeg` from `plugins/reverse/core/features.ts` and
//! `noSlotsWhy` from `plugins/reverse/app/reFeatureInput.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! A slot is the field as the box shows it (an extrude's To, a hole's Diameter), which is not
//! always the field as the feature stores it (a radius, a draft with the opposite sign). Its
//! path is the key the feature keeps its expression under. Two tables of slots live here, one
//! mapping between them:
//!
//! - [`json_slots`] is ScanRuler's own table, on a feature as ScanRuler writes it — its fields
//!   under ScanRuler's names in a [`JsonFeature`] — for every kind ScanRuler has. ScanRuler reads
//!   a feature's fields untyped, and so does this: a number missing where ScanRuler falls back to
//!   0 reads as 0, and one missing where it does not reads as NaN, which (as `undefined` in
//!   JavaScript arithmetic) no expression ever writes over.
//! - [`model_slots`] is the same table on Polymix's `model::FeatureKind`, for the kinds the model
//!   has, and [`ModelFeature`] a feature of the model with the expressions it keeps. Both target
//!   the eight-kind model copied into this crate. mm-reverse's model has since grown to
//!   ScanRuler's whole list, with `exprs` of its own (see the crate's head).
//!
//! | kind | slot path | label | kind of number | ScanRuler field | `model::FeatureKind` |
//! |---|---|---|---|---|---|
//! | extrude | `extent.from`, `extent.to` (a distance); `extent.total` (symmetric, sign dropped); `extent.from` (any other extent but between planes) | From, To, Total | length | `extent.*` | `Extent::Distance { from, to }`, `Extent::Symmetric { total }` |
//! | extrude | `taper` | Taper | angle | −`draftDeg` | `taper_deg` |
//! | revolve | `angleDeg` (0.1° up to a turn, or a thousand with a pitch); with a pitch also `pitch`, `turns`, `height` | Angle, Pitch, Turns, Height | angle, length, number, length | `angleDeg`, `pitch` | `angle_deg` |
//! | shell | `thickness` | Wall | length, sign dropped | `thickness` | `Shell::thickness` |
//! | fillet | `radius` | Radius | length, sign dropped | `radius` | `EdgeBlend::size` |
//! | chamfer | `distance`; `distance2` (two distances); `angleDeg` (distance and angle, 0.1°–89.9°) | Distance, Distance 2, Angle | length, length, angle | the same | `EdgeBlend::size` |
//! | hole | `diameter`; `depth` when it has one | Diameter, Depth | length | 2 × `radius`, `depth` | — |
//! | thread | `diameter`, `pitch`; `length`, `start` when it does not run full length; `offset` | Diameter, Pitch, Length, Start, Offset | length | the same | — |
//! | extract, thicken | `thickness` | Thickness | length, sign dropped | `thickness` | — |
//! | draft | `angleDeg` | Angle | angle | `angleDeg` | — |
//! | moveFace | `offset` | Offset | length | `offset` | — |
//! | offsetBody | `distance` | Distance | length | `distance` | — |
//! | moveBody | `translate.x` … `translate.z`, `rotate.x` … `rotate.z` | Move X … Turn Z | length, angle | `translate[i]`, `rotate[i]` | — |
//! | sheet | `from`, `to` | From, To | length | the same | — |
//! | sweep | `route.bendRadius`, when it has a route | Bend radius | length, sign dropped | `route.bendRadius` | — |
//! | pattern | `count` (at least 2), `step`; a linear one with a second axis also `count2` (at least 1), `step2` | Count, Spacing or Over, Count 2, Spacing 2 | integer, length or angle | the same | — |
//! | sketch | `offset`, when its plane takes one; `inset`, a profile moved in from another sketch | X = / Y = / Z = or Offset, Inset | length | the plane's `offset` (the frame carried with it), `inset.by` | `PlaneRef` offset |
//! | cutPlane, mirror | `plane.offset`, when its plane takes one (not a face) | Offset | length | `plane.offset` | `Mirror::plane` offset |
//! | surface | `region.box.min.x` … `region.box.max.z`, when it was read in a box | Box min X … Box max Z | length | `regionBox.box.*[i]`, marking the region stale | — |
//! | split, trim | `tolerance`, `overlap` (percent, 5–100) | Tolerance, Overlap | length, number | `tolerance`, `overlap` × 100 | — |
//! | any other (boolean, loft, combine …) | none | | | | |
//!
//! A loft's own expressions (`sections.<k>.offset`, `inner`) are not slots of the loft: the
//! command that takes them writes them onto its station sketches' `offset` and `inset` slots.

use std::cmp::Ordering;

use glam::DVec3;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::frame::SectionFrame;
use crate::js_number::{js_max, js_min, js_round};
use crate::model::{Feature, FeatureId, FeatureKind};
use crate::parameters::{ExprMap, ParameterKind, ParametricFeature, RegionPick, RegionPicks, Slot, SketchInset, Slotted};
use crate::sketch::Sketch;

/// The angle a revolve with no pitch turns at most: a whole turn.
pub const FULL_TURN_DEG: f64 = 360.0;

/// The angle a coil turns at most: a thousand turns.
pub const MAX_COIL_DEG: f64 = 360.0 * 1000.0;

/// Whether a number is "truthy" as JavaScript reads it: not zero and not NaN.
fn truthy_number(v: f64) -> bool {
    v != 0.0 && !v.is_nan()
}

/// The most a revolve may turn, with its pitch or without one.
pub fn revolve_max_deg(pitch: f64) -> f64 {
    if truthy_number(pitch) {
        MAX_COIL_DEG
    } else {
        FULL_TURN_DEG
    }
}

/// A typed number with its sign dropped — what most sizes go through.
fn positive(v: f64) -> f64 {
    v.abs()
}

/// A revolve's angle as a typed one leaves it: its sign dropped, at least a tenth of a degree
/// and at most what the pitch lets it turn.
fn revolve_angle(v: f64, pitch: f64) -> f64 {
    js_min(revolve_max_deg(pitch), js_max(0.1, v.abs()))
}

// ---- a feature as ScanRuler writes it ----------------------------------------------------

/// A feature — or the draft of one in its box — as ScanRuler writes it: its fields under
/// ScanRuler's names (`kind`, `id`, `name`, `extent`, `draftDeg`, `plane`, `frame`, `inset`,
/// `regions`, `regionPick` …), its sketch as Polymix holds one, and the expressions it keeps by
/// slot. A draft of an extrude or a revolve has no `kind` of its own; the slots are then asked
/// with the kind given ([`json_slots`], `apply_slots_with`).
///
/// Two features are equal when their fields are, numbers compared by value (`10` is `10.0`), so
/// a number written back as it was is no change.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JsonFeature {
    /// A sketch's pieces and what holds them (ScanRuler's `sketch`), or a sketch being drawn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sketch: Option<Sketch>,
    /// The expressions driving its numbers, by slot path; none when every number was typed as
    /// one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exprs: Option<ExprMap>,
    /// Every other field.
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

impl PartialEq for JsonFeature {
    fn eq(&self, other: &Self) -> bool {
        self.sketch == other.sketch && self.exprs == other.exprs && fields_eq(&self.fields, &other.fields)
    }
}

/// Two JSON values alike, numbers compared by value.
fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64().partial_cmp(&y.as_f64()) == Some(Ordering::Equal),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_eq(p, q)),
        (Value::Object(x), Value::Object(y)) => fields_eq(x, y),
        _ => a == b,
    }
}

fn fields_eq(a: &Map<String, Value>, b: &Map<String, Value>) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| json_eq(v, w)))
}

/// A number as JSON writes it, as JavaScript's `JSON.stringify` does: a whole number without
/// a point, negative zero as zero, and a NaN or an infinity as null.
pub fn json_number(v: f64) -> Value {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 9_007_199_254_740_992.0 {
        // In range and whole: the cast is exact.
        return Value::from(v as i64);
    }
    serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number)
}

fn json_vec3(v: DVec3) -> Value {
    Value::Array(vec![json_number(v.x), json_number(v.y), json_number(v.z)])
}

/// A `[x, y, z]` of JSON as a vector, or none.
fn vec3_of(v: Option<&Value>) -> Option<DVec3> {
    match v?.as_array()?.as_slice() {
        [x, y, z] => Some(DVec3::new(x.as_f64()?, y.as_f64()?, z.as_f64()?)),
        _ => None,
    }
}

/// Whether a field is "truthy" as JavaScript reads it.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(truthy_number),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(_)) | Some(Value::Object(_)) => true,
    }
}

/// Whether a field is there and not null.
fn present(v: Option<&Value>) -> bool {
    !matches!(v, None | Some(Value::Null))
}

/// A number field, or 0 when it has none (ScanRuler's `?? 0`).
fn num_or_zero(fields: &Map<String, Value>, key: &str) -> f64 {
    fields.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

/// A number field ScanRuler reads with no fallback: NaN when it has none.
fn num_or_nan(v: Option<&Value>) -> f64 {
    v.and_then(Value::as_f64).unwrap_or(f64::NAN)
}

/// A string field.
fn text<'a>(fields: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    fields.get(key).and_then(Value::as_str)
}

/// A list of strings.
fn strings(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default()
}

/// The object a field holds, or an empty one.
fn object(v: Option<&Value>) -> Map<String, Value> {
    v.and_then(Value::as_object).cloned().unwrap_or_default()
}

impl JsonFeature {
    /// A feature from its JSON, as ScanRuler writes it (the sketch, when it has one, as Polymix
    /// writes a sketch).
    pub fn from_value(value: Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value)
    }

    /// Its kind, or none for the draft of an extrude or a revolve.
    pub fn kind(&self) -> Option<&str> {
        text(&self.fields, "kind")
    }

    /// A field.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }

    /// It with a field set.
    pub fn with(&self, key: &str, value: Value) -> Self {
        let mut next = self.clone();
        next.fields.insert(key.to_string(), value);
        next
    }
}

// ---- planes --------------------------------------------------------------------------

/// Whether a plane takes an offset: any but a plane fitted where the scan was clicked.
pub fn plane_takes_offset(plane: &Value) -> bool {
    plane.get("kind").and_then(Value::as_str) != Some("fitted")
}

/// How far a plane is moved along its normal: a coordinate plane's coordinate, the offset of
/// any other — none for a plane fitted where the scan was clicked, which takes no offset.
pub fn plane_offset(plane: &Value) -> f64 {
    if !plane_takes_offset(plane) {
        return 0.0;
    }
    plane.get("offset").and_then(Value::as_f64).unwrap_or(0.0)
}

/// A frame (`{ origin, normal, basisU, basisV }`) carried `delta` along its normal; as it was
/// when it has no origin or normal to carry.
fn shift_frame(frame: &Value, delta: f64) -> Value {
    let (Some(origin), Some(normal)) = (vec3_of(frame.get("origin")), vec3_of(frame.get("normal"))) else {
        return frame.clone();
    };
    let mut out = object(Some(frame));
    out.insert("origin".to_string(), json_vec3(origin + normal * delta));
    Value::Object(out)
}

/// A plane moved to an offset along its normal — a face's last-found frame with it. One that
/// takes no offset comes back as it was.
pub fn plane_at_offset(plane: &Value, offset: f64) -> Value {
    if !plane_takes_offset(plane) || !offset.is_finite() {
        return plane.clone();
    }
    let mut out = object(Some(plane));
    if plane.get("kind").and_then(Value::as_str) == Some("bodyFace") {
        let delta = offset - plane_offset(plane);
        if let Some(frame) = plane.get("frame") {
            out.insert("frame".to_string(), shift_frame(frame, delta));
        }
    }
    out.insert("offset".to_string(), json_number(offset));
    Value::Object(out)
}

/// A sketch — or its draft — moved to an offset of its plane: the plane's offset set and the
/// sketch's frame carried along its normal by the change, a face's last-found frame with it.
/// Its pieces keep their sheet coordinates.
pub fn with_plane_offset(x: &JsonFeature, offset: f64) -> JsonFeature {
    let Some(plane) = x.get("plane") else { return x.clone() };
    if !plane_takes_offset(plane) || !offset.is_finite() {
        return x.clone();
    }
    let delta = offset - plane_offset(plane);
    if delta.abs() < 1e-12 {
        return x.clone();
    }
    let mut moved = object(Some(plane));
    if plane.get("kind").and_then(Value::as_str) == Some("bodyFace") {
        if let Some(frame) = plane.get("frame") {
            moved.insert("frame".to_string(), shift_frame(frame, delta));
        }
    }
    moved.insert("offset".to_string(), json_number(offset));
    let mut next = x.with("plane", Value::Object(moved));
    if let Some(frame) = x.get("frame") {
        next.fields.insert("frame".to_string(), shift_frame(frame, delta));
    }
    next
}

/// A frame as ScanRuler writes it (`origin`, `normal`, `basisU`, `basisV`), or none.
pub fn frame_of(v: Option<&Value>) -> Option<SectionFrame> {
    let v = v?;
    Some(SectionFrame { origin: vec3_of(v.get("origin"))?, normal: vec3_of(v.get("normal"))?, u: vec3_of(v.get("basisU"))?, v: vec3_of(v.get("basisV"))? })
}

/// A frame as ScanRuler writes it.
pub fn frame_json(f: &SectionFrame) -> Value {
    serde_json::json!({ "origin": json_vec3(f.origin), "normal": json_vec3(f.normal), "basisU": json_vec3(f.u), "basisV": json_vec3(f.v) })
}

// ---- the table ------------------------------------------------------------------------

type Get = Box<dyn Fn(&JsonFeature) -> f64>;

/// A slot on one top-level number field. `tidy` is what a typed number goes through — its
/// sign dropped, a count rounded.
fn field(path: &'static str, label: &str, kind: ParameterKind, tidy: fn(f64) -> f64) -> Slot<JsonFeature> {
    Slot::new(path, label, kind, move |y: &JsonFeature| num_or_zero(&y.fields, path), move |y: &JsonFeature, v| y.with(path, json_number(tidy(v))))
}

fn same(v: f64) -> f64 {
    v
}

/// A slot on a number inside an object field: `extent.to`, `route.bendRadius`.
fn inner(path: &str, label: &str, kind: ParameterKind, outer: &'static str, key: &'static str, get: Get, tidy: fn(f64) -> f64) -> Slot<JsonFeature> {
    Slot::new(path, label, kind, get, move |y: &JsonFeature, v| {
        let mut o = object(y.get(outer));
        o.insert(key.to_string(), json_number(tidy(v)));
        y.with(outer, Value::Object(o))
    })
}

fn pitch_of(y: &JsonFeature) -> f64 {
    num_or_zero(&y.fields, "pitch")
}

/// The slots a feature or a draft offers, as its box shows them. `kind` is the feature's own
/// kind unless given — a draft of an extrude or a revolve has none of its own.
pub fn json_slots(x: &JsonFeature, kind: Option<&str>) -> Vec<Slot<JsonFeature>> {
    let f = &x.fields;
    use ParameterKind::{Angle, Integer, Length, Number};
    match kind.or_else(|| x.kind()) {
        Some("extrude") => {
            let mut out = Vec::new();
            let ext = |key: &'static str, label: &str, tidy: fn(f64) -> f64| {
                let get: Get = Box::new(move |y: &JsonFeature| y.get("extent").and_then(|e| e.get(key)).and_then(Value::as_f64).unwrap_or(0.0));
                inner(&format!("extent.{key}"), label, Length, "extent", key, get, tidy)
            };
            // An extrude with no extent at all offers none of its numbers (ScanRuler throws on
            // one). An extent of any kind but these three — one with no kind among them — has a
            // start, as ScanRuler's `extent.kind !== 'betweenPlanes'` reads it.
            if let Some(extent) = f.get("extent").filter(|e| present(Some(e))) {
                match extent.get("kind").and_then(Value::as_str) {
                    Some("distance") => out.extend([ext("from", "From", same), ext("to", "To", same)]),
                    Some("symmetric") => out.push(ext("total", "Total", positive)),
                    Some("betweenPlanes") => {}
                    _ => out.push(ext("from", "From", same)),
                }
            }
            // Shown as a taper, Fusion's way round: positive leans the sides out.
            out.push(Slot::new(
                "taper",
                "Taper",
                Angle,
                |y: &JsonFeature| {
                    let d = num_or_zero(&y.fields, "draftDeg");
                    if d == 0.0 {
                        0.0
                    } else {
                        -d
                    }
                },
                |y: &JsonFeature, v| y.with("draftDeg", json_number(if v == 0.0 { 0.0 } else { -v })),
            ));
            out
        }
        Some("revolve") => {
            let pitch = num_or_zero(f, "pitch");
            let mut out = vec![Slot::new("angleDeg", "Angle", Angle, |y: &JsonFeature| num_or_zero(&y.fields, "angleDeg"), |y: &JsonFeature, v| y.with("angleDeg", json_number(revolve_angle(v, pitch_of(y)))))];
            if truthy_number(pitch) {
                out.push(field("pitch", "Pitch", Length, same));
                out.push(Slot::new("turns", "Turns", Number, |y: &JsonFeature| num_or_zero(&y.fields, "angleDeg") / 360.0, |y: &JsonFeature, v| y.with("angleDeg", json_number(revolve_angle(v.abs() * 360.0, pitch_of(y))))));
                out.push(Slot::new(
                    "height",
                    "Height",
                    Length,
                    |y: &JsonFeature| (pitch_of(y).abs() * num_or_zero(&y.fields, "angleDeg")) / 360.0,
                    |y: &JsonFeature, v| {
                        let p = pitch_of(y);
                        if truthy_number(p) {
                            y.with("angleDeg", json_number(revolve_angle((v.abs() / p.abs()) * 360.0, p)))
                        } else {
                            y.clone()
                        }
                    },
                ));
            }
            out
        }
        Some("shell") => vec![field("thickness", "Wall", Length, positive)],
        Some("fillet") => vec![field("radius", "Radius", Length, positive)],
        Some("chamfer") => {
            let mut out = vec![field("distance", "Distance", Length, positive)];
            match text(f, "mode") {
                Some("twoDistances") => out.push(field("distance2", "Distance 2", Length, positive)),
                Some("distanceAngle") => out.push(field("angleDeg", "Angle", Angle, |v| js_min(89.9, js_max(0.1, v.abs())))),
                _ => {}
            }
            out
        }
        Some("hole") => {
            let mut out = vec![Slot::new("diameter", "Diameter", Length, |y: &JsonFeature| 2.0 * num_or_zero(&y.fields, "radius"), |y: &JsonFeature, v| y.with("radius", json_number(v.abs() / 2.0)))];
            if present(f.get("depth")) {
                out.push(field("depth", "Depth", Length, positive));
            }
            out
        }
        Some("thread") => {
            let mut out = vec![field("diameter", "Diameter", Length, positive), field("pitch", "Pitch", Length, positive)];
            let runs = if f.contains_key("fullLength") { !truthy(f.get("fullLength")) } else { present(f.get("length")) };
            if runs {
                out.push(field("length", "Length", Length, positive));
                out.push(field("start", "Start", Length, positive));
            }
            out.push(field("offset", "Offset", Length, same));
            out
        }
        Some("extract") | Some("thicken") => vec![field("thickness", "Thickness", Length, positive)],
        Some("draft") => vec![field("angleDeg", "Angle", Angle, same)],
        Some("moveFace") => vec![field("offset", "Offset", Length, same)],
        Some("offsetBody") => vec![field("distance", "Distance", Length, same)],
        Some("moveBody") => {
            // Each of the six numbers on its own: `translate.x`, `rotate.z`.
            let axis = |key: &'static str, i: usize| {
                let letter = ["x", "y", "z"][i];
                let label = format!("{} {}", if key == "translate" { "Move" } else { "Turn" }, letter.to_uppercase());
                Slot::new(
                    format!("{key}.{letter}"),
                    label,
                    if key == "translate" { Length } else { Angle },
                    move |y: &JsonFeature| y.get(key).and_then(Value::as_array).and_then(|a| a.get(i)).and_then(Value::as_f64).unwrap_or(0.0),
                    move |y: &JsonFeature, v| {
                        let mut next = y.get(key).and_then(Value::as_array).cloned().unwrap_or_else(|| vec![json_number(0.0); 3]);
                        if next.len() <= i {
                            next.resize(i + 1, Value::Null);
                        }
                        next[i] = json_number(v);
                        y.with(key, Value::Array(next))
                    },
                )
            };
            (0..3).map(|i| axis("translate", i)).chain((0..3).map(|i| axis("rotate", i))).collect()
        }
        Some("sheet") => vec![field("from", "From", Length, same), field("to", "To", Length, same)],
        Some("sweep") => {
            if !truthy(f.get("route")) {
                return Vec::new();
            }
            let get: Get = Box::new(|y: &JsonFeature| num_or_nan(y.get("route").and_then(|r| r.get("bendRadius"))));
            vec![inner("route.bendRadius", "Bend radius", Length, "route", "bendRadius", get, positive)]
        }
        Some("pattern") => {
            let circular = text(f, "mode") == Some("circular");
            let mut out = vec![
                field("count", "Count", Integer, |v| js_max(2.0, js_round(v.abs()))),
                field("step", if circular { "Over" } else { "Spacing" }, if circular { Angle } else { Length }, positive),
            ];
            if text(f, "mode") == Some("linear") && truthy(f.get("axis2")) {
                out.push(field("count2", "Count 2", Integer, |v| js_max(1.0, js_round(v.abs()))));
                out.push(field("step2", "Spacing 2", Length, positive));
            }
            out
        }
        Some("sketch") => {
            // The plane's offset: a coordinate plane's coordinate, how far off a face or a
            // measured plane the sketch stands.
            let mut out = Vec::new();
            if let Some(plane) = f.get("plane").filter(|p| truthy(Some(p)) && plane_takes_offset(p)) {
                let label = if plane.get("kind").and_then(Value::as_str) == Some("world") {
                    format!("{} =", plane.get("axis").and_then(Value::as_str).unwrap_or("").to_uppercase())
                } else {
                    "Offset".to_string()
                };
                out.push(Slot::new("offset", label, Length, |y: &JsonFeature| y.get("plane").map_or(0.0, plane_offset), with_plane_offset));
            }
            // A profile moved in from another sketch: the wall it was moved by. The number
            // alone is written here; `apply_parameters` moves the profile in again.
            if truthy(f.get("inset")) {
                let get: Get = Box::new(|y: &JsonFeature| num_or_nan(y.get("inset").and_then(|i| i.get("by"))));
                out.push(inner("inset", "Inset", Length, "inset", "by", get, same));
            }
            out
        }
        Some("cutPlane") | Some("mirror") => {
            // The plane's offset, as a sketch's: a coordinate plane's coordinate, how far off a
            // face or a measured plane the cut or the mirror is.
            let Some(plane) = f.get("plane").filter(|p| truthy(Some(p))) else { return Vec::new() };
            if plane.get("kind").and_then(Value::as_str) == Some("face") || !plane_takes_offset(plane) {
                return Vec::new();
            }
            vec![Slot::new(
                "plane.offset",
                "Offset",
                Length,
                |y: &JsonFeature| y.get("plane").map_or(0.0, plane_offset),
                |y: &JsonFeature, v| match y.get("plane") {
                    Some(plane) => y.with("plane", plane_at_offset(plane, v)),
                    None => y.clone(),
                },
            )]
        }
        Some("surface") => {
            // The box the region was read in: each of its six numbers. A number moved leaves
            // the region to be read again in the box (`regionStale`).
            if !truthy(f.get("regionBox")) {
                return Vec::new();
            }
            let corner = |end: &'static str, i: usize| {
                let letter = ["x", "y", "z"][i];
                Slot::new(
                    format!("region.box.{end}.{letter}"),
                    format!("Box {end} {}", letter.to_uppercase()),
                    Length,
                    move |y: &JsonFeature| num_or_nan(y.get("regionBox").and_then(|r| r.get("box")).and_then(|b| b.get(end)).and_then(|c| c.get(i))),
                    move |y: &JsonFeature, v| {
                        let was = object(y.get("regionBox"));
                        let mut bx = object(was.get("box"));
                        let mut at = bx.get(end).and_then(Value::as_array).cloned().unwrap_or_default();
                        if at.len() <= i {
                            at.resize(i + 1, Value::Null);
                        }
                        at[i] = json_number(v);
                        bx.insert(end.to_string(), Value::Array(at));
                        let mut region_box = was;
                        region_box.insert("box".to_string(), Value::Object(bx));
                        y.with("regionBox", Value::Object(region_box)).with("regionStale", Value::Bool(true))
                    },
                )
            };
            ["min", "max"].into_iter().flat_map(|end| (0..3).map(move |i| corner(end, i))).collect()
        }
        Some("split") | Some("trim") => vec![
            field("tolerance", "Tolerance", Length, positive),
            Slot::new("overlap", "Overlap", Number, |y: &JsonFeature| num_or_zero(&y.fields, "overlap") * 100.0, |y: &JsonFeature, v| y.with("overlap", json_number(js_min(1.0, js_max(0.05, v.abs() / 100.0))))),
        ],
        _ => Vec::new(),
    }
}

/// The slot of a feature or a draft at a path, or none.
pub fn json_slot_at(x: &JsonFeature, path: &str, kind: Option<&str>) -> Option<Slot<JsonFeature>> {
    json_slots(x, kind).into_iter().find(|s| s.path == path)
}

/// Why a feature of a kind that takes expressions has no number for one, as a refusal
/// finishes its sentence: the plane of a cut or a mirror on a face, or fitted where the scan
/// was clicked; a surface whose region was not read in a box. Empty for any other.
pub fn no_slots_why(kind: Option<&str>, x: &JsonFeature) -> &'static str {
    let kind = kind.or_else(|| x.kind());
    let plane = x.get("plane").filter(|p| present(Some(p)));
    if let (Some("cutPlane") | Some("mirror"), Some(plane)) = (kind, plane) {
        return if plane.get("kind").and_then(Value::as_str) == Some("face") {
            ": its plane is a face of the body, which has no offset — give a plane for one"
        } else {
            ": its plane was fitted where the scan was clicked, and takes no offset"
        };
    }
    if kind == Some("surface") {
        return ": its region was not read in a box — give region: { box } with the expressions";
    }
    ""
}

impl Slotted for JsonFeature {
    fn slots(&self) -> Vec<Slot<Self>> {
        json_slots(self, None)
    }

    fn exprs(&self) -> Option<&ExprMap> {
        self.exprs.as_ref()
    }

    fn with_exprs(&self, exprs: Option<ExprMap>) -> Self {
        JsonFeature { exprs, ..self.clone() }
    }
}

impl ParametricFeature for JsonFeature {
    fn id(&self) -> FeatureId {
        self.get("id").and_then(Value::as_u64).and_then(|v| FeatureId::try_from(v).ok()).unwrap_or(0)
    }

    fn name(&self) -> &str {
        text(&self.fields, "name").unwrap_or("")
    }

    fn sketch(&self) -> Option<&Sketch> {
        self.sketch.as_ref()
    }

    fn sketch_frame(&self) -> Option<SectionFrame> {
        frame_of(self.get("frame"))
    }

    fn with_sketch(&self, sketch: Sketch) -> Self {
        JsonFeature { sketch: Some(sketch), ..self.clone() }
    }

    fn source_sketch(&self) -> Option<FeatureId> {
        if !matches!(self.kind(), Some("extrude") | Some("revolve")) {
            return None;
        }
        self.get("sketchId").and_then(Value::as_u64).and_then(|v| FeatureId::try_from(v).ok())
    }

    fn inset(&self) -> Option<SketchInset> {
        self.get("inset").and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    fn with_inset(&self, inset: Option<SketchInset>) -> Self {
        let mut next = self.clone();
        match inset {
            Some(i) => {
                let keep = i.keep.iter().map(|&k| Value::from(k)).collect();
                next.fields.insert("inset".to_string(), serde_json::json!({ "of": i.of, "by": json_number(i.by), "keep": Value::Array(keep) }));
            }
            None => {
                next.fields.remove("inset");
            }
        }
        next
    }
}

impl RegionPicks for JsonFeature {
    fn regions(&self) -> Vec<String> {
        strings(self.get("regions"))
    }

    fn region_pick(&self) -> Option<RegionPick> {
        self.get("regionPick").and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    fn regions_by_place(&self) -> Option<Vec<String>> {
        self.get("regionsByPlace").filter(|v| present(Some(v))).map(|v| strings(Some(v)))
    }

    /// A pick of none leaves the feature's `regionPick` as it stands — ScanRuler's `rekeyed`
    /// never takes one away, and one this cannot read (`region_pick` none) is kept as written.
    fn with_picks(&self, regions: Vec<String>, pick: Option<RegionPick>, by_place: Option<Vec<String>>) -> Self {
        let mut next = self.with("regions", Value::from(regions));
        if let Some(p) = pick.and_then(|p| serde_json::to_value(p).ok()) {
            next.fields.insert("regionPick".to_string(), p);
        }
        match by_place {
            Some(b) => {
                next.fields.insert("regionsByPlace".to_string(), Value::from(b));
            }
            None => {
                next.fields.remove("regionsByPlace");
            }
        }
        next
    }
}

// ---- Polymix's model -----------------------------------------------------------------

// mm-reverse's model (`crate::model`) keeps a feature in ScanRuler's JSON shape — the same
// fields under the same names — so its slots are ScanRuler's own table read through that JSON:
// the feature written out, the slot read or set on it, and the feature read back.

/// A feature of the model as ScanRuler writes it.
pub fn feature_json(f: &Feature) -> JsonFeature {
    serde_json::to_value(f).ok().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

/// The feature a slot left, read back from its JSON: the feature as it was where the JSON no
/// longer reads as one (which a slot never leaves it).
fn feature_from_json(x: &JsonFeature, was: &Feature) -> Feature {
    serde_json::to_value(x).ok().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_else(|| was.clone())
}

/// The slots a feature of the model offers, as ScanRuler's box shows the same kind (see the
/// table at the head of this module).
pub fn model_slots(f: &Feature) -> Vec<Slot<Feature>> {
    let x = feature_json(f);
    let kind = x.fields.get("kind").and_then(Value::as_str).map(str::to_string);
    json_slots(&x, kind.as_deref())
        .into_iter()
        .map(|slot| {
            let (path, label, kind) = (slot.path.clone(), slot.label.clone(), slot.kind);
            let slot = std::rc::Rc::new(slot);
            let get = slot.clone();
            Slot::new(path, label, kind, move |f: &Feature| get.get(&feature_json(f)), move |f: &Feature, v| feature_from_json(&slot.set(&feature_json(f), v), f))
        })
        .collect()
}

impl Slotted for Feature {
    fn slots(&self) -> Vec<Slot<Self>> {
        model_slots(self)
    }

    fn exprs(&self) -> Option<&ExprMap> {
        self.kind.exprs().filter(|e| !e.is_empty())
    }

    fn with_exprs(&self, exprs: Option<ExprMap>) -> Self {
        let mut next = self.clone();
        if let Some(e) = next.kind.exprs_mut() {
            *e = exprs.unwrap_or_default();
        }
        next
    }
}

impl ParametricFeature for Feature {
    fn id(&self) -> FeatureId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn sketch(&self) -> Option<&Sketch> {
        match &self.kind {
            FeatureKind::Sketch(s) => Some(&s.sketch),
            _ => None,
        }
    }

    fn sketch_frame(&self) -> Option<SectionFrame> {
        match &self.kind {
            FeatureKind::Sketch(s) => Some(s.frame),
            _ => None,
        }
    }

    fn with_sketch(&self, sketch: Sketch) -> Self {
        let mut next = self.clone();
        if let FeatureKind::Sketch(s) = &mut next.kind {
            s.sketch = sketch;
        }
        next
    }

    fn source_sketch(&self) -> Option<FeatureId> {
        match &self.kind {
            FeatureKind::Extrude(e) => Some(e.sketch_id),
            FeatureKind::Revolve(r) => Some(r.sketch_id),
            _ => None,
        }
    }

    fn inset(&self) -> Option<SketchInset> {
        match &self.kind {
            FeatureKind::Sketch(s) => s.inset.clone(),
            _ => None,
        }
    }

    fn with_inset(&self, inset: Option<SketchInset>) -> Self {
        let mut next = self.clone();
        if let FeatureKind::Sketch(s) = &mut next.kind {
            s.inset = inset;
        }
        next
    }
}

/// A model's region pick as the parameters read one (the same JSON).
fn pick_of(p: &crate::model::RegionPick) -> Option<RegionPick> {
    serde_json::to_value(p).ok().and_then(|v| serde_json::from_value(v).ok())
}

/// A region pick of the parameters as the model keeps one.
fn model_pick(p: &RegionPick) -> Option<crate::model::RegionPick> {
    serde_json::to_value(p).ok().and_then(|v| serde_json::from_value(v).ok())
}

impl RegionPicks for Feature {
    fn regions(&self) -> Vec<String> {
        match &self.kind {
            FeatureKind::Extrude(e) => e.regions.clone(),
            FeatureKind::Revolve(r) => r.regions.clone(),
            _ => Vec::new(),
        }
    }

    fn region_pick(&self) -> Option<RegionPick> {
        match &self.kind {
            FeatureKind::Extrude(e) => e.region_pick.as_ref().and_then(pick_of),
            FeatureKind::Revolve(r) => r.region_pick.as_ref().and_then(pick_of),
            _ => None,
        }
    }

    fn regions_by_place(&self) -> Option<Vec<String>> {
        let v = match &self.kind {
            FeatureKind::Extrude(e) => &e.regions_by_place,
            FeatureKind::Revolve(r) => &r.regions_by_place,
            _ => return None,
        };
        (!v.is_empty()).then(|| v.clone())
    }

    fn with_picks(&self, regions: Vec<String>, pick: Option<RegionPick>, by_place: Option<Vec<String>>) -> Self {
        let mut next = self.clone();
        let (r, p, b) = match &mut next.kind {
            FeatureKind::Extrude(e) => (&mut e.regions, &mut e.region_pick, &mut e.regions_by_place),
            FeatureKind::Revolve(r) => (&mut r.regions, &mut r.region_pick, &mut r.regions_by_place),
            _ => return next,
        };
        *r = regions;
        if let Some(pick) = pick.as_ref().and_then(model_pick) {
            *p = Some(pick);
        }
        *b = by_place.unwrap_or_default();
        next
    }
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reParameters.test.ts` ("slots",
    //! "sketches", "who reads a parameter"), the slot-level parts of `reExprsEveryKind.test.ts`
    //! (a cut's and a mirror's plane offset, a loft station's wall, a surface's box, a kind with
    //! no numbers) and the parts of `reParameterStore.test.ts` that follow a parameter through
    //! the features. The commands, the kernel builds and the store's undo, grips and drafts
    //! around them are not this crate's; their checks here are the slots' and the features'
    //! numbers they come down to.
    use std::borrow::Cow;

    use glam::DVec2;
    use serde_json::json;

    use super::*;
    use crate::datums::DatumTrace;
    use crate::expression::{NoNames, ParameterKind};
    use crate::frame::WorldAxis;
    use crate::model::{AxisRef, BodyId, BodyOp, BooleanFeature, Exprs, BooleanOp, ChamferFeature, ExtrudeExtent, ExtrudeFeature, FeatureFlags, FilletFeature, MirrorFeature, MirrorPlane, PlaneRef, RevolveFeature, ShellFeature, SketchFeature};
    use crate::parameters::*;
    use crate::sketch::{Anchor, AnchorAt, ConstraintId, Entity, Rule, Shape};

    fn feature(v: Value) -> JsonFeature {
        JsonFeature::from_value(v).expect("a feature's JSON")
    }
    fn xy() -> Value {
        json!({ "origin": [0, 0, 0], "normal": [0, 0, 1], "basisU": [1, 0, 0], "basisV": [0, 1, 0] })
    }
    fn xy_frame() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }
    fn extrude() -> JsonFeature {
        feature(json!({ "id": 2, "kind": "extrude", "name": "Extrude 1", "sketchId": 1, "regions": ["r:1"], "extent": { "kind": "distance", "from": 0, "to": 10 }, "draftDeg": 0, "op": "new", "visible": true }))
    }
    fn paths(x: &JsonFeature) -> Vec<String> {
        json_slots(x, None).into_iter().map(|s| s.path).collect()
    }
    fn slot(x: &JsonFeature, path: &str) -> Slot<JsonFeature> {
        json_slot_at(x, path, None).expect("the slot")
    }
    fn num(x: &JsonFeature, path: &[&str]) -> f64 {
        let mut v = x.get(path[0]);
        for key in &path[1..] {
            v = v.and_then(|o| o.get(*key));
        }
        v.and_then(Value::as_f64).unwrap_or(f64::NAN)
    }
    fn param(id: u32, name: &str, expression: &str) -> Parameter {
        Parameter { id, name: name.to_string(), kind: ParameterKind::Length, expression: expression.to_string(), comment: None }
    }
    fn table(ps: &[(&str, &str)]) -> ParameterTable {
        let list: Vec<Parameter> = ps.iter().enumerate().map(|(i, (n, e))| param(i as u32 + 1, n, e)).collect();
        evaluate_parameters(&list)
    }
    fn exprs(pairs: &[(&str, &str)]) -> Option<ExprMap> {
        Some(pairs.iter().copied().collect())
    }
    fn errors(pairs: &[(&str, &str)]) -> ExprErrors {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn slots_read_and_write_an_extrude_as_its_box_shows_it() {
        let x = extrude();
        assert_eq!(paths(&x), ["extent.from", "extent.to", "taper"]);
        let to = slot(&x, "extent.to");
        assert_eq!(to.get(&x), 10.0);
        assert!(json_eq(to.set(&x, 25.0).get("extent").unwrap_or(&Value::Null), &json!({ "kind": "distance", "from": 0, "to": 25 })));
        let taper = slot(&x, "taper");
        assert_eq!(taper.get(&x.with("draftDeg", json!(-5))), 5.0);
        assert_eq!(num(&taper.set(&x, 5.0), &["draftDeg"]), -5.0);
        assert_eq!(num(&taper.set(&x, 0.0), &["draftDeg"]), 0.0);
        assert_eq!(paths(&x.with("extent", json!({ "kind": "symmetric", "total": 8 }))), ["extent.total", "taper"]);
        assert_eq!(paths(&x.with("extent", json!({ "kind": "toNext", "from": 2 }))), ["extent.from", "taper"]);
        let between = json!({ "kind": "betweenPlanes", "a": { "kind": "world", "axis": "z", "offset": 0 }, "b": { "kind": "world", "axis": "z", "offset": 5 } });
        assert_eq!(paths(&x.with("extent", between)), ["taper"]);
    }

    #[test]
    fn slots_read_and_write_the_others() {
        let hole = feature(json!({ "id": 3, "kind": "hole", "name": "Hole 1", "body": 2, "axis": { "kind": "element", "id": 1 }, "radius": 3, "depth": null, "visible": true }));
        assert_eq!(paths(&hole), ["diameter"]);
        assert_eq!(slot(&hole, "diameter").get(&hole), 6.0);
        assert_eq!(num(&slot(&hole, "diameter").set(&hole, 10.0), &["radius"]), 5.0);
        assert_eq!(paths(&hole.with("depth", json!(4))), ["diameter", "depth"]);
        let revolve = feature(json!({ "id": 4, "kind": "revolve", "name": "Revolve 1", "sketchId": 1, "regions": [], "axis": { "kind": "world", "axis": "z" }, "angleDeg": 180, "op": "new", "visible": true }));
        assert_eq!(paths(&revolve), ["angleDeg"]);
        assert_eq!(num(&slot(&revolve, "angleDeg").set(&revolve, 720.0), &["angleDeg"]), 360.0);
        let coil = revolve.with("pitch", json!(2)).with("angleDeg", json!(720));
        assert_eq!(paths(&coil), ["angleDeg", "pitch", "turns", "height"]);
        assert_eq!(slot(&coil, "turns").get(&coil), 2.0);
        assert_eq!(num(&slot(&coil, "turns").set(&coil, 3.0), &["angleDeg"]), 1080.0);
        assert_eq!(slot(&coil, "height").get(&coil), 4.0);
        assert_eq!(num(&slot(&coil, "height").set(&coil, 10.0), &["angleDeg"]), 1800.0);
        let pattern = feature(json!({ "id": 5, "kind": "pattern", "name": "Pattern 1", "body": 2, "mode": "circular", "axis": { "kind": "world", "axis": "z" }, "count": 4, "step": 360, "visible": true }));
        let kinds = |x: &JsonFeature| json_slots(x, None).into_iter().map(|s| (s.path, s.kind)).collect::<Vec<_>>();
        assert_eq!(kinds(&pattern), [("count".to_string(), ParameterKind::Integer), ("step".to_string(), ParameterKind::Angle)]);
        assert_eq!(num(&slot(&pattern, "count").set(&pattern, 7.3), &["count"]), 7.0);
        assert_eq!(num(&slot(&pattern, "count").set(&pattern, 1.0), &["count"]), 2.0);
        assert_eq!(kinds(&pattern.with("mode", json!("linear"))), [("count".to_string(), ParameterKind::Integer), ("step".to_string(), ParameterKind::Length)]);
        let split = feature(json!({ "id": 6, "kind": "split", "name": "Split 1", "body": 2, "tools": [], "keep": "scan", "tolerance": 0.3, "overlap": 0.5, "visible": true }));
        assert_eq!(slot(&split, "overlap").get(&split), 50.0);
        assert_eq!(num(&slot(&split, "overlap").set(&split, 80.0), &["overlap"]), 0.8);
        let shell = feature(json!({ "id": 7, "kind": "shell", "name": "Shell 1", "body": 2, "faces": [], "thickness": 2, "visible": true }));
        assert_eq!(num(&slot(&shell, "thickness").set(&shell, -3.0), &["thickness"]), 3.0);
        // A sketch's one slot is its plane's offset: moving it carries the frame.
        let mut sketch = feature(json!({ "id": 1, "kind": "sketch", "name": "Sketch 1", "plane": { "kind": "world", "axis": "z", "offset": 0 }, "frame": xy(), "visible": true }));
        sketch.sketch = Some(Sketch::default());
        assert_eq!(paths(&sketch), ["offset"]);
        let lifted = slot(&sketch, "offset").set(&sketch, 7.0);
        assert!(json_eq(lifted.get("plane").unwrap_or(&Value::Null), &json!({ "kind": "world", "axis": "z", "offset": 7 })));
        assert_eq!(lifted.sketch_frame().map(|f| f.origin), Some(DVec3::new(0.0, 0.0, 7.0)));
        let fitted = sketch.with("plane", json!({ "kind": "fitted", "frame": xy() }));
        assert!(json_slots(&fitted, None).is_empty());
    }

    #[test]
    fn slots_write_an_expression_through_and_let_it_go_for_a_number() {
        let x = extrude();
        let scope = table(&[("height", "20")]);
        let bound = with_slot(&x, "extent.to", 25.0, Some("height + 5"));
        assert_eq!(num(&bound, &["extent", "to"]), 25.0);
        assert_eq!(bound.exprs, exprs(&[("extent.to", "height + 5")]));
        let r = apply_slots(&bound, &table(&[("height", "30")]));
        assert!(r.errors.is_empty());
        assert_eq!(num(&r.next, &["extent", "to"]), 35.0);
        assert!(matches!(apply_slots(&bound, &scope).next, Cow::Borrowed(_)));
        let typed = with_slot(&bound, "extent.to", 12.0, None);
        assert_eq!(num(&typed, &["extent", "to"]), 12.0);
        assert_eq!(typed.exprs, None);
        assert_eq!(unlink_slot(&bound, "extent.to").exprs, None);
        assert!(matches!(unlink_slot(&x, "extent.to"), Cow::Borrowed(_)));
    }

    #[test]
    fn slots_leave_the_number_standing_when_the_expression_cannot_be_read_and_drop_one_with_no_slot() {
        let bound = with_slot(&extrude(), "extent.to", 25.0, Some("height + 5"));
        let r = apply_slots(&bound, &NoNames);
        assert!(matches!(r.next, Cow::Borrowed(_)));
        assert_eq!(r.errors, errors(&[("extent.to", "there is no parameter “height”")]));
        assert_eq!(feature_expression_errors(&bound, &NoNames), errors(&[("extent.to", "there is no parameter “height”")]));
        let stale = JsonFeature { exprs: exprs(&[("extent.to", "height"), ("taper", "2")]), ..bound.with("extent", json!({ "kind": "symmetric", "total": 8 })) };
        let dropped = apply_slots(&stale, &table(&[("height", "20")])).next;
        assert_eq!(dropped.exprs, exprs(&[("taper", "2")]));
        assert_eq!(num(&dropped, &["draftDeg"]), -2.0);
    }

    /// Not in ScanRuler's tests: the errors come in the order the expressions were read, as
    /// ScanRuler's object keeps its keys — a sketch's slots before its dimensions.
    #[test]
    fn slots_report_their_errors_in_the_order_read() {
        let x = driven(extrude(), &[("taper", "lean"), ("extent.to", "tall")]);
        let order = |e: &ExprErrors| e.paths().map(str::to_string).collect::<Vec<_>>();
        assert_eq!(order(&apply_slots(&x, &NoNames).errors), ["taper", "extent.to"]);
        assert_eq!(order(&feature_expression_errors(&x, &NoNames)), ["taper", "extent.to"]);
        let (sketch, width) = square(10.0);
        let placed = driven(sketch_feature(1, sketch), &[("offset", "z")]);
        let r = apply(&[placed], &NoNames);
        assert_eq!(r.errors.get(&1).map(order), Some(vec!["offset".to_string(), format!("dim:{width}")]));
    }

    /// A square of `size` from the origin, its bottom side dimensioned to `width`.
    fn square(size: f64) -> (Sketch, ConstraintId) {
        let mut sketch = Sketch::default();
        let ids = sketch.add_rectangle(DVec2::ZERO, DVec2::splat(size));
        let origin = sketch.add(Entity { fixed: true, ..Entity::new(Shape::Point { p: DVec2::ZERO }) });
        sketch.constrain(Rule::Coincident { p: Anchor::new(ids[0], AnchorAt::A), q: Anchor::new(origin, AnchorAt::Center) });
        let width = sketch.constrain(Rule::Length { entity: ids[0], value: size });
        if let Some(c) = sketch.constraint_mut(width) {
            c.expr = Some("width".to_string());
        }
        sketch.constrain(Rule::Length { entity: ids[1], value: size });
        (sketch, width)
    }
    fn bottom_length(sketch: &Sketch) -> f64 {
        match sketch.entities[0].shape {
            Shape::Line { a, b } => (b - a).length(),
            _ => f64::NAN,
        }
    }
    fn dim_value(sketch: &Sketch, id: ConstraintId) -> Option<f64> {
        sketch.constraint(id).and_then(|c| c.rule.value())
    }

    #[test]
    fn sketches_write_a_dimensions_expression_through_and_solve_the_sketch_again_on_it() {
        let (sketch, width) = square(10.0);
        let scope = table(&[("width", "30")]);
        let r = apply_sketch_expressions(&sketch, &xy_frame(), &scope);
        assert!(r.errors.is_empty());
        let Cow::Owned(next) = r.next else { panic!("the sketch did not change") };
        assert!((bottom_length(&next) - 30.0).abs() < 0.5e-6, "{}", bottom_length(&next));
        assert!((bottom_length(&sketch) - 10.0).abs() < 0.5e-6);
        assert_eq!(dim_value(&next, width), Some(30.0));
        assert!(matches!(apply_sketch_expressions(&next, &xy_frame(), &scope).next, Cow::Borrowed(_)));
    }

    #[test]
    fn sketches_leave_a_dimension_whose_expression_cannot_be_read_as_it_is() {
        let (sketch, width) = square(10.0);
        let r = apply_sketch_expressions(&sketch, &xy_frame(), &NoNames);
        assert!(matches!(r.next, Cow::Borrowed(_)));
        assert_eq!(r.errors, errors(&[(&format!("dim:{width}"), "there is no parameter “width”")]));
    }

    fn sketch_feature(id: u32, sketch: Sketch) -> JsonFeature {
        JsonFeature { sketch: Some(sketch), ..feature(json!({ "id": id, "kind": "sketch", "name": format!("Sketch {id}"), "plane": { "kind": "world", "axis": "z", "offset": 0 }, "frame": xy(), "visible": true })) }
    }

    /// No regions module here (mm-reverse's needs the kernel), so the features on a changed
    /// sketch keep their keys as they are — as a pick that matches nothing is kept.
    fn keep_keys(fs: &[JsonFeature], id: FeatureId, s: Sketch) -> Vec<JsonFeature> {
        replace_sketch(fs, id, &s, &mut |_| None)
    }
    fn no_inset(_: &JsonFeature, _: &[JsonFeature]) -> Result<Sketch, String> {
        Err("no profile moved in here".to_string())
    }
    fn apply(fs: &[JsonFeature], scope: &dyn crate::expression::Scope) -> ParametersApplied<JsonFeature> {
        apply_parameters(fs, scope, &mut keep_keys, &mut no_inset)
    }

    #[test]
    fn sketches_keep_the_features_on_a_changed_sketch_pointing_at_their_regions() {
        let (sketch, width) = square(10.0);
        let features = vec![
            sketch_feature(1, sketch.clone()),
            JsonFeature { exprs: exprs(&[("extent.to", "width / 2")]), ..feature(json!({ "id": 2, "kind": "extrude", "name": "Extrude 1", "sketchId": 1, "regions": ["r:1"], "extent": { "kind": "distance", "from": 0, "to": 10 }, "op": "new", "visible": true })) },
            feature(json!({ "id": 3, "kind": "fillet", "name": "Fillet 1", "body": 2, "edges": [], "radius": 1, "chain": true, "visible": true })),
        ];
        let wide = table(&[("width", "30")]);
        let r = apply(&features, &wide);
        assert!(r.changed);
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert_ne!(r.features[0].sketch.as_ref(), Some(&sketch));
        assert_eq!(num(&r.features[1], &["extent", "to"]), 15.0);
        assert_eq!(r.features[1].regions().len(), 1);
        // The fillet read nothing and is as it was.
        assert_eq!(r.features[2], features[2]);
        // Nothing moves a second time.
        let again = apply(&r.features, &wide);
        assert!(!again.changed);
        assert_eq!(again.features, r.features);
        // A parameter that cannot be read leaves everything standing and says so.
        let broken = apply(&r.features, &table(&[("width", "nothing")]));
        assert!(!broken.changed);
        assert!(broken.errors.get(&1).and_then(|e| e.get(&format!("dim:{width}"))).is_some_and(|e| e.contains("“width” has no value")), "{:?}", broken.errors);
        assert!(broken.errors.get(&2).and_then(|e| e.get("extent.to")).is_some_and(|e| e.contains("“width” has no value")));
    }

    /// The regions of a sketch where no key matches any more.
    struct NoRegions;
    impl RegionMatch for NoRegions {
        fn key(&self, _region: usize) -> &str {
            ""
        }
        fn match_region(&self, _key: &str) -> Option<usize> {
            None
        }
        fn regions_by_pick(&self, _given: &PickGiven) -> Option<Vec<usize>> {
            None
        }
        fn was_region(&self, _key: &str) -> bool {
            false
        }
        fn region_by_place(&self, _was_key: &str, _taken: &[usize]) -> Option<usize> {
            None
        }
        fn split_parts(&self, _was_key: &str) -> Vec<usize> {
            Vec::new()
        }
    }

    #[test]
    fn sketches_replace_a_sketch_and_rekey_the_features_on_it() {
        let features = vec![sketch_feature(1, square(10.0).0), feature(json!({ "id": 2, "kind": "extrude", "name": "Extrude 1", "sketchId": 1, "regions": ["r:nothing"], "extent": { "kind": "distance", "from": 0, "to": 10 }, "op": "new", "visible": true }))];
        let other = square(20.0).0;
        let next = replace_sketch(&features, 1, &other, &mut |f| match rekeyed(f, &NoRegions) {
            Cow::Owned(g) => Some(g),
            Cow::Borrowed(_) => None,
        });
        // ScanRuler's patch (the new name) is the caller's to write.
        let renamed = next[0].with("name", json!("Renamed"));
        assert_eq!(renamed.name(), "Renamed");
        assert_eq!(renamed.sketch.as_ref(), Some(&other));
        // A pick that matches nothing is kept as it was, for the feature to report.
        assert_eq!(next[1].regions(), ["r:nothing"]);
    }

    fn readers() -> (Vec<Parameter>, Vec<JsonFeature>) {
        let parameters = vec![param(1, "width", "40"), param(2, "height", "width / 2"), param(3, "depth", "5")];
        let features = vec![
            sketch_feature(1, square(10.0).0),
            JsonFeature { exprs: exprs(&[("extent.to", "height + 1")]), ..feature(json!({ "id": 2, "kind": "extrude", "name": "Extrude 1", "sketchId": 1, "regions": [], "extent": { "kind": "distance", "from": 0, "to": 10 }, "op": "new", "visible": true })) },
        ];
        (parameters, features)
    }

    #[test]
    fn who_reads_a_parameter_lists_them() {
        let (parameters, features) = readers();
        let user = |kind, id, name: &str| ParameterUser { kind, id, name: name.to_string() };
        assert_eq!(users_of("width", &parameters, &features), [user(UserKind::Parameter, 2, "height"), user(UserKind::Feature, 1, "Sketch 1")]);
        assert_eq!(users_of("height", &parameters, &features), [user(UserKind::Feature, 2, "Extrude 1")]);
        assert!(users_of("depth", &parameters, &features).is_empty());
    }

    #[test]
    fn who_reads_a_parameter_renames_it_everywhere() {
        let (parameters, features) = readers();
        let renamed = rename_in_parameters(&parameters, 1, "w");
        let rows: Vec<(&str, &str)> = renamed.iter().map(|p| (p.name.as_str(), p.expression.as_str())).collect();
        assert_eq!(rows, [("w", "40"), ("height", "w / 2"), ("depth", "5")]);
        let s = rename_in_feature(&features[0], "width", "w");
        let dim = s.sketch.as_ref().and_then(|k| k.constraints.iter().find(|c| c.rule.is_dimension() && c.expr.is_some())).and_then(|c| c.expr.clone());
        assert_eq!(dim.as_deref(), Some("w"));
        assert!(matches!(rename_in_feature(&features[1], "width", "w"), Cow::Borrowed(_)));
        assert_eq!(rename_in_feature(&features[1], "height", "h").exprs, exprs(&[("extent.to", "h + 1")]));
    }

    /// Not in ScanRuler's tests: where a feature reads a parameter, and the parameters that
    /// drive nothing.
    #[test]
    fn who_reads_a_parameter_where_and_which_drive_nothing() {
        let (mut parameters, features) = readers();
        assert_eq!(fields_reading(&features[0], "width"), [format!("dimension {}", square(10.0).1)]);
        assert_eq!(fields_reading(&features[1], "height"), ["extent.to"]);
        parameters.push(param(4, "spare", "depth * 2"));
        let idle: Vec<&str> = idle_parameters(&parameters, &features).into_iter().map(|p| p.name.as_str()).collect();
        assert_eq!(idle, ["depth", "spare"]);
    }

    // ---- reExprsEveryKind: every kind with numbers ----------------------------------------

    fn driven(x: JsonFeature, pairs: &[(&str, &str)]) -> JsonFeature {
        JsonFeature { exprs: exprs(pairs), ..x }
    }

    #[test]
    fn every_kind_drives_a_cuts_plane_offset_and_follows_the_parameter() {
        let cut = feature(json!({ "id": 3, "kind": "cutPlane", "name": "Shoulder", "body": 2, "plane": { "kind": "world", "axis": "z", "offset": 5 }, "keep": -1 }));
        assert_eq!(paths(&cut), ["plane.offset"]);
        let edited = driven(cut, &[("plane.offset", "z_cut + 1")]);
        let at3 = apply_slots(&edited, &table(&[("z_cut", "3")])).next.into_owned();
        assert_eq!(num(&at3, &["plane", "offset"]), 4.0);
        let at6 = apply_slots(&at3, &table(&[("z_cut", "6")])).next.into_owned();
        assert_eq!(num(&at6, &["plane", "offset"]), 7.0);
        assert_eq!(at6.exprs, exprs(&[("plane.offset", "z_cut + 1")]));
    }

    #[test]
    fn every_kind_drives_a_mirrors_plane_offset_and_refuses_one_on_a_face() {
        let mirror = feature(json!({ "id": 3, "kind": "mirror", "name": "Mirror", "body": 2, "plane": { "kind": "world", "axis": "x", "offset": 0 }, "join": false }));
        let edited = driven(mirror.clone(), &[("plane.offset", "mx")]);
        let at20 = apply_slots(&edited, &table(&[("mx", "20")])).next.into_owned();
        assert_eq!(num(&at20, &["plane", "offset"]), 20.0);
        let at25 = apply_slots(&at20, &table(&[("mx", "25")])).next.into_owned();
        assert_eq!(num(&at25, &["plane", "offset"]), 25.0);
        // A face of the body has no offset, and the answer says so.
        let on_face = mirror.with("plane", json!({ "kind": "face", "face": { "kind": "plane", "between": [[1, 0, 0]] } }));
        assert!(json_slots(&on_face, None).is_empty());
        assert!(no_slots_why(None, &on_face).contains("a face of the body, which has no offset"));
        let fitted = mirror.with("plane", json!({ "kind": "fitted", "frame": xy() }));
        assert!(json_slots(&fitted, None).is_empty());
        assert!(no_slots_why(None, &fitted).contains("takes no offset"));
        // A face's last-found frame is carried with its offset.
        let face = json!({ "kind": "bodyFace", "body": 2, "face": {}, "frame": xy(), "offset": 1 });
        let moved = plane_at_offset(&face, 4.0);
        assert_eq!(plane_offset(&moved), 4.0);
        assert_eq!(frame_of(moved.get("frame")).map(|f| f.origin), Some(DVec3::new(0.0, 0.0, 3.0)));
    }

    #[test]
    fn every_kind_drives_a_loft_stations_wall_moving_the_profile_in_again() {
        // The loft's own `inner` and `sections.<k>.offset` are written by the command onto its
        // station sketches: the wall onto an inner station's inset, the plane onto its offset.
        let foot = sketch_feature(1, {
            let mut s = Sketch::default();
            s.add_rectangle(DVec2::splat(-10.0), DVec2::splat(10.0));
            s
        });
        let inner = sketch_feature(5, Sketch::default()).with("name", json!("Boss inner 1")).with("inset", json!({ "of": 1, "by": 1, "keep": [] }));
        assert_eq!(paths(&inner), ["offset", "inset"]);
        assert_eq!(slot(&inner, "offset").label, "Z =");
        let inner = driven(inner, &[("inset", "wall"), ("offset", "z_top")]);
        // The move itself is the stations' (`insetChains`): here a rectangle moved in by the wall.
        let chains = |of: &Sketch, inset: &SketchInset, _: &[DatumTrace], _: &str| -> Result<Sketch, String> {
            let (lo, hi) = of.bounds().ok_or("an empty profile")?;
            if (hi - lo).min_element() <= 2.0 * inset.by {
                return Err(format!("a wall of {} closes the profile", inset.by));
            }
            let mut s = Sketch::default();
            s.add_rectangle(lo + DVec2::splat(inset.by), hi - DVec2::splat(inset.by));
            Ok(s)
        };
        let width = |f: &JsonFeature| f.sketch.as_ref().and_then(Sketch::bounds).map_or(f64::NAN, |(lo, hi)| hi.x - lo.x);
        let features = vec![foot, inner];
        let r = apply_parameters(&features, &table(&[("wall", "1.5"), ("z_top", "12")]), &mut keep_keys, &mut |f, fs| inset_again(f, fs, &chains));
        assert!(r.changed && r.errors.is_empty(), "{:?}", r.errors);
        assert!((width(&r.features[1]) - 17.0).abs() < 1e-6, "{}", width(&r.features[1]));
        assert_eq!(r.features[1].inset().map(|i| i.by), Some(1.5));
        assert_eq!(num(&r.features[1], &["plane", "offset"]), 12.0);
        assert_eq!(r.features[1].sketch_frame().map(|f| f.origin.z), Some(12.0));
        // A wall that closes the profile is refused, and the wall stays what it was.
        let closed = apply_parameters(&r.features, &table(&[("wall", "12"), ("z_top", "12")]), &mut keep_keys, &mut |f, fs| inset_again(f, fs, &chains));
        assert_eq!(closed.errors.get(&5).and_then(|e| e.get("inset")), Some("a wall of 12 closes the profile"));
        assert_eq!(closed.features[1].inset().map(|i| i.by), Some(1.5));
        // A station whose source is gone says so.
        let gone = apply_parameters(&r.features[1..], &table(&[("wall", "2"), ("z_top", "12")]), &mut keep_keys, &mut |f, fs| inset_again(f, fs, &chains));
        assert_eq!(gone.errors.get(&5).and_then(|e| e.get("inset")), Some("the sketch it was moved in from (1) is gone"));
    }

    #[test]
    fn every_kind_drives_the_box_a_surfaces_region_is_read_in() {
        let skin = feature(json!({ "id": 4, "kind": "surface", "name": "Skin", "regionBox": { "box": { "min": [-25, -20, -1], "max": [25, 20, 10] }, "normal": { "dir": [0, 0, -1], "withinDeg": 40 } } }));
        assert_eq!(paths(&skin), ["region.box.min.x", "region.box.min.y", "region.box.min.z", "region.box.max.x", "region.box.max.y", "region.box.max.z"]);
        let made = apply_slots(&driven(skin, &[("region.box.max.x", "bx")]), &table(&[("bx", "0")])).next.into_owned();
        assert!(json_eq(made.get("regionBox").and_then(|r| r.get("box")).and_then(|b| b.get("max")).unwrap_or(&Value::Null), &json!([0, 20, 10])));
        assert_eq!(made.get("regionStale"), Some(&json!(true)));
        let edited = apply_slots(&driven(made, &[("region.box.max.x", "bx + 10")]), &table(&[("bx", "0")])).next.into_owned();
        assert_eq!(edited.get("regionBox").and_then(|r| r.get("box")).and_then(|b| b.get("max")).and_then(|m| m.get(0)).and_then(Value::as_f64), Some(10.0));
        // Points given as a list have no box to drive.
        let listed = feature(json!({ "id": 5, "kind": "surface", "name": "Listed", "region": [0, 7, 14] }));
        assert!(json_slots(&listed, None).is_empty());
        assert!(no_slots_why(None, &listed).contains("not read in a box"));
    }

    #[test]
    fn every_kind_says_a_kind_without_numbers_has_none() {
        let both = feature(json!({ "id": 6, "kind": "boolean", "name": "Both", "op": "join", "body": 2, "tool": 3 }));
        assert!(json_slots(&both, None).is_empty());
        assert_eq!(no_slots_why(None, &both), "");
        // An expression kept for a number it has not is dropped when the features follow.
        let kept = apply_slots(&driven(both, &[("offset", "z")]), &NoNames).next.into_owned();
        assert_eq!(kept.exprs, None);
    }

    // ---- reParameterStore: the features following the parameter list ------------------------

    fn edit(list: &[Parameter], table: &ParameterTable, id: u32, patch: ParameterPatch) -> ParameterEdit {
        match edit_parameter(list, table, id, &patch) {
            Ok(e) => e,
            Err(why) => panic!("{patch:?}: {why}"),
        }
    }
    fn follow(features: &[JsonFeature], e: &ParameterEdit) -> ParametersApplied<JsonFeature> {
        let renamed = e.renamed.as_ref().map(|(a, b)| (a.as_str(), b.as_str()));
        follow_parameters(features, &e.table, renamed, &mut keep_keys, &mut no_inset)
    }
    fn height_list(expression: &str) -> (Vec<Parameter>, ParameterTable) {
        let list = vec![new_parameter(&[], 1, ParameterKind::Length)];
        let t = evaluate_parameters(&list);
        let e = edit(&list, &t, 1, ParameterPatch { name: Some("height".into()), expression: Some(expression.into()), ..Default::default() });
        (e.parameters, e.table)
    }

    #[test]
    fn a_feature_reading_a_parameter_follows_it_and_asks_for_a_rebuild() {
        let (list, t) = height_list("20");
        let features = vec![
            sketch_feature(1, square(10.0).0),
            JsonFeature { exprs: exprs(&[("extent.to", "height + 5")]), ..feature(json!({ "id": 2, "kind": "extrude", "name": "Extrude 1", "sketchId": 1, "regions": ["r:1"], "extent": { "kind": "distance", "from": 0, "to": 25 }, "op": "new", "visible": true })) },
        ];
        let e = edit(&list, &t, 1, ParameterPatch { expression: Some("30".into()), ..Default::default() });
        let r = follow(&features, &e);
        assert!(r.changed);
        assert!(json_eq(r.features[1].get("extent").unwrap_or(&Value::Null), &json!({ "kind": "distance", "from": 0, "to": 35 })));
        // A change that moves no number asks for no rebuild.
        let quiet = edit(&e.parameters, &e.table, 1, ParameterPatch { comment: Some("the boss".into()), ..Default::default() });
        assert!(!follow(&r.features, &quiet).changed);
    }

    #[test]
    fn a_feature_reading_a_parameter_is_named_when_its_parameter_is_to_be_removed_and_follows_a_rename() {
        let (list, t) = height_list("20");
        let mut features = vec![sketch_feature(1, square(10.0).0)];
        // The extrude being set up: a draft has no kind of its own.
        let as_extrude = |y: &JsonFeature| json_slots(y, Some("extrude"));
        let draft = driven(feature(json!({ "sketchId": 1, "regions": [], "extent": { "kind": "distance", "from": 0, "to": 10 } })), &[("extent.to", "height * 2")]);
        let draft = apply_slots_with(&draft, &t, &as_extrude).next.into_owned();
        assert_eq!(num(&draft, &["extent", "to"]), 40.0);
        let drafts: Vec<&str> = if feature_reads(&draft, "height") { vec![READ_BY_FEATURE_DRAFT] } else { Vec::new() };
        assert!(removal_problem(&list, &features, 1, &drafts).is_some_and(|e| e.contains("the feature being set up")));
        let h = edit(&list, &t, 1, ParameterPatch { name: Some("h".into()), ..Default::default() });
        assert_eq!(h.renamed, Some(("height".to_string(), "h".to_string())));
        let draft = apply_slots_with(rename_in_feature(&draft, "height", "h").as_ref(), &h.table, &as_extrude).next.into_owned();
        assert_eq!(draft.exprs, exprs(&[("extent.to", "h * 2")]));
        // Committed, it is a feature of the list, and named by its name.
        features.push(draft.with("kind", json!("extrude")).with("id", json!(2)).with("name", json!("Extrude 1")));
        assert!(removal_problem(&h.parameters, &features, 1, &[]).is_some_and(|e| e.contains("Extrude 1")));
        let tall = edit(&h.parameters, &h.table, 1, ParameterPatch { name: Some("tall".into()), ..Default::default() });
        let r = follow(&features, &tall);
        assert_eq!(r.features[1].exprs, exprs(&[("extent.to", "tall * 2")]));
        assert_eq!(num(&r.features[1], &["extent", "to"]), 40.0);
    }

    #[test]
    fn a_sketch_dimension_reading_a_parameter_is_solved_again_the_features_on_it_keeping_their_regions() {
        let list = vec![new_parameter(&[], 1, ParameterKind::Length)];
        let t0 = evaluate_parameters(&list);
        let w = edit(&list, &t0, 1, ParameterPatch { name: Some("width".into()), expression: Some("10".into()), ..Default::default() });
        let (sketch, dim) = square(10.0);
        let features = vec![sketch_feature(1, sketch), feature(json!({ "id": 2, "kind": "extrude", "name": "Extrude 1", "sketchId": 1, "regions": ["r:1"], "extent": { "kind": "distance", "from": 0, "to": 10 }, "op": "new", "visible": true }))];
        assert_eq!(features[0].sketch.as_ref().and_then(|s| s.constraint(dim)).and_then(|c| c.expr.as_deref()), Some("width"));
        let e = edit(&w.parameters, &w.table, 1, ParameterPatch { expression: Some("30".into()), ..Default::default() });
        let r = follow(&features, &e);
        assert!(r.changed);
        let solved = r.features[0].sketch.as_ref().map_or(f64::NAN, bottom_length);
        assert!((solved - 30.0).abs() < 0.5e-6, "{solved}");
        assert_eq!(r.features[0].sketch.as_ref().and_then(|s| dim_value(s, dim)), Some(30.0));
        assert_eq!(r.features[1].regions().len(), 1);
        assert!(removal_problem(&e.parameters, &r.features, 1, &[]).is_some_and(|e| e.contains("Sketch 1")));
    }

    // ---- not in ScanRuler's tests: the JSON form, and Polymix's model -----------------------

    #[test]
    fn a_json_feature_reads_back_as_written() {
        let x = with_slot(&extrude(), "extent.to", 25.0, Some("height + 5"));
        let text = serde_json::to_string(&x).unwrap_or_default();
        assert!(text.contains(r#""exprs":{"extent.to":"height + 5"}"#), "{text}");
        assert!(text.contains(r#""to":25"#), "{text}");
        let back: JsonFeature = serde_json::from_str(&text).unwrap_or_default();
        assert_eq!(back, x);
        assert_eq!(back.with("draftDeg", json!(0.0)), x);
        assert_eq!(json_number(-0.0), json!(0));
        assert_eq!(json_number(2.5), json!(2.5));
        assert_eq!(json_number(f64::NAN), Value::Null);
        let pick = RegionPick { given: PickGiven::Picks(vec![PickItem::Key("r:1".into()), PickItem::At { at: DVec2::new(1.0, 2.0) }]), keys: vec!["r:1".into()] };
        let text = serde_json::to_string(&pick).unwrap_or_default();
        assert_eq!(text, r#"{"given":["r:1",{"at":[1.0,2.0]}],"keys":["r:1"]}"#);
        assert_eq!(serde_json::from_str::<RegionPick>(&text).ok(), Some(pick));
        assert_eq!(serde_json::to_string(&PickGiven::Largest).unwrap_or_default(), r#""largest""#);
    }

    #[test]
    fn the_more_kinds_read_and_write_their_numbers() {
        let chamfer = feature(json!({ "kind": "chamfer", "distance": 1, "mode": "distanceAngle", "angleDeg": 45 }));
        assert_eq!(paths(&chamfer), ["distance", "angleDeg"]);
        assert_eq!(num(&slot(&chamfer, "angleDeg").set(&chamfer, 95.0), &["angleDeg"]), 89.9);
        assert_eq!(paths(&chamfer.with("mode", json!("twoDistances"))), ["distance", "distance2"]);
        let thread = feature(json!({ "kind": "thread", "diameter": 10, "pitch": 1.5, "fullLength": true, "offset": 0 }));
        assert_eq!(paths(&thread), ["diameter", "pitch", "offset"]);
        assert_eq!(paths(&thread.with("fullLength", json!(false))), ["diameter", "pitch", "length", "start", "offset"]);
        let mut short = thread.clone();
        short.fields.remove("fullLength");
        assert_eq!(paths(&short.with("length", json!(8))), ["diameter", "pitch", "length", "start", "offset"]);
        let moved = feature(json!({ "kind": "moveBody", "translate": [25.4, 0, 0], "rotate": [0, 0, 0] }));
        assert_eq!(paths(&moved), ["translate.x", "translate.y", "translate.z", "rotate.x", "rotate.y", "rotate.z"]);
        assert_eq!(slot(&moved, "rotate.z").label, "Turn Z");
        assert!(json_eq(slot(&moved, "rotate.z").set(&moved, 90.0).get("rotate").unwrap_or(&Value::Null), &json!([0, 0, 90])));
        let sweep = feature(json!({ "kind": "sweep", "route": { "bendRadius": 4 } }));
        assert_eq!(num(&slot(&sweep, "route.bendRadius").set(&sweep, -6.0), &["route", "bendRadius"]), 6.0);
        assert!(json_slots(&sweep.with("route", Value::Null), None).is_empty());
        let linear = feature(json!({ "kind": "pattern", "mode": "linear", "axis2": { "kind": "world", "axis": "y" }, "count": 3, "step": 5, "count2": 2, "step2": 4 }));
        assert_eq!(paths(&linear), ["count", "step", "count2", "step2"]);
        assert_eq!(num(&slot(&linear, "count2").set(&linear, 0.2), &["count2"]), 1.0);
        for (kind, path) in [("extract", "thickness"), ("thicken", "thickness"), ("draft", "angleDeg"), ("moveFace", "offset"), ("offsetBody", "distance"), ("trim", "tolerance")] {
            assert!(paths(&feature(json!({ "kind": kind }))).contains(&path.to_string()), "{kind}");
        }
        assert_eq!(paths(&feature(json!({ "kind": "sheet", "from": 0, "to": 2 }))), ["from", "to"]);
        // A draft of an extrude has no kind of its own, and is asked with one.
        let draft = feature(json!({ "extent": { "kind": "symmetric", "total": 6 } }));
        assert!(json_slots(&draft, None).is_empty());
        assert_eq!(json_slots(&draft, Some("extrude")).into_iter().map(|s| s.path).collect::<Vec<_>>(), ["extent.total", "taper"]);
    }

    /// Not in ScanRuler's tests: what its code does where the port first read it otherwise. An
    /// extent of no kind among ScanRuler's three has a start (`extent.kind !== 'betweenPlanes'`).
    #[test]
    fn an_extent_of_no_known_kind_has_a_start() {
        let x = extrude();
        assert_eq!(paths(&x.with("extent", json!({ "from": 3 }))), ["extent.from", "taper"]);
        assert_eq!(paths(&x.with("extent", json!({ "kind": "throughAll" }))), ["extent.from", "taper"]);
        assert_eq!(slot(&x.with("extent", json!({ "from": 3 })), "extent.from").get(&x.with("extent", json!({ "from": 3 }))), 3.0);
        let mut bare = x.clone();
        bare.fields.remove("extent");
        assert_eq!(paths(&bare), ["taper"]);
    }

    /// The regions of a sketch where the old "r:1" is now "r:2".
    struct Renamed;
    impl RegionMatch for Renamed {
        fn key(&self, _region: usize) -> &str {
            "r:2"
        }
        fn match_region(&self, key: &str) -> Option<usize> {
            (key == "r:1").then_some(0)
        }
        fn regions_by_pick(&self, _given: &PickGiven) -> Option<Vec<usize>> {
            Some(vec![0])
        }
        fn was_region(&self, _key: &str) -> bool {
            false
        }
        fn region_by_place(&self, _was_key: &str, _taken: &[usize]) -> Option<usize> {
            None
        }
        fn split_parts(&self, _was_key: &str) -> Vec<usize> {
            Vec::new()
        }
    }

    /// ScanRuler's `rekeyed` writes a pick over or keeps it; it never takes one away — not even
    /// one written in a form this port does not read.
    #[test]
    fn rekeying_keeps_a_pick_it_cannot_read_and_writes_one_it_can() {
        let unread = json!({ "given": "everything", "keys": ["r:1"] });
        let f = extrude().with("regionPick", unread.clone());
        let r = rekeyed(&f, &Renamed).into_owned();
        assert_eq!(r.regions(), ["r:2"]);
        assert_eq!(r.get("regionPick"), Some(&unread));
        let largest = extrude().with("regionPick", json!({ "given": "largest", "keys": ["r:1"] })).with("regionsByPlace", json!(["r:1"]));
        let r = rekeyed(&largest, &Renamed).into_owned();
        assert_eq!(r.region_pick(), Some(RegionPick { given: PickGiven::Largest, keys: vec!["r:2".into()] }));
        assert_eq!(r.get("regionsByPlace"), None);
    }

    /// A `replace_sketch` that hands back a shorter list ends the walk where the list ends, as
    /// ScanRuler's loop reads the list's length on every step — it does not run past it.
    #[test]
    fn applying_the_parameters_walks_the_list_as_it_stands() {
        let (sketch, _) = square(10.0);
        let features = vec![sketch_feature(1, sketch), driven(extrude(), &[("extent.to", "width / 2")])];
        let mut keep_first = |fs: &[JsonFeature], id: FeatureId, s: Sketch| {
            let mut out = replace_sketch(fs, id, &s, &mut |_| None);
            out.truncate(1);
            out
        };
        let r = apply_parameters(&features, &table(&[("width", "30")]), &mut keep_first, &mut no_inset);
        assert!(r.changed);
        assert_eq!(r.features.len(), 1);
    }

    /// The sketch an inset was moved in from is the first sketch of that id, as ScanRuler's
    /// `find` asks for the id and the kind together.
    #[test]
    fn an_inset_is_moved_in_again_from_the_sketch_of_its_id() {
        let mut rect = Sketch::default();
        rect.add_rectangle(DVec2::splat(-10.0), DVec2::splat(10.0));
        let other = feature(json!({ "id": 1, "kind": "fillet", "name": "Fillet 1", "radius": 1 }));
        let foot = sketch_feature(1, rect);
        let inner = sketch_feature(5, Sketch::default()).with("inset", json!({ "of": 1, "by": 2, "keep": [] }));
        let chains = |of: &Sketch, inset: &SketchInset, _: &[DatumTrace], name: &str| -> Result<Sketch, String> {
            let (lo, hi) = of.bounds().ok_or("an empty profile")?;
            assert_eq!(name, "Sketch 1");
            let mut s = Sketch::default();
            s.add_rectangle(lo + DVec2::splat(inset.by), hi - DVec2::splat(inset.by));
            Ok(s)
        };
        let moved = inset_again(&inner, &[other, foot, inner.clone()], &chains);
        let width = moved.as_ref().ok().and_then(Sketch::bounds).map_or(f64::NAN, |(lo, hi)| hi.x - lo.x);
        assert!((width - 16.0).abs() < 1e-6, "{moved:?}");
    }

    fn model(id: u32, name: &str, kind: FeatureKind) -> Feature {
        Feature { id, name: name.to_string(), flags: FeatureFlags::default(), kind }
    }

    fn with_exprs_of(f: Feature, e: &[(&str, &str)]) -> Feature {
        Slotted::with_exprs(&f, exprs(e))
    }

    fn pad(extent: ExtrudeExtent, op: BodyOp) -> FeatureKind {
        FeatureKind::Extrude(ExtrudeFeature::new(1, vec!["r:1".into()], extent, op))
    }

    #[test]
    fn polymix_model_features_offer_the_same_slots() {
        let paths = |k: FeatureKind| model_slots(&model(9, "F", k)).into_iter().map(|s| s.path).collect::<Vec<_>>();
        let sketch = model(1, "Sketch 1", FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, Sketch::default())));
        assert_eq!(model_slots(&sketch).into_iter().map(|s| s.path).collect::<Vec<_>>(), ["offset"]);
        assert_eq!(model_slots(&sketch)[0].label, "Z =");
        assert_eq!(paths(pad(ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)), ["extent.from", "extent.to", "taper"]);
        let symmetric = model(2, "Extrude 1", pad(ExtrudeExtent::Symmetric { total: 8.0 }, BodyOp::New));
        let total = model_slots(&symmetric).into_iter().find(|s| s.path == "extent.total").expect("a total");
        assert!(matches!(&total.set(&symmetric, -6.0).kind, FeatureKind::Extrude(ExtrudeFeature { extent: ExtrudeExtent::Symmetric { total }, .. }) if *total == 6.0));
        let revolve = model(3, "Revolve 1", FeatureKind::Revolve(RevolveFeature::new(1, vec![], AxisRef::World { axis: WorldAxis::Z }, 180.0, BodyOp::New)));
        assert!(matches!(&model_slots(&revolve)[0].set(&revolve, 720.0).kind, FeatureKind::Revolve(RevolveFeature { angle_deg, .. }) if *angle_deg == 360.0));
        let fillet = FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: vec![], edges_where: vec![], radius: 1.0, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() });
        assert_eq!(paths(fillet), ["radius"]);
        let chamfer = FeatureKind::Chamfer(ChamferFeature { body: BodyId::new(2), edges: vec![], edges_where: vec![], distance: 1.0, chain: false, skip_failing: false, mode: None, distance2: None, angle_deg: None, flip: false, exprs: Exprs::new() });
        assert_eq!(paths(chamfer), ["distance"]);
        let shell = model(4, "Shell 1", FeatureKind::Shell(ShellFeature { body: BodyId::new(2), faces: vec![], faces_where: vec![], thickness: 2.0, exprs: Exprs::new() }));
        assert!(matches!(&model_slots(&shell)[0].set(&shell, -3.0).kind, FeatureKind::Shell(ShellFeature { thickness, .. }) if *thickness == 3.0));
        let mirror = model(5, "Mirror 1", FeatureKind::Mirror(MirrorFeature { body: BodyId::new(2), plane: MirrorPlane::Plane(PlaneRef::world(WorldAxis::X, 0.0)), join: false, exprs: Exprs::new() }));
        assert_eq!(model_slots(&mirror).into_iter().map(|s| s.path).collect::<Vec<_>>(), ["plane.offset"]);
        // A mirror's plane takes any finite offset (`planeAtOffset`); a sketch's moves only by
        // 1e-12 or more (`withPlaneOffset`).
        assert!(matches!(&model_slots(&mirror)[0].set(&mirror, 1e-13).kind, FeatureKind::Mirror(MirrorFeature { plane: MirrorPlane::Plane(PlaneRef::World { offset, .. }), .. }) if *offset == 1e-13));
        assert!(matches!(&model_slots(&sketch)[0].set(&sketch, 1e-13).kind, FeatureKind::Sketch(SketchFeature { plane: PlaneRef::World { offset, .. }, .. }) if *offset == 0.0));
        assert!(paths(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Join, body: BodyId::new(2), tool: BodyId::new(3), keep: false })).is_empty());
    }

    #[test]
    fn polymix_model_features_follow_the_parameters() {
        let (sketch, dim) = square(10.0);
        let features = vec![
            model(1, "Sketch 1", FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, sketch))),
            with_exprs_of(model(2, "Extrude 1", pad(ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)), &[("extent.to", "width / 2"), ("taper", "lean")]),
            with_exprs_of(model(3, "Mirror 1", FeatureKind::Mirror(MirrorFeature { body: BodyId::new(2), plane: MirrorPlane::Plane(PlaneRef::world(WorldAxis::X, 0.0)), join: false, exprs: Exprs::new() })), &[("plane.offset", "width")]),
        ];
        let mut keep = |fs: &[Feature], id: FeatureId, s: Sketch| replace_sketch(fs, id, &s, &mut |_| None);
        let mut none = |_: &Feature, _: &[Feature]| -> Result<Sketch, String> { Err("no insets here".to_string()) };
        let list = vec![param(1, "width", "30"), Parameter { kind: ParameterKind::Angle, ..param(2, "lean", "3") }];
        let r = apply_parameters(&features, &evaluate_parameters(&list), &mut keep, &mut none);
        assert!(r.changed && r.errors.is_empty(), "{:?}", r.errors);
        let solved = r.features[0].sketch().map_or(f64::NAN, bottom_length);
        assert!((solved - 30.0).abs() < 0.5e-6, "{solved}");
        assert_eq!(r.features[0].sketch().and_then(|s| dim_value(s, dim)), Some(30.0));
        // The box's taper is ScanRuler's draftDeg with its sign turned.
        assert!(matches!(&r.features[1].kind, FeatureKind::Extrude(ExtrudeFeature { extent: ExtrudeExtent::Distance { to, .. }, draft_deg, .. }) if *to == 15.0 && *draft_deg == -3.0), "{:?}", r.features[1].kind);
        assert!(matches!(&r.features[2].kind, FeatureKind::Mirror(MirrorFeature { plane: MirrorPlane::Plane(PlaneRef::World { offset, .. }), .. }) if *offset == 30.0));
        assert!(!apply_parameters(&r.features, &evaluate_parameters(&list), &mut keep, &mut none).changed);
        assert_eq!(users_of("width", &list, &r.features).len(), 3);
    }
}
