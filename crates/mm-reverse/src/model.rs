// SPDX-License-Identifier: AGPL-3.0-only
//! The features of the Reverse Engineer workspace, as they are stored. Ported from ScanRuler
//! (`plugins/reverse/core/features.ts`, with the part's own lists from `parameters.ts`,
//! `bodyFolders.ts`, `notes.ts`, `thread.ts`, `sweepRoute.ts`, `autoConstrain.ts`,
//! `sketchImage.ts`, `kernelTypes.ts` and `src/core/types.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.
//!
//! A feature list is linear: a sketch on a plane, then extrusions and revolutions that each
//! pick the sketch's regions to use — the way a CAD extrude asks for its profiles — and make a
//! new body or join or cut one made before, then the features that work on a body: a shell, a
//! round, a hole, a cut at a plane, a mirror, a pattern. Features refer to the part's other
//! things by id — a measured element, a section plane, a coordinate plane, a line — and are
//! resolved against them at build time into plain planes and axes. A face or an edge a feature
//! works on is stored as its fingerprint (see `topo.rs`), found again on the body as it stands
//! before the feature, never as an index into a list the next rebuild remakes.
//!
//! ScanRuler's features refer to the elements and sections of its 3D Measure workspace, which
//! Polymix does not have: the part keeps a list of its own elements here (planes, lines,
//! circles, cylinders, cones, spheres, tori, points and section planes), each a geometric
//! description with an id and a name, so that the references keep ScanRuler's shape.
//!
//! The part saves as JSON in ScanRuler's shape — every feature an object with its `kind` —
//! and its bodies are never saved: they are rebuilt.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use glam::{DVec2, DVec3};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::frame::{SectionFrame, WorldAxis};
use crate::pick_query::StoredPick;
use crate::sketch::{EntityId, Sketch};
use crate::topo::{EdgeFingerprint, FaceFingerprint, FaceKind};

/// A feature's id: counts from one, never reused within a part.
pub type FeatureId = u32;

/// The id of one of the part's elements (see [`Element`]).
pub type ElementId = u32;

/// A face of a body as a feature stores its pick: a fingerprint found again on the body as it
/// stands before the feature (see `topo.rs`).
pub type FaceRef = FaceFingerprint;

/// An edge of a body as a feature stores its pick.
pub type EdgeRef = EdgeFingerprint;

/// The expressions driving a feature's numbers, by the field as its box shows it — see
/// ScanRuler's `parameters.ts`; the numbers hold what they last came to.
/// The expressions a feature keeps beside its numbers, by slot path, in the order they were set
/// (ScanRuler evaluates them in that order).
pub type Exprs = crate::parameters::ExprMap;

fn is_false(b: &bool) -> bool {
    !*b
}

fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

fn yes() -> bool {
    true
}

// ---- PICKLIST ----
fn is_true(b: &bool) -> bool {
    *b
}

// ---- bodies ----

/// How many further bodies a step can leave: a body is named by the feature that made it and,
/// for every further body that feature's step left — a solid in pieces, two profiles apart, a
/// cut through a body's waist, a pattern of separate copies — by the feature's id and a
/// fraction, n / ISLAND_IDS. One body is one solid.
pub const ISLAND_IDS: u32 = 1024;

/// A body's id: the feature that made it and which of the bodies that feature's step left it
/// is (0 for the first). ScanRuler keeps it as one number, `feature + island / 1024`, and so
/// does the part's JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct BodyId {
    pub feature: FeatureId,
    pub island: u32,
}

impl BodyId {
    /// The first body a step leaves: the step's own id.
    pub const fn new(feature: FeatureId) -> BodyId {
        BodyId { feature, island: 0 }
    }

    /// The feature that made the body (ScanRuler's `bodyMaker`).
    pub fn maker(self) -> FeatureId {
        self.feature
    }

    /// The id as ScanRuler writes it: the feature and a fraction.
    pub fn as_number(self) -> f64 {
        self.feature as f64 + self.island as f64 / ISLAND_IDS as f64
    }

    /// The id a number written by ScanRuler stands for; `None` for one that is no body's.
    pub fn from_number(x: f64) -> Option<BodyId> {
        if !x.is_finite() || x < 0.0 || x > u32::MAX as f64 {
            return None;
        }
        let feature = x.floor();
        let island = ((x - feature) * ISLAND_IDS as f64).round() as u32;
        if island >= ISLAND_IDS {
            return Some(BodyId::new(feature as u32 + 1));
        }
        Some(BodyId { feature: feature as u32, island })
    }
}

/// The id of the n-th further body a step leaves (n ≥ 1); the first is the step's own id.
pub fn island_id(step: FeatureId, n: u32) -> BodyId {
    BodyId { feature: step, island: n }
}

/// The feature that made a body, from the body's id.
pub fn body_maker(id: BodyId) -> FeatureId {
    id.feature
}

/// Which of the bodies its step left a body is: 0 for the first.
pub fn island_of(id: BodyId) -> u32 {
    id.island
}

impl fmt::Display for BodyId {
    /// "Body 3", or "Body 3.0009765625" for the second body step 3 left — the name ScanRuler's
    /// body list falls back on.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.island == 0 {
            write!(f, "Body {}", self.feature)
        } else {
            write!(f, "Body {}", self.as_number())
        }
    }
}

impl Serialize for BodyId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.island == 0 {
            s.serialize_u32(self.feature)
        } else {
            s.serialize_f64(self.as_number())
        }
    }
}

impl<'de> Deserialize<'de> for BodyId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<BodyId, D::Error> {
        struct V;
        impl Visitor<'_> for V {
            type Value = BodyId;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a body id: a feature's id, with a fraction for a further body of its step")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<BodyId, E> {
                u32::try_from(v).map(BodyId::new).map_err(|_| E::custom(format!("{v} is no body id")))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<BodyId, E> {
                u32::try_from(v).map(BodyId::new).map_err(|_| E::custom(format!("{v} is no body id")))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<BodyId, E> {
                BodyId::from_number(v).ok_or_else(|| E::custom(format!("{v} is no body id")))
            }
        }
        d.deserialize_any(V)
    }
}

/// A map keyed by body, written as ScanRuler writes a `Record<number, T>`: the body's id as the
/// key's text.
mod body_keyed {
    use super::*;

    pub fn serialize<S: Serializer, T: Serialize>(map: &BTreeMap<BodyId, T>, s: S) -> Result<S::Ok, S::Error> {
        let text: BTreeMap<String, &T> = map.iter().map(|(k, v)| (if k.island == 0 { k.feature.to_string() } else { k.as_number().to_string() }, v)).collect();
        text.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<BTreeMap<BodyId, T>, D::Error> {
        let text: BTreeMap<String, T> = BTreeMap::deserialize(d)?;
        text.into_iter()
            .map(|(k, v)| {
                let id = k.parse::<f64>().ok().and_then(BodyId::from_number).ok_or_else(|| de::Error::custom(format!("{k} is no body id")))?;
                Ok((id, v))
            })
            .collect()
    }
}

// ---- planes and axes ----

/// How far inside a face the scan is cut for a sketch on it: on the face itself the plane
/// would pass through the face's own triangles and come back as a tangle; a hair inside, the
/// cut crosses the walls that meet the face and gives the outline of the part at that face.
pub const FACE_SLICE_INSET_MM: f64 = 0.25;

/// What a radial plane turns about: a line, cylinder, cone or circle element's axis, or a
/// coordinate axis through the origin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum RadialAxis {
    Element { id: ElementId },
    World { axis: WorldAxis },
}

impl From<RadialAxis> for AxisRef {
    fn from(a: RadialAxis) -> AxisRef {
        match a {
            RadialAxis::Element { id } => AxisRef::Element { id },
            RadialAxis::World { axis } => AxisRef::World { axis },
        }
    }
}

/// Where a sketch lies: a section's plane, a plane element, or a coordinate plane at an
/// offset along its axis. A section's plane, a plane element's and a body's face take an
/// `offset` too: millimetres the plane is moved along its normal. `up` is the coordinate axis
/// a sketch's v runs up, as near as the plane lets it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PlaneRef {
    Section {
        id: ElementId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        up: Option<WorldAxis>,
    },
    Plane {
        id: ElementId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        up: Option<WorldAxis>,
    },
    /// A coordinate plane: the plane square to `axis`, `offset` along it.
    World {
        axis: WorldAxis,
        offset: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        up: Option<WorldAxis>,
    },
    /// A plane through an axis, turned `angle_deg` about it: u out along the radius, v up the
    /// axis. The sheet's zero is on the axis, level with the part's origin.
    #[serde(rename_all = "camelCase")]
    Radial {
        axis: RadialAxis,
        angle_deg: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<f64>,
    },
    /// A plane fitted through a face of the scan that was clicked to sketch on, belonging to no
    /// element; the frame is the plane. `label` names what the plane is where it is not a
    /// clicked face (the scan's symmetry plane).
    Fitted {
        frame: SectionFrame,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    /// A flat face of a body, as a CAD sketch on a face: the face by its fingerprint, found
    /// again on the body as it stands when a feature on the sketch is built. `frame` is the
    /// plane as last found, moved out by the offset.
    #[serde(rename_all = "camelCase")]
    BodyFace {
        body: BodyId,
        face: FaceRef,
        frame: SectionFrame,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<f64>,
    },
}

impl PlaneRef {
    /// A coordinate plane: the plane square to `axis`, `offset` along it.
    pub fn world(axis: WorldAxis, offset: f64) -> PlaneRef {
        PlaneRef::World { axis, offset, up: None }
    }

    /// The frame the plane stands for where nothing of the part's is needed to say it: a
    /// coordinate plane's (the sketch rule's zero and X on it), a fitted plane's, a face's as
    /// last found. `None` for a plane that names an element or turns about an axis.
    pub fn own_frame(&self) -> Option<SectionFrame> {
        match self {
            PlaneRef::World { axis, offset, .. } => {
                let dir = axis.dir();
                SectionFrame::sketch_on(dir * if offset.is_finite() { *offset } else { 0.0 }, dir, None)
            }
            PlaneRef::Fitted { frame, .. } | PlaneRef::BodyFace { frame, .. } => Some(*frame),
            _ => None,
        }
    }
}

/// What a revolve turns about: a line or cylinder element's axis, a line of the sketch itself,
/// or a coordinate axis through the origin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum AxisRef {
    Element { id: ElementId },
    SketchLine { entity: EntityId },
    World { axis: WorldAxis },
}

/// A plane moved to an offset along its normal — a face's last-found frame with it. One that
/// takes no offset comes back as it was.
pub fn plane_at_offset(r: &PlaneRef, offset: f64) -> PlaneRef {
    if !plane_takes_offset(r) || !offset.is_finite() {
        return r.clone();
    }
    let delta = offset - plane_offset(r);
    let mut out = r.clone();
    match &mut out {
        PlaneRef::Section { offset: o, .. } | PlaneRef::Plane { offset: o, .. } | PlaneRef::Radial { offset: o, .. } => *o = Some(offset),
        PlaneRef::World { offset: o, .. } => *o = offset,
        PlaneRef::BodyFace { offset: o, frame, .. } => {
            *o = Some(offset);
            frame.origin += frame.normal * delta;
        }
        PlaneRef::Fitted { .. } => {}
    }
    out
}

/// How far a plane is moved along its normal: a coordinate plane's coordinate, the offset of
/// any other — none for a plane fitted where the scan was clicked, which takes no offset.
pub fn plane_offset(r: &PlaneRef) -> f64 {
    match r {
        PlaneRef::Fitted { .. } => 0.0,
        PlaneRef::World { offset, .. } => *offset,
        PlaneRef::Section { offset, .. } | PlaneRef::Plane { offset, .. } | PlaneRef::Radial { offset, .. } | PlaneRef::BodyFace { offset, .. } => offset.unwrap_or(0.0),
    }
}

/// Whether a plane takes an offset.
pub fn plane_takes_offset(r: &PlaneRef) -> bool {
    !matches!(r, PlaneRef::Fitted { .. })
}

/// A sketch's plane and frame moved to an offset of its plane: the plane's offset set and the
/// frame carried along its normal by the change, a face's last-found frame with it. Its pieces
/// keep their sheet coordinates.
pub fn with_plane_offset(plane: &PlaneRef, frame: &SectionFrame, offset: f64) -> (PlaneRef, SectionFrame) {
    if !plane_takes_offset(plane) || !offset.is_finite() {
        return (plane.clone(), *frame);
    }
    let delta = offset - plane_offset(plane);
    if delta.abs() < 1e-12 {
        return (plane.clone(), *frame);
    }
    (plane_at_offset(plane, offset), frame.offset(delta))
}

/// What a solid feature does with what it makes. Join makes one with every solid body it
/// meets — a new body where it meets none, and one body of all of them where it bridges
/// several; Cut takes it out of every solid body it meets; New body keeps it a body of its
/// own, whatever it overlaps. A feature saved with a `target` joins or cuts that body alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BodyOp {
    New,
    Join,
    Cut,
}

impl BodyOp {
    pub const ALL: [BodyOp; 3] = [BodyOp::Join, BodyOp::Cut, BodyOp::New];

    pub fn name(self) -> &'static str {
        match self {
            BodyOp::Join => "Join",
            BodyOp::Cut => "Cut",
            BodyOp::New => "New body",
        }
    }
}

/// What a Combine does with its two bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BooleanOp {
    Join,
    Cut,
    Intersect,
}

impl BooleanOp {
    pub const ALL: [BooleanOp; 3] = [BooleanOp::Join, BooleanOp::Cut, BooleanOp::Intersect];

    pub fn name(self) -> &'static str {
        match self {
            BooleanOp::Join => "Join",
            BooleanOp::Cut => "Cut",
            BooleanOp::Intersect => "Intersect",
        }
    }
}

// ---- sketches ----

/// A picture laid under a sketch to trace — shown while the sketch is drawn, nothing to the
/// build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SketchImage {
    /// The picture, as a data URL.
    pub src: String,
    pub width: u32,
    pub height: u32,
    /// Where its bottom-left corner lies on the sheet, mm.
    pub origin: DVec2,
    pub mm_per_px: f64,
    /// Turned counter-clockwise about its bottom-left corner, degrees.
    pub angle_deg: f64,
    /// How much of it shows, 0–1.
    pub opacity: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Where a sketch's inset profile came from: the sketch, the wall, and the pieces kept where
/// they lie.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SketchInset {
    pub of: FeatureId,
    pub by: f64,
    pub keep: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SketchFeature {
    pub plane: PlaneRef,
    /// The plane frozen when the sketch was made — as a section's is — so a re-fitted plane
    /// element does not move a sketch under its regions.
    pub frame: SectionFrame,
    pub sketch: Sketch,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<SketchImage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inset: Option<SketchInset>,
}

impl SketchFeature {
    /// A sketch on a coordinate plane, its frame the plane's.
    pub fn on_world(axis: WorldAxis, offset: f64, sketch: Sketch) -> SketchFeature {
        let plane = PlaneRef::world(axis, offset);
        let frame = plane.own_frame().unwrap_or(SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y });
        SketchFeature { plane, frame, sketch, exprs: Exprs::new(), image: None, inset: None }
    }

    /// The sketch moved to an offset of its plane (see [`with_plane_offset`]).
    pub fn at_offset(&self, offset: f64) -> SketchFeature {
        let (plane, frame) = with_plane_offset(&self.plane, &self.frame, offset);
        SketchFeature { plane, frame, ..self.clone() }
    }
}

// ---- extrude and revolve ----

/// One region asked for by more than its key: its key, or a point inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RegionAsk {
    Key(String),
    At { at: DVec2 },
}

/// Every region, or the largest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegionWord {
    All,
    Largest,
}

/// How an extrude's or a revolve's regions were asked for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RegionGiven {
    Word(RegionWord),
    List(Vec<RegionAsk>),
}

/// How an extrude's or a revolve's regions were asked for, where by more than their keys:
/// every region, the largest, or keys and points inside regions — taken again from the sketch
/// as it stands whenever it is edited. `keys` are the regions it came to: while they are still
/// the feature's regions it stands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionPick {
    pub given: RegionGiven,
    pub keys: Vec<String>,
}

/// A flat face of a body as an extrude's profile: the face itself, by its fingerprint, taken
/// as it stands when the extrude runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceProfile {
    pub body: BodyId,
    pub face: FaceRef,
}

/// How far an extrusion reaches, along the sketch plane's normal: offsets from the plane, a
/// plane to stop at, planes on both sides, a symmetric span about the sketch, or an end found
/// on the part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ExtrudeExtent {
    Distance { from: f64, to: f64 },
    ToPlane { from: f64, plane: PlaneRef },
    BetweenPlanes { a: PlaneRef, b: PlaneRef },
    Symmetric { total: f64 },
    /// Up to a face of a body, found again on the body as it stands when the extrude runs.
    ToFace { from: f64, body: BodyId, face: FaceRef },
    /// Up to the surface of a body, the way the body lies from the sketch — against the normal
    /// with `reverse` where it lies both ways.
    ToBody {
        from: f64,
        body: BodyId,
        #[serde(default, skip_serializing_if = "is_false")]
        reverse: bool,
    },
    /// Until the profile first meets the body it joins or cuts (or any body, for a new one).
    ToNext {
        from: f64,
        #[serde(default, skip_serializing_if = "is_false")]
        reverse: bool,
    },
    /// To the far end of that body: a boss as long as the part, a cut through all of it.
    ToLast {
        from: f64,
        #[serde(default, skip_serializing_if = "is_false")]
        reverse: bool,
    },
}

impl ExtrudeExtent {
    pub fn name(&self) -> &'static str {
        match self {
            ExtrudeExtent::Distance { .. } => "Distance",
            ExtrudeExtent::ToPlane { .. } => "To plane",
            ExtrudeExtent::BetweenPlanes { .. } => "Between planes",
            ExtrudeExtent::Symmetric { .. } => "Symmetric",
            ExtrudeExtent::ToFace { .. } => "To face",
            ExtrudeExtent::ToBody { .. } => "To body",
            ExtrudeExtent::ToNext { .. } => "To next",
            ExtrudeExtent::ToLast { .. } => "Through all",
        }
    }
}

/// Whether an extent starts at a typed offset and ends at something found — the kinds with a
/// From and no To.
pub fn extent_ends_on_part(e: &ExtrudeExtent) -> bool {
    matches!(e, ExtrudeExtent::ToPlane { .. } | ExtrudeExtent::ToFace { .. } | ExtrudeExtent::ToBody { .. } | ExtrudeExtent::ToNext { .. } | ExtrudeExtent::ToLast { .. })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtrudeFeature {
    /// The sketch the profiles come from — 0, and `regions` empty, for an extrude pulled from
    /// a face of a body (`face`).
    pub sketch_id: FeatureId,
    /// The sketch's regions to extrude, by their keys. An inner loop left unpicked is a hole.
    pub regions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_pick: Option<RegionPick>,
    /// Regions an edit of the sketch found again by where they lie.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions_by_place: Vec<String>,
    /// A face of a body pulled as it stands, in place of a sketch's regions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub face: Option<FaceProfile>,
    pub extent: ExtrudeExtent,
    /// Degrees the sides lean in as the extrusion goes away from the sketch plane — a moulded
    /// boss's draft; 0 for straight sides. The box shows it as a taper the way Fusion does,
    /// the other way round: a positive taper leans the sides out.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub draft_deg: f64,
    /// The way the profile is swept, a unit vector in the part's frame — an oblique prism.
    /// Absent: the sketch plane's normal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<DVec3>,
    pub op: BodyOp,
    /// The one body a join or a cut works on, in a feature saved before a join met every body
    /// it touched. Absent, a join or a cut works on every body it meets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<BodyId>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RevolveFeature {
    pub sketch_id: FeatureId,
    pub regions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_pick: Option<RegionPick>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions_by_place: Vec<String>,
    pub axis: AxisRef,
    /// At most a full turn for a plain revolve; any number of turns for one with a pitch.
    pub angle_deg: f64,
    /// Millimetres the profile climbs along the axis every full turn — a coil or a thread;
    /// absent or 0 for a plain revolve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f64>,
    /// With a pitch: the screw turns clockwise seen down the way it climbs.
    #[serde(default, skip_serializing_if = "is_false")]
    pub left_handed: bool,
    pub op: BodyOp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<BodyId>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

impl ExtrudeFeature {
    /// An extrude of a sketch's regions, straight, with nothing else set.
    pub fn new(sketch_id: FeatureId, regions: Vec<String>, extent: ExtrudeExtent, op: BodyOp) -> ExtrudeFeature {
        ExtrudeFeature { sketch_id, regions, region_pick: None, regions_by_place: Vec::new(), face: None, extent, draft_deg: 0.0, direction: None, op, target: None, exprs: Exprs::new() }
    }
}

impl RevolveFeature {
    /// A revolve of a sketch's regions about an axis, with nothing else set.
    pub fn new(sketch_id: FeatureId, regions: Vec<String>, axis: AxisRef, angle_deg: f64, op: BodyOp) -> RevolveFeature {
        RevolveFeature { sketch_id, regions, region_pick: None, regions_by_place: Vec::new(), axis, angle_deg, pitch: None, left_handed: false, op, target: None, exprs: Exprs::new() }
    }
}

/// The angle a revolve with no pitch turns at most: a whole turn.
pub const FULL_TURN_DEG: f64 = 360.0;

/// The least an extrude's direction leaves its sketch plane by.
pub const MIN_SWEEP_RAD: f64 = std::f64::consts::PI / 180.0;

/// The angle a coil turns at most: a thousand turns.
pub const MAX_COIL_DEG: f64 = 360.0 * 1000.0;

/// The most a revolve may turn, with its pitch or without one.
pub fn revolve_max_deg(pitch: Option<f64>) -> f64 {
    // As ScanRuler's `pitch ? …`: no pitch, a zero one and a NaN one are a plain revolve.
    if pitch.is_some_and(|p| p != 0.0 && !p.is_nan()) {
        MAX_COIL_DEG
    } else {
        FULL_TURN_DEG
    }
}

// ---- features on a body ----

/// Hollow a body to a wall: the faces named are taken away to open it; none named leaves it
/// closed, hollow inside.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellFeature {
    pub body: BodyId,
    pub faces: Vec<FaceRef>,
    /// The queries the faces were picked by — see [`FilletFeature::edges_where`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces_where: Vec<StoredPick<FaceRef>>,
    pub thickness: f64,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// An edge a fillet rounds at a varying radius: from `radius[0]` at its first end to
/// `radius[1]` at its other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariableEdge {
    pub edge: EdgeRef,
    pub radius: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FilletFeature {
    pub body: BodyId,
    pub edges: Vec<EdgeRef>,
    /// The queries the edges were picked by, each with what it found then — run again at build
    /// on the body as it stands before the feature; `edges` holds what each found, the
    /// fallback where it finds nothing now.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges_where: Vec<StoredPick<EdgeRef>>,
    pub radius: f64,
    /// Whether the edges tangent-continuous with the picked ones are rounded too.
    pub chain: bool,
    /// Round what the kernel can: a chain it cannot round is left out, and the result's note
    /// says where, instead of the whole round failing.
    #[serde(default, skip_serializing_if = "is_false")]
    pub skip_failing: bool,
    /// Edges rounded at a radius running evenly from the first to the second along each.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variable: Vec<VariableEdge>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// How a chamfer cuts each face back from its edge: by one distance on both, by two —
/// `distance` on the edge's reference face, `distance2` on the other — or by `distance` on
/// the reference face and the angle the chamfer makes with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ChamferMode {
    Equal,
    TwoDistances,
    DistanceAngle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChamferFeature {
    pub body: BodyId,
    pub edges: Vec<EdgeRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges_where: Vec<StoredPick<EdgeRef>>,
    pub distance: f64,
    pub chain: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub skip_failing: bool,
    /// Absent: one distance on both faces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<ChamferMode>,
    /// Two distances: the setback on the face across from the reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance2: Option<f64>,
    /// Distance and angle: degrees between the chamfer and the reference face, under 90.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle_deg: Option<f64>,
    /// The reference face of each edge is the face shared by the most of the edges chamfered,
    /// or the kernel's first; flipped, the other face.
    #[serde(default, skip_serializing_if = "is_false")]
    pub flip: bool,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// What a hole is drilled along: a cylinder or circle element's axis, a cylindrical face of
/// the body itself — both ways from its point — or a flat face of the body, drilled into the
/// body square to it from a point on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum HoleAxis {
    Element { id: ElementId },
    Face { face: FaceRef },
    Point { face: FaceRef, at: DVec3 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HoleFeature {
    pub body: BodyId,
    pub axis: HoleAxis,
    pub radius: f64,
    /// Millimetres along the axis from its point, both ways — into the body alone from a flat
    /// face; `None` (null) drills through everything.
    pub depth: Option<f64>,
    /// The heat-set insert it is sized for, by its label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub insert: Option<String>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// The standard a thread's sizes come from, and with them its form: the 60° one of the metric
/// and Unified tables, or the 45° one for printing (ScanRuler's `thread.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThreadStandard {
    Metric,
    Unc,
    Unf,
    #[serde(rename = "print45")]
    Print45,
}

impl ThreadStandard {
    pub const ALL: [ThreadStandard; 4] = [ThreadStandard::Metric, ThreadStandard::Unc, ThreadStandard::Unf, ThreadStandard::Print45];

    pub fn label(self) -> &'static str {
        match self {
            ThreadStandard::Metric => "ISO metric",
            ThreadStandard::Unc => "Unified coarse (UNC)",
            ThreadStandard::Unf => "Unified fine (UNF)",
            ThreadStandard::Print45 => "3D printing — 45° flanks",
        }
    }
}

/// A screw thread cut on a full round face of a body: a bolt's on a boss, a nut's in a bore.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadFeature {
    pub body: BodyId,
    pub face: FaceRef,
    pub standard: ThreadStandard,
    /// The major diameter, millimetres.
    pub diameter: f64,
    pub pitch: f64,
    pub left_handed: bool,
    /// Millimetres along the face; `None` (null) runs the whole face.
    pub length: Option<f64>,
    /// Millimetres in from the face's end that a thread with a length starts at.
    pub start: f64,
    /// Measured from the other end of the face.
    pub flip: bool,
    /// A clearance for printing: millimetres every face of the thread is moved into the
    /// material.
    pub offset: f64,
    /// A 45° chamfer where the thread runs out of the body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chamfer: Option<bool>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// A measured element as a body at its extents: a plane as a slab, a cylinder or a cone as a
/// solid of its length, a sphere as a ball — or a section as the block it stands as. One of
/// `element` and `section`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractFeature {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub element: Option<ElementId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<ElementId>,
    /// A slab's thickness, centred on the plane.
    pub thickness: f64,
    pub op: BodyOp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<BodyId>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CutPlaneFeature {
    pub body: BodyId,
    pub plane: PlaneRef,
    /// Which side stays: the side the plane's normal points to (1) or the other (−1).
    pub keep: i8,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// What a mirror is made about, and a draft's neutral plane: a plane, or a flat face of the
/// body by its fingerprint.
#[derive(Debug, Clone, PartialEq)]
pub enum MirrorPlane {
    Plane(PlaneRef),
    Face { face: FaceRef },
}

impl Serialize for MirrorPlane {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct FaceOut<'a> {
            kind: &'static str,
            face: &'a FaceRef,
        }
        match self {
            MirrorPlane::Plane(p) => p.serialize(s),
            MirrorPlane::Face { face } => FaceOut { kind: "face", face }.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for MirrorPlane {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<MirrorPlane, D::Error> {
        let v = Value::deserialize(d)?;
        if v.get("kind").and_then(Value::as_str) == Some("face") {
            let face = v.get("face").cloned().ok_or_else(|| de::Error::missing_field("face"))?;
            let face: FaceRef = serde_json::from_value(face).map_err(de::Error::custom)?;
            return Ok(MirrorPlane::Face { face });
        }
        serde_json::from_value(v).map(MirrorPlane::Plane).map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MirrorFeature {
    pub body: BodyId,
    pub plane: MirrorPlane,
    /// Joined with the source into one body, or a body of its own.
    pub join: bool,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// Faces a draft leans at an angle of their own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DraftGroup {
    pub faces: Vec<FaceRef>,
    pub angle_deg: f64,
}

/// Faces of a body leaned about a neutral plane — a moulding's draft, put on after the fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DraftFeature {
    pub body: BodyId,
    pub neutral: MirrorPlane,
    pub faces: Vec<FaceRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces_where: Vec<StoredPick<FaceRef>>,
    pub angle_deg: f64,
    /// More faces, each group at an angle of its own, leaned after `faces` in the order given.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<DraftGroup>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// Faces of a body moved by one offset: a flat face along its outward normal, a full
/// cylinder's radius changed by it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", from = "MoveFaceFeatureJson")]
pub struct MoveFaceFeature {
    pub body: BodyId,
    pub faces: Vec<FaceRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces_where: Vec<StoredPick<FaceRef>>,
    pub offset: f64,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// Faces of a body taken off and the faces round them run on to close the gap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteFaceFeature {
    pub body: BodyId,
    pub faces: Vec<FaceRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces_where: Vec<StoredPick<FaceRef>>,
}

/// A body grown all round by a distance, or shrunk by a negative one. Kept, the body stays as
/// it was and the offset is a body of its own; not, the body becomes the offset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OffsetBodyFeature {
    pub body: BodyId,
    /// Millimetres out — in, negative.
    pub distance: f64,
    pub keep: bool,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// A STEP file as an Import holds it: the file's name and size as it came in, and the key its
/// text is kept under in the part's `step_files` — a digest of the text, so the same file
/// imported twice is kept once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepFileRef {
    pub key: String,
    pub name: String,
    pub size: u64,
}

/// The key a STEP file's text is kept under: its length and a 53-bit digest of it (cyrb53),
/// so one file imported twice is one file. Counted in UTF-16 units, as ScanRuler counts them,
/// so a part's keys agree with ScanRuler's.
pub fn step_file_key(text: &str) -> String {
    let mut h1: u32 = 0xdead_beef;
    let mut h2: u32 = 0x41c6_ce57;
    let mut n: u64 = 0;
    for c in text.encode_utf16() {
        let c = c as u32;
        h1 = (h1 ^ c).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ c).wrapping_mul(1_597_334_677);
        n += 1;
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507) ^ (h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507) ^ (h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909);
    let digest: u64 = 4_294_967_296u64 * (2_097_151 & h2) as u64 + h1 as u64;
    format!("{}-{}", base36(n), base36(digest))
}

fn base36(mut x: u64) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if x == 0 {
        return "0".into();
    }
    let mut out = Vec::new();
    while x > 0 {
        out.push(DIGITS[(x % 36) as usize]);
        x /= 36;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// A STEP file brought in as bodies of their own — a body per solid in the file, named after
/// its part where the file names it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportFeature {
    pub file: StepFileRef,
}

/// Bodies moved and turned — or copies of them: turned about `pivot` about X, then Y, then Z
/// by `rotate` degrees, then carried by `translate` millimetres.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveBodyFeature {
    pub bodies: Vec<BodyId>,
    pub pivot: DVec3,
    pub translate: DVec3,
    pub rotate: DVec3,
    /// The bodies stay where they are and copies of them are moved.
    pub copy: bool,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// Two bodies made separately into one: the tool goes into the body — or, kept, stays a body
/// of its own beside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooleanFeature {
    pub op: BooleanOp,
    pub body: BodyId,
    pub tool: BodyId,
    #[serde(default, skip_serializing_if = "is_false")]
    pub keep: bool,
}

/// A sketch's pieces extruded as a surface, not a solid: a cutter to split a body with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SheetFeature {
    pub sketch_id: FeatureId,
    /// Millimetres along the sketch plane's normal, either way.
    pub from: f64,
    pub to: f64,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// Which fragments of a split body are kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitKeep {
    Scan,
    Largest,
    Smallest,
    Between,
}

/// Which way a sheet is thickened: along its normal (out), against it (in), or half each way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThickenSide {
    In,
    Out,
    Both,
}

/// A sheet made a solid of a thickness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThickenFeature {
    /// The sheet body thickened.
    pub sheet: BodyId,
    pub thickness: f64,
    pub side: ThickenSide,
    pub op: BodyOp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<BodyId>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitFeature {
    pub body: BodyId,
    /// The sheet bodies (and solid bodies) cutting it.
    pub tools: Vec<BodyId>,
    pub keep: SplitKeep,
    /// How far a sampled point of a fragment's face may lie from the scan and still count as
    /// on it, millimetres.
    pub tolerance: f64,
    /// The share of a fragment's samples that must lie on the scan.
    pub overlap: f64,
    /// Cut within the fitted sheets' regions, each grown by this many millimetres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within: Option<f64>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// A solid through the profiles of two or more sketches, each on its own plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoftFeature {
    /// In order; each sketch's one region, or its largest.
    pub sketches: Vec<FeatureId>,
    /// Straight between the profiles, or a smooth surface through them.
    pub ruled: bool,
    pub op: BodyOp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<BodyId>,
}

/// A centreline of straights and bends in the part's frame, for a sweep with no path sketch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepRoute {
    /// In order, either end first: the sweep starts from the end nearest the profile.
    pub points: Vec<DVec3>,
    /// The radius of every bend `radii` does not name, on the centreline.
    pub bend_radius: f64,
    /// One per inner point in order: that bend's own radius, or null for `bend_radius`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radii: Option<Vec<Option<f64>>>,
}

/// A profile swept along a path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepFeature {
    /// The sketch whose largest region is swept.
    pub profile: FeatureId,
    /// The sketch whose one chain the profile follows. Absent where the sweep has a route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<FeatureId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<SweepRoute>,
    pub op: BodyOp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<BodyId>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PatternMode {
    Circular,
    Linear,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PatternFeature {
    pub body: BodyId,
    pub mode: PatternMode,
    /// Circular: the axis turned about. Linear: the direction stepped along.
    pub axis: AxisRef,
    pub count: u32,
    /// Circular: the angle from the first copy to the last, degrees — 360 is a full turn,
    /// `count` copies round with none on the first (see [`pattern_pitch`]). Linear: the
    /// spacing, millimetres.
    pub step: f64,
    /// Linear only: a second direction the row is repeated along — a grid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis2: Option<AxisRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count2: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step2: Option<f64>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// Whether a circular pattern's step is a full turn.
pub fn full_turn(step_deg: f64) -> bool {
    (step_deg.abs() - 360.0).abs() < 1e-9
}

/// The angle from one copy of a circular pattern to the next, degrees: a full turn shared out
/// among `count`, an open span among `count − 1`.
pub fn pattern_pitch(step_deg: f64, count: f64) -> f64 {
    let n = count.round().max(1.0);
    if n > 1.0 {
        step_deg / if full_turn(step_deg) { n } else { n - 1.0 }
    } else {
        0.0
    }
}

/// The net size of a freeform fit's control net.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NetSize {
    pub u: u32,
    pub v: u32,
}

/// The parameters a freeform surface is fitted with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceFitParams {
    /// The control net, U × V, or null for the fit to choose.
    pub net: Option<NetSize>,
    pub degree: u32,
    /// How far past the region's bounds the surface reaches, as a share of its extent.
    pub expand_pct: f64,
    /// The weight of the thin-plate term relative to the data's.
    pub smoothness: f64,
    /// Fit again without the points that stand far off the first fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust: Option<bool>,
    /// The height direction the surface is fitted along. Absent: the region's own normal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal: Option<DVec3>,
}

/// A box in the part's frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoxBounds {
    pub min: DVec3,
    pub max: DVec3,
}

/// The way a region read in a box faces.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Facing {
    pub dir: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_deg: Option<f64>,
}

/// The box a surface's region was read in off the scan, with what else the query asked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionBox {
    #[serde(rename = "box")]
    pub bounds: BoxBounds,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal: Option<Facing>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<Vec<BoxBounds>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// A fitted surface's kind: the freeform B-spline, or a plane or a cylinder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SurfaceShape {
    Freeform,
    Plane,
    Cylinder,
}

/// How a fitted plane or cylinder relates to a reference (ScanRuler's `autoConstrain.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Relation {
    Along,
    Across,
    On,
}

/// A plane's or a cylinder's Auto Constrain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum ConstrainMode {
    /// Hold what the free fit lands within the tolerances of.
    Auto,
    /// The free fit, whatever it is near.
    Free,
    /// This relation to this reference, however far the free fit is from it.
    Force {
        relation: Relation,
        #[serde(rename = "ref")]
        reference: String,
    },
}

/// That the region's box moved with the parameters since the region was read — or, a string,
/// why the points could not be read again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RegionStale {
    Yes(bool),
    Why(String),
}

/// A freeform surface fitted through a region of the scan — a body of its own, marked as a
/// surface. The region is a set of the scan's vertices, held by the feature alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceFeature {
    pub region: Vec<u32>,
    pub fit: SurfaceFitParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<SurfaceShape>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrain: Option<ConstrainMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_box: Option<RegionBox>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_stale: Option<RegionStale>,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

/// Where a network's node lies on a body: along an edge, or at one of its ends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NodeOn {
    Edge { body: BodyId, edge: EdgeRef, t: f64 },
    Vertex { body: BodyId, vertex: EdgeRef, end: u8 },
}

/// A point of a patch network: a curve's end, or a corner — on the scan, or on a body's edge
/// or corner (`on`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: u32,
    pub at: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on: Option<NodeOn>,
    /// Whether the node is a corner of the cells it bounds, over the rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner: Option<bool>,
}

/// A curve running along a body's edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveAlong {
    pub body: BodyId,
    pub edge: EdgeRef,
}

/// How the patch beside a body's face meets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SideMode {
    Contact,
    Tangent,
}

/// The patch beside a body's face meets it there: on its edge (contact) or tangent to it too.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveSide {
    pub body: BodyId,
    pub face: FaceRef,
    pub mode: SideMode,
}

/// A curve of a patch network, from one node to another through its grips (`via`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkCurve {
    pub id: u32,
    pub from: u32,
    pub to: u32,
    pub via: Vec<DVec3>,
    /// A fold or a parting line: the patches either side meet at an angle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crease: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub along: Option<CurveAlong>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub side: Option<CurveSide>,
}

/// A cell of a network filled with a patch: which region was filled, by a point inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkCell {
    pub seed: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centre: Option<DVec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corners: Option<Vec<u32>>,
}

/// How a network's patches are fitted to the scan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkFitParams {
    pub smoothness: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
}

/// A Patch network: curves drawn on the scan, and the regions they close filled with B-spline
/// patches fitted to the scan together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkFeature {
    pub nodes: Vec<NetworkNode>,
    pub curves: Vec<NetworkCurve>,
    pub cells: Vec<NetworkCell>,
    /// The span count of a strip of patches, keyed by the id of a curve in it.
    pub spans: BTreeMap<String, u32>,
    pub fit: NetworkFitParams,
}

/// What an automatic trim cuts space with: a surface body or a measured element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TrimTool {
    Body { id: BodyId },
    Element { id: ElementId },
}

/// A set of surfaces closed into a solid: a box round the scan is split by every tool, and of
/// the faces that makes, the ones lying on the scan are kept and sewn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrimFeature {
    pub tools: Vec<TrimTool>,
    pub tolerance: f64,
    pub overlap: f64,
    #[serde(default, skip_serializing_if = "Exprs::is_empty")]
    pub exprs: Exprs,
}

// ---- the feature ----

/// What every feature carries beside its own fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Left out of the build — the part built as if the feature were not there — and kept in
    /// the list, its picks and numbers as they were, to be built again.
    #[serde(default, skip_serializing_if = "is_false")]
    pub suppressed: bool,
    /// Shown in the viewport (a sketch drawn on the part).
    #[serde(default = "yes")]
    pub visible: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        FeatureFlags { suppressed: false, visible: true }
    }
}

/// Every kind of feature, with its own fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum FeatureKind {
    Sketch(SketchFeature),
    Extrude(ExtrudeFeature),
    Revolve(RevolveFeature),
    Extract(ExtractFeature),
    Sheet(SheetFeature),
    Loft(LoftFeature),
    Sweep(SweepFeature),
    Thicken(ThickenFeature),
    Surface(SurfaceFeature),
    Network(NetworkFeature),
    Trim(TrimFeature),
    Import(ImportFeature),
    MoveBody(MoveBodyFeature),
    Shell(ShellFeature),
    Fillet(FilletFeature),
    Chamfer(ChamferFeature),
    Hole(HoleFeature),
    Thread(ThreadFeature),
    CutPlane(CutPlaneFeature),
    Mirror(MirrorFeature),
    Boolean(BooleanFeature),
    Split(SplitFeature),
    Pattern(PatternFeature),
    Draft(DraftFeature),
    MoveFace(MoveFaceFeature),
    DeleteFace(DeleteFaceFeature),
    OffsetBody(OffsetBodyFeature),
}

impl FeatureKind {
    /// The kind as ScanRuler writes it.
    pub fn kind(&self) -> &'static str {
        match self {
            FeatureKind::Sketch(_) => "sketch",
            FeatureKind::Extrude(_) => "extrude",
            FeatureKind::Revolve(_) => "revolve",
            FeatureKind::Extract(_) => "extract",
            FeatureKind::Sheet(_) => "sheet",
            FeatureKind::Loft(_) => "loft",
            FeatureKind::Sweep(_) => "sweep",
            FeatureKind::Thicken(_) => "thicken",
            FeatureKind::Surface(_) => "surface",
            FeatureKind::Network(_) => "network",
            FeatureKind::Trim(_) => "trim",
            FeatureKind::Import(_) => "import",
            FeatureKind::MoveBody(_) => "moveBody",
            FeatureKind::Shell(_) => "shell",
            FeatureKind::Fillet(_) => "fillet",
            FeatureKind::Chamfer(_) => "chamfer",
            FeatureKind::Hole(_) => "hole",
            FeatureKind::Thread(_) => "thread",
            FeatureKind::CutPlane(_) => "cutPlane",
            FeatureKind::Mirror(_) => "mirror",
            FeatureKind::Boolean(_) => "boolean",
            FeatureKind::Split(_) => "split",
            FeatureKind::Pattern(_) => "pattern",
            FeatureKind::Draft(_) => "draft",
            FeatureKind::MoveFace(_) => "moveFace",
            FeatureKind::DeleteFace(_) => "deleteFace",
            FeatureKind::OffsetBody(_) => "offsetBody",
        }
    }

    /// The kind as the feature list names it (ScanRuler's `KIND_LABELS`): a new feature is
    /// this and a count — Combine 1, Cut 2.
    pub fn kind_name(&self) -> &'static str {
        kind_label(self.kind())
    }

    /// The expressions driving the feature's numbers, on the kinds that have them.
    pub fn exprs(&self) -> Option<&Exprs> {
        match self {
            FeatureKind::Sketch(f) => Some(&f.exprs),
            FeatureKind::Extrude(f) => Some(&f.exprs),
            FeatureKind::Revolve(f) => Some(&f.exprs),
            FeatureKind::Extract(f) => Some(&f.exprs),
            FeatureKind::Sheet(f) => Some(&f.exprs),
            FeatureKind::Sweep(f) => Some(&f.exprs),
            FeatureKind::Thicken(f) => Some(&f.exprs),
            FeatureKind::Surface(f) => Some(&f.exprs),
            FeatureKind::Trim(f) => Some(&f.exprs),
            FeatureKind::MoveBody(f) => Some(&f.exprs),
            FeatureKind::Shell(f) => Some(&f.exprs),
            FeatureKind::Fillet(f) => Some(&f.exprs),
            FeatureKind::Chamfer(f) => Some(&f.exprs),
            FeatureKind::Hole(f) => Some(&f.exprs),
            FeatureKind::Thread(f) => Some(&f.exprs),
            FeatureKind::CutPlane(f) => Some(&f.exprs),
            FeatureKind::Mirror(f) => Some(&f.exprs),
            FeatureKind::Split(f) => Some(&f.exprs),
            FeatureKind::Pattern(f) => Some(&f.exprs),
            FeatureKind::Draft(f) => Some(&f.exprs),
            FeatureKind::MoveFace(f) => Some(&f.exprs),
            FeatureKind::OffsetBody(f) => Some(&f.exprs),
            FeatureKind::Loft(_) | FeatureKind::Network(_) | FeatureKind::Import(_) | FeatureKind::Boolean(_) | FeatureKind::DeleteFace(_) => None,
        }
    }

    /// The expressions of the kinds that have them, to write.
    pub fn exprs_mut(&mut self) -> Option<&mut Exprs> {
        match self {
            FeatureKind::Sketch(f) => Some(&mut f.exprs),
            FeatureKind::Extrude(f) => Some(&mut f.exprs),
            FeatureKind::Revolve(f) => Some(&mut f.exprs),
            FeatureKind::Extract(f) => Some(&mut f.exprs),
            FeatureKind::Sheet(f) => Some(&mut f.exprs),
            FeatureKind::Sweep(f) => Some(&mut f.exprs),
            FeatureKind::Thicken(f) => Some(&mut f.exprs),
            FeatureKind::Surface(f) => Some(&mut f.exprs),
            FeatureKind::Trim(f) => Some(&mut f.exprs),
            FeatureKind::MoveBody(f) => Some(&mut f.exprs),
            FeatureKind::Shell(f) => Some(&mut f.exprs),
            FeatureKind::Fillet(f) => Some(&mut f.exprs),
            FeatureKind::Chamfer(f) => Some(&mut f.exprs),
            FeatureKind::Hole(f) => Some(&mut f.exprs),
            FeatureKind::Thread(f) => Some(&mut f.exprs),
            FeatureKind::CutPlane(f) => Some(&mut f.exprs),
            FeatureKind::Mirror(f) => Some(&mut f.exprs),
            FeatureKind::Split(f) => Some(&mut f.exprs),
            FeatureKind::Pattern(f) => Some(&mut f.exprs),
            FeatureKind::Draft(f) => Some(&mut f.exprs),
            FeatureKind::MoveFace(f) => Some(&mut f.exprs),
            FeatureKind::OffsetBody(f) => Some(&mut f.exprs),
            FeatureKind::Loft(_) | FeatureKind::Network(_) | FeatureKind::Import(_) | FeatureKind::Boolean(_) | FeatureKind::DeleteFace(_) => None,
        }
    }

    /// The one body a feature on a body works on, for the kinds that work on one.
    pub fn body(&self) -> Option<BodyId> {
        match self {
            FeatureKind::Shell(f) => Some(f.body),
            FeatureKind::Fillet(f) => Some(f.body),
            FeatureKind::Chamfer(f) => Some(f.body),
            FeatureKind::Hole(f) => Some(f.body),
            FeatureKind::Thread(f) => Some(f.body),
            FeatureKind::CutPlane(f) => Some(f.body),
            FeatureKind::Mirror(f) => Some(f.body),
            FeatureKind::Boolean(f) => Some(f.body),
            FeatureKind::Split(f) => Some(f.body),
            FeatureKind::Pattern(f) => Some(f.body),
            FeatureKind::Draft(f) => Some(f.body),
            FeatureKind::MoveFace(f) => Some(f.body),
            FeatureKind::DeleteFace(f) => Some(f.body),
            FeatureKind::OffsetBody(f) => Some(f.body),
            _ => None,
        }
    }

    /// The sketches a feature takes its profiles from (an extrude's, a loft's), in order.
    pub fn source_sketches(&self) -> Vec<FeatureId> {
        match self {
            FeatureKind::Extrude(e) if e.face.is_none() => vec![e.sketch_id],
            FeatureKind::Revolve(r) => vec![r.sketch_id],
            FeatureKind::Sheet(s) => vec![s.sketch_id],
            FeatureKind::Loft(l) => l.sketches.clone(),
            FeatureKind::Sweep(s) => std::iter::once(s.profile).chain(s.path).collect(),
            _ => Vec::new(),
        }
    }
}

/// The label of a feature kind as ScanRuler's feature list names it.
pub fn kind_label(kind: &str) -> &'static str {
    match kind {
        "sketch" => "Sketch",
        "extrude" => "Extrude",
        "revolve" => "Revolve",
        "coil" => "Coil",
        "shell" => "Shell",
        "fillet" => "Fillet",
        "chamfer" => "Chamfer",
        "hole" => "Hole",
        "thread" => "Thread",
        "extract" => "Extract",
        "cutPlane" => "Cut",
        "mirror" => "Mirror",
        "boolean" => "Combine",
        "sheet" => "Sheet",
        "split" => "Split",
        "loft" => "Loft",
        "sweep" => "Sweep",
        "thicken" => "Thicken",
        "pattern" => "Pattern",
        "offsetBody" => "Offset",
        "import" => "Import",
        "moveBody" => "Move",
        "draft" => "Draft",
        "moveFace" => "Move face",
        "deleteFace" => "Delete face",
        "surface" => "Surface",
        "network" => "Patch network",
        "trim" => "Trim",
        _ => "Feature",
    }
}

/// A feature: its id and name, its flags, and its kind with the kind's own fields — written as
/// one object, as ScanRuler writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    pub id: FeatureId,
    pub name: String,
    #[serde(flatten)]
    pub flags: FeatureFlags,
    #[serde(flatten)]
    pub kind: FeatureKind,
}

impl Feature {
    pub fn suppressed(&self) -> bool {
        self.flags.suppressed
    }

    pub fn visible(&self) -> bool {
        self.flags.visible
    }
}

/// A feature edited in its box, which holds no queries: the queries its picks were given by
/// (`edges_where`, `faces_where`) carried over from what it was. Each stands while what it
/// found is still picked (see `pick_query::live_picks`).
pub fn with_kept_queries(old: &Feature, next: Feature) -> Feature {
    if old.kind.kind() != next.kind.kind() {
        return next;
    }
    let mut next = next;
    match (&old.kind, &mut next.kind) {
        (FeatureKind::Fillet(a), FeatureKind::Fillet(b)) if b.edges_where.is_empty() => b.edges_where = a.edges_where.clone(),
        (FeatureKind::Chamfer(a), FeatureKind::Chamfer(b)) if b.edges_where.is_empty() => b.edges_where = a.edges_where.clone(),
        (FeatureKind::Shell(a), FeatureKind::Shell(b)) if b.faces_where.is_empty() => b.faces_where = a.faces_where.clone(),
        (FeatureKind::Draft(a), FeatureKind::Draft(b)) if b.faces_where.is_empty() => b.faces_where = a.faces_where.clone(),
        (FeatureKind::MoveFace(a), FeatureKind::MoveFace(b)) if b.faces_where.is_empty() => b.faces_where = a.faces_where.clone(),
        (FeatureKind::DeleteFace(a), FeatureKind::DeleteFace(b)) if b.faces_where.is_empty() => b.faces_where = a.faces_where.clone(),
        _ => {}
    }
    next
}

// ---- resolved solids ----

/// One piece of a profile loop, in the part's coordinates. An arc is given by three points on
/// it; a spline is a clamped B-spline. `entity` is the sketch piece it was lifted from, when it
/// was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum LoopPiece {
    Line {
        a: DVec3,
        b: DVec3,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entity: Option<EntityId>,
    },
    Arc {
        a: DVec3,
        m: DVec3,
        b: DVec3,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entity: Option<EntityId>,
    },
    Circle {
        center: DVec3,
        normal: DVec3,
        radius: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entity: Option<EntityId>,
    },
    Spline {
        poles: Vec<DVec3>,
        knots: Vec<f64>,
        mults: Vec<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entity: Option<EntityId>,
    },
}

/// A closed region to make a face of: an outer loop and the loops of any holes in it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub outer: Vec<LoopPiece>,
    pub holes: Vec<Vec<LoopPiece>>,
}

/// The solid an extracted element or section becomes, resolved to plain geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Primitive {
    #[serde(rename_all = "camelCase")]
    Slab { origin: DVec3, normal: DVec3, basis_u: DVec3, basis_v: DVec3, extent_u: f64, extent_v: f64, thickness: f64 },
    Cylinder { origin: DVec3, axis: DVec3, radius: f64, length: f64 },
    Cone { origin: DVec3, axis: DVec3, radius1: f64, radius2: f64, length: f64 },
    Sphere { center: DVec3, radius: f64 },
    #[serde(rename_all = "camelCase")]
    Torus { center: DVec3, axis: DVec3, major_radius: f64, minor_radius: f64 },
    /// Profiles in the part stood along `dir` from `from` to `to` mm off their plane.
    Prism { profiles: Vec<Profile>, dir: DVec3, from: f64, to: f64 },
}

/// A measured element as a trim tool, resolved: a plane by its frame, the others as the
/// primitives Extract makes, which the builder oversizes to reach past the part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum TrimPrimitive {
    #[serde(rename_all = "camelCase")]
    Plane { origin: DVec3, normal: DVec3, basis_u: DVec3, basis_v: DVec3 },
    Cylinder { origin: DVec3, axis: DVec3, radius: f64, length: f64 },
    Cone { origin: DVec3, axis: DVec3, radius1: f64, radius2: f64, length: f64 },
    Sphere { center: DVec3, radius: f64 },
    #[serde(rename_all = "camelCase")]
    Torus { center: DVec3, axis: DVec3, major_radius: f64, minor_radius: f64 },
}

// ---- faces ----

/// The frame a sketch on a body's planar face lies in: the face's plane with its normal
/// outward, the sheet's zero and X by the sketch rule — or `None` for a face that is not a
/// plane.
pub fn frame_on_face(face: &FaceFingerprint) -> Option<SectionFrame> {
    if face.kind != FaceKind::Plane {
        return None;
    }
    SectionFrame::sketch_on(face.centroid, face.dir, None)
}

/// Whether Move face takes a face: a flat one, or a round one with a radius to change.
pub fn movable_face(face: &FaceFingerprint) -> bool {
    matches!(face.kind, FaceKind::Plane | FaceKind::Cylinder)
}

/// Whether a face of a build is a round one known to stop short of a full turn. A pick taken
/// back from a feature does not know (`None`), and passes.
pub fn partial_round(full_turn: Option<bool>) -> bool {
    full_turn == Some(false)
}

/// The sketch plane a body's face makes, by its fingerprint, or `None` for a face that is not
/// flat.
pub fn body_face_plane(body: BodyId, face: &FaceFingerprint) -> Option<PlaneRef> {
    let frame = frame_on_face(face)?;
    Some(PlaneRef::BodyFace { body, face: *face, frame, offset: None })
}

// ---- what a feature builds on ----

/// The ids a feature builds on: its sketch, the body it joins, cuts, works on or combines
/// with — for a sketch, the body whose face it lies on. A sketch's id is written as the body
/// id of island 0, as ScanRuler mixes the two in one list of numbers.
///
/// ScanRuler also counts, for a sketch, the bodies and sketches its projected pieces were taken
/// from (`SketchEntity.source`); this crate's sketch pieces keep no source yet (nothing projects
/// them), so a sketch here builds on the face it lies on alone.
pub fn references_of(f: &Feature) -> Vec<BodyId> {
    let mut refs: Vec<BodyId> = Vec::new();
    let add = |id: BodyId, refs: &mut Vec<BodyId>| {
        if !refs.contains(&id) {
            refs.push(id);
        }
    };
    match &f.kind {
        FeatureKind::Sketch(s) => {
            if let PlaneRef::BodyFace { body, .. } = s.plane {
                add(body, &mut refs);
            }
        }
        FeatureKind::Extrude(e) => {
            match &e.face {
                Some(p) => add(p.body, &mut refs),
                None => add(BodyId::new(e.sketch_id), &mut refs),
            }
            if let Some(t) = e.target {
                add(t, &mut refs);
            }
            if let ExtrudeExtent::ToFace { body, .. } | ExtrudeExtent::ToBody { body, .. } = e.extent {
                add(body, &mut refs);
            }
        }
        FeatureKind::Revolve(r) => {
            add(BodyId::new(r.sketch_id), &mut refs);
            if let Some(t) = r.target {
                add(t, &mut refs);
            }
        }
        FeatureKind::Extract(x) => {
            if let Some(t) = x.target {
                add(t, &mut refs);
            }
        }
        FeatureKind::Boolean(b) => {
            refs.push(b.body);
            refs.push(b.tool);
        }
        FeatureKind::Sheet(s) => add(BodyId::new(s.sketch_id), &mut refs),
        FeatureKind::Surface(s) => {
            // Held by hand to a surface fitted before it; what Auto finds is not a reference.
            if let Some(ConstrainMode::Force { reference, .. }) = &s.constrain {
                if let Some(id) = reference.strip_prefix("surface:").and_then(|n| n.parse::<f64>().ok()).and_then(BodyId::from_number) {
                    add(id, &mut refs);
                }
            }
        }
        FeatureKind::Network(n) => {
            for node in &n.nodes {
                match &node.on {
                    Some(NodeOn::Edge { body, .. }) | Some(NodeOn::Vertex { body, .. }) => add(*body, &mut refs),
                    None => {}
                }
            }
            for c in &n.curves {
                if let Some(a) = &c.along {
                    add(a.body, &mut refs);
                }
                if let Some(s) = &c.side {
                    add(s.body, &mut refs);
                }
            }
        }
        FeatureKind::Trim(t) => {
            for tool in &t.tools {
                if let TrimTool::Body { id } = tool {
                    refs.push(*id);
                }
            }
        }
        FeatureKind::Split(s) => {
            refs.push(s.body);
            refs.extend(s.tools.iter().copied());
        }
        FeatureKind::Loft(l) => {
            refs.extend(l.sketches.iter().map(|id| BodyId::new(*id)));
            if let Some(t) = l.target {
                refs.push(t);
            }
        }
        FeatureKind::Sweep(s) => {
            refs.push(BodyId::new(s.profile));
            if let Some(p) = s.path {
                refs.push(BodyId::new(p));
            }
            if let Some(t) = s.target {
                refs.push(t);
            }
        }
        FeatureKind::Thicken(t) => {
            refs.push(t.sheet);
            if let Some(x) = t.target {
                refs.push(x);
            }
        }
        FeatureKind::Import(_) => {}
        FeatureKind::MoveBody(m) => refs.extend(m.bodies.iter().copied()),
        other => {
            if let Some(b) = other.body() {
                refs.push(b);
            }
        }
    }
    refs
}

/// Features that depend on the given one — what deleting it would orphan: a feature on a body
/// it made counts, a further body of its step too.
pub fn dependents_of(id: FeatureId, features: &[Feature]) -> Vec<FeatureId> {
    features.iter().filter(|f| references_of(f).iter().any(|r| body_maker(*r) == id)).map(|f| f.id).collect()
}

/// The features one builds on, by id, for its place in the list: what it references, the
/// features that made the faces and edges it picks (their fingerprints' origin, a query's
/// madeBy), and the bodies whose faces its planes lie on.
pub fn built_on(f: &Feature) -> Vec<FeatureId> {
    let mut ids: Vec<FeatureId> = Vec::new();
    let add = |id: FeatureId, ids: &mut Vec<FeatureId>| {
        if !ids.contains(&id) {
            ids.push(id);
        }
    };
    for r in references_of(f) {
        add(body_maker(r), &mut ids);
    }
    let maker = |v: &Value| v.as_f64().and_then(BodyId::from_number).map(body_maker);
    fn walk(v: &Value, key: Option<&str>, sketch: bool, ids: &mut Vec<FeatureId>, maker: &dyn Fn(&Value) -> Option<FeatureId>) {
        match v {
            Value::Array(a) => {
                for x in a {
                    walk(x, key, sketch, ids, maker);
                }
            }
            Value::Object(o) => {
                // A face's or an edge's fingerprint: what first made it.
                if o.get("origin").is_some_and(Value::is_number) && (o.contains_key("centroid") || o.contains_key("mid")) {
                    if let Some(id) = o.get("origin").and_then(maker) {
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
                if o.get("kind").and_then(Value::as_str) == Some("bodyFace") {
                    if let Some(id) = o.get("body").and_then(maker) {
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
                for (k, x) in o {
                    if k != "sketch" || !sketch {
                        walk(x, Some(k), sketch, ids, maker);
                    }
                }
            }
            other => {
                if key == Some("madeBy") {
                    if let Some(id) = maker(other) {
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
    }
    if let Ok(v) = serde_json::to_value(f) {
        walk(&v, None, matches!(f.kind, FeatureKind::Sketch(_)), &mut ids, &maker);
    }
    ids.retain(|x| *x != f.id);
    ids
}

/// The list with the features `ids` moved, in the order they stand, to just after the feature
/// `after` (`None`: to the top) — or why not: a feature moved that would then come before one
/// it builds on, or one it is moved past that builds on it. Ids, names and picks stay as they
/// are.
pub fn moved_list(features: &[Feature], ids: &[FeatureId], after: Option<FeatureId>) -> Result<Vec<Feature>, String> {
    if after.is_some_and(|a| ids.contains(&a)) {
        return Err("after names a feature that is moved — name one that stays.".into());
    }
    let moving: Vec<&Feature> = features.iter().filter(|f| ids.contains(&f.id)).collect();
    let rest: Vec<&Feature> = features.iter().filter(|f| !ids.contains(&f.id)).collect();
    let at = match after {
        None => 0,
        Some(a) => rest.iter().position(|f| f.id == a).map_or(0, |i| i + 1),
    };
    let order: Vec<&Feature> = rest[..at].iter().copied().chain(moving).chain(rest[at..].iter().copied()).collect();
    let place = |id: FeatureId| order.iter().position(|f| f.id == id);
    for (fi, f) in order.iter().enumerate() {
        for used in built_on(f) {
            let Some(ui) = place(used) else { continue };
            if ui < fi || !(ids.contains(&f.id) || ids.contains(&used)) {
                continue;
            }
            let u = order[ui];
            return Err(if ids.contains(&f.id) {
                format!("{} builds on {}, which would then come after it — move {} with it, or put {} lower.", f.name, u.name, u.name, f.name)
            } else {
                format!("{} builds on {}, which the move puts after it — move {} with it, or put {} higher.", f.name, u.name, f.name, u.name)
            });
        }
    }
    Ok(order.into_iter().cloned().collect())
}

// ---- the part's own lists ----

/// The fit statistics of an element fitted to the scan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FitStats {
    pub rms: f64,
    pub sigma: f64,
    pub used_points: u32,
    pub region_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_max: Option<f64>,
}

/// What an element is, as geometry (ScanRuler's `FitData`, and its sections' planes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ElementGeometry {
    /// A reference point.
    Point { center: DVec3 },
    /// A line (an axis): its drawn segment's middle, direction and extent.
    Line { center: DVec3, dir: DVec3, length: f64 },
    /// A circle: its centre, the normal of its plane, its radius.
    Circle { center: DVec3, normal: DVec3, radius: f64 },
    /// A plane: the middle of its patch, its normal, and the patch's axes and half-extents.
    #[serde(rename_all = "camelCase")]
    Plane { center: DVec3, normal: DVec3, basis_u: DVec3, basis_v: DVec3, extent_u: f64, extent_v: f64 },
    /// A cylinder: a point on the axis at the middle of the surface, the axis, the radius, the
    /// axial extent and the degrees of arc it wraps.
    Cylinder { center: DVec3, axis: DVec3, radius: f64, length: f64, coverage: f64 },
    /// A cone: the axis pointing the way the radius grows, the half angle in degrees, the
    /// radius at `center` and at the small and large ends.
    #[serde(rename_all = "camelCase")]
    Cone { center: DVec3, axis: DVec3, half_angle: f64, radius: f64, radius1: f64, radius2: f64, length: f64, coverage: f64 },
    Sphere { center: DVec3, radius: f64 },
    /// A torus: the centre of the spine circle, its axis, the spine's and the tube's radii.
    #[serde(rename_all = "camelCase")]
    Torus { center: DVec3, axis: DVec3, major_radius: f64, minor_radius: f64, tube_coverage: f64, spine_coverage: f64 },
    /// A section plane: where it cuts, as its frame.
    Section { frame: SectionFrame },
}

impl ElementGeometry {
    pub fn kind(&self) -> &'static str {
        match self {
            ElementGeometry::Point { .. } => "point",
            ElementGeometry::Line { .. } => "line",
            ElementGeometry::Circle { .. } => "circle",
            ElementGeometry::Plane { .. } => "plane",
            ElementGeometry::Cylinder { .. } => "cylinder",
            ElementGeometry::Cone { .. } => "cone",
            ElementGeometry::Sphere { .. } => "sphere",
            ElementGeometry::Torus { .. } => "torus",
            ElementGeometry::Section { .. } => "section",
        }
    }
}

/// A measured element or a section plane, which features refer to by id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub id: ElementId,
    pub name: String,
    #[serde(flatten)]
    pub geometry: ElementGeometry,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<FitStats>,
}

/// The kind of a parameter's value, and a named expression of a kind, in one list for the
/// whole part, outside the feature list and the rollback bar — the evaluator's own types
/// (`expression.rs`, `parameters.rs`), saved in the same shape.
pub use crate::expression::ParameterKind;
pub use crate::parameters::Parameter;

/// A STEP file the part holds for its imports, by key (see [`step_file_key`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepFile {
    pub name: String,
    pub size: u64,
    pub text: String,
}

/// What a body is called and wears where it is given its own; a body with none is Solid n.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BodyStyle {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// #rrggbb.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

/// A component of the body list: a named folder the bodies are sorted into, one inside another
/// (ScanRuler's `bodyFolders.ts`, the data alone).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BodyFolder {
    pub id: u32,
    pub name: String,
    /// The component it is in, or `None` (null) at the head of the list.
    pub parent: Option<u32>,
    /// The bodies put in it — those not built at the moment too.
    pub bodies: Vec<BodyId>,
}

/// Who pinned a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NoteBy {
    Agent,
}

/// What a note was pinned on: the scan, or a face, an edge or a corner of a body — kept by
/// fingerprint, as a feature keeps its picks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum NoteTarget {
    Scan,
    Face { body: BodyId, face: FaceFingerprint },
    Edge { body: BodyId, edge: EdgeFingerprint },
    Vertex { body: BodyId, point: DVec3 },
}

/// A note pinned to the part between the person at the screen and an agent (ScanRuler's
/// `notes.ts`, the data alone).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Note {
    /// Counts from one; the number on the pin.
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<NoteBy>,
    pub text: String,
    /// Where the pin stands, mm, in the part's frame.
    pub at: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal: Option<DVec3>,
    pub on: Vec<NoteTarget>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply: Option<String>,
}

/// The version of the part's JSON this model reads and writes. A part saved by the earlier
/// model has none and is refused, cleanly.
pub const PART_VERSION: u32 = 2;

/// The part: its features in order, and the lists they refer to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    pub version: u32,
    pub features: Vec<Feature>,
    pub next_id: FeatureId,
    /// How many of each kind were named, for the next one's name.
    #[serde(default)]
    pub next_of_kind: BTreeMap<String, u32>,
    /// Features after this many are rolled back (not built); `None` builds them all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rolled_to: Option<usize>,
    /// The measured elements and section planes features refer to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub elements: Vec<Element>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<Parameter>,
    /// The id the next parameter takes (ScanRuler's `nextParameterId`); none in a part saved
    /// without one, which counts on from the largest (see `part_parameters`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_parameter_id: Option<u32>,
    /// The STEP files the imports hold, by key.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub step_files: BTreeMap<String, StepFile>,
    /// Names and colours given to bodies, by body.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty", with = "body_keyed")]
    pub body_styles: BTreeMap<BodyId, BodyStyle>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub body_folders: Vec<BodyFolder>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<Note>,
    /// ---- PICKLIST ---- Whether the volume two bodies share where they run into each other is
    /// drawn (ScanRuler's `overlapsShown`): a view setting kept with the part, written only when
    /// off, and no step of the undo history.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub overlaps_shown: bool,
}

impl Default for Part {
    fn default() -> Self {
        Part {
            version: PART_VERSION,
            features: Vec::new(),
            next_id: 1,
            next_of_kind: BTreeMap::new(),
            rolled_to: None,
            elements: Vec::new(),
            parameters: Vec::new(),
            next_parameter_id: None,
            step_files: BTreeMap::new(),
            body_styles: BTreeMap::new(),
            body_folders: Vec::new(),
            notes: Vec::new(),
            overlaps_shown: true,
        }
    }
}

impl Part {
    pub fn feature(&self, id: FeatureId) -> Option<&Feature> {
        self.features.iter().find(|f| f.id == id)
    }

    pub fn feature_mut(&mut self, id: FeatureId) -> Option<&mut Feature> {
        self.features.iter_mut().find(|f| f.id == id)
    }

    pub fn index_of(&self, id: FeatureId) -> Option<usize> {
        self.features.iter().position(|f| f.id == id)
    }

    pub fn element(&self, id: ElementId) -> Option<&Element> {
        self.elements.iter().find(|e| e.id == id)
    }

    /// The name the next feature of a kind takes: its label and one past the count of that
    /// kind named so far.
    pub fn next_name(&self, kind: &FeatureKind) -> String {
        let n = self.next_of_kind.get(kind.kind()).copied().unwrap_or(0) + 1;
        format!("{} {n}", kind.kind_name())
    }

    /// Add a feature at the end, named after its kind and the count of that kind.
    pub fn add(&mut self, kind: FeatureKind) -> FeatureId {
        let name = self.next_name(&kind);
        self.add_named(kind, name)
    }

    /// Add a feature under a name of its own: at the end of the features built — at the roll-back
    /// bar where the list is rolled back, before the features under it, as ScanRuler appends to
    /// its `features` and keeps those under the bar in `rolledBack`.
    pub fn add_named(&mut self, kind: FeatureKind, name: String) -> FeatureId {
        self.next_id = self.next_id.max(1).max(self.features.iter().map(|f| f.id + 1).max().unwrap_or(1));
        let id = self.next_id;
        self.next_id += 1;
        *self.next_of_kind.entry(kind.kind().to_string()).or_insert(0) += 1;
        let at = self.rolled_to.map_or(self.features.len(), |r| r.min(self.features.len()));
        self.features.insert(at, Feature { id, name, flags: FeatureFlags::default(), kind });
        if let Some(r) = self.rolled_to.as_mut() {
            *r = at + 1;
        }
        id
    }

    /// Remove a feature, and every feature that depends on it, at any depth (an extrude of a
    /// deleted sketch, a round on a deleted extrude's body). The ids removed.
    pub fn remove(&mut self, id: FeatureId) -> Vec<FeatureId> {
        let mut gone: Vec<FeatureId> = vec![id];
        let mut i = 0;
        while i < gone.len() {
            for d in dependents_of(gone[i], &self.features) {
                if !gone.contains(&d) {
                    gone.push(d);
                }
            }
            i += 1;
        }
        let set: HashSet<FeatureId> = gone.iter().copied().collect();
        let before = self.features.len();
        let rolled = self.rolled_to.map(|r| self.features.iter().take(r).filter(|f| set.contains(&f.id)).count());
        self.features.retain(|f| !set.contains(&f.id));
        if let (Some(r), Some(n)) = (self.rolled_to.as_mut(), rolled) {
            *r = r.saturating_sub(n).min(self.features.len());
        }
        debug_assert!(self.features.len() <= before);
        gone
    }

    pub fn sketch(&self, id: FeatureId) -> Option<&SketchFeature> {
        match &self.feature(id)?.kind {
            FeatureKind::Sketch(s) => Some(s),
            _ => None,
        }
    }

    pub fn sketch_mut(&mut self, id: FeatureId) -> Option<&mut SketchFeature> {
        match &mut self.feature_mut(id)?.kind {
            FeatureKind::Sketch(s) => Some(s),
            _ => None,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// The part a JSON text holds. A part saved by the earlier model of the workspace, or by a
    /// later one, is refused with a sentence that says so.
    pub fn from_json(text: &str) -> Result<Part, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("The part file is not JSON: {e}"))?;
        match v.get("version").and_then(Value::as_u64) {
            None => return Err("The part was saved by an earlier version of the Reverse Engineer workspace, whose features this one does not read.".into()),
            Some(n) if n > PART_VERSION as u64 => return Err(format!("The part was saved by a later version of the Reverse Engineer workspace (part version {n}; this one reads {PART_VERSION}).")),
            Some(_) => {}
        }
        let part: Part = serde_json::from_value(v).map_err(|e| format!("The part file could not be read: {e}"))?;
        // ---- PICKLIST ---- the body list's components, checked as ScanRuler checks them.
        crate::body_folders::validate_folders(&part.body_folders).map_err(|e| format!("The part file could not be read: {e}"))?;
        Ok(part)
    }
}

// ---- DRAFT ----

/// A Move face as a file holds it: saved before several faces could be picked, it holds its
/// one `face` (ScanRuler's `projectSection.ts`, `featureFromJson`).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MoveFaceFeatureJson {
    body: BodyId,
    #[serde(default)]
    faces: Vec<FaceRef>,
    #[serde(default)]
    face: Option<FaceRef>,
    #[serde(default)]
    faces_where: Vec<StoredPick<FaceRef>>,
    offset: f64,
    #[serde(default)]
    exprs: Exprs,
}

impl From<MoveFaceFeatureJson> for MoveFaceFeature {
    fn from(j: MoveFaceFeatureJson) -> Self {
        let mut faces = j.faces;
        if faces.is_empty() {
            faces.extend(j.face);
        }
        MoveFaceFeature { body: j.body, faces, faces_where: j.faces_where, offset: j.offset, exprs: j.exprs }
    }
}

#[cfg(test)]
mod tests;
