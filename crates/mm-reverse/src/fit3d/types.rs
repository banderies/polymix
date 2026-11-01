// SPDX-License-Identifier: AGPL-3.0-only
//! The measuring core's vocabulary as far as the 3D fits use it: the fitted shapes (sphere,
//! plane, cylinder, cone, torus), the fit settings, and every element kind's geometry as it
//! is fitted, picked or constructed, with what each fit reports beside its numbers (σ, the
//! form error and the residual extremes it is the span of, how many points it used). Plain
//! data throughout, so a fit crosses threads and saves as JSON as it is.
//!
//! Ported from ScanRuler (`src/core/types.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. The parsed-file and mesh-graph types of that file are not here: Polymix
//! has its own mesh (`mm_core::Mesh`), and the fits take its positions and normals as slices.

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// A sphere: centre and radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    pub center: DVec3,
    pub r: f64,
}

/// A plane as `normal · p = d`, with `|normal| = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Plane {
    pub normal: DVec3,
    pub d: f64,
}

/// An infinite cylinder: a point on the axis, a unit axis direction, a radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cylinder {
    pub point: DVec3,
    pub axis: DVec3,
    pub r: f64,
}

/// An infinite cone: a point on the axis, a unit axis direction, the surface radius at that
/// point, and the half-angle in radians. Anchored at a point on the axis rather than at the
/// apex, so the parametrisation stays tame as the half-angle approaches zero and the shape
/// becomes a cylinder. The radius grows along +axis, so the half-angle is never negative.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cone {
    pub point: DVec3,
    pub axis: DVec3,
    pub r: f64,
    pub phi: f64,
}

/// A torus: a tube of radius `minor` round the circle of radius `major` about the axis
/// through `center` — the shape of a fillet's round or a bend.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Torus {
    pub center: DVec3,
    pub axis: DVec3,
    /// The spine circle's radius (ScanRuler's `R`).
    pub major: f64,
    /// The tube's radius (ScanRuler's `r`).
    pub minor: f64,
}

/// The fit method. Gaussian best fit is the one there is; Chebyshev, minimum circumscribed
/// and maximum inscribed slot in here later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FitMethodId {
    #[default]
    Gaussian,
}

/// How a fit is made: the method, and the outlier cut-off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FitSettings {
    pub method: FitMethodId,
    /// 0 = use all points, otherwise the k in "discard residuals beyond k·sigma" (1, 2 or 3;
    /// ScanRuler's `SigmaPreset`).
    pub sigma: u8,
    /// A sphere's diameter as it is known to be, mm — a gauge ball's, a ballbar's — held by
    /// the fit, which then places the centre alone. An element's own, never the default a
    /// new one starts on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known_diameter: Option<f64>,
}

impl Default for FitSettings {
    fn default() -> Self {
        Self { method: FitMethodId::Gaussian, sigma: 3, known_diameter: None }
    }
}

impl FitSettings {
    /// The clipping factor the clipped refits take: 0 for all points.
    pub fn k(&self) -> f64 {
        f64::from(self.sigma)
    }
}

/// The span a cylinder's best fit is confined to, as millimetres past each end of the
/// surface it was measured on — the same two numbers the element is drawn with. Only a
/// negative side leaves anything out: it pulls that end of the window in, and the surface
/// beyond it — a badly captured rim at the mouth of a bore, say — stays out of the fit. A
/// positive side reaches past the surface, where there is nothing more to take in.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct AxialWindow {
    pub start: f64,
    pub end: f64,
}

/// Every kind of element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ElementKind {
    Point,
    Line,
    Plane,
    Sphere,
    Cylinder,
    Cone,
    Circle,
    Torus,
}

impl ElementKind {
    /// The kinds that are measured by fitting to the scan surface. Points are picked, lines
    /// only constructed; these five run the fit pipeline.
    pub fn is_fitted(self) -> bool {
        matches!(self, ElementKind::Sphere | ElementKind::Cylinder | ElementKind::Cone | ElementKind::Plane | ElementKind::Torus)
    }
}

/// What every fit reports regardless of geometry: the RMS form deviation and how many points
/// went into it. Picked and constructed elements have no residuals — they carry zeros here.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FitBase {
    pub sigma: f64,
    pub used_points: usize,
    pub region_size: usize,
    /// Peak-to-peak form deviation over the used points — max minus min residual, the number
    /// GD&T calls flatness / circularity / cylindricity / sphericity. Absent on picked and
    /// constructed geometry, which has no residuals to span.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form_error: Option<f64>,
    /// The signed residual extremes `form_error` is the difference of — how far the measured
    /// surface reaches below and above the fit, in the fit's own convention: along the
    /// outward normal for a plane, radially outward for a cylinder, sphere, cone or circle.
    /// On their own they say where the surface actually lies, which is what a min / max
    /// diameter and a caliper-style distance range are read from. Absent with `form_error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_max: Option<f64>,
}

/// A reference point: picked on the scan surface or constructed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointFit {
    #[serde(flatten)]
    pub base: FitBase,
    pub center: DVec3,
}

/// A reference line (an axis). Always constructed — from two points, a cylinder axis, or a
/// plane–plane intersection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LineFit {
    #[serde(flatten)]
    pub base: FitBase,
    /// Midpoint of the drawn segment.
    pub center: DVec3,
    pub dir: DVec3,
    /// Extent to draw and to sanity-check measurements against; the line the math uses is
    /// still infinite.
    pub length: f64,
}

/// What holding a known diameter did to a sphere fit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeldDiameter {
    /// The same points fitted with the radius free: their diameter, against the known one,
    /// says how far the scan sits off the ball.
    pub free_diameter: f64,
    /// How far holding the diameter moved the centre, mm.
    pub shift: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SphereFit {
    #[serde(flatten)]
    pub base: FitBase,
    pub center: DVec3,
    pub radius: f64,
    /// Set when the fit held a known diameter (`FitSettings::known_diameter`) — `radius` is
    /// then half of it, and the residuals are against it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held: Option<HeldDiameter>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CylinderFit {
    #[serde(flatten)]
    pub base: FitBase,
    /// Point on the axis at the middle of the fitted surface.
    pub center: DVec3,
    pub axis: DVec3,
    pub radius: f64,
    /// Axial extent of the fitted surface.
    pub length: f64,
    /// Degrees of arc the fitted surface wraps around the axis — under about 90° the axis
    /// position is only weakly determined, which the UI shows.
    pub coverage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConeFit {
    #[serde(flatten)]
    pub base: FitBase,
    /// Point on the axis at the middle of the fitted surface.
    pub center: DVec3,
    /// Unit axis, pointing the way the radius grows.
    pub axis: DVec3,
    /// Half of the apex angle, in degrees — like `coverage`, a number the UI shows as it is.
    pub half_angle: f64,
    /// Surface radius at `center`.
    pub radius: f64,
    /// Radii at the small and the large end of the fitted surface.
    pub radius1: f64,
    pub radius2: f64,
    /// Axial extent of the fitted surface.
    pub length: f64,
    /// Degrees of arc the fitted surface wraps around the axis — under about 90° the axis
    /// position is only weakly determined, which the UI shows.
    pub coverage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TorusFit {
    #[serde(flatten)]
    pub base: FitBase,
    /// The centre of the spine circle.
    pub center: DVec3,
    pub axis: DVec3,
    /// The spine circle's radius and the tube's.
    pub major_radius: f64,
    pub minor_radius: f64,
    /// Degrees of the tube's cross section the fitted surface wraps, and of the spine it
    /// reaches round — a fillet is a quarter of the one over a stretch of the other.
    pub tube_coverage: f64,
    pub spine_coverage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaneFit {
    #[serde(flatten)]
    pub base: FitBase,
    /// Middle of the fitted patch (lies in the plane).
    pub center: DVec3,
    pub normal: DVec3,
    /// In-plane principal axes and half-extents of the patch, for drawing it.
    pub basis_u: DVec3,
    pub basis_v: DVec3,
    pub extent_u: f64,
    pub extent_v: f64,
}

/// A circle: a centre, the unit normal of the plane it lies in, a radius. Fitted through
/// picked points, or constructed — a plane–cylinder or plane–sphere intersection, or
/// typed-in coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CircleFit {
    #[serde(flatten)]
    pub base: FitBase,
    pub center: DVec3,
    pub normal: DVec3,
    pub radius: f64,
}

/// The geometry of one element — everything except the (large) list of mesh vertices a
/// fitted one was measured on.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum FitData {
    Sphere(SphereFit),
    Cylinder(CylinderFit),
    Cone(ConeFit),
    Plane(PlaneFit),
    Point(PointFit),
    Line(LineFit),
    Circle(CircleFit),
    Torus(TorusFit),
}

impl FitData {
    pub fn kind(&self) -> ElementKind {
        match self {
            FitData::Sphere(_) => ElementKind::Sphere,
            FitData::Cylinder(_) => ElementKind::Cylinder,
            FitData::Cone(_) => ElementKind::Cone,
            FitData::Plane(_) => ElementKind::Plane,
            FitData::Point(_) => ElementKind::Point,
            FitData::Line(_) => ElementKind::Line,
            FitData::Circle(_) => ElementKind::Circle,
            FitData::Torus(_) => ElementKind::Torus,
        }
    }

    /// What every fit reports, whatever its geometry.
    pub fn base(&self) -> &FitBase {
        match self {
            FitData::Sphere(f) => &f.base,
            FitData::Cylinder(f) => &f.base,
            FitData::Cone(f) => &f.base,
            FitData::Plane(f) => &f.base,
            FitData::Point(f) => &f.base,
            FitData::Line(f) => &f.base,
            FitData::Circle(f) => &f.base,
            FitData::Torus(f) => &f.base,
        }
    }

    /// The element's centre — every kind has one.
    pub fn center(&self) -> DVec3 {
        match self {
            FitData::Sphere(f) => f.center,
            FitData::Cylinder(f) => f.center,
            FitData::Cone(f) => f.center,
            FitData::Plane(f) => f.center,
            FitData::Point(f) => f.center,
            FitData::Line(f) => f.center,
            FitData::Circle(f) => f.center,
            FitData::Torus(f) => f.center,
        }
    }
}

/// The other surface a fit's first region took in — a bore's wider step: its radius, where
/// it lies along the fit's axis from the fit's centre (mm) and its share of that region.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SecondSurface {
    pub radius: f64,
    pub band: [f64; 2],
    pub share: f64,
}

/// What a fit from a click did that its numbers do not show.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitTold {
    /// In a sentence: a cylinder whose first region took in two surfaces, fitted again on
    /// the one the seed is on.
    pub note: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second: Option<SecondSurface>,
}

/// A fit result plus the mesh region it was measured on. Picked points carry an empty region
/// — there is no surface patch to tint for them. Like the region, `told` is never kept with
/// the fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitOutput {
    pub fit: FitData,
    pub region: Vec<u32>,
    pub told: Option<FitTold>,
}

/// A fit of one kind with the mesh region it was measured on, as a fit from a click or from a
/// painted selection returns it (ScanRuler's `PlaneFit & { region }` and the like), and what
/// it did that its numbers do not show. `FitOutput` is the same with the kind erased.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionFit<T> {
    pub fit: T,
    /// The vertices the fit was measured on.
    pub region: Vec<u32>,
    pub told: Option<FitTold>,
}

impl<T> RegionFit<T> {
    /// A fit and its region, nothing told.
    pub fn new(fit: T, region: Vec<u32>) -> Self {
        Self { fit, region, told: None }
    }
}

impl<T> std::ops::Deref for RegionFit<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.fit
    }
}

macro_rules! region_fit_into_output {
    ($($fit:ident => $kind:ident),*) => {$(
        impl From<RegionFit<$fit>> for FitOutput {
            fn from(r: RegionFit<$fit>) -> Self {
                FitOutput { fit: FitData::$kind(r.fit), region: r.region, told: r.told }
            }
        }
    )*};
}
region_fit_into_output!(PlaneFit => Plane, SphereFit => Sphere, CylinderFit => Cylinder, ConeFit => Cone, TorusFit => Torus);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_data_carries_its_kind_as_scanruler_writes_it() {
        let fit = FitData::Sphere(SphereFit {
            base: FitBase { sigma: 0.01, used_points: 100, region_size: 120, form_error: Some(0.05), ..FitBase::default() },
            center: DVec3::new(1.0, 2.0, 3.0),
            radius: 5.0,
            held: None,
        });
        let json = serde_json::to_value(fit).unwrap();
        assert_eq!(json["kind"], "sphere");
        assert_eq!(json["usedPoints"], 100);
        assert_eq!(json["formError"], 0.05);
        assert!(json.get("residualMin").is_none());
        let back: FitData = serde_json::from_value(json).unwrap();
        assert_eq!(back, fit);
        assert_eq!(back.kind(), ElementKind::Sphere);
        assert!(back.kind().is_fitted());
    }
}
