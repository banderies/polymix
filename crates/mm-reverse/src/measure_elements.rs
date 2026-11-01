// SPDX-License-Identifier: AGPL-3.0-only
//! The measured elements a part is modelled against — planes, cylinders, cones, spheres, tori,
//! lines, circles and points — and the section planes: how each is made, named, coloured, read
//! out and drawn, and the planes a sketch is offered. Ported from ScanRuler
//! (`src/core/elements/kinds.ts`, `construct.ts`, `refs.ts` and `registry.ts`, `src/core/summary.ts`
//! (`formatPrimary`, `formatDetail`), `src/core/dimensions.ts` (`diameterRange`),
//! `src/state/elementModel.ts` (`elementColor`, `defaultMethod`), `src/state/palette.ts`,
//! `src/core/section/frame.ts` (`SECTION_REF_KINDS`, `describeCut`) and, from the Reverse
//! engineer workspace, `plugins/reverse/core/resolve.ts` (`coordinatePlaneOffers`,
//! `worldPlaneKey`) and `plugins/reverse/ui/PlaneSelect.tsx` (`planeOptions`, `planeKey`)),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! In ScanRuler the elements come from its 3D Measure workspace: a click on the scan grows the
//! surface under it and best-fits a primitive to it, a hand-marked surface is fitted as it is,
//! a point is picked, a circle goes through three or more picks, and lines, planes, points and
//! circles are constructed from others. Polymix has no Measure workspace; the Reverse Engineer
//! workspace makes its elements itself, through the functions here, and keeps them in the
//! part's own list (`Part::elements`), which features refer to by id.
//!
//! Where an element keeps less than ScanRuler's: the part stores an element's geometry and its
//! fit statistics, not the recipe it was made by — the seeds, the marked surface, the sources of
//! a construction — so a constructed element is evaluated once, when it is made, and does not
//! follow its sources afterwards.

use glam::DVec3;

use crate::display_units::{fmt_len, DisplayUnits, FmtOptions};
use crate::fit3d::{
    circle_from_points, fit_cone_from_seed, fit_cone_on_selection, fit_cylinder_from_seed, fit_cylinder_on_selection, fit_plane_from_seed, fit_plane_on_selection, fit_sphere_from_seed,
    fit_sphere_on_selection, fit_torus_from_seed, fit_torus_on_selection, CircleFit, ConeFit, CylinderFit, ElementKind, FitBase, FitData, FitOutput, FitSettings, LineFit, PlaneFit, PointFit, SphereFit,
    TorusFit,
};
use crate::frame::{world_cut_axis, CutAxis, SectionFrame, WorldAxis};
use crate::js_fmt::{grouped, js_round, num, to_fixed};
use crate::linalg::ortho_basis;
use crate::mesh_graph::MeshGraph;
use crate::model::{Element, ElementGeometry, ElementId, FitStats, Part, PlaneRef, RadialAxis};

// ---- the kinds ----

/// What the interface calls an element kind: a title-case label for keys and names, a noun for
/// the middle of a sentence (ScanRuler's `ELEMENT_KINDS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindInfo {
    pub kind: ElementKind,
    pub label: &'static str,
    pub noun: &'static str,
}

/// Every element kind, in the order the Create element keys show them.
pub const ELEMENT_KINDS: [KindInfo; 8] = [
    KindInfo { kind: ElementKind::Point, label: "Point", noun: "point" },
    KindInfo { kind: ElementKind::Line, label: "Line", noun: "line" },
    KindInfo { kind: ElementKind::Plane, label: "Plane", noun: "plane" },
    KindInfo { kind: ElementKind::Sphere, label: "Sphere", noun: "sphere" },
    KindInfo { kind: ElementKind::Cylinder, label: "Cylinder", noun: "cylinder" },
    KindInfo { kind: ElementKind::Cone, label: "Cone", noun: "cone" },
    KindInfo { kind: ElementKind::Circle, label: "Circle", noun: "circle" },
    KindInfo { kind: ElementKind::Torus, label: "Torus", noun: "round" },
];

pub fn kind_info(kind: ElementKind) -> KindInfo {
    ELEMENT_KINDS.iter().copied().find(|k| k.kind == kind).unwrap_or(ELEMENT_KINDS[0])
}

/// The kind of an element's geometry; `None` for a section plane, which is no element.
pub fn element_kind(g: &ElementGeometry) -> Option<ElementKind> {
    Some(match g {
        ElementGeometry::Point { .. } => ElementKind::Point,
        ElementGeometry::Line { .. } => ElementKind::Line,
        ElementGeometry::Circle { .. } => ElementKind::Circle,
        ElementGeometry::Plane { .. } => ElementKind::Plane,
        ElementGeometry::Cylinder { .. } => ElementKind::Cylinder,
        ElementGeometry::Cone { .. } => ElementKind::Cone,
        ElementGeometry::Sphere { .. } => ElementKind::Sphere,
        ElementGeometry::Torus { .. } => ElementKind::Torus,
        ElementGeometry::Section { .. } => return None,
    })
}

/// The eight accent colours everything numbered is drawn in, as 0xRRGGBB (ScanRuler's
/// `PALETTE`).
pub const PALETTE: [u32; 8] = [0x1877c0, 0xe8590c, 0x2e7d46, 0xb5179e, 0xc99a0a, 0x0f9b9b, 0xb3361c, 0x6d4bbd];

/// The colour of the nth element (counting from one): the element and, while it is a draft, the
/// surface picked for it.
pub fn element_color(num: u32) -> u32 {
    PALETTE[(num.max(1) as usize - 1) % PALETTE.len()]
}

// ---- the part's list ----

/// The id the next element or section takes: one past the highest in the part.
pub fn next_element_id(part: &Part) -> ElementId {
    part.elements.iter().map(|e| e.id).max().map_or(1, |m| m + 1)
}

/// The name the next element of a label takes — "Cylinder 2": one past the highest number an
/// element of that label carries (ScanRuler counts per kind and never goes back; the part keeps
/// no counter, so a number freed at the end of the list is given again).
pub fn next_element_name(part: &Part, label: &str) -> String {
    let n = part
        .elements
        .iter()
        .filter_map(|e| e.name.strip_prefix(label).and_then(|rest| rest.strip_prefix(' ')).and_then(|k| k.parse::<u32>().ok()))
        .max()
        .unwrap_or(0);
    format!("{label} {}", n + 1)
}

/// Add an element (or a section) to the part under a name; its id.
pub fn add_element(part: &mut Part, name: String, geometry: ElementGeometry, stats: Option<FitStats>) -> ElementId {
    let id = next_element_id(part);
    part.elements.push(Element { id, name, geometry, stats });
    id
}

// ---- fits and elements ----

/// The statistics a fit keeps beside its geometry; none for picked and constructed geometry,
/// which has no residuals (no points used).
pub fn stats_of(base: &FitBase) -> Option<FitStats> {
    (base.used_points > 0).then(|| FitStats {
        rms: base.sigma,
        sigma: base.sigma,
        used_points: base.used_points as u32,
        region_size: base.region_size as u32,
        form_error: base.form_error,
        residual_min: base.residual_min,
        residual_max: base.residual_max,
    })
}

fn base_of(stats: Option<&FitStats>) -> FitBase {
    match stats {
        Some(s) => FitBase { sigma: s.sigma, used_points: s.used_points as usize, region_size: s.region_size as usize, form_error: s.form_error, residual_min: s.residual_min, residual_max: s.residual_max },
        None => FitBase::default(),
    }
}

/// An element's geometry and statistics from a fit, as the part keeps them.
pub fn geometry_of(fit: &FitData) -> (ElementGeometry, Option<FitStats>) {
    let g = match *fit {
        FitData::Point(p) => ElementGeometry::Point { center: p.center },
        FitData::Line(l) => ElementGeometry::Line { center: l.center, dir: l.dir, length: l.length },
        FitData::Circle(c) => ElementGeometry::Circle { center: c.center, normal: c.normal, radius: c.radius },
        FitData::Plane(p) => ElementGeometry::Plane { center: p.center, normal: p.normal, basis_u: p.basis_u, basis_v: p.basis_v, extent_u: p.extent_u, extent_v: p.extent_v },
        FitData::Cylinder(c) => ElementGeometry::Cylinder { center: c.center, axis: c.axis, radius: c.radius, length: c.length, coverage: c.coverage },
        FitData::Cone(c) => ElementGeometry::Cone { center: c.center, axis: c.axis, half_angle: c.half_angle, radius: c.radius, radius1: c.radius1, radius2: c.radius2, length: c.length, coverage: c.coverage },
        FitData::Sphere(s) => ElementGeometry::Sphere { center: s.center, radius: s.radius },
        FitData::Torus(t) => ElementGeometry::Torus { center: t.center, axis: t.axis, major_radius: t.major_radius, minor_radius: t.minor_radius, tube_coverage: t.tube_coverage, spine_coverage: t.spine_coverage },
    };
    (g, stats_of(fit.base()))
}

/// An element as the fit it stands for — what it is read out and constructed from. A section
/// stands for the plane it was cut on, its patch `fallback_size` across each way.
pub fn fit_of(e: &Element, fallback_size: f64) -> FitData {
    let base = base_of(e.stats.as_ref());
    match e.geometry {
        ElementGeometry::Point { center } => FitData::Point(PointFit { base, center }),
        ElementGeometry::Line { center, dir, length } => FitData::Line(LineFit { base, center, dir, length }),
        ElementGeometry::Circle { center, normal, radius } => FitData::Circle(CircleFit { base, center, normal, radius }),
        ElementGeometry::Plane { center, normal, basis_u, basis_v, extent_u, extent_v } => FitData::Plane(PlaneFit { base, center, normal, basis_u, basis_v, extent_u, extent_v }),
        ElementGeometry::Cylinder { center, axis, radius, length, coverage } => FitData::Cylinder(CylinderFit { base, center, axis, radius, length, coverage }),
        ElementGeometry::Cone { center, axis, half_angle, radius, radius1, radius2, length, coverage } => FitData::Cone(ConeFit { base, center, axis, half_angle, radius, radius1, radius2, length, coverage }),
        ElementGeometry::Sphere { center, radius } => FitData::Sphere(SphereFit { base, center, radius, held: None }),
        ElementGeometry::Torus { center, axis, major_radius, minor_radius, tube_coverage, spine_coverage } => FitData::Torus(TorusFit { base, center, axis, major_radius, minor_radius, tube_coverage, spine_coverage }),
        ElementGeometry::Section { frame } => {
            let extent = (fallback_size * 0.3).max(1.0);
            FitData::Plane(PlaneFit { base, center: frame.origin, normal: frame.normal, basis_u: frame.u, basis_v: frame.v, extent_u: extent, extent_v: extent })
        }
    }
}

/// A fit carried through a linear map of space — the scene's axes to the part's (a rotation):
/// every point and direction mapped, the numbers kept.
pub fn map_fit(fit: &FitData, f: impl Fn(DVec3) -> DVec3) -> FitData {
    let mut out = *fit;
    match &mut out {
        FitData::Point(p) => p.center = f(p.center),
        FitData::Line(l) => {
            l.center = f(l.center);
            l.dir = f(l.dir);
        }
        FitData::Circle(c) => {
            c.center = f(c.center);
            c.normal = f(c.normal);
        }
        FitData::Plane(p) => {
            p.center = f(p.center);
            p.normal = f(p.normal);
            p.basis_u = f(p.basis_u);
            p.basis_v = f(p.basis_v);
        }
        FitData::Cylinder(c) => {
            c.center = f(c.center);
            c.axis = f(c.axis);
        }
        FitData::Cone(c) => {
            c.center = f(c.center);
            c.axis = f(c.axis);
        }
        FitData::Sphere(s) => s.center = f(s.center),
        FitData::Torus(t) => {
            t.center = f(t.center);
            t.axis = f(t.axis);
        }
    }
    out
}

/// Fit a surface element to the scan from the places clicked on it (ScanRuler's `getFitter`): the
/// surface grows from the seeds — the corners of each clicked triangle — and is best-fitted. The
/// refusal is the fit's own sentence.
pub fn fit_from_seeds(graph: &MeshGraph, kind: ElementKind, seeds: &[u32], settings: &FitSettings) -> Result<FitOutput, String> {
    let r = match kind {
        ElementKind::Plane => fit_plane_from_seed(graph, seeds, settings, None, None).map(Into::into),
        ElementKind::Sphere => fit_sphere_from_seed(graph, seeds, settings, None, None).map(Into::into),
        ElementKind::Cylinder => fit_cylinder_from_seed(graph, seeds, settings, None, None).map(Into::into),
        ElementKind::Cone => fit_cone_from_seed(graph, seeds, settings, None, None).map(Into::into),
        ElementKind::Torus => fit_torus_from_seed(graph, seeds, settings, None, None).map(Into::into),
        other => return Err(format!("Unknown element type \"{}\".", kind_info(other).noun)),
    };
    r.map_err(|e| e.0)
}

/// The same fit on a surface marked by hand (ScanRuler's `getSelectionFitter`): no search, the
/// marked vertices are the region.
pub fn fit_on_marked(graph: &MeshGraph, kind: ElementKind, selection: &[u32], settings: &FitSettings) -> Result<FitOutput, String> {
    let r = match kind {
        ElementKind::Plane => fit_plane_on_selection(graph, selection, settings).map(Into::into),
        ElementKind::Sphere => fit_sphere_on_selection(graph, selection, settings).map(Into::into),
        ElementKind::Cylinder => fit_cylinder_on_selection(graph, selection, settings, None).map(Into::into),
        ElementKind::Cone => fit_cone_on_selection(graph, selection, settings).map(Into::into),
        ElementKind::Torus => fit_torus_on_selection(graph, selection, settings).map(Into::into),
        other => return Err(format!("Element type \"{}\" cannot be fitted to a marked surface.", kind_info(other).noun)),
    };
    r.map_err(|e| e.0)
}

/// A point picked on the scan: the place itself.
pub fn picked_point(at: DVec3) -> FitData {
    FitData::Point(PointFit { base: FitBase::default(), center: at })
}

/// A circle through three or more picked points.
pub fn circle_through(points: &[DVec3]) -> Result<FitData, String> {
    circle_from_points(points).map(FitData::Circle).map_err(|e| e.0)
}

// ---- how an element is made ----

/// What an element stands for when something is built on it: a point, an axis or a plane
/// (ScanRuler's `RefRole`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefRole {
    Point,
    Axis,
    Plane,
}

/// Whether a kind plays a role: spheres, points and circles as a point; cylinders, cones, lines,
/// circles and tori as an axis; planes as a plane.
pub fn provides_role(kind: ElementKind, role: RefRole) -> bool {
    match role {
        RefRole::Point => matches!(kind, ElementKind::Point | ElementKind::Sphere | ElementKind::Circle),
        RefRole::Axis => matches!(kind, ElementKind::Line | ElementKind::Cylinder | ElementKind::Cone | ElementKind::Circle | ElementKind::Torus),
        RefRole::Plane => kind == ElementKind::Plane,
    }
}

/// How an element is made: fitted to the scan, picked on it, or constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodMode {
    Fit,
    Pick,
    Construct,
}

/// One source a construction takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotSpec {
    pub role: RefRole,
    pub label: &'static str,
    /// Narrower than the role: the kinds alone that fill it.
    pub kinds: Option<&'static [ElementKind]>,
}

/// One number a construction takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    /// A length, held in millimetres.
    pub length: bool,
    /// Found on the scan, not typed.
    pub hidden: bool,
}

/// One way of making an element (ScanRuler's `CreationMethod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreationMethod {
    pub id: &'static str,
    pub kind: ElementKind,
    pub mode: MethodMode,
    pub label: &'static str,
    pub hint: &'static str,
    pub slots: &'static [SlotSpec],
    pub params: &'static [ParamSpec],
    /// Picks a pick-mode method needs before it has a result.
    pub min_picks: usize,
    /// A construction that searches the scan, confined to a marked surface where one is marked.
    pub surface: bool,
}

const fn slot(role: RefRole, label: &'static str) -> SlotSpec {
    SlotSpec { role, label, kinds: None }
}

const fn len_param(key: &'static str, label: &'static str) -> ParamSpec {
    ParamSpec { key, label, length: true, hidden: false }
}

const fn plain_param(key: &'static str, label: &'static str) -> ParamSpec {
    ParamSpec { key, label, length: false, hidden: false }
}

const fn found(p: ParamSpec) -> ParamSpec {
    ParamSpec { hidden: true, ..p }
}

const XYZ: [ParamSpec; 3] = [len_param("x", "X"), len_param("y", "Y"), len_param("z", "Z")];
const NONE_SLOTS: [SlotSpec; 0] = [];
const NONE_PARAMS: [ParamSpec; 0] = [];
const TWO_POINTS: [SlotSpec; 2] = [slot(RefRole::Point, "Point A"), slot(RefRole::Point, "Point B")];
const TWO_PLANES: [SlotSpec; 2] = [slot(RefRole::Plane, "Plane A"), slot(RefRole::Plane, "Plane B")];
const CYLINDER_ONLY: [ElementKind; 1] = [ElementKind::Cylinder];
const SPHERE_ONLY: [ElementKind; 1] = [ElementKind::Sphere];

const fn fit_method(kind: ElementKind, hint: &'static str) -> CreationMethod {
    CreationMethod { id: "fit", kind, mode: MethodMode::Fit, label: "Fit to scan", hint, slots: &NONE_SLOTS, params: &NONE_PARAMS, min_picks: 1, surface: false }
}

const fn construct(id: &'static str, kind: ElementKind, label: &'static str, hint: &'static str, slots: &'static [SlotSpec], params: &'static [ParamSpec]) -> CreationMethod {
    CreationMethod { id, kind, mode: MethodMode::Construct, label, hint, slots, params, min_picks: 0, surface: false }
}

/// Every way an element is made, as the Create element box offers them (ScanRuler's
/// `CREATION_METHODS`, without the symmetry plane of the scan: in Polymix the Mirror box finds
/// that and keeps it as a plane element).
pub const CREATION_METHODS: &[CreationMethod] = &[
    // ---- Point ----
    CreationMethod { id: "pick", kind: ElementKind::Point, mode: MethodMode::Pick, label: "Pick on scan", hint: "Click the point on the scan you want to measure to.", slots: &NONE_SLOTS, params: &NONE_PARAMS, min_picks: 1, surface: false },
    construct("point-coords", ElementKind::Point, "From coordinates", "A fixed reference point, e.g. a datum from the drawing.", &NONE_SLOTS, &XYZ),
    CreationMethod {
        id: "point-centroid",
        kind: ElementKind::Point,
        mode: MethodMode::Construct,
        label: "Centroid of the scan",
        hint: "The centre of the volume the scan encloses, read off the mesh the moment this is chosen. An open scan gets the centre of its surface instead, and says so; a marked surface gets the centre of the marking.",
        slots: &NONE_SLOTS,
        params: &[found(len_param("x", "X")), found(len_param("y", "Y")), found(len_param("z", "Z"))],
        min_picks: 0,
        surface: true,
    },
    construct("point-midpoint", ElementKind::Point, "Midpoint of two points", "Halfway between two points or sphere centers.", &TWO_POINTS, &NONE_PARAMS),
    construct("point-line-plane", ElementKind::Point, "Line–plane intersection", "Where a line or cylinder axis pierces a plane.", &[slot(RefRole::Axis, "Line or cylinder"), slot(RefRole::Plane, "Plane")], &NONE_PARAMS),
    // ---- Line ----
    construct("line-two-points", ElementKind::Line, "Through two points", "The line through two points or sphere centers.", &TWO_POINTS, &NONE_PARAMS),
    construct("line-axis", ElementKind::Line, "Cylinder axis", "A cylinder's axis as a standalone line element.", &[slot(RefRole::Axis, "Cylinder")], &NONE_PARAMS),
    construct("line-plane-plane", ElementKind::Line, "Plane–plane intersection", "The edge line where two planes meet.", &TWO_PLANES, &NONE_PARAMS),
    // ---- Plane ----
    fit_method(ElementKind::Plane, "Click a point on the flat surface in the 3D view."),
    construct("plane-three-points", ElementKind::Plane, "Through three points", "The plane spanned by three points or sphere centers.", &[slot(RefRole::Point, "Point A"), slot(RefRole::Point, "Point B"), slot(RefRole::Point, "Point C")], &NONE_PARAMS),
    construct("plane-offset", ElementKind::Plane, "Offset from plane", "A parallel plane, shifted along the source normal.", &[slot(RefRole::Plane, "Plane")], &[len_param("offset", "Offset")]),
    construct("plane-midplane", ElementKind::Plane, "Midplane of two planes", "The symmetry plane between two (near-parallel) planes.", &TWO_PLANES, &NONE_PARAMS),
    construct(
        "plane-coords",
        ElementKind::Plane,
        "From coordinates",
        "A fixed datum plane: a normal direction and a point it passes through.",
        &NONE_SLOTS,
        &[plain_param("nx", "Normal X"), plain_param("ny", "Normal Y"), plain_param("nz", "Normal Z"), len_param("px", "Point X"), len_param("py", "Point Y"), len_param("pz", "Point Z")],
    ),
    // ---- Sphere / Cylinder / Cone / Torus ----
    fit_method(ElementKind::Sphere, "Click a point on the sphere in the 3D view."),
    fit_method(ElementKind::Cylinder, "Click a point on the cylindrical surface in the 3D view."),
    fit_method(ElementKind::Cone, "Click a point on the tapered surface in the 3D view."),
    fit_method(ElementKind::Torus, "Click a point on the middle of a round — a fillet, a bend, an O-ring seat — in the 3D view."),
    // ---- Circle ----
    CreationMethod {
        id: "circle-points",
        kind: ElementKind::Circle,
        mode: MethodMode::Pick,
        label: "Through picked points (3+)",
        hint: "Click three or more points around the feature — a hole rim, a boss edge. More points refine the fit.",
        slots: &NONE_SLOTS,
        params: &NONE_PARAMS,
        min_picks: 3,
        surface: false,
    },
    construct(
        "circle-plane-cylinder",
        ElementKind::Circle,
        "Plane–cylinder intersection",
        "The circle where a cylinder crosses a plane — a bore's rim, a boss at its base.",
        &[slot(RefRole::Plane, "Plane"), SlotSpec { role: RefRole::Axis, label: "Cylinder", kinds: Some(&CYLINDER_ONLY) }],
        &NONE_PARAMS,
    ),
    construct(
        "circle-plane-sphere",
        ElementKind::Circle,
        "Plane–sphere intersection",
        "The circle where a sphere crosses a plane.",
        &[slot(RefRole::Plane, "Plane"), SlotSpec { role: RefRole::Point, label: "Sphere", kinds: Some(&SPHERE_ONLY) }],
        &NONE_PARAMS,
    ),
    construct(
        "circle-coords",
        ElementKind::Circle,
        "From coordinates",
        "A fixed reference circle: its diameter, the normal of its plane and its center.",
        &NONE_SLOTS,
        &[len_param("d", "Diameter"), plain_param("nx", "Normal X"), plain_param("ny", "Normal Y"), plain_param("nz", "Normal Z"), len_param("cx", "Center X"), len_param("cy", "Center Y"), len_param("cz", "Center Z")],
    ),
];

pub fn methods_for_kind(kind: ElementKind) -> Vec<&'static CreationMethod> {
    CREATION_METHODS.iter().filter(|m| m.kind == kind).collect()
}

pub fn creation_method(kind: ElementKind, id: &str) -> Option<&'static CreationMethod> {
    CREATION_METHODS.iter().find(|m| m.kind == kind && m.id == id)
}

/// The method a kind's box opens on: the one that touches the scan.
pub fn default_method(kind: ElementKind) -> &'static str {
    match kind {
        ElementKind::Point => "pick",
        ElementKind::Line => "line-two-points",
        ElementKind::Circle => "circle-points",
        _ => "fit",
    }
}

/// Whether a slot takes an element of a kind.
pub fn slot_takes(s: &SlotSpec, kind: ElementKind) -> bool {
    provides_role(kind, s.role) && s.kinds.is_none_or(|k| k.contains(&kind))
}

/// The point an element stands for: a point's, a sphere's or a circle's centre.
pub fn ref_point(fit: &FitData) -> Option<DVec3> {
    match fit {
        FitData::Point(p) => Some(p.center),
        FitData::Sphere(s) => Some(s.center),
        FitData::Circle(c) => Some(c.center),
        _ => None,
    }
}

/// An axis an element stands for: a point on it, its direction, and half its measured extent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisOf {
    pub origin: DVec3,
    pub dir: DVec3,
    pub half_length: f64,
}

pub fn ref_axis(fit: &FitData) -> Option<AxisOf> {
    match fit {
        FitData::Cylinder(c) => Some(AxisOf { origin: c.center, dir: c.axis, half_length: c.length / 2.0 }),
        FitData::Cone(c) => Some(AxisOf { origin: c.center, dir: c.axis, half_length: c.length / 2.0 }),
        FitData::Line(l) => Some(AxisOf { origin: l.center, dir: l.dir, half_length: l.length / 2.0 }),
        FitData::Torus(t) => Some(AxisOf { origin: t.center, dir: t.axis, half_length: t.minor_radius }),
        FitData::Circle(c) => Some(AxisOf { origin: c.center, dir: c.normal, half_length: c.radius }),
        _ => None,
    }
}

pub fn ref_plane(fit: &FitData) -> Option<PlaneFit> {
    match fit {
        FitData::Plane(p) => Some(*p),
        _ => None,
    }
}

/// Two planes are parallel for a construction below this angle (degrees).
const INTERSECT_MIN_ANGLE: f64 = 0.2;
/// How far a cylinder may lean against a plane before their intersection is no circle.
const SECTION_MAX_TILT: f64 = 5.0;

/// The acute angle between two directions, degrees.
fn acute_angle(a: DVec3, b: DVec3) -> f64 {
    let (la, lb) = (a.length(), b.length());
    if la == 0.0 || lb == 0.0 {
        return 0.0;
    }
    let c = (a.dot(b) / (la * lb)).abs().min(1.0);
    c.acos().to_degrees()
}

fn normalize(v: DVec3) -> Option<DVec3> {
    let l = v.length();
    (l > 0.0 && l.is_finite()).then(|| v / l)
}

fn foot_on_line(p: DVec3, origin: DVec3, dir: DVec3) -> DVec3 {
    origin + dir * (p - origin).dot(dir)
}

fn need<T>(v: Option<T>, what: &str) -> Result<T, String> {
    v.ok_or_else(|| format!("Missing {what}."))
}

fn point_of(fit: Option<&FitData>, label: &str) -> Result<DVec3, String> {
    need(fit.and_then(ref_point), &format!("a point from {label}"))
}

/// A plane through `center` facing `normal`, drawn at a share of the part's size.
fn plane_at(center: DVec3, normal: DVec3, fallback_size: f64) -> PlaneFit {
    let (basis_u, basis_v) = ortho_basis(normal);
    let extent = (fallback_size * 0.3).max(1.0);
    PlaneFit { base: FitBase::default(), center, normal, basis_u, basis_v, extent_u: extent, extent_v: extent }
}

/// The geometry of a constructed element from its sources and numbers (ScanRuler's
/// `evaluateConstruction`). `fallback_size` (the part's radius) sizes what has no extent of its
/// own. A degenerate input is refused with a sentence for the user.
pub fn evaluate_construction(method: &str, refs: &[FitData], params: &[f64], fallback_size: f64) -> Result<FitData, String> {
    let p = |i: usize| params.get(i).copied().unwrap_or(f64::NAN);
    let r = |i: usize| refs.get(i);
    let none = FitBase::default();
    match method {
        "point-coords" | "point-centroid" => {
            let (x, y, z) = (p(0), p(1), p(2));
            if ![x, y, z].iter().all(|v| v.is_finite()) {
                return Err(if method == "point-centroid" { "The centroid has not been measured yet.".into() } else { "Enter all three coordinates.".into() });
            }
            Ok(FitData::Point(PointFit { base: none, center: DVec3::new(x, y, z) }))
        }
        "point-midpoint" => {
            let a = point_of(r(0), "point A")?;
            let b = point_of(r(1), "point B")?;
            Ok(FitData::Point(PointFit { base: none, center: (a + b) / 2.0 }))
        }
        "point-line-plane" => {
            let axis = need(r(0).and_then(ref_axis), "a line or cylinder axis")?;
            let pl = need(r(1).and_then(ref_plane), "the plane")?;
            if 90.0 - acute_angle(axis.dir, pl.normal) < INTERSECT_MIN_ANGLE {
                return Err("The line runs parallel to the plane — they never meet.".into());
            }
            let t = (pl.center - axis.origin).dot(pl.normal) / axis.dir.dot(pl.normal);
            Ok(FitData::Point(PointFit { base: none, center: axis.origin + axis.dir * t }))
        }
        "line-two-points" => {
            let a = point_of(r(0), "point A")?;
            let b = point_of(r(1), "point B")?;
            let dir = normalize(b - a).ok_or("The two points coincide — no line direction.")?;
            Ok(FitData::Line(LineFit { base: none, center: (a + b) / 2.0, dir, length: (b - a).length() }))
        }
        "line-axis" => {
            let axis = need(r(0).and_then(ref_axis), "an axis from the cylinder")?;
            Ok(FitData::Line(LineFit { base: none, center: axis.origin, dir: axis.dir, length: (axis.half_length * 2.0).max(fallback_size * 0.1) }))
        }
        "line-plane-plane" => {
            let pa = need(r(0).and_then(ref_plane), "plane A")?;
            let pb = need(r(1).and_then(ref_plane), "plane B")?;
            if acute_angle(pa.normal, pb.normal) < INTERSECT_MIN_ANGLE {
                return Err("The planes are parallel — they have no intersection line.".into());
            }
            let dir_raw = pa.normal.cross(pb.normal);
            let dir = normalize(dir_raw).ok_or("The planes are parallel — they have no intersection line.")?;
            let da = pa.normal.dot(pa.center);
            let db = pb.normal.dot(pb.center);
            let d2 = dir_raw.dot(dir_raw);
            let point = (pb.normal.cross(dir_raw) * da + dir_raw.cross(pa.normal) * db) / d2;
            let anchor = foot_on_line((pa.center + pb.center) / 2.0, point, dir);
            let length = (pa.extent_u.max(pa.extent_v).max(pb.extent_u).max(pb.extent_v) * 2.0).max(fallback_size * 0.05);
            Ok(FitData::Line(LineFit { base: none, center: anchor, dir, length }))
        }
        "plane-three-points" => {
            let a = point_of(r(0), "point A")?;
            let b = point_of(r(1), "point B")?;
            let c = point_of(r(2), "point C")?;
            let (ab, ac) = (b - a, c - a);
            let n_raw = ab.cross(ac);
            let span = ab.length().max(ac.length());
            if span < 1e-9 || n_raw.length() < 1e-6 * span * span {
                return Err("The three points are collinear — they do not span a plane.".into());
            }
            let normal = n_raw.normalize();
            let center = (a + b + c) / 3.0;
            let basis_u = ab.normalize();
            let basis_v = normal.cross(basis_u);
            let (mut eu, mut ev) = (0.0f64, 0.0f64);
            for q in [a, b, c] {
                let d = q - center;
                eu = eu.max(d.dot(basis_u).abs());
                ev = ev.max(d.dot(basis_v).abs());
            }
            let pad = eu.max(ev) * 0.15;
            Ok(FitData::Plane(PlaneFit { base: none, center, normal, basis_u, basis_v, extent_u: eu + pad, extent_v: ev + pad }))
        }
        "plane-offset" => {
            let pl = need(r(0).and_then(ref_plane), "the source plane")?;
            let offset = p(0);
            if !offset.is_finite() {
                return Err("Enter the offset distance.".into());
            }
            Ok(FitData::Plane(PlaneFit { base: none, center: pl.center + pl.normal * offset, ..pl }))
        }
        "plane-midplane" => {
            let pa = need(r(0).and_then(ref_plane), "plane A")?;
            let pb = need(r(1).and_then(ref_plane), "plane B")?;
            if 90.0 - acute_angle(pa.normal, pb.normal) < INTERSECT_MIN_ANGLE {
                return Err("The planes are perpendicular — no midplane between them.".into());
            }
            let nb = if pa.normal.dot(pb.normal) < 0.0 { -pb.normal } else { pb.normal };
            let normal = normalize(pa.normal + nb).ok_or("The planes are perpendicular — no midplane between them.")?;
            let center = (pa.center + pb.center) / 2.0;
            let basis_u = normalize(pa.basis_u - normal * pa.basis_u.dot(normal)).unwrap_or_else(|| ortho_basis(normal).0);
            let basis_v = normal.cross(basis_u);
            Ok(FitData::Plane(PlaneFit { base: none, center, normal, basis_u, basis_v, extent_u: pa.extent_u.max(pb.extent_u), extent_v: pa.extent_v.max(pb.extent_v) }))
        }
        "plane-coords" => {
            let v: Vec<f64> = (0..6).map(p).collect();
            if !v.iter().all(|x| x.is_finite()) {
                return Err("Enter the normal direction and a point on the plane.".into());
            }
            let normal = normalize(DVec3::new(v[0], v[1], v[2])).ok_or("The normal direction must not be zero.")?;
            Ok(FitData::Plane(plane_at(DVec3::new(v[3], v[4], v[5]), normal, fallback_size)))
        }
        "plane-symmetry" => {
            let v: Vec<f64> = (0..9).map(p).collect();
            if !v[..6].iter().all(|x| x.is_finite()) {
                return Err("Find the symmetry plane first.".into());
            }
            let normal = normalize(DVec3::new(v[0], v[1], v[2])).ok_or("The symmetry plane has no direction.")?;
            let mut plane = plane_at(DVec3::new(v[3], v[4], v[5]), normal, fallback_size);
            plane.base.sigma = if v[6].is_finite() { v[6] } else { 0.0 };
            plane.base.used_points = if v[7].is_finite() { v[7] as usize } else { 0 };
            plane.base.region_size = if v[8].is_finite() { v[8] as usize } else { 0 };
            Ok(FitData::Plane(plane))
        }
        "circle-plane-cylinder" => {
            let pl = need(r(0).and_then(ref_plane), "the plane")?;
            let Some(FitData::Cylinder(cyl)) = r(1) else { return Err("The second reference must be a cylinder.".into()) };
            let tilt = acute_angle(cyl.axis, pl.normal);
            if 90.0 - tilt < INTERSECT_MIN_ANGLE {
                return Err("The cylinder runs parallel to the plane — they never meet.".into());
            }
            if tilt > SECTION_MAX_TILT {
                return Err(format!("The cylinder leans {}° against the plane — the section is an ellipse, not a circle.", to_fixed(tilt, 1)));
            }
            let t = (pl.center - cyl.center).dot(pl.normal) / cyl.axis.dot(pl.normal);
            Ok(FitData::Circle(CircleFit { base: none, center: cyl.center + cyl.axis * t, normal: pl.normal, radius: cyl.radius }))
        }
        "circle-plane-sphere" => {
            let pl = need(r(0).and_then(ref_plane), "the plane")?;
            let Some(FitData::Sphere(sph)) = r(1) else { return Err("The second reference must be a sphere.".into()) };
            let d = (sph.center - pl.center).dot(pl.normal);
            if d.abs() >= sph.radius {
                return Err(if d.abs() - sph.radius < sph.radius * 1e-6 { "The plane only touches the sphere — the circle would have no size.".into() } else { "The plane misses the sphere — there is no intersection.".into() });
            }
            Ok(FitData::Circle(CircleFit { base: none, center: sph.center - pl.normal * d, normal: pl.normal, radius: (sph.radius * sph.radius - d * d).sqrt() }))
        }
        "circle-coords" => {
            let v: Vec<f64> = (0..7).map(p).collect();
            if !v.iter().all(|x| x.is_finite()) {
                return Err("Enter the diameter, the normal direction and the center.".into());
            }
            if !(v[0] > 0.0) {
                return Err("The diameter must be positive.".into());
            }
            let normal = normalize(DVec3::new(v[1], v[2], v[3])).ok_or("The normal direction must not be zero.")?;
            Ok(FitData::Circle(CircleFit { base: none, center: DVec3::new(v[4], v[5], v[6]), normal, radius: v[0] / 2.0 }))
        }
        other => Err(format!("Unknown construction \"{other}\".")),
    }
}

// ---- readings ----

/// The outlier cut-offs a fit offers, as the box names them.
pub const SIGMA_PRESETS: [(u8, &str); 4] = [(3, "3 sigma"), (2, "2 sigma"), (1, "1 sigma"), (0, "All points")];

fn len(mm: f64, units: DisplayUnits, decimals: usize) -> String {
    fmt_len(mm, units, decimals, FmtOptions::default())
}

fn bare(mm: f64, units: DisplayUnits, decimals: usize) -> String {
    fmt_len(mm, units, decimals, FmtOptions { bare: true, signed: false })
}

/// A position, its coordinates written bare.
fn format_pos(v: DVec3, units: DisplayUnits, decimals: usize) -> String {
    format!("({}, {}, {})", bare(v.x, units, decimals), bare(v.y, units, decimals), bare(v.z, units, decimals))
}

fn is_fitted(fit: &FitData) -> bool {
    fit.base().used_points > 0
}

/// The one number an element is listed by (ScanRuler's `formatPrimary`).
pub fn format_primary(fit: &FitData, units: DisplayUnits) -> String {
    match fit {
        FitData::Sphere(s) => format!("Ø {}", len(s.radius * 2.0, units, 3)),
        FitData::Cylinder(c) => format!("Ø {}", len(c.radius * 2.0, units, 3)),
        FitData::Circle(c) => format!("Ø {}", len(c.radius * 2.0, units, 3)),
        FitData::Cone(c) => format!("∠ {}°", to_fixed(c.half_angle * 2.0, 2)),
        FitData::Torus(t) => format!("R {}", len(t.minor_radius, units, 3)),
        FitData::Plane(p) => {
            if is_fitted(fit) {
                format!("σ {}", len(p.base.sigma, units, 4))
            } else {
                String::new()
            }
        }
        FitData::Point(p) => format_pos(p.center, units, 1),
        FitData::Line(_) => String::new(),
    }
}

fn form_error_label(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::Plane => Some("flatness"),
        ElementKind::Cylinder => Some("cylindricity"),
        ElementKind::Sphere => Some("sphericity"),
        ElementKind::Circle => Some("circularity"),
        _ => None,
    }
}

/// The smallest and largest diameter the measured surface reaches: the Gaussian radius moved in
/// to the lowest and out to the highest radial residual, doubled (ScanRuler's `diameterRange`).
pub fn diameter_range(fit: &FitData) -> Option<(f64, f64)> {
    let r = match fit {
        FitData::Sphere(s) => s.radius,
        FitData::Cylinder(c) => c.radius,
        FitData::Circle(c) => c.radius,
        _ => return None,
    };
    let b = fit.base();
    Some((2.0 * (r + b.residual_min?), 2.0 * (r + b.residual_max?)))
}

pub fn format_diameter_range(fit: &FitData, units: DisplayUnits) -> Option<String> {
    diameter_range(fit).map(|(a, b)| format!("Ø min {} · max {}", bare(a, units, 3), len(b, units, 3)))
}

/// The supporting numbers of an element, for its row's hover and the box's note (ScanRuler's
/// `formatDetail`).
pub fn format_detail(fit: &FitData, units: DisplayUnits) -> String {
    let b = fit.base();
    let points = format!("{} of {} points", grouped(b.used_points as f64), grouped(b.region_size as f64));
    let form = match b.form_error {
        Some(f) => format!(" · {} {}", form_error_label(fit.kind()).unwrap_or("form"), len(f, units, 4)),
        None => String::new(),
    };
    let dia = format_diameter_range(fit, units).map_or(String::new(), |r| format!(" · {r}"));
    let sigma = len(b.sigma, units, 4);
    match fit {
        FitData::Sphere(s) => {
            let held = s.held.map_or(String::new(), |h| format!("Ø held · free Ø {} · ", len(h.free_diameter, units, 3)));
            format!("{held}σ {sigma}{form}{dia} · {points}")
        }
        FitData::Cylinder(c) => format!("σ {sigma}{form}{dia} · length {} · arc {}° · {points}", len(c.length, units, 3), num(js_round(c.coverage))),
        FitData::Cone(c) => format!("σ {sigma}{form} · Ø {}–{} · arc {}° · {points}", bare(c.radius1 * 2.0, units, 3), len(c.radius2 * 2.0, units, 3), num(js_round(c.coverage))),
        FitData::Torus(t) => format!(
            "σ {sigma}{form} · ring Ø {} · tube {}° · round {}° · {points}",
            len(t.major_radius * 2.0, units, 3),
            num(js_round(t.tube_coverage)),
            num(js_round(t.spine_coverage))
        ),
        FitData::Plane(p) => {
            let patch = format!("{} × {} patch", bare(p.extent_u * 2.0, units, 2), len(p.extent_v * 2.0, units, 2));
            if is_fitted(fit) {
                format!("{patch}{form} · {points}")
            } else {
                patch
            }
        }
        FitData::Circle(c) => {
            if is_fitted(fit) {
                format!("σ {sigma}{form}{dia} · from {} points", b.used_points)
            } else {
                format!("center {}", format_pos(c.center, units, 3))
            }
        }
        FitData::Point(p) => format!("at {}", format_pos(p.center, units, 3)),
        FitData::Line(l) => format!("direction ({}, {}, {})", to_fixed(l.dir.x, 4), to_fixed(l.dir.y, 4), to_fixed(l.dir.z, 4)),
    }
}

// ---- sections ----

/// The element kinds a section is taken across (ScanRuler's `SECTION_REF_KINDS`).
pub const SECTION_REF_KINDS: [ElementKind; 5] = [ElementKind::Plane, ElementKind::Cylinder, ElementKind::Cone, ElementKind::Line, ElementKind::Circle];

/// What a section plane is taken across: a coordinate axis, or an element's direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionAlong {
    World(WorldAxis),
    Element(ElementId),
}

/// The line a section slides along: a coordinate axis through the origin, or an element's
/// direction through its centre — a plane's normal, a cylinder's, a cone's or a line's axis, a
/// circle's normal.
pub fn section_axis(along: SectionAlong, part: &Part) -> Option<CutAxis> {
    match along {
        SectionAlong::World(axis) => Some(world_cut_axis(axis, DVec3::ZERO)),
        SectionAlong::Element(id) => {
            let e = part.element(id)?;
            let (origin, dir) = match e.geometry {
                ElementGeometry::Plane { center, normal, .. } => (center, normal),
                ElementGeometry::Cylinder { center, axis, .. } | ElementGeometry::Cone { center, axis, .. } => (center, axis),
                ElementGeometry::Line { center, dir, .. } => (center, dir),
                ElementGeometry::Circle { center, normal, .. } => (center, normal),
                _ => return None,
            };
            let dir = normalize(dir)?;
            Some(CutAxis { origin, dir, u: ortho_basis(dir).0 })
        }
    }
}

/// The plane a section cuts on: `offset` along what it is taken across — for a coordinate axis
/// the plane's coordinate, for an element the distance from its centre.
pub fn section_frame(along: SectionAlong, offset: f64, part: &Part) -> Option<SectionFrame> {
    SectionFrame::along(section_axis(along, part)?, offset, None)
}

/// How a section's place reads: "along Cylinder 1, +2.000 mm" (ScanRuler's `describeCut`).
pub fn describe_cut(ref_name: Option<&str>, offset: f64, units: DisplayUnits) -> String {
    let mm = format!("{}{}", if offset >= 0.0 { "+" } else { "−" }, len(offset.abs(), units, 3));
    match ref_name {
        Some(n) => format!("along {n}, {mm}"),
        None => format!("{mm} from a deleted element"),
    }
}

// ---- the planes a sketch is offered ----

/// One plane on offer to a sketch, or to an extrusion's end: its key, its name, the reference it
/// stands for and the group the list shows it in.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneOffer {
    pub key: String,
    pub label: String,
    pub plane: PlaneRef,
    pub group: &'static str,
}

/// The key a coordinate plane on offer is told apart by: the plane at zero and the one through
/// the centre are two offers on one axis.
pub fn world_plane_key(axis: WorldAxis, offset: f64) -> String {
    let a = match axis {
        WorldAxis::X => "x",
        WorldAxis::Y => "y",
        WorldAxis::Z => "z",
    };
    format!("world:{a}:{}", num(js_round(offset * 1000.0) / 1000.0))
}

/// The coordinate planes a sketch is offered: at zero once the part is aligned or there is no
/// scan, and — with a scan — the same planes through its centre (ScanRuler's
/// `coordinatePlaneOffers`).
pub fn coordinate_plane_offers(centre: Option<DVec3>, at_zero: bool) -> Vec<PlaneOffer> {
    let mut out = Vec::new();
    if at_zero {
        for axis in WorldAxis::ALL {
            out.push(PlaneOffer { key: world_plane_key(axis, 0.0), label: axis.plane_name().to_string(), plane: PlaneRef::world(axis, 0.0), group: "Coordinate planes" });
        }
    }
    if let Some(c) = centre {
        for axis in WorldAxis::ALL {
            let offset = c.dot(axis.dir());
            out.push(PlaneOffer { key: world_plane_key(axis, offset), label: format!("{} through the centre", axis.plane_name()), plane: PlaneRef::world(axis, offset), group: "Coordinate planes" });
        }
    }
    out
}

/// The planes a sketch can lie on: every section, every plane element, then the coordinate
/// planes (ScanRuler's `planeOptions`).
pub fn plane_offers(part: &Part, centre: Option<DVec3>, at_zero: bool) -> Vec<PlaneOffer> {
    let mut out = Vec::new();
    for e in &part.elements {
        if let ElementGeometry::Section { .. } = e.geometry {
            out.push(PlaneOffer { key: format!("section:{}", e.id), label: e.name.clone(), plane: PlaneRef::Section { id: e.id, offset: None, up: None }, group: "Sections" });
        }
    }
    for e in &part.elements {
        if let ElementGeometry::Plane { .. } = e.geometry {
            out.push(PlaneOffer { key: format!("plane:{}", e.id), label: e.name.clone(), plane: PlaneRef::Plane { id: e.id, offset: None, up: None }, group: "Planes" });
        }
    }
    out.extend(coordinate_plane_offers(centre, at_zero));
    out
}

/// The key a plane reference is found by in the offers (ScanRuler's `planeKey`).
pub fn plane_key(r: &PlaneRef) -> String {
    match r {
        PlaneRef::World { axis, offset, .. } => world_plane_key(*axis, *offset),
        PlaneRef::Fitted { .. } => "fitted".into(),
        PlaneRef::BodyFace { .. } => "bodyFace".into(),
        PlaneRef::Radial { .. } => "radial".into(),
        PlaneRef::Section { id, .. } => format!("section:{id}"),
        PlaneRef::Plane { id, .. } => format!("plane:{id}"),
    }
}

/// The axes a plane can be laid through, at an angle about them (a radial plane): every line,
/// cylinder, cone and circle element, then the coordinate axes — each with its name.
pub fn radial_axis_offers(part: &Part) -> Vec<(RadialAxis, String)> {
    let mut out: Vec<(RadialAxis, String)> = part
        .elements
        .iter()
        .filter(|e| matches!(e.geometry, ElementGeometry::Line { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Circle { .. }))
        .map(|e| (RadialAxis::Element { id: e.id }, e.name.clone()))
        .collect();
    for (axis, name) in [(WorldAxis::X, "the X axis"), (WorldAxis::Y, "the Y axis"), (WorldAxis::Z, "the Z axis")] {
        out.push((RadialAxis::World { axis }, name.to_string()));
    }
    out
}

/// A plane fitted through a face of the scan, to sketch on (the reference keeps the frame and
/// no element): the sketch rule's frame on the fitted plane, its normal out of the face.
pub fn fitted_plane(fit: &PlaneFit) -> Option<PlaneRef> {
    let frame = SectionFrame::sketch_on(fit.center, fit.normal, None)?;
    Some(PlaneRef::Fitted { frame, label: None })
}

/// The frame a sketch's slice is cut on: a plane fitted on a face, or a body's face, a hair
/// inside it (`FACE_SLICE_INSET_MM`) — on the face itself the cut would run through the face's
/// own triangles; any other plane where it lies.
pub fn slice_frame(plane: &PlaneRef, frame: &SectionFrame) -> SectionFrame {
    match plane {
        PlaneRef::Fitted { .. } | PlaneRef::BodyFace { .. } => frame.offset(-crate::model::FACE_SLICE_INSET_MM),
        _ => *frame,
    }
}

// ---- drawing ----

fn circle_points(center: DVec3, axis: DVec3, radius: f64, n: usize) -> Vec<DVec3> {
    let (u, v) = ortho_basis(axis.try_normalize().unwrap_or(DVec3::Z));
    (0..=n).map(|i| {
        let a = std::f64::consts::TAU * i as f64 / n as f64;
        center + (u * a.cos() + v * a.sin()) * radius
    })
    .collect()
}

/// The lines an element is drawn with on the part, in part coordinates: a plane's patch outline,
/// a cylinder's or a cone's end circles and four of its lines, a sphere's three great circles, a
/// torus's spine with its outer and inner rims and two tube circles, a circle, a line's segment,
/// a point's three small crossing strokes, a section's square (`size` across each way of its
/// centre). `size` — the part's radius — sizes what has no size of its own.
pub fn element_wires(g: &ElementGeometry, size: f64) -> Vec<Vec<DVec3>> {
    const N: usize = 48;
    match *g {
        ElementGeometry::Point { center } => {
            let s = (size * 0.02).max(0.2);
            [DVec3::X, DVec3::Y, DVec3::Z].iter().map(|d| vec![center - *d * s, center + *d * s]).collect()
        }
        ElementGeometry::Line { center, dir, length } => {
            let d = dir.try_normalize().unwrap_or(DVec3::Z) * (length.max(size * 0.05) / 2.0);
            vec![vec![center - d, center + d]]
        }
        ElementGeometry::Circle { center, normal, radius } => vec![circle_points(center, normal, radius, N)],
        ElementGeometry::Plane { center, basis_u, basis_v, extent_u, extent_v, .. } => {
            let (u, v) = (basis_u * extent_u, basis_v * extent_v);
            vec![vec![center - u - v, center + u - v, center + u + v, center - u + v, center - u - v]]
        }
        ElementGeometry::Section { frame } => {
            let (u, v) = (frame.u * size, frame.v * size);
            let c = frame.origin;
            vec![vec![c - u - v, c + u - v, c + u + v, c - u + v, c - u - v]]
        }
        ElementGeometry::Cylinder { center, axis, radius, length, .. } => tube_wires(center, axis, radius, radius, length),
        ElementGeometry::Cone { center, axis, radius1, radius2, length, .. } => tube_wires(center, axis, radius1, radius2, length),
        ElementGeometry::Sphere { center, radius } => [DVec3::X, DVec3::Y, DVec3::Z].iter().map(|a| circle_points(center, *a, radius, N)).collect(),
        ElementGeometry::Torus { center, axis, major_radius, minor_radius, .. } => {
            let a = axis.try_normalize().unwrap_or(DVec3::Z);
            let (u, _) = ortho_basis(a);
            let mut out = vec![circle_points(center, a, major_radius, N), circle_points(center, a, major_radius + minor_radius, N), circle_points(center, a, (major_radius - minor_radius).max(0.0), N)];
            for side in [1.0, -1.0] {
                let c = center + u * major_radius * side;
                out.push(circle_points(c, a.cross(u), minor_radius, N / 2));
            }
            out
        }
    }
}

fn tube_wires(center: DVec3, axis: DVec3, r_low: f64, r_high: f64, length: f64) -> Vec<Vec<DVec3>> {
    let a = axis.try_normalize().unwrap_or(DVec3::Z);
    let (u, v) = ortho_basis(a);
    let (lo, hi) = (center - a * (length / 2.0), center + a * (length / 2.0));
    let mut out = vec![circle_points(lo, a, r_low, 48), circle_points(hi, a, r_high, 48)];
    for k in 0..4 {
        let t = std::f64::consts::FRAC_PI_2 * k as f64;
        let d = u * t.cos() + v * t.sin();
        out.push(vec![lo + d * r_low, hi + d * r_high]);
    }
    out
}

/// The surface an element is drawn with, see-through, in part coordinates: a plane's patch, a
/// section's square, a cylinder's or a cone's wall, a sphere, a torus. Lines, circles and points
/// have none.
pub fn element_surface(g: &ElementGeometry, size: f64) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let quad = |c: DVec3, u: DVec3, v: DVec3| (vec![c - u - v, c + u - v, c + u + v, c - u + v], vec![[0u32, 1, 2], [0, 2, 3]]);
    match *g {
        ElementGeometry::Plane { center, basis_u, basis_v, extent_u, extent_v, .. } => quad(center, basis_u * extent_u, basis_v * extent_v),
        ElementGeometry::Section { frame } => quad(frame.origin, frame.u * size, frame.v * size),
        ElementGeometry::Cylinder { center, axis, radius, length, .. } => revolved(center, axis, |t| (t * length - length / 2.0, radius), 2, 40),
        ElementGeometry::Cone { center, axis, radius1, radius2, length, .. } => revolved(center, axis, |t| (t * length - length / 2.0, radius1 + (radius2 - radius1) * t), 2, 40),
        ElementGeometry::Sphere { center, radius } => revolved(center, DVec3::Z, |t| {
            let a = std::f64::consts::PI * (t - 0.5);
            (radius * a.sin(), radius * a.cos())
        }, 17, 32),
        ElementGeometry::Torus { center, axis, major_radius, minor_radius, .. } => {
            let a = axis.try_normalize().unwrap_or(DVec3::Z);
            let (u, v) = ortho_basis(a);
            let (nu, nv) = (40usize, 16usize);
            let mut pts = Vec::with_capacity(nu * nv);
            for i in 0..nu {
                let p = std::f64::consts::TAU * i as f64 / nu as f64;
                let radial = u * p.cos() + v * p.sin();
                for j in 0..nv {
                    let q = std::f64::consts::TAU * j as f64 / nv as f64;
                    pts.push(center + radial * (major_radius + minor_radius * q.cos()) + a * (minor_radius * q.sin()));
                }
            }
            let mut tris = Vec::new();
            for i in 0..nu {
                for j in 0..nv {
                    let (i1, j1) = ((i + 1) % nu, (j + 1) % nv);
                    let (a0, b0, c0, d0) = ((i * nv + j) as u32, (i1 * nv + j) as u32, (i1 * nv + j1) as u32, (i * nv + j1) as u32);
                    tris.push([a0, b0, c0]);
                    tris.push([a0, c0, d0]);
                }
            }
            (pts, tris)
        }
        _ => (Vec::new(), Vec::new()),
    }
}

/// A surface of revolution about `axis` through `center`: `profile(t)` gives the height along the
/// axis and the radius for t from 0 to 1 over `rows` rings of `cols` points.
fn revolved(center: DVec3, axis: DVec3, profile: impl Fn(f64) -> (f64, f64), rows: usize, cols: usize) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let a = axis.try_normalize().unwrap_or(DVec3::Z);
    let (u, v) = ortho_basis(a);
    let mut pts = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let (h, rad) = profile(r as f64 / (rows - 1).max(1) as f64);
        for c in 0..cols {
            let p = std::f64::consts::TAU * c as f64 / cols as f64;
            pts.push(center + a * h + (u * p.cos() + v * p.sin()) * rad);
        }
    }
    let mut tris = Vec::new();
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols {
            let c1 = (c + 1) % cols;
            let (a0, b0, c0, d0) = ((r * cols + c) as u32, (r * cols + c1) as u32, ((r + 1) * cols + c1) as u32, ((r + 1) * cols + c) as u32);
            tris.push([a0, b0, c0]);
            tris.push([a0, c0, d0]);
        }
    }
    (pts, tris)
}

/// The point an element's name stands at, and where a click finds it nearest: its centre (a
/// section's origin).
pub fn element_anchor(g: &ElementGeometry) -> DVec3 {
    match *g {
        ElementGeometry::Point { center }
        | ElementGeometry::Line { center, .. }
        | ElementGeometry::Circle { center, .. }
        | ElementGeometry::Plane { center, .. }
        | ElementGeometry::Cylinder { center, .. }
        | ElementGeometry::Cone { center, .. }
        | ElementGeometry::Sphere { center, .. }
        | ElementGeometry::Torus { center, .. } => center,
        ElementGeometry::Section { frame } => frame.origin,
    }
}

#[cfg(test)]
mod tests;
