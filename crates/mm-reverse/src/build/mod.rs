// SPDX-License-Identifier: AGPL-3.0-only
//! The build: the features played in order into bodies, on the SolidCraft kernel. The part's
//! bookkeeping follows ScanRuler (`plugins/reverse/core/build.ts`, `buildExtrude.ts`,
//! `buildTypes.ts`, `resolve.ts`, `primitives.ts`, `move.ts`, `kernelOps.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only, with Open CASCADE's calls replaced by SolidCraft's
//! (`solidcraft_kernel`, MIT OR Apache-2.0).
//!
//! What the bodies do follows ScanRuler: a join is made one with every solid body it touches
//! (a body of its own where it touches none, one body of all of them where it bridges several,
//! under the oldest one's id); a cut goes through every solid body it meets and fails, saying
//! so, when it meets none; a new body keeps the solid apart whatever it overlaps. One body is
//! one solid: what a step leaves in pieces is a body a piece, the largest keeping the id, each
//! other a further id of the step's (see [`BodyId`]). A feature that fails is reported by name
//! and the rest still build.
//!
//! Every face of a body carries the id of the feature that first made it, through every step,
//! by the kernel's history where the operation reports one and by where the face lies where it
//! does not — so a fingerprint (`topo.rs`) says who made the face or edge it names, and a pick
//! is found again on the body as it stands before the feature that uses it.
//!
//! Each feature kind is built in a file of its own (`build/<kind>.rs`); this module holds what
//! they share: the bodies and their ids, placing a solid into them, finding picks, resolving
//! planes, axes and elements into geometry, reading a body's faces and edges, meshing and STEP.

mod boolean;
mod chamfer;
mod cut_plane;
mod delete_face;
mod draft;
mod extract;
mod extrude;
mod fillet;
mod hole;
mod import;
mod loft;
mod mirror;
mod move_body;
mod move_face;
mod network;
mod offset_body;
mod pattern;
mod revolve;
mod sheet;
mod shell;
mod sketch;
mod split;
mod surface;
mod sweep;
mod thicken;
mod thread;
mod trim;
// ---- BUILD ---- (the engine: incremental and parallel builds, stops, words of splits and
// cuts, the boolean guard, failure diagnosis)
mod cut_elsewhere;
mod diagnose;
mod engine;
pub mod every_face;
mod groups;
pub mod guard;
pub mod split_words;
pub use engine::{step_keys, BuildAsk, BuildControl, BuildSteps, Engine, EngineBuild, InPlay, StepKey, CACHE_BYTES, KERNEL_VERSION};
pub use groups::{group_steps, Grouped, StepGroup};
pub use split_words::{box_words, plane_words, side_of_box, side_words, split_words, takes_all_words, PieceSaid, Side};
pub use cut_elsewhere::{cut_elsewhere_words, cuts_elsewhere, CutElsewhere};
use engine::{Tracked, Tracking};
// ---- SURF ----
mod surf;
mod surf_kernel;
mod surf_split;
pub use surf::{build_before_scanned, build_scanned, BuildScan, OffScanReport, SurfReport, SurfaceFitReport, TrimReport};
pub use surface::{constrain_refs, fit_params, SURFACE_MIN_REGION};
pub use thicken::thicken_way;

use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::{DVec2, DVec3};
use solidcraft_geom::{Loop2, Plane, Region2, Vec2 as SVec2, Vec3 as SVec3};
use solidcraft_kernel::history::FaceOrigin;
use solidcraft_kernel::{self as kernel, BoolOp, Body, FaceSurface};

use crate::frame::{radial_frame, SectionFrame, WorldAxis};
use crate::model::{AxisRef, BodyId, BodyOp, EdgeRef, Element, ElementGeometry, FaceRef, Feature, FeatureId, FeatureKind, Part, PlaneRef, Primitive, RegionAsk, RegionGiven, RegionPick, RegionWord, SketchFeature, ISLAND_IDS};
use crate::pick_query::{live_picks, run_stored_where, sharp_by_surfaces, where_words, StoredPick, What};
use crate::regions::{region_at, Region};
use crate::topo::{canonical_dir, describe_edge, describe_face, edge_distance, face_distance, match_edge, match_face, match_tolerance, EdgeFingerprint, EdgeInfo, EdgeKind, EdgeSide, FaceFingerprint, FaceInfo, FaceKind, MatchTolerance, Topology};

/// The kernel's solid, for callers that hold bodies.
pub use solidcraft_kernel::Body as KernelBody;
// ---- NET ----
/// What a build reads for its patch networks — the scan, the open box's feature — and what
/// each says it came to (`network.rs`).
pub use network::{reading_feature, NetworkContext, NetworkOutcome, NetworkScope};

// ---- small conversions ----

pub fn sv3(p: DVec3) -> SVec3 {
    SVec3::new(p.x, p.y, p.z)
}

pub fn dv3(p: SVec3) -> DVec3 {
    DVec3::new(p.x, p.y, p.z)
}

pub(crate) fn sv2(p: DVec2) -> SVec2 {
    SVec2::new(p.x, p.y)
}

pub(crate) fn kerr(e: kernel::KernelError) -> String {
    e.to_string()
}

/// A frame as the kernel's plane.
pub fn kernel_plane(frame: &SectionFrame) -> Result<Plane, String> {
    Plane::new(sv3(frame.origin), sv3(frame.u), sv3(frame.v)).ok_or_else(|| "The sketch plane has no direction.".to_string())
}

/// A triangle soup welded: points closer than `tol` (relative to the soup's size) made one,
/// triangles that collapse dropped.
pub fn welded(positions: &[DVec3], triangles: &[[u32; 3]], tol: f64) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let size = positions.iter().fold(0.0f64, |m, p| m.max(p.abs().max_element())).max(1.0);
    let q = (tol * size).max(1e-12);
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut out = Vec::new();
    let index: Vec<u32> = positions
        .iter()
        .map(|p| {
            let key = ((p.x / q).round() as i64, (p.y / q).round() as i64, (p.z / q).round() as i64);
            *map.entry(key).or_insert_with(|| {
                out.push(*p);
                (out.len() - 1) as u32
            })
        })
        .collect();
    let tris = triangles
        .iter()
        .filter(|t| t.iter().all(|v| (*v as usize) < index.len()))
        .map(|t| [index[t[0] as usize], index[t[1] as usize], index[t[2] as usize]])
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0])
        .collect();
    (out, tris)
}

/// The mirror of a point set about a plane, as a column-major 4×4.
pub fn mirror_matrix(origin: DVec3, normal: DVec3) -> [[f64; 4]; 4] {
    let n = normal.normalize_or_zero();
    let d = 2.0 * origin.dot(n);
    let mut m = [[0.0; 4]; 4];
    for c in 0..3 {
        for r in 0..3 {
            let identity = if r == c { 1.0 } else { 0.0 };
            m[c][r] = identity - 2.0 * n[r] * n[c];
        }
    }
    m[3] = [d * n.x, d * n.y, d * n.z, 1.0];
    m
}

/// A turn of `angle` radians about the axis through `origin` along `dir`, as a column-major
/// 4×4.
pub fn rotation_matrix(origin: DVec3, dir: DVec3, angle: f64) -> [[f64; 4]; 4] {
    let r = glam::DMat3::from_axis_angle(dir.normalize_or_zero(), angle);
    let t = origin - r * origin;
    [[r.x_axis.x, r.x_axis.y, r.x_axis.z, 0.0], [r.y_axis.x, r.y_axis.y, r.y_axis.z, 0.0], [r.z_axis.x, r.z_axis.y, r.z_axis.z, 0.0], [t.x, t.y, t.z, 1.0]]
}

/// A carry by `t`, as a column-major 4×4.
pub fn translation_matrix(t: DVec3) -> [[f64; 4]; 4] {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [t.x, t.y, t.z, 1.0]]
}

/// A Move's turn of three angles in degrees — about X first, then Y, then Z, each about the
/// part's own axis: Rz · Ry · Rx (ScanRuler's `move.ts` rotationOf).
pub fn move_rotation(rotate: DVec3) -> glam::DMat3 {
    glam::DMat3::from_rotation_z(rotate.z.to_radians()) * glam::DMat3::from_rotation_y(rotate.y.to_radians()) * glam::DMat3::from_rotation_x(rotate.x.to_radians())
}

/// The whole of a Move, as a column-major 4×4: p′ = R (p − pivot) + pivot + translate
/// (ScanRuler's `move.ts` moveMatrix).
pub fn move_matrix(pivot: DVec3, translate: DVec3, rotate: DVec3) -> [[f64; 4]; 4] {
    let r = move_rotation(rotate);
    let t = pivot + translate - r * pivot;
    [[r.x_axis.x, r.x_axis.y, r.x_axis.z, 0.0], [r.y_axis.x, r.y_axis.y, r.y_axis.z, 0.0], [r.z_axis.x, r.z_axis.y, r.z_axis.z, 0.0], [t.x, t.y, t.z, 1.0]]
}

/// A point moved by a column-major 4×4.
pub fn apply_matrix(m: &[[f64; 4]; 4], p: DVec3) -> DVec3 {
    DVec3::new(
        m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0],
        m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1],
        m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2],
    )
}

/// A body's size: the diagonal of its box.
pub fn diagonal_of(body: &Body) -> f64 {
    let bb = body.outline_bounds();
    if bb.is_empty() {
        return 0.0;
    }
    (dv3(bb.max) - dv3(bb.min)).length()
}

/// The chord tolerance a body's faces and edges are read at, off its size (ScanRuler's
/// `meshDeflection`): capped at 0.05 mm.
pub fn mesh_deflection(diagonal: f64) -> f64 {
    (diagonal * 5e-4).clamp(0.005, 0.05)
}

/// A body's volume, mm³.
pub fn volume(body: &Body) -> Option<f64> {
    kernel::measure(body).ok().map(|m| m.volume)
}

// ---- what a build hands back ----

/// A body the build made.
#[derive(Clone)]
pub struct BuiltBody {
    pub id: BodyId,
    /// As the body list names it: Solid n (Shell n for a surface), in the order of the
    /// features that made the bodies, or the name given it (see [`name_bodies`]).
    pub name: String,
    pub body: Body,
    /// A sheet rather than a solid.
    pub surface: bool,
    /// The name a body came with from a STEP file.
    pub label: Option<String>,
    /// Every face and edge with its fingerprint, for the viewport's picks; a face's and an
    /// edge's hash is its index in the kernel's order (the display mesh's too).
    pub topology: Arc<Topology>,
}

/// What a sketch is as the build reads it: its frame and its closed regions.
#[derive(Clone)]
pub struct BuiltSketch {
    pub frame: SectionFrame,
    pub regions: Vec<Region>,
    /// The sketch as placed, its projected pieces taken again from their sources, when that
    /// differs from the one stored (see `crate::project::place_projections`).
    pub placed: Option<crate::sketch::Sketch>,
}

/// The part as built.
#[derive(Clone, Default)]
pub struct Built {
    pub bodies: Vec<BuiltBody>,
    pub sketches: HashMap<FeatureId, BuiltSketch>,
    /// The features that failed, and why.
    pub errors: HashMap<FeatureId, String>,
    /// What a feature that built did its own way, or a pick found again only loosely.
    pub warnings: HashMap<FeatureId, Vec<String>>,
    /// The bodies each feature joined, cut, made or worked on, as they came out of it.
    pub touched: HashMap<FeatureId, Vec<BodyId>>,
    // ---- DRAFT ----
    /// Where a feature that failed failed, for the part to show red: the outlines of the faces
    /// its diagnosis names (a draft's faces that will not lean).
    pub failed_where: HashMap<FeatureId, Vec<Vec<DVec3>>>,
    // ---- EXT ----
    /// The volume of the solid an extrude or a revolve made before it was placed into the
    /// bodies: what its box's preview line reads as the ghost's.
    pub ghost_volumes: HashMap<FeatureId, f64>,
    // ---- BUILD ----
    /// Where a failed feature failed, as its diagnosis found it: the edges or outlines to draw
    /// red on the part (ScanRuler's `where`).
    pub failed_at: HashMap<FeatureId, Vec<Vec<DVec3>>>,
    // ---- SURF ----
    /// What the surface features said of themselves, built or failed (`surf::SurfReport`).
    pub surf: HashMap<FeatureId, SurfReport>,
}

impl Built {
    pub fn body(&self, id: BodyId) -> Option<&BuiltBody> {
        self.bodies.iter().find(|b| b.id == id)
    }
}

/// The bodies named as a CAD body list names them (ScanRuler's `nameBodies`): Solid 1, Solid 2
/// for the closed ones, Shell 1 for a surface — counted in the order of the features that made
/// them, then of their islands — a body from a STEP file by its part's name (the second of one
/// name by the name and a count), and a name given in the part's body styles over all of them.
pub fn name_bodies(bodies: &mut [BuiltBody], part: &Part) {
    let order: HashMap<FeatureId, usize> = part.features.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
    let at = |id: BodyId| order.get(&id.maker()).copied().unwrap_or(usize::MAX);
    let mut sorted: Vec<usize> = (0..bodies.len()).collect();
    sorted.sort_by_key(|i| (at(bodies[*i].id), bodies[*i].id.island));
    let (mut solids, mut shells) = (0, 0);
    let mut labelled: HashMap<String, usize> = HashMap::new();
    let mut names: HashMap<BodyId, String> = HashMap::new();
    for i in sorted {
        let b = &bodies[i];
        let counted = if b.surface {
            shells += 1;
            format!("Shell {shells}")
        } else {
            solids += 1;
            format!("Solid {solids}")
        };
        let name = match &b.label {
            Some(l) => {
                let n = labelled.entry(l.clone()).or_insert(0);
                *n += 1;
                if *n == 1 {
                    l.clone()
                } else {
                    format!("{l} ({n})")
                }
            }
            None => counted,
        };
        names.insert(b.id, name);
    }
    for b in bodies.iter_mut() {
        let given = part.body_styles.get(&b.id).and_then(|s| s.name.clone()).filter(|n| !n.is_empty());
        b.name = given.or_else(|| names.get(&b.id).cloned()).unwrap_or_else(|| b.id.to_string());
    }
}

// ---- the builder ----

/// A body in the builder: its shape, which feature first made each of its faces (by face
/// index, -1 where none is known), and its faces and edges read once.
#[derive(Clone)]
pub(crate) struct BodyState {
    pub(crate) shape: Body,
    pub(crate) origins: Vec<i64>,
    pub(crate) surface: bool,
    pub(crate) label: Option<String>,
    topology: std::sync::OnceLock<Arc<Topology>>,
}

impl BodyState {
    pub(crate) fn new(shape: Body, origins: Vec<i64>) -> BodyState {
        BodyState { shape, origins, surface: false, label: None, topology: std::sync::OnceLock::new() }
    }

    /// Every face and edge of the body with its fingerprint.
    pub(crate) fn topology(&self) -> Arc<Topology> {
        self.topology.get_or_init(|| Arc::new(read_topology(&self.shape, &self.origins))).clone()
    }

    pub(crate) fn diagonal(&self) -> f64 {
        diagonal_of(&self.shape)
    }
}

/// The state of a build as it plays the features: the bodies standing, in the order they were
/// made, the ids given out, and what it says of each feature.
#[derive(Clone)]
pub(crate) struct Builder {
    /// Each body shared with the states the engine keeps; a step puts a new one in place of the
    /// one it changes, never changes one in place (`body_state_mut` copies one shared).
    bodies: Tracked<Vec<(BodyId, Arc<BodyState>)>>,
    /// A body gone into another (a join that bridged two), and the one it went into.
    aliases: Tracked<HashMap<BodyId, BodyId>>,
    issued: Tracked<HashSet<BodyId>>,
    /// The further bodies a step split off a body, by the body's id (ScanRuler's `pieces`): a
    /// Combine naming the body takes them with it (`family_of`).
    pieces: Tracked<HashMap<BodyId, Vec<BodyId>>>,
    pub(crate) out: Built,
    /// The feature being built.
    pub(crate) step: FeatureId,
    warnings: Vec<String>,
    // ---- HOLE ----
    /// The threads whose faceted cut waits (see `thread::flush`); kept with the engine's states
    /// too, so a build taken up after a thread still cuts it.
    pub(crate) threads: Vec<thread::PendingCut>,
    // ---- BUILD ----
    /// What the engine keeps beside the maps: what the step playing read, the sketches'
    /// marks, the features' names, how a split is said.
    pub(crate) track: Tracking,
    // ---- SURF ----
    /// The scan and what the surface features keep beside the bodies.
    pub(crate) surf: surf::SurfState,
}

/// A solid in pieces whose largest pieces are within this share of each other's volume are
/// told apart by where the body stood, not by the kernel's order.
const SPLIT_TIE: f64 = 0.01;

impl Builder {
    pub(crate) fn new() -> Builder {
        Builder { bodies: Tracked::new(Vec::new()), aliases: Tracked::new(HashMap::new()), issued: Tracked::new(HashSet::new()), pieces: Tracked::new(HashMap::new()), out: Built::default(), step: 0, warnings: Vec::new(), threads: Vec::new(), track: Tracking::default(), surf: surf::SurfState::default() }
    }

    /// Something the feature being built did its own way, for its row.
    pub(crate) fn warn(&mut self, s: impl Into<String>) {
        self.warnings.push(s.into());
    }

    /// The bodies a feature joined, cut, made or worked on, as they came out of it.
    pub(crate) fn touched(&mut self, ids: Vec<BodyId>) {
        self.out.touched.insert(self.step, ids);
    }

    /// A body's id as it stands now: one gone into another names the other.
    pub(crate) fn resolve(&self, id: BodyId) -> BodyId {
        let mut at = id;
        for _ in 0..64 {
            self.track.note_alias(at);
            match self.aliases.raw.get(&at) {
                Some(next) if *next != at => at = *next,
                _ => break,
            }
        }
        at
    }

    /// A body standing, by id.
    pub(crate) fn body(&self, id: BodyId) -> Option<&BodyState> {
        let id = self.resolve(id);
        self.track.note_body(id);
        self.bodies.raw.iter().find(|(b, _)| *b == id).map(|(_, s)| &**s)
    }

    /// The faces and edges of a body standing, read once and kept with the body — what its
    /// picks are found on and what a feature on it reads them by, the same reading for both.
    pub(crate) fn topology_of(&self, id: BodyId) -> Option<Arc<Topology>> {
        self.body(id).map(BodyState::topology)
    }

    /// The body a feature works on, as it stands before the feature, with its id as it stands.
    pub(crate) fn body_for(&self, id: BodyId) -> Result<(BodyId, BodyState), String> {
        let at = self.resolve(id);
        self.body(at).map(|s| (at, s.clone())).ok_or_else(|| "The body this feature works on is gone, or failed to build.".to_string())
    }

    /// The ids of the bodies standing, in the order they were made.
    pub(crate) fn body_ids(&self) -> Vec<BodyId> {
        self.track.note_bodies_all();
        self.bodies.raw.iter().map(|(id, _)| *id).collect()
    }

    /// A body put in place: in the place of the one with its id, or at the end.
    pub(crate) fn set_body(&mut self, id: BodyId, state: BodyState) {
        self.issued.raw.insert(id);
        let state = Arc::new(state);
        match self.bodies.raw.iter_mut().find(|(b, _)| *b == id) {
            Some(slot) => slot.1 = state,
            None => self.bodies.raw.push((id, state)),
        }
    }

    /// A body standing, by id, to change in place — a copy of it where a kept state shares it.
    pub(crate) fn body_state_mut(&mut self, id: BodyId) -> Option<&mut BodyState> {
        let id = self.resolve(id);
        self.track.note_body(id);
        self.bodies.raw.iter_mut().find(|(b, _)| *b == id).map(|(_, s)| Arc::make_mut(s))
    }

    pub(crate) fn remove_body(&mut self, id: BodyId) {
        self.bodies.raw.retain(|(b, _)| *b != id);
    }

    /// A body gone into another: a project that names it finds the other.
    pub(crate) fn alias(&mut self, gone: BodyId, into: BodyId) {
        if gone != into {
            self.aliases.raw.insert(gone, into);
        }
    }

    /// A body id for the step's next body: the step's own, then its own and a fraction.
    pub(crate) fn fresh_id(&mut self) -> Result<BodyId, String> {
        for n in 0..ISLAND_IDS {
            let id = BodyId { feature: self.step, island: n };
            self.track.note_issued(id);
            self.track.note_body(id);
            if self.issued.raw.contains(&id) || self.bodies.raw.iter().any(|(b, _)| *b == id) {
                continue;
            }
            self.issued.raw.insert(id);
            return Ok(id);
        }
        Err(format!("The feature leaves more than {ISLAND_IDS} separate bodies."))
    }

    /// The faces of a new solid, every one made by the step.
    pub(crate) fn own_origins(&self, body: &Body) -> Vec<i64> {
        vec![self.step as i64; body.face_count()]
    }

    /// A shape into the body `id`, as many bodies as it has solids: the largest keeps the id,
    /// each other piece a fresh one of the step's. Pieces the same size go by where the body
    /// stood (`by_place`): the one nearest it keeps the id. The ids the pieces went to, `id`
    /// first.
    pub(crate) fn settle(&mut self, id: BodyId, shape: Body, origins: Vec<i64>, by_place: bool) -> Result<Vec<BodyId>, String> {
        let label = self.body(id).and_then(|b| b.label.clone());
        let was_centroid = if by_place { self.body(id).and_then(|b| kernel::measure(&b.shape).ok()).map(|m| dv3(m.centroid)) } else { None };
        let mut pieces = islands(&shape);
        if pieces.len() <= 1 {
            let mut state = BodyState::new(shape, origins);
            state.label = label;
            self.set_body(id, state);
            return Ok(vec![id]);
        }
        if let Some(at) = was_centroid {
            let tied = pieces.iter().filter(|p| p.1 >= pieces[0].1 * (1.0 - SPLIT_TIE)).count();
            if tied >= 2 {
                let far: Vec<f64> = pieces[..tied].iter().map(|p| kernel::measure(&p.0).map(|m| (dv3(m.centroid) - at).length()).unwrap_or(f64::INFINITY)).collect();
                let mut best = 0;
                for i in 1..tied {
                    if far[i] < far[best] - 1e-6 * far[best].max(1.0) {
                        best = i;
                    }
                }
                if best > 0 {
                    let p = pieces.remove(best);
                    pieces.insert(0, p);
                }
            }
        }
        let mut ids = Vec::new();
        for (i, (piece, _)) in pieces.into_iter().enumerate() {
            let at = if i == 0 { id } else { self.fresh_id()? };
            let own = origins_after(&piece, &[(&shape, &origins)], self.step);
            let mut state = BodyState::new(piece, own);
            state.label = label.clone();
            self.set_body(at, state);
            ids.push(at);
        }
        // The pieces kept with the body (BOOL, BUILD), and a cut's split said on its row (BUILD).
        self.note_pieces(id, &ids[1..]);
        if by_place {
            self.say_split(&ids);
        }
        Ok(ids)
    }

    /// The solid a feature makes into the bodies: a body of its own (a body a piece), joined to
    /// every solid body it meets or cut out of them — or, for a feature saved with a target,
    /// joined to or cut from that body alone. The ids of the bodies it went into.
    pub(crate) fn place(&mut self, op: BodyOp, target: Option<BodyId>, solids: Vec<Body>) -> Result<Vec<BodyId>, String> {
        let ids = match (op, target) {
            (BodyOp::New, _) => {
                let mut ids = Vec::new();
                for s in solids {
                    let id = self.fresh_id()?;
                    let own = self.own_origins(&s);
                    ids.extend(self.settle(id, s, own, false)?);
                }
                if ids.len() > 1 {
                    let list: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
                    self.warn(format!("The feature made {} separate solids — {}. A Combine on it takes them all; a feature on one body works on the one it names.", ids.len(), list.join(", ")));
                }
                ids
            }
            (_, Some(t)) => self.place_on_target(op, t, solids)?,
            (BodyOp::Join, None) => {
                let mut ids = Vec::new();
                for s in solids {
                    for id in self.join_all(s)? {
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
                ids.retain(|id| self.body(*id).is_some());
                ids
            }
            (BodyOp::Cut, None) => self.cut_all(&solids)?,
        };
        if target.is_none() && op != BodyOp::Cut {
            self.note_made_apart(&ids);
        }
        self.touched(ids.clone());
        Ok(ids)
    }

    /// Join a solid to every solid body it meets: the bodies it bridges and the solid made one,
    /// a body of its own where it meets none. Each piece of what comes out keeps the id of the
    /// oldest body in it, or takes a fresh one of the step's; a body gone into another is
    /// aliased to it.
    fn join_all(&mut self, solid: Body) -> Result<Vec<BodyId>, String> {
        self.track.note_bodies_all();
        let near: Vec<BodyId> = self.bodies.raw.iter().filter(|(_, b)| !b.surface && b.shape.meets(&solid).unwrap_or(false)).map(|(id, _)| *id).collect();
        if near.is_empty() {
            let id = self.fresh_id()?;
            let own = self.own_origins(&solid);
            return self.settle(id, solid, own, false);
        }
        let mut acc = solid.clone();
        let mut acc_origins = self.own_origins(&solid);
        for id in &near {
            let Some(b) = self.body(*id).cloned() else { continue };
            let joined = kernel::boolean(&acc, &b.shape, BoolOp::Union).map_err(kerr)?.ok_or("The join produced nothing.")?;
            acc_origins = origins_after(&joined, &[(&acc, &acc_origins), (&b.shape, &b.origins)], self.step);
            acc = joined;
        }
        // ---- BUILD ---- never less than any part, never more than all of them together.
        let parts: Vec<f64> = std::iter::once(&solid).chain(near.iter().filter_map(|id| self.body(*id).map(|b| &b.shape))).map(|s| volume(s).unwrap_or(0.0)).collect();
        guard::guarded("Joining", &guard::GuardAsk::Join { parts, result: volume(&acc).unwrap_or(0.0) })?;
        let pieces = islands(&acc);
        // Which bodies went into which piece, by a point just inside each.
        let probes: Vec<(BodyId, Option<DVec3>)> = near.iter().map(|id| (*id, self.body(*id).and_then(|b| probe_inside(&b.shape)))).collect();
        let tool_probe = probe_inside(&solid);
        let mut claimed: HashSet<BodyId> = HashSet::new();
        let mut ids = Vec::new();
        for (piece, _) in &pieces {
            let of: Vec<BodyId> = probes.iter().filter(|(id, p)| !claimed.contains(id) && p.is_some_and(|p| piece.contains(sv3(p)))).map(|(id, _)| *id).collect();
            for id in &of {
                claimed.insert(*id);
            }
            let has_tool = tool_probe.is_some_and(|p| piece.contains(sv3(p)));
            // A body the new solid does not reach, standing in a piece of its own: as it was.
            if !has_tool && of.len() == 1 && pieces.len() > 1 {
                continue;
            }
            let id = match of.first() {
                Some(id) => *id,
                None => self.fresh_id()?,
            };
            for gone in &of {
                if gone != &id {
                    self.remove_body(*gone);
                    self.alias(*gone, id);
                }
            }
            let own = if pieces.len() == 1 { acc_origins.clone() } else { origins_after(piece, &[(&acc, &acc_origins)], self.step) };
            let label = self.body(id).and_then(|b| b.label.clone());
            let mut state = BodyState::new(piece.clone(), own);
            state.label = label;
            self.set_body(id, state);
            ids.push(id);
        }
        // A body no piece claimed went into the first piece.
        for id in &near {
            if claimed.contains(id) {
                continue;
            }
            if let Some(first) = ids.first().copied() {
                if first != *id {
                    self.remove_body(*id);
                    self.alias(*id, first);
                }
            }
        }
        Ok(ids)
    }

    /// Cut solids out of every solid body they meet: a body taken whole is gone, one parted is a
    /// body a piece. A cut that meets no body is an error, as a CAD cut with nothing to cut is.
    fn cut_all(&mut self, solids: &[Body]) -> Result<Vec<BodyId>, String> {
        let mut any = false;
        let mut ids = Vec::new();
        for id in self.body_ids() {
            let Some(b) = self.body(id).cloned() else { continue };
            if b.surface {
                continue;
            }
            let mut acc = Some(b.shape.clone());
            let mut origins = b.origins.clone();
            let mut cut = false;
            for tool in solids {
                let Some(a) = acc.clone() else { break };
                if !a.meets(tool).unwrap_or(false) {
                    continue;
                }
                let was = volume(&a).unwrap_or(0.0);
                let r = kernel::boolean(&a, tool, BoolOp::Cut).map_err(kerr)?;
                match r {
                    Some(r) => {
                        let left = volume(&r).unwrap_or(0.0);
                        // A body the solid only touches is not cut.
                        if (was - left).abs() <= 1e-7 * was.max(1.0) {
                            continue;
                        }
                        // ---- BUILD ---- never more than the body was.
                        if let Some(why) = guard::volume_refusal(&guard::GuardAsk::Cut { body: was, result: left }) {
                            return Err(format!("The cut of {} cannot stand: {why}.", self.body_words(id)));
                        }
                        let tool_origins = self.own_origins(tool);
                        origins = origins_after(&r, &[(&a, &origins), (tool, &tool_origins)], self.step);
                        acc = Some(r);
                    }
                    None => acc = None,
                }
                cut = true;
            }
            if !cut {
                continue;
            }
            any = true;
            match acc {
                Some(shape) if shape.face_count() > 0 => ids.extend(self.settle(id, shape, origins, true)?),
                _ => self.remove_body(id),
            }
        }
        if !any {
            return Err("The cut meets no body — there is nothing for it to take out.".into());
        }
        Ok(ids)
    }

    /// A join or a cut on the one body a feature saved with a target names.
    fn place_on_target(&mut self, op: BodyOp, target: BodyId, solids: Vec<Body>) -> Result<Vec<BodyId>, String> {
        let (id, b) = self.body_for(target).map_err(|_| "The body to join or cut is gone, or failed to build.".to_string())?;
        if b.surface {
            return Err(format!("{id} is a surface, not a solid — a solid cannot {} it. Set the result to New body.", if op == BodyOp::Join { "join" } else { "cut" }));
        }
        let mut acc = b.shape.clone();
        let mut origins = b.origins.clone();
        for s in &solids {
            let bop = if op == BodyOp::Join { BoolOp::Union } else { BoolOp::Cut };
            let r = kernel::boolean(&acc, s, bop).map_err(kerr)?.ok_or("The cut left no closed solid (did it take the whole body?)")?;
            // ---- BUILD ---- held to its volume.
            let (before, tool, after) = (volume(&acc).unwrap_or(0.0), volume(s).unwrap_or(0.0), volume(&r).unwrap_or(0.0));
            if op == BodyOp::Join {
                guard::guarded("Joining", &guard::GuardAsk::Join { parts: vec![before, tool], result: after })?;
            } else {
                guard::guarded("The cut", &guard::GuardAsk::Cut { body: before, result: after })?;
            }
            let own = self.own_origins(s);
            origins = origins_after(&r, &[(&acc, &origins), (s, &own)], self.step);
            acc = r;
        }
        self.settle(id, acc, origins, op == BodyOp::Cut)
    }

    // ---- picks ----

    /// The faces of a body that fingerprints name, by index — or the first fingerprint that
    /// names none, described, for the feature to fail with. Each face is taken once. The
    /// queries the picks were given by (`asked`) are run again first, on the body as it
    /// stands; one that names nothing falls back on the fingerprints of what it found.
    pub(crate) fn find_faces(&mut self, body: BodyId, wanted: &[FaceRef], asked: &[StoredPick<FaceRef>]) -> Result<Vec<usize>, String> {
        let Some(state) = self.body(body) else { return Err("the body is gone".into()) };
        let t = state.topology();
        let tol = match_tolerance(state.diagonal());
        let same = |f: &FaceRef, p: &FaceRef| face_distance(f, p, &tol).is_some();
        let (live, rest) = live_picks(wanted, Some(asked), Some(&same));
        let mut hashes: Vec<usize> = Vec::new();
        let mut prints: Vec<FaceRef> = rest;
        let candidates = t.face_prints();
        let name = body.to_string();
        for p in live {
            let at = run_stored_where(&p.query, &t, &name, What::Face, None, &tol);
            if let Some(found) = at.as_ref().filter(|a| !a.is_empty()) {
                for i in found {
                    if !hashes.contains(i) {
                        hashes.push(*i);
                    }
                }
                continue;
            }
            let back: Vec<FaceRef> = if p.query.optional == Some(true) { p.found.iter().filter(|f| match_face(f, &candidates, &tol).is_some()).copied().collect() } else { p.found.clone() };
            if back.is_empty() {
                continue;
            }
            self.warn(format!("{} {}: its {} earlier face{} used — check {}.", where_words(&p.query), if at.is_some() { "names no face now" } else { "cannot be read now" }, back.len(), if back.len() == 1 { "" } else { "s" }, if back.len() == 1 { "it" } else { "them" }));
            prints.extend(back);
        }
        let found = pick_each(&prints, &candidates, &tol, |w, c, t| match_face(w, c, t), describe_face)?;
        self.note_found_by_place(&prints, &candidates, &found, describe_face, |w, c| face_distance(w, c, &tol).is_some(), |f| f.centroid, |f| f.origin);
        for i in found {
            if !hashes.contains(&i) {
                hashes.push(i);
            }
        }
        Ok(hashes)
    }

    /// As [`Builder::find_faces`], for edges.
    pub(crate) fn find_edges(&mut self, body: BodyId, wanted: &[EdgeRef], asked: &[StoredPick<EdgeRef>]) -> Result<Vec<usize>, String> {
        let Some(state) = self.body(body) else { return Err("the body is gone".into()) };
        let t = state.topology();
        let tol = match_tolerance(state.diagonal());
        let shape = state.shape.clone();
        let same = |f: &EdgeRef, p: &EdgeRef| edge_distance(f, p, &tol).is_some();
        let (live, rest) = live_picks(wanted, Some(asked), Some(&same));
        let mut hashes: Vec<usize> = Vec::new();
        let mut prints: Vec<EdgeRef> = rest;
        let candidates = t.edge_prints();
        let name = body.to_string();
        // The faces' normals for `sharp`, off one mesh of the body, made the first time a query
        // asks.
        let normals_mesh: OnceCell<Option<FaceMesh>> = OnceCell::new();
        let normals = |face: u32, points: &[DVec3]| match normals_mesh.get_or_init(|| FaceMesh::of(&shape)) {
            Some(fm) => fm.normals(face as usize, points),
            None => vec![None; points.len()],
        };
        let sharp_of = |i: usize| t.edges.get(i).is_some_and(|e| sharp_by_surfaces(e, &normals));
        for p in live {
            let at = run_stored_where(&p.query, &t, &name, What::Edge, Some(&sharp_of), &tol);
            if let Some(found) = at.as_ref().filter(|a| !a.is_empty()) {
                for i in found {
                    if !hashes.contains(i) {
                        hashes.push(*i);
                    }
                }
                continue;
            }
            let back: Vec<EdgeRef> = if p.query.optional == Some(true) { p.found.iter().filter(|f| match_edge(f, &candidates, &tol).is_some()).cloned().collect() } else { p.found.clone() };
            if back.is_empty() {
                continue;
            }
            self.warn(format!("{} {}: its {} earlier edge{} used — check {}.", where_words(&p.query), if at.is_some() { "names no edge now" } else { "cannot be read now" }, back.len(), if back.len() == 1 { "" } else { "s" }, if back.len() == 1 { "it" } else { "them" }));
            prints.extend(back);
        }
        let found = pick_each(&prints, &candidates, &tol, |w, c, t| match_edge(w, c, t), describe_edge)?;
        self.note_found_by_place(&prints, &candidates, &found, describe_edge, |w, c| edge_distance(w, c, &tol).is_some(), |e| e.mid, |e| e.origin);
        for i in found {
            if !hashes.contains(&i) {
                hashes.push(i);
            }
        }
        Ok(hashes)
    }

    /// Picks found again where they lie, on a face or an edge another feature made than the
    /// one that made it when it was picked, or only by the looser second look: said on the
    /// feature's row.
    #[allow(clippy::too_many_arguments)]
    fn note_found_by_place<F>(&mut self, wanted: &[F], candidates: &[F], found: &[usize], describe: impl Fn(&F) -> String, strict: impl Fn(&F, &F) -> bool, place: impl Fn(&F) -> DVec3, origin: impl Fn(&F) -> i64) {
        let mut away = Vec::new();
        let mut by_place = Vec::new();
        for (w, i) in wanted.iter().zip(found) {
            let Some(got) = candidates.get(*i) else { continue };
            if origin(w) >= 0 && origin(got) != origin(w) {
                by_place.push(describe(w));
                continue;
            }
            if strict(w, got) {
                continue;
            }
            away.push(format!("{} {:.1} mm off", describe(w), (place(got) - place(w)).length()));
        }
        if by_place.len() == 1 {
            let s = &by_place[0];
            let mut c = s.chars();
            let first = c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default();
            self.warn(format!("{first} found again by where it lies, not by its maker — check it."));
        } else if by_place.len() > 1 {
            self.warn(format!("{} picks found again by where they lie, not by their maker — check them.", by_place.len()));
        }
        if !away.is_empty() {
            const NAMED: usize = 3;
            let one = away.len() == 1;
            let more = if away.len() > NAMED { format!(" and {} more", away.len() - NAMED) } else { String::new() };
            let head = if one { "A pick".to_string() } else { format!("{} picks", away.len()) };
            self.warn(format!("{head} found again only loosely, the size let go: {}{more}. Check {}.", away[..away.len().min(NAMED)].join("; "), if one { "it" } else { "them" }));
        }
    }

    // ---- sketches ----

    /// A sketch's frame and the regions a feature picked of it, as the kernel takes them — or
    /// why there are none: nothing picked, or the sketch changed under the pick.
    pub(crate) fn sketch_profiles(&self, part: &Part, sketch_id: FeatureId, keys: &[String], pick: Option<&RegionPick>, what: &str) -> Result<(SectionFrame, Vec<Region2>), String> {
        self.track.note_sketch(sketch_id);
        let Some(built) = self.out.sketches.get(&sketch_id) else { return Err(absent_sketch(part, sketch_id, self.step, what)) };
        if keys.is_empty() {
            return Err("No profile picked — click the sketch's regions in the viewport.".into());
        }
        let regions = &built.regions;
        // A pick kept as it was given, while it stands, is taken as given; the keys otherwise.
        let given = pick.filter(|p| p.keys.len() == keys.len() && p.keys.iter().all(|k| keys.contains(k))).and_then(|p| regions_by_pick(&p.given, regions));
        let picked: Vec<&Region> = match given {
            Some(g) => g,
            None => {
                let mut out: Vec<&Region> = Vec::new();
                for k in keys {
                    if let Some(r) = regions.iter().find(|r| &r.key == k) {
                        if !out.iter().any(|o| o.key == r.key) {
                            out.push(r);
                        }
                    }
                }
                out
            }
        };
        if picked.is_empty() {
            let name = part.feature(sketch_id).map_or("the sketch".to_string(), |f| f.name.clone());
            return Err(if regions.is_empty() {
                "The picked profiles are no longer closed — the sketch changed under them.".to_string()
            } else {
                let listed: Vec<String> = regions.iter().take(8).map(|r| format!("{} ({:.3} mm²)", r.key, r.area)).collect();
                let n = keys.len();
                format!("None of the {} picked is in {name} as it is now — an edit remade the pieces round {}. It has {} closed region{}: {} — pick again.", if n == 1 { "profile".to_string() } else { format!("{n} profiles") }, if n == 1 { "it" } else { "them" }, regions.len(), if regions.len() == 1 { "" } else { "s" }, listed.join("; "))
            });
        }
        Ok((built.frame, picked.into_iter().map(|r| r.region.clone()).collect()))
    }
}

/// Why a sketch a feature names is not in the build: it stands further down, or is suppressed
/// — said so, not "gone", which reads as deleted.
pub(crate) fn absent_sketch(part: &Part, id: FeatureId, feature: FeatureId, what: &str) -> String {
    let at = part.index_of(id);
    let me = part.index_of(feature);
    match (at.and_then(|i| part.features.get(i)), at, me) {
        (Some(s), Some(at), Some(me)) if matches!(s.kind, FeatureKind::Sketch(_)) => {
            let named = format!("Sketch {id} ({})", s.name);
            if s.suppressed() {
                format!("{named} is suppressed — build it again, or take it out of this {what}.")
            } else if at > me {
                format!("{named} stands after this {what} — a feature takes only the sketches above it. Move it above.")
            } else {
                format!("The sketch this {what} builds on is gone.")
            }
        }
        _ => format!("The sketch this {what} builds on is gone."),
    }
}

/// The regions a pick kept as it was given names in the sketch as it stands: every region, the
/// largest, or each key's region and each point's — `None` where one of them names nothing now,
/// or the sketch has none.
pub fn regions_by_pick<'a>(given: &RegionGiven, regions: &'a [Region]) -> Option<Vec<&'a Region>> {
    if regions.is_empty() {
        return None;
    }
    match given {
        RegionGiven::Word(RegionWord::All) => Some(regions.iter().collect()),
        RegionGiven::Word(RegionWord::Largest) => regions.iter().max_by(|a, b| a.area.total_cmp(&b.area)).map(|r| vec![r]),
        RegionGiven::List(list) => {
            let mut out: Vec<&Region> = Vec::new();
            for p in list {
                let r = match p {
                    RegionAsk::Key(k) => regions.iter().find(|r| &r.key == k),
                    RegionAsk::At { at } => region_at(regions, *at),
                }?;
                if !out.iter().any(|o| o.key == r.key) {
                    out.push(r);
                }
            }
            Some(out)
        }
    }
}

/// The regions of a sketch an extrude takes when none is picked by hand: every region that is
/// no hole of another — an inner loop left unpicked is a hole.
pub fn default_regions(regions: &[Region]) -> Vec<String> {
    let holes: Vec<String> = regions.iter().flat_map(|r| r.holes.iter().map(|h| crate::regions::region_key(h))).collect();
    regions.iter().filter(|r| !holes.contains(&r.key)).map(|r| r.key.clone()).collect()
}

/// Match every wanted fingerprint to a different candidate, in order — or the first that names
/// none, described.
pub(crate) fn pick_each<F: Clone>(wanted: &[F], candidates: &[F], tol: &MatchTolerance, matcher: impl Fn(&F, &[F], &MatchTolerance) -> Option<usize>, describe: impl Fn(&F) -> String) -> Result<Vec<usize>, String> {
    let mut taken: HashSet<usize> = HashSet::new();
    let mut out = Vec::new();
    for w in wanted {
        let open: Vec<usize> = (0..candidates.len()).filter(|i| !taken.contains(i)).collect();
        let pool: Vec<F> = open.iter().map(|i| candidates[*i].clone()).collect();
        let j = matcher(w, &pool, tol).ok_or_else(|| describe(w))?;
        taken.insert(open[j]);
        out.push(open[j]);
    }
    Ok(out)
}

/// A shape as the solids it holds, largest first, with their volumes — the shape itself when
/// it is one solid or none.
pub(crate) fn islands(shape: &Body) -> Vec<(Body, f64)> {
    match shape.lumps() {
        Ok(pieces) if pieces.len() > 1 => {
            let mut out: Vec<(Body, f64)> = pieces.into_iter().map(|p| {
                let v = volume(&p).unwrap_or(0.0);
                (p, v)
            }).collect();
            out.sort_by(|a, b| b.1.total_cmp(&a.1));
            out
        }
        _ => vec![(shape.clone(), 0.0)],
    }
}

/// A point just inside a solid: off the middle of its largest facet, into the body.
fn probe_inside(body: &Body) -> Option<DVec3> {
    let tol = mesh_deflection(diagonal_of(body)).max(1e-3);
    let m = body.tessellate(tol).ok()?;
    let mut best: Option<(f64, DVec3, DVec3)> = None;
    for t in &m.triangles {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.positions.get(i as usize).copied().map(dv3).unwrap_or_default());
        let n = (b - a).cross(c - a);
        let area = n.length();
        if best.is_none_or(|(x, _, _)| area > x) {
            best = Some((area, (a + b + c) / 3.0, n.normalize_or_zero()));
        }
    }
    let (_, at, n) = best?;
    Some(at - n * (diagonal_of(body) * 1e-3).max(1e-3))
}

/// Which feature first made each face of an operation's result, from its operands' (each with
/// its faces' makers) — ScanRuler's `propagate` and `geometricHistory`. By the kernel's history
/// where the operation reports one: a face that came from an operand's face keeps that face's
/// maker, and a face several went into keeps the maker of the first that is not the step
/// itself — the top of a block stays the block's when a boss is joined flush with it, and is
/// the step's own only when nothing else went into it. Where the operation reports no history,
/// each face is traced to the face it most resembles — the same kind, near where it was, facing
/// much the same way — among the faces the step did not make (the body as it stood, never the
/// tool), and a face that resembles none is the step's own.
pub(crate) fn origins_after(result: &Body, operands: &[(&Body, &Vec<i64>)], step: FeatureId) -> Vec<i64> {
    let n = result.face_count();
    let own = step as i64;
    if let Some(h) = result.history() {
        if h.operands == operands.len() && h.faces.len() == n {
            return h
                .faces
                .iter()
                .map(|prov| {
                    let mut first: Option<i64> = None;
                    for o in &prov.origins {
                        let FaceOrigin::Face { operand, face } = o else { continue };
                        let Some(x) = operands.get(*operand).and_then(|(_, orig)| orig.get(*face)).copied().filter(|x| *x >= 0) else { continue };
                        if x != own {
                            return x;
                        }
                        first.get_or_insert(x);
                    }
                    first.unwrap_or(own)
                })
                .collect();
        }
    }
    let before: Vec<FaceFingerprint> = operands.iter().flat_map(|(b, o)| read_faces(b, o, None).into_iter().map(|f| f.fingerprint)).filter(|f| f.origin >= 0 && f.origin != own).collect();
    if before.is_empty() {
        return vec![own; n];
    }
    let tol = match_tolerance(diagonal_of(result));
    let after = read_faces(result, &vec![-1; n], None);
    if after.len() != n {
        return vec![own; n];
    }
    after
        .into_iter()
        .map(|f| {
            let mut best: Option<(f64, i64)> = None;
            for b in &before {
                let wanted = FaceFingerprint { origin: 0, ..*b };
                let got = FaceFingerprint { origin: 0, ..f.fingerprint };
                if let Some(d) = face_distance(&wanted, &got, &tol) {
                    if best.is_none_or(|(x, _)| d < x) {
                        best = Some((d, b.origin));
                    }
                }
            }
            best.map_or(own, |(_, o)| o)
        })
        .collect()
}

// ---- reading a body ----

/// What one tessellation of a body says of its faces: per face, the triangles on it.
struct FaceMesh {
    mesh: solidcraft_geom::Mesh,
}

impl FaceMesh {
    fn of(body: &Body) -> Option<FaceMesh> {
        let tol = mesh_deflection(diagonal_of(body));
        body.display_mesh_angular(tol, kernel::DISPLAY_ANGLE).ok().map(|mesh| FaceMesh { mesh })
    }

    fn tri(&self, t: &[u32; 3]) -> [DVec3; 3] {
        [t[0], t[1], t[2]].map(|i| self.mesh.positions.get(i as usize).copied().map(dv3).unwrap_or_default())
    }
}

/// The faces of a body with their fingerprints, each face's maker from `origins`.
fn read_faces(body: &Body, origins: &[i64], fm: Option<&FaceMesh>) -> Vec<FaceInfo> {
    let own;
    let fm = match fm {
        Some(f) => f,
        None => match FaceMesh::of(body) {
            Some(f) => {
                own = f;
                &own
            }
            None => return Vec::new(),
        },
    };
    let nf = body.face_count();
    // Area, area-weighted centre, and the triangles of each face.
    let mut area = vec![0.0f64; nf];
    let mut centre = vec![DVec3::ZERO; nf];
    let mut tris: Vec<Vec<usize>> = vec![Vec::new(); nf];
    for (k, (t, f)) in fm.mesh.triangles.iter().zip(&fm.mesh.tri_face).enumerate() {
        let f = *f as usize;
        if f >= nf {
            continue;
        }
        let [a, b, c] = fm.tri(t);
        let ar = (b - a).cross(c - a).length() * 0.5;
        area[f] += ar;
        centre[f] += (a + b + c) * (ar / 3.0);
        tris[f].push(k);
    }
    // Each face's surface: its kind, direction, radius and axis point; a round one's axis
    // too, for the turn it makes.
    let mut faces: Vec<FaceInfo> = Vec::with_capacity(nf);
    let mut axes: Vec<Option<(DVec3, DVec3)>> = Vec::with_capacity(nf);
    for i in 0..nf {
        let centroid = if area[i] > 0.0 { centre[i] / area[i] } else { DVec3::ZERO };
        // The middle of the face: the triangle nearest its centroid, its normal.
        let mut middle: Option<(f64, DVec3, DVec3)> = None;
        for k in &tris[i] {
            let [a, b, c] = fm.tri(&fm.mesh.triangles[*k]);
            let m = (a + b + c) / 3.0;
            let d = (m - centroid).length_squared();
            if middle.is_none_or(|(x, _, _)| d < x) {
                middle = Some((d, m, (b - a).cross(c - a).normalize_or_zero()));
            }
        }
        let (point, normal) = middle.map_or((centroid, DVec3::Z), |(_, m, n)| (m, if n == DVec3::ZERO { DVec3::Z } else { n }));
        let (kind, dir, radius, center, axis) = match kernel::face_surface(body, i) {
            Some(FaceSurface::Plane { normal: n, .. }) => (FaceKind::Plane, dv3(n.into()), None, None, None),
            Some(FaceSurface::Cylinder { origin, axis, radius }) => {
                let (o, a) = (dv3(origin), dv3(axis.into()));
                let c = o + a * (centroid - o).dot(a);
                (FaceKind::Cylinder, canonical_dir(a), Some(radius), Some(c), Some((a, c)))
            }
            Some(FaceSurface::Cone { apex, axis, .. }) => {
                let (o, a) = (dv3(apex), dv3(axis.into()));
                let c = o + a * (centroid - o).dot(a);
                (FaceKind::Cone, canonical_dir(a), None, Some(c), Some((a, c)))
            }
            Some(FaceSurface::Sphere { center, radius }) => (FaceKind::Sphere, normal, Some(radius), Some(dv3(center)), None),
            Some(FaceSurface::Torus { center, axis, .. }) => {
                let a = dv3(axis.into());
                (FaceKind::Torus, canonical_dir(a), None, None, Some((a, dv3(center))))
            }
            None => (FaceKind::Other, normal, None, None, None),
        };
        faces.push(FaceInfo { hash: i as u32, fingerprint: FaceFingerprint { kind, centroid, dir, area: area[i], radius, center, origin: origins.get(i).copied().unwrap_or(-1) }, full_turn: None, point });
        axes.push(axis);
    }
    // Whether a round face goes the whole way round its axis — with the faces lying on its
    // surface beside it: the kernel keeps a whole bore as two halves.
    for i in 0..nf {
        let Some((axis, on)) = axes[i] else { continue };
        let a = &faces[i].fingerprint;
        let same: Vec<usize> = (0..nf)
            .filter(|j| {
                let b = &faces[*j].fingerprint;
                *j == i
                    || match a.kind {
                        FaceKind::Torus => b.kind == FaceKind::Torus && axes[*j].is_some_and(|(x, c)| (c - on).length() < 1e-3 && x.cross(axis).length() < 1e-4),
                        _ => crate::topo::on_one_surface(a, b),
                    }
            })
            .collect();
        let points = same.iter().flat_map(|j| tris[*j].iter().flat_map(|k| fm.tri(&fm.mesh.triangles[*k])));
        faces[i].full_turn = Some(full_turn_of(points, on, axis));
    }
    faces
}

/// Whether the points of a round face go the whole way round its axis: no gap between them,
/// seen down the axis, wider than the mesh's own facets leave.
fn full_turn_of(points: impl Iterator<Item = DVec3>, on: DVec3, axis: DVec3) -> bool {
    let a = axis.normalize_or_zero();
    let (u, v) = crate::linalg::ortho_basis(a);
    let mut angles: Vec<f64> = points
        .filter_map(|p| {
            let d = p - on;
            let r = d - a * d.dot(a);
            (r.length() > 1e-9).then(|| r.dot(v).atan2(r.dot(u)))
        })
        .collect();
    if angles.len() < 3 {
        return false;
    }
    angles.sort_by(|x, y| x.total_cmp(y));
    let mut gap: f64 = angles[0] + std::f64::consts::TAU - angles[angles.len() - 1];
    for w in angles.windows(2) {
        gap = gap.max(w[1] - w[0]);
    }
    gap < 45f64.to_radians()
}

/// Every face and edge of a body with its fingerprint (ScanRuler's `readTopology`): a face's
/// kind, centroid, direction, area, radius and axis point off the kernel's surface and a mesh
/// of it; an edge's kind, middle, direction, length and radius, the faces either side, its
/// ends and the tangent at each. The edge belongs to the newest of its faces: a fillet's edge
/// on a face the fillet made is the fillet's, not the extrude's.
pub fn read_topology(body: &Body, origins: &[i64]) -> Topology {
    let Some(fm) = FaceMesh::of(body) else { return Topology::default() };
    let faces = read_faces(body, origins, Some(&fm));
    let size = diagonal_of(body).max(1.0);
    let edges = fm
        .mesh
        .edges
        .iter()
        .enumerate()
        .map(|(i, pts)| {
            let polyline: Vec<DVec3> = pts.iter().map(|p| dv3(*p)).collect();
            let mut on: Vec<u32> = fm.mesh.edge_faces.get(i).cloned().unwrap_or_default();
            let seam = fm.mesh.seams.get(i).copied().unwrap_or(false);
            if seam && on.len() == 1 {
                on.push(on[0]);
            }
            let first = polyline.first().copied().unwrap_or_default();
            let last = polyline.last().copied().unwrap_or(first);
            let along: f64 = polyline.windows(2).map(|w| (w[1] - w[0]).length()).sum();
            let mid = polyline_mid(&polyline);
            let tangent_at = |k: usize| polyline.get(k + 1).zip(polyline.get(k)).map(|(b, a)| (*b - *a).normalize_or_zero()).unwrap_or(DVec3::X);
            let start_tangent = tangent_at(0);
            let end_tangent = tangent_at(polyline.len().saturating_sub(2));
            let straight = polyline.len() >= 2 && (last - first).length() > 1e-9 && polyline.iter().all(|p| dist_to_segment(*p, first, last) <= 1e-6 * size);
            let circle = if straight && between_flats(&faces, &on) { None } else { kernel::edge_circle(body, i) };
            let (kind, dir, length, radius) = match &circle {
                Some(c) => (EdgeKind::Circle, canonical_dir(dv3(c.axis.into())), c.radius * c.angle, Some(c.radius)),
                None if straight => (EdgeKind::Line, canonical_dir(last - first), (last - first).length(), None),
                None => {
                    let k = polyline.len().saturating_sub(1) / 2;
                    (EdgeKind::Other, canonical_dir(tangent_at(k.min(polyline.len().saturating_sub(2)))), along, None)
                }
            };
            // The tangent at each end, as a chain is walked by: a circle's and a line's exact,
            // a curve's read off its first and last two facets, a facet's turn ahead of the
            // chord's.
            let (start_tangent, end_tangent) = match (&circle, kind) {
                (Some(c), _) => {
                    let (centre, axis) = (dv3(c.center), dv3(c.axis.into()));
                    let at = |p: DVec3| axis.cross(p - centre).normalize_or_zero();
                    (at(first), at(last))
                }
                (None, EdgeKind::Line) => {
                    let d = (last - first).normalize_or_zero();
                    (d, d)
                }
                _ if polyline.len() >= 3 => {
                    let n = polyline.len();
                    let lead = |a: DVec3, b: DVec3| (a * 1.5 - b * 0.5).normalize_or_zero();
                    (lead(tangent_at(0), tangent_at(1)), lead(tangent_at(n - 2), tangent_at(n - 3)))
                }
                _ => (start_tangent, end_tangent),
            };
            let mut unique: Vec<u32> = Vec::new();
            for f in &on {
                if !unique.contains(f) {
                    unique.push(*f);
                }
            }
            let sides: Vec<&FaceInfo> = unique.iter().filter_map(|h| faces.get(*h as usize)).collect();
            let origin = sides.iter().map(|f| f.fingerprint.origin).fold(-1, i64::max);
            EdgeInfo {
                hash: i as u32,
                fingerprint: EdgeFingerprint { kind, mid, dir, length, radius, faces: sides.iter().map(|f| EdgeSide { kind: f.fingerprint.kind, dir: f.fingerprint.dir }).collect(), origin },
                faces: on,
                ends: [first, last],
                end_tangents: [start_tangent, end_tangent],
                polyline,
            }
        })
        .collect();
    Topology { faces, edges }
}

impl FaceMesh {
    /// The unit normals of a face at the points of it nearest those asked — what a `sharp`
    /// query reads how two faces meet by (ScanRuler's kernel `faceNormals`).
    fn normals(&self, face: usize, points: &[DVec3]) -> Vec<Option<DVec3>> {
        let tris: Vec<[DVec3; 3]> = self.mesh.triangles.iter().zip(&self.mesh.tri_face).filter(|(_, f)| **f as usize == face).map(|(t, _)| self.tri(t)).collect();
        points
            .iter()
            .map(|p| {
                tris.iter()
                    .map(|[a, b, c]| (((*a + *b + *c) / 3.0 - *p).length_squared(), (*b - *a).cross(*c - *a).normalize_or_zero()))
                    .min_by(|x, y| x.0.total_cmp(&y.0))
                    .map(|(_, n)| n)
                    .filter(|n| *n != DVec3::ZERO)
            })
            .collect()
    }
}

fn polyline_mid(pts: &[DVec3]) -> DVec3 {
    let total: f64 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    let mut acc = 0.0;
    for w in pts.windows(2) {
        let l = (w[1] - w[0]).length();
        if acc + l >= total / 2.0 && l > 0.0 {
            return w[0] + (w[1] - w[0]) * ((total / 2.0 - acc) / l);
        }
        acc += l;
    }
    pts.first().copied().unwrap_or_default()
}

fn dist_to_segment(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let d = b - a;
    let t = if d.length_squared() > 0.0 { ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
    (a + d * t - p).length()
}

// ---- resolving planes, axes and elements ----

/// How far off a plane's normal `up` has to lie to run up the plane: its share in the plane.
const UP_MIN_SHARE: f64 = 0.1;

/// The u a plane's `up` asks for: up × normal, so v runs up the plane as near as it lets — or
/// `None` with no `up`, or one along the normal.
pub fn up_u(up: Option<WorldAxis>, normal: DVec3) -> Option<DVec3> {
    let n = normal.try_normalize()?;
    let u = up?.dir().cross(n);
    (u.length() >= UP_MIN_SHARE).then(|| u.normalize())
}

/// The frame a plane reference stands for now, or `None` when the thing it names is gone. The
/// plane is the reference's; the sheet's zero and X on it are the sketch rule's. A plane fitted
/// or a face's carries the frame it was given.
pub fn resolve_plane_frame(r: &PlaneRef, part: &Part) -> Option<SectionFrame> {
    let off = |frame: SectionFrame, offset: Option<f64>| match offset {
        Some(o) if o != 0.0 && o.is_finite() => frame.offset(o),
        _ => frame,
    };
    match r {
        PlaneRef::Fitted { frame, .. } | PlaneRef::BodyFace { frame, .. } => Some(*frame),
        PlaneRef::Radial { axis, angle_deg, offset } => {
            let (o, d) = resolve_axis(&(*axis).into(), None, part)?;
            let d = d.try_normalize()?;
            let frame = radial_frame(o - d * o.dot(d), d, *angle_deg)?;
            Some(off(frame, *offset))
        }
        PlaneRef::Section { id, offset, up } => match &part.element(*id)?.geometry {
            ElementGeometry::Section { frame } => Some(off(SectionFrame::sketch_on(frame.origin, frame.normal, up_u(*up, frame.normal))?, *offset)),
            _ => None,
        },
        PlaneRef::World { axis, offset, up } => {
            let dir = axis.dir();
            SectionFrame::sketch_on(dir * if offset.is_finite() { *offset } else { 0.0 }, dir, up_u(*up, dir))
        }
        PlaneRef::Plane { id, offset, up } => match &part.element(*id)?.geometry {
            ElementGeometry::Plane { center, normal, .. } => Some(off(SectionFrame::sketch_on(*center, *normal, up_u(*up, *normal))?, *offset)),
            _ => None,
        },
    }
}

/// A plane in the part: a point on it and its unit normal.
pub fn resolve_plane(r: &PlaneRef, part: &Part) -> Option<(DVec3, DVec3)> {
    resolve_plane_frame(r, part).map(|f| (f.origin, f.normal))
}

/// The name a plane reference reads as.
pub fn describe_plane_ref(r: &PlaneRef, part: &Part) -> String {
    let off = |o: Option<f64>| match o {
        Some(o) if o != 0.0 => format!(", {}{:.2} mm off it", if o > 0.0 { "+" } else { "−" }, o.abs()),
        _ => String::new(),
    };
    let name = |id: ElementId, gone: &str| part.element(id).map_or(gone.to_string(), |e| e.name.clone());
    match r {
        PlaneRef::Section { id, offset, .. } => name(*id, "deleted section") + &off(*offset),
        PlaneRef::Plane { id, offset, .. } => name(*id, "deleted plane") + &off(*offset),
        PlaneRef::Fitted { label, .. } => label.clone().unwrap_or_else(|| "a fitted plane".into()),
        PlaneRef::BodyFace { offset, .. } => "a face of the body".to_string() + &off(*offset),
        PlaneRef::Radial { axis, angle_deg, offset } => {
            let a = match axis {
                crate::model::RadialAxis::World { axis } => format!("the {:?} axis", axis),
                crate::model::RadialAxis::Element { id } => name(*id, "a deleted axis"),
            };
            format!("the plane through {a} at {}°", (angle_deg * 1000.0).round() / 1000.0) + &off(*offset)
        }
        PlaneRef::World { axis, offset, .. } => format!("{} at {:?} = {:.2}", axis.plane_name(), axis, offset),
    }
}

use crate::model::ElementId;

/// The axis a revolve or a pattern turns about, as a point and a direction, or `None`. A line of
/// the sketch needs the sketch.
pub fn resolve_axis(r: &AxisRef, sketch: Option<&SketchFeature>, part: &Part) -> Option<(DVec3, DVec3)> {
    match r {
        AxisRef::World { axis } => Some((DVec3::ZERO, axis.dir())),
        AxisRef::SketchLine { entity } => {
            let s = sketch?;
            match s.sketch.entity(*entity).map(|e| e.shape.clone()) {
                Some(crate::sketch::Shape::Line { a, b }) => {
                    let (pa, pb) = (s.frame.to_world(a), s.frame.to_world(b));
                    let d = (pb - pa).try_normalize()?;
                    Some((pa, d))
                }
                _ => None,
            }
        }
        AxisRef::Element { id } => match &part.element(*id)?.geometry {
            ElementGeometry::Line { center, dir, .. } => Some((*center, *dir)),
            ElementGeometry::Cylinder { center, axis, .. } | ElementGeometry::Cone { center, axis, .. } => Some((*center, *axis)),
            ElementGeometry::Circle { center, normal, .. } => Some((*center, *normal)),
            _ => None,
        },
    }
}

/// The solid a measured element stands for, or why there is none (ScanRuler's `primitiveOf`).
pub fn primitive_of(element: Option<&Element>, thickness: f64) -> Result<Primitive, String> {
    let Some(e) = element else { return Err("The element this body is taken from is gone.".into()) };
    match &e.geometry {
        ElementGeometry::Plane { center, normal, basis_u, basis_v, extent_u, extent_v } => {
            if !(thickness > 0.0) {
                return Err("A slab needs a thickness.".into());
            }
            Ok(Primitive::Slab { origin: *center, normal: *normal, basis_u: *basis_u, basis_v: *basis_v, extent_u: *extent_u, extent_v: *extent_v, thickness })
        }
        ElementGeometry::Cylinder { center, axis, radius, length, .. } => Ok(Primitive::Cylinder { origin: *center, axis: *axis, radius: *radius, length: *length }),
        ElementGeometry::Cone { center, axis, radius1, radius2, length, .. } => Ok(Primitive::Cone { origin: *center, axis: *axis, radius1: *radius1, radius2: *radius2, length: *length }),
        ElementGeometry::Sphere { center, radius } => Ok(Primitive::Sphere { center: *center, radius: *radius }),
        ElementGeometry::Torus { center, axis, major_radius, minor_radius, .. } => Ok(Primitive::Torus { center: *center, axis: *axis, major_radius: *major_radius, minor_radius: *minor_radius }),
        other => Err(format!("A {} has no volume to make a body of.", other.kind())),
    }
}

/// A plane with this origin and normal, as the kernel takes it, its X the sketch rule's.
pub(crate) fn plane_through(origin: DVec3, normal: DVec3) -> Result<Plane, String> {
    let n = normal.try_normalize().ok_or("The plane has no direction.")?;
    let u = crate::frame::sketch_x_axis(n);
    let u = (u - n * u.dot(n)).try_normalize().ok_or("The plane has no direction.")?;
    Plane::new(sv3(origin), sv3(u), sv3(n.cross(u))).ok_or_else(|| "The plane has no direction.".to_string())
}

/// A closed polygon in a plane's coordinates as a region.
pub(crate) fn polygon_region(points: &[DVec2]) -> Region2 {
    let pts: Vec<SVec2> = points.iter().map(|p| sv2(*p)).collect();
    Region2 { outer: Loop2::polygon(&pts), holes: Vec::new() }
}

/// A primitive's solid (ScanRuler's `primitiveSolid`).
pub(crate) fn primitive_solid(p: &Primitive) -> Result<Body, String> {
    match p {
        Primitive::Slab { origin, normal: _, basis_u, basis_v, extent_u, extent_v, thickness } => {
            let u = basis_u.try_normalize().unwrap_or(DVec3::X);
            let v = basis_v.try_normalize().unwrap_or(DVec3::Y);
            let plane = Plane::new(sv3(*origin), sv3(u), sv3(v)).ok_or("The slab's plane has no direction.")?;
            let region = polygon_region(&[DVec2::new(-extent_u, -extent_v), DVec2::new(*extent_u, -extent_v), DVec2::new(*extent_u, *extent_v), DVec2::new(-extent_u, *extent_v)]);
            let mut out = kernel::extrude(&plane, &[region], -thickness / 2.0, thickness / 2.0).map_err(kerr)?;
            out.pop().ok_or_else(|| "The slab could not be made.".to_string())
        }
        Primitive::Cylinder { origin, axis, radius, length } => {
            let a = axis.try_normalize().ok_or("The axis has no direction.")?;
            kernel::cylinder(sv3(*origin - a * (length / 2.0)), sv3(a), *radius, *length).map_err(kerr)
        }
        Primitive::Sphere { center, radius } => kernel::sphere(sv3(*center), *radius).map_err(kerr),
        Primitive::Cone { origin, axis, radius1, radius2, length } => {
            // A trapezoid in a plane through the axis, turned about it.
            let n = axis.try_normalize().ok_or("The axis has no direction.")?;
            let u = crate::linalg::ortho_basis(n).0;
            let a = *origin - n * (length / 2.0);
            let plane = Plane::new(sv3(a), sv3(u), sv3(n)).ok_or("The cone's axis has no direction.")?;
            let region = polygon_region(&[DVec2::new(0.0, 0.0), DVec2::new(radius1.max(1e-6), 0.0), DVec2::new(radius2.max(1e-6), *length), DVec2::new(0.0, *length)]);
            let mut out = kernel::revolve(&plane, &[region], SVec2::new(0.0, 0.0), SVec2::new(0.0, 1.0), std::f64::consts::TAU).map_err(kerr)?;
            out.pop().ok_or_else(|| "The cone could not be made.".to_string())
        }
        Primitive::Torus { center, axis, major_radius, minor_radius } => {
            let made = kernel::torus(SVec3::new(0.0, 0.0, 0.0), *major_radius, *minor_radius).map_err(kerr)?;
            let n = axis.try_normalize().ok_or("The axis has no direction.")?;
            let r = glam::DQuat::from_rotation_arc(DVec3::Z, n);
            let m = glam::DMat3::from_quat(r);
            let t = *center;
            let matrix = [[m.x_axis.x, m.x_axis.y, m.x_axis.z, 0.0], [m.y_axis.x, m.y_axis.y, m.y_axis.z, 0.0], [m.z_axis.x, m.z_axis.y, m.z_axis.z, 0.0], [t.x, t.y, t.z, 1.0]];
            kernel::transform_matrix(&made, matrix).map_err(kerr)
        }
        Primitive::Prism { .. } => Err("A section's block is not built yet.".into()),
    }
}

// ---- the build ----

/// Build the part: every feature in order, up to the roll-back bar.
pub fn build(part: &Part) -> Built {
    let upto = part.rolled_to.unwrap_or(part.features.len()).min(part.features.len());
    build_upto(part, upto)
}

/// Build the part's first `upto` features.
pub fn build_upto(part: &Part, upto: usize) -> Built {
    play(part, upto).finish(part)
}

/// The part built, and the part as it stood before one of its features — the bodies a feature
/// box picks on, as ScanRuler builds only the features before the one being edited
/// (`featuresToBuild`) so that a pick is taken on the body the feature will find it on again.
/// The rest is played on from there, not built again. `(before, whole)`; a feature not in the
/// part, or past the roll-back bar, has the whole build before it.
pub fn build_before(part: &Part, feature: FeatureId) -> (Built, Built) {
    let upto = part.rolled_to.unwrap_or(part.features.len()).min(part.features.len());
    let at = part.index_of(feature).map_or(upto, |i| i.min(upto));
    let mut b = Builder::new();
    play_range(&mut b, part, 0, at);
    // Each body read once, for both builds.
    for (_, s) in &b.bodies.raw {
        s.topology();
    }
    // The picks are taken on the bodies with no thread's faceted cut made yet (`thread::flush`).
    let mut pick = b.clone();
    pick.threads.clear();
    let before = pick.finish(part);
    play_range(&mut b, part, at, upto);
    (before, b.finish(part))
}

/// The part's first `upto` features played into a builder.
pub(crate) fn play(part: &Part, upto: usize) -> Builder {
    let mut b = Builder::new();
    play_range(&mut b, part, 0, upto);
    b
}

/// The part's features from `from` to `to` (by place in the list) played into a builder.
fn play_range(b: &mut Builder, part: &Part, from: usize, to: usize) {
    let to = to.min(part.features.len());
    if b.track.names.is_empty() {
        b.track.names = Arc::new(part.features.iter().map(|f| (f.id, f.name.clone())).collect());
    }
    for f in &part.features[from.min(to)..to] {
        if f.suppressed() {
            continue;
        }
        // ---- BUILD ---- one step as the engine plays it (diagnosis, words), nothing kept. The
        // threads' faceted cuts waiting are made first where the step is no thread or sketch
        // (HOLE, `thread::flush`), inside `play_step` so the engine's builds make them too.
        engine::play_step(b, part, f, false);
    }
}

impl Builder {
    /// What the build hands back: the bodies standing, named, with their faces and edges.
    pub(crate) fn finish(mut self, part: &Part) -> Built {
        thread::flush(&mut self);
        let mut bodies: Vec<BuiltBody> = self
            .bodies
            .raw
            .iter()
            .map(|(id, s)| BuiltBody { id: *id, name: String::new(), body: s.shape.clone(), surface: s.surface, label: s.label.clone(), topology: s.topology() })
            .collect();
        name_bodies(&mut bodies, part);
        self.out.bodies = bodies;
        self.out
    }
}

/// One feature, by its kind's own file.
fn build_one(b: &mut Builder, part: &Part, f: &Feature) -> Result<(), String> {
    match &f.kind {
        FeatureKind::Sketch(x) => sketch::build(b, part, f, x),
        FeatureKind::Extrude(x) => extrude::build(b, part, f, x),
        FeatureKind::Revolve(x) => revolve::build(b, part, f, x),
        FeatureKind::Extract(x) => extract::build(b, part, f, x),
        FeatureKind::Sheet(x) => sheet::build(b, part, f, x),
        FeatureKind::Loft(x) => loft::build(b, part, f, x),
        FeatureKind::Sweep(x) => sweep::build(b, part, f, x),
        FeatureKind::Thicken(x) => thicken::build(b, part, f, x),
        FeatureKind::Surface(x) => surface::build(b, part, f, x),
        FeatureKind::Network(x) => network::build(b, part, f, x),
        FeatureKind::Trim(x) => trim::build(b, part, f, x),
        FeatureKind::Import(x) => import::build(b, part, f, x),
        FeatureKind::MoveBody(x) => move_body::build(b, part, f, x),
        FeatureKind::Shell(x) => shell::build(b, part, f, x),
        FeatureKind::Fillet(x) => fillet::build(b, part, f, x),
        FeatureKind::Chamfer(x) => chamfer::build(b, part, f, x),
        FeatureKind::Hole(x) => hole::build(b, part, f, x),
        FeatureKind::Thread(x) => thread::build(b, part, f, x),
        FeatureKind::CutPlane(x) => cut_plane::build(b, part, f, x),
        FeatureKind::Mirror(x) => mirror::build(b, part, f, x),
        FeatureKind::Boolean(x) => boolean::build(b, part, f, x),
        FeatureKind::Split(x) => split::build(b, part, f, x),
        FeatureKind::Pattern(x) => pattern::build(b, part, f, x),
        FeatureKind::Draft(x) => draft::build(b, part, f, x),
        FeatureKind::MoveFace(x) => move_face::build(b, part, f, x),
        FeatureKind::DeleteFace(x) => delete_face::build(b, part, f, x),
        FeatureKind::OffsetBody(x) => offset_body::build(b, part, f, x),
    }
}

// ---- meshes and STEP ----

/// A body tessellated for display: triangles with their B-rep face, and the edges drawn (seams
/// left out), each with its index in the kernel's edge order.
#[derive(Debug, Clone, Default)]
pub struct BodyMesh {
    pub positions: Vec<DVec3>,
    pub normals: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
    pub tri_face: Vec<u32>,
    pub edges: Vec<Vec<DVec3>>,
    /// The index of each drawn edge in the kernel's edge order (the topology's hash).
    pub edge_ids: Vec<u32>,
    /// The faces the mesher left bare or drew in small part, drawn again on their own.
    pub faults: Vec<every_face::MeshFault>,
}

/// The display mesh of a body, at a chord tolerance that follows its size.
pub fn body_mesh(body: &Body) -> Result<BodyMesh, String> {
    let bb = body.outline_bounds();
    let size = if bb.is_empty() { 10.0 } else { (dv3(bb.max) - dv3(bb.min)).length() };
    let tol = (size * 4e-4).clamp(0.002, 0.2);
    let m = body.display_mesh(tol).map_err(kerr)?;
    // ---- BUILD ---- every face drawn: a face the mesher left bare drawn again on its own.
    let (m, faults) = every_face::mesh_every_face(body, m, tol);
    let kept: Vec<usize> = (0..m.edges.len()).filter(|i| !m.seams.get(*i).copied().unwrap_or(false)).collect();
    Ok(BodyMesh {
        positions: m.positions.iter().map(|p| dv3(*p)).collect(),
        normals: m.normals.iter().map(|p| dv3(*p)).collect(),
        triangles: m.triangles.clone(),
        tri_face: m.tri_face.clone(),
        edges: kept.iter().map(|i| m.edges[*i].iter().map(|p| dv3(*p)).collect()).collect(),
        edge_ids: kept.iter().map(|i| *i as u32).collect(),
        faults,
    })
}

/// The bodies as a STEP file (AP242), named.
pub fn export_step(bodies: &[(&str, &Body)]) -> Result<String, String> {
    let list: Vec<kernel::ExportBody> = bodies.iter().map(|(name, body)| kernel::ExportBody { name: name.to_string(), body, color: body.color() }).collect();
    let header = kernel::StepHeader { organization: "Polymix".to_string(), ..Default::default() };
    kernel::step_export_bodies(&list, &header).map_err(kerr)
}

/// The ids of a sketch's lines, for an axis pick.
pub fn sketch_lines(part: &Part, sketch: FeatureId) -> Vec<crate::sketch::EntityId> {
    part.sketch(sketch).map_or(Vec::new(), |s| s.sketch.entities.iter().filter(|e| matches!(e.shape, crate::sketch::Shape::Line { .. })).map(|e| e.id).collect())
}

/// The plane of a sketch, for the feature list.
pub fn plane_of(part: &Part, sketch: FeatureId) -> Option<PlaneRef> {
    part.sketch(sketch).map(|s| s.plane.clone())
}

// ---- HOLE ----

/// Whether an edge runs between two flat faces that are not one plane — a line, which the
/// kernel is not asked about: a threaded body is cut faceted, with thousands of them, and the
/// kernel finds an edge's curve by walking all its edges.
fn between_flats(faces: &[FaceInfo], on: &[u32]) -> bool {
    let (Some(a), Some(b)) = (on.first().and_then(|h| faces.get(*h as usize)), on.get(1).and_then(|h| faces.get(*h as usize))) else { return false };
    let (a, b) = (&a.fingerprint, &b.fingerprint);
    a.kind == FaceKind::Plane && b.kind == FaceKind::Plane && a.dir.cross(b.dir).length() > 1e-9
}

// The Hole's and the Thread's own words and readings, for their boxes and script steps.
pub use hole::{hole_problem, hole_unpicked, is_unpicked, resolve_hole_axis, unpicked_face, HoleAxisAt, HOLE_LEAD_MM, HOLE_UNPICKED};
pub use thread::{core_standard, model_standard, thread_cut_of, thread_resolve_problem, THREAD_UNPICKED};

#[cfg(test)]
mod tests;

// ---- BOOL ----

pub use cut_plane::{no_plane, plane_chosen, NO_PLANE};
pub use offset_body::{offset_fault, offset_solid};

#[cfg(test)]
mod bool_tests;

impl Builder {
    /// Bodies a step split off `id`, kept as its pieces (ScanRuler's `pieces`, set by
    /// `settle`).
    /// The read is noted for the engine (BUILD): a step that splits a body read its pieces.
    pub(crate) fn note_pieces(&mut self, id: BodyId, split: &[BodyId]) {
        if split.is_empty() {
            return;
        }
        self.track.note_piece(id);
        let list = self.pieces.raw.entry(id).or_default();
        for p in split {
            if *p != id && !list.contains(p) {
                list.push(*p);
            }
        }
    }

    /// What a feature placed made apart: where every body it made is new — a body of its own,
    /// or joined to nothing that stood — the bodies after the first are that one's pieces: two
    /// profiles extruded together are a body and a piece of it, as ScanRuler settles one shape
    /// of two solids. A solid joined into a body that stood is no piece of it, nor is a solid
    /// of the same step standing apart beside it (ScanRuler's `joinAll` notes none).
    fn note_made_apart(&mut self, ids: &[BodyId]) {
        let Some(first) = ids.first().copied() else { return };
        if !ids.iter().all(|i| i.feature == self.step) {
            return;
        }
        self.note_pieces(first, &ids[1..]);
    }

    /// A sheet put in place as one body, its pieces apart and all (ScanRuler's `settle` of a
    /// surface): a sheet is never taken apart into bodies a piece. Returns its id.
    pub(crate) fn settle_sheet(&mut self, id: BodyId, shape: Body, origins: Vec<i64>) -> Vec<BodyId> {
        let label = self.body(id).and_then(|b| b.label.clone());
        let mut state = BodyState::new(shape, origins);
        state.surface = true;
        state.label = label;
        self.set_body(id, state);
        vec![id]
    }

    /// What the build's words call a body: the name of the feature that made it — a piece a
    /// later step split off a body goes by that body's name, as ScanRuler's `settle` names a
    /// piece after the body it was.
    pub(crate) fn name_of(&self, part: &Part, id: BodyId) -> String {
        let mut at = self.resolve(id);
        for _ in 0..64 {
            let parent = self.pieces.iter().filter(|(p, list)| **p != at && list.contains(&at)).map(|(p, _)| *p).min_by_key(|p| (p.feature, p.island));
            match parent {
                Some(p) => at = p,
                None => break,
            }
        }
        boolean::body_name(part, at)
    }

    /// A body standing and every body split off it since — and off those — that stands
    /// (ScanRuler's `familyOf`): a plug patterned into four copies apart is four bodies, and a
    /// Combine naming the plug takes all four, as it did while they were one body.
    pub(crate) fn family_of(&self, id: BodyId) -> Vec<BodyId> {
        let mut out = Vec::new();
        let mut seen: HashSet<BodyId> = HashSet::new();
        let mut stack = vec![id];
        while let Some(at) = stack.pop() {
            let own = self.resolve(at);
            if !seen.insert(own) {
                continue;
            }
            if self.body(own).is_some() {
                out.push(own);
            }
            // The engine (BUILD) notes the pieces read, key by key.
            self.track.note_piece(own);
            if let Some(list) = self.pieces.raw.get(&own) {
                stack.extend(list.iter().rev().copied());
            }
        }
        out
    }
}
