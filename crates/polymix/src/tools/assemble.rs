//! Assemble: pose kit parts (a miniature's torso, arms, legs, weapons) by their joints.
//!
//! Each part owns **joints** (see [`crate::assembly`]): balls, sockets, pins, holes, flat faces,
//! points. **Add joint** makes one from a click on the part (the surface is recognized,
//! [`mm_edit::mate::recognize`]) or from a selection (Select over this tool), and the joint
//! editor changes its shape, radius, length, depth (proud of the surface or into it) and
//! position. A joint stays with its part.
//!
//! **Connect** joins a joint on one part to a joint on another: click each (an existing joint's
//! marker or surface, or a new surface, which becomes a joint when connecting). The part that is
//! not held by a grounded part is seated on the other, with everything already connected to it:
//! the joints' anchors meet ([`assembly::anchors`]), or the surfaces are fitted touching
//! ([`mate::contact_fit`]). The interaction (concentric, tangent, coaxial, flush, coincident)
//! and the motion (ball, hinge, fixed, …) are suggested from the joints' shapes and can be
//! changed. A joint holds one connection: connecting it again replaces the one it had.
//!
//! **Pose:** drag a part to turn it about its connection; what hangs from it follows and
//! grounded parts stay put. Ball: the grabbed point follows the cursor (Shift: twist about the
//! limb). Hinge: turn about the axis. Slider: slide along it. ⌘ turns in steps of 15°. The
//! selected connection has handles for the motions it allows, one turn or slide at a time: a
//! ball's three rings (its x, y and twist, turning with the part), a hinge's ring, a slider's
//! arrow, a cylindrical joint's ring and arrow, a planar joint's ring and two arrows (⌘: steps of
//! 15° or 1 mm, Shift: 1° or 0.1 mm, of the connection's own angle or slide). Its angles can
//! also be typed in the panel.

use super::*;
use super::transform::{dist_to_segment, AXIS_COLORS};
use crate::assembly::{self, anchors, initial_seat, Assembly, Connection, Drive, Interaction, JFrame, Joint, JointShape, Motion, Rigid, Selected};
use crate::ui_util;
use egui::{Color32, RichText};
use glam::{DQuat, EulerRot};
use mm_core::Aabb;
use mm_edit::mate::{self, Contact, ContactOptions, Feature, RecognizeOptions, Shape, ShapeKind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Pose,
    Connect,
    AddJoint,
}

/// What was picked for connecting.
#[derive(Clone)]
enum PickWhat {
    /// A joint the part has.
    Joint(u64),
    /// A surface, to become a joint when connecting (fitted to the part's selection, or
    /// recognized from the click).
    Surface { feature: Feature, selected: bool },
}

#[derive(Clone)]
struct Pick {
    object: ObjectId,
    /// The mesh it was found on (another mesh makes it stale).
    mesh: Arc<Mesh>,
    what: PickWhat,
}

impl Pick {
    /// The joint it is, or would be.
    fn joint(&self, asm: &Assembly) -> Option<Joint> {
        match &self.what {
            PickWhat::Joint(id) => asm.joint(*id).cloned(),
            PickWhat::Surface { feature, .. } => Some(Joint::from_feature(self.object, feature, &self.mesh)),
        }
    }

    fn label(&self, asm: &Assembly) -> String {
        match &self.what {
            PickWhat::Joint(id) => asm.joint(*id).map_or_else(|| "(a joint that is gone)".into(), |j| j.title()),
            PickWhat::Surface { feature, selected } => format!("{}{}", feature.describe(), if *selected { " (selection)" } else { " (new joint)" }),
        }
    }
}

/// How the picked parts are to be seated.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SeatChoice {
    /// The interaction (none: the usual one for the joints' shapes).
    interaction: Option<Interaction>,
    /// A ball's part keeps its turn (otherwise its limb comes out of the socket's opening).
    keep_turn: bool,
    /// Which way round keyed flat faces meet (0: the nearest).
    turn: usize,
    /// Off: connect the parts where they are.
    move_into_place: bool,
    /// Fit the surfaces touching (else the anchors meet).
    touch: bool,
    /// Moves the part from its seat, in the fixed joint's anchor axes.
    offset: DVec3,
}

impl Default for SeatChoice {
    fn default() -> Self {
        Self { interaction: None, keep_turn: false, turn: 0, move_into_place: true, touch: true, offset: DVec3::ZERO }
    }
}

/// What Connect would do with the two picks.
#[derive(Clone)]
struct Plan {
    /// Which pick's part moves (0 or 1); the other's joint is the connection's `a`.
    moving: usize,
    /// The parts that move with it (its whole group).
    parts: Vec<ObjectId>,
    motion: Rigid,
    /// The interactions the joints can make, and the one used.
    interactions: Vec<Interaction>,
    interaction: Interaction,
    suggested: Motion,
    seat: Rigid,
    /// Where it will turn.
    pivot: DVec3,
    clearance: Option<f64>,
    /// Connections the picked joints hold, to be replaced.
    replaces: Vec<String>,
}

/// What Select (open over this tool) is choosing a surface for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RefineFor {
    /// Pick `i` (the next pick when past the last).
    Pick(usize),
    /// A new joint on the active part.
    NewJoint,
    /// Joint `id`, fitted again.
    Joint(u64),
}

/// A pose handle on the selected connection: only the turns and slides its motion allows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Handle {
    /// A ring turning about axis `i` of the gizmo (a ball's x, y and twist; the axis of a
    /// hinge, cylindrical or planar connection is 2).
    Turn(usize),
    /// An arrow sliding along axis `i` (a slider's or cylindrical connection's axis is 2; a
    /// planar connection slides along 0 and 1).
    Slide(usize),
}

/// The selected connection's handles (Pose).
#[derive(Clone, Debug)]
struct Gizmo {
    connection: u64,
    kind: Motion,
    /// The parts the handles move, and whether that is the connection's child's side.
    parts: Vec<ObjectId>,
    moving_child: bool,
    /// On the moving side's frame; axes the moving frame's for a ball (they turn with the part),
    /// else the parent's (in which the connection's angle and slides are measured).
    centre: DVec3,
    axes: [DVec3; 3],
}

impl Gizmo {
    fn handles(&self) -> &'static [Handle] {
        match self.kind {
            Motion::Ball => &[Handle::Turn(0), Handle::Turn(1), Handle::Turn(2)],
            Motion::Hinge => &[Handle::Turn(2)],
            Motion::Cylindrical => &[Handle::Turn(2), Handle::Slide(2)],
            Motion::Slider => &[Handle::Slide(2)],
            Motion::Planar => &[Handle::Turn(2), Handle::Slide(0), Handle::Slide(1)],
            Motion::Fixed => &[],
        }
    }

    /// Moved by a drag's motion, as drawn while it goes on.
    fn moved(&self, m: &Rigid) -> Gizmo {
        let mut g = self.clone();
        g.centre = m.apply(self.centre);
        if self.kind == Motion::Ball {
            g.axes = self.axes.map(|a| m.rot * a);
        }
        g
    }

    /// The rings' radius and the arrows' half length, in world units (constant on screen).
    fn sizes(&self, cx: &ToolCx) -> (f64, f64) {
        let per = cx.world_per_point(self.centre);
        (per * 70.0, per * 100.0)
    }

    fn name(&self, h: Handle) -> &'static str {
        match (self.kind, h) {
            (Motion::Ball, Handle::Turn(0)) => "X",
            (Motion::Ball, Handle::Turn(1)) => "Y",
            (Motion::Ball, Handle::Turn(_)) => "Twist",
            (_, Handle::Turn(_)) => "Angle",
            (Motion::Planar, Handle::Slide(0)) => "Slide X",
            (Motion::Planar, Handle::Slide(_)) => "Slide Y",
            (_, Handle::Slide(_)) => "Slide",
        }
    }

    /// Points round ring `i`.
    fn ring(&self, i: usize, r: f64, n: usize) -> Vec<DVec3> {
        let (u, v) = mm_core::geom::orthonormal_basis(self.axes[i]);
        (0..=n).map(|k| {
            let t = k as f64 / n as f64 * std::f64::consts::TAU;
            self.centre + (u * t.cos() + v * t.sin()) * r
        }).collect()
    }

    /// The handle under `pos` (the nearest within a few points), and where on it.
    fn handle_at(&self, pos: egui::Pos2, cx: &ToolCx) -> Option<(Handle, DVec3)> {
        let (r, len) = self.sizes(cx);
        let mut best: Option<(f32, Handle, DVec3)> = None;
        for &h in self.handles() {
            let (d, at) = match h {
                Handle::Turn(i) => {
                    let pts = self.ring(i, r, 64);
                    let mut near = (f32::MAX, self.centre);
                    for w in pts.windows(2) {
                        if let (Some(a), Some(b)) = (cx.project(w[0]), cx.project(w[1])) {
                            let d = dist_to_segment(pos, a, b);
                            if d < near.0 {
                                near = (d, if pos.distance(a) < pos.distance(b) { w[0] } else { w[1] });
                            }
                        }
                    }
                    near
                }
                Handle::Slide(i) => match (cx.project(self.centre - self.axes[i] * len), cx.project(self.centre + self.axes[i] * len)) {
                    (Some(a), Some(b)) => (dist_to_segment(pos, a, b), self.centre),
                    _ => (f32::MAX, self.centre),
                },
            };
            if d < 7.0 && best.is_none_or(|b| d < b.0) {
                best = Some((d, h, at));
            }
        }
        best.map(|b| (b.1, b.2))
    }

    fn paint(&self, painter: &egui::Painter, cx: &ToolCx, active: Option<Handle>) {
        let Some(c) = cx.project(self.centre) else { return };
        let (r, len) = self.sizes(cx);
        for &h in self.handles() {
            let hot = active == Some(h);
            let (i, width) = match h {
                Handle::Turn(i) | Handle::Slide(i) => (i, if hot { 3.5 } else { 2.0 }),
            };
            let col = if hot { Color32::YELLOW } else { AXIS_COLORS[i] };
            match h {
                Handle::Turn(_) => {
                    let pts: Vec<egui::Pos2> = self.ring(i, r, 64).into_iter().filter_map(|p| cx.project(p)).collect();
                    painter.add(egui::Shape::line(pts, egui::Stroke::new(width, col)));
                }
                Handle::Slide(_) => {
                    for s in [1.0, -1.0] {
                        if let Some(tip) = cx.project(self.centre + self.axes[i] * (len * s)) {
                            painter.arrow(c, tip - c, egui::Stroke::new(width + 0.5, col));
                        }
                    }
                }
            }
        }
    }
}

/// A drag on a pose handle.
#[derive(Clone, Debug)]
struct HandleDrag {
    gizmo: Gizmo,
    handle: Handle,
    /// Where on the ring it was grabbed.
    grab: DVec3,
    /// The cursor's ray met the ring's plane (else the turn follows the cursor's move along the
    /// way the grabbed point goes on screen); for a slide, the ray was not along the axis.
    exact: bool,
    /// For a slide, where along the axis the cursor's ray passed at the press.
    along0: f64,
    /// The connection's angle (one-axis turns) or slide along the handle's axis at the press,
    /// as the panel shows it, and +1 or −1 as the moving side's motion adds to it or takes from it.
    base: f64,
    sign: f64,
    /// This drag's turn (radians) or slide (mm) so far, and the cursor.
    amount: f64,
    at: egui::Pos2,
}

impl HandleDrag {
    /// What the drag has done so far: a ball's turn, else the connection's angle or slide.
    fn readout(&self) -> String {
        let name = self.gizmo.name(self.handle);
        match self.handle {
            Handle::Turn(_) if self.gizmo.kind == Motion::Ball => format!("{name} {:+.1}°", self.amount.to_degrees()),
            Handle::Turn(_) => {
                let d = (self.base + self.sign * self.amount).to_degrees();
                format!("{name} {:.1}° ({:+.1}°)", (d + 180.0).rem_euclid(360.0) - 180.0, (self.sign * self.amount).to_degrees())
            }
            Handle::Slide(_) => format!("{name} {:.3} mm ({:+.3})", self.base + self.sign * self.amount, self.sign * self.amount),
        }
    }
}

/// Text on a dark plate, for readouts over the model.
fn plate_text(painter: &egui::Painter, at: egui::Pos2, text: String, col: Color32) {
    let galley = painter.layout_no_wrap(text, egui::FontId::proportional(13.0), col);
    let rect = egui::Rect::from_min_size(at, galley.size()).expand(3.0);
    painter.rect_filled(rect, 3.0, Color32::from_black_alpha(170));
    painter.galley(at, galley, col);
}

struct Drag {
    part: ObjectId,
    drive: Drive,
    /// Dragging one of the selected connection's handles (else the part itself).
    handle: Option<HandleDrag>,
    /// Where the part was grabbed (world), and the cursor then.
    grab: DVec3,
    start: egui::Pos2,
    /// Shift held when the drag began: twist a ball, slide a cylindrical joint, turn on a face.
    alt: bool,
    /// Centre of the moving parts (for turning a free group).
    centre: DVec3,
    /// From the connection to the moving parts (a ball's twist axis).
    limb: DVec3,
    motion: Rigid,
    /// History generation at the press (an undo meanwhile drops the drag).
    generation: u64,
}

/// Crossing triangles between parts (Check collisions).
struct Collisions {
    /// Shape versions of the parts checked.
    shapes: Vec<(ObjectId, u64)>,
    /// (part, part, crossing triangle pairs, connected to each other).
    pairs: Vec<(ObjectId, ObjectId, usize, bool)>,
    lines: Vec<(DVec3, DVec3)>,
}

pub struct AssembleTool {
    mode: Mode,
    picks: Vec<Pick>,
    /// The shape to look for (none: the best fit).
    shape: Option<ShapeKind>,
    /// How far a curved surface may stray from its shape (percent of its radius).
    tolerance_pct: f64,
    /// The motion chosen for the next connection (none: the usual one).
    motion: Option<Motion>,
    seat: SeatChoice,
    /// The touching fit for the picks as seated (redone when they or the seat change).
    contact: Option<Contact>,
    refining: Option<(RefineFor, Arc<AtomicBool>)>,
    request: Option<Box<dyn Tool>>,
    plan: Result<Option<Plan>, String>,
    hover: Option<(ObjectId, egui::Pos2)>,
    /// The selected connection's handle under the cursor (Pose), and the cursor.
    hover_handle: Option<(Handle, egui::Pos2)>,
    drag: Option<Drag>,
    pressed_at: Option<egui::Pos2>,
    /// The grounded parts, joints and connections when last planned (the object browser,
    /// deletes and undo change them).
    seen: Seen,
    /// An edit of the selected joint's or connection's numbers is going on (later changes join
    /// its undo step).
    editing: bool,
    snap_deg: f64,
    collisions: Option<Collisions>,
}

const SEL: Color32 = Color32::from_rgb(255, 205, 60);
const JOINT: Color32 = Color32::from_rgb(80, 200, 255);
const FREE: Color32 = Color32::from_rgb(170, 175, 185);
const APART: Color32 = Color32::from_rgb(255, 90, 80);
const PICK: [Vec4; 2] = [Vec4::new(1.0, 0.55, 0.15, 1.0), Vec4::new(0.35, 0.9, 0.4, 1.0)];

fn pick_color(i: usize) -> Color32 {
    let c = PICK[i.min(1)];
    Color32::from_rgb((c.x * 255.0) as u8, (c.y * 255.0) as u8, (c.z * 255.0) as u8)
}

/// What a plan depends on: the grounded parts, each joint's shape, size and place, and the
/// connections.
type Seen = (Vec<ObjectId>, Vec<(u64, JointShape, [u64; 6])>, Vec<(u64, ObjectId, ObjectId)>);

fn seen_of(asm: &Assembly) -> Seen {
    let bits = |j: &Joint| {
        let f = j.frame();
        [j.radius.to_bits(), j.length.to_bits(), f.origin.x.to_bits(), f.origin.y.to_bits(), f.origin.z.to_bits(), (f.rot.x + 2.0 * f.rot.y + 3.0 * f.rot.z).to_bits()]
    };
    (asm.grounded.clone(), asm.joints.iter().map(|j| (j.id, j.shape, bits(j))).collect(), asm.connections.iter().map(|c| (c.id, c.parent, c.child)).collect())
}

impl AssembleTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        assembly::follow(cx.scene);
        // Nothing grounded yet: hold the largest part (usually the body) still.
        let parts: Vec<&crate::scene::SceneObject> = cx.scene.objects.iter().filter(|o| o.visible && !o.reference).collect();
        if cx.scene.assembly.grounded.is_empty() && parts.len() >= 2 {
            let volume = |o: &crate::scene::SceneObject| {
                let e = o.bounds().extents();
                e.x * e.y * e.z
            };
            if let Some(big) = parts.iter().max_by(|a, b| volume(a).total_cmp(&volume(b))) {
                *cx.status = format!("Grounded {}, the largest part: it stays put while you pose the others (the ground symbol beside each part in the object browser changes it).", big.name);
                let id = big.id;
                cx.scene.assembly.set_grounded(id, true);
            }
        }
        let connected = !cx.scene.assembly.connections.is_empty();
        Self {
            mode: if connected { Mode::Pose } else { Mode::Connect },
            picks: Vec::new(),
            shape: None,
            tolerance_pct: 6.0,
            motion: None,
            seat: SeatChoice::default(),
            contact: None,
            refining: None,
            request: None,
            plan: Ok(None),
            hover: None,
            hover_handle: None,
            drag: None,
            pressed_at: None,
            seen: seen_of(&cx.scene.assembly),
            editing: false,
            snap_deg: 15.0,
            collisions: None,
        }
    }

    fn options(&self) -> RecognizeOptions {
        RecognizeOptions { kind: self.shape, tolerance: self.tolerance_pct / 100.0, ..Default::default() }
    }

    /// One undo step for an edit, joined by the changes that follow while it is dragged.
    fn begin_edit(&self, cx: &mut ToolCx, label: &str) {
        if self.editing {
            cx.history.touch();
        } else {
            cx.history.push(label, cx.scene);
        }
    }

    /// The part and triangle under `pos` as the view shows them: the parts the Connect preview
    /// draws seated are hit where they are drawn.
    fn pick_shown(&self, pos: egui::Pos2, cx: &mut ToolCx) -> Option<(ObjectId, u32)> {
        let preview = match (&self.plan, self.mode) {
            (Ok(Some(p)), Mode::Connect) if p.motion != Rigid::IDENTITY => Some((p.parts.clone(), p.motion.inverse())),
            _ => None,
        };
        let ray = cx.ray_at(pos);
        let mut best: Option<(f64, ObjectId, u32)> = None;
        for o in cx.scene.objects.iter_mut() {
            if !o.visible || o.see_through || o.reference {
                continue;
            }
            // A seated part: the ray taken back into where its mesh is.
            let r = match &preview {
                Some((parts, back)) if parts.contains(&o.id) => mm_core::Ray::new(back.apply(ray.origin), back.rot * ray.dir),
                _ => ray,
            };
            let bvh = o.bvh();
            if let Some(h) = bvh.raycast(&o.mesh, &r) {
                if best.is_none_or(|b| h.t < b.0) {
                    best = Some((h.t, o.id, h.tid));
                }
            }
        }
        best.map(|b| (b.1, b.2))
    }

    /// The joint whose marker is within a few points of `pos`, nearest first (where it is
    /// drawn: moved with its part by the Connect preview).
    fn joint_marker_at(&self, pos: egui::Pos2, cx: &ToolCx) -> Option<u64> {
        let preview = match (&self.plan, self.mode) {
            (Ok(Some(p)), Mode::Connect) => Some((p.parts.clone(), p.motion)),
            _ => None,
        };
        let shown = |j: &Joint| match &preview {
            Some((parts, m)) if parts.contains(&j.part) => m.apply(j.frame().origin),
            _ => j.frame().origin,
        };
        cx.scene
            .assembly
            .joints
            .iter()
            .filter(|j| cx.scene.get(j.part).is_some_and(|o| o.visible))
            .filter_map(|j| cx.project(shown(j)).map(|p| (p.distance(pos), j.id)))
            .filter(|(d, _)| *d < 9.0)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|x| x.1)
    }

    /// The joint of part `id` made of triangle `tid`, if any.
    fn joint_with_triangle(scene: &Scene, id: ObjectId, tid: u32) -> Option<u64> {
        let o = scene.get(id)?;
        scene.assembly.joints_of(id).find(|j| j.surface_on(&o.mesh).is_some_and(|t| t.contains(&tid))).map(|j| j.id)
    }

    /// The joint of part `id` that `feature` is, found by shape and place, for joints that do not
    /// know their triangles (loaded from a project): it takes the feature's triangles.
    fn joint_like(scene: &mut Scene, id: ObjectId, feature: &Feature) -> Option<u64> {
        let mesh = scene.get(id)?.mesh.clone();
        let made = Joint::from_feature(id, feature, &mesh);
        let reach = 0.25 * made.size().max(1e-6);
        let found = scene
            .assembly
            .joints_of(id)
            .filter(|j| j.surface_on(&mesh).is_none() && j.shape == made.shape)
            .filter(|j| (j.base.origin - made.base.origin).length() <= reach && (j.shape.round() || j.base.axis().dot(made.base.axis()).abs() > 0.95))
            .min_by(|a, b| (a.base.origin - made.base.origin).length().total_cmp(&(b.base.origin - made.base.origin).length()))
            .map(|j| j.id)?;
        if let Some(j) = scene.assembly.joint_mut(found) {
            j.surface = made.surface.clone();
        }
        Some(found)
    }

    /// The surface at triangle `tid` of part `id`: fitted to the selection when the triangle is
    /// selected, else recognized.
    fn feature_at(&self, scene: &Scene, id: ObjectId, tid: u32) -> Result<(Feature, bool), String> {
        let o = scene.get(id).ok_or("the part is gone")?;
        let selected = o.selection.contains(tid);
        let f = if selected { mate::fit_region(&o.mesh, &o.selection.to_vec(), self.shape) } else { mate::recognize(&o.mesh, tid, &self.options()) };
        f.map(|f| (f, selected)).ok_or(format!("No ball, pin or flat face found there on {}. Choose the shape to look for, or select the surface (Select) and click it.", o.name))
    }

    /// A click in Connect: a joint's marker, a joint's surface, or a new surface.
    fn click_connect(&mut self, pos: egui::Pos2, cx: &mut ToolCx) {
        if let Some(j) = self.joint_marker_at(pos, cx) {
            self.pick_joint(j, cx);
        } else if let Some((id, tid)) = self.pick_shown(pos, cx) {
            self.pick_on(id, tid, cx);
        }
    }

    /// Pick joint `j`.
    fn pick_joint(&mut self, j: u64, cx: &mut ToolCx) -> bool {
        let Some(joint) = cx.scene.assembly.joint(j) else { return false };
        let Some(o) = cx.scene.get(joint.part) else { return false };
        *cx.status = format!("{}: {}", o.name, joint.title());
        let pick = Pick { object: joint.part, mesh: o.mesh.clone(), what: PickWhat::Joint(j) };
        self.add_pick(pick, cx.scene);
        true
    }

    /// Pick at triangle `tid` of part `id`: the joint made of it, or the surface there.
    fn pick_on(&mut self, id: ObjectId, tid: u32, cx: &mut ToolCx) -> bool {
        if let Some(j) = Self::joint_with_triangle(cx.scene, id, tid) {
            return self.pick_joint(j, cx);
        }
        match self.feature_at(cx.scene, id, tid) {
            Ok((feature, selected)) => {
                // A joint loaded without its triangles, found by shape and place.
                if let Some(j) = Self::joint_like(cx.scene, id, &feature) {
                    return self.pick_joint(j, cx);
                }
                let o = cx.scene.get(id).unwrap();
                *cx.status = format!("{}: {}{}", o.name, feature.describe(), if selected { " (fitted to the selection)" } else { "" });
                let pick = Pick { object: id, mesh: o.mesh.clone(), what: PickWhat::Surface { feature, selected } };
                self.add_pick(pick, cx.scene);
                true
            }
            Err(e) => {
                *cx.status = e;
                false
            }
        }
    }

    /// Add a pick: a third starts again; a second on the first part replaces its pick.
    fn add_pick(&mut self, pick: Pick, scene: &Scene) {
        let id = pick.object;
        if self.picks.len() >= 2 || self.picks.first().is_some_and(|p| p.object == id) {
            self.picks.clear();
        }
        self.picks.push(pick);
        self.seat = SeatChoice { move_into_place: self.seat.move_into_place, touch: self.seat.touch, ..SeatChoice::default() };
        self.motion = None;
        self.mode = Mode::Connect;
        self.recontact(scene);
    }

    /// Make a joint on part `id` at triangle `tid` (Add joint), or select the joint that surface
    /// is already. Returns the joint and whether it is new.
    fn add_joint_at(&mut self, id: ObjectId, tid: u32, cx: &mut ToolCx) -> Result<(u64, bool), String> {
        let existing = match Self::joint_with_triangle(cx.scene, id, tid) {
            Some(j) => Some(j),
            None => {
                let (feature, _) = self.feature_at(cx.scene, id, tid)?;
                match Self::joint_like(cx.scene, id, &feature) {
                    Some(j) => Some(j),
                    None => return Ok((self.add_joint_from(id, &feature, cx), true)),
                }
            }
        };
        let j = existing.unwrap();
        cx.scene.assembly.selected = Some(Selected::Joint(j));
        let other = |js: JointShape| self.shape.is_some_and(|k| js.kind() != Some(k));
        let known = |jj: &Joint| cx.scene.get(jj.part).is_some_and(|o| jj.surface_on(&o.mesh).is_some());
        if cx.scene.assembly.joint(j).is_some_and(|jj| known(jj) && other(jj.shape)) {
            cx.history.push("Joint shape", cx.scene);
            *cx.status = match assembly::refit_joint(cx.scene, j, self.shape, None) {
                Ok(m) => format!("That surface is a joint already, fitted again as you look for: {m}"),
                Err(m) => m,
            };
        } else {
            *cx.status = "That surface is a joint already: selected (Look for fits it again as another shape).".into();
        }
        Ok((j, false))
    }

    fn add_joint_from(&mut self, id: ObjectId, feature: &Feature, cx: &mut ToolCx) -> u64 {
        let mesh = cx.scene.get(id).map(|o| o.mesh.clone()).unwrap_or_default();
        cx.history.push("Add joint", cx.scene);
        let j = cx.scene.assembly.add_joint(Joint::from_feature(id, feature, &mesh));
        cx.scene.assembly.selected = Some(Selected::Joint(j));
        cx.scene.active = Some(id);
        let name = cx.scene.get(id).map_or_else(String::new, |o| o.name.clone());
        *cx.status = format!("Added {} to {name}: edit it in the panel, then Connect it.", cx.scene.assembly.joint(j).map_or_else(String::new, |j| j.title()));
        j
    }

    /// Open Select (for a joint's surface) over this tool. For pick `i`, or joint `id` fitted
    /// again, it starts from the triangles found; for a new pick or joint, from nothing. "Use
    /// this surface" there fits the joint to the selection when this tool is back.
    fn start_refine(&mut self, target: RefineFor, cx: &mut ToolCx) {
        let seed: Option<(ObjectId, Vec<u32>)> = match target {
            RefineFor::Pick(i) => self.picks.get(i).and_then(|p| match &p.what {
                PickWhat::Surface { feature, selected: false } => Some((p.object, feature.tris.clone())),
                PickWhat::Surface { selected: true, .. } => None,
                PickWhat::Joint(j) => cx.scene.assembly.joint(*j).and_then(|j| j.surface.as_ref().map(|s| (p.object, s.tris.clone()))),
            }),
            RefineFor::Joint(j) => cx.scene.assembly.joint(j).and_then(|j| j.surface.as_ref().map(|s| (j.part, s.tris.clone()))),
            RefineFor::NewJoint => None,
        };
        let on = match target {
            RefineFor::Pick(i) => self.picks.get(i).map(|p| p.object),
            RefineFor::Joint(j) => cx.scene.assembly.joint(j).map(|j| j.part),
            RefineFor::NewJoint => None,
        };
        if let Some(id) = on {
            cx.scene.active = Some(id);
        }
        // Start from the surface found, or from nothing.
        let keep = matches!(target, RefineFor::Pick(i) if self.picks.get(i).is_some_and(|p| matches!(p.what, PickWhat::Surface { selected: true, .. })));
        if !keep && (seed.is_some() || cx.scene.active().is_some_and(|o| !o.selection.is_empty())) {
            cx.history.push("Select joint surface", cx.scene);
            if let Some(o) = cx.scene.active_mut() {
                o.selection.clear();
                if let Some((_, tris)) = &seed {
                    for &t in tris.iter().filter(|&&t| o.mesh.is_triangle(t)) {
                        o.selection.insert(t);
                    }
                }
                o.touch_selection();
            }
        }
        let part = cx.scene.active().map_or_else(String::new, |o| o.name.clone());
        let title = match target {
            RefineFor::Pick(i) if i < self.picks.len() => format!("Surface {} on {part}: add with the brush, remove with Shift.", i + 1),
            RefineFor::Pick(_) => format!("Surface {}: paint {}.", self.picks.len().min(1) + 1, if self.picks.len() == 1 { "where it fits on the other part (socket, hole, recess)" } else { "the joint on one part (ball, pin, peg end)" }),
            RefineFor::NewJoint => "A new joint: paint its surface on the part.".into(),
            RefineFor::Joint(j) => format!("{} on {part}: paint its surface again.", cx.scene.assembly.joint(j).map_or_else(String::new, |j| j.title())),
        };
        let used = Arc::new(AtomicBool::new(false));
        self.refining = Some((target, used.clone()));
        self.request = Some(Box::new(super::select::SelectTool::for_joint_surface(title, used)));
        *cx.status = "Select the joint's surface, then Use this surface (selecting never changes the parts).".into();
    }

    /// The feature fitted to part `object`'s selection.
    fn selection_feature(&self, scene: &Scene, object: ObjectId) -> Result<(Feature, Arc<Mesh>, usize), String> {
        let o = scene.get(object).ok_or("the part is gone")?;
        let sel = o.selection.to_vec();
        if sel.is_empty() {
            return Err(format!("Nothing is selected on {}.", o.name));
        }
        let f = mate::fit_region(&o.mesh, &sel, self.shape).ok_or(format!("No ball, pin or flat face fits the selection on {}.", o.name))?;
        Ok((f, o.mesh.clone(), sel.len()))
    }

    /// Back from Select: use the selection for what it was chosen for.
    fn use_selection(&mut self, target: RefineFor, cx: &mut ToolCx) -> Result<String, String> {
        match target {
            RefineFor::Pick(i) => {
                let object = match self.picks.get(i) {
                    Some(p) => p.object,
                    None => cx.scene.active.ok_or("there is no active part")?,
                };
                let (f, mesh, n) = self.selection_feature(cx.scene, object)?;
                let msg = format!("{} fitted to {n} selected triangles", f.describe());
                let what = match Self::joint_like(cx.scene, object, &f) {
                    Some(j) => PickWhat::Joint(j),
                    None => PickWhat::Surface { feature: f, selected: true },
                };
                let pick = Pick { object, mesh, what };
                match self.picks.get_mut(i) {
                    Some(p) => {
                        *p = pick;
                        self.recontact(cx.scene);
                    }
                    None => self.add_pick(pick, cx.scene),
                }
                Ok(msg)
            }
            RefineFor::NewJoint => {
                let object = cx.scene.active.ok_or("there is no active part")?;
                let (f, _, _) = self.selection_feature(cx.scene, object)?;
                self.add_joint_from(object, &f, cx);
                Ok(cx.status.clone())
            }
            RefineFor::Joint(j) => {
                let object = cx.scene.assembly.joint(j).ok_or("the joint is gone")?.part;
                let (f, mesh, n) = self.selection_feature(cx.scene, object)?;
                let fresh = Joint::from_feature(object, &f, &mesh);
                cx.history.push("Fit joint again", cx.scene);
                let notes = assembly::edit_joint(cx.scene, j, |jj| {
                    let (name, offset) = (std::mem::take(&mut jj.name), jj.offset);
                    *jj = Joint { id: jj.id, name, offset, ..fresh };
                })?;
                Ok(format!("Fitted again to {n} selected triangles: {}. {}", f.describe(), notes.join(" ")))
            }
        }
    }

    /// Recognize the picked surfaces again (after the shape or tolerance changed).
    fn refit(&mut self, scene: &Scene) {
        let opts = self.options();
        for p in &mut self.picks {
            let Some(o) = scene.get(p.object) else { continue };
            if let PickWhat::Surface { feature, selected } = &mut p.what {
                let f = if *selected {
                    mate::fit_region(&o.mesh, &o.selection.to_vec(), self.shape)
                } else {
                    let seed = feature.tris.first().copied().unwrap_or(0);
                    mate::recognize(&o.mesh, seed, &opts)
                };
                if let Some(f) = f {
                    *feature = f;
                }
            }
        }
        self.recontact(scene);
    }

    fn replan(&mut self, scene: &Scene) {
        self.plan = plan(scene, &self.picks, &self.seat, self.contact.as_ref());
    }

    /// Fit the picked surfaces touching (when that is how they are seated), then plan.
    fn recontact(&mut self, scene: &Scene) {
        self.contact = None;
        if self.seat.touch && self.seat.move_into_place {
            let bare = SeatChoice { offset: DVec3::ZERO, ..self.seat };
            // A tangent connection rests on the ball's radius, not on the surfaces.
            if let Ok(Some(p)) = plan(scene, &self.picks, &bare, None).map(|p| p.filter(|p| p.interaction != Interaction::Tangent)) {
                let (mv, fx) = (&self.picks[p.moving], &self.picks[1 - p.moving]);
                let asm = &scene.assembly;
                if let (Some(jb), Some(ja)) = (mv.joint(asm), fx.joint(asm)) {
                    // Only surfaces that still describe their joints (not edited since).
                    if let (Some(tb), Some(ta)) = (jb.fitted_surface_on(&mv.mesh), ja.fitted_surface_on(&fx.mesh)) {
                        let opts = ContactOptions { turn: !(ja.shape.round() || jb.shape.round()), ..Default::default() };
                        self.contact = mate::contact_fit(&mv.mesh, tb, &fx.mesh, ta, (p.motion.rot, p.motion.shift), &opts);
                    }
                }
            }
        }
        self.replan(scene);
    }

    /// Connect the picked joints (making the new ones).
    fn connect(&mut self, cx: &mut ToolCx) {
        // Plan afresh: connections may have come or gone since (a delete, an undo).
        self.recontact(cx.scene);
        let plan = match self.plan.clone() {
            Ok(Some(p)) => p,
            Ok(None) => return,
            Err(e) => {
                *cx.status = e;
                return;
            }
        };
        let (mv, fx) = (self.picks[plan.moving].clone(), self.picks[1 - plan.moving].clone());
        let name = |id: ObjectId| cx.scene.get(id).map_or_else(String::new, |o| o.name.clone());
        let (mv_name, fx_name) = (name(mv.object), name(fx.object));
        let kind = self.motion.unwrap_or(plan.suggested);
        let touching = self.contact.filter(|_| self.seat.touch && self.seat.move_into_place);
        cx.history.push(format!("Connect {mv_name} to {fx_name}"), cx.scene);
        let asm = &mut cx.scene.assembly;
        // A joint holds one connection: the ones the picked joints hold go.
        for p in [&mv, &fx] {
            if let PickWhat::Joint(j) = p.what {
                if let Some(c) = asm.connection_at(j).map(|c| c.id) {
                    asm.remove_connection(c);
                }
            }
        }
        let add = |asm: &mut Assembly, p: &Pick| match &p.what {
            PickWhat::Joint(j) => *j,
            PickWhat::Surface { feature, .. } => asm.add_joint(Joint::from_feature(p.object, feature, &p.mesh)),
        };
        let a = add(asm, &fx);
        let b = add(asm, &mv);
        let about = format!("{} in {}", asm.joint(b).map_or_else(String::new, |j| j.describe()), asm.joint(a).map_or_else(String::new, |j| j.describe()));
        asm.reroot(mv.object);
        assembly::move_parts(cx.scene, &plan.parts, &plan.motion);
        match cx.scene.assembly.connect(a, b, plan.interaction, kind, plan.seat, about) {
            Ok(id) => {
                cx.scene.assembly.selected = Some(Selected::Connection(id));
                let gap = touching.map_or_else(String::new, |c| format!(" The surfaces touch (gap {:.3} mm, was {:.3}; deepest overlap {:.3}).", c.gap_after, c.gap_before, c.overlap));
                *cx.status = format!("Connected {mv_name} to {fx_name} ({}, {}).{gap} Drag {mv_name} to pose it.", plan.interaction.label().to_lowercase(), kind.label().to_lowercase());
            }
            Err(e) => *cx.status = e,
        }
        assembly::follow(cx.scene);
        self.picks.clear();
        self.plan = Ok(None);
        self.motion = None;
        self.seat = SeatChoice { move_into_place: self.seat.move_into_place, touch: self.seat.touch, ..SeatChoice::default() };
        self.contact = None;
        self.mode = Mode::Pose;
    }

    /// Move connection `cid`'s child by `m` relative to its parent (the parent's side the other
    /// way if the child's is held).
    fn move_connection(&mut self, cx: &mut ToolCx, cid: u64, m: Rigid, label: &str) {
        self.begin_edit(cx, label);
        if let Err(e) = assembly::move_connection(cx.scene, cid, &m) {
            *cx.status = e;
        }
    }

    /// Fit connection `cid`'s joints' surfaces touching, where its parts are now; it keeps its
    /// pose (the fit becomes its seat).
    fn fit_touching(&mut self, cx: &mut ToolCx, cid: u64) -> Result<String, String> {
        let asm = &cx.scene.assembly;
        let c = asm.connection(cid).ok_or(format!("there is no connection {cid}"))?.clone();
        let (ja, jb) = (asm.joint(c.a).ok_or("a joint is gone")?.clone(), asm.joint(c.b).ok_or("a joint is gone")?.clone());
        let (ma, mb) = match (cx.scene.get(c.part_a), cx.scene.get(c.part_b)) {
            (Some(x), Some(y)) => (x.mesh.clone(), y.mesh.clone()),
            _ => return Err("a part is gone".into()),
        };
        if c.interaction == Interaction::Tangent {
            return Err("A tangent connection rests on the ball's radius: change the ball joint's radius to move it in or out.".into());
        }
        let (Some(ta), Some(tb)) = (ja.fitted_surface_on(&ma), jb.fitted_surface_on(&mb)) else {
            return Err("A joint's surface no longer describes it (edited since it was found, loaded from a file, or its part's triangles changed): fit it again from a selection first.".into());
        };
        let child_side = asm.side(cid, c.child);
        let parent_side = asm.side(cid, c.parent);
        let held = |side: &[ObjectId]| side.iter().any(|&p| asm.is_grounded(p));
        // The side that moves, its joint's surface onto the other's.
        let (parts, moving_a) = if !held(&child_side) {
            (child_side, c.child == c.part_a)
        } else if !held(&parent_side) {
            (parent_side, c.parent == c.part_a)
        } else {
            return Err("Both sides of this connection are held by grounded parts.".into());
        };
        let opts = ContactOptions { turn: !(ja.shape.round() || jb.shape.round()), ..Default::default() };
        let start = (DQuat::IDENTITY, DVec3::ZERO);
        let fit = if moving_a { mate::contact_fit(&ma, ta, &mb, tb, start, &opts) } else { mate::contact_fit(&mb, tb, &ma, ta, start, &opts) };
        let fit = fit.ok_or("The surfaces do not face each other here.")?;
        let (rel, off) = c.pose();
        cx.history.push("Fit touching", cx.scene);
        assembly::move_parts(cx.scene, &parts, &Rigid { rot: fit.rot, shift: fit.shift });
        if let Some(cm) = cx.scene.assembly.connection_mut(cid) {
            cm.keep_pose(rel, off);
        }
        Ok(format!("The surfaces touch: gap {:.3} mm (was {:.3}), deepest overlap {:.3} mm.", fit.gap_after, fit.gap_before, fit.overlap))
    }

    /// Put every connection back in its rest pose (as connected), from the grounded parts out.
    fn reset_all(&mut self, cx: &mut ToolCx) {
        let asm = &cx.scene.assembly;
        if asm.connections.is_empty() {
            return;
        }
        let mut order: Vec<(usize, u64)> = asm.connections.iter().map(|c| (asm.path(c.child, asm.root_of(c.child)).len().min(asm.path(c.parent, asm.root_of(c.parent)).len()), c.id)).collect();
        order.sort();
        cx.history.push("Reset pose", cx.scene);
        self.editing = true;
        for (_, cid) in order {
            if let Some(c) = cx.scene.assembly.connection(cid) {
                let m = c.motion_to(DQuat::IDENTITY, DVec3::ZERO);
                self.move_connection(cx, cid, m, "Reset pose");
            }
        }
        self.editing = false;
    }

    fn drag_motion(&self, d: &Drag, pos: egui::Pos2, snap: bool, cx: &ToolCx) -> Option<Rigid> {
        let ray = cx.ray_at(pos);
        let dir = ray.dir.normalize();
        let moved = pos - d.start;
        let snap_angle = |a: f64| {
            let step = self.snap_deg.to_radians();
            if snap && step > 0.0 { (a / step).round() * step } else { a }
        };
        let screen_turn = |axis: DVec3, about: DVec3| snap_angle(screen_angle(cx, d.grab, d.start, pos, axis, about));
        match &d.drive {
            Drive::Grounded(_) => None,
            Drive::Free(_) if d.alt => {
                let k = 0.01;
                let r = DQuat::from_axis_angle(cx.camera.up(), moved.x as f64 * k) * DQuat::from_axis_angle(cx.camera.right(), moved.y as f64 * k);
                Some(Rigid::about(r, d.centre))
            }
            Drive::Free(_) => ray_plane(&ray, d.grab, cx.camera.forward()).map(|h| Rigid::translation(h - d.grab)),
            Drive::Connection { connection, frame, .. } => {
                let kind = cx.scene.assembly.connection(*connection)?.kind;
                let (c, axis) = (frame.origin, frame.axis());
                let turn = || {
                    let a = turn_angle(&ray, d.grab, c, axis, 0.3).map(snap_angle).unwrap_or_else(|| screen_turn(axis, c));
                    Rigid::about(DQuat::from_axis_angle(axis, a), c)
                };
                let slide = || {
                    let s = slide_along(&ray, d.grab, axis).unwrap_or_else(|| screen_slide(cx, d.grab, d.start, pos, axis));
                    Rigid::translation(axis * s)
                };
                Some(match (kind, d.alt) {
                    (Motion::Fixed, _) => Rigid::IDENTITY,
                    (Motion::Ball, true) => {
                        let a = snap_angle(moved.x as f64 * 0.01);
                        Rigid::about(DQuat::from_axis_angle(d.limb, a), c)
                    }
                    (Motion::Ball, false) => {
                        let r0 = d.grab - c;
                        let radius = r0.length();
                        if radius < 1e-12 {
                            return None;
                        }
                        // The grabbed point follows the cursor over the sphere it can reach, on
                        // the side of it that it started on.
                        let front = r0.dot(dir) < 0.0;
                        let w = ray.origin - c;
                        let b = dir.dot(w);
                        let disc = b * b - (w.length_squared() - radius * radius);
                        let h = if disc >= 0.0 {
                            let t = if front { -b - disc.sqrt() } else { -b + disc.sqrt() };
                            ray.origin + dir * t
                        } else {
                            c + (ray.origin + dir * -b - c).normalize_or_zero() * radius
                        };
                        let mut rot = DQuat::from_rotation_arc(r0 / radius, (h - c).normalize_or(r0 / radius));
                        if snap {
                            let (ax, ang) = rot.to_axis_angle();
                            rot = DQuat::from_axis_angle(ax, snap_angle(ang));
                        }
                        Rigid::about(rot, c)
                    }
                    (Motion::Hinge, _) | (Motion::Cylindrical, false) | (Motion::Planar, true) => turn(),
                    (Motion::Slider, _) | (Motion::Cylindrical, true) => slide(),
                    (Motion::Planar, false) => {
                        let h = ray_plane(&ray, d.grab, axis)?;
                        let v = h - d.grab;
                        Rigid::translation(v - axis * v.dot(axis))
                    }
                })
            }
        }
    }

    /// A press on a part (Pose): drag it about the connection that moves it.
    fn press_part(&mut self, pos: egui::Pos2, shift: bool, cx: &mut ToolCx) {
        let Some((id, hit)) = cx.pick(pos, None) else { return };
        let drive = cx.scene.assembly.drive(id);
        cx.scene.active = Some(id);
        let name = cx.scene.get(id).map_or_else(String::new, |o| o.name.clone());
        match &drive {
            Drive::Grounded(g) => {
                let gname = cx.scene.get(*g).map_or_else(String::new, |o| o.name.clone());
                *cx.status = if *g == id { format!("{name} is grounded: it stays put (click its ground symbol in the object browser to free it).") } else { format!("{name} is held by {gname}, which is grounded.") };
                let asm = &mut cx.scene.assembly;
                if let Some(c) = asm.parent_connection(id).map(|c| c.id) {
                    asm.selected = Some(Selected::Connection(c));
                }
                return;
            }
            Drive::Connection { connection, .. } => cx.scene.assembly.selected = Some(Selected::Connection(*connection)),
            Drive::Free(_) => {}
        }
        let parts = drive.parts();
        let mut b = Aabb::EMPTY;
        for o in parts.iter().filter_map(|&p| cx.scene.get(p)) {
            b = b.union(&o.bounds());
        }
        let centre = b.center();
        let limb = match &drive {
            Drive::Connection { frame, .. } => (centre - frame.origin).normalize_or(frame.axis()),
            _ => DVec3::Z,
        };
        self.drag = Some(Drag { part: id, drive, handle: None, grab: hit.point, start: pos, alt: shift, centre, limb, motion: Rigid::IDENTITY, generation: cx.history.generation() });
    }

    /// The selected connection's handles, in Pose: none for a fixed connection, one that came
    /// apart (Snap together puts it back first), or one held by grounded parts on both sides.
    fn gizmo(&self, scene: &Scene) -> Option<Gizmo> {
        if self.mode != Mode::Pose {
            return None;
        }
        let asm = &scene.assembly;
        let Some(Selected::Connection(cid)) = asm.selected else { return None };
        let c = asm.connection(cid)?;
        if c.kind == Motion::Fixed || c.is_apart() {
            return None;
        }
        let Some(Drive::Connection { parts, moving_child, .. }) = asm.connection_drive(cid) else { return None };
        if !parts.iter().any(|&p| scene.get(p).is_some_and(|o| o.visible)) {
            return None;
        }
        let moving = if moving_child { c.on_child() } else { c.on_parent() };
        let rot = if c.kind == Motion::Ball { moving.rot } else { c.on_parent().rot };
        Some(Gizmo { connection: cid, kind: c.kind, parts, moving_child, centre: moving.origin, axes: [rot * DVec3::X, rot * DVec3::Y, rot * DVec3::Z] })
    }

    /// Start dragging handle `h` of `g`, grabbed at `at` (on a ring), the cursor at `pos`.
    fn start_handle(&mut self, g: Gizmo, h: Handle, at: DVec3, pos: egui::Pos2, cx: &ToolCx) {
        let Some(c) = cx.scene.assembly.connection(g.connection) else { return };
        let (_, off) = c.pose();
        let ray = cx.ray_at(pos);
        let mut at = at;
        let (exact, along0, base) = match h {
            Handle::Turn(i) => {
                let exact = ray.dir.normalize().dot(g.axes[i]).abs() > 0.2;
                // Turn from under the cursor, not from the ring's nearest drawn point.
                if let Some(hit) = ray_plane(&ray, g.centre, g.axes[i]).filter(|_| exact) {
                    at = hit;
                }
                (exact, 0.0, c.angle())
            }
            Handle::Slide(i) => {
                let along = slide_along(&ray, g.centre, g.axes[i]);
                (along.is_some(), along.unwrap_or(0.0), off[i])
            }
        };
        let part = if g.moving_child { c.child } else { c.parent };
        let frame = if g.moving_child { c.on_parent() } else { c.on_child() };
        let drive = Drive::Connection { connection: g.connection, parts: g.parts.clone(), frame, moving_child: g.moving_child };
        // Turning or sliding the parent's side takes from the connection's angle and slides.
        let sign = if g.moving_child { 1.0 } else { -1.0 };
        let (centre, limb) = (g.centre, g.axes[2]);
        let hd = HandleDrag { gizmo: g, handle: h, grab: at, exact, along0, base, sign, amount: 0.0, at: pos };
        self.drag = Some(Drag { part, drive, handle: Some(hd), grab: at, start: pos, alt: false, centre, limb, motion: Rigid::IDENTITY, generation: cx.history.generation() });
    }

    /// The motion a handle drag gives the moving parts with the cursor at `pos`. ⌘: steps of
    /// 15° (1 mm), Shift: 1° (0.1 mm); a one-axis turn or a slide steps the connection's own
    /// angle or slide (what the panel shows), a ball's ring this drag's turn.
    fn handle_motion(&self, h: &mut HandleDrag, start: egui::Pos2, pos: egui::Pos2, coarse: bool, fine: bool, cx: &ToolCx) -> Rigid {
        let ray = cx.ray_at(pos);
        h.at = pos;
        let (centre, axes, kind, sign, base) = (h.gizmo.centre, h.gizmo.axes, h.gizmo.kind, h.sign, h.base);
        let stepped = |raw: f64, step: f64, absolute: bool| {
            if step <= 0.0 {
                raw
            } else if absolute {
                sign * (((base + sign * raw) / step).round() * step - base)
            } else {
                (raw / step).round() * step
            }
        };
        match h.handle {
            Handle::Turn(i) => {
                let axis = axes[i];
                let raw = if h.exact { turn_angle(&ray, h.grab, centre, axis, 0.0).unwrap_or(h.amount) } else { screen_angle(cx, h.grab, start, pos, axis, centre) };
                let step = if coarse { self.snap_deg.to_radians() } else if fine { 1f64.to_radians() } else { 0.0 };
                h.amount = stepped(raw, step, kind != Motion::Ball);
                Rigid::about(DQuat::from_axis_angle(axis, h.amount), centre)
            }
            Handle::Slide(i) => {
                let axis = axes[i];
                let raw = if h.exact { slide_along(&ray, centre, axis).map_or(h.amount, |s| s - h.along0) } else { screen_slide(cx, centre, start, pos, axis) };
                h.amount = stepped(raw, if coarse { 1.0 } else if fine { 0.1 } else { 0.0 }, true);
                Rigid::translation(axis * h.amount)
            }
        }
    }

    fn finish_drag(&mut self, cx: &mut ToolCx) {
        let Some(d) = self.drag.take() else { return };
        if d.generation != cx.history.generation() {
            return;
        }
        let parts = d.drive.parts().to_vec();
        let size = parts.iter().filter_map(|&p| cx.scene.get(p)).map(|o| o.bounds().diagonal()).fold(0.0, f64::max);
        if parts.is_empty() || d.motion.is_identity_within(size, 1e-9 * size.max(1e-9)) {
            return;
        }
        let name = cx.scene.get(d.part).map_or_else(String::new, |o| o.name.clone());
        cx.history.push(format!("Pose {name}"), cx.scene);
        assembly::move_parts(cx.scene, &parts, &d.motion);
    }

    fn check_collisions(&mut self, cx: &mut ToolCx) {
        let ids: Vec<ObjectId> = cx.scene.objects.iter().filter(|o| o.visible && !o.reference).map(|o| o.id).collect();
        let mut pairs = Vec::new();
        let mut lines = Vec::new();
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                let (Some(oa), Some(ob)) = (cx.scene.get(a), cx.scene.get(b)) else { continue };
                if !oa.bounds().intersects(&ob.bounds()) {
                    continue;
                }
                // Walk the smaller part's triangles against the larger's tree.
                let (small, big) = if oa.mesh.triangle_count() <= ob.mesh.triangle_count() { (a, b) } else { (b, a) };
                let tree = cx.scene.get_mut(big).unwrap().bvh();
                let (ms, mb) = (cx.scene.get(small).unwrap().mesh.clone(), cx.scene.get(big).unwrap().mesh.clone());
                let overlap = overlap_box(&ms.bounds(), &mb.bounds());
                let mut n = 0;
                for t in ms.triangle_ids() {
                    let ts = ms.tri_positions(t);
                    let tb = Aabb::from_points(ts);
                    if !tb.intersects(&overlap) {
                        continue;
                    }
                    tree.query_aabb(&tb, |u| {
                        let tu = mb.tri_positions(u);
                        if mm_core::intersect::triangles_intersect(ts, tu) {
                            n += 1;
                            if lines.len() < 60_000 {
                                for k in 0..3 {
                                    lines.push((ts[k], ts[(k + 1) % 3]));
                                    lines.push((tu[k], tu[(k + 1) % 3]));
                                }
                            }
                        }
                    });
                    if n > 200_000 {
                        break;
                    }
                }
                if n > 0 {
                    let joined = cx.scene.assembly.connections.iter().any(|c| c.joins(a) && c.other(a) == Some(b));
                    pairs.push((a, b, n, joined));
                }
            }
        }
        let shapes = ids.iter().filter_map(|&id| cx.scene.get(id).map(|o| (id, o.mesh.shape_version()))).collect();
        *cx.status = match pairs.iter().filter(|p| !p.3).count() {
            0 if pairs.is_empty() => "No parts cross each other.".into(),
            0 => "Only connected parts touch (at their joints).".into(),
            n => format!("{n} pair{} of parts cross each other (red).", if n == 1 { "" } else { "s" }),
        };
        self.collisions = Some(Collisions { shapes, pairs, lines });
    }

    /// The collision results, if the parts are as they were checked.
    fn current_collisions(&self, scene: &Scene) -> Option<&Collisions> {
        self.collisions.as_ref().filter(|c| c.shapes.iter().all(|&(id, v)| scene.get(id).is_some_and(|o| o.mesh.shape_version() == v)))
    }

    fn look_for_ui(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) {
        let before = (self.shape, self.tolerance_pct);
        ui.horizontal(|ui| {
            ui.label("Look for");
            egui::ComboBox::from_id_salt("assemble_shape").selected_text(self.shape.map_or("Any (best fit)", |k| k.label())).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.shape, None, "Any (best fit)");
                for k in ShapeKind::ALL {
                    ui.selectable_value(&mut self.shape, Some(k), k.label());
                }
            });
        });
        ui.add(egui::Slider::new(&mut self.tolerance_pct, 1.0..=20.0).suffix("%").text("Tolerance"))
            .on_hover_text("How far a sculpted ball or pin may stray from a true sphere or cylinder, in percent of its radius");
        if (self.shape, self.tolerance_pct) != before && !self.picks.is_empty() {
            self.refit(cx.scene);
        }
    }

    fn connect_ui(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) {
        self.look_for_ui(ui, cx);
        ui.add_space(4.0);
        let mut refine = None;
        let asm = &cx.scene.assembly;
        let mut step = |ui: &mut egui::Ui, i: usize, pick: Option<&Pick>, ask: &str| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(format!("{}.", i + 1)).color(pick_color(i)).strong());
                match pick.and_then(|p| cx.scene.get(p.object).map(|o| (p, o))) {
                    Some((p, o)) => {
                        ui.label(format!("{}: {}", o.name, p.label(asm)));
                        if matches!(p.what, PickWhat::Surface { .. }) && ui.small_button("Refine…").on_hover_text("Paint exactly this surface's triangles with Select; Use this surface fits the joint to them").clicked() {
                            refine = Some(i);
                        }
                    }
                    None => {
                        ui.label(RichText::new(ask).weak());
                    }
                }
            });
        };
        step(ui, 0, self.picks.first(), "Click a joint on one part (its marker), or a surface to make one: a ball, a pin, a peg's flat end…");
        step(ui, 1, self.picks.get(1), "Click where it fits on the other part: the socket, the hole, the recess, or a joint there…");
        if let Some(i) = refine {
            self.start_refine(RefineFor::Pick(i), cx);
        }
        match self.plan.clone() {
            Err(e) => {
                ui.label(RichText::new(e).color(APART));
            }
            Ok(None) => {}
            Ok(Some(plan)) => {
                let name = |id: ObjectId| cx.scene.get(id).map_or_else(String::new, |o| o.name.clone());
                let mv = self.picks[plan.moving].object;
                let others = plan.parts.len() - 1;
                ui.label(format!("{} moves{}.", name(mv), if others > 0 { format!(" (with {others} part{} connected to it)", if others == 1 { "" } else { "s" }) } else { String::new() }));
                for r in &plan.replaces {
                    ui.label(RichText::new(r).color(SEL));
                }
                if let Some(c) = plan.clearance {
                    if c >= 0.0 {
                        ui.label(format!("Clearance {c:.3} (the socket or hole is that much wider in radius)."));
                    } else {
                        ui.label(RichText::new(format!("Interference {:.3}: the ball or pin is that much bigger in radius.", -c)).color(APART));
                    }
                }
                let before = self.seat;
                let mut inter = plan.interaction;
                ui.horizontal(|ui| {
                    ui.label("Interaction");
                    egui::ComboBox::from_id_salt("plan_interaction").selected_text(inter.label()).show_ui(ui, |ui| {
                        for i in &plan.interactions {
                            ui.selectable_value(&mut inter, *i, i.label()).on_hover_text(i.about());
                        }
                    });
                });
                if inter != plan.interaction {
                    self.seat.interaction = Some(inter);
                    self.motion = None;
                }
                ui.small(inter.about());
                let mut kind = self.motion.unwrap_or(plan.suggested);
                ui.horizontal(|ui| {
                    ui.label("Motion");
                    egui::ComboBox::from_id_salt("plan_motion").selected_text(kind.label()).show_ui(ui, |ui| {
                        for k in Motion::ALL {
                            let text = if k == plan.suggested { format!("{} (usual)", k.label()) } else { k.label().to_string() };
                            ui.selectable_value(&mut kind, k, text).on_hover_text(k.about());
                        }
                    });
                });
                if kind != self.motion.unwrap_or(plan.suggested) {
                    self.motion = Some(kind);
                }
                ui.checkbox(&mut self.seat.move_into_place, "Move into place").on_hover_text("Seat the moving part on the other. Off: connect the parts where they are (a kit modelled assembled).");
                if self.seat.move_into_place && inter == Interaction::Tangent {
                    ui.small("The face rests on the ball at its radius: change the ball joint's radius to move it in or out.");
                } else if self.seat.move_into_place {
                    ui.horizontal(|ui| {
                        ui.label("Seat");
                        ui.selectable_value(&mut self.seat.touch, true, "Surfaces touch").on_hover_text("Move the part until its joint's surface lies on the other's (for sculpted joints that are not true spheres or cylinders)");
                        ui.selectable_value(&mut self.seat.touch, false, "Joints meet").on_hover_text("Put the joints' anchors together (the centres, axes or faces), leaving any clearance all round");
                    });
                    if self.seat.touch {
                        match &self.contact {
                            Some(c) => ui.small(format!("Gap {:.3} → {:.3} mm, deepest overlap {:.3} mm (over {} facing points). Seat offset z backs it off.", c.gap_before, c.gap_after, c.overlap, c.pairs)),
                            None => ui.small(RichText::new("The surfaces cannot be fitted (they do not face each other, or a joint does not know its surface): seated by the joints.").color(APART)),
                        };
                    }
                    ui.horizontal(|ui| {
                        if inter == Interaction::Concentric {
                            ui.checkbox(&mut self.seat.keep_turn, "Keep its turn").on_hover_text("Seat the part without turning it (otherwise its limb comes out of the socket's opening)");
                        }
                        if inter == Interaction::Flush && ui.button("Turn").on_hover_text("The next way round the faces fit (a rectangle fits two ways, a square four)").clicked() {
                            self.seat.turn += 1;
                        }
                    });
                }
                let speed = (plan.parts.len() as f64 * 0.0 + 0.01).max(1e-4);
                let mut off = self.seat.offset;
                let (changed, _) = vec3_edit(ui, "Seat offset", &mut off, speed, "Move the part from its seat, in the fixed joint's axes (z along its axis: for a ball, along the limb)");
                if changed {
                    self.seat.offset = off;
                }
                if self.seat != before {
                    // The offset moves the seat found; everything else seats again.
                    if (SeatChoice { offset: before.offset, ..self.seat }) == before {
                        self.replan(cx.scene);
                    } else {
                        self.recontact(cx.scene);
                    }
                }
            }
        }
        ui.horizontal(|ui| {
            let ready = matches!(self.plan, Ok(Some(_)));
            if ui.add_enabled(ready, egui::Button::new(RichText::new("Connect").strong())).on_hover_text("Enter").clicked() {
                self.connect(cx);
            }
            if ui.add_enabled(!self.picks.is_empty(), egui::Button::new("Clear")).on_hover_text("Esc").clicked() {
                self.picks.clear();
                self.plan = Ok(None);
            }
        });
        ui.small("Joints are drawn on the parts: click a marker to connect that joint. On a lumpy sculpt, Refine… (or Select, S) paints the surface by hand.");
    }

    fn add_joint_ui(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) {
        let before = self.shape;
        self.look_for_ui(ui, cx);
        // Choosing what to look for with a joint selected fits that joint again as it.
        if self.shape != before {
            let scene = &*cx.scene;
            if let Some(j) = scene.assembly.selected_joint().filter(|j| scene.get(j.part).is_some_and(|o| j.surface_on(&o.mesh).is_some())).map(|j| j.id) {
                cx.history.push("Joint shape", cx.scene);
                *cx.status = match assembly::refit_joint(cx.scene, j, self.shape, None) {
                    Ok(m) | Err(m) => m,
                };
            }
        }
        ui.small("Click a part's surface to give it a joint there (a ball, socket, pin, hole or flat face is recognized; Look for forces one, and fits the selected joint again as it), or paint the surface with Select (S) and use it. Then edit the joint below: its shape, radius, depth and position stay with the part.");
        if ui.button("Select a surface…").on_hover_text("S").clicked() {
            self.start_refine(RefineFor::NewJoint, cx);
        }
    }

    /// Make joint `id` a `shape`: fitted again to its surface as that shape when it is another
    /// kind of surface (a pin's triangles fitted as a ball), else (a ball made a socket, the
    /// surface not known or not fitting) the same size. Returns what was done, for the status.
    fn reshape_joint(scene: &mut Scene, id: u64, shape: JointShape) -> String {
        let Some(j) = scene.assembly.joint(id) else { return "The joint is gone.".into() };
        let old = j.shape;
        let mut why = None;
        if shape.kind().is_some() && shape.kind() != j.shape.kind() {
            match assembly::refit_joint(scene, id, shape.kind(), Some(shape)) {
                Ok(m) => return m,
                Err(e) => why = Some(e),
            }
        }
        match assembly::edit_joint(scene, id, |jj| jj.set_shape(shape)) {
            Ok(notes) => {
                scene.assembly.rename_for_shape(id, old);
                let j = scene.assembly.joint(id).map_or_else(String::new, |j| format!("{}: {}.", j.name, j.describe()));
                let kept = why.map_or_else(String::new, |e| format!("{e} Made it a {} of the same size: set its radius and place here. ", shape.label().to_lowercase()));
                format!("{kept}{j} {}", notes.join(" ")).trim_end().to_string()
            }
            Err(e) => e,
        }
    }

    /// The selected joint's editor.
    fn joint_ui(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) {
        let Some(j) = cx.scene.assembly.selected_joint().cloned() else { return };
        let part = cx.scene.get(j.part).map_or_else(String::new, |o| o.name.clone());
        ui_util::section(ui, &format!("Joint on {part}"));
        let mut name = j.name.clone();
        let mut naming = false;
        ui.horizontal(|ui| {
            ui.label("Name");
            let r = ui.text_edit_singleline(&mut name);
            if r.changed() {
                // One undo step for the whole name.
                self.begin_edit(cx, "Rename joint");
                if let Some(jm) = cx.scene.assembly.joint_mut(j.id) {
                    jm.name = name.clone();
                }
            }
            naming = r.has_focus();
        });
        let mut shape = j.shape;
        ui.horizontal(|ui| {
            ui.label("Shape");
            egui::ComboBox::from_id_salt("joint_shape").selected_text(shape.label()).show_ui(ui, |ui| {
                for s in JointShape::ALL {
                    ui.selectable_value(&mut shape, s, s.label());
                }
            });
        });
        let mut edit: Option<Box<dyn FnOnce(&mut Joint)>> = None;
        let mut dragging = naming;
        let reshape = (shape != j.shape).then_some(shape);
        let speed = (j.size().max(0.5) * 0.01).max(1e-4);
        ui.horizontal_wrapped(|ui| {
            if j.shape.round() || j.shape.axial() {
                let mut r = j.radius;
                ui.label("Radius");
                let resp = ui.add(egui::DragValue::new(&mut r).speed(speed).range(1e-4..=1e6).max_decimals(3));
                dragging |= resp.dragged();
                if resp.changed() && edit.is_none() {
                    edit = Some(Box::new(move |jj: &mut Joint| jj.radius = r));
                }
            }
            if j.shape.axial() {
                let mut l = j.length;
                ui.label("Length");
                let resp = ui.add(egui::DragValue::new(&mut l).speed(speed).range(0.0..=1e6).max_decimals(3));
                dragging |= resp.dragged();
                if resp.changed() && edit.is_none() {
                    edit = Some(Box::new(move |jj: &mut Joint| jj.length = l));
                }
            }
        });
        let mut off = j.offset;
        ui.horizontal(|ui| {
            ui.label("Depth").on_hover_text("Along the joint's axis: proud of the surface (+) or into the part (−)");
            let resp = ui.add(egui::DragValue::new(&mut off.z).speed(speed).max_decimals(3));
            dragging |= resp.dragged();
            ui.label("Nudge");
            let r1 = ui.add(egui::DragValue::new(&mut off.x).speed(speed).max_decimals(3).prefix("x "));
            let r2 = ui.add(egui::DragValue::new(&mut off.y).speed(speed).max_decimals(3).prefix("y "));
            dragging |= r1.dragged() || r2.dragged();
            if (resp.changed() || r1.changed() || r2.changed()) && edit.is_none() {
                edit = Some(Box::new(move |jj: &mut Joint| jj.offset = off));
            }
            if j.offset != DVec3::ZERO && ui.small_button("0").on_hover_text("Back where it was fitted").clicked() && edit.is_none() {
                edit = Some(Box::new(|jj: &mut Joint| jj.offset = DVec3::ZERO));
            }
        });
        if !j.fitted.is_empty() {
            ui.small(format!("Found as {}", j.fitted));
        }
        if let Some(shape) = reshape {
            self.editing = false;
            cx.history.push("Joint shape", cx.scene);
            *cx.status = Self::reshape_joint(cx.scene, j.id, shape);
        } else if let Some(e) = edit {
            self.begin_edit(cx, "Edit joint");
            match assembly::edit_joint(cx.scene, j.id, e) {
                Ok(notes) if !notes.is_empty() => *cx.status = notes.join(" "),
                Ok(_) => {}
                Err(err) => *cx.status = err,
            }
            self.editing = dragging;
        } else if !dragging {
            self.editing = false;
        }
        // What it holds.
        let held = cx.scene.assembly.connection_at(j.id).cloned();
        match &held {
            Some(c) => {
                let other = c.other(j.part).and_then(|o| cx.scene.get(o)).map_or_else(String::new, |o| o.name.clone());
                if ui.link(format!("Connected to {other} ({} · {})", c.interaction.label().to_lowercase(), c.kind.label().to_lowercase())).on_hover_text("Select the connection").clicked() {
                    cx.scene.assembly.selected = Some(Selected::Connection(c.id));
                }
            }
            None => {
                ui.small("Not connected.");
            }
        }
        ui.horizontal(|ui| {
            if ui.button("Fit again…").on_hover_text("Paint its surface with Select and fit it to that").clicked() {
                self.start_refine(RefineFor::Joint(j.id), cx);
            }
            if ui.button("Connect from here").on_hover_text("Pick this joint as the first of a connection; then click where it fits").clicked() {
                self.picks.clear();
                self.pick_joint(j.id, cx);
            }
            if ui.button("Delete joint").on_hover_text("Its connection goes too (the parts stay where they are)").clicked() {
                cx.history.push("Delete joint", cx.scene);
                cx.scene.assembly.remove_joint(j.id);
            }
        });
    }

    /// The selected connection's editor.
    fn connection_ui(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) {
        let Some(c) = cx.scene.assembly.selected_connection().cloned() else { return };
        let name = |id: ObjectId| cx.scene.get(id).map_or_else(String::new, |o| o.name.clone());
        ui_util::section(ui, &format!("Connection: {} → {}", name(c.child), name(c.parent)));
        ui.small(&c.about);
        let asm = &cx.scene.assembly;
        let (ja, jb) = (asm.joint(c.a).cloned(), asm.joint(c.b).cloned());
        let mut select = None;
        ui.horizontal_wrapped(|ui| {
            for (j, part) in [(&ja, c.part_a), (&jb, c.part_b)] {
                if let Some(j) = j {
                    if ui.link(format!("{} on {}", j.title(), name(part))).on_hover_text("Select the joint").clicked() {
                        select = Some(j.id);
                    }
                }
            }
        });
        if let Some(j) = select {
            cx.scene.assembly.selected = Some(Selected::Joint(j));
            return;
        }
        let (Some(ja), Some(jb)) = (ja, jb) else { return };
        let mut inter = c.interaction;
        let mut kind = c.kind;
        ui.horizontal(|ui| {
            ui.label("Interaction");
            egui::ComboBox::from_id_salt("connection_interaction").selected_text(inter.label()).show_ui(ui, |ui| {
                for i in Interaction::between(ja.shape, jb.shape) {
                    ui.selectable_value(&mut inter, i, i.label()).on_hover_text(i.about());
                }
            });
            ui.label("Motion");
            egui::ComboBox::from_id_salt("connection_motion").selected_text(kind.label()).show_ui(ui, |ui| {
                for k in Motion::ALL {
                    ui.selectable_value(&mut kind, k, k.label()).on_hover_text(k.about());
                }
            });
        });
        if inter != c.interaction || kind != c.kind {
            cx.history.push("Connection", cx.scene);
            if let Err(e) = assembly::edit_connection(cx.scene, c.id, |cm| {
                cm.interaction = inter;
                cm.kind = kind;
            }) {
                *cx.status = e;
            }
            return;
        }
        let (rel, off) = c.pose();
        let deg = |r: f64| r.to_degrees();
        let mut changed: Option<(DQuat, DVec3)> = None;
        let mut dragging = false;
        let speed = (c.size * 0.02).max(1e-4);
        let mut angle_box = |ui: &mut egui::Ui, label: &str, v: &mut f64| {
            ui.label(label);
            let r = ui.add(egui::DragValue::new(v).speed(0.5).suffix("°").max_decimals(1));
            dragging |= r.dragged();
            r.changed()
        };
        ui.horizontal_wrapped(|ui| match c.kind {
            Motion::Ball => {
                let (x, y, z) = rel.to_euler(EulerRot::XYZ);
                let (mut x, mut y, mut z) = (deg(x), deg(y), deg(z));
                let ch = angle_box(ui, "X", &mut x) | angle_box(ui, "Y", &mut y) | angle_box(ui, "Twist", &mut z);
                if ch {
                    changed = Some((DQuat::from_euler(EulerRot::XYZ, x.to_radians(), y.to_radians(), z.to_radians()), DVec3::ZERO));
                }
            }
            Motion::Hinge | Motion::Cylindrical | Motion::Planar => {
                let mut a = deg(c.angle());
                if angle_box(ui, "Angle", &mut a) {
                    changed = Some((DQuat::from_rotation_z(a.to_radians()), off));
                }
            }
            _ => {}
        });
        let a = c.angle();
        ui.horizontal_wrapped(|ui| match c.kind {
            Motion::Slider | Motion::Cylindrical => {
                let mut s = off.z;
                ui.label("Slide");
                let r = ui.add(egui::DragValue::new(&mut s).speed(speed).max_decimals(3));
                dragging |= r.dragged();
                if r.changed() {
                    changed = Some((if c.kind == Motion::Slider { DQuat::IDENTITY } else { DQuat::from_rotation_z(a) }, DVec3::new(0.0, 0.0, s)));
                }
            }
            Motion::Planar => {
                let (mut x, mut y) = (off.x, off.y);
                ui.label("Slide X");
                let r1 = ui.add(egui::DragValue::new(&mut x).speed(speed).max_decimals(3));
                ui.label("Y");
                let r2 = ui.add(egui::DragValue::new(&mut y).speed(speed).max_decimals(3));
                dragging |= r1.dragged() || r2.dragged();
                if r1.changed() || r2.changed() {
                    changed = Some((DQuat::from_rotation_z(a), DVec3::new(x, y, 0.0)));
                }
            }
            _ => {}
        });
        let mut seat = c.seat.shift;
        let (sc, sd) = vec3_edit(ui, "Seat offset", &mut seat, (c.size * 0.01).max(1e-4), "Move the part in its seat, in the axes of the joint it was connected to (z along that joint's axis); the pose stays");
        dragging |= sd;
        if sc {
            self.begin_edit(cx, "Move seat");
            if let Err(e) = assembly::edit_connection(cx.scene, c.id, |cm| cm.seat.shift = seat) {
                *cx.status = e;
            }
            self.editing = dragging;
        } else if let Some((r, o)) = changed {
            let m = c.motion_to(r, o);
            self.move_connection(cx, c.id, m, "Pose");
            self.editing = dragging;
        } else if !dragging {
            self.editing = false;
        }
        ui.horizontal(|ui| {
            if c.interaction == Interaction::Flush && ui.button("Turn").on_hover_text("Turn the part to the next way the faces fit").clicked() {
                let step = if ja.half.y > 0.87 * ja.half.x && jb.half.y > 0.87 * jb.half.x { std::f64::consts::FRAC_PI_2 } else { std::f64::consts::PI };
                cx.history.push("Turn", cx.scene);
                if let Err(e) = assembly::edit_connection(cx.scene, c.id, |cm| cm.seat.rot = (cm.seat.rot * DQuat::from_rotation_z(step)).normalize()) {
                    *cx.status = e;
                }
            }
            if ui.button("Fit surfaces touching").on_hover_text("Move the part in its seat until its joint's surface lies on the other's; the pose stays").clicked() {
                *cx.status = match self.fit_touching(cx, c.id) {
                    Ok(m) | Err(m) => m,
                };
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Rest pose").on_hover_text("As connected").clicked() {
                let m = c.motion_to(DQuat::IDENTITY, DVec3::ZERO);
                self.move_connection(cx, c.id, m, "Rest pose");
            }
            if c.is_apart() && ui.button(RichText::new("Snap together").color(APART)).on_hover_text("A part was moved off this connection by another tool: put it back").clicked() {
                self.move_connection(cx, c.id, c.snap_motion(), "Snap together");
            }
            if ui.button("Disconnect").on_hover_text("The joints stay on their parts, and the parts where they are").clicked() {
                cx.history.push("Disconnect", cx.scene);
                cx.scene.assembly.remove_connection(c.id);
            }
        });
        ui.small("To move the pivot or change a radius, edit a joint (the links above): the connected part follows.");
    }
}

/// Draw connection `c`'s marker (2D, on top: balls sit inside their sockets) with its frames on
/// the parent and child where they are shown: a ring (a square when fixed), the axis of a hinge
/// or slider, a red line to where the child's frame is when it came apart, the motion beside the
/// selected one.
pub fn paint_connection(painter: &egui::Painter, cx: &ToolCx, c: &Connection, sel: bool, f: JFrame, fc: JFrame) {
    let apart = (f.origin - fc.origin).length() > 1e-3 * c.size.max(1e-6) && c.is_apart();
    let col = if apart { APART } else if sel { SEL } else { JOINT };
    let Some(p) = cx.project(f.origin) else { return };
    let len = c.size * 2.5;
    if c.kind != Motion::Ball && c.kind != Motion::Fixed {
        if let (Some(a), Some(b)) = (cx.project(f.origin - f.axis() * len), cx.project(f.origin + f.axis() * len)) {
            painter.line_segment([a, b], egui::Stroke::new(if sel { 2.5 } else { 1.5 }, col));
            if matches!(c.kind, Motion::Slider | Motion::Cylindrical) {
                painter.arrow(p, b - p, egui::Stroke::new(1.5, col));
                painter.arrow(p, a - p, egui::Stroke::new(1.5, col));
            }
        }
    }
    match c.kind {
        Motion::Fixed => {
            painter.rect_stroke(egui::Rect::from_center_size(p, egui::vec2(9.0, 9.0)), 1.0, egui::Stroke::new(2.0, col), egui::StrokeKind::Middle);
        }
        _ => {
            painter.circle(p, if sel { 6.5 } else { 5.0 }, Color32::from_black_alpha(160), egui::Stroke::new(2.0, col));
        }
    }
    if apart {
        if let Some(q) = cx.project(fc.origin) {
            painter.line_segment([p, q], egui::Stroke::new(1.0, APART));
        }
    }
    if sel {
        painter.text(p + egui::vec2(9.0, -9.0), egui::Align2::LEFT_BOTTOM, format!("{} · {}", c.interaction.label(), c.kind.label()), egui::FontId::proportional(12.0), col);
    }
}

/// Draw a joint on its part as its shape seen on screen (2D, on top): a ball's or socket's
/// outline (a socket's dashed), a pin's or hole's axis and ends, a flat face's outline and
/// normal, a point's cross; its name beside it when `label`.
pub fn paint_joint(painter: &egui::Painter, cx: &ToolCx, j: &Joint, f: JFrame, col: Color32, label: bool) {
    let Some(p) = cx.project(f.origin) else { return };
    let per = cx.world_per_point(f.origin).max(1e-12);
    let stroke = egui::Stroke::new(1.5, col);
    let dashed_circle = |r: f32| {
        let n = 24;
        for i in (0..n).step_by(2) {
            let (a0, a1) = (i as f32 / n as f32 * std::f32::consts::TAU, (i + 1) as f32 / n as f32 * std::f32::consts::TAU);
            painter.line_segment([p + egui::vec2(a0.cos(), a0.sin()) * r, p + egui::vec2(a1.cos(), a1.sin()) * r], stroke);
        }
    };
    match j.shape {
        JointShape::Ball => {
            painter.circle_stroke(p, ((j.radius / per) as f32).clamp(3.0, 400.0), stroke);
        }
        JointShape::Socket => dashed_circle(((j.radius / per) as f32).clamp(3.0, 400.0)),
        JointShape::Pin | JointShape::Hole => {
            let h = f.axis() * (0.5 * j.length.max(j.radius));
            if let (Some(a), Some(b)) = (cx.project(f.origin - h), cx.project(f.origin + h)) {
                painter.line_segment([a, b], egui::Stroke::new(if j.shape == JointShape::Pin { 2.5 } else { 1.0 }, col));
                for e in [a, b] {
                    painter.circle_stroke(e, ((j.radius / per) as f32).clamp(2.0, 200.0) * 0.5, stroke);
                }
            }
        }
        JointShape::Flat => {
            let (u, v) = (f.rot * DVec3::X, f.rot * DVec3::Y);
            let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(a, b)| cx.project(f.origin + u * (a * j.half.x) + v * (b * j.half.y)));
            if corners.iter().all(Option::is_some) {
                let c: Vec<egui::Pos2> = corners.iter().map(|x| x.unwrap()).collect();
                painter.add(egui::Shape::closed_line(c, stroke));
            }
            if let Some(n) = cx.project(f.origin + f.axis() * j.half.x.max(j.half.y)) {
                painter.line_segment([p, n], stroke);
            }
        }
        JointShape::Point => {
            painter.line_segment([p - egui::vec2(5.0, 0.0), p + egui::vec2(5.0, 0.0)], stroke);
            painter.line_segment([p - egui::vec2(0.0, 5.0), p + egui::vec2(0.0, 5.0)], stroke);
        }
    }
    painter.circle_filled(p, 2.5, col);
    if label {
        painter.text(p + egui::vec2(8.0, 8.0), egui::Align2::LEFT_TOP, &j.name, egui::FontId::proportional(12.0), col);
    }
}

/// A small picture of a joint's shape, for lists.
pub fn joint_glyph(ui: &mut egui::Ui, shape: JointShape, col: Color32) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    let s = egui::Stroke::new(1.4, col);
    match shape {
        JointShape::Ball => {
            p.circle_filled(c, 4.5, col);
        }
        JointShape::Socket => {
            p.circle_stroke(c, 4.5, s);
        }
        JointShape::Pin => {
            p.line_segment([c + egui::vec2(0.0, -6.0), c + egui::vec2(0.0, 6.0)], egui::Stroke::new(3.0, col));
        }
        JointShape::Hole => {
            p.rect_stroke(egui::Rect::from_center_size(c, egui::vec2(5.0, 12.0)), 1.0, s, egui::StrokeKind::Middle);
        }
        JointShape::Flat => {
            p.rect_stroke(egui::Rect::from_center_size(c, egui::vec2(11.0, 6.0)), 0.0, s, egui::StrokeKind::Middle);
        }
        JointShape::Point => {
            p.line_segment([c + egui::vec2(-5.0, 0.0), c + egui::vec2(5.0, 0.0)], s);
            p.line_segment([c + egui::vec2(0.0, -5.0), c + egui::vec2(0.0, 5.0)], s);
        }
    }
    r
}

/// A small ground-symbol toggle.
pub fn ground_toggle(ui: &mut egui::Ui, on: bool) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::click());
    let col = if on {
        ui_util::ACCENT
    } else if r.hovered() {
        ui.visuals().text_color()
    } else {
        ui.visuals().weak_text_color().gamma_multiply(0.6)
    };
    let p = ui.painter();
    let s = egui::Stroke::new(if on { 2.0 } else { 1.4 }, col);
    let c = rect.center();
    p.line_segment([c + egui::vec2(0.0, -7.0), c + egui::vec2(0.0, 1.0)], s);
    for (i, w) in [6.5f32, 4.5, 2.5].into_iter().enumerate() {
        let y = 1.0 + 3.0 * i as f32;
        p.line_segment([c + egui::vec2(-w, y), c + egui::vec2(w, y)], s);
    }
    r
}

/// Three drag values for a vector, with a label. Returns (changed, being dragged).
fn vec3_edit(ui: &mut egui::Ui, label: &str, v: &mut DVec3, speed: f64, hint: &str) -> (bool, bool) {
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hint);
        let mut changed = false;
        let mut dragging = false;
        for (axis, x) in [("x", &mut v.x), ("y", &mut v.y), ("z", &mut v.z)] {
            let r = ui.add(egui::DragValue::new(x).speed(speed).max_decimals(3).prefix(format!("{axis} "))).on_hover_text(hint);
            changed |= r.changed();
            dragging |= r.dragged();
        }
        if *v != DVec3::ZERO && ui.small_button("0").on_hover_text("Back to none").clicked() {
            *v = DVec3::ZERO;
            changed = true;
        }
        (changed, dragging)
    })
    .inner
}

/// What connecting the two picks would do: the free part's joint seated on the other's (by
/// `contact`, a touching fit, if given), moved by the seat offset.
fn plan(scene: &Scene, picks: &[Pick], seat: &SeatChoice, contact: Option<&Contact>) -> Result<Option<Plan>, String> {
    let [p0, p1] = picks else { return Ok(None) };
    if p0.object == p1.object {
        return Err("Pick the second joint on another part.".into());
    }
    if [p0, p1].iter().any(|p| scene.get(p.object).is_none_or(|o| !Arc::ptr_eq(&o.mesh, &p.mesh))) {
        return Err("A part changed since it was clicked: click it again.".into());
    }
    let asm = &scene.assembly;
    // The connections the picked joints hold are replaced.
    let mut after = asm.clone();
    let mut replaces = Vec::new();
    for p in [p0, p1] {
        if let PickWhat::Joint(j) = p.what {
            if let Some(c) = after.connection_at(j).cloned() {
                let other = c.other(p.object).and_then(|o| scene.get(o)).map_or_else(String::new, |o| o.name.clone());
                let jn = asm.joint(j).map_or_else(String::new, |j| j.title());
                replaces.push(format!("{jn} holds {other}: connecting replaces that connection."));
                after.remove_connection(c.id);
            }
        }
    }
    if after.connected(p0.object, p1.object) {
        return Err("These parts are already connected (perhaps through others): disconnect them first.".into());
    }
    let held = |id: ObjectId| after.component(id).iter().any(|&p| after.is_grounded(p));
    let moving = match (held(p0.object), held(p1.object)) {
        (false, _) => 0,
        (true, false) => 1,
        (true, true) => return Err("Both parts are held by grounded parts: free one (the ground symbol in the object browser) to move it into place.".into()),
    };
    let (mv, fx) = (&picks[moving], &picks[1 - moving]);
    let (Some(ja), Some(jb)) = (fx.joint(asm), mv.joint(asm)) else { return Err("A picked joint is gone: pick it again.".into()) };
    let interactions = Interaction::between(ja.shape, jb.shape);
    let interaction = seat.interaction.filter(|i| interactions.contains(i)).unwrap_or(interactions[0]);
    let (fa, fb) = anchors(&ja, &jb, interaction);
    let mut s = if seat.move_into_place {
        let mut s = initial_seat(&ja, &jb, interaction, seat.keep_turn, seat.turn);
        if let Some(c) = contact {
            // Where the touching fit puts b's anchor.
            s = fa.local_to(&fb.moved(&Rigid { rot: c.rot, shift: c.shift }));
        }
        s
    } else {
        // As they are.
        fa.local_to(&fb)
    };
    s.shift += seat.offset;
    let motion = Rigid::between(&fb, &fa.then_local(&s));
    let clearance = match (ja.shape, jb.shape) {
        (JointShape::Socket, JointShape::Ball) | (JointShape::Hole, JointShape::Pin) => Some(ja.radius - jb.radius),
        (JointShape::Ball, JointShape::Socket) | (JointShape::Pin, JointShape::Hole) => Some(jb.radius - ja.radius),
        _ => None,
    };
    Ok(Some(Plan {
        moving,
        parts: after.component(mv.object),
        motion,
        interactions,
        interaction,
        suggested: interaction.motion(&ja, &jb),
        seat: s,
        pivot: fa.then_local(&s).origin,
        clearance,
        replaces,
    }))
}

fn overlap_box(a: &Aabb, b: &Aabb) -> Aabb {
    Aabb { min: a.min.max(b.min), max: a.max.min(b.max) }
}

fn ray_plane(ray: &mm_core::Ray, p0: DVec3, n: DVec3) -> Option<DVec3> {
    let d = ray.dir.normalize();
    let den = d.dot(n);
    if den.abs() < 1e-9 {
        return None;
    }
    let t = (p0 - ray.origin).dot(n) / den;
    (t > 0.0).then(|| ray.origin + d * t)
}

/// The turn about `axis` (through `c`) that brings the grabbed point under the ray, when the axis
/// is not too side-on to the view (the cosine between them at least `min_cos`).
fn turn_angle(ray: &mm_core::Ray, grab: DVec3, c: DVec3, axis: DVec3, min_cos: f64) -> Option<f64> {
    if ray.dir.normalize().dot(axis).abs() < min_cos {
        return None;
    }
    let cp = c + axis * (grab - c).dot(axis);
    let r0 = grab - cp;
    if r0.length_squared() < 1e-24 {
        return None;
    }
    let h = ray_plane(ray, cp, axis)?;
    let r1 = h - cp;
    Some(r0.cross(r1).dot(axis).atan2(r0.dot(r1)))
}

/// A turn about `axis` (through `about`) from the cursor's move along the way the grabbed point
/// goes on screen (for an axis seen side-on).
fn screen_angle(cx: &ToolCx, grab: DVec3, start: egui::Pos2, pos: egui::Pos2, axis: DVec3, about: DVec3) -> f64 {
    let cp = about + axis * (grab - about).dot(axis);
    let r0 = grab - cp;
    let eps = 0.01;
    let (Some(a), Some(b)) = (cx.project(grab), cx.project(cp + DQuat::from_axis_angle(axis, eps) * r0)) else { return 0.0 };
    let per_radian = (b - a) / eps as f32;
    if per_radian.length_sq() < 1e-6 {
        // The point turns straight towards the viewer: use the sideways move.
        return (pos - start).x as f64 * 0.01;
    }
    ((pos - start).dot(per_radian) / per_radian.length_sq()) as f64
}

/// How far along `axis` (through `grab`) the ray passes closest to it, unless they are nearly
/// parallel.
fn slide_along(ray: &mm_core::Ray, grab: DVec3, axis: DVec3) -> Option<f64> {
    let d = ray.dir.normalize();
    let b = axis.dot(d);
    let den = 1.0 - b * b;
    if den < 0.05 {
        return None;
    }
    let w = grab - ray.origin;
    Some((b * d.dot(w) - axis.dot(w)) / den)
}

/// A slide along `axis` from the cursor's move along the axis on screen.
fn screen_slide(cx: &ToolCx, grab: DVec3, start: egui::Pos2, pos: egui::Pos2, axis: DVec3) -> f64 {
    let unit = cx.world_per_point(grab) * 50.0;
    let (Some(a), Some(b)) = (cx.project(grab), cx.project(grab + axis * unit)) else { return 0.0 };
    let per_unit = (b - a) / unit as f32;
    if per_unit.length_sq() < 1e-9 {
        return -(pos - start).y as f64 * cx.world_per_point(grab);
    }
    ((pos - start).dot(per_unit) / per_unit.length_sq()) as f64
}

fn rgba(c: Color32) -> Vec4 {
    Vec4::new(c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, 1.0)
}

/// Wire outline of a recognized shape, and the edges round its triangles.
fn draw_feature(ov: &mut Overlay, mesh: &Mesh, f: &Feature, m: &Rigid, col: Vec4) {
    let at = |p: DVec3| m.apply(p);
    let circle = |ov: &mut Overlay, c: DVec3, n: DVec3, r: f64| {
        let (u, v) = mm_core::geom::orthonormal_basis(n);
        let k = 40;
        for i in 0..k {
            let (a0, a1) = (i as f64 / k as f64 * std::f64::consts::TAU, (i + 1) as f64 / k as f64 * std::f64::consts::TAU);
            ov.line(at(c + (u * a0.cos() + v * a0.sin()) * r), at(c + (u * a1.cos() + v * a1.sin()) * r), col);
        }
    };
    match f.shape {
        Shape::Sphere { centre, radius } => {
            for n in [DVec3::X, DVec3::Y, DVec3::Z] {
                circle(ov, centre, n, radius);
            }
        }
        Shape::Cylinder { centre, axis, radius, length } => {
            let h = axis * (0.5 * length);
            circle(ov, centre - h, axis, radius);
            circle(ov, centre + h, axis, radius);
            let (u, v) = mm_core::geom::orthonormal_basis(axis);
            for d in [u, -u, v, -v] {
                ov.line(at(centre - h + d * radius), at(centre + h + d * radius), col);
            }
            ov.line(at(centre - h * 1.6), at(centre + h * 1.6), col);
        }
        Shape::Plane { centre, normal, u, half, .. } => {
            let v = normal.cross(u);
            let p = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(a, b)| centre + u * (a * half.x) + v * (b * half.y));
            for i in 0..4 {
                ov.line(at(p[i]), at(p[(i + 1) % 4]), col);
            }
            ov.line(at(centre), at(centre + normal * half.x.max(half.y)), col);
        }
    }
    // The recognized surface's border.
    let inside: mm_core::hash::FastSet<u32> = f.tris.iter().copied().collect();
    let mut n = 0;
    for &t in &f.tris {
        if !mesh.is_triangle(t) {
            continue;
        }
        let tv = mesh.triangle(t);
        for (k, nb) in mesh.tri_neighbors(t).into_iter().enumerate() {
            if nb == mm_core::INVALID || !inside.contains(&nb) {
                ov.line(at(mesh.vertex(tv[k])), at(mesh.vertex(tv[(k + 1) % 3])), col);
                n += 1;
            }
        }
        if n > 20_000 {
            break;
        }
    }
}

impl Tool for AssembleTool {
    fn name(&self) -> &'static str {
        "Assemble"
    }

    fn survives_scene_change(&self) -> bool {
        true
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        assembly::follow(cx.scene);
        // Back from Select: use the surface chosen there.
        if let Some((target, used)) = self.refining.take() {
            *cx.status = if used.load(Ordering::Relaxed) {
                match self.use_selection(target, cx) {
                    Ok(m) | Err(m) => m,
                }
            } else {
                "The selection was not used.".into()
            };
        }
        // Picks on parts that changed (an undo, another tool) or joints that went are dropped.
        let n = self.picks.len();
        let asm = &cx.scene.assembly;
        self.picks.retain(|p| cx.scene.get(p.object).is_some_and(|o| Arc::ptr_eq(&o.mesh, &p.mesh)) && !matches!(p.what, PickWhat::Joint(j) if asm.joint(j).is_none()));
        // Grounding, joints or connections changed (the object browser, a delete, an undo):
        // which part moves, and what may be connected, may have changed too.
        let seen = seen_of(&cx.scene.assembly);
        if self.picks.len() != n || seen != self.seen {
            self.seen = seen;
            self.recontact(cx.scene);
        }
        ui_util::section(ui, "Assemble");
        ui.horizontal(|ui| {
            let mut mode = self.mode;
            ui.selectable_value(&mut mode, Mode::Pose, RichText::new("Pose").strong()).on_hover_text("Drag parts to turn them about their connections");
            ui.selectable_value(&mut mode, Mode::Connect, RichText::new("Connect").strong()).on_hover_text("Connect a joint on one part to a joint on another");
            ui.selectable_value(&mut mode, Mode::AddJoint, RichText::new("Add joint").strong()).on_hover_text("Give a part a joint (a ball, socket, pin, hole, flat face) without connecting it yet");
            if mode != self.mode {
                self.mode = mode;
                self.drag = None;
                self.hover_handle = None;
            }
        });
        match self.mode {
            Mode::Pose => {
                ui.small("Drag a part to turn it about its connection; what hangs from it follows. Shift+drag: twist a ball, slide, turn on a face. ⌘: steps of 15°. A part with no connection moves freely (Shift: turns).");
                ui.small("The selected connection has handles for what its motion allows (a ball: rings for X, Y and twist; a hinge: one ring; a slider: an arrow). Drag one for that turn or slide alone; ⌘: 15° (1 mm) steps, Shift: 1° (0.1 mm).");
            }
            Mode::Connect => self.connect_ui(ui, cx),
            Mode::AddJoint => self.add_joint_ui(ui, cx),
        }
        if cx.scene.objects.iter().all(|o| o.reference) {
            ui.label("Import the kit's parts first (one file each, or Separate Shells).");
        } else if cx.scene.assembly.selected.is_none() {
            ui.small("The object browser shows each part's joints and what is connected to them: click a joint or a connection there to edit it.");
        }
        self.joint_ui(ui, cx);
        self.connection_ui(ui, cx);
        ui_util::section(ui, "Check");
        ui.horizontal(|ui| {
            if ui.add_enabled(!cx.scene.assembly.connections.is_empty(), egui::Button::new("Reset all to rest")).on_hover_text("Every connection back as connected").clicked() {
                self.reset_all(cx);
            }
            if ui.button("Check collisions").on_hover_text("Find parts that cross each other in this pose").clicked() {
                self.check_collisions(cx);
            }
        });
        let scene = &*cx.scene;
        if let Some(c) = self.collisions.as_ref() {
            let stale = self.current_collisions(scene).is_none();
            for &(a, b, n, joined) in &c.pairs {
                let name = |id: ObjectId| scene.get(id).map_or("?".to_string(), |o| o.name.clone());
                let text = format!("{} × {}: {n} crossing triangles{}", name(a), name(b), if joined { " (connected: at the joint)" } else { "" });
                ui.label(if joined || stale { RichText::new(text).weak() } else { RichText::new(text).color(APART) });
            }
            if stale {
                ui.small("The parts moved since: check again.");
            }
        }
        ui.add_space(6.0);
        if ui.button(RichText::new("Done").strong()).on_hover_text("Esc").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else {
            self.hover = None;
            return;
        };
        if self.drag.is_none() && !input.down {
            self.hover = self.pick_shown(pos, cx).map(|(id, _)| (id, pos));
        }
        match self.mode {
            Mode::Connect | Mode::AddJoint => {
                if input.pressed {
                    self.pressed_at = Some(pos);
                }
                if input.released && self.pressed_at.take().is_some_and(|p| p.distance(pos) < 5.0) {
                    if self.mode == Mode::Connect {
                        self.click_connect(pos, cx);
                    } else if let Some(j) = self.joint_marker_at(pos, cx) {
                        cx.scene.assembly.selected = Some(Selected::Joint(j));
                    } else if let Some((id, tid)) = self.pick_shown(pos, cx) {
                        if let Err(e) = self.add_joint_at(id, tid, cx) {
                            *cx.status = e;
                        }
                    }
                }
            }
            Mode::Pose => {
                if self.drag.is_none() && !input.down {
                    self.hover_handle = self.gizmo(cx.scene).and_then(|g| g.handle_at(pos, cx)).map(|(h, _)| (h, pos));
                }
                if input.pressed {
                    assembly::follow(cx.scene);
                    match self.gizmo(cx.scene).and_then(|g| g.handle_at(pos, cx).map(|(h, at)| (g, h, at))) {
                        Some((g, h, at)) => self.start_handle(g, h, at, pos, cx),
                        None => self.press_part(pos, input.shift, cx),
                    }
                }
                if input.down {
                    if let Some(mut d) = self.drag.take() {
                        if let Some(h) = d.handle.as_mut() {
                            d.motion = self.handle_motion(h, d.start, pos, input.command, input.shift, cx);
                        } else if let Some(m) = self.drag_motion(&d, pos, input.command, cx) {
                            d.motion = m;
                        }
                        self.drag = Some(d);
                    }
                }
                if input.released {
                    self.finish_drag(cx);
                }
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        // Picked surfaces (the moving one where Connect would put it).
        let plan = self.plan.as_ref().ok().and_then(|p| p.as_ref());
        for (i, p) in self.picks.iter().enumerate() {
            let m = match plan {
                Some(pl) if pl.moving == i && self.mode == Mode::Connect => pl.motion,
                _ => Rigid::IDENTITY,
            };
            if let PickWhat::Surface { feature, .. } = &p.what {
                draw_feature(ov, &p.mesh, feature, &m, PICK[i]);
            }
        }
        if let Some(c) = self.current_collisions(cx.scene) {
            let red = rgba(APART);
            for &(a, b) in &c.lines {
                ov.line(a, b, red);
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        // Where things are drawn: moved by a drag, or by the Connect preview.
        let plan = self.plan.as_ref().ok().and_then(|p| p.as_ref());
        let live = match (&self.drag, plan, self.mode) {
            (Some(d), _, _) => Some((d.drive.parts().to_vec(), d.motion)),
            (None, Some(pl), Mode::Connect) => Some((pl.parts.clone(), pl.motion)),
            _ => None,
        };
        let moved = |part: ObjectId, f: &JFrame| match &live {
            Some((parts, m)) if parts.contains(&part) => f.moved(m),
            _ => *f,
        };
        let asm = &cx.scene.assembly;
        // Joints on the parts.
        let picked: Vec<(usize, u64)> = self.picks.iter().enumerate().filter_map(|(i, p)| match p.what {
            PickWhat::Joint(j) => Some((i, j)),
            _ => None,
        }).collect();
        for j in &asm.joints {
            if cx.scene.get(j.part).is_none_or(|o| !o.visible) {
                continue;
            }
            let sel = asm.selected == Some(Selected::Joint(j.id));
            let pick = picked.iter().find(|(_, id)| *id == j.id).map(|(i, _)| *i);
            let col = match pick {
                Some(i) => pick_color(i),
                None if sel => SEL,
                None if asm.connection_at(j.id).is_some() => JOINT.gamma_multiply(0.6),
                None => FREE,
            };
            paint_joint(painter, cx, j, moved(j.part, &j.frame()), col, sel || pick.is_some() || self.mode == Mode::AddJoint);
        }
        // Connections, on top (balls sit inside their sockets).
        for c in &asm.connections {
            let sel = asm.selected == Some(Selected::Connection(c.id));
            paint_connection(painter, cx, c, sel, moved(c.parent, &c.on_parent()), moved(c.child, &c.on_child()));
        }
        // Where the picked surfaces are.
        for (i, p) in self.picks.iter().enumerate() {
            let PickWhat::Surface { feature, .. } = &p.what else { continue };
            let m = match plan {
                Some(pl) if pl.moving == i && self.mode == Mode::Connect => pl.motion,
                _ => Rigid::IDENTITY,
            };
            if let Some(c) = cx.project(m.apply(feature.shape.origin())) {
                painter.circle_filled(c, 4.0, pick_color(i));
                painter.text(c + egui::vec2(7.0, 7.0), egui::Align2::LEFT_TOP, format!("{}: {}", i + 1, feature.name()), egui::FontId::proportional(12.0), pick_color(i));
            }
        }
        // Where the connection will turn.
        if let (Mode::Connect, Some(pl)) = (self.mode, plan) {
            if let Some(c) = cx.project(pl.pivot) {
                let s = egui::Stroke::new(2.0, SEL);
                painter.line_segment([c - egui::vec2(6.0, 0.0), c + egui::vec2(6.0, 0.0)], s);
                painter.line_segment([c - egui::vec2(0.0, 6.0), c + egui::vec2(0.0, 6.0)], s);
                painter.text(c + egui::vec2(8.0, -8.0), egui::Align2::LEFT_BOTTOM, "pivot", egui::FontId::proportional(12.0), SEL);
            }
        }
        // The selected connection's handles, moved with a drag of its parts.
        let gizmo = match &self.drag {
            Some(Drag { handle: Some(h), motion, .. }) => Some((h.gizmo.moved(motion), Some(h.handle))),
            Some(d) => self.gizmo(cx.scene).filter(|g| d.drive.parts() == g.parts.as_slice()).map(|g| (g.moved(&d.motion), None)),
            None => self.gizmo(cx.scene).map(|g| (g, self.hover_handle.map(|h| h.0))),
        };
        if let Some((g, active)) = &gizmo {
            g.paint(painter, cx, *active);
        }
        if let Some(Drag { handle: Some(h), .. }) = &self.drag {
            if let (Handle::Turn(i), Some(c)) = (h.handle, cx.project(h.gizmo.centre)) {
                // The wedge turned through, from where the ring was grabbed.
                let axis = h.gizmo.axes[i];
                let (r, _) = h.gizmo.sizes(cx);
                let v = h.grab - h.gizmo.centre;
                let r0 = (v - axis * v.dot(axis)).normalize_or_zero() * r;
                let n = ((h.amount.abs().to_degrees() / 3.0).ceil() as usize).clamp(2, 120);
                let pts: Vec<egui::Pos2> = (0..=n).filter_map(|k| cx.project(h.gizmo.centre + DQuat::from_axis_angle(axis, h.amount * k as f64 / n as f64) * r0)).collect();
                if r0 != DVec3::ZERO && pts.len() > 1 {
                    let s = egui::Stroke::new(1.5, Color32::YELLOW);
                    painter.line_segment([c, pts[0]], s);
                    painter.line_segment([c, pts[pts.len() - 1]], s);
                    painter.add(egui::Shape::line(pts, egui::Stroke::new(5.0, Color32::YELLOW.gamma_multiply(0.6))));
                }
            }
            plate_text(painter, h.at + egui::vec2(18.0, -26.0), h.readout(), Color32::YELLOW);
        }
        // What a drag on a handle, or on the part under the cursor, would do.
        if let (Mode::Pose, None, Some((h, pos)), Some((g, _))) = (self.mode, &self.drag, self.hover_handle, &gizmo) {
            let how = match h {
                Handle::Turn(_) => "drag to turn · ⌘ 15° steps · Shift 1°",
                Handle::Slide(_) => "drag to slide · ⌘ 1 mm steps · Shift 0.1 mm",
            };
            plate_text(painter, pos + egui::vec2(16.0, -24.0), format!("{}: {how}", g.name(h)), Color32::from_gray(235));
        } else if self.mode == Mode::Pose && self.drag.is_none() {
            if let Some((id, pos)) = self.hover {
                let what = match asm.drive(id) {
                    Drive::Grounded(g) if g == id => "grounded".to_string(),
                    Drive::Grounded(_) => "held by a grounded part".to_string(),
                    Drive::Free(p) if p.len() > 1 => format!("free: moves with {} connected parts", p.len() - 1),
                    Drive::Free(_) => "free: drag to move, Shift+drag to turn".to_string(),
                    Drive::Connection { connection, .. } => asm.connection(connection).map_or_else(String::new, |c| format!("{} · {}", c.interaction.label().to_lowercase(), c.kind.label().to_lowercase())),
                };
                if let Some(o) = cx.scene.get(id) {
                    painter.text(pos + egui::vec2(14.0, -6.0), egui::Align2::LEFT_BOTTOM, format!("{} · {what}", o.name), egui::FontId::proportional(12.0), Color32::from_gray(230));
                }
            }
        }
    }

    fn take_request(&mut self) -> Option<Box<dyn Tool>> {
        self.request.take()
    }

    /// The Assembly toolbar's buttons: `connect`, `pose`, `addjoint`, `select`, `collisions`.
    fn script(&mut self, cmd: &str, cx: &mut ToolCx) {
        match cmd {
            "connect" => self.mode = Mode::Connect,
            "pose" => self.mode = Mode::Pose,
            "addjoint" => self.mode = Mode::AddJoint,
            // Select a surface by hand: a new joint (Add joint) or the next pick.
            "select" if self.mode == Mode::AddJoint => self.start_refine(RefineFor::NewJoint, cx),
            "select" => {
                self.mode = Mode::Connect;
                self.start_refine(RefineFor::Pick(self.picks.len().min(2)), cx)
            }
            "collisions" => self.check_collisions(cx),
            _ => {}
        }
        self.drag = None;
    }

    fn rigid_previews(&self) -> Vec<(ObjectId, glam::DMat4)> {
        if let Some(d) = &self.drag {
            let m = d.motion.mat4();
            return d.drive.parts().iter().map(|&p| (p, m)).collect();
        }
        match (&self.plan, self.mode) {
            (Ok(Some(p)), Mode::Connect) if p.motion != Rigid::IDENTITY => p.parts.iter().map(|&id| (id, p.motion.mat4())).collect(),
            _ => Vec::new(),
        }
    }

    fn cancel(&mut self, _cx: &mut ToolCx) -> bool {
        if self.drag.take().is_some() {
            return false;
        }
        if !self.picks.is_empty() {
            self.picks.clear();
            self.plan = Ok(None);
            return false;
        }
        true
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.mode == Mode::Connect && matches!(self.plan, Ok(Some(_))) {
            self.connect(cx);
            return false;
        }
        true
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        let motions: Vec<&str> = Motion::ALL.iter().map(|k| k.label()).collect();
        let interactions: Vec<&str> = Interaction::ALL.iter().map(|k| k.label()).collect();
        let shapes: Vec<&str> = JointShape::ALL.iter().map(|k| k.key()).collect();
        let at = || arg::point("A point on or near the part's surface, mm");
        let part = || arg::integer("The part's object id (default: the nearest visible part)");
        vec![
            Action::new("pick_surface", "Pick for connecting: the joint made of the surface nearest `at`, or else that surface (recognized: ball, socket, pin, hole, flat face), to become a joint; the second pick plans the connection", json!({"at": at(), "part": part()})).required(&["at"]),
            Action::new("pick_joint", "Pick a joint for connecting", json!({"joint": arg::integer("Joint id")})).required(&["joint"]),
            Action::new("connect", "Connect the two picks: the part not held by a grounded part is seated on the other; new surfaces become joints; a connection a picked joint held is replaced", json!({"interaction": arg::choice(&interactions, "Interaction (default: the usual one for the joints' shapes)"), "motion": arg::choice(&motions, "Motion (default: the usual one)"), "move_into_place": arg::boolean("Seat the moving part (default true); false connects the parts where they are")})),
            Action::new("seat", "How the picked parts are seated before connecting", json!({"mode": arg::choice(&["touch", "joints"], "Surfaces touch (fit one surface onto the other) or the joints' anchors meet"), "offset": arg::point("Move the part from its seat, mm, in the fixed joint's axes (z along its axis)"), "keep_turn": arg::boolean("A ball's part keeps its turn"), "turn": arg::integer("Which way round two flat faces meet (0 = nearest)")})),
            Action::new("add_joint", "Give a part a joint at the surface nearest `at` (recognized), without connecting it", json!({"at": at(), "part": part(), "shape": arg::choice(&shapes, "Make it this shape instead of what was recognized")})).required(&["at"]),
            Action::new("edit_joint", "Change a joint (connected parts are re-seated, keeping their pose)", json!({"joint": arg::integer("Joint id"), "shape": arg::choice(&shapes, "Shape"), "radius": arg::number("Radius, mm"), "length": arg::number("Pin / hole length, mm"), "depth": arg::number("Along its axis from where it was found: proud (+) or into the part (−), mm"), "nudge_x": arg::number("Across its axis (its x), mm"), "nudge_y": arg::number("Across its axis (its y), mm"), "name": arg::text("Name")})).required(&["joint"]),
            Action::new("delete_joint", "Remove a joint and its connection (the parts stay where they are)", json!({"joint": arg::integer("Joint id")})).required(&["joint"]),
            Action::new("disconnect", "Remove a connection (the joints stay)", json!({"connection": arg::integer("Connection id")})).required(&["connection"]),
            Action::new("pose", "Set a connection's pose relative to its rest pose (as connected): angles in degrees, slides in mm; what its motion does not allow is ignored", json!({"connection": arg::integer("Connection id"), "x": arg::number("Ball: turn about the anchor's x"), "y": arg::number("Ball: turn about y"), "twist": arg::number("Ball: twist about z"), "angle": arg::number("Hinge, cylindrical, planar: turn about the axis"), "slide": arg::number("Slider, cylindrical: along the axis")})).required(&["connection"]),
            Action::new("set_connection", "Change a connection's interaction, motion or seat offset (the pose keeps what the motion allows)", json!({"connection": arg::integer("Connection id"), "interaction": arg::choice(&interactions, "Interaction"), "motion": arg::choice(&motions, "Motion"), "offset": arg::point("Seat offset, mm, in the axes of the joint it was connected to (its joints[0])")})).required(&["connection"]),
            Action::new("fit_touching", "Move a connection's part in its seat until its joints' surfaces touch", json!({"connection": arg::integer("Connection id")})).required(&["connection"]),
            Action::new("rest_pose", "Put a connection (or every connection) back as connected", json!({"connection": arg::integer("Connection id (default: every connection)")})),
            Action::new("ground", "Ground a part (it stays put while others are posed) or free it", json!({"part": arg::integer("Object id"), "on": arg::boolean("Grounded (default true)")})).required(&["part"]),
            Action::new("refine_from_selection", "Fit a pick to its part's current selection (select the triangles first with mesh_select); the pick after the last makes a new pick from the active part's selection", json!({"pick": arg::integer("1 or 2")})).required(&["pick"]),
            Action::plain("check_collisions", "Find parts that cross each other in this pose"),
            Action::plain("clear", "Forget the picks"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        let motion_of = |s: &str| Motion::ALL.into_iter().find(|k| k.label().eq_ignore_ascii_case(s)).ok_or(format!("no motion {s}"));
        let inter_of = |s: &str| Interaction::from_label(s).ok_or(format!("no interaction {s}"));
        let shape_of = |s: &str| JointShape::from_key(s).ok_or(format!("no joint shape {s}"));
        let joint_id = |cx: &ToolCx| -> Result<u64, String> {
            let id = a.usize("joint")? as u64;
            cx.scene.assembly.joint(id).map(|j| j.id).ok_or(format!("there is no joint {id}"))
        };
        let conn_id = |cx: &ToolCx| -> Result<u64, String> {
            let id = a.usize("connection")? as u64;
            cx.scene.assembly.connection(id).map(|c| c.id).ok_or(format!("there is no connection {id}"))
        };
        // The part's triangle nearest `at`.
        let nearest = |cx: &mut ToolCx| -> Result<(ObjectId, u32), String> {
            let at = DVec3::from_array(a.point("at")?);
            let only = a.opt_usize("part").map(|p| p as ObjectId);
            let mut best: Option<(f64, ObjectId, u32)> = None;
            for o in cx.scene.objects.iter_mut().filter(|o| o.visible && !o.reference && only.is_none_or(|p| p == o.id)) {
                let bvh = o.bvh();
                if let Some(h) = bvh.nearest(&o.mesh, at, f64::INFINITY) {
                    if best.is_none_or(|b| h.dist_sq < b.0) {
                        best = Some((h.dist_sq, o.id, h.tid));
                    }
                }
            }
            best.map(|b| (b.1, b.2)).ok_or("there is no such visible part".into())
        };
        let planned = |t: &Self| match &t.plan {
            Err(e) => Err(e.clone()),
            Ok(_) => Ok(String::new()),
        };
        match action {
            "pick_surface" => {
                let (id, tid) = nearest(cx)?;
                if !self.pick_on(id, tid, cx) {
                    return Err(cx.status.clone());
                }
                planned(self).map(|_| cx.status.clone())
            }
            "pick_joint" => {
                let j = joint_id(cx)?;
                self.pick_joint(j, cx);
                planned(self).map(|_| cx.status.clone())
            }
            "connect" => {
                if let Some(i) = a.opt_str("interaction") {
                    self.seat.interaction = Some(inter_of(i)?);
                }
                if let Some(m) = a.opt_str("motion") {
                    self.motion = Some(motion_of(m)?);
                }
                if let Some(m) = a.opt_bool("move_into_place") {
                    self.seat.move_into_place = m;
                }
                self.recontact(cx.scene);
                match &self.plan {
                    Err(e) => return Err(e.clone()),
                    Ok(None) => return Err("pick two joints or surfaces first (pick_surface / pick_joint)".into()),
                    Ok(Some(_)) => {}
                }
                self.connect(cx);
                Ok(cx.status.clone())
            }
            "seat" => {
                if let Some(m) = a.opt_str("mode") {
                    self.seat.touch = m == "touch";
                }
                if let Some(p) = a.opt_point("offset") {
                    self.seat.offset = DVec3::from_array(p);
                }
                if let Some(f) = a.opt_bool("keep_turn") {
                    self.seat.keep_turn = f;
                }
                if let Some(t) = a.opt_usize("turn") {
                    self.seat.turn = t;
                }
                self.recontact(cx.scene);
                match (&self.plan, &self.contact) {
                    (Err(e), _) => Err(e.clone()),
                    (Ok(Some(_)), Some(c)) if self.seat.touch => Ok(format!("Seated touching: gap {:.3} mm (was {:.3}), deepest overlap {:.3} mm", c.gap_after, c.gap_before, c.overlap)),
                    (Ok(Some(_)), _) => Ok("Seated by the joints".into()),
                    (Ok(None), _) => Ok("Set; pick two joints or surfaces to seat them".into()),
                }
            }
            "add_joint" => {
                let (id, tid) = nearest(cx)?;
                let shape = a.opt_str("shape").map(shape_of).transpose()?;
                // Looked for as that shape (a joint there already is fitted again as it); where
                // there is none (a ball on a flat face), what is there made that shape.
                let look = self.shape;
                let mut added = Err(String::new());
                if let Some(k) = shape.and_then(JointShape::kind) {
                    self.shape = Some(k);
                    added = self.add_joint_at(id, tid, cx);
                    self.shape = look;
                }
                let (j, new) = match added {
                    Ok(a) => a,
                    Err(_) => self.add_joint_at(id, tid, cx)?,
                };
                if let Some((shape, old)) = shape.and_then(|s| cx.scene.assembly.joint(j).filter(|jj| jj.shape != s).map(|jj| (s, jj.shape))) {
                    // Part of the step that added it.
                    cx.history.touch();
                    assembly::edit_joint(cx.scene, j, |jj| jj.set_shape(shape))?;
                    cx.scene.assembly.rename_for_shape(j, old);
                }
                Ok(format!("Joint {j}{}: {}", if new { "" } else { " (already there)" }, cx.scene.assembly.joint(j).map_or_else(|| cx.status.clone(), |j| j.title())))
            }
            "edit_joint" => {
                let j = joint_id(cx)?;
                let shape = a.opt_str("shape").map(shape_of).transpose()?;
                let (radius, length, depth) = (a.opt_f64("radius"), a.opt_f64("length"), a.opt_f64("depth"));
                let (nx, ny) = (a.opt_f64("nudge_x"), a.opt_f64("nudge_y"));
                let name = a.opt_str("name").map(str::to_string);
                cx.history.push("Edit joint", cx.scene);
                // Another kind of shape: fitted again to its surface as it, as in the panel.
                let mut said = String::new();
                if let Some(s) = shape.filter(|&s| cx.scene.assembly.joint(j).is_some_and(|jj| jj.shape != s)) {
                    said = Self::reshape_joint(cx.scene, j, s);
                }
                let notes = assembly::edit_joint(cx.scene, j, |jj| {
                    if let Some(r) = radius.filter(|r| *r > 0.0) {
                        jj.radius = r;
                    }
                    if let Some(l) = length.filter(|l| *l >= 0.0) {
                        jj.length = l;
                    }
                    if let Some(d) = depth {
                        jj.offset.z = d;
                    }
                    if let Some(x) = nx {
                        jj.offset.x = x;
                    }
                    if let Some(y) = ny {
                        jj.offset.y = y;
                    }
                    if let Some(n) = name {
                        jj.name = n;
                    }
                })?;
                Ok(format!("Joint {j}: {}. {said} {}", cx.scene.assembly.joint(j).map_or_else(String::new, |j| j.title()), notes.join(" ")).trim_end().to_string())
            }
            "delete_joint" => {
                let j = joint_id(cx)?;
                cx.history.push("Delete joint", cx.scene);
                cx.scene.assembly.remove_joint(j);
                Ok(format!("Deleted joint {j}"))
            }
            "disconnect" => {
                let c = conn_id(cx)?;
                cx.history.push("Disconnect", cx.scene);
                cx.scene.assembly.remove_connection(c);
                Ok(format!("Removed connection {c}"))
            }
            "pose" => {
                let id = conn_id(cx)?;
                let c = cx.scene.assembly.connection(id).unwrap().clone();
                let r = |k: &str| a.opt_f64(k).unwrap_or(0.0).to_radians();
                let rel = match c.kind {
                    Motion::Ball => DQuat::from_euler(EulerRot::XYZ, r("x"), r("y"), r("twist")),
                    k if k.turns() => DQuat::from_rotation_z(r("angle")),
                    _ => DQuat::IDENTITY,
                };
                let off = DVec3::new(0.0, 0.0, a.opt_f64("slide").unwrap_or(0.0));
                let (rel, off) = assembly::constrain(c.kind, rel, off);
                self.move_connection(cx, id, c.motion_to(rel, off), "Pose");
                Ok(format!("Posed connection {id}"))
            }
            "set_connection" => {
                let id = conn_id(cx)?;
                let inter = a.opt_str("interaction").map(inter_of).transpose()?;
                let kind = a.opt_str("motion").map(motion_of).transpose()?;
                let offset = a.opt_point("offset").map(DVec3::from_array);
                cx.history.push("Connection", cx.scene);
                assembly::edit_connection(cx.scene, id, |c| {
                    if let Some(i) = inter {
                        c.interaction = i;
                    }
                    if let Some(k) = kind {
                        c.kind = k;
                    }
                    if let Some(o) = offset {
                        c.seat.shift = o;
                    }
                })?;
                let c = cx.scene.assembly.connection(id).unwrap();
                Ok(format!("Connection {id}: {} · {}", c.interaction.label(), c.kind.label()))
            }
            "fit_touching" => {
                let id = conn_id(cx)?;
                self.fit_touching(cx, id)
            }
            "rest_pose" => {
                if a.has("connection") {
                    let id = conn_id(cx)?;
                    let m = cx.scene.assembly.connection(id).unwrap().motion_to(DQuat::IDENTITY, DVec3::ZERO);
                    self.move_connection(cx, id, m, "Rest pose");
                } else {
                    self.reset_all(cx);
                }
                Ok("Back in the rest pose".into())
            }
            "ground" => {
                let id = a.usize("part")? as ObjectId;
                if cx.scene.get(id).is_none() {
                    return Err(format!("there is no object {id}"));
                }
                let on = a.opt_bool("on").unwrap_or(true);
                cx.scene.assembly.set_grounded(id, on);
                self.seen = seen_of(&cx.scene.assembly);
                self.recontact(cx.scene);
                Ok(if on { "Grounded".into() } else { "Freed".into() })
            }
            "refine_from_selection" => {
                let i = a.usize("pick")?;
                if !(1..=2).contains(&i) || i > self.picks.len() + 1 {
                    return Err(format!("pick is 1 or 2 (and at most one past the {} picked)", self.picks.len()));
                }
                self.mode = Mode::Connect;
                self.use_selection(RefineFor::Pick(i - 1), cx)
            }
            "check_collisions" => {
                self.check_collisions(cx);
                Ok(cx.status.clone())
            }
            "clear" => {
                self.picks.clear();
                self.plan = Ok(None);
                Ok("Cleared".into())
            }
            _ => Err(format!("Assemble has no action {action}")),
        }
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        let name = |id: ObjectId| scene.get(id).map_or_else(String::new, |o| o.name.clone());
        let asm = &scene.assembly;
        let plan = self.plan.as_ref().ok().and_then(|p| p.as_ref());
        serde_json::json!({
            "mode": match self.mode { Mode::Pose => "pose", Mode::Connect => "connect", Mode::AddJoint => "add_joint" },
            "grounded": asm.grounded.iter().map(|&g| name(g)).collect::<Vec<_>>(),
            "selected": match asm.selected {
                Some(Selected::Joint(j)) => serde_json::json!({"joint": j}),
                Some(Selected::Connection(c)) => serde_json::json!({"connection": c}),
                None => serde_json::Value::Null,
            },
            "parts": scene.objects.iter().filter(|o| !o.reference).map(|o| serde_json::json!({ "id": o.id, "name": o.name })).collect::<Vec<_>>(),
            "joints": asm.joints.iter().map(|j| serde_json::json!({
                "id": j.id,
                "part": name(j.part),
                "name": j.name,
                "shape": j.shape.key(),
                "radius": j.radius,
                "length": j.length,
                "depth": j.offset.z,
                "nudge": [j.offset.x, j.offset.y],
                "centre": j.frame().origin.to_array(),
                "axis": j.frame().axis().to_array(),
                "knows_surface": j.surface.is_some(),
                "connection": asm.connection_at(j.id).map(|c| c.id),
            })).collect::<Vec<_>>(),
            "connections": asm.connections.iter().map(|c| serde_json::json!({
                "id": c.id,
                "joints": [c.a, c.b],
                "part": name(c.child),
                "on": name(c.parent),
                "interaction": c.interaction.label(),
                "motion": c.kind.label(),
                "about": c.about,
                "angle_deg": c.angle().to_degrees(),
                "seat_offset": c.seat.shift.to_array(),
                "pivot": c.on_parent().origin.to_array(),
                "apart": c.is_apart(),
            })).collect::<Vec<_>>(),
            "picked": self.picks.iter().map(|p| serde_json::json!({ "part": name(p.object), "what": p.label(asm) })).collect::<Vec<_>>(),
            "error": self.plan.as_ref().err(),
            "plan": plan.map(|p| serde_json::json!({
                "moves": name(self.picks[p.moving].object),
                "interaction": p.interaction.label(),
                "motion": self.motion.unwrap_or(p.suggested).label(),
                "pivot": p.pivot.to_array(),
                "clearance_mm": p.clearance,
                "replaces": p.replaces,
            })),
            "seat": {
                "mode": if self.seat.touch { "touch" } else { "joints" },
                "gap_before_mm": self.contact.map(|c| c.gap_before),
                "gap_after_mm": self.contact.map(|c| c.gap_after),
                "overlap_mm": self.contact.map(|c| c.overlap),
                "offset": self.seat.offset.to_array(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;
    use serde_json::json;

    fn arm_motion() -> Rigid {
        Rigid::about(DQuat::from_euler(EulerRot::XYZ, 0.3, 1.2, -0.4), DVec3::ZERO).then(&Rigid::translation(DVec3::new(12.0, 5.0, -3.0)))
    }

    /// A torso block with a socket cut in its side, and an arm: a ball on a stick, lying away
    /// from it at an angle.
    fn kit() -> (Scene, ObjectId, ObjectId) {
        let mut block = primitives::cube(1.0, 6);
        block.transform(|p| p * 4.0);
        let mut cutter = primitives::icosphere(1.05, 4);
        cutter.transform(|p| p + DVec3::new(2.3, 0.0, 0.0));
        let torso = mm_edit::boolean::mesh_boolean(&block, &cutter, mm_edit::boolean::BooleanOp::Difference).unwrap().mesh;
        let mut ball = primitives::icosphere(1.0, 4);
        ball.transform(|p| p + DVec3::new(0.0, 0.0, 4.6));
        let stick = primitives::cylinder(0.4, 4.4, 24, 12, true);
        let mut arm = mm_edit::boolean::mesh_boolean(&ball, &stick, mm_edit::boolean::BooleanOp::Union).unwrap().mesh;
        let r = arm_motion();
        arm.transform(|p| r.apply(p));
        let mut scene = Scene::default();
        let t = scene.add("torso", torso);
        let a = scene.add("arm", arm);
        (scene, t, a)
    }

    fn cx_for<'a>(scene: &'a mut Scene, history: &'a mut crate::scene::History, camera: &'a mut crate::camera::Camera, status: &'a mut String, ctx: &'a egui::Context) -> ToolCx<'a> {
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        ToolCx { scene, history, camera, status, egui: ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift }
    }

    fn t_near(m: &Mesh, p: DVec3) -> u32 {
        m.triangle_ids().min_by(|&a, &b| (m.tri_centroid(a) - p).length().total_cmp(&(m.tri_centroid(b) - p).length())).unwrap()
    }

    #[test]
    fn connect_makes_both_joints_seats_the_ball_and_posing_turns_the_arm_about_it() {
        let (mut scene, torso, arm) = kit();
        let ball_top = arm_motion().apply(DVec3::new(0.0, 0.0, 5.6));
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let mut cx = cx_for(&mut scene, &mut history, &mut camera, &mut status, &ctx);
        let mut t = AssembleTool::new(&mut cx);
        assert!(cx.scene.assembly.is_grounded(torso), "the largest part is grounded");
        t.act("pick_surface", &json!({"at": ball_top.to_array()}), &mut cx).unwrap();
        t.act("pick_surface", &json!({"at": [3.3, 0.0, 0.0]}), &mut cx).unwrap();
        let plan = t.plan.clone().unwrap().unwrap();
        assert_eq!((plan.interaction, plan.suggested), (Interaction::Concentric, Motion::Ball));
        assert_eq!(plan.parts, vec![arm]);
        assert!((plan.clearance.unwrap() - 0.05).abs() < 0.02);
        let c = t.contact.expect("touching fit");
        assert!(c.gap_after < c.gap_before, "{c:?}");
        assert!(plan.pivot.x < 2.29 && (plan.pivot - DVec3::new(2.3, 0.0, 0.0)).length() < 0.15, "{}", plan.pivot);
        // The joints meeting instead: the ball's centre on the socket's.
        t.act("seat", &json!({"mode": "joints"}), &mut cx).unwrap();
        let centred = t.plan.clone().unwrap().unwrap();
        assert!((centred.pivot - DVec3::new(2.3, 0.0, 0.0)).length() < 0.02, "{}", centred.pivot);
        t.act("connect", &json!({}), &mut cx).unwrap();
        let asm = &cx.scene.assembly;
        assert_eq!((asm.joints.len(), asm.connections.len()), (2, 1), "a joint on each part, and the connection");
        let c = asm.connections[0].clone();
        assert_eq!((c.parent, c.child, c.kind, c.interaction), (torso, arm, Motion::Ball, Interaction::Concentric));
        assert_eq!(asm.joint(c.a).unwrap().shape, JointShape::Socket);
        assert_eq!(asm.joint(c.b).unwrap().shape, JointShape::Ball);
        assert!((c.on_parent().origin - DVec3::new(2.3, 0.0, 0.0)).length() < 0.02);
        assert!(c.error() < 1e-9 && c.pose().0.angle_between(DQuat::IDENTITY) < 1e-9, "at rest");
        // The arm's ball sits in the socket, its stick coming out of the opening (+x).
        let ball = mate::recognize(&cx.scene.get(arm).unwrap().mesh, t_near(&cx.scene.get(arm).unwrap().mesh, c.on_parent().origin - DVec3::X), &RecognizeOptions::default()).unwrap();
        assert_eq!(ball.name(), "ball");
        assert!((ball.shape.origin() - c.on_parent().origin).length() < 0.02);
        assert!(ball.body.unwrap().dot(DVec3::X) > 0.95, "{:?}", ball.body);
        // Swing by 30°: still together.
        t.act("pose", &json!({"connection": c.id, "x": 30.0}), &mut cx).unwrap();
        let c2 = cx.scene.assembly.connections[0].clone();
        assert!((c2.pose().0.angle_between(DQuat::IDENTITY) - 30f64.to_radians()).abs() < 1e-6);
        assert!(c2.error() < 1e-9);
        assert_eq!(cx.history.undo_label(), Some("Pose"));
        // A hinge keeps only the turn about its axis; then back to rest.
        t.act("set_connection", &json!({"connection": c.id, "motion": "Hinge"}), &mut cx).unwrap();
        assert!(cx.scene.assembly.connections[0].error() < 1e-9);
        t.act("rest_pose", &json!({}), &mut cx).unwrap();
        assert!(cx.scene.assembly.connections[0].pose().0.angle_between(DQuat::IDENTITY) < 1e-9);
        // The socket's joint moved 0.2 out: the arm follows (joint edits re-seat).
        let before = cx.scene.get(arm).unwrap().mesh.vertex(0);
        t.act("edit_joint", &json!({"joint": c.a, "depth": 0.2}), &mut cx).unwrap();
        let socket_axis = cx.scene.assembly.joint(c.a).unwrap().frame().axis();
        assert!((cx.scene.get(arm).unwrap().mesh.vertex(0) - (before + socket_axis * 0.2)).length() < 1e-9);
        // A moved joint no longer fits its old triangles; back where it was found, it does: fit the
        // surfaces touching, the seat changes, the pose stays.
        t.act("set_connection", &json!({"connection": c.id, "motion": "Ball"}), &mut cx).unwrap();
        assert!(t.act("fit_touching", &json!({"connection": c.id}), &mut cx).is_err());
        t.act("edit_joint", &json!({"joint": c.a, "depth": 0.0}), &mut cx).unwrap();
        t.act("fit_touching", &json!({"connection": c.id}), &mut cx).unwrap();
        let cf = cx.scene.assembly.connections[0].clone();
        assert!(cf.error() < 1e-9 && cf.pose().0.angle_between(DQuat::IDENTITY) < 1e-9);
        // Joints loaded from a project do not know their triangles: a click on the socket still
        // finds its joint (by shape and place).
        for j in cx.scene.assembly.joints.iter_mut() {
            j.surface = None;
        }
        t.act("pick_joint", &json!({"joint": c.b}), &mut cx).unwrap();
        let again = t.act("pick_surface", &json!({"at": [1.25, 0.0, 0.0], "part": torso}), &mut cx);
        assert!(again.is_ok(), "the socket is picked as its joint");
        assert!(matches!(t.picks[1].what, PickWhat::Joint(j) if j == c.a), "the click on the socket's surface picks its joint");
        assert!(cx.scene.assembly.joint(c.a).unwrap().surface.is_some(), "and gives it its triangles again");
        let state = t.state(cx.scene);
        assert_eq!(state["connections"][0]["motion"], "Ball");
        assert_eq!(state["grounded"][0], "torso");
    }

    /// Press at `from`, drag to `to`, release (⌘ and Shift as given).
    fn drag_pointer(t: &mut AssembleTool, cx: &mut ToolCx, from: egui::Pos2, to: egui::Pos2, command: bool, shift: bool) {
        let ev = |pos: egui::Pos2, pressed: bool, down: bool, released: bool| PointerInput { pos: Some(pos), pressed, down, released, clicked: false, double_clicked: false, shift, command };
        t.pointer(&ev(from, false, false, false), cx);
        t.pointer(&ev(from, true, true, false), cx);
        t.pointer(&ev(from + (to - from) * 0.5, false, true, false), cx);
        t.pointer(&ev(to, false, true, false), cx);
        t.pointer(&ev(to, false, false, true), cx);
    }

    /// Drag ring `i` of the selected connection's handles by `deg` degrees (seen along its axis).
    fn turn_ring(t: &mut AssembleTool, cx: &mut ToolCx, i: usize, deg: f64, command: bool) {
        let g = t.gizmo(cx.scene).expect("handles");
        assert!(g.handles().contains(&Handle::Turn(i)));
        let axis = g.axes[i];
        cx.camera.look_along(-axis);
        let (r, _) = g.sizes(cx);
        let (u, v) = mm_core::geom::orthonormal_basis(axis);
        let p0 = g.centre + (u * 0.3f64.cos() + v * 0.3f64.sin()) * r;
        let p1 = g.centre + DQuat::from_axis_angle(axis, deg.to_radians()) * (p0 - g.centre);
        let (a, b) = (cx.project(p0).unwrap(), cx.project(p1).unwrap());
        drag_pointer(t, cx, a, b, command, false);
    }

    #[test]
    fn pose_handles_turn_and_slide_only_as_the_connection_allows() {
        let (mut scene, torso, arm) = kit();
        let ball_top = arm_motion().apply(DVec3::new(0.0, 0.0, 5.6));
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let mut cx = cx_for(&mut scene, &mut history, &mut camera, &mut status, &ctx);
        let mut t = AssembleTool::new(&mut cx);
        t.act("pick_surface", &json!({"at": ball_top.to_array()}), &mut cx).unwrap();
        t.act("pick_surface", &json!({"at": [3.3, 0.0, 0.0]}), &mut cx).unwrap();
        t.act("seat", &json!({"mode": "joints"}), &mut cx).unwrap();
        t.act("connect", &json!({}), &mut cx).unwrap();
        let cid = cx.scene.assembly.connections[0].id;
        cx.scene.assembly.selected = Some(Selected::Connection(cid));
        t.mode = Mode::Pose;
        let mut b = Aabb::EMPTY;
        for o in &cx.scene.objects {
            b = b.union(&o.bounds());
        }
        cx.camera.focus(&b);
        let arm_at = |cx: &ToolCx| cx.scene.get(arm).unwrap().mesh.vertex(0);
        let torso_at = |cx: &ToolCx| cx.scene.get(torso).unwrap().mesh.vertex(0);

        // A ball: three rings, turning with the part. The twist ring turns the arm about its own
        // axis by just what the cursor went round.
        let g = t.gizmo(cx.scene).unwrap();
        assert_eq!(g.handles(), &[Handle::Turn(0), Handle::Turn(1), Handle::Turn(2)]);
        turn_ring(&mut t, &mut cx, 2, 40.0, false);
        let c = cx.scene.assembly.connection(cid).unwrap().clone();
        let (rel, off) = c.pose();
        assert!((rel.angle_between(DQuat::IDENTITY).to_degrees() - 40.0).abs() < 0.05, "{rel:?}");
        assert!((assembly::twist_z(rel).angle_between(DQuat::IDENTITY).to_degrees() - 40.0).abs() < 0.05, "all of it a twist");
        assert!(off.length() < 1e-9 && c.error() < 1e-9, "still in its socket");
        assert_eq!(cx.history.undo_label(), Some("Pose arm"));

        // A hinge: one ring, about its axis; ⌘ steps the connection's angle by 15°.
        t.act("set_connection", &json!({"connection": cid, "motion": "Hinge"}), &mut cx).unwrap();
        t.act("rest_pose", &json!({}), &mut cx).unwrap();
        assert_eq!(t.gizmo(cx.scene).unwrap().handles(), &[Handle::Turn(2)]);
        turn_ring(&mut t, &mut cx, 2, 37.0, true);
        let c = cx.scene.assembly.connection(cid).unwrap().clone();
        assert!((c.angle().to_degrees() - 30.0).abs() < 1e-6, "{}", c.angle().to_degrees());
        assert!(c.error() < 1e-9);

        // The arm grounded instead: the torso turns, the other way round its angle.
        t.act("rest_pose", &json!({}), &mut cx).unwrap();
        cx.scene.assembly.set_grounded(torso, false);
        cx.scene.assembly.set_grounded(arm, true);
        let (arm0, torso0) = (arm_at(&cx), torso_at(&cx));
        turn_ring(&mut t, &mut cx, 2, 37.0, true);
        let c = cx.scene.assembly.connection(cid).unwrap().clone();
        assert!((c.angle().to_degrees() + 30.0).abs() < 1e-6, "{}", c.angle().to_degrees());
        assert_eq!(arm_at(&cx), arm0, "the grounded arm stays");
        assert!((torso_at(&cx) - torso0).length() > 0.1, "the torso turned");
        assert_eq!(cx.history.undo_label(), Some("Pose torso"));
        cx.scene.assembly.set_grounded(arm, false);
        cx.scene.assembly.set_grounded(torso, true);
        t.act("rest_pose", &json!({}), &mut cx).unwrap();

        // A slider: one arrow, no ring; Shift steps the slide by 0.1 mm.
        t.act("set_connection", &json!({"connection": cid, "motion": "Slider"}), &mut cx).unwrap();
        let g = t.gizmo(cx.scene).unwrap();
        assert_eq!(g.handles(), &[Handle::Slide(2)]);
        let axis = g.axes[2];
        cx.camera.look_along(mm_core::geom::orthonormal_basis(axis).0);
        let (_, len) = g.sizes(&cx);
        let p0 = g.centre + axis * (0.5 * len);
        let (a, b) = (cx.project(p0).unwrap(), cx.project(p0 + axis * 0.73).unwrap());
        drag_pointer(&mut t, &mut cx, a, b, false, true);
        let c = cx.scene.assembly.connection(cid).unwrap().clone();
        let (rel, off) = c.pose();
        assert!((off.z - 0.7).abs() < 1e-9 && off.x.abs() < 1e-9 && off.y.abs() < 1e-9, "{off}");
        assert!(rel.angle_between(DQuat::IDENTITY) < 1e-9, "a slider does not turn");

        // Planar: a ring and two arrows. Fixed: none.
        t.act("set_connection", &json!({"connection": cid, "motion": "Planar"}), &mut cx).unwrap();
        assert_eq!(t.gizmo(cx.scene).unwrap().handles(), &[Handle::Turn(2), Handle::Slide(0), Handle::Slide(1)]);
        t.act("set_connection", &json!({"connection": cid, "motion": "Fixed"}), &mut cx).unwrap();
        assert!(t.gizmo(cx.scene).is_none());
        // Nor any outside Pose.
        t.act("set_connection", &json!({"connection": cid, "motion": "Ball"}), &mut cx).unwrap();
        t.mode = Mode::Connect;
        assert!(t.gizmo(cx.scene).is_none());
    }

    #[test]
    fn a_flat_arm_rests_on_the_torso_ball_and_its_radius_tunes_it() {
        // The user's case: the torso has a ball, the arm only a flat face.
        let mut scene = Scene::default();
        let mut torso = primitives::cube(1.0, 4);
        torso.transform(|p| p * 4.0);
        let torso = scene.add("torso", torso);
        let mut arm = primitives::cube(1.0, 4);
        arm.transform(|p| p + DVec3::new(10.0, 3.0, 0.0));
        let arm = scene.add("arm", arm);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let mut cx = cx_for(&mut scene, &mut history, &mut camera, &mut status, &ctx);
        let mut t = AssembleTool::new(&mut cx);
        // A joint on the torso's +x face, turned into a ball r 1 sitting 0.5 proud of it.
        t.act("add_joint", &json!({"at": [2.0, 0.3, 0.2], "part": torso, "shape": "ball"}), &mut cx).unwrap();
        let ball = cx.scene.assembly.joints[0].id;
        t.act("edit_joint", &json!({"joint": ball, "radius": 1.0, "depth": 0.5}), &mut cx).unwrap();
        let bj = cx.scene.assembly.joint(ball).unwrap().clone();
        assert!((bj.frame().origin - DVec3::new(2.5, 0.0, 0.0)).length() < 1e-6, "{}", bj.frame().origin);
        assert!(bj.frame().axis().dot(DVec3::X) > 0.999);
        // The arm's flat face: the −x face of its cube.
        t.act("pick_joint", &json!({"joint": ball}), &mut cx).unwrap();
        t.act("pick_surface", &json!({"at": [9.5, 3.1, 0.1], "part": arm}), &mut cx).unwrap();
        // With the defaults ("Surfaces touch"): a tangent connection still rests on the radius.
        let plan = t.plan.clone().unwrap().unwrap();
        assert_eq!(plan.interaction, Interaction::Tangent);
        assert!(t.contact.is_none(), "no touching fit for a tangent connection");
        t.act("connect", &json!({}), &mut cx).unwrap();
        let c = cx.scene.assembly.connections[0].clone();
        let face = cx.scene.assembly.joint(c.b).unwrap().frame();
        // The face rests on the ball: its centre 1 out from the ball's centre, facing it.
        assert!((face.origin - DVec3::new(3.5, 0.0, 0.0)).length() < 1e-6, "{}", face.origin);
        assert!(face.axis().dot(-DVec3::X) > 0.999);
        // Swing it: the face stays tangent; a bigger ball pushes it out, keeping the swing.
        t.act("pose", &json!({"connection": c.id, "y": 25.0}), &mut cx).unwrap();
        t.act("edit_joint", &json!({"joint": ball, "radius": 1.4}), &mut cx).unwrap();
        let face = cx.scene.assembly.joint(c.b).unwrap().frame();
        assert!(((face.origin - DVec3::new(2.5, 0.0, 0.0)).length() - 1.4).abs() < 1e-6);
        assert!(face.axis().dot((DVec3::new(2.5, 0.0, 0.0) - face.origin).normalize()) > 1.0 - 1e-9);
        let c = cx.scene.assembly.connections[0].clone();
        assert!((c.pose().0.angle_between(DQuat::IDENTITY) - 25f64.to_radians()).abs() < 1e-6);
    }

    #[test]
    fn a_ball_made_from_a_flat_face_seats_by_its_centre_not_the_old_triangles() {
        // The user's other approach: the arm's flat end becomes a ball standing 0.5 proud.
        let (mut scene, torso, arm) = kit();
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let mut cx = cx_for(&mut scene, &mut history, &mut camera, &mut status, &ctx);
        let mut t = AssembleTool::new(&mut cx);
        // The stick's flat end (its cap at z = 0 before the arm was turned and moved).
        let end = arm_motion().apply(DVec3::new(0.1, 0.1, 0.0));
        t.act("add_joint", &json!({"at": end.to_array(), "part": arm, "shape": "ball"}), &mut cx).unwrap();
        let j = cx.scene.assembly.joints.last().unwrap().id;
        t.act("edit_joint", &json!({"joint": j, "radius": 1.0, "depth": 0.5}), &mut cx).unwrap();
        let jj = cx.scene.assembly.joint(j).unwrap();
        assert_eq!(jj.shape, JointShape::Ball);
        assert!(jj.fitted_surface_on(&cx.scene.get(arm).unwrap().mesh).is_none(), "its flat triangles no longer describe it");
        t.act("pick_joint", &json!({"joint": j}), &mut cx).unwrap();
        t.act("pick_surface", &json!({"at": [1.25, 0.0, 0.0], "part": torso}), &mut cx).unwrap();
        assert!(t.contact.is_none(), "no touching fit from the old triangles");
        t.act("connect", &json!({}), &mut cx).unwrap();
        let c = cx.scene.assembly.connections[0].clone();
        assert_eq!(c.interaction, Interaction::Concentric);
        let ball = cx.scene.assembly.joint(j).unwrap().frame();
        assert!((ball.origin - DVec3::new(2.3, 0.0, 0.0)).length() < 0.02, "the ball's centre on the socket's: {}", ball.origin);
    }

    #[test]
    fn surfaces_selected_by_hand_become_picks_and_joints() {
        let (mut scene, torso, arm) = kit();
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let mut cx = cx_for(&mut scene, &mut history, &mut camera, &mut status, &ctx);
        let mut t = AssembleTool::new(&mut cx);
        // The Select toolbar button: a joint-surface Select opens over Assemble for pick 1.
        cx.scene.active = Some(arm);
        t.script("select", &mut cx);
        let sub = t.take_request().expect("Select over Assemble");
        assert_eq!(sub.name(), "Select");
        let (target, used) = t.refining.clone().unwrap();
        assert_eq!(target, RefineFor::Pick(0));
        let centre = arm_motion().apply(DVec3::new(0.0, 0.0, 4.6));
        let o = cx.scene.get_mut(arm).unwrap();
        let ball: Vec<u32> = o.mesh.triangle_ids().filter(|&t| (o.mesh.tri_centroid(t) - centre).length() < 1.05 && o.mesh.tri_normal(t).dot(o.mesh.tri_centroid(t) - centre) > 0.9 * (o.mesh.tri_centroid(t) - centre).length()).collect();
        for &tr in &ball {
            o.selection.insert(tr);
        }
        used.store(true, Ordering::Relaxed);
        t.refining = None;
        t.use_selection(target, &mut cx).unwrap();
        assert_eq!(t.picks.len(), 1);
        assert!(matches!(&t.picks[0].what, PickWhat::Surface { feature, selected: true } if feature.name() == "ball"));
        // The socket by hand too, through the agent action on the torso's selection.
        cx.scene.active = Some(torso);
        let o = cx.scene.get_mut(torso).unwrap();
        let cup: Vec<u32> = o.mesh.triangle_ids().filter(|&t| (o.mesh.tri_centroid(t) - DVec3::new(2.3, 0.0, 0.0)).length() < 1.1 && o.mesh.tri_centroid(t).x < 1.95).collect();
        for &tr in &cup {
            o.selection.insert(tr);
        }
        t.act("refine_from_selection", &json!({"pick": 2}), &mut cx).unwrap();
        let c = t.contact.expect("seated touching");
        assert!(c.gap_after <= c.gap_before + 1e-9);
        t.act("connect", &json!({}), &mut cx).unwrap();
        assert_eq!(cx.scene.assembly.connections.len(), 1);
        // Add joint by hand: Select for a new joint, then use it.
        t.script("addjoint", &mut cx);
        cx.scene.active = Some(torso);
        t.script("select", &mut cx);
        assert_eq!(t.refining.as_ref().unwrap().0, RefineFor::NewJoint);
        let o = cx.scene.get_mut(torso).unwrap();
        let top: Vec<u32> = o.mesh.triangle_ids().filter(|&t| o.mesh.tri_normal(t).z > 0.99).collect();
        for &tr in &top {
            o.selection.insert(tr);
        }
        t.refining = None;
        t.use_selection(RefineFor::NewJoint, &mut cx).unwrap();
        let flat = cx.scene.assembly.joints.last().unwrap();
        assert_eq!((flat.part, flat.shape), (torso, JointShape::Flat));
        assert_eq!(cx.scene.assembly.selected, Some(Selected::Joint(flat.id)));
    }

    /// The joint editor's Shape menu and Add joint's Look for, clicked as a user would (found
    /// through AccessKit): another kind of shape fits the joint's own surface again as it.
    #[test]
    fn choosing_a_shape_fits_the_joint_again_as_it() {
        let (mut scene, _, arm) = kit();
        let ball_centre = arm_motion().apply(DVec3::new(0.0, 0.0, 4.6));
        let ball_top = arm_motion().apply(DVec3::new(0.0, 0.0, 5.6));
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let (mut t, id) = {
            let mut cx = cx_for(&mut scene, &mut history, &mut camera, &mut status, &ctx);
            let mut t = AssembleTool::new(&mut cx);
            t.act("add_joint", &json!({"at": ball_top.to_array()}), &mut cx).unwrap();
            t.mode = Mode::AddJoint;
            (t, cx.scene.assembly.joints.last().unwrap().id)
        };
        assert_eq!(scene.assembly.joint(id).unwrap().name, "Ball 1");
        struct Ui<'a> {
            ctx: &'a egui::Context,
            t: &'a mut AssembleTool,
            scene: &'a mut Scene,
            history: &'a mut crate::scene::History,
            camera: &'a mut crate::camera::Camera,
            status: &'a mut String,
        }
        impl Ui<'_> {
            fn frame(&mut self, events: Vec<egui::Event>) -> Vec<(egui::accesskit::NodeId, egui::accesskit::Node)> {
                let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), events, ..Default::default() };
                let ctx = self.ctx;
                let out = ctx.run_ui(input, |ui| {
                    let mut cx = cx_for(self.scene, self.history, self.camera, self.status, ctx);
                    self.t.panel(ui, &mut cx);
                });
                out.platform_output.accesskit_update.unwrap().nodes
            }
            fn click(&mut self, at: egui::Pos2) {
                let b = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
                for e in [vec![egui::Event::PointerMoved(at)], vec![b(true)], vec![b(false)], vec![]] {
                    self.frame(e);
                }
            }
            /// Open the menu showing `menu` and click `option` in it.
            fn choose(&mut self, menu: &str, option: &str) {
                fn find(nodes: &[(egui::accesskit::NodeId, egui::accesskit::Node)], f: impl Fn(&egui::accesskit::Node) -> bool) -> Option<egui::Pos2> {
                    nodes.iter().find(|(_, n)| f(n)).and_then(|(_, n)| n.bounds()).map(|b| egui::pos2(((b.x0 + b.x1) / 2.0) as f32, ((b.y0 + b.y1) / 2.0) as f32))
                }
                let nodes = self.frame(vec![]);
                // The last menu showing it (the joint editor's is below Look for).
                let rev: Vec<_> = nodes.into_iter().rev().collect();
                let combo = find(&rev, |n| n.role() == egui::accesskit::Role::ComboBox && n.value() == Some(menu)).unwrap_or_else(|| panic!("the {menu} menu"));
                self.click(combo);
                let nodes = self.frame(vec![]);
                let at = find(&nodes, |n| n.role() != egui::accesskit::Role::ComboBox && (n.label().as_deref() == Some(option) || n.value() == Some(option))).unwrap_or_else(|| panic!("{option} in the open menu"));
                self.click(at);
                self.frame(vec![]);
            }
        }
        let mut ui = Ui { ctx: &ctx, t: &mut t, scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status };
        // Look for a flat face: the selected ball is fitted again as one.
        ui.choose("Any (best fit)", "Flat face");
        let j = ui.scene.assembly.joint(id).unwrap().clone();
        assert_eq!((j.shape, j.name.as_str()), (JointShape::Flat, "Flat face 1"), "{}", ui.status);
        assert!(j.fitted_surface_on(&ui.scene.get(arm).unwrap().mesh).is_some(), "fitted to its surface");
        // Made a ball in the editor: the ball's surface fitted as a sphere again, not a ball the
        // flat face's size where the face was.
        ui.choose("Flat face", "Ball");
        let j = ui.scene.assembly.joint(id).unwrap().clone();
        assert_eq!((j.shape, j.name.as_str()), (JointShape::Ball, "Ball 1"), "{}", ui.status);
        assert!((j.radius - 1.0).abs() < 0.03 && (j.frame().origin - ball_centre).length() < 0.03, "{} at {}: {}", j.radius, j.frame().origin, ui.status);
        // A socket: the same sphere, only the shape changes.
        ui.choose("Ball", "Socket");
        let s = ui.scene.assembly.joint(id).unwrap().clone();
        assert_eq!(s.shape, JointShape::Socket);
        assert_eq!((s.radius, s.frame().origin), (j.radius, j.frame().origin));
        // Each change is a step of its own.
        assert_eq!(ui.history.labels().0.iter().filter(|&&l| l == "Joint shape").count(), 3);
    }

    #[test]
    fn refit_joint_keeps_given_names_and_says_why_it_cannot() {
        let (mut scene, torso, _) = kit();
        let id = scene.assembly.add_joint(Joint::point(torso, assembly::JFrame::new(DVec3::X, DVec3::X, None)));
        let e = assembly::refit_joint(&mut scene, id, Some(ShapeKind::Sphere), None).unwrap_err();
        assert!(e.contains("not known"), "{e}");
        // Made a ball anyway, of the same size, with the reason.
        let said = AssembleTool::reshape_joint(&mut scene, id, JointShape::Ball);
        assert!(said.contains("not known") && said.contains("same size"), "{said}");
        assert_eq!(scene.assembly.joint(id).unwrap().shape, JointShape::Ball);
        // Its name follows the shape it was named for; a name given by hand stays.
        assert_eq!(scene.assembly.joint(id).unwrap().name, "Ball 1");
        scene.assembly.joint_mut(id).unwrap().name = "Neck".into();
        AssembleTool::reshape_joint(&mut scene, id, JointShape::Socket);
        assert_eq!(scene.assembly.joint(id).unwrap().name, "Neck");
        // A ball asked for on a flat face (no ball there): the face found, made a ball, named one.
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let mut cx = cx_for(&mut scene, &mut history, &mut camera, &mut status, &ctx);
        let mut t = AssembleTool::new(&mut cx);
        let said = t.act("add_joint", &json!({"at": [2.0, 0.3, 0.2], "part": torso, "shape": "ball"}), &mut cx).unwrap();
        let j = cx.scene.assembly.joints.last().unwrap();
        assert_eq!((j.shape, j.name.as_str()), (JointShape::Ball, "Ball 1"), "{said}");
        assert_eq!(said.matches("ball r").count(), 1, "described once: {said}");
    }
}
