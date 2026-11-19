//! Assemblies of kit parts (the Assembly tab): which parts are held still (grounded), the
//! joints each part owns, and the connections between joints.
//!
//! A **joint** belongs to one part: a ball, socket, pin, hole, flat face or bare point. Its
//! frame's z axis points out of the part: from a ball's limb through the ball, out of a socket's
//! or hole's opening, along a pin to its tip, along a flat face's outward normal. It is found on
//! the part's surface (or placed there) and can be edited: its shape, radius, depth (proud of the
//! surface or into it) and position. It stays with its part, whatever is connected to it.
//!
//! A **connection** joins a joint on one part (`a`: the part it was connected to) to a joint on
//! another (`b`: the part that was moved), with an interaction: concentric (the balls' centres
//! meet), tangent (a flat face rests on a ball), coaxial (the pins' axes meet), flush (face to
//! face) or coincident (the frames meet). Each joint gives the connection an anchor frame
//! ([`anchors`]); b's is turned half round about x so that the two z axes face each other. At
//! rest b's anchor sits on a's moved by the connection's `seat` (the turn of a keyed face, the
//! fit of surfaces touching, the seat offset); the pose is how far it is from there, within the
//! motion the connection allows (ball, hinge, slider, …). A joint holds one connection at a time.
//!
//! Meshes hold their own world positions (objects have no transforms), so joint frames and
//! anchors are kept in world coordinates and ride along with their part: [`move_parts`] moves
//! parts with them, and parts moved by other tools are caught by [`follow`] (a rigid fit of a few
//! anchor vertices). Editing a joint ([`edit_joint`]) re-seats what is connected to it, keeping
//! its pose.
//!
//! The connections make a forest. Dragging a part turns it about the first connection that can
//! move on the way from it to the part its group hangs from (a grounded part, or else the part
//! that is no connection's child), carrying everything on its side. Fixed connections weld
//! parts into one.

use crate::scene::{ObjectId, Scene};
use glam::{DMat3, DMat4, DQuat, DVec2, DVec3};
use mm_core::hash::FastMap;
use mm_core::Mesh;
use mm_edit::mate::{self, Feature, Shape, ShapeKind};
pub use mm_edit::mate::JointKind as Motion;
use std::f64::consts::PI;
use std::sync::Arc;

/// A rigid motion `x ↦ rot x + shift`. Also used for a move within a frame (its axes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid {
    pub rot: DQuat,
    pub shift: DVec3,
}

impl Rigid {
    pub const IDENTITY: Rigid = Rigid { rot: DQuat::IDENTITY, shift: DVec3::ZERO };

    /// Turning by `rot` about the point `centre`.
    pub fn about(rot: DQuat, centre: DVec3) -> Rigid {
        Rigid { rot, shift: centre - rot * centre }
    }

    pub fn translation(d: DVec3) -> Rigid {
        Rigid { rot: DQuat::IDENTITY, shift: d }
    }

    pub fn apply(&self, p: DVec3) -> DVec3 {
        self.rot * p + self.shift
    }

    /// `self`, then `next`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn then(&self, next: &Rigid) -> Rigid {
        Rigid { rot: (next.rot * self.rot).normalize(), shift: next.rot * self.shift + next.shift }
    }

    pub fn inverse(&self) -> Rigid {
        let r = self.rot.inverse();
        Rigid { rot: r, shift: -(r * self.shift) }
    }

    pub fn mat4(&self) -> DMat4 {
        DMat4::from_rotation_translation(self.rot, self.shift)
    }

    /// The motion taking frame `from` onto frame `to`.
    pub fn between(from: &JFrame, to: &JFrame) -> Rigid {
        let rot = (to.rot * from.rot.inverse()).normalize();
        Rigid { rot, shift: to.origin - rot * from.origin }
    }

    /// It moves nothing by more than `eps` within `reach` of the origin.
    pub fn is_identity_within(&self, reach: f64, eps: f64) -> bool {
        self.shift.length() + self.rot.angle_between(DQuat::IDENTITY) * reach.max(1.0) <= eps
    }
}

/// A frame: an origin and axes (the z axis a joint's axis).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JFrame {
    pub origin: DVec3,
    pub rot: DQuat,
}

impl JFrame {
    /// A frame at `origin` with z along `axis` and x along `x` (made perpendicular), if given.
    pub fn new(origin: DVec3, axis: DVec3, x: Option<DVec3>) -> JFrame {
        let z = axis.normalize_or(DVec3::Z);
        let x = x.map(|x| x - z * x.dot(z)).filter(|x| x.length_squared() > 1e-18).map(|x| x.normalize()).unwrap_or_else(|| mm_core::geom::orthonormal_basis(z).0);
        JFrame { origin, rot: DQuat::from_mat3(&DMat3::from_cols(x, z.cross(x), z)).normalize() }
    }

    pub fn axis(&self) -> DVec3 {
        self.rot * DVec3::Z
    }

    pub fn moved(&self, m: &Rigid) -> JFrame {
        JFrame { origin: m.apply(self.origin), rot: (m.rot * self.rot).normalize() }
    }

    /// This frame moved by `r` within itself (along and about its own axes).
    pub fn then_local(&self, r: &Rigid) -> JFrame {
        JFrame { origin: self.origin + self.rot * r.shift, rot: (self.rot * r.rot).normalize() }
    }

    /// `other` within this frame: the `r` with `self.then_local(r) == other`.
    pub fn local_to(&self, other: &JFrame) -> Rigid {
        let inv = self.rot.inverse();
        Rigid { rot: (inv * other.rot).normalize(), shift: inv * (other.origin - self.origin) }
    }
}

/// The shapes a joint can have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JointShape {
    Ball,
    Socket,
    Pin,
    Hole,
    Flat,
    Point,
}

impl JointShape {
    pub const ALL: [JointShape; 6] = [JointShape::Ball, JointShape::Socket, JointShape::Pin, JointShape::Hole, JointShape::Flat, JointShape::Point];

    pub fn label(self) -> &'static str {
        match self {
            JointShape::Ball => "Ball",
            JointShape::Socket => "Socket",
            JointShape::Pin => "Pin",
            JointShape::Hole => "Hole",
            JointShape::Flat => "Flat face",
            JointShape::Point => "Point",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            JointShape::Ball => "ball",
            JointShape::Socket => "socket",
            JointShape::Pin => "pin",
            JointShape::Hole => "hole",
            JointShape::Flat => "flat",
            JointShape::Point => "point",
        }
    }

    pub fn from_key(k: &str) -> Option<JointShape> {
        JointShape::ALL.into_iter().find(|s| s.key() == k || s.label().eq_ignore_ascii_case(k))
    }

    /// A ball or a socket.
    pub fn round(self) -> bool {
        matches!(self, JointShape::Ball | JointShape::Socket)
    }

    /// A pin or a hole.
    pub fn axial(self) -> bool {
        matches!(self, JointShape::Pin | JointShape::Hole)
    }

    /// The surface it is fitted as (none for a point).
    pub fn kind(self) -> Option<ShapeKind> {
        match self {
            JointShape::Ball | JointShape::Socket => Some(ShapeKind::Sphere),
            JointShape::Pin | JointShape::Hole => Some(ShapeKind::Cylinder),
            JointShape::Flat => Some(ShapeKind::Plane),
            JointShape::Point => None,
        }
    }
}

/// How two joints meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interaction {
    /// The centres meet (a ball in a socket).
    Concentric,
    /// A flat face rests on a ball: it turns about the ball's centre, the radius away.
    Tangent,
    /// The axes meet (a pin in a hole).
    Coaxial,
    /// Face to face, the outlines lined up (a keyed peg).
    Flush,
    /// The frames meet.
    Coincident,
}

impl Interaction {
    pub const ALL: [Interaction; 5] = [Interaction::Concentric, Interaction::Tangent, Interaction::Coaxial, Interaction::Flush, Interaction::Coincident];

    pub fn label(self) -> &'static str {
        match self {
            Interaction::Concentric => "Concentric",
            Interaction::Tangent => "Tangent",
            Interaction::Coaxial => "Coaxial",
            Interaction::Flush => "Flush",
            Interaction::Coincident => "Coincident",
        }
    }

    pub fn about(self) -> &'static str {
        match self {
            Interaction::Concentric => "The centres meet (a ball in its socket)",
            Interaction::Tangent => "The flat face rests on the ball and turns about its centre; the ball's radius sets how far out it sits",
            Interaction::Coaxial => "The axes meet (a pin in its hole)",
            Interaction::Flush => "Face to face, the outlines lined up (a keyed peg in its recess)",
            Interaction::Coincident => "The joints' frames meet",
        }
    }

    pub fn from_label(k: &str) -> Option<Interaction> {
        Interaction::ALL.into_iter().find(|i| i.label().eq_ignore_ascii_case(k))
    }

    /// The interactions joints of these shapes can make, the usual one first.
    pub fn between(a: JointShape, b: JointShape) -> Vec<Interaction> {
        use Interaction::*;
        let (ra, rb) = (a.round(), b.round());
        let flat = |s: JointShape| s == JointShape::Flat;
        if ra && rb {
            vec![Concentric, Coincident]
        } else if (a == JointShape::Ball && flat(b)) || (flat(a) && b == JointShape::Ball) {
            vec![Tangent, Concentric, Coincident]
        } else if (ra && flat(b)) || (flat(a) && rb) {
            // A flat end centred in a cup.
            vec![Concentric, Coincident]
        } else if a.axial() && b.axial() {
            vec![Coaxial, Coincident]
        } else if flat(a) && flat(b) {
            vec![Flush, Coincident]
        } else if (ra && b.axial()) || (a.axial() && rb) {
            vec![Concentric, Coincident]
        } else {
            vec![Coincident]
        }
    }

    /// The motion a connection of this interaction usually allows.
    pub fn motion(self, a: &Joint, b: &Joint) -> Motion {
        match self {
            Interaction::Concentric | Interaction::Tangent => Motion::Ball,
            Interaction::Coaxial => Motion::Hinge,
            Interaction::Flush if a.round && b.round => Motion::Hinge,
            Interaction::Flush | Interaction::Coincident => Motion::Fixed,
        }
    }

    /// The part is turned only about the joint's axis when it is seated (anything else lines
    /// its axis up first).
    fn free_twist(self) -> bool {
        matches!(self, Interaction::Concentric | Interaction::Tangent | Interaction::Coaxial)
    }
}

/// A joint: a place on a part where another part can be connected.
#[derive(Clone, Debug)]
pub struct Joint {
    pub id: u64,
    pub part: ObjectId,
    pub name: String,
    pub shape: JointShape,
    /// A ball's, socket's, pin's or hole's radius.
    pub radius: f64,
    /// A pin's or hole's length.
    pub length: f64,
    /// A flat face's half sizes along x (its long side) and y, and whether its outline is round.
    pub half: DVec2,
    pub round: bool,
    /// Where it was found or placed (its z axis out of the part). Rides with the part.
    pub base: JFrame,
    /// Moved from there, along its own axes (z: proud of the surface, or into it below zero).
    pub offset: DVec3,
    /// The triangles it was fitted to, to fit them again or fit them touching another joint's.
    pub surface: Option<Arc<JointSurface>>,
    /// What it was found as, e.g. "ball r 2.07".
    pub fitted: String,
}

/// The triangles of a joint's surface, with its part's topology version then (another version:
/// they are no longer those triangles).
#[derive(Debug)]
pub struct JointSurface {
    pub tris: Vec<u32>,
    pub topo: u64,
    /// The shape and radius the joint was fitted as: once it is edited away from them, the
    /// triangles no longer describe it (they still say where it came from).
    pub shape: JointShape,
    pub radius: f64,
}

impl Joint {
    /// A joint on `part` (no id yet) made of a recognized surface of its mesh `mesh`.
    pub fn from_feature(part: ObjectId, f: &Feature, mesh: &Mesh) -> Joint {
        let out = |fallback: DVec3| f.body.map(|b| -b).unwrap_or(fallback);
        let (shape, radius, length, half, round, base) = match f.shape {
            Shape::Sphere { centre, radius } => {
                let z = out((centre - mesh.bounds().center()).normalize_or(DVec3::Z));
                (if f.convex { JointShape::Ball } else { JointShape::Socket }, radius, 0.0, DVec2::ZERO, false, JFrame::new(centre, z, None))
            }
            Shape::Cylinder { centre, axis, radius, length } => (if f.convex { JointShape::Pin } else { JointShape::Hole }, radius, length, DVec2::ZERO, false, JFrame::new(centre, axis, None)),
            Shape::Plane { centre, normal, u, half, round } => (JointShape::Flat, half.x.min(half.y), 0.0, half, round, JFrame::new(centre, normal, Some(u))),
        };
        Joint {
            id: 0,
            part,
            name: String::new(),
            shape,
            radius,
            length,
            half,
            round,
            base,
            offset: DVec3::ZERO,
            surface: Some(Arc::new(JointSurface { tris: f.tris.clone(), topo: mesh.topology_version(), shape, radius })),
            fitted: f.describe(),
        }
    }

    /// A bare point joint at `frame`.
    pub fn point(part: ObjectId, frame: JFrame) -> Joint {
        Joint { id: 0, part, name: String::new(), shape: JointShape::Point, radius: 0.0, length: 0.0, half: DVec2::ZERO, round: false, base: frame, offset: DVec3::ZERO, surface: None, fitted: String::new() }
    }

    /// Where it is: its base moved by its offset.
    pub fn frame(&self) -> JFrame {
        self.base.then_local(&Rigid::translation(self.offset))
    }

    /// Its size (radius, or half a flat face's diagonal), to draw it.
    pub fn size(&self) -> f64 {
        match self.shape {
            JointShape::Flat => self.half.length(),
            JointShape::Point => 0.0,
            _ => self.radius,
        }
    }

    /// e.g. "ball r 2.07", "pin r 0.50 × 4.00", "flat face 3.57 × 3.57".
    pub fn describe(&self) -> String {
        match self.shape {
            JointShape::Ball | JointShape::Socket => format!("{} r {:.2}", self.shape.label().to_lowercase(), self.radius),
            JointShape::Pin | JointShape::Hole => format!("{} r {:.2} × {:.2}", self.shape.label().to_lowercase(), self.radius, self.length),
            JointShape::Flat if self.round => format!("round flat face, diameter {:.2}", self.half.x + self.half.y),
            JointShape::Flat => format!("flat face {:.2} × {:.2}", 2.0 * self.half.x, 2.0 * self.half.y),
            JointShape::Point => "point".into(),
        }
    }

    /// Its name, or what it is when it has none.
    pub fn title(&self) -> String {
        if self.name.is_empty() {
            self.describe()
        } else {
            format!("{} ({})", self.name, self.describe())
        }
    }

    /// The triangles it was fitted to, if they are still those of `mesh`.
    pub fn surface_on(&self, mesh: &Mesh) -> Option<&[u32]> {
        self.surface.as_deref().filter(|s| s.topo == mesh.topology_version()).map(|s| s.tris.as_slice())
    }

    /// The triangles it was fitted to, if they still describe it: still those of `mesh`, and the
    /// joint not edited since (same shape and radius, not moved). Fitting surfaces touching uses
    /// only these.
    pub fn fitted_surface_on(&self, mesh: &Mesh) -> Option<&[u32]> {
        let s = self.surface.as_deref()?;
        (s.shape == self.shape && (s.radius - self.radius).abs() <= 1e-9 * self.radius.abs().max(1.0) && self.offset == DVec3::ZERO).then_some(())?;
        self.surface_on(mesh)
    }

    /// Make it `shape`, giving it a radius and a face size if it had none.
    pub fn set_shape(&mut self, shape: JointShape) {
        if self.radius <= 0.0 {
            self.radius = self.half.x.min(self.half.y).max(self.length * 0.25).max(0.5);
        }
        if shape == JointShape::Flat && self.half == DVec2::ZERO {
            self.half = DVec2::splat(self.radius.max(0.5));
        }
        self.shape = shape;
    }
}

/// The anchor frames joints `a` and `b` give a connection of `inter`: each joint's frame (a flat
/// face's lifted by the ball's radius for a tangent connection), b's turned half round about x
/// so that the two z axes face each other.
pub fn anchors(a: &Joint, b: &Joint, inter: Interaction) -> (JFrame, JFrame) {
    let lift = |j: &Joint, other: &Joint| {
        let mut f = j.frame();
        if inter == Interaction::Tangent && j.shape == JointShape::Flat && other.shape == JointShape::Ball {
            f.origin += f.axis() * other.radius;
        }
        f
    };
    let fa = lift(a, b);
    let mut fb = lift(b, a);
    fb.rot = (fb.rot * DQuat::from_rotation_x(PI)).normalize();
    (fa, fb)
}

/// A connection between a joint on one part and a joint on another.
#[derive(Clone, Debug)]
pub struct Connection {
    pub id: u64,
    /// The joints: `a` on the part it was connected to, `b` on the part that was moved.
    pub a: u64,
    pub b: u64,
    pub part_a: ObjectId,
    pub part_b: ObjectId,
    /// The part it hangs from and the part it carries (`part_a` and `part_b` as connected;
    /// turned round when another part of the group is connected to something else).
    pub parent: ObjectId,
    pub child: ObjectId,
    pub interaction: Interaction,
    pub kind: Motion,
    /// Where b's anchor sits at rest, within a's anchor.
    pub seat: Rigid,
    /// The joints' anchors (see [`anchors`]), riding with the parts.
    pub on_a: JFrame,
    pub on_b: JFrame,
    /// The size of the joints, to draw it.
    pub size: f64,
    /// What it joins, e.g. "ball r 2.07 in socket r 2.72".
    pub about: String,
}

impl Connection {
    /// The part at the other end from `id`.
    pub fn other(&self, id: ObjectId) -> Option<ObjectId> {
        if id == self.part_a {
            Some(self.part_b)
        } else if id == self.part_b {
            Some(self.part_a)
        } else {
            None
        }
    }

    pub fn joins(&self, id: ObjectId) -> bool {
        self.part_a == id || self.part_b == id
    }

    /// Where b's anchor sits at rest.
    fn seated_a(&self) -> JFrame {
        self.on_a.then_local(&self.seat)
    }

    /// The frame on the parent and the frame on the child: equal at rest.
    pub fn on_parent(&self) -> JFrame {
        if self.parent == self.part_a {
            self.seated_a()
        } else {
            self.on_b
        }
    }

    pub fn on_child(&self) -> JFrame {
        if self.parent == self.part_a {
            self.on_b
        } else {
            self.seated_a()
        }
    }

    /// The child's frame in the parent's: its turn and offset (identity and zero at rest).
    pub fn pose(&self) -> (DQuat, DVec3) {
        let r = self.on_parent().local_to(&self.on_child());
        (r.rot, r.shift)
    }

    /// The motion of the child's side giving the connection the pose (`rel`, `off`).
    pub fn motion_to(&self, rel: DQuat, off: DVec3) -> Rigid {
        let target = self.on_parent().then_local(&Rigid { rot: rel, shift: off });
        Rigid::between(&self.on_child(), &target)
    }

    /// The motion of the child's side that puts the connection together the way its motion
    /// allows, keeping as much of its pose as that allows (after the motion changed, a joint was
    /// edited, or a part was moved off it by another tool).
    pub fn snap_motion(&self) -> Rigid {
        let (rel, off) = self.pose();
        let (rel, off) = constrain(self.kind, rel, off);
        self.motion_to(rel, off)
    }

    /// How far the child's frame is from where the motion allows (distance, plus the stray turn
    /// at the joint's size).
    pub fn error(&self) -> f64 {
        let (rel, off) = self.pose();
        let (r2, o2) = constrain(self.kind, rel, off);
        (off - o2).length() + rel.angle_between(r2) * self.size.max(1e-9)
    }

    /// It has come apart (a part was moved off it by another tool).
    pub fn is_apart(&self) -> bool {
        self.error() > 1e-3 * self.size.max(1e-6)
    }

    /// The turn about the connection's axis (radians, −π to π).
    pub fn angle(&self) -> f64 {
        let t = twist_z(self.pose().0);
        let a = 2.0 * t.z.atan2(t.w);
        if a > PI {
            a - 2.0 * PI
        } else if a < -PI {
            a + 2.0 * PI
        } else {
            a
        }
    }

    /// Its parts moved: make the seat what keeps the pose (`rel`, `off`), so the connection goes
    /// with them as it is.
    pub fn keep_pose(&mut self, rel: DQuat, off: DVec3) {
        let pose = Rigid { rot: rel, shift: off };
        let seated = if self.parent == self.part_a { self.on_b.then_local(&pose.inverse()) } else { self.on_b.then_local(&pose) };
        self.seat = self.on_a.local_to(&seated);
    }
}

/// The part of `q` that turns about z (swing–twist decomposition).
pub fn twist_z(q: DQuat) -> DQuat {
    let t = DQuat::from_xyzw(0.0, 0.0, q.z, q.w);
    if t.length_squared() < 1e-24 {
        DQuat::IDENTITY
    } else {
        t.normalize()
    }
}

/// The pose nearest (`rel`, `off`) that a connection of motion `kind` allows.
pub fn constrain(kind: Motion, rel: DQuat, off: DVec3) -> (DQuat, DVec3) {
    match kind {
        Motion::Fixed => (DQuat::IDENTITY, DVec3::ZERO),
        Motion::Ball => (rel, DVec3::ZERO),
        Motion::Hinge => (twist_z(rel), DVec3::ZERO),
        Motion::Slider => (DQuat::IDENTITY, DVec3::new(0.0, 0.0, off.z)),
        Motion::Cylindrical => (twist_z(rel), DVec3::new(0.0, 0.0, off.z)),
        Motion::Planar => (twist_z(rel), DVec3::new(off.x, off.y, 0.0)),
    }
}

/// The seat that puts `b`'s anchor on `a`'s turning the part as little as it can: with only a
/// turn about the shared axis where the interaction leaves that free (or, with `keep_turn` and a
/// ball, none at all), and with the turns a keyed flat face can take (`turn`: 0 the nearest,
/// then the others).
pub fn initial_seat(a: &Joint, b: &Joint, inter: Interaction, keep_turn: bool, turn: usize) -> Rigid {
    let (fa, fb) = anchors(a, b, inter);
    let rot = if keep_turn && matches!(inter, Interaction::Concentric | Interaction::Tangent) {
        fa.rot.inverse() * fb.rot
    } else {
        // Line the z axes up with the smallest turn: what is left is a turn about z.
        let lined = DQuat::from_rotation_arc(fb.axis(), fa.axis()) * fb.rot;
        let mut twist = twist_z(fa.rot.inverse() * lined);
        if inter == Interaction::Flush && a.shape == JointShape::Flat && b.shape == JointShape::Flat && !(a.round && b.round) {
            // Keyed faces: the long sides along each other (squares any of four ways).
            let square = a.half.y > 0.87 * a.half.x && b.half.y > 0.87 * b.half.x;
            let step = if square { PI / 2.0 } else { PI };
            let now = 2.0 * twist.z.atan2(twist.w);
            let n = (2.0 * PI / step).round() as usize;
            let mut turns: Vec<f64> = (0..n).map(|k| k as f64 * step).collect();
            let wrap = |x: f64| (x + PI).rem_euclid(2.0 * PI) - PI;
            turns.sort_by(|x, y| wrap(x - now).abs().total_cmp(&wrap(y - now).abs()));
            twist = DQuat::from_rotation_z(turns[turn % turns.len()]);
        } else if !inter.free_twist() && !(inter == Interaction::Flush && a.round && b.round) {
            twist = DQuat::IDENTITY;
        }
        twist
    };
    Rigid { rot: rot.normalize(), shift: DVec3::ZERO }
}

/// A few vertices spread over a part, and where they were, to tell how it moved.
#[derive(Clone, Debug)]
struct Anchor {
    shape: u64,
    topo: u64,
    verts: Vec<u32>,
    at: Vec<DVec3>,
}

/// What dragging a part moves.
#[derive(Clone, Debug, PartialEq)]
pub enum Drive {
    /// It is held by this grounded part (itself, or through fixed connections).
    Grounded(ObjectId),
    /// No connection holds it: its whole group moves freely.
    Free(Vec<ObjectId>),
    /// It turns (or slides) at `connection`, moving `parts`; `frame` is the connection's frame
    /// on the side that stays.
    Connection { connection: u64, parts: Vec<ObjectId>, frame: JFrame, moving_child: bool },
}

impl Drive {
    pub fn parts(&self) -> &[ObjectId] {
        match self {
            Drive::Grounded(_) => &[],
            Drive::Free(p) | Drive::Connection { parts: p, .. } => p,
        }
    }
}

/// What is selected in the object browser or the Assemble tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selected {
    Joint(u64),
    Connection(u64),
}

/// The grounded parts, joints and connections of a scene.
#[derive(Clone, Debug, Default)]
pub struct Assembly {
    pub grounded: Vec<ObjectId>,
    pub joints: Vec<Joint>,
    pub connections: Vec<Connection>,
    /// The joint or connection selected (not part of the undo history: undo keeps it while it
    /// is there).
    pub selected: Option<Selected>,
    anchors: FastMap<ObjectId, Anchor>,
    next_id: u64,
}

impl Assembly {
    pub fn is_empty(&self) -> bool {
        self.grounded.is_empty() && self.joints.is_empty() && self.connections.is_empty()
    }

    pub fn is_grounded(&self, id: ObjectId) -> bool {
        self.grounded.contains(&id)
    }

    pub fn set_grounded(&mut self, id: ObjectId, on: bool) {
        self.grounded.retain(|&g| g != id);
        if on {
            self.grounded.push(id);
        }
    }

    fn new_id(&mut self) -> u64 {
        let used = self.joints.iter().map(|j| j.id).chain(self.connections.iter().map(|c| c.id)).max().unwrap_or(0);
        self.next_id = self.next_id.max(used) + 1;
        self.next_id
    }

    pub fn joint(&self, id: u64) -> Option<&Joint> {
        self.joints.iter().find(|j| j.id == id)
    }

    pub fn joint_mut(&mut self, id: u64) -> Option<&mut Joint> {
        self.joints.iter_mut().find(|j| j.id == id)
    }

    /// Part `id`'s joints.
    pub fn joints_of(&self, id: ObjectId) -> impl Iterator<Item = &Joint> {
        self.joints.iter().filter(move |j| j.part == id)
    }

    /// Add a joint (named after its shape when it has no name). Returns its id.
    pub fn add_joint(&mut self, mut j: Joint) -> u64 {
        j.id = self.new_id();
        if j.name.is_empty() {
            j.name = self.free_name(j.part, j.shape, j.id);
        }
        let id = j.id;
        self.joints.push(j);
        id
    }

    /// The first "<Shape> n" no other joint on `part` than `except` is called.
    pub fn free_name(&self, part: ObjectId, shape: JointShape, except: u64) -> String {
        (1..)
            .map(|n| format!("{} {n}", shape.label()))
            .find(|name| !self.joints_of(part).any(|o| o.id != except && o.name == *name))
            .unwrap()
    }

    /// Joint `id`, made another shape than `old`: a name it was given as an `old` ("Pin 1")
    /// follows the new shape ("Ball 1"); a name the user gave it stays.
    pub fn rename_for_shape(&mut self, id: u64, old: JointShape) {
        let Some(j) = self.joint(id) else { return };
        let auto = j.name.strip_prefix(old.label()).and_then(|n| n.strip_prefix(' ')).is_some_and(|n| n.parse::<u32>().is_ok());
        if auto && j.shape != old {
            let name = self.free_name(j.part, j.shape, id);
            self.joint_mut(id).unwrap().name = name;
        }
    }

    /// Remove a joint and its connection.
    pub fn remove_joint(&mut self, id: u64) {
        self.joints.retain(|j| j.id != id);
        self.connections.retain(|c| c.a != id && c.b != id);
        self.drop_stale_selection();
    }

    pub fn connection(&self, id: u64) -> Option<&Connection> {
        self.connections.iter().find(|c| c.id == id)
    }

    pub fn connection_mut(&mut self, id: u64) -> Option<&mut Connection> {
        self.connections.iter_mut().find(|c| c.id == id)
    }

    /// The connection held by joint `joint`.
    pub fn connection_at(&self, joint: u64) -> Option<&Connection> {
        self.connections.iter().find(|c| c.a == joint || c.b == joint)
    }

    pub fn remove_connection(&mut self, id: u64) {
        self.connections.retain(|c| c.id != id);
        self.drop_stale_selection();
    }

    /// Forget the selection if what it selects is gone.
    fn drop_stale_selection(&mut self) {
        let there = match self.selected {
            Some(Selected::Joint(j)) => self.joint(j).is_some(),
            Some(Selected::Connection(c)) => self.connection(c).is_some(),
            None => true,
        };
        if !there {
            self.selected = None;
        }
    }

    /// The selected connection, if a connection is selected.
    pub fn selected_connection(&self) -> Option<&Connection> {
        match self.selected {
            Some(Selected::Connection(c)) => self.connection(c),
            _ => None,
        }
    }

    /// The selected joint, if a joint is selected.
    pub fn selected_joint(&self) -> Option<&Joint> {
        match self.selected {
            Some(Selected::Joint(j)) => self.joint(j),
            _ => None,
        }
    }

    /// Drop everything about part `id` (it left the scene).
    pub fn forget(&mut self, id: ObjectId) {
        self.grounded.retain(|&g| g != id);
        self.connections.retain(|c| !c.joins(id));
        self.joints.retain(|j| j.part != id);
        self.anchors.remove(&id);
        self.drop_stale_selection();
    }

    /// The connection `id` hangs from.
    pub fn parent_connection(&self, id: ObjectId) -> Option<&Connection> {
        self.connections.iter().find(|c| c.child == id)
    }

    /// Every part joined to `id`, directly or through others, `id` first, nearest first.
    pub fn component(&self, id: ObjectId) -> Vec<ObjectId> {
        self.reach(id, None)
    }

    /// The parts on `id`'s side of connection `cut`.
    pub fn side(&self, cut: u64, id: ObjectId) -> Vec<ObjectId> {
        self.reach(id, Some(cut))
    }

    fn reach(&self, id: ObjectId, cut: Option<u64>) -> Vec<ObjectId> {
        let mut out = vec![id];
        let mut i = 0;
        while i < out.len() {
            let p = out[i];
            i += 1;
            for c in self.connections.iter().filter(|c| Some(c.id) != cut) {
                if let Some(o) = c.other(p) {
                    if !out.contains(&o) {
                        out.push(o);
                    }
                }
            }
        }
        out
    }

    pub fn connected(&self, a: ObjectId, b: ObjectId) -> bool {
        self.component(a).contains(&b)
    }

    /// The part `id`'s group hangs from: the nearest grounded part in it, or else the part
    /// reached by going up from `id` (to each connection's parent) as far as it goes.
    pub fn root_of(&self, id: ObjectId) -> ObjectId {
        if let Some(&g) = self.component(id).iter().find(|&&p| self.is_grounded(p)) {
            return g;
        }
        let mut at = id;
        let mut seen = vec![id];
        while let Some(c) = self.parent_connection(at) {
            if seen.contains(&c.parent) {
                break;
            }
            at = c.parent;
            seen.push(at);
        }
        at
    }

    /// The connections on the way from `from` to `to`, in order (empty if not connected).
    pub fn path(&self, from: ObjectId, to: ObjectId) -> Vec<u64> {
        let mut came: Vec<(ObjectId, Option<(ObjectId, u64)>)> = vec![(from, None)];
        let mut i = 0;
        while i < came.len() {
            let p = came[i].0;
            if p == to {
                break;
            }
            i += 1;
            for c in &self.connections {
                if let Some(o) = c.other(p) {
                    if !came.iter().any(|x| x.0 == o) {
                        came.push((o, Some((p, c.id))));
                    }
                }
            }
        }
        let mut out = Vec::new();
        let mut at = to;
        while let Some(&(_, Some((prev, cid)))) = came.iter().find(|x| x.0 == at) {
            out.push(cid);
            at = prev;
        }
        out.reverse();
        out
    }

    /// What dragging part `id` moves.
    pub fn drive(&self, id: ObjectId) -> Drive {
        let root = self.root_of(id);
        for cid in self.path(id, root) {
            let Some(c) = self.connection(cid) else { continue };
            if c.kind == Motion::Fixed {
                continue;
            }
            let parts = self.side(cid, id);
            if let Some(&g) = parts.iter().find(|&&p| self.is_grounded(p)) {
                return Drive::Grounded(g);
            }
            let moving_child = parts.contains(&c.child);
            let frame = if moving_child { c.on_parent() } else { c.on_child() };
            return Drive::Connection { connection: cid, parts, frame, moving_child };
        }
        if self.is_grounded(root) {
            Drive::Grounded(root)
        } else {
            Drive::Free(self.component(id))
        }
    }

    /// Make `id` the top of its group (no connection's child), turning round the connections
    /// above it (their joints and seats stay; only which side is the parent changes).
    pub fn reroot(&mut self, id: ObjectId) {
        let mut up = Vec::new();
        let mut at = id;
        let mut seen = vec![id];
        while let Some(c) = self.connections.iter().find(|c| c.child == at && !up.contains(&c.id)) {
            up.push(c.id);
            if seen.contains(&c.parent) {
                break;
            }
            seen.push(c.parent);
            at = c.parent;
        }
        for c in self.connections.iter_mut().filter(|c| up.contains(&c.id)) {
            std::mem::swap(&mut c.parent, &mut c.child);
        }
    }

    /// Connect joint `b` (its part to be carried) to joint `a`, seated by `seat` (within a's
    /// anchor). A connection either joint held before is removed (one per joint). The parts must
    /// be in different groups. Returns the connection's id.
    pub fn connect(&mut self, a: u64, b: u64, interaction: Interaction, kind: Motion, seat: Rigid, about: String) -> Result<u64, String> {
        let (ja, jb) = match (self.joint(a), self.joint(b)) {
            (Some(x), Some(y)) => (x.clone(), y.clone()),
            _ => return Err("a joint is gone".into()),
        };
        if ja.part == jb.part {
            return Err("Both joints are on the same part.".into());
        }
        self.connections.retain(|c| c.a != a && c.b != a && c.a != b && c.b != b);
        if self.connected(ja.part, jb.part) {
            return Err("These parts are already connected (perhaps through others): disconnect them first.".into());
        }
        let (on_a, on_b) = anchors(&ja, &jb, interaction);
        let id = self.new_id();
        // Two points have no size: a millimetre, to see it.
        let size = joints_size(&ja, &jb).unwrap_or(1.0);
        self.connections.push(Connection { id, a, b, part_a: ja.part, part_b: jb.part, parent: ja.part, child: jb.part, interaction, kind, seat, on_a, on_b, size, about });
        Ok(id)
    }

    /// Work out connection `id`'s anchors from its joints again (after a joint or the
    /// interaction changed).
    pub fn refresh_anchors(&mut self, id: u64) {
        let Some(c) = self.connection(id) else { return };
        let (Some(ja), Some(jb)) = (self.joint(c.a).cloned(), self.joint(c.b).cloned()) else { return };
        let (on_a, on_b) = anchors(&ja, &jb, c.interaction);
        let size = joints_size(&ja, &jb).unwrap_or(c.size);
        if let Some(c) = self.connection_mut(id) {
            (c.on_a, c.on_b, c.size) = (on_a, on_b, size);
        }
    }

    /// Move the frames on `parts` by `m` (their meshes moved by it).
    fn move_frames(&mut self, parts: &[ObjectId], m: &Rigid) {
        for j in &mut self.joints {
            if parts.contains(&j.part) {
                j.base = j.base.moved(m);
            }
        }
        for c in &mut self.connections {
            if parts.contains(&c.part_a) {
                c.on_a = c.on_a.moved(m);
            }
            if parts.contains(&c.part_b) {
                c.on_b = c.on_b.moved(m);
            }
        }
    }

    /// Every part with a joint or a connection.
    fn jointed(&self) -> Vec<ObjectId> {
        let mut v: Vec<ObjectId> = self.joints.iter().map(|j| j.part).chain(self.connections.iter().flat_map(|c| [c.part_a, c.part_b])).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// A connection's size from its joints' (the smaller, or the one that has a size), none for two
/// points.
fn joints_size(a: &Joint, b: &Joint) -> Option<f64> {
    match (a.size(), b.size()) {
        (x, y) if x > 0.0 && y > 0.0 => Some(x.min(y)),
        (x, y) if x > 0.0 || y > 0.0 => Some(x.max(y)),
        _ => None,
    }
}

/// Move `parts` (meshes, frames, pivots, joints and connection anchors) by `m`.
pub fn move_parts(scene: &mut Scene, parts: &[ObjectId], m: &Rigid) {
    for &id in parts {
        let Some(o) = scene.get_mut(id) else { continue };
        o.mesh_mut().transform(|p| m.apply(p));
        o.frame = (m.rot * o.frame).normalize();
        for p in &mut o.pivots {
            p.position = m.apply(p.position);
            p.frame = (m.rot * p.frame).normalize();
        }
        o.touch_selection();
    }
    scene.assembly.move_frames(parts, m);
    // The anchor vertices moved with their meshes.
    for &id in parts {
        let Some(o) = scene.objects.iter().find(|o| o.id == id) else { continue };
        if let Some(a) = scene.assembly.anchors.get_mut(&id) {
            if a.topo == o.mesh.topology_version() && a.verts.iter().all(|&v| o.mesh.is_vertex(v)) {
                a.at = a.verts.iter().map(|&v| o.mesh.vertex(v)).collect();
                a.shape = o.mesh.shape_version();
            }
        }
    }
}

/// Move one side of connection `id` so that its child moves by `m` relative to its parent: the
/// child's side if nothing grounded is on it, else the parent's side the other way.
pub fn move_connection(scene: &mut Scene, id: u64, m: &Rigid) -> Result<(), String> {
    let asm = &scene.assembly;
    let c = asm.connection(id).ok_or("the connection is gone")?;
    let child_side = asm.side(id, c.child);
    let parent_side = asm.side(id, c.parent);
    let held = |side: &[ObjectId]| side.iter().any(|&p| asm.is_grounded(p));
    let (parts, m) = if !held(&child_side) {
        (child_side, *m)
    } else if !held(&parent_side) {
        (parent_side, m.inverse())
    } else {
        return Err("Both sides of this connection are held by grounded parts.".into());
    };
    move_parts(scene, &parts, &m);
    Ok(())
}

/// Change joint `id` with `edit` and re-seat what is connected to it, keeping its pose (a new
/// radius moves a part resting on a ball in or out, a moved joint takes the part along). A
/// connection whose interaction the new shape cannot make takes the usual one (said in the
/// notes returned).
pub fn edit_joint(scene: &mut Scene, id: u64, edit: impl FnOnce(&mut Joint)) -> Result<Vec<String>, String> {
    let asm = &mut scene.assembly;
    let poses: Vec<(u64, DQuat, DVec3)> = asm.connections.iter().filter(|c| c.a == id || c.b == id).map(|c| (c.id, c.pose().0, c.pose().1)).collect();
    edit(asm.joint_mut(id).ok_or("the joint is gone")?);
    let mut notes = Vec::new();
    for &(cid, _, _) in &poses {
        let asm = &mut scene.assembly;
        let Some(c) = asm.connection(cid) else { continue };
        let (Some(ja), Some(jb)) = (asm.joint(c.a).cloned(), asm.joint(c.b).cloned()) else { continue };
        let allowed = Interaction::between(ja.shape, jb.shape);
        let c = asm.connection_mut(cid).unwrap();
        c.about = format!("{} in {}", jb.describe(), ja.describe());
        if !allowed.contains(&c.interaction) {
            c.interaction = allowed[0];
            notes.push(format!("A {} and a {} make it {} now.", ja.shape.label().to_lowercase(), jb.shape.label().to_lowercase(), c.interaction.label().to_lowercase()));
        }
    }
    let mut result = Ok(notes);
    for (cid, rel, off) in poses {
        scene.assembly.refresh_anchors(cid);
        let m = scene.assembly.connection(cid).map(|c| c.motion_to(rel, off));
        if let Some(m) = m {
            if let Err(e) = move_connection(scene, cid, &m) {
                result = Err(e);
            }
        }
    }
    result
}

/// Fit joint `id` again to its surface as `kind` (`None`: whichever fits best), made `shape`
/// when given (a ball fitted to a concave surface stays a ball). Its place and size come from
/// the fit; its offset stays, and so does its name unless it was the old shape's ("Pin 1"
/// becomes "Ball 1"). Connected parts are re-seated, keeping their pose (see [`edit_joint`]).
/// Returns what it is now, and edit_joint's notes.
pub fn refit_joint(scene: &mut Scene, id: u64, kind: Option<ShapeKind>, shape: Option<JointShape>) -> Result<String, String> {
    let j = scene.assembly.joint(id).ok_or("the joint is gone")?;
    let o = scene.get(j.part).ok_or("the part is gone")?;
    let what = |k: Option<ShapeKind>| match (shape, k) {
        (Some(s), _) => s.label().to_lowercase(),
        (None, Some(k)) => k.label().to_lowercase(),
        (None, None) => "ball, pin or flat face".into(),
    };
    let tris = j.surface_on(&o.mesh).ok_or_else(|| format!("{}'s surface is not known (a joint from a saved project, or the part was edited): Fit again… paints it.", j.name))?;
    let f = mate::fit_region(&o.mesh, tris, kind).ok_or_else(|| format!("No {} fits the {} triangles of {}'s surface.", what(kind), tris.len(), j.name))?;
    let mut fresh = Joint::from_feature(j.part, &f, &o.mesh);
    if let Some(s) = shape.filter(|&s| s != fresh.shape) {
        fresh.set_shape(s);
        let surf = fresh.surface.as_deref().expect("fitted");
        fresh.surface = Some(Arc::new(JointSurface { tris: surf.tris.clone(), topo: surf.topo, shape: s, radius: fresh.radius }));
    }
    let (old, old_shape, offset, n) = (j.name.clone(), j.shape, j.offset, tris.len());
    fresh.name = old.clone();
    let now = fresh.describe();
    let notes = edit_joint(scene, id, |jj| *jj = Joint { id, offset, ..fresh })?;
    scene.assembly.rename_for_shape(id, old_shape);
    let name = scene.assembly.joint(id).map_or_else(String::new, |j| j.name.clone());
    let renamed = if name != old { format!("{old} is {name} now: ") } else { format!("{name}: ") };
    Ok(format!("{renamed}{now}, fitted to its {n} triangles. {}", notes.join(" ")).trim_end().to_string())
}

/// Change connection `id` with `edit` (its interaction, motion or seat), then re-seat the child
/// so that it keeps the pose the new motion allows.
pub fn edit_connection(scene: &mut Scene, id: u64, edit: impl FnOnce(&mut Connection)) -> Result<(), String> {
    let c = scene.assembly.connection(id).ok_or("the connection is gone")?;
    let (rel, off) = c.pose();
    edit(scene.assembly.connection_mut(id).unwrap());
    scene.assembly.refresh_anchors(id);
    let c = scene.assembly.connection(id).unwrap();
    let (rel, off) = constrain(c.kind, rel, off);
    let m = c.motion_to(rel, off);
    move_connection(scene, id, &m)
}

/// Bring the assembly up to date with the scene: forget parts that left it, and move the frames
/// of parts that other tools moved. Cheap when nothing changed.
pub fn follow(scene: &mut Scene) {
    if scene.assembly.is_empty() {
        return;
    }
    let (objects, asm) = (&scene.objects, &mut scene.assembly);
    let alive = |id: ObjectId| objects.iter().any(|o| o.id == id);
    asm.grounded.retain(|&g| alive(g));
    asm.joints.retain(|j| alive(j.part));
    let joint_ids: Vec<u64> = asm.joints.iter().map(|j| j.id).collect();
    asm.connections.retain(|c| alive(c.part_a) && alive(c.part_b) && joint_ids.contains(&c.a) && joint_ids.contains(&c.b));
    asm.drop_stale_selection();
    let parts = asm.jointed();
    asm.anchors.retain(|id, _| parts.contains(id));
    for id in parts {
        let Some(o) = objects.iter().find(|o| o.id == id) else { continue };
        let m = &o.mesh;
        let size = m.bounds().diagonal().max(1e-9);
        let moved = match asm.anchors.get(&id) {
            Some(a) if a.shape == m.shape_version() => continue,
            Some(a) if a.topo == m.topology_version() && !a.verts.is_empty() && a.verts.iter().all(|&v| m.is_vertex(v)) => {
                let now: Vec<DVec3> = a.verts.iter().map(|&v| m.vertex(v)).collect();
                let r = rigid_fit(&a.at, &now);
                // Only a rigid move (Transform, Align) carries the frames; a reshape (a brush,
                // Smooth, a scale) leaves them where they are.
                let residual = a.at.iter().zip(&now).map(|(p, q)| (r.apply(*p) - *q).length()).fold(0.0, f64::max);
                (residual <= 1e-6 * size).then_some(r)
            }
            // New, or remade (its triangles changed): it is taken to be where it is.
            _ => None,
        };
        if let Some(r) = moved {
            if !r.is_identity_within(size, 1e-9 * size) {
                asm.move_frames(&[id], &r);
            }
        }
        asm.anchors.insert(id, anchor_of(m));
    }
}

fn frame_text(f: &JFrame) -> String {
    format!("{} {} {} {} {} {} {}", f.origin.x, f.origin.y, f.origin.z, f.rot.x, f.rot.y, f.rot.z, f.rot.w)
}

fn frame_of(v: &[f64]) -> JFrame {
    JFrame { origin: DVec3::new(v[0], v[1], v[2]), rot: DQuat::from_xyzw(v[3], v[4], v[5], v[6]).normalize() }
}

/// The assembly as text for a project file (`assembly.txt`), parts by their index in `order`;
/// none when nothing is grounded, has a joint or is connected.
///
/// ```text
/// polymix-assembly 2
/// ground 0
/// joint <id> <part> <shape> <radius> <length> <half x> <half y> <round 0/1> <base: x y z qx qy qz qw> <offset: x y z>
/// name Shoulder L
/// fitted socket r 2.72
/// connection <id> <joint a> <joint b> <parent> <child> <interaction> <motion> <size> <seat: qx qy qz qw x y z> <anchor a: 7> <anchor b: 7>
/// about ball r 2.07 in socket r 2.72
/// ```
///
/// Version 1 files (connections only, from before joints belonged to parts) still read.
pub fn to_text(asm: &Assembly, order: &[ObjectId]) -> Option<String> {
    let index = |id: ObjectId| order.iter().position(|&o| o == id);
    let one_line = |s: &str| s.replace(['\n', '\r'], " ");
    let mut s = String::from("polymix-assembly 2\n");
    let mut any = false;
    for &g in &asm.grounded {
        if let Some(i) = index(g) {
            s += &format!("ground {i}\n");
            any = true;
        }
    }
    let mut saved = Vec::new();
    for j in &asm.joints {
        let Some(p) = index(j.part) else { continue };
        let o = j.offset;
        s += &format!("joint {} {p} {} {} {} {} {} {} {} {} {} {}\n", j.id, j.shape.key(), j.radius, j.length, j.half.x, j.half.y, j.round as u8, frame_text(&j.base), o.x, o.y, o.z);
        s += &format!("name {}\n", one_line(&j.name));
        s += &format!("fitted {}\n", one_line(&j.fitted));
        saved.push(j.id);
        any = true;
    }
    for c in &asm.connections {
        let (Some(p), Some(ch)) = (index(c.parent), index(c.child)) else { continue };
        if !saved.contains(&c.a) || !saved.contains(&c.b) {
            continue;
        }
        let (q, t) = (c.seat.rot, c.seat.shift);
        s += &format!("connection {} {} {} {p} {ch} {} {} {} {} {} {} {} {} {} {} {} {}\n", c.id, c.a, c.b, c.interaction.label(), c.kind.label(), c.size, q.x, q.y, q.z, q.w, t.x, t.y, t.z, frame_text(&c.on_a), frame_text(&c.on_b));
        s += &format!("about {}\n", one_line(&c.about));
    }
    any.then_some(s)
}

/// The assembly saved by [`to_text`], for the parts `ids` (in the saved order). Lines it cannot
/// read are passed by.
pub fn from_text(text: &str, ids: &[ObjectId]) -> Assembly {
    let mut asm = Assembly::default();
    let mut lines = text.lines();
    let version = match lines.next().map(str::trim) {
        Some("polymix-assembly 1") => 1,
        Some("polymix-assembly 2") => 2,
        _ => return asm,
    };
    let part = |s: &str| s.parse::<usize>().ok().and_then(|i| ids.get(i).copied());
    let nums = |w: &[&str]| -> Vec<f64> { w.iter().filter_map(|x| x.parse().ok()).collect() };
    // What the last joint or connection line was, for its name / fitted / about line.
    enum Last {
        None,
        Joint(u64),
        Connection(u64),
    }
    let mut last = Last::None;
    for line in lines {
        let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
        let w: Vec<&str> = rest.split_whitespace().collect();
        match key {
            "ground" => {
                if let Some(id) = part(rest.trim()) {
                    asm.set_grounded(id, true);
                }
            }
            // Version 1: a connection between two parts, as its two frames (joints as points).
            "joint" if version == 1 => {
                let Some(kind) = w.first().and_then(|k| Motion::ALL.into_iter().find(|j| j.label() == *k)) else { continue };
                let (Some(p), Some(c)) = (w.get(1).and_then(|x| part(x)), w.get(2).and_then(|x| part(x))) else { continue };
                let v = nums(&w[3..]);
                if !(v.len() == 15 || v.len() == 21) || !v.iter().all(|x| x.is_finite()) || p == c {
                    continue;
                }
                let (fp, fc) = (frame_of(&v[1..8]), frame_of(&v[8..15]));
                let a = asm.add_joint(Joint::point(p, fp));
                // b's anchor is its frame turned half round about x: undo that here.
                let b = asm.add_joint(Joint::point(c, JFrame { origin: fc.origin, rot: (fc.rot * DQuat::from_rotation_x(PI)).normalize() }));
                if let Ok(id) = asm.connect(a, b, Interaction::Coincident, kind, Rigid::IDENTITY, String::new()) {
                    if let Some(cn) = asm.connection_mut(id) {
                        cn.size = v[0];
                    }
                    last = Last::Connection(id);
                }
            }
            "joint" => {
                let Some(&id_s) = w.first() else { continue };
                let (Ok(id), Some(p), Some(shape)) = (id_s.parse::<u64>(), w.get(1).and_then(|x| part(x)), w.get(2).and_then(|k| JointShape::from_key(k))) else { continue };
                let v = nums(&w[3..]);
                if v.len() != 15 || !v.iter().all(|x| x.is_finite()) {
                    continue;
                }
                asm.joints.push(Joint {
                    id,
                    part: p,
                    name: String::new(),
                    shape,
                    radius: v[0],
                    length: v[1],
                    half: DVec2::new(v[2], v[3]),
                    round: v[4] != 0.0,
                    base: frame_of(&v[5..12]),
                    offset: DVec3::new(v[12], v[13], v[14]),
                    surface: None,
                    fitted: String::new(),
                });
                last = Last::Joint(id);
            }
            "name" | "fitted" => {
                if let Last::Joint(id) = last {
                    if let Some(j) = asm.joint_mut(id) {
                        if key == "name" {
                            j.name = rest.to_string();
                        } else {
                            j.fitted = rest.to_string();
                        }
                    }
                }
            }
            "connection" => {
                let ids3: Vec<Option<u64>> = w.iter().take(3).map(|x| x.parse().ok()).collect();
                let (Some(Some(id)), Some(Some(a)), Some(Some(b))) = (ids3.first(), ids3.get(1), ids3.get(2)) else { continue };
                let (Some(p), Some(ch)) = (w.get(3).and_then(|x| part(x)), w.get(4).and_then(|x| part(x))) else { continue };
                let (Some(inter), Some(kind)) = (w.get(5).and_then(|k| Interaction::from_label(k)), w.get(6).and_then(|k| Motion::ALL.into_iter().find(|m| m.label() == *k))) else { continue };
                let v = nums(&w[7..]);
                let (Some(ja), Some(jb)) = (asm.joint(*a).cloned(), asm.joint(*b).cloned()) else { continue };
                if v.len() != 22 || !v.iter().all(|x| x.is_finite()) || ja.part == jb.part || ![ja.part, jb.part].contains(&p) || ![ja.part, jb.part].contains(&ch) || p == ch {
                    continue;
                }
                asm.connections.push(Connection {
                    id: *id,
                    a: *a,
                    b: *b,
                    part_a: ja.part,
                    part_b: jb.part,
                    parent: p,
                    child: ch,
                    interaction: inter,
                    kind,
                    size: v[0],
                    seat: Rigid { rot: DQuat::from_xyzw(v[1], v[2], v[3], v[4]).normalize(), shift: DVec3::new(v[5], v[6], v[7]) },
                    on_a: frame_of(&v[8..15]),
                    on_b: frame_of(&v[15..22]),
                    about: String::new(),
                });
                last = Last::Connection(*id);
            }
            "about" => {
                if let Last::Connection(id) = last {
                    if let Some(c) = asm.connection_mut(id) {
                        c.about = rest.to_string();
                    }
                }
            }
            _ => {}
        }
    }
    asm.next_id = asm.joints.iter().map(|j| j.id).chain(asm.connections.iter().map(|c| c.id)).max().unwrap_or(0);
    asm
}

/// Up to six vertices spread over `m` (each the farthest from those before).
fn anchor_of(m: &Mesh) -> Anchor {
    let ids: Vec<u32> = m.vertex_ids().collect();
    let mut verts = Vec::new();
    if !ids.is_empty() {
        let c = ids.iter().map(|&v| m.vertex(v)).sum::<DVec3>() / ids.len() as f64;
        let far = |from: DVec3| ids.iter().copied().max_by(|&a, &b| (m.vertex(a) - from).length_squared().total_cmp(&(m.vertex(b) - from).length_squared()));
        let first = far(c).unwrap();
        verts.push(first);
        let mut near: Vec<f64> = ids.iter().map(|&v| (m.vertex(v) - m.vertex(first)).length_squared()).collect();
        while verts.len() < 6 {
            let (i, &d) = near.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            if d <= 0.0 {
                break;
            }
            let v = ids[i];
            verts.push(v);
            let p = m.vertex(v);
            for (k, &w) in ids.iter().enumerate() {
                near[k] = near[k].min((m.vertex(w) - p).length_squared());
            }
        }
    }
    Anchor { shape: m.shape_version(), topo: m.topology_version(), at: verts.iter().map(|&v| m.vertex(v)).collect(), verts }
}

/// The rigid motion that best takes the points `from` onto `to` (Kabsch).
fn rigid_fit(from: &[DVec3], to: &[DVec3]) -> Rigid {
    let n = from.len().min(to.len());
    if n == 0 {
        return Rigid::IDENTITY;
    }
    let ca = from[..n].iter().sum::<DVec3>() / n as f64;
    let cb = to[..n].iter().sum::<DVec3>() / n as f64;
    let mut s = DMat3::ZERO;
    for (a, b) in from[..n].iter().zip(&to[..n]) {
        let (a, b) = (*a - ca, *b - cb);
        s += DMat3::from_cols(b * a.x, b * a.y, b * a.z);
    }
    let rot = if n >= 3 { DQuat::from_mat3(&mm_core::linalg::closest_rotation(s)).normalize() } else { DQuat::IDENTITY };
    Rigid { rot, shift: cb - rot * ca }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn near(a: DVec3, b: DVec3) -> bool {
        (a - b).length() < 1e-6
    }

    fn joint(part: ObjectId, shape: JointShape, origin: DVec3, axis: DVec3, radius: f64) -> Joint {
        let mut j = Joint::point(part, JFrame::new(origin, axis, None));
        j.shape = shape;
        j.radius = radius;
        j.half = DVec2::splat(radius);
        j
    }

    /// Torso (grounded) ← arm (ball in the shoulder socket) ← hand (pin in the elbow hole) ←
    /// sword (flush peg); a loose shield.
    fn kit() -> (Scene, [ObjectId; 5], [u64; 3]) {
        let mut s = Scene::default();
        let torso = s.add("torso", primitives::cube(2.0, 1));
        let mut arm = primitives::cube(0.5, 1);
        arm.transform(|p| p + DVec3::new(1.5, 0.0, 0.0));
        let arm = s.add("arm", arm);
        let mut hand = primitives::cube(0.3, 1);
        hand.transform(|p| p + DVec3::new(2.2, 0.0, 0.0));
        let hand = s.add("hand", hand);
        let mut sword = primitives::cube(0.2, 1);
        sword.transform(|p| p + DVec3::new(2.2, 0.0, 0.5));
        let sword = s.add("sword", sword);
        let shield = s.add("shield", primitives::cube(0.4, 1));
        let a = &mut s.assembly;
        a.set_grounded(torso, true);
        // The socket opens along +x; the arm's ball points back into it (−x).
        let socket = a.add_joint(joint(torso, JointShape::Socket, DVec3::new(1.0, 0.0, 0.0), DVec3::X, 0.26));
        let ball = a.add_joint(joint(arm, JointShape::Ball, DVec3::new(1.0, 0.0, 0.0), -DVec3::X, 0.25));
        let hole = a.add_joint(joint(arm, JointShape::Hole, DVec3::new(1.9, 0.0, 0.0), DVec3::Y, 0.1));
        let pin = a.add_joint(joint(hand, JointShape::Pin, DVec3::new(1.9, 0.0, 0.0), -DVec3::Y, 0.09));
        let recess = a.add_joint(joint(hand, JointShape::Flat, DVec3::new(2.2, 0.0, 0.3), DVec3::Z, 0.1));
        let peg = a.add_joint(joint(sword, JointShape::Flat, DVec3::new(2.2, 0.0, 0.3), -DVec3::Z, 0.1));
        let shoulder = a.connect(socket, ball, Interaction::Concentric, Motion::Ball, initial_seat(a.joint(socket).unwrap(), a.joint(ball).unwrap(), Interaction::Concentric, false, 0), "ball".into()).unwrap();
        let elbow = a.connect(hole, pin, Interaction::Coaxial, Motion::Hinge, initial_seat(a.joint(hole).unwrap(), a.joint(pin).unwrap(), Interaction::Coaxial, false, 0), "pin".into()).unwrap();
        let grip = a.connect(recess, peg, Interaction::Flush, Motion::Fixed, initial_seat(a.joint(recess).unwrap(), a.joint(peg).unwrap(), Interaction::Flush, false, 0), "peg".into()).unwrap();
        (s, [torso, arm, hand, sword, shield], [shoulder, elbow, grip])
    }

    #[test]
    fn connections_start_at_rest_and_dragging_turns_the_first_that_can_move() {
        let (s, [torso, arm, hand, sword, shield], [shoulder, elbow, grip]) = kit();
        let a = &s.assembly;
        for c in [shoulder, elbow, grip] {
            let c = a.connection(c).unwrap();
            assert!(c.error() < 1e-9 && c.pose().0.angle_between(DQuat::IDENTITY) < 1e-9 && c.pose().1.length() < 1e-9, "{} at rest", c.about);
        }
        assert_eq!(a.drive(torso), Drive::Grounded(torso));
        match a.drive(arm) {
            Drive::Connection { connection, parts, moving_child, .. } => {
                assert_eq!(connection, shoulder);
                assert_eq!(parts, vec![arm, hand, sword]);
                assert!(moving_child);
            }
            d => panic!("{d:?}"),
        }
        for p in [hand, sword] {
            assert!(matches!(a.drive(p), Drive::Connection { connection, .. } if connection == elbow));
        }
        assert_eq!(a.drive(shield), Drive::Free(vec![shield]));
        assert_eq!(a.path(sword, torso).len(), 3);
    }

    #[test]
    fn a_joint_holds_one_connection_and_parts_connect_once() {
        let (mut s, [torso, arm, ..], [shoulder, ..]) = kit();
        let a = &mut s.assembly;
        let socket = a.connection(shoulder).unwrap().a;
        // Another arm in the same socket replaces the first one's connection.
        let arm2 = 999;
        let ball2 = a.add_joint(joint(arm2, JointShape::Ball, DVec3::new(5.0, 0.0, 0.0), -DVec3::X, 0.25));
        let c = a.connect(socket, ball2, Interaction::Concentric, Motion::Ball, Rigid::IDENTITY, String::new()).unwrap();
        assert!(a.connection(shoulder).is_none() && a.connection(c).is_some());
        assert!(!a.connected(torso, arm));
        // Two joints of parts already connected: refused.
        let elsewhere = a.add_joint(joint(torso, JointShape::Socket, DVec3::new(-1.0, 0.0, 0.0), -DVec3::X, 0.2));
        let other_ball = a.add_joint(joint(arm2, JointShape::Ball, DVec3::new(6.0, 0.0, 0.0), DVec3::X, 0.2));
        assert!(a.connect(elsewhere, other_ball, Interaction::Concentric, Motion::Ball, Rigid::IDENTITY, String::new()).is_err());
        // Deleting a joint takes its connection.
        a.remove_joint(ball2);
        assert!(a.connection(c).is_none());
        assert_eq!(a.joints_of(torso).count(), 2);
    }

    #[test]
    fn grounding_the_hand_turns_the_drags_round() {
        let (mut s, [torso, arm, hand, _, _], [shoulder, elbow, _]) = kit();
        s.assembly.set_grounded(torso, false);
        s.assembly.set_grounded(hand, true);
        match s.assembly.drive(torso) {
            Drive::Connection { connection, parts, moving_child, .. } => {
                assert_eq!((connection, parts), (shoulder, vec![torso]));
                assert!(!moving_child);
            }
            d => panic!("{d:?}"),
        }
        assert!(matches!(s.assembly.drive(arm), Drive::Connection { connection, ref parts, .. } if connection == elbow && *parts == vec![arm, torso]));
        s.assembly.set_grounded(hand, false);
        assert_eq!(s.assembly.root_of(hand), torso);
        // Turned round: the poses read the same, only the sides swap.
        let side = s.assembly.side(shoulder, arm);
        move_parts(&mut s, &side, &Rigid::about(DQuat::from_rotation_z(0.4), DVec3::new(1.0, 0.0, 0.0)));
        let before = s.assembly.connection(shoulder).unwrap().pose().0;
        s.assembly.reroot(hand);
        assert_eq!(s.assembly.root_of(torso), hand);
        let c = s.assembly.connection(shoulder).unwrap();
        assert_eq!((c.parent, c.child), (arm, torso));
        assert!(c.error() < 1e-9);
        assert!((c.pose().0 * before).angle_between(DQuat::IDENTITY) < 1e-9, "the inverse pose");
    }

    #[test]
    fn moving_parts_carries_their_joints_and_anchors() {
        let (mut s, [_, arm, hand, sword, _], [shoulder, elbow, grip]) = kit();
        follow(&mut s);
        let before = s.get(hand).unwrap().mesh.vertex(0);
        let hole = s.assembly.connection(elbow).unwrap().a;
        let hole_before = s.assembly.joint(hole).unwrap().frame();
        let r = Rigid::about(DQuat::from_rotation_z(0.7), DVec3::new(1.0, 0.0, 0.0));
        move_parts(&mut s, &[arm, hand, sword], &r);
        assert!(near(s.get(hand).unwrap().mesh.vertex(0), r.apply(before)));
        assert!(near(s.assembly.joint(hole).unwrap().frame().origin, r.apply(hole_before.origin)));
        let a = &s.assembly;
        let j = a.connection(shoulder).unwrap();
        assert!(j.error() < 1e-9, "the shoulder turned about its centre");
        assert!((j.pose().0.angle_between(DQuat::IDENTITY) - 0.7).abs() < 1e-9);
        assert!(a.connection(elbow).unwrap().pose().0.angle_between(DQuat::IDENTITY) < 1e-9, "the elbow kept its pose");
        assert!(a.connection(grip).unwrap().error() < 1e-9);
        let frames: Vec<_> = a.connections.iter().map(|c| (c.on_a, c.on_b)).collect();
        follow(&mut s);
        assert_eq!(frames, s.assembly.connections.iter().map(|c| (c.on_a, c.on_b)).collect::<Vec<_>>());
    }

    #[test]
    fn a_part_moved_by_another_tool_takes_its_frames_along_but_not_when_reshaped() {
        let (mut s, [torso, arm, ..], [shoulder, ..]) = kit();
        follow(&mut s);
        let r = Rigid::about(DQuat::from_rotation_y(0.4), DVec3::new(0.3, -0.2, 0.1)).then(&Rigid::translation(DVec3::new(5.0, 1.0, -2.0)));
        s.get_mut(torso).unwrap().mesh_mut().transform(|p| r.apply(p));
        follow(&mut s);
        let c = s.assembly.connection(shoulder).unwrap().clone();
        assert!(near(c.on_a.origin, r.apply(DVec3::new(1.0, 0.0, 0.0))));
        assert!(c.is_apart());
        move_connection(&mut s, shoulder, &c.snap_motion()).unwrap();
        let c = s.assembly.connection(shoulder).unwrap();
        assert!(c.error() < 1e-9 && c.pose().0.angle_between(DQuat::IDENTITY) > 0.3, "back together, keeping its turn");
        // A brush stroke on the arm: no rigid move, the frames stay.
        let frames = (c.on_a, c.on_b);
        let m = s.get_mut(arm).unwrap().mesh_mut();
        let p = m.vertex(0);
        m.set_vertex(0, p + DVec3::new(0.05, 0.02, 0.0));
        follow(&mut s);
        let c = s.assembly.connection(shoulder).unwrap();
        assert_eq!((c.on_a, c.on_b), frames);
    }

    #[test]
    fn a_flat_face_rests_on_a_ball_and_moves_with_its_radius() {
        let mut s = Scene::default();
        let torso = s.add("torso", primitives::cube(2.0, 1));
        let arm = s.add("arm", primitives::cube(0.5, 1));
        let a = &mut s.assembly;
        a.set_grounded(torso, true);
        // A ball r 1 on the torso pointing +x, a flat face on the arm facing −x at x = 5.
        let ball = a.add_joint(joint(torso, JointShape::Ball, DVec3::new(1.0, 0.0, 0.0), DVec3::X, 1.0));
        let face = a.add_joint(joint(arm, JointShape::Flat, DVec3::new(5.0, 0.0, 0.0), -DVec3::X, 0.3));
        assert_eq!(Interaction::between(JointShape::Ball, JointShape::Flat)[0], Interaction::Tangent);
        let seat = initial_seat(a.joint(ball).unwrap(), a.joint(face).unwrap(), Interaction::Tangent, false, 0);
        let c = a.connect(ball, face, Interaction::Tangent, Motion::Ball, seat, String::new()).unwrap();
        // Seat it: the face moves to touch the ball at x = 2 (centre 1 + radius 1).
        let m = Rigid::between(&a.connection(c).unwrap().on_b, &a.connection(c).unwrap().on_parent());
        move_parts(&mut s, &[arm], &m);
        assert!(near(s.assembly.joint(face).unwrap().frame().origin, DVec3::new(2.0, 0.0, 0.0)));
        // Swing it by 30°: the face stays on the sphere of radius 1.
        let cn = s.assembly.connection(c).unwrap().clone();
        move_connection(&mut s, c, &cn.motion_to(DQuat::from_rotation_z(0.5), DVec3::ZERO)).unwrap();
        let f = s.assembly.joint(face).unwrap().frame();
        assert!(((f.origin - DVec3::new(1.0, 0.0, 0.0)).length() - 1.0).abs() < 1e-9);
        assert!(f.axis().dot((DVec3::new(1.0, 0.0, 0.0) - f.origin).normalize()) > 1.0 - 1e-9, "facing the centre");
        // A bigger ball pushes the arm out, keeping the swing.
        edit_joint(&mut s, ball, |j| j.radius = 1.5).unwrap();
        let f = s.assembly.joint(face).unwrap().frame();
        assert!(((f.origin - DVec3::new(1.0, 0.0, 0.0)).length() - 1.5).abs() < 1e-9, "{}", f.origin);
        let c = s.assembly.connection(c).unwrap();
        assert!(c.error() < 1e-9 && (c.pose().0.angle_between(DQuat::IDENTITY) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn editing_a_joint_and_the_seat_keep_the_pose() {
        let (mut s, [_, arm, ..], [shoulder, ..]) = kit();
        let side = s.assembly.side(shoulder, arm);
        move_parts(&mut s, &side, &Rigid::about(DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2), DVec3::new(1.0, 0.0, 0.0)));
        let socket = s.assembly.connection(shoulder).unwrap().a;
        // The socket moved 0.1 along its axis: the arm goes with it, still at 90°.
        let before = s.get(arm).unwrap().mesh.vertex(0);
        edit_joint(&mut s, socket, |j| j.offset.z = 0.1).unwrap();
        assert!(near(s.get(arm).unwrap().mesh.vertex(0), before + DVec3::new(0.1, 0.0, 0.0)));
        let c = s.assembly.connection(shoulder).unwrap();
        assert!(c.error() < 1e-9 && (c.pose().0.angle_between(DQuat::IDENTITY) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        // A seat offset moves the arm along the parent's anchor, also keeping the pose.
        edit_connection(&mut s, shoulder, |c| c.seat.shift = DVec3::new(0.0, 0.0, 0.05)).unwrap();
        let c = s.assembly.connection(shoulder).unwrap();
        assert!(c.error() < 1e-9 && (c.pose().0.angle_between(DQuat::IDENTITY) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        assert!(near(s.get(arm).unwrap().mesh.vertex(0), before + DVec3::new(0.15, 0.0, 0.0)));
        // A hinge keeps only the turn about its axis.
        edit_connection(&mut s, shoulder, |c| c.kind = Motion::Hinge).unwrap();
        assert!(s.assembly.connection(shoulder).unwrap().error() < 1e-9);
    }

    #[test]
    fn keyed_faces_turn_the_nearer_way_and_keep_pose_resets_the_rest() {
        let a = joint(1, JointShape::Flat, DVec3::ZERO, DVec3::Z, 1.0);
        let mut b = Joint::point(2, JFrame::new(DVec3::new(0.0, 0.0, 3.0), -DVec3::Z, Some(DVec3::new(1.0, 0.1, 0.0))));
        b.shape = JointShape::Flat;
        b.half = DVec2::new(2.0, 1.0);
        let mut a = a;
        a.half = DVec2::new(2.0, 1.0);
        let s0 = initial_seat(&a, &b, Interaction::Flush, false, 0);
        let s1 = initial_seat(&a, &b, Interaction::Flush, false, 1);
        assert!(s0.rot.angle_between(DQuat::IDENTITY) < 1e-9, "already lined up");
        assert!((s1.rot.angle_between(DQuat::IDENTITY) - PI).abs() < 1e-9, "the other way round");
        let mut c = Connection { id: 1, a: 1, b: 2, part_a: 1, part_b: 2, parent: 1, child: 2, interaction: Interaction::Flush, kind: Motion::Fixed, seat: s0, on_a: anchors(&a, &b, Interaction::Flush).0, on_b: anchors(&a, &b, Interaction::Flush).1, size: 1.0, about: String::new() };
        let (rel, off) = (DQuat::from_rotation_z(0.3), DVec3::new(0.0, 0.0, 0.2));
        c.keep_pose(rel, off);
        let (r2, o2) = c.pose();
        assert!(r2.angle_between(rel) < 1e-9 && near(o2, off));
    }

    #[test]
    fn the_assembly_reads_back_as_saved_and_old_files_still_read() {
        let (mut s, parts, [shoulder, ..]) = kit();
        let side = s.assembly.side(shoulder, parts[1]);
        move_parts(&mut s, &side, &Rigid::about(DQuat::from_rotation_z(0.7), DVec3::new(1.0, 0.0, 0.0)));
        s.assembly.joint_mut(s.assembly.connection(shoulder).unwrap().a).unwrap().name = "Shoulder L".into();
        let order: Vec<ObjectId> = s.objects.iter().map(|o| o.id).collect();
        let text = to_text(&s.assembly, &order).unwrap();
        let ids: Vec<ObjectId> = (100..105).collect();
        let map = |id: ObjectId| ids[order.iter().position(|&o| o == id).unwrap()];
        let back = from_text(&text, &ids);
        assert_eq!(back.grounded, vec![100]);
        assert_eq!((back.joints.len(), back.connections.len()), (6, 3));
        for (x, y) in s.assembly.joints.iter().zip(&back.joints) {
            assert_eq!((x.id, map(x.part), x.shape, x.name.clone()), (y.id, y.part, y.shape, y.name.clone()));
            assert!(near(x.frame().origin, y.frame().origin) && (x.radius - y.radius).abs() < 1e-12);
        }
        for (x, y) in s.assembly.connections.iter().zip(&back.connections) {
            assert_eq!((x.a, x.b, map(x.parent), map(x.child), x.interaction, x.kind), (y.a, y.b, y.parent, y.child, y.interaction, y.kind));
            assert!(x.pose().0.angle_between(y.pose().0) < 1e-9 && near(x.pose().1, y.pose().1));
        }
        assert_eq!(back.joints[0].name, "Shoulder L");
        // New ids continue after the loaded ones.
        let mut back = back;
        let nj = back.add_joint(Joint::point(100, JFrame::new(DVec3::ZERO, DVec3::Z, None)));
        assert!(nj > back.connections.iter().map(|c| c.id).max().unwrap());
        // Version 1: connections as two frames: points joined, the same pose.
        let f = JFrame::new(DVec3::new(1.0, 0.0, 0.0), DVec3::X, None);
        let fc = f.then_local(&Rigid { rot: DQuat::from_rotation_z(0.4), shift: DVec3::ZERO });
        let v1 = format!("polymix-assembly 1\nground 0\njoint Ball 0 1 0.25 {} {}\nabout ball\n", frame_text(&f), frame_text(&fc));
        let old = from_text(&v1, &ids);
        assert_eq!((old.joints.len(), old.connections.len()), (2, 1));
        let c = &old.connections[0];
        assert_eq!((c.parent, c.child, c.kind, c.about.as_str()), (100, 101, Motion::Ball, "ball"));
        assert!((c.pose().0.angle_between(DQuat::IDENTITY) - 0.4).abs() < 1e-9);
        assert!(to_text(&Assembly::default(), &order).is_none());
        assert!(from_text("something else", &ids).is_empty());
    }

    #[test]
    fn the_selection_survives_undo_but_not_what_it_selects() {
        let (mut s, [torso, arm, ..], [shoulder, elbow, _]) = kit();
        let mut h = crate::scene::History::default();
        s.assembly.selected = Some(Selected::Connection(elbow));
        h.push("pose", &s);
        let side = s.assembly.side(shoulder, arm);
        move_parts(&mut s, &side, &Rigid::about(DQuat::from_rotation_z(0.3), DVec3::new(1.0, 0.0, 0.0)));
        s.assembly.selected = Some(Selected::Connection(shoulder));
        h.undo(&mut s);
        assert_eq!(s.assembly.selected, Some(Selected::Connection(shoulder)));
        h.push("delete", &s);
        s.assembly.remove_connection(shoulder);
        assert_eq!(s.assembly.selected, None);
        let socket = s.assembly.joints[0].id;
        s.assembly.selected = Some(Selected::Joint(socket));
        h.undo(&mut s);
        assert_eq!(s.assembly.selected, Some(Selected::Joint(socket)));
        s.remove(torso);
        assert_eq!(s.assembly.selected, None, "removing a part drops its joints and their selection");
        assert!(s.assembly.connections.iter().all(|c| !c.joins(torso)));
    }

    #[test]
    fn small_rules_tangent_for_balls_only_sizes_and_grounding_over_undo() {
        assert_eq!(Interaction::between(JointShape::Socket, JointShape::Flat)[0], Interaction::Concentric);
        assert!(!Interaction::between(JointShape::Flat, JointShape::Socket).contains(&Interaction::Tangent));
        assert_eq!(Interaction::between(JointShape::Flat, JointShape::Ball)[0], Interaction::Tangent);
        // Two points: a visible size, kept when the anchors are worked out again.
        let mut a = Assembly::default();
        let p = a.add_joint(Joint::point(1, JFrame::new(DVec3::ZERO, DVec3::Z, None)));
        let q = a.add_joint(Joint::point(2, JFrame::new(DVec3::X, DVec3::Z, None)));
        let c = a.connect(p, q, Interaction::Coincident, Motion::Hinge, Rigid::IDENTITY, String::new()).unwrap();
        assert_eq!(a.connection(c).unwrap().size, 1.0);
        a.connection_mut(c).unwrap().size = 0.25;
        a.refresh_anchors(c);
        assert_eq!(a.connection(c).unwrap().size, 0.25);
        // A part brought back by undo is grounded as it was.
        let (mut s, [torso, ..], _) = kit();
        let mut h = crate::scene::History::default();
        h.push("delete", &s);
        s.remove(torso);
        h.undo(&mut s);
        assert!(s.assembly.is_grounded(torso));
        // Edited joints no longer fit their old triangles.
        let mut m = primitives::icosphere(1.0, 2);
        m.transform(|x| x + DVec3::new(3.0, 0.0, 0.0));
        let f = mm_edit::mate::recognize(&m, 0, &Default::default()).unwrap();
        let mut j = Joint::from_feature(7, &f, &m);
        assert!(j.fitted_surface_on(&m).is_some());
        j.radius *= 1.1;
        assert!(j.fitted_surface_on(&m).is_none() && j.surface_on(&m).is_some());
    }

    #[test]
    fn kabsch_finds_the_motion() {
        let r = Rigid::about(DQuat::from_euler(glam::EulerRot::XYZ, 0.3, -0.8, 1.9), DVec3::new(1.0, 2.0, 3.0));
        let pts = [DVec3::ZERO, DVec3::X, DVec3::new(0.0, 2.0, 0.5), DVec3::new(-1.0, 0.3, 4.0)];
        let moved: Vec<DVec3> = pts.iter().map(|&p| r.apply(p)).collect();
        let f = rigid_fit(&pts, &moved);
        for p in pts {
            assert!(near(f.apply(p), r.apply(p)));
        }
    }
}
