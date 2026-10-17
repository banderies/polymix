//! Orbit camera with Meshmixer-style tumble / pan / zoom.
//!
//! Tumbling is a free trackball by default: drags rotate about the *view's* own axes, so the
//! model can be turned to any orientation (including rolling it). A turntable mode that keeps
//! the up axis vertical is available as an option.

use glam::dcamera::rh::{proj::directx, view as camview};
use glam::{DMat4, DQuat, DVec2, DVec3, Mat4};
use mm_core::{Aabb, Ray};
use std::f64::consts::FRAC_PI_2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpAxis {
    Y,
    Z,
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub target: DVec3,
    pub distance: f64,
    /// Camera orientation relative to the up-axis frame (see [`Camera::rotation`]).
    pub orient: DQuat,
    /// Free trackball tumbling (true) or turntable around the up axis (false).
    pub free_orbit: bool,
    pub fov_y: f64,
    pub orthographic: bool,
    pub up_axis: UpAxis,
    /// Radius of the scene, used for clip planes.
    pub scene_radius: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: DVec3::ZERO,
            distance: 5.0,
            orient: view_orientation(0.6, 0.35),
            free_orbit: true,
            fov_y: 35f64.to_radians(),
            orthographic: false,
            up_axis: UpAxis::Y,
            scene_radius: 1.0,
        }
    }
}

/// A saved view (for Previous / Next view).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewState {
    pub target: DVec3,
    pub distance: f64,
    pub orient: DQuat,
}

impl ViewState {
    /// Whether two views differ enough to keep both in the history.
    pub fn differs(&self, o: &ViewState, scene_radius: f64) -> bool {
        self.target.distance(o.target) > scene_radius * 1e-3
            || (self.distance / o.distance.max(1e-12) - 1.0).abs() > 1e-3
            || self.orient.angle_between(o.orient) > 1e-3
    }
}

/// Back / forward history of views.
#[derive(Default)]
pub struct ViewHistory {
    states: Vec<ViewState>,
    /// Index of the current view when browsing the history; `None` = live (past the end).
    pos: Option<usize>,
}

impl ViewHistory {
    const MAX: usize = 50;

    /// Remember `current` before the view changes (navigation start, focus, standard views…).
    pub fn record(&mut self, current: ViewState, scene_radius: f64) {
        if let Some(p) = self.pos.take() {
            self.states.truncate(p + 1);
        }
        if self.states.last().is_none_or(|l| l.differs(&current, scene_radius)) {
            self.states.push(current);
            if self.states.len() > Self::MAX {
                self.states.remove(0);
            }
        }
    }

    pub fn back(&mut self, current: ViewState, scene_radius: f64) -> Option<ViewState> {
        let p = match self.pos {
            Some(p) => p.checked_sub(1)?,
            None => {
                // Keep the live view so Next can return to it.
                self.record(current, scene_radius);
                self.states.len().checked_sub(2)?
            }
        };
        self.pos = Some(p);
        Some(self.states[p])
    }

    pub fn forward(&mut self) -> Option<ViewState> {
        let p = self.pos? + 1;
        if p >= self.states.len() {
            return None;
        }
        self.pos = Some(p);
        Some(self.states[p])
    }

    pub fn can_back(&self) -> bool {
        self.pos.map_or(!self.states.is_empty(), |p| p > 0)
    }

    pub fn can_forward(&self) -> bool {
        self.pos.is_some_and(|p| p + 1 < self.states.len())
    }
}

impl Camera {
    pub fn view_state(&self) -> ViewState {
        ViewState { target: self.target, distance: self.distance, orient: self.orient }
    }

    pub fn set_view_state(&mut self, v: ViewState) {
        self.target = v.target;
        self.distance = v.distance;
        self.orient = v.orient;
    }

    pub fn world_up(&self) -> DVec3 {
        match self.up_axis {
            UpAxis::Y => DVec3::Y,
            UpAxis::Z => DVec3::Z,
        }
    }

    /// Frame whose Y axis is the world up axis.
    fn base(&self) -> DQuat {
        match self.up_axis {
            UpAxis::Y => DQuat::IDENTITY,
            // Map camera-space Y-up onto world Z-up, looking along +Y by default.
            UpAxis::Z => DQuat::from_rotation_x(FRAC_PI_2),
        }
    }

    /// Camera orientation (camera-to-world rotation; camera looks down its local -Z).
    pub fn rotation(&self) -> DQuat {
        self.base() * self.orient
    }

    pub fn forward(&self) -> DVec3 {
        self.rotation() * -DVec3::Z
    }
    pub fn right(&self) -> DVec3 {
        self.rotation() * DVec3::X
    }
    pub fn up(&self) -> DVec3 {
        self.rotation() * DVec3::Y
    }
    pub fn eye(&self) -> DVec3 {
        self.target - self.forward() * self.distance
    }

    /// The view uses the camera's own up vector (not the world's), so it never degenerates
    /// when looking straight up or down.
    pub fn view(&self) -> DMat4 {
        camview::look_at_mat4(self.eye(), self.target, self.up())
    }

    /// Near plane for the reversed-Z infinite projection: depth precision is excellent
    /// everywhere, so the near plane can sit close to the eye.
    fn near_plane(&self) -> f64 {
        (self.distance * 1e-3).max(self.scene_radius * 1e-6).max(1e-9)
    }

    /// Reversed-Z projection (depth 1 at the near plane, 0 at infinity / far).
    pub fn proj(&self, aspect: f64) -> DMat4 {
        if self.orthographic {
            let h = self.distance * (self.fov_y * 0.5).tan();
            let w = h * aspect;
            let far = self.distance + 10.0 * self.scene_radius.max(1e-6);
            // Swapping near and far reverses the depth range.
            directx::orthographic(-w, w, -h, h, far, -far)
        } else {
            directx::perspective_infinite_reverse(self.fov_y, aspect, self.near_plane())
        }
    }

    pub fn view_proj(&self, aspect: f64) -> DMat4 {
        self.proj(aspect) * self.view()
    }

    pub fn view_proj_f32(&self, aspect: f64) -> Mat4 {
        self.view_proj(aspect).as_mat4()
    }

    /// Ray through a viewport position given in normalized device coordinates (-1..1, y up).
    pub fn ray_from_ndc(&self, ndc: DVec2, aspect: f64) -> Ray {
        // Reversed Z: depth 1 is the near plane; 0 is infinity, so unproject two finite depths.
        let inv = self.view_proj(aspect).inverse();
        let near = inv.project_point3(DVec3::new(ndc.x, ndc.y, 1.0));
        let mid = inv.project_point3(DVec3::new(ndc.x, ndc.y, 0.5));
        Ray::new(near, mid - near)
    }

    /// Project a world point to NDC (x, y in -1..1, z depth) — `None` if behind the camera.
    pub fn project(&self, p: DVec3, aspect: f64) -> Option<DVec3> {
        let clip = self.view_proj(aspect) * p.extend(1.0);
        if clip.w <= 1e-12 {
            return None;
        }
        Some(clip.truncate() / clip.w)
    }

    /// World-space size of one pixel at the target distance for a viewport `height_px` tall.
    pub fn pixel_size_at_target(&self, height_px: f64) -> f64 {
        2.0 * self.distance * (self.fov_y * 0.5).tan() / height_px.max(1.0)
    }

    /// Rotate the view for a drag of (`dx`, `dy`) screen points.
    pub fn tumble(&mut self, dx: f64, dy: f64) {
        const SPEED: f64 = 0.008;
        let turn = DQuat::from_rotation_y(-dx * SPEED);
        let tilt = DQuat::from_rotation_x(-dy * SPEED);
        self.orient = if self.free_orbit {
            // Trackball: both rotations about the current view axes, so nothing is locked.
            self.orient * turn * tilt
        } else {
            // Turntable: turn about the up axis (reversed when upside down so the model still
            // follows the cursor), tilt about the view's horizontal axis.
            let upright = (self.orient * DVec3::Y).y >= 0.0;
            let turn = if upright { turn } else { turn.inverse() };
            turn * self.orient * tilt
        }
        .normalize();
    }

    pub fn pan(&mut self, dx: f64, dy: f64, viewport_h: f64) {
        let s = self.pixel_size_at_target(viewport_h);
        self.target += (-self.right() * dx + self.up() * dy) * s;
    }

    pub fn zoom(&mut self, factor: f64) {
        self.distance = (self.distance * factor).clamp(self.scene_radius * 1e-4, self.scene_radius * 1e3);
    }

    /// Zoom toward a world point under the cursor so that point stays put.
    pub fn zoom_towards(&mut self, factor: f64, point: DVec3) {
        let old = self.distance;
        self.zoom(factor);
        let k = 1.0 - self.distance / old;
        // Move the target along the view-plane offset of the point.
        let off = point - self.target;
        let off_plane = off - self.forward() * off.dot(self.forward());
        self.target += off_plane * k;
    }

    /// Fit the view to bounds.
    pub fn focus(&mut self, b: &Aabb) {
        if b.is_empty() {
            return;
        }
        let r = (b.diagonal() * 0.5).max(1e-6);
        self.target = b.center();
        self.scene_radius = self.scene_radius.max(r);
        self.distance = r / (self.fov_y * 0.5).sin() * 1.1;
    }

    /// Turn the model by (`rx`, `ry`, `rz`) radians about the view's axes (x right, y up,
    /// z toward the viewer), as a 3D mouse does in object mode. The turntable mode keeps the
    /// horizon level: spin is about the world up axis and roll is ignored.
    pub fn rotate_view(&mut self, rx: f64, ry: f64, rz: f64) {
        // The camera turns opposite to the model.
        self.orient = if self.free_orbit {
            self.orient * DQuat::from_scaled_axis(DVec3::new(-rx, -ry, -rz))
        } else {
            let upright = (self.orient * DVec3::Y).y >= 0.0;
            let spin = DQuat::from_rotation_y(if upright { -ry } else { ry });
            spin * self.orient * DQuat::from_rotation_x(-rx)
        }
        .normalize();
    }

    /// Snap to the nearest axis-aligned view (view direction and screen-up both along axes).
    pub fn snap_to_axis(&mut self) {
        let back = nearest_axis(self.orient * DVec3::Z, None);
        let up = nearest_axis(self.orient * DVec3::Y, Some(back));
        let right = up.cross(back);
        self.orient = DQuat::from_mat3(&glam::DMat3::from_cols(right, up, back)).normalize();
    }

    /// Standard view from a heading `yaw` around the up axis and an elevation `pitch`.
    pub fn set_view(&mut self, yaw: f64, pitch: f64) {
        self.orient = view_orientation(yaw, pitch);
    }
}

fn view_orientation(yaw: f64, pitch: f64) -> DQuat {
    DQuat::from_rotation_y(yaw) * DQuat::from_rotation_x(-pitch)
}

/// The signed coordinate axis closest to `v`, optionally excluding the line of `not`.
fn nearest_axis(v: DVec3, not: Option<DVec3>) -> DVec3 {
    let mut best = DVec3::Y;
    let mut best_dot = f64::MIN;
    for a in [DVec3::X, DVec3::Y, DVec3::Z, -DVec3::X, -DVec3::Y, -DVec3::Z] {
        if not.is_some_and(|n| n.dot(a).abs() > 0.5) {
            continue;
        }
        if v.dot(a) > best_dot {
            best_dot = v.dot(a);
            best = a;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orthonormal(c: &Camera) -> bool {
        c.forward().dot(c.up()).abs() < 1e-9 && c.forward().dot(c.right()).abs() < 1e-9 && (c.up().length() - 1.0).abs() < 1e-9
    }

    #[test]
    fn tumble_goes_over_the_top() {
        let mut c = Camera::default();
        c.set_view(0.0, 0.0);
        // Drag far enough to rotate 200 degrees vertically.
        let steps = (200f64.to_radians() / 0.008 / 10.0).ceil() as usize;
        for _ in 0..steps {
            c.tumble(0.0, 10.0);
            assert!(orthonormal(&c));
            assert!(c.view().is_finite());
        }
        // Now upside down: the camera's up points against the world up.
        assert!(c.up().dot(c.world_up()) < 0.0);
    }

    #[test]
    fn free_orbit_can_roll_the_view() {
        // A turntable can never tilt the horizon; the trackball must, so the model can be
        // turned to any orientation.
        let mut c = Camera::default();
        c.set_view(0.0, 0.0);
        // Up, right, down, left: a closed loop of drags leaves a net roll about the view axis.
        for (dx, dy) in [(0.0, 100.0), (100.0, 0.0), (0.0, -100.0), (-100.0, 0.0)] {
            for _ in 0..10 {
                c.tumble(dx / 10.0, dy / 10.0);
            }
        }
        assert!(orthonormal(&c));
        assert!(c.right().dot(c.world_up()).abs() > 0.05, "horizon must tilt in free orbit");
        // Turntable keeps the horizon level for the same drags.
        let mut t = Camera { free_orbit: false, ..Default::default() };
        t.set_view(0.0, 0.0);
        for (dx, dy) in [(0.0, 100.0), (100.0, 0.0), (0.0, -100.0), (-100.0, 0.0)] {
            for _ in 0..10 {
                t.tumble(dx / 10.0, dy / 10.0);
            }
        }
        assert!(t.right().dot(t.world_up()).abs() < 1e-9);
    }

    #[test]
    fn straight_down_views_are_valid() {
        let mut c = Camera::default();
        for pitch in [FRAC_PI_2, -FRAC_PI_2, std::f64::consts::PI] {
            c.set_view(0.3, pitch);
            assert!(c.view().is_finite());
            assert!(orthonormal(&c));
        }
    }

    #[test]
    fn view_history_back_and_forward() {
        let mut c = Camera::default();
        let mut h = ViewHistory::default();
        let v0 = c.view_state();
        h.record(c.view_state(), 1.0);
        c.tumble(50.0, 0.0);
        let v1 = c.view_state();
        h.record(c.view_state(), 1.0);
        c.tumble(0.0, 50.0);
        let v2 = c.view_state();
        assert_eq!(h.back(c.view_state(), 1.0), Some(v1));
        assert_eq!(h.back(v1, 1.0), Some(v0));
        assert_eq!(h.back(v0, 1.0), None);
        assert_eq!(h.forward(), Some(v1));
        assert_eq!(h.forward(), Some(v2));
        assert_eq!(h.forward(), None);
        // A new change drops the forward history.
        h.back(v2, 1.0);
        h.record(v1, 1.0);
        assert!(!h.can_forward());
    }

    #[test]
    fn snap_aligns_to_axes() {
        let mut c = Camera::default();
        c.tumble(37.0, -12.0);
        c.tumble(5.0, 80.0);
        c.snap_to_axis();
        assert!(orthonormal(&c));
        for v in [c.forward(), c.up(), c.right()] {
            assert!((v.abs().max_element() - 1.0).abs() < 1e-9, "{v:?} is not an axis");
        }
    }
}
