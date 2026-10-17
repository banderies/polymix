//! Orbit camera with Meshmixer-style tumble / pan / zoom.

use glam::dcamera::rh::{proj::directx, view as camview};
use glam::{DMat4, DQuat, DVec2, DVec3, Mat4};
use mm_core::{Aabb, Ray};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpAxis {
    Y,
    Z,
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub target: DVec3,
    pub distance: f64,
    /// Rotation around the world up axis (radians).
    pub yaw: f64,
    /// Elevation above the horizon (radians), unbounded in effect: past ±90° the camera goes
    /// over the top / underneath and ends up upside down. Kept in (-π, π].
    pub pitch: f64,
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
            yaw: 0.6,
            pitch: 0.35,
            fov_y: 35f64.to_radians(),
            orthographic: false,
            up_axis: UpAxis::Y,
            scene_radius: 1.0,
        }
    }
}

impl Camera {
    pub fn world_up(&self) -> DVec3 {
        match self.up_axis {
            UpAxis::Y => DVec3::Y,
            UpAxis::Z => DVec3::Z,
        }
    }

    /// Camera orientation (camera-to-world rotation; camera looks down its local -Z).
    pub fn rotation(&self) -> DQuat {
        let base = match self.up_axis {
            UpAxis::Y => DQuat::IDENTITY,
            // Map camera-space Y-up onto world Z-up, looking along +Y by default.
            UpAxis::Z => DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
        };
        base * DQuat::from_rotation_y(self.yaw) * DQuat::from_rotation_x(-self.pitch)
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

    /// Free orbit: vertical drags rotate over the top and underneath without limit.
    pub fn tumble(&mut self, dx: f64, dy: f64) {
        // Upside down, horizontal drags turn the other way so the model still follows the cursor.
        let upright = if self.pitch.cos() >= 0.0 { 1.0 } else { -1.0 };
        self.yaw = wrap_angle(self.yaw - dx * 0.008 * upright);
        self.pitch = wrap_angle(self.pitch + dy * 0.008);
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

    /// Snap to the nearest axis-aligned view.
    pub fn snap_to_axis(&mut self) {
        self.yaw = wrap_angle((self.yaw / FRAC_PI_2).round() * FRAC_PI_2);
        self.pitch = wrap_angle((self.pitch / FRAC_PI_2).round() * FRAC_PI_2);
    }

    pub fn set_view(&mut self, yaw: f64, pitch: f64) {
        self.yaw = wrap_angle(yaw);
        self.pitch = wrap_angle(pitch);
    }
}

/// Wrap an angle into (-π, π].
fn wrap_angle(a: f64) -> f64 {
    let w = a.rem_euclid(TAU);
    if w > PI {
        w - TAU
    } else {
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tumble_goes_over_the_top() {
        let mut c = Camera { pitch: 0.0, yaw: 0.0, ..Default::default() };
        // Drag far enough to rotate 200 degrees vertically.
        let steps = (200f64.to_radians() / 0.008 / 10.0).ceil() as usize;
        let mut max_pitch: f64 = 0.0;
        for _ in 0..steps {
            c.tumble(0.0, 10.0);
            max_pitch = max_pitch.max(c.pitch);
            // The view basis stays orthonormal the whole way round.
            assert!(c.forward().dot(c.up()).abs() < 1e-9);
            assert!(c.view().is_finite());
        }
        assert!(max_pitch > FRAC_PI_2, "pitch must pass straight up");
        // Now upside down: the camera's up points against the world up.
        assert!(c.up().dot(c.world_up()) < 0.0);
    }

    #[test]
    fn straight_down_views_are_valid() {
        let mut c = Camera::default();
        for pitch in [FRAC_PI_2, -FRAC_PI_2, PI] {
            c.set_view(0.3, pitch);
            assert!(c.view().is_finite());
            assert!(c.forward().dot(c.up()).abs() < 1e-9);
        }
        c.set_view(0.0, 3.0 * PI);
        assert!((c.pitch - PI).abs() < 1e-12);
    }
}
