// SPDX-License-Identifier: AGPL-3.0-only
//! The part's three coordinate planes as a sketch sees them. Ported from ScanRuler
//! (`plugins/reverse/core/datums.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Each coordinate plane cuts the sketch plane in a line (or runs parallel to it); that line
//! is the plane's *trace*, and a dimension to the plane is the distance of a point from the
//! trace, measured in the sheet. The traces are never stored: they follow from the frame.

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::frame::SectionFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatumPlane {
    Xy,
    Yz,
    Zx,
}

impl DatumPlane {
    pub const ALL: [DatumPlane; 3] = [DatumPlane::Xy, DatumPlane::Yz, DatumPlane::Zx];

    pub fn name(self) -> &'static str {
        match self {
            DatumPlane::Xy => "XY plane",
            DatumPlane::Yz => "YZ plane",
            DatumPlane::Zx => "XZ plane",
        }
    }

    pub fn normal(self) -> DVec3 {
        match self {
            DatumPlane::Xy => DVec3::Z,
            DatumPlane::Yz => DVec3::X,
            DatumPlane::Zx => DVec3::Y,
        }
    }
}

/// A coordinate plane's line in the sheet: the points with `n · p + d = 0`, `n` a unit
/// vector, so `n · p + d` is a point's signed distance from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatumTrace {
    pub plane: DatumPlane,
    pub n: DVec2,
    pub d: f64,
}

impl DatumTrace {
    /// A point's signed distance from the trace.
    pub fn offset(&self, p: DVec2) -> f64 {
        self.n.dot(p) + self.d
    }

    /// The foot of a point on the trace.
    pub fn foot(&self, p: DVec2) -> DVec2 {
        p - self.n * self.offset(p)
    }

    /// The trace as a point on it and its unit direction.
    pub fn line(&self) -> (DVec2, DVec2) {
        (-self.n * self.d, DVec2::new(-self.n.y, self.n.x))
    }

    /// A stretch of the trace to draw: `half` either way of the foot of `near`.
    pub fn segment(&self, near: DVec2, half: f64) -> (DVec2, DVec2) {
        let foot = self.foot(near);
        let dir = DVec2::new(-self.n.y, self.n.x);
        (foot - dir * half, foot + dir * half)
    }
}

/// The traces of the coordinate planes in a sketch's sheet. A plane the sketch lies parallel
/// to has none.
pub fn datum_traces(frame: &SectionFrame) -> Vec<DatumTrace> {
    let mut out = Vec::new();
    for plane in DatumPlane::ALL {
        let normal = plane.normal();
        let a = frame.u.dot(normal);
        let b = frame.v.dot(normal);
        let l = a.hypot(b);
        if l < 1e-6 {
            continue;
        }
        out.push(DatumTrace { plane, n: DVec2::new(a / l, b / l), d: frame.origin.dot(normal) / l });
    }
    out
}

pub fn trace_of(traces: &[DatumTrace], plane: DatumPlane) -> Option<DatumTrace> {
    traces.iter().copied().find(|t| t.plane == plane)
}

/// The trace within reach of a spot, the nearest where several are.
pub fn trace_near(traces: &[DatumTrace], spot: DVec2, radius: f64) -> Option<DatumTrace> {
    let mut best = None;
    let mut best_d = radius;
    for t in traces {
        let d = t.offset(spot).abs();
        if d <= best_d {
            best_d = d;
            best = Some(*t);
        }
    }
    best
}
