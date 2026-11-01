// SPDX-License-Identifier: AGPL-3.0-only
//! Script steps for the Mirror, Combine, Cut by plane and Offset body boxes, so they can be
//! driven headlessly (`POLYMIX_SCRIPT`): `bool:<what>[:<args>]`.
//!
//! * `bool:open:<mirror|boolean|cutPlane|offsetBody>` — open a new box of the kind, as its
//!   button does (a mirror starts the symmetry search);
//! * `bool:plane:<x|y|z>[:<offset>]` — the open cut's or mirror's plane, a coordinate plane;
//! * `bool:flip` — the open cut keeps the other side;
//! * `bool:create` — make the open feature, as Create does;
//! * `bool:mirror:<x|y|z>[:<offset>[:apart]]` — a mirror made in one step;
//! * `bool:combine:<join|cut|intersect>:<body>:<tool>[:keep]` — bodies by id (a feature's id,
//!   with a fraction for a further body of its step);
//! * `bool:cut:<x|y|z>:<offset>[:flip]`, `bool:offset:<distance>[:replace]`.

use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{BodyId, BooleanOp, FeatureKind, MirrorPlane, PlaneRef};

use crate::reverse::{Mode, ReverseWorkspace};

fn axis(s: Option<&&str>) -> WorldAxis {
    match s.copied() {
        Some("x") | Some("yz") => WorldAxis::X,
        Some("y") | Some("xz") => WorldAxis::Y,
        _ => WorldAxis::Z,
    }
}

fn num(args: &[&str], i: usize, default: f64) -> f64 {
    args.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(default)
}

fn body(args: &[&str], i: usize) -> Option<BodyId> {
    args.get(i).and_then(|v| v.parse::<f64>().ok()).and_then(BodyId::from_number)
}

/// Open a new box of a kind, as its button in the panel does.
fn open(ws: &mut ReverseWorkspace, kind: &str) -> bool {
    match super::new_feature(kind, ws) {
        Ok(k) => {
            ws.open_feature(k, None);
            true
        }
        Err(e) => {
            ws.status = e;
            log::warn!("bool script: {kind} cannot open: {}", ws.status);
            false
        }
    }
}

/// The open box's feature, to change.
fn open_kind(ws: &mut ReverseWorkspace) -> Option<&mut FeatureKind> {
    match &mut ws.mode {
        Mode::Feature(edit) => Some(&mut edit.kind),
        _ => None,
    }
}

fn set_plane(ws: &mut ReverseWorkspace, plane: PlaneRef) {
    match open_kind(ws) {
        Some(FeatureKind::CutPlane(x)) => x.plane = plane,
        Some(FeatureKind::Mirror(x)) => x.plane = MirrorPlane::Plane(plane),
        _ => log::warn!("bool script: no cut or mirror is open to take a plane"),
    }
    ws.request_build();
}

/// One `bool:` step.
pub(crate) fn bool_script_step(ws: &mut ReverseWorkspace, args: &[&str]) {
    let what = args.first().copied().unwrap_or("");
    let rest = &args[1.min(args.len())..];
    match what {
        "open" => {
            open(ws, rest.first().copied().unwrap_or(""));
        }
        "plane" => set_plane(ws, PlaneRef::world(axis(rest.first()), num(rest, 1, 0.0))),
        "flip" => {
            if let Some(FeatureKind::CutPlane(x)) = open_kind(ws) {
                x.keep = if x.keep > 0 { -1 } else { 1 };
            }
            ws.request_build();
        }
        "create" => ws.commit_feature(),
        "mirror" => {
            if open(ws, "mirror") {
                set_plane(ws, PlaneRef::world(axis(rest.first()), num(rest, 1, 0.0)));
                if let Some(FeatureKind::Mirror(m)) = open_kind(ws) {
                    m.join = rest.get(2) != Some(&"apart");
                }
                ws.commit_feature();
            }
        }
        "combine" => {
            let op = match rest.first().copied() {
                Some("cut") => BooleanOp::Cut,
                Some("intersect") => BooleanOp::Intersect,
                _ => BooleanOp::Join,
            };
            if open(ws, "boolean") {
                let (b, t) = (body(rest, 1), body(rest, 2));
                if let Some(FeatureKind::Boolean(c)) = open_kind(ws) {
                    c.op = op;
                    if let Some(b) = b {
                        c.body = b;
                    }
                    if let Some(t) = t {
                        c.tool = t;
                    }
                    c.keep = rest.get(3) == Some(&"keep");
                }
                ws.commit_feature();
            }
        }
        "cut" => {
            if open(ws, "cutPlane") {
                set_plane(ws, PlaneRef::world(axis(rest.first()), num(rest, 1, 0.0)));
                if let Some(FeatureKind::CutPlane(x)) = open_kind(ws) {
                    x.keep = if rest.get(2) == Some(&"flip") { -1 } else { 1 };
                }
                ws.commit_feature();
            }
        }
        "offset" => {
            if open(ws, "offsetBody") {
                if let Some(FeatureKind::OffsetBody(x)) = open_kind(ws) {
                    x.distance = num(rest, 0, 1.0);
                    x.keep = rest.get(1) != Some(&"replace");
                }
                ws.commit_feature();
            }
        }
        other => log::warn!("unknown bool script step {other}"),
    }
}
