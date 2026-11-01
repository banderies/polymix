// SPDX-License-Identifier: AGPL-3.0-only
//! A patch network as a feature stores it: its nodes, its curves through their grips, the
//! cells filled by a point inside each, the strips' span counts and the fit's settings.
//!
//! Ported from ScanRuler (`plugins/reverse/core/features.ts`: `NetworkNode`, `NetworkCurve`,
//! `NetworkCell`, `NetworkFitParams`, `NetworkFeature`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. The pick fingerprints they hold are [`crate::topo`]'s.
//!
//! The feature's id, name, visibility and suppression are the Polymix feature's own
//! (`model::Feature`), so [`NetworkFeature`] is what its kind carries, as `Extrude` is. The
//! fingerprints are data only here: finding them again on a body as built is the kernel's
//! side, behind [`crate::network::NetworkBodies`].

use std::collections::BTreeMap;

use glam::DVec3;
use serde::{Deserialize, Serialize};

pub use crate::topo::{EdgeFingerprint, FaceFingerprint};

/// A face of a body as a network's curve refers to it: its fingerprint ([`crate::topo`]).
pub type FaceRef = FaceFingerprint;
/// An edge of a body as a network's node or curve refers to it: its fingerprint.
pub type EdgeRef = EdgeFingerprint;

/// Where a node rests on a body: partway along an edge (`t` its share of the way from the
/// edge's first end), or at one of its ends (a corner).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NodeOn {
    Edge { body: u32, edge: EdgeRef, t: f64 },
    Vertex { body: u32, vertex: EdgeRef, end: u8 },
}

impl NodeOn {
    pub fn body(&self) -> u32 {
        match self {
            NodeOn::Edge { body, .. } | NodeOn::Vertex { body, .. } => *body,
        }
    }

    /// The edge it is on, or the edge whose end it is.
    pub fn edge(&self) -> &EdgeRef {
        match self {
            NodeOn::Edge { edge, .. } => edge,
            NodeOn::Vertex { vertex, .. } => vertex,
        }
    }
}

/// A point of a patch network: a curve's end, or a corner; on the scan, in the scan's frame,
/// and taken onto it again at every build; or on a body's edge or corner (`on`), where it is
/// found again on the body as built.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: u32,
    pub at: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on: Option<NodeOn>,
    /// Whether the node is a corner of the cells it bounds, over the rule (the boundary
    /// turning more than 30° there). Absent: the rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner: Option<bool>,
}

/// A curve runs along a body's edge: its points are the edge's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveAlong {
    pub body: u32,
    pub edge: EdgeRef,
}

/// How the patch beside a body's face meets it: on its edge, or tangent to the face too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SideMode {
    Contact,
    Tangent,
}

/// The patch beside a body's face meets it along the curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveSide {
    pub body: u32,
    pub face: FaceRef,
    pub mode: SideMode,
}

/// A curve of a patch network, from one node to another through its grips (`via`), along
/// the scan between them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkCurve {
    pub id: u32,
    pub from: u32,
    pub to: u32,
    #[serde(default)]
    pub via: Vec<DVec3>,
    /// A fold or a parting line: the patches either side meet at an angle. Absent: they meet
    /// tangent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crease: Option<bool>,
    /// The curve runs along a body's edge: its points are the edge's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub along: Option<CurveAlong>,
    /// The patch beside a body's face meets it there: on its edge (contact) or tangent to the
    /// face too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub side: Option<CurveSide>,
}

/// A cell of a network filled with a patch: which region was filled, by a point inside it
/// (it stays filled when its curves move) and, for a cell of three, five or six sides split
/// into four-sided ones, where its centre was dragged to. A patch placed by its corners keeps
/// them (`corners`, node ids in order round it): they are its corners however little the
/// boundary turns there, and the box puts its seed back in its middle when they move.
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
    /// The fairing term's weight, as Fit Surface's smoothness.
    pub smoothness: f64,
    /// Fit again without the points past three sigmas of the first pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust: Option<bool>,
    /// How far Auto lets a patch stray from the scan beyond its noise before its strips get
    /// more spans, mm (`network_curve::DEFAULT_TOLERANCE` if none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
}

impl Default for NetworkFitParams {
    fn default() -> Self {
        NetworkFitParams { smoothness: 0.001, robust: None, tolerance: None }
    }
}

/// A Patch network: curves drawn on the scan, and the regions they close filled with
/// B-spline patches fitted to the scan together, sharing their edges and meeting tangent
/// across every curve not marked a crease ([`crate::network`]). One body: a sheet, or a
/// solid where the network closes.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NetworkFeature {
    pub nodes: Vec<NetworkNode>,
    pub curves: Vec<NetworkCurve>,
    pub cells: Vec<NetworkCell>,
    /// The span count of a strip of patches (the network fit), keyed by the id of a curve in
    /// it; a strip with none is Auto.
    #[serde(default)]
    pub spans: BTreeMap<String, f64>,
    pub fit: NetworkFitParams,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_scanruler_json() {
        let json = r#"{
            "nodes": [
                { "id": 1, "at": [0, 0, 0] },
                { "id": 2, "at": [10, 0, 0], "corner": true,
                  "on": { "body": 3, "vertex": { "kind": "line", "mid": [5, 0, 0], "dir": [1, 0, 0], "length": 10,
                          "faces": [{ "kind": "plane", "dir": [0, 0, 1] }], "origin": 4 }, "end": 1 } }
            ],
            "curves": [{ "id": 1, "from": 1, "to": 2, "via": [[5, 1, 0]], "crease": true }],
            "cells": [{ "seed": [5, 5, 0], "corners": [1, 2] }],
            "spans": { "1": 6 },
            "fit": { "smoothness": 0.001 }
        }"#;
        let f: NetworkFeature = serde_json::from_str(json).expect("parses");
        assert_eq!(f.nodes.len(), 2);
        assert!(matches!(f.nodes[1].on, Some(NodeOn::Vertex { end: 1, body: 3, .. })));
        assert_eq!(f.curves[0].via, vec![DVec3::new(5.0, 1.0, 0.0)]);
        assert_eq!(f.spans.get("1"), Some(&6.0));
        let back: NetworkFeature = serde_json::from_str(&serde_json::to_string(&f).expect("writes")).expect("parses again");
        assert_eq!(back, f);
    }
}
