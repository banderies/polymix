// SPDX-License-Identifier: AGPL-3.0-only
//! The Patch network feature's box: curves drawn on the scan, the regions they close filled
//! with patches fitted to the scan together. Its fields, its layer in the view and its keys are
//! `reverse/network_layer.rs`; this is its place among the feature boxes. Ported from ScanRuler
//! (`plugins/reverse/ui/NetworkEditor.tsx`, `plugins/reverse/ui/ReversePanel.tsx`), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::model::{FeatureKind, NetworkFeature};
use mm_reverse::network_draft::empty_network;

use crate::reverse::network_layer::{self, NET_FRESH};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Patch network", "Curves drawn on the scan, and the regions they close filled with patches fitted to the scan together — tangent across every curve not marked a crease, sharing their edges: a sheet, or a solid where the network closes. For a freeform skin that folds back or wants more than one patch");
/// What a click picks first when the box opens: the box takes a fresh draft of its network.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Custom(NET_FRESH);

/// The feature the panel's button starts, or why it cannot: a network is drawn on a scan.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    if ws.scan.is_empty() {
        return Err("A patch network is drawn on a scan: open one in the Mesh workspace.".into());
    }
    Ok(FeatureKind::Network(empty_network()))
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut NetworkFeature, pick: &mut PickWhat) -> bool {
    network_layer::box_ui(ws, ui, x, pick)
}

pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut NetworkFeature, pick: &mut PickWhat) -> bool {
    network_layer::pointer(ws, vp, p, clicked, x, pick)
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &NetworkFeature) {
    network_layer::paint(ws, painter, vp, x);
}
