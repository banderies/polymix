// SPDX-License-Identifier: AGPL-3.0-only
//! What the sketch being drawn can project (ScanRuler's `projectablesOf`, in
//! `plugins/reverse/app/useReverseWorkspace.ts`, © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only): the edges of every body on show and the pieces of every other sketch on
//! show, each as the piece it would become on the sketch's plane — those that project at all —
//! with what it is taken from, so it follows that (`mm_reverse::project`).

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use mm_reverse::model::FeatureKind;
use mm_reverse::project::{project_edge, project_sketch_entity, projectable_polyline, ProjectedEdge, Projectable};
use mm_reverse::sketch::ProjectionSource;

use super::sketcher::{SketchEdit, Tool};
use super::ReverseWorkspace;

impl ReverseWorkspace {
    /// Find again what the open sketch's Project tool can take, when the bodies, the sketches
    /// or the plane changed since it was last found.
    pub(crate) fn refresh_projectables(&self, edit: &mut SketchEdit) {
        if edit.tool != Tool::Project {
            return;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for b in self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)) {
            (Arc::as_ptr(&b.topology) as usize).hash(&mut h);
        }
        self.part.features.len().hash(&mut h);
        for f in &self.part.features {
            f.id.hash(&mut h);
            f.visible().hash(&mut h);
        }
        format!("{:?}", edit.frame).hash(&mut h);
        let key = h.finish();
        if edit.projectables_key == Some(key) {
            return;
        }
        edit.projectables = self.projectables(edit);
        edit.projectables_key = Some(key);
    }

    fn projectables(&self, edit: &SketchEdit) -> Vec<Projectable> {
        let frame = edit.frame;
        let mut out = Vec::new();
        let mut take = |entity: Option<mm_reverse::sketch::Entity>, source: String| {
            if let Some(entity) = entity {
                let polyline = projectable_polyline(&entity);
                out.push(Projectable { entity, polyline, source });
            }
        };
        for b in self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)) {
            for edge in &b.topology.edges {
                let e = project_edge(ProjectedEdge::from(edge), &frame).map(|mut e| {
                    e.source = Some(ProjectionSource::Edge { body: b.id, edge: edge.fingerprint.clone() });
                    e
                });
                take(e, format!("an edge of {}", b.name));
            }
        }
        // A sketch further down the list than the one being edited is not there yet where this
        // one is built: its pieces are taken as they lie, and stay so.
        let at = match edit.feature {
            None => Some(self.part.features.len()),
            Some(id) => self.part.features.iter().position(|f| f.id == id),
        };
        for (i, f) in self.part.features.iter().enumerate() {
            let FeatureKind::Sketch(s) = &f.kind else { continue };
            if !f.visible() || Some(f.id) == edit.feature {
                continue;
            }
            let from = self.sketch_frame(f.id).unwrap_or(s.frame);
            for piece in &s.sketch.entities {
                let p = project_sketch_entity(piece, &from, &frame).map(|mut p| {
                    if at.is_none_or(|at| i < at) {
                        p.source = Some(ProjectionSource::Sketch { sketch: f.id, entity: piece.id });
                    }
                    p
                });
                take(p, format!("a piece of {}", f.name));
            }
        }
        out
    }
}
