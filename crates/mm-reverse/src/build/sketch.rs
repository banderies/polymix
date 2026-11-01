// SPDX-License-Identifier: AGPL-3.0-only
//! The Sketch feature's build: the sketch's frame and its closed regions, for the features
//! that pick from it. Ported from ScanRuler (`plugins/reverse/core/build.ts` placeSketch,
//! `resolve.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A sketch makes no body. One on a body's face is placed again on the face as it stands: the
//! face found by its fingerprint and the sketch moved with it, so a face pushed out by an edit
//! to an earlier feature takes its sketch along; a face lost leaves the sketch where it was,
//! and the row says so.

use super::{BuiltSketch, Builder};
use crate::model::{Feature, Part, PlaneRef, SketchFeature};
use crate::regions::find_regions;
use crate::frame::SectionFrame;

pub(crate) fn build(b: &mut Builder, part: &Part, f: &Feature, s: &SketchFeature) -> Result<(), String> {
    let mut frame = s.frame;
    // ---- SKETCH ---- the face found, for its outline's projected pieces.
    let mut face_found: Option<(crate::model::BodyId, usize)> = None;
    if let PlaneRef::BodyFace { body, face, offset, .. } = &s.plane {
        let found = b.find_faces(*body, std::slice::from_ref(face), &[]);
        let fingerprint = found.as_ref().ok().and_then(|f| f.first().copied()).and_then(|i| b.body(*body).and_then(|st| st.topology().faces.get(i).map(|x| x.fingerprint)));
        match (found, fingerprint) {
            (Ok(at), Some(fp)) if fp.kind == crate::topo::FaceKind::Plane => {
                if let Some(on) = SectionFrame::sketch_on(fp.centroid, fp.dir, Some(s.frame.u)) {
                    frame = on.offset(offset.unwrap_or(0.0));
                }
                face_found = at.first().map(|i| (*body, *i));
            }
            (Err(lost), _) => b.warn(format!("The face the sketch lies on was not found ({lost}); the sketch stays where it was.")),
            _ => b.warn("The face the sketch lies on was not found (the face is not flat any more); the sketch stays where it was."),
        }
    }
    let placed = place(b, part, s, &frame, face_found);
    let regions = find_regions(placed.as_ref().unwrap_or(&s.sketch));
    b.out.sketches.insert(f.id, BuiltSketch { frame, regions, placed });
    Ok(())
}

// ---- SKETCH ----

/// The sketch's projected pieces taken again from their sources (ScanRuler's `placeSketch`):
/// the sketch as placed when any moved, its row told of what was not found.
fn place(b: &mut Builder, part: &Part, s: &SketchFeature, frame: &SectionFrame, face: Option<(crate::model::BodyId, usize)>) -> Option<crate::sketch::Sketch> {
    if !s.sketch.entities.iter().any(|e| e.projected) {
        return None;
    }
    let topology = |b: &Builder, id: crate::model::BodyId| b.body(id).map(|st| (st.topology(), crate::topo::match_tolerance(st.diagonal())));
    let outline: Vec<crate::topo::EdgeInfo> = face
        .and_then(|(body, i)| {
            let t = b.body(body)?.topology();
            let hash = t.faces.get(i)?.hash;
            Some(t.edges.iter().filter(|e| e.faces.contains(&hash)).cloned().collect())
        })
        .unwrap_or_default();
    let others = |id: crate::model::FeatureId| -> Option<(SectionFrame, crate::sketch::Sketch)> {
        let built = b.out.sketches.get(&id);
        let stored = part.sketch(id)?;
        let frame = built.map_or(stored.frame, |x| x.frame);
        Some((frame, built.and_then(|x| x.placed.clone()).unwrap_or_else(|| stored.sketch.clone())))
    };
    let placed = crate::project::place_projections(&s.sketch, frame, face.map(|(body, _)| (body, outline.as_slice())), |id| topology(b, id), others);
    for note in placed.notes {
        b.warn(note);
    }
    placed.sketch
}
