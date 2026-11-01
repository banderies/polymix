// SPDX-License-Identifier: AGPL-3.0-only
//! The steps of a build in groups that work on bodies of their own. Ported from ScanRuler
//! (`plugins/reverse/core/buildGroups.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Two groups never read or change a body the other made, so each can be built on a thread of
//! its own at the same time — the base of a screw cap and its knurl on one, the cap and its
//! knurl on another — and the build of the whole list takes every step over from them (the
//! engine's take-over, which checks what each step read). Worked out from the features' fields
//! alone, before anything is built: which bodies a step reads, which it changes, whether it
//! makes one. A step that joins or cuts whatever it meets reads every body there is, and puts
//! them all in one group. A guess that turns out wrong costs time, never the result.

use std::collections::{HashMap, HashSet};

use crate::model::{body_maker, references_of, BodyId, BodyOp, ExtrudeExtent, Feature, FeatureId, FeatureKind, Part, PlaneRef, SplitKeep};

/// A group of steps: their places in the build, in order, and whether one of them reads the
/// scan.
#[derive(Debug, Clone, PartialEq)]
pub struct StepGroup {
    pub steps: Vec<usize>,
    pub scan: bool,
}

/// The steps in groups, and by step whether it has to be played — it changed, or it reads what
/// a step that has to be played makes.
#[derive(Debug, Clone, PartialEq)]
pub struct Grouped {
    pub groups: Vec<StepGroup>,
    pub dirty: Vec<bool>,
}

/// What one step reads and does to the bodies, by its fields (ScanRuler's `touches`).
struct Touches {
    reads: Vec<BodyId>,
    changes: Vec<BodyId>,
    makes: bool,
    all: bool,
    sketches: Vec<FeatureId>,
    scan: bool,
}

fn touches(f: &Feature) -> Touches {
    let mut t = Touches { reads: Vec::new(), changes: Vec::new(), makes: false, all: false, sketches: Vec::new(), scan: false };
    let solid = |t: &mut Touches, op: BodyOp, target: Option<BodyId>| {
        if op == BodyOp::New {
            t.makes = true;
        } else if let Some(x) = target {
            t.reads.push(x);
            t.changes.push(x);
        } else {
            // Joined to, or cut out of, whatever it meets.
            t.all = true;
            t.makes = op == BodyOp::Join;
        }
    };
    match &f.kind {
        FeatureKind::Sketch(s) => {
            if let PlaneRef::BodyFace { body, .. } = &s.plane {
                t.reads.push(*body);
            }
        }
        FeatureKind::Extrude(e) => {
            match &e.face {
                Some(p) => t.reads.push(p.body),
                None => t.sketches.push(e.sketch_id),
            }
            match &e.extent {
                ExtrudeExtent::ToFace { body, .. } | ExtrudeExtent::ToBody { body, .. } => t.reads.push(*body),
                ExtrudeExtent::ToNext { .. } | ExtrudeExtent::ToLast { .. } => match e.target {
                    Some(x) => t.reads.push(x),
                    None => t.all = true,
                },
                _ => {}
            }
            solid(&mut t, e.op, e.target);
        }
        FeatureKind::Revolve(r) => {
            t.sketches.push(r.sketch_id);
            solid(&mut t, r.op, r.target);
        }
        FeatureKind::Extract(x) => solid(&mut t, x.op, x.target),
        FeatureKind::Loft(l) => {
            t.sketches.extend(l.sketches.iter().copied());
            solid(&mut t, l.op, l.target);
        }
        FeatureKind::Sweep(s) => {
            t.sketches.push(s.profile);
            t.sketches.extend(s.path);
            solid(&mut t, s.op, s.target);
        }
        FeatureKind::Thicken(x) => {
            t.reads.push(x.sheet);
            solid(&mut t, x.op, x.target);
        }
        FeatureKind::Sheet(s) => {
            t.sketches.push(s.sketch_id);
            t.makes = true;
        }
        FeatureKind::Import(_) => t.makes = true,
        // The surfaces it is held to, the bodies its curves run along and its patches meet,
        // the bodies it trims: what the model names as its references.
        FeatureKind::Surface(_) | FeatureKind::Network(_) | FeatureKind::Trim(_) => {
            t.reads.extend(references_of(f));
            t.makes = true;
            t.scan = true;
        }
        FeatureKind::MoveBody(m) => {
            t.reads.extend(m.bodies.iter().copied());
            if m.copy {
                t.makes = true;
            } else {
                t.changes.extend(m.bodies.iter().copied());
            }
        }
        FeatureKind::OffsetBody(o) => {
            t.reads.push(o.body);
            if o.keep {
                t.makes = true;
            } else {
                t.changes.push(o.body);
            }
        }
        FeatureKind::Mirror(m) => {
            t.reads.push(m.body);
            if m.join {
                t.changes.push(m.body);
            } else {
                t.makes = true;
            }
        }
        FeatureKind::Boolean(c) => {
            t.reads.extend([c.body, c.tool]);
            t.changes.push(c.body);
        }
        FeatureKind::Split(s) => {
            t.reads.push(s.body);
            t.reads.extend(s.tools.iter().copied());
            t.changes.push(s.body);
            t.scan = s.keep == SplitKeep::Scan;
        }
        other => {
            if let Some(b) = other.body() {
                t.reads.push(b);
                t.changes.push(b);
            }
        }
    }
    t
}

/// The steps in groups that touch no body of another's, in the order of their first steps; and
/// which steps have to be played, by `changed`: the steps no kept build has (ScanRuler's
/// `groupSteps`).
///
/// Beyond ScanRuler: every step that works on bodies counts as writing the bodies of its own
/// id too — a pattern's copies stood apart, a cut through a body's waist — so a later step on
/// one of them goes with it.
pub fn group_steps(_part: &Part, steps: &[&Feature], changed: &[bool]) -> Grouped {
    let n = steps.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    fn join(parent: &mut [usize], a: usize, b: usize) {
        let (x, y) = (find(parent, a), find(parent, b));
        if x != y {
            parent[x.max(y)] = x.min(y);
        }
    }
    // The step that last made or changed each body, by the feature that made it.
    let mut writer: HashMap<FeatureId, usize> = HashMap::new();
    // A body gone into another — a Combine's tool — to the body it went into.
    let mut into: HashMap<FeatureId, FeatureId> = HashMap::new();
    let mut live: Vec<FeatureId> = Vec::new();
    let mut sketch_step: HashMap<FeatureId, usize> = HashMap::new();
    let mut scan: HashSet<usize> = HashSet::new();
    let mut dirty: Vec<bool> = (0..n).map(|i| changed.get(i).copied().unwrap_or(true)).collect();
    let resolve = |into: &HashMap<FeatureId, FeatureId>, id: BodyId| {
        let mut b = body_maker(id);
        for _ in 0..64 {
            match into.get(&b) {
                Some(x) => b = *x,
                None => break,
            }
        }
        b
    };
    for (i, f) in steps.iter().enumerate() {
        let t = touches(f);
        if t.scan {
            scan.insert(i);
        }
        if matches!(f.kind, FeatureKind::Sketch(_)) {
            sketch_step.insert(f.id, i);
        }
        for id in &t.sketches {
            if let Some(at) = sketch_step.get(id).copied().filter(|at| *at != i) {
                join(&mut parent, i, at);
                if dirty[at] {
                    dirty[i] = true;
                }
            }
        }
        let mut read: Vec<FeatureId> = if t.all { live.clone() } else { Vec::new() };
        read.extend(t.reads.iter().map(|id| resolve(&into, *id)));
        for b in read {
            let Some(w) = writer.get(&b).copied() else { continue };
            join(&mut parent, i, w);
            if dirty[w] {
                dirty[i] = true;
            }
        }
        if t.all {
            for b in &live {
                writer.insert(*b, i);
            }
        }
        for b in t.changes.iter().map(|id| resolve(&into, *id)) {
            writer.insert(b, i);
            if !live.contains(&b) {
                live.push(b);
            }
        }
        // What it makes, and the further bodies a step on a body may leave under its own id.
        if t.makes || t.all || !t.changes.is_empty() {
            writer.insert(f.id, i);
            if !live.contains(&f.id) {
                live.push(f.id);
            }
        }
        if let FeatureKind::Boolean(c) = &f.kind {
            if !c.keep {
                let (tool, body) = (resolve(&into, c.tool), resolve(&into, c.body));
                if tool != body {
                    into.insert(tool, body);
                    live.retain(|x| *x != tool);
                }
            }
        }
    }
    let mut groups: Vec<(usize, StepGroup)> = Vec::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        match groups.iter_mut().find(|(r, _)| *r == root) {
            Some((_, g)) => {
                g.steps.push(i);
                g.scan |= scan.contains(&i);
            }
            None => groups.push((root, StepGroup { steps: vec![i], scan: scan.contains(&i) })),
        }
    }
    let mut groups: Vec<StepGroup> = groups.into_iter().map(|(_, g)| g).collect();
    groups.sort_by_key(|g| g.steps[0]);
    Grouped { groups, dirty }
}
