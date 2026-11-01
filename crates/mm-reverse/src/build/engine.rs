// SPDX-License-Identifier: AGPL-3.0-only
//! The build engine: what a build keeps between builds, and how the next one takes up from it.
//! Ported from ScanRuler (`plugins/reverse/core/build.ts` Builder.build, playFrom, takeOver,
//! snapshot, restore, stand, hold, letGo, trim, withChangesSaid; `buildReads.ts`;
//! `buildTypes.ts` stepKeys, Snapshot, PrefixNode, KeptBuild, BuildSteps; `reParallel.ts`;
//! `reEvents.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The state after every step of a build is kept, in a tree of steps by their keys (what each
//! step is built from — the feature and the sketches it reads, as one hash). A build whose
//! first steps the tree holds restores the state after them and plays only the rest: an
//! appended feature costs its own step, the roll-back bar moved is a state looked up. Past the
//! first step changed, a step whose reads of the builder's state stand as they did in a kept
//! build — the very bodies, ids and sketches it looked up — is taken over from it rather than
//! played: what it did there (the bodies it set and took away, what it said of its feature) is
//! laid on the build as it stands. The builder notes every lookup a step makes; a map reached
//! any way but by the builder's own lookups counts as read whole.
//!
//! Where the steps fall into groups that work on bodies of their own (`groups.rs`) and there is
//! work worth sharing, the groups are built at the same time on helper threads — each a
//! builder of its own, taking over from the kept builds where it can — and the build of the
//! whole list takes their steps over. A guess that turns out wrong costs time, never the
//! result: a step whose reads do not stand is played again.
//!
//! The kept states are bounded by the bytes of the shapes they hold (`CACHE_BYTES`); the build
//! standing's are kept whatever they come to, the others let go from the far ends of the tree,
//! the oldest used first. A build is stopped between two steps when its control says so; a
//! kernel call under way runs to its end first (a kernel call cannot be interrupted).

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use solidcraft_kernel as kernel;

use super::{BodyState, Builder, Built, BuiltSketch};
use crate::model::{BodyId, BodyOp, Feature, FeatureId, FeatureKind, Part};

/// The kernel the build runs on, as the Model group names it — the SolidCraft kernel pinned in
/// the workspace's Cargo.toml (keep the two together).
pub const KERNEL_VERSION: &str = "SolidCraft kernel 0.29.0 (e4a67d1)";

/// What the states the engine keeps may hold, bytes of their shapes (SolidCraft's own reckoning
/// of a body's B-rep, `Body::retained_bytes`) — the build standing's are kept whatever they
/// come to (ScanRuler's `CACHE_BYTES`).
pub const CACHE_BYTES: usize = 96 * (1 << 20);

/// How many steps' play times are kept, the oldest let go first.
const PLAY_TIMES: usize = 2000;

/// A failure's note on what changed since it built names this many features at most.
const CHANGES_NAMED: usize = 5;

/// The work, ms, a build's second largest group has to be worth for the build to be shared out
/// (ScanRuler's `SHARE_MIN_MS`; its `SHARE_COLD_MS`, for a helper worker's first start, has no
/// counterpart: a helper thread starts at once).
pub const SHARE_MIN_MS: f64 = 800.0;

/// What a step of no known time is taken to cost, ms: one of these kinds — a boolean, a round,
/// a coil — and the rest.
const HEAVY_MS: f64 = 1000.0;
const LIGHT_MS: f64 = 20.0;

// ---- a map that notes being reached ----

/// One of the builder's maps, reached by code outside the builder's own lookups through `Deref`
/// — which counts as a read of all of it, so a step that goes through the map itself is never
/// taken over on a guess. The builder's lookups use `raw` and note the key they look up.
#[derive(Clone, Default)]
pub(crate) struct Tracked<T> {
    pub(crate) raw: T,
    touched: Cell<bool>,
}

impl<T> Tracked<T> {
    pub(crate) fn new(raw: T) -> Tracked<T> {
        Tracked { raw, touched: Cell::new(false) }
    }

    /// Whether the map was reached whole since this was last asked.
    fn take_touched(&self) -> bool {
        self.touched.replace(false)
    }
}

impl<T> Deref for Tracked<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.touched.set(true);
        &self.raw
    }
}

impl<T> DerefMut for Tracked<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.touched.set(true);
        &mut self.raw
    }
}

// ---- what a step read ----

/// The keys a step looked up in one map, or all of it (ScanRuler's `StepReads` entry).
#[derive(Clone, Debug)]
pub(crate) enum Keys<K> {
    Some(HashSet<K>),
    All,
}

impl<K: std::hash::Hash + Eq> Keys<K> {
    fn note(slot: &mut Option<Keys<K>>, key: K) {
        match slot {
            Some(Keys::All) => {}
            Some(Keys::Some(set)) => {
                set.insert(key);
            }
            None => *slot = Some(Keys::Some(HashSet::from([key]))),
        }
    }
}

/// What a step read of the builder's state, map by map: the keys it looked up, or all of it. A
/// map it never looked at is `None` (ScanRuler's `StepReads`).
#[derive(Clone, Debug, Default)]
pub(crate) struct StepReads {
    bodies: Option<Keys<BodyId>>,
    aliases: Option<Keys<BodyId>>,
    issued: Option<Keys<BodyId>>,
    pieces: Option<Keys<BodyId>>,
    sketches: Option<Keys<FeatureId>>,
}

/// What the builder keeps for the engine beside its maps: what the step playing reads, the
/// sketches' marks, the features' names, and what a step that splits a body says it did.
#[derive(Clone, Default)]
pub(crate) struct Tracking {
    reading: RefCell<Option<StepReads>>,
    /// A mark per sketch built, a fresh one each time the sketch's step is played: what a step
    /// that read the sketch is checked against.
    pub(crate) sketch_marks: HashMap<FeatureId, u64>,
    /// The features' names, by id — what the words of a split or a failure name them by.
    pub(crate) names: Arc<HashMap<FeatureId, String>>,
    /// How the step playing says a split it makes: "The cut at y 5", "The combine", "The cut".
    pub(crate) split_what: Option<String>,
}

impl Tracking {
    fn with<F: FnOnce(&mut StepReads)>(&self, f: F) {
        if let Some(r) = self.reading.borrow_mut().as_mut() {
            f(r);
        }
    }

    pub(crate) fn note_body(&self, id: BodyId) {
        self.with(|r| Keys::note(&mut r.bodies, id));
    }

    pub(crate) fn note_bodies_all(&self) {
        self.with(|r| r.bodies = Some(Keys::All));
    }

    pub(crate) fn note_alias(&self, id: BodyId) {
        self.with(|r| Keys::note(&mut r.aliases, id));
    }

    pub(crate) fn note_issued(&self, id: BodyId) {
        self.with(|r| Keys::note(&mut r.issued, id));
    }

    pub(crate) fn note_piece(&self, id: BodyId) {
        self.with(|r| Keys::note(&mut r.pieces, id));
    }

    pub(crate) fn note_sketch(&self, id: FeatureId) {
        self.with(|r| Keys::note(&mut r.sketches, id));
    }

    /// A feature's name, for words.
    pub(crate) fn name_of(&self, id: FeatureId) -> Option<&str> {
        self.names.get(&id).map(String::as_str)
    }
}

/// A fresh mark for a sketch built, unique across every builder of the run.
fn next_mark() -> u64 {
    static MARK: AtomicU64 = AtomicU64::new(1);
    MARK.fetch_add(1, Ordering::Relaxed)
}

// ---- the builder's state after a step ----

/// A sketch as a state holds it: its mark and what the build made of it.
pub(crate) struct SketchEntry {
    mark: u64,
    sketch: BuiltSketch,
}

/// Everything the builder holds after a step, and what the step said of its features — what a
/// later build that starts with the same steps takes up from (ScanRuler's `Snapshot`). The
/// bodies are shared with the builder and with the states after it, never changed: a step puts
/// a new body in place of the one it works on.
pub(crate) struct Snapshot {
    bodies: Vec<(BodyId, Arc<BodyState>)>,
    aliases: HashMap<BodyId, BodyId>,
    issued: HashSet<BodyId>,
    pieces: HashMap<BodyId, Vec<BodyId>>,
    /// The threads whose faceted cut waits (HOLE, `thread::flush`): made by a later step.
    threads: Vec<super::thread::PendingCut>,
    sketches: Arc<HashMap<FeatureId, Arc<SketchEntry>>>,
    /// ---- SURF ---- What the surface features keep beside the bodies (fits, the shapes fitted
    /// as, sheets reshaped, sheet chains); its scan is the engine's, put back by `restore`.
    surf: super::surf::SurfState,
    /// What the step said of its features: the results it added (bodies and sketches empty).
    said: Built,
}

impl Snapshot {
    /// The state before any step.
    fn nothing() -> Snapshot {
        Snapshot { bodies: Vec::new(), aliases: HashMap::new(), issued: HashSet::new(), pieces: HashMap::new(), threads: Vec::new(), sketches: Arc::new(HashMap::new()), surf: Default::default(), said: Built::default() }
    }

    /// The solids it holds, each once, by the kernel's identity of a solid.
    fn shapes(&self, into: &mut HashMap<usize, kernel::Body>) {
        for (_, b) in &self.bodies {
            into.entry(b.shape.solid_id()).or_insert_with(|| b.shape.clone());
        }
    }
}

/// The keys of the results a build holds, before a step: what tells the results it adds.
/// Written out field by field — a field added to `Built` has to be named here and in
/// `results_since` and `absorb`, or the crate does not build: a result a build taken up from a
/// kept one would otherwise lose.
struct ResultKeys {
    errors: HashSet<FeatureId>,
    warnings: HashSet<FeatureId>,
    touched: HashSet<FeatureId>,
    failed_at: HashSet<FeatureId>,
    failed_where: HashSet<FeatureId>,
    ghost_volumes: HashSet<FeatureId>,
    surf: HashSet<FeatureId>,
}

impl ResultKeys {
    fn of(out: &Built) -> ResultKeys {
        let Built { bodies: _, sketches: _, errors, warnings, touched, failed_at, failed_where, ghost_volumes, surf } = out;
        ResultKeys {
            errors: errors.keys().copied().collect(),
            warnings: warnings.keys().copied().collect(),
            touched: touched.keys().copied().collect(),
            failed_at: failed_at.keys().copied().collect(),
            failed_where: failed_where.keys().copied().collect(),
            ghost_volumes: ghost_volumes.keys().copied().collect(),
            surf: surf.keys().copied().collect(),
        }
    }
}

/// The results a step added: its own feature's, and any it gave for a feature none was given
/// for before it.
fn results_since(out: &Built, before: &ResultKeys, step: FeatureId) -> Built {
    let Built { bodies: _, sketches: _, errors, warnings, touched, failed_at, failed_where, ghost_volumes, surf } = out;
    fn new<V: Clone>(map: &HashMap<FeatureId, V>, had: &HashSet<FeatureId>, step: FeatureId) -> HashMap<FeatureId, V> {
        map.iter().filter(|(k, _)| **k == step || !had.contains(k)).map(|(k, v)| (*k, v.clone())).collect()
    }
    Built {
        bodies: Vec::new(),
        sketches: HashMap::new(),
        errors: new(errors, &before.errors, step),
        warnings: new(warnings, &before.warnings, step),
        touched: new(touched, &before.touched, step),
        failed_at: new(failed_at, &before.failed_at, step),
        failed_where: new(failed_where, &before.failed_where, step),
        ghost_volumes: new(ghost_volumes, &before.ghost_volumes, step),
        surf: new(surf, &before.surf, step),
    }
}

/// A step's results laid on a build's.
fn absorb(into: &mut Built, said: &Built) {
    let Built { bodies: _, sketches: _, errors, warnings, touched, failed_at, failed_where, ghost_volumes, surf } = said;
    into.errors.extend(errors.iter().map(|(k, v)| (*k, v.clone())));
    into.warnings.extend(warnings.iter().map(|(k, v)| (*k, v.clone())));
    into.touched.extend(touched.iter().map(|(k, v)| (*k, v.clone())));
    into.failed_at.extend(failed_at.iter().map(|(k, v)| (*k, v.clone())));
    into.failed_where.extend(failed_where.iter().map(|(k, v)| (*k, v.clone())));
    into.ghost_volumes.extend(ghost_volumes.iter().map(|(k, v)| (*k, *v)));
    into.surf.extend(surf.iter().map(|(k, v)| (*k, v.clone())));
}

impl Builder {
    /// The builder as it stands after a step, the step's results with it. The sketches are
    /// shared with the state before where the step changed none of them.
    fn snapshot(&self, before: Option<&Snapshot>, said: Built) -> Snapshot {
        let prev = before.map(|s| &s.sketches);
        let same = prev.is_some_and(|p| p.len() == self.track.sketch_marks.len() && self.track.sketch_marks.iter().all(|(id, m)| p.get(id).is_some_and(|e| e.mark == *m)));
        let sketches = match (same, prev) {
            (true, Some(p)) => p.clone(),
            _ => Arc::new(
                self.track
                    .sketch_marks
                    .iter()
                    .filter_map(|(id, m)| {
                        let kept = prev.and_then(|p| p.get(id)).filter(|e| e.mark == *m).cloned();
                        let entry = kept.or_else(|| self.out.sketches.get(id).map(|s| Arc::new(SketchEntry { mark: *m, sketch: s.clone() })))?;
                        Some((*id, entry))
                    })
                    .collect(),
            ),
        };
        Snapshot { bodies: self.bodies.raw.clone(), aliases: self.aliases.raw.clone(), issued: self.issued.raw.clone(), pieces: self.pieces.raw.clone(), threads: self.threads.clone(), sketches, surf: super::surf::SurfState { scan: None, ..self.surf.clone() }, said }
    }

    /// The builder put back as it stood after a step — or as it is before any build — with the
    /// results of the steps up to it.
    fn restore(&mut self, state: Option<&Snapshot>, results: Built) {
        let empty = Snapshot::nothing();
        let s = state.unwrap_or(&empty);
        self.bodies = Tracked::new(s.bodies.clone());
        self.aliases = Tracked::new(s.aliases.clone());
        self.issued = Tracked::new(s.issued.clone());
        self.pieces = Tracked::new(s.pieces.clone());
        self.threads = s.threads.clone();
        // ---- SURF ---- the state kept, over the scan this build reads.
        let scan = self.surf.scan.take();
        self.surf = super::surf::SurfState { scan, ..s.surf.clone() };
        self.out = results;
        self.out.sketches = s.sketches.iter().map(|(id, e)| (*id, e.sketch.clone())).collect();
        self.track.sketch_marks = s.sketches.iter().map(|(id, e)| (*id, e.mark)).collect();
    }

    /// The sketches' marks after a step was played: the step's own sketch, and any sketch new
    /// to the build, marked afresh; a sketch gone, forgotten.
    fn mark_sketches(&mut self, step: FeatureId) {
        let ids: Vec<FeatureId> = self.out.sketches.keys().copied().collect();
        self.track.sketch_marks.retain(|id, _| ids.contains(id));
        for id in ids {
            if id == step || !self.track.sketch_marks.contains_key(&id) {
                self.track.sketch_marks.insert(id, next_mark());
            }
        }
    }
}

/// Whether what a step read stands as it did when the step played on `then` — every key it
/// looked up holding the very same value now, every map it went through the same throughout
/// (ScanRuler's `readsStand`).
fn reads_stand(reads: &StepReads, now: &Builder, then: &Snapshot) -> bool {
    fn find<'a>(list: &'a [(BodyId, Arc<BodyState>)], id: &BodyId) -> Option<&'a Arc<BodyState>> {
        list.iter().find(|(b, _)| b == id).map(|(_, s)| s)
    }
    // HOLE: threads' faceted cuts waiting, here or where the step was kept, are no read the
    // engine notes — the step made them, or would have — so such a step is played again.
    if !now.threads.is_empty() || !then.threads.is_empty() {
        return false;
    }
    match &reads.bodies {
        None => {}
        Some(Keys::All) => {
            let (a, b) = (&now.bodies.raw, &then.bodies);
            if a.len() != b.len() || !a.iter().zip(b).all(|(x, y)| x.0 == y.0 && Arc::ptr_eq(&x.1, &y.1)) {
                return false;
            }
        }
        Some(Keys::Some(keys)) => {
            for k in keys {
                match (find(&now.bodies.raw, k), find(&then.bodies, k)) {
                    (None, None) => {}
                    (Some(a), Some(b)) if Arc::ptr_eq(a, b) => {}
                    _ => return false,
                }
            }
        }
    }
    fn map_stands<K: std::hash::Hash + Eq, V: PartialEq>(read: &Option<Keys<K>>, now: &HashMap<K, V>, then: &HashMap<K, V>) -> bool {
        match read {
            None => true,
            Some(Keys::All) => now == then,
            Some(Keys::Some(keys)) => keys.iter().all(|k| now.get(k) == then.get(k)),
        }
    }
    if !map_stands(&reads.aliases, &now.aliases.raw, &then.aliases) || !map_stands(&reads.pieces, &now.pieces.raw, &then.pieces) {
        return false;
    }
    match &reads.issued {
        None => {}
        Some(Keys::All) => {
            if now.issued.raw != then.issued {
                return false;
            }
        }
        Some(Keys::Some(keys)) => {
            if keys.iter().any(|k| now.issued.raw.contains(k) != then.issued.contains(k)) {
                return false;
            }
        }
    }
    let then_marks: HashMap<FeatureId, u64> = match &reads.sketches {
        None => HashMap::new(),
        Some(_) => then.sketches.iter().map(|(id, e)| (*id, e.mark)).collect(),
    };
    map_stands(&reads.sketches, &now.track.sketch_marks, &then_marks)
}

/// What a step did, from the state before it to the state after it, laid on the builder as it
/// stands — entries set where they changed, taken away where they went — and the results it
/// gave (ScanRuler's `replay`).
fn replay(before: &Snapshot, after: &Snapshot, b: &mut Builder) {
    for (id, _) in &before.bodies {
        if !after.bodies.iter().any(|(x, _)| x == id) {
            b.bodies.raw.retain(|(x, _)| x != id);
        }
    }
    for (id, s) in &after.bodies {
        let was = before.bodies.iter().find(|(x, _)| x == id).map(|(_, s)| s);
        if was.is_some_and(|w| Arc::ptr_eq(w, s)) {
            continue;
        }
        match b.bodies.raw.iter_mut().find(|(x, _)| x == id) {
            Some(slot) => slot.1 = s.clone(),
            None => b.bodies.raw.push((*id, s.clone())),
        }
    }
    fn lay<K: std::hash::Hash + Eq + Copy, V: PartialEq + Clone>(before: &HashMap<K, V>, after: &HashMap<K, V>, now: &mut HashMap<K, V>) {
        for k in before.keys() {
            if !after.contains_key(k) {
                now.remove(k);
            }
        }
        for (k, v) in after {
            if before.get(k) != Some(v) {
                now.insert(*k, v.clone());
            }
        }
    }
    lay(&before.aliases, &after.aliases, &mut b.aliases.raw);
    lay(&before.pieces, &after.pieces, &mut b.pieces.raw);
    for k in &before.issued {
        if !after.issued.contains(k) {
            b.issued.raw.remove(k);
        }
    }
    for k in &after.issued {
        b.issued.raw.insert(*k);
    }
    for id in before.sketches.keys() {
        if !after.sketches.contains_key(id) {
            b.out.sketches.remove(id);
            b.track.sketch_marks.remove(id);
        }
    }
    for (id, e) in after.sketches.iter() {
        if before.sketches.get(id).is_some_and(|w| Arc::ptr_eq(w, e)) {
            continue;
        }
        b.out.sketches.insert(*id, e.sketch.clone());
        b.track.sketch_marks.insert(*id, e.mark);
    }
    // ---- SURF ---- the surface features' state, entries set where they changed.
    fn lay_by<K: std::hash::Hash + Eq + Copy, V: Clone>(before: &HashMap<K, V>, after: &HashMap<K, V>, now: &mut HashMap<K, V>, same: impl Fn(&V, &V) -> bool) {
        for k in before.keys() {
            if !after.contains_key(k) {
                now.remove(k);
            }
        }
        for (k, v) in after {
            if !before.get(k).is_some_and(|w| same(w, v)) {
                now.insert(*k, v.clone());
            }
        }
    }
    lay_by(&before.surf.fits, &after.surf.fits, &mut b.surf.fits, |a, c| Arc::ptr_eq(a, c));
    lay(&before.surf.fitted, &after.surf.fitted, &mut b.surf.fitted);
    lay(&before.surf.reshaped, &after.surf.reshaped, &mut b.surf.reshaped);
    lay_by(&before.surf.chains, &after.surf.chains, &mut b.surf.chains, |a, c| Arc::ptr_eq(a, c));
    absorb(&mut b.out, &after.said);
}

// ---- one step played ----

/// One feature played on the builder as it stands (ScanRuler's `play` as `playFrom` calls it):
/// its kind's build, a failure said on its row — diagnosed where the kind has a diagnosis — a
/// cut that also cut elsewhere warned of, and what it did its own way. With `track`, what it
/// read of the builder's state, for a later build to take it over by.
pub(crate) fn play_step(b: &mut Builder, part: &Part, f: &Feature, track: bool) -> StepReads {
    // HOLE: the threads' faceted cuts waiting are made when a feature other than a thread or a
    // sketch comes to work on the bodies (`thread::flush`).
    if !matches!(f.kind, FeatureKind::Thread(_) | FeatureKind::Sketch(_)) {
        super::thread::flush(b);
    }
    b.step = f.id;
    b.warnings.clear();
    b.track.split_what = Some(super::split_words::split_what(part, f));
    // What a cut takes, read off the bodies before it — not a read of the step's.
    let cutting = super::cut_elsewhere::cut_makers(b, f);
    let uncut: Option<Vec<(BodyId, Arc<BodyState>)>> = cutting.as_ref().map(|_| b.bodies.raw.clone());
    // What reached the maps before the step is none of the step's reading.
    let _ = (b.bodies.take_touched(), b.aliases.take_touched(), b.issued.take_touched(), b.pieces.take_touched());
    if track {
        *b.track.reading.borrow_mut() = Some(StepReads::default());
    }
    let began = Instant::now();
    let result = kernel::guard(&f.name, || Ok(super::build_one(b, part, f))).unwrap_or_else(|e| Err(e.to_string()));
    let result = match result {
        Err(e) => Err(super::diagnose::diagnosed(b, part, f, e, began.elapsed())),
        Ok(()) => Ok(()),
    };
    let mut reads = b.track.reading.borrow_mut().take().unwrap_or_default();
    // A map reached past the builder's lookups was read whole.
    if b.bodies.take_touched() {
        reads.bodies = Some(Keys::All);
    }
    if b.aliases.take_touched() {
        reads.aliases = Some(Keys::All);
    }
    if b.issued.take_touched() {
        reads.issued = Some(Keys::All);
    }
    if b.pieces.take_touched() {
        reads.pieces = Some(Keys::All);
    }
    // The sketches the feature takes its profiles from are read, by whatever way.
    for id in sketches_read(f, part) {
        Keys::note(&mut reads.sketches, id);
    }
    match result {
        Err(e) => {
            b.out.errors.insert(f.id, e);
        }
        Ok(()) => {
            if let (Some(mine), Some(uncut)) = (cutting, uncut) {
                super::cut_elsewhere::note_cuts_elsewhere(b, f, &mine, &uncut);
            }
        }
    }
    if !b.warnings.is_empty() {
        let mut said: Vec<String> = Vec::new();
        for w in std::mem::take(&mut b.warnings) {
            if !said.contains(&w) {
                said.push(w);
            }
        }
        b.out.warnings.insert(f.id, said);
    }
    b.mark_sketches(f.id);
    b.track.split_what = None;
    reads
}

/// Whether a step works with the threads' faceted cuts waiting (HOLE): a thread queues one, and
/// any step but a sketch makes those waiting. The engine notes neither as a read nor lays it in
/// a take-over, so such a step is always played, never taken over.
fn thread_bound(b: &Builder, f: &Feature) -> bool {
    matches!(f.kind, FeatureKind::Thread(_)) || (!b.threads.is_empty() && !matches!(f.kind, FeatureKind::Sketch(_)))
}

/// The sketches a step reads beside its own fields, by id: the sketches its profiles come from
/// (ScanRuler's `sketchesRead`).
pub(crate) fn sketches_read(f: &Feature, part: &Part) -> Vec<FeatureId> {
    let mut out = f.kind.source_sketches();
    if let FeatureKind::Extrude(e) = &f.kind {
        // A face pulled has no sketch; the sketch id still names one an old part kept.
        if e.face.is_some() && part.sketch(e.sketch_id).is_some() && !out.contains(&e.sketch_id) {
            out.push(e.sketch_id);
        }
    }
    out.retain(|id| *id != f.id);
    out
}

// ---- the steps' keys ----

/// What one step of a build is built from, as one hash: the feature, the sketches it reads,
/// and what of the part every step may read (its elements, parameters and STEP files) — two
/// builds whose first steps have the same keys stand the same after them.
pub type StepKey = u128;

/// FNV-1a, 128 bits: a key no two different steps come to in practice.
struct Fnv(u128);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0x6c62_272e_07bb_0142_62b8_2175_6295_c58d)
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u128;
            self.0 = self.0.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
        }
        // A separator, so that "ab" + "c" is not "a" + "bc".
        self.0 ^= 0xff;
        self.0 = self.0.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
    }
}

/// A value as text, telling apart what JSON alone would not: ±infinity and NaN all write as
/// null, so their places are written after it (ScanRuler's `keyOf`).
fn key_text<T: serde::Serialize + std::fmt::Debug>(value: &T) -> String {
    let mut text = serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"));
    let debug = format!("{value:?}");
    if debug.contains("NaN") || debug.contains("inf") {
        let marks: Vec<&str> = debug.split(|c: char| !c.is_alphanumeric() && c != '-').filter(|w| *w == "NaN" || *w == "inf" || *w == "-inf").collect();
        text.push('#');
        text.push_str(&marks.join(","));
    }
    text
}

/// The keys of the steps (ScanRuler's `stepKeys`).
pub fn step_keys(part: &Part, steps: &[&Feature]) -> Vec<StepKey> {
    let mut context = Fnv::new();
    context.write(key_text(&part.elements).as_bytes());
    context.write(key_text(&part.parameters).as_bytes());
    for k in part.step_files.keys() {
        context.write(k.as_bytes());
    }
    // Whether a feature is shown in the view is none of the build's: a sketch hidden or shown
    // builds nothing again.
    let as_built = |f: &Feature| {
        let mut f = f.clone();
        f.flags.visible = true;
        key_text(&f)
    };
    steps
        .iter()
        .map(|f| {
            let mut h = Fnv(context.0);
            h.write(as_built(f).as_bytes());
            for id in sketches_read(f, part) {
                h.write(&id.to_le_bytes());
                match part.feature(id) {
                    Some(s) => h.write(as_built(s).as_bytes()),
                    None => h.write(b"null"),
                }
            }
            h.0
        })
        .collect()
}

// ---- the build's control ----

/// What a build is at, and a way to stop it between steps — shared by the build's thread and
/// the window (ScanRuler's `BuildInPlay` and `reEvents.ts`: each kernel tells the step it is
/// about to play).
#[derive(Clone, Default)]
pub struct BuildControl {
    inner: Arc<ControlInner>,
}

#[derive(Default)]
struct ControlInner {
    stopped: AtomicBool,
    /// By kernel — 0 the build's own thread, n the n-th helper — the step in play and since
    /// when.
    at: Mutex<HashMap<usize, (FeatureId, Instant)>>,
    /// Whether a step of the build has been told.
    told: AtomicBool,
    /// How long each step is held before it is played, ms — for scripted tests of a stop.
    slow_ms: AtomicU64,
}

/// The step a build is at the longest, on whichever thread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InPlay {
    pub feature: FeatureId,
    pub for_how_long: Duration,
}

impl BuildControl {
    pub fn new() -> BuildControl {
        BuildControl::default()
    }

    /// Stop the build at its next step.
    pub fn stop(&self) {
        self.inner.stopped.store(true, Ordering::SeqCst);
    }

    pub fn stopped(&self) -> bool {
        self.inner.stopped.load(Ordering::SeqCst)
    }

    /// The step in play the longest, or `None` between steps.
    pub fn in_play(&self) -> Option<InPlay> {
        let at = self.inner.at.lock().ok()?;
        at.values().min_by_key(|(_, since)| *since).map(|(id, since)| InPlay { feature: *id, for_how_long: since.elapsed() })
    }

    /// Whether any step of the build under way has been told: with none in play, the build is
    /// past its last step (meshing the bodies) rather than before its first.
    pub fn played(&self) -> bool {
        self.inner.told.load(Ordering::SeqCst)
    }

    /// A build begins: nothing told yet.
    pub fn begin(&self) {
        self.inner.told.store(false, Ordering::SeqCst);
        if let Ok(mut at) = self.inner.at.lock() {
            at.clear();
        }
    }

    /// Every step held this long before it is played: a build made slow, for scripted tests of
    /// a stop (a hold is cut short by the stop).
    pub fn slow_steps(&self, by: Duration) {
        self.inner.slow_ms.store(by.as_millis() as u64, Ordering::SeqCst);
    }

    /// The hold `slow_steps` asked for, cut short by a stop.
    fn dwell(&self) {
        let ms = self.inner.slow_ms.load(Ordering::SeqCst);
        let until = Instant::now() + Duration::from_millis(ms);
        while ms > 0 && Instant::now() < until && !self.stopped() {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// A kernel is about to play a step, or is done with the steps given it.
    pub(crate) fn tell(&self, kernel: usize, step: Option<FeatureId>) {
        if let Ok(mut at) = self.inner.at.lock() {
            match step {
                Some(id) => {
                    at.insert(kernel, (id, Instant::now()));
                    self.inner.told.store(true, Ordering::SeqCst);
                }
                None => {
                    at.remove(&kernel);
                }
            }
        }
    }
}

// ---- the engine ----

/// What a build did with its steps: played them (recomputed), or took them up from a state the
/// engine held or over from another build (restored), of how many; the steps played with how
/// long each took, ms (ScanRuler's `BuildSteps`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuildSteps {
    pub recomputed: usize,
    pub restored: usize,
    pub of: usize,
    pub played: Vec<(FeatureId, f64)>,
}

/// What a build is asked for.
#[derive(Debug, Clone)]
pub struct BuildAsk {
    /// How many of the part's features to build (the roll-back bar).
    pub upto: usize,
    /// The feature a box is open on: the part as it stood before it is handed back too, for
    /// the box's picks.
    pub before: Option<FeatureId>,
    /// Take up from the kept builds and keep this one (ScanRuler's `reuse`); without, every
    /// step is played and nothing is kept.
    pub reuse: bool,
    /// Share the build out between threads where it is worth it.
    pub parallel: bool,
}

/// A build: the part as built, and as it stood before the feature asked for.
pub struct EngineBuild {
    pub built: Built,
    pub before: Option<Built>,
    pub steps: BuildSteps,
}

/// A node of the tree of steps: a step by its key, below the steps before it — the state after
/// it and what it read, while they are held (ScanRuler's `PrefixNode`).
struct Node {
    key: StepKey,
    parent: usize,
    children: HashMap<StepKey, usize>,
    state: Option<Arc<Snapshot>>,
    reads: Option<Arc<StepReads>>,
    used: u64,
}

/// A build kept: its steps' keys, the state after each and what each read (ScanRuler's
/// `KeptBuild`), and which features built.
#[derive(Clone, Default)]
struct Kept {
    keys: Vec<StepKey>,
    states: Vec<Arc<Snapshot>>,
    reads: Vec<Arc<StepReads>>,
    ok: HashSet<FeatureId>,
}

impl Kept {
    fn index(&self, key: StepKey) -> Option<usize> {
        self.keys.iter().position(|k| *k == key)
    }
}

const ROOT: usize = 0;

/// The builds kept, and the build of a part taken up from them (ScanRuler's `Builder` as a
/// whole: its tree, its build standing and its budget).
pub struct Engine {
    nodes: Vec<Option<Node>>,
    free: Vec<usize>,
    by_key: HashMap<StepKey, Vec<usize>>,
    standing: Option<Kept>,
    clock: u64,
    /// What the kept states may hold, bytes ([`CACHE_BYTES`]).
    pub budget: usize,
    /// How many hold each state (by its address) and each solid (by its identity, with its
    /// bytes), and what the solids held come to.
    holders: HashMap<usize, usize>,
    shape_holders: HashMap<usize, (usize, usize)>,
    held_bytes: usize,
    play_ms: HashMap<StepKey, (f64, u64)>,
    play_order: VecDeque<(StepKey, u64)>,
    /// What the last build did with its steps.
    pub last_steps: BuildSteps,
    /// The helper threads a build may share its groups out to (ScanRuler's helper pool: at most
    /// two, two cores left free).
    pub helpers: usize,
    /// The work a build's second largest group has to be worth to be shared out, ms.
    pub share_min_ms: f64,
    /// ---- SURF ---- The scan every build reads (a split, a trim, a fitted surface, a shadow):
    /// set by [`Engine::set_scan`], which lets go of the builds kept when it changes.
    scan: Option<Arc<super::surf::BuildScan>>,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new()
    }
}

impl Engine {
    pub fn new() -> Engine {
        let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
        Engine {
            nodes: vec![Some(Node { key: 0, parent: ROOT, children: HashMap::new(), state: None, reads: None, used: 0 })],
            free: Vec::new(),
            by_key: HashMap::new(),
            standing: None,
            clock: 0,
            budget: CACHE_BYTES,
            holders: HashMap::new(),
            shape_holders: HashMap::new(),
            held_bytes: 0,
            play_ms: HashMap::new(),
            play_order: VecDeque::new(),
            last_steps: BuildSteps::default(),
            helpers: cores.saturating_sub(2).min(2),
            share_min_ms: SHARE_MIN_MS,
            scan: None,
        }
    }

    fn node(&self, i: usize) -> Option<&Node> {
        self.nodes.get(i).and_then(Option::as_ref)
    }

    fn node_mut(&mut self, i: usize) -> Option<&mut Node> {
        self.nodes.get_mut(i).and_then(Option::as_mut)
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Everything kept let go: what a build after a change to what every step reads (the
    /// scan) starts from.
    pub fn clear(&mut self) {
        *self = Engine { helpers: self.helpers, share_min_ms: self.share_min_ms, budget: self.budget, scan: self.scan.take(), ..Engine::new() };
    }

    /// ---- SURF ---- The scan the builds read from now on. The steps' keys do not hold it: a
    /// scan other than the one the kept builds read lets go of them all ([`Engine::clear`]).
    pub fn set_scan(&mut self, scan: Option<Arc<super::surf::BuildScan>>) {
        let same = match (&self.scan, &scan) {
            (None, None) => true,
            (Some(a), Some(c)) => Arc::ptr_eq(a, c),
            _ => false,
        };
        if !same {
            self.clear();
            self.scan = scan;
        }
    }

    /// How much the states kept hold: bytes, solids and states — for the tests.
    pub fn cache_held(&self) -> (usize, usize, usize) {
        (self.held_bytes, self.shape_holders.len(), self.holders.len())
    }

    /// Whether a kept build has a step of this key.
    pub fn knows(&self, key: StepKey) -> bool {
        self.by_key.get(&key).is_some_and(|ns| ns.iter().any(|n| self.node(*n).is_some_and(|x| x.state.is_some())))
    }

    /// How long a step of this key took when it was last played, ms.
    pub fn played_ms(&self, key: StepKey) -> Option<f64> {
        self.play_ms.get(&key).map(|(ms, _)| *ms)
    }

    fn timed(&mut self, key: StepKey, ms: f64) {
        let t = self.tick();
        self.play_ms.insert(key, (ms, t));
        self.play_order.push_back((key, t));
        while self.play_ms.len() > PLAY_TIMES {
            let Some((k, at)) = self.play_order.pop_front() else { break };
            if self.play_ms.get(&k).is_some_and(|(_, when)| *when == at) {
                self.play_ms.remove(&k);
            }
        }
        if self.play_order.len() > 4 * PLAY_TIMES {
            let live = &self.play_ms;
            self.play_order.retain(|(k, at)| live.get(k).is_some_and(|(_, when)| when == at));
        }
    }

    /// Build the part (ScanRuler's `Builder.build`): the longest run of its first steps the
    /// tree holds a state for restored, the rest played or taken over. `None` when the control
    /// stopped it.
    pub fn build(&mut self, part: &Part, ask: &BuildAsk, control: &BuildControl) -> Option<EngineBuild> {
        let upto = ask.upto.min(part.features.len());
        let steps: Vec<&Feature> = part.features[..upto].iter().filter(|f| !f.suppressed()).collect();
        let keys = if ask.reuse { step_keys(part, &steps) } else { Vec::new() };
        let names: Arc<HashMap<FeatureId, String>> = Arc::new(part.features.iter().map(|f| (f.id, f.name.clone())).collect());
        // The longest run of first steps the tree holds a state for.
        let mut path: Vec<usize> = Vec::new();
        let mut at = ROOT;
        while path.len() < keys.len() {
            match self.node(at).and_then(|n| n.children.get(&keys[path.len()])).copied() {
                Some(next) if self.node(next).is_some_and(|n| n.state.is_some()) => {
                    path.push(next);
                    at = next;
                }
                _ => break,
            }
        }
        let mut states: Vec<Arc<Snapshot>> = path.iter().filter_map(|n| self.node(*n).and_then(|x| x.state.clone())).collect();
        let mut reads: Vec<Arc<StepReads>> = path.iter().filter_map(|n| self.node(*n).and_then(|x| x.reads.clone())).collect();
        let mut b = Builder::new();
        b.track.names = names.clone();
        b.surf.scan = self.scan.clone();
        let mut results = Built::default();
        for s in &states {
            absorb(&mut results, &s.said);
        }
        b.restore(states.last().map(|s| &**s), results);
        // The groups worth sharing out, built on helper threads (and this one) first.
        let offered = if ask.reuse && ask.parallel && states.len() < steps.len() { self.prebuild(part, &steps, &keys, &names, control) } else { Vec::new() };
        let mut played: Vec<(FeatureId, f64)> = Vec::new();
        control.tell(0, None);
        for i in states.len()..steps.len() {
            if control.stopped() {
                control.tell(0, None);
                return None;
            }
            let f = steps[i];
            let taken = if ask.reuse && !thread_bound(&b, f) { self.take_over(&mut b, keys[i], &offered, true) } else { None };
            let (read, said) = match taken {
                Some(t) => t,
                None => {
                    control.tell(0, Some(f.id));
                    control.dwell();
                    if control.stopped() {
                        control.tell(0, None);
                        return None;
                    }
                    let before = ResultKeys::of(&b.out);
                    let started = Instant::now();
                    let read = play_step(&mut b, part, f, ask.reuse);
                    let ms = started.elapsed().as_secs_f64() * 1000.0;
                    played.push((f.id, ms));
                    if ask.reuse {
                        self.timed(keys[i], ms);
                    }
                    (Arc::new(read), results_since(&b.out, &before, f.id))
                }
            };
            let state = Arc::new(b.snapshot(states.last().map(|s| &**s), said));
            states.push(state);
            reads.push(read);
        }
        control.tell(0, None);
        self.last_steps = BuildSteps { recomputed: played.len(), restored: steps.len() - played.len(), of: steps.len(), played };
        let ok: HashSet<FeatureId> = steps.iter().map(|f| f.id).filter(|id| !b.out.errors.contains_key(id)).collect();
        let was = self.standing.clone();
        let kept = Kept { keys: keys.clone(), states: states.clone(), reads, ok };
        self.stand(kept);
        // The part as it stood before the feature a box is open on.
        let before = ask.before.map(|fid| {
            let at = part.index_of(fid).map_or(upto, |i| i.min(upto));
            let k = steps.iter().filter(|f| part.index_of(f.id).is_some_and(|i| i < at)).count();
            let mut pb = Builder::new();
            pb.track.names = names.clone();
            pb.surf.scan = self.scan.clone();
            let mut results = Built::default();
            for s in &states[..k] {
                absorb(&mut results, &s.said);
            }
            pb.restore(states[..k].last().map(|s| &**s), results);
            // The picks are taken on the bodies with no thread's faceted cut made yet (HOLE).
            pb.threads.clear();
            pb.finish(part)
        });
        let mut built = b.finish(part);
        with_changes_said(&mut built, &steps, &keys, was.as_ref(), &names);
        Some(EngineBuild { built, before, steps: self.last_steps.clone() })
    }

    /// A step's outcome taken over from a kept build instead of played: the same step (by its
    /// key) that read there nothing that does not stand the same now — then what it did there
    /// is done here. What it read and what it said; `None` where no kept build has the step so
    /// (ScanRuler's `takeOver`). `mark`: note the node as used.
    fn take_over(&mut self, b: &mut Builder, key: StepKey, offered: &[Kept], mark: bool) -> Option<(Arc<StepReads>, Built)> {
        let found = self.find_taken(b, key, offered);
        let (n, before, after, reads) = found?;
        replay(&before, &after, b);
        if mark {
            if let Some(n) = n {
                let t = self.tick();
                if let Some(x) = self.node_mut(n) {
                    x.used = t;
                }
            }
        }
        Some((reads, clone_said(&after.said)))
    }

    /// The kept step of this key whose reads stand: its node (in the tree), the states before
    /// and after it, and what it read.
    #[allow(clippy::type_complexity)]
    fn find_taken(&self, b: &Builder, key: StepKey, offered: &[Kept]) -> Option<(Option<usize>, Arc<Snapshot>, Arc<Snapshot>, Arc<StepReads>)> {
        let nothing = Arc::new(Snapshot::nothing());
        let mut held: Vec<usize> = self.by_key.get(&key).map_or(Vec::new(), |ns| ns.iter().copied().filter(|n| self.node(*n).is_some_and(|x| x.state.is_some() && x.reads.is_some())).collect());
        held.sort_by_key(|n| std::cmp::Reverse(self.node(*n).map_or(0, |x| x.used)));
        for n in held {
            let Some(node) = self.node(n) else { continue };
            let before = if node.parent == ROOT { Some(nothing.clone()) } else { self.node(node.parent).and_then(|p| p.state.clone()) };
            let (Some(before), Some(after), Some(reads)) = (before, node.state.clone(), node.reads.clone()) else { continue };
            if reads_stand(&reads, b, &before) {
                return Some((Some(n), before, after, reads));
            }
        }
        for part in offered {
            let Some(j) = part.index(key) else { continue };
            let (Some(after), Some(reads)) = (part.states.get(j), part.reads.get(j)) else { continue };
            let before = if j > 0 { part.states[j - 1].clone() } else { nothing.clone() };
            if reads_stand(reads, b, &before) {
                return Some((None, before, after.clone(), reads.clone()));
            }
        }
        None
    }

    /// Some of the steps — a group of them — built on a builder of their own, taken over from
    /// the kept builds where they can be (ScanRuler's `buildPart`). Reads the tree, changes
    /// nothing of it, so helper threads can run it at once.
    fn build_part(&self, part: &Part, steps: &[&Feature], keys: &[StepKey], names: &Arc<HashMap<FeatureId, String>>, control: &BuildControl, kernel_no: usize) -> Option<Kept> {
        let mut b = Builder::new();
        b.track.names = names.clone();
        b.surf.scan = self.scan.clone();
        let mut kept = Kept::default();
        for (f, key) in steps.iter().zip(keys) {
            if control.stopped() {
                control.tell(kernel_no, None);
                return None;
            }
            let found = if thread_bound(&b, f) { None } else { self.find_taken(&b, *key, &[]) };
            let (read, said) = match found {
                Some((_, before, after, reads)) => {
                    replay(&before, &after, &mut b);
                    (reads, clone_said(&after.said))
                }
                None => {
                    control.tell(kernel_no, Some(f.id));
                    control.dwell();
                    let before = ResultKeys::of(&b.out);
                    let read = play_step(&mut b, part, f, true);
                    (Arc::new(read), results_since(&b.out, &before, f.id))
                }
            };
            let state = Arc::new(b.snapshot(kept.states.last().map(|s| &**s), said));
            kept.keys.push(*key);
            kept.states.push(state);
            kept.reads.push(read);
        }
        control.tell(kernel_no, None);
        Some(kept)
    }

    /// The groups of the steps worth sharing out built — here and on helper threads — and
    /// offered to the build of the whole list, which follows (ScanRuler's `prebuild`). Nothing
    /// when there is nothing worth sharing.
    fn prebuild(&self, part: &Part, steps: &[&Feature], keys: &[StepKey], names: &Arc<HashMap<FeatureId, String>>, control: &BuildControl) -> Vec<Kept> {
        if self.helpers == 0 {
            return Vec::new();
        }
        let changed: Vec<bool> = keys.iter().map(|k| !self.knows(*k)).collect();
        let grouped = super::groups::group_steps(part, steps, &changed);
        struct Work {
            steps: Vec<usize>,
            cost: f64,
            fresh: bool,
        }
        let work: Vec<Work> = grouped
            .groups
            .iter()
            .map(|g| Work {
                cost: g.steps.iter().filter(|i| grouped.dirty[**i]).map(|i| estimate(steps[*i], self.played_ms(keys[*i]))).sum(),
                fresh: g.steps.iter().all(|i| !self.knows(keys[*i])),
                steps: g.steps.clone(),
            })
            .filter(|w| w.cost > 0.0)
            .collect();
        let mut movable: Vec<&Work> = work.iter().collect();
        movable.sort_by(|a, b| b.cost.total_cmp(&a.cost));
        if movable.len() < 2 || movable[1].cost < self.share_min_ms {
            return Vec::new();
        }
        // New work to the least loaded, the largest first; a group of little work, or one built
        // here before, stays here.
        let mut load = vec![0.0f64; self.helpers + 1];
        let mut to: Vec<(usize, usize)> = Vec::new();
        let mut order: Vec<usize> = (0..work.len()).collect();
        order.sort_by(|a, b| work[*b].cost.total_cmp(&work[*a].cost));
        for w in order {
            let at = if work[w].fresh && work[w].cost >= self.share_min_ms {
                let mut best = 0;
                for (i, l) in load.iter().enumerate() {
                    if *l < load[best] {
                        best = i;
                    }
                }
                best
            } else {
                0
            };
            load[at] += work[w].cost;
            to.push((w, at));
        }
        if !to.iter().any(|(_, at)| *at > 0) {
            return Vec::new();
        }
        let share = |kernel_no: usize| -> Vec<Vec<usize>> {
            let mut groups: Vec<Vec<usize>> = to.iter().filter(|(_, at)| *at == kernel_no).map(|(w, _)| work[*w].steps.clone()).collect();
            if kernel_no == 0 {
                // This thread's own share, as one part.
                let mut mine: Vec<usize> = groups.concat();
                mine.sort_unstable();
                groups = if mine.is_empty() { Vec::new() } else { vec![mine] };
            }
            groups
        };
        let part_of = |idx: &[usize], kernel_no: usize| -> Option<Kept> {
            let s: Vec<&Feature> = idx.iter().map(|i| steps[*i]).collect();
            let k: Vec<StepKey> = idx.iter().map(|i| keys[*i]).collect();
            self.build_part(part, &s, &k, names, control, kernel_no)
        };
        let mut offered: Vec<Kept> = Vec::new();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for h in 1..=self.helpers {
                let groups = share(h);
                if groups.is_empty() {
                    continue;
                }
                let part_of = &part_of;
                let spawned = std::thread::Builder::new().name(format!("reverse build helper {h}")).spawn_scoped(scope, move || groups.iter().filter_map(|g| part_of(g, h)).collect::<Vec<Kept>>());
                // A helper that does not start leaves its groups to the build of the whole list,
                // which plays what it could not take over.
                if let Ok(handle) = spawned {
                    handles.push(handle);
                }
            }
            for g in share(0) {
                if let Some(k) = part_of(&g, 0) {
                    offered.push(k);
                }
            }
            for handle in handles {
                // A helper that failed leaves its groups to the build of the whole list.
                if let Ok(parts) = handle.join() {
                    offered.extend(parts);
                }
            }
        });
        offered
    }

    /// The build standing made so: its states held, and put in the tree when it has keys; the
    /// one before let go of where nothing else holds it; the tree brought within its budget
    /// (ScanRuler's `stand`).
    fn stand(&mut self, build: Kept) {
        let before = self.standing.replace(build.clone());
        self.hold(&build.states);
        let mut node = ROOT;
        for (i, key) in build.keys.iter().enumerate() {
            let next = match self.node(node).and_then(|n| n.children.get(key)).copied() {
                Some(n) => n,
                None => {
                    let fresh = Node { key: *key, parent: node, children: HashMap::new(), state: None, reads: None, used: 0 };
                    let n = match self.free.pop() {
                        Some(slot) => {
                            self.nodes[slot] = Some(fresh);
                            slot
                        }
                        None => {
                            self.nodes.push(Some(fresh));
                            self.nodes.len() - 1
                        }
                    };
                    if let Some(p) = self.node_mut(node) {
                        p.children.insert(*key, n);
                    }
                    self.by_key.entry(*key).or_default().push(n);
                    n
                }
            };
            let same = self.node(next).and_then(|x| x.state.as_ref()).is_some_and(|s| Arc::ptr_eq(s, &build.states[i]));
            if !same {
                let old = self.node_mut(next).and_then(|x| x.state.take());
                if let Some(old) = old {
                    self.let_go(&[old]);
                }
                let state = build.states[i].clone();
                self.hold(std::slice::from_ref(&state));
                if let Some(x) = self.node_mut(next) {
                    x.state = Some(state);
                    x.reads = build.reads.get(i).cloned();
                }
            }
            let t = self.tick();
            if let Some(x) = self.node_mut(next) {
                x.used = t;
            }
            node = next;
        }
        if let Some(b) = before {
            self.let_go(&b.states);
        }
        self.trim();
    }

    /// States held once more — and the solids in them, each counted once for the bytes held the
    /// first time it is held.
    fn hold(&mut self, states: &[Arc<Snapshot>]) {
        for s in states {
            let key = Arc::as_ptr(s) as usize;
            let n = self.holders.entry(key).or_insert(0);
            *n += 1;
            if *n > 1 {
                continue;
            }
            let mut shapes = HashMap::new();
            s.shapes(&mut shapes);
            for (id, shape) in shapes {
                let slot = self.shape_holders.entry(id).or_insert((0, 0));
                if slot.0 == 0 {
                    slot.1 = shape.retained_bytes();
                    self.held_bytes += slot.1;
                }
                slot.0 += 1;
            }
        }
    }

    /// States held once less: a state nothing holds any more lets go of its solids, and a solid
    /// no state holds any more is no longer counted.
    fn let_go(&mut self, states: &[Arc<Snapshot>]) {
        for s in states {
            let key = Arc::as_ptr(s) as usize;
            let Some(n) = self.holders.get_mut(&key) else { continue };
            *n -= 1;
            if *n > 0 {
                continue;
            }
            self.holders.remove(&key);
            let mut shapes = HashMap::new();
            s.shapes(&mut shapes);
            for id in shapes.keys() {
                let Some(slot) = self.shape_holders.get_mut(id) else { continue };
                slot.0 -= 1;
                if slot.0 == 0 {
                    self.held_bytes = self.held_bytes.saturating_sub(slot.1);
                    self.shape_holders.remove(id);
                }
            }
        }
    }

    fn drop_state(&mut self, n: usize) {
        let old = self.node_mut(n).and_then(|x| {
            x.reads = None;
            x.state.take()
        });
        if let Some(old) = old {
            self.let_go(&[old]);
        }
    }

    /// The tree within its budget: the far ends of its branches let go — a state none of whose
    /// node's children holds one, but never one of the build standing — the oldest used first,
    /// until the solids held come within it. A node with nothing under it and no state is taken
    /// out of the tree (ScanRuler's `trim`).
    fn trim(&mut self) {
        if self.held_bytes <= self.budget {
            return;
        }
        let standing: HashSet<usize> = self.standing.as_ref().map_or(HashSet::new(), |s| s.states.iter().map(|x| Arc::as_ptr(x) as usize).collect());
        let is_end = |e: &Engine, n: usize| -> bool {
            let Some(x) = e.node(n) else { return false };
            let Some(s) = &x.state else { return false };
            !standing.contains(&(Arc::as_ptr(s) as usize)) && !x.children.values().any(|c| e.node(*c).is_some_and(|y| y.state.is_some()))
        };
        let mut ends: Vec<usize> = (1..self.nodes.len()).filter(|n| is_end(self, *n)).collect();
        // The oldest used last, to be popped first.
        ends.sort_by_key(|n| std::cmp::Reverse(self.node(*n).map_or(0, |x| x.used)));
        while self.held_bytes > self.budget {
            let Some(n) = ends.pop() else { break };
            self.drop_state(n);
            let parent = self.node(n).map_or(ROOT, |x| x.parent);
            // Nodes with nothing under them and no state go.
            let mut gone = n;
            while gone != ROOT && self.node(gone).is_some_and(|x| x.state.is_none() && x.children.is_empty()) {
                let Some(x) = self.nodes[gone].take() else { break };
                self.free.push(gone);
                if let Some(p) = self.node_mut(x.parent) {
                    p.children.remove(&x.key);
                }
                if let Some(list) = self.by_key.get_mut(&x.key) {
                    list.retain(|i| *i != gone);
                    if list.is_empty() {
                        self.by_key.remove(&x.key);
                    }
                }
                gone = x.parent;
            }
            if parent != ROOT && is_end(self, parent) && !ends.contains(&parent) {
                let used = self.node(parent).map_or(0, |x| x.used);
                let mut at = ends.len();
                while at > 0 && self.node(ends[at - 1]).map_or(0, |x| x.used) < used {
                    at -= 1;
                }
                ends.insert(at, parent);
            }
        }
    }
}

fn clone_said(said: &Built) -> Built {
    let mut out = Built::default();
    absorb(&mut out, said);
    out
}

/// What a step of no known time is taken to cost, ms (ScanRuler's `estimate`).
fn estimate(f: &Feature, known: Option<f64>) -> f64 {
    if let Some(ms) = known {
        return ms;
    }
    let heavy = matches!(
        f.kind,
        FeatureKind::Boolean(_)
            | FeatureKind::Pattern(_)
            | FeatureKind::Thread(_)
            | FeatureKind::Fillet(_)
            | FeatureKind::Chamfer(_)
            | FeatureKind::Shell(_)
            | FeatureKind::Split(_)
            | FeatureKind::Trim(_)
            | FeatureKind::Loft(_)
            | FeatureKind::Sweep(_)
            | FeatureKind::Import(_)
            | FeatureKind::Draft(_)
            | FeatureKind::OffsetBody(_)
            | FeatureKind::Thicken(_)
            | FeatureKind::Mirror(_)
            | FeatureKind::MoveFace(_)
            | FeatureKind::DeleteFace(_)
            | FeatureKind::Surface(_)
    );
    if heavy {
        return HEAVY_MS;
    }
    let op = match &f.kind {
        FeatureKind::Extrude(x) => Some(x.op),
        FeatureKind::Revolve(x) => Some(x.op),
        FeatureKind::Extract(x) => Some(x.op),
        _ => None,
    };
    if op.is_some_and(|o| o != BodyOp::New) {
        return HEAVY_MS;
    }
    if let FeatureKind::Revolve(r) = &f.kind {
        if r.pitch.is_some_and(|p| p != 0.0) {
            return HEAVY_MS;
        }
    }
    LIGHT_MS
}

/// A build's results as they go out: a round or a chamfer that built in the build before and
/// fails now says which features before it changed since — by the keys, so what no change
/// reached is not named (ScanRuler's `withChangesSaid`).
fn with_changes_said(built: &mut Built, steps: &[&Feature], keys: &[StepKey], was: Option<&Kept>, names: &HashMap<FeatureId, String>) {
    let Some(was) = was.filter(|w| !w.keys.is_empty() && !keys.is_empty()) else { return };
    let known: HashSet<StepKey> = was.keys.iter().copied().collect();
    for (i, f) in steps.iter().enumerate() {
        if !matches!(f.kind, FeatureKind::Fillet(_) | FeatureKind::Chamfer(_)) || !was.ok.contains(&f.id) {
            continue;
        }
        let Some(message) = built.errors.get_mut(&f.id) else { continue };
        let changed: Vec<String> = steps[..i].iter().zip(keys).filter(|(_, k)| !known.contains(k)).map(|(s, _)| names.get(&s.id).cloned().unwrap_or_else(|| format!("feature {}", s.id))).collect();
        if changed.is_empty() {
            continue;
        }
        let named = changed[..changed.len().min(CHANGES_NAMED)].join(", ");
        let more = if changed.len() > CHANGES_NAMED { format!(" and {} more", changed.len() - CHANGES_NAMED) } else { String::new() };
        *message = format!("{message} It built in the last build; changed before it since: {named}{more}.");
    }
}

// `Builder::family_of` and `Builder::note_pieces` (ScanRuler's `familyOf` and `pieces`) are in
// the BOOL block of build/mod.rs, noting what they read for the engine.

#[cfg(test)]
mod tests;
