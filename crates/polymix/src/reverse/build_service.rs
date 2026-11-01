// SPDX-License-Identifier: AGPL-3.0-only
//! The build thread: the CAD kernel's side of the workspace. Ported from ScanRuler
//! (`plugins/reverse/core/reQueue.ts`, `reClient.ts`, `reService.ts`, `app/reSession.ts`
//! build, stop, busy, `app/reBuildStop.ts`, `ui/BuildChip.tsx`, and the kernel status of
//! `ui/ReversePanel.tsx`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! One thread holds the build engine — the builds it keeps, which the next build takes up
//! from — and takes the builds asked of it one at a time: a build asked for while one runs
//! waits, and only the newest of those waiting is built; the ones it overtook are answered as
//! skipped and never played. A build that does not finish is stopped: a kernel call cannot be
//! interrupted, so the thread is let go — it stops at its next step, or once the call it is in
//! returns, and its answer is nobody's — and a fresh one started in its place (the kernel
//! "started again", the builds it kept gone with it); the feature whose step was in play is
//! suppressed, its picks and numbers kept, and the part built without it.
//!
//! The bodies are meshed for the view here too, each once: a body a build left as it was is
//! the very view the build before handed back.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use mm_core::Mesh;
use mm_reverse::build::{body_mesh, volume, BuildAsk, BuildControl, BuildScan, BuildSteps, Built, BuiltBody, Engine, InPlay, NetworkContext, NetworkScope, KERNEL_VERSION};
use mm_reverse::model::{BodyId, FeatureId, Part};
use mm_reverse::topo::Topology;

use super::BodyView;

/// A build is shown on the stage only after this long, so a quick one never flashes the chip
/// (ScanRuler's `SHOW_AFTER_MS`).
pub(crate) const SHOW_AFTER: Duration = Duration::from_millis(300);

/// How long a step has been in play before the chip says how to stop it (ScanRuler's
/// `STOP_SAID_MS`).
pub(crate) const STOP_SAID: Duration = Duration::from_secs(30);

/// A build asked of the thread: the part as the view shows it, the feature a box is open on.
pub(crate) struct BuildRequest {
    pub generation: u64,
    pub part: Part,
    pub preview: Option<FeatureId>,
    pub flip: bool,
    /// ---- NET ---- What the part's patch networks read and say (the scan, the open box's
    /// feature, the outcomes' sink), entered on the build thread for the build; `None` for a
    /// part with no network.
    pub net: Option<NetworkContext>,
    /// ---- SURF ---- The scan in the part's frame, as the surface features read it (a split, a
    /// trim, a fitted surface, a shadow); `None` with no scan shown.
    pub scan: Option<Arc<BuildScan>>,
}

/// The result of a build, meshed, from the build thread.
pub(crate) struct BuildDone {
    pub generation: u64,
    pub built: Built,
    pub bodies: Vec<BodyView>,
    /// With a feature box open, the bodies as they stood before its feature.
    pub before: Option<Vec<BodyView>>,
    /// What the build did with its steps.
    pub steps: BuildSteps,
}

/// What the build thread says.
pub(crate) enum Answer {
    /// It is up and can build.
    Up,
    Done(Box<BuildDone>),
    /// A build overtaken by a newer one, never played.
    Skipped(u64),
    /// A build that broke off on an internal error; the thread goes on with a fresh engine.
    Failed(u64, String),
}

/// Where the CAD kernel stands, as the Model group says it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum KernelStatus {
    Idle,
    Loading,
    Ready,
    Unavailable(String),
}

/// The handle on one build thread.
struct Service {
    tx: Sender<BuildRequest>,
    rx: Receiver<Answer>,
    control: BuildControl,
    /// The generation the thread is building, 0 for none.
    at: Arc<AtomicU64>,
}

impl Service {
    fn start(slow: Duration) -> Result<Service, String> {
        let (tx, requests) = channel::<BuildRequest>();
        let (answers, rx) = channel::<Answer>();
        let control = BuildControl::new();
        let at = Arc::new(AtomicU64::new(0));
        let (c, a) = (control.clone(), at.clone());
        std::thread::Builder::new().name("reverse build".into()).spawn(move || serve(requests, answers, c, a, slow)).map_err(|e| format!("The CAD kernel could not be started: {e}"))?;
        Ok(Service { tx, rx, control, at })
    }
}

/// The build thread: every build asked of it, the newest of those waiting first and the rest
/// skipped (ScanRuler's `requestQueue`), until it is let go.
fn serve(requests: Receiver<BuildRequest>, answers: Sender<Answer>, control: BuildControl, at: Arc<AtomicU64>, slow: Duration) {
    let mut engine = Engine::new();
    let mut views = ViewCache::default();
    if answers.send(Answer::Up).is_err() {
        return;
    }
    while let Ok(mut req) = requests.recv() {
        // A build followed straight by another is overtaken: answered as skipped.
        loop {
            match requests.try_recv() {
                Ok(next) => {
                    if answers.send(Answer::Skipped(req.generation)).is_err() {
                        return;
                    }
                    req = next;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if control.stopped() {
            return;
        }
        at.store(req.generation, Ordering::SeqCst);
        control.begin();
        if !slow.is_zero() {
            control.slow_steps(slow);
        }
        let generation = req.generation;
        // ---- SURF ---- the scan the build reads; another than the kept builds read lets them go.
        engine.set_scan(req.scan.take());
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let upto = req.part.rolled_to.unwrap_or(req.part.features.len()).min(req.part.features.len());
            // ---- NET ---- The networks read the scan through a scope on this thread. The
            // engine's step keys hold neither the scan nor which network is the box's preview,
            // its helper threads have no scope, and a step taken over says no outcome: a part
            // with a network is played whole, here, nothing kept.
            let networked = req.net.is_some();
            let _net = req.net.take().map(NetworkScope::enter);
            let ask = BuildAsk { upto, before: req.preview, reuse: !networked, parallel: !networked };
            let out = engine.build(&req.part, &ask, &control)?;
            let bodies = views.views(&out.built.bodies, req.flip);
            let before = out.before.map(|b| views.views(&b.bodies, req.flip));
            views.keep(&bodies, before.as_deref());
            Some(BuildDone { generation, built: out.built, bodies, before, steps: out.steps })
        }));
        at.store(0, Ordering::SeqCst);
        let answer = match run {
            Ok(Some(done)) => Answer::Done(Box::new(done)),
            // Stopped: let go, its answer nobody's.
            Ok(None) => return,
            Err(p) => {
                let why = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
                engine = Engine::new();
                views = ViewCache::default();
                Answer::Failed(generation, format!("The build stopped on an internal error ({why}); the CAD kernel was started again."))
            }
        };
        if answers.send(answer).is_err() {
            return;
        }
    }
}

/// The views of the bodies the last build handed back, by the reading of each — a body a build
/// left as it was is its view again, not meshed anew.
#[derive(Default)]
struct ViewCache {
    held: Vec<(Arc<Topology>, BodyId, bool, BodyView)>,
}

impl ViewCache {
    fn views(&mut self, bodies: &[BuiltBody], flip: bool) -> Vec<BodyView> {
        bodies
            .iter()
            .filter_map(|b| match self.held.iter().find(|(t, id, f, _)| Arc::ptr_eq(t, &b.topology) && *id == b.id && *f == flip) {
                Some((_, _, _, v)) => Some(BodyView { name: b.name.clone(), ..v.clone() }),
                None => {
                    let v = view_of(b, flip)?;
                    self.held.push((b.topology.clone(), b.id, flip, v.clone()));
                    Some(v)
                }
            })
            .collect()
    }

    /// Only what the last build handed back is kept.
    fn keep(&mut self, bodies: &[BodyView], before: Option<&[BodyView]>) {
        let wanted: Vec<&Arc<Topology>> = bodies.iter().chain(before.unwrap_or(&[])).map(|v| &v.topology).collect();
        self.held.retain(|(t, _, _, _)| wanted.iter().any(|w| Arc::ptr_eq(w, t)));
    }
}

/// A body meshed for the view, in scene coordinates.
fn view_of(b: &BuiltBody, flip: bool) -> Option<BodyView> {
    let to_scene = |p: DVec3| if flip { crate::app::flip_zy_in(p) } else { p };
    let m = body_mesh(&b.body).ok()?;
    let scene: Vec<DVec3> = m.positions.iter().map(|p| to_scene(*p)).collect();
    let (mesh, _) = Mesh::from_triangles(&scene, &m.triangles, Some(&m.tri_face.iter().map(|f| *f as i32).collect::<Vec<_>>()));
    let edges = m.edges.iter().map(|e| e.iter().map(|p| to_scene(*p)).collect()).collect();
    Some(BodyView { id: b.id, name: b.name.clone(), mesh: Arc::new(mesh), edges, volume: volume(&b.body).unwrap_or(0.0), part_mesh: Arc::new(m), topology: b.topology.clone() })
}

/// What a build stop did (ScanRuler's `Stopped`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Stopped {
    /// The feature whose step was in play, suppressed — `None` when no step was, or when the
    /// build was of the list before a change since.
    pub feature: Option<(FeatureId, String)>,
    /// The feature in play was an open box's, not yet made: the box is put away.
    pub put_away: bool,
    pub message: String,
}

/// The builds of the workspace: the thread, what it was asked and when, what the kernel is at.
pub(crate) struct Builds {
    service: Option<Service>,
    pub kernel: KernelStatus,
    /// The newest build asked for and not landed: its generation and when it was asked.
    pub pending: Option<(u64, Instant)>,
    /// What the last build that landed did with its steps.
    pub last_steps: BuildSteps,
    /// A slowness put on every step of the builds after it — for scripted tests of the stop.
    slow: Duration,
}

impl Default for Builds {
    fn default() -> Self {
        Builds { service: None, kernel: KernelStatus::Idle, pending: None, last_steps: BuildSteps::default(), slow: Duration::ZERO }
    }
}

impl Builds {
    /// The thread brought up, once — what the first build does.
    fn up(&mut self) -> Option<&Service> {
        if self.service.is_none() && !matches!(self.kernel, KernelStatus::Unavailable(_)) {
            match Service::start(self.slow) {
                Ok(s) => {
                    self.service = Some(s);
                    self.kernel = KernelStatus::Loading;
                }
                Err(e) => self.kernel = KernelStatus::Unavailable(e),
            }
        }
        self.service.as_ref()
    }

    /// Ask for a build; the newest asked is the one that counts.
    pub fn ask(&mut self, req: BuildRequest) {
        let generation = req.generation;
        let sent = self.up().is_some_and(|s| s.tx.send(req).is_ok());
        if sent {
            self.pending = Some((generation, Instant::now()));
        } else {
            self.service = None;
            self.pending = None;
            if !matches!(self.kernel, KernelStatus::Unavailable(_)) {
                self.kernel = KernelStatus::Unavailable("The CAD kernel stopped and could not take the build.".into());
            }
        }
    }

    /// What the thread has said since last asked; `Err` when it is gone.
    pub fn poll(&mut self) -> Result<Vec<Answer>, String> {
        let Some(s) = &self.service else { return Ok(Vec::new()) };
        let mut out = Vec::new();
        loop {
            match s.rx.try_recv() {
                Ok(Answer::Up) => {
                    self.kernel = KernelStatus::Ready;
                    out.push(Answer::Up);
                }
                Ok(a) => out.push(a),
                Err(TryRecvError::Empty) => return Ok(out),
                Err(TryRecvError::Disconnected) => {
                    self.service = None;
                    self.pending = None;
                    let why = "The build stopped on an internal error.".to_string();
                    self.kernel = KernelStatus::Unavailable(why.clone());
                    return Err(why);
                }
            }
        }
    }

    /// Whether the newest build asked for has not landed.
    pub fn building(&self) -> bool {
        self.pending.is_some()
    }

    /// How long the newest build has been waited for.
    pub fn waited(&self) -> Option<Duration> {
        self.pending.map(|(_, at)| at.elapsed())
    }

    /// The step the thread is at the longest, and the generation of the build it is in.
    pub fn in_play(&self) -> (Option<InPlay>, u64, bool) {
        match &self.service {
            Some(s) => (s.control.in_play(), s.at.load(Ordering::SeqCst), s.control.played()),
            None => (None, 0, false),
        }
    }

    /// The kernel let go and started again: the thread under way stopped and dropped, a fresh
    /// one in its place, nothing pending.
    pub fn restart(&mut self) {
        if let Some(s) = self.service.take() {
            s.control.stop();
        }
        self.pending = None;
        self.kernel = KernelStatus::Idle;
        self.up();
    }

    /// Every step of the builds after this made slower — for scripted tests of the stop.
    pub fn set_slow(&mut self, slow: Duration) {
        self.slow = slow;
        self.restart();
    }

    /// Whether the builds are made slow on purpose.
    pub fn slowed(&self) -> bool {
        !self.slow.is_zero()
    }

    /// The Model group's word on the kernel.
    pub fn kernel_words(&self) -> String {
        match &self.kernel {
            KernelStatus::Idle => "CAD kernel: waiting.".into(),
            KernelStatus::Loading => "Loading the CAD kernel…".into(),
            KernelStatus::Ready => format!("CAD kernel ready — {KERNEL_VERSION}."),
            KernelStatus::Unavailable(why) => format!("⚠ {why}"),
        }
    }
}

/// The amber of a warning, as the panel writes one.
const WARN: egui::Color32 = egui::Color32::from_rgb(226, 164, 59);

/// Under a feature's row, indented as the row's hint: why it failed, and what it did its own
/// way (ScanRuler's `rowerror` hints under an element row).
pub(crate) fn row_notes(ui: &mut egui::Ui, built: &Built, id: FeatureId) {
    let mut lines: Vec<String> = built.errors.get(&id).map(|e| vec![format!("⚠ {e}")]).unwrap_or_default();
    lines.extend(built.warnings.get(&id).into_iter().flatten().cloned());
    for line in lines {
        ui.horizontal(|ui| {
            ui.add_space(17.0);
            ui.add(egui::Label::new(egui::RichText::new(line).small().color(WARN)).wrap());
        });
    }
}

/// What the status says of a build that landed: the features that built in the build before
/// and fail now, the first by name with why — `None` when none newly fails.
pub(crate) fn newly_failed(part: &Part, was: &Built, now: &Built) -> Option<String> {
    let mut failed: Vec<(usize, FeatureId)> = now.errors.keys().filter(|id| !was.errors.contains_key(id)).filter_map(|id| part.index_of(*id).map(|i| (i, *id))).collect();
    failed.sort_unstable();
    let (_, first) = *failed.first()?;
    let name = part.feature(first).map_or_else(|| format!("Feature {first}"), |f| f.name.clone());
    let more = if failed.len() > 1 { format!(" — and {} more failed", failed.len() - 1) } else { String::new() };
    Some(format!("⚠ {name}: {}{more}", now.errors.get(&first).cloned().unwrap_or_default()))
}

/// A step in play said: its feature's name and its place in the list — "Flange plate (step
/// 12)" (ScanRuler's `stepSaid`; the SolidCraft kernel tells no call names).
pub(crate) fn step_said(part: &Part, id: FeatureId) -> String {
    let at = part.index_of(id);
    let name = part.feature(id).map_or_else(|| format!("Feature {id}"), |f| f.name.clone());
    match at {
        Some(i) => format!("{name} (step {})", i + 1),
        None => name,
    }
}

/// What a stop says (ScanRuler's session `stop`, asked): which step was in play and for how
/// long, and what was done — the feature suppressed, or nothing to suppress.
/// `part` is the part as the view shows it — with an open box's feature in it — and `saved`
/// tells a feature of the part as kept from one a box has not made yet.
pub(crate) fn stop_words(part: &Part, in_play: Option<InPlay>, was_latest: bool, played: bool, build_time: Duration, saved: impl Fn(FeatureId) -> bool) -> Stopped {
    let secs = |d: Duration| js_secs(d.as_secs_f64());
    match in_play {
        None => {
            let place = if played { "after its last step" } else { "before its first step" };
            Stopped { feature: None, put_away: false, message: format!("The build was stopped after {} s, {place}: no step was in play, so nothing was suppressed. The CAD kernel was started again and the part is building again.", secs(build_time)) }
        }
        Some(p) => {
            let said = step_said(part, p.feature);
            let late = format!("was stopped after {} s", secs(p.for_how_long));
            match part.feature(p.feature).filter(|_| was_latest) {
                None => Stopped { feature: None, put_away: false, message: format!("{said} {late}: it was a build of the feature list before a change since. The CAD kernel was started again and the part built as the list stands — nothing was suppressed.") },
                Some(f) if !saved(f.id) => Stopped { feature: None, put_away: true, message: format!("{said} {late}: the CAD kernel was stopped and started again, and its box put away — {} was not made.", f.name) },
                Some(f) => Stopped { feature: Some((f.id, f.name.clone())), put_away: false, message: format!("{said} {late}: the CAD kernel was stopped and started again, and the part built without it — {} is suppressed. Change it before you build it again.", f.name) },
            }
        }
    }
}

/// A feature box's join or cut lets go of the one body it was saved with: saved again from its
/// box, it joins or cuts every body it meets (ScanRuler's `withoutTarget`, which the Extrude
/// and Revolve boxes save by).
pub(crate) fn without_target(kind: &mut mm_reverse::model::FeatureKind) {
    match kind {
        mm_reverse::model::FeatureKind::Extrude(x) => x.target = None,
        mm_reverse::model::FeatureKind::Revolve(x) => x.target = None,
        _ => {}
    }
}

fn js_secs(s: f64) -> String {
    mm_reverse::js_number::js_string(mm_reverse::js_number::js_round(s))
}

/// The build chip at the top of the stage while the kernel loads or builds — only after
/// `SHOW_AFTER`, so a quick build never flashes it (ScanRuler's `BuildChip`), with a Stop key
/// and, once a step has been in play long, what it is at. True when Stop was pressed.
pub(crate) fn build_chip(ctx: &egui::Context, stage: egui::Rect, builds: &Builds, part: &Part, has_features: bool) -> bool {
    let label = if builds.kernel == KernelStatus::Loading && has_features {
        "Loading the CAD kernel…"
    } else if builds.building() {
        "Building the part…"
    } else {
        return false;
    };
    if builds.waited().is_some_and(|w| w < SHOW_AFTER) {
        ctx.request_repaint_after(SHOW_AFTER);
        return false;
    }
    let (in_play, _, _) = builds.in_play();
    let mut stop = false;
    egui::Area::new(egui::Id::NULL.with("re_build_chip"))
        .order(egui::Order::Foreground)
        .pivot(egui::Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(stage.center().x, stage.top() + 10.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).corner_radius(10.0).show(ui, |ui| {
                ui.horizontal(|ui| {
                    polymix_workspace::spinner(ui);
                    ui.label(label);
                    if builds.building()
                        && ui
                            .small_button("Stop")
                            .on_hover_text("Stop the build: the CAD kernel is started again, the feature whose step is in play suppressed (its picks and numbers kept), and the part built without it")
                            .clicked()
                    {
                        stop = true;
                    }
                });
                if let Some(p) = in_play.filter(|p| p.for_how_long >= STOP_SAID) {
                    ui.weak(format!("{}, {} s so far — Stop builds the part without it.", step_said(part, p.feature), js_secs(p.for_how_long.as_secs_f64())));
                }
            });
        });
    ctx.request_repaint_after(Duration::from_millis(250));
    stop
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_reverse::frame::WorldAxis;
    use mm_reverse::model::{BodyOp, ExtrudeExtent, ExtrudeFeature, FeatureKind, SketchFeature};
    use mm_reverse::regions::find_regions;
    use mm_reverse::sketch::Sketch;

    fn plate(width: f64) -> Part {
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_rectangle(glam::DVec2::ZERO, glam::DVec2::new(width, 20.0));
        let regions = mm_reverse::build::default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 5.0 }, BodyOp::New)));
        part
    }

    fn wait(builds: &mut Builds, until: impl Fn(&[Answer]) -> bool) -> Vec<Answer> {
        let mut all = Vec::new();
        for _ in 0..2000 {
            match builds.poll() {
                Ok(a) => all.extend(a),
                Err(e) => panic!("{e}"),
            }
            if until(&all) {
                return all;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("no answer");
    }

    #[test]
    fn reverse_builds_only_the_newest_of_the_builds_waiting_and_skips_the_rest() {
        let mut builds = Builds::default();
        // Slow steps hold the first build while the next three are asked for.
        builds.set_slow(Duration::from_millis(150));
        builds.ask(BuildRequest { generation: 1, part: plate(40.0), preview: None, flip: false, net: None, scan: None });
        for _ in 0..400 {
            let _ = builds.poll();
            if builds.in_play().0.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        for (g, w) in [(2, 41.0), (3, 42.0), (4, 43.0)] {
            builds.ask(BuildRequest { generation: g, part: plate(w), preview: None, flip: false, net: None, scan: None });
        }
        let answers = wait(&mut builds, |a| a.iter().any(|x| matches!(x, Answer::Done(d) if d.generation == 4)));
        let done: Vec<u64> = answers.iter().filter_map(|a| if let Answer::Done(d) = a { Some(d.generation) } else { None }).collect();
        let skipped: Vec<u64> = answers.iter().filter_map(|a| if let Answer::Skipped(g) = a { Some(*g) } else { None }).collect();
        assert_eq!(done, vec![1, 4]);
        assert_eq!(skipped, vec![2, 3]);
        assert_eq!(builds.kernel, KernelStatus::Ready);
        let last = answers.iter().rev().find_map(|a| if let Answer::Done(d) = a { Some(d) } else { None }).unwrap_or_else(|| panic!("no build"));
        assert_eq!(last.bodies.len(), 1);
        assert!((last.bodies[0].volume - 43.0 * 20.0 * 5.0).abs() < 1e-3);
    }

    #[test]
    fn reverse_a_body_a_build_left_as_it_was_keeps_its_view() {
        let mut builds = Builds::default();
        let mut part = plate(40.0);
        builds.ask(BuildRequest { generation: 1, part: part.clone(), preview: None, flip: true, net: None, scan: None });
        let first = wait(&mut builds, |a| a.iter().any(|x| matches!(x, Answer::Done(_))));
        let mesh = first.iter().find_map(|a| if let Answer::Done(d) = a { Some(d.bodies[0].mesh.clone()) } else { None }).unwrap_or_else(|| panic!("no build"));
        // A sketch hidden: no step changes, the body is the same view.
        if let Some(f) = part.feature_mut(1) {
            f.flags.visible = false;
        }
        builds.ask(BuildRequest { generation: 2, part, preview: None, flip: true, net: None, scan: None });
        let second = wait(&mut builds, |a| a.iter().any(|x| matches!(x, Answer::Done(_))));
        let again = second.iter().find_map(|a| if let Answer::Done(d) = a { Some((d.bodies[0].mesh.clone(), d.steps.recomputed)) } else { None }).unwrap_or_else(|| panic!("no build"));
        assert!(Arc::ptr_eq(&mesh, &again.0));
        assert_eq!(again.1, 0, "a sketch hidden builds nothing again");
    }

    #[test]
    fn reverse_a_stop_says_what_it_stopped_and_suppresses_the_step_in_play() {
        let part = plate(40.0);
        let p = InPlay { feature: 2, for_how_long: Duration::from_secs(12) };
        let s = stop_words(&part, Some(p), true, true, Duration::from_secs(13), |_| true);
        assert_eq!(s.feature, Some((2, "Extrude 1".to_string())));
        assert_eq!(s.message, "Extrude 1 (step 2) was stopped after 12 s: the CAD kernel was stopped and started again, and the part built without it — Extrude 1 is suppressed. Change it before you build it again.");
        let s = stop_words(&part, Some(p), false, true, Duration::from_secs(13), |_| true);
        assert_eq!(s.feature, None);
        assert!(s.message.contains("it was a build of the feature list before a change since"));
        // A box's feature, not made yet: the box put away.
        let s = stop_words(&part, Some(p), true, true, Duration::from_secs(13), |_| false);
        assert!(s.put_away && s.feature.is_none());
        assert!(s.message.ends_with("and its box put away — Extrude 1 was not made."), "{}", s.message);
        let s = stop_words(&part, None, false, false, Duration::from_secs(3), |_| true);
        assert_eq!(s.message, "The build was stopped after 3 s, before its first step: no step was in play, so nothing was suppressed. The CAD kernel was started again and the part is building again.");
    }

    #[test]
    fn reverse_stops_a_build_under_way_and_starts_the_kernel_again() {
        let mut builds = Builds::default();
        builds.set_slow(Duration::from_millis(400));
        builds.ask(BuildRequest { generation: 1, part: plate(40.0), preview: None, flip: false, net: None, scan: None });
        // Until a step is in play.
        for _ in 0..400 {
            let _ = builds.poll();
            if builds.in_play().0.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let (p, at, _) = builds.in_play();
        assert!(p.is_some());
        assert_eq!(at, 1);
        builds.slow = Duration::ZERO;
        builds.restart();
        assert!(!builds.building());
        builds.ask(BuildRequest { generation: 2, part: plate(30.0), preview: None, flip: false, net: None, scan: None });
        let answers = wait(&mut builds, |a| a.iter().any(|x| matches!(x, Answer::Done(_))));
        let done = answers.iter().find_map(|a| if let Answer::Done(d) = a { Some(d) } else { None }).unwrap_or_else(|| panic!("no build"));
        assert_eq!(done.generation, 2);
        assert_eq!(done.steps.recomputed, 2, "a kernel started again keeps nothing");
    }
}
