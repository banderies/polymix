// SPDX-License-Identifier: AGPL-3.0-only
//! The Mirror feature's box: the body, the plane it is mirrored about — a plane from the list,
//! or a flat face of the body clicked on the part — and whether the copy joins the body. A new
//! mirror opens on the scan's own symmetry plane: one kept as a plane element before, or the
//! one mirror registration finds on the scan (`mm_reverse::symmetry_search`, on a worker thread,
//! once per scan), which the box says it was set to — or that the scan has none. Ported from
//! ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the mirror case, takeSymmetryPlane,
//! keepSymmetryPlane, `app/useReverseWorkspace.ts` the search a new mirror starts,
//! `core/symmetryPlane.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

use egui::{Pos2, RichText};
use glam::DVec3;
use mm_reverse::build::{no_plane, plane_chosen};
use mm_reverse::model::{Element, ElementGeometry, Exprs, FeatureKind, FitStats, MirrorFeature, MirrorPlane, PlaneRef};
use mm_reverse::symmetry_plane::{symmetry_plane_ref, SymmetryPlane, SYMMETRY_LABEL};
use mm_reverse::topo::{describe_face, FaceKind};

use super::bool_fields::{body_select, clear_plane_hint, field, hint, paint_plane_hint, plane_select};
use super::paint_faces;
use crate::reverse::view::Hit;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Mirror", "Mirror the body about a plane, joined with the source");
/// A box just opened: the search for the scan's symmetry plane is yet to be asked for.
const OPENED: u32 = 0x4d49_0001;
/// The box as it stays open.
const OPEN: u32 = 0x4d49_0002;
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Custom(OPENED);

/// The name a plane element kept from the search goes by — and how a symmetry plane kept before
/// is known again when the next mirror opens.
const KEPT_NAME: &str = "Symmetry plane";

pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = super::boolean::last_solid(ws).ok_or("A mirror copies a body: make a body first.")?;
    Ok(FeatureKind::Mirror(MirrorFeature { body, plane: MirrorPlane::Plane(no_plane()), join: true, exprs: Exprs::new() }))
}

/// Whether the mirror can be made: a plane chosen (ScanRuler's `featureDraftReady`).
pub(crate) fn ready(m: &MirrorFeature) -> Result<(), String> {
    match &m.plane {
        MirrorPlane::Plane(p) if !plane_chosen(p) => Err("Click a flat face of the body, or choose a plane.".into()),
        _ => Ok(()),
    }
}

// ---- the search for the scan's symmetry plane ----

/// What the box knows of the scan's symmetry (ScanRuler's `SymmetryFinding`).
#[derive(Clone)]
enum Finding {
    Idle,
    Working,
    Found(SymmetryPlane),
    /// Set to a symmetry plane kept as an element, by its name.
    Element(String),
    /// None found: the best mirror image's σ, where the search got one.
    None(Option<f64>),
}

/// The search's answer per scan, kept for the session: the scan it was for, the answer, and
/// the search under way.
#[derive(Default)]
struct Searches {
    key: u64,
    answer: Option<Result<SymmetryPlane, String>>,
    job: Option<(u64, Receiver<Result<SymmetryPlane, String>>)>,
    /// What the search under way is doing.
    saying: Arc<Mutex<String>>,
}

type Shared = Arc<Mutex<Searches>>;

fn searches(ctx: &egui::Context) -> Shared {
    let id = egui::Id::NULL.with("re_bool_symmetry_searches");
    ctx.data_mut(|d| match d.get_temp::<Shared>(id) {
        Some(s) => s,
        None => {
            let s: Shared = Arc::new(Mutex::new(Searches::default()));
            d.insert_temp(id, s.clone());
            s
        }
    })
}

fn finding_id() -> egui::Id {
    egui::Id::NULL.with("re_bool_mirror_finding")
}

/// Which scan the workspace shows: its objects, their shapes and the axes it is read in.
fn scan_key(ws: &ReverseWorkspace) -> u64 {
    ws.scan.iter().filter(|o| o.visible).fold(ws.flip as u64 + 7, |h, o| {
        let p = Arc::as_ptr(&o.mesh) as u64;
        h.wrapping_mul(1_000_003).wrapping_add(p ^ o.mesh.shape_version().rotate_left(17) ^ o.mesh.topology_version().rotate_left(31))
    })
}

/// The search on the scan as it stands, on a worker thread: every visible object of it in one
/// surface, in the part's frame.
fn start_search(ws: &mut ReverseWorkspace, s: &mut Searches, key: u64) {
    let parts = ws.scan_parts();
    let (tx, rx) = channel();
    let saying = s.saying.clone();
    std::thread::Builder::new()
        .name("reverse symmetry".into())
        .spawn(move || {
            let mut positions: Vec<DVec3> = Vec::new();
            let mut triangles: Vec<[u32; 3]> = Vec::new();
            for p in parts.iter() {
                let base = positions.len() as u32;
                positions.extend_from_slice(&p.positions);
                triangles.extend(p.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
            }
            let say = |t: &str| {
                if let Ok(mut s) = saying.lock() {
                    *s = t.to_string();
                }
            };
            let _ = tx.send(mm_reverse::symmetry_search::scan_symmetry(&positions, &triangles, &say));
        })
        .ok();
    s.job = Some((key, rx));
}

/// The search's answer landed on the box (ScanRuler's `land`): a symmetry worth defaulting to
/// sets the plane while none is chosen; otherwise the box says there is none.
fn land(m: &mut MirrorFeature, r: &Result<SymmetryPlane, String>) -> (Finding, bool) {
    match r {
        Ok(r) => match symmetry_plane_ref(r) {
            Some(plane) => {
                let unset = matches!(&m.plane, MirrorPlane::Plane(p) if !plane_chosen(p));
                if unset {
                    m.plane = MirrorPlane::Plane(plane);
                }
                (Finding::Found(*r), unset)
            }
            None => (Finding::None(r.rms.is_finite().then_some(r.rms)), false),
        },
        Err(_) => (Finding::None(None), false),
    }
}

/// The last plane element kept from the search, if any: its id and name.
fn measured_symmetry(ws: &ReverseWorkspace) -> Option<(u32, String)> {
    ws.part.elements.iter().rev().find(|e| matches!(e.geometry, ElementGeometry::Plane { .. }) && e.name.starts_with(KEPT_NAME)).map(|e| (e.id, e.name.clone()))
}

/// A new mirror opens on the scan's own symmetry plane: one kept as an element before, else
/// the search's — run once per scan; a plane chosen meanwhile stands. Returns whether the
/// mirror changed.
fn open_on_symmetry(ws: &mut ReverseWorkspace, ctx: &egui::Context, m: &mut MirrorFeature) -> (Finding, bool) {
    if !matches!(&m.plane, MirrorPlane::Plane(p) if !plane_chosen(p)) {
        return (Finding::Idle, false);
    }
    if let Some((id, name)) = measured_symmetry(ws) {
        m.plane = MirrorPlane::Plane(PlaneRef::Plane { id, offset: None, up: None });
        return (Finding::Element(name), true);
    }
    let shared = searches(ctx);
    let Ok(mut s) = shared.lock() else { return (Finding::None(None), false) };
    let key = scan_key(ws);
    if s.key == key {
        if let Some(r) = &s.answer {
            return land(m, r);
        }
    }
    if ws.scan.is_empty() {
        return (Finding::None(None), false);
    }
    if !matches!(&s.job, Some((k, _)) if *k == key) {
        start_search(ws, &mut s, key);
    }
    (Finding::Working, false)
}

/// The search under way, asked once a frame: its answer kept for the scan and landed on the box.
fn poll_search(ws: &ReverseWorkspace, ctx: &egui::Context, m: &mut MirrorFeature) -> Option<(Finding, bool)> {
    let shared = searches(ctx);
    let mut s = shared.lock().ok()?;
    let (key, rx) = s.job.as_ref()?;
    match rx.try_recv() {
        Ok(r) => {
            let key = *key;
            s.job = None;
            s.key = key;
            s.answer = Some(r.clone());
            // A scan changed since then: the plane is of another part.
            if key != scan_key(ws) {
                return Some((Finding::None(None), false));
            }
            Some(land(m, &r))
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            None
        }
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            s.job = None;
            Some((Finding::None(None), false))
        }
    }
}

/// What the search under way says it is doing.
fn search_saying(ctx: &egui::Context) -> String {
    let shared = searches(ctx);
    let s = shared.lock().ok().and_then(|s| s.saying.lock().ok().map(|t| t.clone()));
    s.unwrap_or_default()
}

/// The found plane made a plane element (ScanRuler's `keepSymmetryPlane`: Measure's
/// symmetry-plane method, the search's numbers as its own) and the mirror moved onto it.
fn keep_as_element(ws: &mut ReverseWorkspace, m: &mut MirrorFeature, r: &SymmetryPlane) -> Finding {
    let id = ws.part.elements.iter().map(|e| e.id).max().map_or(1, |i| i + 1);
    let kept = ws.part.elements.iter().filter(|e| e.name.starts_with(KEPT_NAME)).count();
    let name = if kept == 0 { KEPT_NAME.to_string() } else { format!("{KEPT_NAME} {}", kept + 1) };
    let normal = r.normal.normalize_or_zero();
    let (basis_u, basis_v) = mm_reverse::linalg::ortho_basis(normal);
    // A patch as large as the model's radius allows (ScanRuler's `planeAt`).
    let radius = ws.scan_parts().iter().flat_map(|p| p.live_vertices.iter().map(|v| p.positions[*v as usize])).fold((DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)), |(lo, hi), p| (lo.min(p), hi.max(p)));
    let fallback = if radius.0.x.is_finite() { (radius.1 - radius.0).length() / 2.0 } else { 10.0 };
    let extent = (fallback * 0.3).max(1.0);
    ws.part.elements.push(Element {
        id,
        name: name.clone(),
        geometry: ElementGeometry::Plane { center: r.point, normal, basis_u, basis_v, extent_u: extent, extent_v: extent },
        // The mirror registration's residual stands where a fit's sigma does.
        stats: Some(FitStats { rms: r.rms, sigma: r.rms, used_points: r.matched as u32, region_size: r.sampled as u32, form_error: None, residual_min: None, residual_max: None }),
    });
    m.plane = MirrorPlane::Plane(PlaneRef::Plane { id, offset: None, up: None });
    ws.status = format!("{name}: the mirror image fits the scan to σ {}, found for a Mirror in Reverse Engineer.", len(r.rms, 4));
    Finding::Element(name)
}

/// A length as the box reads it, to `decimals` places, with its unit.
fn len(mm: f64, decimals: usize) -> String {
    mm_reverse::display_units::fmt_len(mm, mm_reverse::display_units::DisplayUnits::Mm, decimals, Default::default())
}

/// A number written with thousands separators, as `toLocaleString('en-US')` writes a count.
fn count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, m: &mut MirrorFeature, pick: &mut PickWhat) -> bool {
    let ctx = ui.ctx().clone();
    let mut changed = false;
    let mut finding: Finding = ctx.data(|d| d.get_temp::<Finding>(finding_id())).unwrap_or(Finding::Idle);
    if *pick == PickWhat::Custom(OPENED) {
        *pick = PickWhat::Custom(OPEN);
        let (f, c) = open_on_symmetry(ws, &ctx, m);
        finding = f;
        changed |= c;
    }
    if matches!(finding, Finding::Working) {
        if let Some((f, c)) = poll_search(ws, &ctx, m) {
            finding = f;
            changed |= c;
        }
    }
    changed |= body_select(ui, ws, "re_mirror_body", "Body", &mut m.body);
    field(ui, "About", None, |ui| {
        let value = match &m.plane {
            MirrorPlane::Plane(p) if plane_chosen(p) => Some(p.clone()),
            _ => None,
        };
        let placeholder = match &m.plane {
            MirrorPlane::Face { face } => describe_face(face),
            _ => "Click a flat face of the body, or choose…".to_string(),
        };
        if let Some(p) = plane_select(ui, ws, "re_mirror_plane", value.as_ref(), &placeholder) {
            m.plane = MirrorPlane::Plane(p);
            changed = true;
        }
    });
    if matches!(m.plane, MirrorPlane::Face { .. }) {
        clear_plane_hint(&ctx);
    }
    match finding.clone() {
        Finding::Working => {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label(RichText::new("Looking for the scan's own symmetry plane…").weak());
            })
            .response
            .on_hover_text(search_saying(&ctx));
        }
        Finding::Found(r) => {
            let on_it = matches!(&m.plane, MirrorPlane::Plane(PlaneRef::Fitted { label: Some(l), .. }) if l == SYMMETRY_LABEL);
            let head = if on_it { "Set to the scan’s own symmetry plane" } else { "The scan has a symmetry plane" };
            hint(ui, &format!("{head}: its mirror image lies on the scan to σ {} over {} of {} samples.", len(r.rms, 3), count(r.matched), count(r.sampled)));
            ui.horizontal(|ui| {
                if !on_it && ui.small_button("Use it").clicked() {
                    if let Some(plane) = symmetry_plane_ref(&r) {
                        m.plane = MirrorPlane::Plane(plane);
                        changed = true;
                    }
                }
                if ui.small_button("Keep as a plane element").on_hover_text("Make it a plane element in 3D Measure — Symmetry plane — for sketches, sections and dimensions to stand on; the mirror then refers to the element").clicked() {
                    finding = keep_as_element(ws, m, &r);
                    changed = true;
                }
            });
        }
        Finding::Element(name) => hint(ui, &format!("Set to {name}, the symmetry plane measured on this scan.")),
        Finding::None(rms) => {
            let off = rms.map_or(String::new(), |s| format!(" (the best mirror image stands off it by σ {})", len(s, 2)));
            hint(ui, &format!("The scan has no symmetry plane of its own{off} — click a flat face of the body or choose a plane."));
        }
        Finding::Idle => {}
    }
    changed |= ui.checkbox(&mut m.join, RichText::new("Join with the source").weak()).changed();
    ctx.data_mut(|d| d.insert_temp(finding_id(), finding));
    changed
}

/// A click on a flat face of the body takes it as the plane; the flat face under the cursor is
/// lit (ScanRuler: a mirror asks the viewport for the faces of its body).
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, m: &mut MirrorFeature, _pick: &mut PickWhat) -> bool {
    let Some(Hit::Body { body, face, .. }) = ws.pick_hit(vp, p) else { return false };
    if body != m.body {
        return false;
    }
    let Some(print) = ws.pick_view(body).and_then(|v| v.topology.faces.get(face)).map(|f| f.fingerprint) else { return false };
    if print.kind != FaceKind::Plane {
        return false;
    }
    ws.view.hover_face = Some((body, face));
    if !clicked {
        return false;
    }
    m.plane = MirrorPlane::Face { face: print };
    true
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, m: &MirrorFeature) {
    match &m.plane {
        MirrorPlane::Face { face } => paint_faces(ws, painter, vp, m.body, std::slice::from_ref(face)),
        MirrorPlane::Plane(_) => paint_plane_hint(ws, painter, vp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_reverse::frame::WorldAxis;

    fn found(rms: f64, matched: usize) -> SymmetryPlane {
        SymmetryPlane { normal: DVec3::Z, point: DVec3::new(0.0, 0.0, 2.0), rms, matched, sampled: 4000, candidate: 0 }
    }

    fn mirror(plane: PlaneRef) -> MirrorFeature {
        MirrorFeature { body: mm_reverse::model::BodyId::new(2), plane: MirrorPlane::Plane(plane), join: true, exprs: Exprs::new() }
    }

    #[test]
    fn a_found_symmetry_sets_the_plane_while_none_is_chosen_and_stands_aside_for_one_that_is() {
        let mut m = mirror(no_plane());
        assert!(ready(&m).is_err());
        let (f, changed) = land(&mut m, &Ok(found(0.02, 3600)));
        assert!(changed && matches!(f, Finding::Found(_)));
        assert!(matches!(&m.plane, MirrorPlane::Plane(PlaneRef::Fitted { label: Some(l), .. }) if l == SYMMETRY_LABEL));
        assert!(ready(&m).is_ok());
        // A plane chosen in the meantime stands.
        let mut chosen = mirror(PlaneRef::world(WorldAxis::X, 0.0));
        let (f, changed) = land(&mut chosen, &Ok(found(0.02, 3600)));
        assert!(!changed && matches!(f, Finding::Found(_)));
        assert_eq!(chosen.plane, MirrorPlane::Plane(PlaneRef::world(WorldAxis::X, 0.0)));
    }

    #[test]
    fn a_loose_mirror_image_or_a_failed_search_is_no_symmetry() {
        let mut m = mirror(no_plane());
        assert!(matches!(land(&mut m, &Ok(found(3.48, 3268))).0, Finding::None(Some(r)) if (r - 3.48).abs() < 1e-12));
        assert!(matches!(land(&mut m, &Ok(found(f64::INFINITY, 0))).0, Finding::None(None)));
        assert!(matches!(land(&mut m, &Err("no scan".into())).0, Finding::None(None)));
        assert!(!plane_chosen(match &m.plane {
            MirrorPlane::Plane(p) => p,
            MirrorPlane::Face { .. } => panic!("a face"),
        }));
    }

    #[test]
    fn writes_counts_as_the_box_reads_them() {
        assert_eq!(count(7), "7");
        assert_eq!(count(3600), "3,600");
        assert_eq!(count(1234567), "1,234,567");
    }
}
