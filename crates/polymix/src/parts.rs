//! The Meshmix parts library: Polymix's built-in parts, the user's own (made with
//! Select > Convert To), and folders of mesh files the user added (Import Parts Folder);
//! plus the user's stamp stencils. User parts and stencils live in the data directory
//! (`parts/<name>.<open|solid>.pmx`, `stencils/<name>.txt`), the list of parts folders in
//! `parts_folders.txt` there (one path per line).

use anyhow::{bail, Context, Result};
use glam::{DVec2, DVec3};
use mm_core::Mesh;
use mm_edit::{Part, PartKind, StencilShape};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub struct LibPart {
    pub part: Arc<Part>,
    /// File of a user or folder part (None for built-ins).
    pub file: Option<PathBuf>,
    /// The parts folder it comes from (index into [`Library::folders`]).
    pub folder: Option<usize>,
    /// Shaded drawing in a unit square, see [`thumbnail`].
    pub thumb: Arc<egui::Mesh>,
    /// The name as its tile shows it (cut short).
    label: String,
    /// The thumbnail placed in the tile's picture, for the rectangle it was placed in.
    placed: Option<(egui::Rect, Arc<egui::Mesh>)>,
}

impl LibPart {
    fn new(part: Arc<Part>, file: Option<PathBuf>, folder: Option<usize>, thumb: Arc<egui::Mesh>) -> Self {
        Self { label: elide(&part.name, 12), part, file, folder, thumb, placed: None }
    }
}

/// Bumped whenever the library's files change (a part saved, a parts folder added or
/// removed), so that a library loaded before is known to be out of date.
static CHANGES: AtomicU64 = AtomicU64::new(0);

/// How many times the library's files have changed (see [`CHANGES`]).
pub fn library_changes() -> u64 {
    CHANGES.load(Ordering::Relaxed)
}

fn library_changed() {
    CHANGES.fetch_add(1, Ordering::Relaxed);
}

/// A folder of mesh files in the library.
pub struct PartsFolder {
    pub path: PathBuf,
    /// Mesh files in it that are not usable parts: (file name, why).
    pub skipped: Vec<(String, String)>,
    /// Why the folder could not be read (moved, deleted…).
    pub error: Option<String>,
}

impl PartsFolder {
    pub fn label(&self) -> String {
        self.path.file_name().map_or_else(|| self.path.display().to_string(), |n| n.to_string_lossy().into_owned())
    }
}

#[derive(Default)]
pub struct Library {
    pub parts: Vec<LibPart>,
    pub folders: Vec<PartsFolder>,
}

fn parts_dir() -> Option<PathBuf> {
    crate::app::data_dir().map(|d| d.join("parts"))
}

/// Where the user's stamp stencils are kept.
pub fn stencils_dir() -> Option<PathBuf> {
    crate::app::data_dir().map(|d| d.join("stencils"))
}

fn folders_file() -> Option<PathBuf> {
    crate::app::data_dir().map(|d| d.join("parts_folders.txt"))
}

impl Library {
    /// Built-in parts, the user's saved parts and the parts in the user's parts folders.
    /// Parts already loaded from files that have not changed since are taken as they were.
    pub fn load() -> Self {
        // Built-ins never change: made once per session.
        static BUILTIN: OnceLock<Vec<(Arc<Part>, Arc<egui::Mesh>)>> = OnceLock::new();
        let builtin = BUILTIN.get_or_init(|| {
            mm_edit::builtin_parts()
                .into_iter()
                .map(|p| {
                    let thumb = Arc::new(thumbnail(&p.mesh, p.kind));
                    (Arc::new(p), thumb)
                })
                .collect()
        });
        let mut parts: Vec<LibPart> = builtin.iter().map(|(p, t)| LibPart::new(p.clone(), None, None, t.clone())).collect();
        let mut files: Vec<PathBuf> = parts_dir().and_then(|d| std::fs::read_dir(d).ok()).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "pmx")).collect();
        files.sort();
        for f in files {
            match user_part(&f) {
                Ok((p, thumb)) => parts.push(LibPart::new(p, Some(f), None, thumb)),
                Err(e) => log::warn!("skipping part {}: {e}", f.display()),
            }
        }
        let mut lib = Self { parts, folders: Vec::new() };
        for dir in parts_folders() {
            lib.add_folder_parts(&dir);
        }
        lib
    }

    /// Scan a parts folder into the library, renaming parts whose names are taken.
    fn add_folder_parts(&mut self, dir: &Path) {
        let scan = scan_parts_folder(dir);
        let index = self.folders.len();
        let mut taken: HashSet<String> = self.parts.iter().map(|p| p.part.name.clone()).collect();
        let label = dir.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        for fp in scan.parts {
            let mut part = fp.part;
            if taken.contains(&part.name) {
                let base = format!("{} ({label})", part.name);
                let name = if taken.contains(&base) { free_name(&base, |n| taken.contains(n)) } else { base };
                part = Arc::new(Part { name, ..(*part).clone() });
            }
            taken.insert(part.name.clone());
            self.parts.push(LibPart::new(part, Some(fp.file), Some(index), fp.thumb));
        }
        self.folders.push(PartsFolder { path: dir.to_path_buf(), skipped: scan.skipped, error: scan.error });
    }

    pub fn find(&self, name: &str) -> Option<Arc<Part>> {
        self.parts.iter().find(|p| p.part.name == name).map(|p| p.part.clone())
    }
}

fn load_part(path: &Path) -> Result<Part> {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("part");
    let (name, kind) = stem.rsplit_once('.').context("part file name must be <name>.<open|solid>.pmx")?;
    let mesh = crate::project::mesh_from_bytes(&std::fs::read(path)?)?;
    Ok(match kind {
        "open" => Part::open(name, mesh)?,
        "solid" => Part::solid(name, mesh)?,
        k => bail!("unknown part kind '{k}'"),
    })
}

/// A part file loaded with its thumbnail (or why it could not be).
type Loaded = std::result::Result<(Arc<Part>, Arc<egui::Mesh>), String>;
/// Loaded part files by path, with the modification time and size they were loaded at.
type PartCache = Mutex<HashMap<PathBuf, (Option<std::time::SystemTime>, u64, Loaded)>>;

/// `load` of the file at `path` (`Err` with the message to show), reusing the last load from
/// `cache` while the file's modification time and size are unchanged.
fn cached_load(cache: &PartCache, path: &Path, load: impl FnOnce() -> Loaded) -> Loaded {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let stamp = (meta.modified().ok(), meta.len());
    if let Some((m, l, r)) = cache.lock().ok().and_then(|c| c.get(path).cloned()) {
        if (m, l) == stamp {
            return r;
        }
    }
    let loaded = load();
    if let Ok(mut c) = cache.lock() {
        c.insert(path.to_path_buf(), (stamp.0, stamp.1, loaded.clone()));
    }
    loaded
}

/// A user part (`.pmx`) with its thumbnail, loaded again only when its file changed.
fn user_part(path: &Path) -> Loaded {
    static CACHE: OnceLock<PartCache> = OnceLock::new();
    cached_load(CACHE.get_or_init(Default::default), path, || {
        let p = load_part(path).map_err(|e| format!("{e:#}"))?;
        let thumb = Arc::new(thumbnail(&p.mesh, p.kind));
        Ok((Arc::new(p), thumb))
    })
}

/// The first of "<prefix> 1", "<prefix> 2", … not `taken`.
fn free_name(prefix: &str, taken: impl Fn(&str) -> bool) -> String {
    (1..).map(|i| format!("{prefix} {i}")).find(|n| !taken(n)).unwrap()
}

/// Save a part to the user library; returns its name.
pub fn save_part(part_mesh: Mesh, kind: PartKind) -> Result<String> {
    let dir = parts_dir().context("no data directory")?;
    std::fs::create_dir_all(&dir)?;
    let label = if kind == PartKind::Open { "open" } else { "solid" };
    let name = free_name("My part", |n| dir.join(format!("{n}.open.pmx")).exists() || dir.join(format!("{n}.solid.pmx")).exists());
    let part = if kind == PartKind::Open { Part::open(&name, part_mesh)? } else { Part::solid(&name, part_mesh)? };
    std::fs::write(dir.join(format!("{name}.{label}.pmx")), crate::project::mesh_to_bytes(&part.mesh))?;
    library_changed();
    Ok(name)
}

// ---------------------------------------------------------------------------------------
// Parts folders
// ---------------------------------------------------------------------------------------

/// Whether a mesh is a solid part (closed) or an open part (exactly one border loop, its
/// rim); anything else is not a part.
pub fn classify_part(mesh: &Mesh) -> std::result::Result<PartKind, String> {
    if mesh.triangle_count() == 0 {
        return Err("no triangles".into());
    }
    if mesh.is_closed() {
        return Ok(PartKind::Solid);
    }
    match mm_core::topology::boundary_loops(mesh).len() {
        1 => Ok(PartKind::Open),
        n => Err(format!("{n} open borders: an open part has exactly one (its rim), a solid part none")),
    }
}

/// A part found in a folder.
pub struct FolderPart {
    pub file: PathBuf,
    pub part: Arc<Part>,
    pub thumb: Arc<egui::Mesh>,
}

/// What [`scan_parts_folder`] found.
#[derive(Default)]
pub struct FolderScan {
    /// In file-name order.
    pub parts: Vec<FolderPart>,
    pub skipped: Vec<(String, String)>,
    pub error: Option<String>,
}

/// Meshes larger than this are drawn reduced in their thumbnails.
const THUMB_TRIANGLES: usize = 6000;

/// Load a mesh file as a part, reusing the last load while the file is unchanged.
fn folder_part(path: &Path) -> Loaded {
    static CACHE: OnceLock<PartCache> = OnceLock::new();
    cached_load(CACHE.get_or_init(Default::default), path, || {
        let imported = mm_io::read_mesh(path).map_err(|e| format!("{e:#}"))?;
        let kind = classify_part(&imported.mesh)?;
        let name = path.file_stem().map_or_else(|| "part".to_string(), |s| s.to_string_lossy().into_owned());
        let part = match kind {
            PartKind::Open => Part::open(name, imported.mesh),
            PartKind::Solid => Part::solid(name, imported.mesh),
        }
        .map_err(|e| e.to_string())?;
        let thumb = if part.mesh.triangle_count() > THUMB_TRIANGLES {
            let mut m = part.mesh.clone();
            let params = mm_remesh::ReduceParams { preserve_boundary: false, preserve_group_boundaries: false, ..mm_remesh::ReduceParams::with_target(mm_remesh::ReduceTarget::TriangleCount(THUMB_TRIANGLES)) };
            let _ = mm_remesh::reduce(&mut m, &params, None);
            thumbnail(&m, kind)
        } else {
            thumbnail(&part.mesh, kind)
        };
        Ok((Arc::new(part), Arc::new(thumb)))
    })
}

/// The mesh files of a folder (not its subfolders) as parts: closed meshes are solid parts,
/// meshes with one border loop open parts; others are listed as skipped.
pub fn scan_parts_folder(dir: &Path) -> FolderScan {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => return FolderScan { error: Some(format!("cannot read {}: {e}", dir.display())), ..Default::default() },
    };
    let mut files: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_file() && mm_io::Format::from_path(p).is_some_and(|f| f.can_read())).collect();
    files.sort_by_key(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()));
    let loaded: Vec<Loaded> = files.par_iter().map(|f| folder_part(f)).collect();
    let mut scan = FolderScan::default();
    for (file, r) in files.into_iter().zip(loaded) {
        match r {
            Ok((part, thumb)) => scan.parts.push(FolderPart { file, part, thumb }),
            Err(e) => scan.skipped.push((file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), e)),
        }
    }
    scan
}

fn read_list(file: &Path) -> Vec<PathBuf> {
    std::fs::read_to_string(file).unwrap_or_default().lines().map(str::trim).filter(|l| !l.is_empty()).map(PathBuf::from).collect()
}

fn write_list(file: &Path, dirs: &[PathBuf]) -> Result<()> {
    if let Some(d) = file.parent() {
        std::fs::create_dir_all(d)?;
    }
    let text: String = dirs.iter().map(|d| format!("{}\n", d.display())).collect();
    std::fs::write(file, text).with_context(|| format!("cannot write {}", file.display()))
}

/// The parts folders in the library, in the order they were added.
pub fn parts_folders() -> Vec<PathBuf> {
    folders_file().map(|f| read_list(&f)).unwrap_or_default()
}

fn add_folder_to(list: &Path, dir: &Path) -> Result<String> {
    if !dir.is_dir() {
        bail!("{} is not a folder", dir.display());
    }
    let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let mut dirs = read_list(list);
    let name = dir.file_name().map_or_else(|| dir.display().to_string(), |n| n.to_string_lossy().into_owned());
    if dirs.contains(&dir) {
        return Ok(format!("Parts folder \"{name}\" is already in the library"));
    }
    dirs.push(dir.clone());
    write_list(list, &dirs)?;
    library_changed();
    let scan = scan_parts_folder(&dir);
    let solid = scan.parts.iter().filter(|p| p.part.kind == PartKind::Solid).count();
    let mut msg = format!("Added parts folder \"{name}\" to Meshmix: {} ({solid} solid, {} open)", count(scan.parts.len(), "part"), scan.parts.len() - solid);
    if !scan.skipped.is_empty() {
        msg += &format!(", {} skipped", count(scan.skipped.len(), "file"));
    }
    Ok(msg)
}

/// "1 part", "2 parts".
fn count(n: usize, noun: &str) -> String {
    format!("{n} {noun}{}", if n == 1 { "" } else { "s" })
}

fn remove_folder_from(list: &Path, dir: &Path) -> Result<()> {
    let mut dirs = read_list(list);
    dirs.retain(|d| d != dir);
    write_list(list, &dirs)?;
    library_changed();
    Ok(())
}

/// Add a folder of mesh files to the parts library (remembered across sessions); returns a
/// status message.
pub fn add_parts_folder(dir: &Path) -> Result<String> {
    add_folder_to(&folders_file().context("no data directory")?, dir)
}

/// Take a folder out of the parts library (its files are left alone).
pub fn remove_parts_folder(dir: &Path) -> Result<()> {
    remove_folder_from(&folders_file().context("no data directory")?, dir)
}

/// File > Import Parts Folder: pick a folder and add it. None if cancelled.
pub fn import_parts_folder_dialog() -> Option<String> {
    let dir = rfd::FileDialog::new().set_title("Add a folder of parts (STL, OBJ, PLY, OFF, 3MF, AMF)").pick_folder()?;
    Some(add_parts_folder(&dir).unwrap_or_else(|e| format!("Import parts folder: {e:#}")))
}

// ---------------------------------------------------------------------------------------
// The Meshmix panel
// ---------------------------------------------------------------------------------------

fn elide(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max - 1).collect::<String>() + "…"
    }
}

/// One part's tile; returns whether it was clicked.
fn part_tile(ui: &mut egui::Ui, p: &mut LibPart) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(72.0, 86.0), egui::Sense::click());
    let v = ui.visuals();
    let bg = if resp.hovered() { v.widgets.hovered.bg_fill } else { v.widgets.inactive.bg_fill };
    ui.painter().rect_filled(rect, 4.0, bg);
    let img = egui::Rect::from_min_size(rect.min + egui::vec2(6.0, 4.0), egui::vec2(60.0, 60.0));
    // Placed once for where the tile is, not copied and moved every frame.
    let placed = match &p.placed {
        Some((r, m)) if *r == img => m.clone(),
        _ => {
            let m = Arc::new(placed_thumbnail(img, &p.thumb));
            p.placed = Some((img, m.clone()));
            m
        }
    };
    ui.painter().add(egui::Shape::mesh(placed));
    ui.painter().text(rect.center_bottom() - egui::vec2(0.0, 6.0), egui::Align2::CENTER_BOTTOM, &p.label, egui::FontId::proportional(11.0), v.text_color());
    crate::app::hover_text_with(resp, || {
        let kind = if p.part.kind == PartKind::Open { "open part: welded into the surface" } else { "solid part" };
        let mut hover = format!("{} ({kind})", p.part.name);
        if let Some(f) = &p.file {
            hover += &format!("\n{}", f.display());
        }
        hover
    })
    .clicked()
}

fn part_grid<'a>(ui: &mut egui::Ui, items: impl Iterator<Item = &'a mut LibPart>, has_obj: bool, drop: &mut Option<String>) {
    ui.horizontal_wrapped(|ui| {
        for p in items {
            if part_tile(ui, p) && has_obj {
                *drop = Some(p.part.name.clone());
            }
        }
    });
}

/// What was done in the Meshmix panel.
#[derive(Default)]
pub struct PanelResponse {
    /// The name of the part clicked (to open Drop Part with).
    pub drop: Option<String>,
    /// The library's files changed or a reload was asked for: load it again.
    pub reload: bool,
}

/// The Meshmix category panel: built-in parts, My parts, one section per parts folder (with
/// its Remove button), and Add parts folder; while there is no library yet (`None`, being
/// loaded), a spinner. Messages go to `status`.
pub fn library_panel(ui: &mut egui::Ui, lib: Option<&mut Library>, has_obj: bool, status: &mut String) -> PanelResponse {
    let Some(l) = lib else {
        ui.horizontal(|ui| {
            polymix_workspace::spinner(ui);
            ui.label("Loading parts…");
        });
        return PanelResponse::default();
    };
    let mut drop = None;
    let mut reload = false;
    for (title, kind) in [("Solid parts", PartKind::Solid), ("Open parts", PartKind::Open)] {
        crate::ui_util::section(ui, title);
        part_grid(ui, l.parts.iter_mut().filter(|p| p.file.is_none() && p.part.kind == kind), has_obj, &mut drop);
    }
    if l.parts.iter().any(|p| p.file.is_some() && p.folder.is_none()) {
        crate::ui_util::section(ui, "My parts");
        part_grid(ui, l.parts.iter_mut().filter(|p| p.file.is_some() && p.folder.is_none()), has_obj, &mut drop);
    }
    for (i, f) in l.folders.iter().enumerate() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(f.label()).strong().color(ui.visuals().strong_text_color())).on_hover_text(f.path.display().to_string());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Remove").on_hover_text("Take this folder out of the parts library (its files are left alone)").clicked() {
                    *status = match remove_parts_folder(&f.path) {
                        Ok(()) => format!("Removed parts folder \"{}\" from the library", f.label()),
                        Err(e) => format!("Remove parts folder: {e:#}"),
                    };
                    reload = true;
                }
            });
        });
        ui.separator();
        if let Some(e) = &f.error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        if f.error.is_none() && !l.parts.iter().any(|p| p.folder == Some(i)) {
            ui.weak("No usable mesh files in this folder.");
        }
        part_grid(ui, l.parts.iter_mut().filter(|p| p.folder == Some(i)), has_obj, &mut drop);
        if !f.skipped.is_empty() {
            let list: String = f.skipped.iter().map(|(n, why)| format!("{n}: {why}\n")).collect();
            ui.small(format!("{} skipped (hover for why)", count(f.skipped.len(), "file"))).on_hover_text(list.trim_end());
        }
    }
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui.button("Add parts folder…").on_hover_text("Add a folder of mesh files (STL, OBJ, PLY, OFF, 3MF, AMF) to the library: closed meshes become solid parts, meshes with one open border open parts").clicked() {
            if let Some(msg) = import_parts_folder_dialog() {
                *status = msg;
                reload = true;
            }
        }
        if ui.button("Reload").on_hover_text("Look again for new and changed parts").clicked() {
            reload = true;
        }
    });
    ui.small("Click a part, then press on the model where it goes and drag to size it. Make your own with Select > Convert To, or add a folder of mesh files.");
    PanelResponse { drop, reload }
}

// ---------------------------------------------------------------------------------------
// Stamp stencils
// ---------------------------------------------------------------------------------------

/// Stencil file text: one `x y` point per line, a blank line between loops.
fn stencil_text(loops: &[Vec<DVec2>]) -> String {
    loops.iter().map(|l| l.iter().map(|p| format!("{} {}\n", p.x, p.y)).collect::<String>()).collect::<Vec<_>>().join("\n")
}

/// Loops of a stencil file (lines that are not two numbers end a loop; `#` starts a comment).
fn parse_stencil(text: &str) -> Vec<Vec<DVec2>> {
    let mut loops = vec![Vec::new()];
    for line in text.lines() {
        let mut it = line.split('#').next().unwrap_or_default().split_whitespace().map(|x| x.parse::<f64>());
        match (it.next(), it.next()) {
            (Some(Ok(x)), Some(Ok(y))) => loops.last_mut().unwrap().push(DVec2::new(x, y)),
            _ => {
                if !loops.last().unwrap().is_empty() {
                    loops.push(Vec::new());
                }
            }
        }
    }
    loops.retain(|l| !l.is_empty());
    loops
}

/// The stencils saved in `dir`, by name; files that are not valid outlines are skipped.
pub fn user_stencils_in(dir: &Path) -> Vec<StencilShape> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("txt"))).collect();
    files.sort();
    files
        .into_iter()
        .filter_map(|p| {
            let loops = parse_stencil(&std::fs::read_to_string(&p).ok()?);
            if let Err(e) = mm_edit::validate_loops(&loops) {
                log::warn!("skipping stencil {}: {e}", p.display());
                return None;
            }
            Some(StencilShape::new(p.file_stem()?.to_string_lossy(), loops))
        })
        .collect()
}

/// The user's stencils (`<data dir>/stencils/*.txt`).
pub fn user_stencils() -> Vec<StencilShape> {
    stencils_dir().map(|d| user_stencils_in(&d)).unwrap_or_default()
}

/// Save a stencil (loops at any scale) in `dir`; returns its name.
pub fn save_stencil_in(dir: &Path, loops: &[Vec<DVec2>]) -> Result<String> {
    mm_edit::validate_loops(loops).map_err(|e| anyhow::anyhow!("not a usable outline: {e}"))?;
    std::fs::create_dir_all(dir)?;
    let name = free_name("My stamp", |n| dir.join(format!("{n}.txt")).exists());
    std::fs::write(dir.join(format!("{name}.txt")), stencil_text(loops))?;
    Ok(name)
}

/// Save a stencil to the user's stencils; returns its name.
pub fn save_stencil(loops: &[Vec<DVec2>]) -> Result<String> {
    save_stencil_in(&stencils_dir().context("no data directory")?, loops)
}

/// Stencil from closed loops of 3D points (a region's border: the outer loop and its holes):
/// projected onto their best-fit plane seen from `view` (the side the region faces),
/// simplified to about `detail` of their size, centred and fitted to the unit circle.
pub fn stencil_from_loops(loops: &[Vec<DVec3>], view: DVec3, detail: f64) -> Result<Vec<Vec<DVec2>>> {
    let all: Vec<DVec3> = loops.iter().flatten().copied().collect();
    if all.len() < 3 {
        bail!("the border is too short to trace");
    }
    let c = all.iter().copied().sum::<DVec3>() / all.len() as f64;
    let mut cov = [[0.0; 3]; 3];
    for p in &all {
        mm_core::linalg::add_outer(&mut cov, *p - c, 1.0);
    }
    let (_, vecs) = mm_core::linalg::sym_eigen3(cov);
    let n = if vecs[0].dot(view) < 0.0 { -vecs[0] } else { vecs[0] };
    let (u, _) = mm_core::geom::orthonormal_basis(n);
    let v = n.cross(u);
    let flat: Vec<Vec<DVec2>> = loops.iter().map(|l| l.iter().map(|p| DVec2::new((*p - c).dot(u), (*p - c).dot(v))).collect()).collect();
    let size = flat.iter().flatten().map(|p| p.length()).fold(0.0, f64::max);
    if size <= 0.0 {
        bail!("the border has no extent");
    }
    // Simplify (more if needed to keep stamping quick), dropping loops that flatten to nothing.
    let mut tol = detail * size;
    let out = loop {
        let mut s: Vec<Vec<DVec2>> = flat.iter().map(|l| mm_edit::stencil::simplify_loop(l, tol)).collect();
        s.retain(|l| l.len() >= 3 && mm_edit::stencil::signed_area(l).abs() > 1e-6 * size * size);
        if s.iter().map(Vec::len).sum::<usize>() <= 2000 || tol > 0.05 * size {
            break s;
        }
        tol *= 2.0;
    };
    let (out, _) = mm_edit::stencil::normalize_loops(out);
    mm_edit::validate_loops(&out).map_err(|e| anyhow::anyhow!("the border overlaps itself when flattened onto its plane ({e}): select a flatter region"))?;
    Ok(out)
}

/// The stamp outline of a selection: every border loop of the selected triangles (holes
/// stay holes) on their best-fit plane, as [`stencil_from_loops`].
pub fn stamp_from_selection(mesh: &Mesh, tris: &[u32]) -> Result<Vec<Vec<DVec2>>> {
    let (sub, _) = mesh.submesh(tris);
    let loops = mm_core::topology::boundary_loops(&sub);
    if loops.is_empty() {
        bail!("the selection has no border to trace (it is a closed shell)");
    }
    let pts: Vec<Vec<DVec3>> = loops.iter().map(|l| l.vertices.iter().map(|&v| sub.vertex(v)).collect()).collect();
    let view: DVec3 = sub.triangle_ids().map(|t| sub.tri_normal(t) * sub.tri_area(t)).sum();
    stencil_from_loops(&pts, view, 0.002)
}

/// What Select > Convert To makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertTo {
    OpenPart,
    SolidPart,
    Stamp,
}

/// Save the selected triangles of `mesh` to the library; returns a status message.
pub fn convert_selection(mesh: &Mesh, tris: &[u32], what: ConvertTo) -> Result<String> {
    if tris.is_empty() {
        bail!("nothing is selected");
    }
    match what {
        ConvertTo::OpenPart => Ok(format!("Added open part \"{}\" to Meshmix > My parts", save_part(mesh.submesh(tris).0, PartKind::Open)?)),
        ConvertTo::SolidPart => Ok(format!("Added solid part \"{}\" to Meshmix > My parts", save_part(mesh.submesh(tris).0, PartKind::Solid)?)),
        ConvertTo::Stamp => {
            let loops = stamp_from_selection(mesh, tris)?;
            let holes = loops.len() - 1;
            let name = save_stencil(&loops)?;
            let extra = if holes > 0 { format!(" with {holes} hole{}", if holes == 1 { "" } else { "s" }) } else { String::new() };
            Ok(format!("Added stamp \"{name}\"{extra} to Sculpt > Stamp > My stamps"))
        }
    }
}

// ---------------------------------------------------------------------------------------
// Thumbnails
// ---------------------------------------------------------------------------------------

/// Flat-shaded drawing of a part in a unit square (y down), seen from above and in front;
/// the part's +z is up.
pub fn thumbnail(mesh: &Mesh, kind: PartKind) -> egui::Mesh {
    let fwd = DVec3::new(-0.45, 0.85, -0.5).normalize();
    let right = fwd.cross(DVec3::Z).normalize();
    let up = right.cross(fwd);
    let light = (-fwd + up * 0.6 + right * 0.3).normalize();
    let base = if kind == PartKind::Open { [205.0, 190.0, 160.0] } else { [190.0, 195.0, 205.0] };
    let screen = |p: DVec3| (p.dot(right), -p.dot(up));
    let pts: Vec<(f64, f64)> = mesh.vertex_ids().map(|v| screen(mesh.vertex(v))).collect();
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for &(x, y) in &pts {
        lo = (lo.0.min(x), lo.1.min(y));
        hi = (hi.0.max(x), hi.1.max(y));
    }
    let span = (hi.0 - lo.0).max(hi.1 - lo.1).max(1e-12);
    let mid = ((lo.0 + hi.0) * 0.5, (lo.1 + hi.1) * 0.5);
    let to2 = |p: DVec3| {
        let (x, y) = screen(p);
        egui::pos2((0.5 + (x - mid.0) / span * 0.9) as f32, (0.5 + (y - mid.1) / span * 0.9) as f32)
    };
    // Painter's algorithm: far triangles first. Open parts show their inside too.
    let mut tris: Vec<(f64, u32)> = mesh.triangle_ids().map(|t| (mesh.tri_centroid(t).dot(fwd), t)).collect();
    tris.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = egui::Mesh::default();
    for (_, t) in tris {
        let mut n = mesh.tri_normal(t);
        if n.dot(fwd) > 0.0 {
            if kind == PartKind::Solid {
                continue;
            }
            n = -n;
        }
        let k = (0.3 + 0.7 * n.dot(light).max(0.0)) as f32;
        let col = egui::Color32::from_rgb((base[0] * k) as u8, (base[1] * k) as u8, (base[2] * k) as u8);
        let i = out.vertices.len() as u32;
        for v in mesh.triangle(t) {
            out.colored_vertex(to2(mesh.vertex(v)), col);
        }
        out.add_triangle(i, i + 1, i + 2);
    }
    out
}

/// A thumbnail placed in `rect`, to draw with `egui::Shape::mesh`.
pub fn placed_thumbnail(rect: egui::Rect, thumb: &egui::Mesh) -> egui::Mesh {
    let mut m = thumb.clone();
    for v in &mut m.vertices {
        v.pos = rect.min + v.pos.to_vec2() * rect.width();
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    /// A fresh folder under the system temp directory, removed when dropped.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!("polymix-parts-{tag}-{}-{:?}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn stencil_from_a_tilted_square_loop() {
        let q = glam::DQuat::from_axis_angle(DVec3::X, 0.7);
        let pts: Vec<DVec3> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].iter().map(|&(x, y)| q * DVec3::new(x, y, 0.0) * 3.0 + DVec3::splat(5.0)).collect();
        let s = stencil_from_loops(&[pts], q * DVec3::Z, 0.002).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].len(), 4);
        assert!(s[0].iter().all(|p| (p.length() - 1.0).abs() < 1e-9));
        // Seen from the side it faces, the loop is counter-clockwise.
        assert!(mm_edit::stencil::signed_area(&s[0]) > 0.0);
    }

    /// An annulus of a plane grid, tilted: a selection with a hole.
    fn annulus_selection() -> (Mesh, Vec<u32>) {
        let mut m = primitives::plane_grid(10.0, 10.0, 40, 40);
        let q = glam::DQuat::from_axis_angle(DVec3::new(1.0, 1.0, 0.0).normalize(), 0.5);
        m.transform(|p| q * p + DVec3::new(3.0, -2.0, 7.0));
        let c = q * DVec3::ZERO + DVec3::new(3.0, -2.0, 7.0);
        let tris = m
            .triangle_ids()
            .filter(|&t| {
                let r = m.tri_centroid(t).distance(c);
                (1.5..4.0).contains(&r)
            })
            .collect();
        (m, tris)
    }

    #[test]
    fn convert_to_stamp_round_trip() {
        let (m, tris) = annulus_selection();
        let loops = stamp_from_selection(&m, &tris).unwrap();
        assert_eq!(loops.len(), 2, "outer border and hole");
        mm_edit::validate_loops(&loops).unwrap();
        // The hole is a hole.
        assert!(!mm_edit::stencil::point_in_loops(DVec2::ZERO, &loops));
        assert!(mm_edit::stencil::point_in_loops(DVec2::new(0.7, 0.0), &loops));
        // Save, list, load: the same outline.
        let dir = TempDir::new("stencils");
        let name = save_stencil_in(&dir.0, &loops).unwrap();
        assert_eq!(name, "My stamp 1");
        assert_eq!(save_stencil_in(&dir.0, &loops).unwrap(), "My stamp 2");
        std::fs::write(dir.0.join("broken.txt"), "0 0\n1 1\n").unwrap();
        let listed = user_stencils_in(&dir.0);
        assert_eq!(listed.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["My stamp 1", "My stamp 2"]);
        let back = &listed[0].loops;
        assert_eq!(back.len(), loops.len());
        for (a, b) in back.iter().zip(&loops) {
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(b).all(|(p, q)| p.distance(*q) < 1e-12));
        }
        // A closed shell has no outline.
        let ball = primitives::icosphere(1.0, 2);
        assert!(stamp_from_selection(&ball, &ball.triangle_ids().collect::<Vec<_>>()).is_err());
    }

    #[test]
    fn stencil_files_with_several_loops_and_comments() {
        let loops = parse_stencil("# a square with a square hole\n0 0\n4 0\n4 4\n0 4\n\n1 1\n1 3\n3 3\n3 1\n");
        assert_eq!(loops.len(), 2);
        assert_eq!(parse_stencil(&stencil_text(&loops)), loops);
    }

    #[test]
    fn parts_folder_scan_tells_open_from_solid() {
        let dir = TempDir::new("folder");
        let w = mm_io::WriteOptions::default();
        mm_io::write_mesh(&dir.0.join("Ball.stl"), &primitives::icosphere(2.0, 2), &w).unwrap();
        mm_io::write_mesh(&dir.0.join("cap.obj"), &primitives::plane_grid(2.0, 2.0, 4, 4), &w).unwrap();
        mm_io::write_mesh(&dir.0.join("tube.ply"), &primitives::cylinder(1.0, 3.0, 12, 3, false), &w).unwrap();
        std::fs::write(dir.0.join("notes.txt"), "not a mesh").unwrap();
        std::fs::write(dir.0.join("empty.stl"), "").unwrap();
        let scan = scan_parts_folder(&dir.0);
        assert!(scan.error.is_none());
        let kinds: Vec<(&str, PartKind)> = scan.parts.iter().map(|p| (p.part.name.as_str(), p.part.kind)).collect();
        assert_eq!(kinds, [("Ball", PartKind::Solid), ("cap", PartKind::Open)]);
        assert!(scan.parts.iter().all(|p| !p.thumb.indices.is_empty()));
        let skipped: Vec<&str> = scan.skipped.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(skipped, ["empty.stl", "tube.ply"], "{:?}", scan.skipped);
        assert!(scan.skipped[1].1.contains("2 open borders"));
        // Remembered, listed once, removed.
        let list = dir.0.join("list").join("parts_folders.txt");
        let msg = add_folder_to(&list, &dir.0).unwrap();
        assert!(msg.contains("2 parts (1 solid, 1 open), 2 files skipped"), "{msg}");
        assert!(add_folder_to(&list, &dir.0).unwrap().contains("already"));
        assert_eq!(read_list(&list), [dir.0.canonicalize().unwrap()]);
        assert!(add_folder_to(&list, &dir.0.join("Ball.stl")).is_err());
        remove_folder_from(&list, &dir.0.canonicalize().unwrap()).unwrap();
        assert!(read_list(&list).is_empty());
        // A missing folder is reported, not fatal.
        assert!(scan_parts_folder(&dir.0.join("gone")).error.is_some());
    }

    #[test]
    fn folder_parts_get_unique_names() {
        let dir = TempDir::new("names");
        mm_io::write_mesh(&dir.0.join("Sphere.stl"), &primitives::icosphere(2.0, 1), &mm_io::WriteOptions::default()).unwrap();
        let mut lib = Library { parts: Vec::new(), folders: Vec::new() };
        lib.add_folder_parts(&dir.0);
        lib.add_folder_parts(&dir.0);
        let names: Vec<&str> = lib.parts.iter().map(|p| p.part.name.as_str()).collect();
        let label = dir.0.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(names, ["Sphere".to_string(), format!("Sphere ({label})")]);
        assert_eq!(lib.parts[1].folder, Some(1));
        assert!(lib.find(&format!("Sphere ({label})")).is_some());
    }

    #[test]
    fn user_parts_are_read_again_only_when_their_file_changes() {
        let dir = TempDir::new("pmx");
        let f = dir.0.join("Knob.solid.pmx");
        std::fs::write(&f, crate::project::mesh_to_bytes(&primitives::icosphere(1.0, 2))).unwrap();
        let (a, b) = (user_part(&f).unwrap(), user_part(&f).unwrap());
        assert!(Arc::ptr_eq(&a.0, &b.0) && Arc::ptr_eq(&a.1, &b.1), "unchanged: the same part");
        assert_eq!(a.0.name, "Knob");
        let fresh = load_part(&f).unwrap();
        assert_eq!(a.0.mesh.to_arrays(), fresh.mesh.to_arrays());
        let t = thumbnail(&fresh.mesh, fresh.kind);
        assert_eq!((&a.1.vertices, &a.1.indices), (&t.vertices, &t.indices));
        // Another mesh in the file (another size): read again.
        std::fs::write(&f, crate::project::mesh_to_bytes(&primitives::icosphere(1.0, 3))).unwrap();
        let c = user_part(&f).unwrap();
        assert!(!Arc::ptr_eq(&a.0, &c.0));
        assert_eq!(c.0.mesh.triangle_count(), primitives::icosphere(1.0, 3).triangle_count());
        // A broken file says why, every time.
        std::fs::write(&f, b"junk, not a mesh at all").unwrap();
        assert!(user_part(&f).err().expect("broken").contains("not a Polymix mesh"));
        assert!(user_part(&dir.0.join("gone.solid.pmx")).is_err());
    }

    #[test]
    fn changes_to_the_library_files_are_counted() {
        let dir = TempDir::new("changes");
        let list = dir.0.join("list").join("parts_folders.txt");
        let before = library_changes();
        add_folder_to(&list, &dir.0).unwrap();
        let added = library_changes();
        assert!(added > before);
        // Already listed: nothing changed.
        add_folder_to(&list, &dir.0).unwrap();
        remove_folder_from(&list, &dir.0.canonicalize().unwrap()).unwrap();
        assert!(library_changes() > added);
    }

    #[test]
    fn tiles_draw_their_thumbnail_placed_once() {
        let p = mm_edit::builtin_parts().into_iter().next().unwrap();
        let thumb = Arc::new(thumbnail(&p.mesh, p.kind));
        let mut part = LibPart::new(Arc::new(p), None, None, thumb.clone());
        assert_eq!(part.label, elide(&part.part.name, 12));
        let ctx = egui::Context::default();
        let mut placed = Vec::new();
        for x in [0.0, 0.0, 30.0] {
            let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0))), ..Default::default() };
            let out = ctx.run_ui(input, |ui| {
                ui.add_space(x);
                part_tile(ui, &mut part);
            });
            let (img, m) = part.placed.clone().expect("placed");
            // The vertices where the old drawing put them: thumbnail scaled into the picture.
            assert!(m.vertices.iter().zip(&thumb.vertices).all(|(a, b)| a.pos == img.min + b.pos.to_vec2() * img.width() && a.color == b.color));
            assert_eq!(m.indices, thumb.indices);
            let meshes = out.shapes.iter().filter(|s| matches!(s.shape, egui::Shape::Mesh(_))).count();
            assert_eq!(meshes, 1);
            placed.push((img, m));
        }
        assert!(Arc::ptr_eq(&placed[0].1, &placed[1].1), "the same place: drawn from the cache");
        assert!(!Arc::ptr_eq(&placed[1].1, &placed[2].1) && placed[1].0 != placed[2].0, "moved: placed again");
    }

    #[test]
    fn thumbnails_cover_their_square() {
        for p in mm_edit::builtin_parts() {
            let t = thumbnail(&p.mesh, p.kind);
            assert!(!t.indices.is_empty(), "{}", p.name);
            assert!(t.vertices.iter().all(|v| (0.0..=1.0).contains(&v.pos.x) && (0.0..=1.0).contains(&v.pos.y)), "{}", p.name);
        }
    }
}
