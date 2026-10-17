//! Mesh file formats: STL, OBJ, PLY, OFF, 3MF.
//!
//! Reading goes through [`read_mesh`] (dispatching on the file extension) or
//! [`read_mesh_from_bytes`]. Every reader produces an indexed triangle set that is turned
//! into a [`Mesh`] with [`Mesh::from_triangles`], so non-manifold edges are split and
//! degenerate/duplicate triangles are skipped (see [`ImportedMesh::report`]). STL stores
//! no shared vertices, so STL corners are welded by exact (bit-identical) position by
//! default; see [`ReadOptions`] and [`weld_vertices`] for tolerance welding.
//!
//! Writing goes through [`write_mesh`], which writes to a temporary file in the target
//! directory and renames it into place, so a failed export never clobbers an existing file.
//!
//! Malformed or truncated input always produces an `Err`, never a panic.

mod obj;
mod off;
mod parse;
mod ply;
mod stl;
mod threemf;
mod weld;

pub use weld::{weld_vertices, WeldResult};

use anyhow::{anyhow, bail, Context, Result};
use glam::DVec3;
use mm_core::mesh::BuildReport;
use mm_core::Mesh;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Cursor, Seek, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Supported file formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    /// Stereolithography, binary or ASCII.
    Stl,
    /// Wavefront OBJ.
    Obj,
    /// Stanford PLY (ASCII, binary little or big endian).
    Ply,
    /// Geomview Object File Format.
    Off,
    /// 3D Manufacturing Format (zip package).
    ThreeMf,
}

impl Format {
    /// All formats, in a stable order.
    pub const ALL: [Format; 5] = [Format::Stl, Format::Obj, Format::Ply, Format::Off, Format::ThreeMf];

    /// Format for a file extension (without the dot, case-insensitive).
    pub fn from_extension(ext: &str) -> Option<Format> {
        match ext.to_ascii_lowercase().as_str() {
            "stl" => Some(Format::Stl),
            "obj" => Some(Format::Obj),
            "ply" => Some(Format::Ply),
            "off" => Some(Format::Off),
            "3mf" => Some(Format::ThreeMf),
            _ => None,
        }
    }

    /// Format implied by a path's extension.
    pub fn from_path(path: &Path) -> Option<Format> {
        path.extension().and_then(|e| e.to_str()).and_then(Format::from_extension)
    }

    /// Canonical lowercase extension (without the dot).
    pub fn extension(self) -> &'static str {
        match self {
            Format::Stl => "stl",
            Format::Obj => "obj",
            Format::Ply => "ply",
            Format::Off => "off",
            Format::ThreeMf => "3mf",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Format::Stl => "STL",
            Format::Obj => "OBJ",
            Format::Ply => "PLY",
            Format::Off => "OFF",
            Format::ThreeMf => "3MF",
        })
    }
}

/// Vertex welding policy applied while importing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Weld {
    /// Exact weld for STL (which has no shared vertices); keep the file's own vertex
    /// sharing for indexed formats.
    #[default]
    Auto,
    /// Never merge vertices (an STL then becomes a triangle soup).
    None,
    /// Merge bit-identical positions in every format (-0.0 equals 0.0).
    Exact,
    /// Exact weld, then merge vertices closer than this distance (file units, after
    /// 3MF unit conversion).
    Tolerance(f64),
}

/// Options for [`read_mesh_with`] / [`read_mesh_from_bytes_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReadOptions {
    /// Vertex welding policy.
    pub weld: Weld,
}

/// A mesh loaded from a file.
#[derive(Clone, Debug)]
pub struct ImportedMesh {
    /// The mesh. Vertex ids may be sparse (vertices not used by any triangle are removed).
    pub mesh: Mesh,
    /// Display name (the file stem, or `"mesh"` when read from bytes).
    pub name: String,
    /// Format the mesh was read from.
    pub format: Format,
    /// What [`Mesh::from_triangles`] had to repair or skip.
    pub report: BuildReport,
    /// Number of file vertices merged away by welding (STL corners, or [`Weld`] options).
    pub welded_vertices: usize,
    /// Per-vertex RGB colors in `0..=1`, indexed by vertex id of [`Self::mesh`]
    /// (length `mesh.max_vid()`), when the file has vertex colors (PLY, OBJ, COFF).
    pub colors: Option<Vec<[f32; 3]>>,
    /// Non-fatal problems noticed while parsing (e.g. ignored trailing bytes).
    pub warnings: Vec<String>,
}

/// Options for writing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WriteOptions {
    /// STL: write ASCII instead of binary.
    pub stl_ascii: bool,
    /// OBJ: emit a `g group_N` statement whenever the face group changes.
    pub obj_groups: bool,
    /// Write coordinates with full `f64` precision (ASCII formats, 3MF, and PLY as
    /// `double`). By default coordinates are written as `f32`, which is what STL and most
    /// tools use; binary STL is always `f32`.
    pub double_precision: bool,
    /// Name stored in the file (STL solid name / header, 3MF object name). Defaults to the
    /// file stem in [`write_mesh`], `"mesh"` otherwise.
    pub name: Option<String>,
}

/// Parser output shared by all formats: an indexed triangle set.
#[derive(Default)]
pub(crate) struct Raw {
    pub positions: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
    /// Per-triangle face group ids.
    pub groups: Option<Vec<i32>>,
    /// Per-position colors.
    pub colors: Option<Vec<[f32; 3]>>,
    pub warnings: Vec<String>,
    /// Vertices merged by the parser itself (STL exact weld).
    pub welded: usize,
}

/// Read a mesh file, choosing the format from the extension (case-insensitive).
pub fn read_mesh(path: &Path) -> Result<ImportedMesh> {
    read_mesh_with(path, &ReadOptions::default())
}

/// [`read_mesh`] with explicit options.
pub fn read_mesh_with(path: &Path, opts: &ReadOptions) -> Result<ImportedMesh> {
    let format = Format::from_path(path)
        .ok_or_else(|| anyhow!("unsupported file extension: {}", path.display()))?;
    let data = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let mut m = read_mesh_from_bytes_with(&data, format, opts)
        .with_context(|| format!("failed to load {} as {format}", path.display()))?;
    if let Some(stem) = path.file_stem() {
        m.name = stem.to_string_lossy().into_owned();
    }
    Ok(m)
}

/// Parse an in-memory file of the given format.
pub fn read_mesh_from_bytes(bytes: &[u8], format: Format) -> Result<ImportedMesh> {
    read_mesh_from_bytes_with(bytes, format, &ReadOptions::default())
}

/// [`read_mesh_from_bytes`] with explicit options.
pub fn read_mesh_from_bytes_with(bytes: &[u8], format: Format, opts: &ReadOptions) -> Result<ImportedMesh> {
    if bytes.is_empty() {
        bail!("file is empty");
    }
    let raw = match format {
        Format::Stl => stl::read(bytes, opts.weld != Weld::None)?,
        Format::Obj => obj::read(bytes)?,
        Format::Ply => ply::read(bytes)?,
        Format::Off => off::read(bytes)?,
        Format::ThreeMf => threemf::read(bytes)?,
    };
    finish(raw, format, opts)
}

/// Validate, weld and build the mesh.
fn finish(mut raw: Raw, format: Format, opts: &ReadOptions) -> Result<ImportedMesh> {
    if raw.triangles.is_empty() {
        bail!("file contains no triangles");
    }
    let nv = raw.positions.len();
    if nv >= u32::MAX as usize {
        bail!("too many vertices ({nv})");
    }
    if let Some(bad) = raw.triangles.iter().flatten().find(|&&v| v as usize >= nv) {
        bail!("face references vertex index {bad}, but the file has only {nv} vertices");
    }
    if raw.groups.as_ref().is_some_and(|g| g.len() != raw.triangles.len()) {
        bail!("internal error: face group count does not match triangle count");
    }

    // Drop triangles touching non-finite coordinates.
    if raw.positions.iter().any(|p| !p.is_finite()) {
        let before = raw.triangles.len();
        let keep: Vec<bool> = raw
            .triangles
            .iter()
            .map(|t| t.iter().all(|&v| raw.positions[v as usize].is_finite()))
            .collect();
        let mut k = keep.iter();
        raw.triangles.retain(|_| *k.next().unwrap());
        if let Some(g) = &mut raw.groups {
            let mut k = keep.iter();
            g.retain(|_| *k.next().unwrap());
        }
        raw.warnings.push(format!(
            "skipped {} triangles with non-finite (NaN/inf) coordinates",
            before - raw.triangles.len()
        ));
        if raw.triangles.is_empty() {
            bail!("all triangles have non-finite coordinates");
        }
    }

    let mut welded = raw.welded;
    let extra = match opts.weld {
        Weld::Auto | Weld::None => None,
        Weld::Exact => (format != Format::Stl).then_some(0.0),
        Weld::Tolerance(t) => Some(t),
    };
    if let Some(tol) = extra {
        let w = weld_vertices(&raw.positions, &raw.triangles, tol);
        if let Some(c) = &raw.colors {
            let first = invert_remap(&w.remap, w.positions.len());
            raw.colors = Some(first.iter().map(|&i| c[i as usize]).collect());
        }
        welded += w.welded;
        raw.positions = w.positions;
        raw.triangles = w.triangles;
    }

    let (mesh, report) = Mesh::from_triangles(&raw.positions, &raw.triangles, raw.groups.as_deref());
    if mesh.triangle_count() == 0 {
        bail!("file contains no valid triangles ({} degenerate)", report.degenerate);
    }
    let colors = raw.colors.map(|c| colors_for_mesh(&mesh, &raw.positions, c));
    Ok(ImportedMesh {
        mesh,
        name: "mesh".to_string(),
        format,
        report,
        welded_vertices: welded,
        colors,
        warnings: raw.warnings,
    })
}

/// First input index for each output id of a remap.
fn invert_remap(remap: &[u32], n_out: usize) -> Vec<u32> {
    let mut first = vec![u32::MAX; n_out];
    for (i, &r) in remap.iter().enumerate().rev() {
        first[r as usize] = i as u32;
    }
    first
}

/// Re-index per-position colors by mesh vertex id. `from_triangles` keeps input vertex
/// `i` as id `i` and appends private copies for non-manifold splits; those copies get the
/// color of an input vertex at the same position.
fn colors_for_mesh(mesh: &Mesh, positions: &[DVec3], mut colors: Vec<[f32; 3]>) -> Vec<[f32; 3]> {
    let n = positions.len();
    colors.resize(n, [1.0; 3]);
    if mesh.max_vid() > n {
        let mut by_pos: weld::FastMap<weld::Key3<u64>, u32> = weld::FastMap::default();
        for (i, &p) in positions.iter().enumerate() {
            by_pos.entry(weld::dvec_key(p)).or_insert(i as u32);
        }
        for v in n..mesh.max_vid() {
            let c = by_pos
                .get(&weld::dvec_key(mesh.vertex(v as u32)))
                .map_or([1.0; 3], |&i| colors[i as usize]);
            colors.push(c);
        }
    }
    colors
}

/// Write a mesh, choosing the format from the extension (case-insensitive).
///
/// The data goes to a temporary file in the same directory, which is flushed, synced and
/// then renamed over `path`.
pub fn write_mesh(path: &Path, mesh: &Mesh, opts: &WriteOptions) -> Result<()> {
    let format = Format::from_path(path)
        .ok_or_else(|| anyhow!("unsupported file extension: {}", path.display()))?;
    let mut opts = opts.clone();
    if opts.name.is_none() {
        opts.name = path.file_stem().map(|s| s.to_string_lossy().into_owned());
    }
    write_atomically(path, |w| write_mesh_to(w, mesh, format, &opts))
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Write a mesh in the given format to a seekable writer (seeking is only used by 3MF).
/// The writer should be buffered.
pub fn write_mesh_to<W: Write + Seek>(w: &mut W, mesh: &Mesh, format: Format, opts: &WriteOptions) -> Result<()> {
    if mesh.triangle_count() == 0 {
        bail!("mesh has no triangles");
    }
    let (pos, tris, groups) = mesh.to_arrays();
    let name = opts.name.as_deref().unwrap_or("mesh");
    match format {
        Format::Stl if opts.stl_ascii => stl::write_ascii(w, &pos, &tris, name, opts.double_precision)?,
        Format::Stl => stl::write_binary(w, &pos, &tris, name)?,
        Format::Obj => obj::write(w, &pos, &tris, opts.obj_groups.then_some(&groups[..]), opts.double_precision)?,
        Format::Ply => ply::write(w, &pos, &tris, opts.double_precision)?,
        Format::Off => off::write(w, &pos, &tris, opts.double_precision)?,
        Format::ThreeMf => threemf::write(w, &pos, &tris, name, opts.double_precision)?,
    }
    w.flush()?;
    Ok(())
}

/// Write a mesh in the given format into memory.
pub fn write_mesh_to_bytes(mesh: &Mesh, format: Format, opts: &WriteOptions) -> Result<Vec<u8>> {
    let mut cur = Cursor::new(Vec::new());
    write_mesh_to(&mut cur, mesh, format, opts)?;
    Ok(cur.into_inner())
}

fn write_atomically(path: &Path, f: impl FnOnce(&mut BufWriter<File>) -> Result<()>) -> Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let file_name = path.file_name().ok_or_else(|| anyhow!("output path has no file name"))?;
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let tmp = dir.join(format!(
        ".{}.{}-{}.tmp",
        file_name.to_string_lossy(),
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .with_context(|| format!("cannot create {}", tmp.display()))?;
        let mut w = BufWriter::with_capacity(1 << 20, file);
        f(&mut w)?;
        let file = w.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Write an `f64` as text with the requested precision (shortest round-trip form).
#[inline]
pub(crate) fn write_float<W: Write + ?Sized>(w: &mut W, x: f64, double: bool) -> std::io::Result<()> {
    if double {
        write!(w, "{x}")
    } else {
        write!(w, "{}", x as f32)
    }
}

/// Append `x y z` (space separated, no newline) to a text buffer.
#[inline]
pub(crate) fn put_xyz(b: &mut Vec<u8>, p: DVec3, double: bool, sep: &[u8]) {
    // Writing into a Vec cannot fail.
    let _ = write_float(b, p.x, double);
    b.extend_from_slice(sep);
    let _ = write_float(b, p.y, double);
    b.extend_from_slice(sep);
    let _ = write_float(b, p.z, double);
}

/// Format `items` into text in parallel chunks and write the chunks in order.
/// `f(buf, index, item)` appends the text for one item.
pub(crate) fn write_parallel<W: Write + ?Sized, T: Sync>(
    w: &mut W,
    items: &[T],
    f: impl Fn(&mut Vec<u8>, usize, &T) + Sync,
) -> std::io::Result<()> {
    use rayon::prelude::*;
    const CHUNK: usize = 8192;
    const BATCH: usize = CHUNK * 64;
    for (bi, batch) in items.chunks(BATCH).enumerate() {
        let bufs: Vec<Vec<u8>> = batch
            .par_chunks(CHUNK)
            .enumerate()
            .map(|(ci, chunk)| {
                let mut b = Vec::with_capacity(chunk.len() * 48);
                let base = bi * BATCH + ci * CHUNK;
                for (k, x) in chunk.iter().enumerate() {
                    f(&mut b, base + k, x);
                }
                b
            })
            .collect();
        for b in &bufs {
            w.write_all(b)?;
        }
    }
    Ok(())
}

/// Fan-triangulate a polygon given as vertex indices, appending to `out`.
#[inline]
pub(crate) fn fan(poly: &[u32], out: &mut Vec<[u32; 3]>) -> usize {
    if poly.len() < 3 {
        return 0;
    }
    for i in 1..poly.len() - 1 {
        out.push([poly[0], poly[i], poly[i + 1]]);
    }
    poly.len() - 2
}
