//! The Meshmix parts library: Polymix's built-in parts plus the user's own (made with
//! Select > Convert To), and the user's stamp stencils. User parts and stencils live in the
//! data directory (`parts/<name>.<open|solid>.pmx`, `stencils/<name>.txt`).

use anyhow::{bail, Context, Result};
use glam::{DVec2, DVec3};
use mm_core::Mesh;
use mm_edit::{Part, PartKind};
use std::path::PathBuf;
use std::sync::Arc;

pub struct LibPart {
    pub part: Arc<Part>,
    /// File of a user part (None for built-ins).
    pub file: Option<PathBuf>,
    /// Shaded drawing in a unit square, see [`thumbnail`].
    pub thumb: egui::Mesh,
}

#[derive(Default)]
pub struct Library {
    pub parts: Vec<LibPart>,
}

fn parts_dir() -> Option<PathBuf> {
    crate::app::data_dir().map(|d| d.join("parts"))
}

fn stencils_dir() -> Option<PathBuf> {
    crate::app::data_dir().map(|d| d.join("stencils"))
}

impl Library {
    /// Built-in parts and the user's saved parts.
    pub fn load() -> Self {
        let mut parts: Vec<LibPart> = mm_edit::builtin_parts().into_iter().map(|p| LibPart { thumb: thumbnail(&p.mesh, p.kind), part: Arc::new(p), file: None }).collect();
        let mut files: Vec<PathBuf> = parts_dir().and_then(|d| std::fs::read_dir(d).ok()).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "pmx")).collect();
        files.sort();
        for f in files {
            match load_part(&f) {
                Ok(p) => parts.push(LibPart { thumb: thumbnail(&p.mesh, p.kind), part: Arc::new(p), file: Some(f) }),
                Err(e) => log::warn!("skipping part {}: {e:#}", f.display()),
            }
        }
        Self { parts }
    }

    pub fn find(&self, name: &str) -> Option<Arc<Part>> {
        self.parts.iter().find(|p| p.part.name == name).map(|p| p.part.clone())
    }
}

fn load_part(path: &std::path::Path) -> Result<Part> {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("part");
    let (name, kind) = stem.rsplit_once('.').context("part file name must be <name>.<open|solid>.pmx")?;
    let mesh = crate::project::mesh_from_bytes(&std::fs::read(path)?)?;
    Ok(match kind {
        "open" => Part::open(name, mesh)?,
        "solid" => Part::solid(name, mesh)?,
        k => bail!("unknown part kind '{k}'"),
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
    Ok(name)
}

/// The user's stencils (name, outline).
pub fn user_stencils() -> Vec<(String, Vec<DVec2>)> {
    let mut out: Vec<(String, Vec<DVec2>)> = stencils_dir()
        .and_then(|d| std::fs::read_dir(d).ok())
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .filter_map(|p| {
            let pts: Vec<DVec2> = std::fs::read_to_string(&p)
                .ok()?
                .lines()
                .filter_map(|l| {
                    let mut it = l.split_whitespace().map(|x| x.parse::<f64>());
                    Some(DVec2::new(it.next()?.ok()?, it.next()?.ok()?))
                })
                .collect();
            let name = p.file_stem()?.to_string_lossy().into_owned();
            (pts.len() >= 3).then_some((name, pts))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Save a stencil outline (any scale) to the user's stencils; returns its name.
pub fn save_stencil(outline: &[DVec2]) -> Result<String> {
    let dir = stencils_dir().context("no data directory")?;
    std::fs::create_dir_all(&dir)?;
    let name = free_name("My stencil", |n| dir.join(format!("{n}.txt")).exists());
    let text: String = outline.iter().map(|p| format!("{} {}\n", p.x, p.y)).collect();
    std::fs::write(dir.join(format!("{name}.txt")), text)?;
    Ok(name)
}

/// Stencil from a closed loop of 3D points: projected onto its best-fit plane, seen from
/// `view` (the side the selection faces), centred, unit radius, at most `max_points` points.
pub fn stencil_from_loop(pts: &[DVec3], view: DVec3, max_points: usize) -> Option<Vec<DVec2>> {
    if pts.len() < 3 {
        return None;
    }
    let c = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
    let mut cov = [[0.0; 3]; 3];
    for p in pts {
        mm_core::linalg::add_outer(&mut cov, *p - c, 1.0);
    }
    let (_, vecs) = mm_core::linalg::sym_eigen3(cov);
    let n = if vecs[0].dot(view) < 0.0 { -vecs[0] } else { vecs[0] };
    let (u, _) = mm_core::geom::orthonormal_basis(n);
    let v = n.cross(u);
    let step = pts.len().div_ceil(max_points).max(1);
    let q: Vec<DVec2> = pts.iter().step_by(step).map(|p| DVec2::new((*p - c).dot(u), (*p - c).dot(v))).collect();
    let r = q.iter().map(|p| p.length()).fold(0.0, f64::max);
    (r > 0.0 && q.len() >= 3).then(|| q.iter().map(|p| *p / r).collect())
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
    let (sub, _) = mesh.submesh(tris);
    match what {
        ConvertTo::OpenPart => Ok(format!("Added open part \"{}\" to Meshmix > My parts", save_part(sub, PartKind::Open)?)),
        ConvertTo::SolidPart => Ok(format!("Added solid part \"{}\" to Meshmix > My parts", save_part(sub, PartKind::Solid)?)),
        ConvertTo::Stamp => {
            let loops = mm_core::topology::boundary_loops(&sub);
            let Some(lp) = loops.iter().max_by_key(|l| l.vertices.len()) else { bail!("the selection has no border to trace") };
            let pts: Vec<DVec3> = lp.vertices.iter().map(|&v| sub.vertex(v)).collect();
            let view: DVec3 = sub.triangle_ids().map(|t| sub.tri_normal(t) * sub.tri_area(t)).sum();
            let outline = stencil_from_loop(&pts, view, 256).context("the selection's border is degenerate")?;
            Ok(format!("Added stencil \"{}\" to the Stamp tool", save_stencil(&outline)?))
        }
    }
}

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

/// Draw a thumbnail into `rect`.
pub fn paint_thumbnail(painter: &egui::Painter, rect: egui::Rect, thumb: &egui::Mesh) {
    let mut m = thumb.clone();
    for v in &mut m.vertices {
        v.pos = rect.min + v.pos.to_vec2() * rect.width();
    }
    painter.add(egui::Shape::mesh(m));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stencil_from_a_tilted_square_loop() {
        let q = glam::DQuat::from_axis_angle(DVec3::X, 0.7);
        let pts: Vec<DVec3> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].iter().map(|&(x, y)| q * DVec3::new(x, y, 0.0) * 3.0 + DVec3::splat(5.0)).collect();
        let s = stencil_from_loop(&pts, q * DVec3::Z, 100).unwrap();
        assert_eq!(s.len(), 4);
        assert!(s.iter().all(|p| (p.length() - 1.0).abs() < 1e-9));
        // Seen from the side it faces, the loop keeps its counter-clockwise order.
        let area: f64 = (0..4).map(|i| s[i].perp_dot(s[(i + 1) % 4])).sum();
        assert!(area > 0.0);
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
