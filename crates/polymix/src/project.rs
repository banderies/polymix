//! Polymix project files (`.polymix`): the whole scene — every object's mesh (with face
//! groups and vertex colours), name, visibility, colour, material, frame and patch names,
//! the active object, the target surface and the view. No undo history.
//!
//! The file is a zip archive holding `manifest.txt` (one `key value` per line) and one
//! lossless binary mesh per object (`objects/N.bin`).

use crate::camera::ViewState;
use anyhow::{anyhow, bail, Context, Result};
use glam::{DQuat, DVec3, Vec4};
use mm_core::Mesh;
use std::io::{Cursor, Read, Write};
use std::path::Path;

pub const EXTENSION: &str = "polymix";
const MESH_MAGIC: &[u8; 8] = b"PMXMESH1";

/// One object as stored in a project.
#[derive(Clone, Debug)]
pub struct ProjectObject {
    pub name: String,
    pub mesh: Mesh,
    pub visible: bool,
    pub color: Vec4,
    pub material: usize,
    pub frame: DQuat,
    pub group_names: Vec<(i32, String)>,
}

/// A loaded project.
#[derive(Clone, Debug, Default)]
pub struct Project {
    pub objects: Vec<ProjectObject>,
    /// Index of the active object and of the target surface.
    pub active: Option<usize>,
    pub target: Option<usize>,
    pub view: Option<ViewState>,
}

/// Lossless binary mesh: positions (f64), triangles, face groups, optional vertex colours.
pub fn mesh_to_bytes(m: &Mesh) -> Vec<u8> {
    let (pos, tris, groups) = m.to_arrays();
    let colors = m.colors_compact();
    let mut b = Vec::with_capacity(16 + pos.len() * 36 + tris.len() * 16);
    b.extend_from_slice(MESH_MAGIC);
    b.extend_from_slice(&(pos.len() as u32).to_le_bytes());
    b.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    b.push(colors.is_some() as u8);
    for p in &pos {
        for c in p.to_array() {
            b.extend_from_slice(&c.to_le_bytes());
        }
    }
    for t in &tris {
        for v in t {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    for g in &groups {
        b.extend_from_slice(&g.to_le_bytes());
    }
    for c in colors.iter().flatten() {
        for x in c {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    b
}

pub fn mesh_from_bytes(b: &[u8]) -> Result<Mesh> {
    let mut r = Cursor::new(b);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MESH_MAGIC {
        bail!("not a Polymix mesh");
    }
    let mut u32b = [0u8; 4];
    let mut read_u32 = |r: &mut Cursor<&[u8]>| -> Result<u32> {
        r.read_exact(&mut u32b)?;
        Ok(u32::from_le_bytes(u32b))
    };
    let nv = read_u32(&mut r)? as usize;
    let nt = read_u32(&mut r)? as usize;
    let mut flag = [0u8; 1];
    r.read_exact(&mut flag)?;
    let need = nv * 24 + nt * 16 + if flag[0] != 0 { nv * 12 } else { 0 };
    if b.len() < 17 + need {
        bail!("truncated mesh data");
    }
    let mut f64b = [0u8; 8];
    let mut pos = Vec::with_capacity(nv);
    for _ in 0..nv {
        let mut c = [0.0; 3];
        for x in &mut c {
            r.read_exact(&mut f64b)?;
            *x = f64::from_le_bytes(f64b);
        }
        pos.push(DVec3::from_array(c));
    }
    let mut tris = Vec::with_capacity(nt);
    for _ in 0..nt {
        let t = [read_u32(&mut r)?, read_u32(&mut r)?, read_u32(&mut r)?];
        if t.iter().any(|&v| v as usize >= nv) {
            bail!("triangle refers to a missing vertex");
        }
        tris.push(t);
    }
    let mut groups = Vec::with_capacity(nt);
    for _ in 0..nt {
        groups.push(read_u32(&mut r)? as i32);
    }
    let (mut mesh, _) = Mesh::from_triangles(&pos, &tris, Some(&groups));
    if flag[0] != 0 {
        let mut f32b = [0u8; 4];
        for v in 0..nv {
            let mut c = [0.0f32; 3];
            for x in &mut c {
                r.read_exact(&mut f32b)?;
                *x = f32::from_le_bytes(f32b);
            }
            if mesh.is_vertex(v as u32) {
                mesh.set_vertex_color(v as u32, c);
            }
        }
    }
    Ok(mesh)
}

/// Text of the manifest.
fn manifest(p: &Project) -> String {
    let mut s = String::from("polymix-project 1\n");
    let one_line = |x: &str| x.replace(['\n', '\r'], " ");
    if let Some(v) = &p.view {
        let q = v.orient;
        s += &format!("view {} {} {} {} {} {} {} {}\n", v.target.x, v.target.y, v.target.z, v.distance, q.x, q.y, q.z, q.w);
    }
    if let Some(i) = p.active {
        s += &format!("active {i}\n");
    }
    if let Some(i) = p.target {
        s += &format!("target {i}\n");
    }
    for (i, o) in p.objects.iter().enumerate() {
        s += "object\n";
        s += &format!("name {}\n", one_line(&o.name));
        s += &format!("file objects/{i}.bin\n");
        s += &format!("visible {}\n", o.visible as u8);
        s += &format!("color {} {} {} {}\n", o.color.x, o.color.y, o.color.z, o.color.w);
        s += &format!("material {}\n", o.material);
        s += &format!("frame {} {} {} {}\n", o.frame.x, o.frame.y, o.frame.z, o.frame.w);
        for (g, n) in &o.group_names {
            s += &format!("group {g} {}\n", one_line(n));
        }
        s += "end\n";
    }
    s
}

/// Write `project` to `path` (via a temporary file, so a failed save never leaves a broken
/// project behind).
pub fn save(path: &Path, project: &Project) -> Result<()> {
    let mut buf = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("manifest.txt", opts)?;
        zip.write_all(manifest(project).as_bytes())?;
        for (i, o) in project.objects.iter().enumerate() {
            zip.start_file(format!("objects/{i}.bin"), opts)?;
            zip.write_all(&mesh_to_bytes(&o.mesh))?;
        }
        zip.finish()?;
    }
    let tmp = path.with_extension(format!("{EXTENSION}.tmp"));
    std::fs::write(&tmp, buf.into_inner()).with_context(|| format!("failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

pub fn load(path: &Path) -> Result<Project> {
    let data = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(Cursor::new(data)).context("not a Polymix project")?;
    let mut text = String::new();
    zip.by_name("manifest.txt").context("project has no manifest")?.read_to_string(&mut text)?;
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("polymix-project 1") {
        bail!("unsupported project version");
    }
    let nums = |rest: &str| -> Vec<f64> { rest.split_whitespace().filter_map(|x| x.parse().ok()).collect() };
    let mut project = Project::default();
    let mut cur: Option<(ProjectObject, String)> = None;
    for line in lines {
        let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
        match (key, cur.as_mut()) {
            ("view", None) => {
                if let [x, y, z, d, qx, qy, qz, qw] = nums(rest)[..] {
                    project.view = Some(ViewState { target: DVec3::new(x, y, z), distance: d, orient: DQuat::from_xyzw(qx, qy, qz, qw).normalize() });
                }
            }
            ("active", None) => project.active = rest.trim().parse().ok(),
            ("target", None) => project.target = rest.trim().parse().ok(),
            ("object", None) => {
                let o = ProjectObject {
                    name: String::new(),
                    mesh: Mesh::new(),
                    visible: true,
                    color: Vec4::new(0.8, 0.8, 0.8, 1.0),
                    material: 0,
                    frame: DQuat::IDENTITY,
                    group_names: Vec::new(),
                };
                cur = Some((o, String::new()));
            }
            ("name", Some((o, _))) => o.name = rest.to_string(),
            ("file", Some((_, f))) => *f = rest.trim().to_string(),
            ("visible", Some((o, _))) => o.visible = rest.trim() != "0",
            ("color", Some((o, _))) => {
                if let [r, g, b, a] = nums(rest)[..] {
                    o.color = Vec4::new(r as f32, g as f32, b as f32, a as f32);
                }
            }
            ("material", Some((o, _))) => o.material = rest.trim().parse().unwrap_or(0),
            ("frame", Some((o, _))) => {
                if let [x, y, z, w] = nums(rest)[..] {
                    o.frame = DQuat::from_xyzw(x, y, z, w).normalize();
                }
            }
            ("group", Some((o, _))) => {
                let (g, n) = rest.split_once(' ').unwrap_or((rest, ""));
                if let Ok(g) = g.parse() {
                    o.group_names.push((g, n.to_string()));
                }
            }
            ("end", Some(_)) => {
                let (mut o, file) = cur.take().unwrap();
                let mut bytes = Vec::new();
                zip.by_name(&file).with_context(|| format!("project is missing {file}"))?.read_to_end(&mut bytes)?;
                o.mesh = mesh_from_bytes(&bytes).with_context(|| format!("in {file}"))?;
                project.objects.push(o);
            }
            ("", _) => {}
            (k, _) => return Err(anyhow!("unexpected '{k}' in the project manifest")),
        }
    }
    Ok(project)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_round_trip() {
        let mut a = mm_core::primitives::icosphere(1.0, 2);
        for t in a.triangle_ids().collect::<Vec<_>>() {
            let g = (a.tri_centroid(t).z > 0.0) as i32 + 3;
            a.set_tri_group(t, g);
        }
        a.set_vertex_color(0, [0.1, 0.2, 0.3]);
        let b = mm_core::primitives::cube(2.0, 2);
        let p = Project {
            objects: vec![
                ProjectObject {
                    name: "vessel with spaces".into(),
                    mesh: a.clone(),
                    visible: true,
                    color: Vec4::new(0.9, 0.4, 0.4, 0.7),
                    material: 5,
                    frame: DQuat::from_rotation_z(0.3),
                    group_names: vec![(3, "wall".into()), (4, "inlet 1".into())],
                },
                ProjectObject {
                    name: "box".into(),
                    mesh: b.clone(),
                    visible: false,
                    color: Vec4::ONE,
                    material: 0,
                    frame: DQuat::IDENTITY,
                    group_names: Vec::new(),
                },
            ],
            active: Some(0),
            target: Some(1),
            view: Some(ViewState { target: DVec3::new(1.0, 2.0, 3.0), distance: 42.0, orient: DQuat::from_rotation_y(0.5) }),
        };
        let dir = std::env::temp_dir().join(format!("polymix-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scene.polymix");
        save(&path, &p).unwrap();
        let q = load(&path).unwrap();
        assert_eq!(q.objects.len(), 2);
        assert_eq!((q.active, q.target), (Some(0), Some(1)));
        assert!((q.view.unwrap().distance - 42.0).abs() < 1e-12);
        let o = &q.objects[0];
        assert_eq!(o.name, "vessel with spaces");
        assert_eq!(o.material, 5);
        assert_eq!(o.group_names, vec![(3, "wall".to_string()), (4, "inlet 1".to_string())]);
        assert!(o.frame.angle_between(DQuat::from_rotation_z(0.3)) < 1e-9);
        assert_eq!(o.mesh.triangle_count(), a.triangle_count());
        // Positions are bit-exact, groups and colours kept.
        for (v, w) in a.vertex_ids().zip(o.mesh.vertex_ids()) {
            assert_eq!(a.vertex(v), o.mesh.vertex(w));
            assert_eq!(a.vertex_color(v), o.mesh.vertex_color(w));
        }
        for (s, t) in a.triangle_ids().zip(o.mesh.triangle_ids()) {
            assert_eq!(a.tri_group(s), o.mesh.tri_group(t));
        }
        assert!(!q.objects[1].visible);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
