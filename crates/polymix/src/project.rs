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
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// Corner UVs in triangle order (see [`crate::scene::UvSet`]).
    pub uvs: Option<Vec<[[f32; 2]; 3]>>,
    pub pivots: Vec<crate::scene::Pivot>,
    /// The texture image's file (the image itself stays where it is).
    pub texture: Option<std::path::PathBuf>,
    /// A reference object (see [`crate::scene::SceneObject::reference`]).
    pub reference: bool,
}

const UV_MAGIC: &[u8; 8] = b"PMXUV001";

fn uvs_to_bytes(uvs: &[[[f32; 2]; 3]]) -> Vec<u8> {
    uv_corners_to_bytes(uvs.len(), uvs.iter())
}

/// [`uvs_to_bytes`] of `n` triangles' corner UVs, given in triangle order.
fn uv_corners_to_bytes<'a>(n: usize, uvs: impl Iterator<Item = &'a [[f32; 2]; 3]>) -> Vec<u8> {
    let mut b = Vec::with_capacity(12 + n * 24);
    b.extend_from_slice(UV_MAGIC);
    b.extend_from_slice(&(n as u32).to_le_bytes());
    for c in uvs {
        put_le(&mut b, c.as_flattened());
    }
    b
}

/// Append numbers as little-endian bytes: in one copy where memory already holds them so.
fn put_le<T: bytemuck::Pod + ToLe>(b: &mut Vec<u8>, values: &[T]) {
    if cfg!(target_endian = "little") {
        b.extend_from_slice(bytemuck::cast_slice(values));
    } else {
        for v in values {
            v.put_le(b);
        }
    }
}

/// A number that can be written as little-endian bytes.
trait ToLe {
    fn put_le(&self, b: &mut Vec<u8>);
}

macro_rules! to_le {
    ($($t:ty),*) => {$(
        impl ToLe for $t {
            fn put_le(&self, b: &mut Vec<u8>) {
                b.extend_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}
to_le!(f32, f64, u32, i32);

fn uvs_from_bytes(b: &[u8]) -> Result<Vec<[[f32; 2]; 3]>> {
    if b.len() < 12 || &b[..8] != UV_MAGIC {
        bail!("not Polymix UVs");
    }
    let n = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
    let data = &b[12..];
    if data.len() != n * 24 {
        bail!("truncated UVs");
    }
    let f = |i: usize| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
    Ok((0..n).map(|t| [0, 1, 2].map(|c| [f(t * 6 + c * 2), f(t * 6 + c * 2 + 1)])).collect())
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
    let mut b = Vec::with_capacity(17 + pos.len() * 24 + tris.len() * 16 + colors.as_ref().map_or(0, |c| c.len() * 12));
    b.extend_from_slice(MESH_MAGIC);
    b.extend_from_slice(&(pos.len() as u32).to_le_bytes());
    b.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    b.push(colors.is_some() as u8);
    // x, y, z of each position in turn, as `DVec3` holds them.
    let coords: &[f64] = bytemuck::cast_slice(&pos);
    put_le(&mut b, coords);
    put_le(&mut b, tris.as_flattened());
    put_le(&mut b, &groups);
    if let Some(c) = &colors {
        put_le(&mut b, c.as_flattened());
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
    // Every value is there (checked just above): take them in bulk.
    let (pos_b, rest) = b[17..].split_at(nv * 24);
    let (tri_b, rest) = rest.split_at(nt * 12);
    let (group_b, color_b) = rest.split_at(nt * 4);
    let f64_of = |c: &[u8]| f64::from_le_bytes(c.try_into().expect("8 bytes"));
    let u32_of = |c: &[u8]| u32::from_le_bytes(c.try_into().expect("4 bytes"));
    let f32_of = |c: &[u8]| f32::from_le_bytes(c.try_into().expect("4 bytes"));
    let pos: Vec<DVec3> = pos_b.as_chunks::<24>().0.iter().map(|p| DVec3::new(f64_of(&p[..8]), f64_of(&p[8..16]), f64_of(&p[16..]))).collect();
    let mut tris = Vec::with_capacity(nt);
    for t in tri_b.as_chunks::<12>().0 {
        let t = [u32_of(&t[..4]), u32_of(&t[4..8]), u32_of(&t[8..])];
        if t.iter().any(|&v| v as usize >= nv) {
            bail!("triangle refers to a missing vertex");
        }
        tris.push(t);
    }
    let groups: Vec<i32> = group_b.as_chunks::<4>().0.iter().map(|g| u32_of(g) as i32).collect();
    let (mut mesh, _) = Mesh::from_triangles(&pos, &tris, Some(&groups));
    if flag[0] != 0 {
        for (v, c) in color_b.as_chunks::<12>().0.iter().take(nv).enumerate() {
            if mesh.is_vertex(v as u32) {
                mesh.set_vertex_color(v as u32, [f32_of(&c[..4]), f32_of(&c[4..8]), f32_of(&c[8..])]);
            }
        }
    }
    Ok(mesh)
}

/// One object as Save and autosave write it: its mesh and UVs shared with the scene, so
/// that taking the snapshot copies nothing (the job writing the file does the work).
pub struct SaveObject {
    pub name: String,
    pub mesh: Arc<Mesh>,
    pub visible: bool,
    pub color: Vec4,
    pub material: usize,
    pub frame: DQuat,
    pub group_names: Vec<(i32, String)>,
    /// The object's UVs, only if they belong to `mesh`'s triangles (see
    /// [`crate::scene::SceneObject::uvs`]); written in triangle order.
    pub uvs: Option<Arc<crate::scene::UvSet>>,
    pub pivots: Vec<crate::scene::Pivot>,
    pub texture: Option<PathBuf>,
    pub reference: bool,
}

/// A scene to save with [`save_shared`]: [`Project`] with [`SaveObject`]s.
pub struct SaveProject {
    pub objects: Vec<SaveObject>,
    pub active: Option<usize>,
    pub target: Option<usize>,
    pub view: Option<ViewState>,
}

/// How hard the archive's entries are compressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    /// Deflate at its usual level (Save).
    Default,
    /// Deflate at its fastest level: larger files, about half the time (autosave).
    Fast,
    /// No compression: quickest, for a save that must be done at once (after a crash).
    Stored,
}

/// What the writer needs of one object, from a [`ProjectObject`] or a [`SaveObject`].
struct ObjectOut<'a> {
    name: &'a str,
    mesh: &'a Mesh,
    visible: bool,
    color: Vec4,
    material: usize,
    frame: DQuat,
    group_names: &'a [(i32, String)],
    uvs: Option<UvsOut<'a>>,
    pivots: &'a [crate::scene::Pivot],
    texture: Option<&'a Path>,
    reference: bool,
}

/// An object's UVs as the writer gets them.
#[derive(Clone, Copy)]
enum UvsOut<'a> {
    /// One entry per triangle, in triangle order.
    Compact(&'a [[[f32; 2]; 3]]),
    /// By triangle id (a [`crate::scene::UvSet`] of the mesh), written in triangle order.
    ById(&'a [[[f32; 2]; 3]]),
}

impl<'a> From<&'a ProjectObject> for ObjectOut<'a> {
    fn from(o: &'a ProjectObject) -> Self {
        Self {
            name: &o.name,
            mesh: &o.mesh,
            visible: o.visible,
            color: o.color,
            material: o.material,
            frame: o.frame,
            group_names: &o.group_names,
            uvs: o.uvs.as_deref().map(UvsOut::Compact),
            pivots: &o.pivots,
            texture: o.texture.as_deref(),
            reference: o.reference,
        }
    }
}

impl<'a> From<&'a SaveObject> for ObjectOut<'a> {
    fn from(o: &'a SaveObject) -> Self {
        Self {
            name: &o.name,
            mesh: &o.mesh,
            visible: o.visible,
            color: o.color,
            material: o.material,
            frame: o.frame,
            group_names: &o.group_names,
            uvs: o.uvs.as_deref().filter(|u| u.topo == o.mesh.topology_version()).map(|u| UvsOut::ById(&u.corners)),
            pivots: &o.pivots,
            texture: o.texture.as_deref(),
            reference: o.reference,
        }
    }
}

/// Text of the manifest.
fn manifest(view: Option<&ViewState>, active: Option<usize>, target: Option<usize>, objects: &[ObjectOut]) -> String {
    let mut s = String::from("polymix-project 1\n");
    let one_line = |x: &str| x.replace(['\n', '\r'], " ");
    if let Some(v) = view {
        let q = v.orient;
        s += &format!("view {} {} {} {} {} {} {} {}\n", v.target.x, v.target.y, v.target.z, v.distance, q.x, q.y, q.z, q.w);
    }
    if let Some(i) = active {
        s += &format!("active {i}\n");
    }
    if let Some(i) = target {
        s += &format!("target {i}\n");
    }
    for (i, o) in objects.iter().enumerate() {
        s += "object\n";
        s += &format!("name {}\n", one_line(o.name));
        s += &format!("file objects/{i}.bin\n");
        s += &format!("visible {}\n", o.visible as u8);
        s += &format!("color {} {} {} {}\n", o.color.x, o.color.y, o.color.z, o.color.w);
        s += &format!("material {}\n", o.material);
        s += &format!("frame {} {} {} {}\n", o.frame.x, o.frame.y, o.frame.z, o.frame.w);
        for (g, n) in o.group_names {
            s += &format!("group {g} {}\n", one_line(n));
        }
        if o.uvs.is_some() {
            s += &format!("uvs objects/{i}.uv\n");
        }
        for p in o.pivots {
            let (x, q) = (p.position, p.frame);
            s += &format!("pivot {} {} {} {} {} {} {}\n", x.x, x.y, x.z, q.x, q.y, q.z, q.w);
        }
        if let Some(t) = o.texture {
            s += &format!("texture {}\n", one_line(&t.to_string_lossy()));
        }
        if o.reference {
            s += "reference 1\n";
        }
        s += "end\n";
    }
    s
}

/// Write `project` to `path` (via a temporary file, so a failed save never leaves a broken
/// project behind).
#[allow(dead_code)] // For code holding a `Project` (tests); the app saves with `save_shared`.
pub fn save(path: &Path, project: &Project) -> Result<()> {
    let objects: Vec<ObjectOut> = project.objects.iter().map(ObjectOut::from).collect();
    write_project(path, project.view.as_ref(), project.active, project.target, &objects, Compression::Default)
}

/// [`save`] for a snapshot of the scene that shares its meshes: the same file, compressed as
/// asked.
pub fn save_shared(path: &Path, project: &SaveProject, compression: Compression) -> Result<()> {
    let objects: Vec<ObjectOut> = project.objects.iter().map(ObjectOut::from).collect();
    write_project(path, project.view.as_ref(), project.active, project.target, &objects, compression)
}

fn write_project(path: &Path, view: Option<&ViewState>, active: Option<usize>, target: Option<usize>, objects: &[ObjectOut], compression: Compression) -> Result<()> {
    let mut buf = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buf);
        let opts = match compression {
            Compression::Default => zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated),
            Compression::Fast => zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated).compression_level(Some(1)),
            Compression::Stored => zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
        };
        zip.start_file("manifest.txt", opts)?;
        zip.write_all(manifest(view, active, target, objects).as_bytes())?;
        for (i, o) in objects.iter().enumerate() {
            zip.start_file(format!("objects/{i}.bin"), opts)?;
            zip.write_all(&mesh_to_bytes(o.mesh))?;
            if let Some(uvs) = o.uvs {
                zip.start_file(format!("objects/{i}.uv"), opts)?;
                zip.write_all(&match uvs {
                    UvsOut::Compact(uvs) => uvs_to_bytes(uvs),
                    UvsOut::ById(corners) => uv_corners_to_bytes(o.mesh.triangle_count(), o.mesh.triangle_ids().map(|t| &corners[t as usize])),
                })?;
            }
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
    let mut uv_file: Option<String> = None;
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
                    uvs: None,
                    pivots: Vec::new(),
                    texture: None,
                    reference: false,
                };
                cur = Some((o, String::new()));
                uv_file = None;
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
            ("uvs", Some(_)) => uv_file = Some(rest.trim().to_string()),
            ("texture", Some((o, _))) => o.texture = Some(std::path::PathBuf::from(rest)),
            ("reference", Some((o, _))) => o.reference = rest.trim() != "0",
            ("pivot", Some((o, _))) => {
                if let [x, y, z, qx, qy, qz, qw] = nums(rest)[..] {
                    o.pivots.push(crate::scene::Pivot { position: DVec3::new(x, y, z), frame: DQuat::from_xyzw(qx, qy, qz, qw).normalize() });
                }
            }
            ("end", Some(_)) => {
                let (mut o, file) = cur.take().unwrap();
                let mut bytes = Vec::new();
                zip.by_name(&file).with_context(|| format!("project is missing {file}"))?.read_to_end(&mut bytes)?;
                o.mesh = mesh_from_bytes(&bytes).with_context(|| format!("in {file}"))?;
                if let Some(f) = uv_file.take() {
                    let mut b = Vec::new();
                    zip.by_name(&f).with_context(|| format!("project is missing {f}"))?.read_to_end(&mut b)?;
                    let uvs = uvs_from_bytes(&b).with_context(|| format!("in {f}"))?;
                    // UVs that don't match the triangles are dropped rather than misapplied.
                    o.uvs = (uvs.len() == o.mesh.triangle_count()).then_some(uvs);
                }
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
                    uvs: Some((0..a.triangle_count()).map(|t| [[t as f32, 0.5], [0.25, 0.125], [1.0, t as f32 * 0.001]]).collect()),
                    pivots: vec![crate::scene::Pivot { position: DVec3::new(1.0, -2.0, 0.5), frame: DQuat::from_rotation_x(0.7) }],
                    texture: Some(std::path::PathBuf::from("/some dir/skin.png")),
                    reference: false,
                },
                ProjectObject {
                    name: "box".into(),
                    mesh: b.clone(),
                    visible: false,
                    color: Vec4::ONE,
                    material: 0,
                    frame: DQuat::IDENTITY,
                    group_names: Vec::new(),
                    uvs: None,
                    pivots: Vec::new(),
                    texture: None,
                    reference: true,
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
        assert_eq!(o.uvs, p.objects[0].uvs);
        assert_eq!(o.pivots.len(), 1);
        assert!((o.pivots[0].position - DVec3::new(1.0, -2.0, 0.5)).length() < 1e-12);
        assert!(o.pivots[0].frame.angle_between(DQuat::from_rotation_x(0.7)) < 1e-9);
        assert!(q.objects[1].uvs.is_none());
        assert_eq!(q.objects[0].texture.as_deref(), Some(std::path::Path::new("/some dir/skin.png")));
        assert!(!q.objects[0].reference && q.objects[1].reference);
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

    /// `mesh_to_bytes` as it was: one value at a time.
    fn to_bytes_before(m: &Mesh) -> Vec<u8> {
        let (pos, tris, groups) = m.to_arrays();
        let colors = m.colors_compact();
        let mut b = Vec::new();
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

    /// `mesh_from_bytes` as it was: one value at a time from a cursor.
    fn from_bytes_before(b: &[u8]) -> Result<Mesh> {
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

    /// Positions, triangles, groups and colours (as bits), and the id ranges.
    type MeshBits = (Vec<[u64; 3]>, Vec<[u32; 3]>, Vec<i32>, Option<Vec<[u32; 3]>>, usize, usize);

    /// Everything of a mesh, as bits.
    fn mesh_bits(m: &Mesh) -> MeshBits {
        let (p, t, g) = m.to_arrays();
        let c = m.colors_compact().map(|c| c.iter().map(|x| x.map(f32::to_bits)).collect());
        (p.iter().map(|v| v.to_array().map(f64::to_bits)).collect(), t, g, c, m.max_vid(), m.max_tid())
    }

    /// Meshes with holes (sparse ids), colours, groups, odd values; and an empty one.
    fn test_meshes() -> Vec<Mesh> {
        let mut a = mm_core::primitives::icosphere(2.0, 3);
        for t in a.triangle_ids().step_by(17).collect::<Vec<_>>() {
            a.remove_triangle(t, true).unwrap();
        }
        for (k, t) in a.triangle_ids().collect::<Vec<_>>().into_iter().enumerate() {
            a.set_tri_group(t, (k % 7) as i32 - 3);
        }
        let v = a.vertex_ids().nth(4).unwrap();
        a.set_vertex_color(v, [0.1, -0.0, 7.5]);
        a.set_vertex(v, DVec3::new(-0.0, 1e-300, -1e300));
        let mut b = mm_core::primitives::cube(3.0, 4);
        b.transform(|p| p * 1.0000001 + DVec3::new(0.1, -0.2, 1e6));
        vec![a, b, mm_core::primitives::plane_grid(1.0, 2.0, 3, 5), Mesh::new()]
    }

    #[test]
    fn mesh_bytes_are_those_written_and_read_one_value_at_a_time() {
        for (k, m) in test_meshes().iter().enumerate() {
            let b = mesh_to_bytes(m);
            assert_eq!(b, to_bytes_before(m), "mesh {k}");
            assert_eq!(mesh_bits(&mesh_from_bytes(&b).unwrap()), mesh_bits(&from_bytes_before(&b).unwrap()), "mesh {k}");
            // Trailing bytes are ignored, a short or broken buffer fails the same way.
            let mut longer = b.clone();
            longer.extend_from_slice(&[1, 2, 3, 4, 5]);
            assert_eq!(mesh_bits(&mesh_from_bytes(&longer).unwrap()), mesh_bits(&from_bytes_before(&longer).unwrap()));
            for cut in [0, 3, 8, 12, 16, 17, b.len() / 2, b.len().saturating_sub(1)] {
                if cut >= b.len() {
                    continue;
                }
                let (x, y) = (mesh_from_bytes(&b[..cut]), from_bytes_before(&b[..cut]));
                assert_eq!(x.err().map(|e| format!("{e:#}")), y.err().map(|e| format!("{e:#}")), "mesh {k} cut at {cut}");
            }
        }
        // A triangle pointing past the vertices, and another magic.
        let mut bad = mesh_to_bytes(&mm_core::primitives::cube(1.0, 1));
        let at = 17 + 8 * 24 + 4;
        bad[at..at + 4].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(format!("{:#}", mesh_from_bytes(&bad).unwrap_err()), format!("{:#}", from_bytes_before(&bad).unwrap_err()));
        bad[0] = b'X';
        assert_eq!(format!("{:#}", mesh_from_bytes(&bad).unwrap_err()), "not a Polymix mesh");
    }

    /// `uvs_to_bytes` as it was.
    fn uvs_to_bytes_before(uvs: &[[[f32; 2]; 3]]) -> Vec<u8> {
        let mut b = Vec::with_capacity(12 + uvs.len() * 24);
        b.extend_from_slice(UV_MAGIC);
        b.extend_from_slice(&(uvs.len() as u32).to_le_bytes());
        for x in uvs.iter().flatten().flatten() {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    /// `manifest` as it was.
    fn manifest_before(p: &Project) -> String {
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
            if o.uvs.is_some() {
                s += &format!("uvs objects/{i}.uv\n");
            }
            for p in &o.pivots {
                let (x, q) = (p.position, p.frame);
                s += &format!("pivot {} {} {} {} {} {} {}\n", x.x, x.y, x.z, q.x, q.y, q.z, q.w);
            }
            if let Some(t) = &o.texture {
                s += &format!("texture {}\n", one_line(&t.to_string_lossy()));
            }
            if o.reference {
                s += "reference 1\n";
            }
            s += "end\n";
        }
        s
    }

    /// `save` as it was, with the writers as they were.
    fn save_before(path: &Path, project: &Project) -> Result<()> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("manifest.txt", opts)?;
            zip.write_all(manifest_before(project).as_bytes())?;
            for (i, o) in project.objects.iter().enumerate() {
                zip.start_file(format!("objects/{i}.bin"), opts)?;
                zip.write_all(&to_bytes_before(&o.mesh))?;
                if let Some(uvs) = &o.uvs {
                    zip.start_file(format!("objects/{i}.uv"), opts)?;
                    zip.write_all(&uvs_to_bytes_before(uvs))?;
                }
            }
            zip.finish()?;
        }
        let tmp = path.with_extension(format!("{EXTENSION}.tmp"));
        std::fs::write(&tmp, buf.into_inner()).with_context(|| format!("failed to write {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("failed to save {}", path.display()))?;
        Ok(())
    }

    /// The entries of a project file, in order, with their uncompressed bytes.
    fn entries(path: &Path) -> Vec<(String, Vec<u8>)> {
        let mut zip = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
        (0..zip.len())
            .map(|i| {
                let mut f = zip.by_index(i).unwrap();
                let mut b = Vec::new();
                f.read_to_end(&mut b).unwrap();
                (f.name().expect("entry name").to_string(), b)
            })
            .collect()
    }

    #[test]
    fn shared_snapshots_save_what_copies_saved() {
        let meshes = test_meshes();
        let pivot = crate::scene::Pivot { position: DVec3::new(1.0, 2.0, 3.0), frame: DQuat::from_rotation_y(0.4) };
        // The scene's objects: UVs by triangle id (one set for an older topology, not saved).
        let mut stale = mm_core::primitives::cube(1.0, 2);
        let old_uvs = Arc::new(crate::scene::UvSet::new(&stale, vec![[[0.5; 2]; 3]; stale.max_tid()]));
        let t = stale.triangle_ids().next().unwrap();
        stale.remove_triangle(t, true).unwrap();
        let mut shared: Vec<SaveObject> = meshes
            .iter()
            .enumerate()
            .map(|(k, m)| SaveObject {
                name: format!("object {k}\nwith a line break"),
                mesh: Arc::new(m.clone()),
                visible: k % 2 == 0,
                color: Vec4::new(0.1 * k as f32, 0.5, 1.0, 0.75),
                material: k,
                frame: DQuat::from_rotation_x(k as f64 * 0.1),
                group_names: vec![(2, "wall".into()), (-1, "outlet 2".into())],
                uvs: (k == 0).then(|| Arc::new(crate::scene::UvSet::new(m, (0..m.max_tid()).map(|t| [[t as f32, -0.0], [0.5, 1.0], [0.25, t as f32 * 1e-3]]).collect()))),
                pivots: if k == 1 { vec![pivot] } else { Vec::new() },
                texture: (k == 0).then(|| PathBuf::from("/a dir/skin.png")),
                reference: k == 2,
            })
            .collect();
        shared.push(SaveObject { name: "stale uvs".into(), mesh: Arc::new(stale), visible: true, color: Vec4::ONE, material: 0, frame: DQuat::IDENTITY, group_names: Vec::new(), uvs: Some(old_uvs), pivots: Vec::new(), texture: None, reference: false });
        // The same as the copies Save and autosave made before: UVs in triangle order, only
        // when they belong to the mesh.
        let copied = Project {
            objects: shared
                .iter()
                .map(|o| ProjectObject {
                    name: o.name.clone(),
                    mesh: (*o.mesh).clone(),
                    visible: o.visible,
                    color: o.color,
                    material: o.material,
                    frame: o.frame,
                    group_names: o.group_names.clone(),
                    uvs: o.uvs.as_ref().filter(|u| u.topo == o.mesh.topology_version()).map(|u| o.mesh.triangle_ids().map(|t| u.corners[t as usize]).collect()),
                    pivots: o.pivots.clone(),
                    texture: o.texture.clone(),
                    reference: o.reference,
                })
                .collect(),
            active: Some(1),
            target: Some(0),
            view: Some(ViewState { target: DVec3::new(1.0, -2.0, 3.0), distance: 12.5, orient: DQuat::from_rotation_z(0.2) }),
        };
        let snapshot = SaveProject { objects: shared, active: copied.active, target: copied.target, view: copied.view };
        let dir = std::env::temp_dir().join(format!("polymix-shared-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Written as before, byte for byte; compressed otherwise, the same entries.
        let before = dir.join("before.polymix");
        save_before(&before, &copied).unwrap();
        let file = |p: &Path| std::fs::read(p).unwrap();
        let path = dir.join("copied.polymix");
        save(&path, &copied).unwrap();
        assert!(file(&path) == file(&before));
        for compression in [Compression::Default, Compression::Fast, Compression::Stored] {
            let path = dir.join(format!("{compression:?}.polymix"));
            save_shared(&path, &snapshot, compression).unwrap();
            assert!(entries(&path) == entries(&before), "{compression:?}");
            assert_eq!(file(&path) == file(&before), compression == Compression::Default, "{compression:?}");
            let (a, b) = (load(&path).unwrap(), load(&before).unwrap());
            assert_eq!(a.objects.len(), 5);
            for (x, y) in a.objects.iter().zip(&b.objects) {
                assert_eq!(mesh_bits(&x.mesh), mesh_bits(&y.mesh));
                assert_eq!((&x.name, &x.uvs, &x.group_names, &x.texture, x.reference), (&y.name, &y.uvs, &y.group_names, &y.texture, y.reference));
            }
        }
        let size = |c: Compression| std::fs::metadata(dir.join(format!("{c:?}.polymix"))).unwrap().len();
        assert!(size(Compression::Fast) < size(Compression::Stored));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn objects_and_errors_come_in_manifest_order() {
        let dir = std::env::temp_dir().join(format!("polymix-load-order-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let objects: Vec<ProjectObject> = test_meshes()
            .into_iter()
            .chain([mm_core::primitives::icosphere(1.0, 4), mm_core::primitives::cylinder(1.0, 2.0, 32, 8, true)])
            .enumerate()
            .map(|(k, mesh)| ProjectObject {
                name: format!("o{k}"),
                uvs: (k % 2 == 1).then(|| vec![[[k as f32, 0.5]; 3]; mesh.triangle_count()]),
                mesh,
                visible: true,
                color: Vec4::ONE,
                material: 0,
                frame: DQuat::IDENTITY,
                group_names: Vec::new(),
                pivots: Vec::new(),
                texture: None,
                reference: false,
            })
            .collect();
        let path = dir.join("many.polymix");
        save(&path, &Project { objects: objects.clone(), ..Default::default() }).unwrap();
        let p = load(&path).unwrap();
        assert_eq!(p.objects.len(), objects.len());
        for (a, b) in p.objects.iter().zip(&objects) {
            assert_eq!(a.name, b.name);
            assert_eq!(mesh_bits(&a.mesh), mesh_bits(&mesh_from_bytes(&mesh_to_bytes(&b.mesh)).unwrap()));
            assert_eq!(a.uvs, b.uvs);
        }
        // Hand-made archives: an object's error comes before a manifest error after it, and
        // the first broken object's error before a later one's.
        let write = |name: &str, manifest: &str, files: &[(&str, &[u8])]| {
            let path = dir.join(name);
            let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
            let opts = zip::write::SimpleFileOptions::default();
            zip.start_file("manifest.txt", opts).unwrap();
            zip.write_all(manifest.as_bytes()).unwrap();
            for (n, b) in files {
                zip.start_file(*n, opts).unwrap();
                zip.write_all(b).unwrap();
            }
            zip.finish().unwrap();
            path
        };
        let good = mesh_to_bytes(&mm_core::primitives::cube(1.0, 1));
        let obj = |k: usize| format!("object\nname o{k}\nfile objects/{k}.bin\nend\n");
        let head = "polymix-project 1\n";
        let missing = write("missing.polymix", &format!("{head}{}{}{}bogus line\n", obj(0), obj(1), obj(2)), &[("objects/0.bin", &good), ("objects/2.bin", b"junk, not a mesh at all")]);
        let e = format!("{:#}", load(&missing).unwrap_err());
        assert!(e.starts_with("project is missing objects/1.bin"), "{e}");
        let broken = write("broken.polymix", &format!("{head}{}{}bogus line\n", obj(0), obj(1)), &[("objects/0.bin", &good), ("objects/1.bin", b"junk, not a mesh at all")]);
        assert_eq!(format!("{:#}", load(&broken).unwrap_err()), "in objects/1.bin: not a Polymix mesh");
        let manifest = write("manifest.polymix", &format!("{head}{}bogus line\n{}", obj(0), obj(1)), &[("objects/0.bin", &good), ("objects/1.bin", b"junk, not a mesh at all")]);
        assert_eq!(format!("{:#}", load(&manifest).unwrap_err()), "unexpected 'bogus' in the project manifest");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
