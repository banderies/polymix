//! 3MF: core-spec meshes and components, build item transforms, model units, and
//! production-extension `p:path` references to other model parts in the package
//! (as written by Bambu Studio / PrusaSlicer). Materials and colors are ignored.

use crate::parse::{parse_f64, parse_i64, strip_bom};
use crate::{write_float, write_parallel, Raw};
use anyhow::{anyhow, bail, Context, Result};
use glam::{DAffine3, DVec3};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{BufWriter, Cursor, Read, Seek, Write};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Refuse to inflate any single package part beyond this (zip-bomb guard).
const MAX_PART_BYTES: u64 = 2 << 30;
const MAX_TRIANGLES: usize = 1 << 30;
const MAX_DEPTH: usize = 16;

#[derive(Clone, Debug)]
struct Ref {
    objectid: u32,
    xf: DAffine3,
    /// Package part holding the object (`None` = same part as the reference).
    path: Option<String>,
}

#[derive(Default)]
struct Object {
    vertices: Vec<DVec3>,
    triangles: Vec<[u32; 3]>,
    components: Vec<Ref>,
}

struct Model {
    /// Millimetres per model unit.
    unit: f64,
    objects: HashMap<u32, Object>,
    build: Vec<Ref>,
}

fn part_name(p: &str) -> String {
    p.trim_start_matches('/').to_string()
}

fn entry_index<R: Read + Seek>(zip: &ZipArchive<R>, name: &str) -> Option<usize> {
    zip.index_for_name(name).or_else(|| {
        // OPC part names are case-insensitive.
        (0..zip.len()).find(|&i| {
            zip.name_for_index(i)
                .and_then(|r| r.ok())
                .is_some_and(|n| n.trim_start_matches('/').eq_ignore_ascii_case(name))
        })
    })
}

fn read_entry<R: Read + Seek>(zip: &mut ZipArchive<R>, name: &str) -> Result<Vec<u8>> {
    let idx = entry_index(zip, name).ok_or_else(|| anyhow!("3MF package has no part '{name}'"))?;
    let f = zip.by_index(idx).with_context(|| format!("cannot open 3MF part '{name}'"))?;
    if f.size() > MAX_PART_BYTES {
        bail!("3MF part '{name}' is too large ({} bytes)", f.size());
    }
    let mut buf = Vec::with_capacity(f.size() as usize);
    f.take(MAX_PART_BYTES + 1)
        .read_to_end(&mut buf)
        .with_context(|| format!("cannot decompress 3MF part '{name}'"))?;
    if buf.len() as u64 > MAX_PART_BYTES {
        bail!("3MF part '{name}' is too large");
    }
    Ok(buf)
}

fn xml_str(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(strip_bom(bytes)).context("3MF XML is not valid UTF-8")
}

/// Iterate `(local name, raw value)` attribute pairs.
fn attrs<'a>(e: &'a BytesStart<'a>) -> impl Iterator<Item = Result<(&'a str, Cow<'a, str>)>> + 'a {
    let mut it = e.attributes();
    it.with_checks(false);
    it.map(|a| {
        let a = a.map_err(|e| anyhow!("bad XML attribute: {e}"))?;
        Ok((a.key.local_name().into_inner(), a.value))
    })
}

/// Locate the root model part through `_rels/.rels`, falling back to the usual name.
fn find_root_model<R: Read + Seek>(zip: &mut ZipArchive<R>) -> Result<String> {
    if entry_index(zip, "_rels/.rels").is_some() {
        let rels = read_entry(zip, "_rels/.rels")?;
        let mut reader = Reader::from_str(xml_str(&rels)?);
        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) | Ok(Event::Empty(e)) if e.local_name().as_ref() == "Relationship" => {
                    let (mut ty, mut target) = (None, None);
                    for a in attrs(&e) {
                        let (k, v) = a?;
                        match k {
                            "Type" => ty = Some(v.into_owned()),
                            "Target" => target = Some(v.into_owned()),
                            _ => {}
                        }
                    }
                    if let (Some(ty), Some(target)) = (ty, target) {
                        if ty.ends_with("/3dmodel") {
                            let name = part_name(&target);
                            if entry_index(zip, &name).is_some() {
                                return Ok(name);
                            }
                        }
                    }
                }
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
        }
    }
    if entry_index(zip, "3D/3dmodel.model").is_some() {
        return Ok("3D/3dmodel.model".to_string());
    }
    let any = (0..zip.len())
        .filter_map(|i| zip.name_for_index(i).and_then(|r| r.ok()).map(|n| n.into_owned()))
        .find(|n| n.to_ascii_lowercase().ends_with(".model"));
    any.ok_or_else(|| anyhow!("3MF package contains no 3D model part"))
}

fn unit_scale(unit: &str) -> Result<f64> {
    Ok(match unit {
        "micron" => 0.001,
        "millimeter" => 1.0,
        "centimeter" => 10.0,
        "inch" => 25.4,
        "foot" => 304.8,
        "meter" => 1000.0,
        _ => bail!("unknown 3MF unit '{unit}'"),
    })
}

fn parse_transform(s: &str) -> Result<DAffine3> {
    let mut m = [0.0f64; 12];
    let mut n = 0;
    for t in s.split_ascii_whitespace() {
        if n == 12 {
            bail!("3MF transform has more than 12 values");
        }
        m[n] = parse_f64(t.as_bytes()).ok_or_else(|| anyhow!("bad 3MF transform value '{t}'"))?;
        n += 1;
    }
    if n != 12 {
        bail!("3MF transform needs 12 values, got {n}");
    }
    // 3MF matrices act on row vectors: p' = [x y z 1] * M (4x3), so rows of M are columns here.
    Ok(DAffine3::from_cols(
        DVec3::new(m[0], m[1], m[2]),
        DVec3::new(m[3], m[4], m[5]),
        DVec3::new(m[6], m[7], m[8]),
        DVec3::new(m[9], m[10], m[11]),
    ))
}

fn parse_ref(e: &BytesStart<'_>) -> Result<Ref> {
    let mut r = Ref { objectid: u32::MAX, xf: DAffine3::IDENTITY, path: None };
    let mut have_id = false;
    for a in attrs(e) {
        let (k, v) = a?;
        match k {
            "objectid" => {
                r.objectid = parse_i64(v.as_bytes())
                    .and_then(|i| u32::try_from(i).ok())
                    .ok_or_else(|| anyhow!("bad 3MF objectid '{v}'"))?;
                have_id = true;
            }
            "transform" => r.xf = parse_transform(&v)?,
            "path" => r.path = Some(part_name(&quick_xml::escape::unescape(&v).map_err(|e| anyhow!("{e}"))?)),
            _ => {}
        }
    }
    if !have_id {
        bail!("3MF <{}> without objectid", e.local_name().as_ref());
    }
    Ok(r)
}

fn parse_model(xml: &str) -> Result<Model> {
    let mut reader = Reader::from_str(xml);
    let mut model = Model { unit: 1.0, objects: HashMap::new(), build: Vec::new() };
    let mut cur: Option<(u32, Object)> = None;
    let mut in_mesh = false;
    let mut in_build = false;
    loop {
        let pos = reader.buffer_position();
        let ev = reader.read_event().map_err(|e| anyhow!("XML error near byte {pos}: {e}"))?;
        let (e, empty) = match ev {
            Event::Start(e) => (e, false),
            Event::Empty(e) => (e, true),
            Event::End(e) => {
                match e.local_name().as_ref() {
                    "mesh" => in_mesh = false,
                    "build" => in_build = false,
                    "object" => {
                        if let Some((id, obj)) = cur.take() {
                            finish_object(id, &obj)?;
                            model.objects.insert(id, obj);
                        }
                    }
                    _ => {}
                }
                continue;
            }
            Event::Eof => break,
            _ => continue,
        };
        match e.local_name().as_ref() {
            "vertex" if in_mesh => {
                let Some((_, obj)) = cur.as_mut() else { continue };
                let mut p = [f64::NAN; 3];
                for a in attrs(&e) {
                    let (k, v) = a?;
                    let i = match k {
                        "x" => 0,
                        "y" => 1,
                        "z" => 2,
                        _ => continue,
                    };
                    p[i] = parse_f64(v.as_bytes()).ok_or_else(|| anyhow!("bad 3MF vertex coordinate '{v}'"))?;
                }
                if p.iter().any(|c| c.is_nan()) {
                    bail!("3MF vertex is missing a coordinate");
                }
                obj.vertices.push(DVec3::from_array(p));
            }
            "triangle" if in_mesh => {
                let Some((_, obj)) = cur.as_mut() else { continue };
                let mut t = [u32::MAX; 3];
                for a in attrs(&e) {
                    let (k, v) = a?;
                    let i = match k {
                        "v1" => 0,
                        "v2" => 1,
                        "v3" => 2,
                        _ => continue,
                    };
                    t[i] = parse_i64(v.as_bytes())
                        .and_then(|i| u32::try_from(i).ok())
                        .filter(|&i| i != u32::MAX)
                        .ok_or_else(|| anyhow!("bad 3MF triangle index '{v}'"))?;
                }
                if t.contains(&u32::MAX) {
                    bail!("3MF triangle is missing a vertex index");
                }
                obj.triangles.push(t);
            }
            "component" => {
                if let Some((_, obj)) = cur.as_mut() {
                    obj.components.push(parse_ref(&e)?);
                }
            }
            "item" if in_build => model.build.push(parse_ref(&e)?),
            "mesh" if cur.is_some() => in_mesh = !empty,
            "build" => in_build = !empty,
            "object" => {
                let mut id = None;
                for a in attrs(&e) {
                    let (k, v) = a?;
                    if k == "id" {
                        id = parse_i64(v.as_bytes()).and_then(|i| u32::try_from(i).ok());
                    }
                }
                let id = id.ok_or_else(|| anyhow!("3MF <object> without a valid id"))?;
                if empty {
                    model.objects.insert(id, Object::default());
                } else {
                    cur = Some((id, Object::default()));
                }
            }
            "model" => {
                for a in attrs(&e) {
                    let (k, v) = a?;
                    if k == "unit" {
                        model.unit = unit_scale(&v)?;
                    }
                }
            }
            _ => {}
        }
    }
    if cur.is_some() {
        bail!("truncated 3MF model: unterminated <object>");
    }
    Ok(model)
}

fn finish_object(id: u32, obj: &Object) -> Result<()> {
    let n = obj.vertices.len();
    if let Some(&v) = obj.triangles.iter().flatten().find(|&&v| v as usize >= n) {
        bail!("3MF object {id}: triangle references vertex {v}, but the object has only {n} vertices");
    }
    Ok(())
}

struct Emitter<'a> {
    models: &'a HashMap<String, Model>,
    raw: Raw,
    groups: Vec<i32>,
    next_group: i32,
}

impl Emitter<'_> {
    fn emit(&mut self, path: &str, id: u32, xf: DAffine3, depth: usize) -> Result<()> {
        if depth > MAX_DEPTH {
            bail!("3MF components nested too deeply (cyclic reference?)");
        }
        let model = self.models.get(path).ok_or_else(|| anyhow!("3MF part '{path}' not loaded"))?;
        let obj = model
            .objects
            .get(&id)
            .ok_or_else(|| anyhow!("3MF references missing object {id} in '{path}'"))?;
        if !obj.triangles.is_empty() {
            let base = self.raw.positions.len();
            if base + obj.vertices.len() >= u32::MAX as usize
                || self.raw.triangles.len() + obj.triangles.len() > MAX_TRIANGLES
            {
                bail!("3MF model is too large");
            }
            let base = base as u32;
            self.raw.positions.extend(obj.vertices.iter().map(|&p| xf.transform_point3(p)));
            // Mirroring transforms would turn the surface inside out.
            let flip = xf.matrix3.determinant() < 0.0;
            self.raw.triangles.extend(obj.triangles.iter().map(|t| {
                let t = t.map(|v| v + base);
                if flip {
                    [t[0], t[2], t[1]]
                } else {
                    t
                }
            }));
            let g = self.next_group;
            self.next_group += 1;
            self.groups.resize(self.raw.triangles.len(), g);
        }
        for c in &obj.components {
            let cpath = c.path.as_deref().unwrap_or(path);
            self.emit(cpath, c.objectid, xf * c.xf, depth + 1)?;
        }
        Ok(())
    }
}

pub(crate) fn read(data: &[u8]) -> Result<Raw> {
    let mut zip = ZipArchive::new(Cursor::new(data)).context("not a valid 3MF package (zip archive)")?;
    let root = find_root_model(&mut zip)?;
    let mut models: HashMap<String, Model> = HashMap::new();
    let mut queue = vec![root.clone()];
    while let Some(path) = queue.pop() {
        if models.contains_key(&path) {
            continue;
        }
        let bytes = read_entry(&mut zip, &path)?;
        let model = parse_model(xml_str(&bytes)?).with_context(|| format!("in 3MF part '{path}'"))?;
        let refs = model.build.iter().chain(model.objects.values().flat_map(|o| o.components.iter()));
        for r in refs {
            if let Some(p) = &r.path {
                if !models.contains_key(p) && *p != path {
                    queue.push(p.clone());
                }
            }
        }
        models.insert(path, model);
    }

    let root_model = &models[&root];
    let unit = root_model.unit;
    let mut warnings = Vec::new();
    let items: Vec<Ref> = if root_model.build.is_empty() {
        warnings.push("3MF build has no items; importing every mesh object".to_string());
        let mut ids: Vec<u32> = root_model.objects.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter().map(|objectid| Ref { objectid, xf: DAffine3::IDENTITY, path: None }).collect()
    } else {
        root_model.build.clone()
    };
    let mut em = Emitter { models: &models, raw: Raw::default(), groups: Vec::new(), next_group: 0 };
    for it in &items {
        let path = it.path.clone().unwrap_or_else(|| root.clone());
        em.emit(&path, it.objectid, it.xf, 0)?;
    }
    let mut raw = em.raw;
    if unit != 1.0 {
        for p in &mut raw.positions {
            *p *= unit;
        }
    }
    if em.next_group > 1 {
        raw.groups = Some(em.groups);
    }
    raw.warnings.extend(warnings);
    Ok(raw)
}

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#;

const RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#;

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// Minimal 3MF package: content types, root relationship, one mesh object in millimetres.
pub(crate) fn write<W: Write + Seek>(w: &mut W, pos: &[DVec3], tris: &[[u32; 3]], name: &str, double: bool) -> Result<()> {
    let mut zip = ZipWriter::new(w);
    let estimate = pos.len() as u64 * 80 + tris.len() as u64 * 60;
    let opts = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(3))
        .large_file(estimate > u32::MAX as u64 / 2);
    zip.start_file("[Content_Types].xml", opts)?;
    zip.write_all(CONTENT_TYPES.as_bytes())?;
    zip.start_file("_rels/.rels", opts)?;
    zip.write_all(RELS.as_bytes())?;
    zip.start_file("3D/3dmodel.model", opts)?;
    {
        let mut b = BufWriter::with_capacity(1 << 16, &mut zip);
        write!(
            b,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <model unit=\"millimeter\" xml:lang=\"en-US\" \
             xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n\
             <metadata name=\"Application\">Polymix</metadata>\n\
             <resources>\n<object id=\"1\" type=\"model\" name=\"{}\">\n<mesh>\n<vertices>\n",
            xml_escape(name)
        )?;
        write_parallel(&mut b, pos, |b, _, &p| {
            // Writing into a Vec cannot fail.
            b.extend_from_slice(b"<vertex x=\"");
            let _ = write_float(b, p.x, double);
            b.extend_from_slice(b"\" y=\"");
            let _ = write_float(b, p.y, double);
            b.extend_from_slice(b"\" z=\"");
            let _ = write_float(b, p.z, double);
            b.extend_from_slice(b"\"/>\n");
        })?;
        b.write_all(b"</vertices>\n<triangles>\n")?;
        write_parallel(&mut b, tris, |b, _, t| {
            let _ = writeln!(b, "<triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>", t[0], t[1], t[2]);
        })?;
        b.write_all(
            b"</triangles>\n</mesh>\n</object>\n</resources>\n<build>\n<item objectid=\"1\"/>\n</build>\n</model>\n",
        )?;
        b.flush()?;
    }
    zip.finish()?;
    Ok(())
}
