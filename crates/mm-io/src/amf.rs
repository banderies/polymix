//! AMF (Additive Manufacturing File Format, ISO/ASTM 52915): plain or zip-compressed XML.
//! All objects are merged; each `<volume>` becomes a face group and coordinates are
//! converted to millimetres. Vertex colours are read and written.

use crate::parse::{parse_f64, parse_i64, strip_bom};
use crate::Raw;
use anyhow::{anyhow, bail, Context, Result};
use glam::DVec3;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::{Cursor, Read, Write};

/// Millimetres per AMF unit.
fn unit_scale(unit: &str) -> Result<f64> {
    Ok(match unit.trim().to_ascii_lowercase().as_str() {
        "" | "millimeter" | "millimetre" => 1.0,
        "inch" => 25.4,
        "feet" | "foot" => 304.8,
        "meter" | "metre" => 1000.0,
        "micron" | "micrometer" => 0.001,
        u => bail!("unknown AMF unit '{u}'"),
    })
}

pub(crate) fn read(bytes: &[u8]) -> Result<Raw> {
    // Compressed AMF is a zip with the XML as its (first) entry.
    let mut owned = Vec::new();
    let xml_bytes: &[u8] = if bytes.starts_with(b"PK") {
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).context("not a valid compressed AMF (zip)")?;
        if zip.is_empty() {
            bail!("compressed AMF is empty");
        }
        zip.by_index(0)?.read_to_end(&mut owned)?;
        &owned
    } else {
        bytes
    };
    let xml = std::str::from_utf8(strip_bom(xml_bytes)).context("AMF is not valid UTF-8")?;
    let mut reader = Reader::from_str(xml);
    let mut raw = Raw::default();
    let mut groups: Vec<i32> = Vec::new();
    let mut colors: Vec<[f32; 3]> = Vec::new();
    let mut any_color = false;
    let mut scale = 1.0;
    // Per object: index of its first vertex in `raw.positions`.
    let mut base = 0u32;
    let mut volume = -1i32;
    let mut path: Vec<String> = Vec::new();
    let mut coord = [f64::NAN; 3];
    let mut color = [1.0f32; 3];
    let mut tri = [u32::MAX; 3];
    loop {
        let pos = reader.buffer_position();
        match reader.read_event().map_err(|e| anyhow!("AMF XML error near byte {pos}: {e}"))? {
            Event::Start(e) => {
                let name = e.local_name().as_ref().to_string();
                match name.as_str() {
                    "amf" => {
                        for a in e.attributes().flatten() {
                            if a.key.local_name().as_ref() == "unit" {
                                scale = unit_scale(&a.value)?;
                            }
                        }
                    }
                    "object" => base = raw.positions.len() as u32,
                    "volume" => volume += 1,
                    "vertex" => {
                        coord = [f64::NAN; 3];
                        color = [1.0; 3];
                    }
                    "triangle" => tri = [u32::MAX; 3],
                    _ => {}
                }
                path.push(name);
            }
            Event::Text(t) => {
                let text = t.xml10_content();
                let text = text.trim();
                let (Some(leaf), parent) = (path.last().map(String::as_str), path.len().checked_sub(2).map(|i| path[i].as_str())) else { continue };
                match (parent, leaf) {
                    (Some("coordinates"), "x" | "y" | "z") => {
                        let i = (leaf.as_bytes()[0] - b'x') as usize;
                        coord[i] = parse_f64(text.as_bytes()).ok_or_else(|| anyhow!("bad AMF coordinate '{text}'"))?;
                    }
                    (Some("color"), "r" | "g" | "b") if path.iter().any(|p| p == "vertex") => {
                        let i = match leaf {
                            "r" => 0,
                            "g" => 1,
                            _ => 2,
                        };
                        color[i] = text.parse::<f32>().unwrap_or(1.0);
                        any_color = true;
                    }
                    (Some("triangle"), "v1" | "v2" | "v3") => {
                        let i = (leaf.as_bytes()[1] - b'1') as usize;
                        let v = parse_i64(text.as_bytes()).and_then(|v| u32::try_from(v).ok()).ok_or_else(|| anyhow!("bad AMF vertex index '{text}'"))?;
                        tri[i] = base + v;
                    }
                    _ => {}
                }
            }
            Event::End(e) => {
                let name = e.local_name();
                match name.as_ref() {
                    "vertex" => {
                        if coord.iter().any(|c| c.is_nan()) {
                            bail!("AMF vertex is missing a coordinate");
                        }
                        raw.positions.push(DVec3::from_array(coord) * scale);
                        colors.push(color);
                    }
                    "triangle" => {
                        if tri.contains(&u32::MAX) {
                            bail!("AMF triangle is missing a vertex");
                        }
                        raw.triangles.push(tri);
                        groups.push(volume.max(0));
                    }
                    _ => {}
                }
                path.pop();
            }
            Event::Eof => break,
            _ => {}
        }
    }
    if raw.triangles.iter().flatten().any(|&v| v as usize >= raw.positions.len()) {
        bail!("AMF triangle refers to a missing vertex");
    }
    if groups.iter().any(|&g| g != 0) {
        raw.groups = Some(groups);
    }
    if any_color {
        raw.colors = Some(colors);
    }
    Ok(raw)
}

/// Write one object whose volumes are the face groups (millimetres).
pub(crate) fn write<W: Write + ?Sized>(w: &mut W, pos: &[DVec3], tris: &[[u32; 3]], groups: &[i32], colors: Option<&[[f32; 3]]>, double: bool) -> Result<()> {
    let num = |x: f64| if double { format!("{x}") } else { format!("{}", x as f32) };
    writeln!(w, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<amf unit=\"millimeter\" version=\"1.1\">\n <object id=\"0\">\n  <mesh>\n   <vertices>")?;
    for (i, p) in pos.iter().enumerate() {
        write!(w, "    <vertex><coordinates><x>{}</x><y>{}</y><z>{}</z></coordinates>", num(p.x), num(p.y), num(p.z))?;
        if let Some(c) = colors.and_then(|c| c.get(i)) {
            write!(w, "<color><r>{:.4}</r><g>{:.4}</g><b>{:.4}</b></color>", c[0], c[1], c[2])?;
        }
        writeln!(w, "</vertex>")?;
    }
    writeln!(w, "   </vertices>")?;
    let mut ids: Vec<i32> = groups.to_vec();
    ids.sort_unstable();
    ids.dedup();
    for g in ids {
        writeln!(w, "   <volume>\n    <metadata type=\"name\">group_{g}</metadata>")?;
        for (t, _) in tris.iter().zip(groups).filter(|(_, &tg)| tg == g) {
            writeln!(w, "    <triangle><v1>{}</v1><v2>{}</v2><v3>{}</v3></triangle>", t[0], t[1], t[2])?;
        }
        writeln!(w, "   </volume>")?;
    }
    writeln!(w, "  </mesh>\n </object>\n</amf>")?;
    Ok(())
}

/// COLLADA 1.4 document with a single triangle mesh (export only).
pub(crate) fn write_dae<W: Write + ?Sized>(w: &mut W, pos: &[DVec3], tris: &[[u32; 3]], name: &str) -> Result<()> {
    let name: String = name.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
    let name = if name.is_empty() { "mesh".to_string() } else { name };
    writeln!(w, "<?xml version=\"1.0\" encoding=\"utf-8\"?>")?;
    writeln!(w, "<COLLADA xmlns=\"http://www.collada.org/2005/11/COLLADASchema\" version=\"1.4.1\">")?;
    writeln!(w, " <asset><unit name=\"millimeter\" meter=\"0.001\"/></asset>")?;
    writeln!(w, " <library_geometries>\n  <geometry id=\"{name}\" name=\"{name}\">\n   <mesh>")?;
    write!(w, "    <source id=\"{name}-pos\"><float_array id=\"{name}-pos-array\" count=\"{}\">", pos.len() * 3)?;
    for p in pos {
        write!(w, "{} {} {} ", p.x as f32, p.y as f32, p.z as f32)?;
    }
    writeln!(w, "</float_array>")?;
    writeln!(
        w,
        "     <technique_common><accessor source=\"#{name}-pos-array\" count=\"{}\" stride=\"3\"><param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/></accessor></technique_common>\n    </source>",
        pos.len()
    )?;
    writeln!(w, "    <vertices id=\"{name}-vtx\"><input semantic=\"POSITION\" source=\"#{name}-pos\"/></vertices>")?;
    write!(w, "    <triangles count=\"{}\"><input semantic=\"VERTEX\" source=\"#{name}-vtx\" offset=\"0\"/><p>", tris.len())?;
    for t in tris {
        write!(w, "{} {} {} ", t[0], t[1], t[2])?;
    }
    writeln!(w, "</p></triangles>\n   </mesh>\n  </geometry>\n </library_geometries>")?;
    writeln!(w, " <library_visual_scenes><visual_scene id=\"scene\"><node id=\"{name}-node\"><instance_geometry url=\"#{name}\"/></node></visual_scene></library_visual_scenes>")?;
    writeln!(w, " <scene><instance_visual_scene url=\"#scene\"/></scene>\n</COLLADA>")?;
    Ok(())
}

/// VRML 2.0 indexed face set, with per-vertex colours when present (export only).
pub(crate) fn write_wrl<W: Write + ?Sized>(w: &mut W, pos: &[DVec3], tris: &[[u32; 3]], colors: Option<&[[f32; 3]]>) -> Result<()> {
    writeln!(w, "#VRML V2.0 utf8\n# Polymix export\nShape {{\n geometry IndexedFaceSet {{\n  solid FALSE\n  coord Coordinate {{ point [")?;
    for p in pos {
        writeln!(w, "   {} {} {},", p.x as f32, p.y as f32, p.z as f32)?;
    }
    writeln!(w, "  ] }}")?;
    if let Some(c) = colors {
        writeln!(w, "  colorPerVertex TRUE\n  color Color {{ color [")?;
        for c in c {
            writeln!(w, "   {:.4} {:.4} {:.4},", c[0], c[1], c[2])?;
        }
        writeln!(w, "  ] }}")?;
    }
    writeln!(w, "  coordIndex [")?;
    for t in tris {
        writeln!(w, "   {}, {}, {}, -1,", t[0], t[1], t[2])?;
    }
    writeln!(w, "  ]\n }}\n}}")?;
    Ok(())
}

/// TetGen `.smesh` piecewise linear complex; each facet's boundary marker is its face group,
/// so CFD patches carry through to the volume mesh (export only).
pub(crate) fn write_smesh<W: Write + ?Sized>(w: &mut W, pos: &[DVec3], tris: &[[u32; 3]], groups: &[i32], double: bool) -> Result<()> {
    writeln!(w, "# Polymix export (TetGen .smesh)\n# Part 1 - node list\n{} 3 0 0", pos.len())?;
    for (i, p) in pos.iter().enumerate() {
        if double {
            writeln!(w, "{i} {} {} {}", p.x, p.y, p.z)?;
        } else {
            writeln!(w, "{i} {} {} {}", p.x as f32, p.y as f32, p.z as f32)?;
        }
    }
    writeln!(w, "# Part 2 - facet list\n{} 1", tris.len())?;
    for (t, g) in tris.iter().zip(groups) {
        writeln!(w, "3 {} {} {} {g}", t[0], t[1], t[2])?;
    }
    writeln!(w, "# Part 3 - hole list\n0\n# Part 4 - region list\n0")?;
    Ok(())
}
