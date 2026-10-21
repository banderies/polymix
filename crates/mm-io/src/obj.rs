//! Wavefront OBJ: `v` (optionally with `r g b`), `vt`, `f` in all index forms with negative
//! (relative) indices, polygons (fan-triangulated), `g`/`o` as face groups, and `mtllib` /
//! `usemtl` for the texture of the most used material. Other statements are ignored.

use crate::parse::{find_byte, parse_f64, parse_i64, show, strip_bom, trim, Lines, Tokens};
use crate::weld::{FastMap, Key3};
use crate::{fan, put_xyz, write_parallel, Raw};
use anyhow::{anyhow, bail, Result};
use glam::DVec3;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::io::Write;

pub(crate) fn read(data: &[u8]) -> Result<Raw> {
    let data = strip_bom(data);
    let mut positions: Vec<DVec3> = Vec::with_capacity(data.len() / 64);
    let mut colors: Option<Vec<[f32; 3]>> = None;
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(data.len() / 32);
    let mut groups: Vec<i32> = Vec::with_capacity(data.len() / 32);
    let mut group_ids: HashMap<Vec<u8>, i32> = HashMap::new();
    let mut cur_o: Vec<u8> = Vec::new();
    let mut cur_g: Vec<u8> = Vec::new();
    let mut cur_gid: Option<i32> = None;
    let mut poly: SmallVec<[u32; 8]> = SmallVec::new();
    let mut poly_vt: SmallVec<[Option<u32>; 8]> = SmallVec::new();
    let mut short_faces = 0usize;
    let mut vts: Vec<[f32; 2]> = Vec::new();
    // Per triangle: its corners' `vt`. Made at the first face that has any (the triangles
    // before it get none), so files without texture coordinates never pay for it.
    let mut tri_vt: Option<Vec<[Option<u32>; 3]>> = None;
    let mut mtllib: Option<String> = None;
    let mut material: Option<String> = None;
    // Triangles per material, in order of first use; `cur_mat` indexes the current
    // material's entry once a face has used it since its `usemtl`.
    let mut material_ids: HashMap<String, usize> = HashMap::new();
    let mut material_faces: Vec<(String, usize)> = Vec::new();
    let mut cur_mat: Option<usize> = None;

    let mut lines = Lines::new(data);
    while let Some(line) = lines.next() {
        let line = match find_byte(line, b'#') {
            Some(i) => &line[..i],
            None => line,
        };
        let mut tk = Tokens::new(line);
        let Some(cmd) = tk.next() else { continue };
        match cmd {
            b"v" => {
                let mut vals = [0f64; 7];
                let mut n = 0;
                for t in tk.by_ref() {
                    if n == vals.len() {
                        break;
                    }
                    vals[n] = parse_f64(t)
                        .ok_or_else(|| anyhow!("line {}: bad vertex value '{}'", lines.line_no, show(t)))?;
                    n += 1;
                }
                if n < 3 {
                    bail!("line {}: vertex needs 3 coordinates", lines.line_no);
                }
                if n >= 6 {
                    let c = colors.get_or_insert_with(|| vec![[1.0; 3]; positions.len()]);
                    c.push([vals[3] as f32, vals[4] as f32, vals[5] as f32]);
                } else if let Some(c) = &mut colors {
                    c.push([1.0; 3]);
                }
                positions.push(DVec3::new(vals[0], vals[1], vals[2]));
            }
            b"vt" => {
                let u = tk.next().and_then(parse_f64).unwrap_or(0.0) as f32;
                let v = tk.next().and_then(parse_f64).unwrap_or(0.0) as f32;
                vts.push([u, v]);
            }
            b"mtllib" => mtllib = Some(String::from_utf8_lossy(trim(tk.rest())).into_owned()),
            b"usemtl" => {
                material = Some(String::from_utf8_lossy(trim(tk.rest())).into_owned());
                cur_mat = None;
            }
            b"f" | b"fo" => {
                poly.clear();
                poly_vt.clear();
                let nv = positions.len() as i64;
                for t in tk.by_ref() {
                    let idx_tok = match find_byte(t, b'/') {
                        Some(i) => {
                            // v/vt or v/vt/vn: the texture coordinate's index (relative ones too).
                            let rest = &t[i + 1..];
                            let vt_tok = match find_byte(rest, b'/') {
                                Some(j) => &rest[..j],
                                None => rest,
                            };
                            poly_vt.push(parse_i64(vt_tok).and_then(|k| {
                                let k = if k > 0 { k - 1 } else { vts.len() as i64 + k };
                                (0..vts.len() as i64).contains(&k).then_some(k as u32)
                            }));
                            &t[..i]
                        }
                        None => {
                            poly_vt.push(None);
                            t
                        }
                    };
                    let i = parse_i64(idx_tok)
                        .ok_or_else(|| anyhow!("line {}: bad face index '{}'", lines.line_no, show(t)))?;
                    let v = if i > 0 {
                        i - 1
                    } else if i < 0 {
                        nv + i
                    } else {
                        bail!("line {}: face index 0 is invalid (OBJ indices are 1-based)", lines.line_no)
                    };
                    if v < 0 || v >= u32::MAX as i64 {
                        bail!("line {}: face index {i} out of range ({nv} vertices so far)", lines.line_no);
                    }
                    poly.push(v as u32);
                }
                let g = match cur_gid {
                    Some(g) => g,
                    None => {
                        let mut key = cur_o.clone();
                        key.push(0);
                        key.extend_from_slice(&cur_g);
                        let next = group_ids.len() as i32;
                        let g = *group_ids.entry(key).or_insert(next);
                        cur_gid = Some(g);
                        g
                    }
                };
                let added = fan(&poly, &mut tris);
                if added == 0 {
                    short_faces += 1;
                }
                if tri_vt.is_none() && added > 0 && poly_vt.iter().any(Option::is_some) {
                    tri_vt = Some(vec![[None; 3]; tris.len() - added]);
                }
                if let Some(tri_vt) = &mut tri_vt {
                    for i in 1..=added {
                        tri_vt.push([poly_vt[0], poly_vt[i], poly_vt[i + 1]]);
                    }
                }
                if let Some(m) = &material {
                    let k = *cur_mat.get_or_insert_with(|| {
                        *material_ids.entry(m.clone()).or_insert_with(|| {
                            material_faces.push((m.clone(), 0));
                            material_faces.len() - 1
                        })
                    });
                    material_faces[k].1 += added;
                }
                groups.extend(std::iter::repeat_n(g, added));
            }
            b"g" => {
                cur_g = trim(tk.rest()).to_vec();
                cur_gid = None;
            }
            b"o" => {
                cur_o = trim(tk.rest()).to_vec();
                cur_g.clear();
                cur_gid = None;
            }
            _ => {}
        }
    }

    if let Some(c) = &mut colors {
        // Some writers use 0..255 instead of 0..1.
        if c.iter().flatten().any(|&x| x > 1.0) {
            for x in c.iter_mut().flatten() {
                *x /= 255.0;
            }
        }
    }
    let mut raw = Raw { positions, triangles: tris, colors, ..Default::default() };
    if let Some(tri_vt) = tri_vt.filter(|t| t.iter().flatten().any(Option::is_some)) {
        raw.uvs = Some(tri_vt.iter().map(|c| c.map(|k| k.map_or([0.0; 2], |k| vts[k as usize]))).collect());
    }
    if let (Some(lib), Some((m, _))) = (mtllib, material_faces.into_iter().max_by_key(|(_, n)| *n)) {
        raw.material = Some((lib, m));
    }
    if group_ids.len() > 1 {
        raw.groups = Some(groups);
    }
    if short_faces > 0 {
        raw.warnings.push(format!("skipped {short_faces} faces with fewer than 3 vertices"));
    }
    Ok(raw)
}

pub(crate) fn write<W: Write + ?Sized>(
    w: &mut W,
    pos: &[DVec3],
    tris: &[[u32; 3]],
    groups: Option<&[i32]>,
    colors: Option<&[[f32; 3]]>,
    uvs: Option<&[[[f32; 2]; 3]]>,
    double: bool,
) -> Result<()> {
    writeln!(w, "# Polymix OBJ export\n# {} vertices, {} triangles", pos.len(), tris.len())?;
    write_parallel(w, pos, |b, i, &p| {
        b.extend_from_slice(b"v ");
        put_xyz(b, p, double, b" ");
        // Vertex colours as the widely read `v x y z r g b` extension.
        if let Some(c) = colors.and_then(|c| c.get(i)) {
            let _ = write!(b, " {:.4} {:.4} {:.4}", c[0], c[1], c[2]);
        }
        b.push(b'\n');
    })?;
    // Texture coordinates: one `vt` per distinct value, faces as `v/vt`.
    let mut vt_index: Vec<[u32; 3]> = Vec::new();
    if let Some(uvs) = uvs {
        // Distinct by bit pattern (so -0.0 and 0.0 stay apart, as written).
        let mut ids: FastMap<Key3<u32>, u32> = FastMap::default();
        let mut vts: Vec<[f32; 2]> = Vec::new();
        vt_index = uvs
            .iter()
            .map(|c| {
                c.map(|q| {
                    *ids.entry(Key3([q[0].to_bits(), q[1].to_bits(), 0])).or_insert_with(|| {
                        vts.push(q);
                        vts.len() as u32 - 1
                    })
                })
            })
            .collect();
        write_parallel(w, &vts, |b, _, q| {
            let _ = writeln!(b, "vt {} {}", q[0], q[1]);
        })?;
    }
    write_parallel(w, tris, |b, i, t| {
        if let Some(g) = groups {
            if i == 0 || g[i] != g[i - 1] {
                let _ = writeln!(b, "g group_{}", g[i]);
            }
        }
        match vt_index.get(i) {
            Some(u) => {
                let _ = writeln!(b, "f {}/{} {}/{} {}/{}", t[0] + 1, u[0] + 1, t[1] + 1, u[1] + 1, t[2] + 1, u[2] + 1);
            }
            None => {
                let _ = writeln!(b, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1);
            }
        }
    })?;
    Ok(())
}

/// The diffuse texture (`map_Kd`) of `material` in the material library `mtl`, next to it.
pub(crate) fn diffuse_map(mtl: &std::path::Path, material: &str) -> Option<std::path::PathBuf> {
    let text = std::fs::read_to_string(mtl).ok()?;
    let mut current = false;
    for line in text.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix("newmtl") {
            current = name.trim() == material;
        } else if let (true, Some(rest)) = (current, line.strip_prefix("map_Kd")) {
            // Options (-s, -o, …) come first; the file name is last.
            let file = rest.split_whitespace().last()?;
            return Some(mtl.parent()?.join(file.replace('\\', "/")));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUAD: &str = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nvt 0 0\nvt 1 0\nvt 1 1\n";

    #[test]
    fn texture_coordinates_start_at_the_first_face_that_has_them() {
        // Two faces without `vt`, then one with, one with `v//vn`, and one with a bad index.
        let text = format!("{QUAD}f 1 2 3\nf 1 3 4\nf 1/1 2/2 3/3 4/1\nf 2//1 5//1 3//1\nf 2/9 5/-1 3\n");
        let raw = read(text.as_bytes()).unwrap();
        assert_eq!(raw.triangles.len(), 6);
        let (z, a, b, c) = ([0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        assert_eq!(raw.uvs.unwrap(), vec![[z; 3], [z; 3], [a, b, c], [a, c, a], [z; 3], [z, c, z]]);
        // `vt` lines that no face uses, or none at all: no UVs.
        for text in [format!("{QUAD}f 1 2 3\nf 1//1 3//1 4//1\n"), "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".to_string(), format!("{QUAD}f 1/7 2/8 3/9\n")] {
            assert!(read(text.as_bytes()).unwrap().uvs.is_none(), "{text}");
        }
        // A face too short for a triangle does not start them.
        let raw = read(format!("{QUAD}f 1/1 2/2\nf 1 2 3\nf 1/1 2/2 3/3\n").as_bytes()).unwrap();
        assert_eq!(raw.uvs.unwrap(), vec![[z; 3], [a, b, c]]);
    }

    #[test]
    fn the_material_with_the_most_triangles_is_used() {
        // `a`: a triangle, then two more after `b`'s quad (2 triangles); faces before the first
        // `usemtl` count for none.
        let text = format!("{QUAD}mtllib lib.mtl\nf 1 2 3 4 5\nusemtl a\nf 1 2 3\nusemtl b\nf 1 2 3 4\nusemtl a\nf 1 3 4\nf 2 3 5\n");
        assert_eq!(read(text.as_bytes()).unwrap().material, Some(("lib.mtl".to_string(), "a".to_string())));
        let text = format!("{QUAD}mtllib lib.mtl\nusemtl a\nf 1 2 3\nusemtl b\nf 1 2 3 4 5\nusemtl a\nf 1 3 4\n");
        assert_eq!(read(text.as_bytes()).unwrap().material, Some(("lib.mtl".to_string(), "b".to_string())));
        // No library, no material.
        assert_eq!(read(format!("{QUAD}usemtl a\nf 1 2 3\n").as_bytes()).unwrap().material, None);
    }

    #[test]
    fn written_texture_coordinates_are_distinct_by_bits() {
        let pos = [DVec3::ZERO, DVec3::X, DVec3::Y];
        let uvs = [[[0.0f32, -0.0], [-0.0, 0.0], [0.0, 0.0]], [[0.0, -0.0], [0.5, 0.5], [-0.0, 0.0]]];
        let mut out = Vec::new();
        write(&mut out, &pos, &[[0, 1, 2], [0, 2, 1]], None, None, Some(&uvs[..]), false).unwrap();
        let text = String::from_utf8(out).unwrap();
        let vt: Vec<&str> = text.lines().filter(|l| l.starts_with("vt ")).collect();
        assert_eq!(vt, ["vt 0 -0", "vt -0 0", "vt 0 0", "vt 0.5 0.5"]);
        assert!(text.contains("f 1/1 2/2 3/3\nf 1/1 3/4 2/2\n"), "{text}");
    }
}
