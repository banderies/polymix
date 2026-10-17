//! Wavefront OBJ: `v` (optionally with `r g b`), `f` in all index forms with negative
//! (relative) indices, polygons (fan-triangulated), and `g`/`o` as face groups. Other
//! statements are ignored.

use crate::parse::{find_byte, parse_f64, parse_i64, show, strip_bom, trim, Lines, Tokens};
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
    let mut short_faces = 0usize;

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
            b"f" | b"fo" => {
                poly.clear();
                let nv = positions.len() as i64;
                for t in tk.by_ref() {
                    let idx_tok = match find_byte(t, b'/') {
                        Some(i) => &t[..i],
                        None => t,
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
    double: bool,
) -> Result<()> {
    writeln!(w, "# Polymix OBJ export\n# {} vertices, {} triangles", pos.len(), tris.len())?;
    write_parallel(w, pos, |b, _, &p| {
        b.extend_from_slice(b"v ");
        put_xyz(b, p, double, b" ");
        b.push(b'\n');
    })?;
    write_parallel(w, tris, |b, i, t| {
        if let Some(g) = groups {
            if i == 0 || g[i] != g[i - 1] {
                let _ = writeln!(b, "g group_{}", g[i]);
            }
        }
        let _ = writeln!(b, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1);
    })?;
    Ok(())
}
