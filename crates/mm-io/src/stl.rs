//! STL, binary and ASCII. Corners are welded by exact position since STL has no
//! shared vertices.

use crate::parse::{parse_f64, show, strip_bom, Tokens};
use crate::weld::{f32_key, f64_key, weld_exact_by, Key3};
use crate::{put_xyz, write_parallel, Raw};
use anyhow::{bail, Result};
use glam::DVec3;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::io::Write;

const HEADER: usize = 80;
const RECORD: usize = 50;

/// True when the size is exactly `84 + 50 * n` for the header's triangle count `n`.
/// This is decisive even when the header starts with `solid` (many exporters do that);
/// ASCII text cannot match because bytes 80..84 would have to be near-zero.
fn exact_binary_size(data: &[u8]) -> bool {
    data.len() >= HEADER + 4 && {
        let n = u32::from_le_bytes(data[80..84].try_into().unwrap()) as u64;
        (HEADER as u64 + 4) + RECORD as u64 * n == data.len() as u64
    }
}

/// Starts with `solid` and has an ASCII STL keyword in the first 4 KB.
fn looks_ascii(data: &[u8]) -> bool {
    let text = strip_bom(data);
    let start = text.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(text.len());
    let text = &text[start..];
    if text.len() < 5 || !text[..5].eq_ignore_ascii_case(b"solid") {
        return false;
    }
    let lower: Vec<u8> = text[..text.len().min(4096)].to_ascii_lowercase();
    let has = |kw: &[u8]| lower.windows(kw.len()).any(|w| w == kw);
    has(b"facet") || has(b"endsolid") || has(b"vertex")
}

pub(crate) fn read(data: &[u8], weld: bool) -> Result<Raw> {
    if exact_binary_size(data) || !looks_ascii(data) {
        return read_binary(data, weld);
    }
    match read_ascii(data, weld) {
        Ok(raw) if !raw.triangles.is_empty() => Ok(raw),
        // A binary file whose header mentions "solid ... facet" and whose size is off
        // (e.g. trailing bytes): fall back to binary if that yields triangles.
        ascii => match read_binary(data, weld) {
            Ok(raw) if !raw.triangles.is_empty() => Ok(raw),
            _ => ascii,
        },
    }
}

fn read_binary(data: &[u8], weld: bool) -> Result<Raw> {
    if data.len() < HEADER + 4 {
        bail!("file too small for binary STL ({} bytes) and not ASCII STL", data.len());
    }
    let mut warnings = Vec::new();
    let declared = u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize;
    let body = data.len() - (HEADER + 4);
    let fits = body / RECORD;
    let n = if declared == 0 && fits > 0 && body.is_multiple_of(RECORD) {
        warnings.push(format!("binary STL header declares 0 triangles; using file size ({fits} triangles)"));
        fits
    } else if declared > fits {
        bail!(
            "truncated binary STL: header declares {declared} triangles but the file holds only {fits} ({} bytes)",
            data.len()
        );
    } else {
        declared
    };
    let extra = body - n * RECORD;
    if extra > 0 {
        warnings.push(format!("ignored {extra} trailing bytes after {n} STL triangles"));
    }
    let recs = &data[HEADER + 4..HEADER + 4 + n * RECORD];
    // Raw coordinate bits of every corner (skipping the facet normal).
    let mut corners: Vec<[u32; 3]> = vec![[0; 3]; n * 3];
    corners.par_chunks_mut(3).zip(recs.par_chunks_exact(RECORD)).for_each(|(c, r)| {
        for (k, ck) in c.iter_mut().enumerate() {
            let o = 12 + 12 * k;
            for (j, x) in ck.iter_mut().enumerate() {
                *x = u32::from_le_bytes(r[o + 4 * j..o + 4 * j + 4].try_into().unwrap());
            }
        }
    });
    let pos_of = |c: [u32; 3]| {
        DVec3::new(f32::from_bits(c[0]) as f64, f32::from_bits(c[1]) as f64, f32::from_bits(c[2]) as f64)
    };
    let mut raw = Raw { warnings, ..Default::default() };
    if weld {
        let (remap, first) = weld_exact_by(corners.len(), |i| {
            let c = corners[i];
            Key3([f32_key(f32::from_bits(c[0])), f32_key(f32::from_bits(c[1])), f32_key(f32::from_bits(c[2]))])
        });
        raw.positions = first.iter().map(|&i| pos_of(corners[i as usize])).collect();
        raw.triangles = remap.as_chunks::<3>().0.to_vec();
        raw.welded = corners.len() - first.len();
    } else {
        raw.positions = corners.iter().map(|&c| pos_of(c)).collect();
        raw.triangles = (0..n as u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
    }
    Ok(raw)
}

fn read_ascii(data: &[u8], weld: bool) -> Result<Raw> {
    let data = strip_bom(data);
    let mut tk = Tokens::new(data);
    // Corner positions (3 per output triangle) and the solid index per triangle.
    let mut corners: Vec<DVec3> = Vec::with_capacity(data.len() / 80);
    let mut solid_of: Vec<i32> = Vec::with_capacity(data.len() / 250);
    let mut poly: SmallVec<[DVec3; 4]> = SmallVec::new();
    let mut solid: i32 = -1;
    let mut in_loop = false;
    let mut short_loops = 0usize;
    let mut saw_end = false;
    let mut trailing_junk = false;

    let mut emit = |poly: &mut SmallVec<[DVec3; 4]>, solid: i32, corners: &mut Vec<DVec3>| {
        if poly.len() < 3 {
            short_loops += 1;
        } else {
            for i in 1..poly.len() - 1 {
                corners.extend_from_slice(&[poly[0], poly[i], poly[i + 1]]);
                solid_of.push(solid.max(0));
            }
        }
        poly.clear();
    };

    while let Some(tok) = tk.next() {
        let kw = |k: &[u8]| tok.eq_ignore_ascii_case(k);
        if kw(b"vertex") {
            if !in_loop {
                bail!("line {}: 'vertex' outside of 'outer loop'", tk.line_no());
            }
            let mut p = [0.0; 3];
            for c in p.iter_mut() {
                let t = tk.next().unwrap_or(b"");
                *c = parse_f64(t)
                    .ok_or_else(|| anyhow::anyhow!("line {}: bad vertex coordinate '{}'", tk.line_no(), show(t)))?;
            }
            poly.push(DVec3::from_array(p));
        } else if kw(b"facet") || kw(b"outer") {
        } else if kw(b"normal") {
            for _ in 0..3 {
                tk.next();
            }
        } else if kw(b"loop") {
            if in_loop {
                bail!("line {}: nested 'outer loop'", tk.line_no());
            }
            in_loop = true;
            poly.clear();
        } else if kw(b"endloop") || (kw(b"endfacet") && in_loop) {
            if !in_loop {
                bail!("line {}: 'endloop' without 'outer loop'", tk.line_no());
            }
            in_loop = false;
            emit(&mut poly, solid, &mut corners);
        } else if kw(b"endfacet") {
        } else if kw(b"solid") {
            if in_loop {
                bail!("line {}: 'solid' inside a facet", tk.line_no());
            }
            solid += 1;
            saw_end = false;
            tk.skip_line();
        } else if kw(b"endsolid") {
            saw_end = true;
            tk.skip_line();
        } else if saw_end {
            // Junk after a complete solid (padding, exporter garbage) is ignored.
            trailing_junk = true;
            break;
        } else {
            bail!("line {}: unexpected token '{}' in ASCII STL", tk.line_no(), show(tok));
        }
    }
    if in_loop || !poly.is_empty() {
        bail!("truncated ASCII STL: file ends inside a facet");
    }
    let mut raw = Raw::default();
    if !saw_end {
        raw.warnings.push("ASCII STL is missing 'endsolid'".to_string());
    }
    if short_loops > 0 {
        raw.warnings.push(format!("skipped {short_loops} facets with fewer than 3 vertices"));
    }
    if trailing_junk {
        raw.warnings.push("ignored trailing data after 'endsolid'".to_string());
    }
    let ntri = solid_of.len();
    if weld {
        let (remap, first) =
            weld_exact_by(corners.len(), |i| Key3([f64_key(corners[i].x), f64_key(corners[i].y), f64_key(corners[i].z)]));
        raw.positions = first.iter().map(|&i| corners[i as usize]).collect();
        raw.triangles = remap.as_chunks::<3>().0.to_vec();
        raw.welded = corners.len() - first.len();
    } else {
        raw.triangles = (0..ntri as u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
        raw.positions = corners;
    }
    if solid > 0 {
        raw.groups = Some(solid_of);
    }
    Ok(raw)
}

#[inline]
fn facet_normal(pos: &[DVec3], t: [u32; 3]) -> DVec3 {
    let [a, b, c] = t.map(|v| pos[v as usize]);
    (b - a).cross(c - a).normalize_or_zero()
}

/// Binary STL. The 80-byte header never starts with `solid`.
pub(crate) fn write_binary<W: Write + ?Sized>(w: &mut W, pos: &[DVec3], tris: &[[u32; 3]], name: &str) -> Result<()> {
    if tris.len() > u32::MAX as usize {
        bail!("too many triangles for binary STL");
    }
    let mut header = [b' '; HEADER];
    let text = format!("Polymix binary STL: {name}");
    let text = text.as_bytes();
    header[..text.len().min(HEADER)].copy_from_slice(&text[..text.len().min(HEADER)]);
    w.write_all(&header)?;
    w.write_all(&(tris.len() as u32).to_le_bytes())?;
    const CHUNK: usize = 1 << 16;
    let mut buf = vec![0u8; RECORD * CHUNK.min(tris.len())];
    for chunk in tris.chunks(CHUNK) {
        let b = &mut buf[..chunk.len() * RECORD];
        b.par_chunks_mut(RECORD).zip(chunk.par_iter()).for_each(|(out, &t)| {
            let n = facet_normal(pos, t);
            let mut vals = [0f32; 12];
            vals[..3].copy_from_slice(&n.as_vec3().to_array());
            for k in 0..3 {
                vals[3 + 3 * k..6 + 3 * k].copy_from_slice(&pos[t[k] as usize].as_vec3().to_array());
            }
            for (j, v) in vals.iter().enumerate() {
                out[4 * j..4 * j + 4].copy_from_slice(&v.to_le_bytes());
            }
            out[48] = 0;
            out[49] = 0;
        });
        w.write_all(b)?;
    }
    Ok(())
}

/// ASCII STL.
pub(crate) fn write_ascii<W: Write + ?Sized>(
    w: &mut W,
    pos: &[DVec3],
    tris: &[[u32; 3]],
    name: &str,
    double: bool,
) -> Result<()> {
    // Solid names end at the newline; keep them on one line.
    let name: String = name.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    writeln!(w, "solid {name}")?;
    write_parallel(w, tris, |b, _, &t| {
        let n = facet_normal(pos, t);
        let _ = write!(b, "facet normal {} {} {}\n outer loop\n", n.x as f32, n.y as f32, n.z as f32);
        for &v in &t {
            b.extend_from_slice(b"  vertex ");
            put_xyz(b, pos[v as usize], double, b" ");
            b.push(b'\n');
        }
        b.extend_from_slice(b" endloop\nendfacet\n");
    })?;
    writeln!(w, "endsolid {name}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection() {
        // Binary with a "solid" header and exact size.
        let mut b = vec![0u8; 84 + 50];
        b[..5].copy_from_slice(b"solid");
        b[80] = 1;
        assert!(exact_binary_size(&b));
        assert!(!exact_binary_size(b"solid x\nfacet normal 0 0 1\n"));
        assert!(looks_ascii(b"solid x\nfacet normal 0 0 1\n"));
        assert!(looks_ascii(b"\xEF\xBB\xBF  SOLID\r\nendsolid\r\n"));
        assert!(!looks_ascii(b"garbage garbage"));
        assert!(!looks_ascii(b"solid but nothing else"));
    }
}
