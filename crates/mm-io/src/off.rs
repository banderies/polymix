//! Geomview OFF (ASCII), including the `C`/`N`/`ST` vertex variants and per-face colors
//! (ignored). `COFF` vertex colors are read.

use crate::parse::{find_byte, parse_f64, parse_int_lenient, show, strip_bom, Lines, Tokens};
use crate::{fan, put_xyz, write_parallel, Raw};
use anyhow::{anyhow, bail, Result};
use glam::DVec3;
use smallvec::SmallVec;
use std::io::Write;

/// Non-empty lines with `#` comments removed, tracking line numbers.
struct OffLines<'a> {
    inner: Lines<'a>,
}

impl<'a> OffLines<'a> {
    fn next_line(&mut self) -> Option<&'a [u8]> {
        for line in self.inner.by_ref() {
            let line = match find_byte(line, b'#') {
                Some(i) => &line[..i],
                None => line,
            };
            if line.iter().any(|b| !b.is_ascii_whitespace()) {
                return Some(line);
            }
        }
        None
    }
    fn line_no(&self) -> usize {
        self.inner.line_no
    }
}

fn count(tok: Option<&[u8]>, what: &str, line: usize) -> Result<usize> {
    let t = tok.ok_or_else(|| anyhow!("line {line}: missing {what} count"))?;
    let v = parse_int_lenient(t).ok_or_else(|| anyhow!("line {line}: bad {what} count '{}'", show(t)))?;
    usize::try_from(v).map_err(|_| anyhow!("line {line}: negative {what} count"))
}

pub(crate) fn read(data: &[u8]) -> Result<Raw> {
    let data = strip_bom(data);
    let mut lines = OffLines { inner: Lines::new(data) };
    let first = lines.next_line().ok_or_else(|| anyhow!("empty OFF file"))?;
    let mut tk = Tokens::new(first);
    let kw = tk.next().unwrap_or(b"");
    let (mut has_c, mut has_n) = (false, false);
    let mut header_tokens: SmallVec<[&[u8]; 4]> = SmallVec::new();
    if let Some(prefix) = kw.strip_suffix(b"OFF").or_else(|| kw.strip_suffix(b"off")) {
        // Texture coordinates (ST) follow the colors on a vertex line, so they are ignored.
        let mut p = prefix.strip_prefix(b"ST").unwrap_or(prefix);
        if let Some(rest) = p.strip_prefix(b"C") {
            has_c = true;
            p = rest;
        }
        if let Some(rest) = p.strip_prefix(b"N") {
            has_n = true;
            p = rest;
        }
        match p {
            b"" => {}
            b"n" => {
                let dim = match tk.next() {
                    Some(t) => t,
                    None => {
                        tk = Tokens::new(lines.next_line().unwrap_or(b""));
                        tk.next().unwrap_or(b"")
                    }
                };
                if dim != b"3" {
                    bail!("only 3-dimensional OFF is supported (got dimension '{}')", show(dim));
                }
            }
            _ => bail!("unsupported OFF variant '{}'", show(kw)),
        }
        header_tokens.extend(tk.by_ref());
        if header_tokens.is_empty() {
            let l = lines.next_line().ok_or_else(|| anyhow!("OFF header is missing the counts line"))?;
            header_tokens.extend(Tokens::new(l));
        }
    } else if parse_int_lenient(kw).is_some() {
        // Keyword omitted: the first line holds the counts.
        header_tokens.push(kw);
        header_tokens.extend(tk.by_ref());
    } else {
        bail!("not an OFF file (starts with '{}')", show(kw));
    }
    if header_tokens.first().is_some_and(|t| t.eq_ignore_ascii_case(b"BINARY")) {
        bail!("binary OFF is not supported");
    }
    let line = lines.line_no();
    let nv = count(header_tokens.first().copied(), "vertex", line)?;
    let nf = count(header_tokens.get(1).copied(), "face", line)?;

    // A vertex line takes at least 5 bytes ("0 0 0"), a face line at least 4.
    let budget = data.len();
    if nv.saturating_mul(5) > budget || nf.saturating_mul(4) > budget {
        bail!("OFF header declares {nv} vertices / {nf} faces, more than the file can hold");
    }

    let mut positions = Vec::with_capacity(nv);
    let mut colors: Option<Vec<[f32; 3]>> = has_c.then(|| Vec::with_capacity(nv));
    for i in 0..nv {
        let l = lines
            .next_line()
            .ok_or_else(|| anyhow!("truncated OFF: expected {nv} vertices, found {i}"))?;
        let mut tk = Tokens::new(l);
        let mut p = [0.0; 3];
        for c in p.iter_mut() {
            let t = tk.next().unwrap_or(b"");
            *c = parse_f64(t).ok_or_else(|| anyhow!("line {}: bad vertex coordinate '{}'", lines.line_no(), show(t)))?;
        }
        positions.push(DVec3::from_array(p));
        if has_n {
            tk.by_ref().take(3).for_each(drop);
        }
        if let Some(c) = &mut colors {
            let mut rgb = [1.0f32; 3];
            for x in rgb.iter_mut() {
                match tk.next() {
                    Some(t) => {
                        *x = parse_f64(t)
                            .ok_or_else(|| anyhow!("line {}: bad vertex color '{}'", lines.line_no(), show(t)))?
                            as f32
                    }
                    None => break,
                }
            }
            c.push(rgb);
        }
    }

    let mut tris = Vec::with_capacity(nf * 2);
    let mut poly: SmallVec<[u32; 8]> = SmallVec::new();
    let mut short_faces = 0;
    for i in 0..nf {
        let l = lines.next_line().ok_or_else(|| anyhow!("truncated OFF: expected {nf} faces, found {i}"))?;
        let mut tk = Tokens::new(l);
        let line = lines.line_no();
        let k = count(tk.next(), "face vertex", line)?;
        poly.clear();
        for _ in 0..k {
            let t = tk.next().ok_or_else(|| anyhow!("line {line}: face has fewer than {k} indices"))?;
            let v = parse_int_lenient(t).ok_or_else(|| anyhow!("line {line}: bad face index '{}'", show(t)))?;
            if v < 0 || v >= u32::MAX as i64 {
                bail!("line {line}: face index {v} out of range");
            }
            poly.push(v as u32);
        }
        if fan(&poly, &mut tris) == 0 {
            short_faces += 1;
        }
    }

    if let Some(c) = &mut colors {
        // COFF colors are either 0..1 floats or 0..255 integers.
        if c.iter().flatten().any(|&x| x > 1.0) {
            for x in c.iter_mut().flatten() {
                *x /= 255.0;
            }
        }
    }
    let mut raw = Raw { positions, triangles: tris, colors, ..Default::default() };
    if short_faces > 0 {
        raw.warnings.push(format!("skipped {short_faces} faces with fewer than 3 vertices"));
    }
    Ok(raw)
}

pub(crate) fn write<W: Write + ?Sized>(w: &mut W, pos: &[DVec3], tris: &[[u32; 3]], double: bool) -> Result<()> {
    writeln!(w, "OFF\n{} {} 0", pos.len(), tris.len())?;
    write_parallel(w, pos, |b, _, &p| {
        put_xyz(b, p, double, b" ");
        b.push(b'\n');
    })?;
    write_parallel(w, tris, |b, _, t| {
        let _ = writeln!(b, "3 {} {} {}", t[0], t[1], t[2]);
    })?;
    Ok(())
}
