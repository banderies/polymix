//! Stanford PLY: ASCII, binary little and big endian. Arbitrary elements and properties
//! (scalars and lists of any type) are parsed or skipped; `vertex` x/y/z and optional
//! red/green/blue, `face` `vertex_indices`/`vertex_index` polygons (fan-triangulated) and
//! `tristrips` are imported.

use crate::parse::{parse_f64, parse_i64, parse_int_lenient, show, Lines, Tokens};
use crate::{fan, Raw};
use anyhow::{anyhow, bail, Result};
use glam::DVec3;
use smallvec::SmallVec;
use std::io::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &[u8]) -> Option<Scalar> {
        Some(match name {
            b"char" | b"int8" => Scalar::I8,
            b"uchar" | b"uint8" => Scalar::U8,
            b"short" | b"int16" => Scalar::I16,
            b"ushort" | b"uint16" => Scalar::U16,
            b"int" | b"int32" => Scalar::I32,
            b"uint" | b"uint32" => Scalar::U32,
            b"int64" | b"long" => Scalar::I64,
            b"uint64" | b"ulong" => Scalar::U64,
            b"float" | b"float32" => Scalar::F32,
            b"double" | b"float64" => Scalar::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Scalar::I8 | Scalar::U8 => 1,
            Scalar::I16 | Scalar::U16 => 2,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::I64 | Scalar::U64 | Scalar::F64 => 8,
        }
    }
    /// Full-scale value for color normalization (1 for floats).
    fn color_scale(self) -> f64 {
        match self {
            Scalar::I8 => 127.0,
            Scalar::U8 => 255.0,
            Scalar::I16 => 32767.0,
            Scalar::U16 => 65535.0,
            Scalar::I32 => i32::MAX as f64,
            Scalar::U32 => u32::MAX as f64,
            Scalar::I64 => i64::MAX as f64,
            Scalar::U64 => u64::MAX as f64,
            Scalar::F32 | Scalar::F64 => 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PropType {
    Scalar(Scalar),
    List { count: Scalar, item: Scalar },
}

#[derive(Clone, Debug)]
struct Property {
    name: Vec<u8>,
    ty: PropType,
}

#[derive(Clone, Debug)]
struct Element {
    name: Vec<u8>,
    count: usize,
    props: Vec<Property>,
}

impl Element {
    fn find(&self, names: &[&[u8]]) -> Option<usize> {
        names.iter().find_map(|n| self.props.iter().position(|p| p.name == *n))
    }
    /// Minimum encoded size of one instance in binary encodings (lists assumed empty).
    fn min_binary_size(&self) -> usize {
        self.props
            .iter()
            .map(|p| match p.ty {
                PropType::Scalar(s) => s.size(),
                PropType::List { count, .. } => count.size(),
            })
            .sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoding {
    Ascii,
    BinaryLe,
    BinaryBe,
}

struct Header {
    encoding: Encoding,
    elements: Vec<Element>,
    data_start: usize,
}

fn parse_header(data: &[u8]) -> Result<Header> {
    let mut lines = Lines::new(data);
    match lines.next() {
        Some(l) if l.trim_ascii_end() == b"ply" => {}
        _ => bail!("not a PLY file (missing 'ply' magic)"),
    }
    let mut encoding = None;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        let line = lines.next().ok_or_else(|| anyhow!("PLY header has no 'end_header'"))?;
        let ln = lines.line_no;
        let mut tk = Tokens::new(line);
        let Some(kw) = tk.next() else { continue };
        match kw {
            b"end_header" => break,
            b"comment" | b"obj_info" => {}
            b"format" => {
                encoding = Some(match tk.next().unwrap_or(b"") {
                    b"ascii" => Encoding::Ascii,
                    b"binary_little_endian" => Encoding::BinaryLe,
                    b"binary_big_endian" => Encoding::BinaryBe,
                    other => bail!("line {ln}: unknown PLY format '{}'", show(other)),
                });
            }
            b"element" => {
                let name = tk.next().ok_or_else(|| anyhow!("line {ln}: element without a name"))?;
                let count = tk
                    .next()
                    .and_then(parse_i64)
                    .and_then(|c| usize::try_from(c).ok())
                    .ok_or_else(|| anyhow!("line {ln}: bad element count"))?;
                elements.push(Element { name: name.to_vec(), count, props: Vec::new() });
            }
            b"property" => {
                let el = elements.last_mut().ok_or_else(|| anyhow!("line {ln}: property before any element"))?;
                let t = tk.next().unwrap_or(b"");
                let ty = if t == b"list" {
                    let c = tk.next().unwrap_or(b"");
                    let i = tk.next().unwrap_or(b"");
                    let count = Scalar::parse(c).ok_or_else(|| anyhow!("line {ln}: bad list count type '{}'", show(c)))?;
                    let item = Scalar::parse(i).ok_or_else(|| anyhow!("line {ln}: bad list item type '{}'", show(i)))?;
                    if matches!(count, Scalar::F32 | Scalar::F64) {
                        bail!("line {ln}: list count type must be an integer");
                    }
                    PropType::List { count, item }
                } else {
                    PropType::Scalar(Scalar::parse(t).ok_or_else(|| anyhow!("line {ln}: bad property type '{}'", show(t)))?)
                };
                let name = tk.next().ok_or_else(|| anyhow!("line {ln}: property without a name"))?;
                el.props.push(Property { name: name.to_vec(), ty });
            }
            // Unknown header lines are tolerated.
            _ => {}
        }
    }
    let encoding = encoding.ok_or_else(|| anyhow!("PLY header has no 'format' line"))?;
    Ok(Header { encoding, elements, data_start: lines.offset() })
}

/// Value source for the element data section.
trait Source {
    fn real(&mut self, ty: Scalar) -> Result<f64>;
    fn int(&mut self, ty: Scalar) -> Result<i64>;
    fn skip(&mut self, ty: Scalar, n: usize) -> Result<()>;
    /// Upper bound on the number of values left (to sanity-check counts).
    fn max_values_left(&self, ty: Scalar) -> usize;
    fn bytes_left(&self) -> usize;
}

struct Binary<'a> {
    data: &'a [u8],
    pos: usize,
    be: bool,
}

impl<'a> Binary<'a> {
    #[inline]
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.pos + N;
        if end > self.data.len() {
            bail!("truncated PLY data (offset {})", self.pos);
        }
        let b: [u8; N] = self.data[self.pos..end].try_into().unwrap();
        self.pos = end;
        Ok(b)
    }
}

macro_rules! rd {
    ($s:expr, $t:ty) => {{
        let b = $s.take::<{ std::mem::size_of::<$t>() }>()?;
        if $s.be {
            <$t>::from_be_bytes(b)
        } else {
            <$t>::from_le_bytes(b)
        }
    }};
}

impl Source for Binary<'_> {
    #[inline]
    fn real(&mut self, ty: Scalar) -> Result<f64> {
        Ok(match ty {
            Scalar::I8 => rd!(self, i8) as f64,
            Scalar::U8 => rd!(self, u8) as f64,
            Scalar::I16 => rd!(self, i16) as f64,
            Scalar::U16 => rd!(self, u16) as f64,
            Scalar::I32 => rd!(self, i32) as f64,
            Scalar::U32 => rd!(self, u32) as f64,
            Scalar::I64 => rd!(self, i64) as f64,
            Scalar::U64 => rd!(self, u64) as f64,
            Scalar::F32 => rd!(self, f32) as f64,
            Scalar::F64 => rd!(self, f64),
        })
    }
    #[inline]
    fn int(&mut self, ty: Scalar) -> Result<i64> {
        Ok(match ty {
            Scalar::I8 => rd!(self, i8) as i64,
            Scalar::U8 => rd!(self, u8) as i64,
            Scalar::I16 => rd!(self, i16) as i64,
            Scalar::U16 => rd!(self, u16) as i64,
            Scalar::I32 => rd!(self, i32) as i64,
            Scalar::U32 => rd!(self, u32) as i64,
            Scalar::I64 => rd!(self, i64),
            Scalar::U64 => i64::try_from(rd!(self, u64)).map_err(|_| anyhow!("PLY integer out of range"))?,
            Scalar::F32 | Scalar::F64 => {
                let f = self.real(ty)?;
                if f.fract() != 0.0 || f.abs() > 9.0e15 {
                    bail!("non-integral PLY index {f}");
                }
                f as i64
            }
        })
    }
    #[inline]
    fn skip(&mut self, ty: Scalar, n: usize) -> Result<()> {
        let end = n.checked_mul(ty.size()).and_then(|b| b.checked_add(self.pos));
        match end {
            Some(e) if e <= self.data.len() => {
                self.pos = e;
                Ok(())
            }
            _ => bail!("truncated PLY data (offset {})", self.pos),
        }
    }
    fn max_values_left(&self, ty: Scalar) -> usize {
        (self.data.len() - self.pos) / ty.size()
    }
    fn bytes_left(&self) -> usize {
        self.data.len() - self.pos
    }
}

struct Ascii<'a> {
    tk: Tokens<'a>,
}

impl Ascii<'_> {
    #[inline]
    fn token(&mut self) -> Result<&[u8]> {
        self.tk.next().ok_or_else(|| anyhow!("truncated PLY data (unexpected end of file)"))
    }
}

impl Source for Ascii<'_> {
    #[inline]
    fn real(&mut self, _ty: Scalar) -> Result<f64> {
        let t = self.token()?;
        parse_f64(t).ok_or_else(|| anyhow!("bad PLY number '{}'", show(t)))
    }
    #[inline]
    fn int(&mut self, _ty: Scalar) -> Result<i64> {
        let t = self.token()?;
        parse_int_lenient(t).ok_or_else(|| anyhow!("bad PLY integer '{}'", show(t)))
    }
    fn skip(&mut self, _ty: Scalar, n: usize) -> Result<()> {
        for _ in 0..n {
            self.token()?;
        }
        Ok(())
    }
    fn max_values_left(&self, _ty: Scalar) -> usize {
        // Each value takes at least one character plus a separator.
        self.tk.remaining() / 2 + 1
    }
    fn bytes_left(&self) -> usize {
        self.tk.remaining()
    }
}

#[inline]
fn list_len<S: Source>(src: &mut S, count: Scalar, item: Scalar) -> Result<usize> {
    let n = src.int(count)?;
    let n = usize::try_from(n).map_err(|_| anyhow!("negative PLY list length {n}"))?;
    if n > src.max_values_left(item) {
        bail!("PLY list length {n} exceeds the remaining data");
    }
    Ok(n)
}

#[inline]
fn skip_prop<S: Source>(src: &mut S, ty: PropType) -> Result<()> {
    match ty {
        PropType::Scalar(s) => src.skip(s, 1),
        PropType::List { count, item } => {
            let n = list_len(src, count, item)?;
            src.skip(item, n)
        }
    }
}

/// Role of a vertex property.
#[derive(Clone, Copy, PartialEq)]
enum VRole {
    Coord(usize),
    Color(usize),
    Other,
}

fn read_body<S: Source>(src: &mut S, header: &Header, binary: bool) -> Result<Raw> {
    let mut raw = Raw::default();
    let mut colors: Vec<[f32; 3]> = Vec::new();
    let mut have_vertices = false;
    let mut poly: SmallVec<[u32; 8]> = SmallVec::new();
    let mut short_faces = 0usize;

    for el in &header.elements {
        // Refuse counts the remaining data cannot possibly hold (protects allocations).
        let min = if binary { el.min_binary_size() } else { 2 * el.props.len() };
        if el.count > 0 && min > 0 && el.count.saturating_mul(min) > src.bytes_left() + 1 {
            bail!(
                "truncated PLY: element '{}' declares {} entries, more than the remaining {} bytes can hold",
                String::from_utf8_lossy(&el.name),
                el.count,
                src.bytes_left()
            );
        }
        match el.name.as_slice() {
            b"vertex" if !have_vertices => {
                have_vertices = true;
                let mut roles = vec![VRole::Other; el.props.len()];
                for (k, name) in [&b"x"[..], b"y", b"z"].into_iter().enumerate() {
                    let i = el.find(&[name]).ok_or_else(|| {
                        anyhow!("PLY vertex element has no '{}' property", String::from_utf8_lossy(name))
                    })?;
                    roles[i] = VRole::Coord(k);
                }
                let color_names: [&[&[u8]]; 3] =
                    [&[b"red", b"r", b"diffuse_red"], &[b"green", b"g", b"diffuse_green"], &[b"blue", b"b", b"diffuse_blue"]];
                let color_idx: Vec<Option<usize>> = color_names.iter().map(|n| el.find(n)).collect();
                let has_color = color_idx.iter().all(Option::is_some);
                if has_color {
                    for (k, i) in color_idx.iter().enumerate() {
                        roles[i.unwrap()] = VRole::Color(k);
                    }
                }
                for (p, role) in el.props.iter().zip(&roles) {
                    if *role != VRole::Other && matches!(p.ty, PropType::List { .. }) {
                        bail!("PLY vertex coordinate/color property cannot be a list");
                    }
                }
                raw.positions.reserve(el.count);
                if has_color {
                    colors.reserve(el.count);
                }
                for _ in 0..el.count {
                    let mut p = [0.0f64; 3];
                    let mut c = [1.0f32; 3];
                    for (prop, role) in el.props.iter().zip(&roles) {
                        match (*role, prop.ty) {
                            (VRole::Coord(k), PropType::Scalar(s)) => p[k] = src.real(s)?,
                            (VRole::Color(k), PropType::Scalar(s)) => c[k] = (src.real(s)? / s.color_scale()) as f32,
                            _ => skip_prop(src, prop.ty)?,
                        }
                    }
                    raw.positions.push(DVec3::from_array(p));
                    if has_color {
                        colors.push(c);
                    }
                }
                if has_color {
                    raw.colors = Some(std::mem::take(&mut colors));
                }
            }
            b"face" | b"tristrips" => {
                let strips = el.name == b"tristrips";
                let li = el.find(&[b"vertex_indices", b"vertex_index"]);
                let Some(li) = li.filter(|&i| matches!(el.props[i].ty, PropType::List { .. })) else {
                    bail!("PLY {} element has no vertex_indices list", String::from_utf8_lossy(&el.name));
                };
                if !strips {
                    raw.triangles.reserve(el.count);
                }
                for _ in 0..el.count {
                    for (k, prop) in el.props.iter().enumerate() {
                        match prop.ty {
                            PropType::List { count, item } if k == li => {
                                let n = list_len(src, count, item)?;
                                poly.clear();
                                for _ in 0..n {
                                    let v = src.int(item)?;
                                    if strips && v < 0 {
                                        // -1 restarts a strip.
                                        poly.push(u32::MAX);
                                        continue;
                                    }
                                    if v < 0 || v >= u32::MAX as i64 {
                                        bail!("PLY face index {v} out of range");
                                    }
                                    poly.push(v as u32);
                                }
                                if strips {
                                    strip_triangles(&poly, &mut raw.triangles);
                                } else if fan(&poly, &mut raw.triangles) == 0 {
                                    short_faces += 1;
                                }
                            }
                            _ => skip_prop(src, prop.ty)?,
                        }
                    }
                }
            }
            _ => {
                for _ in 0..el.count {
                    for prop in &el.props {
                        skip_prop(src, prop.ty)?;
                    }
                }
            }
        }
    }
    if short_faces > 0 {
        raw.warnings.push(format!("skipped {short_faces} faces with fewer than 3 vertices"));
    }
    Ok(raw)
}

/// Triangle strips with `u32::MAX` as restart marker; winding alternates.
fn strip_triangles(strip: &[u32], out: &mut Vec<[u32; 3]>) {
    for run in strip.split(|&v| v == u32::MAX) {
        for i in 2..run.len() {
            let (a, b, c) = (run[i - 2], run[i - 1], run[i]);
            if i % 2 == 0 {
                out.push([a, b, c]);
            } else {
                out.push([b, a, c]);
            }
        }
    }
}

pub(crate) fn read(data: &[u8]) -> Result<Raw> {
    let header = parse_header(data)?;
    let body = &data[header.data_start..];
    let raw = match header.encoding {
        Encoding::Ascii => read_body(&mut Ascii { tk: Tokens::new(body) }, &header, false)?,
        Encoding::BinaryLe => read_body(&mut Binary { data: body, pos: 0, be: false }, &header, true)?,
        Encoding::BinaryBe => read_body(&mut Binary { data: body, pos: 0, be: true }, &header, true)?,
    };
    if raw.positions.is_empty() && !header.elements.iter().any(|e| e.name == b"vertex") {
        bail!("PLY file has no vertex element");
    }
    Ok(raw)
}

/// Binary little-endian PLY with `float` (or `double`) coordinates.
pub(crate) fn write<W: Write + ?Sized>(w: &mut W, pos: &[DVec3], tris: &[[u32; 3]], colors: Option<&[[f32; 3]]>, double: bool) -> Result<()> {
    let ty = if double { "double" } else { "float" };
    let rgb = if colors.is_some() { "property uchar red\nproperty uchar green\nproperty uchar blue\n" } else { "" };
    write!(
        w,
        "ply\nformat binary_little_endian 1.0\ncomment Polymix export\nelement vertex {}\n\
         property {ty} x\nproperty {ty} y\nproperty {ty} z\n{rgb}element face {}\n\
         property list uchar int vertex_indices\nend_header\n",
        pos.len(),
        tris.len()
    )?;
    if tris.len() > i32::MAX as usize || pos.len() > i32::MAX as usize {
        bail!("mesh too large for PLY int indices");
    }
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
    for (i, p) in pos.iter().enumerate() {
        if double {
            for c in p.to_array() {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        } else {
            for c in p.as_vec3().to_array() {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        }
        if let Some(c) = colors.and_then(|c| c.get(i)) {
            buf.extend(c.map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8));
        }
        if buf.len() >= (1 << 16) - 32 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    for t in tris {
        buf.push(3);
        for &v in t {
            buf.extend_from_slice(&(v as i32).to_le_bytes());
        }
        if buf.len() >= (1 << 16) - 32 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)?;
    Ok(())
}
