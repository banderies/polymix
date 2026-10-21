//! JPEG-LS decoding (ITU-T T.87 / ISO 14495-1), lossless and near-lossless, 2–16 bits, one
//! or more components in any interleave mode, so DICOM files compressed with JPEG-LS
//! (transfer syntaxes 1.2.840.10008.1.2.4.80 / .81) can be shown. Written from the standard;
//! checked against files made by CharLS (the reference implementation).

use anyhow::{bail, ensure, Context, Result};

/// A decoded image: samples (component values interleaved per pixel), row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub components: usize,
    pub bits: u32,
    pub samples: Vec<u16>,
}

const J: [u32; 32] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Coding parameters of a scan.
#[derive(Clone, Copy, Debug)]
struct Params {
    maxval: i32,
    near: i32,
    t1: i32,
    t2: i32,
    t3: i32,
    reset: i32,
    range: i32,
    qbpp: u32,
    limit: u32,
}

fn clamp_t(i: i32, j: i32, maxval: i32) -> i32 {
    if i > maxval || i < j { j } else { i }
}

fn ceil_log2(n: i32) -> u32 {
    let mut k = 0;
    while (1i64 << k) < n as i64 {
        k += 1;
    }
    k
}

impl Params {
    fn new(bits: u32, near: i32, custom: Option<[i32; 5]>) -> Self {
        let mut maxval = (1i32 << bits) - 1;
        let (mut t1, mut t2, mut t3, mut reset) = (0, 0, 0, 64);
        if let Some([m, a, b, c, r]) = custom {
            if m > 0 {
                maxval = m;
            }
            (t1, t2, t3) = (a, b, c);
            if r > 0 {
                reset = r;
            }
        }
        // Default thresholds (T.87 C.2.4.1.1.1), for those not given.
        // Basic thresholds 3, 7 and 21.
        let (d1, d2, d3) = if maxval >= 128 {
            let f = (maxval.min(4095) + 128) / 256;
            let d1 = clamp_t(f + 2 + 3 * near, near + 1, maxval);
            let d2 = clamp_t(f * 4 + 3 + 5 * near, d1, maxval);
            (d1, d2, clamp_t(f * 17 + 4 + 7 * near, d2, maxval))
        } else {
            let f = 256 / (maxval + 1);
            let d1 = clamp_t((3 / f + 3 * near).max(2), near + 1, maxval);
            let d2 = clamp_t((7 / f + 5 * near).max(3), d1, maxval);
            (d1, d2, clamp_t((21 / f + 7 * near).max(4), d2, maxval))
        };
        if t1 == 0 {
            t1 = d1;
        }
        if t2 == 0 {
            t2 = d2;
        }
        if t3 == 0 {
            t3 = d3;
        }
        let range = (maxval + 2 * near) / (2 * near + 1) + 1;
        let bpp = ceil_log2(maxval + 1).max(2);
        Self { maxval, near, t1, t2, t3, reset, range, qbpp: ceil_log2(range), limit: 2 * (bpp + bpp.max(8)) }
    }

    fn quantize(&self, d: i32) -> i32 {
        if d <= -self.t3 {
            -4
        } else if d <= -self.t2 {
            -3
        } else if d <= -self.t1 {
            -2
        } else if d < -self.near {
            -1
        } else if d <= self.near {
            0
        } else if d < self.t1 {
            1
        } else if d < self.t2 {
            2
        } else if d < self.t3 {
            3
        } else {
            4
        }
    }

    fn context(&self, ra: i32, rb: i32, rc: i32, rd: i32) -> i32 {
        (self.quantize(rd - rb) * 9 + self.quantize(rb - rc)) * 9 + self.quantize(rc - ra)
    }

    fn reconstruct(&self, px: i32, err: i32) -> i32 {
        let mut x = px + err * (2 * self.near + 1);
        if x < -self.near {
            x += self.range * (2 * self.near + 1);
        } else if x > self.maxval + self.near {
            x -= self.range * (2 * self.near + 1);
        }
        x.clamp(0, self.maxval)
    }
}

/// Context variables of regular mode.
#[derive(Clone, Copy)]
struct Ctx {
    a: i32,
    b: i32,
    c: i32,
    n: i32,
}

/// Context variables of run interruption samples.
#[derive(Clone, Copy)]
struct RiCtx {
    a: i32,
    n: i32,
    nn: i32,
    ri_type: i32,
}

/// Bits of the entropy-coded data, with the 0 bit stuffed after every 0xFF removed.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    cache: u64,
    count: u32,
    prev_ff: bool,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, cache: 0, count: 0, prev_ff: false }
    }

    fn fill(&mut self) {
        while self.count <= 56 {
            let Some(&b) = self.data.get(self.pos) else {
                // Past the end (or at a marker): zeros.
                self.cache <<= 8;
                self.count += 8;
                continue;
            };
            if self.prev_ff {
                if b & 0x80 != 0 {
                    // A marker: no more data.
                    self.data = &self.data[..self.pos];
                    continue;
                }
                self.cache = (self.cache << 7) | b as u64;
                self.count += 7;
            } else {
                self.cache = (self.cache << 8) | b as u64;
                self.count += 8;
            }
            self.prev_ff = b == 0xff;
            self.pos += 1;
        }
    }

    fn bit(&mut self) -> bool {
        if self.count == 0 {
            self.fill();
        }
        self.count -= 1;
        (self.cache >> self.count) & 1 != 0
    }

    fn value(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        if self.count < n {
            self.fill();
        }
        self.count -= n;
        ((self.cache >> self.count) & ((1u64 << n) - 1)) as i32
    }

    /// Zeros before the next 1 (which is consumed). The zeros are counted a cache at a time
    /// rather than bit by bit; more than 64 of them is corrupt data, as before.
    fn unary(&mut self) -> Result<u32> {
        let mut q = 0;
        loop {
            if self.count == 0 {
                self.fill();
            }
            // The unread bits, first one at the top (`count` is 1 to 64 here).
            let lz = (self.cache << (64 - self.count)).leading_zeros();
            if lz < self.count {
                self.count -= lz + 1;
                q += lz;
                if q > 64 {
                    bail!("corrupt JPEG-LS data");
                }
                return Ok(q);
            }
            q += self.count;
            self.count = 0;
            if q > 64 {
                bail!("corrupt JPEG-LS data");
            }
        }
    }

    /// A Golomb-coded value with the length limit of T.87 A.5.3.
    fn golomb(&mut self, k: u32, limit: u32, qbpp: u32) -> Result<i32> {
        let high = self.unary()?;
        if high >= limit.saturating_sub(qbpp + 1) {
            return Ok(self.value(qbpp) + 1);
        }
        Ok(((high as i32) << k) + self.value(k))
    }
}

struct Decoder<'a> {
    p: Params,
    bits: Bits<'a>,
    ctx: Vec<Ctx>,
    ri: [RiCtx; 2],
    run_index: usize,
}

fn med(ra: i32, rb: i32, rc: i32) -> i32 {
    if rc >= ra.max(rb) {
        ra.min(rb)
    } else if rc <= ra.min(rb) {
        ra.max(rb)
    } else {
        ra + rb - rc
    }
}

fn sign(n: i32) -> i32 {
    if n < 0 { -1 } else { 1 }
}

impl<'a> Decoder<'a> {
    fn new(p: Params, data: &'a [u8]) -> Self {
        let a = ((p.range + 32) / 64).max(2);
        Self { p, bits: Bits::new(data), ctx: vec![Ctx { a, b: 0, c: 0, n: 1 }; 365], ri: [RiCtx { a, n: 1, nn: 0, ri_type: 0 }, RiCtx { a, n: 1, nn: 0, ri_type: 1 }], run_index: 0 }
    }

    /// A regular-mode sample with signed context `qs` and prediction `pred`.
    fn regular(&mut self, qs: i32, pred: i32) -> Result<i32> {
        let s = sign(qs);
        let q = qs.unsigned_abs() as usize;
        let p = self.p;
        let c = self.ctx[q];
        let mut k = 0;
        while (c.n << k) < c.a {
            k += 1;
            if k > 31 {
                bail!("corrupt JPEG-LS data");
            }
        }
        let px = (pred + s * c.c).clamp(0, p.maxval);
        let m = self.bits.golomb(k, p.limit, p.qbpp)?;
        let mut err = if m & 1 == 0 { m >> 1 } else { -((m + 1) >> 1) };
        if k == 0 && p.near == 0 && 2 * c.b + c.n - 1 < 0 {
            err = -err - 1;
        }
        // Context update (A.6).
        let x = &mut self.ctx[q];
        x.b += err * (2 * p.near + 1);
        x.a += err.abs();
        if x.n == p.reset {
            x.a >>= 1;
            x.b = if x.b >= 0 { x.b >> 1 } else { -((1 - x.b) >> 1) };
            x.n >>= 1;
        }
        x.n += 1;
        if x.b <= -x.n {
            x.b += x.n;
            if x.c > -128 {
                x.c -= 1;
            }
            if x.b <= -x.n {
                x.b = -x.n + 1;
            }
        } else if x.b > 0 {
            x.b -= x.n;
            if x.c < 127 {
                x.c += 1;
            }
            if x.b > 0 {
                x.b = 0;
            }
        }
        Ok(p.reconstruct(px, s * err))
    }

    /// The error of a run interruption sample (A.7.2).
    fn ri_error(&mut self, t: usize) -> Result<i32> {
        let c = self.ri[t];
        let temp = c.a + (c.n >> 1) * c.ri_type;
        let mut k = 0;
        while (c.n << k) < temp {
            k += 1;
            if k > 31 {
                bail!("corrupt JPEG-LS data");
            }
        }
        let limit = self.p.limit - J[self.run_index] - 1;
        let em = self.bits.golomb(k, limit, self.p.qbpp)?;
        let tmp = em + c.ri_type;
        let map = tmp & 1;
        let abs = (tmp + map) / 2;
        let err = if ((k != 0 || 2 * c.nn >= c.n) as i32) == map { -abs } else { abs };
        let x = &mut self.ri[t];
        if err < 0 {
            x.nn += 1;
        }
        x.a += (em + 1 - x.ri_type) >> 1;
        if x.n == self.p.reset {
            x.a >>= 1;
            x.n >>= 1;
            x.nn >>= 1;
        }
        x.n += 1;
        Ok(err)
    }

    /// Length of a run starting at `x` in a line of `width` samples (A.7.1).
    fn run_length(&mut self, x: usize, width: usize) -> Result<(usize, bool)> {
        let left = width - x;
        let mut n = 0;
        while self.bits.bit() {
            let block = 1usize << J[self.run_index];
            let take = block.min(left - n);
            n += take;
            if take == block {
                self.run_index = (self.run_index + 1).min(31);
            }
            if n == left {
                return Ok((n, true));
            }
        }
        n += self.bits.value(J[self.run_index]) as usize;
        ensure!(n < left, "corrupt JPEG-LS run");
        Ok((n, false))
    }

    /// Decode one line of one component; `prev` and `cur` have one guard sample on each side.
    fn line(&mut self, prev: &mut [i32], cur: &mut [i32]) -> Result<()> {
        let w = cur.len() - 2;
        prev[w + 1] = prev[w];
        cur[0] = prev[1];
        let mut x = 1;
        while x <= w {
            let (ra, rb, rc, rd) = (cur[x - 1], prev[x], prev[x - 1], prev[x + 1]);
            let qs = self.p.context(ra, rb, rc, rd);
            if qs != 0 {
                cur[x] = self.regular(qs, med(ra, rb, rc))?;
                x += 1;
                continue;
            }
            let (n, to_end) = self.run_length(x - 1, w)?;
            cur[x..x + n].fill(ra);
            x += n;
            if to_end {
                break;
            }
            let rb = prev[x];
            cur[x] = if (ra - rb).abs() <= self.p.near {
                let e = self.ri_error(1)?;
                self.p.reconstruct(ra, e)
            } else {
                let e = self.ri_error(0)?;
                self.p.reconstruct(rb, e * sign(rb - ra))
            };
            self.run_index = self.run_index.saturating_sub(1);
            x += 1;
        }
        Ok(())
    }

    /// Decode one line of `nc` sample-interleaved components (pixel `x` of component `c` at
    /// `x * nc + c`, with one guard pixel on each side).
    fn line_interleaved(&mut self, prev: &mut [i32], cur: &mut [i32], nc: usize) -> Result<()> {
        let w = cur.len() / nc - 2;
        for c in 0..nc {
            prev[(w + 1) * nc + c] = prev[w * nc + c];
            cur[c] = prev[nc + c];
        }
        let mut x = 1;
        while x <= w {
            let mut qs = [0i32; 4];
            for c in 0..nc {
                qs[c] = self.p.context(cur[(x - 1) * nc + c], prev[x * nc + c], prev[(x - 1) * nc + c], prev[(x + 1) * nc + c]);
            }
            if qs[..nc].iter().any(|&q| q != 0) {
                for c in 0..nc {
                    let (ra, rb, rc) = (cur[(x - 1) * nc + c], prev[x * nc + c], prev[(x - 1) * nc + c]);
                    cur[x * nc + c] = self.regular(qs[c], med(ra, rb, rc))?;
                }
                x += 1;
                continue;
            }
            let (n, to_end) = self.run_length(x - 1, w)?;
            for p in x..x + n {
                for c in 0..nc {
                    cur[p * nc + c] = cur[(x - 1) * nc + c];
                }
            }
            x += n;
            if to_end {
                break;
            }
            for c in 0..nc {
                let (ra, rb) = (cur[(x - 1) * nc + c], prev[x * nc + c]);
                let e = self.ri_error(0)?;
                cur[x * nc + c] = self.p.reconstruct(rb, e * sign(rb - ra));
            }
            self.run_index = self.run_index.saturating_sub(1);
            x += 1;
        }
        Ok(())
    }
}

fn u16_at(b: &[u8], at: usize) -> Result<usize> {
    Ok(u16::from_be_bytes([*b.get(at).context("truncated JPEG-LS data")?, *b.get(at + 1).context("truncated JPEG-LS data")?]) as usize)
}

/// Position of the next marker (0xFF followed by a byte with its top bit set) at or after `at`.
fn next_marker(b: &[u8], mut at: usize) -> Option<usize> {
    while at + 1 < b.len() {
        if b[at] == 0xff && b[at + 1] >= 0x80 && b[at + 1] != 0xff {
            return Some(at);
        }
        at += 1;
    }
    None
}

/// Decode a JPEG-LS image (a whole `.jls` file or a DICOM frame).
pub fn decode(data: &[u8]) -> Result<Image> {
    ensure!(data.len() >= 4 && data[0] == 0xff && data[1] == 0xd8, "not a JPEG-LS stream");
    let mut at = 2;
    let mut frame: Option<(u32, usize, usize, Vec<u8>)> = None;
    let mut custom: Option<[i32; 5]> = None;
    let mut planes: Vec<Vec<i32>> = Vec::new();
    let mut out: Option<Image> = None;
    loop {
        at = next_marker(data, at).context("JPEG-LS data ends early")?;
        let marker = data[at + 1];
        at += 2;
        match marker {
            0xd9 => break,
            0xf7 => {
                // SOF55: precision, lines, columns, components.
                let len = u16_at(data, at)?;
                let seg = data.get(at + 2..at + len).context("truncated frame header")?;
                ensure!(seg.len() >= 6, "bad frame header");
                let bits = seg[0] as u32;
                let (h, w, nc) = (u16_at(seg, 1)?, u16_at(seg, 3)?, seg[5] as usize);
                ensure!((2..=16).contains(&bits) && w > 0 && h > 0 && (1..=4).contains(&nc), "unsupported JPEG-LS frame ({bits} bits, {w}×{h}, {nc} components)");
                let ids = (0..nc).map(|c| seg.get(6 + 3 * c).copied().unwrap_or(c as u8)).collect();
                frame = Some((bits, w, h, ids));
                planes = vec![vec![0; w * h]; nc];
                at += len;
            }
            0xf8 => {
                // LSE: preset coding parameters (id 1); mapping tables are not supported.
                let len = u16_at(data, at)?;
                let seg = data.get(at + 2..at + len).context("truncated LSE segment")?;
                match seg.first() {
                    Some(1) if seg.len() >= 11 => {
                        let v = |k: usize| u16::from_be_bytes([seg[1 + 2 * k], seg[2 + 2 * k]]) as i32;
                        custom = Some([v(0), v(1), v(2), v(3), v(4)]);
                    }
                    Some(id) => bail!("JPEG-LS extension {id} is not supported"),
                    None => {}
                }
                at += len;
            }
            0xda => {
                let (bits, w, h, ids) = frame.clone().context("scan before the frame header")?;
                let len = u16_at(data, at)?;
                let seg = data.get(at + 2..at + len).context("truncated scan header")?;
                let ns = seg[0] as usize;
                ensure!(seg.len() >= 1 + 2 * ns + 3, "bad scan header");
                let comps: Vec<usize> = (0..ns).map(|c| ids.iter().position(|&id| id == seg[1 + 2 * c]).unwrap_or(c)).collect();
                let near = seg[1 + 2 * ns] as i32;
                let ilv = seg[2 + 2 * ns];
                ensure!(seg[3 + 2 * ns] & 0x0f == 0, "JPEG-LS point transform is not supported");
                at += len;
                let p = Params::new(bits, near, custom);
                let mut d = Decoder::new(p, &data[at..]);
                match (ilv, ns) {
                    (0, _) | (_, 1) => {
                        for &c in &comps {
                            // Each component of a non-interleaved image has its own scan.
                            let mut prev = vec![0i32; w + 2];
                            let mut cur = vec![0i32; w + 2];
                            for y in 0..h {
                                d.line(&mut prev, &mut cur)?;
                                planes[c][y * w..(y + 1) * w].copy_from_slice(&cur[1..=w]);
                                std::mem::swap(&mut prev, &mut cur);
                            }
                        }
                    }
                    (1, _) => {
                        let mut prev = vec![vec![0i32; w + 2]; ns];
                        let mut cur = vec![vec![0i32; w + 2]; ns];
                        let mut runs = vec![0usize; ns];
                        for y in 0..h {
                            for (s, &c) in comps.iter().enumerate() {
                                d.run_index = runs[s];
                                d.line(&mut prev[s], &mut cur[s])?;
                                runs[s] = d.run_index;
                                planes[c][y * w..(y + 1) * w].copy_from_slice(&cur[s][1..=w]);
                                std::mem::swap(&mut prev[s], &mut cur[s]);
                            }
                        }
                    }
                    (2, _) => {
                        let mut prev = vec![0i32; (w + 2) * ns];
                        let mut cur = vec![0i32; (w + 2) * ns];
                        for y in 0..h {
                            d.line_interleaved(&mut prev, &mut cur, ns)?;
                            for x in 0..w {
                                for (s, &c) in comps.iter().enumerate() {
                                    planes[c][y * w + x] = cur[(x + 1) * ns + s];
                                }
                            }
                            std::mem::swap(&mut prev, &mut cur);
                        }
                    }
                    _ => bail!("unknown JPEG-LS interleave mode {ilv}"),
                }
                let nc = planes.len();
                // One component needs no interleaving.
                let samples = if nc == 1 { planes[0].iter().map(|&p| p as u16).collect() } else { (0..w * h).flat_map(|i| planes.iter().map(move |p| p[i] as u16)).collect::<Vec<_>>() };
                out = Some(Image { width: w, height: h, components: nc, bits, samples });
            }
            0xd8 | 0x01 | 0xd0..=0xd7 => {}
            0xdd => bail!("JPEG-LS restart intervals are not supported"),
            _ => {
                // Other segments (APPn, COM, …): skip.
                let len = u16_at(data, at)?;
                at += len;
            }
        }
    }
    out.context("no image in the JPEG-LS data")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> (Vec<u8>, Vec<u16>) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/jpegls");
        let jls = std::fs::read(dir.join(format!("{name}.jls"))).unwrap();
        let raw = std::fs::read(dir.join(format!("{name}.u16"))).unwrap();
        (jls, raw.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    #[test]
    fn matches_charls() {
        for (name, w, h, nc, bits) in [("grey8", 37, 23, 1, 8), ("grey12", 40, 30, 1, 12), ("grey16", 33, 17, 1, 16), ("near3", 37, 23, 1, 8), ("rgb", 37, 23, 3, 8)] {
            let (jls, expected) = fixture(name);
            let img = decode(&jls).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_eq!((img.width, img.height, img.components, img.bits), (w, h, nc, bits), "{name}");
            let bad = img.samples.iter().zip(&expected).position(|(a, b)| a != b);
            assert!(bad.is_none(), "{name}: first difference at sample {bad:?}");
            assert_eq!(img.samples.len(), expected.len());
        }
        assert!(decode(b"\xff\xd8\xff\xd9").is_err());
        assert!(decode(b"junk").is_err());
    }

    /// The bit-by-bit count of zeros the decoder used before.
    fn unary_by_bits(b: &mut Bits) -> Result<u32> {
        let mut q = 0;
        while !b.bit() {
            q += 1;
            if q > 64 {
                bail!("corrupt JPEG-LS data");
            }
        }
        Ok(q)
    }

    #[test]
    fn unary_codes_match_bit_by_bit_reading() {
        // Random streams heavy in zero bytes (long runs), 0xFF bytes (stuffed bits) and markers,
        // read as unary codes and plain values in turn, both ways: the codes, the errors and
        // everything left to read must agree.
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for round in 0..3000 {
            let len = 1 + (rnd() % 160) as usize;
            let data: Vec<u8> = (0..len)
                .map(|_| match rnd() % 10 {
                    0..=4 => 0,
                    5 => 0xff,
                    6 => 0x80 | rnd() as u8,
                    _ => rnd() as u8,
                })
                .collect();
            let (mut a, mut b) = (Bits::new(&data), Bits::new(&data));
            for _ in 0..300 {
                match (a.unary(), unary_by_bits(&mut b)) {
                    (Ok(x), Ok(y)) => assert_eq!(x, y, "round {round}"),
                    (Err(_), Err(_)) => break,
                    (x, y) => panic!("round {round}: {x:?} / {y:?}"),
                }
                let unread = |b: &Bits| (b.count, b.pos, b.data.len(), b.prev_ff, if b.count == 0 { 0 } else { b.cache << (64 - b.count) });
                assert_eq!(unread(&a), unread(&b), "round {round}");
                let n = (rnd() % 17) as u32;
                assert_eq!(a.value(n), b.value(n));
            }
        }
    }
}
