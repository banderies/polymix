//! Allocation-free lexing helpers shared by the ASCII parsers: line and token iterators
//! over byte slices plus fast number parsing.

/// Horizontal whitespace (everything that separates tokens within a line).
#[inline]
pub(crate) fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | 0x0b | 0x0c)
}

/// Any whitespace including newlines.
#[inline]
pub(crate) fn is_ws(b: u8) -> bool {
    is_space(b) || b == b'\n'
}

/// Strip a UTF-8 byte-order mark.
pub(crate) fn strip_bom(data: &[u8]) -> &[u8] {
    data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(data)
}

/// Trim leading and trailing whitespace.
pub(crate) fn trim(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|&b| !is_ws(b)).unwrap_or(s.len());
    let end = s.iter().rposition(|&b| !is_ws(b)).map_or(start, |e| e + 1);
    &s[start..end]
}

/// Index of the first occurrence of `needle` (word-at-a-time scan).
#[inline]
pub(crate) fn find_byte(hay: &[u8], needle: u8) -> Option<usize> {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let rep = LO.wrapping_mul(needle as u64);
    let mut i = 0;
    while i + 8 <= hay.len() {
        let w = u64::from_le_bytes(hay[i..i + 8].try_into().unwrap()) ^ rep;
        // The lowest set bit marks the first zero byte exactly.
        let z = w.wrapping_sub(LO) & !w & HI;
        if z != 0 {
            return Some(i + (z.trailing_zeros() / 8) as usize);
        }
        i += 8;
    }
    hay[i..].iter().position(|&b| b == needle).map(|p| p + i)
}

/// Line iterator: yields lines without the terminator (a trailing `\r` is removed).
pub(crate) struct Lines<'a> {
    data: &'a [u8],
    pos: usize,
    /// 1-based number of the line most recently returned.
    pub line_no: usize,
}

impl<'a> Lines<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Lines { data, pos: 0, line_no: 0 }
    }
    /// Byte offset of the start of the next line.
    pub fn offset(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = &'a [u8];
    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.pos >= self.data.len() {
            return None;
        }
        let rest = &self.data[self.pos..];
        let (line, adv) = match find_byte(rest, b'\n') {
            Some(i) => (&rest[..i], i + 1),
            None => (rest, rest.len()),
        };
        self.pos += adv;
        self.line_no += 1;
        Some(line.strip_suffix(b"\r").unwrap_or(line))
    }
}

/// Whitespace-separated tokens of a byte slice (newlines count as whitespace).
pub(crate) struct Tokens<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Tokens<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Tokens { data, pos: 0 }
    }
    /// Unconsumed remainder (starting at the current position).
    pub fn rest(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }
    /// Bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }
    /// Skip to just after the next newline.
    pub fn skip_line(&mut self) {
        match find_byte(&self.data[self.pos..], b'\n') {
            Some(i) => self.pos += i + 1,
            None => self.pos = self.data.len(),
        }
    }
    /// 1-based line number of the current position (O(n); for error messages only).
    pub fn line_no(&self) -> usize {
        1 + self.data[..self.pos].iter().filter(|&&b| b == b'\n').count()
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = &'a [u8];
    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        let d = self.data;
        let mut i = self.pos;
        while i < d.len() && is_ws(d[i]) {
            i += 1;
        }
        if i >= d.len() {
            self.pos = i;
            return None;
        }
        let s = i;
        while i < d.len() && !is_ws(d[i]) {
            i += 1;
        }
        self.pos = i;
        Some(&d[s..i])
    }
}

/// Printable form of a token for error messages.
pub(crate) fn show(tok: &[u8]) -> String {
    let s = String::from_utf8_lossy(&tok[..tok.len().min(40)]).into_owned();
    if tok.len() > 40 {
        s + "..."
    } else {
        s
    }
}

const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18,
    1e19, 1e20, 1e21, 1e22,
];

/// Parse a complete token as `f64`.
///
/// Short decimals (≤ 19 significant digits, mantissa ≤ 2^53, |exp| ≤ 22) take Clinger's
/// fast path, which is exact because both operands are exactly representable and IEEE
/// multiplication/division round correctly. Anything else (long mantissas, large
/// exponents, `inf`/`nan`) falls back to the standard library's correctly rounded parser.
#[inline]
pub(crate) fn parse_f64(s: &[u8]) -> Option<f64> {
    let n = s.len();
    let mut i = 0;
    let neg = match s.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let mut mant: u64 = 0;
    let mut ndig = 0u32;
    let mut exp: i32 = 0;
    let mut any = false;
    let mut overflow = false;
    while i < n {
        let d = s[i].wrapping_sub(b'0');
        if d > 9 {
            break;
        }
        any = true;
        if ndig < 19 {
            mant = mant * 10 + d as u64;
            if mant != 0 {
                ndig += 1;
            }
        } else {
            overflow = true;
        }
        i += 1;
    }
    if i < n && s[i] == b'.' {
        i += 1;
        while i < n {
            let d = s[i].wrapping_sub(b'0');
            if d > 9 {
                break;
            }
            any = true;
            if ndig < 19 {
                mant = mant * 10 + d as u64;
                if mant != 0 {
                    ndig += 1;
                }
                exp -= 1;
            } else {
                overflow = true;
            }
            i += 1;
        }
    }
    if !any {
        return parse_f64_slow(s);
    }
    if i < n && (s[i] | 0x20) == b'e' {
        i += 1;
        let eneg = match s.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let start = i;
        let mut e: i32 = 0;
        while i < n {
            let d = s[i].wrapping_sub(b'0');
            if d > 9 {
                break;
            }
            if e < 100_000 {
                e = e * 10 + d as i32;
            }
            i += 1;
        }
        if i == start {
            return None;
        }
        exp += if eneg { -e } else { e };
    }
    if i != n {
        return parse_f64_slow(s);
    }
    if mant == 0 && !overflow {
        return Some(if neg { -0.0 } else { 0.0 });
    }
    if !overflow && mant <= (1u64 << 53) && (-22..=22).contains(&exp) {
        let v = mant as f64;
        let v = if exp >= 0 { v * POW10[exp as usize] } else { v / POW10[(-exp) as usize] };
        return Some(if neg { -v } else { v });
    }
    parse_f64_slow(s)
}

#[cold]
fn parse_f64_slow(s: &[u8]) -> Option<f64> {
    std::str::from_utf8(s).ok()?.parse::<f64>().ok()
}

/// Parse a complete token as a signed integer (no overflow, at least one digit).
#[inline]
pub(crate) fn parse_i64(s: &[u8]) -> Option<i64> {
    let (neg, digits) = match s.first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let mut v: i64 = 0;
    for &b in digits {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_sub(d as i64)?;
    }
    // Accumulated negatively so that i64::MIN parses.
    if neg {
        Some(v)
    } else {
        v.checked_neg()
    }
}

/// Parse an integer token, also accepting integral floats such as `3.0`.
pub(crate) fn parse_int_lenient(s: &[u8]) -> Option<i64> {
    parse_i64(s).or_else(|| {
        let f = parse_f64(s)?;
        (f.fract() == 0.0 && f.abs() < 9.0e15).then_some(f as i64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_match_std() {
        let cases = [
            "0", "-0", "1", "-1", "1.5", "+2.25", ".5", "5.", "1e3", "1E-3", "-1.25e+10", "3.14159265358979",
            "0.1", "0.2", "0.30000000000000004", "123456789012345678901234", "1e-300", "1e308", "2.2250738585072014e-308",
            "9007199254740993", "0.000001234", "-0.0000000000000000000000001", "1.7976931348623157e308",
            "4.9e-324", "inf", "-inf", "NaN", "12345678901234567890.123", "0000.0001", "1.000000000000000000000001",
        ];
        for c in cases {
            let a = parse_f64(c.as_bytes());
            let b: Option<f64> = c.parse().ok();
            match (a, b) {
                (Some(a), Some(b)) if a.is_nan() && b.is_nan() => {}
                (a, b) => assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits), "{c}"),
            }
        }
        for bad in ["", "-", "+", ".", "e5", "1e", "1e+", "1.2.3", "abc", "1x", "--1", "1 "] {
            assert!(parse_f64(bad.as_bytes()).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn float_roundtrip_random() {
        // Deterministic LCG over f32 / f64 values formatted via Display and LowerExp.
        let mut s: u64 = 0x1234_5678;
        for _ in 0..20000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x32 = f32::from_bits((s >> 32) as u32);
            let x64 = f64::from_bits(s);
            for v in [x32 as f64, x64, (s % 100000) as f64 / 997.0] {
                if !v.is_finite() {
                    continue;
                }
                for t in [format!("{v}"), format!("{v:e}"), format!("{}", v as f32), format!("{:.6}", v)] {
                    let a = parse_f64(t.as_bytes()).unwrap();
                    let b: f64 = t.parse().unwrap();
                    assert_eq!(a.to_bits(), b.to_bits(), "{t}");
                }
            }
        }
    }

    #[test]
    fn ints() {
        assert_eq!(parse_i64(b"0"), Some(0));
        assert_eq!(parse_i64(b"-17"), Some(-17));
        assert_eq!(parse_i64(b"+5"), Some(5));
        assert_eq!(parse_i64(b"-9223372036854775808"), Some(i64::MIN));
        assert_eq!(parse_i64(b"9223372036854775808"), None);
        assert_eq!(parse_i64(b""), None);
        assert_eq!(parse_i64(b"-"), None);
        assert_eq!(parse_i64(b"1.0"), None);
        assert_eq!(parse_int_lenient(b"3.0"), Some(3));
        assert_eq!(parse_int_lenient(b"3.5"), None);
    }

    #[test]
    fn lexing() {
        let d = b"a b\r\n\r\n  c\td  \nlast";
        let lines: Vec<&[u8]> = Lines::new(d).collect();
        assert_eq!(lines, vec![&b"a b"[..], b"", b"  c\td  ", b"last"]);
        let toks: Vec<&[u8]> = Tokens::new(d).collect();
        assert_eq!(toks, vec![&b"a"[..], b"b", b"c", b"d", b"last"]);
        assert_eq!(trim(b"  x y \r"), b"x y");
        assert_eq!(trim(b"   "), b"");
        for n in 0..40 {
            let mut v = vec![b'a'; n];
            assert_eq!(find_byte(&v, b'\n'), None);
            v.push(b'\n');
            v.extend_from_slice(b"zz\n");
            assert_eq!(find_byte(&v, b'\n'), Some(n));
        }
        let mut t = Tokens::new(b"solid my name\nfacet");
        assert_eq!(t.next(), Some(&b"solid"[..]));
        t.skip_line();
        assert_eq!(t.next(), Some(&b"facet"[..]));
        assert_eq!(t.line_no(), 2);
    }
}
