//! JSON with Python's semantics: what qev's server does to a request body and its response.
//!
//! - [`parse`] is `json.loads` (as Starlette's `request.json()` calls it on the body bytes): object key order is
//!   kept, a duplicate key keeps its first position and its last value, integers have arbitrary precision, and
//!   `NaN`, `Infinity` and `-Infinity` are accepted.
//! - [`Json::dumps_indent2`] is `json.dumps(v, ensure_ascii=False, indent=2)` (qev's `render_entry`/`render_state`)
//!   and [`Json::dumps_compact`] is Starlette's `JSONResponse` rendering, byte for byte. Floats use Python's `repr`.
//! - Helpers for the Python built-ins the frontend relies on: `str.strip`, `repr(str)`, `round(x, 6)` and the
//!   compensated float `sum` of Python 3.12.

use std::collections::HashMap;
use std::fmt::Write as _;

/// A JSON value as Python's `json` module holds it.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer literal in canonical form (optional `-`, no leading zeros, no `-0`), any size.
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    /// Keys in first-insertion order, unique.
    Object(Vec<(String, Json)>),
}

/// Python refuses to convert integers with more digits than this (`sys.get_int_max_str_digits()`), so `json.loads`
/// fails on them.
const MAX_INT_DIGITS: usize = 4300;
/// Nesting limit. CPython's scanner raises `RecursionError` near its recursion limit; deeper inputs are refused
/// here as invalid JSON instead of risking the worker's stack.
const MAX_DEPTH: usize = 900;

impl Json {
    pub fn int(v: i64) -> Self {
        Json::Int(v.to_string())
    }

    pub fn str(s: impl Into<String>) -> Self {
        Json::Str(s.into())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// The value of `key` in an object (None for a missing key or a non-object).
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// `json.dumps(self, ensure_ascii=False, indent=2)`.
    pub fn dumps_indent2(&self) -> String {
        let mut out = String::new();
        self.write_indent2(&mut out, 0);
        out
    }

    /// `json.dumps(self, ensure_ascii=False, separators=(",", ":"))`, as Starlette's `JSONResponse` renders a body.
    /// Errors on NaN or an infinity, like its `allow_nan=False`.
    pub fn dumps_compact(&self) -> Result<String, String> {
        let mut out = String::new();
        self.write_compact(&mut out)?;
        Ok(out)
    }

    fn write_indent2(&self, out: &mut String, level: usize) {
        match self {
            Json::Array(a) if !a.is_empty() => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    push_indent(out, level + 1);
                    v.write_indent2(out, level + 1);
                }
                out.push('\n');
                push_indent(out, level);
                out.push(']');
            }
            Json::Object(m) if !m.is_empty() => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    push_indent(out, level + 1);
                    write_str(out, k);
                    out.push_str(": ");
                    v.write_indent2(out, level + 1);
                }
                out.push('\n');
                push_indent(out, level);
                out.push('}');
            }
            other => other.write_scalar(out, true).expect("allow_nan"),
        }
    }

    fn write_compact(&self, out: &mut String) -> Result<(), String> {
        match self {
            Json::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write_compact(out)?;
                }
                out.push(']');
            }
            Json::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(out, k);
                    out.push(':');
                    v.write_compact(out)?;
                }
                out.push('}');
            }
            other => other.write_scalar(out, false)?,
        }
        Ok(())
    }

    fn write_scalar(&self, out: &mut String, allow_nan: bool) -> Result<(), String> {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(s) => out.push_str(s),
            Json::Float(f) if !f.is_finite() => {
                if !allow_nan {
                    return Err(format!("Out of range float values are not JSON compliant: {}", py_float_repr(*f)));
                }
                out.push_str(if f.is_nan() { "NaN" } else if *f > 0.0 { "Infinity" } else { "-Infinity" });
            }
            Json::Float(f) => out.push_str(&py_float_repr(*f)),
            Json::Str(s) => write_str(out, s),
            // Non-empty containers are written by the caller.
            Json::Array(_) => out.push_str("[]"),
            Json::Object(_) => out.push_str("{}"),
        }
        Ok(())
    }
}

fn push_indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

/// A JSON string as `json.encoder.encode_basestring` writes it (ensure_ascii=False): only `"`, `\` and the C0
/// controls are escaped.
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    let mut last = 0;
    for (i, b) in s.bytes().enumerate() {
        let esc: &str = match b {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            b'\n' => "\\n",
            b'\r' => "\\r",
            b'\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            0..=0x1f => "",
            _ => continue,
        };
        out.push_str(&s[last..i]);
        if esc.is_empty() {
            let _ = write!(out, "\\u{:04x}", b);
        } else {
            out.push_str(esc);
        }
        last = i + 1;
    }
    out.push_str(&s[last..]);
    out.push('"');
}

/// Python's `repr(float)` (shortest round-trip digits; exponent form when the decimal exponent is < -4 or >= 16).
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` prints the shortest round-trip digits as d[.ddd]e<exp>.
    let e = format!("{:e}", x.abs());
    let (mant, exp) = e.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let n = digits.len() as i32;
    let decpt = exp + 1;
    let mut out = String::with_capacity(24);
    if x < 0.0 {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt >= n {
        out.push_str(&digits);
        for _ in 0..decpt - n {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// Python's `round(x, 6)` for a float: the exactly-rounded decimal (ties to even), read back as a float.
pub fn py_round6(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    // A tie at the 7th decimal needs x * 10^7 to be an odd multiple of 5, so x is a multiple of 1/128. Those are
    // handled exactly with integers; every other value has an unambiguous rounding, which `{:.6}` gets right.
    let t = x * 128.0;
    if t.fract() == 0.0 && t.abs() < 1e15 {
        let x7 = (t as i128) * 78_125; // x * 10^7, exact
        if x7 % 10 != 0 && x7.abs() % 10 == 5 {
            let mut q = x7 / 10; // truncated toward zero
            if q % 2 != 0 {
                q += x7.signum();
            }
            return format!("{q}e-6").parse().expect("decimal");
        }
    }
    format!("{x:.6}").parse().expect("decimal")
}

/// Python 3.12's `sum()` over floats (Neumaier compensated summation, as in `builtin_sum_impl`), starting from the
/// integer 0.
pub fn py_fsum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut it = xs.into_iter();
    let Some(first) = it.next() else { return 0.0 };
    // `0 + x`: an int start turns into the first float (and -0.0 into 0.0).
    let mut f = 0.0 + first;
    let mut c = 0.0f64;
    for x in it {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// Python's `str.isspace` for one character (what `str.strip()` removes).
pub fn py_isspace(c: char) -> bool {
    matches!(c,
        '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c'..='\x1f' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}'
        | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

/// Python's `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// Python's `str.isprintable` for one character.
fn py_isprintable(c: char) -> bool {
    use unicode_general_category::{GeneralCategory as G, get_general_category};
    if c == ' ' {
        return true;
    }
    !matches!(
        get_general_category(c),
        G::Control | G::Format | G::Surrogate | G::PrivateUse | G::Unassigned | G::LineSeparator
            | G::ParagraphSeparator | G::SpaceSeparator
    )
}

/// Python's `repr(str)`.
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c if c.is_ascii() || py_isprintable(c) => out.push(c),
            c if (c as u32) <= 0xff => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c if (c as u32) <= 0xffff => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => {
                let _ = write!(out, "\\U{:08x}", c as u32);
            }
        }
    }
    out.push(quote);
    out
}

/// A body `json.loads` refuses (Starlette's `request.json()` raises, and qev answers "invalid JSON").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub pos: usize,
    pub msg: &'static str,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.msg, self.pos)
    }
}

impl std::error::Error for ParseError {}

/// `json.loads(body)` for a request body. A UTF-8 BOM is skipped (Python detects `utf-8-sig`); other encodings
/// are refused.
pub fn parse(body: &[u8]) -> Result<Json, ParseError> {
    let body = body.strip_prefix(b"\xef\xbb\xbf").unwrap_or(body);
    let text = std::str::from_utf8(body).map_err(|e| ParseError { pos: e.valid_up_to(), msg: "invalid UTF-8" })?;
    parse_str(text)
}

/// `json.loads(text)`.
pub fn parse_str(text: &str) -> Result<Json, ParseError> {
    let mut p = Parser { s: text.as_bytes(), i: 0, depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("extra data"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &'static str) -> ParseError {
        ParseError { pos: self.i, msg }
    }

    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Json, ParseError> {
        match self.s.get(self.i) {
            None => Err(self.err("expecting value")),
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'N') if self.eat("NaN") => Ok(Json::Float(f64::NAN)),
            Some(b'I') if self.eat("Infinity") => Ok(Json::Float(f64::INFINITY)),
            Some(b'-') if self.eat("-Infinity") => Ok(Json::Float(f64::NEG_INFINITY)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("expecting value")),
        }
    }

    fn nested(&mut self, f: fn(&mut Self) -> Result<Json, ParseError>) -> Result<Json, ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        let v = f(self);
        self.depth -= 1;
        v
    }

    fn object(&mut self) -> Result<Json, ParseError> {
        self.i += 1;
        let mut items: Vec<(String, Json)> = Vec::new();
        // Index for duplicate detection once an object gets large; small objects scan linearly.
        let mut index: Option<HashMap<String, usize>> = None;
        self.ws();
        if self.eat("}") {
            return Ok(Json::Object(items));
        }
        loop {
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("expecting property name enclosed in double quotes"));
            }
            let key = self.string()?;
            self.ws();
            if !self.eat(":") {
                return Err(self.err("expecting ':' delimiter"));
            }
            self.ws();
            let v = self.value()?;
            if index.is_none() && items.len() >= 32 {
                index = Some(items.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect());
            }
            let found = match &index {
                Some(ix) => ix.get(&key).copied(),
                None => items.iter().position(|(k, _)| *k == key),
            };
            match found {
                Some(at) => items[at].1 = v,
                None => {
                    if let Some(ix) = &mut index {
                        ix.insert(key.clone(), items.len());
                    }
                    items.push((key, v));
                }
            }
            self.ws();
            if self.eat(",") {
                self.ws();
                continue;
            }
            if self.eat("}") {
                return Ok(Json::Object(items));
            }
            return Err(self.err("expecting ',' delimiter"));
        }
    }

    fn array(&mut self) -> Result<Json, ParseError> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            if self.eat(",") {
                self.ws();
                continue;
            }
            if self.eat("]") {
                return Ok(Json::Array(items));
            }
            return Err(self.err("expecting ',' delimiter"));
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| self.err("invalid \\uXXXX escape"))?;
        let h = std::str::from_utf8(h).map_err(|_| self.err("invalid \\uXXXX escape"))?;
        if !h.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(self.err("invalid \\uXXXX escape"));
        }
        self.i += 4;
        Ok(u32::from_str_radix(h, 16).expect("hex"))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(&b) = self.s.get(self.i) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.i += 1;
            }
            // Safe: the input is valid UTF-8 and we stopped on an ASCII byte.
            out.push_str(std::str::from_utf8(&self.s[start..self.i]).expect("utf-8"));
            match self.s.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let Some(&e) = self.s.get(self.i) else { return Err(self.err("unterminated string")) };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\x08'),
                        b'f' => out.push('\x0c'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut c = self.hex4()?;
                            if (0xd800..0xdc00).contains(&c) && self.s[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    c = 0x10000 + ((c - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    self.i = save;
                                }
                            }
                            // Python keeps a lone surrogate in the str; a Rust String cannot hold one.
                            out.push(char::from_u32(c).ok_or_else(|| self.err("lone surrogate (not supported)"))?);
                        }
                        _ => return Err(self.err("invalid \\escape")),
                    }
                }
                Some(_) => return Err(self.err("invalid control character")),
            }
        }
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let s = p.i;
            while let Some(b'0'..=b'9') = p.s.get(p.i) {
                p.i += 1;
            }
            p.i - s
        };
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return Err(self.err("expecting value")),
        }
        let int_end = self.i;
        let mut is_float = false;
        // A '.' or exponent without digits is not part of the number (Python then fails on the extra data).
        if self.s.get(self.i) == Some(&b'.') && matches!(self.s.get(self.i + 1), Some(b'0'..=b'9')) {
            self.i += 1;
            digits(self);
            is_float = true;
        }
        if let Some(b'e' | b'E') = self.s.get(self.i) {
            let save = self.i;
            self.i += 1;
            if let Some(b'+' | b'-') = self.s.get(self.i) {
                self.i += 1;
            }
            if digits(self) == 0 {
                self.i = save;
            } else {
                is_float = true;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).expect("ascii");
        if is_float {
            return Ok(Json::Float(text.parse().expect("float literal")));
        }
        let lit = &self.s[start..int_end];
        let (neg, mag) = match lit.strip_prefix(b"-") {
            Some(m) => (true, m),
            None => (false, lit),
        };
        if mag.len() > MAX_INT_DIGITS {
            return Err(self.err("integer too large for Python's int conversion"));
        }
        let mag = std::str::from_utf8(mag).expect("ascii");
        Ok(Json::Int(if neg && mag != "0" { format!("-{mag}") } else { mag.to_string() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-0.0, "-0.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (5e-324, "5e-324"),
            (1e21, "1e+21"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (323.47, "323.47"),
            (2.5, "2.5"),
            (100000.0, "100000.0"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    fn dumps_indent2_matches_python() {
        let v = parse_str(r#"{"a": [1, 2.50, {"b": null, "c": []}], "d": {}, "e": "x\"\n\u0001é\u2028", "f": -0}"#).unwrap();
        assert_eq!(
            v.dumps_indent2(),
            "{\n  \"a\": [\n    1,\n    2.5,\n    {\n      \"b\": null,\n      \"c\": []\n    }\n  ],\n  \"d\": {},\n  \
             \"e\": \"x\\\"\\n\\u0001é\u{2028}\",\n  \"f\": 0\n}"
        );
        assert_eq!(v.dumps_compact().unwrap(), "{\"a\":[1,2.5,{\"b\":null,\"c\":[]}],\"d\":{},\"e\":\"x\\\"\\n\\u0001é\u{2028}\",\"f\":0}");
    }

    #[test]
    fn parse_follows_python_json_loads() {
        // Duplicate keys: first position, last value.
        let v = parse_str(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(v, Json::Object(vec![("a".into(), Json::int(3)), ("b".into(), Json::int(2))]));
        assert_eq!(parse_str("18446744073709551617").unwrap(), Json::Int("18446744073709551617".into()));
        assert_eq!(parse_str("-0").unwrap(), Json::Int("0".into()));
        assert_eq!(parse_str("1E400").unwrap(), Json::Float(f64::INFINITY));
        assert!(matches!(parse_str("NaN").unwrap(), Json::Float(f) if f.is_nan()));
        assert_eq!(parse_str("\"\\ud83d\\ude00\"").unwrap(), Json::str("😀"));
        for bad in ["", "01", "1.", "[1,]", "{\"a\":1,}", "\"a\nb\"", "tru", "{'a':1}", "[1] x", "\"\\x\""] {
            assert!(parse_str(bad).is_err(), "{bad:?} should fail");
        }
        assert_eq!(parse(b"\xef\xbb\xbf[]").unwrap(), Json::Array(vec![]));
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert!(parse_str(&deep).is_err());
    }

    #[test]
    fn python_builtins() {
        assert_eq!(py_strip("\u{1c} a b\u{a0}\u{3000}\n"), "a b");
        assert_eq!(py_strip("\u{200b}x"), "\u{200b}x");
        assert_eq!(py_repr_str("it's"), "\"it's\"");
        assert_eq!(py_repr_str("a'\"b\n\x7f\u{85}é\u{200b}😀"), "'a\\'\"b\\n\\x7f\\x85é\\u200b😀'");
        assert_eq!(py_round6(0.0078125), 0.007812);
        assert_eq!(py_round6(0.0234375), 0.023438);
        assert_eq!(py_round6(0.1234564999), 0.123456);
        assert_eq!(py_round6(-0.0078125), -0.007812);
        assert_eq!(py_fsum([1e16, 1.0, -1e16]), 1.0);
        assert_eq!(py_fsum([0.1; 10]), 1.0);
    }
}
