//! Query parameters as qev's server reads them: Starlette's `QueryParams` (`urllib.parse.parse_qsl` with
//! `keep_blank_values=True`; a repeated key yields its last value), `bool(...)` for `debug` and Python's `int(...)`
//! for `debias`.

use unicode_general_category::{GeneralCategory, get_general_category};

use crate::frontend::pyjson::py_strip;

/// `parse_qsl(query, keep_blank_values=True)`: pairs split on `&`, `+` as space, `%XX` decoded (invalid UTF-8
/// replaced).
pub fn parse_qsl(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|f| !f.is_empty())
        .map(|f| match f.split_once('=') {
            Some((k, v)) => (unquote_plus(k), unquote_plus(v)),
            None => (unquote_plus(f), String::new()),
        })
        .collect()
}

fn unquote_plus(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => match (hex(b[i + 1]), hex(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h * 16 + l);
                    i += 2;
                }
                _ => out.push(b'%'),
            },
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The last value of `key` (Starlette's `QueryParams.get`).
pub fn get<'a>(params: &'a [(String, String)], key: &str) -> Option<&'a str> {
    params.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// The value of a Unicode decimal digit (category Nd): digits come in runs of ten starting at zero.
fn digit_value(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return Some(c as u32 - '0' as u32);
    }
    if get_general_category(c) != GeneralCategory::DecimalNumber {
        return None;
    }
    let mut run = 0u32;
    let mut cp = c as u32;
    while cp > 0 {
        match char::from_u32(cp - 1) {
            Some(p) if get_general_category(p) == GeneralCategory::DecimalNumber => {
                run += 1;
                cp -= 1;
            }
            _ => break,
        }
    }
    Some(run % 10)
}

/// Python's `int(s)` for a string in base 10, saturated to the i64 range; None where Python raises `ValueError`.
/// Accepts surrounding whitespace, one sign, any Unicode decimal digits, and single underscores between digits.
pub fn py_int(s: &str) -> Option<i64> {
    let s = py_strip(s);
    let (neg, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let mut value: i128 = 0;
    let mut prev_digit = false;
    let mut any = false;
    for c in digits.chars() {
        if c == '_' {
            if !prev_digit {
                return None;
            }
            prev_digit = false;
            continue;
        }
        let d = digit_value(c)?;
        value = (value * 10 + d as i128).min(i64::MAX as i128 + 1);
        prev_digit = true;
        any = true;
    }
    if !any || !prev_digit {
        return None;
    }
    let v = if neg { -value } else { value };
    Some(v.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_int_semantics() {
        assert_eq!(py_int("3"), Some(3));
        assert_eq!(py_int(" -3\n"), Some(-3));
        assert_eq!(py_int("+007"), Some(7));
        assert_eq!(py_int("1_000"), Some(1000));
        assert_eq!(py_int("\u{0663}"), Some(3)); // ARABIC-INDIC DIGIT THREE
        assert_eq!(py_int("\u{FF15}"), Some(5)); // FULLWIDTH DIGIT FIVE
        assert_eq!(py_int("99999999999999999999999"), Some(i64::MAX));
        assert_eq!(py_int("-99999999999999999999999"), Some(i64::MIN));
        for bad in ["", " ", "+", "-", "1.0", "0x10", "_1", "1_", "1__0", "1 2", "--1", "three", "\u{00B2}"] {
            assert_eq!(py_int(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn parse_qsl_like_starlette() {
        let p = parse_qsl("debias=2&debug&x=a+b%21&debias=%33&bad=%zz&&");
        assert_eq!(get(&p, "debias"), Some("3"));
        assert_eq!(get(&p, "debug"), Some(""));
        assert_eq!(get(&p, "x"), Some("a b!"));
        assert_eq!(get(&p, "bad"), Some("%zz"));
        assert_eq!(get(&p, "missing"), None);
        assert_eq!(get(&parse_qsl("a=%"), "a"), Some("%"));
        assert_eq!(get(&parse_qsl("a=%4"), "a"), Some("%4"));
        assert_eq!(get(&parse_qsl("a=%ff"), "a"), Some("\u{FFFD}"));
    }
}
