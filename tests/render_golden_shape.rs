//! Consistency of the render goldens (tests/golden/dev/render, written by reference/render_golden.py) with the
//! contract: every accepted request packs into a valid `DecisionBatch`, and the recorded token counts, limits and
//! responses agree with each other. This checks the goldens, not the frontend (that is W2's frontend_golden.rs).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use sysone::contract::{DecisionBatch, MAX_SLOTS};

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/dev/render")
}

fn ids(v: &Value) -> Result<Vec<u32>> {
    v.as_array()
        .context("not an array")?
        .iter()
        .map(|x| x.as_u64().map(|x| x as u32).context("not a token id"))
        .collect()
}

fn records() -> Result<Vec<Value>> {
    let prov: Value = serde_json::from_str(&fs::read_to_string(golden_dir().join("provenance.json"))?)?;
    let files = prov["files"].as_object().context("provenance has no file manifest")?;
    let mut out = Vec::new();
    for (name, meta) in files {
        let text = fs::read_to_string(golden_dir().join(name))?;
        let lines: Vec<&str> = text.split('\n').filter(|l| !l.is_empty()).collect();
        ensure!(lines.len() as u64 == meta["cases"].as_u64().unwrap(), "{name}: case count differs from the manifest");
        for l in lines {
            out.push(serde_json::from_str(l).with_context(|| format!("{name}: bad JSON line"))?);
        }
    }
    ensure!(out.len() as u64 == prov["counts"]["cases"].as_u64().unwrap(), "case total differs from provenance");
    Ok(out)
}

/// Checks one `render` object; returns (prefix, suffixes with slot counts) when the split ran.
fn check_render(id: &str, r: &Value, max_branch: u64, max_request: u64) -> Result<(Vec<u32>, Vec<(Vec<u32>, u16)>)> {
    let prefix = ids(&r["prefix_ids"]).with_context(|| format!("{id}: prefix_ids"))?;
    ensure!(r["prefix_tokens"].as_u64() == Some(prefix.len() as u64), "{id}: prefix_tokens");
    let head = r["prompt_head"].as_str().context("prompt_head")?;
    let tail = r["prompt_tail"].as_str().context("prompt_tail")?;
    ensure!(head.starts_with("<|im_start|>system\n") && head.ends_with("</state>\n\n"), "{id}: prompt_head shape");
    ensure!(tail == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "{id}: prompt_tail {tail:?}");
    let mut suffixes = Vec::new();
    for b in r["branches"].as_array().context("branches")? {
        let suffix = ids(&b["suffix_ids"]).with_context(|| format!("{id}: suffix_ids"))?;
        let n = b["n_slots"].as_u64().context("n_slots")?;
        ensure!((1..=MAX_SLOTS as u64).contains(&n), "{id}: n_slots {n}");
        ensure!(b["labels"].as_array().map(|a| a.len() as u64) == Some(n), "{id}: labels vs n_slots");
        ensure!(b["order"].as_array().map(|a| a.len() as u64) == Some(n), "{id}: order vs n_slots");
        let text = b["suffix_text"].as_str().context("suffix_text")?;
        ensure!(text.starts_with("Question: ") || text.starts_with("Rate the state: "), "{id}: suffix header");
        suffixes.push((suffix, n as u16));
    }
    let p = prefix.len() as u64;
    let longest = p + suffixes.iter().map(|(s, _)| s.len() as u64).max().unwrap_or(0);
    let total = p + suffixes.iter().map(|(s, _)| s.len() as u64).sum::<u64>();
    ensure!(r["longest"].as_u64() == Some(longest), "{id}: longest {:?} != {longest}", r["longest"]);
    ensure!(r["total"].as_u64() == Some(total), "{id}: total {:?} != {total}", r["total"]);
    // The reference checks the branch limit first, then the request limit.
    let expected = if longest > max_branch {
        Some(format!("ValueError: state plus longest question is {longest} tokens, limit is {max_branch}"))
    } else if total > max_request {
        Some(format!("ValueError: request is {total} tokens, limit is {max_request}"))
    } else {
        None
    };
    if let Some(e) = &expected {
        ensure!(r["error"].as_str() == Some(e.as_str()), "{id}: error {:?}, expected {e:?}", r["error"]);
        ensure!(!r["split_ran"].as_bool().unwrap(), "{id}: split ran despite a limit error");
    }
    Ok((prefix, suffixes))
}

#[test]
fn render_goldens_are_consistent_with_the_contract() -> Result<()> {
    let prov: Value = serde_json::from_str(&fs::read_to_string(golden_dir().join("provenance.json"))?)?;
    let max_branch = prov["limits"]["max_branch_tokens"].as_u64().unwrap();
    let max_request = prov["limits"]["max_request_tokens"].as_u64().unwrap();
    ensure!(prov["slots"].as_array().map(|a| a.len()) == Some(MAX_SLOTS), "slot table size");
    let (mut accepted, mut packed) = (0, DecisionBatch::default());
    for rec in records()? {
        let id = rec["id"].as_str().unwrap();
        let status = rec["http"]["status"].as_u64().unwrap();
        for (key, r) in [("render", &rec["render"]), ("engine_direct", &rec["engine_direct"]["render"])] {
            if r.is_null() {
                continue;
            }
            let (prefix, suffixes) = check_render(&format!("{id}/{key}"), r, max_branch, max_request)?;
            if key == "render" && status == 200 {
                let body = &rec["http"]["body"];
                ensure!(body["usage"]["input_tokens"] == r["total"], "{id}: usage.input_tokens");
                ensure!(body["qev"]["prefix_tokens"] == r["prefix_tokens"], "{id}: qev.prefix_tokens");
                ensure!(body["qev"]["branches"].as_u64() == Some(suffixes.len() as u64), "{id}: qev.branches");
                ensure!(r["split_ran"].as_bool() == Some(true), "{id}: accepted without _split");
                let refs: Vec<(&[u32], u16)> = suffixes.iter().map(|(s, n)| (s.as_slice(), *n)).collect();
                let mut one = DecisionBatch::default();
                one.push_request(&prefix, &refs)?;
                one.validate().with_context(|| format!("{id}: batch"))?;
                if packed.num_tokens() < 200_000 {
                    packed.push_request(&prefix, &refs)?;
                }
                accepted += 1;
            }
        }
        if status == 422 && !rec["render"].is_null() {
            let msg = rec["http"]["body"]["detail"][0]["msg"].as_str().unwrap();
            let err = rec["render"]["error"].as_str().context("422 with a plan but no recorded error")?;
            ensure!(err == format!("ValueError: {msg}"), "{id}: response {msg:?} vs plan error {err:?}");
        }
    }
    packed.validate()?;
    ensure!(accepted >= 250, "only {accepted} accepted requests");
    Ok(())
}
