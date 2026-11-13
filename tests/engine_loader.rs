//! Loader and weights against the real checkpoints (DESIGN §3, §3.1, §6.5; PLATFORM.md P2).
//!
//! - Headers: text weights only, unused shards skipped.
//! - Load with page-cache dropping: load time, bytes, and page-cache residency of the checkpoint files
//!   before/after (`mincore`) plus `/proc/meminfo` `Cached:`.
//! - Byte-for-byte validation of sampled device tensors against the safetensors file, including every part
//!   of each fused projection, the NVFP4 packed/scale/global-scale tensors and the gathered slot rows.
//!
//! The checkpoints live outside the repo; set `SYSONE_BF16_DIR` / `SYSONE_NVFP4_DIR` to override the
//! default paths. Tests skip (with a message) when a checkpoint is absent. Timing runs should hold the GPU
//! lock: `flock /tmp/sysone-gpu.lock cargo test --release --test engine_loader -- --nocapture --test-threads=1`.

use cuda_core::Context;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use sysone::engine::config::ModelConfig;
use sysone::engine::graph::tensor::DevTensor;
use sysone::engine::loader::{
    CacheMode, Checkpoint, StreamOptions, device_is_integrated, drop_file_cache, file_resident_bytes,
    meminfo_cached_available,
};
use sysone::engine::weights::{Layer, Linear, LoadOptions, ModelWeights};

fn dir(var: &str, default: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.into()));
    if p.join("config.json").exists() {
        Some(p)
    } else {
        println!("skipping: {} not found (set {var})", p.display());
        None
    }
}

fn bf16_dir() -> Option<PathBuf> {
    dir("SYSONE_BF16_DIR", "/home/velvet/Desktop/hf_models/elephant-1")
}

fn nvfp4_dir() -> Option<PathBuf> {
    dir("SYSONE_NVFP4_DIR", "/home/velvet/Desktop/hf_models/elephant-1-nvfp4-mlp")
}

/// Bytes `[off, off + len)` of a tensor, read straight from its file (independent of the streamer).
fn file_bytes(ck: &Checkpoint, name: &str, off: usize, len: usize) -> Vec<u8> {
    let t = ck.get(name).unwrap();
    assert!(off + len <= t.bytes());
    let f = File::open(ck.file_path(t)).unwrap();
    let mut b = vec![0u8; len];
    f.read_exact_at(&mut b, t.start + off as u64).unwrap();
    b
}

struct Checker<'a> {
    ck: &'a Checkpoint,
    stream: std::sync::Arc<cuda_core::Stream>,
    bytes: usize,
    tensors: usize,
}

impl Checker<'_> {
    /// Compare rows `rows` of device tensor `dev` (rows starting at `dev_row0`) with rows of file tensor `name`.
    fn rows(&mut self, dev: &DevTensor, dev_row0: usize, name: &str, rows: std::ops::Range<usize>) {
        let row_bytes = dev.cols() * dev.dtype.size_in_bytes();
        let got = dev.rows_view(dev_row0 + rows.start, rows.len()).unwrap().to_host_bytes(&self.stream).unwrap();
        let want = file_bytes(self.ck, name, rows.start * row_bytes, rows.len() * row_bytes);
        assert!(got == want, "{name} rows {rows:?} differ from the file");
        self.bytes += got.len();
    }

    /// First, middle and last rows (or all rows if `full`) of each part.
    fn sampled(&mut self, dev: &DevTensor, dev_row0: usize, name: &str, nrows: usize, full: bool) {
        if full || nrows <= 48 {
            self.rows(dev, dev_row0, name, 0..nrows);
        } else {
            for s in [0, nrows / 2 - 8, nrows - 16] {
                self.rows(dev, dev_row0, name, s..s + 16);
            }
        }
        self.tensors += 1;
    }

    fn whole(&mut self, dev: &DevTensor, name: &str) {
        let n = if dev.shape.is_empty() { 1 } else { dev.shape[0] };
        let d = if dev.shape.len() == 1 { dev.reshape(&[n, 1]).unwrap() } else { dev.clone() };
        let d = if d.shape.len() > 2 { d.reshape(&[n, d.cols()]).unwrap() } else { d };
        self.sampled(&d, 0, name, n, true);
    }

    fn linear(&mut self, l: &Linear, full: bool) {
        match l {
            Linear::Bf16 { w, parts } => {
                for p in parts {
                    self.sampled(w, p.row0, &format!("{}.weight", p.name), p.rows, full);
                }
            }
            Linear::Nvfp4 { packed, scales, parts, a_gs, .. } => {
                for p in parts {
                    self.sampled(packed, p.row0, &format!("{}.weight_packed", p.name), p.rows, full);
                    self.sampled(scales, p.row0, &format!("{}.weight_scale", p.name), p.rows, full);
                    let w_gs = self.ck.read_f32_scalar(&format!("{}.weight_global_scale", p.name)).unwrap();
                    let i_gs = self.ck.read_f32_scalar(&format!("{}.input_global_scale", p.name)).unwrap();
                    assert_eq!(p.w_gs.unwrap().to_bits(), w_gs.to_bits(), "{} w_gs", p.name);
                    assert_eq!(a_gs.to_bits(), i_gs.to_bits(), "{} a_gs", p.name);
                }
            }
        }
    }
}

fn files_resident(ck: &Checkpoint) -> u64 {
    ck.files.iter().map(|f| file_resident_bytes(f).unwrap()).sum()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn slot_ids(vocab: usize) -> Vec<u32> {
    // 256 distinct pseudo-random ids, including the first and last rows of the head.
    let mut ids: Vec<u32> = (0..254u64).map(|i| ((i * 2654435761 + 977) % vocab as u64) as u32).collect();
    ids.push(0);
    ids.push(vocab as u32 - 1);
    ids
}

fn load_and_check(model_dir: &Path, mode: CacheMode) {
    let cfg = ModelConfig::load(model_dir).unwrap();
    let ck = Checkpoint::open(model_dir).unwrap();
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    println!("== {} ({mode:?}); integrated device: {}", model_dir.display(), device_is_integrated(&ctx).unwrap());
    println!("files used: {:?}; skipped shards: {:?}", ck.files, ck.skipped_files);

    // Cold start: drop whatever the page cache holds of these files.
    for f in &ck.files {
        drop_file_cache(f).unwrap();
    }
    let (cached0, avail0) = meminfo_cached_available().unwrap();
    let res0 = files_resident(&ck);

    let ids = slot_ids(cfg.text_config.vocab_size);
    let opts = LoadOptions {
        slot_token_ids: Some(ids.clone()),
        full_lm_head: false,
        stream: StreamOptions { mode, ..Default::default() },
    };
    let (w, rep) = ModelWeights::load(&cfg, &ck, &stream, &opts).unwrap();
    let (cached1, avail1) = meminfo_cached_available().unwrap();
    let res1 = files_resident(&ck);
    let s = &rep.stream;
    println!(
        "load: {:.2} GiB to device ({} copies, {} sequential runs, O_DIRECT={}) in {:.2} s = {:.2} GB/s copied \
         ({:.2} GiB read); {} NVFP4 projections",
        gib(s.bytes_copied), s.items, s.runs, s.direct, s.seconds, s.bytes_copied as f64 / s.seconds / 1e9,
        gib(s.bytes_read), rep.nvfp4_projections
    );
    println!("device weights: {:.2} GiB", gib(rep.device_bytes as u64));
    println!(
        "page cache of checkpoint files: before {:.3} GiB, after {:.3} GiB; meminfo Cached: {:.2} -> {:.2} GiB \
         (Δ {:+.2} GiB, other processes included); MemAvailable {:.2} -> {:.2} GiB",
        gib(res0), gib(res1), gib(cached0), gib(cached1), (cached1 as f64 - cached0 as f64) / (1u64 << 30) as f64,
        gib(avail0), gib(avail1)
    );
    // No page-cache copy of the weights may stay resident (P2): the load must not add resident pages.
    // (`before` can be non-zero when another process has the file mapped: such pages cannot be dropped.)
    assert!(res1 <= res0 + (16 << 20), "the load left {:.3} GiB of the checkpoint in the page cache",
            gib(res1.saturating_sub(res0)));

    // Byte-for-byte validation.
    let mut c = Checker { ck: &ck, stream: stream.clone(), bytes: 0, tensors: 0 };
    let lm = "model.language_model";
    c.sampled(&w.embed, 0, &format!("{lm}.embed_tokens.weight"), cfg.text_config.vocab_size, false);
    c.whole(&w.final_norm, &format!("{lm}.norm.weight"));
    for (i, l) in w.layers.iter().enumerate() {
        let p = format!("{lm}.layers.{i}");
        c.whole(l.input_norm(), &format!("{p}.input_layernorm.weight"));
        c.whole(l.post_norm(), &format!("{p}.post_attention_layernorm.weight"));
        let full = i == 0 || i == 3; // one GDN and one attention layer compared in full
        c.linear(&l.mlp().gate_up, full);
        c.linear(&l.mlp().down, full);
        match l {
            Layer::Gdn(g) => {
                c.linear(&g.in_proj, full);
                c.linear(&g.out_proj, full);
                c.whole(&g.conv_w, &format!("{p}.linear_attn.conv1d.weight"));
                c.whole(&g.a_log, &format!("{p}.linear_attn.A_log"));
                c.whole(&g.dt_bias, &format!("{p}.linear_attn.dt_bias"));
                c.whole(&g.norm_w, &format!("{p}.linear_attn.norm.weight"));
                assert_eq!(g.in_proj.out_features(), 12352);
            }
            Layer::Attn(a) => {
                c.linear(&a.q_proj, full);
                c.linear(&a.kv_proj, full);
                c.linear(&a.o_proj, full);
                c.whole(&a.q_norm, &format!("{p}.self_attn.q_norm.weight"));
                c.whole(&a.k_norm, &format!("{p}.self_attn.k_norm.weight"));
                assert_eq!(a.kv_proj.out_features(), 2048);
            }
        }
        assert_eq!(l.mlp().gate_up.out_features(), 24576);
    }
    // Slot rows: row i of the slot head is lm_head[ids[i]].
    let sh = w.slot_head.as_ref().unwrap();
    let got = sh.w.to_host_bytes(&stream).unwrap();
    let rb = cfg.text_config.hidden_size * 2;
    for (i, &id) in ids.iter().enumerate() {
        assert!(got[i * rb..(i + 1) * rb] == file_bytes(&ck, "lm_head.weight", id as usize * rb, rb), "slot row {i} (id {id})");
    }
    println!(
        "validated byte-for-byte: {} tensors/parts, {:.1} MiB sampled + {} slot rows",
        c.tensors, c.bytes as f64 / (1 << 20) as f64, ids.len()
    );
    // The validation reads went through the page cache; leave it as found.
    for f in &ck.files {
        drop_file_cache(f).unwrap();
    }
}

#[test]
fn headers_text_only() {
    let Some(d) = nvfp4_dir() else { return };
    let ck = Checkpoint::open(&d).unwrap();
    assert_eq!(ck.files.len(), 1, "only the backbone shard is used");
    assert_eq!(ck.skipped_files.len(), 1, "the vision shard is skipped");
    assert!(ck.names().all(|n| !n.contains("visual")));
    let Some(d) = bf16_dir() else { return };
    let ck = Checkpoint::open(&d).unwrap();
    assert!(ck.names().all(|n| !n.contains("visual")));
    assert_eq!(ck.names().count(), 760 - 333);
}

#[test]
fn load_bf16_direct() {
    if let Some(d) = bf16_dir() {
        load_and_check(&d, CacheMode::Direct);
    }
}

#[test]
fn load_nvfp4_direct() {
    if let Some(d) = nvfp4_dir() {
        load_and_check(&d, CacheMode::Direct);
    }
}

#[test]
fn load_nvfp4_drop_behind() {
    if let Some(d) = nvfp4_dir() {
        load_and_check(&d, CacheMode::DropBehind);
    }
}

/// Baseline for comparison: plain buffered reads leave the whole checkpoint in the page cache (the doubled
/// copy PLATFORM.md P2 warns about). Dropped again at the end.
#[test]
fn load_nvfp4_keep_baseline() {
    let Some(d) = nvfp4_dir() else { return };
    let cfg = ModelConfig::load(&d).unwrap();
    let ck = Checkpoint::open(&d).unwrap();
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    for f in &ck.files {
        drop_file_cache(f).unwrap();
    }
    let res0 = files_resident(&ck);
    let opts = LoadOptions { stream: StreamOptions { mode: CacheMode::Keep, ..Default::default() }, ..Default::default() };
    let (_w, rep) = ModelWeights::load(&cfg, &ck, &stream, &opts).unwrap();
    let res1 = files_resident(&ck);
    for f in &ck.files {
        drop_file_cache(f).unwrap();
    }
    println!(
        "Keep (buffered, no drop): {:.2} GiB in {:.2} s = {:.2} GB/s; checkpoint page cache {:.3} -> {:.3} GiB",
        gib(rep.stream.bytes_copied), rep.stream.seconds, rep.stream.bytes_copied as f64 / rep.stream.seconds / 1e9,
        gib(res0), gib(res1)
    );
    assert!(res1 > res0 + (4u64 << 30), "buffered reads were expected to populate the page cache");
}
