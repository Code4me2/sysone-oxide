//! The GPU [`DecisionEngine`] (DESIGN §6, §7; Phase 2a, I1): weights, the packed 32-layer forward
//! ([`forward::build`]), T-bucket arenas and CUDA graphs, batch metadata, slot readout.
//!
//! One step:
//! 1. pick the smallest bucket that fits the batch's tokens, answers and segments;
//! 2. pack the batch metadata into a pinned host slot and upload it; rewrite the kernels' segment tables and
//!    attention work list for the bucket ([`TileKernels::update`]);
//! 3. replay the bucket's CUDA graph, or run the forward eagerly (first use of a bucket, or graphs disabled);
//!    after the first eager run of a bucket its graph is captured, so later steps replay it;
//! 4. copy the `[Q, n_slots]` f32 slot logits into a persistent, pre-touched pinned host buffer (PLATFORM P3)
//!    and return them.
//!
//! A batch that exceeds the largest bucket's answers or segments, or the GDN scratch's prefix capacity, is
//! split by request into several steps (results are reassembled in the batch's answer order); a single request
//! that does not fit is an error.

use crate::contract::{DecisionBatch, DecisionEngine, MAX_SLOTS, SlotLogits};
use crate::engine::config::ModelConfig;
use crate::engine::forward::{self, Forward, Taps};
use crate::engine::gemm::Blas;
use crate::engine::graph::tensor::GraphDType;
use crate::engine::graph::{Bucket, BucketGraphs, BucketPlan, Executor, Profiler, ValueId};
use crate::engine::kernels::TileKernels;
use crate::engine::loader::{Checkpoint, StreamOptions};
use crate::engine::meta::{MetaShape, MetaUploader};
use crate::engine::weights::{LoadOptions, LoadReport, ModelWeights};
use crate::kernels::gdn::GdnBackend;
use anyhow::{Context as _, Result, bail, ensure};
use cuda_core::{Context, PinnedHostBuffer, Stream};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Engine sizing and behaviour.
#[derive(Clone, Debug)]
pub struct EngineOptions {
    /// T buckets (packed tokens per step). The largest is the engine's `max_tokens_per_step`.
    pub buckets: Vec<usize>,
    /// Most answer rows (branches) one step holds.
    pub max_answers: usize,
    /// Most segments (prefixes + suffixes) one step holds.
    pub max_segments: usize,
    /// Most prefix segments (requests) one step holds: GDN state slots, 2 MiB each.
    pub max_prefixes: usize,
    /// Capture a CUDA graph per bucket and replay it (else every step runs eagerly).
    pub cuda_graphs: bool,
    /// Run one small batch at load so kernel JIT happens before the first request.
    pub warmup: bool,
    /// Debug taps (per-layer hidden states).
    pub taps: Taps,
    /// Implementation of the GDN delta rule. `None`: [`GdnBackend::default_for_arch`] of the device. Either way
    /// the choice and its reason are logged at load and reported by [`DecisionEngine::info`]; SIMT without a
    /// kernel artifact for the device's architecture fails the load.
    pub gdn_backend: Option<GdnBackend>,
}

/// Environment variable the engine tests and benchmarks read the GDN backend from (`tile`, `simt` or `auto`).
pub const GDN_BACKEND_ENV: &str = "SYSONE_GDN_BACKEND";

/// Parse a GDN backend choice: `tile`, `simt`, or `auto` (`None`: the device's default).
pub fn parse_gdn_backend(s: &str) -> Result<Option<GdnBackend>> {
    match s {
        "auto" => Ok(None),
        other => Ok(Some(other.parse()?)),
    }
}

/// The GDN backend choice in `$SYSONE_GDN_BACKEND` (unset: `auto`).
pub fn gdn_backend_from_env() -> Result<Option<GdnBackend>> {
    match std::env::var(GDN_BACKEND_ENV) {
        Ok(v) => parse_gdn_backend(&v).with_context(|| format!("{GDN_BACKEND_ENV}={v:?}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(e).context(GDN_BACKEND_ENV),
    }
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            buckets: vec![256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
            max_answers: 2048,
            max_segments: 2048,
            max_prefixes: 256,
            cuda_graphs: true,
            warmup: true,
            taps: Taps::default(),
            gdn_backend: None,
        }
    }
}

impl EngineOptions {
    /// Buckets up to the smallest power of two ≥ `max_tokens` (at least 256), doubling from 256.
    pub fn buckets_up_to(max_tokens: usize) -> Vec<usize> {
        let mut b = vec![256usize];
        while *b.last().unwrap() < max_tokens {
            b.push(b.last().unwrap() * 2);
        }
        b
    }
}

/// What the last step did (tests, metrics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepInfo {
    pub bucket_tokens: usize,
    /// Graph replay (else eager).
    pub replayed: bool,
    /// Steps the batch was split into.
    pub steps: usize,
}

pub struct GpuEngine {
    pub cfg: ModelConfig,
    pub opts: EngineOptions,
    pub load_report: LoadReport,
    pub fwd: Forward,
    // Field order is drop order: graphs and buffers go before the context they live in.
    graphs: BucketGraphs,
    kernels: TileKernels,
    meta: MetaUploader,
    host_logits: PinnedHostBuffer<f32>,
    blas: Blas,
    weights: ModelWeights,
    stream: Arc<Stream>,
    ctx: Arc<Context>,
    pub last: StepInfo,
    /// Device architecture (`sm_121`).
    pub arch: String,
}

// SAFETY: every member is either plain data or a device/pinned allocation owned by this engine; the scheduler
// drives the engine from one worker thread at a time (`DecisionEngine: Send`, not `Sync`).
unsafe impl Send for GpuEngine {}

impl GpuEngine {
    /// Load `model_dir` (bf16 checkpoint) onto device 0 and build the forward. `slot_ids` are the slot label
    /// token ids from the frontend (the readout rows).
    pub fn load(model_dir: &Path, slot_ids: &[u32], opts: EngineOptions) -> Result<Self> {
        let t0 = Instant::now();
        let cfg = ModelConfig::load(model_dir)?;
        forward::check_dims(&cfg)?;
        ensure!(!slot_ids.is_empty() && slot_ids.len() <= MAX_SLOTS, "{} slot ids", slot_ids.len());
        ensure!(!opts.buckets.is_empty(), "no buckets");
        let ctx = Context::new(0)?;
        let stream = ctx.new_stream()?;
        let ckpt = Checkpoint::open(model_dir)?;
        let lopts = LoadOptions { slot_token_ids: Some(slot_ids.to_vec()), full_lm_head: false, stream: StreamOptions::default() };
        let (weights, load_report) = ModelWeights::load(&cfg, &ckpt, &stream, &lopts)?;
        let fwd = forward::build(&cfg, &weights, slot_ids.len(), &opts.taps)?;
        let plan = BucketPlan::new(&opts.buckets, opts.max_answers, opts.max_segments)?;
        let largest = plan.largest();
        let graphs = BucketGraphs::new(&fwd.graph, plan, &stream)?;
        let (major, minor) = ctx.compute_capability()?;
        let arch = format!("sm_{major}{minor}");
        let (gdn_backend, why) = match opts.gdn_backend {
            Some(b) => (b, "requested"),
            None => (GdnBackend::default_for_arch(major, minor), "default for this architecture"),
        };
        eprintln!("sysone engine: GDN backend {} ({why}) on {arch}", gdn_backend.name());
        let kernels = TileKernels::new(&stream, cfg.text_config.rms_norm_eps as f32, largest.tokens, opts.max_prefixes,
                                       gdn_backend)
            .with_context(|| format!("GDN backend {} on {arch}", gdn_backend.name()))?;
        let meta = MetaUploader::new(&stream, MetaShape {
            tokens: largest.tokens,
            segments: largest.segments,
            answers: largest.answers,
        })?;
        // Pre-touched (zeroed) pinned readback buffer for the largest step's logits.
        let host_logits = PinnedHostBuffer::zeroed(&ctx, largest.answers * slot_ids.len())?;
        let blas = Blas::new(&stream)?;
        let mut e = Self {
            cfg,
            opts,
            load_report,
            fwd,
            graphs,
            kernels,
            meta,
            host_logits,
            blas,
            weights,
            stream,
            ctx,
            last: StepInfo::default(),
            arch,
        };
        eprintln!(
            "sysone engine: loaded {:.1} GiB of weights in {:.1} s; arena {:.2} GiB (largest bucket {} tokens), GDN scratch \
             {:.2} GiB ({} prefix states)",
            e.load_report.device_bytes as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64(),
            e.graphs.arena.bytes() as f64 / (1u64 << 30) as f64,
            largest.tokens,
            e.kernels.scratch_bytes() as f64 / (1u64 << 30) as f64,
            e.opts.max_prefixes
        );
        if e.opts.warmup {
            let t = Instant::now();
            let mut b = DecisionBatch::default();
            let ids: Vec<u32> = (0..100).map(|i| 1000 + i).collect();
            b.push_request(&ids[..60], &[(&ids[60..], 2)])?;
            e.run(&b)?;
            eprintln!("sysone engine: warm-up step (kernel JIT) {:.1} s", t.elapsed().as_secs_f64());
        }
        Ok(e)
    }

    /// The GDN delta-rule backend this engine runs.
    pub fn gdn_backend(&self) -> GdnBackend {
        self.kernels.gdn_backend()
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    pub fn bucket_plan(&self) -> &BucketPlan {
        &self.graphs.plan
    }

    /// Bytes of the activation arena.
    pub fn arena_bytes(&self) -> usize {
        self.graphs.arena.bytes()
    }

    /// The bucket a batch runs in: the smallest that fits, or the smallest with at least `min_tokens` tokens.
    pub fn pick_bucket(&self, batch: &DecisionBatch, min_tokens: usize) -> Option<Bucket> {
        let (t, q, s) = (batch.num_tokens().max(min_tokens), batch.num_branches(), batch.segments.len());
        let b = self.graphs.plan.pick(t, q, s)?;
        // `pick` accounts for the pad segment only when the bucket is larger than `t`.
        let pad = (b.tokens > batch.num_tokens()) as usize;
        (s + pad <= b.segments).then_some(b)
    }

    fn executor<'a>(&'a self, b: Bucket, profiler: Option<&'a Profiler>) -> Result<Executor<'a>> {
        Ok(Executor {
            graph: &self.fwd.graph,
            plan: self.graphs.pool(b)?,
            arena: &self.graphs.arena,
            weights: &self.weights,
            kernels: &self.kernels,
            blas: &self.blas,
            meta: &self.meta.dev,
            cfg: Some(&self.cfg),
            profiler,
        })
    }

    /// Upload `batch` for bucket `b`: metadata and the kernels' tables.
    fn stage(&mut self, batch: &DecisionBatch, b: Bucket) -> Result<()> {
        self.ctx.bind_to_thread()?;
        self.kernels.prepare(b, &self.stream)?;
        let slot = self.meta.pack(batch, b.tokens)?;
        self.meta.upload(slot, &self.stream)?;
        self.kernels.update(b, batch, &self.stream)
    }

    /// One step in bucket `b` (it must fit). `eager` forces an eager run even when a graph exists.
    pub fn step(&mut self, batch: &DecisionBatch, b: Bucket, eager: bool) -> Result<SlotLogits> {
        let q = batch.num_branches();
        ensure!(q <= b.answers, "{q} answers over the bucket's {}", b.answers);
        self.stage(batch, b)?;
        let replay = self.opts.cuda_graphs && !eager && self.graphs.is_captured(b);
        if replay {
            self.graphs.launch(b, &self.stream)?;
        } else {
            self.executor(b, None)?.run_eager(&self.stream)?;
            if self.opts.cuda_graphs && !self.graphs.is_captured(b) {
                let g = self.executor(b, None)?.capture(&self.stream).with_context(|| format!("capturing bucket {b:?}"))?;
                self.graphs.insert(b, g)?;
            }
        }
        let logits = self.read_logits(b, q)?;
        self.last = StepInfo { bucket_tokens: b.tokens, replayed: replay, steps: 1 };
        Ok(logits)
    }

    /// Copy the first `q` logit rows into the pinned buffer and widen them to `MAX_SLOTS` columns.
    fn read_logits(&mut self, b: Bucket, q: usize) -> Result<SlotLogits> {
        let n = self.fwd.n_slots;
        let src = self.executor(b, None)?.value(self.fwd.logits)?;
        let bytes = q * n * 4;
        if bytes > 0 {
            unsafe {
                cuda_core::memory::memcpy_dtoh_async(self.host_logits.as_mut_ptr(), src.ptr(), bytes,
                                                     self.stream.cu_stream())?
            };
        }
        self.stream.synchronize()?;
        let host = &self.host_logits.as_slice()[..q * n];
        let mut data = vec![0f32; q * MAX_SLOTS];
        for (r, row) in host.chunks_exact(n.max(1)).enumerate().take(q) {
            data[r * MAX_SLOTS..r * MAX_SLOTS + n].copy_from_slice(row);
        }
        Ok(SlotLogits { rows: q, data })
    }

    /// Host copy of a value of the last step (first `rows` rows), e.g. a tap.
    pub fn read_value(&self, v: ValueId, rows: usize) -> Result<Vec<u8>> {
        let b = self.graphs.plan.buckets.iter().copied().find(|b| b.tokens == self.last.bucket_tokens)
            .context("no step has run")?;
        self.executor(b, None)?.value(v)?.rows_view(0, rows)?.to_host_bytes(&self.stream)
    }

    /// A tap of the last step as f32, `[rows, hidden]`.
    pub fn read_tap(&self, name: &str, rows: usize) -> Result<Vec<f32>> {
        let v = self.fwd.taps.iter().find(|(n, _)| n == name).with_context(|| format!("no tap {name}"))?.1;
        let spec = self.fwd.graph.spec(v);
        ensure!(spec.dtype == GraphDType::BF16, "tap {name} is {:?}", spec.dtype);
        Ok(crate::engine::graph::tensor::bf16_from_bytes(&self.read_value(v, rows)?))
    }

    /// Run `batch` eagerly in bucket `b` with a CUDA-event mark per op; returns `(label, ms, launches)` per op
    /// class and the forward's total ms.
    pub fn profile(&mut self, batch: &DecisionBatch, b: Bucket) -> Result<(Vec<(String, f64, usize)>, f64)> {
        self.stage(batch, b)?;
        let prof = Profiler::new(&self.ctx);
        self.executor(b, Some(&prof))?.run_eager(&self.stream)?;
        self.stream.synchronize()?;
        prof.summary()
    }

    /// Batches that fit one step, as `(sub-batch, original answer index of each of its rows)`.
    fn split(&self, batch: &DecisionBatch) -> Result<Vec<(DecisionBatch, Vec<usize>)>> {
        let fits = |b: &DecisionBatch| {
            let prefixes = b.segments.iter().filter(|s| s.is_prefix()).count();
            prefixes <= self.kernels.max_prefixes() && self.pick_bucket(b, 0).is_some()
        };
        split_batch(batch, fits).with_context(|| {
            format!("largest bucket {:?}, {} prefix states per step", self.graphs.plan.largest(), self.kernels.max_prefixes())
        })
    }
}

/// Split `batch` by request into batches that each satisfy `fits`, packing requests greedily in prefix order.
/// Returns each sub-batch with the original answer index of each of its answer rows. A batch that fits is
/// returned whole; a single request that does not fit is an error.
pub fn split_batch(batch: &DecisionBatch, fits: impl Fn(&DecisionBatch) -> bool) -> Result<Vec<(DecisionBatch, Vec<usize>)>> {
    if fits(batch) {
        return Ok(vec![(batch.clone(), (0..batch.num_branches()).collect())]);
    }
    // Answer index of every suffix segment (answers are in suffix-segment order).
    let mut answer_of = vec![usize::MAX; batch.segments.len()];
    let mut a = 0;
    for (i, s) in batch.segments.iter().enumerate() {
        if !s.is_prefix() {
            answer_of[i] = a;
            a += 1;
        }
    }
    let ids = |s: &crate::contract::Segment| &batch.token_ids[s.start as usize..s.end() as usize];
    let mut out: Vec<(DecisionBatch, Vec<usize>)> = Vec::new();
    let mut cur = (DecisionBatch::default(), Vec::new());
    for (pi, p) in batch.segments.iter().enumerate().filter(|(_, s)| s.is_prefix()) {
        let sufs: Vec<usize> = (0..batch.segments.len()).filter(|&i| batch.segments[i].parent == pi as i32).collect();
        let slots: Vec<(&[u32], u16)> = sufs.iter().map(|&i| (ids(&batch.segments[i]), batch.answer_slots[answer_of[i]])).collect();
        let mut trial = cur.0.clone();
        trial.push_request(ids(p), &slots)?;
        if !fits(&trial) {
            let tokens = p.len + sufs.iter().map(|&i| batch.segments[i].len).sum::<u32>();
            ensure!(!cur.0.segments.is_empty(), "the request at segment {pi} ({tokens} tokens, {} branches) does not fit \
                    one step", sufs.len());
            out.push(std::mem::take(&mut cur));
            trial = DecisionBatch::default();
            trial.push_request(ids(p), &slots)?;
            ensure!(fits(&trial), "the request at segment {pi} ({tokens} tokens, {} branches) does not fit one step",
                    sufs.len());
        }
        cur.0 = trial;
        cur.1.extend(sufs.iter().map(|&i| answer_of[i]));
    }
    if !cur.0.segments.is_empty() {
        out.push(cur);
    }
    Ok(out)
}

impl DecisionEngine for GpuEngine {
    fn max_tokens_per_step(&self) -> usize {
        self.graphs.plan.largest().tokens
    }

    fn info(&self) -> Vec<(&'static str, String)> {
        vec![("engine", "bf16".into()), ("gdn_backend", self.gdn_backend().name().into()), ("arch", self.arch.clone())]
    }

    fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
        let max = self.max_tokens_per_step();
        ensure!(batch.num_tokens() <= max, "batch of {} tokens over the engine's {max} per step", batch.num_tokens());
        if batch.num_branches() == 0 {
            return Ok(SlotLogits { rows: 0, data: vec![] });
        }
        let parts = self.split(batch)?;
        let n = parts.len();
        if n == 1 && parts[0].1.iter().enumerate().all(|(i, &a)| i == a) {
            let b = self.pick_bucket(batch, 0).context("no bucket fits")?;
            return self.step(batch, b, false);
        }
        let mut data = vec![0f32; batch.num_branches() * MAX_SLOTS];
        for (sub, rows) in parts {
            let Some(b) = self.pick_bucket(&sub, 0) else { bail!("no bucket fits a split batch") };
            let l = self.step(&sub, b, false)?;
            for (r, &a) in rows.iter().enumerate() {
                data[a * MAX_SLOTS..(a + 1) * MAX_SLOTS].copy_from_slice(&l.data[r * MAX_SLOTS..(r + 1) * MAX_SLOTS]);
            }
        }
        self.last.steps = n;
        Ok(SlotLogits { rows: batch.num_branches(), data })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_backend_choice_parses() {
        assert_eq!(parse_gdn_backend("auto").unwrap(), None);
        assert_eq!(parse_gdn_backend("tile").unwrap(), Some(GdnBackend::Tile));
        assert_eq!(parse_gdn_backend("simt").unwrap(), Some(GdnBackend::Simt));
        assert!(parse_gdn_backend("SIMT").is_err());
    }

    #[test]
    fn split_keeps_answer_order_and_refuses_oversize_requests() {
        let mut b = DecisionBatch::default();
        b.push_request(&[1; 10], &[(&[2; 3], 2), (&[3; 4], 3)]).unwrap();
        b.push_request(&[4; 6], &[(&[5; 2], 4)]).unwrap();
        b.push_request(&[6; 5], &[(&[7; 1], 5), (&[8; 2], 6), (&[9; 3], 7)]).unwrap();
        // At most 2 prefixes per step.
        let fits = |x: &DecisionBatch| x.segments.iter().filter(|s| s.is_prefix()).count() <= 2;
        let parts = split_batch(&b, fits).unwrap();
        assert_eq!(parts.len(), 2);
        let rows: Vec<usize> = parts.iter().flat_map(|p| p.1.clone()).collect();
        assert_eq!(rows, [0, 1, 2, 3, 4, 5]);
        for (sub, idx) in &parts {
            sub.validate().unwrap();
            for (r, &a) in idx.iter().enumerate() {
                assert_eq!(sub.answer_slots[r], b.answer_slots[a]);
            }
        }
        assert_eq!(parts[1].0.token_ids, [6, 6, 6, 6, 6, 7, 8, 8, 9, 9, 9]);
        // A batch that fits is not split; a request over the limit is an error.
        assert_eq!(split_batch(&b, |_| true).unwrap().len(), 1);
        assert!(split_batch(&b, |x| x.num_tokens() <= 12).is_err());
    }
}
