//! The GPU [`DecisionEngine`] (DESIGN §6, §7; Phase 2a, I1): weights, the packed 32-layer forward
//! ([`forward::build`]), T-bucket arenas and CUDA graphs, batch metadata, slot readout.
//!
//! One step:
//! 1. pick the smallest bucket that fits the batch's tokens, answers and segments;
//! 2. pack the batch metadata into a pinned host slot and upload it; rewrite the kernels' segment tables and
//!    attention work list for the bucket ([`TileKernels::update`]);
//! 3. in legacy mode, replay the bucket's CUDA graph or run bucket-sized eager work; in exact eager mode, run
//!    actual logical T/Q rows over the bucket's capacity-backed arena and never capture/replay;
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
use crate::engine::graph::{Bucket, BucketGraphs, BucketPlan, Executor, GemmRequest, Profiler, StepShape, ValueId};
use crate::engine::kernels::TileKernels;
use crate::engine::loader::{Checkpoint, StreamOptions};
use crate::engine::meta::{MetaShape, MetaUploader};
use crate::engine::weights::{LoadOptions, LoadReport, ModelWeights, debug_permutation};
use crate::engine::warmup::{self, WarmupReport};
use crate::kernels::gdn::GdnBackend;
use anyhow::{Context as _, Result, bail, ensure};
use cuda_core::{Context, PinnedHostBuffer, Stream};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Dense row execution policy. The default preserves the existing bucket-sized graph/eager behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    #[default]
    Bucketed,
    ExactEager,
}

impl ExecutionMode {
    pub fn name(self, cuda_graphs: bool) -> &'static str {
        match (self, cuda_graphs) {
            (Self::Bucketed, true) => "bucket_graph",
            (Self::Bucketed, false) => "bucket_eager",
            (Self::ExactEager, _) => "exact_eager",
        }
    }
}

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
    /// Capture a CUDA graph per bucket and replay it (else every step runs eagerly). Exact eager requires false.
    pub cuda_graphs: bool,
    /// Bucket-sized legacy execution or exact logical token/answer rows backed by a capacity bucket.
    pub execution_mode: ExecutionMode,
    /// Run one small batch at load so kernel JIT happens before the first request.
    pub warmup: bool,
    /// Opt-in fixed 51-call exact/explicit-Tile specialization warmup; requires warmup and no graphs.
    pub bounded_exact_warmup: bool,
    /// Debug taps (per-layer hidden states).
    pub taps: Taps,
    /// Debug (numerics diagnostics only): run the model in a hidden basis permuted with this seed
    /// ([`ModelWeights::permute_hidden`]); [`GpuEngine::read_tap`] returns taps in the original basis.
    pub debug_permute_hidden: Option<u64>,
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
            execution_mode: ExecutionMode::Bucketed,
            warmup: true,
            bounded_exact_warmup: false,
            taps: Taps::default(),
            debug_permute_hidden: None,
            gdn_backend: None,
        }
    }
}

impl EngineOptions {
    /// Validate known execution choices on CPU; an automatic backend is checked after arch resolution.
    fn validate_execution(&self, backend: Option<GdnBackend>) -> Result<()> {
        if self.execution_mode == ExecutionMode::ExactEager {
            ensure!(!self.cuda_graphs, "exact eager mode must not enable CUDA graphs");
            if let Some(backend) = backend {
                ensure!(backend == GdnBackend::Tile, "exact eager mode supports only the Tile GDN backend");
            }
        }
        Ok(())
    }

    fn resolve_gdn_backend(&self, major: i32, minor: i32) -> Result<(GdnBackend, &'static str)> {
        let (backend, why) = match self.gdn_backend {
            Some(b) => (b, "requested"),
            None => (GdnBackend::default_for_arch(major, minor), "default for this architecture"),
        };
        self.validate_execution(Some(backend))?;
        Ok((backend, why))
    }

    /// CPU-only validation of the opt-in contract, before any CUDA context or model allocation.
    /// Legacy configurations retain their existing validation path when the flag is absent.
    pub fn validate_bounded_exact_warmup(&self) -> Result<()> {
        if !self.bounded_exact_warmup { return Ok(()); }
        ensure!(self.warmup, "bounded exact warmup requires warmup=true");
        ensure!(self.execution_mode == ExecutionMode::ExactEager, "bounded exact warmup requires exact eager execution");
        ensure!(!self.cuda_graphs, "bounded exact warmup must not enable CUDA graphs");
        ensure!(self.gdn_backend == Some(GdnBackend::Tile), "bounded exact warmup requires explicit Tile GDN backend");
        ensure!(self.debug_permute_hidden.is_none() && self.taps.layers.is_empty() && !self.taps.norm
            && !self.taps.sublayers, "bounded exact warmup does not support debug permutations/taps");
        ensure!(self.max_prefixes >= 1 && self.max_answers >= 16 && self.max_segments >= 17,
            "bounded exact warmup requires capacity Q16/S17/P1; coverage must not shrink");
        let plan = BucketPlan::new(&self.buckets, self.max_answers, self.max_segments)?;
        for layout in warmup::plan() {
            ensure!(plan.pick(layout.tokens, layout.answers, layout.segments).is_some(),
                "bounded exact warmup capacity cannot fit {} T{} Q{} S{} P1 (including padding)",
                layout.phase, layout.tokens, layout.answers, layout.segments);
        }
        Ok(())
    }

    pub fn warmup_policy(&self) -> &'static str {
        if self.bounded_exact_warmup { warmup::POLICY } else if self.warmup { "standard" } else { "disabled" }
    }

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
    /// Selected allocation/table capacity T/Q/S.
    pub bucket_tokens: usize,
    pub bucket_answers: usize,
    pub bucket_segments: usize,
    /// Actual batch dimensions of the last sub-step (debug/readback bounds).
    pub logical_tokens: usize,
    pub logical_answers: usize,
    pub logical_segments: usize,
    /// Dense rows requested from kernels/GEMMs. Legacy mode uses capacity rows; exact mode equals logical rows.
    pub execution_tokens: usize,
    pub execution_answers: usize,
    pub execution_segments: usize,
    pub execution_mode: ExecutionMode,
    /// Graph replay (else eager). Always false in exact eager mode.
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
    /// The debug hidden-basis permutation, when [`EngineOptions::debug_permute_hidden`] is set.
    hidden_perm: Option<Vec<usize>>,
    /// Device architecture (`sm_121`).
    pub arch: String,
    warmup_report: Option<WarmupReport>,
}

// SAFETY: every member is either plain data or a device/pinned allocation owned by this engine; the scheduler
// drives the engine from one worker thread at a time (`DecisionEngine: Send`, not `Sync`).
unsafe impl Send for GpuEngine {}

impl GpuEngine {
    /// Load `model_dir` (bf16 checkpoint) onto device 0 and build the forward. `slot_ids` are the slot label
    /// token ids from the frontend (the readout rows).
    pub fn load(model_dir: &Path, slot_ids: &[u32], opts: EngineOptions) -> Result<Self> {
        let t0 = Instant::now();
        opts.validate_bounded_exact_warmup()?;
        opts.validate_execution(opts.gdn_backend)?;
        let cfg = ModelConfig::load(model_dir)?;
        if opts.bounded_exact_warmup {
            ensure!(cfg.text_config.vocab_size >= 1100 && slot_ids.len() >= 2,
                "bounded exact warmup requires vocabulary IDs1000..1099 and at least two readout slots");
        }
        forward::check_dims(&cfg)?;
        ensure!(!slot_ids.is_empty() && slot_ids.len() <= MAX_SLOTS, "{} slot ids", slot_ids.len());
        ensure!(!opts.buckets.is_empty(), "no buckets");
        let ctx = Context::new(0)?;
        let (major, minor) = ctx.compute_capability()?;
        let arch = format!("sm_{major}{minor}");
        let (gdn_backend, why) = opts.resolve_gdn_backend(major, minor)?;
        eprintln!("sysone engine: GDN backend {} ({why}) on {arch}; execution mode {}", gdn_backend.name(),
                  opts.execution_mode.name(opts.cuda_graphs));
        let stream = ctx.new_stream()?;
        let ckpt = Checkpoint::open(model_dir)?;
        let lopts = LoadOptions { slot_token_ids: Some(slot_ids.to_vec()), full_lm_head: false, stream: StreamOptions::default() };
        let (weights, load_report) = ModelWeights::load(&cfg, &ckpt, &stream, &lopts)?;
        let hidden_perm = opts.debug_permute_hidden.map(|seed| debug_permutation(cfg.text_config.hidden_size, seed));
        if let Some(p) = &hidden_perm {
            weights.permute_hidden(&stream, p)?;
        }
        let fwd = forward::build(&cfg, &weights, slot_ids.len(), &opts.taps)?;
        let plan = BucketPlan::new(&opts.buckets, opts.max_answers, opts.max_segments)?;
        let largest = plan.largest();
        let graphs = BucketGraphs::new(&fwd.graph, plan, &stream)?;
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
            hidden_perm,
            arch,
            warmup_report: None,
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
        if e.opts.bounded_exact_warmup {
            let t = Instant::now();
            let mut report = WarmupReport::default();
            let result = warmup::execute(&mut report, |batch| {
                e.run(batch)?;
                ensure!(!e.last.replayed && e.last.steps == 1, "warmup must be one uncaptured exact step");
                Ok(())
            }, cutile::tile_kernel::jit_compile_count, || t.elapsed());
            report.startup_ms = t0.elapsed().as_secs_f64() * 1000.;
            eprintln!("SYSONE_WARMUP_REPORT {}", serde_json::to_string(&report)?);
            result?;
            eprintln!("SYSONE_WARMUP_READY {}", serde_json::json!({"policy":warmup::POLICY,
                "calls":report.calls.len(),"jit_at_ready":report.jit_after,"startup_ms":report.startup_ms}));
            e.warmup_report = Some(report);
        } else if e.opts.warmup {
            let t = Instant::now();
            let mut b = DecisionBatch::default();
            let ids: Vec<u32> = (0..100).map(|i| 1000 + i).collect();
            b.push_request(&ids[..60], &[(&ids[60..], 2)])?;
            e.run(&b)?;
            eprintln!("sysone engine: warm-up step (kernel JIT) {:.1} s", t.elapsed().as_secs_f64());
        }
        Ok(e)
    }

    /// Bounded startup report, completed before load returns. None preserves the unflagged legacy path.
    pub fn warmup_report(&self) -> Option<&WarmupReport> { self.warmup_report.as_ref() }

    /// Captured allocation buckets (diagnostic proof; exact warmup must always report zero).
    pub fn captured_bucket_count(&self) -> usize {
        self.graphs.plan.buckets.iter().filter(|&&b| self.graphs.is_captured(b)).count()
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

    fn execution_shape(&self, batch: &DecisionBatch, b: Bucket) -> StepShape {
        match self.opts.execution_mode {
            ExecutionMode::Bucketed => b.shape(),
            ExecutionMode::ExactEager => StepShape {
                tokens: batch.num_tokens(),
                answers: batch.num_branches(),
                segments: batch.segments.len(),
            },
        }
    }

    fn executor<'a>(&'a self, b: Bucket, shape: StepShape, profiler: Option<&'a Profiler>) -> Result<Executor<'a>> {
        Ok(Executor {
            graph: &self.fwd.graph,
            plan: self.graphs.pool(b)?,
            shape,
            arena: &self.graphs.arena,
            weights: &self.weights,
            kernels: &self.kernels,
            blas: &self.blas,
            meta: &self.meta.dev,
            cfg: Some(&self.cfg),
            profiler,
        })
    }

    /// Upload `batch` for capacity bucket `b`: metadata and fixed-capacity kernel tables. In exact mode the
    /// metadata header and GDN logical row count use the batch's actual T; legacy mode retains bucket padding.
    fn stage(&mut self, batch: &DecisionBatch, b: Bucket, shape: StepShape) -> Result<()> {
        self.ctx.bind_to_thread()?;
        ensure!(shape.fits(b), "logical shape {shape:?} exceeds capacity {b:?}");
        self.kernels.prepare(b, &self.stream)?;
        let slot = self.meta.pack(batch, shape.tokens)?;
        self.meta.upload(slot, &self.stream)?;
        self.kernels.update(b, batch, shape.tokens, &self.stream)
    }

    /// One step in bucket `b` (it must fit). `eager` forces an eager run even when a graph exists.
    pub fn step(&mut self, batch: &DecisionBatch, b: Bucket, eager: bool) -> Result<SlotLogits> {
        let q = batch.num_branches();
        ensure!(q <= b.answers, "{q} answers over the bucket's {}", b.answers);
        let shape = self.execution_shape(batch, b);
        self.stage(batch, b, shape)?;
        let graphs_enabled = self.opts.execution_mode == ExecutionMode::Bucketed && self.opts.cuda_graphs;
        let replay = graphs_enabled && !eager && self.graphs.is_captured(b);
        if replay {
            self.graphs.launch(b, &self.stream)?;
        } else {
            self.executor(b, shape, None)?.run_eager(&self.stream)?;
            if graphs_enabled && !self.graphs.is_captured(b) {
                let g = self.executor(b, shape, None)?.capture(&self.stream)
                    .with_context(|| format!("capturing bucket {b:?}"))?;
                self.graphs.insert(b, g)?;
            }
        }
        let logits = self.read_logits(b, shape, q)?;
        self.last = StepInfo {
            bucket_tokens: b.tokens,
            bucket_answers: b.answers,
            bucket_segments: b.segments,
            logical_tokens: batch.num_tokens(),
            logical_answers: batch.num_branches(),
            logical_segments: batch.segments.len(),
            execution_tokens: shape.tokens,
            execution_answers: shape.answers,
            execution_segments: shape.segments,
            execution_mode: self.opts.execution_mode,
            replayed: replay,
            steps: 1,
        };
        Ok(logits)
    }

    /// Copy the first `q` logit rows into the pinned buffer and widen them to `MAX_SLOTS` columns.
    fn read_logits(&mut self, b: Bucket, shape: StepShape, q: usize) -> Result<SlotLogits> {
        let n = self.fwd.n_slots;
        let src = self.executor(b, shape, None)?.value(self.fwd.logits)?;
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
        let logical = StepShape {
            tokens: self.last.logical_tokens,
            answers: self.last.logical_answers,
            segments: self.last.logical_segments,
        };
        let execution = StepShape {
            tokens: self.last.execution_tokens,
            answers: self.last.execution_answers,
            segments: self.last.execution_segments,
        };
        let logical_rows = self.fwd.graph.spec(v).concrete(&logical).first().copied().unwrap_or(1);
        ensure!(rows <= logical_rows, "read of {rows} rows from value {} exceeds last step's logical bound {logical_rows}", v.0);
        self.executor(b, execution, None)?.value(v)?.rows_view(0, rows)?.to_host_bytes(&self.stream)
    }

    /// Requested GEMM row dimensions of the last sub-step. This is the host launch shape trace, not measured
    /// hardware FLOPs.
    pub fn last_gemm_requests(&self) -> Result<Vec<GemmRequest>> {
        let b = self.graphs.plan.buckets.iter().copied().find(|b| b.tokens == self.last.bucket_tokens)
            .context("no step has run")?;
        let execution = StepShape {
            tokens: self.last.execution_tokens,
            answers: self.last.execution_answers,
            segments: self.last.execution_segments,
        };
        Ok(self.executor(b, execution, None)?.gemm_requests())
    }

    /// A tap of the last step as f32, `[rows, hidden]`.
    pub fn read_tap(&self, name: &str, rows: usize) -> Result<Vec<f32>> {
        let v = self.fwd.taps.iter().find(|(n, _)| n == name).with_context(|| format!("no tap {name}"))?.1;
        let spec = self.fwd.graph.spec(v);
        ensure!(spec.dtype == GraphDType::BF16, "tap {name} is {:?}", spec.dtype);
        let vals = crate::engine::graph::tensor::bf16_from_bytes(&self.read_value(v, rows)?);
        let Some(perm) = &self.hidden_perm else { return Ok(vals) };
        let h = perm.len();
        ensure!(vals.len() % h == 0, "tap {name}: {} values for hidden {h}", vals.len());
        let mut out = vec![0f32; vals.len()];
        for (o, x) in out.chunks_exact_mut(h).zip(vals.chunks_exact(h)) {
            for (j, &p) in perm.iter().enumerate() {
                o[p] = x[j];
            }
        }
        Ok(out)
    }

    /// Run `batch` eagerly in bucket `b` with a CUDA-event mark per op; returns `(label, ms, launches)` per op
    /// class and the forward's total ms.
    pub fn profile(&mut self, batch: &DecisionBatch, b: Bucket) -> Result<(Vec<(String, f64, usize)>, f64)> {
        let shape = self.execution_shape(batch, b);
        self.stage(batch, b, shape)?;
        let prof = Profiler::new(&self.ctx);
        self.executor(b, shape, Some(&prof))?.run_eager(&self.stream)?;
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
        let graph_policy = match (self.opts.execution_mode, self.opts.cuda_graphs) {
            (ExecutionMode::ExactEager, _) => "never",
            (ExecutionMode::Bucketed, true) => "enabled",
            (ExecutionMode::Bucketed, false) => "disabled",
        };
        vec![
            ("engine", "bf16".into()),
            ("execution_mode", self.opts.execution_mode.name(self.opts.cuda_graphs).into()),
            ("cuda_graphs", graph_policy.into()),
            ("warmup_policy", self.opts.warmup_policy().into()),
            ("gdn_backend", self.gdn_backend().name().into()),
            ("arch", self.arch.clone()),
        ]
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
        assert_eq!(ExecutionMode::Bucketed.name(true), "bucket_graph");
        assert_eq!(ExecutionMode::Bucketed.name(false), "bucket_eager");
        assert_eq!(ExecutionMode::ExactEager.name(true), "exact_eager");
    }

    #[test]
    fn exact_invalid_options_fail_before_even_opening_model_or_cuda() {
        for backend in [None, Some(GdnBackend::Tile), Some(GdnBackend::Simt)] {
            let opts = EngineOptions { execution_mode: ExecutionMode::ExactEager, cuda_graphs: true,
                gdn_backend: backend, ..Default::default() };
            let err = GpuEngine::load(Path::new("nonexistent-model"), &[1, 2], opts).err().unwrap();
            assert_eq!(err.to_string(), "exact eager mode must not enable CUDA graphs");
        }
        let opts = EngineOptions { execution_mode: ExecutionMode::ExactEager, cuda_graphs: false,
            gdn_backend: Some(GdnBackend::Simt), ..Default::default() };
        let err = GpuEngine::load(Path::new("nonexistent-model"), &[1, 2], opts).err().unwrap();
        assert_eq!(err.to_string(), "exact eager mode supports only the Tile GDN backend");
    }

    #[test]
    fn exact_auto_backend_is_checked_after_arch_resolution() {
        let mut opts = EngineOptions { execution_mode: ExecutionMode::ExactEager, cuda_graphs: false,
            ..Default::default() };
        opts.validate_bounded_exact_warmup().unwrap();
        opts.validate_execution(opts.gdn_backend).unwrap();
        assert_eq!(opts.resolve_gdn_backend(12, 0).unwrap(), (GdnBackend::Tile, "default for this architecture"));
        assert_eq!(opts.resolve_gdn_backend(12, 1).unwrap_err().to_string(),
            "exact eager mode supports only the Tile GDN backend");
        opts.gdn_backend = Some(GdnBackend::Tile);
        assert_eq!(opts.resolve_gdn_backend(12, 1).unwrap(), (GdnBackend::Tile, "requested"));
        opts.bounded_exact_warmup = true;
        opts.gdn_backend = None;
        assert_eq!(opts.validate_bounded_exact_warmup().unwrap_err().to_string(),
            "bounded exact warmup requires explicit Tile GDN backend");
    }

    #[test]
    fn legacy_engine_defaults_remain_unchanged() {
        let opts = EngineOptions::default();
        assert_eq!(opts.buckets, vec![256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]);
        assert_eq!((opts.max_answers, opts.max_segments, opts.max_prefixes), (2048, 2048, 256));
        assert_eq!(opts.execution_mode, ExecutionMode::Bucketed);
        assert!(opts.cuda_graphs && opts.warmup && !opts.bounded_exact_warmup);
        assert_eq!(opts.gdn_backend, None);
        assert!(opts.debug_permute_hidden.is_none() && opts.taps.layers.is_empty()
            && !opts.taps.norm && !opts.taps.sublayers);
        assert_eq!(opts.warmup_policy(), "standard");
        opts.validate_bounded_exact_warmup().unwrap();
        opts.validate_execution(opts.gdn_backend).unwrap();
        assert_eq!(opts.resolve_gdn_backend(12, 0).unwrap().0, GdnBackend::Tile);
        assert_eq!(opts.resolve_gdn_backend(12, 1).unwrap().0, GdnBackend::Simt);
        for backend in GdnBackend::ALL {
            let explicit = EngineOptions { gdn_backend: Some(backend), ..opts.clone() };
            explicit.validate_execution(explicit.gdn_backend).unwrap();
            assert_eq!(explicit.resolve_gdn_backend(12, 0).unwrap(), (backend, "requested"));
        }
    }

    fn bounded_options() -> EngineOptions {
        EngineOptions { bounded_exact_warmup: true, execution_mode: ExecutionMode::ExactEager,
            cuda_graphs: false, gdn_backend: Some(GdnBackend::Tile), ..Default::default() }
    }

    #[test]
    fn bounded_configuration_is_opt_in_and_rejects_contradictions() {
        assert!(!EngineOptions::default().bounded_exact_warmup);
        assert_eq!(EngineOptions::default().warmup_policy(), "standard");
        bounded_options().validate_bounded_exact_warmup().unwrap();
        let mut bad = Vec::new();
        let mut o = bounded_options(); o.warmup = false; bad.push(o);
        let mut o = bounded_options(); o.cuda_graphs = true; bad.push(o);
        let mut o = bounded_options(); o.execution_mode = ExecutionMode::Bucketed; bad.push(o);
        let mut o = bounded_options(); o.gdn_backend = None; bad.push(o);
        let mut o = bounded_options(); o.gdn_backend = Some(GdnBackend::Simt); bad.push(o);
        let mut o = bounded_options(); o.taps.norm = true; bad.push(o);
        let mut o = bounded_options(); o.debug_permute_hidden = Some(1); bad.push(o);
        for o in bad { assert!(o.validate_bounded_exact_warmup().is_err(), "{o:?}"); }
    }

    #[test]
    fn bounded_capacity_must_cover_standard_and_full_lattice_not_shrink() {
        let mut good = bounded_options();
        good.buckets = vec![100]; good.max_answers = 16; good.max_segments = 17; good.max_prefixes = 1;
        good.validate_bounded_exact_warmup().unwrap();
        for dim in 0..4 {
            let mut bad = good.clone();
            match dim { 0 => bad.buckets = vec![80], 1 => bad.max_answers = 15,
                2 => bad.max_segments = 16, _ => bad.max_prefixes = 0 }
            assert!(bad.validate_bounded_exact_warmup().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn bounded_invalid_options_fail_before_even_opening_model_or_cuda() {
        let mut o = bounded_options(); o.warmup = false;
        let err = GpuEngine::load(Path::new("nonexistent-model"), &[1,2], o).err().unwrap();
        assert!(err.to_string().contains("requires warmup=true"));
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
