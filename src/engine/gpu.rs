//! The GPU [`DecisionEngine`] (DESIGN §6, §7; Phase 2a, I1): weights, the packed 32-layer forward
//! ([`forward::build`]), T-bucket arenas and CUDA graphs, batch metadata, slot readout.
//!
//! One step:
//! 1. pick the smallest bucket that fits the batch's tokens, answers and segments;
//! 2. pack the batch metadata into a pinned host slot and upload it; rewrite the kernels' segment tables and
//!    attention work list for the bucket ([`TileKernels::update`]);
//! 3. in bucketed mode, replay the bucket's CUDA graph or run bucket-sized eager work; in exact eager mode, run
//!    actual logical T/Q rows over the bucket's capacity-backed arena and never capture/replay;
//! 4. copy the `[Q, n_slots]` f32 slot logits into a persistent, pre-touched pinned host buffer (PLATFORM P3)
//!    and return them.
//!
//! A batch that exceeds the largest bucket's answers or segments, or the GDN scratch's prefix capacity, is
//! split by request into several steps (results are reassembled in the batch's answer order); a single request
//! that does not fit is an error.

use crate::contract::{DecisionBatch, DecisionEngine, MAX_SLOTS, SlotLogits, StepLimits};
use crate::engine::config::{ModelConfig, WeightFormat};
use crate::engine::forward::{self, Forward, ForwardOptions, Taps};
use crate::engine::gemm::{Blas, Fp4Gemm, GemmMode, fp4_gemm_from_env};
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

/// Dense row execution policy: bucket-sized graph/eager work by default, or opt-in exact eager rows.
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
    /// Bucket-sized execution or exact logical token/answer rows backed by a capacity bucket.
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
    /// The weight format the caller expects (`--engine bf16 | nvfp4`). `None`: whatever the checkpoint's profile
    /// says ([`ModelConfig::profile`]). A mismatch fails the load before any weight is read.
    pub weights: Option<WeightFormat>,
    /// NVFP4 checkpoints: quantize activations in the epilogue of the preceding norm / SiLU·mul (default) instead of
    /// a standalone `quantize_nvfp4` launch. Both are bit-identical; the fused form saves a bf16 round trip.
    pub nvfp4_fused: bool,
    /// NVFP4 checkpoints with `nvfp4_fused`: run SiLU·mul + quantize in the gate/up GEMM's epilogue (F1, default)
    /// instead of a separate `silu_mul_nvfp4` launch over a bf16 `[T, 2I]` gate/up output. Byte-identical; the
    /// gate/up weight rows are interleaved for it at load (`ModelWeights::interleave_gate_up_nvfp4`).
    pub nvfp4_swiglu: bool,
    /// NVFP4 checkpoints: add the MLP down projection's output to the residual stream in the down GEMM's epilogue
    /// (S2, default; `ForwardOptions::nvfp4_residual`) instead of in the next layer's input norm. Byte-identical;
    /// saves writing and reading the bf16 MLP output.
    pub nvfp4_residual: bool,
    /// Size the buckets to the device: keep only the buckets (of `buckets`) whose activation arena, GDN scratch
    /// and readback fit in this fraction of the device memory free after the weights are loaded. None: use
    /// `buckets` as given (an arena that does not fit fails the load).
    pub memory_fraction: Option<f64>,
    /// How the bf16 GEMMs pick their cuBLAS kernel (L1). [`GemmMode::Pinned`] (default): one cuBLASLt algorithm
    /// per weight shape at every T, without split-K, so a row's result never depends on the batch it is in.
    /// [`GemmMode::Heuristic`]: cuBLAS's per-call choice (Phase 2a / N1 behaviour; not batch-invariant).
    pub gemm: GemmMode,
    /// L1: run the last layer's queries, attention, MLP and the final norm only on the answer rows
    /// ([`ForwardOptions::trim_last_layer`]). Turned off automatically when a debug tap needs the last layer's
    /// rows (`taps.norm`, or the last layer in `taps.layers`).
    pub trim_last_layer: bool,
    /// L1: a bucket of T tokens holds at most `ceil(T / tokens_per_answer)` answer rows (and at most
    /// `max_answers`; [`BucketPlan::with_answer_density`]). The answer-row values (the readout and the trimmed
    /// last layer) are sized by it, so a lower capacity is less padding; a batch denser in answers runs in a
    /// larger bucket. Every branch has at least the chat template's closing tokens as its suffix, so real batches
    /// stay far below one answer per 8 tokens. 1: `min(T, max_answers)` (the Phase 2a capacities).
    pub tokens_per_answer: usize,
    /// C1: the kernel of the NVFP4 MLP down projection's GEMM. [`Fp4Gemm::CuTile`] (W3c's `gemm_nvfp4_raw`) or
    /// [`Fp4Gemm::CuBlasLt`] (cuBLASLt's block-scaled FP4 GEMM, one pinned algorithm per weight shape, plus a
    /// re-layout of the activation scales; docs/C1.md; bit-identical on GB10). `None`: `$SYSONE_FP4_GEMM` if set,
    /// else [`Fp4Gemm::default_for_arch`] (cuBLASLt on sm_121, cuTile elsewhere). The gate/up GEMM is cuTile either
    /// way. Ignored by the bf16 engine.
    pub fp4_gemm: Option<Fp4Gemm>,
}

/// Environment variable the engine tests and benchmarks read the expected weight format from (`bf16`, `nvfp4`;
/// unset: the checkpoint's own).
pub const ENGINE_ENV: &str = "SYSONE_ENGINE";

/// Parse a weight-format / engine name: `bf16` or `nvfp4`.
pub fn parse_weight_format(s: &str) -> Result<WeightFormat> {
    match s {
        "bf16" => Ok(WeightFormat::Bf16),
        "nvfp4" => Ok(WeightFormat::Nvfp4),
        other => bail!("unknown engine {other:?} (bf16 or nvfp4)"),
    }
}

/// The weight format in `$SYSONE_ENGINE` (unset or `auto`: `None`, the checkpoint decides).
pub fn weight_format_from_env() -> Result<Option<WeightFormat>> {
    match std::env::var(ENGINE_ENV) {
        Ok(v) if v == "auto" => Ok(None),
        Ok(v) => parse_weight_format(&v).map(Some).with_context(|| format!("{ENGINE_ENV}={v:?}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(e).context(ENGINE_ENV),
    }
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
            weights: None,
            nvfp4_fused: true,
            nvfp4_swiglu: true,
            nvfp4_residual: true,
            memory_fraction: None,
            gemm: GemmMode::Pinned,
            trim_last_layer: true,
            tokens_per_answer: 8,
            fp4_gemm: None,
        }
    }
}

impl EngineOptions {
    /// The bucket capacities for `tokens` under these options.
    pub fn bucket_plan(&self, tokens: &[usize]) -> Result<BucketPlan> {
        BucketPlan::with_answer_density(tokens, self.max_answers, self.max_segments, self.tokens_per_answer)
    }

    /// Validate known execution choices on CPU; an automatic backend is checked after arch resolution.
    fn validate_execution(&self, backend: Option<GdnBackend>) -> Result<()> {
        if self.execution_mode == ExecutionMode::ExactEager {
            ensure!(!self.cuda_graphs, "exact eager mode must not enable CUDA graphs");
            ensure!(self.weights != Some(WeightFormat::Nvfp4), "exact eager mode supports only bf16 weights");
            if let Some(backend) = backend {
                ensure!(backend == GdnBackend::Tile, "exact eager mode supports only the Tile GDN backend");
            }
        }
        Ok(())
    }

    fn validate_weight_format(&self, format: WeightFormat) -> Result<()> {
        ensure!(self.execution_mode != ExecutionMode::ExactEager || format == WeightFormat::Bf16,
                "exact eager mode supports only bf16 weights");
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

    /// CPU-only validation of the opt-in warmup contract, before any CUDA context or model allocation.
    /// Does nothing when bounded exact warmup is disabled; execution choices are validated separately.
    pub fn validate_bounded_exact_warmup(&self) -> Result<()> {
        if !self.bounded_exact_warmup { return Ok(()); }
        ensure!(self.warmup, "bounded exact warmup requires warmup=true");
        ensure!(self.execution_mode == ExecutionMode::ExactEager, "bounded exact warmup requires exact eager execution");
        ensure!(!self.cuda_graphs, "bounded exact warmup must not enable CUDA graphs");
        ensure!(self.gdn_backend == Some(GdnBackend::Tile), "bounded exact warmup requires explicit Tile GDN backend");
        ensure!(self.debug_permute_hidden.is_none() && self.taps.layers.is_empty() && !self.taps.norm
            && !self.taps.sublayers && !self.taps.answer_rows,
            "bounded exact warmup does not support debug permutations/taps");
        ensure!(self.max_prefixes >= 1 && self.max_answers >= 16 && self.max_segments >= 17,
            "bounded exact warmup requires capacity Q16/S17/P1; coverage must not shrink");
        let plan = self.bucket_plan(&self.buckets)?;
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
    /// Dense rows requested from kernels/GEMMs. Bucketed mode uses capacity rows; exact eager uses logical rows.
    pub execution_tokens: usize,
    pub execution_answers: usize,
    pub execution_segments: usize,
    pub execution_mode: ExecutionMode,
    /// Graph replay (else eager). Always false in exact eager mode.
    pub replayed: bool,
    /// Steps the batch was split into.
    pub steps: usize,
    /// Allocation bucket tokens summed over those steps (not executed rows or hardware FLOPs in exact mode).
    pub total_bucket_tokens: usize,
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
    /// The checkpoint's weight format (the engine's name in `/metrics`).
    pub format: WeightFormat,
    warmup_report: Option<WarmupReport>,
}

// SAFETY: every member is either plain data or a device/pinned allocation owned by this engine; the scheduler
// drives the engine from one worker thread at a time (`DecisionEngine: Send`, not `Sync`).
unsafe impl Send for GpuEngine {}

impl GpuEngine {
    /// Load `model_dir` (a bf16 or NVFP4-MLP checkpoint, [`ModelConfig::profile`]) onto device 0 and build the
    /// forward. `slot_ids` are the slot label token ids from the frontend (the readout rows).
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
        let format = cfg.profile()?.weights;
        opts.validate_weight_format(format)?;
        if let Some(want) = opts.weights {
            ensure!(
                want == format,
                "the {} engine needs a {} checkpoint, but {} is {} (config.json quantization_config: {:?})",
                want.name(),
                want.name(),
                model_dir.display(),
                format.name(),
                cfg.quantization_config.as_ref().map(|q| q.format.as_str())
            );
        }
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
        let (mut weights, load_report) = ModelWeights::load(&cfg, &ckpt, &stream, &lopts)?;
        let hidden_perm = opts.debug_permute_hidden.map(|seed| debug_permutation(cfg.text_config.hidden_size, seed));
        if let Some(p) = &hidden_perm {
            weights.permute_hidden(&stream, p)?;
        }
        check_weight_format(&weights, format)?;
        let swiglu = format == WeightFormat::Nvfp4 && opts.nvfp4_fused && opts.nvfp4_swiglu;
        if swiglu {
            // Row blocks of half the GEMM's BN: one weight tile = the gate and up rows of BN/2 product columns.
            weights.interleave_gate_up_nvfp4(&stream, crate::kernels::nvfp4::GemmTiles::DEFAULT.bn / 2)?;
        }
        let (cc_major, cc_minor) = ctx.compute_capability()?;
        let (fp4_gemm, fp4_why) = match (opts.fp4_gemm, fp4_gemm_from_env()?) {
            (Some(g), _) => (g, "requested"),
            (None, Some(g)) => (g, "$SYSONE_FP4_GEMM"),
            (None, None) => (Fp4Gemm::default_for_arch(cc_major as u32, cc_minor as u32), "default for this architecture"),
        };
        let fp4_lt = format == WeightFormat::Nvfp4 && fp4_gemm == Fp4Gemm::CuBlasLt;
        if fp4_lt {
            let (n, bytes) = weights.add_lt_scales_down(&stream)?;
            eprintln!("sysone engine: FP4 down GEMM on cuBLASLt: {n} down projections given tiled block scales \
                       ({:.1} MiB)", bytes as f64 / (1u64 << 20) as f64);
        }
        // S2's residual add in the down GEMM epilogue exists only for the cuTile FP4 GEMM: cuBLASLt's beta=1 epilogue
        // rounds once, not bf16(bf16(acc*alpha) + r), so it cannot carry it bit-identically. Where cuBLASLt runs the
        // down projection (the sm_121 default) the add stays in the next norm; C1's GEMM saves more than S2's fusion
        // there (docs/S2.md, "C1 and S2").
        let nvfp4_residual = opts.nvfp4_residual && !fp4_lt;
        let fopts = ForwardOptions { nvfp4_fused: opts.nvfp4_fused, nvfp4_swiglu: swiglu, trim_last_layer: opts.trim_last_layer,
                                     nvfp4_residual };
        let fwd = forward::build_opts(&cfg, &weights, slot_ids.len(), &opts.taps, fopts)?;
        if opts.trim_last_layer && !fwd.trimmed {
            eprintln!("sysone engine: last layer not trimmed (a debug tap needs its rows, or it is not full attention)");
        }
        let mut opts = opts;
        opts.fp4_gemm = Some(fp4_gemm);
        opts.nvfp4_residual = nvfp4_residual;
        if let Some(frac) = opts.memory_fraction {
            ctx.bind_to_thread()?;
            opts.buckets = fit_buckets(&fwd, &opts, gdn_backend, slot_ids.len(), frac)?;
        }
        // Memory fitting must not silently shrink the fixed warmup coverage.
        opts.validate_bounded_exact_warmup()?;
        let plan = opts.bucket_plan(&opts.buckets)?;
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
        let mut blas = Blas::with_mode(&stream, opts.gemm)?;
        if fp4_lt {
            let inter = cfg.text_config.intermediate_size;
            blas.enable_fp4_lt(largest.tokens, inter / crate::kernels::nvfp4::GROUP, (major as u32, minor as u32))?;
        }
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
            format,
            warmup_report: None,
        };
        eprintln!(
            "sysone engine: {} weights ({} NVFP4 projections{}): loaded {:.1} GiB in {:.1} s; arena {:.2} GiB (largest bucket {} tokens), GDN scratch \
             {:.2} GiB ({} prefix states)",
            format.name(),
            e.load_report.nvfp4_projections,
            if format == WeightFormat::Nvfp4 {
                if !e.opts.nvfp4_fused {
                    ", standalone quantize"
                } else if e.opts.nvfp4_swiglu && e.opts.nvfp4_residual {
                    ", quantize fused, SiLU·mul in the gate/up GEMM, residual add in the down GEMM"
                } else if e.opts.nvfp4_swiglu {
                    ", quantize fused, SiLU·mul in the gate/up GEMM"
                } else {
                    ", quantize fused"
                }
            } else {
                ""
            },
            e.load_report.device_bytes as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64(),
            e.graphs.arena.bytes() as f64 / (1u64 << 30) as f64,
            largest.tokens,
            e.kernels.scratch_bytes() as f64 / (1u64 << 30) as f64,
            e.opts.max_prefixes
        );
        eprintln!("sysone engine: bf16 GEMMs {} ({}), last layer {}, {} answer rows at the largest bucket",
                  e.opts.gemm.name(),
                  if e.opts.gemm == GemmMode::Pinned { "batch-invariant" } else { "not batch-invariant" },
                  if e.fwd.trimmed { "trimmed to the answer rows" } else { "untrimmed" }, largest.answers);
        if e.format == WeightFormat::Nvfp4 {
            eprintln!("sysone engine: FP4 down GEMM {} ({fp4_why}){}", fp4_gemm.name(),
                      if fp4_lt { "; one pinned algorithm per weight shape, no split-K: batch-invariant" } else { "" });
        }
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

    /// The device's name (`NVIDIA GB10`).
    pub fn device_name(&self) -> Result<String> {
        Ok(self.ctx.device_name()?)
    }

    /// Free and total device memory (binds the engine's context to the calling thread).
    pub fn memory(&self) -> Result<(usize, usize)> {
        self.ctx.bind_to_thread()?;
        device_memory()
    }

    /// Bounded startup report, completed before load returns; None for standard or disabled warmup.
    pub fn warmup_report(&self) -> Option<&WarmupReport> { self.warmup_report.as_ref() }

    /// Captured allocation buckets (diagnostic proof; exact warmup must always report zero).
    pub fn captured_bucket_count(&self) -> usize {
        self.graphs.plan.buckets.iter().filter(|&&b| self.graphs.is_captured(b)).count()
    }

    /// The GDN delta-rule backend this engine runs.
    pub fn gdn_backend(&self) -> GdnBackend {
        self.kernels.gdn_backend()
    }

    /// Change the speed-only launch choices of the element-wise kernels (F1 A/B benchmarks and tests; every
    /// choice is byte-identical). Drops the captured CUDA graphs, which recorded the old launches.
    pub fn set_kernel_tuning(&mut self, tuning: crate::engine::kernels::KernelTuning) {
        if self.kernels.tuning != tuning {
            self.kernels.tuning = tuning;
            self.graphs.clear_captured();
        }
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
    /// metadata header and GDN logical row count use the batch's actual T; bucketed mode pads to capacity.
    fn stage(&mut self, batch: &DecisionBatch, b: Bucket, shape: StepShape) -> Result<()> {
        self.ctx.bind_to_thread()?;
        ensure!(shape.fits(b), "logical shape {shape:?} exceeds capacity {b:?}");
        self.kernels.prepare(b, &self.stream)?;
        let slot = self.meta.pack(batch, shape.tokens)?;
        self.meta.upload(slot, &self.stream)?;
        self.kernels.update_for_tokens(b, batch, shape.tokens, &self.stream)
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
            total_bucket_tokens: b.tokens,
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

/// The loaded weights must be what `format` promises: no NVFP4 projection for bf16; for NVFP4 every MLP projection
/// (and nothing else, which `forward::build` also refuses).
fn check_weight_format(w: &ModelWeights, format: WeightFormat) -> Result<()> {
    for (i, l) in w.layers.iter().enumerate() {
        let mlp = l.mlp();
        for (name, lin) in [("gate/up", &mlp.gate_up), ("down", &mlp.down)] {
            ensure!(lin.is_nvfp4() == (format == WeightFormat::Nvfp4),
                    "layer {i} MLP {name} is {} in a checkpoint whose profile is {}",
                    if lin.is_nvfp4() { "NVFP4" } else { "bf16" }, format.name());
        }
    }
    Ok(())
}

/// Free and total device memory of the current context.
pub fn device_memory() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: plain out-parameters; the caller's context is current on this thread.
    let r = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _) };
    ensure!(r == cuda_core::sys::cudaError_enum_CUDA_SUCCESS, "cuMemGetInfo failed: {r:?}");
    Ok((free, total))
}

/// Device bytes an engine with largest bucket `tokens` allocates beyond its weights: the activation arena, the GDN
/// scratch, the metadata block and the logits readback (the pinned host buffer is not device memory but is
/// counted too: on an integrated GPU it comes out of the same pool).
pub fn step_bytes(fwd: &Forward, opts: &EngineOptions, backend: GdnBackend, n_slots: usize, tokens: usize) -> Result<usize> {
    let plan = opts.bucket_plan(&[tokens])?;
    let b = plan.largest();
    let arena = fwd.graph.plan(b)?.arena_bytes;
    let scratch = crate::kernels::gdn::GdnScratch::engine_bytes(backend, tokens, opts.max_prefixes);
    // Metadata: ids, positions, token_seg (3 x u32 per token), segments (16 B) and seg_state, answers (rows, slots, positions).
    let meta = 2 * (12 * b.tokens + 20 * b.segments + 12 * b.answers + 4096);
    let logits = b.answers * n_slots * 4 * 2;
    Ok(arena + scratch + meta + logits)
}

/// [`EngineOptions::memory_fraction`]: the configured buckets up to the largest whose [`step_bytes`] fit in `frac`
/// of the device memory still free (weights are loaded at this point; GB10's unified pool counts host use too).
fn fit_buckets(fwd: &Forward, opts: &EngineOptions, backend: GdnBackend, n_slots: usize, frac: f64) -> Result<Vec<usize>> {
    ensure!(frac > 0.0 && frac <= 1.0, "memory fraction {frac} must be in (0, 1]");
    let (free, total) = device_memory()?;
    let budget = (free as f64 * frac) as usize;
    let mut buckets = opts.buckets.clone();
    buckets.sort_unstable();
    buckets.dedup();
    let mut keep = Vec::new();
    for &t in &buckets {
        let need = step_bytes(fwd, opts, backend, n_slots, t)?;
        if need > budget {
            eprintln!(
                "sysone engine: bucket {t} needs {:.2} GiB, over {frac} of the {:.2} GiB free ({:.2} GiB total): \
                 largest bucket is {:?}",
                need as f64 / GIB, free as f64 / GIB, total as f64 / GIB, keep.last()
            );
            break;
        }
        keep.push(t);
    }
    ensure!(!keep.is_empty(), "not even the smallest bucket ({:?}) fits in {frac} of {:.2} GiB free device memory",
            buckets.first(), free as f64 / GIB);
    Ok(keep)
}

const GIB: f64 = (1u64 << 30) as f64;

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
            ("engine", self.format.name().into()),
            ("gemm", self.opts.gemm.name().into()),
            ("trim_last_layer", self.fwd.trimmed.to_string()),
            ("execution_mode", self.opts.execution_mode.name(self.opts.cuda_graphs).into()),
            ("cuda_graphs", graph_policy.into()),
            ("warmup_policy", self.opts.warmup_policy().into()),
            ("gdn_backend", self.gdn_backend().name().into()),
            ("arch", self.arch.clone()),
        ]
    }

    fn step_limits(&self) -> StepLimits {
        let largest = self.graphs.plan.largest();
        StepLimits {
            buckets: self.graphs.plan.buckets.iter().map(|b| b.tokens).collect(),
            max_answers: largest.answers,
            // One segment slot of the largest bucket is the padding segment (used whenever a batch is smaller).
            max_segments: largest.segments - 1,
            max_prefixes: self.kernels.max_prefixes(),
            tokens_per_answer: self.opts.tokens_per_answer,
        }
    }

    fn last_bucket_tokens(&self) -> Option<usize> {
        Some(self.last.total_bucket_tokens)
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
        let mut total_bucket_tokens = 0;
        for (sub, rows) in parts {
            let Some(b) = self.pick_bucket(&sub, 0) else { bail!("no bucket fits a split batch") };
            let l = self.step(&sub, b, false)?;
            total_bucket_tokens += b.tokens;
            for (r, &a) in rows.iter().enumerate() {
                data[a * MAX_SLOTS..(a + 1) * MAX_SLOTS].copy_from_slice(&l.data[r * MAX_SLOTS..(r + 1) * MAX_SLOTS]);
            }
        }
        self.last.steps = n;
        self.last.total_bucket_tokens = total_bucket_tokens;
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
    fn exact_rejects_explicit_and_autodetected_nvfp4_before_allocation() {
        let mut opts = EngineOptions { execution_mode: ExecutionMode::ExactEager, cuda_graphs: false,
            gdn_backend: Some(GdnBackend::Tile), ..Default::default() };
        opts.validate_weight_format(WeightFormat::Bf16).unwrap();
        assert!(opts.validate_weight_format(WeightFormat::Nvfp4).is_err());
        opts.weights = Some(WeightFormat::Nvfp4);
        let err = GpuEngine::load(Path::new("nonexistent-model"), &[1, 2], opts).err().unwrap();
        assert_eq!(err.to_string(), "exact eager mode supports only bf16 weights");
        // Bucketed execution also accepts NVFP4 weights.
        EngineOptions::default().validate_weight_format(WeightFormat::Nvfp4).unwrap();
    }

    #[test]
    fn bounded_capacity_uses_upstream_answer_density() {
        let mut opts = bounded_options();
        opts.buckets = vec![100];
        assert!(opts.validate_bounded_exact_warmup().is_err()); // Q cap = ceil(100/8) = 13.
        opts.buckets = vec![128];
        opts.validate_bounded_exact_warmup().unwrap();
        opts.tokens_per_answer = 16;
        assert!(opts.validate_bounded_exact_warmup().is_err());
        opts.buckets = vec![256];
        opts.validate_bounded_exact_warmup().unwrap();
        opts.taps.answer_rows = true;
        assert!(opts.validate_bounded_exact_warmup().is_err());
    }

    #[test]
    fn legacy_engine_defaults_remain_unchanged() {
        let opts = EngineOptions::default();
        assert_eq!(opts.buckets, vec![256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]);
        assert_eq!((opts.max_answers, opts.max_segments, opts.max_prefixes), (2048, 2048, 256));
        assert_eq!(opts.execution_mode, ExecutionMode::Bucketed);
        assert!(opts.cuda_graphs && opts.warmup && !opts.bounded_exact_warmup);
        assert_eq!(opts.gdn_backend, None);
        assert_eq!(opts.weights, None);
        assert_eq!(opts.gemm, GemmMode::Pinned);
        assert!(opts.trim_last_layer && opts.nvfp4_fused && opts.nvfp4_swiglu && opts.nvfp4_residual);
        assert_eq!(opts.tokens_per_answer, 8);
        assert_eq!(opts.memory_fraction, None);
        assert_eq!(opts.fp4_gemm, None);
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
        good.buckets = vec![128]; good.max_answers = 16; good.max_segments = 17; good.max_prefixes = 1;
        good.validate_bounded_exact_warmup().unwrap();
        for dim in 0..4 {
            let mut bad = good.clone();
            match dim { 0 => bad.buckets = vec![100], 1 => bad.max_answers = 15,
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
