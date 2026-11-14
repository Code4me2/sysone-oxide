//! The engine's [`Kernels`] over the W3 cuTile kernels (`crate::kernels`): GDN (W3a), attention, norms,
//! pointwise ops (W3b). Phase 2a, I1.
//!
//! The adapters translate arena views ([`DevTensor`]) into the kernels' own argument types (borrowed cutile
//! tensors, [`DView`]s, raw device pointers) and launch on the step's execution context, so every launch can be
//! recorded into a T-bucket CUDA graph. Batch-dependent state lives in fixed device buffers per bucket:
//! - a [`GdnPlan`] (chunk and segment tables) sized for the bucket's token and segment capacities;
//! - the [`AttnPlan`] work list, padded to the bucket's maximum item count (the grid of a captured launch).
//!
//! [`TileKernels::update`] rewrites both for each batch before the forward runs (eagerly or as a replay).
//!
//! GDN scratch (gates, chunk inverses, prefix states; f32 q, k and the intra-chunk scores for the cuTile path) is
//! one [`GdnScratch`] ([`GdnScratch::for_engine`]) sized for the largest bucket and the engine's prefix capacity,
//! shared by all buckets and layers (one step runs at a time). The conv output and the delta-rule output are arena
//! values instead of scratch fields, so the scratch holds only what the scan kernels read implicitly.
//!
//! The chunked delta rule runs on one of two backends ([`GdnBackend`], fixed per engine): the cuTile kernels
//! (`gdn_prep` + `gdn_chunk_scan`) or the cuda-oxide SIMT kernels (`gdn::simt::{gates, chunk_scan}`, docs/G1.md).
//! The conv and the gated norm are cuTile in both. Profile marks name the SIMT launches (`gdn.simt_*`).

use crate::contract::DecisionBatch;
use crate::engine::graph::tensor::{DevTensor, GraphDType};
use crate::engine::graph::{Bucket, Kernels, LayerWeight, Launch, WeightRef};
use crate::kernels::attention::{self as attn, AttnPlan, AttnTiles, SegAttnArgs, WORK_FIELDS};
use crate::engine::graph::GraphOp;
use crate::kernels::gdn::{self, DView, GdnBackend, GdnPlan, GdnScratch, GdnTables, GdnWeights};
use crate::engine::weights::Linear;
use crate::kernels::nvfp4::{self, GemmTiles, QuantTiles, RawAlpha};
use crate::kernels::{norm, pointwise};
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use anyhow::{Context as _, Result, anyhow, ensure};
use cuda_async::device_buffer::DevicePointer;
use cuda_async::device_operation::DeviceOp;
use cuda_core::{PinnedHostBuffer, Stream};
use cutile::half::bf16;
use std::collections::HashMap;
use std::sync::Arc;

/// Fixed per-bucket device state of the segment-aware kernels.
struct BucketState {
    gdn: GdnPlan,
    /// `[attn_items, WORK_FIELDS]` i32 (an [`AttnPlan`] padded with empty items).
    attn_work: DevTensor,
    attn_items: usize,
    /// Pinned staging for the work list (rewritten only after the previous step has completed).
    attn_host: PinnedHostBuffer<i32>,
    /// The answer rows' work list ([`AttnPlan::answers`], padded to the bucket's answer capacity) and its staging,
    /// for [`GraphOp::SegAttnAnswers`] (L1, the trimmed last layer).
    ans_work: DevTensor,
    ans_items: usize,
    ans_host: PinnedHostBuffer<i32>,
}

pub struct TileKernels {
    eps: f32,
    pub attn_tiles: AttnTiles,
    /// `inv_freq` of the partial RoPE, `[32]` f32.
    inv_freq: DevTensor,
    scratch: GdnScratch,
    gdn_backend: GdnBackend,
    buckets: HashMap<Bucket, BucketState>,
    /// Multiprocessors of the device (persistent NVFP4 GEMM schedules).
    sms: usize,
    /// Tiles of the NVFP4 GEMM ([`GemmTiles::DEFAULT`]: fixed at every T, so a row's result never depends on the
    /// batch it is in).
    pub nvfp4_tiles: GemmTiles,
    /// Launch shapes and occupancy hints that change speed only (F1; every choice is byte-identical).
    pub tuning: KernelTuning,
}

/// Speed-only launch choices of the element-wise kernels (F1, docs/F1.md). None of them changes a result bit:
/// `tests/kernels_nvfp4_f1.rs` checks each against the default launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelTuning {
    /// cuTile `occupancy` hint of `add_zc_rmsnorm` and `add_zc_rmsnorm_nvfp4` (tile blocks per SM).
    pub norm_occupancy: Option<i32>,
    /// Tile shape / occupancy of `silu_mul_nvfp4`.
    pub silu: nvfp4::SiluTiles,
    /// cuTile `occupancy` hint of the fused gate/up + SiLU·mul GEMM (`GemmNvfp4SwiGlu`).
    pub swiglu_occupancy: Option<i32>,
}

impl KernelTuning {
    /// GB10-measured (bench_nvfp4_mlp): two norm rows per SM (+9% / +26% bandwidth), SiLU·mul + quantize at three
    /// tile blocks per SM (+55%), two fused gate/up GEMM tiles per SM (its epilogue then overlaps the other tile's
    /// main loop: 16.3 -> 13.2 ms at 16k, the cost of the plain gate/up GEMM alone).
    pub const DEFAULT: KernelTuning = KernelTuning {
        norm_occupancy: Some(2),
        silu: nvfp4::SiluTiles { rows: 1, cols: nvfp4::SILU_Q_BLOCK, occupancy: Some(3) },
        swiglu_occupancy: Some(2),
    };
    /// The launches as N1 had them (compiler defaults).
    pub const N1: KernelTuning = KernelTuning { norm_occupancy: None, silu: nvfp4::SiluTiles::N1, swiglu_occupancy: None };
}

// SAFETY: the cutile tensors and plans hold device allocations only; the engine's single worker thread drives
// every launch, and `&self` methods only enqueue work.
unsafe impl Send for TileKernels {}
unsafe impl Sync for TileKernels {}

fn launch_err(what: &str) -> impl FnOnce(cuda_async::error::DeviceError) -> anyhow::Error + '_ {
    move |e| anyhow!("{what} launch: {e:?}")
}

/// A [`DView`] over a rank-2 (possibly column-strided) arena view.
fn dview<T: cuda_core::DType>(t: &DevTensor, dtype: GraphDType) -> Result<DView<T>> {
    ensure!(t.dtype == dtype && t.shape.len() == 2, "expected a rank-2 {dtype:?} view, got {t:?}");
    // SAFETY: the address is a live device allocation held by the view's buffer.
    let ptr = unsafe { DevicePointer::from_cu_deviceptr(t.ptr()) };
    Ok(DView { ptr, rows: t.shape[0], cols: t.shape[1], row_stride: t.row_stride })
}

fn dptr<T>(t: &DevTensor) -> DevicePointer<T> {
    // SAFETY: as in `dview`.
    unsafe { DevicePointer::from_cu_deviceptr(t.ptr()) }
}

impl TileKernels {
    /// Kernels for an engine whose largest bucket has `max_tokens` tokens and whose batches hold at most
    /// `max_prefixes` prefix segments (GDN state slots: 2 MiB each), running the delta rule on `gdn_backend`.
    /// For [`GdnBackend::Simt`] the SIMT module for the current device is loaded here, so a missing or stale
    /// kernel artifact for the device's architecture is a load error (never a fallback to cuTile).
    pub fn new(stream: &Arc<Stream>, eps: f32, max_tokens: usize, max_prefixes: usize, gdn_backend: GdnBackend)
        -> Result<Self> {
        ensure!(max_prefixes > 0, "max_prefixes must be > 0");
        if gdn_backend == GdnBackend::Simt {
            let device = cuda_async::device_context::get_default_device();
            crate::kernels::simt::SimtModule::get(device).context("GDN backend simt")?;
        }
        let inv: Vec<u8> = attn::rope_inv_freq().iter().flat_map(|x| x.to_le_bytes()).collect();
        let inv_freq = DevTensor::alloc(stream.context(), GraphDType::F32, &[attn::ROTARY_DIM / 2])?;
        inv_freq.copy_from_host(stream, &inv)?;
        let scratch = GdnScratch::for_engine(gdn_backend, max_tokens, max_prefixes, stream)?;
        let sms = stream.context().multiprocessor_count()? as usize;
        Ok(Self {
            eps,
            attn_tiles: AttnTiles::default(),
            inv_freq,
            scratch,
            gdn_backend,
            buckets: HashMap::new(),
            sms,
            nvfp4_tiles: GemmTiles::DEFAULT,
            tuning: KernelTuning::DEFAULT,
        })
    }

    /// The delta-rule backend.
    pub fn gdn_backend(&self) -> GdnBackend {
        self.gdn_backend
    }

    /// Prefix segments a batch may hold (GDN state slots).
    pub fn max_prefixes(&self) -> usize {
        self.scratch.slots
    }

    /// Device bytes of the GDN scratch.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.bytes()
    }

    /// Allocate `b`'s plan buffers (idempotent; never during capture).
    pub fn prepare(&mut self, b: Bucket, stream: &Arc<Stream>) -> Result<()> {
        if self.buckets.contains_key(&b) {
            return Ok(());
        }
        ensure!(b.tokens <= self.scratch.tokens, "bucket of {} tokens over the scratch's {}", b.tokens, self.scratch.tokens);
        let max_chunks = GdnTables::chunk_capacity(b.tokens, b.segments);
        let gdn = GdnPlan::with_capacity(&[], b.tokens, max_chunks, b.segments, stream)?;
        let attn_items = AttnPlan::max_items(b.tokens, b.segments, self.attn_tiles.bm);
        let attn_work = DevTensor::zeros(stream, GraphDType::I32, &[attn_items * WORK_FIELDS])?;
        let attn_host = PinnedHostBuffer::zeroed(stream.context(), attn_items * WORK_FIELDS)?;
        let ans_items = b.answers.max(1);
        let ans_work = DevTensor::zeros(stream, GraphDType::I32, &[ans_items * WORK_FIELDS])?;
        let ans_host = PinnedHostBuffer::zeroed(stream.context(), ans_items * WORK_FIELDS)?;
        self.buckets.insert(b, BucketState { gdn, attn_work, attn_items, attn_host, ans_work, ans_items, ans_host });
        Ok(())
    }

    /// Rewrite `b`'s segment tables and attention work list for `batch` (validated; it must fit `b`). The
    /// previous step must have completed (the work list's pinned staging is reused). Fails, before anything
    /// runs, when the batch has more prefixes than the GDN scratch has state slots: a replayed scan would
    /// otherwise write past the state buffer (`GdnPlan::update` does not check the scratch).
    pub fn update(&mut self, b: Bucket, batch: &DecisionBatch, stream: &Arc<Stream>) -> Result<()> {
        let slots = self.scratch.slots;
        let st = self.buckets.get_mut(&b).with_context(|| format!("bucket {b:?} not prepared"))?;
        let prefixes = batch.segments.iter().filter(|s| s.is_prefix()).count();
        ensure!(prefixes <= slots, "batch has {prefixes} prefixes; the GDN scratch holds {slots} prefix states");
        st.gdn.update(&batch.segments, stream)?;
        ensure!(st.gdn.num_prefixes <= slots, "GDN plan has {} prefixes for {slots} state slots", st.gdn.num_prefixes);
        let mut plan = AttnPlan::new(&batch.segments, self.attn_tiles.bm);
        ensure!(plan.num_items() <= st.attn_items, "{} attention items over the bucket's {}", plan.num_items(),
                st.attn_items);
        plan.pad_to(st.attn_items);
        st.attn_host.as_mut_slice().copy_from_slice(&plan.work);
        let mut ans = AttnPlan::answers(&batch.segments);
        ensure!(ans.num_items() <= st.ans_items, "{} answer rows over the bucket's {}", ans.num_items(), st.ans_items);
        ans.pad_to(st.ans_items);
        st.ans_host.as_mut_slice().copy_from_slice(&ans.work);
        stream.context().bind_to_thread()?;
        unsafe {
            cuda_core::memory::memcpy_htod_async(st.attn_work.ptr(), st.attn_host.as_ptr(), plan.work.len() * 4,
                                                 stream.cu_stream())?;
            cuda_core::memory::memcpy_htod_async(st.ans_work.ptr(), st.ans_host.as_ptr(), ans.work.len() * 4,
                                                 stream.cu_stream())?
        };
        Ok(())
    }

    fn bucket(&self, b: Bucket) -> Result<&BucketState> {
        self.buckets.get(&b).with_context(|| format!("bucket {b:?} not prepared"))
    }

    fn gdn_weights<'a>(cx: &'a Launch, layer: usize) -> Result<[&'a DevTensor; 4]> {
        let w = |s| cx.weights.tensor(WeightRef::Layer(layer, s));
        Ok([w(LayerWeight::GdnConv)?, w(LayerWeight::GdnALog)?, w(LayerWeight::GdnDtBias)?, w(LayerWeight::GdnNorm)?])
    }
}

impl Kernels for TileKernels {
    fn zc_rmsnorm(&self, cx: &Launch, x: &DevTensor, w: &DevTensor, out: &DevTensor) -> Result<()> {
        unsafe {
            let (x, w, mut o) = (x.tile::<bf16>()?, w.tile::<bf16>()?, out.tile::<bf16>()?);
            norm::zc_rmsnorm(&mut o, &x, &w, self.eps)?.execute(cx.exec).map(|_| ()).map_err(launch_err("zc_rmsnorm"))
        }
    }

    fn add_zc_rmsnorm(&self, cx: &Launch, residual: &DevTensor, x: &DevTensor, w: &DevTensor, out: &DevTensor)
        -> Result<()> {
        unsafe {
            let (x, w) = (x.tile::<bf16>()?, w.tile::<bf16>()?);
            let (mut r, mut o) = (residual.tile::<bf16>()?, out.tile::<bf16>()?);
            norm::add_zc_rmsnorm_with(&mut r, &mut o, &x, &w, self.eps, self.tuning.norm_occupancy)?
                .execute(cx.exec)
                .map(|_| ())
                .map_err(launch_err("add_zc_rmsnorm"))
        }
    }

    fn silu_mul(&self, cx: &Launch, gu: &DevTensor, out: &DevTensor) -> Result<()> {
        unsafe {
            let (gu, mut o) = (gu.tile::<bf16>()?, out.tile::<bf16>()?);
            pointwise::silu_mul(&mut o, &gu)?.execute(cx.exec).map(|_| ()).map_err(launch_err("silu_mul"))
        }
    }

    fn quantize_nvfp4(&self, cx: &Launch, x: &DevTensor, a_gs: f32, q: &DevTensor, s: &DevTensor) -> Result<()> {
        ensure!(x.is_contiguous(), "quantize_nvfp4 of a strided view {x:?}");
        let (t, k) = (x.rows(), x.cols());
        unsafe {
            let x = Arc::new(x.tile::<bf16>()?);
            let out = nvfp4::Nvfp4Act { t, k, packed: q.tile::<f4e2m1fnx2>()?, scales: s.tile::<f8e4m3fn>()? };
            nvfp4::quantize_op(&x, a_gs, out, QuantTiles::DEFAULT)?
                .execute(cx.exec)
                .map(|_| ())
                .map_err(launch_err("quantize_nvfp4"))
        }
    }

    fn zc_rmsnorm_nvfp4(&self, cx: &Launch, x: &DevTensor, norm: &DevTensor, a_gs: f32, q: &DevTensor, s: &DevTensor)
        -> Result<()> {
        unsafe {
            let (x, w) = (x.tile::<bf16>()?, norm.tile::<bf16>()?);
            let (mut q, mut s) = (q.tile::<f4e2m1fnx2>()?, s.tile::<f8e4m3fn>()?);
            nvfp4::zc_rmsnorm_nvfp4(&mut q, &mut s, &x, &w, self.eps, a_gs)?
                .execute(cx.exec)
                .map(|_| ())
                .map_err(launch_err("zc_rmsnorm_nvfp4"))
        }
    }

    fn add_zc_rmsnorm_nvfp4(&self, cx: &Launch, residual: &DevTensor, x: &DevTensor, norm: &DevTensor, a_gs: f32,
                            q: &DevTensor, s: &DevTensor) -> Result<()> {
        unsafe {
            let (x, w, mut r) = (x.tile::<bf16>()?, norm.tile::<bf16>()?, residual.tile::<bf16>()?);
            let (mut q, mut s) = (q.tile::<f4e2m1fnx2>()?, s.tile::<f8e4m3fn>()?);
            nvfp4::add_zc_rmsnorm_nvfp4_with(&mut r, &mut q, &mut s, &x, &w, self.eps, a_gs, self.tuning.norm_occupancy)?
                .execute(cx.exec)
                .map(|_| ())
                .map_err(launch_err("add_zc_rmsnorm_nvfp4"))
        }
    }

    fn silu_mul_nvfp4(&self, cx: &Launch, gu: &DevTensor, a_gs: f32, q: &DevTensor, s: &DevTensor) -> Result<()> {
        unsafe {
            let gu = gu.tile::<bf16>()?;
            let (mut q, mut s) = (q.tile::<f4e2m1fnx2>()?, s.tile::<f8e4m3fn>()?);
            if self.tuning.silu == nvfp4::SiluTiles::N1 {
                // N1's own entry point (the reference launch of the F1 bit-identity tests).
                nvfp4::silu_mul_nvfp4(&mut q, &mut s, &gu, a_gs)?
                    .execute(cx.exec)
                    .map(|_| ())
                    .map_err(launch_err("silu_mul_nvfp4"))
            } else {
                nvfp4::silu_mul_nvfp4_with(&mut q, &mut s, &gu, a_gs, self.tuning.silu)?
                    .execute(cx.exec)
                    .map(|_| ())
                    .map_err(launch_err("silu_mul_nvfp4"))
            }
        }
    }

    fn gemm_nvfp4(&self, cx: &Launch, q: &DevTensor, s: &DevTensor, w: &Linear, out: &DevTensor) -> Result<()> {
        let Linear::Nvfp4 { packed, scales, parts, a_gs, gate_up_block, .. } = w else {
            anyhow::bail!("gemm_nvfp4 on a bf16 projection");
        };
        ensure!(gate_up_block.is_none(), "gemm_nvfp4 on a gate/up weight interleaved for the fused SiLU·mul GEMM");
        ensure!(q.is_contiguous() && s.is_contiguous() && out.is_contiguous(), "gemm_nvfp4 needs contiguous operands");
        let parts: Vec<(usize, f32)> =
            parts.iter().map(|p| Ok((p.rows, p.w_gs.context("NVFP4 part without weight_global_scale")?))).collect::<Result<_>>()?;
        let alpha = RawAlpha::for_parts(*a_gs, &parts)?;
        unsafe {
            let (x, xs) = (Arc::new(q.tile::<f4e2m1fnx2>()?), Arc::new(s.tile::<f8e4m3fn>()?));
            let (wp, ws) = (Arc::new(packed.tile::<f4e2m1fnx2>()?), Arc::new(scales.tile::<f8e4m3fn>()?));
            let z = Arc::new(out.tile::<bf16>()?);
            nvfp4::gemm_raw_op(&x, &xs, &wp, &ws, alpha, z, self.nvfp4_tiles, self.sms)?
                .execute(cx.exec)
                .map(|_| ())
                .map_err(launch_err("gemm_nvfp4"))
        }
    }

    fn gemm_nvfp4_swiglu(&self, cx: &Launch, q: &DevTensor, s: &DevTensor, w: &Linear, a_gs_down: f32,
                         q_out: &DevTensor, s_out: &DevTensor) -> Result<()> {
        let Linear::Nvfp4 { packed, scales, parts, a_gs, gate_up_block, .. } = w else {
            anyhow::bail!("gemm_nvfp4_swiglu on a bf16 projection");
        };
        let tiles = self.nvfp4_tiles;
        ensure!(*gate_up_block == Some(tiles.bn / 2),
                "gemm_nvfp4_swiglu needs the gate/up rows interleaved in blocks of {} (BN / 2), got {gate_up_block:?}",
                tiles.bn / 2);
        ensure!(q.is_contiguous() && s.is_contiguous() && q_out.is_contiguous() && s_out.is_contiguous(),
                "gemm_nvfp4_swiglu needs contiguous operands");
        let parts: Vec<(usize, f32)> =
            parts.iter().map(|p| Ok((p.rows, p.w_gs.context("NVFP4 part without weight_global_scale")?))).collect::<Result<_>>()?;
        let alpha = RawAlpha::for_parts(*a_gs, &parts)?;
        unsafe {
            let (x, xs) = (Arc::new(q.tile::<f4e2m1fnx2>()?), Arc::new(s.tile::<f8e4m3fn>()?));
            let (wp, ws) = (Arc::new(packed.tile::<f4e2m1fnx2>()?), Arc::new(scales.tile::<f8e4m3fn>()?));
            let (qo, so) = (Arc::new(q_out.tile::<f4e2m1fnx2>()?), Arc::new(s_out.tile::<f8e4m3fn>()?));
            let opts = cutile::tile_kernel::CompileOptions { occupancy: self.tuning.swiglu_occupancy, ..Default::default() };
            nvfp4::gemm_swiglu_il_op_with(&x, &xs, &wp, &ws, alpha, a_gs_down, qo, so, tiles, self.sms, opts)?
                .execute(cx.exec)
                .map(|_| ())
                .map_err(launch_err("gemm_nvfp4_swiglu"))
        }
    }

    fn tree_conv1d_silu(&self, cx: &Launch, _layer: usize, x: &DevTensor, conv_w: &DevTensor, out: &DevTensor)
        -> Result<()> {
        let st = self.bucket(cx.bucket)?;
        unsafe {
            let w = conv_w.tile::<bf16>()?;
            let op = gdn::tree_conv1d_silu(&st.gdn, dview(x, GraphDType::BF16)?, &w, dview(out, GraphDType::BF16)?)?;
            op.execute(cx.exec).map_err(launch_err("tree_conv1d_silu"))
        }
    }

    fn gdn_prep(&self, cx: &Launch, layer: usize, conv: &DevTensor, a: &DevTensor, b: &DevTensor, _a_log: &DevTensor,
                _dt_bias: &DevTensor) -> Result<()> {
        let st = self.bucket(cx.bucket)?;
        let [cw, al, dt, nw] = Self::gdn_weights(cx, layer)?;
        unsafe {
            let (cw, al, dt, nw) = (cw.tile::<bf16>()?, al.tile::<bf16>()?, dt.tile::<bf16>()?, nw.tile::<bf16>()?);
            let w = GdnWeights { conv_w: &cw, a_log: &al, dt_bias: &dt, norm_w: &nw };
            let (a, b) = (dview(a, GraphDType::BF16)?, dview(b, GraphDType::BF16)?);
            match self.gdn_backend {
                GdnBackend::Tile => {
                    let conv = dview(conv, GraphDType::BF16)?;
                    let op = gdn::gdn_prep(&st.gdn, conv, a, b, &w, &self.scratch)?;
                    op.execute(cx.exec).map_err(launch_err("gdn_prep"))
                }
                // The SIMT scan normalises q and k from the conv output itself: only the gates are needed here.
                GdnBackend::Simt => {
                    let op = gdn::simt::gates(&st.gdn, a, b, &w, &self.scratch)?;
                    op.execute(cx.exec).map_err(launch_err("gdn_simt_gates"))
                }
            }
        }
    }

    fn gdn_chunk_scan(&self, cx: &Launch, _layer: usize, conv: &DevTensor, out: &DevTensor) -> Result<()> {
        let st = self.bucket(cx.bucket)?;
        ensure!(st.gdn.num_prefixes <= self.scratch.slots, "GDN plan has {} prefixes for {} state slots",
                st.gdn.num_prefixes, self.scratch.slots);
        let conv = dview(conv, GraphDType::BF16)?;
        let out = dview(out, GraphDType::BF16)?;
        let (mut ops, labels) = match self.gdn_backend {
            // Reads v (the conv output's v columns) and the f32 q, k that gdn_prep stored in the scratch.
            GdnBackend::Tile => (
                unsafe { gdn::gdn_chunk_scan(&st.gdn, conv.columns(2 * gdn::KEY_DIM, gdn::VALUE_DIM)?, &self.scratch, out)? },
                ["gdn.chunk_tinv", "gdn.chunk_wy", "gdn.scan_prefix", "gdn.scan_suffix"],
            ),
            // Reads q, k and v from the whole conv output.
            GdnBackend::Simt => (
                unsafe { gdn::simt::chunk_scan(&st.gdn, conv, &self.scratch, out)? },
                ["gdn.simt_tinv", "gdn.simt_slot_guard", "gdn.simt_scan_prefix", "gdn.simt_scan_suffix"],
            ),
        };
        // Both build four operations in this order: inverse, WY slot (cuTile: empty; SIMT: slot guard), prefix scan,
        // suffix scan.
        for label in labels {
            cx.mark(label)?;
            let op = ops.remove(0);
            unsafe { op.execute(cx.exec) }.map_err(launch_err(label))?;
        }
        Ok(())
    }

    fn profile_label(&self, op: &GraphOp) -> String {
        match (op, self.gdn_backend) {
            (GraphOp::GdnPrep { .. }, GdnBackend::Simt) => "gdn.simt_gates".into(),
            _ => op.profile_label(),
        }
    }

    fn gated_rmsnorm(&self, cx: &Launch, _layer: usize, o: &DevTensor, z: &DevTensor, w: &DevTensor, out: &DevTensor)
        -> Result<()> {
        let st = self.bucket(cx.bucket)?;
        unsafe {
            let w = w.tile::<bf16>()?;
            let op = gdn::gated_rmsnorm(&st.gdn, dview(o, GraphDType::BF16)?, dview(z, GraphDType::BF16)?, &w, self.eps,
                                        dview(out, GraphDType::BF16)?)?;
            op.execute(cx.exec).map_err(launch_err("gated_rmsnorm"))
        }
    }

    fn qk_norm_rope(&self, cx: &Launch, _layer: usize, qg: &DevTensor, kv: &DevTensor, q_norm: &DevTensor,
                    k_norm: &DevTensor, q: &DevTensor, k: &DevTensor) -> Result<()> {
        self.qk_norm_rope_at(cx, &cx.meta.positions, qg, kv, q_norm, k_norm, q, k)
    }

    fn qk_norm_rope_answers(&self, cx: &Launch, _layer: usize, qg: &DevTensor, kv: &DevTensor, q_norm: &DevTensor,
                            k_norm: &DevTensor, q: &DevTensor, k: &DevTensor) -> Result<()> {
        self.qk_norm_rope_at(cx, &cx.meta.answer_pos, qg, kv, q_norm, k_norm, q, k)
    }

    fn seg_attn_answers(&self, cx: &Launch, _layer: usize, q: &DevTensor, k: &DevTensor, kv: &DevTensor,
                        qg: &DevTensor, out: &DevTensor) -> Result<()> {
        let st = self.bucket(cx.bucket)?;
        let (a, t) = (q.rows(), k.rows());
        ensure!(a == st.ans_items, "seg_attn_answers: {a} query rows for {} answer items", st.ans_items);
        for (name, x, rows, cols) in [("q", q, a, 4096), ("k", k, t, 1024), ("kv", kv, t, 2048), ("qg", qg, a, 8192),
                                      ("out", out, a, 4096)] {
            ensure!(x.dtype == GraphDType::BF16 && x.is_contiguous() && x.rows() == rows && x.cols() == cols,
                    "seg_attn_answers {name}: {x:?}");
        }
        let args = SegAttnArgs {
            out: dptr(out),
            q: dptr(q),
            k: dptr(k),
            kv: dptr(kv),
            qg: dptr(qg),
            work: dptr(&st.ans_work),
            num_items: st.ans_items,
        };
        unsafe { attn::seg_attn(args, self.attn_tiles).execute(cx.exec) }.map(|_| ()).map_err(launch_err("seg_attn"))
    }

    fn seg_attn(&self, cx: &Launch, _layer: usize, q: &DevTensor, k: &DevTensor, kv: &DevTensor, qg: &DevTensor,
                out: &DevTensor) -> Result<()> {
        let st = self.bucket(cx.bucket)?;
        let t = q.rows();
        for (name, x, cols) in [("q", q, 4096), ("k", k, 1024), ("kv", kv, 2048), ("qg", qg, 8192), ("out", out, 4096)] {
            ensure!(x.dtype == GraphDType::BF16 && x.is_contiguous() && x.rows() == t && x.cols() == cols,
                    "seg_attn {name}: {x:?}");
        }
        let args = SegAttnArgs {
            out: dptr(out),
            q: dptr(q),
            k: dptr(k),
            kv: dptr(kv),
            qg: dptr(qg),
            work: dptr(&st.attn_work),
            num_items: st.attn_items,
        };
        unsafe { attn::seg_attn(args, self.attn_tiles).execute(cx.exec) }.map(|_| ()).map_err(launch_err("seg_attn"))
    }

    fn gather_rows(&self, cx: &Launch, src: &DevTensor, out: &DevTensor) -> Result<()> {
        let idx = cx.meta.answer_rows.rows_view(0, out.rows())?;
        unsafe {
            let (src, idx, mut o) = (src.tile::<bf16>()?, idx.tile::<u32>()?, out.tile::<bf16>()?);
            pointwise::gather_rows(&mut o, &src, &idx)?.execute(cx.exec).map(|_| ()).map_err(launch_err("gather_rows"))
        }
    }
}

/// The first `t` RoPE positions of a u32 metadata array as the i32 tensor the kernels read (all < 2^31).
fn rope_positions(pos: &DevTensor, t: usize) -> Result<DevTensor> {
    ensure!(pos.numel() >= t, "{t} positions from an array of {}", pos.numel());
    DevTensor::new(pos.buf().clone(), pos.offset(), GraphDType::I32, vec![t])
}

impl TileKernels {
    /// `qk_norm_rope` over `qg.rows()` rows at the positions in `pos` (every token, or the answer rows).
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope_at(&self, cx: &Launch, pos: &DevTensor, qg: &DevTensor, kv: &DevTensor, q_norm: &DevTensor,
                       k_norm: &DevTensor, q: &DevTensor, k: &DevTensor) -> Result<()> {
        let t = qg.rows();
        let (hq, hk, d) = (attn::NUM_Q_HEADS, attn::NUM_KV_HEADS, attn::HEAD_DIM);
        ensure!(t % attn::ROPE_TOKENS == 0, "qk_norm_rope over {t} rows (not a multiple of {})", attn::ROPE_TOKENS);
        let pos = rope_positions(pos, t)?;
        unsafe {
            let qg = qg.reshape(&[t, hq, 2 * d])?.tile::<bf16>()?;
            let kv = kv.reshape(&[t, 2 * hk, d])?.tile::<bf16>()?;
            let mut qo = q.reshape(&[t, hq, d])?.tile::<bf16>()?;
            let mut ko = k.reshape(&[t, hk, d])?.tile::<bf16>()?;
            let (pos, qw, kw) = (pos.tile::<i32>()?, q_norm.tile::<bf16>()?, k_norm.tile::<bf16>()?);
            let inv = self.inv_freq.tile::<f32>()?;
            attn::qk_norm_rope(&mut qo, &mut ko, &qg, &kv, &pos, &qw, &kw, &inv, self.eps)?
                .execute(cx.exec)
                .map(|_| ()).map_err(launch_err("qk_norm_rope"))
        }
    }
}
