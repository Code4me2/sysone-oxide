//! GatedDeltaNet mixer kernels over packed segment trees (DESIGN.md §3, §6.1, §6.3, §6.4). Owner: W3a.
//!
//! The GDN half of a linear-attention layer, after the fused `[T, 12352]` input projection (`qkv 8192 | z 4096 |
//! a 32 | b 32`) and before `out_proj`:
//!
//! ```text
//! conv   = tree_conv1d_silu(qkv, conv_w)                  bf16 [T, 8192]
//! q,k,g,beta = gdn_prep(conv, a, b, A_log, dt_bias)       f32 [T,16,128] x2, f32 [T,32] x2
//! tinv,p = gdn_chunk_tp(q, k, g, beta)                   f32 [T,32,64] x2     (parallel over chunks)
//! o      = gdn_chunk_scan_fused(q, k, v, g, beta, tinv, p) bf16 [T, 4096]      (prefixes, then suffixes)
//! y      = gated_rmsnorm(o, z, norm_w)                    bf16 [T, 4096]
//! ```
//!
//! [`gdn_mixer`] fuses the first two steps differently: the conv also writes each q/k head's inverse L2 norm
//! ([`tree_conv1d_silu_inv`], into the plan), `gdn_gates` computes g and beta only, and the inverse and scan
//! kernels rebuild the normalised f32 q and k from the bf16 conv output ([`gdn_chunk_scan_conv`]) instead of
//! reading 16 KB per token of stored f32 rows.
//!
//! Numerics follow `torch_chunk_gated_delta_rule` (chunk 64, UT/WY form, fp32), with deliberate, mathematically
//! identical changes: fp32 matmuls run as split-TF32 ("3xTF32", ~2^-21 per product) because cuTile's plain f32
//! `mma` is emulated and spills; the decay to the chunk end uses a reverse scan instead of `cum_last - cum` (which
//! cancels catastrophically for strongly decaying heads); the unit lower-triangular inverse is computed blockwise
//! (forward substitution inside 16-row blocks, then block doubling, as fla's `solve_tril`); and the scan uses
//! `v_new = Tinv (v*beta - (k*beta*exp(cum)) S)` instead of the reference's `u - w S` with stored `u = Tinv v*beta`,
//! `w = Tinv k*beta*exp(cum)`, so no `[T, 4096]` f32 intermediate is written (P1). Measured against an f64 oracle,
//! these stay inside the error envelope of transformers' own f32 implementation (tests/kernels_gdn.rs).
//! [`GdnPrecision`] selects faster, less precise products (DESIGN §1 "Precision knobs").
//!
//! Segment structure comes from a [`GdnPlan`] (device tables built from the batch's [`Segment`]s). Grids depend
//! only on the plan's capacities, so a launch sequence can be captured once per T bucket and replayed after
//! [`GdnPlan::update`].
//!
//! Every launcher here is `unsafe`: it hands raw device addresses to the kernel, so the caller must keep every
//! buffer alive, and not written by anyone else, until the returned operation has completed on its stream.

pub mod simt;
pub mod tile;

use crate::contract::Segment;
use anyhow::{Result, bail, ensure};
use cutile::cuda_core::Stream;
use cutile::half::bf16;
use cutile::prelude::*;
use cuda_async::device_buffer::DevicePointer;
use cutile::tile_kernel::CompileOptions;
use tile::gdn_tile;

/// Chunk length of the chunked delta rule (`torch_chunk_gated_delta_rule`'s default, used by the reference).
pub const CHUNK: usize = 64;
pub const NUM_V_HEADS: usize = 32;
pub const NUM_K_HEADS: usize = 16;
pub const HEAD_DIM: usize = 128;
/// Conv channels: q (16×128) | k (16×128) | v (32×128).
pub const CONV_DIM: usize = 8192;
pub const CONV_KERNEL: usize = 4;
pub const KEY_DIM: usize = NUM_K_HEADS * HEAD_DIM;
pub const VALUE_DIM: usize = NUM_V_HEADS * HEAD_DIM;
/// One prefix's recurrent state, `[32, 128, 128]` f32 elements.
pub const STATE_ELEMS: usize = NUM_V_HEADS * HEAD_DIM * HEAD_DIM;

/// Which implementation runs the chunked delta rule (the conv, gate and norm kernels are cuTile in both).
///
/// `Simt` is the cuda-oxide path (`simt.rs`, docs/G1.md): tensor-core `mma.sync` with split-bf16 operands,
/// recomputing the WY operands inside the scan instead of storing them. `Tile` is the cuTile path, kept as the
/// fallback and as the test reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GdnBackend {
    #[default]
    Tile,
    Simt,
}

impl GdnBackend {
    pub const ALL: [GdnBackend; 2] = [GdnBackend::Tile, GdnBackend::Simt];

    /// Default [`GdnOptions`] with this backend.
    pub fn options(self) -> GdnOptions {
        GdnOptions { backend: self, ..GdnOptions::default() }
    }

    pub fn name(self) -> &'static str {
        match self {
            GdnBackend::Tile => "tile",
            GdnBackend::Simt => "simt",
        }
    }
}

impl std::str::FromStr for GdnBackend {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "tile" => Ok(GdnBackend::Tile),
            "simt" => Ok(GdnBackend::Simt),
            _ => bail!("unknown GDN backend {s:?} (tile | simt)"),
        }
    }
}

/// Channel block of the conv kernel.
const CONV_BC: usize = 512;
/// Row block of `gdn_prep` and `gated_rmsnorm`.
const ROW_BLOCK: usize = 16;
/// Value-column block of the state scan; 128 / STATE_BV programs per (segment, head).
const STATE_BV: usize = 64;
/// Warps per CTA of the scan kernels (swept on GB10: 8 warps and BV 16/32/128 were no faster).
const WY_WARPS: i32 = 4;
const TINV_WARPS: i32 = 4;
const STATE_WARPS: i32 = 4;

/// A row-major 2-D device view: `rows × cols` elements, row `r` starting `r * row_stride` elements after `ptr`.
/// Used for the strided column slices of the fused projection output.
#[derive(Debug)]
pub struct DView<T> {
    pub ptr: DevicePointer<T>,
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
}

impl<T> Clone for DView<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DView<T> {}

impl<T: DType> DView<T> {
    /// The whole of a contiguous 2-D tensor.
    pub fn of(t: &Tensor<T>) -> Result<Self> {
        ensure!(t.shape().len() == 2 && t.is_contiguous(), "expected a contiguous 2-D tensor, got shape {:?}", t.shape());
        let (rows, cols) = (t.shape()[0] as usize, t.shape()[1] as usize);
        Ok(Self { ptr: t.device_pointer(), rows, cols, row_stride: cols })
    }

    /// Columns `c0 .. c0 + n` of this view.
    pub fn columns(self, c0: usize, n: usize) -> Result<Self> {
        ensure!(c0 + n <= self.cols, "columns {c0}..{} out of {}", c0 + n, self.cols);
        let dptr = self.ptr.cu_deviceptr() + (c0 * size_of::<T>()) as u64;
        // SAFETY: the address stays inside the same allocation (checked against cols above).
        let ptr = unsafe { DevicePointer::from_cu_deviceptr(dptr) };
        Ok(Self { ptr, cols: n, ..self })
    }

    fn check(&self, what: &str, rows: usize, cols: usize) -> Result<()> {
        ensure!(self.rows >= rows && self.cols == cols, "{what}: view is {}x{}, need {rows}x{cols}", self.rows, self.cols);
        ensure!(self.row_stride >= self.cols, "{what}: row stride {} < {} columns", self.row_stride, self.cols);
        ensure!(self.row_stride * rows <= i32::MAX as usize, "{what}: offsets overflow i32");
        Ok(())
    }
}

fn check_tensor<T: DType>(what: &str, t: &Tensor<T>, shape: &[usize]) -> Result<DevicePointer<T>> {
    let have: Vec<usize> = t.shape().iter().map(|&d| d as usize).collect();
    ensure!(t.is_contiguous(), "{what}: not contiguous");
    ensure!(have.iter().product::<usize>() >= shape.iter().product::<usize>(), "{what}: shape {have:?}, need {shape:?}");
    Ok(t.device_pointer())
}

/// Device-side segment metadata for the GDN kernels of one batch.
pub struct GdnPlan {
    /// Rows the per-token kernels process (`gdn_prep`, `gated_rmsnorm`): the batch's tokens, or its T bucket.
    pub tokens: usize,
    pub max_chunks: usize,
    pub max_segs: usize,
    /// Prefix-state scratch slots the plan needs (= number of prefix segments).
    pub num_prefixes: usize,
    chunks: Tensor<i32>,
    segs: Tensor<i32>,
    /// Per-(token, head) inverse L2 norms of the conv output's q and k heads, f32 `[tokens, 32]` (q heads, then k
    /// heads), written by [`tree_conv1d_silu_inv`] and read by [`gdn_chunk_scan_conv`]. Per-batch data, kept in
    /// the plan so that [`GdnScratch`] keeps its fields.
    qk_inv: Tensor<f32>,
}

/// Host-side tables of a [`GdnPlan`] (see `tile.rs` for the layouts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnTables {
    pub chunks: Vec<i32>,
    pub segs: Vec<i32>,
    pub num_prefixes: usize,
}

impl GdnTables {
    /// Number of 64-row chunks a segment list needs.
    pub fn num_chunks(segments: &[Segment]) -> usize {
        segments.iter().map(|s| (s.len as usize).div_ceil(CHUNK)).sum()
    }

    /// An upper bound on [`num_chunks`](Self::num_chunks) for any batch of `tokens` tokens in `segments`
    /// segments, for sizing a T bucket's plan.
    pub fn chunk_capacity(tokens: usize, segments: usize) -> usize {
        tokens.div_ceil(CHUNK) + segments
    }

    pub fn build(segments: &[Segment], tokens: usize, max_chunks: usize, max_segs: usize) -> Result<Self> {
        ensure!(segments.len() <= max_segs, "{} segments over the plan's {max_segs}", segments.len());
        let mut chunks = vec![0i32; max_chunks * 4];
        let mut segs = vec![0i32; max_segs * 4];
        for s in segs.chunks_mut(4) {
            s[2] = -1;
        }
        let mut slot_of = vec![-1i32; segments.len()];
        let mut num_prefixes = 0usize;
        let mut c = 0usize;
        for (i, s) in segments.iter().enumerate() {
            ensure!(s.len > 0 && s.end() as usize <= tokens, "segment {i} ({}..{}) outside {tokens} tokens", s.start, s.end());
            let (phase, slot, parent) = if s.is_prefix() {
                slot_of[i] = num_prefixes as i32;
                num_prefixes += 1;
                (0, slot_of[i], None)
            } else {
                let p = s.parent as usize;
                ensure!(p < i && segments[p].is_prefix(), "segment {i}: parent {p} is not an earlier prefix");
                (1, slot_of[p], Some(segments[p]))
            };
            segs[i * 4..i * 4 + 4].copy_from_slice(&[s.start as i32, s.len as i32, phase, slot]);
            for off in (0..s.len).step_by(CHUNK) {
                ensure!(c < max_chunks, "more than the plan's {max_chunks} chunks");
                let start = s.start + off;
                let len = (s.len - off).min(CHUNK as u32);
                let (prev_end, prev_avail) = match (off, parent) {
                    (0, Some(p)) => (p.end(), p.len.min(3)),
                    (0, None) => (start, 0),
                    _ => (start, 3),
                };
                chunks[c * 4..c * 4 + 4].copy_from_slice(&[start as i32, len as i32, prev_end as i32, prev_avail as i32]);
                c += 1;
            }
        }
        Ok(Self { chunks, segs, num_prefixes })
    }
}

impl GdnPlan {
    /// A plan sized exactly for `segments` (no spare capacity).
    pub fn new(segments: &[Segment], tokens: usize, stream: &Arc<Stream>) -> Result<Self> {
        let max_chunks = GdnTables::num_chunks(segments).max(1);
        Self::with_capacity(segments, tokens, max_chunks, segments.len().max(1), stream)
    }

    pub fn with_capacity(
        segments: &[Segment],
        tokens: usize,
        max_chunks: usize,
        max_segs: usize,
        stream: &Arc<Stream>,
    ) -> Result<Self> {
        ensure!(tokens > 0 && tokens * VALUE_DIM <= i32::MAX as usize, "tokens {tokens} out of range");
        let t = GdnTables::build(segments, tokens, max_chunks, max_segs)?;
        let chunks = api::copy_host_vec_to_device(&Arc::new(t.chunks)).sync_on(stream)?;
        let segs = api::copy_host_vec_to_device(&Arc::new(t.segs)).sync_on(stream)?;
        let qk_inv = api::zeros::<f32>(&[tokens, 2 * NUM_K_HEADS]).sync_on(stream)?;
        Ok(Self { tokens, max_chunks, max_segs, num_prefixes: t.num_prefixes, chunks, segs, qk_inv })
    }

    /// Rewrite the tables in place for another batch with the same capacities (for graph replay).
    ///
    /// The launchers check `num_prefixes <= scratch.slots` when they build their operations, which a replayed graph
    /// does not repeat: size the scratch of a captured mixer with [`GdnScratch::for_plan`] (one slot per segment the
    /// plan can hold). If a replay still sees more prefixes than slots, the SIMT path writes NaN over the affected
    /// segments' output (it never leaves stale output); the cuTile scan does not detect it.
    pub fn update(&mut self, segments: &[Segment], stream: &Arc<Stream>) -> Result<()> {
        let t = GdnTables::build(segments, self.tokens, self.max_chunks, self.max_segs)?;
        let chunks = api::copy_host_vec_to_device(&Arc::new(t.chunks)).sync_on(stream)?;
        let segs = api::copy_host_vec_to_device(&Arc::new(t.segs)).sync_on(stream)?;
        api::memcpy(&mut self.chunks, &chunks).sync_on(stream)?;
        api::memcpy(&mut self.segs, &segs).sync_on(stream)?;
        self.num_prefixes = t.num_prefixes;
        Ok(())
    }
}

/// Per-batch intermediates of the GDN mixer, sized for `tokens` rows and `slots` prefix states.
pub struct GdnScratch {
    pub tokens: usize,
    pub slots: usize,
    /// conv1d+SiLU output, bf16 `[T, 8192]` (q | k | v).
    pub conv: Tensor<bf16>,
    /// L2-normalised (q also scaled) queries and keys, f32 `[T, 16 * 128]`, written by [`gdn_prep`] or
    /// [`tree_conv1d_silu_qk`]. [`gdn_mixer`] neither writes nor reads them (its scan rebuilds q and k from the
    /// bf16 conv output).
    pub q: Tensor<f32>,
    pub k: Tensor<f32>,
    /// f32 `[T, 32]`.
    pub g: Tensor<f32>,
    pub beta: Tensor<f32>,
    /// Stored WY operands of the Phase 1 scan ([`gdn_chunk_scan_w3a`] only): WY-transformed keys and values,
    /// decayed queries and keys, f32 `[T, 32 * 128]` each when allocated by
    /// [`with_w3a_buffers`](Self::with_w3a_buffers). [`new`](Self::new) allocates them as `[1, 32 * 128]`
    /// placeholders: the default scan recomputes them per chunk (64 KB per token saved).
    pub w: Tensor<f32>,
    pub u: Tensor<f32>,
    pub qd: Tensor<f32>,
    pub kd: Tensor<f32>,
    /// Masked, decayed intra-chunk attention, f32 `[T, 32 * 64]` (row = query, columns = the chunk's keys).
    pub p: Tensor<f32>,
    /// Inverse of each chunk's UT system, f32 `[T, 32 * 64]` (same layout as `p`).
    pub tinv: Tensor<f32>,
    /// Delta-rule output, bf16 `[T, 32 * 128]`.
    pub core: Tensor<bf16>,
    /// Prefix final states, f32 `[slots, 32, 128, 128]`.
    pub state: Tensor<f32>,
}

impl GdnScratch {
    /// Scratch for the cuTile backend (every buffer at full size).
    pub fn new(tokens: usize, slots: usize, stream: &Arc<Stream>) -> Result<Self> {
        Self::alloc(tokens, slots, tokens, 1, stream)
    }

    /// Like [`new`](Self::new), with full-size `w, u, qd, kd` for [`gdn_chunk_scan_w3a`].
    pub fn with_w3a_buffers(tokens: usize, slots: usize, stream: &Arc<Stream>) -> Result<Self> {
        Self::alloc(tokens, slots, tokens, tokens, stream)
    }

    /// Scratch for `backend`'s mixer. `Tile` is [`new`](Self::new). The SIMT backend does not use q, k or p either
    /// (it normalises q, k and recomputes P inside the scan), so those are also one-row placeholders.
    pub fn for_backend(backend: GdnBackend, tokens: usize, slots: usize, stream: &Arc<Stream>) -> Result<Self> {
        match backend {
            GdnBackend::Tile => Self::new(tokens, slots, stream),
            GdnBackend::Simt => Self::alloc(tokens, slots, 1, 1, stream),
        }
    }

    /// `qkp_rows`: rows of q, k and p (`tokens`, or 1 when unused); `wy_rows`: rows of w, u, qd, kd.
    fn alloc(tokens: usize, slots: usize, qkp_rows: usize, wy_rows: usize, stream: &Arc<Stream>) -> Result<Self> {
        let slots = slots.max(1);
        Ok(Self {
            tokens,
            slots,
            conv: api::zeros::<bf16>(&[tokens, CONV_DIM]).sync_on(stream)?,
            q: api::zeros::<f32>(&[qkp_rows, KEY_DIM]).sync_on(stream)?,
            k: api::zeros::<f32>(&[qkp_rows, KEY_DIM]).sync_on(stream)?,
            g: api::zeros::<f32>(&[tokens, NUM_V_HEADS]).sync_on(stream)?,
            beta: api::zeros::<f32>(&[tokens, NUM_V_HEADS]).sync_on(stream)?,
            w: api::zeros::<f32>(&[wy_rows, VALUE_DIM]).sync_on(stream)?,
            u: api::zeros::<f32>(&[wy_rows, VALUE_DIM]).sync_on(stream)?,
            qd: api::zeros::<f32>(&[wy_rows, VALUE_DIM]).sync_on(stream)?,
            kd: api::zeros::<f32>(&[wy_rows, VALUE_DIM]).sync_on(stream)?,
            p: api::zeros::<f32>(&[qkp_rows, NUM_V_HEADS * CHUNK]).sync_on(stream)?,
            tinv: api::zeros::<f32>(&[tokens, NUM_V_HEADS * CHUNK]).sync_on(stream)?,
            core: api::zeros::<bf16>(&[tokens, VALUE_DIM]).sync_on(stream)?,
            state: api::zeros::<f32>(&[slots, NUM_V_HEADS, HEAD_DIM, HEAD_DIM]).sync_on(stream)?,
        })
    }

    /// Scratch for every batch `plan` can describe (graph replay): one state slot per segment the plan can hold,
    /// since `GdnPlan::update` may raise the prefix count up to that.
    pub fn for_plan(backend: GdnBackend, plan: &GdnPlan, stream: &Arc<Stream>) -> Result<Self> {
        Self::for_backend(backend, plan.tokens, plan.max_segs, stream)
    }

    pub fn conv_view(&self) -> Result<DView<bf16>> {
        DView::of(&self.conv)
    }
}

/// The GDN weights of one linear-attention layer (bf16, as on disk).
pub struct GdnWeights<'a> {
    /// `conv1d.weight` `[8192, 1, 4]`.
    pub conv_w: &'a Tensor<bf16>,
    /// `A_log` `[32]`.
    pub a_log: &'a Tensor<bf16>,
    /// `dt_bias` `[32]`.
    pub dt_bias: &'a Tensor<bf16>,
    /// `norm.weight` `[128]`.
    pub norm_w: &'a Tensor<bf16>,
}

fn grid(x: usize, y: usize, z: usize) -> (u32, u32, u32) {
    (x as u32, y as u32, z as u32)
}

/// `out = silu(causal_conv1d(x, conv_w))` over the plan's segment tree. x, out: `[T, 8192]` bf16 views.
///
/// # Safety
/// Every buffer must stay alive and unaliased by other writers until the returned operation completes.
pub unsafe fn tree_conv1d_silu(
    plan: &GdnPlan,
    x: DView<bf16>,
    conv_w: &Tensor<bf16>,
    out: DView<bf16>,
) -> Result<BoxedDeviceOp<()>> {
    unsafe { conv_impl(plan, x, conv_w, out, ConvQk::None) }
}

/// [`tree_conv1d_silu`] that also writes the L2-normalised (q also scaled) f32 q and k into `scratch.{q,k}`,
/// exactly as [`gdn_prep`] computes them from the conv output; pair it with [`gdn_gates`]. Saves reading the
/// q and k columns back.
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn tree_conv1d_silu_qk(
    plan: &GdnPlan,
    x: DView<bf16>,
    conv_w: &Tensor<bf16>,
    out: DView<bf16>,
    scratch: &GdnScratch,
) -> Result<BoxedDeviceOp<()>> {
    ensure!(plan.tokens <= scratch.tokens, "plan has {} tokens, scratch {}", plan.tokens, scratch.tokens);
    unsafe { conv_impl(plan, x, conv_w, out, ConvQk::F32(scratch)) }
}

/// [`tree_conv1d_silu`] that also writes the per-(token, head) inverse L2 norms of the q and k heads into the
/// plan, for [`gdn_chunk_scan_conv`], which rebuilds the normalised f32 q and k from `out` with the same
/// arithmetic (bit-identical to [`tree_conv1d_silu_qk`]'s) instead of reading 16 KB per token of f32 rows.
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn tree_conv1d_silu_inv(
    plan: &GdnPlan,
    x: DView<bf16>,
    conv_w: &Tensor<bf16>,
    out: DView<bf16>,
) -> Result<BoxedDeviceOp<()>> {
    unsafe { conv_impl(plan, x, conv_w, out, ConvQk::InvNorms) }
}

/// What the conv writes besides its bf16 output.
enum ConvQk<'a> {
    None,
    /// Normalised f32 q and k into the scratch (`gdn_prep` layout).
    F32(&'a GdnScratch),
    /// Inverse norms into the plan.
    InvNorms,
}

unsafe fn conv_impl(
    plan: &GdnPlan,
    x: DView<bf16>,
    conv_w: &Tensor<bf16>,
    out: DView<bf16>,
    qk: ConvQk<'_>,
) -> Result<BoxedDeviceOp<()>> {
    x.check("conv input", plan.tokens, CONV_DIM)?;
    out.check("conv output", plan.tokens, CONV_DIM)?;
    let w = check_tensor("conv_w", conv_w, &[CONV_DIM, CONV_KERNEL])?;
    let inv = plan.qk_inv.device_pointer();
    let (q, k, mode) = match qk {
        ConvQk::F32(s) => (s.q.device_pointer(), s.k.device_pointer(), 1),
        // SAFETY: the q, k pointers are never dereferenced by the kernel in these modes.
        ConvQk::None | ConvQk::InvNorms => unsafe {
            let p = DevicePointer::<f32>::from_cu_deviceptr(out.ptr.cu_deviceptr());
            (p, p, if matches!(qk, ConvQk::None) { 0 } else { 2 })
        },
    };
    let op = unsafe {
        gdn_tile::tree_conv1d_silu(
            x.ptr,
            value(x.row_stride as i32),
            w,
            out.ptr,
            value(out.row_stride as i32),
            q,
            k,
            inv,
            plan.chunks.device_pointer(),
            value(plan.max_chunks as i32),
            value(plan.tokens as i32),
        )
    }
    .generics(vec![CONV_DIM.to_string(), CONV_BC.to_string(), (CONV_BC / HEAD_DIM).to_string(), mode.to_string()])
    .grid(grid(plan.max_chunks, CONV_DIM / CONV_BC, 1));
    Ok(op.map(|_| ()).boxed())
}

/// Split the conv output into normalised q, k (f32 `[T, 2048]`) and compute the gates g, beta (f32 `[T, 32]`)
/// into `scratch`. `a`, `b`: `[T, 32]` bf16 views of the fused projection output.
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_prep(
    plan: &GdnPlan,
    conv: DView<bf16>,
    a: DView<bf16>,
    b: DView<bf16>,
    w: &GdnWeights<'_>,
    scratch: &GdnScratch,
) -> Result<BoxedDeviceOp<()>> {
    unsafe { prep_impl(plan, conv, a, b, w, scratch, true) }
}

/// The gate half of [`gdn_prep`] (g, beta only), for use after [`tree_conv1d_silu_qk`].
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_gates(
    plan: &GdnPlan,
    conv: DView<bf16>,
    a: DView<bf16>,
    b: DView<bf16>,
    w: &GdnWeights<'_>,
    scratch: &GdnScratch,
) -> Result<BoxedDeviceOp<()>> {
    unsafe { prep_impl(plan, conv, a, b, w, scratch, false) }
}

unsafe fn prep_impl(
    plan: &GdnPlan,
    conv: DView<bf16>,
    a: DView<bf16>,
    b: DView<bf16>,
    w: &GdnWeights<'_>,
    scratch: &GdnScratch,
    qk: bool,
) -> Result<BoxedDeviceOp<()>> {
    let t = plan.tokens;
    ensure!(t <= scratch.tokens, "plan has {t} tokens, scratch {}", scratch.tokens);
    conv.check("conv", t, CONV_DIM)?;
    a.check("a", t, NUM_V_HEADS)?;
    b.check("b", t, NUM_V_HEADS)?;
    let a_log = check_tensor("A_log", w.a_log, &[NUM_V_HEADS])?;
    let dt_bias = check_tensor("dt_bias", w.dt_bias, &[NUM_V_HEADS])?;
    let op = unsafe {
        gdn_tile::gdn_prep(
            conv.ptr,
            value(conv.row_stride as i32),
            a.ptr,
            value(a.row_stride as i32),
            b.ptr,
            value(b.row_stride as i32),
            a_log,
            dt_bias,
            scratch.q.device_pointer(),
            scratch.k.device_pointer(),
            scratch.g.device_pointer(),
            scratch.beta.device_pointer(),
            value(t as i32),
        )
    }
    .generics(vec![ROW_BLOCK.to_string(), (qk as i32).to_string()])
    .grid(grid(t.div_ceil(ROW_BLOCK), if qk { NUM_K_HEADS } else { 1 }, 1));
    Ok(op.map(|_| ()).boxed())
}

/// Arithmetic of the matrix products inside the chunked delta rule (DESIGN §1 "Precision knobs": the end-to-end
/// gate decides whether a faster variant is acceptable; the default is the precise one).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GdnPrecision {
    /// Split TF32 ("3xTF32", ~2^-21 relative per product): fp32-grade, within the tolerances of
    /// tests/kernels_gdn.rs.
    #[default]
    Fp32Split3,
    /// One TF32 product (10-bit mantissa operands, fp32 accumulation) everywhere.
    Tf32,
    /// bf16 operands (fp32 accumulation) in the sequential scan; TF32 for the chunk inverse and the intra-chunk
    /// attention, where bf16 would compound through the triangular solve.
    Bf16Operands,
    /// Split TF32 for every product that feeds the recurrent state (`(k*beta) k^T` and the chunk inverse,
    /// `k S`, `Tinv (..)`, `k^T v_new`) and one TF32 product for the output-only ones (`q k^T`, `q S`,
    /// `p v_new`): output errors do not compound across chunks.
    Fp32StateTf32Output,
}

impl GdnPrecision {
    /// `[element type, split, output element type, output split]` generics of the chunk-inverse kernel and of
    /// the scan kernel.
    fn generics(self) -> ([&'static str; 4], [&'static str; 4]) {
        match self {
            Self::Fp32Split3 => (["tf32", "3", "tf32", "3"], ["tf32", "3", "tf32", "3"]),
            Self::Tf32 => (["tf32", "1", "tf32", "1"], ["tf32", "1", "tf32", "1"]),
            Self::Bf16Operands => (["tf32", "1", "tf32", "1"], ["bf16", "1", "bf16", "1"]),
            Self::Fp32StateTf32Output => (["tf32", "3", "tf32", "1"], ["tf32", "3", "tf32", "1"]),
        }
    }
}

/// Tuning and precision options of [`gdn_chunk_scan_with`] / [`gdn_mixer_with`]. `Default` is what
/// [`gdn_chunk_scan`] and [`gdn_mixer`] use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnOptions {
    /// Implementation of the delta rule in [`gdn_mixer_with`]. The other fields tune the cuTile (`Tile`) kernels
    /// only; the SIMT kernels have one configuration. [`gdn_chunk_scan_with`] is always cuTile.
    pub backend: GdnBackend,
    pub precision: GdnPrecision,
    /// Value columns per program of the prefix scan (phase 0: few, long segments; smaller blocks give more
    /// programs). One of 16, 32, 64, 128.
    pub prefix_bv: usize,
    /// Value columns per program of the suffix scan (phase 1: many short segments).
    pub suffix_bv: usize,
    /// Warps per CTA of the prefix and suffix scans.
    pub prefix_warps: i32,
    pub suffix_warps: i32,
    /// Warps per CTA of the chunk-inverse kernel.
    pub tp_warps: i32,
    /// Block size of the chunk inverse solved by forward substitution before block doubling takes over (a
    /// power of two, 2..=64; 2 = doubling only, 64 = forward substitution only).
    pub tp_fsb: i32,
}

impl Default for GdnOptions {
    fn default() -> Self {
        Self {
            backend: GdnBackend::default(),
            precision: GdnPrecision::default(),
            prefix_bv: 32,
            suffix_bv: 32,
            prefix_warps: 8,
            suffix_warps: 8,
            tp_warps: 8,
            tp_fsb: 16,
        }
    }
}

/// The chunked gated delta rule over the plan's segment tree with default [`GdnOptions`]; see
/// [`gdn_chunk_scan_with`].
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_chunk_scan(
    plan: &GdnPlan,
    v: DView<bf16>,
    scratch: &GdnScratch,
    out: DView<bf16>,
) -> Result<DeviceOpVec<()>> {
    unsafe { gdn_chunk_scan_with(plan, v, scratch, out, &GdnOptions::default()) }
}

/// The chunked gated delta rule over the plan's segment tree, as four operations: the chunk inverse `Tinv` and
/// masked intra-chunk attention `p` (parallel over chunks, into `scratch.{tinv,p}`), an empty operation in the
/// slot of the Phase 1 WY launch (now fused into the scans), then the inter-chunk scan for prefixes (writing
/// their final states into `scratch.state`), then for suffixes (starting from their parent's state). Reads `scratch.{q,k,g,beta}` and v (`[T, 4096]` bf16 view, normally columns 4096.. of the
/// conv output); writes bf16 `[T, 4096]` into `out`.
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_chunk_scan_with(
    plan: &GdnPlan,
    v: DView<bf16>,
    scratch: &GdnScratch,
    out: DView<bf16>,
    opts: &GdnOptions,
) -> Result<DeviceOpVec<()>> {
    unsafe { scan_impl(plan, None, v, scratch, out, opts) }
}

/// [`gdn_chunk_scan_with`] for the mixer's conv output: `conv` is the full `[T, 8192]` bf16 view written by
/// [`tree_conv1d_silu_inv`] (same plan), whose v columns are the scan's values. Instead of reading f32 q and k
/// from `scratch.{q,k}` (16 KB per token, written by the conv or `gdn_prep`), it rebuilds them from the bf16 q, k
/// columns and the plan's inverse norms with the same arithmetic, so the result is bit-identical to
/// [`tree_conv1d_silu_qk`] followed by [`gdn_chunk_scan_with`]. `scratch.{q,k}` are neither read nor written.
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_chunk_scan_conv(
    plan: &GdnPlan,
    conv: DView<bf16>,
    scratch: &GdnScratch,
    out: DView<bf16>,
    opts: &GdnOptions,
) -> Result<DeviceOpVec<()>> {
    conv.check("conv", plan.tokens, CONV_DIM)?;
    unsafe { scan_impl(plan, Some(conv), conv.columns(2 * KEY_DIM, VALUE_DIM)?, scratch, out, opts) }
}

unsafe fn scan_impl(
    plan: &GdnPlan,
    conv: Option<DView<bf16>>,
    v: DView<bf16>,
    scratch: &GdnScratch,
    out: DView<bf16>,
    opts: &GdnOptions,
) -> Result<DeviceOpVec<()>> {
    let t = plan.tokens;
    ensure!(t <= scratch.tokens, "plan has {t} tokens, scratch {}", scratch.tokens);
    ensure!(plan.num_prefixes <= scratch.slots, "{} prefixes, scratch has {} state slots", plan.num_prefixes, scratch.slots);
    v.check("v", t, VALUE_DIM)?;
    out.check("scan output", t, VALUE_DIM)?;
    // q, k source: QKB 1 = rebuilt from the bf16 conv output and the plan's inverse norms; 0 = scratch.{q,k}
    // (the conv pointer is then unused; v's stands in).
    let (conv_ptr, conv_stride, qkb) = match conv {
        Some(c) => (c.ptr, c.row_stride, 1),
        None => (v.ptr, v.row_stride, 0),
    };
    ensure!(opts.tp_fsb >= 2 && opts.tp_fsb <= 64 && opts.tp_fsb.count_ones() == 1, "tp_fsb {} (power of two, 2..=64)", opts.tp_fsb);
    for bv in [opts.prefix_bv, opts.suffix_bv] {
        ensure!(matches!(bv, 16 | 32 | 64 | 128), "scan block of {bv} value columns (16, 32, 64 or 128)");
    }
    let sp = |t: &Tensor<f32>| t.device_pointer();
    let (tp_gen, scan_gen) = opts.precision.generics();
    let mut ops = DeviceOpVec::with_capacity(4);
    let tp = unsafe {
        gdn_tile::gdn_chunk_tp(
            sp(&scratch.q),
            sp(&scratch.k),
            conv_ptr,
            value(conv_stride as i32),
            sp(&plan.qk_inv),
            sp(&scratch.g),
            sp(&scratch.beta),
            sp(&scratch.tinv),
            sp(&scratch.p),
            plan.chunks.device_pointer(),
            value(plan.max_chunks as i32),
        )
    }
    .generics(tp_gen.iter().map(|g| g.to_string()).chain([opts.tp_fsb.to_string(), qkb.to_string()]).collect())
    .compile_options(CompileOptions::default().num_worker_warps_per_cta(opts.tp_warps))
    // Heads fastest: the two v-heads of a k-head (and the heads of one chunk) read the same q, k rows together.
    .grid(grid(NUM_V_HEADS, plan.max_chunks, 1));
    ops.push(tp.map(|_| ()));
    // The Phase 1 WY launch is fused into the scans; an empty op keeps the four-launch shape
    // (inverse, WY, prefix scan, suffix scan) that callers label and replay.
    ops.push(value(()).boxed());
    for (phase, bv, warps) in [(0i32, opts.prefix_bv, opts.prefix_warps), (1, opts.suffix_bv, opts.suffix_warps)] {
        let scan = unsafe {
            gdn_tile::gdn_chunk_scan_fused(
                sp(&scratch.q),
                sp(&scratch.k),
                conv_ptr,
                value(conv_stride as i32),
                sp(&plan.qk_inv),
                v.ptr,
                value(v.row_stride as i32),
                sp(&scratch.g),
                sp(&scratch.beta),
                sp(&scratch.tinv),
                sp(&scratch.p),
                sp(&scratch.state),
                out.ptr,
                value(out.row_stride as i32),
                plan.segs.device_pointer(),
                value(plan.max_segs as i32),
                value(scratch.slots as i32),
                value(phase),
            )
        }
        .generics(scan_gen.iter().map(|g| g.to_string()).chain([bv.to_string(), qkb.to_string()]).collect())
        .compile_options(CompileOptions::default().num_worker_warps_per_cta(warps))
        // Value blocks fastest, then heads: the CTAs that read one chunk's q, k, Tinv and p (the value blocks of
        // a head, and the two v-heads of a k-head) are launched together and share those reads through L2.
        .grid(grid(HEAD_DIM / bv, NUM_V_HEADS, plan.max_segs));
        ops.push(scan.map(|_| ()));
    }
    Ok(ops)
}

/// The Phase 1 (W3a) four-launch scan: inverse by forward substitution, stored WY intermediates
/// (`scratch.{w,u,qd,kd,p}`, which need [`GdnScratch::with_w3a_buffers`]), then the state scan. Kept for
/// comparison in `bench_gdn`.
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_chunk_scan_w3a(
    plan: &GdnPlan,
    v: DView<bf16>,
    scratch: &GdnScratch,
    out: DView<bf16>,
) -> Result<DeviceOpVec<()>> {
    let t = plan.tokens;
    ensure!(t <= scratch.tokens, "plan has {t} tokens, scratch {}", scratch.tokens);
    ensure!(plan.num_prefixes <= scratch.slots, "{} prefixes, scratch has {} state slots", plan.num_prefixes, scratch.slots);
    v.check("v", t, VALUE_DIM)?;
    out.check("scan output", t, VALUE_DIM)?;
    ensure!(scratch.w.shape()[0] as usize >= t, "gdn_chunk_scan_w3a needs GdnScratch::with_w3a_buffers");
    let mut ops = DeviceOpVec::with_capacity(4);
    let sp = |t: &Tensor<f32>| t.device_pointer();
    let tinv = unsafe {
        gdn_tile::gdn_chunk_tinv(
            sp(&scratch.k),
            sp(&scratch.g),
            sp(&scratch.beta),
            sp(&scratch.tinv),
            plan.chunks.device_pointer(),
            value(plan.max_chunks as i32),
        )
    }
    .compile_options(CompileOptions::default().num_worker_warps_per_cta(TINV_WARPS))
    .grid(grid(plan.max_chunks, NUM_V_HEADS, 1));
    ops.push(tinv.map(|_| ()));
    let wy = unsafe {
        gdn_tile::gdn_chunk_wy(
            sp(&scratch.q),
            sp(&scratch.k),
            v.ptr,
            value(v.row_stride as i32),
            sp(&scratch.g),
            sp(&scratch.beta),
            sp(&scratch.tinv),
            sp(&scratch.w),
            sp(&scratch.u),
            sp(&scratch.qd),
            sp(&scratch.kd),
            sp(&scratch.p),
            plan.chunks.device_pointer(),
            value(plan.max_chunks as i32),
        )
    }
    .compile_options(CompileOptions::default().num_worker_warps_per_cta(WY_WARPS))
    .grid(grid(plan.max_chunks, NUM_V_HEADS, 1));
    ops.push(wy.map(|_| ()));
    for phase in [0i32, 1] {
        let scan = unsafe {
            gdn_tile::gdn_chunk_state(
                sp(&scratch.w),
                sp(&scratch.u),
                sp(&scratch.qd),
                sp(&scratch.kd),
                sp(&scratch.p),
                sp(&scratch.g),
                sp(&scratch.state),
                out.ptr,
                value(out.row_stride as i32),
                plan.segs.device_pointer(),
                value(plan.max_segs as i32),
                value(scratch.slots as i32),
                value(phase),
            )
        }
        .generics(vec![STATE_BV.to_string()])
        .compile_options(CompileOptions::default().num_worker_warps_per_cta(STATE_WARPS))
        .grid(grid(plan.max_segs, NUM_V_HEADS, HEAD_DIM / STATE_BV));
        ops.push(scan.map(|_| ()));
    }
    Ok(ops)
}

/// `out = gated_rmsnorm(x, z) * norm_w` per 128-wide head. x, z, out: `[T, 4096]` bf16 views.
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gated_rmsnorm(
    plan: &GdnPlan,
    x: DView<bf16>,
    z: DView<bf16>,
    norm_w: &Tensor<bf16>,
    eps: f32,
    out: DView<bf16>,
) -> Result<BoxedDeviceOp<()>> {
    let t = plan.tokens;
    x.check("norm input", t, VALUE_DIM)?;
    z.check("z", t, VALUE_DIM)?;
    out.check("norm output", t, VALUE_DIM)?;
    let w = check_tensor("norm_w", norm_w, &[HEAD_DIM])?;
    let op = unsafe {
        gdn_tile::gated_rmsnorm(
            x.ptr,
            value(x.row_stride as i32),
            z.ptr,
            value(z.row_stride as i32),
            w,
            out.ptr,
            value(out.row_stride as i32),
            value(t as i32),
            value(eps),
        )
    }
    .generics(vec![ROW_BLOCK.to_string()])
    .grid(grid(t.div_ceil(ROW_BLOCK), NUM_V_HEADS, 1));
    Ok(op.map(|_| ()).boxed())
}

/// The whole GDN mixer between the fused input projection and `out_proj`: `proj` is the `[T, 12352]` bf16 GEMM
/// output (`qkv | z | a | b`), `out` the `[T, 4096]` bf16 input of `out_proj`. Launches
/// [`tree_conv1d_silu_inv`], [`gdn_gates`], [`gdn_chunk_scan_conv`] and [`gated_rmsnorm`].
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_mixer(
    plan: &GdnPlan,
    proj: DView<bf16>,
    w: &GdnWeights<'_>,
    eps: f32,
    scratch: &GdnScratch,
    out: DView<bf16>,
) -> Result<DeviceOpVec<()>> {
    unsafe { gdn_mixer_with(plan, proj, w, eps, scratch, out, &GdnOptions::default()) }
}

/// [`gdn_mixer`] with explicit [`GdnOptions`] (including the [`GdnBackend`]).
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_mixer_with(
    plan: &GdnPlan,
    proj: DView<bf16>,
    w: &GdnWeights<'_>,
    eps: f32,
    scratch: &GdnScratch,
    out: DView<bf16>,
    opts: &GdnOptions,
) -> Result<DeviceOpVec<()>> {
    if proj.cols != CONV_DIM + VALUE_DIM + 2 * NUM_V_HEADS {
        bail!("fused projection has {} columns, expected {}", proj.cols, CONV_DIM + VALUE_DIM + 2 * NUM_V_HEADS);
    }
    let qkv = proj.columns(0, CONV_DIM)?;
    let z = proj.columns(CONV_DIM, VALUE_DIM)?;
    let a = proj.columns(CONV_DIM + VALUE_DIM, NUM_V_HEADS)?;
    let b = proj.columns(CONV_DIM + VALUE_DIM + NUM_V_HEADS, NUM_V_HEADS)?;
    let conv = scratch.conv_view()?;
    let core = DView::of(&scratch.core)?;
    let mut ops = DeviceOpVec::with_capacity(6);
    unsafe {
        match opts.backend {
            GdnBackend::Tile => {
                ops.push(tree_conv1d_silu_inv(plan, qkv, w.conv_w, conv)?);
                ops.push(gdn_gates(plan, conv, a, b, w, scratch)?);
                ops.push(gdn_chunk_scan_conv(plan, conv, scratch, core, opts)?.map(|_| ()));
            }
            // The SIMT scan normalises q and k itself (no inverse norms needed) and has its own gates kernel.
            GdnBackend::Simt => {
                ops.push(tree_conv1d_silu(plan, qkv, w.conv_w, conv)?);
                ops.push(simt::gates(plan, a, b, w, scratch)?);
                ops.push(simt::chunk_scan(plan, conv, scratch, core)?.map(|_| ()).boxed());
            }
        }
        ops.push(gated_rmsnorm(plan, core, z, w.norm_w, eps, out)?);
    }
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::DecisionBatch;

    #[test]
    fn tables_follow_the_tree() {
        let mut b = DecisionBatch::default();
        let prefix: Vec<u32> = (0..130).collect();
        b.push_request(&prefix, &[(&[1, 2], 2), (&[3; 70], 2)]).unwrap();
        b.push_request(&[9, 9], &[(&[4], 2)]).unwrap();
        let t = GdnTables::build(&b.segments, b.num_tokens(), 8, 6).unwrap();
        assert_eq!(t.num_prefixes, 2);
        // prefix: 3 chunks; suffix 1: 1; suffix 2: 2; prefix 2: 1; its suffix: 1 → 8.
        assert_eq!(GdnTables::num_chunks(&b.segments), 8);
        let chunk = |c: usize| &t.chunks[c * 4..c * 4 + 4];
        assert_eq!(chunk(0), [0, 64, 0, 0]);
        assert_eq!(chunk(1), [64, 64, 64, 3]);
        assert_eq!(chunk(2), [128, 2, 128, 3]);
        assert_eq!(chunk(3), [130, 2, 130, 3]); // first suffix: taps from the prefix's tail
        assert_eq!(chunk(4), [132, 64, 130, 3]);
        assert_eq!(chunk(5), [196, 6, 196, 3]);
        assert_eq!(chunk(6), [202, 2, 202, 0]);
        assert_eq!(chunk(7), [204, 1, 204, 2]); // a 2-token prefix offers only 2 taps
        let seg = |s: usize| &t.segs[s * 4..s * 4 + 4];
        assert_eq!(seg(0), [0, 130, 0, 0]);
        assert_eq!(seg(2), [132, 70, 1, 0]);
        assert_eq!(seg(4), [204, 1, 1, 1]);
        assert_eq!(seg(5), [0, 0, -1, 0]);
        assert!(GdnTables::build(&b.segments, b.num_tokens(), 7, 6).is_err());
        assert!(GdnTables::num_chunks(&b.segments) <= GdnTables::chunk_capacity(b.num_tokens(), b.segments.len()));
    }
}

