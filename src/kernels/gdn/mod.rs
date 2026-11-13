//! GatedDeltaNet mixer kernels over packed segment trees (DESIGN.md §3, §6.1, §6.3, §6.4). Owner: W3a.
//!
//! The GDN half of a linear-attention layer, after the fused `[T, 12352]` input projection (`qkv 8192 | z 4096 |
//! a 32 | b 32`) and before `out_proj`:
//!
//! ```text
//! conv   = tree_conv1d_silu(qkv, conv_w)                  bf16 [T, 8192]
//! q,k,g,beta = gdn_prep(conv, a, b, A_log, dt_bias)       f32 [T,16,128] x2, f32 [T,32] x2
//! tinv   = gdn_chunk_tinv(k, g, beta)                     f32 [T,32,64]        (parallel over chunks)
//! w,u,qd,kd,p = gdn_chunk_wy(q, k, v=conv[:, 4096:], ..)  f32 [T,32,128] x4, [T,32,64]   (parallel)
//! o      = gdn_chunk_state(w, u, qd, kd, p, g)            bf16 [T, 4096]       (prefixes, then suffixes)
//! y      = gated_rmsnorm(o, z, norm_w)                    bf16 [T, 4096]
//! ```
//!
//! Numerics follow `torch_chunk_gated_delta_rule` (chunk 64, UT/WY form, fp32), with two deliberate,
//! mathematically identical changes: fp32 matmuls run as split-TF32 ("3xTF32", ~2^-21 per product) because cuTile's
//! plain f32 `mma` is emulated and spills; and the decay to the chunk end uses a reverse scan instead of
//! `cum_last - cum` (which cancels catastrophically for strongly decaying heads). Measured against an f64 oracle,
//! both stay inside the error envelope of transformers' own f32 implementation (tests/kernels_gdn.rs).
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
        Ok(Self { tokens, max_chunks, max_segs, num_prefixes: t.num_prefixes, chunks, segs })
    }

    /// Rewrite the tables in place for another batch with the same capacities (for graph replay).
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
    /// L2-normalised (q also scaled) queries and keys, f32 `[T, 16 * 128]`.
    pub q: Tensor<f32>,
    pub k: Tensor<f32>,
    /// f32 `[T, 32]`.
    pub g: Tensor<f32>,
    pub beta: Tensor<f32>,
    /// Phase A of the scan, per v-head: WY-transformed keys and values, decayed queries and keys,
    /// f32 `[T, 32 * 128]` each.
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
    pub fn new(tokens: usize, slots: usize, stream: &Arc<Stream>) -> Result<Self> {
        let slots = slots.max(1);
        Ok(Self {
            tokens,
            slots,
            conv: api::zeros::<bf16>(&[tokens, CONV_DIM]).sync_on(stream)?,
            q: api::zeros::<f32>(&[tokens, KEY_DIM]).sync_on(stream)?,
            k: api::zeros::<f32>(&[tokens, KEY_DIM]).sync_on(stream)?,
            g: api::zeros::<f32>(&[tokens, NUM_V_HEADS]).sync_on(stream)?,
            beta: api::zeros::<f32>(&[tokens, NUM_V_HEADS]).sync_on(stream)?,
            w: api::zeros::<f32>(&[tokens, VALUE_DIM]).sync_on(stream)?,
            u: api::zeros::<f32>(&[tokens, VALUE_DIM]).sync_on(stream)?,
            qd: api::zeros::<f32>(&[tokens, VALUE_DIM]).sync_on(stream)?,
            kd: api::zeros::<f32>(&[tokens, VALUE_DIM]).sync_on(stream)?,
            p: api::zeros::<f32>(&[tokens, NUM_V_HEADS * CHUNK]).sync_on(stream)?,
            tinv: api::zeros::<f32>(&[tokens, NUM_V_HEADS * CHUNK]).sync_on(stream)?,
            core: api::zeros::<bf16>(&[tokens, VALUE_DIM]).sync_on(stream)?,
            state: api::zeros::<f32>(&[slots, NUM_V_HEADS, HEAD_DIM, HEAD_DIM]).sync_on(stream)?,
        })
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
    x.check("conv input", plan.tokens, CONV_DIM)?;
    out.check("conv output", plan.tokens, CONV_DIM)?;
    let w = check_tensor("conv_w", conv_w, &[CONV_DIM, CONV_KERNEL])?;
    let op = unsafe {
        gdn_tile::tree_conv1d_silu(
            x.ptr,
            value(x.row_stride as i32),
            w,
            out.ptr,
            value(out.row_stride as i32),
            plan.chunks.device_pointer(),
            value(plan.max_chunks as i32),
            value(plan.tokens as i32),
        )
    }
    .generics(vec![CONV_DIM.to_string(), CONV_BC.to_string()])
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
    .generics(vec![ROW_BLOCK.to_string()])
    .grid(grid(t.div_ceil(ROW_BLOCK), NUM_K_HEADS, 1));
    Ok(op.map(|_| ()).boxed())
}

/// The chunked gated delta rule over the plan's segment tree: WY transform (parallel over chunks), then the
/// inter-chunk scan for prefixes (writing their final states into `scratch.state`), then for suffixes (starting
/// from their parent's state). Reads `scratch.{q,k,g,beta}` and v (`[T, 4096]` bf16 view, normally columns
/// 4096.. of the conv output); writes bf16 `[T, 4096]` into `out`.
///
/// # Safety
/// As for [`tree_conv1d_silu`].
pub unsafe fn gdn_chunk_scan(
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
/// output (`qkv | z | a | b`), `out` the `[T, 4096]` bf16 input of `out_proj`.
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
        ops.push(tree_conv1d_silu(plan, qkv, w.conv_w, conv)?);
        ops.push(gdn_prep(plan, conv, a, b, w, scratch)?);
        ops.push(gdn_chunk_scan(plan, conv.columns(2 * KEY_DIM, VALUE_DIM)?, scratch, core)?.map(|_| ()));
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

