//! Full-attention layer kernels (DESIGN.md §3, §6.1, §6.3, §6.4). Owner: W3b.
//!
//! - [`qk_norm_rope`]: `q_norm`/`k_norm` (zero-centred RMSNorm per 256-wide head) then partial RoPE (first 64
//!   dims, rotate-half, θ = 1e7) at per-token positions. It reads the `q_proj` output in place, viewed
//!   `[T, 16, 512]` = per head (query 256 | gate 256), and the fused `[Wk; Wv]` output `[T, 8, 256]` (k heads 0..4,
//!   v heads 4..8).
//! - [`seg_attn`]: non-causal segment attention, GQA 16 q / 4 kv heads, head_dim 256, scale 1/16, softmax in
//!   fp32, epilogue `* sigmoid(gate)`. The mask is implicit in an [`AttnPlan`] built from the segment table.
//!
//! # Layouts (all row-major, bf16)
//!
//! | tensor | shape | notes |
//! |---|---|---|
//! | `qg` | `[T, 16, 512]` | `q_proj` output; gate is read in place by `seg_attn` |
//! | `kv` | `[T, 8, 256]` | `[Wk; Wv]` output; v is read in place by `seg_attn` |
//! | `q` | `[T, 16, 256]` | normed + roped query (written by `qk_norm_rope`) |
//! | `k` | `[T, 4, 256]` | normed + roped key (written by `qk_norm_rope`) |
//! | `out` | `[T, 16, 256]` = `[T, 4096]` | gated attention output, the `o_proj` input |
//!
//! # Reference rounding points (transformers 5.19.0 `Qwen3_5Attention.forward`)
//!
//! - `q_norm`/`k_norm`: fp32 statistics, `* (1 + w)`, rounded to bf16.
//! - RoPE: `cos`/`sin` are computed in fp32 from `pos * inv_freq` (fp32 product) and cast to bf16. Then
//!   `(x * cos) + (rotate_half(x) * sin)` is evaluated on bf16 tensors: each product and the sum are rounded.
//! - Attention: qev runs `sdpa`, which for head_dim 256 with a mask dispatches to the MATH backend on
//!   GB10 (flash and mem-efficient refuse it), i.e. fp32 attention with only the output rounded to bf16. The
//!   kernel keeps scores, softmax and P in fp32 (P enters the bf16 MMA as a hi + lo pair) and rounds the
//!   output to bf16.
//! - Gate: `attn_output * torch.sigmoid(gate)` on bf16 tensors: `sigmoid(gate)` is rounded to bf16, then the
//!   product is rounded.

use crate::contract::Segment;
use anyhow::{Result, ensure};
use cuda_async::device_buffer::DevicePointer;
use cutile::half::bf16;
use cutile::prelude::*;

pub const NUM_Q_HEADS: usize = 16;
pub const NUM_KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 256;
/// Rotary dims (`head_dim * partial_rotary_factor`).
pub const ROTARY_DIM: usize = 64;
pub const ROPE_THETA: f64 = 1e7;
/// `head_dim ** -0.5`.
pub const ATTN_SCALE: f32 = 1.0 / 16.0;

/// `inv_freq = 1 / theta ** (arange(0, 64, 2) / 64)` in fp32, as `Qwen3_5TextRotaryEmbedding` computes it
/// (torch CPU fp32 `pow` and division). The bit patterns are checked against torch 2.13 in the tests.
pub fn rope_inv_freq() -> [f32; ROTARY_DIM / 2] {
    let mut out = [0f32; ROTARY_DIM / 2];
    for (i, o) in out.iter_mut().enumerate() {
        let e = (2 * i) as f32 / ROTARY_DIM as f32;
        *o = 1.0f32 / (ROPE_THETA as f32).powf(e);
    }
    out
}

/// RoPE position of every token of the packed stream: `pos0 + offset` within its segment.
pub fn positions(segments: &[Segment]) -> Vec<i32> {
    let mut pos = Vec::with_capacity(segments.last().map_or(0, |s| s.end() as usize));
    for s in segments {
        pos.extend((0..s.len).map(|i| (s.pos0 + i) as i32));
    }
    pos
}

#[cutile::module]
pub mod qk_norm_rope_module {
    use cutile::core::*;

    /// Zero-centred RMSNorm of `H` heads (`x` is `[H, D]` fp32) and RoPE on the first `2 * R2` dims.
    /// `x_lo`/`x_hi` are dims `0..R2` and `R2..2*R2` of the same heads, loaded separately; they go through
    /// the same arithmetic as `x`, so they equal the corresponding slices of the normed `x` bit for bit.
    fn norm_rope<const H: i32, const D: i32, const R2: i32, const NC: i32>(
        x: Tile<f32, { [H, D] }>,
        x_lo: Tile<f32, { [H, R2] }>,
        x_hi: Tile<f32, { [H, R2] }>,
        opw: Tile<f32, { [1, D] }>,
        opw_lo: Tile<f32, { [1, R2] }>,
        opw_hi: Tile<f32, { [1, R2] }>,
        cos_b: Tile<f32, { [1, R2] }>,
        sin_b: Tile<f32, { [1, R2] }>,
        eps: f32,
        s3: Shape<{ [H, NC, R2] }>,
    ) -> Tile<bf16, { [H, D] }> {
        let ss: Tile<f32, { [H] }> = reduce_sum(x * x, 1i32);
        let d: f32 = convert_scalar(D);
        let mean: Tile<f32, { [H] }> = true_div(ss, broadcast_scalar(d, const_shape![H]));
        let inv: Tile<f32, { [H] }> = rsqrt(mean + broadcast_scalar(eps, const_shape![H]), ftz::Disabled);
        let inv: Tile<f32, { [H, 1] }> = inv.reshape(const_shape![H, 1]);

        let n: Tile<f32, { [H, D] }> =
            (x * inv.broadcast(const_shape![H, D])) * opw.broadcast(const_shape![H, D]);
        let n: Tile<bf16, { [H, D] }> = convert_tile(n);
        let n: Tile<f32, { [H, D] }> = convert_tile(n);
        let n_lo: Tile<f32, { [H, R2] }> =
            (x_lo * inv.broadcast(const_shape![H, R2])) * opw_lo.broadcast(const_shape![H, R2]);
        let n_lo: Tile<bf16, { [H, R2] }> = convert_tile(n_lo);
        let n_lo: Tile<f32, { [H, R2] }> = convert_tile(n_lo);
        let n_hi: Tile<f32, { [H, R2] }> =
            (x_hi * inv.broadcast(const_shape![H, R2])) * opw_hi.broadcast(const_shape![H, R2]);
        let n_hi: Tile<bf16, { [H, R2] }> = convert_tile(n_hi);
        let n_hi: Tile<f32, { [H, R2] }> = convert_tile(n_hi);

        // View each head as NC chunks of R2 dims. Chunk 0 rotates with -chunk1, chunk 1 with chunk 0, the
        // rest pass through (cos = 1, sin = 0 gives back n exactly).
        let n3: Tile<f32, { [H, NC, R2] }> = n.reshape(s3);
        let chunk: Tile<i32, { [NC] }> = iota(const_shape![NC]);
        let chunk: Tile<i32, { [H, NC, R2] }> = chunk.reshape(const_shape![1, NC, 1]).broadcast(s3);
        let zero_i: Tile<i32, { [H, NC, R2] }> = constant(0i32, s3);
        let one_i: Tile<i32, { [H, NC, R2] }> = constant(1i32, s3);
        let two_i: Tile<i32, { [H, NC, R2] }> = constant(2i32, s3);
        let is0: Tile<bool, { [H, NC, R2] }> = eq_tile(chunk, zero_i);
        let is1: Tile<bool, { [H, NC, R2] }> = eq_tile(chunk, one_i);
        let rot: Tile<bool, { [H, NC, R2] }> = lt_tile(chunk, two_i);
        let one_f: Tile<f32, { [H, NC, R2] }> = constant(1.0f32, s3);
        let zero_f: Tile<f32, { [H, NC, R2] }> = constant(0.0f32, s3);
        let cos3: Tile<f32, { [H, NC, R2] }> = cos_b.reshape(const_shape![1, 1, R2]).broadcast(s3);
        let sin3: Tile<f32, { [H, NC, R2] }> = sin_b.reshape(const_shape![1, 1, R2]).broadcast(s3);
        let lo3: Tile<f32, { [H, NC, R2] }> = n_lo.reshape(const_shape![H, 1, R2]).broadcast(s3);
        let hi3: Tile<f32, { [H, NC, R2] }> = n_hi.reshape(const_shape![H, 1, R2]).broadcast(s3);
        let c: Tile<f32, { [H, NC, R2] }> = select(rot, cos3, one_f);
        let s: Tile<f32, { [H, NC, R2] }> = select(rot, sin3, zero_f);
        let r: Tile<f32, { [H, NC, R2] }> = select(is0, negf(hi3), select(is1, lo3, zero_f));
        let a: Tile<bf16, { [H, NC, R2] }> = convert_tile(n3 * c);
        let b: Tile<bf16, { [H, NC, R2] }> = convert_tile(r * s);
        let a: Tile<f32, { [H, NC, R2] }> = convert_tile(a);
        let b: Tile<f32, { [H, NC, R2] }> = convert_tile(b);
        let y: Tile<bf16, { [H, NC, R2] }> = convert_tile(a + b);
        y.reshape(const_shape![H, D])
    }

    /// One token per tile block: q heads from `qg[t, :, :D]`, k heads from `kv[t, :HK, :]`.
    #[cutile::entry()]
    pub fn qk_norm_rope<const HQ: i32, const HK: i32, const D: i32, const R2: i32, const NC: i32>(
        q_out: &mut Tensor<bf16, { [1, HQ, D] }>,
        k_out: &mut Tensor<bf16, { [1, HK, D] }>,
        qg: &Tensor<bf16, { [-1, HQ, -1] }>,
        kv: &Tensor<bf16, { [-1, -1, D] }>,
        pos: &Tensor<i32, { [-1] }>,
        q_w: &Tensor<bf16, { [D] }>,
        k_w: &Tensor<bf16, { [D] }>,
        inv_freq: &Tensor<f32, { [R2] }>,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t: i32 = pid.0;

        // cos/sin of the fp32 angle `pos * inv_freq`, rounded to bf16 as `.to(x.dtype)` does.
        let tp: Tile<i32, { [1] }> = pos.partition(const_shape![1]).load([t]);
        let p: i32 = tile_to_scalar(tp.reshape(const_shape![]));
        let pf: f32 = convert_scalar(p);
        let invf: Tile<f32, { [R2] }> = inv_freq.partition(const_shape![R2]).load([0i32]);
        let ang: Tile<f32, { [R2] }> = invf * broadcast_scalar(pf, const_shape![R2]);
        let cb: Tile<bf16, { [R2] }> = convert_tile(cos(ang));
        let sb: Tile<bf16, { [R2] }> = convert_tile(sin(ang));
        let cos_b: Tile<f32, { [R2] }> = convert_tile(cb);
        let sin_b: Tile<f32, { [R2] }> = convert_tile(sb);
        let cos_b: Tile<f32, { [1, R2] }> = cos_b.reshape(const_shape![1, R2]);
        let sin_b: Tile<f32, { [1, R2] }> = sin_b.reshape(const_shape![1, R2]);

        // q heads.
        let qw: Tile<bf16, { [D] }> = q_w.partition(const_shape![D]).load([0i32]);
        let qw: Tile<f32, { [D] }> = convert_tile(qw);
        let qopw: Tile<f32, { [1, D] }> = constant(1.0f32, const_shape![1, D]) + qw.reshape(const_shape![1, D]);
        let qw_part: Partition<bf16, { [R2] }> = q_w.partition(const_shape![R2]);
        let qw_lo: Tile<f32, { [R2] }> = convert_tile(qw_part.load([0i32]));
        let qw_hi: Tile<f32, { [R2] }> = convert_tile(qw_part.load([1i32]));
        let qopw_lo: Tile<f32, { [1, R2] }> = constant(1.0f32, const_shape![1, R2]) + qw_lo.reshape(const_shape![1, R2]);
        let qopw_hi: Tile<f32, { [1, R2] }> = constant(1.0f32, const_shape![1, R2]) + qw_hi.reshape(const_shape![1, R2]);
        let qx: Tile<bf16, { [1, HQ, D] }> = qg.partition(const_shape![1, HQ, D]).load([t, 0i32, 0i32]);
        let qx: Tile<f32, { [1, HQ, D] }> = convert_tile(qx);
        let q_lo_part: Partition<bf16, { [1, HQ, R2] }> = qg.partition(const_shape![1, HQ, R2]);
        let qx_lo: Tile<bf16, { [1, HQ, R2] }> = q_lo_part.load([t, 0i32, 0i32]);
        let qx_hi: Tile<bf16, { [1, HQ, R2] }> = q_lo_part.load([t, 0i32, 1i32]);
        let qx_lo: Tile<f32, { [1, HQ, R2] }> = convert_tile(qx_lo);
        let qx_hi: Tile<f32, { [1, HQ, R2] }> = convert_tile(qx_hi);
        let qy: Tile<bf16, { [HQ, D] }> = norm_rope(
            qx.reshape(const_shape![HQ, D]),
            qx_lo.reshape(const_shape![HQ, R2]),
            qx_hi.reshape(const_shape![HQ, R2]),
            qopw,
            qopw_lo,
            qopw_hi,
            cos_b,
            sin_b,
            eps,
            const_shape![HQ, NC, R2],
        );
        q_out.store(qy.reshape(const_shape![1, HQ, D]));

        // k heads.
        let kw: Tile<bf16, { [D] }> = k_w.partition(const_shape![D]).load([0i32]);
        let kw: Tile<f32, { [D] }> = convert_tile(kw);
        let kopw: Tile<f32, { [1, D] }> = constant(1.0f32, const_shape![1, D]) + kw.reshape(const_shape![1, D]);
        let kw_part: Partition<bf16, { [R2] }> = k_w.partition(const_shape![R2]);
        let kw_lo: Tile<f32, { [R2] }> = convert_tile(kw_part.load([0i32]));
        let kw_hi: Tile<f32, { [R2] }> = convert_tile(kw_part.load([1i32]));
        let kopw_lo: Tile<f32, { [1, R2] }> = constant(1.0f32, const_shape![1, R2]) + kw_lo.reshape(const_shape![1, R2]);
        let kopw_hi: Tile<f32, { [1, R2] }> = constant(1.0f32, const_shape![1, R2]) + kw_hi.reshape(const_shape![1, R2]);
        let kx: Tile<bf16, { [1, HK, D] }> = kv.partition(const_shape![1, HK, D]).load([t, 0i32, 0i32]);
        let kx: Tile<f32, { [1, HK, D] }> = convert_tile(kx);
        let k_lo_part: Partition<bf16, { [1, HK, R2] }> = kv.partition(const_shape![1, HK, R2]);
        let kx_lo: Tile<bf16, { [1, HK, R2] }> = k_lo_part.load([t, 0i32, 0i32]);
        let kx_hi: Tile<bf16, { [1, HK, R2] }> = k_lo_part.load([t, 0i32, 1i32]);
        let kx_lo: Tile<f32, { [1, HK, R2] }> = convert_tile(kx_lo);
        let kx_hi: Tile<f32, { [1, HK, R2] }> = convert_tile(kx_hi);
        let ky: Tile<bf16, { [HK, D] }> = norm_rope(
            kx.reshape(const_shape![HK, D]),
            kx_lo.reshape(const_shape![HK, R2]),
            kx_hi.reshape(const_shape![HK, R2]),
            kopw,
            kopw_lo,
            kopw_hi,
            cos_b,
            sin_b,
            eps,
            const_shape![HK, NC, R2],
        );
        k_out.store(ky.reshape(const_shape![1, HK, D]));
    }
}

/// `q = rope(q_norm(qg[:, :, :256]))`, `k = rope(k_norm(kv[:, :4, :]))` at per-token `pos` (`[T]` i32).
/// `inv_freq` is [`rope_inv_freq`] on the device (`[32]` f32).
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope<'a>(
    q: &'a mut Tensor<bf16>,
    k: &'a mut Tensor<bf16>,
    qg: &'a Tensor<bf16>,
    kv: &'a Tensor<bf16>,
    pos: &'a Tensor<i32>,
    q_w: &'a Tensor<bf16>,
    k_w: &'a Tensor<bf16>,
    inv_freq: &'a Tensor<f32>,
    eps: f32,
) -> Result<impl DeviceOp + GraphNode + 'a> {
    let t = qg.shape()[0];
    let (hq, hk, d) = (NUM_Q_HEADS as i32, NUM_KV_HEADS as i32, HEAD_DIM as i32);
    ensure!(qg.shape() == [t, hq, 2 * d].as_slice(), "qg is {:?}, expected [T, 16, 512]", qg.shape());
    ensure!(kv.shape() == [t, 2 * hk, d].as_slice(), "kv is {:?}, expected [T, 8, 256]", kv.shape());
    ensure!(q.shape() == [t, hq, d].as_slice(), "q is {:?}, expected [T, 16, 256]", q.shape());
    ensure!(k.shape() == [t, hk, d].as_slice(), "k is {:?}, expected [T, 4, 256]", k.shape());
    ensure!(pos.shape() == [t].as_slice(), "pos is {:?}, expected [{t}]", pos.shape());
    ensure!(q_w.shape() == [d].as_slice() && k_w.shape() == [d].as_slice(), "norm weights must be [256]");
    ensure!(inv_freq.shape() == [ROTARY_DIM as i32 / 2].as_slice(), "inv_freq must be [32]");
    let r2 = ROTARY_DIM / 2;
    Ok(qk_norm_rope_module::qk_norm_rope(
        q.partition([1, NUM_Q_HEADS, HEAD_DIM]),
        k.partition([1, NUM_KV_HEADS, HEAD_DIM]),
        qg,
        kv,
        pos,
        q_w,
        k_w,
        inv_freq,
        eps,
    )
    .generics(vec![
        hq.to_string(),
        hk.to_string(),
        d.to_string(),
        r2.to_string(),
        (HEAD_DIM / r2).to_string(),
    ]))
}

// ------------------------------------------------------------------------------------------------------
// Segment attention
// ------------------------------------------------------------------------------------------------------

/// Number of i32 fields per work item.
pub const WORK_FIELDS: usize = 8;

/// The attention schedule for one packed batch, derived from its segment table.
///
/// One work item covers up to `bm` consecutive query rows **of one segment** and one kv head (the grid's
/// y axis). It attends to two non-causal key ranges:
/// - range A: the prefix: the segment itself for a prefix tile, the parent prefix for a suffix tile;
/// - range B: the suffix segment itself (empty for a prefix tile).
///
/// The two ranges are combined by the online-softmax (log-sum-exp) merge. Tiles start at the segment's
/// first row and key blocks start at each range's first key, so a row's arithmetic does not depend on the
/// rest of the batch: a branch gives bit-identical outputs alone or packed with others.
///
/// All suffix tiles of a request read the same prefix keys. Items are ordered by decreasing key count, which
/// keeps one request's suffix tiles adjacent in launch order, so the prefix keys they share are streamed
/// from L2 rather than DRAM.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AttnPlan {
    /// `[n, WORK_FIELDS]`: `q0, q_len, a0, a_len, b0, b_len, 0, 0`. `q_len == 0` marks an empty item.
    pub work: Vec<i32>,
}

impl AttnPlan {
    pub fn num_items(&self) -> usize {
        self.work.len() / WORK_FIELDS
    }

    /// Upper bound on [`num_items`](Self::num_items) for any batch of `t` tokens and `segments` segments,
    /// for sizing a captured graph's grid: every segment adds at most one partial tile.
    pub fn max_items(t: usize, segments: usize, bm: usize) -> usize {
        t.div_ceil(bm) + segments
    }

    /// Pad the work list with empty items up to `n` (a captured graph launches a fixed grid).
    pub fn pad_to(&mut self, n: usize) {
        assert!(n >= self.num_items(), "plan has {} items, more than {n}", self.num_items());
        self.work.resize(n * WORK_FIELDS, 0);
    }

    pub fn new(segments: &[Segment], bm: usize) -> Self {
        let mut items: Vec<[i32; WORK_FIELDS]> = Vec::new();
        for s in segments {
            let (a0, a_len, b0, b_len) = if s.is_prefix() {
                (s.start, s.len, 0, 0)
            } else {
                let p = segments[s.parent as usize];
                (p.start, p.len, s.start, s.len)
            };
            let mut q = s.start;
            while q < s.end() {
                let n = (s.end() - q).min(bm as u32);
                items.push([q, n, a0, a_len, b0, b_len, 0, 0].map(|v| v as i32));
                q += n;
            }
        }
        // Stable: equal-cost items keep stream order.
        items.sort_by_key(|w| std::cmp::Reverse(w[3] + w[5]));
        AttnPlan { work: items.into_iter().flatten().collect() }
    }
}

#[cutile::module]
pub mod seg_attn_module {
    use cutile::core::*;

    fn scalar_at(part: &Partition<i32, { [1] }>, i: i32) -> i32 {
        let t: Tile<i32, { [1] }> = load_view_tko(part, [i], ordering::Weak, scope::TileBlock, None, tma::Disabled);
        tile_to_scalar(t.reshape(const_shape![]))
    }

    /// Grid `(num_items, HK)`. See [`super::AttnPlan`] for the work items and the module docs for layouts.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn seg_attn<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const G: i32,
        const ME: i32, // BM * G
        const HQ: i32,
        const HK: i32,
        const LAT: i32,
        const PS: i32, // bf16 parts of P in the P @ V product: 1, 2 or 3
    >(
        out_ptr: *mut bf16,
        q_ptr: *mut bf16,
        k_ptr: *mut bf16,
        kv_ptr: *mut bf16,
        qg_ptr: *mut bf16,
        work_ptr: *mut i32,
        num_items: i32,
        qk_scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let item: i32 = pid.0;
        let kvh: i32 = pid.1;
        let tok: Token = new_token_unordered();

        let work_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[num_items * 8i32] };
        let work_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let work_tv: Tensor<i32, { [-1] }> =
            unsafe { make_tensor_view(pointer_to_tile(work_ptr), work_shape, work_strides, tok) };
        let work_part: Partition<i32, { [1] }> = work_tv.partition(const_shape![1]);
        let q0: i32 = scalar_at(&work_part, item * 8i32);
        let q_len: i32 = scalar_at(&work_part, item * 8i32 + 1i32);
        let a0: i32 = scalar_at(&work_part, item * 8i32 + 2i32);
        let a_len: i32 = scalar_at(&work_part, item * 8i32 + 3i32);
        let b0: i32 = scalar_at(&work_part, item * 8i32 + 4i32);
        let b_len: i32 = scalar_at(&work_part, item * 8i32 + 5i32);

        if q_len > 0i32 {
            // exp(x * scale) = exp2(x * scale * log2(e)).
            let scale_log2: f32 = qk_scale * 1.4426950408889634f32;
            let scale_t: Tile<f32, { [ME, BN] }> = scale_log2.broadcast(const_shape![ME, BN]);
            let scale_c: Tile<f32, { [ME, 1] }> = scale_log2.broadcast(const_shape![ME, 1]);
            let neg_big: Tile<f32, { [ME, BN] }> = constant(-1.0e30f32, const_shape![ME, BN]);
            let zero_mn: Tile<f32, { [ME, BN] }> = constant(0.0f32, const_shape![ME, BN]);
            let zero_md: Tile<f32, { [ME, D] }> = constant(0.0f32, const_shape![ME, D]);
            let offs_n: Tile<i32, { [BN] }> = iota(const_shape![BN]);
            let offs_n: Tile<i32, { [ME, BN] }> = offs_n.reshape(const_shape![1, BN]).broadcast(const_shape![ME, BN]);

            // Q rows [q0, q0 + q_len), the G heads of kv head `kvh`: [BM, G, D] -> [ME, D] (row = tok*G + g).
            // Rows past q_len are zero-padded on load and dropped on store.
            let q_shape: Shape<{ [-1, HQ, D] }> = Shape::<{ [-1, HQ, D] }> { dims: &[q_len] };
            let q_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> { dims: &[HQ * D, D] };
            let q_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(q_ptr).offset(q0 * HQ * D);
            let q_tv: Tensor<bf16, { [-1, HQ, D] }> = unsafe { make_tensor_view(q_base, q_shape, q_strides, tok) };
            let q_part: Partition<bf16, { [BM, G, D] }> = q_tv.partition(const_shape![BM, G, D]);
            let tq: Tile<bf16, { [BM, G, D] }> =
                load_view_tko(&q_part, [0i32, kvh, 0i32], ordering::Weak, scope::TileBlock, Some(LAT), tma::Enabled);
            let tq: Tile<bf16, { [ME, D] }> = tq.reshape(const_shape![ME, D]);

            let mut m_i: Tile<f32, { [ME, 1] }> = constant(-1.0e30f32, const_shape![ME, 1]);
            let mut l_i: Tile<f32, { [ME, 1] }> = constant(0.0f32, const_shape![ME, 1]);
            let mut acc: Tile<f32, { [ME, D] }> = constant(0.0f32, const_shape![ME, D]);
            let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> { dims: &[1i32, 0i32] };
            let k_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[HK * D] };
            let v_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[2i32 * HK * D] };

            // Range A then range B; the online softmax carries (m, l, acc) across both (the LSE merge).
            for r in 0i32..2i32 {
                let r0: i32 = if r == 0i32 { a0 } else { b0 };
                let r_len: i32 = if r == 0i32 { a_len } else { b_len };
                let kv_shape: Shape<{ [-1, D] }> = Shape::<{ [-1, D] }> { dims: &[max(r_len, 1i32)] };
                let k_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(k_ptr).offset((r0 * HK + kvh) * D);
                let v_base: PointerTile<*mut bf16, { [] }> =
                    pointer_to_tile(kv_ptr).offset((r0 * 2i32 * HK + HK + kvh) * D);
                let k_tv: Tensor<bf16, { [-1, D] }> = unsafe { make_tensor_view(k_base, kv_shape, k_strides, tok) };
                let v_tv: Tensor<bf16, { [-1, D] }> = unsafe { make_tensor_view(v_base, kv_shape, v_strides, tok) };
                let k_part: Partition<bf16, { [BN, D] }> = k_tv.partition(const_shape![BN, D]);
                let v_part: Partition<bf16, { [BN, D] }> = v_tv.partition(const_shape![BN, D]);
                let n_full: i32 = r_len / BN;
                let n_blocks: i32 = ceil_div(r_len, BN);
                for j in 0i32..n_blocks {
                    let tk: Tile<bf16, { [BN, D] }> =
                        load_view_tko(&k_part, [j, 0i32], ordering::Weak, scope::TileBlock, Some(LAT), tma::Enabled);
                    let tkt: Tile<bf16, { [D, BN] }> = permute(tk, transpose);
                    let mut qk: Tile<f32, { [ME, BN] }> = mma(tq, tkt, zero_mn);
                    if j >= n_full {
                        // Partial last block: keys past the range are masked out.
                        let key: Tile<i32, { [ME, BN] }> = (j * BN).broadcast(const_shape![ME, BN]) + offs_n;
                        let valid: Tile<bool, { [ME, BN] }> = lt_tile(key, r_len.broadcast(const_shape![ME, BN]));
                        qk = select(valid, qk, neg_big);
                    }
                    let qk_max: Tile<f32, { [ME] }> = reduce_max(qk, 1i32);
                    let m_ij: Tile<f32, { [ME, 1] }> = max_tile(m_i, qk_max.reshape(const_shape![ME, 1]) * scale_c);
                    let p: Tile<f32, { [ME, BN] }> =
                        exp2(qk * scale_t - m_ij.broadcast(const_shape![ME, BN]), ftz::Disabled);
                    let l_ij: Tile<f32, { [ME] }> = reduce_sum(p, 1i32);
                    let alpha: Tile<f32, { [ME, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
                    l_i = l_i * alpha + l_ij.reshape(const_shape![ME, 1]);
                    let tv: Tile<bf16, { [BN, D] }> =
                        load_view_tko(&v_part, [j, 0i32], ordering::Weak, scope::TileBlock, Some(LAT), tma::Enabled);
                    // P @ V with P split into PS bf16 parts (hi, mid, lo: ~24 mantissa bits at PS = 3). The
                    // reference's sdpa (MATH backend for head_dim 256 with a mask) keeps P in fp32; a single bf16
                    // P is 4x off its error envelope (see tests/kernels_attn.rs). The block product starts from
                    // zero and is added to the running sum in fp32 rather than accumulated inside the MMA.
                    let p_hi: Tile<bf16, { [ME, BN] }> = convert_tile(p);
                    let mut pv: Tile<f32, { [ME, D] }> = zero_md;
                    if PS > 1i32 {
                        let p_hi_f: Tile<f32, { [ME, BN] }> = convert_tile(p_hi);
                        let p_r1: Tile<f32, { [ME, BN] }> = p - p_hi_f;
                        let p_mid: Tile<bf16, { [ME, BN] }> = convert_tile(p_r1);
                        if PS > 2i32 {
                            let p_mid_f: Tile<f32, { [ME, BN] }> = convert_tile(p_mid);
                            let p_lo: Tile<bf16, { [ME, BN] }> = convert_tile(p_r1 - p_mid_f);
                            pv = mma(p_lo, tv, pv);
                        }
                        pv = mma(p_mid, tv, pv);
                    }
                    pv = mma(p_hi, tv, pv);
                    acc = acc * alpha.broadcast(const_shape![ME, D]) + pv;
                    m_i = m_ij;
                }
            }

            // Epilogue: o = bf16(acc / l); out = bf16(o * bf16(sigmoid(gate))).
            let o: Tile<f32, { [ME, D] }> = true_div(acc, l_i.broadcast(const_shape![ME, D]));
            let o: Tile<bf16, { [ME, D] }> = convert_tile(o);
            let o: Tile<f32, { [ME, D] }> = convert_tile(o);
            let g_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> { dims: &[HQ * 2i32 * D, 2i32 * D] };
            let g_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(qg_ptr).offset(q0 * HQ * 2i32 * D + D);
            let g_tv: Tensor<bf16, { [-1, HQ, D] }> = unsafe { make_tensor_view(g_base, q_shape, g_strides, tok) };
            let g_part: Partition<bf16, { [BM, G, D] }> = g_tv.partition(const_shape![BM, G, D]);
            let tg: Tile<bf16, { [BM, G, D] }> =
                load_view_tko(&g_part, [0i32, kvh, 0i32], ordering::Weak, scope::TileBlock, None, tma::Enabled);
            let gf: Tile<f32, { [ME, D] }> = convert_tile(tg.reshape(const_shape![ME, D]));
            let one: Tile<f32, { [ME, D] }> = constant(1.0f32, const_shape![ME, D]);
            let sg: Tile<f32, { [ME, D] }> = true_div(one, one + exp(negf(gf)));
            let sg: Tile<bf16, { [ME, D] }> = convert_tile(sg);
            let sg: Tile<f32, { [ME, D] }> = convert_tile(sg);
            let y: Tile<bf16, { [ME, D] }> = convert_tile(o * sg);

            let o_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr).offset(q0 * HQ * D);
            let o_tv: Tensor<bf16, { [-1, HQ, D] }> = unsafe { make_tensor_view(o_base, q_shape, q_strides, tok) };
            let mut o_part: PartitionMut<bf16, { [BM, G, D] }> =
                unsafe { o_tv.partition_full_mut(const_shape![BM, G, D]) };
            let y: Tile<bf16, { [BM, G, D] }> = y.reshape(const_shape![BM, G, D]);
            unsafe {
                store_view_tko_mut(
                    &mut o_part,
                    y,
                    [0i32, kvh, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
        }
    }
}

/// Tile sizes of [`seg_attn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnTiles {
    /// Query tokens per tile block (× 4 grouped heads = rows of the MMA).
    pub bm: usize,
    /// Keys per inner step.
    pub bn: usize,
    /// Load pipelining hint.
    pub latency: usize,
    /// bf16 parts P is split into for `P @ V` (1, 2 or 3). 3 matches the fp32 reference (default); 1 is the
    /// classic flash-attention bf16 P, ~4x outside the reference's error envelope.
    pub p_split: usize,
}

impl Default for AttnTiles {
    fn default() -> Self {
        AttnTiles { bm: 16, bn: 64, latency: 2, p_split: 3 }
    }
}

/// Device buffers a [`seg_attn`] launch reads (all live for the duration of the launch).
#[derive(Clone, Copy, Debug)]
pub struct SegAttnArgs {
    /// `[T, 16, 256]` output.
    pub out: DevicePointer<bf16>,
    /// `[T, 16, 256]` from [`qk_norm_rope`].
    pub q: DevicePointer<bf16>,
    /// `[T, 4, 256]` from [`qk_norm_rope`].
    pub k: DevicePointer<bf16>,
    /// `[T, 8, 256]` `[Wk; Wv]` output (v = heads 4..8).
    pub kv: DevicePointer<bf16>,
    /// `[T, 16, 512]` `q_proj` output (gate = second half of each head).
    pub qg: DevicePointer<bf16>,
    /// [`AttnPlan::work`] on the device.
    pub work: DevicePointer<i32>,
    /// Items in `work` (the grid's x extent), including padding items.
    pub num_items: usize,
}

/// Launch `seg_attn` over `args.num_items` work items × 4 kv heads.
///
/// # Safety
/// Every pointer must reference a live device buffer of the documented shape for the batch the plan was
/// built from, and `out` must not alias any input. The plan must have been built with `tiles.bm`.
pub unsafe fn seg_attn(args: SegAttnArgs, tiles: AttnTiles) -> impl DeviceOp + GraphNode {
    let generics = vec![
        tiles.bm.to_string(),
        tiles.bn.to_string(),
        HEAD_DIM.to_string(),
        (NUM_Q_HEADS / NUM_KV_HEADS).to_string(),
        (tiles.bm * NUM_Q_HEADS / NUM_KV_HEADS).to_string(),
        NUM_Q_HEADS.to_string(),
        NUM_KV_HEADS.to_string(),
        tiles.latency.to_string(),
        tiles.p_split.to_string(),
    ];
    unsafe {
        seg_attn_module::seg_attn(
            args.out,
            args.q,
            args.k,
            args.kv,
            args.qg,
            args.work,
            args.num_items as i32,
            ATTN_SCALE,
        )
    }
    .generics(generics)
    .grid((args.num_items as u32, NUM_KV_HEADS as u32, 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::DecisionBatch;

    #[test]
    fn plan_covers_every_row_once_with_the_right_ranges() {
        let mut b = DecisionBatch::default();
        b.push_request(&[1; 40], &[(&[2; 5], 2), (&[3; 30], 2), (&[4; 1], 2)]).unwrap();
        b.push_request(&[5; 3], &[(&[6; 17], 2)]).unwrap();
        let plan = AttnPlan::new(&b.segments, 16);
        let mut seen = vec![0; b.num_tokens()];
        for w in plan.work.chunks(WORK_FIELDS) {
            let [q0, n, a0, alen, b0, blen, ..] = *w else { unreachable!() };
            assert!((1..=16).contains(&n));
            let seg = b.segments.iter().find(|s| s.start as i32 <= q0 && q0 < s.end() as i32).unwrap();
            assert!(q0 + n <= seg.end() as i32, "a tile never crosses a segment end");
            assert_eq!((q0 - seg.start as i32) % 16, 0, "tiles start at the segment start");
            let pre = if seg.is_prefix() { seg } else { &b.segments[seg.parent as usize] };
            assert_eq!((a0, alen), (pre.start as i32, pre.len as i32));
            if seg.is_prefix() {
                assert_eq!(blen, 0);
            } else {
                assert_eq!((b0, blen), (seg.start as i32, seg.len as i32));
            }
            for t in q0..q0 + n {
                seen[t as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        assert!(plan.num_items() <= AttnPlan::max_items(b.num_tokens(), b.segments.len(), 16));
        let costs: Vec<i32> = plan.work.chunks(WORK_FIELDS).map(|w| w[3] + w[5]).collect();
        assert!(costs.windows(2).all(|c| c[0] >= c[1]));
    }

    #[test]
    fn positions_follow_pos0() {
        let mut b = DecisionBatch::default();
        b.push_request(&[1, 2, 3], &[(&[4, 5], 2), (&[6], 2)]).unwrap();
        assert_eq!(positions(&b.segments), [0, 1, 2, 3, 4, 3]);
    }
}
