//! cuTile device code for the GatedDeltaNet mixer (DESIGN.md §3, §6.1, §6.3).
//!
//! Every entry takes raw device pointers plus element strides, so callers can hand in strided column views of the
//! fused `[T, 12352]` projection output without copies. Segment metadata is read from device tables built by
//! [`super::GdnPlan`]; grids depend only on the plan's capacities, so one launch sequence can be graph-captured per
//! T bucket. All entries are `unsafe`: the safety contract is the host wrapper's (`super`), which checks shapes.
//!
//! Tables (i32, row-major):
//! - chunk table `[max_chunks, 4]`: `start, len, prev_end, prev_avail`. A chunk is ≤ 64 rows of one segment,
//!   64-aligned to the segment start. `len == 0` marks an unused entry. For the conv, the causal predecessors of
//!   row `start` are rows `prev_end-3 .. prev_end`, of which only the last `prev_avail` exist.
//! - segment table `[max_segs, 4]`: `start, len, phase, slot`. `phase` 0 = prefix, 1 = suffix, -1 = unused;
//!   `slot` is the prefix-state scratch slot (the prefix's own, or the parent's for a suffix).

#[cutile::module]
pub mod gdn_tile {
    use cutile::core::*;

    /// Read `table[idx]` (i32) from a device table view.
    fn table_at(part: &Partition<i32, { [1] }>, idx: i32) -> i32 {
        let t: Tile<i32, { [1] }> = part.load([idx]);
        tile_to_scalar(t.reshape(const_shape![]))
    }

    /// `a @ b + acc` in fp32 via split TF32 ("3xTF32"): with `x = hi(x) + lo(x)`, both TF32-rounded,
    /// `a b ≈ hi(a) hi(b) + hi(a) lo(b) + lo(a) hi(b)` on the tensor cores with fp32 accumulation; the dropped
    /// `lo lo` term and the rounding of `lo` leave ~2^-21 relative error per product. Plain f32 `mma` is
    /// emulated off the tensor cores and spills heavily (DESIGN §6.4 risk Q3).
    fn mma_f32x3<const M: i32, const N: i32, const K: i32>(
        a: Tile<f32, { [M, K] }>,
        b: Tile<f32, { [K, N] }>,
        acc: Tile<f32, { [M, N] }>,
    ) -> Tile<f32, { [M, N] }> {
        let a_hi: Tile<tf32, { [M, K] }> = convert_tile(a);
        let a_hi_f: Tile<f32, { [M, K] }> = convert_tile(a_hi);
        let a_lo: Tile<tf32, { [M, K] }> = convert_tile(a - a_hi_f);
        let b_hi: Tile<tf32, { [K, N] }> = convert_tile(b);
        let b_hi_f: Tile<f32, { [K, N] }> = convert_tile(b_hi);
        let b_lo: Tile<tf32, { [K, N] }> = convert_tile(b - b_hi_f);
        let acc: Tile<f32, { [M, N] }> = mma(a_lo, b_hi, acc);
        let acc: Tile<f32, { [M, N] }> = mma(a_hi, b_lo, acc);
        mma(a_hi, b_hi, acc)
    }

    /// `a @ b + acc` at the precision selected by `SPLIT` / `E`: `SPLIT == 3` is [`mma_f32x3`] (fp32-grade);
    /// otherwise both operands are rounded once to `E` (tf32 or bf16) and multiplied on the tensor cores with
    /// fp32 accumulation.
    fn mma_p<E: ElementType, const SPLIT: i32, const M: i32, const N: i32, const K: i32>(
        a: Tile<f32, { [M, K] }>,
        b: Tile<f32, { [K, N] }>,
        acc: Tile<f32, { [M, N] }>,
    ) -> Tile<f32, { [M, N] }> {
        if SPLIT == 3i32 {
            mma_f32x3(a, b, acc)
        } else {
            let ae: Tile<E, { [M, K] }> = convert_tile(a);
            let be: Tile<E, { [K, N] }> = convert_tile(b);
            mma(ae, be, acc)
        }
    }

    fn silu_f32<const S: [i32; 2]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let one: Tile<f32, S> = constant(1.0f32, x.shape());
        true_div(x, one + exp(negf(x)))
    }

    /// Depthwise causal conv1d (kernel 4, no bias) over `C` channels, then SiLU, with taps before a segment's first
    /// token taken from the parent prefix's tail (suffix) or zero (prefix).
    ///
    /// Grid `(max_chunks, C / BC, 1)`. x: `[T, C]` bf16 with row stride `x_stride`; w: `[C, 4]` bf16 contiguous;
    /// out: `[T, C]` bf16 with row stride `out_stride`. Rounding follows the reference's torch fallback
    /// (`causal_conv1d_fn`): fp32 accumulate, round to bf16, SiLU in fp32, round to bf16.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn tree_conv1d_silu<const C: i32, const BC: i32, const HB: i32, const QK: i32>(
        x_ptr: *mut bf16,
        x_stride: i32,
        w_ptr: *mut bf16,
        out_ptr: *mut bf16,
        out_stride: i32,
        q_ptr: *mut f32,
        k_ptr: *mut f32,
        chunks_ptr: *mut i32,
        max_chunks: i32,
        tokens: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let cb = pid.1;
        let tok: Token = new_token_unordered();
        let tab_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(chunks_ptr);
        let tab_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[max_chunks * 4i32] };
        let tab_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let tab: Tensor<i32, { [-1] }> = unsafe { make_tensor_view(tab_base, tab_shape, tab_strides, tok) };
        let tab_part: Partition<i32, { [1] }> = tab.partition(const_shape![1]);
        let start: i32 = table_at(&tab_part, pid.0 * 4i32);
        let len: i32 = table_at(&tab_part, pid.0 * 4i32 + 1i32);
        let prev_end: i32 = table_at(&tab_part, pid.0 * 4i32 + 2i32);
        let prev_avail: i32 = table_at(&tab_part, pid.0 * 4i32 + 3i32);
        if len > 0i32 {
            let x_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(x_ptr);
            let x_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, C] };
            let x_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[x_stride] };
            let x: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(x_base, x_shape, x_strides, tok) };
            let x_part: Partition<bf16, { [1, BC] }> = x.partition(const_shape![1, BC]);
            // w viewed as [4, C] (stride 1 between taps, 4 between channels): row j is tap j.
            let w_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(w_ptr);
            let w_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[4i32, C] };
            let w_strides: Array<{ [1, 4] }> = Array::<{ [1, 4] }> { dims: &[] };
            let w: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(w_base, w_shape, w_strides, tok) };
            let w_part: Partition<bf16, { [1, BC] }> = w.partition(const_shape![1, BC]);
            let w0: Tile<f32, { [1, BC] }> = convert_tile(w_part.load([0i32, cb]));
            let w1: Tile<f32, { [1, BC] }> = convert_tile(w_part.load([1i32, cb]));
            let w2: Tile<f32, { [1, BC] }> = convert_tile(w_part.load([2i32, cb]));
            let w3: Tile<f32, { [1, BC] }> = convert_tile(w_part.load([3i32, cb]));
            let out_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
            let out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, C] };
            let out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[out_stride] };
            let out: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(out_base, out_shape, out_strides, tok) };
            let mut out_part: PartitionMut<bf16, { [1, BC] }> =
                unsafe { out.partition_full_mut(const_shape![1, BC]) };
            // QK: channel blocks of q (0..2048) or k (2048..4096) also write the L2-normalised f32 rows that
            // `gdn_prep` would compute (same arithmetic), so the mixer does not read them back.
            // q and k viewed as [T * 16, 128] (row = token * 16 + head), stored [HB, 128] at a time.
            let is_q: bool = QK == 1i32 && cb * BC < 2048i32;
            let is_k: bool = QK == 1i32 && cb * BC >= 2048i32 && cb * BC < 4096i32;
            let hd_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens * 16i32, 128i32] };
            let hd_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[128i32] };
            let q_out: Tensor<f32, { [-1, -1] }> = unsafe { make_tensor_view(pointer_to_tile(q_ptr), hd_shape, hd_strides, tok) };
            let hd_shape2: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens * 16i32, 128i32] };
            let hd_strides2: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[128i32] };
            let k_out: Tensor<f32, { [-1, -1] }> = unsafe { make_tensor_view(pointer_to_tile(k_ptr), hd_shape2, hd_strides2, tok) };
            let mut q_part: PartitionMut<f32, { [HB, 128] }> =
                unsafe { q_out.partition_full_mut(const_shape![HB, 128]) };
            let mut k_part: PartitionMut<f32, { [HB, 128] }> =
                unsafe { k_out.partition_full_mut(const_shape![HB, 128]) };
            // First head of this channel block within q (or k).
            let head0: i32 = (cb * BC) / 128i32 - if is_k { 16i32 } else { 0i32 };
            let zero: Tile<f32, { [1, BC] }> = constant(0.0f32, const_shape![1, BC]);
            // Sliding window of the three causal predecessors, oldest first.
            let mut h0: Tile<f32, { [1, BC] }> = if prev_avail >= 3i32 {
                let t: Tile<bf16, { [1, BC] }> = x_part.load([prev_end - 3i32, cb]);
                let t: Tile<f32, { [1, BC] }> = convert_tile(t);
                t
            } else {
                zero
            };
            let mut h1: Tile<f32, { [1, BC] }> = if prev_avail >= 2i32 {
                let t: Tile<bf16, { [1, BC] }> = x_part.load([prev_end - 2i32, cb]);
                let t: Tile<f32, { [1, BC] }> = convert_tile(t);
                t
            } else {
                zero
            };
            let mut h2: Tile<f32, { [1, BC] }> = if prev_avail >= 1i32 {
                let t: Tile<bf16, { [1, BC] }> = x_part.load([prev_end - 1i32, cb]);
                let t: Tile<f32, { [1, BC] }> = convert_tile(t);
                t
            } else {
                zero
            };
            for i in 0i32..len {
                let xi: Tile<f32, { [1, BC] }> = convert_tile(x_part.load([start + i, cb]));
                let acc: Tile<f32, { [1, BC] }> = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xi;
                let conv: Tile<bf16, { [1, BC] }> = convert_tile(acc);
                let conv_f: Tile<f32, { [1, BC] }> = convert_tile(conv);
                let act: Tile<bf16, { [1, BC] }> = convert_tile(silu_f32(conv_f));
                out_part.store(act, [start + i, cb]);
                if is_q || is_k {
                    let a: Tile<f32, { [1, BC] }> = convert_tile(act);
                    let a: Tile<f32, { [HB, 128] }> = a.reshape(const_shape![HB, 128]);
                    let ss: Tile<f32, { [HB] }> = reduce_sum(a * a, 1i32);
                    let eps: Tile<f32, { [HB, 1] }> = constant(1e-6f32, const_shape![HB, 1]);
                    let inv: Tile<f32, { [HB, 1] }> =
                        rsqrt(ss.reshape(const_shape![HB, 1]) + eps, ftz::Disabled);
                    let n: Tile<f32, { [HB, 128] }> = a * inv.broadcast(const_shape![HB, 128]);
                    let idx: i32 = ((start + i) * 16i32 + head0) / HB;
                    if is_q {
                        let sc: Tile<f32, { [HB, 128] }> = constant(0.08838834764831845f32, const_shape![HB, 128]);
                        q_part.store(n * sc, [idx, 0i32]);
                    } else {
                        k_part.store(n, [idx, 0i32]);
                    }
                }
                h0 = h1;
                h1 = h2;
                h2 = xi;
            }
        }
    }

    /// Split, normalise and gate: q, k `[T, 16, 128]` f32 (fp32 L2 norm with eps 1e-6; q also times 128^-0.5),
    /// g `[T, 32]` f32 = -exp(A_log) * softplus(a + dt_bias), beta `[T, 32]` f32 = sigmoid(b) rounded to bf16.
    ///
    /// Grid `(ceil(T / BR), 16, 1)`. qkv: conv output `[T, 8192]` bf16 (q at column 0, k at 2048), row stride
    /// `qkv_stride`. a, b: `[T, 32]` bf16 column views with row strides `a_stride`, `b_stride`.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gdn_prep<const BR: i32, const QK: i32>(
        qkv_ptr: *mut bf16,
        qkv_stride: i32,
        a_ptr: *mut bf16,
        a_stride: i32,
        b_ptr: *mut bf16,
        b_stride: i32,
        a_log_ptr: *mut bf16,
        dt_bias_ptr: *mut bf16,
        q_ptr: *mut f32,
        k_ptr: *mut f32,
        g_ptr: *mut f32,
        beta_ptr: *mut f32,
        tokens: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let rb = pid.0;
        let hk = pid.1;
        let tok: Token = new_token_unordered();
        let qkv_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(qkv_ptr);
        let qkv_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 4096i32] };
        let qkv_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[qkv_stride] };
        let qkv: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(qkv_base, qkv_shape, qkv_strides, tok) };
        let qkv_part: Partition<bf16, { [BR, 128] }> = qkv.partition(const_shape![BR, 128]);
        // QK == 0: q and k come from the fused conv (`tree_conv1d_silu` with QK); only the gates are computed.
        if QK == 1i32 {
            let q_out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(q_ptr);
            let q_out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 2048i32] };
            let q_out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[2048i32] };
            let q_out: Tensor<f32, { [-1, -1] }> = unsafe { make_tensor_view(q_out_base, q_out_shape, q_out_strides, tok) };
            let k_out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(k_ptr);
            let k_out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 2048i32] };
            let k_out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[2048i32] };
            let k_out: Tensor<f32, { [-1, -1] }> = unsafe { make_tensor_view(k_out_base, k_out_shape, k_out_strides, tok) };
            let mut q_part: PartitionMut<f32, { [BR, 128] }> = unsafe { q_out.partition_full_mut(const_shape![BR, 128]) };
            let mut k_part: PartitionMut<f32, { [BR, 128] }> = unsafe { k_out.partition_full_mut(const_shape![BR, 128]) };
            let eps: Tile<f32, { [BR, 1] }> = constant(1e-6f32, const_shape![BR, 1]);
            let scale: Tile<f32, { [BR, 128] }> = constant(0.08838834764831845f32, const_shape![BR, 128]);

            let q: Tile<f32, { [BR, 128] }> = convert_tile(qkv_part.load([rb, hk]));
            let q_ss: Tile<f32, { [BR] }> = reduce_sum(q * q, 1i32);
            let q_inv: Tile<f32, { [BR, 1] }> = rsqrt(q_ss.reshape(const_shape![BR, 1]) + eps, ftz::Disabled);
            let q_n: Tile<f32, { [BR, 128] }> = q * q_inv.broadcast(const_shape![BR, 128]);
            q_part.store(q_n * scale, [rb, hk]);

            let k: Tile<f32, { [BR, 128] }> = convert_tile(qkv_part.load([rb, hk + 16i32]));
            let k_ss: Tile<f32, { [BR] }> = reduce_sum(k * k, 1i32);
            let k_inv: Tile<f32, { [BR, 1] }> = rsqrt(k_ss.reshape(const_shape![BR, 1]) + eps, ftz::Disabled);
            k_part.store(k * k_inv.broadcast(const_shape![BR, 128]), [rb, hk]);
        }

        if hk == 0i32 {
            let a_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(a_ptr);
            let a_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 32i32] };
            let a_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[a_stride] };
            let a: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(a_base, a_shape, a_strides, tok) };
            let b_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(b_ptr);
            let b_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 32i32] };
            let b_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[b_stride] };
            let b: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(b_base, b_shape, b_strides, tok) };
            let vec_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[32i32] };
            let unit: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
            let a_log_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(a_log_ptr);
            let dt_bias_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(dt_bias_ptr);
            let a_log: Tensor<bf16, { [-1] }> =
                unsafe { make_tensor_view(a_log_base, vec_shape, unit, tok) };
            let dt_bias: Tensor<bf16, { [-1] }> =
                unsafe { make_tensor_view(dt_bias_base, vec_shape, unit, tok) };
            let a_t: Tile<f32, { [BR, 32] }> = convert_tile(a.partition(const_shape![BR, 32]).load([rb, 0i32]));
            let b_t: Tile<f32, { [BR, 32] }> = convert_tile(b.partition(const_shape![BR, 32]).load([rb, 0i32]));
            let a_log_t: Tile<f32, { [32] }> = convert_tile(a_log.partition(const_shape![32]).load([0i32]));
            let dt_t: Tile<f32, { [32] }> = convert_tile(dt_bias.partition(const_shape![32]).load([0i32]));
            let neg_a: Tile<f32, { [BR, 32] }> =
                negf(exp(a_log_t)).reshape(const_shape![1, 32]).broadcast(const_shape![BR, 32]);
            let x: Tile<f32, { [BR, 32] }> = a_t + dt_t.reshape(const_shape![1, 32]).broadcast(const_shape![BR, 32]);
            // torch softplus (beta 1, threshold 20): x if x > 20 else log1p(exp(x)). log1p via the exact
            // `log(1+u) * u / ((1+u) - 1)` correction, so tiny exp(x) keeps full relative precision.
            let one: Tile<f32, { [BR, 32] }> = constant(1.0f32, const_shape![BR, 32]);
            let u: Tile<f32, { [BR, 32] }> = exp(x);
            let w: Tile<f32, { [BR, 32] }> = one + u;
            let dw: Tile<f32, { [BR, 32] }> = w - one;
            let zero: Tile<f32, { [BR, 32] }> = constant(0.0f32, const_shape![BR, 32]);
            let twenty: Tile<f32, { [BR, 32] }> = constant(20.0f32, const_shape![BR, 32]);
            let tiny: Tile<bool, { [BR, 32] }> = eq_tile(dw, zero);
            let safe_dw: Tile<f32, { [BR, 32] }> = select(tiny, one, dw);
            let l1p: Tile<f32, { [BR, 32] }> = select(tiny, u, log(w) * true_div(u, safe_dw));
            let sp: Tile<f32, { [BR, 32] }> = select(gt_tile(x, twenty), x, l1p);
            let g: Tile<f32, { [BR, 32] }> = neg_a * sp;
            // sigmoid in fp32, rounded to bf16 as torch does for a bf16 input.
            let sig_f: Tile<f32, { [BR, 32] }> = true_div(one, one + exp(negf(b_t)));
            let sig: Tile<bf16, { [BR, 32] }> = convert_tile(sig_f);
            let beta: Tile<f32, { [BR, 32] }> = convert_tile(sig);

            let g_out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(g_ptr);
            let g_out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 32i32] };
            let g_out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[32i32] };
            let g_out: Tensor<f32, { [-1, -1] }> = unsafe { make_tensor_view(g_out_base, g_out_shape, g_out_strides, tok) };
            let beta_out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(beta_ptr);
            let beta_out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 32i32] };
            let beta_out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[32i32] };
            let beta_out: Tensor<f32, { [-1, -1] }> = unsafe { make_tensor_view(beta_out_base, beta_out_shape, beta_out_strides, tok) };
            let mut g_part: PartitionMut<f32, { [BR, 32] }> = unsafe { g_out.partition_full_mut(const_shape![BR, 32]) };
            let mut beta_part: PartitionMut<f32, { [BR, 32] }> =
                unsafe { beta_out.partition_full_mut(const_shape![BR, 32]) };
            g_part.store(g, [rb, 0i32]);
            beta_part.store(beta, [rb, 0i32]);
        }
    }

    /// A `[rows, cols]` f32 view starting at row `r0` of a row-major buffer with row stride `stride`.
    fn rows_f32(ptr: *mut f32, r0: i32, rows: i32, cols: i32, stride: i32, tok: Token) -> Tensor<f32, { [-1, -1] }> {
        let base: PointerTile<*mut f32, { [] }> = addptr(pointer_to_tile(ptr), r0 * stride);
        let shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[rows, cols] };
        let strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[stride] };
        unsafe { make_tensor_view(base, shape, strides, tok) }
    }

    /// Read one chunk-table entry: `(start, len)`.
    fn chunk_at(chunks_ptr: *mut i32, max_chunks: i32, c: i32, tok: Token) -> (i32, i32) {
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(chunks_ptr);
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[max_chunks * 4i32] };
        let strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let tab: Tensor<i32, { [-1] }> = unsafe { make_tensor_view(base, shape, strides, tok) };
        let part: Partition<i32, { [1] }> = tab.partition(const_shape![1]);
        (table_at(&part, c * 4i32), table_at(&part, c * 4i32 + 1i32))
    }

    /// Phase A1 of the chunked gated delta rule, parallel over (chunk, v-head): the UT system's inverse.
    ///
    /// For one chunk (≤ 64 rows of one segment) and v-head h (k-head h/2), with `cum` the in-chunk cumulative sum
    /// of g: `A = strict_lower((k*beta) k^T ⊙ exp(cum_i - cum_j))` and `Tinv = (I + A)^-1` by forward
    /// substitution, the reference's unit lower-triangular solve. Grid `(max_chunks, 32, 1)`.
    /// k: `[T, 2048]` f32; g, beta: `[T, 32]` f32; tinv out: `[T, 32 * 64]` f32 (row i of the chunk at row
    /// start + i, columns h*64..).
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gdn_chunk_tinv(
        k_ptr: *mut f32,
        g_ptr: *mut f32,
        beta_ptr: *mut f32,
        tinv_ptr: *mut f32,
        chunks_ptr: *mut i32,
        max_chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h = pid.1;
        let tok: Token = new_token_unordered();
        let (start, len): (i32, i32) = chunk_at(chunks_ptr, max_chunks, pid.0, tok);
        if len > 0i32 {
            let g: Tensor<f32, { [-1, -1] }> = rows_f32(g_ptr, start, len, 32i32, 32i32, tok);
            let beta: Tensor<f32, { [-1, -1] }> = rows_f32(beta_ptr, start, len, 32i32, 32i32, tok);
            let gt: Tile<f32, { [64, 1] }> = g.partition(const_shape![64, 1]).load([0i32, h]);
            let bt: Tile<f32, { [64, 1] }> = beta.partition(const_shape![64, 1]).load([0i32, h]);
            let cum: Tile<f32, { [64, 1] }> = scan_sum(gt, 0i32, reverse::Forward, 0.0f32);
            let idx: Tile<i32, { [64] }> = iota(const_shape![64]);
            let rows: Tile<i32, { [64, 64] }> = idx.reshape(const_shape![64, 1]).broadcast(const_shape![64, 64]);
            let cols: Tile<i32, { [64, 64] }> = idx.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
            let zero: Tile<f32, { [64, 64] }> = constant(0.0f32, const_shape![64, 64]);
            let one: Tile<f32, { [64, 64] }> = constant(1.0f32, const_shape![64, 64]);
            let cum_i: Tile<f32, { [64, 64] }> = cum.broadcast(const_shape![64, 64]);
            let cum_j: Tile<f32, { [64, 64] }> = cum.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
            let k: Tensor<f32, { [-1, -1] }> = rows_f32(k_ptr, start, len, 2048i32, 2048i32, tok);
            let kt: Tile<f32, { [64, 128] }> = k.partition(const_shape![64, 128]).load([0i32, h / 2i32]);
            let kb: Tile<f32, { [64, 128] }> = kt * bt.broadcast(const_shape![64, 128]);
            let a: Tile<f32, { [64, 64] }> =
                select(gt_tile(rows, cols), mma_f32x3(kb, kt.transpose(), zero) * exp(cum_i - cum_j), zero);

            // Forward substitution: row i of Tinv = e_i - sum_{j<i} A[i, j] Tinv[j, :].
            let mut tinv: Tile<f32, { [64, 64] }> = select(eq_tile(rows, cols), one, zero);
            for i in 1i32..64i32 {
                let sel: Tile<bool, { [64, 64] }> = eq_tile(rows, broadcast_scalar(i, const_shape![64, 64]));
                let a_row: Tile<f32, { [64] }> = reduce_sum(select(sel, a, zero), 0i32);
                let contrib: Tile<f32, { [64] }> =
                    reduce_sum(a_row.reshape(const_shape![64, 1]).broadcast(const_shape![64, 64]) * tinv, 0i32);
                let new_row: Tile<f32, { [64, 64] }> =
                    tinv - contrib.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
                tinv = select(sel, new_row, tinv);
            }
            let out: Tensor<f32, { [-1, -1] }> = rows_f32(tinv_ptr, start, len, 2048i32, 2048i32, tok);
            let mut part: PartitionMut<f32, { [64, 64] }> = unsafe { out.partition_full_mut(const_shape![64, 64]) };
            part.store(tinv, [0i32, h]);
        }
    }

    /// Phase A2 of the chunked gated delta rule, parallel over (chunk, v-head): every scan operand that does not
    /// depend on the recurrent state, with `D[i, j] = exp(cum_i - cum_j)`:
    /// - `w = Tinv (k*beta*exp(cum))` and `u = Tinv (v*beta)` (the reference's `k_cumdecay` and `new_values`);
    /// - `qd = q*exp(cum)`, `kd = k*exp(cum_last - cum)` and `p = (q k^T) ⊙ D` on and below the diagonal.
    ///
    /// Grid `(max_chunks, 32, 1)`. q, k: `[T, 2048]` f32; v: `[T, 4096]` bf16 view (row stride `v_stride`);
    /// g, beta: `[T, 32]` f32; tinv: `[T, 32 * 64]` f32. Outputs: w, u, qd, kd `[T, 4096]` f32; p
    /// `[T, 32 * 64]` f32.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gdn_chunk_wy(
        q_ptr: *mut f32,
        k_ptr: *mut f32,
        v_ptr: *mut bf16,
        v_stride: i32,
        g_ptr: *mut f32,
        beta_ptr: *mut f32,
        tinv_ptr: *mut f32,
        w_ptr: *mut f32,
        u_ptr: *mut f32,
        qd_ptr: *mut f32,
        kd_ptr: *mut f32,
        p_ptr: *mut f32,
        chunks_ptr: *mut i32,
        max_chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h = pid.1;
        let hk = h / 2i32;
        let tok: Token = new_token_unordered();
        let (start, len): (i32, i32) = chunk_at(chunks_ptr, max_chunks, pid.0, tok);
        if len > 0i32 {
            let g: Tensor<f32, { [-1, -1] }> = rows_f32(g_ptr, start, len, 32i32, 32i32, tok);
            let beta: Tensor<f32, { [-1, -1] }> = rows_f32(beta_ptr, start, len, 32i32, 32i32, tok);
            let gt: Tile<f32, { [64, 1] }> = g.partition(const_shape![64, 1]).load([0i32, h]);
            let bt: Tile<f32, { [64, 1] }> = beta.partition(const_shape![64, 1]).load([0i32, h]);
            let cum: Tile<f32, { [64, 1] }> = scan_sum(gt, 0i32, reverse::Forward, 0.0f32);
            let ecum: Tile<f32, { [64, 1] }> = exp(cum);
            let zero_k: Tile<f32, { [64, 128] }> = constant(0.0f32, const_shape![64, 128]);

            let tinv_v: Tensor<f32, { [-1, -1] }> = rows_f32(tinv_ptr, start, len, 2048i32, 2048i32, tok);
            let tinv: Tile<f32, { [64, 64] }> = tinv_v.partition(const_shape![64, 64]).load([0i32, h]);
            let k: Tensor<f32, { [-1, -1] }> = rows_f32(k_ptr, start, len, 2048i32, 2048i32, tok);
            let kt: Tile<f32, { [64, 128] }> = k.partition(const_shape![64, 128]).load([0i32, hk]);
            let w_out: Tensor<f32, { [-1, -1] }> = rows_f32(w_ptr, start, len, 4096i32, 4096i32, tok);
            let mut w_part: PartitionMut<f32, { [64, 128] }> = unsafe { w_out.partition_full_mut(const_shape![64, 128]) };
            w_part.store(mma_f32x3(tinv, kt * (bt * ecum).broadcast(const_shape![64, 128]), zero_k), [0i32, h]);

            let v_base: PointerTile<*mut bf16, { [] }> = addptr(pointer_to_tile(v_ptr), start * v_stride);
            let v_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[len, 4096i32] };
            let v_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[v_stride] };
            let v: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(v_base, v_shape, v_strides, tok) };
            let vt: Tile<bf16, { [64, 128] }> = v.partition(const_shape![64, 128]).load([0i32, h]);
            let vt: Tile<f32, { [64, 128] }> = convert_tile(vt);
            let u_out: Tensor<f32, { [-1, -1] }> = rows_f32(u_ptr, start, len, 4096i32, 4096i32, tok);
            let mut u_part: PartitionMut<f32, { [64, 128] }> = unsafe { u_out.partition_full_mut(const_shape![64, 128]) };
            u_part.store(mma_f32x3(tinv, vt * bt.broadcast(const_shape![64, 128]), zero_k), [0i32, h]);

            // exp(cum_last - cum_i) = exp(sum_{j>i} g_j) from a reverse scan: for strongly decaying heads
            // cum_last - cum_i cancels catastrophically (|cum| ~ 1e3, ulp ~ 1e-4) while the suffix sum is exact
            // where it matters (the last rows). Mathematically identical to the reference.
            let rcum: Tile<f32, { [64, 1] }> = scan_sum(gt, 0i32, reverse::Reverse, 0.0f32);
            let kd_out: Tensor<f32, { [-1, -1] }> = rows_f32(kd_ptr, start, len, 4096i32, 4096i32, tok);
            let mut kd_part: PartitionMut<f32, { [64, 128] }> = unsafe { kd_out.partition_full_mut(const_shape![64, 128]) };
            kd_part.store(kt * exp(rcum - gt).broadcast(const_shape![64, 128]), [0i32, h]);

            let idx: Tile<i32, { [64] }> = iota(const_shape![64]);
            let rows: Tile<i32, { [64, 64] }> = idx.reshape(const_shape![64, 1]).broadcast(const_shape![64, 64]);
            let cols: Tile<i32, { [64, 64] }> = idx.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
            let zero: Tile<f32, { [64, 64] }> = constant(0.0f32, const_shape![64, 64]);
            let cum_i: Tile<f32, { [64, 64] }> = cum.broadcast(const_shape![64, 64]);
            let cum_j: Tile<f32, { [64, 64] }> = cum.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
            let q: Tensor<f32, { [-1, -1] }> = rows_f32(q_ptr, start, len, 2048i32, 2048i32, tok);
            let qt: Tile<f32, { [64, 128] }> = q.partition(const_shape![64, 128]).load([0i32, hk]);
            let p: Tile<f32, { [64, 64] }> =
                select(ge_tile(rows, cols), mma_f32x3(qt, kt.transpose(), zero) * exp(cum_i - cum_j), zero);
            let p_out: Tensor<f32, { [-1, -1] }> = rows_f32(p_ptr, start, len, 2048i32, 2048i32, tok);
            let mut p_part: PartitionMut<f32, { [64, 64] }> = unsafe { p_out.partition_full_mut(const_shape![64, 64]) };
            p_part.store(p, [0i32, h]);
            let qd_out: Tensor<f32, { [-1, -1] }> = rows_f32(qd_ptr, start, len, 4096i32, 4096i32, tok);
            let mut qd_part: PartitionMut<f32, { [64, 128] }> = unsafe { qd_out.partition_full_mut(const_shape![64, 128]) };
            qd_part.store(qt * ecum.broadcast(const_shape![64, 128]), [0i32, h]);
        }
    }

    /// Phase B of the chunked gated delta rule: the sequential inter-chunk scan of one segment, one v-head, one
    /// block of BV value columns, from phase A's operands. Only segments whose table `phase` equals `phase` run.
    ///
    /// Per chunk: `v_new = u - w S`, `o = qd S + p v_new`, `S = exp(sum g) S + kd^T v_new`. Prefixes (phase 0) start from
    /// a zero state and write their final state to `state[slot]`; suffixes (phase 1) start from `state[slot]`
    /// (their parent's) and write nothing. Grid `(max_segs, 32, 128 / BV)`.
    ///
    /// The state `[128, BV]` is held as four `[32, BV]` row blocks and the `[64, 128]` operands are streamed as
    /// `[64, 32]` column blocks, so no MMA operand (with its TF32 hi/lo split) is larger than `[64, 64]`.
    /// state: `[slots, 32, 128, 128]` f32. out: `[T, 4096]` bf16 with row stride `out_stride`.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gdn_chunk_state<const BV: i32>(
        w_ptr: *mut f32,
        u_ptr: *mut f32,
        qd_ptr: *mut f32,
        kd_ptr: *mut f32,
        p_ptr: *mut f32,
        g_ptr: *mut f32,
        state_ptr: *mut f32,
        out_ptr: *mut bf16,
        out_stride: i32,
        segs_ptr: *mut i32,
        max_segs: i32,
        num_slots: i32,
        phase: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h = pid.1;
        let vb = pid.2;
        let vcol = h * (128i32 / BV) + vb;
        let kc = h * 4i32; // first [*, 32] column block of head h in a [T, 4096] buffer
        let tok: Token = new_token_unordered();
        let tab_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(segs_ptr);
        let tab_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[max_segs * 4i32] };
        let tab_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let tab: Tensor<i32, { [-1] }> = unsafe { make_tensor_view(tab_base, tab_shape, tab_strides, tok) };
        let tab_part: Partition<i32, { [1] }> = tab.partition(const_shape![1]);
        let seg_start: i32 = table_at(&tab_part, pid.0 * 4i32);
        let seg_len: i32 = table_at(&tab_part, pid.0 * 4i32 + 1i32);
        let seg_phase: i32 = table_at(&tab_part, pid.0 * 4i32 + 2i32);
        let slot: i32 = table_at(&tab_part, pid.0 * 4i32 + 3i32);
        if seg_phase == phase {
            // state viewed as [slots * 32 * 128, 128] in [32, BV] tiles; (slot, h) rows start at tile row
            // (slot * 32 + h) * 4.
            let state: Tensor<f32, { [-1, -1] }> = rows_f32(state_ptr, 0i32, num_slots * 4096i32, 128i32, 128i32, tok);
            let srow: i32 = (slot * 32i32 + h) * 4i32;
            let zero_s: Tile<f32, { [32, BV] }> = constant(0.0f32, const_shape![32, BV]);
            let state_part: Partition<f32, { [32, BV] }> = state.partition(const_shape![32, BV]);
            let mut s0: Tile<f32, { [32, BV] }> =
                if phase == 0i32 { zero_s } else { state_part.load([srow, vb]) };
            let mut s1: Tile<f32, { [32, BV] }> =
                if phase == 0i32 { zero_s } else { state_part.load([srow + 1i32, vb]) };
            let mut s2: Tile<f32, { [32, BV] }> =
                if phase == 0i32 { zero_s } else { state_part.load([srow + 2i32, vb]) };
            let mut s3: Tile<f32, { [32, BV] }> =
                if phase == 0i32 { zero_s } else { state_part.load([srow + 3i32, vb]) };
            let zero_o: Tile<f32, { [64, BV] }> = constant(0.0f32, const_shape![64, BV]);
            // Views over the whole segment: chunks are 64-aligned to its start, so chunk c is partition row c.
            let w: Tensor<f32, { [-1, -1] }> = rows_f32(w_ptr, seg_start, seg_len, 4096i32, 4096i32, tok);
            let u: Tensor<f32, { [-1, -1] }> = rows_f32(u_ptr, seg_start, seg_len, 4096i32, 4096i32, tok);
            let qd: Tensor<f32, { [-1, -1] }> = rows_f32(qd_ptr, seg_start, seg_len, 4096i32, 4096i32, tok);
            let kd: Tensor<f32, { [-1, -1] }> = rows_f32(kd_ptr, seg_start, seg_len, 4096i32, 4096i32, tok);
            let p: Tensor<f32, { [-1, -1] }> = rows_f32(p_ptr, seg_start, seg_len, 2048i32, 2048i32, tok);
            let g: Tensor<f32, { [-1, -1] }> = rows_f32(g_ptr, seg_start, seg_len, 32i32, 32i32, tok);
            let num_chunks: i32 = (seg_len + 63i32) / 64i32;
            let out_base: PointerTile<*mut bf16, { [] }> = addptr(pointer_to_tile(out_ptr), seg_start * out_stride);
            let out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[seg_len, 4096i32] };
            let out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[out_stride] };
            let out: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(out_base, out_shape, out_strides, tok) };
            let w_part: Partition<f32, { [64, 32] }> = w.partition(const_shape![64, 32]);
            let u_part: Partition<f32, { [64, BV] }> = u.partition(const_shape![64, BV]);
            let qd_part: Partition<f32, { [64, 32] }> = qd.partition(const_shape![64, 32]);
            let kd_part: Partition<f32, { [64, 32] }> = kd.partition(const_shape![64, 32]);
            let p_part: Partition<f32, { [64, 64] }> = p.partition(const_shape![64, 64]);
            let g_part: Partition<f32, { [64, 1] }> = g.partition(const_shape![64, 1]);
            let mut out_part: PartitionMut<bf16, { [64, BV] }> = unsafe { out.partition_full_mut(const_shape![64, BV]) };
            for c in 0i32..num_chunks {
                let w0: Tile<f32, { [64, 32] }> = w_part.load([c, kc]);
                let ws: Tile<f32, { [64, BV] }> = mma_f32x3(w0, s0, zero_o);
                let w1: Tile<f32, { [64, 32] }> = w_part.load([c, kc + 1i32]);
                let ws: Tile<f32, { [64, BV] }> = mma_f32x3(w1, s1, ws);
                let w2: Tile<f32, { [64, 32] }> = w_part.load([c, kc + 2i32]);
                let ws: Tile<f32, { [64, BV] }> = mma_f32x3(w2, s2, ws);
                let w3: Tile<f32, { [64, 32] }> = w_part.load([c, kc + 3i32]);
                let ws: Tile<f32, { [64, BV] }> = mma_f32x3(w3, s3, ws);
                let ut: Tile<f32, { [64, BV] }> = u_part.load([c, vcol]);
                let v_new: Tile<f32, { [64, BV] }> = ut - ws;

                let q0: Tile<f32, { [64, 32] }> = qd_part.load([c, kc]);
                let o: Tile<f32, { [64, BV] }> = mma_f32x3(q0, s0, zero_o);
                let q1: Tile<f32, { [64, 32] }> = qd_part.load([c, kc + 1i32]);
                let o: Tile<f32, { [64, BV] }> = mma_f32x3(q1, s1, o);
                let q2: Tile<f32, { [64, 32] }> = qd_part.load([c, kc + 2i32]);
                let o: Tile<f32, { [64, BV] }> = mma_f32x3(q2, s2, o);
                let q3: Tile<f32, { [64, 32] }> = qd_part.load([c, kc + 3i32]);
                let o: Tile<f32, { [64, BV] }> = mma_f32x3(q3, s3, o);
                let pt: Tile<f32, { [64, 64] }> = p_part.load([c, h]);
                let o: Tile<f32, { [64, BV] }> = mma_f32x3(pt, v_new, o);
                let o_bf: Tile<bf16, { [64, BV] }> = convert_tile(o);
                out_part.store(o_bf, [c, vcol]);

                // State decay over the chunk: exp(cum_last) = exp(sum of the chunk's g; padded rows are 0).
                let gt: Tile<f32, { [64, 1] }> = g_part.load([c, h]);
                let total: Tile<f32, { [1] }> = reduce_sum(gt, 0i32);
                let cdb: Tile<f32, { [32, BV] }> = exp(total).reshape(const_shape![1, 1]).broadcast(const_shape![32, BV]);
                let k0: Tile<f32, { [64, 32] }> = kd_part.load([c, kc]);
                s0 = mma_f32x3(k0.transpose(), v_new, s0 * cdb);
                let k1: Tile<f32, { [64, 32] }> = kd_part.load([c, kc + 1i32]);
                s1 = mma_f32x3(k1.transpose(), v_new, s1 * cdb);
                let k2: Tile<f32, { [64, 32] }> = kd_part.load([c, kc + 2i32]);
                s2 = mma_f32x3(k2.transpose(), v_new, s2 * cdb);
                let k3: Tile<f32, { [64, 32] }> = kd_part.load([c, kc + 3i32]);
                s3 = mma_f32x3(k3.transpose(), v_new, s3 * cdb);
            }
            if phase == 0i32 {
                let mut sp: PartitionMut<f32, { [32, BV] }> = unsafe { state.partition_full_mut(const_shape![32, BV]) };
                sp.store(s0, [srow, vb]);
                sp.store(s1, [srow + 1i32, vb]);
                sp.store(s2, [srow + 2i32, vb]);
                sp.store(s3, [srow + 3i32, vb]);
            }
        }
    }

    /// Phase A of the chunked gated delta rule (P1), parallel over (chunk, v-head): the two `[64, 64]` operands
    /// that need the whole chunk, with `cum` the in-chunk cumulative sum of g and `D[i, j] = exp(cum_i - cum_j)`:
    /// - `Tinv = (I + A)^-1` for `A = strict_lower((k*beta) k^T ⊙ D)`, the reference's unit lower-triangular
    ///   solve, by block doubling (fla's `solve_tril` merges 16x16 blocks the same way): with `T_b` the inverse
    ///   of the `b`-block-diagonal part of `I + A` and `E_b` the part of `A` inside the `2b`-blocks but outside
    ///   the `b`-blocks, `T_2b = T_b - T_b (E_b T_b)`. From `T_2 = I - E_1`, five doublings (ten `[64, 64]`
    ///   MMAs) reach `T_64`, with no sequential row loop.
    /// - `p = lower_incl((q k^T) ⊙ D)`, the masked intra-chunk attention.
    ///
    /// Grid `(max_chunks, 32, 1)`. q, k: `[T, 2048]` f32; g, beta: `[T, 32]` f32; tinv, p out: `[T, 32 * 64]` f32
    /// (row i of the chunk at row start + i, columns h*64..).
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gdn_chunk_tp<E: ElementType, const SPLIT: i32, EO: ElementType, const SPLIT_O: i32, const FSB: i32>(
        q_ptr: *mut f32,
        k_ptr: *mut f32,
        g_ptr: *mut f32,
        beta_ptr: *mut f32,
        tinv_ptr: *mut f32,
        p_ptr: *mut f32,
        chunks_ptr: *mut i32,
        max_chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h = pid.1;
        let tok: Token = new_token_unordered();
        let (start, len): (i32, i32) = chunk_at(chunks_ptr, max_chunks, pid.0, tok);
        if len > 0i32 {
            let g: Tensor<f32, { [-1, -1] }> = rows_f32(g_ptr, start, len, 32i32, 32i32, tok);
            let beta: Tensor<f32, { [-1, -1] }> = rows_f32(beta_ptr, start, len, 32i32, 32i32, tok);
            let gt: Tile<f32, { [64, 1] }> = g.partition(const_shape![64, 1]).load([0i32, h]);
            let bt: Tile<f32, { [64, 1] }> = beta.partition(const_shape![64, 1]).load([0i32, h]);
            let cum: Tile<f32, { [64, 1] }> = scan_sum(gt, 0i32, reverse::Forward, 0.0f32);
            let idx: Tile<i32, { [64] }> = iota(const_shape![64]);
            let rows: Tile<i32, { [64, 64] }> = idx.reshape(const_shape![64, 1]).broadcast(const_shape![64, 64]);
            let cols: Tile<i32, { [64, 64] }> = idx.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
            let zero: Tile<f32, { [64, 64] }> = constant(0.0f32, const_shape![64, 64]);
            let one: Tile<f32, { [64, 64] }> = constant(1.0f32, const_shape![64, 64]);
            let cum_i: Tile<f32, { [64, 64] }> = cum.broadcast(const_shape![64, 64]);
            let cum_j: Tile<f32, { [64, 64] }> = cum.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
            let d: Tile<f32, { [64, 64] }> = exp(cum_i - cum_j);
            // (k*beta) k^T and q k^T over four [64, 32] column blocks of the k-head (keeps operands small).
            let k: Tensor<f32, { [-1, -1] }> = rows_f32(k_ptr, start, len, 2048i32, 2048i32, tok);
            let q: Tensor<f32, { [-1, -1] }> = rows_f32(q_ptr, start, len, 2048i32, 2048i32, tok);
            let k_part: Partition<f32, { [64, 32] }> = k.partition(const_shape![64, 32]);
            let q_part: Partition<f32, { [64, 32] }> = q.partition(const_shape![64, 32]);
            let kc = (h / 2i32) * 4i32;
            let bs: Tile<f32, { [64, 32] }> = bt.broadcast(const_shape![64, 32]);
            let mut kk: Tile<f32, { [64, 64] }> = zero;
            let mut qk: Tile<f32, { [64, 64] }> = zero;
            for j in 0i32..4i32 {
                let kj: Tile<f32, { [64, 32] }> = k_part.load([0i32, kc + j]);
                let kjt: Tile<f32, { [32, 64] }> = kj.transpose();
                kk = mma_p::<E, SPLIT, 64, 64, 32>(kj * bs, kjt, kk);
                let qj: Tile<f32, { [64, 32] }> = q_part.load([0i32, kc + j]);
                qk = mma_p::<EO, SPLIT_O, 64, 64, 32>(qj, kjt, qk);
            }
            let p: Tile<f32, { [64, 64] }> = select(ge_tile(rows, cols), qk * d, zero);
            let p_out: Tensor<f32, { [-1, -1] }> = rows_f32(p_ptr, start, len, 2048i32, 2048i32, tok);
            let mut p_part: PartitionMut<f32, { [64, 64] }> = unsafe { p_out.partition_full_mut(const_shape![64, 64]) };
            p_part.store(p, [0i32, h]);
            let a: Tile<f32, { [64, 64] }> = select(gt_tile(rows, cols), kk * d, zero);

            // With x = row ^ col, (i, j) share a b-block iff x < b (b a power of two).
            let x: Tile<i32, { [64, 64] }> = xori(rows, cols);
            let ident: Tile<f32, { [64, 64] }> = select(eq_tile(rows, cols), one, zero);
            // 1. Inverse of the FSB-block-diagonal part by forward substitution, all 64 / FSB blocks at once: step
            //    i updates row i of every block, T[r, :] -= sum_j A[r, j] T[j, :]. Within a block at most one
            //    row is selected and T is block-diagonal, so one column reduction serves every block.
            let fsb: Tile<i32, { [64, 64] }> = broadcast_scalar(FSB, const_shape![64, 64]);
            let same: Tile<bool, { [64, 64] }> = lt_tile(x, fsb);
            let a_blk: Tile<f32, { [64, 64] }> = select(same, a, zero);
            let in_blk: Tile<i32, { [64, 64] }> = andi(rows, broadcast_scalar(FSB - 1i32, const_shape![64, 64]));
            let mut tinv: Tile<f32, { [64, 64] }> = ident;
            for i in 1i32..FSB {
                let sel: Tile<bool, { [64, 64] }> = eq_tile(in_blk, broadcast_scalar(i, const_shape![64, 64]));
                let a_sel: Tile<f32, { [64] }> = reduce_sum(select(sel, a_blk, zero), 0i32);
                let contrib: Tile<f32, { [64] }> =
                    reduce_sum(a_sel.reshape(const_shape![64, 1]).broadcast(const_shape![64, 64]) * tinv, 0i32);
                let cb: Tile<f32, { [64, 64] }> = contrib.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
                tinv = select(sel, select(same, tinv - cb, tinv), tinv);
            }
            // 2. Block doubling (fla's `solve_tril` merges blocks the same way): with E_b the part of A inside the
            //    2b-blocks but outside the b-blocks (b <= x < 2b, i.e. (x & -b) == b), T_2b = T_b - T_b (E_b T_b).
            let mut b: i32 = FSB;
            for _step in 0i32..6i32 {
                if b < 64i32 {
                    let bt_: Tile<i32, { [64, 64] }> = broadcast_scalar(b, const_shape![64, 64]);
                    let nb: Tile<i32, { [64, 64] }> = broadcast_scalar(0i32 - b, const_shape![64, 64]);
                    let e: Tile<f32, { [64, 64] }> = select(eq_tile(andi(x, nb), bt_), a, zero);
                    let m: Tile<f32, { [64, 64] }> = mma_p::<E, SPLIT, 64, 64, 64>(e, tinv, zero);
                    tinv = mma_p::<E, SPLIT, 64, 64, 64>(negf(tinv), m, tinv);
                }
                b = b * 2i32;
            }
            let out: Tensor<f32, { [-1, -1] }> = rows_f32(tinv_ptr, start, len, 2048i32, 2048i32, tok);
            let mut part: PartitionMut<f32, { [64, 64] }> = unsafe { out.partition_full_mut(const_shape![64, 64]) };
            part.store(tinv, [0i32, h]);
        }
    }

    /// Phase B of the chunked gated delta rule (P1): the sequential inter-chunk scan of one segment, one v-head
    /// and one block of BV value columns. It recomputes every per-chunk operand from q, k, v, g, beta and phase
    /// A's `Tinv` and `p` instead of reading stored WY intermediates. Per chunk, with `cum` the in-chunk
    /// cumulative sum of g:
    /// ```text
    /// v_new = Tinv ((v*beta) - (k*beta*exp(cum)) S)        (= u - w S of the reference, re-associated)
    /// o     = (q*exp(cum)) S + p v_new
    /// S     = exp(sum g) S + (k*exp(sum_{j>i} g_j))^T v_new
    /// ```
    /// Only segments whose table `phase` equals `phase` run. Prefixes (phase 0) start from zero and write their
    /// final state to `state[slot]`; suffixes (phase 1) start from `state[slot]` and write nothing.
    /// Grid `(max_segs, 32, 128 / BV)`. q, k: `[T, 2048]` f32; v: `[T, 4096]` bf16 view (row stride
    /// `v_stride`); g, beta: `[T, 32]` f32; tinv, p: `[T, 32 * 64]` f32; state: `[slots, 32, 128, 128]` f32;
    /// out: `[T, 4096]` bf16 with row stride `out_stride`.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gdn_chunk_scan_fused<
        E: ElementType,
        const SPLIT: i32,
        EO: ElementType,
        const SPLIT_O: i32,
        const BV: i32,
    >(
        q_ptr: *mut f32,
        k_ptr: *mut f32,
        v_ptr: *mut bf16,
        v_stride: i32,
        g_ptr: *mut f32,
        beta_ptr: *mut f32,
        tinv_ptr: *mut f32,
        p_ptr: *mut f32,
        state_ptr: *mut f32,
        out_ptr: *mut bf16,
        out_stride: i32,
        segs_ptr: *mut i32,
        max_segs: i32,
        num_slots: i32,
        phase: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h = pid.1;
        let hk = h / 2i32;
        let vb = pid.2;
        let vcol = h * (128i32 / BV) + vb;
        let tok: Token = new_token_unordered();
        let tab_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(segs_ptr);
        let tab_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[max_segs * 4i32] };
        let tab_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let tab: Tensor<i32, { [-1] }> = unsafe { make_tensor_view(tab_base, tab_shape, tab_strides, tok) };
        let tab_part: Partition<i32, { [1] }> = tab.partition(const_shape![1]);
        let seg_start: i32 = table_at(&tab_part, pid.0 * 4i32);
        let seg_len: i32 = table_at(&tab_part, pid.0 * 4i32 + 1i32);
        let seg_phase: i32 = table_at(&tab_part, pid.0 * 4i32 + 2i32);
        let slot: i32 = table_at(&tab_part, pid.0 * 4i32 + 3i32);
        if seg_phase == phase {
            // state viewed as [slots * 32 * 128, 128] in [32, BV] tiles, held as four row blocks (k rows
            // 32j..32j+32); (slot, h) rows start at tile row (slot * 32 + h) * 4. The [64, 128] operands are
            // streamed as [64, 32] column blocks, so no MMA operand (with its TF32 hi/lo split) is larger than
            // [64, 64] and the kernel stays in registers.
            let state: Tensor<f32, { [-1, -1] }> = rows_f32(state_ptr, 0i32, num_slots * 4096i32, 128i32, 128i32, tok);
            let srow: i32 = (slot * 32i32 + h) * 4i32;
            let state_part: Partition<f32, { [32, BV] }> = state.partition(const_shape![32, BV]);
            let zero_s: Tile<f32, { [32, BV] }> = constant(0.0f32, const_shape![32, BV]);
            let mut s0: Tile<f32, { [32, BV] }> = if phase == 0i32 { zero_s } else { state_part.load([srow, vb]) };
            let mut s1: Tile<f32, { [32, BV] }> = if phase == 0i32 { zero_s } else { state_part.load([srow + 1i32, vb]) };
            let mut s2: Tile<f32, { [32, BV] }> = if phase == 0i32 { zero_s } else { state_part.load([srow + 2i32, vb]) };
            let mut s3: Tile<f32, { [32, BV] }> = if phase == 0i32 { zero_s } else { state_part.load([srow + 3i32, vb]) };
            let zero_o: Tile<f32, { [64, BV] }> = constant(0.0f32, const_shape![64, BV]);
            let q: Tensor<f32, { [-1, -1] }> = rows_f32(q_ptr, seg_start, seg_len, 2048i32, 2048i32, tok);
            let k: Tensor<f32, { [-1, -1] }> = rows_f32(k_ptr, seg_start, seg_len, 2048i32, 2048i32, tok);
            let tinv: Tensor<f32, { [-1, -1] }> = rows_f32(tinv_ptr, seg_start, seg_len, 2048i32, 2048i32, tok);
            let p: Tensor<f32, { [-1, -1] }> = rows_f32(p_ptr, seg_start, seg_len, 2048i32, 2048i32, tok);
            let g: Tensor<f32, { [-1, -1] }> = rows_f32(g_ptr, seg_start, seg_len, 32i32, 32i32, tok);
            let beta: Tensor<f32, { [-1, -1] }> = rows_f32(beta_ptr, seg_start, seg_len, 32i32, 32i32, tok);
            let v_base: PointerTile<*mut bf16, { [] }> = addptr(pointer_to_tile(v_ptr), seg_start * v_stride);
            let v_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[seg_len, 4096i32] };
            let v_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[v_stride] };
            let v: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(v_base, v_shape, v_strides, tok) };
            let out_base: PointerTile<*mut bf16, { [] }> = addptr(pointer_to_tile(out_ptr), seg_start * out_stride);
            let out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[seg_len, 4096i32] };
            let out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[out_stride] };
            let out: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(out_base, out_shape, out_strides, tok) };
            let q_part: Partition<f32, { [64, 32] }> = q.partition(const_shape![64, 32]);
            let k_part: Partition<f32, { [64, 32] }> = k.partition(const_shape![64, 32]);
            let tinv_part: Partition<f32, { [64, 64] }> = tinv.partition(const_shape![64, 64]);
            let p_part: Partition<f32, { [64, 64] }> = p.partition(const_shape![64, 64]);
            let g_part: Partition<f32, { [64, 1] }> = g.partition(const_shape![64, 1]);
            let beta_part: Partition<f32, { [64, 1] }> = beta.partition(const_shape![64, 1]);
            let v_part: Partition<bf16, { [64, BV] }> = v.partition(const_shape![64, BV]);
            let mut out_part: PartitionMut<bf16, { [64, BV] }> = unsafe { out.partition_full_mut(const_shape![64, BV]) };
            let kc = hk * 4i32; // first [*, 32] column block of k-head hk in a [T, 2048] buffer
            let num_chunks: i32 = (seg_len + 63i32) / 64i32;
            for c in 0i32..num_chunks {
                // Views end at the segment end, so rows past it load as zeros (g = 0: no decay).
                let gt: Tile<f32, { [64, 1] }> = g_part.load([c, h]);
                let bt: Tile<f32, { [64, 1] }> = beta_part.load([c, h]);
                let cum: Tile<f32, { [64, 1] }> = scan_sum(gt, 0i32, reverse::Forward, 0.0f32);
                let ecum: Tile<f32, { [64, 1] }> = exp(cum);
                let kbe_s: Tile<f32, { [64, 32] }> = (bt * ecum).broadcast(const_shape![64, 32]);
                let qd_s: Tile<f32, { [64, 32] }> = ecum.broadcast(const_shape![64, 32]);

                // x = (k*beta*exp(cum)) S, then v_new, then o = (q*exp(cum)) S + p v_new (x first: fewer live tiles).
                let k0: Tile<f32, { [64, 32] }> = k_part.load([c, kc]);
                let x: Tile<f32, { [64, BV] }> = mma_p::<E, SPLIT, 64, BV, 32>(k0 * kbe_s, s0, zero_o);
                let k1: Tile<f32, { [64, 32] }> = k_part.load([c, kc + 1i32]);
                let x: Tile<f32, { [64, BV] }> = mma_p::<E, SPLIT, 64, BV, 32>(k1 * kbe_s, s1, x);
                let k2: Tile<f32, { [64, 32] }> = k_part.load([c, kc + 2i32]);
                let x: Tile<f32, { [64, BV] }> = mma_p::<E, SPLIT, 64, BV, 32>(k2 * kbe_s, s2, x);
                let k3: Tile<f32, { [64, 32] }> = k_part.load([c, kc + 3i32]);
                let x: Tile<f32, { [64, BV] }> = mma_p::<E, SPLIT, 64, BV, 32>(k3 * kbe_s, s3, x);
                let vt: Tile<bf16, { [64, BV] }> = v_part.load([c, vcol]);
                let vt: Tile<f32, { [64, BV] }> = convert_tile(vt);
                let y: Tile<f32, { [64, BV] }> = vt * bt.broadcast(const_shape![64, BV]) - x;
                let tt: Tile<f32, { [64, 64] }> = tinv_part.load([c, h]);
                let v_new: Tile<f32, { [64, BV] }> = mma_p::<E, SPLIT, 64, BV, 64>(tt, y, zero_o);
                let pt: Tile<f32, { [64, 64] }> = p_part.load([c, h]);
                let o: Tile<f32, { [64, BV] }> = mma_p::<EO, SPLIT_O, 64, BV, 64>(pt, v_new, zero_o);
                let q0: Tile<f32, { [64, 32] }> = q_part.load([c, kc]);
                let o: Tile<f32, { [64, BV] }> = mma_p::<EO, SPLIT_O, 64, BV, 32>(q0 * qd_s, s0, o);
                let q1: Tile<f32, { [64, 32] }> = q_part.load([c, kc + 1i32]);
                let o: Tile<f32, { [64, BV] }> = mma_p::<EO, SPLIT_O, 64, BV, 32>(q1 * qd_s, s1, o);
                let q2: Tile<f32, { [64, 32] }> = q_part.load([c, kc + 2i32]);
                let o: Tile<f32, { [64, BV] }> = mma_p::<EO, SPLIT_O, 64, BV, 32>(q2 * qd_s, s2, o);
                let q3: Tile<f32, { [64, 32] }> = q_part.load([c, kc + 3i32]);
                let o: Tile<f32, { [64, BV] }> = mma_p::<EO, SPLIT_O, 64, BV, 32>(q3 * qd_s, s3, o);
                let o_bf: Tile<bf16, { [64, BV] }> = convert_tile(o);
                out_part.store(o_bf, [c, vcol]);

                // S = exp(sum g) S + (k*exp(sum_{j>i} g_j))^T v_new. The suffix sum comes from a reverse scan:
                // cum_last - cum_i cancels catastrophically for strongly decaying heads.
                let rcum: Tile<f32, { [64, 1] }> = scan_sum(gt, 0i32, reverse::Reverse, 0.0f32);
                let kd_s: Tile<f32, { [64, 32] }> = exp(rcum - gt).broadcast(const_shape![64, 32]);
                let total: Tile<f32, { [1] }> = reduce_sum(gt, 0i32);
                let decay: Tile<f32, { [32, BV] }> = exp(total).reshape(const_shape![1, 1]).broadcast(const_shape![32, BV]);
                let kd0: Tile<f32, { [64, 32] }> = k_part.load([c, kc]);
                s0 = mma_p::<E, SPLIT, 32, BV, 64>((kd0 * kd_s).transpose(), v_new, s0 * decay);
                let kd1: Tile<f32, { [64, 32] }> = k_part.load([c, kc + 1i32]);
                s1 = mma_p::<E, SPLIT, 32, BV, 64>((kd1 * kd_s).transpose(), v_new, s1 * decay);
                let kd2: Tile<f32, { [64, 32] }> = k_part.load([c, kc + 2i32]);
                s2 = mma_p::<E, SPLIT, 32, BV, 64>((kd2 * kd_s).transpose(), v_new, s2 * decay);
                let kd3: Tile<f32, { [64, 32] }> = k_part.load([c, kc + 3i32]);
                s3 = mma_p::<E, SPLIT, 32, BV, 64>((kd3 * kd_s).transpose(), v_new, s3 * decay);
            }
            if phase == 0i32 {
                let mut sp: PartitionMut<f32, { [32, BV] }> = unsafe { state.partition_full_mut(const_shape![32, BV]) };
                sp.store(s0, [srow, vb]);
                sp.store(s1, [srow + 1i32, vb]);
                sp.store(s2, [srow + 2i32, vb]);
                sp.store(s3, [srow + 3i32, vb]);
            }
        }
    }

    /// Gated RMSNorm per 128-wide v-head: `bf16(bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w) * silu(z))`, with the
    /// normalisation and the gate in fp32 and the weight product in bf16 (`Qwen3_5RMSNormGated`).
    ///
    /// Grid `(ceil(T / BR), 32, 1)`. x: `[T, 4096]` bf16 (row stride `x_stride`); z: `[T, 4096]` bf16 view (row
    /// stride `z_stride`); w: `[128]` bf16; out: `[T, 4096]` bf16 (row stride `out_stride`).
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gated_rmsnorm<const BR: i32>(
        x_ptr: *mut bf16,
        x_stride: i32,
        z_ptr: *mut bf16,
        z_stride: i32,
        w_ptr: *mut bf16,
        out_ptr: *mut bf16,
        out_stride: i32,
        tokens: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tok: Token = new_token_unordered();
        let x_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(x_ptr);
        let x_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 4096i32] };
        let x_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[x_stride] };
        let x: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(x_base, x_shape, x_strides, tok) };
        let z_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(z_ptr);
        let z_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 4096i32] };
        let z_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[z_stride] };
        let z: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(z_base, z_shape, z_strides, tok) };
        let w_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(w_ptr);
        let w_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[128i32] };
        let w_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let w: Tensor<bf16, { [-1] }> = unsafe { make_tensor_view(w_base, w_shape, w_strides, tok) };
        let out_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
        let out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[tokens, 4096i32] };
        let out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[out_stride] };
        let out: Tensor<bf16, { [-1, -1] }> = unsafe { make_tensor_view(out_base, out_shape, out_strides, tok) };
        let xt: Tile<f32, { [BR, 128] }> = convert_tile(x.partition(const_shape![BR, 128]).load([pid.0, pid.1]));
        let zt: Tile<f32, { [BR, 128] }> = convert_tile(z.partition(const_shape![BR, 128]).load([pid.0, pid.1]));
        let w_row: Tile<bf16, { [1, 128] }> = w.partition(const_shape![128]).load([0i32]).reshape(const_shape![1, 128]);
        let w_row: Tile<f32, { [1, 128] }> = convert_tile(w_row);
        let wt: Tile<f32, { [BR, 128] }> = w_row.broadcast(const_shape![BR, 128]);
        let ss: Tile<f32, { [BR] }> = reduce_sum(xt * xt, 1i32);
        let n128: Tile<f32, { [BR] }> = constant(128.0f32, const_shape![BR]);
        let var: Tile<f32, { [BR, 1] }> = true_div(ss, n128).reshape(const_shape![BR, 1]);
        let inv: Tile<f32, { [BR, 1] }> = rsqrt(var + broadcast_scalar(eps, const_shape![BR, 1]), ftz::Disabled);
        let normed: Tile<bf16, { [BR, 128] }> = convert_tile(xt * inv.broadcast(const_shape![BR, 128]));
        let normed: Tile<f32, { [BR, 128] }> = convert_tile(normed);
        let weighted: Tile<bf16, { [BR, 128] }> = convert_tile(normed * wt);
        let weighted: Tile<f32, { [BR, 128] }> = convert_tile(weighted);
        let y: Tile<f32, { [BR, 128] }> = weighted * silu_f32(zt);
        let mut out_part: PartitionMut<bf16, { [BR, 128] }> = unsafe { out.partition_full_mut(const_shape![BR, 128]) };
        let y_bf: Tile<bf16, { [BR, 128] }> = convert_tile(y);
        out_part.store(y_bf, [pid.0, pid.1]);
    }
}
