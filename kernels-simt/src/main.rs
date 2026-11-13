//! cuda-oxide SIMT kernels for sysone's GatedDeltaNet mixer (docs/DESIGN.md §6.4, docs/G1.md).
//!
//! This crate is device code only. `build.sh` compiles it with `cargo oxide` to PTX for one architecture and
//! writes `artifacts/gdn_simt.<arch>.ptx` plus a manifest (source hash, PTX hash); sysone embeds the checked-in
//! PTX and loads it with cuda-core at run time (`sysone::kernels::simt`). Every kernel takes raw device pointers
//! and i32 scalars, so the host side launches it with an untyped argument packet: keep the parameter lists here
//! and in `src/kernels/gdn/simt.rs` in sync.
//!
//! # Numerics
//!
//! The chunked gated delta rule (chunk 64, WY/UT form) as in `torch_chunk_gated_delta_rule`, with the same
//! operands as the cuTile kernels (`src/kernels/gdn/tile.rs`), but on the bf16 tensor cores (`mma.sync`
//! m16n8k16, fp32 accumulate) without losing fp32 accuracy:
//! - q and k are the conv output (bf16, exact) times a per-row scale (`rsqrt(sum x^2 + 1e-6)`, times 128^-0.5 for
//!   q), so `k k^T`, `q k^T`, `k S` and `q S` take the raw bf16 rows as one exact operand and apply the scales in
//!   fp32 afterwards;
//! - fp32 operands (the state S, Tinv, P, the WY residuals) are split `x = hi + lo` into two bf16 numbers
//!   (relative residual <= 2^-17); products with one exact operand use 2 MMAs (hi, lo), fp32 x fp32 products 3
//!   (hi hi + hi lo + lo hi);
//! - the 64x64 unit-lower-triangular inverse is a blocked (16x16) forward substitution in fp32 SIMT.
//!
//! # Fragment layouts (PTX `mma.m16n8k16.row.col`, lane = 4 g + t)
//! A (16x16): a[0] = (row g, cols 2t, 2t+1), a[1] = (g+8, 2t..), a[2] = (g, 2t+8..), a[3] = (g+8, 2t+8..).
//! B (16x8, k x n): b[0] = (k 2t, 2t+1; n g), b[1] = (k 2t+8, 2t+9; n g).
//! C (16x8): c[0], c[1] = (row g, cols 2t, 2t+1); c[2], c[3] = (row g+8, cols 2t, 2t+1).
//! Two adjacent C tiles (n 0..8, 8..16) are exactly one A fragment of a 16x16 operand, which is how the state and
//! the per-chunk intermediates flow from one product into the next without shared memory.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use core::ptr::addr_of_mut;
use cuda_device::wmma::mma_m16n8k16_f32_bf16;
use cuda_device::{SharedArray, cuda_module, kernel, thread, warp};

/// Chunk length.
const CH: usize = 64;
/// Row stride (u32 words) of a [64, 128] bf16 tile in shared memory: 64 data words + 4 padding, so that the
/// fragment loads of 8 rows x 4 words hit 32 distinct banks.
const KW: usize = 68;
/// Row stride (f32) of a [64, 64] f32 tile in shared memory.
const AW: usize = 65;
/// Conv output layout: q at column 0, k at 2048, v at 4096 (bf16 columns).
const K_OFF: usize = 2048;
const V_OFF: usize = 4096;
const L2_EPS: f32 = 1e-6;
const Q_SCALE: f32 = 0.088_388_35; // 128^-0.5

/// Full unrolling with a constant index per copy (cuda-oxide's `opt -O2` does not unroll loops around the
/// `mma.sync` inline asm, and a register array indexed by a loop variable is demoted to local memory).
macro_rules! unroll {
    ($i:ident in 0..4 $body:block) => { unroll!(@ $i [0, 1, 2, 3] $body) };
    ($i:ident in 0..8 $body:block) => { unroll!(@ $i [0, 1, 2, 3, 4, 5, 6, 7] $body) };
    ($i:ident in 0..16 $body:block) => { unroll!(@ $i [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] $body) };
    (@ $i:ident [$($n:literal),*] $body:block) => { $( { #[allow(unused_variables)] let $i: usize = $n; $body } )* };
}

#[inline(always)]
fn bf16_rne(x: f32) -> u32 {
    let b = x.to_bits();
    b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16
}

/// `x = hi + lo` with hi, lo bf16 (as the low 16 bits).
#[inline(always)]
fn split(x: f32) -> (u32, u32) {
    let h = bf16_rne(x);
    let l = bf16_rne(x - f32::from_bits(h << 16));
    (h, l)
}

/// Pack two f32 as bf16x2 (lo in bits 0..16) split into (hi pair, lo pair).
#[inline(always)]
fn split2(x0: f32, x1: f32) -> (u32, u32) {
    let (h0, l0) = split(x0);
    let (h1, l1) = split(x1);
    (h0 | (h1 << 16), l0 | (l1 << 16))
}

#[inline(always)]
fn lo_bf16(w: u32) -> f32 {
    f32::from_bits(w << 16)
}

#[inline(always)]
fn hi_bf16(w: u32) -> f32 {
    f32::from_bits(w & 0xffff_0000)
}

#[inline(always)]
fn mma(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    unsafe { mma_m16n8k16_f32_bf16(c, a, b) }
}

/// Load the 64 x 128 bf16 tile at `src` (row stride `stride` bf16) into `dst` ([64][KW] words), zeroing rows
/// `>= len`. All 128 threads participate.
#[inline(always)]
unsafe fn load_tile(dst: *mut u32, src: *const u16, stride: usize, len: usize, tid: usize) {
    let mut it = 0usize;
    while it < 32 {
        let e = tid + 128 * it;
        let row = e >> 6;
        let wd = e & 63;
        let v = if row < len { unsafe { *(src.add(row * stride + 2 * wd) as *const u32) } } else { 0 };
        unsafe { *dst.add(row * KW + wd) = v };
        it += 1;
    }
}

/// Inverse l2 norm of row `tid / 2` of a [64][KW] tile, reduced over the two threads of a row (all 128 threads
/// call this; both threads of the pair get the result).
#[inline(always)]
unsafe fn row_inv_norm(tile: *const u32, tid: usize) -> f32 {
    let row = tid >> 1;
    let half = tid & 1;
    let mut ss = 0f32;
    let mut w = 0usize;
    while w < 32 {
        let v = unsafe { *tile.add(row * KW + half * 32 + w) };
        let a = lo_bf16(v);
        let b = hi_bf16(v);
        ss += a * a + b * b;
        w += 1;
    }
    ss += warp::shuffle_xor_f32(ss, 1);
    1.0 / (ss + L2_EPS).sqrt()
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Integration probe: `out[i] = in[i] * 3 + i` for `i < n` (one element per thread).
    #[kernel]
    pub unsafe fn simt_probe(input: *const u32, out: *mut u32, n: i32) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if i < n {
            unsafe {
                *out.add(i as usize) = (*input.add(i as usize)).wrapping_mul(3).wrapping_add(i as u32);
            }
        }
    }

    /// The GDN gates (the part of `gdn_prep` the SIMT scan needs; q and k are normalised inside the scan):
    /// `g = -exp(A_log) * softplus(a + dt_bias)` (torch softplus, threshold 20) and `beta = bf16(sigmoid(b))`, f32
    /// `[T, 32]` each. Grid `(ceil(T * 32 / 256), 1, 1)`, block 256, one (row, head) per thread.
    /// a, b: `[T, 32]` bf16 views (row strides `a_stride`, `b_stride`); a_log, dt_bias: `[32]` bf16.
    #[kernel]
    pub unsafe fn gdn_simt_gates(
        a: *const u16,
        a_stride: i32,
        b: *const u16,
        b_stride: i32,
        a_log: *const u16,
        dt_bias: *const u16,
        g: *mut f32,
        beta: *mut f32,
        tokens: i32,
    ) {
        let e = (thread::blockIdx_x() * 256 + thread::threadIdx_x()) as usize;
        if e >= tokens as usize * 32 {
            return;
        }
        let row = e >> 5;
        let h = e & 31;
        unsafe {
            let av = lo_bf16(*a.add(row * a_stride as usize + h) as u32);
            let bv = lo_bf16(*b.add(row * b_stride as usize + h) as u32);
            let neg_a = -(lo_bf16(*a_log.add(h) as u32)).exp();
            let x = av + lo_bf16(*dt_bias.add(h) as u32);
            // log1p(exp(x)) with the exact `log(1+u) * u / ((1+u) - 1)` correction (as the cuTile prep).
            let u = x.exp();
            let w = 1.0 + u;
            let dw = w - 1.0;
            let l1p = if dw == 0.0 { u } else { w.ln() * (u / dw) };
            let sp = if x > 20.0 { x } else { l1p };
            *g.add(e) = neg_a * sp;
            let sig = 1.0 / (1.0 + (-bv).exp());
            *beta.add(e) = f32::from_bits(bf16_rne(sig) << 16);
        }
    }

    /// Phase A of the chunked gated delta rule, parallel over (chunk, v-head): `Tinv = (I + A)^-1` with
    /// `A = strict_lower((k*beta) k^T ⊙ exp(cum_i - cum_j))`, `cum` the in-chunk inclusive cumulative sum of g.
    ///
    /// Grid `(32, max_chunks, 1)` (the two v-heads of a k-head are adjacent, so k is read from DRAM once), block
    /// 128. conv: `[T, 8192]` bf16 (row stride `conv_stride`); g, beta:
    /// `[T, 32]` f32; tinv out: `[T, 32 * 64]` f32 (row i of the chunk at row start + i, columns h*64..; only the
    /// chunk's `len` rows and the 16x16 blocks on and below the diagonal are written; the scan reads only those). chunks: `[max_chunks, 4]` (start, len, ..).
    #[kernel]
    pub unsafe fn gdn_simt_tinv(
        conv: *const u16,
        conv_stride: i32,
        g: *const f32,
        beta: *const f32,
        tinv: *mut f32,
        chunks: *const i32,
        max_chunks: i32,
    ) {
        static mut KB: SharedArray<u32, 4352, 16> = SharedArray::UNINIT; // [64][KW]; reused for T [64][AW]
        static mut AM: SharedArray<f32, 4160> = SharedArray::UNINIT; // [64][AW]
        static mut MS: SharedArray<f32, 768> = SharedArray::UNINIT; // 3 x [16][16]
        static mut SC: SharedArray<f32, 192> = SharedArray::UNINIT; // gv[64] | bk[64] = beta*sk | sk[64]
        let h = thread::blockIdx_x() as usize;
        let c = thread::blockIdx_y() as usize;
        if c >= max_chunks as usize {
            return;
        }
        let start = unsafe { *chunks.add(c * 4) } as usize;
        let len = unsafe { *chunks.add(c * 4 + 1) } as usize;
        if len == 0 {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let w = tid >> 5;
        let (gq, tq) = (lane >> 2, lane & 3);
        let kb = addr_of_mut!(KB) as *mut u32;
        let am = addr_of_mut!(AM) as *mut f32;
        let ms = addr_of_mut!(MS) as *mut f32;
        let sc = addr_of_mut!(SC) as *mut f32;
        let stride = conv_stride as usize;
        let hk = h / 2;
        unsafe {
            load_tile(kb, conv.add(start * stride + K_OFF + hk * 128), stride, len, tid);
            if tid < CH {
                let valid = tid < len;
                *sc.add(tid) = if valid { *g.add((start + tid) * 32 + h) } else { 0.0 };
                *sc.add(64 + tid) = if valid { *beta.add((start + tid) * 32 + h) } else { 0.0 };
            }
        }
        thread::sync_threads();
        let inv = unsafe { row_inv_norm(kb, tid) };
        if tid & 1 == 0 {
            unsafe {
                let r = tid >> 1;
                *sc.add(128 + r) = inv;
                *sc.add(64 + r) *= inv; // beta * sk
            }
        }
        if tid == 0 {
            // In-chunk inclusive cumulative sum of g, in place.
            let mut acc = 0f32;
            let mut i = 0usize;
            while i < CH {
                unsafe {
                    acc += *sc.add(i);
                    *sc.add(i) = acc;
                }
                i += 1;
            }
        }
        thread::sync_threads();

        // Gram rows 16w..16w+16 (only column tiles on or below the diagonal block): G = kb kb^T, exact products.
        let i0 = 16 * w;
        let mut acc = [[0f32; 4]; 8];
        unroll!(ks in 0..8 {
            let a = unsafe {
                [
                    *kb.add((i0 + gq) * KW + ks * 8 + tq),
                    *kb.add((i0 + gq + 8) * KW + ks * 8 + tq),
                    *kb.add((i0 + gq) * KW + ks * 8 + 4 + tq),
                    *kb.add((i0 + gq + 8) * KW + ks * 8 + 4 + tq),
                ]
            };
            unroll!(n in 0..8 {
                if n <= 2 * w + 1 {
                    let b = unsafe { [*kb.add((8 * n + gq) * KW + ks * 8 + tq), *kb.add((8 * n + gq) * KW + ks * 8 + 4 + tq)] };
                    acc[n] = mma(acc[n], a, b);
                }
            });
        });
        unroll!(n in 0..8 {
            if n <= 2 * w + 1 {
                unroll!(r in 0..4 {
                    let i = i0 + gq + 8 * (r >> 1);
                    let j = 8 * n + 2 * tq + (r & 1);
                    let v = if i > j {
                        unsafe { *sc.add(64 + i) * *sc.add(128 + j) * acc[n][r] * (*sc.add(i) - *sc.add(j)).exp() }
                    } else {
                        0.0
                    };
                    unsafe { *am.add(i * AW + j) = v };
                });
            }
        });
        thread::sync_threads();

        // T = (I + A)^-1, blocked by 16. Diagonal blocks: warp b inverts block b, lane l < 16 owns column l.
        let tm = kb; // T as f32 bits, [64][AW]
        if lane < 16 {
            let b0 = 16 * w;
            let mut x = [0f32; 16];
            unroll!(i in 0..16 {
                let mut s = if i == lane { 1.0f32 } else { 0.0 };
                unroll!(m in 0..16 {
                    if m < i {
                        s -= unsafe { *am.add((b0 + i) * AW + b0 + m) } * x[m];
                    }
                });
                x[i] = s;
                unsafe { *tm.add((b0 + i) * AW + b0 + lane) = s.to_bits() };
            });
        }
        thread::sync_threads();
        // Off-diagonal blocks, one block row at a time: T_IJ = -T_II sum_{K=J}^{I-1} A_IK T_KJ.
        let mut bi = 1usize;
        while bi < 4 {
            let outs = bi * 256;
            let mut o = tid;
            while o < outs {
                let bj = o >> 8;
                let r = (o >> 4) & 15;
                let cc = o & 15;
                let mut s = 0f32;
                let mut bk = bj;
                while bk < bi {
                    let mut k = 0usize;
                    while k < 16 {
                        s += unsafe {
                            *am.add((16 * bi + r) * AW + 16 * bk + k) * f32::from_bits(*tm.add((16 * bk + k) * AW + 16 * bj + cc))
                        };
                        k += 1;
                    }
                    bk += 1;
                }
                unsafe { *ms.add(o) = s };
                o += 128;
            }
            thread::sync_threads();
            let mut o = tid;
            while o < outs {
                let bj = o >> 8;
                let r = (o >> 4) & 15;
                let cc = o & 15;
                let mut s = 0f32;
                let mut k = 0usize;
                while k < 16 {
                    s += unsafe { f32::from_bits(*tm.add((16 * bi + r) * AW + 16 * bi + k)) * *ms.add(bj * 256 + k * 16 + cc) };
                    k += 1;
                }
                unsafe { *tm.add((16 * bi + r) * AW + 16 * bj + cc) = (-s).to_bits() };
                o += 128;
            }
            thread::sync_threads();
            bi += 1;
        }
        let mut e = tid;
        while e < len * CH {
            let i = e >> 6;
            let j = e & 63;
            if (j >> 4) <= (i >> 4) {
                unsafe { *tinv.add((start + i) * 2048 + h * 64 + j) = f32::from_bits(*tm.add(i * AW + j)) };
            }
            e += 128;
        }
    }

    /// Phase B of the chunked gated delta rule: the sequential inter-chunk scan of one segment, one v-head and
    /// one block of 64 value columns, recomputing every per-chunk operand from the conv output and `tinv` instead
    /// of storing them. Only segments whose table `phase` equals `phase` run.
    ///
    /// Per chunk (rows i, j of the chunk; `cum` the in-chunk cumulative sum of g; `rc_i = sum_{j>i} g_j`):
    /// ```text
    /// X = k S,  Y = q S                                   (k, q: normalised rows; S: [128 k, 128 v] state)
    /// v_new = Tinv (beta ⊙ (v - e^cum ⊙ X))               (= u - w S of the WY form)
    /// o = e^cum ⊙ Y + P v_new,  P = (q k^T ⊙ e^(cum_i - cum_j)) on and below the diagonal
    /// S = e^(sum g) S + k^T (e^rc ⊙ v_new)
    /// ```
    /// computed transposed (`S^T` as the MMA accumulator, 16 value rows per warp), so the C fragments of one
    /// product are the A operand of the next. Prefixes (phase 0) start from zero and write their final state to
    /// `state[slot]`; suffixes (phase 1) start from `state[slot]` (their parent's) and write nothing.
    ///
    /// Grid `(64, max_segs, 1)` with `blockIdx.x = 2 h + value block` (the four CTAs reading the same q/k rows are
    /// adjacent, so those rows come from DRAM once), block 128 (4 warps). conv: `[T, 8192]` bf16 (row stride `conv_stride`); g, beta:
    /// `[T, 32]` f32; tinv: `[T, 32 * 64]` f32; state: `[slots, 32, 128, 128]` f32 (k-major); out: `[T, 4096]` bf16
    /// (row stride `out_stride`); segs: `[max_segs, 4]` (start, len, phase, slot).
    #[kernel]
    pub unsafe fn gdn_simt_scan(
        conv: *const u16,
        conv_stride: i32,
        g: *const f32,
        beta: *const f32,
        tinv: *const f32,
        state: *mut f32,
        out: *mut u16,
        out_stride: i32,
        segs: *const i32,
        max_segs: i32,
        num_slots: i32,
        phase: i32,
    ) {
        static mut KB: SharedArray<u32, 4352, 16> = SharedArray::UNINIT; // k rows [64][KW], raw bf16
        static mut PH: SharedArray<u32, 2304, 16> = SharedArray::UNINIT; // P hi, bf16x2 [64][36]
        static mut PL: SharedArray<u32, 2304, 16> = SharedArray::UNINIT; // P lo
        static mut OS: SharedArray<u32, 2304, 16> = SharedArray::UNINIT; // output staging, bf16x2 [64][36]
        // Per-row scalars: 0 cum | 64 beta | 128 sk | 192 sq | 256 e^cum sk | 320 e^cum sq | 384 e^rc sk | 448 gv
        static mut SC: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut GT: SharedArray<f32, 4> = SharedArray::UNINIT; // e^(sum g)
        let h = (thread::blockIdx_x() >> 1) as usize;
        let vb = (thread::blockIdx_x() & 1) as usize;
        let sidx = thread::blockIdx_y() as usize;
        if sidx >= max_segs as usize {
            return;
        }
        let seg_start = unsafe { *segs.add(sidx * 4) } as usize;
        let seg_len = unsafe { *segs.add(sidx * 4 + 1) } as usize;
        let seg_phase = unsafe { *segs.add(sidx * 4 + 2) };
        let slot = unsafe { *segs.add(sidx * 4 + 3) } as usize;
        if seg_phase != phase || seg_len == 0 || slot >= num_slots as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let w = tid >> 5;
        let (gq, tq) = (lane >> 2, lane & 3);
        let kb = addr_of_mut!(KB) as *mut u32;
        let ph = addr_of_mut!(PH) as *mut u32;
        let pl = addr_of_mut!(PL) as *mut u32;
        let os = addr_of_mut!(OS) as *mut u32;
        let sc = addr_of_mut!(SC) as *mut f32;
        let gt = addr_of_mut!(GT) as *mut f32;
        let stride = conv_stride as usize;
        let ostride = out_stride as usize;
        let hk = h / 2;
        // This warp's 16 value columns: vc + g and vc + g + 8 (absolute columns of head h: vb*64 + 16w + ..).
        let vc = vb * 64 + 16 * w;
        // State base for (slot, h): element (f, v) at f*128 + v.
        let sbase = unsafe { state.add((slot * 32 + h) * 128 * 128) };

        // S^T [v, f]: sacc[nf] holds (v = g, f = 8nf + 2t, +1) and (v = g + 8, same f).
        let mut sacc = [[0f32; 4]; 16];
        if phase != 0 {
            unroll!(nf in 0..16 {
                unroll!(r in 0..4 {
                    let v = vc + gq + 8 * (r >> 1);
                    let f = 8 * nf + 2 * tq + (r & 1);
                    sacc[nf][r] = unsafe { *sbase.add(f * 128 + v) };
                });
            });
        }

        let nch = seg_len.div_ceil(CH);
        let mut c = 0usize;
        while c < nch {
            let r0 = seg_start + c * CH;
            let len = if seg_len - c * CH < CH { seg_len - c * CH } else { CH };
            let qrow = unsafe { conv.add(r0 * stride + hk * 128) } as *const u32; // q row 0 of the chunk
            let wstride = stride / 2; // row stride in u32 words

            // (A) k tile and the gates of this head.
            unsafe {
                load_tile(kb, conv.add(r0 * stride + K_OFF + hk * 128), stride, len, tid);
                if tid < CH {
                    let valid = tid < len;
                    *sc.add(448 + tid) = if valid { *g.add((r0 + tid) * 32 + h) } else { 0.0 };
                    *sc.add(64 + tid) = if valid { *beta.add((r0 + tid) * 32 + h) } else { 0.0 };
                }
            }
            thread::sync_threads();

            // (B) row scales and the in-chunk cumulative sums.
            let sk = unsafe { row_inv_norm(kb, tid) };
            let sq = {
                let row = tid >> 1;
                let half = tid & 1;
                let mut ss = 0f32;
                if row < len {
                    // Unrolled so that the 32 loads are in flight together.
                    let mut v = [0u32; 32];
                    unroll!(k in 0..16 {
                        v[2 * k] = unsafe { *qrow.add(row * wstride + half * 32 + 2 * k) };
                        v[2 * k + 1] = unsafe { *qrow.add(row * wstride + half * 32 + 2 * k + 1) };
                    });
                    unroll!(k in 0..16 {
                        let (a, b, c, d) = (lo_bf16(v[2 * k]), hi_bf16(v[2 * k]), lo_bf16(v[2 * k + 1]), hi_bf16(v[2 * k + 1]));
                        ss += a * a + b * b + c * c + d * d;
                    });
                }
                ss += warp::shuffle_xor_f32(ss, 1);
                Q_SCALE / (ss + L2_EPS).sqrt()
            };
            if tid & 1 == 0 {
                unsafe {
                    *sc.add(128 + (tid >> 1)) = sk;
                    *sc.add(192 + (tid >> 1)) = sq;
                }
            }
            if w == 0 {
                // Lane l owns rows 2l, 2l+1: inclusive forward scan and exclusive reverse scan over the warp.
                let g0 = unsafe { *sc.add(448 + 2 * lane) };
                let g1 = unsafe { *sc.add(448 + 2 * lane + 1) };
                let pair = g0 + g1;
                let mut incl = pair;
                let mut d = 1u32;
                while d < 32 {
                    let o = warp::shuffle_up_f32(incl, d);
                    if lane as u32 >= d {
                        incl += o;
                    }
                    d *= 2;
                }
                let excl = incl - pair;
                let mut rincl = pair; // sum over lanes >= l
                let mut d = 1u32;
                while d < 32 {
                    let o = warp::shuffle_down_f32(rincl, d);
                    if (lane as u32) + d < 32 {
                        rincl += o;
                    }
                    d *= 2;
                }
                let rexcl = rincl - pair; // sum over lanes > l
                unsafe {
                    *sc.add(2 * lane) = excl + g0;
                    *sc.add(2 * lane + 1) = excl + pair;
                    // rc_i = sum_{j > i} g_j, kept here until (C) turns it into e^rc sk.
                    *sc.add(384 + 2 * lane) = rexcl + g1;
                    *sc.add(384 + 2 * lane + 1) = rexcl;
                }
                let total = warp::shuffle_f32(incl, 31);
                if lane == 0 {
                    unsafe { *gt = total.exp() };
                }
            }
            thread::sync_threads();

            // (C) per-row decay products.
            if tid < CH {
                unsafe {
                    let e = (*sc.add(tid)).exp();
                    *sc.add(256 + tid) = e * *sc.add(128 + tid);
                    *sc.add(320 + tid) = e * *sc.add(192 + tid);
                    *sc.add(384 + tid) = (*sc.add(384 + tid)).exp() * *sc.add(128 + tid);
                }
            }

            // (D1) P rows 16w..16w+16: (q k^T) scaled, decayed, masked to j <= i < len; stored split in PH/PL.
            {
                let i0 = 16 * w;
                let mut acc = [[0f32; 4]; 8];
                let ra = i0 + gq;
                let rb = i0 + gq + 8;
                unroll!(ks in 0..8 {
                    let a = unsafe {
                        [
                            if ra < len { *qrow.add(ra * wstride + ks * 8 + tq) } else { 0 },
                            if rb < len { *qrow.add(rb * wstride + ks * 8 + tq) } else { 0 },
                            if ra < len { *qrow.add(ra * wstride + ks * 8 + 4 + tq) } else { 0 },
                            if rb < len { *qrow.add(rb * wstride + ks * 8 + 4 + tq) } else { 0 },
                        ]
                    };
                    unroll!(n in 0..8 {
                        if n <= 2 * w + 1 {
                            let b = unsafe { [*kb.add((8 * n + gq) * KW + ks * 8 + tq), *kb.add((8 * n + gq) * KW + ks * 8 + 4 + tq)] };
                            acc[n] = mma(acc[n], a, b);
                        }
                    });
                });
                unroll!(n in 0..8 {
                    if n <= 2 * w + 1 {
                        unroll!(half in 0..4 {
                            if half < 2 {
                                let i = i0 + gq + 8 * half;
                                let j = 8 * n + 2 * tq;
                                let p = |jj: usize, v: f32| -> f32 {
                                    if jj <= i && i < len {
                                        unsafe { *sc.add(192 + i) * *sc.add(128 + jj) * v * (*sc.add(i) - *sc.add(jj)).exp() }
                                    } else {
                                        0.0
                                    }
                                };
                                let (hi, lo) = split2(p(j, acc[n][2 * half]), p(j + 1, acc[n][2 * half + 1]));
                                unsafe {
                                    *ph.add(i * 36 + 4 * n + tq) = hi;
                                    *pl.add(i * 36 + 4 * n + tq) = lo;
                                }
                            }
                        });
                    }
                });
            }

            // (D2) X^T = S^T k^T over the 64 rows (8 n-tiles of tokens).
            let mut xa = [[0f32; 4]; 8];
            unroll!(ks in 0..8 {
                let (h0, l0) = split2(sacc[2 * ks][0], sacc[2 * ks][1]);
                let (h1, l1) = split2(sacc[2 * ks][2], sacc[2 * ks][3]);
                let (h2, l2) = split2(sacc[2 * ks + 1][0], sacc[2 * ks + 1][1]);
                let (h3, l3) = split2(sacc[2 * ks + 1][2], sacc[2 * ks + 1][3]);
                let ah = [h0, h1, h2, h3];
                let al = [l0, l1, l2, l3];
                unroll!(n in 0..8 {
                    let row = 8 * n + gq;
                    let bk = unsafe { [*kb.add(row * KW + ks * 8 + tq), *kb.add(row * KW + ks * 8 + 4 + tq)] };
                    xa[n] = mma(mma(xa[n], ah, bk), al, bk);
                });
            });
            thread::sync_threads();

            // (E) R^T = beta ⊙ (v - e^cum sk ⊙ X^T), in place in xa.
            let vrow = unsafe { conv.add(r0 * stride + V_OFF + h * 128 + vc) };
            unroll!(n in 0..8 {
                unroll!(r in 0..4 {
                    let i = 8 * n + 2 * tq + (r & 1);
                    let v = gq + 8 * (r >> 1);
                    let vv = if i < len { lo_bf16(unsafe { *vrow.add(i * stride + v) } as u32) } else { 0.0 };
                    xa[n][r] = unsafe { *sc.add(64 + i) * (vv - *sc.add(256 + i) * xa[n][r]) };
                });
            });

            // (F) v_new^T = R^T Tinv^T (Tinv lower triangular: k-step ks meets n-tile n only if 16ks <= 8n + 7).
            let trow = unsafe { tinv.add(r0 * 2048 + h * 64) };
            let mut vn = [[0f32; 4]; 8];
            unroll!(ks in 0..4 {
                let (h0, l0) = split2(xa[2 * ks][0], xa[2 * ks][1]);
                let (h1, l1) = split2(xa[2 * ks][2], xa[2 * ks][3]);
                let (h2, l2) = split2(xa[2 * ks + 1][0], xa[2 * ks + 1][1]);
                let (h3, l3) = split2(xa[2 * ks + 1][2], xa[2 * ks + 1][3]);
                let ah = [h0, h1, h2, h3];
                let al = [l0, l1, l2, l3];
                unroll!(n in 0..8 {
                    if 16 * ks <= 8 * n + 7 {
                        let i = 8 * n + gq;
                        let (t0, t1, t8, t9) = if i < len {
                            unsafe {
                                let p = trow.add(i * 2048 + 16 * ks + 2 * tq);
                                (*p, *p.add(1), *p.add(8), *p.add(9))
                            }
                        } else {
                            (0.0, 0.0, 0.0, 0.0)
                        };
                        let (b0h, b0l) = split2(t0, t1);
                        let (b1h, b1l) = split2(t8, t9);
                        let bh = [b0h, b1h];
                        let bl = [b0l, b1l];
                        vn[n] = mma(mma(mma(vn[n], ah, bh), ah, bl), al, bh);
                    }
                });
            });

            // (G) Y^T = S^T q^T (old S), then O^T = e^cum sq ⊙ Y^T + v_new^T P^T, staged as bf16 in OS[i][v].
            let mut ya = [[0f32; 4]; 8];
            unroll!(ks in 0..8 {
                let (h0, l0) = split2(sacc[2 * ks][0], sacc[2 * ks][1]);
                let (h1, l1) = split2(sacc[2 * ks][2], sacc[2 * ks][3]);
                let (h2, l2) = split2(sacc[2 * ks + 1][0], sacc[2 * ks + 1][1]);
                let (h3, l3) = split2(sacc[2 * ks + 1][2], sacc[2 * ks + 1][3]);
                let ah = [h0, h1, h2, h3];
                let al = [l0, l1, l2, l3];
                unroll!(n in 0..8 {
                    let row = 8 * n + gq;
                    let bq = if row < len {
                        unsafe { [*qrow.add(row * wstride + ks * 8 + tq), *qrow.add(row * wstride + ks * 8 + 4 + tq)] }
                    } else {
                        [0, 0]
                    };
                    ya[n] = mma(mma(ya[n], ah, bq), al, bq);
                });
            });
            unroll!(n in 0..8 {
                unroll!(r in 0..4 {
                    let i = 8 * n + 2 * tq + (r & 1);
                    ya[n][r] *= unsafe { *sc.add(320 + i) };
                });
            });
            unroll!(ks in 0..4 {
                let (h0, l0) = split2(vn[2 * ks][0], vn[2 * ks][1]);
                let (h1, l1) = split2(vn[2 * ks][2], vn[2 * ks][3]);
                let (h2, l2) = split2(vn[2 * ks + 1][0], vn[2 * ks + 1][1]);
                let (h3, l3) = split2(vn[2 * ks + 1][2], vn[2 * ks + 1][3]);
                let ah = [h0, h1, h2, h3];
                let al = [l0, l1, l2, l3];
                unroll!(n in 0..8 {
                    if 16 * ks <= 8 * n + 7 {
                        let i = 8 * n + gq;
                        let bh = unsafe { [*ph.add(i * 36 + 8 * ks + tq), *ph.add(i * 36 + 8 * ks + 4 + tq)] };
                        let bl = unsafe { [*pl.add(i * 36 + 8 * ks + tq), *pl.add(i * 36 + 8 * ks + 4 + tq)] };
                        ya[n] = mma(mma(mma(ya[n], ah, bh), ah, bl), al, bh);
                    }
                });
            });
            let osh = os as *mut u16;
            unroll!(n in 0..8 {
                unroll!(r in 0..4 {
                    let i = 8 * n + 2 * tq + (r & 1);
                    let v = 16 * w + gq + 8 * (r >> 1);
                    unsafe { *osh.add(i * 72 + v) = bf16_rne(ya[n][r]) as u16 };
                });
            });

            // (H) S^T = e^(sum g) S^T + Z^T k, Z^T = v_new^T ⊙ e^rc sk.
            let decay = unsafe { *gt };
            unroll!(nf in 0..16 {
                unroll!(r in 0..4 {
                    sacc[nf][r] *= decay;
                });
            });
            let kbh = kb as *const u16;
            unroll!(ks in 0..4 {
                let z = |n: usize, r: usize| -> f32 {
                    let i = 8 * n + 2 * tq + (r & 1);
                    vn[n][r] * unsafe { *sc.add(384 + i) }
                };
                let (h0, l0) = split2(z(2 * ks, 0), z(2 * ks, 1));
                let (h1, l1) = split2(z(2 * ks, 2), z(2 * ks, 3));
                let (h2, l2) = split2(z(2 * ks + 1, 0), z(2 * ks + 1, 1));
                let (h3, l3) = split2(z(2 * ks + 1, 2), z(2 * ks + 1, 3));
                let ah = [h0, h1, h2, h3];
                let al = [l0, l1, l2, l3];
                let i0 = 16 * ks + 2 * tq;
                unroll!(nf in 0..16 {
                    let f = 8 * nf + gq;
                    let b = unsafe {
                        [
                            (*kbh.add(i0 * 2 * KW + f) as u32) | ((*kbh.add((i0 + 1) * 2 * KW + f) as u32) << 16),
                            (*kbh.add((i0 + 8) * 2 * KW + f) as u32) | ((*kbh.add((i0 + 9) * 2 * KW + f) as u32) << 16),
                        ]
                    };
                    sacc[nf] = mma(mma(sacc[nf], ah, b), al, b);
                });
            });
            thread::sync_threads();

            // (I) coalesced output of the chunk's rows.
            let obase = unsafe { out.add(r0 * ostride + h * 128 + vb * 64) } as *mut u32;
            let mut e = tid;
            while e < len * 32 {
                let i = e >> 5;
                let wd = e & 31;
                unsafe { *obase.add(i * (ostride / 2) + wd) = *os.add(i * 36 + wd) };
                e += 128;
            }
            c += 1;
        }

        if phase == 0 {
            unroll!(nf in 0..16 {
                unroll!(r in 0..4 {
                    let v = vc + gq + 8 * (r >> 1);
                    let f = 8 * nf + 2 * tq + (r & 1);
                    unsafe { *sbase.add(f * 128 + v) = sacc[nf][r] };
                });
            });
        }
    }

}

fn main() {}
