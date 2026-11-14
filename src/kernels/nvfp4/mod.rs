//! NVFP4 W4A4 MLP kernels (DESIGN.md §3.1, §6.6): activation quantization and block-scaled FP4 GEMMs. Owner: W3c.
//!
//! The numerics follow compressed-tensors 0.20.1 as the checkpoint `elephant-1-nvfp4-mlp` was evaluated: transformers
//! 5.19 `Qwen3_5ForCausalLM`, bf16, `CompressedTensorsConfig(run_compressed=False)`. In that mode every MLP
//! projection ends up `DECOMPRESSED`: its weight is materialized once to bf16, weight QDQ is skipped, and the input
//! activations are fake-quantized on every forward. `reference/nvfp4/probe_dtypes.py` loads the real modules and checks
//! the formulas below against them bit for bit (weights and outputs, `torch.equal`).
//!
//! # Notation
//!
//! `rne_T(v)` rounds to the nearest `T` value, ties to even. Every arithmetic step is one IEEE operation in the stated
//! type, in the stated order: `a / b` is an f32 division, not a multiply by a reciprocal. `e2m1` values are
//! `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a sign bit.
//!
//! # Weights (checkpoint layout, per projection `[N, K]`)
//!
//! - `weight_packed` u8 `[N, K/2]`: byte `j` of row `n` holds element `2j` in its low nibble and `2j+1` in its high
//!   nibble (`unpack_fp4_from_uint8`). A nibble is `sign << 3 | magnitude index` into the e2m1 table.
//! - `weight_scale` e4m3 `[N, K/16]`, `weight_global_scale` (`wgs`) f32 `[1]`.
//! - Dequantization (`NVFP4PackedCompressor.decompress` → `_dequantize`; the scale is cast e4m3 → bf16, exactly):
//!
//!   `W[n, k] = rne_bf16( e2m1(n, k) · (f32(scale[n, k/16]) / wgs) )`
//!
//!   The division and the product are f32 (bf16 scale ÷ f32 `[1]` tensor promotes to f32); the product is exact,
//!   so the only rounding after the division is the final bf16 cast.
//!
//! # Activations (`dynamic: local`, `tensor_group`, group 16, static `input_global_scale` = `igs`, f32)
//!
//! For each token row and each group `g` of 16 consecutive bf16 values `x` (`compute_dynamic_scales_and_zp` →
//! `calculate_qparams`, then `_quantize_dequantize`):
//!
//! 1. `amax = max |x_i|` (exact in bf16).
//! 2. `s_bf = rne_bf16(f32(amax) / 6)` — torch bf16 ÷ bf16 computes in f32 and rounds to bf16.
//! 3. `s8 = rne_e4m3(min(igs · f32(s_bf), 448))` — f32 product, clamped to ±448, cast to `float8_e4m3fn` (torch's
//!    cast is RNE; subnormals down to 2^-9 are kept, `2^-10` rounds to 0).
//! 4. If `s8 == 0` the scale becomes `0.125` (`_get_dtype_eps(float8_e4m3fn)`); this is also the scale byte we
//!    store (0x20). The 16 values of such a group all quantize to 0.
//! 5. `eff = f32(s8) / igs` (f32 division).
//! 6. `y_i = clamp(f32(x_i) / eff, -6, 6)` (f32 division).
//! 7. `q_i = rne_e2m1(y_i)`. compressed-tensors' threshold cascade (`cast_to_fp4`: `<=0.25 → 0`, `<0.75 → 0.5`,
//!    `<=1.25 → 1`, `<1.75 → 1.5`, `<=2.5 → 2`, `<3.5 → 3`, `<=5 → 4`, else 6) is exactly round-to-nearest-even on
//!    the e2m1 grid. The sign of a zero result follows `torch.sign(y) · 0`: `-0` for `y < 0`, `+0` for `y = ±0`.
//! 8. The fake-quantized activation fed to the bf16 GEMM is `rne_bf16(q_i · eff)`.
//!
//! The kernel stores `q_i` packed like `weight_packed` (low nibble first) and `s8` as e4m3 `[T, K/16]`. These are
//! bit-exact with compressed-tensors' own `quantize` + `pack_fp4_to_uint8` on the same inputs
//! (`reference/nvfp4/gen_quant_vectors.py`, `tests/kernels_nvfp4.rs`).
//!
//! # GEMM epilogue
//!
//! The reference computes `y = rne_bf16(Σ_k Xq[t, k] · W[n, k])` with fp32 accumulation, where both operands were
//! rounded to bf16 after dequantization. The block-scaled MMA computes
//! `acc = Σ_k q_x · s8_x · q_w · s8_w` (every product exact in f32) and the epilogue applies
//!
//!   `y = rne_bf16(acc · alpha[n])`, `alpha[n] = rne_f32(1 / (igs · wgs(n)))` (computed in f64),
//!
//! where `wgs(n)` is the global scale of the projection that owns output column `n`; the fused gate/up GEMM has a
//! different `alpha` per half. The only numerical difference from the reference is that the reference rounds each
//! dequantized operand to bf16 (relative 2^-9 per element) before the dot product, while this path keeps them exact;
//! the measured effect is reported by `tests/kernels_nvfp4.rs`.
//!
//! # Layouts used by the kernels
//!
//! - Activations: packed `[T, K/2]` (`f4e2m1fnx2`), scales `[T, K/16]` (`f8e4m3fn`), row-major, as quantized.
//! - Weights: the checkpoint layout unchanged (`[N, K/2]`, `[N, K/16]` row-major), with gate and up concatenated
//!   along N for the fused projection, plus `alpha` (per column, and as `alpha_lo`/`alpha_hi` split at the gate/up
//!   boundary). [`prepare_weights`] builds these once at load; [`Nvfp4Linear::upload`] puts them on the device.
//!
//! # Kernels
//!
//! - [`kernels::quantize_nvfp4`]: standalone quantization, one `[BR, BC]` tile per tile block.
//!   [`kernels::nvfp4_quantize_tile`] is the same math as an inlinable device function for norm / SiLU·mul
//!   epilogues (input must already be bf16-rounded, as in the reference).
//! - [`kernels::gemm_nvfp4_raw`] (production, [`GemmKernel::Raw`]): `mmaf_scaled` over `[BM, BK] × [BK, BN]`
//!   NVFP4 tiles, fp32 accumulate, grouped-M rasterization computed in the kernel from a 1-D grid (optionally
//!   persistent), scalar alpha per half. No in-kernel bounds checks: [`gemm_op`] validates every shape first.
//! - [`kernels::gemm_nvfp4`] ([`GemmKernel::Mapped`]): the same math through cutile's safe mapped-partition
//!   schedule with a per-column alpha vector. Checked, but its `[BM, BN]` alpha broadcast and residual checks cost
//!   registers (128×256 tiles collapse to ~6 TFLOP/s); kept as the checked variant for tests.

use anyhow::{Result, bail, ensure};
use cutile::cuda_core::{Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;

/// Group size along K shared by the activation and weight block scales.
pub const GROUP: usize = 16;
/// Largest finite `float8_e4m3fn` value; compressed-tensors clamps the scale to it before the cast.
pub const E4M3_MAX: f32 = 448.0;
/// Scale used for a group whose e4m3 scale rounds to zero (`_get_dtype_eps(torch.float8_e4m3fn)`).
pub const ZERO_SCALE_EPS: f32 = 0.125;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Quantize a `[R, C]` tile of activations to NVFP4 (module docs, steps 1–7). `C = 16 · G`.
    ///
    /// `x` must hold bf16-representable values (the reference quantizes a bf16 tensor). Fused epilogues
    /// (RMSNorm, SiLU·mul) must round their f32 result to bf16 before calling this, as the reference does. Returns
    /// the e2m1 values (pack them with `.pack(...)`) and the e4m3 group scales `[R, G]`.
    pub fn nvfp4_quantize_tile<const R: i32, const C: i32, const G: i32>(
        x: Tile<f32, { [R, C] }>,
        igs: f32,
    ) -> (Tile<f4e2m1fn, { [R, C] }>, Tile<f8e4m3fn, { [R, G] }>) {
        let xg: Tile<f32, { [R, G, 16] }> = x.reshape(const_shape![R, G, 16]);
        // 1. amax per group.
        let amax: Tile<f32, { [R, G] }> = reduce_max(absf(xg), 2i32);
        // 2. bf16(amax / 6).
        let six: Tile<f32, { [R, G] }> = constant(6.0f32, const_shape![R, G]);
        let s_div: Tile<f32, { [R, G] }> = divf(amax, six, rounding::NearestEven, ftz::Disabled);
        let s_bf: Tile<bf16, { [R, G] }> = ftof(s_div, rounding::NearestEven);
        let s_bf32: Tile<f32, { [R, G] }> = convert_tile(s_bf);
        // 3. e4m3(min(igs * s, 448)).
        let igs_g: Tile<f32, { [R, G] }> = broadcast_scalar(igs, const_shape![R, G]);
        let t: Tile<f32, { [R, G] }> = mulf(igs_g, s_bf32, rounding::NearestEven, ftz::Disabled);
        let e4m3_max: Tile<f32, { [R, G] }> = constant(448.0f32, const_shape![R, G]);
        let t: Tile<f32, { [R, G] }> = minf(t, e4m3_max, nan::Disabled, ftz::Disabled);
        let s8: Tile<f8e4m3fn, { [R, G] }> = ftof(t, rounding::NearestEven);
        let s8f: Tile<f32, { [R, G] }> = convert_tile(s8);
        // 4. zero scale -> 0.125.
        let zero: Tile<f32, { [R, G] }> = constant(0.0f32, const_shape![R, G]);
        let is_zero = cmpf(s8f, zero, predicate::Equal, cmp_ordering::Ordered);
        let eps: Tile<f32, { [R, G] }> = constant(0.125f32, const_shape![R, G]);
        let scale: Tile<f32, { [R, G] }> = select(is_zero, eps, s8f);
        let scale8: Tile<f8e4m3fn, { [R, G] }> = ftof(scale, rounding::NearestEven);
        // 5. eff = scale / igs.
        let eff: Tile<f32, { [R, G] }> = divf(scale, igs_g, rounding::NearestEven, ftz::Disabled);
        let eff_b: Tile<f32, { [R, G, 16] }> =
            eff.reshape(const_shape![R, G, 1]).broadcast(const_shape![R, G, 16]);
        // 6. y = clamp(x / eff, -6, 6); `+ 0.0` turns a -0 quotient into +0 (torch.sign(-0.0) == 0).
        let y: Tile<f32, { [R, G, 16] }> = divf(xg, eff_b, rounding::NearestEven, ftz::Disabled);
        let lim: Tile<f32, { [R, G, 16] }> = constant(6.0f32, const_shape![R, G, 16]);
        let nlim: Tile<f32, { [R, G, 16] }> = constant(-6.0f32, const_shape![R, G, 16]);
        let y: Tile<f32, { [R, G, 16] }> = minf(y, lim, nan::Disabled, ftz::Disabled);
        let y: Tile<f32, { [R, G, 16] }> = maxf(y, nlim, nan::Disabled, ftz::Disabled);
        let pzero: Tile<f32, { [R, G, 16] }> = constant(0.0f32, const_shape![R, G, 16]);
        let y: Tile<f32, { [R, G, 16] }> = addf(y, pzero, rounding::NearestEven, ftz::Disabled);
        // 7. e2m1, round to nearest even.
        let q: Tile<f4e2m1fn, { [R, G, 16] }> = ftof(y, rounding::NearestEven);
        (q.reshape(const_shape![R, C]), scale8)
    }

    /// `x` bf16 `[T, K]` → packed e2m1 `[T, K/2]` + e4m3 scales `[T, K/16]`. Grid: `(ceil(T/BR), K/BC)`.
    /// Generics: `BR, BC, BC/2, BC/16`.
    #[cutile::entry()]
    pub fn quantize_nvfp4<const BR: i32, const BC: i32, const BCP: i32, const G: i32>(
        q: &mut Tensor<f4e2m1fnx2, { [BR, BCP] }>,
        s: &mut Tensor<f8e4m3fn, { [BR, G] }>,
        x: &Tensor<bf16, { [-1, -1] }>,
        igs: f32,
    ) {
        let pid = get_tile_block_id();
        let part_x = x.partition(const_shape![BR, BC]);
        let xt: Tile<bf16, { [BR, BC] }> = part_x.load([pid.0, pid.1]);
        let xf: Tile<f32, { [BR, BC] }> = convert_tile(xt);
        let (qt, st) = nvfp4_quantize_tile::<BR, BC, G>(xf, igs);
        q.store(qt.pack(const_shape![BR, BCP]));
        s.store(st);
    }

    /// `(x * rsqrt(sum(x²)/N + eps)) * (1 + w)` in fp32: the same operations in the same order as
    /// `crate::kernels::norm`'s `zc_scale` (a copy: cuTile modules cannot call each other's device functions).
    fn zc_scale_q<const N: i32>(
        xf: Tile<f32, { [1, N] }>,
        one_plus_w: Tile<f32, { [1, N] }>,
        eps: f32,
    ) -> Tile<f32, { [1, N] }> {
        let ss: Tile<f32, { [1] }> = reduce_sum(xf * xf, 1i32);
        let ss: f32 = tile_to_scalar(ss.reshape(const_shape![]));
        let n: f32 = convert_scalar(N);
        let inv: Tile<f32, { [] }> = rsqrt(scalar_to_tile(ss / n + eps), ftz::Disabled);
        let inv: f32 = tile_to_scalar(inv);
        (xf * inv.broadcast(const_shape![1, N])) * one_plus_w
    }

    /// `zc_rmsnorm` (as `crate::kernels::norm::zc_rmsnorm`) with the NVFP4 quantization of its bf16 output fused
    /// into the epilogue: the normed row is rounded to bf16 (the reference's rounding point) and quantized, and
    /// only the packed values and scales are stored. One row per tile block. Generics: `N, N/2, N/16`.
    #[cutile::entry()]
    pub fn zc_rmsnorm_nvfp4<const N: i32, const NP: i32, const G: i32>(
        q: &mut Tensor<f4e2m1fnx2, { [1, NP] }>,
        s: &mut Tensor<f8e4m3fn, { [1, G] }>,
        x: &Tensor<bf16, { [-1, N] }>,
        w: &Tensor<bf16, { [N] }>,
        eps: f32,
        igs: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_shape: Shape<{ [1, N] }> = const_shape![1, N];
        let tx: Tile<bf16, { [1, N] }> = x.partition(row_shape).load([pid.0, 0i32]);
        let xf: Tile<f32, { [1, N] }> = convert_tile(tx);
        let tw: Tile<bf16, { [N] }> = w.partition(const_shape![N]).load([0i32]);
        let wf: Tile<f32, { [1, N] }> = convert_tile(tw.reshape(row_shape));
        let one_plus_w: Tile<f32, { [1, N] }> = constant(1.0f32, row_shape) + wf;
        let y: Tile<f32, { [1, N] }> = zc_scale_q(xf, one_plus_w, eps);
        let yb: Tile<bf16, { [1, N] }> = convert_tile(y);
        let yq: Tile<f32, { [1, N] }> = convert_tile(yb);
        let (qt, st) = nvfp4_quantize_tile::<1, N, G>(yq, igs);
        q.store(qt.pack(const_shape![1, NP]));
        s.store(st);
    }

    /// `add_zc_rmsnorm` (as `crate::kernels::norm::add_zc_rmsnorm`: `residual = bf16(residual + x)` in place,
    /// then the norm) with the NVFP4 quantization of the bf16 normed row fused into the epilogue. Generics:
    /// `N, N/2, N/16`.
    #[cutile::entry()]
    pub fn add_zc_rmsnorm_nvfp4<const N: i32, const NP: i32, const G: i32>(
        residual: &mut Tensor<bf16, { [1, N] }>,
        q: &mut Tensor<f4e2m1fnx2, { [1, NP] }>,
        s: &mut Tensor<f8e4m3fn, { [1, G] }>,
        x: &Tensor<bf16, { [-1, N] }>,
        w: &Tensor<bf16, { [N] }>,
        eps: f32,
        igs: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_shape: Shape<{ [1, N] }> = const_shape![1, N];
        let tr: Tile<bf16, { [1, N] }> = residual.load();
        let tx: Tile<bf16, { [1, N] }> = x.partition(row_shape).load([pid.0, 0i32]);
        let trf: Tile<f32, { [1, N] }> = convert_tile(tr);
        let txf: Tile<f32, { [1, N] }> = convert_tile(tx);
        let sum: Tile<f32, { [1, N] }> = trf + txf;
        let new_res: Tile<bf16, { [1, N] }> = convert_tile(sum);
        residual.store(new_res);
        let xf: Tile<f32, { [1, N] }> = convert_tile(new_res);
        let tw: Tile<bf16, { [N] }> = w.partition(const_shape![N]).load([0i32]);
        let wf: Tile<f32, { [1, N] }> = convert_tile(tw.reshape(row_shape));
        let one_plus_w: Tile<f32, { [1, N] }> = constant(1.0f32, row_shape) + wf;
        let y: Tile<f32, { [1, N] }> = zc_scale_q(xf, one_plus_w, eps);
        let yb: Tile<bf16, { [1, N] }> = convert_tile(y);
        let yq: Tile<f32, { [1, N] }> = convert_tile(yb);
        let (qt, st) = nvfp4_quantize_tile::<1, N, G>(yq, igs);
        q.store(qt.pack(const_shape![1, NP]));
        s.store(st);
    }

    /// `silu_mul` (as `crate::kernels::pointwise::silu_mul`: `bf16(bf16(silu(g)) · u)`) with the NVFP4
    /// quantization of the bf16 product fused into the epilogue. `B` columns of one row per tile block; grid
    /// `(T, I/B)`. Generics: `I, B, B/2, B/16`.
    #[cutile::entry()]
    pub fn silu_mul_nvfp4<const I: i32, const B: i32, const BP: i32, const G: i32>(
        q: &mut Tensor<f4e2m1fnx2, { [1, BP] }>,
        s: &mut Tensor<f8e4m3fn, { [1, G] }>,
        gu: &Tensor<bf16, { [-1, -1] }>,
        igs: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<bf16, { [1, B] }> = gu.partition(const_shape![1, B]);
        let tg: Tile<bf16, { [1, B] }> = part.load([pid.0, pid.1]);
        let tu: Tile<bf16, { [1, B] }> = part.load([pid.0, pid.1 + I / B]);
        let g: Tile<f32, { [1, B] }> = convert_tile(tg);
        let u: Tile<f32, { [1, B] }> = convert_tile(tu);
        let one: Tile<f32, { [1, B] }> = constant(1.0f32, const_shape![1, B]);
        let sl: Tile<f32, { [1, B] }> = true_div(g, one + exp(negf(g)));
        let s_b: Tile<bf16, { [1, B] }> = convert_tile(sl);
        let s_f: Tile<f32, { [1, B] }> = convert_tile(s_b);
        let o: Tile<bf16, { [1, B] }> = convert_tile(s_f * u);
        let of: Tile<f32, { [1, B] }> = convert_tile(o);
        let (qt, st) = nvfp4_quantize_tile::<1, B, G>(of, igs);
        q.store(qt.pack(const_shape![1, BP]));
        s.store(st);
    }

    /// [`silu_mul_nvfp4`] over an `[R, B]` tile (`R` rows, `B` columns of the `I` outputs) per tile block; grid
    /// `(T/R, I/B)`. The same element-wise operations and the same 16-column quantization groups, so the result
    /// is identical for every `(R, B)`; the shape only changes registers and occupancy (F1). Generics:
    /// `I, R, B, B/2, B/16`.
    #[cutile::entry()]
    pub fn silu_mul_nvfp4_rb<const I: i32, const R: i32, const B: i32, const BP: i32, const G: i32>(
        q: &mut Tensor<f4e2m1fnx2, { [R, BP] }>,
        s: &mut Tensor<f8e4m3fn, { [R, G] }>,
        gu: &Tensor<bf16, { [-1, -1] }>,
        igs: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<bf16, { [R, B] }> = gu.partition(const_shape![R, B]);
        let tg: Tile<bf16, { [R, B] }> = part.load([pid.0, pid.1]);
        let tu: Tile<bf16, { [R, B] }> = part.load([pid.0, pid.1 + I / B]);
        let g: Tile<f32, { [R, B] }> = convert_tile(tg);
        let u: Tile<f32, { [R, B] }> = convert_tile(tu);
        let one: Tile<f32, { [R, B] }> = constant(1.0f32, const_shape![R, B]);
        let sl: Tile<f32, { [R, B] }> = true_div(g, one + exp(negf(g)));
        let s_b: Tile<bf16, { [R, B] }> = convert_tile(sl);
        let s_f: Tile<f32, { [R, B] }> = convert_tile(s_b);
        let o: Tile<bf16, { [R, B] }> = convert_tile(s_f * u);
        let of: Tile<f32, { [R, B] }> = convert_tile(o);
        let (qt, st) = nvfp4_quantize_tile::<R, B, G>(of, igs);
        q.store(qt.pack(const_shape![R, BP]));
        s.store(st);
    }

    /// The gate/up GEMM with SiLU·mul and the NVFP4 quantization of the product in its epilogue (F1), on a
    /// gate/up weight whose rows are interleaved in blocks of `H = BN/2` ([`interleave_gate_up`]: row block `j` of
    /// `BN` rows = gate rows `j·H..` then up rows `j·H..`). The main loop is exactly [`gemm_nvfp4_raw`]'s (one
    /// `[BN, BK]` weight tile, one accumulator), so every gate and up element is the same `mmaf_scaled` chain over K
    /// as in the unfused GEMM; the epilogue extracts the two `[BM, H]` halves and applies exactly
    /// [`gemm_nvfp4_raw`]'s epilogue (`bf16(acc · alpha)` per half) followed by [`silu_mul_nvfp4`]'s
    /// (`bf16(bf16(silu(g)) · u)`, quantized under `igs`, the down projection's input scale). `q`/`s` are
    /// byte-identical to `gemm_nvfp4_raw` + `silu_mul_nvfp4` (tests/kernels_nvfp4_f1.rs), and the bf16 `[T, 2I]`
    /// gate/up tensor is never written. Generics: `BM, BN, BK, BK/2, BK/16, GM, H, H/2, H/16`; `num_bid_n = I / H`.
    /// Safety: no bounds checks (the host validates every shape).
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gemm_nvfp4_swiglu_il<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BKP: i32,
        const BKS: i32,
        const GM: i32,
        const H: i32,
        const HP: i32,
        const HS: i32,
    >(
        q: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        s: &Tensor<f8e4m3fn, { [-1, -1] }>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha_g: f32,
        alpha_u: f32,
        igs: f32,
        num_bid_m: i32,
        num_bid_n: i32,
        num_blocks: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BKP]);
        let part_w = w.partition(const_shape![BN, BKP]);
        let part_xs = x_scales.partition(const_shape![BM, BKS]);
        let part_ws = w_scales.partition(const_shape![BN, BKS]);
        let mut part_q = unsafe { q.partition_full_mut(const_shape![BM, HP]) };
        let mut part_s = unsafe { s.partition_full_mut(const_shape![BM, HS]) };
        let total = num_bid_m * num_bid_n;
        let group = GM * num_bid_n;
        let mut tile_id = get_tile_block_id().0;
        while tile_id < total {
            let first_m = (tile_id / group) * GM;
            let rem_m = num_bid_m - first_m;
            let gsz = if rem_m < GM { rem_m } else { GM };
            let in_group = tile_id % group;
            let bid_m = first_m + in_group % gsz;
            let bid_n = in_group / gsz;
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for k in 0i32..num_tiles(&part_x, 1) {
                let tx = part_x.load([bid_m, k]).unpack(const_shape![BM, BK]);
                let tw = part_w.load([bid_n, k]).unpack(const_shape![BN, BK]).transpose();
                let txs = part_xs.load([bid_m, k]);
                let tws = part_ws.load([bid_n, k]).transpose();
                acc = mmaf_scaled(tx, tw, acc, txs, tws);
            }
            let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
            let one_i: Tile<i32, { [] }> = scalar_to_tile(1i32);
            let ag: Tile<f32, { [BM, H] }> = extract(acc, [zero, zero]);
            let au: Tile<f32, { [BM, H] }> = extract(acc, [zero, one_i]);
            let gb: Tile<bf16, { [BM, H] }> = convert_tile(ag * broadcast_scalar(alpha_g, const_shape![BM, H]));
            let ub: Tile<bf16, { [BM, H] }> = convert_tile(au * broadcast_scalar(alpha_u, const_shape![BM, H]));
            let g: Tile<f32, { [BM, H] }> = convert_tile(gb);
            let u: Tile<f32, { [BM, H] }> = convert_tile(ub);
            let one: Tile<f32, { [BM, H] }> = constant(1.0f32, const_shape![BM, H]);
            let sl: Tile<f32, { [BM, H] }> = true_div(g, one + exp(negf(g)));
            let s_b: Tile<bf16, { [BM, H] }> = convert_tile(sl);
            let s_f: Tile<f32, { [BM, H] }> = convert_tile(s_b);
            let o: Tile<bf16, { [BM, H] }> = convert_tile(s_f * u);
            let of: Tile<f32, { [BM, H] }> = convert_tile(o);
            let (qt, sct) = nvfp4_quantize_tile::<BM, H, HS>(of, igs);
            part_q.store(qt.pack(const_shape![BM, HP]), [bid_m, bid_n]);
            part_s.store(sct, [bid_m, bid_n]);
            tile_id = tile_id + num_blocks;
        }
    }

    /// `z[T, N] = bf16(alpha[n] · Σ_k x[t, k] · w[n, k])` with NVFP4 operands and e4m3 group-16 block scales.
    ///
    /// Persistent, grouped-M rasterization over the output tiles (`MAP_SHAPE = [GROUP_M, 1]`). Generics:
    /// `BM, BN, BK, BK/2, BK/16, GROUP_M, 1`. Host contract: `z` partitioned `[BM, BN]` and mapped
    /// `.map([GROUP_M, 1], num_tile_blocks)`; `alpha` has one f32 per output column.
    #[cutile::entry(preconditions = (
        dim(x, 1) == dim(w, 1),
        dim(x_scales, 1) == dim(w_scales, 1),
        dim(x, 0) == dim(x_scales, 0),
        dim(w, 0) == dim(w_scales, 0),
    ))]
    pub fn gemm_nvfp4<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BKP: i32,
        const BKS: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<bf16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: &Tensor<f32, { [-1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BKP]);
        let part_w = w.partition(const_shape![BN, BKP]);
        let part_xs = x_scales.partition(const_shape![BM, BKS]);
        let part_ws = w_scales.partition(const_shape![BN, BKS]);
        let part_a = alpha.partition(const_shape![BN]);
        for out_idx in z.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for k in 0i32..num_tiles(&part_x, 1) {
                let tx = part_x.load([bid_m, k]).unpack(const_shape![BM, BK]);
                let tw = part_w.load([bid_n, k]).unpack(const_shape![BN, BK]).transpose();
                let txs = part_xs.load([bid_m, k]);
                let tws = part_ws.load([bid_n, k]).transpose();
                acc = mmaf_scaled(tx, tw, acc, txs, tws);
            }
            let a: Tile<f32, { [BN] }> = part_a.load([bid_n]);
            let a2: Tile<f32, { [BM, BN] }> = a.reshape(const_shape![1, BN]).broadcast(const_shape![BM, BN]);
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc * a2);
            z.store(out, out_idx);
        }
    }

    /// Raw-schedule variant: 1-D grid, grouped-M swizzle computed in the kernel, scalar per-half alpha.
    /// Output columns `< split_tile * BN` get `alpha_lo`, the rest `alpha_hi`. Generics: `BM, BN, BK, BK/2, BK/16, GM`.
    /// Safety: no bounds checks; the host wrapper validates every shape.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn gemm_nvfp4_raw<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BKP: i32,
        const BKS: i32,
        const GM: i32,
    >(
        z: &Tensor<bf16, { [-1, -1] }>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha_lo: f32,
        alpha_hi: f32,
        split_tile: i32,
        num_bid_m: i32,
        num_bid_n: i32,
        num_blocks: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BKP]);
        let part_w = w.partition(const_shape![BN, BKP]);
        let part_xs = x_scales.partition(const_shape![BM, BKS]);
        let part_ws = w_scales.partition(const_shape![BN, BKS]);
        let mut part_z = unsafe { z.partition_full_mut(const_shape![BM, BN]) };
        let total = num_bid_m * num_bid_n;
        let group = GM * num_bid_n;
        let mut tile_id = get_tile_block_id().0;
        while tile_id < total {
            let first_m = (tile_id / group) * GM;
            let rem_m = num_bid_m - first_m;
            let gsz = if rem_m < GM { rem_m } else { GM };
            let in_group = tile_id % group;
            let bid_m = first_m + in_group % gsz;
            let bid_n = in_group / gsz;
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for k in 0i32..num_tiles(&part_x, 1) {
                let tx = part_x.load([bid_m, k]).unpack(const_shape![BM, BK]);
                let tw = part_w.load([bid_n, k]).unpack(const_shape![BN, BK]).transpose();
                let txs = part_xs.load([bid_m, k]);
                let tws = part_ws.load([bid_n, k]).transpose();
                acc = mmaf_scaled(tx, tw, acc, txs, tws);
            }
            let a: f32 = if bid_n < split_tile { alpha_lo } else { alpha_hi };
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc * broadcast_scalar(a, const_shape![BM, BN]));
            part_z.store(out, [bid_m, bid_n]);
            tile_id = tile_id + num_blocks;
        }
    }
}

/// Which GEMM entry point to launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmKernel {
    /// [`kernels::gemm_nvfp4_raw`]: grouped-M swizzle computed in the kernel, scalar per-half alpha, no in-kernel
    /// bounds checks (the host validates every shape). The production path.
    Raw,
    /// [`kernels::gemm_nvfp4`]: safe mapped-partition schedule, per-column alpha vector. Slower (the [BM, BN] alpha
    /// broadcast and the remaining in-kernel checks cost registers); kept as the checked reference variant.
    Mapped,
}

/// Tile configuration of the NVFP4 GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmTiles {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    /// Output-tile rows per rasterization band (grouped-M); 1 = row-major walk.
    pub group_m: usize,
    /// Persistent tile blocks per SM; 0 = one tile block per output tile (no persistence).
    pub blocks_per_sm: usize,
    pub kernel: GemmKernel,
}

impl GemmTiles {
    /// Default for elephant-1's MLP shapes on GB10 (sm_121), from `benchmarks/s1/bench_nvfp4.rs`.
    pub const DEFAULT: GemmTiles =
        GemmTiles { bm: 128, bn: 128, bk: 128, group_m: 8, blocks_per_sm: 0, kernel: GemmKernel::Raw };

    /// Generics of the selected kernel (`BM, BN, BK, BK/2, BK/16, GROUP_M` and, for `Mapped`, the map's `1`).
    pub fn generics(&self) -> Vec<String> {
        let mut g = vec![self.bm, self.bn, self.bk, self.bk / 2, self.bk / GROUP, self.group_m];
        if self.kernel == GemmKernel::Mapped {
            g.push(1);
        }
        g.iter().map(|v| v.to_string()).collect()
    }

    /// Number of tile blocks to launch for an `[m, n]` output on a device with `sms` multiprocessors.
    pub fn num_tile_blocks(&self, m: usize, n: usize, sms: usize) -> u32 {
        let tiles = m.div_ceil(self.bm) * n.div_ceil(self.bn);
        let blocks = if self.blocks_per_sm == 0 { tiles } else { tiles.min(self.blocks_per_sm * sms) };
        blocks.max(1) as u32
    }

    fn validate(&self, n: usize, k: usize) -> Result<()> {
        ensure!(self.bk % 32 == 0 && self.bm > 0 && self.bn > 0 && self.group_m > 0, "bad tiles {self:?}");
        ensure!(k % self.bk == 0, "K = {k} is not a multiple of BK = {}", self.bk);
        ensure!(n % self.bn == 0, "N = {n} is not a multiple of BN = {}", self.bn);
        Ok(())
    }
}

/// Tile configuration of [`kernels::quantize_nvfp4`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantTiles {
    pub br: usize,
    pub bc: usize,
}

impl QuantTiles {
    pub const DEFAULT: QuantTiles = QuantTiles { br: 16, bc: 256 };

    pub fn generics(&self) -> Vec<String> {
        [self.br, self.bc, self.bc / 2, self.bc / GROUP].iter().map(|v| v.to_string()).collect()
    }
}

/// One NVFP4 projection as stored in the checkpoint (`weight_packed`, `weight_scale`, global scales).
#[derive(Debug, Clone, Copy)]
pub struct CheckpointProj<'a> {
    /// Output features.
    pub n: usize,
    /// Input features.
    pub k: usize,
    /// `weight_packed` u8 `[n, k/2]`.
    pub weight_packed: &'a [u8],
    /// `weight_scale` e4m3 bytes `[n, k/16]`.
    pub weight_scale: &'a [u8],
    pub weight_global_scale: f32,
    pub input_global_scale: f32,
}

/// Host-side kernel layout of one (possibly fused) NVFP4 linear: `[N, K]` with N = Σ projection rows.
#[derive(Debug, Clone)]
pub struct PreparedNvfp4 {
    pub n: usize,
    pub k: usize,
    /// e2m1 pairs `[n, k/2]`, low nibble = even k.
    pub packed: Vec<u8>,
    /// e4m3 bytes `[n, k/16]`.
    pub scales: Vec<u8>,
    /// Per output column: `1 / (input_global_scale · weight_global_scale)` rounded once to f32.
    pub alpha: Vec<f32>,
    /// The shared `input_global_scale` the activations are quantized with.
    pub input_global_scale: f32,
    /// Columns `< split_n` use `alpha_lo`, the rest `alpha_hi` (the two projections of a fused gate/up; for one
    /// projection `split_n = n` and both are equal). Used by [`kernels::gemm_nvfp4_raw`].
    pub split_n: usize,
    pub alpha_lo: f32,
    pub alpha_hi: f32,
}

/// Convert checkpoint projections sharing one input (gate and up, or down alone) into the kernel layout: rows
/// concatenated in the given order. Run once at load. Refuses projections whose `input_global_scale` or K
/// differ, since the input is quantized once for all of them (DESIGN §3.1).
pub fn prepare_weights(projs: &[CheckpointProj<'_>]) -> Result<PreparedNvfp4> {
    let Some(first) = projs.first() else { bail!("prepare_weights: no projections") };
    let (k, igs) = (first.k, first.input_global_scale);
    ensure!(k % GROUP == 0, "K = {k} is not a multiple of {GROUP}");
    ensure!(igs.is_finite() && igs > 0.0, "input_global_scale {igs} is not a positive finite value");
    let n: usize = projs.iter().map(|p| p.n).sum();
    let mut out = PreparedNvfp4 {
        n,
        k,
        packed: Vec::with_capacity(n * k / 2),
        scales: Vec::with_capacity(n * k / GROUP),
        alpha: Vec::with_capacity(n),
        input_global_scale: igs,
        split_n: first.n,
        alpha_lo: 0.0,
        alpha_hi: 0.0,
    };
    ensure!(projs.len() <= 2, "prepare_weights: at most two projections share one input (gate, up)");
    for (i, p) in projs.iter().enumerate() {
        ensure!(p.k == k, "projection {i}: K = {} differs from {k}", p.k);
        ensure!(
            p.input_global_scale.to_bits() == igs.to_bits(),
            "projection {i}: input_global_scale {} differs from {igs}; the shared input cannot be quantized once",
            p.input_global_scale
        );
        ensure!(p.weight_packed.len() == p.n * k / 2, "projection {i}: weight_packed has {} bytes", p.weight_packed.len());
        ensure!(p.weight_scale.len() == p.n * k / GROUP, "projection {i}: weight_scale has {} bytes", p.weight_scale.len());
        let wgs = p.weight_global_scale;
        ensure!(wgs.is_finite() && wgs > 0.0, "projection {i}: weight_global_scale {wgs} is not positive finite");
        out.packed.extend_from_slice(p.weight_packed);
        out.scales.extend_from_slice(p.weight_scale);
        let alpha = (1.0f64 / (igs as f64 * wgs as f64)) as f32;
        out.alpha.extend(std::iter::repeat_n(alpha, p.n));
        if i == 0 {
            out.alpha_lo = alpha;
        }
        out.alpha_hi = alpha;
    }
    if projs.len() == 1 {
        out.split_n = n;
    }
    Ok(out)
}

/// Device-resident NVFP4 linear in the kernel layout.
pub struct Nvfp4Linear {
    pub n: usize,
    pub k: usize,
    pub packed: Arc<Tensor<f4e2m1fnx2>>,
    pub scales: Arc<Tensor<f8e4m3fn>>,
    /// Per-column alpha (used by the `Mapped` kernel).
    pub alpha: Arc<Tensor<f32>>,
    pub input_global_scale: f32,
    /// Columns `< split_n` use `alpha_lo`, the rest `alpha_hi` (used by the `Raw` kernel).
    pub split_n: usize,
    pub alpha_lo: f32,
    pub alpha_hi: f32,
}

impl Nvfp4Linear {
    pub fn upload(p: &PreparedNvfp4, stream: &Arc<Stream>) -> Result<Self> {
        let packed = Arc::new(p.packed.iter().map(|&b| f4e2m1fnx2::from_bits(b)).collect::<Vec<_>>());
        let scales = Arc::new(p.scales.iter().map(|&b| f8e4m3fn(b)).collect::<Vec<_>>());
        Ok(Nvfp4Linear {
            n: p.n,
            k: p.k,
            packed: api::copy_host_vec_to_device(&packed).reshape(&[p.n, p.k / 2]).sync_on(stream)?.into(),
            scales: api::copy_host_vec_to_device(&scales).reshape(&[p.n, p.k / GROUP]).sync_on(stream)?.into(),
            alpha: api::copy_host_vec_to_device(&Arc::new(p.alpha.clone())).sync_on(stream)?.into(),
            input_global_scale: p.input_global_scale,
            split_n: p.split_n,
            alpha_lo: p.alpha_lo,
            alpha_hi: p.alpha_hi,
        })
    }
}

/// NVFP4 activations: packed `[T, K/2]` plus e4m3 scales `[T, K/16]`.
pub struct Nvfp4Act {
    pub t: usize,
    pub k: usize,
    pub packed: Tensor<f4e2m1fnx2>,
    pub scales: Tensor<f8e4m3fn>,
}

impl Nvfp4Act {
    pub fn zeros(t: usize, k: usize, stream: &Arc<Stream>) -> Result<Self> {
        Ok(Nvfp4Act {
            t,
            k,
            packed: api::zeros::<f4e2m1fnx2>(&[t, k / 2]).sync_on(stream)?,
            scales: api::zeros::<f8e4m3fn>(&[t, k / GROUP]).sync_on(stream)?,
        })
    }
}

/// Quantize bf16 `x [T, K]` into `out`, as a device operation (enqueue with `execute`/`sync_on`, or capture).
pub fn quantize_op(
    x: &Arc<Tensor<bf16>>,
    igs: f32,
    out: Nvfp4Act,
    tiles: QuantTiles,
) -> Result<impl DeviceOp<Output = Nvfp4Act>> {
    let shape = x.shape();
    ensure!(shape.len() == 2, "quantize: x must be rank 2, got {shape:?}");
    let (t, k) = (shape[0] as usize, shape[1] as usize);
    ensure!(t == out.t && k == out.k, "quantize: x is [{t}, {k}] but out is [{}, {}]", out.t, out.k);
    ensure!(k % tiles.bc == 0 && tiles.bc % GROUP == 0, "quantize: K = {k} vs BC = {}", tiles.bc);
    let q = out.packed.partition([tiles.br, tiles.bc / 2]);
    let s = out.scales.partition([tiles.br, tiles.bc / GROUP]);
    Ok(kernels::quantize_nvfp4(q, s, x.clone(), igs)
        .generics(tiles.generics())
        .map(move |(q, s, _, _)| Nvfp4Act { t, k, packed: q.unpartition(), scales: s.unpartition() }))
}

/// Synchronous [`quantize_op`].
pub fn quantize(
    x: &Arc<Tensor<bf16>>,
    igs: f32,
    out: Nvfp4Act,
    tiles: QuantTiles,
    stream: &Arc<Stream>,
) -> Result<Nvfp4Act> {
    Ok(quantize_op(x, igs, out, tiles)?.sync_on(stream)?)
}

fn check_gemm_shapes(
    x: &Tensor<f4e2m1fnx2>,
    x_scales: &Tensor<f8e4m3fn>,
    w: &Nvfp4Linear,
    z: &Tensor<bf16>,
    tiles: &GemmTiles,
) -> Result<usize> {
    let t = x.shape()[0] as usize;
    ensure!(x.shape() == [t as i32, (w.k / 2) as i32], "gemm: x is {:?}, W has K = {}", x.shape(), w.k);
    ensure!(x_scales.shape() == [t as i32, (w.k / GROUP) as i32], "gemm: x_scales is {:?}", x_scales.shape());
    ensure!(w.packed.shape() == [w.n as i32, (w.k / 2) as i32], "gemm: W packed is {:?}", w.packed.shape());
    ensure!(w.scales.shape() == [w.n as i32, (w.k / GROUP) as i32], "gemm: W scales is {:?}", w.scales.shape());
    ensure!(w.alpha.shape() == [w.n as i32], "gemm: alpha is {:?}", w.alpha.shape());
    ensure!(z.shape() == [t as i32, w.n as i32], "gemm: z is {:?}, expected [{t}, {}]", z.shape(), w.n);
    tiles.validate(w.n, w.k)?;
    ensure!(w.split_n % tiles.bn == 0, "gemm: alpha split at column {} is not a multiple of BN", w.split_n);
    Ok(t)
}

/// `z [T, N] = bf16(alpha ⊙ (x_q @ W_q^T))` with the `Raw` kernel, as a device operation writing into `z`.
/// `sms` is the device's multiprocessor count (for persistent schedules).
pub fn gemm_op(
    x: &Arc<Tensor<f4e2m1fnx2>>,
    x_scales: &Arc<Tensor<f8e4m3fn>>,
    w: &Nvfp4Linear,
    z: Arc<Tensor<bf16>>,
    tiles: GemmTiles,
    sms: usize,
) -> Result<impl DeviceOp<Output = Arc<Tensor<bf16>>>> {
    ensure!(tiles.kernel == GemmKernel::Raw, "gemm_op launches the Raw kernel");
    check_gemm_shapes(x, x_scales, w, &z, &tiles)?;
    let alpha = RawAlpha { lo: w.alpha_lo, hi: w.alpha_hi, split_n: w.split_n };
    gemm_raw_op(x, x_scales, &w.packed, &w.scales, alpha, z, tiles, sms)
}

/// The scalar epilogue of [`kernels::gemm_nvfp4_raw`]: output columns `< split_n` are scaled by `lo`, the rest by
/// `hi` (the two halves of a fused gate/up; one projection has `lo == hi`, `split_n = N`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawAlpha {
    pub lo: f32,
    pub hi: f32,
    pub split_n: usize,
}

impl RawAlpha {
    /// The epilogue of projections that share one `input_global_scale` `igs`, given each part's
    /// `(rows, weight_global_scale)` in row order: `1 / (igs · wgs)` rounded once to f32 (as [`prepare_weights`]).
    pub fn for_parts(igs: f32, parts: &[(usize, f32)]) -> Result<Self> {
        ensure!(!parts.is_empty() && parts.len() <= 2, "{} parts: the raw epilogue takes one or two", parts.len());
        let alpha = |wgs: f32| (1.0f64 / (igs as f64 * wgs as f64)) as f32;
        let n: usize = parts.iter().map(|p| p.0).sum();
        Ok(RawAlpha {
            lo: alpha(parts[0].1),
            hi: alpha(parts[parts.len() - 1].1),
            split_n: if parts.len() == 1 { n } else { parts[0].0 },
        })
    }
}

/// The `Raw` GEMM on borrowed weight tensors (`w_packed [N, K/2]`, `w_scales [N, K/16]`), as a device operation
/// writing into `z [T, N]`. Every shape is validated here; the kernel itself has no bounds checks.
#[allow(clippy::too_many_arguments)]
pub fn gemm_raw_op(
    x: &Arc<Tensor<f4e2m1fnx2>>,
    x_scales: &Arc<Tensor<f8e4m3fn>>,
    w_packed: &Arc<Tensor<f4e2m1fnx2>>,
    w_scales: &Arc<Tensor<f8e4m3fn>>,
    alpha: RawAlpha,
    z: Arc<Tensor<bf16>>,
    tiles: GemmTiles,
    sms: usize,
) -> Result<impl DeviceOp<Output = Arc<Tensor<bf16>>>> {
    gemm_raw_op_with(x, x_scales, w_packed, w_scales, alpha, z, tiles, sms, CompileOptions::default())
}

/// [`gemm_raw_op`] with explicit cuTile compile options (occupancy / warps per CTA hints; F1 experiments).
#[allow(clippy::too_many_arguments)]
pub fn gemm_raw_op_with(
    x: &Arc<Tensor<f4e2m1fnx2>>,
    x_scales: &Arc<Tensor<f8e4m3fn>>,
    w_packed: &Arc<Tensor<f4e2m1fnx2>>,
    w_scales: &Arc<Tensor<f8e4m3fn>>,
    alpha: RawAlpha,
    z: Arc<Tensor<bf16>>,
    tiles: GemmTiles,
    sms: usize,
    opts: CompileOptions,
) -> Result<impl DeviceOp<Output = Arc<Tensor<bf16>>>> {
    ensure!(tiles.kernel == GemmKernel::Raw, "gemm_raw_op launches the Raw kernel");
    ensure!(w_packed.shape().len() == 2 && x.shape().len() == 2, "gemm: rank-2 operands expected");
    let (n, k) = (w_packed.shape()[0] as usize, 2 * w_packed.shape()[1] as usize);
    let t = x.shape()[0] as usize;
    ensure!(x.shape() == [t as i32, (k / 2) as i32], "gemm: x is {:?}, W has K = {k}", x.shape());
    ensure!(x_scales.shape() == [t as i32, (k / GROUP) as i32], "gemm: x_scales is {:?}", x_scales.shape());
    ensure!(w_scales.shape() == [n as i32, (k / GROUP) as i32], "gemm: W scales is {:?}", w_scales.shape());
    ensure!(z.shape() == [t as i32, n as i32], "gemm: z is {:?}, expected [{t}, {n}]", z.shape());
    tiles.validate(n, k)?;
    ensure!(alpha.split_n % tiles.bn == 0 && alpha.split_n <= n, "gemm: alpha split at column {} (N = {n}, BN = {})",
            alpha.split_n, tiles.bn);
    let (nm, nn) = (t.div_ceil(tiles.bm), n / tiles.bn);
    let ntb = tiles.num_tile_blocks(t, n, sms);
    // Safety: every operand was validated against (T, N, K) and the tiles above; the kernel only touches tiles
    // bid_m < ceil(T/BM), bid_n < N/BN, k < K/BK, and partial row tiles are clipped by the partition views.
    let op = unsafe {
        kernels::gemm_nvfp4_raw(
            z,
            x.clone(),
            w_packed.clone(),
            x_scales.clone(),
            w_scales.clone(),
            alpha.lo,
            alpha.hi,
            (alpha.split_n / tiles.bn) as i32,
            nm as i32,
            nn as i32,
            ntb as i32,
        )
    };
    Ok(op.generics(tiles.generics()).grid((ntb, 1, 1)).compile_options(opts).map(|out| out.0))
}

/// Packed NVFP4 values and their block scales, as device tensors.
pub type PackedAct = (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>);

/// The fused gate/up GEMM + SiLU·mul + quantize ([`kernels::gemm_nvfp4_swiglu_il`]) on a gate/up projection
/// whose rows were interleaved by [`interleave_gate_up`] with block `tiles.bn / 2`: `w_packed_il [2I, K/2]`,
/// `w_scales_il [2I, K/16]`, `alpha` the gate/up epilogue (gate part's `lo`, up part's `hi`; its split must be at
/// `I`), `igs` the down projection's `input_global_scale`. Writes the quantized product `q [T, I/2]`,
/// `s [T, I/16]`: `tiles.bn / 2` product columns per tile. Every shape is validated here.
#[allow(clippy::too_many_arguments)]
pub fn gemm_swiglu_il_op(
    x: &Arc<Tensor<f4e2m1fnx2>>,
    x_scales: &Arc<Tensor<f8e4m3fn>>,
    w_packed_il: &Arc<Tensor<f4e2m1fnx2>>,
    w_scales_il: &Arc<Tensor<f8e4m3fn>>,
    alpha: RawAlpha,
    igs: f32,
    q: Arc<Tensor<f4e2m1fnx2>>,
    s: Arc<Tensor<f8e4m3fn>>,
    tiles: GemmTiles,
    sms: usize,
) -> Result<impl DeviceOp<Output = PackedAct>> {
    ensure!(tiles.kernel == GemmKernel::Raw, "the fused gate/up GEMM uses the raw schedule");
    ensure!(w_packed_il.shape().len() == 2 && x.shape().len() == 2, "gemm: rank-2 operands expected");
    let (n2, k) = (w_packed_il.shape()[0] as usize, 2 * w_packed_il.shape()[1] as usize);
    ensure!(n2.is_multiple_of(2), "gate/up weight has an odd number of rows ({n2})");
    let i = n2 / 2;
    let t = x.shape()[0] as usize;
    ensure!(alpha.split_n == i, "gate/up alpha split at {} but I = {i}", alpha.split_n);
    ensure!(x.shape() == [t as i32, (k / 2) as i32], "gemm: x is {:?}, W has K = {k}", x.shape());
    ensure!(x_scales.shape() == [t as i32, (k / GROUP) as i32], "gemm: x_scales is {:?}", x_scales.shape());
    ensure!(w_scales_il.shape() == [n2 as i32, (k / GROUP) as i32], "gemm: W scales is {:?}", w_scales_il.shape());
    ensure!(q.shape() == [t as i32, (i / 2) as i32], "swiglu: q is {:?}, expected [{t}, {}]", q.shape(), i / 2);
    ensure!(s.shape() == [t as i32, (i / GROUP) as i32], "swiglu: s is {:?}, expected [{t}, {}]", s.shape(), i / GROUP);
    tiles.validate(n2, k)?;
    // Product columns per tile.
    let h = tiles.bn / 2;
    ensure!(tiles.bn.is_multiple_of(2) && h.is_multiple_of(GROUP) && i.is_multiple_of(h), "{h} product columns per tile do not tile I = {i}");
    let (nm, nn) = (t.div_ceil(tiles.bm), i / h);
    let ntb = tiles.num_tile_blocks(t, n2, sms);
    let mut g = tiles.generics();
    g.extend([h.to_string(), (h / 2).to_string(), (h / GROUP).to_string()]);
    // Safety: every operand was validated against (T, I, K) and the tiles above; the kernel only touches weight
    // row tiles bid_n < 2I/BN, activation tiles bid_m < ceil(T/BM), k < K/BK, output tiles [BM, H/2] / [BM, H/16]
    // at (bid_m, bid_n < I/H); partial row tiles are clipped by the partition views.
    let op = unsafe {
        kernels::gemm_nvfp4_swiglu_il(
            q, s, x.clone(), w_packed_il.clone(), x_scales.clone(), w_scales_il.clone(), alpha.lo, alpha.hi, igs,
            nm as i32, nn as i32, ntb as i32,
        )
    };
    Ok(op.generics(g).grid((ntb, 1, 1)).map(|out| (out.0, out.1)))
}

/// Host-side row interleave of a concatenated gate/up projection (`rows = 2I`, `row_bytes` bytes per row, gate
/// rows first): output row block `j` (of `2·block` rows) is gate rows `j·block..(j+1)·block` followed by up rows
/// `j·block..(j+1)·block`. Values are unchanged (a permutation of rows), for packed values and scales alike.
pub fn interleave_gate_up(data: &[u8], rows: usize, row_bytes: usize, block: usize) -> Result<Vec<u8>> {
    ensure!(rows.is_multiple_of(2) && data.len() == rows * row_bytes, "interleave: {} bytes for {rows} x {row_bytes}", data.len());
    let i = rows / 2;
    ensure!(block > 0 && i.is_multiple_of(block), "interleave: I = {i} is not a multiple of {block}");
    let mut out = Vec::with_capacity(data.len());
    for j in 0..i / block {
        for half in 0..2 {
            let r0 = half * i + j * block;
            out.extend_from_slice(&data[r0 * row_bytes..(r0 + block) * row_bytes]);
        }
    }
    Ok(out)
}


/// Tile shape and occupancy hint of [`silu_mul_nvfp4_with`] (F1). Every shape gives the same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiluTiles {
    /// Rows per tile block.
    pub rows: usize,
    /// Columns (of the `I` outputs) per tile block; a multiple of 16 dividing `I`.
    pub cols: usize,
    /// cuTile `occupancy` hint (tile blocks per SM the compiler should allow); `None`: compiler default.
    pub occupancy: Option<i32>,
}

impl SiluTiles {
    /// N1's shape: one row of 1,024 columns, compiler default (registers cap it at one tile block per SM).
    pub const N1: SiluTiles = SiluTiles { rows: 1, cols: SILU_Q_BLOCK, occupancy: None };
}

/// Columns of one row per tile block of [`kernels::silu_mul_nvfp4`].
pub const SILU_Q_BLOCK: usize = 1024;

/// [`silu_mul_nvfp4`] with an explicit tile shape ([`kernels::silu_mul_nvfp4_rb`]). `T` must be a multiple of
/// `tiles.rows` and `I` of `tiles.cols`.
pub fn silu_mul_nvfp4_with<'a>(
    q: &'a mut Tensor<f4e2m1fnx2>,
    s: &'a mut Tensor<f8e4m3fn>,
    gu: &'a Tensor<bf16>,
    igs: f32,
    tiles: SiluTiles,
) -> Result<impl GraphNode + 'a> {
    let sh = gu.shape();
    ensure!(sh.len() == 2 && (sh[1] as usize).is_multiple_of(2), "gu must be [T, 2I], got {sh:?}");
    let (t, i) = (sh[0] as usize, sh[1] as usize / 2);
    let (r, b) = (tiles.rows, tiles.cols);
    ensure!(b.is_multiple_of(GROUP) && i.is_multiple_of(b), "I = {i} is not a multiple of B = {b} (B % 16 == 0)");
    ensure!(r > 0 && t.is_multiple_of(r), "T = {t} is not a multiple of R = {r}");
    check_act(t, i, q, s)?;
    let g = vec![i.to_string(), r.to_string(), b.to_string(), (b / 2).to_string(), (b / GROUP).to_string()];
    let opts = CompileOptions { occupancy: tiles.occupancy, ..Default::default() };
    Ok(kernels::silu_mul_nvfp4_rb(q.partition([r, b / 2]), s.partition([r, b / GROUP]), gu, igs)
        .generics(g)
        .compile_options(opts))
}

fn check_act(t: usize, k: usize, q: &Tensor<f4e2m1fnx2>, s: &Tensor<f8e4m3fn>) -> Result<()> {
    ensure!(k % GROUP == 0, "K = {k} is not a multiple of {GROUP}");
    ensure!(q.shape() == [t as i32, (k / 2) as i32], "q is {:?}, expected [{t}, {}]", q.shape(), k / 2);
    ensure!(s.shape() == [t as i32, (k / GROUP) as i32], "s is {:?}, expected [{t}, {}]", s.shape(), k / GROUP);
    Ok(())
}

/// `q, s = nvfp4(bf16(zc_rmsnorm(x, w)))` for `x [T, N]`, `w [N]`: [`kernels::zc_rmsnorm_nvfp4`].
pub fn zc_rmsnorm_nvfp4<'a>(
    q: &'a mut Tensor<f4e2m1fnx2>,
    s: &'a mut Tensor<f8e4m3fn>,
    x: &'a Tensor<bf16>,
    w: &'a Tensor<bf16>,
    eps: f32,
    igs: f32,
) -> Result<impl GraphNode + 'a> {
    ensure!(x.shape().len() == 2, "x must be [T, N], got {:?}", x.shape());
    let (t, n) = (x.shape()[0] as usize, x.shape()[1] as usize);
    ensure!(w.shape() == [n as i32], "w is {:?}, expected [{n}]", w.shape());
    check_act(t, n, q, s)?;
    let g = vec![n.to_string(), (n / 2).to_string(), (n / GROUP).to_string()];
    Ok(kernels::zc_rmsnorm_nvfp4(q.partition([1, n / 2]), s.partition([1, n / GROUP]), x, w, eps, igs).generics(g))
}

/// `residual = bf16(residual + x)` in place, then `q, s = nvfp4(bf16(zc_rmsnorm(residual, w)))`:
/// [`kernels::add_zc_rmsnorm_nvfp4`].
#[allow(clippy::too_many_arguments)]
pub fn add_zc_rmsnorm_nvfp4<'a>(
    residual: &'a mut Tensor<bf16>,
    q: &'a mut Tensor<f4e2m1fnx2>,
    s: &'a mut Tensor<f8e4m3fn>,
    x: &'a Tensor<bf16>,
    w: &'a Tensor<bf16>,
    eps: f32,
    igs: f32,
) -> Result<impl GraphNode + 'a> {
    add_zc_rmsnorm_nvfp4_with(residual, q, s, x, w, eps, igs, None)
}

/// [`add_zc_rmsnorm_nvfp4`] with a cuTile `occupancy` hint (F1; byte-identical results).
#[allow(clippy::too_many_arguments)]
pub fn add_zc_rmsnorm_nvfp4_with<'a>(
    residual: &'a mut Tensor<bf16>,
    q: &'a mut Tensor<f4e2m1fnx2>,
    s: &'a mut Tensor<f8e4m3fn>,
    x: &'a Tensor<bf16>,
    w: &'a Tensor<bf16>,
    eps: f32,
    igs: f32,
    occupancy: Option<i32>,
) -> Result<impl GraphNode + 'a> {
    ensure!(x.shape().len() == 2 && residual.shape() == x.shape(), "residual {:?} vs x {:?}", residual.shape(), x.shape());
    let (t, n) = (x.shape()[0] as usize, x.shape()[1] as usize);
    ensure!(w.shape() == [n as i32], "w is {:?}, expected [{n}]", w.shape());
    check_act(t, n, q, s)?;
    let g = vec![n.to_string(), (n / 2).to_string(), (n / GROUP).to_string()];
    let opts = CompileOptions { occupancy, ..Default::default() };
    Ok(kernels::add_zc_rmsnorm_nvfp4(residual.partition([1, n]), q.partition([1, n / 2]), s.partition([1, n / GROUP]),
                                     x, w, eps, igs)
        .generics(g)
        .compile_options(opts))
}

/// `q, s = nvfp4(bf16(bf16(silu(gu[:, :I])) · gu[:, I:]))` for `gu [T, 2I]`: [`kernels::silu_mul_nvfp4`].
/// `I` must be a multiple of [`SILU_Q_BLOCK`].
pub fn silu_mul_nvfp4<'a>(
    q: &'a mut Tensor<f4e2m1fnx2>,
    s: &'a mut Tensor<f8e4m3fn>,
    gu: &'a Tensor<bf16>,
    igs: f32,
) -> Result<impl GraphNode + 'a> {
    let sh = gu.shape();
    ensure!(sh.len() == 2 && (sh[1] as usize).is_multiple_of(2), "gu must be [T, 2I], got {sh:?}");
    let (t, i) = (sh[0] as usize, sh[1] as usize / 2);
    ensure!(i.is_multiple_of(SILU_Q_BLOCK), "I = {i} is not a multiple of {SILU_Q_BLOCK}");
    check_act(t, i, q, s)?;
    let b = SILU_Q_BLOCK;
    let g = vec![i.to_string(), b.to_string(), (b / 2).to_string(), (b / GROUP).to_string()];
    Ok(kernels::silu_mul_nvfp4(q.partition([1, b / 2]), s.partition([1, b / GROUP]), gu, igs).generics(g))
}

/// Synchronous GEMM with either kernel. `z` must be `[T, N]` bf16.
pub fn gemm(
    x: &Arc<Tensor<f4e2m1fnx2>>,
    x_scales: &Arc<Tensor<f8e4m3fn>>,
    w: &Nvfp4Linear,
    z: Tensor<bf16>,
    tiles: GemmTiles,
    sms: usize,
    stream: &Arc<Stream>,
) -> Result<Tensor<bf16>> {
    match tiles.kernel {
        GemmKernel::Raw => {
            let z = gemm_op(x, x_scales, w, Arc::new(z), tiles, sms)?.sync_on(stream)?;
            Arc::try_unwrap(z).map_err(|_| anyhow::anyhow!("gemm: output tensor still shared"))
        }
        GemmKernel::Mapped => {
            let t = check_gemm_shapes(x, x_scales, w, &z, &tiles)?;
            let zp = z.partition([tiles.bm, tiles.bn]).map([tiles.group_m, 1], tiles.num_tile_blocks(t, w.n, sms));
            let (zp, ..) =
                kernels::gemm_nvfp4(zp, x.clone(), w.packed.clone(), x_scales.clone(), w.scales.clone(), w.alpha.clone())
                    .generics(tiles.generics())
                    .sync_on(stream)?;
            Ok(zp.unpartition())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_concatenates_and_sets_per_half_alpha() {
        let (k, ng, nu) = (32usize, 2usize, 3usize);
        let gp: Vec<u8> = (0..ng * k / 2).map(|i| i as u8).collect();
        let gs: Vec<u8> = (0..ng * k / 16).map(|i| 0x30 + i as u8).collect();
        let up: Vec<u8> = (0..nu * k / 2).map(|i| 200 - i as u8).collect();
        let us: Vec<u8> = (0..nu * k / 16).map(|i| 0x40 + i as u8).collect();
        let proj = |n, p: &'static [u8], s: &'static [u8], wgs| CheckpointProj {
            n,
            k,
            weight_packed: p,
            weight_scale: s,
            weight_global_scale: wgs,
            input_global_scale: 628.0,
        };
        let (gp, gs, up, us): (&'static [u8], &'static [u8], &'static [u8], &'static [u8]) =
            (gp.leak(), gs.leak(), up.leak(), us.leak());
        let p = prepare_weights(&[proj(ng, gp, gs, 9408.0), proj(nu, up, us, 4128.0)]).unwrap();
        assert_eq!((p.n, p.k), (ng + nu, k));
        assert_eq!(&p.packed[..gp.len()], gp);
        assert_eq!(&p.packed[gp.len()..], up);
        assert_eq!(&p.scales[..gs.len()], gs);
        assert_eq!(&p.scales[gs.len()..], us);
        assert_eq!(p.alpha[0], (1.0 / (628.0f64 * 9408.0)) as f32);
        assert_eq!(p.alpha[ng], (1.0 / (628.0f64 * 4128.0)) as f32);
        assert_eq!(p.alpha.len(), ng + nu);
    }

    #[test]
    fn raw_alpha_for_parts_matches_prepare_weights() {
        // The engine computes the epilogue from the loaded parts (RawAlpha); it must equal prepare_weights'.
        let k = 32usize;
        let (p, s) = (vec![0u8; 256 * k / 2], vec![0x38u8; 256 * k / 16]);
        let proj = |wgs| CheckpointProj {
            n: 256,
            k,
            weight_packed: &p,
            weight_scale: &s,
            weight_global_scale: wgs,
            input_global_scale: 628.0,
        };
        let two = prepare_weights(&[proj(9408.0), proj(4128.0)]).unwrap();
        let a = RawAlpha::for_parts(628.0, &[(256, 9408.0), (256, 4128.0)]).unwrap();
        assert_eq!((a.lo, a.hi, a.split_n), (two.alpha_lo, two.alpha_hi, two.split_n));
        let one = prepare_weights(&[proj(9408.0)]).unwrap();
        let a = RawAlpha::for_parts(628.0, &[(256, 9408.0)]).unwrap();
        assert_eq!((a.lo, a.hi, a.split_n), (one.alpha_lo, one.alpha_hi, one.split_n));
        assert!(RawAlpha::for_parts(628.0, &[]).is_err());
    }

    #[test]
    fn prepare_refuses_mismatched_input_global_scale() {
        let p = [0u8; 16];
        let s = [0x38u8; 2];
        let a = CheckpointProj {
            n: 1,
            k: 32,
            weight_packed: &p,
            weight_scale: &s,
            weight_global_scale: 1.0,
            input_global_scale: 2.0,
        };
        let b = CheckpointProj { input_global_scale: 3.0, ..a };
        assert!(prepare_weights(&[a, b]).is_err());
        assert!(prepare_weights(&[a, a]).is_ok());
    }

    #[test]
    fn gemm_tiles_generics() {
        assert_eq!(GemmTiles::DEFAULT.generics(), ["128", "128", "128", "64", "8", "8"]);
        assert_eq!(GemmTiles { blocks_per_sm: 2, ..GemmTiles::DEFAULT }.num_tile_blocks(16384, 4096, 48), 96);
        assert_eq!(GemmTiles::DEFAULT.num_tile_blocks(100, 256, 48), 2);
    }
}
