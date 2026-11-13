//! Zero-centred RMSNorm (DESIGN.md §3, §6.3, §6.4). Owner: W3b.
//!
//! `Qwen3_5RMSNorm` is `y = (x_f32 * rsqrt(mean(x_f32²) + eps)) * (1 + w_f32)`, cast to bf16 at the end. The
//! kernels keep the reference's fp32 rounding points: the statistic is `sum / N` in fp32, the row is scaled
//! by `rsqrt` first and then multiplied by the fp32 value `1 + w`.
//!
//! The fused variant adds the sublayer output into the residual stream first. The decoder layer does that add
//! in bf16 (`residual + hidden_states`), so the new residual is rounded to bf16 and the norm reads the rounded
//! value, exactly as the next layer's `input_layernorm` does in the reference.
//!
//! One tile block normalises one row of `N` (4096 for elephant-1) columns, held in registers.

use cutile::half::bf16;
use cutile::prelude::*;

/// elephant-1 `rms_norm_eps`.
pub const RMS_EPS: f32 = 1e-6;

#[cutile::module]
pub mod zc_rmsnorm_module {
    use cutile::core::*;

    /// `out[t] = zc_rmsnorm(x[t]) * (1 + w)`, one row per tile block.
    #[cutile::entry()]
    pub fn zc_rmsnorm<const N: i32>(
        out: &mut Tensor<bf16, { [1, N] }>,
        x: &Tensor<bf16, { [-1, N] }>,
        w: &Tensor<bf16, { [N] }>,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_shape: Shape<{ [1, N] }> = const_shape![1, N];
        let x_part: Partition<bf16, { [1, N] }> = x.partition(row_shape);
        let tx: Tile<bf16, { [1, N] }> = x_part.load([pid.0, 0i32]);
        let xf: Tile<f32, { [1, N] }> = convert_tile(tx);
        let tw: Tile<bf16, { [N] }> = w.partition(const_shape![N]).load([0i32]);
        let wf: Tile<f32, { [1, N] }> = convert_tile(tw.reshape(row_shape));
        let one_plus_w: Tile<f32, { [1, N] }> = constant(1.0f32, row_shape) + wf;
        let y: Tile<f32, { [1, N] }> = zc_scale(xf, one_plus_w, eps);
        let yb: Tile<bf16, { [1, N] }> = convert_tile(y);
        out.store(yb);
    }

    /// Residual add then norm: `residual[t] = bf16(residual[t] + x[t])`; `out[t] = zc_rmsnorm(residual[t])`.
    /// `residual` is updated in place.
    #[cutile::entry()]
    pub fn add_zc_rmsnorm<const N: i32>(
        residual: &mut Tensor<bf16, { [1, N] }>,
        out: &mut Tensor<bf16, { [1, N] }>,
        x: &Tensor<bf16, { [-1, N] }>,
        w: &Tensor<bf16, { [N] }>,
        eps: f32,
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
        let y: Tile<f32, { [1, N] }> = zc_scale(xf, one_plus_w, eps);
        let yb: Tile<bf16, { [1, N] }> = convert_tile(y);
        out.store(yb);
    }

    /// `(x * rsqrt(sum(x²)/N + eps)) * (1 + w)` in fp32, in the reference's order.
    fn zc_scale<const N: i32>(
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
}

fn hidden_of(x: &Tensor<bf16>, w: &Tensor<bf16>) -> anyhow::Result<(usize, usize)> {
    let s = x.shape();
    anyhow::ensure!(s.len() == 2, "x must be [T, N], got {s:?}");
    anyhow::ensure!(w.shape() == [s[1]].as_slice(), "w is {:?}, expected [{}]", w.shape(), s[1]);
    Ok((s[0] as usize, s[1] as usize))
}

/// `out = zc_rmsnorm(x, w)` for `x`, `out` of shape `[T, N]` and `w` of shape `[N]`.
pub fn zc_rmsnorm<'a>(
    out: &'a mut Tensor<bf16>,
    x: &'a Tensor<bf16>,
    w: &'a Tensor<bf16>,
    eps: f32,
) -> anyhow::Result<impl GraphNode + 'a> {
    let (_, n) = hidden_of(x, w)?;
    anyhow::ensure!(out.shape() == x.shape(), "out {:?} != x {:?}", out.shape(), x.shape());
    Ok(zc_rmsnorm_module::zc_rmsnorm(out.partition([1, n]), x, w, eps).generics(vec![n.to_string()]))
}

/// `residual = bf16(residual + x)` in place, then `out = zc_rmsnorm(residual, w)`.
pub fn add_zc_rmsnorm<'a>(
    residual: &'a mut Tensor<bf16>,
    out: &'a mut Tensor<bf16>,
    x: &'a Tensor<bf16>,
    w: &'a Tensor<bf16>,
    eps: f32,
) -> anyhow::Result<impl GraphNode + 'a> {
    let (_, n) = hidden_of(x, w)?;
    anyhow::ensure!(out.shape() == x.shape() && residual.shape() == x.shape(), "shape mismatch");
    Ok(zc_rmsnorm_module::add_zc_rmsnorm(residual.partition([1, n]), out.partition([1, n]), x, w, eps)
        .generics(vec![n.to_string()]))
}
