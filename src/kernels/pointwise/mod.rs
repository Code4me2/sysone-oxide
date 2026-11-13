//! Token-parallel pointwise kernels (DESIGN.md §6.3, §6.4). Owner: W3b.
//!
//! - [`silu_mul`]: the SwiGLU product over the fused gate/up GEMM output. `Qwen3_5MLP` computes
//!   `act_fn(gate_proj(x)) * up_proj(x)` on bf16 tensors, so `silu(g)` is rounded to bf16 before the product,
//!   and the product is rounded again. The kernel keeps both rounding points.
//! - [`gather_rows`]: `out[i] = src[idx[i]]`. Serves the embedding lookup (`src` = `embed_tokens`,
//!   `idx` = token ids) and the readout gather (`src` = hidden states, `idx` = answer rows).

use cutile::half::bf16;
use cutile::prelude::*;

/// Columns handled per tile block by [`silu_mul`].
pub const SILU_BLOCK: usize = 1024;
/// Columns copied per tile block by [`gather_rows`].
pub const GATHER_BLOCK: usize = 1024;

#[cutile::module]
pub mod pointwise_module {
    use cutile::core::*;

    /// `out[t, j] = bf16(bf16(silu(gu[t, j])) * gu[t, I + j])` for a fused `[T, 2I]` gate/up row.
    #[cutile::entry()]
    pub fn silu_mul<const I: i32, const B: i32>(
        out: &mut Tensor<bf16, { [1, B] }>,
        gu: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<bf16, { [1, B] }> = gu.partition(const_shape![1, B]);
        let tg: Tile<bf16, { [1, B] }> = part.load([pid.0, pid.1]);
        let tu: Tile<bf16, { [1, B] }> = part.load([pid.0, pid.1 + I / B]);
        let g: Tile<f32, { [1, B] }> = convert_tile(tg);
        let u: Tile<f32, { [1, B] }> = convert_tile(tu);
        let one: Tile<f32, { [1, B] }> = constant(1.0f32, const_shape![1, B]);
        let s: Tile<f32, { [1, B] }> = true_div(g, one + exp(negf(g)));
        let s_b: Tile<bf16, { [1, B] }> = convert_tile(s);
        let s_f: Tile<f32, { [1, B] }> = convert_tile(s_b);
        let o: Tile<bf16, { [1, B] }> = convert_tile(s_f * u);
        out.store(o);
    }

    /// `out[i, :] = src[idx[i], :]`, `B` columns per tile block.
    #[cutile::entry()]
    pub fn gather_rows<const B: i32>(
        out: &mut Tensor<bf16, { [1, B] }>,
        src: &Tensor<bf16, { [-1, -1] }>,
        idx: &Tensor<u32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ti: Tile<u32, { [1] }> = idx.partition(const_shape![1]).load([pid.0]);
        let ti: Tile<i32, { [1] }> = bitcast(ti);
        let row: i32 = tile_to_scalar(ti.reshape(const_shape![]));
        let part: Partition<bf16, { [1, B] }> = src.partition(const_shape![1, B]);
        let t: Tile<bf16, { [1, B] }> = part.load([row, pid.1]);
        out.store(t);
    }
}

/// `out = silu(gu[:, :I]) * gu[:, I:]` for `gu` of shape `[T, 2I]` and `out` of shape `[T, I]`.
/// `I` must be a multiple of [`SILU_BLOCK`].
pub fn silu_mul<'a>(out: &'a mut Tensor<bf16>, gu: &'a Tensor<bf16>) -> anyhow::Result<impl DeviceOp + GraphNode + 'a> {
    let (s, o) = (gu.shape(), out.shape());
    anyhow::ensure!(s.len() == 2 && s[1] % 2 == 0, "gu must be [T, 2I], got {s:?}");
    let i = s[1] as usize / 2;
    anyhow::ensure!(o == [s[0], i as i32].as_slice(), "out is {o:?}, expected [{}, {i}]", s[0]);
    anyhow::ensure!(i % SILU_BLOCK == 0, "I = {i} is not a multiple of {SILU_BLOCK}");
    Ok(pointwise_module::silu_mul(out.partition([1, SILU_BLOCK]), gu)
        .generics(vec![i.to_string(), SILU_BLOCK.to_string()]))
}

/// `out[i] = src[idx[i]]` for `src` `[R, N]`, `idx` `[Q]` (u32, every entry `< R`), `out` `[Q, N]`.
/// `N` must be a multiple of [`GATHER_BLOCK`]. Indices are not bounds-checked on the device.
pub fn gather_rows<'a>(
    out: &'a mut Tensor<bf16>,
    src: &'a Tensor<bf16>,
    idx: &'a Tensor<u32>,
) -> anyhow::Result<impl DeviceOp + GraphNode + 'a> {
    let (s, o) = (src.shape(), out.shape());
    anyhow::ensure!(s.len() == 2 && s[1] as usize % GATHER_BLOCK == 0, "src must be [R, N*{GATHER_BLOCK}], got {s:?}");
    anyhow::ensure!(idx.shape().len() == 1, "idx must be 1-D");
    anyhow::ensure!(o == [idx.shape()[0], s[1]].as_slice(), "out is {o:?}, expected [{}, {}]", idx.shape()[0], s[1]);
    Ok(pointwise_module::gather_rows(out.partition([1, GATHER_BLOCK]), src, idx).generics(vec![GATHER_BLOCK.to_string()]))
}

/// Embedding lookup: `out[t] = table[token_ids[t]]`. The same kernel as [`gather_rows`].
pub fn embedding<'a>(
    out: &'a mut Tensor<bf16>,
    table: &'a Tensor<bf16>,
    token_ids: &'a Tensor<u32>,
) -> anyhow::Result<impl DeviceOp + GraphNode + 'a> {
    gather_rows(out, table, token_ids)
}
