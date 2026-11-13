//! Slot readout (DESIGN.md §6.3): `logits[Q, S] = hs[Q, K] @ W_slot[S, K]^T` with bf16 operands, fp32
//! accumulation and fp32 output. Owner: W3b.
//!
//! qev upcasts the LM head to fp32 and casts the (final-normed, bf16) hidden state to fp32 before it
//! (`_upcast_head`). Products of two bf16 values are exact in fp32, so a bf16 MMA with an fp32 accumulator
//! computes the same sums, up to summation order. `W_slot` is `lm_head.weight[slot_ids]` (256 × 4096).

use cutile::half::bf16;
use cutile::prelude::*;

/// Rows of `hs` per tile block. Q is small (one row per branch), so the tile is short.
pub const READOUT_BM: usize = 16;
/// Slot columns per tile block.
pub const READOUT_BN: usize = 64;
/// K step.
pub const READOUT_BK: usize = 128;

#[cutile::module]
pub mod readout_module {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn slot_logits<const BM: i32, const BN: i32, const BK: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        hs: &Tensor<bf16, { [-1, -1] }>,
        w: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_h: Partition<bf16, { [BM, BK] }> = hs.partition(const_shape![BM, BK]);
        let part_w: Partition<bf16, { [BN, BK] }> = w.partition(const_shape![BN, BK]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for k in 0i32..num_tiles(&part_h, 1) {
            let th: Tile<bf16, { [BM, BK] }> = part_h.load([pid.0, k]);
            let tw: Tile<bf16, { [BN, BK] }> = part_w.load([pid.1, k]);
            let twt: Tile<bf16, { [BK, BN] }> = tw.transpose();
            // Each K step starts from a zero accumulator and is added in fp32: tensor-core accumulation
            // over the whole K = 4096 loses ~30x more precision than qev's cuBLAS fp32 head.
            let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            acc = acc + mma(th, twt, zero);
        }
        out.store(acc);
    }
}

/// `out[Q, S] = hs[Q, K] @ w[S, K]^T` (fp32 out). `K` must be a multiple of [`READOUT_BK`] and `S` of
/// [`READOUT_BN`]; `Q` is arbitrary.
pub fn slot_logits<'a>(
    out: &'a mut Tensor<f32>,
    hs: &'a Tensor<bf16>,
    w: &'a Tensor<bf16>,
) -> anyhow::Result<impl DeviceOp + GraphNode + 'a> {
    let (h, ws, o) = (hs.shape(), w.shape(), out.shape());
    anyhow::ensure!(h.len() == 2 && ws.len() == 2 && h[1] == ws[1], "hs {h:?} and w {ws:?} disagree on K");
    anyhow::ensure!(h[1] as usize % READOUT_BK == 0, "K = {} is not a multiple of {READOUT_BK}", h[1]);
    anyhow::ensure!(ws[0] as usize % READOUT_BN == 0, "S = {} is not a multiple of {READOUT_BN}", ws[0]);
    anyhow::ensure!(o == [h[0], ws[0]].as_slice(), "out is {o:?}, expected [{}, {}]", h[0], ws[0]);
    Ok(readout_module::slot_logits(out.partition([READOUT_BM, READOUT_BN]), hs, w).generics(vec![
        READOUT_BM.to_string(),
        READOUT_BN.to_string(),
        READOUT_BK.to_string(),
    ]))
}
