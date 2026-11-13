//! Host side of the cuda-oxide SIMT GDN kernels (`kernels-simt/src/main.rs`, docs/G1.md).
//!
//! Same operands and scratch layout as the cuTile path (`super`), so the two backends are interchangeable per
//! launch and can be compared stage by stage. The argument lists below must match the `#[kernel]` signatures.

use super::{CHUNK, DView, GdnPlan, GdnScratch, HEAD_DIM, NUM_V_HEADS, VALUE_DIM, CONV_DIM};
use crate::kernels::simt::{Arg, launch};
use anyhow::{Result, ensure};
use cuda_async::device_operation::{BoxedDeviceOp, DeviceOpVec};
use cutile::half::bf16;
use cutile::prelude::*;

fn ptr<T>(t: &Tensor<T>) -> Arg
where
    T: DType,
{
    Arg::Ptr(t.device_pointer().cu_deviceptr())
}

fn vptr<T>(v: &DView<T>) -> Arg {
    Arg::Ptr(v.ptr.cu_deviceptr())
}

/// `Tinv = (I + A)^-1` per (chunk, v-head) into `tinv` (`[T, 32 * 64]` f32, the layout of `GdnScratch::tinv`),
/// from the conv output (k columns), `scratch.g` and `scratch.beta`.
///
/// # Safety
/// Every buffer must stay alive and unaliased by other writers until the returned operation completes.
pub unsafe fn chunk_tinv(plan: &GdnPlan, conv: DView<bf16>, scratch: &GdnScratch, tinv: &Tensor<f32>) -> Result<BoxedDeviceOp<()>> {
    let t = plan.tokens;
    conv.check("conv", t, CONV_DIM)?;
    ensure!(t <= scratch.tokens, "plan has {t} tokens, scratch {}", scratch.tokens);
    ensure!(tinv.shape().iter().product::<i32>() as usize >= t * NUM_V_HEADS * CHUNK, "tinv too small");
    unsafe {
        launch(
            "gdn_simt_tinv",
            (plan.max_chunks as u32, NUM_V_HEADS as u32, 1),
            (128, 1, 1),
            0,
            &[
                vptr(&conv),
                Arg::I32(conv.row_stride as i32),
                ptr(&scratch.g),
                ptr(&scratch.beta),
                ptr(tinv),
                ptr(&plan.chunks),
                Arg::I32(plan.max_chunks as i32),
            ],
        )
    }
}


/// The SIMT chunked gated delta rule: `Tinv` per (chunk, v-head) into `scratch.tinv`, then the fused scan for
/// prefixes (writing their final states into `scratch.state`) and for suffixes (from their parent's state).
/// q, k and v are read from the conv output `conv` (`[T, 8192]` bf16: q | k | v) and normalised in-kernel; g and
/// beta from `scratch`. Writes bf16 `[T, 4096]` into `out`. `scratch.q`, `scratch.k` and the WY buffers are not
/// used.
///
/// # Safety
/// Every buffer must stay alive and unaliased by other writers until the returned operations complete.
pub unsafe fn chunk_scan(plan: &GdnPlan, conv: DView<bf16>, scratch: &GdnScratch, out: DView<bf16>) -> Result<DeviceOpVec<()>> {
    let t = plan.tokens;
    ensure!(plan.num_prefixes <= scratch.slots, "{} prefixes, scratch has {} state slots", plan.num_prefixes, scratch.slots);
    out.check("scan output", t, VALUE_DIM)?;
    ensure!(conv.row_stride % 2 == 0 && out.row_stride % 2 == 0, "SIMT scan needs even row strides");
    let mut ops = DeviceOpVec::with_capacity(3);
    ops.push(unsafe { chunk_tinv(plan, conv, scratch, &scratch.tinv)? });
    for phase in [0i32, 1] {
        ops.push(unsafe {
            launch(
                "gdn_simt_scan",
                (plan.max_segs as u32, NUM_V_HEADS as u32, (HEAD_DIM / 64) as u32),
                (128, 1, 1),
                0,
                &[
                    vptr(&conv),
                    Arg::I32(conv.row_stride as i32),
                    ptr(&scratch.g),
                    ptr(&scratch.beta),
                    ptr(&scratch.tinv),
                    ptr(&scratch.state),
                    vptr(&out),
                    Arg::I32(out.row_stride as i32),
                    ptr(&plan.segs),
                    Arg::I32(plan.max_segs as i32),
                    Arg::I32(scratch.slots as i32),
                    Arg::I32(phase),
                ],
            )?
        });
    }
    Ok(ops)
}
