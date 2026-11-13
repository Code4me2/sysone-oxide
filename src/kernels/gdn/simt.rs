//! Host side of the cuda-oxide SIMT GDN kernels (`kernels-simt/src/main.rs`, docs/G1.md).
//!
//! Same operands and scratch layout as the cuTile path (`super`), so the two backends are interchangeable per
//! launch and can be compared stage by stage. The argument lists below must match the `#[kernel]` signatures.

use super::{CHUNK, CONV_DIM, DView, GdnPlan, GdnScratch, GdnWeights, HEAD_DIM, NUM_V_HEADS, VALUE_DIM, check_tensor};
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

/// The SIMT kernels read and write bf16 views as pairs (u32 words): they need an even row stride and a 4-byte
/// aligned base.
fn check_pairs(name: &str, v: &DView<bf16>) -> Result<()> {
    ensure!(
        v.row_stride % 2 == 0 && v.ptr.cu_deviceptr() % 4 == 0,
        "SIMT GDN kernels need {name} with an even row stride and a 4-byte aligned base (stride {}, address {:#x})",
        v.row_stride,
        v.ptr.cu_deviceptr()
    );
    Ok(())
}

/// The gates g and beta (f32 `[T, 32]`) into `scratch` from the `a`, `b` columns of the fused projection
/// (`[T, 32]` bf16 views): the part of `gdn_prep` the SIMT scan needs.
///
/// # Safety
/// Every buffer must stay alive and unaliased by other writers until the returned operation completes.
pub unsafe fn gates(plan: &GdnPlan, a: DView<bf16>, b: DView<bf16>, w: &GdnWeights<'_>, scratch: &GdnScratch) -> Result<BoxedDeviceOp<()>> {
    let t = plan.tokens;
    ensure!(t <= scratch.tokens, "plan has {t} tokens, scratch {}", scratch.tokens);
    a.check("a", t, NUM_V_HEADS)?;
    b.check("b", t, NUM_V_HEADS)?;
    let a_log = check_tensor("A_log", w.a_log, &[NUM_V_HEADS])?;
    let dt_bias = check_tensor("dt_bias", w.dt_bias, &[NUM_V_HEADS])?;
    unsafe {
        launch(
            "gdn_simt_gates",
            ((t * NUM_V_HEADS).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                vptr(&a),
                Arg::I32(a.row_stride as i32),
                vptr(&b),
                Arg::I32(b.row_stride as i32),
                Arg::Ptr(a_log.cu_deviceptr()),
                Arg::Ptr(dt_bias.cu_deviceptr()),
                ptr(&scratch.g),
                ptr(&scratch.beta),
                Arg::I32(t as i32),
            ],
        )
    }
}

/// `Tinv = (I + A)^-1` per (chunk, v-head) into `tinv` (`[T, 32 * 64]` f32, the layout of `GdnScratch::tinv`),
/// from the conv output (k columns), `scratch.g` and `scratch.beta`.
///
/// # Safety
/// Every buffer must stay alive and unaliased by other writers until the returned operation completes.
pub unsafe fn chunk_tinv(plan: &GdnPlan, conv: DView<bf16>, scratch: &GdnScratch, tinv: &Tensor<f32>) -> Result<BoxedDeviceOp<()>> {
    let t = plan.tokens;
    conv.check("conv", t, CONV_DIM)?;
    check_pairs("conv", &conv)?;
    ensure!(t <= scratch.tokens, "plan has {t} tokens, scratch {}", scratch.tokens);
    ensure!(tinv.shape().iter().product::<i32>() as usize >= t * NUM_V_HEADS * CHUNK, "tinv too small");
    unsafe {
        launch(
            "gdn_simt_tinv",
            (NUM_V_HEADS as u32, plan.max_chunks as u32, 1),
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


/// The SIMT chunked gated delta rule, as four operations like the cuTile scan (`gdn_chunk_scan`): `Tinv` per
/// (chunk, v-head) into `scratch.tinv`, an empty operation in the slot of the cuTile WY launch, then the fused
/// scan for prefixes (writing their final states into `scratch.state`) and for suffixes (from their parent's
/// state). q, k and v are read from the conv output `conv` (`[T, 8192]` bf16: q | k | v) and normalised in-kernel;
/// g and beta from `scratch`. Writes bf16 `[T, 4096]` into `out`. `scratch.q`, `scratch.k` and the WY buffers are
/// not used.
///
/// # Safety
/// Every buffer must stay alive and unaliased by other writers until the returned operations complete.
pub unsafe fn chunk_scan(plan: &GdnPlan, conv: DView<bf16>, scratch: &GdnScratch, out: DView<bf16>) -> Result<DeviceOpVec<()>> {
    let t = plan.tokens;
    ensure!(t <= scratch.tokens, "plan has {t} tokens, scratch {}", scratch.tokens);
    ensure!(plan.num_prefixes <= scratch.slots, "{} prefixes, scratch has {} state slots", plan.num_prefixes, scratch.slots);
    conv.check("conv", t, CONV_DIM)?;
    out.check("scan output", t, VALUE_DIM)?;
    check_pairs("conv", &conv)?;
    check_pairs("scan output", &out)?;
    let mut ops = DeviceOpVec::with_capacity(4);
    ops.push(unsafe { chunk_tinv(plan, conv, scratch, &scratch.tinv)? });
    ops.push(value(()).boxed());
    for phase in [0i32, 1] {
        ops.push(unsafe {
            launch(
                "gdn_simt_scan",
                // x = 2 * head + value block: the four CTAs reading the same q/k rows are adjacent.
                ((NUM_V_HEADS * HEAD_DIM / 64) as u32, plan.max_segs as u32, 1),
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
