//! C1: the cuBLASLt NVFP4 GEMM path (docs/C1.md).
//!
//! - `scales_to_lt_matches_host_layout`: the engine's activation-scale re-layout kernel writes exactly
//!   `to_lt_scales` (cuBLASLt's tiled block-scale layout, `lt_scale_offset`), including partial row blocks.
//! - `cublaslt_fp4_gemm_matches_cutile_and_is_batch_invariant`: `Blas::gemm_nvfp4_lt` (pinned algorithm) against the
//!   engine's cuTile `gemm_nvfp4_raw` on the same operands (bf16 ulps; the f64 comparison of both is in
//!   `bench_fp4_cublaslt --check`), and a row's result independent of the call's row count and of its position.

use cuda_core::{Context, Stream};
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use std::sync::Arc;
use sysone::engine::gemm::{Blas, GemmMode};
use sysone::engine::graph::tensor::{DevTensor, GraphDType, bf16_from_bytes};
use sysone::kernels::nvfp4::{
    self, CheckpointProj, GemmTiles, Nvfp4Act, Nvfp4Linear, QuantTiles, lt_scale_bytes, to_lt_scales,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn dev(stream: &Arc<Stream>, dtype: GraphDType, shape: &[usize], bytes: &[u8]) -> DevTensor {
    let t = DevTensor::zeros(stream, dtype, shape).unwrap();
    t.copy_from_host(stream, bytes).unwrap();
    t
}

#[test]
fn scales_to_lt_matches_host_layout() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut rng = Rng(0x51ed_270b_9a3c_77e1);
    for (rows, cols) in [(1usize, 256usize), (300, 256), (1024, 768), (2049, 768), (128, 4)] {
        let host: Vec<u8> = (0..rows * cols).map(|_| (rng.next() % 0x7f) as u8).collect();
        let s = dev(&stream, GraphDType::F8E4M3, &[rows, cols], &host);
        // Fill the output with a non-zero pattern: the kernel must write every byte (padding rows as zeros).
        let out = dev(&stream, GraphDType::F8E4M3, &[lt_scale_bytes(rows, cols)], &vec![0xa5u8; lt_scale_bytes(rows, cols)]);
        unsafe {
            nvfp4::scales_to_lt_op(Arc::new(s.tile::<f8e4m3fn>().unwrap()), out.tile::<f8e4m3fn>().unwrap())
                .unwrap()
                .sync_on(&stream)
                .unwrap();
        }
        let got = out.to_host_bytes(&stream).unwrap();
        let want = to_lt_scales(&host, rows, cols);
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("scales_to_lt [{rows}, {cols}]: {bad} of {} bytes differ from the host layout", want.len());
        assert_eq!(bad, 0);
    }
}

fn bf16_ulps(a: f32, b: f32) -> i64 {
    let key = |v: f32| {
        let u = bf16::from_f32(v).to_bits() as i64;
        if u & 0x8000 != 0 { -(u & 0x7fff) } else { u }
    };
    (key(a) - key(b)).abs()
}

#[test]
fn cublaslt_fp4_gemm_matches_cutile_and_is_batch_invariant() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let sms = ctx.multiprocessor_count().unwrap() as usize;
    let (major, minor) = ctx.compute_capability().unwrap();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let t = 1024usize;
    // (N, K, igs, wgs): the down projection and one gate/up half.
    for (n, k, igs, wgs) in [(4096usize, 12288usize, 233.0f32, 7000.0f32), (12288, 4096, 628.0, 9408.0)] {
        let wp: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
        let ws: Vec<u8> = (0..n * k / 16).map(|_| ((4 + rng.next() % 7) << 3 | (rng.next() & 7)) as u8).collect();
        let proj = CheckpointProj { n, k, weight_packed: &wp, weight_scale: &ws, weight_global_scale: wgs,
                                    input_global_scale: igs };
        let prep = nvfp4::prepare_weights(&[proj]).unwrap();
        let lin = Nvfp4Linear::upload(&prep, &stream).unwrap();
        let xh: Vec<bf16> =
            (0..t * k).map(|_| bf16::from_f32(((rng.next() >> 40) as f32 / 16777216.0 - 0.5) * 4.0)).collect();
        let x: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(xh)).reshape(&[t, k]).sync_on(&stream).unwrap().into();
        let a = nvfp4::quantize(&x, igs, Nvfp4Act::zeros(t, k, &stream).unwrap(), QuantTiles::DEFAULT, &stream).unwrap();
        let (xp, xs): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) = (a.packed.into(), a.scales.into());
        let qb: Vec<u8> = xp.clone().to_host_vec().sync_on(&stream).unwrap().iter().map(|v| v.to_bits()).collect();
        let sb: Vec<u8> = xs.clone().to_host_vec().sync_on(&stream).unwrap().iter().map(|v| v.0).collect();
        let z: Arc<Tensor<bf16>> = api::zeros::<bf16>(&[t, n]).sync_on(&stream).unwrap().into();
        let z = nvfp4::gemm_op(&xp, &xs, &lin, z, GemmTiles::DEFAULT, sms).unwrap().sync_on(&stream).unwrap();
        let y_tile: Vec<f32> = z.clone().to_host_vec().sync_on(&stream).unwrap().iter().map(|v| v.to_f32()).collect();

        let mut blas = Blas::with_mode(&stream, GemmMode::Pinned).unwrap();
        blas.enable_fp4_lt(t, k / 16, (major as u32, minor as u32)).unwrap();
        let w = dev(&stream, GraphDType::U8, &[n, k / 2], &wp);
        let w_lt_bytes = to_lt_scales(&ws, n, k / 16);
        let w_lt = dev(&stream, GraphDType::F8E4M3, &[w_lt_bytes.len()], &w_lt_bytes);
        // Rows [start, start + rows) of the activation through cuBLASLt as a call of `rows` rows.
        let run = |start: usize, rows: usize| -> Vec<f32> {
            let q = dev(&stream, GraphDType::U8, &[rows, k / 2], &qb[start * k / 2..(start + rows) * k / 2]);
            let s = dev(&stream, GraphDType::F8E4M3, &[rows, k / 16], &sb[start * k / 16..(start + rows) * k / 16]);
            let scratch = blas.fp4_scratch(rows, k / 16).unwrap();
            unsafe {
                nvfp4::scales_to_lt_op(Arc::new(s.tile::<f8e4m3fn>().unwrap()), scratch.tile::<f8e4m3fn>().unwrap())
                    .unwrap()
                    .sync_on(&stream)
                    .unwrap();
            }
            let y = DevTensor::zeros(&stream, GraphDType::BF16, &[rows, n]).unwrap();
            blas.gemm_nvfp4_lt(&stream, &q, &scratch, &w, &w_lt, prep.alpha_lo, &y).unwrap();
            stream.synchronize().unwrap();
            bf16_from_bytes(&y.to_host_bytes(&stream).unwrap())
        };
        let y_lt = run(0, t);
        let (mut differ, mut max_ulp) = (0usize, 0i64);
        for (a, b) in y_lt.iter().zip(&y_tile) {
            let u = bf16_ulps(*a, *b);
            differ += (u != 0) as usize;
            max_ulp = max_ulp.max(u);
        }
        println!("N={n} K={k}: cuBLASLt {:?} vs cuTile raw: {differ} of {} outputs differ, max {max_ulp} bf16 ulp",
                 blas.fp4_algos(), y_lt.len());
        assert!(max_ulp <= 1, "cuBLASLt FP4 GEMM differs from the cuTile GEMM by {max_ulp} ulp");
        // Batch invariance: the same rows in calls of other sizes and at other positions are bit-identical.
        for (start, rows) in [(0usize, 1usize), (0, 37), (5, 300), (128, 129), (700, 324)] {
            let y = run(start, rows);
            let bad = y.iter().zip(&y_lt[start * n..(start + rows) * n]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!("  rows {start}..{} as a {rows}-row call: {bad} of {} values differ from the {t}-row call",
                     start + rows, rows * n);
            assert_eq!(bad, 0, "the pinned cuBLASLt FP4 GEMM is not batch-invariant");
        }
    }
}
