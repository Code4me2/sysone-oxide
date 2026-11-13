//! Phase 0: cuTile GEMM rates on this GPU at elephant-1's MLP shapes, bf16 vs NVFP4 block-scaled (DESIGN.md §13 Q8).
//!
//! Both kernels use the checkpoint's layouts: activations `[T, K]`, weights `[N, K]` row-major (NVFP4: packed
//! e2m1 pairs `[N, K/2]` plus e4m3 block scales `[N, K/16]`), fp32 accumulation, bf16 output. These are
//! deliberately plain tile kernels with a small tile sweep, a floor for W3, not a tuned implementation.

use anyhow::Result;
use clap::Parser;
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use std::time::Instant;

#[cutile::module]
mod probe_kernels {
    use cutile::core::*;

    /// z[T, N] = x[T, K] @ w[N, K]^T, bf16 in/out, fp32 accumulate.
    #[cutile::entry()]
    fn gemm_bf16<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<bf16, { [BM, BN] }>,
        x: &Tensor<bf16, { [-1, -1] }>,
        w: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let part_x = x.partition(shape![BM, BK]);
        let part_w = w.partition(shape![BN, BK]);
        let mut acc = constant(0.0f32, shape![BM, BN]);
        for k in 0i32..num_tiles(&part_x, 1) {
            let tx = part_x.load([pid.0, k]);
            let tw = part_w.load([pid.1, k]).transpose();
            acc = mma(tx, tw, acc);
        }
        let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc);
        z.store(out);
    }

    /// z[T, N] = alpha * x[T, K] @ w[N, K]^T with NVFP4 operands and e4m3 block scales (group 16), bf16 out.
    #[cutile::entry()]
    fn gemm_nvfp4<const BM: i32, const BN: i32, const BK: i32, const BK_PACKED: i32, const BK_SCALES: i32>(
        z: &mut Tensor<bf16, { [BM, BN] }>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let part_x = x.partition(shape![BM, BK_PACKED]);
        let part_w = w.partition(shape![BN, BK_PACKED]);
        let part_xs = x_scales.partition(shape![BM, BK_SCALES]);
        let part_ws = w_scales.partition(shape![BN, BK_SCALES]);
        let mut acc = constant(0.0f32, shape![BM, BN]);
        for k in 0i32..num_tiles(&part_x, 1) {
            let tx = part_x.load([pid.0, k]).unpack(shape![BM, BK]);
            let tw = part_w.load([pid.1, k]).unpack(shape![BN, BK]).transpose();
            let txs = part_xs.load([pid.0, k]);
            let tws = part_ws.load([pid.1, k]).transpose();
            acc = mmaf_scaled(tx, tw, acc, txs, tws);
        }
        let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc * broadcast_scalar(alpha, z.shape()));
        z.store(out);
    }
}

use probe_kernels::{gemm_bf16, gemm_nvfp4};

#[derive(Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_value = "4096,16384")]
    tokens: Vec<usize>,
    #[arg(long, default_value_t = 10)]
    iters: usize,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn host_bf16(n: usize, rng: &mut Rng) -> Vec<bf16> {
    (0..n).map(|_| bf16::from_f32(((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1)).collect()
}

fn host_fp4(n: usize, rng: &mut Rng) -> Vec<f4e2m1fnx2> {
    (0..n).map(|_| { let b = rng.next(); f4e2m1fnx2::from_nibbles((b & 0xf) as u8, ((b >> 4) & 0xf) as u8) }).collect()
}

/// e4m3 scales between 2^-1 and 2^1 (exponent field 6..8), never NaN.
fn host_scales(n: usize, rng: &mut Rng) -> Vec<f8e4m3fn> {
    (0..n).map(|_| { let b = rng.next(); f8e4m3fn((((6 + b % 3) << 3) | ((b >> 8) & 7)) as u8) }).collect()
}

fn report(kind: &str, t: usize, n: usize, k: usize, tiles: &str, secs: f64) {
    let tflops = 2.0 * (t * n * k) as f64 / secs / 1e12;
    println!("{kind:<6} {t:>6} {n:>6} {k:>6} {tiles:<14} {:>9.3} {tflops:>9.1}", secs * 1e3);
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let shapes = [(24576usize, 4096usize), (4096, 12288)]; // fused gate/up, down
    let bf16_tiles: &[(usize, usize, usize)] = &[(128, 128, 64), (128, 256, 64), (256, 128, 64), (64, 128, 64)];
    let fp4_tiles: &[(usize, usize, usize)] = &[(128, 128, 128), (128, 256, 128), (256, 128, 128), (128, 128, 256)];
    println!("{:<6} {:>6} {:>6} {:>6} {:<14} {:>9} {:>9}", "kind", "T", "N", "K", "BMxBNxBK", "ms", "TFLOP/s");

    for &t in &args.tokens {
        for &(n, k) in &shapes {
            let x: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(host_bf16(t * k, &mut rng)))
                .reshape(&[t, k]).sync_on(&stream)?.into();
            let w: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(host_bf16(n * k, &mut rng)))
                .reshape(&[n, k]).sync_on(&stream)?.into();
            for &(bm, bn, bk) in bf16_tiles {
                let gens = vec![bm.to_string(), bn.to_string(), bk.to_string()];
                let mut z = api::zeros::<bf16>(&[t, n]).sync_on(&stream)?.partition([bm, bn]);
                let mut best = f64::MAX;
                for i in 0..=args.iters {
                    let start = Instant::now();
                    let (z2, _, _) = gemm_bf16(z, x.clone(), w.clone()).generics(gens.clone()).sync_on(&stream)?;
                    z = z2;
                    if i > 0 { best = best.min(start.elapsed().as_secs_f64()); } // i = 0 includes JIT compile
                }
                report("bf16", t, n, k, &format!("{bm}x{bn}x{bk}"), best);
            }
            drop((x, w));

            let xq: Arc<Tensor<f4e2m1fnx2>> = api::copy_host_vec_to_device(&Arc::new(host_fp4(t * k / 2, &mut rng)))
                .reshape(&[t, k / 2]).sync_on(&stream)?.into();
            let wq: Arc<Tensor<f4e2m1fnx2>> = api::copy_host_vec_to_device(&Arc::new(host_fp4(n * k / 2, &mut rng)))
                .reshape(&[n, k / 2]).sync_on(&stream)?.into();
            let xs: Arc<Tensor<f8e4m3fn>> = api::copy_host_vec_to_device(&Arc::new(host_scales(t * k / 16, &mut rng)))
                .reshape(&[t, k / 16]).sync_on(&stream)?.into();
            let ws: Arc<Tensor<f8e4m3fn>> = api::copy_host_vec_to_device(&Arc::new(host_scales(n * k / 16, &mut rng)))
                .reshape(&[n, k / 16]).sync_on(&stream)?.into();
            for &(bm, bn, bk) in fp4_tiles {
                let gens = vec![bm.to_string(), bn.to_string(), bk.to_string(), (bk / 2).to_string(), (bk / 16).to_string()];
                let mut z = api::zeros::<bf16>(&[t, n]).sync_on(&stream)?.partition([bm, bn]);
                let mut best = f64::MAX;
                for i in 0..=args.iters {
                    let start = Instant::now();
                    let out = gemm_nvfp4(z, xq.clone(), wq.clone(), xs.clone(), ws.clone(), 1e-3f32)
                        .generics(gens.clone())
                        .sync_on(&stream);
                    match out {
                        Ok((z2, ..)) => z = z2,
                        Err(e) => { println!("nvfp4  {t:>6} {n:>6} {k:>6} {bm}x{bn}x{bk}: {e}"); best = f64::MAX; break; }
                    }
                    if i > 0 { best = best.min(start.elapsed().as_secs_f64()); }
                }
                if best < f64::MAX { report("nvfp4", t, n, k, &format!("{bm}x{bn}x{bk}"), best); }
            }
        }
    }
    Ok(())
}
