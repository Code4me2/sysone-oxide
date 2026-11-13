//! cuBLAS bf16 GEMM (fp32 compute) against a CPU f64 reference on random data (DESIGN §6.4, §6.6).
//!
//! Reference: `y[t, n] = Σ_k x[t, k] · w[n, k]` in f64 over the exact bf16 inputs. Tolerance per element:
//! the output rounding (half a bf16 ulp, ≤ 2^-9 |y|; none for f32 output) plus an fp32-accumulation bound
//! of `K · 2^-23 · Σ_k |x·w|` (generous: worst-case linear growth, any summation order).

use cuda_core::Context;
use sysone::engine::gemm::Blas;
use sysone::engine::graph::tensor::{DevTensor, GraphDType, bf16_bytes, bf16_from_bytes, f32_from_bytes};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
    fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| half::bf16::from_f32(self.f() * scale).to_f32()).collect()
    }
}

struct Stats {
    max_ratio: f64,
    max_abs: f64,
    p99_abs: f64,
    max_rel: f64,
}

/// Compare `y` against the f64 reference; returns error statistics. `ratio` is |err| / tolerance.
#[allow(clippy::too_many_arguments)]
fn check(y: &[f32], x: &[f32], x_stride: usize, w: &[f32], t: usize, n: usize, k: usize, bf16_out: bool) -> Stats {
    let mut abs = Vec::with_capacity(t * n);
    let (mut max_ratio, mut max_rel) = (0.0f64, 0.0f64);
    for i in 0..t {
        for j in 0..n {
            let (mut s, mut mag) = (0.0f64, 0.0f64);
            for kk in 0..k {
                let p = x[i * x_stride + kk] as f64 * w[j * k + kk] as f64;
                s += p;
                mag += p.abs();
            }
            let e = (y[i * n + j] as f64 - s).abs();
            let acc = k as f64 * 2f64.powi(-23) * mag;
            let tol = if bf16_out { (s.abs() + acc) * 2f64.powi(-8) } else { 0.0 } + acc + 1e-30;
            max_ratio = max_ratio.max(e / tol);
            if s.abs() > 1e-3 * mag.max(1e-30) {
                max_rel = max_rel.max(e / s.abs());
            }
            abs.push(e);
        }
    }
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Stats { max_ratio, max_abs: *abs.last().unwrap(), p99_abs: abs[abs.len() * 99 / 100], max_rel }
}

#[test]
fn cublas_bf16_gemm_matches_f64_reference() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let blas = Blas::new(&stream).unwrap();
    let mut rng = Rng(0x9e3779b97f4a7c15);
    // (T, N, K, x column offset inside a wider row, output dtype)
    let cases = [
        (1usize, 64usize, 128usize, 0usize, GraphDType::BF16),
        (37, 96, 256, 0, GraphDType::BF16),
        (128, 2048, 4096, 0, GraphDType::BF16),
        (200, 64, 4096, 0, GraphDType::BF16),   // in_proj_a/b shape
        (65, 256, 4096, 0, GraphDType::F32),    // slot readout: f32 out
        (48, 320, 512, 128, GraphDType::BF16),  // x is a column slice of a wider buffer
        (300, 128, 12288, 0, GraphDType::BF16), // down_proj K
    ];
    for (t, n, k, col0, out_dt) in cases {
        let width = k + col0 + if col0 > 0 { 64 } else { 0 };
        let xf = rng.bf16_vec(t * width, 1.0);
        let wf = rng.bf16_vec(n * k, 0.05);
        let xb = DevTensor::alloc(&ctx, GraphDType::BF16, &[t, width]).unwrap();
        xb.copy_from_host(&stream, &bf16_bytes(&xf)).unwrap();
        let x = xb.cols_view(col0, k).unwrap();
        let w = DevTensor::alloc(&ctx, GraphDType::BF16, &[n, k]).unwrap();
        w.copy_from_host(&stream, &bf16_bytes(&wf)).unwrap();
        let y = DevTensor::zeros(&stream, out_dt, &[t, n]).unwrap();
        blas.gemm_bf16(&stream, &x, &w, &y).unwrap();
        stream.synchronize().unwrap();
        let yb = y.to_host_bytes(&stream).unwrap();
        let yv = if out_dt == GraphDType::BF16 { bf16_from_bytes(&yb) } else { f32_from_bytes(&yb) };
        let xs: Vec<f32> = xf.iter().skip(col0).copied().collect();
        let s = check(&yv, &xs, width, &wf, t, n, k, out_dt == GraphDType::BF16);
        println!(
            "gemm T={t:4} N={n:5} K={k:5} x_col0={col0:3} out={out_dt:?}: max|err| {:.3e} p99|err| {:.3e} \
             max rel {:.3e}, max err/tol {:.3}",
            s.max_abs, s.p99_abs, s.max_rel, s.max_ratio
        );
        assert!(s.max_ratio <= 1.0, "T={t} N={n} K={k}: error {:.3} × tolerance", s.max_ratio);
    }
}
