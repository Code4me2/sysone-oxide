//! cuBLAS bf16 GEMM (fp32 compute) against a CPU f64 reference on random data (DESIGN §6.4, §6.6).
//!
//! Reference: `y[t, n] = Σ_k x[t, k] · w[n, k]` in f64 over the exact bf16 inputs. Tolerance per element:
//! the output rounding (half a bf16 ulp, ≤ 2^-9 |y|; none for f32 output) plus an fp32-accumulation bound
//! of `K · 2^-23 · Σ_k |x·w|` (generous: worst-case linear growth, any summation order).

use cuda_core::Context;
use sysone::engine::gemm::{Blas, GemmMode};
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
    for mode in [GemmMode::Heuristic, GemmMode::Pinned] {
        println!("mode {}", mode.name());
        accuracy(mode);
    }
}

fn accuracy(mode: GemmMode) {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let blas = Blas::with_mode(&stream, mode).unwrap();
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

/// L1: in pinned mode a row's output does not depend on how many rows the GEMM has or where the row sits. For every
/// projection shape of elephant-1 (and a slot readout), a [T, K] GEMM is compared bit for bit with GEMMs over
/// sub-batches of its rows (placed at other positions, M from 1 to T - 256). The heuristic mode runs the same check
/// and its mismatches are reported, not asserted (cuBLAS may legitimately pick another kernel per M).
#[test]
fn pinned_gemm_rows_do_not_depend_on_the_batch() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut rng = Rng(0x2545f4914f6cdd1d);
    // (N, K, f32 out): GDN in / out, attention q / kv / o, MLP gate-up / down (bf16 engine), slot readout.
    let shapes = [(12352usize, 4096usize, false), (4096, 4096, false), (8192, 4096, false), (2048, 4096, false),
                  (24576, 4096, false), (4096, 12288, false), (26, 4096, true)];
    let t = 2304usize;
    let subs = [1usize, 2, 7, 64, 129, 300, 1000, 2048];
    for mode in [GemmMode::Pinned, GemmMode::Heuristic] {
        let blas = Blas::with_mode(&stream, mode).unwrap();
        let mut bad_total = 0usize;
        for &(n, k, f32_out) in &shapes {
            let out_dt = if f32_out { GraphDType::F32 } else { GraphDType::BF16 };
            let es = if f32_out { 4 } else { 2 };
            let xf = rng.bf16_vec(t * k, 1.0);
            let wf = rng.bf16_vec(n * k, 0.05);
            let w = DevTensor::alloc(&ctx, GraphDType::BF16, &[n, k]).unwrap();
            w.copy_from_host(&stream, &bf16_bytes(&wf)).unwrap();
            let x = DevTensor::alloc(&ctx, GraphDType::BF16, &[t, k]).unwrap();
            x.copy_from_host(&stream, &bf16_bytes(&xf)).unwrap();
            let y = DevTensor::zeros(&stream, out_dt, &[t, n]).unwrap();
            blas.gemm_bf16(&stream, &x, &w, &y).unwrap();
            let full = y.to_host_bytes(&stream).unwrap();
            let mut bad_rows = 0usize;
            for (si, &m) in subs.iter().enumerate() {
                // The sub-batch: a strided selection of the full batch's rows, each at a new position.
                let rows: Vec<usize> = (0..m).map(|i| (i * 7 + si * 13 + 3) % t).collect();
                let mut sx = Vec::with_capacity(m * k);
                for &r in &rows {
                    sx.extend_from_slice(&xf[r * k..(r + 1) * k]);
                }
                let xs = DevTensor::alloc(&ctx, GraphDType::BF16, &[m, k]).unwrap();
                xs.copy_from_host(&stream, &bf16_bytes(&sx)).unwrap();
                let ys = DevTensor::zeros(&stream, out_dt, &[m, n]).unwrap();
                blas.gemm_bf16(&stream, &xs, &w, &ys).unwrap();
                let got = ys.to_host_bytes(&stream).unwrap();
                for (i, &r) in rows.iter().enumerate() {
                    if got[i * n * es..(i + 1) * n * es] != full[r * n * es..(r + 1) * n * es] {
                        bad_rows += 1;
                    }
                }
            }
            let checked: usize = subs.iter().sum();
            println!("{}: N={n:5} K={k:5} {}: {bad_rows}/{checked} sub-batch rows differ from the T={t} GEMM",
                     mode.name(), if f32_out { "f32" } else { "bf16" });
            bad_total += bad_rows;
        }
        for ((n, k, ldx, ldy, f32_out), (algo, tile, stages)) in blas.pinned_algos() {
            println!("  pinned N={n} K={k} ldx={ldx} ldy={ldy} f32={f32_out}: algo {algo} tile {tile} stages {stages}");
        }
        if mode == GemmMode::Pinned {
            assert_eq!(bad_total, 0, "pinned GEMM rows depend on the batch");
        }
    }
}
