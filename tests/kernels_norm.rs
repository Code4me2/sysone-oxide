//! GPU tests for the norm, pointwise and readout kernels (W3b) against CPU f64 references written from
//! transformers 5.19.0 `modeling_qwen3_5.py` (`Qwen3_5RMSNorm`, `Qwen3_5DecoderLayer`, `Qwen3_5MLP`, the
//! embedding and the `lm_head` rows).
//!
//! The references compute in f64 and round to bf16 at the points where the reference materialises a bf16
//! tensor. The kernels compute in fp32, so a result may land on the neighbouring bf16 value when the exact
//! value sits close to a rounding boundary. Error is measured in bf16 ulps of the reference value.

use cutile::half::bf16;
use cutile::prelude::*;
use sysone::kernels::{norm, pointwise, readout};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn bf16s(&mut self, n: usize, scale: f64) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f64(self.uniform() * scale)).collect()
    }
}

fn f(x: bf16) -> f64 {
    x.to_f64()
}

fn round_bf16(x: f64) -> f64 {
    bf16::from_f64(x).to_f64()
}

/// Distance between a bf16 result and an exact reference value, in bf16 ulps at the reference's magnitude.
fn ulps(got: f64, want: f64) -> f64 {
    let mag = want.abs().max(1e-30);
    let ulp = 2f64.powi(mag.log2().floor() as i32 - 7);
    (got - want).abs() / ulp
}

struct Stats {
    max: f64,
    p99: f64,
    mismatched: f64,
    n: usize,
}

/// `want_exact` is the reference value before its final bf16 rounding. `mismatched` is the fraction of
/// elements that differ from the correctly rounded reference.
fn bf16_stats(got: &[bf16], want_exact: &[f64]) -> Stats {
    assert_eq!(got.len(), want_exact.len());
    let mut errs: Vec<f64> = got.iter().zip(want_exact).map(|(&g, &w)| ulps(f(g), w)).collect();
    let mism = got.iter().zip(want_exact).filter(|(g, w)| f(**g) != round_bf16(**w)).count();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = errs.len();
    Stats { max: errs[n - 1], p99: errs[(n * 99 / 100).min(n - 1)], mismatched: mism as f64 / n as f64, n }
}

fn report(name: &str, s: &Stats, max_ulps: f64, max_mismatch: f64) {
    println!(
        "{name}: n={} max={:.3} ulp p99={:.3} ulp mismatched={:.4}% (tolerance: max <= {max_ulps} ulp, mismatched <= {}%)",
        s.n,
        s.max,
        s.p99,
        s.mismatched * 100.0,
        max_mismatch * 100.0
    );
    assert!(s.max <= max_ulps, "{name}: max error {} ulp over {max_ulps}", s.max);
    assert!(s.mismatched <= max_mismatch, "{name}: {}% mismatched", s.mismatched * 100.0);
}

fn upload<T: DType>(v: Vec<T>, shape: &[usize], stream: &Arc<cuda_core::Stream>) -> Tensor<T> {
    api::copy_host_vec_to_device(&Arc::new(v)).reshape(shape).sync_on(stream).unwrap()
}

fn setup() -> Arc<cuda_core::Stream> {
    let ctx = Context::new(0).unwrap();
    ctx.new_stream().unwrap()
}

/// `Qwen3_5RMSNorm.forward` before the final `.type_as(x)`.
fn ref_zc_rmsnorm(x: &[f64], w: &[bf16], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let ms = x.iter().map(|v| v * v).sum::<f64>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * inv * (1.0 + f(*w))).collect()
}

const HIDDEN: usize = 4096;

#[test]
fn zc_rmsnorm_matches_reference() {
    let stream = setup();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for &t in &[1usize, 3, 257] {
        // Activations with a heavy row-to-row scale spread (residual streams grow with depth).
        let mut x = Vec::with_capacity(t * HIDDEN);
        for r in 0..t {
            let scale = [0.02, 1.0, 40.0][r % 3];
            x.extend(rng.bf16s(HIDDEN, scale));
        }
        let w = rng.bf16s(HIDDEN, 0.5);
        let xd = upload(x.clone(), &[t, HIDDEN], &stream);
        let wd = upload(w.clone(), &[HIDDEN], &stream);
        let mut out = api::zeros::<bf16>(&[t, HIDDEN]).sync_on(&stream).unwrap();
        norm::zc_rmsnorm(&mut out, &xd, &wd, norm::RMS_EPS).unwrap().sync_on(&stream).unwrap();
        let got = out.to_host_vec().sync_on(&stream).unwrap();
        let mut want = Vec::with_capacity(t * HIDDEN);
        for r in 0..t {
            let row: Vec<f64> = x[r * HIDDEN..(r + 1) * HIDDEN].iter().map(|v| f(*v)).collect();
            want.extend(ref_zc_rmsnorm(&row, &w, 1e-6));
        }
        report(&format!("zc_rmsnorm T={t}"), &bf16_stats(&got, &want), 1.0, 0.01);
    }
}

#[test]
fn add_zc_rmsnorm_matches_reference() {
    let stream = setup();
    let mut rng = Rng(0x2468_ace0_1357_9bdf);
    let t = 129;
    let res = rng.bf16s(t * HIDDEN, 8.0);
    let x = rng.bf16s(t * HIDDEN, 2.0);
    let w = rng.bf16s(HIDDEN, 0.5);
    let mut resd = upload(res.clone(), &[t, HIDDEN], &stream);
    let xd = upload(x.clone(), &[t, HIDDEN], &stream);
    let wd = upload(w.clone(), &[HIDDEN], &stream);
    let mut out = api::zeros::<bf16>(&[t, HIDDEN]).sync_on(&stream).unwrap();
    norm::add_zc_rmsnorm(&mut resd, &mut out, &xd, &wd, norm::RMS_EPS).unwrap().sync_on(&stream).unwrap();
    let got_res = resd.to_host_vec().sync_on(&stream).unwrap();
    let got = out.to_host_vec().sync_on(&stream).unwrap();
    let nan = got.iter().position(|v| v.is_nan());
    assert!(nan.is_none(), "NaN in output at {nan:?} (row {:?})", nan.map(|i| i / HIDDEN));

    // `hidden_states = residual + hidden_states` in bf16, then the next norm reads the rounded sum.
    let new_res: Vec<f64> = res.iter().zip(&x).map(|(a, b)| f(*a) + f(*b)).collect();
    let res_bits_equal = got_res.iter().zip(&new_res).filter(|(g, w)| f(**g) == round_bf16(**w)).count();
    println!("add_zc_rmsnorm residual: {res_bits_equal}/{} bit-exact (tolerance: all)", new_res.len());
    assert_eq!(res_bits_equal, new_res.len(), "residual add must be the correctly rounded bf16 sum");
    let mut want = Vec::with_capacity(t * HIDDEN);
    for r in 0..t {
        let row: Vec<f64> = new_res[r * HIDDEN..(r + 1) * HIDDEN].iter().map(|v| round_bf16(*v)).collect();
        want.extend(ref_zc_rmsnorm(&row, &w, 1e-6));
    }
    report("add_zc_rmsnorm out", &bf16_stats(&got, &want), 1.0, 0.01);
}

/// `Qwen3_5MLP.forward`'s `act_fn(gate_proj(x)) * up_proj(x)` on bf16 tensors: torch rounds `silu(g)` to
/// bf16, then rounds the product.
#[test]
fn silu_mul_matches_reference() {
    let stream = setup();
    let mut rng = Rng(0x0bad_cafe_dead_beef);
    let (t, i) = (37usize, 12288usize);
    let gu = rng.bf16s(t * 2 * i, 6.0);
    let gud = upload(gu.clone(), &[t, 2 * i], &stream);
    let mut out = api::zeros::<bf16>(&[t, i]).sync_on(&stream).unwrap();
    pointwise::silu_mul(&mut out, &gud).unwrap().sync_on(&stream).unwrap();
    let got = out.to_host_vec().sync_on(&stream).unwrap();
    let mut want = Vec::with_capacity(t * i);
    for r in 0..t {
        for j in 0..i {
            let g = f(gu[r * 2 * i + j]);
            let u = f(gu[r * 2 * i + i + j]);
            let s = round_bf16(g / (1.0 + (-g).exp()));
            want.push(s * u);
        }
    }
    report("silu_mul", &bf16_stats(&got, &want), 1.0, 0.01);
}

#[test]
fn gather_rows_and_embedding_are_exact() {
    let stream = setup();
    let mut rng = Rng(0x5eed_5eed_5eed_5eed);
    let (rows, n) = (1000usize, HIDDEN);
    let src = rng.bf16s(rows * n, 1.0);
    let srcd = upload(src.clone(), &[rows, n], &stream);
    let idx: Vec<u32> = (0..53).map(|_| (rng.next() % rows as u64) as u32).chain([0, rows as u32 - 1]).collect();
    let idxd = upload(idx.clone(), &[idx.len()], &stream);
    let mut out = api::zeros::<bf16>(&[idx.len(), n]).sync_on(&stream).unwrap();
    pointwise::gather_rows(&mut out, &srcd, &idxd).unwrap().sync_on(&stream).unwrap();
    let got = out.to_host_vec().sync_on(&stream).unwrap();
    for (i, &r) in idx.iter().enumerate() {
        assert_eq!(&got[i * n..(i + 1) * n], &src[r as usize * n..(r as usize + 1) * n], "row {i} (src row {r})");
    }
    let mut out = api::zeros::<bf16>(&[idx.len(), n]).sync_on(&stream).unwrap();
    pointwise::embedding(&mut out, &srcd, &idxd).unwrap().sync_on(&stream).unwrap();
    assert_eq!(out.to_host_vec().sync_on(&stream).unwrap(), got);
    println!("gather_rows/embedding: {} rows bit-exact (tolerance: exact)", idx.len());
}

/// fp32 logits of `hs @ W_slot^T`, as qev's upcast fp32 LM head computes them.
#[test]
fn slot_logits_match_reference() {
    let stream = setup();
    let mut rng = Rng(0xfeed_face_1234_4321);
    for &q in &[1usize, 7, 16, 100] {
        let (s, k) = (256usize, HIDDEN);
        let hs = rng.bf16s(q * k, 3.0);
        let w = rng.bf16s(s * k, 0.05);
        let hsd = upload(hs.clone(), &[q, k], &stream);
        let wd = upload(w.clone(), &[s, k], &stream);
        let mut out = api::zeros::<f32>(&[q, s]).sync_on(&stream).unwrap();
        readout::slot_logits(&mut out, &hsd, &wd).unwrap().sync_on(&stream).unwrap();
        let got = out.to_host_vec().sync_on(&stream).unwrap();
        // Error relative to the dot product's condition (sum of |h*w|), in fp32 epsilons.
        let mut errs = Vec::with_capacity(q * s);
        let mut max_abs = 0f64;
        for r in 0..q {
            for c in 0..s {
                let (mut dot, mut mag) = (0f64, 0f64);
                for j in 0..k {
                    let p = f(hs[r * k + j]) * f(w[c * k + j]);
                    dot += p;
                    mag += p.abs();
                }
                let e = (got[r * s + c] as f64 - dot).abs();
                max_abs = max_abs.max(e);
                errs.push(e / (mag * f32::EPSILON as f64));
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (max, p99) = (errs[errs.len() - 1], errs[errs.len() * 99 / 100]);
        println!(
            "slot_logits Q={q}: max abs err={max_abs:.3e}, normalised max={max:.3} p99={p99:.3} \
             (units: eps32 * sum|h*w|; tolerance: normalised max <= 1; torch fp32 head on GB10 measures 0.13 on cuda, 0.81 on cpu)"
        );
        assert!(max <= 1.0, "slot_logits Q={q}: normalised error {max}");
    }
}
