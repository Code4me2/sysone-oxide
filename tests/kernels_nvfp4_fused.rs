//! GPU tests for the NVFP4 quantization fused into the norm and SiLU·mul epilogues (N1; DESIGN §6.6).
//!
//! Each fused kernel must be **bit-identical** to its two-kernel equivalent: the bf16 op (`norm::zc_rmsnorm`,
//! `norm::add_zc_rmsnorm`, `pointwise::silu_mul`, all checked against the reference in `tests/kernels_norm.rs`)
//! followed by the standalone `nvfp4::quantize_nvfp4` (bit-exact vs compressed-tensors in `tests/kernels_nvfp4.rs`).
//! Packed bytes, scale bytes and (for the residual variant) the updated residual are compared byte for byte, on
//! the engine's shapes (K = 4096 after the norms, 12288 after SiLU·mul), rows with a wide spread of magnitudes,
//! all-zero rows, and several activation global scales (including one that saturates the e4m3 scale at 448).
//!
//! Run: `scripts/gpu.sh cargo test --release --test kernels_nvfp4_fused -- --nocapture --test-threads 1`

use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use sysone::kernels::nvfp4::{self, Nvfp4Act, QuantTiles};
use sysone::kernels::{norm, pointwise};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit().max(1e-300), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    /// `t` rows of `k` values: row scales spread over 2^-8 .. 2^8, group scales over 2^-4 .. 2^4, every 7th row
    /// zero (an all-zero group takes the 0.125 scale path).
    fn rows(&mut self, t: usize, k: usize) -> Vec<bf16> {
        let mut v = Vec::with_capacity(t * k);
        for r in 0..t {
            let row = 2f64.powi((r % 17) as i32 - 8);
            for i in 0..k {
                let x = if r % 7 == 6 { 0.0 } else { self.normal() * row * 2f64.powi(((i / 16) % 9) as i32 - 4) };
                v.push(bf16::from_f64(x));
            }
        }
        v
    }
}

fn stream() -> Arc<cuda_core::Stream> {
    static STREAM: std::sync::OnceLock<Arc<cuda_core::Stream>> = std::sync::OnceLock::new();
    STREAM.get_or_init(|| Context::new(0).unwrap().new_stream().unwrap()).clone()
}

fn upload<T: DType>(v: Vec<T>, shape: &[usize], s: &Arc<cuda_core::Stream>) -> Tensor<T> {
    api::copy_host_vec_to_device(&Arc::new(v)).reshape(shape).sync_on(s).unwrap()
}

fn act_bytes(packed: Tensor<f4e2m1fnx2>, scales: Tensor<f8e4m3fn>, s: &Arc<cuda_core::Stream>) -> (Vec<u8>, Vec<u8>) {
    let p = packed.to_host_vec().sync_on(s).unwrap().iter().map(|b| b.to_bits()).collect();
    let sc = scales.to_host_vec().sync_on(s).unwrap().iter().map(|b| b.0).collect();
    (p, sc)
}

/// The standalone path's quantization of a bf16 `[t, k]` device tensor.
fn standalone_quantize(x: Tensor<bf16>, t: usize, k: usize, igs: f32, s: &Arc<cuda_core::Stream>) -> (Vec<u8>, Vec<u8>) {
    let out = Nvfp4Act::zeros(t, k, s).unwrap();
    let act = nvfp4::quantize(&Arc::new(x), igs, out, QuantTiles::DEFAULT, s).unwrap();
    act_bytes(act.packed, act.scales, s)
}

fn zeros_act(t: usize, k: usize, s: &Arc<cuda_core::Stream>) -> (Tensor<f4e2m1fnx2>, Tensor<f8e4m3fn>) {
    let a = Nvfp4Act::zeros(t, k, s).unwrap();
    (a.packed, a.scales)
}

fn mismatches(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Activation global scales: two in the range of the checkpoint's calibrated values and one tiny one, which
/// drives most group scales to the 448 clamp.
const IGS: [f32; 3] = [628.0, 37.5, 2.0e5];

#[test]
fn zc_rmsnorm_nvfp4_is_bit_identical_to_norm_then_quantize() {
    let s = stream();
    let (t, k) = (300usize, 4096usize);
    let mut rng = Rng(0x0f0f_1234_5678_9abc);
    let x = rng.rows(t, k);
    let w: Vec<bf16> = (0..k).map(|_| bf16::from_f64(rng.normal() * 0.5)).collect();
    let mut bad = Vec::new();
    for igs in IGS {
        let (xd, wd) = (upload(x.clone(), &[t, k], &s), upload(w.clone(), &[k], &s));
        let mut out = api::zeros::<bf16>(&[t, k]).sync_on(&s).unwrap();
        norm::zc_rmsnorm(&mut out, &xd, &wd, norm::RMS_EPS).unwrap().sync_on(&s).unwrap();
        let (wp, ws) = standalone_quantize(out, t, k, igs, &s);
        let (mut q, mut sc) = zeros_act(t, k, &s);
        nvfp4::zc_rmsnorm_nvfp4(&mut q, &mut sc, &xd, &wd, norm::RMS_EPS, igs).unwrap().sync_on(&s).unwrap();
        let (gp, gs) = act_bytes(q, sc, &s);
        let (mp, ms) = (mismatches(&gp, &wp), mismatches(&gs, &ws));
        println!("zc_rmsnorm_nvfp4 T={t} K={k} igs={igs}: packed mismatches {mp}/{}, scale mismatches {ms}/{}", gp.len(), gs.len());
        if mp + ms > 0 {
            bad.push(igs);
        }
    }
    assert!(bad.is_empty(), "fused norm + quantize differs from norm then quantize at igs {bad:?}");
}

#[test]
fn add_zc_rmsnorm_nvfp4_is_bit_identical_to_add_norm_then_quantize() {
    let s = stream();
    let (t, k) = (257usize, 4096usize);
    let mut rng = Rng(0x7777_0000_1111_2222);
    let res = rng.rows(t, k);
    let x = rng.rows(t, k);
    let w: Vec<bf16> = (0..k).map(|_| bf16::from_f64(rng.normal() * 0.5)).collect();
    let mut bad = Vec::new();
    for igs in IGS {
        let (xd, wd) = (upload(x.clone(), &[t, k], &s), upload(w.clone(), &[k], &s));
        let mut r1 = upload(res.clone(), &[t, k], &s);
        let mut out = api::zeros::<bf16>(&[t, k]).sync_on(&s).unwrap();
        norm::add_zc_rmsnorm(&mut r1, &mut out, &xd, &wd, norm::RMS_EPS).unwrap().sync_on(&s).unwrap();
        let (wp, ws) = standalone_quantize(out, t, k, igs, &s);
        let mut r2 = upload(res.clone(), &[t, k], &s);
        let (mut q, mut sc) = zeros_act(t, k, &s);
        nvfp4::add_zc_rmsnorm_nvfp4(&mut r2, &mut q, &mut sc, &xd, &wd, norm::RMS_EPS, igs).unwrap().sync_on(&s).unwrap();
        let (gp, gs) = act_bytes(q, sc, &s);
        let bits = |v: Vec<bf16>| v.iter().map(|b| b.to_bits()).collect::<Vec<u16>>();
        let (ra, rb) = (bits(r1.to_host_vec().sync_on(&s).unwrap()), bits(r2.to_host_vec().sync_on(&s).unwrap()));
        let mr = ra.iter().zip(&rb).filter(|(a, b)| a != b).count();
        let (mp, ms) = (mismatches(&gp, &wp), mismatches(&gs, &ws));
        println!("add_zc_rmsnorm_nvfp4 T={t} K={k} igs={igs}: residual mismatches {mr}/{}, packed {mp}/{}, scales {ms}/{}",
                 ra.len(), gp.len(), gs.len());
        if mr + mp + ms > 0 {
            bad.push(igs);
        }
    }
    assert!(bad.is_empty(), "fused add + norm + quantize differs from the two-kernel path at igs {bad:?}");
}

#[test]
fn silu_mul_nvfp4_is_bit_identical_to_silu_mul_then_quantize() {
    let s = stream();
    let (t, i) = (131usize, 12288usize);
    let mut rng = Rng(0x5151_aaaa_cccc_0001);
    let gu = rng.rows(t, 2 * i);
    let mut bad = Vec::new();
    for igs in IGS {
        let gud = upload(gu.clone(), &[t, 2 * i], &s);
        let mut out = api::zeros::<bf16>(&[t, i]).sync_on(&s).unwrap();
        pointwise::silu_mul(&mut out, &gud).unwrap().sync_on(&s).unwrap();
        let (wp, ws) = standalone_quantize(out, t, i, igs, &s);
        let (mut q, mut sc) = zeros_act(t, i, &s);
        nvfp4::silu_mul_nvfp4(&mut q, &mut sc, &gud, igs).unwrap().sync_on(&s).unwrap();
        let (gp, gs) = act_bytes(q, sc, &s);
        let (mp, ms) = (mismatches(&gp, &wp), mismatches(&gs, &ws));
        println!("silu_mul_nvfp4 T={t} I={i} igs={igs}: packed mismatches {mp}/{}, scale mismatches {ms}/{}", gp.len(), gs.len());
        if mp + ms > 0 {
            bad.push(igs);
        }
    }
    assert!(bad.is_empty(), "fused SiLU·mul + quantize differs from silu_mul then quantize at igs {bad:?}");
}
