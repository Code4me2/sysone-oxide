//! Numeric check of cuda-oxide's warp-level `mma.sync` fragments on the local GPU (G1 step 1).
//!
//! For `TRIALS` random problems `D[16, 8] = C + A[16, K] B[K, 8]` (K = 64, chained MMAs) it runs:
//! - `m16n8k8.f32.tf32` with operands rounded by `cvt.rna.tf32.f32` (1xTF32);
//! - the 3xTF32 split `a_lo b_hi + a_hi b_lo + a_hi b_hi` (hi = rna(x), lo = rna(x - hi));
//! - `m16n8k16.f32.bf16` with operands rounded to bf16 (RNE) on the host;
//! - the bf16x3 split (same three products, bf16 hi/lo);
//! and compares each lane's documented C/D fragment against an f64 CPU result. A wrong fragment layout shows as
//! O(1) errors; a correct one leaves only rounding. Errors are reported relative to `sum_k |a_k b_k|` (the
//! natural scale of an fp32 dot product) in units of 2^-23 (f32 ulp at 1).
//!
//! Run: `scripts/gpu.sh cargo oxide run` from this directory (or `kernels-simt/build.sh check`).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::convert::cvt_rna_tf32_f32;
use cuda_device::wmma::{mma_m16n8k8_f32_tf32, mma_m16n8k16_f32_bf16};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const K: usize = 64;
const TRIALS: usize = 512;

#[cuda_module]
mod kernels {
    use super::*;

    /// Grid (TRIALS), block 32. a: [TRIALS, 16, K] f32, b: [TRIALS, K, 8] f32, c: [TRIALS, 16, 8] f32.
    /// d1 = 1xTF32, d3 = 3xTF32, both [TRIALS, 16, 8].
    #[kernel]
    pub fn tf32_check(a: &[f32], b: &[f32], c: &[f32], mut d1: DisjointSlice<f32>, mut d3: DisjointSlice<f32>) {
        let lane = thread::threadIdx_x() as usize;
        let trial = thread::blockIdx_x() as usize;
        let (g, t) = (lane / 4, lane % 4);
        let (ab, bb, cb) = (trial * 16 * K, trial * K * 8, trial * 128);
        let c_idx = [cb + g * 8 + 2 * t, cb + g * 8 + 2 * t + 1, cb + (g + 8) * 8 + 2 * t, cb + (g + 8) * 8 + 2 * t + 1];
        let mut acc1 = [c[c_idx[0]], c[c_idx[1]], c[c_idx[2]], c[c_idx[3]]];
        let mut acc3 = acc1;
        let mut kk = 0usize;
        while kk < K {
            // A register j: row g + 8*(j&1), col t + 4*(j>>1). B register j: row t + 4j, col g.
            let av = [
                a[ab + g * K + kk + t],
                a[ab + (g + 8) * K + kk + t],
                a[ab + g * K + kk + t + 4],
                a[ab + (g + 8) * K + kk + t + 4],
            ];
            let bv = [b[bb + (kk + t) * 8 + g], b[bb + (kk + t + 4) * 8 + g]];
            let ah = [cvt_rna_tf32_f32(av[0]), cvt_rna_tf32_f32(av[1]), cvt_rna_tf32_f32(av[2]), cvt_rna_tf32_f32(av[3])];
            let al = [
                cvt_rna_tf32_f32(av[0] - f32::from_bits(ah[0])),
                cvt_rna_tf32_f32(av[1] - f32::from_bits(ah[1])),
                cvt_rna_tf32_f32(av[2] - f32::from_bits(ah[2])),
                cvt_rna_tf32_f32(av[3] - f32::from_bits(ah[3])),
            ];
            let bh = [cvt_rna_tf32_f32(bv[0]), cvt_rna_tf32_f32(bv[1])];
            let bl = [cvt_rna_tf32_f32(bv[0] - f32::from_bits(bh[0])), cvt_rna_tf32_f32(bv[1] - f32::from_bits(bh[1]))];
            unsafe {
                acc1 = mma_m16n8k8_f32_tf32(acc1, ah, bh);
                acc3 = mma_m16n8k8_f32_tf32(acc3, al, bh);
                acc3 = mma_m16n8k8_f32_tf32(acc3, ah, bl);
                acc3 = mma_m16n8k8_f32_tf32(acc3, ah, bh);
            }
            kk += 8;
        }
        unsafe {
            for j in 0..4 {
                *d1.get_unchecked_mut(c_idx[j]) = acc1[j];
                *d3.get_unchecked_mut(c_idx[j]) = acc3[j];
            }
        }
    }

    /// Same problem with bf16 operands: `a`, `b` are f32 and are split on the device into bf16 hi = rne(x) and
    /// lo = rne(x - hi). d1 = hi*hi only, d3 = lo*hi + hi*lo + hi*hi.
    #[kernel]
    pub fn bf16_check(a: &[f32], b: &[f32], c: &[f32], mut d1: DisjointSlice<f32>, mut d3: DisjointSlice<f32>) {
        let lane = thread::threadIdx_x() as usize;
        let trial = thread::blockIdx_x() as usize;
        let (g, t) = (lane / 4, lane % 4);
        let (ab, bb, cb) = (trial * 16 * K, trial * K * 8, trial * 128);
        let c_idx = [cb + g * 8 + 2 * t, cb + g * 8 + 2 * t + 1, cb + (g + 8) * 8 + 2 * t, cb + (g + 8) * 8 + 2 * t + 1];
        let mut acc1 = [c[c_idx[0]], c[c_idx[1]], c[c_idx[2]], c[c_idx[3]]];
        let mut acc3 = acc1;
        let mut kk = 0usize;
        while kk < K {
            // A element j (packed a[j/2], low half first): row g (j in 0,1,4,5) else g+8; col 2t + (j&1) + 8*(j>=4).
            // B element j: row 2t + (j&1) + 8*(j>=2), col g.
            let mut ah = [0u32; 4];
            let mut al = [0u32; 4];
            for r in 0..4 {
                let row = if r % 2 == 0 { g } else { g + 8 };
                let col = 2 * t + if r >= 2 { 8 } else { 0 };
                let x0 = a[ab + row * K + kk + col];
                let x1 = a[ab + row * K + kk + col + 1];
                let (h0, l0) = split_bf16(x0);
                let (h1, l1) = split_bf16(x1);
                ah[r] = (h0 as u32) | ((h1 as u32) << 16);
                al[r] = (l0 as u32) | ((l1 as u32) << 16);
            }
            let mut bh = [0u32; 2];
            let mut bl = [0u32; 2];
            for r in 0..2 {
                let row = 2 * t + if r == 1 { 8 } else { 0 };
                let (h0, l0) = split_bf16(b[bb + (kk + row) * 8 + g]);
                let (h1, l1) = split_bf16(b[bb + (kk + row + 1) * 8 + g]);
                bh[r] = (h0 as u32) | ((h1 as u32) << 16);
                bl[r] = (l0 as u32) | ((l1 as u32) << 16);
            }
            unsafe {
                acc1 = mma_m16n8k16_f32_bf16(acc1, ah, bh);
                acc3 = mma_m16n8k16_f32_bf16(acc3, al, bh);
                acc3 = mma_m16n8k16_f32_bf16(acc3, ah, bl);
                acc3 = mma_m16n8k16_f32_bf16(acc3, ah, bh);
            }
            kk += 16;
        }
        unsafe {
            for j in 0..4 {
                *d1.get_unchecked_mut(c_idx[j]) = acc1[j];
                *d3.get_unchecked_mut(c_idx[j]) = acc3[j];
            }
        }
    }
}

/// bf16 round-to-nearest-even of `x` and of the remainder `x - hi`.
#[inline(always)]
fn split_bf16(x: f32) -> (u16, u16) {
    let h = rne_bf16(x);
    let l = rne_bf16(x - f32::from_bits((h as u32) << 16));
    (h, l)
}

#[inline(always)]
fn rne_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

/// Host model of `cvt.rna.tf32.f32` (round to nearest, ties away from zero, 10 explicit mantissa bits).
fn rna_tf32(x: f32) -> f32 {
    f32::from_bits(x.to_bits().wrapping_add(0x1000) & 0xffff_e000)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Signed values over several binades, like GDN operands (normalised keys, decayed states).
    fn value(&mut self) -> f32 {
        let m = 2.0 * self.next() - 1.0;
        let e = (self.next() * 8.0).floor() as i32 - 4;
        (m * 2f64.powi(e)) as f32
    }
}

/// Max and mean of |got - want| / sum_k |a_k b_k| over all outputs, in units of 2^-23.
fn stats(got: &[f32], want: &[f64], scale: &[f64]) -> (f64, f64) {
    let ulp = 2f64.powi(-23);
    let mut max = 0f64;
    let mut sum = 0f64;
    for i in 0..got.len() {
        let e = (got[i] as f64 - want[i]).abs() / scale[i] / ulp;
        assert!(e.is_finite(), "non-finite result at {i}");
        max = max.max(e);
        sum += e;
    }
    (max, sum / got.len() as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let a: Vec<f32> = (0..TRIALS * 16 * K).map(|_| rng.value()).collect();
    let b: Vec<f32> = (0..TRIALS * K * 8).map(|_| rng.value()).collect();
    let c: Vec<f32> = (0..TRIALS * 128).map(|_| rng.value()).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let c_dev = DeviceBuffer::from_host(&stream, &c)?;
    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (TRIALS as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };

    // f64 references: exact products of the (rounded) operands, plus the scale sum |a b|.
    let reference = |ra: &dyn Fn(f32) -> f64, rb: &dyn Fn(f32) -> f64| -> (Vec<f64>, Vec<f64>) {
        let mut want = vec![0f64; TRIALS * 128];
        let mut scale = vec![0f64; TRIALS * 128];
        for tr in 0..TRIALS {
            for m in 0..16 {
                for n in 0..8 {
                    let (mut s, mut sc) = (c[tr * 128 + m * 8 + n] as f64, (c[tr * 128 + m * 8 + n] as f64).abs());
                    for k in 0..K {
                        let p = ra(a[tr * 16 * K + m * K + k]) * rb(b[tr * K * 8 + k * 8 + n]);
                        s += p;
                        sc += p.abs();
                    }
                    want[tr * 128 + m * 8 + n] = s;
                    scale[tr * 128 + m * 8 + n] = sc;
                }
            }
        }
        (want, scale)
    };
    let exact = |x: f32| x as f64;
    let tf = |x: f32| rna_tf32(x) as f64;
    let bf = |x: f32| f32::from_bits((rne_bf16(x) as u32) << 16) as f64;
    let (want_exact, scale) = reference(&exact, &exact);
    let (want_tf, _) = reference(&tf, &tf);
    let (want_bf, _) = reference(&bf, &bf);

    let mut failures = 0;
    let mut check = |name: &str, got: &[f32], want: &[f64], bound: f64| {
        let (max, mean) = stats(got, want, &scale);
        let ok = max <= bound;
        println!("{name:<38} max {max:>10.2} mean {mean:>8.3} (x 2^-23 x sum|ab|)  bound {bound:>8.0}  {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    for kind in ["tf32", "bf16"] {
        let mut d1 = DeviceBuffer::<f32>::zeroed(&stream, TRIALS * 128)?;
        let mut d3 = DeviceBuffer::<f32>::zeroed(&stream, TRIALS * 128)?;
        unsafe {
            if kind == "tf32" {
                module.tf32_check(&stream, cfg, &a_dev, &b_dev, &c_dev, &mut d1, &mut d3)?;
            } else {
                module.bf16_check(&stream, cfg, &a_dev, &b_dev, &c_dev, &mut d1, &mut d3)?;
            }
        }
        let d1 = d1.to_host_vec(&stream)?;
        let d3 = d3.to_host_vec(&stream)?;
        if kind == "tf32" {
            // Products of TF32 operands are exact in f32; only the fp32 accumulation rounds (K/8 + 1 chained adds,
            // each <= a few ulp of the running sum, which is <= sum|ab|).
            check("tf32 m16n8k8: vs f64(tf32 operands)", &d1, &want_tf, 64.0);
            check("tf32 m16n8k8: vs f64(f32 operands)", &d1, &want_exact, f64::INFINITY);
            check("3xTF32 m16n8k8: vs f64(f32 operands)", &d3, &want_exact, 64.0);
        } else {
            check("bf16 m16n8k16: vs f64(bf16 operands)", &d1, &want_bf, 64.0);
            check("bf16 m16n8k16: vs f64(f32 operands)", &d1, &want_exact, f64::INFINITY);
            check("3xbf16 m16n8k16: vs f64(f32 operands)", &d3, &want_exact, f64::INFINITY);
        }
    }
    println!("{TRIALS} trials of [16x{K}]x[{K}x8] on sm_{major}{minor}");
    if failures > 0 {
        eprintln!("FAILED: {failures} checks");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
