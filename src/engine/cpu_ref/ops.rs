//! Token-parallel building blocks of the reference forward, each a direct transcription of the transformers 5.19
//! `modeling_qwen3_5.py` op it names. Statistics (norms, softmax, dot products outside the GEMMs) are accumulated
//! in f64; GEMMs accumulate in f32 like the fp32 reference does.

use super::weights::Mat;
use rayon::prelude::*;

/// Lanes of the f32 GEMM accumulator. Every output element is `sum over lanes of (sum over k = lane mod LANES)`,
/// in a fixed order that depends only on K, so a token's result never depends on which other tokens share the
/// batch or where it sits in it (batch invariance by construction).
const LANES: usize = 4;
/// Weight rows per GEMM task (converted to f32 once per task).
const ROW_BLOCK: usize = 32;

#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    let mut acc = [0.0f32; LANES];
    let full = k / LANES * LANES;
    let mut i = 0;
    while i < full {
        for l in 0..LANES {
            acc[l] += a[i + l] * b[i + l];
        }
        i += LANES;
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for j in full..k {
        s += a[j] * b[j];
    }
    s
}

/// 4 tokens x 4 rows with exactly `dot_f32`'s operation order per output.
#[inline(always)]
fn dot_4x4(x: [&[f32]; 4], w: [&[f32]; 4]) -> [[f32; 4]; 4] {
    let k = x[0].len();
    let full = k / LANES * LANES;
    let mut acc = [[[0.0f32; LANES]; 4]; 4];
    let mut i = 0;
    while i < full {
        let xs: [[f32; LANES]; 4] = std::array::from_fn(|t| x[t][i..i + LANES].try_into().unwrap());
        let ws: [[f32; LANES]; 4] = std::array::from_fn(|r| w[r][i..i + LANES].try_into().unwrap());
        for t in 0..4 {
            for r in 0..4 {
                for l in 0..LANES {
                    acc[t][r][l] += xs[t][l] * ws[r][l];
                }
            }
        }
        i += LANES;
    }
    let mut out = [[0.0f32; 4]; 4];
    for t in 0..4 {
        for r in 0..4 {
            let a = &acc[t][r];
            let mut s = (a[0] + a[1]) + (a[2] + a[3]);
            for j in full..k {
                s += x[t][j] * w[r][j];
            }
            out[t][r] = s;
        }
    }
    out
}

/// `y[t, n] = sum_k x[t, k] * w[n, k]` (`nn.Linear` without bias), x `[t, w.cols]`, y `[t, w.rows]`.
pub fn matmul(x: &[f32], t: usize, w: &Mat) -> Vec<f32> {
    let (n, k) = (w.rows, w.cols);
    assert_eq!(x.len(), t * k, "matmul: x is not [{t}, {k}]");
    if t == 0 {
        return Vec::new();
    }
    // Computed transposed ([n, t]) so each task owns a contiguous block of rows.
    let mut yt = vec![0.0f32; n * t];
    yt.par_chunks_mut(ROW_BLOCK * t).enumerate().for_each(|(bi, ych)| {
        let r0 = bi * ROW_BLOCK;
        let nr = ych.len() / t;
        let mut wb = vec![0.0f32; nr * k];
        for r in 0..nr {
            w.row_into(r0 + r, &mut wb[r * k..(r + 1) * k]);
        }
        let wrow = |r: usize| &wb[r * k..(r + 1) * k];
        let xrow = |i: usize| &x[i * k..(i + 1) * k];
        let (t4, r4) = (t / 4 * 4, nr / 4 * 4);
        for i in (0..t4).step_by(4) {
            let xs = [xrow(i), xrow(i + 1), xrow(i + 2), xrow(i + 3)];
            for r in (0..r4).step_by(4) {
                let o = dot_4x4(xs, [wrow(r), wrow(r + 1), wrow(r + 2), wrow(r + 3)]);
                for dt in 0..4 {
                    for dr in 0..4 {
                        ych[(r + dr) * t + i + dt] = o[dt][dr];
                    }
                }
            }
            for r in r4..nr {
                for dt in 0..4 {
                    ych[r * t + i + dt] = dot_f32(xs[dt], wrow(r));
                }
            }
        }
        for i in t4..t {
            for r in 0..nr {
                ych[r * t + i] = dot_f32(xrow(i), wrow(r));
            }
        }
    });
    let mut y = vec![0.0f32; t * n];
    y.par_chunks_mut(n).enumerate().for_each(|(i, yr)| {
        for (j, v) in yr.iter_mut().enumerate() {
            *v = yt[j * t + i];
        }
    });
    y
}

/// `Qwen3_5RMSNorm` (zero-centred): `x * rsqrt(mean(x^2) + eps) * (1 + w)`, per row of `w.len()`.
pub fn zc_rmsnorm(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let d = w.len();
    let mut y = vec![0.0f32; x.len()];
    y.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(yr, xr)| zc_rmsnorm_row(xr, w, eps, yr));
    y
}

pub fn zc_rmsnorm_row(x: &[f32], w: &[f32], eps: f64, y: &mut [f32]) {
    let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    for ((o, &v), &wv) in y.iter_mut().zip(x).zip(w) {
        *o = (v as f64 * inv * (1.0 + wv as f64)) as f32;
    }
}

/// `Qwen3_5RMSNormGated` in an fp32 model: `w * (x * rsqrt(mean(x^2) + eps)) * silu(z)`, one row of `w.len()`.
pub fn gated_rmsnorm_row(x: &[f32], z: &[f32], w: &[f32], eps: f64, y: &mut [f32]) {
    let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    for (((o, &v), &zv), &wv) in y.iter_mut().zip(x).zip(z).zip(w) {
        *o = (wv as f64 * (v as f64 * inv) * silu(zv as f64)) as f32;
    }
}

/// `l2norm` (FLA-aligned): `x * rsqrt(sum(x^2) + eps)`.
pub fn l2norm_row(x: &mut [f64], eps: f64) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= inv);
}

pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `F.softplus` with its defaults (beta 1, threshold 20: linear above 20).
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// cos/sin of RoPE for one position, `rot/2` frequencies, at the reference's rounding points:
/// `inv_freq = 1 / theta^(2i/rot)` in f32, `freq = f32(pos) * inv_freq` in f32, cos/sin of that f32 angle.
pub fn rope_cos_sin(pos: u32, rot: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = rot / 2;
    let mut c = vec![0.0f32; half];
    let mut s = vec![0.0f32; half];
    for i in 0..half {
        let expo = (2 * i) as f32 / rot as f32;
        let inv = 1.0f32 / (theta as f32).powf(expo);
        let f = pos as f32 * inv;
        c[i] = (f as f64).cos() as f32;
        s[i] = (f as f64).sin() as f32;
    }
    (c, s)
}

/// `apply_rotary_pos_emb` on the first `2*cos.len()` dims of one head (rotate-half style); the rest pass.
pub fn rope_apply(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for i in 0..half {
        let (a, b) = (x[i], x[i + half]);
        // q_embed = q*cos + rotate_half(q)*sin with rotate_half(q) = [-q2, q1], cos/sin = cat(f, f).
        x[i] = a * cos[i] + (-b) * sin[i];
        x[i + half] = b * cos[i] + a * sin[i];
    }
}

/// Numerically stable softmax in f64.
pub fn softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
    let e: Vec<f64> = x.iter().map(|&v| (v as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn matmul_matches_f64_and_is_batch_invariant() {
        let mut s = 7u64;
        let (t, k, n) = (11, 37, 70);
        let x: Vec<f32> = (0..t * k).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let m = Mat::from_f32(n, k, w.clone());
        let y = matmul(&x, t, &m);
        let mut max_err = 0.0f64;
        for i in 0..t {
            for j in 0..n {
                let r: f64 = (0..k).map(|q| x[i * k + q] as f64 * w[j * k + q] as f64).sum();
                max_err = max_err.max((r - y[i * n + j] as f64).abs());
            }
        }
        println!("matmul vs f64: max abs err {max_err:.3e} (tol 1e-5)");
        assert!(max_err < 1e-5);
        // Any subset of rows, at any offset, gives bit-identical results.
        for (a, b) in [(0, 1), (3, 9), (5, 11), (0, 4)] {
            let ys = matmul(&x[a * k..b * k], b - a, &m);
            assert_eq!(ys, y[a * n..b * n]);
        }
    }

    #[test]
    fn rope_rotates_pairs() {
        let (c, s) = rope_cos_sin(5, 8, 10000.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let orig = x.clone();
        rope_apply(&mut x[..], &c, &s);
        // Rotation keeps each (i, i+half) pair's norm; dims past the rotary part pass through.
        for i in 0..4 {
            let n0 = orig[i] * orig[i] + orig[i + 4] * orig[i + 4];
            let n1 = x[i] * x[i] + x[i + 4] * x[i + 4];
            assert!((n0 - n1).abs() < 1e-4);
        }
        assert_eq!(&x[8..], &orig[8..]);
        // Frequency 0 is pos * 1 radian.
        assert!((x[0] - (1.0 * 5f32.cos() - 5.0 * 5f32.sin())).abs() < 1e-5);
    }
}
