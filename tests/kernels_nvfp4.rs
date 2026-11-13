//! GPU tests for `sysone::kernels::nvfp4` (DESIGN.md §3.1, §6.6, §9.2 "NVFP4" row).
//!
//! 1. `quantize_nvfp4` bit-exact (packed bytes and scale bytes) against compressed-tensors 0.20.1 on the vectors in
//!    `reference/nvfp4/vectors` (random, wide-range, ties, zeros/-0/inf/extremes), and against an independent CPU
//!    port of the compressed-tensors math on further shapes (partial row tiles, K = 4096 and 12288).
//! 2. `gemm_nvfp4` vs a CPU f64 dequant-matmul on small shapes, with per-half alpha.
//! 3. Real layer-0 MLP weights of elephant-1-nvfp4-mlp with random activations vs the compressed-tensors QDQ path in
//!    torch (`reference/nvfp4/gen_layer0.py`); fails (with how to generate it) when its data file is
//!    absent, rather than passing vacuously.
//!
//! Run: `cargo test --release --test kernels_nvfp4 -- --nocapture --test-threads 1`

use cutile::cuda_core::{Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use std::path::{Path, PathBuf};
use sysone::kernels::nvfp4::{self, CheckpointProj, GemmKernel, GemmTiles, Nvfp4Act, Nvfp4Linear, QuantTiles};

const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Gpu {
    ctx: Arc<Context>,
    stream: Arc<Stream>,
    sms: usize,
}

fn gpu() -> Gpu {
    let ctx = Context::new(0).expect("CUDA context");
    let stream = ctx.new_stream().expect("stream");
    let sms = ctx.multiprocessor_count().expect("SM count") as usize;
    Gpu { ctx, stream, sms }
}

// ---------------------------------------------------------------------------------------------------------------
// Independent CPU references (written from the compressed-tensors source, see module docs of kernels::nvfp4).
// ---------------------------------------------------------------------------------------------------------------

/// e4m3fn decode (bias 7, no inf, 0x7f/0xff = NaN).
fn e4m3_to_f64(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f64;
    if e == 15 && (b & 7) == 7 {
        return f64::NAN;
    }
    if e == 0 { sign * m / 8.0 * 2f64.powi(-6) } else { sign * (1.0 + m / 8.0) * 2f64.powi(e - 7) }
}

/// f32 → e4m3fn, round to nearest even, input already in [-448, 448] (torch's `.to(float8_e4m3fn)` after clamp).
fn f32_to_e4m3(v: f32) -> u8 {
    // Enumerate the 127 non-negative finite codes and pick the nearest; ties to the even code (mantissa LSB 0).
    let a = v.abs() as f64;
    let mut best = 0u8;
    let mut best_d = f64::INFINITY;
    for code in 0u8..0x7f {
        let d = (e4m3_to_f64(code) - a).abs();
        if d < best_d || (d == best_d && code & 1 == 0) {
            best_d = d;
            best = code;
        }
    }
    if v.is_sign_negative() && best != 0 { best | 0x80 } else { best }
}

/// f32 → bf16 round to nearest even (torch's bf16 rounding), finite inputs and inf.
fn f32_to_bf16_rne(v: f32) -> f32 {
    bf16::from_f32(v).to_f32() // half's from_f32 is RNE
}

/// compressed-tensors' fp4 threshold cascade (`cast_to_fp4`), returns the nibble.
fn ct_e2m1_nibble(y: f32) -> u8 {
    let a = y.abs();
    let mag = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    // torch.sign(y) * |q|: negative only when y < 0 (y = -0.0 was turned into +0.0 by the zero-point add).
    if y < 0.0 { mag | 8 } else { mag }
}

/// CPU port of the compressed-tensors activation path for one row: (packed bytes, scale bytes, qdq bf16 values).
fn cpu_quantize_row(x: &[bf16], igs: f32) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let mut nibbles = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len() / 16);
    let mut qdq = Vec::with_capacity(x.len());
    for g in x.chunks(16) {
        let amax = g.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
        let s_bf = f32_to_bf16_rne(amax / 6.0);
        let t = (igs * s_bf).min(448.0);
        let mut s8 = f32_to_e4m3(t);
        let mut s = e4m3_to_f64(s8) as f32;
        if s == 0.0 {
            s = 0.125;
            s8 = 0x20;
        }
        scales.push(s8);
        let eff = s / igs;
        for v in g {
            let y = (v.to_f32() / eff + 0.0).clamp(-6.0, 6.0);
            let nib = ct_e2m1_nibble(y);
            nibbles.push(nib);
            let q = E2M1[(nib & 7) as usize] as f32 * if nib & 8 != 0 { -1.0 } else { 1.0 };
            qdq.push(f32_to_bf16_rne(q * eff));
        }
    }
    let packed = nibbles.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
    (packed, scales, qdq)
}

fn nibble_value(b: u8, hi: bool) -> f64 {
    let n = if hi { b >> 4 } else { b & 0xf };
    E2M1[(n & 7) as usize] * if n & 8 != 0 { -1.0 } else { 1.0 }
}

/// f64 `alpha[n] · Σ_k (q_x s_x)(q_w s_w)`: the exact value the block-scaled GEMM approximates, and Σ|terms|.
fn cpu_gemm_f64(
    t: usize,
    n: usize,
    k: usize,
    xp: &[u8],
    xs: &[u8],
    wp: &[u8],
    ws: &[u8],
    alpha: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let deq = |p: &[u8], s: &[u8], rows: usize| -> Vec<f64> {
        let mut out = vec![0.0; rows * k];
        for r in 0..rows {
            for c in 0..k {
                let b = p[r * k / 2 + c / 2];
                out[r * k + c] = nibble_value(b, c % 2 == 1) * e4m3_to_f64(s[r * k / 16 + c / 16]);
            }
        }
        out
    };
    let (xd, wd) = (deq(xp, xs, t), deq(wp, ws, n));
    let mut y = vec![0.0; t * n];
    let mut mag = vec![0.0; t * n];
    for i in 0..t {
        for j in 0..n {
            let (mut acc, mut abs) = (0.0f64, 0.0f64);
            for c in 0..k {
                let p = xd[i * k + c] * wd[j * k + c];
                acc += p;
                abs += p.abs();
            }
            y[i * n + j] = acc * alpha[j] as f64;
            mag[i * n + j] = abs * alpha[j] as f64;
        }
    }
    (y, mag)
}

fn bf16_ulp(v: f64) -> f64 {
    let a = v.abs().max(f32::MIN_POSITIVE as f64);
    2f64.powi(a.log2().floor() as i32 - 7)
}

struct Stats {
    v: Vec<f64>,
}

impl Stats {
    fn new() -> Self {
        Stats { v: Vec::new() }
    }
    fn push(&mut self, x: f64) {
        self.v.push(x);
    }
    fn max(&self) -> f64 {
        self.v.iter().cloned().fold(0.0, f64::max)
    }
    fn pct(&self, p: f64) -> f64 {
        let mut s = self.v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if s.is_empty() { 0.0 } else { s[((s.len() - 1) as f64 * p).round() as usize] }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------------------------------------

fn upload_bf16(g: &Gpu, data: &[bf16], t: usize, k: usize) -> Arc<Tensor<bf16>> {
    api::copy_host_vec_to_device(&Arc::new(data.to_vec())).reshape(&[t, k]).sync_on(&g.stream).unwrap().into()
}

fn run_quantize(g: &Gpu, x: &[bf16], t: usize, k: usize, igs: f32, tiles: QuantTiles) -> (Vec<u8>, Vec<u8>) {
    let xd = upload_bf16(g, x, t, k);
    let out = Nvfp4Act::zeros(t, k, &g.stream).unwrap();
    let act = nvfp4::quantize(&xd, igs, out, tiles, &g.stream).expect("quantize launch");
    let p: Vec<u8> = act.packed.to_host_vec().sync_on(&g.stream).unwrap().iter().map(|b| b.to_bits()).collect();
    let s: Vec<u8> = act.scales.to_host_vec().sync_on(&g.stream).unwrap().iter().map(|b| b.0).collect();
    (p, s)
}

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
}

// ---------------------------------------------------------------------------------------------------------------
// 1. quantize_nvfp4
// ---------------------------------------------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct QuantCase {
    name: String,
    t: usize,
    k: usize,
    igs: f32,
    x_bf16: usize,
    packed_u8: usize,
    scales_u8: usize,
    qdq_bf16: usize,
}

#[derive(serde::Deserialize)]
struct QuantIndex {
    cases: Vec<QuantCase>,
}

fn bf16_at(blob: &[u8], off: usize, n: usize) -> Vec<bf16> {
    blob[off..off + 2 * n].chunks(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect()
}

fn diff_bytes(a: &[u8], b: &[u8]) -> Vec<usize> {
    a.iter().zip(b).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect()
}

#[test]
fn quantize_bit_exact_vs_compressed_tensors() {
    let dir = repo().join("reference/nvfp4/vectors");
    let index: QuantIndex = serde_json::from_slice(&std::fs::read(dir.join("quant.json")).unwrap()).unwrap();
    let blob = std::fs::read(dir.join("quant.bin")).unwrap();
    let g = gpu();
    let mut failures = Vec::new();
    for c in &index.cases {
        let x = bf16_at(&blob, c.x_bf16, c.t * c.k);
        let ref_p = &blob[c.packed_u8..c.packed_u8 + c.t * c.k / 2];
        let ref_s = &blob[c.scales_u8..c.scales_u8 + c.t * c.k / 16];
        let ref_q = bf16_at(&blob, c.qdq_bf16, c.t * c.k);

        // The CPU port must agree with compressed-tensors too (it is the reference for the extra shapes below).
        let (mut cp, mut cs, mut cq) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..c.t {
            let (p, s, q) = cpu_quantize_row(&x[r * c.k..(r + 1) * c.k], c.igs);
            cp.extend(p);
            cs.extend(s);
            cq.extend(q);
        }
        let cpu_ok = cp == ref_p
            && cs == ref_s
            && cq.iter().zip(&ref_q).all(|(a, b)| *a == b.to_f32() || (a.is_nan() && b.to_f32().is_nan()));

        let bc = if c.k % 256 == 0 { 256 } else { 16 * (c.k / 16).min(16) };
        for tiles in [QuantTiles::DEFAULT, QuantTiles { br: 4, bc }, QuantTiles { br: 32, bc: 128 }] {
            if c.k % tiles.bc != 0 {
                continue;
            }
            let (p, s) = run_quantize(&g, &x, c.t, c.k, c.igs, tiles);
            let (dp, ds) = (diff_bytes(&p, ref_p), diff_bytes(&s, ref_s));
            println!(
                "quantize {:<26} T={:<3} K={:<5} igs={:<11} tiles={:>2}x{:<3} packed mismatches {}/{} scale mismatches {}/{} cpu-port={}",
                c.name, c.t, c.k, c.igs, tiles.br, tiles.bc, dp.len(), p.len(), ds.len(), s.len(),
                if cpu_ok { "exact" } else { "MISMATCH" }
            );
            for &i in dp.iter().take(4) {
                let (row, col) = (i / (c.k / 2), 2 * (i % (c.k / 2)));
                println!(
                    "   packed[{row},{col}]: got {:#04x} want {:#04x}  x={} {} scale(got {:#04x} want {:#04x})",
                    p[i], ref_p[i], x[row * c.k + col], x[row * c.k + col + 1],
                    s[row * c.k / 16 + col / 16], ref_s[row * c.k / 16 + col / 16]
                );
            }
            for &i in ds.iter().take(4) {
                println!("   scale[{i}]: got {:#04x} want {:#04x}", s[i], ref_s[i]);
            }
            if !dp.is_empty() || !ds.is_empty() {
                failures.push(format!("{} {}x{}", c.name, tiles.br, tiles.bc));
            }
        }
        if !cpu_ok {
            failures.push(format!("{} cpu port", c.name));
        }
    }
    assert!(failures.is_empty(), "not bit-exact: {failures:?}");
}

#[test]
fn quantize_bit_exact_vs_cpu_port_shapes() {
    // Shapes the engine uses (K = 4096 for gate/up, 12288 for down), with T not a multiple of the row tile.
    let g = gpu();
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut failures = Vec::new();
    for &(t, k, igs) in &[(1usize, 4096usize, 628.0f32), (37, 4096, 628.0), (19, 12288, 692.0), (130, 4096, 3.0e-2)] {
        let x: Vec<bf16> = (0..t * k)
            .map(|i| {
                let group_mag = 2f64.powf(((i / 16) % 23) as f64 - 11.0);
                bf16::from_f64(rng.normal() * group_mag)
            })
            .collect();
        let (mut rp, mut rs) = (Vec::new(), Vec::new());
        for r in 0..t {
            let (p, s, _) = cpu_quantize_row(&x[r * k..(r + 1) * k], igs);
            rp.extend(p);
            rs.extend(s);
        }
        let (p, s) = run_quantize(&g, &x, t, k, igs, QuantTiles::DEFAULT);
        let (dp, ds) = (diff_bytes(&p, &rp), diff_bytes(&s, &rs));
        println!("quantize cpu-port T={t:<4} K={k:<6} igs={igs:<8} packed mismatches {}/{} scale mismatches {}/{}",
                 dp.len(), p.len(), ds.len(), s.len());
        if !dp.is_empty() || !ds.is_empty() {
            failures.push((t, k));
        }
    }
    assert!(failures.is_empty(), "not bit-exact: {failures:?}");
}

// ---------------------------------------------------------------------------------------------------------------
// 2. gemm_nvfp4 vs CPU f64
// ---------------------------------------------------------------------------------------------------------------

/// e4m3 scale bytes with exponents in [2^-3, 2^3], never NaN.
fn rand_scale(rng: &mut Rng) -> u8 {
    let e = 4 + (rng.next() % 7) as u8; // exponent field 4..10 -> 2^-3..2^3
    (e << 3) | (rng.next() & 7) as u8
}

#[test]
fn gemm_small_vs_cpu_f64() {
    let g = gpu();
    let mut rng = Rng(0xfeed_0000_0000_0042);
    let configs = [
        GemmTiles::DEFAULT,
        GemmTiles { bm: 128, bn: 64, bk: 128, group_m: 4, blocks_per_sm: 1, kernel: GemmKernel::Raw },
        GemmTiles { bm: 128, bn: 128, bk: 256, group_m: 1, blocks_per_sm: 1, kernel: GemmKernel::Raw },
        GemmTiles { bm: 128, bn: 256, bk: 128, group_m: 8, blocks_per_sm: 2, kernel: GemmKernel::Raw },
        GemmTiles { kernel: GemmKernel::Mapped, ..GemmTiles::DEFAULT },
        GemmTiles { bm: 64, bn: 128, bk: 128, group_m: 4, blocks_per_sm: 2, kernel: GemmKernel::Mapped },
    ];
    let mut failures = Vec::new();
    for &(t, nh, k) in &[(1usize, 256usize, 256usize), (37, 256, 512), (128, 256, 256), (300, 256, 1024)] {
        // Two "projections" (gate, up) with different weight global scales, one shared input scale.
        let igs = 628.0f32;
        let wgs = [9408.0f32, 4128.0f32];
        let wp: Vec<Vec<u8>> = (0..2).map(|_| (0..nh * k / 2).map(|_| rng.next() as u8).collect()).collect();
        let ws: Vec<Vec<u8>> = (0..2).map(|_| (0..nh * k / 16).map(|_| rand_scale(&mut rng)).collect()).collect();
        let projs: Vec<CheckpointProj> = (0..2)
            .map(|h| CheckpointProj {
                n: nh,
                k,
                weight_packed: &wp[h],
                weight_scale: &ws[h],
                weight_global_scale: wgs[h],
                input_global_scale: igs,
            })
            .collect();
        let prep = nvfp4::prepare_weights(&projs).unwrap();
        let lin = Nvfp4Linear::upload(&prep, &g.stream).unwrap();
        let xp: Vec<u8> = (0..t * k / 2).map(|_| rng.next() as u8).collect();
        let xs: Vec<u8> = (0..t * k / 16).map(|_| rand_scale(&mut rng)).collect();
        let xpd: Arc<Tensor<f4e2m1fnx2>> = api::copy_host_vec_to_device(&Arc::new(
            xp.iter().map(|&b| f4e2m1fnx2::from_bits(b)).collect::<Vec<_>>(),
        ))
        .reshape(&[t, k / 2])
        .sync_on(&g.stream)
        .unwrap()
        .into();
        let xsd: Arc<Tensor<f8e4m3fn>> =
            api::copy_host_vec_to_device(&Arc::new(xs.iter().map(|&b| f8e4m3fn(b)).collect::<Vec<_>>()))
                .reshape(&[t, k / 16])
                .sync_on(&g.stream)
                .unwrap()
                .into();
        let n = 2 * nh;
        let (want, mag) = cpu_gemm_f64(t, n, k, &xp, &xs, &prep.packed, &prep.scales, &prep.alpha);
        for tiles in configs {
            let z = api::zeros::<bf16>(&[t, n]).sync_on(&g.stream).unwrap();
            let z = nvfp4::gemm(&xpd, &xsd, &lin, z, tiles, g.sms, &g.stream).expect("gemm launch");
            let got: Vec<f32> = z.to_host_vec().sync_on(&g.stream).unwrap().iter().map(|v| v.to_f32()).collect();
            // Tolerance: 1 bf16 ulp of the exact value (output rounding) + a worst-case f32 accumulation bound.
            let (mut ulps, mut bad) = (Stats::new(), 0usize);
            for i in 0..t * n {
                let err = (got[i] as f64 - want[i]).abs();
                let tol = bf16_ulp(want[i]) + k as f64 * 2f64.powi(-24) * mag[i];
                ulps.push(err / bf16_ulp(want[i]));
                if !(err <= tol) {
                    bad += 1;
                }
            }
            println!(
                "gemm T={t:<4} N={n:<4} K={k:<5} tiles={}x{}x{} gm={} bps={} {} |err| in bf16 ulps: max {:.3} p99 {:.3}  out of tol {bad}",
                tiles.bm, tiles.bn, tiles.bk, tiles.group_m, tiles.blocks_per_sm,
                format!("{:?}", tiles.kernel),
                ulps.max(), ulps.pct(0.99)
            );
            if bad > 0 {
                failures.push((t, n, k, tiles));
            }
        }
    }
    let _ = &g.ctx;
    assert!(failures.is_empty(), "outside tolerance: {failures:?}");
}

// ---------------------------------------------------------------------------------------------------------------
// 3. Real layer-0 MLP weights vs the compressed-tensors QDQ path (torch, GPU, bf16 out)
// ---------------------------------------------------------------------------------------------------------------

/// Minimal safetensors reader: returns (dtype, shape, bytes) of one tensor.
fn safetensor(path: &Path, name: &str) -> (String, Vec<usize>, Vec<u8>) {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).unwrap();
    let mut len = [0u8; 8];
    f.read_exact(&mut len).unwrap();
    let hlen = u64::from_le_bytes(len) as usize;
    let mut h = vec![0u8; hlen];
    f.read_exact(&mut h).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&h).unwrap();
    let e = &header[name];
    let dtype = e["dtype"].as_str().unwrap_or_else(|| panic!("{name} not in {path:?}")).to_string();
    let shape = e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let off = e["data_offsets"].as_array().unwrap();
    let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
    let mut data = vec![0u8; (b - a) as usize];
    f.seek(SeekFrom::Start(8 + hlen as u64 + a)).unwrap();
    f.read_exact(&mut data).unwrap();
    (dtype, shape, data)
}

fn f32_scalar(path: &Path, name: &str) -> f32 {
    let (dt, shape, d) = safetensor(path, name);
    assert_eq!((dt.as_str(), shape.as_slice()), ("F32", &[1usize][..]), "{name}");
    f32::from_le_bytes([d[0], d[1], d[2], d[3]])
}

#[derive(serde::Deserialize)]
struct Layer0Index {
    checkpoint: String,
    t: usize,
    /// name -> (byte offset, element count) in layer0.bin; bf16 unless noted.
    arrays: std::collections::BTreeMap<String, (usize, usize)>,
}

#[test]
fn gemm_layer0_vs_compressed_tensors_qdq() {
    let dir = repo().join("reference/nvfp4/data");
    // The data is generated, not committed; a missing file must fail rather than pass vacuously.
    let idx = std::fs::read(dir.join("layer0.json")).unwrap_or_else(|e| {
        panic!("{dir:?}/layer0.json: {e}; generate it with reference/nvfp4/gen_layer0.py (reference venv)")
    });
    let idx: Layer0Index = serde_json::from_slice(&idx).unwrap();
    let blob = std::fs::read(dir.join("layer0.bin")).unwrap();
    let get_bf16 = |name: &str| {
        let (off, n) = idx.arrays[name];
        bf16_at(&blob, off, n)
    };
    let get_f32 = |name: &str| -> Vec<f32> {
        let (off, n) = idx.arrays[name];
        blob[off..off + 4 * n].chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let ckpt = Path::new(&idx.checkpoint).join("model_backbone.safetensors");
    let pre = "model.language_model.layers.0.mlp.";
    let load = |p: &str| {
        let (_, shape, packed) = safetensor(&ckpt, &format!("{pre}{p}.weight_packed"));
        let (_, _, scale) = safetensor(&ckpt, &format!("{pre}{p}.weight_scale"));
        let wgs = f32_scalar(&ckpt, &format!("{pre}{p}.weight_global_scale"));
        let igs = f32_scalar(&ckpt, &format!("{pre}{p}.input_global_scale"));
        (shape[0], shape[1] * 2, packed, scale, wgs, igs)
    };
    let (gate, up, down) = (load("gate_proj"), load("up_proj"), load("down_proj"));
    fn proj(p: &(usize, usize, Vec<u8>, Vec<u8>, f32, f32)) -> CheckpointProj<'_> {
        CheckpointProj {
            n: p.0,
            k: p.1,
            weight_packed: &p.2,
            weight_scale: &p.3,
            weight_global_scale: p.4,
            input_global_scale: p.5,
        }
    }
    let t = idx.t;
    let g = gpu();
    let gu = Nvfp4Linear::upload(&nvfp4::prepare_weights(&[proj(&gate), proj(&up)]).unwrap(), &g.stream).unwrap();
    let dn = Nvfp4Linear::upload(&nvfp4::prepare_weights(&[proj(&down)]).unwrap(), &g.stream).unwrap();
    println!("layer0: gate wgs {} up wgs {} igs {}; down wgs {} igs {}", gate.4, up.4, gate.5, down.4, down.5);

    let mut ok = true;
    let mut run = |label: &str, lin: &Nvfp4Linear, x: Vec<bf16>, ref_bf16: Vec<bf16>, ideal: Vec<f32>| {
        let k = lin.k;
        let xd = upload_bf16(&g, &x, t, k);
        let act = nvfp4::quantize(&xd, lin.input_global_scale, Nvfp4Act::zeros(t, k, &g.stream).unwrap(),
                                  QuantTiles::DEFAULT, &g.stream).unwrap();
        let (xp, xs): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) = (act.packed.into(), act.scales.into());
        let z = api::zeros::<bf16>(&[t, lin.n]).sync_on(&g.stream).unwrap();
        let z = nvfp4::gemm(&xp, &xs, lin, z, GemmTiles::DEFAULT, g.sms, &g.stream).unwrap();
        let got: Vec<f32> = z.to_host_vec().sync_on(&g.stream).unwrap().iter().map(|v| v.to_f32()).collect();
        let rms = |v: &[f32]| (v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let refv: Vec<f32> = ref_bf16.iter().map(|v| v.to_f32()).collect();
        let r = rms(&refv);
        // (i) vs the exact dequant-matmul (f64, torch): only bf16 output rounding + f32 accumulation remain.
        let (mut e_ideal, mut bad_ideal) = (Stats::new(), 0usize);
        // (ii) vs the compressed-tensors QDQ bf16 output: adds the reference's bf16 operand rounding.
        let (mut e_ref_norm, mut e_ref_rel) = (Stats::new(), Stats::new());
        for i in 0..got.len() {
            let ei = (got[i] as f64 - ideal[i] as f64).abs();
            e_ideal.push(ei / bf16_ulp(ideal[i] as f64));
            // f32 accumulation over K: 1e-6 * rms bound is loose for these magnitudes; mostly the bf16 rounding.
            if !(ei <= bf16_ulp(ideal[i] as f64) + 1e-5 * r) {
                bad_ideal += 1;
            }
            let er = (got[i] - refv[i]).abs() as f64;
            e_ref_norm.push(er / r);
            if refv[i].abs() as f64 >= 0.1 * r {
                e_ref_rel.push(er / refv[i].abs() as f64);
            }
        }
        let (mut ref_vs_ideal_norm, mut ref_vs_ideal_rel) = (Stats::new(), Stats::new());
        for i in 0..got.len() {
            let e = (refv[i] as f64 - ideal[i] as f64).abs();
            ref_vs_ideal_norm.push(e / r);
            if refv[i].abs() as f64 >= 0.1 * r {
                ref_vs_ideal_rel.push(e / refv[i].abs() as f64);
            }
        }
        println!("{label}: T={t} N={} K={k} rms(ref)={r:.4}", lin.n);
        println!("   ours vs exact f64 dequant-matmul: |err| in bf16 ulps max {:.3} p99 {:.3}; out of tol (1 ulp + 1e-5 rms): {bad_ideal}",
                 e_ideal.max(), e_ideal.pct(0.99));
        println!("   ours vs CT QDQ bf16 ref: |err|/rms max {:.3e} p99 {:.3e}; rel (|ref|>=0.1 rms) max {:.3e} p99 {:.3e}",
                 e_ref_norm.max(), e_ref_norm.pct(0.99), e_ref_rel.max(), e_ref_rel.pct(0.99));
        println!("   (CT QDQ ref vs exact, for scale: |err|/rms max {:.3e} p99 {:.3e}; rel max {:.3e} p99 {:.3e})",
                 ref_vs_ideal_norm.max(), ref_vs_ideal_norm.pct(0.99), ref_vs_ideal_rel.max(), ref_vs_ideal_rel.pct(0.99));
        if bad_ideal > 0 {
            ok = false;
        }
    };
    run("gate/up", &gu, get_bf16("x"), get_bf16("gateup_ref"), get_f32("gateup_exact_f32"));
    run("down", &dn, get_bf16("m"), get_bf16("down_ref"), get_f32("down_exact_f32"));
    assert!(ok, "layer-0 GEMM outside tolerance vs the exact dequant-matmul");
}
