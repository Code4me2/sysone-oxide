//! GPU tests for the full-attention kernels (W3b): `qk_norm_rope` and `seg_attn`, against CPU f64
//! references written from transformers 5.19.0 `modeling_qwen3_5.py` (`Qwen3_5Attention.forward`,
//! `Qwen3_5RMSNorm`, `apply_rotary_pos_emb`, `Qwen3_5TextRotaryEmbedding`) and `masking_utils.py` (the
//! block-bidirectional mask of DESIGN.md §3: prefix rows see the whole prefix, suffix rows see the whole
//! prefix plus their whole own suffix).
//!
//! References compute in f64 and round to bf16 where the reference materialises a bf16 tensor.

use cutile::half::bf16;
use cutile::prelude::*;
use sysone::contract::{DecisionBatch, Segment};
use sysone::kernels::attention::{self as attn, AttnPlan, AttnTiles, SegAttnArgs};
use sysone::kernels::norm::RMS_EPS;

const HQ: usize = attn::NUM_Q_HEADS;
const HK: usize = attn::NUM_KV_HEADS;
const D: usize = attn::HEAD_DIM;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
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

fn rb(x: f64) -> f64 {
    bf16::from_f64(x).to_f64()
}

fn upload<T: DType>(v: Vec<T>, shape: &[usize], stream: &Arc<cuda_core::Stream>) -> Tensor<T> {
    api::copy_host_vec_to_device(&Arc::new(v)).reshape(shape).sync_on(stream).unwrap()
}

/// One CUDA context and stream shared by every test in this binary.
fn setup() -> Arc<cuda_core::Stream> {
    static STREAM: std::sync::OnceLock<Arc<cuda_core::Stream>> = std::sync::OnceLock::new();
    STREAM.get_or_init(|| Context::new(0).unwrap().new_stream().unwrap()).clone()
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

// ---------------------------------------------------------------------------------------------------------
// RoPE frequencies
// ---------------------------------------------------------------------------------------------------------

/// `1.0 / (10000000 ** (torch.arange(0, 64, 2, dtype=torch.float) / 64))` from torch 2.13.0 (CPU, fp32).
const TORCH_INV_FREQ_BITS: [u32; 32] = [
    0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177, 0x3c91ad39,
    0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02, 0x39a5cb60, 0x394860c1,
    0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4, 0x36bcb0c1, 0x36640cc6, 0x3609cf4a,
    0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44,
];

#[test]
fn rope_inv_freq_matches_torch_bits() {
    let ours = attn::rope_inv_freq();
    let bad: Vec<usize> = (0..32).filter(|&i| ours[i].to_bits() != TORCH_INV_FREQ_BITS[i]).collect();
    println!("rope_inv_freq: {}/32 bit-identical to torch (tolerance: all)", 32 - bad.len());
    assert!(bad.is_empty(), "inv_freq differs from torch at {bad:?}");
}

// ---------------------------------------------------------------------------------------------------------
// qk_norm_rope
// ---------------------------------------------------------------------------------------------------------

/// `Qwen3_5RMSNorm` (zero-centred) on one head, rounded to bf16.
fn ref_head_norm(x: &[bf16], w: &[bf16]) -> Vec<f64> {
    let ms = x.iter().map(|v| f(*v) * f(*v)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + 1e-6).sqrt();
    x.iter().zip(w).map(|(v, w)| rb(f(*v) * inv * (1.0 + f(*w)))).collect()
}

/// `apply_rotary_pos_emb` on one normed head at position `pos`. Returns `(exact, scale)` per dim: the value
/// before the final bf16 rounding, and `max(|n[i]|, |n[partner]|, |exact|)`, the magnitude its error is
/// measured at.
///
/// Error budget at that scale: the normed inputs are bf16 values the kernel computes in fp32 (as torch does)
/// and this reference in f64, so a value sitting on a rounding boundary can come out one ulp apart (the
/// zc_rmsnorm test measures ~0.01% such flips). One flipped input moves `x*cos` or `rot*sin` by up to one
/// ulp at the scale, and the three roundings add at most 1.5 ulp, so the bound is 2 ulps (rounding is
/// measured from the exact value, not from the rounded one).
fn ref_rope(n: &[f64], pos: i32, inv: &[f32; 32]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = n.iter().map(|v| (*v, v.abs())).collect();
    for i in 0..64 {
        let fi = i % 32;
        // freqs = position_ids.float() * inv_freq: an fp32 product; cos/sin in fp32, then `.to(bf16)`.
        let ang = (pos as f32 * inv[fi]) as f64;
        let (c, s) = (rb(ang.cos()), rb(ang.sin()));
        let rot = if i < 32 { -n[i + 32] } else { n[i - 32] };
        let (a, b) = (rb(n[i] * c), rb(rot * s));
        out[i] = (a + b, n[i].abs().max(rot.abs()).max((a + b).abs()));
    }
    out
}

#[test]
fn qk_norm_rope_matches_reference() {
    let stream = setup();
    let mut rng = Rng(0x7777_1111_3333_9999);
    // Positions of a real packed batch, then large ones up to the 65,536-token request limit.
    let mut b = DecisionBatch::default();
    b.push_request(&[1; 300], &[(&[2; 40], 2), (&[3; 7], 2)]).unwrap();
    b.push_request(&[1; 5], &[(&[2; 3], 2)]).unwrap();
    let mut pos = attn::positions(&b.segments);
    pos.extend((0..245).map(|_| (rng.next() % 65536) as i32));
    pos.extend([32767, 32768, 65535]);
    let t = pos.len();

    let mut qg = Vec::with_capacity(t * HQ * 2 * D);
    for r in 0..t {
        qg.extend(rng.bf16s(HQ * 2 * D, [0.05, 1.0, 30.0][r % 3]));
    }
    let kv = rng.bf16s(t * 2 * HK * D, 2.0);
    let qw = rng.bf16s(D, 0.5);
    let kw = rng.bf16s(D, 0.5);
    let inv = attn::rope_inv_freq();

    let qg_d = upload(qg.clone(), &[t, HQ, 2 * D], &stream);
    let kv_d = upload(kv.clone(), &[t, 2 * HK, D], &stream);
    let pos_d = upload(pos.clone(), &[t], &stream);
    let qw_d = upload(qw.clone(), &[D], &stream);
    let kw_d = upload(kw.clone(), &[D], &stream);
    let inv_d = upload(inv.to_vec(), &[32], &stream);
    let mut q = api::zeros::<bf16>(&[t, HQ, D]).sync_on(&stream).unwrap();
    let mut k = api::zeros::<bf16>(&[t, HK, D]).sync_on(&stream).unwrap();
    attn::qk_norm_rope(&mut q, &mut k, &qg_d, &kv_d, &pos_d, &qw_d, &kw_d, &inv_d, RMS_EPS)
        .unwrap()
        .sync_on(&stream)
        .unwrap();
    let gq = q.to_host_vec().sync_on(&stream).unwrap();
    let gk = k.to_host_vec().sync_on(&stream).unwrap();

    for (name, heads, got, src, stride, w) in
        [("q", HQ, &gq, &qg, 2 * D, &qw), ("k", HK, &gk, &kv, D, &kw)]
    {
        let mut errs = Vec::new();
        let mut mism = 0usize;
        let mut rot_mism = 0usize;
        for r in 0..t {
            for h in 0..heads {
                let base = (r * heads + h) * stride;
                // q heads are [query | gate] (stride 512); k heads are the first 4 of 8 (stride 256).
                let base = if name == "q" { base } else { (r * 2 * HK + h) * D };
                let n = ref_head_norm(&src[base..base + D], w);
                let want = ref_rope(&n, pos[r], &inv);
                for (i, &(exact, scale)) in want.iter().enumerate() {
                    let g = f(got[(r * heads + h) * D + i]);
                    let ulp = 2f64.powi(scale.max(1e-30).log2().floor() as i32 - 7);
                    errs.push((g - exact).abs() / ulp);
                    if g != rb(exact) {
                        mism += 1;
                        if i < 64 {
                            rot_mism += 1;
                        }
                    }
                }
            }
        }
        let n = errs.len();
        let e = sorted(errs);
        println!(
            "qk_norm_rope {name}: n={n} max={:.3} p99={:.3} (bf16 ulps at the operand scale) mismatched={:.4}% \
             ({rot_mism} in rotary dims) (tolerance: max <= 2, mismatched <= 1%)",
            e[n - 1],
            e[n * 99 / 100],
            100.0 * mism as f64 / n as f64
        );
        assert!(e[n - 1] <= 2.0, "qk_norm_rope {name}: max error {}", e[n - 1]);
        assert!(mism as f64 / n as f64 <= 0.01, "qk_norm_rope {name}: {mism} mismatched");
    }
}

// ---------------------------------------------------------------------------------------------------------
// seg_attn
// ---------------------------------------------------------------------------------------------------------

/// Per-token inputs of the attention kernel, generated per (request, segment, offset) so that the same
/// branch gets the same data in any batch.
struct TokenData {
    q: Vec<bf16>,  // [HQ, D]
    k: Vec<bf16>,  // [HK, D]
    v: Vec<bf16>,  // [HK, D]
    g: Vec<bf16>,  // [HQ, D]
}

fn token_data(seed: u64) -> TokenData {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    for _ in 0..4 {
        rng.next();
    }
    // Head-dependent query scale gives flat to peaky attention (scores up to ~30 after the 1/16 scale).
    let mut q = Vec::with_capacity(HQ * D);
    for h in 0..HQ {
        q.extend(rng.bf16s(D, [0.5, 2.0, 6.0, 12.0][h % 4]));
    }
    TokenData { q, k: rng.bf16s(HK * D, 1.0), v: rng.bf16s(HK * D, 1.0), g: rng.bf16s(HQ * D, 4.0) }
}

/// A request: prefix length and suffix lengths; `id` keys its token data.
struct Req {
    id: u64,
    prefix: usize,
    suffixes: Vec<usize>,
}

struct Packed {
    batch: DecisionBatch,
    data: Vec<TokenData>,
}

fn pack(reqs: &[(&Req, Vec<usize>)]) -> Packed {
    let mut batch = DecisionBatch::default();
    let mut data = Vec::new();
    for (r, which) in reqs {
        let pre = vec![0u32; r.prefix];
        data.extend((0..r.prefix).map(|i| token_data(r.id << 40 | i as u64)));
        let sufs: Vec<Vec<u32>> = which.iter().map(|&s| vec![0u32; r.suffixes[s]]).collect();
        for &s in which {
            data.extend((0..r.suffixes[s]).map(|i| token_data(r.id << 40 | (s as u64 + 1) << 24 | i as u64)));
        }
        let branches: Vec<(&[u32], u16)> = sufs.iter().map(|s| (s.as_slice(), 2u16)).collect();
        batch.push_request(&pre, &branches).unwrap();
    }
    batch.validate().unwrap();
    Packed { batch, data }
}

struct Device {
    q: Tensor<bf16>,
    k: Tensor<bf16>,
    kv: Tensor<bf16>,
    qg: Tensor<bf16>,
}

fn device_inputs(p: &Packed, stream: &Arc<cuda_core::Stream>) -> Device {
    let t = p.data.len();
    let (mut q, mut k, mut kv, mut qg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for d in &p.data {
        q.extend_from_slice(&d.q);
        k.extend_from_slice(&d.k);
        // kv rows are [k heads | v heads]; the kernel reads only v from here (k comes from `k`).
        kv.extend_from_slice(&d.k);
        kv.extend_from_slice(&d.v);
        for h in 0..HQ {
            // q_proj rows are per head [query | gate]; the kernel reads only the gate from here.
            qg.extend_from_slice(&d.q[h * D..(h + 1) * D]);
            qg.extend_from_slice(&d.g[h * D..(h + 1) * D]);
        }
    }
    Device {
        q: upload(q, &[t, HQ, D], stream),
        k: upload(k, &[t, HK, D], stream),
        kv: upload(kv, &[t, 2 * HK, D], stream),
        qg: upload(qg, &[t, HQ, 2 * D], stream),
    }
}

/// Run seg_attn on `plan`; `out` starts filled with `fill`.
fn run_attn(dev: &Device, plan: &AttnPlan, tiles: AttnTiles, fill: bf16, stream: &Arc<cuda_core::Stream>) -> Vec<bf16> {
    let t = dev.q.shape()[0] as usize;
    let out = upload(vec![fill; t * HQ * D], &[t, HQ, D], stream);
    let work = upload(plan.work.clone(), &[plan.work.len().max(1)], stream);
    let args = SegAttnArgs {
        out: out.device_pointer(),
        q: dev.q.device_pointer(),
        k: dev.k.device_pointer(),
        kv: dev.kv.device_pointer(),
        qg: dev.qg.device_pointer(),
        work: work.device_pointer(),
        num_items: plan.num_items(),
    };
    unsafe { attn::seg_attn(args, tiles) }.sync_on(stream).unwrap();
    out.to_host_vec().sync_on(stream).unwrap()
}

/// The keys token `t` attends to under the block-bidirectional mask.
fn key_ranges(segs: &[Segment], t: usize) -> Vec<std::ops::Range<usize>> {
    let s = segs.iter().find(|s| s.start as usize <= t && t < s.end() as usize).unwrap();
    let range = |s: &Segment| s.start as usize..s.end() as usize;
    if s.is_prefix() { vec![range(s)] } else { vec![range(&segs[s.parent as usize]), range(s)] }
}

/// `eager_attention_forward` (fp32 softmax, exact here) for one (token, head), then `* sigmoid(gate)` with
/// the reference's bf16 rounding points. Returns the exact value before the final rounding and the
/// rounded `o` (attention output) row.
fn ref_attn_row(p: &Packed, t: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let kvh = h / (HQ / HK);
    let qrow = &p.data[t].q[h * D..(h + 1) * D];
    let keys: Vec<usize> = key_ranges(&p.batch.segments, t).into_iter().flatten().collect();
    let scores: Vec<f64> = keys
        .iter()
        .map(|&j| {
            let krow = &p.data[j].k[kvh * D..(kvh + 1) * D];
            qrow.iter().zip(krow).map(|(a, b)| f(*a) * f(*b)).sum::<f64>() / 16.0
        })
        .collect();
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = e.iter().sum();
    let mut o = vec![0f64; D];
    for (&j, w) in keys.iter().zip(&e) {
        let vrow = &p.data[j].v[kvh * D..(kvh + 1) * D];
        for (acc, v) in o.iter_mut().zip(vrow) {
            *acc += w / z * f(*v);
        }
    }
    let grow = &p.data[t].g[h * D..(h + 1) * D];
    let o_b: Vec<f64> = o.iter().map(|v| rb(*v)).collect();
    let exact = o_b.iter().zip(grow).map(|(o, g)| o * rb(1.0 / (1.0 + (-f(*g)).exp()))).collect();
    (exact, o_b)
}

/// Error envelope of the reference itself on this distribution: torch 2.13 `scaled_dot_product_attention`
/// (bf16, block-bidirectional mask, head_dim 256 -> MATH backend on GB10; flash, mem-efficient and cuDNN
/// refuse it) then `* sigmoid(gate)`, against the same f64 computation and metric (a 1500-token prefix with
/// suffixes 1, 37, 200, 64, 3; script in the W3b report): max 2.297, p99 0.484 ulp, 0.05% mismatched.
/// Torch eager (bf16 P) measures max 29.2, p99 5.53, 61% mismatched. Gate per DESIGN.md §9.2: 1.5x the
/// sdpa envelope.
const ENVELOPE_MAX: f64 = 1.5 * 2.297;
const ENVELOPE_P99: f64 = 1.5 * 0.484;
const ENVELOPE_MISMATCH: f64 = 1.5 * 0.0005;

/// Compare kernel output against the reference for all rows, multithreaded. Asserts the envelope when
/// `gate` is set; otherwise only reports.
fn check_attn(p: &Packed, got: &[bf16], label: &str, gate: bool) {
    let t = p.data.len();
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(16);
    // Error unit: one bf16 ulp at the row's scale, max(|ref|, rms of the reference row).
    let results: Vec<(Vec<f64>, usize, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|w| {
                sc.spawn(move || {
                    let (mut errs, mut mism, mut max_abs) = (Vec::new(), 0usize, 0f64);
                    for r in (w..t).step_by(threads) {
                        for h in 0..HQ {
                            let (exact, _) = ref_attn_row(p, r, h);
                            let rms = (exact.iter().map(|v| v * v).sum::<f64>() / D as f64).sqrt();
                            for (i, want) in exact.iter().enumerate() {
                                let g = f(got[(r * HQ + h) * D + i]);
                                let scale = want.abs().max(rms).max(1e-30);
                                let ulp = 2f64.powi(scale.log2().floor() as i32 - 7);
                                let err = (g - want).abs();
                                max_abs = max_abs.max(err);
                                errs.push(if err.is_nan() { f64::INFINITY } else { err / ulp });
                                if g != rb(*want) {
                                    mism += 1;
                                }
                            }
                        }
                    }
                    (errs, mism, max_abs)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mism: usize = results.iter().map(|r| r.1).sum();
    let max_abs = results.iter().map(|r| r.2).fold(0.0, f64::max);
    let errs = sorted(results.into_iter().flat_map(|r| r.0).collect());
    let n = errs.len();
    println!(
        "seg_attn {label}: T={t} n={n} max={:.3} p99={:.3} p50={:.3} (bf16 ulps at row scale) max abs={max_abs:.3e} \
         mismatched={:.4}% (tolerance{}: max <= {ENVELOPE_MAX:.3}, p99 <= {ENVELOPE_P99:.3}, \
         mismatched <= {:.3}%)",
        errs[n - 1],
        errs[n * 99 / 100],
        errs[n / 2],
        100.0 * mism as f64 / n as f64,
        if gate { "" } else { " NOT ASSERTED" },
        100.0 * ENVELOPE_MISMATCH,
    );
    if gate {
        assert!(errs[n - 1] <= ENVELOPE_MAX, "seg_attn {label}: max error {} ulp", errs[n - 1]);
        assert!(errs[n * 99 / 100] <= ENVELOPE_P99, "seg_attn {label}: p99 error {} ulp", errs[n * 99 / 100]);
        assert!(mism as f64 / n as f64 <= ENVELOPE_MISMATCH, "seg_attn {label}: {mism}/{n} mismatched");
    }
}

fn requests() -> Vec<Req> {
    vec![
        Req { id: 1, prefix: 1500, suffixes: vec![1, 37, 200, 64, 3] },
        Req { id: 2, prefix: 1, suffixes: vec![1, 2] },
        Req { id: 3, prefix: 2048, suffixes: vec![2000] },
        Req { id: 4, prefix: 17, suffixes: vec![16, 15, 33, 1] },
        Req { id: 5, prefix: 64, suffixes: vec![65, 63] },
    ]
}

#[test]
fn seg_attn_matches_reference() {
    let stream = setup();
    let reqs = requests();
    let all: Vec<(&Req, Vec<usize>)> = reqs.iter().map(|r| (r, (0..r.suffixes.len()).collect())).collect();
    let p = pack(&all);
    let dev = device_inputs(&p, &stream);
    let d = AttnTiles::default();
    for (tiles, gate) in [
        (d, true),
        (AttnTiles { bm: 32, bn: 32, ..d }, true),
        (AttnTiles { p_split: 2, ..d }, false),
        (AttnTiles { p_split: 1, ..d }, false),
    ] {
        let plan = AttnPlan::new(&p.batch.segments, tiles.bm);
        let got = run_attn(&dev, &plan, tiles, bf16::NAN, &stream);
        check_attn(&p, &got, &format!("bm={} bn={} p_split={}", tiles.bm, tiles.bn, tiles.p_split), gate);
    }
}

#[test]
fn seg_attn_branch_alone_is_bit_identical() {
    let stream = setup();
    let reqs = requests();
    let tiles = AttnTiles::default();
    let all: Vec<(&Req, Vec<usize>)> = reqs.iter().map(|r| (r, (0..r.suffixes.len()).collect())).collect();
    let packed = pack(&all);
    let dev = device_inputs(&packed, &stream);
    let got_all = run_attn(&dev, &AttnPlan::new(&packed.batch.segments, tiles.bm), tiles, bf16::NAN, &stream);

    let mut checked = 0;
    for (ri, r) in reqs.iter().enumerate() {
        for s in 0..r.suffixes.len() {
            let alone = pack(&[(r, vec![s])]);
            let dev1 = device_inputs(&alone, &stream);
            let got1 = run_attn(&dev1, &AttnPlan::new(&alone.batch.segments, tiles.bm), tiles, bf16::NAN, &stream);
            // Locate the request's prefix and suffix s in the full batch.
            let pre_seg = packed.batch.segments.iter().filter(|x| x.is_prefix()).nth(ri).unwrap();
            let pidx = packed.batch.segments.iter().position(|x| x == pre_seg).unwrap() as i32;
            let suf_seg = packed.batch.segments.iter().filter(|x| x.parent == pidx).nth(s).unwrap();
            let row = |v: &[bf16], start: u32, len: u32| v[start as usize * HQ * D..(start + len) as usize * HQ * D].to_vec();
            let (a_pre, a_suf) = (alone.batch.segments[0], alone.batch.segments[1]);
            assert_eq!(row(&got_all, pre_seg.start, pre_seg.len), row(&got1, a_pre.start, a_pre.len), "request {ri} prefix");
            assert_eq!(
                row(&got_all, suf_seg.start, suf_seg.len),
                row(&got1, a_suf.start, a_suf.len),
                "request {ri} branch {s} differs alone vs packed"
            );
            checked += 1;
        }
    }
    println!("seg_attn branch invariance: {checked} branches bit-identical alone vs packed (tolerance: exact)");
}

#[test]
fn seg_attn_writes_only_its_rows() {
    let stream = setup();
    let r = Req { id: 9, prefix: 40, suffixes: vec![5, 3, 20] };
    let p = pack(&[(&r, vec![0, 1, 2])]);
    let dev = device_inputs(&p, &stream);
    let tiles = AttnTiles::default();
    let full = AttnPlan::new(&p.batch.segments, tiles.bm);
    // Only the items of the 5-row suffix (rows 40..45): a 16-row tile whose last 11 rows belong to others.
    let work: Vec<i32> = full.work.chunks(8).filter(|w| w[0] == 40).flatten().copied().collect();
    let sentinel = bf16::from_f32(-123.0);
    let got = run_attn(&dev, &AttnPlan { work }, tiles, sentinel, &stream);
    let t = p.data.len();
    for row in 0..t {
        let written = got[row * HQ * D..(row + 1) * HQ * D].iter().filter(|v| **v != sentinel).count();
        let expect = if (40..45).contains(&row) { HQ * D } else { 0 };
        assert_eq!(written, expect, "row {row}: {written} elements written");
    }
    println!("seg_attn store clipping: rows outside the tile's segment untouched (tolerance: exact)");
}
