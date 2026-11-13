//! GPU tests for the GatedDeltaNet kernels (`sysone::kernels::gdn`) against independent CPU f64 references written
//! from transformers 5.19.0 `modeling_qwen3_5.py`:
//! - `causal_conv1d_fn` (torch fallback: bf16 conv with fp32 accumulation, SiLU, bf16 out);
//! - the `Qwen3_5GatedDeltaNet.forward` gate math and `l2norm`;
//! - `torch_recurrent_gated_delta_rule`, step by step, as the oracle for the *chunked* kernel;
//! - `Qwen3_5RMSNormGated`.
//!
//! Each test prints the measured error distribution and the tolerance. Run with
//! `cargo test --release --test kernels_gdn -- --nocapture --test-threads=1`.

use cutile::half::bf16;
use cutile::prelude::*;
use cutile::cuda_core::Stream;
use sysone::contract::{DecisionBatch, Segment};
use sysone::kernels::gdn::*;

const PROJ_COLS: usize = CONV_DIM + VALUE_DIM + 2 * NUM_V_HEADS;
const EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------------------------------------------
// Helpers

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn on_test_thread(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(64 << 20).spawn(f).unwrap().join().unwrap();
}

/// Round as the reference does: compute in f32, store bf16 (round to nearest even).
fn r16(x: f64) -> f64 {
    bf16::from_f32(x as f32).to_f64()
}

/// One bf16 ulp at the magnitude of `x`.
fn ulp16(x: f64) -> f64 {
    let a = x.abs().max(1e-30);
    2f64.powi(a.log2().floor() as i32 - 7)
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn to_f64<T: Copy + Into<f64>>(v: &[T]) -> Vec<f64> {
    v.iter().map(|&x| x.into()).collect()
}

fn bf16_to_f64(v: &[bf16]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

fn upload<T: DType + Clone + Send + Sync + 'static>(v: Vec<T>, shape: &[usize], s: &Arc<Stream>) -> Tensor<T> {
    api::copy_host_vec_to_device(&Arc::new(v)).reshape(shape).sync_on(s).unwrap()
}

fn download<T: DType + Clone + Send + Sync + 'static>(t: &Tensor<T>, s: &Arc<Stream>) -> Vec<T> {
    t.dup().to_host_vec().sync_on(s).unwrap()
}

/// Error distribution of `got` vs `want` with a per-element tolerance; prints a one-line report and returns the
/// number of elements over tolerance.
fn report(name: &str, got: &[f64], want: &[f64], tol_desc: &str, tol: impl Fn(usize, f64) -> f64) -> usize {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut abs: Vec<f64> = Vec::with_capacity(got.len());
    let (mut max_ulp, mut over, mut exact, mut worst) = (0f64, 0usize, 0usize, (0usize, 0f64, 0f64));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{name}: non-finite output at {i}: {g} (want {w})");
        let d = (g - w).abs();
        abs.push(d);
        if d == 0.0 {
            exact += 1;
        }
        max_ulp = max_ulp.max(d / ulp16(w));
        if d > tol(i, w) {
            if over == 0 || d - tol(i, w) > worst.1 - tol(worst.0, worst.2) {
                worst = (i, g, w);
            }
            over += 1;
        }
    }
    let mut sorted = abs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
    println!(
        "{name:<34} n={:<9} exact={:>6.2}% abs p50={:.2e} p99={:.2e} max={:.2e} (ref rms {:.2e}) max={:.2} bf16-ulp | tol {tol_desc}: {over} over{}",
        got.len(),
        100.0 * exact as f64 / got.len() as f64,
        p(0.5),
        p(0.99),
        p(1.0),
        rms,
        max_ulp,
        if over > 0 { format!(" (worst at {}: got {:e} want {:e})", worst.0, worst.1, worst.2) } else { String::new() },
    );
    over
}

// ---------------------------------------------------------------------------------------------------------------
// CPU f64 references (from modeling_qwen3_5.py)

struct Weights {
    conv_w: Vec<f64>, // [8192, 4]
    a_log: Vec<f64>,
    dt_bias: Vec<f64>,
    norm_w: Vec<f64>,
}

/// The causal history of segment `s`: its parent's rows (suffix) then its own, as packed-stream row indices.
fn history(segs: &[Segment], s: usize) -> (Vec<usize>, usize) {
    let seg = segs[s];
    let mut rows = Vec::new();
    if !seg.is_prefix() {
        let p = segs[seg.parent as usize];
        rows.extend(p.start as usize..p.end() as usize);
    }
    let own = rows.len();
    rows.extend(seg.start as usize..seg.end() as usize);
    (rows, own)
}

/// `causal_conv1d_fn`: `F.conv1d(x, w, padding=3)[..., :L]` on the branch's whole causal sequence, bf16 conv
/// output, then SiLU rounded to bf16. x: packed `[T, PROJ_COLS]`-strided bf16 values, qkv at columns 0..8192.
/// Also returns the unrounded conv sum per element (for the rounding-flip tolerance).
fn conv_ref(x: &[f64], stride: usize, w: &Weights, segs: &[Segment], t: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0f64; t * CONV_DIM];
    let mut pre = vec![0f64; t * CONV_DIM];
    for s in 0..segs.len() {
        let (rows, own) = history(segs, s);
        for i in own..rows.len() {
            for c in 0..CONV_DIM {
                let mut acc = 0f64;
                for j in 0..CONV_KERNEL {
                    let src = i as isize - 3 + j as isize;
                    if src >= 0 {
                        acc += w.conv_w[c * 4 + j] * x[rows[src as usize] * stride + c];
                    }
                }
                out[rows[i] * CONV_DIM + c] = r16(silu(r16(acc)));
                pre[rows[i] * CONV_DIM + c] = acc;
            }
        }
    }
    (out, pre)
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

struct Prep {
    q: Vec<f64>,
    k: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
}

/// Split, gates, repeat-free l2norm (`l2norm`: x * rsqrt(sum x^2 + 1e-6)) and the 128^-0.5 query scale.
fn prep_ref(conv: &[f64], a: &[f64], b: &[f64], ab_stride: usize, w: &Weights, t: usize) -> Prep {
    let mut p = Prep { q: vec![0.0; t * KEY_DIM], k: vec![0.0; t * KEY_DIM], g: vec![0.0; t * 32], beta: vec![0.0; t * 32] };
    for r in 0..t {
        for hk in 0..NUM_K_HEADS {
            for (dst, off) in [(&mut p.q, 0usize), (&mut p.k, KEY_DIM)] {
                let x = &conv[r * CONV_DIM + off + hk * 128..][..128];
                let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
                let scale = if off == 0 { 1.0 / 128f64.sqrt() } else { 1.0 };
                for d in 0..128 {
                    dst[r * KEY_DIM + hk * 128 + d] = x[d] * inv * scale;
                }
            }
        }
        for h in 0..32 {
            let av = a[r * ab_stride + h];
            let bv = b[r * ab_stride + h];
            p.g[r * 32 + h] = -w.a_log[h].exp() * softplus(av + w.dt_bias[h]);
            p.beta[r * 32 + h] = r16(1.0 / (1.0 + (-bv).exp()));
        }
    }
    p
}

/// `torch_recurrent_gated_delta_rule`, token by token, per branch: prefixes from a zero state, suffixes from
/// their prefix's final state. q is already scaled (as `gdn_prep` emits it). Returns the f64 output `[T, 4096]`
/// (before the bf16 cast) and each prefix's final state `[32, 128, 128]`.
fn scan_ref(p: &Prep, v: &[f64], v_stride: usize, segs: &[Segment], t: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let prefixes: Vec<usize> = (0..segs.len()).filter(|&i| segs[i].is_prefix()).collect();
    let per_head: Vec<(Vec<f64>, Vec<Vec<f64>>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..NUM_V_HEADS)
            .map(|h| {
                let prefixes = &prefixes;
                sc.spawn(move || {
                    let hk = h / 2;
                    let mut out = vec![0f64; t * 128];
                    let mut finals: Vec<Vec<f64>> = vec![Vec::new(); segs.len()];
                    let run = |seg: &Segment, s: &mut Vec<f64>, out: &mut Vec<f64>| {
                        for r in seg.start as usize..seg.end() as usize {
                            let decay = p.g[r * 32 + h].exp();
                            let beta = p.beta[r * 32 + h];
                            let k = &p.k[r * KEY_DIM + hk * 128..][..128];
                            let q = &p.q[r * KEY_DIM + hk * 128..][..128];
                            let vv = &v[r * v_stride + h * 128..][..128];
                            s.iter_mut().for_each(|x| *x *= decay);
                            let mut delta = [0f64; 128];
                            for j in 0..128 {
                                let mem: f64 = (0..128).map(|i| s[i * 128 + j] * k[i]).sum();
                                delta[j] = (vv[j] - mem) * beta;
                            }
                            for i in 0..128 {
                                for j in 0..128 {
                                    s[i * 128 + j] += k[i] * delta[j];
                                }
                            }
                            for j in 0..128 {
                                out[r * 128 + j] = (0..128).map(|i| s[i * 128 + j] * q[i]).sum();
                            }
                        }
                    };
                    for &pi in prefixes {
                        let mut s = vec![0f64; 128 * 128];
                        run(&segs[pi], &mut s, &mut out);
                        finals[pi] = s;
                    }
                    for seg in segs.iter().filter(|s| !s.is_prefix()) {
                        let mut s = finals[seg.parent as usize].clone();
                        run(seg, &mut s, &mut out);
                    }
                    (out, prefixes.iter().map(|&pi| finals[pi].clone()).collect::<Vec<_>>())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = vec![0f64; t * VALUE_DIM];
    let mut states = vec![vec![0f64; STATE_ELEMS]; prefixes.len()];
    for (h, (o, fin)) in per_head.into_iter().enumerate() {
        for r in 0..t {
            out[r * VALUE_DIM + h * 128..][..128].copy_from_slice(&o[r * 128..][..128]);
        }
        for (slot, st) in fin.into_iter().enumerate() {
            states[slot][h * 128 * 128..][..128 * 128].copy_from_slice(&st);
        }
    }
    (out, states)
}

/// `Qwen3_5RMSNormGated` per 128-wide head, x and z given as bf16 values.
fn norm_ref(x: &[f64], z: &[f64], z_stride: usize, w: &[f64], t: usize) -> Vec<f64> {
    let mut out = vec![0f64; t * VALUE_DIM];
    for r in 0..t {
        for h in 0..NUM_V_HEADS {
            let xs = &x[r * VALUE_DIM + h * 128..][..128];
            let var = xs.iter().map(|v| v * v).sum::<f64>() / 128.0;
            let inv = 1.0 / (var + EPS as f64).sqrt();
            for d in 0..128 {
                let normed = r16(xs[d] * inv);
                let weighted = r16(w[d] * normed);
                out[r * VALUE_DIM + h * 128 + d] = r16(weighted * silu(z[r * z_stride + h * 128 + d]));
            }
        }
    }
    out
}

/// rms of `x` (`[T, 32 * 128]`) over each (segment, head), broadcast back to `[T * 32]` (row, head) entries.
fn segment_head_rms(x: &[f64], segs: &[Segment], t: usize) -> Vec<f64> {
    let mut out = vec![0f64; t * NUM_V_HEADS];
    for seg in segs {
        let rows = seg.start as usize..seg.end() as usize;
        for h in 0..NUM_V_HEADS {
            let ss: f64 = rows.clone().map(|r| x[r * VALUE_DIM + h * 128..][..128].iter().map(|v| v * v).sum::<f64>()).sum();
            let rms = (ss / (128 * rows.len()) as f64).sqrt();
            rows.clone().for_each(|r| out[r * NUM_V_HEADS + h] = rms);
        }
    }
    out
}

/// Write a run's scan inputs and outputs as raw little-endian f64 files plus the segment table, for offline
/// calibration against torch (`GDN_DUMP_DIR=...`).
fn dump(dir: &str, run: &Run, segs: &[Segment]) {
    std::fs::create_dir_all(dir).unwrap();
    let put = |name: &str, v: &[f64]| {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/{name}.f64"), bytes).unwrap();
    };
    for (name, v) in [("conv", &run.conv), ("q", &run.q), ("k", &run.k), ("g", &run.g), ("beta", &run.beta), ("core", &run.core), ("state", &run.state)] {
        put(name, v);
    }
    let table: Vec<String> = segs.iter().map(|s| format!("{} {} {}", s.start, s.len, s.parent)).collect();
    std::fs::write(format!("{dir}/segments.txt"), table.join("\n")).unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// Fixtures

fn weights(rng: &mut Rng) -> Weights {
    // Ranges seen in elephant-1's checkpoint: A_log in [-4.7, 3.7], dt_bias in [-8.7, 18.5], conv std 0.03..0.08
    // with tails to ±1.2, norm weight in [0.1, 1.3].
    Weights {
        conv_w: (0..CONV_DIM * 4).map(|_| r16(rng.normal() * 0.25)).collect(),
        a_log: (0..32).map(|_| r16(-4.7 + 8.4 * rng.uniform())).collect(),
        dt_bias: (0..32).map(|_| r16(-8.7 + 27.2 * rng.uniform())).collect(),
        norm_w: (0..128).map(|_| r16(0.1 + 1.2 * rng.uniform())).collect(),
    }
}

/// The fused projection output `[T, 12352]` (qkv | z | a | b), bf16.
fn projection(t: usize, rng: &mut Rng) -> Vec<f64> {
    (0..t * PROJ_COLS)
        .map(|i| {
            let c = i % PROJ_COLS;
            let sd = if c >= CONV_DIM + VALUE_DIM { 2.5 } else { 1.0 };
            r16(rng.normal() * sd)
        })
        .collect()
}

fn to_bf16(v: &[f64]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f64(x)).collect()
}

struct GpuWeights {
    conv_w: Tensor<bf16>,
    a_log: Tensor<bf16>,
    dt_bias: Tensor<bf16>,
    norm_w: Tensor<bf16>,
}

impl GpuWeights {
    fn new(w: &Weights, s: &Arc<Stream>) -> Self {
        Self {
            conv_w: upload(to_bf16(&w.conv_w), &[CONV_DIM, 1, CONV_KERNEL], s),
            a_log: upload(to_bf16(&w.a_log), &[32], s),
            dt_bias: upload(to_bf16(&w.dt_bias), &[32], s),
            norm_w: upload(to_bf16(&w.norm_w), &[128], s),
        }
    }
    fn view(&self) -> GdnWeights<'_> {
        GdnWeights { conv_w: &self.conv_w, a_log: &self.a_log, dt_bias: &self.dt_bias, norm_w: &self.norm_w }
    }
}

/// Everything one GPU run of the mixer produces.
struct Run {
    conv: Vec<f64>,
    q: Vec<f64>,
    k: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    core: Vec<f64>,
    state: Vec<f64>,
    out: Vec<f64>,
}

/// Run the whole mixer on `proj` with the given plan capacities and download every intermediate.
fn run_gpu(proj: &[f64], segs: &[Segment], w: &GpuWeights, caps: Option<(usize, usize)>, s: &Arc<Stream>) -> Run {
    run_gpu_with(proj, segs, w, caps, &GdnOptions::default(), s)
}

/// [`run_gpu`] with explicit scan options.
fn run_gpu_with(
    proj: &[f64],
    segs: &[Segment],
    w: &GpuWeights,
    caps: Option<(usize, usize)>,
    opts: &GdnOptions,
    s: &Arc<Stream>,
) -> Run {
    let t = proj.len() / PROJ_COLS;
    let plan = match caps {
        None => GdnPlan::new(segs, t, s).unwrap(),
        Some((mc, ms)) => GdnPlan::with_capacity(segs, t, mc, ms, s).unwrap(),
    };
    let proj_t = upload(to_bf16(proj), &[t, PROJ_COLS], s);
    let scratch = GdnScratch::new(t, plan.num_prefixes, s).unwrap();
    let out = api::zeros::<bf16>(&[t, VALUE_DIM]).sync_on(s).unwrap();
    let ops = unsafe {
        gdn_mixer_with(&plan, DView::of(&proj_t).unwrap(), &w.view(), EPS, &scratch, DView::of(&out).unwrap(), opts).unwrap()
    };
    ops.sync_on(s).unwrap();
    Run {
        conv: bf16_to_f64(&download(&scratch.conv, s)),
        q: to_f64(&download(&scratch.q, s)),
        k: to_f64(&download(&scratch.k, s)),
        g: to_f64(&download(&scratch.g, s)),
        beta: to_f64(&download(&scratch.beta, s)),
        core: bf16_to_f64(&download(&scratch.core, s)),
        state: to_f64(&download(&scratch.state, s)),
        out: bf16_to_f64(&download(&out, s)),
    }
}

/// Requests covering segment lengths 1, 2, 3, 63, 64, 65, 200, 1000+ (prefix and suffix), several requests and
/// several suffixes per prefix, and prefixes shorter than the conv window.
fn tree_batch() -> DecisionBatch {
    let mut b = DecisionBatch::default();
    let ids = |n: usize| vec![7u32; n];
    let (s1, s3, s63, s64, s65, s200) = (ids(1), ids(3), ids(63), ids(64), ids(65), ids(200));
    b.push_request(&ids(1037), &[(&s1, 2), (&s3, 2), (&s63, 2), (&s64, 2), (&s65, 2), (&s200, 2)]).unwrap();
    b.push_request(&ids(1), &[(&s3, 2), (&s1, 2)]).unwrap();
    b.push_request(&ids(64), &[(&s65, 2)]).unwrap();
    b.push_request(&ids(2), &[(&s1, 2), (&ids(2), 2)]).unwrap();
    b.push_request(&ids(200), &[(&s64, 2), (&ids(1100), 2)]).unwrap();
    b.push_request(&ids(63), &[(&s63, 2)]).unwrap();
    b.validate().unwrap();
    b
}

// ---------------------------------------------------------------------------------------------------------------
// Tests

/// Each kernel of the mixer against its f64 reference, fed with the GPU's own inputs to that stage, on a packed
/// tree batch; plus the prefix final states and the whole chain from the projection output.
#[test]
fn gdn_kernels_match_f64_references_on_a_tree_batch() {
    on_test_thread(|| {
        let ctx = Context::new(0).unwrap();
        let s = ctx.new_stream().unwrap();
        let mut rng = Rng(0x2545f4914f6cdd1d);
        let batch = tree_batch();
        let segs = &batch.segments;
        let t = batch.num_tokens();
        let w = weights(&mut rng);
        let gw = GpuWeights::new(&w, &s);
        let proj = projection(t, &mut rng);
        println!("tree batch: {t} tokens, {} segments", segs.len());
        let run = run_gpu(&proj, segs, &gw, None, &s);
        let mut fails = Vec::new();
        let mut check = |name: &str, over: usize| {
            if over > 0 {
                fails.push(format!("{name}: {over} elements over tolerance"));
            }
        };

        // 1. conv: rounding points match, so only f32-vs-f64 rounding flips are allowed: one at the bf16 conv
        //    output (an ulp of the conv sum, through SiLU' <= 1.1) and one at the bf16 SiLU output.
        let (conv, conv_pre) = conv_ref(&proj, PROJ_COLS, &w, segs, t);
        check(
            "conv",
            report("tree_conv1d_silu", &run.conv, &conv, "1 ulp(out) + 1.1 ulp(conv sum)", |i, r| {
                ulp16(r) + 1.1 * ulp16(conv_pre[i])
            }),
        );

        // 2. prep from the GPU conv output: q, k and g at fp32 level; beta (bf16 sigmoid) within 1 bf16 ulp.
        let a: Vec<f64> = (0..t).flat_map(|r| proj[r * PROJ_COLS + CONV_DIM + VALUE_DIM..][..32].to_vec()).collect();
        let b: Vec<f64> = (0..t).flat_map(|r| proj[r * PROJ_COLS + CONV_DIM + VALUE_DIM + 32..][..32].to_vec()).collect();
        let prep = prep_ref(&run.conv, &a, &b, 32, &w, t);
        check("q", report("gdn_prep q", &run.q, &prep.q, "1e-6 rel + 1e-9", |_, r| 1e-6 * r.abs() + 1e-9));
        check("k", report("gdn_prep k", &run.k, &prep.k, "1e-6 rel + 1e-8", |_, r| 1e-6 * r.abs() + 1e-8));
        check("g", report("gdn_prep g", &run.g, &prep.g, "1e-5 rel + 1e-12", |_, r| 1e-5 * r.abs() + 1e-12));
        check("beta", report("gdn_prep beta", &run.beta, &prep.beta, "1 bf16 ulp", |_, r| ulp16(r)));

        if let Ok(dir) = std::env::var("GDN_DUMP_DIR") {
            dump(&dir, &run, segs);
        }
        // 3. chunked scan vs the recurrent oracle, from the GPU's q, k, g, beta and v.
        let gpu_prep = Prep { q: run.q.clone(), k: run.k.clone(), g: run.g.clone(), beta: run.beta.clone() };
        let v: Vec<f64> = (0..t).flat_map(|r| run.conv[r * CONV_DIM + 2 * KEY_DIM..][..VALUE_DIM].to_vec()).collect();
        let (core, states) = scan_ref(&gpu_prep, &v, VALUE_DIM, segs, t);
        // Output scale per (segment, head): the f32 error is relative to the magnitude of the sums, not of each
        // (possibly cancelled) output element. Envelope: transformers' own f32 `torch_chunk_gated_delta_rule`
        // (GPU, TF32 off) on these inputs deviates from the f64 recurrence by up to 2.5e-4 x (segment, head) rms
        // beyond one bf16 rounding flip (sysone: 1.7e-4); the tolerance is twice the reference's envelope.
        let seg_head_rms = segment_head_rms(&core, segs, t);
        let core_b: Vec<f64> = core.iter().map(|&x| r16(x)).collect();
        check(
            "scan",
            report("gdn_chunk_scan out (bf16)", &run.core, &core_b, "1 bf16 ulp + 5e-4 * seg-head rms", |i, r| {
                ulp16(r) + 5e-4 * seg_head_rms[i / 128]
            }),
        );
        // Prefix states (f32) per prefix, relative to that prefix's state rms. Envelope: transformers' own f32
        // `torch_chunk_gated_delta_rule` (GPU, TF32 off) vs the f64 recurrence on these inputs reaches 1.1e-4 x rms
        // (measured with the GDN_DUMP_DIR dump); the tolerance is 3e-4 x rms.
        for (slot, st) in states.iter().enumerate() {
            let rms = (st.iter().map(|x| x * x).sum::<f64>() / st.len() as f64).sqrt();
            let got = &run.state[slot * STATE_ELEMS..][..STATE_ELEMS];
            let name = format!("prefix {slot} final state (f32)");
            check(&name.clone(), report(&name, got, st, "3e-4 * state rms", |_, _| 3e-4 * rms));
        }

        // 4. gated RMSNorm from the GPU scan output and z.
        let z: Vec<f64> = (0..t).flat_map(|r| proj[r * PROJ_COLS + CONV_DIM..][..VALUE_DIM].to_vec()).collect();
        let normed = norm_ref(&run.core, &z, VALUE_DIM, &w.norm_w, t);
        check("norm", report("gated_rmsnorm", &run.out, &normed, "2 bf16 ulp", |_, r| 2.0 * ulp16(r)));

        // 5. The whole chain from the projection output in f64 (bf16 rounding at the reference's points). Rounding
        //    flips upstream propagate, so the tolerance is wider; it is set before measuring, not tuned to it.
        let prep64 = prep_ref(&conv, &a, &b, 32, &w, t);
        let v64: Vec<f64> = (0..t).flat_map(|r| conv[r * CONV_DIM + 2 * KEY_DIM..][..VALUE_DIM].to_vec()).collect();
        let (core64, _) = scan_ref(&prep64, &v64, VALUE_DIM, segs, t);
        let core64: Vec<f64> = core64.iter().map(|&x| r16(x)).collect();
        let chain = norm_ref(&core64, &z, VALUE_DIM, &w.norm_w, t);
        let out_rms = (chain.iter().map(|x| x * x).sum::<f64>() / chain.len() as f64).sqrt();
        check(
            "chain",
            report("mixer chain (proj -> norm)", &run.out, &chain, "4 bf16 ulp + 1e-2 * out rms", |_, r| {
                4.0 * ulp16(r) + 1e-2 * out_rms
            }),
        );
        assert!(fails.is_empty(), "failures: {fails:#?}");
    });
}

/// Each branch of the tree batch, run alone as one causal sequence (prefix + suffix in a single prefix segment,
/// so the conv and the chunking see one contiguous stream), against the tree-batched run's rows.
#[test]
fn tree_batched_matches_each_branch_alone() {
    on_test_thread(|| {
        let ctx = Context::new(0).unwrap();
        let s = ctx.new_stream().unwrap();
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let batch = tree_batch();
        let segs = &batch.segments;
        let t = batch.num_tokens();
        let w = weights(&mut rng);
        let gw = GpuWeights::new(&w, &s);
        let proj = projection(t, &mut rng);
        // Spare capacity in the plan: unused chunk and segment entries must be no-ops.
        let caps = (GdnTables::chunk_capacity(t, segs.len()) + 5, segs.len() + 3);
        let tree = run_gpu(&proj, segs, &gw, Some(caps), &s);
        let (mut got_core, mut want_core, mut got_out, mut want_out) = (vec![], vec![], vec![], vec![]);
        let mut scale = Vec::new(); // (suffix, head) rms of the alone run's scan output, per (row, head)
        for (i, seg) in segs.iter().enumerate().filter(|(_, s)| !s.is_prefix()) {
            let (rows, own) = history(segs, i);
            let alone_proj: Vec<f64> = rows.iter().flat_map(|&r| proj[r * PROJ_COLS..][..PROJ_COLS].to_vec()).collect();
            let alone_segs = [Segment { start: 0, len: rows.len() as u32, parent: -1, pos0: 0 }];
            let alone = run_gpu(&alone_proj, &alone_segs, &gw, None, &s);
            assert_eq!(rows.len() - own, seg.len as usize);
            let suffix_only = [Segment { start: own as u32, len: seg.len, parent: -1, pos0: 0 }];
            let shr = segment_head_rms(&alone.core, &suffix_only, rows.len());
            scale.extend(rows.iter().enumerate().skip(own).flat_map(|(l, _)| shr[l * NUM_V_HEADS..][..NUM_V_HEADS].to_vec()));
            for (local, &r) in rows.iter().enumerate().skip(own) {
                got_core.extend_from_slice(&tree.core[r * VALUE_DIM..][..VALUE_DIM]);
                want_core.extend_from_slice(&alone.core[local * VALUE_DIM..][..VALUE_DIM]);
                got_out.extend_from_slice(&tree.out[r * VALUE_DIM..][..VALUE_DIM]);
                want_out.extend_from_slice(&alone.out[local * VALUE_DIM..][..VALUE_DIM]);
            }
            // The conv output of the suffix rows must be bit-identical (same taps, same arithmetic).
            let conv_diff = rows
                .iter()
                .enumerate()
                .skip(own)
                .flat_map(|(l, &r)| (0..CONV_DIM).map(move |c| (l, r, c)))
                .filter(|&(l, r, c)| tree.conv[r * CONV_DIM + c] != alone.conv[l * CONV_DIM + c])
                .count();
            assert_eq!(conv_diff, 0, "suffix segment {i}: conv differs from the branch run alone");
        }
        // Only the chunk alignment differs (the alone run chunks across the prefix/suffix boundary), so the scan
        // outputs differ by fp32 noise: the same envelope as the scan test (twice transformers' own f32 chunked
        // noise). Flips there propagate through the gated norm's three bf16 roundings, so the mixer output gets
        // the chain test's tolerance.
        let a = report("suffix scan out: tree vs alone", &got_core, &want_core, "1 bf16 ulp + 5e-4 * seg-head rms", |i, r| {
            ulp16(r) + 5e-4 * scale[i / 128]
        });
        let out_rms = (want_out.iter().map(|x| x * x).sum::<f64>() / want_out.len() as f64).sqrt();
        let b = report("suffix mixer out: tree vs alone", &got_out, &want_out, "4 bf16 ulp + 1e-2 * out rms", |_, r| {
            4.0 * ulp16(r) + 1e-2 * out_rms
        });
        assert_eq!(a + b, 0, "tree-batched suffix rows differ from the branches run alone");
    });
}

/// Replaying a plan through `update` with a different tree (same capacities) gives the same result as a fresh
/// plan; segments of one request are independent of the others in the batch.
#[test]
fn plan_update_and_batch_invariance() {
    on_test_thread(|| {
        let ctx = Context::new(0).unwrap();
        let s = ctx.new_stream().unwrap();
        let mut rng = Rng(0xdeadbeefcafef00d);
        let w = weights(&mut rng);
        let gw = GpuWeights::new(&w, &s);
        let mut big = DecisionBatch::default();
        big.push_request(&[1; 300], &[(&[2; 40], 2), (&[3; 129], 2)]).unwrap();
        big.push_request(&[4; 70], &[(&[5; 5], 2)]).unwrap();
        let mut small = DecisionBatch::default();
        small.push_request(&[4; 70], &[(&[5; 5], 2)]).unwrap();
        let t = big.num_tokens();
        let proj = projection(t, &mut rng);
        let off = 300 + 40 + 129;
        let small_proj = proj[off * PROJ_COLS..].to_vec();
        let alone = run_gpu(&small_proj, &small.segments, &gw, None, &s);

        // Plan sized for `big`, first used for `small` padded to the same T, then updated to `big`.
        let caps = (GdnTables::chunk_capacity(t, 8), 8);
        let mut plan = GdnPlan::with_capacity(&small.segments, t, caps.0, caps.1, &s).unwrap();
        let mut padded = small_proj.clone();
        padded.resize(t * PROJ_COLS, 0.0);
        let scratch = GdnScratch::new(t, 4, &s).unwrap();
        let out = api::zeros::<bf16>(&[t, VALUE_DIM]).sync_on(&s).unwrap();
        let wv = gw.view();
        for (step, (p, b)) in [(&padded, &small), (&proj, &big)].into_iter().enumerate() {
            if step == 1 {
                plan.update(&b.segments, &s).unwrap();
            }
            let pt = upload(to_bf16(p), &[t, PROJ_COLS], &s);
            unsafe { gdn_mixer(&plan, DView::of(&pt).unwrap(), &wv, EPS, &scratch, DView::of(&out).unwrap()).unwrap() }
                .sync_on(&s)
                .unwrap();
            let got = bf16_to_f64(&download(&out, &s));
            let base = if step == 0 { 0 } else { off };
            let n = small.num_tokens() * VALUE_DIM;
            let same = got[base * VALUE_DIM..][..n] == alone.out[..n];
            println!("plan step {step}: request rows bit-identical to a fresh single-request plan: {same}");
            assert!(same, "step {step}: request output depends on the rest of the batch or on plan reuse");
        }
    });
}

/// The scan's tuning options (block sizes, warps, inverse blocking) must not change its accuracy: each variant
/// is held to the default's tolerances against the f64 recurrence. The precision variants (DESIGN §1 "Precision
/// knobs") are measured against the same oracle and reported; only the default precision must meet the
/// fp32-grade tolerance, the faster ones are judged by the end-to-end gate.
#[test]
fn scan_options_and_precision_variants_vs_f64() {
    on_test_thread(|| {
        let ctx = Context::new(0).unwrap();
        let s = ctx.new_stream().unwrap();
        let mut rng = Rng(0x2545f4914f6cdd1d);
        let batch = tree_batch();
        let segs = &batch.segments;
        let t = batch.num_tokens();
        let w = weights(&mut rng);
        let gw = GpuWeights::new(&w, &s);
        let proj = projection(t, &mut rng);
        let base = run_gpu(&proj, segs, &gw, None, &s);
        // The oracle from the GPU's own scan inputs (conv and prep do not depend on the options).
        let prep = Prep { q: base.q.clone(), k: base.k.clone(), g: base.g.clone(), beta: base.beta.clone() };
        let v: Vec<f64> = (0..t).flat_map(|r| base.conv[r * CONV_DIM + 2 * KEY_DIM..][..VALUE_DIM].to_vec()).collect();
        let (core, states) = scan_ref(&prep, &v, VALUE_DIM, segs, t);
        let seg_head_rms = segment_head_rms(&core, segs, t);
        let core_b: Vec<f64> = core.iter().map(|&x| r16(x)).collect();
        let d = GdnOptions::default();
        let variants: Vec<(&str, GdnOptions, bool)> = vec![
            ("default", d, true),
            ("prefix_bv 16, suffix_bv 128", GdnOptions { prefix_bv: 16, suffix_bv: 128, ..d }, true),
            ("prefix_bv 64, suffix_bv 32", GdnOptions { prefix_bv: 64, suffix_bv: 32, ..d }, true),
            ("tp_fsb 2 (doubling only)", GdnOptions { tp_fsb: 2, ..d }, true),
            ("tp_fsb 64 (substitution only)", GdnOptions { tp_fsb: 64, ..d }, true),
            ("4 / 8 / 4 warps", GdnOptions { prefix_warps: 4, suffix_warps: 8, tp_warps: 4, ..d }, true),
            ("precision Tf32", GdnOptions { precision: GdnPrecision::Tf32, ..d }, false),
            ("precision Bf16Operands", GdnOptions { precision: GdnPrecision::Bf16Operands, ..d }, false),
        ];
        let mut fails = Vec::new();
        for (name, opts, gated) in variants {
            let run = run_gpu_with(&proj, segs, &gw, None, &opts, &s);
            assert!(run.core.iter().all(|x| x.is_finite()), "{name}: non-finite scan output");
            // Error beyond one bf16 rounding flip, in units of the (segment, head) rms (the scan tolerance is 5e-4).
            let worst = run
                .core
                .iter()
                .zip(&core_b)
                .enumerate()
                .map(|(i, (g, r))| ((g - r).abs() - ulp16(*r)).max(0.0) / seg_head_rms[i / 128].max(1e-30))
                .fold(0f64, f64::max);
            let mut state_worst = 0f64;
            for (slot, st) in states.iter().enumerate() {
                let rms = (st.iter().map(|x| x * x).sum::<f64>() / st.len() as f64).sqrt();
                let got = &run.state[slot * STATE_ELEMS..][..STATE_ELEMS];
                let e = got.iter().zip(st).map(|(g, r)| (g - r).abs()).fold(0f64, f64::max) / rms;
                state_worst = state_worst.max(e);
            }
            let over = report(&format!("scan out [{name}]"), &run.core, &core_b, "1 bf16 ulp + 5e-4 * seg-head rms", |i, r| {
                ulp16(r) + 5e-4 * seg_head_rms[i / 128]
            });
            println!(
                "  {name}: max scan error {worst:.2e} x seg-head rms (tol 5e-4), max prefix-state error {state_worst:.2e} x state rms (tol 3e-4){}",
                if gated { "" } else { " [reported only]" }
            );
            if gated && (over > 0 || state_worst > 3e-4) {
                fails.push(format!("{name}: {over} scan elements over tolerance, state error {state_worst:.2e} x rms"));
            }
        }
        assert!(fails.is_empty(), "failures: {fails:#?}");
    });
}
