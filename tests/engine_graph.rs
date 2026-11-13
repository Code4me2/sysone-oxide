//! StepGraph execution: embedding (cutile-kernels) + chained bf16 GEMMs (cuBLAS) over a planned arena,
//! eager and as T-bucket CUDA graphs replayed with fresh batch metadata, against a CPU f64 reference.
//! Also checks the double-buffered metadata upload byte for byte.

use cuda_core::Context;
use std::collections::HashMap;
use sysone::contract::DecisionBatch;
use sysone::engine::gemm::Blas;
use sysone::engine::graph::tensor::{DevTensor, GraphDType, bf16_bytes, bf16_from_bytes, f32_from_bytes};
use sysone::engine::graph::{
    BucketGraphs, BucketPlan, Dim, Executor, GraphOp, LayerWeight, NoKernels, StepGraph, TensorRef, WeightRef,
    WeightStore,
};
use sysone::engine::meta::{MetaShape, MetaUploader, PAD_PARENT};
use sysone::engine::weights::{Linear, LinearPart};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
    fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| half::bf16::from_f32(self.f() * scale).to_f32()).collect()
    }
}

struct Store {
    embed: DevTensor,
    linears: HashMap<WeightRef, Linear>,
}

impl WeightStore for Store {
    fn tensor(&self, w: WeightRef) -> anyhow::Result<&DevTensor> {
        match w {
            WeightRef::Embed => Ok(&self.embed),
            _ => anyhow::bail!("no {w:?}"),
        }
    }
    fn linear(&self, w: WeightRef) -> anyhow::Result<&Linear> {
        self.linears.get(&w).ok_or_else(|| anyhow::anyhow!("no {w:?}"))
    }
}

const V: usize = 500;
const D: usize = 1024;
const N1: usize = 384;
const C: usize = 256; // column slice of x1 fed to the second GEMM
const N2: usize = 128;
const N3: usize = 64;

fn round_bf16(x: f64) -> f32 {
    half::bf16::from_f64(x).to_f32()
}

/// CPU reference of the test graph: x0 = E[ids]; x1 = bf16(x0·W1ᵀ); x2 = bf16(x1[:, :C]·W2ᵀ); y = x2·W3ᵀ (f32).
/// Returns (x2, y) per token. Products in f64; intermediate rounding to bf16 as the graph does.
fn reference(ids: &[u32], e: &[f32], w1: &[f32], w2: &[f32], w3: &[f32]) -> (Vec<f32>, Vec<f64>, Vec<f64>) {
    let mut x2 = Vec::new();
    let mut mag2 = Vec::new();
    let mut y = Vec::new();
    for &id in ids {
        let x0 = &e[id as usize * D..(id as usize + 1) * D];
        let x1: Vec<f32> = (0..N1).map(|n| round_bf16((0..D).map(|k| x0[k] as f64 * w1[n * D + k] as f64).sum())).collect();
        let r2: Vec<f32> = (0..N2).map(|n| round_bf16((0..C).map(|k| x1[k] as f64 * w2[n * C + k] as f64).sum())).collect();
        mag2.extend((0..N2).map(|n| (0..C).map(|k| (x1[k] as f64 * w2[n * C + k] as f64).abs()).sum::<f64>()));
        y.extend((0..N3).map(|n| (0..N2).map(|k| r2[k] as f64 * w3[n * N2 + k] as f64).sum::<f64>()));
        x2.extend(r2);
    }
    (x2, mag2, y)
}

fn batch(rng: &mut Rng, prefix: usize, suffixes: &[usize]) -> DecisionBatch {
    let mut b = DecisionBatch::default();
    let p: Vec<u32> = (0..prefix).map(|_| (rng.next() % V as u64) as u32).collect();
    let s: Vec<Vec<u32>> = suffixes.iter().map(|&n| (0..n).map(|_| (rng.next() % V as u64) as u32).collect()).collect();
    let refs: Vec<(&[u32], u16)> = s.iter().map(|v| (v.as_slice(), 4u16)).collect();
    b.push_request(&p, &refs).unwrap();
    b.validate().unwrap();
    b
}

#[test]
fn step_graph_eager_and_captured_buckets() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let blas = Blas::new(&stream).unwrap();
    let mut rng = Rng(0x1234_5678_9abc_def1);

    let e = rng.bf16_vec(V * D, 1.0);
    let w1 = rng.bf16_vec(N1 * D, 0.05);
    let w2 = rng.bf16_vec(N2 * C, 0.1);
    let w3 = rng.bf16_vec(N3 * N2, 0.1);
    let up = |v: &[f32], shape: &[usize]| {
        let t = DevTensor::alloc(&ctx, GraphDType::BF16, shape).unwrap();
        t.copy_from_host(&stream, &bf16_bytes(v)).unwrap();
        t
    };
    let lin = |w: DevTensor| {
        let rows = w.shape[0];
        Linear::Bf16 { w, parts: vec![LinearPart { name: "test".into(), row0: 0, rows, w_gs: None }] }
    };
    let (r1, r2, r3) = (
        WeightRef::Layer(0, LayerWeight::MlpGateUp),
        WeightRef::Layer(0, LayerWeight::MlpDown),
        WeightRef::Layer(1, LayerWeight::MlpDown),
    );
    let mut linears = HashMap::new();
    linears.insert(r1, lin(up(&w1, &[N1, D])));
    linears.insert(r2, lin(up(&w2, &[N2, C])));
    linears.insert(r3, lin(up(&w3, &[N3, N2])));
    let store = Store { embed: up(&e, &[V, D]), linears };

    let mut g = StepGraph::new();
    let x0 = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(D)]);
    let x1 = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(N1)]);
    let x2 = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(N2)]);
    let y = g.value(GraphDType::F32, vec![Dim::Tokens, Dim::Fixed(N3)]);
    g.push(GraphOp::Embedding { out: x0 });
    g.push(GraphOp::Gemm { x: x0.into(), w: r1, out: x1 });
    g.push(GraphOp::Gemm { x: TensorRef::Cols { v: x1, start: 0, len: C }, w: r2, out: x2 });
    g.push(GraphOp::Gemm { x: x2.into(), w: r3, out: y });
    g.outputs.extend([x0, x2, y]); // x0 kept live so the test can read the gather back

    let plan = BucketPlan::new(&[64, 128, 256], 16, 64).unwrap();
    let largest = plan.largest();
    let mut up_meta =
        MetaUploader::new(&stream, MetaShape { tokens: largest.tokens, segments: largest.segments, answers: largest.answers })
            .unwrap();
    let mut bg = BucketGraphs::new(&g, plan.clone(), &stream).unwrap();
    for b in &plan.buckets {
        let p = bg.pool(*b).unwrap();
        println!("bucket {:?}: arena {} B (values separately: {} B)", b, p.arena_bytes, p.naive_bytes);
    }

    let read = |ex: &Executor, t: usize| {
        let x2h = bf16_from_bytes(&ex.value(x2).unwrap().rows_view(0, t).unwrap().to_host_bytes(&stream).unwrap());
        let yh = f32_from_bytes(&ex.value(y).unwrap().rows_view(0, t).unwrap().to_host_bytes(&stream).unwrap());
        (x2h, yh)
    };
    let compare = |what: &str, b: &DecisionBatch, x2h: &[f32], yh: &[f32]| {
        let (rx2, mag2, ry) = reference(&b.token_ids, &e, &w1, &w2, &w3);
        // x2 = bf16(x1[:, :C] · W2ᵀ) with x1 itself a bf16-rounded GEMM output. The device and the reference
        // may round an x1 element to neighbouring bf16 values (fp32 vs f64 sum near a tie), so the bound is
        // x2's own half-ulp rounding plus one x1 ulp (2^-7 relative) propagated through W2, plus fp32
        // accumulation: tol = 2^-8 (|r| + δ) + δ, δ = 2^-7 Σ|x1·w2| + C·2^-23 Σ|x1·w2|.
        let mut bad = 0;
        let mut exact = 0;
        let mut max_x2 = 0.0f64;
        for ((a, r), m) in x2h.iter().zip(&rx2).zip(&mag2) {
            let d = (*a as f64 - *r as f64).abs();
            let delta = (2f64.powi(-7) + C as f64 * 2f64.powi(-23)) * m;
            let tol = 2f64.powi(-8) * (r.abs() as f64 + delta) + delta;
            max_x2 = max_x2.max(d / tol);
            exact += (d == 0.0) as usize;
            if d > tol {
                bad += 1;
            }
        }
        let mut errs: Vec<f64> = yh.iter().zip(&ry).map(|(a, r)| (*a as f64 - r).abs()).collect();
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ymax = ry.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        println!("{what}: T={} x2 max err/tol {max_x2:.3} ({exact}/{} bit-equal to the f64 reference); \
                  y vs reference max|err| {:.3e} p99 {:.3e} (|y|max {ymax:.3})",
                 b.num_tokens(), rx2.len(), errs.last().unwrap(), errs[errs.len() * 99 / 100]);
        assert_eq!(bad, 0, "{what}: x2 elements beyond tolerance");
        // y = f32 GEMM of the device x2: compare against the reference fed with the device's own x2.
        let mut max_y = 0.0f64;
        for t in 0..b.num_tokens() {
            for n in 0..N3 {
                let (mut s, mut mag) = (0.0f64, 0.0f64);
                for k in 0..N2 {
                    let p = x2h[t * N2 + k] as f64 * w3[n * N2 + k] as f64;
                    s += p;
                    mag += p.abs();
                }
                let e = (yh[t * N3 + n] as f64 - s).abs();
                max_y = max_y.max(e / (N2 as f64 * 2f64.powi(-23) * mag + 1e-30));
            }
        }
        println!("{what}: y vs f64(device x2) max err/tol {max_y:.3} (tol = K·2^-23·Σ|x·w|)");
        assert!(max_y <= 1.0);
    };

    // Embedding must be a bit-exact gather.
    let check_embed = |ex: &Executor, b: &DecisionBatch| {
        let got = ex.value(x0).unwrap().rows_view(0, b.num_tokens()).unwrap().to_host_bytes(&stream).unwrap();
        let want: Vec<u8> = b.token_ids.iter().flat_map(|&i| bf16_bytes(&e[i as usize * D..(i as usize + 1) * D])).collect();
        if got != want {
            let n = got.chunks(2).zip(want.chunks(2)).filter(|(a, b)| a != b).count();
            let first = got.chunks(2).zip(want.chunks(2)).position(|(a, b)| a != b).unwrap();
            let z = got.iter().filter(|&&x| x == 0).count();
            panic!("embedding gather is not bit-exact: {n} of {} elements differ, first at {first} ({}, {}); zero bytes {z}",
                   got.len() / 2, first / D, first % D);
        }
    };

    // Eager run of one batch, then capture each bucket and replay several batches through it.
    let mut eager_batch = batch(&mut rng, 40, &[5, 9]);
    let b0 = plan.pick(eager_batch.num_tokens(), eager_batch.num_branches(), eager_batch.segments.len()).unwrap();
    let slot = up_meta.pack(&eager_batch, b0.tokens).unwrap();
    up_meta.upload(slot, &stream).unwrap();
    {
        let ex = Executor {
            graph: &g, plan: bg.pool(b0).unwrap(), arena: &bg.arena, weights: &store, kernels: &NoKernels,
            blas: &blas, meta: &up_meta.dev, gdn_state: None, cfg: None,
        };
        ex.run_eager(&stream).unwrap();
        check_embed(&ex, &eager_batch);
        let (x2h, yh) = read(&ex, eager_batch.num_tokens());
        compare("eager", &eager_batch, &x2h, &yh);
    }

    for b in plan.buckets.clone() {
        let graph = {
            let ex = Executor {
                graph: &g, plan: bg.pool(b).unwrap(), arena: &bg.arena, weights: &store, kernels: &NoKernels,
                blas: &blas, meta: &up_meta.dev, gdn_state: None, cfg: None,
            };
            ex.run_eager(&stream).unwrap(); // warm-up outside capture
            ex.capture(&stream).unwrap()
        };
        bg.insert(b, graph).unwrap();
    }

    let shapes: [(usize, &[usize]); 5] = [(30, &[3, 4]), (100, &[10, 7, 2]), (50, &[60]), (200, &[20, 20]), (64, &[])];
    for (i, (p, s)) in shapes.into_iter().enumerate() {
        eager_batch = if s.is_empty() { batch(&mut rng, p - 1, &[1]) } else { batch(&mut rng, p, s) };
        let bt = &eager_batch;
        let b = plan.pick(bt.num_tokens(), bt.num_branches(), bt.segments.len()).unwrap();
        let slot = up_meta.pack(bt, b.tokens).unwrap();
        up_meta.upload(slot, &stream).unwrap();
        bg.launch(b, &stream).unwrap();
        stream.synchronize().unwrap();
        let ex = Executor {
            graph: &g, plan: bg.pool(b).unwrap(), arena: &bg.arena, weights: &store, kernels: &NoKernels,
            blas: &blas, meta: &up_meta.dev, gdn_state: None, cfg: None,
        };
        check_embed(&ex, bt);
        let (x2h, yh) = read(&ex, bt.num_tokens());
        compare(&format!("replay {i} bucket {}", b.tokens), bt, &x2h, &yh);
    }
}

/// Packs batches into alternating host slots and checks the device arrays against an independent rebuild.
#[test]
fn meta_upload_matches_batch() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let shape = MetaShape { tokens: 512, segments: 65, answers: 64 };
    let mut up = MetaUploader::new(&stream, shape).unwrap();
    let mut rng = Rng(77);
    let mut slots = Vec::new();
    for (pi, (plen, suf)) in [(30usize, vec![4usize, 6, 1]), (100, vec![12]), (7, vec![3, 3, 3, 3])].into_iter().enumerate() {
        let mut b = batch(&mut rng, plen, &suf);
        // A second request in the same batch.
        b.push_request(&[1, 2, 3], &[(&[4, 5], 2)]).unwrap();
        b.validate().unwrap();
        let bucket = if pi == 1 { b.num_tokens() } else { 256 };
        let slot = up.pack(&b, bucket).unwrap();
        slots.push(slot);
        let c = up.upload(slot, &stream).unwrap();
        stream.synchronize().unwrap();
        let rd = |t: &DevTensor| -> Vec<u32> {
            t.to_host_bytes(&stream).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let d = &up.dev;
        let t = b.num_tokens();
        let pad = bucket - t;
        // Independent rebuild of the expected arrays.
        let mut pos = vec![0u32; bucket];
        let mut seg = vec![0u32; bucket];
        for (si, s) in b.segments.iter().enumerate() {
            for k in 0..s.len {
                pos[(s.start + k) as usize] = s.pos0 + k;
                seg[(s.start + k) as usize] = si as u32;
            }
        }
        for k in 0..pad {
            pos[t + k] = k as u32;
            seg[t + k] = b.segments.len() as u32;
        }
        let ids = rd(&d.token_ids);
        assert_eq!(&ids[..t], &b.token_ids[..]);
        assert!(ids[t..bucket].iter().all(|&v| v == 0));
        assert_eq!(&rd(&d.positions)[..bucket], &pos[..]);
        assert_eq!(&rd(&d.token_seg)[..bucket], &seg[..]);
        let segs = rd(&d.segments);
        for (si, s) in b.segments.iter().enumerate() {
            assert_eq!(&segs[si * 4..si * 4 + 4], &[s.start, s.len, s.parent as u32, s.pos0]);
        }
        if pad > 0 {
            let si = b.segments.len();
            assert_eq!(&segs[si * 4..si * 4 + 4], &[t as u32, pad as u32, PAD_PARENT as u32, 0]);
        }
        let state = rd(&d.seg_state);
        let mut ord = 0u32;
        let mut prefix_ord = vec![0u32; b.segments.len()];
        for (si, s) in b.segments.iter().enumerate() {
            if s.is_prefix() {
                prefix_ord[si] = ord;
                ord += 1;
                assert_eq!(state[si], prefix_ord[si]);
            } else {
                assert_eq!(state[si], prefix_ord[s.parent as usize]);
            }
        }
        let q = b.num_branches();
        assert_eq!(&rd(&d.answer_rows)[..q], &b.answer_rows[..]);
        let slots_dev = rd(&d.answer_slots);
        assert!(slots_dev[..q].iter().zip(&b.answer_slots).all(|(a, b)| *a == *b as u32));
        assert!(slots_dev[q..].iter().all(|&v| v == 0));
        let h = rd(&d.header);
        assert_eq!(&h[..5], &[t as u32, b.segments.len() as u32, q as u32, ord, bucket as u32]);
        assert_eq!(h[5], if pad > 0 { b.segments.len() as u32 } else { u32::MAX });
        assert_eq!((c.tokens, c.answers, c.prefixes), (t, q, ord as usize));
        println!("meta batch {pi}: T={t} bucket={bucket} S={} Q={q} prefixes={ord} slot={slot}: ok", b.segments.len());
    }
    assert_eq!(slots, [0, 1, 0], "host slots alternate");
}
