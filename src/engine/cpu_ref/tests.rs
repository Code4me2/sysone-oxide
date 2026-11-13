//! Tree semantics on a tiny random model: the packed tree forward equals one flat forward per branch with the
//! explicit block mask (the reference's single-forward semantics), and is batch-invariant.

use super::model::{self, AttnWeights, Dims, GdnWeights, Layer, Mixer, Model};
use super::weights::Mat;
use super::*;
use crate::engine::config::LayerType;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }

    fn mat(&mut self, r: usize, c: usize) -> Mat {
        let s = (c as f32).powf(-0.5);
        Mat::from_f32(r, c, self.vec(r * c, s))
    }
}

fn tiny_dims() -> Dims {
    Dims {
        hidden: 32,
        intermediate: 48,
        vocab: 40,
        eps: 1e-6,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        rotary_dim: 4,
        rope_theta: 1e4,
        conv_kernel: 4,
        k_heads: 2,
        k_dim: 8,
        v_heads: 4,
        v_dim: 8,
    }
}

fn tiny_model(seed: u64) -> Model {
    let d = tiny_dims();
    let mut r = Rng(seed);
    let h = d.hidden;
    let layers = d
        .layer_types
        .iter()
        .map(|ty| {
            let mixer = match ty {
                LayerType::LinearAttention => Mixer::Gdn(GdnWeights {
                    in_proj_qkv: r.mat(d.conv_dim(), h),
                    in_proj_z: r.mat(d.value_dim(), h),
                    in_proj_b: r.mat(d.v_heads, h),
                    in_proj_a: r.mat(d.v_heads, h),
                    conv: r.vec(d.conv_dim() * d.conv_kernel, 0.6),
                    a_log: r.vec(d.v_heads, 1.0),
                    dt_bias: r.vec(d.v_heads, 1.0),
                    norm: r.vec(d.v_dim, 0.3).iter().map(|x| 1.0 + x).collect(),
                    out_proj: r.mat(h, d.value_dim()),
                }),
                LayerType::FullAttention => Mixer::Attn(AttnWeights {
                    q_proj: r.mat(2 * d.heads * d.head_dim, h),
                    k_proj: r.mat(d.kv_heads * d.head_dim, h),
                    v_proj: r.mat(d.kv_heads * d.head_dim, h),
                    o_proj: r.mat(h, d.heads * d.head_dim),
                    q_norm: r.vec(d.head_dim, 0.3),
                    k_norm: r.vec(d.head_dim, 0.3),
                }),
            };
            Layer {
                input_layernorm: r.vec(h, 0.3),
                post_attention_layernorm: r.vec(h, 0.3),
                mixer,
                gate_proj: r.mat(d.intermediate, h),
                up_proj: r.mat(d.intermediate, h),
                down_proj: r.mat(h, d.intermediate),
            }
        })
        .collect();
    let model = Model {
        embed: Mat::from_f32(d.vocab, h, r.vec(d.vocab * h, 1.0)),
        norm: r.vec(h, 0.3),
        lm_head: r.mat(d.vocab, h),
        layers,
        dims: d,
    };
    model.check().unwrap();
    model
}

type Request = (Vec<u32>, Vec<Vec<u32>>);

fn requests() -> Vec<Request> {
    vec![
        // A 2-token prefix: the first suffix tokens' conv taps reach past the prefix start (zero padding).
        (vec![3, 7], vec![vec![1, 2, 3, 4, 5], vec![9], vec![11, 12]]),
        (vec![5, 6, 7, 8, 9, 10, 11, 12, 13], vec![vec![20, 21, 22], vec![30, 31, 32, 33, 34, 35, 36]]),
        (vec![1], vec![vec![2, 2, 2, 2]]),
    ]
}

fn pack(reqs: &[Request]) -> DecisionBatch {
    let mut b = DecisionBatch::default();
    for (p, sufs) in reqs {
        let s: Vec<(&[u32], u16)> = sufs.iter().map(|s| (s.as_slice(), 3)).collect();
        b.push_request(p, &s).unwrap();
    }
    b.validate().unwrap();
    b
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

#[test]
fn tree_equals_flat_block_mask_forward() {
    let m = tiny_model(1);
    let reqs = requests();
    let batch = pack(&reqs);
    let layout = Layout::from_batch(&batch, m.dims.conv_kernel);
    let tree = model::forward_hidden(&m, &batch.token_ids, &layout, &batch.answer_rows);
    let h = m.dims.hidden;
    let mut row = 0;
    let mut worst = 0.0f64;
    for (p, sufs) in &reqs {
        for s in sufs {
            let toks: Vec<u32> = p.iter().chain(s).copied().collect();
            let flat = Layout::flat(toks.len(), p.len(), m.dims.conv_kernel);
            let f = model::forward_hidden(&m, &toks, &flat, &[toks.len() as u32 - 1]);
            worst = worst.max(max_abs(&tree[row * h..(row + 1) * h], &f));
            row += 1;
        }
    }
    println!("tree vs flat block-mask forward: max |dh| {worst:.3e} over {row} branches (tol 1e-5)");
    assert!(worst <= 1e-5, "tree forward differs from the flat forward by {worst}");
}

#[test]
fn packed_equals_alone_bitwise() {
    let mut e = CpuRefEngine::from_parts(tiny_model(2), (0..MAX_SLOTS as u32).map(|i| i % 40).collect()).unwrap();
    let reqs = requests();
    let packed = e.run(&pack(&reqs)).unwrap();
    let mut row = 0;
    for r in &reqs {
        let alone = e.run(&pack(std::slice::from_ref(r))).unwrap();
        for i in 0..alone.rows {
            assert_eq!(alone.row(i, MAX_SLOTS), packed.row(row, MAX_SLOTS), "branch {row} differs alone vs packed");
            row += 1;
        }
    }
}

/// Negative controls: dropping the parent's delta-rule state, or its conv taps, or the parent keys changes the
/// answer, so the tree test above exercises each of them.
#[test]
fn each_parent_link_matters() {
    let m = tiny_model(3);
    let batch = pack(&requests());
    let good = Layout::from_batch(&batch, m.dims.conv_kernel);
    let base = model::forward_hidden(&m, &batch.token_ids, &good, &batch.answer_rows);
    let mut no_state = good.clone();
    no_state.runs.iter_mut().for_each(|r| r.init = None);
    let mut no_taps = good.clone();
    for s in batch.segments.iter().filter(|s| !s.is_prefix()) {
        for t in s.start..s.end() {
            for (d, prev) in no_taps.conv_prev[t as usize].iter_mut().enumerate() {
                if (t - s.start) as usize <= d {
                    *prev = None;
                }
            }
        }
    }
    let mut no_keys = good.clone();
    for s in batch.segments.iter().filter(|s| !s.is_prefix()) {
        for t in s.start..s.end() {
            no_keys.keys[t as usize].retain(|&j| j >= s.start);
        }
    }
    for (name, l) in [("state", no_state), ("conv taps", no_taps), ("prefix keys", no_keys)] {
        let h = model::forward_hidden(&m, &batch.token_ids, &l, &batch.answer_rows);
        let d = max_abs(&base, &h);
        println!("without parent {name}: max |dh| {d:.3e}");
        assert!(d > 1e-3, "dropping the parent {name} changed nothing ({d})");
    }
}

#[test]
fn flat_mask_is_block_bidirectional() {
    let l = Layout::flat(5, 2, 4);
    assert_eq!(l.keys[0], [0, 1]);
    assert_eq!(l.keys[1], [0, 1]);
    assert_eq!(l.keys[2], [0, 1, 2, 3, 4]);
    assert_eq!(l.conv_prev[1], [Some(0), None, None]);
    let mut b = DecisionBatch::default();
    b.push_request(&[1, 2], &[(&[3, 4, 5], 2)]).unwrap();
    let t = Layout::from_batch(&b, 4);
    assert_eq!(t.keys, l.keys);
    assert_eq!(t.conv_prev, l.conv_prev);
    assert_eq!(t.pos, l.pos);
}

#[test]
fn slot_labels_match_qev() {
    let l = slot_labels();
    assert_eq!(l.len(), MAX_SLOTS);
    assert_eq!(l[0], "A");
    assert_eq!(l[25], "Z");
    assert_eq!(l[26], "AA");
    assert!(!l.contains(&"BQ".to_string()));
    // 26 + 676 - 25 excluded > 256, so the list is cut inside the two-letter labels.
    assert_eq!(l[255], "JU"); // qev.prompt.SLOT_LABELS[255]
}
