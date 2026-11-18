//! CPU-only integration; no GPU, checkpoint weights or ignored tests.
//! Index: metadata-only NVFP4 rejection before CUDA; synthetic trimmed IR spans/aliases and T/Q GEMM rows.
//! Direct config and IR fixtures test validation/planning, not full-model numerics.
use std::{fs, path::PathBuf};
use sysone::engine::{gpu::{EngineOptions, ExecutionMode, GpuEngine},
    graph::{Bucket, Dim, GraphOp, LayerWeight, StepGraph, StepShape, WeightRef},
    graph::tensor::GraphDType};
use sysone::kernels::gdn::GdnBackend;

struct Fixture(PathBuf);
impl Drop for Fixture { fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); } }

// A config-only directory proves format rejection happens before opening missing weights or CUDA.
#[test]
fn autoloaded_nvfp4_is_refused_before_cuda_or_checkpoint_open() {
    let dir = std::env::temp_dir().join(format!("exact-format-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let fixture = Fixture(dir);
    let cfg = serde_json::json!({
        "model_type":"qwen3_5", "quantization_config":{"format":"nvfp4-pack-quantized"},
        "text_config":{
            "hidden_size":4096,"intermediate_size":12288,"num_hidden_layers":1,
            "layer_types":["full_attention"],"vocab_size":248320,"rms_norm_eps":1e-6,
            "hidden_act":"silu","tie_word_embeddings":false,"eos_token_id":1,
            "num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,
            "attn_output_gate":true,"attention_bias":false,
            "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"rope_type":"default"},
            "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
            "linear_num_value_heads":32,"linear_value_head_dim":128
        },
        "modeling_overrides":{"attention":{"full_attention":"block_bidirectional","linear_attention":"causal"},
            "readout":"slot_lm","n_slots":256}
    });
    fs::write(fixture.0.join("config.json"), cfg.to_string()).unwrap();
    let opts = EngineOptions { execution_mode:ExecutionMode::ExactEager, cuda_graphs:false,
        gdn_backend:Some(GdnBackend::Tile), weights:None, ..Default::default() };
    let err = GpuEngine::load(&fixture.0, &[1,2], opts).err().unwrap();
    assert_eq!(err.to_string(), "exact eager mode supports only bf16 weights");
}

// Minimal trim-operator chain: all-token KV, answer-only queries/attention/MLP/readout.
fn trimmed_ir(nvfp4: bool) -> StepGraph {
    use Dim::{Answers as A, Tokens as T};
    let mut g = StepGraph::new();
    let val = |g: &mut StepGraph, rows, cols| g.value(GraphDType::BF16, vec![rows,Dim::Fixed(cols)]);
    let w = |p| WeightRef::Layer(0,p);
    let x = val(&mut g,T,4096);
    g.push(GraphOp::Embedding {out:x});
    let kv = val(&mut g,T,2048);
    g.push(GraphOp::Gemm {x:x.into(),w:w(LayerWeight::AttnKv),out:kv});
    let unused = val(&mut g,T,8192);
    let qu = val(&mut g,T,4096);
    let k = val(&mut g,T,1024);
    g.push(GraphOp::Uninit {out:unused});
    g.push(GraphOp::QkNormRope {layer:0,qg:unused.into(),kv:kv.into(),q:qu,k});
    let xa = val(&mut g,A,4096);
    g.push(GraphOp::GatherRows {src:x.into(),out:xa});
    let qg = val(&mut g,A,8192);
    g.push(GraphOp::Gemm {x:xa.into(),w:w(LayerWeight::AttnQ),out:qg});
    let kva = val(&mut g,A,2048);
    g.push(GraphOp::GatherRows {src:kv.into(),out:kva});
    let qa = val(&mut g,A,4096);
    let ka = val(&mut g,A,1024);
    g.push(GraphOp::QkNormRopeAnswers {layer:0,qg:qg.into(),kv:kva.into(),q:qa,k:ka});
    let ao = val(&mut g,A,4096);
    g.push(GraphOp::SegAttnAnswers {layer:0,q:qa.into(),k:k.into(),kv:kv.into(),qg:qg.into(),out:ao});
    let mix = val(&mut g,A,4096);
    g.push(GraphOp::Gemm {x:ao.into(),w:w(LayerWeight::AttnO),out:mix});
    let res = val(&mut g,A,4096);
    let hn = val(&mut g,A,4096);
    g.push(GraphOp::AddZcRmsNorm {residual:xa,x:mix.into(),w:w(LayerWeight::PostNorm),out:hn,residual_out:res});
    let out = if nvfp4 {
        let act = |g: &mut StepGraph,k: usize| (g.value(GraphDType::U8,vec![A,Dim::Fixed(k/2)]),
            g.value(GraphDType::F8E4M3,vec![A,Dim::Fixed(k/16)]));
        let (q,s) = act(&mut g,4096);
        g.push(GraphOp::QuantizeNvfp4 {x:hn.into(),w:w(LayerWeight::MlpGateUp),q,s});
        let (q2,s2) = act(&mut g,12288);
        g.push(GraphOp::GemmNvfp4SwiGlu {q:q.into(),s:s.into(),w:w(LayerWeight::MlpGateUp),
            w_down:w(LayerWeight::MlpDown),q_out:q2,s_out:s2});
        let res2 = val(&mut g,A,4096);
        g.push(GraphOp::GemmNvfp4Residual {q:q2.into(),s:s2.into(),w:w(LayerWeight::MlpDown),residual:res,residual_out:res2});
        let norm = val(&mut g,A,4096);
        g.push(GraphOp::ZcRmsNormRes {residual:res2,w:WeightRef::FinalNorm,out:norm});
        norm
    } else { hn };
    let logits = g.value(GraphDType::F32,vec![A,Dim::Fixed(7)]);
    g.push(GraphOp::Gemm {x:out.into(),w:WeightRef::SlotHead,out:logits});
    g.outputs.push(logits);
    g
}

// Walk each op's live values to check capacity-root separation, in-place aliases and logical row traces.
#[test]
fn trimmed_ops_keep_t_q_views_aliases_and_strides_inside_capacity() {
    for nvfp4 in [false,true] {
        let g = trimmed_ir(nvfp4);
        let b = Bucket {tokens:256,answers:32,segments:65};
        let p = g.plan(b).unwrap();
        // Include bucketed NVFP4 IR without enabling NVFP4 exact execution in the engine.
        let shapes = if nvfp4 {vec![b.shape()]} else {vec![b.shape(),
            StepShape {tokens:65,answers:1,segments:2},StepShape {tokens:129,answers:16,segments:17}]};
        for shape in shapes {
            p.validate_shape(&g,shape).unwrap();
            for (i,op) in g.ops.iter().enumerate() {
                let live: Vec<_> = op.inputs().into_iter().chain(op.outputs()).collect();
                for a in &live { for c in &live {
                    if p.roots[a.0] != p.roots[c.0] {
                        let (oa,oc) = (p.offsets[a.0],p.offsets[c.0]);
                        assert!(oa+p.storage_bytes[a.0] <= oc || oc+p.storage_bytes[c.0] <= oa,
                            "live capacity roots overlap at op {i}");
                    }
                }}
                for v in live {
                    let spec = g.spec(v);
                    let dims = spec.concrete(shape);
                    assert_eq!(dims[0],shape.dim(spec.shape[0]));
                    assert!(spec.bytes(shape) <= p.storage_bytes[v.0]);
                    if let Some((a,c)) = op.in_place() {
                        assert_eq!((p.roots[a.0],p.offsets[a.0]),(p.roots[c.0],p.offsets[c.0]));
                    }
                }
            }
            let trace = g.gemm_requests(shape);
            assert_eq!(trace[0].row_kind,"tokens");
            assert_eq!(trace[0].rows,shape.tokens);
            assert!(trace[1..].iter().all(|r|r.row_kind == "answers" && r.rows == shape.answers));
            assert_eq!(trace.len(),if nvfp4 {6} else {4});
        }
        assert_eq!(b.dim(Dim::Tokens),256); // Original capacity API remains available.
        assert_eq!(g.spec(g.outputs[0]).concrete(&b),[32,7]);
    }
}
