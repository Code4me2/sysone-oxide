//! Portable, synthetic public-model correctness check. CPU planning is the default; the full-model
//! test is ignored and must run alone in a fresh process under a real GPU lock. No private goldens.
use std::{collections::BTreeMap, fs::{File, OpenOptions}, io::Write, path::PathBuf, time::Instant};
use cutile::tile_kernel::jit_compile_count;
use serde::Serialize;
use serde_json::{json, Value};
use sysone::{contract::{DecisionBatch, DecisionEngine, SlotLogits},
    engine::{config::{ModelConfig, WeightFormat}, gemm::GemmMode,
        gpu::{EngineOptions, ExecutionMode, GpuEngine}, graph::{Dim, GraphOp, LayerWeight, StepShape, WeightRef}, warmup},
    frontend::{Frontend, Limits}, kernels::gdn::GdnBackend, tolerances::{Stats, Tolerances}};

#[derive(Clone, Debug, Serialize)]
struct Request { id: &'static str, prefix: usize, suffixes: Vec<usize>, seed: u64 }
#[derive(Clone, Debug, Serialize)]
struct Case { id: &'static str, requests: Vec<Request>, repeat: Option<usize> }
fn request(id: &'static str, prefix: usize, suffixes: Vec<usize>, seed: u64) -> Request {
    Request { id, prefix, suffixes, seed }
}
fn cases() -> Vec<Case> {
    let one = |id, t, q, seed| Case { id, requests: vec![request(id,t-2*q,vec![2;q],seed)], repeat:None };
    let mut c = vec![one("t63",63,1,63),one("t64",64,1,64),one("t65",65,1,65),
        one("t255",255,1,255),one("t256",256,1,256),one("t257",257,1,257),
        one("long",16384,2,16384)];
    let restore = |c: &mut Vec<Case>, i: usize| { let mut x=c[i].clone(); x.repeat=Some(i); c.push(x); };
    restore(&mut c,2); restore(&mut c,6);
    c.push(one("many16",80,16,8016)); c.push(one("few80",80,1,8001)); restore(&mut c,9);
    c.push(one("many64",257,64,25764)); restore(&mut c,0); restore(&mut c,12);
    let a=request("A",31,vec![2,3],31023); let b=request("B",45,vec![2],4502);
    c.push(Case {id:"AB",requests:vec![a.clone(),b.clone()],repeat:None});
    c.push(Case {id:"BA",requests:vec![b,a],repeat:None}); restore(&mut c,15);
    c
}
fn tokens(n: usize, seed: u64, pool: &[u32]) -> Vec<u32> {
    assert!(!pool.is_empty()); let mut s=seed;
    (0..n).map(|_| { s=s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        pool[(s >> 32) as usize % pool.len()] }).collect()
}
// Identity is semantic request/branch identity, NOT the push_request return value or a reused AB map.
fn row_keys(c: &Case) -> Vec<String> {
    c.requests.iter().flat_map(|r| (0..r.suffixes.len()).map(move |q| format!("{}:{q}",r.id))).collect()
}
fn batch(c: &Case, pool: &[u32], slots: usize) -> DecisionBatch {
    assert!(slots>=2);
    let mut b=DecisionBatch::default();
    for r in &c.requests {
        let p=tokens(r.prefix,r.seed,pool);
        let ss: Vec<_>=r.suffixes.iter().enumerate().map(|(i,&n)| tokens(n,r.seed+100+i as u64,pool)).collect();
        let branches: Vec<_>=ss.iter().enumerate().map(|(i,s)| (s.as_slice(),(2+i%(slots.min(8)-1)) as u16)).collect();
        b.push_request(&p,&branches).unwrap();
    }
    b.validate().unwrap(); b
}
fn options(exact: bool, graph: bool) -> EngineOptions {
    EngineOptions { buckets:EngineOptions::buckets_up_to(16384), max_prefixes:32,
        execution_mode:if exact {ExecutionMode::ExactEager} else {ExecutionMode::Bucketed},
        cuda_graphs:graph, warmup:exact, bounded_exact_warmup:exact,
        weights:Some(WeightFormat::Bf16),gdn_backend:Some(GdnBackend::Tile),..Default::default() }
}
fn frozen_limits() -> [f64;5] {
    let t=Tolerances::parse(include_str!("tolerances.toml")).unwrap();
    let v=["dlogit_max","dlogit_p99","dp_max","dp_p99","top1_margin"].map(|k|t.f64(&format!("e2e.bf16.{k}")).unwrap());
    assert_eq!(v,[0.2842,0.1201,0.04241,0.01846,0.02]); v
}
#[test]
fn case_plan_cpu_validation() {
    let c=cases(); assert_eq!(c.len(),18); let pool:Vec<_>=(0..257).collect();
    let expected=[(63,1,1),(64,1,1),(65,1,1),(255,1,1),(256,1,1),(257,1,1),
        (16384,2,1),(65,1,1),(16384,2,1),(80,16,1),(80,1,1),(80,16,1),
        (257,64,1),(63,1,1),(257,64,1),(83,3,2),(83,3,2),(83,3,2)];
    let opts=options(true,false); opts.validate_bounded_exact_warmup().unwrap();
    assert_eq!(opts.gemm,GemmMode::Pinned); assert!(opts.trim_last_layer);
    let plan=opts.bucket_plan(&opts.buckets).unwrap();
    let bs:Vec<_>=c.iter().map(|c|batch(c,&pool,8)).collect();
    for (i,b) in bs.iter().enumerate() {
        let (t,q,p)=expected[i];
        assert_eq!((b.num_tokens(),b.num_branches(),b.segments.iter().filter(|s|s.is_prefix()).count()),(t,q,p));
        assert_eq!(b.segments.len(),p+q); assert!(b.answer_slots.iter().all(|&n|n>=2 && n<=8));
        assert!(b.token_ids.iter().all(|&t|t<257));
        let bucket=plan.pick(t,q,p+q).unwrap(); assert!(bucket.tokens<=16384);
        if i==12 || i==14 {assert_eq!((bucket.tokens,bucket.answers),(512,64));}
        if let Some(j)=c[i].repeat { assert_eq!(b,&bs[j]); assert_eq!(row_keys(&c[i]),row_keys(&c[j])); }
    }
    assert_eq!(row_keys(&c[15]),["A:0","A:1","B:0"]);
    assert_eq!(row_keys(&c[16]),["B:0","A:0","A:1"]);
    let ba: BTreeMap<_,_>=row_keys(&c[16]).into_iter().enumerate().map(|(i,k)|(k,i)).collect();
    let permutation:Vec<_>=row_keys(&c[15]).iter().map(|k|ba[k]).collect();
    assert_eq!(permutation,[1,2,0]); assert_ne!(permutation,[0,1,2]);
    // Independently check per-request tokens, RoPE positions and suffix order survived reordering.
    for (ai,bi) in [(0,2),(3,0)] {
        let (a,b)=(&bs[15],&bs[16]); let n=if ai==0 {3} else {2};
        for j in 0..n {let (x,y)=(a.segments[ai+j],b.segments[bi+j]);
            assert_eq!((x.len,x.pos0,x.is_prefix()),(y.len,y.pos0,y.is_prefix()));
            assert_eq!(&a.token_ids[x.start as usize..x.end() as usize],&b.token_ids[y.start as usize..y.end() as usize]);}
    }
    assert_eq!(warmup::plan().len(),51); frozen_limits();
}
struct Records(Option<File>);
impl Records {
    fn open() -> Self {
        Self(std::env::var_os("SYSONE_EXACT_GPU_OUTPUT").map(|p| {
            let p=PathBuf::from(p); assert!(p.is_absolute(),"explicit absolute output path required");
            OpenOptions::new().write(true).create_new(true).open(p).expect("exclusive JSONL output")
        }))
    }
    fn put(&mut self, v: Value) {
        if let Some(f)=&mut self.0 {serde_json::to_writer(&mut *f,&v).unwrap(); writeln!(f).unwrap(); f.flush().unwrap();}
        else {println!("{v}");}
    }
}
#[cfg(unix)]
fn require_real_lock() {
    use std::os::fd::AsRawFd;
    unsafe extern "C" {fn flock(fd:i32, op:i32)->i32;}
    assert_eq!(std::env::var("SYSONE_GPU_LOCKED").as_deref(),Ok("1"));
    let p=std::env::var_os("SYSONE_EXACT_GPU_LOCK").expect("real wrapper lock file required");
    let f=File::open(p).expect("open actual held lock");
    let rc=unsafe {flock(f.as_raw_fd(),2|4)};
    assert_eq!(rc,-1,"real lock must already be held by wrapper");
    assert_eq!(std::io::Error::last_os_error().kind(),std::io::ErrorKind::WouldBlock,"lock contention proof");
}
#[cfg(not(unix))]
fn require_real_lock() {panic!("real flock required");}
fn trace(e:&GpuEngine, b:&DecisionBatch) -> Value {
    let s=e.last; assert_eq!(s.steps,1);
    assert_eq!((s.logical_tokens,s.logical_answers,s.logical_segments),(b.num_tokens(),b.num_branches(),b.segments.len()));
    let exact=e.opts.execution_mode==ExecutionMode::ExactEager;
    let shape=StepShape {tokens:s.execution_tokens,answers:s.execution_answers,segments:s.execution_segments};
    assert_eq!((shape.tokens,shape.answers,shape.segments),if exact {
        (s.logical_tokens,s.logical_answers,s.logical_segments)
    } else {(s.bucket_tokens,s.bucket_answers,s.bucket_segments)});
    let got=e.last_gemm_requests().unwrap();
    // Independent enumeration of actual graph ops/specs, not a historical fixed GEMM count.
    let specs:Vec<_>=e.fwd.graph.ops.iter().enumerate().filter_map(|(i,op)|match op {
        GraphOp::Gemm {out,..}=>Some((i,e.fwd.graph.spec(*out))), _=>None }).collect();
    assert_eq!(got.len(),specs.len()); assert!(!got.is_empty());
    let mut has_t=false; let mut has_q=false;
    for (r,(i,spec)) in got.iter().zip(specs) {
        assert_eq!(r.op_index,i); assert_eq!(r.rows,spec.concrete(shape)[0]);
        let (kind,n)=match spec.shape[0] {Dim::Tokens=>{has_t=true;("tokens",shape.tokens)},
            Dim::Answers=>{has_q=true;("answers",shape.answers)},_=>panic!("unexpected BF16 GEMM row kind")};
        assert_eq!((r.row_kind,r.rows),(kind,n));
    }
    assert!(has_t && has_q); assert_eq!(got.last().unwrap().row_kind,"answers");
    let last_layer=e.cfg.text_config.num_hidden_layers-1;
    let mut trimmed=vec![];
    for op in &e.fwd.graph.ops {
        if let GraphOp::Gemm {w:WeightRef::Layer(layer,weight),out,..}=op {
            if *layer==last_layer {
                let dim=e.fwd.graph.spec(*out).shape[0];
                assert_eq!(dim,if *weight==LayerWeight::AttnKv {Dim::Tokens} else {Dim::Answers});
                trimmed.push(json!({"weight":format!("{weight:?}"),"rows":e.fwd.graph.spec(*out).concrete(shape)[0],"kind":format!("{dim:?}")}));
            }
        }
    }
    assert!(!trimmed.is_empty());
    json!({"logical":[s.logical_tokens,s.logical_answers,s.logical_segments],
        "capacity":[s.bucket_tokens,s.bucket_answers,s.bucket_segments],"execution":[shape.tokens,shape.answers,shape.segments],
        "steps":s.steps,"replayed":s.replayed,"total_bucket_tokens":s.total_bucket_tokens,"trimmed_last_layer":trimmed,
        "gemm_requested_rows":got.iter().map(|r|json!({"op":r.op_index,"label":r.label,"kind":r.row_kind,"rows":r.rows})).collect::<Vec<_>>()})
}
type Rows=Vec<Vec<u32>>;
fn active(l:&SlotLogits,b:&DecisionBatch)->Rows {
    assert_eq!(l.rows,b.num_branches());
    b.answer_slots.iter().enumerate().map(|(i,&n)|l.row(i,n as usize).iter().map(|x|x.to_bits()).collect()).collect()
}
fn floats(row:&[u32])->Vec<f64> {row.iter().map(|&x|f32::from_bits(x) as f64).collect()}
fn compare(label:&str,a:&Rows,b:&Rows, lim:[f64;5],out:&mut Records) {
    assert_eq!(a.len(),b.len()); let (mut dl,mut dp,mut flips)=(vec![],vec![],vec![]);
    for (i,(x,y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.len(),y.len()); let (x,y)=(floats(x),floats(y));
        assert!(x.iter().chain(&y).all(|x|x.is_finite()),"nonfinite {label} row {i}");
        let (px,py)=(sysone::engine::goldens::softmax(&x),sysone::engine::goldens::softmax(&y));
        dl.extend(x.iter().zip(&y).map(|(x,y)|(x-y).abs())); dp.extend(px.iter().zip(&py).map(|(x,y)|(x-y).abs()));
        let top=|p:&[f64]| {let mut s:Vec<_>=p.iter().copied().enumerate().collect();s.sort_by(|a,b|b.1.total_cmp(&a.1));(s[0].0,s[0].1-s[1].1)};
        let (ta,tb)=(top(&px),top(&py));
        if ta.0!=tb.0 && (ta.1>lim[4] || tb.1>lim[4]) {flips.push(i);}
    }
    let (l,p)=(Stats::of(&dl).unwrap(),Stats::of(&dp).unwrap());
    let ok=l.max<=lim[0] && l.p99<=lim[1] && p.max<=lim[2] && p.p99<=lim[3] && flips.is_empty();
    out.put(json!({"kind":"comparison","label":label,"values":dl.len(),"dlogit":{"max":l.max,"p99":l.p99,"p50":l.p50},
        "dp":{"max":p.max,"p99":p.p99,"p50":p.p50},"confident_flips":flips,"limits":lim,"ok":ok}));
    assert!(ok,"numerical gate failed: {label}");
}
#[test]
#[ignore = "full public model + real GPU lock; independent review and main readiness required"]
fn full_public_model_exact_state_warmup() {
    assert_eq!(jit_compile_count(),0,"FIRST fresh test process required");
    require_real_lock(); let mut out=Records::open(); let start=Instant::now();
    let model=PathBuf::from(std::env::var_os("SYSONE_MODEL_DIR").expect("SYSONE_MODEL_DIR required"));
    let fe=Frontend::load(&model,Limits {max_branch_tokens:16384,max_request_tokens:16384}).unwrap();
    assert!(fe.slot_ids.len()>=2);
    let cfg=ModelConfig::load(&model).unwrap();
    let pool=fe.tokenizer.encode("The public synthetic decision check uses deterministic tokens for state isolation. ABC xyz 0123456789").unwrap();
    assert!(pool.iter().chain(&fe.slot_ids).all(|&x|(x as usize)<cfg.text_config.vocab_size));
    let c=cases(); let bs:Vec<_>=c.iter().map(|c|batch(c,&pool,fe.slot_ids.len())).collect();
    let lim=frozen_limits();
    out.put(json!({"kind":"plan","first_process_jit":jit_compile_count(),"cases":c,"batches":bs.iter().map(|b|json!({"token_ids":b.token_ids,
        "segments":b.segments.iter().map(|s|json!([s.start,s.len,s.parent,s.pos0])).collect::<Vec<_>>(),
        "answer_rows":b.answer_rows,"answer_slots":b.answer_slots})).collect::<Vec<_>>(),"slot_ids":fe.slot_ids,"limits":lim}));
    let mut modes=vec![];
    for (mode,exact,graph) in [("exact_eager",true,false),("bucket_eager",false,false),("bucket_graph",false,true)] {
        let load=Instant::now(); let mut e=GpuEngine::load(&model,&fe.slot_ids,options(exact,graph)).unwrap();
        let info:BTreeMap<_,_>=e.info().into_iter().collect();
        out.put(json!({"kind":"engine","mode":mode,"info":info,"load_ms":load.elapsed().as_secs_f64()*1000.,"jit":jit_compile_count()}));
        assert_eq!(info["engine"],"bf16"); assert_eq!(info["gemm"],"pinned"); assert_eq!(info["trim_last_layer"],"true");
        assert_eq!(info["execution_mode"],mode); assert_eq!(info["gdn_backend"],"tile");
        assert_eq!(info["cuda_graphs"],if exact {"never"} else if graph {"enabled"} else {"disabled"});
        assert!(e.fwd.trimmed && e.opts.gemm==GemmMode::Pinned);
        assert_eq!(e.opts.max_prefixes,32); assert_eq!(e.opts.buckets,EngineOptions::buckets_up_to(16384));
        let ready_jit=jit_compile_count();
        if exact {
            let r=e.warmup_report().expect("complete bounded report");
            out.put(json!({"kind":"warmup","mode":mode,"report":r,"ready_jit":ready_jit}));
            assert_eq!(info["warmup_policy"],warmup::POLICY);
            assert!(r.complete && r.warmup_ms<90000.); assert_eq!(r.plan,warmup::plan());
            assert_eq!((r.calls.len(),r.standard_calls,r.coverage_calls,r.validation_calls),(51,1,25,25));
            assert_eq!(r.validation_new_installations,0); assert_eq!(r.jit_after,ready_jit);
            for (i,call) in r.calls.iter().enumerate() {assert!(call.ok);assert_eq!(call.sequence,i);assert_eq!(call.layout,r.plan[i]);
                if call.layout.phase=="validation" {assert_eq!(call.jit_before,call.jit_after);}}
            assert_eq!(e.captured_bucket_count(),0);
        } else {assert!(e.warmup_report().is_none());assert_eq!(info["warmup_policy"],"disabled");}
        let mut results:Vec<Rows>=vec![];
        let mut captured_tokens=std::collections::BTreeSet::new();
        for (i,(case,b)) in c.iter().zip(&bs).enumerate() {
            let mut previous=None;
            for run in 0..if graph {2} else {1} {
                let before=jit_compile_count(); let t=Instant::now(); let logits=e.run(b).unwrap();
                let after=jit_compile_count(); let rows=active(&logits,b); let shape=trace(&e,b);
                let probabilities:Vec<Option<Vec<u64>>>=rows.iter().map(|r| {
                    let f=floats(r); if f.iter().all(|x|x.is_finite()) {
                        Some(sysone::engine::goldens::softmax(&f).iter().map(|p|p.to_bits()).collect())
                    } else {None}
                }).collect();
                out.put(json!({"kind":"run","mode":mode,"step":i+1,"case":case.id,"run":run,"keys":row_keys(case),
                    "active_bits":rows,"active_softmax_bits64":probabilities,"shape":shape,"jit_before":before,"jit_after":after,"ready_jit":ready_jit,
                    "ms":t.elapsed().as_secs_f64()*1000.,"captured":e.captured_bucket_count()}));
                // Write raw bits BEFORE checking finiteness, preserving NaN/Inf payloads on failure.
                assert!(rows.iter().flatten().all(|&x|f32::from_bits(x).is_finite()),"nonfinite logits");
                if exact {assert_eq!(after,ready_jit,"post-ready installation at step {}",i+1);}
                if graph {
                    let seen=!captured_tokens.insert(e.last.bucket_tokens);
                    assert_eq!(e.last.replayed,seen,"new bucket captures; all subsequent uses replay");
                    if run==1 {assert!(e.last.replayed); assert!(e.captured_bucket_count()>0);}
                }
                if !graph {assert!(!e.last.replayed);assert_eq!(e.captured_bucket_count(),0);}
                if let Some(prev)=&previous {assert_eq!(&rows,prev,"same graph case capture/replay bits");}
                previous=Some(rows);
            }
            let rows=previous.unwrap();
            if let Some(j)=case.repeat {
                let same=rows==results[j];
                out.put(json!({"kind":"restoration","mode":mode,"step":i+1,"original_step":j+1,"bits_equal":same}));
                assert!(same,"interleaved restoration {mode} step {}",i+1);
            }
            results.push(rows);
        }
        let map:BTreeMap<_,_>=row_keys(&c[16]).into_iter().enumerate().map(|(i,k)|(k,i)).collect();
        let reordered:Rows=row_keys(&c[15]).iter().map(|k|results[16][map[k]].clone()).collect();
        compare(&format!("{mode}:AB/BA"),&results[15],&reordered,lim,&mut out);
        modes.push(results);
        drop(e); // All device weights/arenas/context dropped BEFORE loading the next model instance.
        out.put(json!({"kind":"dropped","mode":mode}));
    }
    for (a,b) in [(0,1),(0,2),(1,2)] {
        let (mut all_a,mut all_b)=(vec![],vec![]);
        for i in 0..18 {compare(&format!("modes{a}-{b}:step{}",i+1),&modes[a][i],&modes[b][i],lim,&mut out);
            all_a.extend(modes[a][i].clone());all_b.extend(modes[b][i].clone());}
        compare(&format!("modes{a}-{b}:all"),&all_a,&all_b,lim,&mut out);
    }
    out.put(json!({"kind":"final","ok":true,"modes":3,"steps_per_mode":18,"wall_ms":start.elapsed().as_secs_f64()*1000.}));
}
