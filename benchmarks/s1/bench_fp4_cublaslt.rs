//! C1: cuBLASLt's NVFP4 block-scaled GEMM (FP4 e2m1 A/B, UE4M3 scales per 16 elements, bf16 out) against the engine's
//! cuTile FP4 GEMMs at elephant-1's MLP shapes (docs/C1.md).
//!
//!     scripts/gpu.sh ./target/release/bench_fp4_cublaslt --probe                       support status only
//!     scripts/gpu.sh ./target/release/bench_fp4_cublaslt --tokens 1024,4096,16384,32768 timing, whole MLP per layer
//!     scripts/gpu.sh ./target/release/bench_fp4_cublaslt --check                       ulps vs f64, batch invariance
//!
//! Layout: the engine's `y [T, N] = x [T, K] · W [N, K]ᵀ` is issued column-major as `yᵀ = W · xᵀ` (A = W with op T,
//! B = x with op N, the "TN" form cuBLASLt requires for FP4). cuBLASLt wants each block-scale tensor in its tiled
//! layout (128 rows x 4 scale columns per 512-byte block, `nvfp4::lt_scale_offset`), not the row-major `[rows, K/16]` the
//! checkpoint and the engine's quantize kernels produce: the weights' scales are re-laid out once (load time), the
//! activations' scales by a small cuTile kernel per GEMM (`nvfp4::scales_to_lt_op`, the engine's, timed and counted below).
//! Epilogue: `D = alpha · Σ (q_x s_x)(q_w s_w)`, rounded once to bf16, with `alpha = 1/(igs·wgs)` per output column
//! (gate and up halves differ: a device alpha vector, `CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO`, or two
//! calls). cuBLASLt has no SiLU·mul epilogue: its gate/up path writes bf16 `[T, 24576]` and then runs the engine's
//! `silu_mul_nvfp4` (F1-tuned), which the cuTile path fuses into its GEMM.

use anyhow::{Result, anyhow, bail, ensure};
use clap::Parser;
use cudarc::cublas::sys as cublas_sys;
use cudarc::cublaslt::{result as lt, sys};
use cutile::cuda_async::device_operation::ExecutionContext;
use cutile::cuda_core::{Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use rayon::prelude::*;
use std::ffi::c_void;
use std::time::Instant;
use sysone::kernels::nvfp4::{
    self, CheckpointProj, GemmTiles, Nvfp4Act, Nvfp4Linear, QuantTiles, RawAlpha, SiluTiles, lt_scale_bytes, to_lt_scales,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    allow_shared_gpu: bool,
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384,32768")]
    tokens: Vec<usize>,
    #[arg(long, default_value_t = 10)]
    iters: usize,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Only report whether cuBLASLt supports the FP4 GEMM here (heuristic results per shape).
    #[arg(long)]
    probe: bool,
    /// Correctness (ulps vs the exact dequantized product, vs cuTile) and batch invariance instead of timing.
    #[arg(long)]
    check: bool,
    /// Algorithms requested from the heuristic.
    #[arg(long, default_value_t = 64)]
    max_algos: usize,
    /// cuBLASLt workspace, MiB.
    #[arg(long, default_value_t = 64)]
    workspace_mib: usize,
    /// Down projection: also pin this algorithm `id:tile:custom` (taken from the heuristic at 16k rows) and report
    /// the whole MLP with it (the engine's sm_121 preference; tile 23 = 128x256).
    #[arg(long, default_value = "67:23:23")]
    down_pin: String,
}

const K: usize = 4096;
const I: usize = 12288;
const PINNED_REF_ROWS: usize = 16384;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Host formats.
// ------------------------------------------------------------------------------------------------------------------

const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1(n: u8) -> f64 {
    let v = E2M1[(n & 7) as usize];
    if n & 8 != 0 { -v } else { v }
}

fn e4m3(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f64;
    if e == 15 && (b & 7) == 7 {
        return f64::NAN;
    }
    if e == 0 { sign * m / 8.0 * 2f64.powi(-6) } else { sign * (1.0 + m / 8.0) * 2f64.powi(e - 7) }
}

fn bf16_rne(v: f64) -> f32 {
    bf16::from_f32(v as f32).to_f32() // f64 → f32 → bf16: double rounding is harmless for the ulp histograms here
}

/// Signed distance in bf16 ulps (monotone integer mapping of the bit patterns).
fn bf16_ulps(a: f32, b: f32) -> i64 {
    let key = |v: f32| {
        let u = bf16::from_f32(v).to_bits() as i64;
        if u & 0x8000 != 0 { -(u & 0x7fff) } else { u }
    };
    key(a) - key(b)
}

// ------------------------------------------------------------------------------------------------------------------
// Device buffers (raw, for cuBLASLt).
// ------------------------------------------------------------------------------------------------------------------

fn up_u8(v: Vec<u8>, stream: &Arc<Stream>) -> Result<Arc<Tensor<u8>>> {
    let n = v.len();
    Ok(api::copy_host_vec_to_device(&Arc::new(v)).reshape(&[n]).sync_on(stream)?.into())
}

fn dptr<T: cutile::cuda_core::DType>(t: &Tensor<T>) -> u64 {
    t.device_pointer().dptr
}

fn down_bf16(t: &Arc<Tensor<bf16>>, stream: &Arc<Stream>) -> Result<Vec<f32>> {
    let v: Vec<bf16> = t.clone().to_host_vec().sync_on(stream)?;
    Ok(v.iter().map(|x| x.to_f32()).collect())
}

// ------------------------------------------------------------------------------------------------------------------
// cuBLASLt FP4 GEMM.
// ------------------------------------------------------------------------------------------------------------------

fn lt_err(what: &'static str) -> impl Fn(lt::CublasError) -> anyhow::Error {
    move |e| anyhow!("{what}: {e:?}")
}

#[derive(Clone, Copy)]
struct AlgoInfo {
    algo: sys::cublasLtMatmulAlgo_t,
    ws: usize,
    waves: f32,
    id: u32,
    tile: u32,
    stages: u32,
    splitk: u32,
    red: u32,
    swizzle: u32,
    custom: u32,
}

impl std::fmt::Display for AlgoInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "algo {:>3} tile {:>3} stages {:>3} splitK {} red {} swz {} cust {} ws {:>8} waves {:.2}", self.id,
               self.tile, self.stages, self.splitk, self.red, self.swizzle, self.custom, self.ws, self.waves)
    }
}

fn cfg_u32(algo: &sys::cublasLtMatmulAlgo_t, attr: sys::cublasLtMatmulAlgoConfigAttributes_t) -> u32 {
    let mut v = 0u32;
    let mut written = 0usize;
    let st = unsafe { sys::cublasLtMatmulAlgoConfigGetAttribute(algo, attr, (&mut v as *mut u32).cast(), 4, &mut written) };
    if st.result().is_err() { u32::MAX } else { v }
}

fn info(r: &sys::cublasLtMatmulHeuristicResult_t) -> AlgoInfo {
    use sys::cublasLtMatmulAlgoConfigAttributes_t as C;
    let a = &r.algo;
    AlgoInfo {
        algo: r.algo,
        ws: r.workspaceSize,
        waves: r.wavesCount,
        id: cfg_u32(a, C::CUBLASLT_ALGO_CONFIG_ID),
        tile: cfg_u32(a, C::CUBLASLT_ALGO_CONFIG_TILE_ID),
        stages: cfg_u32(a, C::CUBLASLT_ALGO_CONFIG_STAGES_ID),
        splitk: cfg_u32(a, C::CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
        red: cfg_u32(a, C::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
        swizzle: cfg_u32(a, C::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
        custom: cfg_u32(a, C::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
    }
}

/// One FP4 GEMM problem `y [T, N] (row stride ldy) = alpha ⊙ (x [T, K] · W [N, K]ᵀ)`.
struct LtGemm {
    handle: sys::cublasLtHandle_t,
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    d: sys::cublasLtMatrixLayout_t,
    n: usize,
    k: usize,
    t: usize,
    ldy: usize,
    alpha_vec: bool,
}

impl Drop for LtGemm {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matrix_layout(self.a);
            let _ = lt::destroy_matrix_layout(self.b);
            let _ = lt::destroy_matrix_layout(self.d);
            let _ = lt::destroy_matmul_desc(self.desc);
        }
    }
}

impl LtGemm {
    fn new(handle: sys::cublasLtHandle_t, n: usize, k: usize, t: usize, ldy: usize, alpha_vec: bool) -> Result<Self> {
        use sys::cublasLtMatmulDescAttributes_t as D;
        let desc = lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType::CUDA_R_32F)
            .map_err(lt_err("cublasLtMatmulDescCreate"))?;
        let fp4 = sys::cudaDataType::CUDA_R_4F_E2M1;
        let a = lt::create_matrix_layout(fp4, k as u64, n as u64, k as i64).map_err(lt_err("layout A"))?;
        let b = lt::create_matrix_layout(fp4, k as u64, t as u64, k as i64).map_err(lt_err("layout B"))?;
        let d = lt::create_matrix_layout(sys::cudaDataType::CUDA_R_16BF, n as u64, t as u64, ldy as i64)
            .map_err(lt_err("layout D"))?;
        let g = Self { handle, desc, a, b, d, n, k, t, ldy, alpha_vec };
        unsafe {
            let ta = cublas_sys::cublasOperation_t::CUBLAS_OP_T;
            let tb = cublas_sys::cublasOperation_t::CUBLAS_OP_N;
            let sz = std::mem::size_of::<cublas_sys::cublasOperation_t>();
            lt::set_matmul_desc_attribute(desc, D::CUBLASLT_MATMUL_DESC_TRANSA, &ta as *const _ as *const c_void, sz)
                .map_err(lt_err("TRANSA"))?;
            lt::set_matmul_desc_attribute(desc, D::CUBLASLT_MATMUL_DESC_TRANSB, &tb as *const _ as *const c_void, sz)
                .map_err(lt_err("TRANSB"))?;
            let mode = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 as i32;
            for attr in [D::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, D::CUBLASLT_MATMUL_DESC_B_SCALE_MODE] {
                lt::set_matmul_desc_attribute(desc, attr, (&mode as *const i32).cast(), 4).map_err(lt_err("SCALE_MODE"))?;
            }
            if alpha_vec {
                let pm = 3i32; // CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO
                lt::set_matmul_desc_attribute(desc, D::CUBLASLT_MATMUL_DESC_POINTER_MODE, (&pm as *const i32).cast(), 4)
                    .map_err(lt_err("POINTER_MODE"))?;
            }
        }
        Ok(g)
    }

    /// Set the scale pointers (device, cuBLASLt tiled layout) before a heuristic query or a launch.
    fn set_scales(&self, w_scales: u64, x_scales: u64) -> Result<()> {
        use sys::cublasLtMatmulDescAttributes_t as D;
        unsafe {
            lt::set_matmul_desc_attribute(self.desc, D::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                          (&w_scales as *const u64).cast(), 8)
                .map_err(lt_err("A_SCALE_POINTER"))?;
            lt::set_matmul_desc_attribute(self.desc, D::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                          (&x_scales as *const u64).cast(), 8)
                .map_err(lt_err("B_SCALE_POINTER"))?;
        }
        Ok(())
    }

    fn heuristic(&self, ws: usize, max: usize) -> Result<(Vec<AlgoInfo>, Vec<sys::cublasStatus_t>)> {
        use sys::cublasLtMatmulPreferenceAttributes_t as P;
        let pref = lt::create_matmul_pref().map_err(lt_err("pref"))?;
        let mut res = vec![
            sys::cublasLtMatmulHeuristicResult_t {
                algo: sys::cublasLtMatmulAlgo_t { data: [0; 8] },
                workspaceSize: 0,
                state: sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                wavesCount: 0.0,
                reserved: [0; 4],
            };
            max
        ];
        let mut found = 0i32;
        let r = unsafe {
            let w = ws as u64;
            lt::set_matmul_pref_attribute(pref, P::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, (&w as *const u64).cast(), 8)
                .map_err(lt_err("MAX_WORKSPACE"))?;
            let r = sys::cublasLtMatmulAlgoGetHeuristic(self.handle, self.desc, self.a, self.b, self.d, self.d, pref,
                                                        max as i32, res.as_mut_ptr(), &mut found);
            let _ = lt::destroy_matmul_pref(pref);
            r
        };
        r.result().map_err(|e| anyhow!("cublasLtMatmulAlgoGetHeuristic: {e:?}"))?;
        let res = &res[..found.max(0) as usize];
        Ok((res.iter().filter(|r| r.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS).map(info).collect(),
            res.iter().map(|r| r.state).collect()))
    }

    /// `cublasLtMatmulAlgoCheck` of `algo` on this problem: Ok(workspace bytes).
    fn check(&self, algo: &sys::cublasLtMatmulAlgo_t) -> Result<usize> {
        let mut res = std::mem::MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::zeroed();
        unsafe {
            sys::cublasLtMatmulAlgoCheck(self.handle, self.desc, self.a, self.b, self.d, self.d, algo, res.as_mut_ptr())
                .result()
                .map_err(|e| anyhow!("cublasLtMatmulAlgoCheck: {e:?}"))?;
            Ok(res.assume_init().workspaceSize)
        }
    }

    /// Enqueue `y = alpha ⊙ x·Wᵀ`. `alpha` is a host scalar or (alpha_vec) a device vector of N f32.
    #[allow(clippy::too_many_arguments)]
    fn run(&self, stream: &Stream, algo: &sys::cublasLtMatmulAlgo_t, w: u64, x: u64, y: u64, alpha_host: f32,
           alpha_dev: u64, ws: (u64, usize)) -> Result<()> {
        let beta = 0.0f32;
        let alpha_ptr: *const c_void =
            if self.alpha_vec { alpha_dev as usize as *const c_void } else { (&alpha_host as *const f32).cast() };
        unsafe {
            lt::matmul(self.handle, self.desc, alpha_ptr, (&beta as *const f32).cast(), w as usize as *const c_void,
                       self.a, x as usize as *const c_void, self.b, y as usize as *const c_void, self.d,
                       y as usize as *mut c_void, self.d, algo, ws.0 as usize as *mut c_void, ws.1,
                       stream.cu_stream() as sys::cudaStream_t)
                .map_err(|e| anyhow!("cublasLtMatmul (t={}, n={}, k={}, ldy={}): {e:?}", self.t, self.n, self.k, self.ldy))
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Problem data.
// ------------------------------------------------------------------------------------------------------------------

/// A random NVFP4 weight `[n, k]` in the checkpoint layout (bytes like bench_nvfp4_mlp's).
fn rand_weight(n: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, Vec<u8>) {
    let p = (0..n * k / 2).map(|_| rng.next() as u8).collect();
    let s = (0..n * k / 16).map(|_| ((4 + rng.next() % 7) << 3 | (rng.next() & 7)) as u8).collect();
    (p, s)
}

/// Device copies of one projection for both GEMMs: the cuTile `Nvfp4Linear` and cuBLASLt's (same packed bytes, tiled
/// scales, alpha vector).
struct Weight {
    host: nvfp4::PreparedNvfp4,
    lin: Nvfp4Linear,
    lt_scales: Arc<Tensor<u8>>,
}

fn weight(parts: &[(Vec<u8>, Vec<u8>, f32)], n_each: usize, k: usize, igs: f32, stream: &Arc<Stream>) -> Result<Weight> {
    let projs: Vec<CheckpointProj> = parts
        .iter()
        .map(|(p, s, wgs)| CheckpointProj {
            n: n_each,
            k,
            weight_packed: p,
            weight_scale: s,
            weight_global_scale: *wgs,
            input_global_scale: igs,
        })
        .collect();
    let host = nvfp4::prepare_weights(&projs)?;
    let lin = Nvfp4Linear::upload(&host, stream)?;
    let lt_scales = up_u8(to_lt_scales(&host.scales, host.n, k / 16), stream)?;
    Ok(Weight { host, lin, lt_scales })
}

/// Quantized activations: host bytes, cuTile tensors, and the tiled scales for cuBLASLt (device copy).
struct Act {
    packed: Vec<u8>,
    scales: Vec<u8>,
    xp: Arc<Tensor<f4e2m1fnx2>>,
    xs: Arc<Tensor<f8e4m3fn>>,
    lt_scales: Arc<Tensor<u8>>,
}

fn act_from_bytes(t: usize, k: usize, packed: Vec<u8>, scales: Vec<u8>, stream: &Arc<Stream>) -> Result<Act> {
    let pv = Arc::new(packed.iter().map(|&b| f4e2m1fnx2::from_bits(b)).collect::<Vec<_>>());
    let sv = Arc::new(scales.iter().map(|&b| f8e4m3fn(b)).collect::<Vec<_>>());
    let xp = api::copy_host_vec_to_device(&pv).reshape(&[t, k / 2]).sync_on(stream)?.into();
    let xs = api::copy_host_vec_to_device(&sv).reshape(&[t, k / 16]).sync_on(stream)?.into();
    let lt_scales = up_u8(to_lt_scales(&scales, t, k / 16), stream)?;
    Ok(Act { packed, scales, xp, xs, lt_scales })
}

/// Random bf16 rows quantized by the engine's own kernel (the engine's activation distribution of scales).
fn rand_act(t: usize, k: usize, igs: f32, rng: &mut Rng, stream: &Arc<Stream>) -> Result<Act> {
    let xh: Vec<bf16> = (0..t * k).map(|_| bf16::from_f32(((rng.next() >> 40) as f32 / 16777216.0 - 0.5) * 4.0)).collect();
    let x: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(xh)).reshape(&[t, k]).sync_on(stream)?.into();
    let a = nvfp4::quantize(&x, igs, Nvfp4Act::zeros(t, k, stream)?, QuantTiles::DEFAULT, stream)?;
    let packed: Vec<u8> = a.packed.to_host_vec().sync_on(stream)?.iter().map(|v| v.to_bits()).collect();
    let scales: Vec<u8> = a.scales.to_host_vec().sync_on(stream)?.iter().map(|v| v.0).collect();
    act_from_bytes(t, k, packed, scales, stream)
}

// ------------------------------------------------------------------------------------------------------------------
// Timing.
// ------------------------------------------------------------------------------------------------------------------

fn time_ms(stream: &Arc<Stream>, iters: usize, reps: usize, mut launch: impl FnMut(&ExecutionContext) -> Result<()>)
           -> Result<f64> {
    let mut best = f64::MAX;
    for rep in 0..=reps {
        let ctx = ExecutionContext::new(stream.clone());
        let n = if rep == 0 { 1 } else { iters };
        let t0 = Instant::now();
        for _ in 0..n {
            launch(&ctx)?;
        }
        stream.synchronize()?;
        if rep > 0 {
            best = best.min(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
        }
    }
    Ok(best)
}

fn tflops(t: usize, n: usize, k: usize, ms: f64) -> f64 {
    2.0 * (t * n * k) as f64 / (ms * 1e-3) / 1e12
}

// ------------------------------------------------------------------------------------------------------------------

struct Ctx {
    stream: Arc<Stream>,
    handle: sys::cublasLtHandle_t,
    ws: Arc<Tensor<u8>>,
    ws_bytes: usize,
    sms: usize,
    args: Args,
}

fn status_str(s: &[sys::cublasStatus_t]) -> String {
    let mut v: Vec<String> = s.iter().map(|x| format!("{x:?}")).collect();
    v.dedup();
    v.join(",")
}

fn probe(c: &Ctx) -> Result<()> {
    let mut rng = Rng(7);
    let dummy = up_u8(vec![0u8; lt_scale_bytes(PINNED_REF_ROWS.max(2 * I), I / 16)], &c.stream)?;
    let alpha = up_u8(vec![0u8; 4 * 2 * I], &c.stream)?;
    let _ = (&mut rng, &alpha);
    println!("# probe: cuBLASLt {} (FP4 e2m1 A/B, VEC16_UE4M3 scales, fp32 compute, bf16 D)", lt_version());
    for (name, n, k, ldy) in [("gate/up", 2 * I, K, 2 * I), ("gate|up half", I, K, 2 * I), ("down", K, I, K)] {
        for &t in &[1024usize, 16384] {
            for av in [false, true] {
                let g = LtGemm::new(c.handle, n, k, t, ldy, av)?;
                g.set_scales(dptr(&dummy), dptr(&dummy))?;
                match g.heuristic(c.ws_bytes, c.args.max_algos) {
                    Ok((ok, st)) => {
                        println!("{name:<13} N={n:<6} K={k:<6} T={t:<6} alpha {}: {} candidates ({} usable) [{}]",
                                 if av { "vector" } else { "scalar" }, st.len(), ok.len(), status_str(&st));
                        for a in ok.iter().take(if t == 16384 && !av { 64 } else { 0 }) {
                            println!("    {a}");
                        }
                    }
                    Err(e) => println!("{name:<13} N={n:<6} K={k:<6} T={t:<6} alpha {}: {e:#}",
                                       if av { "vector" } else { "scalar" }),
                }
            }
        }
    }
    Ok(())
}

fn lt_version() -> String {
    let v = unsafe { sys::cublasLtGetVersion() };
    format!("{}.{}.{}", v / 10000, v / 100 % 100, v % 100)
}

/// First candidate without split-K / reduction (the L1 pinning rule), from the heuristic at PINNED_REF_ROWS.
fn pinned(list: &[AlgoInfo]) -> Option<AlgoInfo> {
    list.iter().copied().find(|a| a.splitk <= 1 && a.red == 0)
}

fn bench(c: &Ctx) -> Result<()> {
    let s = &c.stream;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (igs1, igs2) = (628.0f32, 233.0f32);
    let (gp, gs) = rand_weight(I, K, &mut rng);
    let (upp, ups) = rand_weight(I, K, &mut rng);
    let (dp, ds) = rand_weight(K, I, &mut rng);
    let gu = weight(&[(gp, gs, 9408.0), (upp, ups, 4704.0)], I, K, igs1, s)?;
    let dn = weight(&[(dp, ds, 7000.0)], K, I, igs2, s)?;
    let alpha_gu: Arc<Tensor<f32>> = gu.lin.alpha.clone();
    let ws = (dptr(&c.ws), c.ws_bytes);
    println!("# cuBLASLt {}; times are ms per launch, best of {} batches of {}", lt_version(), c.args.reps, c.args.iters);
    let mut summary = Vec::new();
    let mut summary2 = Vec::new();
    for &t in &c.args.tokens {
        println!("\n## T = {t}");
        let a1 = rand_act(t, K, igs1, &mut rng, s)?;
        let a2 = rand_act(t, I, igs2, &mut rng, s)?;
        let y_gu: Arc<Tensor<bf16>> = api::zeros::<bf16>(&[t, 2 * I]).sync_on(s)?.into();
        let y_dn: Arc<Tensor<bf16>> = api::zeros::<bf16>(&[t, K]).sync_on(s)?.into();

        // --- cuTile ---
        let g_raw = time_ms(s, c.args.iters, c.args.reps, |ctx| {
            unsafe { nvfp4::gemm_op(&a1.xp, &a1.xs, &gu.lin, y_gu.clone(), GemmTiles::DEFAULT, c.sms)?.execute(ctx)? };
            Ok(())
        })?;
        let d_raw = time_ms(s, c.args.iters, c.args.reps, |ctx| {
            unsafe { nvfp4::gemm_op(&a2.xp, &a2.xs, &dn.lin, y_dn.clone(), GemmTiles::DEFAULT, c.sms)?.execute(ctx)? };
            Ok(())
        })?;
        let q2 = Nvfp4Act::zeros(t, I, s)?;
        let (qa, sa): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) = (q2.packed.into(), q2.scales.into());
        let alpha = RawAlpha { lo: gu.lin.alpha_lo, hi: gu.lin.alpha_hi, split_n: gu.lin.split_n };
        let swiglu_opts = cutile::tile_kernel::CompileOptions { occupancy: Some(2), ..Default::default() };
        let g_fused = time_ms(s, c.args.iters, c.args.reps, |ctx| {
            unsafe {
                nvfp4::gemm_swiglu_il_op_with(&a1.xp, &a1.xs, &gu.lin.packed, &gu.lin.scales, alpha, igs2, qa.clone(),
                                              sa.clone(), GemmTiles::DEFAULT, c.sms, swiglu_opts.clone())?
                    .execute(ctx)?
            };
            Ok(())
        })?;
        drop((qa, sa));
        let q2 = Nvfp4Act::zeros(t, I, s)?;
        let (mut q, mut qs) = (q2.packed, q2.scales);
        let silu_t = SiluTiles { rows: 1, cols: nvfp4::SILU_Q_BLOCK, occupancy: Some(3) };
        let silu = time_ms(s, c.args.iters, c.args.reps, |ctx| {
            unsafe { nvfp4::silu_mul_nvfp4_with(&mut q, &mut qs, &y_gu, igs2, silu_t)?.execute(ctx)? };
            Ok(())
        })?;
        println!("cuTile  gate/up raw          {g_raw:>8.3} ms {:>6.1} TFLOP/s", tflops(t, 2 * I, K, g_raw));
        println!("cuTile  gate/up fused SwiGLU {g_fused:>8.3} ms {:>6.1} TFLOP/s (engine)", tflops(t, 2 * I, K, g_fused));
        println!("cuTile  down raw             {d_raw:>8.3} ms {:>6.1} TFLOP/s (engine)", tflops(t, K, I, d_raw));
        println!("silu_mul_nvfp4 (F1 tuning)   {silu:>8.3} ms");

        // --- scale re-layout kernels ---
        let mut relay = [0.0f64; 2];
        for (i, (sc, cols)) in [(&a1.scales, K / 16), (&a2.scales, I / 16)].into_iter().enumerate() {
            let sv = Arc::new(sc.iter().map(|&b| f8e4m3fn(b)).collect::<Vec<_>>());
            let src: Arc<Tensor<f8e4m3fn>> = api::copy_host_vec_to_device(&sv).reshape(&[t, cols]).sync_on(s)?.into();
            let mut out = Some(api::zeros::<f8e4m3fn>(&[lt_scale_bytes(t, cols)]).sync_on(s)?);
            relay[i] = time_ms(s, c.args.iters, c.args.reps, |ctx| {
                let o = out.take().unwrap();
                out = Some(unsafe { nvfp4::scales_to_lt_op(src.clone(), o)?.execute(ctx)? });
                Ok(())
            })?;
            // The device re-layout must equal the host one (the layout cuBLASLt is fed in the timing runs).
            let dev: Vec<u8> = out.take().unwrap().to_host_vec().sync_on(s)?.iter().map(|v| v.0).collect();
            ensure!(dev == to_lt_scales(sc, t, cols), "device scale re-layout differs from the host one");
        }
        println!("scale re-layout [T,{}] {:.3} ms, [T,{}] {:.3} ms (device == host layout)", K / 16, relay[0], I / 16,
                 relay[1]);

        // --- cuBLASLt: every algorithm of the heuristic ---
        let mut best = [f64::MAX; 3];
        let mut pin_ms = [f64::NAN; 3];
        let mut pref_ms = f64::NAN;
        let cases: [(&str, usize, usize, usize, bool); 3] =
            [("gate/up 1 call (timing only: one alpha for both halves)", 2 * I, K, 2 * I, false),
             ("gate/up 2 calls (exact: alpha per half)", I, K, 2 * I, false), ("down", K, I, K, false)];
        for (ci, &(name, n, k, ldy, av)) in cases.iter().enumerate() {
            let (w, a, y) = if ci < 2 { (&gu, &a1, &y_gu) } else { (&dn, &a2, &y_dn) };
            let g = LtGemm::new(c.handle, n, k, t, ldy, av)?;
            g.set_scales(dptr(&w.lt_scales), dptr(&a.lt_scales))?;
            // The pinned algorithm (L1 rule) comes from the heuristic at PINNED_REF_ROWS rows.
            let gref = LtGemm::new(c.handle, n, k, PINNED_REF_ROWS, ldy, av)?;
            gref.set_scales(dptr(&w.lt_scales), dptr(&a.lt_scales))?;
            let ref_list = gref.heuristic(c.ws_bytes, c.args.max_algos)?.0;
            let pin = pinned(&ref_list);
            let want: Vec<u32> = c.args.down_pin.split(':').map(|v| v.parse().unwrap()).collect();
            let pref = if ci == 2 {
                ref_list.iter().copied().find(|a| a.id == want[0] && a.tile == want[1] && a.custom == want[2]
                                                && a.splitk <= 1 && a.red == 0)
            } else {
                None
            };
            let (list, st) = g.heuristic(c.ws_bytes, c.args.max_algos)?;
            println!("cuBLASLt {name}: {} candidates, {} usable [{}]", st.len(), list.len(), status_str(&st));
            let half_bytes = (I * k / 2) as u64;
            let half_scale_bytes = (lt_scale_bytes(I, k / 16)) as u64;
            let mut timed: Vec<(AlgoInfo, f64)> = Vec::new();
            let mut cands = list.clone();
            for p in [pin, pref].into_iter().flatten() {
                if !cands.iter().any(|a| a.algo.data == p.algo.data) {
                    cands.push(p);
                }
            }
            for al in &cands {
                if g.check(&al.algo).map(|need| need > c.ws_bytes).unwrap_or(true) {
                    println!("    {al}: AlgoCheck refuses T={t}");
                    continue;
                }
                let r = time_ms(s, c.args.iters, c.args.reps, |_| {
                    if ci == 1 {
                        // gate then up, each [T, 12288] with its scalar alpha, into the two column halves.
                        g.set_scales(dptr(&w.lt_scales), dptr(&a.lt_scales))?;
                        g.run(s, &al.algo, dptr(&w.lin.packed), dptr(&a.xp), dptr(y), w.host.alpha_lo, 0, ws)?;
                        g.set_scales(dptr(&w.lt_scales) + half_scale_bytes, dptr(&a.lt_scales))?;
                        g.run(s, &al.algo, dptr(&w.lin.packed) + half_bytes, dptr(&a.xp), dptr(y) + 2 * I as u64,
                              w.host.alpha_hi, 0, ws)?;
                    } else {
                        g.run(s, &al.algo, dptr(&w.lin.packed), dptr(&a.xp), dptr(y), w.host.alpha_lo,
                              dptr(&alpha_gu), ws)?;
                    }
                    Ok(())
                });
                match r {
                    Ok(ms) => timed.push((*al, ms)),
                    Err(e) => println!("    {al}: {e:#}"),
                }
            }
            let ntot = if ci == 1 { 2 * I } else { n };
            for (al, ms) in &timed {
                let is_pin = pin.map(|p| p.algo.data == al.algo.data).unwrap_or(false);
                println!("    {al}: {ms:>8.3} ms {:>6.1} TFLOP/s{}", tflops(t, ntot, k, *ms),
                         if is_pin { "  <- pinned (L1 rule, heuristic at 16k)" } else { "" });
                best[ci] = best[ci].min(*ms);
                if is_pin {
                    pin_ms[ci] = *ms;
                }
                if pref.map(|p| p.algo.data == al.algo.data).unwrap_or(false) {
                    pref_ms = *ms;
                    println!("      ^ preferred down algorithm (--down-pin {})", c.args.down_pin);
                }
            }
            if let Some(p) = pin {
                if !timed.iter().any(|(a, _)| a.algo.data == p.algo.data) {
                    println!("    pinned {p}: not runnable at T={t}");
                }
            } else {
                println!("    no candidate without split-K at {PINNED_REF_ROWS} rows");
            }
        }
        let gu_lt = best[1]; // the exact path (alpha per half); best[0] is reported for information
        let gu_lt_pin = pin_ms[1];
        let cutile_mlp = g_fused + d_raw;
        let lt_mlp = gu_lt + silu + best[2] + relay[0] + relay[1];
        let lt_mlp_pin = gu_lt_pin + silu + pin_ms[2] + relay[0] + relay[1];
        let lt_mlp_fusedrel = gu_lt + silu + best[2];
        let hybrid = g_fused + best[2] + relay[1];
        let hybrid_pin = g_fused + pin_ms[2] + relay[1];
        println!("\nwhole MLP per layer (GEMMs + SiLU·mul/quantize + re-layouts; the norm+quantize is common to all):");
        println!("  cuTile (engine): fused gate/up {g_fused:.3} + down {d_raw:.3} = {cutile_mlp:.3} ms");
        println!("  cuBLASLt best:   gate/up {gu_lt:.3} + silu_mul_nvfp4 {silu:.3} + down {:.3} + re-layouts {:.3} = {lt_mlp:.3} ms ({:+.1}%)",
                 best[2], relay[0] + relay[1], 100.0 * (lt_mlp / cutile_mlp - 1.0));
        println!("  cuBLASLt pinned: {lt_mlp_pin:.3} ms ({:+.1}%)", 100.0 * (lt_mlp_pin / cutile_mlp - 1.0));
        println!("  cuBLASLt best, re-layout fused into the producers: {lt_mlp_fusedrel:.3} ms ({:+.1}%)",
                 100.0 * (lt_mlp_fusedrel / cutile_mlp - 1.0));
        println!("  hybrid (cuTile fused gate/up + cuBLASLt down + re-layout): best {hybrid:.3} ms ({:+.1}%), pinned {hybrid_pin:.3} ms ({:+.1}%)",
                 100.0 * (hybrid / cutile_mlp - 1.0), 100.0 * (hybrid_pin / cutile_mlp - 1.0));
        let hybrid_pref = g_fused + pref_ms + relay[1];
        let hybrid_pref_fused = g_fused + pref_ms;
        let lt_pref = gu_lt_pin + silu + pref_ms + relay[0] + relay[1];
        println!("  preferred down algorithm: down {pref_ms:.3} ms; hybrid {hybrid_pref:.3} ms ({:+.1}%), hybrid with the re-layout fused {hybrid_pref_fused:.3} ms ({:+.1}%); all cuBLASLt (gate/up pinned) {lt_pref:.3} ms ({:+.1}%)",
                 100.0 * (hybrid_pref / cutile_mlp - 1.0), 100.0 * (hybrid_pref_fused / cutile_mlp - 1.0),
                 100.0 * (lt_pref / cutile_mlp - 1.0));
        summary.push((t, cutile_mlp, lt_mlp, lt_mlp_pin, hybrid, hybrid_pin, g_raw, g_fused, d_raw, gu_lt, best[2]));
        summary2.push((t, cutile_mlp, pref_ms, hybrid_pref, lt_pref));
    }
    println!("\n| T | cuTile gate/up raw | cuTile fused | cuTile down | cuBLASLt gate/up | cuBLASLt down | MLP cuTile | MLP cuBLASLt best | pinned | hybrid best | hybrid pinned |");
    for (t, cm, lm, lp, h, hp, gr, gf, dr, gl, dl) in summary {
        println!("| {t} | {gr:.3} ({:.0}) | {gf:.3} | {dr:.3} ({:.0}) | {gl:.3} ({:.0}) | {dl:.3} ({:.0}) | {cm:.3} | {lm:.3} ({:+.1}%) | {lp:.3} ({:+.1}%) | {h:.3} ({:+.1}%) | {hp:.3} ({:+.1}%) |",
                 tflops(t, 2 * I, K, gr), tflops(t, K, I, dr), tflops(t, 2 * I, K, gl), tflops(t, K, I, dl),
                 100.0 * (lm / cm - 1.0), 100.0 * (lp / cm - 1.0), 100.0 * (h / cm - 1.0), 100.0 * (hp / cm - 1.0));
    }
    println!("\n| T | MLP cuTile | cuBLASLt down (preferred) | hybrid (preferred down) | all cuBLASLt (pinned gate/up, preferred down) |");
    for (t, cm, pd, hp, lp) in summary2 {
        println!("| {t} | {cm:.3} | {pd:.3} ({:.0}) | {hp:.3} ({:+.1}%) | {lp:.3} ({:+.1}%) |", tflops(t, K, I, pd),
                 100.0 * (hp / cm - 1.0), 100.0 * (lp / cm - 1.0));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------
// Correctness and batch invariance.
// ------------------------------------------------------------------------------------------------------------------

/// Exact `alpha[n] · Σ_k e2m1·s_x·e2m1·s_w` in f64 for rows `rows` of `a`.
fn reference(a: &Act, w: &nvfp4::PreparedNvfp4, rows: usize) -> Vec<f64> {
    let (n, k) = (w.n, w.k);
    let deq = |packed: &[u8], scales: &[u8], r: usize| -> Vec<f64> {
        (0..k).map(|j| {
            let b = packed[r * k / 2 + j / 2];
            let nib = if j % 2 == 0 { b & 0xf } else { b >> 4 };
            e2m1(nib) * e4m3(scales[r * k / 16 + j / 16])
        }).collect()
    };
    let wd: Vec<Vec<f64>> = (0..n).into_par_iter().map(|c| deq(&w.packed, &w.scales, c)).collect();
    let xd: Vec<Vec<f64>> = (0..rows).map(|r| deq(&a.packed, &a.scales, r)).collect();
    (0..rows * n).into_par_iter().map(|i| {
        let (r, c) = (i / n, i % n);
        let acc: f64 = xd[r].iter().zip(&wd[c]).map(|(x, y)| x * y).sum();
        acc * w.alpha[c] as f64
    }).collect()
}

fn ulp_report(name: &str, got: &[f32], want: &[f32]) {
    let mut h = [0usize; 5];
    let mut max = 0i64;
    for (g, w) in got.iter().zip(want) {
        let u = bf16_ulps(*g, *w).abs();
        max = max.max(u);
        h[(u as usize).min(4)] += 1;
    }
    let n = got.len() as f64;
    println!("  {name:<38} n={:<8} 0 ulp {:>6.2}%  1 ulp {:>6.2}%  2 ulp {:>6.3}%  3 {:>6.3}%  >=4 {:>6.3}%  max {max}",
             got.len(), 100.0 * h[0] as f64 / n, 100.0 * h[1] as f64 / n, 100.0 * h[2] as f64 / n,
             100.0 * h[3] as f64 / n, 100.0 * h[4] as f64 / n);
}

/// Relative error of the bf16 outputs vs the exact f64 value, in units of 2^-8 (one bf16 ulp at the value).
fn rel_report(name: &str, got: &[f32], exact: &[f64]) {
    let mut v: Vec<f64> = got.iter().zip(exact).filter(|(_, e)| e.abs() > 1e-30)
        .map(|(g, e)| ((*g as f64 - e) / e).abs() * 256.0).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    println!("  {name:<38} |rel err| in 2^-8 units: p50 {:.3} p99 {:.3} max {:.3}", q(0.5), q(0.99), q(1.0));
}

fn check(c: &Ctx) -> Result<()> {
    let s = &c.stream;
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (igs1, igs2) = (628.0f32, 233.0f32);
    let (gp, gs) = rand_weight(I, K, &mut rng);
    let (upp, ups) = rand_weight(I, K, &mut rng);
    let (dp, ds) = rand_weight(K, I, &mut rng);
    let gu = weight(&[(gp, gs, 9408.0), (upp, ups, 4704.0)], I, K, igs1, s)?;
    let dn = weight(&[(dp, ds, 7000.0)], K, I, igs2, s)?;
    let ws = (dptr(&c.ws), c.ws_bytes);
    let big = 4096usize;
    let a1 = rand_act(big, K, igs1, &mut rng, s)?;
    let a2 = rand_act(big, I, igs2, &mut rng, s)?;
    println!("# check: cuBLASLt {}", lt_version());

    // cuBLASLt rejects a per-column alpha vector for FP4 (CUBLAS_STATUS_NOT_SUPPORTED), so gate/up runs as two calls
    // (gate, up), each [T, 12288] with its own scalar alpha, into the two column halves of y [T, 24576].
    for (name, w, a, n, k, halves) in [("gate/up", &gu, &a1, 2 * I, K, 2usize), ("down", &dn, &a2, K, I, 1)] {
        let nc = n / halves;
        // Pinned algorithm (L1 rule): the heuristic at 16k rows, first candidate without split-K.
        let gref = LtGemm::new(c.handle, nc, k, PINNED_REF_ROWS, n, false)?;
        gref.set_scales(dptr(&w.lt_scales), dptr(&a.lt_scales))?;
        let list = gref.heuristic(c.ws_bytes, c.args.max_algos)?.0;
        let nosplit: Vec<AlgoInfo> = list.iter().copied().filter(|a| a.splitk <= 1 && a.red == 0).collect();
        println!("\n## {name} [T, {k}] x [{n}, {k}]ᵀ ({halves} call(s) of N={nc}): {} candidates at 16k, {} without split-K",
                 list.len(), nosplit.len());
        let Some(pin) = nosplit.first().copied() else { bail!("no split-K-free algorithm for {name}") };
        println!("pinned: {pin}");

        // Run an activation (host bytes, rows `rows`) through cuBLASLt with algorithm `al`.
        let run_lt = |al: &AlgoInfo, act: &Act, rows: usize| -> Result<Vec<f32>> {
            let g = LtGemm::new(c.handle, nc, k, rows, n, false)?;
            let y: Arc<Tensor<bf16>> = api::zeros::<bf16>(&[rows, n]).sync_on(s)?.into();
            for h in 0..halves {
                g.set_scales(dptr(&w.lt_scales) + (h * lt_scale_bytes(nc, k / 16)) as u64, dptr(&act.lt_scales))?;
                let need = g.check(&al.algo)?;
                ensure!(need <= c.ws_bytes, "workspace");
                let alpha = if h == 0 { w.host.alpha_lo } else { w.host.alpha_hi };
                g.run(s, &al.algo, dptr(&w.lin.packed) + (h * nc * k / 2) as u64, dptr(&act.xp),
                      dptr(&y) + (h * nc * 2) as u64, alpha, 0, ws)?;
            }
            s.synchronize()?;
            down_bf16(&y, s)
        };
        let run_tile = |act: &Act, rows: usize| -> Result<Vec<f32>> {
            let y: Arc<Tensor<bf16>> = api::zeros::<bf16>(&[rows, n]).sync_on(s)?.into();
            let y = nvfp4::gemm_op(&act.xp, &act.xs, &w.lin, y, GemmTiles::DEFAULT, c.sms)?.sync_on(s)?;
            down_bf16(&y, s)
        };
        let sub = |act: &Act, start: usize, rows: usize| -> Result<Act> {
            act_from_bytes(rows, k, act.packed[start * k / 2..(start + rows) * k / 2].to_vec(),
                           act.scales[start * k / 16..(start + rows) * k / 16].to_vec(), s)
        };

        // 1. Accuracy on 64 rows vs the exact dequantized product.
        let rows = 64;
        let a64 = sub(a, 0, rows)?;
        let exact = reference(&a64, &w.host, rows);
        let exact_bf: Vec<f32> = exact.iter().map(|&v| bf16_rne(v)).collect();
        let y_tile = run_tile(&a64, rows)?;
        let y_pin = run_lt(&pin, &a64, rows)?;
        ulp_report("cuTile raw vs bf16(exact)", &y_tile, &exact_bf);
        ulp_report("cuBLASLt pinned vs bf16(exact)", &y_pin, &exact_bf);
        ulp_report("cuBLASLt pinned vs cuTile", &y_pin, &y_tile);
        rel_report("cuTile raw", &y_tile, &exact);
        rel_report("cuBLASLt pinned", &y_pin, &exact);
        for al in nosplit.iter().skip(1).take(6) {
            match run_lt(al, &a64, rows) {
                Ok(y) => ulp_report(&format!("algo {} tile {} stages {} vs cuTile", al.id, al.tile, al.stages), &y, &y_tile),
                Err(e) => println!("  algo {} tile {}: {e:#}", al.id, al.tile),
            }
        }
        for al in list.iter().filter(|a| a.splitk > 1 || a.red != 0).take(2) {
            match run_lt(al, &a64, rows) {
                Ok(y) => ulp_report(&format!("split-K {} algo {} vs cuTile", al.splitk, al.id), &y, &y_tile),
                Err(e) => println!("  split-K algo {}: {e:#}", al.id),
            }
        }

        // 2. Batch invariance of the pinned algorithm: rows [0, 64) at T = 64, 300, 1024, 4096, and rows [r, r+64)
        //    moved to the front of a call (shifts of 1, 3, 128, 1000 rows) vs the same rows inside the 4096-row call.
        let y_big = run_lt(&pin, a, big)?;
        let ref64 = &y_big[..rows * n];
        let mut lines = Vec::new();
        for &t in &[64usize, 300, 1024, 4096] {
            let yt = if t == big { y_big.clone() } else { run_lt(&pin, &sub(a, 0, t)?, t)? };
            let diff = yt[..rows * n].iter().zip(ref64).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            lines.push(format!("rows 0..64 at T={t}: {diff} of {} values differ from T={big}", rows * n));
        }
        for &shift in &[1usize, 3, 128, 1000] {
            let yt = run_lt(&pin, &sub(a, shift, 300)?, 300)?;
            let want = &y_big[shift * n..(shift + rows) * n];
            let diff = yt[..rows * n].iter().zip(want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            lines.push(format!("rows {shift}..{} moved to row 0 of a T=300 call: {diff} differ", shift + rows));
        }
        // Every other split-K-free algorithm too (whichever a pinning rule might pick).
        for al in nosplit.iter().skip(1) {
            let y4 = run_lt(al, a, big)?;
            let y1 = run_lt(al, &sub(a, 3, 300)?, 300)?;
            let diff = y1[..rows * n].iter().zip(&y4[3 * n..(3 + rows) * n]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            lines.push(format!("algo {} tile {}: rows 3..67 at T=300 vs T=4096: {diff} differ", al.id, al.tile));
        }
        for l in lines {
            println!("  {l}");
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("# {}", sysone::engine::platform::require_idle_gpu(args.allow_shared_gpu)?);
    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    let sms = device.multiprocessor_count()? as usize;
    let handle = lt::create_handle().map_err(|e| anyhow!("cublasLtCreate: {e:?}"))?;
    let ws_bytes = args.workspace_mib << 20;
    let ws = up_u8(vec![0u8; ws_bytes], &stream)?;
    let c = Ctx { stream, handle, ws, ws_bytes, sms, args };
    if c.args.probe {
        probe(&c)
    } else if c.args.check {
        check(&c)
    } else {
        bench(&c)
    }
}
