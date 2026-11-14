//! cuBLAS bf16 GEMM with fp32 compute (DESIGN §6.4, §6.6), ported from grout's `cublas.rs`.
//!
//! Layout is the engine's: row-major activations `x [T, K]` times row-major weights `w [N, K]`, giving
//! `y [T, N] = x · wᵀ`. cuBLAS is column-major, so this is issued as `yᵀ = w · xᵀ` (op_T on the weight,
//! op_N on the activations, m = N, n = T).
//!
//! Only cudarc's cuBLAS *sys/result* bindings are used: the stream comes from cuda-core and the
//! workspace is a [`DevBuf`], so no cudarc driver objects are created. A fixed workspace is set on the
//! handle so cuBLAS never allocates during CUDA-graph capture.
//!
//! Numerics: `CUBLAS_COMPUTE_32F` (no TF32, no fp16 accumulation) and
//! `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`, so split-K partial sums also stay fp32. The output
//! is rounded once, to bf16 or not at all (f32 output).
//!
//! Two modes ([`GemmMode`], L1):
//! - [`GemmMode::Heuristic`]: `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT`. cuBLAS picks the kernel, tile and
//!   split-K per call shape, so the same row can round differently at different T (batch sizes): the engine is
//!   then not batch-invariant (N1: 103/164 branches bit-identical packed vs alone).
//! - [`GemmMode::Pinned`]: cuBLASLt with **one algorithm per weight shape**, chosen once (cuBLASLt's heuristic at
//!   a fixed reference row count, [`PINNED_REF_ROWS`]) and reused for every T, restricted to algorithms without
//!   split-K (`SPLITK_NUM <= 1`, no reduction scheme). Each output element is then one fp32 dot product over K in
//!   an order fixed by the kernel's K tiling, independent of how many rows the call has or where the row sits,
//!   so a row's result depends only on that row (tests/engine_gemm.rs checks this bit for bit). Same numerics
//!   otherwise (fp32 compute and scale, one output rounding).
//!
//! C1 adds an NVFP4 GEMM on cuBLASLt ([`Fp4Gemm::CuBlasLt`], [`Blas::gemm_nvfp4_lt`], docs/C1.md): FP4 e2m1 A and B
//! with `VEC16_UE4M3` block scales in cuBLASLt's tiled scale layout, fp32 compute, a scalar alpha
//! `1/(igs·wgs)`, bf16 out; one algorithm per weight shape at every T, without split-K, as in pinned mode.

use crate::engine::graph::tensor::{DevBuf, DevTensor, GraphDType};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use cuda_core::Stream;
use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::cublaslt::{result as lt_result, sys as lt_sys};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

/// Workspace given to cuBLAS (enough for split-K on large GEMMs; cuBLAS recommends ≥ 32 MiB on Hopper+).
pub const CUBLAS_WORKSPACE_BYTES: usize = 64 << 20;

/// Row count (cuBLAS `n`) the pinned algorithm of a weight shape is chosen at: the 16k bucket, where the GEMMs
/// cost the most. The choice depends only on the weight shape, the strides and the output type, never on the
/// call's T, so every bucket (and every engine configuration) runs the same kernel per projection.
pub const PINNED_REF_ROWS: usize = 16384;

/// How [`Blas::gemm_bf16`] picks its kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum GemmMode {
    /// cuBLAS's default heuristic per call (`cublasGemmEx`); not batch-invariant.
    Heuristic,
    /// One cuBLASLt algorithm per weight shape at every T, no split-K: batch-invariant (the engine default).
    #[default]
    Pinned,
}

impl GemmMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::Pinned => "pinned",
        }
    }
}

impl std::str::FromStr for GemmMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "heuristic" => Ok(Self::Heuristic),
            "pinned" => Ok(Self::Pinned),
            other => Err(anyhow!("unknown GEMM mode {other:?} (pinned or heuristic)")),
        }
    }
}

/// Which kernel runs the NVFP4 engine's plain (non-fused) FP4 GEMMs, i.e. the MLP down projection (C1). The
/// gate/up GEMM with SiLU·mul + quantize in its epilogue (F1) stays a cuTile kernel either way: cuBLASLt has no
/// such epilogue, and its gate/up GEMM plus a separate `silu_mul_nvfp4` is slower (docs/C1.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Fp4Gemm {
    /// cuTile `gemm_nvfp4_raw` (W3c, the engine default).
    #[default]
    CuTile,
    /// cuBLASLt's block-scaled FP4 GEMM with one pinned algorithm per weight shape, after a re-layout of the
    /// activation scales into cuBLASLt's tiled layout.
    CuBlasLt,
}

impl Fp4Gemm {
    pub fn name(self) -> &'static str {
        match self {
            Self::CuTile => "cutile",
            Self::CuBlasLt => "cublaslt",
        }
    }
}

impl std::str::FromStr for Fp4Gemm {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "cutile" => Ok(Self::CuTile),
            "cublaslt" => Ok(Self::CuBlasLt),
            other => Err(anyhow!("unknown FP4 GEMM {other:?} (cutile or cublaslt)")),
        }
    }
}

/// Environment variable the engine reads the FP4 GEMM from when [`crate::engine::gpu::EngineOptions::fp4_gemm`]
/// is `None` (`cutile`, `cublaslt`; unset: cutile). Lets the engine tests and benchmarks run either kernel.
pub const FP4_GEMM_ENV: &str = "SYSONE_FP4_GEMM";

/// The FP4 GEMM in `$SYSONE_FP4_GEMM` (unset: `None`).
pub fn fp4_gemm_from_env() -> Result<Option<Fp4Gemm>> {
    match std::env::var(FP4_GEMM_ENV) {
        Ok(v) => v.parse().map(Some).with_context(|| format!("{FP4_GEMM_ENV}={v:?}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(e).context(FP4_GEMM_ENV),
    }
}

/// Preferred cuBLASLt FP4 algorithms per `(compute capability, N, K)`: `(algorithm id, tile id, custom option)`,
/// used when the heuristic at [`PINNED_REF_ROWS`] rows offers that algorithm without split-K; otherwise the first
/// split-K-free candidate (L1's rule). Measured with `bench_fp4_cublaslt` (docs/C1.md): on GB10 the down projection's
/// best is algorithm 67 with a 128x256 tile (id 23), custom option 23, 377-383 TFLOP/s at 16k-32k rows against
/// 346-350 for the heuristic's first candidate (128x128). A fixed table keeps the choice deterministic (no timing at
/// load), so every run and every bucket uses the same kernel.
const FP4_PREFERRED: &[((u32, u32), usize, usize, (u32, u32, u32))] = &[((12, 1), 4096, 12288, (67, 23, 23))];

/// The pinned FP4 algorithm of one weight shape `(N, K)` and its B / D layouts per row count.
struct Fp4Plan {
    desc: lt_sys::cublasLtMatmulDesc_t,
    a: lt_sys::cublasLtMatrixLayout_t,
    algo: lt_sys::cublasLtMatmulAlgo_t,
    /// (algorithm id, tile id, stages id, custom option), for reports.
    ids: (u32, u32, u32, u32),
    rows: HashMap<usize, (lt_sys::cublasLtMatrixLayout_t, lt_sys::cublasLtMatrixLayout_t)>,
}

struct Fp4State {
    handle: lt_sys::cublasLtHandle_t,
    plans: HashMap<(usize, usize), Fp4Plan>,
    /// The device's compute capability (selects [`FP4_PREFERRED`] entries).
    cc: (u32, u32),
    /// Activation scales of the current call in cuBLASLt's tiled layout (written by the re-layout kernel right
    /// before the GEMM; one buffer for every call, like the cuBLAS workspace: the forward's ops run in order on
    /// one stream).
    scratch: Arc<DevBuf>,
}

// SAFETY: as LtState.
unsafe impl Send for Fp4State {}

impl Drop for Fp4State {
    fn drop(&mut self) {
        unsafe {
            for p in self.plans.values() {
                for (b, d) in p.rows.values() {
                    let _ = lt_result::destroy_matrix_layout(*b);
                    let _ = lt_result::destroy_matrix_layout(*d);
                }
                let _ = lt_result::destroy_matrix_layout(p.a);
                let _ = lt_result::destroy_matmul_desc(p.desc);
            }
            let _ = lt_result::destroy_handle(self.handle);
        }
    }
}

impl Fp4State {
    fn layouts(n: usize, k: usize, t: usize)
               -> Result<(lt_sys::cublasLtMatrixLayout_t, lt_sys::cublasLtMatrixLayout_t)> {
        let b = lt_layout(lt_sys::cudaDataType::CUDA_R_4F_E2M1, k, t, k)?;
        match lt_layout(lt_sys::cudaDataType::CUDA_R_16BF, n, t, n) {
            Ok(d) => Ok((b, d)),
            Err(e) => {
                unsafe { let _ = lt_result::destroy_matrix_layout(b); }
                Err(e)
            }
        }
    }

    /// cuBLASLt's heuristic at [`PINNED_REF_ROWS`] rows; the preferred split-K-free algorithm, else the first.
    fn choose(&self, n: usize, k: usize, ws_bytes: usize) -> Result<Fp4Plan> {
        use lt_sys::cublasLtMatmulAlgoConfigAttributes_t as C;
        use lt_sys::cublasLtMatmulDescAttributes_t as D;
        use lt_sys::cublasLtMatmulPreferenceAttributes_t as P;
        let desc = lt_result::create_matmul_desc(lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                                                 lt_sys::cudaDataType::CUDA_R_32F)
            .map_err(lt_err("cublasLtMatmulDescCreate"))?;
        let a = match lt_layout(lt_sys::cudaDataType::CUDA_R_4F_E2M1, k, n, k) {
            Ok(a) => a,
            Err(e) => {
                unsafe { let _ = lt_result::destroy_matmul_desc(desc); }
                return Err(e);
            }
        };
        let (b, d) = match Self::layouts(n, k, PINNED_REF_ROWS) {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    let _ = lt_result::destroy_matrix_layout(a);
                    let _ = lt_result::destroy_matmul_desc(desc);
                }
                return Err(e);
            }
        };
        let pref = lt_result::create_matmul_pref().map_err(lt_err("cublasLtMatmulPreferenceCreate"));
        let picked = (|| -> Result<Fp4Plan> {
            let pref = *pref.as_ref().map_err(|e| anyhow!("{e}"))?;
            let ta = cublas_sys::cublasOperation_t::CUBLAS_OP_T;
            let tb = cublas_sys::cublasOperation_t::CUBLAS_OP_N;
            let mut cands = [lt_sys::cublasLtMatmulHeuristicResult_t {
                algo: lt_sys::cublasLtMatmulAlgo_t { data: [0; 8] },
                workspaceSize: 0,
                state: lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                wavesCount: 0.0,
                reserved: [0; 4],
            }; 32];
            let mut found = 0i32;
            unsafe {
                let sz = std::mem::size_of::<cublas_sys::cublasOperation_t>();
                lt_result::set_matmul_desc_attribute(desc, D::CUBLASLT_MATMUL_DESC_TRANSA,
                                                     (&ta as *const cublas_sys::cublasOperation_t).cast(), sz)
                    .map_err(lt_err("TRANSA"))?;
                lt_result::set_matmul_desc_attribute(desc, D::CUBLASLT_MATMUL_DESC_TRANSB,
                                                     (&tb as *const cublas_sys::cublasOperation_t).cast(), sz)
                    .map_err(lt_err("TRANSB"))?;
                let mode = lt_sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 as i32;
                for attr in [D::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, D::CUBLASLT_MATMUL_DESC_B_SCALE_MODE] {
                    lt_result::set_matmul_desc_attribute(desc, attr, (&mode as *const i32).cast(), 4)
                        .map_err(lt_err("SCALE_MODE"))?;
                }
                // Scale pointers must be set for the heuristic; the scratch stands in for both.
                let p = self.scratch.ptr();
                for attr in [D::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, D::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER] {
                    lt_result::set_matmul_desc_attribute(desc, attr, (&p as *const u64).cast(), 8)
                        .map_err(lt_err("SCALE_POINTER"))?;
                }
                let ws = ws_bytes as u64;
                lt_result::set_matmul_pref_attribute(pref, P::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     (&ws as *const u64).cast(), 8)
                    .map_err(lt_err("MAX_WORKSPACE_BYTES"))?;
                lt_sys::cublasLtMatmulAlgoGetHeuristic(self.handle, desc, a, b, d, d, pref, cands.len() as i32,
                                                       cands.as_mut_ptr(), &mut found)
                    .result()
                    .map_err(lt_err("cublasLtMatmulAlgoGetHeuristic (FP4 e2m1, VEC16_UE4M3 scales)"))?;
            }
            let mut ok = Vec::new();
            for cand in &cands[..found.max(0) as usize] {
                if cand.state != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    continue;
                }
                let splitk = algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?;
                let red = algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?;
                if splitk > 1 || red != 0 {
                    continue;
                }
                let ids = (algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_ID)?,
                           algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
                           algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
                           algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?);
                ok.push((cand.algo, ids));
            }
            let want = FP4_PREFERRED.iter().find(|e| e.0 == self.cc && e.1 == n && e.2 == k).map(|e| e.3);
            let pick = want
                .and_then(|w| ok.iter().find(|(_, ids)| (ids.0, ids.1, ids.3) == w))
                .or_else(|| ok.first())
                .copied();
            let Some((algo, ids)) = pick else {
                bail!("none of cuBLASLt's {found} FP4 candidates for N={n}, K={k} runs without split-K");
            };
            Ok(Fp4Plan { desc, a, algo, ids, rows: HashMap::new() })
        })();
        unsafe {
            let _ = lt_result::destroy_matrix_layout(b);
            let _ = lt_result::destroy_matrix_layout(d);
            if let Ok(p) = pref {
                let _ = lt_result::destroy_matmul_pref(p);
            }
            if picked.is_err() {
                let _ = lt_result::destroy_matrix_layout(a);
                let _ = lt_result::destroy_matmul_desc(desc);
            }
        }
        picked
    }
}

/// One cuBLAS handle bound to one cuda-core stream (and, in [`GemmMode::Pinned`], one cuBLASLt handle; with
/// [`Blas::enable_fp4_lt`], another for the FP4 GEMM).
pub struct Blas {
    handle: Mutex<Handle>,
    lt: Option<Mutex<LtState>>,
    fp4: Option<Mutex<Fp4State>>,
    stream: Arc<Stream>,
    workspace: Arc<DevBuf>,
}

/// What selects a pinned algorithm: everything about a call except its row count T.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LtKey {
    n: usize,
    k: usize,
    ldx: usize,
    ldy: usize,
    f32_out: bool,
}

impl LtKey {
    /// Byte alignments (x, w, y) the pinned algorithm is chosen for: the largest power of two up to 16 that
    /// divides the row pitch (arena values and weights are 256-byte aligned, so the pitch decides).
    fn alignments(&self) -> (u32, u32, u32) {
        let a = |pitch: usize| (1usize << pitch.trailing_zeros().min(4)) as u32;
        (a(self.ldx * 2), a(self.k * 2), a(self.ldy * if self.f32_out { 4 } else { 2 }))
    }
}

/// The pinned algorithm of one [`LtKey`], its descriptors, and the B / C layouts per row count.
struct LtPlan {
    desc: lt_sys::cublasLtMatmulDesc_t,
    a: lt_sys::cublasLtMatrixLayout_t,
    algo: lt_sys::cublasLtMatmulAlgo_t,
    /// (algorithm id, tile id, stages id), for reports.
    ids: (u32, u32, u32),
    /// Per T: the B and C (= D) layouts, checked against the algorithm once.
    rows: HashMap<usize, (lt_sys::cublasLtMatrixLayout_t, lt_sys::cublasLtMatrixLayout_t)>,
}

struct LtState {
    handle: lt_sys::cublasLtHandle_t,
    plans: HashMap<LtKey, LtPlan>,
}

// SAFETY: cuBLASLt handles and descriptors are host objects; the Mutex around LtState serializes every use.
unsafe impl Send for LtState {}

impl Drop for LtState {
    fn drop(&mut self) {
        unsafe {
            for p in self.plans.values() {
                for (b, c) in p.rows.values() {
                    let _ = lt_result::destroy_matrix_layout(*b);
                    let _ = lt_result::destroy_matrix_layout(*c);
                }
                let _ = lt_result::destroy_matrix_layout(p.a);
                let _ = lt_result::destroy_matmul_desc(p.desc);
            }
            let _ = lt_result::destroy_handle(self.handle);
        }
    }
}

fn lt_err(what: &'static str) -> impl Fn(lt_result::CublasError) -> anyhow::Error {
    move |e| anyhow!("{what}: {e:?}")
}

fn lt_layout(dt: lt_sys::cudaDataType, rows: usize, cols: usize, ld: usize) -> Result<lt_sys::cublasLtMatrixLayout_t> {
    lt_result::create_matrix_layout(dt, rows as u64, cols as u64, ld as i64).map_err(lt_err("cublasLtMatrixLayoutCreate"))
}

fn algo_config_u32(algo: &lt_sys::cublasLtMatmulAlgo_t, attr: lt_sys::cublasLtMatmulAlgoConfigAttributes_t)
    -> Result<u32> {
    let mut v = 0u32;
    let mut written = 0usize;
    unsafe {
        lt_sys::cublasLtMatmulAlgoConfigGetAttribute(algo, attr, (&mut v as *mut u32).cast(), 4, &mut written)
            .result()
            .map_err(lt_err("cublasLtMatmulAlgoConfigGetAttribute"))?;
    }
    Ok(v)
}

impl LtState {
    /// The pinned plan of `key`, chosen on first use.
    fn plan(&mut self, key: LtKey, ws_bytes: usize) -> Result<&mut LtPlan> {
        if !self.plans.contains_key(&key) {
            let p = self.choose(key, ws_bytes).with_context(|| format!("pinning a cuBLASLt algorithm for {key:?}"))?;
            self.plans.insert(key, p);
        }
        Ok(self.plans.get_mut(&key).expect("inserted above"))
    }

    /// cuBLASLt's heuristic for [`PINNED_REF_ROWS`] rows; the first candidate without split-K or a reduction
    /// scheme.
    fn choose(&self, key: LtKey, ws_bytes: usize) -> Result<LtPlan> {
        use lt_sys::cublasLtMatmulAlgoConfigAttributes_t as C;
        use lt_sys::cublasLtMatmulDescAttributes_t as D;
        use lt_sys::cublasLtMatmulPreferenceAttributes_t as P;
        let bf = lt_sys::cudaDataType::CUDA_R_16BF;
        let c_type = if key.f32_out { lt_sys::cudaDataType::CUDA_R_32F } else { bf };
        let desc = lt_result::create_matmul_desc(lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                                                 lt_sys::cudaDataType::CUDA_R_32F)
            .map_err(lt_err("cublasLtMatmulDescCreate"))?;
        let a = lt_layout(bf, key.k, key.n, key.k)?;
        let b = lt_layout(bf, key.k, PINNED_REF_ROWS, key.ldx)?;
        let c = lt_layout(c_type, key.n, PINNED_REF_ROWS, key.ldy)?;
        let pref = lt_result::create_matmul_pref().map_err(lt_err("cublasLtMatmulPreferenceCreate"))?;
        let picked = (|| -> Result<LtPlan> {
            let ta = cublas_sys::cublasOperation_t::CUBLAS_OP_T;
            let tb = cublas_sys::cublasOperation_t::CUBLAS_OP_N;
            let mut cands = [lt_sys::cublasLtMatmulHeuristicResult_t {
                algo: lt_sys::cublasLtMatmulAlgo_t { data: [0; 8] },
                workspaceSize: 0,
                state: lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                wavesCount: 0.0,
                reserved: [0; 4],
            }; 32];
            let mut found = 0i32;
            unsafe {
                let sz = std::mem::size_of::<cublas_sys::cublasOperation_t>();
                lt_result::set_matmul_desc_attribute(desc, D::CUBLASLT_MATMUL_DESC_TRANSA,
                                                     (&ta as *const cublas_sys::cublasOperation_t).cast(), sz)
                    .map_err(lt_err("TRANSA"))?;
                lt_result::set_matmul_desc_attribute(desc, D::CUBLASLT_MATMUL_DESC_TRANSB,
                                                     (&tb as *const cublas_sys::cublasOperation_t).cast(), sz)
                    .map_err(lt_err("TRANSB"))?;
                let ws = ws_bytes as u64;
                lt_result::set_matmul_pref_attribute(pref, P::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     (&ws as *const u64).cast(), 8)
                    .map_err(lt_err("MAX_WORKSPACE_BYTES"))?;
                // The alignments every call with this key guarantees (checked per call), so the pinned
                // algorithm never relies on a pointer alignment a later call lacks.
                let (ax, aw, ay) = key.alignments();
                for (p, align) in [(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, aw),
                                   (P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, ax),
                                   (P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, ay),
                                   (P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, ay)] {
                    lt_result::set_matmul_pref_attribute(pref, p, (&align as *const u32).cast(), 4)
                        .map_err(lt_err("MIN_ALIGNMENT"))?;
                }
                lt_sys::cublasLtMatmulAlgoGetHeuristic(self.handle, desc, a, b, c, c, pref, cands.len() as i32,
                                                       cands.as_mut_ptr(), &mut found)
                    .result()
                    .map_err(lt_err("cublasLtMatmulAlgoGetHeuristic"))?;
            }
            for cand in &cands[..found.max(0) as usize] {
                if cand.state != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    continue;
                }
                let splitk = algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?;
                let red = algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?;
                if splitk > 1 || red != 0 {
                    continue;
                }
                let ids = (algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_ID)?,
                           algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
                           algo_config_u32(&cand.algo, C::CUBLASLT_ALGO_CONFIG_STAGES_ID)?);
                return Ok(LtPlan { desc, a, algo: cand.algo, ids, rows: HashMap::new() });
            }
            Err(anyhow!("none of cuBLASLt's {found} candidates runs without split-K"))
        })();
        unsafe {
            let _ = lt_result::destroy_matrix_layout(b);
            let _ = lt_result::destroy_matrix_layout(c);
            let _ = lt_result::destroy_matmul_pref(pref);
            if picked.is_err() {
                let _ = lt_result::destroy_matrix_layout(a);
                let _ = lt_result::destroy_matmul_desc(desc);
            }
        }
        picked
    }
}

struct Handle(cublas_sys::cublasHandle_t);

// SAFETY: a cuBLAS handle may be used from any thread as long as calls are not concurrent; the Mutex
// serializes them.
unsafe impl Send for Handle {}

impl Blas {
    /// A handle in [`GemmMode::Heuristic`] (cuBLAS picks the kernel per call).
    pub fn new(stream: &Arc<Stream>) -> Result<Self> {
        Self::with_mode(stream, GemmMode::Heuristic)
    }

    pub fn with_mode(stream: &Arc<Stream>, mode: GemmMode) -> Result<Self> {
        stream.context().bind_to_thread()?;
        let workspace = DevBuf::alloc(stream.context(), CUBLAS_WORKSPACE_BYTES)?;
        let handle = cublas_result::create_handle().map_err(|e| anyhow!("cublasCreate: {e:?}"))?;
        let lt = match mode {
            GemmMode::Heuristic => None,
            GemmMode::Pinned => Some(Mutex::new(LtState {
                handle: lt_result::create_handle().map_err(|e| anyhow!("cublasLtCreate: {e:?}"))?,
                plans: HashMap::new(),
            })),
        };
        let blas =
            Self { handle: Mutex::new(Handle(handle)), lt, fp4: None, stream: stream.clone(), workspace: workspace.clone() };
        unsafe {
            cublas_result::set_stream(handle, stream.cu_stream() as cublas_sys::cudaStream_t)
                .map_err(|e| anyhow!("cublasSetStream: {e:?}"))?;
            cublas_sys::cublasSetWorkspace_v2(handle, workspace.ptr() as usize as *mut c_void, workspace.bytes())
                .result()
                .map_err(|e| anyhow!("cublasSetWorkspace: {e:?}"))?;
            cublas_sys::cublasSetPointerMode_v2(handle, cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
                .result()
                .map_err(|e| anyhow!("cublasSetPointerMode: {e:?}"))?;
            let math = cublas_sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
            cublas_sys::cublasSetMathMode(handle, math).result().map_err(|e| anyhow!("cublasSetMathMode: {e:?}"))?;
        }
        Ok(blas)
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    pub fn mode(&self) -> GemmMode {
        if self.lt.is_some() { GemmMode::Pinned } else { GemmMode::Heuristic }
    }

    /// Pinned mode: `((N, K, ldx, ldy, f32 out), (algorithm id, tile id, stages id))` of every algorithm pinned
    /// so far (reports).
    pub fn pinned_algos(&self) -> Vec<((usize, usize, usize, usize, bool), (u32, u32, u32))> {
        let Some(lt) = &self.lt else { return Vec::new() };
        let lt = lt.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = lt.plans.iter().map(|(k, p)| ((k.n, k.k, k.ldx, k.ldy, k.f32_out), p.ids)).collect();
        v.sort_unstable();
        v
    }

    /// C1: enable [`Blas::gemm_nvfp4_lt`] for activations of up to `max_rows` rows and `max_scale_cols` scale
    /// columns (K/16) on a device of compute capability `cc`. Allocates the tiled activation-scale scratch.
    pub fn enable_fp4_lt(&mut self, max_rows: usize, max_scale_cols: usize, cc: (u32, u32)) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        let bytes = crate::kernels::nvfp4::lt_scale_bytes(max_rows.max(1), max_scale_cols.max(4));
        let scratch = DevBuf::alloc(self.stream.context(), bytes)?;
        let handle = lt_result::create_handle().map_err(|e| anyhow!("cublasLtCreate: {e:?}"))?;
        self.fp4 = Some(Mutex::new(Fp4State { handle, plans: HashMap::new(), cc, scratch }));
        Ok(())
    }

    /// Whether [`Blas::enable_fp4_lt`] was called.
    pub fn fp4_lt(&self) -> bool {
        self.fp4.is_some()
    }

    /// Bytes of the FP4 activation-scale scratch (0 when disabled).
    pub fn fp4_scratch_bytes(&self) -> usize {
        self.fp4.as_ref().map_or(0, |f| f.lock().unwrap_or_else(|e| e.into_inner()).scratch.bytes())
    }

    /// The scratch as a flat f8e4m3 tensor of `lt_scale_bytes(rows, cols)` bytes: where the re-layout kernel writes
    /// a `[rows, cols]` activation-scale matrix for the next [`Blas::gemm_nvfp4_lt`].
    pub fn fp4_scratch(&self, rows: usize, cols: usize) -> Result<DevTensor> {
        let f = self.fp4.as_ref().context("the cuBLASLt FP4 GEMM is not enabled")?;
        let buf = f.lock().unwrap_or_else(|e| e.into_inner()).scratch.clone();
        let need = crate::kernels::nvfp4::lt_scale_bytes(rows, cols);
        ensure!(need <= buf.bytes(), "FP4 scale scratch: {rows}x{cols} needs {need} bytes of {}", buf.bytes());
        DevTensor::new(buf, 0, GraphDType::F8E4M3, vec![need])
    }

    /// Pinned FP4 algorithms so far: `((N, K), (algorithm id, tile id, stages id, custom option))` (reports).
    pub fn fp4_algos(&self) -> Vec<((usize, usize), (u32, u32, u32, u32))> {
        let Some(f) = &self.fp4 else { return Vec::new() };
        let f = f.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = f.plans.iter().map(|(k, p)| (*k, p.ids)).collect();
        v.sort_unstable();
        v
    }

    /// C1: `y [T, N] = bf16(alpha · Σ_k (q_x s_x)(q_w s_w))` on cuBLASLt: `q` packed e2m1 `[T, K/2]`, `s_lt` its
    /// block scales in cuBLASLt's tiled layout (`fp4_scratch` after the re-layout kernel), `w` packed `[N, K/2]`,
    /// `w_lt` the weight's tiled scales (`Linear::Nvfp4::lt_scales`), `alpha = 1/(igs·wgs)` (one part). One pinned
    /// algorithm per `(N, K)` at every T, never split-K.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_nvfp4_lt(&self, stream: &Stream, q: &DevTensor, s_lt: &DevTensor, w: &DevTensor, w_lt: &DevTensor,
                         alpha: f32, y: &DevTensor) -> Result<()> {
        use crate::kernels::nvfp4::lt_scale_bytes;
        use lt_sys::cublasLtMatmulDescAttributes_t as D;
        let f = self.fp4.as_ref().context("the cuBLASLt FP4 GEMM is not enabled")?;
        ensure!(q.dtype == GraphDType::U8 && w.dtype == GraphDType::U8 && y.dtype == GraphDType::BF16,
                "gemm_nvfp4_lt operands {q:?} {w:?} {y:?}");
        ensure!(q.shape.len() == 2 && w.shape.len() == 2 && y.shape.len() == 2, "gemm_nvfp4_lt needs rank-2 operands");
        ensure!(q.is_contiguous() && w.is_contiguous() && y.is_contiguous(), "gemm_nvfp4_lt needs contiguous operands");
        let (t, k, n) = (q.shape[0], q.shape[1] * 2, w.shape[0]);
        ensure!(w.shape[1] * 2 == k && y.shape == [t, n], "gemm_nvfp4_lt shapes: q {:?} w {:?} y {:?}", q.shape, w.shape,
                y.shape);
        ensure!(s_lt.bytes() >= lt_scale_bytes(t, k / 16) && w_lt.bytes() >= lt_scale_bytes(n, k / 16),
                "gemm_nvfp4_lt: tiled scales too small");
        if t == 0 {
            return Ok(());
        }
        for p in [q.ptr(), w.ptr(), y.ptr(), s_lt.ptr(), w_lt.ptr()] {
            ensure!(p % 16 == 0, "gemm_nvfp4_lt operand at {p:#x} is not 16-byte aligned");
        }
        let mut st = f.lock().unwrap_or_else(|e| e.into_inner());
        let ws = self.workspace.bytes();
        if !st.plans.contains_key(&(n, k)) {
            let p = st.choose(n, k, ws).with_context(|| format!("pinning a cuBLASLt FP4 algorithm for N={n}, K={k}"))?;
            st.plans.insert((n, k), p);
        }
        let handle = st.handle;
        let plan = st.plans.get_mut(&(n, k)).expect("inserted above");
        unsafe {
            for (attr, p) in [(D::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, w_lt.ptr()),
                              (D::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, s_lt.ptr())] {
                lt_result::set_matmul_desc_attribute(plan.desc, attr, (&p as *const u64).cast(), 8)
                    .map_err(lt_err("SCALE_POINTER"))?;
            }
        }
        if !plan.rows.contains_key(&t) {
            let (b, d) = Fp4State::layouts(n, k, t)?;
            let mut res = std::mem::MaybeUninit::<lt_sys::cublasLtMatmulHeuristicResult_t>::zeroed();
            let ok = unsafe { lt_sys::cublasLtMatmulAlgoCheck(handle, plan.desc, plan.a, b, d, d, &plan.algo, res.as_mut_ptr()) };
            let need = unsafe { res.assume_init() }.workspaceSize;
            if ok.result().is_err() || need > ws {
                unsafe {
                    let _ = lt_result::destroy_matrix_layout(b);
                    let _ = lt_result::destroy_matrix_layout(d);
                }
                return Err(anyhow!("the pinned cuBLASLt FP4 algorithm {:?} of N={n}, K={k} does not run {t} rows \
                                    ({ok:?}, workspace {need} of {ws})", plan.ids));
            }
            plan.rows.insert(t, (b, d));
        }
        let (b, d) = plan.rows[&t];
        let beta = 0.0f32;
        unsafe {
            lt_result::matmul(handle, plan.desc, (&alpha as *const f32).cast(), (&beta as *const f32).cast(),
                              w.ptr() as usize as *const c_void, plan.a, q.ptr() as usize as *const c_void, b,
                              y.ptr() as usize as *const c_void, d, y.ptr() as usize as *mut c_void, d, &plan.algo,
                              self.workspace.ptr() as usize as *mut c_void, ws, stream.cu_stream() as lt_sys::cudaStream_t)
                .map_err(|e| anyhow!("cublasLtMatmul FP4 (t={t}, n={n}, k={k}): {e:?}"))
        }
    }

    /// Pinned mode: run `y = x · wᵀ` with the weight shape's pinned algorithm.
    #[allow(clippy::too_many_arguments)]
    fn gemm_pinned(&self, lt: &Mutex<LtState>, stream: &Stream, x: &DevTensor, w: &DevTensor, y: &DevTensor, t: usize,
                   n: usize, k: usize) -> Result<()> {
        let key = LtKey { n, k, ldx: x.row_stride.max(k), ldy: y.row_stride.max(n), f32_out: y.dtype == GraphDType::F32 };
        let (ax, aw, ay) = key.alignments();
        for (p, a) in [(x.ptr(), ax), (w.ptr(), aw), (y.ptr(), ay)] {
            ensure!(p % a as u64 == 0, "pinned GEMM operand at {p:#x} is not {a}-byte aligned");
        }
        let mut st = lt.lock().unwrap_or_else(|e| e.into_inner());
        let handle = st.handle;
        let ws = self.workspace.bytes();
        let plan = st.plan(key, ws)?;
        if !plan.rows.contains_key(&t) {
            let ct = if key.f32_out { lt_sys::cudaDataType::CUDA_R_32F } else { lt_sys::cudaDataType::CUDA_R_16BF };
            let b = lt_layout(lt_sys::cudaDataType::CUDA_R_16BF, k, t, key.ldx)?;
            let c = lt_layout(ct, n, t, key.ldy)?;
            let mut res = std::mem::MaybeUninit::<lt_sys::cublasLtMatmulHeuristicResult_t>::zeroed();
            let ok = unsafe {
                lt_sys::cublasLtMatmulAlgoCheck(handle, plan.desc, plan.a, b, c, c, &plan.algo, res.as_mut_ptr())
            };
            let need = unsafe { res.assume_init() }.workspaceSize;
            if ok.result().is_err() || need > ws {
                unsafe {
                    let _ = lt_result::destroy_matrix_layout(b);
                    let _ = lt_result::destroy_matrix_layout(c);
                }
                return Err(anyhow!("the pinned cuBLASLt algorithm {:?} of {key:?} does not run {t} rows ({ok:?}, \
                                    workspace {need} of {ws})", plan.ids));
            }
            plan.rows.insert(t, (b, c));
        }
        let (b, c) = plan.rows[&t];
        let (alpha, beta) = (1.0f32, 0.0f32);
        unsafe {
            lt_result::matmul(
                handle,
                plan.desc,
                (&alpha as *const f32).cast(),
                (&beta as *const f32).cast(),
                w.ptr() as usize as *const c_void,
                plan.a,
                x.ptr() as usize as *const c_void,
                b,
                y.ptr() as usize as *const c_void,
                c,
                y.ptr() as usize as *mut c_void,
                c,
                &plan.algo,
                self.workspace.ptr() as usize as *mut c_void,
                ws,
                stream.cu_stream() as lt_sys::cudaStream_t,
            )
            .map_err(|e| anyhow!("cublasLtMatmul (t={t}, n={n}, k={k}): {e:?}"))
        }
    }

    /// `y[T, N] = x[T, K] · w[N, K]ᵀ` on `stream` (normally the handle's own stream, or a capture stream
    /// forked from it). `x` and `y` may be row-strided views (column slices of a wider buffer); `w` must
    /// be contiguous rows. `y` is bf16 or f32.
    pub fn gemm_bf16(&self, stream: &Stream, x: &DevTensor, w: &DevTensor, y: &DevTensor) -> Result<()> {
        ensure!(x.dtype == GraphDType::BF16 && w.dtype == GraphDType::BF16, "gemm_bf16 inputs {x:?} {w:?}");
        ensure!(x.shape.len() == 2 && w.shape.len() == 2 && y.shape.len() == 2, "gemm needs rank-2 operands");
        let (t, k) = (x.shape[0], x.shape[1]);
        let n = w.shape[0];
        ensure!(w.shape[1] == k, "gemm K mismatch: x {:?} w {:?}", x.shape, w.shape);
        ensure!(y.shape == [t, n], "gemm out {:?}, expected [{t}, {n}]", y.shape);
        ensure!(w.is_contiguous(), "weight must be contiguous");
        let c_type = match y.dtype {
            GraphDType::BF16 => cublas_sys::cudaDataType_t::CUDA_R_16BF,
            GraphDType::F32 => cublas_sys::cudaDataType_t::CUDA_R_32F,
            d => return Err(anyhow!("gemm output dtype {d:?}")),
        };
        if t == 0 || n == 0 {
            return Ok(());
        }
        for d in [t, n, k, x.row_stride, y.row_stride] {
            ensure!(d <= i32::MAX as usize, "gemm dim {d} too large for cuBLAS");
        }
        if let Some(lt) = &self.lt {
            return self.gemm_pinned(lt, stream, x, w, y, t, n, k);
        }
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let guard = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        let handle = guard.0;
        unsafe {
            cublas_result::set_stream(handle, stream.cu_stream() as cublas_sys::cudaStream_t)
                .map_err(|e| anyhow!("cublasSetStream: {e:?}"))?;
            cublas_result::gemm_ex(
                handle,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                t as i32,
                k as i32,
                (&alpha as *const f32).cast(),
                w.ptr() as usize as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                x.ptr() as usize as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                x.row_stride.max(k) as i32,
                (&beta as *const f32).cast(),
                y.ptr() as usize as *mut c_void,
                c_type,
                y.row_stride.max(n) as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| anyhow!("cublasGemmEx (t={t}, n={n}, k={k}): {e:?}"))?;
        }
        Ok(())
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        let h = self.handle.get_mut().unwrap_or_else(|e| e.into_inner()).0;
        let _ = unsafe { cublas_result::destroy_handle(h) };
    }
}
