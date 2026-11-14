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

use crate::engine::graph::tensor::{DevBuf, DevTensor, GraphDType};
use anyhow::{Context as _, Result, anyhow, ensure};
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

/// One cuBLAS handle bound to one cuda-core stream (and, in [`GemmMode::Pinned`], one cuBLASLt handle).
pub struct Blas {
    handle: Mutex<Handle>,
    lt: Option<Mutex<LtState>>,
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
        let blas = Self { handle: Mutex::new(Handle(handle)), lt, stream: stream.clone(), workspace: workspace.clone() };
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
