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

use crate::engine::graph::tensor::{DevBuf, DevTensor, GraphDType};
use anyhow::{Result, anyhow, ensure};
use cuda_core::Stream;
use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

/// Workspace given to cuBLAS (enough for split-K on large GEMMs; cuBLAS recommends ≥ 32 MiB on Hopper+).
pub const CUBLAS_WORKSPACE_BYTES: usize = 64 << 20;

/// One cuBLAS handle bound to one cuda-core stream.
pub struct Blas {
    handle: Mutex<Handle>,
    stream: Arc<Stream>,
    _workspace: Arc<DevBuf>,
}

struct Handle(cublas_sys::cublasHandle_t);

// SAFETY: a cuBLAS handle may be used from any thread as long as calls are not concurrent; the Mutex
// serializes them.
unsafe impl Send for Handle {}

impl Blas {
    pub fn new(stream: &Arc<Stream>) -> Result<Self> {
        stream.context().bind_to_thread()?;
        let workspace = DevBuf::alloc(stream.context(), CUBLAS_WORKSPACE_BYTES)?;
        let handle = cublas_result::create_handle().map_err(|e| anyhow!("cublasCreate: {e:?}"))?;
        let blas = Self { handle: Mutex::new(Handle(handle)), stream: stream.clone(), _workspace: workspace.clone() };
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
