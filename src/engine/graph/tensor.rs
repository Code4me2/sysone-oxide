//! Device memory for the engine: owned allocations ([`DevBuf`]) and typed, possibly row-strided views into
//! them ([`DevTensor`]). Weights, the activation arena and batch metadata all live in these.
//!
//! The engine passes raw pointers to cuBLAS and hands cuTile kernels borrowed `cutile` tensors over the same
//! memory ([`DevTensor::tile`]). Ownership stays with the engine: every `DevBuf` outlives the graphs and
//! launches that use it.

use anyhow::{Context as _, Result, ensure};
use cuda_core::sys::CUdeviceptr;
use cuda_core::{Context, DeviceAllocation, Stream};
use std::sync::Arc;

/// Element types a graph value or weight can have (DESIGN §6.2, §3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphDType {
    BF16,
    F32,
    /// Packed e2m1 pairs (two FP4 values per byte, NVFP4 `weight_packed` layout).
    U8,
    F8E4M3,
    /// Metadata only (token ids, positions).
    U32,
    I32,
}

impl GraphDType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::U8 | Self::F8E4M3 => 1,
        }
    }

    /// The safetensors dtype string, if this type appears in checkpoints.
    pub fn from_safetensors(s: &str) -> Option<Self> {
        Some(match s {
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "U8" => Self::U8,
            "F8_E4M3" => Self::F8E4M3,
            "U32" => Self::U32,
            "I32" => Self::I32,
            _ => return None,
        })
    }

    fn matches(self, id: cuda_core::DTypeId) -> bool {
        use cuda_core::DTypeId as D;
        matches!(
            (self, id),
            (Self::BF16, D::BF16)
                | (Self::F32, D::F32)
                | (Self::U8, D::U8)
                | (Self::U8, D::F4E2M1FNX2)
                | (Self::F8E4M3, D::F8E4M3FN)
                | (Self::U8, D::F8E4M3FN)
                | (Self::U32, D::U32)
                | (Self::I32, D::I32)
                // bf16 bits viewed as f16 for pure data movement (embedding gather)
                | (Self::BF16, D::F16)
        )
    }
}

/// An owned, synchronous (`cuMemAlloc`) device allocation, freed on drop.
pub struct DevBuf {
    ptr: CUdeviceptr,
    bytes: usize,
    ctx: Arc<Context>,
}

// SAFETY: the pointer is a plain device address owned by this value; the driver synchronizes access.
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

impl DevBuf {
    /// Uninitialized allocation of `bytes` (at least 1) bytes.
    pub fn alloc(ctx: &Arc<Context>, bytes: usize) -> Result<Arc<Self>> {
        ctx.bind_to_thread()?;
        let n = bytes.max(1);
        let ptr = unsafe { cuda_core::memory::malloc_sync(n) }
            .with_context(|| format!("cuMemAlloc of {} MiB", n >> 20))?;
        Ok(Arc::new(Self { ptr, bytes: n, ctx: ctx.clone() }))
    }

    /// Zero-filled allocation; the fill completes before this returns.
    pub fn zeroed(stream: &Arc<Stream>, bytes: usize) -> Result<Arc<Self>> {
        let b = Self::alloc(stream.context(), bytes)?;
        unsafe { cuda_core::memory::memset_d8_async(b.ptr, 0, b.bytes, stream.cu_stream())? };
        stream.synchronize()?;
        Ok(b)
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn ctx(&self) -> &Arc<Context> {
        &self.ctx
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            // A destructor cannot report a refused free.
            let _ = unsafe { cuda_core::memory::free_sync(self.ptr) };
        }
    }
}

// SAFETY: `ptr` is a live `bytes`-long allocation in `ctx` until drop.
unsafe impl DeviceAllocation for DevBuf {
    fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn context(&self) -> Arc<Context> {
        self.ctx.clone()
    }
}

/// A byte range of a [`DevBuf`], kept alive by its `Arc`; the owner object for borrowed cutile tensors.
struct SubAlloc {
    buf: Arc<DevBuf>,
    offset: usize,
    bytes: usize,
}

// SAFETY: a sub-range of a live allocation, kept alive by the Arc.
unsafe impl DeviceAllocation for SubAlloc {
    fn device_ptr(&self) -> CUdeviceptr {
        self.buf.ptr + self.offset as u64
    }
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn context(&self) -> Arc<Context> {
        self.buf.ctx.clone()
    }
}

/// A typed view of device memory: `shape` row-major, rows `row_stride` elements apart (a column slice of
/// a wider tensor has `row_stride > cols`). Rank 1 and 2 cover every engine value; higher ranks are
/// contiguous only.
#[derive(Clone)]
pub struct DevTensor {
    buf: Arc<DevBuf>,
    offset: usize,
    pub dtype: GraphDType,
    pub shape: Vec<usize>,
    /// Elements between consecutive rows (dim 0) for rank-2 views; equals `shape[1]` when contiguous.
    pub row_stride: usize,
}

impl std::fmt::Debug for DevTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DevTensor({:?} {:?} stride {} @{:#x})", self.dtype, self.shape, self.row_stride, self.ptr())
    }
}

impl DevTensor {
    /// A contiguous view of `buf` starting at byte `offset`.
    pub fn new(buf: Arc<DevBuf>, offset: usize, dtype: GraphDType, shape: Vec<usize>) -> Result<Self> {
        let t = Self { row_stride: if shape.len() >= 2 { shape[1..].iter().product() } else { 0 }, buf, offset, dtype, shape };
        ensure!(t.offset + t.span_bytes() <= t.buf.bytes, "view {:?} at {} past a {}-byte buffer", t.shape,
                t.offset, t.buf.bytes);
        Ok(t)
    }

    /// Allocate a fresh contiguous tensor (uninitialized).
    pub fn alloc(ctx: &Arc<Context>, dtype: GraphDType, shape: &[usize]) -> Result<Self> {
        let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
        Self::new(DevBuf::alloc(ctx, bytes)?, 0, dtype, shape.to_vec())
    }

    /// Allocate a fresh zero-filled contiguous tensor.
    pub fn zeros(stream: &Arc<Stream>, dtype: GraphDType, shape: &[usize]) -> Result<Self> {
        let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
        Self::new(DevBuf::zeroed(stream, bytes)?, 0, dtype, shape.to_vec())
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.buf.ptr + self.offset as u64
    }

    pub fn buf(&self) -> &Arc<DevBuf> {
        &self.buf
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Elements per row (product of all dims after the first).
    pub fn cols(&self) -> usize {
        self.shape.iter().skip(1).product()
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.len() < 2 || self.row_stride == self.cols()
    }

    /// Logical bytes (what a contiguous copy would hold).
    pub fn bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }

    /// Bytes from the first to one past the last addressed element.
    pub fn span_bytes(&self) -> usize {
        if self.numel() == 0 {
            return 0;
        }
        let elems = if self.shape.len() >= 2 { (self.rows() - 1) * self.row_stride + self.cols() } else { self.numel() };
        elems * self.dtype.size_in_bytes()
    }

    /// Reinterpret a contiguous tensor with another shape of the same element count.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        ensure!(self.is_contiguous(), "reshape of a strided view {self:?}");
        ensure!(shape.iter().product::<usize>() == self.numel(), "reshape {:?} -> {shape:?}", self.shape);
        Self::new(self.buf.clone(), self.offset, self.dtype, shape.to_vec())
    }

    /// Rows `start..start+len` (dim 0) as a view.
    pub fn rows_view(&self, start: usize, len: usize) -> Result<Self> {
        ensure!(start + len <= self.rows(), "rows {start}+{len} of {self:?}");
        let mut t = self.clone();
        t.offset += start * self.row_stride.max(self.cols()) * self.dtype.size_in_bytes();
        if t.shape.is_empty() {
            t.shape = vec![len];
        } else {
            t.shape[0] = len;
        }
        Ok(t)
    }

    /// Columns `start..start+len` of a rank-2 tensor, as a row-strided view.
    pub fn cols_view(&self, start: usize, len: usize) -> Result<Self> {
        ensure!(self.shape.len() == 2, "column view of rank-{} tensor", self.shape.len());
        ensure!(start + len <= self.shape[1], "cols {start}+{len} of {self:?}");
        let mut t = self.clone();
        t.offset += start * self.dtype.size_in_bytes();
        t.shape[1] = len;
        Ok(t)
    }

    /// A borrowed cutile tensor over this view, for cuTile kernels (W3). Contiguous views get contiguous
    /// strides; a column view keeps its row stride.
    ///
    /// # Safety
    /// The usual cutile foreign-memory aliasing contract: while a kernel writes through the returned tensor
    /// nothing else may access the bytes, and the bytes must hold valid `T` values. Liveness is guaranteed
    /// (the tensor holds the allocation).
    pub unsafe fn tile<T: cuda_core::DType>(&self) -> Result<cutile::tensor::Tensor<T>> {
        ensure!(self.dtype.matches(T::DTYPE), "{self:?} viewed as {:?}", T::DTYPE);
        let elem = self.dtype.size_in_bytes();
        let tsize = std::mem::size_of::<T>();
        ensure!(elem == tsize, "element size {elem} vs {tsize}");
        let shape: Vec<i32> = self.shape.iter().map(|&d| d as i32).collect();
        let strides: Vec<i32> = if self.shape.len() == 2 {
            vec![self.row_stride as i32, 1]
        } else {
            let mut s = vec![1i32; self.shape.len()];
            for i in (0..self.shape.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * self.shape[i + 1] as i32;
            }
            s
        };
        let owner = Arc::new(SubAlloc { buf: self.buf.clone(), offset: self.offset, bytes: self.span_bytes().max(elem) });
        Ok(unsafe { cutile::tensor::Tensor::<T>::from_foreign(owner, shape, strides) })
    }

    /// Copy host bytes into a contiguous view (blocking).
    pub fn copy_from_host(&self, stream: &Stream, bytes: &[u8]) -> Result<()> {
        ensure!(self.is_contiguous() && bytes.len() == self.bytes(), "host copy of {} bytes into {self:?}", bytes.len());
        if bytes.is_empty() {
            return Ok(());
        }
        stream.context().bind_to_thread()?;
        unsafe { cuda_core::memory::memcpy_htod_async(self.ptr(), bytes.as_ptr(), bytes.len(), stream.cu_stream())? };
        stream.synchronize()?;
        Ok(())
    }

    /// Copy the view to the host, row by row for strided views (blocking). Debug and test path.
    pub fn to_host_bytes(&self, stream: &Stream) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.bytes()];
        if out.is_empty() {
            return Ok(out);
        }
        stream.context().bind_to_thread()?;
        if self.is_contiguous() {
            unsafe { cuda_core::memory::memcpy_dtoh_async(out.as_mut_ptr(), self.ptr(), out.len(), stream.cu_stream())? };
        } else {
            let row = self.cols() * self.dtype.size_in_bytes();
            let pitch = self.row_stride * self.dtype.size_in_bytes();
            for r in 0..self.rows() {
                unsafe {
                    cuda_core::memory::memcpy_dtoh_async(out[r * row..].as_mut_ptr(), self.ptr() + (r * pitch) as u64,
                                                         row, stream.cu_stream())?
                };
            }
        }
        stream.synchronize()?;
        Ok(out)
    }
}

/// bf16 bit patterns of a host f32 slice, round-to-nearest-even (test and setup helper).
pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes()).collect()
}

/// Decode little-endian bf16 bytes to f32.
pub fn bf16_from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
}

/// Decode little-endian f32 bytes.
pub fn f32_from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
