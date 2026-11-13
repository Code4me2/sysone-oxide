//! Checkpoint loading without a resident page-cache copy (DESIGN §6.5, PLATFORM.md P1/P2).
//!
//! [`Checkpoint`] reads the safetensors headers (`model.safetensors`, or the shards named by
//! `model.safetensors.index.json`) and keeps only text weights (`model.language_model.*`, `lm_head.*`);
//! shards holding none of them (the NVFP4 checkpoint's vision `model_mtp.safetensors`) are never opened.
//!
//! [`Streamer`] copies file ranges into device memory through a ring of pinned staging buffers: one
//! sequential pass per file, reads overlapping the previous chunk's host-to-device copy. With
//! [`CacheMode::Direct`] the file is read with `O_DIRECT` and never enters the page cache; with
//! [`CacheMode::DropBehind`] it is read normally and each chunk's pages are dropped with
//! `posix_fadvise(DONTNEED)` as soon as they are in the staging buffer. Either way no page-cache copy of
//! the weights stays resident, which matters on unified-memory systems where it would compete with the
//! device copy for the same DRAM.

use crate::engine::graph::tensor::{DevTensor, GraphDType};
use anyhow::{Context as _, Result, bail, ensure};
use cuda_core::{Event, PinnedHostBuffer, Stream};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Is this tensor part of the text model v1 serves?
pub fn is_text_weight(name: &str) -> bool {
    name.starts_with("model.language_model.") || name.starts_with("lm_head.")
}

/// One tensor in a safetensors file. Offsets are absolute file offsets.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub file: usize,
    pub dtype: GraphDType,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn bytes(&self) -> usize {
        (self.end - self.start) as usize
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Deserialize)]
struct IndexFile {
    weight_map: BTreeMap<String, String>,
}

/// The text-model tensors of a checkpoint directory.
pub struct Checkpoint {
    pub dir: PathBuf,
    /// Shard files actually used, in index order.
    pub files: Vec<PathBuf>,
    tensors: HashMap<String, TensorInfo>,
    /// Shards named by the index that hold no text weights (not opened).
    pub skipped_files: Vec<PathBuf>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let (files, skipped) = if index.exists() {
            let text = std::fs::read_to_string(&index).with_context(|| format!("reading {}", index.display()))?;
            let idx: IndexFile = serde_json::from_str(&text).with_context(|| format!("parsing {}", index.display()))?;
            let mut used: Vec<String> = Vec::new();
            let mut all: Vec<String> = Vec::new();
            for (name, file) in &idx.weight_map {
                if !all.contains(file) {
                    all.push(file.clone());
                }
                if is_text_weight(name) && !used.contains(file) {
                    used.push(file.clone());
                }
            }
            let skipped = all.iter().filter(|f| !used.contains(f)).map(|f| dir.join(f)).collect();
            (used.iter().map(|f| dir.join(f)).collect::<Vec<_>>(), skipped)
        } else {
            (vec![dir.join("model.safetensors")], vec![])
        };
        ensure!(!files.is_empty(), "{} has no text weights", dir.display());
        let mut tensors = HashMap::new();
        for (fi, path) in files.iter().enumerate() {
            for t in read_header(path, fi)? {
                if is_text_weight(&t.name) {
                    tensors.insert(t.name.clone(), t);
                }
            }
        }
        Ok(Self { dir: dir.to_path_buf(), files, tensors, skipped_files: skipped })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not in {}", self.dir.display()))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn file_path(&self, t: &TensorInfo) -> &Path {
        &self.files[t.file]
    }

    /// Read a small tensor (scalars, norms) straight from the file, bypassing the page cache bookkeeping.
    pub fn read_small(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.get(name)?;
        ensure!(t.bytes() <= 1 << 20, "read_small of {} bytes ({name})", t.bytes());
        let f = File::open(self.file_path(t))?;
        let mut buf = vec![0u8; t.bytes()];
        f.read_exact_at(&mut buf, t.start)?;
        Ok(buf)
    }

    /// A scalar f32 tensor (`weight_global_scale`, `input_global_scale`).
    pub fn read_f32_scalar(&self, name: &str) -> Result<f32> {
        let t = self.get(name)?;
        ensure!(t.dtype == GraphDType::F32 && t.numel() == 1, "{name} is {:?} {:?}, expected one f32", t.dtype, t.shape);
        let b = self.read_small(name)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

fn read_header(path: &Path, file: usize) -> Result<Vec<TensorInfo>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact_at(&mut len, 0)?;
    let n = u64::from_le_bytes(len);
    ensure!(n < 1 << 28, "{}: header of {n} bytes", path.display());
    let mut hdr = vec![0u8; n as usize];
    f.read_exact_at(&mut hdr, 8)?;
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&hdr)?;
    let base = 8 + n;
    let file_len = f.metadata()?.len();
    let mut out = Vec::new();
    for (name, v) in raw {
        if name == "__metadata__" {
            continue;
        }
        let e: RawEntry = serde_json::from_value(v).with_context(|| format!("{}: entry {name}", path.display()))?;
        let Some(dtype) = GraphDType::from_safetensors(&e.dtype) else {
            if is_text_weight(&name) {
                bail!("{}: {name} has unsupported dtype {}", path.display(), e.dtype);
            }
            continue;
        };
        let (start, end) = (base + e.data_offsets[0], base + e.data_offsets[1]);
        ensure!(end <= file_len && start <= end, "{}: {name} range {start}..{end} past EOF", path.display());
        let numel: usize = e.shape.iter().product();
        ensure!((end - start) as usize == numel * dtype.size_in_bytes(), "{}: {name} size mismatch", path.display());
        out.push(TensorInfo { name, file, dtype, shape: e.shape, start, end });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------
// Page-cache control and measurement (Linux).

mod sys {
    use std::ffi::{c_int, c_long, c_void};
    unsafe extern "C" {
        pub fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
        pub fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
        pub fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
        pub fn sysconf(name: c_int) -> c_long;
    }
    pub const POSIX_FADV_SEQUENTIAL: c_int = 2;
    pub const POSIX_FADV_DONTNEED: c_int = 4;
    pub const PROT_READ: c_int = 1;
    pub const MAP_SHARED: c_int = 1;
    pub const SC_PAGESIZE: c_int = 30;
    #[cfg(target_arch = "aarch64")]
    pub const O_DIRECT: c_int = 0o200000;
    #[cfg(not(target_arch = "aarch64"))]
    pub const O_DIRECT: c_int = 0o40000;
}

fn page_size() -> usize {
    let p = unsafe { sys::sysconf(sys::SC_PAGESIZE) };
    if p > 0 { p as usize } else { 4096 }
}

/// Drop a file's clean pages from the page cache (for cold-load measurements and after loading).
pub fn drop_file_cache(path: &Path) -> Result<()> {
    let f = File::open(path)?;
    let r = unsafe { sys::posix_fadvise(f.as_raw_fd(), 0, 0, sys::POSIX_FADV_DONTNEED) };
    ensure!(r == 0, "posix_fadvise(DONTNEED) on {}: errno {r}", path.display());
    Ok(())
}

/// Bytes of `path` currently resident in the page cache (`mincore` over a read-only mapping; the mapping
/// faults nothing in).
pub fn file_resident_bytes(path: &Path) -> Result<u64> {
    let f = File::open(path)?;
    let len = f.metadata()?.len() as usize;
    if len == 0 {
        return Ok(0);
    }
    let ps = page_size();
    let addr = unsafe { sys::mmap(std::ptr::null_mut(), len, sys::PROT_READ, sys::MAP_SHARED, f.as_raw_fd(), 0) };
    ensure!(addr as isize != -1, "mmap of {} failed", path.display());
    let pages = len.div_ceil(ps);
    let mut vec = vec![0u8; pages];
    let r = unsafe { sys::mincore(addr, len, vec.as_mut_ptr()) };
    unsafe { sys::munmap(addr, len) };
    ensure!(r == 0, "mincore on {} failed", path.display());
    Ok(vec.iter().filter(|&&b| b & 1 != 0).count() as u64 * ps as u64)
}

/// `Cached:` and `MemAvailable:` from `/proc/meminfo`, in bytes.
pub fn meminfo_cached_available() -> Result<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo")?;
    let field = |key: &str| -> Result<u64> {
        let line = text.lines().find(|l| l.starts_with(key)).with_context(|| format!("{key} not in /proc/meminfo"))?;
        let kb: u64 = line[key.len()..].trim().trim_end_matches("kB").trim().parse()?;
        Ok(kb * 1024)
    };
    Ok((field("Cached:")?, field("MemAvailable:")?))
}

// ---------------------------------------------------------------------------------------------------------
// Streaming upload.

/// How file pages are kept out of the page cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheMode {
    /// `O_DIRECT` reads (falls back to `DropBehind` if the filesystem refuses `O_DIRECT`).
    Direct,
    /// Buffered reads, each chunk's pages dropped with `posix_fadvise(DONTNEED)` right after the read.
    DropBehind,
    /// Buffered reads, page cache left alone (for comparison only).
    Keep,
}

#[derive(Clone, Debug)]
pub struct StreamOptions {
    pub mode: CacheMode,
    /// Staging chunk size (a multiple of the page size).
    pub chunk_bytes: usize,
    /// Number of pinned staging buffers in the ring.
    pub staging_buffers: usize,
    /// Reads separated by at most this many unused bytes are merged into one sequential run.
    pub merge_gap: u64,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self { mode: CacheMode::Direct, chunk_bytes: 32 << 20, staging_buffers: 3, merge_gap: 8 << 20 }
    }
}

/// One copy from a file range into device memory.
struct CopyItem {
    file: usize,
    start: u64,
    len: u64,
    dst: DevTensor,
    dst_offset: u64,
}

/// What a [`Streamer`] run did.
#[derive(Clone, Debug, Default)]
pub struct StreamStats {
    pub bytes_copied: u64,
    pub bytes_read: u64,
    pub seconds: f64,
    pub items: usize,
    pub runs: usize,
    /// Whether `O_DIRECT` was used for every file.
    pub direct: bool,
}

/// Collects file-to-device copies, then performs them in one sequential pass per file.
pub struct Streamer<'a> {
    ckpt: &'a Checkpoint,
    items: Vec<CopyItem>,
}

impl<'a> Streamer<'a> {
    pub fn new(ckpt: &'a Checkpoint) -> Self {
        Self { ckpt, items: Vec::new() }
    }

    /// Copy the whole tensor `name` into `dst` (a contiguous device view of exactly its size).
    pub fn tensor(&mut self, name: &str, dst: &DevTensor) -> Result<()> {
        let t = self.ckpt.get(name)?;
        ensure!(dst.is_contiguous() && dst.bytes() == t.bytes(), "{name}: {} bytes into {dst:?}", t.bytes());
        ensure!(dst.dtype.size_in_bytes() == t.dtype.size_in_bytes(), "{name}: dtype {:?} into {dst:?}", t.dtype);
        self.items.push(CopyItem { file: t.file, start: t.start, len: t.bytes() as u64, dst: dst.clone(), dst_offset: 0 });
        Ok(())
    }

    /// Copy rows `rows` (dim 0) of tensor `name` into consecutive rows of `dst`.
    pub fn rows(&mut self, name: &str, rows: &[u32], dst: &DevTensor) -> Result<()> {
        let t = self.ckpt.get(name)?;
        ensure!(!t.shape.is_empty(), "{name} is a scalar");
        let row_bytes = (t.bytes() / t.shape[0]) as u64;
        ensure!(dst.is_contiguous() && dst.bytes() as u64 == row_bytes * rows.len() as u64, "{name}: rows into {dst:?}");
        for (i, &r) in rows.iter().enumerate() {
            ensure!((r as usize) < t.shape[0], "{name}: row {r} of {}", t.shape[0]);
            self.items.push(CopyItem {
                file: t.file,
                start: t.start + r as u64 * row_bytes,
                len: row_bytes,
                dst: dst.clone(),
                dst_offset: i as u64 * row_bytes,
            });
        }
        Ok(())
    }

    pub fn pending_bytes(&self) -> u64 {
        self.items.iter().map(|i| i.len).sum()
    }

    /// Perform every queued copy on `stream`; returns once all of them have completed.
    pub fn run(self, stream: &Arc<Stream>, opts: &StreamOptions) -> Result<StreamStats> {
        let t0 = Instant::now();
        let ctx = stream.context().clone();
        let ps = page_size();
        ensure!(opts.chunk_bytes % ps == 0 && opts.chunk_bytes > 0, "chunk size must be a page multiple");
        let nbuf = opts.staging_buffers.max(2);
        // Pre-touched (zeroed) pinned staging ring; each slot has an event marking its last copy done.
        let mut staging: Vec<PinnedHostBuffer<u8>> =
            (0..nbuf).map(|_| PinnedHostBuffer::zeroed(&ctx, opts.chunk_bytes + ps)).collect::<Result<_, _>>()?;
        let events: Vec<Event> = (0..nbuf).map(|_| ctx.new_event()).collect::<Result<_, _>>()?;
        let mut used = vec![false; nbuf];
        let mut slot = 0usize;
        let mut stats = StreamStats { items: self.items.len(), direct: true, ..Default::default() };

        let mut by_file: BTreeMap<usize, Vec<CopyItem>> = BTreeMap::new();
        for it in self.items {
            by_file.entry(it.file).or_default().push(it);
        }
        for (file, mut items) in by_file {
            items.sort_by_key(|i| i.start);
            let path = &self.ckpt.files[file];
            let (f, direct) = open_for(path, opts.mode)?;
            stats.direct &= direct;
            let fd = f.as_raw_fd();
            unsafe { sys::posix_fadvise(fd, 0, 0, sys::POSIX_FADV_SEQUENTIAL) };
            // Merge items into sequential runs.
            let mut runs: Vec<(u64, u64, std::ops::Range<usize>)> = Vec::new();
            for (i, it) in items.iter().enumerate() {
                match runs.last_mut() {
                    Some((_, end, r)) if it.start <= *end + opts.merge_gap => {
                        *end = (*end).max(it.start + it.len);
                        r.end = i + 1;
                    }
                    _ => runs.push((it.start, it.start + it.len, i..i + 1)),
                }
            }
            stats.runs += runs.len();
            for (rstart, rend, range) in runs {
                let align = if direct { ps as u64 } else { 1 };
                let mut pos = rstart / align * align;
                let mut first = range.start;
                while pos < rend {
                    let want = (opts.chunk_bytes as u64).min((rend - pos).div_ceil(align) * align);
                    // Reuse the slot only after its previous copies finished.
                    if used[slot] {
                        events[slot].synchronize()?;
                    }
                    let buf = &mut staging[slot];
                    let base = buf.as_mut_ptr();
                    let pad = base.align_offset(ps);
                    let dst = &mut buf.as_mut_slice()[pad..pad + want as usize];
                    let got = read_full(&f, dst, pos)? as u64;
                    ensure!(got > 0 || pos >= rend, "{}: short read at {pos}", path.display());
                    stats.bytes_read += got;
                    if opts.mode == CacheMode::DropBehind || (opts.mode == CacheMode::Direct && !direct) {
                        unsafe { sys::posix_fadvise(fd, pos as i64, got as i64, sys::POSIX_FADV_DONTNEED) };
                    }
                    let (cs, ce) = (pos, pos + got);
                    let host = unsafe { staging[slot].as_ptr().add(pad) };
                    ctx.bind_to_thread()?;
                    for it in &items[first..range.end] {
                        let (is, ie) = (it.start, it.start + it.len);
                        if is >= ce {
                            break;
                        }
                        let (s, e) = (is.max(cs), ie.min(ce));
                        if s < e {
                            let d = it.dst.ptr() + it.dst_offset + (s - is);
                            unsafe {
                                cuda_core::memory::memcpy_htod_async(d, host.add((s - cs) as usize), (e - s) as usize,
                                                                     stream.cu_stream())?
                            };
                            stats.bytes_copied += e - s;
                        }
                    }
                    while first < range.end && items[first].start + items[first].len <= ce {
                        first += 1;
                    }
                    events[slot].record(stream)?;
                    used[slot] = true;
                    slot = (slot + 1) % nbuf;
                    pos = ce;
                    if got < want {
                        ensure!(pos >= rend, "{}: EOF at {pos} before {rend}", path.display());
                        break;
                    }
                }
            }
        }
        stream.synchronize()?;
        drop(staging);
        stats.seconds = t0.elapsed().as_secs_f64();
        Ok(stats)
    }
}

fn open_for(path: &Path, mode: CacheMode) -> Result<(File, bool)> {
    if mode == CacheMode::Direct
        && let Ok(f) = std::fs::OpenOptions::new().read(true).custom_flags(sys::O_DIRECT).open(path)
    {
        return Ok((f, true));
    }
    Ok((File::open(path).with_context(|| format!("opening {}", path.display()))?, false))
}

/// Read until `buf` is full or EOF; returns bytes read.
fn read_full(f: &File, buf: &mut [u8], off: u64) -> Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match f.read_at(&mut buf[n..], off + n as u64) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e).with_context(|| format!("pread at {}", off + n as u64)),
        }
    }
    Ok(n)
}

/// Whether the device shares physical memory with the host (`CU_DEVICE_ATTRIBUTE_INTEGRATED`): true on
/// GB10, false on discrete GPUs.
pub fn device_is_integrated(ctx: &Arc<cuda_core::Context>) -> Result<bool> {
    use cuda_core::IntoResult;
    ctx.bind_to_thread()?;
    let mut v = 0i32;
    unsafe {
        cuda_core::sys::cuDeviceGetAttribute(&mut v, cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_INTEGRATED,
                                             ctx.cu_device())
            .result()?
    };
    Ok(v != 0)
}
