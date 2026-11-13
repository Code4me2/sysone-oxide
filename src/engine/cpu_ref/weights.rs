//! bf16 safetensors, memory-mapped, read as f32 on the fly.
//!
//! The reference holds no converted copy of the big matrices: a [`Mat`] points into the mapping and each GEMM
//! converts the rows it is about to use. Small vectors (norm weights, gates, conv taps) are converted once at
//! load. Test models build [`Mat::F32`] directly.

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// A row-major `[rows, cols]` matrix, bf16 in a mapped file or owned f32.
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    store: Store,
}

#[derive(Clone)]
enum Store {
    /// `off` is the byte offset of element 0 in the mapping.
    Bf16 { map: Arc<Mmap>, off: usize },
    F32(Arc<Vec<f32>>),
}

impl Mat {
    pub fn from_f32(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data does not match {rows}x{cols}");
        Mat { rows, cols, store: Store::F32(Arc::new(data)) }
    }

    /// Row `r` as f32 into `out` (`cols` long). bf16 -> f32 is exact.
    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        debug_assert!(r < self.rows && out.len() == self.cols);
        match &self.store {
            Store::F32(v) => out.copy_from_slice(&v[r * self.cols..(r + 1) * self.cols]),
            Store::Bf16 { map, off } => {
                let base = off + 2 * r * self.cols;
                let bytes = &map[base..base + 2 * self.cols];
                for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                }
            }
        }
    }

    pub fn row(&self, r: usize) -> Vec<f32> {
        let mut v = vec![0.0; self.cols];
        self.row_into(r, &mut v);
        v
    }

    /// The whole matrix as f32 (for vectors and small tensors).
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = vec![0.0; self.rows * self.cols];
        for (r, chunk) in v.chunks_exact_mut(self.cols.max(1)).enumerate() {
            self.row_into(r, chunk);
        }
        v
    }
}

struct Entry {
    file: usize,
    dtype: String,
    shape: Vec<usize>,
    /// Absolute byte range in the file.
    begin: usize,
    end: usize,
}

/// The tensors of a checkpoint directory: `model.safetensors`, or the shards named by
/// `model.safetensors.index.json`.
pub struct SafeTensors {
    maps: Vec<Arc<Mmap>>,
    entries: HashMap<String, Entry>,
}

impl SafeTensors {
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let files: Vec<String> = if index.exists() {
            let text = std::fs::read_to_string(&index).with_context(|| format!("reading {}", index.display()))?;
            let v: Value = serde_json::from_str(&text)?;
            let map = v["weight_map"].as_object().context("index has no weight_map")?;
            let mut f: Vec<String> = map.values().filter_map(|x| x.as_str().map(str::to_owned)).collect();
            f.sort();
            f.dedup();
            f
        } else {
            vec!["model.safetensors".to_owned()]
        };
        let mut st = SafeTensors { maps: Vec::new(), entries: HashMap::new() };
        for name in files {
            st.add_file(&dir.join(name))?;
        }
        Ok(st)
    }

    fn add_file(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the checkpoint is read-only for the life of the process; nobody truncates it under us.
        let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
        ensure!(map.len() >= 8, "{} is too short for safetensors", path.display());
        let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        ensure!(8 + n <= map.len(), "{}: header length {n} past end of file", path.display());
        let header: Value = serde_json::from_slice(&map[8..8 + n]).context("parsing safetensors header")?;
        let fi = self.maps.len();
        for (name, e) in header.as_object().context("header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = e["dtype"].as_str().context("tensor without dtype")?.to_owned();
            let shape = e["shape"].as_array().context("tensor without shape")?.iter()
                .map(|d| d.as_u64().map(|d| d as usize).context("bad dim"))
                .collect::<Result<Vec<_>>>()?;
            let offs = e["data_offsets"].as_array().context("tensor without data_offsets")?;
            let begin = 8 + n + offs[0].as_u64().context("bad offset")? as usize;
            let end = 8 + n + offs[1].as_u64().context("bad offset")? as usize;
            ensure!(end <= map.len(), "tensor {name} past end of {}", path.display());
            self.entries.insert(name.clone(), Entry { file: fi, dtype, shape, begin, end });
        }
        self.maps.push(Arc::new(map));
        Ok(())
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// A bf16 tensor viewed as `[shape[0], prod(shape[1..])]` (a vector is `[1, n]`).
    pub fn mat(&self, name: &str) -> Result<Mat> {
        let e = self.entries.get(name).with_context(|| format!("tensor {name} not in checkpoint"))?;
        if e.dtype != "BF16" {
            bail!("tensor {name} is {}, the CPU reference reads bf16 only", e.dtype);
        }
        let numel: usize = e.shape.iter().product();
        ensure!(e.end - e.begin == 2 * numel, "tensor {name}: {} bytes for {numel} bf16", e.end - e.begin);
        let (rows, cols) = match e.shape.len() {
            0 => (1, 1),
            1 => (1, e.shape[0]),
            _ => (e.shape[0], numel / e.shape[0]),
        };
        Ok(Mat { rows, cols, store: Store::Bf16 { map: self.maps[e.file].clone(), off: e.begin } })
    }

    /// A bf16 tensor converted to f32, with its element count checked.
    pub fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let v = self.mat(name)?.to_vec();
        ensure!(v.len() == len, "tensor {name} has {} elements, expected {len}", v.len());
        Ok(v)
    }
}
