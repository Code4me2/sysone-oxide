//! Loader and launcher for the cuda-oxide SIMT kernels in `kernels-simt/` (DESIGN.md §6.4, docs/G1.md).
//!
//! The kernels are compiled by `kernels-simt/build.sh <arch>` (toolchain 1.99.1 + cargo-oxide) into a checked-in
//! PTX artifact with a hash manifest; `build.rs` embeds every artifact whose manifest matches the current kernel
//! sources. At run time the module for the device's architecture is JIT-loaded once per device (the driver caches
//! the JIT result on disk). A missing or stale artifact is an error naming the rebuild command; nothing falls back
//! to another backend silently.
//!
//! Launches are untyped (`AsyncKernelLaunchBuilder` with raw device addresses and scalars): the argument lists in
//! the callers must match the `#[kernel]` signatures in `kernels-simt/src/main.rs`.

use anyhow::{Context as _, Result, anyhow, bail};
use cuda_async::device_context::{get_default_device, load_module_from_ptx, with_device};
use cuda_async::device_operation::{BoxedDeviceOp, DeviceOp};
use cuda_async::launch::AsyncKernelLaunchBuilder;
use cuda_core::{Function, LaunchConfig, Module};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

mod artifacts {
    include!(concat!(env!("OUT_DIR"), "/simt_artifacts.rs"));
}

/// Architectures with an embedded SIMT artifact, and the reason any other artifact is unusable.
pub fn artifacts() -> Vec<(&'static str, std::result::Result<(), &'static str>)> {
    artifacts::ARTIFACTS.iter().map(|(arch, r)| (*arch, r.map(|_| ()))).collect()
}

/// The PTX for compute capability `major.minor`: the artifact built for exactly `sm_<major><minor>` (with or
/// without the `a` suffix). PTX is forward-compatible, but a kernel tuned and validated for one architecture is
/// not silently run on another: build an artifact for it.
fn ptx_for(major: i32, minor: i32) -> Result<&'static str> {
    let want = format!("sm_{major}{minor}");
    let mut seen = Vec::new();
    for (arch, ptx) in artifacts::ARTIFACTS {
        if arch.trim_end_matches('a') == want {
            return ptx.map_err(|e| anyhow!("SIMT kernels for {arch}: {e}; rebuild with `kernels-simt/build.sh {arch}`"));
        }
        seen.push(*arch);
    }
    bail!(
        "no SIMT kernel artifact for {want} (have: {}); build it with `kernels-simt/build.sh {want}`",
        if seen.is_empty() { "none".to_string() } else { seen.join(", ") }
    )
}

/// A loaded SIMT module with its functions resolved by name (and their dynamic shared memory limits raised).
pub struct SimtModule {
    module: Arc<Module>,
    functions: Mutex<HashMap<&'static str, Arc<Function>>>,
}

impl SimtModule {
    /// The module for `device`, loaded on first use.
    pub fn get(device: usize) -> Result<Arc<SimtModule>> {
        static MODULES: OnceLock<Mutex<HashMap<usize, Arc<SimtModule>>>> = OnceLock::new();
        let mut map = MODULES.get_or_init(Default::default).lock().map_err(|_| anyhow!("SIMT module map poisoned"))?;
        if let Some(m) = map.get(&device) {
            return Ok(m.clone());
        }
        let (major, minor) = with_device(device, |ctx| ctx.compute_capability())??;
        let ptx = ptx_for(major, minor)?;
        let module = load_module_from_ptx(ptx, device).with_context(|| format!("loading the SIMT PTX on device {device}"))?;
        let m = Arc::new(SimtModule { module, functions: Mutex::new(HashMap::new()) });
        map.insert(device, m.clone());
        Ok(m)
    }

    /// The kernel `name`, allowed `dyn_smem` bytes of dynamic shared memory.
    pub fn function(&self, name: &'static str, dyn_smem: u32) -> Result<Arc<Function>> {
        let mut fns = self.functions.lock().map_err(|_| anyhow!("SIMT function map poisoned"))?;
        if let Some(f) = fns.get(name) {
            return Ok(f.clone());
        }
        let f = self.module.load_function(name).with_context(|| format!("SIMT kernel {name} not in the artifact"))?;
        if dyn_smem > 48 * 1024 {
            // SAFETY: the handle comes from a live module this struct keeps alive.
            let r = unsafe {
                cuda_core::sys::cuFuncSetAttribute(
                    f.cu_function(),
                    cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dyn_smem as i32,
                )
            };
            if r != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
                bail!("SIMT kernel {name}: cannot allow {dyn_smem} B of dynamic shared memory (CUDA error {r})");
            }
        }
        let f = Arc::new(f);
        fns.insert(name, f.clone());
        Ok(f)
    }
}

/// One kernel argument: a device address or a 32-bit scalar.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    Ptr(u64),
    I32(i32),
    F32(f32),
}

/// An enqueueable launch of SIMT kernel `name` on the default device.
///
/// # Safety
/// `args` must match the kernel's parameter list exactly, the grid/block/shared-memory shape must be one the
/// kernel supports, and every addressed buffer must stay alive and unaliased by other writers until the returned
/// operation has completed on its stream.
pub unsafe fn launch(
    name: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dyn_smem: u32,
    args: &[Arg],
) -> Result<BoxedDeviceOp<()>> {
    let device = get_default_device();
    let f = SimtModule::get(device)?.function(name, dyn_smem)?;
    let mut b = AsyncKernelLaunchBuilder::new(f);
    for a in args {
        match *a {
            Arg::Ptr(p) => b.push_scalar_arg(p),
            Arg::I32(v) => b.push_scalar_arg(v),
            Arg::F32(v) => b.push_scalar_arg(v),
        };
    }
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: dyn_smem };
    // SAFETY: forwarded to the caller (see above).
    Ok(unsafe { b.finalize_unchecked(cfg) }.boxed())
}

/// `out[i] = input[i] * 3 + i` for `i < n` (u32): the integration probe of the SIMT build/load path.
///
/// # Safety
/// `input` and `out` must address at least `n` u32 elements, live until the operation completes.
pub unsafe fn probe(input: u64, out: u64, n: i32) -> Result<BoxedDeviceOp<()>> {
    let blocks = (n.max(0) as u32).div_ceil(256);
    unsafe { launch("simt_probe", (blocks.max(1), 1, 1), (256, 1, 1), 0, &[Arg::Ptr(input), Arg::Ptr(out), Arg::I32(n)]) }
}
