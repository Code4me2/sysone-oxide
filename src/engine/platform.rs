//! Platform profile (PLATFORM.md P1, P4; DESIGN §6.5, §7): what this machine measured, written by
//! `hwprobe --profile-out` and read by the engine and scheduler to size token budgets and buckets instead
//! of using constants. GB10 numbers are never extrapolated to another machine: every machine gets its own
//! profile.

use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One row of the GEMM-only forward estimate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForwardPoint {
    /// Packed tokens per forward.
    pub tokens: usize,
    pub ms: f64,
    pub tflops: f64,
    pub tokens_per_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlatformProfile {
    /// Profile format version.
    pub version: u32,
    pub device_name: String,
    /// Compute capability, e.g. `[12, 1]`.
    pub compute_capability: [i32; 2],
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED`: device memory is the host's DRAM (GB10).
    pub integrated: bool,
    pub sm_count: u32,
    /// Device memory visible to CUDA, bytes (on an integrated system this is shared with the OS).
    pub total_mem_bytes: u64,
    /// Measured device-to-device copy bandwidth, GB/s.
    pub mem_bw_gbs: f64,
    /// Best measured bf16 GEMM rate (cuBLAS, fp32 accumulate) over the probed projections, TFLOP/s.
    pub gemm_bf16_tflops: f64,
    /// Bytes of GEMM weights streamed per forward in the probe (elephant-1 layers).
    pub gemm_weight_bytes: u64,
    /// GEMM-only forward estimate vs packed tokens, ascending in `tokens`.
    pub forward: Vec<ForwardPoint>,
    /// Free text: where and when it was measured.
    #[serde(default)]
    pub note: String,
}

impl PlatformProfile {
    pub const VERSION: u32 = 1;

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let p: Self = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        p.check()?;
        Ok(p)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.check()?;
        std::fs::write(path, serde_json::to_string_pretty(self)? + "\n").with_context(|| format!("writing {}", path.display()))
    }

    pub fn check(&self) -> Result<()> {
        ensure!(self.version == Self::VERSION, "platform profile version {} (expected {})", self.version, Self::VERSION);
        ensure!(self.mem_bw_gbs > 0.0 && self.gemm_bf16_tflops > 0.0, "platform profile has no measurements");
        ensure!(self.forward.windows(2).all(|w| w[0].tokens < w[1].tokens), "forward points not ascending");
        Ok(())
    }

    /// Time to stream the GEMM weights once at the measured bandwidth (the per-forward floor), ms.
    pub fn weight_floor_ms(&self) -> f64 {
        self.gemm_weight_bytes as f64 / (self.mem_bw_gbs * 1e9) * 1e3
    }

    /// Packed tokens at which a forward's GEMM FLOPs take as long as streaming the weights: below this a
    /// forward is bandwidth-bound and batching is nearly free. `flop_per_token` is the model's GEMM FLOPs per
    /// token (2 × GEMM parameters).
    pub fn crossover_tokens(&self, flop_per_token: f64) -> usize {
        let floor_s = self.weight_floor_ms() * 1e-3;
        (floor_s * self.gemm_bf16_tflops * 1e12 / flop_per_token).ceil() as usize
    }

    /// Smallest probed token count whose throughput reaches `fraction` of the best probed tokens/s: the
    /// default token budget per step (DESIGN §7), so a step is no larger than it needs to be to reach the
    /// compute ceiling. Falls back to the largest probed point.
    pub fn token_budget(&self, fraction: f64) -> Option<usize> {
        let best = self.forward.iter().map(|p| p.tokens_per_s).fold(0.0, f64::max);
        self.forward.iter().find(|p| p.tokens_per_s >= fraction * best).or(self.forward.last()).map(|p| p.tokens)
    }

    /// Interpolated (piecewise linear in tokens) GEMM-only forward time for `tokens`, ms.
    pub fn forward_ms(&self, tokens: usize) -> Option<f64> {
        let f = &self.forward;
        let first = f.first()?;
        if tokens <= first.tokens {
            return Some(first.ms.max(self.weight_floor_ms()));
        }
        for w in f.windows(2) {
            if tokens <= w[1].tokens {
                let a = (tokens - w[0].tokens) as f64 / (w[1].tokens - w[0].tokens) as f64;
                return Some(w[0].ms + a * (w[1].ms - w[0].ms));
            }
        }
        let last = f.last()?;
        Some(last.ms * tokens as f64 / last.tokens as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Phase 0 GB10 numbers (docs/PHASE0.md).
    fn gb10() -> PlatformProfile {
        let pt = |tokens, ms: f64| ForwardPoint {
            tokens,
            ms,
            tflops: 0.0,
            tokens_per_s: tokens as f64 / (ms * 1e-3),
        };
        PlatformProfile {
            version: 1,
            device_name: "NVIDIA GB10".into(),
            compute_capability: [12, 1],
            integrated: true,
            sm_count: 48,
            total_mem_bytes: 128 << 30,
            mem_bw_gbs: 243.0,
            gemm_bf16_tflops: 101.0,
            gemm_weight_bytes: 13_840_000_000,
            forward: vec![pt(256, 90.8), pt(1024, 154.5), pt(4096, 625.4), pt(8192, 1184.1), pt(16384, 2320.1)],
            note: String::new(),
        }
    }

    #[test]
    fn roundtrip_and_derived_numbers() {
        let p = gb10();
        let s = serde_json::to_string(&p).unwrap();
        let q: PlatformProfile = serde_json::from_str(&s).unwrap();
        assert_eq!(p, q);
        assert!((p.weight_floor_ms() - 56.95).abs() < 0.1);
        // 2 × 6.92 B GEMM parameters per token.
        let x = p.crossover_tokens(2.0 * 6.92e9);
        assert!((300..600).contains(&x), "crossover {x}");
        assert_eq!(p.token_budget(0.9), Some(1024));
        assert!((p.forward_ms(2560).unwrap() - 390.0).abs() < 1.0);
    }

    #[test]
    fn bad_profiles_are_refused() {
        let mut p = gb10();
        p.version = 9;
        assert!(p.check().is_err());
        let mut p = gb10();
        p.forward.swap(0, 1);
        assert!(p.check().is_err());
    }
}

/// A PCI address `domain:bus:device.function`. nvidia-smi prints the domain with 8 hex digits
/// (`0000000F:01:00.0`), the CUDA driver with 4 (`000F:01:00.0`), so addresses are compared parsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciAddr {
    pub domain: u32,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl PciAddr {
    /// Parses `[domain:]bus:device[.function]` (hex, any width; domain and function default to 0).
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let (rest, function) = match s.rsplit_once('.') {
            Some((r, f)) => (r, u8::from_str_radix(f, 16).ok()?),
            None => (s, 0),
        };
        let parts: Vec<&str> = rest.split(':').collect();
        let (domain, bus, device) = match parts[..] {
            [d, b, v] => (u32::from_str_radix(d, 16).ok()?, b, v),
            [b, v] => (0, b, v),
            _ => return None,
        };
        Some(Self { domain, bus: u8::from_str_radix(bus, 16).ok()?, device: u8::from_str_radix(device, 16).ok()?, function })
    }
}

/// The PCI address of CUDA device `ordinal` as this process numbers it (after `CUDA_VISIBLE_DEVICES` and
/// `CUDA_DEVICE_ORDER`), from the driver. This ties a CUDA ordinal to nvidia-smi's view of the GPUs whatever the
/// enumeration order or the form of `CUDA_VISIBLE_DEVICES` (indices or UUIDs).
pub fn cuda_device_pci(ordinal: i32) -> Result<PciAddr> {
    use cuda_core::sys;
    let mut dev: sys::CUdevice = 0;
    let mut buf = [0 as std::ffi::c_char; 32];
    // SAFETY: driver queries into local out-parameters; cuInit is idempotent and creates no context.
    let r = unsafe {
        let mut r = sys::cuInit(0);
        if r == sys::cudaError_enum_CUDA_SUCCESS {
            r = sys::cuDeviceGet(&mut dev, ordinal);
        }
        if r == sys::cudaError_enum_CUDA_SUCCESS {
            r = sys::cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as i32, dev);
        }
        r
    };
    ensure!(r == sys::cudaError_enum_CUDA_SUCCESS, "CUDA device {ordinal}: cannot read its PCI address (CUDA error {r})");
    // SAFETY: the driver wrote a NUL-terminated string into the zeroed buffer (one byte is never written).
    let s = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned();
    PciAddr::parse(&s).with_context(|| format!("CUDA device {ordinal}: unparsable PCI address {s:?}"))
}

/// The rows of `nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_bus_id --format=csv,noheader`
/// that run on `gpu` (every row when `gpu` is `None`), excluding process `me`, as `pid, name, memory`. A row whose
/// bus id is missing or unparsable counts as on the GPU (contention is never under-reported).
pub fn compute_apps_on(csv: &str, gpu: Option<PciAddr>, me: u32) -> Vec<String> {
    let me = me.to_string();
    csv.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            let cols: Vec<&str> = l.split(',').map(str::trim).collect();
            if cols.first() == Some(&me.as_str()) {
                return None;
            }
            // The bus id is the last column (a process name may itself contain commas).
            let (head, bus) = match cols.split_last() {
                Some((last, rest)) if cols.len() >= 4 => (rest.join(", "), PciAddr::parse(last)),
                _ => (cols.join(", "), None),
            };
            match (gpu, bus) {
                (Some(want), Some(have)) if want != have => None,
                _ => Some(head),
            }
        })
        .collect()
}

/// Other processes currently holding a CUDA context on this process's GPU, as `nvidia-smi` lists them (`pid, name,
/// memory`), excluding this process. "This process's GPU" is CUDA device 0, i.e. the first device of
/// `CUDA_VISIBLE_DEVICES`, which is the device every benchmark and test uses; processes on the other GPUs of a
/// multi-GPU host do not count. Graphics clients (the desktop) are not compute apps.
pub fn gpu_contention() -> Result<Vec<String>> {
    gpu_contention_on(0)
}

/// [`gpu_contention`] for CUDA device `ordinal`.
pub fn gpu_contention_on(ordinal: i32) -> Result<Vec<String>> {
    let gpu = cuda_device_pci(ordinal)?;
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,process_name,used_memory,gpu_bus_id", "--format=csv,noheader"])
        .output()
        .context("running nvidia-smi")?;
    ensure!(out.status.success(), "nvidia-smi failed: {}", String::from_utf8_lossy(&out.stderr));
    Ok(compute_apps_on(&String::from_utf8_lossy(&out.stdout), Some(gpu), std::process::id()))
}

/// Guard for benchmarks: timings taken while other processes use the GPU are not comparable (Phase 1 lost
/// most of its speed numbers this way). Refuses to run while another process uses this process's GPU
/// ([`gpu_contention`]; other GPUs of the host may be busy) unless `allow_shared` is set, and returns a one-line
/// note for the run's output saying which case applied. Run benchmarks through `scripts/gpu.sh`, which also takes
/// the selected GPU's lock.
pub fn require_idle_gpu(allow_shared: bool) -> Result<String> {
    let others = gpu_contention()?;
    let locked = std::env::var_os("SYSONE_GPU_LOCKED").is_some();
    let lock_note = if locked { "gpu lock held" } else { "WARNING: not run via scripts/gpu.sh (no GPU lock)" };
    if others.is_empty() {
        return Ok(format!("gpu idle at start; {lock_note}"));
    }
    anyhow::ensure!(
        allow_shared,
        "other processes are using the GPU, so timings would not be authoritative:\n  {}\n\
         stop them, or pass --allow-shared-gpu to measure anyway",
        others.join("\n  ")
    );
    Ok(format!("SHARED GPU (not authoritative): {}; {lock_note}", others.join(" | ")))
}

#[cfg(test)]
mod contention_tests {
    use super::*;

    #[test]
    fn pci_addresses_parse_across_widths() {
        let smi = PciAddr::parse("0000000F:01:00.0").unwrap();
        assert_eq!(smi, PciAddr::parse("000F:01:00.0").unwrap());
        assert_eq!(smi, PciAddr { domain: 0xf, bus: 1, device: 0, function: 0 });
        assert_eq!(PciAddr::parse("81:00.1"), Some(PciAddr { domain: 0, bus: 0x81, device: 0, function: 1 }));
        assert_eq!(PciAddr::parse("00000000:C1:00"), Some(PciAddr { domain: 0, bus: 0xc1, device: 0, function: 0 }));
        assert_eq!(PciAddr::parse("[N/A]"), None);
        assert_eq!(PciAddr::parse(""), None);
    }

    /// Three GPUs (the x86 host): only the selected GPU's processes count, and never this process.
    #[test]
    fn contention_is_per_gpu() {
        let csv = "111, python, 2000 MiB, 00000000:01:00.0\n\
                   222, /opt/x, y/bench, 500 MiB, 00000000:41:00.0\n\
                   333, vllm, 80000 MiB, 00000000:81:00.0\n\
                   444, me, 10 MiB, 00000000:41:00.0\n";
        let g = |s| Some(PciAddr::parse(s).unwrap());
        assert_eq!(compute_apps_on(csv, g("0000:01:00.0"), 444), vec!["111, python, 2000 MiB"]);
        assert_eq!(compute_apps_on(csv, g("0000:41:00.0"), 444), vec!["222, /opt/x, y/bench, 500 MiB"]);
        assert_eq!(compute_apps_on(csv, g("0000:81:00.0"), 444), vec!["333, vllm, 80000 MiB"]);
        assert!(compute_apps_on(csv, g("0000:02:00.0"), 444).is_empty());
        assert_eq!(compute_apps_on(csv, None, 444).len(), 3);
    }

    /// GB10 (one GPU, domain 000F): as before, every other compute app counts; so do rows without a bus id.
    #[test]
    fn single_gpu_and_unparsable_rows_count() {
        let csv = "10, a, 1 MiB, 0000000F:01:00.0\n11, b, 2 MiB, [N/A]\n12, c, 3 MiB\n\n";
        let gb10 = Some(PciAddr::parse("000F:01:00.0").unwrap());
        assert_eq!(compute_apps_on(csv, gb10, 1), vec!["10, a, 1 MiB", "11, b, 2 MiB", "12, c, 3 MiB"]);
        assert_eq!(compute_apps_on(csv, gb10, 10), vec!["11, b, 2 MiB", "12, c, 3 MiB"]);
        assert!(compute_apps_on("", gb10, 1).is_empty());
    }
}
