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
