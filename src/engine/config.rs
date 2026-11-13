//! `config.json` + `readout.json` of a System One checkpoint (DESIGN.md §3).
//!
//! Only what v1 can serve is accepted: Qwen3.5 text decoder, `slot_lm` readout, block-bidirectional full
//! attention over causal linear attention. Anything else is refused at load instead of served differently
//! from how it was trained.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub mrope_interleaved: bool,
    pub rope_type: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<LayerType>,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
    // full attention
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attn_output_gate: bool,
    pub attention_bias: bool,
    pub rope_parameters: RopeParameters,
    // linear attention (GatedDeltaNet)
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AttentionOverrides {
    pub full_attention: String,
    pub linear_attention: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelingOverrides {
    pub attention: AttentionOverrides,
    pub readout: String,
    pub n_slots: usize,
}

/// The `quantization_config` of a compressed-tensors checkpoint (only what selects the profile).
#[derive(Clone, Debug, Deserialize)]
pub struct QuantizationConfig {
    pub format: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub text_config: TextConfig,
    pub image_token_id: Option<u32>,
    pub modeling_overrides: ModelingOverrides,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

/// Weight format of a checkpoint's projections, which selects the engine (`--engine bf16 | nvfp4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    /// Every projection bf16 (`baselabs/elephant-1`).
    Bf16,
    /// compressed-tensors `nvfp4-pack-quantized` W4A4 on the MLP projections, everything else bf16
    /// (`Code4me2/elephant-1-NVFP4-MLP`, DESIGN §3.1).
    Nvfp4,
}

impl WeightFormat {
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

/// What a checkpoint needs from the server beyond `config.json`'s architecture (DESIGN §13 item 9): its weight
/// format and its request limits. Derived from `config.json` ([`ModelConfig::profile`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointProfile {
    pub weights: WeightFormat,
    /// Longest prefix + one question suffix (qev `max_branch_tokens`).
    pub max_branch_tokens: usize,
    /// Longest prefix + all suffixes of one request (qev `max_request_tokens`).
    pub max_request_tokens: usize,
    /// Where the limits come from, for logs.
    pub limits_source: &'static str,
}

/// The reference's limits (qev defaults; DESIGN §1).
pub const QEV_MAX_BRANCH_TOKENS: usize = 32768;
pub const QEV_MAX_REQUEST_TOKENS: usize = 65536;
/// The NVFP4 checkpoint's documented limit (its calibration data reaches 16,017 tokens). E1 established that the
/// checkpoint's own evaluation applies it per evaluated question, i.e. to prefix + one suffix (reference/eval/
/// README.md, "Eligibility"): it bounds `max_branch_tokens`; the request total keeps qev's limit.
pub const NVFP4_MAX_BRANCH_TOKENS: usize = 16384;

impl ModelConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        cfg.check()?;
        Ok(cfg)
    }

    /// Refuse configurations v1 does not implement.
    pub fn check(&self) -> Result<()> {
        let t = &self.text_config;
        ensure!(self.model_type == "qwen3_5", "model_type {:?} is not qwen3_5", self.model_type);
        ensure!(t.layer_types.len() == t.num_hidden_layers, "layer_types does not match num_hidden_layers");
        ensure!(t.hidden_act == "silu", "hidden_act {:?} unsupported", t.hidden_act);
        ensure!(!t.tie_word_embeddings, "tied embeddings unsupported");
        ensure!(t.attn_output_gate, "attention without output gate unsupported");
        ensure!(!t.attention_bias, "attention bias unsupported");
        ensure!(t.num_attention_heads % t.num_key_value_heads == 0, "heads not divisible by kv heads");
        ensure!(t.linear_num_value_heads % t.linear_num_key_heads == 0, "v heads not divisible by k heads");
        ensure!(t.rope_parameters.rope_type == "default", "rope_type {:?} unsupported", t.rope_parameters.rope_type);
        let rot = self.rotary_dim();
        ensure!(rot % 2 == 0 && rot <= t.head_dim, "rotary dim {rot} invalid for head_dim {}", t.head_dim);
        let o = &self.modeling_overrides;
        ensure!(o.attention.full_attention == "block_bidirectional", "full_attention mask {:?} unsupported",
                o.attention.full_attention);
        ensure!(o.attention.linear_attention == "causal", "linear_attention mask {:?} unsupported",
                o.attention.linear_attention);
        ensure!(o.readout == "slot_lm", "readout {:?} unsupported in v1", o.readout);
        Ok(())
    }

    /// The checkpoint profile: bf16 checkpoints get qev's limits; `nvfp4-pack-quantized` gets the NVFP4 engine and
    /// its documented 16,384-token limit per branch. Any other quantization format is refused.
    pub fn profile(&self) -> Result<CheckpointProfile> {
        match self.quantization_config.as_ref().map(|q| q.format.as_str()) {
            None => Ok(CheckpointProfile {
                weights: WeightFormat::Bf16,
                max_branch_tokens: QEV_MAX_BRANCH_TOKENS,
                max_request_tokens: QEV_MAX_REQUEST_TOKENS,
                limits_source: "qev defaults",
            }),
            Some("nvfp4-pack-quantized") => Ok(CheckpointProfile {
                weights: WeightFormat::Nvfp4,
                max_branch_tokens: NVFP4_MAX_BRANCH_TOKENS,
                max_request_tokens: QEV_MAX_REQUEST_TOKENS,
                limits_source: "NVFP4 checkpoint profile (documented 16,384-token limit per evaluated question)",
            }),
            Some(f) => anyhow::bail!("quantization format {f:?} is not supported (bf16 or nvfp4-pack-quantized)"),
        }
    }

    pub fn rotary_dim(&self) -> usize {
        let t = &self.text_config;
        (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor) as usize
    }

    /// Channels of the GatedDeltaNet conv1d: q and k (key dim each) plus v (value dim).
    pub fn linear_conv_dim(&self) -> usize {
        let t = &self.text_config;
        2 * t.linear_num_key_heads * t.linear_key_head_dim + t.linear_num_value_heads * t.linear_value_head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parts of elephant-1's config.json the engine reads (layer_types abbreviated to the real pattern).
    fn elephant1() -> ModelConfig {
        let layer_types: Vec<&str> = (0..32)
            .map(|i| if i % 4 == 3 { "full_attention" } else { "linear_attention" })
            .collect();
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "text_config": {
                "attention_bias": false, "attn_output_gate": true, "eos_token_id": 248044, "head_dim": 256,
                "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 12288, "layer_types": layer_types,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
                "linear_num_value_heads": 32, "linear_value_head_dim": 128, "num_attention_heads": 16,
                "num_hidden_layers": 32, "num_key_value_heads": 4, "rms_norm_eps": 1e-6,
                "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                                    "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
                "tie_word_embeddings": false, "vocab_size": 248320
            },
            "modeling_overrides": {
                "attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"},
                "readout": "slot_lm", "head_dim": 256, "n_slots": 256
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn elephant1_is_accepted() {
        let c = elephant1();
        c.check().unwrap();
        assert_eq!(c.rotary_dim(), 64);
        assert_eq!(c.linear_conv_dim(), 8192);
        let full = c.text_config.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count();
        assert_eq!(full, 8);
    }

    #[test]
    fn profile_follows_the_quantization_format() {
        let c = elephant1();
        let p = c.profile().unwrap();
        assert_eq!((p.weights, p.max_branch_tokens, p.max_request_tokens), (WeightFormat::Bf16, 32768, 65536));
        let mut c = elephant1();
        c.quantization_config = Some(QuantizationConfig { format: "nvfp4-pack-quantized".into() });
        let p = c.profile().unwrap();
        assert_eq!((p.weights, p.max_branch_tokens, p.max_request_tokens), (WeightFormat::Nvfp4, 16384, 65536));
        c.quantization_config = Some(QuantizationConfig { format: "float-quantized".into() });
        assert!(c.profile().is_err());
    }

    #[test]
    fn causal_or_head_readout_is_refused() {
        let mut c = elephant1();
        c.modeling_overrides.attention.full_attention = "causal".into();
        assert!(c.check().is_err());
        let mut c = elephant1();
        c.modeling_overrides.readout = "head".into();
        assert!(c.check().is_err());
    }
}
