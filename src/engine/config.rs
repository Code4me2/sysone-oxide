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

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub text_config: TextConfig,
    pub image_token_id: Option<u32>,
    pub modeling_overrides: ModelingOverrides,
}

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
    fn causal_or_head_readout_is_refused() {
        let mut c = elephant1();
        c.modeling_overrides.attention.full_attention = "causal".into();
        assert!(c.check().is_err());
        let mut c = elephant1();
        c.modeling_overrides.readout = "head".into();
        assert!(c.check().is_err());
    }
}
