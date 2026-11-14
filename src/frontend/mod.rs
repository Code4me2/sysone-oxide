//! Request → branches: schema, rendering, chat template, tokenization, prefix split, chunking, debias, limits,
//! and answer assembly with calibration (DESIGN.md §4). CPU only. Owner: W2.
//!
//! Use from a server (W5):
//! ```text
//! let req = SystemOneRequest::from_body(&body)?;            // 422 ApiError on failure
//! schema::check_model(&req.model, served_name)?;            // 404
//! let plan = frontend.plan(&req, debias)?;                   // PlanError::Value(msg) → ApiError::value_error(msg)
//! let first = plan.push_into(&mut batch)?;                   // contract::DecisionBatch
//! ... engine.run(&batch) → SlotLogits ...
//! let out = frontend.assemble(&req, &plan, |i| logits.row(first + i, plan.branches[i].n_slots()), None)?;
//! let body = frontend.response(&plan, out.answers, served_name).to_json(Some(frontend.meta(&plan, mode).to_json()));
//! ```
//! `Frontend` is `Sync` and holds no locks of its own, so any number of CPU workers can share one.

pub mod assemble;
pub mod plan;
pub mod pyjson;
pub mod render;
pub mod template;
pub mod tokenize;

use anyhow::{Context, Result, bail, ensure};
use std::path::Path;

use crate::api::schema::{Answer, QevMeta, SystemOneRequest, SystemOneResponse, Usage};
use crate::contract::MAX_SLOTS;
pub use assemble::{Assembled, Calibration};
pub use plan::{Branch, Limits, Plan, PlanError, PreparedQuestions};

pub struct Frontend {
    pub tokenizer: tokenize::Tokenizer,
    pub template: template::ChatTemplate,
    /// Token id of each slot label, in slot order (`DecisionEngine.slot_ids`): the engine's readout rows.
    pub slot_ids: Vec<u32>,
    /// Slots the tokenizer supports (256 for elephant-1); choices with more options are chunked.
    pub max_slots: usize,
    pub limits: Limits,
    /// Block-bidirectional full attention (serve.patch `noncausal`): enables `_split`'s prefix-block check.
    pub noncausal: bool,
    pub calibration: Calibration,
}

/// The parts of `config.json` (`modeling_overrides`) and `readout.json` the frontend needs, read as
/// `qev.readout.ReadoutConfig.load` reads them.
fn readout_attention(model_dir: &Path) -> Result<bool> {
    let read = |name: &str| -> Result<Option<pyjson::Json>> {
        let p = model_dir.join(name);
        if !p.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        Ok(Some(pyjson::parse(&bytes).with_context(|| format!("parsing {}", p.display()))?))
    };
    let readout = read("readout.json")?;
    let config = read("config.json")?;
    let field = |k: &str| readout.as_ref().and_then(|r| r.get(k)).and_then(|v| v.as_str()).map(str::to_string);
    if let Some(r) = field("readout")
        && r != "slot_lm"
    {
        bail!("readout {r:?} is not supported (slot_lm only)");
    }
    let mut attention = field("attention").unwrap_or_else(|| "causal".into());
    let overrides = config.as_ref().and_then(|c| c.get("modeling_overrides")).filter(|o| !o.is_null());
    if let Some(o) = overrides {
        if let Some(r) = o.get("readout").and_then(|v| v.as_str())
            && r != "slot_lm"
        {
            bail!("readout {r:?} is not supported (slot_lm only)");
        }
        let mask = o.get("attention").and_then(|a| a.get("full_attention")).and_then(|v| v.as_str());
        attention = match mask {
            Some("causal") => "causal".into(),
            Some("block_bidirectional") => "noncausal".into(),
            other => bail!("unsupported full_attention mask {other:?}"),
        };
        if let Some(a) = field("attention") {
            ensure!(a == attention, "readout.json says attention {a:?} but config.json says {mask:?}");
        }
    }
    match attention.as_str() {
        "causal" => Ok(false),
        "noncausal" => Ok(true),
        a => bail!("attention must be causal or noncausal, got {a:?}"),
    }
}

impl Frontend {
    /// Load `tokenizer.json`, `chat_template.jinja`, the readout settings and `calibration.json` (if any) from a
    /// checkpoint directory, and verify the slot labels.
    pub fn load(model_dir: &Path, limits: Limits) -> Result<Self> {
        let tokenizer = tokenize::Tokenizer::load(model_dir)?;
        let template = template::ChatTemplate::load(model_dir)?;
        let slot_ids = tokenizer.verify_slots()?;
        ensure!(slot_ids.len() <= MAX_SLOTS, "{} slots exceed the contract's {MAX_SLOTS}", slot_ids.len());
        Ok(Frontend {
            max_slots: slot_ids.len(),
            slot_ids,
            tokenizer,
            template,
            limits,
            noncausal: readout_attention(model_dir)?,
            calibration: Calibration::load_dir(model_dir)?,
        })
    }

    /// Answers from the slot logits of a planned request (see [`assemble::assemble`]).
    pub fn assemble<'a>(
        &self,
        req: &SystemOneRequest,
        plan: &Plan,
        row: impl Fn(usize) -> &'a [f32],
        masses: Option<&[f64]>,
    ) -> Result<Assembled> {
        assemble::assemble(req, plan, &self.calibration, self.max_slots, row, masses)
    }

    /// The response body qev builds (`model` set to the served name).
    pub fn response(&self, plan: &Plan, answers: Vec<(String, Answer)>, served_model: &str) -> SystemOneResponse {
        SystemOneResponse {
            model: served_model.to_string(),
            answers,
            usage: Usage { input_tokens: plan.total as u64, output_tokens: plan.num_questions as u64 },
        }
    }

    /// The `qev` meta of a 200 response, without `timing`.
    pub fn meta(&self, plan: &Plan, fork_mode: &str) -> QevMeta {
        QevMeta {
            prompt_version: render::PROMPT_VERSION.into(),
            prompt_sha256: plan.prompt_sha256.clone(),
            fork_mode: fork_mode.into(),
            prefix_tokens: plan.prefix_ids.len() as u64,
            branches: plan.branches.len() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frontend_is_shareable_across_worker_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::Frontend>();
        assert_send_sync::<super::Plan>();
    }
}
