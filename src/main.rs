use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use sysone::engine::config::ModelConfig;

#[derive(Parser, Debug)]
#[command(version, about = "System One decision-model server")]
struct Args {
    /// Model directory (config.json, model.safetensors, tokenizer.json, chat_template.jinja).
    #[arg(long)]
    model: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = ModelConfig::load(&args.model)?;
    let t = &cfg.text_config;
    println!(
        "{}: {} layers, hidden {}, vocab {}, readout {} — serving is not implemented yet (DESIGN.md §11)",
        args.model.display(), t.num_hidden_layers, t.hidden_size, t.vocab_size, cfg.modeling_overrides.readout
    );
    Ok(())
}
