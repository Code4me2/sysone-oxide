//! `cpu_ref_check`: run the CPU reference over the requests recorded by `reference/hf_check/run_qev.py`, packed
//! into one DecisionBatch and each alone, and compare slot logits / probabilities with qev's fp32 results.
//!
//! ```text
//! cargo run --release --bin cpu_ref_check -- \
//!     --model /home/velvet/Desktop/hf_models/elephant-1 --recorded reference/hf_check/out/qev_fp32.json
//! ```

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use sysone::engine::cpu_ref::{CpuRefEngine, check};

#[derive(Parser, Debug)]
#[command(about = "Compare the CPU f32 reference forward with qev's recorded fp32 slot logits")]
struct Args {
    /// bf16 checkpoint directory (config.json, model.safetensors, tokenizer.json).
    #[arg(long, default_value = "/home/velvet/Desktop/hf_models/elephant-1")]
    model: PathBuf,
    /// JSON written by reference/hf_check/run_qev.py.
    #[arg(long, default_value = "reference/hf_check/out/qev_fp32.json")]
    recorded: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let mut engine = CpuRefEngine::load(&args.model)?;
    println!("loaded {} in {:.1}s ({} threads)", args.model.display(), t0.elapsed().as_secs_f64(),
             rayon::current_num_threads());
    let rec = check::Recorded::load(&args.recorded)?;
    println!("reference: {}", rec.provenance);
    let report = check::compare(&mut engine, &rec, true)?;
    report.print();
    Ok(())
}
