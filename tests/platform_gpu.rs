//! The GPU discipline on a real GPU (`engine::platform`): CUDA device 0 of this process is the GPU that
//! `scripts/gpu.sh` selected, so `require_idle_gpu` looks at the right GPU of a multi-GPU host. Needs a GPU; run it
//! through the lock: `scripts/gpu.sh [--gpu N] cargo test --release --test platform_gpu`.
//! Messages never print PCI addresses (test output may be committed as x86 results).

use std::process::Command;
use sysone::engine::platform::{PciAddr, cuda_device_pci, gpu_contention};

fn smi_gpus() -> Vec<(u32, PciAddr)> {
    let out = Command::new("nvidia-smi").args(["--query-gpu=index,pci.bus_id", "--format=csv,noheader"]).output().unwrap();
    assert!(out.status.success(), "nvidia-smi failed");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let (i, b) = l.split_once(',')?;
            Some((i.trim().parse().ok()?, PciAddr::parse(b)?))
        })
        .collect()
}

#[test]
fn cuda_device_0_is_the_selected_gpu() {
    let gpus = smi_gpus();
    assert!(!gpus.is_empty(), "nvidia-smi lists no GPU");
    let mine = cuda_device_pci(0).unwrap();
    let idx = gpus.iter().find(|(_, a)| *a == mine).map(|(i, _)| *i);
    assert!(idx.is_some(), "CUDA device 0 is not among nvidia-smi's GPUs");
    // gpu.sh exports CUDA_DEVICE_ORDER=PCI_BUS_ID, so an index in CUDA_VISIBLE_DEVICES is nvidia-smi's index.
    let order = std::env::var("CUDA_DEVICE_ORDER").unwrap_or_default();
    if let Ok(first) = std::env::var("CUDA_VISIBLE_DEVICES").map(|v| v.split(',').next().unwrap_or("").trim().to_string()) {
        if let (Ok(want), "PCI_BUS_ID") = (first.parse::<u32>(), order.as_str()) {
            assert_eq!(idx, Some(want), "CUDA device 0 is nvidia-smi GPU {idx:?}, but CUDA_VISIBLE_DEVICES selects {want}");
        }
    }
    println!("CUDA device 0 = nvidia-smi GPU {}", idx.unwrap());
}

#[test]
fn contention_query_runs() {
    // Only that it runs and never lists this process; whether other processes use the GPU depends on the host.
    let me = std::process::id().to_string();
    let others = gpu_contention().unwrap();
    assert!(others.iter().all(|l| l.split(',').next().map(str::trim) != Some(me.as_str())));
    println!("other compute processes on this GPU: {}", others.len());
}
