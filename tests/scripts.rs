//! The shell scripts' own tests (no GPU, no checkpoint): `scripts/gpu.sh`'s GPU selection and per-device locks
//! against a fake nvidia-smi, and `scripts/x86/check_outgoing.sh` on throw-away git repositories. Needs bash, flock
//! and git.

use std::path::PathBuf;
use std::process::Command;

fn run(script: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(script);
    let out = Command::new("bash").arg(&path).output().unwrap_or_else(|e| panic!("running {script}: {e}"));
    let text = String::from_utf8_lossy(&out.stdout);
    println!("{text}{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{script} failed:\n{}", text.lines().filter(|l| l.starts_with("FAIL")).collect::<Vec<_>>().join("\n"));
}

#[test]
fn gpu_sh_selection_and_locks() {
    run("scripts/tests/test_gpu_sh.sh");
}

#[test]
fn check_outgoing_flags_leaks() {
    run("scripts/tests/test_check_outgoing.sh");
}
