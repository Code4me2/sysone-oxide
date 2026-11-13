//! Embeds the checked-in cuda-oxide PTX of `kernels-simt/` (docs/G1.md, "Build").
//!
//! The SIMT kernels build with a different toolchain (1.99.1 + cargo-oxide) than sysone (1.98.0), so sysone does not
//! compile them. `kernels-simt/build.sh <arch>` writes `kernels-simt/artifacts/gdn_simt.<arch>.ptx` and a manifest
//! with the hash of the kernel sources and of the PTX. This script embeds each artifact whose manifest matches both
//! the current sources and the PTX file; any other artifact is embedded as an error string, so selecting the SIMT
//! backend with a stale or missing artifact fails loudly at run time instead of running old kernels.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn rs_files(dir: &Path, rel: &str, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let rel = format!("{rel}/{name}");
        let path = e.path();
        if path.is_dir() {
            rs_files(&path, &rel, out);
        } else if name.ends_with(".rs") {
            out.push(rel);
        }
    }
}

/// Same algorithm as `source_hash` in kernels-simt/build.sh.
fn source_hash(root: &Path) -> Option<String> {
    let mut files = vec!["Cargo.toml".to_string(), "Cargo.lock".to_string(), "rust-toolchain.toml".to_string()];
    rs_files(&root.join("src"), "src", &mut files);
    files.sort();
    let mut listing = String::new();
    for f in &files {
        let bytes = std::fs::read(root.join(f)).ok()?;
        let _ = write!(listing, "{f}\n{}\n", sha256(&bytes));
    }
    Some(sha256(listing.as_bytes()))
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.join("kernels-simt");
    for p in ["src", "artifacts", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        println!("cargo:rerun-if-changed={}", root.join(p).display());
    }
    let current = source_hash(&root);
    let mut entries = Vec::new();
    if let Ok(dir) = std::fs::read_dir(root.join("artifacts")) {
        let mut names: Vec<String> = dir.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        for name in names {
            let Some(arch) = name.strip_prefix("gdn_simt.").and_then(|n| n.strip_suffix(".ptx")) else { continue };
            let ptx_path = root.join("artifacts").join(&name);
            let manifest = std::fs::read_to_string(root.join("artifacts").join(format!("gdn_simt.{arch}.manifest")))
                .unwrap_or_default();
            let field = |k: &str| {
                manifest.lines().find_map(|l| l.strip_prefix(k).and_then(|v| v.strip_prefix(' '))).map(str::to_string)
            };
            let ptx = std::fs::read(&ptx_path).unwrap_or_default();
            let status = if manifest.is_empty() {
                Err(format!("{name} has no manifest"))
            } else if field("ptx_sha256").as_deref() != Some(&sha256(&ptx)) {
                Err(format!("{name} does not match the PTX hash in its manifest"))
            } else if field("source_sha256") != current {
                Err(format!("{name} is stale: kernels-simt sources changed since it was built"))
            } else {
                Ok(())
            };
            match status {
                Ok(()) => entries.push(format!("    ({arch:?}, Ok(include_str!({:?}))),", ptx_path.display().to_string())),
                Err(e) => {
                    println!("cargo:warning=SIMT GDN kernels: {e}; rebuild with `kernels-simt/build.sh {arch}`");
                    entries.push(format!("    ({arch:?}, Err({e:?})),"));
                }
            }
        }
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("simt_artifacts.rs");
    let src = format!(
        "/// `(arch, PTX or the reason it is unusable)` for every artifact under kernels-simt/artifacts.\n\
         pub const ARTIFACTS: &[(&str, Result<&str, &str>)] = &[\n{}\n];\n",
        entries.join("\n")
    );
    std::fs::write(out, src).unwrap();
}
