//! cuda-oxide SIMT kernels for sysone's GatedDeltaNet mixer (docs/DESIGN.md §6.4, docs/G1.md).
//!
//! This crate is device code only. `build.sh` compiles it with `cargo oxide` to PTX for one architecture and
//! writes `artifacts/gdn_simt.<arch>.ptx` plus a manifest (source hash, PTX hash); sysone embeds the checked-in
//! PTX and loads it with cuda-core at run time (`sysone::kernels::simt`). Every kernel takes raw device pointers
//! and i32 scalars, so the host side launches it with an untyped argument packet: keep the parameter lists here
//! and in `src/kernels/gdn/simt.rs` in sync.

#![allow(clippy::too_many_arguments)]

use cuda_device::{cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Integration probe: `out[i] = in[i] * 3 + i` for `i < n` (grid-stride free; one element per thread).
    #[kernel]
    pub unsafe fn simt_probe(input: *const u32, out: *mut u32, n: i32) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if i < n {
            unsafe {
                *out.add(i as usize) = (*input.add(i as usize)).wrapping_mul(3).wrapping_add(i as u32);
            }
        }
    }
}

fn main() {}
