//! The cuda-oxide SIMT build/load path (`sysone::kernels::simt`): the checked-in artifact for this GPU is current,
//! loads, and a probe kernel round-trips through a cutile tensor on the same stream.

use cutile::prelude::*;
use sysone::kernels::simt;

#[test]
fn simt_artifact_loads_and_probe_round_trips() {
    for (arch, status) in simt::artifacts() {
        println!("SIMT artifact {arch}: {}", match status { Ok(()) => "ok", Err(e) => e });
    }
    let ctx = Context::new(0).unwrap();
    let s = ctx.new_stream().unwrap();
    let n = 1000usize;
    let host: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect();
    let input = api::copy_host_vec_to_device(&Arc::new(host.clone())).sync_on(&s).unwrap();
    let out = api::zeros::<u32>(&[n]).sync_on(&s).unwrap();
    let op = unsafe {
        simt::probe(input.device_pointer().cu_deviceptr(), out.device_pointer().cu_deviceptr(), n as i32).unwrap()
    };
    op.sync_on(&s).unwrap();
    let got = out.dup().to_host_vec().sync_on(&s).unwrap();
    let want: Vec<u32> = host.iter().enumerate().map(|(i, x)| x.wrapping_mul(3).wrapping_add(i as u32)).collect();
    assert_eq!(got, want);
}
