//! Pinning the GPU worker, the batcher and the CPU pool to performance cores (docs/PLATFORM.md P7).
//!
//! GB10 has 10 Cortex-X925 and 10 Cortex-A725 cores. The kernel describes them in sysfs: `cpu_capacity`
//! (X925 997–1024, A725 718–731) and `cpufreq/cpuinfo_max_freq` (3.9 vs 2.8 GHz). A core counts as a performance
//! core when its value is at least [`PERFORMANCE_SHARE`] of the largest one. Detection gives up (and nothing is
//! pinned) when neither file exists for every core, or when all cores look alike, as on most x86 hosts. Pinning is
//! opt-in (`--pin-cores`); a failure to pin is reported once and otherwise ignored.

use std::path::Path;

/// A core is a performance core when its capacity (or maximum frequency) is at least this share of the largest.
pub const PERFORMANCE_SHARE: f64 = 0.85;

/// The performance cores this process may run on, ascending, or None when there is no performance/efficiency
/// split to exploit (or it cannot be read).
pub fn performance_cores() -> Option<Vec<usize>> {
    let cores = performance_cores_in(Path::new("/sys/devices/system/cpu"))?;
    match allowed_cores() {
        Some(allowed) => {
            let both: Vec<usize> = cores.into_iter().filter(|c| allowed.contains(c)).collect();
            (!both.is_empty()).then_some(both)
        }
        None => Some(cores),
    }
}

/// [`performance_cores`] over a sysfs-like directory (`cpuN/cpu_capacity`, `cpuN/cpufreq/cpuinfo_max_freq`),
/// without the affinity check.
pub fn performance_cores_in(root: &Path) -> Option<Vec<usize>> {
    let mut cpus: Vec<usize> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok()?.file_name().to_str()?.strip_prefix("cpu")?.parse().ok())
        .collect();
    cpus.sort_unstable();
    if cpus.is_empty() {
        return None;
    }
    let read = |cpu: usize, file: &str| -> Option<u64> {
        std::fs::read_to_string(root.join(format!("cpu{cpu}")).join(file)).ok()?.trim().parse().ok()
    };
    for file in ["cpu_capacity", "cpufreq/cpuinfo_max_freq"] {
        let Some(values) = cpus.iter().map(|&c| read(c, file)).collect::<Option<Vec<u64>>>() else { continue };
        let max = *values.iter().max()?;
        if max == 0 {
            continue;
        }
        let fast: Vec<usize> =
            cpus.iter().zip(&values).filter(|&(_, &v)| v as f64 >= PERFORMANCE_SHARE * max as f64).map(|(&c, _)| c).collect();
        // All cores alike: nothing to gain, and pinning would only take the scheduler's freedom away.
        return (fast.len() < cpus.len()).then_some(fast);
    }
    None
}

/// Size of glibc's `cpu_set_t` in bytes (1024 CPUs).
#[cfg(target_os = "linux")]
const CPU_SET_BYTES: usize = 128;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u8) -> i32;
}

/// The cores the calling thread may run on (None where unknown).
pub fn allowed_cores() -> Option<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        let mut mask = [0u8; CPU_SET_BYTES];
        // SAFETY: the mask is a valid, writable cpu_set_t-sized buffer.
        if unsafe { sched_getaffinity(0, CPU_SET_BYTES, mask.as_mut_ptr()) } != 0 {
            return None;
        }
        Some((0..CPU_SET_BYTES * 8).filter(|&c| mask[c / 8] & (1 << (c % 8)) != 0).collect())
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// Restrict the calling thread to `cores`.
pub fn pin_current_thread(cores: &[usize]) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let mut mask = [0u8; CPU_SET_BYTES];
        for &c in cores {
            if c >= CPU_SET_BYTES * 8 {
                return Err(std::io::Error::other(format!("core {c} is beyond the {}-core mask", CPU_SET_BYTES * 8)));
            }
            mask[c / 8] |= 1 << (c % 8);
        }
        // SAFETY: the mask is a valid cpu_set_t-sized buffer; pid 0 is the calling thread.
        if unsafe { sched_setaffinity(0, CPU_SET_BYTES, mask.as_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cores;
        Err(std::io::Error::other("thread pinning is only implemented on Linux"))
    }
}

/// Pin the calling thread to `cores` if given; a failure is logged and otherwise harmless.
pub fn maybe_pin(cores: Option<&[usize]>, what: &str) {
    if let Some(cores) = cores
        && let Err(e) = pin_current_thread(cores)
    {
        eprintln!("sysone: could not pin {what} to cores {cores:?}: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_sysfs(name: &str, cpus: &[(Option<u64>, Option<u64>)]) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("sysone-affinity-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (i, (cap, freq)) in cpus.iter().enumerate() {
            let d = root.join(format!("cpu{i}"));
            std::fs::create_dir_all(d.join("cpufreq")).unwrap();
            if let Some(c) = cap {
                std::fs::write(d.join("cpu_capacity"), format!("{c}\n")).unwrap();
            }
            if let Some(f) = freq {
                std::fs::write(d.join("cpufreq/cpuinfo_max_freq"), format!("{f}\n")).unwrap();
            }
        }
        // Entries that are not cores.
        std::fs::create_dir_all(root.join("cpufreq")).unwrap();
        std::fs::write(root.join("online"), "0-19\n").unwrap();
        root
    }

    #[test]
    fn detects_gb10_performance_cores() {
        // The GB10's sysfs as read on this machine.
        let caps = [718, 718, 718, 718, 718, 997, 997, 997, 997, 997, 731, 731, 731, 731, 731, 1017, 1017, 1017, 1017, 1024];
        let cpus: Vec<_> = caps.iter().map(|&c| (Some(c), Some(if c > 900 { 3_900_000 } else { 2_808_000 }))).collect();
        let root = fake_sysfs("gb10", &cpus);
        assert_eq!(performance_cores_in(&root), Some(vec![5, 6, 7, 8, 9, 15, 16, 17, 18, 19]));
        // Frequency alone gives the same split.
        let cpus: Vec<_> = cpus.iter().map(|&(_, f)| (None, f)).collect();
        let _ = std::fs::remove_dir_all(&root);
        let root = fake_sysfs("gb10-freq", &cpus);
        assert_eq!(performance_cores_in(&root), Some(vec![5, 6, 7, 8, 9, 15, 16, 17, 18, 19]));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn homogeneous_or_unreadable_means_no_pinning() {
        // One core without data: that source is skipped.
        for (name, cpus) in [
            ("x86", vec![(None, Some(3_000_000)); 8]),
            ("none", vec![(None, None); 4]),
            ("partial", vec![(Some(1024), None), (None, None)]),
        ] {
            let root = fake_sysfs(name, &cpus);
            assert_eq!(performance_cores_in(&root), None, "{name}");
            let _ = std::fs::remove_dir_all(&root);
        }
        assert_eq!(performance_cores_in(Path::new("/nonexistent/sysone")), None);
    }

    #[test]
    fn pins_and_restores_the_current_thread() {
        let Some(before) = allowed_cores() else { return };
        std::thread::spawn(move || {
            let one = [before[0]];
            pin_current_thread(&one).unwrap();
            assert_eq!(allowed_cores().unwrap(), one);
            pin_current_thread(&before).unwrap();
            assert_eq!(allowed_cores().unwrap(), before);
        })
        .join()
        .unwrap();
        #[cfg(target_os = "linux")]
        assert!(pin_current_thread(&[CPU_SET_BYTES * 8]).is_err());
    }
}
