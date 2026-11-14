//! The results file of `sysone bulk`: append-only JSONL, one complete line per finished job, safe to resume.
//!
//! - **Exclusive.** The file is locked (`flock`, non-blocking) while a run writes it, so two runs never append to
//!   the same file.
//! - **Crash-safe.** Every result is one `write` of a whole line ending in `\n`. A crash can leave at most a
//!   partial last line (no newline); [`Output::open`] cuts it off before appending. A complete line that does not
//!   parse is never cut: that is corruption from outside, and the run refuses to start.
//! - **Resumable.** [`Output::open`] returns the ids already present ([`IdKey`]s), and the run skips those jobs.
//! - **Durable.** [`Output::write`] appends; [`Output::sync`] calls `fdatasync`, done at least every
//!   [`SYNC_EVERY`] and at the end.

use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::job::{IdKey, id_key, result_id};

pub const SYNC_EVERY: Duration = Duration::from_secs(2);

mod sys {
    use std::os::raw::c_int;
    pub const LOCK_EX: c_int = 2;
    pub const LOCK_NB: c_int = 4;
    unsafe extern "C" {
        pub fn flock(fd: c_int, op: c_int) -> c_int;
        pub fn fdatasync(fd: c_int) -> c_int;
    }
}

pub struct Output {
    file: File,
    path: PathBuf,
    last_sync: Instant,
    dirty: bool,
}

/// What [`Output::open`] found in an existing file.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Resume {
    /// Complete result lines already in the file.
    pub lines: u64,
    /// Bytes of a partial last line that were cut off.
    pub truncated_bytes: u64,
}

impl Output {
    /// Open (or create) `path` for appending, lock it, cut a partial last line, and return the ids it holds.
    pub fn open(path: &Path) -> Result<(Self, HashSet<IdKey>, Resume)> {
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: flock on a file descriptor this function owns.
        if unsafe { sys::flock(file.as_raw_fd(), sys::LOCK_EX | sys::LOCK_NB) } != 0 {
            bail!("{} is locked by another process (another bulk run writing it?)", path.display());
        }
        let mut done = HashSet::new();
        let mut resume = Resume::default();
        let mut good_end = 0u64;
        {
            let mut r = BufReader::with_capacity(1 << 20, &file);
            r.seek(SeekFrom::Start(0))?;
            let mut buf = Vec::new();
            loop {
                buf.clear();
                let n = r.read_until(b'\n', &mut buf)?;
                if n == 0 {
                    break;
                }
                if buf.last() != Some(&b'\n') {
                    // A partial last line: the previous run stopped in the middle of writing it.
                    resume.truncated_bytes = n as u64;
                    break;
                }
                let Some(id) = result_id(&buf[..n - 1]) else {
                    bail!(
                        "{} line {}: a complete line that is not a result ({} bytes); refusing to append to a corrupt file",
                        path.display(),
                        resume.lines + 1,
                        n
                    );
                };
                done.insert(id_key(&id));
                resume.lines += 1;
                good_end += n as u64;
            }
        }
        if resume.truncated_bytes > 0 {
            file.set_len(good_end).with_context(|| format!("cutting the partial last line of {}", path.display()))?;
        }
        Ok((Output { file, path: path.to_path_buf(), last_sync: Instant::now(), dirty: false }, done, resume))
    }

    /// Append whole lines (each ending in `\n`) in one write.
    pub fn write(&mut self, lines: &[u8]) -> Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        debug_assert!(lines.ends_with(b"\n"));
        self.file.write_all(lines).with_context(|| format!("writing {}", self.path.display()))?;
        self.dirty = true;
        if self.last_sync.elapsed() >= SYNC_EVERY {
            self.sync()?;
        }
        Ok(())
    }

    /// Flush appended lines to the device.
    pub fn sync(&mut self) -> Result<()> {
        if self.dirty {
            // SAFETY: fdatasync on an open descriptor.
            if unsafe { sys::fdatasync(self.file.as_raw_fd()) } != 0 {
                return Err(std::io::Error::last_os_error()).with_context(|| format!("syncing {}", self.path.display()));
            }
            self.dirty = false;
        }
        self.last_sync = Instant::now();
        Ok(())
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        let _ = self.sync();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bulk::job::result_line;
    use crate::frontend::pyjson::Json;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sysone-bulk-out-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&d);
        d
    }

    #[test]
    fn resume_cuts_a_partial_line_and_reads_done_ids() {
        let p = tmp("partial");
        let ok = Json::Object(vec![]);
        let mut text = result_line(&Json::str("a"), 200, &ok).unwrap();
        text.extend(result_line(&Json::Int("2".into()), 422, &ok).unwrap());
        let whole = text.len();
        text.extend_from_slice(b"{\"id\":\"c\",\"sta");
        std::fs::write(&p, &text).unwrap();
        {
            let (mut out, done, resume) = Output::open(&p).unwrap();
            assert_eq!(resume, Resume { lines: 2, truncated_bytes: 14 });
            assert!(done.contains(&id_key(&Json::str("a"))) && done.contains(&id_key(&Json::Int("2".into()))));
            assert!(!done.contains(&id_key(&Json::str("c"))));
            assert_eq!(std::fs::metadata(&p).unwrap().len() as usize, whole);
            // A second opener is refused while this one holds the file.
            assert!(Output::open(&p).err().unwrap().to_string().contains("locked"));
            out.write(&result_line(&Json::str("c"), 200, &ok).unwrap()).unwrap();
        }
        let (_, done, resume) = Output::open(&p).unwrap();
        assert_eq!((resume.lines, resume.truncated_bytes, done.len()), (3, 0, 3));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn a_corrupt_complete_line_is_refused() {
        let p = tmp("corrupt");
        std::fs::write(&p, b"{\"id\":1,\"status\":200,\"body\":{}}\nnot json\n").unwrap();
        let e = Output::open(&p).err().unwrap().to_string();
        assert!(e.contains("line 2"), "{e}");
        assert_eq!(std::fs::metadata(&p).unwrap().len(), 41);
        std::fs::remove_file(&p).unwrap();
    }
}
