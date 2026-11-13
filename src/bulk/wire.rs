//! The pipe protocol between the bulk coordinator and a worker process (`sysone bulk worker`): length-prefixed
//! frames on the worker's stdin (coordinator → worker) and stdout (worker → coordinator), little-endian.
//!
//! ```text
//! frame   = u32 length of (kind + payload) | u8 kind | payload
//! HELLO   (1, worker → coordinator, once, after the engine is loaded): UTF-8 JSON of `Hello`
//! STEP    (2, coordinator → worker): u64 step id | u8 wants_mass | batch
//! DONE    (3, worker → coordinator): u64 step id | f64 step seconds | u64 bucket tokens (0: unknown) | u32 rows |
//!         per row its answer_slots[i] f32 logits | u8 has_mass | rows x f64 masses (if has_mass)
//! FAILED  (4, worker → coordinator): u64 step id | UTF-8 message
//! batch   = u32 T | T x u32 token ids | u32 S | S x (u32 start, u32 len, i32 parent, u32 pos0) |
//!           u32 Q | Q x u32 answer rows | Q x u16 answer slots
//! ```
//! A worker exits when its stdin closes. Logits travel compactly (only each row's slots), so a step's result is a
//! few hundred KB at most against seconds of GPU time.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

use crate::contract::{DecisionBatch, MAX_SLOTS, Segment, SlotLogits, StepLimits};

pub const HELLO: u8 = 1;
pub const STEP: u8 = 2;
pub const DONE: u8 = 3;
pub const FAILED: u8 = 4;

/// Largest frame accepted (a 1M-token step is 4 MB of ids; anything far beyond is a broken stream).
const MAX_FRAME: usize = 1 << 30;

/// What a worker reports once its engine is ready.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Hello {
    /// e.g. "NVIDIA GB10" or "mock".
    pub device: String,
    pub arch: String,
    pub total_mem_bytes: u64,
    pub buckets: Vec<usize>,
    pub max_answers: usize,
    pub max_segments: usize,
    pub max_prefixes: usize,
    /// `(bucket tokens, tok/s)` for packing decisions (the platform profile's forward points, or a default).
    pub throughput: Vec<(usize, f64)>,
    /// `DecisionEngine::info` pairs.
    pub info: Vec<(String, String)>,
    pub load_seconds: f64,
}

impl Hello {
    pub fn limits(&self) -> StepLimits {
        StepLimits {
            buckets: self.buckets.clone(),
            max_answers: self.max_answers,
            max_segments: self.max_segments,
            max_prefixes: self.max_prefixes,
        }
    }
}

/// A finished step as the worker reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct Done {
    pub id: u64,
    pub step_seconds: f64,
    pub bucket_tokens: Option<usize>,
    pub logits: SlotLogits,
    pub masses: Option<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Frame {
    Hello(Hello),
    Step { id: u64, wants_mass: bool, batch: DecisionBatch },
    /// Logits arrive compact; [`read_frame`] widens them to `MAX_SLOTS` per row using the step's answer slots,
    /// which only the coordinator knows: it is given them through `slots_of`.
    Done(Done),
    Failed { id: u64, msg: String },
}

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn len(&mut self, n: usize) -> Result<()> {
        self.u32(u32::try_from(n).context("length over u32")?);
        Ok(())
    }
}

struct R<'a>(&'a [u8]);

impl R<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        ensure!(self.0.len() >= n, "frame truncated");
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }
    /// A count of items of at least `item` bytes each, checked against what is left.
    fn count(&mut self, item: usize) -> Result<usize> {
        let n = self.u32()? as usize;
        ensure!(n.saturating_mul(item) <= self.0.len(), "count {n} over the frame");
        Ok(n)
    }
}

fn put_batch(w: &mut W, b: &DecisionBatch) -> Result<()> {
    w.len(b.token_ids.len())?;
    for &t in &b.token_ids {
        w.u32(t);
    }
    w.len(b.segments.len())?;
    for s in &b.segments {
        w.u32(s.start);
        w.u32(s.len);
        w.i32(s.parent);
        w.u32(s.pos0);
    }
    w.len(b.answer_rows.len())?;
    for &r in &b.answer_rows {
        w.u32(r);
    }
    for &s in &b.answer_slots {
        w.u16(s);
    }
    Ok(())
}

fn get_batch(r: &mut R) -> Result<DecisionBatch> {
    let n = r.count(4)?;
    let token_ids = (0..n).map(|_| r.u32()).collect::<Result<_>>()?;
    let n = r.count(16)?;
    let segments = (0..n)
        .map(|_| Ok(Segment { start: r.u32()?, len: r.u32()?, parent: r.i32()?, pos0: r.u32()? }))
        .collect::<Result<_>>()?;
    let n = r.count(6)?;
    let answer_rows = (0..n).map(|_| r.u32()).collect::<Result<_>>()?;
    let answer_slots = (0..n).map(|_| r.u16()).collect::<Result<_>>()?;
    Ok(DecisionBatch { token_ids, segments, answer_rows, answer_slots })
}

/// Encode a frame. For `Done`, `slots` gives each row's slot count (the compact logits).
pub fn encode(f: &Frame, slots: &[u16]) -> Result<Vec<u8>> {
    let mut w = W(vec![0; 4]);
    match f {
        Frame::Hello(h) => {
            w.u8(HELLO);
            w.0.extend_from_slice(&serde_json::to_vec(h)?);
        }
        Frame::Step { id, wants_mass, batch } => {
            w.u8(STEP);
            w.u64(*id);
            w.u8(*wants_mass as u8);
            put_batch(&mut w, batch)?;
        }
        Frame::Done(d) => {
            w.u8(DONE);
            w.u64(d.id);
            w.f64(d.step_seconds);
            w.u64(d.bucket_tokens.unwrap_or(0) as u64);
            ensure!(slots.len() == d.logits.rows, "{} slot counts for {} rows", slots.len(), d.logits.rows);
            w.len(d.logits.rows)?;
            for (r, &k) in slots.iter().enumerate() {
                for &x in &d.logits.data[r * MAX_SLOTS..r * MAX_SLOTS + k as usize] {
                    w.f32(x);
                }
            }
            match &d.masses {
                Some(m) => {
                    ensure!(m.len() == d.logits.rows, "{} masses for {} rows", m.len(), d.logits.rows);
                    w.u8(1);
                    for &x in m {
                        w.f64(x);
                    }
                }
                None => w.u8(0),
            }
        }
        Frame::Failed { id, msg } => {
            w.u8(FAILED);
            w.u64(*id);
            w.0.extend_from_slice(msg.as_bytes());
        }
    }
    let n = u32::try_from(w.0.len() - 4).context("frame over 4 GiB")?;
    w.0[..4].copy_from_slice(&n.to_le_bytes());
    Ok(w.0)
}

/// Decode one frame body (kind + payload). `slots_of(step id)` returns the answer slots of a step this side sent
/// (needed to widen a `Done`'s compact logits).
pub fn decode(body: &[u8], slots_of: &dyn Fn(u64) -> Option<Vec<u16>>) -> Result<Frame> {
    let mut r = R(body);
    let kind = r.u8()?;
    let f = match kind {
        HELLO => Frame::Hello(serde_json::from_slice(r.0).context("HELLO payload")?),
        STEP => {
            let id = r.u64()?;
            let wants_mass = r.u8()? != 0;
            let batch = get_batch(&mut r)?;
            ensure!(r.0.is_empty(), "trailing bytes in STEP");
            Frame::Step { id, wants_mass, batch }
        }
        DONE => {
            let id = r.u64()?;
            let step_seconds = r.f64()?;
            let bucket = r.u64()? as usize;
            let rows = r.u32()? as usize;
            let slots = slots_of(id).with_context(|| format!("DONE for unknown step {id}"))?;
            ensure!(slots.len() == rows, "DONE step {id}: {rows} rows, the step has {}", slots.len());
            let mut data = vec![0f32; rows * MAX_SLOTS];
            for (row, &k) in slots.iter().enumerate() {
                for x in &mut data[row * MAX_SLOTS..row * MAX_SLOTS + k as usize] {
                    *x = r.f32()?;
                }
            }
            let masses = match r.u8()? {
                0 => None,
                _ => Some((0..rows).map(|_| r.f64()).collect::<Result<Vec<_>>>()?),
            };
            ensure!(r.0.is_empty(), "trailing bytes in DONE");
            Frame::Done(Done {
                id,
                step_seconds,
                bucket_tokens: (bucket > 0).then_some(bucket),
                logits: SlotLogits { rows, data },
                masses,
            })
        }
        FAILED => {
            let id = r.u64()?;
            Frame::Failed { id, msg: String::from_utf8_lossy(r.0).into_owned() }
        }
        k => bail!("unknown frame kind {k}"),
    };
    Ok(f)
}

/// Read one frame body; Ok(None) at a clean end of stream (before a frame starts).
pub fn read_body(r: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let n = u32::from_le_bytes(len) as usize;
    ensure!(n > 0 && n <= MAX_FRAME, "bad frame length {n}");
    let mut body = vec![0u8; n];
    r.read_exact(&mut body).context("stream ended inside a frame")?;
    Ok(Some(body))
}

pub fn write_frame(w: &mut impl Write, bytes: &[u8]) -> Result<()> {
    w.write_all(bytes)?;
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch() -> DecisionBatch {
        let mut b = DecisionBatch::default();
        b.push_request(&[1, 2, 3], &[(&[10, 11], 3), (&[12], 2)]).unwrap();
        b.push_request(&[4, 5], &[(&[20, 21, 22], 5)]).unwrap();
        b
    }

    fn round(f: &Frame, slots: &[u16]) -> Frame {
        let bytes = encode(f, slots).unwrap();
        let body = read_body(&mut &bytes[..]).unwrap().unwrap();
        let s = slots.to_vec();
        decode(&body, &move |_| Some(s.clone())).unwrap()
    }

    #[test]
    fn frames_round_trip() {
        let b = batch();
        let step = Frame::Step { id: 42, wants_mass: true, batch: b.clone() };
        assert_eq!(round(&step, &[]), step);
        let hello = Frame::Hello(Hello {
            device: "mock".into(),
            arch: "none".into(),
            total_mem_bytes: 1,
            buckets: vec![256, 512],
            max_answers: 9,
            max_segments: 10,
            max_prefixes: 3,
            throughput: vec![(256, 1.5)],
            info: vec![("engine".into(), "mock".into())],
            load_seconds: 0.25,
        });
        assert_eq!(round(&hello, &[]), hello);
        // Logits: only each row's slots travel; the rest of the row comes back as zeros.
        let mut data = vec![0f32; 3 * MAX_SLOTS];
        for (r, &k) in b.answer_slots.iter().enumerate() {
            for j in 0..k as usize {
                data[r * MAX_SLOTS + j] = (r * 100 + j) as f32 + 0.5;
            }
        }
        let done = Frame::Done(Done {
            id: 7,
            step_seconds: 1.25,
            bucket_tokens: Some(512),
            logits: SlotLogits { rows: 3, data },
            masses: Some(vec![0.5, 0.25, 0.125]),
        });
        assert_eq!(round(&done, &b.answer_slots), done);
        let failed = Frame::Failed { id: 9, msg: "device lost".into() };
        assert_eq!(round(&failed, &[]), failed);
        // A clean end of stream, and a truncated frame.
        assert!(read_body(&mut &[][..]).unwrap().is_none());
        let bytes = encode(&step, &[]).unwrap();
        assert!(read_body(&mut &bytes[..bytes.len() - 1]).is_err());
        assert!(decode(&bytes[4..bytes.len() - 2], &|_| None).is_err());
    }
}
