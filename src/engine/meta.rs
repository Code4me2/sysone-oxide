//! Per-step batch metadata on the device (DESIGN §6.1, §6.2, §7; PLATFORM.md P3).
//!
//! A captured T-bucket graph reads everything batch-specific from fixed device buffers, so one graph serves
//! any [`DecisionBatch`] that fits its bucket. [`MetaUploader`] packs a batch into a persistent, pinned,
//! pre-touched host slot and copies it into [`DeviceMeta`] with one async H2D per array. Two host slots
//! alternate (double buffering), so the CPU can pack batch n+1 while batch n's copy and forward are still
//! in flight; a slot is reused only after the event recorded behind its last copy has completed.
//!
//! Layout for a bucket of `tokens` tokens, `segments` segment slots and `answers` answer slots:
//! - `header [8] u32`: `[num_tokens T, num_segments S (real), num_answers Q, num_prefixes, bucket_tokens,
//!   pad_segment index (S, or u32::MAX when T == bucket_tokens), 0, 0]`;
//! - `token_ids [tokens] u32` (0 for padding);
//! - `positions [tokens] u32`: RoPE position (`pos0 + offset in segment`; padding counts from 0);
//! - `token_seg [tokens] i32`: segment index of each token (padding tokens point at the pad segment);
//! - `segments [segments] × {start, len, parent, pos0}` (`Segment`, 16 bytes; the pad segment has parent
//!   [`PAD_PARENT`]);
//! - `seg_state [segments] i32`: prefix-state scratch slot of a segment's prefix (its own ordinal for a prefix,
//!   the parent's for a suffix, -1 for padding);
//! - `answer_rows [answers] u32` (padding repeats row 0), `answer_slots [answers] u32` (0 for padding).

use crate::contract::{DecisionBatch, Segment};
use crate::engine::graph::tensor::{DevBuf, DevTensor, GraphDType};
use anyhow::{Result, ensure};
use cuda_core::{Event, PinnedHostBuffer, Stream};
use std::sync::Arc;

/// `parent` of the dummy segment that holds a bucket's padding tokens. Kernels skip it.
pub const PAD_PARENT: i32 = -2;
pub const HEADER_LEN: usize = 8;

/// Capacities of one bucket's metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MetaShape {
    pub tokens: usize,
    pub segments: usize,
    pub answers: usize,
}

impl MetaShape {
    fn layout(&self) -> [(usize, usize); 8] {
        // (offset, bytes) of each array in one contiguous block, 256-byte aligned.
        let sizes = [
            HEADER_LEN * 4,
            self.tokens * 4,
            self.tokens * 4,
            self.tokens * 4,
            self.segments * 16,
            self.segments * 4,
            self.answers * 4,
            self.answers * 4,
        ];
        let mut out = [(0, 0); 8];
        let mut off = 0;
        for (i, s) in sizes.iter().enumerate() {
            out[i] = (off, *s);
            off += s.div_ceil(256) * 256;
        }
        out
    }

    pub fn total_bytes(&self) -> usize {
        let l = self.layout();
        l[7].0 + l[7].1
    }
}

/// Device-side metadata arrays (views into one allocation). Pointers are stable for the engine's lifetime,
/// so graphs captured against them stay valid.
#[derive(Clone, Debug)]
pub struct DeviceMeta {
    pub shape: MetaShape,
    pub header: DevTensor,
    pub token_ids: DevTensor,
    pub positions: DevTensor,
    pub token_seg: DevTensor,
    /// `[segments, 4]` i32: start, len, parent, pos0.
    pub segments: DevTensor,
    pub seg_state: DevTensor,
    pub answer_rows: DevTensor,
    pub answer_slots: DevTensor,
}

impl DeviceMeta {
    pub fn alloc(stream: &Arc<Stream>, shape: MetaShape) -> Result<Self> {
        let buf = DevBuf::zeroed(stream, shape.total_bytes())?;
        let l = shape.layout();
        let v = |i: usize, dt: GraphDType, sh: Vec<usize>| DevTensor::new(buf.clone(), l[i].0, dt, sh);
        Ok(Self {
            shape,
            header: v(0, GraphDType::U32, vec![HEADER_LEN])?,
            token_ids: v(1, GraphDType::U32, vec![shape.tokens])?,
            positions: v(2, GraphDType::U32, vec![shape.tokens])?,
            token_seg: v(3, GraphDType::I32, vec![shape.tokens])?,
            segments: v(4, GraphDType::I32, vec![shape.segments, 4])?,
            seg_state: v(5, GraphDType::I32, vec![shape.segments])?,
            answer_rows: v(6, GraphDType::U32, vec![shape.answers])?,
            answer_slots: v(7, GraphDType::U32, vec![shape.answers])?,
        })
    }

    pub fn base_ptr(&self) -> u64 {
        self.header.ptr()
    }
}

/// Host-side image of one packed batch, in exactly the device layout.
pub struct HostSlot {
    bytes: PinnedHostBuffer<u8>,
    done: Event,
    in_flight: bool,
    /// Counts of the batch currently packed here.
    pub packed: Option<PackedCounts>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedCounts {
    pub tokens: usize,
    pub segments: usize,
    pub answers: usize,
    pub prefixes: usize,
    /// Bucket size the batch was padded to.
    pub bucket_tokens: usize,
}

/// Double-buffered metadata uploader for one device metadata block.
pub struct MetaUploader {
    pub dev: DeviceMeta,
    slots: [HostSlot; 2],
    next: usize,
}

impl MetaUploader {
    pub fn new(stream: &Arc<Stream>, shape: MetaShape) -> Result<Self> {
        let dev = DeviceMeta::alloc(stream, shape)?;
        let ctx = stream.context();
        // `zeroed` writes every byte, so all pages are touched and resident before the first batch (P3).
        let mk = || -> Result<HostSlot> {
            Ok(HostSlot {
                bytes: PinnedHostBuffer::zeroed(ctx, shape.total_bytes())?,
                done: ctx.new_event()?,
                in_flight: false,
                packed: None,
            })
        };
        Ok(Self { dev, slots: [mk()?, mk()?], next: 0 })
    }

    /// Pack `batch` (already validated) for a bucket of `bucket_tokens` tokens into the next free host
    /// slot, blocking only if that slot's previous upload has not completed. Returns the slot index.
    pub fn pack(&mut self, batch: &DecisionBatch, bucket_tokens: usize) -> Result<usize> {
        let shape = self.dev.shape;
        let t = batch.num_tokens();
        let q = batch.num_branches();
        let s = batch.segments.len();
        let pad = bucket_tokens - t.min(bucket_tokens);
        ensure!(t <= bucket_tokens && bucket_tokens <= shape.tokens, "batch of {t} tokens, bucket {bucket_tokens}, \
                capacity {}", shape.tokens);
        ensure!(s + (pad > 0) as usize <= shape.segments, "{s} segments (+pad) over capacity {}", shape.segments);
        ensure!(q <= shape.answers, "{q} answers over capacity {}", shape.answers);
        let i = self.next;
        self.next ^= 1;
        let slot = &mut self.slots[i];
        if slot.in_flight {
            slot.done.synchronize()?;
            slot.in_flight = false;
        }
        let l = shape.layout();
        let buf = slot.bytes.as_mut_slice();
        let mut put_u32 = |arr: usize, idx: usize, v: u32| {
            let o = l[arr].0 + idx * 4;
            buf[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        // Per-token arrays.
        let mut prefixes = 0u32;
        let mut ordinal = vec![-1i32; s];
        for (si, seg) in batch.segments.iter().enumerate() {
            let state = if seg.is_prefix() {
                ordinal[si] = prefixes as i32;
                prefixes += 1;
                ordinal[si]
            } else {
                ordinal[seg.parent as usize]
            };
            write_segment(&mut put_u32, si, seg);
            put_u32(5, si, state as u32);
            for k in 0..seg.len as usize {
                let tok = seg.start as usize + k;
                put_u32(1, tok, batch.token_ids[tok]);
                put_u32(2, tok, seg.pos0 + k as u32);
                put_u32(3, tok, si as u32);
            }
        }
        let pad_seg = if pad > 0 {
            let seg = Segment { start: t as u32, len: pad as u32, parent: PAD_PARENT, pos0: 0 };
            write_segment(&mut put_u32, s, &seg);
            put_u32(5, s, u32::MAX);
            for k in 0..pad {
                put_u32(1, t + k, 0);
                put_u32(2, t + k, k as u32);
                put_u32(3, t + k, s as u32);
            }
            s as u32
        } else {
            u32::MAX
        };
        for (r, (&row, &slots)) in batch.answer_rows.iter().zip(&batch.answer_slots).enumerate() {
            put_u32(6, r, row);
            put_u32(7, r, slots as u32);
        }
        for r in q..shape.answers {
            put_u32(6, r, 0);
            put_u32(7, r, 0);
        }
        let header = [t as u32, s as u32, q as u32, prefixes, bucket_tokens as u32, pad_seg, 0, 0];
        for (k, v) in header.into_iter().enumerate() {
            put_u32(0, k, v);
        }
        slot.packed = Some(PackedCounts {
            tokens: t,
            segments: s,
            answers: q,
            prefixes: prefixes as usize,
            bucket_tokens,
        });
        Ok(i)
    }

    /// Enqueue the H2D copy of a packed slot on `stream` (ordered before whatever is launched after it on
    /// that stream). Only the used prefix of each array is copied.
    pub fn upload(&mut self, slot: usize, stream: &Stream) -> Result<PackedCounts> {
        let shape = self.dev.shape;
        let s = &mut self.slots[slot];
        let Some(c) = s.packed else { anyhow::bail!("upload of an empty metadata slot") };
        let l = shape.layout();
        let used = [
            HEADER_LEN * 4,
            c.bucket_tokens * 4,
            c.bucket_tokens * 4,
            c.bucket_tokens * 4,
            (c.segments + (c.bucket_tokens > c.tokens) as usize) * 16,
            (c.segments + (c.bucket_tokens > c.tokens) as usize) * 4,
            shape.answers * 4,
            shape.answers * 4,
        ];
        stream.context().bind_to_thread()?;
        let host = s.bytes.as_ptr();
        let dev = self.dev.base_ptr();
        for (k, &n) in used.iter().enumerate() {
            if n == 0 {
                continue;
            }
            unsafe {
                cuda_core::memory::memcpy_htod_async(dev + l[k].0 as u64, host.add(l[k].0), n, stream.cu_stream())?
            };
        }
        s.done.record(stream)?;
        s.in_flight = true;
        Ok(c)
    }

    /// Host view of a packed slot (tests, debugging).
    pub fn host_slot(&self, slot: usize) -> &[u8] {
        self.slots[slot].bytes.as_slice()
    }

    /// Offsets of the arrays in the packed layout.
    pub fn layout(&self) -> [(usize, usize); 8] {
        self.dev.shape.layout()
    }
}

fn write_segment(put: &mut impl FnMut(usize, usize, u32), si: usize, seg: &Segment) {
    put(4, si * 4, seg.start);
    put(4, si * 4 + 1, seg.len);
    put(4, si * 4 + 2, seg.parent as u32);
    put(4, si * 4 + 3, seg.pos0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_aligned_and_disjoint() {
        let s = MetaShape { tokens: 1000, segments: 33, answers: 17 };
        let l = s.layout();
        for w in l.windows(2) {
            assert!(w[0].0 + w[0].1 <= w[1].0);
            assert_eq!(w[1].0 % 256, 0);
        }
        assert_eq!(s.total_bytes(), l[7].0 + 17 * 4);
    }
}
