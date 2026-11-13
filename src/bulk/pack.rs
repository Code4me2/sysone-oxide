//! Packing for throughput: which requests go into the next engine step, and in which bucket.
//!
//! The engine pads every step up to a T bucket, and padded rows cost almost as much as real ones (every GEMM,
//! norm and pointwise op is token-parallel; only attention and the GDN scan skip the pad segment). The server's
//! batcher packs requests first-in first-out up to its token budget, so a step holds on average about half a
//! request less than the bucket (and a batch over a power of two pays up to the next one). For bulk work order is
//! free, so [`choose`] fills each step from a *window* of planned requests, largest first (first-fit decreasing,
//! one bin at a time):
//!
//! 1. Walk the window from the largest request down and take every request that still fits the bucket's tokens,
//!    the engine's answer rows, segments and prefix slots. Large requests go early, small ones fill the gaps.
//! 2. While more input is coming (`flush == false`) only the largest bucket is filled, and only when the window
//!    can fill it to at least [`PackPolicy::min_fill`]; otherwise the caller reads more input first.
//! 3. When no more input is coming (the tail of the run), every bucket of the ladder is tried and the one with the
//!    most real tokens per second of step time wins, using the worker's throughput curve (`tokens / bucket ×
//!    tok/s(bucket)`). A finer ladder ([`ladder`]) lowers the padding of these last steps.
//!
//! [`simulate_fifo`] reproduces the server's batcher on the same requests, for the padding comparison in
//! docs/BULK.md.

use crate::contract::StepLimits;

/// The size of one request in a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    /// Prefix + every suffix.
    pub tokens: usize,
    /// Suffixes (answer rows).
    pub branches: usize,
}

/// Speed of a step by bucket size: `(bucket tokens, bucket tokens per second)` points, ascending, interpolated
/// linearly and held flat outside them (a step of bucket `b` takes `b / at(b)` seconds, padding included). Only
/// ratios matter.
#[derive(Clone, Debug, PartialEq)]
pub struct Throughput(pub Vec<(usize, f64)>);

impl Throughput {
    /// The bf16 forward on GB10 (SIMT GDN backend, graph replay, bench_forward 2026-11-13, docs/BULK.md): 4,536 /
    /// 4,849 / 4,424 / 4,855 / 5,298 / 5,384 / 5,406 / 5,348 bucket tok/s at 512 ... 65,536. The default when a
    /// worker has no platform profile: nearly flat, so at the end of a run the packer mostly minimises padding.
    pub fn gb10_default() -> Self {
        Throughput(vec![
            (512, 4536.0),
            (1024, 4849.0),
            (2048, 4424.0),
            (4096, 4855.0),
            (8192, 5298.0),
            (16384, 5384.0),
            (32768, 5406.0),
            (65536, 5348.0),
        ])
    }

    pub fn at(&self, tokens: usize) -> f64 {
        let p = &self.0;
        if p.is_empty() {
            return 1.0;
        }
        if tokens <= p[0].0 {
            return p[0].1;
        }
        for w in p.windows(2) {
            let ((t0, v0), (t1, v1)) = (w[0], w[1]);
            if tokens <= t1 {
                let f = (tokens - t0) as f64 / (t1 - t0) as f64;
                return v0 + f * (v1 - v0);
            }
        }
        p.last().unwrap().1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PackPolicy {
    /// While more input is coming, fill the largest bucket at least this much before running it.
    pub min_fill: f64,
}

impl Default for PackPolicy {
    fn default() -> Self {
        PackPolicy { min_fill: 0.98 }
    }
}

/// A chosen step: indices into the window (in the order to pack them), its tokens and its bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub picks: Vec<usize>,
    pub tokens: usize,
    pub bucket: usize,
}

/// Whether one request fits a worker's step at all (else it must go to a larger worker).
pub fn fits_alone(s: Size, lim: &StepLimits) -> bool {
    s.tokens <= lim.max_tokens() && s.branches <= lim.max_answers && s.branches < lim.max_segments && lim.max_prefixes >= 1
}

/// First-fit decreasing into one bin of `cap` tokens. `order` lists window indices from the largest request down.
fn fill(window: &[Size], order: &[usize], cap: usize, lim: &StepLimits) -> (Vec<usize>, usize) {
    let (mut tokens, mut answers, mut prefixes) = (0usize, 0usize, 0usize);
    let max_segments = lim.max_segments.min(cap);
    let mut picks = Vec::new();
    for &i in order {
        let s = window[i];
        if tokens + s.tokens <= cap
            && answers + s.branches <= lim.max_answers.min(cap)
            && prefixes + 1 <= lim.max_prefixes
            && prefixes + 1 + answers + s.branches <= max_segments
        {
            tokens += s.tokens;
            answers += s.branches;
            prefixes += 1;
            picks.push(i);
            if tokens == cap {
                break;
            }
        }
    }
    (picks, tokens)
}

/// The next step for a worker with limits `lim` and throughput curve `tput`, from `window`; None when it should
/// wait for more input (or nothing in the window fits it). See the module docs.
pub fn choose(window: &[Size], lim: &StepLimits, tput: &Throughput, policy: &PackPolicy, flush: bool) -> Option<Choice> {
    let mut order: Vec<usize> = (0..window.len()).filter(|&i| fits_alone(window[i], lim)).collect();
    if order.is_empty() {
        return None;
    }
    // Largest first; ties by window position, so the choice depends only on the window's contents and order.
    order.sort_by(|&a, &b| window[b].tokens.cmp(&window[a].tokens).then(a.cmp(&b)));
    let largest = lim.max_tokens();
    if !flush {
        let (picks, tokens) = fill(window, &order, largest, lim);
        return (tokens as f64 >= policy.min_fill * largest as f64).then_some(Choice { picks, tokens, bucket: largest });
    }
    let mut best: Option<(f64, Choice)> = None;
    for &b in &lim.buckets {
        let (picks, tokens) = fill(window, &order, b, lim);
        if picks.is_empty() {
            continue;
        }
        let rate = tokens as f64 / b as f64 * tput.at(b);
        // Ties go to the larger bucket (fewer steps).
        if best.as_ref().is_none_or(|(r, _)| rate >= *r) {
            best = Some((rate, Choice { picks, tokens, bucket: b }));
        }
    }
    best.map(|(_, c)| c)
}

/// A bucket ladder up to `max` tokens (inclusive, `max` is always the top): powers of two from 256 (`fine ==
/// false`), or also the midpoints 1.5 x 2^k (`fine == true`: 256, 384, 512, 768, ...).
pub fn ladder(max: usize, fine: bool) -> Vec<usize> {
    let mut v = Vec::new();
    let mut t = 256usize;
    while t < max {
        v.push(t);
        if fine && t + t / 2 < max {
            v.push(t + t / 2);
        }
        t *= 2;
    }
    v.push(max);
    v
}

/// Totals of a sequence of steps: real tokens, bucket tokens, steps, and the estimated time at `tput`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackStats {
    pub steps: usize,
    pub tokens: usize,
    pub bucket_tokens: usize,
    /// Sum over steps of `bucket / tok/s(bucket)`: estimated GPU seconds.
    pub est_seconds: f64,
}

impl PackStats {
    pub fn add(&mut self, tokens: usize, bucket: usize, tput: &Throughput) {
        self.steps += 1;
        self.tokens += tokens;
        self.bucket_tokens += bucket;
        self.est_seconds += bucket as f64 / tput.at(bucket);
    }

    pub fn padding(&self) -> f64 {
        if self.bucket_tokens == 0 { 0.0 } else { 1.0 - self.tokens as f64 / self.bucket_tokens as f64 }
    }
}

/// The server's batcher on `sizes` in arrival order under saturating load: whole requests first-in first-out
/// until the next one would cross `budget` (a request over the budget runs alone), each batch padded to the
/// smallest bucket of `lim` that holds it (`GpuEngine::pick_bucket`; answer and prefix limits make the engine
/// split a batch, which this ignores: it does not happen at these sizes).
pub fn simulate_fifo(sizes: &[Size], budget: usize, lim: &StepLimits, tput: &Throughput) -> PackStats {
    let mut st = PackStats::default();
    let mut cur = 0usize;
    for s in sizes {
        if cur > 0 && cur + s.tokens > budget {
            st.add(cur, lim.bucket_for(cur).unwrap_or(cur), tput);
            cur = 0;
        }
        cur += s.tokens;
        if cur >= budget {
            st.add(cur, lim.bucket_for(cur).unwrap_or(cur), tput);
            cur = 0;
        }
    }
    if cur > 0 {
        st.add(cur, lim.bucket_for(cur).unwrap_or(cur), tput);
    }
    st
}

/// [`choose`] over `sizes` arriving in order with a window of `window` requests, one worker: the steps a bulk run
/// would make. Requests that fit no step are skipped (counted in neither total).
pub fn simulate_bulk(sizes: &[Size], window: usize, lim: &StepLimits, tput: &Throughput, policy: &PackPolicy) -> PackStats {
    let mut st = PackStats::default();
    let mut win: Vec<Size> = Vec::new();
    let mut next = 0;
    let fit: Vec<Size> = sizes.iter().copied().filter(|&s| fits_alone(s, lim)).collect();
    loop {
        while win.len() < window && next < fit.len() {
            win.push(fit[next]);
            next += 1;
        }
        if win.is_empty() {
            break;
        }
        let flush = next == fit.len();
        let c = choose(&win, lim, tput, policy, flush)
            .or_else(|| if flush { None } else { choose(&win, lim, tput, policy, true) })
            .expect("a non-empty window of requests that fit always yields a step when flushing");
        st.add(c.tokens, c.bucket, tput);
        let mut picked = vec![false; win.len()];
        for &i in &c.picks {
            picked[i] = true;
        }
        let mut k = 0;
        win.retain(|_| {
            k += 1;
            !picked[k - 1]
        });
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lim(buckets: Vec<usize>) -> StepLimits {
        StepLimits { buckets, max_answers: 2048, max_segments: 2048, max_prefixes: 256 }
    }

    fn sizes(n: usize, seed: u64) -> Vec<Size> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                Size { tokens: 300 + (x % 7700) as usize, branches: 4 + (x >> 20) as usize % 13 }
            })
            .collect()
    }

    #[test]
    fn ladders() {
        assert_eq!(ladder(4096, false), [256, 512, 1024, 2048, 4096]);
        assert_eq!(ladder(4096, true), [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]);
        assert_eq!(ladder(3000, false), [256, 512, 1024, 2048, 3000]);
        assert_eq!(ladder(100, true), [100]);
    }

    #[test]
    fn throughput_interpolates_and_holds() {
        let t = Throughput(vec![(1000, 1.0), (3000, 3.0)]);
        assert_eq!((t.at(10), t.at(2000), t.at(9000)), (1.0, 2.0, 3.0));
    }

    #[test]
    fn choose_respects_every_limit_and_waits_for_a_full_step() {
        let l = StepLimits { buckets: vec![1024, 2048], max_answers: 10, max_segments: 2048, max_prefixes: 3 };
        let t = Throughput::gb10_default();
        let w = vec![
            Size { tokens: 900, branches: 2 },
            Size { tokens: 600, branches: 5 },
            Size { tokens: 500, branches: 4 },
            Size { tokens: 400, branches: 2 },
            Size { tokens: 3000, branches: 1 }, // never fits this worker
        ];
        let c = choose(&w, &l, &t, &PackPolicy { min_fill: 0.9 }, false).unwrap();
        // Largest first: 900 (2), 600 (5): then 500 would make 9+4 answers > 10; 400 fits (3 prefixes, 9 answers).
        assert_eq!(c, Choice { picks: vec![0, 1, 3], tokens: 1900, bucket: 2048 });
        // Not full enough while input is coming: wait.
        assert_eq!(choose(&w[1..], &l, &t, &PackPolicy { min_fill: 0.99 }, false), None);
        // Nothing fits: None even when flushing.
        assert_eq!(choose(&w[4..], &l, &t, &PackPolicy::default(), true), None);
    }

    #[test]
    fn flushing_prefers_the_bucket_with_the_best_rate() {
        let l = lim(vec![256, 512, 1024, 2048, 4096]);
        let flat = Throughput(vec![(1, 1.0)]);
        // 1,000 tokens left: a 1024 bucket (98% full) beats 2048 or 4096.
        let w = vec![Size { tokens: 600, branches: 1 }, Size { tokens: 400, branches: 1 }];
        assert_eq!(choose(&w, &l, &flat, &PackPolicy::default(), true).unwrap().bucket, 1024);
        // With a curve that makes small steps slow, filling 512 alone (one request) loses to 1024 with both.
        let w = vec![Size { tokens: 500, branches: 1 }, Size { tokens: 300, branches: 1 }];
        let slow_small = Throughput(vec![(256, 1000.0), (512, 1500.0), (1024, 4000.0)]);
        let c = choose(&w, &l, &slow_small, &PackPolicy::default(), true).unwrap();
        assert_eq!((c.bucket, c.tokens), (1024, 800));
        // With GB10's nearly flat curve, the full 512 step wins (the 300-token request goes in the next one).
        let c = choose(&w, &l, &Throughput::gb10_default(), &PackPolicy::default(), true).unwrap();
        assert_eq!((c.bucket, c.tokens), (512, 500));
        let c = choose(&w, &l, &slow_small, &PackPolicy::default(), true).unwrap();
        assert_eq!((c.bucket, c.tokens), (1024, 800));
    }

    #[test]
    fn bulk_packing_beats_fifo_on_a_mixed_corpus() {
        let s = sizes(2000, 0x2545f4914f6cdd1d);
        let l = lim(ladder(16384, false));
        let t = Throughput::gb10_default();
        let fifo = simulate_fifo(&s, 16384, &l, &t);
        let bulk = simulate_bulk(&s, 512, &l, &t, &PackPolicy::default());
        assert_eq!(fifo.tokens, bulk.tokens);
        assert!(fifo.padding() > 0.05, "fifo padding {}", fifo.padding());
        assert!(bulk.padding() < 0.02, "bulk padding {}", bulk.padding());
        assert!(bulk.est_seconds < fifo.est_seconds);
        // A finer ladder only changes the tail.
        let fine = simulate_bulk(&s, 512, &lim(ladder(16384, true)), &t, &PackPolicy::default());
        assert!(fine.bucket_tokens <= bulk.bucket_tokens);
    }

    #[test]
    fn packing_is_deterministic() {
        let s = sizes(500, 7);
        let l = lim(ladder(8192, true));
        let t = Throughput::gb10_default();
        let a = simulate_bulk(&s, 64, &l, &t, &PackPolicy::default());
        let b = simulate_bulk(&s, 64, &l, &t, &PackPolicy::default());
        assert_eq!(a, b);
    }
}
