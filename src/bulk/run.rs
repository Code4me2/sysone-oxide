//! The bulk coordinator: input → planning → window → packing → workers → answers → results file.
//!
//! ```text
//! reader thread ──► planner threads ──► dispatcher (this thread) ──► worker 0..N ──► dispatcher ──► writer thread
//!  parse lines,      server::prepare      reorder by input order,     one engine       logits rows       server::finish,
//!  skip done ids     (validate, plan)      window of W plans,          each             per request       append lines
//!                                          choose(), send steps
//! ```
//!
//! Every channel is bounded, so memory stays bounded however long the input is: at most `W` planned requests in
//! the window, the steps in flight (`depth` per worker), and a few per planner. The done and seen id sets hold 16
//! bytes per id.
//!
//! Determinism: planned requests enter the window in input order, and a step is chosen only from a full window (or
//! once the input is exhausted), so with the same input, window, worker limits and resume state the sequence of
//! steps is the same on every run. Which worker takes which step depends on timing; with one worker (or workers
//! with equal limits) the batches are identical, and so are the GPU results.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::job::{IdKey, Line, Payload, doc_file, parse_doc_line, parse_line, result_line};
use super::qset::QuestionSet;
use super::output::Output;
use super::pack::{PackPolicy, Size, Throughput, choose, fits_alone};
use super::wire::Hello;
use super::worker::{Event, Handle, StepMsg};
use crate::api::server::{self, Outcome, Ready, ServerInfo};
use crate::contract::{DecisionBatch, MAX_SLOTS, StepLimits};
use crate::frontend::Frontend;
use crate::frontend::pyjson::Json;
use crate::sched::Scored;

/// How to start one worker.
pub enum WorkerStart {
    Thread(super::worker::EngineSpec),
    Process(std::process::Command),
}

pub struct Worker {
    pub name: String,
    pub start: WorkerStart,
}

/// What a run reads.
pub enum Input {
    /// A JSONL file of jobs (`{"id", "body"}`).
    Jobs(PathBuf),
    /// A question set and its documents: a JSONL file (`{"id", "state"}`) or a directory of text files.
    Docs { questions: Arc<QuestionSet>, docs: PathBuf },
}

pub struct RunOptions {
    pub input: Input,
    pub output: PathBuf,
    pub workers: Vec<Worker>,
    /// Planned requests the packer chooses from.
    pub window: usize,
    /// Steps queued per worker (1 running + the rest waiting), so a GPU never waits for the coordinator.
    pub depth: usize,
    pub policy: PackPolicy,
    /// Query string for jobs without their own `query`.
    pub query: Option<String>,
    pub info: ServerInfo,
    pub cpu_workers: usize,
    pub progress: Option<Duration>,
    /// Process at most this many jobs (not counting the ones skipped as done).
    pub limit: Option<u64>,
    /// Write one line per step here: `{"step", "worker", "bucket", "tokens", "ids"}` (determinism checks, tuning).
    pub trace: Option<PathBuf>,
}

/// The input as a stream of parsed records, read by the reader thread.
enum Source {
    /// JSONL: jobs, or documents of a question set.
    Lines { r: BufReader<std::fs::File>, docs: bool, line: u64, buf: Vec<u8> },
    /// A directory of text files, in file-name order (names read up front, contents one at a time).
    Dir { dir: PathBuf, names: std::vec::IntoIter<Result<String, String>>, index: u64 },
}

impl Source {
    fn open(input: &Input) -> Result<Self> {
        let (path, docs) = match input {
            Input::Jobs(p) => (p, false),
            Input::Docs { docs, .. } => (docs, true),
        };
        if docs && path.is_dir() {
            Ok(Source::Dir { dir: path.clone(), names: doc_names(path)?.into_iter(), index: 0 })
        } else {
            let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            Ok(Source::Lines { r: BufReader::with_capacity(1 << 20, f), docs, line: 0, buf: Vec::new() })
        }
    }

    /// How reports name a record.
    fn what(&self) -> &'static str {
        match self {
            Source::Lines { .. } => "input line",
            Source::Dir { .. } => "document file #",
        }
    }

    /// The next record; None at the end of the input.
    fn next(&mut self) -> Result<Option<Line>> {
        match self {
            Source::Lines { r, docs, line, buf } => {
                buf.clear();
                if r.read_until(b'\n', buf)? == 0 {
                    return Ok(None);
                }
                *line += 1;
                if buf.last() == Some(&b'\n') {
                    buf.pop();
                }
                Ok(Some(if *docs { parse_doc_line(*line, buf) } else { parse_line(*line, buf) }))
            }
            Source::Dir { dir, names, index } => {
                let Some(name) = names.next() else { return Ok(None) };
                *index += 1;
                Ok(Some(match name {
                    Err(msg) => Line::Bad { line: *index, msg },
                    Ok(name) => match std::fs::read(dir.join(&name)) {
                        Ok(bytes) => doc_file(*index, &name, bytes),
                        Err(e) => Line::Bad { line: *index, msg: format!("reading {name:?}: {e}") },
                    },
                }))
            }
        }
    }
}

/// The documents of a directory: regular files (or links to them) whose names do not start with `.`, sorted by
/// name; a name that is not UTF-8 is reported as a bad record.
fn doc_names(dir: &std::path::Path) -> Result<Vec<Result<String, String>>> {
    let mut names = Vec::new();
    for e in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let e = e?;
        let is_file = std::fs::metadata(e.path()).map(|m| m.is_file()).unwrap_or(false);
        let name = e.file_name();
        if !is_file || name.as_encoded_bytes().first() == Some(&b'.') {
            continue;
        }
        names.push(name.into_string().map_err(|n| format!("file name {n:?} is not UTF-8")));
    }
    names.sort_by(|a, b| match (a, b) {
        (Ok(a), Ok(b)) => a.cmp(b),
        (Err(a), Err(b)) => a.cmp(b),
        (Ok(_), Err(_)) => std::cmp::Ordering::Less,
        (Err(_), Ok(_)) => std::cmp::Ordering::Greater,
    });
    Ok(names)
}

/// Per-worker totals.
#[derive(Clone, Debug, Default, Serialize)]
pub struct WorkerSummary {
    pub name: String,
    pub device: String,
    pub buckets: Vec<usize>,
    pub load_s: f64,
    pub steps: u64,
    pub requests: u64,
    pub decisions: u64,
    pub packed_tokens: u64,
    pub bucket_tokens: u64,
    pub padding_frac: f64,
    /// The engine's `info` pairs (GDN backend, architecture; for mock workers where gpu.sh placed them).
    pub info: Vec<(String, String)>,
    /// Sum of step wall times.
    pub busy_s: f64,
    /// `busy_s` over the time from this worker's HELLO to the end of the run.
    pub busy_frac: f64,
    /// This worker's share of all packed tokens.
    pub token_share: f64,
    pub packed_tok_per_s: f64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub input_lines: u64,
    /// Jobs read and not skipped (processed or failed).
    pub jobs: u64,
    pub skipped_done: u64,
    pub resumed_lines: u64,
    pub truncated_partial_bytes: u64,
    pub bad_lines: u64,
    pub duplicate_ids: u64,
    /// Results written, by HTTP status.
    pub written: BTreeMap<u16, u64>,
    pub ok: u64,
    /// Not written (a rerun retries them): engine failures and requests that fit no worker.
    pub failed: u64,
    pub unrunnable: u64,
    pub decisions: u64,
    pub steps: u64,
    pub packed_tokens: u64,
    pub bucket_tokens: u64,
    pub padding_frac: f64,
    /// From the start of the run (input opened) to the last result written.
    pub wall_s: f64,
    /// From the first worker's HELLO to the last result: the throughput window (excludes model loading).
    pub run_s: f64,
    pub docs_per_s: f64,
    pub decisions_per_s: f64,
    pub packed_tok_per_s: f64,
    pub window: usize,
    pub workers: Vec<WorkerSummary>,
    /// Planning time (validation, rendering, tokenization) summed over the planner threads, and per job planned:
    /// the CPU cost of feeding the GPUs (wall time inside planning, so it includes any wait for a core).
    pub plan_s: f64,
    pub plan_ms_per_job: f64,
    pub planner_threads: usize,
    /// Question-set runs only.
    pub question_set: Option<QuestionSetSummary>,
    /// Set when the run stopped early (every worker gone); a rerun resumes.
    pub aborted: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct QuestionSetSummary {
    pub questions: usize,
    /// At the default debias.
    pub branches: usize,
    pub longest_suffix_tokens: usize,
    pub suffix_tokens: usize,
    /// Documents planned from the prepared question set, and from their full body (they override `model` or
    /// `questions`).
    pub docs_prepared: u64,
    pub docs_full_body: u64,
}

/// A planned request waiting in the window or in a step.
struct Item {
    id: Json,
    ready: Ready,
    size: Size,
}

struct Planned {
    seq: u64,
    id: Json,
    result: Result<Ready, Outcome>,
}

enum WriteMsg {
    Answer { id: Json, ready: Ready, scored: Scored },
    Error { id: Json, outcome: Outcome },
}

struct InFlight {
    worker: usize,
    items: Vec<(Item, usize)>,
    tokens: usize,
}

struct WState {
    handle: Handle,
    hello: Option<Hello>,
    limits: Option<StepLimits>,
    tput: Throughput,
    alive: bool,
    reported: bool,
    ready_at: Option<Instant>,
    inflight: usize,
    sum: WorkerSummary,
}

/// Counters shared with the writer thread.
#[derive(Default)]
struct WriteStats {
    written: BTreeMap<u16, u64>,
    ok: u64,
    decisions: u64,
    last_write: Option<Instant>,
}

pub fn run(opts: RunOptions, frontend: Arc<Frontend>) -> Result<Summary> {
    let t0 = Instant::now();
    anyhow::ensure!(!opts.workers.is_empty(), "no workers");
    anyhow::ensure!(opts.window > 0 && opts.depth > 0, "window and depth must be positive");
    let (out, done, resume) = Output::open(&opts.output)?;
    if resume.lines > 0 || resume.truncated_bytes > 0 {
        eprintln!(
            "sysone bulk: resuming: {} results already in {} ({} bytes of a partial last line cut)",
            resume.lines,
            opts.output.display(),
            resume.truncated_bytes
        );
    }
    let mut source = Source::open(&opts.input)?;
    let qset = match &opts.input {
        Input::Docs { questions, .. } => Some(questions.clone()),
        Input::Jobs(_) => None,
    };
    let mut summary = Summary {
        resumed_lines: resume.lines,
        truncated_partial_bytes: resume.truncated_bytes,
        window: opts.window,
        ..Default::default()
    };

    // Workers first: GPU workers load for tens of seconds while the input is planned.
    let (ev_tx, ev_rx) = mpsc::channel::<Event>();
    let mut ws: Vec<WState> = Vec::new();
    for (i, w) in opts.workers.into_iter().enumerate() {
        let handle = match w.start {
            WorkerStart::Thread(spec) => Handle::thread(i, w.name.clone(), spec, ev_tx.clone())?,
            WorkerStart::Process(cmd) => Handle::process(i, w.name.clone(), cmd, ev_tx.clone())?,
        };
        ws.push(WState {
            handle,
            hello: None,
            limits: None,
            tput: Throughput::gb10_default(),
            alive: true,
            reported: false,
            ready_at: None,
            inflight: 0,
            sum: WorkerSummary { name: w.name, ..Default::default() },
        });
    }
    drop(ev_tx);

    // Reader → planners.
    let cpu = opts.cpu_workers.max(1);
    let (job_tx, job_rx) = mpsc::sync_channel::<(u64, super::job::Job)>(cpu * 4);
    let reader_stats = Arc::new(Mutex::new((0u64, 0u64, 0u64, 0u64, 0u64))); // lines, jobs, skipped, bad, dup
    let rs = reader_stats.clone();
    let limit = opts.limit;
    // Credits: the reader stays at most `ahead` jobs past the last one the dispatcher took in, which bounds what
    // the planners and the reorder buffer hold (and lets the dispatcher always drain the planners' channel).
    let ahead = (opts.window + cpu * 4) as u64;
    let consumed = Arc::new((Mutex::new(0u64), std::sync::Condvar::new()));
    let credits = consumed.clone();
    let reader = std::thread::Builder::new().name("bulk-reader".into()).spawn(move || -> Result<()> {
        let mut seen: HashSet<IdKey> = HashSet::new();
        let (mut lines, mut jobs, mut skipped, mut bad, mut dup) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let mut shown = 0;
        let what = source.what();
        loop {
            let Some(parsed) = source.next()? else { break };
            lines += 1;
            match parsed {
                Line::Blank => {}
                Line::Bad { line, msg } => {
                    bad += 1;
                    if shown < 20 {
                        eprintln!("sysone bulk: {what} {line}: {msg}; skipped");
                        shown += 1;
                    }
                }
                Line::Job(j) => {
                    if done.contains(&j.key) {
                        skipped += 1;
                    } else if !seen.insert(j.key) {
                        dup += 1;
                        if shown < 20 {
                            eprintln!("sysone bulk: {what} {}: duplicate id {}; skipped", j.line, j.id.dumps_compact().unwrap_or_default());
                            shown += 1;
                        }
                    } else {
                        if limit.is_some_and(|l| jobs >= l) {
                            break;
                        }
                        let (m, cv) = &*credits;
                        let mut c = m.lock().unwrap();
                        while jobs >= *c + ahead {
                            c = cv.wait(c).unwrap();
                        }
                        drop(c);
                        if job_tx.send((jobs, j)).is_err() {
                            break;
                        }
                        jobs += 1;
                    }
                }
            }
            *rs.lock().unwrap() = (lines, jobs, skipped, bad, dup);
        }
        *rs.lock().unwrap() = (lines, jobs, skipped, bad, dup);
        Ok(())
    })?;

    let (plan_tx, plan_rx) = mpsc::sync_channel::<Planned>(opts.window.max(cpu * 2));
    let job_rx = Arc::new(Mutex::new(job_rx));
    let mut planners = Vec::new();
    // (nanoseconds planning, jobs planned) over all planners.
    let plan_time = Arc::new((AtomicU64::new(0), AtomicU64::new(0)));
    for k in 0..cpu {
        let (rx, tx, fe, info, query) = (job_rx.clone(), plan_tx.clone(), frontend.clone(), opts.info.clone(), opts.query.clone());
        let (qset, plan_time) = (qset.clone(), plan_time.clone());
        planners.push(std::thread::Builder::new().name(format!("bulk-plan-{k}")).spawn(move || {
            loop {
                let next = rx.lock().unwrap().recv();
                let Ok((seq, job)) = next else { return };
                let t = Instant::now();
                let p = server::params(job.query.as_deref().or(query.as_deref()), info.default_debias);
                let result = match (job.payload, &qset) {
                    (Payload::Body(body), _) => server::prepare(&fe, &info, body, &p),
                    (Payload::Doc(doc), Some(qs)) => qs.prepare(&fe, &info, doc, &p),
                    (Payload::Doc(_), None) => Err(Outcome::Internal),
                };
                plan_time.0.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                plan_time.1.fetch_add(1, Ordering::Relaxed);
                if tx.send(Planned { seq, id: job.id, result }).is_err() {
                    return;
                }
            }
        })?);
    }
    drop(plan_tx);

    // Writer.
    let (w_tx, w_rx) = mpsc::sync_channel::<WriteMsg>(opts.window.max(64));
    let wstats = Arc::new(Mutex::new(WriteStats::default()));
    let (fe, info, wst) = (frontend.clone(), opts.info.clone(), wstats.clone());
    let writer = std::thread::Builder::new().name("bulk-writer".into()).spawn(move || writer_main(out, w_rx, fe, info, wst))?;

    let mut trace = match &opts.trace {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p).with_context(|| format!("creating {}", p.display()))?)),
        None => None,
    };
    // Dispatcher.
    let mut reorder: BTreeMap<u64, Planned> = BTreeMap::new();
    let mut next_seq = 0u64;
    let mut window: VecDeque<Item> = VecDeque::new();
    let mut inflight: HashMap<u64, InFlight> = HashMap::new();
    let mut next_step = 0u64;
    let mut planned_open = true;
    let mut first_ready: Option<Instant> = None;
    let mut last_progress = Instant::now();
    let (mut failed, mut unrunnable) = (0u64, 0u64);
    let mut abort: Option<String> = None;
    let w_cap = opts.window;
    loop {
        // 1. Intake, in input order, up to the window (the rest waits in the bounded channel).
        let mut progressed = false;
        while planned_open {
            let r = if window.len() < w_cap && !reorder.contains_key(&next_seq) {
                plan_rx.recv_timeout(Duration::from_millis(2)).map_err(|e| match e {
                    RecvTimeoutError::Timeout => TryRecvError::Empty,
                    RecvTimeoutError::Disconnected => TryRecvError::Disconnected,
                })
            } else {
                plan_rx.try_recv()
            };
            match r {
                Ok(p) => {
                    reorder.insert(p.seq, p);
                    progressed = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => planned_open = false,
            }
        }
        while window.len() < w_cap
            && let Some(p) = reorder.remove(&next_seq)
        {
            next_seq += 1;
            match p.result {
                Ok(ready) => {
                    let size = Size { tokens: ready.plan.total, branches: ready.plan.branches.len() };
                    window.push_back(Item { id: p.id, ready, size });
                }
                Err(outcome) => {
                    let _ = w_tx.send(WriteMsg::Error { id: p.id, outcome });
                }
            }
            progressed = true;
        }
        {
            let (m, cv) = &*consumed;
            let mut c = m.lock().unwrap();
            if *c != next_seq {
                *c = next_seq;
                cv.notify_all();
            }
        }
        let input_done = !planned_open && reorder.is_empty();

        // 2. Worker events.
        loop {
            match ev_rx.try_recv() {
                Ok(ev) => {
                    progressed = true;
                    handle_event(ev, &mut ws, &mut inflight, &mut window, &w_tx, &mut failed, &mut first_ready, &mut summary)?;
                }
                Err(_) => break,
            }
        }

        // 3. Requests no live worker can hold (decided once every worker has reported).
        if ws.iter().all(|w| w.reported) {
            let alive: Vec<&StepLimits> = ws.iter().filter(|w| w.alive).filter_map(|w| w.limits.as_ref()).collect();
            if alive.is_empty() {
                abort = Some("no worker is running".into());
                break;
            }
            let before = window.len();
            window.retain(|it| {
                let ok = alive.iter().any(|l| fits_alone(it.size, l));
                if !ok {
                    eprintln!(
                        "sysone bulk: id {}: {} tokens / {} branches fit no worker's step; not run",
                        it.id.dumps_compact().unwrap_or_default(),
                        it.size.tokens,
                        it.size.branches
                    );
                }
                ok
            });
            if window.len() != before {
                unrunnable += (before - window.len()) as u64;
                progressed = true;
            }
        }

        // 4. Dispatch: only from a full window or the tail of the input, so the choice is deterministic.
        for wi in 0..ws.len() {
            if window.len() < w_cap && !input_done {
                break;
            }
            {
                let w = &ws[wi];
                if !w.alive || w.inflight >= opts.depth {
                    continue;
                }
                let Some(lim) = w.limits.clone() else { continue };
                let sizes: Vec<Size> = window.iter().map(|it| it.size).collect();
                let pick = choose(&sizes, &lim, &w.tput, &opts.policy, input_done)
                    .or_else(|| if input_done { None } else { choose(&sizes, &lim, &w.tput, &opts.policy, true) });
                let Some(c) = pick else { continue };
                let mut chosen = vec![false; window.len()];
                for &i in &c.picks {
                    chosen[i] = true;
                }
                let mut items = Vec::with_capacity(c.picks.len());
                let mut taken: Vec<Option<Item>> = Vec::with_capacity(window.len());
                let mut rest = VecDeque::with_capacity(window.len());
                for (k, it) in window.drain(..).enumerate() {
                    if chosen[k] {
                        taken.push(Some(it));
                    } else {
                        taken.push(None);
                        rest.push_back(it);
                    }
                }
                window = rest;
                let mut batch = DecisionBatch::default();
                let mut wants_mass = false;
                for &i in &c.picks {
                    let it = taken[i].take().unwrap();
                    let first = it.ready.plan.push_into(&mut batch)?;
                    wants_mass |= it.ready.debug;
                    items.push((it, first));
                }
                batch.validate().context("packed batch")?;
                let id = next_step;
                next_step += 1;
                let tokens = batch.num_tokens();
                if let Some(t) = &mut trace {
                    use std::io::Write as _;
                    let ids: Vec<String> = items.iter().map(|(it, _)| it.id.dumps_compact().unwrap_or_default()).collect();
                    writeln!(
                        t,
                        "{{\"step\":{id},\"worker\":{:?},\"bucket\":{},\"tokens\":{tokens},\"ids\":[{}]}}",
                        ws[wi].handle.name,
                        c.bucket,
                        ids.join(",")
                    )?;
                }
                if ws[wi].handle.send(StepMsg { id, wants_mass, batch }) {
                    ws[wi].inflight += 1;
                    inflight.insert(id, InFlight { worker: wi, items, tokens });
                } else {
                    // The worker is gone; its Exited event follows. Put the requests back.
                    for (it, _) in items.into_iter().rev() {
                        window.push_front(it);
                    }
                    ws[wi].alive = false;
                }
                progressed = true;
            }
        }

        // 5. Done?
        if input_done && window.is_empty() && inflight.is_empty() {
            break;
        }
        if ws.iter().all(|w| !w.alive) {
            abort = Some("every worker stopped".into());
            break;
        }
        if let Some(every) = opts.progress
            && last_progress.elapsed() >= every
        {
            last_progress = Instant::now();
            progress_line(&ws, &wstats, &reader_stats, first_ready, window.len());
        }
        if !progressed {
            match ev_rx.recv_timeout(Duration::from_millis(5)) {
                Ok(ev) => handle_event(ev, &mut ws, &mut inflight, &mut window, &w_tx, &mut failed, &mut first_ready, &mut summary)?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    if !inflight.is_empty() || !window.is_empty() {
                        abort = Some("every worker stopped".into());
                        break;
                    }
                }
            }
        }
    }

    // Shut down: workers, reader and planners, writer.
    let end_dispatch = Instant::now();
    if abort.is_some() {
        failed += inflight.values().map(|f| f.items.len() as u64).sum::<u64>();
        for w in &mut ws {
            w.handle.kill();
        }
    }
    for w in &mut ws {
        // A worker that never reported (still loading, or waiting for its GPU's lock) is not needed any more.
        if !w.reported {
            eprintln!("sysone bulk: worker {} never became ready; stopping it", w.handle.name);
            w.sum.error.get_or_insert_with(|| "never ready (loading, or waiting for its GPU lock)".into());
            w.handle.kill();
        }
        w.handle.close();
    }
    if let Some(mut t) = trace.take() {
        use std::io::Write as _;
        t.flush()?;
    }
    drop(plan_rx);
    {
        // Unblock a reader waiting for credits; with the planners' channel gone it stops at its next send.
        let (m, cv) = &*consumed;
        *m.lock().unwrap() = u64::MAX / 2;
        cv.notify_all();
    }
    let _ = reader.join();
    for p in planners {
        let _ = p.join();
    }
    drop(w_tx);
    let write_result = writer.join().map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
    let (lines, jobs, skipped, bad, dup) = *reader_stats.lock().unwrap();
    let w = wstats.lock().unwrap();
    let end = w.last_write.unwrap_or(end_dispatch).max(end_dispatch);
    summary.input_lines = lines;
    summary.jobs = jobs;
    summary.skipped_done = skipped;
    summary.bad_lines = bad;
    summary.duplicate_ids = dup;
    summary.written = w.written.clone();
    summary.ok = w.ok;
    summary.decisions = w.decisions;
    summary.failed = failed;
    summary.unrunnable = unrunnable;
    summary.wall_s = end.duration_since(t0).as_secs_f64();
    summary.run_s = first_ready.map_or(0.0, |f| end.saturating_duration_since(f).as_secs_f64());
    let run = summary.run_s.max(1e-9);
    summary.docs_per_s = summary.ok as f64 / run;
    summary.decisions_per_s = summary.decisions as f64 / run;
    summary.packed_tok_per_s = summary.packed_tokens as f64 / run;
    summary.padding_frac =
        if summary.bucket_tokens > 0 { 1.0 - summary.packed_tokens as f64 / summary.bucket_tokens as f64 } else { 0.0 };
    let total_tokens = summary.packed_tokens.max(1) as f64;
    for w in &mut ws {
        let s = &mut w.sum;
        let up = w.ready_at.map_or(0.0, |r| end.saturating_duration_since(r).as_secs_f64());
        s.busy_frac = if up > 0.0 { s.busy_s / up } else { 0.0 };
        s.token_share = s.packed_tokens as f64 / total_tokens;
        s.packed_tok_per_s = if up > 0.0 { s.packed_tokens as f64 / up } else { 0.0 };
        s.padding_frac = if s.bucket_tokens > 0 { 1.0 - s.packed_tokens as f64 / s.bucket_tokens as f64 } else { 0.0 };
        summary.workers.push(s.clone());
    }
    drop(w);
    write_result?;
    let (plan_ns, planned) = (plan_time.0.load(Ordering::Relaxed), plan_time.1.load(Ordering::Relaxed));
    summary.plan_s = plan_ns as f64 * 1e-9;
    summary.plan_ms_per_job = if planned > 0 { plan_ns as f64 * 1e-6 / planned as f64 } else { 0.0 };
    summary.planner_threads = cpu;
    summary.question_set = qset.map(|q| QuestionSetSummary {
        questions: q.num_questions,
        branches: q.num_branches,
        longest_suffix_tokens: q.suffix_tokens.0,
        suffix_tokens: q.suffix_tokens.1,
        docs_prepared: q.docs_prepared.load(Ordering::Relaxed),
        docs_full_body: q.docs_full_body.load(Ordering::Relaxed),
    });
    summary.aborted = abort;
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn handle_event(
    ev: Event,
    ws: &mut [WState],
    inflight: &mut HashMap<u64, InFlight>,
    window: &mut VecDeque<Item>,
    w_tx: &SyncSender<WriteMsg>,
    failed: &mut u64,
    first_ready: &mut Option<Instant>,
    summary: &mut Summary,
) -> Result<()> {
    match ev {
        Event::Ready(i, hello) => {
            let w = &mut ws[i];
            eprintln!(
                "sysone bulk: worker {} ready: {} ({}), buckets {:?}, {} prefixes, loaded in {:.1} s",
                w.handle.name,
                hello.device,
                hello.arch,
                hello.buckets,
                hello.max_prefixes,
                hello.load_seconds
            );
            let now = Instant::now();
            first_ready.get_or_insert(now);
            w.ready_at = Some(now);
            w.limits = Some(hello.limits());
            w.tput = Throughput(hello.throughput.clone());
            w.sum.device = hello.device.clone();
            w.sum.buckets = hello.buckets.clone();
            w.sum.load_s = hello.load_seconds;
            w.sum.info = hello.info.clone();
            w.reported = true;
            w.hello = Some(hello);
        }
        Event::Done(i, d) => {
            let Some(f) = inflight.remove(&d.id) else { bail!("worker {i} finished unknown step {}", d.id) };
            let w = &mut ws[f.worker];
            w.inflight -= 1;
            let now = Instant::now();
            let step = Duration::from_secs_f64(d.step_seconds);
            let step_started = now.checked_sub(step).unwrap_or(now);
            let n_req = f.items.len();
            let bucket = d.bucket_tokens.unwrap_or(f.tokens);
            w.sum.steps += 1;
            w.sum.requests += n_req as u64;
            w.sum.packed_tokens += f.tokens as u64;
            w.sum.bucket_tokens += bucket as u64;
            w.sum.busy_s += d.step_seconds;
            summary.steps += 1;
            summary.packed_tokens += f.tokens as u64;
            summary.bucket_tokens += bucket as u64;
            for (it, first) in f.items {
                let n = it.ready.plan.branches.len();
                w.sum.decisions += it.ready.plan.num_questions as u64;
                let scored = Scored {
                    logits: d.logits.data[first * MAX_SLOTS..(first + n) * MAX_SLOTS].to_vec(),
                    allowed_mass: d.masses.as_ref().map(|m| m[first..first + n].to_vec()),
                    queue_wait: step_started.saturating_duration_since(it.ready.planned),
                    step_started,
                    step,
                    batch_tokens: f.tokens,
                    batch_requests: n_req,
                };
                w_tx.send(WriteMsg::Answer { id: it.id, ready: it.ready, scored }).map_err(|_| anyhow::anyhow!("writer stopped"))?;
            }
        }
        Event::Failed(i, id, msg) => {
            let Some(f) = inflight.remove(&id) else { bail!("worker {i} failed unknown step {id}") };
            ws[f.worker].inflight -= 1;
            *failed += f.items.len() as u64;
            eprintln!(
                "sysone bulk: worker {}: step {id} ({} requests, {} tokens) failed: {msg}; its requests are not written \
                 (rerun to retry them)",
                ws[f.worker].handle.name,
                f.items.len(),
                f.tokens
            );
        }
        Event::Exited(i, msg) => {
            let w = &mut ws[i];
            let was_alive = w.alive;
            w.alive = false;
            w.reported = true;
            // Steps it had in flight go back to the window (front, in their order), for another worker.
            let mine: Vec<u64> = inflight.iter().filter(|(_, f)| f.worker == i).map(|(&id, _)| id).collect();
            let mut back = Vec::new();
            for id in mine {
                let f = inflight.remove(&id).unwrap();
                back.extend(f.items.into_iter().map(|(it, _)| it));
            }
            w.inflight = 0;
            if was_alive {
                eprintln!("sysone bulk: worker {} stopped: {msg}; {} requests requeued", w.handle.name, back.len());
                w.sum.error = Some(msg);
            }
            for it in back.into_iter().rev() {
                window.push_front(it);
            }
        }
    }
    Ok(())
}

fn writer_main(
    mut out: Output,
    rx: Receiver<WriteMsg>,
    frontend: Arc<Frontend>,
    info: ServerInfo,
    stats: Arc<Mutex<WriteStats>>,
) -> Result<()> {
    let mut buf = Vec::new();
    let mut res = Ok(());
    while let Ok(first) = rx.recv() {
        let mut msgs = vec![first];
        while let Ok(m) = rx.try_recv() {
            msgs.push(m);
            if msgs.len() >= 256 {
                break;
            }
        }
        buf.clear();
        let mut st = Vec::new();
        for m in msgs {
            let (id, outcome, decisions) = match m {
                WriteMsg::Answer { id, ready, scored } => {
                    let o = server::finish(&frontend, &info, &ready, &scored);
                    let d = if o.status() == 200 { ready.plan.num_questions as u64 } else { 0 };
                    (id, o, d)
                }
                WriteMsg::Error { id, outcome } => (id, outcome, 0),
            };
            let status = outcome.status();
            let item = outcome.into_item();
            let body = item.get("body").cloned().unwrap_or(Json::Null);
            match result_line(&id, status, &body) {
                Ok(l) => {
                    buf.extend_from_slice(&l);
                    st.push((status, decisions));
                }
                Err(e) => eprintln!("sysone bulk: id {}: cannot serialise the result: {e}", id.dumps_compact().unwrap_or_default()),
            }
        }
        if res.is_ok() {
            res = out.write(&buf);
        }
        let mut s = stats.lock().unwrap();
        for (status, d) in st {
            *s.written.entry(status).or_default() += 1;
            if status == 200 {
                s.ok += 1;
                s.decisions += d;
            }
        }
        s.last_write = Some(Instant::now());
    }
    res?;
    out.sync()
}

fn progress_line(
    ws: &[WState],
    wstats: &Mutex<WriteStats>,
    reader: &Mutex<(u64, u64, u64, u64, u64)>,
    first_ready: Option<Instant>,
    window: usize,
) {
    let (_, jobs, skipped, bad, dup) = *reader.lock().unwrap();
    let s = wstats.lock().unwrap();
    let written: u64 = s.written.values().sum();
    let t = first_ready.map_or(0.0, |f| f.elapsed().as_secs_f64()).max(1e-9);
    let tokens: u64 = ws.iter().map(|w| w.sum.packed_tokens).sum();
    let bucket: u64 = ws.iter().map(|w| w.sum.bucket_tokens).sum();
    let pad = if bucket > 0 { 1.0 - tokens as f64 / bucket as f64 } else { 0.0 };
    let per: Vec<String> = ws
        .iter()
        .map(|w| format!("{}:{}{}", w.handle.name, w.sum.steps, if w.alive { "" } else { "(stopped)" }))
        .collect();
    eprintln!(
        "sysone bulk: {written}/{jobs} written ({} ok), {skipped} skipped, {} bad/dup; {:.2} docs/s, {:.1} decisions/s, \
         {:.0} packed tok/s, padding {:.1}%, window {window}; steps {}",
        s.ok,
        bad + dup,
        s.ok as f64 / t,
        s.decisions as f64 / t,
        tokens as f64 / t,
        pad * 100.0,
        per.join(" ")
    );
}
