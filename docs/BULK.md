# Bulk tagging: `sysone bulk` (B1)

`sysone bulk` answers a file of `/v1/systemone` requests as fast as the GPUs allow. It reads a JSONL file of jobs,
writes one JSONL result line per job keyed by the job's id, packs requests into steps for throughput rather than
latency, and spreads the steps over one engine per GPU. The answers are the server's: every job goes through the
same validation, planning and answer assembly code as `POST /v1/systemone`, so a result's status and body are what
the server returns for that body.

Numbers: [Results on GB10](#results-on-gb10). Design: [Design](#design). x86 task: [docs/x86/TASKS.md](x86/TASKS.md) T7.

## Usage

```bash
# A synthetic corpus: 2,000 documents, states of 300..8,000 tokens (log-uniform), 4..16 questions (loadgen's
# request shape: noul / choice with 5 options / score with 4 levels).
./target/release/sysone bulk gen --docs 2000 --output corpus.jsonl

# Tag it on every GPU of the host (one worker process per GPU, each started through scripts/gpu.sh --gpu N, which
# takes that GPU's lock). Rerunning the same command resumes: ids already in the output are skipped.
./target/release/sysone bulk --model /path/to/elephant-1 --input corpus.jsonl --output results.jsonl

# Some GPUs only; progress every 30 s on stderr; the final summary (JSON) on stdout and in a file.
./target/release/sysone bulk --model M --input jobs.jsonl --output out.jsonl --gpus 0,2 --summary summary.json

# CPU only: how the server's batcher and bulk windows would pack this file (padding, steps, estimated GPU time).
./target/release/sysone bulk simulate --model M --input corpus.jsonl --max-step 16384,32768 --window 256,1024

# Compare two result files by id (bulk vs bulk, or bulk vs `loadgen --corpus ... --responses FILE`).
./target/release/sysone bulk compare a.jsonl b.jsonl
```

**Input** (one job per line; blank lines are skipped):

```json
{"id": "doc-1", "body": {"state": "...", "questions": {"late": {"type": "noul", "instructions": "Is it late?"}}}}
{"id": 2, "body": {...}, "query": "debias=2&debug=1"}
{"id": "doc-3", "body_text": "{\"state\": ... raw request text ...}"}
```

- `id`: a string or an integer, unique in the file (`"7"` and `7` are different ids).
- `body`: the request body as JSON. `body_text` instead: the body as raw text, parsed exactly like an HTTP body
  (text that is not JSON gets the server's invalid-JSON 422).
- `query` (optional): the query string the request would carry; default `--query` (none: the server's defaults).
- A line that is not a JSON object, has no usable id or body, or repeats an id is a *bad line*: reported on stderr
  with its line number, counted in the summary, not processed (it has no id to key a result by).

**Output** (one line per job, in completion order, appended):

```json
{"id":"doc-1","status":200,"body":{"model":"elephant-1","answers":{...},"usage":{...},"qev":{...}}}
{"id":"doc-3","status":422,"body":{"detail":[...]}}
```

`status` and `body` are what `POST /v1/systemone` answers for that body and query (the body bytes are the server's
compact JSON). `qev.timing` has the server's keys with bulk's own times (`queue_s` includes waiting in the window).
A 500 is written as `{"detail": "Internal Server Error"}`, as `/v1/systemone:batch` reports it.

**Not written** (a rerun retries them): requests in a step the engine failed, and requests larger than every
worker's largest step (the server answers those with a 500; bulk reports them as `unrunnable` and exits 1, so a
run with bigger GPUs or `--max-step-tokens` can pick them up). The exit code is 0 only when every job was written.

**Resume and crash safety.** The output file is locked while a run appends to it. Each result is one write of a
whole line; a crash can leave only a partial last line, which the next run cuts off before appending. On start the
run reads the ids already in the file and skips those jobs (16 bytes per id held in memory). A complete line that
does not parse is corruption from outside: the run refuses to start rather than truncate it.

**Summary** (stdout, `--summary FILE`): jobs, skipped, bad lines, results written by status, failed, unrunnable;
docs/s, decisions/s and packed tok/s over the run (from the first worker ready to the last result); padding
fraction; per worker: device, buckets, steps, requests, packed and bucket tokens, busy seconds and busy fraction,
token share. `--trace FILE` writes one line per step (worker, bucket, tokens, ids).

Options that matter for throughput (`sysone bulk --help` lists all):

| Option | Default | Effect |
|---|---|---|
| `--window` | 1024 | Planned requests the packer chooses each step from. Larger: fuller steps, more memory (a plan holds its token ids and prompt text). |
| `--max-step-tokens` | 32768 | Largest bucket; then cut to what fits the device's memory (`--memory-fraction`, default 0.85 of the memory free after the weights). |
| `--fine-ladder` | off | Buckets at 1.5 x 2^k too: less padding in the last steps of a run, one more graph capture per extra bucket. |
| `--min-fill` | 0.98 | While input is still coming, a step runs only once the window fills the largest bucket this much. |
| `--depth` | 2 | Steps queued per worker (one running, one waiting), so a GPU never waits for the coordinator. |
| `--gpus` | auto | GPUs (nvidia-smi indices or UUIDs); `auto` = `$CUDA_VISIBLE_DEVICES`, else every GPU. |
| `--profile` | none | Platform profiles (`hwprobe --profile-out`); each worker uses the one matching its device name for its throughput curve. |
| `--require-idle` | off | Workers refuse to start while another process uses their GPU (benchmarks). |

## Design

```text
reader thread ──► planner threads ──► dispatcher ──► worker 0..N (one engine each) ──► dispatcher ──► writer thread
 parse lines,      validate + plan     window of W       one process per GPU            logits rows     answer assembly,
 skip done ids     (server code)       planned reqs,     (gpu.sh --gpu N), or a thread  per request     append lines
                                       choose a step
```

Code: `src/bulk/` (module docs in each file). The server's request path (`api::server::prepare` and `finish`) is
shared, not copied.

### Packing for throughput

The engine pads every step to a T bucket, and a padded row costs nearly as much as a real one: every GEMM, norm and
pointwise op is token-parallel (GEMMs are 76% of the forward); only attention and the GDN scan skip the pad segment.
The server's batcher packs requests first-in first-out up to its token budget (16,384 by default), so each step is
short of its bucket by about half a request, and a batch over a power of two pays for the next one.

Bulk work has no order to keep, so `pack::choose` builds each step from a window of W planned requests:

1. **First-fit decreasing, one bin at a time.** Walk the window from the largest request down and take every
   request that still fits the bucket's tokens and the engine's answer rows, segments and prefix slots
   (`DecisionEngine::step_limits`). Large requests go first; small ones fill the gaps.
2. **Only full steps while input is coming.** The largest bucket is filled, and only once the window fills it to
   `--min-fill` (98%); otherwise the dispatcher reads more input first.
3. **The tail of the run** (input exhausted): every bucket of the ladder is tried and the one with the most real
   tokens per second of step time wins, from the worker's throughput curve (`Throughput`: the profile's points, or
   GB10's measured bf16 curve, nearly flat from 512 to 65,536 tokens, so this mostly minimises padding).
4. **Deterministic.** Planned requests enter the window in input order and a step is chosen only from a full window
   (or the exhausted input), so the same input, window, limits and resume state give the same steps on every run,
   whatever the timing; with equal workers the step sequence is the same for one or several workers
   (`tests/bulk.rs::packing_is_deterministic`, `tests/bulk_gpu.rs`). Which worker runs which step depends on timing.

Requests in a step are reordered (by size), and steps hold several requests: results are keyed by id, never by
position.

### Bucket ladder for bulk

`bench_forward` on GB10 (SIMT GDN backend, graph replay, idle GPU, 2026-11-13), bucket tokens per second:

| bucket | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| replay ms | 112.9 | 211.2 | 462.9 | 843.7 | 1,546 | 3,043 | 6,062 | 12,255 |
| bucket tok/s | 4,536 | 4,849 | 4,424 | 4,855 | 5,298 | 5,384 | 5,406 | 5,348 |

The step cost is nearly proportional to the bucket from 1k up: the only thing a smaller step saves is padding.
So for bulk the ladder hardly matters once steps are full, and the largest bucket should be as large as memory
allows, up to the point where throughput stops rising (8k-32k here). The default top is 32,768: as fast per token
as 16,384, and its larger bins pack slightly better (fewer, fuller steps; table below). `--fine-ladder` adds the
1.5 x 2^k buckets; it changes only the last few steps (simulation below: 0.02-0.03 percentage points of padding).

Each card sizes its own ladder: `EngineOptions::memory_fraction` keeps the buckets whose activation arena, GDN scratch
and readback fit in a fraction of the memory free after the weights (`gpu::step_bytes`, computed from the same arena
plan the engine allocates; the engine logs what it dropped). The top bucket needs ~3.8 GiB at 32,768 and ~7.0 GiB
at 65,536 (SIMT backend; 14.8 GiB of weights besides). So with the default ceiling every card here fits 32,768,
including the 32 GB RTX 5090 (~16 GiB free after the weights); the memory rule is what decides for a larger
`--max-step-tokens` (e.g. 131,072: ~13.6 GiB, about the 5090's whole 85% budget) or a smaller card. The packer
fills each worker's steps to that worker's own largest bucket, so mixed ladders work in one run.

### Multi-GPU: one process per GPU

Each GPU gets a worker *process* (`sysone bulk worker`, started as `scripts/gpu.sh --gpu N sysone bulk worker
--spec ...`), not an engine per device inside one process. Reasons:

- **The default-device hazard.** The engine and the libraries under it bind to the default device in several places:
  `GpuEngine::load` creates its context on device 0; `TileKernels::new` loads the SIMT module on cuda-async's
  thread-local `get_default_device()` and every SIMT launch (`kernels::simt::launch`) looks the module up there;
  cuTile's JIT and launch path and cuBLAS handles follow the current context; `PlatformProfile`/`require_idle_gpu`
  look at CUDA device 0. In-process multi-GPU would need each of these made device-explicit and audited (and kept
  so). With `CUDA_VISIBLE_DEVICES=N` every one of them *is* GPU N, with no code path to audit.
- **Locks.** gpu.sh already implements the per-device lock semantics (docs/x86/AGENT.md §4): each worker holds its
  GPU's lock for its lifetime; run under `scripts/gpu.sh --gpu 0,1,2`, the workers' nested gpu.sh reuse the held
  locks; a worker for a GPU whose lock is held elsewhere waits (and gets no work) instead of overlapping.
- **Isolation.** An OOM or a crash on one card ends that worker only: its queued steps go back to the window and the
  other GPUs finish them (`tests/bulk.rs`: a worker killed after 3 steps; every result is still written).
- **Cost.** A step crosses a pipe: token ids and segments in (4 bytes per token), only each answer row's slot logits
  out. At 32k tokens that is ~130 KB per 6 s step.

**Dynamic load balancing.** One queue (the window) feeds every worker; a worker gets its next step when it has
fewer than `--depth` (2) in flight. A faster GPU comes back sooner and takes more steps; a smaller card gets steps
sized to its own largest bucket, and requests too large for it stay in the window for a larger card. With three
simulated workers of which one is 4x slower with 4k steps, the slow one took 11% of the tokens and the others 45%
and 43% (`tests/bulk.rs`).

A thread worker (`--workers thread`) runs the engine in the coordinator's process instead: the mock engine (tests),
or one GPU when the coordinator itself runs under gpu.sh.

### Equality with the server

- Mock engine (`tests/bulk.rs::results_equal_the_servers_responses`): 394 jobs (the 298-case render corpus with its
  debias values, the 36 raw schema bodies incl. invalid JSON, 60 synthetic documents with debug and debias queries)
  plus bad lines, through three worker processes: every status and body equals the HTTP server's byte for byte apart
  from `qev.timing` (341 x 200, 49 x 422, 4 x 404).
- GPU (`tests/bulk_gpu.rs`, 112 jobs: the 52-request model corpus, the render corpus's error and limits cases, 40
  synthetic documents up to 6,000 tokens):
  - every request alone in its step (`--window 1`) equals the server's response to the same request sent alone,
    **byte for byte** (0 of 112 differ): same batch, same bucket, same CUDA graph;
  - packed (`--window 64`, 11 steps, 0.12% padding): statuses and every non-float field equal; probabilities differ
    by p99 4.9e-3, max 1.38e-2 (gate `[batch_invariance].dp_max` 1.82e-2), 880 of 1,874 bit-identical; debug slot
    logits max 5.2e-2 (gate 0.18); no choice flip. This is the engine's known batch variance (packing a request with
    others changes its rows slightly, docs/PHASE2A.md), not something bulk adds;
  - the same packed run twice, and on two engines sharing the GPU (`--gpus 0,0`, 3 + 8 steps), gives identical bytes.

## Results on GB10

2,000 synthetic documents (`sysone bulk gen --docs 2000`, seed 1: states 300..8,000 words, about one token each,
4..16 questions; 6,303,460 packed tokens, mean 3,152 and max 9,049 per request, 20,078 decisions), bf16 engine,
SIMT GDN backend, idle GPU under its lock, 2026-11-13. Both runs are throughput only: no warm-up excluded except
model loading (bulk: from the worker's HELLO; server: already loaded, from the first request).

| | bulk (`sysone bulk`, defaults: window 1,024, steps up to 32,768) | server + `loadgen --corpus --concurrency 256` (defaults: budget 16,384) |
|---|---:|---:|
| wall time | 1,245 s | 1,418 s |
| docs/s | **1.606** | 1.410 |
| decisions/s | **16.12** | 14.16 |
| packed tok/s | **5,062** | 4,444 |
| steps | 202 | 446 |
| padding fraction (bucket tokens not holding a request) | **0.20%** | 13.35% |
| GPU busy fraction (step time / wall) | 99.99% | 99.998% |
| latency p50 / p99 | n/a (bulk) | 186 s / 203 s (256 clients queued) |

Bulk is **13.9% faster** on the same GPU and the same requests. The difference is the padding: both keep the GPU
busy all the time, but 13% of the server's step tokens were padding (FIFO batches of whole requests stop short
of the 16,384 budget, and the bucket is a power of two), against 0.2% for bulk. `sysone bulk simulate` predicted
this from the token counts alone (CPU only, seconds): padding 13.8% (server FIFO, 16k) vs 0.2% (bulk, window 1,024,
32k), estimated GPU time 1,375 s vs 1,187 s (measured 1,418 s and 1,245 s; the estimate uses the bucket curve of
short-document steps, and long documents spend more in attention, so both measured runs are 3-5% slower).

Per-token speed is set by the forward itself (5,062 packed tok/s here vs 5,384 bucket tok/s for the 16k bench
step of ~600-token requests): the long documents of this mix cost more attention per token, so the first steps
(largest requests first) ran at ~4,500 tok/s and the last ones above 5,200.

The answers agree (`sysone bulk compare server.jsonl bulk.jsonl`): all 2,000 ids, statuses and every non-float
field equal; 1,076 of 2,000 bodies identical apart from timing; probabilities |dp| p99 5.6e-3, max 1.59e-2
(under `[batch_invariance].dp_max` 1.82e-2: the two sides pack the requests differently); 17 choice flips, all
with a top-1 margin below 0.0075.

Packing simulation on the same corpus (`sysone bulk simulate --max-step 16384,32768 --window 256,1024`):

| packing | steps | padding | est. GPU s |
|---|---:|---:|---:|
| server FIFO, budget 16,384, pow2 buckets | 448 | 13.83% | 1,375 |
| server FIFO, budget 32,768 | 207 | 7.07% | 1,274 |
| bulk, window 256, top 16,384 (pow2 / fine) | 390 | 0.97% / 0.95% | 1,196 |
| bulk, window 1,024, top 16,384 | 388 | 0.52% / 0.50% | 1,191 |
| bulk, window 256, top 32,768 | 198 / 197 | 0.36% / 0.35% | 1,188 |
| bulk, window 1,024, top 32,768 (default) | 210 / 211 | 0.20% / 0.17% | 1,187 |

A larger server budget halves the server's padding but keeps half of it; the window is what removes it. The fine
ladder changes only the last steps (a few hundredths of a percent).

**What concurrency does on this machine.** With one GPU, more concurrency does not raise throughput once the
engine is never idle: at 256 clients the server's engine was busy 99.998% of the time, so its throughput is the
forward's speed minus the padding its FIFO batches carry, and the extra clients only wait (p50 latency 186 s). The
way to more throughput on one GB10 is fuller steps (bulk: +13.9%), then a faster forward (GEMMs are 76% of it).

## Open items

1. On x86 (T7): bulk on each card alone and on all three at once, vs the server per card; the 5090's ladder from
   its memory; whether larger steps (65,536) help on the RTX PRO 6000.
2. Throughput per token depends on document length (attention grows with the prefix): the GB10 numbers above are
   for this corpus' length mix.
3. `allowed_token_mass` (`?debug=1`) is still absent with the GPU engine (as in the server).
