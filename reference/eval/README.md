# reference/eval/: labeled eval set and the statistical end-to-end gates

DESIGN §9.2 gates NVFP4 statistically, because the QDQ reference itself moves by up to 0.2 per option when only
the batch shape changes (§3.1). This directory builds a labeled set from the NVFP4 checkpoint card's public
sources, runs it through the patched qev reference on several kernel-shape paths, derives the gate from how much
those reference paths disagree with each other, and scores any `/v1/systemone` server against it. The same
machinery gives a bf16 gate against the bf16 reference paths.

| File | What it does | Runs with |
|---|---|---|
| `fetch.sh` | Pinned downloads (hf CLI) of the three public sources into `sysone-ref/eval-data/`, sha256 check, parquet to JSONL (`convert.py`, in a separate pyarrow tool venv `sysone-ref/eval-venv`; the reference venv is not modified) | bash |
| `build_set.py` | Builds the set (fixed seed), checks eligibility with qev's own planner (`qev_shim.ShimEngine`), writes `sysone-ref/golden/dev/eval/requests.jsonl` and the committed manifest `tests/golden/dev/eval/set.json` | ref venv, CPU, ~4 min |
| `evalset.py` | Loads the set (sha256-checked) and turns any response into answer-level probability vectors | stdlib |
| `run_eval.py` | Runs the set through qev's FastAPI app on one path, time-boxed and resumable; `--merge` writes the committed run | ref venv, `scripts/gpu.sh` |
| `run_chunks.sh` | Loops `scripts/gpu.sh run_eval.py` (each call ≤ ~9 min including the model load) until a path is complete, then merges | bash |
| `stats.py` | Accuracy/CI/Brier/ECE per run, pairwise agreement and \|dp\|, the envelopes, the gate; `--write-tolerances` writes `[eval.*]` into `tests/tolerances.toml` | stdlib |
| `score.py` | Scores a live server (or a saved run) and applies the gate | stdlib |
| `serve_ref.py` | Serves the reference over HTTP (qev's app around the text `DecisionEngine`) for testing `score.py` | ref venv, `scripts/gpu.sh` |

```sh
bash reference/eval/fetch.sh
PY=/home/velvet/Desktop/sysone-ref/venv/bin/python
$PY -I reference/eval/build_set.py
bash reference/eval/run_chunks.sh nvfp4_perq     --nvfp4 --path per_question
bash reference/eval/run_chunks.sh nvfp4_perq_bN  --nvfp4 --path per_question_batched
bash reference/eval/run_chunks.sh nvfp4_batched  --nvfp4 --path batched
bash reference/eval/run_chunks.sh nvfp4_sequential --nvfp4 --path sequential
bash reference/eval/run_chunks.sh bf16_batched    --path batched
bash reference/eval/run_chunks.sh bf16_sequential --path sequential
bash reference/eval/run_chunks.sh bf16_perq       --path per_question
bash reference/eval/run_chunks.sh fp32_batched    --dtype float32 --path batched   # truth path of the bf16 tier
python3 -I reference/eval/stats.py --write-tolerances
# a candidate server (sysone, qev, ...):
python3 -I reference/eval/score.py --server http://127.0.0.1:8080 --tier nvfp4
```

## The set

The card evaluated **278 labeled tasks (128 BANKING77, 128 LEDGAR, 22 eligible LongBench-v2) plus 556 unlabeled
support questions**, rendered as Decision Index / qev requests; its records were not published. This set is
built the same way from the same public sources. **It is equivalent in composition, not identical**: the sampled
rows, the question wording and the support questions are ours, so accuracies are comparable to the card's only
loosely.

| | Source (pinned) | Records |
|---|---|---|
| BANKING77 | `PolyAI/banking77@1846984` test split (3,080 rows, 77 intents) | 128, `random.Random("20261113:banking77").sample` |
| LEDGAR | `coastalcph/lex_glue@c23fdff` `ledgar` test split (10,000 provisions, 100 headings) | 128, `random.Random("20261113:ledgar").sample` |
| LongBench-v2 | `zai-org/LongBench-v2@2b48e49` (= `THUDM/LongBench-v2`; 503 items) | all 22 eligible |

Every record is one request with three questions, so 278 labeled questions + 556 support questions:

- `task` (labeled): a `choice`.
  - BANKING77: instructions "Which intent does the customer's message express?", one option per intent, the
    dataset's label name verbatim (`activate_my_card`, `Refund_not_showing_up`, `reverted_card_payment?`), in
    label-id order, with no description (qev renders `A. activate_my_card`; 77 options use two-letter labels
    after Z). State: the customer message.
  - LEDGAR: "Which heading best describes this contract provision?", 100 headings verbatim in label-id order, no
    descriptions. State: the provision text.
  - LongBench-v2: the item's question as instructions; the four choice texts as option names in A–D order (so
    slot letter = original letter), no descriptions. State: the context. One item whose choices are not four
    distinct non-empty strings is skipped.
- two unlabeled support questions per record (a `noul` and a 3-level `score`, wording per source in
  `build_set.py::support_questions`). They carry no labels; they add branches whose probabilities the path
  comparisons use.
- `model` is `"jev-latest"` (qev's default alias, accepted whatever the served name); `debias=1`. No question is
  chunked (≤ 255 options). With debias 1 the answer probabilities are the branch softmax (T = 1).

**Eligibility.** Every record must fit qev's limits (32,768 per branch, 65,536 per request) *and* the NVFP4 card's
documented limit of 16,384 rendered tokens per evaluated question (prefix + that question's suffix, i.e. qev's
`longest`). The card limit is binding only for LongBench-v2: 111 items fit qev's limits, **exactly 22** fit 16,384,
which reproduces the card's "22 eligible" count and is why we read its limit as per-branch. The 89 items that only
qev's limits admit are listed in `set.json` (`longbench_v2_eligibility.qev_only_eligible_ids`); they are not in
the gate set (outside the NVFP4 checkpoint's documented range, and 16–32k-token per-question QDQ forwards would
cost hours of shared GPU time per path). The longest branch in the set is 16,012 tokens.

Files: `tests/golden/dev/eval/set.json` (manifest: per record source, index, gold, option count, token counts,
request sha256; the requests file sha256; source file sha256s and revisions); `requests.jsonl` (2.2 MB) stays in
`sysone-ref/golden/dev/eval/` and is verified against the manifest on every load.

## Reference runs

All through qev's FastAPI app (OneJev@81ce62f + serve.patch, text `DecisionEngine`, TF32 off, fp32 head,
`?debug=1&debias=1`), on this GB10 dev tier (torch 2.13.0+cu132, transformers 5.19.0, torch-fallback GDN, sdpa).
NVFP4 is compressed-tensors QDQ (`run_compressed=False`), as the card evaluated it.

| Run | Path | Kernel shape |
|---|---|---|
| `nvfp4_perq` | per question, batch 1, no cache (the card's path) | one `[1, P+S_i]` forward per question |
| `nvfp4_perq_bN` | per question, batched | one `[3, P+max S]` right-padded forward per record, no cache |
| `nvfp4_batched` | shared prefix, forked cache (qev `fork_mode=batched`) | `[1, P]` prefix, then `[3, max S]` suffix batch |
| `nvfp4_sequential` | shared prefix, one suffix at a time (qev `fork_mode=sequential`) | `[1, P]`, then `[1, S_i]` × 3 |
| `bf16_batched` | bf16, shared prefix batched (the production reference) | as above |
| `bf16_sequential` | bf16, shared prefix sequential | as above |
| `bf16_perq` | bf16, per question batch 1 | as above |
| `fp32_batched` | fp32 weights and activations, shared prefix batched (the truth path of the bf16 tier, DESIGN §9.2) | as `bf16_batched` |

Committed: `tests/golden/dev/eval/runs/<run>.json` (answer-level vectors rounded to 1e-6 as qev's response is,
per-record seconds and prefix tokens, provenance). Full slot logits and response JSON: the part files under
`sysone-ref/golden/dev/eval/runs/<run>/`, sha256 in `tests/golden/dev/eval/manifest.json`.

Runs were driven in ≤ ~9-minute `scripts/gpu.sh` chunks (2–3 chunks per path, model load 45–115 s each;
`fp32_batched` took 4 chunks, ~26 min of GPU, peak 34.4 GB; the
long LongBench-v2 records dominate: the last 15–39 records of a path take as long as the first ~250). Every path
answered all 278 records with HTTP 200.

## Results (dev tier, 2026-11-13; `tests/golden/dev/eval/report.json`)

Labeled-task accuracy (record-bootstrap 95% CI, B = 10,000), multi-class Brier, ECE (10 bins, top-1 confidence):

| Run | Overall | BANKING77 | LEDGAR | LongBench-v2 | Brier | ECE |
|---|---|---|---|---|---|---|
| `bf16_batched` | 227/278 81.7% [77.0, 86.0] | 116/128 90.6% | 102/128 79.7% | 9/22 40.9% | 0.2481 | 0.062 |
| `bf16_sequential` | 227/278 81.7% | 116/128 90.6% | 102/128 79.7% | 9/22 40.9% | 0.2475 | 0.063 |
| `bf16_perq` | 227/278 81.7% | 116/128 90.6% | 102/128 79.7% | 9/22 40.9% | 0.2475 | 0.062 |
| `fp32_batched` | 227/278 81.7% | 116/128 90.6% | 102/128 79.7% | 9/22 40.9% | 0.2477 | 0.061 |
| `nvfp4_perq` | 225/278 80.9% [76.3, 85.3] | 114/128 89.1% | 103/128 80.5% | 8/22 36.4% | 0.2651 | 0.086 |
| `nvfp4_perq_bN` | 227/278 81.7% | 115/128 89.8% | 104/128 81.2% | 8/22 36.4% | 0.2554 | 0.079 |
| `nvfp4_batched` | 227/278 81.7% | 116/128 90.6% | 102/128 79.7% | 9/22 40.9% | 0.2579 | 0.077 |
| `nvfp4_sequential` | 227/278 81.7% | 115/128 89.8% | 103/128 80.5% | 9/22 40.9% | 0.2586 | 0.070 |

The per-source CIs are wide (BANKING77 ±~5pp, LEDGAR ±~7pp, LongBench-v2 ±~20pp with 22 items): the set
resolves gross regressions, not fine quality differences.

**Reference envelopes** (pairwise over each tier's paths; all 834 questions, 24,134 options compared per pair):

| Tier | Pairs | Disagree (questions) | mean \|dp\| | p99 \|dp\| | max \|dp\| | acc diff CI |
|---|---|---|---|---|---|---|
| NVFP4 QDQ, 4 kernel shapes | 6 | 13–23 of 834 (1.6–2.8%) | 1.57–1.87e-3 | 0.043–0.052 | 0.28–0.45 | widest [-2.88, +1.44]pp |
| bf16, 3 paths | 3 | 0 | 1.36–1.47e-4 | 0.0036–0.0039 | 0.027–0.040 | [0, 0] |
| bf16 paths vs fp32 truth | 3 | 1 (margin < 0.02; 0 above it) | 1.23–1.33e-4 | 0.0032–0.0036 | 0.024–0.025 | [0, 0] |
| NVFP4 vs bf16 (info) | 4 | 14–24 (1.7–2.9%) | 2.1–2.2e-3 | 0.058–0.059 | 0.37–0.49 | dAcc -0.72 to 0pp |

On this set the bf16 reference's error vs fp32 truth is *smaller* than its own fork-path spread, so adding the fp32
path to the bf16 envelope (DESIGN §9.2's binding rule) does not change any bf16 limit; it does add fp32 as a
reference the candidate is gated against directly.

NVFP4 QDQ moves top-1 answers on ~2% of questions and single options by up to 0.45 just by changing the kernel
shape, about as much as it moves away from bf16; bf16 paths never disagree on a top-1 answer. NVFP4 costs no
measurable accuracy on this set (all CIs straddle 0) but is less calibrated (Brier +0.008 to +0.018 vs bf16).

## The gate (exact rule; thresholds in `tests/tolerances.toml` `[eval.nvfp4]`, `[eval.bf16]`)

For a tier with reference paths R (NVFP4: `nvfp4_perq`, `nvfp4_perq_bN`, `nvfp4_batched`, `nvfp4_sequential`;
bf16: `bf16_batched`, `bf16_sequential`, `bf16_perq` **and the fp32 truth path `fp32_batched`**), each metric's
envelope is its worst value over all pairs of R, and its limit is `max(factor × envelope, floor)` with `factor` =
the file's top-level 1.5, **rounded up** to 6 significant digits (never below the rule; `measured_*` and `floor_*`
are written at full precision and `src/tolerances.rs` checks `limit >= max(factor × measured, floor)` exactly).
A candidate **passes iff, against every path r in R**, on the records it answered:

1. top-1 disagreement ≤ `disagree`: the fraction of all questions (labeled + support) whose top-1 differs from
   r's **while r's top-1 margin exceeds `disagree_margin`**. NVFP4: `disagree_margin = 0` (every question counts).
   bf16: `disagree_margin = [e2e.bf16].top1_margin = 0.02`, DESIGN §9.2's end-to-end rule ("top choice agrees
   wherever the reference margin is > 0.02"), so the near-tie flips the design allows never count;
2. mean |dp| ≤ `dp_mean` and p99 |dp| (nearest rank) ≤ `dp_p99`, dp per option of every question;
3. lower bound of the paired record-bootstrap 95% CI of acc(C) − acc(r) ≥ −`acc_drop` (non-inferiority;
   the envelope is the largest |CI bound| among reference pairs);
4. upper bound of the paired record-bootstrap 95% CI of Brier(C) − Brier(r) ≤ `brier_rise` (same construction).

Floors only where a count is discrete:
- `disagree` ≥ 2 questions of 834;
- `acc_drop` ≥ **one labeled record on the CI scale**: −(CI lower bound) of a candidate that differs from the
  reference by one labeled loss, or by one loss plus one gain (net zero), whichever is lower. It is computed
  exactly (multinomial resampling counts, `stats.acc_floor`) and is 3/278 = 0.01079: one lost record is drawn up
  to 3 times in the 2.5% tail. The tail mass beyond it is 0.019 (one loss) and 0.008 (net zero), far enough from
  0.025 that a 10,000-resample bootstrap lands on the same bound. (The first version used 1/278 as a point floor
  on the CI bound, which rejected a single lost record; it also rounded limits to nearest, which put the
  2-question floor at 0.002398 < 2/834 and so allowed only 1.)

| Limit | NVFP4 | bf16 |
|---|---|---|
| `disagree_margin` | 0 | 0.02 (`[e2e.bf16].top1_margin`) |
| `disagree` | 0.041367 (measured 0.02758) | 0.00239809 (floor 2/834; measured 0 above margin) |
| `dp_mean` | 0.00281023 (measured 0.001873) | 0.000220301 (measured 0.0001469) |
| `dp_p99` | 0.0785671 (measured 0.05238) | 0.00582 (measured 0.00388) |
| `acc_drop` | 0.0431655 (measured 0.02878) | 0.0107914 (floor; measured 0) |
| `brier_rise` | 0.0318085 (measured 0.02121) | 0.00236366 (measured 0.001576) |

`dp_mean` and `dp_p99` pool every option of every question (24,134 options over 834 questions), so the 77- and
100-option BANKING77/LEDGAR questions dominate them and most of those options sit near 0: mean |dp| is diluted.
A large error on a few top options moves `dp_p99` and the top-1/accuracy/Brier checks more than `dp_mean`.

**Validation of the gate itself** (`report.json` → `validation`):

- *Leave-one-out*: rebuilding each tier's limits without one path and gating that path against the rest passes
  for all 4 NVFP4 and all 3 bf16 paths (fp32 truth stays in the bf16 reference set; it is the yardstick, not a
  kernel shape): a new kernel shape of the same reference is accepted. Because the bf16 paths never change a
  top-1 answer, the bf16 leave-one-out cannot exercise a flip; the minimal-flip probes below cover that. (A first version
  gated the point Brier difference on bf16 only; its leave-one-out failed for `bf16_batched`, because the other
  two bf16 paths had identical point Briers, i.e. an envelope of 0. Hence rule 4's CI form, for both tiers.)
- *Cross-tier*: NVFP4 runs gated as bf16 candidates fail every metric against every bf16 path; bf16 runs gated
  as NVFP4 candidates pass (bf16 sits inside the NVFP4 reference's own noise, so the NVFP4 gate is a
  "no worse than the reference's kernel-shape noise" gate, not a "matches NVFP4 bit-for-bit" one).
- *Minimal allowed deviations* (`validation.minimal_flips`; the top two probabilities of a question swapped
  on a reference run, gated against its own tier): the 2 smallest-margin support flips (the `disagree` floor),
  one labeled loss (the `acc_drop` floor; `ledgar-06992`, bf16 margin 0.052), one loss + one gain, and (bf16) all
  5 flips at reference margin ≤ 0.02 in every bf16/fp32 path: all PASS on both tiers. Just outside them, 3
  flips at margin > 0.03 fail the bf16 `disagree` check and 2 labeled losses fail `acc_ci_lo` and `brier_ci_hi`.
- *Cross-language*: `report.json` → `rust_fixture` holds every validation candidate's comparison values and
  failed checks; `src/tolerances.rs` (`eval_gate_matches_stats_py`) re-applies the committed limits with
  `EvalGate::failures` and must reach the same verdict on each.
- *Synthetic distortion* (a reference run with every answer re-tempered, p^(1/T) renormalized): NVFP4 gate
  passes T = 0.97, 1.03, 1.1 and fails T = 1.25 (`dp_mean`) and 1.5 (`dp_mean`, `dp_p99`); the bf16 gate fails
  every tested T including 0.97 and 1.03. So the NVFP4 gate tolerates roughly a ±10% logit-scale error and the
  bf16 gate about none; it does not catch rare large errors confined to < ~2% of questions (by construction:
  the reference itself makes those).

## Scoring a server

`score.py --server URL --tier {nvfp4,bf16}` POSTs every record (or a `--limit`/`--only`/`--sources` subset) to
`/v1/systemone?debias=1`, prints accuracies next to the reference runs restricted to the same records, and applies
the gate with the committed limits. Exit status: 0 full-set pass, 1 fail, 2 request errors, **4 subset pass**
(subset-calibrated limits; indicative only and deliberately not 0, so a script cannot mistake it for a gate pass). `smoke_score.sh` starts `serve_ref.py` (qev's FastAPI app around the
text engine), waits for `/health`, scores, and stops it, all under one GPU lock.

**Tested end to end** (2026-11-13) on a 28-record subset (every 10th record: 13 BANKING77, 13 LEDGAR, 2
LongBench-v2; `smoke/qev_nvfp4_batched_http_28.json`) against a qev NVFP4 server started by `smoke_score.sh`
(`--nvfp4 --path batched`, model load + 28 records in one ~2-minute GPU lock):

- the HTTP answers equal the committed in-process `nvfp4_batched` run exactly (dp = 0, 0 disagreements) on all 28
  records, so the HTTP path and the reference runner agree;
- `--tier nvfp4`: against the committed full-set limits the subset FAILS the two CI checks against three paths
  (one lost record on 28 gives a CI lower bound of −10.7pp vs −4.3pp allowed), which is why subset verdicts use
  subset-calibrated limits (same rule on the reference envelope over the same records; `score.py` docstring).
  With them: PASS (exit 0 at the time; such a run now exits 4);
- `--tier bf16` (the NVFP4 answers gated as a bf16 candidate, `--run-file`): FAIL on dp_mean, dp_p99 against all
  bf16 paths under both sets of limits (exit 1).

A first version computed disagreement as `1 − agree/n`, which made a candidate exactly at the 2-question floor
fail by one ulp; it is now `count/n`.

## sysone bf16 on the gate (A2)

`run_sysone.sh` drives the real sysone server (`target/release/sysone --engine bf16`, the default production
configuration: T buckets 256…65,536 with CUDA graphs, token budget 16,384, max wait 2 ms) over HTTP. Each chunk
is one `scripts/gpu.sh` job: start the server, wait for `/health` (it listens only after the engine has loaded and
warmed up), POST records with `run_sysone.py` for up to `CHUNK_SECONDS` (420), stop the server. Every record is
sent with its unchanged request body as `/v1/systemone?debug=1&debias=1` (the reference runs' query), one request
at a time. A non-200 answer is kept as an error record, never dropped (`score.py` then exits 2). The merge checks
that every part has the same engine build (git sha, binary sha256, model sha256) and that each answer vector
equals the softmax of the debug slot logits (< 2e-6), and writes `tests/golden/dev/eval/candidates/<name>.json`
in the reference runs' format (outside `runs/`, which `stats.py` reads as reference paths); the part files (full
responses) stay in `sysone-ref/golden/dev/eval/candidates/<name>/`, sha256 in `candidates/manifest.json`.

```sh
cargo build --release --bin sysone
bash reference/eval/run_sysone.sh                     # NAME=sysone_bf16
python3 -I reference/eval/score.py --run-file tests/golden/dev/eval/candidates/sysone_bf16.json --tier bf16
```

Protocol, fixed before the run: the full 278-record set, the committed `[eval.bf16]` limits unchanged, against all
four bf16-tier paths (`bf16_batched`, `bf16_sequential`, `bf16_perq`, `fp32_batched`).

**Result (2026-11-13, `candidates/sysone_bf16.json`; engine at 7198c65 = main bb5e6aa's `src/`, binary sha256
`919e1c1f…`, model.safetensors `682f0234…`, GB10 driver 580.178.04): GATE PASS**, 278/278 records HTTP 200, no
missing records, answers = softmax(debug slot logits) to 5e-7. `score.py` output: `candidates/sysone_bf16.score.txt`;
`diagnose.py` output: `candidates/sysone_bf16.diagnose.json`.

| vs | disagree (≤ 0.002398) | dp_mean (≤ 2.203e-4) | dp_p99 (≤ 0.00582) | acc CI lo (≥ −0.01079) | Brier CI hi (≤ 0.002364) | max \|dp\| |
|---|---|---|---|---|---|---|
| `bf16_batched` | 0.001199 (1/834) | 1.536e-4 | 0.004127 | 0 | −2.2e-5 | 0.034 |
| `bf16_sequential` | 0.001199 | 1.528e-4 | 0.004033 | 0 | 6.0e-4 | 0.032 |
| `bf16_perq` | 0.001199 | 1.431e-4 | 0.003702 | 0 | 4.2e-4 | 0.030 |
| `fp32_batched` | 0.001199 | 1.360e-4 | 0.003483 | 0 | 3.5e-4 | 0.029 |

Accuracy 228/278 = 82.0% [77.7, 86.3] (BANKING77 116/128, LEDGAR 103/128, LongBench-v2 9/22) vs 227/278 for every
bf16-tier reference path; dAcc +0.36pp, paired CI [0, +1.08]pp; Brier 0.2471, ECE 0.058. One top-1 change counts
against the gate: `ledgar-04056` (labeled task; reference margin 0.025–0.046 depending on the path, sysone margin
0.013), where sysone picks the gold heading the references miss. Against `fp32_batched` there is a second change,
`ledgar-04431` (support question, fp32 margin 0.0012: a near tie the gate exempts). It uses half of the
`disagree` limit (the 2-question floor); the dp metrics use 61–71% of theirs.

Throughput (information; one request at a time, server batching idle): 278 records in 181.4 s of request time
(sum of per-request seconds; chunk wall 181.5 s after an 18.1 s server start incl. 13.4 s JIT warm-up), 550,538
rendered tokens, 3,034 tok/s; BANKING77 37.2 s, LEDGAR 53.0 s, LongBench-v2 91.2 s (22 records, 3,279 tok/s).
The `bf16_batched` reference (HF eager) took 492 s for the same records.
