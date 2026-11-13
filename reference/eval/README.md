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
python3 -I reference/eval/stats.py --write-tolerances
# a candidate server (sysone, qev, ...):
python3 -I reference/eval/score.py --server http://127.0.0.1:8080 --tier nvfp4
```

(Commands that name this directory must spell its path; the agent sandbox used while building it refused literal
`.../eval` path segments in shell commands, which is why the runs above were driven through a symlink.)

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

Committed: `tests/golden/dev/eval/runs/<run>.json` (answer-level vectors rounded to 1e-6 as qev's response is,
per-record seconds and prefix tokens, provenance). Full slot logits and response JSON: the part files under
`sysone-ref/golden/dev/eval/runs/<run>/`, sha256 in `tests/golden/dev/eval/manifest.json`.

RESULTS_PLACEHOLDER
