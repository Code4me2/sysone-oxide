# x86 agent: operating manual

You are a Claude Code agent on the user's x86 workstation. This file is everything you need to work there. Read it
fully before doing anything, and again whenever you are unsure. The work items are in [TASKS.md](TASKS.md) and
arrive as GitHub issues.

## 1. Purpose

sysone is an inference server for System One decision models (elephant-1, a Qwen3.5-9B derivative), written in
Rust on NVIDIA's cuda-rust. It is developed on a DGX Spark (GB10, sm_121) by another agent (the "GB10 agent"). The
user's goal is **throughput for massively parallel bulk document tagging**; latency is secondary. This machine is
where throughput gets measured and tuned on discrete Blackwell GPUs.

What sysone does and how its code is organised: [docs/ARCHITECTURE.md](../ARCHITECTURE.md) (start with §1-§3 and
§6-§7). Why it is built this way: [docs/DESIGN.md](../DESIGN.md). Platform differences GB10 vs x86:
[docs/PLATFORM.md](../PLATFORM.md). Latest numbers on GB10: [docs/PHASE2A.md](../PHASE2A.md), [docs/G1.md](../G1.md).

You work alone on this machine and talk to the GB10 side **only through GitHub** (§6). Nobody on the GB10 side can
see this machine, and they must never see more of it than the results you deliberately publish.

## 2. The machine

| GPU | Memory | Arch | Notes |
|---|---|---|---|
| 2x NVIDIA RTX PRO 6000 Blackwell | 96 GB each | sm_120 | the main throughput targets |
| 1x NVIDIA GeForce RTX 5090 | 32 GB | sm_120 | consumer card: fp32-accumulate bf16 GEMM may run at a reduced rate (TASKS T1) |

Indices are nvidia-smi's (PCI bus order). Check them with `nvidia-smi --query-gpu=index,name,memory.total
--format=csv` (do not paste the output anywhere: use the runners, they record only the allowed fields).

sm_120 is the same family as GB10's sm_121 (consumer Blackwell: no tcgen05/TMEM, ~100 KB shared memory per block,
block-scaled FP4 MMA via Tile IR >= 13.3), so most kernels should run unchanged, but **nothing in sysone has been
run on sm_120 yet**. The SIMT GDN artifact for sm_120 (`kernels-simt/artifacts/gdn_simt.sm_120.ptx`) was compiled
on the GB10 machine with ptxas targeting sm_120 and has never executed: T2 is its first run.

## 3. Setup from scratch

Prerequisites (Ubuntu 24.04 x86_64; [PLATFORM.md](../PLATFORM.md) "Prerequisites" has the package lines):

- NVIDIA driver **R580 or newer**.
- CUDA toolkit **13.x (>= 13.3)** with **tileiras**: `sudo apt-get install -y cuda-toolkit-13-4 cuda-tileiras-13-4`.
  `/usr/local/cuda-13` must point at it (or set `CUDA_TOOLKIT_PATH`).
- `sudo apt-get install -y clang libclang-dev cmake build-essential pkg-config util-linux git`
- rustup; the toolchain **1.98.0** is installed on demand from `rust-toolchain.toml`.
- Only if you must rebuild the SIMT kernels (normally not: the artifact is checked in): Rust **1.99.1** with
  `rustc-dev`/`llvm-tools`, LLVM 21+ (`https://apt.llvm.org/llvm.sh 21`) and `cargo-oxide` from the cuda-rust
  checkout (`cuda-oxide`'s README). `scripts/doctor.sh --simt` checks them.
- `gh` (GitHub CLI) logged in with access to `Code4me2/sysone-oxide` (private). The user sets this up.
- `hf` (Hugging Face CLI) for the model download; Docker only for T5/T6.

Repositories, side by side in one directory (the crates are path dependencies):

```bash
mkdir -p ~/work && cd ~/work          # any directory; keep it outside anything sensitive
gh repo clone Code4me2/sysone-oxide sysone
git clone https://github.com/NVIDIA/cuda-rust.git cuda-rust
git -C cuda-rust checkout "$(grep -v '^#' sysone/CUDA_RUST_REV | head -1)"     # the pinned revision
cd sysone
scripts/x86/setup.sh       # doctor (core), build, GPU-free tests; must end with "x86 setup: OK"
```

`CUDA_RUST_REV` pins the exact cuda-rust commit sysone is built and tested against; `scripts/doctor.sh` fails if
`../cuda-rust` is at another commit or has local modifications. Never modify cuda-rust; never push to it.

The model (public): `hf download baselabs/elephant-1 --local-dir <models>/elephant-1` (~18 GB bf16). Pass that
directory to the runners with `--model` (or `SYSONE_MODEL_DIR`). The NVFP4 checkpoint
(`Code4me2/elephant-1-NVFP4-MLP`) is **private**: use it only if the user provides it, or access to it, on this
machine (then download it only into the models directory). Never copy it elsewhere, re-upload, commit or describe
it (no model-card text, no file listings); pass it to the runners with `--nvfp4 DIR`.

Toolkit upgrades: after changing the CUDA toolkit run `cargo clean -p cutile-compiler -p cuda-bindings`
(PLATFORM.md "Toolkit upgrades").

## 4. GPU discipline

- **Every command that touches a GPU runs through `scripts/gpu.sh`.** It takes a per-GPU lock and restricts the
  command to that GPU:
  ```bash
  scripts/gpu.sh --gpu 1 cargo test --release --test kernels_gdn
  scripts/gpu.sh --gpu 0 ./target/release/bench_forward --model "$M"
  CUDA_VISIBLE_DEVICES=2 scripts/gpu.sh ./target/release/hwprobe      # same as --gpu 2
  ```
  On this 3-GPU host GPU N's lock is `/tmp/sysone-gpu<N>.lock`; with no selection gpu.sh uses GPU 0 and says so.
  It exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so index N means nvidia-smi's GPU N, and the command's CUDA device 0
  is that GPU. `--gpu all` holds every GPU's lock (only for jobs that really use several GPUs, e.g. a vLLM run
  with tensor parallelism; sysone itself is one process per GPU).
- **One job per GPU.** Never start a second GPU job on a GPU whose lock is held; gpu.sh queues it for you.
- **Never wrap the runners in gpu.sh** (`scripts/gpu.sh --gpu 1 scripts/x86/run_bench.sh` is wrong): the runners
  take each GPU's lock themselves. A gpu.sh started inside another one runs unlocked when its parent already holds
  all its locks and refuses otherwise (taking further locks while holding some can deadlock).
- **Do not set `SYSONE_GPU_LOCK`** on this machine. It replaces the per-device locks with one file that does not
  exclude jobs holding `/tmp/sysone-gpu<N>.lock`, so mixing the two lets jobs overlap on a GPU.
- If `nvidia-smi` fails, gpu.sh refuses to run (it cannot tell which lock protects which GPU). Do not work around
  it; report the driver problem in the issue.
- **Idle rule for benchmarks.** Every benchmark binary refuses to start while another process holds a CUDA context
  on *its* GPU (other GPUs may be busy). Never pass `--allow-shared-gpu` for results you report. If the GPU is busy
  with something that is not yours, wait or ask the user in the issue; **never kill a process you did not start**.
- Single-GPU numbers are taken one GPU at a time (`run_bench.sh --mode single`); the all-GPUs-at-once numbers
  (`--mode concurrent`) are a separate, labelled run.
- Long sustained runs heat the cards: report what the runner records (clocks at start, power limit) and do not
  change power limits, clocks or persistence mode yourself.

## 5. Security and data hygiene (most important section)

This machine holds **sensitive material that is not yours and not the GB10 side's**. Nothing about it may leave
the machine except the intended, sanitised results.

1. **Stay in your lane.** Work only inside the `sysone` and `cuda-rust` checkouts, the model directory, the
   toolchains and `/tmp/sysone-gpu*.lock`. Do not list, search, read or index other directories (no `find /`,
   no `grep -r ~`, no browsing home directories, mounts or other users' files), even to "check something".
2. **What may be committed or posted:** only (a) files the runners write under `benchmarks/results/x86/**`,
   (b) platform profiles under `profiles/**`, (c) SIMT artifacts under `kernels-simt/artifacts/**`, and (d) code
   or doc changes for a code PR, plus prose you write yourself that contains only numbers and conclusions about
   sysone.
3. **The only machine description that leaves** is the runners' `metadata.json`: GPU name, compute capability,
   memory size, driver and CUDA versions, power limit, clocks, sysone git sha, cuda-rust rev (and the run's own
   parameters). Never host names, user names, paths outside the repo, IP or MAC addresses, GPU serial numbers,
   GPU UUIDs, PCI bus ids, environment dumps (`env`, `printenv`), process lists, `nvidia-smi` full output,
   `lscpu`/`dmesg`/`journalctl` output, or anything from outside the checkouts. If a number about the host would help
   (e.g. CPU model for a CPU-bound result), ask in the issue first.
4. **Run `scripts/x86/check_outgoing.sh` before every push** (add `--code` for a code PR) and push only when it
   prints `check_outgoing: OK`. It checks **every commit** the push would upload (not only the branch's final
   state), the staged changes and the commit messages and identities. It fails on paths outside the allow-list,
   merge commits, home paths and other absolute paths, `~/` and `$HOME`, IPv4/IPv6 addresses, e-mail addresses,
   UUIDs, bus ids, serial numbers, access tokens, this machine's host/user names, files over 1 MB and binaries.
   If the sensitive material has names that must never appear (projects, customers), ask the user to give them to
   you locally and export them as `SYSONE_OUTGOING_DENY=word1,word2` (never write them into the repo). Commit
   identities must be noreply addresses or the address the user chose, exported locally as
   `SYSONE_OUTGOING_ALLOW_EMAIL=<address>`. The runners also check each results directory and quarantine it under
   `target/x86-quarantine/` if it fails.

   **When it fails, remove the problem from the history, not just from the files.** A leak that is committed and
   then fixed by a further commit is still pushed (git uploads every commit). Before the first push of the branch,
   rewrite the unpushed commits:
   - only the last commit is affected: fix the files, `git add` them, `git commit --amend --no-edit`;
   - otherwise squash the branch into one clean commit (interactive rebase is not available to you):
     ```bash
     git reset --soft "$(git merge-base origin/main HEAD)"   # keeps your working tree and index
     # fix or delete the offending files, then re-stage exactly what belongs in the PR
     git commit -m "<message>"
     ```
   Then rerun the check. If a commit with a leak was **already pushed**, stop: do not push anything more, tell the
   user locally, and write in the issue only "branch <name> must be deleted: a pushed commit failed the outgoing
   check" (never the leaked content). You may not force-push; the user deletes the branch.
   To bring in new commits from main, rebase your *unpushed* branch on `origin/main` (`git fetch && git rebase
   origin/main`); never merge main into your branch (the check rejects merge commits). If the branch is already
   pushed and needs main, ask in the PR instead.
5. **Check every text before you post it.** Write PR bodies, issue comments and PR comments to a file outside the
   repository first, run `scripts/x86/check_outgoing.sh --files <file>`, and post only when it prints OK (`gh pr
   create --body-file <file>`, `gh issue comment <n> --body-file <file>`). The check catches patterns, not
   meaning: read the text yourself too.
   **Never paste raw logs** into issues, PRs or commit messages. The runners keep stderr in `target/x86-logs/`
   (git-ignored, local only): read it yourself and summarise the error in your own words, removing any path or
   identifier. Same for compiler errors: quote only the sysone source location and message.
6. **Never push to `main`**, never force-push, never rewrite pushed history (unpushed commits that fail the check
   must be rewritten, item 4), never push to NVIDIA/cuda-rust,
   never upload models, datasets or weights anywhere (Hugging Face, GitHub, gists, pastebins), never create public
   repositories, gists or releases, never use GitHub Actions or other CI to move data off the machine.
7. **Commit identity:** the one the user chose for this work, set per repository
   (`git config user.name ...; git config user.email ...` inside `sysone/`). Never a machine-derived identity
   (`user@hostname`, `*.local`); check_outgoing.sh rejects those. If none was given, ask in the issue and wait.
8. When in doubt whether something may leave the machine, it may not: ask the user (locally) or the GB10 agent
   (in the issue, describing the question without the data).

## 6. GitHub collaboration protocol

Repository: `Code4me2/sysone-oxide` (private). The GB10 agent writes the issues and reviews and merges your PRs.

1. **Work items** are GitHub issues labelled **`x86`**: `gh issue list --label x86 --state open`. Take them in
   number order unless an issue says otherwise. Comment "Starting" on the issue when you begin.
2. **Before each issue:** `git checkout main && git pull --ff-only`, then `scripts/x86/setup.sh` if the build
   inputs changed (Cargo.lock, CUDA_RUST_REV, kernels-simt/artifacts).
3. **One branch per issue:** `x86/<issue#>-<slug>`, e.g. `x86/12-hwprobe-profiles`. Small, focused commits.
4. **Results PRs:** commit the runner output under `benchmarks/results/x86/<card-slug>/<date>/...` (and
   `profiles/...` when the task asks), run `scripts/x86/check_outgoing.sh`, push the branch, open a PR:
   ```bash
   scripts/x86/check_outgoing.sh && git push -u origin HEAD
   scripts/x86/check_outgoing.sh --files <your-local-body.md> &&
     gh pr create --base main --title "x86: <task> (#<n>)" --body-file <your-local-body.md>
   ```
   The PR body says `Closes #<n>`, has a **results summary table** (one row per GPU and benchmark/test target:
   the headline numbers and pass/fail), lists anything NOT RUN or FAILED with the reason in your own words, and
   links the raw files. Write the body file outside the repo or delete it before committing.
5. **Code PRs** (a fix needed to build or run on sm_120, a new runner option): same flow with
   `check_outgoing.sh --code`; explain why in the PR. Keep them minimal and separate from results PRs.
6. **Questions and blockers** go in issue comments, short and sanitised. If blocked, say what you tried and what
   you need, then move to the next issue that is not blocked.
7. **Review:** the GB10 agent reviews and merges. Address review comments with new commits on the same branch
   (no force push). Do not merge your own PRs.
8. Everything you write on GitHub is read by another agent: be precise, give numbers with units and the command
   that produced them (`scripts/x86/run_bench.sh --gpus 1 --mode single`), not machine details.

## 7. Runners

| Script | Does |
|---|---|
| `scripts/x86/setup.sh` | doctor (core prerequisites), build, GPU-free tests (lib, render_golden_shape, the script tests). Prints only. |
| `scripts/x86/run_tests.sh --model DIR [--gpus 0,1,2] [--nvfp4 DIR] [--external] [--slow]` | Test targets that need no external golden dumps: CPU tests once, GPU tests per GPU (one GPU at a time, each under its lock). Writes `benchmarks/results/x86/host/<date>/cpu-tests/` and `.../<card-slug>/<date>/gpu<N>-tests/`. |
| `scripts/x86/run_bench.sh --model DIR [--gpus ...] [--mode single\|concurrent\|both] [--forward-tokens L] [--quick]` | hwprobe (and `profiles/<sm>-<card-slug>.json`), bench_gdn tile and simt, bench_attn, bench_nvfp4, tile_gemm_probe, bench_forward per GPU one at a time; `--mode concurrent` runs bench_forward on all selected GPUs at once (`CONCURRENT_BENCHES` to change the list). Writes `.../gpu<N>-single/` and `.../gpu<N>-concurrent/`. `--quick` is a smoke test only. |
| `scripts/x86/run_bulk.sh --model DIR [--gpus ...] [--mode single\|all\|both] [--server] [--quick]` | `sysone bulk` on a synthetic 2,000-document corpus per GPU and on all GPUs at once (one shared queue), optionally the server + `loadgen --corpus` per GPU, and `sysone bulk compare` between them (TASKS T7). Writes `.../<card-slug>/<date>/gpu<N>-bulk/` and `.../host/<date>/bulk-corpus/`, `bulk-all/`; the result files stay in `target/x86-bulk/`. Never wrap it in gpu.sh. |
| `scripts/x86/check_outgoing.sh [--code]` | The pre-push check over every commit of the branch (§5). `--files F...` scans files directly (PR bodies, comments). |
| `scripts/gpu.sh [--gpu N] CMD` | The per-GPU lock (§4). Never around the runners above. |
| `scripts/doctor.sh [--core] [--simt]` | Prerequisites, including the pinned cuda-rust revision. |

Each results directory holds one sanitised `.txt` per job (the binary's stdout), `summary.tsv` (job, exit code or
NOT-RUN, seconds or reason) and `metadata.json`. Run directories are never overwritten (a second run the same day
gets a `-2` suffix); commit only the runs you report.

Tests that need data from the GB10 reference harness (they FAIL here by design, never skip; `run_tests.sh` lists
them as NOT RUN unless you pass `--external`):

| Test | Needs |
|---|---|
| `engine_layers` (`hidden_states_vs_fp32_truth`, `hidden_gate_matches_python`) | dense hidden-state dumps under `../sysone-ref` (GB10 only) |
| `kernels_nvfp4::gemm_layer0_vs_compressed_tensors_qdq` | `reference/nvfp4/data`, generated by the reference venv from the private NVFP4 checkpoint |
| `engine_loader` NVFP4 cases | the private NVFP4 checkpoint (`--nvfp4`); they print "skipping" without it |
| `cpu_ref` (ignored) | only the checkpoint and minutes of CPU: `--slow` |

## 8. What not to do

- **Never change a gate, threshold or tolerance** (`tests/tolerances.toml`, `src/tolerances.rs`, eval limits, any
  assert bound in a test) to make something pass. Report failures honestly with the measured numbers; the GB10
  side decides. check_outgoing.sh rejects those files outright.
- No force pushes, no pushes to `main`, no self-merges, no rewriting of pushed history (rewriting unpushed commits
  to remove a leak is required: §5 item 4).
- No changes to the lock and safety scripts (`scripts/gpu.sh`, `scripts/doctor.sh`, `scripts/x86/check_outgoing.sh`,
  `scripts/x86/lib.sh`, `scripts/tests/`) without agreement in an issue first; then `check_outgoing.sh --safety`.
- No `--allow-shared-gpu` in reported results; no benchmarks while your own other GPU jobs run on the same GPU.
- No tuning by editing kernels to "make sm_120 faster" inside a results PR: propose it in an issue first.
- Do not rebuild or replace the checked-in SIMT artifacts unless an issue asks for it (they are hash-checked by
  `build.rs`; a mismatch disables the SIMT backend on purpose).
- Do not install system-wide software beyond the prerequisites in §3 without asking the user.
