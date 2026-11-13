"""Record fp32 reference slot logits from OneJev's qev DecisionEngine (+ elephant-1's serve.patch).

The output JSON is what `cpu_ref_check` (src/bin/cpu_ref_check.rs) and `tests/cpu_ref.rs` compare against: per
request the prefix ids, and per branch the suffix ids, slot count and slot logits from the reference's own
sequential path (prefix forward, then one forward per suffix off a copy of the cache) and from its batched path
(forked cache, right-padded suffixes), so the reference's own path-to-path spread is recorded next to it.

Run (no GPU lock needed: a correctness run, ~36 GB of fp32 weights on the GPU):

    /home/velvet/Desktop/sysone-ref/venv/bin/python -I reference/hf_check/run_qev.py \
        --out reference/hf_check/out/qev_fp32.json

It copies OneJev@81ce62f to a temporary directory, applies serve.patch there and imports qev from that copy only,
so the installed / shared copies are never used or modified.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ONEJEV = Path("/home/velvet/Desktop/sysone-ref/OneJev")
MODEL = Path("/home/velvet/Desktop/hf_models/elephant-1")


def patched_qev(workdir: Path) -> Path:
    dst = workdir / "OneJev"
    shutil.copytree(ONEJEV, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    subprocess.run(["git", "apply", str(MODEL / "serve.patch")], cwd=dst, check=True)
    return dst


# Short requests: state ~50-300 tokens, 2-4 questions, every question kind.
REQUESTS = [
    {
        "name": "refund_ticket",
        "state": {"ticket": "My card was charged twice for the same order #4471 and support has not replied in 3 days. "
                            "I want the duplicate charge reversed today.",
                  "customer": {"tenure_months": 14, "vip": False, "plan": "basic"}},
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "charges, invoices and refunds", "technical": "bugs and outages",
                                  "sales": "pricing and upgrades"}},
            "mood": {"type": "score", "instructions": "How frustrated is the customer?",
                     "criteria": ["calm", "irritated", "angry"]},
        },
    },
    {
        "name": "incident_report",
        "state": ("At 02:14 UTC the primary database for the payments service failed over to the replica in "
                  "us-east-2 after a disk on the primary host reported uncorrectable read errors. The failover took "
                  "41 seconds, during which roughly 1,900 card authorisations were rejected with a generic error. "
                  "After the failover, p99 latency of the authorisation endpoint rose from 180 ms to 950 ms because "
                  "the replica's buffer pool was cold, and it returned to normal by 02:40 UTC. No data was lost: the "
                  "replication lag at the time of failure was under one second and every committed transaction was "
                  "present on the replica. The on-call engineer was paged at 02:15 and acknowledged at 02:22. A "
                  "follow-up ticket proposes keeping the replica warm with synthetic read traffic and adding an "
                  "alert on disk error counters, which were visible in the host metrics for six hours before the "
                  "failure."),
        "questions": {
            "severity": {"type": "choice", "instructions": "How severe was this incident?",
                         "criteria": {"sev1": "customer-facing outage with data loss or a security impact",
                                      "sev2": "customer-facing degradation or partial outage without data loss",
                                      "sev3": "internal impact only", "sev4": "no impact, near miss",
                                      "unclear": "the report does not allow a judgement"}},
            "data_loss": {"type": "noul", "instructions": "Was any committed data lost?",
                          "criteria": {"true": "some committed data was lost", "false": "no committed data was lost"}},
        },
    },
    {
        "name": "loan_application",
        "state": {"applicant": {"age": 34, "employment": "full-time nurse, 6 years at the same hospital",
                                "annual_income_usd": 78000, "monthly_debt_payments_usd": 650,
                                "credit_score": 712, "prior_defaults": 0, "bankruptcies": 0},
                  "request": {"product": "auto loan", "amount_usd": 24000, "term_months": 60,
                              "vehicle": "2023 Toyota Corolla, 18,000 miles", "down_payment_usd": 3000},
                  "history": [{"date": "2025-03-02", "event": "credit card limit increase approved"},
                              {"date": "2025-11-19", "event": "late payment, 12 days, on a store card"},
                              {"date": "2026-06-30", "event": "store card closed by the applicant"}],
                  "notes": "Applicant asked whether a co-signer would lower the rate. Income verified with two pay "
                           "stubs and a W-2. The dealer quote is valid until the end of the month."},
        "questions": {
            "decision": {"type": "choice", "instructions": "What should the lender do with this application?",
                         "criteria": {"approve": "approve at the standard rate",
                                      "approve_conditions": "approve with conditions such as a co-signer or a "
                                                            "larger down payment",
                                      "refer": "refer to a human underwriter",
                                      "decline": "decline the application"}},
            "risk": {"type": "score", "instructions": "How risky is this borrower?",
                     "criteria": ["very low", "low", "moderate", "high", "very high"]},
            "verified": {"type": "noul", "instructions": "Was the applicant's income verified?"},
            "cosigner": {"type": "noul", "instructions": "Did the applicant ask about a co-signer?"},
        },
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="sysone_hf_check_"))
    try:
        root = patched_qev(tmp)
        sys.path.insert(0, str(root))
        import torch
        import transformers

        import qev
        from qev import engine as qe
        from qev.prompt import render_question
        from qev.schema import SystemOneRequest

        if Path(qev.__file__).resolve().parent != (root / "qev").resolve():
            raise SystemExit(f"imported qev from {qev.__file__}, not the patched copy in {root}")

        # fp32 truth: no TF32 anywhere.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        t0 = time.perf_counter()
        eng = qe.DecisionEngine(args.model, device=args.device, dtype=args.dtype, fork_mode="sequential",
                                head_dtype=args.dtype)
        print(f"loaded in {time.perf_counter() - t0:.1f}s; attn={eng.model.config._attn_implementation} "
              f"readout={eng.info.extra}", flush=True)

        out = {
            "provenance": {
                "runtime": "OneJev @81ce62f + elephant-1 serve.patch (temporary copy)",
                "model": args.model, "dtype": args.dtype, "device": args.device, "tf32": False,
                "attn_implementation": eng.model.config._attn_implementation,
                "transformers": transformers.__version__, "torch": torch.__version__,
                "gdn_backend": "torch_chunk_gated_delta_rule (fla not installed)",
                "attention": eng.info.extra.get("attention"), "readout": eng.info.extra.get("readout"),
            },
            "slot_ids": list(eng.slot_ids),
            "requests": [],
        }
        for req in REQUESTS:
            request = SystemOneRequest(state=req["state"], questions=req["questions"])
            rendered = [render_question(qid, q, style=eng.style) for qid, q in request.questions.items()]
            prefix_ids, suffixes, _ = eng._split(request.state, rendered)
            t1 = time.perf_counter()
            seq = eng._run_sequential(request.state, rendered)
            t2 = time.perf_counter()
            bat = eng._run_batched(request.state, rendered)
            t3 = time.perf_counter()
            resp, _ = eng.decide(request)
            branches = []
            for r, suf, s, b in zip(rendered, suffixes, seq, bat):
                ps = qe.softmax(s.logits)
                pb = qe.softmax(b.logits)
                branches.append({"qid": r.qid, "kind": r.kind, "labels": r.labels, "n_slots": r.n_slots,
                                 "suffix_ids": suf, "logits_seq": s.logits, "logits_bat": b.logits,
                                 "allowed_mass_seq": s.allowed_mass,
                                 "max_dp_seq_vs_bat": max(abs(x - y) for x, y in zip(ps, pb))})
            out["requests"].append({"name": req["name"], "prefix_ids": prefix_ids, "branches": branches,
                                    "answers": json.loads(resp.model_dump_json())["answers"],
                                    "seconds": {"sequential": t2 - t1, "batched": t3 - t2}})
            print(f"{req['name']}: prefix {len(prefix_ids)} tokens, suffixes {[len(s) for s in suffixes]}, "
                  f"seq {t2 - t1:.2f}s bat {t3 - t2:.2f}s, max |dp| seq vs bat "
                  f"{max(b['max_dp_seq_vs_bat'] for b in branches):.2e}", flush=True)
            for b in branches:
                p = qe.softmax(b["logits_seq"])
                print(f"   {b['qid']:10s} {b['kind']:6s} p={[round(x, 4) for x in p]}", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
