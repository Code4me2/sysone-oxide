"""Interim model-side golden corpus (W1b), until W1a's tests/golden/dev/render corpus lands on main.

Same categories as DESIGN.md §9.1, smaller: every question type; option counts 2, 5, 26, 27, 255 (and 300, which
the schema rejects); debias 1 and 3; string, JSON, list, number and null states; Unicode; states from ~10 to ~16k
tokens; over-limit requests of each kind. Deterministic: rerunning rewrites the same requests.jsonl byte for byte.

    venv/bin/python -I reference/corpus_model/build_corpus.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qevref import BF16_MODEL  # noqa: E402

OUT = Path(__file__).resolve().parent / "requests.jsonl"

WORDS = ("account order refund charge invoice customer support ticket delivery package delayed damaged warehouse "
         "shipment carrier tracking payment card bank transfer subscription renewal cancel upgrade plan pricing "
         "discount coupon error crash login password reset email phone address contract clause liability "
         "termination notice period landlord tenant deposit repair inspection report revenue quarter forecast "
         "margin growth churn retention cohort survey feedback rating complaint escalation manager policy "
         "compliance audit risk fraud alert suspicious transaction device location session token server latency "
         "outage incident deploy rollback database query index cache memory disk network firewall certificate "
         "patient symptom dosage clinic appointment insurance claim coverage deductible referral diagnosis "
         "student course grade assignment deadline lecture exam tuition scholarship housing meal library").split()

UNICODE = ["Café déjà vu — naïve façade, Größe, smørrebrød.", "東京の配送センターで荷物が二日遅れています。",
           "客户说退款还没有到账，已经等了一周。", "Клиент просит вернуть деньги за двойное списание.",
           "العميل يطلب استرداد المبلغ بسبب خصم مكرر.", "고객이 이중 결제에 대해 환불을 요청했습니다.",
           "Emoji check: 📦🚚💳✅❌🙂 and ZWJ 👩‍💻 plus flags 🇯🇵🇧🇷.", "Combining: é ä ñ; NBSP: end.",
           "Math: ∀x∈ℝ, x² ≥ 0; ½ ≠ ⅓; tabs\tand\nnewlines."]


def sentence(rng: random.Random) -> str:
    n = rng.randint(6, 18)
    s = " ".join(rng.choice(WORDS) for _ in range(n))
    return s[0].upper() + s[1:] + rng.choice([".", ".", ".", "?", "!"])


def text_of_tokens(rng: random.Random, tok, target: int) -> str:
    """A state string of about `target` tokens (exact count checked with the model's tokenizer)."""
    paras, n = [], 0
    while n < target:
        para = " ".join(sentence(rng) for _ in range(rng.randint(3, 7)))
        paras.append(para)
        n += len(tok.encode(para + "\n\n", add_special_tokens=False))
    text = "\n\n".join(paras)
    ids = tok.encode(text, add_special_tokens=False)
    return tok.decode(ids[:target]).strip() if len(ids) > target else text


def opts(rng: random.Random, k: int, desc_words: int = 6, prefix: str = "opt") -> dict:
    out = {}
    while len(out) < k:
        name = f"{prefix}_{len(out)}_{rng.choice(WORDS)}"
        out[name] = " ".join(rng.choice(WORDS) for _ in range(rng.randint(max(1, desc_words // 2), desc_words)))
    return out


def noul(instr=None, true=None, false=None):
    q = {"type": "noul"}
    if instr is not None:
        q["instructions"] = instr
    if true is not None or false is not None:
        q["criteria"] = {"true": true, "false": false}
    return q


def choice(instr, criteria):
    return {"type": "choice", "instructions": instr, "criteria": criteria}


def score(instr, levels):
    return {"type": "score", "instructions": instr, "criteria": levels}


def mixed(rng: random.Random, n: int) -> dict:
    qs = {}
    for i in range(n):
        kind = ["noul", "choice", "score"][i % 3]
        if kind == "noul":
            qs[f"q{i}_noul"] = noul(f"Does the state mention a {rng.choice(WORDS)} problem?")
        elif kind == "choice":
            qs[f"q{i}_choice"] = choice("Which category best fits the state?", opts(rng, rng.choice([3, 4, 5, 6])))
        else:
            qs[f"q{i}_score"] = score("How severe is the issue described?",
                                      ["none", "minor", "moderate", "serious", "critical"][: rng.choice([3, 4, 5])])
    return qs


def build() -> list[dict]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(BF16_MODEL), local_files_only=True)
    rng = random.Random(20261113)
    items: list[dict] = []

    def add(cat, state, questions, debias=1, dump=False, expect=None):
        items.append({"id": f"m{len(items):03d}_{cat}", "category": cat, "debias": debias, "dump_hidden": dump,
                      "expect": expect or "ok", "request": {"state": state, "questions": questions}})

    ticket = "I was charged twice this month, please refund one of them."
    # --- question types, short string states
    add("noul_default", ticket, {"refund": noul("Is the customer asking for a refund?")}, dump=True)
    add("noul_criteria", ticket, {"r": noul("Is the tone polite?", "Polite or neutral wording", "Rude or hostile")})
    add("noul_no_instr", "The parcel arrived broken.", {"x": noul()})
    add("noul_entry_obj", ticket, {"x": noul({"ask": "refund?", "strict": True}, ["yes", "explicitly"], None)})
    add("choice_2", ticket, {"team": choice("Which team?", {"billing": "Charges", "technical": "Faults"})})
    add("choice_5", ticket, {"team": choice("Which team should handle this?", opts(rng, 5))}, dump=True)
    add("choice_5_debias3", ticket, {"team": choice("Which team should handle this?", opts(rng, 5))}, debias=3)
    add("choice_26", "Route this support ticket: " + ticket, {"route": choice("Route to which queue?", opts(rng, 26, 4))})
    add("choice_27", "Route this support ticket: " + ticket, {"route": choice("Route to which queue?", opts(rng, 27, 4))},
        dump=True)
    add("choice_27_debias3", ticket, {"route": choice("Pick a queue.", opts(rng, 27, 3))}, debias=3)
    add("choice_255", "A long routing table follows the request. " + ticket,
        {"route": choice("Which of the 255 queues fits?", opts(rng, 255, 3))}, dump=True)
    add("choice_null_desc", ticket, {"c": choice(None, {"billing": None, "tech": "", "other": None})})
    add("choice_entry_types", ticket, {"c": choice({"task": "classify", "lang": "en"},
                                                   {"a": {"desc": "billing", "w": 1}, "b": ["tech", "faults"],
                                                    "c": "other"})})
    add("score_3", ticket, {"u": score("How urgent is this?", ["Can wait", "Today", "Blocking now"])}, dump=True)
    add("score_5_debias3", ticket, {"u": score("How angry is the customer?",
                                               ["calm", "mild", "annoyed", "angry", "furious"])}, debias=3)
    add("score_10", ticket, {"u": score("Rate the priority from 0 to 9.", [f"priority {i}" for i in range(10)])})
    add("score_1", ticket, {"u": score("Only one level.", ["the only level"])})
    add("choice_1", ticket, {"c": choice("Only one option.", {"only": "the only option"})})
    add("readme_example", ticket, {
        "refund": noul("Is the customer asking for a refund?"),
        "team": choice("Which team should handle this?", {"billing": "Charges, refunds, invoices",
                                                          "technical": "App or site faults",
                                                          "fraud": "Suspected unauthorised use"}),
        "urgency": score("How urgent is this?", ["Can wait", "Today", "Blocking the customer now"])})
    # --- state types
    add("state_json", {"ticket": ticket, "customer": {"tenure_months": 14, "vip": False, "tags": ["billing", "eu"]}},
        mixed(rng, 3), dump=True)
    add("state_json_nested", {"events": [{"t": i, "kind": rng.choice(WORDS), "ok": i % 3 != 0} for i in range(40)],
                              "meta": {"source": "webhook", "version": 3}}, mixed(rng, 4), debias=3)
    add("state_list", [sentence(rng) for _ in range(12)], mixed(rng, 2))
    add("state_number", 42, {"x": noul("Is the state an even number?")})
    add("state_null", None, {"x": noul("Is there any state at all?")})
    add("state_empty_string", "", {"x": noul("Is the state empty?")})
    add("state_bool", True, {"x": noul("Is the state true?")})
    # --- Unicode
    add("unicode_state", "\n".join(UNICODE), mixed(rng, 3), dump=True)
    add("unicode_options", "配送が遅れています。Le colis est en retard.",
        {"lang": choice("Which language is used first?", {"日本語": "Japanese 🇯🇵", "français": "French — langue",
                                                          "Русский": "Russian", "العربية": "Arabic", "emoji 🙂": "😀"}),
         "sev": score("Severity ★", ["☆", "★", "★★"])}, debias=3)
    add("unicode_json_state", {"名前": "山田", "メモ": "返金希望 💳", "ñ": "ü"}, mixed(rng, 2))
    # --- multi-question requests
    for n, deb in ((3, 1), (5, 1), (8, 1), (8, 3), (24, 1)):
        add(f"multi_{n}q_debias{deb}", " ".join(sentence(rng) for _ in range(rng.randint(2, 6))), mixed(rng, n),
            debias=deb)
    # --- length sweep (approximate state tokens)
    for target, deb, dump in ((10, 1, False), (64, 1, False), (256, 1, False), (1024, 1, True), (1024, 3, False),
                              (2048, 1, False), (4096, 1, True), (4096, 3, False), (8192, 1, False),
                              (12288, 1, False), (16000, 1, True)):
        state = text_of_tokens(rng, tok, target)
        add(f"len{target}_debias{deb}", state, mixed(rng, 3), debias=deb, dump=dump)
    add("len4096_json", {"log": text_of_tokens(rng, tok, 4000), "user": "u-17"}, mixed(rng, 2))
    add("len2048_choice_27", text_of_tokens(rng, tok, 2048), {"route": choice("Route?", opts(rng, 27, 4))})
    # --- limits (no forward: rejected before the model runs)
    add("over_branch_limit", text_of_tokens(rng, tok, 33000), {"x": noul("Too long?")}, expect="422")
    big = {f"q{i}": choice("Pick one.", opts(rng, 100, 60)) for i in range(9)}
    add("over_request_limit", text_of_tokens(rng, tok, 20000), big, expect="422")
    add("schema_300_options", ticket, {"c": choice("Too many options.", opts(rng, 300, 2))}, expect="422")
    add("schema_11_levels", ticket, {"s": score("Too many levels.", [f"l{i}" for i in range(11)])}, expect="422")
    add("schema_no_questions", ticket, {}, expect="422")
    return items


def main() -> None:
    items = build()
    OUT.write_text("".join(json.dumps(it, ensure_ascii=False, separators=(",", ":")) + "\n" for it in items))
    dumps = sum(it["dump_hidden"] for it in items)
    print(f"wrote {len(items)} requests ({dumps} with hidden-state dumps) to {OUT}")


if __name__ == "__main__":
    main()
