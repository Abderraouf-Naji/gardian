"""
Isolate the reader prompt: same questions, same retrieved passages, different prompt.

Motivation
----------
On PubMedQA-labeled the readers score at or below the 55.2% majority-class
baseline, while the gold evidence is present in the context 87% of the time.
The bottleneck is therefore the reader, not retrieval. The dominant error is
*hedging*: gold-yes/no questions answered "maybe" (25-44% of all questions,
depending on the reader).

This script replays a completed end-to-end QA run with the passages frozen and
only the system prompt varied, so any change in accuracy is attributable to the
prompt alone.

    python scripts/ablate_reader_prompt.py \
        --qa-json results/qa_hybrid_bm25_faiss_Llama3-8B_n300.json \
        --dataset pubmedqa_labeled --system gardian \
        --prompts baseline decisive \
        --n 300 --out results/reader_prompt_ablation.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import statistics
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from src.common.passage_lookup import scan_corpus_for_pids  # noqa: E402
from src.common.repro import set_seed  # noqa: E402

VERDICT_RE = re.compile(r"(?i)\banswer\s*:\s*(yes|no|maybe)\b")
CITE_RE = re.compile(r"\[P(\d+)\]")


# --------------------------------------------------------------------------
# Prompt variants
# --------------------------------------------------------------------------
# "baseline" is the prompt used for the submitted results, reproduced verbatim
# from src/pipeline/rag/prompts.py so the A/B is honest.
BASELINE = """You are answering a biomedical research question using ONLY the passages below.
The passages are labeled [P1] through [P{k}] in the order they were ranked.

Instructions:
1. Read the question carefully.
2. Use all passages that are relevant to the question; ignore only those that are
   completely off-topic.
3. Write 2-3 sentences summarising the evidence. Cite every passage that supports
   each sentence using [P#] tags - a single sentence may carry multiple citations
   (e.g. "X has been shown [P2][P5].").
4. Last line MUST be exactly one of:
     Answer: yes
     Answer: no
     Answer: maybe

When to choose each answer:
- yes  : the passages collectively support the claim in the question.
- no   : the passages contradict the claim or consistently show no effect.
- maybe: the evidence is insufficient, mixed, or the key question is not
         addressed by the passages.

Do not say "I don't know". Do not cite a passage you did not use.
Do not fabricate [P#] tags."""

# "decisive" narrows the licence for "maybe" and states the label prior.
# The baseline definition ("insufficient, mixed, or not addressed") is broad
# enough that a cautious reader can justify "maybe" almost anywhere.
DECISIVE = """You are answering a biomedical research question using ONLY the passages below.
The passages are labeled [P1] through [P{k}] in the order they were ranked.

Instructions:
1. Read the question carefully.
2. Use every passage relevant to the question; ignore only those completely off-topic.
3. Write 2-4 sentences summarising the evidence. Cite EVERY passage that supports
   each sentence using [P#] tags - a sentence may carry several (e.g. "X was
   observed [P2][P5][P7]."). Cite each distinct supporting passage at least once.
4. Last line MUST be exactly one of:
     Answer: yes
     Answer: no
     Answer: maybe

Choosing the verdict - read this carefully:
- This question came from a study that reached a definite conclusion in about
  9 cases out of 10. "maybe" is the rare case, not the safe default.
- yes  : the passages, taken together, support the claim in the question.
- no   : the passages contradict the claim, or show no significant effect.
         A negative or null finding is "no". It is NOT "maybe".
- maybe: reserve this for genuine contradiction between passages, or for a
         claim the passages address but leave explicitly unresolved.

Do NOT answer "maybe" merely because the passages are incomplete, indirect, or do
not restate the question. Weigh the evidence you have and commit to the more
likely verdict. If the passages lean one way, answer that way.

Do not say "I don't know". Do not cite a passage you did not use.
Do not fabricate [P#] tags."""

# "calibrated" keeps the decisive framing but restores a principled route to
# "maybe", so the ~11% of genuinely unresolved questions are not sacrificed to
# suppress hedging. It also asks explicitly for full citation coverage, which is
# where citation_recall is lost (readers cite ~1.75 of ~3.36 gold passages).
CALIBRATED = """You are answering a biomedical research question using ONLY the passages below.
The passages are labeled [P1] through [P{k}] in the order they were ranked.

Step 1 - Evidence. Write 2-4 sentences summarising what the passages show.
Cite EVERY passage that supports a sentence with [P#] tags; a sentence may carry
several (e.g. "X was observed [P2][P5][P7]."). Every passage you relied on must
appear at least once.

Step 2 - Direction. State in one short sentence which way the evidence leans:
supporting the claim, contradicting it, or genuinely split.

Step 3 - Verdict. Last line MUST be exactly one of:
     Answer: yes
     Answer: no
     Answer: maybe

How to decide:
- yes  : the evidence leans toward supporting the claim.
- no   : the evidence leans against the claim, or shows no significant effect.
         A null or negative result is "no", not "maybe".
- maybe: ONLY when passages directly conflict with each other, or the study
         itself reports an inconclusive result. Roughly 1 question in 10.

Incomplete or indirect evidence is not a reason for "maybe". If the passages lean
one way at all, commit to that direction. Reserve "maybe" for true conflict.

Do not say "I don't know". Do not cite a passage you did not use.
Do not fabricate [P#] tags."""

PROMPTS = {"baseline": BASELINE, "decisive": DECISIVE, "calibrated": CALIBRATED}


def build_user_message(question: str, passages: List[str], max_chars: int) -> str:
    blocks = [
        f"[P{i + 1}] {p[:max_chars]}" for i, p in enumerate(passages)
    ]
    return "Passages:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n"


def score_answer(
    answer: str,
    gold_label: str,
    gold_pids: List[str],
    passage_ids: List[str],
) -> Dict[str, Any]:
    """Accuracy plus the grounding metrics, on one answer."""
    m = VERDICT_RE.search(answer or "")
    pred = m.group(1).lower() if m else None

    cited = []
    for tag in CITE_RE.findall(answer or ""):
        idx = int(tag) - 1
        if 0 <= idx < len(passage_ids) and idx not in cited:
            cited.append(idx)

    gold_set = set(gold_pids)
    cited_pids = [passage_ids[i] for i in cited]
    n_correct_cites = sum(1 for p in cited_pids if p in gold_set)

    ctx_hits = [p for p in passage_ids if p in gold_set]
    return {
        "pred": pred,
        "accuracy": 1.0 if pred == gold_label else 0.0,
        "n_citations": len(cited),
        "citation_precision": (n_correct_cites / len(cited_pids)) if cited_pids else None,
        "citation_recall": (n_correct_cites / len(gold_set)) if gold_set else None,
        "unsupported_claim_rate": (
            1.0 - n_correct_cites / len(cited_pids) if cited_pids else 0.0
        ),
        "gold_evidence_in_context_rate": 1.0 if ctx_hits else 0.0,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean_of(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    preds = collections.Counter(r["pred"] for r in rows)
    return {
        "n": len(rows),
        "accuracy": mean_of("accuracy"),
        "gold_evidence_in_context_rate": mean_of("gold_evidence_in_context_rate"),
        "citation_precision": mean_of("citation_precision"),
        "citation_recall": mean_of("citation_recall"),
        "unsupported_claim_rate": mean_of("unsupported_claim_rate"),
        "mean_citations_per_answer": mean_of("n_citations"),
        "predicted_distribution": {str(k): v for k, v in preds.most_common()},
        "unparseable_verdicts": preds.get(None, 0),
    }


def _write_results(path: str, results: Dict[str, Any]) -> None:
    """Write the report atomically, so an interrupted run leaves a valid file."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.replace(out)
    logger.info(f"Wrote {out} ({len(results.get('arms', {}))} arm(s) complete)")


def _resume_completed_arms(path: str) -> Dict[str, Any]:
    """Reload arms already finished in a previous run so they are not repeated."""
    out = pathlib.Path(path)
    if not out.is_file():
        return {}
    try:
        with open(out, "r", encoding="utf-8") as f:
            prior = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return prior.get("arms", {}) if isinstance(prior, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa-json", required=True, help="Completed QA run to replay.")
    ap.add_argument("--dataset", default="pubmedqa_labeled")
    ap.add_argument("--system", default="gardian")
    ap.add_argument("--eval-jsonl", default="data/pubmedqa_labeled_eval.jsonl")
    ap.add_argument("--corpus", default="data/corpus_pubmedqa_labeled.jsonl")
    ap.add_argument("--prompts", nargs="+", default=["baseline", "decisive"])
    ap.add_argument("--reader", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--max-chars-per-passage", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/reader_prompt_ablation.json")
    ap.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Re-run arms already present in --out instead of resuming past them.",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    # ---- replay source: questions, gold labels, and the FROZEN passage lists ----
    qa = json.load(open(args.qa_json, encoding="utf-8"))
    recs = qa["datasets"][args.dataset]["per_question"][args.system][: args.n]
    logger.info(f"Replaying {len(recs)} questions from {args.qa_json} [{args.system}]")

    meta: Dict[str, Dict[str, Any]] = {}
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                meta[r["id"]] = r

    want = {pid for r in recs for pid in r["passage_ids"]}
    logger.info(f"Resolving {len(want)} passage texts from {args.corpus}")
    texts = scan_corpus_for_pids(pathlib.Path(args.corpus), want)
    missing = len(want - set(texts))
    if missing:
        raise SystemExit(
            f"{missing} passage ids could not be resolved from {args.corpus}; "
            "the replay would not use the same context as the original run."
        )

    # ---- reader ----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading reader {args.reader}")
    tok = AutoTokenizer.from_pretrained(args.reader)
    model = AutoModelForCausalLM.from_pretrained(
        args.reader, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results: Dict[str, Any] = {
        "meta": {
            "qa_json": args.qa_json,
            "dataset": args.dataset,
            "system": args.system,
            "reader": args.reader,
            "n": len(recs),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "note": (
                "Passages are frozen from the source run; only the system prompt "
                "varies across arms, so accuracy differences are attributable to "
                "the prompt."
            ),
        },
        "arms": {},
    }

    completed = {} if args.rerun_completed else _resume_completed_arms(args.out)
    if completed:
        results["arms"].update(completed)
        logger.info(f"Resuming: {sorted(completed)} already complete, skipping")

    for arm in args.prompts:
        if arm not in PROMPTS:
            raise SystemExit(f"Unknown prompt {arm!r}; choose from {sorted(PROMPTS)}")
        if arm in results["arms"]:
            continue
        logger.info(f"=== arm: {arm} ===")
        rows: List[Dict[str, Any]] = []
        t0 = time.time()
        for i, rec in enumerate(recs):
            qid = rec["qid"]
            src = meta.get(qid)
            if src is None:
                continue
            pids = rec["passage_ids"]
            passages = [texts[p] for p in pids]
            system = PROMPTS[arm].format(k=len(passages))
            user = build_user_message(
                src["question"], passages, args.max_chars_per_passage
            )
            chat = tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok(chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tok.pad_token_id,
                )
            answer = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            scored = score_answer(
                answer,
                str(src.get("answer") or "").strip().lower(),
                list(src.get("gold_passage_ids") or []),
                pids,
            )
            scored.update({"qid": qid, "gold": str(src.get("answer") or "").lower(),
                           "answer": answer})
            rows.append(scored)
            if (i + 1) % 25 == 0:
                acc = statistics.fmean(r["accuracy"] for r in rows)
                logger.info(f"  [{arm}] {i + 1}/{len(recs)}  running acc={acc:.3f}")

        agg = aggregate(rows)
        agg["wall_clock_sec"] = round(time.time() - t0, 1)
        results["arms"][arm] = {"aggregate": agg, "per_question": rows}

        # Persist after EVERY arm. Each arm is ~30 minutes of GPU time; writing
        # only at the end means an interruption discards completed work.
        _write_results(args.out, results)

        logger.success(
            f"[{arm}] acc={agg['accuracy']:.4f}  "
            f"cit_recall={agg['citation_recall']:.4f}  "
            f"uns={agg['unsupported_claim_rate']:.4f}  "
            f"pred={agg['predicted_distribution']}"
        )

    _write_results(args.out, results)

    # ---- summary table ----
    print("\n" + "=" * 84)
    print(f"READER PROMPT ABLATION  |  {args.reader}  |  {args.dataset}  |  {args.system}")
    print("=" * 84)
    hdr = f"{'arm':<12}{'Acc':>8}{'Ctx':>8}{'CitP':>8}{'CitR':>8}{'Uns':>8}{'cites':>8}{'maybe%':>8}"
    print(hdr)
    print("-" * 84)
    for arm, block in results["arms"].items():
        a = block["aggregate"]
        maybe = 100 * a["predicted_distribution"].get("maybe", 0) / a["n"]
        print(f"{arm:<12}{a['accuracy']:>8.4f}{a['gold_evidence_in_context_rate']:>8.4f}"
              f"{a['citation_precision']:>8.4f}{a['citation_recall']:>8.4f}"
              f"{a['unsupported_claim_rate']:>8.4f}{a['mean_citations_per_answer']:>8.2f}"
              f"{maybe:>8.1f}")
    print("-" * 84)
    print("majority-class baseline on PubMedQA-labeled = 0.5520")


if __name__ == "__main__":
    main()
