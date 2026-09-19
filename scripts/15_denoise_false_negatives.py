"""
Remove false negatives from rank JSONL using a cross-encoder.

Why
---
MedMCQA's gold passage is the *explanation authored for that one question*
(``pid = mcq_<id>_explanation``), so questions and passages are in 1:1
correspondence. Medical exam questions are highly redundant, so many
explanations legitimately answer many questions -- but exactly one is labelled
gold and every other is labelled negative. Measured with
``cross-encoder/ms-marco-MiniLM-L-6-v2`` over 150 test queries:

    collection   gold/q   top-10 negatives scoring >= gold   median(gold - best_neg)
    medmcqa       1.00                 25.0%                        -0.998
    nfcorpus     10.37                  9.7%                        +1.460
    scifact       1.14                 14.2%                        +0.958

One in four MedMCQA "hard negatives" is judged at least as relevant as the
gold, and the median margin is *negative*. SciFact has the same sparsity
(1.14 gold/query) with a +0.958 margin, so this is a label defect, not a
density artifact. With ``hard_negative_fraction: 0.85`` and
``hard_negative_top_n: 120`` the trainer draws most negatives from exactly
where those false negatives are.

What this does
--------------
For each query, scores the gold passages and the candidate negatives with a
cross-encoder and drops negatives scoring at or above the best gold (optionally
with a margin). Output is a new rank JSONL; the input is never modified.

Why the stored features are NOT recomputed
------------------------------------------
Dimensions 3/4/5 of each branch (min-max, z-score, rank) and ``pool_stats`` are
pool-relative, computed at generation time over the full pool. Recomputing them
over the truncated pool would make training see normalisations that can never
occur at evaluation time, where nothing is dropped. Leaving them untouched
keeps train and eval on the same footing: this is a label fix, not a pool
change. The listwise sampler already subsamples pools to
``listwise_group_size``, so dropping rows only changes what it may pick.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, ".")

import numpy as np
from loguru import logger


def load_corpus_text(corpus_path: pathlib.Path) -> Dict[str, str]:
    """pid -> text for a whole corpus (the biomedical corpora here are small)."""
    out: Dict[str, str] = {}
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("id")
            if isinstance(pid, str):
                title = str(obj.get("title") or "").strip()
                text = str(obj.get("text") or "")
                out[pid] = f"{title}. {text}".strip(". ") if title else text
    logger.info(f"Loaded {len(out):,} passages from {corpus_path.name}")
    return out


def iter_query_groups(path: pathlib.Path):
    """Yield one query's contiguous block of records at a time."""
    cur = None
    group: List[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("qid") != cur:
                if group:
                    yield cur, group
                cur = rec.get("qid")
                group = []
            group.append(rec)
    if group:
        yield cur, group


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank-data", required=True, help="input rank JSONL")
    ap.add_argument("--corpus", required=True, help="corpus JSONL for passage text")
    ap.add_argument("--out", default=None,
                    help="output rank JSONL (default: <input>_denoised.jsonl)")
    ap.add_argument("--report", default=None, help="write a JSON report here")
    ap.add_argument("--mode", choices=("drop", "relabel"), default="drop",
                    help="drop: remove false negatives from the pool. relabel: keep "
                         "them and promote them to a positive grade, which also "
                         "raises label density (MedMCQA 0.84 -> ~9 positives/query) "
                         "and un-flattens the alpha surface. Relabelling asserts the "
                         "cross-encoder's opinion as a training label, so validate on "
                         "real graded qrels, never on the denoised collection itself.")
    ap.add_argument("--gold-grade", type=int, default=2,
                    help="relabel mode: grade written for the original gold passages.")
    ap.add_argument("--promoted-grade", type=int, default=1,
                    help="relabel mode: grade written for promoted false negatives. "
                         "Listwise gain is 2^label - 1, so the default 2/1 leaves the "
                         "original gold worth 3x a promoted passage rather than "
                         "treating them as interchangeable.")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="drop a negative when ce >= best_gold_ce - margin. 0.0 drops "
                         "only negatives at or above gold; a positive margin is stricter.")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="pilot on the first N queries")
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure what would be dropped without writing the output")
    args = ap.parse_args()

    rank_path = pathlib.Path(args.rank_data)
    if not rank_path.is_file():
        raise SystemExit(f"missing rank data: {rank_path}")
    out_path = pathlib.Path(args.out) if args.out else rank_path.with_name(
        rank_path.stem + "_denoised.jsonl")

    text = load_corpus_text(pathlib.Path(args.corpus))

    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.model, max_length=args.max_length, device=args.device)
    logger.info(f"Cross-encoder: {args.model} on {args.device}")

    n_q = n_q_written = n_q_dropped = n_q_unjudgeable = 0
    n_rec = n_neg = n_dropped = n_missing_text = 0
    # If a false negative is instead *promoted* to positive, how much denser do
    # the labels get? Reported, never applied -- relabelling asserts the
    # cross-encoder's opinion as ground truth and needs separate validation.
    promoted_hist: List[int] = []
    pos_per_q: List[int] = []
    n_promoted = 0
    drop_frac_per_q: List[float] = []
    margins: List[float] = []

    writer = None if (args.dry_run) else out_path.open("w", encoding="utf-8")
    t0 = time.time()
    try:
        for qid, group in iter_query_groups(rank_path):
            if args.max_queries and n_q >= args.max_queries:
                break
            n_q += 1
            n_rec += len(group)

            gold = [r for r in group if int(r.get("label", 0)) == 1]
            negs = [r for r in group if int(r.get("label", 0)) != 1]
            n_neg += len(negs)

            question = next((r.get("question") for r in group if r.get("question")), None)
            scorable_gold = [r for r in gold if text.get(r["pid"])]
            scorable_negs = [r for r in negs if text.get(r["pid"])]
            n_missing_text += (len(gold) - len(scorable_gold)) + (len(negs) - len(scorable_negs))

            # Nothing to compare against -- most often the gold passage was not
            # retrieved into the pool at all (MedMCQA pool recall is 0.838).
            # Keep the query untouched.
            if not question or not scorable_gold or not scorable_negs:
                n_q_unjudgeable += 1
                n_q_written += 1
                if writer:
                    for r in group:
                        writer.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue

            pairs = [(question, text[r["pid"]]) for r in scorable_gold + scorable_negs]
            scores = np.asarray(
                ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=False),
                dtype=np.float64,
            )
            g_scores = scores[: len(scorable_gold)]
            n_scores = scores[len(scorable_gold):]
            best_gold = float(g_scores.max())
            threshold = best_gold - float(args.margin)

            drop_ids = {
                r["pid"] for r, s in zip(scorable_negs, n_scores) if float(s) >= threshold
            }
            n_dropped += len(drop_ids)
            drop_frac_per_q.append(len(drop_ids) / max(len(negs), 1))
            promoted_hist.append(len(gold) + len(drop_ids))
            if n_scores.size:
                margins.append(best_gold - float(n_scores.max()))

            if args.mode == "drop":
                kept = [r for r in group if r["pid"] not in drop_ids]
            else:
                # Promote instead of removing: the pool is unchanged, the labels
                # get a third level. gold_passage_ids is kept consistent with the
                # labels so the record does not contradict itself downstream.
                promoted = sorted(drop_ids)
                kept = []
                for r in group:
                    r = dict(r)
                    if r["pid"] in drop_ids:
                        r["label"] = int(args.promoted_grade)
                    elif int(r.get("label", 0)) >= 1:
                        r["label"] = int(args.gold_grade)
                    if promoted:
                        r["gold_passage_ids"] = sorted(
                            set(r.get("gold_passage_ids") or []) | drop_ids
                        )
                    kept.append(r)
                n_promoted += len(drop_ids)

            # A query needs at least one positive and one negative to contribute
            # to either objective; otherwise it is dead weight.
            n_pos_kept = sum(1 for r in kept if int(r.get("label", 0)) >= 1)
            n_neg_kept = sum(1 for r in kept if int(r.get("label", 0)) == 0)
            if n_pos_kept == 0 or n_neg_kept == 0 or len(kept) < 2:
                n_q_dropped += 1
                continue
            pos_per_q.append(n_pos_kept)
            n_q_written += 1
            if writer:
                for r in kept:
                    writer.write(json.dumps(r, ensure_ascii=False) + "\n")

            if n_q % 500 == 0:
                el = time.time() - t0
                logger.info(
                    f"  {n_q:,} queries | {n_dropped:,} negatives flagged "
                    f"({100 * n_dropped / max(n_neg, 1):.1f}%) | "
                    f"{n_q / el:.1f} q/s | {el / 60:.1f} min"
                )
    finally:
        if writer:
            writer.close()

    elapsed = time.time() - t0
    report = {
        "rank_data": str(rank_path),
        "out": None if args.dry_run else str(out_path),
        "model": args.model,
        "margin": args.margin,
        "queries_seen": n_q,
        "queries_written": n_q_written,
        "queries_dropped_no_usable_pool": n_q_dropped,
        "queries_unjudgeable_gold_not_in_pool": n_q_unjudgeable,
        "records_seen": n_rec,
        "negatives_seen": n_neg,
        "mode": args.mode,
        "negatives_flagged_false": n_dropped,
        "negatives_dropped": n_dropped if args.mode == "drop" else 0,
        "negatives_promoted": n_promoted,
        "positives_per_query_after": round(float(np.mean(pos_per_q)), 2) if pos_per_q else None,
        "negatives_dropped_pct": round(100 * n_dropped / max(n_neg, 1), 2),
        "mean_drop_fraction_per_query": round(float(np.mean(drop_frac_per_q)), 4) if drop_frac_per_q else 0.0,
        "queries_with_any_false_negative_pct": round(
            100 * float(np.mean([f > 0 for f in drop_frac_per_q])), 2) if drop_frac_per_q else 0.0,
        "median_gold_minus_best_negative": round(float(np.median(margins)), 4) if margins else None,
        "positives_per_query_if_relabelled": round(float(np.mean(promoted_hist)), 2) if promoted_hist else None,
        "passages_without_text": n_missing_text,
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info("=" * 72)
    for k, v in report.items():
        logger.info(f"  {k:42s} {v}")
    logger.info("=" * 72)
    if args.report:
        p = pathlib.Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
        logger.info(f"report -> {p}")


if __name__ == "__main__":
    main()
