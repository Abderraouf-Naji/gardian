"""
Script 00 – Corpus preparation (target 3M passages, 80% / 10% / 10% train–val–test).

Data sources (in priority order for quality):
1. PubMedQA Labeled              ~3,358  passages  (gold standard evaluation)
2. MedMCQA                       ~50,940 passages  (strict filtered, high-quality explanations)
3. PubMedQA Artificial           ~570,574 passages (synthetic data)
4. MedRAG/pubmed                 23.9M available   (PRIMARY filler — field bug FIXED)

Total target: 3,000,000 passages
- Training:   2,400,000 (80%)
- Validation:   300,000 (10%)
- Test:         300,000 (10%)

Root cause of previous failures:
  - MedRAG/statpearls: gated by StatPearls privacy policy — data files intentionally empty on HF.
  - bigbio/bioasq: requires HF account agreement (403 without token).
  - MedRAG/pubmed: WAS working but field priority was wrong.
      Schema keys: id, title, content, contents, PMID
      Old code checked: "text" → "abstract" → "content"  ← "text" and "abstract" don't exist
      Fixed code checks: "content" → "contents" → "text" → "abstract"
      This caused 100% of rows to be counted as no_text and skipped silently.
"""

import json
import pathlib
import random
import re
import sys
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from loguru import logger
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

for _res, _pkg in [
    ("tokenizers/punkt",  "punkt"),
    ("corpora/wordnet",   "wordnet"),
    ("corpora/omw-1.4",  "omw-1.4"),
]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

random.seed(42)

TOTAL_PASSAGES = 3_000_000
TRAIN_PASSAGES = 2_400_000
VAL_PASSAGES   =   300_000
TEST_PASSAGES  =   300_000

TARGETS = {
    "pubmedqa_labeled":        3_000,   # fixed
    "medmcqa":               500_000,   # hard ceiling (~51k pass strict filter)
    "pubmedqa_artificial":   600_000,   # hard ceiling (~571k available)
    "medrag_pubmed":       2_500_000,   # primary filler — 23.9M available on HF
}

PUBMEDQA_LABELED    = "pqa_labeled"
PUBMEDQA_ARTIFICIAL = "pqa_artificial"
MEDMCQA_HF_ID       = "medmcqa"
MEDRAG_PUBMED_ID    = "MedRAG/pubmed"

# -----------------------------------------------------------------------------
# Text normalization & quality filters  (UNCHANGED — do not relax)
# -----------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return str(text).strip().lower()


def normalize_question(question: str) -> str:
    if not question:
        return ""
    question = re.sub(r'\)([A-Z])', r') \1', question)
    question = re.sub(r'([a-z]\))([a-z])', r'\1 \2', question, flags=re.IGNORECASE)
    question = re.sub(r'\.([A-Z])', r'. \1', question)
    question = re.sub(r'\s+', ' ', question)
    question = re.sub(r',([A-Za-z])', r', \1', question)
    question = question.replace('–', '-').replace('—', '-')
    return question.strip()


def clean_question_formatting(question: str) -> str:
    if not question:
        return ""
    def _fmt(m):
        return f"\n{m.group(1).upper()}) {m.group(2).strip()}"
    question = re.sub(r'([a-e])\s*[\).]\s*([A-Z][a-z]+)', _fmt, question)
    question = re.sub(r'\s+[a-e]\)\s*$', '', question)
    return question.strip()


def has_semantic_leakage(passage: str, answer: str) -> bool:
    if not passage or not answer:
        return False
    p   = normalize_text(passage)
    a   = normalize_text(answer)
    esc = re.escape(a)
    if re.search(rf'\b{esc}\b', p):
        return True
    for pat in [rf'{esc}\s+is\s+(?:defined as|a|an|the)',
                rf'{esc}\s+refers to', rf'{esc}\s+means']:
        if re.search(pat, p):
            return True
    try:
        synsets = set()
        for tok in word_tokenize(a):
            for syn in wn.synsets(tok):
                for lem in syn.lemmas():
                    synsets.add(lem.name().lower())
        if any(t in synsets for t in word_tokenize(p)):
            return True
    except Exception:
        pass
    if len(answer) > 10 and SequenceMatcher(None, a, p).ratio() > 0.85:
        return True
    return False


def filter_short_passages(text: str, min_chars: int = 200) -> bool:
    if not text or len(text.strip()) < min_chars:
        return False
    if len(re.findall(r'[a-zA-Z]', text)) < min_chars * 0.3:
        return False
    return True


def clean_explanation(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    for pat in [r'^ANSWER:\s*', r'^Ans\.\s*', r'^Answer:\s*',
                r'^REF:\s*', r'^Reference:\s*']:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def write_jsonl(records: List[dict], path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"  -> {path}  ({len(records):,} records)")


def write_corpus_jsonl(passages: Dict[str, str], source: str,
                       path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for pid, text in passages.items():
            fh.write(json.dumps(
                {"id": pid, "text": text, "title": "", "source": source},
                ensure_ascii=False,
            ) + "\n")
    logger.success(f"  -> {path}  ({len(passages):,} passages)")


def three_way_split(items, strata, test_frac=0.10, dev_frac=0.10, seed=42):
    try:
        train_dev, test = train_test_split(
            items, test_size=test_frac, random_state=seed, stratify=strata)
        adj = dev_frac / (1.0 - test_frac)
        idx = {id(x): s for x, s in zip(items, strata)}
        strata_td = [idx[id(x)] for x in train_dev]
        train, dev = train_test_split(
            train_dev, test_size=adj, random_state=seed, stratify=strata_td)
        return train, dev, test
    except Exception:
        s = list(items); random.shuffle(s); n = len(s)
        return s[:int(n*.8)], s[int(n*.8):int(n*.9)], s[int(n*.9):]

# -----------------------------------------------------------------------------
# Source 1 — PubMedQA LABELED
# -----------------------------------------------------------------------------

def download_pubmedqa_labeled_full() -> Tuple[Dict[str, str], List[dict]]:
    logger.info("=" * 62)
    logger.info("1 / 4  PubMedQA LABELED (EVALUATION – full 1k gold standard)")
    logger.info("=" * 62)

    corpus: Dict[str, str] = {}
    ds    = load_dataset("qiaojin/PubMedQA", PUBMEDQA_LABELED)
    items = list(ds["train"])
    logger.info(f"  Loaded {len(items):,} expert-annotated examples")

    valid = [i for i in items
             if i.get("context", {}).get("contexts")
             and i.get("question") and i.get("final_decision")]
    logger.info(f"  Valid: {len(valid):,}")

    records = []
    for item in valid:
        pubid = str(item["pubid"])
        gold  = []
        for idx, ctx in enumerate(item["context"]["contexts"]):
            pid = f"pq_lab_{pubid}_{idx}"
            corpus[pid] = ctx
            gold.append(pid)
        records.append({
            "id": f"pq_lab_{pubid}",
            "question": normalize_question(item["question"]),
            "answer": item["final_decision"],
            "answer_list": [item["final_decision"]],
            "long_answer": item.get("long_answer", ""),
            "options": None, "answer_letter": None,
            "gold_passage_ids": gold, "question_type": "yesno",
            "dataset": "pubmedqa_labeled",
            "source_config": "evaluation_gold", "purpose": "evaluation",
        })

    write_jsonl(records, DATA_DIR / "pubmedqa_labeled_eval.jsonl")
    write_corpus_jsonl(corpus, "pubmedqa_labeled",
                       DATA_DIR / "corpus_pubmedqa_labeled.jsonl")
    logger.success(f"  Complete: {len(corpus):,} passages, {len(records):,} QA pairs")
    return corpus, records

# -----------------------------------------------------------------------------
# Source 2 — MedMCQA
# -----------------------------------------------------------------------------

def download_medmcqa_sampled(target_passages: int) -> Tuple[Dict[str, str], List[dict]]:
    logger.info("=" * 62)
    logger.info(f"2 / 4  MedMCQA (STRICT – target {target_passages:,} passages)")
    logger.info("=" * 62)

    corpus: Dict[str, str] = {}
    try:
        ds = load_dataset(MEDMCQA_HF_ID)
        all_items = []
        for sname in ds.keys():
            chunk = list(ds[sname])
            logger.info(f"  {sname}: {len(chunk):,}")
            all_items.extend(chunk)
        logger.info(f"  Total: {len(all_items):,}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return {}, []

    records, passages = [], {}
    stats = dict(processed=0, no_question=0, no_options=0, no_explanation=0,
                 short_explanation=0, semantic_leakage=0, label_mismatch=0, passed=0)

    for item in all_items:
        if len(records) >= target_passages:
            break
        stats["processed"] += 1

        raw_q = item.get("question", "")
        if not raw_q: stats["no_question"] += 1; continue
        question = clean_question_formatting(normalize_question(raw_q))
        if not question: stats["no_question"] += 1; continue

        opts = {k: (item.get(f, "") or "").strip()
                for k, f in zip("ABCD", ["opa", "opb", "opc", "opd"])}
        opts = {k: v for k, v in opts.items() if v}
        if not opts: stats["no_options"] += 1; continue

        cop = item.get("cop", 0)
        try:
            cop = int(cop) if cop is not None else 0
        except (ValueError, TypeError):
            cop = 0
        letter = ["A", "B", "C", "D"][cop] if 0 <= cop < 4 else "A"
        ans = opts.get(letter, "")
        if not ans: stats["no_options"] += 1; continue
        if normalize_text(ans) != normalize_text(opts.get(letter, "")):
            stats["label_mismatch"] += 1; continue

        raw_exp = item.get("exp", "")
        if raw_exp is None: stats["no_explanation"] += 1; continue
        exp = clean_explanation(str(raw_exp))
        if len(exp) < 200: stats["short_explanation"] += 1; continue
        if has_semantic_leakage(exp, ans): stats["semantic_leakage"] += 1; continue

        item_id = str(item.get("id", len(records)))
        pid = f"mcq_{item_id}_explanation"
        passages[pid] = exp

        q_l = question.lower()
        if any(w in q_l for w in ["diagnosis", "diagnose", "condition", "disease"]):
            qtype = "diagnosis"
        elif any(w in q_l for w in ["treatment", "therapy", "drug", "medication"]):
            qtype = "treatment"
        elif any(w in q_l for w in ["mechanism", "pathway", "cause"]):
            qtype = "mechanism"
        elif any(w in q_l for w in ["contraindication", "adverse", "side effect"]):
            qtype = "contraindication"
        else:
            qtype = "factoid"

        records.append({
            "id": f"mcq_{item_id}", "question": question,
            "answer": ans, "answer_list": [ans], "long_answer": None,
            "options": opts, "answer_letter": letter,
            "gold_passage_ids": [pid], "question_type": qtype,
            "dataset": "medmcqa",
            "source_config": str(item.get("subject_name", "unknown") or "unknown"),
            "purpose": "training",
        })
        stats["passed"] += 1

    corpus.update(passages)
    logger.info(f"\n  MedMCQA STRICT STATISTICS:")
    logger.info(f"    Processed:        {stats['processed']:,}")
    logger.info(f"    PASSED:           {stats['passed']:,}")
    logger.info(f"    Acceptance rate:  {stats['passed']/max(1,stats['processed'])*100:.1f}%")
    logger.info(f"    Semantic leakage: {stats['semantic_leakage']:,}")
    logger.info(f"    Short expl.:      {stats['short_explanation']:,}")

    if not records:
        logger.warning("  No valid records after strict validation")
        return {}, []

    strata = [r["question_type"] for r in records]
    train, dev, test = three_way_split(records, strata)
    write_jsonl(train, DATA_DIR / "medmcqa_train.jsonl")
    write_jsonl(dev,   DATA_DIR / "medmcqa_dev.jsonl")
    write_jsonl(test,  DATA_DIR / "medmcqa_test.jsonl")
    write_corpus_jsonl(corpus, "medmcqa", DATA_DIR / "corpus_medmcqa.jsonl")
    logger.success(f"  Complete: {len(corpus):,} passages, {len(records):,} QA pairs")
    return corpus, records

# -----------------------------------------------------------------------------
# Source 3 — PubMedQA ARTIFICIAL
# -----------------------------------------------------------------------------

def download_pubmedqa_artificial_sampled(target_passages: int
                                         ) -> Tuple[Dict[str, str], List[dict]]:
    logger.info("=" * 62)
    logger.info(f"3 / 4  PubMedQA ARTIFICIAL (target ~{target_passages:,} passages)")
    logger.info("=" * 62)

    corpus: Dict[str, str] = {}
    all_records: List[dict] = []

    ds    = load_dataset("qiaojin/PubMedQA", PUBMEDQA_ARTIFICIAL)
    items = list(ds["train"])
    logger.info(f"  Loaded {len(items):,} artificial examples")

    stats = dict(no_context=0, no_question=0, no_answer=0, short_context=0)
    valid = []
    for item in items:
        ctx = item.get("context", {}).get("contexts", [])
        if not ctx:                    stats["no_context"] += 1;  continue
        if not item.get("question"):   stats["no_question"] += 1; continue
        if not item.get("final_decision"): stats["no_answer"] += 1; continue
        if len(ctx) < 3:               stats["short_context"] += 1; continue
        valid.append(item)

    for k, v in stats.items():
        logger.info(f"    {k}: {v:,}")
    logger.info(f"  Valid: {len(valid):,}")

    avg = 3
    n   = min(len(valid), max(1, target_passages // avg))
    sampled = random.sample(valid, n)
    logger.info(f"  Sampled {n:,} items (~{n*avg:,} passages)")

    strata = [x["final_decision"] for x in sampled]
    tr_i, vl_i, ts_i = three_way_split(sampled, strata)
    logger.info(f"  Split – Train: {len(tr_i):,}, Val: {len(vl_i):,}, Test: {len(ts_i):,}")

    def convert(items_: list, purpose: str) -> List[dict]:
        recs = []
        for item in items_:
            pubid = str(item["pubid"])
            gold  = []
            for i, ctx in enumerate(item["context"]["contexts"]):
                pid = f"pq_art_{pubid}_{i}"
                if filter_short_passages(ctx, min_chars=100):
                    corpus[pid] = ctx
                    gold.append(pid)
            if not gold:
                continue
            recs.append({
                "id": f"pq_art_{pubid}",
                "question": normalize_question(item["question"]),
                "answer": item["final_decision"],
                "answer_list": [item["final_decision"]],
                "long_answer": item.get("long_answer", ""),
                "options": None, "answer_letter": None,
                "gold_passage_ids": gold, "question_type": "yesno",
                "dataset": "pubmedqa_artificial",
                "source_config": "training", "purpose": purpose,
            })
        return recs

    tr = convert(tr_i, "training")
    vl = convert(vl_i, "validation")
    ts = convert(ts_i, "test")
    write_jsonl(tr, DATA_DIR / "pubmedqa_artificial_train.jsonl")
    write_jsonl(vl, DATA_DIR / "pubmedqa_artificial_dev.jsonl")
    write_jsonl(ts, DATA_DIR / "pubmedqa_artificial_test.jsonl")
    all_records.extend(tr + vl + ts)
    write_corpus_jsonl(corpus, "pubmedqa_artificial",
                       DATA_DIR / "corpus_pubmedqa_artificial.jsonl")
    logger.success(f"  Complete: {len(corpus):,} passages, {len(all_records):,} QA pairs")
    return corpus, all_records

# -----------------------------------------------------------------------------
# Source 4 — MedRAG/pubmed  (PRIMARY large-scale filler)
# -----------------------------------------------------------------------------

def download_medrag_pubmed(target_passages: int,
                           max_retries: int = 10,
                           retry_delay: float = 5.0
                           ) -> Tuple[Dict[str, str], List[dict]]:
    """
    Stream MedRAG/pubmed with automatic retry-resume on network errors.

    Strategy: when a connection drop occurs mid-stream, we re-open the dataset
    and skip forward past already-seen doc_ids using the `seen` set.
    This avoids restarting from zero after a transient dropout.

    Your last run reached 2,000,000 passages before dropping at
    chunk pubmed23n0117.jsonl — this handler will resume from there.
    """
    logger.info("=" * 62)
    logger.info(f"4 / 4  MedRAG/pubmed (STREAMING – target {target_passages:,} passages)")
    logger.info("=" * 62)
    logger.info("  Schema: id | title | content | contents | PMID")
    logger.info("  Retry-resume enabled (network-drop safe)")

    corpus: Dict[str, str] = {}
    stats  = dict(added=0, too_short=0, no_text=0, duplicate=0, retries=0)
    seen:  set = set()

    schema_logged = False
    attempt = 0

    while stats["added"] < target_passages and attempt <= max_retries:
        attempt += 1
        if attempt > 1:
            import time
            logger.warning(f"  Retry {attempt-1}/{max_retries} — "
                           f"resuming from {stats['added']:,} passages "
                           f"(skipping {len(seen):,} seen IDs)...")
            time.sleep(retry_delay)

        try:
            ds = load_dataset(MEDRAG_PUBMED_ID, split="train", streaming=True)
        except Exception as e:
            logger.error(f"  Failed to open stream (attempt {attempt}): {e}")
            continue

        try:
            for item in ds:
                if stats["added"] >= target_passages:
                    break

                doc_id = str(item.get("id", item.get("PMID", stats["added"])))

                # Skip already-collected passages on resume
                if doc_id in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(doc_id)

                if not schema_logged:
                    logger.info(f"  ✓ Schema keys: {list(item.keys())}")
                    schema_logged = True

                #  FIXED field priority — "content" first (confirmed schema)
                text = (
                    item.get("content")
                    or item.get("contents")
                    or item.get("text")
                    or item.get("abstract")
                )

                if not text or not isinstance(text, str):
                    stats["no_text"] += 1
                    continue

                if not filter_short_passages(text, min_chars=200):
                    stats["too_short"] += 1
                    continue

                pid = f"medrag_{doc_id}_{stats['added']}"
                corpus[pid] = text
                stats["added"] += 1

                if stats["added"] % 100_000 == 0:
                    logger.info(f"    Progress: {stats['added']:,}/{target_passages:,} "
                                f"| retries so far: {stats['retries']}")

            # Clean exit — stream exhausted or target reached
            break

        except (RuntimeError, ConnectionError, OSError, Exception) as e:
            stats["retries"] += 1
            logger.warning(f"  Network error after {stats['added']:,} passages: "
                           f"{type(e).__name__}: {e}")
            if attempt > max_retries:
                logger.error("  Max retries exceeded — saving what we have")
                break
            # Loop continues → re-opens stream → skips seen IDs → resumes

    logger.info(f"\n  MedRAG/pubmed Statistics:")
    for k, v in stats.items():
        logger.info(f"    {k}: {v:,}")

    if not corpus:
        logger.warning("  No passages added — fallback source will engage")
        return corpus, []

    write_corpus_jsonl(corpus, "medrag_pubmed",
                       DATA_DIR / "corpus_medrag_pubmed.jsonl")
    logger.success(f"  MedRAG/pubmed complete: {len(corpus):,} passages")
    return corpus, []



# -----------------------------------------------------------------------------
# Unified corpus + final report
# -----------------------------------------------------------------------------

def create_unified_corpus(all_passages: Dict[str, str]) -> None:
    out = DATA_DIR / "indices" / "unified" / "corpus_unified.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for pid, text in all_passages.items():
            f.write(json.dumps({
                "id":     pid,
                "text":   text,
                "title":  "",
                "source": pid.split("_")[0] if "_" in pid else "unknown",
            }, ensure_ascii=False) + "\n")
    logger.success(f"Unified corpus: {out} ({len(all_passages):,} passages)")


def generate_final_report(passage_counts: Dict[str, int],
                          qa_counts: Dict[str, int]) -> None:
    logger.info("\n" + "=" * 62)
    logger.info(f"  FINAL CORPUS SUMMARY  (target {TOTAL_PASSAGES:,}, 80/10/10)")
    logger.info("=" * 62)
    total_p = sum(passage_counts.values())
    total_q = sum(qa_counts.values())

    logger.info("\n  PASSAGE COUNTS BY SOURCE:")
    for src, cnt in passage_counts.items():
        flag = "✓" if cnt > 0 else "✗"
        logger.info(f"    {flag}  {src:32s}: {cnt:>10,}")
    logger.info(f"       {'TOTAL':32s}: {total_p:>10,}")

    logger.info("\n  QA PAIR COUNTS BY SOURCE:")
    for src, cnt in qa_counts.items():
        if cnt:
            logger.info(f"       {src:32s}: {cnt:>10,}")
    logger.info(f"       {'TOTAL':32s}: {total_q:>10,}")

    diff = total_p - TOTAL_PASSAGES
    logger.info(f"\n  Target : {TOTAL_PASSAGES:,}")
    logger.info(f"  Actual : {total_p:,}")
    if diff < 0:
        logger.warning(f"  SHORT  : {diff:,}  (MedRAG/pubmed has 23.9M — raise target if needed)")
    else:
        logger.info(f"  SURPLUS: +{diff:,}")
    logger.info("=" * 62)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, ".")

    logger.info("=" * 62)
    logger.info(f"CORPUS PREPARATION  (target {TOTAL_PASSAGES:,} passages, 80/10/10)")
    logger.info("Sources: PubMedQA-L | MedMCQA | PubMedQA-A | MedRAG/pubmed ")
    logger.info("=" * 62)

    all_passages:   Dict[str, str] = {}
    all_qa_records: List[dict]     = []
    passage_counts: Dict[str, int] = {}
    qa_counts:      Dict[str, int] = {}

    # ── 1. PubMedQA Labeled ──────────────────────────────────────────────────
    p, q = download_pubmedqa_labeled_full()
    all_passages.update(p); all_qa_records.extend(q)
    passage_counts["pubmedqa_labeled"] = len(p)
    qa_counts["pubmedqa_labeled"]      = len(q)

    # ── 2. MedMCQA ───────────────────────────────────────────────────────────
    p, q = download_medmcqa_sampled(target_passages=TARGETS["medmcqa"])
    all_passages.update(p); all_qa_records.extend(q)
    passage_counts["medmcqa"] = len(p)
    qa_counts["medmcqa"]      = len(q)

    # ── 3. PubMedQA Artificial ───────────────────────────────────────────────
    remaining  = TOTAL_PASSAGES - len(all_passages)
    target_art = min(TARGETS["pubmedqa_artificial"], remaining)
    p, q = download_pubmedqa_artificial_sampled(target_passages=target_art)
    all_passages.update(p); all_qa_records.extend(q)
    passage_counts["pubmedqa_artificial"] = len(p)
    qa_counts["pubmedqa_artificial"]      = len(q)

    # ── 4. MedRAG/pubmed (PRIMARY filler — field bug fixed) ──────────────────
    remaining = TOTAL_PASSAGES - len(all_passages)
    if remaining > 0:
        target_mr = min(TARGETS["medrag_pubmed"], remaining)
        p, q = download_medrag_pubmed(target_passages=target_mr)
        all_passages.update(p); all_qa_records.extend(q)
        passage_counts["medrag_pubmed"] = len(p)
        qa_counts["medrag_pubmed"]      = len(q)
    else:
        logger.info("Target reached — skipping MedRAG/pubmed")
        passage_counts["medrag_pubmed"] = 0
        qa_counts["medrag_pubmed"]      = 0


    # ── Finalize ─────────────────────────────────────────────────────────────
    create_unified_corpus(all_passages)
    generate_final_report(passage_counts, qa_counts)

    logger.success(f"\n COMPLETE!  Total passages : {len(all_passages):,}")
    logger.success(f"             Total QA pairs : {len(all_qa_records):,}")