"""
Download a BEIR retrieval collection and convert it to GARDIAN JSONL (additive).

Supported collections (``--dataset``)
-------------------------------------
``trec_covid``  eval-only. Public NIST pooled qrels (~493 judgments/query) and a
                ~171k corpus: the strongest answer to incomplete-judgment
                critiques on PubMedQA/MedMCQA, and the paper's zero-shot split.
``nfcorpus``    train/dev/test. Graded (0-2) medical IR judgments over a small
                (~3.6k) corpus.
``scifact``     train/test. Scientific claim verification against ~5k abstracts.

Why nfcorpus / scifact matter
-----------------------------
Per-query alpha is close to unlearnable on the QA splits: MedMCQA carries a
single gold passage per query (~1% of the pool), so nDCG@10 is 1/log2(rank+1)
and the alpha surface is quantised and mostly flat. A controller trained only
there has almost no gradient telling it which way to move alpha, which is why
it does not transfer to a densely-judged collection. These two collections are
real retrieval data with multiple relevant documents per query, and adding them
to training keeps TREC-COVID strictly zero-shot.

Check the label density before spending GPU on indices and rank data:

    .venv/bin/python scripts/00b_download_beir.py --dataset nfcorpus --stats-only

Outputs (never overwrite other datasets)
----------------------------------------
  data/corpus_<name>.jsonl
  data/<name>_<split>.jsonl        one file per available split

Relevance: every qrel with grade >= 1 becomes a gold_passage_id, and the graded
value is preserved in ``relevance_grades`` so evaluation can use graded gains
(see src/evaluation/qrels.py). Binary and graded nDCG differ wherever grades
exceed 1; report which one a table uses.

Usage:
    .venv/bin/python scripts/00b_download_beir.py --dataset nfcorpus
    .venv/bin/python scripts/00b_download_beir.py --dataset scifact --force
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

# ``md5`` pins the archive where we have verified it; None means "warn and
# report the digest so it can be pinned" rather than silently trusting the
# download. ``splits`` lists the qrels files BEIR ships for the collection.
BEIR_DATASETS: Dict[str, dict] = {
    "trec_covid": {
        "slug": "trec-covid",
        "md5": "ce62140cb23feb9becf6270d0d1fe6d1",
        "splits": ("test",),
        "eval_only": True,
    },
    "nfcorpus": {
        "slug": "nfcorpus",
        "md5": "a89dba18a62ef92f7d323ec890a0d38d",
        "splits": ("train", "dev", "test"),
        "eval_only": False,
    },
    "scifact": {
        "slug": "scifact",
        "md5": "5f7d1de60b170fc8027bb7898e2efca1",
        "splits": ("train", "test"),
        "eval_only": False,
    },
}

DATA_DIR = pathlib.Path("data")


def corpus_out(name: str) -> pathlib.Path:
    return DATA_DIR / f"corpus_{name}.jsonl"


def queries_out(name: str, split: str) -> pathlib.Path:
    return DATA_DIR / f"{name}_{split}.jsonl"


def _md5_file(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_zip(dest: pathlib.Path, slug: str, expected_md5: Optional[str]) -> pathlib.Path:
    url = f"{BEIR_BASE}/{slug}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        digest = _md5_file(dest)
        if expected_md5 is None:
            logger.info(f"Using cached zip (md5 {digest}, unpinned): {dest}")
            return dest
        if digest == expected_md5:
            logger.info(f"Using cached zip (md5 ok): {dest}")
            return dest
        logger.warning(f"Cached zip md5 {digest} != pinned {expected_md5}; re-downloading")

    logger.info(f"Downloading BEIR {slug} from {url}")
    urllib.request.urlretrieve(url, dest)
    digest = _md5_file(dest)
    if expected_md5 is None:
        logger.warning(
            f"No pinned md5 for {slug}; downloaded digest is {digest}. "
            f"Add it to BEIR_DATASETS['{slug}']['md5'] to make this reproducible."
        )
    elif digest != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {dest}: got {digest}, expected {expected_md5}"
        )
    logger.success(f"Downloaded {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def _open_maybe_gz(path: pathlib.Path):
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_jsonl(path: pathlib.Path) -> Iterable[dict]:
    with _open_maybe_gz(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _find_member(root: pathlib.Path, *candidates: str) -> pathlib.Path:
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    matches = []
    for pat in candidates:
        name = pathlib.Path(pat).name
        matches.extend(root.rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any of {candidates} under {root}"
        )
    return matches[0]


def _load_qrels(path: pathlib.Path) -> Dict[str, Dict[str, int]]:
    """qid -> {docid: grade}.

    Accepts BEIR 3-column ``qid docid score`` and classic TREC 4-column
    ``qid 0 docid score`` (whitespace or tab separated).
    """
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.lower().startswith("query-id"):
                continue
            parts = line.replace(",", "\t").split()
            if len(parts) == 3:
                qid, docid, grade_s = parts
            elif len(parts) >= 4:
                qid, docid, grade_s = parts[0], parts[2], parts[3]
            else:
                raise ValueError(f"Bad qrels line {line_no}: {line!r}")
            qrels[qid][docid] = int(grade_s)
    return qrels


def _query_text(rec: dict) -> str:
    title = str(rec.get("title") or "").strip()
    text = str(rec.get("text") or "").strip()
    if title and text and title not in text:
        return f"{title} {text}".strip()
    return text or title


def _split_stats(qrels: Dict[str, Dict[str, int]]) -> dict:
    """
    Label density for one split -- the number that decides whether a collection
    can teach a per-query fusion weight at all.

    A query with one relevant document gives nDCG@10 = 1/log2(rank+1): alpha
    only matters when it moves that single document, so the objective is
    quantised and mostly flat in alpha. Multiple relevant documents per query
    are what make the alpha surface informative.
    """
    per_q = [len([g for g in gs.values() if g >= 1]) for gs in qrels.values()]
    per_q = [n for n in per_q if n > 0]
    grades: Dict[int, int] = defaultdict(int)
    for gs in qrels.values():
        for g in gs.values():
            if g >= 1:
                grades[int(g)] += 1
    if not per_q:
        return {"queries": 0}
    arr = sorted(per_q)
    return {
        "queries": len(arr),
        "positives": sum(arr),
        "mean_pos_per_query": sum(arr) / len(arr),
        "median_pos_per_query": arr[len(arr) // 2],
        "min_pos": arr[0],
        "max_pos": arr[-1],
        "frac_single_positive": sum(1 for n in arr if n == 1) / len(arr),
        "grade_histogram": dict(sorted(grades.items())),
    }


def convert_beir_dir(
    beir_root: pathlib.Path,
    name: str,
    splits: Iterable[str],
    *,
    write: bool = True,
) -> dict:
    """
    Convert one extracted BEIR collection. Returns per-split density stats.

    ``write=False`` reads the qrels and reports density without touching disk,
    which is how ``--stats-only`` answers "is this collection worth indexing?"
    before any GPU time is spent.
    """
    corpus_path = _find_member(beir_root, "corpus.jsonl.gz", "corpus.jsonl")
    queries_path = _find_member(beir_root, "queries.jsonl.gz", "queries.jsonl")

    logger.info(f"corpus  : {corpus_path}")
    logger.info(f"queries : {queries_path}")

    report: dict = {"dataset": name, "splits": {}}

    # Locate the qrels each split actually ships; BEIR omits e.g. scifact/dev.
    split_qrels: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split in splits:
        try:
            path = _find_member(beir_root, f"qrels/{split}.tsv")
        except FileNotFoundError:
            logger.warning(f"{name}: no qrels/{split}.tsv in the archive; skipping split")
            continue
        logger.info(f"qrels   : {path}  ({split})")
        split_qrels[split] = _load_qrels(path)

    if not split_qrels:
        raise FileNotFoundError(f"{name}: archive contained no usable qrels")

    for split, qrels in split_qrels.items():
        report["splits"][split] = _split_stats(qrels)

    if not write:
        return report

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    with corpus_out(name).open("w", encoding="utf-8") as out:
        for rec in _iter_jsonl(corpus_path):
            doc_id = str(rec.get("_id") or rec.get("id") or "").strip()
            if not doc_id:
                continue
            title = str(rec.get("title") or "")
            text = str(rec.get("text") or "").strip()
            if not text and title:
                text = title
            if not text:
                continue
            out.write(
                json.dumps(
                    {"id": doc_id, "text": text, "title": title, "source": name},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_docs += 1
    logger.success(f"Wrote {corpus_out(name)} ({n_docs:,} passages)")
    report["passages"] = n_docs

    queries_by_id = {
        str(rec.get("_id") or rec.get("id")): rec for rec in _iter_jsonl(queries_path)
    }

    def _sort_key(qid: str):
        return (0, int(qid)) if qid.isdigit() else (1, qid)

    for split, qrels in split_qrels.items():
        dest = queries_out(name, split)
        n_queries = n_pos = skipped_no_pos = 0
        with dest.open("w", encoding="utf-8") as out:
            for qid in sorted(qrels.keys(), key=_sort_key):
                grades = qrels[qid]
                gold_ids = sorted(did for did, g in grades.items() if g >= 1)
                if not gold_ids:
                    skipped_no_pos += 1
                    continue
                qrec = queries_by_id.get(qid)
                if qrec is None:
                    logger.warning(f"qrel qid={qid} missing from queries file; skipping")
                    continue
                question = _query_text(qrec)
                if not question:
                    logger.warning(f"empty query text for qid={qid}; skipping")
                    continue
                out.write(
                    json.dumps(
                        {
                            "id": f"{name}_{qid}",
                            "question": question,
                            "answer": None,
                            "answer_list": [],
                            "long_answer": None,
                            "gold_passage_ids": gold_ids,
                            "relevance_grades": {did: int(grades[did]) for did in gold_ids},
                            "question_type": "other",
                            "dataset": name,
                            "source_config": f"beir/{name}",
                            # Drives how scripts/03 and 04 treat the split; the
                            # test split of a training collection is still only
                            # ever used for evaluation.
                            "purpose": "training" if split in ("train", "dev") else "evaluation",
                            "beir_query_id": qid,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_queries += 1
                n_pos += len(gold_ids)
        logger.success(
            f"Wrote {dest} ({n_queries:,} queries, {n_pos:,} positive qrels, "
            f"skipped_no_pos={skipped_no_pos}, "
            f"{n_pos / max(n_queries, 1):.1f} positives/query)"
        )
        report["splits"][split]["written"] = n_queries

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a BEIR collection into GARDIAN JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(BEIR_DATASETS),
        required=True,
        help="BEIR collection to fetch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite this collection's existing corpus/query files "
             "(other datasets are never touched).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Download and report per-split label density WITHOUT writing any "
             "GARDIAN JSONL. Use this to decide whether a collection can teach "
             "a per-query fusion weight before building indices.",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Where to store/reuse the BEIR zip (default: data/raw/<slug>.zip).",
    )
    args = parser.parse_args()

    name = args.dataset
    spec = BEIR_DATASETS[name]
    slug = spec["slug"]

    outputs = [corpus_out(name)] + [queries_out(name, sp) for sp in spec["splits"]]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.force and not args.stats_only:
        raise SystemExit(
            "Outputs already exist:\n  "
            + "\n  ".join(str(p) for p in existing)
            + "\nPass --force to overwrite, or --stats-only to inspect without writing."
        )

    zip_path = pathlib.Path(args.zip_path or f"data/raw/{slug}.zip")
    zip_path = _download_zip(zip_path, slug, spec["md5"])

    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
        tmp_root = pathlib.Path(tmp)
        logger.info(f"Extracting {zip_path} -> {tmp_root}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_root)
        # The zip normally contains a single top-level <slug>/ directory.
        beir_root = tmp_root / slug
        if not beir_root.is_dir():
            beir_root = tmp_root
        report = convert_beir_dir(
            beir_root, name, spec["splits"], write=not args.stats_only
        )

    print(f"\n{'=' * 78}")
    print(f"LABEL DENSITY | {name}")
    print(f"{'=' * 78}")
    print(f"{'split':8s}{'queries':>9s}{'pos/query':>11s}{'median':>8s}"
          f"{'max':>6s}{'% single-positive':>19s}   grades")
    print("-" * 78)
    for split, st in report["splits"].items():
        if not st.get("queries"):
            continue
        print(f"{split:8s}{st['queries']:>9,}{st['mean_pos_per_query']:>11.1f}"
              f"{st['median_pos_per_query']:>8}{st['max_pos']:>6}"
              f"{st['frac_single_positive'] * 100:>18.1f}%   {st['grade_histogram']}")
    print(f"{'=' * 78}\n")

    if args.stats_only:
        logger.info("--stats-only: no files written.")
        return

    logger.success(
        f"{name} ready. Next (additive indices only):\n"
        f"  .venv/bin/python scripts/01_build_bm25_faiss_indices.py --dataset {name}\n"
        f"  .venv/bin/python scripts/01_build_spladepp_medcpt_indices.py --dataset {name}\n"
        f"  .venv/bin/python scripts/03_generate_rank_data.py --retriever all --dataset {name}"
    )


if __name__ == "__main__":
    main()
