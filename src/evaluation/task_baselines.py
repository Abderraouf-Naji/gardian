"""
Trivial task baselines that every reported accuracy must be compared against.

PubMedQA-labeled is 55.2% "yes", so a constant "yes" scores 0.552. An
end-to-end accuracy below that is not a weak result, it is no result -- and a
reader will compute the number in seconds whether or not the paper does. These
helpers put the baseline in the results file so the comparison cannot be
omitted by accident.
"""

from __future__ import annotations

import collections
import json
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence


def majority_class_baseline(gold_labels: Sequence[str]) -> Dict[str, Any]:
    """
    Accuracy of always predicting the most frequent gold label.

    Returns the baseline accuracy, the label it corresponds to, and the full
    label distribution, so the reader of a results file can check it.
    """
    labels = [str(g).strip().lower() for g in gold_labels if str(g).strip()]
    if not labels:
        return {"accuracy": None, "label": None, "distribution": {}, "n": 0}
    counts = collections.Counter(labels)
    label, n = counts.most_common(1)[0]
    total = sum(counts.values())
    return {
        "accuracy": n / total,
        "label": label,
        "distribution": {k: v for k, v in counts.most_common()},
        "n": total,
    }


def random_choice_baseline(n_options: int) -> Optional[float]:
    """Accuracy of picking uniformly at random among *n_options* choices."""
    return 1.0 / n_options if n_options and n_options > 0 else None


def gold_labels_from_jsonl(
    path: str | pathlib.Path, *, field: str = "answer"
) -> List[str]:
    """Read the gold answer label of every question in an eval JSONL."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                value = rec.get(field)
                if value is not None:
                    out.append(str(value))
    return out


def baseline_block(
    gold_labels: Sequence[str],
    *,
    n_options: Optional[int] = None,
) -> Dict[str, Any]:
    """The baseline block embedded in every QA results file."""
    majority = majority_class_baseline(gold_labels)
    block: Dict[str, Any] = {
        "majority_class": majority,
        "note": (
            "Any reported answer_accuracy at or below majority_class.accuracy is "
            "indistinguishable from answering with a constant label."
        ),
    }
    if n_options:
        block["random_choice"] = {
            "accuracy": random_choice_baseline(n_options),
            "n_options": n_options,
        }
    return block


def compare_to_baseline(accuracy: Optional[float], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Signed delta of one system's accuracy against the majority-class baseline."""
    base = (baseline.get("majority_class") or {}).get("accuracy")
    if accuracy is None or base is None:
        return {"delta_vs_majority": None, "beats_majority": None}
    return {
        "delta_vs_majority": float(accuracy) - float(base),
        "beats_majority": bool(float(accuracy) > float(base)),
    }


def annotate_dataset_block(
    dataset_block: Dict[str, Any],
    gold_labels: Iterable[str],
    *,
    n_options: Optional[int] = None,
    accuracy_key: str = "answer_accuracy",
) -> Dict[str, Any]:
    """
    Attach the baseline block and per-system deltas to one dataset's results.

    ``answer_accuracy`` is stored as a bootstrap triple ``[mean, lo, hi]``, so
    the mean is element 0 when the value is a list.
    """
    baseline = baseline_block(list(gold_labels), n_options=n_options)
    dataset_block["baselines"] = baseline

    for system, metrics in (dataset_block.get("aggregate") or {}).items():
        if not isinstance(metrics, dict):
            continue
        acc = metrics.get(accuracy_key)
        if isinstance(acc, (list, tuple)) and acc:
            acc = acc[0]
        metrics.update(compare_to_baseline(acc, baseline))
    return dataset_block


# Datasets whose answer is a choice among enumerated options. For these the
# meaningful constant-answer baseline is "always pick the same LETTER", not
# "always emit the same answer STRING": MedMCQA option texts are free-form, so
# the most frequent string ("all of the above") covers 1.7% of the test split
# and is not a baseline any system could be compared against. The letter
# distribution (A 26.9 / B 25.2 / D 24.7 / C 23.2 on medmcqa_test) is.
MCQ_DATASETS = frozenset({"medmcqa", "medqa"})


def baseline_inputs_for_dataset(
    dataset: str, questions: Sequence[Dict[str, Any]]
) -> tuple[List[str], Optional[int]]:
    """
    The gold labels and option count a dataset's trivial baselines need.

    Returns ``(labels, n_options)`` ready for :func:`annotate_dataset_block`.
    Multiple-choice datasets are keyed on ``answer_letter`` and carry the
    number of options so a random-choice baseline is reported beside the
    majority one; free-text yes/no datasets are keyed on ``answer``.
    """
    ds = (dataset or "").strip().lower()
    if ds in MCQ_DATASETS:
        labels = [str(q.get("answer_letter") or "").strip() for q in questions]
        counts = {len(q.get("options") or {}) for q in questions}
        counts.discard(0)
        # Only meaningful when every question offers the same number of options.
        n_options = counts.pop() if len(counts) == 1 else None
        return [l for l in labels if l], n_options
    return [str(q.get("answer") or "") for q in questions], None
