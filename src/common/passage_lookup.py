"""Resolve passage id -> text for rank JSONL rows that omit ``text``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Set


def scan_corpus_for_pids(corpus_path: Path, want: Set[str]) -> Dict[str, str]:
    """Scan one JSONL corpus for passage ids in ``want``; return id -> text."""
    out: Dict[str, str] = {}
    if not want or not corpus_path.is_file():
        return out
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("id")
            if isinstance(pid, str) and pid in want and pid not in out:
                out[pid] = str(obj.get("text") or "")
                if len(out) >= len(want):
                    break
    return out


def build_passage_text_lookup(
    pids: Iterable[str],
    corpus_paths: Sequence[Path],
) -> Dict[str, str]:
    """Load passage texts from the first corpus files that contain each pid."""
    need = {str(pid) for pid in pids if pid}
    if not need:
        return {}
    out: Dict[str, str] = {}
    for path in corpus_paths:
        if not need:
            break
        found = scan_corpus_for_pids(path, need)
        out.update(found)
        need -= set(found.keys())
    return out


def passage_text_for_record(
    rec: dict,
    lookup: Dict[str, str],
) -> str:
    text = rec.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    pid = str(rec.get("pid", ""))
    return lookup.get(pid, "")
