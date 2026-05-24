#!/usr/bin/env python3
"""Merge per-cell QA JSON files (one retriever + one reader) into one matrix JSON."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List


def _cell_to_run(obj: Dict[str, Any], path: pathlib.Path) -> Dict[str, Any]:
    if "runs" in obj and isinstance(obj["runs"], list):
        if len(obj["runs"]) != 1:
            raise ValueError(f"{path}: expected one run per cell file, got {len(obj['runs'])}")
        return obj["runs"][0]
    meta = obj.get("meta") or {}
    datasets = obj.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"{path}: missing 'datasets' or 'runs'")
    reader = meta.get("reader_model") or (meta.get("reader_models") or [None])[0]
    retriever = meta.get("retriever")
    if not reader or not retriever:
        raise ValueError(f"{path}: meta missing reader_model or retriever")
    return {
        "reader_model": str(reader),
        "retriever": str(retriever),
        "query_emb_cache": meta.get("query_emb_cache") or [],
        "online_retrieval": bool(meta.get("online_retrieval", False)),
        "datasets": datasets,
    }


def merge(paths: List[pathlib.Path]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    seen = set()
    for p in paths:
        obj = json.loads(p.read_text(encoding="utf-8"))
        meta = obj.get("meta") or {}
        if meta.get("checkpoint"):
            ds = obj.get("datasets") or {}
            print(
                f"WARNING: {p} is a checkpoint (datasets={list(ds.keys())}); "
                "finish the run or omit this file from merge."
            )
        run = _cell_to_run(obj, p)
        key = (run["reader_model"], run["retriever"])
        if key in seen:
            raise ValueError(f"Duplicate cell: {key} (from {p})")
        seen.add(key)
        runs.append(run)
    runs.sort(key=lambda r: (r["retriever"], r["reader_model"]))
    return {
        "meta": {
            "merged_from": [str(p) for p in paths],
            "n_runs": len(runs),
            "matrix_mode": True,
        },
        "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge single-cell QA JSON outputs into qa_e2e_matrix_1k.json shape."
    )
    p.add_argument(
        "inputs",
        nargs="*",
        help="JSON files or directories (dirs: all *.json, non-recursive).",
    )
    p.add_argument(
        "-o",
        "--out",
        type=str,
        default="results/qa_e2e_matrix_1k.json",
    )
    args = p.parse_args()
    paths: List[pathlib.Path] = []
    for raw in args.inputs:
        path = pathlib.Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            paths.append(path)
    if not paths:
        raise SystemExit("No input JSON files.")
    payload = merge(paths)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(payload['runs'])} cells -> {out}")
    for r in payload["runs"]:
        print(f"  {r['retriever']} | {r['reader_model']}")


if __name__ == "__main__":
    main()
