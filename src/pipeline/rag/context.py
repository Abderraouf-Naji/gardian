"""Format retrieved passages for the reader context block."""

from __future__ import annotations

from typing import Any, Dict, List


def format_passages(
    passages: List[Dict[str, Any]],
    *,
    top_k: int,
    max_chars_per_passage: int,
    include_signal_features: bool = False,
) -> str:
    lines: List[str] = []
    for i, p in enumerate(passages[:top_k]):
        pid = p.get("id", f"doc_{i}")
        text = (p.get("text") or "")[:max_chars_per_passage]
        if include_signal_features:
            bits = []
            for key in (
                "gardian_score",
                "bm25_score",
                "spladepp_score",
                "dense_score",
                "medcpt_score",
                "hybrid_rrf_score",
            ):
                if key in p:
                    try:
                        bits.append(f"{key}={float(p.get(key, 0.0)):.4f}")
                    except (TypeError, ValueError):
                        pass
            meta = f"; {'; '.join(bits)}" if bits else ""
            lines.append(f"[P{i + 1}] (id={pid}{meta}): {text}")
        else:
            lines.append(f"[P{i + 1}] (id={pid}): {text}")
    return "\n\n".join(lines)
