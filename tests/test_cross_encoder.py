"""Unit tests for cross-encoder backend selection and scoring wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.cross_encoder import (
    CROSS_ENCODER_PRESETS,
    CrossEncoderScorer,
    resolve_cross_encoder_backend,
    resolve_device,
)


def test_resolve_backend_auto():
    assert resolve_cross_encoder_backend("castorini/monot5-base-msmarco", "auto") == "monot5"
    assert resolve_cross_encoder_backend("castorini/monobert-large-msmarco", "auto") == "monobert"
    assert resolve_cross_encoder_backend("BAAI/bge-reranker-v2-m3", "auto") == "st"


def test_resolve_backend_explicit():
    assert resolve_cross_encoder_backend("any-model", "monot5") == "monot5"


def test_presets_defined():
    assert "monot5_med" in CROSS_ENCODER_PRESETS
    assert "bge_v2_m3" in CROSS_ENCODER_PRESETS


@patch("src.retrieval.cross_encoder.build_pair_scorer")
def test_cross_encoder_scorer_delegates(mock_build):
    backend = MagicMock()
    backend.score_pairs.return_value = [0.5, 0.9]
    mock_build.return_value = backend

    scorer = CrossEncoderScorer(
        "castorini/monot5-base-msmarco-10k",
        device="cpu",
        backend="monot5",
        fp16=False,
    )
    out = scorer.score_pairs(["q1", "q2"], ["p1", "p2"])
    assert out == [0.5, 0.9]
    backend.score_pairs.assert_called_once()


def test_cross_encoder_length_mismatch():
    with patch("src.retrieval.cross_encoder.build_pair_scorer"):
        scorer = CrossEncoderScorer("x", device="cpu", fp16=False)
    with pytest.raises(ValueError, match="mismatch"):
        scorer.score_pairs(["q"], ["p1", "p2"])


def test_resolve_device_auto_cuda(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.cross_encoder.torch.cuda.is_available",
        lambda: True,
    )
    assert resolve_device("auto") == "cuda"
