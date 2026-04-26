"""Lightweight validators for evaluation artifact schemas."""

from __future__ import annotations

from typing import Any, Dict


REQUIRED_METRICS = ("ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@20", "mrr")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_metric_block(block: Dict[str, Any], label: str) -> None:
    _require(isinstance(block, dict), f"{label} must be an object")
    for key in REQUIRED_METRICS:
        _require(key in block, f"{label} missing metric '{key}'")


def validate_evaluation_results(payload: Dict[str, Any]) -> None:
    _require(isinstance(payload, dict), "evaluation payload must be an object")
    _require("meta" in payload and isinstance(payload["meta"], dict), "evaluation payload missing meta")
    _require("results" in payload and isinstance(payload["results"], dict), "evaluation payload missing results")
    for retriever, ds_block in payload["results"].items():
        _require(isinstance(ds_block, dict), f"results.{retriever} must be an object")
        for dataset, sys_block in ds_block.items():
            _require(isinstance(sys_block, dict), f"results.{retriever}.{dataset} must be an object")
            for name in ["bm25", "dense", "hybrid", "gardian"]:
                _require(name in sys_block, f"results.{retriever}.{dataset} missing '{name}'")
                _validate_metric_block(sys_block[name], f"results.{retriever}.{dataset}.{name}")


def validate_paper_bundle(payload: Dict[str, Any]) -> None:
    _require(isinstance(payload, dict), "paper bundle must be an object")
    _require("meta" in payload and isinstance(payload["meta"], dict), "paper bundle missing meta")
    _require("results" in payload and isinstance(payload["results"], dict), "paper bundle missing results")
    for retriever, ds_block in payload["results"].items():
        _require(isinstance(ds_block, dict), f"paper results.{retriever} must be object")
        for dataset, abl_block in ds_block.items():
            _require(isinstance(abl_block, dict), f"paper results.{retriever}.{dataset} must be object")
            for ablation, eval_block in abl_block.items():
                _require(isinstance(eval_block, dict), f"paper block {retriever}.{dataset}.{ablation} must be object")
                _require("gardian" in eval_block, f"paper block {retriever}.{dataset}.{ablation} missing gardian")
                _validate_metric_block(
                    eval_block["gardian"],
                    f"paper results.{retriever}.{dataset}.{ablation}.gardian",
                )


def validate_controller_weights(payload: Dict[str, Any]) -> None:
    _require(isinstance(payload, dict), "controller payload must be object")
    _require("stats" in payload and isinstance(payload["stats"], dict), "controller payload missing stats")
    stats = payload["stats"]
    _require("branch_names" in stats and isinstance(stats["branch_names"], list), "controller payload missing branch_names")
    _require("by_question_type" in stats and isinstance(stats["by_question_type"], dict), "controller payload missing by_question_type")
    for qtype, block in stats["by_question_type"].items():
        _require(isinstance(block, dict), f"controller stats for {qtype} must be object")
        for key in ["n_queries", "mean", "std"]:
            _require(key in block, f"controller stats for {qtype} missing '{key}'")
