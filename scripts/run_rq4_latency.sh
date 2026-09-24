#!/usr/bin/env bash
# RQ4: timing pass. Run this ONLY on an otherwise idle GPU.
#
# Effectiveness is throughput-bound and unaffected by a busy device, so
# scripts/run_rq4_reranker_suite.sh can share the GPU. A p50 latency cannot:
# measured under contention it reports queueing, not compute. This re-reads the
# comparison files produced by that suite and fills in their latency blocks.
#
# --eval-only skips cross-encoder backfill, so the tagged _ce_<tag>.jsonl files
# from the effectiveness run are reused and no scores are recomputed.
set -uo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
SEED="${SEED:-42}"
NQ="${NQ:-200}"
WARMUP="${WARMUP:-10}"
LOG_DIR="logs/rq4"
mkdir -p "$LOG_DIR"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [[ "$busy" -gt 0 ]]; then
  echo "WARNING: ${busy} process(es) already hold the GPU."
  echo "Latency measured now is not reportable. Set FORCE=1 to run anyway."
  [[ "${FORCE:-0}" != "1" ]] && exit 1
fi

for R in "hybrid_bm25_faiss" "hybrid_spladepp_medcpt"; do
  echo "=== [$(date -Is)] ${R}: timing PQA-L + MedMCQA ==="
  "$PY" scripts/14_compare_rerankers.py \
    --retriever "$R" \
    --datasets pubmedqa_labeled,medmcqa \
    --seed "$SEED" \
    --eval-only \
    --measure-latency \
    --latency-queries "$NQ" \
    --latency-warmup "$WARMUP" \
    --output "results/reranker_comparison_${R}.json" \
    2>&1 | tee "${LOG_DIR}/latency_${R}_main.log"

  echo "=== [$(date -Is)] ${R}: timing PQA-A ==="
  "$PY" scripts/14_compare_rerankers.py \
    --retriever "$R" \
    --datasets pubmedqa_artificial \
    --max-queries 1000 \
    --seed "$SEED" \
    --eval-only \
    --measure-latency \
    --latency-queries "$NQ" \
    --latency-warmup "$WARMUP" \
    --output "results/reranker_comparison_${R}_pqa_a_q1000.json" \
    2>&1 | tee "${LOG_DIR}/latency_${R}_pqa_a.log"
done

echo "=== [$(date -Is)] RQ4 timing pass done ==="
