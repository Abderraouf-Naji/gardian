#!/usr/bin/env bash
# RQ4: effectiveness half of the re-ranker comparison.
#
# Scores every system on the same candidate pools for the two back-ends in the
# RQ4 figure. Latency is NOT measured here -- cross-encoder scoring is
# throughput-bound and unaffected by a contended GPU, but a p50 measured while
# another job holds the device is not reportable. Timing runs separately via
# scripts/run_rq4_latency.sh on an idle GPU.
#
# Output names match what scripts/plot_reranker_comparison.py expects.
set -uo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
SEED="${SEED:-42}"
LOG_DIR="logs/rq4"
mkdir -p "$LOG_DIR" results

RETRIEVERS=("hybrid_bm25_faiss" "hybrid_spladepp_medcpt")

for R in "${RETRIEVERS[@]}"; do
  echo "=== [$(date -Is)] ${R}: PQA-L + MedMCQA (full test splits) ==="
  "$PY" scripts/14_compare_rerankers.py \
    --retriever "$R" \
    --datasets pubmedqa_labeled,medmcqa \
    --seed "$SEED" \
    --skip-backfill \
    --output "results/reranker_comparison_${R}.json" \
    2>&1 | tee "${LOG_DIR}/compare_${R}_main.log"

  echo "=== [$(date -Is)] ${R}: PQA-A (1000-query subsample) ==="
  "$PY" scripts/14_compare_rerankers.py \
    --retriever "$R" \
    --datasets pubmedqa_artificial \
    --max-queries 1000 \
    --seed "$SEED" \
    --skip-backfill \
    --output "results/reranker_comparison_${R}_pqa_a_q1000.json" \
    2>&1 | tee "${LOG_DIR}/compare_${R}_pqa_a.log"
done

echo "=== [$(date -Is)] RQ4 re-ranker effectiveness suite done ==="
