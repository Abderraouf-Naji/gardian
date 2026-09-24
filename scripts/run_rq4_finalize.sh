#!/usr/bin/env bash
# RQ4: everything that must happen on an idle GPU, then the paper artifacts.
#
# Run this after run_rq4_reranker_suite.sh, run_rq4_ltr_suite.sh and the
# GARDIAN-Lite training have all finished, and after any unrelated GPU job has
# exited. It:
#
#   1. times every re-ranker on a quiet device (fills the latency blocks)
#   2. evaluates and times GARDIAN-Lite on the same pools
#   3. regenerates the cost tables from the artifacts
#   4. rebuilds the RQ4 figure
#
# Nothing here trains or re-scores anything, so it is cheap to re-run.
set -uo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR="logs/rq4"
mkdir -p "$LOG_DIR" results/figures

echo "=== [$(date -Is)] RQ4 finalize ==="

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [[ "$busy" -gt 0 && "${FORCE:-0}" != "1" ]]; then
  echo "REFUSING: ${busy} process(es) hold the GPU."
  echo "Latency measured under contention reports queueing, not compute."
  echo "Wait for them to exit, or set FORCE=1 if you accept unreportable timings."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

echo
echo "--- 1/4  timing pass (all re-rankers) ---"
bash scripts/run_rq4_latency.sh || echo "WARN: timing pass returned $?"

echo
echo "--- 2/4  GARDIAN-Lite evaluation + timing ---"
bash scripts/run_rq4_lite_eval.sh || echo "WARN: Lite evaluation returned $?"

echo
echo "--- 3/4  cost tables ---"
"$PY" scripts/report_rq4_cost.py 2>&1 | tee "${LOG_DIR}/report_cost.log"

echo
echo "--- 4/4  figure ---"
"$PY" scripts/plot_reranker_comparison.py \
  --out results/figures/reranker_comparison.png \
  2>&1 | tee "${LOG_DIR}/plot_reranker.log"

echo
echo "=== [$(date -Is)] done ==="
echo "Artifacts:"
echo "  results/rq4_cost_summary.json"
echo "  paper/table_rq4_cost.tex"
echo "  paper/table_rq4_training_cost.tex"
echo "  paper/table_rq4_lite.tex"
echo "  results/figures/reranker_comparison.pdf"
echo
echo "Check that latency_trustworthy is true in results/rq4_cost_summary.json"
echo "before quoting any millisecond figure."
