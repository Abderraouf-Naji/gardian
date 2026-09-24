#!/usr/bin/env bash
# RQ4: evaluate and time GARDIAN-Lite on the same pools as the main arm.
#
# Loads the --no-controller checkpoint from results/gardian_lite and runs it
# through the identical evaluation path, so the only difference between these
# numbers and the main GARDIAN row is the controller.
#
# --no-cross-encoders: the cross-encoder rows are unchanged by which GARDIAN
# checkpoint is loaded, so re-scoring them here would burn GPU for nothing.
#
# Run the timing part on an idle GPU (same reason as run_rq4_latency.sh); pass
# MEASURE_LATENCY=0 to collect effectiveness only.
set -uo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
SEED="${SEED:-42}"
LITE_DIR="${LITE_DIR:-results/gardian_lite}"
MEASURE_LATENCY="${MEASURE_LATENCY:-1}"
NQ="${NQ:-200}"
WARMUP="${WARMUP:-10}"
LOG_DIR="logs/rq4"
mkdir -p "$LOG_DIR"

LAT_ARGS=()
if [[ "$MEASURE_LATENCY" == "1" ]]; then
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [[ "$busy" -gt 0 && "${FORCE:-0}" != "1" ]]; then
    echo "WARNING: ${busy} process(es) hold the GPU; Lite latency would not be"
    echo "comparable to the main arm's. Set FORCE=1, or MEASURE_LATENCY=0."
    exit 1
  fi
  LAT_ARGS=(--measure-latency --latency-queries "$NQ" --latency-warmup "$WARMUP")
fi

for R in "hybrid_bm25_faiss" "hybrid_spladepp_medcpt"; do
  CKPT="${LITE_DIR}/seeds/seed_${SEED}/gardian_best_${R}.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "SKIP ${R}: no Lite checkpoint at ${CKPT}"
    continue
  fi

  echo "=== [$(date -Is)] GARDIAN-Lite ${R}: PQA-L + MedMCQA ==="
  "$PY" scripts/14_compare_rerankers.py \
    --retriever "$R" \
    --datasets pubmedqa_labeled,medmcqa \
    --seed "$SEED" \
    --gardian-results-dir "$LITE_DIR" \
    --no-cross-encoders \
    --eval-only \
    "${LAT_ARGS[@]}" \
    --output "results/reranker_comparison_${R}_lite.json" \
    2>&1 | tee "${LOG_DIR}/lite_eval_${R}_main.log"

  echo "=== [$(date -Is)] GARDIAN-Lite ${R}: PQA-A ==="
  "$PY" scripts/14_compare_rerankers.py \
    --retriever "$R" \
    --datasets pubmedqa_artificial \
    --max-queries 1000 \
    --seed "$SEED" \
    --gardian-results-dir "$LITE_DIR" \
    --no-cross-encoders \
    --eval-only \
    "${LAT_ARGS[@]}" \
    --output "results/reranker_comparison_${R}_lite_pqa_a_q1000.json" \
    2>&1 | tee "${LOG_DIR}/lite_eval_${R}_pqa_a.log"
done

echo "=== [$(date -Is)] GARDIAN-Lite evaluation done ==="
