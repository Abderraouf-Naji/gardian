#!/usr/bin/env bash
# RQ4: LambdaMART control on identical features.
#
# scripts/04_train_gardian.py rebuilds rank_data_<retriever>_train_all.jsonl
# from its per-dataset sources at the start of every training run, so reading
# that file while a training job is starting yields a torn line. This waits for
# the file to stop changing before parsing it.
set -uo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR="logs/rq4"
mkdir -p "$LOG_DIR" results

RETRIEVERS=("hybrid_bm25_faiss" "hybrid_spladepp_medcpt")

wait_until_stable() {
  local f="$1" last=-1 cur quiet=0
  echo "[$(date -Is)] waiting for ${f} to stop changing..."
  while true; do
    cur=$(stat -c %s "$f" 2>/dev/null || echo -1)
    if [[ "$cur" == "$last" && "$cur" != "-1" ]]; then
      quiet=$((quiet + 1))
      # Three consecutive quiet samples (90s) means no writer is active.
      [[ $quiet -ge 3 ]] && break
    else
      quiet=0
    fi
    last="$cur"
    sleep 30
  done
  echo "[$(date -Is)] ${f} stable at ${cur} bytes"
}

for R in "${RETRIEVERS[@]}"; do
  TRAIN="data/${R}/rank_data_${R}_train_all.jsonl"
  wait_until_stable "$TRAIN"
  echo "=== [$(date -Is)] LambdaMART: ${R} ==="
  "$PY" scripts/run_ltr_baseline.py \
    --retriever "$R" \
    --n-jobs 8 \
    --out "results/ltr_baseline_${R}.json" \
    2>&1 | tee "${LOG_DIR}/ltr_${R}.log"
done

echo "=== [$(date -Is)] RQ4 LambdaMART suite done ==="
