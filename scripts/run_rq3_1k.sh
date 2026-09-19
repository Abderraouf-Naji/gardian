#!/usr/bin/env bash
# RQ3 end-to-end matrix at n=1000.
#
#   2 back-ends x 2 readers x 3 datasets, RRF (hybrid) vs GARDIAN only.
#
# Scope decisions behind this file:
#   * systems=hybrid,gardian -- the sparse/dense channel rows double reader cost
#     and feed the ablation, not RQ3. --rq4 no longer overrides an explicit
#     --systems (scripts/06_end_to_end_qa.py).
#   * seed 42 -- reader decoding is greedy (do_sample=False), so the only
#     stochastic component in an E2E cell is the GARDIAN checkpoint. The 5-seed
#     spread is reported on the retrieval tables, which cost no reader time.
#   * n=1000 -- at n=300 a 2-3 pp accuracy delta cannot be resolved (paired
#     bootstrap p=0.19-0.48 on the previous runs).
#
# Resumable: --checkpoint-every-dataset writes after each dataset, and a cell
# whose output already covers the requested systems/questions is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
N=${N:-1000}
SEED=${SEED:-42}
OUTDIR=results/rq3_1k
LOGDIR=logs/rq3_1k
mkdir -p "$OUTDIR" "$LOGDIR"

declare -a BACKENDS=(hybrid_bm25_faiss hybrid_spladepp_medcpt)
declare -a READERS=("meta-llama/Meta-Llama-3-8B-Instruct:Llama3-8B" "Qwen/Qwen2.5-14B-Instruct:Qwen14B")

for be in "${BACKENDS[@]}"; do
  for entry in "${READERS[@]}"; do
    model="${entry%%:*}"; tag="${entry##*:}"
    out="$OUTDIR/qa_${be}_${tag}_n${N}.json"
    log="$LOGDIR/${be}_${tag}.log"
    if [[ -f "$out" ]]; then
      echo "[skip] $out exists"
      continue
    fi
    echo "[run ] $be x $tag -> $out"
    "$PY" scripts/06_end_to_end_qa.py \
      --rq4 --systems hybrid,gardian \
      --online-retrieval --pubmedqa-open-domain \
      --gardian-adaptive-retrieval \
      --checkpoint-every-dataset \
      --datasets pubmedqa_labeled,pubmedqa_artificial,medmcqa \
      --max-questions "$N" \
      --retriever "$be" \
      --reader-models "$model" \
      --faiss-cpu \
      --top-k-passages 10 \
      --seed "$SEED" \
      --out "$out" > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[FAIL] $be x $tag (exit $rc) -- see $log"
    else
      echo "[done] $out"
    fi
  done
done
echo "matrix complete"
