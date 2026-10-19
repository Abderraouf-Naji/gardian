#!/usr/bin/env bash
# Rerun the BM25+FAISS end-to-end GARDIAN arm after the live-feature fix.
#
# The original BM25+FAISS runs built GARDIAN's IDF-weighted overlap feature
# without the corpus IDF table (src/features/idf.py explains the bug). RRF
# never reads that feature, so its answers in results/rq3_1k/ stay valid and
# only the GARDIAN arm is rerun here. SPLADE++ +MedCPT is not rerun: its model
# ignores this feature (live and offline MedMCQA Hit@10 70.2 vs 70.3).
#
# Same protocol as scripts/run_rq3_1k.sh except --systems gardian, and output
# goes to a separate directory so the original files are never overwritten.
#
#   READERS="Qwen/Qwen2.5-7B-Instruct:Qwen7B" bash scripts/run_rq3_rerun_bm25_gardian.sh
#
# Qwen2.5-32B does not fit the A40 in fp16, so it runs in 4-bit NF4. The old
# 32B RRF answers were fp16 and are incomplete, so RRF is rerun as well to keep
# the paired comparison on the same reader:
#   READERS="Qwen/Qwen2.5-32B-Instruct:Qwen32B" SYSTEMS=hybrid,gardian \
#   EXTRA_ARGS="--load-in-4bit" bash scripts/run_rq3_rerun_bm25_gardian.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
N=${N:-1000}
SEED=${SEED:-42}
BE=hybrid_bm25_faiss
SYSTEMS=${SYSTEMS:-gardian}
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
OUTDIR=results/rq3_1k_fixidf
LOGDIR=logs/rq3_1k_fixidf
mkdir -p "$OUTDIR" "$LOGDIR"

read -r -a READER_LIST <<< "${READERS:-Qwen/Qwen2.5-7B-Instruct:Qwen7B Qwen/Qwen2.5-14B-Instruct:Qwen14B}"

for entry in "${READER_LIST[@]}"; do
  model="${entry%%:*}"; tag="${entry##*:}"
  out="$OUTDIR/qa_${BE}_${tag}_n${N}.json"
  log="$LOGDIR/${BE}_${tag}.log"
  if [[ -f "$out" ]] && grep -q '"medmcqa"' "$out"; then
    echo "[skip] $out already has all datasets"
    continue
  fi
  echo "[run ] $BE x $tag (systems=$SYSTEMS ${EXTRA[*]:-}) -> $out   $(date '+%F %T')"
  "$PY" scripts/06_end_to_end_qa.py \
    --rq4 --systems "$SYSTEMS" "${EXTRA[@]}" \
    --online-retrieval --pubmedqa-open-domain \
    --gardian-adaptive-retrieval \
    --checkpoint-every-dataset \
    --datasets pubmedqa_labeled,pubmedqa_artificial,medmcqa \
    --max-questions "$N" \
    --retriever "$BE" \
    --reader-models "$model" \
    --faiss-cpu \
    --top-k-passages 10 \
    --seed "$SEED" \
    --out "$out" > "$log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $BE x $tag (exit $rc) -- see $log   $(date '+%F %T')"
  else
    echo "[done] $out   $(date '+%F %T')"
  fi
done
echo "rerun complete   $(date '+%F %T')"
