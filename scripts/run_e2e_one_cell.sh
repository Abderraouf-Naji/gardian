#!/usr/bin/env bash
# Run one E2E QA cell: one hybrid family + one reader LLM.
# Usage:
#   ./scripts/run_e2e_one_cell.sh hybrid_bm25_faiss meta-llama/Meta-Llama-3-8B-Instruct
#   ./scripts/run_e2e_one_cell.sh hybrid_spladepp_medcpt BioMistral/BioMistral-7B-DARE
set -euo pipefail
cd "$(dirname "$0")/.."

RETRIEVER="${1:?retriever, e.g. hybrid_bm25_faiss}"
READER="${2:?reader HF id, e.g. meta-llama/Meta-Llama-3-8B-Instruct}"
OUT_DIR="${OUT_DIR:-results/qa_e2e_cells}"
mkdir -p "$OUT_DIR"

# Safe filename from reader id
READER_SLUG="$(echo "$READER" | tr '/:' '__')"
OUT_JSON="${OUT_DIR}/${RETRIEVER}__${READER_SLUG}.json"

echo "=== Cell: retriever=$RETRIEVER reader=$READER ==="
echo "=== Out: $OUT_JSON ==="

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/06_end_to_end_qa.py \
  --online-retrieval \
  --pubmedqa-open-domain \
  --gardian-adaptive-retrieval \
  --faiss-gpu \
  --checkpoint-every-dataset \
  --datasets pubmedqa_labeled,medmcqa \
  --max-questions 1000 \
  --systems bm25,dense,hybrid,gardian \
  --retriever "$RETRIEVER" \
  --reader-models "$READER" \
  --out "$OUT_JSON"

echo "Done: $OUT_JSON"
echo "Merge all cells: python scripts/merge_qa_matrix_json.py $OUT_DIR -o results/qa_e2e_matrix_1k.json"
