#!/usr/bin/env bash
# Run all 8 cells for Table tab:e2e-ablation (4 families x 2 readers).
# Retrieval: per-dataset indices only (pubmedqa_labeled / medmcqa corpora).
# Do NOT pass --unified-indices; keep cfg.qa.use_per_dataset_indices: true.
# Usage: ./scripts/run_e2e_ablation_cells.sh [cell_name|all]
#   cell examples: bm25_medcpt_llama, spladepp_faiss_biomistral
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

LLAMA="meta-llama/Meta-Llama-3-8B-Instruct"
BIOMISTRAL="BioMistral/BioMistral-7B-DARE"
COMMON=(
  --rq4
  --online-retrieval
  --pubmedqa-open-domain
  --gardian-adaptive-retrieval
  --checkpoint-every-dataset
  --datasets pubmedqa_labeled,medmcqa
  --max-questions 1000
  --systems sparse,dense,hybrid,gardian
)

run_cell() {
  local retriever="$1"
  local reader="$2"
  local slug="$3"
  local extra=("${@:4}")
  local out="results/qa_${slug}.json"
  echo "========== $slug =========="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/06_end_to_end_qa.py \
    "${COMMON[@]}" \
    --retriever "$retriever" \
    --reader-models "$reader" \
    --out "$out" \
    "${extra[@]}"
}

case "${1:-all}" in
  spladepp_faiss_llama)    run_cell hybrid_spladepp_faiss "$LLAMA" "hybrid_spladepp_faiss_Llama-3-8B" --faiss-gpu ;;
  spladepp_faiss_biomistral) run_cell hybrid_spladepp_faiss "$BIOMISTRAL" "hybrid_spladepp_faiss_BioMistral-7B" --faiss-gpu ;;
  spladepp_medcpt_llama)   run_cell hybrid_spladepp_medcpt "$LLAMA" "hybrid_spladepp_medcpt_Llama-3-8B" ;;
  spladepp_medcpt_biomistral) run_cell hybrid_spladepp_medcpt "$BIOMISTRAL" "hybrid_spladepp_medcpt_BioMistral-7B" ;;
  bm25_faiss_llama)        run_cell hybrid_bm25_faiss "$LLAMA" "hybrid_bm25_faiss_Llama-3-8B" --faiss-gpu ;;
  bm25_faiss_biomistral)   run_cell hybrid_bm25_faiss "$BIOMISTRAL" "hybrid_bm25_faiss_BioMistral-7B" --faiss-gpu ;;
  bm25_medcpt_llama)       run_cell hybrid_bm25_medcpt "$LLAMA" "hybrid_bm25_medcpt_Llama-3-8B" ;;
  bm25_medcpt_biomistral)  run_cell hybrid_bm25_medcpt "$BIOMISTRAL" "hybrid_bm25_medcpt_BioMistral-7B" ;;
  all)
    run_cell hybrid_spladepp_faiss "$LLAMA" "hybrid_spladepp_faiss_Llama-3-8B" --faiss-gpu
    run_cell hybrid_spladepp_faiss "$BIOMISTRAL" "hybrid_spladepp_faiss_BioMistral-7B" --faiss-gpu
    run_cell hybrid_spladepp_medcpt "$LLAMA" "hybrid_spladepp_medcpt_Llama-3-8B"
    run_cell hybrid_spladepp_medcpt "$BIOMISTRAL" "hybrid_spladepp_medcpt_BioMistral-7B"
    run_cell hybrid_bm25_faiss "$LLAMA" "hybrid_bm25_faiss_Llama-3-8B" --faiss-gpu
    run_cell hybrid_bm25_faiss "$BIOMISTRAL" "hybrid_bm25_faiss_BioMistral-7B" --faiss-gpu
    run_cell hybrid_bm25_medcpt "$LLAMA" "hybrid_bm25_medcpt_Llama-3-8B"
    run_cell hybrid_bm25_medcpt "$BIOMISTRAL" "hybrid_bm25_medcpt_BioMistral-7B"
    python3 scripts/merge_qa_matrix_json.py results/qa_hybrid_*.json -o results/qa_e2e_matrix_1k.json
    python3 scripts/summarize_qa_matrix_latex.py results/qa_e2e_matrix_1k.json --body-only
    ;;
  *)
    echo "Unknown cell: $1"
    echo "Cells: spladepp_faiss_llama, spladepp_faiss_biomistral, spladepp_medcpt_llama, spladepp_medcpt_biomistral,"
    echo "       bm25_faiss_llama, bm25_faiss_biomistral, bm25_medcpt_llama, bm25_medcpt_biomistral, all"
    exit 1
    ;;
esac
