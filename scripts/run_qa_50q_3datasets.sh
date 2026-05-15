#!/usr/bin/env bash
# 50 questions × 3 datasets × 3 systems (llm_only, hybrid, gardian).
# Do NOT use --quick-qa for PubMedQA yes/no (it drives ~50% "maybe" answers).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
OUT="${OUT:-results/qa_50q_3datasets.json}"
LOG="${LOG:-results/qa_50q_3datasets.log}"

exec "$PYTHON" scripts/06_end_to_end_qa.py \
  --retriever hybrid_bm25_faiss \
  --datasets pubmedqa_labeled,pubmedqa_artificial,medmcqa \
  --max-questions 50 \
  --systems llm_only,hybrid,gardian \
  --top-k-passages 10 \
  --max-new-tokens 512 \
  --reader-max-input-length 4096 \
  --max-chars-per-passage 600 \
  --query-emb-cache "data/query_emb_cache_hybrid_bm25_faiss_all.pkl,data/query_emb_cache_hybrid_bm25_faiss_pubmedqa_labeled_eval.pkl,data/query_emb_cache_hybrid_bm25_faiss_pubmedqa_artificial_test.pkl,data/query_emb_cache_hybrid_bm25_faiss_medmcqa_test.pkl" \
  --bootstrap 200 \
  --out "$OUT" \
  2>&1 | tee "$LOG"
