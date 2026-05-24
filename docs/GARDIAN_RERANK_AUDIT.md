# GARDIAN rerank vs E2E QA — audit (for fixes)

**RQ4 (end-to-end impact):** see **`docs/RQ4_END_TO_END.md`** for what GARDIAN is, which metrics to report, and the canonical `06` command.

## Symptom

- **Offline retrieval** (`05_evaluate_gardian.py` / `rank_jsonl_eval.py`): GARDIAN often **beats** RRF / hybrid on **nDCG@10**, **Hit@10**, **MRR**.
- **End-to-end QA** (`06_end_to_end_qa.py`): **Hybrid RAG** often **beats** GARDIAN on **answer accuracy**, even with a stronger reader (Qwen-32B).

This is **not automatically a bug** — retrieval rank quality ≠ reader success — but this repo had **protocol mismatches** that made the gap look “absurd” when the paper claims the same top-$k$ stack.

---

## Root causes (verified in code)

### 1. Different tasks (inherent)

| Retrieval eval | E2E QA |
|----------------|--------|
| Is gold passage in **top-$k$ of ranked list**? | Does **LLM** output correct yes/no / MCQ given **$k$ passages**? |
| Trained with **pairwise BCE** on scalar scores | Reader adds noise, prompt effects, citation format |
| Full list ranking (~100 candidates) | Only **10 passages** in context (after fix; was **6**) |

**Implication:** GARDIAN can improve rank@10 and still lose QA if the reader fails or distractors in top-10 hurt.

### 2. Reader top-$k$ ≠ retrieval @10 (fixed)

- Paper text said top-$k{=}10$; `configs/base.yaml` had `top_k_passages: 6`.
- **Fix:** `top_k_passages` / `yesno_top_k_passages` / `mcq_top_k_passages` = **10**.

### 3. Hybrid reader used “balanced” slots (fixed)

- `hybrid_balanced_top_k: true` gave Hybrid **5 sparse + 5 dense** from the pool, not **top-10 by RRF**.
- GARDIAN used **top-10 by `gardian_score`** → unfair RQ4 comparison.
- **Fix:** `hybrid_balanced_top_k: false`, `rq4_align_retrieval_top_k: true` → Hybrid = **RRF top-10**, GARDIAN = **rerank top-10**.

### 4. Offline GARDIAN scored a **subset** of the pool (fixed)

In `rank_jsonl_eval.py`, with `gardian_adaptive_retrieval=true`, GARDIAN only scored  
`subset_rank_records_adaptive()` (~top-$\alpha$ sparse + top-$\beta$ dense), while **RRF ranked the full ~100 candidates**.

Gold can be **high under RRF** but **excluded from the subset** (weak single channel) → retrieval comparison biased.

**Fix:** Score **all** candidates in rank JSONL for GARDIAN; adaptive weights are diagnostic only, no pre-subset.

### 5. QA pool ≠ rank-training pool (fixed for RQ4)

| Source | Pool |
|--------|------|
| Rank data (`03_generate_rank_data.py`) | `hybrid.retrieve()` → **50+50 union**, up to ~100 |
| QA (old, adaptive on) | `retrieve_adaptive_candidates_live()` → **smaller** $\alpha$-weighted union |
| QA (new, `rq4_full_union_pool: true`) | `retrieve_hybrid_candidates(pool_k=100)` → **same as training** |

**Fix:** `qa.rq4_full_union_pool: true` (default in `configs/base.yaml`) uses full union for E2E when aligning with retrieval tables.

### 6. “Sparse / Dense” rows in adaptive QA are not standalone retrievers

With adaptive pool, **all systems share one pool**. Sparse = top-$k$ by BM25 **within that pool**; not a full-corpus BM25 run. Same for dense. Only meaningful as **channel ablations on the shared pool**.

### 7. Train / eval index scope

- Rank JSONL may be built on one index layout; QA uses `qa.use_per_dataset_indices` + open-domain.
- Ensure rank data for each `hybrid_*` family matches QA indices (per-dataset vs unified).

---

## Files to hand to another assistant (priority order)

### Core logic (edit first)

1. `src/evaluation/rank_jsonl_eval.py` — retrieval metrics, RRF vs GARDIAN scoring (**subset fix applied**).
2. `src/evaluation/qa_eval.py` — live QA pools, passage selection, citations, aggregates.
3. `src/pipeline/gardian_adaptive.py` — adaptive retrieve, `subset_rank_records_adaptive` (offline only).
4. `src/model/gardian.py` — fusion, controller, `rerank()`.
5. `src/pipeline/rag_reader.py` — reader load, RAG prompts, `retrieve_hybrid_candidates`.
6. `configs/base.yaml` — `top_k_passages`, `rq4_*`, `gardian_adaptive_retrieval`, pool caps.

### Scripts

7. `scripts/06_end_to_end_qa.py` — CLI, checkpoint, matrix runs.
8. `scripts/05_evaluate_gardian.py` — offline retrieval table.
9. `scripts/03_generate_rank_data.py` — training pool = hybrid union ~100.
10. `scripts/backfill_hit_metrics.py` — Hit@k backfill via `evaluate_all_from_rank_data`.

### Features / training

11. `src/features/sparse.py` — 3-d sparse features.
12. `src/features/dense_feat.py` — 4-d dense features.
13. `src/training/rank_data.py` — JSONL format.
14. `src/training/trainer.py` — pairwise BCE, dev nDCG@10.

### Tests

15. `tests/test_qa_reader_top_k.py` — RRF vs GARDIAN top-$k$ selection.
16. `tests/test_rank_extra_baselines.py` — rank JSONL baselines.

---

## Recommended experiment protocol (500q, 3 LLMs)

```bash
python scripts/06_end_to_end_qa.py \
  --rq4 \
  --online-retrieval --pubmedqa-open-domain \
  --gardian-adaptive-retrieval \
  --checkpoint-every-dataset \
  --datasets pubmedqa_labeled,medmcqa --max-questions 500 \
  --retriever hybrid_spladepp_medcpt \
  --reader-models <LLM> \
  --load-in-4bit   # if 32B/70B \
  --faiss-cpu \
  --top-k-passages 10 \
  --fresh \
  --out results/qa_<family>_<llm>_n500_v3.json
```

**Check logs:**

```text
QA reader top-k=10 | hybrid_balanced=False | rq4_align_retrieval=True | pool=full_union_rrf
```

Re-run **retrieval** after rank_jsonl fix:

```bash
python scripts/05_evaluate_gardian.py --retriever hybrid_spladepp_medcpt ...
```

---

## Paper wording (honest)

- Report retrieval on **full candidate pool** (Hit@10, nDCG@10, MRR).
- Report E2E on **top-10 passages** after RRF vs GARDIAN rerank, **same union pool** as rank-data when `rq4_full_union_pool` is on.
- Do **not** claim “GARDIAN improves QA because it improves Hit@10” unless you measure **gold-in-reader-context rate** (diagnostic script optional).

---

## Changes applied in this audit (2026-05-20)

- `rank_jsonl_eval.py`: GARDIAN scores **full pool**, not adaptive subset.
- `qa_eval.py`: `rq4_full_union_pool` → `retrieve_hybrid_candidates` for RQ4.
- `configs/base.yaml`: top-$k$=10, `rq4_full_union_pool: true`.
- `supported_citation_rate` in QA aggregates.
- `gold_evidence_in_context_rate` diagnostic in QA aggregates.
- `--rq4` flag locks protocol cfg; `--fresh` skips stale checkpoints.
- `docs/RQ4_END_TO_END.md` — RQ4 metrics and GARDIAN definition.
