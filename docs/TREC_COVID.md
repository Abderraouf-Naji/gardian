# TREC-COVID (BEIR) — optional IR evaluation

## Why this dataset

ECIR reviewers may ask why results are not compared on standard biomedical IR
benchmarks with **public qrels**. Among BEIR biomedical options:

| Dataset | Why / why not |
|---|---|
| **TREC-COVID** (chosen) | NIST pooled judgments (~493 / query). Strongest answer to incomplete-relevance critiques. Free BEIR zip. ~171k docs — indexable beside existing corpora. |
| NFCorpus | Public qrels, but automatic link-based grades; tiny corpus (~3.6k). Weaker rebuttal. |
| BioASQ | Not in the public BEIR download set; ~15M docs — heavy and gated. |

TREC-COVID is **eval-only**. It is never mixed into `train_all` / `dev_all` and
never merged into the unified 3M index, so existing PubMedQA / MedMCQA artifacts
stay untouched.

## Safety guarantees

- `--dataset all` on scripts `01` / `03` / `05` still means the paper QA sets only.
- Request TREC-COVID explicitly: `--dataset trec_covid` / `--datasets trec_covid`.
- New files only: `data/corpus_trec_covid.jsonl`, `data/trec_covid_test.jsonl`,
  `data/indices/*/trec_covid/`, `data/hybrid_*/rank_data_*_trec_covid_*.jsonl`.
- Query-cache merge does **not** overwrite `query_emb_cache_*_{all,train_all}.pkl`
  when generating TREC-COVID alone.

## Labels vs BEIR leaderboard

GARDIAN uses **binary-gain** nDCG (every qrel with grade ≥ 1 is a positive).
Published BEIR tables often use **graded** gains. Report GARDIAN numbers as
binary-gain nDCG on TREC-COVID, and note the distinction if citing BEIR
leaderboard figures.

## Commands

```bash
cd ~/gardian
source .venv/bin/activate

# 1) Download + convert (additive JSONL only)
python scripts/00b_download_trec_covid.py

# 2) Indices for trec_covid only (does not rebuild unified / other corpora)
python scripts/01_build_bm25_faiss_indices.py --dataset trec_covid
python scripts/01_build_spladepp_medcpt_indices.py --dataset trec_covid

# 3) Rank data for one hybrid family (repeat for the other three if needed)
python scripts/03_generate_rank_data.py \
  --retriever hybrid_bm25_faiss --dataset trec_covid

# 4) Evaluate an existing checkpoint on TREC-COVID only
python scripts/05_evaluate_gardian.py \
  --retriever hybrid_bm25_faiss --datasets trec_covid --seeds 42
```

After step 1 you should see ~50 test queries and mean positives/query on the
order of hundreds (NIST pooling).
