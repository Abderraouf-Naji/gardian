# gardian-reranker

Query-adaptive sparse/dense/KG reranker for RAG systems.

## Architecture Docs

- `docs/ARCHITECTURE_ASTAR.md` (A*-conference architecture package)
- `docs/REVIEWER_GUIDE.md` (code-grounded reviewer map)

## Install

```bash
pip install .
```

After publishing to PyPI:

```bash
pip install gardian-reranker
```

## What it returns

`gardian-reranker` returns:
- query-level adaptive branch weights: sparse/dense/kg
- ranked passages with final score
- per-passage branch contributions for explainable routing into the reader

## Python API

```python
from gardian_reranker import GardianReranker, Candidate, QueryFeatures

reranker = GardianReranker.from_checkpoint("gardian_best.pt", {
    "sparse_dim": 3,
    "dense_dim": 4,
    "kg_dim": 6,
    "branch_hidden": 256,
    "controller_hidden": 256,
    "query_feat_dim": 1024,
    "n_qtypes": 7,
    "dropout": 0.2
})

query_features = QueryFeatures(
    query_emb=[0.0] * 1024,
    qtype_onehot=[0, 0, 0, 0, 0, 0, 1],
    kg_coverage=0.8
)

candidates = [
    Candidate(
        id="doc1",
        text="Passage text",
        sparse_feats=[0.1, 0.2, 0.3],
        dense_feats=[0.4, 0.5, 0.6, 0.7],
        kg_feats=[0.1, 0.2, 0.3, 0.1, 0.0, 0.4],
    )
]

result = reranker.rerank(query_features, candidates)
print(result["query_weights"])
print(result["ranked"][0]["contrib"])
```

## CLI

```bash
gardian-rerank \
  --checkpoint results/gardian_best_hybrid.pt \
  --config config.json \
  --input input.json \
  --output output.json
```

## PyPI release (maintainers)

1. Replace placeholder repository URLs in `pyproject.toml`.
2. Configure [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) 
3. Bump `version` in `pyproject.toml`.
4. Push a tag, e.g. `v0.1.1`:

```bash
git tag v0.1.1
git push origin v0.1.1
```

The GitHub Action `.github/workflows/publish-pypi.yml` will build and publish automatically.
