# What changed since the CoopIS submission

Reviewers rejected the CoopIS version on four grounds: single seed, missing
fusion baselines, unclear features, and gains too small to distinguish from
noise. This document records what was measured and changed in response. Every
number here is reproducible from the scripts named beside it.

---

## 1. The central claim did not survive measurement

The submitted paper claimed query-adaptive sparse--dense fusion as its
contribution. Six independent tests say the controller contributes almost
nothing, and on the one collection with real relevance judgments it is harmful.

| test | result |
| --- | ---: |
| controller ablated to fixed 0.5/0.5 (old checkpoint) | +0.0007 |
| z-normalising branch outputs before fusion | no effect (oracle 0.6153 -> 0.6151) |
| LightGBM given the query embedding directly (no saturating gradient) | +0.0049 |
| TREC-COVID, leave-one-out prediction of oracle alpha | r=+0.42, but **worse** than a constant |
| isolation ladder, PubMedQA-artificial | **+0.0025** (4.5% of headroom) |
| isolation ladder, MedMCQA | **+0.0052** (4.6% of headroom) |
| isolation ladder, TREC-COVID (real NIST qrels) | **-0.0280** |

The controller is not broken -- its outputs vary (mean 0.49, sd 0.24, range
[0.11, 0.87]). There is simply no per-query signal worth exploiting on these
pools. Two structural reasons, both measured:

* **The synthetic benchmarks cannot show an effect.** In PubMedQA and MedMCQA
  the gold passage is derived from the question's own record, so there is
  exactly one gold per query and nDCG@10 is a step function of where it lands.
  55--57% of the alpha grid ties the per-query maximum and 28--31% of queries
  are completely alpha-invariant.
* **Where the alpha surface *is* sharp** (TREC-COVID: 25% ties, 4% invariant),
  the controller trained on the synthetic splits transfers badly and loses to a
  single cross-validated weight.

## 2. What actually produces the gains

`scripts/ablate_fusion_ladder.py` -- each rung changes exactly one thing.

| rung | PubMedQA-art. | MedMCQA | TREC-COVID |
| --- | ---: | ---: | ---: |
| 1 sum, unnormalised | 0.6500 | 0.4345 | 0.6294 |
| 2 RRF (k=60) | 0.7331 | 0.4734 | 0.6368 |
| 3 weighted RRF (tuned) | 0.7471 | 0.4784 | 0.6622 |
| 4 global-alpha (tuned) | 0.7517 | 0.4856 | 0.6541 |
| **5 + learned branches** | **0.7944** | **0.5026** | **0.7274** |
| 6 + per-query controller | 0.7970 | 0.5078 | 0.6994 |
| 7 [ceiling] oracle-alpha | 0.8509 | 0.6147 | 0.8100 |
| **value of learned branches (5-4)** | **+4.27** | **+1.70** | **+7.33** |
| value of adaptivity (6-5) | +0.25 | +0.52 | **-2.80** |

Learned branch scoring over within-pool-normalised statistics is the
contribution. Per-query weighting is not.

## 3. Training objective and learning rate

The submitted model used `softplus(gamma - (r_pos - r_neg))`. That objective
saturates: measured over 34,152 real training pairs, the median value of the
gating term `sigmoid(gamma - diff)` is **0.0092**, so the median pair supplies
under 1% of the gradient available at the margin boundary. Dev nDCG@10 peaked
at epoch 1 and then declined.

Replaced with a listwise LambdaRank objective (`src/training/losses.py`), which
weights every comparison by the nDCG@k change of swapping it. Also swept the
learning rate, which was a transformer fine-tuning value applied to
from-scratch MLPs over tabular features.

| objective (MedMCQA-only, best lr) | test nDCG@10 |
| --- | ---: |
| pairwise softplus | 0.5037 |
| approxNDCG | 0.5023 (diverges at 1e-3) |
| lambdarank | 0.5070 |
| listnet | 0.5081 |

Objective alone is worth +0.003--0.004; the learning rate (3e-5 -> **1e-3**) is
the larger half. Both are reported separately rather than credited to the loss.

**Result on the real training regime** (combined splits, 10 epochs):
dev nDCG@10 **0.6786 -> 0.7114** (BM25+FAISS), **0.7192** (BM25+MedCPT), and no
mid-training collapse.

## 4. Retrieval results against the strongest tuned baseline

Seed 42, BM25+FAISS. Every tuned quantity fitted on dev (or by 5-fold CV where
a collection has no dev split), never on the reported split.

| benchmark | best baseline | GARDIAN | margin |
| --- | ---: | ---: | ---: |
| PubMedQA-Labeled | 0.8922 (dense) | 0.9074 | +1.5 |
| PubMedQA-Artificial | 0.7300 (global-a) | 0.7932 | **+6.3** |
| MedMCQA | 0.3977 (global-a) | 0.4256 | +2.8 |
| TREC-COVID (real qrels, zero-shot) | 0.6553 (global-a, CV) | 0.6994 | **+4.4** |

Against the CoopIS table: PubMedQA-Artificial **76.0 -> 79.3**, MedMCQA
**41.6 -> 42.6**.

## 5. Features

Reviewers asked what the features are. `src/features/schema.py` is now the
single machine-checkable answer: 8 sparse + 8 dense, each dimension named and
documented, with `assert_feature_dims` failing loudly if config, data and
schema disagree.

Feature-count ablation (`--dropped-features` selects columns at tensor-build
time, so a feature set can change **without regenerating rank data**):

| feature set | MedMCQA | PubMedQA |
| --- | ---: | ---: |
| **16 (all)** | **0.5126** | **0.8058** |
| 14 (drop retrieval indicators) | 0.5116 | 0.8063 |
| 8 (best-8 by tree gain) | 0.5031 | 0.7827 |
| 7 (CoopIS set) | 0.5004 | 0.7975 |
| 4 (normalisations only) | 0.4867 | 0.7578 |
| 2 (raw scores only) | 0.4914 | 0.7592 |

All 16 are kept. A methodological finding worth reporting: the four
within-pool normalisations carry **77% of tree-attributed gain** yet score
*below* the two raw scores (which carry 1.7%) when used alone. Selecting
features by importance ranking here would delete the wrong ones.

## 6. Baselines added

The submitted paper compared only against Sum and RRF. Now, all fitted on dev
and reported on the same query population as GARDIAN:

* **Global-alpha** -- grid-searched over 101 points on dev, applied at test.
* **Group-alpha** -- one alpha per question type (only non-vacuous on MedMCQA;
  PubMedQA is 100% yes/no by construction).
* **Weighted RRF** -- channel weight and k both tuned.
* **Oracle-alpha** -- per-query best alpha from test labels. Reported as a
  *ceiling*, never a competitor, and only at the cutoff it optimises.
* **Oracle re-ranking / pool_recall** -- shows the ranking headroom, which is
  2.3x the alpha headroom on every benchmark.
* **LambdaMART** on the identical 16 features (`scripts/run_ltr_baseline.py`),
  which lands slightly *above* the neural model and must be reported. On the
  gold-in-pool population it scores 0.8065 / 0.5092; on the full population
  used by section 4 and the paper tables it scores 0.8041 (PubMedQA-art.),
  0.4274 (MedMCQA) and 0.9093 (PubMedQA-Labeled), against GARDIAN's 0.7932 /
  0.4256 / 0.9074. See section 10 -- the two populations are not
  interchangeable.

## 7. Removed

* **Knowledge-graph code** -- there is no KG; the code was dead.
* **Question-type model input** -- the one-hot is constant within each
  benchmark, so it acted as a dataset indicator rather than a query signal.
  Retained as an analysis dimension only (`docs/QUESTION_TYPE.md`).
* **`no_qtype` ablation** -- outlived the input it named and would have raised
  at evaluation time.
* Two broken, unimported modules (`src/training/rank_data.py`,
  `src/pipeline/online_feature_cache.py`).

## 8. Reproducibility and correctness

* **Multi-seed protocol** (`src/common/seeds.py`): per-seed artifact
  directories, a guard that refuses to overwrite completed results, and
  aggregation into mean/std.
* **Query-embedding caching in evaluation** -- previously every evaluation
  re-encoded the whole split with a BERT pass per query while a 234k-query
  cache sat unused on disk (18,218 needless encodes per PubMedQA run).
* **Memory** 19 GB -> 2.47 GB; **speed** 82 -> ~9 min/epoch.
* **181 tests** (was 96), including regression tests for five bugs found
  during this work: an oracle row that scored *below* the system it bounds at
  cutoffs it does not optimise; a Group-alpha row that silently reported
  Global-alpha; a stale ablation name that would have crashed a paper run; a
  latency benchmark that encoded the query *outside* its own timer, so the
  controller's cost was invisible; and `load_checkpoint_state` dereferencing
  `model.controller` unconditionally, which made every GARDIAN-Lite checkpoint
  unloadable.

## 9. Open items

Done since this list was written:

* **Multi-seed.** All four back-ends are trained at all five seeds (13, 21, 42,
  87, 100); `results/aggregated/training_summary.json` carries mean/std for
  every cell. The SPLADE++ back-ends are included.
* **LambdaMART is reproducible again.** `scripts/run_ltr_baseline.py` was cited
  in `docs/OBJECTIVE.md` but never committed; it now exists and trains on the
  full 16.4M-row pool in ~8 min on CPU.

Still open:

* GARDIAN-Lite (`--no-controller`, 4.25M params, no query encoder): training
  and evaluation are in progress at seed 42 on all four back-ends
  (`results/gardian_lite/`, `scripts/run_rq4_lite_eval.sh`). Not yet
  multi-seed.
* A flat MLP over all 16 concatenated features -- the "why two branches?"
  control -- is not yet run.
* Paired-bootstrap significance on the retrieval tables.
* Real-judgment collections beyond TREC-COVID (NFCorpus, SciFact).

## 10. A note on comparing numbers across documents

Two evaluation populations appear in these documents and they are not
interchangeable:

* **Full population** -- every query in the split, with per-query pool labels
  where no qrels entry exists. This is what `src/evaluation/rank_jsonl_eval.py`
  reports, and what section 4 above and the paper tables use.
* **Gold-in-pool population** -- only queries whose gold passage is in the
  candidate pool. This is what the tables in `docs/OBJECTIVE.md` section 3a
  use, and it reads 5-8 points higher on MedMCQA.

Quoting a gold-in-pool number beside a full-population one overstates the gap.
`scripts/run_ltr_baseline.py` reports the full population, so its output is
directly comparable to the paper tables and *not* to `docs/OBJECTIVE.md`
section 3a.
