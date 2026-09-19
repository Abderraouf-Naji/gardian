# Why the training objective changed

Short version: the submitted model was trained with a pairwise margin loss that
stops producing gradient long before the ranking stops improving, at a learning
rate 33x below optimal. Retrained listwise at the measured learning rate,
GARDIAN scores **0.8015 on PubMedQA-artificial** and **0.5057 on MedMCQA** --
**+5.0 and +2.0 nDCG@10 over dev-tuned Global-alpha**, and within half a point
of a LambdaMART on the same features. The query-adaptive controller, which the
submission claimed as its contribution, is worth **<= +0.005** and is not the
source of any of it.

Everything below is measured, not argued. Reproduce with
`scripts/diagnose_objective.py` and `scripts/run_ltr_baseline.py`.

---

## 1. The margin loss saturates

The submitted objective is `softplus(gamma - (r_pos - r_neg))` with `gamma = 1`.
Writing `r = a*s_sp + (1-a)*s_de` and `diff = r_pos - r_neg`:

```
dL/da = -sigmoid(gamma - diff) * (D_sp - D_de)
```

where `D_sp`, `D_de` are the two branches' positive-minus-negative differences.
Every gradient reaching the controller passes through the gate
`sigmoid(gamma - diff)`, which closes as the branches learn to separate
positives from negatives.

Measured on 34,152 real training pairs, epoch-1 checkpoint, MedMCQA test:

| quantity | value |
| --- | ---: |
| `r_pos - r_neg` | mean 5.33, median 5.68 |
| `sigmoid(gamma - diff)` | mean 0.181, **median 0.0092** |
| pairs with gate < 0.05 | **63.8%** |

The median pair supplies under 1% of the gradient available at the margin
boundary. This is also why dev nDCG@10 *fell* from epoch 1 (0.6786) to epoch 2
(0.6715): the branches keep sharpening, so the gate keeps closing.

## 2. The controller contributes ~nothing — four independent tests

| test | result |
| --- | ---: |
| ablate controller to fixed 0.5/0.5 (`uniform_alpha`) | **+0.0007** |
| z-normalise branch outputs before fusion (scale-confound hypothesis) | 0.4945 -> 0.4933, oracle 0.6153 -> 0.6151 — **no effect** |
| give LightGBM the query embedding directly (no saturating gradient, free to use it) | **+0.0049**, 6.3% of total gain |
| TREC-COVID, leave-one-out prediction of the per-query oracle alpha | r = +0.42 but nDCG 0.6442 < **0.6554** Global-alpha — *worse than a constant* |

The fourth row is the decisive one. TREC-COVID is the only collection here with
real pooled NIST judgments, and it is the regime where adaptivity *should* pay:
its per-query alpha surface is genuinely peaked (4% alpha-invariant queries,
0.25 grid-tie fraction) where the synthetic collections are flat (29% and 0.56).
Even there, a predictor that correlates with the oracle at r = +0.42 still loses
to a single tuned constant, because the alpha surface is peaked enough that
prediction error costs more than adaptivity gains.

Alpha is not collapsed and the controller is not broken — its outputs have mean
0.493, sd 0.239, range [0.106, 0.872]. There is simply no per-query signal worth
exploiting on these pools.

### Why the synthetic collections cannot show an effect

In PubMedQA and MedMCQA the gold passage is derived from the question's own
record (`pq_art_<pmid>_<i>` are sentences of the abstract the question was
generated from; `mcq_<uuid>_explanation` is that question's explanation). There
is exactly one gold per query, so nDCG@10 is a step function of where that
single document lands, and 56% of the alpha grid ties the maximum. There is
nothing for a per-query controller to learn.

## 3. What the gain actually comes from

### 3a. The headline: GARDIAN retrained

All trained on the combined PubMedQA+MedMCQA pools, evaluated on the
gold-in-pool population of each test split. Global-alpha is tuned *on test*, so
it is if anything flattered.

| method | MedMCQA (4,269 q) | PubMedQA-art. (18,132 q) |
| --- | ---: | ---: |
| BM25 only | 0.4297 | 0.6451 |
| FAISS only | 0.3913 | 0.7129 |
| RRF (k=60) | 0.4734 | 0.7331 |
| Global-alpha (*tuned on test*) | 0.4861 | 0.7517 |
| GARDIAN, pairwise softplus, lr 3e-5 (CoopIS) | 0.4952 | 0.7576 |
| LambdaMART on the same 16 features | 0.5092 | 0.8065 |
| **GARDIAN, lambdarank, lr 1e-3** | **0.5057** | **0.8015** |
| Oracle-alpha (*ceiling, uses test labels*) | 0.5873 | 0.8119 |

**+2.0 / +5.0 nDCG@10 over the strongest tuned non-adaptive baseline**, and
within 0.4-0.5 of the gradient-boosted ranker. The neural model does not need to
be replaced by trees.

### 3b. How much of that is the objective alone

Two effects are confounded in the CoopIS checkpoint: the objective *and* a
learning rate 33x too low. Isolating the objective on MedMCQA-only training,
everything else identical (12 epochs, same init, same pools, best dev epoch):

| objective | 3e-5 | 1e-4 | 3e-4 | 1e-3 |
| --- | ---: | ---: | ---: | ---: |
| pairwise softplus | 0.5020 | 0.5014 | **0.5037** | 0.5001 |
| lambdarank | 0.5056 | 0.5054 | 0.5066 | **0.5070** |
| approxndcg | 0.4963 | 0.5014 | 0.5023 | 0.4374 |
| listnet | 0.5065 | 0.5038 | 0.5061 | **0.5081** |

At its own best learning rate the listwise objective is worth **+0.003 to
+0.004** over the pairwise one -- real and consistent, but small. The larger
part of the improvement over the CoopIS checkpoint is the learning rate. Both
are reported rather than attributing the whole gap to the loss.

On the combined training set lambdarank beats listnet at every setting
(0.8015 vs 0.7968 on PubMedQA), which is why it is the configured default.
``approxndcg`` diverges at 1e-3 and is not recommended.

### 3c. Learning rate

Swept over {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} x {lambdarank, listnet} x 2 seeds.
3e-5 -- a transformer fine-tuning rate applied to from-scratch MLPs over tabular
features -- underfits badly. 1e-3 is the optimum; 3e-3 diverged in 3 of 4
configurations.

## 4. Oracle-alpha is a ceiling, not a beatable baseline

It selects alpha per query *using the test labels*. It is reported as a
diagnostic, and no honest system is expected to exceed it. It is also not
reachable with these features: pushing LightGBM to 255 leaves, adding 15
cross-channel interaction features, and adding a 64-dimensional query-embedding
PCA all plateau at ~0.52 on MedMCQA.

| variant | MedMCQA test |
| --- | ---: |
| 16 features, moderate capacity | 0.5150 |
| 16 features, 255 leaves | 0.5088 |
| + 15 cross-channel interactions | 0.5080 |
| + query PCA-64 | 0.5136 |
| + query PCA-32 (best) | **0.5199** |

## 5. How many features are needed

Every subset retrained and evaluated identically; only the columns differ.

| feature set | MedMCQA | PubMedQA | delta MedMCQA | delta PubMedQA |
| --- | ---: | ---: | ---: | ---: |
| **16 (all)** | **0.5126** | **0.8058** | -- | -- |
| 14 (drop `retrieved_by_*`) | 0.5116 | 0.8063 | -0.0010 | +0.0005 |
| 12 (drop indicators + dispersion) | 0.5080 | 0.8030 | -0.0046 | -0.0028 |
| 8 (best-8 by tree gain) | 0.5031 | 0.7827 | -0.0095 | -0.0231 |
| 7 (CoopIS submission set) | 0.5004 | 0.7975 | -0.0122 | -0.0083 |
| 6 (normalisations + ranks) | 0.4868 | 0.7593 | -0.0258 | -0.0466 |
| 4 (normalisations only) | 0.4867 | 0.7578 | -0.0260 | -0.0480 |
| 2 (raw scores only) | 0.4914 | 0.7592 | -0.0212 | -0.0466 |

All 16 are kept. Dropping the two retrieval indicators is a wash in every
measurement (LambdaMART -0.0010/+0.0005; GARDIAN -0.0008/-0.0010 over 2 seeds
and 2 objectives), so there is no evidence for removing them. Cutting harder is
not a wash: the best 8 features lose 0.0095/0.0231.

### Importance is not necessity

LambdaMART gain attribution over the 16 features:

| feature | gain |
| --- | ---: |
| `sp_minmax` | 34.9% |
| `sp_z` | 21.1% |
| `de_minmax` | 15.2% |
| `de_z` | 5.6% |
| `sp_idf` | 5.4% |
| all others | 17.8% |
| -- of which raw `sp_score` + `de_score` | 1.7% |
| -- of which `retrieved_by_*` | 0.2% |

The four within-pool normalisations carry **77% of tree gain** -- yet on their
own they score 0.4867/0.7578, *below* the two raw scores alone (0.4914/0.7592)
which carry 1.7%. Gain measures how often a tree splits on a feature, not
whether the information exists nowhere else; selecting features by importance
ranking here would delete the wrong ones. The normalisations still matter --
this is a statement about attribution, not about their value.

## 6. What changed in the code

* `src/training/losses.py` — new. `lambdarank`, `approxndcg`, `listnet`, plus
  the original `pairwise_softplus_margin` retained so the loss ablation runs
  both arms through one code path.
* `src/training/trainer.py` — `StreamingRankDataset` gained a `"groups"` mode
  emitting whole candidate pools; `collate_groups` pads them with a mask;
  `GARDIANTrainer` selects the objective, the dataset mode and the collate
  function from `cfg.training.loss`.
* `configs/base.yaml` — `training.loss: lambdarank`, plus
  `listwise_group_size`, `listwise_ndcg_k`, `lambdarank_sigma`,
  `approxndcg_temperature`. In listwise mode `batch_size` counts **queries**,
  not pairs.

Hard-negative sampling is shared between the two modes, so the loss ablation
compares objectives and not candidate distributions.
