# Question-type conditioning: measurements and decision

`model.use_question_type` in `configs/base.yaml` controls whether the controller
input is `h_q` or `[h_q || one-hot(type)]`. It defaults to **false**. This file
records the measurements behind that default so the choice can be defended, and
reversed, on evidence rather than on the reviewers' summary of one table.

Reproduce every number here with:

```bash
python scripts/analyze_question_types.py
```

## 1. The one-hot is constant on two of the three benchmarks

Distinct question types per evaluation set (unique queries, `hybrid_bm25_faiss`
rank data):

| dataset | queries | distinct types | composition |
|---|---:|---:|---|
| pubmedqa_labeled | 1 000 | **1** | `yesno` 100% |
| pubmedqa_artificial | 18 218 | **1** | `yesno` 100% |
| medmcqa | 5 094 | 5 | `factoid` 78.4%, `diagnosis` 9.7%, `mechanism` 6.1%, `treatment` 5.3%, `contraindication` 0.5% |

PubMedQA is entirely yes/no by construction. On both PubMedQA splits the one-hot
is a constant vector: it carries zero information and can only add parameters.

**Consequence for the published ablation.** Table 2 compares NoQType against the
full model in six columns. Four of them are PubMedQA columns, where the feature
is mathematically incapable of helping. The ablation is therefore informative in
**two** cells (MedMCQA under two back-ends), not six — one tie and one win at a
single seed. The reviewers read a null result; the experiment was underpowered.

## 2. The one-hot encodes dataset identity

Across every rank file:

| dataset | share of queries tagged `yesno` |
|---|---:|
| pubmedqa_labeled | 100% |
| pubmedqa_artificial | 100% |
| medmcqa | 0% |

Training concatenates all three datasets into one file
(`rank_data_*_train_all.jsonl`). In that combined set the `yesno` bit separates
PubMedQA from MedMCQA **perfectly**. A controller given this feature can learn
"if `yesno`, apply the PubMedQA policy, otherwise the MedMCQA policy".

This is the strongest argument against keeping the feature, and it is not one the
reviewers made: a component sold as *query*-adaptive is partly *dataset*-adaptive.
It also explains the observed ablation pattern — the full model wins exactly
where the bit is informative about the dataset (MedMCQA) and loses where the bit
is constant noise (PubMedQA).

## 3. One declared category is never used

`model.question_types` declares seven categories. Counts over all rank data:

| category | queries |
|---|---:|
| yesno | 14 676 |
| factoid | 5 884 |
| diagnosis | 716 |
| mechanism | 449 |
| treatment | 346 |
| contraindication | 40 |
| **other** | **0** |

`other` never fires, so the conditioned controller carries a permanently-zero
input dimension. `contraindication` is supported by 40 queries.

## 4. Inference-time check

Evaluating the existing seed-42 `hybrid_bm25_faiss` checkpoint on MedMCQA with
the one-hot columns dropped from the controller's first layer:

| variant | nDCG@10 |
|---|---:|
| with one-hot (as trained) | 0.416352 |
| one-hot columns dropped | 0.416943 |

All four first-stage baselines are bit-identical across the two runs, so the
difference is attributable to the controller alone. This is **not** a clean
comparison — the checkpoint was trained *with* the feature — and it is reported
only as a consistency check. The decisive comparison is a five-seed retrain of
both variants.

## 5. Decision

Ship `use_question_type: false` as the main system, and keep the conditioned
variant runnable via the flag.

- The claim in Section 3.3 survives unchanged: the controller still adapts per
  query. What changes is that type-dependence is *discovered* from the query
  embedding rather than supplied as a label.
- The keyword heuristic that assigns clinical types disappears from the model,
  and with it the reviewers' question about how PubMedQA categories were
  constructed.
- Reporting keeps question type as an analysis dimension: the per-type
  breakdown and the α-by-type export are unaffected, since both read the label
  from the data rather than from the model.

## 6. Reporting rules for the ablation

1. Report the NoQType comparison **only on MedMCQA**, and state explicitly that
   PubMedQA cannot test it because the feature is constant there.
2. Run both variants across all five seeds and report mean ± std with a paired
   test, rather than a single-seed delta.
3. Pair the result with the linear probe on the frozen PubMedBERT `[CLS]`
   embedding: if the embedding already predicts question type, the one-hot is
   redundant by construction, which is the mechanism behind the null result.
4. If MedMCQA shows a robust significant gain across seeds, report it as a
   conditional observation — explicit typing helps only where queries are
   heterogeneous — while shipping the simpler model as the main system.
