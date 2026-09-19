# Where end-to-end RAG accuracy is actually lost

Measurements behind the reader and metric changes. Regenerate everything with:

```bash
python scripts/diagnose_rag.py --qa-json results/qa_hybrid_bm25_faiss_Llama3-8B_n300.json
```

Output: `results/rag_diagnosis.json`.

## 1. PubMedQA accuracy sits at the majority-class baseline

PubMedQA-labeled (1 000 questions) is 55.2% `yes`, 33.8% `no`, 11.0% `maybe`.
**Answering `yes` to everything scores 0.552.**

| system (300-question eval) | Acc | vs baseline |
|---|---:|---|
| GARDIAN + Llama-3-8B-Instruct | 0.550 | −0.002 |
| GARDIAN + Qwen2.5-14B-Instruct | 0.480 | −0.072 |
| GARDIAN + Qwen2.5-32B-Instruct | 0.577 | +0.025 |

Two of three readers are at or below a constant answer. Any accuracy reported
for this dataset must be printed next to this baseline, which
`src/evaluation/task_baselines.py` now enforces by writing a `baselines` block
and a per-system `delta_vs_majority` into every QA results file.

## 2. The bottleneck is the reader, not retrieval

For GARDIAN + Llama-3-8B on 300 questions:

- at least one gold passage is in context for **99%** of questions
- gold-passage coverage (the paper's `Ctx`) averages **0.87**
- of the questions where gold evidence is present, **42.6% are still answered wrong**

Retrieval is not what is failing on PubMedQA.

Note the two context metrics are different quantities. `Ctx = 0.87` means "87%
of the gold passages were retrieved on average", **not** "13% of questions had
no evidence". The binary companion `gold_evidence_in_context_any` is now
reported alongside it so that reading is unavailable.

## 3. The dominant error is hedging to "maybe"

GARDIAN + Llama-3-8B, gold vs predicted:

| gold \ pred | yes | no | maybe | recall |
|---|---:|---:|---:|---:|
| yes | 117 | 4 | **43** | 0.713 |
| no | 19 | 34 | **32** | 0.400 |
| maybe | 29 | 8 | 14 | 0.275 |

Decidable (gold yes/no) questions answered `maybe`:

| reader | hedged | share | oracle gain if resolved |
|---|---:|---:|---:|
| Llama-3-8B | 75 | 25.0% | +0.250 |
| Qwen2.5-14B | 132 | **44.0%** | +0.440 |
| Qwen2.5-32B | 86 | 28.7% | +0.287 |

Gold `maybe` is 11%; Llama predicts 29.7% and Qwen-14B 44.7%. The hedging rate
explains why a 14B reader scores below an 8B one.

**Cause.** The submitted prompt defines the class as *"maybe: the evidence is
insufficient, mixed, or the key question is not addressed by the passages"* —
broad enough to justify `maybe` almost anywhere, and the reader uses it even
though the gold evidence is present 99% of the time. Verdict parsing is not
implicated: 0% of answers were unparseable.

**The obvious fix does not work.** `scripts/ablate_reader_prompt.py` measures
prompt variants head-to-head with the retrieved passages **frozen**, so the
prompt is the only variable. Over 300 PubMedQA-labeled questions with
Llama-3-8B:

| arm | Acc | CitR | Uns | maybe | unparseable |
|---|---:|---:|---:|---:|---:|
| `v1_permissive` (submitted) | **0.5333** | 0.5285 | 0.1068 | 79 | 7 |
| anti-hedging rewrite | 0.5300 | 0.5363 | 0.1540 | **26** | 27 |

The rewrite did exactly what it was designed to do -- hedging fell from 79 to 26
and "no" predictions rose from 75 to 116 -- and accuracy did not improve. Forced
off the fence, the reader splits roughly evenly between right and wrong, and
unparseable verdicts quadrupled.

**The hedging is a symptom, not the cause.** The reader hedges because it cannot
tell, not because the prompt permits it. The +25 pp figure in the table above is
an upper bound on *perfect* resolution of hedged questions and should not be
read as recoverable headroom. `qa.yesno_prompt` therefore still defaults to
`v1_permissive`; `v2_calibrated` is implemented and runnable but has no
completed 300-question measurement behind it.

This closes off prompt engineering as the route to PubMedQA accuracy. The
remaining levers are the reader itself (a stronger or fine-tuned model) and
accepting that this benchmark, in this setting, is near its practical ceiling
for an 8B reader.

## 4. Citation recall is a coverage problem

The reader emits **1.75** distinct citations against **3.43** gold passages —
coverage 0.51, which reproduces the observed `citation_recall` of ~0.48.

Separately, `qa.reader_max_citations_yesno` was **3**, while 21% of questions
have 4 or more gold passages (max 9). That capped recall structurally. Raised to
**6**, which covers 99% of questions.

## 5. Two metric defects

**Uncited answers scored as perfectly grounded.** `unsupported_claim_rate`
returned `0.0` when an answer cited nothing, so a reader that never cites
achieved a perfect score. 7.3% of Llama-3-8B answers cite nothing. It now
returns `None`, is aggregated only over answers that cited, and
`answer_without_citation_rate` is reported beside it.

**Accuracy and unsupported-citation rate trade off.** In a 12-question pilot,
the decisive prompt moved Acc 0.500 → 0.583 while `Uns` rose 0.175 → 0.275: more
citations means more opportunities to cite a non-gold passage. Both must be
reported together; picking the arm that flatters one metric would be misleading.

## 6. The ceiling is ranking, not first-stage retrieval

`pool_recall@K'` (gold passage present anywhere in the candidate pool) against
the oracle re-ranking bound (perfect ordering of that same pool), seed 42:

| dataset | lost to ranking | lost to pool | dominant |
|---|---:|---:|---|
| pubmedqa_labeled | 0.088 | 0.000 | ranking (all of it) |
| pubmedqa_artificial | 0.225 | 0.004 | ranking (**63.7×**) |
| medmcqa | 0.416 | 0.141 | ranking (**2.9×**) |

**This contradicts the submitted paper.** Section 5.4 claims first-stage
retrieval quality is the limiting factor on MedMCQA; ranking in fact loses 2.9×
more. On PubMedQA, pool recall is exactly 1.000 — *every* loss is ranking.

This is the better story: roughly 40 nDCG@10 points on MedMCQA are recoverable
by re-ranking alone, which motivates the paper's own contribution rather than
excusing its weakest column.
