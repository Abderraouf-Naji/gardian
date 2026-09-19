"""
Evaluation: retrieval metrics, rank-file scoring, end-to-end QA, statistics.

  metrics          binary-gain nDCG / recall / hit / MRR -- the canonical
                   definitions; every other module imports these rather than
                   re-implementing them
  rank_jsonl_eval  scores all systems from a rank JSONL in one pass
  qa_eval          end-to-end RAG evaluation (accuracy + grounding)
  task_baselines   majority-class baseline attached to every reported accuracy
  qtype_breakdown  per-question-type analysis (reporting only)
  stats            bootstrap CIs and paired significance tests
"""
