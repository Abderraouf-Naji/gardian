import gc
import json
import math
import random
from typing import Dict, List, Optional, Sequence
import os
from pathlib import Path
import time

import numpy as np
import torch
from src.features.schema import active_feature_indices
from src.model.gardian import dropped_features_from_cfg
from src.training.losses import build_loss, is_listwise, pairwise_softplus_margin
from src.common.query_emb_cache import (
    QueryEmbStore,
    load_query_emb_store,
    save_query_emb_cache,
)
import torch.optim as optim
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.features.pool_features import POOL_FEATURE_DIM, pool_features_from_records
from tqdm import tqdm

torch.set_float32_matmul_precision("high")


class StreamingRankDataset(IterableDataset):
    """
    Stream JSONL rank lines grouped by query id.

    Two emission modes, selected by ``mode``:

    ``"pairs"``
        One item per in-query (positive, negative) pair -- what a pairwise
        margin loss consumes.
    ``"groups"``
        One item per query: the whole sampled candidate pool with its labels,
        which is what a listwise objective needs in order to know where the
        nDCG cutoff falls. Pool size is capped at ``group_size``; all
        positives are kept and the remainder is filled by
        :meth:`_sample_negatives`, so hard-negative mixing is identical in
        both modes.
    """

    def __init__(
        self,
        path: str,
        num_negatives: int = 4,
        query_feat_dim: int = 768,
        query_encoder_name: Optional[str] = None,
        query_encoder_device: str = "cpu",
        max_pairs_per_query: Optional[int] = None,
        precompute_query_emb: bool = True,
        query_encoder_batch_size: int = 512,
        query_emb_cache_path: Optional[str] = None,
        query_emb_cache_max_entries: Optional[int] = None,
        hard_negative_top_n: Optional[int] = None,
        hard_negative_fraction: float = 0.0,
        shuffle_buffer: int = 20_000,
        shuffle_seed: int = 42,
        mode: str = "pairs",
        group_size: int = 64,
        dropped_features: Optional[Sequence[str]] = None,
        emit_pool_features: bool = False,
        emit_query_emb: bool = True,
    ):
        if mode not in ("pairs", "groups"):
            raise ValueError(f"mode must be 'pairs' or 'groups', got {mode!r}")
        self.mode = mode
        self.group_size = max(2, int(group_size))
        # Only computed when the controller actually consumes it: this runs per
        # query group inside the DataLoader workers, where data construction is
        # already the throughput bottleneck.
        self.emit_pool_features = bool(emit_pool_features)
        # With the controller disabled (GARDIAN-Lite) the model never reads the
        # query embedding, so loading the cache, precomputing misses and
        # materialising a 768-d vector per group is pure overhead. Gating it
        # here is what makes "no controller means no query encoder" true of the
        # training path as well as the inference path -- otherwise the measured
        # training cost of the Lite arm includes work its model discards.
        self.emit_query_emb = bool(emit_query_emb)
        # Rank JSONL always stores the full 8+8 schema; the model may consume a
        # subset, so columns are selected here rather than in the data.
        self.sparse_cols, self.dense_cols = active_feature_indices(dropped_features)
        self.path = path
        self.num_neg = max(1, int(num_negatives))
        self.query_feat_dim = max(1, int(query_feat_dim))
        self.query_encoder_name = query_encoder_name
        self.query_encoder_device = query_encoder_device
        self.max_pairs_per_query = (
            None if max_pairs_per_query in (None, 0) else max(1, int(max_pairs_per_query))
        )
        self.precompute_query_emb = bool(precompute_query_emb)
        self.query_encoder_batch_size = max(1, int(query_encoder_batch_size))
        self.query_emb_cache_path = query_emb_cache_path
        self._query_encoder = None
        self.hard_negative_top_n = (
            None
            if hard_negative_top_n in (None, 0)
            else max(1, int(hard_negative_top_n))
        )
        self.hard_negative_fraction = min(1.0, max(0.0, float(hard_negative_fraction)))
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.shuffle_seed = int(shuffle_seed)
        # Runtime additions are capped; the preloaded matrix is never evicted
        # (it is shared memory, so eviction would save nothing).
        self._emb_store = QueryEmbStore.empty(
            self.query_feat_dim,
            extra_max=(
                None
                if query_emb_cache_max_entries in (None, 0)
                else max(1, int(query_emb_cache_max_entries))
            ),
        )
        if self.emit_query_emb:
            self._load_query_emb_cache()
            self._maybe_precompute_query_embeddings()
        else:
            logger.info(
                "Controller disabled: skipping query-embedding cache and precompute."
            )

    @property
    def _zero_query_emb(self) -> np.ndarray:
        cached = getattr(self, "_zero_qemb_cache", None)
        if cached is None:
            cached = np.zeros(int(self.query_feat_dim), dtype=np.float32)
            self._zero_qemb_cache = cached
        return cached

    def _emb_cache_get(self, qid: str):
        return self._emb_store.get(qid)

    def _emb_cache_set(self, qid: str, emb_list):
        self._emb_store.set(qid, emb_list)

    def _load_query_emb_cache(self):
        if not self.query_emb_cache_path:
            return
        p = Path(self.query_emb_cache_path)
        if not p.exists():
            return
        self._emb_store = load_query_emb_store(
            p,
            expected_dim=self.query_feat_dim,
            extra_max=self._emb_store._extra_max,
        )

    def _save_query_emb_cache(self):
        if not self.query_emb_cache_path:
            return
        try:
            save_query_emb_cache(
                self.query_emb_cache_path,
                dict(self._emb_store.items()),
                expected_dim=self.query_feat_dim,
            )
        except Exception as e:
            logger.warning(f"Failed to save query_emb cache {self.query_emb_cache_path}: {e}")

    def release_query_encoder(self) -> None:
        """Drop the SentenceTransformer before the DataLoader forks workers.

        Otherwise every worker inherits a copy of the encoder weights it will
        never use once the embedding cache covers the file.
        """
        if self._query_encoder is not None:
            self._query_encoder = None
            gc.collect()

    def _maybe_precompute_query_embeddings(self):
        if not self.precompute_query_emb:
            return
        if len(self._emb_store):
            logger.info("Skipping query_emb precompute (cache already loaded).")
            return
        # Fast probe: if query_emb exists, no need to precompute.
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    q = rec.get("query_emb")
                    if isinstance(q, list) and len(q) == int(self.query_feat_dim):
                        return
                    break
        except Exception:
            return
        if not self.query_encoder_name:
            return
        logger.info("Precomputing missing query_emb values (batched, one-time)...")
        qid_to_question: Dict[str, str] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = rec.get("qid")
                question = rec.get("question")
                if qid is None or not isinstance(question, str):
                    continue
                sqid = str(qid)
                if sqid not in qid_to_question:
                    qid_to_question[sqid] = question
        if not qid_to_question:
            return
        encoder = self._get_query_encoder()
        qids = list(qid_to_question.keys())
        questions = [qid_to_question[qid] for qid in qids]
        for i in tqdm(
            range(0, len(questions), self.query_encoder_batch_size),
            desc="Precompute query_emb",
            unit="batch",
        ):
            batch_q = questions[i : i + self.query_encoder_batch_size]
            embs = encoder.encode(
                batch_q,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=self.query_encoder_batch_size,
            )
            for j, emb in enumerate(embs):
                emb_list = emb.tolist()
                if len(emb_list) != int(self.query_feat_dim):
                    raise ValueError(
                        f"Precomputed query_emb dim mismatch: got={len(emb_list)} expected={self.query_feat_dim}"
                    )
                self._emb_cache_set(qids[i + j], emb_list)
        logger.info(f"Precomputed query_emb for {len(self._emb_store):,} unique queries.")
        self._save_query_emb_cache()
        # Reload so the freshly computed vectors live in one shared matrix
        # rather than as one small array object per query.
        self._load_query_emb_cache()

    def _get_query_encoder(self):
        if self._query_encoder is None:
            if not self.query_encoder_name:
                raise ValueError(
                    "query_emb is missing in rank-data and no query encoder is configured."
                )
            self._query_encoder = SentenceTransformer(
                self.query_encoder_name, device=self.query_encoder_device
            )
        return self._query_encoder

    def _resolve_query_emb(self, rec: Dict):
        if not self.emit_query_emb:
            return self._zero_query_emb
        query_emb = rec.get("query_emb")
        if isinstance(query_emb, list) and len(query_emb) == int(self.query_feat_dim):
            return query_emb

        qid = str(rec.get("qid", ""))
        cached = self._emb_cache_get(qid)
        if cached is not None:
            return cached

        question = rec.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Missing query_emb and question in rank-data record; cannot recompute query embedding."
            )
        encoder = self._get_query_encoder()
        emb = encoder.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        emb_list = emb.tolist()
        if len(emb_list) != int(self.query_feat_dim):
            raise ValueError(
                f"Recomputed query_emb dim mismatch: got={len(emb_list)} expected={self.query_feat_dim}"
            )
        return self._emb_store.set(qid, emb_list)

    def _emit_pairs_for_group(self, group: List[dict]):
        positives = [rec for rec in group if int(rec.get("label", 0)) >= 1]
        negatives = [rec for rec in group if int(rec.get("label", 0)) == 0]
        if not positives or not negatives:
            return

        emitted = 0
        for pos in positives:
            num_negs = min(self.num_neg, len(negatives))
            sampled_negs = self._sample_negatives(negatives, num_negs)
            for neg in sampled_negs:
                if self.max_pairs_per_query is not None and emitted >= self.max_pairs_per_query:
                    return
                yield (
                    record_to_row(
                        pos,
                        query_feat_dim=self.query_feat_dim,
                        query_emb_override=self._resolve_query_emb(pos),
                        sparse_cols=self.sparse_cols,
                        dense_cols=self.dense_cols,
                    ),
                    record_to_row(
                        neg,
                        query_feat_dim=self.query_feat_dim,
                        query_emb_override=self._resolve_query_emb(neg),
                        sparse_cols=self.sparse_cols,
                        dense_cols=self.dense_cols,
                    ),
                )
                emitted += 1

    def _emit_pool_for_group(self, group: List[dict]):
        """
        Yield one query's pool as
        ``(sparse[n,8], dense[n,8], query_emb[768], labels[n], pool_feats[P])``.

        Candidates are shuffled before emission so that pool position, which in
        the rank JSONL still carries the first-stage order, cannot leak the
        label through padding position.

        ``pool_feats`` is computed on the **full** ``group``, before the
        subsampling below. That matters: the pool is cut to ``group_size`` with
        a high hard-negative fraction, so the sampled pool is not the pool the
        model is scored on. Deriving the controller's evidence from the sampled
        pool would train it on a pool composition that never occurs at
        evaluation time.
        """
        positives = [rec for rec in group if int(rec.get("label", 0)) >= 1]
        negatives = [rec for rec in group if int(rec.get("label", 0)) == 0]
        if not positives or not negatives:
            return

        pool_feats = (
            pool_features_from_records(group) if self.emit_pool_features else None
        )

        keep_pos = positives[: self.group_size - 1]
        num_negs = min(self.group_size - len(keep_pos), len(negatives))
        sampled = self._sample_negatives(negatives, num_negs)
        # Keep the stored grade. Every listwise loss uses gain = 2^label - 1, so a
        # 3-level label separates "the gold passage" from "also relevant" instead
        # of asserting they are interchangeable. On binary data (label in {0, 1})
        # this is identical to the previous hard-coded 1.0/0.0.
        records = ([(r, float(int(r.get("label", 1)))) for r in keep_pos]
                   + [(r, 0.0) for r in sampled])
        random.shuffle(records)

        query_emb = self._resolve_query_emb(records[0][0])
        sparse = np.asarray(
            [r["sparse_feats"] for r, _ in records], dtype=np.float32
        )[:, self.sparse_cols]
        dense = np.asarray(
            [r["dense_feats"] for r, _ in records], dtype=np.float32
        )[:, self.dense_cols]
        labels = np.asarray([lab for _, lab in records], dtype=np.float32)
        row = (sparse, dense, np.asarray(query_emb, dtype=np.float32), labels)
        yield row + (pool_feats,) if pool_feats is not None else row

    def _sample_negatives(self, negatives: List[dict], num_negs: int) -> List[dict]:
        """
        Mix hard negatives from the top of the candidate pool with random negatives.

        Rank JSONL rows are query-contiguous and keep the first-stage candidate order.
        Sampling from the top of this list focuses training on top-k errors instead
        of spending most pairs on easy tail negatives.
        """
        if num_negs <= 0:
            return []
        if not self.hard_negative_top_n or self.hard_negative_fraction <= 0:
            return random.sample(negatives, min(num_negs, len(negatives)))

        hard_pool = negatives[: min(self.hard_negative_top_n, len(negatives))]
        n_hard = min(len(hard_pool), int(round(num_negs * self.hard_negative_fraction)))
        n_hard = max(1, n_hard) if hard_pool else 0
        hard = random.sample(hard_pool, n_hard) if n_hard > 0 else []
        hard_ids = {id(x) for x in hard}
        rest_pool = [x for x in negatives if id(x) not in hard_ids]
        n_rest = min(num_negs - len(hard), len(rest_pool))
        rest = random.sample(rest_pool, n_rest) if n_rest > 0 else []
        out = hard + rest
        if len(out) < num_negs:
            remaining = [x for x in negatives if id(x) not in {id(y) for y in out}]
            out.extend(random.sample(remaining, min(num_negs - len(out), len(remaining))))
        random.shuffle(out)
        return out

    def __iter__(self):
        """
        Stream (positive, negative) pairs, sharding by *query group* across workers.

        Rank JSONL rows are query-contiguous. Sharding must therefore happen at
        group boundaries: assigning individual lines round-robin would split one
        query's candidates across workers, so each worker would see a partial
        pool and pair positives against a fraction of the true negatives.

        Pairs then pass through a reservoir shuffle buffer. Without it the model
        sees the datasets in file order -- for the combined file that is 78%
        PubMedQA-artificial followed by 22% MedMCQA, so every epoch is a block
        of one distribution followed by a block of another. The buffer costs
        ``shuffle_buffer`` pairs of memory and removes that structure while
        keeping the stream lazy. ``shuffle_buffer`` is the budget for the whole
        loader, split across workers, so raising ``num_workers`` does not
        multiply resident memory.
        """
        worker_info = get_worker_info()
        num_workers = int(worker_info.num_workers) if worker_info else 1
        worker_id = int(worker_info.id) if worker_info else 0

        def _owns(idx: int) -> bool:
            return num_workers <= 1 or idx % num_workers == worker_id

        emit = (
            self._emit_pool_for_group if self.mode == "groups"
            else self._emit_pairs_for_group
        )

        def _raw_pairs():
            current_qid = None
            current_group: List[dict] = []
            group_idx = 0
            with open(self.path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    qid = rec.get("qid")
                    if current_qid is None:
                        current_qid = qid
                    if qid != current_qid:
                        if _owns(group_idx):
                            yield from emit(current_group)
                        group_idx += 1
                        current_group = [rec]
                        current_qid = qid
                    else:
                        current_group.append(rec)
                    if line_idx % 500_000 == 0:
                        gc.collect()
                if current_group and _owns(group_idx):
                    yield from emit(current_group)

        buf_size = int(self.shuffle_buffer) // max(1, num_workers)
        if buf_size <= 1:
            yield from _raw_pairs()
            return

        rng = random.Random(self.shuffle_seed + worker_id)
        buf: List[tuple] = []
        for item in _raw_pairs():
            if len(buf) < buf_size:
                buf.append(item)
                continue
            j = rng.randrange(buf_size)
            yield buf[j]
            buf[j] = item
        rng.shuffle(buf)
        yield from buf


def record_to_row(
    rec: Dict,
    query_feat_dim: int = 768,
    query_emb_override: Optional[List[float]] = None,
    sparse_cols: Optional[Sequence[int]] = None,
    dense_cols: Optional[Sequence[int]] = None,
) -> tuple:
    """
    One rank-JSONL record as plain Python lists.

    Deliberately returns lists rather than tensors: building a torch tensor per
    field per record meant ~1,536 small allocations per batch, which capped
    throughput at 22 batch/s against a 58 batch/s GPU ceiling. ``collate_fn``
    now converts a whole batch in three array constructions instead.
    """
    query_emb = query_emb_override if query_emb_override is not None else rec.get("query_emb")
    # ndarray views come from the shared QueryEmbStore; lists come from the
    # rank JSONL itself. Both are accepted, an unset/short vector is not.
    if isinstance(query_emb, np.ndarray):
        if query_emb.size != int(query_feat_dim):
            raise ValueError(
                f"query_emb dim mismatch: got={query_emb.size} expected={query_feat_dim}"
            )
    elif not (isinstance(query_emb, list) and len(query_emb) == int(query_feat_dim)):
        raise ValueError(
            "query_emb missing or invalid and no runtime recomputation provided."
        )
    # float32 arrays, not Python lists: a list of 768 floats costs ~18 KB
    # (24 bytes per object) against 3 KB as float32. With a shuffle buffer
    # holding many thousands of pairs that difference decides whether the run
    # fits in RAM.
    sparse = np.asarray(rec["sparse_feats"], dtype=np.float32)
    dense = np.asarray(rec["dense_feats"], dtype=np.float32)
    if sparse_cols is not None:
        sparse = sparse[list(sparse_cols)]
    if dense_cols is not None:
        dense = dense[list(dense_cols)]
    return (sparse, dense, np.asarray(query_emb, dtype=np.float32))


# Backwards-compatible alias for callers/tests that still expect tensors.
def record_to_tensor(
    rec: Dict,
    query_feat_dim: int = 768,
    query_emb_override: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    sf, df, qe = record_to_row(rec, query_feat_dim, query_emb_override)
    return {
        "sparse_feats": torch.tensor(sf, dtype=torch.float32),
        "dense_feats": torch.tensor(df, dtype=torch.float32),
        "query_emb": torch.tensor(qe, dtype=torch.float32),
    }


def collate_fn(batch):
    """
    Stack a batch of (positive, negative) rows into two tensor dicts.

    Three ``np.asarray`` calls per side instead of one ``torch.tensor`` per
    field per record. ``torch.from_numpy`` shares memory, so there is no extra
    copy beyond the array construction itself.
    """
    if batch and isinstance(batch[0][0], dict):        # legacy tensor rows
        pos_list, neg_list = zip(*batch)
        stack = lambda items, key: torch.stack([i[key] for i in items])  # noqa: E731
        return ({k: stack(pos_list, k) for k in pos_list[0]},
                {k: stack(neg_list, k) for k in neg_list[0]})

    pos, neg = zip(*batch)

    def pack(rows):
        sf, df, qe = zip(*rows)
        return {
            "sparse_feats": torch.from_numpy(np.asarray(sf, dtype=np.float32)),
            "dense_feats": torch.from_numpy(np.asarray(df, dtype=np.float32)),
            "query_emb": torch.from_numpy(np.asarray(qe, dtype=np.float32)),
        }

    return pack(pos), pack(neg)


def collate_groups(batch):
    """
    Pad a batch of variable-size candidate pools into ``(B, N, ...)`` tensors.

    Returns ``sparse_feats``/``dense_feats`` ``(B, N, F)``, ``query_emb``
    ``(B, D)``, ``pool_feats`` ``(B, P)``, ``labels`` and ``mask`` ``(B, N)``.
    Padding rows are zero and
    are excluded by ``mask``; every listwise loss in ``src.training.losses``
    honours that mask, so padded columns cannot influence the objective.
    """
    n_max = max(g[0].shape[0] for g in batch)
    b = len(batch)
    sp_dim = batch[0][0].shape[1]
    de_dim = batch[0][1].shape[1]

    sparse = np.zeros((b, n_max, sp_dim), dtype=np.float32)
    dense = np.zeros((b, n_max, de_dim), dtype=np.float32)
    labels = np.zeros((b, n_max), dtype=np.float32)
    mask = np.zeros((b, n_max), dtype=np.float32)
    qemb = np.empty((b, batch[0][2].shape[0]), dtype=np.float32)
    # Present only when the dataset was built with emit_pool_features.
    with_pool = len(batch[0]) >= 5
    pool = np.zeros((b, POOL_FEATURE_DIM), dtype=np.float32) if with_pool else None

    for i, row in enumerate(batch):
        sf, df, qe, lab = row[:4]
        n = sf.shape[0]
        sparse[i, :n] = sf
        dense[i, :n] = df
        labels[i, :n] = lab
        mask[i, :n] = 1.0
        qemb[i] = qe
        if with_pool:
            pool[i] = row[4]

    out = {
        "sparse_feats": torch.from_numpy(sparse),
        "dense_feats": torch.from_numpy(dense),
        "query_emb": torch.from_numpy(qemb),
        "labels": torch.from_numpy(labels),
        "mask": torch.from_numpy(mask),
    }
    if with_pool:
        out["pool_feats"] = torch.from_numpy(pool)
    return out


# The pairwise objective now lives in src/training/losses.py alongside the
# listwise ones, so both arms of the loss ablation are defined in one place and
# selected by name. This alias keeps the old import path working.
bce_loss = pairwise_softplus_margin


class GARDIANTrainer:
    """
    Train GARDIAN with either a pairwise or a listwise objective.

    ``cfg.training.loss`` selects the objective and, with it, the whole input
    pipeline: a listwise loss needs whole candidate pools, so the dataset
    switches to group mode and ``cfg.training.batch_size`` starts counting
    queries rather than pairs. Everything else -- schedule, AMP, clipping,
    early stopping -- is shared, so the two arms of the loss ablation differ
    only in the objective.
    """

    def __init__(self, model, cfg, device: str = "cuda"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self._use_amp = device == "cuda"
        self.epoch_logs: List[Dict] = []
        self.margin = float(getattr(cfg.training, "margin", 0.0) or 0.0)

        self.opt = optim.AdamW(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        self.loss_name = str(getattr(cfg.training, "loss", "pairwise_softplus_margin"))
        self.loss_fn = build_loss(self.loss_name)
        self.listwise = is_listwise(self.loss_name)
        self.group_size = int(getattr(cfg.training, "listwise_group_size", 64))
        self.dropped_features = dropped_features_from_cfg(cfg.model)
        self.loss_kwargs = {}
        if self.listwise:
            self.loss_kwargs["k"] = int(getattr(cfg.training, "listwise_ndcg_k", 10))
            if self.loss_name == "lambdarank":
                self.loss_kwargs["sigma"] = float(
                    getattr(cfg.training, "lambdarank_sigma", 1.0)
                )
            elif self.loss_name == "approxndcg":
                self.loss_kwargs["temperature"] = float(
                    getattr(cfg.training, "approxndcg_temperature", 1.0)
                )
        logger.info(
            f"Objective: {self.loss_name} "
            f"({'listwise, pools of <=' + str(self.group_size) if self.listwise else 'pairwise'})"
        )
        # The controller only reads pool features when configured to; computing
        # them otherwise is pure dataloader cost.
        self.needs_pool_features = "pool_feats" in tuple(
            getattr(cfg.model, "controller_inputs", ("query_emb",)) or ("query_emb",)
        )
        if self.needs_pool_features:
            logger.info("Controller consumes pool features; emitting them per group")
        # With no controller there is no consumer for the query embedding, so
        # the encoder leaves the training path exactly as it leaves the
        # inference path.
        self.needs_query_emb = getattr(model, "controller", None) is not None
        if not self.needs_query_emb:
            logger.info(
                "GARDIAN-Lite: no controller, so no query encoder in the training path"
            )
        self.base_lr = float(cfg.training.lr)
        self.warmup_epochs = max(0, int(getattr(cfg.training, "warmup_epochs", 2)))
        self.min_lr_ratio = float(getattr(cfg.training, "min_lr_ratio", 0.05))

    def forward_batch(self, batch):
        """Score one collated batch of (query, candidate) pairs."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        outputs = self.model(
            sparse_feats=batch["sparse_feats"],
            dense_feats=batch["dense_feats"],
            query_emb=batch["query_emb"],
        )
        scores = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return scores.float()

    def forward_groups(self, batch):
        """
        Score a padded batch of candidate pools, returning ``(B, N)`` scores.

        Pools are passed to the model in grouped ``(B, N, F)`` form rather than
        flattened. The branch MLPs are per-candidate either way, but keeping the
        pool axis is what lets the model standardise each branch within its pool
        (``normalize_branches``) -- flattening would mix candidates from
        different queries into one normalisation.

        One query embedding and one pool-feature vector serve the whole pool,
        which is exactly the property that makes the controller weights constant
        within a query and variable between queries.
        """
        outputs = self.model(
            sparse_feats=batch["sparse_feats"],
            dense_feats=batch["dense_feats"],
            query_emb=batch["query_emb"],
            pool_feats=batch.get("pool_feats"),
            mask=batch.get("mask"),
        )
        scores = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return scores.float()

    def _listwise_step(self, batch):
        """
        One listwise step: pools scored under AMP, objective evaluated in fp32.

        The nDCG-based objectives divide by IDCG and take reciprocal logs of
        rank positions. Those terms are small and are compared across a whole
        pool, so they are kept out of the autocast region -- only the model
        forward runs in half precision.
        """
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            scores = self.forward_groups(batch)
        return self.loss_fn(
            scores.float(), batch["labels"], batch["mask"], **self.loss_kwargs
        )

    def _pairwise_step(self, pos_batch, neg_batch):
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            pos_scores = self.forward_batch(pos_batch)
            neg_scores = self.forward_batch(neg_batch)
        return self.loss_fn(pos_scores.float(), neg_scores.float(), margin=self.margin)

    def _pairs_per_query(self, batch) -> float:
        """
        Mean number of (relevant, non-relevant) pairs per query in ``batch``.

        LambdaRank sums a softplus term over every such pair, so the objective's
        magnitude tracks pool composition -- a PQA-A query with three positives
        in a pool of 64 contributes three times the terms a single-gold MedMCQA
        query does. Dividing by the pair count turns the logged figure into a
        per-pair average that is comparable across datasets and epochs. This is
        for logging only: the optimiser still sees the summed form, because
        gradient clipping at norm 1.0 makes the objective's scale part of the
        training recipe rather than a free constant.
        """
        mask = batch.get("mask") if isinstance(batch, dict) else None
        if not self.listwise or self.loss_name != "lambdarank" or mask is None:
            return 1.0
        with torch.no_grad():
            mask = mask.bool()
            pos = ((batch["labels"] > 0) & mask).sum(dim=1).float()
            neg = mask.sum(dim=1).float() - pos
            pairs = pos * neg
            pairs = pairs[pairs > 0]
            if pairs.numel() == 0:
                return 1.0
            return float(pairs.mean().item())

    def _set_lr_for_epoch(self, epoch: int, total_epochs: int) -> float:
        """Linear warmup then cosine decay to min_lr."""
        base = self.base_lr
        min_lr = max(base * self.min_lr_ratio, 1e-9)
        we = min(int(self.warmup_epochs), max(int(total_epochs) - 1, 0))
        if we > 0 and epoch <= we:
            lr = base * float(epoch) / float(we)
        else:
            n_cos = max(int(total_epochs) - int(we), 1)
            k = int(epoch) - int(we) - 1
            denom = max(n_cos - 1, 1)
            t = min(max(float(k) / float(denom), 0.0), 1.0)
            lr = min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_loss_per_pair = 0.0
        steps = 0
        pbar = tqdm(loader, desc="Training", unit="batch")

        for batch in pbar:
            self.opt.zero_grad(set_to_none=True)
            compute = (
                (lambda: self._listwise_step(batch)) if self.listwise
                else (lambda: self._pairwise_step(batch[0], batch[1]))
            )

            if self._use_amp:
                loss = compute()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss = compute()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

            raw_loss = loss.item()
            per_pair = raw_loss / self._pairs_per_query(batch)
            total_loss += raw_loss
            total_loss_per_pair += per_pair
            steps += 1
            pbar.set_postfix(loss=f"{per_pair:.4f}", sum=f"{raw_loss:.2f}")

            if steps % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return total_loss / max(steps, 1), total_loss_per_pair / max(steps, 1)

    def fit(self, train_path, dev_path, *, on_best=None, on_epoch=None):
        """
        Train, evaluating on dev once per epoch.

        ``on_best(state_dict, dev_ndcg, epoch)`` fires whenever dev nDCG@10
        improves, and ``on_epoch(row)`` after every epoch. The caller uses these
        to persist the best checkpoint and the epoch log *as training runs*:
        previously both were written only after ``fit`` returned, so a run that
        was interrupted -- or simply one you wanted to evaluate early -- left
        nothing on disk.
        """
        from src.evaluation.metrics import evaluate_rank_data

        num_negs = max(1, int(self.cfg.training.num_negatives))
        max_pairs_per_query = getattr(self.cfg.training, "max_pairs_per_query", None)
        hard_negative_top_n = getattr(self.cfg.training, "hard_negative_top_n", None)
        hard_negative_fraction = float(getattr(self.cfg.training, "hard_negative_fraction", 0.0) or 0.0)
        precompute_query_emb = bool(getattr(self.cfg.training, "precompute_query_emb", True))
        query_encoder_batch_size = int(getattr(self.cfg.training, "query_encoder_batch_size", 512))
        query_emb_cache_path = getattr(self.cfg.training, "query_emb_cache_path", None)
        eval_query_emb_cache_path = query_emb_cache_path
        if query_emb_cache_path and str(query_emb_cache_path).endswith("_train_all.pkl"):
            all_cache = str(query_emb_cache_path).replace("_train_all.pkl", "_all.pkl")
            if os.path.exists(all_cache):
                eval_query_emb_cache_path = all_cache
                logger.info(f"Using all-split query_emb cache for evaluation: {all_cache}")
        eval_batch_size = int(getattr(self.cfg.training, "eval_batch_size", 8192))
        raw_max = getattr(self.cfg.training, "query_emb_cache_max_entries", None)
        query_emb_cache_max_entries = (
            int(raw_max) if raw_max is not None else None
        )
        num_workers = int(getattr(self.cfg.training, "num_workers", 4))
        if num_workers < 0:
            num_workers = 0
        if num_workers > 0 and os.cpu_count():
            num_workers = min(num_workers, os.cpu_count())
        train_ds = StreamingRankDataset(
            train_path,
            num_negatives=num_negs,
            query_feat_dim=int(self.cfg.model.query_feat_dim),
            query_encoder_name=str(self.cfg.encoder.model_name),
            query_encoder_device="cpu",
            max_pairs_per_query=max_pairs_per_query,
            precompute_query_emb=precompute_query_emb,
            query_encoder_batch_size=query_encoder_batch_size,
            query_emb_cache_path=query_emb_cache_path,
            query_emb_cache_max_entries=query_emb_cache_max_entries,
            hard_negative_top_n=hard_negative_top_n,
            hard_negative_fraction=hard_negative_fraction,
            shuffle_buffer=int(getattr(self.cfg.training, "shuffle_buffer", 100_000)),
            shuffle_seed=int(getattr(self.cfg, "seed", 42)),
            mode=("groups" if self.listwise else "pairs"),
            group_size=self.group_size,
            dropped_features=self.dropped_features,
            emit_pool_features=self.needs_pool_features,
            emit_query_emb=self.needs_query_emb,
        )
        # Forked workers inherit whatever the parent holds. The embedding cache
        # is a single shared float32 matrix, but the encoder is not needed once
        # that cache covers the file, so drop it before the fork.
        train_ds.release_query_encoder()
        logger.info(
            f"DataLoader | num_workers={num_workers} | "
            f"query_emb store={train_ds._emb_store.nbytes / 1024 ** 3:.2f} GB shared | "
            f"shuffle_buffer={int(getattr(self.cfg.training, 'shuffle_buffer', 20_000)):,} "
            f"{'pools' if self.listwise else 'pairs'} total (split across workers)"
        )
        loader_kwargs = {}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        train_dl = DataLoader(
            train_ds,
            batch_size=int(self.cfg.training.batch_size),
            collate_fn=(collate_groups if self.listwise else collate_fn),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **loader_kwargs,
        )

        self.epoch_logs = []
        best_metric = -1.0
        patience = 0
        best_state = None

        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            cur_lr = self._set_lr_for_epoch(epoch, int(self.cfg.training.epochs))
            epoch_start = time.time()
            logger.info(f"\n{'=' * 50}\nEpoch {epoch}/{self.cfg.training.epochs} (lr={cur_lr:.2e})\n{'=' * 50}")
            train_loss, train_loss_per_pair = self.train_epoch(train_dl)
            dev_ndcg = None
            did_eval = False
            is_best = False
            stopped_early = False

            did_eval = True
            dev_ndcg = evaluate_rank_data(
                self.model,
                dev_path,
                self.device,
                k=10,
                query_encoder_name=str(self.cfg.encoder.model_name),
                query_encoder_device="cpu",
                query_emb_cache_path=eval_query_emb_cache_path,
                batch_size=eval_batch_size,
                dropped_features=self.dropped_features,
            )
            logger.info(
                f"Epoch {epoch:02d} | Loss={train_loss_per_pair:.4f} "
                f"(sum={train_loss:.4f}) | nDCG@10={dev_ndcg:.4f}"
            )
            if dev_ndcg > best_metric:
                best_metric = dev_ndcg
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience = 0
                is_best = True
                logger.info(f"  New best nDCG@10={dev_ndcg:.4f}")
                if on_best is not None:
                    try:
                        on_best(best_state, float(dev_ndcg), int(epoch))
                    except Exception as exc:          # persisting must never kill a run
                        logger.warning(f"  could not persist best checkpoint: {exc}")
            else:
                patience += 1
                if patience >= int(self.cfg.training.early_stopping_patience):
                    logger.info(f"Early stopping after epoch {epoch}")
                    stopped_early = True

            epoch_log = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_loss_per_pair": float(train_loss_per_pair),
                "did_eval": bool(did_eval),
                "dev_ndcg@10": (float(dev_ndcg) if dev_ndcg is not None else None),
                "is_best": bool(is_best),
                "best_ndcg@10_so_far": (float(best_metric) if best_metric >= 0 else None),
                "patience_counter": int(patience),
                "epoch_elapsed_sec": float(time.time() - epoch_start),
                "lr": float(cur_lr),
            }
            self.epoch_logs.append(epoch_log)
            if on_epoch is not None:
                try:
                    on_epoch(epoch_log)
                except Exception as exc:
                    logger.warning(f"  could not persist epoch log: {exc}")

            if stopped_early:
                break

        if best_state:
            self.model.load_state_dict(best_state)

        return best_metric
