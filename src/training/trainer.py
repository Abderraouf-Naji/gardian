import gc
import json
import random
from typing import Dict, List, Optional
import os
import pickle
from pathlib import Path
import time

import torch
import torch.optim as optim
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

torch.set_float32_matmul_precision("high")


class StreamingRankDataset(IterableDataset):
    """
    Stream JSONL rank lines grouped by query id, yielding in-query
    (positive, negative) tensor pairs.
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
    ):
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
        self._qid_emb_cache: Dict[str, List[float]] = {}
        self._load_query_emb_cache()
        self._maybe_precompute_query_embeddings()

    def _load_query_emb_cache(self):
        if not self.query_emb_cache_path:
            return
        p = Path(self.query_emb_cache_path)
        if not p.exists():
            return
        try:
            with p.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self._qid_emb_cache = {str(k): v for k, v in data.items()}
                logger.info(
                    f"Loaded query_emb cache: {len(self._qid_emb_cache):,} queries from {p}"
                )
        except Exception as e:
            logger.warning(f"Failed to load query_emb cache {p}: {e}")

    def _save_query_emb_cache(self):
        if not self.query_emb_cache_path:
            return
        p = Path(self.query_emb_cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with p.open("wb") as f:
                pickle.dump(self._qid_emb_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                f"Saved query_emb cache: {len(self._qid_emb_cache):,} queries -> {p}"
            )
        except Exception as e:
            logger.warning(f"Failed to save query_emb cache {p}: {e}")

    def _maybe_precompute_query_embeddings(self):
        if not self.precompute_query_emb:
            return
        if self._qid_emb_cache:
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
                self._qid_emb_cache[qids[i + j]] = emb_list
        logger.info(f"Precomputed query_emb for {len(self._qid_emb_cache):,} unique queries.")
        self._save_query_emb_cache()

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

    def _resolve_query_emb(self, rec: Dict) -> List[float]:
        query_emb = rec.get("query_emb")
        if isinstance(query_emb, list) and len(query_emb) == int(self.query_feat_dim):
            return query_emb

        qid = str(rec.get("qid", ""))
        if qid in self._qid_emb_cache:
            return self._qid_emb_cache[qid]

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
        self._qid_emb_cache[qid] = emb_list
        return emb_list

    def _emit_pairs_for_group(self, group: List[dict]):
        positives = [rec for rec in group if int(rec.get("label", 0)) == 1]
        negatives = [rec for rec in group if int(rec.get("label", 0)) == 0]
        if not positives or not negatives:
            return

        emitted = 0
        for pos in positives:
            num_negs = min(self.num_neg, len(negatives))
            for neg in random.sample(negatives, num_negs):
                if self.max_pairs_per_query is not None and emitted >= self.max_pairs_per_query:
                    return
                yield (
                    record_to_tensor(
                        pos,
                        query_feat_dim=self.query_feat_dim,
                        query_emb_override=self._resolve_query_emb(pos),
                    ),
                    record_to_tensor(
                        neg,
                        query_feat_dim=self.query_feat_dim,
                        query_emb_override=self._resolve_query_emb(neg),
                    ),
                )
                emitted += 1

    def __iter__(self):
        worker_info = get_worker_info()
        current_qid = None
        current_group: List[dict] = []

        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if worker_info and idx % worker_info.num_workers != worker_info.id:
                    continue
                if not line.strip():
                    continue

                rec = json.loads(line)
                qid = rec.get("qid")
                if current_qid is None:
                    current_qid = qid

                if qid != current_qid:
                    yield from self._emit_pairs_for_group(current_group)
                    current_group = [rec]
                    current_qid = qid
                else:
                    current_group.append(rec)

                if idx % 50000 == 0:
                    gc.collect()

            if current_group:
                yield from self._emit_pairs_for_group(current_group)


def record_to_tensor(
    rec: Dict,
    query_feat_dim: int = 768,
    query_emb_override: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    query_emb = query_emb_override if query_emb_override is not None else rec.get("query_emb")
    if not (isinstance(query_emb, list) and len(query_emb) == int(query_feat_dim)):
        raise ValueError(
            "query_emb missing or invalid and no runtime recomputation provided."
        )
    return {
        "sparse_feats": torch.tensor(rec["sparse_feats"], dtype=torch.float32),
        "dense_feats": torch.tensor(rec["dense_feats"], dtype=torch.float32),
        "kg_feats": torch.tensor(rec["kg_feats"], dtype=torch.float32),
        "query_emb": torch.tensor(query_emb, dtype=torch.float32),
        "qtype_onehot": torch.tensor(rec["qtype_onehot"], dtype=torch.float32),
        "kg_coverage": torch.tensor(rec["kg_coverage"], dtype=torch.float32),
    }


def collate_fn(batch):
    pos_list, neg_list = zip(*batch)

    def stack(items, key):
        return torch.stack([i[key] for i in items])

    pos_batch = {k: stack(pos_list, k) for k in pos_list[0].keys()}
    neg_batch = {k: stack(neg_list, k) for k in neg_list[0].keys()}
    return pos_batch, neg_batch


def bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    diff = pos_scores - neg_scores
    return -torch.log(torch.sigmoid(diff) + 1e-8).mean()


class GARDIANTrainer:
    """Pairwise BCE training with mixed precision on CUDA only."""

    def __init__(self, model, cfg, device: str = "cuda"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self._use_amp = device == "cuda"
        self.epoch_logs: List[Dict] = []

        self.opt = optim.AdamW(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        self.loss_fn = bce_loss

    def forward_batch(self, batch):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        outputs = self.model(
            sparse_feats=batch["sparse_feats"],
            dense_feats=batch["dense_feats"],
            kg_feats=batch["kg_feats"],
            query_emb=batch["query_emb"],
            qtype_onehot=batch["qtype_onehot"],
            kg_coverage=batch["kg_coverage"],
        )
        scores = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return scores.float()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(loader, desc="Training", unit="batch")

        for pos_batch, neg_batch in pbar:
            self.opt.zero_grad(set_to_none=True)

            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    pos_scores = self.forward_batch(pos_batch)
                    neg_scores = self.forward_batch(neg_batch)
                    loss = self.loss_fn(pos_scores, neg_scores)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                pos_scores = self.forward_batch(pos_batch)
                neg_scores = self.forward_batch(neg_batch)
                loss = self.loss_fn(pos_scores, neg_scores)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if steps % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return total_loss / max(steps, 1)

    def fit(self, train_path, dev_path):
        from src.evaluation.metrics import evaluate_rank_data

        num_negs = max(1, int(self.cfg.training.num_negatives))
        max_pairs_per_query = getattr(self.cfg.training, "max_pairs_per_query", None)
        precompute_query_emb = bool(getattr(self.cfg.training, "precompute_query_emb", True))
        query_encoder_batch_size = int(getattr(self.cfg.training, "query_encoder_batch_size", 512))
        query_emb_cache_path = getattr(self.cfg.training, "query_emb_cache_path", None)
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
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=int(self.cfg.training.batch_size),
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )

        self.epoch_logs = []
        best_metric = -1.0
        patience = 0
        best_state = None

        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            epoch_start = time.time()
            logger.info(f"\n{'=' * 50}\nEpoch {epoch}/{self.cfg.training.epochs}\n{'=' * 50}")
            train_loss = self.train_epoch(train_dl)
            dev_ndcg = None
            did_eval = False
            is_best = False
            stopped_early = False

            if epoch % 2 == 0 or epoch == 1:
                did_eval = True
                dev_ndcg = evaluate_rank_data(
                    self.model,
                    dev_path,
                    self.device,
                    k=10,
                    query_encoder_name=str(self.cfg.encoder.model_name),
                    query_encoder_device="cpu",
                    query_emb_cache_path=getattr(self.cfg.training, "query_emb_cache_path", None),
                )
                logger.info(
                    f"Epoch {epoch:02d} | Loss={train_loss:.4f} | nDCG@10={dev_ndcg:.4f}"
                )
                if dev_ndcg > best_metric:
                    best_metric = dev_ndcg
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                    is_best = True
                    logger.info(f"  New best nDCG@10={dev_ndcg:.4f}")
                else:
                    patience += 1
                    if patience >= int(self.cfg.training.early_stopping_patience):
                        logger.info(f"Early stopping after epoch {epoch}")
                        stopped_early = True
            else:
                logger.info(f"Epoch {epoch:02d} | Loss={train_loss:.4f}")

            epoch_log = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "did_eval": bool(did_eval),
                "dev_ndcg@10": (float(dev_ndcg) if dev_ndcg is not None else None),
                "is_best": bool(is_best),
                "best_ndcg@10_so_far": (float(best_metric) if best_metric >= 0 else None),
                "patience_counter": int(patience),
                "epoch_elapsed_sec": float(time.time() - epoch_start),
            }
            self.epoch_logs.append(epoch_log)

            if stopped_early:
                break

        if best_state:
            self.model.load_state_dict(best_state)

        return best_metric
