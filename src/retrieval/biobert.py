"""BioBERT dense retriever for medical QA."""

import json
import torch
import numpy as np
from typing import List, Dict, Optional
from loguru import logger
import pathlib
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    BIOBERT_AVAILABLE = True
except ImportError:
    BIOBERT_AVAILABLE = False
    logger.warning("BioBERT requires transformers: pip install transformers")


class BioBERTRetriever:
    """BioBERT dense retriever using mean pooling."""
    
    def __init__(
        self,
        index_path: str,
        checkpoint: str = "dmis-lab/biobert-v1.1",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        if not BIOBERT_AVAILABLE:
            raise ImportError("Transformers not available. Run: pip install transformers")
        
        self.index_path = pathlib.Path(index_path)
        self.checkpoint = checkpoint
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        logger.info(f"Loading BioBERT from {checkpoint} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        try:
            # Transformers now blocks unsafe torch.load paths on torch<2.6.
            # For compatible checkpoints (including BioBERT on HF), force
            # safetensors to avoid falling back to legacy .bin loading.
            self.model = AutoModel.from_pretrained(checkpoint, use_safetensors=True)
        except Exception as e:
            raise RuntimeError(
                "Failed to load BioBERT with safetensors. "
                "Ensure the checkpoint provides .safetensors files, or upgrade "
                "torch to >=2.6 if legacy .bin loading is required."
            ) from e
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for index
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        
        # Try to load existing index
        self._load_index()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling on top of token embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode texts to dense embeddings."""
        embeddings = []
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Pooling
            pooled = self._mean_pooling(outputs, inputs["attention_mask"])
            
            # Normalize
            pooled = F.normalize(pooled, p=2, dim=1)
            
            embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def build_index(
        self,
        corpus_path: str,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Build BioBERT index from corpus JSONL."""
        if output_dir:
            self.index_path = pathlib.Path(output_dir)
        
        index_file = self.index_path / "biobert_index.pt"
        
        if index_file.exists() and not overwrite:
            logger.info(f"Index already exists at {index_file}, loading...")
            self._load_index()
            return
        
        logger.info(f"Building BioBERT index from {corpus_path}")
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Load passages
        passages = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    passages.append((data["id"], data["text"]))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {line_num}: {e}")
        
        if not passages:
            logger.error(f"No valid passages found in {corpus_path}")
            return
        
        self.doc_ids = [p[0] for p in passages]
        self.doc_texts = [p[1] for p in passages]
        
        # Encode all passages
        logger.info(f"Encoding {len(passages)} passages...")
        self.doc_embeddings = self.encode_texts(self.doc_texts, show_progress=True)
        
        # Save index
        self._save_index(index_file)
        
        logger.success(f"BioBERT index built: {len(self.doc_ids)} docs, embedding dim={self.doc_embeddings.shape[1]}")
    
    def _save_index(self, index_file: pathlib.Path):
        """Save index to disk."""
        if self.doc_embeddings is None:
            logger.error("No embeddings to save")
            return
        
        torch.save({
            'doc_ids': self.doc_ids,
            'doc_texts': self.doc_texts,
            'doc_embeddings': self.doc_embeddings,
            'checkpoint': self.checkpoint,
        }, index_file)
        logger.info(f"Saved index to {index_file}")
    
    def _load_index(self):
        """Load index from disk."""
        index_file = self.index_path / "biobert_index.pt"
        
        if not index_file.exists():
            logger.info(f"No existing index found at {index_file}")
            return
        
        try:
            data = torch.load(index_file, map_location='cpu')
            self.doc_ids = data['doc_ids']
            self.doc_texts = data['doc_texts']
            self.doc_embeddings = data['doc_embeddings']
            logger.info(f"Loaded index with {len(self.doc_ids)} docs from {index_file}")
        except Exception as e:
            logger.warning(f"Failed to load index from {index_file}: {e}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 200,
        return_scores: bool = True,
    ) -> List[Dict]:
        """Retrieve passages using dense dot product."""
        if self.doc_embeddings is None or len(self.doc_ids) == 0:
            logger.warning("Index not built or empty")
            return []
        
        # Encode query
        query_embedding = self.encode_texts([query])
        
        if query_embedding.shape[0] == 0:
            return []
        
        # Compute similarity scores
        scores = np.dot(self.doc_embeddings, query_embedding.T).flatten()
        
        # Get top-k
        if top_k > len(scores):
            top_k = len(scores)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                "id": self.doc_ids[idx],
                "text": self.doc_texts[idx],
                "biobert_score": float(scores[idx]),
                "rank": len(results) + 1,
            })
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 200,
    ) -> List[List[Dict]]:
        """Retrieve for multiple queries."""
        if self.doc_embeddings is None or len(self.doc_ids) == 0:
            return [[] for _ in queries]
        
        # Encode all queries
        query_embeddings = self.encode_texts(queries)
        
        # Compute all similarity scores
        scores = np.dot(self.doc_embeddings, query_embeddings.T)
        
        # Get top-k for each query
        results = []
        for i in range(len(queries)):
            query_scores = scores[:, i]
            top_indices = np.argsort(query_scores)[-top_k:][::-1]
            
            query_results = []
            for idx in top_indices:
                query_results.append({
                    "id": self.doc_ids[idx],
                    "text": self.doc_texts[idx],
                    "biobert_score": float(query_scores[idx]),
                    "rank": len(query_results) + 1,
                })
            results.append(query_results)
        
        return results