"""diagnose_dimensions.py - Run this first to detect actual feature dimensions."""

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.common.question_types import N_QTYPES, ORDERED_QUESTION_TYPES


def detect_feature_dimensions(file_path):
    """Detect actual feature dimensions from rank data file."""
    print(f"\nAnalyzing: {file_path}")
    print("=" * 60)
    
    dimensions = {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:  # Check first 5 samples
                break
            if line.strip():
                data = json.loads(line)
                
                sparse_len = len(data.get("sparse_feats", []))
                dense_len = len(data.get("dense_feats", []))
                kg_len = len(data.get("kg_feats", []))
                qtype_len = len(data.get("qtype_onehot", []))
                
                print(f"\nSample {i+1}:")
                print(f"  sparse_feats: {sparse_len} dimensions")
                print(f"  dense_feats:  {dense_len} dimensions")
                print(f"  kg_feats:     {kg_len} dimensions")
                print(f"  qtype_onehot: {qtype_len} dimensions")
                print(f"  Total features: {sparse_len + dense_len + kg_len}")
                
                dimensions[f"sample_{i+1}"] = {
                    "sparse": sparse_len,
                    "dense": dense_len,
                    "kg": kg_len,
                    "total": sparse_len + dense_len + kg_len
                }
    
    # Get most common dimensions
    if dimensions:
        sparse_dims = [d["sparse"] for d in dimensions.values()]
        dense_dims = [d["dense"] for d in dimensions.values()]
        kg_dims = [d["kg"] for d in dimensions.values()]
        
        print("\n" + "=" * 60)
        print("RECOMMENDED config.yaml settings:")
        print(f"  sparse_feat_dim: {max(set(sparse_dims), key=sparse_dims.count)}")
        print(f"  dense_feat_dim: {max(set(dense_dims), key=dense_dims.count)}")
        print(f"  kg_feat_dim: {max(set(kg_dims), key=kg_dims.count)}")
        print(f"  query_feat_dim: 384  # all-MiniLM-L6-v2 default")
        print(
            f"  question_types: {list(ORDERED_QUESTION_TYPES)} "
            f"(N={N_QTYPES}) — cfg.model.question_types must match this order exactly."
        )

if __name__ == "__main__":
    # Check both files
    detect_feature_dimensions("data/rank_data_pubmedqa_artificial_train.jsonl")
    detect_feature_dimensions("data/rank_data_medmcqa_train.jsonl")