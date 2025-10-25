#!/usr/bin/env python3
"""
Evaluation script for BGE-M3 embedding model
Computes retrieval metrics: MRR, Recall@K, Precision@K

Usage:
    python evaluate_model.py
    python evaluate_model.py --checkpoint checkpoints/bgem3_projection_best.pt
    python evaluate_model.py --data data/test-set.json --examples 10
"""

import json
import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict

# Auto-detect project root
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import BGEM3WithHead


def resolve_path(path_str):
    """Resolve path relative to script directory"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint"""
    checkpoint_path = resolve_path(checkpoint_path)
    print(f"📦 Loading model from: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = BGEM3WithHead(d_out=128, freeze_encoder=True)
    
    # Handle both state_dict and full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    print(f"✅ Model loaded successfully")
    return model


def compute_similarity_matrix(
    queries: torch.Tensor, 
    documents: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity matrix
    
    Args:
        queries: [N_queries, D]
        documents: [N_docs, D]
    
    Returns:
        similarities: [N_queries, N_docs]
    """
    # Already L2-normalized, so dot product = cosine similarity
    return queries @ documents.T


def compute_mrr(similarities: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank
    
    Args:
        similarities: [N_queries, N_docs] - similarity scores
        labels: [N_queries] - index of correct document for each query
    
    Returns:
        MRR score (0 to 1, higher is better)
    """
    # Get ranking of documents for each query (descending order)
    ranks = torch.argsort(similarities, dim=1, descending=True)
    
    reciprocal_ranks = []
    for i, label in enumerate(labels):
        # Find position of correct document
        rank_pos = (ranks[i] == label).nonzero(as_tuple=True)[0].item()
        reciprocal_rank = 1.0 / (rank_pos + 1)  # +1 because 0-indexed
        reciprocal_ranks.append(reciprocal_rank)
    
    return np.mean(reciprocal_ranks)


def compute_recall_at_k(
    similarities: torch.Tensor, 
    labels: torch.Tensor, 
    k: int = 10
) -> float:
    """
    Compute Recall@K: % of queries where correct doc is in top-K
    
    Args:
        similarities: [N_queries, N_docs]
        labels: [N_queries]
        k: Top-K to consider
    
    Returns:
        Recall@K score (0 to 1)
    """
    # Get top-K documents for each query
    topk_indices = torch.topk(similarities, k=min(k, similarities.size(1)), dim=1).indices
    
    correct = 0
    for i, label in enumerate(labels):
        if label in topk_indices[i]:
            correct += 1
    
    return correct / len(labels)


def compute_precision_at_k(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10
) -> float:
    """
    Compute Precision@K
    
    For our task (1 correct doc per query), Precision@K = Recall@K / K
    """
    recall = compute_recall_at_k(similarities, labels, k)
    return recall / k


def evaluate_on_dataset(
    model: BGEM3WithHead,
    test_data: List[Dict],
    device: str = "cuda",
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained model
        test_data: List of {query, pos, ...}
        device: Device to use
        batch_size: Batch size for encoding
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n📊 Evaluating on {len(test_data)} examples...")
    
    # Extract queries and positives
    queries = [item['query'] for item in test_data]
    positives = [item['pos'] for item in test_data]
    
    # Encode in batches
    print("🔄 Encoding queries...")
    query_embs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="Queries"):
        batch = queries[i:i+batch_size]
        with torch.no_grad():
            embs = model(batch, device=device)
        query_embs.append(embs)
    query_embs = torch.cat(query_embs, dim=0)
    
    print("🔄 Encoding documents...")
    doc_embs = []
    for i in tqdm(range(0, len(positives), batch_size), desc="Documents"):
        batch = positives[i:i+batch_size]
        with torch.no_grad():
            embs = model(batch, device=device)
        doc_embs.append(embs)
    doc_embs = torch.cat(doc_embs, dim=0)
    
    # Compute similarities
    print("🔄 Computing similarities...")
    similarities = compute_similarity_matrix(query_embs, doc_embs)
    
    # Labels: diagonal (query i matches doc i)
    labels = torch.arange(len(test_data), device=device)
    
    # Compute metrics
    print("🔄 Computing metrics...")
    metrics = {
        'MRR': compute_mrr(similarities, labels),
        'Recall@1': compute_recall_at_k(similarities, labels, k=1),
        'Recall@5': compute_recall_at_k(similarities, labels, k=5),
        'Recall@10': compute_recall_at_k(similarities, labels, k=10),
        'Recall@50': compute_recall_at_k(similarities, labels, k=50),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Pretty print metrics"""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    for metric, value in metrics.items():
        percentage = value * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{metric:12s}: {percentage:6.2f}% │{bar}│")
    
    print("="*60)
    
    # Interpretation
    print("\n💡 Interpretation:")
    mrr = metrics['MRR']
    if mrr > 0.8:
        print("   🟢 Excellent! Model retrieves correct docs very accurately.")
    elif mrr > 0.6:
        print("   🟡 Good! Model performs well on most queries.")
    elif mrr > 0.4:
        print("   🟠 Fair. Model needs improvement.")
    else:
        print("   🔴 Poor. Consider retraining with different settings.")
    
    recall10 = metrics['Recall@10']
    print(f"\n   In top-10 results: {recall10*100:.1f}% queries find correct match")


def evaluate_examples(
    model: BGEM3WithHead,
    test_data: List[Dict],
    device: str = "cuda",
    n_examples: int = 5
):
    """Show example predictions"""
    print("\n" + "="*60)
    print("🔍 EXAMPLE PREDICTIONS")
    print("="*60)
    
    for i, item in enumerate(test_data[:n_examples]):
        query = item['query']
        positive = item['pos']
        
        # Get hard negatives if available
        hard_negs = []
        if 'hard_neg' in item:
            hard_negs = [hn['text'] if isinstance(hn, dict) else hn 
                        for hn in item['hard_neg'][:3]]
        
        # Encode
        with torch.no_grad():
            q_emb = model([query], device=device)
            candidates = [positive] + hard_negs
            c_embs = model(candidates, device=device)
            
            # Similarities
            sims = (q_emb @ c_embs.T).squeeze().cpu().numpy()
        
        # Print
        print(f"\n📝 Example {i+1}:")
        print(f"   Query: {query[:80]}...")
        print(f"\n   Similarities:")
        print(f"   ✅ Positive:    {sims[0]:.4f} - {positive[:60]}...")
        
        for j, (neg, sim) in enumerate(zip(hard_negs, sims[1:])):
            print(f"   ❌ Negative {j+1}:  {sim:.4f} - {neg[:60]}...")
        
        # Check if positive has highest score
        if len(sims) > 1:
            if sims[0] == max(sims):
                print(f"   ✅ Correct! Positive has highest similarity")
            else:
                print(f"   ❌ Wrong! A negative has higher similarity")
        
        # Show margin
        if len(sims) > 1:
            margin = sims[0] - max(sims[1:])
            print(f"   📊 Margin: {margin:+.4f} (higher is better)")


def main():
    """Main evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained BGE-M3 model")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/bgem3_projection_best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data",
        default="data/gen-data-set.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data to use as test set (default: 0.1 = last 10%%)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of examples to show (0 to disable)"
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Don't show example predictions"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎯 BGE-M3 MODEL EVALUATION")
    print("="*60)
    print(f"Device: {args.device}")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load test data
    data_path = resolve_path(args.data)
    print(f"\n📂 Loading data from: {data_path}")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return 1
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use last X% as test set
    test_size = max(1, int(len(data) * args.test_split))
    test_data = data[-test_size:]
    print(f"✅ Using last {args.test_split*100:.0f}% of data: {len(test_data)} test examples")
    
    # Evaluate
    metrics = evaluate_on_dataset(model, test_data, args.device, args.batch_size)
    print_metrics(metrics)
    
    # Show examples
    if not args.no_examples and args.examples > 0:
        evaluate_examples(model, test_data, args.device, args.examples)
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())

