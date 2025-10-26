#!/usr/bin/env python3
"""
BGE-M3 Embedding Visualization Tool
====================================

Visualizes trained embedding space to understand what the model learned.

Features:
  - t-SNE/UMAP 2D projections of embedding space
  - Similarity heatmaps (query-to-document)
  - Training curves (loss, learning rate)
  - Top-K retrieval visualization
  - Similarity distribution analysis
  - Location/feature clustering analysis

Usage:
    python visualize_embeddings.py --checkpoint checkpoints/best_model.pt \\
                                     --data data/gen-data-set.json \\
                                     --output visualizations/

Author: BGE-M3 Fine-tuning Project
Date: October 2025
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For dimensionality reduction
try:
    from sklearn.manifold import TSNE
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not available. Install with: pip install umap-learn")

from model import BGEM3WithHead


class EmbeddingVisualizer:
    """Visualizes trained embedding space."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize visualizer with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device
        print(f"🔧 Loading model from: {model_path}")
        
        # Load model
        self.model = BGEM3WithHead(
            d_out=128,
            freeze_encoder=True
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
            print("✅ Loaded model weights")
        
        self.model.eval()
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Embeddings array [N, d_out]
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            emb = self.model.encode(batch)
            embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def compute_similarity_matrix(
        self,
        query_embs: np.ndarray,
        doc_embs: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix.
        
        Args:
            query_embs: Query embeddings [N_q, d]
            doc_embs: Document embeddings [N_d, d]
            
        Returns:
            Similarity matrix [N_q, N_d]
        """
        # Embeddings are already L2-normalized, so dot product = cosine similarity
        return query_embs @ doc_embs.T
    
    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        colors: List[str],
        title: str,
        output_path: Path,
        perplexity: int = 30
    ):
        """
        Plot t-SNE 2D projection of embeddings.
        
        Args:
            embeddings: Embeddings array [N, d]
            labels: Labels for each point
            colors: Colors for each point
            title: Plot title
            output_path: Where to save plot
            perplexity: t-SNE perplexity parameter
        """
        print(f"🎨 Computing t-SNE (perplexity={perplexity})...")
        
        # Compute t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(embeddings) - 1),
            random_state=42,
            n_iter=1000
        )
        coords_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Get unique label-color combinations
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = [l == label for l in labels]
            label_colors = [c for c, m in zip(colors, mask) if m]
            plt.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=label_colors[0] if label_colors else 'gray',
                label=label,
                alpha=0.6,
                s=50
            )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved t-SNE plot to: {output_path}")
    
    def plot_umap(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        colors: List[str],
        title: str,
        output_path: Path,
        n_neighbors: int = 15
    ):
        """
        Plot UMAP 2D projection of embeddings.
        
        Args:
            embeddings: Embeddings array [N, d]
            labels: Labels for each point
            colors: Colors for each point
            title: Plot title
            output_path: Where to save plot
            n_neighbors: UMAP n_neighbors parameter
        """
        if not UMAP_AVAILABLE:
            print("⚠️  UMAP not available, skipping...")
            return
        
        print(f"🎨 Computing UMAP (n_neighbors={n_neighbors})...")
        
        # Compute UMAP
        umap = UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(embeddings) - 1),
            random_state=42,
            min_dist=0.1
        )
        coords_2d = umap.fit_transform(embeddings)
        
        # Plot (same as t-SNE)
        plt.figure(figsize=(12, 10))
        
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = [l == label for l in labels]
            label_colors = [c for c, m in zip(colors, mask) if m]
            plt.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=label_colors[0] if label_colors else 'gray',
                label=label,
                alpha=0.6,
                s=50
            )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("UMAP Dimension 1", fontsize=12)
        plt.ylabel("UMAP Dimension 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved UMAP plot to: {output_path}")
    
    def plot_similarity_heatmap(
        self,
        sim_matrix: np.ndarray,
        query_labels: List[str],
        doc_labels: List[str],
        output_path: Path,
        max_items: int = 50
    ):
        """
        Plot similarity heatmap.
        
        Args:
            sim_matrix: Similarity matrix [N_q, N_d]
            query_labels: Labels for queries
            doc_labels: Labels for documents
            output_path: Where to save plot
            max_items: Max items to show (for readability)
        """
        print(f"🎨 Plotting similarity heatmap...")
        
        # Truncate if too large
        if sim_matrix.shape[0] > max_items:
            sim_matrix = sim_matrix[:max_items, :max_items]
            query_labels = query_labels[:max_items]
            doc_labels = doc_labels[:max_items]
        
        plt.figure(figsize=(14, 12))
        
        sns.heatmap(
            sim_matrix,
            xticklabels=doc_labels,
            yticklabels=query_labels,
            cmap="RdYlGn",
            center=0.5,
            vmin=0.0,
            vmax=1.0,
            cbar_kws={'label': 'Cosine Similarity'},
            square=False
        )
        
        plt.title("Query-Document Similarity Matrix", fontsize=16, fontweight='bold')
        plt.xlabel("Documents", fontsize=12)
        plt.ylabel("Queries", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved heatmap to: {output_path}")
    
    def plot_similarity_distribution(
        self,
        pos_similarities: np.ndarray,
        neg_similarities: np.ndarray,
        output_path: Path
    ):
        """
        Plot distribution of positive vs negative similarities.
        
        Args:
            pos_similarities: Similarities for positive pairs
            neg_similarities: Similarities for negative pairs
            output_path: Where to save plot
        """
        print(f"🎨 Plotting similarity distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(pos_similarities, bins=50, alpha=0.7, label='Positive', color='green', edgecolor='black')
        axes[0].hist(neg_similarities, bins=50, alpha=0.7, label='Negative', color='red', edgecolor='black')
        axes[0].axvline(pos_similarities.mean(), color='green', linestyle='--', linewidth=2, label=f'Pos Mean: {pos_similarities.mean():.3f}')
        axes[0].axvline(neg_similarities.mean(), color='red', linestyle='--', linewidth=2, label=f'Neg Mean: {neg_similarities.mean():.3f}')
        axes[0].set_xlabel("Cosine Similarity", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Similarity Distribution", fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [pos_similarities, neg_similarities]
        bp = axes[1].boxplot(data_to_plot, labels=['Positive', 'Negative'], patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        axes[1].set_ylabel("Cosine Similarity", fontsize=12)
        axes[1].set_title("Similarity Box Plot", fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved distribution plot to: {output_path}")
        
        # Print statistics
        margin = pos_similarities.mean() - neg_similarities.mean()
        print(f"\n📊 Similarity Statistics:")
        print(f"   Positive: {pos_similarities.mean():.4f} ± {pos_similarities.std():.4f}")
        print(f"   Negative: {neg_similarities.mean():.4f} ± {neg_similarities.std():.4f}")
        print(f"   Margin:   {margin:.4f}")
        print(f"   Separation: {margin / (pos_similarities.std() + neg_similarities.std()):.4f}σ")
    
    def plot_top_k_predictions(
        self,
        queries: List[str],
        documents: List[str],
        sim_matrix: np.ndarray,
        ground_truth: List[int],
        output_path: Path,
        k: int = 10,
        num_examples: int = 5
    ):
        """
        Show top-K predictions for example queries.
        
        Args:
            queries: Query texts
            documents: Document texts
            sim_matrix: Similarity matrix [N_q, N_d]
            ground_truth: Ground truth indices for each query
            output_path: Where to save plot
            k: Number of top results to show
            num_examples: Number of example queries to visualize
        """
        print(f"🎨 Plotting top-{k} predictions for {num_examples} examples...")
        
        fig, axes = plt.subplots(num_examples, 1, figsize=(14, 3 * num_examples))
        if num_examples == 1:
            axes = [axes]
        
        for i in range(min(num_examples, len(queries))):
            # Get top-k predictions
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]
            top_k_scores = sim_matrix[i, top_k_indices]
            
            # Color bars based on correctness
            colors = ['green' if idx == ground_truth[i] else 'red' for idx in top_k_indices]
            
            # Plot
            axes[i].barh(range(k), top_k_scores, color=colors, alpha=0.7, edgecolor='black')
            axes[i].set_yticks(range(k))
            axes[i].set_yticklabels([f"Rank {j+1}" for j in range(k)], fontsize=9)
            axes[i].set_xlabel("Cosine Similarity", fontsize=10)
            axes[i].set_title(f"Query {i+1}: \"{queries[i][:80]}...\"", fontsize=11, fontweight='bold')
            axes[i].set_xlim([0, 1])
            axes[i].grid(True, alpha=0.3, axis='x')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Correct (Positive)'),
                Patch(facecolor='red', label='Incorrect')
            ]
            axes[i].legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved top-K predictions to: {output_path}")


def load_dataset(data_path: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str], List[int]]:
    """
    Load dataset for visualization.
    
    Args:
        data_path: Path to dataset JSON
        max_samples: Max samples to load (for faster visualization)
        
    Returns:
        (queries, positives, ground_truth_indices)
    """
    print(f"📁 Loading dataset from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    queries = [item['query'] for item in data]
    positives = [item['pos'] for item in data]
    ground_truth = list(range(len(queries)))  # Each query matches its corresponding positive
    
    print(f"✅ Loaded {len(queries)} query-positive pairs")
    
    return queries, positives, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Visualize BGE-M3 trained embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--output", type=str, default="visualizations/", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples for visualization")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP visualization")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE visualization")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎨 BGE-M3 Embedding Visualization Tool")
    print("="*70 + "\n")
    
    # Load model
    visualizer = EmbeddingVisualizer(args.checkpoint, device=args.device)
    
    # Load data
    queries, positives, ground_truth = load_dataset(args.data, max_samples=args.max_samples)
    
    # Encode
    print("\n📊 Encoding texts...")
    query_embs = visualizer.encode_batch(queries, batch_size=args.batch_size)
    pos_embs = visualizer.encode_batch(positives, batch_size=args.batch_size)
    
    # Compute similarities
    print("\n🔢 Computing similarity matrix...")
    sim_matrix = visualizer.compute_similarity_matrix(query_embs, pos_embs)
    
    # Extract positive and negative similarities
    pos_similarities = np.array([sim_matrix[i, i] for i in range(len(queries))])
    neg_mask = ~np.eye(len(queries), dtype=bool)
    neg_similarities = sim_matrix[neg_mask]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total pairs: {len(queries)}")
    print(f"   Embedding dim: {query_embs.shape[1]}")
    print(f"   Positive pairs: {len(pos_similarities)}")
    print(f"   Negative pairs: {len(neg_similarities)}")
    
    # Plot 1: t-SNE
    if not args.skip_tsne:
        print("\n" + "-"*70)
        all_embs = np.vstack([query_embs, pos_embs])
        labels = ['Query'] * len(queries) + ['Positive'] * len(positives)
        colors = ['blue'] * len(queries) + ['green'] * len(positives)
        
        visualizer.plot_tsne(
            all_embs,
            labels,
            colors,
            "t-SNE Projection of Embedding Space",
            output_dir / "tsne_projection.png"
        )
    
    # Plot 2: UMAP
    if not args.skip_umap and UMAP_AVAILABLE:
        print("\n" + "-"*70)
        all_embs = np.vstack([query_embs, pos_embs])
        labels = ['Query'] * len(queries) + ['Positive'] * len(positives)
        colors = ['blue'] * len(queries) + ['green'] * len(positives)
        
        visualizer.plot_umap(
            all_embs,
            labels,
            colors,
            "UMAP Projection of Embedding Space",
            output_dir / "umap_projection.png"
        )
    
    # Plot 3: Similarity heatmap
    print("\n" + "-"*70)
    visualizer.plot_similarity_heatmap(
        sim_matrix,
        [f"Q{i+1}" for i in range(len(queries))],
        [f"P{i+1}" for i in range(len(positives))],
        output_dir / "similarity_heatmap.png",
        max_items=min(50, len(queries))
    )
    
    # Plot 4: Similarity distribution
    print("\n" + "-"*70)
    visualizer.plot_similarity_distribution(
        pos_similarities,
        neg_similarities,
        output_dir / "similarity_distribution.png"
    )
    
    # Plot 5: Top-K predictions
    print("\n" + "-"*70)
    visualizer.plot_top_k_predictions(
        queries,
        positives,
        sim_matrix,
        ground_truth,
        output_dir / "top_k_predictions.png",
        k=10,
        num_examples=min(5, len(queries))
    )
    
    print("\n" + "="*70)
    print("✅ Visualization Complete!")
    print("="*70)
    print(f"\n📁 All plots saved to: {output_dir.absolute()}")
    print("\n📊 Generated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"   ✓ {file.name}")


if __name__ == "__main__":
    main()

