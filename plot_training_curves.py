#!/usr/bin/env python3
"""
Training Curves Visualization
==============================

Plots training and validation loss curves from training history.

Usage:
    python plot_training_curves.py [--history PATH] [--output PATH]
    
Example:
    python plot_training_curves.py --history checkpoints/loss_history.json
    
Author: BGE-M3 Fine-tuning Project
Date: October 2025
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np


def load_loss_history(history_path):
    """Load loss history from JSON file"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(history, output_path="training_curves.png"):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with keys 'epochs', 'train_loss', 'val_loss'
        output_path: Where to save the plot
    """
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    # Filter out None values from val_loss
    val_epochs = [e for e, v in zip(epochs, val_loss) if v is not None]
    val_loss_filtered = [v for v in val_loss if v is not None]
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot training loss
    ax.plot(epochs, train_loss, 
            marker='o', markersize=8, linewidth=2.5,
            label='Training Loss', color='#2E86DE', alpha=0.9)
    
    # Plot validation loss (if available)
    if val_loss_filtered:
        ax.plot(val_epochs, val_loss_filtered,
                marker='s', markersize=8, linewidth=2.5,
                label='Validation Loss', color='#EE5A6F', alpha=0.9)
    
    # Find best points
    best_train_epoch = epochs[np.argmin(train_loss)]
    best_train_loss = min(train_loss)
    
    # Mark best training loss
    ax.plot(best_train_epoch, best_train_loss, 
            marker='*', markersize=20, color='gold', 
            markeredgecolor='black', markeredgewidth=1.5,
            label=f'Best Train: {best_train_loss:.4f} (Epoch {best_train_epoch})',
            zorder=10)
    
    # Mark best validation loss (if available)
    if val_loss_filtered:
        best_val_epoch = val_epochs[np.argmin(val_loss_filtered)]
        best_val_loss = min(val_loss_filtered)
        ax.plot(best_val_epoch, best_val_loss,
                marker='*', markersize=20, color='gold',
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'Best Val: {best_val_loss:.4f} (Epoch {best_val_epoch})',
                zorder=10)
    
    # Labels and title
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training & Validation Loss Curves', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Set integer ticks for epochs
    ax.set_xticks(epochs)
    
    # Y-axis formatting
    ax.set_ylim(bottom=0)
    
    # Add metadata text
    metadata_text = (
        f"LR: {history.get('learning_rate', 'N/A')}\n"
        f"Batch Size: {history.get('batch_size', 'N/A')}"
    )
    ax.text(0.02, 0.98, metadata_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to: {output_path}")


def plot_loss_comparison(histories, labels, output_path="loss_comparison.png"):
    """
    Plot multiple training runs for comparison.
    
    Args:
        histories: List of loss history dictionaries
        labels: List of labels for each history
        output_path: Where to save the plot
    """
    sns.set_style("whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    # Plot training losses
    for hist, label, color in zip(histories, labels, colors):
        epochs = hist['epochs']
        train_loss = hist['train_loss']
        ax1.plot(epochs, train_loss, marker='o', linewidth=2,
                 label=label, color=color, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot validation losses
    for hist, label, color in zip(histories, labels, colors):
        epochs = hist['epochs']
        val_loss = hist['val_loss']
        val_epochs = [e for e, v in zip(epochs, val_loss) if v is not None]
        val_loss_filtered = [v for v in val_loss if v is not None]
        
        if val_loss_filtered:
            ax2.plot(val_epochs, val_loss_filtered, marker='s', linewidth=2,
                     label=label, color=color, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Loss comparison saved to: {output_path}")


def print_training_summary(history):
    """Print training summary statistics"""
    train_loss = history['train_loss']
    val_loss = [v for v in history['val_loss'] if v is not None]
    epochs = history['epochs']
    
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    
    print(f"\n🏋️  Training:")
    print(f"   Total epochs:     {len(epochs)}")
    print(f"   Initial loss:     {train_loss[0]:.4f}")
    print(f"   Final loss:       {train_loss[-1]:.4f}")
    print(f"   Best loss:        {min(train_loss):.4f} (Epoch {epochs[np.argmin(train_loss)]})")
    print(f"   Improvement:      {train_loss[0] - train_loss[-1]:.4f} (-{100 * (train_loss[0] - train_loss[-1]) / train_loss[0]:.1f}%)")
    
    if val_loss:
        val_epochs = [e for e, v in zip(epochs, history['val_loss']) if v is not None]
        print(f"\n📊 Validation:")
        print(f"   Total validations: {len(val_loss)}")
        print(f"   Initial loss:      {val_loss[0]:.4f}")
        print(f"   Final loss:        {val_loss[-1]:.4f}")
        print(f"   Best loss:         {min(val_loss):.4f} (Epoch {val_epochs[np.argmin(val_loss)]})")
        print(f"   Improvement:       {val_loss[0] - val_loss[-1]:.4f} (-{100 * (val_loss[0] - val_loss[-1]) / val_loss[0]:.1f}%)")
        
        # Check for overfitting
        train_best = min(train_loss)
        val_best = min(val_loss)
        gap = abs(train_best - val_best)
        
        print(f"\n🔍 Train-Val Gap:")
        print(f"   Best train loss: {train_best:.4f}")
        print(f"   Best val loss:   {val_best:.4f}")
        print(f"   Gap:             {gap:.4f}")
        
        if gap < 0.05:
            print(f"   Status:          ✅ Good generalization")
        elif gap < 0.15:
            print(f"   Status:          ⚠️  Some overfitting")
        else:
            print(f"   Status:          ❌ Significant overfitting")
    
    print(f"\n⚙️  Hyperparameters:")
    print(f"   Learning rate:   {history.get('learning_rate', 'N/A')}")
    print(f"   Batch size:      {history.get('batch_size', 'N/A')}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument(
        "--history",
        type=str,
        default="checkpoints/loss_history.json",
        help="Path to loss history JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_curves.png",
        help="Output path for plot"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs='+',
        help="Paths to multiple history files for comparison"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        help="Labels for comparison plots"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("📈 TRAINING CURVES VISUALIZATION")
    print("=" * 60)
    
    # Comparison mode
    if args.compare:
        print(f"\n🔍 Comparison mode: {len(args.compare)} runs")
        
        histories = []
        for path in args.compare:
            print(f"   Loading: {path}")
            history = load_loss_history(path)
            histories.append(history)
        
        labels = args.labels if args.labels else [f"Run {i+1}" for i in range(len(histories))]
        
        plot_loss_comparison(histories, labels, args.output)
        
        # Print summary for each
        for hist, label in zip(histories, labels):
            print(f"\n{'=' * 60}")
            print(f"Summary for: {label}")
            print_training_summary(hist)
    
    # Single run mode
    else:
        print(f"\n📁 Loading history from: {args.history}")
        
        if not Path(args.history).exists():
            print(f"❌ Error: File not found: {args.history}")
            print(f"\n💡 Tip: Make sure you've run training first:")
            print(f"   python train_script.py")
            return
        
        history = load_loss_history(args.history)
        
        # Print summary
        print_training_summary(history)
        
        # Plot curves
        print(f"\n📊 Generating plot...")
        plot_training_curves(history, args.output)
        
        print(f"\n✅ Done!")
        print(f"   View plot: {args.output}")


if __name__ == "__main__":
    main()

