# 🎨 Visualization Tool - Implementation Summary

**Date:** October 26, 2025  
**Author:** BGE-M3 Fine-tuning Project  
**Purpose:** Visualize trained embedding space to understand model learning

---

## 📋 Overview

The visualization tool provides **5 comprehensive visualizations** to help understand:
- ✅ What the model learned
- ✅ How well embeddings separate
- ✅ Which queries work well/poorly
- ✅ Whether to deploy or retrain

---

## 🎯 Motivation

### **Why Visualization?**

After training and evaluation, you know:
- ✅ Training loss converged
- ✅ MRR and Recall metrics

But you DON'T know:
- ❓ **HOW** does the model group similar items?
- ❓ **WHERE** do failures occur?
- ❓ **WHAT** patterns did it learn?

**Visualization answers these questions!**

---

## 📦 Files Created

### 1. **`visualize_embeddings.py`** (554 lines)

**Purpose:** Main visualization script

**Features:**
- Loads trained model checkpoint
- Encodes queries and documents in batches
- Computes similarity matrices
- Generates 5 types of plots
- Outputs to `visualizations/` directory

**Key Classes:**
- `EmbeddingVisualizer`: Main class
  - `encode_batch()`: Batch encoding with progress bar
  - `compute_similarity_matrix()`: Cosine similarity
  - `plot_tsne()`: t-SNE 2D projection
  - `plot_umap()`: UMAP 2D projection
  - `plot_similarity_heatmap()`: Heatmap visualization
  - `plot_similarity_distribution()`: Histogram + boxplot
  - `plot_top_k_predictions()`: Ranked predictions

**Dependencies:**
```python
torch, numpy, matplotlib, seaborn
sklearn.manifold.TSNE
umap.UMAP (optional)
```

---

### 2. **`VISUALIZATION_GUIDE.md`** (611 lines)

**Purpose:** Comprehensive user guide

**Sections:**
1. Overview & Installation
2. Quick Start (1-command usage)
3. Visualization Types (detailed explanation of each)
4. Understanding Plots (good vs bad examples)
5. Command-Line Options
6. Examples (training progress, dataset quality)
7. Troubleshooting
8. Metrics Interpretation
9. Best Practices

---

### 3. **Updated `requirements.txt`**

**Added dependencies:**
```
# Visualization
matplotlib>=3.7.0,<3.10.0
seaborn>=0.12.0,<0.14.0

# Optional (for better visualizations)
# umap-learn>=0.5.0  # Uncomment for UMAP support
```

---

### 4. **Updated `00_START_HERE.md`**

**Changes:**
- Added "Visualize embeddings" to navigation table
- Added Step 3: Visualize (3 minutes) to Fastest Path
- Updated Complete Workflow to include visualization
- Added visualization commands to Quick Commands Reference
- Added "During Visualization" example output
- Added Visualization Success criteria
- Added visualization troubleshooting (UMAP, OOM)
- Updated version to 3.0

---

## 🎨 Visualization Types

### **1. t-SNE Projection** 📍

**What it shows:**
- 2D projection of 256-dimensional embeddings
- Spatial relationships between queries and positives

**Implementation:**
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
coords_2d = tsne.fit_transform(embeddings)
```

**Good indicators:**
- ✅ Queries and their positives cluster together
- ✅ Clear separation between different clusters

**Bad indicators:**
- ❌ Random scatter (no clustering)
- ❌ All points in one cluster

---

### **2. UMAP Projection** 🗺️

**What it shows:**
- Alternative 2D projection (often better than t-SNE)
- Preserves both local and global structure

**Implementation:**
```python
from umap import UMAP

umap = UMAP(n_components=2, n_neighbors=15, random_state=42)
coords_2d = umap.fit_transform(embeddings)
```

**Advantages:**
- ⚡ Faster than t-SNE (important for >500 samples)
- 📐 Better preserves global structure
- 🎯 More consistent across runs

**Optional:** Automatically skipped if `umap-learn` not installed

---

### **3. Similarity Heatmap** 🔥

**What it shows:**
- Matrix: `sim[i, j] = cosine_similarity(query_i, positive_j)`
- Diagonal cells = query-to-own-positive (should be HIGH!)
- Off-diagonal = query-to-other-positive (should be LOWER)

**Implementation:**
```python
sim_matrix = query_embs @ pos_embs.T  # [N_q, N_d]

sns.heatmap(
    sim_matrix,
    cmap="RdYlGn",
    center=0.5,
    vmin=0.0,
    vmax=1.0
)
```

**Good indicators:**
- ✅ Bright diagonal (similarities > 0.8)
- ✅ Darker off-diagonal (similarities < 0.6)
- ✅ Clear pattern: diagonal >> others

**Bad indicators:**
- ❌ Dim diagonal (< 0.6)
- ❌ Bright off-diagonal (false positives)
- ❌ No clear pattern

---

### **4. Similarity Distribution** 📊

**What it shows:**
- **Left:** Histogram of positive vs negative similarities
- **Right:** Box plot comparison

**Implementation:**
```python
# Extract similarities
pos_sim = [sim_matrix[i, i] for i in range(N)]
neg_sim = sim_matrix[~np.eye(N, dtype=bool)]

# Plot histogram
plt.hist(pos_sim, alpha=0.7, label='Positive', color='green')
plt.hist(neg_sim, alpha=0.7, label='Negative', color='red')

# Statistics
margin = pos_sim.mean() - neg_sim.mean()
separation = margin / (pos_sim.std() + neg_sim.std())
```

**Key Metrics:**
```
Positive: 0.8234 ± 0.0876  ← Mean ± std
Negative: 0.4521 ± 0.1234
Margin:   0.3713            ← pos_mean - neg_mean
Separation: 2.45σ           ← margin / (pos_std + neg_std)
```

**Good indicators:**
- ✅ Margin > 0.25
- ✅ Separation > 2.0σ (statistically significant)
- ✅ Minimal overlap between distributions

**Bad indicators:**
- ❌ Margin < 0.15
- ❌ Separation < 1.5σ
- ❌ Heavy overlap

---

### **5. Top-K Predictions** 🎯

**What it shows:**
- For each example query, show top-10 most similar documents
- Green bar = correct positive match
- Red bars = incorrect matches

**Implementation:**
```python
# For each query
for i in range(num_examples):
    # Get top-k predictions
    top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]
    top_k_scores = sim_matrix[i, top_k_indices]
    
    # Color by correctness
    colors = ['green' if idx == ground_truth[i] else 'red' 
              for idx in top_k_indices]
    
    # Plot horizontal bar chart
    plt.barh(range(k), top_k_scores, color=colors)
```

**Good indicators:**
- ✅ Green bar at Rank 1 for most queries
- ✅ Large gap between Rank 1 and Rank 2
- ✅ All scores > 0.5

**Bad indicators:**
- ❌ Green bar at Rank 3+
- ❌ Small gaps between ranks
- ❌ Low absolute scores

---

## 🚀 Usage Examples

### **Example 1: Basic Visualization**

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json
```

**Output:**
```
visualizations/
├── tsne_projection.png
├── umap_projection.png
├── similarity_heatmap.png
├── similarity_distribution.png
└── top_k_predictions.png
```

---

### **Example 2: Quick Visualization (100 samples)**

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json \
    --max-samples 100 \
    --output quick_viz/
```

**Time:** ~30 seconds

---

### **Example 3: Full Dataset (2000 samples)**

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json \
    --max-samples 2000 \
    --output full_viz/
```

**Time:** ~3-5 minutes (t-SNE is slow)

---

### **Example 4: Skip UMAP (faster)**

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json \
    --skip-umap
```

**Use when:** UMAP not installed or want faster execution

---

### **Example 5: Compare Training Progress**

```bash
# Epoch 1
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_epoch1.pt \
    --data data/gen-data-set.json \
    --output viz_epoch1/

# Epoch 5
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_epoch5.pt \
    --data data/gen-data-set.json \
    --output viz_epoch5/

# Best model
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json \
    --output viz_best/

# Compare plots side-by-side!
```

---

## 📊 Interpretation Guidelines

### **What Good Embeddings Look Like**

| Visualization | Good Indicators |
|---------------|-----------------|
| **t-SNE/UMAP** | Clear clusters, queries close to positives |
| **Heatmap** | Bright diagonal (>0.8), dark off-diagonal (<0.6) |
| **Distribution** | Margin >0.25, Separation >2σ, minimal overlap |
| **Top-K** | >85% green at Rank 1, large gaps |

---

### **What Bad Embeddings Look Like**

| Visualization | Bad Indicators |
|---------------|----------------|
| **t-SNE/UMAP** | Random scatter, no clear clusters |
| **Heatmap** | Dim diagonal (<0.6), bright off-diagonal |
| **Distribution** | Margin <0.15, heavy overlap |
| **Top-K** | Green at Rank 3+, small gaps |

---

## 🐛 Troubleshooting

### **Issue 1: UMAP Not Available**

```bash
pip install umap-learn
```

Or skip UMAP:
```bash
python visualize_embeddings.py ... --skip-umap
```

---

### **Issue 2: CUDA OOM**

```bash
# Option 1: Reduce batch size
python visualize_embeddings.py ... --batch-size 8

# Option 2: Reduce samples
python visualize_embeddings.py ... --max-samples 200

# Option 3: Use CPU
python visualize_embeddings.py ... --device cpu
```

---

### **Issue 3: t-SNE Too Slow**

```bash
# Skip t-SNE, use UMAP only
python visualize_embeddings.py ... --skip-tsne
```

---

## 🎓 Best Practices

### **1. Sample Size**

| Samples | Speed | Coverage | Recommendation |
|---------|-------|----------|----------------|
| 100 | ⚡ Fast (~30s) | Low | Quick check |
| 500 | 🚶 Medium (~2m) | Good | ✅ **Recommended** |
| 2000 | 🐌 Slow (~5m) | High | Full analysis |

---

### **2. When to Visualize**

✅ **Do visualize:**
- After training completes (understand what model learned)
- When metrics seem strange (debug issues)
- Before deployment (final validation)
- When comparing models (A/B testing)

❌ **Don't visualize:**
- During training (use tensorboard for that)
- On every epoch (expensive)

---

### **3. Interpretation Workflow**

```
1. Check Distribution first
   ├─ Margin < 0.2? → Model needs more training
   └─ Margin > 0.25? → Continue to step 2

2. Check t-SNE/UMAP
   ├─ Clear clusters? → Good!
   └─ Random scatter? → Dataset/model issue

3. Check Heatmap
   ├─ Bright diagonal? → Retrieval working
   └─ Dim diagonal? → Model not learning positives

4. Check Top-K
   ├─ Green at Rank 1? → Deploy!
   └─ Green at Rank 3+? → Retrain or tune
```

---

## ✅ Success Checklist

Use this checklist to decide if your model is ready:

- [ ] **Distribution:** Margin > 0.25 ✅
- [ ] **Distribution:** Separation > 2.0σ ✅
- [ ] **t-SNE/UMAP:** Clear query-positive clustering ✅
- [ ] **Heatmap:** Diagonal brightness > 0.8 ✅
- [ ] **Heatmap:** Off-diagonal < 0.6 ✅
- [ ] **Top-K:** >85% queries have green at Rank 1 ✅
- [ ] **Top-K:** Gap between Rank 1 and Rank 2 > 0.1 ✅

**If all checked → Your model is production-ready! 🚀**

---

## 🔗 Integration with Workflow

### **Complete Training → Evaluation → Visualization Pipeline**

```bash
# Step 1: Train
python train_script.py --epochs 10

# Step 2: Evaluate
python evaluate_model.py

# Step 3: Visualize
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json

# Step 4: Analyze
# - Open visualizations/*.png
# - Check success criteria
# - Decide: Deploy or Retrain
```

---

## 📈 Typical Results

### **Expected Output (Well-Trained Model)**

```
📊 Similarity Statistics:
   Positive: 0.8234 ± 0.0876  ✅ High mean
   Negative: 0.4521 ± 0.1234  ✅ Low mean
   Margin:   0.3713           ✅ Large gap
   Separation: 2.45σ          ✅ Significant

Visualizations:
✓ tsne_projection.png       ✅ Clear clusters
✓ umap_projection.png       ✅ Good separation
✓ similarity_heatmap.png    ✅ Bright diagonal
✓ similarity_distribution.png ✅ Minimal overlap
✓ top_k_predictions.png     ✅ 92% correct at Rank 1
```

---

## 🎯 Key Takeaways

1. **Visualization is essential** for understanding what your model learned
2. **5 complementary views** provide different insights:
   - t-SNE/UMAP: Global structure
   - Heatmap: Pairwise relationships
   - Distribution: Statistical separation
   - Top-K: Practical retrieval performance
3. **Quantitative metrics + Visual inspection = Complete picture**
4. **Use visualization to:**
   - Validate training success
   - Debug failures
   - Compare models
   - Build confidence before deployment

---

## 📚 Related Documentation

- **User Guide:** `VISUALIZATION_GUIDE.md` (611 lines)
- **Training:** `TRAIN.md`
- **Evaluation:** `EVALUATION_GUIDE.md`
- **Getting Started:** `00_START_HERE.md`

---

## 🎉 Conclusion

The visualization tool provides a **comprehensive, production-ready** solution for understanding trained embedding models.

**Key Features:**
- ✅ 5 types of visualizations
- ✅ Automatic statistics computation
- ✅ Batch processing for efficiency
- ✅ GPU/CPU support
- ✅ Extensive documentation
- ✅ Easy to use (1-command)

**Result:** You can now confidently answer:
- ✅ "Did my model learn correctly?"
- ✅ "Is it ready for deployment?"
- ✅ "Where are the failure modes?"

---

**Status:** ✅ Complete and Production-Ready  
**Date:** October 26, 2025  
**Version:** 1.0

