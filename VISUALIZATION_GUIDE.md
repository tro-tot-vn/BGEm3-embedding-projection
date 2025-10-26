# 🎨 Embedding Visualization Guide

**Last Updated:** October 26, 2025

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Visualization Types](#visualization-types)
5. [Understanding the Plots](#understanding-the-plots)
6. [Command-Line Options](#command-line-options)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

After training your model, visualization helps you understand:

✅ **What did the model learn?**
- Are queries and their positives close in embedding space?
- Are negatives pushed away?

✅ **How well does it generalize?**
- Do similar features cluster together?
- Are locations separable?

✅ **Where can it be improved?**
- Which queries have low similarity to their positives?
- Which negatives are too close to positives?

---

## 📦 Installation

### Required Dependencies

```bash
# Core dependencies (already in requirements.txt)
pip install torch transformers sentence-transformers
pip install matplotlib seaborn numpy scikit-learn

# For UMAP visualization (optional but recommended)
pip install umap-learn
```

**Note:** UMAP is optional. If not installed, t-SNE will still work.

---

## 🚀 Quick Start

### Basic Usage

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/gen-data-set.json \
    --output visualizations/
```

This will generate **5 types of plots** in the `visualizations/` directory:

1. `tsne_projection.png` - 2D t-SNE embedding space
2. `umap_projection.png` - 2D UMAP embedding space (if UMAP available)
3. `similarity_heatmap.png` - Query-document similarity matrix
4. `similarity_distribution.png` - Positive vs negative similarity histograms
5. `top_k_predictions.png` - Top-10 predictions for example queries

---

## 📊 Visualization Types

### 1. **t-SNE Projection** 📍

**What it shows:**
- 2D projection of high-dimensional embeddings (256D → 2D)
- Spatial relationships between queries and positives

**What to look for:**
- ✅ **Good:** Queries and their positives cluster together
- ❌ **Bad:** Queries and positives are scattered randomly

**Example interpretation:**

```
🟢 Query cluster → Close to → 🟢 Positive cluster = Good!
🔵 Query cluster → Far from → 🔴 Negative cluster = Great!
```

---

### 2. **UMAP Projection** 🗺️

**What it shows:**
- Similar to t-SNE but preserves more global structure
- Often clearer clustering than t-SNE

**Advantages over t-SNE:**
- Faster for large datasets
- Better at preserving both local and global structure
- More interpretable distances

**When to use:**
- Use UMAP for large datasets (>1000 samples)
- Use t-SNE for smaller datasets or when UMAP unavailable

---

### 3. **Similarity Heatmap** 🔥

**What it shows:**
- Matrix of cosine similarities: `sim[i, j] = cos(query_i, pos_j)`
- Diagonal = query-to-own-positive similarity (should be high!)
- Off-diagonal = query-to-other-positive similarity (should be lower)

**What to look for:**

```
Perfect heatmap:
     P1   P2   P3   P4
Q1  [🟢] [🟡] [🟡] [🔴]  ← Q1 matches P1 (diagonal = high)
Q2  [🟡] [🟢] [🟡] [🔴]  ← Q2 matches P2 (diagonal = high)
Q3  [🟡] [🟡] [🟢] [🔴]  ← Q3 matches P3 (diagonal = high)
Q4  [🔴] [🔴] [🔴] [🟢]  ← Q4 matches P4 (diagonal = high)

Legend: 🟢 High sim (>0.8) | 🟡 Medium (0.5-0.8) | 🔴 Low (<0.5)
```

**Red flags:**
- ❌ Diagonal is not the brightest (positives not ranking #1)
- ❌ Many bright off-diagonal cells (false positives)

---

### 4. **Similarity Distribution** 📈

**What it shows:**
- **Left plot:** Histogram of positive vs negative similarities
- **Right plot:** Box plot comparison

**What to look for:**

```
Good separation:
  Positive │         [━━━━━━━━━━]      Mean: 0.85
           │                              ↑ High!
           │ [━━━━━━]                  Mean: 0.45
  Negative │                              ↑ Low!
           └─────────────────────────────
           0.0    0.5    1.0

Bad separation:
  Positive │      [━━━━━━━━━━]         Overlap!
  Negative │   [━━━━━━━━━━]            ← Both around 0.6
           └─────────────────────────────
```

**Key metrics printed:**

```
📊 Similarity Statistics:
   Positive: 0.8234 ± 0.0876   ← Mean ± std
   Negative: 0.4521 ± 0.1234
   Margin:   0.3713             ← Difference (higher = better)
   Separation: 2.45σ            ← In units of std (>2 is good!)
```

**Interpretation:**
- ✅ **Margin > 0.3:** Good separation
- ✅ **Separation > 2σ:** Statistically significant
- ⚠️ **Margin < 0.2:** Model may struggle to distinguish

---

### 5. **Top-K Predictions** 🎯

**What it shows:**
- For each query, the top-10 most similar documents
- Green bars = correct positive match
- Red bars = incorrect matches

**What to look for:**

```
Good example:
Query: "Tìm trọ Q10 25m2 5.5tr"
  Rank 1: [🟢━━━━━━━━━━━━━━] 0.92  ← Correct! (green)
  Rank 2: [🔴━━━━━━━━━] 0.67       ← Wrong location
  Rank 3: [🔴━━━━━━━] 0.55         ← Wrong price
  ...

Bad example:
Query: "Tìm trọ Q10 25m2 5.5tr"
  Rank 1: [🔴━━━━━━━━━] 0.68       ← Wrong! Should be green
  Rank 2: [🔴━━━━━━━] 0.65
  Rank 3: [🟢━━━━━━] 0.62          ← Correct but rank 3 :(
  ...
```

**Good model:**
- Green bar is Rank 1 for most queries
- Large gap between Rank 1 and Rank 2

**Needs improvement:**
- Green bar is Rank 3+ (low MRR)
- Small gap between ranks (ambiguous predictions)

---

## ⚙️ Command-Line Options

### Full Options

```bash
python visualize_embeddings.py \
    --checkpoint PATH        # Required: Model checkpoint
    --data PATH              # Required: Dataset JSON
    --output DIR             # Output directory (default: visualizations/)
    --max-samples N          # Limit samples (default: 500)
    --batch-size N           # Encoding batch size (default: 32)
    --device cuda|cpu        # Device (default: auto-detect)
    --skip-umap              # Skip UMAP (faster)
    --skip-tsne              # Skip t-SNE (faster)
```

### Examples

**1. Quick visualization (100 samples):**

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/gen-data-set.json \
    --max-samples 100 \
    --output quick_viz/
```

**2. Full dataset visualization:**

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/gen-data-set.json \
    --max-samples 2000 \
    --output full_viz/
```

**3. Only t-SNE and heatmap (skip UMAP):**

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/gen-data-set.json \
    --skip-umap
```

**4. CPU-only (no GPU):**

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/gen-data-set.json \
    --device cpu
```

---

## 📚 Examples

### Example 1: Training Progress

Compare checkpoints at different epochs:

```bash
# Early epoch
python visualize_embeddings.py \
    --checkpoint checkpoints/epoch_1.pt \
    --data data/gen-data-set.json \
    --output viz_epoch1/

# Mid training
python visualize_embeddings.py \
    --checkpoint checkpoints/epoch_5.pt \
    --data data/gen-data-set.json \
    --output viz_epoch5/

# Best model
python visualize_embeddings.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/gen-data-set.json \
    --output viz_best/
```

Compare plots side-by-side to see improvement!

---

### Example 2: Dataset Quality Check

Visualize your dataset **before training**:

```bash
# Use untrained model (just projection head initialized)
python visualize_embeddings.py \
    --checkpoint model_init.pt \
    --data data/gen-data-set.json \
    --output viz_before_training/
```

**What to look for:**
- Even before training, frozen BGE-M3 should show some clustering
- If random → dataset might have issues

---

### Example 3: Feature-Specific Analysis

Generate embeddings grouped by feature:

1. **By Location:**
   - Create separate datasets per district
   - Visualize each: Do Q10 queries cluster separately from Q1?

2. **By Price Range:**
   - Low (<4tr), Medium (4-7tr), High (>7tr)
   - Do they form distinct clusters?

3. **By Amenities:**
   - With AC vs without
   - Do they separate?

---

## 🔍 Understanding the Plots

### What Good Embeddings Look Like

#### ✅ t-SNE/UMAP
```
      [Query Cluster]
            ↓
     (close together)
            ↓
    [Positive Cluster]

    (far from negatives)
```

#### ✅ Heatmap
```
Bright diagonal (>0.8)
Darker off-diagonal (<0.6)
```

#### ✅ Distribution
```
Positive mean: >0.75
Negative mean: <0.55
Margin: >0.20
```

#### ✅ Top-K
```
90%+ of queries have green bar at Rank 1
```

---

### What Bad Embeddings Look Like

#### ❌ t-SNE/UMAP
```
[Queries and Positives scattered randomly]
No clear clusters
```

#### ❌ Heatmap
```
No clear diagonal pattern
Many bright off-diagonal cells
```

#### ❌ Distribution
```
Positive and Negative overlap heavily
Margin < 0.15
```

#### ❌ Top-K
```
Many queries have green bar at Rank 3+
Small score gaps
```

---

## 🐛 Troubleshooting

### Issue 1: UMAP Not Available

**Error:**
```
⚠️  UMAP not available. Install with: pip install umap-learn
```

**Fix:**
```bash
pip install umap-learn
```

Or skip UMAP:
```bash
python visualize_embeddings.py ... --skip-umap
```

---

### Issue 2: Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Fix 1:** Reduce batch size:
```bash
python visualize_embeddings.py ... --batch-size 8
```

**Fix 2:** Reduce samples:
```bash
python visualize_embeddings.py ... --max-samples 200
```

**Fix 3:** Use CPU:
```bash
python visualize_embeddings.py ... --device cpu
```

---

### Issue 3: t-SNE Too Slow

**Problem:** t-SNE takes forever on large datasets

**Fix 1:** Skip t-SNE, use UMAP only:
```bash
python visualize_embeddings.py ... --skip-tsne
```

**Fix 2:** Reduce samples:
```bash
python visualize_embeddings.py ... --max-samples 300
```

---

### Issue 4: Plots Look Weird

**Problem 1:** All points in one cluster
- **Cause:** Model not trained yet / embeddings not normalized
- **Fix:** Check model checkpoint is correct

**Problem 2:** No clear separation
- **Cause:** Model undertrained / dataset too hard
- **Fix:** Train longer or check dataset quality

**Problem 3:** Heatmap is all yellow
- **Cause:** All similarities around 0.5-0.7
- **Fix:** Model needs more training epochs

---

## 📊 Metrics Interpretation

### Similarity Margin

| Margin | Quality | Interpretation |
|--------|---------|----------------|
| > 0.40 | Excellent | Very strong separation |
| 0.30-0.40 | Good | Clear distinction between pos/neg |
| 0.20-0.30 | Fair | Usable but room for improvement |
| 0.10-0.20 | Poor | Model struggles to distinguish |
| < 0.10 | Bad | Model failed to learn |

---

### Separation (in σ)

| Separation | Quality | Interpretation |
|------------|---------|----------------|
| > 3.0σ | Excellent | Statistically very strong |
| 2.0-3.0σ | Good | Statistically significant |
| 1.5-2.0σ | Fair | Noticeable difference |
| 1.0-1.5σ | Poor | Weak separation |
| < 1.0σ | Bad | No real separation |

---

## 🎯 Best Practices

### 1. Visualize Multiple Checkpoints
- Compare early vs late training
- Identify overfitting (if final checkpoint worse than earlier ones)

### 2. Sample Size Balance
- Too small (<100): Not representative
- Too large (>1000): t-SNE very slow
- **Recommended:** 300-500 samples

### 3. Interpret Together
- Don't rely on one plot
- Cross-reference: t-SNE + Distribution + Top-K

### 4. Dataset Quality Check
- Visualize BEFORE training (with random weights)
- If BGE-M3 base already shows good clustering → good dataset
- If random even with base BGE-M3 → check dataset

---

## 📝 What to Report

When sharing results, include:

```markdown
## Embedding Visualization Results

### Model Info
- Checkpoint: checkpoints/epoch_10.pt
- Dataset: data/gen-data-set.json (500 samples)

### Key Metrics
- Positive similarity: 0.83 ± 0.09
- Negative similarity: 0.42 ± 0.12
- Margin: 0.41
- Separation: 3.1σ

### Visual Analysis
- t-SNE: Clear query-positive clustering ✅
- Heatmap: Strong diagonal pattern ✅
- Distribution: Good separation (minimal overlap) ✅
- Top-K: 92% queries have correct positive at Rank 1 ✅

### Conclusion
Model successfully learned to:
- ✅ Cluster similar queries and positives
- ✅ Separate negatives with clear margin
- ✅ Achieve high retrieval accuracy

[Attach plots]
```

---

## 🎓 Further Analysis Ideas

### 1. Error Analysis
- Find queries where green bar is NOT Rank 1
- Manually inspect: Why did model fail?
- Common patterns: specific feature types, locations, price ranges?

### 2. Feature Importance
- Compare similarities when only one feature differs
- Which features create largest distance?
- Does this match your weight-config.json?

### 3. Location Clustering
- Do all Q10 queries cluster together?
- Are neighboring districts (Q1, Q3, Q10) closer than distant ones?

### 4. Price Continuity
- Are 5tr and 5.5tr queries close?
- Is there a smooth gradient in price space?

---

## 🔗 Related Files

- **Main Training:** `train_script.py`, `TRAIN.md`
- **Evaluation:** `evaluate_model.py`, `EVALUATION_GUIDE.md`
- **Model:** `model.py`
- **Dataset:** `pair_dataset.py`, `data/gen-data-set.json`

---

## ✅ Quick Checklist

Before considering your model "done":

- [ ] Generated all 5 visualization types
- [ ] Positive mean similarity > 0.75
- [ ] Negative mean similarity < 0.55
- [ ] Margin > 0.25
- [ ] Separation > 2.0σ
- [ ] >85% queries have correct positive at Rank 1
- [ ] Clear clusters in t-SNE/UMAP
- [ ] Bright diagonal in heatmap

If all checked → **Your model is ready for deployment!** 🚀

---

**Last Updated:** October 26, 2025  
**Part of:** BGE-M3 Fine-tuning for Vietnamese Rental Market

