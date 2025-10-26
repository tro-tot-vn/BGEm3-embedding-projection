# 🎯 START HERE - Complete Training & Evaluation Guide

## ✅ **Project Status: READY TO USE**

Everything is set up and ready for training & evaluation!

---

## 📋 **Quick Navigation**

| What You Want | Read This | Then Run |
|---------------|-----------|----------|
| **Train model NOW** | `QUICK_START.md` | `python train_script.py` |
| **Detailed training guide** | `TRAIN.md` | - |
| **Evaluate model** | `EVALUATION_GUIDE.md` | `python evaluate_model.py` |
| **Visualize embeddings** 🎨 | `VISUALIZATION_GUIDE.md` | `python visualize_embeddings.py` |
| **Understand changes** | `CHANGES_SUMMARY.md` | - |
| **Project architecture** | `PROJECT_SUMMARY.md` | - |

---

## 🚀 **Fastest Path to Results**

### **Step 1: Train (5 minutes to start)**

```bash
# Activate environment
source venv/bin/activate

# Fix NumPy if needed
pip install "numpy<2.0"

# Train!
python train_script.py
```

**Expected time:**
- GPU (Tesla T4): ~20 minutes for 10 epochs
- CPU: ~2-3 hours

---

### **Step 2: Evaluate (2 minutes)**

```bash
# After training completes
python evaluate_model.py
```

**You'll see:**
```
📊 EVALUATION RESULTS
MRR         :  78.45%
Recall@10   :  94.50%
```

**Good results:**
- MRR > 70% ✅
- Recall@10 > 90% ✅

---

### **Step 3: Visualize (3 minutes)** 🎨

#### **3a. Training Curves** 📈
```bash
# Plot loss over time
python plot_training_curves.py
```

**You'll get:**
- 📈 Training loss curve
- 📉 Validation loss curve
- ⭐ Best checkpoints marked
- 📊 Training statistics

---

#### **3b. Embedding Space** 🎨
```bash
# Visualize what the model learned
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json \
    --max-samples 500
```

**You'll get 5 plots:**
- 📍 t-SNE projection (embedding space clustering)
- 🗺️ UMAP projection (alternative view)
- 🔥 Similarity heatmap (query-document matches)
- 📊 Distribution analysis (positive vs negative)
- 🎯 Top-K predictions (example queries)

**What to look for:**
- Clear query-positive clustering ✅
- Positive-negative separation ✅
- Bright diagonal in heatmap ✅

---

## 📁 **Project Files Overview**

### **🏃 Executable Scripts**

```
train_script.py          ⭐ Main training script
evaluate_model.py        ⭐ Evaluation script
visualize_embeddings.py  🎨 Embedding visualization
plot_training_curves.py  📈 Loss curves (NEW!)
test_weighted_pipeline.py  Test suite
```

**Run these!**

---

### **📚 Documentation**

```
00_START_HERE.md         ← You are here!
QUICK_START.md          ⭐ 5-minute quick start
TRAIN.md                 📖 Complete training guide (600+ lines)
EVALUATION_GUIDE.md      📊 Evaluation guide
VISUALIZATION_GUIDE.md   🎨 Visualization guide (NEW!)
CHANGES_SUMMARY.md       🔧 What was fixed
PROJECT_SUMMARY.md       📐 Technical architecture
README_FINAL.md          📋 Summary
```

**Read in this order:**
1. This file (00_START_HERE.md)
2. QUICK_START.md
3. EVALUATION_GUIDE.md (after training)
4. VISUALIZATION_GUIDE.md (to understand embeddings)

---

### **⚙️ Core Code**

```
model.py                 ✅ Fixed - Model definition
train.py                 ✅ Fixed - Training logic (weighted loss)
pair_dataset.py          ✅ Fixed - Dataset loader
```

**These are production-ready!**

---

### **📦 Configuration**

```
config.example.json      Template for custom config
requirements.txt         ✅ Restored - Dependencies
```

---

## 🎓 **Complete Workflow**

### **Full Pipeline:**

```
1. Setup Environment
   ↓
2. Train Model
   python train_script.py
   ↓
3. Monitor Loss (during training)
   Watch: Train Loss ↓, Val Loss ↓
   ↓
4. Checkpoints Saved
   checkpoints/bgem3_projection_best.pt ⭐
   checkpoints/loss_history.json 📊
   ↓
5. Plot Training Curves 📈
   python plot_training_curves.py
   Check: Converged? Overfitting?
   ↓
6. Evaluate Model
   python evaluate_model.py
   MRR, Recall@K
   ↓
7. Visualize Embeddings 🎨
   python visualize_embeddings.py
   t-SNE, Heatmap, Distribution
   ↓
8. Analyze Results
   • Loss converged? → Check ✅
   • Metrics good? → Check ✅
   • Visualizations clear? → Check ✅
   → Deploy! 🚀
```

---

## 💡 **Quick Commands Reference**

### **Training:**

```bash
# Default training
python train_script.py

# Custom settings
python train_script.py --epochs 20 --batch-size 64

# See all options
python train_script.py --help
```

### **Evaluation:**

```bash
# Evaluate best model
python evaluate_model.py

# Evaluate specific checkpoint
python evaluate_model.py --checkpoint checkpoints/bgem3_projection_epoch8.pt

# Show more examples
python evaluate_model.py --examples 10

# See all options
python evaluate_model.py --help
```

### **Visualization:** 🎨📈

#### **Training Curves:**
```bash
# Plot training loss curves
python plot_training_curves.py

# Custom history file
python plot_training_curves.py --history checkpoints/loss_history.json

# Compare multiple runs
python plot_training_curves.py \
    --compare run1/loss_history.json run2/loss_history.json \
    --labels "Baseline" "Improved" \
    --output comparison.png
```

#### **Embedding Space:**
```bash
# Basic visualization (500 samples)
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json

# Quick viz (100 samples, faster)
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json \
    --max-samples 100 \
    --output quick_viz/

# Full dataset (2000 samples, slower)
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json \
    --max-samples 2000 \
    --output full_viz/

# Skip UMAP (faster)
python visualize_embeddings.py ... --skip-umap

# See all options
python visualize_embeddings.py --help
```

### **Testing:**

```bash
# Run test suite (after fixing NumPy)
python test_weighted_pipeline.py
```

---

## 📊 **What to Expect**

### **During Training:**

```
================================================================================
🚀 BGE-M3 PROJECTION HEAD TRAINING
================================================================================
🖥️  Device: cuda
   GPU: Tesla T4
   Memory: 15.8 GB

📊 Loading dataset
✅ Loaded 1000 examples
   Train: 900 examples
   Val:   100 examples

🤖 Initializing model
✅ Trainable params: 262,144 (0.05% of total)

🏋️  Starting training for 10 epochs
================================================================================
Epoch 1/10: 100%|████| 7/7 [00:15<00:00, loss=1.2345, avg=1.3456]

📈 Epoch 1 Summary:
   Train Loss: 1.3456
   Val Loss:   1.4567
   ⭐ New best val loss! Saved to: checkpoints/bgem3_projection_best.pt

[... 9 more epochs ...]

✅ TRAINING COMPLETE!
📁 Output: checkpoints/bgem3_projection_best.pt
```

---

### **During Evaluation:**

```
============================================================
🎯 BGE-M3 MODEL EVALUATION
============================================================
📦 Loading model from: checkpoints/bgem3_projection_best.pt
✅ Model loaded successfully

📊 Evaluating on 100 examples...
🔄 Encoding queries... 100%
🔄 Encoding documents... 100%

============================================================
📊 EVALUATION RESULTS
============================================================
MRR         :  78.45% │███████████████████████████████████████░░░░░░░░░░│
Recall@1    :  65.30% │████████████████████████████████░░░░░░░░░░░░░░░░░│
Recall@5    :  89.20% │████████████████████████████████████████████░░░░░│
Recall@10   :  94.50% │███████████████████████████████████████████████░░│
Recall@50   :  99.10% │█████████████████████████████████████████████████│
============================================================

💡 Interpretation:
   🟢 Excellent! Model retrieves correct docs very accurately.
```

---

### **During Visualization:** 🎨

```
======================================================================
🎨 BGE-M3 Embedding Visualization Tool
======================================================================
🔧 Loading model from: checkpoints/bgem3_projection_best.pt
✅ Loaded checkpoint from epoch 10

📁 Loading dataset from: data/gen-data-set.json
✅ Loaded 500 query-positive pairs

📊 Encoding texts...
Encoding: 100%|████████████████████| 16/16 [00:12<00:00]

🔢 Computing similarity matrix...

----------------------------------------------------------------------
🎨 Computing t-SNE (perplexity=30)...
✅ Saved t-SNE plot to: visualizations/tsne_projection.png

----------------------------------------------------------------------
🎨 Computing UMAP (n_neighbors=15)...
✅ Saved UMAP plot to: visualizations/umap_projection.png

----------------------------------------------------------------------
🎨 Plotting similarity heatmap...
✅ Saved heatmap to: visualizations/similarity_heatmap.png

----------------------------------------------------------------------
🎨 Plotting similarity distribution...

📊 Similarity Statistics:
   Positive: 0.8234 ± 0.0876
   Negative: 0.4521 ± 0.1234
   Margin:   0.3713
   Separation: 2.45σ

✅ Saved distribution plot to: visualizations/similarity_distribution.png

----------------------------------------------------------------------
🎨 Plotting top-10 predictions for 5 examples...
✅ Saved top-K predictions to: visualizations/top_k_predictions.png

======================================================================
✅ Visualization Complete!
======================================================================

📁 All plots saved to: /path/to/visualizations

📊 Generated files:
   ✓ similarity_distribution.png
   ✓ similarity_heatmap.png
   ✓ top_k_predictions.png
   ✓ tsne_projection.png
   ✓ umap_projection.png
```

---

## 🎯 **Success Criteria**

### **Training Success:** 📈
- ✅ Loss giảm dần qua epochs (check training curves!)
- ✅ Loss converged (plateau in curve)
- ✅ Val loss < 1.0 (good), < 0.5 (excellent)
- ✅ Val loss không tăng (no overfitting in curve)
- ✅ Train-val gap < 0.15 (good generalization)
- ✅ Checkpoints được save

### **Evaluation Success:**
- ✅ MRR > 0.7 (good), > 0.8 (excellent)
- ✅ Recall@10 > 0.9
- ✅ Examples show correct predictions
- ✅ Margin > 0.05 between positive and negatives

### **Visualization Success:** 🎨
- ✅ t-SNE/UMAP: Clear query-positive clustering
- ✅ Heatmap: Bright diagonal (>0.8)
- ✅ Distribution: Margin > 0.25, Separation > 2.0σ
- ✅ Top-K: >85% queries have correct at Rank 1
- ✅ No strange patterns (e.g., all points in one cluster)

---

## 🐛 **Common Issues & Quick Fixes**

### **Issue 1: NumPy Error**
```bash
pip install "numpy<2.0"
```

### **Issue 2: CUDA OOM**
```bash
python train_script.py --batch-size 64
```

### **Issue 3: File Not Found (Colab)**
```bash
# Already fixed! train_script.py auto-detects paths
python train_script.py  # Just works!
```

### **Issue 4: Low Metrics**
- Check if training converged (loss should be < 1.0)
- Train more epochs: `--epochs 20`
- Check data quality

### **Issue 5: UMAP Not Available** 🎨
```bash
pip install umap-learn
# Or skip UMAP
python visualize_embeddings.py ... --skip-umap
```

### **Issue 6: Visualization OOM** 🎨
```bash
# Reduce samples
python visualize_embeddings.py ... --max-samples 200
# Or use CPU
python visualize_embeddings.py ... --device cpu
```

---

## 📈 **What's Next?**

### **After Successful Training & Evaluation:**

1. **Deploy Model:**
   ```python
   # inference.py
   model = BGEM3WithHead(d_out=128, freeze_encoder=True)
   model.load_state_dict(torch.load("checkpoints/bgem3_projection_best.pt"))
   model.eval()
   
   # Use for search
   embeddings = model(["phòng trọ q10 giá rẻ"])
   ```

2. **A/B Test:**
   - Compare with baseline
   - Measure real user metrics

3. **Iterate:**
   - Collect more data
   - Tune hyperparameters
   - Try different architectures

---

## 🔧 **Advanced Usage**

### **Custom Configuration:**

```bash
# Copy example config
cp config.example.json my_config.json

# Edit settings
nano my_config.json

# Train with custom config
python train_script.py --config my_config.json
```

### **Distributed Training (Multi-GPU):**

```bash
# See TRAIN.md section "Advanced Topics"
```

### **TensorBoard Logging:**

```bash
# See TRAIN.md section "Advanced Topics"
```

---

## 📚 **Learning Resources**

### **Understand the Code:**
1. Read `PROJECT_SUMMARY.md` - Architecture overview
2. Read `CHANGES_SUMMARY.md` - What was fixed and why
3. Look at code: `model.py` → `train.py` → `pair_dataset.py`

### **Improve Results:**
1. Experiment with `increment_ratio` in `data/weight-config.json`
2. Try different `d_out` dimensions (128, 512)
3. Adjust learning rate and epochs

### **Deploy to Production:**
1. Export to ONNX for faster inference
2. Quantize model for smaller size
3. Build API service with FastAPI

---

## ✅ **Checklist Before You Start**

- [ ] Environment activated: `source venv/bin/activate`
- [ ] NumPy fixed: `pip install "numpy<2.0"`
- [ ] Dataset exists: `ls data/gen-data-set.json`
- [ ] Read QUICK_START.md
- [ ] Ready to train!

---

## 🎉 **You're All Set!**

**Just run:**

```bash
# 1. Train
python train_script.py

# 2. Plot training curves 📈
python plot_training_curves.py

# 3. Evaluate
python evaluate_model.py

# 4. Visualize embeddings 🎨
python visualize_embeddings.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/gen-data-set.json

# Done! Check training_curves.png and visualizations/ 🚀
```

---

## 📞 **Need Help?**

**Read these in order:**
1. `QUICK_START.md` - For training
2. `EVALUATION_GUIDE.md` - For evaluation
3. `VISUALIZATION_GUIDE.md` - For understanding embeddings 🎨
4. `TRAIN.md` - For detailed training info
5. `CHANGES_SUMMARY.md` - For technical details

**Still stuck?**
- Check error messages carefully
- Re-read relevant documentation
- Verify environment setup

---

**Last Updated:** October 26, 2025  
**Status:** ✅ Production Ready  
**Version:** 3.1 (with Loss Curves 📈)

---

**Happy Training, Evaluating & Visualizing! 🚀📊📈🎨**

