# 🚀 Quick Start - Training in 5 Minutes

## ⚡ Fastest Way to Start Training

### **Step 1: Activate Environment**

```bash
source venv/bin/activate
```

### **Step 2: Fix NumPy (if needed)**

```bash
pip install "numpy<2.0"
```

### **Step 3: Run Training**

```bash
# Default settings (recommended for first run)
python train_script.py

# Or with custom settings
python train_script.py --epochs 5 --batch-size 64
```

**That's it! Training will start immediately.**

---

## 📊 What You'll See

```
================================================================================
🚀 BGE-M3 PROJECTION HEAD TRAINING
   Vietnamese Rental Market (Phòng Trọ)
================================================================================
Started at: 2025-10-22 12:00:00
🖥️  Device: cuda
   GPU: NVIDIA RTX 3090
   Memory: 24.0 GB

📊 Loading dataset from: data/gen-data-set.json
✅ Loaded 1000 examples
   Train: 900 examples
   Val:   100 examples
✅ Train batches: 7
✅ Val batches:   1

🤖 Initializing model
✅ Model initialized
   Trainable params: 262,144
   Total params:     560,394,240
   Trainable ratio:  0.05%

⚙️  Optimizer: AdamW
   Learning rate: 0.0002
   Weight decay:  0.01

🏋️  Starting training for 10 epochs
================================================================================
Epoch 1/10: 100%|████████| 7/7 [00:15<00:00, loss=1.2345, avg=1.3456]

📈 Epoch 1 Summary:
   Train Loss: 1.3456
   Val Loss:   1.4567
   ⭐ New best val loss! Saved to: checkpoints/bgem3_projection_best.pt
```

---

## ⚙️ Command-Line Options

### **Basic Usage:**

```bash
# Use all defaults
python train_script.py

# Specify epochs
python train_script.py --epochs 20

# Custom batch size (smaller for less GPU memory)
python train_script.py --batch-size 64

# Custom learning rate
python train_script.py --lr 0.0001

# Force CPU (if no GPU)
python train_script.py --device cpu

# Custom output directory
python train_script.py --output my_checkpoints

# Combine multiple options
python train_script.py --epochs 15 --batch-size 64 --lr 0.00015
```

### **Using Config File:**

```bash
# Copy example config
cp config.example.json my_config.json

# Edit my_config.json to your liking
nano my_config.json

# Train with config
python train_script.py --config my_config.json
```

---

## 📁 Where Are My Models?

After training, check `checkpoints/` directory:

```
checkpoints/
├── config.json                    # Your training config
├── bgem3_projection_best.pt       # Best model (lowest validation loss)
├── bgem3_projection_final.pt      # Final model (last epoch)
├── bgem3_projection_epoch2.pt     # Checkpoint at epoch 2
├── bgem3_projection_epoch4.pt     # Checkpoint at epoch 4
└── ...
```

**Use `bgem3_projection_best.pt` for inference!**

---

## 🧪 Test Before Training

```bash
# Verify everything works
python test_weighted_pipeline.py
```

**Expected output:**
```
TEST 1: Weight Calculator ✓
TEST 2: Dataset Loading ✓
TEST 3: PairDataset Class ✓
TEST 4: Model Forward Pass ✓
TEST 5: Training Step (Weighted Loss) ✓
TEST 6: Backward Pass & Gradients ✓

🎉 ALL TESTS PASSED!
```

---

## 🔧 Common Issues & Fixes

### **Issue: CUDA Out of Memory**

```bash
# Solution 1: Reduce batch size
python train_script.py --batch-size 64

# Solution 2: Use CPU
python train_script.py --device cpu
```

### **Issue: NumPy Error**

```bash
# Solution: Downgrade NumPy
pip install "numpy<2.0"
```

### **Issue: Dataset Not Found**

```bash
# Check if data exists
ls data/gen-data-set.json

# If not, you need to prepare your dataset first
# See TRAIN.md for details
```

---

## 📊 Expected Training Time

| Setup | Batch Size | Time per Epoch | Total (10 epochs) |
|-------|------------|----------------|-------------------|
| RTX 3090 | 128 | ~2 min | ~20 min |
| RTX 2080 Ti | 128 | ~3 min | ~30 min |
| RTX 2080 Ti | 64 | ~4 min | ~40 min |
| CPU (16 cores) | 64 | ~15 min | ~2.5 hours |

---

## 🎯 After Training: Inference

```python
import torch
from model import BGEM3WithHead

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BGEM3WithHead(d_out=256, freeze_encoder=True)
model.load_state_dict(torch.load("checkpoints/bgem3_projection_best.pt"))
model.eval()
model.to(device)

# Encode texts
with torch.no_grad():
    queries = ["phòng trọ q10 giá rẻ", "căn hộ quận 1"]
    embeddings = model(queries, device=device)  # [2, 256]
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"L2-normalized: {torch.norm(embeddings, dim=1)}")
```

---

## 🔄 Resume Training

```python
# Load checkpoint
checkpoint = torch.load("checkpoints/bgem3_projection_epoch4.pt")

# Resume from epoch 4
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']

# Continue training from start_epoch...
```

---

## 📖 Need More Details?

- **Full guide:** See `TRAIN.md`
- **Technical details:** See `CHANGES_SUMMARY.md`
- **Project overview:** See `PROJECT_SUMMARY.md`

---

## 💡 Quick Tips

1. **Start with default settings** - They're tuned for good results
2. **Monitor validation loss** - Use it for early stopping
3. **Save checkpoints frequently** - Don't lose hours of training
4. **Use GPU if possible** - 10-20x faster than CPU
5. **Check logs** - Progress bar shows loss in real-time

---

## 🎉 Ready to Train!

```bash
# Just run this:
python train_script.py

# Then wait for:
# ✅ TRAINING COMPLETE!
```

**Good luck! 🚀**

