# 🚀 Training Guide - BGE-M3 Embedding Projection

**Complete guide for training BGE-M3 with weighted hard negatives for Vietnamese rental market**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Training Configuration](#training-configuration)
6. [Training Process](#training-process)
7. [Monitoring & Evaluation](#monitoring--evaluation)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## 🎯 Overview

### What You'll Train

- **Base Model:** BGE-M3 (BAAI/bge-m3, 560M params, **FROZEN**)
- **Trainable Component:** Projection Head (1024 → 128 dims)
- **Training Method:** Contrastive learning with weighted hard negatives
- **Loss Function:** Symmetric InfoNCE with feature-based weighting

### Why This Approach?

✅ **Fast Training:** Only ~260K parameters to train (vs 560M)  
✅ **Less Overfitting:** Preserve pre-trained multilingual knowledge  
✅ **Domain Adaptation:** Learn Vietnamese rental market specifics  
✅ **Feature Importance:** Model understands location > price > amenities

---

## 📚 Prerequisites

### System Requirements

```yaml
Python: 3.10+
GPU Memory: 8GB+ recommended (can train on CPU but slower)
Disk Space: 5GB+ (model weights + dataset)
```

### Required Knowledge

- Basic Python and PyTorch
- Understanding of contrastive learning (helpful but not required)
- Familiarity with command line

---

## 🔧 Environment Setup

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd "/home/lamdx4/Projects/BGEm3 embedding projection"

# Create virtual environment
python3.10 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

**Expected output:**
```
PyTorch: 2.x.x
Transformers: 4.x.x
```

### Step 3: Verify GPU (Optional but Recommended)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## 📊 Data Preparation

### Step 1: Understand Dataset Structure

Your dataset (`data/gen-data-set.json`) should have this format:

```json
[
    {
        "query": "phòng trọ q10 wc riêng 25m2 5tr5",
        "pos": "Cho thuê phòng Quận 10, 25m², WC khép kín, giá 5.5 triệu/tháng",
        "hard_neg": [
            {
                "text": "Cho thuê phòng Quận 11, 25m², WC khép kín, giá 5.5 triệu",
                "type": ["location"],
                "weight": 2.5
            },
            {
                "text": "Cho thuê phòng Quận 10, 25m², WC khép kín, giá 7 triệu",
                "type": ["price"],
                "weight": 2.0
            }
        ]
    }
]
```

**Fields Explained:**
- `query`: User search query (real, messy text)
- `pos`: Ground truth positive listing
- `hard_neg`: Similar but wrong listings
  - `text`: The listing text
  - `type`: Feature types that differ (e.g., location, price)
  - `weight`: Calculated importance (2.5=critical, 1.0=minor)

### Step 2: Populate Weights (If Not Done)

If your dataset has `weight: 0` placeholders:

```bash
# Run weight populator
python scripts/populate_weights.py

# This will:
# 1. Calculate weights based on feature types
# 2. Create backup (gen-data-set.json.bak)
# 3. Update weights in-place
```

**Weight Calculation Logic:**
```python
# Single feature:
["location"] → 2.5

# Multiple features (Max + Incremental):
["location", "price"] → max(2.5, 2.0) + 2.0×0.3 = 3.1
["location", "price", "area"] → 2.5 + (2.0+1.5)×0.3 = 3.55
```

### Step 3: Verify Dataset

```bash
python -c "
import json
with open('data/gen-data-set.json') as f:
    data = json.load(f)
print(f'✅ Loaded {len(data)} training examples')

# Check first item
item = data[0]
print(f'✅ Query: {item[\"query\"][:50]}...')
print(f'✅ Hard negatives: {len(item.get(\"hard_neg\", []))}')
if item.get('hard_neg'):
    print(f'✅ First weight: {item[\"hard_neg\"][0].get(\"weight\", \"N/A\")}')
"
```

**Expected output:**
```
✅ Loaded 1000 training examples
✅ Query: phòng trọ q10 wc riêng 25m2 5tr5...
✅ Hard negatives: 3
✅ First weight: 2.5
```

---

## ⚙️ Training Configuration

### Hyperparameters Explained

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **d_out** | 128 | 128-512 | Output embedding dimension |
| **batch_size** | 128 | 64-128 | Batch size (adjust for GPU memory) |
| **learning_rate** | 2e-4 | 1e-5 - 5e-4 | AdamW learning rate |
| **weight_decay** | 0.01 | 0.001-0.1 | L2 regularization |
| **temperature** | 0.07 | 0.05-0.1 | InfoNCE temperature (τ) |
| **epochs** | 10 | 5-20 | Number of training epochs |
| **max_length** | 512 | 128-512 | Maximum token length |

### Recommended Configurations

#### 🚀 **Fast Prototyping (Small GPU/CPU)**
```python
batch_size = 64
d_out = 128
epochs = 5
max_length = 128
```

#### ⚡ **Standard Training (8GB GPU)**
```python
batch_size = 128
d_out = 128
epochs = 10
max_length = 512
```

#### 🔥 **High Quality (16GB+ GPU)**
```python
batch_size = 128
d_out = 512
epochs = 15
max_length = 512
```

---

## 🏋️ Training Process

### Option 1: Quick Start Script

Create `train_script.py`:

```python
#!/usr/bin/env python3
"""
Training script for BGE-M3 projection head with weighted hard negatives
"""

import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BGEM3WithHead
from train import ContrastiveTrainer
from pair_dataset import PairDataset, collate

# ===== Configuration =====
CONFIG = {
    "data_path": "data/gen-data-set.json",
    "output_dir": "checkpoints",
    "d_out": 128,
    "batch_size": 128,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "epochs": 10,
    "max_length": 512,
    "freeze_encoder": True,
    "use_hard_neg": True,
    "save_every": 2,  # Save checkpoint every N epochs
}

def main():
    # ===== Setup =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Create output directory
    import os
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # ===== Load Data =====
    print(f"\n📊 Loading dataset from {CONFIG['data_path']}")
    with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} training examples")
    
    # Create dataset with hard negatives
    dataset = PairDataset(data, use_hard_neg=CONFIG["use_hard_neg"])
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate,
        drop_last=True,  # Important for stable batch negatives
        num_workers=0,   # Set to 2-4 if you have multi-core CPU
    )
    print(f"✅ Created DataLoader: {len(loader)} batches per epoch")
    
    # ===== Initialize Model =====
    print(f"\n🤖 Initializing model (d_out={CONFIG['d_out']})")
    model = BGEM3WithHead(
        d_out=CONFIG["d_out"],
        freeze_encoder=CONFIG["freeze_encoder"],
        use_layernorm=False
    ).to(device)
    
    trainer = ContrastiveTrainer(model)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"   ({100 * trainable_params / total_params:.2f}% of total)")
    
    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(
        model.head.parameters(),  # Only optimize projection head
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Optional: Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    # )
    
    # ===== Training Loop =====
    print(f"\n🏋️  Starting training for {CONFIG['epochs']} epochs")
    print("=" * 70)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0.0
        
        # Progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            optimizer.zero_grad()
            loss = trainer.training_step(batch)
            
            # Backward pass
            loss.backward()
            
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })
        
        # Epoch statistics
        avg_loss = epoch_loss / len(loader)
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Average Loss: {avg_loss:.4f}")
        
        # Optional: Update learning rate
        # scheduler.step()
        # print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % CONFIG["save_every"] == 0 or avg_loss < best_loss:
            checkpoint_path = f"{CONFIG['output_dir']}/bgem3_projection_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': CONFIG,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save best model
                best_path = f"{CONFIG['output_dir']}/bgem3_projection_best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"⭐ New best model saved: {best_path}")
        
        print("-" * 70)
    
    # ===== Final Save =====
    final_path = f"{CONFIG['output_dir']}/bgem3_projection_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training complete! Final model saved: {final_path}")
    print(f"📊 Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
```

### Run Training

```bash
# Make script executable (optional)
chmod +x train_script.py

# Run training
python train_script.py
```

**Expected Output:**
```
🖥️  Using device: cuda
📊 Loading dataset from data/gen-data-set.json
✅ Loaded 1000 training examples
✅ Created DataLoader: 7 batches per epoch
🤖 Initializing model (d_out=128)
✅ Trainable parameters: 262,144 / 560,394,240
   (0.05% of total)
🏋️  Starting training for 10 epochs
======================================================================
Epoch 1/10: 100%|████████| 7/7 [00:15<00:00, loss=1.2345, avg_loss=1.3456]
📈 Epoch 1 Summary:
   Average Loss: 1.3456
💾 Saved checkpoint: checkpoints/bgem3_projection_epoch1.pt
⭐ New best model saved: checkpoints/bgem3_projection_best.pt
----------------------------------------------------------------------
...
```

### Option 2: Interactive Training (Jupyter/Python REPL)

```python
import json
import torch
from torch.utils.data import DataLoader
from model import BGEM3WithHead
from train import ContrastiveTrainer
from pair_dataset import PairDataset, collate

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
with open("data/gen-data-set.json") as f:
    data = json.load(f)

# Create dataset with hard negatives
dataset = PairDataset(data, use_hard_neg=True)
loader = DataLoader(dataset, batch_size=128, shuffle=True, 
                   collate_fn=collate, drop_last=True)

# Initialize model
model = BGEM3WithHead(d_out=128, freeze_encoder=True).to(device)
trainer = ContrastiveTrainer(model)

# Optimizer
optimizer = torch.optim.AdamW(model.head.parameters(), lr=2e-4, weight_decay=0.01)

# Training loop
model.train()
for epoch in range(10):
    epoch_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = trainer.training_step(batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "checkpoints/bgem3_projection.pt")
```

---

## 📊 Monitoring & Evaluation

### Loss Interpretation

**Expected Loss Trajectory:**
```
Epoch 1: 1.5-2.0  (High, model learning basics)
Epoch 3: 0.8-1.2  (Decreasing, good progress)
Epoch 5: 0.5-0.8  (Stabilizing)
Epoch 10: 0.3-0.6 (Converged)
```

**Warning Signs:**
- ⚠️ Loss > 3.0: Learning rate too high or data issues
- ⚠️ Loss = NaN: Numerical instability (reduce lr, check data)
- ⚠️ Loss not decreasing: Model not learning (check gradients, data)

### Monitoring Gradients

Add to training loop:

```python
# After loss.backward()
grad_norms = []
for name, param in model.head.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        print(f"{name}: {grad_norm:.6f}")

avg_grad_norm = sum(grad_norms) / len(grad_norms)
print(f"Average gradient norm: {avg_grad_norm:.6f}")
```

**Healthy gradients:** 1e-4 to 1e-1  
**Too small:** < 1e-6 (vanishing gradients)  
**Too large:** > 10 (exploding gradients, need clipping)

### Validation (Recommended)

Split your dataset:

```python
from sklearn.model_selection import train_test_split

# Load data
with open("data/gen-data-set.json") as f:
    data = json.load(f)

# Split 90% train, 10% validation
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Create separate loaders
train_dataset = PairDataset(train_data, use_hard_neg=True)
val_dataset = PairDataset(val_data, use_hard_neg=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                         collate_fn=collate, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, 
                       collate_fn=collate, drop_last=False)
```

**Validation loop:**

```python
def validate(model, trainer, val_loader, device):
    """Compute validation loss"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            loss = trainer.training_step(batch)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader)

# In training loop:
for epoch in range(10):
    # ... training code ...
    
    # Validation
    val_loss = validate(model, trainer, val_loader, device)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 3:  # Stop if no improvement for 3 epochs
            print("Early stopping triggered!")
            break
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```python
batch_size = 64  # or 32
```

2. **Reduce max_length:**
```python
max_length = 128  # or 128
```

3. **Use gradient accumulation:**
```python
accumulation_steps = 4  # Effective batch size = batch_size × accumulation_steps

for batch_idx, batch in enumerate(loader):
    loss = trainer.training_step(batch)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

4. **Enable mixed precision (FP16):**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    optimizer.zero_grad()
    
    with autocast():  # FP16 forward pass
        loss = trainer.training_step(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Issue 2: Loss is NaN

**Causes:**
- Learning rate too high
- Numerical overflow in exp()
- Invalid data

**Solutions:**

1. **Reduce learning rate:**
```python
learning_rate = 1e-4  # or 5e-5
```

2. **Add gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
```

3. **Check data for issues:**
```python
# Verify no NaN in embeddings
def check_batch(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN found in {key}!")
```

### Issue 3: Model Not Learning (Loss Stuck)

**Symptoms:**
- Loss stays constant across epochs
- Loss decreases very slowly

**Solutions:**

1. **Check if encoder is frozen:**
```python
# Should print True
print(f"Encoder frozen: {not any(p.requires_grad for p in model.encoder.parameters())}")

# Should print > 0
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
```

2. **Verify hard negatives are used:**
```python
# In training loop
batch = next(iter(loader))
print(f"Has hard_neg: {'hard_neg' in batch}")
if 'hard_neg' in batch:
    print(f"Hard neg counts: {[len(hn) for hn in batch['hard_neg']]}")
```

3. **Increase learning rate (carefully):**
```python
learning_rate = 5e-4  # Try higher
```

### Issue 4: Training Too Slow

**Solutions:**

1. **Use DataLoader num_workers:**
```python
loader = DataLoader(dataset, batch_size=128, num_workers=4, ...)
```

2. **Reduce max_length:**
```python
max_length = 128  # Faster tokenization
```

3. **Profile bottlenecks:**
```python
import time

start = time.time()
for batch in loader:
    batch_time = time.time()
    loss = trainer.training_step(batch)
    forward_time = time.time() - batch_time
    
    loss.backward()
    backward_time = time.time() - batch_time - forward_time
    
    optimizer.step()
    total_time = time.time() - start
    
    print(f"Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")
    break
```

---

## 🔬 Advanced Topics

### 1. Learning Rate Scheduling

```python
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)

# Warmup + cosine
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine(step):
    warmup_steps = 100
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)

# In training loop (call after each batch or epoch)
scheduler.step()
```

### 2. Tensorboard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/bgem3_training')

# In training loop
writer.add_scalar('Loss/train', avg_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

# View: tensorboard --logdir=runs
```

### 3. Distributed Training (Multi-GPU)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model
model = BGEM3WithHead(...).to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler, ...)

# Run: torchrun --nproc_per_node=2 train_script.py
```

### 4. Custom Weight Strategies

Experiment with different weighting strategies:

```python
# In data/weight-config.json
{
    "method": "max_incremental",  # or "capped_linear", "average"
    "increment_ratio": 0.3,       # Try 0.35, 0.4 for stronger effect
    "cap": 4.0
}

# Regenerate weights
python scripts/populate_weights.py
```

### 5. Inference & Deployment

```python
# Load trained model
model = BGEM3WithHead(d_out=128, freeze_encoder=True)
model.load_state_dict(torch.load("checkpoints/bgem3_projection_best.pt"))
model.eval()
model.to(device)

# Encode queries
with torch.no_grad():
    queries = ["phòng trọ q10 giá rẻ", "căn hộ quận 1"]
    query_embs = model(queries, device=device)  # [2, 128]

# Encode documents
with torch.no_grad():
    docs = ["Cho thuê phòng Quận 10...", "Căn hộ Quận 1..."]
    doc_embs = model(docs, device=device)  # [2, 128]

# Compute similarities
similarities = query_embs @ doc_embs.T  # [2, 2]
print(similarities)

# Get top-k results
scores, indices = similarities[0].topk(k=5)
```

---

## 📚 Additional Resources

### Papers & References

- **BGE-M3:** [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
- **InfoNCE Loss:** [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- **Hard Negative Mining:** [In Defense of the Triplet Loss](https://arxiv.org/abs/1703.07737)

### Useful Commands

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f training.log

# Find checkpoints
ls -lh checkpoints/

# Clean cache
rm -rf __pycache__
python -c "import torch; torch.cuda.empty_cache()"
```

### Tips & Best Practices

1. **Start small:** Test with 100 samples first
2. **Monitor early:** Check first epoch carefully
3. **Save often:** Don't lose hours of training
4. **Validate:** Always use a validation set
5. **Document:** Keep notes on what works

---

## ✅ Checklist Before Training

- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset prepared and weights populated
- [ ] GPU available (optional but recommended)
- [ ] Checkpoint directory created
- [ ] Test run passes (`python test_weighted_pipeline.py`)
- [ ] Configuration reviewed and adjusted
- [ ] Backup important data

---

## 🎯 Quick Start Summary

```bash
# 1. Setup
source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data
python scripts/populate_weights.py

# 3. Test pipeline
python test_weighted_pipeline.py

# 4. Train
python train_script.py

# 5. Monitor
# Watch loss decrease, save best checkpoint

# 6. Inference
# Load model, encode texts, compute similarities
```

---

**Happy Training! 🚀**

For questions or issues, refer to:
- `PROJECT_SUMMARY.md` for technical context
- `README.md` for project overview
- `test_weighted_pipeline.py` for examples

**Last Updated:** October 22, 2025  
**Version:** 1.0

