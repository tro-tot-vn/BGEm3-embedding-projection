# 📊 Evaluation Guide - Đánh Giá Model

## 🎯 Overview

File `evaluate_model.py` giúp bạn đánh giá model sau khi train xong.

**Metrics được tính:**
- **MRR (Mean Reciprocal Rank):** Vị trí trung bình của kết quả đúng
- **Recall@K:** % queries tìm thấy đúng trong top-K
- **Example Predictions:** Xem cụ thể model predict như thế nào

---

## 🚀 Quick Start

### **Cách 1: Default (Đơn giản nhất)**

```bash
python evaluate_model.py
```

**Sẽ:**
- Load model từ `checkpoints/bgem3_projection_best.pt`
- Test trên 10% cuối của `data/gen-data-set.json`
- Show 5 examples

---

### **Cách 2: Custom Settings**

```bash
# Evaluate specific checkpoint
python evaluate_model.py --checkpoint checkpoints/bgem3_projection_epoch10.pt

# Use different test data
python evaluate_model.py --data data/test-set.json

# Show more examples
python evaluate_model.py --examples 10

# Faster evaluation (smaller batch)
python evaluate_model.py --batch-size 16

# CPU only
python evaluate_model.py --device cpu

# Use 20% of data as test set
python evaluate_model.py --test-split 0.2

# No examples, just metrics
python evaluate_model.py --no-examples
```

---

## 📊 Expected Output

```
============================================================
🎯 BGE-M3 MODEL EVALUATION
============================================================
Device: cuda
📦 Loading model from: checkpoints/bgem3_projection_best.pt
   Epoch: 10
   Loss: 0.4567
✅ Model loaded successfully

📂 Loading data from: data/gen-data-set.json
✅ Using last 10% of data: 100 test examples

📊 Evaluating on 100 examples...
🔄 Encoding queries...
Queries: 100%|████████████████████| 4/4 [00:02<00:00]
🔄 Encoding documents...
Documents: 100%|██████████████████| 4/4 [00:02<00:00]
🔄 Computing similarities...
🔄 Computing metrics...

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
   
   In top-10 results: 94.5% queries find correct match

============================================================
🔍 EXAMPLE PREDICTIONS
============================================================

📝 Example 1:
   Query: phòng trọ q10 wc riêng 25m2 5tr5...

   Similarities:
   ✅ Positive:    0.8523 - Cho thuê phòng Quận 10, 25m², WC khép kín, giá 5.5...
   ❌ Negative 1:  0.7234 - Cho thuê phòng Quận 11, 25m², WC khép kín, giá 5.5...
   ❌ Negative 2:  0.6891 - Cho thuê phòng Quận 10, 30m², WC khép kín, giá 6...
   ❌ Negative 3:  0.6512 - Cho thuê phòng Quận 10, 25m², WC chung, giá 5.5...
   ✅ Correct! Positive has highest similarity
   📊 Margin: +0.1289 (higher is better)

...

============================================================
✅ Evaluation complete!
============================================================
```

---

## 📈 Hiểu Metrics

### **MRR (Mean Reciprocal Rank)**

**Ý nghĩa:** Trung bình nghịch đảo vị trí của kết quả đúng

**Công thức:** `MRR = average(1 / rank_of_correct_doc)`

**Ví dụ:**
- Query 1: Kết quả đúng ở vị trí #1 → RR = 1.0
- Query 2: Kết quả đúng ở vị trí #2 → RR = 0.5
- Query 3: Kết quả đúng ở vị trí #5 → RR = 0.2
- **MRR = (1.0 + 0.5 + 0.2) / 3 = 0.567**

**Threshold:**
- **> 0.8:** 🟢 Excellent
- **0.6 - 0.8:** 🟡 Good
- **0.4 - 0.6:** 🟠 Fair
- **< 0.4:** 🔴 Poor

---

### **Recall@K**

**Ý nghĩa:** % queries có kết quả đúng trong top-K

**Ví dụ:**
- 100 queries total
- 65 queries có kết quả đúng ở vị trí #1 → **Recall@1 = 65%**
- 89 queries có kết quả đúng trong top-5 → **Recall@5 = 89%**
- 95 queries có kết quả đúng trong top-10 → **Recall@10 = 95%**

**Threshold (Cho rental search):**
- **Recall@1 > 60%:** Good
- **Recall@5 > 85%:** Good
- **Recall@10 > 90%:** Good
- **Recall@50 > 95%:** Good

---

## 🔍 Example Predictions Explained

```
📝 Example 1:
   Query: phòng trọ q10 wc riêng 25m2 5tr5...

   Similarities:
   ✅ Positive:    0.8523  ← HIGHEST = Good! ✅
   ❌ Negative 1:  0.7234  ← Lower (location wrong)
   ❌ Negative 2:  0.6891  ← Lower (area/price wrong)
   ❌ Negative 3:  0.6512  ← Lowest (amenity wrong)
```

**Good model:**
- ✅ Positive có similarity **cao nhất**
- ✅ Margin (positive - best_negative) **> 0.05**
- ✅ Hard negatives có order đúng (location error > area error > amenity error)

**Bad model:**
- ❌ Negative có similarity cao hơn positive
- ❌ Margin quá nhỏ (< 0.02)
- ❌ Order của negatives sai

---

## 🎯 Khi Nào Cần Re-train?

### **Nên Re-train nếu:**

1. **MRR < 0.5**
   - Model không học tốt
   - Try: Increase epochs, adjust learning rate

2. **Recall@10 < 80%**
   - Model không phân biệt được positive/negative
   - Try: Check hard negatives quality, increase batch size

3. **Margin < 0.05 trong examples**
   - Model không confident
   - Try: Train longer, use more hard negatives

4. **Val loss tăng (overfitting)**
   - Model memorize training data
   - Try: More data, stronger regularization

---

## 📝 Compare Multiple Checkpoints

```bash
# Evaluate all checkpoints
for epoch in 2 4 6 8 10; do
    echo "Evaluating epoch $epoch..."
    python evaluate_model.py \
        --checkpoint checkpoints/bgem3_projection_epoch${epoch}.pt \
        --no-examples \
        2>&1 | grep -E "MRR|Recall@10"
done

# Expected output:
# Evaluating epoch 2...
# MRR         :  65.30%
# Recall@10   :  85.20%
# Evaluating epoch 4...
# MRR         :  72.45%
# Recall@10   :  90.10%
# ...
```

---

## 🔬 Advanced: Create Separate Test Set

**Tốt nhất:** Tách test set riêng, KHÔNG dùng trong training

```python
# split_data.py
import json
from sklearn.model_selection import train_test_split

with open('data/gen-data-set.json') as f:
    data = json.load(f)

train, test = train_test_split(data, test_size=0.1, random_state=42)

with open('data/train-set.json', 'w') as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open('data/test-set.json', 'w') as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train)}, Test: {len(test)}")
```

**Then evaluate:**
```bash
python evaluate_model.py --data data/test-set.json --test-split 1.0
```

---

## 💡 Tips

1. **Always evaluate on unseen data**
   - Đừng evaluate trên training data!
   - Use last 10% hoặc separate test set

2. **Compare with baseline**
   - Evaluate untrained model (random weights)
   - Good model should be much better than random

3. **Check examples visually**
   - Numbers có thể misleading
   - Xem examples để hiểu model behavior

4. **Track metrics over epochs**
   - Plot MRR/Recall vs epochs
   - Detect overfitting early

---

## 📚 Full Command Reference

```bash
# Basic
python evaluate_model.py

# Custom checkpoint
python evaluate_model.py --checkpoint path/to/model.pt

# Custom data
python evaluate_model.py --data data/test-set.json

# Custom test split (20% of data)
python evaluate_model.py --test-split 0.2

# Device
python evaluate_model.py --device cuda
python evaluate_model.py --device cpu

# Batch size (for memory constraints)
python evaluate_model.py --batch-size 16

# Examples
python evaluate_model.py --examples 10     # Show 10 examples
python evaluate_model.py --examples 0      # No examples
python evaluate_model.py --no-examples     # No examples (flag)

# Combine
python evaluate_model.py \
    --checkpoint checkpoints/bgem3_projection_best.pt \
    --data data/test-set.json \
    --device cuda \
    --batch-size 32 \
    --examples 5
```

---

## ❓ Troubleshooting

### **Error: FileNotFoundError**
```
❌ Checkpoint not found
```
**Fix:** Check checkpoint path
```bash
ls checkpoints/
python evaluate_model.py --checkpoint checkpoints/bgem3_projection_final.pt
```

### **Error: CUDA OOM**
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce batch size
```bash
python evaluate_model.py --batch-size 16  # or smaller
```

### **Metrics are 0% or very low**
**Possible causes:**
1. Model not trained (using random weights)
2. Wrong checkpoint loaded
3. Data format mismatch

**Fix:** 
```bash
# Check training loss in checkpoint
python -c "
import torch
ckpt = torch.load('checkpoints/bgem3_projection_best.pt')
print('Loss:', ckpt.get('loss', 'N/A'))
print('Epoch:', ckpt.get('epoch', 'N/A'))
"
```

---

**Happy Evaluating! 📊**

