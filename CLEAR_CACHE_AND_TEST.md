# 🔧 Fix NaN Issue - Clear Cache & Test

## ❌ Vấn đề

Nếu bạn thấy `[nan]` khi test model ở repo khác, có thể do **HF cache cũ** (trước khi fix bug).

## ✅ Giải pháp

### 1. Clear Hugging Face Cache

```bash
# Xóa cache của model này
rm -rf ~/.cache/huggingface/hub/models--lamdx4--bge-m3-vietnamese-rental-projection

# Hoặc xóa toàn bộ cache (nếu cần)
# rm -rf ~/.cache/huggingface/hub/*
```

### 2. Test Standalone

Tạo file `test.py` ở bất kỳ đâu (NGOÀI project này):

```python
#!/usr/bin/env python3
"""Test model từ HF Hub (standalone)"""

import torch
from transformers import AutoModel

# Load model
print("Downloading model from HF Hub...")
model = AutoModel.from_pretrained(
    "lamdx4/bge-m3-vietnamese-rental-projection",
    trust_remote_code=True
)

# Test
texts = ["Phòng trọ Quận 10, 25m², 5tr"]
embeddings = model.encode(texts)

print(f"Shape: {embeddings.shape}")
print(f"Sample: {embeddings[0][:5]}")

# Check NaN
if torch.isnan(embeddings).any():
    print("❌ ERROR: Found NaN!")
else:
    print("✅ SUCCESS: No NaN!")
```

Chạy:
```bash
python test.py
```

### 3. Kết quả mong đợi

```
✅ SUCCESS: No NaN!
Shape: torch.Size([1, 128])
Sample: tensor([-0.0486, -0.1078, -0.0124,  0.1243,  0.0297])
```

---

## 🐛 Bugs đã fix

### Bug 1: Missing `head.` prefix (CRITICAL!)
- **Vấn đề:** SafeTensors có key `linear.weight` thay vì `head.linear.weight`
- **Kết quả:** Embeddings hoàn toàn sai (max diff = 0.38)
- **Fix:** Giữ nguyên prefix khi save weights

### Bug 2: Missing tokenizer (CRITICAL!)
- **Vấn đề:** `encode()` gọi `self.tokenizer` nhưng không tồn tại
- **Kết quả:** NaN values khi test standalone
- **Fix:** Lazy load tokenizer trong property

---

## ✅ Verified

Model đã được test:
- ✅ Local test (trong project folder)
- ✅ Standalone test (ở /tmp, không có local code)
- ✅ So sánh với original model (max diff = 0.0)
- ✅ Similarity ranking chính xác

---

## 🌐 Model đã sẵn sàng

**URL:** https://huggingface.co/lamdx4/bge-m3-vietnamese-rental-projection

**Usage:**
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "lamdx4/bge-m3-vietnamese-rental-projection",
    trust_remote_code=True
)

# Encode texts
embeddings = model.encode(["Your text here"])
```

---

## 📝 Notes

- Model size: ~513 KB (chỉ projection head)
- Users vẫn cần load BGE-M3 base (tự động download khi khởi tạo)
- Tokenizer được lazy load (download lần đầu tiên gọi `encode()`)

