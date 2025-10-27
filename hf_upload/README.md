---
language:
- vi
license: mit
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- embeddings
- vietnamese
- rental
- real-estate
library_name: transformers
pipeline_tag: feature-extraction
---

# BGE-M3 Vietnamese Rental Property Search

Fine-tuned projection head for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) optimized for **Vietnamese rental property search** (Phòng trọ).

This model adds a lightweight trainable projection head (128 dimensions) on top of the frozen BGE-M3 encoder, trained with **weighted hard negatives** using contrastive learning (InfoNCE loss).

## 🎯 Model Description

- **Base Model**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (frozen)
- **Task**: Semantic search for Vietnamese rental properties
- **Training Strategy**: Weighted contrastive learning with hard negatives
- **Output Dimension**: 128 (projected from 1024)
- **Training Data**: 10,384 Vietnamese rental property query-document pairs

## 📊 Performance

Evaluated on 96 test examples:

| Metric | Score |
|--------|-------|
| **MRR** | **98.44%** |
| **Recall@1** | **96.88%** |
| **Recall@5** | **100.00%** |
| **Recall@10** | **100.00%** |
| **Recall@50** | **100.00%** |

### Interpretation

- **98.44% MRR**: On average, the correct match appears at position ~1.02 (nearly always rank 1!)
- **96.88% Recall@1**: 93 out of 96 queries find the correct match at the top position
- **100% Recall@5+**: All queries find their correct match within top-5 results

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch safetensors
```

### Usage

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model
model = AutoModel.from_pretrained(
    "your-username/bge-m3-vietnamese-rental-projection",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Encode texts
texts = [
    "Phòng trọ Quận 10, 25m², giá 5 triệu, WC riêng, máy lạnh",
    "Cho thuê phòng Bình Thạnh, 20m², 4 triệu/tháng"
]

# Method 1: Using encode (recommended)
embeddings = model.encode(texts, device=device)  # [2, 128]

# Method 2: Using forward
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # [2, 128], L2-normalized

print(embeddings.shape)  # torch.Size([2, 128])

# Compute similarity (cosine)
similarity = embeddings[0] @ embeddings[1]
print(f"Similarity: {similarity:.4f}")
```

### Search Example

```python
# Build a search engine
class RentalSearchEngine:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.database_embeddings = None
        self.database_texts = None
    
    def index(self, property_descriptions):
        """Index a database of property descriptions"""
        self.database_texts = property_descriptions
        self.database_embeddings = self.model.encode(
            property_descriptions,
            device=self.device
        )
    
    def search(self, query, top_k=5):
        """Search for most similar properties"""
        query_emb = self.model.encode([query], device=self.device)[0]
        
        # Compute similarities
        similarities = query_emb @ self.database_embeddings.T
        
        # Get top-k
        top_k = min(top_k, len(similarities))
        scores, indices = torch.topk(similarities, k=top_k)
        
        results = []
        for idx, score in zip(indices.tolist(), scores.tolist()):
            results.append({
                "text": self.database_texts[idx],
                "score": score
            })
        
        return results

# Example usage
engine = RentalSearchEngine(model, tokenizer, device)

# Index properties
properties = [
    "Phòng trọ 25m² Quận 10, WC riêng, máy lạnh, giá 5.5tr/tháng",
    "Cho thuê phòng 30m² Quận 1, full nội thất, giá 8tr/tháng",
    "Phòng 20m² Thủ Đức, WC chung, giá 3.5tr/tháng",
    "Studio 35m² Quận 3, ban công, bếp riêng, giá 9tr/tháng",
]
engine.index(properties)

# Search
results = engine.search("phòng trọ q10 25m2 wc riêng 5tr5", top_k=3)

for i, result in enumerate(results, 1):
    print(f"{i}. [{result['score']:.4f}] {result['text']}")
```

## 🎓 Training Details

### Dataset

- **Size**: 10,384 examples
- **Split**: 9,345 train / 1,039 validation
- **Format**: Query-positive-hard negatives triplets
- **Hard Negatives**: 3 per example, weighted by feature type

### Weighted Hard Negatives Strategy

The model uses feature-based weighting for hard negatives:

| Feature Type | Weight | Importance |
|--------------|--------|------------|
| Location (Quận) | 2.5 | Highest |
| Price | 2.0 | High |
| Area (m²) | 1.8 | Medium |
| Amenities | 1.5 | Lower |

This teaches the model that location mismatches are more critical than amenity differences.

### Training Configuration

```json
{
  "base_model": "BAAI/bge-m3",
  "d_out": 128,
  "freeze_encoder": true,
  "epochs": 17,
  "batch_size": 128,
  "learning_rate": 0.0002,
  "optimizer": "AdamW",
  "weight_decay": 0.01,
  "loss": "Weighted InfoNCE (symmetric)",
  "temperature": 0.07,
  "device": "Tesla T4 (Google Colab)",
  "training_time": "~2.5 hours"
}
```

### Training Progress

| Epoch | Train Loss | Val Loss | Status |
|-------|------------|----------|--------|
| 1 | 2.9054 | 2.4529 | ⭐ Best |
| 5 | 2.1609 | 2.0078 | ⭐ Best |
| 9 | 2.0237 | 1.8906 | ⭐ Best |
| 12 | 1.9722 | 1.8760 | ⭐ Best |
| **16** | **1.9297** | **1.8215** | ⭐ **Best** |
| 17 | 1.9191 | 1.8276 | Final |

**Improvement**: -34% train loss, -26% validation loss

### Model Architecture

```
BAAI/bge-m3 (frozen)
    ↓ [1024-dim]
ProjectionHead
    ├─ Linear(1024 → 128, bias=False)
    └─ L2 Normalization
    ↓ [128-dim, L2-normalized]
Output Embeddings
```

**Parameters**:
- Trainable: 131,072 (0.02%)
- Total: 567,885,824
- Strategy: Only projection head is trainable

## 🎯 Use Cases

This model is optimized for:

✅ **Vietnamese rental property search**
- Matching user queries to property listings
- Finding similar properties
- Semantic search for rental accommodations

✅ **Supported features**:
- Location (districts, neighborhoods)
- Price range
- Area/size (m²)
- Amenities (WC, máy lạnh, ban công, bếp, etc.)
- Room type (phòng trọ, studio, etc.)

## ⚠️ Limitations

- **Domain-specific**: Optimized for Vietnamese rental properties only
- **Geographic focus**: Primarily trained on properties in Ho Chi Minh City and Hanoi
- **Language**: Vietnamese only (not multilingual like base BGE-M3)
- **Frozen encoder**: Base BGE-M3 encoder is not fine-tuned, only projection head
- **Not for**: General-purpose Vietnamese embeddings or other domains

## 🔍 Example Predictions

### Example 1: Location Sensitivity

```
Query: "phòng trọ Gò Vấp 18m² 3tr5 có wc riêng"

Positive (0.947):  Gò Vấp 18m² 3tr5 wc riêng ✅
Negative 1 (0.366): Quận 12 18m² 3tr5 wc riêng (wrong district!)
Negative 2 (0.411): Gò Vấp 18m² 4tr2 wc riêng (wrong price)
Negative 3 (0.828): Gò Vấp 18m² 3tr5 wc chung (wrong amenity)

→ Model correctly penalizes location mismatch most heavily
```

### Example 2: Feature Understanding

```
Query: "phòng trọ q10 4tr 20m² có máy lạnh wc riêng gần chợ"

Positive (0.904):  Q10 20m² 4tr máy lạnh wc riêng ✅
Negative 1 (0.542): Q3 20m² 4tr máy lạnh wc riêng (wrong district)
Negative 2 (0.418): Q10 20m² 5.5tr máy lạnh wc riêng (wrong price)
Negative 3 (0.257): Q10 15m² 4tr máy lạnh wc chung (multiple diffs)

→ Strong margin (+0.36) between positive and top negative
```

## 📖 Citation

If you use this model, please cite:

```bibtex
@misc{bge-m3-vietnamese-rental,
  author = {Your Name},
  title = {BGE-M3 Vietnamese Rental Property Search},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/your-username/bge-m3-vietnamese-rental-projection}},
}
```

## 📜 License

MIT License - Free to use for commercial and non-commercial purposes.

## 🙏 Acknowledgments

- Base model: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- Framework: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Training: Google Colab (Tesla T4)

## 📧 Contact

For questions or feedback, please open an issue on the model repository.

---

**Last updated**: October 2025

