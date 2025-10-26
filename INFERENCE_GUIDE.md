# 🚀 Inference Guide: Using Your Trained Model

After training, you'll have checkpoint files (`.pt`) that contain your trained model. This guide shows you how to use them.

---

## 📁 Checkpoint Files

After training, you'll find these files in `checkpoints/`:

```
checkpoints/
├── best_model.pt           # Best validation loss model (recommended)
├── checkpoint_epoch_5.pt   # Checkpoint from specific epoch
├── checkpoint_epoch_10.pt
├── final_model.pt          # Final epoch model
├── config.json             # Training configuration
└── loss_history.json       # Training metrics
```

### Which File to Use?

- **`best_model.pt`** ✅ **Recommended** - Best performing model
- **`final_model.pt`** - Last epoch (may be overfitted)
- **`checkpoint_epoch_N.pt`** - Specific epoch checkpoint

---

## 🔧 Method 1: Using the Inference Script (Easiest)

### Quick Start

```python
from inference import RentalSearchEngine

# 1. Load trained model
engine = RentalSearchEngine(
    model_path="checkpoints/best_model.pt",
    device="auto",  # "cuda", "cpu", or "auto"
    d_out=128       # Must match training config!
)

# 2. Index your property database
properties = [
    "Phòng trọ 25m2 Quận 10, WC riêng, giá 5.5 triệu",
    "Studio 35m2 Quận 3, full nội thất, giá 9 triệu",
    # ... more properties
]
engine.index_database(properties)

# 3. Search!
results = engine.search("phòng q10 wc riêng 5tr", top_k=5)

for rank, (text, score) in enumerate(results, 1):
    print(f"{rank}. [{score:.4f}] {text}")
```

### Run Demo

```bash
python inference.py
```

---

## 🔧 Method 2: Direct Model Loading (Advanced)

If you need more control:

```python
import torch
from model import BGEM3WithHead

# Initialize model architecture
model = BGEM3WithHead(
    d_out=128,           # Must match training config!
    freeze_encoder=True,
    use_layernorm=False
).to("cuda")

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pt")

# Check checkpoint type
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    # Full checkpoint (with optimizer, epoch info)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
else:
    # Just model weights (final_model.pt)
    model.load_state_dict(checkpoint)

model.eval()

# Use the model
with torch.no_grad():
    embeddings = model(["query text here"], device="cuda")
    print(embeddings.shape)  # (1, 128)
```

---

## 🎯 Common Use Cases

### 1. Semantic Search

Find properties matching a user query:

```python
from inference import RentalSearchEngine

engine = RentalSearchEngine("checkpoints/best_model.pt")
engine.index_database(your_properties)

# User searches
query = "phòng q10 25m2 wc riêng 5 triệu"
results = engine.search(query, top_k=10)

for text, score in results:
    print(f"[{score:.3f}] {text}")
```

### 2. Find Similar Properties

Find properties similar to a given one:

```python
# User is viewing property A, show similar ones
current_property = "Phòng 25m2 Q10, WC riêng, 5.5tr"

# Search using this as query
similar = engine.search(current_property, top_k=5)
print("Similar properties:")
for text, score in similar[1:]:  # Skip first (itself)
    print(f"  - {text}")
```

### 3. Compute Similarity Score

Check if a property matches user requirements:

```python
user_query = "cần phòng q10 wc riêng dưới 6 triệu"
property_desc = "Phòng 25m2 Quận 10, WC riêng, 5.5tr"

similarity = engine.compute_similarity(user_query, property_desc)

if similarity > 0.8:
    print("✅ Highly relevant!")
elif similarity > 0.6:
    print("⚠️  Somewhat relevant")
else:
    print("❌ Not relevant")
```

### 4. Batch Encoding

Encode many texts at once:

```python
# Encode all properties for database
properties = load_all_properties()  # 10,000 properties

embeddings = engine.encode(properties, batch_size=64)
# Shape: (10000, 128)

# Save to disk for fast loading
torch.save({
    'embeddings': embeddings,
    'texts': properties
}, 'property_embeddings.pt')
```

### 5. Re-ranking

Use embeddings to re-rank results from another system:

```python
# Get initial results from keyword search
keyword_results = your_keyword_search(query, top_k=100)

# Encode query
query_emb = engine.encode([query])[0]

# Encode candidates
candidate_embs = engine.encode(keyword_results)

# Compute similarities
scores = query_emb @ candidate_embs.T

# Re-rank
sorted_indices = torch.argsort(scores, descending=True)
reranked_results = [keyword_results[i] for i in sorted_indices[:10]]
```

---

## 🔄 Resume Training from Checkpoint

If you want to continue training:

```python
import torch
from model import BGEM3WithHead

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_epoch_5.pt")

# Restore model
model = BGEM3WithHead(d_out=128).to("cuda")
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer
optimizer = torch.optim.AdamW(model.head.parameters(), lr=2e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Resume training
start_epoch = checkpoint['epoch'] + 1
print(f"Resuming from epoch {start_epoch}")
```

---

## ⚙️ Configuration

### Important Parameters

When loading a model, **these must match your training config**:

```python
engine = RentalSearchEngine(
    model_path="checkpoints/best_model.pt",
    d_out=128,           # ⚠️  MUST MATCH training config!
    freeze_encoder=True, # ⚠️  MUST MATCH training config!
    use_layernorm=False  # ⚠️  MUST MATCH training config!
)
```

Check `checkpoints/config.json` to see your training config:

```bash
cat checkpoints/config.json
```

### Device Selection

```python
# Auto-detect (GPU if available, else CPU)
engine = RentalSearchEngine(model_path, device="auto")

# Force GPU
engine = RentalSearchEngine(model_path, device="cuda")

# Force CPU
engine = RentalSearchEngine(model_path, device="cpu")
```

---

## 🐛 Troubleshooting

### Error: "checkpoint not found"

```bash
# Check available checkpoints
ls checkpoints/*.pt

# Make sure you trained a model first
python train_script.py
```

### Error: "size mismatch"

You're loading a model with wrong `d_out`:

```python
# ❌ Wrong
engine = RentalSearchEngine(model_path, d_out=256)  # Model was trained with 128

# ✅ Correct
engine = RentalSearchEngine(model_path, d_out=128)  # Match training config
```

### Slow Inference

```python
# Use larger batch size for encoding
embeddings = engine.encode(texts, batch_size=128)  # Faster than batch_size=1

# Pre-index database (don't re-encode every search)
engine.index_database(properties)  # Do this once at startup
```

### GPU Out of Memory

```python
# Reduce batch size
embeddings = engine.encode(texts, batch_size=16)  # Smaller batches

# Or use CPU
engine = RentalSearchEngine(model_path, device="cpu")
```

---

## 📊 Performance Tips

### 1. Pre-index Your Database

**Don't** encode on every search:

```python
# ❌ Slow: Re-encode database every time
for query in user_queries:
    embeddings = engine.encode(database)  # Wasteful!
    results = search(query, embeddings)
```

**Do** index once at startup:

```python
# ✅ Fast: Index once, search many times
engine.index_database(database)  # Do once at startup

for query in user_queries:
    results = engine.search(query)  # Fast!
```

### 2. Batch Encoding

Encode multiple texts together:

```python
# ❌ Slow: One at a time
embeddings = [engine.encode([text])[0] for text in texts]

# ✅ Fast: Batch encoding
embeddings = engine.encode(texts, batch_size=64)
```

### 3. Save Pre-computed Embeddings

```python
# Compute once
embeddings = engine.encode(large_database)

# Save to disk
torch.save(embeddings, "database_embeddings.pt")

# Load later (instant)
embeddings = torch.load("database_embeddings.pt")
```

---

## 🎯 Integration Examples

### Flask API

```python
from flask import Flask, request, jsonify
from inference import RentalSearchEngine

app = Flask(__name__)

# Load model at startup
engine = RentalSearchEngine("checkpoints/best_model.pt")
engine.index_database(load_properties())

@app.route("/search", methods=["POST"])
def search():
    query = request.json["query"]
    top_k = request.json.get("top_k", 10)
    
    results = engine.search(query, top_k=top_k)
    
    return jsonify({
        "query": query,
        "results": [
            {"text": text, "score": float(score)}
            for text, score in results
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
from inference import RentalSearchEngine

app = FastAPI()
engine = RentalSearchEngine("checkpoints/best_model.pt")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/search")
def search(req: SearchRequest):
    results = engine.search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {"text": text, "score": score}
            for text, score in results
        ]
    }
```

---

## 📚 Next Steps

1. **Try the demo**: `python inference.py`
2. **Load your data**: Replace sample database with real properties
3. **Tune threshold**: Experiment with similarity thresholds for your use case
4. **Monitor performance**: Track latency and relevance metrics
5. **A/B test**: Compare with your existing search

---

## 🔗 Related Files

- `inference.py` - Main inference script
- `evaluate_model.py` - Evaluate model performance
- `visualize_embeddings.py` - Visualize what the model learned
- `train_script.py` - Training script

---

**Happy searching! 🚀**

