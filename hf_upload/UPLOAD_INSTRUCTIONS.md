# Hugging Face Hub Upload Instructions

## Files Ready for Upload

All files are in the `hf_upload/` directory:

```
hf_upload/
├── model.safetensors          # Projection head weights (512 KB)
├── config.json                 # Model configuration
├── modeling_bgem3_projection.py # Model class definition
├── training_info.json          # Training metrics and details
└── README.md                   # Model Card
```

## Step-by-Step Upload Process

### 1. Install Hugging Face CLI (if not already installed)

```bash
pip install huggingface_hub
```

### 2. Login to Hugging Face

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted. Get your token from: https://huggingface.co/settings/tokens

### 3. Create Repository

```bash
huggingface-cli repo create bge-m3-vietnamese-rental-projection --type model
```

This creates a new model repository: `https://huggingface.co/YOUR_USERNAME/bge-m3-vietnamese-rental-projection`

### 4. Upload Files

#### Option A: Using huggingface-cli (Recommended)

```bash
cd hf_upload

# Upload all files at once
huggingface-cli upload YOUR_USERNAME/bge-m3-vietnamese-rental-projection . . --repo-type model
```

#### Option B: Using Git

```bash
cd hf_upload

# Clone the empty repo
git clone https://huggingface.co/YOUR_USERNAME/bge-m3-vietnamese-rental-projection
cd bge-m3-vietnamese-rental-projection

# Copy files
cp ../model.safetensors .
cp ../config.json .
cp ../modeling_bgem3_projection.py .
cp ../training_info.json .
cp ../README.md .

# Commit and push
git add .
git commit -m "Initial upload: BGE-M3 Vietnamese rental projection head"
git push
```

#### Option C: Using Python

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload each file
api.upload_file(
    path_or_fileobj="model.safetensors",
    path_in_repo="model.safetensors",
    repo_id="YOUR_USERNAME/bge-m3-vietnamese-rental-projection",
    repo_type="model",
)

# Repeat for other files...
```

### 5. Update README.md

Before uploading, update `README.md` with your Hugging Face username:

1. Replace `your-username` with your actual username (appears 2 times)
2. Update the citation section with your name
3. Add your contact information if desired

### 6. Verify Upload

After uploading, visit:
```
https://huggingface.co/YOUR_USERNAME/bge-m3-vietnamese-rental-projection
```

You should see:
- ✅ Model Card (README.md) displayed
- ✅ Files tab shows all 5 files
- ✅ Model can be loaded with `from_pretrained()`

### 7. Test Download (Important!)

```python
from transformers import AutoTokenizer
import sys
sys.path.insert(0, "path/to/hf_upload")  # Add for trust_remote_code

# Import model class
from modeling_bgem3_projection import BGEM3ProjectionModel, BGEM3ProjectionConfig

# Load from Hub
config = BGEM3ProjectionConfig.from_pretrained(
    "YOUR_USERNAME/bge-m3-vietnamese-rental-projection"
)
model = BGEM3ProjectionModel.from_pretrained(
    "YOUR_USERNAME/bge-m3-vietnamese-rental-projection",
    config=config,
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

# Test encoding
texts = ["Phòng trọ Quận 10, 25m², giá 5tr"]
embeddings = model.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")  # Should be [1, 128]
```

## Troubleshooting

### Issue: "trust_remote_code" error

**Solution**: Make sure to use `trust_remote_code=True` when loading the model.

### Issue: Weight loading warnings

The warnings about encoder weights not being initialized are **expected**. We only upload projection head weights; the encoder is loaded from BAAI/bge-m3 separately.

### Issue: NumPy version error

**Solution**: Use `pip install "numpy<2.0"` if you encounter TensorFlow compatibility issues.

## Additional Configuration

### Add Model Tags

You can add tags to your model page for better discoverability. In the README.md front matter:

```yaml
---
language:
- vi
tags:
- sentence-transformers
- vietnamese
- rental
- real-estate
- bge-m3
---
```

### Add to a Collection

Consider adding your model to Vietnamese NLP or real estate collections on Hugging Face.

## License

The model is released under MIT License. Make sure this is acceptable for your use case.

## Support

For issues or questions:
- Open an issue on the model repository
- Contact Hugging Face support
- Check Hugging Face documentation: https://huggingface.co/docs

---

**Ready to upload!** 🚀

Follow the steps above and your model will be publicly available for the community to use.

