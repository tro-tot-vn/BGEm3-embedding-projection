#!/usr/bin/env python3
"""
Final verification: Compare HF Hub model vs Original model
"""

import torch
from transformers import AutoModel
from model import BGEM3WithHead
from pathlib import Path

print("=" * 80)
print("🔍 FINAL VERIFICATION: HF Hub vs Original")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
project_root = Path(__file__).parent

# Test texts
texts = [
    "Phòng trọ Quận 10, 25m², giá 5 triệu, WC riêng",
    "phòng trọ q1 30m2 8tr full nội thất",
    "Cho thuê phòng Bình Thạnh, 20m², 4 triệu/tháng"
]

print(f"\n📝 Test texts: {len(texts)}")
print(f"🖥️  Device: {device}")

# Load HF model from Hub
print("\n" + "=" * 80)
print("1. Loading model from HF Hub")
print("=" * 80)

print("🔄 Downloading from lamdx4/bge-m3-vietnamese-rental-projection...")
hf_model = AutoModel.from_pretrained(
    "lamdx4/bge-m3-vietnamese-rental-projection",
    trust_remote_code=True
).to(device)
hf_model.eval()
print("   ✅ Loaded")

# Load original model
print("\n" + "=" * 80)
print("2. Loading original model")
print("=" * 80)

print("🔄 Loading from local checkpoint...")
orig_model = BGEM3WithHead(d_out=128, freeze_encoder=True, use_layernorm=False).to(device)
checkpoint = torch.load(
    project_root / "checkpoints" / "bgem3_projection_best.pt",
    map_location=device
)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    orig_model.load_state_dict(checkpoint['model_state_dict'])
else:
    orig_model.load_state_dict(checkpoint)
orig_model.eval()
print("   ✅ Loaded")

# Encode with both models
print("\n" + "=" * 80)
print("3. Encoding texts")
print("=" * 80)

print("🔄 Encoding with HF model...")
hf_embeddings = hf_model.encode(texts, device=device)
print(f"   ✅ Shape: {hf_embeddings.shape}")

print("🔄 Encoding with original model...")
with torch.no_grad():
    orig_embeddings = orig_model(texts, device=device)
print(f"   ✅ Shape: {orig_embeddings.shape}")

# Compare embeddings
print("\n" + "=" * 80)
print("4. Comparing embeddings")
print("=" * 80)

diff = torch.abs(hf_embeddings.cpu() - orig_embeddings.cpu())
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"\n📊 Difference statistics:")
print(f"   Max difference:  {max_diff:.10f}")
print(f"   Mean difference: {mean_diff:.10f}")
print(f"   Std difference:  {diff.std().item():.10f}")

if max_diff < 1e-6:
    print(f"\n✅ PERFECT! Models are identical (diff < 1e-6)")
    status = "✅ PASS"
elif max_diff < 1e-4:
    print(f"\n✅ EXCELLENT! Models are nearly identical (diff < 1e-4)")
    status = "✅ PASS"
elif max_diff < 1e-2:
    print(f"\n⚠️  WARNING! Models have small differences (diff < 1e-2)")
    status = "⚠️  WARN"
else:
    print(f"\n❌ FAILED! Models are different (diff >= 1e-2)")
    status = "❌ FAIL"
    print(f"\nSample embeddings:")
    print(f"   HF:       {hf_embeddings[0][:5].tolist()}")
    print(f"   Original: {orig_embeddings[0][:5].tolist()}")

# Test similarity consistency
print("\n" + "=" * 80)
print("5. Testing similarity consistency")
print("=" * 80)

# Compute similarity matrix with HF model
hf_sim_matrix = (hf_embeddings @ hf_embeddings.T).cpu()
print(f"\n📊 HF model similarity matrix:")
for i in range(len(texts)):
    row = " ".join([f"{hf_sim_matrix[i, j]:.4f}" for j in range(len(texts))])
    print(f"   [{row}]")

# Compute similarity matrix with original model
orig_sim_matrix = (orig_embeddings @ orig_embeddings.T).cpu()
print(f"\n📊 Original model similarity matrix:")
for i in range(len(texts)):
    row = " ".join([f"{orig_sim_matrix[i, j]:.4f}" for j in range(len(texts))])
    print(f"   [{row}]")

# Compare matrices
sim_diff = torch.abs(hf_sim_matrix - orig_sim_matrix).max().item()
print(f"\n📊 Similarity matrix difference: {sim_diff:.10f}")

if sim_diff < 1e-6:
    print(f"   ✅ Similarity matrices are identical")
else:
    print(f"   ⚠️  Similarity matrices have small differences")

# Final result
print("\n" + "=" * 80)
print(f"FINAL RESULT: {status}")
print("=" * 80)

if status == "✅ PASS":
    print("\n🎉 Model on HF Hub is working correctly!")
    print("   Users can safely download and use it.")
    print(f"\n🌐 https://huggingface.co/lamdx4/bge-m3-vietnamese-rental-projection")
else:
    print("\n⚠️  There may be issues with the uploaded model.")
    print("   Please investigate before sharing with users.")

print("\n" + "=" * 80)

