#!/usr/bin/env python3
"""
Test script for Hugging Face model upload

This script tests:
1. Loading model from local directory
2. Encoding texts
3. Computing similarities
4. Comparing with original checkpoint
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_local_loading():
    """Test loading model from local hf_upload directory"""
    print("=" * 80)
    print("TEST 1: Loading Model from Local Directory")
    print("=" * 80)
    
    from transformers import AutoTokenizer
    
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "hf_upload"
    
    print(f"\n📂 Loading from: {model_dir}")
    
    # Add hf_upload to path so we can import the model class
    sys.path.insert(0, str(model_dir))
    
    try:
        # Import the custom model class
        from modeling_bgem3_projection import BGEM3ProjectionModel, BGEM3ProjectionConfig
        
        # Load config
        config = BGEM3ProjectionConfig.from_pretrained(str(model_dir))
        
        # Load model
        model = BGEM3ProjectionModel.from_pretrained(
            str(model_dir),
            config=config
        )
        print("   ✓ Model loaded successfully")
        
        # Load tokenizer (from base model)
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        print("   ✓ Tokenizer loaded")
        
        # Check model structure
        print(f"\n📊 Model structure:")
        print(f"   - Config: {model.config.model_type}")
        print(f"   - Base model: {model.config.base_model}")
        print(f"   - d_out: {model.config.d_out}")
        print(f"   - use_layernorm: {model.config.use_layernorm}")
        
        # Check parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n🔢 Parameters:")
        print(f"   - Trainable: {trainable_params:,}")
        print(f"   - Total: {total_params:,}")
        print(f"   - Ratio: {trainable_params/total_params*100:.2f}%")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        raise


def test_encoding(model, tokenizer):
    """Test encoding functionality"""
    print("\n" + "=" * 80)
    print("TEST 2: Encoding Texts")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"\n🖥️  Device: {device}")
    
    # Test texts
    texts = [
        "Phòng trọ Quận 10, 25m², giá 5 triệu, WC riêng, máy lạnh",
        "Cho thuê phòng Bình Thạnh, 20m², 4 triệu/tháng",
        "phòng trọ q1 30m2 8tr full nội thất"
    ]
    
    print(f"\n📝 Test texts: {len(texts)} examples")
    
    # Method 1: Using encode
    print("\n🔄 Method 1: Using model.encode()")
    embeddings1 = model.encode(texts, device=device)
    print(f"   ✓ Shape: {embeddings1.shape}")
    print(f"   ✓ Device: {embeddings1.device}")
    print(f"   ✓ Dtype: {embeddings1.dtype}")
    
    # Check L2 normalization
    norms = torch.norm(embeddings1, p=2, dim=1)
    print(f"   ✓ L2 norms: {norms.tolist()}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Not L2 normalized!"
    print(f"   ✓ L2 normalized: ✅")
    
    # Method 2: Using forward
    print("\n🔄 Method 2: Using model(**inputs)")
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings2 = outputs.last_hidden_state
    
    print(f"   ✓ Shape: {embeddings2.shape}")
    print(f"   ✓ Device: {embeddings2.device}")
    
    # Check consistency
    diff = torch.abs(embeddings1.cpu() - embeddings2.cpu()).max().item()
    print(f"\n🔍 Consistency check:")
    print(f"   Max difference: {diff:.6f}")
    assert diff < 1e-5, f"Methods give different results! diff={diff}"
    print(f"   ✓ Both methods consistent: ✅")
    
    return embeddings1


def test_similarity(model, tokenizer):
    """Test similarity computation"""
    print("\n" + "=" * 80)
    print("TEST 3: Computing Similarities")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test pairs
    pairs = [
        ("phòng trọ quận 10 25m2 5tr", "Phòng trọ Q10, 25m², giá 5 triệu"),
        ("phòng trọ quận 10 25m2 5tr", "Phòng trọ Quận 1, 30m², giá 8 triệu"),
    ]
    
    print("\n📊 Similarity scores:")
    for text1, text2 in pairs:
        embs = model.encode([text1, text2], device=device)
        sim = (embs[0] @ embs[1]).item()
        
        print(f"\n   Text 1: {text1}")
        print(f"   Text 2: {text2}")
        print(f"   Similarity: {sim:.4f}")
    
    # Expected: first pair should have higher similarity
    embs1 = model.encode([pairs[0][0], pairs[0][1]], device=device)
    sim1 = (embs1[0] @ embs1[1]).item()
    
    embs2 = model.encode([pairs[1][0], pairs[1][1]], device=device)
    sim2 = (embs2[0] @ embs2[1]).item()
    
    print(f"\n✓ Similar pair similarity: {sim1:.4f}")
    print(f"✓ Different pair similarity: {sim2:.4f}")
    
    assert sim1 > sim2, f"Similar pair should have higher similarity! {sim1} <= {sim2}"
    print(f"✓ Similarity ranking correct: ✅")


def test_comparison_with_original():
    """Compare HF model with original checkpoint"""
    print("\n" + "=" * 80)
    print("TEST 4: Comparing with Original Checkpoint")
    print("=" * 80)
    
    from model import BGEM3WithHead
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).parent.parent
    
    # Add hf_upload to path
    sys.path.insert(0, str(project_root / "hf_upload"))
    from modeling_bgem3_projection import BGEM3ProjectionModel, BGEM3ProjectionConfig
    
    # Load HF model
    print("\n📦 Loading HF model...")
    config = BGEM3ProjectionConfig.from_pretrained(str(project_root / "hf_upload"))
    hf_model = BGEM3ProjectionModel.from_pretrained(
        str(project_root / "hf_upload"),
        config=config
    ).to(device)
    hf_model.eval()
    
    # Load original model
    print("📦 Loading original model...")
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
    
    # Test texts
    texts = [
        "Phòng trọ Quận 10, 25m², giá 5tr",
        "phòng trọ q1 30m2 8tr"
    ]
    
    print(f"\n🔄 Encoding test texts...")
    
    # HF model
    hf_embeddings = hf_model.encode(texts, device=device)
    
    # Original model
    with torch.no_grad():
        orig_embeddings = orig_model(texts, device=device)
    
    # Compare
    diff = torch.abs(hf_embeddings.cpu() - orig_embeddings.cpu()).max().item()
    mean_diff = torch.abs(hf_embeddings.cpu() - orig_embeddings.cpu()).mean().item()
    
    print(f"\n📊 Comparison results:")
    print(f"   Max difference: {diff:.8f}")
    print(f"   Mean difference: {mean_diff:.8f}")
    
    if diff < 1e-5:
        print(f"   ✓ Models are identical: ✅")
    elif diff < 1e-3:
        print(f"   ✓ Models are very similar: ✅")
    else:
        print(f"   ⚠️  Models have noticeable differences")
        print(f"\n   HF embedding sample: {hf_embeddings[0][:5].tolist()}")
        print(f"   Original embedding sample: {orig_embeddings[0][:5].tolist()}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 HUGGING FACE MODEL TESTING")
    print("=" * 80)
    
    try:
        # Test 1: Load model
        model, tokenizer = test_local_loading()
        
        # Test 2: Encoding
        embeddings = test_encoding(model, tokenizer)
        
        # Test 3: Similarity
        test_similarity(model, tokenizer)
        
        # Test 4: Compare with original
        test_comparison_with_original()
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\n📝 Next steps:")
        print("   1. Review the generated files in hf_upload/")
        print("   2. Update README.md with your username")
        print("   3. Upload to Hugging Face Hub:")
        print("      cd hf_upload")
        print("      huggingface-cli login")
        print("      huggingface-cli repo create bge-m3-vietnamese-rental-projection")
        print("      huggingface-cli upload bge-m3-vietnamese-rental-projection . .")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

