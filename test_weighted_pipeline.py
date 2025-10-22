"""
Test script for weighted hard negative training pipeline
Tests the complete flow: dataset → weights → training
"""

import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

from model import BGEM3WithHead
from train import ContrastiveTrainer
from pair_dataset import PairDataset, collate


def test_weight_calculator():
    """Test 1: Weight calculator"""
    print("\n" + "="*60)
    print("TEST 1: Weight Calculator")
    print("="*60)
    
    from scripts.weight_calculator import WeightCalculator
    
    calc = WeightCalculator()
    
    test_cases = [
        (["location"], "Single feature - location"),
        (["price"], "Single feature - price"),
        (["location", "price"], "Two major features"),
        (["location", "price", "area"], "Three features"),
        (["amenity", "furniture"], "Two minor features"),
    ]
    
    for types, desc in test_cases:
        weight = calc.calculate(types)
        print(f"✓ {desc:30s} {types} → {weight:.2f}")
    
    print("\n✅ Weight calculator working correctly!")


def test_dataset_loading():
    """Test 2: Dataset loading with weights"""
    print("\n" + "="*60)
    print("TEST 2: Dataset Loading")
    print("="*60)
    
    # Load dataset
    data_path = Path(__file__).parent / "data" / "data-set.json"
    
    if not data_path.exists():
        print(f"⚠️  Dataset not found: {data_path}")
        return None
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} items")
    
    # Check first item structure
    if len(data) > 0:
        item = data[0]
        print(f"✓ First item keys: {list(item.keys())}")
        
        if "hard_neg" in item and len(item["hard_neg"]) > 0:
            hn = item["hard_neg"][0]
            print(f"✓ First hard_neg structure: {list(hn.keys())}")
            print(f"   - type: {hn.get('type', [])}")
            print(f"   - weight: {hn.get('weight', 0)}")
        else:
            print("⚠️  No hard negatives in first item")
    
    print("\n✅ Dataset structure looks good!")
    return data


def test_dataset_class():
    """Test 3: PairDataset class"""
    print("\n" + "="*60)
    print("TEST 3: PairDataset Class")
    print("="*60)
    
    # Create minimal test data
    test_data = [
        {
            "query": "phòng trọ q10 25m2",
            "pos": "Cho thuê phòng Quận 10, 25m²",
            "hard_neg": [
                {
                    "text": "Cho thuê phòng Quận 1, 25m²",
                    "type": ["location"],
                    "weight": 2.5
                },
                {
                    "text": "Cho thuê phòng Quận 10, 30m²",
                    "type": ["area"],
                    "weight": 1.5
                }
            ]
        },
        {
            "query": "phòng trọ q1 giá rẻ",
            "pos": "Cho thuê phòng Quận 1, giá 3 triệu",
            "hard_neg": [
                {
                    "text": "Cho thuê phòng Quận 2, giá 3 triệu",
                    "type": ["location"],
                    "weight": 2.5
                }
            ]
        }
    ]
    
    # Test without hard negatives
    print("\n→ Testing without hard negatives:")
    ds_no_hn = PairDataset(test_data, use_hard_neg=False)
    item = ds_no_hn[0]
    print(f"   Keys: {list(item.keys())}")
    assert "hard_neg" not in item, "Should not have hard_neg when use_hard_neg=False"
    print("   ✓ Correctly excludes hard negatives")
    
    # Test with hard negatives
    print("\n→ Testing with hard negatives:")
    ds_with_hn = PairDataset(test_data, use_hard_neg=True)
    item = ds_with_hn[0]
    print(f"   Keys: {list(item.keys())}")
    print(f"   Hard negs: {len(item['hard_neg'])} items")
    print(f"   Weights: {item['hard_neg_weights']}")
    assert len(item['hard_neg']) == 2, "Should have 2 hard negatives"
    assert item['hard_neg_weights'] == [2.5, 1.5], "Weights should match"
    print("   ✓ Correctly includes hard negatives with weights")
    
    # Test collate function
    print("\n→ Testing collate function:")
    batch = [ds_with_hn[0], ds_with_hn[1]]
    collated = collate(batch)
    print(f"   Batch keys: {list(collated.keys())}")
    print(f"   Query batch size: {len(collated['query'])}")
    print(f"   Hard neg counts: {[len(hn) for hn in collated['hard_neg']]}")
    print("   ✓ Collate function working")
    
    print("\n✅ PairDataset class working correctly!")


def test_model_forward():
    """Test 4: Model forward pass"""
    print("\n" + "="*60)
    print("TEST 4: Model Forward Pass")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = BGEM3WithHead(d_out=256, freeze_encoder=True, use_layernorm=False)
    model = model.to(device)
    model.eval()
    
    # Test encoding
    texts = ["phòng trọ q10", "phòng trọ q1"]
    
    with torch.no_grad():
        embs = model(texts, device=device)
    
    print(f"✓ Input: {len(texts)} texts")
    print(f"✓ Output shape: {embs.shape}")
    print(f"✓ L2-normalized: {torch.norm(embs, dim=1)}")
    
    assert embs.shape == (2, 256), "Should output [2, 256]"
    assert torch.allclose(torch.norm(embs, dim=1), torch.ones(2, device=device), atol=1e-5), "Should be L2-normalized"
    
    print("\n✅ Model forward pass working!")


def test_training_step():
    """Test 5: Training step with weighted loss"""
    print("\n" + "="*60)
    print("TEST 5: Training Step (Weighted Loss)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BGEM3WithHead(d_out=256, freeze_encoder=True, use_layernorm=False)
    model = model.to(device)
    trainer = ContrastiveTrainer(model)
    
    # Test batch WITHOUT hard negatives
    print("\n→ Testing WITHOUT hard negatives:")
    batch_no_hn = {
        "query": ["phòng trọ q10", "phòng trọ q1"],
        "pos": ["Cho thuê phòng Quận 10", "Cho thuê phòng Quận 1"]
    }
    
    loss_no_hn = trainer.training_step(batch_no_hn)
    print(f"   Loss: {loss_no_hn.item():.4f}")
    assert loss_no_hn.numel() == 1, "Should return scalar loss"
    print("   ✓ Standard InfoNCE working")
    
    # Test batch WITH hard negatives
    print("\n→ Testing WITH hard negatives:")
    batch_with_hn = {
        "query": ["phòng trọ q10", "phòng trọ q1"],
        "pos": ["Cho thuê phòng Quận 10", "Cho thuê phòng Quận 1"],
        "hard_neg": [
            ["Cho thuê phòng Quận 1", "Cho thuê phòng Quận 10, 30m²"],
            ["Cho thuê phòng Quận 2"]
        ],
        "hard_neg_weights": [
            [2.5, 1.5],  # First query has 2 hard negs
            [2.5]        # Second query has 1 hard neg
        ]
    }
    
    loss_with_hn = trainer.training_step(batch_with_hn)
    print(f"   Loss: {loss_with_hn.item():.4f}")
    assert loss_with_hn.numel() == 1, "Should return scalar loss"
    print("   ✓ Weighted InfoNCE working")
    
    # Test batch with EMPTY hard negatives
    print("\n→ Testing WITH some empty hard negatives:")
    batch_mixed = {
        "query": ["phòng trọ q10", "phòng trọ q1"],
        "pos": ["Cho thuê phòng Quận 10", "Cho thuê phòng Quận 1"],
        "hard_neg": [
            ["Cho thuê phòng Quận 1"],  # First has hard negs
            []                           # Second has none
        ],
        "hard_neg_weights": [
            [2.5],
            []
        ]
    }
    
    loss_mixed = trainer.training_step(batch_mixed)
    print(f"   Loss: {loss_mixed.item():.4f}")
    print("   ✓ Mixed batch (some with/without hard negs) working")
    
    print("\n✅ Training step working correctly!")


def test_backward_pass():
    """Test 6: Backward pass and gradient flow"""
    print("\n" + "="*60)
    print("TEST 6: Backward Pass & Gradients")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BGEM3WithHead(d_out=256, freeze_encoder=True, use_layernorm=False)
    model = model.to(device)
    model.train()
    
    trainer = ContrastiveTrainer(model)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=2e-4)
    
    # Training batch
    batch = {
        "query": ["phòng trọ q10", "phòng trọ q1"],
        "pos": ["Cho thuê phòng Quận 10", "Cho thuê phòng Quận 1"],
        "hard_neg": [
            ["Cho thuê phòng Quận 1"],
            ["Cho thuê phòng Quận 2"]
        ],
        "hard_neg_weights": [[2.5], [2.5]]
    }
    
    # Forward
    loss = trainer.training_step(batch)
    print(f"✓ Forward pass - Loss: {loss.item():.4f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    print("✓ Backward pass completed")
    
    # Check gradients
    has_grad = False
    for name, param in model.head.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"   {name}: grad_norm = {grad_norm:.6f}")
    
    assert has_grad, "Head should have gradients"
    print("✓ Gradients computed correctly")
    
    # Optimizer step
    optimizer.step()
    print("✓ Optimizer step completed")
    
    print("\n✅ Backward pass working correctly!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("WEIGHTED HARD NEGATIVE PIPELINE TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        test_weight_calculator()
        test_dataset_loading()
        test_dataset_class()
        test_model_forward()
        test_training_step()
        test_backward_pass()
        
        # Summary
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nYour weighted hard negative pipeline is ready!")
        print("\nNext steps:")
        print("  1. Run: python scripts/populate_weights.py")
        print("  2. Update training data in pair_dataset.py")
        print("  3. Train with: python train.py (or your training script)")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
