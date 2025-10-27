#!/usr/bin/env python3
"""
Prepare BGE-M3 Projection Head for Hugging Face Hub Upload

This script:
1. Loads the trained checkpoint
2. Extracts only projection head weights
3. Converts to SafeTensors format
4. Saves metadata
"""

import torch
import json
from pathlib import Path
from safetensors.torch import save_file
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_projection_weights(checkpoint_path: Path):
    """
    Extract projection head weights from checkpoint
    
    Returns:
        dict: Projection head state dict
        dict: Metadata (epoch, loss, etc.)
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check checkpoint structure
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with optimizer, epoch, etc.
        full_state_dict = checkpoint['model_state_dict']
        metadata = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'loss': float(checkpoint.get('loss', 0.0)),
            'val_loss': float(checkpoint.get('val_loss', 0.0)) if 'val_loss' in checkpoint else None
        }
        print(f"   ✓ Full checkpoint detected")
        print(f"   ✓ Epoch: {metadata['epoch']}")
        print(f"   ✓ Loss: {metadata['loss']:.4f}")
    else:
        # Just state_dict
        full_state_dict = checkpoint
        metadata = {}
        print(f"   ✓ State dict only (no metadata)")
    
    # Extract only projection head weights
    projection_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith('head.'):
            # Remove 'head.' prefix for cleaner naming
            new_key = key.replace('head.', '')
            projection_weights[new_key] = value
            print(f"   ✓ Extracted: {key} -> {new_key} {list(value.shape)}")
    
    if not projection_weights:
        raise ValueError("No projection head weights found! Keys in checkpoint: " + str(list(full_state_dict.keys())))
    
    return projection_weights, metadata


def save_as_safetensors(weights: dict, output_path: Path, metadata: dict = None):
    """Save weights in SafeTensors format"""
    print(f"\n💾 Saving to SafeTensors: {output_path}")
    
    # Convert all tensors to contiguous memory layout (required by safetensors)
    weights_contiguous = {k: v.contiguous() for k, v in weights.items()}
    
    # Add metadata
    if metadata is None:
        metadata = {}
    
    # Convert metadata values to strings (safetensors requirement)
    metadata_str = {k: str(v) for k, v in metadata.items()}
    metadata_str['format'] = 'pt'  # Required by transformers
    
    save_file(weights_contiguous, output_path, metadata=metadata_str)
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Saved! Size: {size_mb:.2f} MB")


def create_config(output_dir: Path, use_layernorm: bool = False):
    """Create config.json for the model"""
    config = {
        "model_type": "bgem3_projection",
        "base_model": "BAAI/bge-m3",
        "d_in": 1024,
        "d_out": 128,
        "use_layernorm": use_layernorm,
        "freeze_encoder": True,
        "max_length": 512,
        "architectures": ["BGEM3ProjectionModel"],
        "auto_map": {
            "AutoConfig": "modeling_bgem3_projection.BGEM3ProjectionConfig",
            "AutoModel": "modeling_bgem3_projection.BGEM3ProjectionModel"
        },
        "torch_dtype": "float32",
        "transformers_version": "4.36.0"
    }
    
    config_path = output_dir / "config.json"
    print(f"\n📝 Creating config: {config_path}")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Config saved")
    
    return config


def create_training_info(output_dir: Path, metadata: dict):
    """Create training_info.json with training details and metrics"""
    
    training_info = {
        "training": {
            "dataset_size": 10384,
            "train_examples": 9345,
            "val_examples": 1039,
            "epochs": 17,
            "best_epoch": metadata.get('epoch', 16),
            "batch_size": 128,
            "learning_rate": 0.0002,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "device": "Tesla T4 (Google Colab)",
            "training_time": "~2.5 hours"
        },
        "loss": {
            "final_train_loss": 1.9191,
            "best_val_loss": metadata.get('loss', 1.8215),
            "initial_train_loss": 2.9054,
            "initial_val_loss": 2.4529,
            "improvement": {
                "train": "-34%",
                "val": "-26%"
            }
        },
        "evaluation": {
            "test_examples": 96,
            "metrics": {
                "MRR": 0.9844,
                "Recall@1": 0.9688,
                "Recall@5": 1.0,
                "Recall@10": 1.0,
                "Recall@50": 1.0
            },
            "interpretation": "Excellent retrieval performance. 96.88% of queries find correct match at rank 1."
        },
        "model_details": {
            "base_model": "BAAI/bge-m3",
            "projection_dim": 128,
            "trainable_params": 131072,
            "total_params": 567885824,
            "trainable_ratio": "0.02%",
            "training_strategy": "Frozen encoder + trainable projection head"
        },
        "loss_function": {
            "type": "Weighted InfoNCE",
            "temperature": 0.07,
            "symmetric": True,
            "weighted_hard_negatives": True,
            "feature_weights": {
                "location": 2.5,
                "price": 2.0,
                "area": 1.8,
                "amenity": 1.5
            }
        }
    }
    
    info_path = output_dir / "training_info.json"
    print(f"\n📊 Creating training info: {info_path}")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Training info saved")
    
    return training_info


def main():
    """Main conversion pipeline"""
    print("=" * 80)
    print("🚀 Preparing BGE-M3 Projection Head for Hugging Face Hub")
    print("=" * 80)
    
    # Paths
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / "checkpoints" / "bgem3_projection_best.pt"
    output_dir = project_root / "hf_upload"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Extract weights
    print("\n" + "=" * 80)
    print("STEP 1: Extract Projection Head Weights")
    print("=" * 80)
    
    projection_weights, metadata = extract_projection_weights(checkpoint_path)
    
    # Step 2: Convert to SafeTensors
    print("\n" + "=" * 80)
    print("STEP 2: Convert to SafeTensors Format")
    print("=" * 80)
    
    safetensors_path = output_dir / "model.safetensors"
    save_as_safetensors(projection_weights, safetensors_path, metadata)
    
    # Step 3: Create config
    print("\n" + "=" * 80)
    print("STEP 3: Create Configuration Files")
    print("=" * 80)
    
    use_layernorm = 'ln.weight' in projection_weights
    config = create_config(output_dir, use_layernorm)
    
    # Step 4: Create training info
    training_info = create_training_info(output_dir, metadata)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📦 Files created:")
    print(f"   ✓ model.safetensors ({(output_dir / 'model.safetensors').stat().st_size / 1024:.1f} KB)")
    print(f"   ✓ config.json")
    print(f"   ✓ training_info.json")
    print(f"\n📝 Next steps:")
    print(f"   1. Create modeling_bgem3_projection.py")
    print(f"   2. Create README.md (Model Card)")
    print(f"   3. Upload to Hugging Face Hub")
    print("=" * 80)


if __name__ == "__main__":
    main()

