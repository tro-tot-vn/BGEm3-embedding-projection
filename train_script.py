#!/usr/bin/env python3
"""
Training script for BGE-M3 projection head with weighted hard negatives
Vietnamese Rental Market - Phòng Trọ Embedding

Usage:
    python train_script.py [--config CONFIG_FILE]
    python train_script.py --help
    
Environment Support:
    - Local development
    - Google Colab: !python path/to/train_script.py
    - Jupyter Notebook
"""

import json
import torch
import os
import sys
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# ===== Auto-detect Project Root =====
# This allows the script to work from any directory (Colab, local, etc.)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

# Add project root to Python path to import modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import BGEM3WithHead
from train import ContrastiveTrainer
from pair_dataset import PairDataset, collate


# ===== Helper Functions for Path Resolution =====
def resolve_path(path_str):
    """Resolve path relative to project root if not absolute"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except:
        return False


# ===== Default Configuration =====
# Note: Paths will be auto-resolved relative to PROJECT_ROOT
DEFAULT_CONFIG = {
    # Data
    "data_path": "data/gen-data-set.json",  # Will be resolved to absolute path
    "train_split": 0.9,  # 90% train, 10% validation
    
    # Model
    "d_out": 256,
    "freeze_encoder": True,
    "use_layernorm": False,
    
    # Training
    "batch_size": 128,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "epochs": 10,
    "max_length": 512,
    "use_hard_neg": True,
    
    # Optimization
    "gradient_clip_norm": 1.0,
    "warmup_steps": 0,  # Set to > 0 to enable warmup
    
    # Checkpointing
    "output_dir": "checkpoints",  # Will be resolved to absolute path
    "save_every": 2,  # Save every N epochs
    "save_best": True,
    
    # Logging
    "log_every": 10,  # Log every N batches
    "validate_every": 1,  # Validate every N epochs
    
    # Device
    "device": "auto",  # "auto", "cuda", or "cpu"
}


def load_config(config_path=None):
    """Load configuration from file or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        print(f"📄 Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)
    
    return config


def setup_device(device_config):
    """Setup device (auto-detect or specified)"""
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def load_dataset(config):
    """Load and split dataset"""
    # Resolve data path relative to project root
    data_path = resolve_path(config["data_path"])
    
    print(f"\n📊 Loading dataset")
    print(f"   Project root: {PROJECT_ROOT}")
    print(f"   Data path: {data_path}")
    
    # Check if file exists with helpful error message
    if not data_path.exists():
        print(f"\n❌ Dataset file not found: {data_path}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check if file exists in project:")
        print(f"      ls {PROJECT_ROOT / 'data' / 'gen-data-set.json'}")
        print(f"   2. Current working directory: {Path.cwd()}")
        print(f"   3. Project root detected: {PROJECT_ROOT}")
        print(f"   4. Available data files:")
        data_dir = PROJECT_ROOT / 'data'
        if data_dir.exists():
            for f in data_dir.glob('*.json'):
                print(f"      - {f.name}")
        else:
            print(f"      ⚠️  Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Split train/val
    if config["train_split"] < 1.0:
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(
            data, 
            train_size=config["train_split"], 
            random_state=42
        )
        print(f"   Train: {len(train_data)} examples")
        print(f"   Val:   {len(val_data)} examples")
    else:
        train_data = data
        val_data = None
        print(f"   Using all data for training (no validation)")
    
    # Create datasets
    train_dataset = PairDataset(train_data, use_hard_neg=config["use_hard_neg"])
    val_dataset = PairDataset(val_data, use_hard_neg=config["use_hard_neg"]) if val_data else None
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=0,  # Set to 2-4 if you have CPU cores
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
            num_workers=0,
        )
    
    print(f"✅ Train batches: {len(train_loader)}")
    if val_loader:
        print(f"✅ Val batches:   {len(val_loader)}")
    
    return train_loader, val_loader


def initialize_model(config, device):
    """Initialize model and trainer"""
    print(f"\n🤖 Initializing model")
    print(f"   Output dimension: {config['d_out']}")
    print(f"   Freeze encoder: {config['freeze_encoder']}")
    print(f"   Use LayerNorm: {config['use_layernorm']}")
    
    model = BGEM3WithHead(
        d_out=config["d_out"],
        freeze_encoder=config["freeze_encoder"],
        use_layernorm=config["use_layernorm"]
    ).to(device)
    
    trainer = ContrastiveTrainer(model)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Model initialized")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable ratio:  {100 * trainable_params / total_params:.2f}%")
    
    return model, trainer


def setup_optimizer(model, config):
    """Setup optimizer"""
    optimizer = torch.optim.AdamW(
        model.head.parameters(),  # Only optimize projection head
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    print(f"\n⚙️  Optimizer: AdamW")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Weight decay:  {config['weight_decay']}")
    
    return optimizer


def validate(model, trainer, val_loader, device):
    """Run validation"""
    if val_loader is None:
        return None
    
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            loss = trainer.training_step(batch)
            total_loss += loss.item()
    
    model.train()
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, config, filename):
    """Save training checkpoint"""
    # Resolve output directory relative to project root
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / filename
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }, checkpoint_path)
    
    return checkpoint_path


def train(config):
    """Main training function"""
    
    # Print header with environment info
    print("=" * 80)
    print("🚀 BGE-M3 PROJECTION HEAD TRAINING")
    print("   Vietnamese Rental Market (Phòng Trọ)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📍 Environment:")
    print(f"   Platform: {'Google Colab' if is_colab() else 'Local/Other'}")
    print(f"   Working dir: {Path.cwd()}")
    print(f"   Project root: {PROJECT_ROOT}")
    print(f"   Python: {sys.version.split()[0]}")
    
    # Setup
    device = setup_device(config["device"])
    train_loader, val_loader = load_dataset(config)
    model, trainer = initialize_model(config, device)
    optimizer = setup_optimizer(model, config)
    
    # Training state
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    # Create output directory (resolved to absolute path)
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Config saved to: {config_path}")
    
    # Training loop
    print(f"\n🏋️  Starting training for {config['epochs']} epochs")
    print("=" * 80)
    
    model.train()
    
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        batch_count = 0
        
        # Progress bar
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{config['epochs']}",
            ncols=100
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            optimizer.zero_grad()
            loss = trainer.training_step(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            if config["gradient_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.head.parameters(), 
                    max_norm=config["gradient_clip_norm"]
                )
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            avg_loss = epoch_loss / batch_count
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}'
            })
            
            # Periodic logging
            if (batch_idx + 1) % config["log_every"] == 0:
                tqdm.write(
                    f"   Batch {batch_idx+1}/{len(train_loader)}: "
                    f"loss={loss.item():.4f}, avg_loss={avg_loss:.4f}"
                )
        
        # Epoch summary
        avg_train_loss = epoch_loss / len(train_loader)
        
        print(f"\n📈 Epoch {epoch+1}/{config['epochs']} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_loss = None
        if val_loader and (epoch + 1) % config["validate_every"] == 0:
            val_loss = validate(model, trainer, val_loader, device)
            print(f"   Val Loss:   {val_loss:.4f}")
            
            # Check if best validation loss
            if config["save_best"] and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = save_checkpoint(
                    model, optimizer, epoch + 1, val_loss, config,
                    "bgem3_projection_best.pt"
                )
                print(f"   ⭐ New best val loss! Saved to: {best_path}")
        
        # Check if best train loss (when no validation)
        if val_loader is None and config["save_best"] and avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_path = save_checkpoint(
                model, optimizer, epoch + 1, avg_train_loss, config,
                "bgem3_projection_best.pt"
            )
            print(f"   ⭐ New best train loss! Saved to: {best_path}")
        
        # Periodic checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch + 1, avg_train_loss, config,
                f"bgem3_projection_epoch{epoch+1}.pt"
            )
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
        
        print("-" * 80)
    
    # Final save
    output_dir = resolve_path(config["output_dir"])
    final_path = output_dir / "bgem3_projection_final.pt"
    torch.save(model.state_dict(), final_path)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   - Final model:  bgem3_projection_final.pt")
    if config["save_best"]:
        print(f"   - Best model:   bgem3_projection_best.pt")
    print(f"   - Config:       config.json")
    
    if val_loss:
        print(f"\n📊 Final Results:")
        print(f"   Best Val Loss:   {best_val_loss:.4f}")
        print(f"   Final Train Loss: {avg_train_loss:.4f}")
    else:
        print(f"\n📊 Final Results:")
        print(f"   Best Train Loss:  {best_train_loss:.4f}")
        print(f"   Final Train Loss: {avg_train_loss:.4f}")
    
    print("\n🎉 Ready for inference!")
    print(f"   Load model: model.load_state_dict(torch.load('{final_path}'))")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train BGE-M3 projection head for Vietnamese rental market"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (optional, uses defaults if not provided)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data JSON (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for checkpoints (overrides config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Device to use (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command-line arguments
    if args.data:
        config["data_path"] = args.data
    if args.output:
        config["output_dir"] = args.output
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.device:
        config["device"] = args.device
    
    # Run training
    try:
        train(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("   Checkpoint saved in:", resolve_path(config["output_dir"]))
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

