"""
Populate weight field in gen-data-set.json
Calculates weights for hard negatives based on their feature types
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from weight_calculator import WeightCalculator


def populate_dataset_weights(
    input_path="data/gen-data-set.json",
    output_path="data/gen-data-set.json",
    backup=True
):
    """
    Populate weight: 0 with calculated weights based on feature types
    
    Args:
        input_path: Path to input dataset JSON
        output_path: Path to save updated dataset
        backup: Whether to create backup of original
    """
    # Convert to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.is_absolute():
        project_root = Path(__file__).parent.parent
        input_path = project_root / input_path
        output_path = project_root / output_path
    
    print(f"📂 Loading dataset from: {input_path}")
    
    # Load dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Create backup if requested
    if backup and input_path == output_path:
        backup_path = input_path.with_suffix('.json.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print(f"💾 Backup created: {backup_path}")
    
    # Initialize weight calculator
    calc = WeightCalculator()
    
    # Process dataset
    updated_count = 0
    total_hard_negs = 0
    
    for item_idx, item in enumerate(dataset):
        if "hard_neg" not in item:
            continue
        
        for hn_idx, hn in enumerate(item["hard_neg"]):
            if not isinstance(hn, dict):
                continue
            
            total_hard_negs += 1
            
            # Calculate weight if not set or is 0
            if hn.get("weight", 0) == 0:
                feature_types = hn.get("type", [])
                
                if not feature_types:
                    print(f"⚠️  Warning: Item {item_idx}, hard_neg {hn_idx} has no types!")
                    continue
                
                # Calculate weight
                weight = calc.calculate(feature_types)
                hn["weight"] = round(weight, 2)
                
                updated_count += 1
                
                # Debug output for first few
                if updated_count <= 3:
                    types_str = ", ".join(feature_types)
                    print(f"   Updated: {types_str} → weight = {weight:.2f}")
    
    # Save updated dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ Processing complete!")
    print(f"   Total items: {len(dataset)}")
    print(f"   Total hard negatives: {total_hard_negs}")
    print(f"   Updated weights: {updated_count}")
    print(f"   Saved to: {output_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Populate weights in dataset based on feature types"
    )
    parser.add_argument(
        "--input",
        default="data/gen-data-set.json",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output",
        default="data/gen-data-set.json",
        help="Output dataset path"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup file"
    )
    
    args = parser.parse_args()
    
    populate_dataset_weights(
        input_path=args.input,
        output_path=args.output,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
