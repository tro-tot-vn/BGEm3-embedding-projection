#!/usr/bin/env python3
"""
Fix duplicate types in hard_neg field
Deduplicates type arrays while preserving order

Example:
    ["location", "amenity", "amenity"] → ["location", "amenity"]
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def deduplicate_types(type_list):
    """Remove duplicates while preserving order"""
    seen = set()
    result = []
    for t in type_list:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def fix_dataset(input_path, output_path=None, backup=True):
    """
    Fix duplicate types in dataset
    
    Args:
        input_path: Path to input JSON
        output_path: Path to save fixed JSON (default: same as input)
        backup: Whether to create backup
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    print(f"📂 Loading dataset from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Create backup if requested
    if backup and input_path == output_path:
        backup_path = input_path.with_suffix('.json.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Backup created: {backup_path}")
    
    # Fix duplicates
    fixed_count = 0
    total_hn = 0
    
    for item_idx, item in enumerate(data):
        if 'hard_neg' not in item:
            continue
        
        for hn_idx, hn in enumerate(item['hard_neg']):
            if not isinstance(hn, dict) or 'type' not in hn:
                continue
            
            total_hn += 1
            types = hn['type']
            
            # Check for duplicates
            if len(types) != len(set(types)):
                # Fix it
                original = types.copy()
                deduplicated = deduplicate_types(types)
                hn['type'] = deduplicated
                fixed_count += 1
                
                # Show first few fixes
                if fixed_count <= 5:
                    print(f"\n  Fix {fixed_count}:")
                    print(f"    Before: {original}")
                    print(f"    After:  {deduplicated}")
    
    # Save fixed dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ FIXING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total hard negatives: {total_hn}")
    print(f"Fixed duplicates: {fixed_count} ({100*fixed_count/total_hn:.1f}%)")
    print(f"Saved to: {output_path}")
    
    return fixed_count


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix duplicate types in hard_neg field"
    )
    parser.add_argument(
        "--input",
        default="data/gen-data-set.json",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dataset path (default: same as input)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup file"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_path = (project_root / args.output) if args.output else None
    
    print("="*60)
    print("🔧 FIX DUPLICATE TYPES IN DATASET")
    print("="*60)
    
    try:
        fixed_count = fix_dataset(
            input_path, 
            output_path, 
            backup=not args.no_backup
        )
        
        if fixed_count == 0:
            print("\n✨ No duplicates found! Dataset is clean.")
        else:
            print(f"\n✨ Fixed {fixed_count} duplicate type entries!")
            print("\n💡 Next step:")
            print("   python scripts/populate_weights.py")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

