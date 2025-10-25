#!/usr/bin/env python3
"""
Fix all datasets in data/ directory
- Remove duplicate types
- Remove location mismatches
- Recalculate weights

Usage:
    python scripts/fix_all_datasets.py
    python scripts/fix_all_datasets.py --dry-run  # Preview only
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from fix_duplicate_types import deduplicate_types
from validate_dataset import validate_location_match
from weight_calculator import WeightCalculator


def fix_dataset(input_path, dry_run=False):
    """
    Fix a single dataset:
    1. Remove duplicate types
    2. Remove location mismatches
    3. Recalculate weights
    
    Args:
        input_path: Path to dataset JSON
        dry_run: If True, only report issues without fixing
    
    Returns:
        dict with statistics
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        return {
            'status': 'not_found',
            'error': f'File not found: {input_path}'
        }
    
    print(f"\n{'='*70}")
    print(f"📁 Processing: {input_path.name}")
    print(f"{'='*70}")
    
    # Load data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Failed to load: {e}'
        }
    
    original_count = len(data)
    print(f"📊 Original: {original_count} examples")
    
    # Stats
    stats = {
        'file': input_path.name,
        'original_examples': original_count,
        'duplicate_types_fixed': 0,
        'location_mismatches_removed': 0,
        'weights_updated': 0,
        'final_examples': 0,
        'status': 'success'
    }
    
    # Step 1: Fix duplicate types
    print(f"\n🔧 Step 1: Fixing duplicate types...")
    for item in data:
        if 'hard_neg' not in item:
            continue
        
        for hn in item['hard_neg']:
            if 'type' not in hn:
                continue
            
            types = hn['type']
            if len(types) != len(set(types)):
                # Has duplicates
                original = types.copy()
                deduplicated = deduplicate_types(types)
                hn['type'] = deduplicated
                stats['duplicate_types_fixed'] += 1
                
                if stats['duplicate_types_fixed'] <= 3:
                    print(f"   Fixed: {original} → {deduplicated}")
    
    print(f"✅ Fixed {stats['duplicate_types_fixed']} duplicate types")
    
    # Step 2: Remove location mismatches
    print(f"\n🔧 Step 2: Removing location mismatches...")
    valid_data = []
    
    for idx, item in enumerate(data):
        query = item.get('query', '')
        pos = item.get('pos', '')
        
        is_valid, query_locs, pos_locs = validate_location_match(query, pos)
        
        if is_valid:
            valid_data.append(item)
        else:
            stats['location_mismatches_removed'] += 1
            if stats['location_mismatches_removed'] <= 3:
                print(f"   Removed #{idx}:")
                print(f"      Query: {query[:60]}... → {query_locs}")
                print(f"      Pos:   {pos[:60]}... → {pos_locs}")
    
    data = valid_data
    print(f"✅ Removed {stats['location_mismatches_removed']} location mismatches")
    
    # Step 3: Recalculate weights
    print(f"\n🔧 Step 3: Recalculating weights...")
    calc = WeightCalculator()
    
    for item in data:
        if 'hard_neg' not in item:
            continue
        
        for hn in item['hard_neg']:
            if not isinstance(hn, dict):
                continue
            
            feature_types = hn.get('type', [])
            if not feature_types:
                continue
            
            # Calculate weight if not set or is 0
            if hn.get('weight', 0) == 0:
                weight = calc.calculate(feature_types)
                hn['weight'] = round(weight, 2)
                stats['weights_updated'] += 1
    
    print(f"✅ Updated {stats['weights_updated']} weights")
    
    stats['final_examples'] = len(data)
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"   Original examples:      {stats['original_examples']}")
    print(f"   Duplicate types fixed:  {stats['duplicate_types_fixed']}")
    print(f"   Location mismatches:    {stats['location_mismatches_removed']}")
    print(f"   Weights updated:        {stats['weights_updated']}")
    print(f"   Final examples:         {stats['final_examples']}")
    print(f"   Data loss:              {stats['original_examples'] - stats['final_examples']} ({100*(stats['original_examples']-stats['final_examples'])/stats['original_examples']:.1f}%)")
    
    # Save if not dry run
    if not dry_run:
        # Create backup
        backup_path = input_path.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        with open(backup_path, 'w', encoding='utf-8') as f:
            # Load original again for backup
            with open(input_path, 'r', encoding='utf-8') as orig:
                original_data = json.load(orig)
            json.dump(original_data, f, ensure_ascii=False, indent=4)
        print(f"\n💾 Backup created: {backup_path.name}")
        
        # Save fixed data
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"💾 Saved fixed data: {input_path.name}")
    else:
        print(f"\n🔍 DRY RUN: No changes saved")
    
    return stats


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix all datasets in data/ directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview issues without fixing"
    )
    parser.add_argument(
        "--files",
        nargs='+',
        default=None,
        help="Specific files to fix (default: all in data/)"
    )
    
    args = parser.parse_args()
    
    # Find datasets
    project_root = Path(__file__).parent.parent
    
    if args.files:
        dataset_files = [project_root / 'data' / f for f in args.files]
    else:
        data_dir = project_root / 'data'
        dataset_files = [
            f for f in data_dir.glob('*.json')
            if f.name not in ['weight-config.json', 'config.json']
            and 'backup' not in f.name
            and 'bak' not in f.name
        ]
    
    print("="*70)
    print("🔧 FIX ALL DATASETS")
    print("="*70)
    print(f"Mode: {'🔍 DRY RUN (preview only)' if args.dry_run else '✅ FIXING'}")
    print(f"Found {len(dataset_files)} dataset(s)")
    
    # Process each dataset
    all_stats = []
    
    for dataset_file in dataset_files:
        stats = fix_dataset(dataset_file, dry_run=args.dry_run)
        all_stats.append(stats)
    
    # Overall summary
    print(f"\n{'='*70}")
    print("📊 OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_original = sum(s.get('original_examples', 0) for s in all_stats if s.get('status') == 'success')
    total_final = sum(s.get('final_examples', 0) for s in all_stats if s.get('status') == 'success')
    total_dups = sum(s.get('duplicate_types_fixed', 0) for s in all_stats if s.get('status') == 'success')
    total_locs = sum(s.get('location_mismatches_removed', 0) for s in all_stats if s.get('status') == 'success')
    total_weights = sum(s.get('weights_updated', 0) for s in all_stats if s.get('status') == 'success')
    
    print(f"\nProcessed {len(all_stats)} file(s)")
    print(f"\nTotal examples:          {total_original} → {total_final}")
    print(f"Duplicate types fixed:   {total_dups}")
    print(f"Location mismatches:     {total_locs}")
    print(f"Weights updated:         {total_weights}")
    print(f"Data loss:               {total_original - total_final} ({100*(total_original-total_final)/total_original:.1f}% if total_original > 0 else 0)%)")
    
    # Per-file summary
    print(f"\n📋 Per-file summary:")
    print(f"{'File':<40} {'Original':<10} {'Final':<10} {'Quality':<10}")
    print("-"*70)
    
    for stats in all_stats:
        if stats.get('status') != 'success':
            print(f"{stats['file']:<40} {'ERROR':<10}")
            continue
        
        file_name = stats['file'][:38]
        orig = stats['original_examples']
        final = stats['final_examples']
        quality = 100 * final / orig if orig > 0 else 0
        
        print(f"{file_name:<40} {orig:<10} {final:<10} {quality:.1f}%")
    
    if args.dry_run:
        print(f"\n💡 To apply fixes, run without --dry-run:")
        print(f"   python scripts/fix_all_datasets.py")
    else:
        print(f"\n✅ All datasets fixed!")
        print(f"\n💡 Next steps:")
        print(f"   1. Validate: python scripts/validate_dataset.py --input data/FILENAME.json")
        print(f"   2. Train: python train_script.py --data data/FILENAME.json")
    
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())

