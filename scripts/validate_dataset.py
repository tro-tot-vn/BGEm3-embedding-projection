#!/usr/bin/env python3
"""
Validate dataset for location mismatches between query and positive examples
Remove or flag invalid entries

Usage:
    python scripts/validate_dataset.py [--fix] [--output cleaned.json]
"""

import json
import re
from pathlib import Path


# Location definitions
LOCATIONS = [
    'Thủ Đức', 'Thu Duc', 'Quận 1', 'Quận 2', 'Quận 3', 'Quận 4', 
    'Quận 5', 'Quận 6', 'Quận 7', 'Quận 8', 'Quận 9', 'Quận 10',
    'Quận 11', 'Quận 12', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 
    'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q.1', 'Q.2', 'Q.3',
    'Q.4', 'Q.5', 'Q.6', 'Q.7', 'Q.8', 'Q.9', 'Q.10', 'Q.11', 'Q.12',
    'Bình Thạnh', 'Binh Thanh', 'Tân Bình', 'Tan Binh',
    'Phú Nhuận', 'Phu Nhuan', 'Gò Vấp', 'Go Vap',
    'Đống Đa', 'Dong Da', 'Ba Đình', 'Ba Dinh', 
    'Hai Bà Trưng', 'Hai Ba Trung', 'Cầu Giấy', 'Cau Giay',
    'Hoàng Mai', 'Hoang Mai', 'Thanh Xuân', 'Thanh Xuan',
    'Long Biên', 'Long Bien', 'Hà Đông', 'Ha Dong'
]


def normalize_location(loc):
    """Normalize location names for comparison"""
    loc = loc.lower().strip()
    
    # Map variations to standard form
    mappings = {
        'thu duc': 'thủ đức',
        'binh thanh': 'bình thạnh',
        'tan binh': 'tân bình',
        'phu nhuan': 'phú nhuận',
        'go vap': 'gò vấp',
        'dong da': 'đống đa',
        'ba dinh': 'ba đình',
        'hai ba trung': 'hai bà trưng',
        'cau giay': 'cầu giấy',
        'hoang mai': 'hoàng mai',
        'thanh xuan': 'thanh xuân',
        'long bien': 'long biên',
        'ha dong': 'hà đông',
    }
    
    for key, val in mappings.items():
        if key in loc:
            return val
    
    # Handle Q1-Q12, Q.1-Q.12 notation
    for i in range(1, 13):
        patterns = [f'q{i}', f'q.{i}', f'q. {i}', f'q {i}', f'quận {i}', f'quan {i}']
        for p in patterns:
            if p in loc:
                return f'quận {i}'
    
    return loc


def extract_locations(text):
    """Extract all locations mentioned in text"""
    text_lower = text.lower()
    found = []
    
    for loc in LOCATIONS:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(loc.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(loc)
    
    return list(set(found))  # Remove duplicates


def validate_location_match(query, pos):
    """
    Check if query and positive example have matching locations
    
    Returns:
        (is_valid, query_locs, pos_locs)
    """
    query_locs = extract_locations(query)
    pos_locs = extract_locations(pos)
    
    # If no location specified in query, assume it's OK
    if not query_locs:
        return True, [], pos_locs
    
    # If query has location but pos doesn't, it's invalid
    if not pos_locs:
        return False, query_locs, []
    
    # Normalize and compare
    query_norm = set([normalize_location(loc) for loc in query_locs])
    pos_norm = set([normalize_location(loc) for loc in pos_locs])
    
    # Check if ANY location matches
    has_match = bool(query_norm.intersection(pos_norm))
    
    return has_match, query_locs, pos_locs


def validate_dataset(input_path, fix=False, output_path=None, backup=True):
    """
    Validate dataset for location mismatches
    
    Args:
        input_path: Path to input JSON
        fix: If True, remove invalid entries
        output_path: Path to save cleaned dataset
        backup: Whether to create backup
    """
    input_path = Path(input_path)
    if not input_path.is_absolute():
        project_root = Path(__file__).parent.parent
        input_path = project_root / input_path
    
    print("="*70)
    print("🔍 VALIDATING DATASET FOR LOCATION MISMATCHES")
    print("="*70)
    print(f"Input: {input_path}")
    
    # Load dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    
    # Validate
    invalid_items = []
    
    for idx, item in enumerate(data):
        query = item.get('query', '')
        pos = item.get('pos', '')
        
        is_valid, query_locs, pos_locs = validate_location_match(query, pos)
        
        if not is_valid:
            invalid_items.append({
                'idx': idx,
                'query': query,
                'pos': pos,
                'query_locs': query_locs,
                'pos_locs': pos_locs
            })
    
    # Report
    print(f"\n🚨 Location Mismatches: {len(invalid_items)} ({100*len(invalid_items)/len(data):.2f}%)")
    
    if len(invalid_items) > 0:
        print(f"\n📋 First 5 invalid examples:")
        for i, item in enumerate(invalid_items[:5], 1):
            print(f"\n  {i}. Index {item['idx']}:")
            print(f"     Query: {item['query'][:70]}...")
            print(f"       → Locations: {item['query_locs']}")
            print(f"     Pos: {item['pos'][:70]}...")
            print(f"       → Locations: {item['pos_locs']}")
    
    # Fix if requested
    if fix:
        print(f"\n🔧 FIXING: Removing {len(invalid_items)} invalid entries...")
        
        # Create backup
        if backup:
            backup_path = input_path.with_suffix('.json.invalid_backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"💾 Backup created: {backup_path}")
        
        # Remove invalid entries
        invalid_indices = set([item['idx'] for item in invalid_items])
        cleaned_data = [item for idx, item in enumerate(data) if idx not in invalid_indices]
        
        # Save cleaned dataset
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                project_root = Path(__file__).parent.parent
                output_path = project_root / output_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ CLEANING COMPLETE!")
        print(f"   Original: {len(data)} examples")
        print(f"   Removed:  {len(invalid_items)} examples")
        print(f"   Cleaned:  {len(cleaned_data)} examples")
        print(f"   Saved to: {output_path}")
    else:
        print(f"\n💡 To fix these issues, run:")
        print(f"   python scripts/validate_dataset.py --fix")
    
    print("="*70)
    
    return len(invalid_items)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate dataset for location mismatches"
    )
    parser.add_argument(
        "--input",
        default="data/gen-data-set.json",
        help="Input dataset path"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove invalid entries"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for cleaned dataset (default: same as input)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup file"
    )
    
    args = parser.parse_args()
    
    invalid_count = validate_dataset(
        args.input,
        fix=args.fix,
        output_path=args.output,
        backup=not args.no_backup
    )
    
    if invalid_count > 0 and not args.fix:
        print("\n⚠️  Dataset has location mismatch errors!")
        return 1
    elif invalid_count > 0 and args.fix:
        print("\n✅ Dataset cleaned successfully!")
        return 0
    else:
        print("\n✅ Dataset is valid!")
        return 0


if __name__ == "__main__":
    exit(main())

