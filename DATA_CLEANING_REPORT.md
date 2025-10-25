# 🔧 Data Cleaning Report

## Overview

This report documents all data quality issues found and fixed in the BGE-M3 embedding projection dataset (`gen-data-set.json`).

---

## 🚨 Issues Found & Fixed

### Issue #1: Duplicate Types in Hard Negatives

**Problem:**
- 679 hard negative examples (3.3%) had duplicate types
- Example: `["location", "price", "amenity", "amenity"]`

**Impact:**
- Weight calculation would be incorrect (inflated)
- Model would over-weight certain features
- Training bias towards duplicated features

**Fix:**
- Created `scripts/fix_duplicate_types.py`
- Deduplicated type arrays while preserving order
- Example fix: `["amenity", "amenity"]` → `["amenity"]`

**Result:**
```
✅ Fixed 679 duplicate entries
✅ Recalculated all weights correctly
```

**Backup:** `data/gen-data-set.json.backup`

---

### Issue #2: Location Mismatches (CRITICAL)

**Problem:**
- 363 examples (5.26%) had mismatched locations between query and positive
- Example:
  ```
  Query: "Tìm trọ khu Thủ Đức diện tích 25m2 giá 5.5tr"
  Pos:   "Phòng trọ 25m2... ở Quận 10..."  ❌ WRONG LOCATION!
  ```

**Why This is Critical:**
In contrastive learning, the positive example **MUST** match the query. Location mismatches teach the model:
- `similarity("Thủ Đức", "Quận 10")` should be HIGH ❌ **WRONG!**
- This causes the model to confuse different locations
- Location is the most important feature (weight 2.5)
- Retrieval results would return wrong areas → Bad UX

**Types of Mismatches Found:**
1. **Cross-city mismatches:**
   - Query: Hà Nội (Cầu Giấy, Đống Đa) → Pos: TPHCM (Q10, Bình Thạnh)
   
2. **Different districts:**
   - Query: Quận 3 → Pos: Quận 10
   
3. **Missing locations:**
   - Query: "phòng trọ q10..." → Pos: "{pos}" (template not filled)

**Fix:**
- Created `scripts/validate_dataset.py`
- Validates location match between query and positive
- Removes invalid entries (363 examples)
- Normalizes location names (Q10 = Q.10 = Quận 10)

**Result:**
```
✅ Removed 363 invalid entries (5.26%)
✅ 0 location mismatches remaining
✅ Dataset size: 6901 → 6538 examples
```

**Backup:** `data/gen-data-set.json.invalid_backup`

---

## 📊 Final Dataset Statistics

```
Total examples:              6,538
Total hard negatives:        19,319
Avg hard negatives/example:  2.95

Weight Distribution:
  Min:     1.00 (single feature errors)
  Max:     3.85 (4 feature errors)
  Mean:    2.06
  Median:  2.30
```

### Type Distribution (Top 5)

| Type Combination | Count | Percentage |
|------------------|-------|------------|
| `['location']` | 4,429 | 22.9% |
| `['amenity']` | 3,388 | 17.5% |
| `['amenity', 'area']` | 2,976 | 15.4% |
| `['amenity', 'price']` | 2,586 | 13.4% |
| `['price']` | 1,807 | 9.4% |

---

## 🔧 Tools Created

### 1. `scripts/fix_duplicate_types.py`
```bash
# Remove duplicate types from hard negatives
python scripts/fix_duplicate_types.py

# Options:
python scripts/fix_duplicate_types.py --input data/gen-data-set.json --output data/cleaned.json
python scripts/fix_duplicate_types.py --no-backup
```

**Features:**
- Deduplicates type arrays while preserving order
- Creates automatic backup
- Shows first 5 fixes for verification

### 2. `scripts/validate_dataset.py`
```bash
# Validate dataset (check only)
python scripts/validate_dataset.py

# Validate and fix
python scripts/validate_dataset.py --fix

# Options:
python scripts/validate_dataset.py --input data/gen-data-set.json --fix
python scripts/validate_dataset.py --fix --output data/cleaned.json
python scripts/validate_dataset.py --fix --no-backup
```

**Features:**
- Validates location match between query and positive
- Normalizes location names for comparison
- Removes invalid entries with detailed reporting
- Creates automatic backup

---

## 🔄 Cleaning Workflow

If you need to clean a new dataset, follow this workflow:

```bash
# Step 1: Fix duplicate types
python scripts/fix_duplicate_types.py
# → Removes duplicate types in hard_neg

# Step 2: Validate and fix location mismatches
python scripts/validate_dataset.py --fix
# → Removes entries with location mismatches

# Step 3: Recalculate weights
python scripts/populate_weights.py
# → Updates weights based on fixed types

# Step 4: Final validation
python scripts/validate_dataset.py
# → Verify dataset is clean (should show 0 errors)
```

---

## ✅ Validation Checklist

Before training, verify:

- [x] No duplicate types in hard negatives
- [x] All query-positive pairs have matching locations
- [x] All weights are non-zero and in valid range (1.0 - 3.85)
- [x] No empty type arrays
- [x] All type values are valid (`location`, `price`, `area`, `amenity`, `requirement`)

---

## 📁 Backup Files

All cleaning operations create backups automatically:

```
data/gen-data-set.json.backup          # Before duplicate fix
data/gen-data-set.json.bak             # Before weight recalc
data/gen-data-set.json.invalid_backup  # Before location fix
```

To restore from backup:
```bash
# Restore from any backup
cp data/gen-data-set.json.backup data/gen-data-set.json
```

---

## 🎯 Impact on Training

### Before Cleaning:
- ❌ 679 duplicate types → incorrect weights
- ❌ 363 location mismatches → model learns wrong associations
- ❌ Training would be unstable and inaccurate

### After Cleaning:
- ✅ Correct weights for all hard negatives
- ✅ All query-positive pairs are valid
- ✅ Dataset size reduced by 5.26% but quality improved
- ✅ Ready for high-quality training

**Expected Improvements:**
- Better location understanding
- More stable training (no conflicting signals)
- Higher retrieval accuracy
- Better metrics (MRR, Recall@K)

---

## 💡 Recommendations

### For Future Data Generation:

1. **Validate location match** in data generation script
   - Ensure query location matches positive location
   - Use normalized location comparison

2. **Deduplicate types** before saving
   - Use `set()` or check for duplicates in generation

3. **Add validation step** to data pipeline
   - Run `validate_dataset.py` after generation
   - Fix issues before training

4. **Create test set** with known good examples
   - Manually verify a sample of 50-100 examples
   - Use as gold standard for validation

### For Training:

1. **Split train/val AFTER cleaning**
   - Current split: 90% train, 10% val
   - Already handled in `train_script.py`

2. **Monitor training metrics**
   - If training is unstable → check data quality
   - If val loss plateaus early → may need more data

3. **Evaluate on held-out test set**
   - Use `evaluate_model.py` after training
   - Check if location retrieval is accurate

---

## 📞 Contact

If you encounter data quality issues not covered here, document them and add fixes to this workflow.

**Last Updated:** 2025-10-25
**Dataset Version:** v1.1 (cleaned)
**Total Examples:** 6,538
**Quality Score:** ✅ High (all known issues fixed)

