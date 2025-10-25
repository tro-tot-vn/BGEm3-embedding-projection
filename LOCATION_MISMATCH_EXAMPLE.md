# 🚨 Location Mismatch Example - Why This is Critical

## The Case You Asked About

```json
{
  "query": "Tìm trọ khu Thủ Đức diện tích 25m2 giá 5.5tr",
  "pos": "Phòng trọ 25m2 có gác, wc khép kín, máy lạnh, ban công, giá 5.5tr ở Quận 10 gần ĐH Bách Khoa"
}
```

## ❌ The Problem

| Field | Location Mentioned | Issue |
|-------|-------------------|-------|
| **Query** | 🏠 **Thủ Đức** | User is searching for Thủ Đức |
| **Positive** | 🏠 **Quận 10** | Example is in Quận 10 |
| **Match?** | ❌ **NO!** | Different locations! |

---

## 🔬 Why This is a CRITICAL Error

### In Contrastive Learning:

Contrastive learning teaches the model by **contrasting** examples:
- **Positive pairs** should be **SIMILAR** (high similarity score)
- **Negative pairs** should be **DIFFERENT** (low similarity score)

### What the Model Would Learn (WRONG):

```python
# With this bad data, model learns:
embedding("Tìm trọ Thủ Đức") ≈ embedding("Phòng trọ Quận 10")
                                        ↑
                                    WRONG!

# This means:
similarity("Thủ Đức", "Quận 10") → 0.85 (HIGH)  ❌ INCORRECT!
```

### Correct Behavior Should Be:

```python
# Model should learn:
embedding("Tìm trọ Thủ Đức") ≈ embedding("Phòng trọ Thủ Đức")
                                        ↑
                                    CORRECT!

# This means:
similarity("Thủ Đức", "Thủ Đức") → 0.90 (HIGH)  ✅ CORRECT!
similarity("Thủ Đức", "Quận 10") → 0.20 (LOW)   ✅ CORRECT!
```

---

## 🌍 Real-World Impact

### Scenario: User searches for room in Thủ Đức

**With bad data (before fix):**
```
User Query: "Tìm phòng trọ Thủ Đức 5tr"

Top Results:
1. 🏠 Phòng trọ Quận 10...    ❌ Wrong location!
2. 🏠 Phòng trọ Bình Thạnh... ❌ Wrong location!
3. 🏠 Phòng trọ Đống Đa...    ❌ Wrong location!
```

**With cleaned data (after fix):**
```
User Query: "Tìm phòng trọ Thủ Đức 5tr"

Top Results:
1. 🏠 Phòng trọ Thủ Đức...    ✅ Correct!
2. 🏠 Phòng trọ Thủ Đức...    ✅ Correct!
3. 🏠 Phòng trọ Thủ Đức...    ✅ Correct!
```

**User satisfaction:**
- Before: 😡 Frustrated (all results wrong area)
- After:  😊 Happy (relevant results)

---

## 📊 Statistics

### Location Mismatches Found:

```
Total mismatches:     363 (5.26% of dataset)
Examples removed:     363
Dataset before:       6,901 examples
Dataset after:        6,538 examples
Quality improvement:  ⚠️ Low → ✅ High
```

### Types of Mismatches:

1. **Cross-city** (most severe):
   ```
   Query: Hà Nội → Pos: TPHCM  (139 cases)
   Query: TPHCM → Pos: Hà Nội  (127 cases)
   ```

2. **Different districts**:
   ```
   Query: Quận 3 → Pos: Quận 10  (51 cases)
   Query: Thủ Đức → Pos: Bình Thạnh  (46 cases)
   ```

---

## 🧪 Technical Explanation

### InfoNCE Loss Function:

```python
# Contrastive loss pushes:
# - Query CLOSE to Positive
# - Query FAR from Negatives

loss = -log(
    exp(sim(query, positive)) / 
    (exp(sim(query, positive)) + Σ exp(sim(query, negative_i)))
)
```

### With Location Mismatch:

```python
Query:    "Tìm trọ Thủ Đức 5tr"
Positive: "Phòng Quận 10 5tr"  ← WRONG location!

# Model is forced to:
# 1. Make query similar to Quận 10 (because it's labeled "positive")
# 2. But user wants Thủ Đức!
# 3. → Model learns wrong associations
```

### Feature Importance:

```python
Feature Weights (from weight-config.json):
- location:    2.5  ← MOST IMPORTANT!
- price:       2.0
- area:        1.5
- amenity:     1.0
- requirement: 1.2
```

**Location is the MOST important feature!** Errors here are most damaging.

---

## ✅ How We Fixed It

### Tool Created: `scripts/validate_dataset.py`

**What it does:**
1. Extracts location mentions from query and positive
2. Normalizes location names (handles variations):
   ```python
   "Q10" = "Q.10" = "Quận 10" = "quan 10"
   ```
3. Checks if ANY location matches
4. Removes entries with no match

**Example validation:**

```python
# VALID PAIRS:
Query: "phòng Q10"     → Pos: "Phòng Quận 10"     ✅ Match!
Query: "tro Thu Duc"   → Pos: "Trọ Thủ Đức"       ✅ Match!

# INVALID PAIRS:
Query: "phòng Q10"     → Pos: "Phòng Bình Thạnh"  ❌ No match
Query: "tro Thu Duc"   → Pos: "Trọ Quận 3"        ❌ No match
```

---

## 🎯 Conclusion

### Is this a dataset error?

**YES! Absolutely!** ✅

This is a **CRITICAL** error because:
- ❌ Violates fundamental principle of contrastive learning
- ❌ Teaches model wrong location associations
- ❌ Most important feature (location) is corrupted
- ❌ Results in poor retrieval quality
- ❌ Bad user experience

### What we did:

- ✅ Detected 363 such cases (5.26%)
- ✅ Removed all invalid entries
- ✅ Created validation tool for future datasets
- ✅ Backed up original data
- ✅ Dataset is now clean and ready

### Your specific case:

```
Status: ✅ REMOVED (correctly identified as invalid)
Reason: Query asked for Thủ Đức, but positive was Quận 10
Action: Entry removed from training data
Backup: Preserved in data/gen-data-set.json.invalid_backup
```

---

## 📚 References

- **Contrastive Learning Theory:** Positive examples must match query intent
- **InfoNCE Loss:** Requires correctly labeled positive/negative pairs
- **Feature Importance:** Location is highest weight (2.5)
- **User Experience:** Wrong location = worst possible error

---

**Last Updated:** 2025-10-25  
**Dataset Version:** v1.1 (cleaned)  
**Validation Status:** ✅ All location mismatches removed

