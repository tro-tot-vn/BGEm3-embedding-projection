# 🗺️ Landmark Strategy Update Report

**Date:** October 25, 2025  
**File Modified:** `GEMINI_PROMPT_DATASET_GENERATION.md`  
**Lines Changed:** 309 → 431 (+122 lines)

---

## 📋 Summary

Updated the landmark/location guidance in the Gemini dataset generation prompt to **prevent location mismatch errors** while maintaining dataset realism.

---

## ❌ Problems with Old Approach

### 1. **Risky Specific Landmarks Without Context**
```markdown
OLD:
**TPHCM:**
- ĐH Bách Khoa, ĐH Kinh Tế, ĐH KHTN, ĐH Sư Phạm
- Chợ Bến Thành, Vincom, Lotte Mart
```

**Issues:**
- ❌ No indication which landmark belongs to which district
- ❌ Gemini AI could randomly mix landmarks with wrong districts
- ❌ Example in prompt had errors: "Quận 3 gần Công viên Tao Đàn" (wrong!), "Quận 10 gần Chợ Bến Thành" (wrong!)

### 2. **Impossible to Complete**
- Cannot list all landmarks for all districts (500+ locations)
- Not scalable or maintainable
- Over-specification limits AI creativity

### 3. **No Guidance for Uncertainty**
- What should AI do if unsure about landmark location?
- No safe fallback options provided

---

## ✅ New Approach: Simplified 3-Strategy System

### **Strategy 1: Specific Landmarks (30%)**
- Use ONLY if 100% certain landmark-district pair is correct
- Examples: "Quận 10 gần ĐH Bách Khoa" (if certain)

### **Strategy 2: Generic Landmarks (50% - Recommended)**
- Use generic terms that work for ANY district
- Always safe, always correct
- Examples: "gần chợ", "gần siêu thị", "gần trường học", "đường chính", "hẻm yên tĩnh"
- Includes generic street terms (no specific names)

### **Strategy 3: No Landmark (20%)**
- Omit landmarks entirely
- Safest option - zero risk
- Example: "Phòng trọ 25m² Quận 10, giá 5tr"

**Note:** Original Strategy 3 (specific street names) was removed because:
- ❌ Same risk as specific landmarks (can't verify all streets in all districts)
- ❌ Inconsistent with "don't list specifics" principle
- ✅ Generic street terms moved to Strategy 2 instead

---

## 🎯 Key Improvements

### **1. Clear Decision Tree**
```
Do I know the landmark-district pair is correct?
│
├─ YES (100% certain) → Strategy 1 (specific)
├─ MAYBE / NOT SURE → Strategy 2 (generic) or 4 (omit)
└─ NO → Strategy 2 (generic) or 4 (omit)
```

### **2. Explicit Warning**
> **Better to have NO landmark than a WRONG landmark!**
> - Wrong landmark = Dataset corruption = Model learns wrong
> - No landmark = Safe = Model focuses on price/area/amenities

### **3. Distribution Target**
Clear guidance on how to mix strategies:
- 30% specific (only when certain)
- 50% generic (most frequent - includes generic street terms)
- 20% no landmark

### **4. Fixed Examples**
**OLD (risky):**
```json
"Phòng trọ 25m² Quận 3 ... gần Công viên Tao Đàn"  ❌
"Phòng trọ 18m² Quận 10 ... gần Chợ Bến Thành"     ❌
```

**NEW (safe):**
```json
"Phòng trọ 25m² Quận 3 ... gần siêu thị"           ✅
"Phòng trọ 18m² Quận 10 ... gần trường học"        ✅
```

---

## 📊 Expected Impact

| Aspect | Before | After |
|--------|--------|-------|
| **Location mismatch risk** | 🔴 HIGH | 🟢 LOW |
| **Guidance clarity** | ⚠️ Vague | ✅ Clear |
| **Scalability** | ❌ Limited | ✅ Flexible |
| **Safe fallbacks** | ❌ None | ✅ 3 options |
| **Data realism** | ✅ Good | ✅ Good (maintained) |
| **Correctness** | ⚠️ Risky | ✅ Safe |

---

## 🔍 Validation Strategy

### **Prevention (New Prompt)**
- Clear instructions to use safe strategies when uncertain
- Multiple fallback options provided
- Strong warnings about wrong landmark consequences

### **Detection (Existing Scripts)**
- `validate_dataset.py` will still catch any remaining errors
- But **prevention >> detection** (better to avoid creating errors)

---

## 💡 Key Principle

```
🎯 GOAL: Realistic data, but NEVER sacrifice correctness!

Hierarchy:
1. Correct data without landmarks >>> Realistic data with wrong landmarks
2. Generic landmarks (always safe) >>> Specific landmarks (risky if wrong)
3. Prevention >>> Detection
```

---

## ✅ Conclusion

This update makes the dataset generation process:
- **Safer:** Multiple fallback strategies prevent errors
- **Clearer:** Explicit decision tree and warnings
- **Scalable:** Works for any location without exhaustive mapping
- **Practical:** Acknowledges that complete landmark mapping is impossible

**Result:** Higher quality datasets with minimal location mismatch errors!

---

## 📝 Files Modified

1. ✅ `GEMINI_PROMPT_DATASET_GENERATION.md`
   - Replaced "Landmarks to Mention" section
   - Updated example with safe generic landmarks
   - Implemented 3-strategy system (removed risky specific street names)
   - Added decision tree and warnings
   - File size: 432 lines → 413 lines

---

**Status:** ✅ Complete  
**Next Step:** Use updated prompt to generate new datasets!

