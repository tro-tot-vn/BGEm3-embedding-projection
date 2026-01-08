

```
# Task: Generate Training Dataset for Vietnamese Rental Market Embedding Model

You are generating training data for a contrastive learning model that learns embeddings for Vietnamese rental property (phòng trọ) search.

## Dataset Structure

Each example must have this exact JSON structure:

{
  "query": "user search query in Vietnamese",
  "pos": "matching positive description",
  "hard_neg": [
    {
      "text": "similar but wrong description",
      "type": ["feature1", "feature2"],
      "weight": 0
    }
  ]
}

## ⚠️ CRITICAL RULES (Must Follow Strictly)

### Rule 1: Location MUST Match Between Query and Positive ⭐ MOST IMPORTANT

- If query mentions "Quận 10" → positive MUST also be in "Quận 10"
- If query mentions "Thủ Đức" → positive MUST also be in "Thủ Đức"
- If query mentions "Hà Nội" → positive MUST also be in "Hà Nội"
- **NEVER mix cities:** Hà Nội ≠ TPHCM
- **NEVER mix districts:** Quận 10 ≠ Quận 3 ≠ Bình Thạnh

This is THE MOST CRITICAL rule. Location mismatches destroy model quality.

❌ BAD EXAMPLE (WRONG):
{
  "query": "phòng trọ Thủ Đức 5tr",
  "pos": "Phòng trọ Quận 10 5tr có máy lạnh..."
}
→ WRONG! Query asks for Thủ Đức but positive is Quận 10!

✅ GOOD EXAMPLE (CORRECT):
{
  "query": "phòng trọ Thủ Đức 5tr",
  "pos": "Phòng trọ Thủ Đức 5tr có máy lạnh, wc riêng..."
}
→ CORRECT! Both query and positive are in Thủ Đức!

### Rule 2: Positive Must Match Query Intent

The positive example must satisfy ALL requirements in the query:
- **Location:** EXACT match (same district/city)
- **Price:** Within ±10% of query (5tr query → 4.5tr-5.5tr positive OK)
- **Area:** Within ±20% of query (25m² query → 20m²-30m² positive OK)
- **Amenities:** All mentioned amenities must be present

Example:
Query: "phòng Q10, 25m², 5tr, có máy lạnh, wc riêng"
✅ Positive must have: Q10 + 22-28m² + 4.5-5.5tr + máy lạnh + wc riêng

### Rule 3: Hard Negatives Should Be "Almost Right"

Hard negatives are examples that look similar but have 1-3 key differences.
They should be "hard to distinguish" but still wrong.

Good hard negatives change ONLY:
- Location (same price, area, amenities, but different district)
- Price (same location, area, but different price)
- Area (same location, price, but different size)
- Amenity (same location, price, area, but missing key feature)

❌ TOO EASY NEGATIVE (avoid):
Query: "Phòng Q10, 25m², 5tr, có máy lạnh"
Bad negative: "Nhà 100m², Hà Nội, 20tr, 3 phòng ngủ"
→ TOO different! Not useful for training.

✅ GOOD HARD NEGATIVE:
Query: "Phòng Q10, 25m², 5tr, có máy lạnh"
Good negative: "Phòng Q3, 25m², 5tr, có máy lạnh, wc riêng"
→ Almost identical, but wrong district!

### Rule 4: Type Field - NO DUPLICATES

Valid types: ["location", "price", "area", "amenity", "requirement"]

For each hard negative, specify which features DIFFER from the positive:

❌ BAD (WRONG):
{
  "type": ["amenity", "amenity", "location"]
}
→ "amenity" appears twice! This breaks weight calculation!

✅ GOOD (CORRECT):
{
  "type": ["amenity", "location"]
}
→ Each type appears once only!

#### Type Classification Guide:

| Type | When to Use | Example |
|------|-------------|---------|
| `location` | Different district or city | Q10 vs Q3, TPHCM vs Hà Nội |
| `price` | Price differs by >10% | 5tr vs 6tr |
| `area` | Area differs by >20% | 20m² vs 30m² |
| `amenity` | Missing or different amenities | Có máy lạnh vs Không máy lạnh, WC riêng vs WC chung |
| `requirement` | Different restrictions | Nữ only vs Nam/Nữ, Không chung chủ vs Chung chủ |

## 📍 Data Generation Guidelines

### Locations to Use (Choose from these)

**TPHCM Districts:**
- Central: Quận 1, Quận 3, Quận 5, Quận 10
- Popular: Bình Thạnh, Tân Bình, Phú Nhuận, Gò Vấp
- Emerging: Quận 2, Quận 7, Quận 9, Thủ Đức
- Others: Quận 4, Quận 6, Quận 8, Quận 11, Quận 12

**Hà Nội Districts:**
- Central: Đống Đa, Ba Đình, Hai Bà Trưng, Hoàn Kiếm
- Popular: Cầu Giấy, Thanh Xuân, Hoàng Mai
- Emerging: Long Biên, Hà Đông, Nam Từ Liêm, Bắc Từ Liêm

### Price Ranges (Realistic)

- **Budget:** 2.5tr - 4tr (basic, shared bathroom, no AC)
- **Mid-range:** 4tr - 6tr (private bathroom, AC, some furniture)
- **Premium:** 6tr - 10tr (fully furnished, balcony, new building)

### Area Ranges (Realistic)

- **Small:** 15m² - 20m² (single room, minimal space)
- **Medium:** 20m² - 30m² (comfortable, can fit furniture)
- **Large:** 30m² - 40m² (spacious, can have loft/gác)

### Common Amenities (Use These Terms)

**Essential:**
- wc riêng / wc khép kín (private bathroom)
- wc chung (shared bathroom)
- điều hòa / máy lạnh (air conditioning)
- ban công (balcony)
- gác / gác lửng (loft)

**Additional:**
- tủ lạnh (fridge)
- máy giặt (washing machine)
- giường (bed)
- bàn, ghế (table, chair)
- tủ quần áo (wardrobe)
- bếp (kitchen)
- máy nước nóng / bình nóng lạnh (water heater)

**Restrictions:**
- nữ only / chỉ nữ (female only)
- nam/nữ (both genders)
- không chung chủ (no landlord living with you)
- chung chủ (landlord lives in same building)
- nuôi pet / không nuôi pet (pets allowed/not allowed)

### Query Variations (Generate Diverse Styles)

1. **Natural language (30%):**
   - "Tìm phòng trọ Quận 10 giá rẻ có máy lạnh"
   - "Cần thuê phòng quận 3 gần Đại học Kinh Tế"

2. **Short form (30%):**
   - "phong tro q10 4tr"
   - "tro q3 25m2 may lanh"

3. **Detailed (20%):**
   - "Cần thuê phòng Q10, diện tích 25m², giá 5tr, có máy lạnh, wc riêng, ban công"
   - "Phòng trọ quận Đống Đa, 20m2, 4tr5, có đầy đủ nội thất"

4. **Conversational (20%):**
   - "Cho thuê phòng quận 10 gần chợ không?"
   - "Có phòng nào Q3 khoảng 5tr không ạ?"
   - "Tìm trọ Bình Thạnh gần trường học có không?"

### Landmarks Strategy (Add Realism Safely)

🎯 **GOAL:** Make data realistic, but NEVER sacrifice correctness!

⚠️ **CRITICAL RULE:** 
**Better to have NO landmark than a WRONG landmark!**
- Wrong landmark-district pair = Dataset corruption = Model learns wrong associations
- No landmark = Safe = Model focuses on other features (price, area, amenities)

---

#### **Strategy 1: Specific Landmarks (Use ONLY if 100% certain)**

Only use specific landmarks if you have CONFIRMED knowledge that the landmark exists in that specific district.

**Decision process:**
```
Ask yourself: Do I KNOW for certain this landmark is in this district?
├─ YES (100% verified) → Use it!
│  Example: "Phòng trọ [District] gần [verified landmark]"
│
└─ NO / UNSURE → Use Strategy 2 (generic) or Strategy 3 (omit) instead!
   DO NOT GUESS!
```

**How to verify:**
- You have reliable knowledge that the landmark belongs to that district
- You would stake the dataset quality on this information being correct
- When in doubt → DON'T USE! Use generic terms instead.

**When to use:** 30% of entries (only when 100% confident!)

---

#### **Strategy 2: Generic Landmarks (Always Safe - Recommended)**

Use generic terms that apply to ANY district:

**Generic Location Terms:**
- "gần chợ" (near local market)
- "gần siêu thị" (near supermarket)
- "gần bệnh viện" (near hospital)
- "gần trường học" (near school)
- "gần công viên" (near park)
- "gần trung tâm quận" (near district center)
- "khu dân cư" (residential area)
- "đường chính" / "đường lớn" (main road)
- "hẻm yên tĩnh" (quiet alley)
- "gần ngã tư" (near intersection)
- "gần bến xe" (near bus station)

**Example Usage:**
```json
"Phòng trọ 25m² Quận 3, giá 5tr gần chợ và trường học"
"Phòng trọ 20m² Bình Thạnh, giá 4tr trên đường chính, hẻm yên tĩnh"
```
✅ Always correct! (every district has markets, schools, main roads, etc.)

**When to use:** 50% of entries (most frequent - recommended default)

---

#### **Strategy 3: No Landmark (Safest)**

Simply describe the room without any landmark:

**Examples:**
```json
"Phòng trọ 25m² Quận 10, có máy lạnh, wc riêng, giá 5tr"
"Phòng trọ 30m² Thanh Xuân, đầy đủ nội thất, giá 6tr"
```

✅ Still complete and useful data!
✅ Zero risk of location mismatch!
✅ Model focuses on price/area/amenities (often more important than landmarks anyway)

**When to use:** 20% of entries

---

#### **Distribution Target:**

Aim for this mix across your dataset:
- 30% Strategy 1 (specific landmarks, only if 100% certain)
- 50% Strategy 2 (generic - safest and most realistic)
- 20% Strategy 3 (no landmark)

---

#### **⛔ What NOT to Do:**

**DON'T guess or assume landmark locations:**
```json
❌ "Phòng trọ Quận 10 gần Chợ Bến Thành"
   → Only if you KNOW Chợ Bến Thành is in Q10!

❌ "Phòng trọ Quận 3 gần ĐH Bách Khoa"
   → Only if you KNOW ĐHBK is in Q3!

❌ "Phòng trọ Bình Thạnh gần Hồ Gươm"
   → Only if you KNOW Hồ Gươm is in Bình Thạnh!
```

**If unsure → Use Strategy 2 (generic) or Strategy 3 (omit)!**

---

### 💡 Summary Decision Tree:

```
Do I know the landmark-district pair is correct?
│
├─ YES (100% certain)
│  └─→ Use Strategy 1 (specific landmark)
│
├─ MAYBE / NOT SURE
│  └─→ Use Strategy 2 (generic) or Strategy 3 (omit)
│
└─ NO
   └─→ Use Strategy 2 (generic) or Strategy 3 (omit)
```

**Remember:** Validation can catch some errors, but prevention is better!

## 📝 Complete Example

Here's a perfect example following all rules:

{
  "query": "Tìm phòng trọ Quận 10, 25m², giá 5tr, có máy lạnh, wc riêng",
  "pos": "Phòng trọ 25m² Quận 10, có máy lạnh, wc riêng, ban công, giá 5tr gần chợ",
  "hard_neg": [
    {
      "text": "Phòng trọ 25m² Quận 3, có máy lạnh, wc riêng, ban công, giá 5tr gần siêu thị",
      "type": ["location"],
      "weight": 0
    },
    {
      "text": "Phòng trọ 25m² Quận 10, có máy lạnh, wc riêng, ban công, giá 7tr",
      "type": ["price"],
      "weight": 0
    },
    {
      "text": "Phòng trọ 18m² Quận 10, có máy lạnh, wc chung, không ban công, giá 5tr gần trường học",
      "type": ["area", "amenity"],
      "weight": 0
    },
    {
      "text": "Phòng trọ 25m² Quận 10, không máy lạnh, wc riêng, ban công, giá 5tr",
      "type": ["amenity"],
      "weight": 0
    }
  ]
}

Explanation:
✅ Used generic landmarks ("gần chợ", "gần siêu thị", "gần trường học") - Always safe!
✅ Some entries have no landmark - Also safe and valid!
✅ All location matches are guaranteed correct
✅ Query asks for Q10 → Positive is in Q10 (location match!)
✅ Query asks for 25m² → Positive is 25m² (area match!)
✅ Query asks for 5tr → Positive is 5tr (price match!)
✅ Query asks for máy lạnh, wc riêng → Positive has both!
✅ Each hard negative differs in 1-2 specific ways
✅ Type field correctly identifies differences
✅ No duplicate types

## ✅ Quality Checklist

Before outputting each example, verify:

- [ ] Query location matches positive location EXACTLY (same district)
- [ ] Positive satisfies ALL query requirements (price, area, amenities)
- [ ] Generated 3-5 hard negatives per example
- [ ] Each hard negative has 1-3 clear, specific differences
- [ ] Type field correctly identifies ALL differences
- [ ] NO duplicate types in any type array
- [ ] All prices in reasonable range (2tr-10tr)
- [ ] All areas in reasonable range (15m²-40m²)
- [ ] All text is in natural, correct Vietnamese
- [ ] All landmarks are either verified-correct, generic, or omitted (no guessing!)

## 🎯 Output Format

Generate 50 examples at a time in valid JSON array format:

[
  {
    "query": "...",
    "pos": "...",
    "hard_neg": [...]
  },
  {
    "query": "...",
    "pos": "...",
    "hard_neg": [...]
  },
  ...
]

Make sure JSON is valid (proper quotes, commas, brackets).

## ❌ Common Mistakes to AVOID

1. ❌ **Location mismatch** between query and positive (MOST CRITICAL!)
   - Query: "phòng Q10" → Pos: "Phòng Q3" ← WRONG!

2. ❌ **Duplicate types** in type array
   - "type": ["amenity", "amenity"] ← WRONG!

3. ❌ **Hard negatives too similar** (no clear difference)
   - All features identical ← Not useful!

4. ❌ **Hard negatives too different** (completely unrelated)
   - Query: Room in TPHCM → Negative: House in Đà Nẵng ← Too different!

5. ❌ **Unrealistic prices or areas**
   - 0.5tr or 50tr ← Unrealistic!
   - 5m² or 200m² ← Not for phòng trọ!

6. ❌ **Missing required amenities from query in positive**
   - Query asks for "có máy lạnh" → Positive doesn't mention it ← WRONG!

7. ❌ **Inconsistent location naming**
   - Use standard forms: "Quận 10" or "Q10" (be consistent)

8. ❌ **Too few hard negatives**
   - Generate at least 3 hard negatives per example

## 🚀 Now Generate!

Please generate 15 high-quality training examples following ALL rules above.

Focus on:
- Diverse locations (mix TPHCM and Hà Nội)
- Diverse price ranges (budget, mid, premium)
- Diverse query styles (natural, short, detailed)
- Perfect location matching between query and positive
- High-quality hard negatives with clear differences
- NO duplicate types

Start generating now!
```
