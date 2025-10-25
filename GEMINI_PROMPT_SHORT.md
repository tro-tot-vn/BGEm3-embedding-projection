# 🤖 Gemini Prompt (Short Version)

**Quick prompt for generating Vietnamese phòng trọ dataset**

---

## 📋 COPY & PASTE THIS TO GEMINI

```
Generate 50 Vietnamese rental property (phòng trọ) training examples for contrastive learning.

CRITICAL RULES:
1. Query location MUST EXACTLY match positive location (same district!)
2. Positive must satisfy ALL query requirements
3. Generate 3-5 hard negatives per example (1-3 differences each)
4. Types: ["location", "price", "area", "amenity", "requirement"] - NO DUPLICATES
5. Weights: always 0 (will be calculated later)

LOCATIONS:
- TPHCM: Q1-12, Thủ Đức, Bình Thạnh, Tân Bình, Phú Nhuận, Gò Vấp
- Hà Nội: Đống Đa, Ba Đình, Cầu Giấy, Hai Bà Trưng, Thanh Xuân

RANGES:
- Price: 2.5tr - 10tr
- Area: 15m² - 40m²
- Amenities: wc riêng/chung, máy lạnh, ban công, gác, tủ lạnh

FORMAT:
[{
  "query": "Tìm phòng Q10, 25m², 5tr, có máy lạnh",
  "pos": "Phòng 25m² Q10, máy lạnh, wc riêng, 5tr gần ĐH Bách Khoa",
  "hard_neg": [
    {"text": "Phòng 25m² Q3, máy lạnh, 5tr...", "type": ["location"], "weight": 0},
    {"text": "Phòng 25m² Q10, máy lạnh, 7tr...", "type": ["price"], "weight": 0},
    {"text": "Phòng 18m² Q10, wc chung, 5tr...", "type": ["area", "amenity"], "weight": 0}
  ]
}]

AVOID:
❌ Query "Q10" → Pos "Q3" (location mismatch!)
❌ ["amenity", "amenity"] (duplicate types!)
❌ Too few hard negatives (<3)

Generate now with diverse locations, prices, and query styles!
```

---

## ⚡ AFTER GENERATION

```bash
# Validate & fix
python scripts/validate_dataset.py --input data/output.json --fix
python scripts/populate_weights.py --input data/output.json

# Train
python train_script.py --data data/output.json
```

**Target:** 0% location mismatches, 0% duplicates, >95% valid!

