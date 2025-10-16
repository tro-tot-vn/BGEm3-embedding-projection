# gen_triplets.py
import json, re, time, random, argparse
from pathlib import Path
from typing import List, Dict, Any

# pip install google-generativeai python-dotenv tqdm
import google.generativeai as genai
from tqdm import trange

# ================== CẤU HÌNH ==================
MODEL_NAME = "gemini-2.5-flash"
API_KEY = "AIzaSyBoutqJZTOfuueOujaGrscfwhIcZNOg-rM"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ALLOWED_TYPES = {"amenity", "price", "location", "area"}  # có thể mở rộng: "deposit", "noise", ...
DEFAULT_WEIGHT = 0

DISTRICTS_HCMC = [
    "Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 7",
    "Quận 10", "Quận 11", "Bình Thạnh", "Phú Nhuận",
    "Gò Vấp", "Tân Bình", "Tân Phú", "Thủ Đức", "Bình Tân"
]

AMENITIES = [
    "WC khép kín", "WC chung", "máy lạnh", "ban công", "thang máy",
    "bếp trong phòng", "bếp chung", "chỗ để xe", "giờ giấc tự do", "máy giặt chung"
]


# ================== TIỆN ÍCH ==================
def extract_json_block(text: str) -> str:
    """
    Cố gắng lấy khối JSON thuần từ response (nếu model có in thêm text).
    Ưu tiên mảng [...] dài nhất; nếu không có sẽ lấy object {...}.
    """
    candidates = re.findall(r'(\[.*\]|\{.*\})', text, flags=re.DOTALL)
    if not candidates:
        return text.strip()
    # chọn khối dài nhất để giảm rủi ro bị cắt
    return max(candidates, key=len).strip()


def loads_safely(text: str) -> Any:
    """Tải JSON với vài sửa lỗi nhỏ thường gặp (dấu phẩy dư)."""
    raw = extract_json_block(text)
    try:
        return json.loads(raw)
    except Exception:
        raw = re.sub(r",\s*([}\]])", r"\1", raw)  # bỏ trailing commas
        return json.loads(raw)


def normalize_type(t: str) -> str:
    t = (t or "").strip().lower()
    if t in ALLOWED_TYPES:
        return t
    # ánh xạ đơn giản
    mapping = {
        "amenities": "amenity",
        "tiện ích": "amenity",
        "khu vực": "location",
        "địa điểm": "location",
        "giá": "price",
        "diện tích": "area",
    }
    return mapping.get(t, "amenity")  # fallback "amenity"


def postprocess_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Đảm bảo đúng schema:
    {
      "query": str,
      "pos": str,
      "hard_neg": [{"text": str, "type": [..], "weight": 0}, ...]
    }
    """
    item["query"] = str(item.get("query", "")).strip()
    item["pos"] = str(item.get("pos", "")).strip()

    hard_neg = item.get("hard_neg") or []
    fixed = []
    for hn in hard_neg:
        text = str(hn.get("text", "")).strip()
        tlist = hn.get("type", [])
        if isinstance(tlist, str):
            tlist = [tlist]
        tlist = [normalize_type(x) for x in (tlist or [])]
        # lọc rỗng + ràng buộc tập allowed
        tlist = [x for x in tlist if x in ALLOWED_TYPES]
        if not tlist:
            # nếu model không gán type hợp lệ, đoán đại loại phổ biến
            tlist = [random.choice(list(ALLOWED_TYPES))]
        fixed.append({
            "text": text,
            "type": tlist,
            "weight": int(hn.get("weight", DEFAULT_WEIGHT))
        })
    item["hard_neg"] = fixed
    return item


# ================== PROMPT ==================
PROMPT = r"""
Bạn là hệ thống tạo dữ liệu cho website TÌM PHÒNG TRỌ tiếng Việt.

DỮ LIỆU GỐC (gọi là "pos"):
---
{pos}
---

MỤC TIÊU
- Sinh dữ liệu HUẤN LUYỆN hợp lý, tự nhiên như người dùng Việt Nam.
- Ưu tiên bám sát "pos". Cho phép SUY DIỄN NHẸ về cách viết số/địa danh/tiện ích tương đương (không mâu thuẫn rõ ràng):
  • "5tr5" ~ "5,5tr" ~ "5.5tr"   • "q10" ~ "q.10" ~ "quận 10"   • "wc riêng" ~ "wc khép kín"

NGÔN NGỮ & PHONG CÁCH QUERIES
- Ngắn gọn, tự nhiên, giống người Việt tìm trọ.
- Cho phép viết tắt/không dấu/slang/thiếu chủ ngữ/ghép số-chữ:
  • ví dụ: "q10", "q.10", "quan 10", "tphcm", "sg", "wc rieng", "full nt", "co gac", "gan dhbk",
           "co may lanh", "5tr5", "5,5tr", "5.5tr", "duoi 6tr", "25m2", "25 m"
- Đa dạng hình thức: câu hỏi ngắn / list từ khóa / mô tả cực ngắn.

HARD NEGATIVES
- Mỗi record phải có EXACTLY {k_neg} phần tử "hard_neg".
- Mỗi "hard_neg" là mô tả GẦN GIỐNG "pos" nhưng CỐ Ý SAI 1 HOẶC NHIỀU NHÓM THUỘC TÍNH:
  "location", "price", "area", "amenity", "other".
- "type" là MẢNG gồm ≥1 phần tử, chọn từ: ["location","price","area","amenity","other"].
- "weight" luôn = 0.

ĐỊNH DẠNG TRẢ VỀ (JSON THUẦN, KHÔNG markdown, KHÔNG chú thích):
- TRẢ VỀ MỘT MẢNG JSON có EXACTLY {k_query} PHẦN TỬ.
- Mỗi PHẦN TỬ (RECORD) có cấu trúc CHÍNH XÁC:
[
  {{
    "query": "chuỗi truy vấn ngắn gọn tự nhiên (có thể viết tắt/không dấu)",
    "pos": "{pos}",
    "hard_neg": [
      {{"text": "mô tả gần đúng nhưng cố ý sai", "type": ["location"],                "weight": 0}},
      {{"text": "mô tả gần đúng nhưng cố ý sai", "type": ["price","amenity"],         "weight": 0}},
      {{"text": "mô tả gần đúng nhưng cố ý sai", "type": ["area","other"],            "weight": 0}}
    ]
  }}
]

RÀNG BUỘC JSON
- CHỈ TRẢ VỀ JSON HỢP LỆ. KHÔNG kèm văn bản khác.
- MẢNG kết quả phải có đúng {k_query} record.
- Trong mỗi record:
  • "pos" PHẢI đúng y chuỗi: "{pos}"
  • "hard_neg" PHẢI có đúng {k_neg} phần tử; mỗi phần tử phải có đủ ba trường:
    "text" (string), "type" (array string, chỉ thuộc tập cho phép), "weight" (số 0).
"""


# ================== GỌI GEMINI ==================
def make_model():
    api_key = API_KEY
    if not api_key:
        raise RuntimeError("Chưa có GEMINI_API_KEY trong .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def gen_batch(model, count: int) -> List[Dict[str, Any]]:
    prompt = PROMPT.replace("{COUNT}", str(count))
    resp = model.generate_content(prompt)
    data = loads_safely(resp.text)

    # Đảm bảo là list
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Model không trả về mảng JSON như yêu cầu.")

    # Hậu xử lý từng item
    fixed = [postprocess_item(x) for x in data]
    return fixed


# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Generate rental search triplets (query, pos, hard_neg[]) via Gemini")
    parser.add_argument("--n", type=int, default=400, help="Tổng số mẫu cần sinh")
    parser.add_argument("--batch", type=int, default=100, help="Kích thước lô cho mỗi lần gọi API")
    parser.add_argument("--out", type=str, default="out_triplets_1.json", help="Đường dẫn file xuất (json/jsonl)")
    parser.add_argument("--jsonl", action="store_true", help="Nếu đặt cờ này sẽ ghi JSONL thay vì JSON mảng")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = make_model()

    results: List[Dict[str, Any]] = []
    total = args.n
    batch = max(1, args.batch)

    steps = (total + batch - 1) // batch
    for _ in trange(steps, desc="Generating"):
        need = min(batch, total - len(results))
        if need <= 0:
            break

        # gọi model, retry nhẹ nếu lỗi parse
        for attempt in range(3):
            try:
                chunk = gen_batch(model, need)
                # lọc item rỗng
                chunk = [x for x in chunk if x.get("query") and x.get("pos") and x.get("hard_neg")]
                results.extend(chunk)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(0.8)

        time.sleep(0.25)  # lịch sự với API

    # Cắt đúng số lượng
    results = results[:total]

    # Ghi file
    if args.jsonl:
        with out_path.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Đã ghi {len(results)} mẫu vào: {out_path.resolve()}")


if __name__ == "__main__":
    main()
