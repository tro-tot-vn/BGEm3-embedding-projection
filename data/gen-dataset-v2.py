import google.generativeai as genai
import json
import time
import os

# Cấu hình API
genai.configure(api_key="AIzaSyCZEygHbWbXtLbO_N-i9-i8f1693AF_XfA")
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

# BỘ TRỌNG SỐ TỐI ƯU
WEIGHTS = {
    "location": 5.0,
    "price": 3.0,
    "area": 1.5,
    "amenity": 1.0,
    "furniture": 0.5,
    "floor": 0.2,
    "other": 0.2
}


def get_teacher_score(query, doc):
    prompt = f"""
    Bạn là chuyên gia thẩm định Bất động sản. Nhiệm vụ: So sánh "Nhu cầu" (Query) và "Căn nhà" (Document) để chấm điểm độ phù hợp (Relevance Score) từ -10 đến 10.

    BẢNG TRỌNG SỐ PHẠT (PENALTY WEIGHTS):
    - Sai Vị trí (Location): {WEIGHTS['location']} (Lỗi nghiêm trọng nhất)
    - Sai Giá (Price): {WEIGHTS['price']}
    - Sai Diện tích (Area): {WEIGHTS['area']}
    - Thiếu Tiện ích (Amenity): {WEIGHTS['amenity']}
    - Thiếu Nội thất (Furniture): {WEIGHTS['furniture']}
    - Sai Tầng/Khác (Floor/Other): {WEIGHTS['floor']}

    QUY TRÌNH CHẤM ĐIỂM:
    1. Điểm khởi đầu: 10 điểm (Giả định hoàn hảo).
    2. Nguyên tắc trừ điểm: Điểm = 10 - (Trọng số * Mức độ nghiêm trọng).
       - Mức độ nghiêm trọng từ 0.0 (không sai) đến 2.0 (sai hoàn toàn/sai rất nặng).

    CÁC QUY TẮC CỨNG (HARD RULES) - BẮT BUỘC TUÂN THỦ:
    - Nếu SAI QUẬN/HUYỆN hoặc SAI LOẠI HÌNH (Mua vs Thuê): Bắt buộc cho điểm ÂM (từ -5 đến -10). Không quan tâm các yếu tố khác.
    - Nếu Giá chênh lệch > 30%: Trừ ít nhất {WEIGHTS['price']} điểm.

    Query: "{query}"
    Document: "{doc}"

    Output JSON format: {{ "score": float, "reason": "Lý do ngắn gọn" }}
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)['score']
    except:
        return 0.0


# --- BƯỚC 1: ĐỌC PROMPT VÀ NHẬP SỐ LƯỢNG ---

# Đọc prompt từ file bên ngoài
prompt_file_path = "GEMINI_PROMPT_DATASET_GENERATION.md"
if not os.path.exists(prompt_file_path):
    print(f"LỖI: Không tìm thấy file '{prompt_file_path}'. Vui lòng tạo file này trước.")
    exit()

with open(prompt_file_path, "r", encoding="utf-8") as f:
    generation_prompt = f.read()

# Nhập số lượng mẫu
try:
    num_samples = int(input("Nhập số lượng mẫu dữ liệu muốn tạo (VD: 5): "))
except ValueError:
    num_samples = 1

# --- BƯỚC 2: SINH DỮ LIỆU THÔ (RAW DATA) ---
raw_data = []
print(f"\n--- Đang sinh {num_samples} bộ dữ liệu thô từ file prompt ---")

for i in range(num_samples):
    try:
        # Gọi Gemini với prompt từ file
        response = model.generate_content(generation_prompt)
        temp_data = json.loads(response.text)

        # Kiểm tra xem AI trả về List hay Object
        if isinstance(temp_data, list):
            # Nếu là list, lấy phần tử đầu tiên
            if len(temp_data) > 0:
                data_item = temp_data[0]
            else:
                raise Exception("AI trả về list rỗng")
        else:
            # Nếu là object thì dùng luôn
            data_item = temp_data
        # ---------------------

        raw_data.append(data_item)
        print(f"-> [Gen] Đã sinh xong mẫu {i + 1}/{num_samples}: {data_item.get('query', 'No Query')}")
        time.sleep(1)  # Nghỉ xíu tránh rate limit
    except Exception as e:
        print(f"-> [Lỗi] Khi sinh mẫu {i + 1}: {e}")

# --- BƯỚC 3: CHẤM ĐIỂM (TEACHER SCORING) - ĐÃ SỬA FORMAT ---
dataset_ready = []
print(f"\n--- Bắt đầu chấm điểm Teacher Score ---")


def smart_get(data_dict, keys_to_find):
    """Hàm tìm key bất chấp hoa thường"""
    data_lower = {k.lower(): v for k, v in data_dict.items()}
    for key in keys_to_find:
        if key.lower() in data_lower:
            return data_lower[key.lower()]
    return None


for idx, item in enumerate(raw_data):
    # 1. Lấy Query
    query_text = smart_get(item, ['query', 'Query', 'nhu_cau', 'question'])

    # 2. Lấy Pos
    pos_doc = smart_get(item, ['pos', 'Pos', 'positive', 'answer', 'can_nha'])

    # 3. Lấy Neg List
    neg_list = smart_get(item, [
        'neg_candidates', 'negatives', 'neg', 'negative_examples',
        'hard_neg', 'hard_negatives'
    ])

    # Kiểm tra dữ liệu
    if not query_text or not pos_doc or not neg_list:
        print(f"-> [SKIP] Mẫu {idx + 1} bị bỏ qua do thiếu dữ liệu.")
        continue

    entry = {
        "query": query_text,
        "pos": [pos_doc],
        "neg": [],  # Sẽ chỉ chứa list các chuỗi string
        "teacher_scores": []  # Sẽ chứa điểm số [pos, neg1, neg2...]
    }

    try:
        # --- A. CHẤM ĐIỂM POSITIVE ---
        score_pos = get_teacher_score(query_text, pos_doc)
        entry['teacher_scores'].append(score_pos)

        # --- B. XỬ LÝ VÀ CHẤM ĐIỂM NEGATIVES ---
        if isinstance(neg_list, list):
            for neg_item in neg_list:
                # === ĐOẠN SỬA QUAN TRỌNG NHẤT Ở ĐÂY ===
                # Kiểm tra: Nếu neg_item là Object (dict) -> chỉ lấy field "text"
                real_neg_text = ""
                if isinstance(neg_item, dict):
                    real_neg_text = neg_item.get('text', neg_item.get('content', str(neg_item)))
                else:
                    # Nếu nó đã là string thì dùng luôn
                    real_neg_text = str(neg_item)
                # =======================================

                # Lưu text sạch vào danh sách
                entry['neg'].append(real_neg_text)

                # Chấm điểm
                score_neg = get_teacher_score(query_text, real_neg_text)
                entry['teacher_scores'].append(score_neg)

                time.sleep(0.5)

            dataset_ready.append(entry)
            print(f"-> [Score] Đã chấm xong mẫu {idx + 1}")
        else:
            print(f"   [Cảnh báo] Mẫu {idx + 1}: Negatives không phải là danh sách.")

    except Exception as e:
        print(f"-> [Lỗi] Chấm điểm thất bại mẫu {idx + 1}: {e}")

# --- BƯỚC 4: LƯU FILE ---
with open("train_data_distillation_2.jsonl", "w", encoding='utf-8') as f:
    for line in dataset_ready:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print(f"\nHoàn tất! Đã lưu {len(dataset_ready)} mẫu vào file train_data_distillation.jsonl")
