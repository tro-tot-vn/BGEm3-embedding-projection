import glob
import re

# --- CẤU HÌNH ---
# Mẫu tên file cần tìm (dấu * đại diện cho số)
INPUT_PATTERN = "train_data_distillation_*.jsonl"

# Tên file kết quả đầu ra
OUTPUT_FILE = "train_data_distillation.jsonl"


def natural_sort_key(s):
    """Hàm hỗ trợ sắp xếp số tự nhiên (để file _2 đứng trước _10)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def merge_jsonl_files():
    # 1. Tìm tất cả các file khớp mẫu
    files = glob.glob(INPUT_PATTERN)

    # Loại bỏ file output nếu nó đã tồn tại trong danh sách (tránh gộp chính nó)
    if OUTPUT_FILE in files:
        files.remove(OUTPUT_FILE)

    if not files:
        print(f"❌ Không tìm thấy file nào có dạng '{INPUT_PATTERN}' trong thư mục này.")
        return

    # 2. Sắp xếp file theo thứ tự số (1, 2, 3... 10)
    files.sort(key=natural_sort_key)

    print(f"✅ Tìm thấy {len(files)} file thành phần. Đang tiến hành gộp...")
    print(f"   Danh sách: {files}")

    total_lines = 0

    # 3. Mở file đích để ghi
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for fname in files:
            print(f"-> Đang đọc và gộp: {fname}")
            try:
                with open(fname, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # Chỉ ghi những dòng có dữ liệu (không phải dòng trống)
                        if line.strip():
                            outfile.write(line)
                            total_lines += 1
            except Exception as e:
                print(f"   ⚠️ Lỗi khi đọc file {fname}: {e}")

    print("-" * 30)
    print(f"🎉 HOÀN TẤT!")
    print(f"📁 File kết quả: {OUTPUT_FILE}")
    print(f"📊 Tổng cộng: {total_lines} mẫu dữ liệu.")


if __name__ == "__main__":
    merge_jsonl_files()