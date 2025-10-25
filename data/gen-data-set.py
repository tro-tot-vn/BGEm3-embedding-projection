# gen_data_batched.py
# Sinh dataset từ .md theo kiểu chia lô: gọi nhiều lượt (per-call) cho đủ target.
# - API key: --api-key hoặc nhập tương tác (ẩn ký tự)
# - Ghi đè số lượng "Generate N examples ..." trong .md theo per-call
# - Gộp tất cả kết quả thành 1 JSON array

import re, json, time, argparse, random, getpass
from pathlib import Path
from typing import Any, List

import google.generativeai as genai
from google.generativeai.types import RequestOptions, GenerationConfig

# ===== Config =====
MODEL_NAME_DEFAULT = "gemini-2.5-flash"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 65536    # đệm lớn để hạn chế bị cắt
REQUEST_TIMEOUT = 240.0

MAX_RETRY = 1                # 1 retry tối đa (tổng 2 lần)
BASE_BACKOFF = 1.5
BACKOFF_CAP = 6.0
COOLDOWN_BETWEEN_CALLS = 0.4 # nghỉ giữa các lượt

# ===== JSON helpers (robust parsing) =====
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s

def _sanitize_json_text(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r'^\s*//.*$', '', s, flags=re.MULTILINE)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    s = s.replace("\ufeff", "")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = s.replace(",]", "]").replace(",}", "}")
    return s

def _extract_outer_json_block(s: str) -> str:
    lb, rb = s.find("["), s.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        return s[lb:rb+1]
    lb, rb = s.find("{"), s.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        return s[lb:rb+1]
    return s

def _collect_objects_from_array(s: str) -> List[str]:
    out, cur, stk, in_obj = [], [], 0, False
    for ch in s:
        if ch == "{":
            stk += 1
            in_obj = True
        if in_obj:
            cur.append(ch)
        if ch == "}":
            stk -= 1
            if stk == 0 and in_obj:
                out.append("".join(cur))
                cur = []
                in_obj = False
    return out

def _safe_json_from_text(txt: str) -> Any:
    s = _strip_code_fences(txt or "")
    s = _sanitize_json_text(s)
    block = _extract_outer_json_block(s)
    block = _sanitize_json_text(block)
    try:
        return json.loads(block)
    except Exception:
        pass
    if block.strip().startswith("["):
        items = []
        for r in _collect_objects_from_array(block):
            try:
                items.append(json.loads(_sanitize_json_text(r)))
            except Exception:
                continue
        if items:
            return items
    try:
        return json.loads(_sanitize_json_text(_extract_outer_json_block(s)))
    except Exception as e:
        raise ValueError(f"Không parse được JSON sau khi sửa lỗi thường gặp: {e}")

# ===== Model helpers =====
def _resp_text(resp) -> str:
    try:
        if getattr(resp, "text", None):
            return resp.text
    except Exception:
        pass
    parts = []
    for c in getattr(resp, "candidates", []) or []:
        content = getattr(c, "content", None)
        for p in getattr(content, "parts", []) or []:
            t = getattr(p, "text", None)
            if t:
                parts.append(t)
    if parts:
        return "\n".join(parts)
    diag = {
        "finish_reasons": [getattr(c, "finish_reason", None)
                           for c in (getattr(resp, "candidates", []) or [])],
        "prompt_feedback": getattr(resp, "prompt_feedback", None),
        "usage_metadata": getattr(resp, "usage_metadata", None),
    }
    raise RuntimeError(f"Model không trả text về. Chẩn đoán: {diag}")

_RECOVERABLE = ("429", "rate limit", "timeout", "timed out", "deadline",
                "server error", "unavailable", "overloaded", "502", "503", "500")

def _recoverable(e: Exception) -> bool:
    return any(h in str(e).lower() for h in _RECOVERABLE)

def _make_model(model_name: str, api_key: str):
    if not api_key:
        raise RuntimeError("Thiếu API key.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
        )
    )

def _call_model_once(model, prompt: str) -> Any:
    for attempt in range(1, MAX_RETRY + 2):
        try:
            resp = model.generate_content(prompt, request_options=RequestOptions(timeout=REQUEST_TIMEOUT))
            return _safe_json_from_text(_resp_text(resp))
        except Exception as e:
            if attempt > MAX_RETRY or not _recoverable(e):
                raise
            backoff = min(BACKOFF_CAP, BASE_BACKOFF * (2 ** (attempt - 1)))
            time.sleep(backoff + random.uniform(0, 0.5))

# ===== Prompt utils =====
# Regex “bắt rộng” các câu yêu cầu số lượng để ghi đè N theo --per-call
_QUANT_RE = re.compile(
    r"(Generate|Please generate)\s+\d+\s+(?:high-quality\s+training\s+)?examples(?:\s+at\s+a\s+time)?",
    flags=re.IGNORECASE
)

def _apply_per_call(prompt: str, n: int) -> str:
    new_clause = f"Generate {n} examples at a time"
    if _QUANT_RE.search(prompt):
        return _QUANT_RE.sub(new_clause, prompt)
    # nếu .md không có câu theo mẫu, gắn hướng dẫn rõ ở cuối
    return prompt.rstrip() + f"\n\nReturn only a valid JSON array. {new_clause}."

# ===== Find .md =====
def _resolve_md_path(user_input: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(user_input).expanduser(),
        script_dir / user_input,
        script_dir.parent / user_input,
    ]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n - ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Không tìm thấy file .md. Đã thử:\n - {tried}")

# ===== Main =====
def main():
    ap = argparse.ArgumentParser(description="Generate rental dataset in batches to avoid token limit.")
    ap.add_argument("--md", required=True, help="Đường dẫn file .md chứa prompt")
    ap.add_argument("--out", default="dataset.json", help="File JSON đầu ra (mảng)")
    ap.add_argument("--target", type=int, default=200, help="Tổng số mẫu cần sinh (gộp nhiều lượt)")
    ap.add_argument("--per-call", type=int, default=20, help="Số mẫu mỗi lượt gọi (10–30 khuyến nghị)")
    ap.add_argument("--model", default=MODEL_NAME_DEFAULT, help="Model ID (mặc định gemini-2.5-flash)")
    ap.add_argument("--api-key", default=None, help="API key Gemini (nếu không truyền, sẽ hỏi tương tác)")
    args = ap.parse_args()

    # Đọc prompt gốc
    md_path = _resolve_md_path(args.md)
    base_prompt = md_path.read_text(encoding="utf-8")
    print(f"[INFO] Using prompt file: {md_path}")

    api_key = args.api_key or getpass.getpass("Enter Gemini API key: ").strip()
    model = _make_model(args.model, api_key)

    results: List[Any] = []
    rounds = (args.target + args.per_call - 1) // args.per_call

    for i in range(rounds):
        need = min(args.per_call, args.target - len(results))
        if need <= 0:
            break

        prompt = _apply_per_call(base_prompt, need)
        print(f"[INFO] Call {i+1}/{rounds} → request {need} examples")

        data = _call_model_once(model, prompt)

        # ép thành list
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            print("[WARN] Model không trả về mảng; bỏ lượt này.")
            data = []

        # lọc item rỗng/thiếu field chính
        filtered = []
        for x in data:
            try:
                q = str(x.get("query", "")).strip()
                p = str(x.get("pos", "")).strip()
                hn = x.get("hard_neg", [])
                if q and p and isinstance(hn, list) and hn:
                    filtered.append({"query": q, "pos": p, "hard_neg": hn})
            except Exception:
                continue

        results.extend(filtered)
        print(f"[INFO] Collected this call: {len(filtered)} | Total: {len(results)}/{args.target}")
        time.sleep(COOLDOWN_BETWEEN_CALLS)

        if len(filtered) == 0:
            print("[WARN] Lượt này không thu được mẫu nào (có thể do JSON hỏng hoặc bị cắt). Tiếp tục lượt kế.")

    # cắt đúng target
    results = results[:args.target]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {len(results)} items → {out_path.resolve()}")

if __name__ == "__main__":
    main()
