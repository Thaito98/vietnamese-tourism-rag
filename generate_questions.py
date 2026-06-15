"""
generate_questions.py - Sinh 8 câu hỏi đều cho mỗi trong 139 topic.

Phân bổ mỗi topic (25% mỗi loại - công bằng giữa các method):
  K1, K2 - Keyword trực tiếp   (hỏi số liệu, tên, địa điểm cụ thể)  => test BM25
  P1, P2 - Paraphrase           (cùng ý, dùng từ hoàn toàn khác)      => test SBERT
  A1, A2 - Ẩn ý / gián tiếp   (không nhắc tên topic, qua đặc điểm)   => test SBERT/Hybrid
  C1, C2 - Cross 2 topic        (so sánh / kết hợp 2 topic liên quan) => test Hybrid

Tổng: 8 × 139 = 1112 câu  (25% mỗi loại, công bằng tuyệt đối)

Kỹ thuật:
  Key Pooling   : Round-robin các key trong keys.py
  Multi-threading: ThreadPoolExecutor, mặc định bằng số key
  Backoff       : 2^n giây khi gặp 429 / 503 / 5xx

Output: generated_questions.json  =>  python merge_questions.py

Cách dùng:
    python generate_questions.py
    python generate_questions.py --workers 5 --batch-size 2
    python generate_questions.py --regen-type A   # sinh lại chỉ A1/A2
"""

import io, sys, os, re, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace",
    line_buffering=True,
    write_through=True,
)

from google import genai
from google.genai import types

ROOT  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
MODEL = "gemini-3.1-flash-lite"

def _load_api_keys() -> list[str]:
    try:
        import keys
    except ImportError:
        print("Không tìm thấy keys.py.")
        print("Tạo file keys.py ở thư mục project rồi điền GEMINI_API_KEYS.")
        print('  GEMINI_API_KEYS = ["AIzaSy..."]')
        sys.exit(1)

    api_keys = getattr(keys, "GEMINI_API_KEYS", None)
    if not api_keys:
        print("keys.py chưa có GEMINI_API_KEYS.")
        print("Thêm vào keys.py:")
        print('  GEMINI_API_KEYS = ["AIzaSy..."]')
        sys.exit(1)

    valid = [
        k.strip() for k in api_keys
        if isinstance(k, str)
        and k.strip()
        and "YOUR_" not in k
        and "..." not in k
    ]
    if not valid:
        print("GEMINI_API_KEYS trong keys.py toàn placeholder chưa điền.")
        sys.exit(1)
    return valid


API_KEYS = _load_api_keys()

# ── Key Pool ──────────────────────────────────────────────────────────────────
_pool_lock = threading.Lock()
_pool_idx  = 0

def _get_key() -> str:
    global _pool_idx
    with _pool_lock:
        key = API_KEYS[_pool_idx % len(API_KEYS)]
        _pool_idx += 1
    return key

# ── Exponential Backoff ───────────────────────────────────────────────────────
MAX_RETRIES = 6

def _call_api(prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        key = _get_key()
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.8, max_output_tokens=800),
            )
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            retryable = any(x in err for x in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL"))
            if retryable and attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                code = next((x for x in ("429","503","500") if x in err), "5xx")
                print(f"\n  [{code}] ...{key[-6:]} ngủ {wait}s", end="", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Lỗi sau {MAX_RETRIES} lần thử")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size",  type=int, default=2)
parser.add_argument("--workers",     type=int, default=0,
                    help="Số luồng song song. Mặc định = số key trong keys.py.")
parser.add_argument("--out",         default="generated_questions.json")
parser.add_argument("--regen-type",  default=None, help="Sinh lại 1 loại: A | K | P | C")
args = parser.parse_args()

OUT_PATH   = os.path.join(ROOT, args.out)
BATCH_SIZE = args.batch_size
WORKERS    = args.workers if args.workers > 0 else len(API_KEYS)

# ── Load topics ───────────────────────────────────────────────────────────────
from topics_bank import TOPICS

# Nhóm topic theo thể loại để ghép cặp C hợp lý
FOOD_TOPICS = {
    "Bánh mì", "Bánh xèo", "Bún bò Huế", "Bún chả", "Cơm tấm",
    "Gỏi cuốn", "Hải sản tươi sống", "Phở", "Chả cá Lã Vọng",
}
BEACH_TOPICS = {
    "Bãi biển Mỹ Khê: Một trong những bãi biển đẹp nhất",
    "Bãi biển Phú Quốc: Bãi Sao & Bãi Dài",
    "Côn Đảo: Hoang sơ và nguyên sơ",
}
CAVE_TOPICS = {
    "Hang Múa và điểm ngắm toàn cảnh",
    "Hang Sơn Đoòng: hang động lớn nhất thế giới",
}

def _find_related(topic: str) -> str:
    """Tìm 1 topic liên quan để ghép cặp C."""
    idx = TOPICS.index(topic) if topic in TOPICS else -1
    # Ưu tiên topic liền kề (cùng nhóm chủ đề)
    candidates = []
    for group in [FOOD_TOPICS, BEACH_TOPICS, CAVE_TOPICS]:
        if topic in group:
            candidates = [t for t in group if t != topic]
            break
    if not candidates:
        # Lấy topic trước/sau trong danh sách (cùng chủ đề gần)
        if idx > 0:
            candidates.append(TOPICS[idx - 1])
        if idx < len(TOPICS) - 1:
            candidates.append(TOPICS[idx + 1])
    return candidates[0] if candidates else TOPICS[(idx + 5) % len(TOPICS)]

# ── Prompt: sinh đủ 8 loại ───────────────────────────────────────────────────
PROMPT_FULL = """\
Sinh 8 câu hỏi về topic du lịch Việt Nam bên dưới.
Quy tắc:
- Mỗi câu kết thúc bằng dấu ?
- K1, K2: hỏi trực tiếp về số liệu / tên / địa điểm cụ thể trong topic
- P1, P2: cùng ý với K1/K2 nhưng HOÀN TOÀN đổi từ ngữ (paraphrase)
- A1, A2: KHÔNG nhắc tên topic, hỏi gián tiếp qua đặc điểm thực tế (xem ví dụ)
- C1, C2: câu hỏi SO SÁNH thực tế giữa 2 topic (ghi rõ tên cả 2, hỏi về thông tin tra cứu được)
- Không trùng lặp ý giữa các câu
- C KHÔNG được dùng: "bạn sẽ", "bạn ưu tiên", "bạn thích", "nếu phải chọn", "bạn muốn"

Trả lời ĐÚNG định dạng (8 dòng, có nhãn):
K1: <câu>
K2: <câu>
P1: <câu>
P2: <câu>
A1: <câu>
A2: <câu>
C1: <câu>
C2: <câu>

=== TOPIC: {topic} ===
Topic liên quan cho C1/C2: {related}
"""

# ── Prompt: sinh lại chỉ A1/A2 (factual, không cảm xúc) ─────────────────────
PROMPT_REGEN_C = """\
Sinh 2 câu hỏi loại C (so sánh 2 topic) cho topic du lịch bên dưới.

Loại C phải thỏa MỌI điều kiện:
  1. Đề cập RÕ RÀNG tên cả 2 topic trong câu hỏi
  2. Hỏi về thông tin thực tế có thể tra cứu được (giá, thời gian, đặc điểm, điều kiện...)
  3. KHÔNG hỏi ý kiến cá nhân: không dùng "bạn sẽ", "bạn ưu tiên", "nếu phải chọn", "bạn thích"
  4. Câu ngắn gọn dưới 25 từ

Ví dụ ĐÚNG:
  "Bún bò Huế và phở Hà Nội món nào có nước dùng cay hơn?"
  "Tour Hang Sơn Đoòng và leo Fansipan, tour nào yêu cầu thể lực cao hơn?"
  "Bảo tàng Dân tộc học và Bảo tàng Chứng tích Chiến tranh, nơi nào mở cửa sớm hơn?"
  "Du thuyền Hạ Long và tàu ra Côn Đảo, hành trình nào dài hơn theo giờ?"

Ví dụ SAI:
  -"Nếu phải chọn giữa X và Y, bạn sẽ ưu tiên điều gì?"
  -"Bạn thích trải nghiệm X hay Y hơn trong chuyến đi?"

Trả lời ĐÚNG định dạng (2 dòng):
C1: <câu>
C2: <câu>

=== TOPIC: {topic} ===
Topic liên quan để ghép cặp: {related}
"""

PROMPT_REGEN_A = """\
Sinh 2 câu hỏi loại A (ẩn ý gián tiếp) cho topic du lịch bên dưới.

Loại A phải thỏa MỌI điều kiện:
  1. KHÔNG nhắc tên topic, không dùng tên riêng của địa điểm/món ăn
  2. Hỏi qua đặc điểm thực tế: số liệu, vị trí, lịch sử, quy trình, điều kiện
  3. Câu hỏi CÓ THỂ TRẢ LỜI được bằng thông tin cụ thể (không hỏi cảm xúc)
  4. KHÔNG dùng từ: "cảm thấy", "trải nghiệm cảm xúc", "bạn muốn", "khiến bạn"

Ví dụ ĐÚNG:
  Topic "Hang Sơn Đoòng" => "Hang động nào ở Quảng Bình cần đặt vé trước 6 tháng?"
  Topic "Bún bò Huế"     => "Món nước cay đặc trưng miền Trung dùng mắm gì làm nước dùng?"
  Topic "Fansipan"       => "Đỉnh núi cao nhất Đông Dương có thể leo bộ mất bao nhiêu ngày?"
  Topic "Chùa Cầu"       => "Công trình nào ở phố cổ do người Nhật xây để nối hai khu phố?"

Ví dụ SAI:
  -"Bạn cảm thấy thế nào khi đứng trên đỉnh hang động lớn nhất thế giới?"
  -"Những trải nghiệm nào khiến bạn muốn quay lại?"

Trả lời ĐÚNG định dạng (2 dòng):
A1: <câu>
A2: <câu>

=== TOPIC: {topic} ===
"""

# ── Parse plain-text response ─────────────────────────────────────────────────
LABELS = ["K1", "K2", "P1", "P2", "A1", "A2", "C1", "C2"]
TYPE_MAP = {"K1": "K", "K2": "K", "P1": "P", "P2": "P",
            "A1": "A", "A2": "A", "C1": "C", "C2": "C"}

def _parse_response(text: str, topic: str) -> list:
    results = []
    for label in LABELS:
        pattern = rf"(?:^|\n){re.escape(label)}\s*[:：]\s*(.+?)(?=\n[A-Z][0-9][:：]|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            q = m.group(1).strip().replace("\n", " ")
            if not q.endswith("?"):
                q += "?"
            results.append({
                "topic": topic,
                "type":  TYPE_MAP[label],
                "label": label,
                "question": q,
            })
    return results

# ── Load file hiện có ────────────────────────────────────────────────────────
all_results: list = []
if os.path.exists(OUT_PATH):
    with open(OUT_PATH, encoding="utf-8") as f:
        all_results = json.load(f)

REGEN_TYPE = args.regen_type.upper() if args.regen_type else None

if REGEN_TYPE:
    # Xóa tất cả câu thuộc loại cần sinh lại, giữ nguyên các loại khác
    label_prefix = REGEN_TYPE   # "A" => xóa A1,A2 | "C" => xóa C1,C2
    kept    = [r for r in all_results if not r.get("label","").startswith(label_prefix)]
    removed = len(all_results) - len(kept)
    all_results = kept
    print(f"Chế độ --regen-type {REGEN_TYPE}: xóa {removed} câu cũ, giữ {len(all_results)} câu.")
    remaining_topics = list(TOPICS)   # chạy lại toàn bộ 139 topic
    n_per_topic = 2  # A1 + A2
else:
    done_topics    = {r["topic"] for r in all_results}
    remaining_topics = [t for t in TOPICS if t not in done_topics]
    if all_results:
        print(f"Resume: {len(all_results)} câu từ {len(done_topics)} topic đã xong.\n")
    n_per_topic = 8

# ── Batch topics ──────────────────────────────────────────────────────────────
topic_batches = [
    remaining_topics[i : i + BATCH_SIZE]
    for i in range(0, len(remaining_topics), BATCH_SIZE)
]

print(f"Topics cần chạy: {len(remaining_topics)} / {len(TOPICS)}")
print(f"Số batch       : {len(topic_batches)}  |  Workers: {WORKERS}  |  Model: {MODEL}")
print(f"Mục tiêu thêm  : ~{len(remaining_topics) * n_per_topic} câu\n")

# ── Xử lý 1 batch ─────────────────────────────────────────────────────────────
def _process_batch(batch_topics: list) -> tuple:
    new_qs = []
    errors = []
    for topic in batch_topics:
        if REGEN_TYPE == "A":
            prompt   = PROMPT_REGEN_A.format(topic=topic)
            min_ok   = 2
        elif REGEN_TYPE == "C":
            related  = _find_related(topic)
            prompt   = PROMPT_REGEN_C.format(topic=topic, related=related)
            min_ok   = 2
        else:
            related  = _find_related(topic)
            prompt   = PROMPT_FULL.format(topic=topic, related=related)
            min_ok   = 6
        try:
            raw    = _call_api(prompt)
            parsed = _parse_response(raw, topic)
            if len(parsed) < min_ok:
                preview = raw[:150].replace("\n", "↵")
                errors.append(f"{topic}: parse {len(parsed)}/{n_per_topic} - {preview!r}")
            new_qs.extend(parsed)
        except Exception as e:
            errors.append(f"{topic}: {str(e)[:80]}")
    return (new_qs, errors)

# ── Multi-threading ───────────────────────────────────────────────────────────
_write_lock  = threading.Lock()
done_batches = 0

def _save():
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Bắt đầu với {WORKERS} luồng...\n")

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(_process_batch, batch): batch for batch in topic_batches}

    for future in as_completed(futures):
        new_qs, errors = future.result()

        with _write_lock:
            done_batches += 1
            all_results.extend(new_qs)

            for err in errors:
                print(f"\n  [WARN] {err}", flush=True)

            print(f"[{done_batches}/{len(topic_batches)}] +{len(new_qs)}câu  tổng={len(all_results)}", end="  ", flush=True)

            if done_batches % 15 == 0:
                _save()
                print(f"=> lưu", end="  ", flush=True)

# ── Lưu cuối ─────────────────────────────────────────────────────────────────
_save()

by_topic = {}
for r in all_results:
    by_topic.setdefault(r["topic"], []).append(r)

expected = 2 if REGEN_TYPE else 8
print(f"\n\nTổng câu trong file: {len(all_results)}")
print(f"Số topic hoàn thành: {len(by_topic)} / {len(TOPICS)}")
under = [(t, len(qs)) for t, qs in by_topic.items() if len(qs) < expected]
if under:
    print(f"Topic thiếu câu ({expected}): {len(under)} topic - chạy lại để bổ sung")
    for t, n in under[:5]:
        print(f"  {t}: {n}/{expected}")
print(f"\nĐã lưu -> {OUT_PATH}")
print("Kiểm tra rồi chạy: python merge_questions.py")
