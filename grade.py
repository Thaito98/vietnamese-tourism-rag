"""
grade.py - Chấm cột grade trong file kết quả đánh giá bằng Gemini/Gemma API
(qua Google AI Studio, gọi thẳng, không qua trung gian).

Thiết kế chấm điểm:
  - Chấm từng đoạn một, không gom batch. Gom batch khiến điểm của một đoạn
    bị ảnh hưởng bởi các đoạn đứng cạnh trong cùng prompt - đã đo được 13,6%
    các đoạn trùng lặp bị chấm ra điểm khác nhau khi dùng batch 20 câu/lần.
  - Khử trùng lặp theo (câu hỏi, đoạn văn) trước khi gọi API. Một đoạn xuất
    hiện ở nhiều phương pháp chỉ được chấm một lần, nên không thể xảy ra
    trường hợp cùng đoạn mà hai điểm khác nhau.
  - Parse thất bại thì để trống, không điền điểm bừa.
  - thinking_level="minimal": tắt hẳn phần suy nghĩ của model. Trước đây
    dùng max_output_tokens=4000 để "chừa chỗ" cho thinking vẫn không đủ,
    7/10 câu bị cắt giữa chừng (finish_reason=MAX_TOKENS, resp.text=None).
    Tắt thinking giải quyết tận gốc: nhanh hơn ~15-20 lần mỗi request,
    không tốn token suy nghĩ, không còn bị cắt.

Kỹ thuật gọi API:
  - Mỗi luồng 1 key : số luồng = số key, luồng thứ i dùng cố định key thứ i.
                      Round-robin toàn cục (kiểu bản grade_with_gemini.py cũ)
                      khiến các luồng tranh nhau cùng dãy key, dồn cục bộ
                      lên một key rồi dội 429. Gắn cố định thì mỗi key chỉ
                      có đúng 1 request đang bay tại một thời điểm, không
                      thể vượt RPM của chính nó, và một key dính 429 cũng
                      không kéo theo key khác.
  - Song song I/O   : gọi API là tác vụ chờ mạng - đo được 1,3s/request thì
                      chỉ ~0,1s là xử lý. Chạy tuần tự nghĩa là 92% thời
                      gian ngồi không và chỉ dùng được 1/3 quota sẵn có.
  - Không rate limiter: đặt RPM cứng ở phía client chỉ là đoán mò quota -
                      đoán thấp thì tự bóp tốc độ, đoán cao thì dội 429.
                      Để server trả 429 rồi backoff là đúng thực tế hơn.
  - Backoff         : 429/503/5xx => ngủ 2^n giây + jitter rồi thử lại, tối đa
                      7 lần, trần 60s mỗi lần. Cặp nào hết lượt thử thì để
                      trống điểm - chạy lại script sau sẽ chấm bù.

Cách dùng:
    python grade.py --csv eval_results_v3.csv
    python grade.py --csv eval_results_v3.csv eval_results_rerank.csv
"""

import io, sys, os, csv, re, time, random, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from google import genai
from google.genai import types

ROOT  = os.path.dirname(os.path.abspath(__file__))
MODEL = "gemma-4-26b-a4b-it"   # đổi tại đây nếu muốn dùng model Gemini/Gemma khác
sys.path.insert(0, ROOT)


def _load_api_keys() -> list:
    """Đọc GEMINI_API_KEYS (danh sách) từ keys.py.

    Hỗ trợ nhiều key để cộng dồn quota - mỗi key AI Studio có RPM/RPD riêng.
    Kiểm tra ở đây thay vì lúc import module, để `--help` vẫn chạy được dù
    chưa có key.
    """
    try:
        import keys
    except ImportError:
        print("Không tìm thấy keys.py.")
        print("Tạo file keys.py ở thư mục project rồi điền GEMINI_API_KEYS.")
        sys.exit(1)

    api_keys = getattr(keys, "GEMINI_API_KEYS", None)
    if not api_keys:
        print("keys.py chưa có GEMINI_API_KEYS.")
        print("Thêm vào keys.py:")
        print('  GEMINI_API_KEYS = ["AIzaSy..."]')
        print("Lấy key tại: https://aistudio.google.com/apikey")
        sys.exit(1)

    valid = [k for k in api_keys if k and not k.startswith("YOUR_")]
    if not valid:
        print("GEMINI_API_KEYS trong keys.py toàn placeholder chưa điền.")
        sys.exit(1)

    return valid


# ── Client theo từng key ─────────────────────────────────────────────────────
# Tạo sẵn một client cho mỗi key, tái sử dụng thay vì dựng client mới mỗi
# request. Mỗi luồng nhận cố định một client nên hai luồng không bao giờ
# dùng chung một key.
def _make_client(api_key: str):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )


# ── Backoff ───────────────────────────────────────────────────────────────────
# 7 lần thử với trần 60s: quota RPM của Google reset theo từng phút, nên khi
# dính 429 phải chờ sang phút mới. Backoff thuần 2^n dừng ở 30s tổng cộng là
# chưa đủ qua cửa sổ đó - đo thực tế 11/400 cặp bị bỏ vì hết lượt thử.
MAX_RETRIES = 7
MAX_BACKOFF = 60.0

def _is_retryable(err: str) -> str:
    """Phân loại lỗi tạm thời đáng thử lại. Trả về nhãn, hoặc chuỗi rỗng."""
    e = err.lower()
    if "429" in err or "resource_exhausted" in e or "rate limit" in e:
        return "429 rate-limit"
    if "503" in err or "unavailable" in e or "overloaded" in e:
        return "503 quá tải"
    if "500" in err or "502" in err or "504" in err or "internal" in e:
        return "5xx lỗi server"
    if "timeout" in e or "deadline" in e or "connection" in e:
        return "timeout/mạng"
    return ""


class ParseError(ValueError):
    """Không đọc được điểm từ phản hồi của model."""


def _parse_grade(response: str) -> str:
    """Đọc một điểm 0/1/2 từ phản hồi. Ném ParseError thay vì đoán bừa.

    Điền điểm bừa khi parse fail sẽ tạo ra điểm rác đi thẳng vào NDCG mà
    không ai biết - đây là lỗi đã có trong thiết kế batch cũ (nhặt chữ số
    bừa bãi rồi điền cứng 0 cho phần thiếu).
    """
    text = response.strip()

    m = re.fullmatch(r"[^\d]*([012])[^\d]*", text)
    if m:
        return m.group(1)

    digits = re.findall(r"[012]", text)
    if len(text) <= 40 and len(set(digits)) == 1:
        return digits[0]

    raise ParseError(f"Không đọc được điểm từ: {text[:80]!r}")


def _extract_text(resp) -> str:
    """
    gemma-4-26b-a4b-it là thinking model: resp.text = None khi chỉ có thought
    parts. Lấy text từ non-thought parts trước, fallback sang thought parts.
    """
    if resp.text is not None:
        return resp.text.strip()
    if resp.candidates:
        non_thought = []
        thought_txt = []
        for part in resp.candidates[0].content.parts:
            t = getattr(part, "text", None) or ""
            if getattr(part, "thought", False):
                thought_txt.append(t)
            else:
                non_thought.append(t)
        if non_thought:
            return "".join(non_thought).strip()
        if thought_txt:
            return "".join(thought_txt).strip()
    return ""


def _call_and_parse_with_backoff(prompt: str, client, key_label: str) -> str:
    """
    Gọi Gemini/Gemma và parse điểm, với exponential backoff kèm jitter.

    Parse thất bại (model trả sai định dạng, ví dụ lặp lại câu hỏi thay vì
    chỉ trả 1 chữ số) CŨNG được coi là lỗi tạm thời đáng thử lại, không phải
    lỗi vĩnh viễn - đã quan sát thực tế: cùng một cặp (câu hỏi, đoạn văn) có
    lúc model trả sai định dạng, gọi lại ngay sau đó model trả đúng. Nếu
    không retry ở đây, những cặp này bị bỏ trống oan dù model thừa khả năng
    trả lời đúng ở lần sau.

    thinking_level="minimal": gemma-4-26b-a4b-it mặc định BẮT BUỘC suy nghĩ
    (thinking_budget=0 bị API từ chối - 400 INVALID_ARGUMENT). Nhưng
    thinking_level="minimal" thì được chấp nhận và tắt hẳn phần suy nghĩ
    (đã đo: thinking_tokens=None, finish_reason=STOP, không còn bị cắt giữa
    chừng). Trước đây dùng max_output_tokens=4000 để "chừa chỗ" cho thinking
    vẫn không đủ - đo thực tế ở mức 4000 có 7/10 câu bị cắt
    (finish_reason=MAX_TOKENS, resp.text=None) vì thinking tự nó có thể tốn
    hàng nghìn token trước khi kịp trả lời. Tắt thinking giải quyết tận gốc
    thay vì đoán ngưỡng token: nhanh hơn (~1-2s so với ~20-30s mỗi request),
    không tốn token suy nghĩ, và không còn bị cắt.

    max_output_tokens=200: câu trả lời chỉ là một chữ số 0/1/2, không cần
    nhiều - vẫn để dư so với 1 token cho các trường hợp model trả thêm vài
    từ giải thích dù đã dặn không giải thích.

    timeout=30_000 (30s): tránh treo vô hạn nếu một request bị nghẽn ở tầng
    kết nối - khi đó backoff không có cơ hội chạy vì request chưa bao giờ
    trả về lỗi lẫn kết quả.

    Retry vẫn dùng đúng key đó chứ không nhảy sang key khác: 429 nghĩa là
    key này đang chạm quota của nó, đẩy request sang key khác chỉ làm key kia
    chạm quota theo. Ngủ rồi thử lại trên chính key đó là cách để mỗi key tự
    giữ nhịp trong giới hạn của mình.
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        key = key_label
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=200,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            )
            text = _extract_text(resp)
            if not text:
                last_err = RuntimeError("Response rỗng (có thể bị cắt giữa phần suy nghĩ)")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1 + random.random())
                    continue
                raise last_err
            return _parse_grade(text)   # ParseError bị bắt ở except bên dưới, sẽ retry
        except ParseError as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                print(f"\n  [sai định dạng] key ...{key[-6:]} - thử lại "
                      f"(lần {attempt + 1}/{MAX_RETRIES})", end="", flush=True)
                time.sleep(0.5 + random.random())
                continue
            raise
        except Exception as e:
            last_err = e
            kind = _is_retryable(str(e))
            if kind and attempt < MAX_RETRIES - 1:
                wait = min(2 ** (attempt + 1), MAX_BACKOFF) + random.random()
                print(f"\n  [{kind}] key ...{key[-6:]} - ngủ {wait:.1f}s "
                      f"(lần {attempt + 1}/{MAX_RETRIES})", end="", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Vẫn lỗi sau {MAX_RETRIES} lần thử: {last_err}")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", nargs="+", default=["eval_results_v3.csv"],
                    help="Một hoặc nhiều file CSV cần chấm. Nhiều file sẽ được "
                         "gom chung để khử trùng lặp trước khi gọi API.")
parser.add_argument("--workers", type=int, default=0,
                    help="Số luồng song song. Mặc định = số key, mỗi luồng giữ "
                         "cố định một key. Đặt thấp hơn nếu dính 429 quá nhiều.")
args = parser.parse_args()

_api_keys = _load_api_keys()
print(f"Số API key: {len(_api_keys)}")

CSV_PATHS  = [os.path.join(ROOT, c) for c in args.csv]
SAVE_EVERY = 200  # lưu file mỗi N đoạn đã chấm

# Chấm từng đoạn một thay vì gom batch - xem docstring đầu file để biết lý do.
PROMPT = """\
Đánh giá đoạn văn dưới đây có trả lời được câu hỏi không.

Thang điểm:
2 = trả lời trực tiếp và đúng câu hỏi
1 = có thông tin liên quan nhưng không trả lời trực tiếp
0 = không liên quan

Câu hỏi: {question}

Đoạn văn: {passage}

Chỉ trả lời đúng một chữ số 0, 1 hoặc 2. Không giải thích."""


def _save_all():
    """Ghi lại toàn bộ các file CSV đang xử lý."""
    for path, (rows, fieldnames) in files.items():
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

# ── Đọc CSV ───────────────────────────────────────────────────────────────────
files: dict = {}   # path -> (rows, fieldnames)

for path in CSV_PATHS:
    if not os.path.exists(path):
        print(f"Không tìm thấy: {path}")
        sys.exit(1)
    rows_f: list = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fn     = reader.fieldnames
        for row in reader:
            rows_f.append(dict(row))
    files[path] = (rows_f, fn)
    print(f"Đọc {len(rows_f):,} dòng từ {os.path.basename(path)}")

# ── Khử trùng lặp ─────────────────────────────────────────────────────────────
# Gộp theo (câu hỏi, đoạn văn) - KHÔNG phải chỉ đoạn văn. Một đoạn văn có thể
# khớp đúng với câu hỏi này nhưng không liên quan tới câu hỏi khác, nên chỉ
# gộp khi cả câu hỏi lẫn đoạn văn đều giống hệt nhau.
work: dict = {}   # (question, passage) -> [(path, row_index), ...]

for path, (rows_f, _) in files.items():
    for i, row in enumerate(rows_f):
        txt = str(row.get("result_text", ""))
        if txt.startswith("ERROR"):
            continue
        if str(row.get("grade", "")).strip() in ("0", "1", "2"):
            continue
        key = (row.get("test_question", ""), txt)
        work.setdefault(key, []).append((path, i))

total_rows = sum(len(r) for r, _ in files.values())
n_pairs    = len(work)
n_dup_rows = sum(len(v) for v in work.values())

print(f"\nTổng dòng     : {total_rows:,}")
print(f"Cần chấm      : {n_dup_rows:,} dòng")
print(f"Cặp duy nhất  : {n_pairs:,}  "
      f"(giảm {100 * (1 - n_pairs / n_dup_rows):.0f}% số request)"
      if n_dup_rows else "")
# Không quá số key: hai luồng dùng chung một key sẽ tự đẩy key đó vượt RPM.
WORKERS = min(args.workers, len(_api_keys)) if args.workers > 0 else len(_api_keys)
print(f"Model={MODEL}  {WORKERS} luồng, mỗi luồng 1 key riêng")

if n_pairs == 0:
    print("\nTất cả đã chấm. Chạy: python eval_score_v2.py")
    sys.exit(0)

# ~1,3s/request đo được sau khi tắt thinking, chia cho số luồng chạy song song.
est_min = n_pairs * 1.3 / 60 / WORKERS
print(f"Ước tính      : ~{est_min:.0f} phút ({est_min/60:.1f} giờ) nếu không dính 429\n")

# ── Chạy song song, mỗi luồng một key ────────────────────────────────────────
# Client tạo sẵn theo key và gắn vào luồng qua thread-local: luồng nào lấy
# được client nào thì dùng nó suốt, nên mỗi key chỉ có đúng 1 request đang
# bay tại một thời điểm - không thể tự vượt RPM của chính nó.
_clients   = [(_make_client(k), k[-6:]) for k in _api_keys]
_slots     = list(range(WORKERS))
_slot_lock = threading.Lock()
_local     = threading.local()

def _my_client():
    """Lấy client gắn với luồng hiện tại, cấp phát lần đầu gọi."""
    if not hasattr(_local, "client"):
        with _slot_lock:
            idx = _slots.pop()
        _local.client, _local.label = _clients[idx]
    return _local.client, _local.label

def _grade_pair(pair_key: tuple) -> tuple:
    """Chấm một cặp. Trả về (pair_key, grade | None, error | None)."""
    question, passage = pair_key
    prompt = PROMPT.format(question=question, passage=passage[:1200])
    client, label = _my_client()
    try:
        return (pair_key, _call_and_parse_with_backoff(prompt, client, label), None)
    except Exception as e:
        return (pair_key, None, str(e))

_save_lock  = threading.Lock()
done_count  = 0
ok_count    = 0
fail_count  = 0
parse_fails = []
t_start     = time.time()

print(f"Bắt đầu chấm {n_pairs:,} cặp...\n")

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = [executor.submit(_grade_pair, k) for k in work]

    for future in as_completed(futures):
        pair_key, grade, err = future.result()

        with _save_lock:
            if err:
                fail_count += 1
                if "Không đọc được điểm" in err:
                    parse_fails.append(err)
                if fail_count <= 10:
                    print(f"\n  LỖI: {err[:100]}", flush=True)
            else:
                # Gán cùng một điểm cho mọi dòng có cặp này
                for path, i in work[pair_key]:
                    files[path][0][i]["grade"] = grade
                ok_count += 1

            done_count += 1

            if done_count % 50 == 0 or done_count == n_pairs:
                pct     = done_count / n_pairs * 100
                elapsed = time.time() - t_start
                rate    = done_count / elapsed * 60      # request/phút thực tế
                eta_min = (n_pairs - done_count) / max(rate, 1e-9)
                print(f"\r[{done_count:,}/{n_pairs:,}] {pct:5.1f}%  "
                      f"ok={ok_count:,} lỗi={fail_count:,}  "
                      f"{rate:.0f} req/phút  còn ~{eta_min/60:.1f}h", end="", flush=True)

            if done_count % SAVE_EVERY == 0:
                _save_all()

# ── Lưu lần cuối ─────────────────────────────────────────────────────────────
_save_all()

remaining = 0
graded    = 0
for path, (rows_f, _) in files.items():
    for r in rows_f:
        g = str(r.get("grade", "")).strip()
        if g in ("0", "1", "2"):
            graded += 1
        elif not str(r.get("result_text", "")).startswith("ERROR"):
            remaining += 1

print(f"\n\nĐã chấm: {graded:,} dòng  |  Còn thiếu: {remaining:,}")
if parse_fails:
    print(f"Parse thất bại: {len(parse_fails)} - những dòng này để trống, "
          f"không điền điểm bừa")
for path in files:
    print(f"  Đã lưu -> {os.path.basename(path)}")

if remaining == 0:
    print("\nChạy tiếp: python eval_score_v2.py --csv " + " ".join(args.csv))
else:
    print(f"\nCòn {remaining:,} dòng - chạy lại script để chấm tiếp.")
