"""
PHASE 1: CÀO DỮ LIỆU TIẾNG VIỆT

THỨ TỰ FALLBACK (toàn tiếng Việt):
  1. Wikipedia VI   -> API, ổn định nhất
  2. WikiVoyage VI  -> API, chuyên du lịch
  3. SKIP           -> Ghi vào not_found, dùng data CSV gốc

KHÔNG dùng:
  - Wikipedia EN / WikiVoyage EN  (tiếng Anh)
  - VnExpress   (cần URL bài cụ thể, không search được)
  - VietnamTourism (JavaScript-rendered, không scrape được)
  - Agoda/Booking (quá nhiều rác)

GHI NGUỒN: Mỗi file lưu đều có TOPIC, SOURCE, LANG, WORDS, ENTITIES
ENTITIES: danh sách liên kết nội bộ Wikipedia (wikilinks) rút từ bài,
          giúp biết loại thực thể (địa danh, người, sự kiện...).
"""
import sys, codecs
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

import os, json, time, re
import urllib.request, urllib.parse
from tqdm import tqdm
import urllib.error

# LOAD TOPICS TỪ FILE JSON (extract từ CSV dataset)
_topics_file = "data/raw/topics_to_crawl.json"
if os.path.exists(_topics_file):
    with open(_topics_file, 'r', encoding='utf-8') as _f:
        TOPICS = json.load(_f)["topics"]
    print(f"Loaded {len(TOPICS)} topics từ CSV dataset")
else:
    print(f"Chưa có {_topics_file} -> chạy extract_topics_from_csv.py trước!")
    sys.exit(1)


def _api_get(url: str, timeout: int = 12) -> dict:
    headers = {
        "User-Agent": "VietnamTourismRAGBot/1.0 (contact: thaisocu2025@gmail.com)",
        "Accept": "application/json"
    }
    for attempt in range(3):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
            elif 500 <= e.code < 600:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"[HTTP ERROR] {e.code} | {url}")
                raise
        except Exception as e:
            print(f"[ERROR] {repr(e)} | {url}")
            raise
    return {}


def _build_url(base: str, params: dict) -> str:
    """Build URL với encode UTF-8 đúng cách cho tiếng Việt"""
    parts = []
    for k, v in params.items():
        parts.append(f"{k}={urllib.parse.quote(str(v), safe='')}")
    return f"{base}?{'&'.join(parts)}"


# HÀM RÚT ENTITY (wikilinks) - chỉ giữ địa danh liên quan du lịch
# Rút liên kết nội bộ Wikipedia để biết loại thực thể.
# Ví dụ: bài "Sơn Trà" có link [[Bán đảo Sơn Trà]], [[Đà Nẵng]]
# -> biết "Sơn Trà" ở đây là địa danh, không phải tên người.

# Từ khóa chỉ địa danh / đơn vị hành chính -> GIỮ nếu title CHỨA từ này
# Lưu ý: bỏ "xã", "phường", "thị trấn", "thị xã" - cấp quá nhỏ, gây nhiễu
_GEO_KEYWORDS = [
    "tỉnh", "thành phố", "quận", "huyện",
    "vịnh", "đảo", "núi", "sông", "hồ", "thác", "rừng", "vườn quốc gia",
    "bán đảo", "hang", "bãi biển", "bãi tắm", "cửa khẩu", "mũi", "đèo",
    "cao nguyên", "di sản", "khu du lịch", "khu bảo tồn", "quần đảo",
    "cù lao", "đầm phá", "dãy núi", "làng cổ", "phố cổ",
]

# Pattern chỉ là số (năm, mã số...) -> BỎ
_NUMBER_RE = re.compile(r'^\d+$')

# Tên người: 2-4 từ bắt đầu bằng họ phổ biến -> BỎ
_PERSON_SURNAMES = {
    "Nguyễn", "Trần", "Lê", "Phạm", "Huỳnh", "Hoàng", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Tô", "Trịnh",
    "Bà",                           # "Bà Triệu", "Bà Trưng"
    "Minh", "Quang", "Bảo", "Gia",  # vua Nguyễn: Minh Mạng, Quang Trung, Bảo Đại, Gia Long
    "Nhà",                           # "Nhà Nguyễn", "Nhà Trần" -> triều đại
}

# Blacklist: tổ chức, khái niệm không phải địa danh du lịch
_BLACKLIST = {
    "BBC", "CNN", "UNESCO", "UNICEF", "WHO", "WTO", "IMF",
    "Anthropocene", "Buồm", "Bầu trời", "Biểu tượng",
    "Biển lùi", "Biển tiến",
    # Từ chung địa lý (không có tên riêng đi kèm)
    "Núi", "Bán đảo", "Đảo", "Sông", "Hồ", "Biển", "Vịnh", "Rừng",
    "Hang", "Thác", "Đèo", "Mũi", "Bãi biển",
    # Khái niệm không phải địa danh
    "An ninh", "Quốc phòng", "Lãnh hải", "Phù sa", "Chiều rộng",
    "Phật giáo", "Thiên Chúa giáo", "Hồi giáo",
    "Loài nguy cấp", "Loài đặc hữu", "Sinh thái học",
}

# Từ khóa trong tên -> chắc chắn KHÔNG phải địa danh -> BỎ
_NOT_GEO_PATTERNS = [
    "chiến tranh", "chiến dịch", "trận ", "quân đội", "lực lượng",
    "triều đại", "vương triều",
    "loài ", "giống ",                     # phân loại sinh vật
    "bồ tát", "quan âm",                   # nhân vật tôn giáo (không bỏ "phật" vì "Chùa Phật Tích")
    "học thuyết", "lý thuyết", "khái niệm",
    # Đơn vị hành chính cấp xã/phường/thôn/huyện dạng định danh -> quá chi tiết, gây nhiễu
    # Ví dụ: "An Biên (xã)", "Châu Thành (huyện An Giang)" - disambiguation page hành chính
    "(xã)", "(phường)", "(thị trấn)", "(thị xã)", "(thôn)", "(huyện",
    # Trang định hướng Wikipedia
    "định hướng",
    # Ẩm thực nước ngoài / khái niệm ẩm thực không liên quan du lịch VN
    "cuisine",
]

# Địa danh nổi tiếng Việt Nam -> GIỮ ngay, không cần phân tích thêm
# Đặc biệt cần liệt kê tỉnh/huyện KHÔNG có dấu (không pass được rule non-ASCII bên dưới)
_KNOWN_PLACES = {
    # Thành phố / tỉnh thành lớn
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hội An", "Huế", "Nha Trang",
    "Đà Lạt", "Sa Pa", "Hạ Long", "Phú Quốc", "Cần Thơ", "Hải Phòng",
    "Quảng Ninh", "Lào Cai", "Phan Thiết", "Mũi Né", "Côn Đảo",
    # Tỉnh không có dấu (sẽ bị rule non-ASCII lọc mất nếu không whitelist)
    "An Giang", "Gia Lai", "Kon Tum", "Long An",
    # Tránh lọc nhầm "Bà" = họ người nhưng đây là địa danh
    "Bà Rịa", "Bà Rịa-Vũng Tàu", "Bà Nà", "Bà Nà Hills",
    # Quốc gia lân cận liên quan du lịch
    "Việt Nam", "Campuchia", "Lào", "Thái Lan", "Trung Quốc",
}


def _is_tourism_entity(name: str) -> bool:
    """
    Trả về True nếu entity là địa danh liên quan du lịch Việt Nam.
    Loại bỏ: số/năm, tên người/triều đại, tổ chức, khái niệm, loài sinh vật.
    """
    name = name.strip()

    # Bỏ số thuần
    if _NUMBER_RE.match(name):
        return False

    # Bỏ dạng "Tên, Tỉnh" - Wikipedia dùng để định danh thôn/xã quá chi tiết
    # Ví dụ: "An Châu, An Giang", "Châu Phú, Cần Thơ"
    if ", " in name:
        return False

    # Bỏ blacklist
    if name in _BLACKLIST:
        return False

    # Giữ ngay nếu là địa danh nổi tiếng đã biết
    if name in _KNOWN_PLACES:
        return True

    name_lower = name.lower()

    # Bỏ nếu chứa pattern chắc chắn không phải địa danh
    for pattern in _NOT_GEO_PATTERNS:
        if pattern in name_lower:
            return False

    parts = name.split()

    # Bỏ nếu từ đầu là họ người / từ chỉ triều đại phổ biến
    if len(parts) >= 2 and parts[0] in _PERSON_SURNAMES:
        return False

    # Bỏ từ đơn (không có tên riêng -> quá chung chung)
    if len(parts) == 1:
        return False

    # Giữ nếu chứa từ khóa địa lý CÓ tên riêng đi kèm
    # (len > 1 đảm bảo không phải chỉ mỗi từ chung như "Núi", "Sông")
    for kw in _GEO_KEYWORDS:
        if kw in name_lower:
            return True

    # Bỏ quốc gia ngoài Đông Nam Á / Việt Nam
    _FOREIGN_COUNTRIES = {
        "hoa kỳ", "mỹ", "pháp", "anh", "đức", "nhật", "trung quốc",
        "tây ban nha", "bồ đào nha", "nga", "ý", "hà lan",
    }
    if name_lower in _FOREIGN_COUNTRIES:
        return False

    # Giữ nếu là tên riêng 2-4 từ viết hoa VÀ có ít nhất 1 ký tự Unicode (dấu tiếng Việt)
    # -> loại bỏ "Barack Obama", "Cajun cuisine", "Bibim guksu" (toàn ASCII)
    # -> giữ "Hội An" (ộ), "Mù Cang Chải" (ù, ả), "Bái Tử Long" (á, ử)
    if 2 <= len(parts) <= 4 and name[0].isupper() and any(ord(c) > 127 for c in name):
        return True

    return False


def wiki_fetch_links(title: str, site: str = "vi.wikipedia", limit: int = 200) -> list:
    """
    Lấy danh sách entity (wikilinks) liên quan du lịch từ bài Wikipedia.
    - Gọi API prop=links, namespace=0 (bài viết chính)
    - Lọc qua _is_tourism_entity() để chỉ giữ địa danh / khái niệm du lịch
    - Bỏ: năm, tên người, tổ chức không liên quan
    """
    base = f"https://{site}.org/w/api.php"
    params = {
        "action":      "query",
        "prop":        "links",
        "plnamespace": "0",
        "pllimit":     str(limit),
        "titles":      title,
        "format":      "json",
        "utf8":        "1"
    }
    try:
        url = _build_url(base, params)
        data = _api_get(url)
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid != "-1":
                all_links = [lk["title"] for lk in page.get("links", [])]
                return [lk for lk in all_links if _is_tourism_entity(lk)]
    except Exception:
        pass
    return []


# TÌM KIẾM VÀ LẤY NỘI DUNG

def wiki_direct_titles(topic: str) -> list:
    """
    Tạo danh sách title để thử lookup trực tiếp (không qua search API).
    Thử topic gốc + các biến thể thường gặp.
    """
    titles = [topic]
    # Bỏ phần sau dấu : hoặc (
    if ":" in topic:
        titles.append(topic.split(":")[0].strip())
    if "(" in topic:
        titles.append(topic.split("(")[0].strip())
    # Bỏ prefix địa danh thường gặp
    for prefix in ["Du lịch ", "Khám phá ", "Thăm quan ", "Tham quan "]:
        if topic.startswith(prefix):
            titles.append(topic[len(prefix):].strip())
    # Không thêm "Việt Nam" vì tạo title không tồn tại trên Wikipedia/WikiVoyage
    seen, unique = set(), []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def wiki_search(query: str, site: str = "vi.wikipedia") -> list:
    """
    Tìm kiếm bài trên Wikipedia VI hoặc WikiVoyage VI.
    site: "vi.wikipedia" | "vi.wikivoyage"
    """
    base = f"https://{site}.org/w/api.php"
    params = {
        "action":   "query",
        "list":     "search",
        "srsearch": query,
        "srlimit":  "3",
        "format":   "json",
        "utf8":     "1"
    }
    try:
        url = _build_url(base, params)
        data = _api_get(url)
        return [r["title"] for r in data.get("query", {}).get("search", [])]
    except Exception:
        return []


def wiki_fetch_content(title: str, site: str = "vi.wikipedia") -> str:
    """Lấy nội dung plain text của bài (explaintext=True xóa bỏ wiki markup)."""
    base = f"https://{site}.org/w/api.php"
    params = {
        "action":          "query",
        "prop":            "extracts",
        "explaintext":     "True",
        "exsectionformat": "plain",
        "titles":          title,
        "format":          "json",
        "utf8":            "1"
    }
    try:
        url = _build_url(base, params)
        data = _api_get(url)
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid != "-1":
                return page.get("extract", "")
    except Exception:
        pass
    return ""


# LÀM SẠCH NỘI DUNG

def clean_text(text: str) -> str:
    BOILERPLATE = [
        "Xem thêm", "Tham khảo", "Liên kết ngoài", "Chú thích",
        "Ghi chú", "Nguồn", "Đọc thêm", "Ghi chép", "Chú giải",
        "Hình ảnh", "Thư viện ảnh", "Bản đồ", "Phương tiện liên quan",
        "Gallery", "Chú thích hình", "Tham chiếu",
    ]
    cutoff = len(text)
    for section in BOILERPLATE:
        # Dạng 1: section nằm trên dòng riêng
        for pattern in (f"\n{section}\n", f"\n{section}\r\n"):
            idx = text.find(pattern)
            if idx != -1 and idx < cutoff:
                cutoff = idx
        # Dạng 2: section nằm ở cuối file (không có \n sau)
        for pattern in (f"\n{section}", f"\n\n{section}"):
            if text.endswith(pattern):
                idx = len(text) - len(pattern)
                if idx < cutoff:
                    cutoff = idx
        # Dạng 3: section xuất hiện ngay đầu text
        if text.startswith(section + "\n"):
            cutoff = 0
    text = text[:cutoff]

    # Bỏ marker tham chiếu [1], [2], [a], [cần dẫn nguồn]...
    text = re.sub(r'\[[^\]]{1,30}\]', '', text)
    # Bỏ tiêu đề section dạng ==Tên section== nếu còn sót
    text = re.sub(r'={2,}[^=]*={2,}', '\n', text)
    # Chuẩn hóa khoảng trắng và dòng trống
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# TẠO QUERY TÌM KIẾM

def make_queries(title: str) -> list:
    """
    Tạo list query từ title CSV.
    "Vịnh Hạ Long: Kỳ quan thiên nhiên thế giới" -> ["Vịnh Hạ Long", ...]
    """
    queries = []
    if ":" in title:
        queries.append(title.split(":")[0].strip())
    if "(" in title:
        queries.append(title.split("(")[0].strip())
    queries.append(title)

    # Synonym: topic không có bài Wikipedia trực tiếp -> thử tên khác
    SYNONYMS = {
        "balo giá rẻ":       "du lịch bụi",
        "hướng dẫn du lịch": "du lịch Việt Nam",
        "tiết kiệm":         "du lịch tiết kiệm",
        "ngân sách":         "lập kế hoạch du lịch",
        "visa":              "thị thực Việt Nam",
        "hộ chiếu":          "hộ chiếu Việt Nam",
        "sim card":          "SIM điện thoại",
        "ứng dụng":          "ứng dụng du lịch",
        "tip":               "văn hóa Việt Nam",
        "lưu trú":           "khách sạn Việt Nam",
        "bảo hiểm":          "bảo hiểm du lịch",
        "sức khỏe":          "y tế Việt Nam",
        "bản làng sa pa":          "Sa Pa",      # "Trekking các bản làng Sa Pa" -> "Sa Pa"
    }
    title_lower = title.lower()
    for kw, syn in SYNONYMS.items():
        if kw in title_lower and syn not in queries:
            queries.append(syn)

    seen, unique = set(), []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


# CÀO TỪNG TOPIC

# NGUỒN THEO THỨ TỰ ƯU TIÊN
SOURCES = [
    # (site,           source_name,      min_words)
    ("vi.wikipedia",  "Wikipedia VI",    300),
    ("vi.wikivoyage", "WikiVoyage VI",   150),
]


def crawl_one_topic(topic: str, output_dir: str, saved_titles: set) -> dict:
    """
    Cào 1 topic. saved_titles chứa wiki_title đã lưu để phát hiện trùng sớm.
    """
    safe_name = re.sub(r'[^\w\u00C0-\u024F\s-]', '', topic).strip().replace(' ', '_')
    filepath = os.path.join(output_dir, f"{safe_name}.txt")

    # File đã có -> skip
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("WIKI_TITLE:"):
                        saved_titles.add(line.replace("WIKI_TITLE:", "").strip())
                        break
        except Exception:
            pass
        return {"topic": topic, "file": filepath, "status": "skipped"}

    queries = make_queries(topic)

    def _try_save(wiki_title, content, source_name):
        if wiki_title in saved_titles:
            return {"topic": topic, "wiki_title": wiki_title,
                    "status": "duplicate",
                    "note": f"Bài '{wiki_title}' đã cào cho topic khác, bỏ qua"}

        content = clean_text(content)
        word_count = len(content.split())

        # Rút entity TRƯỚC khi quyết định lưu
        # -> dùng để lọc bài không liên quan du lịch (người, tổ chức, sự kiện...)
        _site = "vi.wikipedia" if "Wikipedia" in source_name else "vi.wikivoyage"
        entities = wiki_fetch_links(wiki_title, site=_site)

        # Bỏ bài nếu: không có entity du lịch nào VÀ tên bài cũng không phải địa danh
        # Ví dụ: tìm "Sơn Trà" -> nếu entity toàn tên người -> bài về người, bỏ
        #        tìm "Sơn Trà" -> nếu entity có "Bán đảo Sơn Trà", "Đà Nẵng" -> giữ
        if not entities and not _is_tourism_entity(wiki_title):
            return {"topic": topic, "wiki_title": wiki_title,
                    "status": "skipped_not_tourism",
                    "note": f"Bài '{wiki_title}': 0 entity du lịch và title không phải địa danh -> bỏ"}

        entities_str = " | ".join(entities) if entities else ""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"TOPIC: {topic}\n")
            f.write(f"WIKI_TITLE: {wiki_title}\n")
            f.write(f"SOURCE: {source_name}\n")
            f.write(f"LANG: vi\n")
            f.write(f"WORDS: {word_count}\n")
            f.write(f"ENTITIES: {entities_str}\n")
            f.write("\n")
            f.write(content)

        saved_titles.add(wiki_title)
        return {"topic": topic, "wiki_title": wiki_title,
                "source": source_name, "lang": "vi",
                "file": filepath, "word_count": word_count,
                "entities_count": len(entities),
                "status": "success"}

    # Bước 1: Thử direct title lookup (chính xác hơn text search)
    # Thứ tự: Wikipedia VI trước (min 300 từ), WikiVoyage VI sau (min 150 từ)
    # Nếu title đã cào cho topic khác -> bỏ qua, thử title/source tiếp theo
    for site, source_name, min_words in SOURCES:
        for title_try in wiki_direct_titles(topic):
            if title_try in saved_titles:
                continue  # trùng bài đã cào -> không fetch, thử title/source khác
            content = wiki_fetch_content(title_try, site=site)
            if content and len(content.split()) >= min_words:
                result = _try_save(title_try, content, source_name)
                if result["status"] not in ("duplicate", "skipped_not_tourism"):
                    return result

    # Bước 2: Thử text search API (tìm gần đúng)
    for site, source_name, min_words in SOURCES:
        for query in queries:
            for wiki_title in wiki_search(query, site=site):
                if wiki_title in saved_titles:
                    continue  # trùng bài đã cào -> thử kết quả search tiếp theo
                content = wiki_fetch_content(wiki_title, site=site)
                if content and len(content.split()) >= min_words:
                    result = _try_save(wiki_title, content, source_name)
                    if result["status"] not in ("duplicate", "skipped_not_tourism"):
                        return result

    return {"topic": topic, "file": None, "status": "not_found",
            "note": "Không tìm thấy bài phù hợp (Wikipedia/WikiVoyage VI)"}


# MAIN

if __name__ == "__main__":
    OUTPUT_DIR = "data/raw/wiki_articles"
    MANIFEST_FILE = "data/raw/wiki_manifest.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    print("")
    print("PHASE 1: CÀO WIKIPEDIA VI + WIKIVOYAGE VI")
    print("")
    print(f"Topics: {len(TOPICS)}")
    print(f"Nguồn: Wikipedia VI -> WikiVoyage VI (toàn tiếng Việt)")
    print(f"Entity: rút wikilinks từ mỗi bài, lưu vào header ENTITIES")
    print(f"Ước tính: {len(TOPICS)*2//60}-{len(TOPICS)*3//60} phút\n")

    saved_titles = set()
    results = []
    for topic in tqdm(TOPICS, desc="Crawling", unit="topic"):
        result = crawl_one_topic(topic, OUTPUT_DIR, saved_titles)
        results.append(result)

        if result["status"] == "success":
            ec = result.get("entities_count", 0)
            tqdm.write(
                f"   [{result['source']}] {topic[:50]:<50} "
                f"({result['word_count']:,} từ | {ec} entities)"
            )
        elif result["status"] == "skipped":
            tqdm.write(f"   [SKIP]          {topic[:50]}")
        elif result["status"] == "duplicate":
            tqdm.write(f"   [TRÙNG]         {topic[:50]} -> bài '{result['wiki_title']}' đã cào")
        elif result["status"] == "skipped_not_tourism":
            tqdm.write(f"   [BỎ-ENTITY]     {topic[:50]} -> '{result['wiki_title']}' không phải địa danh")
        else:
            tqdm.write(f"   [NOT FOUND]     {topic[:50]}")

        time.sleep(1.0)

    # Lưu manifest
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    success       = [r for r in results if r["status"] == "success"]
    skipped       = [r for r in results if r["status"] == "skipped"]
    duplicate     = [r for r in results if r["status"] == "duplicate"]
    not_tourism   = [r for r in results if r["status"] == "skipped_not_tourism"]
    not_found     = [r for r in results if r["status"] == "not_found"]

    by_source = {}
    for r in success:
        src = r["source"]
        by_source[src] = by_source.get(src, 0) + 1

    print("")
    print("TỔNG KẾT:")
    print(f"   Thành công:      {len(success):>4} topics")
    for src, cnt in by_source.items():
        print(f"     [{src}]: {cnt} topics")
    print(f"   Đã có (skip):   {len(skipped):>4} topics")
    print(f"   Trùng bài wiki: {len(duplicate):>4} topics -> không lưu lại")
    print(f"   Bỏ (ko du lịch):{len(not_tourism):>4} topics -> entity=0 và title không phải địa danh")
    print(f"   Không tìm thấy: {len(not_found):>4} topics")

    if duplicate:
        print(f"\n   Topics bị trùng bài Wikipedia:")
        for r in duplicate:
            print(f"     - {r['topic']} -> trùng bài '{r['wiki_title']}'")

    if not_tourism:
        print(f"\n   Topics bị bỏ vì entity không phải địa danh du lịch:")
        for r in not_tourism:
            print(f"     - {r['topic']} -> bài '{r['wiki_title']}'")

    if not_found:
        print(f"\n   Topics không tìm thấy bài:")
        for r in not_found:
            print(f"     - {r['topic']}")

    if success:
        total_words = sum(r.get("word_count", 0) for r in success)
        total_entities = sum(r.get("entities_count", 0) for r in success)
        print(f"\n   Tổng từ đã cào:    {total_words:,}")
        print(f"   Trung bình/topic:  {total_words//len(success):,} từ")
        print(f"   Tổng entity rút:   {total_entities:,}")
        print(f"   TB entity/topic:   {total_entities//len(success)}")
