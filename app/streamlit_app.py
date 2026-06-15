"""
STREAMLIT WEB APP - CHATBOT HỎI ĐÁP DU LỊCH VIỆT NAM
"""

import streamlit as st
import pickle
import numpy as np
import json
import time
import re
import urllib.request
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.sbert_retriever import SBERTRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker

# PAGE CONFIG
st.set_page_config(
    page_title="Chatbot Du Lịch Việt Nam",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #FF4B4B;
    }
    .question-text {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .answer-text {
        font-size: 1rem;
        color: #333;
        line-height: 1.6;
    }
    .score-badge {
        background-color: #FF4B4B;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .method-badge {
        background-color: #0068C9;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# QUERY NORMALIZATION - chuẩn hóa viết tắt trước khi truy hồi
# VnCoreNLP tokenize "TP. Hồ Chí Minh" -> token "tp", nếu expand thì "thành_phố" khác "tp"
# Chỉ expand viết tắt thành tên riêng (bỏ phần "tp"/"thành phố")
ABBREV_MAP = {
    r"\btp\.?hcm\b":  "hồ chí minh",
    r"\btphcm\b":     "hồ chí minh",
    r"\bsài gòn\b":   "hồ chí minh",
    r"\bhn\b":        "hà nội",
    r"\bđn\b":        "đà nẵng",
    r"\bnt\b":        "nha trang",
    r"\bpq\b":        "phú quốc",
    r"\bđl\b":        "đà lạt",
    r"\bha\b":        "hội an",
    r"\bhl\b":        "hạ long",
}

def normalize_query(query: str) -> str:
    """Chuẩn hóa viết tắt phổ biến trong câu hỏi."""
    q = query.lower().strip()
    for pattern, replacement in ABBREV_MAP.items():
        q = re.sub(pattern, replacement, q)
    return q


# CATEGORY DETECTION
CATEGORY_RULES: dict[str, list[str]] = {
    "địa điểm": [
        "ở đâu", "địa điểm", "tham quan", "thăm", "điểm đến",
        "vịnh", "núi", "đảo", "bãi biển", "hang", "thác",
        "phố cổ", "khu du lịch", "thành phố", "tỉnh",
    ],
    "ẩm thực": [
        "ăn gì", "ăn ở đâu", "món", "đặc sản", "ẩm thực",
        "quán", "nhà hàng", "uống", "café", "cà phê", "hải sản",
        "phở", "bánh", "bún",
    ],
    "văn hóa": [
        "lễ hội", "phong tục", "lịch sử", "truyền thống",
        "di tích", "chùa", "đền", "miếu", "văn hóa", "di sản",
        "bảo tàng", "làng",
    ],
    "mẹo": [
        "visa", "vé", "chi phí", "giá", "bao nhiêu tiền",
        "an toàn", "lưu trú", "khách sạn", "hostel", "resort",
        "đi lại", "phương tiện", "xe", "tàu", "máy bay",
        "thời tiết", "mùa", "nên đi",
    ],
}


CATEGORY_PROMPTS = {
    "địa điểm": "Đây là câu hỏi về địa điểm du lịch tại Việt Nam. Hãy trả lời cụ thể về địa danh, vị trí, đặc điểm nổi bật.",
    "ẩm thực":  "Đây là câu hỏi về ẩm thực và đặc sản Việt Nam. Hãy nêu tên món, địa điểm thưởng thức và đặc điểm.",
    "văn hóa":  "Đây là câu hỏi về văn hóa, lịch sử, lễ hội tại Việt Nam. Hãy cung cấp thông tin về nét đặc trưng văn hóa.",
    "mẹo":      "Đây là câu hỏi thực tế về mẹo du lịch (chi phí, đi lại, lưu trú, visa...). Hãy trả lời cụ thể, thực tiễn.",
    "chung":    "",
}


def classify_question(query: str) -> str:
    """Phân loại câu hỏi theo category du lịch dựa trên từ khóa."""
    q_lower = query.lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in q_lower for kw in keywords):
            return category
    return "chung"


# LOAD DATA & MODELS (with caching)
@st.cache_data
def load_database():
    """Load Q&A database"""
    import contextlib
    import io

    # Suppress all output during load to avoid encoding errors
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with open('data/processed/questions.pkl', 'rb') as f:
            questions = pickle.load(f)
        with open('data/processed/answers.pkl', 'rb') as f:
            answers = pickle.load(f)
        with open('data/processed/ids.pkl', 'rb') as f:
            ids = pickle.load(f)
        with open('data/processed/contexts.pkl', 'rb') as f:
            contexts = pickle.load(f)
        with open('data/processed/titles.pkl', 'rb') as f:
            titles = pickle.load(f)
        with open('data/processed/answer_starts.pkl', 'rb') as f:
            answer_starts = pickle.load(f)
    return questions, answers, ids, contexts, titles, answer_starts

def expand_answer(answer, answer_start, context, expand_chars=150):
    """
    Mở rộng câu trả lời ngắn bằng cách lấy TOÀN BỘ CÂU chứa answer
    (bao gồm cả dấu chấm cuối, BỎ QUA dấu chấm viết tắt như TP., Tp., Dr., etc.)
    
    Args:
        answer: Câu trả lời gốc (ngắn)
        answer_start: Vị trí bắt đầu của answer trong context
        context: Context đầy đủ
        expand_chars: Số ký tự mở rộng về mỗi phía
    
    Returns:
        Câu trả lời mở rộng (toàn bộ câu)
    """
    if answer_start < 0 or not context:
        return answer  # Fallback to original
    
    # Helper function: Check if a period is an abbreviation
    def is_abbreviation(text, pos):
        """Kiểm tra xem dấu chấm có phải là viết tắt không"""
        if pos <= 0 or pos >= len(text):
            return False
        
        # Lấy từ trước dấu chấm
        before_chars = ""
        i = pos - 1
        while i >= 0 and text[i].isalpha():
            before_chars = text[i] + before_chars
            i -= 1
        
        if not before_chars or len(before_chars) > 5:
            return False
        
        # Danh sách viết tắt phổ biến (chữ cái đầu hoa)
        common_abbreviations = ['Dr', 'Mr', 'Mrs', 'Ms', 'St', 'Tp', 'TP', 'USA', 'UK']
        
        # Kiểm tra xem có phải viết tắt phổ biến không
        is_common_abbr = before_chars in common_abbreviations
        
        # Hoặc là viết tắt toàn chữ HOA (1-4 ký tự)
        is_uppercase_abbr = (1 <= len(before_chars) <= 4 and before_chars.isupper())
        
        if not (is_common_abbr or is_uppercase_abbr):
            return False
        
        # Kiểm tra sau dấu chấm phải có chữ HOA
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            # Ngay sau là chữ HOA: TP.Hồ, Dr.Smith
            if next_char.isupper():
                return True
            # Space + chữ HOA: TP. Hồ, Dr. Smith
            if next_char == ' ' and pos + 2 < len(text) and text[pos + 2].isupper():
                return True
        
        return False
    
    # Tìm đầu câu hiện tại (KHÔNG lấy qua dấu chấm trước đó)
    start = answer_start
    while start > 0 and context[start - 1] not in '.!?\n':
        start -= 1
    
    # Skip dấu câu và khoảng trắng ở đầu
    while start < len(context) and context[start] in '.!?\n \t':
        start += 1
    
    # Tìm cuối câu hiện tại (BỎ QUA dấu chấm viết tắt)
    end = answer_start + len(answer)
    while end < len(context):
        if context[end] in '.!?\n':
            # Nếu là dấu chấm, kiểm tra xem có phải viết tắt không
            if context[end] == '.' and is_abbreviation(context, end):
                # Bỏ qua dấu chấm viết tắt, tiếp tục tìm
                end += 1
                continue
            else:
                # Đây là dấu kết thúc câu thật sự
                break
        end += 1
    
    # Bao gồm cả dấu chấm cuối (nếu có)
    if end < len(context) and context[end] in '.!?':
        end += 1
    
    # Lấy text và strip
    expanded = context[start:end].strip()
    
    # Nếu expanded giống hệt answer gốc, return answer gốc
    if expanded == answer:
        return answer
    
    # Nếu expanded chỉ dài hơn 1-2 ký tự (thêm dấu chấm), vẫn return expanded
    # (không check 1.2x nữa vì có thể chỉ thêm dấu chấm)
    return expanded

@st.cache_resource
def load_tfidf_model():
    """Load TF-IDF model"""
    import contextlib
    import io

    retriever = TFIDFRetriever()

    # Suppress all output during load to avoid encoding errors
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retriever.load('models/tfidf')

    return retriever

@st.cache_resource
def load_bm25_model():
    """Load BM25 model"""
    import contextlib
    import io

    retriever = BM25Retriever()

    # Suppress all output during load to avoid encoding errors
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retriever.load('models/bm25')

    return retriever

@st.cache_resource
def load_sbert_model():
    """Load SBERT Q&A bằng SBERTRetriever (nhất quán với load_tfidf_model, load_bm25_model)."""
    import contextlib, io
    retriever = SBERTRetriever('keepitreal/vietnamese-sbert')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retriever.load_embeddings('data/embeddings/sbert_embeddings.npy')
    return retriever


# WIKI - LOAD FUNCTIONS (mỗi method load index riêng của nó)

@st.cache_resource
def load_wiki_chunks():
    """Load wiki_chunks.csv - dùng chung cho mọi method."""
    import contextlib, io, pandas as pd
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        df = pd.read_csv("data/processed/wiki_chunks.csv")
    return df

@st.cache_resource
def load_tfidf_wiki():
    """Load TF-IDF wiki index bằng TFIDFRetriever (nhất quán với load_tfidf_model).
    Index xây bởi 12_tfidf_wiki.py. Nếu thiếu -> chạy: python 12_tfidf_wiki.py"""
    import contextlib, io
    retriever = TFIDFRetriever()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retriever.load('models/tfidf_wiki')
    return retriever

@st.cache_resource
def load_bm25_wiki():
    """Load BM25 wiki index bằng BM25Retriever (nhất quán với load_bm25_model).
    Index xây bởi 10_index_wiki.py. Nếu thiếu -> chạy: python 10_index_wiki.py"""
    import contextlib, io
    retriever = BM25Retriever()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retriever.load('models/bm25_wiki')
    return retriever

@st.cache_resource
def load_sbert_wiki():
    """Load SBERT wiki bằng SBERTRetriever (nhất quán với load_sbert_model)."""
    import contextlib, io
    retriever = SBERTRetriever('keepitreal/vietnamese-sbert')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retriever.load_embeddings('data/embeddings/wiki_embeddings.npy')
    return retriever

def load_rag_pipeline():
    """Trả về 3 object đã cache: wiki chunks, BM25Retriever wiki, SBERTRetriever wiki."""
    return load_wiki_chunks(), load_bm25_wiki(), load_sbert_wiki()

def search_hybrid_rrf_wiki(query, top_k=5, bm25_k=50, sbert_k=50, rrf_k=60):
    """Hybrid RRF wiki: BM25 wiki + SBERT wiki chạy độc lập, gộp hai ranked lists qua RRF."""
    df_chunks  = load_wiki_chunks()
    bm25_retr  = load_bm25_wiki()
    sbert_retr = load_sbert_wiki()

    bm25_idx, _  = bm25_retr.search(normalize_query(query), bm25_k)
    sbert_idx, _ = sbert_retr.search(query, sbert_k)

    rrf_scores = {}
    for rank, idx in enumerate(bm25_idx):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(sbert_idx):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    final_indices = [i for i, _ in sorted_items]
    final_scores  = [s for _, s in sorted_items]

    return _wiki_rows_to_results(df_chunks, final_indices, final_scores)

# WIKI SEARCH THEO TỪNG PHƯƠNG PHÁP (nhất quán với method chính)

def _wiki_rows_to_results(df_chunks, indices, scores):
    """Chuyển (indices, scores) -> list dict wiki results (child-parent schema)."""
    results = []
    for i, s in zip(indices, scores):
        row = df_chunks.iloc[int(i)]
        results.append({
            "wiki_title": str(row["wiki_title"]),
            "child_text": str(row["child_text"]),   # phần khớp query (hiển thị)
            "chunk_text": str(row["parent_text"]),  # đoạn đầy đủ => Ollama nhận
            "parent_id":  str(row["parent_id"]),    # dùng để dedup
            "source":     str(row.get("source", "")),
            "score":      float(s),
        })
    return results

def search_tfidf_wiki(query, top_k=5):
    """TF-IDF search trên wiki_chunks - dùng TFIDFRetriever.search() (nhất quán với Q&A)."""
    retriever = load_tfidf_wiki()
    df_chunks = load_wiki_chunks()
    indices, scores = retriever.search(normalize_query(query), top_k)
    return _wiki_rows_to_results(df_chunks, indices, scores)

def search_bm25_wiki(query, top_k=5):
    """BM25 search trên wiki_chunks - dùng BM25Retriever.search() (nhất quán với Q&A)."""
    retriever = load_bm25_wiki()
    df_chunks = load_wiki_chunks()
    indices, scores = retriever.search(normalize_query(query), top_k)
    return _wiki_rows_to_results(df_chunks, indices, scores)

def search_sbert_wiki(query, top_k=5):
    """SBERT search trên wiki_chunks - dùng SBERTRetriever.search() (nhất quán với Q&A)."""
    retriever = load_sbert_wiki()
    df_chunks = load_wiki_chunks()
    indices, scores = retriever.search(query, top_k)
    return _wiki_rows_to_results(df_chunks, indices, scores)

def search_wiki_for_method(method, query, top_k=5):
    """Router: chọn đúng wiki search function theo method đang dùng.

    TF-IDF          -> TF-IDF wiki  (cosine [0,1])
    BM25            -> BM25 wiki    (normalized [0,1])
    Vietnamese SBERT-> SBERT wiki   (cosine [0,1])
    Hybrid          -> BM25 wiki + SBERT wiki, gộp qua RRF
    """
    if method == "TF-IDF":
        return search_tfidf_wiki(query, top_k=top_k)
    elif method == "BM25":
        return search_bm25_wiki(query, top_k=top_k)
    elif method == "Vietnamese SBERT":
        return search_sbert_wiki(query, top_k=top_k)
    else:  # Hybrid
        return search_hybrid_rrf_wiki(query, top_k=top_k)

def _wiki_method_label(method):
    """Tên method wiki để hiển thị trong UI."""
    return {
        "TF-IDF":            "TF-IDF wiki",
        "BM25":              "BM25 wiki (normalized)",
        "Vietnamese SBERT":  "SBERT wiki",
        "Hybrid":            "BM25+SBERT wiki (RRF)",
    }.get(method, "wiki")

def _call_ollama(prompt: str, model: str) -> str:
    """Gọi Ollama API, trả về text response."""
    payload = json.dumps({
        "model": model, "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300}
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()

def ask_ollama_streamlit(query, chunks, model="qwen2.5:3b"):
    """Gọi Ollama tổng hợp câu trả lời từ list wiki chunks."""
    context = "\n\n".join([
        f"Thông tin {i+1} (Nguồn: {c['wiki_title']}):\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    ])
    return ask_ollama_streamlit_ctx(query, context, model)

def ask_ollama_streamlit_ctx(query, context: str, model="qwen2.5:3b",
                             category_hint: str = ""):
    """Gọi Ollama tổng hợp câu trả lời từ context string đã dựng sẵn.
    Prompt được thiết kế để LLM CHỈ tổng hợp từ context, KHÔNG tự sinh câu trả lời mới.
    category_hint: gợi ý loại câu hỏi để LLM trả lời đúng trọng tâm hơn."""
    category_line = f"\n{category_hint}" if category_hint else ""
    prompt = f"""Bạn là trợ lý du lịch Việt Nam chuyên nghiệp. Dựa HOÀN TOÀN vào thông tin dưới đây để trả lời câu hỏi.{category_line}
KHÔNG tự bịa thêm. Nếu không đủ thông tin, hãy nói "Tôi không có đủ thông tin."

{context}

Câu hỏi: {query}
Trả lời (tiếng Việt, ngắn gọn và rõ ràng):"""
    try:
        return _call_ollama(prompt, model)
    except Exception as e:
        return f" Ollama chưa chạy hoặc lỗi: {e}\n\nContext đầu tiên:\n{context[:300]}"

def llm_rerank_bm25(query, indices, scores, questions, answers, answer_starts, contexts, titles, ids,
                    top_k_rerank=20, final_top_k=5, model="qwen2.5:3b"):
    """BM25 + LLM Re-ranking:
    Bước 1: BM25 lấy top_k_rerank kết quả (nhanh).
    Bước 2: LLM đánh giá và xếp hạng lại top_k_rerank đó, trả về final_top_k tốt nhất.
    LLM CHỈ chọn từ danh sách có sẵn, KHÔNG tự sinh câu trả lời mới.
    """
    # Lấy top_k_rerank ứng viên từ BM25
    candidates = []
    for idx, score in zip(indices[:top_k_rerank], scores[:top_k_rerank]):
        exp_ans = expand_answer(answers[idx], answer_starts[idx], contexts[idx])
        candidates.append({
            "idx": idx, "bm25_score": float(score),
            "question": questions[idx],
            "answer": exp_ans,
            "title": titles[idx],
            "id": ids[idx],
        })

    # Xây prompt để LLM chọn những kết quả liên quan nhất
    candidate_text = "\n".join([
        f"[{i+1}] Câu hỏi: {c['question']}\n    Câu trả lời: {c['answer'][:150]}"
        for i, c in enumerate(candidates)
    ])
    rerank_prompt = f"""Bạn là chuyên gia đánh giá kết quả tìm kiếm du lịch Việt Nam.
Dưới đây là {len(candidates)} kết quả tìm kiếm cho câu hỏi: "{query}"

{candidate_text}

Nhiệm vụ: Chọn {final_top_k} kết quả LIÊN QUAN NHẤT với câu hỏi trên.
Chỉ trả về {final_top_k} số thứ tự, cách nhau bởi dấu phẩy. Ví dụ: 3,1,5,2,4
KHÔNG giải thích thêm. Chỉ trả về dãy số:"""

    try:
        raw = _call_ollama(rerank_prompt, model).strip()
        # Parse danh sách số từ response
        import re as _re
        nums = [int(x.strip()) for x in _re.findall(r'\d+', raw) if 1 <= int(x.strip()) <= len(candidates)]
        # Loại trùng, giữ thứ tự
        seen, ranked = set(), []
        for n in nums:
            if n not in seen:
                seen.add(n)
                ranked.append(n)
        # Nếu LLM trả thiếu thì bổ sung từ BM25 gốc
        for i in range(1, len(candidates) + 1):
            if len(ranked) >= final_top_k:
                break
            if i not in seen:
                ranked.append(i)
        ranked = ranked[:final_top_k]
        return [candidates[n - 1] for n in ranked]
    except Exception:
        # Fallback: trả nguyên BM25 top
        return candidates[:final_top_k]

# SEARCH FUNCTIONS
def search_tfidf(query, top_k=5):
    """Search using TF-IDF"""
    retriever = load_tfidf_model()
    indices, scores = retriever.search(query, top_k=top_k)
    return indices, scores

def search_bm25(query, top_k=5):
    """Search using BM25 - trả về raw scores (không normalize).
    Gọi search_bm25_normalized() nếu cần so sánh với TF-IDF/SBERT [0,1]."""
    retriever = load_bm25_model()
    indices, scores = retriever.search(query, top_k=top_k)
    return indices, scores

def search_bm25_normalized(query, top_k=5):
    """BM25 search với scores normalize về [0,1] (min-max trong kết quả trả về).
    Dùng cho display và confidence threshold để nhất quán với TF-IDF/SBERT."""
    indices, scores = search_bm25(query, top_k=top_k)
    s = np.array(scores, dtype=float)
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-10:
        norm_scores = np.ones_like(s).tolist()
    else:
        norm_scores = ((s - mn) / (mx - mn)).tolist()
    return indices, norm_scores

def search_sbert(query, top_k=5):
    """SBERT Q&A search - dùng SBERTRetriever.search() (nhất quán với search_tfidf, search_bm25)."""
    retriever = load_sbert_model()
    return retriever.search(query, top_k)

def search_hybrid_rrf(query, top_k=5, bm25_k=50, sbert_k=50, rrf_k=60):
    """Hybrid RRF: BM25 và SBERT chạy độc lập, gộp hai ranked lists qua RRF.

    Mỗi retriever trả ra ranked list riêng, không trộn điểm số trực tiếp.
    RRF score = 1/(rrf_k + rank + 1) - cộng dồn từ cả hai lists.
    """
    bm25_idx, _  = search_bm25(query, top_k=bm25_k)
    sbert_idx, _ = search_sbert(query, top_k=sbert_k)

    rrf_scores = {}
    for rank, idx in enumerate(bm25_idx):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(sbert_idx):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [int(i) for i, _ in sorted_items], [float(s) for _, s in sorted_items]




# UNIFIED SEARCH - RRF merge QA + Wiki

def rrf_merge_results(qa_pairs, wiki_items, k: int = 60):
    """
    Reciprocal Rank Fusion: gộp kết quả từ Q&A KB và Wiki KB.
    Dùng thứ hạng thay vì điểm số vì 2 KB dùng model khác nhau,
    điểm số không so sánh trực tiếp được.
    RRF(item) = 1/(k + rank)
    """
    pool = []
    rrf_score = {}

    for rank, (idx, score) in enumerate(qa_pairs):
        uid = f"qa_{idx}"
        rrf_score[uid] = rrf_score.get(uid, 0.0) + 1.0 / (k + rank + 1)
        pool.append({"uid": uid, "source": "qa", "qa_idx": int(idx),
                     "orig_score": float(score)})

    for rank, chunk in enumerate(wiki_items):
        uid = f"wiki_{rank}"
        rrf_score[uid] = rrf_score.get(uid, 0.0) + 1.0 / (k + rank + 1)
        pool.append({"uid": uid, "source": "wiki", "chunk": chunk,
                     "orig_score": float(chunk["score"])})

    pool.sort(key=lambda x: rrf_score[x["uid"]], reverse=True)
    for item in pool:
        item["rrf_score"] = rrf_score[item["uid"]]
    return pool


# CROSS-ENCODER RERANKING

@st.cache_resource
def load_reranker():
    """Load cross-encoder reranker. Nạp 1 lần, dùng lại cho mọi query."""
    import contextlib, io
    reranker = CrossEncoderReranker()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        reranker.load()
    return reranker


def _pool_item_text(item, questions, answers, answer_starts, contexts, titles) -> str:
    """Lấy nội dung văn bản của một item trong pool để đưa cho reranker chấm.

    Q&A: ghép câu hỏi với câu trả lời đã mở rộng, vì cả hai đều mang thông tin
    quyết định mức liên quan.
    Wiki: dùng child_text - đúng đoạn đã khớp query, ngắn nên không bị cắt bởi
    giới hạn 512 token của model.
    """
    if item["source"] == "qa":
        idx = item["qa_idx"]
        exp = expand_answer(answers[idx], answer_starts[idx], contexts[idx])
        return f"{questions[idx]} {exp}"
    else:
        c = item["chunk"]
        return c.get("child_text") or c.get("chunk_text", "")


def rerank_pool(query, pool, questions, answers, answer_starts, contexts, titles,
                top_k: int = 6):
    """Xếp hạng lại pool (Q&A + Wiki đã gộp qua RRF) bằng cross-encoder.

    RRF gộp theo thứ hạng nên không biết nội dung thực sự liên quan tới đâu.
    Cross-encoder đọc trực tiếp cặp (câu hỏi, nội dung) nên sửa được những
    trường hợp RRF xếp cao nhưng nội dung lệch chủ đề.

    Trả về pool đã sắp xếp lại, mỗi item có thêm khóa 'rerank_score'.
    """
    if not pool:
        return []

    reranker = load_reranker()
    docs = [
        _pool_item_text(it, questions, answers, answer_starts, contexts, titles)
        for it in pool
    ]
    ranked = reranker.rerank(query, docs, top_k=top_k)

    out = []
    for orig_idx, score in ranked:
        item = dict(pool[orig_idx])
        item["rerank_score"] = float(score)
        out.append(item)
    return out


def build_unified_context(pool, questions, answers, answer_starts, contexts,
                          titles, max_items: int = 6) -> str:
    """Ghép context từ pool (Q&A + Wiki) để gửi cho Ollama.
    Wiki items được dedup theo parent_id: nếu nhiều children từ cùng 1 đoạn văn,
    chỉ giữ 1 (cái có score cao nhất, vì pool đã được sắp xếp theo score).
    """
    parts = []
    seen_parents: set = set()
    count = 0
    for item in pool:
        if count >= max_items:
            break
        if item["source"] == "qa":
            idx = item["qa_idx"]
            exp = expand_answer(answers[idx], answer_starts[idx], contexts[idx])
            parts.append(
                f"[Nguồn Q&A - {titles[idx]}]\n"
                f"Câu hỏi: {questions[idx]}\n"
                f"Câu trả lời: {exp}"
            )
            count += 1
        else:
            c = item["chunk"]
            pid = c.get("parent_id", "")
            if pid and pid in seen_parents:
                continue  # bỏ qua: cùng đoạn văn, tránh lặp nội dung
            seen_parents.add(pid)
            parts.append(
                f"[Nguồn Wikipedia - {c['wiki_title']}]\n"
                f"{c['chunk_text']}"
            )
            count += 1
    return "\n\n".join(parts)


# MAIN APP
def main():
    # Header
    st.markdown('<div class="main-header">Chatbot Du Lịch Việt Nam</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hỏi đáp thông tin du lịch Việt Nam bằng AI</div>', unsafe_allow_html=True)
    
    # Load data
    try:
        questions, answers, ids, contexts, titles, answer_starts = load_database()
        st.sidebar.success(f" Loaded {len(questions):,} Q&A pairs")
    except Exception as e:
        st.error(f" Không thể load database: {e}")
        return

    # Sidebar - Settings
    st.sidebar.header(" Cài đặt")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Chọn phương pháp tìm kiếm:",
        ["TF-IDF", "BM25", "Vietnamese SBERT", "Hybrid"],
        index=0,
        help=(
            "**TF-IDF**: Bag-of-words + cosine similarity.\n"
            "**BM25**: Probabilistic keyword ranking (Okapi BM25).\n"
            "**Vietnamese SBERT**: Semantic embedding, hiểu ngữ nghĩa.\n"
            "**Hybrid**: BM25 và SBERT chạy song song độc lập, "
            "gộp hai ranked lists qua Reciprocal Rank Fusion (RRF).\n\n"
            "Tất cả 4 phương pháp đều hỗ trợ: Wiki bổ sung  Ollama synthesis."
        ),
    )

    # Top-k selection
    top_k = st.sidebar.slider("Số kết quả hiển thị:", min_value=1, max_value=10, value=5)

    # Ngưỡng tin cậy (confidence threshold) - áp cho cả Q&A và Wikipedia
    score_threshold      = 0.0
    wiki_score_threshold = 0.0
    if method in ["TF-IDF", "BM25", "Vietnamese SBERT", "Hybrid"]:
        st.sidebar.subheader(" Ngưỡng tin cậy")
        # Hybrid dùng RRF score (max ~0.033), các method khác dùng cosine/BM25 (0-1)
        if method == "Hybrid":
            _default_thresh      = {"Hybrid": 0.010}
            _default_thresh_wiki = {"Hybrid": 0.010}
            slider_max, slider_step, slider_format = 0.05, 0.002, "%0.3f"
            thresh_help      = "RRF score tối đa ~0.033 (item xuất hiện ở cả BM25 và SBERT rank 1)"
            thresh_wiki_help = "RRF score tối đa ~0.033 (item xuất hiện ở cả BM25 và SBERT rank 1)"
        else:
            _default_thresh      = {"TF-IDF": 0.25, "BM25": 0.25, "Vietnamese SBERT": 0.50}
            _default_thresh_wiki = {"TF-IDF": 0.20, "BM25": 0.20, "Vietnamese SBERT": 0.30}
            slider_max, slider_step, slider_format = 1.0, 0.05, "%0.2f"
            thresh_help      = "Nếu score top-1 Q&A < ngưỡng -> cảnh báo câu hỏi ngoài phạm vi dataset"
            thresh_wiki_help = "Nếu score top-1 Wikipedia < ngưỡng -> cảnh báo không tìm thấy nội dung Wiki liên quan"
        score_threshold = st.sidebar.slider(
            "Q&A - Score tối thiểu:", 0.0, slider_max,
            float(_default_thresh.get(method, 0.25 if method != "Hybrid" else 0.010)),
            slider_step,
            format=slider_format,
            key=f"score_threshold_{method}",
            help=thresh_help
        )
        wiki_score_threshold = st.sidebar.slider(
            "Wikipedia - Score tối thiểu:", 0.0, slider_max,
            float(_default_thresh_wiki.get(method, 0.20 if method != "Hybrid" else 0.010)),
            slider_step,
            format=slider_format,
            key=f"wiki_score_threshold_{method}",
            help=thresh_wiki_help
        )

    # Hybrid parameters
    hybrid_candidates = 60
    if method == "Hybrid":
        st.sidebar.subheader("Hybrid Parameters")
        hybrid_candidates = st.sidebar.slider(
            "Candidates mỗi retriever (K):", 20, 200, 60, 10,
            help="BM25 lấy top-K và SBERT lấy top-K, sau đó RRF gộp hai ranked lists."
        )
        st.sidebar.caption(
            f"BM25 top-{hybrid_candidates} + SBERT top-{hybrid_candidates} "
            f"=> RRF (k=60) => top-{top_k}"
        )

    # Luôn dùng cả Q&A + Wikipedia (không còn chế độ chọn riêng)
    wiki_only  = False
    use_wiki   = True
    wiki_top_k = top_k

    # Cross-encoder reranking
    st.sidebar.subheader(" Reranking")
    use_rerank = st.sidebar.checkbox(
        "Bật cross-encoder rerank", value=False,
        help=(
            "Sau khi RRF gộp Q&A + Wikipedia, dùng cross-encoder "
            "(BAAI/bge-reranker-v2-m3) xếp hạng lại theo mức liên quan thực sự.\n\n"
            "RRF chỉ gộp theo thứ hạng nên không biết nội dung. Cross-encoder đọc "
            "trực tiếp cặp (câu hỏi, nội dung) nên chính xác hơn.\n\n"
            "Lần bật đầu tiên sẽ tải model khoảng 600MB."
        ),
    )
    rerank_candidates = 20
    if use_rerank:
        rerank_candidates = st.sidebar.slider(
            "Số ứng viên đưa vào rerank:", 10, 50, 20, 5,
            help="Lấy nhiều ứng viên từ RRF rồi rerank chọn ra nguồn tốt nhất. "
                 "Nhiều hơn thì chính xác hơn nhưng chậm hơn."
        )

    # Ollama luôn bật - chỉ cho chọn model và số nguồn
    use_ollama_synth = True
    st.sidebar.subheader(" Ollama LLM")
    ollama_synth_model = st.sidebar.selectbox(
        "Model:", ["qwen2.5:3b", "qwen2.5:7b"], index=0, key="ollama_synth_model"
    )
    ollama_synth_ctx_k = st.sidebar.slider(
        "Số nguồn gửi LLM:", 2, 10, min(top_k * 2, 6), key="ollama_synth_k",
        help="Số kết quả (Q&A + Wiki gộp chung) đưa vào context cho Ollama tổng hợp"
    )

    
    # Main area - Search
    st.header(" Đặt câu hỏi của bạn")

    # Query input
    query = st.text_input(
        "Nhập câu hỏi:",
        value=st.session_state.get('query', ''),
        placeholder="Ví dụ: Hà Nội có những địa điểm du lịch nào?",
        key="query_input"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 1, 4])
    search_clicked = col1.button(" Tìm kiếm", type="primary", use_container_width=True)
    clear_clicked = col2.button(" Xóa", use_container_width=True)
    
    if clear_clicked:
        st.session_state['query'] = ''
        st.rerun()
    
    # Perform search - luôn dùng cả Q&A + Wiki + Ollama
    if search_clicked and query.strip():

        # WIKI-ONLY MODE (giữ lại để tránh lỗi reference, nhưng wiki_only luôn False)
        if wiki_only:
            wiki_method_label = _wiki_method_label(method)
            _synth_tag = f"   {ollama_synth_model}" if use_ollama_synth else ""
            st.markdown(
                f'<span class="method-badge">Chỉ Wikipedia  {wiki_method_label}{_synth_tag}</span>',
                unsafe_allow_html=True
            )
            st.markdown("---")

            with st.spinner(f"Đang tìm trong Wikipedia ({wiki_method_label})..."):
                t0 = time.time()
                try:
                    wiki_results = search_wiki_for_method(method, query, top_k=wiki_top_k)
                    elapsed_w = (time.time() - t0) * 1000


                    st.success(
                        f" Tìm thấy {len(wiki_results)} kết quả Wikipedia "
                        f"trong {elapsed_w:.1f}ms  |  Phương pháp: {wiki_method_label}"
                    )

                    # ── Ollama synthesis trên wiki chunks (thay thế RAG cũ) ───
                    if use_ollama_synth:
                        synth_k_w = min(ollama_synth_ctx_k, len(wiki_results))
                        wiki_ctx = "\n\n".join(
                            f"Thông tin {i+1} (Nguồn Wikipedia: {c['wiki_title']}):\n{c['chunk_text']}"
                            for i, c in enumerate(wiki_results[:synth_k_w])
                        )
                        with st.spinner(f" Ollama ({ollama_synth_model}) đang tổng hợp từ wiki..."):
                            t_llm0 = time.time()
                            try:
                                synth_w = ask_ollama_streamlit_ctx(query, wiki_ctx, model=ollama_synth_model)
                                t_llm_ms = (time.time() - t_llm0) * 1000
                                st.markdown("###  Câu trả lời tổng hợp (Ollama + Wikipedia)")
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                            padding: 1.4rem 1.6rem; border-radius: 0.8rem; margin-bottom: 1rem;">
                                    <div style="font-size:0.75rem; color:rgba(255,255,255,0.8); margin-bottom:0.5rem;">
                                          <b>{ollama_synth_model}</b> &nbsp;|&nbsp;
                                        {wiki_method_label} top-{synth_k_w} &nbsp;|&nbsp; {t_llm_ms:.0f}ms
                                    </div>
                                    <div style="font-size:1.05rem; color:white; line-height:1.8;">
                                        {synth_w.replace(chr(10), '<br>')}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown("---")
                            except Exception as e_llm:
                                st.error(f" Ollama lỗi: {e_llm} - Kiểm tra Ollama tại localhost:11434")

                    # Danh sách wiki chunks
                    st.markdown("###  Nguồn Wiki tham khảo")
                    for rank, chunk in enumerate(wiki_results, 1):
                        with st.expander(f"#{rank} [{chunk['score']:.4f}]  {chunk['wiki_title'][:60]}"):
                            st.markdown(f"""
                            <div style="background-color:#f0f4f8; padding:1rem; border-radius:0.5rem;
                                        border-left:4px solid #764ba2;">
                                <div style="font-size:0.95rem; color:#333; line-height:1.6;">
                                    {chunk['child_text']}
                                </div>
                                <div style="margin-top:0.4rem; font-size:0.75rem; color:#888;">
                                    Nguồn: {chunk['source']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("Xem đoạn văn đầy đủ (parent)"):
                                st.markdown(f"""
                                <div style="background-color:#f9f9f9; padding:0.8rem; border-radius:0.4rem;
                                            border-left:3px solid #aaa; font-size:0.9rem; color:#555; line-height:1.7;">
                                    {chunk['chunk_text']}
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f" Lỗi tìm kiếm Wikipedia: {e}")
                    st.exception(e)

        # UNIFIED MODE - luôn dùng cả Q&A + Wiki + Ollama
        else:
            # Category detection 
            q_category     = classify_question(query)
            cat_hint       = CATEGORY_PROMPTS.get(q_category, "")

            # Khi bật rerank, lấy nhiều ứng viên hơn để cross-encoder có cái mà chọn.
            # Không có rerank thì giữ nguyên top_k như cũ.
            retrieve_k = max(top_k, rerank_candidates) if use_rerank else top_k

            # Tìm kiếm song song QA + Wiki
            with st.spinner(f"Đang tìm kiếm bằng {method}..."):
                t0 = time.time()
                try:
                    if method == "TF-IDF":
                        qa_indices, qa_scores = search_tfidf(query, top_k=retrieve_k)
                    elif method == "BM25":
                        qa_indices, qa_scores = search_bm25_normalized(query, top_k=retrieve_k)
                    elif method == "Vietnamese SBERT":
                        qa_indices, qa_scores = search_sbert(query, top_k=retrieve_k)
                    else:  # Hybrid
                        qa_indices, qa_scores = search_hybrid_rrf(
                            query, top_k=retrieve_k,
                            bm25_k=hybrid_candidates, sbert_k=hybrid_candidates
                        )

                    wiki_results = search_wiki_for_method(method, query, top_k=retrieve_k)
                    elapsed_ms = (time.time() - t0) * 1000

                except Exception as e:
                    st.error(f"Lỗi tìm kiếm: {e}")
                    st.exception(e)
                    st.stop()

            # RRF: gộp QA + Wiki theo thứ hạng 
            qa_pairs  = list(zip(qa_indices, qa_scores))
            rrf_pool  = rrf_merge_results(qa_pairs, wiki_results)

            _retr_note = f" (lấy {retrieve_k} ứng viên cho rerank)" if use_rerank else ""
            st.success(
                f"Tìm thấy {len(qa_indices)} Q&A + {len(wiki_results)} Wiki -> "
                f"gộp {len(rrf_pool)} kết quả{_retr_note} | {elapsed_ms:.0f}ms | {method}"
            )
            _rr_tag = " &nbsp;|&nbsp; + Rerank" if use_rerank else ""
            st.markdown(
                f'<span class="method-badge">Method: {method} &nbsp;|&nbsp; Q&A + Wikipedia + Ollama'
                f'{_rr_tag}&nbsp;|&nbsp; {q_category.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

            # Confidence check - lọc nguồn không đủ ngưỡng ra khỏi context Ollama
            top_qa_score   = float(qa_scores[0]) if qa_scores else 0.0
            top_wiki_score = float(wiki_results[0]["score"]) if wiki_results else 0.0

            qa_passes   = top_qa_score   >= score_threshold
            wiki_passes = top_wiki_score >= wiki_score_threshold

            if not qa_passes:
                st.warning(
                    f"Score Q&A cao nhất: **{top_qa_score:.4f}** < ngưỡng **{score_threshold:.4f}**. "
                    "Nguồn Q&A bị loại khỏi context - Ollama chỉ dùng Wikipedia."
                )
            if not wiki_passes:
                st.warning(
                    f"Score Wikipedia cao nhất: **{top_wiki_score:.4f}** < ngưỡng **{wiki_score_threshold:.4f}**. "
                    "Nguồn Wikipedia bị loại khỏi context - Ollama chỉ dùng Q&A."
                )
            if not qa_passes and not wiki_passes:
                st.error(
                    "Cả Q&A và Wikipedia đều không đạt ngưỡng tin cậy. "
                    "Ollama sẽ trả lời dựa trên kiến thức nội tại, không có nguồn tham khảo."
                )

            # Lọc pool: chỉ giữ nguồn đạt ngưỡng để gửi Ollama
            ollama_pool = [
                item for item in rrf_pool
                if (item["source"] == "qa"   and qa_passes)
                or (item["source"] == "wiki" and wiki_passes)
            ]

            # Cross-encoder rerank: xếp lại pool theo mức liên quan thực sự
            rerank_ms = None
            if use_rerank and ollama_pool:
                with st.spinner("Cross-encoder đang xếp hạng lại..."):
                    t_rr = time.time()
                    try:
                        ollama_pool = rerank_pool(
                            query, ollama_pool,
                            questions, answers, answer_starts, contexts, titles,
                            top_k=ollama_synth_ctx_k,
                        )
                        rerank_ms = (time.time() - t_rr) * 1000
                    except Exception as e_rr:
                        st.warning(
                            f"Reranker lỗi, dùng thứ tự RRF gốc: {e_rr}"
                        )

                if rerank_ms is not None:
                    st.caption(
                        f"Rerank {len(rrf_pool)} ứng viên -> {len(ollama_pool)} nguồn "
                        f"| {rerank_ms:.0f}ms | BAAI/bge-reranker-v2-m3"
                    )
                    with st.expander("Xem thứ hạng sau rerank"):
                        for r, it in enumerate(ollama_pool, 1):
                            if it["source"] == "qa":
                                label = questions[it["qa_idx"]]
                            else:
                                label = f"[Wiki] {it['chunk']['wiki_title']}"
                            st.markdown(
                                f"**#{r}** &nbsp; `{it['rerank_score']:.4f}` &nbsp; "
                                f"{label[:90]}"
                            )

            # Ollama synthesis
            unified_ctx = build_unified_context(
                ollama_pool, questions, answers, answer_starts, contexts, titles,
                max_items=ollama_synth_ctx_k
            )
            with st.spinner(f" Ollama ({ollama_synth_model}) đang tổng hợp câu trả lời..."):
                t_llm = time.time()
                try:
                    synth_answer = ask_ollama_streamlit_ctx(
                        query, unified_ctx, model=ollama_synth_model,
                        category_hint=cat_hint,
                    )
                    t_llm_ms = (time.time() - t_llm) * 1000
                    st.markdown("###  Câu trả lời")
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 1.4rem 1.6rem; border-radius: 0.8rem; margin-bottom: 1rem;">
                        <div style="font-size:0.75rem; color:rgba(255,255,255,0.8); margin-bottom:0.6rem;">
                             <b>{ollama_synth_model}</b> &nbsp;|&nbsp; {method}
                            &nbsp;|&nbsp; top-{ollama_synth_ctx_k} (Q&A+Wiki)
                            &nbsp;|&nbsp; {q_category}
                            &nbsp;|&nbsp; {t_llm_ms:.0f}ms
                        </div>
                        <div style="font-size:1.05rem; color:white; line-height:1.8;">
                            {synth_answer.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e_llm:
                    st.error(f"Ollama lỗi: {e_llm}")
                    st.info("Kiểm tra Ollama đang chạy tại localhost:11434  ->  `ollama serve`")

            st.markdown("---")

            # Nguồn tham khảo (tab QA | tab Wikipedia)
            # Chỉ hiển thị top_k như người dùng chọn, dù retrieval có lấy nhiều hơn
            # để phục vụ rerank.
            qa_show   = list(zip(qa_indices, qa_scores))[:top_k]
            wiki_show = wiki_results[:top_k]

            st.markdown("###  Nguồn tham khảo")
            tab_qa, tab_wiki = st.tabs([
                f" Q&A Dataset ({len(qa_show)} kết quả)",
                f" Wikipedia ({len(wiki_show)} kết quả)",
            ])

            with tab_qa:
                for rank, (idx, score) in enumerate(qa_show, 1):
                    expanded_answer = expand_answer(
                        answers[idx], answer_starts[idx], contexts[idx], expand_chars=150
                    )
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <span style="font-size:1.1rem; font-weight:bold; color:#FF4B4B;">#{rank}</span>
                            <span class="score-badge">Score: {score:.4f}</span>
                        </div>
                        <div class="question-text"> {questions[idx]}</div>
                        <div class="answer-text"> {expanded_answer}</div>
                        <div style="margin-top:0.5rem; font-size:0.8rem; color:#888;">
                            <b>Nguồn:</b> {titles[idx]} &nbsp;|&nbsp; <b>ID:</b> {ids[idx]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Xem toàn bộ context"):
                        st.markdown(f"""
                        <div style="background-color:#f9f9f9; padding:1rem; border-radius:0.5rem;
                                    border-left:3px solid #0068C9; font-size:0.9rem; color:#555; line-height:1.8;">
                            {contexts[idx]}
                        </div>""", unsafe_allow_html=True)

            with tab_wiki:
                for rank, chunk in enumerate(wiki_show, 1):
                    st.markdown(f"""
                    <div style="background-color:#f0f4f8; padding:1.2rem; border-radius:0.5rem;
                                margin-bottom:0.8rem; border-left:4px solid #764ba2;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                            <span style="font-size:1.1rem; font-weight:bold; color:#764ba2;">#{rank}</span>
                            <span style="background:#764ba2; color:white; padding:0.2rem 0.5rem;
                                         border-radius:0.3rem; font-size:0.85rem;">
                                Score: {chunk['score']:.4f}
                            </span>
                        </div>
                        <div style="font-weight:bold; margin-bottom:0.3rem;"> {chunk['wiki_title']}</div>
                        <div style="font-size:0.95rem; color:#333; line-height:1.6;">
                            {chunk['child_text']}
                        </div>
                        <div style="margin-top:0.4rem; font-size:0.75rem; color:#888;">
                            Nguồn: {chunk['source']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Xem đoạn văn đầy đủ (parent)"):
                        st.markdown(f"""
                        <div style="background-color:#f9f9f9; padding:0.8rem; border-radius:0.4rem;
                                    border-left:3px solid #aaa; font-size:0.9rem; color:#555; line-height:1.7;">
                            {chunk['chunk_text']}
                        </div>
                        """, unsafe_allow_html=True)

    elif search_clicked:
        st.warning("Vui lòng nhập câu hỏi!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p> Đồ án: Chatbot Hỏi Đáp Du Lịch Việt Nam</p>
        <p> Powered by TF-IDF, BM25, Vietnamese SBERT, Hybrid (BM25+SBERT via RRF) & RAG (Wiki + Ollama LLM)</p>
    </div>
    """, unsafe_allow_html=True)

# RUN APP
if __name__ == "__main__":
    main()
