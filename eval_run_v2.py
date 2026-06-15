"""
eval_run_v2.py - Đánh giá cả Q&A KB và Wiki KB, cộng kết quả RRF gộp.

Khác eval_run.py:
  - Câu hỏi test lấy từ nội dung thực tế trong dataset (khớp với phong cách Q&A)
  - Mỗi câu hỏi search trên 3 nguồn: Q&A KB, Wiki KB, RRF gộp
  - Đầu ra: eval_results_v3.csv (không rerank) / eval_results_rerank.csv (--rerank)
            ->  grade.py  ->  eval_score_v2.py  ->  eval_metrics_v2.csv
"""

import io, sys, os, pickle, warnings, argparse
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace",
    line_buffering=True,
    write_through=True,
)
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))

import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--skip-search", action="store_true")
parser.add_argument("--rerank", action="store_true",
                    help="Chạy các biến thể có cross-encoder rerank, "
                         "ghi ra eval_results_rerank.csv")
parser.add_argument("--out", default=None,
                    help="File CSV đầu ra. Mặc định tùy theo có --rerank hay không.")
args = parser.parse_args()

_default_out = "eval_results_rerank.csv" if args.rerank else "eval_results_v3.csv"
OUT_PATH     = os.path.join(ROOT, args.out or _default_out)
FIELDNAMES = ["query_id", "test_question", "source",
              "method", "rank", "score", "result_text", "grade"]

# Câu hỏi test - lấy từ topics_bank.py 
from topics_bank import get_questions as _get_questions
TEST_QUESTIONS = _get_questions()   # [{"id": ..., "question": ...}, ...]

# 4 phương pháp truy hồi nền - tên khớp với web
# (SBERT = "Vietnamese SBERT", Hybrid-RRF = web's "Hybrid")
BASE_METHODS = [
    "TF-IDF",
    "BM25",
    "SBERT",
    "Hybrid-RRF",
]

# Chế độ chạy:
#   --rerank      : chạy 4 biến thể có cross-encoder rerank
#   (không cờ)    : chạy 4 phương pháp gốc, không rerank
#
# Hai chế độ ghi ra hai file riêng vì RETRIEVE_K khác nhau, không chạy chung
# một lượt được. Cả hai file phải được chạy lại và chấm lại cùng một đợt: trộn
# kết quả mới với điểm từ lần chấm cũ sẽ so sánh hai thang chấm khác nhau.
# Wilcoxon ghép theo query_id nên so được giữa hai file, miễn là cùng bộ câu hỏi.
if args.rerank:
    METHODS    = [m + "+Rerank" for m in BASE_METHODS]
    RETRIEVE_K = 20     # lấy 20 ứng viên để cross-encoder có cái mà lọc
else:
    METHODS    = list(BASE_METHODS)
    RETRIEVE_K = 5      # lấy thẳng 5, không có bước lọc lại phía sau

FINAL_K = 5
# Chấm điểm được thực hiện bởi grade.py (sau khi eval_run_v2.py tạo CSV)

def _save_csv(rows):
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

# Phase 1: Search 


def rrf_merge(qa_list, wiki_list, k=60, top_k=5):
    """
    Trả về top_k kết quả gộp từ qa_list và wiki_list theo RRF.
    qa_list  = [(text, score), ...]  đã sắp xếp rank 1..n
    wiki_list= [(text, score), ...]  đã sắp xếp rank 1..n
    """
    scores = {}
    for rank, (text, _) in enumerate(qa_list, 1):
        scores[text] = scores.get(text, 0) + 1/(k + rank)
    for rank, (text, _) in enumerate(wiki_list, 1):
        scores[text] = scores.get(text, 0) + 1/(k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

if args.skip_search:
    if not os.path.exists(OUT_PATH):
        print(f"Không tìm thấy {OUT_PATH}.")
        sys.exit(1)
    rows = []
    with open(OUT_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    print(f"Đọc {len(rows)} dòng từ {OUT_PATH}")

else:
    os.chdir(ROOT)
    print("Đang load dữ liệu và models...")

    # Q&A data
    with open("data/processed/questions.pkl",     "rb") as f: questions     = pickle.load(f)
    with open("data/processed/answers.pkl",       "rb") as f: answers       = pickle.load(f)
    with open("data/processed/contexts.pkl",      "rb") as f: contexts      = pickle.load(f)
    with open("data/processed/answer_starts.pkl", "rb") as f: answer_starts = pickle.load(f)
    print(f"  Q&A: {len(questions)} cặp  |  contexts: {len(contexts)}")

    # Wiki data
    import pandas as pd
    df_wiki = pd.read_csv("data/processed/wiki_chunks.csv")
    # child_text: dùng để index/retrieval (BM25/SBERT/TF-IDF đã build trên child)
    # parent_text: dùng cho result_text => grader chấm (khớp với web gửi Ollama)
    _has_parent = "parent_text" in df_wiki.columns
    wiki_texts   = df_wiki["child_text"].tolist()
    wiki_parents = df_wiki["parent_text"].tolist() if _has_parent else wiki_texts
    wiki_titles  = df_wiki["wiki_title"].tolist() if "wiki_title" in df_wiki.columns else [""] * len(df_wiki)
    if not _has_parent:
        print("  [CẢNH BÁO] wiki_chunks.csv thiếu parent_text - dùng child_text thay thế")
    print(f"  Wiki: {len(wiki_texts)} chunks  |  parent_text={'OK' if _has_parent else 'THIẾU'}")

    # TF-IDF Q&A
    from src.retrieval.tfidf_retriever import TFIDFRetriever
    tfidf_qa = TFIDFRetriever(); tfidf_qa.load("models/tfidf")
    print("  TF-IDF Q&A: OK")

    # TF-IDF Wiki
    tfidf_wiki = TFIDFRetriever(); tfidf_wiki.load("models/tfidf_wiki")
    print("  TF-IDF Wiki: OK")

    # BM25 Q&A
    from src.retrieval.bm25_retriever import BM25Retriever
    bm25_qa = BM25Retriever(); bm25_qa.load("models/bm25")
    print("  BM25 Q&A: OK")

    # BM25 Wiki
    bm25_wiki = BM25Retriever(); bm25_wiki.load("models/bm25_wiki")
    print("  BM25 Wiki: OK")

    # SBERT Q&A + Wiki
    from sentence_transformers import SentenceTransformer
    import torch
    qa_emb   = np.load("data/embeddings/sbert_embeddings.npy")
    wiki_emb = np.load("data/embeddings/wiki_embeddings.npy")
    sbert    = SentenceTransformer("keepitreal/vietnamese-sbert")
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  SBERT: OK (device={device}, qa={qa_emb.shape}, wiki={wiki_emb.shape})")

    # Cross-encoder reranker (chỉ nạp nếu có phương pháp nào cần)
    reranker = None
    if any("Rerank" in m for m in METHODS):
        from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(device=device)
        reranker.load()
        print(f"  Cross-encoder: OK ({reranker.model_name}, device={device})")

    # Helper: expand_answer - copy chính xác từ streamlit_app.py
    def _expand_answer(answer, answer_start, context):
        if answer_start < 0 or not context:
            return answer
        def _is_abbr(text, pos):
            if pos <= 0 or pos >= len(text): return False
            bc = ""; i = pos - 1
            while i >= 0 and text[i].isalpha(): bc = text[i] + bc; i -= 1
            if not bc or len(bc) > 5: return False
            if bc in ['Dr','Mr','Mrs','Ms','St','Tp','TP','USA','UK']: pass
            elif not (1 <= len(bc) <= 4 and bc.isupper()): return False
            if pos + 1 < len(text):
                nc = text[pos + 1]
                if nc.isupper(): return True
                if nc == ' ' and pos + 2 < len(text) and text[pos + 2].isupper(): return True
            return False
        start = answer_start
        while start > 0 and context[start - 1] not in '.!?\n': start -= 1
        while start < len(context) and context[start] in '.!?\n \t': start += 1
        end = answer_start + len(answer)
        while end < len(context):
            if context[end] in '.!?\n':
                if context[end] == '.' and _is_abbr(context, end): end += 1; continue
                else: break
            end += 1
        if end < len(context) and context[end] in '.!?': end += 1
        expanded = context[start:end].strip()
        return expanded if expanded != answer else answer

    # Helper: RRF merge (khớp với web's Hybrid = BM25+SBERT=>RRF)
    def _rrf(bm_idx, sb_idx, k=60, top_k=5):
        sc = {}
        for rank, i in enumerate(bm_idx): sc[i] = sc.get(i, 0.0) + 1.0/(k + rank + 1)
        for rank, i in enumerate(sb_idx): sc[i] = sc.get(i, 0.0) + 1.0/(k + rank + 1)
        items = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [i for i, _ in items], [s for _, s in items]

    # Helper search functions

    def _sbert_search(query, emb_matrix, top_k=5):
        q = sbert.encode(query, convert_to_tensor=True, device=device).cpu().numpy()
        q /= np.linalg.norm(q) + 1e-10
        m  = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)
        s  = np.dot(m, q)
        t  = np.argsort(s)[-top_k:][::-1]
        return t.tolist(), s[t].tolist()

    # 4 phương pháp × 2 KBs

    def search_qa(method, query, top_k=5, cands=50):
        if method == "TF-IDF":
            return tfidf_qa.search(query, top_k)
        elif method == "BM25":
            return bm25_qa.search(query, top_k)
        elif method == "SBERT":
            return _sbert_search(query, qa_emb, top_k)
        else:  # Hybrid-RRF - khớp với web's "Hybrid" (BM25+SBERT=>RRF)
            bm_idx, _ = bm25_qa.search(query, top_k=cands)
            sb_idx, _ = _sbert_search(query, qa_emb, top_k=cands)
            return _rrf(bm_idx, sb_idx, top_k=top_k)

    def search_wiki(method, query, top_k=5, cands=50):
        if method == "TF-IDF":
            return tfidf_wiki.search(query, top_k)
        elif method == "BM25":
            return bm25_wiki.search(query, top_k)
        elif method == "SBERT":
            return _sbert_search(query, wiki_emb, top_k)
        else:  # Hybrid-RRF - khớp với web's "Hybrid" (BM25+SBERT=>RRF)
            bm_idx, _ = bm25_wiki.search(query, top_k=cands)
            sb_idx, _ = _sbert_search(query, wiki_emb, top_k=cands)
            return _rrf(bm_idx, sb_idx, top_k=top_k)

    def qa_text(idx):
        # Dùng expand_answer - khớp với web (build_unified_context gửi cho Ollama)
        ans = answers[idx]       if idx < len(answers)       else ""
        ast = answer_starts[idx] if idx < len(answer_starts) else -1
        ctx = contexts[idx]      if idx < len(contexts)      else ""
        return _expand_answer(ans, ast, ctx)

    def wiki_text(idx):
        # Dùng parent_text - khớp với web (build_unified_context gửi cho Ollama)
        t = wiki_titles[idx]  if idx < len(wiki_titles)  else ""
        p = wiki_parents[idx] if idx < len(wiki_parents) else ""
        return f"[{t}] {p}" if t else p

    # Run search 

    n_total = len(TEST_QUESTIONS) * len(METHODS)
    _mode = "CÓ RERANK" if args.rerank else "KHÔNG RERANK"
    print(f"\nChế độ: {_mode}  ->  {os.path.basename(OUT_PATH)}")
    print(f"Chạy {len(TEST_QUESTIONS)} câu × {len(METHODS)} phương pháp "
          f"× {FINAL_K} kết quả RRF = {n_total * FINAL_K} dòng")
    print(f"  Truy hồi {RETRIEVE_K} ứng viên/KB, lấy {FINAL_K} kết quả cuối")
    print("Tiến trình: ", end="", flush=True)

    # Nếu CSV đã tồn tại, đọc lại để resume từ câu còn thiếu
    done_qids = set()
    rows = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, newline="", encoding="utf-8-sig") as _f:
            for _row in csv.DictReader(_f):
                rows.append({k: v for k, v in _row.items() if k in FIELDNAMES})
                done_qids.add(int(_row["query_id"]))
        if done_qids:
            print(f"Resume: đã có {len(done_qids)} câu ({min(done_qids)}-{max(done_qids)}), bỏ qua.")

    for q in TEST_QUESTIONS:
        qid      = q["id"]
        if qid in done_qids:
            continue
        question = q["question"]

        for method in METHODS:
            try:
                use_rr      = method.endswith("+Rerank")
                base_method = method[:-len("+Rerank")] if use_rr else method

                # Lấy RETRIEVE_K ứng viên từ mỗi KB, gộp RRF, rồi lấy FINAL_K
                # kết quả cuối - bằng cross-encoder nếu có rerank, bằng thứ tự
                # RRF nếu không.
                qa_idx,   qa_sc   = search_qa(base_method,   question, top_k=RETRIEVE_K)
                wiki_idx, wiki_sc = search_wiki(base_method, question, top_k=RETRIEVE_K)

                # Chỉ lưu RRF - đây là kết quả hệ thống thực sự trả về người dùng
                qa_pairs   = [(qa_text(i),  s) for i, s in zip(qa_idx,  qa_sc)]
                wiki_pairs = [(wiki_text(i), s) for i, s in zip(wiki_idx, wiki_sc)]
                merged = rrf_merge(qa_pairs, wiki_pairs, top_k=RETRIEVE_K)

                if use_rr:
                    cand_texts = [t for t, _ in merged]
                    ranked = reranker.rerank(question, cand_texts, top_k=FINAL_K)
                    merged = [(cand_texts[i], s) for i, s in ranked]
                else:
                    merged = merged[:FINAL_K]

                for rank, (text, sc) in enumerate(merged, 1):
                    rows.append({
                        "query_id": qid, "test_question": question,
                        "source": "rrf", "method": method, "rank": rank,
                        "score": round(float(sc), 6),
                        "result_text": text,
                        "grade": "",
                    })

            except Exception as e:
                print(f"\n  LỖI [{method}] q#{qid}: {e}")
                for rank in range(1, FINAL_K + 1):
                    rows.append({
                        "query_id": qid, "test_question": question,
                        "source": "rrf", "method": method, "rank": rank,
                        "score": 0, "result_text": f"ERROR: {e}", "grade": "",
                    })

        n_total_q = len(TEST_QUESTIONS)
        print("." if qid % 10 != 0 else f"[{qid}/{n_total_q}]", end="", flush=True)
        if qid % 10 == 0:          # lưu tạm mỗi 10 câu
            _save_csv(rows)

    print(f"\n\nLưu {len(rows)} dòng -> {OUT_PATH}")
    _save_csv(rows)
    print(f"  ({len(TEST_QUESTIONS)} câu × {len(METHODS)} phương pháp × {FINAL_K} kết quả RRF)")

# Phase 2 (tuỳ chọn): Auto-grade 

print(f"\nĐã lưu -> {OUT_PATH}")
print(f"Bước tiếp: python grade.py --csv {os.path.basename(OUT_PATH)}"
      f"  ->  python eval_score_v2.py --csv {os.path.basename(OUT_PATH)}")
