"""
BM25 RETRIEVER
Tải index đã build sẵn (bm25_model.pkl + tokenized_docs.pkl + params.pkl) và tìm kiếm.

Luồng sử dụng:
  Build index (1 lần):  python 05_bm25_method.py  -> models/bm25/
                        python 10_index_wiki.py   -> models/bm25_wiki/
  Runtime Streamlit:    BM25Retriever().load(thư_mục) -> retriever.search(query)
"""

import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.vietnamese_stopwords import remove_stopwords
from src.preprocessing.vncorenlp_processor import VnCoreNLPProcessor


class BM25Retriever:
    """
    BM25 Retriever - chỉ dùng để load index đã build và tìm kiếm.
    Streamlit gọi: load() một lần khi khởi động, search() mỗi khi có query.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_docs = None

        # Khởi tạo VnCoreNLP để tokenize query khi search
        try:
            self.vncore_processor = VnCoreNLPProcessor()
            self.use_vncorenlp = True
        except Exception as e:
            print(f"Cảnh báo: VnCoreNLP không khả dụng, dùng underthesea thay thế. Lỗi: {e}")
            self.use_vncorenlp = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Tách từ tiếng Việt với VnCoreNLP + bỏ stopwords.
        Dùng cùng pipeline với lúc build index để kết quả nhất quán.
        """
        if self.use_vncorenlp:
            # Tách từ bằng VnCoreNLP + loại stopwords (phương pháp chính)
            return self.vncore_processor.tokenize(text.lower(), remove_stopwords_flag=True)
        else:
            # Fallback: underthesea + loại stopwords
            try:
                from underthesea import word_tokenize
                tokens = word_tokenize(text.lower(), format="text").split()
                return remove_stopwords(tokens)
            except:
                # Fallback cuối cùng: tách theo khoảng trắng + loại stopwords
                tokens = text.lower().split()
                return remove_stopwords(tokens)

    def load(self, save_dir: str = 'models/bm25'):
        """
        Load BM25 model và tokenized docs từ thư mục đã build.
        Streamlit gọi hàm này 1 lần lúc khởi động (@st.cache_resource).

        Args:
            save_dir: 'models/bm25' cho Q&A, 'models/bm25_wiki' cho wiki
        """
        with open(f'{save_dir}/bm25_model.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        # Hỗ trợ cả hai tên file (Q&A dùng tokenized_docs, wiki dùng tokenized_chunks)
        import os as _os
        docs_path = f'{save_dir}/tokenized_docs.pkl'
        if not _os.path.exists(docs_path):
            docs_path = f'{save_dir}/tokenized_chunks.pkl'
        with open(docs_path, 'rb') as f:
            self.tokenized_docs = pickle.load(f)
        with open(f'{save_dir}/params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.k1 = params['k1']
            self.b = params['b']

    def search(self, query: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Tìm kiếm top_k tài liệu phù hợp nhất với query.
        Streamlit gọi hàm này mỗi khi user nhập câu hỏi.

        Luồng xử lý:
          1. Tokenize query bằng VnCoreNLP + bỏ stopwords
          2. Tính điểm BM25 cho toàn bộ tài liệu
          3. Trả về top_k chỉ số và điểm số

        Args:
            query:  Câu hỏi của người dùng
            top_k:  Số kết quả trả về

        Returns:
            (indices, scores) - chỉ số tài liệu và điểm BM25 thô
        """
        # Tách từ câu hỏi nhất quán với lúc build index
        tokenized_query = self._tokenize(query)

        # Tính điểm BM25 cho toàn bộ tài liệu
        scores = self.bm25.get_scores(tokenized_query)

        # Lấy top_k chỉ số có điểm cao nhất
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]

        return top_indices.tolist(), top_scores.tolist()
