"""
TF-IDF RETRIEVER
Tải index đã build sẵn (vectorizer.pkl + doc_vectors.pkl) và tìm kiếm bằng cosine similarity.

Luồng sử dụng:
  Build index (1 lần):  python 04_tfidf_method.py  -> models/tfidf/
                        python 12_tfidf_wiki.py    -> models/tfidf_wiki/
  Runtime Streamlit:    TFIDFRetriever().load(thư_mục) -> retriever.search(query)
"""

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.vncorenlp_processor import VnCoreNLPProcessor


class TFIDFRetriever:
    """
    TF-IDF Retriever - chỉ dùng để load index đã build và tìm kiếm.
    Streamlit gọi: load() một lần khi khởi động, search() mỗi khi có query.
    """

    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        # Khởi tạo VnCoreNLP để tokenize query khi search
        try:
            self.vncore_processor = VnCoreNLPProcessor()
            self.use_vncorenlp = True

            def vncorenlp_tokenizer(text):
                return self.vncore_processor.tokenize(text.lower(), remove_stopwords_flag=True)

            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=False,       # đã lowercase trong tokenizer
                tokenizer=vncorenlp_tokenizer,
                token_pattern=None,    # dùng tokenizer tùy chỉnh
            )
        except Exception as e:
            print(f"Cảnh báo: VnCoreNLP không khả dụng, dùng regex thay thế. Lỗi: {e}")
            self.use_vncorenlp = False
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                strip_accents=None,
                analyzer='word',
                token_pattern=r'\w+',
            )

        self.doc_vectors = None

    def load(self, save_dir: str = 'models/tfidf'):
        """
        Load vectorizer và doc_vectors từ thư mục đã build.
        Streamlit gọi hàm này 1 lần lúc khởi động (@st.cache_resource).

        Args:
            save_dir: 'models/tfidf' cho Q&A, 'models/tfidf_wiki' cho wiki
        """
        with open(f'{save_dir}/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(f'{save_dir}/doc_vectors.pkl', 'rb') as f:
            self.doc_vectors = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Tìm kiếm top_k tài liệu phù hợp nhất với query.
        Streamlit gọi hàm này mỗi khi user nhập câu hỏi.

        Luồng xử lý:
          1. Tokenize query bằng VnCoreNLP + bỏ stopwords
          2. Chuyển query thành vector TF-IDF (cùng không gian với doc_vectors)
          3. Tính cosine similarity với toàn bộ doc_vectors
          4. Trả về top_k chỉ số và điểm số

        Args:
            query:  Câu hỏi của người dùng
            top_k:  Số kết quả trả về

        Returns:
            (indices, scores) - chỉ số tài liệu và điểm cosine [0, 1]
        """
        # Tokenize query nhất quán với lúc build index
        if self.use_vncorenlp and self.vncore_processor:
            tokens = self.vncore_processor.tokenize(query.lower(), remove_stopwords_flag=True)
            query_preprocessed = ' '.join(tokens)
        else:
            query_preprocessed = query.lower()

        # Chuyển query thành vector TF-IDF
        query_vector = self.vectorizer.transform([query_preprocessed])

        # Tính cosine similarity giữa query và toàn bộ tài liệu
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]

        # Lấy top_k chỉ số có điểm cao nhất
        top_indices = scores.argsort()[-top_k:][::-1]
        top_scores = scores[top_indices]

        return top_indices.tolist(), top_scores.tolist()
