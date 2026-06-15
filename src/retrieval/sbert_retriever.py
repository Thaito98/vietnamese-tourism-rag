"""
SBERT RETRIEVER
Tải embeddings đã encode sẵn (.npy) và tìm kiếm bằng cosine similarity.

Luồng sử dụng:
  Build embeddings (1 lần):  python 06_sbert_method.py -> data/embeddings/sbert_embeddings.npy
                              python 13_sbert_wiki.py  -> data/embeddings/wiki_embeddings.npy
  Runtime Streamlit:         SBERTRetriever(model).load_embeddings(path) -> retriever.search(query)
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class SBERTRetriever:
    """
    Vietnamese SBERT Retriever - chỉ dùng để load embeddings đã encode và tìm kiếm.
    Streamlit gọi: __init__() + load_embeddings() một lần, search() mỗi khi có query.
    """

    def __init__(self, model_name: str = 'keepitreal/vietnamese-sbert'):
        """
        Load model SentenceTransformer từ HuggingFace (hoặc cache cục bộ).
        Tự động chọn GPU nếu có, ngược lại dùng CPU.

        Args:
            model_name: Tên model trên HuggingFace
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.doc_embeddings = None

    def load_embeddings(self, embedding_path: str = 'data/embeddings/sbert_embeddings.npy'):
        """
        Load embeddings đã encode sẵn từ file .npy.
        Streamlit gọi hàm này 1 lần lúc khởi động (@st.cache_resource).

        Args:
            embedding_path: 'data/embeddings/sbert_embeddings.npy' cho Q&A,
                            'data/embeddings/wiki_embeddings.npy'  cho wiki
        """
        self.doc_embeddings = np.load(embedding_path)

    def search(self, query: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Tìm kiếm top_k tài liệu phù hợp nhất với query.
        Streamlit gọi hàm này mỗi khi user nhập câu hỏi.

        Luồng xử lý:
          1. Encode query thành vector (cùng không gian với doc_embeddings)
          2. Tính cosine similarity bằng dot product (embeddings đã normalize)
          3. Trả về top_k chỉ số và điểm số

        Args:
            query:  Câu hỏi của người dùng
            top_k:  Số kết quả trả về

        Returns:
            (indices, scores) - chỉ số tài liệu và điểm cosine [0, 1]
        """
        # Encode và normalize query
        query_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )

        # Normalize doc embeddings rồi tính cosine bằng dot product
        doc_norms = self.doc_embeddings / np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        scores = np.dot(doc_norms, query_emb)

        # Lấy top_k chỉ số có điểm cao nhất
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]

        return top_indices.tolist(), top_scores.tolist()
