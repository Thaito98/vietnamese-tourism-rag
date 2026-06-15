"""
CROSS-ENCODER RERANKER
Xếp hạng lại kết quả truy hồi bằng cross-encoder đa ngôn ngữ.

Khác với SBERT (bi-encoder) vốn mã hóa query và document thành hai vector riêng
rồi so cosine, cross-encoder đưa cặp (query, document) qua model trong cùng một
lần forward. Model đọc được cả hai cùng lúc nên đánh giá mức liên quan chính xác
hơn, đổi lại chậm hơn nhiều - chỉ dùng được cho vài chục ứng viên, không quét
được toàn kho. Vì vậy nó đứng sau retrieval, không thay thế retrieval.

Model: BAAI/bge-reranker-v2-m3 (XLM-RoBERTa, đa ngôn ngữ, hỗ trợ tiếng Việt)
Không cần VnCoreNLP - model tự tokenize ở cấp subword.

Luồng sử dụng:
    reranker = CrossEncoderReranker()
    ranked = reranker.rerank(query, documents, top_k=6)
"""

from typing import List, Tuple, Optional


class CrossEncoderReranker:
    """
    Cross-encoder reranker dùng chung cho Streamlit app và script đánh giá.

    Model được nạp lazy ở lần rerank đầu tiên để không làm chậm lúc khởi động
    khi người dùng chưa bật reranking.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512,
                 device: Optional[str] = None):
        """
        Args:
            model_name: Tên model cross-encoder trên HuggingFace
            max_length: Số token tối đa cho mỗi cặp (query, document).
                        Cặp dài hơn sẽ bị cắt bớt phần document.
            device: 'cuda' hoặc 'cpu'. Để None thì tự phát hiện.
        """
        self.model_name = model_name
        self.max_length = max_length
        self._device = device
        self._model = None   # nạp lazy

    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return self._device

    def load(self):
        """Nạp model. Gọi tường minh khi muốn kiểm soát thời điểm nạp."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
        return self._model

    def score(self, query: str, documents: List[str],
              batch_size: int = 32) -> List[float]:
        """
        Chấm điểm liên quan cho từng document so với query.

        Args:
            query:      Câu hỏi của người dùng
            documents:  Danh sách nội dung văn bản cần chấm
            batch_size: Số cặp xử lý mỗi batch

        Returns:
            Danh sách điểm, cùng thứ tự với documents. Điểm cao là liên quan hơn.
        """
        if not documents:
            return []

        model = self.load()
        pairs = [(query, doc) for doc in documents]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, documents: List[str], top_k: int = 6,
               batch_size: int = 32) -> List[Tuple[int, float]]:
        """
        Xếp hạng lại documents theo mức liên quan với query.

        Args:
            query:      Câu hỏi của người dùng
            documents:  Danh sách nội dung văn bản
            top_k:      Số kết quả trả về
            batch_size: Số cặp xử lý mỗi batch

        Returns:
            List[(chỉ_số_gốc, điểm)] đã sắp xếp giảm dần theo điểm.
            Chỉ số gốc là vị trí trong danh sách documents đầu vào, dùng để
            ánh xạ ngược về metadata của kết quả.
        """
        scores = self.score(query, documents, batch_size=batch_size)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
