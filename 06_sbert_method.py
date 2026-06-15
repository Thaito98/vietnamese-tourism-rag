"""
PHẦN 2.3: VIETNAMESE SBERT RETRIEVER
Model: keepitreal/vietnamese-sbert (Sentence-BERT cho tiếng Việt)

- Nếu chưa có sbert_embeddings.npy -> encode toàn bộ kho Q&A và lưu
- Nếu đã có -> load và dùng luôn
- Chạy với tham số --force để encode lại dù đã có (khi data thay đổi):
    python 06_sbert_method.py --force
"""

import sys
import os
import pickle
import numpy as np
import json
import time

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

force_encode = '--force' in sys.argv

print("")
print("PHẦN 2.3: VIETNAMESE SBERT RETRIEVER (keepitreal/vietnamese-sbert)")
print("")

embeddings_path = 'data/embeddings/sbert_embeddings.npy'
metadata_path   = 'data/embeddings/sbert_metadata.json'

# LOAD DATABASE
print("[1] LOAD DATABASE")
print("")

with open('data/processed/questions.pkl', 'rb') as f:
    questions = pickle.load(f)

print(f" Loaded {len(questions):,} questions")

# LOAD MODEL
print("\n[2] LOAD VIETNAMESE SBERT MODEL")
print("")

from sentence_transformers import SentenceTransformer
import torch

if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f" GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
else:
    device = 'cpu'
    print(" Không có GPU - dùng CPU")

model = SentenceTransformer('keepitreal/vietnamese-sbert', device=device)
print(f" Model loaded! Device: {device}")

# ENCODE NẾU CHƯA CÓ HOẶC --force
need_encode = not os.path.exists(embeddings_path) or force_encode

if need_encode:
    if force_encode:
        print(f"\n[3] ENCODE LẠI (--force) {len(questions):,} câu hỏi")
    else:
        print(f"\n[3] CHƯA CÓ EMBEDDINGS - ENCODE {len(questions):,} câu hỏi")
    print("")

    batch_size = 128 if device == 'cuda' else 32
    print(f" batch_size: {batch_size}  |  device: {device}")

    start_time = time.time()
    doc_embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    elapsed = time.time() - start_time
    print(f"\n Encode xong trong {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f" Shape: {doc_embeddings.shape}")

    os.makedirs('data/embeddings', exist_ok=True)
    np.save(embeddings_path, doc_embeddings)
    print(f" Saved: {embeddings_path}")

    metadata = {
        'model': 'keepitreal/vietnamese-sbert',
        'num_documents': len(questions),
        'embedding_dim': int(doc_embeddings.shape[1]),
        'device': device,
        'encoded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f" Saved: {metadata_path}")

else:
    print(f"\n[3] LOAD EMBEDDINGS ĐÃ CÓ")
    print("")
    doc_embeddings = np.load(embeddings_path)
    print(f" Loaded: {embeddings_path}")
    print(f" Shape: {doc_embeddings.shape}")

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        print(f" Encoded at: {meta.get('encoded_at', 'N/A')}  |  Device: {meta.get('device', 'N/A')}")

    if len(questions) != doc_embeddings.shape[0]:
        print(f"\n WARNING: Số câu hỏi ({len(questions):,}) != số embeddings ({doc_embeddings.shape[0]:,})")
        print(" Chạy lại: python 06_sbert_method.py --force")

# SEARCH FUNCTION
print("\n[4] SEARCH FUNCTION")
print("")

doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

def search_sbert(query, top_k=5):
    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True, device=device)
    scores = np.dot(doc_norms, query_emb)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return top_indices.tolist(), scores[top_indices].tolist()

print(" Search function ready")

print(f"""
 HOÀN THÀNH

 Embeddings: {doc_embeddings.shape[0]:,} x {doc_embeddings.shape[1]}  |  Device: {device}

 Chạy lại để encode lại (khi data thay đổi):
   python 06_sbert_method.py --force
""")
