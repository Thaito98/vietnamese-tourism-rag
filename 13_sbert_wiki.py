"""
ENCODE SBERT EMBEDDINGS CHO WIKI CHUNKS (chạy trên máy local)
Model: keepitreal/vietnamese-sbert
Encode raw chunk_text (SBERT không cần VnCoreNLP tokenize)
Lưu: data/embeddings/wiki_embeddings.npy + wiki_metadata.json

- Nếu chưa có wiki_embeddings.npy -> encode toàn bộ wiki chunks và lưu
- Nếu đã có -> bỏ qua, không encode lại
- Chạy với tham số --force để encode lại dù đã có (khi wiki_chunks.csv thay đổi):
    python 13_sbert_wiki.py --force
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import time

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

force_encode = '--force' in sys.argv

print("")
print("PHẦN 4.2: ENCODE SBERT WIKI EMBEDDINGS (local)")
print("")

emb_path  = "data/embeddings/wiki_embeddings.npy"
meta_path = "data/embeddings/wiki_metadata.json"

# LOAD WIKI CHUNKS
print("[1] LOAD WIKI CHUNKS")
print("")

df = pd.read_csv("data/processed/wiki_chunks.csv")
texts = [t.lower().strip() for t in df["child_text"].fillna("").tolist()]
print(f" Loaded {len(texts):,} chunks")

# KIỂM TRA EMBEDDINGS ĐÃ CÓ
if os.path.exists(emb_path) and not force_encode:
    existing = np.load(emb_path)
    print(f"\n Embeddings đã tồn tại: {emb_path}")
    print(f" Shape: {existing.shape}")

    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        print(f" Encoded at: {meta.get('encoded_at', 'N/A')}  |  Device: {meta.get('device_used', 'N/A')}")

    if existing.shape[0] != len(texts):
        print(f"\n WARNING: Số chunks ({len(texts):,}) != số embeddings ({existing.shape[0]:,})")
        print(" Chạy lại: python 13_sbert_wiki.py --force")
    else:
        print("\n Bỏ qua encode. Dùng --force để encode lại khi data thay đổi.")

    sys.exit(0)

# LOAD SBERT MODEL
print("\n[2] LOAD SBERT MODEL")
print("")

from sentence_transformers import SentenceTransformer

model_name = 'keepitreal/vietnamese-sbert'

# Kiểm tra GPU trước khi load model
import torch
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f" GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
else:
    device = "cpu"
    print(" Không có GPU - dùng CPU")

print(f" Loading model: {model_name}")
print(" (Lần đầu sẽ tải ~500MB từ HuggingFace...)")
model = SentenceTransformer(model_name, device=device)
print(f" Model loaded OK  |  device: {device}")

# ENCODE
if force_encode:
    print(f"\n[3] ENCODE LẠI (--force) {len(texts):,} CHUNKS")
else:
    print(f"\n[3] CHƯA CÓ EMBEDDINGS - ENCODE {len(texts):,} CHUNKS")
print("")

# GPU: batch_size lớn hơn để tận dụng VRAM; CPU: nhỏ để tránh chậm
batch_size = 128 if device == "cuda" else 32
print(f" batch_size: {batch_size}  |  device: {device}")
print(" Đang encode... (có thể mất vài phút nếu dùng CPU)")

start_time = time.time()
embeddings = model.encode(
    texts,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    device=device,
)
elapsed = time.time() - start_time

print(f"\n Encode hoàn thành trong {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f" Embeddings shape: {embeddings.shape}")
print(f" dtype: {embeddings.dtype}")

# SAVE
print("\n[4] SAVE")
print("")

os.makedirs("data/embeddings", exist_ok=True)

np.save(emb_path, embeddings)
print(f" Saved: {emb_path}")

metadata = {
    "n_chunks": len(texts),
    "model": model_name,
    "dim": int(embeddings.shape[1]),
    "device_used": device,
    "encode_time_seconds": round(elapsed, 1),
    "encoded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "normalized": True,
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f" Saved: {meta_path}")

print(f"""

 ĐÃ HOÀN THÀNH SBERT WIKI EMBEDDINGS

 KẾT QUẢ:
   - Số chunks: {len(texts):,}
   - Embedding dim: {embeddings.shape[1]}
   - Encode time: {elapsed:.1f}s ({elapsed/60:.1f} min)
   - Device: {device}

 FILES ĐÃ TẠO:
   - data/embeddings/wiki_embeddings.npy
   - data/embeddings/wiki_metadata.json
""")
