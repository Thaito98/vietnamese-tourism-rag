"""
Script để chạy Streamlit web app
"""

import subprocess
import sys
import os

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("")
print(" KHỞI ĐỘNG STREAMLIT WEB APP")
print("")

# Check if streamlit is installed
try:
    import streamlit
    print(" Streamlit đã cài đặt")
except ImportError:
    print(" Streamlit chưa cài đặt!")
    print(" Đang cài đặt streamlit...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "streamlit"])
    print(" Đã cài đặt streamlit")

# Check required files
print("\n Kiểm tra files cần thiết...")
required_files = [
    # Kho Q&A
    'data/processed/questions.pkl',
    'data/processed/answers.pkl',
    'data/processed/ids.pkl',
    'data/processed/contexts.pkl',
    'data/processed/titles.pkl',
    'data/processed/answer_starts.pkl',
    'models/tfidf/vectorizer.pkl',
    'models/bm25/bm25_model.pkl',
    'data/embeddings/sbert_embeddings.npy',
    # Kho Wikipedia (app luôn truy hồi cả hai kho)
    'data/processed/wiki_chunks.csv',
    'models/tfidf_wiki/vectorizer.pkl',
    'models/bm25_wiki/bm25_model.pkl',
    'data/embeddings/wiki_embeddings.npy',
]

missing = []
for f in required_files:
    if os.path.exists(f):
        print(f"   {f}")
    else:
        print(f"   {f} (missing)")
        missing.append(f)

if missing:
    print(f"\n Thiếu {len(missing)} files:")
    for f in missing:
        print(f"  - {f}")
    print("\n Giải pháp - chạy pipeline xây dựng chỉ mục theo thứ tự:")
    print("  1. python 03_prepare_database.py   # Chuẩn bị database Q&A")
    print("  2. python 04_tfidf_method.py       # Xây TF-IDF index")
    print("  3. python 05_bm25_method.py        # Xây BM25 index")
    print("  4. python 06_sbert_method.py       # Encode SBERT embeddings")
    print("\n  Nếu thiếu index Wikipedia, chạy tiếp:")
    print("  5. python 08_crawl_wiki.py         # Cào Wikipedia (cần internet)")
    print("  6. python 09_chunk_wiki.py         # Phân đoạn child-parent")
    print("  7. python 10_index_wiki.py         # Xây BM25 wiki index")
    print("  8. python 12_tfidf_wiki.py         # Xây TF-IDF wiki index")
    print("  9. python 13_sbert_wiki.py         # Encode SBERT wiki embeddings")
    sys.exit(1)

print("\n Tất cả files đều sẵn sàng!")

# Run streamlit
print("")
print(" ĐANG KHỞI ĐỘNG WEB APP...")
print("")
print("\n Web app sẽ mở tại: http://localhost:8501")
print(" Nhấn Ctrl+C để dừng server\n")

# Run streamlit
subprocess.run([
    sys.executable, "-m", "streamlit", "run", 
    "app/streamlit_app.py",
    "--server.port=8501",
    "--server.headless=false"
])
