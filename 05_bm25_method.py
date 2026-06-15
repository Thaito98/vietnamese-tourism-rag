"""
PHẦN 2.2: BM25 RETRIEVER
Xây dựng BM25 index từ 13,360 Q&A

Cách tiếp cận:
  1. Pre-tokenize toàn bộ docs bằng VnCoreNLP + stopwords -> list of list tokens
     - Bao gồm: lowercase, loại bỏ dấu câu/ký tự đặc biệt, loại bỏ stopwords
  2. Build BM25Okapi trực tiếp (không dùng BM25Retriever class)
"""

import sys
import os
import pickle
import re
import time
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing.vncorenlp_processor import VnCoreNLPProcessor

# Set UTF-8 encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("")
print("PHẦN 2.2: BM25 RETRIEVER (VnCoreNLP + BM25Okapi trực tiếp)")
print("")

# LOAD DATABASE
print("\n[1] LOAD DATABASE")
print("")

with open('data/processed/questions.pkl', 'rb') as f:
    questions = pickle.load(f)

print(f" Loaded {len(questions):,} questions")

# PRE-TOKENIZE VỚI VnCoreNLP (list of list tokens -> cho BM25Okapi)
print("\n[2] PRE-TOKENIZE VỚI VnCoreNLP + STOPWORDS")
print("")

vncore = VnCoreNLPProcessor()
print(" VnCoreNLP processor khởi tạo thành công")
print(" Tokenizing 13,360 documents")

start_time = time.time()
tokenized_docs = []

for i, doc in enumerate(tqdm(questions, desc="  Tokenizing", unit="doc")):
    tokens = vncore.tokenize(re.sub(r'\s+', ' ', doc.lower().strip()), remove_stopwords_flag=True)
    tokenized_docs.append(tokens)
    if (i + 1) % 500 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(questions) - (i + 1)) / rate / 60
        print(f"  [{i+1:,}/{len(questions):,}] ETA: {eta:.1f} min")

elapsed_tokenize = time.time() - start_time
print(f"\n Tokenization hoàn thành trong {elapsed_tokenize:.1f}s ({elapsed_tokenize/60:.1f} min)")

# BUILD BM25 INDEX
print("\n[3] BUILD BM25 INDEX")
print("")

start_time = time.time()
bm25_model = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
fit_time = time.time() - start_time

print(f" k1=1.5, b=0.75")
print(f" Build completed in {fit_time:.2f}s")

# SAVE MODEL
print("\n[4] SAVE MODEL")
print("")

os.makedirs('models/bm25', exist_ok=True)

with open('models/bm25/bm25_model.pkl', 'wb') as f:
    pickle.dump(bm25_model, f)
with open('models/bm25/tokenized_docs.pkl', 'wb') as f:
    pickle.dump(tokenized_docs, f)

params = {'k1': 1.5, 'b': 0.75}
with open('models/bm25/params.pkl', 'wb') as f:
    pickle.dump(params, f)

print(" Saved: models/bm25/bm25_model.pkl")
print(" Saved: models/bm25/tokenized_docs.pkl")
print(" Saved: models/bm25/params.pkl")

print(f"""
 ĐÃ HOÀN THÀNH BM25 METHOD

 MÔ HÌNH:
   - Tokenizer: VnCoreNLP + stopwords
   - k1=1.5, b=0.75
   - Tokenize time: {elapsed_tokenize:.1f}s ({elapsed_tokenize/60:.1f} min)
   - Build time: {fit_time:.2f}s

 FILES ĐÃ TẠO:
   - models/bm25/bm25_model.pkl
   - models/bm25/tokenized_docs.pkl
   - models/bm25/params.pkl
""")
