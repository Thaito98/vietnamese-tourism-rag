"""
PHẦN 2.1: TF-IDF RETRIEVER
Xây dựng TF-IDF index từ 13,360 Q&A

Cách tiếp cận (pickle-safe):
  1. Pre-tokenize toàn bộ docs bằng VnCoreNLP + stopwords -> list string
     - Bao gồm: lowercase, loại bỏ dấu câu/ký tự đặc biệt, loại bỏ stopwords
  2. Fit TfidfVectorizer KHÔNG có custom tokenizer -> pickle an toàn
"""

import sys
import os
import pickle
import re
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing.vncorenlp_processor import VnCoreNLPProcessor

# Set UTF-8 encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("")
print("PHẦN 2.1: TF-IDF RETRIEVER (VnCoreNLP + pickle-safe)")
print("")

# LOAD DATABASE
print("\n[1] LOAD DATABASE")
print("")

with open('data/processed/questions.pkl', 'rb') as f:
    questions = pickle.load(f)

print(f" Loaded {len(questions):,} questions")

# PRE-TOKENIZE VỚI VnCoreNLP (tách ra khỏi vectorizer -> pickle an toàn)
print("\n[2] PRE-TOKENIZE VỚI VnCoreNLP + STOPWORDS")
print("")

vncore = VnCoreNLPProcessor()
print(" VnCoreNLP processor khởi tạo thành công")
print(" Tokenizing 13,360 documents")

start_time = time.time()
tokenized_strings = []

for i, doc in enumerate(tqdm(questions, desc="  Tokenizing", unit="doc")):
    tokens = vncore.tokenize(re.sub(r'\s+', ' ', doc.lower().strip()), remove_stopwords_flag=True)
    tokenized_strings.append(' '.join(tokens))
    if (i + 1) % 500 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(questions) - (i + 1)) / rate / 60
        print(f"  [{i+1:,}/{len(questions):,}] ETA: {eta:.1f} min")

elapsed_tokenize = time.time() - start_time
print(f"\n Tokenization hoàn thành trong {elapsed_tokenize:.1f}s ({elapsed_tokenize/60:.1f} min)")

# FIT TF-IDF VECTORIZER (không nhúng custom tokenizer -> pickle an toàn)
print("\n[3] FIT TF-IDF MODEL")
print("")

start_time = time.time()
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    lowercase=False,              # đã lowercase trong bước tokenize
    token_pattern=r'(?u)\b\w+\b', # default sklearn, tách theo khoảng trắng
)
doc_vectors = vectorizer.fit_transform(tokenized_strings)
fit_time = time.time() - start_time

print(f" Vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f" doc_vectors shape: {doc_vectors.shape}")
print(f" Fit completed in {fit_time:.2f}s")

# SAVE MODEL
print("\n[4] SAVE MODEL")
print("")

os.makedirs('models/tfidf', exist_ok=True)
with open('models/tfidf/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('models/tfidf/doc_vectors.pkl', 'wb') as f:
    pickle.dump(doc_vectors, f)

print(" Saved: models/tfidf/vectorizer.pkl  (không có custom tokenizer, load được từ Streamlit)")
print(" Saved: models/tfidf/doc_vectors.pkl")

print(f"""
 ĐÃ HOÀN THÀNH TF-IDF METHOD

 MÔ HÌNH:
   - Tokenizer: VnCoreNLP + stopwords (pre-tokenized, pickle-safe)
   - max_features: 10,000  |  ngram_range: (1,2)
   - Tokenize time: {elapsed_tokenize:.1f}s ({elapsed_tokenize/60:.1f} min)
   - Fit time: {fit_time:.2f}s

 FILES ĐÃ TẠO:
   - models/tfidf/vectorizer.pkl
   - models/tfidf/doc_vectors.pkl
""")
