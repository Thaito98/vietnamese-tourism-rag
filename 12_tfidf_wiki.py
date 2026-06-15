"""
PHẦN 4.1: TF-IDF INDEX CHO WIKI CHUNKS
Nhất quán với 04_tfidf_method.py:
  1. Pre-tokenize với VnCoreNLP + loại dấu câu + loại stopwords
  2. Fit TfidfVectorizer KHÔNG có custom tokenizer (pickle an toàn)
  3. Lưu models/tfidf_wiki/
"""

import sys
import os
import pickle
import re
import time
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocessing.vncorenlp_processor import VnCoreNLPProcessor

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("")
print("PHẦN 4.1: TF-IDF WIKI INDEX (VnCoreNLP + pickle-safe)")
print("")

# LOAD WIKI CHUNKS
print("[1] LOAD WIKI CHUNKS")
print("")

df = pd.read_csv("data/processed/wiki_chunks.csv")
texts = df["child_text"].fillna("").tolist()
print(f" Loaded {len(texts):,} chunks")
print(f" Columns: {list(df.columns)}")

# PRE-TOKENIZE VỚI VnCoreNLP
print("\n[2] PRE-TOKENIZE VỚI VnCoreNLP + loại dấu câu + stopwords")
print("")

vncore = VnCoreNLPProcessor()
print(" VnCoreNLP processor khởi tạo thành công")
print(f" Tokenizing {len(texts):,} chunks...")

start_time = time.time()
tokenized_strings = []

for i, text in enumerate(tqdm(texts, desc="  Tokenizing", unit="chunk")):
    tokens = vncore.tokenize(re.sub(r'\s+', ' ', str(text).lower().strip()), remove_stopwords_flag=True)
    tokenized_strings.append(' '.join(tokens))
    if (i + 1) % 200 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(texts) - (i + 1)) / rate / 60
        print(f"  [{i+1:,}/{len(texts):,}] ETA: {eta:.1f} min")

elapsed_tokenize = time.time() - start_time
print(f"\n Tokenization hoàn thành trong {elapsed_tokenize:.1f}s ({elapsed_tokenize/60:.1f} min)")
print(f" Sample: {tokenized_strings[0][:80]}...")

# FIT TF-IDF VECTORIZER
print("\n[3] FIT TF-IDF MODEL")
print("")

start_time = time.time()
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    lowercase=False,
    token_pattern=r'(?u)\b\w+\b',
)
doc_vectors = vectorizer.fit_transform(tokenized_strings)
fit_time = time.time() - start_time

print(f" Vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f" doc_vectors shape: {doc_vectors.shape}")
print(f" Fit completed in {fit_time:.2f}s")

# SAVE
print("\n[4] SAVE")
print("")

os.makedirs("models/tfidf_wiki", exist_ok=True)

with open("models/tfidf_wiki/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f, protocol=4)
print(" Saved: models/tfidf_wiki/vectorizer.pkl")

with open("models/tfidf_wiki/doc_vectors.pkl", "wb") as f:
    pickle.dump(doc_vectors, f, protocol=4)
print(" Saved: models/tfidf_wiki/doc_vectors.pkl")

print(f"""

 ĐÃ HOÀN THÀNH TF-IDF WIKI INDEX

 MÔ HÌNH:
   - Tokenizer: VnCoreNLP + loại dấu câu + stopwords
   - max_features: 15.000  |  ngram_range: (1,2)
   - Tokenize time: {elapsed_tokenize:.1f}s ({elapsed_tokenize/60:.1f} min)
   - Fit time: {fit_time:.2f}s

 FILES ĐÃ TẠO:
   - models/tfidf_wiki/vectorizer.pkl
   - models/tfidf_wiki/doc_vectors.pkl
""")
