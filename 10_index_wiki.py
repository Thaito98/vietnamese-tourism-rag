"""
PHASE 3: BUILD BM25 INDEX CHO WIKI CHUNKS
- Dùng VnCoreNLP + stopwords (nhất quán với BM25 gốc)
- Lưu: models/bm25_wiki/
"""
import sys, codecs
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

import os, pickle, re, time
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

print("")
print("PHASE 3: BUILD BM25 INDEX - WIKI CHUNKS")
print("")

# Load chunks
print("\n1. Loading wiki chunks...")
df = pd.read_csv("data/processed/wiki_chunks.csv")
print(f"   Loaded {len(df):,} chunks")
print(f"   Columns: {df.columns.tolist()}")

# Tokenize với VnCoreNLP + stopwords
print("\n2. Tokenizing chunks với VnCoreNLP + stopwords...")
print("   (Nhất quán với BM25 gốc)")

from src.preprocessing.vncorenlp_processor import VnCoreNLPProcessor
vncore = VnCoreNLPProcessor()
print("    VnCoreNLP ready")

start = time.time()
tokenized_chunks = []
texts = df['child_text'].tolist()

for text in tqdm(texts, desc="   Tokenizing", unit="chunk"):
    tokens = vncore.tokenize(re.sub(r'\s+', ' ', str(text).lower().strip()), remove_stopwords_flag=True)
    tokenized_chunks.append(tokens)

elapsed = time.time() - start
print(f"    Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Build BM25 index
print("\n3. Building BM25 index...")
start = time.time()
bm25_wiki = BM25Okapi(tokenized_chunks, k1=1.5, b=0.75)
print(f"    Done in {time.time()-start:.1f}s")

# Save
print("\n4. Saving BM25 wiki index...")
os.makedirs("models/bm25_wiki", exist_ok=True)

with open("models/bm25_wiki/bm25_model.pkl", "wb") as f:
    pickle.dump(bm25_wiki, f)
with open("models/bm25_wiki/tokenized_chunks.pkl", "wb") as f:
    pickle.dump(tokenized_chunks, f)
with open("models/bm25_wiki/params.pkl", "wb") as f:
    pickle.dump({"k1": 1.5, "b": 0.75, "n_chunks": len(texts)}, f)

print("    Saved models/bm25_wiki/")


