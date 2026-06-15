"""
EXTRACT TOPICS FROM CSV
Trích xuất danh sách topic (title) từ database.csv để cào Wikipedia.
Output: data/raw/topics_to_crawl.json
Chạy trước 08_crawl_wiki.py nếu topics_to_crawl.json chưa có.
"""
import sys, codecs, json, os
import pandas as pd

if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("EXTRACT TOPICS FROM CSV")
print("")

# Load database
df = pd.read_csv('data/processed/database.csv')
print(f"Loaded database: {len(df):,} rows")

# Extract unique titles
topics = sorted(df['title'].dropna().unique().tolist())
print(f"Unique topics: {len(topics)}")

# Save
os.makedirs('data/raw', exist_ok=True)
output = {'topics': topics}
with open('data/raw/topics_to_crawl.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved: data/raw/topics_to_crawl.json")
print(f"First 5 topics: {topics[:5]}")
print("")
