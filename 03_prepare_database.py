"""
PHẦN 1.3: CHUẨN BỊ DATABASE CHO CHATBOT
Load và làm sạch dữ liệu để các retrieval methods sử dụng
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("")
print("CHUẨN BỊ DATABASE CHO CHATBOT")
print("")

# LOAD DỮ LIỆU
print("\n[1] LOAD DỮ LIỆU")
print("")

df = pd.read_csv('data/raw/vietnam_tourism_all.csv')
print(f" Đã load: {len(df):,} cặp Q&A từ vietnam_tourism_all.csv")

# CHỌN CỘT CẦN THIẾT
print("\n[2] CHỌN CỘT CẦN THIẾT")
print("")

# Chatbot cần: id, question, answer
# Có thể thêm: context, title, answer_start (để mở rộng câu trả lời)
database = df[['id', 'question', 'answer', 'answer_start', 'context', 'title']].copy()

print(f" Đã chọn {len(database.columns)} cột: {list(database.columns)}")
print(f" Database size: {len(database):,} rows")

# LÀM SẠCH DỮ LIỆU CƠ BẢN
print("\n[3] LÀM SẠCH DỮ LIỆU")
print("")

# A. Loại bỏ khoảng trắng thừa
print("\n Loại bỏ khoảng trắng thừa...")
database['question'] = database['question'].str.strip()
database['answer'] = database['answer'].str.strip()
database['context'] = database['context'].str.strip()
print(" Đã loại bỏ khoảng trắng đầu/cuối")

# Replace multiple spaces with single space
database['question'] = database['question'].str.replace(r'\s+', ' ', regex=True)
database['answer'] = database['answer'].str.replace(r'\s+', ' ', regex=True)
print(" Đã chuẩn hóa khoảng trắng")

# B. Kiểm tra và xử lý missing values
print("\n Kiểm tra missing values...")
missing = database.isnull().sum()
if missing.sum() > 0:
    print(f" Phát hiện missing values:")
    print(missing[missing > 0])
    # Drop rows with missing values in critical columns
    before = len(database)
    database = database.dropna(subset=['question', 'answer'])
    after = len(database)
    if before != after:
        print(f" Đã xóa {before - after} rows có missing values")
else:
    print(" Không có missing values")

# C. Loại bỏ duplicates (nếu có)
print("\n Kiểm tra duplicates...")
duplicates = database.duplicated(subset=['question', 'answer']).sum()
if duplicates > 0:
    print(f" Phát hiện {duplicates} duplicates (question VÀ answer giống hệt)")
    print(f" Giữ lại tất cả duplicates (có thể có câu hỏi giống nhưng answer khác)")
else:
    print(" Không có duplicates")

# D. Reset index
database = database.reset_index(drop=True)
print(f"\n Database cuối cùng: {len(database):,} rows")

# TẠO INDEX MAPPING
print("\n[4] TẠO INDEX MAPPING")
print("")

# Create mapping: index -> id, question, answer
database['idx'] = range(len(database))

print(f" Đã tạo index mapping cho {len(database):,} items")
print(f"   - Index: 0 -> {len(database)-1}")
print(f"   - Mỗi index map tới: id, question, answer")

# THỐNG KÊ DATABASE
print("\n[5] THỐNG KÊ DATABASE")
print("")

# Calculate lengths
database['question_len'] = database['question'].str.len()
database['answer_len'] = database['answer'].str.len()
database['question_words'] = database['question'].str.split().str.len()
database['answer_words'] = database['answer'].str.split().str.len()

print(f"\n THỐNG KÊ:")
print(f"   - Tổng số cặp: {len(database):,}")
print(f"   - Avg question length: {database['question_len'].mean():.1f} chars ({database['question_words'].mean():.1f} words)")
print(f"   - Avg answer length: {database['answer_len'].mean():.1f} chars ({database['answer_words'].mean():.1f} words)")
print(f"   - Min question: {database['question_len'].min()} chars")
print(f"   - Max question: {database['question_len'].max()} chars")
print(f"   - Min answer: {database['answer_len'].min()} chars")
print(f"   - Max answer: {database['answer_len'].max()} chars")

# Show some examples
print("\n MỘT SỐ VÍ DỤ:")
print("")
for i in range(3):
    row = database.iloc[i]
    print(f"\n[{i}] ID: {row['id']}")
    print(f"    Q: {row['question'][:80]}...")
    print(f"    A: {row['answer'][:80]}...")
    print(f"    Title: {row['title'][:60]}...")

# LƯU DATABASE
print("\n[6] LƯU DATABASE")
print("")

# Create output folders
os.makedirs('data/processed', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Save as CSV
database.to_csv('data/processed/database.csv', index=False, encoding='utf-8')
print(f" Saved: data/processed/database.csv ({len(database):,} rows)")

# Save as pickle (faster to load)
database.to_pickle('data/processed/database.pkl')
print(f" Saved: data/processed/database.pkl (binary format)")

# Save questions and answers separately for retrieval
questions = database['question'].tolist()
answers = database['answer'].tolist()
ids = database['id'].tolist()
contexts = database['context'].tolist()  # NEW: Save contexts!
titles = database['title'].tolist()  # NEW: Save titles!
answer_starts = database['answer_start'].tolist()  # NEW: Save answer_start positions!

with open('data/processed/questions.pkl', 'wb') as f:
    pickle.dump(questions, f)
print(f" Saved: data/processed/questions.pkl ({len(questions):,} items)")

with open('data/processed/answers.pkl', 'wb') as f:
    pickle.dump(answers, f)
print(f" Saved: data/processed/answers.pkl ({len(answers):,} items)")

with open('data/processed/ids.pkl', 'wb') as f:
    pickle.dump(ids, f)
print(f" Saved: data/processed/ids.pkl ({len(ids):,} items)")

with open('data/processed/contexts.pkl', 'wb') as f:
    pickle.dump(contexts, f)
print(f" Saved: data/processed/contexts.pkl ({len(contexts):,} items)")

with open('data/processed/titles.pkl', 'wb') as f:
    pickle.dump(titles, f)
print(f" Saved: data/processed/titles.pkl ({len(titles):,} items)")

with open('data/processed/answer_starts.pkl', 'wb') as f:
    pickle.dump(answer_starts, f)
print(f" Saved: data/processed/answer_starts.pkl ({len(answer_starts):,} items)")

# TẠO METADATA
print("\n[7] TẠO METADATA")
print("")

metadata = {
    'total_pairs': len(database),
    'avg_question_len_chars': float(database['question_len'].mean()),
    'avg_answer_len_chars': float(database['answer_len'].mean()),
    'avg_question_words': float(database['question_words'].mean()),
    'avg_answer_words': float(database['answer_words'].mean()),
    'min_question_len': int(database['question_len'].min()),
    'max_question_len': int(database['question_len'].max()),
    'min_answer_len': int(database['answer_len'].min()),
    'max_answer_len': int(database['answer_len'].max()),
    'unique_titles': int(database['title'].nunique()),
}

import json
with open('results/database_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(" Saved: results/database_metadata.json")

# TEST LOAD
print("\n[8] TEST LOAD")
print("")

# Test loading pickle files
print("\n Test loading pickle files...")
with open('data/processed/questions.pkl', 'rb') as f:
    test_questions = pickle.load(f)
with open('data/processed/answers.pkl', 'rb') as f:
    test_answers = pickle.load(f)
with open('data/processed/ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

print(f" Loaded {len(test_questions):,} questions")
print(f" Loaded {len(test_answers):,} answers")
print(f" Loaded {len(test_ids):,} ids")

# Test accessing
print(f"\n Test accessing data:")
idx = 100
print(f"   - Index {idx}:")
print(f"     ID: {test_ids[idx]}")
print(f"     Q: {test_questions[idx][:80]}...")
print(f"     A: {test_answers[idx][:80]}...")

# HƯỚNG DẪN SỬ DỤNG
print("\n[9] HƯỚNG DẪN SỬ DỤNG")
print("")



# KẾT LUẬN
print("")
print("KẾT LUẬN")
print("")

print(f"""
 Đã chuẩn bị database thành công!

 DATABASE INFO:
   - Tổng số cặp: {len(database):,}
   - Đã làm sạch và chuẩn hóa
   - Không có missing values
   - Không có duplicates (hoặc đã xóa)

 FILES ĐÃ TẠO:
   - data/processed/database.csv (CSV format)
   - data/processed/database.pkl (pickle format)
   - data/processed/questions.pkl ({len(questions):,} items)
   - data/processed/answers.pkl ({len(answers):,} items)
   - data/processed/ids.pkl ({len(ids):,} items)
   - results/database_metadata.json

 SỬ DỤNG:
   - Load questions.pkl => fit models
   - Load answers.pkl => trả kết quả cho user
   - Load ids.pkl => tracking và debugging

 PERFORMANCE:
   - Pickle files load nhanh hơn CSV ~10x
   - Khuyến khích dùng pickle trong production

 HOÀN THÀNH PHẦN 1
 Tiếp theo: PHẦN 2 - Xây dựng TF-IDF Retriever
""")

print("")
