"""
merge_questions.py - Gộp generated_questions.json vào topics_bank.py.

Cách dùng:
    python merge_questions.py
    python merge_questions.py --in generated_questions.json --preview
"""

import io, sys, os, json, re, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--in",      dest="src", default="generated_questions.json")
parser.add_argument("--preview", action="store_true", help="Chỉ in ra, không ghi file")
args = parser.parse_args()

SRC_PATH = os.path.join(ROOT, args.src)
TB_PATH  = os.path.join(ROOT, "topics_bank.py")

# Đọc câu đã sinh
with open(SRC_PATH, encoding="utf-8") as f:
    new_qs = json.load(f)

# Gán id từ 1 (tạo lại hoàn toàn, sắp xếp theo topic rồi label)
label_order = {"K1":0,"K2":1,"P1":2,"P2":3,"A1":4,"A2":5,"C1":6,"C2":7}
from topics_bank import TOPICS
topic_order = {t: i for i, t in enumerate(TOPICS)}

sorted_qs = sorted(
    new_qs,
    key=lambda x: (topic_order.get(x.get("topic",""), 9999),
                   label_order.get(x.get("label",""), 9))
)

seen = set()
to_add = []
for idx, item in enumerate(sorted_qs, start=1):
    q = str(item.get("question", "")).strip()
    if not q or q.lower() in seen:
        continue
    seen.add(q.lower())
    to_add.append({"id": idx, "question": q})

print(f"Câu mới trong file : {len(new_qs)}")
print(f"Sau khi lọc trùng  : {len(to_add)}")
print(f"ID range           : 1 => {len(to_add)}")

if args.preview:
    print("\nPreview 10 câu đầu:")
    for q in to_add[:10]:
        print(f"  {q['id']}: {q['question']}")
    sys.exit(0)

# Đọc nội dung file topics_bank.py
with open(TB_PATH, encoding="utf-8") as f:
    content = f.read()

# Tìm vị trí cuối của QUESTIONS_BANK (dòng cuối trước dấu `]`)
# Thêm vào trước dấu `]` cuối cùng
lines_to_add = "\n".join(
    f'    {{"id": {q["id"]:>4}, "question": {json.dumps(q["question"], ensure_ascii=False)}}},'
    for q in to_add
)

# Thay thế chính xác block QUESTIONS_BANK = [...] bằng regex
new_block = f"QUESTIONS_BANK = [\n{lines_to_add}\n]"
new_content = re.sub(
    r"QUESTIONS_BANK\s*=\s*\[.*?\]",
    new_block,
    content,
    count=1,
    flags=re.DOTALL,
)

if new_content == content:
    print("\nLỖI: Không tìm được QUESTIONS_BANK trong topics_bank.py")
    sys.exit(1)

with open(TB_PATH, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"\nĐã ghi {len(to_add)} câu vào {TB_PATH}")
print("Kiểm tra: python -c \"from topics_bank import QUESTIONS_BANK; print(len(QUESTIONS_BANK))\"")
