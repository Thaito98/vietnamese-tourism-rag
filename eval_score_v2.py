"""
eval_score_v2.py - Tính P@3 và NDCG@5 từ kết quả đánh giá.

Nhận nhiều file CSV để so sánh các lần chạy khác nhau. Wilcoxon ghép theo
query_id nên các phương pháp ở những file khác nhau vẫn so được với nhau,
miễn là cùng chạy trên một bộ câu hỏi.

Cách dùng:
    # Chỉ 4 phương pháp gốc
    python eval_score_v2.py

    # Gộp cả 4 gốc và 4 biến thể rerank để so sánh
    python eval_score_v2.py --csv eval_results_v3.csv eval_results_rerank.csv
"""

import io, sys, os, csv, math, argparse
from itertools import combinations
import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_METHODS = [
    "TF-IDF",
    "BM25",
    "SBERT",
    "Hybrid-RRF",
]
# Mỗi phương pháp nền có 2 biến thể: không rerank và có cross-encoder rerank
METHODS = BASE_METHODS + [m + "+Rerank" for m in BASE_METHODS]
SOURCES = ["rrf"]

# Công thức 

def precision_at_k(grades, k=3):
    return sum(1 for g in grades[:k] if g >= 1) / k

def dcg_at_k(grades, k=5):
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades[:k]))

def ndcg_at_k(grades, k=5):
    ideal = sorted(grades[:k], reverse=True)
    idcg  = dcg_at_k(ideal, k)
    return dcg_at_k(grades, k) / idcg if idcg > 0 else 0.0

# Đọc CSV 

parser = argparse.ArgumentParser()
parser.add_argument("--csv", nargs="+", default=["eval_results_v3.csv"],
                    help="Một hoặc nhiều file CSV kết quả. Nhiều file sẽ được gộp.")
parser.add_argument("--out", default="eval_metrics_v2.csv",
                    help="File CSV để ghi metrics tổng hợp")
args = parser.parse_args()

csv_paths = []
for c in args.csv:
    p = os.path.join(ROOT, c)
    if not os.path.exists(p):
        print(f"Không tìm thấy: {p}")
        print("Chạy eval_run_v2.py trước.")
        sys.exit(1)
    csv_paths.append(p)

# data[source][method][qid] = {rank: grade}
data = {src: {m: {} for m in METHODS} for src in SOURCES}
all_qids = set()

for csv_path in csv_paths:
    n_rows = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qid       = int(row["query_id"])
            method    = row["method"]
            source    = row["source"]
            rank      = int(row["rank"])
            grade_raw = row.get("grade", "").strip()

            try:
                grade = int(grade_raw)
            except (ValueError, TypeError):
                grade = -1

            if source not in data or method not in data[source]:
                continue
            if qid not in data[source][method]:
                data[source][method][qid] = {}
            data[source][method][qid][rank] = grade
            all_qids.add(qid)
            n_rows += 1
    print(f"Đọc {n_rows:,} dòng từ {os.path.basename(csv_path)}")

# Chỉ giữ phương pháp thực sự có dữ liệu - cho phép chạy khi mới có 1 trong 2 file
METHODS = [m for m in METHODS if any(data[src][m] for src in SOURCES)]
if not METHODS:
    print("Không có phương pháp nào có dữ liệu.")
    sys.exit(1)
print(f"Phương pháp có dữ liệu: {', '.join(METHODS)}\n")

# Kiểm tra chưa chấm
ungraded = set()
for src in SOURCES:
    for m in METHODS:
        for qid, ranks in data[src][m].items():
            if any(g == -1 for g in ranks.values()):
                ungraded.add(qid)

if ungraded:
    print(f"CẢNH BÁO: {len(ungraded)} câu hỏi chưa chấm xong: {sorted(ungraded)}")
    print("Các câu chưa chấm sẽ bị bỏ qua.\n")

# Tính metrics 

def get_grades(source, method, qid):
    ranks_dict = data[source][method].get(qid, {})
    grades = [ranks_dict.get(r, 0) for r in range(1, 6)]
    if any(g == -1 for g in grades):
        return None
    return grades

# results[source][method] = {"p3": [...], "ndcg5": [...], "qids": [...]}
results = {src: {m: {"p3": [], "ndcg5": [], "qids": []} for m in METHODS} for src in SOURCES}

for qid in sorted(all_qids):
    for src in SOURCES:
        for method in METHODS:
            grades = get_grades(src, method, qid)
            if grades is None:
                continue
            results[src][method]["p3"].append(precision_at_k(grades, 3))
            results[src][method]["ndcg5"].append(ndcg_at_k(grades, 5))
            results[src][method]["qids"].append(qid)

# Bảng: Source × Method 

def avg(lst): return sum(lst)/len(lst) if lst else 0.0

# Bảng dọc: mỗi phương pháp một dòng. Dễ đọc hơn bảng ngang khi có nhiều
# phương pháp, và sắp xếp được theo NDCG@5 để thấy ngay thứ hạng.
W = 62

print("=" * W)
print("TRUNG BÌNH P@3 / NDCG@5 THEO METHOD")
print("=" * W)

best_combo = ("", "", -1)
for src in SOURCES:
    rows_tbl = []
    for m in METHODS:
        n5 = avg(results[src][m]["ndcg5"])
        p3 = avg(results[src][m]["p3"])
        n  = len(results[src][m]["ndcg5"])
        rows_tbl.append((m, p3, n5, n))
        if n5 > best_combo[2]:
            best_combo = (src, m, n5)

    rows_tbl.sort(key=lambda r: r[2], reverse=True)

    print(f"\nSource: {src}")
    print("-" * W)
    print(f"  {'Phương pháp':<26} {'P@3':>8} {'NDCG@5':>9} {'n':>8}")
    print("-" * W)
    for m, p3, n5, n in rows_tbl:
        print(f"  {m:<26} {p3:>8.3f} {n5:>9.3f} {n:>8}")
    print("-" * W)

print("=" * W)
print()

# Tóm tắt 

n_graded = len(results["rrf"][METHODS[0]]["qids"])
src_b, meth_b, ndcg_b = best_combo
print(f"Số câu đã chấm : {n_graded}")
print(f"Tốt nhất       : {meth_b}  (source={src_b}, NDCG@5={ndcg_b:.3f})")
print()

# Ghi eval_metrics_v2.csv 

out_path = os.path.join(ROOT, args.out)
with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    hdr_cols = ["source"] + [f"{m}_P@3" for m in METHODS] + [f"{m}_NDCG@5" for m in METHODS]
    writer.writerow(hdr_cols)
    for src in SOURCES:
        row = [src]
        row += [f"{avg(results[src][m]['p3']):.3f}"   for m in METHODS]
        row += [f"{avg(results[src][m]['ndcg5']):.3f}" for m in METHODS]
        writer.writerow(row)

print(f"Đã lưu -> {out_path}")

# Kiểm định thống kê 

SOURCE_STAT = "rrf"

ndcg_per_q = {}
for m in METHODS:
    r = results[SOURCE_STAT][m]
    ndcg_per_q[m] = dict(zip(r["qids"], r["ndcg5"]))

common_qids = sorted(set.intersection(*[set(d.keys()) for d in ndcg_per_q.values()]))
n_q = len(common_qids)
ndcg_arr = {m: np.array([ndcg_per_q[m][q] for q in common_qids]) for m in METHODS}

print()
print(f"KIỂM ĐỊNH THỐNG KÊ  (source = RRF, NDCG@5, n = {n_q} câu)")
print()

# 1. Khoảng tin cậy 95% (bootstrap) 
print()
print("1. Khoảng tin cậy 95% cho NDCG@5 (bootstrap, 2 000 lần)")
print()
print(f"  {'Phương pháp':<24} {'NDCG@5 TB':>10}   {'CI 95%':>18}")
print()

np.random.seed(42)
N_BOOT = 2000
for m in METHODS:
    arr = ndcg_arr[m]
    boots = np.array([np.mean(np.random.choice(arr, size=n_q, replace=True))
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {m:<24} {arr.mean():>10.3f}   [{lo:.3f} - {hi:.3f}]")

print()
print("  CI không giao nhau -> khác biệt có ý nghĩa thực tế.")
print("  CI giao nhau       -> không thể khẳng định chắc chắn.")

# 2. Wilcoxon signed-rank test (Holm-Bonferroni) 
print()
print("2. Wilcoxon Signed-Rank Test - tất cả cặp (Holm-Bonferroni correction)")
print()

if not HAS_SCIPY:
    print("  [!] scipy chưa cài. Chạy:  pip install scipy")
else:
    pair_results = []
    for m1, m2 in combinations(METHODS, 2):
        a1, a2 = ndcg_arr[m1], ndcg_arr[m2]
        delta = float(a1.mean() - a2.mean())
        p = 1.0 if np.allclose(a1, a2) else float(
            stats.wilcoxon(a1, a2, alternative="two-sided", zero_method="wilcox")[1])
        pair_results.append((m1, m2, delta, p))

    pair_results.sort(key=lambda x: x[3])
    n_pairs = len(pair_results)

    print(f"  {'Cặp so sánh':<40} {'delta NDCG@5':>9} {'p-value':>9}  {'Kết quả':>14}")
    print()
    sig_flags = []
    for i, (m1, m2, delta, p) in enumerate(pair_results):
        sig = p < 0.05 / (n_pairs - i)
        sig_flags.append(sig)
        label = f"{m1} vs {m2}"
        print(f"  {label:<40} {delta:>+9.3f} {p:>9.4f}  {' có ý nghĩa' if sig else '- không đủ':>14}")

    print()
    print(f"  Tổng {n_pairs} cặp - có ý nghĩa: {sum(sig_flags)} | không đủ: {n_pairs-sum(sig_flags)}")
    print("  (alpha = 0.05, Holm-Bonferroni)")

# 3. So sánh với phương pháp tốt nhất 
print()
print("3. So sánh với phương pháp tốt nhất (RRF)")
print()

best_m = max(METHODS, key=lambda m: ndcg_arr[m].mean())
print(f"  Tốt nhất: {best_m}  (NDCG@5 = {ndcg_arr[best_m].mean():.3f})")
print()
print(f"  {'Phương pháp':<24} {'delta':>10} {'p-value':>10}  {'Kết quả':>14}")
print()

if HAS_SCIPY:
    for m in METHODS:
        if m == best_m:
            continue
        delta = float(ndcg_arr[m].mean() - ndcg_arr[best_m].mean())
        p = 1.0 if np.allclose(ndcg_arr[m], ndcg_arr[best_m]) else float(
            stats.wilcoxon(ndcg_arr[m], ndcg_arr[best_m],
                           alternative="two-sided", zero_method="wilcox")[1])
        sig = " thua rõ rệt" if (p < 0.05 and delta < 0) else "- chưa rõ"
        print(f"  {m:<24} {delta:>+10.3f} {p:>10.4f}  {sig:>14}")
else:
    print("Cần scipy.")

print()

# 4. Tác động của reranking - so từng cặp cùng phương pháp nền
# So sánh cả pipeline: bản gốc lấy top-5 trực tiếp, bản +Rerank lấy 20 ứng viên
# rồi cross-encoder chọn ra 5. Delta phản ánh lợi ích của việc mở rộng tập ứng
# viên kết hợp với reranking, đúng như cách hệ thống chạy thực tế.
_rr_pairs = [(m, m + "+Rerank") for m in BASE_METHODS
             if m in ndcg_arr and (m + "+Rerank") in ndcg_arr]

if _rr_pairs:
    print("4. Tác động của cross-encoder reranking")
    print("   (bản gốc: top-5 trực tiếp | bản +Rerank: 20 ứng viên -> rerank -> 5)")
    print()
    print(f"  {'Phương pháp nền':<20} {'NDCG@5 gốc':>11} {'+Rerank':>9} "
          f"{'delta':>8} {'p-value':>9}  {'Kết quả':>14}")
    print()

    for base, rr in _rr_pairs:
        a_base, a_rr = ndcg_arr[base], ndcg_arr[rr]
        m_base, m_rr = a_base.mean(), a_rr.mean()
        delta = float(m_rr - m_base)

        if not HAS_SCIPY:
            print(f"  {base:<20} {m_base:>11.3f} {m_rr:>9.3f} {delta:>+8.3f} "
                  f"{'(cần scipy)':>9}")
            continue

        p = 1.0 if np.allclose(a_base, a_rr) else float(
            stats.wilcoxon(a_rr, a_base, alternative="two-sided",
                           zero_method="wilcox")[1])
        if p < 0.05:
            verdict = " rerank tốt hơn" if delta > 0 else " rerank kém hơn"
        else:
            verdict = "- chưa rõ"
        print(f"  {base:<20} {m_base:>11.3f} {m_rr:>9.3f} {delta:>+8.3f} "
              f"{p:>9.4f}  {verdict:>14}")

    print()
    print("  Mỗi dòng so hai biến thể của cùng một phương pháp truy hồi trên cùng")
    print("  bộ câu hỏi, ghép theo query_id. Delta gồm cả lợi ích của việc mở rộng")
    print("  tập ứng viên từ 5 lên 20 lẫn lợi ích của cross-encoder.")
    print()

