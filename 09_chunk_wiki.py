"""
PHASE 2: CHUNKING WIKIPEDIA ARTICLES - Child-Parent Schema
- Child (~100 tu): don vi retrieval (BM25/SBERT index)
- Parent (doan van day du + 2 cau overlap moi ben): gui Ollama
- Output: data/processed/wiki_chunks.csv
  cols: child_id, child_text, child_word_count,
        parent_id, parent_text, parent_word_count,
        topic, wiki_title, source, entities
"""
import sys, codecs
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

import os, re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
ARTICLES_DIR    = "data/raw/wiki_articles"
OUTPUT_CSV      = "data/processed/wiki_chunks.csv"
MAX_CHILD_WORDS = 100   # child toi da (index & retrieval)
MIN_CHILD_WORDS = 30    # child toi thieu (ngan hon thi bo)
MIN_PARA_WORDS  = 50    # doan van ngan hon thi bo qua
MAX_PARENT_WORDS = 500  # parent toi da (Ollama context)
N_OVERLAP_SENTS  = 2    # so cau overlap moi ben cho parent


# ── HELPERS ───────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    """Tach cau tieng Viet (khong tach viet tat TP., TS., ...)"""
    sents = re.split(
        r'(?<=[.!?])\s+(?=[A-ZÁÀẢÃẠĂẮẶẴẨẤẦẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐĐ\"])',
        text
    )
    return [s.strip() for s in sents if s.strip()]


def build_parent_text(paragraphs: list, i: int,
                      n_overlap: int = N_OVERLAP_SENTS,
                      max_words: int = MAX_PARENT_WORDS) -> tuple:
    """
    Parent = [n cau cuoi para[i-1]] + [toan bo para[i]] + [n cau dau para[i+1]]
    Cap tai max_words: cat bot overlap truoc; neu core van vuot thi cat bot core theo cau.
    Tra ve (parent_text, effective_core) de split_into_children dung cung content voi parent.
    """
    core = paragraphs[i]
    core_words = len(core.split())

    # Lay overlap truoc (n cau cuoi doan truoc)
    prev_sents = []
    if i > 0 and n_overlap > 0:
        prev_sents = split_sentences(paragraphs[i - 1])[-n_overlap:]

    # Lay overlap sau (n cau dau doan sau)
    next_sents = []
    if i < len(paragraphs) - 1 and n_overlap > 0:
        next_sents = split_sentences(paragraphs[i + 1])[:n_overlap]

    # Kiem tra tong so tu - cat bot overlap neu vuot max
    prev_words = sum(len(s.split()) for s in prev_sents)
    next_words = sum(len(s.split()) for s in next_sents)

    # Neu tong vuot max_words thi giam overlap theo thu tu: next truoc, prev sau
    while core_words + prev_words + next_words > max_words and next_sents:
        removed = next_sents.pop()
        next_words -= len(removed.split())
    while core_words + prev_words + next_words > max_words and prev_sents:
        removed = prev_sents.pop(0)
        prev_words -= len(removed.split())

    # Fallback: khi ban than doan chinh vuot max_words, cat bot core theo cau
    # effective_core duoc dung cho ca parent va split_into_children (dam bao child nam trong parent)
    if core_words > max_words:
        core_sents = split_sentences(core)
        trimmed, tw = [], 0
        for s in core_sents:
            sw = len(s.split())
            if tw + sw > max_words and trimmed:
                break
            trimmed.append(s)
            tw += sw
        core = ' '.join(trimmed) if trimmed else core

    parts = []
    if prev_sents:
        parts.append(' '.join(prev_sents))
    parts.append(core)
    if next_sents:
        parts.append(' '.join(next_sents))

    return ' '.join(parts).strip(), core


def split_into_children(para_text: str,
                        max_words: int = MAX_CHILD_WORDS,
                        min_words: int = MIN_CHILD_WORDS) -> list:
    """
    Tach doan van thanh children ~max_words tu, khong overlap.
    Tach theo cau, tich luy den max_words thi luu va sang child moi.
    Tra ve list[str] - moi phan tu la 1 child.
    """
    words = para_text.split()
    if len(words) <= max_words:
        return [para_text] if len(words) >= min_words else []

    sentences = split_sentences(para_text)
    if len(sentences) <= 1:
        # Khong tach duoc theo cau -> giu nguyen (du co the dai)
        return [para_text]

    children = []
    current_sents = []
    current_words = 0

    for sent in sentences:
        sw = len(sent.split())
        if current_words + sw > max_words and current_sents:
            child_text = ' '.join(current_sents)
            if len(child_text.split()) >= min_words:
                children.append(child_text)
            current_sents = []
            current_words = 0
        current_sents.append(sent)
        current_words += sw

    # Phan con lai
    if current_sents:
        child_text = ' '.join(current_sents)
        if len(child_text.split()) >= min_words:
            children.append(child_text)

    return children if children else [para_text]


def parse_article_file(filepath: str) -> dict:
    """Doc file article, tach phan header va content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    meta = {}
    content_lines = []
    in_content = False

    for line in lines:
        if line.startswith("=" * 10):
            in_content = True
            continue
        if not in_content:
            if line.startswith("TOPIC:"):
                meta["topic"] = line.replace("TOPIC:", "").strip()
            elif line.startswith("WIKI_TITLE:"):
                meta["wiki_title"] = line.replace("WIKI_TITLE:", "").strip()
            elif line.startswith("SOURCE:"):
                meta["source"] = line.replace("SOURCE:", "").strip()
            elif line.startswith("LANG:"):
                meta["lang"] = line.replace("LANG:", "").strip()
            elif line.startswith("ENTITIES:"):
                meta["entities"] = line.replace("ENTITIES:", "").strip()
        else:
            content_lines.append(line)

    meta["content"] = ''.join(content_lines)
    return meta


def chunk_article(meta: dict) -> list:
    """
    Tao child-parent rows tu 1 article.
    Moi doan van la 1 parent (+ overlap), split thanh N children.
    """
    topic      = meta.get("topic", "")
    wiki_title = meta.get("wiki_title", "")
    source     = meta.get("source", "Wikipedia VI")
    entities   = meta.get("entities", "")
    content    = meta.get("content", "")
    title_slug = re.sub(r'[^A-Za-z0-9]', '_', wiki_title[:20])

    if not content.strip():
        return []

    # Tach thanh doan van theo \n\n, chuan hoa whitespace (loai newline, double-space trong doan)
    paragraphs = [re.sub(r'\s+', ' ', p.strip()) for p in re.split(r'\n{2,}', content) if p.strip()]

    rows = []
    para_idx = 0   # chi dem paragraph du dai (>= MIN_PARA_WORDS)

    for i, para in enumerate(paragraphs):
        if len(para.split()) < MIN_PARA_WORDS:
            continue

        # Parent: doan van day du + overlap 2 cau moi ben
        # effective_core: noi dung chinh sau khi cat (co the ngan hon para goc neu para qua dai)
        parent_text, effective_core = build_parent_text(paragraphs, i)
        parent_id   = f"WIKI_{title_slug}_p{para_idx:03d}"
        parent_wc   = len(parent_text.split())

        # Children: split tu effective_core (dam bao child_text la subset cua parent_text)
        children = split_into_children(effective_core)

        for c_idx, child_text in enumerate(children):
            child_id = f"WIKI_{title_slug}_p{para_idx:03d}_c{c_idx:02d}"
            rows.append({
                "child_id":          child_id,
                "child_text":        child_text,
                "child_word_count":  len(child_text.split()),
                "parent_id":         parent_id,
                "parent_text":       parent_text,
                "parent_word_count": parent_wc,
                "topic":             topic,
                "wiki_title":        wiki_title,
                "source":            source,
                "entities":          entities,
            })

        para_idx += 1

    return rows


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2: CHUNKING WIKIPEDIA - Child-Parent Schema")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  MAX_CHILD_WORDS  = {MAX_CHILD_WORDS}")
    print(f"  MIN_CHILD_WORDS  = {MIN_CHILD_WORDS}")
    print(f"  MIN_PARA_WORDS   = {MIN_PARA_WORDS}")
    print(f"  MAX_PARENT_WORDS = {MAX_PARENT_WORDS}")
    print(f"  N_OVERLAP_SENTS  = {N_OVERLAP_SENTS}")

    article_files = sorted(Path(ARTICLES_DIR).glob("*.txt"))
    print(f"\nSo files: {len(article_files)}")

    all_rows = []

    for filepath in tqdm(article_files, desc="Chunking", unit="file"):
        try:
            meta = parse_article_file(str(filepath))
            rows = chunk_article(meta)
            all_rows.extend(rows)
        except Exception as e:
            tqdm.write(f"   Error {filepath.name}: {e}")

    if not all_rows:
        print("Khong co du lieu!")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Dedup theo child_text (bai bi crao nhieu lan)
    before = len(df)
    df = df.drop_duplicates(subset=['child_text']).reset_index(drop=True)
    print(f"\n  Removed duplicate children: {before - len(df):,} (con {len(df):,})")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # ── Summary ───────────────────────────────────────────────────────────────
    n_children = len(df)
    n_parents  = df['parent_id'].nunique()
    avg_c = df['child_word_count'].mean()
    avg_p = df['parent_word_count'].mean()
    max_p = df['parent_word_count'].max()
    children_per_parent = df.groupby('parent_id').size()

    print()
    print("TONG KET:")
    print(f"  Files processed        : {len(article_files):,}")
    print(f"  Total children (rows)  : {n_children:,}")
    print(f"  Unique parents         : {n_parents:,}")
    print(f"  Avg children/parent    : {children_per_parent.mean():.1f}")
    print(f"  Max children/parent    : {children_per_parent.max()}")
    print(f"  Avg child_word_count   : {avg_c:.1f}")
    print(f"  Avg parent_word_count  : {avg_p:.1f}")
    print(f"  Max parent_word_count  : {max_p}")
    over500 = (df['parent_word_count'] > MAX_PARENT_WORDS).sum()
    print(f"  Parent > {MAX_PARENT_WORDS} tu (lo)    : {over500}")
    print(f"  Output: {OUTPUT_CSV}")

    # Preview
    print("\n--- PREVIEW 2 PARENTS DAU ---")
    for pid, grp in list(df.groupby('parent_id', sort=False))[:2]:
        print(f"\n[{pid}] {grp.iloc[0]['wiki_title']} | parent={grp.iloc[0]['parent_word_count']} tu")
        print(f"  Parent (60 ky tu dau): {grp.iloc[0]['parent_text'][:60]}...")
        for _, r in grp.iterrows():
            print(f"  Child [{r['child_id']}] {r['child_word_count']} tu: {r['child_text'][:60]}...")
