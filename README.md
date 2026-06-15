# Vietnamese Tourism Q&A Chatbot

A Vietnamese question-answering system built on **Retrieval-Augmented Generation**, retrieving from two knowledge bases and synthesizing answers with a locally hosted language model. Everything runs offline - no external API calls at serving time.

The centerpiece of this project is its **evaluation**: eight retrieval configurations compared over 1,112 test questions, with 20,360 question-passage pairs graded by an LLM judge and full statistical testing.

**Stack:** Python, Streamlit, Ollama, sentence-transformers, VnCoreNLP, scikit-learn, rank-bm25, scipy

---

## Results

Eight configurations: four retrieval methods, each run with and without cross-encoder reranking. Evaluated over 1,112 questions using P@3 and NDCG@5.

| Method | P@3 | NDCG@5 |
|---|---|---|
| **Hybrid-RRF + Rerank** | **0.595** | **0.742** |
| SBERT + Rerank | 0.576 | 0.723 |
| BM25 + Rerank | 0.561 | 0.709 |
| TF-IDF + Rerank | 0.519 | 0.671 |
| Hybrid-RRF | 0.415 | 0.601 |
| BM25 | 0.379 | 0.561 |
| Vietnamese SBERT | 0.368 | 0.548 |
| TF-IDF | 0.299 | 0.480 |

`P@3` counts relevant results among the top 3. `NDCG@5` measures the quality of the top 5, accounting for both relevance grade and rank position, on a 0 to 1 scale. Grades come from `gemma-4-26b-a4b-it` on a 0 to 2 scale.

Three findings:

**Cross-encoder reranking is the single largest factor.** All four methods gain between 0.141 and 0.191 NDCG@5 when reranking is added, every pair statistically significant. By contrast, the choice of base retrieval method leaves 2 of 3 comparisons without sufficient evidence of any difference.

**Weaker methods gain more from reranking.** TF-IDF gains 0.191 while Hybrid-RRF gains only 0.141. This follows from how reranking works: the cross-encoder can only reorder the 20 candidates the retrieval layer surfaces, so a method that already ranks well leaves less room to improve.

**The top position is not established.** `Hybrid-RRF + Rerank` leads `SBERT + Rerank` by 0.019 NDCG@5, but a Wilcoxon signed-rank test returns p = 0.2970, far above the 0.05 threshold. With 1,112 questions, that gap is not enough to rule out chance. The honest conclusion is that the two are equivalent.

Full methodology, the 28-pair Wilcoxon table and grading details are in [Evaluation Methodology](#evaluation-methodology).

---

## Engineering Highlights

**Deduplication before grading cut API calls by 54%.** Eight configurations run over the same question set, so many passages get returned by several methods at once. Keying on the (question, passage) pair reduced 44,480 result rows to 20,360 pairs actually needing a grade. Beyond the cost saving, this guarantees an identical passage receives an identical grade across every method - a prerequisite for fair comparison.

**Grading one pair per request instead of batching.** An initial design packed 20 passages into a single prompt to save calls, but measurement showed 13.6% of duplicate passages received different grades depending on which passages sat beside them in the prompt. Switching to one pair per request made each judgment independent, at the cost of more API calls.

**Disabling the judge model's reasoning mode cut per-call latency from 25 seconds to 1.3.** Gemma 4 writes a lengthy internal reasoning trace before answering. For a task whose entire output is a single digit, that trace is wasted work - and worse, it truncated responses by exhausting the token budget before the model reached its answer, failing 7 of every 10 calls. Setting `thinking_level = "minimal"` addressed the root cause rather than raising the token ceiling.

**Child-parent chunking for the Wikipedia store.** Small chunks serve as the indexing and matching unit because they are precise; the larger parent chunk containing them is what reaches the language model because it carries enough context. Separating these two roles keeps retrieval accurate without starving the answer of context.

**Handling rate limits across 20,360 graded pairs.** Each worker thread holds a dedicated API key rather than rotating through a shared pool, so one key hitting its limit never drags the others down. Failures back off exponentially, up to 7 attempts. The script skips rows that already carry a grade, making the job safely resumable after any interruption.

---

## Quick Start

```bash
git clone <repo-url> && cd THTT
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

The first run requires building indexes, which takes 15 to 30 minutes depending on GPU availability. See [Building the Indexes](#building-the-indexes) for all nine steps. Subsequent runs need only:

```bash
python run_web.py
```

Also required: Java 8 or later for VnCoreNLP, and at least 8GB of RAM. Details in [Installation](#installation).

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Text Preprocessing](#text-preprocessing)
- [Knowledge Base Organization](#knowledge-base-organization)
- [Retrieval Methods](#retrieval-methods)
- [Cross-Encoder Reranking](#cross-encoder-reranking)
- [Evaluation Methodology](#evaluation-methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Building the Indexes](#building-the-indexes)
- [Re-running the Evaluation](#re-running-the-evaluation)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Overview

Users ask questions about destinations, food, culture, or practical travel tips in Vietnam. The system handles each query in five stages:

1. **Query normalization** - expands common abbreviations, turning `TPHCM` into `hồ chí minh` and `ĐN` into `đà nẵng`
2. **Query classification** - assigns the question to one of four categories: destinations, food, culture, or travel tips
3. **Parallel retrieval** - searches both knowledge bases at once: a Q&A store of 13,360 question-answer pairs and a Vietnamese Wikipedia store of 7,979 passages
4. **Result fusion** - merges the two ranked lists using Reciprocal Rank Fusion
5. **Answer synthesis** - generates the final answer with an Ollama model running entirely on the local machine

```
User question
  => Normalize + Classify
  => [Q&A retrieval] + [Wikipedia retrieval]   run in parallel
  => RRF merge, fusing both lists by rank
  => Confidence threshold filter
  => Ollama synthesizes the answer
  => Display answer with sources, split into Q&A and Wikipedia tabs
```

---

## System Architecture

### Two Knowledge Bases

| Knowledge base | Content | Size |
|---|---|---|
| **Q&A Dataset** | Vietnamese tourism question-answer pairs from Kaggle | 13,360 pairs |
| **Vietnamese Wikipedia** | Articles on Vietnamese landmarks, culture and cuisine, collected through the Wikipedia API | 139 articles => 7,979 child chunks / 2,584 parent chunks |

### Six Indexes

| Index | Store | Location | Notes |
|---|---|---|---|
| TF-IDF | Q&A | `models/tfidf/` | 10,000-term vocabulary, 1-2 grams |
| TF-IDF | Wikipedia | `models/tfidf_wiki/` | 15,000-term vocabulary, 1-2 grams |
| BM25 | Q&A | `models/bm25/` | k1=1.5, b=0.75 |
| BM25 | Wikipedia | `models/bm25_wiki/` | k1=1.5, b=0.75 |
| SBERT | Q&A | `data/embeddings/sbert_embeddings.npy` | 13,360 x 768 matrix |
| SBERT | Wikipedia | `data/embeddings/wiki_embeddings.npy` | 7,979 x 768 matrix |

### Child-Parent Chunking for Wikipedia

Every Wikipedia paragraph is split into two layers:

- **Child chunks** of roughly 100 words serve as the retrieval unit. Being short, they match queries more precisely.
- **Parent chunks** of up to 500 words, carrying two overlapping sentences from each neighboring paragraph, are what the language model actually reads. Being longer, they carry enough context to support a complete answer.

This resolves the tension between **retrieval precision**, which favors short passages, and **contextual completeness**, which favors long ones.

### Confidence Threshold Filtering

Before assembling the context for the language model, the system checks the top score from each knowledge base. Thresholds are tuned per retrieval method and adjustable from the interface:

- Q&A falls below threshold => drop Q&A, use Wikipedia only
- Wikipedia falls below threshold => drop Wikipedia, use Q&A only
- Both fall below threshold => the model answers from its own knowledge, with a warning shown to the user

---

## Text Preprocessing

The preprocessing pipeline runs **identically** during offline indexing and online query handling, which keeps the vocabulary space of documents and queries perfectly aligned.

It applies four steps in sequence:

1. **Lowercasing** - normalizes the entire text to lower case
2. **Vietnamese word segmentation with VnCoreNLP** - correctly identifies multi-syllable compound words, so `Hà Nội` becomes the single token `Hà_Nội` rather than two separate tokens. VnCoreNLP runs as a Java subprocess with the `wseg` annotator and returns results in CoNLL format.
3. **Special character filtering** - keeps only tokens containing at least one alphanumeric character, discarding tokens made up purely of punctuation
4. **Stopword removal** - uses a hand-curated list of roughly 60 Vietnamese stopwords covering prepositions, pronouns, conjunctions and high-frequency verbs such as *của, và, là, có, trong, được*

The ordering matters: VnCoreNLP needs punctuation intact to determine sentence and word boundaries correctly. Stripping special characters before segmentation would degrade the quality of that step.

VnCoreNLP is a hard requirement with no substitute. Other segmentation libraries produce different word boundaries, which leaves queries misaligned with the vocabulary space used at indexing time and degrades retrieval quality.

SBERT bypasses this pipeline entirely. The model performs its own subword tokenization internally, so the steps above serve TF-IDF and BM25 only.

---

## Knowledge Base Organization

The system maintains **two independent knowledge bases**, each with its own set of indexes for every retrieval method.

### Store 1: Q&A Dataset

**Source:** A Vietnamese tourism dataset from Kaggle in SQuAD format.

**Content:** 13,360 question-answer pairs, each carrying the full source context and the character offset where the answer begins.

**Indexing unit:** The question. TF-IDF and BM25 index questions, and SBERT encodes questions as well.

**Answer post-processing:** Answers in the dataset are extractive and often only a few words long. The `expand_answer` function widens them into complete sentences by scanning backward from the answer offset to find the sentence start and forward to find the sentence end. The function recognizes and skips abbreviation periods such as TP., Dr. and TS. so that sentences are not cut at the wrong place.

### Store 2: Vietnamese Wikipedia

#### Stage 1 - Collection

**Topic source:** 139 unique topics pulled from the `title` column of the processed Q&A database. In the source dataset, `title` names the article each question-answer pair was extracted from, so it effectively serves as a topic label. Deriving topics this way guarantees the Wikipedia store covers exactly the subjects users actually ask about.

**Source priority**, all in Vietnamese:

1. **Vietnamese Wikipedia** - articles must have at least 300 words
2. **Vietnamese WikiVoyage** - at least 150 words, a lower bar because travel articles run shorter than encyclopedic ones

**Article lookup** proceeds in two stages:

- Stage one performs a direct title lookup, which yields the most accurate match. The crawler tries the original title first, then variants: dropping everything after a colon or opening parenthesis, and stripping prefixes such as "Du lịch" and "Khám phá".
- Stage two falls back to the search API when direct lookup finds nothing, taking the top three results.

Some topics have no corresponding Wikipedia article, so the crawler carries a synonym table. The topic "balo giá rẻ" is searched as "du lịch bụi", and "visa" is searched as "thị thực Việt Nam".

**Filtering irrelevant articles:** After fetching an article, the crawler extracts all of its internal Wikipedia links and runs them through a place-name recognition filter. An article is rejected when it both contains no links to tourism-related places and carries a title that is not itself a place name. This guards against storing articles about historical figures, organizations or events unrelated to tourism. Searching for "Sơn Trà" illustrates the point: if the returned article links only to personal names, it is a biography and gets discarded; if it links to "Bán đảo Sơn Trà" and "Đà Nẵng", it is a geographic article and gets kept.

**Content cleaning:** Trailing sections such as "Xem thêm", "Tham khảo" and "Liên kết ngoài" are cut. Reference markers like `[1]` and `[cần dẫn nguồn]` are stripped. Whitespace and blank lines are normalized.

**Output:** 139 text files, each opening with a header recording the topic, Wikipedia title, source, word count and entity list, followed by the article body as plain text. Across all 139 articles the crawler extracted 11,340 entity mentions in total, counted per article, so places referenced in several articles are counted more than once.

#### Stage 2 - Child-Parent Chunking

Each article is split into paragraphs on blank lines, and every paragraph produces one parent chunk along with several children:

```
Wikipedia article
  => split on blank lines => Paragraph
      => Parent chunk: full paragraph + 2 overlapping sentences per side, capped at 500 words
      => Child chunks: split by sentence, accumulating to roughly 100 words per chunk
```

**Limits and filtering criteria:**

| Parameter | Value | Purpose |
|---|---|---|
| `MIN_PARA_WORDS` | 50 words | Shorter paragraphs are skipped and produce no chunks |
| `MAX_CHILD_WORDS` | 100 words | Child ceiling, kept short for precise query matching |
| `MIN_CHILD_WORDS` | 30 words | Child floor, anything shorter is dropped |
| `MAX_PARENT_WORDS` | 500 words | Parent ceiling, enough context for the language model |
| `N_OVERLAP_SENTS` | 2 sentences | Overlap taken from each neighboring paragraph |

**Overlap mechanism:** A parent chunk concatenates the last two sentences of the preceding paragraph, the entire current paragraph, and the first two sentences of the following paragraph. The overlap preserves narrative flow when an idea spans a paragraph boundary. When the total exceeds 500 words, the system trims the overlap first, dropping trailing sentences before leading ones. When the core paragraph alone exceeds 500 words, it is truncated at a sentence boundary.

**Guaranteeing children fall inside parents:** Children are split from the same trimmed core used to build the parent, so every child text is always a substring of its parent. This matters for the interface, where the user sees the child chunk that matched their query and can expand it to read the parent containing it.

**Deduplication:** Once chunking completes, duplicate children are removed, which handles articles crawled more than once under different topics.

**Output:** `data/processed/wiki_chunks.csv`, holding **7,979 child chunks** across **2,584 parent chunks** drawn from 139 articles.

---

## Retrieval Methods

The system implements four methods, selectable directly from the interface. Each carries its own confidence threshold, described in the Confidence Threshold Filtering section above.

### 1. TF-IDF

Represents questions and documents as weighted keyword vectors. A term appearing often within one question but rarely across the corpus receives a higher weight, since it discriminates well. Relevance is measured by cosine similarity.

- Word segmentation uses VnCoreNLP, ensuring compounds like `Hà_Nội` and `Hội_An` are treated as single vocabulary units
- The vocabulary holds 10,000 terms for the Q&A store and 15,000 for Wikipedia, since encyclopedic prose uses a wider range of words than short questions do
- Uses 1-2 grams to capture both single words and adjacent word pairs
- Documents are tokenized ahead of fitting the vectorizer rather than embedding a custom tokenizer inside `TfidfVectorizer`. This keeps the pickled model safe to serialize and loadable from Streamlit without errors.

### 2. BM25

An improvement on TF-IDF adding two mechanisms. The first is term frequency saturation: a repeated term counts only up to a point, preventing keyword-stuffed documents from dominating. The second is length normalization: long documents gain no advantage over short ones.

Raw BM25 scores are unbounded, so the system applies min-max normalization to the 0-1 range before display and threshold comparison, keeping it consistent with TF-IDF and SBERT.

- Parameters k1 = 1.5 and b = 0.75, the standard values

### 3. Vietnamese SBERT

Rather than counting keywords, the `keepitreal/vietnamese-sbert` model maps each sentence to a 768-dimensional vector capturing its meaning. Two sentences worded differently but expressing the same idea still land close together in that space, something neither TF-IDF nor BM25 can achieve.

Embeddings for the entire corpus are precomputed and written to disk, so a new query only requires encoding one sentence and comparing it against the stored matrix.

- 768-dimensional vectors compared by cosine similarity
- Embeddings are pre-normalized, which reduces the comparison to a dot product and speeds up search

### 4. Hybrid-RRF

BM25 and SBERT run independently in parallel, each producing its own ranked list. The two lists are combined through Reciprocal Rank Fusion:

```
RRF(d) = sum( 1 / (k + rank_r(d)) )    with k = 60
```

The two score types cannot be added directly because BM25 scores and cosine similarities live on different scales. RRF sidesteps this by using **rank position only**, discarding absolute score values. A document ranking highly in both lists accumulates score from both sides and is therefore strongly favored.

- Fusion constant k = 60
- Each retriever contributes 60 candidates by default, adjustable from the interface

RRF is applied **twice** in the pipeline. The first pass fuses BM25 with SBERT to produce the best list within each knowledge base. The second pass fuses the Q&A list with the Wikipedia list into a single unified pool before it reaches the language model.

---

## Cross-Encoder Reranking

All four methods above are **retrieval**: they scan the entire corpus for candidates, which forces them to be fast and therefore to trade away accuracy. Reranking is a second stage that handles only a few dozen candidates, so it can afford a far heavier and more accurate model.

### Why a cross-encoder beats a bi-encoder

SBERT is a **bi-encoder**: it encodes the query into one vector, the document into another, then compares them by cosine similarity. The model never sees query and document together - it only compares two independently produced 768-dimensional summaries. That is the price of speed, since document embeddings can be computed ahead of time.

A **cross-encoder** feeds the query and document through a single forward pass as one pair, letting attention layers relate every part of the query to every part of the document directly. The result is substantially more accurate, but it must be recomputed for each pair, so it cannot scan a 13,360-document corpus. That is why it sits behind retrieval rather than replacing it.

### Configuration

| Parameter | Value |
|---|---|
| Model | `BAAI/bge-reranker-v2-m3` |
| Base architecture | XLM-RoBERTa, multilingual, supports Vietnamese |
| Pipeline position | After RRF merge and threshold filtering, before the Ollama call |
| Candidates in | 20, adjustable from 10 to 50 in the interface |
| Sources kept | Matches the count sent to the LLM, default 6 |
| Max pair length | 512 tokens |

The model needs no VnCoreNLP since it tokenizes at the subword level internally, like SBERT. Adding reranking leaves the TF-IDF and BM25 segmentation pipeline untouched.

### What gets scored

For Q&A results, the system pairs the question with its expanded answer, since both carry information that determines relevance. For Wikipedia results, it uses `child_text` rather than `parent_text`, because the child chunk is the passage that actually matched the query and is short enough to avoid the 512-token cutoff.

This feature is **off by default** in the interface, enabled by a sidebar checkbox. The first activation downloads the model, roughly 600MB.

Measured over the 1,112 test questions, reranking raises NDCG@5 by 0.141 to 0.191 depending on the base method. Details in [Evaluation Methodology](#evaluation-methodology).

---

## Evaluation Methodology

This section covers how the figures at the top of this README were produced: the test set, the grading procedure, the four metrics and the statistical testing.

### Test question set

1,112 questions, eight per topic across 139 topics. Each topic contributes four question types in equal proportion, two per type, so that no retrieval method enjoys an unfair advantage:

| Type | Characteristics | Targets |
|---|---|---|
| Direct keyword | Asks outright for figures, proper nouns, locations | BM25, TF-IDF |
| Paraphrase | Same intent, entirely different wording | SBERT |
| Indirect | Never names the topic, asks via its attributes | SBERT, Hybrid |
| Cross-topic | Compares two related topics | Hybrid |

The even split matters: if most questions were keyword-style, BM25 would gain an advantage that reflects the test set rather than the method.

### Grading procedure

Grading runs through `gemma-4-26b-a4b-it` on Google AI Studio using a three-point scale: 2 means the passage answers the question directly, 1 means related but not a direct answer, 0 means irrelevant.

Two design decisions bear directly on how trustworthy the numbers are:

- **One API call per pair.** Packing several passages into a single prompt lets a passage's grade drift with whatever sits beside it - measurement showed 13.6% of duplicate passages received different grades when batched 20 at a time. Grading each pair alone keeps every judgment independent.
- **Deduplication on the (question, passage) pair.** Eight configurations run over the same question set, so many passages surface repeatedly: 44,480 result rows collapse to 20,360 pairs needing a grade. Each pair is graded once and the grade is copied to every row containing it, which makes it impossible for the same passage to score 2 under one method and 0 under another.

Two parameters make the grading reproducible:

- `temperature = 0` removes randomness. Language models sample the next token by probability, so the same prompt can yield different answers. Setting it to zero forces the most likely choice every time, meaning an identical pair always receives an identical grade.
- `thinking_level = "minimal"` disables internal reasoning. Gemma 4 writes out a reasoning trace before answering by default, consuming thousands of hidden tokens. For a task that outputs a single digit it is wasted effort, and it truncated answers by exhausting the token budget before the model produced a grade. Disabling it cut each call from 20-30 seconds to about 1.3.

### Reranking impact

All four methods improve significantly with the cross-encoder, p < 0.0001 in every case:

| Base method | NDCG@5 without rerank | NDCG@5 with rerank | Delta |
|---|---|---|---|
| TF-IDF | 0.480 | 0.671 | +0.191 |
| SBERT | 0.548 | 0.723 | +0.175 |
| BM25 | 0.561 | 0.709 | +0.148 |
| Hybrid-RRF | 0.601 | 0.742 | +0.141 |

The largest gain lands on TF-IDF, the weakest method, and the smallest on Hybrid-RRF, the strongest. That is expected: the cross-encoder can only reorder the 20 candidates retrieval surfaces, so a method that already ranks well has less room to improve.

This delta cannot be attributed to the cross-encoder alone, because the two variants differ in two ways at once. The non-rerank variant pulls 5 candidates per store and merges them by RRF into 5 final results. The rerank variant pulls 20 per store, merges by RRF, then has the cross-encoder rescore everything and pick 5. So the figure measures both changes together: a candidate pool four times wider, and the reranking itself. Isolating the cross-encoder's contribution would require a third variant that pulls 20 candidates but selects 5 by RRF order, which this study did not run.

### The four metrics

**P@3** counts how many of the top 3 results are relevant, meaning graded 1 or 2. It measures quantity, not ordering: a correct result at position 1 counts the same as one at position 3.

**NDCG@5** measures the quality of the top 5, accounting for both relevance grade and rank position. Results that are both more relevant and closer to the top score higher, because each position's contribution is divided by the logarithm of its rank:

```
DCG@5 = sum of ( grade_i / log2(i + 1) )   for i from 1 to 5
NDCG@5 = DCG@5 / ideal DCG
```

Dividing by the ideal DCG - the best ordering achievable with those same results - keeps NDCG between 0 and 1 and makes it comparable across questions of differing difficulty. This is the primary metric used to rank the eight configurations.

**95% bootstrap confidence intervals** answer a different question: would this number hold up on a different set of 1,112 questions? The procedure resamples the 1,112 questions with replacement to build a synthetic set of the same size, recomputes NDCG@5 on it, repeats 2,000 times, then takes the middle 95% of the resulting values.

| Method | NDCG@5 | 95% CI | Width |
|---|---|---|---|
| Hybrid-RRF + Rerank | 0.742 | [0.721 - 0.762] | 0.041 |
| SBERT + Rerank | 0.723 | [0.701 - 0.746] | 0.045 |
| BM25 + Rerank | 0.709 | [0.685 - 0.731] | 0.046 |
| TF-IDF + Rerank | 0.671 | [0.647 - 0.695] | 0.048 |
| Hybrid-RRF | 0.601 | [0.579 - 0.623] | 0.044 |
| BM25 | 0.561 | [0.538 - 0.585] | 0.047 |
| Vietnamese SBERT | 0.548 | [0.525 - 0.569] | 0.044 |
| TF-IDF | 0.480 | [0.457 - 0.503] | 0.046 |

Confidence intervals are **not** the tool for ranking methods - Wilcoxon does that more rigorously. They serve two other purposes:

- **They show how stable each figure is.** Every interval spans roughly 0.04 to 0.05, meaning that with 1,112 questions, a measured NDCG@5 fluctuates within that band. Gaps narrower than 0.04 between two methods should not be read as real differences from the means alone.
- **They flag when two methods sit too close.** `Hybrid-RRF + Rerank` at [0.721 - 0.762] and `SBERT + Rerank` at [0.701 - 0.746] overlap across 0.721 to 0.746, a warning sign that the difference may not hold. Wilcoxon later confirms exactly that with p = 0.2970.

The converse does not hold: overlapping intervals do **not** imply no difference. `BM25 + Rerank` at [0.685 - 0.731] overlaps `Hybrid-RRF + Rerank` at [0.721 - 0.762], yet Wilcoxon returns p = 0.0031, a real difference. Wilcoxon compares question by question and is therefore more sensitive, while confidence intervals look only at each method's mean in isolation.

**Wilcoxon signed-rank** tests whether the gap between two methods could be chance. It pairs by `query_id`, comparing both methods on the **same question** before aggregating, which removes the influence of some questions being harder than others. It ranks the differences rather than using their raw magnitudes, so it makes no normality assumption - appropriate for NDCG scores, which are skewed and bounded to [0, 1].

### Wilcoxon across all 28 pairs

All eight configurations go into a **single test family**, not two groups of four. Every pair is compared: C(8,2) = 28 pairs, one Wilcoxon test each over n = 1,112 questions.

Running 28 tests at once means the sheer number of attempts will produce a few results that look significant by chance alone. **Holm-Bonferroni correction** handles this by sorting p-values ascending and tightening the threshold by position. The pair ranked `i` among 28 must clear:

```
threshold_i = 0.05 / (28 - i)      with i counting from 0
```

The smallest p-value must beat 0.05/28 = 0.00179; the last only needs to beat 0.05. Correcting across all 28 rather than within smaller groups is the stricter choice.

The table below is sorted by p-value ascending, matching the order Holm-Bonferroni applies thresholds. Delta is the NDCG@5 difference between the first and second method named in each pair:

| # | Comparison | Delta NDCG@5 | p-value | HB threshold | Result |
|---|---|---|---|---|---|
| 1 | TF-IDF vs Hybrid-RRF+Rerank | -0.262 | < 0.0001 | 0.00179 | significant |
| 2 | TF-IDF vs BM25+Rerank | -0.229 | < 0.0001 | 0.00185 | significant |
| 3 | TF-IDF vs SBERT+Rerank | -0.244 | < 0.0001 | 0.00192 | significant |
| 4 | TF-IDF vs TF-IDF+Rerank | -0.191 | < 0.0001 | 0.00200 | significant |
| 5 | SBERT vs Hybrid-RRF+Rerank | -0.194 | < 0.0001 | 0.00208 | significant |
| 6 | SBERT vs SBERT+Rerank | -0.175 | < 0.0001 | 0.00217 | significant |
| 7 | BM25 vs Hybrid-RRF+Rerank | -0.181 | < 0.0001 | 0.00227 | significant |
| 8 | SBERT vs BM25+Rerank | -0.161 | < 0.0001 | 0.00238 | significant |
| 9 | Hybrid-RRF vs Hybrid-RRF+Rerank | -0.141 | < 0.0001 | 0.00250 | significant |
| 10 | BM25 vs BM25+Rerank | -0.148 | < 0.0001 | 0.00263 | significant |
| 11 | BM25 vs SBERT+Rerank | -0.162 | < 0.0001 | 0.00278 | significant |
| 12 | Hybrid-RRF vs SBERT+Rerank | -0.123 | < 0.0001 | 0.00294 | significant |
| 13 | Hybrid-RRF vs BM25+Rerank | -0.108 | < 0.0001 | 0.00313 | significant |
| 14 | SBERT vs TF-IDF+Rerank | -0.123 | < 0.0001 | 0.00333 | significant |
| 15 | BM25 vs TF-IDF+Rerank | -0.110 | < 0.0001 | 0.00357 | significant |
| 16 | TF-IDF vs Hybrid-RRF | -0.121 | < 0.0001 | 0.00385 | significant |
| 17 | TF-IDF vs BM25 | -0.081 | < 0.0001 | 0.00417 | significant |
| 18 | Hybrid-RRF vs TF-IDF+Rerank | -0.070 | < 0.0001 | 0.00455 | significant |
| 19 | SBERT vs Hybrid-RRF | -0.052 | < 0.0001 | 0.00500 | significant |
| 20 | TF-IDF+Rerank vs Hybrid-RRF+Rerank | -0.071 | < 0.0001 | 0.00556 | significant |
| 21 | TF-IDF vs SBERT | -0.068 | < 0.0001 | 0.00625 | significant |
| 22 | TF-IDF+Rerank vs SBERT+Rerank | -0.053 | < 0.0001 | 0.00714 | significant |
| 23 | TF-IDF+Rerank vs BM25+Rerank | -0.038 | < 0.0001 | 0.00833 | significant |
| 24 | BM25 vs Hybrid-RRF | -0.039 | < 0.0001 | 0.01000 | significant |
| 25 | BM25+Rerank vs Hybrid-RRF+Rerank | -0.033 | 0.0031 | 0.01250 | significant |
| 26 | BM25+Rerank vs SBERT+Rerank | -0.015 | 0.1015 | 0.01667 | insufficient |
| 27 | BM25 vs SBERT | +0.013 | 0.1241 | 0.02500 | insufficient |
| 28 | SBERT+Rerank vs Hybrid-RRF+Rerank | -0.019 | 0.2970 | 0.05000 | insufficient |

**25 pairs significant, 3 without sufficient evidence.**

All three inconclusive pairs sit near the top of the ranking, where methods are already strong and separated by little:

- `BM25` vs `SBERT`, a gap of 0.013, p = 0.1241. Two fundamentally different approaches - one counting keywords, the other modeling meaning - yet equivalent on this question set.
- `BM25+Rerank` vs `SBERT+Rerank`, a gap of 0.015, p = 0.1015. Once both pass through the same cross-encoder, retrieval-layer differences flatten further.
- `SBERT+Rerank` vs `Hybrid-RRF+Rerank`, a gap of 0.019, p = 0.2970. The most consequential of the three, since it concerns the leading method.

`Hybrid-RRF + Rerank` tops the table but leads `SBERT + Rerank` by only 0.019 NDCG@5, with substantially overlapping confidence intervals of [0.721 - 0.762] against [0.701 - 0.746]. Two independent signals point the same way: **there is not enough evidence to claim `Hybrid-RRF + Rerank` genuinely outperforms `SBERT + Rerank`**, even though it edges ahead on the mean.

Conversely, every comparison between a non-reranked and a reranked method is significant, at p < 0.0001 or p = 0.0031. The takeaway: **adding a cross-encoder produces a far more reliable difference than choosing which retrieval method to build on.**

Aggregated metrics are stored in [`eval_metrics_v2.csv`](eval_metrics_v2.csv).

---

## Installation

### Requirements

- Python 3.10 or later
- Java, required to run VnCoreNLP
- [Ollama](https://ollama.com) installed and running locally
- At least 8GB of RAM. The SBERT model occupies roughly 500MB, Q&A embeddings about 40MB and Wikipedia embeddings about 24MB.
- A GPU is optional. The system detects CUDA automatically and uses it when present; without one it falls back to CPU, which is only slower during the initial SBERT encoding.

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd THTT
pip install -r requirements.txt
```

### 2. Install VnCoreNLP

Download VnCoreNLP from https://github.com/vncorenlp/VnCoreNLP and place the following in the project root:

- The `VnCoreNLP-1.2.jar` file
- The contents of VnCoreNLP's `models` directory, namely `wordsegmenter`, `postagger`, `ner` and `dep`, placed directly inside the project's own `models` directory

The resulting layout:

```
THTT/
├── VnCoreNLP-1.2.jar
└── models/
    ├── wordsegmenter/     # VnCoreNLP wseg annotator, the only one this system uses
    ├── postagger/         # VnCoreNLP
    ├── ner/               # VnCoreNLP
    ├── dep/               # VnCoreNLP
    ├── tfidf/             # Project index, generated by the pipeline
    ├── tfidf_wiki/
    ├── bm25/
    └── bm25_wiki/
```

The `models` directory is shared between VnCoreNLP's models and the project's indexes. `VnCoreNLPProcessor` looks for its models at the project root's `models` directory, while the retrievers look for indexes under `models/tfidf`, `models/bm25` and so on.

### 3. Install Ollama and pull a model

```bash
ollama pull qwen2.5:3b      # lightweight, suits modest hardware
ollama pull qwen2.5:7b      # more detailed answers, needs more RAM
```

### 4. Prepare data and indexes

The `data` and `models` directories are excluded from the repository. You have two options:

- Download the source dataset into `data/raw` and run the index-building pipeline described below
- Or obtain the prebuilt `.pkl`, `.npy` and `.csv` files directly from the team

---

## Usage

### Launching the app

```bash
python run_web.py
```

Or run Streamlit directly:

```bash
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501

`run_web.py` verifies all 13 required data and index files before starting, reporting exactly which files are missing and which script generates them.

### Interface

**The sidebar** controls:

- Retrieval method: TF-IDF, BM25, Vietnamese SBERT or Hybrid
- Number of results displayed, from 1 to 10
- Separate confidence thresholds for the Q&A and Wikipedia stores
- Candidates per retriever before fusion, shown only when Hybrid is selected
- Cross-encoder reranking toggle, with the candidate count fed into the reranker
- Ollama model: `qwen2.5:3b` for speed or `qwen2.5:7b` for more detail
- Number of sources fed into the model's context, from 2 to 10, defaulting to 6

**The main area** displays, in order:

- The question input and search button
- The synthesized answer from Ollama, badged with the model, retrieval method, question category and processing time
- A Q&A Dataset tab listing results from the Q&A store, expandable to reveal the full source context
- A Wikipedia tab listing the child chunks that matched, expandable to reveal the parent chunk containing each one

**Sample questions:**

```
Hà Nội có những địa điểm du lịch nổi tiếng nào?
Đặc sản Hội An là gì?
Visa đến Việt Nam cần những gì?
Vịnh Hạ Long ở đâu và có gì đặc biệt?
```

---

## Building the Indexes

Run the following scripts in order. They only need to run once; the results persist and are reused on every subsequent launch.

### Q&A store

```bash
python 03_prepare_database.py   # Clean the CSV, emit questions/answers/contexts.pkl
python 04_tfidf_method.py       # Build the TF-IDF index into models/tfidf/
python 05_bm25_method.py        # Build the BM25 index into models/bm25/
python 06_sbert_method.py       # Encode SBERT into data/embeddings/sbert_embeddings.npy
```

### Wikipedia store

```bash
python 08_crawl_wiki.py         # Collect 139 Vietnamese Wikipedia and WikiVoyage articles
python 09_chunk_wiki.py         # Chunk into child-parent pairs at data/processed/wiki_chunks.csv
python 10_index_wiki.py         # Build the BM25 wiki index into models/bm25_wiki/
python 12_tfidf_wiki.py         # Build the TF-IDF wiki index into models/tfidf_wiki/
python 13_sbert_wiki.py         # Encode SBERT into data/embeddings/wiki_embeddings.npy
```

A few notes:

- Steps 10 and 12 write to separate directories, so they can run in parallel
- `08_crawl_wiki.py` requires internet access and sleeps one second between requests to respect the Wikipedia API rate limit
- Both SBERT scripts use the GPU automatically when CUDA is available, with a batch size of 128 on GPU and 32 on CPU
- Both SBERT scripts skip encoding when the embeddings file already exists. Pass `--force` to re-encode after the source data changes, for example `python 13_sbert_wiki.py --force`

---

## Re-running the Evaluation

Three steps, with the first run twice for the two groups of configurations:

```bash
# Step 1: retrieval. Run twice, once without reranking and once with
python eval_run_v2.py --out eval_results_v3.csv
python eval_run_v2.py --rerank --out eval_results_rerank.csv

# Step 2: grading. Pass both files so duplicates collapse before any API call
python grade.py --csv eval_results_v3.csv eval_results_rerank.csv

# Step 3: metrics across all eight configurations
python eval_score_v2.py --csv eval_results_v3.csv eval_results_rerank.csv
```

The four retrieval methods run in two variants each, giving eight configurations. The non-rerank variant sets `RETRIEVE_K = 5`; the `+Rerank` variant sets `RETRIEVE_K = 20` and lets the cross-encoder pick 5. All eight return exactly 5 results, so P@3 and NDCG@5 compare directly.

Both `eval_run_v2.py` and `grade.py` support interruption and resumption. `eval_run_v2.py` reads back the CSV and skips questions already processed; `grade.py` skips rows that already carry a grade of 0, 1 or 2 and only fills the blanks, so re-running the same command picks up any pairs that failed.

The grading step must receive **both files in one command**. Grading them separately loses the cross-file deduplication: the 5,329 pairs present in both files would be sent to the API twice, wasting quota and risking the same passage receiving two different grades.

`eval_results_v3.csv` and `eval_results_rerank.csv` are intermediate artifacts of 22,240 rows each, excluded by `.gitignore` and regenerated by running the pipeline. Only `eval_metrics_v2.csv` holding the aggregated metrics is committed.

Both `grade.py` and `generate_questions.py` read `GEMINI_API_KEYS` from a `keys.py` file in the project root:

```python
# keys.py
GEMINI_API_KEYS = ["AIzaSy...", "AIzaSy..."]   # multiple keys pool their quota
```

`keys.py` is gitignored so keys never reach the repository. `generate_questions.py` only needs running when rebuilding the test set, which already ships in `topics_bank.py`.

Both scripts run in parallel through `ThreadPoolExecutor`, defaulting to one thread per key with each thread bound to a dedicated key, so no key ever carries more than one in-flight request. Adjust with `--workers`, though exceeding the key count is counterproductive since two threads sharing a key will push it past its own rate limit. On a 429 or 5xx the script backs off exponentially and retries, up to 7 attempts capped at 60 seconds in `grade.py`.

---

## Project Structure

```
THTT/
├── app/
│   └── streamlit_app.py          # Web app containing the full RAG pipeline
│
├── src/
│   ├── retrieval/
│   │   ├── tfidf_retriever.py    # TFIDFRetriever class
│   │   ├── bm25_retriever.py     # BM25Retriever class
│   │   ├── sbert_retriever.py    # SBERTRetriever class
│   │   └── cross_encoder_reranker.py  # CrossEncoderReranker class
│   ├── preprocessing/
│   │   └── vncorenlp_processor.py  # Invokes VnCoreNLP via Java subprocess
│   └── utils/
│       └── vietnamese_stopwords.py # About 60 Vietnamese stopwords
│
├── data/                          # Excluded from the repository
│   ├── raw/
│   │   ├── vietnam_tourism_all.csv       # Source Q&A dataset from Kaggle
│   │   ├── topics_to_crawl.json          # The 139 topics to crawl
│   │   ├── wiki_articles/                # 139 Wikipedia articles as .txt
│   │   └── wiki_manifest.json            # Crawl manifest, 139 articles with 11,340 entity mentions
│   ├── processed/
│   │   ├── questions.pkl / answers.pkl / contexts.pkl / titles.pkl / ...
│   │   └── wiki_chunks.csv               # 7,979 child chunks, 2,584 parent chunks
│   └── embeddings/
│       ├── sbert_embeddings.npy          # 13,360 x 768
│       └── wiki_embeddings.npy           # 7,979 x 768
│
├── models/                        # Excluded from the repository
│   ├── tfidf/                     # TF-IDF index, Q&A
│   ├── tfidf_wiki/                # TF-IDF index, Wikipedia
│   ├── bm25/                      # BM25 index, Q&A
│   ├── bm25_wiki/                 # BM25 index, Wikipedia
│   ├── wordsegmenter/             # VnCoreNLP word segmentation model
│   ├── postagger/                 # VnCoreNLP
│   ├── ner/                       # VnCoreNLP
│   └── dep/                       # VnCoreNLP
│
├── results/
│   └── database_metadata.json     # Q&A database statistics
│
├── VnCoreNLP-1.2.jar              # Excluded from the repository, download separately
│
├── 03_prepare_database.py         # Prepare the Q&A database
├── 04_tfidf_method.py             # Build TF-IDF, Q&A
├── 05_bm25_method.py              # Build BM25, Q&A
├── 06_sbert_method.py             # Encode SBERT, Q&A
├── 08_crawl_wiki.py               # Crawl Wikipedia
├── 09_chunk_wiki.py               # Child-parent chunking
├── 10_index_wiki.py               # Build BM25, Wikipedia
├── 12_tfidf_wiki.py               # Build TF-IDF, Wikipedia
├── 13_sbert_wiki.py               # Encode SBERT, Wikipedia
│
├── eval_run_v2.py                 # Retrieval runs, --rerank flag for the rerank variants
├── grade.py                       # Automated grading, requires an API key
├── eval_score_v2.py               # Compute P@3 and NDCG@5
├── eval_metrics_v2.csv            # Aggregated metrics, the only committed CSV
│
├── topics_bank.py                 # The 1,112 test questions
├── generate_questions.py          # Generate questions via API, requires an API key
├── merge_questions.py             # Merge question sets
├── extract_topics_from_csv.py     # Extract topics from the dataset into topics_to_crawl.json
│
├── run_web.py                     # Entry point for launching the app
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| Web UI | Streamlit | User interface |
| Local LLM | Ollama running Qwen2.5 | Synthesizes answers from retrieved context |
| Vietnamese segmentation | VnCoreNLP 1.2 on Java | Tokenization for TF-IDF and BM25 |
| TF-IDF | scikit-learn TfidfVectorizer | Index construction and cosine similarity |
| BM25 | rank-bm25, BM25Okapi class | Index construction and BM25 scoring |
| Dense retrieval | sentence-transformers with `keepitreal/vietnamese-sbert` | Encodes sentences into 768-dimensional vectors |
| Vector operations | NumPy | Dot products and embedding normalization |
| Data handling | Pandas | CSV parsing and wiki chunk management |
| Deep learning backend | PyTorch | Underlying framework for sentence-transformers |
| Reranking | sentence-transformers CrossEncoder with `BAAI/bge-reranker-v2-m3` | Rescores candidates by true relevance |
| Statistical testing | SciPy | Wilcoxon signed-rank and bootstrap resampling |
| Automated evaluation | Google AI Studio, `gemma-4-26b-a4b-it` | LLM-as-judge grading of retrieval results |
