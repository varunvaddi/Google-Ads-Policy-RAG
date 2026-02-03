---
title: Google Ads Policy RAG
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# Google Ads Policy RAG System

A **production-grade Retrieval-Augmented Generation (RAG) system** for interpreting, enforcing, and evaluating **Google Ads policies** with citations, confidence scoring, and human-in-the-loop escalation.

---

## 🎯 Project Goal

Build an AI-powered assistant that can:

* Answer questions about Google Ads policies
* Review ad text for policy compliance
* Cite the **exact policy sections** used in decisions
* Quantify confidence and **escalate ambiguous cases** for human review

This system is designed for **policy QA, compliance tooling, and trust & safety workflows**.

---

## 🏗️ System Architecture

```
User Query / Ad Text
        ↓
Dense Embeddings (BGE-large)
        ↓
Hybrid Retrieval (FAISS + BM25)
        ↓
Cross-Encoder Reranking
        ↓
LLM Policy Decision + Confidence
        ↓
Escalation / Citation Output
```

**Design principle:** explicit, modular pipelines over black-box frameworks.

---

## 🧠 Design Philosophy (Why No LangChain?)

This project intentionally avoids heavy RAG frameworks (e.g., LangChain / LangGraph) in favor of **custom Python pipelines**, enabling:

* Full control over retrieval, ranking, and scoring
* Transparent evaluation with RAGAS
* Easier debugging and profiling
* Production-aligned system design

Frameworks can be layered later if needed — the core logic is framework-agnostic.

---

## 🛠️ Tech Stack

* **Embeddings**: BGE-large-en-v1.5 (1024-dim)
* **Vector Store**: FAISS (cosine similarity)
* **Sparse Retrieval**: BM25
* **Reranker**: Cross-Encoder
* **LLM**: Gemini (free tier)
* **Evaluation**: RAGAS
* **Language**: Python

---

## 📁 Project Structure

```
google-ads-policy-rag/
├── data/
│   ├── raw/              # Scraped HTML pages (not in git)
│   ├── processed/        # Parsed sections & chunks
│   └── embeddings/       # Vector embeddings (not in git)
├── src/
│   ├── ingestion/        # Scraping, parsing, chunking
│   ├── retrieval/        # FAISS, BM25, hybrid search
│   ├── generation/       # LLM prompts & decision logic
│   └── evaluation/       # RAGAS evaluation
├── run_phase1_DataIngestion.py
├── run_phase2_Embeddings_VectorStore.py
├── run_phase3_RetrievalRanking.py
├── run_phase4_Generation.py
└── tests/
```

---

## 📊 Implementation Status

* **Phase 1**: Ingestion & Hierarchical Chunking ✅
* **Phase 2**: Dense + Sparse Retrieval ✅
* **Phase 3**: Reranking & LLM Decisioning ✅
* **Phase 4**: RAGAS Evaluation & Metrics ✅

---

## 📈 Final Production Metrics

### 🎯 Core Performance
```
| Metric             | Value     | Grade | Notes                             |
| ------------------ | --------- | ----- | --------------------------------- |
| Decision Accuracy  | **80.0%** | A-    | 8/10 correct decisions            |
| Retrieval Recall@5 | **77.8%** | B+    | Correct policy found 7/9 times    |
| MRR                | **0.778** | B+    | Avg correct rank: 1.3             |
| Policy Match       | **66.7%** | C+    | Exact section cited               |
| Confidence Score   | **29.2%** | D     | Conservative calibration          |
| Escalation Rate    | **100%**  | —     | All flagged for review (expected) |
```
---

### ⏱️ Pipeline Performance
```
| Phase                  | Time     | Throughput     | Status |
| ---------------------- | -------- | -------------- | ------ |
| Scraping (25 pages)    | 39.4s    | 0.6 pages/s    | ✅      |
| Parsing (236 sections) | 0.6s     | 393 sections/s | ✅      |
| Chunking (341 chunks)  | 0.2s     | 1,705 chunks/s | ✅      |
| Embeddings (341)       | 23.5s    | 14.5 vec/s     | ✅      |
| FAISS Index            | ~0s      | Instant        | ✅      |
| BM25 Index             | 0.02s    | Instant        | ✅      |
| **Total Pipeline**     | **~64s** | ~5 chunks/s    | ✅      |
```
---

### 🔍 Search & Decision Latency
```
| Stage           | Time | Notes              |
| --------------- | ---- | ------------------ |
| Semantic Search | ~10s | Model load + FAISS |
| BM25 Search     | <1s  | Very fast          |
| Hybrid + Rerank | ~27s | Cross-encoder load |
| LLM Decision    | ~53s | Gemini API (3 ads) |
```
---

### 💾 Data Footprint
```
| Component    | Size    | Status     |
| ------------ | ------- | ---------- |
| Raw HTML     | ~50 MB  | Not in git |
| Chunked JSON | 356 KB  | ✅          |
| Embeddings   | 1.33 MB | Not in git |
| FAISS Index  | 1.33 MB | Not in git |
| BM25 Index   | ~500 KB | Not in git |
```
---

## ✅ Key Achievements

* **80% decision accuracy** — suitable for assisted review workflows
* **Hybrid retrieval** improves recall over dense-only search
* **Sub-minute end-to-end pipeline** for policy ingestion
* **Clean chunking** (no UI junk like “Was this helpful?”)
* **Zero-cost inference** using Gemini free tier

---

## ⚠️ Known Limitations

* **Low confidence calibration** (29%) — overly conservative
* **Low reranker scores** on complex policy queries
* **100% escalation rate** — needs confidence threshold tuning
* **Gemini rate limits** on free tier

These are expected tradeoffs for a safety-first system.

---

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Phase 1: Ingestion
python run_phase1_DataIngestion.py

# Phase 2: Retrieval
python run_phase2_Embeddings_VectorStore.py

# Phase 3: Generation
python run_phase3_RetrievalRanking.py

# Phase 4: Evaluation
python run_phase4_Generation.py
```

---

## 🧪 Evaluation

Evaluation is performed using **RAGAS**, measuring:

* Faithfulness
* Answer relevance
* Context recall
* Policy grounding

Results are stored in:

```
evaluation/evaluation_results.json
```

---

## 📌 Future Improvements

* Confidence calibration & threshold tuning
* Query rewriting for complex ads
* Policy section-level supervision
* Streaming inference & caching
* UI for reviewer workflows
* LangChain/ LangGraph

---

## 📝 License

MIT

---

## 👤 Author

**Varun Vaddi**
MS in Data Science, University of Houston
Focus: RAG systems, policy AI, trust & safety
