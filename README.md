# Google Ads Policy RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for interpreting and enforcing Google Ads policies.

## 🎯 Project Goal

Build an AI-powered assistant that can:
- Answer questions about Google Ads policies
- Review ad text for policy compliance
- Provide citations to official policy documentation
- Flag ambiguous cases for human review

## 🏗️ Architecture

```
User Query → Embeddings → Hybrid Search (Dense + BM25) → Reranking → LLM Decision
```

## 📊 Current Status

- [x] Phase 1: Data Ingestion (In Progress)
- [ ] Phase 2: Vector Search & Retrieval
- [ ] Phase 3: Reranking & Generation
- [ ] Phase 4: Evaluation & Optimization

## 🛠️ Tech Stack

- **Embeddings**: BGE-large-en-v1.5
- **Vector DB**: FAISS
- **LLM**: OpenAI GPT-4
- **Framework**: LangChain
- **Evaluation**: RAGAS

## 📁 Project Structure

```
google-ads-policy-rag/
├── data/
│   ├── raw/              # Scraped HTML pages
│   ├── processed/        # Parsed and chunked data
│   └── embeddings/       # Vector embeddings
├── src/
│   ├── ingestion/        # Data scraping and processing
│   ├── retrieval/        # Search and ranking
│   ├── generation/       # LLM prompts and responses
│   └── evaluation/       # Metrics and testing
├── notebooks/            # Exploration and experiments
└── tests/               # Unit tests
```

## 🚀 Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd google-ads-policy-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data ingestion (Phase 1)
python src/ingestion/scrape_policies.py
```

## 📈 Progress Log

### Phase 1: Data Ingestion (Current)
- Setting up project structure
- Implementing policy scraper
- Creating hierarchical chunking strategy

## 🤝 Contributing

This is a learning project. Feedback and suggestions welcome!

## 📝 License

MIT