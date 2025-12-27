# Git Workflow for This Project

## Initial Setup (One Time)

```bash
# Navigate to project directory
cd google-ads-policy-rag

# Initialize git (if not done already)
git init
git branch -M main

# Create all files from artifacts
# (Copy paste the code I provided into respective files)

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/embeddings/.gitkeep
```

## Phase 1: Data Ingestion (4 Commits)

### Commit 1: Initial Setup
```bash
git add .gitignore README.md requirements.txt
git add src/__init__.py src/ingestion/__init__.py
git add data/
git commit -m "feat: initial project structure and documentation

- Added .gitignore for Python and data files
- Created README with project overview
- Added initial requirements.txt
- Set up src/ directory structure"

git push -u origin main
```

### Commit 2: Policy Scraper
```bash
git add src/ingestion/scrape_policies.py
git commit -m "feat: implement Google Ads policy scraper

- Scrapes 7 major policy categories
- Saves HTML and metadata
- Includes rate limiting and error handling
- Outputs to data/raw/"

git push
```

### Commit 3: Policy Parser
```bash
git add src/ingestion/parse_policies.py
git commit -m "feat: add hierarchical policy parser

- Extracts sections from HTML
- Preserves heading hierarchy (h1-h4)
- Categorizes content types (policy/example/exception)
- Outputs structured JSON to data/processed/"

git push
```

### Commit 4: Chunking Strategy
```bash
git add src/ingestion/chunking.py run_phase1.py
git commit -m "feat: implement intelligent chunking strategy

- Keeps small sections intact (< 800 chars)
- Splits large sections on paragraphs
- Adds hierarchical context to chunks
- Includes overlap for continuity
- Added run_phase1.py orchestration script"

git push
```

### Commit 5: Tests
```bash
git add tests/test_ingestion.py
git commit -m "test: add unit tests for ingestion pipeline

- Tests for scraper, parser, and chunker
- Validates data structures
- Checks chunking logic
- End-to-end integration tests"

git push
```

### Commit 6: Update README with Phase 1 Complete
```bash
git add README.md
git commit -m "docs: mark Phase 1 as complete in README

- Updated progress checklist
- Added usage instructions
- Documented output files"

git push
```

## Phase 2: Vector Store & Retrieval (Next)

### Commit 7: Add Embedding Dependencies
```bash
# Update requirements.txt with:
# - sentence-transformers
# - faiss-cpu
# - langchain

git add requirements.txt
git commit -m "deps: add embedding and vector store dependencies"

git push
```

### Commit 8: Embedding Generator
```bash
git add src/retrieval/embeddings.py
git commit -m "feat: implement embedding generation with BGE

- Uses BAAI/bge-large-en-v1.5
- Batch processing for efficiency
- Progress tracking with tqdm
- Saves embeddings as numpy arrays"

git push
```

### Continue similar pattern...

## Best Practices

### Commit Message Format
```
<type>: <brief description>

[optional detailed description]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks
- `deps`: Dependency updates

### Good Commit Examples
```bash
git commit -m "feat: add hybrid search with BM25 and dense retrieval"

git commit -m "fix: handle empty sections in parser

Fixed crash when encountering sections with no content.
Now skips empty sections gracefully."

git commit -m "perf: optimize embedding batch size

Increased batch size from 32 to 128, reducing embedding time by 40%"

git commit -m "docs: add architecture diagram to README"
```

### Bad Commit Examples
```bash
git commit -m "update"  # Too vague
git commit -m "fix stuff"  # Not descriptive
git commit -m "WIP"  # Never commit work-in-progress to main
```

## Branching Strategy (Optional but Recommended)

For new features, use branches:
```bash
# Create feature branch
git checkout -b feature/reranking

# Make changes
git add src/retrieval/reranker.py
git commit -m "feat: add cross-encoder reranking"

# Push branch
git push -u origin feature/reranking

# Merge back to main (or create PR on GitHub)
git checkout main
git merge feature/reranking
git push
```

## Checking Status
```bash
# See what's changed
git status

# See what's different
git diff

# See commit history
git log --oneline --graph

# See changes in specific file
git diff src/ingestion/scrape_policies.py
```

## Undoing Changes

```bash
# Undo changes in working directory
git checkout -- filename

# Undo last commit (keeps changes)
git reset --soft HEAD~1

# Undo last commit (discards changes)
git reset --hard HEAD~1

# Undo specific file from staging
git reset HEAD filename
```

## Tags for Milestones

```bash
# Tag major milestones
git tag -a v1.0-phase1-complete -m "Phase 1: Data ingestion complete"
git push origin v1.0-phase1-complete

git tag -a v2.0-phase2-complete -m "Phase 2: Vector store complete"
git push origin v2.0-phase2-complete
```

## .gitignore Tips

Already included in `.gitignore`:
- `data/raw/*.html` (large files)
- `data/embeddings/*.npy` (binary files)
- `__pycache__/` (Python cache)
- `.env` (secrets)

If you need to commit a sample:
```bash
# Force add one example file
git add -f data/raw/sample_policy.html
```