"""
BM25 Keyword Search

WHAT IS BM25?
- "Best Match 25" - probabilistic keyword ranking algorithm
- Like TF-IDF but better (handles document length)
- Industry standard for keyword search

WHY WE NEED IT:
Dense search (Phase 2) can miss exact keyword matches
Example: "unapproved pharmaceuticals" (exact legal term)
- Dense might miss if not in training data
- BM25 catches it every time ✅

HOW IT WORKS:
1. Tokenize: "crypto ads" → ["crypto", "ads"]
2. Build index: word → documents containing it
3. Score: combines term frequency + document frequency
4. Rank: sort by score

FORMULA:
BM25(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))

Don't worry about formula - library handles it!
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import pickle


class BM25Search:
    """
    BM25-based keyword search for policy chunks
    
    Complements dense search by catching exact keyword matches
    """
    
    def __init__(
        self,
        chunks_file: str = "data/processed/chunks.json",
        index_file: str = "data/embeddings/bm25.pkl"
    ):
        """
        Initialize BM25 search
        
        Args:
            chunks_file: Where chunks are stored
            index_file: Where to save/load BM25 index
        """
        self.chunks_file = Path(chunks_file)
        self.index_file = Path(index_file)
        
        self.chunks = []
        self.bm25 = None
        self.tokenized_corpus = []
    
    def simple_tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization
        
        WHY SIMPLE?
        - BM25 doesn't need fancy NLP
        - Split on whitespace, lowercase
        - Remove punctuation
        
        Args:
            text: Input text
        
        Returns:
            List of tokens (words)
        """
        # Lowercase and split
        tokens = text.lower().split()
        
        # Remove punctuation
        tokens = [''.join(c for c in token if c.isalnum()) 
                  for token in tokens]
        
        # Remove empty strings
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def build_index(self):
        """
        Build BM25 index from chunks
        
        STEPS:
        1. Load all chunks
        2. Tokenize each chunk's content
        3. Build BM25 index
        4. Save index for reuse
        """
        print("📚 Loading chunks...")
        with open(self.chunks_file, 'r') as f:
            self.chunks = json.load(f)
        
        print(f"✅ Loaded {len(self.chunks)} chunks")
        
        # Tokenize all chunks
        print("🔤 Tokenizing corpus...")
        self.tokenized_corpus = [
            self.simple_tokenize(chunk['content']) 
            for chunk in self.chunks
        ]
        
        # Build BM25 index
        print("🏗️  Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Save index
        print(f"💾 Saving index to {self.index_file}")
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus,
                'chunks': self.chunks
            }, f)
        
        print("✅ BM25 index built!")
    
    def load_index(self):
        """
        Load pre-built BM25 index
        
        Much faster than rebuilding
        """
        if not self.index_file.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {self.index_file}\n"
                "Run build_index() first!"
            )
        
        print(f"📂 Loading BM25 index from {self.index_file}")
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.tokenized_corpus = data['tokenized_corpus']
        self.chunks = data['chunks']
        
        print(f"✅ Loaded BM25 index ({len(self.chunks)} chunks)")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[Dict], List[float]]:
        """
        Search using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            Tuple of (chunks, scores)
        
        Example:
            query = "unapproved pharmaceuticals"
            chunks, scores = bm25.search(query, top_k=5)
        """
        if self.bm25 is None:
            raise ValueError("Index not loaded. Call load_index() first!")
        
        # Tokenize query
        tokenized_query = self.simple_tokenize(query)
        
        # Get scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Get corresponding chunks and scores
        results = []
        result_scores = []
        
        for idx in top_indices:
            if scores[idx] > 0:  # Only return if score > 0
                results.append(self.chunks[idx])
                result_scores.append(float(scores[idx]))
        
        return results, result_scores
    
    def run(self):
        """Build the BM25 index"""
        print("=" * 60)
        print("BM25 KEYWORD SEARCH INDEX")
        print("=" * 60)
        print()
        
        self.build_index()
        
        print("\n" + "=" * 60)
        print("✨ BM25 INDEX READY!")
        print("=" * 60)
        print(f"\n📊 Statistics:")
        print(f"   Total chunks indexed: {len(self.chunks)}")
        print(f"   Average tokens per chunk: {np.mean([len(t) for t in self.tokenized_corpus]):.1f}")
        print(f"\n🎯 Ready for keyword search!")


def main():
    """Build BM25 index"""
    bm25 = BM25Search()
    bm25.run()
    
    # Demo search
    print("\n🧪 Testing BM25 search...")
    test_queries = [
        "unapproved pharmaceuticals",
        "cryptocurrency advertising",
        "weight loss products"
    ]
    
    for query in test_queries:
        results, scores = bm25.search(query, top_k=2)
        print(f"\n📝 Query: '{query}'")
        for i, (chunk, score) in enumerate(zip(results, scores), 1):
            hierarchy = " > ".join(chunk['metadata']['hierarchy'])
            print(f"   {i}. [{score:.2f}] {hierarchy}")


if __name__ == "__main__":
    main()