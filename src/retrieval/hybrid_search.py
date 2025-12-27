"""
Hybrid Search - Combining Dense + Sparse Retrieval

CONCEPT:
Dense (semantic) + BM25 (keyword) = Best of both worlds!

WHY HYBRID?
Dense search: "crypto" finds "cryptocurrency", "bitcoin"
BM25 search: Finds exact "unapproved pharmaceuticals"
Together: Catches everything! ✅

HOW WE MERGE: Reciprocal Rank Fusion (RRF)
- Simple but effective merging algorithm
- No training required
- Used by Elasticsearch, Google

RRF FORMULA:
score(d) = Σ 1 / (k + rank_i(d))
where k=60 (empirically determined constant)

EXAMPLE:
Dense results:  [doc_A (rank 1), doc_B (rank 2), doc_C (rank 3)]
BM25 results:   [doc_B (rank 1), doc_D (rank 2), doc_A (rank 3)]

RRF scores:
doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  ← highest!
doc_C: 1/(60+3) = 0.0159
doc_D: 1/(60+2) = 0.0161

Final ranking: B, A, D, C
"""

from typing import List, Dict, Tuple
from .vector_store import VectorStore
from .bm25_search import BM25Search
from sentence_transformers import SentenceTransformer
import numpy as np


class HybridSearch:
    """
    Combines dense (vector) and sparse (BM25) search
    
    This is the recommended approach for production RAG systems
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        embeddings_dir: str = "data/embeddings",
        chunks_file: str = "data/processed/chunks.json"
    ):
        """
        Initialize hybrid search
        
        Args:
            model_name: Embedding model
            embeddings_dir: Where vector store is
            chunks_file: Where chunks are
        """
        print("🔄 Initializing HybridSearch...")
        
        # Load embedding model (for dense search)
        print("   Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        
        # Load vector store (dense search)
        print("   Loading vector store...")
        self.vector_store = VectorStore(embeddings_dir=embeddings_dir)
        self.vector_store.load_embeddings()
        self.vector_store.load_index()
        
        # Load BM25 (sparse search)
        print("   Loading BM25 index...")
        self.bm25_search = BM25Search(chunks_file=chunks_file)
        self.bm25_search.load_index()
        
        print("✅ HybridSearch ready!")
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Dict, float]],
        bm25_results: List[Tuple[Dict, float]],
        k: int = 60
    ) -> List[Tuple[Dict, float]]:
        """
        Merge results using Reciprocal Rank Fusion
        
        WHY k=60?
        - Empirically determined by research
        - Works well across different domains
        - You can tune it, but 60 is a good default
        
        Args:
            dense_results: List of (chunk, score) from vector search
            bm25_results: List of (chunk, score) from BM25
            k: RRF constant (default: 60)
        
        Returns:
            Merged and sorted list of (chunk, rrf_score)
        """
        # Dictionary to store RRF scores
        rrf_scores = {}
        
        # Process dense results
        for rank, (chunk, _) in enumerate(dense_results, 1):
            chunk_id = chunk['chunk_id']
            rrf_score = 1.0 / (k + rank)
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {'chunk': chunk, 'score': 0.0}
            
            rrf_scores[chunk_id]['score'] += rrf_score
        
        # Process BM25 results
        for rank, (chunk, _) in enumerate(bm25_results, 1):
            chunk_id = chunk['chunk_id']
            rrf_score = 1.0 / (k + rank)
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {'chunk': chunk, 'score': 0.0}
            
            rrf_scores[chunk_id]['score'] += rrf_score
        
        # Sort by RRF score (descending)
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Return as list of tuples
        return [(item['chunk'], item['score']) for item in sorted_results]
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 20,
        bm25_k: int = 20
    ) -> List[Dict]:
        """
        Hybrid search: dense + BM25 + RRF fusion
        
        PROCESS:
        1. Dense search → top 20
        2. BM25 search → top 20
        3. RRF fusion
        4. Return top K
        
        Args:
            query: Search query
            top_k: Final number of results to return
            dense_k: How many to get from dense search
            bm25_k: How many to get from BM25
        
        Returns:
            List of result dictionaries with:
            - rank: Position (1, 2, 3...)
            - score: RRF score
            - content: Full chunk text
            - metadata: Category, hierarchy, etc.
        """
        # Step 1: Dense search
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        dense_chunks, dense_scores = self.vector_store.search(
            query_embedding,
            top_k=dense_k
        )
        dense_results = list(zip(
            [self.bm25_search.chunks[i] for i, _ in enumerate(dense_chunks)],
            dense_scores
        ))
        
        # Map metadata back to full chunks
        dense_results_full = []
        for meta, score in zip(dense_chunks, dense_scores):
            # Find full chunk by chunk_id
            full_chunk = next(
                (c for c in self.bm25_search.chunks 
                 if c['chunk_id'] == meta['chunk_id']),
                None
            )
            if full_chunk:
                dense_results_full.append((full_chunk, score))
        
        # Step 2: BM25 search
        bm25_chunks, bm25_scores = self.bm25_search.search(query, top_k=bm25_k)
        bm25_results = list(zip(bm25_chunks, bm25_scores))
        
        # Step 3: RRF fusion
        fused_results = self.reciprocal_rank_fusion(
            dense_results_full,
            bm25_results
        )
        
        # Step 4: Format and return top K
        formatted_results = []
        for rank, (chunk, score) in enumerate(fused_results[:top_k], 1):
            result = {
                'rank': rank,
                'score': score,
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content'],
                'metadata': chunk['metadata'],
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def print_results(self, results: List[Dict]):
        """Pretty print search results"""
        print("\n" + "=" * 80)
        print(f"📋 HYBRID SEARCH RESULTS ({len(results)} found)")
        print("=" * 80)
        
        for result in results:
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            
            print(f"\n{'─' * 80}")
            print(f"🏆 Rank #{result['rank']} | RRF Score: {result['score']:.6f}")
            print(f"📂 {hierarchy}")
            print(f"🔗 {result['metadata']['url']}")
            print(f"\n{result['content'][:300]}...")
        
        print("\n" + "=" * 80)


def main():
    """Demo hybrid search"""
    hybrid = HybridSearch()
    
    test_queries = [
        "Are cryptocurrency ads allowed?",
        "unapproved pharmaceuticals",
        "weight loss miracle products"
    ]
    
    print("\n" + "=" * 80)
    print("🔍 HYBRID SEARCH DEMO")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n\n🔎 Query: \"{query}\"")
        print("─" * 80)
        
        results = hybrid.search(query, top_k=3)
        
        for result in results:
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            print(f"\n{result['rank']}. [RRF: {result['score']:.4f}] {hierarchy}")
            print(f"   {result['content'][:150]}...")


if __name__ == "__main__":
    main()