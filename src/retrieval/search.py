"""
Search Interface - User-friendly semantic search

This brings everything together:
1. Take a text query from user
2. Convert to embedding
3. Search FAISS index
4. Return formatted results

USAGE:
    from src.retrieval.search import PolicySearch
    
    search = PolicySearch()
    results = search.search("Are crypto ads allowed?")
    
    for result in results:
        print(result['content'])
"""

import json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from .vector_store import VectorStore


class PolicySearch:
    """
    High-level search interface for Google Ads policies
    
    This is what you'll actually use to search!
    Abstracts away all the embedding/vector complexity.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        embeddings_dir: str = "data/embeddings"
    ):
        """
        Initialize search system
        
        Args:
            model_name: Embedding model (must match what you used to generate embeddings!)
            embeddings_dir: Where embeddings and FAISS index are stored
        """
        print("🔍 Initializing PolicySearch...")
        
        # Load embedding model
        # (This is the same model used to create embeddings)
        print(f"   Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Load vector store
        print(f"   Loading vector store from: {embeddings_dir}")
        self.vector_store = VectorStore(embeddings_dir=embeddings_dir)
        
        # Load everything
        self.vector_store.load_embeddings()
        self.vector_store.load_index()
        
        # Load full chunks (with complete content, not just preview)
        chunks_file = Path("data/processed/chunks.json")
        with open(chunks_file, 'r') as f:
            self.full_chunks = json.load(f)
        
        print("✅ PolicySearch ready!")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search for policy chunks relevant to query
        
        WORKFLOW:
        1. User types query: "Are crypto ads allowed?"
        2. Convert query to embedding: [0.23, -0.45, ...]
        3. FAISS finds similar chunk embeddings
        4. Return those chunks with scores
        
        Args:
            query: Natural language question
            top_k: How many results to return (default: 5)
            min_score: Minimum similarity score (0.0 - 1.0)
                      Higher = more strict filtering
        
        Returns:
            List of result dictionaries with:
            - content: Full chunk text
            - metadata: Category, hierarchy, URL
            - score: Similarity score
            - rank: Position in results (1, 2, 3...)
        """
        # Step 1: Convert query to embedding
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True  # Must match training!
        )
        
        # Step 2: Search vector store
        results_meta, scores = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )
        
        # Step 3: Get full chunk content and format results
        formatted_results = []
        
        for rank, (meta, score) in enumerate(zip(results_meta, scores), 1):
            # Skip if below minimum score threshold
            if score < min_score:
                continue
            
            # Find full chunk by chunk_id
            chunk_id = meta['chunk_id']
            full_chunk = next(
                (c for c in self.full_chunks if c['chunk_id'] == chunk_id),
                None
            )
            
            if full_chunk:
                result = {
                    'rank': rank,
                    'score': score,
                    'chunk_id': chunk_id,
                    'content': full_chunk['content'],
                    'metadata': meta['metadata'],
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def print_results(self, results: List[Dict]):
        """
        Pretty print search results
        
        Makes results human-readable
        """
        print("\n" + "=" * 80)
        print(f"📋 SEARCH RESULTS ({len(results)} found)")
        print("=" * 80)
        
        for result in results:
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            
            print(f"\n{'─' * 80}")
            print(f"🏆 Rank #{result['rank']} | Similarity: {result['score']:.4f}")
            print(f"📂 {hierarchy}")
            print(f"🔗 {result['metadata']['url']}")
            print(f"\n{result['content']}")
        
        print("\n" + "=" * 80)


def main():
    """
    Demo the search functionality
    
    Try different queries to see how semantic search works!
    """
    # Initialize search
    search = PolicySearch()
    
    # Example queries
    queries = [
        "Are cryptocurrency ads allowed?",
        "Can I advertise weight loss supplements?",
        "What are the rules for political advertising?",
    ]
    
    print("\n" + "=" * 80)
    print("🔍 POLICY SEARCH DEMO")
    print("=" * 80)
    
    for query in queries:
        print(f"\n\n🔎 Query: \"{query}\"")
        print("─" * 80)
        
        results = search.search(query, top_k=3)
        
        # Print condensed results
        for i, result in enumerate(results, 1):
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            print(f"\n{i}. [{result['score']:.3f}] {hierarchy}")
            print(f"   {result['content'][:150]}...")
    
    print("\n\n" + "=" * 80)
    print("✨ Try your own queries!")
    print("=" * 80)
    print("\nExample usage:")
    print("  from src.retrieval.search import PolicySearch")
    print("  search = PolicySearch()")
    print('  results = search.search("your question here")')
    print("  search.print_results(results)")


if __name__ == "__main__":
    main()