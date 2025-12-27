"""
Vector Store - FAISS-based similarity search

CONCEPT EXPLANATION:
Imagine you have 1000 policy chunks, each with a 1024-number embedding.
To find relevant chunks for a query:
- Bad way: Compare query to ALL 1000 chunks (slow!)
- Good way: Use FAISS (finds nearest neighbors in milliseconds)

WHAT IS FAISS?
- Facebook's library for fast similarity search
- Uses clever algorithms (like tree structures) to skip irrelevant chunks
- Can search millions of vectors in milliseconds

HOW IT WORKS:
1. Build an "index" (special data structure for fast search)
2. Query → Find K nearest neighbors
3. Return those chunks

SIMILARITY METRIC: Cosine Similarity
- Measures angle between vectors
- 1.0 = identical meaning
- 0.0 = completely unrelated
- Works because we normalized embeddings (length = 1)
"""

import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Dict, Tuple


class VectorStore:
    """
    FAISS-based vector database for policy chunks
    
    Think of this as a specialized database that:
    - Stores vectors (not SQL rows)
    - Searches by similarity (not exact match)
    - Returns "closest" items (not filtered items)
    """
    
    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        index_path: str = "data/embeddings/faiss.index"
    ):
        """
        Initialize vector store
        
        Args:
            embeddings_dir: Where embeddings.npy and metadata.json are
            index_path: Where to save FAISS index
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.index_path = Path(index_path)
        
        self.embeddings = None
        self.metadata = None
        self.index = None
        self.dimension = None
    
    def load_embeddings(self):
        """
        Load embeddings and metadata from disk
        
        These were created by generate_embeddings.py
        """
        embeddings_file = self.embeddings_dir / "embeddings.npy"
        metadata_file = self.embeddings_dir / "metadata.json"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {embeddings_file}\n"
                "Run generate_embeddings.py first!"
            )
        
        # Load embeddings (numpy array)
        print(f"📂 Loading embeddings from {embeddings_file}")
        self.embeddings = np.load(embeddings_file)
        self.dimension = self.embeddings.shape[1]
        
        # Load metadata (JSON)
        print(f"📂 Loading metadata from {metadata_file}")
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Loaded {len(self.embeddings)} embeddings (dim={self.dimension})")
        
        # Sanity check
        assert len(self.embeddings) == len(self.metadata), \
            "Embeddings and metadata count mismatch!"
    
    def build_index(self, index_type: str = "FlatIP"):
        """
        Build FAISS index for fast search
        
        FAISS INDEX TYPES:
        
        1. FlatIP (Inner Product - what we use)
           - Exact search (100% accurate)
           - Fast for small datasets (< 1M vectors)
           - "IP" = inner product = dot product = cosine similarity (when normalized)
           
        2. IVFFlat (for larger datasets)
           - Approximate search (99% accurate but faster)
           - Good for 100K+ vectors
           
        3. HNSW (for huge datasets)
           - Graph-based search
           - Good for millions of vectors
        
        For our ~200-300 chunks, FlatIP is perfect!
        
        Args:
            index_type: Type of FAISS index
        """
        print(f"\n🏗️  Building FAISS index (type: {index_type})")
        
        if self.embeddings is None:
            raise ValueError("Load embeddings first!")
        
        # Create FAISS index
        # IndexFlatIP = Flat (exhaustive search) + IP (inner product)
        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add all embeddings to index
        # FAISS requires float32 (not float64)
        embeddings_float32 = self.embeddings.astype('float32')
        self.index.add(embeddings_float32)
        
        print(f"✅ Index built! Contains {self.index.ntotal} vectors")
    
    def save_index(self):
        """
        Save FAISS index to disk
        
        Why save?
        - Building index takes time
        - Once built, we can load it instantly
        - Essential for production systems
        """
        if self.index is None:
            raise ValueError("Build index first!")
        
        faiss.write_index(self.index, str(self.index_path))
        print(f"💾 Saved FAISS index to {self.index_path}")
    
    def load_index(self):
        """
        Load pre-built FAISS index from disk
        
        Much faster than rebuilding!
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        
        self.index = faiss.read_index(str(self.index_path))
        print(f"📂 Loaded FAISS index ({self.index.ntotal} vectors)")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[Dict], List[float]]:
        """
        Search for similar chunks
        
        HOW IT WORKS:
        1. Query embedding: [0.23, -0.45, ...] (1024 numbers)
        2. FAISS compares to all stored embeddings
        3. Returns indices of K most similar chunks
        4. We look up metadata for those indices
        
        Args:
            query_embedding: The query vector (1024 dimensions)
            top_k: How many results to return
        
        Returns:
            Tuple of (results, scores)
            - results: List of chunk metadata dicts
            - scores: Similarity scores (higher = more similar)
        
        Example:
            query = "Are crypto ads allowed?"
            query_emb = model.encode(query)
            results, scores = vector_store.search(query_emb, top_k=3)
            # Returns 3 most relevant policy chunks
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() or load_index()")
        
        # Ensure query is 2D array (FAISS requirement)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert to float32
        query_embedding = query_embedding.astype('float32')
        
        # Search!
        # D = distances (similarity scores)
        # I = indices (which chunks matched)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get metadata for matched chunks
        results = []
        scores = []
        
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):  # Valid index
                results.append(self.metadata[idx])
                scores.append(float(score))
        
        return results, scores
    
    def run(self):
        """
        Build and save the vector store
        
        This is called once to create the index
        """
        print("=" * 60)
        print("PHASE 2: BUILDING VECTOR STORE")
        print("=" * 60)
        print()
        
        # Load embeddings
        self.load_embeddings()
        
        # Build index
        self.build_index()
        
        # Save index
        self.save_index()
        
        print("\n" + "=" * 60)
        print("✨ VECTOR STORE READY!")
        print("=" * 60)
        print(f"\n📊 Index Statistics:")
        print(f"   Vectors indexed: {self.index.ntotal}")
        print(f"   Dimension: {self.dimension}")
        print(f"   Index type: FlatIP (exact search)")
        print(f"\n🎯 Ready for semantic search!")


def main():
    """Build the vector store"""
    store = VectorStore()
    store.run()
    
    # Demo: Test search with a sample embedding
    print("\n🧪 Testing search functionality...")
    test_query_emb = store.embeddings[0]  # Use first embedding as test
    results, scores = store.search(test_query_emb, top_k=3)
    
    print(f"\n📋 Test search results (top 3):")
    for i, (result, score) in enumerate(zip(results, scores), 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Category: {result['metadata']['category']}")
        print(f"   Preview: {result['content_preview'][:100]}...")


if __name__ == "__main__":
    main()