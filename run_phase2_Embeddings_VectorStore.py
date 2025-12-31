"""
Run Phase 2: Embeddings & Vector Store

This script:
1. Generates embeddings for all chunks (from Phase 1)
2. Builds FAISS vector index
3. Tests search functionality

Usage:
    python run_phase2.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.retrieval.generate_embeddings import EmbeddingGenerator
from src.retrieval.build_vector_store import VectorStore
from src.retrieval.search import PolicySearch


def main():
    
    print("=" * 60)
    print("PHASE 2: EMBEDDINGS & VECTOR STORE")
    print("=" * 60)
    print()
    
    # Step 1: Generate embeddings
    print("🔢 STEP 1: Generating Embeddings")
    print("-" * 60)
    generator = EmbeddingGenerator()
    embeddings, chunks = generator.run()
    print()
    
    if embeddings is None:
        print("❌ Embedding generation failed. Exiting.")
        return
    


    # Step 2: Build vector store
    print("🗄️  STEP 2: Building Vector Store (FAISS)")
    print("-" * 60)
    store = VectorStore()
    store.run()
    print()
    


    # Step 3: Test search
    print("🔍 STEP 3: Testing Search")
    print("-" * 60)
    search = PolicySearch()
    
    # Test queries
    test_queries = [
        "Are cryptocurrency ads allowed?",
        "Can I advertise weight loss products?",
        "What are the healthcare advertising rules?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        results = search.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            print(f"   {i}. [{result['score']:.3f}] {hierarchy}")
            print(f"      {result['content'][:100]}...")
    
    print()
    

    
    # Summary
    print("\n📊 PHASE II: SUMMARY")
    print("-" * 60)
    print()
    print("📊 What You Built:")
    print(f"   • Generated {len(embeddings)} embeddings (1024-dim each)")
    print(f"   • Built FAISS vector index")
    print(f"   • Enabled semantic search")
    print()
    print("📁 Output files:")
    print("   • data/embeddings/embeddings.npy")
    print("   • data/embeddings/metadata.json")
    print("   • data/embeddings/faiss.index")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise