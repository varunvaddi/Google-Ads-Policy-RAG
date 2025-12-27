"""
Run Phase 3: Advanced Retrieval

This script:
1. Builds BM25 index
2. Tests hybrid search (Dense + BM25)
3. Tests reranking with cross-encoder
4. Compares all methods

Usage:
    python run_phase3.py
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.bm25_search import BM25Search
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from retrieval.search import PolicySearch


def main():
    """Run complete Phase 3 pipeline"""
    
    print("=" * 80)
    print("PHASE 3: ADVANCED RETRIEVAL")
    print("=" * 80)
    print()
    
    # Step 1: Build BM25 index
    print("🔤 STEP 1: Building BM25 Index")
    print("-" * 80)
    bm25 = BM25Search()
    bm25.run()
    print()
    
    # Step 2: Test hybrid search
    print("🔄 STEP 2: Testing Hybrid Search (Dense + BM25)")
    print("-" * 80)
    print("   Initializing...")
    hybrid = HybridSearch()
    
    test_query = "Are cryptocurrency trading courses allowed?"
    print(f"\n   Query: '{test_query}'")
    
    start_time = time.time()
    hybrid_results = hybrid.search(test_query, top_k=5)
    hybrid_time = time.time() - start_time
    
    print(f"\n   ✅ Hybrid search completed in {hybrid_time*1000:.0f}ms")
    print(f"   Top result: {' > '.join(hybrid_results[0]['metadata']['hierarchy'])}")
    print()
    
    # Step 3: Test reranking
    print("🎯 STEP 3: Testing Cross-Encoder Reranking")
    print("-" * 80)
    print("   Loading reranker model...")
    reranker = Reranker()
    
    # Get more candidates for reranking
    candidates = hybrid.search(test_query, top_k=10)
    
    start_time = time.time()
    reranked_results = reranker.rerank(test_query, candidates, top_k=5)
    rerank_time = time.time() - start_time
    
    print(f"\n   ✅ Reranking completed in {rerank_time*1000:.0f}ms")
    print()
    
    # Step 4: Comparison
    print("📊 STEP 4: Method Comparison")
    print("-" * 80)
    
    # Compare different methods
    print("\nTesting query: '{}'".format(test_query))
    print()
    
    # Dense only (Phase 2)
    print("1️⃣  Dense Search Only (Phase 2):")
    dense_search = PolicySearch()
    dense_results = dense_search.search(test_query, top_k=3)
    for r in dense_results:
        h = " > ".join(r['metadata']['hierarchy'])
        print(f"   [{r['score']:.4f}] {h[:50]}")
    
    print()
    
    # Hybrid (Dense + BM25)
    print("2️⃣  Hybrid Search (Dense + BM25):")
    hybrid_results = hybrid.search(test_query, top_k=3)
    for r in hybrid_results:
        h = " > ".join(r['metadata']['hierarchy'])
        print(f"   [{r['score']:.6f}] {h[:50]}")
    
    print()
    
    # Hybrid + Reranking
    print("3️⃣  Hybrid + Reranking (Full Pipeline):")
    candidates = hybrid.search(test_query, top_k=10)
    final_results = reranker.rerank(test_query, candidates, top_k=3)
    for r in final_results:
        h = " > ".join(r['metadata']['hierarchy'])
        print(f"   [{r['score']:.4f}] {h[:50]}")
    
    print()
    
    # Summary
    print("=" * 80)
    print("✨ PHASE 3 COMPLETE!")
    print("=" * 80)
    print()
    print("📊 What You Built:")
    print("   • BM25 keyword search index")
    print("   • Hybrid search (Dense + BM25 with RRF)")
    print("   • Cross-encoder reranking")
    print()
    print("📁 Output files:")
    print("   • data/embeddings/bm25.pkl")
    print()
    print("⚡ Performance:")
    print(f"   • Hybrid search: ~{hybrid_time*1000:.0f}ms")
    print(f"   • Reranking: ~{rerank_time*1000:.0f}ms")
    print(f"   • Total: ~{(hybrid_time + rerank_time)*1000:.0f}ms")
    print()
    print("📈 Expected Improvements:")
    print("   • Recall: +15-20% over dense-only")
    print("   • Precision: +10-15% with reranking")
    print()
    print("🚀 Next: Phase 4 (LLM Generation & Structured Output)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise