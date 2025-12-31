"""
Hybrid Search V2 - Using Clean Data
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Dict
from sentence_transformers import CrossEncoder

from src.retrieval.search_v2_clean import PolicySearchV2Clean
from src.retrieval.bm25_search_v2_clean import BM25SearchV2Clean


class HybridSearchV2Clean:
    """
    Hybrid search using clean data (no junk chunks)
    """
    
    def __init__(self):
        print("="*80)
        print("HYBRID SEARCH V2 - CLEAN DATA (341 chunks, no junk)")
        print("="*80)
        
        print("\n📊 Loading BM25...")
        self.bm25 = BM25SearchV2Clean()
        
        print("\n🧠 Loading semantic search...")
        self.semantic = PolicySearchV2Clean()
        
        print("\n🎯 Loading reranker...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-large')
        
        print("\n✅ Clean hybrid search ready!")
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        semantic_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """Reciprocal Rank Fusion"""
        
        rrf_scores = {}
        
        # Add BM25 rankings
        for result in bm25_results:
            chunk_idx = self._find_chunk_index(result['content'])
            rank = result['rank']
            score = 1.0 / (k + rank)
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + score
        
        # Add semantic rankings
        for result in semantic_results:
            chunk_idx = self._find_chunk_index(result['content'])
            rank = result['rank']
            score = 1.0 / (k + rank)
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + score
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build merged results
        merged_results = []
        for rank, (chunk_idx, rrf_score) in enumerate(sorted_indices[:20], 1):
            chunk = self.semantic.chunks[chunk_idx]
            
            merged_results.append({
                'rank': rank,
                'score': rrf_score,
                'rrf_score': rrf_score,
                'content': chunk['content'],
                'metadata': chunk['metadata']
            })
        
        return merged_results
    
    def _find_chunk_index(self, content: str) -> int:
        """Find chunk index by content matching"""
        for idx, chunk in enumerate(self.semantic.chunks):
            if chunk['content'] == content:
                return idx
        return 0
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank using cross-encoder"""
        
        pairs = [[query, candidate['content']] for candidate in candidates]
        scores = self.reranker.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        reranked = []
        for new_rank, idx in enumerate(sorted_indices, 1):
            candidate = candidates[idx].copy()
            candidate['rank'] = new_rank
            candidate['rerank_score'] = float(scores[idx])
            reranked.append(candidate)
        
        return reranked
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Full hybrid search pipeline"""
        
        bm25_results = self.bm25.search(query, top_k=20)
        semantic_results = self.semantic.search(query, top_k=20)
        merged_results = self.reciprocal_rank_fusion(bm25_results, semantic_results)
        final_results = self.rerank(query, merged_results, top_k=top_k)
        
        return final_results
    
    def print_results(self, results: List[Dict], query: str = ""):
        """Pretty print results"""
        print("\n" + "="*80)
        print("HYBRID SEARCH RESULTS - CLEAN DATA")
        if query:
            print(f"Query: \"{query}\"")
        print("="*80)
        
        for result in results:
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            
            print(f"\n🏆 Rank #{result['rank']} | Score: {result['rerank_score']:.4f}")
            print(f"📂 {hierarchy}")
            print(f"�� {result['metadata']['url']}")
            print(f"\n💬 {result['content'][:250]}...")
            print()


def main():
    """Test clean hybrid search"""
    
    search = HybridSearchV2Clean()
    
    test_queries = [
        "cryptocurrency ads",
        "weight loss miracle pill",
        "unapproved pharmaceuticals"
    ]
    
    for query in test_queries:
        print(f"\n\n{'='*80}")
        print(f"Testing: \"{query}\"")
        print('='*80)
        
        results = search.search(query, top_k=5)
        search.print_results(results, query)


if __name__ == "__main__":
    main()
