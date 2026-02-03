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

from src.retrieval.search import PolicySearch
from src.retrieval.bm25_search import BM25Search


class HybridSearch:
    """
    Hybrid search using clean data (no junk chunks)
    """
    
    def __init__(self):
        print("="*80)
        print("HYBRID SEARCH V2 - CLEAN DATA (341 chunks, no junk)")
        print("="*80)
        
        print("\n📊 Loading BM25...")
        self.bm25 = BM25Search()
        
        print("\n🧠 Loading semantic search...")
        self.semantic = PolicySearch()
        
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
        raw_scores = self.reranker.predict(pairs)
        # Combine rerank scores with policy boost
        combined_scores = []
        for i, (raw_score, candidate) in enumerate(zip(raw_scores, candidates)):
            policy_boost = candidate.get('policy_boost', 0.0)
            
            # Add boost as a bonus to raw score
            # Boost value of 1.0 adds 0.001 to score (10x typical score range)
            combined_score = raw_score + (policy_boost * 0.001)
            combined_scores.append(combined_score)
        
        combined_scores = np.array(combined_scores)
        sorted_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        reranked = []
        for new_rank, idx in enumerate(sorted_indices, 1):
            candidate = candidates[idx].copy()
            candidate['rank'] = new_rank
            candidate['rerank_score'] = float(raw_scores[idx])  # Keep raw score
            candidate['combined_score'] = float(combined_scores[idx])
            reranked.append(candidate)
        
        return reranked
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Full hybrid search pipeline"""    
    
        # Query expansion for common patterns
        expanded_query = query
        query_lower = query.lower()
        
        # Financial guarantees → unreliable claims
        if any(word in query_lower for word in ['guarantee', 'promised', '100%']) and \
        any(word in query_lower for word in ['return', 'profit', 'income', 'earning']):
            expanded_query = query + " unreliable claims improbable result inaccurate misrepresentation"
        
        # Use expanded query
        bm25_results = self.bm25.search(expanded_query, top_k=30)
        semantic_results = self.semantic.search(expanded_query, top_k=30)
        
        merged_results = self.reciprocal_rank_fusion(bm25_results, semantic_results)
        
        # Pattern detection with sets (cleaner)
        financial_terms = {'return', 'returns', 'profit', 'income', '%', 'apy', 'apr'}
        guarantee_terms = {'guarantee', 'guaranteed', 'promised', 'certain', 'sure', 'risk-free', 'no risk'}
        
        query_words = set(query_lower.split())
        financial_guarantee = (
            any(term in query_lower for term in guarantee_terms) and
            any(term in query_lower for term in financial_terms)
        )
        
        # Apply policy boost (not rank hacking)
        if financial_guarantee:
            print("   🎯 Detected financial guarantee claim — boosting Misrepresentation policies")
            
            boosted_count = 0
            for result in merged_results:
                hierarchy = ' > '.join(result['metadata']['hierarchy'])
                
                # Boost relevant policy types
                if any(keyword in hierarchy for keyword in ['Misrepresentation', 'Unreliable', 'Unacceptable business']):
                    result['policy_boost'] = 1.0
                    boosted_count += 1
                else:
                    result['policy_boost'] = 0.0
            print(f"   → Boosted {boosted_count} policies in merged results")
        else:
            # No boost needed
            for result in merged_results:
                result['policy_boost'] = 0.0
        
        # Sort by boost (boosted items go to top for reranker to see)
        merged_results.sort(
            key=lambda x: x.get('policy_boost', 0.0),
            reverse=True
        )
        # Show top 5 after boost sort
        print(f"   Top 5 after boost:")
        for i, r in enumerate(merged_results[:5], 1):
            h = ' > '.join(r['metadata']['hierarchy'])
            print(f"     {i}. {h[:60]}... (boost: {r.get('policy_boost', 0)})")
        # Reranker still makes final decision on ordering
        final_results = self.rerank(query, merged_results[:20], top_k=top_k)
        
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
            print(f"🌐 {result['metadata']['url']}")
            print(f"\n💬 {result['content'][:250]}...")
            print()


def main():
    """Test clean hybrid search"""
    
    search = HybridSearch()
    
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
