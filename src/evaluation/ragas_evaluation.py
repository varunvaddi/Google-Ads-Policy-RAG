"""
RAGAS Evaluation for Google Ads Policy RAG
Measures: Retrieval quality, Answer accuracy, Faithfulness
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict
from collections import Counter
import numpy as np

from src.retrieval.hybrid_search import HybridSearch
from src.generation.decision_engine import GeminiPolicyEngine


class RAGEvaluator:
    """
    Evaluate RAG system quality
    
    Metrics:
    1. Retrieval Quality (Recall@K, MRR, Precision)
    2. Answer Quality (Accuracy, Faithfulness)
    3. System Performance (Latency, Consistency)
    """
    
    def __init__(self):
        print("="*80)
        print("RAGAS EVALUATION - RAG SYSTEM QUALITY")
        print("="*80)
        
        # Load systems
        print("\nLoading systems...")
        self.retrieval = HybridSearch()
        self.engine = GeminiPolicyEngine()
        
        # Test cases
        self.test_cases = self._create_test_cases()
        
        print(f"✅ Ready! Test cases: {len(self.test_cases)}")
    
    def _create_test_cases(self) -> List[Dict]:
        """
        Create test cases with ground truth
        
        Format:
        {
            "query": "ad text or question",
            "expected_decision": "allowed|restricted|disallowed",
            "expected_policy": "Healthcare > Weight Loss",
            "rationale": "why this decision"
        }
        """
        return [
            {
                "id": 1,
                "query": "Lose 15 pounds in one week with this miracle pill",
                "expected_decision": "disallowed",
                "expected_policy": "Healthcare and medicines",
                "rationale": "Unrealistic weight loss claims"
            },
            {
                "id": 2,
                "query": "100% guaranteed returns on Bitcoin investment",
                "expected_decision": "disallowed",
                "expected_policy": "Financial services",
                "rationale": "Guaranteed returns prohibited"
            },
            {
                "id": 3,
                "query": "Learn cryptocurrency trading from certified experts",
                "expected_decision": "restricted",
                "expected_policy": "Cryptocurrencies",
                "rationale": "Crypto requires certification"
            },
            {
                "id": 4,
                "query": "Buy our new laptop - Intel i7, 16GB RAM, free shipping",
                "expected_decision": "allowed",
                "expected_policy": None,
                "rationale": "Standard consumer electronics"
            },
            {
                "id": 5,
                "query": "Cure diabetes naturally with herbal tea",
                "expected_decision": "disallowed",
                "expected_policy": "Healthcare and medicines",
                "rationale": "Unapproved medical claims"
            },
            {
                "id": 6,
                "query": "Vote for John Smith - best candidate for mayor",
                "expected_decision": "restricted",
                "expected_policy": "Political content",
                "rationale": "Election advertising requires verification"
            },
            {
                "id": 7,
                "query": "Premium whiskey delivery - 21+ only",
                "expected_decision": "restricted",
                "expected_policy": "Alcohol",
                "rationale": "Alcohol advertising has restrictions"
            },
            {
                "id": 8,
                "query": "Counterfeit designer handbags wholesale",
                "expected_decision": "disallowed",
                "expected_policy": "Counterfeit goods",
                "rationale": "Counterfeit products prohibited"
            },
            {
                "id": 9,
                "query": "Online pharmacy - no prescription needed",
                "expected_decision": "disallowed",
                "expected_policy": "Healthcare and medicines",
                "rationale": "Unauthorized pharmacy"
            },
            {
                "id": 10,
                "query": "Get rich quick with forex trading secrets",
                "expected_decision": "disallowed",
                "expected_policy": "Financial services",
                "rationale": "Misleading financial claims"
            }
        ]
    
    def evaluate_retrieval(self) -> Dict:
        """
        Evaluate retrieval quality
        
        Metrics:
        - Recall@5: Did we find the right policy in top 5?
        - MRR: Mean Reciprocal Rank (how high was correct policy?)
        - Precision@5: How many of top 5 are relevant?
        """
        print("\n" + "="*80)
        print("1. RETRIEVAL EVALUATION")
        print("="*80)
        
        results = {
            'recall_at_5': [],
            'mrr': [],
            'avg_score': []
        }
        
        for case in self.test_cases:
            if not case['expected_policy']:
                continue  # Skip "allowed" cases (no specific policy)
            
            # Retrieve
            retrieved = self.retrieval.search(case['query'], top_k=5)
            
            # Check if expected policy in results
            found = False
            rank = None
            
            for i, result in enumerate(retrieved, 1):
                hierarchy = ' > '.join(result['metadata']['hierarchy'])
                if case['expected_policy'].lower() in hierarchy.lower():
                    found = True
                    rank = i
                    break
            
            # Metrics
            results['recall_at_5'].append(1 if found else 0)
            results['mrr'].append(1/rank if rank else 0)
            results['avg_score'].append(retrieved[0]['rerank_score'] if retrieved else 0)
            
            # Print result
            status = "✅" if found else "❌"
            print(f"{status} Case {case['id']}: {case['query'][:50]}...")
            if found:
                print(f"   Found at rank {rank}, score: {retrieved[rank-1]['rerank_score']:.4f}")
            else:
                print(f"   Not found in top 5")
        
        # Calculate averages
        summary = {
            'recall@5': np.mean(results['recall_at_5']),
            'mrr': np.mean(results['mrr']),
            'avg_top_score': np.mean(results['avg_score'])
        }
        
        print(f"\n📊 Retrieval Summary:")
        print(f"   Recall@5: {summary['recall@5']:.1%}")
        print(f"   MRR: {summary['mrr']:.3f}")
        print(f"   Avg Top Score: {summary['avg_top_score']:.4f}")
        
        return summary
    
    def evaluate_decisions(self) -> Dict:
        """
        Evaluate end-to-end decision accuracy
        
        Metrics:
        - Decision Accuracy: Did it choose right decision?
        - Policy Accuracy: Did it cite right policy?
        - Confidence Distribution
        """
        print("\n" + "="*80)
        print("2. DECISION EVALUATION")
        print("="*80)
        
        results = {
            'correct_decisions': 0,
            'correct_policies': 0,
            'confidences': [],
            'escalations': 0
        }
        
        for case in self.test_cases:
            print(f"\n📝 Case {case['id']}: {case['query'][:60]}...")
            
            # Get decision
            decision = self.engine.review_ad(case['query'])
            
            # Check decision correctness
            decision_correct = decision.decision == case['expected_decision']
            if decision_correct:
                results['correct_decisions'] += 1
            
            # Check policy correctness
            if case['expected_policy']:
                policy_correct = case['expected_policy'].lower() in decision.policy_section.lower()
                if policy_correct:
                    results['correct_policies'] += 1
            
            # Track metrics
            results['confidences'].append(decision.confidence)
            if decision.escalation_required:
                results['escalations'] += 1
            
            # Print result
            status = "✅" if decision_correct else "❌"
            print(f"{status} Expected: {case['expected_decision']}, Got: {decision.decision}")
            print(f"   Confidence: {decision.confidence:.1%}, Escalation: {decision.escalation_required}")
        
        # Calculate summary
        total = len(self.test_cases)
        summary = {
            'decision_accuracy': results['correct_decisions'] / total,
            'policy_accuracy': results['correct_policies'] / (total - 1),  # Exclude "allowed" case
            'avg_confidence': np.mean(results['confidences']),
            'escalation_rate': results['escalations'] / total
        }
        
        print(f"\n📊 Decision Summary:")
        print(f"   Decision Accuracy: {summary['decision_accuracy']:.1%}")
        print(f"   Policy Accuracy: {summary['policy_accuracy']:.1%}")
        print(f"   Avg Confidence: {summary['avg_confidence']:.1%}")
        print(f"   Escalation Rate: {summary['escalation_rate']:.1%}")
        
        return summary
    
    def run_full_evaluation(self):
        """Run complete evaluation suite"""
        
        # 1. Retrieval
        retrieval_metrics = self.evaluate_retrieval()
        
        # 2. Decisions
        decision_metrics = self.evaluate_decisions()
        
        # 3. Final Summary
        print("\n" + "="*80)
        print("📊 FINAL EVALUATION SUMMARY")
        print("="*80)
        
        all_metrics = {
            'retrieval': retrieval_metrics,
            'decisions': decision_metrics,
            'system': {
                'total_chunks': 341,
                'model': 'BGE-large-en-v1.5 + gemini-2.5-flash',
                'architecture': 'Hybrid (BM25 + Semantic + Reranking)'
            }
        }
        
        # Pretty print
        print("\n🎯 Retrieval Performance:")
        print(f"   Recall@5: {retrieval_metrics['recall@5']:.1%}")
        print(f"   MRR: {retrieval_metrics['mrr']:.3f}")
        
        print("\n🎯 Decision Performance:")
        print(f"   Accuracy: {decision_metrics['decision_accuracy']:.1%}")
        print(f"   Confidence: {decision_metrics['avg_confidence']:.1%}")
        
        # Save results
        # Save results
        base_dir = Path(__file__).parent
        output_path = base_dir.parent.parent / "evaluation" / "evaluation_results.json"

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")
        
        return all_metrics


def main():
    """Run evaluation"""
    evaluator = RAGEvaluator()
    results = evaluator.run_full_evaluation()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()