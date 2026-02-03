"""
Policy Decision Engine - CLEAN DATA VERSION
Uses Google Gemini (FREE) + Clean Hybrid Search (341 chunks, no junk)
"""

import os
import json
import sys
from pathlib import Path
from typing import Union
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Setup imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generation.decision_schema import PolicyDecision, PolicyQuestion
from src.generation.prompts import format_policy_review_prompt, format_policy_qa_prompt
from src.retrieval.hybrid_search import HybridSearch  # ← UPDATED

load_dotenv()


class GeminiPolicyEngine:
    """Main Policy Decision Engine - Uses Clean Hybrid Search"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-09-2025"):
        print("🚀 Initializing Policy Engine (Clean Data)...")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found!\n"
                "Get key: https://aistudio.google.com/app/apikey\n"
                "Set: echo 'GOOGLE_API_KEY=your-key' > .env"
            )
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        print("   Loading clean hybrid search (BM25 + Semantic + Reranking)...")
        self.hybrid_search = HybridSearch()  # ← UPDATED
        
        print("✅ Ready with 341 clean chunks!")
    
    def retrieve_policies(self, query: str, top_k: int = 5) -> list:
        """
        Retrieve relevant policies using hybrid search
        
        This already includes:
        - BM25 keyword search
        - Semantic search
        - RRF fusion
        - Cross-encoder reranking
        
        No need for separate reranker!
        """
        return self.hybrid_search.search(query, top_k=top_k)
    
    def calculate_confidence(self, retrieved_chunks: list, decision: Union[PolicyDecision, PolicyQuestion]) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not retrieved_chunks:
            return 0.0
        
        # Get raw rerank scores
        scores = [chunk.get('rerank_score', 0.0) for chunk in retrieved_chunks[:5]]
        
        if len(scores) < 2:
            retrieval_factor = 0.5  # Single result, moderate confidence
        else:
            # Use rank + margin (best practice for BGE reranker)
            top_score = scores[0]
            second_score = scores[1]
            margin = top_score - second_score
            
            # High margin = clear winner = high confidence
            if margin > 0.002:
                retrieval_factor = 0.8
            elif margin > 0.001:
                retrieval_factor = 0.65
            elif margin > 0.0005:
                retrieval_factor = 0.5
            else:
                retrieval_factor = 0.35  # Low margin = uncertain
            
            # Boost if top score is absolutely high
            if top_score > 0.003:
                retrieval_factor = min(1.0, retrieval_factor + 0.1)
        
        retrieval_factor *= 0.25  # Apply weight
        
        # Decision clarity
        if isinstance(decision, PolicyDecision):
            clarity_factor = 0.3 * 0.25 if decision.decision == "unclear" else 0.8 * 0.25
        else:
            clarity_factor = 0.6 * 0.25
        
        # Multi-source: Check if multiple high-margin results
        high_quality_count = sum(1 for s in scores[:3] if s > 0.002)
        multi_source_factor = (high_quality_count / 3) * 0.15
        
        # LLM confidence (highest weight)
        llm_confidence = getattr(decision, 'confidence', 0.5) * 0.35
        
        total = retrieval_factor + clarity_factor + multi_source_factor + llm_confidence
        return max(0.0, min(1.0, total))
    
    def review_ad(self, ad_text: str) -> PolicyDecision:
        """Review ad for policy compliance"""
        print(f"\n🔍 Retrieving policies (hybrid search)...")
        policies = self.retrieve_policies(ad_text, top_k=5)
        print(f"   Found {len(policies)} relevant policies")
        
        # Show retrieval quality
        if policies:
            top_score = policies[0].get('rerank_score', policies[0].get('score', 0))
            print(f"   Top match score: {top_score:.4f}")

            # DEBUG: Show all scores
            all_scores = [p.get('rerank_score', p.get('score', 0)) for p in policies[:5]]
            print(f"   All top 5 scores: {all_scores}")

            # DEBUG: Show what was retrieved
            for i, p in enumerate(policies[:3], 1):
                hierarchy = " > ".join(p['metadata']['hierarchy'])
                print(f"   {i}. {hierarchy[:80]}... (score: {p.get('rerank_score', 0):.4f})")
        
        prompts = format_policy_review_prompt(ad_text, policies)
        
        print(f"🤖 Calling Gemini ({self.model_name})...")
        
        try:
            full_prompt = f"""{prompts["system"]}

{prompts["user"]}

Respond ONLY with valid JSON:
{{
  "decision": "allowed"|"restricted"|"disallowed"|"unclear",
  "confidence": 0.0-1.0,
  "policy_section": "string",
  "citation_url": "string",
  "justification": "string",
  "policy_quote": "string",
  "risk_factors": ["string"] or null,
  "escalation_required": boolean
}}"""
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            response_text = response.text.strip()
            
            # Clean markdown
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            decision_dict = json.loads(response_text)
            decision = PolicyDecision(**decision_dict)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            decision = PolicyDecision(
                decision="unclear",
                confidence=0.0,
                policy_section="Error",
                citation_url="",
                justification=f"Error: {str(e)}",
                policy_quote="",
                escalation_required=True
            )
        
        # Calculate final confidence
        final_confidence = self.calculate_confidence(policies, decision)
        decision.confidence = final_confidence
        
        if final_confidence < 0.7:
            decision.escalation_required = True
        
        print(f"✅ Decision: {decision.decision} ({decision.confidence:.1%})")
        
        return decision
    
    def print_decision(self, decision: PolicyDecision):
        """Pretty print decision"""
        print("\n" + "=" * 80)
        print("📋 POLICY DECISION (Clean Hybrid Search)")
        print("=" * 80)
        print(f"\n🎯 Decision: {decision.decision.upper()}")
        print(f"📊 Confidence: {decision.confidence:.1%}")
        print(f"\n📂 Policy: {decision.policy_section}")
        print(f"🔗 Source: {decision.citation_url}")
        print(f"\n💬 Justification:\n{decision.justification}")
        print(f"\n📝 Policy Quote:\n\"{decision.policy_quote}\"")
        
        if decision.risk_factors:
            print(f"\n⚠️  Risk Factors:")
            for factor in decision.risk_factors:
                print(f"   • {factor}")
        
        if decision.escalation_required:
            print(f"\n🚨 ESCALATION REQUIRED")
        
        print("\n" + "=" * 80)


def main():
    """Test the engine"""
    engine = GeminiPolicyEngine()
    
    test_ads = [
        "Lose 15 pounds in one week with this miracle pill!",
        "Learn cryptocurrency trading from certified instructors",
        "Buy our new laptop - Intel i7, 16GB RAM"
    ]
    
    for ad in test_ads:
        print(f"\n\n{'='*80}")
        print(f"Testing: {ad}")
        print('='*80)
        
        decision = engine.review_ad(ad)
        engine.print_decision(decision)


if __name__ == "__main__":
    main()