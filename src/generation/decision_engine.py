"""
Policy Decision Engine - FINAL VERSION
Uses Google Gemini (FREE)
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
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker

load_dotenv()


class GeminiPolicyEngine:
    """Main Policy Decision Engine - Uses Gemini (FREE)"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-09-2025", use_reranking: bool = True):
        print("🚀 Initializing Policy Engine...")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found!\n"
                "Get key: https://aistudio.google.com/app/apikey\n"
                "Set: echo 'GOOGLE_API_KEY=your-key' > .env"
            )
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        print("   Loading retrieval...")
        self.hybrid_search = HybridSearch()
        
        self.use_reranking = use_reranking
        if use_reranking:
            print("   Loading reranker...")
            self.reranker = Reranker()
        
        print("✅ Ready!")
    
    def retrieve_policies(self, query: str, top_k: int = 5) -> list:
        """Retrieve relevant policies"""
        candidate_k = top_k * 2 if self.use_reranking else top_k
        candidates = self.hybrid_search.search(query, top_k=candidate_k)
        
        if self.use_reranking and self.reranker:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        
        return candidates[:top_k]
    
    def calculate_confidence(self, retrieved_chunks: list, decision: Union[PolicyDecision, PolicyQuestion]) -> float:
        """Calculate confidence score"""
        if not retrieved_chunks:
            return 0.0
        
        retrieval_scores = [chunk.get('score', chunk.get('rerank_score', 0.0)) for chunk in retrieved_chunks[:3]]
        avg_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        retrieval_factor = avg_retrieval * 0.4
        
        if isinstance(decision, PolicyDecision):
            clarity_factor = 0.3 * 0.3 if decision.decision == "unclear" else 0.8 * 0.3
        else:
            clarity_factor = 0.6 * 0.3
        
        high_score_count = sum(1 for score in retrieval_scores if score > 0.7)
        multi_source_factor = min(high_score_count / 3, 1.0) * 0.2
        
        llm_confidence = getattr(decision, 'confidence', 0.5) * 0.1
        
        total = retrieval_factor + clarity_factor + multi_source_factor + llm_confidence
        return max(0.0, min(1.0, total))
    
    def review_ad(self, ad_text: str) -> PolicyDecision:
        """Review ad for policy compliance"""
        print(f"\n🔍 Retrieving policies...")
        policies = self.retrieve_policies(ad_text, top_k=5)
        print(f"   Found {len(policies)} policies")
        
        prompts = format_policy_review_prompt(ad_text, policies)
        
        print(f"🤖 Calling Gemini...")
        
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
        
        final_confidence = self.calculate_confidence(policies, decision)
        decision.confidence = final_confidence
        
        if final_confidence < 0.7:
            decision.escalation_required = True
        
        print(f"✅ Decision: {decision.decision} ({decision.confidence:.1%})")
        
        return decision
    
    def print_decision(self, decision: PolicyDecision):
        """Pretty print decision"""
        print("\n" + "=" * 80)
        print("📋 POLICY DECISION")
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