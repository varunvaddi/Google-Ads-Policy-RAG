"""
Run Phase 4 - FINAL VERSION
Complete RAG system with Google Gemini
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.generation.decision_engine import GeminiPolicyEngine


def main():
    print("=" * 60)
    print("PHASE 4: COMPLETE RAG SYSTEM")
    print("=" * 60)
    print()
    
    # Initialize
    print("🚀 Initializing...")
    print("-" * 60)
    try:
        engine = GeminiPolicyEngine(model_name="gemini-2.5-flash-preview-09-2025", use_reranking=True)
    except ValueError as e:
        print(f"\n❌ {e}")
        return
    
    print()
    
    # Test cases
    test_cases = [
        {
            "name": "Misleading Health Claims",
            "ad": "Lose 15 pounds in one week with this miracle pill! Guaranteed!",
            "expected": "disallowed"
        },
        {
            "name": "Crypto Education",
            "ad": "Learn cryptocurrency trading from certified instructors.",
            "expected": "restricted"
        },
        {
            "name": "Standard Product",
            "ad": "Buy our smartphone - 5G, 128GB. Free shipping over $50.",
            "expected": "allowed"
        }
    ]
    
    print("🧪 Testing...")
    print("-" * 60)
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test['name']}")
        print('='*60)
        print(f"\n📝 Ad: \"{test['ad']}\"")
        print(f"📌 Expected: {test['expected']}")
        
        decision = engine.review_ad(test['ad'])
        engine.print_decision(decision)
        
        match = decision.decision == test['expected']
        status = "✅ PASS" if match else "⚠️  FAIL"
        print(f"\n{status} - Actual: {decision.decision}")
        
        results.append({
            'test': test['name'],
            'expected': test['expected'],
            'actual': decision.decision,
            'confidence': decision.confidence,
            'match': match
        })
    


    # Summary
    print("\n📊 PHASE IV: SUMMARY")
    print("-" * 60)
    print()
    
    for r in results:
        status = "✅" if r['match'] else "❌"
        print(f"{status} {r['test']}: {r['actual']} (expected: {r['expected']}) - {r['confidence']:.0%}")
    
    passed = sum(1 for r in results if r['match'])
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} passed ({passed/total:.0%})")
    
    # Final
    print("\n\n" + "=" * 60)
    print("✨ COMPLETE RAG SYSTEM BUILT!")
    print("=" * 60)
  


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise