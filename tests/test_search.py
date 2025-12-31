"""
Interactive Policy Search
Try different queries and see what policies match!
"""

from src.retrieval.search import PolicySearch

def main():
    print("=" * 80)
    print("🔍 GOOGLE ADS POLICY SEARCH")
    print("=" * 80)
    print("\nInitializing search system...")
    
    search = PolicySearch()
    
    print("\n✅ Ready! Type your queries below (or 'quit' to exit)\n")
    
    while True:
        query = input("🔎 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # Search
        results = search.search(query, top_k=3)
        
        # Display results
        print("\n" + "─" * 80)
        print(f"📋 Top {len(results)} Results:")
        print("─" * 80)
        
        for result in results:
            hierarchy = " > ".join(result['metadata']['hierarchy'])
            print(f"\n🏆 Rank #{result['rank']} | Score: {result['score']:.4f}")
            print(f"📂 {hierarchy}")
            print(f"\n{result['content'][:300]}...")
            print()

if __name__ == "__main__":
    main()
