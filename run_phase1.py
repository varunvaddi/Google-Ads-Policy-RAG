"""
Run Phase 1: Data Ingestion Pipeline

This script orchestrates the entire data ingestion process:
1. Scrape Google Ads policy pages
2. Parse HTML into structured sections
3. Chunk sections for embedding

Usage:
    python run_phase1.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion.scrape_policies import GoogleAdsPolicyScraper
from ingestion.parse_policies import PolicyParser
from ingestion.chunking import PolicyChunker


def main():
    """Run complete Phase 1 pipeline"""
    
    print("=" * 60)
    print("PHASE 1: DATA INGESTION PIPELINE")
    print("=" * 60)
    print()
    
    # Step 1: Scrape policies
    print("🌐 STEP 1: Scraping Google Ads Policies")
    print("-" * 60)
    scraper = GoogleAdsPolicyScraper(output_dir="data/raw")
    pages = scraper.run()
    print()
    
    if not pages:
        print("❌ Scraping failed. Exiting.")
        return
    
    # Step 2: Parse into sections
    print("📖 STEP 2: Parsing HTML into Sections")
    print("-" * 60)
    parser = PolicyParser(
        input_dir="data/raw",
        output_dir="data/processed"
    )
    sections = parser.run()
    print()
    
    if not sections:
        print("❌ Parsing failed. Exiting.")
        return
    
    # Step 3: Chunk sections
    print("✂️  STEP 3: Chunking Sections")
    print("-" * 60)
    chunker = PolicyChunker(
        input_file="data/processed/parsed_sections.json",
        output_file="data/processed/chunks.json"
    )
    chunks = chunker.run()
    print()
    
    if not chunks:
        print("❌ Chunking failed. Exiting.")
        return
    
    # Summary
    print("=" * 60)
    print("✨ PHASE 1 COMPLETE!")
    print("=" * 60)
    print()
    print("📊 Final Statistics:")
    print(f"  • Policy pages scraped: {len(pages)}")
    print(f"  • Sections extracted: {len(sections)}")
    print(f"  • Chunks created: {len(chunks)}")
    print()
    print("📁 Output files:")
    print("  • data/raw/metadata.json")
    print("  • data/raw/*.html")
    print("  • data/processed/parsed_sections.json")
    print("  • data/processed/chunks.json")
    print()
    print("🚀 Next step: Run Phase 2 (Embeddings & Vector Store)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise