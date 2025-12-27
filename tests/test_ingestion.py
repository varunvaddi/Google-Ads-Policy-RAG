"""
Tests for data ingestion pipeline

Run with: pytest tests/test_ingestion.py -v
"""

import pytest
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.scrape_policies import GoogleAdsPolicyScraper, PolicyPage
from ingestion.parse_policies import PolicyParser, PolicySection
from ingestion.chunking import PolicyChunker, PolicyChunk


class TestScraper:
    """Test policy scraper"""
    
    def test_scraper_initialization(self):
        """Test that scraper initializes correctly"""
        scraper = GoogleAdsPolicyScraper(output_dir="data/raw")
        assert scraper.output_dir.exists()
        assert len(scraper.policy_urls) > 0
    
    def test_policy_page_dataclass(self):
        """Test PolicyPage structure"""
        page = PolicyPage(
            url="https://example.com",
            title="Test Policy",
            category="test",
            content_html="<p>Test content</p>",
            scraped_at="2024-01-01T00:00:00"
        )
        assert page.url == "https://example.com"
        assert page.title == "Test Policy"


class TestParser:
    """Test policy parser"""
    
    def test_parser_initialization(self):
        """Test parser setup"""
        parser = PolicyParser(
            input_dir="data/raw",
            output_dir="data/processed"
        )
        assert parser.output_dir.exists()
    
    def test_section_dataclass(self):
        """Test PolicySection structure"""
        section = PolicySection(
            url="https://example.com",
            category="healthcare",
            hierarchy=["Healthcare", "Weight Loss"],
            section_title="Weight Loss Products",
            content="Test content",
            content_type="policy"
        )
        assert section.hierarchy == ["Healthcare", "Weight Loss"]
        assert section.content_type == "policy"


class TestChunker:
    """Test chunking logic"""
    
    def test_chunker_initialization(self):
        """Test chunker setup"""
        chunker = PolicyChunker(
            input_file="data/processed/parsed_sections.json",
            output_file="data/processed/chunks.json"
        )
        assert chunker.max_chunk_chars == 800
        assert chunker.overlap_chars == 100
    
    def test_small_section_chunking(self):
        """Test that small sections stay intact"""
        chunker = PolicyChunker()
        
        small_section = {
            'url': 'https://example.com',
            'category': 'test',
            'hierarchy': ['Test', 'Section'],
            'section_title': 'Small Section',
            'content': 'This is a small section with less than 800 characters.',
            'content_type': 'policy'
        }
        
        chunks = chunker.chunk_section(small_section)
        
        # Should create exactly 1 chunk
        assert len(chunks) == 1
        assert chunks[0].metadata['chunk_type'] == 'complete'
    
    def test_large_section_chunking(self):
        """Test that large sections get split"""
        chunker = PolicyChunker(max_chunk_chars=100)
        
        large_section = {
            'url': 'https://example.com',
            'category': 'test',
            'hierarchy': ['Test', 'Section'],
            'section_title': 'Large Section',
            'content': 'Paragraph one. ' * 50 + '\n\n' + 'Paragraph two. ' * 50,
            'content_type': 'policy'
        }
        
        chunks = chunker.chunk_section(large_section)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        assert all(c.metadata['chunk_type'] == 'partial' for c in chunks)
    
    def test_chunk_metadata_preservation(self):
        """Test that metadata is preserved in chunks"""
        chunker = PolicyChunker()
        
        section = {
            'url': 'https://support.google.com/test',
            'category': 'healthcare',
            'hierarchy': ['Healthcare', 'Weight Loss'],
            'section_title': 'Test Section',
            'content': 'Test content',
            'content_type': 'policy'
        }
        
        chunks = chunker.chunk_section(section)
        chunk = chunks[0]
        
        assert chunk.metadata['url'] == section['url']
        assert chunk.metadata['category'] == section['category']
        assert chunk.metadata['hierarchy'] == section['hierarchy']


class TestEndToEnd:
    """Test full pipeline integration"""
    
    def test_pipeline_outputs_exist(self):
        """Test that all pipeline stages create expected files"""
        # This assumes you've run the pipeline
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        
        # Note: These will fail until you actually run the pipeline
        # That's expected - they're here to remind you what to check
        assert raw_dir.exists(), "Run scraper first"
        assert processed_dir.exists(), "Run parser first"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])