"""
Google Ads Policy Scraper
Fetches policy pages from support.google.com/adspolicy
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
import time
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm


@dataclass
class PolicyPage:
    """Represents a single policy page"""
    url: str
    title: str
    category: str
    content_html: str
    scraped_at: str
    
    def to_dict(self):
        return asdict(self)


class GoogleAdsPolicyScraper:
    """
    Scrapes Google Ads policy documentation
    
    Why we're doing this:
    - Google Ads policies are publicly available HTML pages
    - We need structured data to feed into our RAG system
    - We preserve hierarchy (category > section > subsection)
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main policy categories to scrape
        # These URLs contain the most important policy information
        self.policy_urls = {
            "prohibited_content": "https://support.google.com/adspolicy/answer/6008942",
            "restricted_content": "https://support.google.com/adspolicy/answer/176031",
            "editorial_quality": "https://support.google.com/adspolicy/answer/6021546",
            "technical_requirements": "https://support.google.com/adspolicy/answer/176032",
            "healthcare_medicines": "https://support.google.com/adspolicy/answer/176031",
            "financial_services": "https://support.google.com/adspolicy/answer/2464998",
            "political_content": "https://support.google.com/adspolicy/answer/6014595",
        }
        
        # Headers to look like a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_page(self, url: str, category: str) -> PolicyPage:
        """
        Scrape a single policy page
        
        Args:
            url: URL to scrape
            category: Policy category (e.g., "prohibited_content")
        
        Returns:
            PolicyPage object with scraped content
        """
        try:
            print(f"Scraping: {category}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('h1') or soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else category
            
            # Get main content area
            # Google support pages typically use specific classes
            content = soup.find('article') or soup.find('div', class_='article-content') or soup.body
            
            return PolicyPage(
                url=url,
                title=title,
                category=category,
                content_html=str(content),
                scraped_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Error scraping {category}: {e}")
            raise
    
    def scrape_all(self) -> List[PolicyPage]:
        """
        Scrape all policy pages
        
        Returns:
            List of PolicyPage objects
        """
        pages = []
        
        for category, url in tqdm(self.policy_urls.items(), desc="Scraping policies"):
            try:
                page = self.scrape_page(url, category)
                pages.append(page)
                
                # Be polite - don't hammer Google's servers
                time.sleep(2)
                
            except Exception as e:
                print(f"Failed to scrape {category}: {e}")
                continue
        
        return pages
    
    def save_pages(self, pages: List[PolicyPage]):
        """
        Save scraped pages to disk
        
        Why JSON and HTML?
        - JSON: Easy to load metadata
        - HTML: Preserves structure for parsing
        """
        # Save individual HTML files
        for page in pages:
            html_path = self.output_dir / f"{page.category}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(page.content_html)
        
        # Save metadata JSON
        metadata = [page.to_dict() for page in pages]
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Saved {len(pages)} policy pages to {self.output_dir}")
        print(f"📄 Metadata: {metadata_path}")
    
    def run(self):
        """Main execution method"""
        print("🚀 Starting Google Ads Policy Scraper\n")
        
        pages = self.scrape_all()
        self.save_pages(pages)
        
        print("\n✨ Scraping complete!")
        return pages


def main():
    """Run the scraper"""
    scraper = GoogleAdsPolicyScraper()
    pages = scraper.run()
    
    # Print summary
    print("\n📊 Summary:")
    for page in pages:
        print(f"  - {page.category}: {page.title}")


if __name__ == "__main__":
    main()