"""
Policy Parser - Extracts hierarchical sections from HTML

Why this matters:
- Google policy pages have nested structure (Category > Section > Subsection)
- We need to preserve this hierarchy for better retrieval
- Each chunk should know its "breadcrumb trail"
"""

from bs4 import BeautifulSoup
from pathlib import Path
import json
from typing import List, Dict
from dataclasses import dataclass, asdict


@dataclass
class PolicySection:
    """
    Represents a single policy section
    
    Example:
    hierarchy: ["Healthcare and Medicines", "Unapproved Pharmaceuticals", "Weight Loss"]
    content: "Weight loss products must not make misleading claims..."
    """
    url: str
    category: str
    hierarchy: List[str]  # Breadcrumb trail
    section_title: str
    content: str
    content_type: str  # "policy" or "example" or "exception"
    
    def to_dict(self):
        return asdict(self)


class PolicyParser:
    """
    Parses HTML policy pages into structured sections
    
    Strategy:
    1. Find all heading tags (h1, h2, h3, h4)
    2. Extract content under each heading
    3. Build hierarchy based on heading levels
    """
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_html(self, html_content: str, url: str, category: str) -> List[PolicySection]:
        """
        Parse HTML into structured sections
        
        Args:
            html_content: Raw HTML string
            url: Source URL
            category: Policy category
        
        Returns:
            List of PolicySection objects
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        sections = []
        
        # Track current hierarchy
        hierarchy_stack = []
        current_level = 0
        
        # Find all headings and their content
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        
        for i, heading in enumerate(headings):
            # Determine heading level (h1=1, h2=2, etc.)
            level = int(heading.name[1])
            
            # Update hierarchy stack based on level
            if level > current_level:
                # Going deeper
                hierarchy_stack.append(heading.get_text(strip=True))
            elif level == current_level:
                # Same level - replace last item
                if hierarchy_stack:
                    hierarchy_stack[-1] = heading.get_text(strip=True)
                else:
                    hierarchy_stack.append(heading.get_text(strip=True))
            else:
                # Going up - pop items
                while len(hierarchy_stack) > level:
                    hierarchy_stack.pop()
                if hierarchy_stack:
                    hierarchy_stack[-1] = heading.get_text(strip=True)
                else:
                    hierarchy_stack.append(heading.get_text(strip=True))
            
            current_level = level
            
            # Extract content until next heading
            content_parts = []
            current_element = heading.find_next_sibling()
            
            # Stop at next heading or end
            next_heading = headings[i + 1] if i + 1 < len(headings) else None
            
            while current_element and current_element != next_heading:
                if current_element.name in ['p', 'li', 'div']:
                    text = current_element.get_text(strip=True)
                    if text:
                        content_parts.append(text)
                current_element = current_element.find_next_sibling()
            
            # Create section if we have content
            if content_parts:
                content = '\n\n'.join(content_parts)
                
                # Detect content type
                content_lower = content.lower()
                if 'example' in content_lower or 'for instance' in content_lower:
                    content_type = 'example'
                elif 'exception' in content_lower or 'allowed' in content_lower:
                    content_type = 'exception'
                else:
                    content_type = 'policy'
                
                section = PolicySection(
                    url=url,
                    category=category,
                    hierarchy=hierarchy_stack.copy(),
                    section_title=heading.get_text(strip=True),
                    content=content,
                    content_type=content_type
                )
                sections.append(section)
        
        return sections
    
    def parse_all_pages(self) -> List[PolicySection]:
        """
        Parse all HTML files in input directory
        
        Returns:
            Combined list of all sections
        """
        metadata_path = self.input_dir / "metadata.json"
        
        if not metadata_path.exists():
            print("❌ No metadata.json found. Run scraper first!")
            return []
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        all_sections = []
        
        print("📖 Parsing policy pages...\n")
        
        for page_meta in metadata:
            category = page_meta['category']
            url = page_meta['url']
            
            html_path = self.input_dir / f"{category}.html"
            
            if not html_path.exists():
                print(f"⚠️  Skipping {category} - file not found")
                continue
            
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            sections = self.parse_html(html_content, url, category)
            all_sections.extend(sections)
            
            print(f"✓ {category}: {len(sections)} sections")
        
        return all_sections
    
    def save_sections(self, sections: List[PolicySection]):
        """
        Save parsed sections to JSON
        
        This becomes our "source of truth" for chunking
        """
        output_path = self.output_dir / "parsed_sections.json"
        
        sections_dict = [s.to_dict() for s in sections]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sections_dict, f, indent=2)
        
        print(f"\n✅ Saved {len(sections)} sections to {output_path}")
    
    def run(self):
        """Main execution"""
        print("🔍 Starting Policy Parser\n")
        
        sections = self.parse_all_pages()
        
        if sections:
            self.save_sections(sections)
            
            # Print sample
            print("\n📋 Sample section:")
            sample = sections[0]
            print(f"  Category: {sample.category}")
            print(f"  Hierarchy: {' > '.join(sample.hierarchy)}")
            print(f"  Title: {sample.section_title}")
            print(f"  Content: {sample.content[:200]}...")
        
        return sections


def main():
    parser = PolicyParser()
    sections = parser.run()
    
    # Statistics
    print("\n📊 Statistics:")
    print(f"  Total sections: {len(sections)}")
    
    categories = {}
    for section in sections:
        categories[section.category] = categories.get(section.category, 0) + 1
    
    print("\n  By category:")
    for cat, count in categories.items():
        print(f"    {cat}: {count}")


if __name__ == "__main__":
    main()