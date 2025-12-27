"""
Intelligent Policy Chunking

Why chunking matters:
1. Embeddings work best on 500-1000 character chunks
2. LLMs have token limits (can't send entire 100-page policy)
3. Smaller chunks = more precise retrieval

Our strategy:
- Keep small sections together (< 1000 chars)
- Split large sections but maintain context
- Preserve hierarchy information in metadata
"""

import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import tiktoken


@dataclass
class PolicyChunk:
    """
    A chunk is a piece of policy text ready for embedding
    
    Why these fields?
    - content: The actual text to embed
    - metadata: Everything needed for citation and filtering
    - chunk_id: Unique identifier
    """
    chunk_id: str
    content: str
    metadata: Dict
    char_count: int
    token_count: int
    
    def to_dict(self):
        return asdict(self)


class PolicyChunker:
    """
    Chunks parsed policy sections intelligently
    
    Strategy:
    1. Small sections (< 800 chars): Keep whole
    2. Large sections: Split on paragraphs
    3. Always preserve hierarchy in metadata
    """
    
    def __init__(
        self, 
        input_file: str = "data/processed/parsed_sections.json",
        output_file: str = "data/processed/chunks.json",
        max_chunk_chars: int = 800,
        overlap_chars: int = 100
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        
        # Token counter (for cost estimation)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens for cost estimation"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4
    
    def chunk_section(self, section: Dict) -> List[PolicyChunk]:
        """
        Chunk a single section
        
        Args:
            section: Parsed section dictionary
        
        Returns:
            List of chunks
        """
        content = section['content']
        char_count = len(content)
        
        chunks = []
        
        # Strategy 1: Small section - keep whole
        if char_count <= self.max_chunk_chars:
            chunk = self._create_chunk(
                content=content,
                section=section,
                chunk_index=0,
                is_partial=False
            )
            chunks.append(chunk)
        
        # Strategy 2: Large section - split intelligently
        else:
            # Split on double newlines (paragraphs)
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            chunk_index = 0
            
            for para in paragraphs:
                # Would adding this paragraph exceed limit?
                if len(current_chunk) + len(para) + 2 <= self.max_chunk_chars:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunk = self._create_chunk(
                            content=current_chunk,
                            section=section,
                            chunk_index=chunk_index,
                            is_partial=True
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # Start new chunk with overlap
                    # Take last sentence of previous chunk for context
                    sentences = current_chunk.split('. ')
                    overlap = sentences[-1] if sentences else ""
                    current_chunk = overlap + ". " + para if overlap else para
            
            # Don't forget the last chunk
            if current_chunk:
                chunk = self._create_chunk(
                    content=current_chunk,
                    section=section,
                    chunk_index=chunk_index,
                    is_partial=True
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self, 
        content: str, 
        section: Dict, 
        chunk_index: int,
        is_partial: bool
    ) -> PolicyChunk:
        """
        Create a PolicyChunk with metadata
        
        Why this metadata?
        - url: For citations
        - hierarchy: For breadcrumb display
        - category: For filtering
        - content_type: Distinguish policy vs examples
        - chunk_type: Is this a complete section or partial?
        """
        
        # Add hierarchical context to content
        # This helps the LLM understand where this chunk fits
        hierarchy_path = " > ".join(section['hierarchy'])
        enriched_content = f"[Policy Section: {hierarchy_path}]\n\n{content}"
        
        chunk_id = f"{section['category']}_{section['section_title'][:20]}_{chunk_index}".replace(" ", "_")
        
        metadata = {
            'url': section['url'],
            'category': section['category'],
            'hierarchy': section['hierarchy'],
            'section_title': section['section_title'],
            'content_type': section['content_type'],
            'chunk_type': 'partial' if is_partial else 'complete',
            'chunk_index': chunk_index
        }
        
        return PolicyChunk(
            chunk_id=chunk_id,
            content=enriched_content,
            metadata=metadata,
            char_count=len(enriched_content),
            token_count=self.count_tokens(enriched_content)
        )
    
    def chunk_all_sections(self) -> List[PolicyChunk]:
        """
        Process all sections into chunks
        
        Returns:
            List of all chunks
        """
        if not self.input_file.exists():
            print(f"❌ Input file not found: {self.input_file}")
            return []
        
        with open(self.input_file, 'r') as f:
            sections = json.load(f)
        
        all_chunks = []
        
        print(f"✂️  Chunking {len(sections)} sections...\n")
        
        for section in sections:
            chunks = self.chunk_section(section)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[PolicyChunk]):
        """Save chunks to JSON"""
        chunks_dict = [c.to_dict() for c in chunks]
        
        with open(self.output_file, 'w') as f:
            json.dump(chunks_dict, f, indent=2)
        
        print(f"\n✅ Saved {len(chunks)} chunks to {self.output_file}")
    
    def run(self):
        """Main execution"""
        print("✂️  Starting Policy Chunker\n")
        
        chunks = self.chunk_all_sections()
        
        if chunks:
            self.save_chunks(chunks)
            
            # Statistics
            total_chars = sum(c.char_count for c in chunks)
            total_tokens = sum(c.token_count for c in chunks)
            avg_chars = total_chars / len(chunks)
            
            print("\n📊 Chunking Statistics:")
            print(f"  Total chunks: {len(chunks)}")
            print(f"  Avg chunk size: {avg_chars:.0f} characters")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Est. embedding cost: ${(total_tokens / 1000) * 0.0001:.4f}")
            
            # Sample
            print("\n📋 Sample chunk:")
            sample = chunks[0]
            print(f"  ID: {sample.chunk_id}")
            print(f"  Hierarchy: {' > '.join(sample.metadata['hierarchy'])}")
            print(f"  Content: {sample.content[:200]}...")
        
        return chunks


def main():
    chunker = PolicyChunker()
    chunks = chunker.run()
    
    # Analysis
    if chunks:
        chunk_types = {}
        for chunk in chunks:
            ct = chunk.metadata['chunk_type']
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
        
        print("\n  By chunk type:")
        for ct, count in chunk_types.items():
            print(f"    {ct}: {count}")


if __name__ == "__main__":
    main()