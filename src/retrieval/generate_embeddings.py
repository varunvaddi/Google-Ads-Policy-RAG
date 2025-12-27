"""
Embedding Generator - Converts text chunks into numerical vectors

CONCEPT EXPLANATION:
- An embedding is a list of numbers that represents text meaning
- Similar texts → similar numbers
- We use BGE-large-en-v1.5 (one of the best open-source models)
- Each chunk becomes a 1024-dimensional vector

WHY THIS MATTERS:
- Enables semantic search (meaning-based, not keyword-based)
- "crypto" and "cryptocurrency" will have similar embeddings
- Allows us to find relevant chunks even with different wording
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Generates embeddings for policy chunks
    
    How it works:
    1. Load a pre-trained embedding model (BGE)
    2. Feed text into model
    3. Model outputs 1024 numbers (the embedding)
    4. Save embeddings for later use
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        input_file: str = "data/processed/chunks.json",
        output_dir: str = "data/embeddings"
    ):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Which embedding model to use
                - BGE-large is one of the best open-source models
                - Produces 1024-dimensional embeddings
                - Trained on diverse text for general understanding
            
            input_file: Where to load chunks from
            output_dir: Where to save embeddings
        """
        self.model_name = model_name
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the embedding model
        # This downloads ~1.3GB model on first run (cached after)
        print(f"Loading embedding model: {model_name}")
        print("(First run will download ~1.3GB - this is cached for future use)")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def load_chunks(self) -> List[Dict]:
        """
        Load chunks from Phase 1
        
        Returns:
            List of chunk dictionaries
        """
        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.input_file}\n"
                "Did you run Phase 1 (run_phase1.py)?"
            )
        
        with open(self.input_file, 'r') as f:
            chunks = json.load(f)
        
        print(f"📚 Loaded {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert texts to embeddings
        
        TECHNICAL DETAILS:
        - Processes in batches for efficiency (32 at a time)
        - Uses GPU if available, otherwise CPU
        - Shows progress bar so you know it's working
        
        Args:
            texts: List of text strings to embed
            batch_size: How many to process at once
                - Larger = faster but more memory
                - 32 is a good balance
        
        Returns:
            numpy array of shape (num_texts, 1024)
            Each row is one embedding (1024 numbers)
        
        Example:
            texts = ["hello world", "crypto policy"]
            embeddings = generate_embeddings(texts)
            # embeddings.shape = (2, 1024)
            # embeddings[0] = [0.23, -0.45, ...]  (1024 numbers)
        """
        print(f"\n🔢 Generating embeddings for {len(texts)} chunks...")
        print(f"   Batch size: {batch_size}")
        print(f"   This will take ~{len(texts) * 0.05:.1f} seconds")
        
        # The actual embedding generation
        # normalize_embeddings=True means all vectors have length 1
        # This makes similarity calculations more stable
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # Important for FAISS!
            convert_to_numpy=True
        )
        
        print(f"✅ Generated embeddings: shape {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Save embeddings and metadata to disk
        
        We save TWO files:
        1. embeddings.npy - The actual vectors (binary, fast to load)
        2. metadata.json - Chunk info (for citations, filtering)
        
        Why separate files?
        - .npy is compact and fast for numpy arrays
        - .json is human-readable for metadata
        - FAISS only needs the vectors, not metadata
        """
        # Save embeddings as numpy array
        embeddings_path = self.output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        # Save metadata (everything except the content text to save space)
        metadata = []
        for chunk in chunks:
            # Keep only essential fields
            meta = {
                'chunk_id': chunk['chunk_id'],
                'metadata': chunk['metadata'],
                'char_count': chunk['char_count'],
                # Store first 200 chars of content for preview
                'content_preview': chunk['content'][:200] + "..."
            }
            metadata.append(meta)
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Saved:")
        print(f"   Embeddings: {embeddings_path} ({embeddings.nbytes / 1024 / 1024:.2f} MB)")
        print(f"   Metadata: {metadata_path}")
    
    def run(self):
        """
        Main execution: Load chunks → Generate embeddings → Save
        
        This is the orchestrator method that does everything
        """
        print("=" * 60)
        print("PHASE 2: EMBEDDING GENERATION")
        print("=" * 60)
        print()
        
        # Step 1: Load chunks from Phase 1
        chunks = self.load_chunks()
        
        # Step 2: Extract just the text content for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Step 3: Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Step 4: Save everything
        self.save_embeddings(embeddings, chunks)
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("✨ EMBEDDING GENERATION COMPLETE!")
        print("=" * 60)
        print(f"\n📊 Statistics:")
        print(f"   Total chunks embedded: {len(chunks)}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   Total vectors: {embeddings.shape[0]}")
        print(f"   Memory size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        print(f"\n🎯 Next: Build vector store (FAISS index)")
        
        return embeddings, chunks


def main():
    """Run the embedding generator"""
    generator = EmbeddingGenerator()
    embeddings, chunks = generator.run()
    
    # Demo: Show what an embedding looks like
    print("\n📋 Sample embedding (first 10 dimensions):")
    print(f"   Chunk: {chunks[0]['content'][:80]}...")
    print(f"   Embedding: {embeddings[0][:10]}")
    print(f"   Full size: {len(embeddings[0])} dimensions")


if __name__ == "__main__":
    main()