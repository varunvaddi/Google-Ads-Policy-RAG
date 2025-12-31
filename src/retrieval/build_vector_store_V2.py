# Copy everything from the artifact, but replace the test_index method with this:

    def test_index(self, index: faiss.Index, embeddings: np.ndarray):
        """
        Test the index with sample queries
        
        WHAT WE TEST:
        1. Can we retrieve vectors?
        2. Are similarities in expected range?
        3. Does it return correct number of results?
        """
        print(f"\n🧪 Testing index...")
        
        # Test query: Use first embedding as query
        query = embeddings[0:1]  # Shape: (1, 1024)
        k = 5  # Get top 5 results
        
        # Search
        start = time.time()
        distances, indices = index.search(query, k)
        elapsed = time.time() - start
        
        print(f"✅ Search completed in {elapsed*1000:.2f}ms")
        print(f"\n   Top {k} results:")
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            print(f"   {i+1}. Index {idx}: similarity {dist:.4f}")
        
        # Verify: At least one result should be perfect match (query itself or duplicate)
        max_similarity = distances[0][0]
        assert max_similarity > 0.99, f"Top similarity should be ~1.0, got {max_similarity}"
        
        # Verify: All similarities should be between 0 and 1
        assert all(0 <= d <= 1 for d in distances[0]), "Similarities out of range"
        
        print(f"\n✅ Index working correctly!")
        
        # Note about duplicates
        duplicates = sum(1 for d in distances[0] if d > 0.999)
        if duplicates > 1:
            print(f"   ℹ️  Note: Found {duplicates} near-duplicate vectors")
            print(f"   This is normal if similar policy sections exist")
