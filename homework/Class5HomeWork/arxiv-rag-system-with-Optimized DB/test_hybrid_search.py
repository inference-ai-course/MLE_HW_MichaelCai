#!/usr/bin/env python3
"""
Simple test script for hybrid search functionality
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_services

async def test_hybrid_search():
    """Test the hybrid search functionality"""
    
    print("Testing hybrid search functionality...")
    
    try:
        # Initialize services
        print("1. Initializing services...")
        faiss_svc, embedding_svc, db_svc, hybrid_svc = get_services()
        
        if hybrid_svc is None:
            print("❌ Hybrid search service is None")
            return False
            
        print("✅ Services initialized successfully!")
        
        # Test query
        test_query = "machine learning"
        print(f"\n2. Testing hybrid search with query: '{test_query}'")
        
        # Test different alpha values
        alpha_values = [0.2, 0.5, 0.8]
        
        for alpha in alpha_values:
            print(f"\n   Testing with alpha={alpha} (vector weight={alpha:.1%}, keyword weight={1-alpha:.1%})")
            
            try:
                results = hybrid_svc.search(test_query, limit=3, alpha=alpha)
                
                if results:
                    print(f"   ✅ Found {len(results)} results")
                    
                    # Show top result details
                    top_result = results[0]
                    print(f"   📄 Top result:")
                    print(f"      Combined score: {top_result['combined_score']:.3f}")
                    print(f"      Vector score: {top_result['vector_score']:.3f}")
                    print(f"      Keyword score: {top_result['keyword_score']:.3f}")
                    print(f"      Content preview: {top_result['content'][:100]}...")
                    
                else:
                    print(f"   ⚠️  No results found for alpha={alpha}")
                    
            except Exception as e:
                print(f"   ❌ Error with alpha={alpha}: {str(e)}")
                return False
        
        print("\n3. ✅ All hybrid search tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_hybrid_search())
    
    if success:
        print("\n🎉 Hybrid search implementation is working correctly!")
        print("\nYou can now use the following endpoints:")
        print("- GET /search_hybrid?q=your_query&limit=10&alpha=0.6")
        print("- GET /search_hybrid?q=machine learning&alpha=0.8  (more vector-focused)")
        print("- GET /search_hybrid?q=deep learning&alpha=0.2   (more keyword-focused)")
    else:
        print("\n❌ Hybrid search test failed. Please check the implementation.")
        sys.exit(1)