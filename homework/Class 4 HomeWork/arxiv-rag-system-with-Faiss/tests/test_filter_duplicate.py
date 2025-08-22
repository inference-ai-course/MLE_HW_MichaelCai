import unittest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.filter_duplicate import text_similarity, text_similarity_edit_distance, filter_duplicate_chunks

class TestFilterDuplicate(unittest.TestCase):
    
    def test_text_similarity_identical(self):
        """Test similarity of identical texts."""
        text1 = "This is a test sentence."
        text2 = "This is a test sentence."
        similarity = text_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)
    
    def test_text_similarity_different(self):
        """Test similarity of completely different texts."""
        text1 = "This is a test sentence."
        text2 = "Completely different content here."
        similarity = text_similarity(text1, text2)
        self.assertLess(similarity, 0.5)
    
    def test_text_similarity_case_insensitive(self):
        """Test that similarity is case insensitive."""
        text1 = "This is a TEST sentence."
        text2 = "this is a test sentence."
        similarity = text_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)
    
    def test_text_similarity_whitespace(self):
        """Test that similarity handles whitespace correctly."""
        text1 = "  This is a test sentence.  "
        text2 = "This is a test sentence."
        similarity = text_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)
    
    def test_text_similarity_edit_distance_available(self):
        """Test edit distance similarity if package is available."""
        try:
            import editdistance
            text1 = "This is a test"
            text2 = "This is a test"
            similarity = text_similarity_edit_distance(text1, text2)
            self.assertEqual(similarity, 1.0)
        except ImportError:
            # Skip if editdistance not installed
            self.skipTest("editdistance package not available")
    
    def test_text_similarity_edit_distance_not_available(self):
        """Test that edit distance raises error when package not available."""
        # Mock the editdistance module to be None
        import utils.filter_duplicate
        original_editdistance = utils.filter_duplicate.editdistance
        utils.filter_duplicate.editdistance = None
        
        try:
            with self.assertRaises(ImportError):
                text_similarity_edit_distance("test1", "test2")
        finally:
            # Restore original state
            utils.filter_duplicate.editdistance = original_editdistance
    
    def test_filter_duplicate_chunks_no_duplicates(self):
        """Test filtering when there are no duplicates."""
        new_chunks = ["First chunk", "Second chunk", "Third chunk"]
        existing_chunks = ["Different chunk", "Another different chunk"]
        
        result = filter_duplicate_chunks(new_chunks, existing_chunks, threshold=0.8)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, new_chunks)
    
    def test_filter_duplicate_chunks_with_duplicates(self):
        """Test filtering when there are duplicates."""
        new_chunks = ["This is chunk one", "This is chunk two", "This is chunk one"]
        existing_chunks = ["This is chunk one", "Some other chunk"]
        
        result = filter_duplicate_chunks(new_chunks, existing_chunks, threshold=0.9)
        # Should filter out the duplicate "This is chunk one" chunks
        self.assertLess(len(result), len(new_chunks))
        self.assertIn("This is chunk two", result)
    
    def test_filter_duplicate_chunks_empty_existing(self):
        """Test filtering with empty existing chunks."""
        new_chunks = ["First chunk", "Second chunk"]
        existing_chunks = []
        
        result = filter_duplicate_chunks(new_chunks, existing_chunks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, new_chunks)
    
    def test_filter_duplicate_chunks_empty_new(self):
        """Test filtering with empty new chunks."""
        new_chunks = []
        existing_chunks = ["Existing chunk"]
        
        result = filter_duplicate_chunks(new_chunks, existing_chunks)
        self.assertEqual(len(result), 0)
        self.assertEqual(result, [])
    
    def test_filter_duplicate_chunks_threshold(self):
        """Test that threshold parameter works correctly."""
        new_chunks = ["This is a test sentence"]
        existing_chunks = ["This is a test sentence!"]  # Very similar but not identical
        
        # With high threshold, should be considered duplicate
        result_high = filter_duplicate_chunks(new_chunks, existing_chunks, threshold=0.9)
        self.assertEqual(len(result_high), 0)
        
        # With low threshold, should not be considered duplicate
        result_low = filter_duplicate_chunks(new_chunks, existing_chunks, threshold=0.99)
        self.assertEqual(len(result_low), 1)


if __name__ == '__main__':
    # Run the tests
    unittest.main()