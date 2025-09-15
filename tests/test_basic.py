"""
Basic tests for Therapist RAG System
"""
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests."""
    
    def test_imports(self):
        """Test that all modules can be imported."""
        try:
            from src import data_processor
            from src import embeddings
            from src import rag_engine
            from src import models
            self.assertTrue(True, "All modules imported successfully")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_data_processor_class(self):
        """Test data processor class instantiation."""
        from src.data_processor import TherapyDataProcessor
        
        processor = TherapyDataProcessor("dummy_input.csv", "dummy_output.csv")
        self.assertIsNotNone(processor)
        self.assertEqual(processor.raw_data_path, "dummy_input.csv")
        self.assertEqual(processor.processed_data_path, "dummy_output.csv")
    
    def test_embedding_manager_class(self):
        """Test embedding manager class instantiation."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager()
        self.assertIsNotNone(manager)
        self.assertEqual(manager.model_name, "all-MiniLM-L6-v2")
    
    def test_rag_engine_class(self):
        """Test RAG engine class instantiation."""
        from src.rag_engine import TherapistRAGEngine
        
        engine = TherapistRAGEngine()
        self.assertIsNotNone(engine)
        self.assertFalse(engine.is_loaded)
    
    def test_pydantic_models(self):
        """Test Pydantic models."""
        from src.models import SearchRequest, GuidanceRequest
        
        # Test SearchRequest
        search_req = SearchRequest(query="test query")
        self.assertEqual(search_req.query, "test query")
        self.assertEqual(search_req.top_k, 5)  # default value
        
        # Test GuidanceRequest
        guidance_req = GuidanceRequest(
            patient_context="Patient has depression",
            therapist_question="What approach to use?"
        )
        self.assertEqual(guidance_req.patient_context, "Patient has depression")
        self.assertEqual(guidance_req.therapist_question, "What approach to use?")

class TestDataProcessing(unittest.TestCase):
    """Data processing tests."""
    
    def test_text_cleaning(self):
        """Test text cleaning function."""
        from src.data_processor import TherapyDataProcessor
        
        processor = TherapyDataProcessor("dummy", "dummy")
        
        # Test basic cleaning
        dirty_text = "  This   has\n\nextra   spaces  and\nnewlines  "
        clean_text = processor.clean_text(dirty_text)
        self.assertEqual(clean_text, "This has extra spaces and newlines")
        
        # Test empty text
        self.assertEqual(processor.clean_text(""), "")
        self.assertEqual(processor.clean_text(None), "")
    
    def test_quality_scoring(self):
        """Test quality scoring function."""
        from src.data_processor import TherapyDataProcessor
        
        processor = TherapyDataProcessor("dummy", "dummy")
        
        # Test good quality content
        good_context = "This is a reasonable length context with good content for therapy discussion."
        good_response = "This is a helpful therapeutic response with appropriate length."
        score = processor.calculate_quality_score(good_context, good_response)
        self.assertGreater(score, 50)
        
        # Test poor quality content
        poor_context = "Short"
        poor_response = "Bad"
        score = processor.calculate_quality_score(poor_context, poor_response)
        self.assertLess(score, 50)

if __name__ == "__main__":
    unittest.main()