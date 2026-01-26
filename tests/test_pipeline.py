"""Tests for the extraction pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.pipeline import ExtractionPipeline


class TestExtractionPipeline:
    """Test cases for ExtractionPipeline."""
    
    @patch('src.pipeline.ORKGClient')
    @patch('src.pipeline.LLMExtractor')
    def test_pipeline_init(self, mock_llm, mock_orkg):
        """Test pipeline initialization."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            pipeline = ExtractionPipeline()
            
            assert pipeline.orkg_client is not None
            assert pipeline.paper_fetcher is not None
            assert pipeline.pdf_parser is not None
            assert pipeline.template_mapper is not None
            assert pipeline.comparison_updater is not None
    
    @patch('src.pipeline.ORKGClient')
    @patch('src.pipeline.LLMExtractor')
    def test_test_connection(self, mock_llm, mock_orkg):
        """Test connection testing."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            pipeline = ExtractionPipeline()
            pipeline.orkg_client.ping = Mock(return_value=True)
            
            results = pipeline.test_connection()
            
            assert results['orkg'] is True
            assert 'llm_extractor' in results
    
    @patch('src.pipeline.ORKGClient')
    @patch('src.pipeline.LLMExtractor')
    def test_get_status(self, mock_llm, mock_orkg):
        """Test status retrieval."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            pipeline = ExtractionPipeline()
            pipeline.orkg_client.ping = Mock(return_value=True)
            
            status = pipeline.get_status()
            
            assert 'orkg_host' in status
            assert 'template_id' in status
            assert 'comparison_id' in status
            assert 'connections' in status

