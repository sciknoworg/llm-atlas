"""Tests for LLM extractor."""

import pytest
from unittest.mock import Mock, patch
from src.llm_extractor import LLMExtractor, LLMProperties, MultiModelResponse


class TestLLMExtractor:
    """Test cases for LLMExtractor."""
    
    def test_llm_properties_model(self):
        """Test LLMProperties Pydantic model."""
        props = LLMProperties(
            model_name="Test Model",
            parameters="7B",
            pretraining_architecture="Transformer",
        )

        assert props.model_name == "Test Model"
        assert props.parameters == "7B"
        assert props.pretraining_architecture == "Transformer"
    
    def test_multi_model_response(self):
        """Test MultiModelResponse model."""
        model1 = LLMProperties(model_name="Model 1", parameters="7B")
        model2 = LLMProperties(model_name="Model 2", parameters="13B")
        
        response = MultiModelResponse(
            models=[model1, model2],
            paper_describes_multiple_models=True
        )
        
        assert len(response.models) == 2
        assert response.paper_describes_multiple_models is True
    
    @patch('src.llm_extractor.OpenAI')
    def test_extractor_init(self, mock_openai):
        """Test extractor initialization."""
        extractor = LLMExtractor(api_key="test_key", model="gpt-4o")

        assert extractor.model_name == "gpt-4o"
        assert extractor.temperature == 0.0
        mock_openai.assert_called_once()
    
    @patch('src.llm_extractor.OpenAI')
    def test_deduplicate_models(self, mock_openai):
        """Test model deduplication."""
        extractor = LLMExtractor(api_key="test_key")

        model1 = LLMProperties(
            model_name="Test",
            model_version="1.0",
            parameters="7B",
            pretraining_architecture="Transformer",
        )
        model2 = LLMProperties(
            model_name="Test",
            model_version="1.0",
            parameters="7B",
            organization="TestOrg",
        )

        deduplicated = extractor._deduplicate_models([model1, model2])

        assert len(deduplicated) == 1
        assert deduplicated[0].pretraining_architecture == "Transformer"
        assert deduplicated[0].organization == "TestOrg"
    
    @patch('src.llm_extractor.OpenAI')
    def test_validate_extraction(self, mock_openai):
        """Test extraction validation."""
        extractor = LLMExtractor(api_key="test_key")
        
        model = LLMProperties(
            model_name="Test Model",
            parameters="7B",
            context_length="4096"
        )
        
        response = MultiModelResponse(
            models=[model],
            paper_describes_multiple_models=False
        )
        
        report = extractor.validate_extraction(response)
        
        assert report["valid"] is True
        assert len(report["errors"]) == 0
    
    @patch('src.llm_extractor.OpenAI')
    def test_validate_extraction_missing_name(self, mock_openai):
        """Test validation with missing model name."""
        extractor = LLMExtractor(api_key="test_key")
        
        model = LLMProperties(
            model_name="",  # Missing name
            parameters="7B"
        )
        
        response = MultiModelResponse(
            models=[model],
            paper_describes_multiple_models=False
        )
        
        report = extractor.validate_extraction(response)
        
        assert report["valid"] is False
        assert len(report["errors"]) > 0

