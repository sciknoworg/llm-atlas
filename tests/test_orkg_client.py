"""Tests for ORKG client wrapper."""

import pytest
from unittest.mock import Mock, patch
from src.orkg_client import ORKGClient


class TestORKGClient:
    """Test cases for ORKGClient."""
    
    @patch('src.orkg_client.ORKG')
    def test_init_without_auth(self, mock_orkg):
        """Test initialization without authentication."""
        client = ORKGClient(host="sandbox")
        assert client.host == "sandbox"
        assert client.timeout == 30
        mock_orkg.assert_called_once()
    
    @patch('src.orkg_client.ORKG')
    def test_init_with_auth(self, mock_orkg):
        """Test initialization with authentication."""
        client = ORKGClient(
            host="sandbox",
            email="test@example.com",
            password="password"
        )
        assert client.host == "sandbox"
        mock_orkg.assert_called_once()
    
    @patch('src.orkg_client.ORKG')
    def test_ping_success(self, mock_orkg):
        """Test successful ping."""
        mock_instance = Mock()
        mock_instance.ping.return_value = True
        mock_orkg.return_value = mock_instance
        
        client = ORKGClient()
        assert client.ping() is True
    
    @patch('src.orkg_client.ORKG')
    def test_ping_failure(self, mock_orkg):
        """Test failed ping."""
        mock_instance = Mock()
        mock_instance.ping.side_effect = Exception("Connection error")
        mock_orkg.return_value = mock_instance
        
        client = ORKGClient()
        assert client.ping() is False
    
    @patch('src.orkg_client.ORKG')
    def test_get_template(self, mock_orkg):
        """Test fetching template."""
        mock_instance = Mock()
        mock_template = {"id": "R609825", "label": "LLM Template"}
        mock_response = Mock()
        mock_response.content = mock_template
        mock_instance.resources.by_id.return_value = mock_response
        mock_orkg.return_value = mock_instance

        client = ORKGClient()
        template = client.get_template("R609825")
        assert template == mock_template
    
    @patch('src.orkg_client.ORKG')
    def test_get_comparison(self, mock_orkg):
        """Test fetching comparison."""
        mock_instance = Mock()
        mock_comparison = {"id": "R1364660", "label": "AI Models"}
        mock_response = Mock()
        mock_response.content = mock_comparison
        mock_instance.resources.by_id.return_value = mock_response
        mock_orkg.return_value = mock_instance

        client = ORKGClient()
        comparison = client.get_comparison("R1364660")
        assert comparison == mock_comparison
    
    @patch('src.orkg_client.ORKG')
    def test_search_papers(self, mock_orkg):
        """Test paper search."""
        mock_instance = Mock()
        mock_papers = [
            {"id": "R1", "title": "Paper 1"},
            {"id": "R2", "title": "Paper 2"},
        ]
        mock_response = Mock()
        mock_response.content = mock_papers
        mock_instance.papers.get.return_value = mock_response
        mock_orkg.return_value = mock_instance

        client = ORKGClient()
        papers = client.search_papers("LLM")
        assert len(papers) == 2

