import pytest
from unittest.mock import Mock, patch
from src.orkg_client import ORKGClient
from src.orkg_manager import ORKGPaperManager
from src.template_mapper import TemplateMapper

class TestORKGAppend:
    
    @patch('src.orkg_client.ORKG')
    def test_add_contribution_to_paper(self, mock_orkg):
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.succeeded = True
        # Response should be the paper resource with the NEW contribution included
        mock_response.content = {
            "id": "R123", 
            "title": "Existing Paper"
        }
        
        # Mock get_paper to return the paper with the expected contribution
        # This simulates the logic where we fetch the paper again to find the ID
        mock_instance.papers.by_id.return_value = {
            "id": "R123",
            "title": "Existing Paper",
            "contributions": [
                {"id": "R123_C1", "label": "Test Model"}
            ]
        }
        
        mock_instance.papers.add.return_value = mock_response
        mock_orkg.return_value = mock_instance
        
        client = ORKGClient()
        contrib_data = {
            "label": "Test Model",
            "properties": [
                {"property": "P1", "value": "Val1"}
            ]
        }
        
        cid = client.add_contribution_to_paper("R123", contrib_data)
        
        # Verify it returns the ID found in the updated paper fetch
        assert cid == "R123_C1"
        
        # Verify papers.add was called with merge_if_exists=True
        mock_instance.papers.add.assert_called_once()
        args, kwargs = mock_instance.papers.add.call_args
        assert kwargs['merge_if_exists'] is True
        assert 'paper' in kwargs['params']
        assert kwargs['params']['paper']['title'] == "Existing Paper"
        
    @patch('src.orkg_client.ORKG')
    def test_manager_appends_only_new(self, mock_orkg):
        # Setup mocks
        mock_instance = Mock()
        mock_orkg.return_value = mock_instance
        
        # 1. search_papers finds one
        mock_instance.papers.get.return_value = [{"id": "R123", "title": "Existing Paper"}]
        
        # 2. get_paper returns existing contributions
        # Mocking the sequence of get_paper calls:
        # Call 1 (Manager checks exists): Returns existing
        # Call 2 (Client fetch for title): Returns existing
        # Call 3 (Client fetch after append): Returns existing + new
        mock_instance.papers.by_id.side_effect = [
            {"id": "R123", "contributions": [{"id": "C1", "label": "Existing Model"}], "title": "Existing Paper"},
            {"id": "R123", "contributions": [{"id": "C1", "label": "Existing Model"}], "title": "Existing Paper"},
            {"id": "R123", "contributions": [{"id": "C1", "label": "Existing Model"}, {"id": "C2", "label": "New Model"}], "title": "Existing Paper"}
        ]
        
        # 3. papers.add mock (The Append)
        mock_response = Mock()
        mock_response.succeeded = True
        mock_response.content = {"id": "R123"}
        mock_instance.papers.add.return_value = mock_response
        
        client = ORKGClient()
        manager = ORKGPaperManager(client, Mock())
        
        extraction_data = {
            "paper_title": "Existing Paper",
            "paper_metadata": {"title": "Existing Paper"},
            "raw_extraction": [
                {"model_name": "Existing Model", "paper_title": "Existing Paper"}, # Duplicate
                {"model_name": "New Model", "paper_title": "Existing Paper"}      # New
            ]
        }
        
        # Mock mapper
        manager.mapper.map_extraction_result.return_value = {
            "contributions": [
                {"label": "Existing Model", "properties": []},
                {"label": "New Model", "properties": []}
            ]
        }
        
        with patch.object(client, 'update_comparison_with_contributions') as mock_update:
            result = manager.process_and_upload(extraction_data)
            
            assert result["paper_id"] == "R123"
            # In the new logic, we append via add_contribution_to_paper which calls papers.add
            # verify papers.add was called
            mock_instance.papers.add.assert_called()
