"""
Comparison Updater

This module handles updating ORKG comparisons with new model data.
"""

import logging
from typing import Dict, List, Any, Optional
from src.orkg_client import ORKGClient
from src.template_mapper import TemplateMapper

logger = logging.getLogger(__name__)


class ComparisonUpdater:
    """Updates ORKG comparisons with new LLM data."""
    
    def __init__(
        self,
        orkg_client: ORKGClient,
        template_mapper: TemplateMapper,
        comparison_id: str = "R1364660"
    ):
        """
        Initialize comparison updater.
        
        Args:
            orkg_client: ORKG client instance
            template_mapper: Template mapper instance
            comparison_id: ORKG comparison ID
        """
        self.orkg_client = orkg_client
        self.template_mapper = template_mapper
        self.comparison_id = comparison_id
        
        logger.info(f"Initialized ComparisonUpdater for comparison: {comparison_id}")
    
    def check_model_exists(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        parameters: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a model already exists in the comparison.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            parameters: Model parameters (e.g., "7B")
        
        Returns:
            Existing contribution if found, None otherwise
        """
        logger.info(f"Checking if model exists: {model_name}")
        
        # Get existing contributions
        contributions = self.orkg_client.get_comparison_contributions(self.comparison_id)
        
        for contribution in contributions:
            # Check if model matches
            contrib_label = contribution.get("label", "")
            
            # Simple matching based on name
            if model_name.lower() in contrib_label.lower():
                # Check version and parameters if provided
                if model_version and model_version not in contrib_label:
                    continue
                if parameters and parameters not in contrib_label:
                    continue
                
                logger.info(f"Found existing model: {contrib_label}")
                return contribution
        
        logger.info(f"Model not found: {model_name}")
        return None
    
    def add_model_to_comparison(
        self,
        contribution_data: Dict[str, Any],
        paper_id: Optional[str] = None,
        check_duplicate: bool = True
    ) -> Optional[str]:
        """
        Add a new model to the comparison.
        
        Args:
            contribution_data: Contribution data in ORKG format
            paper_id: ORKG paper ID
            check_duplicate: Whether to check for duplicates
        
        Returns:
            Contribution ID if successful, None otherwise
        """
        model_label = contribution_data.get("label", "Unknown")
        logger.info(f"Adding model to comparison: {model_label}")
        
        # Check for duplicates if requested
        if check_duplicate:
            existing = self.check_model_exists(model_label)
            if existing:
                logger.warning(f"Model already exists: {model_label}")
                return existing.get("id")
        
        # Validate contribution data
        validation = self.template_mapper.validate_mapping(contribution_data)
        if not validation["valid"]:
            logger.error(f"Invalid contribution data: {validation['errors']}")
            return None
        
        # Add contribution to comparison
        contribution_id = self.orkg_client.add_contribution(
            comparison_id=self.comparison_id,
            paper_id=paper_id,
            contribution_data=contribution_data
        )
        
        if contribution_id:
            logger.info(f"Successfully added model: {model_label} (ID: {contribution_id})")
        else:
            logger.error(f"Failed to add model: {model_label}")
        
        return contribution_id
    
    def update_existing_model(
        self,
        contribution_id: str,
        contribution_data: Dict[str, Any]
    ) -> bool:
        """
        Update an existing model in the comparison.
        
        Args:
            contribution_id: ORKG contribution ID
            contribution_data: Updated contribution data
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating contribution: {contribution_id}")
        
        # Validate contribution data
        validation = self.template_mapper.validate_mapping(contribution_data)
        if not validation["valid"]:
            logger.error(f"Invalid contribution data: {validation['errors']}")
            return False
        
        # Update contribution
        success = self.orkg_client.update_contribution(
            contribution_id=contribution_id,
            contribution_data=contribution_data
        )
        
        if success:
            logger.info(f"Successfully updated contribution: {contribution_id}")
        else:
            logger.error(f"Failed to update contribution: {contribution_id}")
        
        return success
    
    def add_or_update_model(
        self,
        contribution_data: Dict[str, Any],
        paper_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Add model if new, or update if exists.
        
        Args:
            contribution_data: Contribution data
            paper_id: ORKG paper ID
        
        Returns:
            Contribution ID
        """
        model_label = contribution_data.get("label", "Unknown")
        
        # Check if model exists
        existing = self.check_model_exists(model_label)
        
        if existing:
            # Update existing model
            contribution_id = existing.get("id")
            success = self.update_existing_model(contribution_id, contribution_data)
            return contribution_id if success else None
        else:
            # Add new model
            return self.add_model_to_comparison(
                contribution_data,
                paper_id,
                check_duplicate=False
            )
    
    def add_multiple_models(
        self,
        contributions: List[Dict[str, Any]],
        paper_id: Optional[str] = None,
        check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Add multiple models to the comparison.
        
        Args:
            contributions: List of contribution data
            paper_id: ORKG paper ID
            check_duplicates: Whether to check for duplicates
        
        Returns:
            Summary of results
        """
        logger.info(f"Adding {len(contributions)} models to comparison")
        
        results = {
            "added": [],
            "updated": [],
            "failed": [],
            "skipped": []
        }
        
        for contribution in contributions:
            model_label = contribution.get("label", "Unknown")
            
            try:
                if check_duplicates:
                    contribution_id = self.add_or_update_model(contribution, paper_id)
                    if contribution_id:
                        # Determine if it was added or updated
                        existing = self.check_model_exists(model_label)
                        if existing and existing.get("id") == contribution_id:
                            results["updated"].append({
                                "label": model_label,
                                "id": contribution_id
                            })
                        else:
                            results["added"].append({
                                "label": model_label,
                                "id": contribution_id
                            })
                    else:
                        results["failed"].append(model_label)
                else:
                    contribution_id = self.add_model_to_comparison(
                        contribution,
                        paper_id,
                        check_duplicate=False
                    )
                    if contribution_id:
                        results["added"].append({
                            "label": model_label,
                            "id": contribution_id
                        })
                    else:
                        results["failed"].append(model_label)
                        
            except Exception as e:
                logger.error(f"Error processing model {model_label}: {e}")
                results["failed"].append(model_label)
        
        logger.info(
            f"Results: {len(results['added'])} added, "
            f"{len(results['updated'])} updated, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    def create_paper_if_needed(
        self,
        paper_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create paper in ORKG if it doesn't exist.
        
        Args:
            paper_metadata: Paper metadata (title, authors, etc.)
        
        Returns:
            ORKG paper ID
        """
        title = paper_metadata.get("title")
        arxiv_id = paper_metadata.get("arxiv_id")
        
        if not title:
            logger.error("Paper title is required")
            return None
        
        logger.info(f"Creating paper in ORKG: {title}")
        
        # Search for existing paper
        if arxiv_id:
            existing_papers = self.orkg_client.search_papers(arxiv_id, size=5)
            for paper in existing_papers:
                if paper.get("title", "").lower() == title.lower():
                    logger.info(f"Paper already exists: {paper.get('id')}")
                    return paper.get("id")
        
        # Create new paper
        paper_id = self.orkg_client.create_paper(
            title=title,
            authors=paper_metadata.get("authors"),
            publication_year=self._extract_year(paper_metadata.get("published")),
            url=paper_metadata.get("pdf_url")
        )
        
        if paper_id:
            logger.info(f"Created paper: {paper_id}")
        else:
            logger.error("Failed to create paper")
        
        return paper_id
    
    def _extract_year(self, date_string: Optional[str]) -> Optional[int]:
        """
        Extract year from date string.
        
        Args:
            date_string: Date string (ISO format)
        
        Returns:
            Year as integer
        """
        if not date_string:
            return None
        
        try:
            # Try to extract year from ISO format
            year = int(date_string[:4])
            return year
        except (ValueError, IndexError):
            return None
    
    def _extract_month(self, date_string: Optional[str]) -> Optional[int]:
        """
        Extract month from date string.
        
        Args:
            date_string: Date string (ISO format)
        
        Returns:
            Month as integer (1-12)
        """
        if not date_string:
            return None
        
        try:
            # Try to extract month from ISO format (YYYY-MM-DD)
            month = int(date_string[5:7])
            return month if 1 <= month <= 12 else None
        except (ValueError, IndexError):
            return None
    
    def process_extraction_result(
        self,
        mapped_result: Dict[str, Any],
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process complete extraction result and create paper with contributions.
        
        Workflow:
        1. Check if paper already exists - if yes, skip (return skipped status)
        2. Create paper with all contributions embedded (papers.add)
        3. Extract contribution IDs from response
        
        Note: Comparison update is skipped because ORKG Python client doesn't support
        updating comparisons. Contributions must be added to comparisons manually
        via the ORKG web interface.
        
        Args:
            mapped_result: Mapped extraction result from TemplateMapper
            paper_metadata: Optional paper metadata
        
        Returns:
            Processing results with paper_id, contribution_ids, and URLs for verification
        """
        logger.info("Processing extraction result for ORKG update")
        
        if not paper_metadata:
            logger.error("Paper metadata is required for ORKG update")
            return {"status": "failed", "error": "Missing paper metadata"}
        
        # Extract paper metadata
        paper_title = paper_metadata.get("title", "Unnamed Paper")
        arxiv_id = paper_metadata.get("arxiv_id", "")
        
        # Check if paper already exists - skip if it does
        existing_papers = self.orkg_client.search_papers(paper_title, size=5)
        for paper in existing_papers:
            if paper.get("title", "").strip().lower() == paper_title.strip().lower():
                existing_paper_id = paper.get("id")
                logger.info(f"Paper already exists: {existing_paper_id} - SKIPPING (will only create new papers)")
                
                # Skip existing papers - return early
                return {
                    "status": "skipped",
                    "reason": "Paper already exists",
                    "paper_id": existing_paper_id,
                    "paper_url": f"https://sandbox.orkg.org/resource/{existing_paper_id}",
                    "message": "Paper already exists in ORKG. Use a different paper or modify the title to test paper creation."
                }
        
        paper_authors = paper_metadata.get("authors", [])
        publication_year = self._extract_year(paper_metadata.get("published"))
        publication_month = self._extract_month(paper_metadata.get("published"))
        paper_url = paper_metadata.get("pdf_url")
        paper_doi = paper_metadata.get("doi")
        
        # Format authors for ORKG
        orkg_authors = [{"name": author if isinstance(author, str) else author.get("name", "")} 
                       for author in paper_authors if author]
        
        # Prepare contributions data for paper creation
        contributions_for_paper = mapped_result.get("contributions", [])
        
        # Limit contributions to avoid API limits
        # For papers with detailed properties, even 20 can be too large
        MAX_CONTRIBUTIONS_PER_PAPER = 15
        if len(contributions_for_paper) > MAX_CONTRIBUTIONS_PER_PAPER:
            logger.warning(f"Too many contributions ({len(contributions_for_paper)}), limiting to {MAX_CONTRIBUTIONS_PER_PAPER}")
            contributions_for_paper = contributions_for_paper[:MAX_CONTRIBUTIONS_PER_PAPER]
        
        # Step 1: Create paper with all its contributions embedded
        logger.info(f"Creating paper with {len(contributions_for_paper)} contributions")
        paper_result = self.orkg_client.create_paper_with_contributions(
            title=paper_title,
            authors=orkg_authors,
            publication_year=publication_year if publication_year else 2024,
            url=paper_url,
            contributions_data=contributions_for_paper,
            doi=paper_doi,
            publication_month=publication_month,
            research_field="R133",  # AI research field
            observatories=[],
            organizations=[],
        )
        
        if not paper_result:
            logger.error("Failed to create paper with contributions in ORKG")
            return {
                "status": "failed",
                "error": "Failed to create paper",
                "paper_id": None,
                "comparison_id": self.comparison_id
            }
        
        paper_id = paper_result.get("paper_id")
        contribution_ids = paper_result.get("contribution_ids", [])
        
        logger.info(f"Created paper {paper_id} with {len(contribution_ids)} contributions")
        
        # Note: Comparison update is skipped because ORKG Python client doesn't support updating comparisons
        # Contributions are successfully created and linked to the paper
        # They can be manually added to comparisons via the ORKG web interface
        
        return {
            "status": "success",
            "paper_id": paper_id,
            "paper_url": f"https://sandbox.orkg.org/resource/{paper_id}",
            "models_extracted": len(contributions_for_paper),
            "contributions_created": len(contribution_ids),
            "contribution_ids": contribution_ids,
            "contribution_urls": [f"https://sandbox.orkg.org/resource/{cid}" for cid in contribution_ids],
            "added": len(contribution_ids),
            "failed": 0,
            "note": "Paper and contributions created successfully. Comparison update skipped (not supported by ORKG Python client)."
        }
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Get summary of current comparison state.
        
        Returns:
            Comparison summary
        """
        comparison = self.orkg_client.get_comparison(self.comparison_id)
        contributions = self.orkg_client.get_comparison_contributions(self.comparison_id)
        
        return {
            "comparison_id": self.comparison_id,
            "label": comparison.get("label") if comparison else None,
            "total_contributions": len(contributions),
            "contribution_labels": [c.get("label") for c in contributions]
        }

