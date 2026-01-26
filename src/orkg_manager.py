import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from src.orkg_client import ORKGClient
from src.template_mapper import TemplateMapper
from src.llm_extractor import MultiModelResponse, LLMProperties

logger = logging.getLogger(__name__)


class ORKGPaperManager:
    """Manages the creation of papers in ORKG with extracted LLM data."""

    def __init__(self, orkg_client: ORKGClient, template_mapper: TemplateMapper):
        self.client = orkg_client
        self.mapper = template_mapper

    def process_and_upload(
        self, extraction_data: Dict[str, Any], paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Orchestrate the extraction-to-upload pipeline.

        Args:
            extraction_data: Dict containing 'raw_extraction' or just the models list
            paper_metadata: Optional metadata (if not already in extraction_data)

        Returns:
            Dict with paper_id and contribution_ids on success, None on failure
        """
        try:
            # 1. Normalize the input data
            # If the user passed the full JSON result from Grete:
            if "raw_extraction" in extraction_data:
                models_raw = extraction_data["raw_extraction"]
                # Use metadata from JSON if not provided as argument
                if not paper_metadata:
                    paper_metadata = extraction_data.get("paper_metadata")
            else:
                # Fallback if it's just the models list
                models_raw = extraction_data.get("models", extraction_data)

            if not paper_metadata:
                logger.error("Mapping failed - no paper metadata provided")
                return None

            # 4. Determine Paper Title (Priority: CLI > LLM Extracted > Fallback)
            cli_title = paper_metadata.get("title") or extraction_data.get("paper_title")
            llm_title = None
            if isinstance(models_raw, list) and len(models_raw) > 0:
                llm_title = models_raw[0].get("paper_title")

            # Use LLM title if CLI title is generic or missing
            if not cli_title or cli_title == "Unknown Paper":
                paper_title = llm_title or cli_title or "Unknown Paper"
            else:
                paper_title = cli_title

            # Robust fallback: ORKG requires a non-blank title
            if not paper_title or not str(paper_title).strip():
                if llm_title:
                    paper_title = llm_title
                elif isinstance(models_raw, list) and len(models_raw) > 0:
                    model_name = models_raw[0].get("model_name", "Unknown Model")
                    paper_title = f"LLM Extraction: {model_name}"
                else:
                    paper_title = (
                        f"Research Paper ({extraction_data.get('paper_url', 'Unknown URL')})"
                    )

            logger.info(f"Final paper title for ORKG: {paper_title}")

            # 2. Convert raw dicts to Pydantic models for mapping
            if isinstance(models_raw, list):
                models = [LLMProperties(**m) if isinstance(m, dict) else m for m in models_raw]
                extraction_result = MultiModelResponse(
                    models=models, paper_describes_multiple_models=len(models) > 1
                )
            else:
                logger.error("Invalid extraction data format - expected list of models")
                return None

            # 3. Map extraction to ORKG template structure
            mapped_data = self.mapper.map_extraction_result(extraction_result)
            if not mapped_data or not mapped_data.get("contributions"):
                logger.error("Mapping failed - no valid contributions to upload")
                return None

            # 4. Check for existing paper to avoid duplicates
            # TEMPORARILY DISABLED FOR TESTING - Always create new papers
            paper_id = None
            contribution_ids = []

            # DISABLED: Paper search logic (for testing - always create new papers)
            # existing_papers = self.client.search_papers(paper_title)
            # for paper in existing_papers:
            #     if paper.get("title", "").strip().lower() == paper_title.strip().lower():
            #         paper_id = paper.get("id")
            #         logger.info(f"Found existing paper in ORKG: {paper_id}")
            #
            #         # Fetch its contributions to check for duplicates
            #         paper_data = self.client.get_paper(paper_id)
            #         existing_contribs = paper_data.get('contributions', []) if paper_data else []
            #         existing_labels = {c.get('label', '').strip().lower() for c in existing_contribs if isinstance(c, dict)}
            #
            #         contribution_ids = [c.get('id') for c in existing_contribs if isinstance(c, dict)]
            #
            #         # Add new contributions that don't exist yet
            #         for contrib_data in mapped_data["contributions"]:
            #             label = contrib_data.get("label", "").strip()
            #             if label.lower() not in existing_labels:
            #                 logger.info(f"Adding new contribution '{label}' to existing paper {paper_id}")
            #                 new_cid = self.client.add_contribution_to_paper(paper_id, contrib_data)
            #                 if new_cid:
            #                     contribution_ids.append(new_cid)
            #                     existing_labels.add(label.lower())
            #             else:
            #                 logger.info(f"Contribution '{label}' already exists on paper {paper_id}. Skipping.")
            #         break

            # Always create new paper (for testing)
            if not paper_id:
                # 5. Create Paper with all Contributions (Step A)
                # Add unique timestamp suffix to avoid ORKG API duplicate title rejection
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_paper_title = f"{paper_title} [TEST-{timestamp}]"
                logger.info(
                    "TESTING MODE: Paper search disabled. Creating new paper in ORKG (duplicates allowed)..."
                )
                logger.info(f"Using unique title: {unique_paper_title}")
                result = self.client.create_paper_with_contributions(
                    title=unique_paper_title,
                    authors=[
                        {"name": a} if isinstance(a, str) else a
                        for a in paper_metadata.get("authors", [])
                    ],
                    publication_year=paper_metadata.get("year", 2024),
                    publication_month=paper_metadata.get("month"),
                    url=paper_metadata.get("url", extraction_data.get("paper_url", "")),
                    doi=paper_metadata.get("doi"),
                    contributions_data=mapped_data["contributions"],
                    research_field="R133",  # AI
                )

                if not result or not result.get("paper_id"):
                    logger.error("Failed to create paper in ORKG")
                    return None

                paper_id = result["paper_id"]
                contribution_ids = result.get("contribution_ids", [])

            # 6. Link to Comparison Table (Step B)
            # Use the sandbox comparison ID
            comparison_id = "R1364660"

            logger.info(
                f"Linking {len(contribution_ids)} contributions to comparison {comparison_id}"
            )
            self.client.update_comparison_with_contributions(
                comparison_id=comparison_id,
                title="Generative AI Model Landscape",
                description="A landscape of Generative AI Models extracted from research papers.",
                new_contribution_ids=contribution_ids,
                research_fields=["R133"],
                authors=[{"name": "Alaa Kefi"}],
            )

            logger.info(f"Successfully processed paper: {paper_id}")
            return {"paper_id": paper_id, "contribution_ids": contribution_ids}

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return None
