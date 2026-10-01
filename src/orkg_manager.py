import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.llm_extractor import LLMProperties, MultiModelResponse
from src.orkg_client import ORKGClient
from src.template_mapper import TemplateMapper

logger = logging.getLogger(__name__)


class ORKGPaperManager:
    """Manages the creation of papers in ORKG with extracted LLM data."""

    def __init__(
        self,
        orkg_client: ORKGClient,
        template_mapper: TemplateMapper,
        comparison_id: str = "R1364660",
        comparison_title: Optional[str] = None,
        comparison_description: Optional[str] = None,
        update_comparison: bool = False,
    ):
        self.client = orkg_client
        self.mapper = template_mapper
        # Comparison to attach contributions to — the id must match the target
        # ORKG instance (sandbox vs live), so it comes from config.
        self.comparison_id = comparison_id
        # Gate for the comparison step. Off by default: only LIVE comparisons can
        # be updated, so pointing this at a published snapshot fails, and the
        # paper upload itself is useful without it.
        self.update_comparison = update_comparison
        self.comparison_title = comparison_title or "Generative AI Model Landscape"
        self.comparison_description = (
            comparison_description
            or "A landscape of Generative AI Models extracted from research papers."
        )

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
            #
            # Uploads often run from a saved extraction JSON in a separate
            # invocation, so the validated row boundaries have to be read back
            # out of that file. Without this, an upload would silently fall back
            # to the heuristic splitter and undo semantic validation.
            rows_by_model = extraction_data.get("validated_rows")
            if rows_by_model:
                logger.info(
                    "Using semantically validated rows for %d model(s)", len(rows_by_model)
                )
            mapped_data = self.mapper.map_extraction_result(
                extraction_result, rows_by_model=rows_by_model
            )
            if not mapped_data or not mapped_data.get("contributions"):
                logger.error("Mapping failed - no valid contributions to upload")
                return None

            # 4. On live, reuse an existing paper and append this extraction to
            #    it as additional contribution tabs. On sandbox/incubating we
            #    intentionally skip the search and always create a fresh test
            #    paper (its title carries a unique [TEST-...] suffix, so a search
            #    by the real title wouldn't match one anyway).
            paper_id = None
            contribution_ids = []      # everything on the paper afterwards
            new_contribution_ids = []  # only what this run added
            is_live = getattr(self.client, "host", "sandbox") == "production"

            if is_live:
                existing_papers = self.client.search_papers(paper_title) or []
                for paper in existing_papers:
                    if paper.get("title", "").strip().lower() == paper_title.strip().lower():
                        paper_id = paper.get("id")
                        logger.info(f"Found existing paper in ORKG: {paper_id}")

                        # Keep the existing contributions and add this
                        # extraction alongside them, as further contribution
                        # tabs — the same shape a paper gets when it introduces
                        # several models. A re-extraction is new information
                        # about the paper, not a duplicate to be suppressed, so
                        # a matching label no longer causes a skip.
                        paper_data = self.client.get_paper(paper_id)
                        existing_contribs = (
                            paper_data.get("contributions", []) if paper_data else []
                        )
                        contribution_ids = [
                            c.get("id")
                            for c in existing_contribs
                            if isinstance(c, dict) and c.get("id")
                        ]
                        logger.info(
                            "Paper %s already has %d contribution(s); adding %d more",
                            paper_id,
                            len(contribution_ids),
                            len(mapped_data["contributions"]),
                        )

                        for contrib_data in mapped_data["contributions"]:
                            label = contrib_data.get("label", "").strip()
                            logger.info(
                                "Adding contribution '%s' to existing paper %s", label, paper_id
                            )
                            new_cid = self.client.add_contribution_to_paper(paper_id, contrib_data)
                            if new_cid:
                                contribution_ids.append(new_cid)
                                new_contribution_ids.append(new_cid)
                            else:
                                logger.error(
                                    "Failed to add contribution '%s' to paper %s",
                                    label,
                                    paper_id,
                                )
                        break

            # Create a new paper when none matched (always, on sandbox)
            if not paper_id:
                # 5. Create Paper with all Contributions (Step A)
                # Live: real title. Sandbox/incubating: unique [TEST-...] suffix so
                # repeated test runs don't collide on duplicate-title rejection.
                if is_live:
                    paper_title_to_use = paper_title
                    logger.info("Creating new paper in ORKG (live)...")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    paper_title_to_use = f"{paper_title} [TEST-{timestamp}]"
                    logger.info("Creating new paper in ORKG (sandbox test mode, unique title)...")
                logger.info(f"Using title: {paper_title_to_use}")
                result = self.client.create_paper_with_contributions(
                    title=paper_title_to_use,
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
                # A new paper carries only what we just uploaded.
                new_contribution_ids = list(contribution_ids)

            # 6. Link to Comparison Table (Step B) — gated by config, because
            # only live comparisons are updatable and the paper upload above is
            # complete and useful on its own.
            comparison_updated = None
            if not self.update_comparison:
                logger.info(
                    "Skipping comparison %s (orkg.update_comparison is false)",
                    self.comparison_id,
                )
            else:
                logger.info(
                    "Adding %d contribution(s) to comparison %s",
                    len(contribution_ids),
                    self.comparison_id,
                )
                comparison_updated = self.client.update_comparison_sources(
                    self.comparison_id, contribution_ids
                )
                if not comparison_updated:
                    # The paper and its contributions are already in ORKG, so
                    # this is a partial success, not a failed upload.
                    logger.error(
                        "Paper %s uploaded, but comparison %s was NOT updated",
                        paper_id,
                        self.comparison_id,
                    )

            logger.info(f"Successfully processed paper: {paper_id}")
            return {
                "paper_id": paper_id,
                # Everything on the paper now, versus only what this run added.
                # The ledger records the latter: those are the tabs to open.
                "contribution_ids": contribution_ids,
                "new_contribution_ids": new_contribution_ids,
                "comparison_updated": comparison_updated,
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return None
