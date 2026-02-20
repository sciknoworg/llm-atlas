"""
Extraction Pipeline

Main orchestrator for the LLM extraction and ORKG update pipeline.
"""

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from src.extraction_normalizer import normalize_extraction
from src.llm_extractor import LLMExtractor, LLMProperties, MultiModelResponse
from src.model_contribution_selector import select_primary_model_contributions
from src.model_variant_merger import merge_model_variants
from src.orkg_client import ORKGClient
from src.orkg_manager import ORKGPaperManager
from src.paper_fetcher import PaperFetcher
from src.pdf_parser import PDFParser
from src.template_mapper import TemplateMapper

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("data/logs/pipeline.log")],
)

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Main pipeline for extracting LLM information and updating ORKG."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize extraction pipeline.

        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing ExtractionPipeline")

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self._initialize_components()

        logger.info("ExtractionPipeline initialized successfully")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return {
                "orkg": {"host": "sandbox", "template_id": "R609825", "comparison_id": "R1364660"},
                "kisski": {
                    "model": "openai-gpt-oss-120b",
                    "temperature": 0.0,
                    "max_tokens": 4000,
                    "base_url": "https://chat-ai.academiccloud.de/v1",
                    "rate_limit_delay": 2.0,
                },
                "arxiv": {"max_results": 10, "download_dir": "data/papers"},
                "extraction": {"max_chunk_size": 8000, "multi_model_extraction": True},
            }

    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Get API keys from environment
        kisski_api_key = os.getenv("KISSKI_API_KEY")
        kisski_base_url = os.getenv("KISSKI_BASE_URL", self.config["kisski"].get("base_url"))
        orkg_email = os.getenv("ORKG_EMAIL")
        orkg_password = os.getenv("ORKG_PASSWORD")

        if not kisski_api_key:
            logger.warning("KISSKI_API_KEY not found in environment")

        # Store ORKG credentials for lazy initialization (only connect when needed)
        self._orkg_email = orkg_email
        self._orkg_password = orkg_password
        self._orkg_client = None  # Will be initialized lazily
        self._orkg_manager = None  # Will be initialized lazily

        # Initialize paper fetcher
        self.paper_fetcher = PaperFetcher(
            download_dir=self.config["arxiv"].get("download_dir", "data/papers")
        )

        # Initialize PDF parser
        self.pdf_parser = PDFParser(method="pdfplumber")

        # Initialize LLM extractor (KISSKI API)
        if kisski_api_key:
            self.llm_extractor = LLMExtractor(
                api_key=kisski_api_key,
                base_url=kisski_base_url,
                model=self.config["kisski"]["model"],
                temperature=self.config["kisski"]["temperature"],
                max_tokens=self.config["kisski"]["max_tokens"],
                timeout=self.config["kisski"].get("timeout", 60),
                rate_limit_delay=self.config["kisski"].get("rate_limit_delay", 2.0),
            )
            logger.info(
                f"Initialized KISSKI API extractor (model: {self.config['kisski']['model']})"
            )
        else:
            self.llm_extractor = None
            logger.warning("LLM extractor not initialized (missing KISSKI_API_KEY)")

        # Initialize template mapper
        self.template_mapper = TemplateMapper(template_id=self.config["orkg"]["template_id"])

    def _get_orkg_client(self):
        """Lazily initialize ORKG client (only when needed)."""
        if self._orkg_client is None:
            logger.info("Initializing ORKG client (lazy initialization)...")
            self._orkg_client = ORKGClient(
                host=self.config["orkg"]["host"],
                email=self._orkg_email,
                password=self._orkg_password,
                timeout=self.config["orkg"].get("timeout", 30),
            )
        return self._orkg_client

    def _get_orkg_manager(self):
        """Lazily initialize ORKG paper manager (only when needed)."""
        if self._orkg_manager is None:
            logger.info("Initializing ORKG paper manager (lazy initialization)...")
            self._orkg_manager = ORKGPaperManager(
                orkg_client=self._get_orkg_client(), template_mapper=self.template_mapper
            )
        return self._orkg_manager

    @property
    def orkg_client(self):
        """Property accessor for ORKG client (lazy initialization)."""
        return self._get_orkg_client()

    @property
    def orkg_manager(self):
        """Property accessor for ORKG paper manager (lazy initialization)."""
        return self._get_orkg_manager()

    def process_paper(
        self, arxiv_id: str, save_intermediate: bool = True, update_orkg: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single paper through the complete pipeline.

        Args:
            arxiv_id: ArXiv paper ID
            save_intermediate: Whether to save intermediate results
            update_orkg: Whether to update ORKG comparison

        Returns:
            Processing results
        """
        logger.info("=" * 80)
        logger.info(f"Processing paper: {arxiv_id}")
        logger.info("=" * 80)

        result = {
            "arxiv_id": arxiv_id,
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "steps": {},
        }

        try:
            # Step 1: Fetch paper from ArXiv
            logger.info("Step 1: Fetching paper from ArXiv")
            paper_metadata = self.paper_fetcher.fetch_paper(arxiv_id, download_pdf=True)

            if not paper_metadata:
                result["status"] = "failed"
                result["error"] = "Failed to fetch paper"
                return result

            result["steps"]["fetch"] = "success"
            result["paper_metadata"] = paper_metadata

            # Step 2: Parse PDF
            logger.info("Step 2: Parsing PDF")
            pdf_path = Path(paper_metadata["pdf_path"])
            parsed_data = self.pdf_parser.parse(pdf_path)

            if not parsed_data:
                result["status"] = "failed"
                result["error"] = "Failed to parse PDF"
                return result

            result["steps"]["parse"] = "success"
            result["text_length"] = parsed_data["text_length"]
            result["word_count"] = parsed_data["word_count"]

            # Step 3: Extract information with LLM
            logger.info("Step 3: Extracting information with LLM")

            if not self.llm_extractor:
                result["status"] = "failed"
                result["error"] = "LLM extractor not available"
                return result

            # Chunk text if needed
            text = parsed_data["cleaned_text"]
            max_chunk_size = self.config["extraction"]["max_chunk_size"]

            if len(text) > max_chunk_size:
                chunks = self.pdf_parser.chunk_text(text, max_chunk_size)
                extraction_result = self.llm_extractor.extract_from_chunks(chunks, paper_metadata)
            else:
                extraction_result = self.llm_extractor.extract(text, paper_metadata)

            if not extraction_result or not extraction_result.models:
                result["status"] = "failed"
                result["error"] = "Failed to extract LLM information"
                return result

            result["steps"]["extract"] = "success"
            result["models_extracted"] = len(extraction_result.models)
            result["extraction_data"] = [model.model_dump() for model in extraction_result.models]
            self._inject_date_created_from_metadata(result["extraction_data"], paper_metadata)
            result["extraction_data"] = normalize_extraction(result["extraction_data"])

            # Step 3.25: Keep contribution-level models, drop auxiliary artifacts
            logger.info("Step 3.25: Selecting primary model contributions")
            result["extraction_data"] = select_primary_model_contributions(
                result["extraction_data"],
                paper_metadata
            )
            result["models_after_selection"] = len(result["extraction_data"])
            logger.info(
                "Models after selection: %s -> %s",
                result["models_extracted"],
                result["models_after_selection"],
            )

            # Step 3.5: Merge size variants (align with gold-standard structure)
            logger.info("Step 3.5: Merging size variants (gold-standard alignment)")
            result["extraction_data"] = merge_model_variants(
                result["extraction_data"],
                paper_metadata
            )
            result["models_after_merge"] = len(result["extraction_data"])
            logger.info(
                f"Models after merge: {result['models_extracted']} -> "
                f"{result['models_after_merge']}"
            )

            # Step 4: Map to ORKG template (use merged models for downstream alignment)
            logger.info("Step 4: Mapping to ORKG template")
            merged_response = MultiModelResponse(
                models=[LLMProperties(**model) for model in result["extraction_data"]],
                paper_describes_multiple_models=(len(result["extraction_data"]) > 1),
            )
            mapped_result = self.template_mapper.map_extraction_result(merged_response)

            result["steps"]["map"] = "success"
            result["contributions"] = mapped_result["contributions"]

            # Save intermediate results if requested
            if save_intermediate:
                saved_path = self._save_intermediate_results(arxiv_id, result)
                result["saved_path"] = str(saved_path) if saved_path else None

            # Step 5: Upload to ORKG (uses model family grouping)
            if update_orkg:
                logger.info("Step 5: Uploading to ORKG (grouping by model family)")

                # Prepare extraction data for ORKGPaperManager
                extraction_data = {
                    "raw_extraction": result["extraction_data"],
                    "paper_title": paper_metadata.get("title"),
                    "arxiv_id": arxiv_id,
                    "paper_url": paper_metadata.get("pdf_url"),
                }

                # Prepare paper metadata for ORKG
                orkg_metadata = {
                    "title": paper_metadata.get("title"),
                    "authors": paper_metadata.get("authors", []),
                    "year": self._extract_year(paper_metadata.get("published")),
                    "month": self._extract_month(paper_metadata.get("published")),
                    "url": paper_metadata.get("pdf_url"),
                    "doi": paper_metadata.get("doi"),
                }

                update_result = self.orkg_manager.process_and_upload(extraction_data, orkg_metadata)

                if update_result:
                    result["steps"]["update_orkg"] = "success"
                    result["orkg_results"] = update_result
                else:
                    result["steps"]["update_orkg"] = "failed"
                    result["orkg_results"] = {"status": "failed", "error": "Upload failed"}
            else:
                logger.info("Step 5: Skipping ORKG update (update_orkg=False)")
                result["steps"]["update_orkg"] = "skipped"

            result["status"] = "completed"
            logger.info(f"Successfully processed paper: {arxiv_id}")

        except Exception as e:
            logger.error(f"Error processing paper {arxiv_id}: {e}", exc_info=True)
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    def _slugify_title(self, title: str, max_len: int = 80) -> str:
        """Create a filesystem-safe slug from paper title."""
        if not title or not str(title).strip():
            return "untitled"
        s = re.sub(r"[^\w\s-]", "", str(title).strip().lower())
        s = re.sub(r"[-\s]+", "_", s).strip("_")
        return s[:max_len] if len(s) > max_len else s or "untitled"

    def process_paper_from_pdf_url(
        self,
        pdf_url: str,
        paper_title: str,
        save_intermediate: bool = True,
        update_orkg: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a paper from a PDF URL (no ArXiv). Download → parse → extract → save.
        Use the same paper_title as in the gold standard for evaluation matching.

        Args:
            pdf_url: URL of the PDF (e.g. OpenAI CDN, conference page).
            paper_title: Paper title (must match gold standard if evaluating).
            save_intermediate: Save extraction JSON to data/extracted/.
            update_orkg: Whether to upload to ORKG.

        Returns:
            Result dict with extraction_data, paper_metadata, and saved_path if saved.
        """
        logger.info("=" * 80)
        logger.info(f"Processing paper from PDF URL: {paper_title}")
        logger.info("=" * 80)

        result = {
            "source": "pdf_url",
            "pdf_url": pdf_url,
            "paper_title": paper_title,
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "steps": {},
        }

        try:
            # Step 1: Download PDF
            logger.info("Step 1: Downloading PDF from URL")
            pdf_path = self.paper_fetcher.download_pdf_from_url(pdf_url)
            if not pdf_path:
                result["status"] = "failed"
                result["error"] = "Failed to download PDF from URL"
                return result

            result["steps"]["download"] = "success"
            result["pdf_path"] = str(pdf_path)

            paper_metadata = {
                "title": paper_title,
                "authors": [],
                "published": None,
                "pdf_url": pdf_url,
                "doi": None,
            }

            # Step 2: Parse PDF
            logger.info("Step 2: Parsing PDF")
            parsed_data = self.pdf_parser.parse(Path(pdf_path))
            if not parsed_data:
                result["status"] = "failed"
                result["error"] = "Failed to parse PDF"
                return result

            result["steps"]["parse"] = "success"
            result["text_length"] = parsed_data["text_length"]
            result["word_count"] = parsed_data["word_count"]

            # Step 3: Extract with LLM
            logger.info("Step 3: Extracting information with LLM")
            if not self.llm_extractor:
                result["status"] = "failed"
                result["error"] = "LLM extractor not available"
                return result

            text = parsed_data["cleaned_text"]
            max_chunk_size = self.config["extraction"]["max_chunk_size"]
            if len(text) > max_chunk_size:
                chunks = self.pdf_parser.chunk_text(text, max_chunk_size)
                extraction_result = self.llm_extractor.extract_from_chunks(chunks, paper_metadata)
            else:
                extraction_result = self.llm_extractor.extract(text, paper_metadata)

            if not extraction_result or not extraction_result.models:
                result["status"] = "failed"
                result["error"] = "Failed to extract LLM information"
                return result

            result["steps"]["extract"] = "success"
            result["models_extracted"] = len(extraction_result.models)
            result["extraction_data"] = [m.model_dump() for m in extraction_result.models]
            self._inject_date_created_from_metadata(result["extraction_data"], paper_metadata)
            result["extraction_data"] = normalize_extraction(result["extraction_data"])
            result["paper_metadata"] = paper_metadata

            # Step 3.25: Keep contribution-level models, drop auxiliary artifacts
            logger.info("Step 3.25: Selecting primary model contributions")
            result["extraction_data"] = select_primary_model_contributions(
                result["extraction_data"],
                paper_metadata
            )
            result["models_after_selection"] = len(result["extraction_data"])
            logger.info(
                "Models after selection: %s -> %s",
                result["models_extracted"],
                result["models_after_selection"],
            )

            # Step 3.5: Merge size variants (align with gold-standard structure)
            logger.info("Step 3.5: Merging size variants (gold-standard alignment)")
            result["extraction_data"] = merge_model_variants(
                result["extraction_data"],
                paper_metadata
            )
            result["models_after_merge"] = len(result["extraction_data"])
            logger.info(
                f"Models after merge: {result['models_extracted']} -> "
                f"{result['models_after_merge']}"
            )

            # Step 4: Map to ORKG template (use merged models for downstream alignment)
            logger.info("Step 4: Mapping to ORKG template")
            merged_response = MultiModelResponse(
                models=[LLMProperties(**model) for model in result["extraction_data"]],
                paper_describes_multiple_models=(len(result["extraction_data"]) > 1),
            )
            mapped_result = self.template_mapper.map_extraction_result(merged_response)
            result["steps"]["map"] = "success"
            result["contributions"] = mapped_result["contributions"]

            if save_intermediate:
                slug = self._slugify_title(paper_title)
                saved_path = self._save_intermediate_results_from_url(slug, result)
                result["saved_path"] = str(saved_path) if saved_path else None

            if update_orkg:
                logger.info("Step 5: Uploading to ORKG")
                extraction_data = {
                    "raw_extraction": result["extraction_data"],
                    "paper_title": paper_title,
                    "arxiv_id": None,
                    "paper_url": pdf_url,
                }
                orkg_metadata = {
                    "title": paper_title,
                    "authors": [],
                    "year": self._extract_year(paper_metadata.get("published")),
                    "month": self._extract_month(paper_metadata.get("published")),
                    "url": pdf_url,
                    "doi": None,
                }
                update_result = self.orkg_manager.process_and_upload(extraction_data, orkg_metadata)
                if update_result:
                    result["steps"]["update_orkg"] = "success"
                    result["orkg_results"] = update_result
                else:
                    result["steps"]["update_orkg"] = "failed"
                    result["orkg_results"] = {"status": "failed", "error": "Upload failed"}
            else:
                result["steps"]["update_orkg"] = "skipped"

            result["status"] = "completed"
            logger.info(f"Successfully processed paper from URL: {paper_title}")

        except Exception as e:
            logger.error(f"Error processing paper from URL: {e}", exc_info=True)
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    def _save_intermediate_results_from_url(
        self, slug: str, result: Dict[str, Any]
    ) -> Optional[Path]:
        """Save extraction result for a PDF-URL paper (slug-based filename)."""
        output_dir = Path("data/extracted")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved intermediate results to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
            return None

    def process_multiple_papers(
        self, arxiv_ids: List[str], save_intermediate: bool = True, update_orkg: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple papers.

        Args:
            arxiv_ids: List of ArXiv IDs
            save_intermediate: Whether to save intermediate results
            update_orkg: Whether to update ORKG

        Returns:
            Summary of results
        """
        logger.info(f"Processing {len(arxiv_ids)} papers")

        results = {"total": len(arxiv_ids), "completed": 0, "failed": 0, "papers": {}}

        for arxiv_id in arxiv_ids:
            paper_result = self.process_paper(
                arxiv_id, save_intermediate=save_intermediate, update_orkg=update_orkg
            )

            results["papers"][arxiv_id] = paper_result

            if paper_result["status"] == "completed":
                results["completed"] += 1
            else:
                results["failed"] += 1

        logger.info(
            f"Batch processing complete: "
            f"{results['completed']} succeeded, {results['failed']} failed"
        )

        return results

    def search_and_process(
        self, query: str, max_results: int = 10, update_orkg: bool = True
    ) -> Dict[str, Any]:
        """
        Search ArXiv and process papers.

        Args:
            query: Search query
            max_results: Maximum papers to process
            update_orkg: Whether to update ORKG

        Returns:
            Processing results
        """
        logger.info(f"Searching ArXiv: {query}")

        # Search for papers
        papers = self.paper_fetcher.search_papers(
            query, max_results=max_results, categories=self.config["arxiv"].get("categories")
        )

        if not papers:
            logger.warning("No papers found")
            return {"status": "no_papers_found"}

        # Extract ArXiv IDs
        arxiv_ids = [paper["arxiv_id"] for paper in papers]

        # Process papers
        return self.process_multiple_papers(arxiv_ids, update_orkg=update_orkg)

    def _save_intermediate_results(self, arxiv_id: str, result: Dict[str, Any]):
        """Save intermediate results to JSON file. Returns path if saved, None otherwise."""
        output_dir = Path("data/extracted")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{arxiv_id.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved intermediate results to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
            return None

    def _inject_date_created_from_metadata(
        self, extraction_data: List[Dict[str, Any]], paper_metadata: Optional[Dict[str, Any]]
    ) -> None:
        """
        Set date_created for all models from paper published date when available.
        Mutates extraction_data in place. Format: YYYY-MM (e.g. 2022-05).
        """
        if not paper_metadata:
            return
        published = paper_metadata.get("published")
        if not published:
            return
        year = self._extract_year(published)
        if year is None:
            return
        month = self._extract_month(published)
        date_created = f"{year}-{month:02d}" if month else f"{year}-01"
        injected = 0
        for model in extraction_data:
            current = model.get("date_created")
            if current in (None, "", "null", "None"):
                model["date_created"] = date_created
                injected += 1
        if injected:
            logger.info(
                "Set missing date_created from paper metadata for %s model(s): %s",
                injected, date_created,
            )

    def _extract_year(self, date_string: Optional[str]) -> Optional[int]:
        """Extract year from ISO date string."""
        if not date_string:
            return None
        try:
            return int(date_string[:4])
        except (ValueError, IndexError):
            return None

    def _extract_month(self, date_string: Optional[str]) -> Optional[int]:
        """Extract month from ISO date string."""
        if not date_string:
            return None
        try:
            month = int(date_string[5:7])
            return month if 1 <= month <= 12 else None
        except (ValueError, IndexError):
            return None

    def test_connection(self) -> Dict[str, bool]:
        """
        Test connections to all services.

        Returns:
            Dictionary with connection test results
        """
        logger.info("Testing connections")

        results = {"orkg": self.orkg_client.ping(), "llm_extractor": self.llm_extractor is not None}

        logger.info(f"Connection test results: {results}")
        return results

    def get_status(self) -> Dict[str, Any]:
        """
        Get pipeline status.

        Returns:
            Status information
        """
        return {
            "orkg_host": self.config["orkg"]["host"],
            "template_id": self.config["orkg"]["template_id"],
            "comparison_id": self.config["orkg"]["comparison_id"],
            "llm_model": self.config["kisski"]["model"],
            "connections": self.test_connection(),
        }


def _run_evaluation(prediction_path: str, gold_path: str) -> bool:
    """
    Run strict evaluation script after successful extraction.
    Returns True if evaluation ran successfully (exit code 0).
    """
    project_root = Path(__file__).resolve().parent.parent
    script = project_root / "scripts" / "evaluation" / "evaluate_extraction_strict.py"
    if not script.exists():
        logger.warning("Evaluation script not found: %s", script)
        return False
    gold = Path(gold_path)
    if not gold.is_absolute():
        gold = project_root / gold_path
    if not gold.exists():
        logger.warning("Gold standard not found: %s", gold)
        return False
    cmd = [sys.executable, str(script), "--prediction", prediction_path, "--gold", str(gold)]
    print("\n" + "=" * 80)
    print("EVALUATION (strict)")
    print("=" * 80)
    r = subprocess.run(cmd)
    return r.returncode == 0


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Extraction Pipeline")
    parser.add_argument("--arxiv-id", help="ArXiv ID to process")
    parser.add_argument(
        "--pdf-url", help="PDF URL (for papers not on ArXiv). Use with --paper-title."
    )
    parser.add_argument(
        "--paper-title",
        help="Paper title (required with --pdf-url; use gold-standard title for evaluation)",
    )
    parser.add_argument("--json-file", help="Upload existing extraction JSON file to ORKG")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--max-results", type=int, default=10, help="Max search results")
    parser.add_argument("--no-update", action="store_true", help="Don't update ORKG")
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation after extraction (default: run strict evaluation when successful)",
    )
    parser.add_argument(
        "--gold",
        default="data/gold_standard/R1364660.json",
        help="Gold-standard JSON for evaluation (default: R1364660.json)",
    )
    parser.add_argument("--test", action="store_true", help="Test connections")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ExtractionPipeline()

    if args.test:
        # Test connections
        results = pipeline.test_connection()
        print("\nConnection Test Results:")
        for service, status in results.items():
            print(f"  {service}: {'✓' if status else '✗'}")

    elif args.status:
        # Show status
        status = pipeline.get_status()
        print("\nPipeline Status:")
        print(json.dumps(status, indent=2))

    elif args.json_file:
        # Upload existing extraction JSON to ORKG
        json_path = Path(args.json_file)
        if not json_path.exists():
            print(f"✗ ERROR: File not found: {args.json_file}")
            sys.exit(1)

        print("=" * 80)
        print("Uploading Extraction JSON to ORKG")
        print("=" * 80)
        print(f"JSON file: {args.json_file}")

        # Load JSON file
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                extraction_data = json.load(f)
        except Exception as e:
            print(f"✗ ERROR: Failed to load JSON file: {e}")
            sys.exit(1)

        # Use ORKGPaperManager (same as Grete pipeline)
        models_list = extraction_data.get(
            "extraction_data", extraction_data.get("raw_extraction", [])
        )

        print(f"\n{'=' * 80}")
        print("EXTRACTION DATA FROM JSON FILE")
        print("=" * 80)
        print(f"Research Paper: {extraction_data.get('paper_title', 'Unknown')}")
        print(f"ArXiv ID: {extraction_data.get('arxiv_id', 'N/A')}")
        print(f"Models extracted: {len(models_list)}")

        # Display key fields for each model version
        if models_list:
            print(f"\n{'=' * 80}")
            print("EXTRACTED MODEL VERSIONS")
            print("=" * 80)

            for i, model in enumerate(models_list, 1):
                print(f"\n--- Model Version {i} ---")
                print(f"  Paper Title:        {model.get('paper_title', 'N/A')}")
                print(f"  Model Name:         {model.get('model_name', 'N/A')}")
                print(f"  Model Family:       {model.get('model_family', 'N/A')}")
                print(f"  Date Created:       {model.get('date_created', 'N/A')}")
                print(f"  Organization:       {model.get('organization', 'N/A')}")
                print(f"  Innovation:         {model.get('innovation', 'N/A')}")
                print(f"  Pretraining Arch:   {model.get('pretraining_architecture', 'N/A')}")
                print(f"  Pretraining Task:   {model.get('pretraining_task', 'N/A')}")
                print(f"  Fine-tuning Task:   {model.get('finetuning_task', 'N/A')}")
                print(f"  Optimizer:          {model.get('optimizer', 'N/A')}")
                print(f"  Parameters:         {model.get('parameters', 'N/A')}")
                print(f"  Parameters (M):     {model.get('parameters_millions', 'N/A')}")
                print(f"  Hardware Used:      {model.get('hardware_used', 'N/A')}")
                print(f"  Extension:          {model.get('extension', 'N/A')}")
                print(f"  Blog Post:          {model.get('blog_post', 'N/A')}")
                print(f"  License:            {model.get('license', 'N/A')}")
                print(f"  Research Problem:   {model.get('research_problem', 'N/A')}")
                print(f"  Pretraining Corpus: {model.get('pretraining_corpus', 'N/A')}")
                print(f"  Application:        {model.get('application', 'N/A')}")

            print(f"\n{'=' * 80}")

        # Prepare paper metadata from extraction data or ArXiv
        paper_metadata = extraction_data.get("paper_metadata", {})

        # If no metadata in JSON, try to fetch from ArXiv
        arxiv_id = extraction_data.get("arxiv_id")
        if not paper_metadata and arxiv_id:
            try:
                paper_meta = pipeline.paper_fetcher.fetch_paper(arxiv_id, download_pdf=False)
                if paper_meta:
                    paper_metadata = {
                        "title": paper_meta.get("title"),
                        "authors": paper_meta.get("authors", []),
                        "year": pipeline._extract_year(paper_meta.get("published")),
                        "month": pipeline._extract_month(paper_meta.get("published")),
                        "url": paper_meta.get("pdf_url"),
                        "doi": paper_meta.get("doi"),
                    }
            except Exception as e:
                logger.debug(f"Could not fetch ArXiv metadata: {e}")

        if not paper_metadata:
            paper_metadata = {
                "title": extraction_data.get("paper_title", "Unknown Paper"),
                "authors": [],
                "year": 2024,
                "url": extraction_data.get("paper_url"),
            }

        print("\nUploading to ORKG (using model family grouping)...")

        # Transform data to match ORKGPaperManager's expected format
        # ORKGPaperManager expects 'raw_extraction' key, not 'extraction_data'
        orkg_extraction_data = {
            "raw_extraction": models_list,  # models_list is already extracted at line 478
            "paper_title": extraction_data.get("paper_title"),
            "arxiv_id": arxiv_id,
            "paper_url": extraction_data.get("paper_url"),
        }

        # Use ORKGPaperManager to upload
        result = pipeline.orkg_manager.process_and_upload(
            orkg_extraction_data, paper_metadata  # Pass the transformed data
        )

        # Display results
        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)

        if result and result.get("paper_id"):
            paper_id = result.get("paper_id")
            contrib_ids = result.get("contribution_ids", [])

            print("✓ SUCCESS: Uploaded to ORKG")
            print(f"\nPaper ID: {paper_id}")
            print(f"Contributions: {len(contrib_ids)}")
            print("\nPaper URL:")
            print(f"https://sandbox.orkg.org/paper/{paper_id}")

            if contrib_ids:
                print("\nContribution URLs:")
                for i, cid in enumerate(contrib_ids[:5], 1):
                    print(f"  {i}. https://sandbox.orkg.org/resource/{cid}")
                if len(contrib_ids) > 5:
                    print(f"  ... and {len(contrib_ids) - 5} more")
        else:
            print("✗ FAILED: Upload to ORKG failed")
            if result:
                print(json.dumps(result, indent=2))

    elif args.pdf_url:
        if not args.paper_title:
            print("✗ ERROR: --paper-title is required when using --pdf-url")
            print("  Use the exact gold-standard paper title for evaluation matching.")
            sys.exit(1)
        result = pipeline.process_paper_from_pdf_url(
            args.pdf_url, args.paper_title, save_intermediate=True, update_orkg=not args.no_update
        )
        print("\n" + "=" * 80)
        print("EXTRACTION RESULT (PDF URL)")
        print("=" * 80)
        if result.get("status") == "completed":
            print(f"[OK] Status: {result.get('status')}")
            print(f"[OK] Models extracted: {result.get('models_extracted', 0)}")
            if result.get("models_after_selection") is not None:
                print(f"[OK] Models after selection: {result.get('models_after_selection')}")
            if result.get("models_after_merge") is not None:
                print(f"[OK] Models after merge: {result.get('models_after_merge')}")
            saved = result.get("saved_path")
            if saved:
                print(f"[OK] Saved to: {saved}")
            # Show ORKG upload results (if uploaded)
            if result.get("orkg_results"):
                orkg_result = result["orkg_results"]
                print("\nORKG Upload:")
                if orkg_result.get("paper_id"):
                    print(f"  Paper ID: {orkg_result.get('paper_id')}")
                    print(
                        f"  Paper URL: https://sandbox.orkg.org/paper/{orkg_result.get('paper_id')}"
                    )
                    print(f"  Contributions: {len(orkg_result.get('contribution_ids', []))}")
                else:
                    print("  Status: Failed")
            if saved and not args.no_evaluate:
                _run_evaluation(saved, args.gold)
            elif saved and args.no_evaluate:
                print("\nRun evaluation manually:")
                print(
                    '  python scripts/evaluation/evaluate_extraction_strict.py '
                    f'--prediction "{saved}" --gold {args.gold}'
                )
        else:
            print(f"[FAIL] Status: {result.get('status', 'unknown')}")
            if result.get("error"):
                print(f"  Error: {result['error']}")
        print("=" * 80)
        sys.exit(0 if result.get("status") == "completed" else 1)

    elif args.arxiv_id:
        # Process single paper
        result = pipeline.process_paper(args.arxiv_id, update_orkg=not args.no_update)

        print("\n" + "=" * 80)
        print("EXTRACTION RESULT")
        print("=" * 80)

        # Show extraction summary
        if result.get("status") == "completed":
            print(f"✓ Status: {result.get('status')}")
            print(f"✓ Models extracted: {result.get('models_extracted', 0)}")
            if result.get("models_after_selection") is not None:
                print(f"✓ Models after selection: {result.get('models_after_selection')}")
            if result.get("models_after_merge") is not None:
                print(f"✓ Models after merge: {result.get('models_after_merge')}")

            # Show extraction data with key fields for each model version
            if result.get("extraction_data"):
                print(f"\n{'=' * 80}")
                print("EXTRACTED MODEL VERSIONS")
                print("=" * 80)

                for i, model in enumerate(result["extraction_data"], 1):
                    print(f"\n--- Model Version {i} ---")
                    print(f"  Paper Title:        {model.get('paper_title', 'N/A')}")
                    print(f"  Model Name:         {model.get('model_name', 'N/A')}")
                    print(f"  Model Family:       {model.get('model_family', 'N/A')}")
                    print(f"  Date Created:       {model.get('date_created', 'N/A')}")
                    print(f"  Organization:       {model.get('organization', 'N/A')}")
                    print(f"  Innovation:         {model.get('innovation', 'N/A')}")
                    print(f"  Pretraining Arch:   {model.get('pretraining_architecture', 'N/A')}")
                    print(f"  Pretraining Task:   {model.get('pretraining_task', 'N/A')}")
                    print(f"  Fine-tuning Task:   {model.get('finetuning_task', 'N/A')}")
                    print(f"  Optimizer:          {model.get('optimizer', 'N/A')}")
                    print(f"  Parameters:         {model.get('parameters', 'N/A')}")
                    print(f"  Parameters (M):     {model.get('parameters_millions', 'N/A')}")
                    print(f"  Hardware Used:      {model.get('hardware_used', 'N/A')}")
                    print(f"  Extension:          {model.get('extension', 'N/A')}")
                    print(f"  Blog Post:          {model.get('blog_post', 'N/A')}")
                    print(f"  License:            {model.get('license', 'N/A')}")
                    print(f"  Research Problem:   {model.get('research_problem', 'N/A')}")
                    print(f"  Pretraining Corpus: {model.get('pretraining_corpus', 'N/A')}")
                    print(f"  Application:        {model.get('application', 'N/A')}")

                print(f"\n{'=' * 80}")
                print("Full JSON structure:")
                print(json.dumps(result["extraction_data"], indent=2, ensure_ascii=False))

            # Show ORKG upload results (if uploaded)
            if result.get("orkg_results"):
                orkg_result = result["orkg_results"]
                print("\nORKG Upload:")
                if orkg_result.get("paper_id"):
                    print(f"  Paper ID: {orkg_result.get('paper_id')}")
                    print(
                        f"  Paper URL: https://sandbox.orkg.org/paper/{orkg_result.get('paper_id')}"
                    )
                    print(f"  Contributions: {len(orkg_result.get('contribution_ids', []))}")
                else:
                    print("  Status: Failed")
        else:
            print(f"✗ Status: {result.get('status', 'unknown')}")
            if result.get("error"):
                print(f"  Error: {result.get('error')}")
            print("\nFull result:")
            print(json.dumps(result, indent=2))

    elif args.search:
        # Search and process
        results = pipeline.search_and_process(
            args.search, max_results=args.max_results, update_orkg=not args.no_update
        )
        print("\nSearch and Process Results:")
        print(json.dumps(results, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
