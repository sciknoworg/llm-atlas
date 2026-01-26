"""
Bachelor Thesis: Automated Knowledge Extraction for LLM Catalog

This package provides tools for automatically extracting information from
LLM research papers and populating the ORKG comparison database.
"""

__version__ = "0.1.0"
__author__ = "Bachelor Thesis Project"

from src.orkg_client import ORKGClient
from src.paper_fetcher import PaperFetcher
from src.pdf_parser import PDFParser
from src.llm_extractor import LLMExtractor
from src.template_mapper import TemplateMapper
from src.orkg_manager import ORKGPaperManager
from src.comparison_updater import ComparisonUpdater  # Deprecated: use ORKGPaperManager
from src.pipeline import ExtractionPipeline

__all__ = [
    "ORKGClient",
    "PaperFetcher",
    "PDFParser",
    "LLMExtractor",
    "TemplateMapper",
    "ORKGPaperManager",
    "ComparisonUpdater",  # Deprecated
    "ExtractionPipeline",
]
