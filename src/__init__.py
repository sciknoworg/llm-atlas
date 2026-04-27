"""
Bachelor Thesis: Automated Knowledge Extraction for LLM Catalog

This package provides tools for automatically extracting information from
LLM research papers and populating the ORKG comparison database.
"""

__version__ = "0.1.0"
__author__ = "Bachelor Thesis Project"

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

_LAZY_IMPORTS = {
    "ORKGClient": ("src.orkg_client", "ORKGClient"),
    "PaperFetcher": ("src.paper_fetcher", "PaperFetcher"),
    "PDFParser": ("src.pdf_parser", "PDFParser"),
    "LLMExtractor": ("src.llm_extractor", "LLMExtractor"),
    "TemplateMapper": ("src.template_mapper", "TemplateMapper"),
    "ORKGPaperManager": ("src.orkg_manager", "ORKGPaperManager"),
    "ComparisonUpdater": ("src.comparison_updater", "ComparisonUpdater"),
    "ExtractionPipeline": ("src.pipeline", "ExtractionPipeline"),
}


def __getattr__(name):
    """Lazily expose package classes without importing CLI modules eagerly."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'src' has no attribute {name!r}")

    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = __import__(module_name, fromlist=[attribute_name])
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
