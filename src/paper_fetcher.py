"""
ArXiv Paper Fetcher

This module handles fetching research papers from ArXiv and downloading PDFs.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime as _datetime

import arxiv
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PaperFetcher:
    """Fetches papers from ArXiv and downloads PDFs."""

    def __init__(self, download_dir: str = "data/papers"):
        """
        Initialize paper fetcher.

        Args:
            download_dir: Directory to save downloaded PDFs
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized PaperFetcher with download_dir: {download_dir}")

    def fetch_paper_metadata(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch paper metadata from ArXiv.

        Args:
            arxiv_id: ArXiv paper ID (e.g., "2307.09288")

        Returns:
            Dictionary with paper metadata, or None if error
        """
        try:
            logger.info(f"Fetching metadata for ArXiv ID: {arxiv_id}")

            # Clean arxiv_id (remove version if present)
            clean_id = arxiv_id.split("v")[0]

            # Search for paper
            search = arxiv.Search(id_list=[clean_id])
            paper = next(search.results())

            metadata = {
                "arxiv_id": clean_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.isoformat() if paper.published else None,
                "updated": paper.updated.isoformat() if paper.updated else None,
                "doi": paper.doi,
                "primary_category": paper.primary_category,
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id,
            }

            logger.info(f"Successfully fetched metadata: {metadata['title']}")
            return metadata

        except StopIteration:
            logger.error(f"Paper not found: {arxiv_id}")
            return None
        except Exception as e:
            logger.error(f"Error fetching metadata for {arxiv_id}: {e}")
            return None

    def download_pdf(
        self, arxiv_id: str, filename: Optional[str] = None, force: bool = False
    ) -> Optional[Path]:
        """
        Download PDF from ArXiv.

        Args:
            arxiv_id: ArXiv paper ID
            filename: Custom filename (optional)
            force: Force re-download if file exists

        Returns:
            Path to downloaded PDF, or None if error
        """
        try:
            # Clean arxiv_id
            clean_id = arxiv_id.split("v")[0]

            # Determine filename
            if filename is None:
                filename = f"{clean_id.replace('/', '_')}.pdf"

            filepath = self.download_dir / filename

            # Check if file already exists
            if filepath.exists() and not force:
                logger.info(f"PDF already exists: {filepath}")
                return filepath

            logger.info(f"Downloading PDF for {clean_id}")

            # Get paper metadata for PDF URL
            metadata = self.fetch_paper_metadata(clean_id)
            if not metadata:
                return None

            pdf_url = metadata["pdf_url"]

            # Download PDF with progress bar
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(filepath, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Successfully downloaded PDF: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error downloading PDF for {arxiv_id}: {e}")
            return None

    def download_pdf_from_url(
        self, url: str, filename: Optional[str] = None, force: bool = False
    ) -> Optional[Path]:
        """
        Download PDF from a URL (e.g. conference, institutional repo).

        Args:
            url: PDF URL (must start with http:// or https://)
            filename: Custom filename (optional); default derived from URL or "paper_YYYYMMDD.pdf"
            force: Force re-download if file exists

        Returns:
            Path to downloaded PDF, or None if error
        """
        try:
            url = url.strip()
            if not url.startswith(("http://", "https://")):
                logger.error(f"Invalid PDF URL: {url[:80]}...")
                return None

            if filename is None:
                last = url.rstrip("/").split("/")[-1]
                if last.lower().endswith(".pdf"):
                    filename = re.sub(r"[^\w\-_.]", "_", last)
                else:
                    filename = f"paper_{_datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            filepath = self.download_dir / filename

            if filepath.exists() and not force:
                logger.info(f"PDF already exists: {filepath}")
                return filepath

            logger.info(f"Downloading PDF from URL: {url[:80]}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            with open(filepath, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Successfully downloaded PDF: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error downloading PDF from URL: {e}")
            return None

    def fetch_paper(self, arxiv_id: str, download_pdf: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch paper metadata and optionally download PDF.

        Args:
            arxiv_id: ArXiv paper ID
            download_pdf: Whether to download PDF

        Returns:
            Dictionary with metadata and PDF path, or None if error
        """
        # Fetch metadata
        metadata = self.fetch_paper_metadata(arxiv_id)
        if not metadata:
            return None

        # Download PDF if requested
        if download_pdf:
            pdf_path = self.download_pdf(arxiv_id)
            metadata["pdf_path"] = str(pdf_path) if pdf_path else None
        else:
            metadata["pdf_path"] = None

        return metadata

    def search_papers(
        self, query: str, max_results: int = 10, categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on ArXiv.

        Args:
            query: Search query
            max_results: Maximum number of results
            categories: Filter by categories (e.g., ["cs.CL", "cs.AI"])

        Returns:
            List of paper metadata dictionaries
        """
        try:
            logger.info(f"Searching ArXiv: {query}")

            # Build search query
            search_query = query
            if categories:
                cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
                search_query = f"{query} AND ({cat_query})"

            # Search ArXiv
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            papers = []
            for paper in search.results():
                metadata = {
                    "arxiv_id": paper.get_short_id(),
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat() if paper.published else None,
                    "primary_category": paper.primary_category,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                }
                papers.append(metadata)

            logger.info(f"Found {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []

    def fetch_multiple_papers(
        self, arxiv_ids: List[str], download_pdfs: bool = True
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetch multiple papers.

        Args:
            arxiv_ids: List of ArXiv IDs
            download_pdfs: Whether to download PDFs

        Returns:
            Dictionary mapping arxiv_id to metadata
        """
        results = {}

        logger.info(f"Fetching {len(arxiv_ids)} papers")

        for arxiv_id in tqdm(arxiv_ids, desc="Fetching papers"):
            results[arxiv_id] = self.fetch_paper(arxiv_id, download_pdf=download_pdfs)

        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Successfully fetched {successful}/{len(arxiv_ids)} papers")

        return results

    def get_pdf_path(self, arxiv_id: str) -> Optional[Path]:
        """
        Get path to downloaded PDF.

        Args:
            arxiv_id: ArXiv paper ID

        Returns:
            Path to PDF if exists, None otherwise
        """
        clean_id = arxiv_id.split("v")[0]
        filename = f"{clean_id.replace('/', '_')}.pdf"
        filepath = self.download_dir / filename

        if filepath.exists():
            return filepath
        return None

    def list_downloaded_papers(self) -> List[str]:
        """
        List all downloaded papers.

        Returns:
            List of ArXiv IDs
        """
        pdfs = list(self.download_dir.glob("*.pdf"))
        arxiv_ids = [pdf.stem.replace("_", "/") for pdf in pdfs]
        return arxiv_ids
