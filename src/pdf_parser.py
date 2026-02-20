"""
PDF Parser

This module handles extracting and preprocessing text from PDF files.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber
import PyPDF2

logger = logging.getLogger(__name__)

# Delimiter so the LLM knows the following block is structured tables from the document
TABLES_BLOCK_START = "\n\n[TABLES FROM DOCUMENT]\n"
TABLES_BLOCK_END = "\n[/TABLES]\n"


class PDFParser:
    """Parses PDF files and extracts text. With pdfplumber, also extracts tables as markdown."""

    def __init__(self, method: str = "pdfplumber", extract_tables: bool = True):
        """
        Initialize PDF parser.

        Args:
            method: Parsing method ("pypdf2" or "pdfplumber")
            extract_tables: If True and method is pdfplumber, extract tables and append as markdown.
        """
        self.method = method
        self.extract_tables = extract_tables
        logger.info(
            "Initialized PDFParser with method=%s, extract_tables=%s",
            method,
            extract_tables,
        )

    def extract_text_pypdf2(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text using PyPDF2.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text, or None if error
        """
        try:
            text = []
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                logger.info(f"Extracting text from {num_pages} pages using PyPDF2")

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text.append(page.extract_text())

            full_text = "\n".join(text)
            logger.info(f"Extracted {len(full_text)} characters")
            return full_text

        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return None

    def extract_text_pdfplumber(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text using pdfplumber (better for complex layouts).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text, or None if error
        """
        try:
            text = []
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)

                logger.info(f"Extracting text from {num_pages} pages using pdfplumber")

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)

            full_text = "\n".join(text)
            logger.info(f"Extracted {len(full_text)} characters")
            return full_text

        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return None

    @staticmethod
    def _table_to_markdown(table: List[List[Any]]) -> str:
        """
        Convert a table (list of rows, each row list of cell values) to markdown.

        Args:
            table: List of rows; each row is a list of cell values (str or None).

        Returns:
            Markdown string with header row and separator.
        """
        if not table:
            return ""
        # Normalize: same number of columns per row, str cells, escape pipe
        rows = []
        ncols = max(len(r) for r in table) if table else 0
        for row in table:
            cells = [str(c).strip() if c is not None else "" for c in row]
            # Pad to ncols
            while len(cells) < ncols:
                cells.append("")
            # Escape pipe so it doesn't break markdown
            cells = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
            rows.append(cells)
        if ncols == 0:
            return ""
        header = "| " + " | ".join(rows[0]) + " |"
        sep = "|" + "---|" * ncols
        body = "\n".join("| " + " | ".join(cells) + " |" for cells in rows[1:])
        if body:
            return header + "\n" + sep + "\n" + body
        return header

    def extract_tables_pdfplumber(self, pdf_path: Path) -> List[List[List[Any]]]:
        """
        Extract all tables from the PDF using pdfplumber (one table = list of rows).

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of tables; each table is a list of rows (list of cell strings).
        """
        all_tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        for t in tables:
                            if t and any(any(cell for cell in row) for row in t):
                                all_tables.append(t)
            logger.info("Extracted %d tables from PDF (pdfplumber)", len(all_tables))
        except Exception as e:
            logger.warning("Table extraction failed: %s", e)
        return all_tables

    def extract_text(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text from PDF using configured method.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text, or None if error
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None

        logger.info(f"Extracting text from: {pdf_path}")

        if self.method == "pypdf2":
            text = self.extract_text_pypdf2(pdf_path)
        elif self.method == "pdfplumber":
            text = self.extract_text_pdfplumber(pdf_path)
        else:
            logger.error(f"Unknown parsing method: {self.method}")
            return None

        # Fallback to alternative method if first fails
        if text is None or len(text.strip()) < 100:
            logger.warning("Primary method failed or extracted too little text, trying fallback")
            if self.method == "pypdf2":
                text = self.extract_text_pdfplumber(pdf_path)
            else:
                text = self.extract_text_pypdf2(pdf_path)

        return text

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove page numbers (common patterns)
        text = re.sub(r"\n\d+\n", "\n", text)

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove non-ASCII characters (optional, can be disabled for multilingual papers)
        # text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract common paper sections.

        Args:
            text: Full paper text

        Returns:
            Dictionary with section names and content
        """
        sections = {}

        # Common section patterns
        section_patterns = {
            "abstract": r"(?i)abstract\s*(.*?)(?=\n\s*\n|\n\s*1\s+introduction|\n\s*introduction)",
            "introduction": r"(?i)(?:1\s+)?introduction\s*(.*?)(?=\n\s*\d+\s+\w+|\Z)",
            "methods": r"(?i)(?:\d+\s+)?(?:methods?|methodology)\s*(.*?)(?=\n\s*\d+\s+\w+|\Z)",
            "results": r"(?i)(?:\d+\s+)?results?\s*(.*?)(?=\n\s*\d+\s+\w+|\Z)",
            "conclusion": r"(?i)(?:\d+\s+)?conclusions?\s*(.*?)(?=\n\s*\d+\s+\w+|\Z)",
        }

        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()[:5000]  # Limit length

        return sections

    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse PDF and return structured data.
        When using pdfplumber with extract_tables=True, appends document tables
        as markdown after the main text so the LLM can use them for extraction.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with parsed data
        """
        # Extract raw text
        raw_text = self.extract_text(pdf_path)
        if not raw_text:
            return None

        # Clean text (body only; tables are appended later and not cleaned)
        cleaned_text = self.clean_text(raw_text)

        # Extract tables as markdown (pdfplumber only) and append
        if self.method == "pdfplumber" and self.extract_tables:
            tables = self.extract_tables_pdfplumber(pdf_path)
            if tables:
                tables_md = "\n\n".join(self._table_to_markdown(t) for t in tables)
                if tables_md.strip():
                    cleaned_text = cleaned_text + TABLES_BLOCK_START + tables_md + TABLES_BLOCK_END
                    logger.info(
                        "Appended %d table(s) as markdown to parsed text",
                        len(tables),
                    )
        # Extract sections (from body text only to avoid matching inside tables)
        sections = self.extract_sections(self.clean_text(raw_text))

        result = {
            "pdf_path": str(pdf_path),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "sections": sections,
            "text_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }

        logger.info(
            "Parsed PDF: %d chars, %d words",
            len(cleaned_text),
            result["word_count"],
        )
        return result

    def chunk_text(self, text: str, max_chunk_size: int = 8000, overlap: int = 500) -> list[str]:
        """
        Split text into chunks for processing.

        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end in the last 200 chars
                last_period = text[max(start, end - 200) : end].rfind(". ")
                if last_period != -1:
                    end = max(start, end - 200) + last_period + 1

            chunks.append(text[start:end])
            start = end - overlap

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from paper text.

        Args:
            text: Paper text

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}

        # Extract title (usually at the beginning)
        title_match = re.search(r"^(.+?)\n", text)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        # Extract year
        year_match = re.search(r"\b(20\d{2})\b", text[:1000])
        if year_match:
            metadata["year"] = year_match.group(1)

        # Extract arXiv ID if present
        arxiv_match = re.search(r"arXiv:(\d+\.\d+)", text[:2000])
        if arxiv_match:
            metadata["arxiv_id"] = arxiv_match.group(1)

        return metadata
