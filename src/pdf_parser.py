"""
PDF Parser

This module handles extracting and preprocessing text from PDF files.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any
import PyPDF2
import pdfplumber

logger = logging.getLogger(__name__)


class PDFParser:
    """Parses PDF files and extracts text."""
    
    def __init__(self, method: str = "pdfplumber"):
        """
        Initialize PDF parser.
        
        Args:
            method: Parsing method ("pypdf2" or "pdfplumber")
        """
        self.method = method
        logger.info(f"Initialized PDFParser with method: {method}")
    
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
            with open(pdf_path, 'rb') as file:
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
            logger.warning(f"Primary method failed or extracted too little text, trying fallback")
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
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
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
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with parsed data
        """
        # Extract raw text
        raw_text = self.extract_text(pdf_path)
        if not raw_text:
            return None
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Extract sections
        sections = self.extract_sections(cleaned_text)
        
        result = {
            "pdf_path": str(pdf_path),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "sections": sections,
            "text_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }
        
        logger.info(f"Parsed PDF: {len(cleaned_text)} chars, {result['word_count']} words")
        return result
    
    def chunk_text(
        self,
        text: str,
        max_chunk_size: int = 8000,
        overlap: int = 500
    ) -> list[str]:
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
                last_period = text[max(start, end-200):end].rfind('. ')
                if last_period != -1:
                    end = max(start, end-200) + last_period + 1
            
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
        title_match = re.search(r'^(.+?)\n', text)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', text[:1000])
        if year_match:
            metadata['year'] = year_match.group(1)
        
        # Extract arXiv ID if present
        arxiv_match = re.search(r'arXiv:(\d+\.\d+)', text[:2000])
        if arxiv_match:
            metadata['arxiv_id'] = arxiv_match.group(1)
        
        return metadata

