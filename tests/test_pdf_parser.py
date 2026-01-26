"""Tests for PDF parser."""

import pytest
from pathlib import Path
from src.pdf_parser import PDFParser


class TestPDFParser:
    """Test cases for PDFParser."""
    
    def test_init(self):
        """Test initialization."""
        parser = PDFParser(method="pdfplumber")
        assert parser.method == "pdfplumber"
    
    def test_clean_text(self):
        """Test text cleaning."""
        parser = PDFParser()
        
        raw_text = "This  is   a    test.\n\n\n\nWith   multiple   spaces."
        cleaned = parser.clean_text(raw_text)
        
        assert "  " not in cleaned
        assert "\n\n\n" not in cleaned
    
    def test_chunk_text(self):
        """Test text chunking."""
        parser = PDFParser()
        
        text = "A" * 10000
        chunks = parser.chunk_text(text, max_chunk_size=3000, overlap=100)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 3000 for chunk in chunks)
    
    def test_chunk_text_small(self):
        """Test chunking small text."""
        parser = PDFParser()
        
        text = "Short text"
        chunks = parser.chunk_text(text, max_chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_extract_metadata_from_text(self):
        """Test metadata extraction."""
        parser = PDFParser()
        
        text = "Test Paper Title\nAuthors: John Doe\narXiv:2307.09288\nYear: 2023"
        metadata = parser.extract_metadata_from_text(text)
        
        assert 'title' in metadata
        assert 'year' in metadata
        assert metadata['year'] == '2023'

