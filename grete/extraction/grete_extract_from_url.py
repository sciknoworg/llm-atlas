#!/usr/bin/env python3
"""
Extract LLM information from a PDF URL (e.g., OpenAI research papers)

Usage:
    python grete_extract_from_url.py <pdf_url> [paper_title]
"""

import sys
import json
import torch
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_parser import PDFParser
from src.llm_extractor_transformers import LLMExtractorTransformers
from src.template_mapper import TemplateMapper


def download_pdf(url: str, output_path: Path) -> bool:
    """Download PDF from URL."""
    try:
        print(f"Downloading PDF from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python grete_extract_from_url.py <pdf_url> [paper_title]")
        sys.exit(1)
    
    # Clean and validate URL - remove newlines, extra whitespace, and truncate at first invalid character
    pdf_url = sys.argv[1].strip()
    # Remove any newlines or control characters
    pdf_url = ''.join(c for c in pdf_url if c.isprintable() or c in ['\n', '\r'])
    # Split on newline and take first line (in case command was accidentally included)
    pdf_url = pdf_url.split('\n')[0].split('\r')[0].strip()
    # Remove any trailing command text (e.g., if sbatch command was included)
    if 'sbatch' in pdf_url.lower():
        pdf_url = pdf_url.split('sbatch')[0].strip()
    if 'grete_extract' in pdf_url.lower():
        pdf_url = pdf_url.split('grete_extract')[0].strip()
    
    # Validate URL format
    if not pdf_url.startswith(('http://', 'https://')):
        print(f"✗ Invalid URL format: {pdf_url[:100]}")
        print("URL must start with http:// or https://")
        sys.exit(1)
    
    raw_title = sys.argv[2].strip() if len(sys.argv) > 2 else ""
    paper_title = raw_title if raw_title else "Unknown Paper"
    
    # Create filename from URL
    url_parts = pdf_url.split('/')
    filename = url_parts[-1] if url_parts[-1].endswith('.pdf') else f"paper_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf_path = Path("data/papers") / filename
    
    print("=" * 70)
    print(f"LLM Extraction from PDF URL")
    print("=" * 70)
    print(f"URL: {pdf_url}")
    if len(pdf_url) > 100:
        print(f"URL (full): {pdf_url[:100]}...")
    print(f"Title: {paper_title}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    parser = PDFParser(method="pdfplumber")
    extractor = LLMExtractorTransformers(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",  # 8B model with 128K context window
        temperature=0.3,
        max_new_tokens=1500
    )
    mapper = TemplateMapper()
    print("✓ Components initialized")
    
    # Download PDF
    print(f"\n[2/5] Downloading PDF...")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if not download_pdf(pdf_url, pdf_path):
        sys.exit(1)
    
    # Parse PDF
    print(f"\n[3/5] Parsing PDF...")
    parsed = parser.parse(pdf_path)
    if not parsed:
        print("✗ Failed to parse PDF")
        sys.exit(1)
    
    text = parsed.get("cleaned_text", "") or parsed.get("raw_text", "")
    print(f"✓ Parsed {len(text)} characters")
    
    # Extract with local LLM
    print(f"\n[4/5] Extracting with local LLM on GPU...")
    print("  (This uses Grete's GPU, no API needed)")
    
    # Try to extract date from arXiv URL (e.g., 1901.02860 -> 2019-01)
    year, month = "", ""
    if "arxiv.org" in pdf_url:
        import re
        match = re.search(r'(\d{2})(\d{2})\.\d+', pdf_url)
        if match:
            yy, mm = match.groups()
            year = f"20{yy}"
            month = mm

    result = extractor.extract(
        paper_text=text,
        paper_metadata={
            "title": paper_title,
            "url": pdf_url,
            "authors": [],
            "year": year,
            "month": month
        }
    )
    
    if not result or not result.models:
        print("✗ Extraction failed or no models found")
        sys.exit(1)
    
    print(f"✓ Extracted {len(result.models)} model(s)")
    for i, model in enumerate(result.models, 1):
        print(f"  Model {i}: {model.model_name} ({model.parameters or 'N/A'})")
    
    # Map to ORKG format
    print(f"\n[5/5] Mapping to ORKG template...")
    mapped = mapper.map_extraction_result(result)
    print(f"✓ Mapped {len(mapped.get('contributions', []))} contribution(s)")
    
    # Save results
    output_dir = Path("data/extracted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = filename.replace('.pdf', '').replace('/', '_')
    output_file = output_dir / f"{safe_filename}_{timestamp}.json"
    
    result_data = {
        "paper_url": pdf_url,
        "paper_title": paper_title,
        "paper_metadata": {
            "year": year,
            "month": month,
            "authors": []
        },
        "extracted_at": timestamp,
        "device": str(extractor.device),
        "model_used": extractor.model_name,
        "models_found": len(result.models),
        "raw_extraction": [m.model_dump() for m in result.models],
        "mapped_to_orkg": mapped
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("✓ EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Models extracted: {len(result.models)}")
    print(f"Results saved to: {output_file}")
    print("=" * 70)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

