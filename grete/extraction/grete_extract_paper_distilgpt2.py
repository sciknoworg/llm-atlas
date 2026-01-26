#!/usr/bin/env python3
"""
LLM Extraction using distilgpt2 (from official GWDG Grete documentation)

This uses the model explicitly mentioned in the GWDG docs, guaranteed to work.
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.paper_fetcher import PaperFetcher
from src.pdf_parser import PDFParser
from src.llm_extractor_transformers import LLMExtractorTransformers
from src.template_mapper import TemplateMapper


def main():
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2302.13971"
    
    print("=" * 70)
    print(f"LLM Extraction on Grete GPU (using distilgpt2)")
    print("=" * 70)
    print(f"Paper: {arxiv_id}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    fetcher = PaperFetcher(download_dir="data/papers")
    parser = PDFParser(method="pdfplumber")
    
    # Use distilgpt2 - from official GWDG documentation
    extractor = LLMExtractorTransformers(
        model_name="distilgpt2",
        temperature=0.7,
        max_new_tokens=500
    )
    mapper = TemplateMapper()
    print("✓ Components initialized")
    
    # Fetch paper
    print(f"\n[2/5] Fetching paper {arxiv_id} from ArXiv...")
    paper = fetcher.fetch_paper(arxiv_id, download_pdf=True)
    if not paper:
        print(f"✗ Failed to fetch paper")
        sys.exit(1)
    print(f"✓ Fetched: {paper['title']}")
    
    # Parse PDF
    print(f"\n[3/5] Parsing PDF...")
    pdf_path = Path(paper["pdf_path"])
    parsed = parser.parse(pdf_path)
    if not parsed:
        print("✗ Failed to parse PDF")
        sys.exit(1)
    
    text = parsed.get("cleaned_text", "") or parsed.get("raw_text", "")
    print(f"✓ Parsed {len(text)} characters")
    
    # Extract with local LLM
    print(f"\n[4/5] Extracting with local LLM on GPU...")
    print("  (Using distilgpt2 from GWDG documentation)")
    
    result = extractor.extract(
        paper_text=text,
        paper_metadata={
            "title": paper["title"],
            "arxiv_id": arxiv_id,
            "authors": paper.get("authors", [])
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
    output_file = output_dir / f"{arxiv_id.replace('/', '_')}_{timestamp}.json"
    
    result_data = {
        "arxiv_id": arxiv_id,
        "paper_title": paper["title"],
        "extracted_at": timestamp,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_used": "distilgpt2",
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









