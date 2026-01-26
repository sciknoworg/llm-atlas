"""
Example usage of the LLM extraction pipeline.

This script demonstrates how to use the pipeline to extract information
from LLM papers and update the ORKG comparison.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ExtractionPipeline


def main():
    """Run example extraction pipeline."""
    
    # Initialize pipeline
    print("Initializing extraction pipeline...")
    pipeline = ExtractionPipeline()
    
    # Example 1: Process a single paper by ArXiv ID
    print("\n=== Example 1: Process Llama 2 paper ===")
    arxiv_id = "2307.09288"  # Llama 2 paper
    
    try:
        result = pipeline.process_paper(arxiv_id)
        print(f"Successfully processed paper: {arxiv_id}")
        print(f"Extracted {len(result.get('models', []))} model(s)")
    except Exception as e:
        print(f"Error processing paper: {e}")
    
    # Example 2: Process multiple papers
    print("\n=== Example 2: Process multiple papers ===")
    arxiv_ids = [
        "2307.09288",  # Llama 2
        "2302.13971",  # Llama 1
        "2203.02155",  # Chinchilla
    ]
    
    for arxiv_id in arxiv_ids:
        try:
            print(f"\nProcessing {arxiv_id}...")
            result = pipeline.process_paper(arxiv_id)
            print(f"✓ Success: {arxiv_id}")
        except Exception as e:
            print(f"✗ Error: {arxiv_id} - {e}")


if __name__ == "__main__":
    main()

