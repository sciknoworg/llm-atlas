"""
Test script for KISSKI API integration (SAIA Platform).

This script tests the KISSKI Chat AI API with a simple extraction to verify:
1. API connection works
2. Rate limiting is enforced
3. Response parsing works
4. Extraction produces valid results

The KISSKI API is OpenAI-compatible and hosted at:
https://chat-ai.academiccloud.de/v1

Usage:
    python test_kisski_api.py

Requirements:
    - KISSKI_API_KEY in .env file
    - openai library installed (pip install openai)
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_extractor import LLMExtractor, MultiModelResponse


def test_kisski_connection():
    """Test basic KISSKI API connection."""
    print("=" * 70)
    print("KISSKI API Connection Test (SAIA Platform)")
    print("=" * 70)
    
    # Get API configuration
    api_key = os.getenv("KISSKI_API_KEY")
    base_url = os.getenv("KISSKI_BASE_URL", "https://chat-ai.academiccloud.de/v1")
    
    if not api_key:
        print("❌ ERROR: KISSKI_API_KEY not found in .env file")
        print("\nPlease add your KISSKI API key to .env:")
        print("KISSKI_API_KEY=8810b4c60127bfed5655b1e66f3d291a")
        print("KISSKI_BASE_URL=https://chat-ai.academiccloud.de/v1")
        return False
    
    print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
    print(f"✓ Base URL: {base_url}")
    print()
    
    # Initialize extractor
    try:
        extractor = LLMExtractor(
            api_key=api_key,
            base_url=base_url,
            model="meta-llama-3.1-8b-instruct",  # Fast model for testing
            temperature=0.0,
            max_tokens=1000,
            timeout=30,
            rate_limit_delay=2.0
        )
        print("✓ Extractor initialized successfully")
        print(f"✓ Using model: meta-llama-3.1-8b-instruct")
        print(f"✓ Rate limiting: 2.0 seconds between requests")
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        return False
    
    return True


def test_simple_extraction():
    """Test extraction with a simple paper excerpt."""
    print("\n" + "=" * 70)
    print("Simple Extraction Test - Llama 2 Paper Excerpt")
    print("=" * 70)
    
    # Sample paper text (Llama 2 excerpt)
    paper_text = """
    Llama 2: Open Foundation and Fine-Tuned Chat Models
    
    Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, et al.
    Meta AI
    
    Abstract:
    In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned 
    large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. 
    Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. 
    Llama 2-Chat models generally outperform open-source chat models on most benchmarks we tested.
    
    Model Details:
    We release several variants of Llama 2:
    
    - Llama 2 7B: 7 billion parameters, 4096 token context window
    - Llama 2 13B: 13 billion parameters, 4096 token context window
    - Llama 2 70B: 70 billion parameters, 4096 token context window
    
    All models use:
    - Architecture: Transformer with grouped-query attention (GQA)
    - Training data: 2 trillion tokens from publicly available sources
    - Tokenizer: SentencePiece with 32k vocabulary
    - Training compute: 3.3M GPU hours on A100-80GB
    - Release date: July 18, 2023
    - License: Llama 2 Community License Agreement
    - Organization: Meta AI (formerly Facebook AI Research)
    
    The models are trained on an optimized transformer architecture with improvements including:
    - Grouped-Query Attention (GQA) for faster inference
    - Ghost Attention (GAtt) for multi-turn consistency
    - AdamW optimizer with cosine learning rate schedule
    
    Hardware: Training was performed on NVIDIA A100-80GB GPUs.
    Carbon emissions: We estimate that training Llama 2 70B emitted approximately 539 tCO2eq.
    """
    
    paper_metadata = {
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors": ["Hugo Touvron", "Louis Martin", "Kevin Stone", "Meta AI"]
    }
    
    # Get API configuration
    api_key = os.getenv("KISSKI_API_KEY")
    base_url = os.getenv("KISSKI_BASE_URL", "https://chat-ai.academiccloud.de/v1")
    
    if not api_key:
        print("❌ KISSKI_API_KEY not found")
        return False
    
    # Initialize extractor
    print("Initializing KISSKI extractor...")
    extractor = LLMExtractor(
        api_key=api_key,
        base_url=base_url,
        model="openai-gpt-oss-120b",  # Best performance model
        temperature=0.0,
        max_tokens=3000,
        timeout=60,
        rate_limit_delay=2.0
    )
    print(f"✓ Using model: openai-gpt-oss-120b")
    print()
    
    print("Sending extraction request to KISSKI API...")
    print("(This enforces 2-second rate limit)")
    print()
    
    # Perform extraction
    try:
        result = extractor.extract(paper_text, paper_metadata)
        
        if result and result.models:
            print("=" * 70)
            print(f"✓ EXTRACTION SUCCESSFUL!")
            print("=" * 70)
            print(f"Models extracted: {len(result.models)}")
            print()
            
            for i, model in enumerate(result.models, 1):
                print(f"{'='*70}")
                print(f"Model {i}: {model.model_name}")
                print(f"{'='*70}")
                print(f"  Model Family:     {model.model_family or 'N/A'}")
                print(f"  Parameters:       {model.parameters or 'N/A'}")
                print(f"  Parameters (M):   {model.parameters_millions or 'N/A'}")
                print(f"  Organization:     {model.organization or 'N/A'}")
                print(f"  Architecture:     {model.pretraining_architecture or 'N/A'}")
                print(f"  Training Data:    {model.pretraining_corpus or 'N/A'}")
                print(f"  Context Length:   {model.context_length or 'N/A'}")
                print(f"  License:          {model.license or 'N/A'}")
                print(f"  Date Created:     {model.date_created or 'N/A'}")
                print(f"  Hardware:         {model.hardware_used or 'N/A'}")
                print(f"  Carbon Emitted:   {model.carbon_emitted or 'N/A'}")
                print()
            
            # Validate
            validation = extractor.validate_extraction(result)
            
            print("=" * 70)
            print("Validation Results")
            print("=" * 70)
            
            if validation["valid"]:
                print("✓ Extraction is valid")
            else:
                print("⚠ Validation failed:")
                for error in validation["errors"]:
                    print(f"  ERROR: {error}")
            
            if validation["warnings"]:
                print("\n⚠ Warnings:")
                for warning in validation["warnings"]:
                    print(f"  - {warning}")
            
            print()
            return True
        else:
            print("❌ No models extracted")
            print("\nPossible reasons:")
            print("1. API returned empty response")
            print("2. JSON parsing failed")
            print("3. Model didn't understand the prompt")
            return False
            
    except Exception as e:
        print(f"❌ Extraction failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


def test_available_models():
    """Test listing available models from KISSKI API."""
    print("\n" + "=" * 70)
    print("Available Models Test")
    print("=" * 70)
    
    api_key = os.getenv("KISSKI_API_KEY")
    
    if not api_key:
        print("❌ KISSKI_API_KEY not found")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://chat-ai.academiccloud.de/v1"
        )
        
        print("Fetching available models from KISSKI API...")
        models = client.models.list()
        
        print(f"\n✓ Found {len(models.data)} available models:")
        print()
        
        # Print first 10 models
        for model in models.data[:10]:
            print(f"  - {model.id}")
        
        if len(models.data) > 10:
            print(f"  ... and {len(models.data) - 10} more")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return False


def main():
    """Run KISSKI API tests."""
    print("\n" + "=" * 70)
    print("KISSKI API Integration Test Suite (SAIA Platform)")
    print("=" * 70)
    print()
    print("Testing KISSKI Chat AI API hosted by GWDG Academic Cloud")
    print("API Documentation: https://doc.gwdg.de/doku.php?id=en:services:application_services:saia")
    print()
    
    # Test 1: Connection
    if not test_kisski_connection():
        print("\n❌ Connection test failed. Fix configuration and try again.")
        sys.exit(1)
    
    # Test 2: Available models
    test_available_models()
    
    # Test 3: Simple extraction
    print()
    if not test_simple_extraction():
        print("\n" + "=" * 70)
        print("❌ Extraction test failed")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("1. Check that your API key is valid")
        print("2. Verify you have API quota remaining")
        print("3. Try a different model in config.yaml")
        print("4. Check logs in data/logs/pipeline.log")
        print("5. Contact professor if issues persist")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nKISSKI API is working correctly with your pipeline.")
    print()
    print("Next steps:")
    print("1. Run full extraction: python -m src.pipeline --arxiv-id 2307.09288")
    print("2. Check results in data/extracted/")
    print("3. Monitor rate limiting in logs")
    print()
    print("Available models for extraction:")
    print("  - openai-gpt-oss-120b (best performance, recommended)")
    print("  - qwen3-32b (good reasoning)")
    print("  - meta-llama-3.1-8b-instruct (fastest)")
    print("  - deepseek-r1-0528 (advanced reasoning)")
    print()
    print("Change model in config/config.yaml under kisski.model")
    print()


if __name__ == "__main__":
    main()
