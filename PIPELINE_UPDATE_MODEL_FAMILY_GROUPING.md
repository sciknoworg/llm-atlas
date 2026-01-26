# Pipeline Update: Model Family Grouping

## Summary

Updated the KISSKI API pipeline to match the Grete HPC pipeline's correct behavior:
- **ONE paper per MODEL FAMILY** (not per research paper)
- **Each model VERSION as a separate contribution**

## What Changed

### Files Modified

1. **`src/pipeline.py`**
   - **Replaced**: `ComparisonUpdater` → `ORKGPaperManager`
   - **Added**: Helper methods `_extract_year()` and `_extract_month()`
   - **Updated**: ORKG upload logic to use model family grouping
   - **Fixed**: Default config fallback (gemini → kisski)
   - **Fixed**: Status method (gemini → kisski)

2. **`src/__init__.py`**
   - **Added**: `ORKGPaperManager` to exports
   - **Marked**: `ComparisonUpdater` as deprecated

3. **`src/comparison_updater.py`**
   - **Removed**: Testing timestamp suffix (no longer needed)

## How It Works Now

### Previous Behavior (WRONG)

```
Research Paper: "Llama 2: Open Foundation and Fine-Tuned Chat Models"
└── Contribution 1: Llama 2-Chat
└── Contribution 2: Llama 2-Chat 7B
└── Contribution 3: Llama 2-Chat 34B
└── Contribution 4: Llama 2-Chat 70B
```

**Problem**: One paper per research publication (not scalable)

### New Behavior (CORRECT - Matches Grete)

```
Paper: "Llama 2"  (model family)
├── Contribution 1: Llama 2 7B
├── Contribution 2: Llama 2 13B
├── Contribution 3: Llama 2 34B
└── Contribution 4: Llama 2 70B

Paper: "GPT-4"  (different model family)
├── Contribution 1: GPT-4
├── Contribution 2: GPT-4 Turbo
└── Contribution 3: GPT-4 Vision
```

**Benefits**:
- Models grouped by family (logical organization)
- Multiple research papers can contribute to the same model family
- New versions added to existing papers automatically

## Key Mechanism: LLM-Extracted `paper_title`

The LLM extractor includes a `paper_title` field for each extracted model:

```json
{
  "model_name": "Llama 2 7B",
  "model_family": "Llama",
  "paper_title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
  "parameters": "7B",
  ...
}
```

The `ORKGPaperManager`:
1. Uses the LLM-extracted `paper_title` (from the first model)
2. Searches ORKG for a paper with that title
3. If found: adds new model versions as contributions
4. If not found: creates a new paper with all versions

This allows the same model family from different research papers to be grouped together.

## Usage

### Extract and Upload

```bash
# Extract paper and upload to ORKG (groups by model family)
python -m src.pipeline --arxiv-id 2307.09288

# Upload existing extraction JSON (uses model family grouping)
python -m src.pipeline --json-file data/extracted/2307.09288_20251130_230219.json
```

### Expected Behavior

When processing a paper about "Llama 2" with versions 7B, 13B, 70B:

1. **Extraction**: Extracts 3 models with `paper_title` = "Llama 2: Open Foundation..."
2. **ORKG Search**: Searches for paper titled "Llama 2: Open Foundation..."
3. **Paper Creation**:
   - If paper exists: Adds new versions as contributions (no duplicates)
   - If paper doesn't exist: Creates paper with all versions
4. **Comparison Update**: Links contributions to comparison R1364660

## Differences from Grete Pipeline

| Feature | Grete Pipeline | KISSKI Pipeline |
|---------|----------------|-----------------|
| **Extraction** | GPU-based transformers | KISSKI API (cloud) |
| **ORKG Upload** | ORKGPaperManager | ORKGPaperManager ✓ |
| **Model Grouping** | By model family ✓ | By model family ✓ |
| **Paper Title** | LLM-extracted ✓ | LLM-extracted ✓ |
| **Duplicate Detection** | Yes ✓ | Yes ✓ |

**Now both pipelines use the same ORKG logic!**

## Testing

To verify the changes work correctly:

```bash
# 1. Run extraction (ArXiv is currently rate-limited, so use existing JSON)
python -m src.pipeline --json-file data/extracted/2307.09288_20251130_230219.json

# 2. Check logs
cat data/logs/pipeline.log | grep "model family"

# 3. Verify in ORKG sandbox
# Visit: https://sandbox.orkg.org/comparison/R1364660
```

## Benefits

1. **Consistency**: KISSKI and Grete pipelines now use the same logic
2. **Correct Organization**: Models grouped by family (matches ORKG template design)
3. **No Duplicates**: Automatically adds new versions to existing papers
4. **Scalability**: Multiple research papers can contribute to the same model family

## What Was Deprecated

- **`ComparisonUpdater`**: Still exists but deprecated
  - Only creates papers based on research paper titles (wrong approach)
  - Use `ORKGPaperManager` instead

## Next Steps

1. Test the updated pipeline with a new paper
2. Verify model family grouping works correctly
3. Check ORKG sandbox to see papers organized by family
4. Remove `ComparisonUpdater` in future cleanup (after confirming no other dependencies)
