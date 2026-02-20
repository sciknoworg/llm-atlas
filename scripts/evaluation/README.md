# Evaluation Scripts

This directory contains scripts for evaluating the extraction pipeline against gold-standard datasets.

## Scripts

### 1. `convert_gold_standard.py`
Converts ORKG comparison CSV exports to gold-standard JSON datasets.

**Usage:**
```bash
python scripts/evaluation/convert_gold_standard.py
```

**Input:** `data/gold_standard/R1364660.csv` (ORKG comparison export)  
**Output:** `data/gold_standard/R1364660.json` (Gold-standard dataset)

### 2. `evaluate_extraction.py`
Evaluates extraction quality by comparing predictions against gold-standard dataset.

**Usage:**
```bash
python scripts/evaluation/evaluate_extraction.py \
    --gold data/gold_standard/R1364660.json \
    --prediction data/extracted/2401.02385_20251207_223913.json
```

**Options:**
- `--gold`: Path to gold-standard JSON (default: `data/gold_standard/R1364660.json`)
- `--prediction`: Path to extraction result JSON (required)
- `--fuzzy-threshold`: Similarity threshold for text matching (default: 0.8)
- `--output`: Optional path to save evaluation report as JSON

**Metrics Calculated:**
- **Accuracy**: Overall correctness = (TP + TN) / Total
- **Precision**: Of all predictions, how many were correct? = TP / (TP + FP)
- **Recall**: Of all gold values, how many were found? = TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall = 2 × (P × R) / (P + R)

**Example Output:**
```
================================================================================
EXTRACTION EVALUATION REPORT
================================================================================

Summary:
  Gold-standard models:    98
  Predicted models:        45
  Matched models:          42
  Unmatched predictions:   3
  Missing models:          56

================================================================================
OVERALL METRICS (All Fields Combined)
================================================================================
  Accuracy:        78.50%
  Precision:       82.30%
  Recall:          75.20%
  F1-Score:        78.60%

  True Positives:  456
  False Positives: 102
  False Negatives: 150
  True Negatives:  892

================================================================================
PER-FIELD METRICS
================================================================================
Field                          Accuracy     Precision    Recall       F1-Score    
--------------------------------------------------------------------------------
model_name                     95.00%       98.00%       92.00%       95.00%
model_family                   88.50%       90.00%       87.00%       88.50%
parameters                     85.20%       87.50%       83.00%       85.20%
...
```

## Evaluation Workflow

1. **Prepare Gold-Standard Dataset**
   ```bash
   # Export ORKG comparison R1364660 as CSV
   # Place CSV in data/gold_standard/R1364660.csv
   
   # Convert to JSON
   python scripts/evaluation/convert_gold_standard.py
   ```

2. **Run Extraction Pipeline**
   ```bash
   # Extract from a paper
   python -m src.pipeline --arxiv-id 2401.02385
   
   # Or use existing extraction
   # Results in: data/extracted/2401.02385_YYYYMMDD_HHMMSS.json
   ```

3. **Evaluate Extraction Quality**
   ```bash
   python scripts/evaluation/evaluate_extraction.py \
       --prediction data/extracted/2401.02385_20251207_223913.json \
       --output data/evaluation_reports/report_20260126.json
   ```

4. **Analyze Results**
   - Review overall F1-score (target: ≥ 80%)
   - Identify low-performing fields
   - Adjust extraction prompts/logic to improve weak areas
   - Re-run evaluation

## Understanding Metrics

### Confusion Matrix for Each Field
```
                    Gold=True   Gold=False
Predicted=True      TP          FP
Predicted=False     FN          TN
```

- **TP (True Positive)**: Field exists in gold, correctly extracted
- **FP (False Positive)**: Field extracted but doesn't match gold
- **FN (False Negative)**: Field exists in gold but not extracted or wrong
- **TN (True Negative)**: Field doesn't exist in gold, not extracted

### Example: model_name field
- Gold: "BERT", Predicted: "BERT" → TP
- Gold: "BERT", Predicted: "GPT-2" → FP
- Gold: "BERT", Predicted: None → FN
- Gold: None, Predicted: None → TN

### Fuzzy Matching
For long text fields (innovation, pretraining_corpus, etc.), the evaluator uses fuzzy string matching with a configurable threshold (default 80% similarity).

### Strict evaluator: smart matching (`evaluate_extraction_strict.py`)
The strict evaluator uses **field-specific rules** so that equivalent information is credited:
- **date_created:** Same year counts as a match (e.g. "2018" vs "2018-10-01"); if both have month, same year+month required.
- **organization:** Exact match, or one string contains the other (e.g. "Google" vs "Google AI Language"), or known aliases (e.g. "Meta AI" vs "Meta").
- **parameters:** Set-based comparison (comma-separated sizes); match when overlap meets threshold.
- **parameters_millions:** Numeric equality (or both null).
Long-text fields still use semantic/fuzzy similarity. This yields a more reliable metric (e.g. BERT date and organization no longer 0% when the information is correct).

## Extraction format and normalizer

The pipeline **normalizes** extraction output before merge and evaluation so stored data is consistent:
- **date_created:** "2018" → "2018-01", "2018-10-01" → "2018-10" (YYYY-MM).
- **organization:** Variants like "Google AI Language" → "Google", "Meta AI" → "Meta" (see `src/extraction_normalizer.py`).

Extraction prompts also ask for **canonical formats**: date as YYYY-MM, organization as canonical name, parameters as comma-separated sizes. Together, extraction rules + normalizer + smart evaluation give a more reliable workflow.

## Improving Extraction Quality

Based on evaluation results:

1. **Low precision** → Extraction is too aggressive (hallucinating values)
   - Tighten prompts
   - Add validation rules
   - Use stricter JSON parsing

2. **Low recall** → Extraction is missing values
   - Expand context window
   - Improve prompt instructions
   - Add few-shot examples

3. **Field-specific issues** → Check ORKG template alignment
   - Review field definitions
   - Update field name normalization
   - Check for synonym handling

## Next Steps

- Run evaluation on multiple papers to get average metrics
- Compare KISSKI API vs. Grete HPC extraction quality
- Track metrics over time as prompts are improved
- Set up automated evaluation in CI/CD
