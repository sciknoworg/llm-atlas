# Bachelor Thesis: Automated Knowledge Extraction for Large-Scale Generative AI Models Catalog

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a Python-based NLP pipeline that automatically extracts structured metadata from Large Language Model (LLM) research papers and populates the ORKG comparison [Generative AI Model Landscape](https://orkg.org/comparisons/R1364660).

The pipeline fetches papers from ArXiv, parses their PDFs, extracts key properties via an LLM (KISSKI Chat AI API), normalises and merges the results, and uploads them to ORKG — all in a single command.

It also supports **HPC cluster** deployment (GPU-based transformers on GWDG Grete) for large-scale batch extraction without API costs.

## Features

- **Automated paper fetching** from ArXiv (by ID or search query)
- **PDF parsing** with fallback mechanisms (`pdfplumber`)
- **LLM-based extraction** via KISSKI Chat AI API (OpenAI-compatible)
- **HPC / GPU extraction** using Hugging Face Transformers on GWDG Grete
- **Extraction normalisation** canonical date, organisation, and parameter formats
- **Model variant merging** groups size variants (e.g. 7B/13B/70B) into one entry
- **Primary contribution selection** filters auxiliary artefacts, keeps main models
- **Structured mapping** to ORKG template [R609825](https://orkg.org/templates/R609825)
- **Automatic ORKG upload** with duplicate detection
- **Strict evaluation framework** match-based F1 + BERTScore for semantic fields
- **Batch processing** for HPC environments

## Project Structure

```
Bachelor-Arbeit-NLP/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI
├── grete/                                  # HPC cluster deployment
│   ├── extraction/
│   │   ├── grete_extract_paper.py
│   │   ├── grete_extract_from_url.py
│   │   └── grete_extract_paper_distilgpt2.py
│   ├── jobs/
│   │   ├── grete_extract_job.sh            # SLURM single job
│   │   ├── grete_extract_batch.sh          # SLURM batch array
│   │   └── grete_extract_url_job.sh
│   └── README.md
├── src/                                    # Core pipeline
│   ├── __init__.py
│   ├── pipeline.py                         # Main orchestration
│   ├── llm_extractor.py                    # KISSKI API extraction
│   ├── llm_extractor_transformers.py       # GPU-based extraction (Grete)
│   ├── pdf_parser.py                       # PDF text extraction
│   ├── paper_fetcher.py                    # ArXiv paper fetching
│   ├── template_mapper.py                  # ORKG template mapping
│   ├── orkg_client.py                      # ORKG API wrapper
│   ├── orkg_manager.py                     # ORKG paper/contribution management
│   ├── extraction_normalizer.py            # Canonical format normalisation
│   ├── model_variant_merger.py             # Size-variant merging
│   ├── model_contribution_selector.py      # Primary model selection
│   ├── baseline_filter.py                  # Baseline/comparison model filter
│   └── comparison_updater.py              # Legacy comparison updater
├── scripts/
│   ├── evaluation/                         # Evaluation scripts and README
│   │   ├── evaluate_extraction_strict.py   # Strict evaluator (thesis metrics)
│   │   ├── evaluate_extraction.py          # Relaxed evaluator
│   │   ├── convert_gold_standard.py        # ORKG CSV → gold JSON converter
│   │   └── README.md
│   ├── batch_extract_all_papers.py
│   ├── build_results_table.py
│   └── import_extracted_to_model_folders.py
├── finetuning/                             # Optional LoRA fine-tuning workflow
├── tests/                                  # Unit tests
│   ├── test_pipeline.py
│   ├── test_llm_extractor.py
│   ├── test_orkg_client.py
│   ├── test_orkg_append.py
│   └── test_pdf_parser.py
├── data/
│   ├── papers/                             # Downloaded PDFs (gitignored)
│   ├── extracted/                          # Extraction results (gitignored)
│   ├── gold_standard/                      # Gold-standard evaluation data
│   └── logs/                              # Pipeline logs (gitignored)
├── results/                                # Aggregated evaluation tables (CSV/JSON)
├── config/
│   └── config.yaml                         # Pipeline configuration
├── .env.example                            # Environment variables template
├── .gitignore
├── LICENSE
├── pyproject.toml                          # Python packaging and tool config
├── requirements.txt                        # Runtime dependencies
└── requirements-dev.txt                    # Development dependencies
```

## Installation

### Prerequisites

- Python 3.9 or higher
- A KISSKI API key (provided by GWDG for thesis work)
- An ORKG account at [sandbox.orkg.org](https://sandbox.orkg.org/) or [orkg.org](https://orkg.org/)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Bachelor-Arbeit-NLP
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the environment template and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

## Configuration

### Environment Variables (`.env`)

```env
# ORKG credentials
ORKG_EMAIL=your.email@example.com
ORKG_PASSWORD=your_password_here

# KISSKI Chat AI API (SAIA platform, OpenAI-compatible)
KISSKI_API_KEY=your_kisski_api_key_here
KISSKI_BASE_URL=https://chat-ai.academiccloud.de/v1
```

> For Grete HPC users: `KISSKI_API_KEY` is not required when running GPU-based extraction. See [grete/README.md](grete/README.md).

### Pipeline Configuration (`config/config.yaml`)

Key settings:

| Section | Key | Description |
|---|---|---|
| `orkg` | `host` | `sandbox` (testing) or `production` |
| `orkg` | `template_id` | ORKG LLM template (`R609825`) |
| `orkg` | `comparison_id` | Target comparison (`R2147679`) |
| `kisski` | `model` | LLM model name (e.g. `qwen3-235b-a22b`) |
| `kisski` | `temperature` | `0.0` for deterministic extraction |
| `extraction` | `max_chunk_size` | Max characters per PDF chunk (default: 6000) |

## Usage

### Process a Single Paper (ArXiv ID)

```bash
python -m src.pipeline --arxiv-id 2302.13971
```

### Process a Paper from PDF URL

```bash
python -m src.pipeline --pdf-url https://example.org/paper.pdf --paper-title "Paper Title"
```

### Upload an Existing Extraction JSON to ORKG

```bash
python -m src.pipeline --json-file data/extracted/2302.13971_20260101_120000.json
```

### Skip ORKG Upload (Extraction Only)

```bash
python -m src.pipeline --arxiv-id 2302.13971 --no-update
```

### Skip Evaluation After Extraction

```bash
python -m src.pipeline --arxiv-id 2302.13971 --no-evaluate
```

### Test Connections

```bash
python -m src.pipeline --test
```

### Python API

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline()
result = pipeline.process_paper("2302.13971")
print(result["extraction_data"])
```

## Evaluation

The evaluation framework measures extraction quality against a gold standard derived from the ORKG comparison [R1364660](https://orkg.org/comparisons/R1364660).

### Run Strict Evaluation

```bash
python scripts/evaluation/evaluate_extraction_strict.py \
    --gold data/gold_standard/R1364660.json \
    --prediction data/extracted/your_extraction.json
```

### Save Evaluation Report

```bash
python scripts/evaluation/evaluate_extraction_strict.py \
    --gold data/gold_standard/R1364660.json \
    --prediction data/extracted/your_extraction.json \
    --output results/evaluation_report.json
```

### Evaluation Metrics

**Structured fields** (model name, parameters, date, organisation, etc.) are evaluated with **match-based F1**:

| Metric | Description |
|---|---|
| Precision | Of all extracted values, how many are correct? |
| Recall | Of all gold values, how many were found? |
| F1-Score | Harmonic mean of precision and recall |

**Semantic fields** (innovation, extension, application, research\_problem, pretraining\_corpus) are evaluated with **BERTScore** (token-level contextual similarity using RoBERTa-large embeddings), which captures paraphrases that exact matching would penalise.

Field-specific matching rules (date year tolerance, organisation aliases, parameter set F1, etc.) are documented in [`scripts/evaluation/README.md`](scripts/evaluation/README.md).

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

## HPC Cluster (Grete)

For large-scale extraction without API limits, the pipeline can run on GWDG Grete using local GPU-based transformers.

```bash
# Submit a single extraction job
sbatch grete/jobs/grete_extract_job.sh 2302.13971

# Submit batch array job
sbatch grete/jobs/grete_extract_batch.sh
```

Full setup guide: [grete/README.md](grete/README.md) and [KISSKI_SETUP.md](KISSKI_SETUP.md).

## Resources

| Resource | Link |
|---|---|
| ORKG Platform | https://orkg.org/ |
| ORKG Sandbox | https://sandbox.orkg.org/ |
| ORKG Python Client | https://orkg.readthedocs.io/en/latest/ |
| LLM Template (R609825) | https://orkg.org/templates/R609825 |
| Target Comparison (R1364660) | https://orkg.org/comparisons/R1364660 |
| KISSKI Chat AI API | https://chat-ai.academiccloud.de |
| ArXiv API | https://info.arxiv.org/help/api/ |
| GWDG Grete HPC | https://info.gwdg.de/docs/doku.php?id=en:services:application_services:high_performance_computing:grete |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ORKG platform ([orkg.org](https://orkg.org/)) for knowledge graph infrastructure
- GWDG for providing HPC resources (Grete cluster) and KISSKI API access
- SAIA platform for Chat AI API services
