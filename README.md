# Bachelor Thesis: Automated Knowledge Extraction for Large-Scale Generative AI Models Catalog

[![CI](https://github.com/yourusername/Bachelor-Arbeit-NLP/workflows/CI/badge.svg)](https://github.com/yourusername/Bachelor-Arbeit-NLP/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a Python-based NLP workflow that automatically extracts key information from Large Language Model (LLM) research papers and populates the ORKG comparison "Generative AI Model Landscape".

The pipeline supports both **API-based** extraction (using KISSKI Chat AI API) and **HPC cluster** deployment (using GPU-based transformers on GWDG Grete).

## Features

- **Automated paper fetching** from ArXiv
- **PDF parsing** and text extraction with fallback mechanisms
- **Dual extraction modes**:
  - API-based using KISSKI Chat AI API (free for thesis work)
  - GPU-based using Hugging Face transformers on HPC cluster
- **Structured data mapping** to ORKG templates
- **Automatic ORKG updates** with duplicate detection
- **Multi-model extraction** from single papers
- **Batch processing** support for HPC environments

## Project Structure

```
Bachelor-Arbeit-NLP/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD
├── docs/
│   ├── deployment/
│   │   ├── grete-setup.md           # HPC cluster setup guide
│   │   └── verify-jobs.md           # Job verification guide
│   ├── troubleshooting/
│   │   ├── phi3-cache-issues.md     # Phi-3 troubleshooting
│   │   └── distilgpt2-issues.md     # DistilGPT2 troubleshooting
│   └── archive/
│       └── cleanup-summary-2025-12.md
├── grete/                            # HPC cluster deployment
│   ├── extraction/
│   │   ├── grete_extract_paper.py
│   │   ├── grete_extract_from_url.py
│   │   └── grete_extract_paper_distilgpt2.py
│   ├── jobs/
│   │   ├── grete_extract_job.sh     # SLURM single job
│   │   ├── grete_extract_batch.sh   # SLURM batch array
│   │   └── grete_extract_url_job.sh
│   └── README.md                     # Grete-specific documentation
├── src/                              # Core pipeline
│   ├── __init__.py
│   ├── comparison_updater.py         # Update ORKG comparisons
│   ├── llm_extractor.py              # KISSKI API extraction
│   ├── llm_extractor_transformers.py # GPU-based extraction
│   ├── orkg_client.py                # ORKG API wrapper
│   ├── orkg_manager.py               # ORKG management
│   ├── paper_fetcher.py              # ArXiv paper fetching
│   ├── pdf_parser.py                 # PDF text extraction
│   ├── pipeline.py                   # Main orchestration
│   └── template_mapper.py            # ORKG template mapping
├── scripts/
│   ├── append_to_paper.py            # Production utilities
│   ├── sandbox_upload.py
│   └── debug/                        # Debug/testing scripts
│       ├── add_to_orkg_manual.py
│       ├── export_to_csv.py
│       └── force_new_paper_upload.py
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_llm_extractor.py
│   ├── test_orkg_append.py
│   ├── test_orkg_client.py
│   ├── test_pdf_parser.py
│   └── test_pipeline.py
├── examples/
│   └── example_usage.py              # Usage examples
├── data/
│   ├── papers/                       # Downloaded PDFs
│   ├── extracted/                    # Extracted JSON data
│   └── logs/                         # Processing logs
├── notebooks/
│   └── validate_extraction.ipynb     # Validation notebook
├── config/
│   └── config.yaml                   # Configuration
├── .env.example                      # Environment variables template
├── .gitignore
├── LICENSE                           # MIT License
├── pyproject.toml                    # Modern Python packaging
├── README.md
├── requirements.txt                  # Production dependencies
└── requirements-dev.txt              # Development dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

## Configuration

### Environment Variables (`.env`)

Required variables:
```env
ORKG_EMAIL=your.email@example.com
ORKG_PASSWORD=your_password_here
ORKG_HOST=sandbox  # or 'production'
KISSKI_API_KEY=your_kisski_api_key_here
KISSKI_API_ENDPOINT=https://kisski.de/api/chat
```

### Pipeline Configuration (`config/config.yaml`)

Customize the extraction pipeline:
- **ORKG settings**: host, template ID, comparison ID
- **KISSKI API**: endpoint, model, temperature, max tokens, rate limiting
- **ArXiv settings**: categories, download directory
- **Extraction fields**: which fields to extract
- **Logging**: level, format, output files

## Usage

### Basic Usage

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline()
pipeline.process_paper("2302.13971")  # ArXiv ID for Llama paper
```

### Command Line

```bash
python -m src.pipeline --arxiv-id 2302.13971
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_pipeline.py -v
```

### Continuous Integration

The project uses GitHub Actions for automated testing. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the CI configuration.

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

## Deployment

### Local Extraction (API-based)

Uses KISSKI Chat AI API for extraction. Fast and reliable, free for thesis work.

**Prerequisites**: KISSKI API key from your professor

See [Usage](#usage) section above for examples.

### HPC Cluster (GPU-based)

Uses local transformers on GPU. No API costs, unlimited processing, but requires HPC setup.

**Full deployment guide**: [docs/deployment/grete-setup.md](docs/deployment/grete-setup.md)

**Quick start**:
1. Upload code to Grete cluster
2. Set up conda environment with PyTorch + transformers
3. Submit SLURM jobs:
   ```bash
   sbatch grete/jobs/grete_extract_job.sh 2302.13971
   ```

**Monitoring**: [docs/deployment/verify-jobs.md](docs/deployment/verify-jobs.md)

## Troubleshooting

Common issues and solutions:

- **Phi-3 cache compatibility**: [docs/troubleshooting/phi3-cache-issues.md](docs/troubleshooting/phi3-cache-issues.md)
- **DistilGPT2 JSON errors**: [docs/troubleshooting/distilgpt2-issues.md](docs/troubleshooting/distilgpt2-issues.md)
- **Job verification**: [docs/deployment/verify-jobs.md](docs/deployment/verify-jobs.md)

## Development

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/

# Sort imports
isort src/
```

### Project Structure

- **`src/`**: Core pipeline code
- **`tests/`**: Unit tests
- **`scripts/`**: Utility scripts
- **`grete/`**: HPC cluster deployment files
- **`docs/`**: Documentation
- **`examples/`**: Usage examples
- **`notebooks/`**: Jupyter notebooks for exploration

### Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/`
4. Run linters: `black src/ && flake8 src/`
5. Submit a pull request

## Resources

### ORKG

- **Platform**: https://orkg.org/
- **Sandbox**: https://sandbox.orkg.org/
- **Python Client**: https://orkg.readthedocs.io/en/latest/
- **LLM Template**: https://orkg.org/templates/R609825
- **Target Comparison**: https://orkg.org/comparisons/R1364660

### APIs

- **KISSKI Chat AI API**: Provided by university for thesis work
- **ArXiv API**: https://info.arxiv.org/help/api/

### HPC

- **GWDG Grete**: https://info.gwdg.de/docs/doku.php?id=en:services:application_services:high_performance_computing:grete

## Citation

If you use this code for your research, please cite:

```bibtex
@thesis{bachelor2025llm,
  title={Automated Knowledge Extraction for Large-Scale Generative AI Models Catalog},
  author={Your Name},
  year={2025},
  school={Your University},
  type={Bachelor's Thesis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed as part of a Bachelor thesis project
- ORKG platform for knowledge graph infrastructure
- GWDG for providing HPC resources and KISSKI API access
- SAIA platform for Chat AI API services

## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).

