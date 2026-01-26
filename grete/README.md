# Grete HPC Deployment

This directory contains all files specific to running the extraction pipeline on the GWDG Grete HPC cluster.

## Overview

The Grete cluster provides GPU resources (A100) for running transformer-based models locally without API costs or quotas. This is particularly useful for processing multiple papers in batch mode.

## Directory Structure

```
grete/
├── extraction/          # Python extraction scripts
│   ├── grete_extract_paper.py              # Main extraction script
│   ├── grete_extract_from_url.py           # Extract from PDF URL
│   └── grete_extract_paper_distilgpt2.py   # Alternative extraction with DistilGPT2
└── jobs/                # SLURM job scripts
    ├── grete_extract_job.sh                # Single paper job
    ├── grete_extract_batch.sh              # Batch job array
    └── grete_extract_url_job.sh            # URL extraction job
```

## Usage

### Prerequisites

1. Access to GWDG Grete cluster
2. Conda environment with PyTorch and transformers installed
3. Code uploaded to Grete

### Running Extractions

**Single paper:**
```bash
sbatch jobs/grete_extract_job.sh 2302.13971
```

**Batch processing:**
```bash
sbatch jobs/grete_extract_batch.sh
```

**From URL:**
```bash
sbatch jobs/grete_extract_url_job.sh https://arxiv.org/pdf/2302.13971.pdf
```

### Monitoring Jobs

```bash
# Check job status
squeue -u YOUR_USERNAME

# View logs
tail -f extract_*.log

# Check results
ls -la ../data/extracted/
```

## Documentation

For detailed setup instructions and troubleshooting, see:
- [`docs/deployment/grete-setup.md`](../docs/deployment/grete-setup.md)
- [`docs/deployment/verify-jobs.md`](../docs/deployment/verify-jobs.md)
- [`docs/troubleshooting/`](../docs/troubleshooting/)

## Key Differences from Local Execution

| Feature | Local (API) | Grete (GPU) |
|---------|------------|-------------|
| Model | Google Gemini API | Local transformers (Phi-3, etc.) |
| Cost | API costs per request | Free (HPC allocation) |
| Speed | Fast (cloud) | Depends on queue |
| Quotas | API rate limits | No quotas |
| Setup | Simple (API key) | Complex (HPC setup) |

## Notes

- These scripts are optimized for the Grete cluster environment
- Extraction scripts use GPU-based transformers instead of API calls
- SLURM job scripts include resource allocation for GPU nodes
- Results are saved to `../data/extracted/`
