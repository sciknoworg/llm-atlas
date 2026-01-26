# GWDG Grete HPC Deployment Guide

This guide covers setting up and running the LLM extraction pipeline on the GWDG Grete HPC cluster using GPU-based transformers.

## Overview

### Benefits of Grete Deployment

✅ **No API costs** - Uses local models on GPU  
✅ **No quotas or rate limits** - Unlimited processing  
✅ **A100 GPU access** - Fast inference with 40GB VRAM  
✅ **Academic infrastructure** - Free HPC resources  
✅ **Scalable** - Process multiple papers in parallel  

### What You Get

- **GPU-based extraction**: Uses Hugging Face transformers on A100 GPUs
- **No API dependencies**: All models run locally
- **SLURM integration**: Batch job management
- **Production-ready scripts**: Pre-configured job templates

---

## Prerequisites

1. **GWDG Grete access** - Account with GPU allocation
2. **SSH configuration** - Set up SSH key-based authentication
3. **Local development** - Code developed and tested locally first

---

## Initial Setup

### Step 1: Connect to Grete

```bash
ssh YOUR_USERNAME@grete.hpc.gwdg.de
```

Or if you have SSH alias configured:
```bash
ssh Grete
```

### Step 2: Create Project Directory

```bash
mkdir -p ~/llm-extraction
cd ~/llm-extraction
```

### Step 3: Set Up Conda Environment

```bash
# Create environment
conda create -n llm-env python=3.10 -y

# Activate environment
conda activate llm-env

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install required packages
pip install transformers accelerate sentencepiece
pip install orkg arxiv pypdf2 pdfplumber python-dotenv pyyaml pydantic requests tqdm
```

### Step 4: Upload Code from Local Machine

From your Windows PC (PowerShell):

```powershell
# Upload main source code
scp -r "C:\Users\ALAKE\OneDrive\Bureau\Bachelor-Arbeit-NLP\src" Grete:~/llm-extraction/

# Upload config
scp -r "C:\Users\ALAKE\OneDrive\Bureau\Bachelor-Arbeit-NLP\config" Grete:~/llm-extraction/

# Upload Grete-specific scripts
scp -r "C:\Users\ALAKE\OneDrive\Bureau\Bachelor-Arbeit-NLP\grete" Grete:~/llm-extraction/

# Upload requirements
scp "C:\Users\ALAKE\OneDrive\Bureau\Bachelor-Arbeit-NLP\requirements.txt" Grete:~/llm-extraction/
```

### Step 5: Create Data Directories

Back on Grete:

```bash
cd ~/llm-extraction
mkdir -p data/{papers,extracted,logs}
```

### Step 6: Make Scripts Executable

```bash
chmod +x grete/jobs/*.sh
```

### Step 7: Configure Environment Variables (Optional)

If you want to upload results to ORKG from Grete:

```bash
# Create .env file
nano .env
```

Add:
```env
ORKG_EMAIL=your-email@example.com
ORKG_PASSWORD=your-password
ORKG_HOST=sandbox
```

---

## Running Extractions

### Single Paper Extraction

```bash
cd ~/llm-extraction
sbatch grete/jobs/grete_extract_job.sh 2302.13971
```

This submits a SLURM job with:
- 1x A100 GPU
- 40GB RAM
- 2 hour time limit
- Internet access (for downloading papers)

### Batch Processing (Multiple Papers)

Edit the batch script to list your papers:

```bash
nano grete/jobs/grete_extract_batch.sh
```

Update the ArXiv IDs:
```bash
ARXIV_IDS=(
    "2302.13971"
    "2307.09288"
    "2203.02155"
    # Add more...
)
```

Submit the batch job:
```bash
sbatch grete/jobs/grete_extract_batch.sh
```

This creates a job array processing papers in parallel.

### Extract from PDF URL

```bash
sbatch grete/jobs/grete_extract_url_job.sh https://arxiv.org/pdf/2302.13971.pdf
```

---

## Monitoring Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# View specific job
squeue -j <JOB_ID>
```

Status codes:
- `PD` - Pending (waiting for resources)
- `R` - Running
- `CG` - Completing
- `CD` - Completed

### View Live Logs

```bash
# Watch the most recent log
tail -f $(ls -t extract_*.log | head -n1)

# Press Ctrl+C to stop
```

### Check Results

```bash
# List extracted files
ls -lt data/extracted/ | head

# View latest extraction
cat $(ls -t data/extracted/*.json | head -n1) | head -100

# Check for errors
cat $(ls -t extract_*.err | head -n1)
```

### Cancel a Job

```bash
scancel <JOB_ID>
```

### Check GPU Usage

While job is running:
```bash
srun --jobid=<JOB_ID> nvidia-smi
```

---

## SLURM Job Configuration

### Single Paper Job Parameters

```bash
#SBATCH --partition=kisski    # GPU partition
#SBATCH -G A100:1             # 1x A100 GPU
#SBATCH --mem=40G             # 40GB RAM
#SBATCH --time=02:00:00       # 2 hour limit
#SBATCH -C inet               # Internet access
```

### Batch Job Parameters

```bash
#SBATCH --array=0-4           # Process 5 papers in parallel
#SBATCH --time=04:00:00       # 4 hour limit per job
```

Adjust based on your needs:
- Increase memory if you get OOM errors
- Increase time limit for long papers
- Adjust array size for more/fewer parallel jobs

---

## Workflow Summary

```
Local Development → Upload to Grete → Submit SLURM Job → Monitor → Download Results
```

### Typical Workflow

1. **Develop locally** - Test pipeline on your machine
2. **Upload changes** - Use scp to update files on Grete
3. **Submit job** - Use sbatch to queue extraction
4. **Monitor** - Check squeue and tail logs
5. **Verify results** - Check data/extracted/ directory
6. **Download** - Copy results back to local machine if needed

---

## Downloading Results

From your local machine:

```powershell
# Download all extracted files
scp -r Grete:~/llm-extraction/data/extracted/*.json "C:\Users\ALAKE\OneDrive\Bureau\Bachelor-Arbeit-NLP\data\extracted\"

# Download logs
scp Grete:~/llm-extraction/extract_*.log "C:\Users\ALAKE\OneDrive\Bureau\Bachelor-Arbeit-NLP\data\logs\"
```

---

## Key Differences from Local API Execution

| Feature | Local (Gemini API) | Grete (GPU) |
|---------|-------------------|-------------|
| **Model** | Google Gemini (cloud) | Phi-3/DistilGPT2 (local) |
| **Cost** | API charges per request | Free (HPC allocation) |
| **Speed** | Fast (cloud infrastructure) | Depends on queue wait |
| **Limits** | API rate limits & quotas | No limits |
| **Setup** | Simple (API key only) | Complex (HPC environment) |
| **Internet** | Always required | Only for downloading papers |
| **Scalability** | Limited by quotas | Limited by GPU allocation |

---

## Best Practices

1. **Test with one paper first** - Verify everything works before batch processing
2. **Monitor resource usage** - Check logs for memory/GPU issues
3. **Use appropriate time limits** - Short papers: 1-2 hours, Long papers: 3-4 hours
4. **Follow professor's guidance** - Process one job at a time when starting
5. **Keep logs** - Save logs for debugging and verification
6. **Regular backups** - Download results regularly to local machine

---

## Troubleshooting

For common issues, see:
- [Phi-3 Cache Issues](../troubleshooting/phi3-cache-issues.md)
- [DistilGPT2 Issues](../troubleshooting/distilgpt2-issues.md)
- [Job Verification Guide](verify-jobs.md)

---

## Next Steps

1. ✅ Follow initial setup steps above
2. ✅ Upload your code to Grete
3. ✅ Test with single paper
4. ✅ Verify extraction quality
5. ✅ Scale to batch processing

Your thesis extraction can now run on Grete's GPUs with unlimited processing capacity!
