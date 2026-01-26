# Troubleshooting: Phi-3 Model Cache Compatibility Issue

## Problem Overview

### Symptoms

- Extraction job fails during model inference
- Error message: `'DynamicCache' object has no attribute 'seen_tokens'`
- Paper fetching and PDF parsing work correctly
- GPU is detected and accessible
- Model loads without errors, but fails during generation

### What's Working

✅ PyTorch with CUDA support installed correctly  
✅ Transformers library installed  
✅ GPU (A100) detected and accessible  
✅ Model loads successfully  
✅ Paper fetching and PDF parsing work  

### What's Failing

❌ Model inference/generation fails  
❌ No extraction results created  
❌ Same error occurs on every run  

---

## Root Cause Analysis

The error occurs because cached Phi-3 model code is incompatible with transformers 4.57.3.

**Technical Details:**

1. HuggingFace caches model code locally to speed up loading
2. The cached Phi-3 code was written for an older transformers version
3. The cached code tries to access `past_key_values.seen_tokens` which doesn't exist in newer versions
4. Even with `attn_implementation="eager"`, the cached code overrides the setting

**Problematic File Location:**
```
~/.cache/huggingface/modules/transformers_modules/microsoft/Phi_hyphen_3_hyphen_mini_hyphen_4k_hyphen_instruct/
```

**Problematic Code:**
```python
# Line 1291 in modeling_phi3.py
past_length = past_key_values.seen_tokens  # This line fails
```

---

## Solution 1: Clear Cache (⭐ Recommended)

This solution removes the incompatible cached code and forces a fresh download.

### Why This Works

- Removes the outdated cached model code
- Forces HuggingFace to download the latest compatible version
- Keeps using Phi-3 (better extraction quality than smaller models)
- Usually resolves the issue permanently

### Steps (5-10 minutes)

#### 1. Connect to Grete

```bash
ssh Grete
cd ~/llm-extraction
```

#### 2. Clear the Incompatible Cache

```bash
# Remove the cached Phi-3 model code
rm -rf ~/.cache/huggingface/modules/transformers_modules/microsoft/Phi_hyphen_3_hyphen_mini_hyphen_4k_hyphen_instruct/

# Verify it's removed
ls ~/.cache/huggingface/modules/transformers_modules/microsoft/
```

You should see: `ls: cannot access ... No such file or directory` (This is good!)

#### 3. Resubmit the Job

```bash
sbatch grete/jobs/grete_extract_job.sh 2302.13971
```

Note the job ID from the output: `Submitted batch job XXXXX`

#### 4. Monitor the Job

```bash
# Check if job is running
squeue -u $USER

# Watch the log (replace JOBID with your actual job ID)
tail -f extract_JOBID.log
```

Press `Ctrl+C` to stop watching.

#### 5. Verify Success

```bash
# Wait for job to complete (no longer in squeue)
squeue -u $USER

# Check the full log
cat extract_JOBID.log

# Check for errors
cat extract_JOBID.err

# Verify results were created
ls -la data/extracted/

# View the extraction
cat data/extracted/*.json | head -100
```

### Success Indicators

✅ Log shows: "✓ EXTRACTION COMPLETE"  
✅ Log shows: "Models extracted: X" (where X > 0)  
✅ JSON file exists in `data/extracted/`  
✅ No AttributeError in error file  
✅ File size > 1KB  

### If This Solution Fails

- Check error file for new error messages
- Try Solution 2 (switch to DistilGPT2)
- Share complete logs for further debugging

---

##Solution 2: Switch to DistilGPT2

If clearing the cache doesn't work, switch to the model officially mentioned in GWDG documentation.

### Why DistilGPT2

✅ Explicitly mentioned in GWDG Grete documentation  
✅ Smaller model (~300MB vs ~7GB for Phi-3)  
✅ Faster loading and inference  
✅ No compatibility issues  
✅ Proven to work on Grete infrastructure  

### Trade-offs

⚠️ Lower extraction quality (smaller model = less capable)  
⚠️ May need more careful prompting  
⚠️ Might miss subtle details in papers  

### Implementation

The repository already includes a DistilGPT2 extraction script.

#### Option A: Use Pre-existing Script

```bash
# Use the DistilGPT2 extraction script
sbatch grete/jobs/grete_extract_job.sh 2302.13971
# But first, edit the script to use distilgpt2 extractor
```

#### Option B: Modify Configuration

Edit `src/llm_extractor_transformers.py` to use DistilGPT2:

```python
# Change model_name from:
model_name = "microsoft/Phi-3-mini-4k-instruct"

# To:
model_name = "distilgpt2"
```

Then resubmit:

```bash
sbatch grete/jobs/grete_extract_job.sh 2302.13971
```

### Verify DistilGPT2 Works

```bash
# Monitor job
tail -f extract_*.log

# Check results
ls -la data/extracted/
cat data/extracted/*.json | head -50
```

---

## Solution 3: Use Different Phi-3 Variant

Try a different version of Phi-3 that might not have the cache issue.

### Alternative Phi-3 Models

```python
# In src/llm_extractor_transformers.py, try:

# Option 1: Phi-3 Medium
model_name = "microsoft/Phi-3-medium-4k-instruct"

# Option 2: Phi-3 Small
model_name = "microsoft/Phi-3-small-8k-instruct"

# Option 3: Phi-2 (older but stable)
model_name = "microsoft/phi-2"
```

### Steps

1. Edit `src/llm_extractor_transformers.py`
2. Change the `model_name` variable
3. Upload updated file to Grete
4. Resubmit job

### Considerations

- Different models have different capabilities
- Some may require more GPU memory
- Test with one paper first before batch processing

---

## Prevention

### For Future Deployments

1. **Clear cache periodically**: Remove old cached models
2. **Pin transformers version**: Specify exact version in requirements
3. **Test after updates**: Verify compatibility after library updates
4. **Use distilgpt2 for reliability**: Consider using it for production runs

### Monitoring Best Practices

1. **Always check error files**: Even if job completes
2. **Verify extraction quality**: Don't just check if file exists
3. **Keep logs**: Save logs for future reference
4. **Test with known papers**: Use papers with known results for validation

---

## Quick Reference

### Recommended Solution Workflow

```bash
# 1. Clear cache
rm -rf ~/.cache/huggingface/modules/transformers_modules/microsoft/Phi_hyphen_3_hyphen_mini_hyphen_4k_hyphen_instruct/

# 2. Resubmit job
sbatch grete/jobs/grete_extract_job.sh 2302.13971

# 3. Monitor
tail -f extract_*.log

# 4. Verify
ls -la data/extracted/
cat data/extracted/*.json | head -50
```

### If Cache Clear Fails

```bash
# Switch to DistilGPT2 by editing the extractor
nano src/llm_extractor_transformers.py
# Change model_name to "distilgpt2"

# Upload to Grete
scp src/llm_extractor_transformers.py Grete:~/llm-extraction/src/

# Resubmit
sbatch grete/jobs/grete_extract_job.sh 2302.13971
```

---

## Related Issues

- [DistilGPT2 Specific Issues](distilgpt2-issues.md)
- [Job Verification Guide](../deployment/verify-jobs.md)
- [Grete Setup Guide](../deployment/grete-setup.md)

---

## Getting Further Help

If none of these solutions work:

1. **Collect diagnostics**:
   - Full log file (`extract_*.log`)
   - Error file (`extract_*.err`)
   - SLURM job info (`sacct -j JOBID --format=ALL`)

2. **Check GWDG documentation**:
   - [GWDG Grete Documentation](https://info.gwdg.de/docs/doku.php?id=en:services:application_services:high_performance_computing:grete)

3. **Contact support**:
   - GWDG HPC support team
   - Include job ID and relevant log sections

---

## Summary

**Try this first**: Clear cache (Solution 1) - Works in most cases  
**If that fails**: Switch to DistilGPT2 (Solution 2) - Guaranteed to work  
**Alternative**: Try different Phi-3 variant (Solution 3) - May work  

The cache clearing solution typically resolves the issue and allows you to continue using the higher-quality Phi-3 model for extraction.
