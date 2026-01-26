# Job Verification Guide

This guide helps you verify that your extraction jobs on Grete are running correctly and producing expected results.

## 1. Monitor Job Status

Check if your job is running, pending, or completed.

### Check Specific Job

```bash
squeue -j [JOB_ID]
```

Replace `[JOB_ID]` with your actual job ID from the sbatch output.

### Check All Your Jobs

```bash
squeue -u $USER
```

### Status Codes

- `PD` - **Pending**: Waiting for GPU resources
- `R` - **Running**: Extraction in progress
- `CG` - **Completing**: Job finishing up
- `CD` - **Completed**: Job finished (won't show in squeue)

### Example Output

```
JOBID   PARTITION   NAME              USER    ST  TIME   NODES  NODELIST(REASON)
123456  kisski      extract_paper     u25486  R   0:15   1      gpu001
```

---

## 2. Monitor Live Logs

Watch the extraction progress in real-time while the job is running.

### Tail Most Recent Log

```bash
# Automatically finds and follows the newest log file
tail -f $(ls -t extract_*.log | head -n1)
```

### Tail Specific Job Log

```bash
tail -f extract_[JOB_ID].log
```

### Stop Watching

Press `Ctrl+C` to stop tailing the log.

### What to Look For

**Good signs:**
```
✓ Paper fetched successfully
✓ PDF parsed successfully
✓ Extracting models from text...
✓ Extraction complete
Models extracted: 3
```

**Warning signs:**
```
ERROR: Failed to parse PDF
WARNING: No models found
AttributeError: ...
```

---

## 3. Verify Output After Completion

Once the job is no longer shown in `squeue`, check the results.

### List Recent Extractions

```bash
# Show most recent files with timestamps
ls -lt data/extracted/ | head

# Count total extractions
ls data/extracted/*.json | wc -l
```

### View Latest Extraction

```bash
# Display the newest extraction file
cat $(ls -t data/extracted/*.json | head -n1)

# View just the beginning
cat $(ls -t data/extracted/*.json | head -n1) | head -100
```

### Check File Size

```bash
# Files should typically be >1KB
ls -lh data/extracted/*.json | tail -5
```

Empty or tiny files (<100 bytes) indicate extraction failures.

---

## 4. Check Specific Fields

Verify that important fields were extracted correctly.

### Search for Specific Field

```bash
# Get latest extraction file
LATEST=$(ls -t data/extracted/*.json | head -n1)

# Check if specific fields exist
grep "model_name" $LATEST
grep "parameters" $LATEST
grep "architecture" $LATEST
grep "training_data" $LATEST
```

### Pretty-Print JSON

If you have `jq` installed:

```bash
cat $(ls -t data/extracted/*.json | head -n1) | jq '.'
```

### Check Extraction Status

```bash
# Look for status field
grep -A 5 '"status"' $(ls -t data/extracted/*.json | head -n1)
```

---

## 5. Check for Errors

Review error logs for any issues during execution.

### View Error Log

```bash
# Check most recent error file
cat $(ls -t extract_*.err | head -n1)
```

### Common Errors to Look For

**CUDA/GPU errors:**
```bash
grep -i "cuda\|gpu\|out of memory" extract_*.err
```

**Python errors:**
```bash
grep -i "error\|traceback\|exception" extract_*.err
```

**Model loading issues:**
```bash
grep -i "transformers\|model" extract_*.err
```

---

## 6. Verify Extraction Quality

Check that the extracted data is meaningful and complete.

### Quick Quality Check

```bash
LATEST=$(ls -t data/extracted/*.json | head -n1)

echo "=== Extraction Quality Check ==="
echo "File: $LATEST"
echo ""
echo "Models found:"
grep -c '"model_name"' $LATEST || echo "0"
echo ""
echo "Model names:"
grep '"model_name"' $LATEST
echo ""
echo "Has parameters:"
grep -c '"parameters"' $LATEST || echo "0"
```

### Expected Results

- **Single model papers**: 1 model extracted
- **Multi-model papers** (like Llama 2): 2-4 models extracted
- **Field completeness**: Most fields should have values (not null/empty)

---

## 7. Compare with Expected Output

For known papers, compare with expected results.

### Llama 2 Paper (2307.09288)

Should extract:
- Llama 2 7B
- Llama 2 13B
- Llama 2 70B
- Llama 2 7B-Chat
- Llama 2 13B-Chat
- Llama 2 70B-Chat

```bash
grep '"model_name"' data/extracted/2307.09288_*.json
```

### GPT-1 Paper (Expected 1 model)

```bash
grep '"model_name"' data/extracted/*gpt-1*.json
```

---

## 8. Job Performance Metrics

### Check Job Completion Time

```bash
# View SLURM accounting info
sacct -j [JOB_ID] --format=JobID,JobName,Elapsed,State,ExitCode
```

### Expected Durations

- **Short papers** (~10 pages): 5-15 minutes
- **Medium papers** (~20 pages): 15-30 minutes
- **Long papers** (~50+ pages): 30-60 minutes

Much longer times may indicate issues.

---

## 9. Batch Job Verification

For job arrays, check each sub-job.

### View Array Job Status

```bash
squeue -j [ARRAY_JOB_ID]
```

### Check Individual Array Tasks

```bash
# View all tasks in array
sacct -j [ARRAY_JOB_ID] --format=JobID,JobName,State,ExitCode

# Count successful completions
sacct -j [ARRAY_JOB_ID] | grep -c "COMPLETED"
```

### Verify All Papers Processed

```bash
# Count expected vs actual extractions
echo "Papers in batch: 5"
echo "Extractions completed: $(ls data/extracted/*.json | wc -l)"
```

---

## 10. Troubleshooting Checklist

If verification fails, check:

- [ ] Job completed (not canceled or failed)
- [ ] No errors in `extract_*.err` file
- [ ] Output file exists in `data/extracted/`
- [ ] Output file is not empty (>1KB)
- [ ] JSON is valid (no parsing errors)
- [ ] Model names extracted
- [ ] Key fields populated (parameters, architecture, etc.)

### Get Help

If issues persist:

1. **Check logs**: Review full log and error files
2. **Check resources**: Verify GPU/memory usage
3. **Try different paper**: Test with known working paper
4. **Consult troubleshooting**: See [troubleshooting docs](../troubleshooting/)

---

## Quick Reference

```bash
# Monitor job
squeue -u $USER
tail -f extract_*.log

# Check results
ls -lt data/extracted/ | head
cat $(ls -t data/extracted/*.json | head -n1) | head -50

# Check errors
cat $(ls -t extract_*.err | head -n1)

# Verify extraction
grep -c '"model_name"' $(ls -t data/extracted/*.json | head -n1)
```

---

## Success Indicators

✅ Job status shows `R` (running) then disappears from queue  
✅ Log file shows "✓ EXTRACTION COMPLETE"  
✅ JSON file created in `data/extracted/`  
✅ File size >1KB  
✅ At least one model extracted  
✅ No errors in `.err` file  
✅ Key fields populated (model_name, parameters, etc.)  

If all indicators are green, your extraction was successful!
