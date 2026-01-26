# Troubleshooting: DistilGPT2 JSON Extraction Failure

## Problem Overview

### Symptoms

- Model loads successfully on GPU
- No CUDA or memory errors
- Model generates output
- But extraction fails with: `JSON parse error: Expecting value: line 1 column 1 (char 0)`
- Output is plain text instead of structured JSON

### What's Working

✅ Model loads without errors  
✅ GPU detected and accessible  
✅ Model generates text (no inference failure)  
✅ Paper fetching and PDF parsing work  

### What's Failing

❌ JSON parsing fails  
❌ Model output is not valid JSON  
❌ No structured extraction results  

---

## Root Cause

**DistilGPT2 is a base language model, NOT an instruction-tuned model.**

### Technical Explanation

1. **Base models** (like DistilGPT2) are trained to continue text, not follow instructions
2. **Instruction-tuned models** (like Phi-3-Instruct, GPT-3.5-Instruct) are trained to follow prompts
3. When you ask DistilGPT2 to "generate JSON with model information", it doesn't understand the instruction
4. Instead, it generates text continuation that looks like natural language
5. The JSON parser then fails because the output isn't valid JSON

### Why This Wasn't Obvious

- DistilGPT2 was mentioned in GWDG documentation as an **example** of running transformers on GPU
- It was NOT recommended specifically for structured extraction tasks
- The documentation showed how to **load and run** models, not how to choose the right model for your task

---

## Solution 1: Use Instruction-Tuned Model (⭐ Recommended)

Switch to a model that's been trained to follow instructions and generate structured output.

### Recommended Models

#### Option A: Phi-3-mini-4k-instruct (Best Quality)

```python
model_name = "microsoft/Phi-3-mini-4k-instruct"
```

- **Size**: 3.8B parameters
- **Pros**: Excellent instruction following, good extraction quality
- **Cons**: Had cache issues (see [phi3-cache-issues.md](phi3-cache-issues.md))
- **Memory**: ~8GB GPU RAM

**Note**: Clear cache before using (see Phi-3 troubleshooting guide)

#### Option B: Qwen2.5-0.5B-Instruct (Fastest)

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
```

- **Size**: 0.5B parameters  
- **Pros**: Very fast, small, instruction-tuned, no known cache issues  
- **Cons**: Lower quality than Phi-3 (smaller model)  
- **Memory**: ~2GB GPU RAM  

**Best for**: Quick testing, batch processing many papers

#### Option C: TinyLlama-1.1B-Chat (Balanced)

```python
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

- **Size**: 1.1B parameters
- **Pros**: Chat-tuned, good balance of speed and quality
- **Cons**: Middle ground (not as fast or as good as alternatives)
- **Memory**: ~3GB GPU RAM

### Implementation

1. **Edit the extraction script**:

```python
# In src/llm_extractor_transformers.py
# Change from:
model_name = "distilgpt2"

# To one of:
model_name = "microsoft/Phi-3-mini-4k-instruct"  # Best quality
model_name = "Qwen/Qwen2.5-0.5B-Instruct"        # Fastest
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Balanced
```

2. **Upload to Grete**:

```powershell
# From local machine
scp src/llm_extractor_transformers.py Grete:~/llm-extraction/src/
```

3. **Test with one paper**:

```bash
# On Grete
sbatch grete/jobs/grete_extract_job.sh 2302.13971
```

4. **Verify extraction**:

```bash
# Monitor
tail -f extract_*.log

# Check results
cat data/extracted/*.json | head -100
```

---

## Solution 2: Fix Phi-3 Cache Issues

If you want to use Phi-3 (recommended for quality), you need to clear the cache first.

### Complete Steps

1. **Clear ALL Phi-3 cache**:

```bash
# On Grete
rm -rf ~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/
rm -rf ~/.cache/huggingface/modules/transformers_modules/microsoft*
```

2. **Modify extraction code** to force fresh download:

```python
# In src/llm_extractor_transformers.py
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
    trust_remote_code=True,
    attn_implementation="eager",
    force_download=True,      # Add this
    local_files_only=False    # Add this
)
```

3. **Upload and test**:

```bash
# Upload updated code
scp src/llm_extractor_transformers.py Grete:~/llm-extraction/src/

# Submit job
sbatch grete/jobs/grete_extract_job.sh 2302.13971
```

See [phi3-cache-issues.md](phi3-cache-issues.md) for detailed Phi-3 troubleshooting.

---

## Solution 3: Post-Process DistilGPT2 Output (Not Recommended)

**⚠️ This is NOT recommended** but included for completeness.

### Why This Doesn't Work Well

- DistilGPT2 doesn't understand what you're asking for
- Output is unpredictable and unstructured
- Would require complex regex/parsing logic
- Very low success rate
- Not reliable for thesis work

### If You Still Want to Try

1. **Capture raw output**:

```python
# In extraction code
raw_output = self.model.generate(...)
print(f"DEBUG: Raw model output: {raw_output}")
```

2. **Add post-processing**:

```python
import re

def extract_from_text(text):
    # Try to find model names
    model_pattern = r'(?:model|architecture):\s*([A-Za-z0-9-]+)'
    models = re.findall(model_pattern, text, re.IGNORECASE)
    
    # Try to find parameters
    param_pattern = r'(\d+\.?\d*)\s*([BMK])\s*parameters?'
    params = re.findall(param_pattern, text, re.IGNORECASE)
    
    # Build JSON manually
    return {
        "model_name": models[0] if models else "Unknown",
        "parameters": params[0] if params else None,
        # ... etc
    }
```

3. **This is fragile and unreliable** - use an instruction-tuned model instead!

---

## Solution 4: API Fallback (Development Only)

For local development/testing, you can use API-based extraction.

### Use Gemini API Locally

```python
# Use the API-based extractor instead
from src.llm_extractor import LLMExtractor  # Not transformers version

extractor = LLMExtractor()  # Uses Gemini API
```

**Pros**: Reliable, high quality  
**Cons**: Costs money, requires API key, defeats purpose of using Grete  

**When to use**: Local testing only, not for production on Grete

---

## Recommended Action Plan

### Quick Start (5 minutes)

1. **Switch to Qwen2.5-0.5B-Instruct** (fastest to test):

```bash
# Edit extractor locally
nano src/llm_extractor_transformers.py
# Change model_name to "Qwen/Qwen2.5-0.5B-Instruct"

# Upload to Grete
scp src/llm_extractor_transformers.py Grete:~/llm-extraction/src/

# Test
ssh Grete
cd ~/llm-extraction
sbatch grete/jobs/grete_extract_job.sh 2302.13971

# Monitor
tail -f extract_*.log
```

### For Best Quality (15 minutes)

1. **Clear Phi-3 cache**
2. **Switch to Phi-3-mini-4k-instruct**
3. **Force fresh download**
4. **Test thoroughly**

See [phi3-cache-issues.md](phi3-cache-issues.md) for complete Phi-3 setup.

---

## Model Comparison

| Model | Size | Speed | Quality | Cache Issues | Recommended For |
|-------|------|-------|---------|--------------|-----------------|
| **Qwen2.5-0.5B-Instruct** | 0.5B | ⚡⚡⚡ Fastest | ⭐⭐ Good | None | Quick testing, batch processing |
| **TinyLlama-1.1B-Chat** | 1.1B | ⚡⚡ Fast | ⭐⭐⭐ Better | None | Balanced use |
| **Phi-3-mini-4k-instruct** | 3.8B | ⚡ Moderate | ⭐⭐⭐⭐ Best | Yes (solvable) | Final thesis runs |
| **DistilGPT2** | 0.08B | ⚡⚡⚡ Fastest | ❌ Unusable | None | **Don't use for extraction** |

---

## Verification

After switching models, verify the extraction works:

```bash
# 1. Check model loaded
grep "Loading model" extract_*.log

# 2. Check JSON generated
grep "Successfully parsed JSON" extract_*.log

# 3. Check models extracted
grep "Models extracted:" extract_*.log

# 4. View results
cat data/extracted/*.json | head -100
```

### Success Indicators

✅ Log shows model loaded successfully  
✅ Log shows JSON parsing succeeded  
✅ At least one model extracted  
✅ Valid JSON in output file  
✅ Model fields populated (name, parameters, architecture, etc.)  

---

## Key Takeaways

1. **DistilGPT2 is NOT suitable** for structured extraction tasks
2. **Use instruction-tuned models** for JSON generation
3. **Qwen2.5-0.5B-Instruct** is fastest for testing
4. **Phi-3-mini-4k-instruct** is best for quality (after cache fix)
5. **Base models** can only continue text, not follow instructions

---

## Related Documentation

- [Phi-3 Cache Issues](phi3-cache-issues.md) - How to fix Phi-3 compatibility
- [Grete Setup Guide](../deployment/grete-setup.md) - Initial cluster setup
- [Job Verification](../deployment/verify-jobs.md) - Verify extraction results

---

## Getting Help

If switching models doesn't work:

1. **Check logs**: Look for model loading errors
2. **Verify GPU**: Ensure model fits in GPU memory
3. **Test locally**: Try the same model on your local machine first
4. **Check model card**: Read the model's documentation on HuggingFace

---

## Summary

**Problem**: DistilGPT2 is a base model that can't follow instructions  
**Solution**: Switch to instruction-tuned model like Qwen2.5-0.5B-Instruct or Phi-3-mini-4k-instruct  
**Quick fix**: Use Qwen2.5-0.5B-Instruct (works immediately, no cache issues)  
**Best quality**: Use Phi-3-mini-4k-instruct (after clearing cache)  

The key insight is that **model selection matters** - not all models can perform structured extraction, even if they run successfully on the GPU.
