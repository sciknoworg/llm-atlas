# KISSKI API Setup Guide (SAIA Platform)

## Overview

KISSKI Chat AI API is part of the **SAIA** (Scalable Artificial Intelligence Accelerator) platform hosted by GWDG Academic Cloud. It provides access to various state-of-the-art LLMs through an OpenAI-compatible API.

### What is SAIA/KISSKI?

- **SAIA**: Scalable AI Accelerator platform by GWDG
- **Chat AI**: The LLM inference service  
- **API**: OpenAI-compatible HTTP API for programmatic access
- **Location**: Hosted by GWDG (German academic infrastructure)
- **Data Privacy**: Highest standards - prompts never stored

### Why Use KISSKI API?

✅ **Free for thesis work** - No API costs  
✅ **OpenAI-compatible** - Uses standard OpenAI Python library  
✅ **Multiple models** - GPT OSS, Llama, Qwen, DeepSeek, etc.  
✅ **High rate limits** - 1000/min, 10000/hour, 50000/day  
✅ **Data privacy** - GDPR-compliant, data never stored  
✅ **Academic infrastructure** - Reliable and maintained  

---

## Quick Setup

### Step 1: Add API Key to .env

Create or edit your `.env` file:

```env
# KISSKI Chat AI API (SAIA Platform)
KISSKI_API_KEY=8810b4c60127bfed5655b1e66f3d291a
KISSKI_BASE_URL=https://chat-ai.academiccloud.de/v1
```

**Security reminders**:
- This key is for your thesis work only
- DO NOT share with others
- DO NOT commit .env to GitHub (already ignored)
- Key will be deleted after thesis completion

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

New dependency: `openai>=1.0.0` (OpenAI Python library for KISSKI)

### Step 3: Test the Integration

```bash
python test_kisski_api.py
```

Expected output:
- ✓ Connection test passes
- ✓ Lists available models
- ✓ Extracts models from sample text
- ✓ Validates extraction format

---

## Available Models

Based on KISSKI documentation, recommended models for extraction:

### Best Performance (Recommended)

**openai-gpt-oss-120b**
- 120B parameters
- Great overall performance
- Fast inference
- Knowledge cutoff: June 2024

Usage in `config/config.yaml`:
```yaml
kisski:
  model: openai-gpt-oss-120b
```

### Fast Performance

**meta-llama-3.1-8b-instruct**
- 8B parameters
- Fastest inference
- Good general performance
- Knowledge cutoff: Dec 2023

### Reasoning Tasks

**qwen3-32b**
- 32B parameters
- Excellent reasoning
- Multilingual support
- Knowledge cutoff: Sep 2024

**deepseek-r1-0528**
- Advanced reasoning model
- Great for problem-solving
- Knowledge cutoff: Dec 2023
- Note: Slower but very capable

### Complete Model List

See full list at: https://doc.gwdg.de/doku.php?id=en:services:application_services:chat_ai:available_models

---

## API Rate Limits

KISSKI API has the following limits:

- **1,000 requests per minute**
- **10,000 requests per hour**
- **50,000 requests per day**

### Client-Side Rate Limiting

The extractor implements **2-second delays** between requests (configurable).

This ensures:
- No server overload (as requested by professor)
- Predictable processing time
- Respectful API usage

Adjust in `config/config.yaml`:
```yaml
kisski:
  rate_limit_delay: 2.0  # Seconds between requests
```

### Check Your Quota

```bash
curl -i -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://chat-ai.academiccloud.de/v1/chat/completions
```

Headers show:
- `x-ratelimit-remaining-minute`: Requests left this minute
- `x-ratelimit-remaining-hour`: Requests left this hour
- `x-ratelimit-remaining-day`: Requests left today

---

## Configuration

### Model Selection

Edit `config/config.yaml`:

```yaml
kisski:
  base_url: https://chat-ai.academiccloud.de/v1
  model: openai-gpt-oss-120b  # Change to any available model
  temperature: 0.0             # 0.0 = deterministic
  max_tokens: 4000             # Response length
  timeout: 60                  # Request timeout (seconds)
  rate_limit_delay: 2.0        # Inter-request delay (seconds)
```

### Model Recommendations by Task

**General extraction**: `openai-gpt-oss-120b`  
**Fast processing**: `meta-llama-3.1-8b-instruct`  
**Reasoning heavy**: `qwen3-32b` or `deepseek-r1-0528`  
**Multilingual**: `qwen3-32b` or `llama-3.1-sauerkrautlm-70b-instruct`  

---

## Usage

### Same Pipeline, No Code Changes!

```python
from src.pipeline import ExtractionPipeline

# Initialize pipeline (automatically uses KISSKI API)
pipeline = ExtractionPipeline()

# Extract a paper
result = pipeline.process_paper("2307.09288")

# Results saved to data/extracted/
```

### Command Line

```bash
# Extract single paper
python -m src.pipeline --arxiv-id 2307.09288

# Extract from existing JSON
python -m src.pipeline --json-file data/extracted/my_file.json

# Test connections
python -m src.pipeline --test
```

---

## Troubleshooting

### Connection Issues

**Problem**: Test fails with connection error

**Solutions**:
1. Verify API key is correct in `.env`
2. Check base URL: `https://chat-ai.academiccloud.de/v1`
3. Test with curl:
   ```bash
   curl -X POST https://chat-ai.academiccloud.de/v1/chat/completions \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"meta-llama-3.1-8b-instruct","messages":[{"role":"user","content":"Hello"}]}'
   ```

### Rate Limit Errors

**Problem**: HTTP 429 (Too Many Requests)

**Solutions**:
1. Increase `rate_limit_delay` to 3.0 or 5.0 seconds
2. Check your quota (see "Check Your Quota" above)
3. Wait until quota resets (check ratelimit-reset header)

### Extraction Quality Issues

**Problem**: Poor extraction quality or missing fields

**Solutions**:
1. Try a better model: `openai-gpt-oss-120b` instead of `meta-llama-3.1-8b-instruct`
2. Increase `max_tokens` to 6000 or 8000
3. Adjust temperature (try 0.1 or 0.2)
4. Check if paper text is being truncated

### JSON Parsing Errors

**Problem**: Model returns text instead of valid JSON

**Solutions**:
1. Update system prompt (already configured in extractor)
2. Try different model (GPT OSS is most reliable for JSON)
3. Increase max_tokens if response is cut off

---

## Differences from Gemini API

| Feature | Gemini API (Old) | KISSKI API (New) |
|---------|------------------|------------------|
| **Cost** | Paid per request | Free (thesis) |
| **Provider** | Google Cloud | GWDG Academic Cloud |
| **Library** | google-generativeai | openai (standard) |
| **Models** | Gemini only | 20+ models (GPT, Llama, Qwen, etc.) |
| **Data privacy** | Google's terms | GWDG (GDPR, data never stored) |
| **Rate limits** | Google's limits | 1000/min, 10000/hour |
| **Setup** | API key from Google | API key from professor |

---

## API Technical Details

### Endpoint

```
Base URL: https://chat-ai.academiccloud.de/v1
Endpoint: /chat/completions
```

### Authentication

```
Header: Authorization: Bearer <api_key>
```

### Request Format (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    api_key="your_api_key",
    base_url="https://chat-ai.academiccloud.de/v1"
)

response = client.chat.completions.create(
    model="openai-gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Extract information from this paper..."}
    ],
    temperature=0.0,
    max_tokens=4000
)

print(response.choices[0].message.content)
```

### Streaming Support

For long responses, streaming is supported:

```python
stream = client.chat.completions.create(
    model="openai-gpt-oss-120b",
    messages=[...],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## Best Practices

### 1. Rate Limiting

- Keep 2-second delay minimum
- Monitor quota usage
- Process in batches if needed

### 2. Model Selection

- Start with `openai-gpt-oss-120b` (best quality)
- Use `meta-llama-3.1-8b-instruct` for speed
- Try reasoning models for complex papers

### 3. Error Handling

- Check API quota before large batches
- Implement retries for transient errors
- Log all API interactions

### 4. Security

- Never commit API key to GitHub
- Rotate key if accidentally exposed
- Use .env file (already gitignored)

---

## Comparison: KISSKI vs Grete HPC

| Feature | KISSKI API | Grete HPC |
|---------|------------|-----------|
| **Setup** | Simple (API key) | Complex (SSH, conda, SLURM) |
| **Speed** | Fast (immediate) | Depends on queue |
| **Cost** | Free (thesis) | Free (academic) |
| **Models** | 20+ predefined | Any HuggingFace model |
| **Control** | Limited | Full control |
| **Best for** | Development, testing | Large batches, experiments |

### When to Use Each

**Use KISSKI API for**:
- Daily development work
- Quick extractions
- Testing and iteration
- Gold standard evaluation

**Use Grete HPC for**:
- Processing 50+ papers
- Custom model experiments
- When KISSKI unavailable
- Specific model requirements

---

## Next Steps

1. ✅ Test API: `python test_kisski_api.py`
2. ✅ Extract a paper: `python -m src.pipeline --arxiv-id 2307.09288`
3. ✅ Verify results: Check `data/extracted/`
4. ✅ Monitor logs: Check `data/logs/pipeline.log`
5. ✅ Adjust model: Try different models in `config/config.yaml`

---

## Support

### Documentation

- **SAIA Platform**: https://doc.gwdg.de/doku.php?id=en:services:application_services:saia
- **Chat AI**: https://doc.gwdg.de/doku.php?id=en:services:application_services:chat_ai
- **Available Models**: https://doc.gwdg.de/doku.php?id=en:services:application_services:chat_ai:available_models

### Contact

- **GWDG Support**: support@gwdg.de
- **Your Professor**: For API key issues

---

## Summary

- ✅ KISSKI API is OpenAI-compatible (easy to use)
- ✅ Uses standard OpenAI Python library
- ✅ Free for your thesis work
- ✅ Rate limiting implemented (2s delay)
- ✅ Multiple high-quality models available
- ✅ GWDG infrastructure (reliable, GDPR-compliant)

Your pipeline is now configured to use KISSKI API for LLM extraction!
