# Migration from Gemini API to KISSKI API

## Summary

Your LLM extraction pipeline has been successfully migrated from Google Gemini API to KISSKI Chat AI API (SAIA Platform).

---

## What Changed

### Files Modified (6)

1. **`src/llm_extractor.py`** - Completely rewritten to use KISSKI API
   - Removed: `google.generativeai` library
   - Added: `openai` library (OpenAI-compatible)
   - Implemented: Rate limiting (2s between requests)
   - API endpoint: `https://chat-ai.academiccloud.de/v1`

2. **`src/pipeline.py`** - Updated initialization
   - Changed: `GEMINI_API_KEY` → `KISSKI_API_KEY`
   - Changed: API endpoint configuration
   - Added: Rate limit parameter

3. **`config/config.yaml`** - Updated configuration
   - Removed: `gemini:` section
   - Added: `kisski:` section with proper settings
   - Default model: `openai-gpt-oss-120b`

4. **`requirements.txt`** - Updated dependencies
   - Removed: `google-generativeai>=0.3.0`
   - Added: `openai>=1.0.0`

5. **`.env.example`** - Updated template
   - Removed: `GEMINI_API_KEY`
   - Added: `KISSKI_API_KEY`, `KISSKI_BASE_URL`

6. **`README.md`** - Updated documentation
   - All Gemini references replaced with KISSKI
   - Updated setup instructions
   - Updated acknowledgments

### Files Created (3)

1. **`test_kisski_api.py`** - Comprehensive test script
2. **`KISSKI_SETUP.md`** - Detailed setup guide
3. **`MIGRATION_TO_KISSKI.md`** - This file

### Files Removed

- All Gemini API related code and dependencies

---

## Why This Migration?

### Benefits of KISSKI API

✅ **Free** - No API costs for your thesis  
✅ **Local** - Hosted by GWDG (German academic infrastructure)  
✅ **Private** - Data never stored (GDPR-compliant)  
✅ **Multiple models** - 20+ models vs 1 with Gemini  
✅ **Higher limits** - 10000/hour vs Gemini's limits  
✅ **Standard** - Uses OpenAI library (industry standard)  

### Previous Issues with Gemini

- Cost per API request
- Google data privacy concerns
- Limited to Gemini models only
- External dependency

---

## How to Use

### 1. Setup (One-time)

```bash
# Step 1: Add API key to .env
echo "KISSKI_API_KEY=8810b4c60127bfed5655b1e66f3d291a" >> .env
echo "KISSKI_BASE_URL=https://chat-ai.academiccloud.de/v1" >> .env

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Test connection
python test_kisski_api.py
```

### 2. Extract Papers (Same as Before!)

```bash
# Extract single paper
python -m src.pipeline --arxiv-id 2307.09288

# Results saved to data/extracted/
```

### 3. Monitor Usage

Check logs for rate limiting:
```bash
tail -f data/logs/pipeline.log
```

---

## API Configuration

### Default Configuration

Located in `config/config.yaml`:

```yaml
kisski:
  base_url: https://chat-ai.academiccloud.de/v1
  model: openai-gpt-oss-120b
  temperature: 0.0
  max_tokens: 4000
  timeout: 60
  rate_limit_delay: 2.0
```

### Recommended Models

**For best extraction quality**:
```yaml
model: openai-gpt-oss-120b  # 120B, best performance
```

**For fastest processing**:
```yaml
model: meta-llama-3.1-8b-instruct  # 8B, very fast
```

**For reasoning tasks**:
```yaml
model: qwen3-32b  # 32B, excellent reasoning
```

Complete model list: See [`KISSKI_SETUP.md`](KISSKI_SETUP.md#available-models)

---

## Rate Limiting

### Why It Matters

Your professor specifically requested:
> "Please do not overload the servers with your queries; make sure to implement 
> an appropriate timer or rate-limiting mechanism when running them."

### Implementation

**Client-side delay**: 2 seconds between requests (configurable)

**Server-side limits**:
- 1,000 requests per minute
- 10,000 requests per hour
- 50,000 requests per day

### Adjusting Rate Limit

Edit `config/config.yaml`:
```yaml
kisski:
  rate_limit_delay: 3.0  # Increase to 3 seconds if needed
```

---

## Technical Details

### API Compatibility

KISSKI API is **OpenAI-compatible**, which means:
- Uses standard OpenAI Python library
- Same request/response format as OpenAI
- Compatible with OpenAI tools and frameworks
- Easy to understand and debug

### Request Flow

```
Your Code → OpenAI Library → KISSKI API → LLM Model → Response → Your Code
```

Detailed:
1. Your code calls `extractor.extract(paper_text)`
2. Extractor enforces 2-second rate limit
3. OpenAI library sends HTTP POST to KISSKI
4. KISSKI routes to selected model (e.g., GPT OSS 120B)
5. Model generates structured JSON response
6. Response parsed and validated
7. Returned as `MultiModelResponse` object

### Error Handling

The extractor handles:
- Connection errors (timeout, network issues)
- Rate limit errors (HTTP 429)
- Invalid responses (malformed JSON)
- Empty responses
- Validation errors

All errors are logged to `data/logs/pipeline.log`.

---

## Verification

### Check Integration

Run all checks:

```bash
# 1. Test API connection
python test_kisski_api.py

# 2. Verify imports work
python -c "from src.pipeline import ExtractionPipeline; print('✓ Imports OK')"

# 3. Extract sample paper
python -m src.pipeline --arxiv-id 2307.09288

# 4. Check results
ls -la data/extracted/
```

### Expected Results

- ✓ Test script passes all tests
- ✓ No import errors
- ✓ Extraction completes successfully
- ✓ JSON files created in `data/extracted/`
- ✓ Rate limiting visible in logs

---

## Rollback (If Needed)

If you need to revert to Gemini API:

```bash
# Checkout previous version
git checkout HEAD~1 src/llm_extractor.py src/pipeline.py config/config.yaml

# Reinstall Gemini dependency
pip install google-generativeai

# Update .env with Gemini key
```

However, **this is not recommended** since KISSKI is free and better for your thesis.

---

## Next Steps After Migration

1. ✅ Test KISSKI API works
2. ✅ Extract 5-10 papers for validation
3. ✅ Compare quality with previous extractions
4. 🔲 Create gold standard dataset (professor's task #2)
5. 🔲 Implement evaluation metrics (professor's task #3)
6. 🔲 Run systematic evaluation

---

## Key Points to Remember

1. **API Key Security**
   - Never share your API key
   - Never commit .env to GitHub
   - Key expires after thesis completion

2. **Rate Limiting**
   - Always keep 2-second delay minimum
   - Monitor your quota usage
   - Don't run massive batches without breaks

3. **Model Selection**
   - Default (openai-gpt-oss-120b) is recommended
   - Try different models for comparison
   - Adjust based on extraction quality

4. **Monitoring**
   - Check logs regularly
   - Verify extraction quality
   - Watch for rate limit warnings

---

## Support

If you encounter issues:

1. **Read documentation**: [`KISSKI_SETUP.md`](KISSKI_SETUP.md)
2. **Check logs**: `data/logs/pipeline.log`
3. **Run test**: `python test_kisski_api.py`
4. **Contact**: Your professor or GWDG support

---

## Migration Complete!

Your pipeline now uses KISSKI API instead of Gemini API.

**Benefits**:
- Free API access
- Multiple model options
- Better data privacy
- Higher rate limits
- Academic infrastructure

All functionality preserved - just better and free!
