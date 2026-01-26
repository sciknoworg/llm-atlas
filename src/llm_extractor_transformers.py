"""
LLM Extractor using Hugging Face Transformers

For use on GWDG Grete GPU cluster. Runs models locally on GPU without API limits.
"""

import logging
import json
import torch
import os
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llm_extractor import LLMProperties, MultiModelResponse

logger = logging.getLogger(__name__)

# #region agent log
# Debug log path for Grete (Linux)
DEBUG_LOG_PATH = os.path.expanduser("~/llm-extraction/debug.log")

def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]):
    """Write debug log entry."""
    import time
    try:
        # Ensure directory exists
        log_dir = os.path.dirname(DEBUG_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # Silently fail if logging fails
# #endregion


class LLMExtractorTransformers:
    """Extracts LLM information using Hugging Face transformers on GPU."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature: float = 0.3,
        max_new_tokens: int = 1500,
        device: str = "auto"
    ):
        """
        Initialize transformers-based extractor.
        
        Args:
            model_name: HuggingFace model name
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            device: Device to use ("auto", "cuda", or "cpu")
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set pad_token if not set (required for batching)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa"  # Use efficient SDPA attention to reduce memory usage
        ).to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Initialize JSON prefix for prompt priming (will be set in _create_extraction_prompt)
        self._json_prefix = None
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
    
    def _create_extraction_prompt(self, paper_text: str, paper_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create extraction prompt."""
        
        # Inject metadata if available
        meta_str = ""
        if paper_metadata:
            meta_str = f"PAPER METADATA:\nTitle: {paper_metadata.get('title', 'Unknown')}\nAuthored: {paper_metadata.get('year', '')}-{paper_metadata.get('month', '')}\nAuthors: {paper_metadata.get('authors', [])}\n"
        schema = MultiModelResponse.model_json_schema()
        
        # Instruction-tuned models (Llama 3.1, Qwen, etc.) use chat template format
        if "instruct" in self.model_name.lower() or "chat" in self.model_name.lower():
            # Try to use tokenizer's chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                # Use up to 65,000 chars
                paper_snippet = paper_text[:65000] if len(paper_text) > 65000 else paper_text
                
                # Prepend metadata to snippet
                if meta_str:
                    paper_snippet = meta_str + "\n\nPAPER CONTENT:\n" + paper_snippet
                
                # FEW-SHOT LEARNING WITH COMPREHENSIVE EXAMPLES
                # We provide full examples to teach the model to extract ALL fields (innovation, tasks, architecture, etc.)
                
                # Example 1: BERT
                example1_input = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Google AI Language. We introduce BERT with 110M, 340M parameters. It uses a Transformer encoder architecture trained on Masked LM and Next Sentence Prediction tasks. It achieves state-of-the-art on GLUE. We use Adam optimizer."
                example1_output = {
                    "models": [{
                        "model_name": "BERT",
                        "model_family": "BERT",
                        "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                        "organization": "Google",
                        "parameters": "340M",
                        "date_created": "2018-10",
                        "pretraining_architecture": "Encoder",
                        "pretraining_task": "Masked LM (MLM), Next Sentence Prediction (NSP)",
                        "optimizer": "Adam",
                        "innovation": "Bidirectional training of Transformer encoder",
                        "research_problem": "Language Understanding"
                    }]
                }
                
                # Example 2: GPT-2
                example2_input = "Language Models are Unsupervised Multitask Learners. OpenAI. We trained a 1.5 billion parameter Transformer decoder language model. It demonstrates zero-shot task transfer. We assume a causal language modeling objective."
                # Example 2: GPT-2
                example2_output = {
                    "models": [{
                        "model_name": "GPT-2", 
                        "model_family": "GPT",
                        "paper_title": "Language Models are Unsupervised Multitask Learners",
                        "organization": "OpenAI", 
                        "parameters": "1.5B",
                        "date_created": "2019-02",
                        "pretraining_architecture": "Decoder", 
                        "pretraining_task": "Causal language modeling",
                        "innovation": "Zero-shot task transfer via large-scale unsupervised learning",
                        "introduction_date": "2019"
                    }]
                }
                
                # Example 3: GPT-1 (Targeting the user's specific case)
                example3_input = "Improving Language Understanding by Generative Pre-Training. Alec Radford, OpenAI. We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. Our approach employs a Transformer-based architecture with 117M parameters. We use the Adam optimizer."
                # Example 3: GPT-1
                example3_output = {
                    "models": [{
                        "model_name": "GPT-1",
                        "model_family": "GPT", 
                        "paper_title": "Improving Language Understanding by Generative Pre-Training",
                        "organization": "OpenAI",
                        "parameters": "117M",
                        "date_created": "2018-06",
                        "pretraining_architecture": "Decoder",
                        "pretraining_task": "Causal language modeling",
                        "finetuning_task": "Supervised discriminative fine-tuning",
                        "optimizer": "Adam",
                        "innovation": "Generative pre-training followed by discriminative fine-tuning",
                        "license": "closed source",
                        "research_problem": "Language Understanding"
                    }]
                }
                
                # Example 4: Multiple model versions (Llama 3.1 with different sizes)
                example4_input = "The Llama 3.1 Herd of Models. Meta AI. We introduce Llama 3.1 with three model sizes: 8B, 70B, and 405B parameters. All models use Transformer decoder architecture. The 8B model has 8 billion parameters, the 70B model has 70 billion parameters, and the 405B model has 405 billion parameters. All models are trained on the same pretraining task."
                example4_output = {
                    "models": [
                        {
                            "model_name": "Llama 3.1 8B",
                            "model_family": "Llama",
                            "paper_title": "The Llama 3.1 Herd of Models",
                            "organization": "Meta",
                            "parameters": "8B",
                            "parameters_millions": 8000,
                            "date_created": "2024-07",
                            "pretraining_architecture": "Transformer",
                            "pretraining_task": "Next-token prediction",
                            "innovation": "Large-scale language models"
                        },
                        {
                            "model_name": "Llama 3.1 70B",
                            "model_family": "Llama",
                            "paper_title": "The Llama 3.1 Herd of Models",
                            "organization": "Meta",
                            "parameters": "70B",
                            "parameters_millions": 70000,
                            "date_created": "2024-07",
                            "pretraining_architecture": "Transformer",
                            "pretraining_task": "Next-token prediction",
                            "innovation": "Large-scale language models"
                        },
                        {
                            "model_name": "Llama 3.1 405B",
                            "model_family": "Llama",
                            "paper_title": "The Llama 3.1 Herd of Models",
                            "organization": "Meta",
                            "parameters": "405B",
                            "parameters_millions": 405000,
                            "date_created": "2024-07",
                            "pretraining_architecture": "Transformer",
                            "pretraining_task": "Next-token prediction",
                            "innovation": "Large-scale language models"
                        }
                    ],
                    "paper_describes_multiple_models": True
                }
                
                messages = [
                    {
                        "role": "system", 
                        "content": "You are an expert AI researcher. Extract DETAILED information about ALL MODEL VERSIONS/VARIANTS introduced in the paper. CRITICAL RULES:\n1. TITLE: Extract the official, full RESEARCH PAPER TITLE and assign it to 'paper_title'.\n2. ALL VARIANTS: Extract ALL model versions, sizes, and variants as SEPARATE entries. If a paper describes multiple model sizes (e.g., 8B, 70B, 405B), versions (e.g., 3.1, 3.2, 3.3), or variants (e.g., Base, Large, XL), create a separate entry for EACH one.\n3. PARAMETERS: Search for 'Our model' or 'Proposed'. Look for 'M' or 'B'. Extract parameter sizes for each variant.\n4. DATES: Priority 1: Metadata. Priority 2: Header/footer dates. Priority 3: Latest citation year.\n5. MULTIPLE MODELS: Set 'paper_describes_multiple_models' to true if the paper describes multiple distinct models, versions, or size variants.\nFields: model_name, model_family, paper_title, organization, parameters, date_created, pretraining_architecture, pretraining_task, finetuning_task, optimizer, innovation, license, hardware_used.\nReturn JSON only."
                    },
                    {
                        "role": "user",
                        "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example1_input}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(example1_output)
                    },
                    {
                        "role": "user", 
                        "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example2_input}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(example2_output)
                    },
                    {
                        "role": "user",
                        "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example3_input}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(example3_output)
                    },
                    {
                        "role": "user",
                        "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example4_input}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(example4_output)
                    },
                    {
                        "role": "user",
                        "content": f"""Extract ALL model versions, variants, and sizes introduced in this paper:

{paper_snippet}

CRITICAL INSTRUCTIONS:
- Extract ALL model versions/variants described in the paper (e.g., if paper mentions "Llama 3.1 8B", "Llama 3.1 70B", "Llama 3.1 405B", create 3 separate entries)
- Extract ALL model sizes mentioned (different parameter counts = different entries)
- Extract ALL model versions mentioned (3.1, 3.2, 3.3 = separate entries)
- Extract ALL architectural variants (Base, Large, XL, etc. = separate entries)
- Each distinct model size/version/variant should be a SEPARATE entry in the models array
- Extract models THIS paper introduces (the main contributions)
- NOT models mentioned as related work or comparisons
- The model name should include version/size if mentioned (e.g., "Llama 3.1 8B" not just "Llama")
- Model name is NOT the architecture (e.g., "GPT" not "Transformer")
- If the paper describes multiple models, set "paper_describes_multiple_models": true

Output JSON:"""
                    }
                ]
                
                # Don't use prefix priming - let few-shot examples guide
                self._json_prefix = None
                
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback to generic chat template (no prefix priming)
                self._json_prefix = None
                paper_snippet = paper_text[:8000]
                example_json = {
                    "models": [
                        {
                            "model_name": "ModelName",
                            "model_family": "FamilyName",
                            "organization": "OrganizationName",
                            "date_created": "YYYY-MM-DD",
                            "parameters": "ParameterSize",
                            "parameters_millions": 0,
                            "pretraining_architecture": "ArchitectureType",
                            "pretraining_task": "Pre-training task",
                            "finetuning_task": "Fine-tuning task",
                            "optimizer": "Optimizer name",
                            "license": "License type",
                            "research_problem": "Research problem",
                            "architecture": "General architecture",
                            "innovation": "Key innovation",
                            "hardware_used": "Hardware type or null"
                        }
                    ],
                    "paper_describes_multiple_models": False
                }
                prompt = f"""<|system|>
You are an expert at extracting structured information from research papers about Large Language Models. You MUST extract information ONLY from the provided paper text. Do NOT use information from your training data. Return valid JSON with no comments or placeholders.
<|end|>
<|user|>
Read this research paper excerpt CAREFULLY and extract ALL LLM model versions/variants/sizes.

CRITICAL: Extract information ONLY from the paper text below. Do NOT use your training data or memory. Do NOT add models that are not mentioned in the paper.

IMPORTANT: Extract ALL model versions, variants, and sizes as SEPARATE entries:
- If paper mentions multiple sizes (8B, 70B, 405B) → create separate entry for each
- If paper mentions multiple versions (3.1, 3.2, 3.3) → create separate entry for each
- If paper mentions multiple variants (Base, Large, XL) → create separate entry for each
- Each distinct model size/version/variant = separate entry in models array

Return JSON matching this structure (replace placeholder values with actual data from the paper):

{json.dumps(example_json, indent=2)}

Paper excerpt:
{paper_snippet}

REQUIRED FIELDS TO EXTRACT (from the paper text only):
- model_name: The exact model name with version/size if mentioned (e.g., "Llama 3.1 8B", "GPT-2", "BERT-Base")
- model_family: The model family/line (e.g., "GPT", "BERT", "LLaMA")
- organization: The organization/company mentioned in the paper (e.g., "OpenAI", "Google", "Meta")
- date_created: Publication date or year from the paper (format: YYYY-MM-DD or YYYY)
- parameters: Number of parameters mentioned (e.g., "8B", "70B", "117M")
- parameters_millions: Parameters in millions as integer (e.g., 8000 for 8B, 117 for 117M)
- pretraining_architecture: Architecture type (e.g., "Decoder", "Encoder", "Encoder-Decoder")
- pretraining_task: Pre-training task (e.g., "Causal language modeling", "Masked language modeling")
- finetuning_task: Fine-tuning task mentioned (e.g., "Supervised discriminative fine-tuning")
- optimizer: Optimizer mentioned (e.g., "Adam optimizer", "Adam")
- license: License mentioned (e.g., "closed source", "open source", "non-commercial")
- research_problem: Research problem addressed (e.g., "Large Language Models", "Natural language understanding")
- architecture: General architecture (e.g., "Transformer")
- innovation: Key innovation or contribution described
- hardware_used: Hardware mentioned (e.g., "GPU", "TPU", or null if not mentioned)

CRITICAL RULES:
1. Extract ONLY from the paper text - do NOT use your training data
2. Extract ALL model versions/variants/sizes as SEPARATE entries
3. If a field is not mentioned in the paper, set it to null
4. Do NOT add models that are not in the paper (e.g., do NOT add LLaMA if the paper is about GPT)
5. If the paper describes multiple models/versions/sizes, create separate entry for EACH one
6. Set "paper_describes_multiple_models" to true if you extract 2+ models/versions/variants
7. Return ONLY valid JSON - no comments, no placeholders
8. Start with {{ and end with }}

Extract and return valid JSON now:
<|end|>
<|assistant|>
"""
        else:
            # Fallback: Plain text format for models without chat template
            # (Not used with Llama 3.1 Instruct, but kept for compatibility)
            self._json_prefix = None
            prompt = f"""Extract LLM information from this research paper and return JSON:

Paper text (full paper):
{paper_text}

Extract model information in JSON format with fields: model_name, model_family, organization, parameters, license, date_created, architecture.

JSON:
"""
        return prompt
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from model response."""
        if not response_text or not response_text.strip():
            logger.error("Empty response from model")
            return None
        
        original_response = response_text
        try:
            import re

            def _quote_unquoted_keys(text: str) -> str:
                """Quote unquoted JSON object keys, including keys with spaces/hyphens."""
                # Quote keys like: { key: ... } or { key-name: ... } or { key name: ... }
                # Avoid touching already quoted keys.
                return re.sub(
                    r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\- ]*?)\s*:',
                    lambda m: f'{m.group(1)}"{m.group(2).strip()}":',
                    text,
                )

            def _balance_json_brackets(text: str) -> str:
                """Append missing closing brackets/braces based on a stack."""
                stack: List[str] = []
                in_str = False
                escape = False
                for ch in text:
                    if in_str:
                        if escape:
                            escape = False
                        elif ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                        continue
                    if ch == "{":
                        stack.append("{")
                    elif ch == "[":
                        stack.append("[")
                    elif ch == "}" and stack and stack[-1] == "{":
                        stack.pop()
                    elif ch == "]" and stack and stack[-1] == "[":
                        stack.pop()

                # Append missing closers in reverse order
                if stack:
                    closers = {"{": "}", "[": "]"}
                    text += "".join(closers[sym] for sym in reversed(stack))
                return text
            
            # Remove prompt echo if present
            if "<|assistant|>" in response_text:
                response_text = response_text.split("<|assistant|>")[-1]
            
            # Remove markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) >= 2:
                    response_text = parts[1].strip()
            
            # Find JSON object - look for the complete JSON structure
            start = response_text.find("{")
            if start < 0:
                logger.error(f"No JSON object found in response. Response preview: {original_response[:200]}")
                return None
            
            # Find matching closing brace by counting braces
            brace_count = 0
            end = start
            for i, char in enumerate(response_text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end <= start:
                # Fallback to simple rfind
                end = response_text.rfind("}") + 1
            
            if end > start:
                response_text = response_text[start:end]
            else:
                logger.error(f"No valid JSON object found. Response preview: {original_response[:200]}")
                return None
            
            # === ENHANCED CLEANING FOR LLAMA 3.1 ===
            
            # Remove dots before field names (e.g., ."field_name" -> "field_name")
            response_text = re.sub(r'\.\s*"', '"', response_text)
            
            # Fix missing opening quotes for field names (handles spaces/hyphens too)
            response_text = _quote_unquoted_keys(response_text)
            
            # Fix missing closing quotes (e.g., "field":"value, -> "field":"value",)
            # Look for unquoted values before comma/brace/bracket
            response_text = re.sub(r':\s*([^",}\]]+?)(\s*[,}\]])', r': "\1"\2', response_text)
            
            # Fix cases like "short_description:"", -> "short_description":"",
            response_text = re.sub(r':\s*"",', r': "",', response_text)
            
            # Fix double quotes (e.g., ""value"" -> "value")
            response_text = re.sub(r'""([^"]+)""', r'"\1"', response_text)
            response_text = re.sub(r'""+', '"', response_text)
            
            # Fix empty comma patterns (e.g., ", ," or ",  ,")
            response_text = re.sub(r',\s*,', ',', response_text)
            
            # Fix field names with trailing spaces (e.g., "field_name " -> "field_name")
            response_text = re.sub(r'"(\w+)\s+":', r'"\1":', response_text)
            
            # Insert missing commas between objects in arrays (e.g., "} {")
            response_text = re.sub(r'}\s*{', '},{', response_text)

            # Insert missing commas between values and the next key (e.g., "value" "next_key":)
            response_text = re.sub(
                r'(?<=[0-9"\]}])\s+(?="[^"]+"\s*:)',
                ',',
                response_text,
            )

            # Remove control characters (newlines, tabs, etc.) that break JSON parsing
            # JSON doesn't allow unescaped control characters in strings
            response_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response_text)
            
            # Remove JSON comments (// and /* */)
            response_text = re.sub(r'//.*?$', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'/\*.*?\*/', '', response_text, flags=re.DOTALL)
            
            # Remove placeholder text like "Add more models here..."
            response_text = re.sub(r',?\s*//.*?Add more.*?$', '', response_text, flags=re.MULTILINE | re.IGNORECASE)
            
            # Remove trailing commas before } or ]
            response_text = re.sub(r',\s*}', '}', response_text)
            response_text = re.sub(r',\s*]', ']', response_text)
            
            # Fix leading commas after { or [
            response_text = re.sub(r'{\s*,', '{', response_text)
            response_text = re.sub(r'\[\s*,', '[', response_text)
            
            # Remove any text after the JSON (like "Note - ...")
            # Find the last } and truncate
            last_brace = response_text.rfind('}')
            if last_brace > 0:
                response_text = response_text[:last_brace + 1]

            # Balance brackets/braces if the JSON is truncated
            response_text = _balance_json_brackets(response_text)
            
            if not response_text.strip():
                logger.error("Response became empty after cleaning")
                logger.debug(f"Original response: {original_response[:500]}")
                return None
            
            # Try to parse JSON - if it fails, attempt repair
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError as parse_error:
                logger.warning(f"Initial JSON parse failed: {parse_error}. Attempting repair...")
                
                # Attempt to repair common issues
                # Fix incomplete strings (e.g., "value, -> "value",)
                response_text = re.sub(r':\s*"([^"]*?)([,}\]])', lambda m: f': "{m.group(1)}"{m.group(2)}' if not m.group(1).endswith('"') else f': {m.group(1)}{m.group(2)}', response_text)
                
                # Fix unclosed strings at end of object
                response_text = re.sub(r':\s*"([^"]*?)(\s*[}\]])', r': "\1"\2', response_text)
                
                # Try parsing again
                try:
                    parsed = json.loads(response_text)
                    logger.info("JSON repair successful")
                except json.JSONDecodeError as repair_error:
                    logger.error(f"JSON repair also failed: {repair_error}")
                    logger.error(f"Response length: {len(original_response)}")
                    logger.error(f"Response preview (first 500 chars): {original_response[:500]}")
                    logger.error(f"Cleaned response (first 500 chars): {response_text[:500] if response_text else 'EMPTY'}")
                    return None
            
            # Normalize field names in the parsed JSON
            parsed = self._normalize_field_names(parsed)
            
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response length: {len(original_response)}")
            logger.error(f"Response preview (first 500 chars): {original_response[:500]}")
            logger.error(f"Cleaned response (first 500 chars): {response_text[:500] if response_text else 'EMPTY'}")
            return None
    
    def _normalize_field_names(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field names to expected format."""
        # Field name mapping: various names -> correct name
        field_mapping = {
            # Model name variations
            "name": "model_name",
            "modelname": "model_name",
            "model": "model_name",
            "the model": "model_name",
            "the_model": "model_name",
            # Family variations
            "family": "model_family",
            "modelfamily": "model_family",
            # Organization variations
            "organisation": "organization",
            "org": "organization",
            "company": "organization",
            "org_full_name": "organization",
            "organizational_affiliation_of_authored_model": "organization",
            "author_organization": "organization",
            # Date variations
            "created_date": "date_created",
            "creation_date": "date_created",
            "date": "date_created",
            "publish_date": "date_created",
            # Parameters variations
            "params": "parameters",
            "param_count": "parameters",
            "params_count": "parameters",
            "parameter_count": "parameters",
            "params_count ": "parameters",  # With trailing space
            "parameters_size_in_million_params": "parameters_millions",
            "params_millions": "parameters_millions",
            "param_millions": "parameters_millions",
            # Architecture variations
            "arch": "architecture",
            "model_architecture": "architecture",
            "pretraining_arch": "pretraining_architecture",
            # Optimizer variations
            "optimizer_algorithm": "optimizer",
            "optim": "optimizer",
            # License variations
            "licence": "license",
            "licence_type": "license",
            "license_type": "license",
            # Other variations
            "hw_used": "hardware_used",
            "hardware": "hardware_used",
        }
        
        def normalize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize field names in a dictionary."""
            normalized = {}
            for key, value in d.items():
                # Normalize key: lowercase, strip whitespace
                norm_key = key.lower().strip().replace(" ", "_")
                
                # Map to correct field name if known
                if norm_key in field_mapping:
                    norm_key = field_mapping[norm_key]
                else:
                    # Keep original key if not in mapping
                    norm_key = key
                
                # Recursively normalize nested dicts
                if isinstance(value, dict):
                    value = normalize_dict(value)
                elif isinstance(value, list):
                    value = [normalize_dict(v) if isinstance(v, dict) else v for v in value]
                
                normalized[norm_key] = value
            
            return normalized
        
        # Normalize the top-level keys
        result = normalize_dict(data)
        
        # Handle case where "Models" is used instead of "models"
        if "Models" in result and "models" not in result:
            result["models"] = result.pop("Models")
        
        return result
    
    def extract(
        self,
        paper_text: str,
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MultiModelResponse]:
        """Extract LLM information using local model on GPU."""
        try:
            # #region agent log
            _debug_log("A", "extract:entry", "Extract method called", {
                "model_name": self.model_name,
                "device": str(self.device),
                "paper_text_len": len(paper_text)
            })
            # #endregion
            
            logger.info("Extracting with transformers on GPU")
            
            prompt = self._create_extraction_prompt(paper_text, paper_metadata)
            
            # #region agent log
            _debug_log("A", "extract:after_prompt", "Prompt created", {
                "prompt_len": len(prompt),
                "prompt_preview": prompt[:200],
                "model_name": self.model_name
            })
            # #endregion
            
            # Llama 3.1 8B supports 128K tokens - use 32K for balance between context and memory
            # This leaves room for prompt and generated response
            max_length = 32768
            
            # #region agent log
            _debug_log("B", "extract:before_tokenize", "Before tokenization", {
                "max_length": max_length,
                "vocab_size": len(self.tokenizer),
                "model_vocab_size": self.model.config.vocab_size if hasattr(self.model, 'config') else None,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "bos_token_id": getattr(self.tokenizer, 'bos_token_id', None)
            })
            # #endregion
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,  # No padding needed for single sequence
                return_attention_mask=True
            )
            
            # #region agent log
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            model_vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else None
            tokenizer_vocab_size = len(self.tokenizer)
            input_ids_min = int(input_ids.min().item())
            input_ids_max = int(input_ids.max().item())
            invalid_tokens = []
            if model_vocab_size:
                invalid_tokens = [int(t.item()) for t in input_ids.flatten() if int(t.item()) >= model_vocab_size]
                if invalid_tokens:
                    logger.error(f"Found {len(invalid_tokens)} invalid token IDs (>= {model_vocab_size})")
                    logger.error(f"Invalid token IDs: {invalid_tokens[:10]}")
                    raise ValueError(f"Invalid token IDs detected: tokens {invalid_tokens[:5]} are >= vocab size {model_vocab_size}")
            
            # Log actual model being used
            actual_model_size = getattr(self.model.config, 'hidden_size', None)
            logger.info(f"Using model: {self.model_name}, hidden_size: {actual_model_size}, vocab_size: {model_vocab_size}")
            
            _debug_log("B", "extract:after_tokenize", "After tokenization", {
                "input_ids_shape": list(input_ids.shape),
                "input_ids_min": input_ids_min,
                "input_ids_max": input_ids_max,
                "input_ids_len": int(input_ids.shape[1]),
                "attention_mask_shape": list(attention_mask.shape) if attention_mask is not None else None,
                "attention_mask_dtype": str(attention_mask.dtype) if attention_mask is not None else None,
                "attention_mask_min": int(attention_mask.min().item()) if attention_mask is not None else None,
                "attention_mask_max": int(attention_mask.max().item()) if attention_mask is not None else None,
                "tokenizer_vocab_size": tokenizer_vocab_size,
                "model_vocab_size": model_vocab_size,
                "vocab_size_match": tokenizer_vocab_size == model_vocab_size if model_vocab_size else None,
                "has_invalid_tokens": len(invalid_tokens) > 0,
                "invalid_token_count": len(invalid_tokens),
                "invalid_token_ids": invalid_tokens[:10] if invalid_tokens else []
            })
            # #endregion
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # #region agent log
            _debug_log("C", "extract:after_to_device", "After moving to device", {
                "device": str(self.device),
                "input_ids_device": str(input_ids.device),
                "attention_mask_device": str(attention_mask.device) if attention_mask is not None else None
            })
            # #endregion
            
            with torch.no_grad():
                # Ensure pad_token_id and eos_token_id are set correctly and valid
                model_vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else None
                
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                if model_vocab_size and pad_token_id is not None and pad_token_id >= model_vocab_size:
                    logger.warning(f"pad_token_id {pad_token_id} >= vocab_size {model_vocab_size}, using eos_token_id instead")
                    pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
                
                eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
                if model_vocab_size and eos_token_id is not None and eos_token_id >= model_vocab_size:
                    logger.warning(f"eos_token_id {eos_token_id} >= vocab_size {model_vocab_size}, using 0 instead")
                    eos_token_id = 0
                
                # Final validation: ensure IDs are within valid range
                if model_vocab_size:
                    if pad_token_id is not None and pad_token_id >= model_vocab_size:
                        pad_token_id = 0
                    if eos_token_id is not None and eos_token_id >= model_vocab_size:
                        eos_token_id = 0
                
                # Llama 3.1 instruction-tuned model: use full token allowance
                actual_max_tokens = self.max_new_tokens
                
                # Validate bos_token_id if it exists
                bos_token_id = None
                if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                    bos_token_id = self.tokenizer.bos_token_id
                    if model_vocab_size and bos_token_id >= model_vocab_size:
                        logger.warning(f"bos_token_id {bos_token_id} >= vocab_size {model_vocab_size}, setting to None")
                        bos_token_id = None
                
                # #region agent log
                _debug_log("D", "extract:before_generate", "Before model.generate", {
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_id,
                    "bos_token_id": bos_token_id,
                    "max_new_tokens": actual_max_tokens,
                    "temperature": self.temperature,
                    "do_sample": True if self.temperature > 0 else False,
                    "input_ids_shape": list(inputs['input_ids'].shape),
                    "attention_mask_passed": inputs.get('attention_mask') is not None
                })
                # #endregion
                
                # Strategy: Use low-temperature sampling for Llama 3.1 8B
                # Greedy decoding was too deterministic and produced malformed JSON
                # Sampling allows the model more flexibility to generate valid JSON
                outputs = None
                generation_successful = False
                
                # Attempt: Low-temperature sampling (better for structured output)
                try:
                    logger.info("Attempting low-temperature sampling for structured output")
                    torch.cuda.empty_cache()  # Clear cache before generation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=actual_max_tokens,  # Use full allowance for detailed extraction
                        temperature=0.1,  # Very low temp for deterministic structured output
                        do_sample=True,  # Enable sampling
                        top_p=0.95,  # Slightly higher top_p for better coherence
                        repetition_penalty=1.1,  # Lower penalty - avoid breaking JSON structure
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        bos_token_id=bos_token_id if bos_token_id is not None else None,
                        use_cache=True
                    )
                    # Move to CPU immediately to protect from future CUDA errors
                    outputs = outputs.cpu()
                    generation_successful = True
                    logger.info("Low-temperature sampling completed successfully")
                except RuntimeError as e:
                    if "CUDA" in str(e) or "device-side assert" in str(e):
                        logger.error(f"CUDA error during sampling: {e}")
                        # Retry with smaller context/output to reduce GPU pressure
                        logger.info("Retrying on GPU with reduced max_new_tokens and no cache")
                        try:
                            torch.cuda.empty_cache()
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=min(512, actual_max_tokens),
                                temperature=0.1,
                                do_sample=True,
                                top_p=0.9,
                                repetition_penalty=1.05,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                                bos_token_id=bos_token_id if bos_token_id is not None else None,
                                use_cache=False
                            )
                            outputs = outputs.cpu()
                            generation_successful = True
                            logger.info("Reduced-output sampling completed successfully")
                        except RuntimeError as e_retry:
                            logger.error(f"Reduced-output GPU retry failed: {e_retry}")
                            logger.info("Attempting CPU-based sampling as fallback")
                        try:
                            # Move model to CPU
                            logger.info("Moving model to CPU for safe sampling")
                            model_cpu = self.model.cpu()
                            inputs_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                            
                            logger.info("Starting CPU-based generation...")
                            outputs = model_cpu.generate(
                                **inputs_cpu,
                                max_new_tokens=actual_max_tokens,  # Match GPU settings
                                temperature=0.1,     # Match GPU settings
                                do_sample=True,
                                top_p=0.95,
                                repetition_penalty=1.1,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                                bos_token_id=bos_token_id if bos_token_id is not None else None,
                                use_cache=True
                            )
                            generation_successful = True
                            logger.info("CPU-based sampling completed successfully")
                            
                            # Move model back to GPU
                            self.model = model_cpu.to(self.device)
                            logger.info("Model moved back to GPU")
                        except Exception as cpu_e:
                            logger.error(f"CPU-based sampling also failed: {cpu_e}")
                            raise RuntimeError(f"Generation failed on both GPU and CPU: {cpu_e}") from cpu_e
                    else:
                        raise
                
                if outputs is None:
                    raise RuntimeError("Failed to generate output: both greedy and sampling attempts failed")
                
                # Note: outputs are already moved to CPU right after generation (see above)
                # This prevents CUDA errors from corrupting the tensor before we can decode it
            
            # #region agent log
            _debug_log("E", "extract:after_generate", "After model.generate (success)", {
                "outputs_shape": list(outputs.shape) if hasattr(outputs, 'shape') else None,
                "outputs_len": int(outputs.shape[0]) if hasattr(outputs, 'shape') and len(outputs.shape) > 0 else None,
                "outputs_device": str(outputs.device) if hasattr(outputs, 'device') else None
            })
            # #endregion
            
            # #region agent log
            try:
                # Decode full output (includes prompt) - now safe because outputs are on CPU
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only newly generated tokens (after input length)
                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[0][input_length:]
                response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # Check for repetitive output (common issue with greedy decoding)
                if len(response_text) > 50:
                    # Check if response is mostly the same character
                    unique_chars = len(set(response_text[:100]))
                    if unique_chars < 3:
                        logger.error(f"Repetitive output detected: response contains mostly same character(s)")
                        logger.error(f"Response preview: {repr(response_text[:200])}")
                        logger.error("This usually indicates the model is stuck in a repetition loop")
                        # Try to extract any JSON that might be at the start before repetition
                        json_start = response_text.find("{")
                        if json_start >= 0:
                            logger.info(f"Found JSON start at position {json_start}, attempting to extract...")
                            response_text = response_text[:json_start + 1000]  # Take first part with JSON
                
                # Log if response is suspiciously short
                if len(response_text.strip()) < 10:
                    logger.warning(f"Very short response generated: {len(response_text)} chars")
                    logger.warning(f"New tokens count: {new_tokens.shape[0]}")
                    logger.warning(f"Response text: {repr(response_text)}")
                
                _debug_log("G", "extract:after_decode", "After tokenizer.decode", {
                    "full_text_length": len(full_text),
                    "full_text_preview": full_text[:500],
                    "response_length": len(response_text),
                    "response_preview": response_text[:500],
                    "response_full": response_text[:2000],  # First 2000 chars
                    "input_length": int(input_length),
                    "output_length": int(outputs[0].shape[0]),
                    "new_tokens_count": int(new_tokens.shape[0]),
                    "response_is_empty": len(response_text.strip()) == 0
                })
            except Exception as decode_err:
                _debug_log("G", "extract:decode_error", "Error during decode", {
                    "error": str(decode_err),
                    "error_type": type(decode_err).__name__
                })
                raise
            # #endregion
            
            # #region agent log
            _debug_log("G", "extract:raw_response", "Raw model response (new tokens only)", {
                "response_length": len(response_text),
                "response_preview": response_text[:500],
                "response_full": response_text[:2000]  # First 2000 chars
            })
            # #endregion
            
            json_data = self._parse_json_response(response_text)
            
            # #region agent log
            _debug_log("G", "extract:after_parse", "After JSON parsing", {
                "json_data": json_data is not None,
                "json_keys": list(json_data.keys()) if json_data else None,
                "parse_success": json_data is not None
            })
            # #endregion
            
            if not json_data:
                logger.warning("Failed to parse JSON from model response")
                # #region agent log
                _debug_log("G", "extract:parse_failed", "JSON parse failed", {
                    "response_text": response_text[:1000]  # First 1000 chars for debugging
                })
                # #endregion
                return None
            
            # 1. Normalize flat JSON to nested structure if model forgot the wrapper
            # Do this first so all subsequent logic works on 'models' list
            if 'models' not in json_data and 'model_name' in json_data:
                logger.info("Normalizing flat JSON response to 'models' list")
                json_data = {'models': [json_data]}

            # 2. Standardize model fields (organization, parameters, etc.)
            if 'models' in json_data:
                for model_data in json_data['models']:
                    # Ensure basic fields exist at least as None
                    for field in ['organization', 'parameters', 'license']:
                        if field not in model_data:
                            model_data[field] = None
                    
                    # Convert parameters string to parameters_millions
                    if model_data.get('parameters'):
                        import re
                        params_str = str(model_data['parameters'])
                        # Only convert if parameters_millions is missing, None, or 0
                        if not model_data.get('parameters_millions') or model_data.get('parameters_millions') == 0:
                            match = re.search(r'(\d+\.?\d*)', params_str.replace(',', ''))
                            if match:
                                num = float(match.group(1))
                                if 'B' in params_str.upper() or 'billion' in params_str.lower():
                                    num = num * 1000
                                model_data['parameters_millions'] = int(num)
                            else:
                                model_data['parameters_millions'] = None

                    # Set defaults for all other required fields
                    defaults = {
                        'model_family': model_data.get('model_name', '').split('-')[0] if model_data.get('model_name') else None,
                        'date_created': None,
                        'pretraining_architecture': None,
                        'pretraining_task': None,
                        'finetuning_task': None,
                        'optimizer': None,
                        'research_problem': None,
                        'architecture': None,
                        'innovation': None,
                        'hardware_used': None
                    }
                    for key, default_value in defaults.items():
                        if key not in model_data or model_data[key] is None:
                            model_data[key] = default_value

            # 3. Add top-level required fields
            if 'paper_describes_multiple_models' not in json_data:
                json_data['paper_describes_multiple_models'] = len(json_data.get('models', [])) > 1
            
            # 4. Final Validation
            result = MultiModelResponse(**json_data)
            
            if result and result.models:
                logger.info(f"Extracted {len(result.models)} model(s)")
                
                if paper_metadata:
                    for model in result.models:
                        if not model.paper_title and 'title' in paper_metadata:
                            model.paper_title = paper_metadata['title']
                
                return result
            
            logger.warning("No models extracted")
            return None
            
        except Exception as e:
            # #region agent log
            import traceback
            _debug_log("F", "extract:exception", "Exception during extraction", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_traceback": traceback.format_exc()[:1000]
            })
            # #endregion
            logger.error(f"Extraction error: {e}", exc_info=True)
            return None
    
    def extract_from_chunks(
        self,
        text_chunks: List[str],
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MultiModelResponse]:
        """Extract from multiple text chunks."""
        all_models = []
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
            result = self.extract(chunk, paper_metadata)
            if result and result.models:
                all_models.extend(result.models)
        
        if not all_models:
            return None
        
        # Deduplicate
        seen = {}
        for model in all_models:
            key = (model.model_name, model.model_version, model.parameters)
            if key not in seen:
                seen[key] = model
            else:
                existing = seen[key]
                for field_name, field_value in model.model_dump().items():
                    if field_value is not None and getattr(existing, field_name) is None:
                        setattr(existing, field_name, field_value)
        
        unique_models = list(seen.values())
        return MultiModelResponse(
            models=unique_models,
            paper_describes_multiple_models=len(unique_models) > 1
        )

