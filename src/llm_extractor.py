"""
LLM Extractor using KISSKI Chat AI API

This module extracts structured information from LLM research papers
using the KISSKI Chat AI API (SAIA platform).

The KISSKI API is OpenAI-compatible and hosted by GWDG Academic Cloud.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMProperties(BaseModel):
    """Properties of an LLM model following ORKG template R609825."""

    model_name: str
    model_family: Optional[str] = None
    date_created: Optional[str] = None
    organization: Optional[str] = None
    innovation: Optional[str] = None
    pretraining_corpus: Optional[str] = None
    parameters: Optional[str] = None
    parameters_millions: Optional[int] = None
    application: Optional[str] = None
    license: Optional[str] = None
    research_problem: Optional[str] = None
    model_version: Optional[str] = None
    pretraining_architecture: Optional[str] = None
    pretraining_task: Optional[str] = None
    training_corpus_size: Optional[str] = None
    knowledge_cutoff_date: Optional[str] = None
    finetuning_task: Optional[str] = None
    finetuning_data: Optional[str] = None
    optimizer: Optional[str] = None
    tokenizer: Optional[str] = None
    context_length: Optional[str] = None
    supported_language: Optional[str] = None
    hardware_used: Optional[str] = None
    hardware_description: Optional[str] = None
    carbon_emitted: Optional[str] = None
    extension: Optional[str] = None
    source_code: Optional[str] = None
    blog_post: Optional[str] = None
    training_data: Optional[str] = None
    training_compute: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    release_date: Optional[str] = None
    model_type: Optional[str] = None
    paper_title: Optional[str] = None


class MultiModelResponse(BaseModel):
    """Response containing multiple extracted models."""

    models: List[LLMProperties]
    paper_describes_multiple_models: bool


class LLMExtractor:
    """Extracts LLM information using KISSKI Chat AI API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://chat-ai.academiccloud.de/v1",
        model: str = "meta-llama-3.1-8b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        timeout: int = 60,
        rate_limit_delay: float = 2.0,
    ):
        """
        Initialize KISSKI API extractor.

        Args:
            api_key: KISSKI API key (provided by professor)
            base_url: KISSKI API base URL
            model: Model name (see KISSKI documentation for available models)
                   Default: meta-llama-3.1-8b-instruct
                   Recommended alternatives:
                   - openai-gpt-oss-120b (best performance)
                   - qwen3-32b (good reasoning)
                   - deepseek-r1-0528 (reasoning tasks)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests in seconds (default: 2.0)

        Note:
            KISSKI API is OpenAI-compatible. Rate limits:
            - 1000 requests per minute
            - 10000 requests per hour
            - 50000 requests per day
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        # Initialize OpenAI client configured for KISSKI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        logger.info(f"Initialized KISSKI extractor with model: {model}")
        logger.info(f"API endpoint: {base_url}")
        logger.info(f"Rate limit: {rate_limit_delay}s between requests")

    def _enforce_rate_limit(self):
        """
        Enforce rate limiting between API requests.

        Implements client-side rate limiting to avoid overloading KISSKI servers
        as requested by professor. Server has limits of:
        - 1000 requests/minute
        - 10000 requests/hour
        - 50000 requests/day
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _create_extraction_messages(
        self, paper_text: str, paper_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Create extraction messages with few-shot examples (matching Grete approach).

        Returns list of messages for OpenAI chat API with few-shot examples.
        """
        # Inject metadata if available
        meta_str = ""
        if paper_metadata:
            meta_str = f"PAPER METADATA:\nTitle: {paper_metadata.get('title', 'Unknown')}\nAuthored: {paper_metadata.get('year', '')}-{paper_metadata.get('month', '')}\nAuthors: {paper_metadata.get('authors', [])}\n"  # noqa: E501

        # Use up to 65,000 chars (matching Grete)
        paper_snippet = paper_text[:65000] if len(paper_text) > 65000 else paper_text

        # Prepend metadata to snippet
        if meta_str:
            paper_snippet = meta_str + "\n\nPAPER CONTENT:\n" + paper_snippet

        # Few-shot examples (matching Grete approach)
        # Example 1: BERT (with all ORKG R609825 required fields)
        example1_input = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Google AI Language. We introduce BERT with 110M, 340M parameters. It uses a Transformer encoder architecture trained on Masked LM and Next Sentence Prediction tasks. It achieves state-of-the-art on GLUE. We use Adam optimizer. Trained on English Wikipedia and BookCorpus."  # noqa: E501
        example1_output = {
            "models": [
                {
                    "model_name": "BERT",
                    "model_family": "BERT",
                    "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",  # noqa: E501
                    "organization": "Google",
                    "parameters": "340M",
                    "parameters_millions": 340,
                    "date_created": "2018-10",
                    "pretraining_architecture": "Encoder",
                    "pretraining_task": "Masked LM (MLM), Next Sentence Prediction (NSP)",
                    "pretraining_corpus": "English Wikipedia, BookCorpus",
                    "optimizer": "Adam",
                    "innovation": "BERT's primary innovation is the masked language model (MLM) approach, inspired by the Cloze task. This method masks random tokens and trains the model to predict them, enabling bidirectional context understanding.",  # noqa: E501
                    "research_problem": "Language Understanding",
                    "application": "Natural language understanding, question answering, text classification",  # noqa: E501
                    "license": "Apache 2.0",
                }
            ]
        }

        # Example 2: GPT-2 (with all ORKG R609825 required fields)
        example2_input = "Language Models are Unsupervised Multitask Learners. OpenAI. We trained a 1.5 billion parameter Transformer decoder language model. It demonstrates zero-shot task transfer. We assume a causal language modeling objective. Trained on WebText dataset."  # noqa: E501
        example2_output = {
            "models": [
                {
                    "model_name": "GPT-2",
                    "model_family": "GPT",
                    "paper_title": "Language Models are Unsupervised Multitask Learners",
                    "organization": "OpenAI",
                    "parameters": "1.5B",
                    "parameters_millions": 1500,
                    "date_created": "2019-02",
                    "pretraining_architecture": "Decoder",
                    "pretraining_task": "Causal language modeling",
                    "pretraining_corpus": "WebText",
                    "innovation": "Zero-shot task transfer via large-scale unsupervised learning",
                    "research_problem": "Large Language Models",
                    "application": "Text generation, language modeling, zero-shot task transfer",  # noqa: E501
                    "license": "Modified MIT License",
                }
            ]
        }

        # Example 3: GPT-1 (with all ORKG R609825 required fields)
        example3_input = "Improving Language Understanding by Generative Pre-Training. Alec Radford, OpenAI. We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. Our approach employs a Transformer-based architecture with 117M parameters. We use the Adam optimizer. Trained on BooksCorpus dataset."  # noqa: E501
        example3_output = {
            "models": [
                {
                    "model_name": "GPT-1",
                    "model_family": "GPT",
                    "paper_title": "Improving Language Understanding by Generative Pre-Training",  # noqa: E501
                    "organization": "OpenAI",
                    "parameters": "117M",
                    "parameters_millions": 117,
                    "date_created": "2018-06",
                    "pretraining_architecture": "Decoder",
                    "pretraining_task": "Causal language modeling",
                    "pretraining_corpus": "BooksCorpus",
                    "finetuning_task": "Supervised discriminative fine-tuning",
                    "optimizer": "Adam",
                    "innovation": "Generative pre-training followed by discriminative fine-tuning",
                    "license": "closed source",
                    "research_problem": "Language Understanding",
                    "application": "Natural language understanding, text classification, question answering",  # noqa: E501
                }
            ]
        }

        # Example 4: Multiple model versions (Llama 3.1 - all ORKG R609825 required fields)
        example4_input = "The Llama 3.1 Herd of Models. Meta AI. We introduce Llama 3.1 with three model sizes: 8B, 70B, and 405B parameters. All models use Transformer decoder architecture. The 8B model has 8 billion parameters, the 70B model has 70 billion parameters, and the 405B model has 405 billion parameters. All models are trained on the same pretraining task. Trained on large-scale text corpus. Applications include chat, instruction following, and general language tasks. Released under Llama 3.1 Community License."  # noqa: E501
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
                    "pretraining_corpus": "Large-scale text corpus",
                    "innovation": "Large-scale language models",
                    "research_problem": "Large Language Models",
                    "application": "Chat, instruction following, general language tasks",
                    "license": "Llama 3.1 Community License",
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
                    "pretraining_corpus": "Large-scale text corpus",
                    "innovation": "Large-scale language models",
                    "research_problem": "Large Language Models",
                    "application": "Chat, instruction following, general language tasks",
                    "license": "Llama 3.1 Community License",
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
                    "pretraining_corpus": "Large-scale text corpus",
                    "innovation": "Large-scale language models",
                    "research_problem": "Large Language Models",
                    "application": "Chat, instruction following, general language tasks",
                    "license": "Llama 3.1 Community License",
                },
            ],
            "paper_describes_multiple_models": True,
        }

        # Build messages with few-shot examples
        messages = [
            {
                "role": "system",
                "content": "You are an expert AI researcher extracting information according to ORKG template R609825. Extract DETAILED information about ALL MODEL VERSIONS/VARIANTS introduced in the paper.\n\nREQUIRED FIELDS (must extract for each model):\n- model_name (required): Exact model name with version/size\n- model_family (required): Model family/series (e.g., GPT, BERT, Llama)\n- date_created (required): Publication date (YYYY-MM-DD or YYYY)\n- organization (required): Organization/company\n- innovation (required): Key innovation or contribution. Prefer the paper's own framing: name the main technique or method (e.g. \"masked language model\", \"Cloze\") and how it differs from prior work (e.g. \"enabling bidirectional context\"). One or two sentences.\n- pretraining_corpus (required): Training dataset/corpus\n- research_problem (required): Research problem addressed\n- parameters (required): Number of parameters as text (e.g., \"7B\", \"175B\")\n- parameters_millions (required): Parameters as integer in millions (e.g., 7000 for 7B)\n- application (required): Use cases/applications\n- license (required): License type\n\nMUST EXTRACT WHEN MENTIONED (use null only if not stated):\n- pretraining_architecture: MUST be exactly one of \"Encoder\", \"Decoder\", or \"Encoder-Decoder\" (encoder-only, decoder-only, or both). Determine from the paper; use null only if not stated.\n- pretraining_task (e.g. Causal language modeling, Masked LM, Next-token prediction)\n- finetuning_task (e.g. Supervised discriminative fine-tuning)\n- optimizer: ONLY if the paper explicitly names an optimizer (e.g. Adam, AdamW). If the paper does NOT mention the optimizer, you MUST use null. Do NOT guess or infer from other papers or prior knowledge.\n- extension: ONLY when the paper explicitly describes an additional technical detail or mechanism that extends the model beyond a baseline (e.g. a specific encoding, module, or technique that enables a capability compared to prior work). One sentence, factual. Example: \"Relative positioned embeddings enable longer-context attention when compared to vanilla Transformer model.\" If the paper does NOT mention such an extension, use null; do NOT guess or infer from other papers.\n\nOPTIONAL: tokenizer, hardware_used, etc.\n\nCRITICAL RULES:\n1. TITLE: Extract the official, full RESEARCH PAPER TITLE and assign it to 'paper_title'.\n2. ALL VARIANTS: Extract ALL model versions, sizes, and variants as SEPARATE entries.\n3. PARAMETERS: Search for 'Our model' or 'Proposed'. Look for 'M' or 'B'. Extract parameter sizes for each variant. Calculate parameters_millions (e.g., 7B = 7000, 117M = 117).\n4. DATES: Prefer YYYY-MM (e.g. 2018-10). Use YYYY-MM-DD when day is known, else YYYY-MM, else YYYY. Priority: metadata > header/footer > citation year.\n5. ORGANIZATION: Use canonical name (e.g. Google, OpenAI, Meta) not long form (e.g. not \"Google AI Language\").\n6. PARAMETERS: For multiple sizes use comma-separated (e.g. \"110M, 340M\").\n7. MULTIPLE MODELS: Set 'paper_describes_multiple_models' to true if the paper describes multiple distinct models, versions, or size variants.\n8. REQUIRED FIELDS: You MUST extract all required fields. If a field is not mentioned in the paper, use null, but prioritize extracting from paper text.\n9. TABLES: If the paper includes a [TABLES FROM DOCUMENT] block, the content is markdown tables from the PDF. Use these tables as the primary source for model names, metrics (e.g. F1, BERTScore), parameter counts, and dataset names; prefer exact values from table cells.\n10. CONTEXT VARIANTS: Do NOT create separate entries for context-window variants of the same model (e.g. 'Llama 3 8K' and 'Llama 3 128K-context' are the SAME model as 'Llama 3'). Record the context length in the context_length field of that single entry instead.\n11. STAGE VARIANTS: Do NOT create separate entries for pre-trained vs post-trained (instruction-tuned) variants of the same model (e.g. 'Llama 3 (pre-trained)' and 'Llama 3 (post-trained)' are ONE entry 'Llama 3'). Mention both stages in the innovation or finetuning_task fields.\n\nFORMAT: date_created=YYYY-MM; organization=canonical name (Google/OpenAI/Meta); parameters=comma-separated sizes when multiple.\n\nReturn JSON only.",  # noqa: E501
            },
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example1_input}",  # noqa: E501
            },
            {"role": "assistant", "content": json.dumps(example1_output)},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example2_input}",  # noqa: E501
            },
            {"role": "assistant", "content": json.dumps(example2_output)},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example3_input}",  # noqa: E501
            },
            {"role": "assistant", "content": json.dumps(example3_output)},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example4_input}",  # noqa: E501
            },
            {"role": "assistant", "content": json.dumps(example4_output)},
            {
                "role": "user",
                "content": f"""Extract ALL model versions, variants, and sizes (ORKG R609825):

{paper_snippet}

REQUIRED FIELDS (must extract for each model):
1. model_name (required): Exact model name with version/size if mentioned
2. model_family (required): Model family/series (e.g., GPT, BERT, Llama)
3. date_created (required): Publication date from paper (YYYY-MM-DD or YYYY)
4. organization (required): Organization/company that created the model
5. innovation (required): Key innovation or contribution. Use the paper's own terms for the main \
method (e.g. MLM, Cloze, bidirectional) and keep to 1-2 sentences.
6. pretraining_corpus (required): Training dataset/corpus mentioned
7. research_problem (required): Research problem addressed
8. parameters (required): Number of parameters as text (e.g., "7B", "175B", "117M")
9. parameters_millions (required): Parameters in millions (e.g., 7000 for 7B, 117 for 117M)
10. application (required): Use cases/applications mentioned
11. license (required): License (e.g., "open source", "closed source", "Apache 2.0")

CRITICAL INSTRUCTIONS:
- Extract ALL model versions/variants (e.g. "Llama 3.1 8B/70B/405B" -> 3 entries)
- Extract ALL model sizes mentioned (different parameter counts = different entries)
- Extract ALL model versions mentioned (3.1, 3.2, 3.3 = separate entries)
- Extract ALL architectural variants (Base, Large, XL, etc. = separate entries)
- Each distinct model size/version/variant = SEPARATE entry in models array
- Extract models THIS paper introduces (main contributions)
- NOT models mentioned as related work or comparisons
- Focus on PRIMARY model contributions intended as standalone released models.
- Do NOT create separate entries for auxiliary artifacts such as tools, guards,
  safety filters, adapters, encoders, tokenizers, pipelines, or infrastructure modules
  when the paper also contains main model contributions.
- If auxiliary artifacts are mentioned, capture them inside innovation/extension fields
  of the relevant primary model instead of as standalone models.
- Do NOT create separate entries for context-window variants of the same model
  (e.g. "Llama 3 8K" and "Llama 3 128K context" are ONE entry "Llama 3"; put the
  context length in the context_length field).
- Do NOT create separate entries for pre-trained vs post-trained variants of the same
  model (e.g. "Llama 3 (pre-trained)" and "Llama 3 (post-trained)" are ONE entry
  "Llama 3"; describe both stages in innovation or finetuning_task).
- Model name include version/size if mentioned (e.g. "Llama 3.1 8B" not just "Llama")
- Model name is NOT the architecture (e.g. "GPT" not "Transformer")
- If multiple models, set "paper_describes_multiple_models": true
- parameters_millions: "7B"->7000, "117M"->117, "1.5B"->1500
- Extract ALL required fields. Use null if not in paper; \
infer when possible.
- pretraining_architecture: use exactly one of Encoder, Decoder, or Encoder-Decoder \
(determine from the paper); null if not stated.
- Always extract pretraining_architecture, pretraining_task, finetuning_task when stated. \
For optimizer: extract ONLY when the paper explicitly mentions it; if the paper does not \
mention the optimizer, use null and do NOT guess or infer from other papers. For extension: \
extract ONLY when the paper explicitly states a technical extension or mechanism (e.g. a \
specific encoding or technique that extends the model vs a baseline); one sentence; if not \
mentioned, use null; do not infer from other papers.
- FORMAT: date_created use YYYY-MM when possible; organization use canonical name \
(Google, OpenAI, Meta); optimizer = algorithm name only when stated in the paper, \
otherwise null; extension = one-sentence technical detail when stated, otherwise null.

Output JSON:""",
            },
        ]

        return messages

    def extract(
        self, paper_text: str, paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MultiModelResponse]:
        """
        Extract LLM information from paper text using KISSKI API with few-shot examples.

        Args:
            paper_text: Full text of the research paper
            paper_metadata: Optional metadata (title, authors, etc.)

        Returns:
            MultiModelResponse with extracted models or None if extraction failed
        """
        try:
            logger.info("Extracting LLM information using KISSKI API")

            # Enforce rate limiting
            self._enforce_rate_limit()

            # Create messages with few-shot examples (matching Grete approach)
            messages = self._create_extraction_messages(paper_text, paper_metadata)

            # Call KISSKI API (OpenAI-compatible)
            logger.debug(f"Sending request to KISSKI API (model: {self.model_name})")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract response text
            if not response.choices or len(response.choices) == 0:
                logger.warning("No response from KISSKI API")
                return None

            response_text = response.choices[0].message.content

            if not response_text or not response_text.strip():
                logger.warning("Empty response from KISSKI API")
                return None

            logger.debug(f"Received response ({len(response_text)} characters)")

            # Parse JSON response
            json_data = self._parse_json_response(response_text)
            if not json_data:
                logger.warning("Failed to parse JSON from response")
                return None

            # Coerce "null" strings to None so evaluation and ORKG get proper nulls
            json_data = self._coerce_null_strings(json_data)

            # Ensure required fields have defaults
            if "models" in json_data:
                for model_data in json_data["models"]:
                    # Set None for missing required fields
                    if not model_data.get("organization"):
                        model_data["organization"] = None
                    if not model_data.get("parameters"):
                        model_data["parameters"] = None
                    if not model_data.get("license"):
                        model_data["license"] = None

            # Validate against schema
            result = MultiModelResponse(**json_data)

            if result and result.models:
                logger.info(f"Successfully extracted {len(result.models)} model(s)")

                # Enrich with paper metadata
                if paper_metadata:
                    for model in result.models:
                        if not model.paper_title and "title" in paper_metadata:
                            model.paper_title = paper_metadata["title"]
                        if not model.organization and "authors" in paper_metadata:
                            model.organization = self._extract_organization(
                                paper_metadata.get("authors", [])
                            )

                return result
            else:
                logger.warning("No models extracted")
                return None

        except Exception as e:
            logger.error(f"Extraction error: {e}", exc_info=True)
            return None

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from model response with robust parsing (matching Grete approach).

        Handles responses wrapped in markdown code blocks and common JSON issues.
        """
        if not response_text or not response_text.strip():
            logger.error("Empty response from model")
            return None

        original_response = response_text
        try:
            import re

            def _quote_unquoted_keys(text: str) -> str:
                """Quote unquoted JSON object keys, including keys with spaces/hyphens."""
                return re.sub(
                    r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\- ]*?)\s*:",
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
                logger.error(
                    f"No JSON object found in response. Response preview: {original_response[:200]}"
                )
                return None

            # Find matching closing brace by counting braces
            brace_count = 0
            end = start
            for i, char in enumerate(response_text[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
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
                logger.error(
                    f"No valid JSON object found. Response preview: {original_response[:200]}"
                )
                return None

            # BEST PRACTICE: Try parsing the minimally cleaned JSON first
            # Only apply aggressive repair if standard parsing fails
            parsed = None
            try:
                parsed = json.loads(response_text)
                # Success: valid JSON, no cleaning needed
            except json.JSONDecodeError as initial_error:
                # Parsing failed - apply enhanced cleaning and repair
                logger.debug(f"Initial parse failed: {initial_error}. Applying cleaning...")

                # Enhanced cleaning (only when needed)
                # Remove dots before field names (e.g., ."field_name" -> "field_name")
                response_text = re.sub(r'\.\s*"', '"', response_text)

                # Fix missing opening quotes for field names (handles spaces/hyphens too)
                response_text = _quote_unquoted_keys(response_text)

                # Fix missing closing quotes for UNQUOTED VALUES (but NOT arrays/objects)
                # OLD (broke []): r':\s*([^",}\]]+?)(\s*[,}\]])'
                # NEW: only match when value doesn't start with [ or { (to preserve [] and {})
                response_text = re.sub(
                    r':\s*([^",}\[\{][^",}\]]*?)(\s*[,}\]])',
                    r': "\1"\2',
                    response_text
                )

                # Fix cases like "short_description:"", -> "short_description":"",
                response_text = re.sub(r':\s*"",', r': "",', response_text)

                # Fix double quotes (e.g., ""value"" -> "value")
                response_text = re.sub(r'""([^"]+)""', r'"\1"', response_text)
                response_text = re.sub(r'""+', '"', response_text)

                # Fix empty comma patterns (e.g., ", ," or ",  ,")
                response_text = re.sub(r",\s*,", ",", response_text)

                # Fix field names with trailing spaces (e.g., "field_name " -> "field_name")
                response_text = re.sub(r'"(\w+)\s+":', r'"\1":', response_text)

                # Insert missing commas between objects in arrays (e.g., "} {")
                response_text = re.sub(r"}\s*{", "},{", response_text)

                # Insert missing commas between values and the next key
                response_text = re.sub(
                    r'(?<=[0-9"\]}])\s+(?="[^"]+"\s*:)',
                    ",",
                    response_text,
                )

                # Remove control characters (newlines, tabs, etc.) that break JSON parsing
                response_text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", response_text)

                # Remove JSON comments (// and /* */)
                response_text = re.sub(r"//.*?$", "", response_text, flags=re.MULTILINE)
                response_text = re.sub(r"/\*.*?\*/", "", response_text, flags=re.DOTALL)

                # Remove placeholder text like "Add more models here..."
                response_text = re.sub(
                    r",?\s*//.*?Add more.*?$", "", response_text, flags=re.MULTILINE | re.IGNORECASE
                )

                # Remove trailing commas before } or ]
                response_text = re.sub(r",\s*}", "}", response_text)
                response_text = re.sub(r",\s*]", "]", response_text)

                # Fix leading commas after { or [
                response_text = re.sub(r"{\s*,", "{", response_text)
                response_text = re.sub(r"\[\s*,", "[", response_text)

                # Remove any text after the JSON (like "Note - ...")
                last_brace = response_text.rfind("}")
                if last_brace > 0:
                    response_text = response_text[: last_brace + 1]

                # Balance brackets/braces if the JSON is truncated
                response_text = _balance_json_brackets(response_text)

                if not response_text.strip():
                    logger.error("Response became empty after cleaning")
                    logger.debug(f"Original response: {original_response[:500]}")
                    return None

                # Try to parse JSON after cleaning
                try:
                    parsed = json.loads(response_text)
                    logger.info("JSON repair successful")
                except json.JSONDecodeError as parse_error:
                    logger.error(f"JSON parse failed after repair: {parse_error}")
                    logger.error(f"Response preview (first 500 chars): {original_response[:500]}")
                    logger.error(
                        "Cleaned response (first 500 chars): "
                        f"{response_text[:500] if response_text else 'EMPTY'}"
                    )
                    return None

            # Normalize field names in the parsed JSON
            parsed = self._normalize_field_names(parsed)

            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response length: {len(original_response)}")
            logger.error(f"Response preview (first 500 chars): {original_response[:500]}")
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

    def _coerce_null_strings(self, data: Any) -> Any:
        """Recursively replace string 'null'/'none'/'n/a' with None."""
        if isinstance(data, dict):
            return {k: self._coerce_null_strings(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._coerce_null_strings(v) for v in data]
        if isinstance(data, str) and data.strip().lower() in ("null", "none", "n/a"):
            return None
        return data

    def extract_from_chunks(
        self, text_chunks: List[str], paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MultiModelResponse]:
        """
        Extract from multiple text chunks (for long papers).

        Args:
            text_chunks: List of text chunks
            paper_metadata: Optional metadata

        Returns:
            MultiModelResponse with deduplicated models
        """
        all_models = []

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
            result = self.extract(chunk, paper_metadata)

            if result and result.models:
                all_models.extend(result.models)

        if not all_models:
            logger.warning("No models extracted from any chunk")
            return None

        # Deduplicate models
        unique_models = self._deduplicate_models(all_models)
        logger.info(f"Deduplicated to {len(unique_models)} unique model(s)")

        return MultiModelResponse(
            models=unique_models, paper_describes_multiple_models=len(unique_models) > 1
        )

    def _deduplicate_models(self, models: List[LLMProperties]) -> List[LLMProperties]:
        """
        Deduplicate models: merge variants that refer to the same model
        (same model_family + parameters). Prefer the most specific/canonical
        model_name (e.g. GPT-1 over GPT 117M) and merge non-null fields.
        """
        # Group by (model_family, parameters). Fallback to (model_name, model_version, parameters).
        groups: Dict[tuple, List[LLMProperties]] = {}
        for m in models:
            fam = (m.model_family or "").strip() or ""
            params = (m.parameters or "").strip() or ""
            params_m = m.parameters_millions
            if fam and (params or params_m is not None):
                key = (fam, params or str(params_m) if params_m is not None else "")
            else:
                key = (m.model_name or "", m.model_version or "", m.parameters or "")
            if key not in groups:
                groups[key] = []
            groups[key].append(m)

        result = []
        for group in groups.values():
            if len(group) == 1:
                result.append(group[0])
                continue

            # Pick representative: prefer canonical "Family-N" (e.g. GPT-1) over "Family 117M"
            def _specificity(m: LLMProperties) -> int:
                n = (m.model_name or "").strip().lower()
                fam = (m.model_family or "").strip().lower()
                s = 0
                if fam and fam in n:
                    s += 2
                if fam and f"{fam}-" in n and any(c.isdigit() for c in n.split(f"{fam}-")[-1][:4]):
                    s += 3  # e.g. GPT-1, BERT-Large
                if any(c.isdigit() for c in n) and ("m" in n or "b" in n or "k" in n):
                    s += 1
                return s

            group_sorted = sorted(group, key=_specificity, reverse=True)
            representative = group_sorted[0]
            for other in group_sorted[1:]:
                for field_name, field_value in other.model_dump().items():
                    if field_value is None:
                        continue
                    existing = getattr(representative, field_name)
                    if existing is None or (
                        isinstance(existing, str) and str(existing).strip() in ("", "null", "none")
                    ):
                        setattr(representative, field_name, field_value)
            result.append(representative)

        return result

    def _extract_organization(self, authors: List[str]) -> Optional[str]:
        """
        Extract organization from author list.

        Looks for common organization keywords in author affiliations.
        """
        if not authors:
            return None

        org_keywords = [
            "Meta",
            "Google",
            "OpenAI",
            "Anthropic",
            "Microsoft",
            "DeepMind",
            "AI",
            "Research",
            "University",
            "Institute",
            "Facebook",
            "Amazon",
            "IBM",
            "NVIDIA",
            "Hugging Face",
            "Alibaba",
            "DeepSeek",
            "Mistral",
            "Stability",
        ]

        for author in authors:
            for keyword in org_keywords:
                if keyword.lower() in str(author).lower():
                    return keyword

        return None

    def validate_extraction(self, result: MultiModelResponse) -> Dict[str, Any]:
        """
        Validate extraction results.

        Args:
            result: Extraction result to validate

        Returns:
            Validation report with errors and warnings
        """
        report = {"valid": True, "warnings": [], "errors": []}

        for i, model in enumerate(result.models):
            # Check required fields
            if not model.model_name:
                report["errors"].append(f"Model {i+1}: Missing model_name (required)")
                report["valid"] = False

            # Check field formats
            if model.parameters and not any(
                c in str(model.parameters) for c in ["B", "M", "K", "billion", "million"]
            ):
                report["warnings"].append(
                    f"Model {i+1}: Unusual parameters format: {model.parameters}"
                )

            if model.context_length and not any(c.isdigit() for c in str(model.context_length)):
                report["warnings"].append(
                    f"Model {i+1}: Unusual context_length format: {model.context_length}"
                )

            if model.parameters_millions is not None and model.parameters_millions <= 0:
                report["warnings"].append(
                    f"Model {i+1}: Invalid parameters_millions: {model.parameters_millions}"
                )

        return report
