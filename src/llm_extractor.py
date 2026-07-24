"""
LLM Extractor using KISSKI Chat AI API

This module extracts structured information from LLM research papers
using the KISSKI Chat AI API (SAIA platform).

The KISSKI API is OpenAI-compatible and hosted by GWDG Academic Cloud.
"""

import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """<Role>
  You are an expert AI researcher extracting structured metadata about language models from research papers, following ORKG template R609825.
  </Role>

  <Task-Description>
  A user will give you the text of a research paper (optionally prefixed with metadata and extracted tables). Extract DETAILED information about ALL model versions, sizes, and variants that THIS paper introduces as its own contribution. Ignore models mentioned only as
  related work, baselines, or comparisons. Return one entry per distinct model in a "models" array.
  </Task-Description>

  <Required-Fields>
  Extract these for every model. If a required field is genuinely absent, use null, but prioritise finding it in the text.
  - model_name: Exact name including version/size (e.g. "Llama 3.1 8B", not "Llama").
  - model_family: Family/series (e.g. GPT, BERT, Llama).
  - date_created: Publication date (YYYY-MM-DD, else YYYY-MM, else YYYY).
  - organization: Canonical name (Google, OpenAI, Meta) — not the long form ("Google AI Language").
  - innovation: The model's key innovation(s). For EACH distinct innovation: (a) name the technique in the paper's own terms (e.g. "masked language model", "RLHF"), (b) explain the mechanism, (c) state how it differs from or improves on prior work. Be specific; avoid
  generic phrases. Separate distinct innovations with "; ".
  - pretraining_corpus: Training dataset/corpus.
  - research_problem: Research problem addressed.
  - parameters: Parameter count as text (e.g. "7B", "175B", "117M").
  - parameters_millions: Parameters as an integer in millions (7B -> 7000, 117M -> 117).
  - application: Use cases/applications.
  - license: License type (e.g. "Apache 2.0", "open source", "closed source").
  </Required-Fields>

  <Conditional-Fields>
  Extract these ONLY when the paper explicitly states them. If not mentioned, use null — do NOT guess or infer from other papers or prior knowledge.
  - pretraining_architecture: EXACTLY one of "Encoder", "Decoder", or "Encoder-Decoder".
  - pretraining_task: e.g. "Causal language modeling", "Masked LM", "Next-token prediction".
  - finetuning_task: e.g. "Supervised discriminative fine-tuning".
  - optimizer: Optimizer name only (e.g. Adam, AdamW).
  - extension: One factual sentence describing a mechanism that extends the model beyond a baseline.
  - hardware_used: Training/inference hardware, in the paper's wording (e.g. "Nvidia V100 GPU", "TPUv3").
  - training_corpus_size: Size of the pretraining corpus (e.g. "300B tokens", "570GB").
  - finetuning_data: Dataset(s) used for fine-tuning. May be multiple — separate with commas.
  - tokenizer: Tokenizer name/scheme (e.g. "BPE", "SentencePiece", "tiktoken").
  - supported_language: Language(s) supported. May be multiple — separate with commas (e.g. "English, French").
  - hardware_description: Description of the training hardware setup (e.g. "256 A100 GPUs for 21 days").
  - carbon_emitted: Reported carbon emissions (e.g. "552 tCO2eq").
  - source_code: URL of the source-code repository (return the bare URL only, e.g. a GitHub link).
  - activated_parameters: [MoE models] Number of ACTIVE parameters per token, distinct from total parameters (e.g. "32B activated").
  - moe_configuration: [MoE models] Expert setup (e.g. "1T total parameters, 32B activated"; "384 experts, 8 active, 1 shared").
  - attention_mechanism: Attention variant (e.g. "Multi-head Latent Attention (MLA)", "Grouped-Query Attention (GQA)", "sparse attention / DSA").
  - number_of_attention_heads: Attention head count (e.g. "64 query / 8 key-value").
  - context_length_max: Maximum supported context window (e.g. "128K tokens", "1M tokens").
  - context_extension_method: Technique used to extend context (e.g. "YaRN", "RoPE scaling", "DSA").
  - base_model: Pre-existing model this one is built upon/initialized from (e.g. "Qwen2.5", "Phi-4").
  - training_pipeline: Multi-stage training recipe (e.g. "pre-training → SFT → RL", "multi-stage post-training").
  - optimizer_innovation: A novel optimizer or optimizer modification introduced (e.g. "MuonClip", "Muon with QK-clip").
  - weight_clipping_mechanism: Weight/gradient/activation clipping technique for stability (e.g. "QK-clip", "gradient clipping").
  - quantization_precision: Numerical precision for training/inference (e.g. "FP8", "BF16", "INT4").
  - synthetic_data_generation_method: How synthetic training data is produced (e.g. "large-scale agentic data synthesis", "rejection sampling", "synthetic augmentation").
  - rl_algorithm: Reinforcement-learning algorithm used in post-training (e.g. "GRPO", "PPO", "asynchronous agent RL").
  - reward_mechanism: How RL rewards are defined (e.g. "verifiable rule-based rewards", "reward model").
  - reasoning_mode: [Reasoning models] Whether/how the model reasons (e.g. "hybrid thinking/non-thinking with explicit mode tokens", "non-thinking", "chain-of-thought").
  - tool_calling_format: Format/protocol for tool or function calling (e.g. "JSON function calling", "ReAct").
  - training_environment_scale: Scale of the (RL) training environments (e.g. "real and synthetic environments", "tens of thousands of environments"). 
  - post_training_infrastructure: System/infrastructure used for post-training (e.g. "asynchronous RL infrastructure decoupling generation from training").
  - training_environment_scale is about environments; post_training_infrastructure is about the training system — keep them distinct.
  - benchmark_result: Reported benchmark scores, verbatim, separated by "; " (e.g. "SWE-Bench Verified: 65.8; AIME 2025: 49.5"). Only real numbers stated in the paper — never invent scores.
  - safety_evaluation_protocol: Safety evaluation method/protocol (e.g. "red-teaming", named safety benchmark).
  - safety_defect_rate: Reported safety defect/failure rate (e.g. "0.3%").
  - fusion_architecture: [Multimodal models] How modalities are fused (e.g. "early fusion", "cross-attention fusion").
  - vision_encoder: [Multimodal models] Vision encoder/backbone (e.g. "ViT", "dynamic-resolution encoder", "CLIP ViT-L").
  </Conditional-Fields>

  <Critical-Rules>
  1. TITLE: Extract the official, full RESEARCH PAPER TITLE and assign it to 'paper_title'.
  2. ALL VARIANTS: Extract ALL model versions, sizes, and variants as SEPARATE entries.
  3. PARAMETERS: Search for 'Our model' or 'Proposed'. Look for 'M' or 'B'. Extract parameter sizes for each variant. Calculate parameters_millions (e.g., 7B = 7000, 117M = 117).
  4. DATES: Prefer YYYY-MM (e.g. 2018-10). Use YYYY-MM-DD when day is known, else YYYY-MM, else YYYY. Priority: metadata > header/footer > citation year.
  5. ORGANIZATION: Use canonical name (e.g. Google, OpenAI, Meta) not long form (e.g. not "Google AI Language").
  6. PARAMETERS: For multiple sizes use comma-separated (e.g. "110M, 340M").
  7. MULTIPLE MODELS: Set 'paper_describes_multiple_models' to true if the paper describes multiple distinct models, versions, or size variants.
  8. REQUIRED FIELDS: You MUST extract all required fields. If a field is not mentioned in the paper, use null, but prioritize extracting from paper text.
  9. TABLES: If the paper includes a [TABLES FROM DOCUMENT] block, the content is markdown tables from the PDF. Use these tables as the primary source for model names, metrics (e.g. F1, BERTScore), parameter counts, and dataset names; prefer exact values from table cells.
  10. CONTEXT VARIANTS: Do NOT create separate entries for context-window variants of the same model (e.g. 'Llama 3 8K' and 'Llama 3 128K-context' are the SAME model as 'Llama 3'). Record the context length in the context_length field of that single entry instead.
  11. STAGE VARIANTS: Do NOT create separate entries for pre-trained vs post-trained (instruction-tuned) variants of the same model (e.g. 'Llama 3 (pre-trained)' and 'Llama 3 (post-trained)' are ONE entry 'Llama 3'). Mention both stages in the innovation or finetuning_task fields.

  Additional rules:
  - Extract ALL model versions mentioned (3.1, 3.2, 3.3 = separate entries).
  - Extract ALL architectural variants (Base, Large, XL, etc. = separate entries).
  - Extract models THIS paper introduces (main contributions), NOT models mentioned as related work or comparisons.
  - Focus on PRIMARY model contributions intended as standalone released models.
  - Do NOT create separate entries for auxiliary artifacts such as tools, guards, safety filters, adapters, encoders, tokenizers, pipelines, or infrastructure modules when the paper also contains main model contributions.
  - If auxiliary artifacts are mentioned, capture them inside innovation/extension fields of the relevant primary model instead of as standalone models.
  - Model name should include version/size if mentioned (e.g. "Llama 3.1 8B" not just "Llama"); model name is NOT the architecture (e.g. "GPT" not "Transformer").
  - parameters_millions: "7B"->7000, "117M"->117, "1.5B"->1500.
  </Critical-Rules>

  <Default>
  For required fields, use null only after genuinely failing to find the value. For conditional fields, always prefer null over guessing when the paper does not state the value.
  </Default>

  <Response-Format>
  Return JSON only, no prose:
  {
    "models": [ { "model_name": "...", "...": "..." } ],
    "paper_describes_multiple_models": true | false
  }
  </Response-Format>"""


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
    activated_parameters: Optional[str] = None
    attention_mechanism: Optional[str] = None
    context_length_max: Optional[str] = None
    context_extension_method: Optional[str] = None
    training_pipeline: Optional[str] = None
    reasoning_mode: Optional[str] = None
    moe_configuration: Optional[str] = None
    quantization_precision: Optional[str] = None
    synthetic_data_generation_method: Optional[str] = None
    rl_algorithm: Optional[str] = None
    reward_mechanism: Optional[str] = None
    tool_calling_format: Optional[str] = None
    training_environment_scale: Optional[str] = None
    safety_evaluation_protocol: Optional[str] = None
    safety_defect_rate: Optional[str] = None
    fusion_architecture: Optional[str] = None
    vision_encoder: Optional[str] = None
    base_model: Optional[str] = None
    optimizer_innovation: Optional[str] = None
    benchmark_result: Optional[str] = None
    weight_clipping_mechanism: Optional[str] = None
    number_of_attention_heads: Optional[str] = None
    post_training_infrastructure: Optional[str] = None 

    @model_validator(mode="before")
    @classmethod
    def _normalize_list_values(cls, data: Any) -> Any:
        """
        Coerce multi-valued shapes the LLM emits for scalar fields.

        Papers describing several model sizes return parameters_millions as a
        list ([8000, 70000]) or a comma-string ('8000, 70000, 405000').
        - Integer fields collapse to their max (predicate P110076 is
          "max params in million").
        - Other scalar fields that arrive as lists are joined into a comma
          string, which the ORKG template mapper later splits into one row each.
        """
        if not isinstance(data, dict):
            return data

        int_fields = {"parameters_millions"}
        for field, value in list(data.items()):
            if field in int_fields:
                nums = cls._to_int_list(value)
                if nums is not None:
                    data[field] = max(nums) if nums else None
            elif isinstance(value, list):
                parts = [str(v).strip() for v in value if v is not None and str(v).strip()]
                data[field] = ", ".join(parts) if parts else None
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # The model sometimes returns a bare number for a string-typed
                # field (e.g. number_of_attention_heads: 64, context_length:
                # 128000). Coerce to str so one numeric value doesn't fail
                # validation and discard the entire extraction.
                data[field] = str(value)
        return data

    @staticmethod
    def _to_int_list(value: Any) -> Optional[List[int]]:
        """Parse an int field that may arrive as an int, list, or comma string."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return [int(value)]
        if isinstance(value, str):
            items: List[Any] = value.split(",")
        elif isinstance(value, list):
            items = value
        else:
            return None
        nums: List[int] = []
        for v in items:
            try:
                nums.append(int(float(str(v).strip())))
            except (TypeError, ValueError):
                continue
        return nums

class MultiModelResponse(BaseModel):
    """Response containing multiple extracted models."""

    models: List[LLMProperties]
    paper_describes_multiple_models: bool = False


class LLMExtractor:
    """Extracts LLM information using KISSKI Chat AI API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://chat-ai.academiccloud.de/v1",
        model: str = "meta-llama-3.1-8b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        timeout: int = 180,
        rate_limit_delay: float = 2.0,
        retry_attempts: int = 5,
        retry_delay: float = 3.0,
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
            timeout: Base request timeout in seconds (escalates on retries)
            rate_limit_delay: Delay between requests in seconds (default: 2.0)
            retry_attempts: Max retries per API call on transient errors
            retry_delay: Base delay between retries in seconds (exponential backoff applied)

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
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.last_request_time = 0

        # Disable the OpenAI client's built-in retries — we handle retries
        # ourselves with escalating timeouts and exponential backoff.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

        logger.info(f"Initialized KISSKI extractor with model: {model}")
        logger.info(f"API endpoint: {base_url}")
        logger.info(f"Timeout: {timeout}s, retries: {retry_attempts}, backoff base: {retry_delay}s")

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
        example1_input = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Google AI Language. We introduce BERT with 110M, 340M parameters. It uses a Transformer encoder architecture trained on Masked LM and Next Sentence Prediction tasks. It achieves state-of-the-art on GLUE. We use Adam optimizer. Trained on English Wikipedia and BookCorpus, totaling 3.3 billion words. We use a WordPiece tokenizer and train on 16 Cloud TPUs (64 TPU chips). The model supports English. Code is available at https://github.com/google-research/bert."  # noqa: E501
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
                    "tokenizer": "WordPiece (Wu et al., 2016), 30,000 token vocabulary; special tokens [CLS], [SEP], [MASK]",
                    "hardware_used": "Cloud TPU",
                    "hardware_description": "BERT_BASE: 4 Cloud TPUs in Pod configuration (16 TPU chips); BERT_LARGE: 16 Cloud TPUs (64 TPU chips); each pre-training run took 4 days. Fine-tuning replicable in ≤1 hour on a single Cloud TPU.",
                    "innovation": "Masked language model (MLM): randomly masks input tokens and trains the model to predict them, enabling deep bidirectional context rather than left-to-right conditioning; Next Sentence Prediction (NSP): jointly pre-trains on sentence-pair coherence so the model captures inter-sentence relationships for tasks like QA and NLI.",
                    "research_problem": "Language Understanding",
                    "application": "Natural language understanding, question answering, text classification",  # noqa: E501
                    "license": "Apache 2.0",
                    "training_corpus_size": "3.3 billion words",
                    "tokenizer": "WordPiece",
                    "hardware_used": "Cloud TPU",
                    "hardware_description": "16 Cloud TPUs (64 TPU chips)",
                    "supported_language": "English",
                    "finetuning_data": "GLUE datasets (MNLI 392k, QQP 363k, QNLI 108k, SST-2 67k, CoLA 8.5k, STS-B 5.7k, MRPC 3.5k, RTE 2.5k), SQuAD v1.1 (100k QA pairs), SQuAD v2.0, SWAG (113k), CoNLL-2003 NER; optional TriviaQA augmentation for SQuAD",
                    "context_length": "512",
                    "activated_parameters": "110M (BASE) / 340M (LARGE) — dense, all parameters active",
                    "attention_mechanism": "Bidirectional (unmasked) multi-head self-attention; A=12 (BASE), A=16 (LARGE)",
                    "training_pipeline": "Two-stage: unsupervised pre-training (MLM + NSP) followed by supervised end-to-end fine-tuning per downstream task; alternatively feature-based extraction of frozen activations",
                    "training_environment_scale": "Batch size 256 sequences (128,000 tokens/batch) for 1,000,000 steps ≈ 40 epochs over 3.3B-word corpus",
                    "benchmark_result": "GLUE average 82.1 (LARGE) / 79.6 (BASE), official leaderboard score 80.5 (+7.7 abs); MNLI-m/mm 86.7/85.9; QQP 72.1; QNLI 92.7; SST-2 94.9; CoLA 60.5; STS-B 86.5; MRPC 89.3; RTE 70.1; SQuAD v1.1 Test EM/F1 87.4/93.2 (ensemble+TriviaQA), single 85.1/91.8; SQuAD v2.0 Test EM/F1 80.0/83.1; SWAG Test 86.3; CoNLL-2003 NER Test F1 92.8",
                    "number_of_attention_heads": "12 (BERT_BASE), 16 (BERT_LARGE)",
                    "source_code": "https://github.com/google-research/bert",
                }
            ]
        }

        # Example 2: GPT-2 (with all ORKG R609825 required fields)
        example2_input = "Language Models are Unsupervised Multitask Learners. OpenAI. We trained a 1.5 billion parameter Transformer decoder language model. It demonstrates zero-shot task transfer. We assume a causal language modeling objective. Trained on the WebText dataset, 40GB of English text. The model uses a byte-level BPE tokenizer. Code: https://github.com/openai/gpt-2."  # noqa: E501
        example2_output = {
            "models": [
                {
                    "model_name": "GPT-2",
                    "model_family": "GPT",
                    "paper_title": "Language Models are Unsupervised Multitask Learners",
                    "organization": "OpenAI",
                    "base_model": "OpenAI GPT (Radford et al., 2018) architecture",
                    "parameters": "1.5B",
                    "parameters_millions": 1500,
                    "date_created": "2019-02",
                    "pretraining_architecture": "Decoder",
                    "pretraining_task": "Causal language modeling",
                    "pretraining_corpus": "WebText",
                    "innovation": "Demonstrates that a large causal-LM Transformer trained on diverse web text (WebText) can perform many NLP tasks zero-shot — without task-specific training data or fine-tuning — by conditioning on natural-language task prompts, in contrast to prior approaches that require supervised fine-tuning per task.",
                    "research_problem": "Large Language Models",
                    "application": "Text generation, language modeling, zero-shot task transfer",  # noqa: E501
                    "license": "Modified MIT License",
                    "training_corpus_size": "40GB of text",
                    "tokenizer": "Byte-level BPE",
                    "supported_language": "English",
                    "context_length": "1024",
                    "supported_language": "English (primarily; non-English pages filtered, ~10MB French detected)",
                    "activated_parameters": "1,542,000,000 (dense, all parameters active)",
                    "attention_mechanism": "Masked (causal) multi-head self-attention",
                    "training_pipeline": "Single-stage unsupervised pretraining only; no supervised fine-tuning or RLHF",
                    "optimizer_innovation": "Residual layer weights scaled at initialization by 1/sqrt(N), N = number of residual layers",
                    "benchmark_result": "SOTA on 7 of 8 LM datasets zero-shot: LAMBADA 8.63 PPL / 63.24 percent acc, CBT-CN 93.30%, CBT-NE 89.05%, WikiText2 18.34 PPL, PTB 35.76 PPL, enwik8 0.93 BPB, text8 0.98 BPC, WikiText103 17.48 PPL, 1BW 42.16 PPL (not SOTA); CoQA 55 F1; Winograd 70.70%; CNN/DM ROUGE-AVG 21.40; WMT-14 En-Fr 5 BLEU, Fr-En 11.5 BLEU; Natural Questions 4.1 percent exact match",
                    "source_code": "https://github.com/openai/gpt-2",
                }
            ]
        }

        # Example 3: GPT-1 (with all ORKG R609825 required fields)
        example3_input = "Improving Language Understanding by Generative Pre-Training. Alec Radford, OpenAI. We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. Our approach employs a Transformer-based architecture with 117M parameters. We use the Adam optimizer. Trained on BooksCorpus dataset using a bytepair encoding (BPE) tokenizer, then fine-tuned on downstream datasets including SNLI and MultiNLI. Code available at https://github.com/openai/finetune-transformer-lm."  # noqa: E501
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
                    "innovation": "Introduces a two-stage semi-supervised recipe: generative pre-training of a Transformer language model on a large unlabeled corpus (BooksCorpus), followed by discriminative fine-tuning on each downstream task via task-aware input transformations, ing one pre-trained model transfer across diverse tasks with minimal architecture changes rather than training task-specific models from scratch.",
                    "license": "closed source",
                    "research_problem": "Language Understanding",
                    "application": "Natural language understanding, text classification, question answering",  # noqa: E501
                    "tokenizer": "Bytepair encoding (BPE)",
                    "finetuning_data": "SNLI, MultiNLI",
                    "context_length": "512",
                    "supported_language": "English",
                    "activated_parameters": "~117M (dense, all parameters active)",
                    "attention_mechanism": "Masked (causal) multi-head self-attention, 12 heads",
                    "training_pipeline": "Two-stage: unsupervised generative pre-training (LM objective) → supervised discriminative fine-tuning with auxiliary LM objective (L3 = L2 + λ·L1, λ=0.5); task-specific input transformations instead of task-specific architectures",
                    "training_environment_scale": "100 epochs on minibatches of 64 randomly sampled contiguous sequences of 512 tokens; fine-tuning batch size 32, lr 6.25e-5, 3 epochs, linear decay with warmup over 0.2 percent of training",
                    "optimizer_innovation": "Decoupled/modified L2 regularization (Loshchilov & Hutter) with w=0.01 on non-bias/gain weights; cosine-annealed LR with linear warmup",
                    "benchmark_result": "SOTA on 9 of 12 datasets. GLUE 72.8 (prev. best 68.9); MNLI-m/mm 82.1/81.4; SNLI 89.9; SciTail 88.3; QNLI 88.1; RTE 56.0; Story Cloze 86.5 (+8.9); RACE 59.0 (RACE-m 62.9, RACE-h 57.4, +5.7); CoLA 45.4 mc (prev. 35.0); SST-2 91.3; MRPC 82.3 F1; STS-B 82.0 pc; QQP 70.3 F1. Ablations: w/o pre-training avg 59.9 vs 74.7 full; LSTM w/ aux LM 69.1",
                    "number_of_attention_heads": "12",
                    "source_code": "https://github.com/openai/finetune-transformer-lm",
                }
            ]
        }

        # Example 4: Multiple model versions (Llama 3.1 - all ORKG R609825 required fields)
        example4_input = "The Llama 3.1 Herd of Models. Meta AI. We introduce Llama 3.1 with three model sizes: 8B, 70B, and 405B parameters. All models use Transformer decoder architecture. The 8B model has 8 billion parameters, the 70B model has 70 billion parameters, and the 405B model has 405 billion parameters. All models are trained on the same pretraining task. Trained on large-scale text corpus. Applications include chat, instruction following, and general language tasks. All models were pretrained on approximately 15 trillion tokens with a 128K-vocabulary BPE tokenizer, on up to 16,000 NVIDIA H100 GPUs, emitting an estimated 11,390 tCO2eq. The models support English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai. Released under Llama 3.1 Community License."  # noqa: E501
        example4_output = {
            "models": [
                {
                    "model_name": "Llama 3.1 8B",
                    "model_family": "Llama",
                    "paper_title": "The Llama 3.1 Herd of Models",
                    "organization": "Meta",
                    "parameters": "8B",
                    "parameters_millions": 8000,
                    "activated_parameters": "8B (dense — all parameters active)",
                    "attention_mechanism": "Grouped Query Attention (GQA) with 8 key-value heads; cross-document attention masking",
                    "date_created": "2024-07",
                    "context_length_max": "128K tokens",
                    "training_corpus_size": "15 trillion tokens",
                    "optimizer": "AdamW",
                    "tokenizer": "128K-vocabulary BPE",
                    "training_pipeline": "Pre-training → long-context pre-training → annealing → post-training (6 rounds of RM → rejection sampling → SFT → DPO)",
                    "hardware_used": "NVIDIA H100 GPU",
                    "hardware_description": "Up to 16,000 H100 GPUs",
                    "reasoning_mode": "Chain-of-thought prompting; interleaved code+text reasoning with execution feedback",
                    "moe_configuration": "None — dense architecture chosen over MoE for training stability",
                    "quantization_precison": "BF16 training; int8 quantized Llama Guard 3 variant (>40 percent size reduction)",
                    "rl_algorithm": "Direct Preference Optimization (DPO); PPO explored but rejected",
                    "reward_mechanism": "Reward model on pre-trained checkpoint, Llama 2 objective minus margin term; edited > chosen > rejected rankings; outcome + step-wise reward models",
                    "tool_calling_format": "Python objects/functions with signature+docstring; JSON conversion for web APIs; multi-message chat protocol with header/termination tokens",
                    "carbon_emitted": "11,390 tCO2eq",
                    "supported_language": "English, German, French, Italian, Portuguese, Hindi, Spanish, Thai",
                    "pretraining_architecture": "Transformer",
                    "pretraining_task": "Next-token prediction",
                    "pretraining_corpus": "Large-scale text corpus",
                    "finetuning_data": "Human-annotated preference data (81.99 percent general English, 6.93 percent coding, 5.19% multilingual, 5.89 percent reasoning/tools); SFT mix of rejection-sampled, synthetic, and human-curated data",
                    "innovation": "A family of decoder-only Transformer language models sharing one architecture and next-token-prediction training recipe, scaled across 8B, 70B, and 405B parameters on a large-scale text corpus to support chat and instruction following.",
                    "research_problem": "Large Language Models",
                    "application": "Chat, instruction following, general language tasks",
                    "supported_language": "English, German, French, Italian, Portuguese, Hindi, Spanish, Thai",
                    "safety_evaluation_protocol": "Violation Rate / False Refusal Rate on >4,000 prompts per capability; red teaming; CyberSecEval; ML Commons hazard taxonomy",
                    "benchmark_results": "MMLU 69.4 (5-shot); MMLU 0-shot CoT 73.0; HumanEval 72.6; GSM8K 84.5; MATH 51.9; IFEval 80.4; MGSM 68.9; BFCL 76.1; GPQA 32.8; ARC-C 83.4",
                    "number_of_attention_heads": "32",
                    "post_training_infrastructure": "PagedAttention for rejection sampling (>2x throughput); 6 iterative post-training rounds",
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
                    "training_corpus_size": "15 trillion tokens",
                    "tokenizer": "128K-vocabulary BPE",
                    "hardware_used": "NVIDIA H100 GPU",
                    "hardware_description": "Up to 16,000 H100 GPUs",
                    "carbon_emitted": "11,390 tCO2eq",
                    "supported_language": "English, German, French, Italian, Portuguese, Hindi, Spanish, Thai",
                    "pretraining_architecture": "Transformer",
                    "pretraining_task": "Next-token prediction",
                    "pretraining_corpus": "Large-scale text corpus",
                    "innovation": "A family of decoder-only Transformer language models sharing one architecture and next-token-prediction training recipe, scaled across 8B, 70B, and 405B parameters on a large-scale text corpus to support chat and instruction following.",
                    "research_problem": "Large Language Models",
                    "application": "Chat, instruction following, general language tasks",
                    "license": "Llama 3.1 Community License",
                    "finetuning_data": "Same preference and SFT mixes as the herd; capability-specific expert models (code expert trained on 1T token mix of >85p code; multilingual expert trained on 90% multilingual tokens)",
                    "context_length": "128000",
                    "activated_parameters": "70B (dense, all parameters active)",
                    "attention_mechanism": "Grouped Query Attention (GQA) with 8 key-value heads",
                    "training_pipeline": "Pre-training -> long-context pre-training -> annealing -> 6 rounds of reward modeling, SFT and DPO with model averaging",
                    "reasoning_mode": "Chain-of-thought; code-interleaved reasoning; self-verification of reasoning traces",
                    "moe_configuration": "None (dense architecture)",
                    "quantization_precision": "BF16",
                    "rl_algorithm": "Direct Preference Optimization (DPO), learning rate 1e-5, beta = 0.1",
                    "reward_mechanism": "Reward model trained on human preference pairs with a third edited response giving edited > chosen > rejected rankings",
                    "tool_calling_format": "Brave Search, Python interpreter and Wolfram Alpha as core tools; zero-shot function calling including nested and parallel calls",
                    "training_environment_scale": "Up to 16K H100 GPUs",
                    "safety_evaluation_protocol": "Violation Rate / False Refusal Rate benchmarks, Llama Guard 3 system-level safety, red teaming, CyberSecEval 2",
                    "safety_defect_rate": "Verbatim memorization 0.60% (English, 50-gram), 0.55% (all, 50-gram), 3.56% (all, 1000-gram); code interpreter abuse compliance 3.8%; spear-phishing attempts judged successful 24p of the time",
                    "benchmark_result": "MMLU 83.6; MMLU 0-shot CoT 86.0; MMLU-Pro 66.4; IFEval 87.5; HumanEval 80.5; MBPP EvalPlus 86.0; GSM8K 95.1; MATH 68.0; ARC-Challenge 94.8; GPQA 46.7; BFCL 84.8; Nexus 56.7; MGSM 86.9; Multilingual MMLU 78.2",
                    "number_of_attention_heads": "64",
                    "post_training_infrastructure": "PagedAttention-accelerated rejection sampling; 6 iterative post-training rounds"
                },
                {
                    "model_name": "Llama 3.1 405B",
                    "model_family": "Llama",
                    "paper_title": "The Llama 3.1 Herd of Models",
                    "organization": "Meta",
                    "parameters": "405B",
                    "parameters_millions": 405000,
                    "date_created": "2024-07",
                    "training_corpus_size": "15 trillion tokens",
                    "tokenizer": "128K-vocabulary BPE",
                    "hardware_used": "NVIDIA H100 GPU",
                    "hardware_description": "Up to 16,000 H100 GPUs",
                    "carbon_emitted": "11,390 tCO2eq",
                    "supported_language": "English, German, French, Italian, Portuguese, Hindi, Spanish, Thai",
                    "pretraining_architecture": "Transformer",
                    "pretraining_task": "Next-token prediction",
                    "pretraining_corpus": "Large-scale text corpus",
                    "innovation": "A family of decoder-only Transformer language models sharing one architecture and next-token-prediction training recipe, scaled across 8B, 70B, and 405B parameters on a large-scale text corpus to support chat and instruction following.",
                    "research_problem": "Large Language Models",
                    "application": "Chat, instruction following, general language tasks",
                    "finetuning_data": "Preference data (Table 6) plus SFT mix (Table 7: 52.66p general English, 21.19p reasoning and tools, 14.89p code, 8.14p exam-like, 3.01% multilingual, 0.11p long context); over 2.7M synthetic coding examples",
                    "context_length": "128000",
                    "activated_parameters": "405B (dense, all parameters active)",
                    "attention_mechanism": "Grouped Query Attention (GQA) with 8 key-value heads; document-boundary attention masking, important for continued pre-training on very long sequences",
                    "training_pipeline": "Initial pre-training (1,200,000 steps, batch size ramping 4M -> 8M -> 16M tokens) -> long-context pre-training -> annealing with Polyak averaging -> 6 post-training rounds of reward modeling, rejection sampling, SFT, DPO and model averaging",
                    "reasoning_mode": "Chain-of-thought; step-wise reasoning traces with self-verification; Monte Carlo Tree Search with learned step-wise reward models; interleaved code and text execution",
                    "quantization_precision": "BF16 training; FP8 row-wise inference quantization applied to feedforward network layers (roughly 50p of inference compute), excluding the first and last Transformer layers and self-attention layers; dynamic scaling factors capped at 1200",
                    "rl_algorithm": "Direct Preference Optimization (DPO)",
                    "reward_mechanism": "Reward model on the pre-trained checkpoint with the margin term removed; multiple responses concatenated per row with random shuffling; edited > chosen > rejected rankings; outcome and step-wise reward models used to filter math reasoning traces",
                    "tool_calling_format": "Core tools as Python objects (Brave Search, Python interpreter, Wolfram Alpha API); JSON format for web API calls; zero-shot function calling from signature and docstring; single, nested, parallel and multi-turn calls",
                    "safety_evaluation_protocol": "Violation Rate and False Refusal Rate internal benchmarks (over 4,000 prompts per capability or language), DocQA and Many-shot long-context safety benchmarks, red teaming including PAIR-style multi-turn automation, CyberSecEval, and CBRNE plus cyber uplift studies with 62 internal volunteers",
                    "benchmark_result": "MMLU 87.3; MMLU 0-shot CoT 88.6; MMLU-Pro 73.3; IFEval 88.6; HumanEval 89.0; MBPP EvalPlus 88.6; GSM8K 96.8; MATH 73.8; ARC-Challenge 96.9; GPQA 51.1; BFCL 88.5; Nexus 58.7; MGSM 91.6; Multilingual MMLU 83.2; ZeroSCROLLS/QuALITY 95.2; InfiniteBench En.MC 83.4; NIH/Multi-needle 98.1",
                    "number_of_attention_heads": "128",
                    "post_training_infrastructure": "PagedAttention for rejection sampling (over 2x throughput); inference pipeline parallelism with micro-batching and FP8 quantization (up to 50% prefill throughput improvement); 6 iterative post-training rounds",
                    "license": "Llama 3.1 Community License",
                },
            ],
            "paper_describes_multiple_models": True,
        }
        # Example 5: Kimi K2 (MoE, optimizer innovation, agentic RL, non-thinking)
        example5_input = "Kimi K2: Open Agentic Intelligence. Moonshot AI. We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments. Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, obtaining 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, 47.3 on SWE-Bench Multilingual, 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond and 27.1 on OJBench, all without extended thinking.We release our base and post-trained model checkpoints."  
        example5_output = {
            "models": [
                {
                      "model_name": "Kimi K2",
                      "model_family": "Kimi",
                      "paper_title": "Kimi K2: Open Agentic Intelligence",
                      "organization": "Moonshot AI",
                      "date_created": "2026-02",
                      "base_model": "Kimi-K2-Base (trained from scratch); builds on K1.5 methodology",
                      "pretraining_architecture": "Ultra-sparse Mixture-of-Experts transformer with Multi-head Latent Attention (MLA), 61 layers, hidden dim 716 MoE expert hidden dim 2048",
                      "pretraining_task": "Autoregressive language modeling (next-token prediction)",
                      "pretraining_corpus": "15.5T high-quality tokens spanning Web Text, Code, Mathematics, Knowledge; rephrasing-based synthetic augmentation",
                      "parameters": "1.04T",
                      "parameters_millions": 1040000,
                      "activated_parameters": "32B",
                      "number_of_attention_heads": "64",
                      "supported_language": "English, Chinese (bilingual; multilingual coding)",
                      "hardware_used": "NVIDIA H800 GPUs",
                      "hardware_description": "H800 cluster; nodes with 2TB RAM + 8 GPUs via NVLink/NVSwitch; 8x400 Gbps RoCE inter-node; node counts in multiples of 32",
                      "moe_configuration": "1T total parameters, 32B activated",
                      "post_training_infrastructure": "Hybrid colocated train/inference architecture; distributed checkpoint engine (<30s full 1T param update); Gym-like RL framework; partial rollout for long-horizon agentic tasks",
                      "optimizer": "MuonClip",
                      "optimizer_innovation": "MuonClip — improves the Muon optimizer with a novel QK-clip technique to address training instability while retaining Muon's token efficiency", 
                      "weight_clipping_mechanism": "QK-clip",
                      "training_corpus_size": "15.5 trillion tokens",
                      "training_pipeline": "Pre-training with MuonClip, then multi-stage post-training: large-scale agentic data synthesis followed by a joint reinforcement learning (RL) stage",  
                      "synthetic_data_generation_method": "Chunk-wise autoregressive rephrasing (knowledge), learning-note rewriting (math), three-stage agentic pipeline: tool spec, agent/task, trajectory generation with LLM judge filtering",
                      "rl_algorithm": "Joint reinforcement learning (RL)",
                      "attention_mechanism": "Multi-head Latent Attention (MLA)",
                      "context_length_max": "128K",
                      "context_extension_method": "YaRN",
                      "training_pipeline": "Pretraining (WSD schedule, 15.5T); annealing; long-context activation; SFT; joint RL (RLVR + self-critique rubric reward)",
                      "tool_calling_format": "Custom token template: tool_declare / tool_call_section with TypeScript (and JSON) tool declarations; constrained decoding enforcer",
                      "safety_evaluation_protocol": "Promptfoo automated red-teaming (Harmful, Criminal, Misinformation, Privacy, Security plugins x Basic, Base64, Prompt Injection, Iterative Jailbreak, Crescendo) with human review",
                      "training_environment_scale": "Kubernetes sandbox infrastructure supporting 10,000+ concurrent instances",
                      "reasoning_mode": "Non-thinking (no extended thinking)",
                      "reward_mechanism": "Verifiable rewards (RLVR) + Self-Critique Rubric Reward with core/prescriptive/human-annotated rubrics; closed-loop critic refinement",
                      "quantization_precision": "BF16 params, FP32 gradient buffers; FP8-E4M3 storage for MoE up-projection/SwiGLU activations (not compute)",
                      "benchmark_result": "Tau2-Bench: 66.1; ACEBench (En): 76.5; SWE-Bench Verified: 65.8; SWE-Bench Multilingual: 47.3; LiveCodeBench v6: 53.7; AIME 2025: 49.5; GPQA-Diamond: 75.1; OJBench: 27.1",  
                      "innovation": "MuonClip optimizer (Muon + QK-clip) enables stable, token-efficient pre-training of a 1T-parameter MoE model on 15.5T tokens with zero loss spikes; a large-scale agentic data-synthesis pipeline plus joint RL in real and synthetic environments produce strong agentic and software-engineering ability in a non-thinking model.", 
                      "research_problem": "Open agentic intelligence with large language models",
                      "application": "Agentic tasks, software engineering, coding, mathematics, reasoning",
                      "license": "open source",
                      "source_code": "https://huggingface.co/moonshotai/Kimi-K2-Instruct"
                }
              ]
          }
        # Example 6: Phi-4-reasoning-vision-15B (multimodal, vision encoder, hybrid reasoning mode)
        example6_input = "Phi-4-reasoning-vision-15B. Microsoft. We present Phi-4-reasoning-vision-15B, a compact open-weight multimodal reasoning model that is good at common vision and language tasks and excels at scientific and mathematical reasoning and understanding user interfaces. Careful architecture choices and rigorous data curation enable smaller, open-weight multimodal models to achieve competitive performance with significantly less training and inference-time compute. The most substantial improvements come from systematic filtering, error correction, and synthetic augmentation. Systematic ablations show that high-resolution, dynamic-resolution encoders yield consistent improvements, as accurate perception is a prerequisite for high-quality reasoning. Finally, a hybrid mix of reasoning and non-reasoning data with explicit mode tokens allows a single model to deliver fast direct answers for simpler tasks and chain-of-thought reasoning for complex problems." 
        example6_output = {
            "models": [
                {
                      "model_name": "Phi-4-reasoning-vision-15B",
                      "model_family": "Phi",
                      "paper_title": "Phi-4-reasoning-vision-15B Technical Report",
                      "base_model": "Phi-4-Reasoning (itself built on Phi-4)",
                      "organization": "Microsoft",
                      "date_created": "2026-03",
                      "parameters": "15B",
                      "parameters_millions": 15000,
                      "pretraining_architecture": "Mid-fusion VLM: SigLIP-2 vision encoder + MLP cross-modality projector + Phi-4-Reasoning LLM backbone",
                      "pretraining_task": "Stage 1: image-text alignment (MLP only, frozen encoder/LLM)",
                      "pretraining_corpus": "200B tokens of multimodal data; backbone Phi-4-Reasoning (16B tokens) on Phi-4 (400B unique tokens); Bunny for Stage 1 alignment",
                      "finetuning_data": "Stage 2: 62.8M samples / 188.5B tokens; Stage 3: 3.2M samples / 12B tokens; sources incl. LLaVA-OneVision, Pixmo, Docmatix, CoSyn, NuminaMath, AGUVis, PhiGround, SeeClick, Open Images, WildGuard, VLGuard",
                      "finetuning_task": "Single-image visual instruction tuning (VQA, math/science reasoning, grounding, captioning, OCR, computer-use); Stage 3 long-context, multi-image, RAI",
                      "optimizer": "AdamW",
                      "context_length": "2048 (Stage 1), 8192 (Stage 2), 16384 (Stage 3)",
                      "supported_language": "English",
                      "activated_parameters": "15B (dense, all active)",
                      "attention_mechanism": "Standard dense transformer attention (Phi-4-Reasoning backbone); vision via SigLIP-2 NaFlex dynamic resolution",
                      "context_length_max": "16,384 (training); 4096 max output tokens at eval",
                      "training_pipeline": "3 stages: (1) MLP pretraining, (2) full-model instruction tuning, (3) long-context + multi-image + RAI; SFT only (no RL)",
                      "hardware_used": "NVIDIA H100 GPUs (used for timing/eval)",
                      "hardware_description": "H100 GPUs, single thread, no concurrency, batch size 1 for latency measurement (Eureka ML Insights)",
                      "training_corpus_size": "~201.9B training tokens total (1.4B + 188.5B + 12B); 68M samples",
                      "vision_encoder": "High-resolution, dynamic-resolution vision encoder",
                      "fusion_architecture": "Mid-fusion",
                      "moe_configuration": "Dense model (no MoE); all 15B parameters active",
                      "quantization_precision": "bf16 mixed precision",
                      "rl_algorithm": "None (SFT only)",
                      "reward_mechanism": "None (SFT only)",
                      "post_training_infrastructure": "DeepSpeed ZeRO-1; evaluation via Eureka ML Insights and VLMEvalKit",
                      "safety_evaluation_protocol": "Automated red teaming on Azure across disallowed content (sexual, violent, hateful, self-harm), copyright/IP, jailbreak susceptibility; RAI training data (Hateful Memes, VLGuard, Think-in-Safety, WildGuard)",
                      "safety_defect_rate": "Text-to-Text 1.4%; Image-to-Text 4.5%",
                      "reasoning_mode": "Hybrid/mixed — default learned switching, with explicit <think> / <nothink> override tokens; ~20 percent reasoning data",
                      "synthetic_data_generation_method": "Systematic filtering, error correction, and synthetic augmentation", 
                      "innovation": "Compact multimodal reasoning model achieving competitive accuracy at far lower compute/token cost; hybrid reasoning/non-reasoning training with explicit <think>/<nothink> mode tokens; systematic data filtering, error correction, and synthetic augmentation",
                      "research_problem": "Building smaller, efficient multimodal reasoning models that push the accuracy-vs-compute Pareto frontier, including when to reason vs. answer directly",
                      "application": "General vision-language tasks, math and science multimodal reasoning, document/chart understanding, OCR, GUI grounding and computer-using agents (CUA)",
                      "benchmark_result": "AI2D 84.8, ChartQA 83.3, HallusionBench 64.4, MathVerse-MINI 44.9, MathVision-MINI 36.2, MathVista-MINI 75.2, MMMU-VAL 54.3, MMStar 64.5, OCRBench 76, ScreenSpot-v2 88.2",
                      "license": "open weight",
                      "source_code": "https://github.com/microsoft/Phi-4-reasoning-vision-15B ; https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B"
                }
              ]
          }
        

        # Build messages with few-shot examples
        messages = [
            {   "role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example1_input}",  # noqa: E501
            },
            {   "role": "assistant", "content": json.dumps(example1_output)},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example2_input}",  # noqa: E501
            },
            {   "role": "assistant", "content": json.dumps(example2_output)},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example3_input}",  # noqa: E501
            },
            {   "role": "assistant", "content": json.dumps(example3_output)},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example4_input}",  # noqa: E501
            },
            {   "role": "assistant", "content": json.dumps(example4_output)},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example5_input}",  # noqa: E501
            },
            {   "role": "assistant", "content": json.dumps(example5_output)},
            {
                "role": "user",
                "content": f"Extract ALL model versions/variants introduced in this paper:\n\n{example6_input}",  # noqa: E501
            },
            {   "role": "assistant", "content": json.dumps(example6_output)},
            {
                  "role": "user",
                  "content": (
                      "Extract ALL model versions, variants, and sizes this paper "
                      "introduces, following the ORKG R609825 rules above. Return JSON only.\n\n"
                      f"{paper_snippet}"
                  ),
              },
        ]

        return messages

    def _call_api_with_retry(self, messages: List[Dict[str, str]]) -> Optional[Any]:
        """
        Call KISSKI API with retry logic, exponential backoff, and escalating
        timeouts.  Handles transient errors (timeouts, connection drops, server
        errors, rate limits) so that individual chunk failures don't silently
        kill the whole extraction.

        Returns the API response object, or ``None`` when all retries are
        exhausted.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._enforce_rate_limit()

                # Escalate timeout: +30 s per retry so slow-but-alive backends
                # eventually get enough headroom.
                attempt_timeout = self.timeout + (attempt - 1) * 30

                logger.info(
                    "API call attempt %d/%d (timeout=%ds, model=%s)",
                    attempt,
                    self.retry_attempts,
                    attempt_timeout,
                    self.model_name,
                )

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=attempt_timeout,
                )

                if attempt > 1:
                    logger.info("API call succeeded on attempt %d", attempt)

                return response

            except (APITimeoutError, APIConnectionError) as exc:
                last_exception = exc
                if attempt < self.retry_attempts:
                    wait = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                    logger.warning(
                        "Transient error on attempt %d/%d: %s. "
                        "Retrying in %.1fs (next timeout=%ds)...",
                        attempt,
                        self.retry_attempts,
                        type(exc).__name__,
                        wait,
                        self.timeout + attempt * 30,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "API call failed after %d attempts: %s: %s",
                        self.retry_attempts,
                        type(exc).__name__,
                        exc,
                    )

            except RateLimitError as exc:
                last_exception = exc
                if attempt < self.retry_attempts:
                    wait = max(self.retry_delay * (2**attempt), 10) + random.uniform(0, 5)
                    logger.warning(
                        "Rate limited on attempt %d/%d. Retrying in %.1fs...",
                        attempt,
                        self.retry_attempts,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Rate limit exceeded after %d attempts",
                        self.retry_attempts,
                    )

            except InternalServerError as exc:
                last_exception = exc
                if attempt < self.retry_attempts:
                    wait = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                    logger.warning(
                        "Server error (HTTP %s) on attempt %d/%d. Retrying in %.1fs...",
                        getattr(exc, "status_code", "5xx"),
                        attempt,
                        self.retry_attempts,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Server error persisted after %d attempts: %s",
                        self.retry_attempts,
                        exc,
                    )

        logger.error(
            "All %d retry attempts exhausted. Last error: %s", self.retry_attempts, last_exception
        )
        return None

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

            messages = self._create_extraction_messages(paper_text, paper_metadata)

            response = self._call_api_with_retry(messages)

            if response is None:
                return None

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
                    if not model_data.get("organization"):
                        model_data["organization"] = None
                    if not model_data.get("parameters"):
                        model_data["parameters"] = None
                    if not model_data.get("license"):
                        model_data["license"] = None
                        
            # Derive the multiple-models flag if the LLM omitted it
            if "models" in json_data and "paper_describes_multiple_models" not in json_data:
                json_data["paper_describes_multiple_models"] = len(json_data["models"]) > 1

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

                # Truncated mid-string (model hit max_tokens): close the dangling
                # string literal so its (cut-off) value still parses as valid JSON.
                if in_str:
                    text += '"'

                # If truncation left a dangling '"key":' with no value, drop it.
                text = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", text)
                # Drop a trailing comma that would otherwise yield invalid ",}".
                text = re.sub(r",\s*$", "", text)

                # Append missing closers in reverse order
                if stack:
                    closers = {"{": "}", "[": "]"}
                    text += "".join(closers[sym] for sym in reversed(stack))
                return text

            # Remove prompt echo if present
            if "<|assistant|>" in response_text:
                response_text = response_text.split("<|assistant|>")[-1]

            # Strip reasoning-model thinking blocks (Qwen3, DeepSeek-R1, etc.)
            # Handles both closed <think>...</think> and unclosed <think>...
            if "<think>" in response_text:
                response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text)
                # Unclosed <think> (model hit token limit while thinking):
                # drop everything from <think> to end, keep anything before it
                if "<think>" in response_text:
                    response_text = response_text[: response_text.index("<think>")]

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
                  # Braces never balanced — the response is truncated. Keep everything
                  # from the first "{" so the repair/balancer below can salvage any
                  # complete objects and close the open brackets.
                  end = len(response_text)

            response_text = response_text[start:end]

            # BEST PRACTICE: Try parsing the minimally cleaned JSON first
            # Only apply aggressive repair if standard parsing fails
            parsed = None
            try:
                parsed = json.loads(response_text)
                # Success: valid JSON, no cleaning needed
            except json.JSONDecodeError as initial_error:
                # Parsing failed - apply enhanced cleaning and repair
                logger.debug(f"Initial parse failed: {initial_error}. Applying cleaning...")

                # Escape double-quotes that appear INSIDE string values — a common
                # LLM error, e.g. '"research_problem": "models that "reason" over
                # images"' or a value like '"MMLU "mini": 88"'.
                # Key vs value matters: a KEY string closes on ':', a VALUE string
                # closes on ',', '}' or ']'. Any other quote inside the string is an
                # inner quote and gets escaped. Tracking key/value avoids mis-closing
                # a value at an inner '"..." :' (the bug in the naive version).
                def _escape_inner_quotes(text: str) -> str:
                    out: List[str] = []
                    in_str = False
                    is_key = False
                    last_sig = ""  # last significant char seen outside a string
                    i, n = 0, len(text)
                    while i < n:
                        ch = text[i]
                        if not in_str:
                            out.append(ch)
                            if ch == '"':
                                in_str = True
                                is_key = last_sig in "{,["
                            elif not ch.isspace():
                                last_sig = ch
                        elif ch == "\\" and i + 1 < n:
                            out.append(ch)
                            out.append(text[i + 1])
                            i += 2
                            continue
                        elif ch == '"':
                            j = i + 1
                            while j < n and text[j] in " \t\r\n":
                                j += 1
                            nxt = text[j] if j < n else ""
                            closers = ":" if is_key else ",}]"
                            if nxt in closers or nxt == "":
                                out.append(ch)
                                in_str = False
                            else:
                                out.append('\\"')
                        else:
                            out.append(ch)
                        i += 1
                    return "".join(out)

                response_text = _escape_inner_quotes(response_text)

                # Enhanced cleaning (only when needed)
                # Remove dots before field names (e.g., ."field_name" -> "field_name")
                response_text = re.sub(r'\.\s*"', '"', response_text)

                # Fix missing opening quotes for field names (handles spaces/hyphens too)
                response_text = _quote_unquoted_keys(response_text)

                # Fix missing closing quotes for UNQUOTED VALUES (but NOT arrays/objects)
                # OLD (broke []): r':\s*([^",}\]]+?)(\s*[,}\]])'
                # NEW: only match when value doesn't start with [ or { (to preserve [] and {})
                response_text = re.sub(
                    r':\s*([^",}\[\{][^",}\]]*?)(\s*[,}\]])', r': "\1"\2', response_text
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
                    # Show the actual break point so failures aren't debugged blind.
                    pos = getattr(parse_error, "pos", None)
                    if pos is not None and response_text:
                        lo, hi = max(0, pos - 100), pos + 100
                        logger.error(
                            "Cleaned response around error (pos %d): ...%s>>><<<%s...",
                            pos,
                            response_text[lo:pos],
                            response_text[pos:hi],
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
        """Recursively replace placeholder strings with None."""
        if isinstance(data, dict):
            return {k: self._coerce_null_strings(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._coerce_null_strings(v) for v in data]
        if isinstance(data, str) and data.strip().lower() in (
            "null",
            "none",
            "n/a",
            "unknown",
            "not specified",
            "unspecified",
            "not available",
        ):
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
        succeeded = 0
        failed = 0

        for i, chunk in enumerate(text_chunks):
            logger.info("Processing chunk %d/%d (%d chars)", i + 1, len(text_chunks), len(chunk))
            result = self.extract(chunk, paper_metadata)

            if result and result.models:
                all_models.extend(result.models)
                succeeded += 1
            else:
                failed += 1
                logger.warning("Chunk %d/%d produced no models", i + 1, len(text_chunks))

        logger.info(
            "Chunk processing complete: %d/%d succeeded, %d/%d failed, %d total models extracted",
            succeeded,
            len(text_chunks),
            failed,
            len(text_chunks),
            len(all_models),
        )

        if not all_models:
            logger.warning("No models extracted from any chunk")
            return None

        # Deduplicate models
        unique_models = self._deduplicate_models(all_models)
        logger.info(f"Deduplicated to {len(unique_models)} unique model(s)")

        return MultiModelResponse(
            models=unique_models, paper_describes_multiple_models=len(unique_models) > 1
        )

    @staticmethod
    def _normalize_name_spacing(name: str) -> str:
        """Normalize whitespace between letters and digits (e.g. Nano1 → Nano 1)."""
        if not name:
            return name
        return re.sub(r"([A-Za-z])(\d)", r"\1 \2", name).strip()

    def _deduplicate_models(self, models: List[LLMProperties]) -> List[LLMProperties]:
        """
        Deduplicate models: merge variants that refer to the same model version.

        CRITICAL: Preserves version distinctions (e.g., Llama 3, 3.1, 3.2, 3.3 stay separate)
        by including version token in the deduplication key.

        Grouping key: (model_family, version_token, parameters)
        - model_family: e.g., "Llama", "GPT", "BERT"
        - version_token: e.g., "3", "3.1", "3.2" (extracted from model_name)
        - parameters: e.g., "8B", "70B", "405B"

        This ensures:
        - "Llama 3 8B" and "Llama 3 70B" merge → "Llama 3" (same version, different sizes)
        - "Llama 3.1 8B" and "Llama 3.2 8B" stay separate (different versions)
        """
        # Quality gate: drop truncated / near-empty extractions early
        _MIN_NAME_LEN = 3
        _MIN_FIELDS = 3
        _CHECK_FIELDS = [
            "model_name",
            "model_family",
            "organization",
            "innovation",
            "parameters",
            "pretraining_architecture",
            "pretraining_task",
            "pretraining_corpus",
            "license",
            "research_problem",
            "application",
        ]
        filtered: List[LLMProperties] = []
        for m in models:
            name = (m.model_name or "").strip()
            if len(name) < _MIN_NAME_LEN:
                logger.warning(
                    "Dedup quality gate: dropping model with short name %r (len=%d < %d)",
                    name,
                    len(name),
                    _MIN_NAME_LEN,
                )
                continue
            filled = sum(
                1 for f in _CHECK_FIELDS if getattr(m, f, None) not in (None, "", "null", "None")
            )
            if filled < _MIN_FIELDS:
                logger.info(
                    "Dedup quality gate: dropping %r (%d/%d fields)",
                    name,
                    filled,
                    len(_CHECK_FIELDS),
                )
                continue
            filtered.append(m)
        models = filtered

        groups: Dict[tuple, List[LLMProperties]] = {}

        for m in models:
            fam = (m.model_family or "").strip() or ""
            params = (m.parameters or "").strip() or ""
            params_m = m.parameters_millions
            model_name = self._normalize_name_spacing((m.model_name or "").strip())

            version_token = self._extract_version_from_name(model_name)

            if fam and version_token:
                key = (fam, version_token, params or str(params_m) if params_m is not None else "")
            elif fam and (params or params_m is not None):
                key = (fam, "", params or str(params_m) if params_m is not None else "")
            else:
                key = (model_name, m.model_version or "", m.parameters or "")

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
                # Prefer names with version numbers (e.g., "Llama 3.1" over "Llama 8B")
                if re.search(r"\d+(?:\.\d+)?(?:\s|$)", n):
                    s += 2
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

    def _extract_version_from_name(self, model_name: str) -> str:
        """
        Extract version token from model name for deduplication.

        This must match the logic in model_variant_merger._extract_version_token()
        to ensure consistent version handling across extraction and merging.

        Args:
            model_name: Model name string (e.g., "Llama 3.1 8B")

        Returns:
            Version token (e.g., "3.1") or empty string if no version found
        """
        if not model_name:
            return ""

        # Pattern: model_family + version_number + (space/dash/underscore/end)
        # MUST MATCH model_variant_merger._extract_version_token() logic
        version_patterns = [
            # Pattern 1: "Model 3.1 ..." or "Model 3" (space separator)
            r"(?:^|\s)([A-Za-z][\w-]*?)\s+([vV]?\d+(?:\.\d+)?)(?:\s|[-_]|$)",
            # Pattern 2: "Model-3.1" or "Model_3.1" (dash/underscore separator)
            r"(?:^|\s)([A-Za-z][\w-]*?)[-_]([vV]?\d+(?:\.\d+)?)(?:\s|[-_]|$)",
        ]

        for pattern in version_patterns:
            match = re.search(pattern, model_name)
            if match:
                version = match.group(2)
                # Strip optional 'v' or 'V' prefix
                if version.lower().startswith("v"):
                    version = version[1:]
                return version

        return ""

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
