"""
Paper Classifier

Uses an LLM call on the paper's title and abstract to determine whether
a paper is an LLM/VLM paper suitable for extraction into the ORKG catalog.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT ="""You are an expert research-paper classifier specializing in machine learning and natural language processing literature. Your task is to decide whether a paper's PRIMARY contribution is to large language models (LLMs) or vision-language models (VLMs).

# Definitions
- LLM: a large neural model trained primarily on text to generate or understand natural language (e.g. GPT, Llama, Mistral).
- VLM: a model that jointly processes vision and language (e.g. image+text input/output).

# Decision rubric
Classify as IN_DOMAIN if the paper's CORE contribution does any of the following:
- Proposes, trains, or releases a new LLM, VLM, or foundation language model
- Advances training methods, architectures, or alignment for LLMs/VLMs (e.g. pretraining, instruction tuning, RLHF, RLAIF, distillation)
- Introduces fine-tuning, prompting, evaluation, or benchmarking frameworks SPECIFICALLY for LLMs/VLMs
- Extends language models to multimodal or cross-modal tasks

Classify as OUT_OF_DOMAIN if the paper:
- Only USES an existing LLM/VLM as a tool to solve a problem in another field (e.g. GPT-4 applied to ecology, medicine, law, finance)
- Belongs to a non-NLP domain (computer architecture, robotics, biology, physics, chemistry, pure systems work, etc.)
- Concerns non-language ML (tabular data, image classification without language, time series, RL on control tasks)

# Key disambiguation
The test is the paper's PRIMARY contribution, not mere mention or use. "We use an LLM to analyze X" is OUT_OF_DOMAIN. "We improve how LLMs do X" is IN_DOMAIN. If genuinely ambiguous, lean on where the novelty lies.

# Paper
Title: {title}
Abstract: {abstract}

# Output
Respond with ONLY a JSON object, no other text:
{{
  "label": "IN_DOMAIN" | "OUT_OF_DOMAIN",
  "primary_domain": "<short domain name, e.g. 'LLM training', 'biology', 'robotics'>"
}}"""


@dataclass
class ClassificationResult:
    is_valid: bool
    reason: str


class PaperClassifier:
    """Classifies papers as LLM/VLM (valid) or out-of-scope (invalid) via an LLM call."""

    def __init__(self, llm_client, model_name: str, timeout: int = 30):
        """
        Args:
            llm_client: openai.OpenAI instance (reused from LLMExtractor.client)
            model_name: KISSKI model name to use for classification
            timeout: Request timeout in seconds
        """
        self.client = llm_client
        self.model_name = model_name
        self.timeout = timeout

    @staticmethod
    def _parse_label(raw: str) -> tuple:
        """
        Extract (label, primary_domain) from a (possibly messy) model response.

        Tries strict JSON first, then falls back to scanning the text for the
        label keyword so a missing/extra wrapper does not break classification.
        Returns ("OUT_OF_DOMAIN", ...) when no label can be found, so unparseable
        responses are treated as invalid rather than silently accepted.
        """
        label = None
        primary_domain = "unknown"

        # Try to locate and parse a JSON object anywhere in the response.
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                data = json.loads(match.group(0))
                label = str(data.get("label", "")).strip().upper() or None
                primary_domain = str(data.get("primary_domain", "unknown")).strip() or "unknown"
            except (json.JSONDecodeError, AttributeError):
                pass

        # Fall back to keyword scanning if JSON parsing didn't yield a label.
        if label not in ("IN_DOMAIN", "OUT_OF_DOMAIN"):
            upper = raw.upper()
            if "OUT_OF_DOMAIN" in upper:
                label = "OUT_OF_DOMAIN"
            elif "IN_DOMAIN" in upper:
                label = "IN_DOMAIN"
            else:
                label = "OUT_OF_DOMAIN"

        return label, primary_domain

    def classify(self, paper_metadata: Dict[str, Any]) -> ClassificationResult:
        """
        Classify whether a paper is an LLM/VLM paper via an LLM call.

        Args:
            paper_metadata: dict with at least 'title'; 'abstract' improves accuracy

        Returns:
            ClassificationResult — is_valid=True means proceed with extraction
        """
        title = (paper_metadata.get("title") or "").strip()
        abstract = (paper_metadata.get("abstract") or "").strip()

        if not title and not abstract:
            logger.warning("No title or abstract for classification; defaulting to valid")
            return ClassificationResult(
                is_valid=True,
                reason="No metadata available; cannot classify",
            )

        prompt = _CLASSIFICATION_PROMPT.format(
            title=title or "(not available)",
            abstract=abstract[:2000] if abstract else "(not available)",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You classify research papers. Respond with ONLY a JSON "
                            "object as specified, no other text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,  # reasoning models (Qwen3, DeepSeek-R1) emit <think> blocks first
                timeout=self.timeout,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Strip <think>...</think> reasoning blocks emitted by Qwen3 / DeepSeek-R1
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw)
            if "<think>" in raw:
                raw = raw[: raw.index("<think>")]

            label, primary_domain = self._parse_label(raw)
            is_valid = label == "IN_DOMAIN"
            logger.info(
                "Paper classifier: '%s...' → %s (%s)",
                title[:60],
                label,
                primary_domain,
            )
            return ClassificationResult(
                is_valid=is_valid,
                reason=f"LLM classification: {label} (domain: {primary_domain})",
            )
        except Exception as exc:
            # On failure default to valid to avoid silently dropping papers
            logger.warning(
                "Paper classification LLM call failed: %s — defaulting to valid", exc
            )
            return ClassificationResult(
                is_valid=True,
                reason=f"Classification call failed ({exc}); proceeding to avoid false rejection",
            )
