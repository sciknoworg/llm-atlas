"""
Paper Classifier

Uses an LLM call on the paper's title and abstract to determine whether
a paper is an LLM/VLM paper suitable for extraction into the ORKG catalog.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT = """\
Does this paper introduce, study, or significantly advance a large language \
model (LLM) or vision-language model (VLM)?

A paper QUALIFIES if it:
- Proposes or trains a new LLM, VLM, or foundation language model
- Studies training methods, architectures, or alignment for LLMs/VLMs \
(e.g. RLHF, instruction tuning, pretraining)
- Presents fine-tuning or evaluation frameworks specifically for LLMs/VLMs
- Extends language models for multimodal or cross-modal tasks

A paper does NOT qualify if it:
- Only uses an existing LLM as a tool (e.g. GPT-4 applied to ecology or biology)
- Is about computer architecture, robotics, biology, physics, or other non-NLP domains
- Studies non-language machine learning (tabular data, image recognition without language)

Title: {title}
Abstract: {abstract}

Answer with exactly one word: YES or NO."""


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
                        "content": "You classify research papers. Answer only YES or NO.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,  # reasoning models (Qwen3, DeepSeek-R1) emit <think> blocks first
                timeout=self.timeout,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Strip <think>...</think> reasoning blocks emitted by Qwen3 / DeepSeek-R1
            import re
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw)
            if "<think>" in raw:
                raw = raw[: raw.index("<think>")]

            answer = raw.strip().upper()
            # Accept YES/NO anywhere in the response in case the model adds punctuation
            is_valid = bool(re.search(r"\bYES\b", answer))
            logger.info("Paper classifier: '%s...' → %s", title[:60], answer)
            return ClassificationResult(
                is_valid=is_valid,
                reason=f"LLM classification: {answer}",
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
