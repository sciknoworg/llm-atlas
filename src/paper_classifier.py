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

classification_prompt = """<Context>
Research paper classification here means looking at a paper's title and abstract and to detect and determine whether it introduces a brand-new model or releases a new version, size, or a new variant of an already existing and released model. In either case one whose metadata can then be extracted. The central/main distinction is between papers that release a new model(a new LLM, VLM, foundation model, or a named variant) and papers that merely use these models for evaluation, surveys or build their systems using them.

A “release" can be intepreted broadly: it includes technical reports and model cards, not only conference papers. The classification depends on whether a new model, a new version of the model or a named variant is actually made available, not on the sophistication of the method described. When evidence of a released model is absent or unclear, the paper falls outside the domain.
</Context>

<Role>
You are an expert research paper classifier.
</Role>

<Task-Description>
A user will provide you with:
1) a paper title, and
2) a paper abstract.

Your task is to decide whether the paper introduces or updates a model. If the paper does not release a new model, classify it as OUT_OF_DOMAIN. Base your judgment solely on the provided title and abstract; if the abstract is incomplete or unclear, use only the information given.
</Task-Description>

<Definitions>
- LLM: A large language model trained mainly on text for language understanding or generation (e.g. GPT, Llama, Mistral, Qwen).
- VLM: A model that works with both images and text.
- "Release": includes technical reports and model cards, not just conference papers.
</Definitions>

<Decision-Rubric>
Classify as IN_DOMAIN if the paper mainly introduces one of these:
1. A new LLM, VLM, or foundation language model.
2. A new version, size, or variant of an existing LLM/VLM (for example a new release, instruction-tuned version, reasoning model, or MoE variant).
3. A new model architecture, training method, attention mechanism, or alignment method (such as RLHF, RLAIF, DPO, or distillation) only if the paper also releases a trained model or a named variant.

Classify as OUT_OF_DOMAIN if any of these apply:
- The paper only uses an existing LLM/VLM to solve another problem.
- It is from another research field such as biology, robotics, chemistry, physics, computer architecture, ecology, mathematics,  or systems.
- The main contribution of the paper is a benchmark, dataset, evaluation, prompting method, agent framework, RAG system, or fine-tuning method without releasing a new model.
- It is a survey, opinion paper, or analysis of existing models (such as interpretability, bias, or safety).
- It focuses on non-language machine learning, such as image-only, audio-only, tabular, reinforcement learning, or time-series models.
- The novelty is in the application or pipeline instead of the model itself.
</Decision-Rubric>

<Edge-Cases>
- A domain-specific model released with its own name (for example Code Llama or BioMedLM): IN_DOMAIN.
- A LoRA, quantized, or distilled model: IN_DOMAIN only if it is released as its own named model. Otherwise OUT_OF_DOMAIN.
- A benchmark or dataset paper that also trains a baseline model: OUT_OF_DOMAIN.
- Multilingual or non-English LLMs/VLMs follow the same rules as English models. A new non-English LLM/VLM is IN_DOMAIN.
- If the abstract is incomplete or unclear, only use the information provided. If there is no clear evidence of a released model, choose OUT_OF_DOMAIN.
</Edge-Cases>

<Default>
If you are unsure, choose OUT_OF_DOMAIN.
</Default>

<Paper>
Title: {title}
Abstract: {abstract}
</Paper>

<Response-Format>
First, write one sentence describing the paper's main contribution. Then make your decision.

Return your response in JSON format:
{{
  "core_contribution": "<one sentence>",
  "releases_model": true | false,
  "label": "IN_DOMAIN" | "OUT_OF_DOMAIN",
  "primary_domain": "<short domain name, e.g. 'LLM pretraining', 'LLM version release', 'biology', 'LLM application', 'benchmark'>"
}}
</Response-Format>
"""

@dataclass
class ClassificationResult:
    is_valid: bool
    reason: str
    error: bool = False


class PaperClassifier:
    """Classifies papers as LLM/VLM (valid) or out-of-scope (invalid) via an LLM call."""

    def __init__(self, llm_client, model_name: str, timeout: int = 30, temperature: float = 0.0, max_tokens: int = 512):
        """
        Args:
            llm_client: openai.OpenAI instance (reused from LLMExtractor.client)
            model_name: KISSKI model name to use for classification
            timeout: Request timeout in seconds
        """
        self.client = llm_client
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

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
                # json.loads(s) converts JSON string (s) into a Python dict. #TODO: Remove this!
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
            paper_metadata: dict with 'title'; 'abstract'

        Returns:
            ClassificationResult — is_valid=True means proceed with extraction
        """
        title = (paper_metadata.get("title") or "").strip()
        abstract = (paper_metadata.get("abstract") or "").strip()
        
        #TODO: How to handle the case where the title and abstract are missing. Currently skip extraction if this happens. Needs to be discused.
        if not title and not abstract:
            logger.warning("No title or abstract for classification; defaulting to in-valid")
            return ClassificationResult(
                is_valid=False,
                reason="No metadata available; cannot classify",
            )

        prompt = classification_prompt.format(
            title=title or "(not available)",
            abstract=abstract[:2000] if abstract else "(not available)",
        )

        try:
            #TODO: Remove the log before the demo:
            logger.info(
              "Classifier prompt for '%s...' (model: %s)\nabstract:\n%s\nPrompt:\n%s",
              title[:60],
              self.model_name,
              abstract[:100],
              prompt,
            )
            # call the LLM to classify the paper ->>
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
                temperature=self.temperature, #TODO: remove the hard-coded value here. Pull from Config
                max_tokens=self.max_tokens,  # reasoning models (Qwen3, DeepSeek-R1) emit <think> blocks first
                timeout=self.timeout,
            )
            raw = (response.choices[0].message.content or "").strip()
            # log the raw response for debugging --> 
            logger.info(
                  "Classifier raw response for '%s...':\n%s",
                  title[:60],
                  raw,
              )
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
              # On API failure, abort rather than silently extracting an unclassified paper
              logger.error(
                  "Paper classification LLM call failed: %s — aborting", exc
              )
              return ClassificationResult(
                  is_valid=False,
                  reason=f"Classification call failed ({exc})",
                  error=True,
              )
