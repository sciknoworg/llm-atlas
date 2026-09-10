"""
Semantic validation of extracted values.

Runs between variant merging and ORKG mapping, and answers one question per
candidate row: *is this value actually an instance of the property it was
extracted for?* Values that are not are dropped before they can become ORKG
statements.

It exists because heuristic splitting and free-form extraction produce three
distinct kinds of garbage, seen in the GLM-5 extraction:

    research_problem   "Agentic, reasoning, and coding (ARC) capabilities; ..."
                       -> split into 5 rows; should have been 3
    rl_algorithm       "... General RL ..."
                       -> "General RL" is not an algorithm
    pretraining_corpus "27 trillion token corpus, prioritizing code and reasoning early on"
                       -> the second half is not a corpus

Three layers, cheapest first, so the LLM only ever sees what the other two
cannot settle:

    Layer 1  value_rules   deterministic, free       fragments, closed vocabularies
    Layer 2  value_cache   disk lookup, free         ~79% of values are repeats
    Layer 3  this module   one LLM call per paper    everything left over

Layer 3 is optional. With it disabled the first two still run, and the pipeline
still benefits from the fragment and vocabulary rules.

WHAT THIS DOES NOT DO
---------------------
It checks a value against the *property description*, not against the paper.
A plausible-but-hallucinated value ("GRPO" in a paper that never mentions GRPO)
passes. Catching that would mean feeding the paper text back in, roughly
doubling extraction cost, and is deliberately out of scope.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.property_descriptions import get_validation_text
from src.value_cache import ValueCache
from src.value_rules import apply_rules
from src.template_mapper import (
    _MULTI_VALUED_FIELDS,
    _RESOURCE_FIELDS,
    TemplateMapper,
)

logger = logging.getLogger(__name__)

# Fields excluded from the default scope because they hold long prose rather
# than entity-like values. Validating them costs a lot of prompt tokens, their
# rows are usually already correct, and a validator has no good basis for
# rejecting a sentence of free text.
_PROSE_FIELDS = {
    "innovation",
    "benchmark_result",
    "hardware_description",
    "moe_configuration",
    "source_code",
}

# Default scope: fields whose values are entity-like and therefore checkable.
DEFAULT_SCOPE = (_RESOURCE_FIELDS | set(_MULTI_VALUED_FIELDS)) - _PROSE_FIELDS

_SYSTEM_PROMPT = """You are validating values extracted from research papers
before they are written to a knowledge graph.

For each property you are given its definition and a list of candidate values
that were extracted for it. Decide, for each candidate value, whether it is
genuinely an INSTANCE of that property as defined.

Reject a value when it is:
- not a thing of the kind the definition describes (e.g. "General RL" for a
  property defined as a named RL algorithm — that is a category of approach,
  not an algorithm)
- a sentence fragment or trailing qualifier rather than a standalone value
  (e.g. "prioritizing code and reasoning early on" for a dataset property)
- a description of what was done rather than the thing itself

Keep a value when it plausibly names a thing of the right kind, even if you
have not heard of it. New models, datasets and algorithms appear constantly;
unfamiliarity is NOT a reason to reject.

Rules:
- Never invent values. Every value you keep must appear in the candidates.
- Do not paraphrase, expand or reword. Copy kept values through
  character-for-character.
- When unsure, keep the value. A wrong rejection loses real data; a wrong
  acceptance is visible and fixable.

Return JSON only, no prose:
{"results": [{"property": "<name>", "value": "<verbatim candidate>",
"keep": true|false, "reason": "<short reason, required only when keep is false>"}]}

Include exactly one entry for every candidate value you were given."""


class SemanticValidator:
    """Validates extracted values against their ORKG property descriptions."""

    def __init__(
        self,
        llm_extractor: Optional[Any] = None,
        scope: Optional[set] = None,
        use_rules: bool = True,
        use_cache: bool = True,
        use_llm: bool = True,
        cache_path: Optional[str] = None,
        mapper: Optional[TemplateMapper] = None,
    ):
        """
        Args:
            llm_extractor: An LLMExtractor. Reused for its retry/backoff and
                           JSON-repair logic rather than reimplementing them.
                           None disables Layer 3.
            scope: Field names to validate. Defaults to DEFAULT_SCOPE.
            use_rules / use_cache / use_llm: Per-layer switches, so a layer can
                           be turned off from config without code changes.
            cache_path: Override the cache file location (used by tests).
            mapper: TemplateMapper used only for its splitting logic, to produce
                    the candidate rows. A default instance is fine — splitting
                    does not depend on the target host.
        """
        self.llm_extractor = llm_extractor
        self.scope = set(scope) if scope else set(DEFAULT_SCOPE)
        self.use_rules = use_rules
        self.use_llm = use_llm and llm_extractor is not None
        self.mapper = mapper or TemplateMapper()
        self.cache = ValueCache(path=cache_path, enabled=use_cache)

        if use_llm and llm_extractor is None:
            logger.warning(
                "Semantic validation: LLM layer requested but no extractor was provided — "
                "running with rules and cache only"
            )
        logger.info(
            "SemanticValidator ready (fields=%d, rules=%s, cache=%s, llm=%s)",
            len(self.scope),
            self.use_rules,
            use_cache,
            self.use_llm,
        )

    # ------------------------------------------------------------------ public

    def validate_models(
        self, models: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Any]]:
        """
        Validate every in-scope field of every model in one paper.

        Args:
            models: Extraction dicts, AFTER variant merging — merging
                    concatenates values, so validating earlier would have its
                    row boundaries destroyed downstream.

        Returns:
            (rows_by_model, report)
              rows_by_model: {model_name: {field: [accepted rows]}} — hand this
                             to TemplateMapper.map_extraction_result().
              report:        an audit trail of what was dropped and why.
        """
        rows_by_model: Dict[str, Dict[str, List[str]]] = {}
        report: Dict[str, Any] = {
            "enabled": True,
            "layers": {"rules": self.use_rules, "cache": self.cache.enabled, "llm": self.use_llm},
            "models": {},
            "counts": {"kept": 0, "dropped": 0, "llm_calls": 0, "values_sent_to_llm": 0},
        }

        if not models:
            return rows_by_model, report

        # Pass 1 — split into candidate rows, then run rules and cache. This
        # collects everything the cheap layers could not settle.
        pending: Dict[Tuple[str, str], List[str]] = {}
        staged: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for model in models:
            if not isinstance(model, dict):
                continue
            model_name = model.get("model_name")
            if not model_name:
                logger.warning("Skipping a model with no model_name — cannot key its rows")
                continue

            staged[model_name] = {}
            for field in sorted(self.scope):
                value = model.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                # Description + value-shape guidance; None means we have no
                # basis to judge this field and must leave it alone.
                description = get_validation_text(field)
                if not description:
                    # No definition means no basis for judgement. Leave the
                    # field entirely alone so the heuristic splitter handles it.
                    logger.debug("No description for %s — skipping validation", field)
                    continue

                candidates = [
                    str(v).strip()
                    for v in self.mapper._split_values(field, value)
                    if str(v).strip()
                ]
                if not candidates:
                    continue

                kept, dropped, unsettled = (
                    apply_rules(field, candidates)
                    if self.use_rules
                    else (list(candidates), [], list(candidates))
                )

                # Cache lookup for whatever the rules had no opinion on.
                to_ask: List[str] = []
                for value_text in unsettled:
                    cached = self.cache.get(field, value_text)
                    if cached is None:
                        to_ask.append(value_text)
                        continue
                    keep, reason = cached
                    if not keep:
                        kept = [k for k in kept if k != value_text]
                        dropped.append((value_text, f"{reason} (cached)"))

                staged[model_name][field] = {
                    "description": description,
                    "candidates": candidates,
                    "kept": kept,
                    "dropped": dropped,
                }
                for value_text in to_ask:
                    pending.setdefault((field, description), [])
                    if value_text not in pending[(field, description)]:
                        pending[(field, description)].append(value_text)

        # Pass 2 — one LLM call for the whole paper, carrying only unseen values.
        verdicts: Dict[Tuple[str, str], Tuple[bool, Optional[str]]] = {}
        if self.use_llm and pending:
            total = sum(len(v) for v in pending.values())
            report["counts"]["values_sent_to_llm"] = total
            logger.info(
                "Semantic validation: asking the model about %d unseen value(s) across %d field(s)",
                total,
                len(pending),
            )
            answered = self._ask_llm(pending)
            if answered is not None:
                report["counts"]["llm_calls"] = 1
                verdicts = answered
                # A value the model did not rule on is kept (see _ask_llm), and
                # that outcome is cached too. Otherwise every such value would be
                # re-sent on every future paper and the cache would never warm up
                # for them.
                for (field, _description), values in pending.items():
                    for value_text in values:
                        verdicts.setdefault((field, value_text), (True, None))
                for (field, value_text), (keep, reason) in verdicts.items():
                    self.cache.put(field, value_text, keep, reason)
                self.cache.save()

        # Pass 3 — apply the verdicts and assemble the result.
        for model_name, fields in staged.items():
            rows_by_model[model_name] = {}
            report["models"][model_name] = {}

            for field, state in fields.items():
                kept = list(state["kept"])
                dropped = list(state["dropped"])

                for value_text in list(kept):
                    verdict = verdicts.get((field, value_text))
                    if verdict is None:
                        continue
                    keep, reason = verdict
                    if not keep:
                        kept.remove(value_text)
                        dropped.append((value_text, reason or "rejected by semantic validation"))

                rows_by_model[model_name][field] = kept
                report["counts"]["kept"] += len(kept)
                report["counts"]["dropped"] += len(dropped)

                if dropped or kept != state["candidates"]:
                    report["models"][model_name][field] = {
                        "candidates": state["candidates"],
                        "accepted": kept,
                        "dropped": [
                            {"value": v, "reason": r} for v, r in dropped
                        ],
                    }
                if not kept:
                    logger.info(
                        "Semantic validation removed every value for %s / %s", model_name, field
                    )

        logger.info(
            "Semantic validation: %d value(s) kept, %d dropped (%d sent to the model)",
            report["counts"]["kept"],
            report["counts"]["dropped"],
            report["counts"]["values_sent_to_llm"],
        )
        return rows_by_model, report

    # ----------------------------------------------------------------- private

    def _build_prompt(self, pending: Dict[Tuple[str, str], List[str]]) -> str:
        """Render the pending values as a compact per-property block."""
        blocks = []
        for (field, description), values in sorted(pending.items()):
            listed = "\n".join(f"  - {v}" for v in values)
            blocks.append(f"PROPERTY: {field}\nDEFINITION: {description}\nCANDIDATES:\n{listed}")
        return "\n\n".join(blocks)

    def _ask_llm(
        self, pending: Dict[Tuple[str, str], List[str]]
    ) -> Optional[Dict[Tuple[str, str], Tuple[bool, Optional[str]]]]:
        """
        Ask the model to judge every pending value in a single call.

        Returns None when the call failed, versus a (possibly empty) dict when it
        succeeded. The caller needs to tell those apart: a successful call with
        no verdict for some value means "the model had nothing to say, keep it
        and remember that", while a failed call must cache nothing at all.

        Semantic validation must never be able to fail an extraction run, so
        every error path here degrades to the pre-existing behaviour.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(pending)},
        ]

        try:
            response = self.llm_extractor._call_api_with_retry(messages)
        except Exception as exc:  # noqa: BLE001 - never let validation break a run
            logger.warning("Semantic validation call raised %s — keeping rule output", exc)
            return None

        if response is None or not getattr(response, "choices", None):
            logger.warning("Semantic validation got no response — keeping rule output")
            return None

        text = response.choices[0].message.content
        if not text or not text.strip():
            logger.warning("Semantic validation got an empty response — keeping rule output")
            return None

        payload = self.llm_extractor._parse_json_response(text)
        if not payload or "results" not in payload:
            logger.warning(
                "Semantic validation response was not usable JSON — keeping rule output"
            )
            return None

        # Index the pending values so we can only accept verdicts about values we
        # actually asked about. Without this a hallucinated or reworded value
        # could silently delete a legitimate row.
        valid = {(field, v) for (field, _), values in pending.items() for v in values}

        verdicts: Dict[Tuple[str, str], Tuple[bool, Optional[str]]] = {}
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            field = item.get("property")
            value_text = item.get("value")
            if not field or value_text is None:
                continue
            key = (field, str(value_text).strip())
            if key not in valid:
                logger.debug("Ignoring verdict for a value we did not ask about: %r", key)
                continue
            keep = bool(item.get("keep", True))
            verdicts[key] = (keep, (item.get("reason") or "").strip() or None)

        unanswered = valid - set(verdicts)
        if unanswered:
            # Missing verdicts default to keeping the value, since the loop in
            # validate_models only acts on verdicts that are present.
            logger.info(
                "Semantic validation returned no verdict for %d value(s) — keeping them",
                len(unanswered),
            )
        return verdicts


def build_validator_from_config(
    config: Dict[str, Any], llm_extractor: Optional[Any], mapper: Optional[TemplateMapper] = None
) -> Optional[SemanticValidator]:
    """
    Construct a SemanticValidator from the ``semantic_validation`` config block,
    or return None when it is switched off.
    """
    settings = (config or {}).get("semantic_validation") or {}
    if not settings.get("enabled", False):
        logger.info("Semantic validation is disabled in config")
        return None

    layers = settings.get("layers") or ["rules", "cache", "llm"]
    scope_setting = settings.get("scope", "default")
    if scope_setting == "all":
        scope = _RESOURCE_FIELDS | set(_MULTI_VALUED_FIELDS)
    elif isinstance(scope_setting, list):
        scope = set(scope_setting)
    else:
        scope = set(DEFAULT_SCOPE)

    return SemanticValidator(
        llm_extractor=llm_extractor,
        scope=scope,
        use_rules="rules" in layers,
        use_cache="cache" in layers,
        use_llm="llm" in layers,
        mapper=mapper,
    )
