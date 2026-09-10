"""
Layer 1 of value validation: deterministic rules, no model, no network.

These rules catch the failure modes that are decidable from the text alone, so
that the (expensive, non-deterministic) LLM pass only ever sees values that
genuinely need judgement.

Two failure modes are handled here:

  1. Split artifacts — a value that is a *fragment* of a sentence rather than a
     standalone entity, produced when the splitter cut a phrase at a comma:
         "a corpus including CommonCrawl, Wikipedia"
             -> "a corpus", "including CommonCrawl", "Wikipedia"
     The "including CommonCrawl" row is obvious garbage.

  2. Closed-vocabulary violations — fields whose value must come from a fixed
     set (pretraining_architecture) or match a fixed shape (date_created).

Everything here is pure: same input, same output, no I/O. That makes the rules
trivial to unit-test and safe to run on every value of every paper.

IMPORTANT — why the fragment rule is field-scoped:
Measured against the project's own extraction corpus (~7,400 distinct values),
a leading-gerund test applied to *all* fields flags 444 values, and most are
correct — "Aligning language models with user intent" is a perfectly good
research_problem. Gerund phrases are the natural grammatical form of a research
problem or an application. So the rule runs only on _ENTITY_NAME_FIELDS, where a
value is supposed to be the *name* of a thing.

These rules are deliberately conservative. Anything they cannot settle is kept
and passed to the LLM layer; a rule that is merely suspicious of a value is a
bug, because a false rejection silently deletes real extracted data.
"""

import logging
import re
from typing import Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


class RuleVerdict(NamedTuple):
    """
    Outcome of running the deterministic rules over one value.

    keep:   False means drop the value outright.
    reason: why it was dropped, or None when kept. Recorded in the validation
            report so a human can audit what the pipeline threw away.
    settled: True means the rules are confident and the LLM layer should not
            re-examine this value. False means "no opinion" — pass it on.
            Keeping these separate matters: most values are kept because no
            rule fired, which is not the same as being verified.
    """

    keep: bool
    reason: Optional[str] = None
    settled: bool = False


# Fields whose values are entity NAMES (a dataset, an optimizer, a piece of
# hardware). A gerund or preposition at the start of one of these is a reliable
# sign of a split artifact.
#
# Deliberately excluded: research_problem, application, finetuning_task,
# pretraining_task, reasoning_mode, reward_mechanism, tool_calling_format,
# synthetic_data_generation_method, training_pipeline. Those are legitimately
# phrased as descriptive phrases and the rule would shred them.
_ENTITY_NAME_FIELDS = {
    "attention_mechanism",
    "base_model",
    "context_extension_method",
    "finetuning_data",
    "hardware_used",
    "license",
    "model_family",
    "optimizer",
    "organization",
    "pretraining_architecture",
    "pretraining_corpus",
    "quantization_precision",
    "rl_algorithm",
    "tokenizer",
    "vision_encoder",
    "weight_clipping_mechanism",
}

# A value starting with one of these words is a continuation of a preceding
# clause, never the start of an entity name.
_LEADING_CONNECTIVE = re.compile(
    r"^(?:with|for|across|based\s+on|including|containing|comprising|spanning|"
    r"such\s+as|which|that|plus|via|using|from|about|over|under|between)\b",
    re.IGNORECASE,
)

# A value opening with one of these verbs is describing what was done rather
# than naming a thing ("prioritizing code and reasoning early on").
#
# This is an explicit verb list, NOT a general "\w+ing" pattern. A morphological
# test looks tempting but is badly wrong in practice: measured over the entity
# fields of this project's corpus it fired on 9 values, of which only ONE was a
# real fragment. It rejected "Hugging Face compute resources" (matched because
# "Hugging" ends in -ing), "Reasoning Reinforcement Learning", "Reasoning and
# tools" and "Coding & Agent ..." — all legitimate. Too many English nouns that
# open an entity name end in -ing.
#
# Values that are garbled but not *fragments* deliberately fall through to the
# LLM layer, which is the right place to judge them.
_LEADING_CLAUSE_VERB = re.compile(
    r"^(?:prioritizing|focusing|emphasizing|leveraging|consisting|ranging|"
    r"targeting|covering|combining|featuring|providing|supporting|allowing|"
    r"involving|addressing|aiming|totalling|totaling|amounting|resulting|"
    r"yielding|achieving|enabling|reaching|drawn\s+from|sourced\s+from|"
    r"curated\s+from|collected\s+from|derived\s+from)\b",
    re.IGNORECASE,
)

# pretraining_architecture is closed by the extraction prompt itself
# ("EXACTLY one of Encoder, Decoder, or Encoder-Decoder"), so anything else is a
# prompt violation we can reject without asking a model.
_CLOSED_VOCABULARIES = {
    "pretraining_architecture": {
        "encoder": "Encoder",
        "decoder": "Decoder",
        "encoder-decoder": "Encoder-Decoder",
        "encoder decoder": "Encoder-Decoder",
    },
}

# Values that carry no information regardless of field. The extractor is told to
# emit null, but it sometimes writes these instead.
_NULL_LIKE = {
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "not specified",
    "not mentioned",
    "not stated",
    "not reported",
    "not applicable",
    "unspecified",
    "-",
    "--",
    "tbd",
}


def _vocabulary_match(vocabulary: Dict[str, str], value: str) -> Optional[str]:
    """
    Resolve a value against a closed vocabulary, tolerating a trailing qualifier.

    "Encoder (image)" is a correct Encoder with a note attached, not a violation,
    so a parenthetical suffix is stripped before matching. Returns the canonical
    spelling, or None when the value is genuinely outside the vocabulary.
    """
    text = value.strip()
    direct = vocabulary.get(text.lower())
    if direct:
        return direct
    without_qualifier = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    if without_qualifier and without_qualifier.lower() in vocabulary:
        return vocabulary[without_qualifier.lower()]
    return None


def canonicalize(field_name: str, value: str) -> str:
    """
    Apply the closed-vocabulary spelling for a field, if one exists.

    "encoder-decoder", "Encoder Decoder" and "Encoder-Decoder (image)" all
    collapse to "Encoder-Decoder" so ORKG does not end up with several resources
    for one concept.
    """
    vocabulary = _CLOSED_VOCABULARIES.get(field_name)
    if not vocabulary:
        return value
    return _vocabulary_match(vocabulary, value) or value


def check_value(field_name: str, value: str) -> RuleVerdict:
    """
    Run every deterministic rule against one already-split value.

    Returns a verdict. `settled=True` means no LLM call is needed for this
    value — either it was rejected outright, or a closed vocabulary confirmed it.
    """
    text = (value or "").strip()

    if not text:
        return RuleVerdict(keep=False, reason="empty value", settled=True)

    if text.lower().strip(" .") in _NULL_LIKE:
        return RuleVerdict(
            keep=False, reason=f"placeholder for a missing value: {text!r}", settled=True
        )

    # A lone separator or bracket left behind by an earlier split.
    if not re.search(r"[A-Za-z0-9]", text):
        return RuleVerdict(keep=False, reason="no alphanumeric content", settled=True)

    vocabulary = _CLOSED_VOCABULARIES.get(field_name)
    if vocabulary is not None:
        if _vocabulary_match(vocabulary, text):
            return RuleVerdict(keep=True, settled=True)
        return RuleVerdict(
            keep=False,
            reason=(
                f"{field_name} accepts only {sorted(set(vocabulary.values()))}; "
                f"got {text!r}"
            ),
            settled=True,
        )

    if field_name in _ENTITY_NAME_FIELDS:
        if _LEADING_CONNECTIVE.match(text):
            return RuleVerdict(
                keep=False,
                reason=(
                    f"starts with a connective, so it is a fragment of a longer phrase "
                    f"rather than a {field_name} name: {text!r}"
                ),
                settled=True,
            )
        if _LEADING_CLAUSE_VERB.match(text):
            return RuleVerdict(
                keep=False,
                reason=(
                    f"describes what was done rather than naming a {field_name}: {text!r}"
                ),
                settled=True,
            )

    # No rule fired. Explicitly NOT settled — the value survives Layer 1 but has
    # not been verified, so the LLM layer still gets a look at it.
    return RuleVerdict(keep=True, settled=False)


def apply_rules(field_name: str, values: List[str]):
    """
    Run check_value over a list of values.

    Returns (kept, dropped, unsettled) where:
      kept      — values that survived, canonicalized
      dropped   — list of (value, reason) for the report
      unsettled — subset of `kept` that no rule had an opinion on, i.e. the
                  values worth spending an LLM call on
    """
    kept: List[str] = []
    dropped = []
    unsettled: List[str] = []

    for value in values:
        text = str(value).strip()
        verdict = check_value(field_name, text)
        if not verdict.keep:
            dropped.append((text, verdict.reason))
            continue
        canonical = canonicalize(field_name, text)
        kept.append(canonical)
        if not verdict.settled:
            unsettled.append(canonical)

    return kept, dropped, unsettled
