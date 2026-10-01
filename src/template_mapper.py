"""
Template Mapper

This module maps extracted LLM data to ORKG template format.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.llm_extractor import LLMProperties, MultiModelResponse

logger = logging.getLogger(__name__)

# Fields whose values represent named entities that should be stored as ORKG Resources
# (not string literals) so they can be linked and reused across papers.
_RESOURCE_FIELDS = {
    "model_family",
    "model_name",
    "organization",
    "pretraining_architecture",
    "pretraining_task",
    "pretraining_corpus",
    "finetuning_task",
    "optimizer",
    "research_problem",
    "license",
    "application",
    "hardware_used",
    "attention_mechanism",
    "context_extension_method",
    "reasoning_mode",
    "quantization_precision",
    "rl_algorithm",
    "tool_calling_format",
    "fusion_architecture",
    "vision_encoder",
    "base_model",
    "weight_clipping_mechanism",
    "supported_language"
}

_MULTI_VALUED_FIELDS = {
      "research_problem": ",",
      "parameters": ",",
      "application": ",",
      "finetuning_task": ",",
      "innovation": ";",   # innovation text contains commas internally; split on ";"
      "benchmark_result": ";",
      "supported_language": ",",
      "finetuning_data": ",",
      "pretraining_corpus": ",",
      "attention_mechanism": ",",
      "quantization_precision": ",",
      "synthetic_data_generation_method": ",",
      "rl_algorithm": ",",
      "reward_mechanism": ",",
      "source_code": ";",
      "vision_encoder": ","
  }

# Comma-configured fields where a ";" in the value should be treated as the ONLY
# top-level separator.
#
# Background: comma fields normally accept both "," and ";" because the model is
# inconsistent about which it uses. But when a value contains BOTH, the ";" is
# almost always the real top-level separator and the "," sits *inside* one value:
#
#   "Agentic, reasoning, and coding (ARC) capabilities; real-world software
#    engineering challenges; long-horizon task execution"
#
# Splitting on both produces five rows and shreds the first concept into
# "Agentic" / "reasoning" / "coding (ARC) capabilities". Honouring only the ";"
# yields the correct three.
#
# This is opt-in per field rather than global on purpose. Only 48 values in the
# whole extraction corpus contain both separators; for research_problem and
# pretraining_corpus semicolon-precedence is clearly right, but for
# reward_mechanism it wrongly joins values that really were comma-separated.
_SEMICOLON_PRECEDENCE_FIELDS = {
      "research_problem",
      "pretraining_corpus",
  }

# Fields whose comma lists are often ONE coordinated phrase rather than several
# values, and so need a grammatical check before splitting.
#
#   "Agentic, reasoning, and coding (ARC) capabilities"     -> 1 value
#   "Agentic engineering, vibe coding, and slide generation" -> 3 values
#
# Both are "A, B, and C" — semicolon precedence alone cannot tell them apart, and
# gets the first right only by accident of a ";" being present. See
# _is_shared_head_coordination for the rule that separates them.
_COORDINATION_AWARE_FIELDS = {
      "research_problem",
  }

  #ORKG predicate IDs per field, one full mapping per target instance. Kept as
  # two complete mappings (not a base + overrides) so each environment's IDs are
  # explicit and editable on their own. Only a few predicates actually differ
  # between instances — training_corpus_size, finetuning_data, context_length,
  # supported_language. The core IDs are currently assumed identical on both;
  # verify them against production before uploading there.
_SANDBOX_FIELD_MAPPING = {
      "model_name": "HAS_MODEL",                    # model name (required)
      "model_family": "P7121",                      # model family (required)
      "date_created": "P49020",                     # date created (required)
      "organization": "P18097",                     # organization (required)
      "innovation": "P15156",                       # innovation (required)
      "pretraining_architecture": "P103000",
      "pretraining_task": "P103001",
      "pretraining_corpus": "P41655",               # (required)
      "finetuning_task": "P116000",
      "optimizer": "P105017",
      "tokenizer": "P43065",
      "parameters": "P103002",                      # number of parameters (required)
      "parameters_millions": "P110076",             # max params in million (required)
      "hardware_used": "P119138",
      "hardware_description": "P119137",
      "carbon_emitted": "P119142",                  # tCO2eq
      "application": "P37544",                       # (required)
      "source_code": "HAS_SOURCE_CODE",             # URI
      "blog_post": "P103003",                        # URI
      "research_problem": "P32",                     # (required)
      # instance-specific predicates (sandbox IDs)
      "training_corpus_size": "P202273",
      "finetuning_data": "P202276",
      "context_length": "P202277",
      "supported_language": "P202278",
      # knowledge_cutoff_date sandbox ID P202274
      "activated_parameters": "P203001",
      "attention_mechanism": "P203008",
      "context_length_max": "P203011",
      "context_extension_method": "P203012",
      "training_pipeline": "P203022",
      "reasoning_mode": "P203018",
      "moe_configuration": "P203014",
      "quantization_precision": "P203016",
      "synthetic_data_generation_method": "P203009",
      "rl_algorithm": "P203004",
      "reward_mechanism": "P203024",
      "tool_calling_format": "P203017",
      "training_environment_scale": "P203015",
      "safety_evaluation_protocol": "P203019",
      "safety_defect_rate": "P203023",
      "fusion_architecture": "P203021",
      "vision_encoder": "P203020",
      "base_model": "P203006",
      "optimizer_innovation": "P203002",
      "benchmark_result": "P203003",
      "weight_clipping_mechanism": "P203007",
      "number_of_attention_heads": "P203010",
      "post_training_infrastructure": "P203025",
  }
  
_PRODUCTION_FIELD_MAPPING = {
      "model_name": "HAS_MODEL",                    # model name (required)
      "model_family": "P7121",                      # model family (required)
      "date_created": "P49020",                     # date created (required)
      "organization": "P18097",                     # organization (required)
      "innovation": "P15156",                       # innovation (required)
      "pretraining_architecture": "P103000",
      "pretraining_task": "P103001",
      "pretraining_corpus": "P41655",               # (required)
      "finetuning_task": "P116000",
      "optimizer": "P105017",
      "tokenizer": "P43065",
      "parameters": "P103002",                      # number of parameters (required)
      "parameters_millions": "P110076",             # max params in million (required)
      "hardware_used": "P119138",
      "hardware_description": "P119137",
      "carbon_emitted": "P119142",                  # tCO2eq
      "application": "P37544",                       # (required)
      "source_code": "HAS_SOURCE_CODE",             # URI
      "blog_post": "P103003",                        # URI
      "research_problem": "P32",                     # (required)
      "training_corpus_size": "P163013",
      "finetuning_data": "P163012",
      "context_length": "P163009",
      "supported_language": "P163010",
      "activated_parameters": "P203084",
      "attention_mechanism": "P203085",
      "context_length_max": "P203086",
      "context_extension_method": "P203087",
      "training_pipeline": "P203088",
      "reasoning_mode": "P203089",
      "moe_configuration": "P203090",
      "quantization_precision": "P203091",
      "synthetic_data_generation_method": "P203092",
      "rl_algorithm": "P203093",
      "reward_mechanism": "P203094",
      "tool_calling_format": "P203095",
      "training_environment_scale": "P203096",
      "safety_evaluation_protocol": "P203097",
      "safety_defect_rate": "P203098",
      "fusion_architecture": "P203099",
      "vision_encoder": "P203100",
      "base_model": "P203101",
      "optimizer_innovation": "P203102",
      "benchmark_result": "P203103",
      "weight_clipping_mechanism": "P203104",
      "number_of_attention_heads": "P203105",
      "post_training_infrastructure": "P203106",
      # knowledge_cutoff_date production ID P163011 — enable when the model extracts it
  }
  
_FIELD_MAPPINGS = {
      "sandbox": _SANDBOX_FIELD_MAPPING,
      "production": _PRODUCTION_FIELD_MAPPING,
  }

  # Hosts that use the production predicate IDs; everything else
  # (sandbox, incubating, unknown) falls back to the sandbox set.
_PRODUCTION_HOSTS = {"production", "prod", "orkg.org", "www.orkg.org"}


class TemplateMapper:
    """Maps extracted LLM data to ORKG template format."""

    def __init__(self, template_id: str = "R609825", host: str = "sandbox"):
          """
          Initialize template mapper.

          Args:
              template_id: ORKG template ID for LLMs
              host: Target ORKG instance ("sandbox", "incubating", or
                    "production"). Selects the matching field-to-predicate
                    mapping, since some predicates live at different IDs per
                    instance.
          """
          self.template_id = template_id
          self.host = "production" if str(host).strip().lower() in _PRODUCTION_HOSTS else "sandbox"
          # Copy so per-instance edits can't mutate the shared module-level dict.
          self.field_mapping = dict(_FIELD_MAPPINGS[self.host])
          logger.info(
              "Initialized TemplateMapper with template: %s (host=%s -> %s predicate IDs)",
              template_id, host, self.host,
          )
          
    def map_model_to_orkg(
        self,
        model: LLMProperties,
        paper_id: Optional[str] = None,
        rows: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Map a single model to ORKG contribution format.

        Args:
            model: Extracted LLM properties
            paper_id: ORKG paper ID (optional)
            rows: Optional {field_name: [row values]} from the semantic
                  validator. Fields present here bypass the heuristic splitter.

        Returns:
            Dictionary in ORKG contribution format
        """
        logger.info(f"Mapping model to ORKG format: {model.model_name}")

        contribution = {"label": model.model_name, "template": self.template_id, "properties": []}

        # Add paper reference if available
        if paper_id:
            contribution["paper_id"] = paper_id

       # Map each field to ORKG property (multi-valued fields expand into
          # one property per value, all sharing the same predicate ID)
        for field_name, property_id in self.field_mapping.items():
            value = getattr(model, field_name, None)

            # A field the validator emptied has no value left to map, even though
            # the model object still carries the original string.
            if rows is not None and field_name in rows and not rows[field_name]:
                logger.debug("Skipping %s — semantic validation rejected all values", field_name)
                continue

            if value is not None:
                props = self._create_properties(property_id, field_name, value, rows=rows)
                if props:
                    contribution["properties"].extend(props)

        logger.info(f"Mapped {len(contribution['properties'])} properties")
        return contribution
    
    @staticmethod #TODO: Capitalize first letter of each new row
    def _capitalize_first(value: str) -> str:
          """
          Uppercase only the first character of a value, preserving the rest.                                                                                                                                                           

          Uses slicing rather than str.capitalize() so acronyms and mixed-case
          tokens stay intact (e.g. "gpt-4 model" -> "Gpt-4 model", not "Gpt-4 model"
          being lowercased elsewhere).
          """
          value = value.strip()
          if not value:
              return value
          return value[0].upper() + value[1:]
    @staticmethod
    def _split_top_level(text: str, seps) -> List[str]:
          """
          Split ``text`` on any separator char in ``seps`` (a single char or an
          iterable of chars), but only when NOT inside (), [] or {} — so a
          parenthetical group containing the separator stays one value.
          e.g. "a, b (x, y), c" split on "," -> ["a", "b (x, y)", "c"].
          """
          sep_set = set(seps)
          parts: List[str] = []
          depth = 0
          buf: List[str] = []
          for ch in text:
              if ch in "([{":
                  depth += 1
                  buf.append(ch)
              elif ch in ")]}":
                  depth = max(0, depth - 1)
                  buf.append(ch)
              elif ch in sep_set and depth == 0:
                  parts.append("".join(buf))
                  buf = []
              else:
                  buf.append(ch)
          parts.append("".join(buf))
          return parts

    @staticmethod
    def _is_shared_head_coordination(parts: List[str]) -> bool:
          """
          Decide whether comma-separated parts are ONE phrase sharing a head noun.

          A bare single-word conjunct is the tell. In "Agentic, reasoning, and
          coding (ARC) capabilities" the words "Agentic" and "reasoning" are not
          research problems on their own — they modify the trailing head noun
          "capabilities", and splitting produces meaningless rows. In "Agentic
          engineering, vibe coding, and slide generation" every conjunct is a
          complete noun phrase that stands alone, so splitting is right.

          Measured over the extraction corpus this keeps 113 of 278
          comma-bearing research_problem values intact that were previously
          shredded into fragments such as "honest", "harmless", "audio" and
          "efficiency" (from "helpful, honest, and harmless" and "text, images,
          and audio").

          Deliberately limited to _COORDINATION_AWARE_FIELDS: on entity fields a
          one-word value is perfectly normal ("English, German, French";
          "WebText, BookCorpus"), and this rule would wrongly fuse them.
          """
          cleaned = [re.sub(r"^and\s+", "", p.strip(), flags=re.IGNORECASE) for p in parts]
          cleaned = [c for c in cleaned if c]
          if len(cleaned) < 2:
              return False
          return any(len(c.split()) == 1 for c in cleaned)

    def _split_coordination_aware(self, text: str) -> List[str]:
          """
          Split on ";" always, and on "," only where the commas separate genuinely
          independent values rather than conjuncts of one coordinated phrase.
          """
          rows: List[str] = []
          for segment in self._split_top_level(text, (";",)):
              if not segment.strip():
                  continue
              parts = self._split_top_level(segment, (","))
              if len(parts) > 1 and not self._is_shared_head_coordination(parts):
                  rows.extend(parts)
              else:
                  rows.append(segment)
          return rows

    @staticmethod
    def _strip_dangling_brackets(text: str) -> str:
          """
          Remove bracket characters left without a partner by an earlier split.

          Handles an unmatched bracket ANYWHERE in the value, not just at the
          ends. The end-only version missed the common case where a split lands
          mid-parenthetical:

              "W4A8 (INT4 for MoE experts, INT8 for Attention/MLP)"
                  -> "W4A8 (INT4 for MoE experts"   <- "(" opens, never closes
                     "INT8 for Attention/MLP)"      <- ")" closes, never opened

          The second row was already cleaned (trailing ")"), but the first was
          not, because its stray "(" sits in the middle rather than at the start.
          Both now come out as plain text.

          Balanced values like "reasoning (advanced)" are left untouched, and
          each bracket type is tracked separately so "f(x]" loses both.
          """
          pairs = {")": "(", "]": "[", "}": "{"}
          open_positions = {"(": [], "[": [], "{": []}
          unmatched = set()

          for index, char in enumerate(text):
              if char in open_positions:
                  open_positions[char].append(index)
              elif char in pairs:
                  opener = pairs[char]
                  if open_positions[opener]:
                      open_positions[opener].pop()
                  else:
                      unmatched.add(index)  # closer with nothing to close

          for positions in open_positions.values():
              unmatched.update(positions)  # openers never closed

          if not unmatched:
              return text

          cleaned = "".join(c for i, c in enumerate(text) if i not in unmatched)
          # Dropping a bracket can leave a double space ("W4A8 ( INT4" -> "W4A8  INT4")
          return re.sub(r"\s{2,}", " ", cleaned).strip()
    
    @staticmethod
    def _dedup_key(text: str) -> str:
          """
          Key used to spot values that are the same statement written differently.

          Exact-match de-duplication is defeated by a single stray character.
          Kimi K2 produced two rows for one method, differing only by a colon:

              "Large-scale agentic data synthesis pipeline systematically generates ..."
              "Large-scale agentic data synthesis pipeline: systematically generates ..."

          Those arrive as genuinely different strings because different chunks of
          the paper phrased the same fact slightly differently and the merge step
          unions them. Comparing on letters and digits alone collapses such
          near-identical variants, while still keeping actually-different values
          apart. Casing is ignored too, since _capitalize_first would otherwise
          make "fp8" and "FP8" identical only AFTER de-duplication has run.

          The class must be "not a word character" rather than "not [a-z0-9]":
          the latter erases every non-Latin value down to the empty string, so
          "English, 中文, 日本語, 한국어" collapsed to two rows — every CJK name
          keyed as "" and deduplicated against the others. supported_language is
          a resource field, so that is a routine value, not an edge case.
          """
          # [\W_] is "not a letter or digit, in any script", plus underscore, so
          # ASCII values key exactly as they did under [^a-z0-9].
          key = re.sub(r"[\W_]+", " ", text.casefold()).strip()
          # A value made only of punctuation still normalizes to nothing; key
          # those on themselves rather than letting them collapse together.
          return key or text.strip()

    def _split_values(
          self, property_name: str, value: Any, rows: Optional[Dict[str, List[str]]] = None
      ) -> List[Any]:
          """
          Break a field value into the individual parts that should each become a
          separate ORKG statement (row).

          - When ``rows`` supplies an entry for this field, it is used verbatim.
            The semantic validator has already decided the row boundaries for that
            field and its decision overrides the heuristics below.
          - List values are always expanded into their elements.
          - String values are split on commas only for fields in _MULTI_VALUED_FIELDS.
          - Everything else is returned as a single-element list unchanged.

          String parts are stripped and empty parts dropped; duplicates are removed
          while preserving first-seen order. Non-string/unhashable parts pass through.
          """
          # Validator-supplied rows win outright. An empty list is a real answer
          # ("every candidate value was rejected"), so test for presence of the
          # key rather than truthiness of the list.
          if rows is not None and property_name in rows:
              return list(rows[property_name])

          if isinstance(value, list):
              raw_parts: List[Any] = value
              did_split = False
          elif property_name in _MULTI_VALUED_FIELDS and isinstance(value, str):
              sep = _MULTI_VALUED_FIELDS[property_name]
              # The model is inconsistent about "," vs ";". Comma-configured
              # fields hold atomic values, so accept BOTH separators. Semicolon
              # fields keep ";" only, because their values contain internal
              # commas (e.g. innovation) that must not be split.
              seps = (",", ";") if sep == "," else (";",)
              if property_name in _COORDINATION_AWARE_FIELDS:
                  # Grammar decides where the commas are real boundaries.
                  raw_parts = self._split_coordination_aware(value)
              else:
                  # ...except for fields where a ";" present in the value marks
                  # the real top-level boundary and the commas are internal. See
                  # _SEMICOLON_PRECEDENCE_FIELDS.
                  if sep == "," and property_name in _SEMICOLON_PRECEDENCE_FIELDS and ";" in value:
                      seps = (";",)
                  raw_parts = self._split_top_level(value, seps)
              did_split = True
          else:
              raw_parts = [value]
              did_split = False

          cleaned: List[Any] = []
          seen = set()
          for part in raw_parts:
              if isinstance(part, str):
                  part = part.strip()
                  if did_split and part.lower().startswith("and "):
                    part = part[4:].strip()
                  if did_split:
                    part = self._strip_dangling_brackets(part)
                  if not part:
                    continue
              # Strings de-duplicate on their normalized form, so punctuation and
              # casing differences don't yield two rows for one statement. The
              # first spelling seen is the one kept.
              marker = self._dedup_key(part) if isinstance(part, str) else part
              try:
                  if marker in seen:
                      continue
                  seen.add(marker)
              except TypeError:
                  pass  # unhashable (e.g. dict) — keep without de-duping
              cleaned.append(part)
          return cleaned

    def _create_properties(
          self,
          property_id: str,
          property_name: str,
          value: Any,
          rows: Optional[Dict[str, List[str]]] = None,
      ) -> List[Dict[str, Any]]:
          """
          Create one or more ORKG property dicts from a single field value.

          Multi-valued fields (and list values) are split into separate property
          dicts that share the same predicate ID, so the ORKG client emits each as
          its own statement row. Works identically for literals and resources
          because the split happens before datatype routing.
          """
          if value is None:
              return []

          props: List[Dict[str, Any]] = []
          for part in self._split_values(property_name, value, rows=rows):
              prop = self._create_property(property_id, property_name, part)
              if prop:
                  props.append(prop)
          return props

    def _create_property(
        self, property_id: str, property_name: str, value: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Create ORKG property from value, assigning the correct datatype so the
        client can route the value to either the resources or literals bucket.

        Datatype routing:
          "resource"  → ORKG Resource node (named entity, reusable across papers)
          "integer"   → xsd:integer literal
          "date"      → xsd:date literal
          "URI"       → reference to an existing resource by URL
          "string"    → xsd:string literal (free-form text)
        """
        if value is None:
            return None

        if isinstance(value, dict):
            return {
                "property": property_id,
                "label": property_name,
                "value": self._format_dict_value(value),
                "datatype": "string",
            }

        if isinstance(value, list):
            return {
                "property": property_id,
                "label": property_name,
                "value": ", ".join(str(v) for v in value),
                "datatype": "string",
            }

        if property_name in _RESOURCE_FIELDS:
            return {
                "property": property_id,
                "label": property_name,
                "value": self._capitalize_first(str(value)),
                "datatype": "resource",
            }

        if isinstance(value, int) or property_name == "parameters_millions":
            return {
                "property": property_id,
                "label": property_name,
                "value": int(value),
                "datatype": "integer",
            }

        if property_name == "date_created":
            return {
                "property": property_id,
                "label": property_name,
                "value": str(value),
                "datatype": "date",
            }

        if property_name in ("source_code", "blog_post") or (
            isinstance(value, str) and value.startswith("http")
        ):
            return {
                "property": property_id,
                "label": property_name,
                "value": str(value),
                "datatype": "URI",
            }

        return {
            "property": property_id,
            "label": property_name,
            "value": self._capitalize_first(str(value)),
            "datatype": "string",
        }

    def _format_dict_value(self, value_dict: Dict[str, Any]) -> str:
        """
        Format dictionary value for ORKG.

        Args:
            value_dict: Dictionary value

        Returns:
            Formatted string
        """
        # Convert dict to readable string format
        items = [f"{k}: {v}" for k, v in value_dict.items()]
        return "; ".join(items)

    def map_multiple_models(
        self,
        models: List[LLMProperties],
        paper_id: Optional[str] = None,
        rows_by_model: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Map multiple models to ORKG format.

        Args:
            models: List of extracted models
            paper_id: ORKG paper ID (optional)
            rows_by_model: Optional {model_name: {field: [rows]}} from the
                           semantic validator.

        Returns:
            List of ORKG contributions
        """
        logger.info(f"Mapping {len(models)} models to ORKG format")

        contributions = []
        for model in models:
            # Keyed by model name because that is the only stable identifier the
            # validator and the mapper share. A model renamed between the two
            # steps simply misses its override and falls back to the heuristic
            # split, which is the pre-existing behaviour.
            rows = (rows_by_model or {}).get(model.model_name)
            if rows_by_model and rows is None:
                logger.debug(
                    "No validated rows for model %r — using heuristic split", model.model_name
                )
            contribution = self.map_model_to_orkg(model, paper_id, rows=rows)
            contributions.append(contribution)

        return contributions

    def map_extraction_result(
        self,
        extraction_result: MultiModelResponse,
        paper_id: Optional[str] = None,
        rows_by_model: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> Dict[str, Any]:
        """
        Map extraction result to ORKG format.

        Args:
            extraction_result: Result from LLM extraction
            paper_id: ORKG paper ID (optional)
            rows_by_model: Optional {model_name: {field: [rows]}} from the
                           semantic validator.

        Returns:
            Dictionary with mapped contributions
        """
        contributions = self.map_multiple_models(
            extraction_result.models, paper_id, rows_by_model=rows_by_model
        )

        return {
            "contributions": contributions,
            "multiple_models": extraction_result.paper_describes_multiple_models,
            "count": len(contributions),
        }

    def create_comparison_entry(
        self, model: LLMProperties, paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a comparison entry for ORKG.

        Args:
            model: Extracted LLM properties
            paper_metadata: Optional paper metadata

        Returns:
            Comparison entry dictionary
        """
        entry = {"label": self._create_model_label(model), "data": {}}

        # Add all available properties
        for field_name in self.field_mapping.keys():
            value = getattr(model, field_name, None)
            if value is not None:
                entry["data"][field_name] = value

        # Add paper metadata if available
        if paper_metadata:
            entry["paper"] = {
                "title": paper_metadata.get("title"),
                "arxiv_id": paper_metadata.get("arxiv_id"),
                "authors": paper_metadata.get("authors"),
                "published": paper_metadata.get("published"),
            }

        return entry

    def _create_model_label(self, model: LLMProperties) -> str:
        """
        Create a descriptive label for the model.

        Args:
            model: LLM properties

        Returns:
            Model label
        """
        parts = [model.model_name]

        if model.model_version:
            parts.append(model.model_version)

        if model.parameters:
            parts.append(model.parameters)

        return " ".join(parts)

    def validate_mapping(self, contribution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mapped contribution.

        Args:
            contribution: Mapped contribution

        Returns:
            Validation report
        """
        report = {"valid": True, "warnings": [], "errors": []}

        # Check required fields
        if "label" not in contribution:
            report["errors"].append("Missing label")
            report["valid"] = False

        if "template" not in contribution:
            report["errors"].append("Missing template")
            report["valid"] = False

        if "properties" not in contribution:
            report["errors"].append("Missing properties")
            report["valid"] = False
        elif len(contribution["properties"]) == 0:
            report["warnings"].append("No properties mapped")

        # Check property structure
        for prop in contribution.get("properties", []):
            if "property" not in prop:
                report["errors"].append("Property missing 'property' field")
                report["valid"] = False
            if "value" not in prop:
                report["errors"].append("Property missing 'value' field")
                report["valid"] = False

        return report

    def merge_duplicate_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge duplicate comparison entries.

        Args:
            entries: List of comparison entries

        Returns:
            Deduplicated and merged entries
        """
        merged = {}

        for entry in entries:
            label = entry.get("label")

            if label not in merged:
                merged[label] = entry
            else:
                # Merge data, preferring non-null values
                existing_data = merged[label].get("data", {})
                new_data = entry.get("data", {})

                for key, value in new_data.items():
                    if value is not None and (
                        key not in existing_data or existing_data[key] is None
                    ):
                        existing_data[key] = value

                merged[label]["data"] = existing_data

        logger.info(f"Merged {len(entries)} entries into {len(merged)} unique entries")
        return list(merged.values())

    def format_for_comparison_update(
        self, entries: List[Dict[str, Any]], comparison_id: str
    ) -> Dict[str, Any]:
        """
        Format entries for comparison update.

        Args:
            entries: List of comparison entries
            comparison_id: ORKG comparison ID

        Returns:
            Formatted update payload
        """
        return {"comparison_id": comparison_id, "entries": entries, "count": len(entries)}
