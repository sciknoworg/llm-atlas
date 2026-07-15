"""
Template Mapper

This module maps extracted LLM data to ORKG template format.
"""

import logging
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
    "tokenizer",
    "research_problem",
    "license",
    "application",
    "hardware_used"
}

_MULTI_VALUED_FIELDS = {
      "research_problem": ",",
      "parameters": ",",
      "application": ",",
      "finetuning_task": ",",
      "innovation": ";",   # innovation text contains commas internally; split on ";"
  }


class TemplateMapper:
    """Maps extracted LLM data to ORKG template format."""

    def __init__(self, template_id: str = "R609825"):
        """
        Initialize template mapper.

        Args:
            template_id: ORKG template ID for LLMs
        """
        self.template_id = template_id
        logger.info(f"Initialized TemplateMapper with template: {template_id}")

        # Define field mapping from extracted data to ORKG properties
        # Based on actual ORKG template R609825
        # NOTE: Some predicates may not exist in sandbox - only use core ones that are verified
        self.field_mapping = {
            # Core fields that exist in sandbox (verified)
            "model_name": "HAS_MODEL",  # model name (required)
            "model_family": "P7121",  # model family (required)
            "date_created": "P49020",  # date created (required)
            "organization": "P18097",  # organization (required)
            "innovation": "P15156",  # innovation (required)
            "pretraining_architecture": "P103000",  # pretraining architecture
            "pretraining_task": "P103001",  # pretraining task
            "pretraining_corpus": "P41655",  # pretraining corpus (required)
            # "training_corpus_size": "P163013",  # DISABLED: not found in sandbox  -> ID on Sandbox: "P202273"
            # "knowledge_cutoff_date": "P163011",  # DISABLED: not found in sandbox  -> ID on Sandbox: "P202274"
            "finetuning_task": "P116000",  # fine-tuning task
            # "finetuning_data": "P163012",  # DISABLED: not found in sandbox -> ID on Sandbox: "P202276"
            "optimizer": "P105017",  # optimizer
            "tokenizer": "P43065",  # tokenizer
            "parameters": "P103002",  # number of parameters (required, text)
            "parameters_millions": "P110076",  # max params in million (required, int)
            # "context_length": "P163009",  # DISABLED: not found in sandbox -> ID on Sandbox: "P202277"
            # "supported_language": "P163010",  # DISABLED: not found in sandbox -> ID on Sandbox: "P202278"
            "hardware_used": "P119138",  # hardware used
            "hardware_description": "P119137",  # hardware description
            "carbon_emitted": "P119142",  # carbon emitted (tCO2eq)
            # "extension": "extension",  # DISABLED: not a valid predicate
            "application": "P37544",  # application (required)
            "source_code": "HAS_SOURCE_CODE",  # source code (URI)
            "blog_post": "P103003",  # blog post (URI)
            # "license": "license",  # DISABLED: not a valid predicate
            "research_problem": "P32",  # research problem (required)
        }

    def map_model_to_orkg(
        self, model: LLMProperties, paper_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Map a single model to ORKG contribution format.

        Args:
            model: Extracted LLM properties
            paper_id: ORKG paper ID (optional)

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

            if value is not None:
                props = self._create_properties(property_id, field_name, value)
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
    
    
    def _split_values(self, property_name: str, value: Any) -> List[Any]:
          """
          Break a field value into the individual parts that should each become a
          separate ORKG statement (row).

          - List values are always expanded into their elements.
          - String values are split on commas only for fields in _MULTI_VALUED_FIELDS.
          - Everything else is returned as a single-element list unchanged.

          String parts are stripped and empty parts dropped; duplicates are removed
          while preserving first-seen order. Non-string/unhashable parts pass through.
          """
          if isinstance(value, list):
              raw_parts: List[Any] = value
              did_split = False
          elif property_name in _MULTI_VALUED_FIELDS and isinstance(value, str):
              raw_parts = value.split(_MULTI_VALUED_FIELDS[property_name])
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
                  if not part:
                      continue
              try:
                  if part in seen:
                      continue
                  seen.add(part)
              except TypeError:
                  pass  # unhashable (e.g. dict) — keep without de-duping
              cleaned.append(part)
          return cleaned

    def _create_properties(
          self, property_id: str, property_name: str, value: Any
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
          for part in self._split_values(property_name, value):
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
        self, models: List[LLMProperties], paper_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Map multiple models to ORKG format.

        Args:
            models: List of extracted models
            paper_id: ORKG paper ID (optional)

        Returns:
            List of ORKG contributions
        """
        logger.info(f"Mapping {len(models)} models to ORKG format")

        contributions = []
        for model in models:
            contribution = self.map_model_to_orkg(model, paper_id)
            contributions.append(contribution)

        return contributions

    def map_extraction_result(
        self, extraction_result: MultiModelResponse, paper_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Map extraction result to ORKG format.

        Args:
            extraction_result: Result from LLM extraction
            paper_id: ORKG paper ID (optional)

        Returns:
            Dictionary with mapped contributions
        """
        contributions = self.map_multiple_models(extraction_result.models, paper_id)

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
