"""
ORKG Client Wrapper

This module provides a wrapper around the ORKG Python client for easier
interaction with the ORKG API, including template and comparison operations.
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from orkg import ORKG, Hosts

logger = logging.getLogger(__name__)

ORKG_HOST_URLS = {
    "sandbox": "https://sandbox.orkg.org",
    "incubating": "https://incubating.orkg.org",
    "production": "https://orkg.org",
}


def normalize_orkg_host(host_or_url: str) -> str:
    """Map ORKG host names or public URLs to the ORKG client host key."""
    value = (host_or_url or "sandbox").strip().rstrip("/")
    parsed = urlparse(value if "://" in value else f"https://{value}")
    hostname = (parsed.hostname or value).lower()

    if hostname == "sandbox.orkg.org":
        return "sandbox"
    if hostname == "incubating.orkg.org":
        return "incubating"
    if hostname == "orkg.org":
        return "production"
    if hostname in ORKG_HOST_URLS:
        return hostname

    logger.warning("Unknown ORKG host/URL '%s'; falling back to sandbox", host_or_url)
    return "sandbox"


def orkg_frontend_url(host_or_url: str) -> str:
    """Return the public ORKG frontend URL for a configured host or endpoint URL."""
    host = normalize_orkg_host(host_or_url)
    return ORKG_HOST_URLS.get(host, ORKG_HOST_URLS["sandbox"])


class ORKGClient:
    """Wrapper for ORKG API operations."""

    def __init__(
        self,
        host: str = "sandbox",
        email: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize ORKG client.

        Args:
            host: ORKG host (sandbox, incubating, production) or public URL
            email: ORKG account email (optional)
            password: ORKG account password (optional)
            timeout: API timeout in seconds
        """
        host = normalize_orkg_host(host)

        # Map host string to Hosts enum
        host_mapping = {
            "sandbox": Hosts.SANDBOX,
            "incubating": Hosts.INCUBATING,
            "production": Hosts.PRODUCTION,
        }

        orkg_host = host_mapping.get(host.lower(), Hosts.SANDBOX)

        # Initialize ORKG client
        if email and password:
            self.orkg = ORKG(host=orkg_host, creds=(email, password))
            logger.info(f"Initialized ORKG client with authentication for {host}")
        else:
            self.orkg = ORKG(host=orkg_host)
            logger.info(f"Initialized ORKG client without authentication for {host}")

        self.host = host
        self.timeout = timeout

    def ping(self) -> bool:
        """
        Test connection to ORKG.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to ping the ORKG service
            result = self.orkg.ping()
            logger.info("ORKG connection test successful")
            return result
        except Exception as e:
            logger.error(f"ORKG connection test failed: {e}")
            return False

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch template structure from ORKG using resources.by_id().

        Templates are resources in ORKG, so we use resources.by_id() to fetch them.

        Args:
            template_id: ORKG template ID (e.g., R609825)

        Returns:
            Template data as dictionary, or None if error
        """
        try:
            logger.info(f"Fetching template {template_id}")
            # Templates are resources, use resources.by_id() as per ORKG Python client
            response = self.orkg.resources.by_id(id=template_id)

            # Handle OrkgResponse object
            if hasattr(response, "content"):
                template = response.content
            else:
                template = response

            logger.info(f"Successfully fetched template {template_id}")
            return template
        except Exception as e:
            logger.error(f"Error fetching template {template_id}: {e}")
            return None

    def get_template_properties(self, template_id: str) -> List[Dict[str, Any]]:
        """
        Get list of properties from a template.

        Args:
            template_id: ORKG template ID

        Returns:
            List of property dictionaries
        """
        template = self.get_template(template_id)
        if template and "properties" in template:
            return template["properties"]
        return []

    def get_comparison(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comparison data from ORKG using resources.by_id.

        Note: Comparisons are resources in ORKG, so we use resources.by_id()
        as per ORKG Python client documentation.

        Args:
            comparison_id: ORKG comparison ID (e.g., R2147679)

        Returns:
            Comparison data as dictionary, or None if error
        """
        try:
            logger.info(f"Fetching comparison {comparison_id}")
            # Comparisons are resources in ORKG, use resources.by_id()
            response = self.orkg.resources.by_id(id=comparison_id)

            # Handle OrkgResponse object
            if hasattr(response, "content"):
                comparison = response.content
            else:
                comparison = response

            logger.info(f"Successfully fetched comparison {comparison_id}")
            return comparison
        except Exception as e:
            logger.error(f"Error fetching comparison {comparison_id}: {e}")
            return None

    def get_comparison_contributions(self, comparison_id: str) -> List[Dict[str, Any]]:
        """
        Get list of contributions from a comparison.

        Args:
            comparison_id: ORKG comparison ID

        Returns:
            List of contribution dictionaries
        """
        comparison = self.get_comparison(comparison_id)
        if comparison and "contributions" in comparison:
            return comparison["contributions"]
        return []

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch paper data from ORKG using papers.by_id().

        Args:
            paper_id: ORKG paper ID

        Returns:
            Paper data as dictionary, or None if error
        """
        try:
            logger.info(f"Fetching paper {paper_id}")
            response = self.orkg.papers.by_id(id=paper_id)

            # Handle OrkgResponse object
            if hasattr(response, "content"):
                paper = response.content
                # If content is bytes (which it seems to be), decode it
                if isinstance(paper, bytes):
                    try:
                        import json

                        paper = json.loads(paper.decode("utf-8"))
                    except Exception as decode_err:
                        logger.error(f"Failed to decode paper content: {decode_err}")
                        return None
            else:
                paper = response

            logger.info(f"Successfully fetched paper {paper_id}")
            return paper
        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None

    def create_paper_with_contributions(
        self,
        title: str,
        authors: List[Dict[str, Any]],
        publication_year: int,
        url: str,
        contributions_data: List[Dict[str, Any]],
        doi: Optional[str] = None,
        publication_month: Optional[int] = None,
        research_field: str = "R133",  # Default to AI
        observatories: Optional[List[str]] = None,
        organizations: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new paper in ORKG with embedded contributions using papers.add().

        According to ORKG Python client documentation, papers.add() accepts a params dict
        with the structure defined at: https://orkg.readthedocs.io/en/latest/client/papers.html

        Args:
            title: Paper title
            authors: List of author dicts with 'name' key
            publication_year: Publication year
            url: Paper URL
            contributions_data: List of contribution dicts with label, classes, statements
            doi: DOI identifier (optional)
            publication_month: Publication month (optional)
            research_field: Research field ID (default: R133 for AI)
            observatories: List of observatory IDs (optional)
            organizations: List of organization IDs (optional)

        Returns:
            Dict with paper_id and contribution_ids if successful, None otherwise
        """
        try:
            logger.info(f"Creating paper with {len(contributions_data)} contributions: {title}")

            # Prepare contributions for the paper structure
            orkg_contributions = []
            orkg_literals = {}
            literal_counter = 0

            for contrib_data in contributions_data:
                contrib_label = contrib_data.get("label", "Unnamed Contribution")
                statements = {}

                for prop in contrib_data.get("properties", []):
                    prop_id = prop.get("property")
                    value = prop.get("value")
                    datatype = prop.get("datatype", "string")

                    # Skip properties with no ID, None values, or empty/whitespace-only values
                    if not prop_id:
                        continue
                    if value is None:
                        continue
                    # For strings, skip empty strings or strings with only whitespace
                    if isinstance(value, str) and (
                        not value.strip() or value.strip().lower() == "none"
                    ):
                        logger.debug(f"Skipping property {prop_id} with empty/whitespace value")
                        continue
                    # For other types, ensure they have valid values
                    if value == "":
                        logger.debug(f"Skipping property {prop_id} with empty string value")
                        continue

                    if prop_id not in statements:
                        statements[prop_id] = []

                    # Format based on datatype per ORKG documentation
                    # Per https://orkg.readthedocs.io/en/latest/client/papers.html example
                    # Use "id" (not "@id") for all references
                    if datatype == "URI" or (isinstance(value, str) and value.startswith("http")):
                        # URI values - reference existing resource or create inline
                        statements[prop_id].append({"id": str(value)})
                    elif datatype in ["Date", "date"]:
                        # Create literal reference for dates
                        literal_id = f"#literal_{literal_counter}"
                        orkg_literals[literal_id] = {"label": str(value), "data_type": "xsd:date"}
                        statements[prop_id].append({"id": literal_id})
                        literal_counter += 1
                    elif datatype in ["Integer", "integer"] or isinstance(value, (int, float)):
                        # Create literal reference for integers
                        literal_id = f"#literal_{literal_counter}"
                        orkg_literals[literal_id] = {
                            "label": str(value),
                            "data_type": "xsd:integer",
                        }
                        statements[prop_id].append({"id": literal_id})
                        literal_counter += 1
                    else:
                        # Text values - create as literals and reference them
                        literal_id = f"#literal_{literal_counter}"
                        orkg_literals[literal_id] = {"label": str(value), "data_type": "xsd:string"}
                        statements[prop_id].append({"id": literal_id})
                        literal_counter += 1

                orkg_contributions.append(
                    {
                        "label": contrib_label,
                        "classes": ["Contribution"],
                        "statements": statements,
                    }
                )

            # Build paper params according to ORKG documentation structure
            paper_params = {
                "title": title,
                "research_fields": [research_field],
                "identifiers": {"doi": [doi]} if doi else {},
                "publication_info": {
                    "published_year": publication_year,
                    "published_month": publication_month,
                    "url": url,
                },
                "authors": authors,
                "contents": {
                    "contributions": orkg_contributions,
                    "literals": orkg_literals,
                },
                "observatories": observatories if observatories else [],
                "organizations": organizations if organizations else [],
                "extraction_method": "AUTOMATIC",
            }

            logger.info(f"Calling papers.add() with {len(orkg_contributions)} contributions")
            response = self.orkg.papers.add(params=paper_params)

            if response.succeeded:
                paper_data = response.content
                paper_id = paper_data.get("id") if isinstance(paper_data, dict) else None

                # Extract contribution IDs from the response
                contribution_ids = []
                if isinstance(paper_data, dict) and "contributions" in paper_data:
                    contribution_ids = [
                        c.get("id") for c in paper_data["contributions"] if isinstance(c, dict)
                    ]

                logger.info(
                    f"Successfully created paper: {paper_id} "
                    f"with {len(contribution_ids)} contributions"
                )
                return {
                    "paper_id": paper_id,
                    "contribution_ids": contribution_ids,
                    "url": response.url,
                }
            else:
                error_msg = (
                    response.content.decode("utf-8")
                    if isinstance(response.content, bytes)
                    else str(response.content)
                )
                logger.error(f"Error creating paper (status {response.status_code}): {error_msg}")

                # Special handling for 401 Unauthorized
                if response.status_code == 401:
                    logger.error("Authentication failed. Check ORKG credentials in .env file.")
                    logger.error("Credentials may have expired or be invalid.")

                return None

        except Exception as e:
            logger.error(f"Error creating paper with contributions: {e}", exc_info=True)
            return None

    def _convert_properties_to_statements(self, properties: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert property list to statements format for ORKG API.

        According to ORKG Python client documentation, statements should be
        formatted as a dictionary where keys are predicate IDs and values
        are lists of statement objects.

        Args:
            properties: List of property dictionaries with 'property' (predicate ID) and 'value'

        Returns:
            Statements dictionary in format: {predicate_id: [statement_objects]}
        """
        statements = {}
        for prop in properties:
            prop_id = prop.get("property")  # This is the predicate ID
            value = prop.get("value")
            datatype = prop.get("datatype", "string")

            if prop_id and value is not None:
                if prop_id not in statements:
                    statements[prop_id] = []

                # Format statement based on datatype
                if datatype == "URI" or (isinstance(value, str) and value.startswith("http")):
                    # URI/literal resource
                    statements[prop_id].append({"id": value})
                elif datatype in ["Date", "date"]:
                    # Date literal
                    statements[prop_id].append({"label": str(value), "datatype": "Date"})
                elif datatype == "Integer" or isinstance(value, int):
                    # Integer literal
                    statements[prop_id].append({"label": str(value), "datatype": "Integer"})
                else:
                    # Text/string literal
                    statements[prop_id].append({"label": str(value)})

        return statements

    def update_contribution(self, contribution_id: str, contribution_data: Dict[str, Any]) -> bool:
        """
        Update an existing contribution using resources.update().

        Note: Contributions are resources in ORKG, so we use resources.update()
        as per ORKG Python client documentation.

        Args:
            contribution_id: ORKG contribution ID
            contribution_data: Updated contribution data

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Updating contribution {contribution_id}")

            # Contributions are resources, use resources.update()
            response = self.orkg.resources.update(
                id=contribution_id,
                label=contribution_data.get("label"),
                statements=self._convert_properties_to_statements(
                    contribution_data.get("properties", [])
                ),
            )

            # Check if update was successful
            if hasattr(response, "succeeded"):
                success = response.succeeded
            else:
                success = response is not None

            if success:
                logger.info(f"Successfully updated contribution {contribution_id}")
            else:
                logger.warning(f"Update may have failed for contribution {contribution_id}")

            return success
        except Exception as e:
            logger.error(f"Error updating contribution: {e}")
            return False

    def add_contribution_to_paper(
        self, paper_id: str, contribution_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Add a new contribution to an existing paper.

        Args:
            paper_id: ORKG paper ID
            contribution_data: Contribution data (label, properties)

        Returns:
            New contribution ID if successful, None otherwise
        """
        try:
            logger.info(
                f"Adding contribution {contribution_data.get('label')} to paper "
                f"{paper_id} using append strategy"
            )

            # Prepare the payload for the 'Old Endpoint' structure which supports merge_if_exists
            # Structure:
            # paper = {
            #    "paper": {
            #       "title": "Title (Required but ignored for merge)",
            #       "researchField": "R11" (Required),
            #       "contributions": [ ... ]
            #    }
            # }

            # We need to fetch the paper title first to satisfy the required field
            paper_info = self.get_paper(paper_id)
            paper_title = (
                paper_info.get("title", "Existing Paper") if paper_info else "Existing Paper"
            )

            # Format statements for the contribution
            # The structure for statements in 'old endpoint' is slightly different?
            # Based on docs: values: { P32: [ { text: "..." } ] }

            # Reuse the existing conversion logic but adapt it if necessary
            # The docs show:
            # "values": { "P32": [ { "text": "...", "@temp": "_..." } ] }

            # Let's construct the properties map
            properties = contribution_data.get("properties", [])
            values_map = {}

            for prop in properties:
                prop_id = prop.get("property")
                value = prop.get("value")
                datatype = prop.get("datatype", "string")

                if prop_id and value is not None:
                    if prop_id not in values_map:
                        values_map[prop_id] = []

                    # Format value object based on type
                    value_obj = {}
                    if datatype == "URI" or (isinstance(value, str) and value.startswith("http")):
                        value_obj["@id"] = str(value)
                    elif datatype in ["Date", "date"]:
                        value_obj["text"] = str(value)
                        value_obj["datatype"] = "xsd:date"
                    elif datatype in ["Integer", "integer"] or isinstance(value, int):
                        value_obj["text"] = str(value)
                        value_obj["datatype"] = "xsd:integer"
                    else:
                        value_obj["text"] = str(value)

                    values_map[prop_id].append(value_obj)

            # Construct the contribution object
            # Include 'classes' field per ORKG documentation for proper classification
            contribution_payload = {
                "name": contribution_data.get("label", "New Contribution"),
                "classes": contribution_data.get(
                    "classes", ["Contribution"]
                ),  # Explicit classification
                "values": values_map,
            }

            # The 'Old Endpoint' structure for merge_if_exists=True
            # explicitely requires the structure:
            # {
            #   "paper": {
            #      "title": ...,
            #      "researchField": ...,
            #      "contributions": [...]
            #   }
            # }

            paper_payload = {
                "paper": {
                    "title": paper_title,
                    "researchField": "R133",
                    "contributions": [contribution_payload],
                }
            }

            logger.info(f"Calling papers.add with merge_if_exists=True for paper '{paper_title}'")

            # IMPORTANT: The python client's papers.add method takes 'params' and expects
            # EITHER the new structure (flat params) OR the old structure (nested 'paper').
            # The 415 error often comes if we mix them or if the library defaults to a media type
            # that the old endpoint doesn't like when combined with this structure.
            # However, we must follow the doc: pass the dict as params.

            response = self.orkg.papers.add(params=paper_payload, merge_if_exists=True)

            if response.succeeded:
                # ... (rest of success handling) ...
                result_content = response.content
                # Decode if needed (add returns dict; get_paper logic similar)
                if isinstance(result_content, bytes):
                    import json

                    result_content = json.loads(result_content.decode("utf-8"))

                if isinstance(result_content, dict):
                    new_contrib_label = contribution_data.get("label")
                    logger.info(f"Append successful. Response ID: {result_content.get('id')}")

                    # We need to return the ID of the new contribution
                    # Fetch paper again to find it
                    updated_paper = self.get_paper(result_content.get("id"))
                    if updated_paper and "contributions" in updated_paper:
                        for c in updated_paper["contributions"]:
                            # Old endpoint may return contributions as list of dicts w/ 'label'
                            c_label = c.get("label")
                            if c_label == new_contrib_label:
                                return c.get("id")

                return result_content.get("id")
            else:
                # ... (error handling) ...
                error_msg = (
                    response.content.decode("utf-8")
                    if isinstance(response.content, bytes)
                    else str(response.content)
                )
                logger.error(f"Failed to append contribution: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Error adding contribution to paper: {e}", exc_info=True)
            return None

    def search_papers(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers in ORKG using papers.get().

        Args:
            query: Search query
            size: Number of results to return

        Returns:
            List of paper dictionaries
        """
        try:
            logger.info(f"Searching papers: {query}")
            # Use papers.get() with title parameter as per ORKG Python client
            response = self.orkg.papers.get(title=query, size=size)

            # Handle OrkgResponse object
            if hasattr(response, "content"):
                papers = response.content
            else:
                papers = response

            # Ensure papers is a list
            if not isinstance(papers, list):
                papers = [papers] if papers else []

            logger.info(f"Found {len(papers)} papers")
            return papers
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []

    def check_model_exists(self, comparison_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Check if a model already exists in the comparison.

        Args:
            comparison_id: ORKG comparison ID
            model_name: Name of the model to check

        Returns:
            Contribution data if found, None otherwise
        """
        try:
            contributions = self.get_comparison_contributions(comparison_id)

            for contribution in contributions:
                # Check if model name matches
                if "properties" in contribution:
                    for prop in contribution["properties"]:
                        if prop.get("label") == "model_name":
                            if prop.get("value") == model_name:
                                logger.info(f"Model {model_name} already exists")
                                return contribution

            logger.info(f"Model {model_name} not found in comparison")
            return None
        except Exception as e:
            logger.error(f"Error checking model existence: {e}")
            return None

    def update_comparison_with_contributions(
        self,
        comparison_id: str,
        title: str,
        description: str,
        new_contribution_ids: List[str],
        research_fields: List[str],
        authors: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Update an existing comparison by adding new contributions.

        Args:
            comparison_id: The ID of the comparison to update
            title: Title for the comparison
            description: Description for the comparison
            new_contribution_ids: List of ALL contribution IDs that should be in the comparison
            research_fields: List of research field IDs
            authors: List of author dictionaries

        Returns:
            The comparison ID if successful, None otherwise
        """
        try:
            logger.info(
                f"Updating comparison {comparison_id} "
                f"with {len(new_contribution_ids)} contributions"
            )

            # ORKG comparisons are updated by creating a new version of the comparison.
            # Argument may be 'comparison_id' or part of the payload depending on client.
            response = self.orkg.comparisons.create(
                comparison_id=comparison_id,
                title=title,
                description=description,
                contributions=new_contribution_ids,
                research_fields=research_fields,
                authors=authors,
            )

            if response.succeeded:
                logger.info(f"Successfully updated comparison {comparison_id}")
                return comparison_id
            else:
                logger.error(f"Failed to update comparison: {response.content}")
                return None
        except Exception as e:
            logger.error(f"Error updating comparison: {e}")
            return None
