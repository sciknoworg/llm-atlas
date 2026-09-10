"""
ORKG Client Wrapper

This module provides a wrapper around the ORKG Python client for easier
interaction with the ORKG API, including template and comparison operations.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests
from orkg import ORKG, Hosts

logger = logging.getLogger(__name__)

# Seconds of remaining token lifetime below which the access token is renewed
# rather than reused. ORKG issues 300-second access tokens, so a request that
# starts at 299s could still arrive after expiry; renewing early avoids that.
_TOKEN_REFRESH_MARGIN = 60.0

_ORKG_RESOURCE_ID_RE = re.compile(r"(R\d+)$")
_ORKG_DEFAULT_API_BASE = "https://sandbox.orkg.org"


def normalize_orkg_location_url(url: str, api_base: str = _ORKG_DEFAULT_API_BASE) -> str:
    """
    Fix malformed Location headers from the ORKG API (e.g. https://host:80/...).

    The official client follows the Location header with requests.get(); when the
    server returns https on port 80, TLS fails with SSL WRONG_VERSION_NUMBER.
    """
    if not url:
        return url

    if url.startswith("/"):
        return f"{api_base.rstrip('/')}{url}"

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host.endswith("orkg.org"):
        return url

    scheme = parsed.scheme
    port = parsed.port

    # https://sandbox.orkg.org:80/... -> https://sandbox.orkg.org/...
    if scheme == "https" and port == 80:
        scheme = "https"
        port = None
    elif scheme == "http":
        # ORKG frontends/APIs are served over HTTPS
        scheme = "https"
        if port == 80:
            port = None

    if port and port != 443:
        netloc = f"{parsed.hostname}:{port}"
    else:
        netloc = parsed.hostname or ""

    return urlunparse(
        (scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def parse_orkg_resource_id(url: str) -> Optional[str]:
    """Extract ORKG resource id (e.g. R2166359) from a Location or API URL."""
    if not url:
        return None
    path = urlparse(url).path.rstrip("/")
    match = _ORKG_RESOURCE_ID_RE.search(path)
    return match.group(1) if match else None


ORKG_HOST_URLS = {
    "sandbox": "https://sandbox.orkg.org",
    "incubating": "https://incubating.orkg.org",
    "production": "https://orkg.org",
}


# ORKG accounts whose resources we are willing to reuse.
#
# ORKG is a shared graph: a search for "Google" returns resources created by
# many different people and by bulk importers (the PWC_* entries), whose
# modelling conventions do not match this template. Reusing one of those links
# our contributions into someone else's vocabulary. So a resource is only reused
# when its `created_by` is on this list; otherwise a fresh one is created.
#
# Override per environment with ORKG_RESOURCE_WHITELIST in .env (comma-separated
# user UUIDs). Setting it to an empty string disables the check entirely and
# restores "reuse the first exact-label match", which is what this code did
# before the whitelist existed.
_DEFAULT_RESOURCE_WHITELIST = (
    "314f389f-59d1-4cad-990b-1d6e65891164", 
    "3a55ad95-645b-4c1f-b614-d2293718ee0b", 
)

# How many exact-label matches to examine before giving up and creating a new
# resource. Must be > 1: ORKG orders matches by its own relevance, not by
# creator, and the whitelisted resource is often not first.
_RESOURCE_LOOKUP_CANDIDATES = 10

# Distinguishes "caller did not specify a whitelist" from "caller explicitly
# passed None to disable the check".
_UNSET = frozenset({"__unset__"})


def load_resource_whitelist() -> Optional[frozenset]:
    """
    Read the reusable-creator whitelist from the environment.

    Returns None when the check is disabled (env var present but empty), or a
    frozenset of user UUIDs otherwise. An unset variable falls back to
    _DEFAULT_RESOURCE_WHITELIST so the behaviour is the same for a fresh clone
    with no .env.
    """
    raw = os.getenv("ORKG_RESOURCE_WHITELIST")
    if raw is None:
        return frozenset(_DEFAULT_RESOURCE_WHITELIST)

    ids = {part.strip() for part in raw.split(",") if part.strip()}
    if not ids:
        logger.warning(
            "ORKG_RESOURCE_WHITELIST is empty — resource reuse is UNFILTERED and may "
            "link to resources created by other ORKG users"
        )
        return None
    return frozenset(ids)


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
        resource_whitelist: Optional[frozenset] = _UNSET,
    ):
        """
        Initialize ORKG client.

        Args:
            host: ORKG host (sandbox, incubating, production) or public URL
            email: ORKG account email (optional)
            password: ORKG account password (optional)
            timeout: API timeout in seconds
            resource_whitelist: ORKG user UUIDs whose resources may be reused.
                Defaults to the environment (ORKG_RESOURCE_WHITELIST, else the
                built-in list). Pass None to disable the check. Note the default
                is a sentinel, not None, precisely so that None can mean
                "disabled" rather than "use the default".
        """
        host = normalize_orkg_host(host)

        # Map host string to Hosts enum
        host_mapping = {
            "sandbox": Hosts.SANDBOX,
            "incubating": Hosts.INCUBATING,
            "production": Hosts.PRODUCTION,
        }

        orkg_host = host_mapping.get(host.lower(), Hosts.SANDBOX)
        self._api_base = ORKG_HOST_URLS.get(host, ORKG_HOST_URLS["sandbox"])

        # Disable automatic Location follow: sandbox returns https://host:80 URLs
        # which break TLS (SSL WRONG_VERSION_NUMBER). We resolve paper IDs ourselves.
        client_kwargs = {"follow_location": False, "timeout": timeout}

        if email and password:
            self.orkg = ORKG(host=orkg_host, creds=(email, password), **client_kwargs)
            logger.info(f"Initialized ORKG client with authentication for {host}")
        else:
            self.orkg = ORKG(host=orkg_host, **client_kwargs)
            logger.info(f"Initialized ORKG client without authentication for {host}")

        self.host = host
        self.timeout = timeout
        # Session-level cache: resource label → ORKG resource ID (positive hits only)
        self._resource_cache: Dict[str, str] = {}

        self.resource_whitelist = (
            load_resource_whitelist() if resource_whitelist is _UNSET else resource_whitelist
        )
        if self.resource_whitelist:
            logger.info(
                "Resource reuse restricted to %d whitelisted creator(s)",
                len(self.resource_whitelist),
            )
        else:
            logger.warning("Resource reuse is UNFILTERED (no creator whitelist in effect)")

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

    def refresh_auth(self, margin: float = _TOKEN_REFRESH_MARGIN) -> bool:
        """
        Re-stamp the bundled client's cached Authorization headers.

        The bundled orkg client freezes the bearer token at construction time:
        every namespaced client (papers, resources, statements, ...) copies
        `Bearer <token>` into its own `auth` dict in `NamespacedClient.__init__`
        and passes that same dict to every later request. Session.get_access_token()
        knows how to renew an expired token, but nothing calls it again — so the
        header stays frozen at whatever the token was when ORKG() was built.

        ORKG access tokens last 300 seconds. A pipeline that keeps one client
        across several papers spends minutes extracting between uploads, so by
        the second paper the header is long dead and papers.add() comes back
        401 even though the credentials are perfectly valid.

        Calling this before ORKG work renews the token when little life is left
        and rewrites every cached header. Cheap when the token is still fresh:
        get_access_token() returns the in-memory token without a network call.

        Returns:
            True if the headers now carry a usable token, False if the client is
            unauthenticated or the token could not be renewed.
        """
        session = getattr(self.orkg, "session", None)
        if session is None:
            return False  # unauthenticated client — nothing to refresh

        try:
            # Force a re-login when the token is spent or nearly so. Zeroing the
            # timestamp makes the session's own expiry check fail, which is the
            # supported way to drive it through _login() without calling it.
            expires_in = (getattr(session, "jwt", None) or {}).get("expires_in", 0)
            issued_at = getattr(session, "timestamp", 0)
            if issued_at + expires_in - margin <= time.time():
                session.timestamp = 0

            token = session.get_access_token()
        except Exception as exc:
            logger.error(f"Could not renew ORKG access token: {exc}")
            return False

        if not token:
            logger.error("ORKG returned an empty access token")
            return False

        header = f"Bearer {token}"
        refreshed = 0
        for attribute in vars(self.orkg).values():
            auth = getattr(attribute, "auth", None)
            if isinstance(auth, dict) and "Authorization" in auth:
                auth["Authorization"] = header
                refreshed += 1

        logger.debug("Refreshed ORKG auth header on %d namespaced client(s)", refreshed)
        return True

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
    #Check if a resource can be re-used if it's added by a whitelisted user
    def _is_reusable(self, resource: Dict[str, Any]) -> bool:
        """
        Decide whether an existing resource may be reused.

        Only resources created by a whitelisted ORKG account qualify.
        """
        if not self.resource_whitelist:
            return True
        return resource.get("created_by") in self.resource_whitelist

    def _find_resource_by_label(self, label: str) -> Optional[str]:
        """
        Look up a reusable existing ORKG resource by exact label.

        Scans up to _RESOURCE_LOOKUP_CANDIDATES exact-label matches and returns
        the first one created by a whitelisted account. Scanning several matters:
        ORKG ranks matches by its own relevance, not by creator, so the
        whitelisted resource is frequently not the first hit (on production the
        whitelisted "Transformer" is the 5th).

        Uses a session-level cache so repeated lookups for the same label
        (e.g. "Google" across many contributions) cost only one API call.
        Misses are not cached so that resources created later in the same
        batch run can still be found on a subsequent paper.

        Args:
            label: Exact label to search for

        Returns:
            ORKG resource ID (e.g. "R186195") if a reusable one exists, else None
        """
        if label in self._resource_cache:
            logger.debug("Resource cache hit for '%s': %s", label, self._resource_cache[label])
            return self._resource_cache[label]

        try:
            response = self.orkg.resources.get(
                q=label, exact=True, size=_RESOURCE_LOOKUP_CANDIDATES
            )

            # Raw lookup payload — shows exactly what ORKG returned for this label
            logger.info(
                "Resource lookup for '%s': succeeded=%s type=%s payload=%r",
                label,
                getattr(response, "succeeded", None),
                type(getattr(response, "content", None)).__name__,
                getattr(response, "content", None),
            )

            if response.succeeded:
                content = response.content
                if isinstance(content, list) and content:
                    rejected = []
                    for candidate in content:
                        if not isinstance(candidate, dict):
                            continue
                        resource_id = candidate.get("id")
                        if not resource_id:
                            continue
                        if not self._is_reusable(candidate):
                            rejected.append((resource_id, candidate.get("created_by")))
                            continue
                        self._resource_cache[label] = resource_id
                        logger.info(
                            "Reusing existing ORKG resource '%s' → %s (created_by %s)",
                            label,
                            resource_id,
                            candidate.get("created_by"),
                        )
                        return resource_id

                    if rejected:
                        logger.info(
                            "Found %d resource(s) labelled '%s' but none from a whitelisted "
                            "creator — a new one will be created. Rejected: %s",
                            len(rejected),
                            label,
                            ", ".join(f"{rid} (by {creator})" for rid, creator in rejected),
                        )
        except Exception as exc:
            logger.warning("Resource lookup failed for '%s': %s", label, exc)

        logger.info("No existing ORKG resource matched '%s' — will create inline", label)
        return None

    def _build_contents(
        self, contributions_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """
        Convert mapped contributions into the ORKG "contents" shape.

        Returns (contributions, resources, literals). The same structure is
        accepted by paper creation and by the per-paper contributions endpoint,
        so both go through this one conversion — including whitelist-filtered
        resource reuse and the anyURI handling for links.
        """
        # Prepare contributions for the paper structure
        orkg_contributions = []
        orkg_literals = {}
        orkg_resources = {}
        literal_counter = 0
        resource_counter = 0

        for contrib_data in contributions_data:
            contrib_label = contrib_data.get("label", "Unnamed Contribution")
            statements = {}

            for prop in contrib_data.get("properties", []):
                prop_id = prop.get("property")
                value = prop.get("value")
                datatype = prop.get("datatype", "string")

                if not prop_id:
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and (
                    not value.strip() or value.strip().lower() == "none"
                ):
                    logger.debug(f"Skipping property {prop_id} with empty/whitespace value")
                    continue
                if value == "":
                    logger.debug(f"Skipping property {prop_id} with empty string value")
                    continue

                if prop_id not in statements:
                    statements[prop_id] = []

                if datatype == "resource":
                    label = str(value)
                    existing_id = self._find_resource_by_label(label)
                    if existing_id:
                        # Reuse the existing resource by its real ORKG ID
                        statements[prop_id].append({"id": existing_id})
                    else:
                        # Not found — declare inline; papers.add creates it server-side
                        inline_id = f"#resource_{resource_counter}"
                        orkg_resources[inline_id] = {"label": label, "classes": []}
                        statements[prop_id].append({"id": inline_id})
                        resource_counter += 1
                elif datatype == "URI" or (
                    isinstance(value, str) and value.startswith("http")
                ):
                    # HTTP URL → xsd:anyURI literal so ORKG renders it as a
                      # clickable link (URL chip) instead of a plain text chip.
                      # Do NOT emit {"id": url}: ORKG treats "id" as a
                      # reference to an existing resource, and a URL is not a
                      # resolvable resource id — that makes papers.add 500.
                    literal_id = f"#literal_{literal_counter}"
                    orkg_literals[literal_id] = {
                        "label": str(value),
                        "data_type": "xsd:anyURI",
                    }
                    statements[prop_id].append({"id": literal_id})
                    literal_counter += 1
                elif datatype in ("date", "Date"):
                    literal_id = f"#literal_{literal_counter}"
                    orkg_literals[literal_id] = {
                        "label": str(value),
                        "data_type": "xsd:date",
                    }
                    statements[prop_id].append({"id": literal_id})
                    literal_counter += 1
                elif datatype in ("integer", "Integer") or isinstance(value, (int, float)):
                    literal_id = f"#literal_{literal_counter}"
                    orkg_literals[literal_id] = {
                        "label": str(value),
                        "data_type": "xsd:integer",
                    }
                    statements[prop_id].append({"id": literal_id})
                    literal_counter += 1
                else:
                    # Free-form text → xsd:string literal
                    literal_id = f"#literal_{literal_counter}"
                    orkg_literals[literal_id] = {
                        "label": str(value),
                        "data_type": "xsd:string",
                    }
                    statements[prop_id].append({"id": literal_id})
                    literal_counter += 1

            orkg_contributions.append(
                {
                    "label": contrib_label,
                    "classes": ["Contribution"],
                    "statements": statements,
                }
            )

        logger.info(
            "Prepared %d contributions, %d resource(s), %d literal(s)",
            len(orkg_contributions),
            resource_counter,
            literal_counter,
        )
        return orkg_contributions, orkg_resources, orkg_literals

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

            orkg_contributions, orkg_resources, orkg_literals = self._build_contents(
                contributions_data
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
                    "resources": orkg_resources,
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

                location_url = normalize_orkg_location_url(
                    response.url or "", api_base=self._api_base
                )
                if not paper_id and location_url:
                    paper_id = parse_orkg_resource_id(location_url)
                    if paper_id:
                        logger.info(
                            "Resolved paper id %s from Location header (follow_location=False)",
                            paper_id,
                        )
                        fetched = self.get_paper(paper_id)
                        if isinstance(fetched, dict) and fetched.get("contributions"):
                            contribution_ids = [
                                c.get("id")
                                for c in fetched["contributions"]
                                if isinstance(c, dict) and c.get("id")
                            ]

                logger.info(
                    f"Successfully created paper: {paper_id} "
                    f"with {len(contribution_ids)} contributions"
                )
                return {
                    "paper_id": paper_id,
                    "contribution_ids": contribution_ids,
                    "url": location_url,
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
                if datatype == "resource":
                    statements[prop_id].append({"label": str(value)})
                elif datatype == "URI" or (isinstance(value, str) and value.startswith("http")):
                    # URL as a literal, not {"id": url} (that is a resource ref).
                    statements[prop_id].append({"label": str(value)})
                elif datatype in ("date", "Date"):
                    statements[prop_id].append({"label": str(value), "datatype": "Date"})
                elif datatype in ("integer", "Integer") or isinstance(value, int):
                    statements[prop_id].append({"label": str(value), "datatype": "Integer"})
                else:
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

    # The per-paper contributions endpoint speaks its own media type, and the
    # bundled orkg client does not implement it at all.
    _CONTRIBUTION_MEDIA_TYPE = "application/vnd.orkg.contribution.v2+json"

    def add_contribution_to_paper(
        self, paper_id: str, contribution_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Append one contribution to an existing paper.

        Uses POST /api/papers/{id}/contributions, the endpoint built for exactly
        this. The previous implementation went through papers.add() with
        mergeIfExists, which the bundled client routes to the LEGACY /api/papers
        endpoint with no media type at all — the server now answers that with
        HTTP 415 (Unsupported Media Type), so appending never succeeded:

            Failed to append contribution: {"status":415,
             "title":"Unsupported Media Type", "path":"/api/papers"}

        Returns the new contribution's ID, or None on failure.
        """
        auth = self._auth_header()
        if auth is None:
            logger.error(
                "Cannot add a contribution to %s: client has no ORKG credentials", paper_id
            )
            return None

        label = contribution_data.get("label", "Unnamed Contribution")
        logger.info("Adding contribution %r to paper %s", label, paper_id)

        # Same conversion as paper creation, so resource reuse and URI handling
        # behave identically whichever path a contribution arrives by.
        contributions, resources, literals = self._build_contents([contribution_data])
        if not contributions:
            logger.error("Contribution %r produced no statements — nothing to add", label)
            return None

        payload = {
            "contribution": contributions[0],
            "resources": resources,
            "literals": literals,
            "predicates": {},
            "lists": {},
            "extraction_method": "AUTOMATIC",
        }

        try:
            response = requests.post(
                f"{self._api_base}/api/papers/{paper_id}/contributions",
                json=payload,
                headers={
                    "Content-Type": self._CONTRIBUTION_MEDIA_TYPE,
                    "Accept": self._CONTRIBUTION_MEDIA_TYPE,
                    **auth,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.error("Error adding contribution to paper %s: %s", paper_id, exc)
            return None

        if response.status_code not in (200, 201, 204):
            logger.error(
                "Failed to add contribution %r to paper %s: HTTP %s %s",
                label, paper_id, response.status_code, response.text[:300],
            )
            return None

        # ORKG returns the new resource in the Location header; the body may be
        # empty on 201/204, so the header is the reliable source.
        contribution_id = parse_orkg_resource_id(
            normalize_orkg_location_url(
                response.headers.get("Location", ""), self._api_base
            )
        )
        if not contribution_id:
            try:
                body = response.json()
                contribution_id = body.get("id") if isinstance(body, dict) else None
            except ValueError:
                contribution_id = None

        if contribution_id:
            logger.info(
                "Added contribution %r to paper %s -> %s", label, paper_id, contribution_id
            )
        else:
            logger.warning(
                "Contribution %r accepted for paper %s but no id was returned", label, paper_id
            )
        return contribution_id

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

    # Media type the ORKG comparison endpoints currently speak. The bundled
    # `orkg` package still sends v2, which the server now answers with HTTP 406,
    # so these calls are made directly rather than through the client.
    _COMPARISON_MEDIA_TYPE = "application/vnd.orkg.comparison.v3+json"

    def _auth_header(self) -> Optional[Dict[str, str]]:
        """
        Bearer token for direct REST calls.

        Reuses the ORKG client's own session so the token is shared with every
        other call and refreshed automatically when it expires. Returns None
        when the client was built without credentials.
        """
        session = getattr(self.orkg, "session", None)
        if session is None:
            return None
        try:
            return {"Authorization": f"Bearer {session.get_access_token()}"}
        except Exception as exc:  # noqa: BLE001 - surfaced as "not authenticated"
            logger.error("Could not obtain an ORKG access token: %s", exc)
            return None

    def _get_comparison_v3(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a comparison through the REST API using the v3 media type.

        Separate from get_comparison(), which returns the *resource*
        representation via the ORKG client and is what comparison_updater and
        build_papers_list expect. This one returns the comparison
        representation, the only shape that carries `published`, `sources` and
        `versions.head` — the fields the update path needs.
        """
        url = f"{self._api_base}/api/comparisons/{comparison_id}"
        try:
            response = requests.get(
                url, headers={"Accept": self._COMPARISON_MEDIA_TYPE}, timeout=self.timeout
            )
            if response.status_code != 200:
                logger.error(
                    "Could not fetch comparison %s: HTTP %s", comparison_id, response.status_code
                )
                return None
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("Error fetching comparison %s: %s", comparison_id, exc)
            return None

    def _comparison_api(
        self, comparison_id: str, path: str = ""
    ) -> Optional[Dict[str, Any]]:
        """GET a comparison sub-resource with the v3 media type."""
        url = f"{self._api_base}/api/comparisons/{comparison_id}{path}"
        try:
            response = requests.get(
                url, headers={"Accept": self._COMPARISON_MEDIA_TYPE}, timeout=self.timeout
            )
            if response.status_code != 200:
                logger.error("GET %s: HTTP %s", url, response.status_code)
                return None
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("Error calling %s: %s", url, exc)
            return None

    def update_comparison_selected_paths(self, comparison_id: str) -> bool:
        """
        Make the comparison actually show property rows.

        Adding sources to a comparison puts the contributions in as COLUMNS, but
        the ROWS — the properties being compared — come from a separate list,
        `selected_paths`, stored on /api/comparisons/{id}/contents. A comparison
        with sources but no selected paths renders as an empty table, which is
        exactly what an update that only touches `sources` produces.

        /table-paths reports every predicate reachable from the current sources,
        so it is the set of rows that could be shown. This selects the union of
        what is already chosen and what is newly available, so an existing
        curated row order is preserved and only genuinely new predicates get
        appended.

        Must run AFTER the sources update: available paths are derived from the
        contributions currently attached.
        """
        auth = self._auth_header()
        if auth is None:
            logger.error("Cannot update comparison table: no ORKG credentials")
            return False

        available = self._comparison_api(comparison_id, "/table-paths")
        if available is None:
            return False
        if isinstance(available, dict):
            available = available.get("content") or available.get("paths") or []

        contents = self._comparison_api(comparison_id, "/contents") or {}
        selected = contents.get("selected_paths") or []

        def as_path(entry: Dict[str, Any]) -> Dict[str, Any]:
            # The request model carries only id/type/children; the response also
            # includes label and description, which must not be sent back.
            return {
                "id": entry.get("id"),
                "type": entry.get("type") or "PREDICATE",
                "children": [as_path(c) for c in (entry.get("children") or [])],
            }

        merged: List[Dict[str, Any]] = []
        seen = set()
        for entry in list(selected) + list(available):
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            if entry["id"] in seen:
                continue
            seen.add(entry["id"])
            merged.append(as_path(entry))

        if len(merged) == len(selected):
            logger.info(
                "Comparison %s already shows all %d available propert(ies)",
                comparison_id,
                len(merged),
            )
            return True

        logger.info(
            "Setting comparison %s property rows: %d -> %d",
            comparison_id,
            len(selected),
            len(merged),
        )

        try:
            response = requests.put(
                f"{self._api_base}/api/comparisons/{comparison_id}/contents",
                json={"selected_paths": merged},
                headers={
                    "Content-Type": self._COMPARISON_MEDIA_TYPE,
                    "Accept": self._COMPARISON_MEDIA_TYPE,
                    **auth,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.error("Error updating comparison table %s: %s", comparison_id, exc)
            return False

        if response.status_code in (200, 204):
            logger.info("Comparison %s now shows %d property row(s)", comparison_id, len(merged))
            return True

        logger.error(
            "Failed to update comparison table %s: HTTP %s %s",
            comparison_id,
            response.status_code,
            response.text[:300],
        )
        return False

    def update_comparison_sources(
        self, comparison_id: str, contribution_ids: List[str]
    ) -> bool:
        """
        Add contributions to a LIVE comparison via PUT /api/comparisons/{id}.

        This is what the ORKG frontend does: it rewrites the comparison's
        `sources` list. The request body is a partial update, so only `sources`
        is sent — title and description are deliberately left untouched, which
        removes any chance of renaming the comparison by accident.

        Published comparisons CANNOT be updated: they are frozen snapshots, and
        the API rejects them (ComparisonAlreadyPublished). Each published
        comparison points at the live one it was cut from via `versions.head`,
        so when the configured ID turns out to be published we say so and name
        the head rather than failing obscurely.

        Existing sources are read and preserved — new contributions are appended.
        A blind overwrite would erase every contribution already in the
        comparison on the first run.

        Returns True only when ORKG accepted the update.
        """
        if not contribution_ids:
            logger.info("No contributions to add to comparison %s", comparison_id)
            return True

        auth = self._auth_header()
        if auth is None:
            logger.error(
                "Cannot update comparison %s: client has no ORKG credentials "
                "(set ORKG_EMAIL and ORKG_PASSWORD)",
                comparison_id,
            )
            return False

        comparison = self._get_comparison_v3(comparison_id)
        if comparison is None:
            return False

        if comparison.get("published"):
            head = ((comparison.get("versions") or {}).get("head") or {}).get("id")
            logger.error(
                "Comparison %s is PUBLISHED and cannot be updated (published comparisons "
                "are frozen snapshots). Point orkg.comparison_id at the live comparison%s.",
                comparison_id,
                f" {head}" if head else "",
            )
            return False

        existing = [
            source.get("id")
            for source in (comparison.get("sources") or [])
            if isinstance(source, dict) and source.get("id")
        ]
        merged = list(dict.fromkeys(existing + list(contribution_ids)))
        added = [cid for cid in contribution_ids if cid not in existing]

        if not added:
            logger.info(
                "All %d contribution(s) are already sources of comparison %s",
                len(contribution_ids),
                comparison_id,
            )
            # Still reconcile the property rows: sources can already be present
            # while selected_paths is empty (e.g. after an update that only set
            # sources), which renders as a table with columns but no rows.
            return self.update_comparison_selected_paths(comparison_id)

        logger.info(
            "Adding %d new contribution(s) to comparison %s (%d -> %d sources)",
            len(added),
            comparison_id,
            len(existing),
            len(merged),
        )

        headers = {
            "Content-Type": self._COMPARISON_MEDIA_TYPE,
            "Accept": self._COMPARISON_MEDIA_TYPE,
            **auth,
        }
        payload = {"sources": [{"id": cid, "type": "THING"} for cid in merged]}

        try:
            response = requests.put(
                f"{self._api_base}/api/comparisons/{comparison_id}",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.error("Error updating comparison %s: %s", comparison_id, exc)
            return False

        if response.status_code in (200, 204):
            logger.info("Successfully updated comparison %s", comparison_id)
            # Sources alone give columns but no rows — the property list has to
            # be set separately or the comparison renders empty.
            return self.update_comparison_selected_paths(comparison_id)

        logger.error(
            "Failed to update comparison %s: HTTP %s %s",
            comparison_id,
            response.status_code,
            response.text[:300],
        )
        return False

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
        Add contributions to a comparison. Kept for call-site compatibility.

        Previously this called ``comparisons.create(comparison_id=...)``, which
        was never a valid call: ``create()`` has no ``comparison_id`` parameter
        and its required ``config``/``data`` arguments were not supplied, so
        every attempt raised TypeError, was swallowed by a broad except, and was
        reported as a generic upload failure. Comparison updates therefore never
        actually worked.

        It now delegates to update_comparison_sources(), which does what the
        ORKG frontend does: PUT the comparison's ``sources`` list. title,
        description, research_fields and authors are accepted but unused — a
        partial update leaves them untouched, which is safer than resending
        them, since a wrong title would rename the live comparison.

        Returns:
            The comparison ID on success, None otherwise.
        """
        if self.update_comparison_sources(comparison_id, new_contribution_ids):
            return comparison_id
        return None
