"""VertexProvider: Google Vertex AI via its OpenAI-compatible endpoint.

Vertex AI exposes an OpenAI-style ``/chat/completions`` surface under
``/v1beta1/projects/{project}/locations/{region}/endpoints/openapi``. Unlike a
plain API key, it authenticates with a short-lived OAuth2 access token minted
from a service-account JSON (or Application Default Credentials). Tokens expire
(~1h), so the bearer header is refreshed on demand rather than baked in once.

Requires ``google-auth`` (install via the ``vertex`` extra, or it is lazily
imported with a clear error if missing).
"""

from __future__ import annotations

import logging
import os
import threading
import time

from ..types import LLMProviderError
from .generic_openai import GenericOpenAIProvider

logger = logging.getLogger(__name__)

DEFAULT_REGION = "global"
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
# Refresh when the cached token is within this many seconds of expiry.
_REFRESH_SKEW_SECONDS = 300

# Module-level credential cache: keyed by resolved credentials path (or
# "__adc__"). Multiple provider instances (tagger, compactor, curator, ...)
# share one Credentials object and therefore one token refresh cycle.
_creds_cache: dict = {}
_creds_lock = threading.Lock()


def _import_google_auth():
    try:
        import google.auth  # noqa: F401
        import google.auth.transport.requests  # noqa: F401
        from google.oauth2 import service_account  # noqa: F401

        return google, service_account
    except ImportError as exc:  # pragma: no cover - depends on env
        raise LLMProviderError(
            "Vertex provider requires the 'google-auth' package. Install it with "
            "`pip install google-auth` (or `pip install virtual-context[vertex]`).",
            provider="vertex",
        ) from exc


def _resolve_credentials_path(explicit: str | None) -> str | None:
    if explicit and os.path.exists(explicit):
        return explicit
    for env_var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = os.environ.get(env_var)
        if path and os.path.exists(path):
            return path
    return None


def _build_base_url(project_id: str, region: str) -> str:
    """OpenAI-compatible base URL for Vertex AI.

    The ``global`` location uses a bare ``aiplatform.googleapis.com`` host; any
    other region prefixes the host with ``{region}-``.
    """
    host = (
        "aiplatform.googleapis.com"
        if region == "global"
        else f"{region}-aiplatform.googleapis.com"
    )
    return f"https://{host}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"


def _get_credentials(credentials_path: str | None):
    """Return (Credentials, project_id), creating + caching on first use."""
    google, service_account = _import_google_auth()
    resolved = _resolve_credentials_path(credentials_path)
    cache_key = resolved or "__adc__"

    with _creds_lock:
        cached = _creds_cache.get(cache_key)
        if cached is None:
            if resolved:
                creds = service_account.Credentials.from_service_account_file(
                    resolved, scopes=_SCOPES,
                )
                project_id = creds.project_id
            else:
                creds, project_id = google.auth.default(scopes=_SCOPES)
            _creds_cache[cache_key] = (creds, project_id)
        else:
            creds, project_id = cached

    override_project = os.environ.get("VERTEX_PROJECT_ID")
    if override_project:
        project_id = override_project
    return creds, project_id


def _refresh_if_needed(creds) -> None:
    google, _ = _import_google_auth()
    needs_refresh = (
        not getattr(creds, "token", None)
        or getattr(creds, "expired", False)
        or (
            getattr(creds, "expiry", None) is not None
            and (creds.expiry.timestamp() - time.time()) < _REFRESH_SKEW_SECONDS
        )
    )
    if needs_refresh:
        with _creds_lock:
            creds.refresh(google.auth.transport.requests.Request())


class VertexProvider(GenericOpenAIProvider):
    """LLM provider for Gemini models via Vertex AI's OpenAI-compatible API."""

    def __init__(
        self,
        credentials_path: str | None = None,
        region: str = DEFAULT_REGION,
        model: str = "google/gemini-2.5-flash",
        temperature: float = 0.3,
        timeout: float = 30.0,
        extra_body: dict | None = None,
    ) -> None:
        self._credentials_path = credentials_path
        self._region = region or DEFAULT_REGION

        creds, project_id = _get_credentials(credentials_path)
        if not project_id:
            raise LLMProviderError(
                "Could not resolve a Google Cloud project_id for Vertex. Set it in "
                "the service-account JSON or via VERTEX_PROJECT_ID.",
                provider="vertex",
            )
        self._creds = creds
        self._project_id = project_id

        super().__init__(
            base_url=_build_base_url(project_id, self._region),
            model=model,
            temperature=temperature,
            api_key="",  # unused; token is minted per request in _get_headers
            extra_body=extra_body,
        )
        self._timeout = timeout

    def _provider_name(self) -> str:
        return "vertex"

    def _get_headers(self) -> dict:
        _refresh_if_needed(self._creds)
        token = getattr(self._creds, "token", None)
        if not token:
            raise LLMProviderError(
                "Vertex credentials produced no access token after refresh.",
                provider="vertex",
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
