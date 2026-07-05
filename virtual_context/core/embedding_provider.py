"""Shared embedding provider — single model load shared across engine components."""

from __future__ import annotations

import logging
import os
import sys
from typing import Callable

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Owns one embedding model, shared across SemanticSearchManager,
    EmbeddingTagGenerator, and any other consumer.

    Three construction modes:
    - Cloud/shared: EmbeddingProvider(embed_fn=my_fn)
    - Standalone local: EmbeddingProvider(model_name="all-MiniLM-L6-v2")
    - Remote OpenAI-compatible: EmbeddingProvider(
          model_name="gemini-embedding-001", base_url="http://localhost:4000/v1")
      which calls POST {base_url}/embeddings (works with LiteLLM, OpenAI,
      Ollama, or any server exposing the OpenAI embeddings API).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/") if base_url else None
        self._api_key = api_key or "not-needed"
        self._embed_fn: Callable[[list[str]], list[list[float]]] | None = embed_fn
        self._loaded = embed_fn is not None
        self._load_failed = False

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_embed_fn(self) -> Callable[[list[str]], list[list[float]]] | None:
        """Return the embed function, lazy-loading the model on first call.

        Returns None if sentence-transformers is not installed or load fails.
        """
        if self._loaded:
            return self._embed_fn
        if self._load_failed:
            return None

        if self._base_url:
            self._embed_fn = self._build_remote_embed_fn()
            self._loaded = True
            logger.info(
                "EmbeddingProvider: using remote embeddings model=%s base_url=%s",
                self._model_name, self._base_url,
            )
            return self._embed_fn

        try:
            from sentence_transformers import SentenceTransformer

            old_stderr = sys.stderr
            try:
                sys.stderr = open(os.devnull, "w")
                model = SentenceTransformer(self._model_name)
            finally:
                try:
                    sys.stderr.close()
                except Exception:
                    pass
                sys.stderr = old_stderr

            def embed(texts: list[str]) -> list[list[float]]:
                return model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False,
                ).tolist()

            self._embed_fn = embed
            self._loaded = True
            logger.info("EmbeddingProvider: loaded model %s", self._model_name)
            return self._embed_fn

        except ImportError:
            logger.debug("sentence-transformers not installed, embeddings disabled")
            self._load_failed = True
            return None
        except Exception:
            logger.debug("Failed to load embedding model %s", self._model_name, exc_info=True)
            self._load_failed = True
            return None

    def _build_remote_embed_fn(self) -> Callable[[list[str]], list[list[float]]]:
        """Build an embed function backed by an OpenAI-compatible /embeddings endpoint."""
        import httpx

        url = f"{self._base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        model = self._model_name
        batch_size = 96  # stay under per-request input limits (e.g. Gemini's 100)

        def embed(texts: list[str]) -> list[list[float]]:
            results: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                last_exc: Exception | None = None
                for _attempt in range(2):  # one retry for transient network errors
                    try:
                        resp = httpx.post(
                            url,
                            headers=headers,
                            json={"model": model, "input": batch},
                            timeout=60.0,
                        )
                        resp.raise_for_status()
                        data = sorted(
                            resp.json().get("data", []),
                            key=lambda item: item.get("index", 0),
                        )
                        results.extend(item["embedding"] for item in data)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                if last_exc is not None:
                    logger.error(
                        "Remote embedding request failed (model=%s, url=%s): %s",
                        model, url, last_exc,
                    )
                    raise last_exc
            return results

        return embed
