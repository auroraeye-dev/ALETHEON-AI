"""
core/embed.py
=============
DAY 2: turn text into vectors using the OpenAI embeddings API.

One function: embed_texts(list[str]) -> list[vector].
Batches requests so we never blow past limits (the Day-8 token work hardens this).
"""

from openai import OpenAI

import time
from core.config import config
from core.logging_setup import log

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


# text-embedding-3-small returns 1536-dim vectors.
EMBED_DIM = 1536


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of strings, batched. Returns one vector per input."""
    if not texts:
        return []
    client = _get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        log.info(f"[embed] embedding {len(batch)} chunks "
                 f"({start + len(batch)}/{len(texts)}) …")
        # E4: retry transient API failures (rate limits, network blips) with
        # simple backoff before giving up.
        resp = None
        last_err = None
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
                break
            except Exception as e:
                last_err = e
                wait = 2 ** attempt  # 1s, 2s, 4s
                log.warning(f"[embed] attempt {attempt + 1} failed ({e}); "
                            f"retrying in {wait}s")
                time.sleep(wait)
        if resp is None:
            from core.errors import PipelineError
            raise PipelineError(f"Embedding failed after 3 attempts: {last_err}")
        vectors.extend(d.embedding for d in resp.data)
        # cost/speed tracking (E2): record token usage if the API reports it
        try:
            from core.metrics import record_embed
            usage = getattr(resp, "usage", None)
            if usage is not None:
                record_embed(getattr(usage, "total_tokens", 0), config.EMBED_MODEL)
        except Exception:
            pass
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
