"""
core/retrieve.py
================
DAY 3: turn a query into its vector, search Qdrant, return the top chunks
as clean dicts the report layer can use (text + citation metadata).
"""

from core.embed import embed_query
from storage import vectorstore
from core.config import config
from core.logging_setup import log


def retrieve(query: str, top_k: int = None, tier: str = None) -> list[dict]:
    """Return the most relevant stored chunks for `query`."""
    top_k = top_k or config.RETRIEVE_TOP_K
    qvec = embed_query(query)
    hits = vectorstore.search(qvec, top_k=top_k, tier=tier)
    results = []
    for h in hits:
        p = h.payload
        results.append({
            "text": p.get("text", ""),
            "source": p.get("source", ""),
            "source_id": p.get("source_id", ""),
            "title": p.get("title", ""),
            "url": p.get("url", ""),
            "tier": p.get("tier", ""),
            "score": h.score,
        })
    log.info(f"[retrieve] {len(results)} chunks for {query!r}")
    return results
