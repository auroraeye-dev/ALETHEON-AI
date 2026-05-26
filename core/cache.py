"""
core/cache.py
=============
E3 — fetch caching.

Caches each source's fetch results on disk, keyed by (source, drug), with a TTL.
Re-running a drug you've already fetched skips the slow network calls and reuses
stored evidence — turning a ~37s/$0.003 run into a near-instant, near-free one.

We cache the FETCH layer (the slow, costly part: API calls). Embeddings are
implicitly reused because the Qdrant store persists between runs unless --reset.

Cache entries are JSON: {"ts": <unix>, "evidence": [<Evidence dicts>]}.
Stale entries (older than CACHE_TTL_HOURS) are ignored and refetched.
"""

import os
import json
import time
import hashlib
from dataclasses import asdict

from core.config import config
from core.models import Evidence
from core.logging_setup import log


def _key_path(source: str, drug: str) -> str:
    key = f"{source}:{drug.lower().strip()}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, f"{source}_{h}.json")


def _fresh(ts: float) -> bool:
    age_hours = (time.time() - ts) / 3600.0
    return age_hours < config.CACHE_TTL_HOURS


def get(source: str, drug: str) -> list[Evidence] | None:
    """Return cached Evidence for (source, drug) if present and fresh, else None."""
    path = _key_path(source, drug)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not _fresh(data.get("ts", 0)):
            return None
        evidence = [Evidence(**e) for e in data.get("evidence", [])]
        log.info(f"[cache] HIT {source}:{drug} ({len(evidence)} items)")
        return evidence
    except Exception as e:
        log.warning(f"[cache] read failed for {source}:{drug}: {e}")
        return None


def put(source: str, drug: str, evidence: list[Evidence]) -> None:
    """Store Evidence for (source, drug)."""
    path = _key_path(source, drug)
    try:
        with open(path, "w") as f:
            json.dump({"ts": time.time(),
                       "evidence": [asdict(e) for e in evidence]}, f)
    except Exception as e:
        log.warning(f"[cache] write failed for {source}:{drug}: {e}")


def cached_fetch(source: str, fetch_fn, drug: str) -> list[Evidence]:
    """Wrap a source fetch with caching: return cached if fresh, else fetch + store."""
    hit = get(source, drug)
    if hit is not None:
        return hit
    evidence = fetch_fn(drug)
    put(source, drug, evidence)
    return evidence


def clear() -> int:
    """Delete all cache files. Returns count removed."""
    if not os.path.isdir(config.CACHE_DIR):
        return 0
    n = 0
    for fn in os.listdir(config.CACHE_DIR):
        if fn.endswith(".json"):
            os.remove(os.path.join(config.CACHE_DIR, fn))
            n += 1
    log.info(f"[cache] cleared {n} entries")
    return n
