"""
core/retrieve.py
================
DAY 7 (retrieval upgrade): section-targeted retrieval.

Instead of one vague query feeding the whole report, we retrieve SEPARATELY
for each section with a focused query — and we can filter by tier (e.g. pull
preprints specifically). This is what fixes the empty Safety/Preprint sections:
the report no longer hopes the right chunks float to the top of one generic
search; it asks for exactly what each section needs.
"""

from core.embed import embed_query
from storage import vectorstore
from core.config import config
from core.logging_setup import log


def _hits_to_dicts(hits) -> list[dict]:
    out = []
    for h in hits:
        p = h.payload
        out.append({
            "text": p.get("text", ""),
            "source": p.get("source", ""),
            "source_id": p.get("source_id", ""),
            "title": p.get("title", ""),
            "url": p.get("url", ""),
            "tier": p.get("tier", ""),
            "score": h.score,
        })
    return out


def retrieve(query: str, top_k: int = None, tier: str = None) -> list[dict]:
    """Single-query retrieval (still used by the `search` command)."""
    top_k = top_k or config.RETRIEVE_TOP_K
    qvec = embed_query(query)
    hits = vectorstore.search(qvec, top_k=top_k, tier=tier)
    results = _hits_to_dicts(hits)
    log.info(f"[retrieve] {len(results)} chunks for {query!r}"
             + (f" (tier={tier})" if tier else ""))
    return results


def retrieve_for_report(drug: str) -> dict[str, list[dict]]:
    """Section-targeted retrieval. Returns a dict: section -> list of chunks.

    Each section gets a focused query (and the preprint section is tier-filtered),
    so every section is fed the evidence it actually needs.
    """
    # Focused queries per section. The drug name is woven in so retrieval stays
    # on-topic while emphasizing the section's angle.
    section_queries = {
        "overview":   f"{drug} overview indication mechanism of action what it treats",
        "efficacy":   f"{drug} efficacy clinical trial outcomes effectiveness results benefit",
        "safety":     f"{drug} adverse effects side effects warnings contraindications risks bleeding toxicity",
    }

    out: dict[str, list[dict]] = {}

    # Main sections: retrieve across all tiers (regulatory + peer-reviewed weigh in).
    for section, q in section_queries.items():
        qvec = embed_query(q)
        hits = vectorstore.search(qvec, top_k=6)
        out[section] = _hits_to_dicts(hits)
        log.info(f"[retrieve:{section}] {len(out[section])} chunks")

    # Preprint section: dedicated pull, FILTERED to preprints only.
    qvec = embed_query(f"{drug} preprint emerging recent findings")
    pre_hits = vectorstore.search(qvec, top_k=5, tier="preprint")
    out["preprint"] = _hits_to_dicts(pre_hits)
    log.info(f"[retrieve:preprint] {len(out['preprint'])} preprint chunks")

    return out
