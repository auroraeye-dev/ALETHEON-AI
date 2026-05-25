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


def _dedup(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        key = (c["source"], c["source_id"], c["text"][:60])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def retrieve_for_report(drug: str) -> dict[str, list[dict]]:
    """Section-targeted retrieval with tier-aware biasing.

    Refinements (polish pass):
      - Efficacy/Findings deliberately pulls from regulatory + peer-reviewed so
        strong RCTs surface instead of being crowded out by preprints.
      - A dedicated 'contradiction' pull retrieves BOTH supportive and
        opposing evidence, so the Contradictions section has material to work with.
      - More chunks per section so good evidence isn't lost.
    """
    drug_key = drug.lower().strip()

    section_queries = {
        "overview": f"{drug} overview indication mechanism of action what it treats",
        "safety":   f"{drug} adverse effects side effects warnings contraindications risks bleeding toxicity",
    }

    out: dict[str, list[dict]] = {}

    # Overview + Safety: standard drug-filtered retrieval, a bit deeper (top_k=8).
    for section, q in section_queries.items():
        qvec = embed_query(q)
        hits = vectorstore.search(qvec, top_k=8, drug=drug_key)
        out[section] = _hits_to_dicts(hits)
        log.info(f"[retrieve:{section}] {len(out[section])} chunks")

    # Efficacy/Findings: pull SEPARATELY from the strongest tiers so RCTs and
    # regulatory evidence drive the findings, not whatever ranks highest overall.
    eff_q = f"{drug} efficacy clinical trial RCT outcomes effectiveness results benefit"
    eff_vec = embed_query(eff_q)
    eff = []
    eff += _hits_to_dicts(vectorstore.search(eff_vec, top_k=5, drug=drug_key, tier="peer_reviewed"))
    eff += _hits_to_dicts(vectorstore.search(eff_vec, top_k=4, drug=drug_key, tier="regulatory"))
    out["efficacy"] = _dedup(eff)
    log.info(f"[retrieve:efficacy] {len(out['efficacy'])} chunks (peer-reviewed + regulatory)")

    # Contradictions: deliberately retrieve BOTH sides so conflicts can surface.
    pro = embed_query(f"{drug} effective benefit reduces risk improves outcomes")
    con = embed_query(f"{drug} no benefit ineffective increased risk harm no significant difference")
    contra = []
    contra += _hits_to_dicts(vectorstore.search(pro, top_k=4, drug=drug_key))
    contra += _hits_to_dicts(vectorstore.search(con, top_k=4, drug=drug_key))
    out["contradiction"] = _dedup(contra)
    log.info(f"[retrieve:contradiction] {len(out['contradiction'])} chunks (both sides)")

    # Preprint section: dedicated pull, filtered to preprints AND this drug.
    qvec = embed_query(f"{drug} preprint emerging recent findings")
    pre_hits = vectorstore.search(qvec, top_k=5, tier="preprint", drug=drug_key)
    out["preprint"] = _hits_to_dicts(pre_hits)
    log.info(f"[retrieve:preprint] {len(out['preprint'])} preprint chunks")

    return out
