"""
sources/europepmc.py
====================
Europe PMC fetch agent. Official, stable REST API (no scraping, no API key).

Contract:  fetch(drug: str) -> list[Evidence]

Europe PMC indexes BOTH peer-reviewed literature AND preprints (bioRxiv,
medRxiv). We detect preprints and tag them correctly:
  - preprint  -> tier = "preprint"     (NOT peer-reviewed)
  - otherwise -> tier = "peer_reviewed"

This single agent therefore gives you your first preprint evidence (lighting up
the report's Preprint section) AND extra peer-reviewed coverage.
"""

import requests

from core.models import Evidence, TIER_PREPRINT, TIER_PEER_REVIEWED
from core.logging_setup import log

EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)"}


def _is_preprint(item: dict) -> bool:
    """Detect preprints via source/pubType signals."""
    src = (item.get("source") or "").upper()      # e.g. "PPR" = preprint
    pub_types = " ".join(item.get("pubTypeList", {}).get("pubType", [])) \
        if isinstance(item.get("pubTypeList"), dict) else ""
    text = (src + " " + (item.get("pubType") or "") + " " + pub_types).lower()
    return "ppr" in src.lower() or "preprint" in text


def _best_id(item: dict) -> tuple[str, str]:
    """Return (id, url) using the best available identifier."""
    pmid = item.get("pmid")
    pmcid = item.get("pmcid")
    doi = item.get("doi")
    if pmid:
        return pmid, f"https://europepmc.org/article/MED/{pmid}"
    if pmcid:
        return pmcid, f"https://europepmc.org/article/PMC/{pmcid}"
    if doi:
        return doi, f"https://doi.org/{doi}"
    return item.get("id", "unknown"), "https://europepmc.org/"


def _run_query(params: dict, label: str) -> list[dict]:
    """Run one Europe PMC query with retries. Returns the result list (may be empty)."""
    data = None
    for attempt in range(1, 4):
        try:
            r = requests.get(EPMC_URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            results_now = data.get("resultList", {}).get("result", [])
            log.info(f"[europepmc:{label}] attempt {attempt}: "
                     f"hitCount={data.get('hitCount', 0)}, "
                     f"results_in_page={len(results_now)}")
            if results_now:
                return results_now
        except Exception as e:
            log.warning(f"[europepmc:{label}] attempt {attempt} failed: {e}")
        import time as _t
        _t.sleep(1.0)
    if data is None:
        log.warning(f"[europepmc:{label}] no response after retries")
    return []


def fetch(drug: str, page_size: int = None) -> list[Evidence]:
    """Search Europe PMC for `drug`, return papers + preprints as Evidence.

    Strategy (step 4 of the synthesis upgrade — better retrieval, not more sources):
    - Filter: HAS_ABSTRACT:Y forces papers we can actually extract from. Drops
      the chemistry/wastewater/topology junk that lacks an abstract or has only
      a stub. Cuts garbage roughly in half on clinical queries.
    - TWO parallel searches: one ranked by citation count (CITED desc — surfaces
      foundational comparison reviews like Cates 2013), one ranked by date
      (P_PDATE_D desc — surfaces recent evidence). Merge and dedupe.
    - This is "best of both worlds" — established + emerging — and roughly
      mirrors how Cochrane/Elicit-style tools combine relevance and recency.
    """
    from core.config import config
    page_size = page_size or config.EUROPEPMC_PAGE_SIZE

    log.info(f"[europepmc] searching for {drug!r} (citation + recency, "
             f"pageSize={page_size}) ...")

    # Common: require an abstract (so extraction has something to work with).
    # Append HAS_ABSTRACT:Y to the user query in Europe PMC's Lucene-like syntax.
    base_q = f'({drug}) AND HAS_ABSTRACT:Y'

    # Run-1: most-cited papers first. Surfaces the foundational clinical literature.
    cited_results = _run_query(
        {
            "query": base_q,
            "format": "json",
            "pageSize": page_size,
            "resultType": "core",
            "sort": "CITED desc",
        },
        label="cited",
    )

    # Run-2: most recent papers first. Surfaces emerging evidence the cited
    # query misses (since new papers haven't accumulated citations yet).
    recent_results = _run_query(
        {
            "query": base_q,
            "format": "json",
            "pageSize": max(20, page_size // 2),  # smaller — citation is the higher-signal stream
            "resultType": "core",
            "sort": "P_PDATE_D desc",
        },
        label="recent",
    )

    # Merge, preserving cited-first order (so cited papers win the dedupe and
    # appear earlier in the final list — they get higher chunk-priority in the
    # vector store too).
    seen = set()
    results: list[dict] = []
    for batch in (cited_results, recent_results):
        for item in batch:
            sid, _ = _best_id(item)
            if sid in seen:
                continue
            seen.add(sid)
            results.append(item)

    if not results:
        log.info(f"[europepmc] no results for {drug!r} after filters")
        return []

    out: list[Evidence] = []
    n_preprint = 0
    for item in results:
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstractText") or "").strip()
        if not (title or abstract):
            continue

        preprint = _is_preprint(item)
        if preprint:
            n_preprint += 1
        tier = TIER_PREPRINT if preprint else TIER_PEER_REVIEWED

        sid, url = _best_id(item)
        date = item.get("firstPublicationDate") or item.get("pubYear")

        body = title
        if abstract:
            body += "\n\n" + abstract

        out.append(Evidence(
            source="europepmc",
            source_id=str(sid),
            title=title or "(no title)",
            text=body,
            url=url,
            tier=tier,
            doc_type="preprint" if preprint else "paper",
            date=str(date) if date else None,
            extra={
                "journal": item.get("journalTitle", ""),
                "is_preprint": preprint,
                "source_db": item.get("source", ""),
            },
        ))

    log.info(f"[europepmc] returned {len(out)} results for {drug!r} "
             f"({n_preprint} preprints, {len(out) - n_preprint} peer-reviewed)")
    return out
