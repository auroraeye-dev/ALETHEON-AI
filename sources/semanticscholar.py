"""
sources/semanticscholar.py
==========================
Semantic Scholar Graph API fetch agent.

Why this source matters:
    The audit benchmark identified Elicit's coverage advantage as the largest
    single gap (audit dimension #4: 138M paper baseline). Semantic Scholar is
    the SAME underlying corpus Elicit builds on — 214M+ papers across all
    scientific disciplines, via the Allen Institute for AI. Adding this source
    closes the data-scope gap structurally, not by handwaving.

    More importantly for Aletheon's specific case: Semantic Scholar's index
    reaches further back than Europe PMC's clinical-leaning ranking. Older
    comparative trials (Cooper 1977, Bloomfield 1974, the classic ibu-vs-asp
    pain studies) and broader cross-discipline comparison papers live here.
    These are exactly the papers our screening gate has been starved for.

API basics (verified June 2026):
    Endpoint:    GET https://api.semanticscholar.org/graph/v1/paper/search
    Auth:        Optional (header `x-api-key: $KEY`). Free without a key.
    Rate limit:  Unauthenticated: 100 requests / 5 min (shared pool).
                 With API key:    1 RPS dedicated (introductory).
    Response:    JSON. Top-level `data: [...]` array of paper records.

Contract (same as every other source in /sources):
    fetch(drug: str) -> list[Evidence]

Tier:
    All returned papers are tagged TIER_PEER_REVIEWED. Semantic Scholar does
    surface preprints (arXiv, bioRxiv) but the API doesn't reliably mark them
    in a single field across all venues. We err toward peer_reviewed for the
    main flow; the screening gate later filters comparative claims anyway, so
    a tier misclassification here doesn't corrupt the synthesis.

Failure mode:
    On any error (network, 429, schema change), this source logs and returns
    []. The combine pipeline already swallows source-level exceptions, so a
    Semantic Scholar outage cannot break the whole run.
"""

import os
import time

import requests

from core.config import config
from core.logging_setup import log
from core.models import Evidence, TIER_PEER_REVIEWED

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Only ask for the fields we actually use. The API tutorial explicitly warns
# that overlong fields lists slow the response — and unused fields would just
# get dropped by our Evidence model anyway.
S2_FIELDS = (
    "paperId,title,abstract,year,authors,citationCount,"
    "externalIds,openAccessPdf,publicationTypes,venue"
)

HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)"}


def _get_api_key() -> str | None:
    """API key is optional. If SEMANTIC_SCHOLAR_API_KEY is set in env, use it
    for the 1 RPS dedicated rate limit. Otherwise we run unauthenticated on
    the shared 100-requests-per-5-min pool."""
    return os.getenv("SEMANTIC_SCHOLAR_API_KEY")


def _build_url(query: str, limit: int) -> tuple[str, dict]:
    """Build the search URL and headers (with API key if available)."""
    params = {"query": query, "limit": limit, "fields": S2_FIELDS}
    headers = dict(HEADERS)
    api_key = _get_api_key()
    if api_key:
        headers["x-api-key"] = api_key
    return params, headers


def _to_evidence(item: dict) -> Evidence | None:
    """Map one Semantic Scholar paper record to an Evidence row. Returns None
    if the paper lacks a usable title or text (in which case it'd just become
    a ✗ failed extraction downstream — drop it now)."""
    title = (item.get("title") or "").strip()
    abstract = (item.get("abstract") or "").strip()
    if not title or not abstract:
        # Without an abstract the extraction step has nothing to work with.
        # Drop the paper now rather than letting it pollute the evidence base.
        return None

    paper_id = item.get("paperId", "")
    if not paper_id:
        return None

    # Build the canonical URL. Semantic Scholar paper pages use the paperId.
    url = f"https://www.semanticscholar.org/paper/{paper_id}"

    # Prefer a DOI as the source_id when available — it links cleanly across
    # systems (and keeps the [tag] referenceable to a real-world identifier
    # the reader can verify). Fall back to paperId.
    ext = item.get("externalIds") or {}
    doi = ext.get("DOI") or ext.get("doi")
    # S2's externalIds shape isn't fully consistent across endpoint versions.
    # PubMed identifier appears as 'PubMed' or 'PMID' depending on the record.
    pmid = ext.get("PubMed") or ext.get("PMID") or ext.get("PubMedCentral")
    source_id = doi or pmid or paper_id

    year = item.get("year")
    date = f"{year}-01-01" if year else None

    # Pull a few useful extras the screening LLM and the reader benefit from.
    authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
    extras = {
        "paper_id": paper_id,
        "citation_count": item.get("citationCount") or 0,
        "authors": authors[:6],            # cap — reports don't need 50 author names
        "venue": item.get("venue") or "",
        "publication_types": item.get("publicationTypes") or [],
    }
    if doi:
        extras["doi"] = doi
    if pmid:
        extras["pmid"] = pmid

    return Evidence(
        source="semanticscholar",
        source_id=str(source_id),
        title=title,
        text=abstract,
        url=url,
        tier=TIER_PEER_REVIEWED,
        doc_type="paper",
        date=date,
        extra=extras,
    )


def fetch(drug: str, limit: int = None) -> list[Evidence]:
    """Search Semantic Scholar for `drug`, return papers as Evidence.

    Two-pass strategy mirroring europepmc.py:
      Pass 1 (relevance): default S2 relevance ranking surfaces the most
              topically-aligned papers (S2 has a learned relevance model — it's
              one of the reasons Elicit uses this corpus).
      We deliberately do NOT also run a citation-sort pass here — S2's relevance
      ranking already incorporates citation signal internally, and we'd waste
      half the rate-limit budget on it.
    """
    limit = limit or getattr(config, "SEMANTIC_SCHOLAR_PAGE_SIZE", 50)

    log.info(f"[semanticscholar] searching for {drug!r} (limit={limit}) ...")

    params, headers = _build_url(drug, limit)

    data = None
    for attempt in range(1, 4):
        try:
            r = requests.get(S2_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                # Rate-limited. Back off and retry (longer each time).
                wait = 2 * attempt
                log.warning(f"[semanticscholar] rate-limited (429), "
                            f"backing off {wait}s and retrying")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            n_in_page = len(data.get("data", []) or [])
            log.info(f"[semanticscholar] attempt {attempt}: "
                     f"total={data.get('total', 'N/A')}, "
                     f"results_in_page={n_in_page}")
            if n_in_page:
                break
        except Exception as e:
            log.warning(f"[semanticscholar] attempt {attempt} failed: {e}")
        time.sleep(1.0)

    if not data:
        log.warning(f"[semanticscholar] no response after retries for {drug!r}")
        return []

    raw = data.get("data") or []
    out: list[Evidence] = []
    skipped_no_abstract = 0
    for item in raw:
        ev = _to_evidence(item)
        if ev is None:
            skipped_no_abstract += 1
            continue
        out.append(ev)

    if skipped_no_abstract:
        log.info(f"[semanticscholar] skipped {skipped_no_abstract} paper(s) "
                 f"with no abstract (would have produced ✗ failed extractions)")
    log.info(f"[semanticscholar] returned {len(out)} usable paper(s) for {drug!r}")
    return out
