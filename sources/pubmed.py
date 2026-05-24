"""
sources/pubmed.py
==================
PubMed fetch agent.

Contract:  fetch(drug: str) -> list[Evidence]

Uses NCBI E-utilities (free). An NCBI_API_KEY in .env raises the rate limit
but is optional. Returns peer-reviewed papers as Evidence objects.
"""

import time
import requests
import xml.etree.ElementTree as ET

from core.models import Evidence, TIER_PEER_REVIEWED
from core.config import config
from core.logging_setup import log

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI asks clients to identify themselves; also helps avoid 403s.
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype; mailto:you@example.com)"}


def _search_pmids(drug: str, retmax: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": drug,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    r = requests.get(ESEARCH, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def _fetch_details(pmids: list[str]) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    r = requests.get(EFETCH, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def _text(el) -> str:
    """Flatten an XML element's text (handles nested tags in abstracts)."""
    return "".join(el.itertext()).strip() if el is not None else ""


def fetch(drug: str, retmax: int = 30) -> list[Evidence]:
    """Search PubMed for `drug` and return up to `retmax` papers as Evidence."""
    log.info(f"[pubmed] searching for {drug!r} ...")
    try:
        pmids = _search_pmids(drug, retmax)
    except Exception as e:
        log.warning(f"[pubmed] search failed: {e}")
        return []

    if not pmids:
        log.info(f"[pubmed] no results for {drug!r}")
        return []

    time.sleep(0.34)  # be polite to NCBI (~3 req/s without key)
    try:
        xml = _fetch_details(pmids)
    except Exception as e:
        log.warning(f"[pubmed] fetch failed: {e}")
        return []

    out: list[Evidence] = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        log.warning(f"[pubmed] XML parse error: {e}")
        return []

    for art in root.findall(".//PubmedArticle"):
        pmid = _text(art.find(".//PMID"))
        title = _text(art.find(".//ArticleTitle"))

        abstract_parts = [_text(a) for a in art.findall(".//Abstract/AbstractText")]
        abstract = " ".join(p for p in abstract_parts if p)

        year = _text(art.find(".//PubDate/Year")) or _text(art.find(".//PubDate/MedlineDate"))

        if not (title or abstract):
            continue

        out.append(Evidence(
            source="pubmed",
            source_id=pmid,
            title=title or "(no title)",
            text=abstract or "(no abstract available)",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            tier=TIER_PEER_REVIEWED,
            doc_type="paper",
            date=year or None,
        ))

    log.info(f"[pubmed] returned {len(out)} papers for {drug!r}")
    return out
