"""
sources/faers.py
================
C1 — FAERS (FDA Adverse Event Reporting System) via openFDA.

A genuinely NEW kind of evidence: real-world adverse events actually reported
for a drug — not what a label warns about (regulatory) or what a trial measured
(peer-reviewed), but what patients/clinicians have reported in practice.

Tier = real_world (between peer-reviewed and preprint in authority): it reflects
real usage, but is unverified, voluntarily reported, and has no denominator — so
it signals *what* reactions occur, not their true rate. We make that caveat
explicit in the evidence text so the report never over-claims from it.

openFDA returns AGGREGATED COUNTS of reaction terms, not prose. We turn the top
reactions into a readable evidence summary.

Contract: fetch(drug) -> list[Evidence]
"""

import requests

from core.models import Evidence, TIER_REAL_WORLD
from core.logging_setup import log

FAERS_URL = "https://api.fda.gov/drug/event.json"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)"}


def fetch(drug: str, top_n: int = None) -> list[Evidence]:
    from core.config import config
    top_n = top_n or config.FAERS_TOP_N
    """Fetch the most-reported adverse reactions for `drug` from FAERS.

    Returns a single Evidence summarizing the top reported reactions (real_world
    tier), or [] if nothing is found / the API errors.
    """
    log.info(f"[faers] fetching adverse-event reports for {drug!r} ...")
    d = drug.strip().lower()

    # Count adverse-reaction terms for reports naming this drug. We search the
    # generic and brand name fields so a brand query still hits, and use
    # openFDA's `count` to aggregate reaction terms server-side.
    search = (f'(patient.drug.openfda.generic_name:"{d}" '
              f'OR patient.drug.openfda.brand_name:"{d}")')
    params = {
        "search": search,
        "count": "patient.reaction.reactionmeddrapt.exact",
    }

    def _request(p):
        r = requests.get(FAERS_URL, params=p, headers=HEADERS, timeout=25)
        r.raise_for_status()
        return r.json()

    try:
        data = _request(params)
    except Exception as e:
        # openFDA returns 404 when a query matches nothing; try a broader search.
        log.info(f"[faers] targeted search failed ({e}); trying broad search")
        try:
            data = _request({"search": f'patient.drug.medicinalproduct:"{d}"',
                             "count": "patient.reaction.reactionmeddrapt.exact"})
        except Exception as e2:
            log.warning(f"[faers] request failed: {e2}")
            return []

    results = data.get("results", [])
    if not results:
        log.info(f"[faers] no adverse-event data for {drug!r}")
        return []

    top = results[:top_n]
    total = sum(r.get("count", 0) for r in top)

    # Build a readable summary of the top reported reactions.
    lines = [
        f"Real-world adverse-event reports for {drug} (FDA FAERS / openFDA).",
        "These are voluntarily reported reactions aggregated across reports; they "
        "indicate which adverse events have been reported in practice, but are "
        "unverified, lack a usage denominator, and do NOT establish causation or "
        "incidence rate.",
        "",
        f"Most frequently reported reactions for {drug} (by report count):",
    ]
    for r in top:
        term = (r.get("term") or "").title()
        cnt = r.get("count", 0)
        lines.append(f"- {term}: {cnt} reports")

    body = "\n".join(lines)

    ev = Evidence(
        source="faers",
        source_id=f"faers-{d}",
        title=f"FAERS adverse-event summary: {drug}",
        text=body,
        url="https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers",
        tier=TIER_REAL_WORLD,
        doc_type="adverse_event_summary",
        date=None,
        extra={"reaction_terms": len(top), "total_reports_top": total},
    )
    log.info(f"[faers] returned top {len(top)} reactions for {drug!r} "
             f"({total} reports across them)")
    return [ev]
