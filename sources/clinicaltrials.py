"""
sources/clinicaltrials.py
=========================
ClinicalTrials.gov fetch agent (API v2). Public, no key needed.

Contract:  fetch(drug: str) -> list[Evidence]
Tier: peer_reviewed (registered trials with protocol + results).

Ported from your working MVP2 scripts. Each study becomes ONE Evidence whose
text weaves together summary + conditions + interventions + OUTCOMES, so the
report can cite efficacy findings (the thing FDA labels alone couldn't give).
"""

import requests

from core.models import Evidence, TIER_PEER_REVIEWED
from core.logging_setup import log

CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)"}


def _join(items, sep=" | "):
    return sep.join(x for x in items if x)


def _build_text(ps: dict) -> str:
    """Assemble a readable evidence body from the protocol section."""
    desc = ps.get("descriptionModule", {})
    cond = ps.get("conditionsModule", {})
    interv = ps.get("armsInterventionsModule", {})
    outcome = ps.get("outcomesModule", {})
    status = ps.get("statusModule", {})

    parts = []

    summary = desc.get("briefSummary", "").strip()
    if summary:
        parts.append(f"Summary: {summary}")

    conditions = cond.get("conditions", [])
    if conditions:
        parts.append(f"Conditions studied: {_join(conditions)}")

    interventions = [
        (i.get("name", "") + (": " + i["description"] if i.get("description") else ""))
        for i in interv.get("interventions", [])
    ]
    if interventions:
        parts.append(f"Interventions: {_join(interventions)}")

    primary = [o.get("measure", "") for o in outcome.get("primaryOutcomes", [])]
    if primary:
        parts.append(f"Primary outcomes measured: {_join(primary)}")

    secondary = [o.get("measure", "") for o in outcome.get("secondaryOutcomes", [])]
    if secondary:
        parts.append(f"Secondary outcomes measured: {_join(secondary)}")

    st = status.get("overallStatus", "")
    if st:
        parts.append(f"Trial status: {st}")

    return "\n\n".join(parts).strip()


def fetch(drug: str, page_size: int = None) -> list[Evidence]:
    from core.config import config
    page_size = page_size or config.CLINICALTRIALS_PAGE_SIZE
    """Search ClinicalTrials.gov for `drug` and return trials as Evidence."""
    log.info(f"[clinicaltrials] searching for {drug!r} ...")
    params = {"query.term": drug, "pageSize": page_size}
    try:
        r = requests.get(CTGOV_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"[clinicaltrials] request failed: {e}")
        return []

    studies = data.get("studies", [])
    if not studies:
        log.info(f"[clinicaltrials] no trials for {drug!r}")
        return []

    out: list[Evidence] = []
    for study in studies:
        ps = study.get("protocolSection", {})
        id_mod = ps.get("identificationModule", {})
        status_mod = ps.get("statusModule", {})
        sponsor_mod = ps.get("sponsorCollaboratorsModule", {})

        nct = id_mod.get("nctId", "")
        title = id_mod.get("briefTitle", "") or id_mod.get("officialTitle", "")
        body = _build_text(ps)
        if not body or not nct:
            continue

        start = status_mod.get("startDateStruct", {}).get("date")
        lead = sponsor_mod.get("leadSponsor", {}).get("name", "")

        out.append(Evidence(
            source="clinicaltrials",
            source_id=nct,
            title=title or f"Trial {nct}",
            text=body,
            url=f"https://clinicaltrials.gov/study/{nct}",
            tier=TIER_PEER_REVIEWED,
            doc_type="trial",
            date=start,
            extra={"sponsor": lead, "has_results": study.get("hasResults", False)},
        ))

    log.info(f"[clinicaltrials] returned {len(out)} trials for {drug!r}")
    return out
