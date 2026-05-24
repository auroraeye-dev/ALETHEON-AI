"""
sources/fda.py
==============
FDA drug-label fetch agent (openFDA). Public API, no key needed.

Contract:  fetch(drug: str) -> list[Evidence]
Tier: regulatory (FDA labels are authoritative regulatory text).

Ported from your working MVP2 fda_s1_data_ingestion.py, reshaped to Evidence.
Each label becomes ONE Evidence with the useful sections concatenated, so the
chunker can split it on natural boundaries later.
"""

import requests

from core.models import Evidence, TIER_REGULATORY
from core.logging_setup import log

FDA_URL = "https://api.fda.gov/drug/label.json"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)"}

# Label sections worth pulling into the evidence text, in a sensible order.
SECTIONS = [
    ("indications_and_usage", "Indications and Usage"),
    ("purpose", "Purpose"),
    ("dosage_and_administration", "Dosage and Administration"),
    ("warnings", "Warnings"),
    ("warnings_and_cautions", "Warnings and Cautions"),
    ("adverse_reactions", "Adverse Reactions"),
    ("contraindications", "Contraindications"),
    ("drug_interactions", "Drug Interactions"),
    ("description", "Description"),
]


def _first(value) -> str:
    """openFDA fields are usually lists of strings; take the first, safely."""
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def _label_name(item: dict, fallback: str) -> str:
    of = item.get("openfda", {})
    brand = _first(of.get("brand_name"))
    generic = _first(of.get("generic_name"))
    name = brand or generic or fallback
    return name


def _label_id(item: dict, idx: int) -> str:
    of = item.get("openfda", {})
    # set_id / spl id is the stable identifier for an FDA label
    sid = _first(of.get("spl_set_id")) or item.get("set_id") or item.get("id")
    return sid or f"fda-{idx}"


def fetch(drug: str, limit: int = 10) -> list[Evidence]:
    """Search openFDA drug labels for `drug` and return Evidence (regulatory)."""
    log.info(f"[fda] searching labels for {drug!r} ...")
    params = {"search": drug, "limit": limit}
    try:
        r = requests.get(FDA_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"[fda] request failed: {e}")
        return []

    results = data.get("results", [])
    if not results:
        log.info(f"[fda] no labels for {drug!r}")
        return []

    out: list[Evidence] = []
    for idx, item in enumerate(results, 1):
        # Build a readable body from whatever sections this label has.
        parts = []
        for key, heading in SECTIONS:
            text = _first(item.get(key)).strip()
            if text:
                parts.append(f"{heading}: {text}")
        body = "\n\n".join(parts).strip()
        if not body:
            continue  # skip labels with no usable text

        name = _label_name(item, drug)
        sid = _label_id(item, idx)
        of = item.get("openfda", {})
        effective = item.get("effective_time")  # YYYYMMDD if present

        out.append(Evidence(
            source="fda",
            source_id=str(sid),
            title=f"FDA Label: {name}",
            text=body,
            url=f"https://labels.fda.gov/" if not sid else
                f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={sid}",
            tier=TIER_REGULATORY,
            doc_type="label",
            date=str(effective) if effective else None,
            extra={
                "brand_name": _first(of.get("brand_name")),
                "generic_name": _first(of.get("generic_name")),
                "manufacturer": _first(of.get("manufacturer_name")),
            },
        ))

    log.info(f"[fda] returned {len(out)} labels for {drug!r}")
    return out
