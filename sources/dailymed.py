"""
sources/dailymed.py
===================
C5 — DailyMed (full FDA prescription labels / SPL).

WHY THIS MATTERS: openFDA's label endpoint often returns OTC consumer
"drug-facts" labels, which lack Mechanism of Action, Clinical Pharmacology,
full Drug Interactions, and Use in Specific Populations sections. DailyMed
serves the FULL prescription Structured Product Labels (SPL) that DO contain
those sections — so this is the source that gives B2's monograph sections real
depth.

Flow:
  1. /spls.json?drug_name=<drug>&name_type=generic  -> list of SPL SETIDs
     (we prefer human prescription labels — they have the rich sections).
  2. /spls/{SETID}.xml  -> the full SPL XML; we extract each <section>'s
     title + text and emit them, tagged `regulatory`, with the section name
     set so the chunker/B2 retrieval can target them.

Contract:  fetch(drug: str) -> list[Evidence]
Tier: regulatory.
"""

import re
import requests
import xml.etree.ElementTree as ET

from core.models import Evidence, TIER_REGULATORY
from core.logging_setup import log

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)"}
SPL_NS = {"v3": "urn:hl7-org:v3"}

# LOINC section codes for the high-value monograph sections we most want.
# (We keep ALL sections, but use these to prioritize / label cleanly.)
SECTION_TITLE_HINTS = [
    "mechanism of action", "clinical pharmacology", "indications",
    "dosage and administration", "contraindications", "warnings",
    "precautions", "adverse reactions", "drug interactions",
    "use in specific populations", "overdosage", "pharmacokinetics",
    "pharmacodynamics",
]


def _strip(text: str) -> str:
    """Collapse whitespace from extracted XML text."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _section_text(el) -> str:
    """Concatenate all visible text under an SPL <section> element."""
    parts = []
    for node in el.iter():
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        if node.tail and node.tail.strip():
            parts.append(node.tail.strip())
    return _strip(" ".join(parts))


def _list_spl_setids(drug: str, max_labels: int) -> list[str]:
    """Return SETIDs for prescription SPLs matching the drug (generic name)."""
    params = {"drug_name": drug, "name_type": "generic", "pagesize": 100}
    try:
        r = requests.get(f"{BASE}/spls.json", params=params, headers=HEADERS, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"[dailymed] spls lookup failed: {e}")
        return []

    rows = data.get("data", []) or []
    setids = []
    for row in rows:
        sid = row.get("setid") or row.get("SETID")
        title = (row.get("title") or row.get("TITLE") or "")
        if not sid:
            continue
        # Prefer human prescription labels — they carry the rich sections.
        # SPL titles usually look like "NAME (INGREDIENT) TABLET [MANUFACTURER]".
        setids.append(sid)
        if len(setids) >= max_labels:
            break
    return setids


def _parse_spl(setid: str, drug: str) -> list[Evidence]:
    """Fetch one SPL XML and emit one Evidence per meaningful labeled section."""
    url = f"{BASE}/spls/{setid}.xml"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning(f"[dailymed] SPL {setid} fetch/parse failed: {e}")
        return []

    # Document title (drug + manufacturer).
    title_el = root.find(".//v3:title", SPL_NS)
    doc_title = _strip("".join(title_el.itertext())) if title_el is not None else drug
    doc_title = doc_title[:160] if doc_title else f"DailyMed label: {drug}"

    out = []
    # Each labeled section: <section> with a <title> and content.
    for sec in root.iter("{urn:hl7-org:v3}section"):
        t_el = sec.find("v3:title", SPL_NS)
        sec_title = _strip("".join(t_el.itertext())) if t_el is not None else ""
        body = _section_text(sec)
        # skip tiny/empty sections and the title-only wrappers
        if not body or len(body) < 40:
            continue
        # normalize a clean section label (strip leading numbers like "12.1")
        clean = re.sub(r"^[\d.\s]+", "", sec_title).strip().lower()
        # keep sections that are substantive; everything substantive is useful
        ev = Evidence(
            source="dailymed",
            source_id=f"{setid}",
            title=f"DailyMed: {doc_title}",
            text=(f"{sec_title}\n{body}" if sec_title else body),
            url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}",
            tier=TIER_REGULATORY,
            doc_type="prescription_label",
            date=None,
            extra={"section": clean, "setid": setid},
        )
        out.append(ev)
    return out


def fetch(drug: str, max_labels: int = None) -> list[Evidence]:
    from core.config import config
    max_labels = max_labels or getattr(config, "DAILYMED_MAX_LABELS", 3)

    log.info(f"[dailymed] searching prescription labels for {drug!r} ...")
    setids = _list_spl_setids(drug, max_labels)
    if not setids:
        log.info(f"[dailymed] no SPLs found for {drug!r}")
        return []

    all_ev = []
    for sid in setids:
        all_ev.extend(_parse_spl(sid, drug))
    log.info(f"[dailymed] returned {len(all_ev)} labeled sections "
             f"from {len(setids)} prescription label(s) for {drug!r}")
    return all_ev
