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
    """Fetch one SPL XML and emit one Evidence per meaningful labeled section.

    Two important behaviors:
      1. We skip WRAPPER sections that have nested <section> children — their
         body would otherwise duplicate the children (8 USE IN SPECIFIC
         POPULATIONS contains the text of 8.1 Pregnancy, 8.2 Lactation, etc.,
         which then appear as standalone sections too). Emitting only leaf
         sections gives each piece of label content exactly one evidence row.
      2. The source_id includes a slug of the section title so each section is
         a DISTINCT evidence row at dedup time. Without this, all sections of
         one SPL share the same setid as source_id and dedup collapses 78
         sections down to 1 (whichever XML-ordering put first — typically not
         pregnancy or PK). This was the root cause of empty Pregnancy / PK /
         Populations sections in reports.
    """
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
    seen_slugs: set[str] = set()
    # Each labeled section: <section> with a <title> and content.
    for sec in root.iter("{urn:hl7-org:v3}section"):
        # Skip wrapper sections that contain nested <section> children — their
        # text would duplicate the leaf-level children we'll emit separately.
        nested = sec.findall("v3:section", SPL_NS)
        if nested:
            continue
        t_el = sec.find("v3:title", SPL_NS)
        sec_title = _strip("".join(t_el.itertext())) if t_el is not None else ""
        body = _section_text(sec)
        # skip tiny/empty sections and the title-only wrappers
        if not body or len(body) < 40:
            continue
        # normalize a clean section label (strip leading numbers like "12.1")
        clean = re.sub(r"^[\d.\s]+", "", sec_title).strip().lower()
        # build a slug for the source_id so each section is uniquely identified
        # in the dedup step downstream. Falls back to a numeric counter if the
        # title is empty (rare).
        slug = re.sub(r"\W+", "_", clean).strip("_")[:50] or f"sec{len(out)}"
        # Disambiguate slug collisions within the same SPL (e.g. two
        # sub-sections both titled "Risk Summary" under different parents).
        original_slug = slug
        counter = 2
        while slug in seen_slugs:
            slug = f"{original_slug}_{counter}"
            counter += 1
        seen_slugs.add(slug)
        ev = Evidence(
            source="dailymed",
            source_id=f"{setid}#{slug}",
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