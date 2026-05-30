"""
sources/pharmgkb.py
===================
PharmGKB — curated pharmacogenomics (drug–gene interactions, dosing guidelines).

Endpoint: https://api.pharmgkb.org/v1  (open REST, no auth for read)
Flow:
  1. /data/chemical?name=<drug>            -> resolve chemical (drug) ID
  2. /data/clinicalAnnotation?relatedChemicals.accessionId=<id>
                                           -> clinical-annotation summaries
                                              (drug-gene pairs with evidence level)

Returns Evidence per clinical annotation describing the drug–gene relationship.
Tier = peer_reviewed (clinical annotations are curated from published evidence).

HONEST CAVEAT: PharmGKB is a fantastic source for drugs with known
pharmacogenomic variants (warfarin, clopidogrel, codeine, etc.); for many
drugs there will be ZERO annotations and that is normal, not an error.
"""

import requests
from core.models import Evidence, TIER_PEER_REVIEWED
from core.logging_setup import log

BASE = "https://api.pharmgkb.org/v1"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)", "Accept": "application/json"}


def _resolve_chemical(drug: str) -> str | None:
    """Return the PharmGKB accessionId (e.g. 'PA452632') for the drug, or None."""
    try:
        r = requests.get(f"{BASE}/data/chemical",
                         params={"name": drug, "view": "max"},
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        return data[0].get("id") or data[0].get("accessionId")
    except Exception as e:
        log.warning(f"[pharmgkb] chemical lookup failed: {e}")
        return None


def _get_clinical_annotations(chem_id: str, max_n: int = 8) -> list[dict]:
    try:
        r = requests.get(f"{BASE}/data/clinicalAnnotation",
                         params={"relatedChemicals.accessionId": chem_id,
                                 "view": "max"},
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        return (r.json().get("data") or [])[:max_n]
    except Exception as e:
        log.warning(f"[pharmgkb] clinical annotations failed: {e}")
        return []


def fetch(drug: str) -> list[Evidence]:
    log.info(f"[pharmgkb] looking up '{drug}' ...")
    chem_id = _resolve_chemical(drug)
    if not chem_id:
        log.info(f"[pharmgkb] no chemical record for '{drug}'")
        return []
    annotations = _get_clinical_annotations(chem_id)
    if not annotations:
        log.info(f"[pharmgkb] no pharmacogenomic annotations for '{drug}' "
                 f"(may simply have no known PGx variants)")
        return []

    out = []
    for ann in annotations:
        ann_id = ann.get("id") or ann.get("accessionId") or ""
        # Genes involved
        genes = ann.get("relatedGenes") or []
        gene_names = ", ".join(g.get("symbol", "") for g in genes if g.get("symbol"))
        # Evidence level (1A/1B/2A/2B/3/4 — PharmGKB's standard scale)
        level = ann.get("levelOfEvidence", {}).get("term", "") if isinstance(
            ann.get("levelOfEvidence"), dict) else (ann.get("levelOfEvidence") or "")
        # Phenotype / summary text
        phenotype = ann.get("phenotypeCategory") or ""
        summary = ann.get("summaryMarkdown", {}).get("markdown", "") if isinstance(
            ann.get("summaryMarkdown"), dict) else (ann.get("summary") or "")

        parts = [f"Drug–gene pair: {drug.title()}"
                 + (f" / {gene_names}" if gene_names else ""),
                 f"PharmGKB evidence level: {level or 'unspecified'}"]
        if phenotype:
            parts.append(f"Phenotype category: {phenotype}")
        if summary:
            # Trim long summaries to keep evidence chunks focused.
            parts.append(summary[:1200])

        text = "\n".join(parts)
        out.append(Evidence(
            source="pharmgkb",
            source_id=str(ann_id),
            title=f"PharmGKB: {drug.title()} – {gene_names or 'pharmacogenomic annotation'}",
            text=text,
            url=f"https://www.pharmgkb.org/clinicalAnnotation/{ann_id}" if ann_id else
                "https://www.pharmgkb.org/",
            tier=TIER_PEER_REVIEWED,
            doc_type="pharmacogenomics",
            date=None,
            extra={"section": "use in specific populations",
                   "evidence_level": level, "genes": gene_names},
        ))
    log.info(f"[pharmgkb] returned {len(out)} pharmacogenomic annotation(s) for '{drug}'")
    return out
