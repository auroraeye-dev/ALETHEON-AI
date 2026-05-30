"""
report/retraction_check.py
==========================
Retraction-Watch checker — flags retracted papers cited in a report.

NOT A SOURCE: this is a quality filter that runs on already-retrieved evidence
and surfaces any item whose DOI has been retracted, corrected, or had a major
editorial concern issued. Retraction Watch / Crossref expose this via the
`update-to` relationship on the Crossref API.

For a tool whose brand is evidence honesty, citing a retracted paper without
flagging it is exactly the kind of thing that destroys trust. This catches it.

Flow:
  - Take the evidence list, pick out items with a DOI we can identify (Europe
    PMC preprints/papers, ClinicalTrials cross-references).
  - For each, query Crossref: https://api.crossref.org/works/{doi}
    Look at the `update-to` array — entries with type "retraction",
    "withdrawal", "correction", or "expression_of_concern" are flags.
  - Return a list of {tag, source_id, issue, original_doi} for the report's
    annotation layer (and a markdown block for the appendix).

LICENSING: Retraction Watch data is licensed for non-commercial use. We rely on
Crossref's free API which exposes the retraction relationship. Before commercial
launch, verify license terms (logged in deployment notes).
"""

import re
import requests

from core.logging_setup import log

CROSSREF = "https://api.crossref.org/works"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype; mailto:contact@aletheon.local)"}

# DOI extraction — only flagged-issue types are returned.
_FLAG_TYPES = {"retraction", "withdrawal", "correction", "expression_of_concern"}

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"']+", re.I)


def _extract_doi(ev_dict: dict) -> str | None:
    """Pull a DOI from an evidence dict (checks source_id and url)."""
    for field in ("source_id", "url"):
        val = ev_dict.get(field, "") or ""
        m = _DOI_RE.search(val)
        if m:
            # Strip trailing punctuation that often clings to DOIs in URLs
            return m.group(0).rstrip(".,;)/")
    return None


def _check_doi(doi: str) -> list[dict]:
    """Return a list of flagged update entries for this DOI (empty if clean)."""
    try:
        r = requests.get(f"{CROSSREF}/{doi}", headers=HEADERS, timeout=15)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        msg = r.json().get("message", {})
        updates = msg.get("update-to") or []
        flags = []
        for u in updates:
            t = (u.get("type") or "").lower().replace(" ", "_")
            if t in _FLAG_TYPES:
                flags.append({
                    "type": t,
                    "label": u.get("label") or t.replace("_", " ").title(),
                    "doi": u.get("DOI", ""),
                })
        return flags
    except Exception as e:
        log.warning(f"[retraction] crossref lookup failed for {doi}: {e}")
        return []


def check_evidence(evidence_items: list[dict]) -> list[dict]:
    """Scan evidence items, return a list of issues found:
    {tag, source_id, doi, issues:[{type,label,doi}]}."""
    issues = []
    seen_dois = set()
    for ev in evidence_items:
        doi = _extract_doi(ev)
        if not doi or doi in seen_dois:
            continue
        seen_dois.add(doi)
        flags = _check_doi(doi)
        if flags:
            issues.append({
                "tag": ev.get("tag", ""),
                "source_id": ev.get("source_id", ""),
                "title": ev.get("title", ""),
                "doi": doi,
                "issues": flags,
            })
    if issues:
        log.warning(f"[retraction] flagged {len(issues)} retracted/corrected/"
                    f"concerning paper(s) in the evidence")
    return issues


def format_retraction_block(issues: list[dict]) -> str:
    """Render a clearly-labeled warning section for the report appendix."""
    if not issues:
        return ""
    lines = ["## ⚠️ Retraction / Correction Notices",
             "_The following cited papers have a flagged status on Crossref "
             "(retraction, withdrawal, correction, or expression of concern). "
             "Treat any claim citing these with extra caution and verify against "
             "the publisher's notice._\n"]
    for it in issues:
        flag_str = "; ".join(f"**{f['label']}**" for f in it["issues"])
        lines.append(f"- {it.get('tag','')} {it.get('title','')[:100]} "
                     f"({it['doi']}) — {flag_str}")
    return "\n".join(lines)
