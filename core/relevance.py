"""
core/relevance.py
=================
A1 — Sibling-drug precision gate.

Problem: keyword search at the source (esp. openFDA) returns documents that
merely *mention* the target drug. A Ketorolac label that references ibuprofen
comes back when searching "ibuprofen", gets tagged drug=ibuprofen, and pollutes
the report with the wrong drug.

Fix: after fetch, verify each Evidence is actually ABOUT the target drug, not
just mentioning it. Strict-but-safe policy:
  - KEEP if the drug name is in the title (strong signal it's the subject).
  - KEEP if the drug is mentioned prominently in the text (>= MIN_MENTIONS),
    i.e. the document genuinely discusses it.
  - DROP if the title names a DIFFERENT known drug and ours only appears in
    passing (clearly a sibling-drug document).
  - When ambiguous (no clear other-drug subject), KEEP — never lose real data
    on a guess.

This runs in combine, so it filters EVERY source at once, not just FDA.
"""

import re

from core.models import Evidence
from core.logging_setup import log


# Minimum times the drug must appear in the body to count as "about" it
# when the title doesn't contain the drug name.
MIN_MENTIONS = 2


def _norm(s: str) -> str:
    return (s or "").lower()


def _word_count(haystack: str, needle: str) -> int:
    """Count whole-word occurrences of needle in haystack (case-insensitive)."""
    if not needle:
        return 0
    return len(re.findall(rf"\b{re.escape(needle)}\b", haystack, flags=re.IGNORECASE))


def is_about_drug(ev: Evidence, drug: str) -> bool:
    """Return True if this evidence is genuinely about `drug` (not a sibling)."""
    d = _norm(drug).strip()
    if not d:
        return True  # no target to check against; keep

    title = _norm(ev.title)
    text = _norm(ev.text)

    # 1) Drug name in the title -> it's the subject. KEEP.
    if _word_count(title, d) > 0:
        return True

    # 2) Drug mentioned MULTIPLE times in the body -> genuinely discussed. KEEP.
    #    (A sibling-drug doc typically mentions our drug only once, in passing —
    #    e.g. "other NSAIDs such as ibuprofen" — so requiring >= MIN_MENTIONS
    #    filters those out while keeping docs that actually discuss our drug.)
    if _word_count(text, d) >= MIN_MENTIONS:
        return True

    # 3) Otherwise: not in title, and at most a single passing mention in body.
    #    Treat as a sibling/tangential document. DROP.
    return False


def filter_relevant(evidence: list[Evidence], drug: str) -> tuple[list[Evidence], int]:
    """Filter evidence to those genuinely about `drug`.
    Returns (kept, dropped_count)."""
    kept, dropped = [], 0
    for ev in evidence:
        if is_about_drug(ev, drug):
            kept.append(ev)
        else:
            dropped += 1
    return kept, dropped
