"""
report/sanity.py
================
Numeric sanity pass — a GUARDRAIL against dangerous/implausible dosing numbers
in the finished report.

WHY: LLMs occasionally misread OTC label tables and emit absurd dosing figures
(observed: "not to exceed 48 tablets in 24 hours" for aspirin — a toxic dose).
For a drug tool, a confidently-stated dangerous dose is worse than no tool. This
pass scans the GENERATED report text and:
  1. flags implausible per-day tablet/unit counts (no oral OTC analgesic is dosed
     at dozens of tablets/day),
  2. flags internal contradictions (e.g. adult daily max wildly exceeding the
     elderly daily max for the same drug),
and annotates the offending lines with a visible warning rather than silently
"fixing" them (we don't invent the correct number — we tell the reader to verify).

HONEST SCOPE: this is a guardrail for *egregious* numeric errors, not a pharmacist.
It catches obvious garbage; it does not validate every clinically-nuanced dose.
"""

import re

# Plausibility ceiling: number of discrete oral units (tablets/capsules) per day.
# Even high-dose regimens rarely exceed ~12 tablets/day for common oral drugs;
# we set a conservative ceiling above which a value is almost certainly a misread.
MAX_PLAUSIBLE_UNITS_PER_DAY = 16

# Phrases that introduce a per-day maximum.
_PER_DAY_MAX_RE = re.compile(
    r"(?:not\s+to\s+exceed|do\s+not\s+exceed|maximum\s+of|no\s+more\s+than|up\s+to)"
    r"\s+(\d{1,3})\s+(tablets?|capsules?|caplets?|pills?)"
    r"\s+(?:in|per|/|a)\s*(?:24\s*hours?|day|daily)", re.I)

WARN = "  \n   ⚠️ **[Aletheon sanity check: this dosing figure looks implausible — verify against the official label before relying on it.]**"


def _find_per_day_maxima(text: str) -> list[tuple]:
    """Return list of (match_obj, count, unit) for 'not to exceed N tablets/day'."""
    out = []
    for m in _PER_DAY_MAX_RE.finditer(text):
        out.append((m, int(m.group(1)), m.group(2).lower()))
    return out


def check_report(report_md: str) -> tuple[str, list[str]]:
    """
    Scan the report for implausible dosing. Returns (annotated_report, warnings).
    - Annotates each implausible per-day maximum line with a visible warning.
    - Detects the adult-vs-elderly contradiction and appends a note.
    Does NOT alter the underlying numbers (we never invent a 'correct' dose).
    """
    warnings = []
    maxima = _find_per_day_maxima(report_md)

    # 1) Implausibly high per-day unit counts.
    flagged_spans = []
    for m, count, unit in maxima:
        if count > MAX_PLAUSIBLE_UNITS_PER_DAY:
            warnings.append(
                f"Implausible dosing: 'not to exceed {count} {unit} in 24h' "
                f"exceeds the plausibility ceiling ({MAX_PLAUSIBLE_UNITS_PER_DAY}).")
            flagged_spans.append(m.end())

    # 2) Internal contradiction: a much higher general max than an elderly max.
    #    If two per-day maxima exist and one is >3x the other, that's suspicious
    #    (a general adult max should never dwarf the elderly max by that margin).
    counts = sorted({c for _, c, _ in maxima})
    if len(counts) >= 2 and counts[-1] > 3 * counts[0]:
        warnings.append(
            f"Dosing inconsistency: report contains both a {counts[0]}/day and a "
            f"{counts[-1]}/day maximum — these conflict; verify against the label.")

    # Insert warnings after each flagged line (work back-to-front to keep offsets).
    annotated = report_md
    for end in sorted(flagged_spans, reverse=True):
        # find end of the line containing this match
        line_end = annotated.find("\n", end)
        if line_end == -1:
            line_end = len(annotated)
        annotated = annotated[:line_end] + WARN + annotated[line_end:]

    return annotated, warnings
