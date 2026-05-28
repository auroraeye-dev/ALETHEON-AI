"""
report/stats_extract.py
=======================
F3 (honest version) — statistical EXTRACTION, not inference.

IMPORTANT DESIGN STANCE: this module does NOT compute new statistics, pool effect
sizes, or run meta-analysis. Doing so from heterogeneous abstract text would
produce unvalidated, potentially wrong biostatistics — unacceptable in a drug
tool. Instead it DETECTS and SURFACES the statistical values that authors already
reported (hazard ratios, odds/risk ratios, confidence intervals, p-values, sample
sizes, percentages), each tied to the source it came from. We report what the
papers say — we do not invent numbers.

This gives a researcher the "show me the actual numbers" view without false
precision. Output is a list of {kind, value, snippet, source_id, tier} dicts.
"""

import re

# Patterns for commonly-reported statistics. Each captures the reported value
# verbatim; we never recompute. Patterns are intentionally conservative — better
# to miss a stat than to mis-extract one.
_PATTERNS = [
    # Hazard ratio / odds ratio / risk ratio / relative risk, e.g. "HR 0.82",
    # "odds ratio of 1.4", "RR=2.1"
    ("ratio", re.compile(
        r"\b(hazard ratio|odds ratio|risk ratio|relative risk|HR|OR|RR)\b"
        r"\s*(?:of|=|:|was|,)?\s*(\d+\.\d+)", re.I)),
    # Confidence interval, e.g. "95% CI 0.74-0.91", "95% CI: 0.74 to 0.91"
    ("ci", re.compile(
        r"\b(\d{2})\s*%\s*CI[:\s]*\(?\s*(\d+\.?\d*)\s*(?:-|–|to|,)\s*(\d+\.?\d*)\)?",
        re.I)),
    # p-value, e.g. "p < 0.001", "p = 0.04", "p=0.012"
    ("p_value", re.compile(r"\bp\s*(<|=|≤|>)\s*(0?\.\d+)", re.I)),
    # Sample size, e.g. "n = 1,200", "N=348", "1,245 patients", "500 subjects",
    # "enrolled 500", "included 348 participants"
    ("sample_size", re.compile(
        r"\b(?:n\s*=\s*|enrolled\s+|included\s+)(\d[\d,]{1,7})\b"
        r"|\b(\d[\d,]{2,7})\s+(?:patients|subjects|participants|individuals)\b", re.I)),
    # Percent outcome, e.g. "reduced by 23%", "incidence of 4.5%". Negative
    # lookahead avoids catching the "95%" that belongs to "95% CI".
    ("percent", re.compile(r"\b(\d+\.?\d*)\s*%(?!\s*CI)", re.I)),
]

# A cap so a single stat-dense abstract doesn't flood the output.
_MAX_PER_EVIDENCE = 6


def _context(text: str, start: int, end: int, pad: int = 45) -> str:
    """Return a short surrounding snippet for human context."""
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    snippet = text[s:e].replace("\n", " ").strip()
    return re.sub(r"\s+", " ", snippet)


def extract_from_text(text: str) -> list[dict]:
    """Find reported statistics in one piece of text. Returns list of dicts:
    {kind, value, snippet}. Verbatim — no recomputation."""
    if not text:
        return []
    found = []
    seen_spans = []

    def _overlaps(a, b):
        return any(not (b <= s or a >= e) for s, e in seen_spans)

    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            if _overlaps(m.start(), m.end()):
                continue
            # Build a clean "value" string per kind.
            if kind == "ratio":
                value = f"{m.group(1).upper()} {m.group(2)}"
            elif kind == "ci":
                value = f"{m.group(1)}% CI {m.group(2)}–{m.group(3)}"
            elif kind == "p_value":
                value = f"p {m.group(1)} {m.group(2)}"
            elif kind == "sample_size":
                num = m.group(1) or m.group(2)
                value = f"n = {num}"
            elif kind == "percent":
                value = f"{m.group(1)}%"
            else:
                value = m.group(0)
            found.append({
                "kind": kind,
                "value": value,
                "snippet": _context(text, m.start(), m.end()),
            })
            seen_spans.append((m.start(), m.end()))
            if len(found) >= _MAX_PER_EVIDENCE:
                return found
    return found


def extract_from_evidence(evidence_items: list[dict]) -> list[dict]:
    """Given retrieved evidence (each a dict with text/source_id/tier), return a
    flat list of surfaced statistics tagged with their source. Percentages are
    only kept when they co-occur with a stronger stat in the same item, to avoid
    surfacing trivial "10% of patients" noise on its own."""
    out = []
    for ev in evidence_items:
        text = ev.get("text", "") or ""
        sid = ev.get("source_id", "") or ev.get("id", "")
        tier = ev.get("tier", "")
        title = ev.get("title", "")
        stats = extract_from_text(text)
        if not stats:
            continue
        kinds = {s["kind"] for s in stats}
        strong = kinds & {"ratio", "ci", "p_value"}
        for s in stats:
            # drop lone percents/sample-sizes unless the item also has a strong stat
            if s["kind"] in ("percent", "sample_size") and not strong:
                continue
            out.append({**s, "source_id": sid, "tier": tier, "title": title})
    return out


def format_stats_markdown(stats: list[dict]) -> str:
    """Render surfaced statistics as a compact, clearly-labeled markdown block.
    Honest framing: 'as reported by the source', not computed."""
    if not stats:
        return ""
    lines = ["## Reported Statistics (extracted as-reported, not computed)"]
    lines.append("_These are statistical values stated by the source documents "
                 "themselves. Aletheon surfaces them verbatim and does not compute "
                 "or pool new estimates._\n")
    # group by source_id
    by_src = {}
    for s in stats:
        by_src.setdefault(s["source_id"], []).append(s)
    for sid, items in by_src.items():
        vals = ", ".join(dict.fromkeys(i["value"] for i in items))  # dedupe, keep order
        tier = items[0].get("tier", "")
        lines.append(f"- **{sid}** ({tier}): {vals}")
    return "\n".join(lines)
