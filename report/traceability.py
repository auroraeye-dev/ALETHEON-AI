"""
report/traceability.py
======================
F2 — evidence traceability.

Makes the claim -> citation -> source-document chain EXPLICIT and exportable,
instead of leaving it implied by inline [E#] tags. For every claim line in the
report that carries citations, we record:
    claim text  ->  [E#] tags  ->  source document(s) (tier, source, url)

This is the honest, ready-now version: a structured map (dict/JSON) plus a
readable markdown appendix. It uses data you ALREADY have (the [E#] -> source
mapping the report builds), so there's no new guessing. When a web UI exists
later, this same structure powers a click-through "why does it say this?" graph.
"""

import re

_TAG_RE = re.compile(r"\[(E\d+)\]")
# A "claim" line is a bullet or sentence that carries at least one [E#] tag.
_CLAIM_LINE_RE = re.compile(r"^\s*[-*]\s+(.*\[E\d+\].*)$", re.M)


def build_traceability(report_md: str, evidence_index: dict) -> dict:
    """
    report_md: the generated report markdown (with inline [E#] tags).
    evidence_index: {"E1": {"source": "...", "source_id": "...", "tier": "...",
                            "title": "...", "url": "..."}, ...}
    Returns a structured traceability map.
    """
    # Only parse claims from the BODY — never the Sources list or appendices,
    # whose bullets also carry [E#] tags and would be mis-read as claims. Cut the
    # report at the first of these section headers.
    body = report_md
    for stop in ("\n## Sources", "\n## Reported Statistics", "\n## Evidence Traceability"):
        idx = body.find(stop)
        if idx != -1:
            body = body[:idx]

    claims = []
    for m in _CLAIM_LINE_RE.finditer(body):
        line = m.group(1).strip()
        tags = _TAG_RE.findall(line)
        if not tags:
            continue
        # Strip the confidence tag and citations from the displayed claim text
        claim_text = re.sub(r"\*\*\(Confidence:.*?\)\*\*", "", line)
        claim_text = _TAG_RE.sub("", claim_text).strip(" .—-")
        sources = []
        seen = set()
        for tag in tags:
            info = evidence_index.get(tag)
            if not info:
                continue
            key = info.get("source_id", tag)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "tag": tag,
                "tier": info.get("tier", ""),
                "source": info.get("source", ""),
                "source_id": info.get("source_id", ""),
                "title": info.get("title", ""),
                "url": info.get("url", ""),
            })
        claims.append({
            "claim": claim_text,
            "tags": tags,
            "sources": sources,
            # highest tier backing this claim (for quick trust read)
            "tiers": sorted({s["tier"] for s in sources if s["tier"]}),
        })
    return {"claims": claims, "n_claims": len(claims)}


def format_traceability_markdown(trace: dict) -> str:
    """Readable appendix: each claim and the exact sources backing it."""
    claims = trace.get("claims", [])
    if not claims:
        return ""
    lines = ["## Evidence Traceability",
             "_Each claim above, mapped to the exact source document(s) that "
             "support it. This is the audit trail behind the report._\n"]
    for i, c in enumerate(claims, 1):
        tier_str = ", ".join(c["tiers"]) if c["tiers"] else "—"
        lines.append(f"**{i}.** {c['claim']}  \n"
                     f"   ↳ backed by ({tier_str}): " +
                     "; ".join(
                         f"{s['tag']} {s['title'][:60]}".strip()
                         for s in c["sources"]) )
    return "\n".join(lines)
