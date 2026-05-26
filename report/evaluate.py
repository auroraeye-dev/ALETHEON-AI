"""
report/evaluate.py
==================
Build 2 flagship — self-evaluation for the feedback loop.

After the first report is generated, this inspects it and decides whether a
corrective pass is warranted. It returns a structured verdict:

  {
    "needs_retry": bool,
    "weak_sections": [...],     # which sections look thin/empty
    "reasons": [...],           # human-readable why
    "boost": {section: extra_k} # how many MORE chunks to pull per weak section
  }

This is deliberately CHEAP and deterministic (no extra LLM call): it reads the
report text structurally — the same signals the eval harness uses. That keeps
the feedback loop's cost bounded and predictable (the only added cost is the
ONE corrective retrieve+regenerate, and only when actually needed).
"""

import re

from core.logging_setup import log


# Sections we care about being substantive, and how much to boost retrieval
# for each if it comes up weak on the first pass.
_SECTION_BOOST = {
    "safety": 8,
    "efficacy": 6,
    "overview": 4,
    "preprint": 4,
}

# Phrases that signal a section had no real content.
_EMPTY_SIGNALS = [
    "insufficient evidence",
    "no preprint evidence",
    "no direct contradictions",  # not weak per se, handled separately
]


def _section_body(md: str, header_regex: str) -> str:
    m = re.search(header_regex + r"[^\n]*\n(.*?)(?:\n## |\Z)", md, flags=re.S)
    return m.group(1).strip() if m else ""


def evaluate_report(md: str) -> dict:
    """Inspect a report; decide if a single corrective pass is warranted."""
    weak, reasons, boost = [], [], {}

    # 1) Safety section thin or empty?
    safety = _section_body(md, r"## Safety / Warnings").lower()
    if "insufficient evidence" in safety or len(safety) < 80:
        weak.append("safety")
        boost["safety"] = _SECTION_BOOST["safety"]
        reasons.append("Safety section is thin or empty")

    # 2) Key Findings too few?
    findings_block = _section_body(md, r"## Key Findings")
    n_findings = len(re.findall(r"^\s*[-*]\s", findings_block, flags=re.M))
    if n_findings < 3:
        weak.append("efficacy")
        boost["efficacy"] = _SECTION_BOOST["efficacy"]
        reasons.append(f"Only {n_findings} key finding(s)")

    # 3) Overview/summary missing?
    summary = _section_body(md, r"## Summary")
    if len(summary) < 60:
        weak.append("overview")
        boost["overview"] = _SECTION_BOOST["overview"]
        reasons.append("Summary is very short")

    # 4) Confidence almost entirely Low? (signals weak evidence base)
    confs = re.findall(r"Confidence:\s*(High|Medium|Low)", md, flags=re.I)
    if confs:
        low = sum(1 for c in confs if c.lower() == "low")
        if low / len(confs) > 0.6:
            # boost efficacy retrieval to find stronger evidence
            weak.append("efficacy")
            boost["efficacy"] = max(boost.get("efficacy", 0), _SECTION_BOOST["efficacy"])
            reasons.append("Most findings are Low confidence")

    needs_retry = len(weak) > 0
    verdict = {
        "needs_retry": needs_retry,
        "weak_sections": sorted(set(weak)),
        "reasons": reasons,
        "boost": boost,
    }
    if needs_retry:
        log.info(f"[evaluate] report weak -> corrective pass. "
                 f"Reasons: {'; '.join(reasons)}")
    else:
        log.info("[evaluate] report is solid -> no corrective pass needed")
    return verdict
