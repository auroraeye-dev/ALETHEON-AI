"""
report/generate.py
==================
DAY 7 (retrieval upgrade): build the report from SECTION-TARGETED evidence.

Each section is fed evidence retrieved specifically for it (safety chunks for
Safety, efficacy chunks for Key Findings, preprint-filtered chunks for the
Preprint section). All chunks are merged into one numbered [E#] list so
citations resolve, and we dedup so the same chunk isn't shown twice.
"""

import os
from datetime import datetime

from openai import OpenAI

from core.config import config
from core.logging_setup import log

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _merge_and_tag(sections: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    """Dedup chunks across sections, assign each a stable [E#] tag.
    Returns (section -> list of tags, ordered unique chunks with .tag)."""
    unique = {}          # (source, source_id, text) -> chunk
    section_tags = {}    # section -> list of E# tags

    # First pass: collect unique chunks, preserving first-seen order.
    ordered = []
    for section, chunks in sections.items():
        for c in chunks:
            key = (c["source"], c["source_id"], c["text"][:50])
            if key not in unique:
                c = dict(c)  # copy
                c["tag"] = f"E{len(ordered) + 1}"
                unique[key] = c
                ordered.append(c)

    # Second pass: map each section to the tags it uses.
    for section, chunks in sections.items():
        tags = []
        for c in chunks:
            key = (c["source"], c["source_id"], c["text"][:50])
            tags.append(unique[key]["tag"])
        section_tags[section] = tags

    return section_tags, ordered


def _evidence_block(section_tags: dict, ordered: list[dict]) -> str:
    """Format the evidence the LLM sees, grouped by which section it's for."""
    lines = []
    label = {
        "overview": "OVERVIEW evidence",
        "efficacy": "EFFICACY / FINDINGS evidence",
        "safety": "SAFETY evidence",
        "contradiction": "CONTRADICTION-CHECK evidence (both supportive AND opposing — compare these for conflicts)",
        "preprint": "PREPRINT evidence (NOT peer-reviewed)",
    }
    for section in ["overview", "efficacy", "safety", "contradiction", "preprint"]:
        tags = section_tags.get(section, [])
        if not tags:
            continue
        lines.append(f"\n=== {label[section]} ===")
        seen = set()
        for c in ordered:
            if c["tag"] in tags and c["tag"] not in seen:
                seen.add(c["tag"])
                lines.append(f"[{c['tag']}] (tier: {c['tier']}, {c['source']}:{c['source_id']}) {c['text']}")
    return "\n".join(lines)


def _tier_tally(ordered: list[dict]) -> str:
    """Summarize the evidence mix so the LLM can ground its confidence judgments."""
    counts = {}
    for c in ordered:
        counts[c["tier"]] = counts.get(c["tier"], 0) + 1
    parts = [f"{n} {tier}" for tier, n in sorted(counts.items())]
    return ", ".join(parts) if parts else "none"


# How authoritative each tier is — given to the LLM to weight confidence.
TIER_AUTHORITY = (
    "Evidence authority (most to least): regulatory (FDA/EMA) and peer-reviewed "
    "RCTs are strongest; observational/real-world is moderate; PREPRINTS are "
    "weakest (NOT peer-reviewed) and must never raise confidence on their own."
)

SYSTEM_PROMPT = """You are Aletheon, a drug-intelligence analyst writing \
evidence-grounded reports for healthcare and pharma researchers. Your defining \
trait is HONESTY ABOUT EVIDENCE: you show how strong the support is and where \
sources disagree, rather than presenting everything as equally certain.

ABSOLUTE RULES:
1. Use ONLY the evidence provided. No outside knowledge. If evidence doesn't \
support a claim, don't make it.
2. Cite every factual claim with its evidence tag(s), e.g. [E3] or [E1][E4].
3. If a section has no supporting evidence, write "Insufficient evidence in sources."
4. PREPRINT evidence belongs ONLY in the Preprint section, flagged as not \
peer-reviewed. Never let a preprint alone raise a finding's confidence.
5. CONFIDENCE: after each Key Finding, append a confidence tag in the form \
**(Confidence: High|Medium|Low — brief reason citing how many sources and which \
tiers support it)**. High = multiple regulatory/peer-reviewed sources agree. \
Medium = some support or only moderate-tier sources. Low = single source, weak \
tier, or only a preprint.
6. CONTRADICTIONS: if two or more pieces of evidence disagree, you MUST surface \
this in the Contradictions section with both sides cited. Do not hide disagreement.
7. Be concise, factual, neutral. No marketing language."""

USER_TEMPLATE = """Drug / query: {drug}

Evidence mix retrieved: {tier_tally}
{tier_authority}

The evidence below is grouped by which section it supports. Use each group for
its section. Cite with [E#] tags.

{evidence}

Write a Markdown report with EXACTLY these sections:

## Summary
2-4 sentence neutral overview (use OVERVIEW evidence), cited.

## Key Findings
Most important efficacy/clinical points (use EFFICACY evidence), bulleted, cited.
After EACH finding, append its confidence tag: \
**(Confidence: High|Medium|Low — reason)**.

## Safety / Warnings
Adverse effects, contraindications, risks (use SAFETY evidence), bulleted, cited.

## Contradictions & Disagreements
Examine the CONTRADICTION-CHECK evidence (and all other evidence) for points where \
findings conflict — e.g. one source reports benefit, another reports no effect or \
harm. Surface each conflict with both sides cited. If genuinely none conflict, \
write "No direct contradictions found in current sources."

## Preprint / Emerging Evidence (not yet peer-reviewed)
ONLY from PREPRINT evidence. Flag as not peer-reviewed. If none, write \
"No preprint evidence in current sources."

Cite using [E#] tags only. Do not add other sections."""


def generate_report(drug: str, sections: dict[str, list[dict]]) -> str:
    """Generate the cited report from section-targeted evidence."""
    section_tags, ordered = _merge_and_tag(sections)
    if not ordered:
        return f"# Aletheon Report: {drug}\n\n_No evidence retrieved._\n"

    evidence_block = _evidence_block(section_tags, ordered)
    client = _get_client()

    log.info(f"[report] generating for {drug!r} from {len(ordered)} unique chunks "
             f"across {len([s for s in sections if sections[s]])} sections …")
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                drug=drug,
                evidence=evidence_block,
                tier_tally=_tier_tally(ordered),
                tier_authority=TIER_AUTHORITY)},
        ],
        temperature=0.2,
    )
    body = resp.choices[0].message.content

    # Sources list from the deduped, tagged chunks.
    sources_lines = ["\n## Sources\n"]
    for c in ordered:
        sources_lines.append(
            f"- **[{c['tag']}]** ({c['tier']}) {c['title']} — "
            f"{c['source']}:{c['source_id']}  \n  {c['url']}"
        )
    sources = "\n".join(sources_lines)

    header = (f"# Aletheon Report: {drug}\n\n"
              f"_Generated {datetime.now():%Y-%m-%d %H:%M} · "
              f"{len(ordered)} evidence chunks (section-targeted) · "
              f"grounded in retrieved sources only_\n\n")

    return header + body + "\n" + sources


def save_report(drug: str, report_md: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in drug.lower())
    folder = os.path.join(config.REPORTS_DIR, safe)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"report_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w") as f:
        f.write(report_md)
    log.info(f"[report] saved -> {path}")
    return path
