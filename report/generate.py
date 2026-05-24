"""
report/generate.py
==================
DAY 3: turn retrieved evidence chunks into a structured, CITED report.

Key design choices (these are what make it trustworthy, not just impressive):
  - The LLM is told to answer ONLY from the provided evidence (low hallucination).
  - Every chunk is given a numbered [E#] tag; the model must cite those tags.
  - Evidence is grouped by TIER, and preprints are forced into their own
    clearly-labeled section (ready for when preprint sources come online).
  - We build a Sources list from the real metadata, so citations resolve.
"""

import os
from datetime import datetime

from openai import OpenAI

from core.config import config
from core.logging_setup import log

# Tier display order + headings for the evidence we feed the model.
TIER_ORDER = ["regulatory", "peer_reviewed", "real_world", "preprint", "patent"]
TIER_LABEL = {
    "regulatory": "Regulatory (FDA/EMA)",
    "peer_reviewed": "Peer-reviewed",
    "real_world": "Real-world / Safety",
    "preprint": "Preprint (NOT peer-reviewed)",
    "patent": "Patent",
}

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _format_evidence(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Number each chunk as [E#] and group by tier for the prompt.
    Returns (evidence_text_block, ordered_chunks_with_tags)."""
    # order chunks by tier so the model sees authoritative evidence first
    ordered = sorted(
        chunks,
        key=lambda c: TIER_ORDER.index(c["tier"]) if c["tier"] in TIER_ORDER else 99,
    )
    lines = []
    for i, c in enumerate(ordered, 1):
        c["tag"] = f"E{i}"
        lines.append(
            f"[E{i}] (tier: {c['tier']}, source: {c['source']}:{c['source_id']}, "
            f"title: {c['title']})\n{c['text']}"
        )
    return "\n\n".join(lines), ordered


SYSTEM_PROMPT = """You are Aletheon, a drug-intelligence analyst. You write \
evidence-grounded reports for healthcare and pharma researchers.

ABSOLUTE RULES:
1. Use ONLY the evidence provided. Do NOT use outside knowledge. If the evidence \
does not support a claim, do not make it.
2. Cite every factual claim with its evidence tag(s), e.g. [E3] or [E1][E4].
3. If evidence is insufficient for a section, write "Insufficient evidence in sources."
4. Keep PREPRINT evidence (tier: preprint) strictly inside the "Preprint / Emerging \
Evidence" section, and note it is NOT peer-reviewed. Never mix preprint claims into \
the main findings.
5. Be concise, factual, and neutral. No marketing language."""

USER_TEMPLATE = """Drug / query: {drug}

EVIDENCE (each item tagged [E#]):
{evidence}

Write a structured report in Markdown with EXACTLY these sections:

## Summary
A 2-4 sentence neutral overview, cited.

## Key Findings
Bulleted, the most important evidence-backed points, each cited.

## Safety / Warnings
Adverse effects, contraindications, risks found in the evidence, cited.

## Preprint / Emerging Evidence (not yet peer-reviewed)
ONLY content from tier: preprint evidence. If there is none, write \
"No preprint evidence in current sources."

Do not add sections beyond these. Cite using [E#] tags only."""


def generate_report(drug: str, chunks: list[dict]) -> str:
    """Generate the cited report body (Markdown) from retrieved chunks."""
    if not chunks:
        return f"# Aletheon Report: {drug}\n\n_No evidence retrieved._\n"

    evidence_block, ordered = _format_evidence(chunks)
    client = _get_client()

    log.info(f"[report] generating report for {drug!r} from {len(chunks)} chunks …")
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                drug=drug, evidence=evidence_block)},
        ],
        temperature=0.2,  # low = factual, less invention
    )
    body = resp.choices[0].message.content

    # Build a real Sources list from metadata so [E#] tags resolve.
    sources_lines = ["\n## Sources\n"]
    for c in ordered:
        sources_lines.append(
            f"- **[{c['tag']}]** ({c['tier']}) {c['title']} — "
            f"{c['source']}:{c['source_id']}  \n  {c['url']}"
        )
    sources = "\n".join(sources_lines)

    header = (f"# Aletheon Report: {drug}\n\n"
              f"_Generated {datetime.now():%Y-%m-%d %H:%M} · "
              f"{len(chunks)} evidence chunks · grounded in retrieved sources only_\n\n")

    return header + body + "\n" + sources


def save_report(drug: str, report_md: str) -> str:
    """Save the report to data/reports/{drug}/report_{timestamp}.md. Returns path."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in drug.lower())
    folder = os.path.join(config.REPORTS_DIR, safe)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"report_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w") as f:
        f.write(report_md)
    log.info(f"[report] saved -> {path}")
    return path
