"""
report/critic.py
================
D2 — Critic agent (opt-in).

Reads a FINISHED report and APPENDS a "## Critical Appraisal" section that
challenges the report's own conclusions: overstated claims, findings resting on
weak/single-source/preprint evidence, glossed-over limitations, and places where
confidence may be too high.

IMPORTANT: this is purely ADDITIVE. It never edits, trims, rewrites, or removes
any existing section. The original report is passed through untouched and the
appraisal is appended at the end — so the output only ever GROWS.

On-brand: a product about "honesty about evidence" should actively look for its
own weaknesses rather than hide them.
"""

from core.config import config
from core.logging_setup import log
from report.generate import _get_client


CRITIC_SYSTEM = """You are a skeptical senior reviewer auditing a drug-intelligence \
report. Your job is to find weaknesses in the report's OWN reasoning and evidence \
use — not to rewrite it. Be rigorous, fair, and specific.

Look for:
- Claims stated more confidently than their evidence supports.
- Findings resting on a single source, weak tier, or preprint treated as settled.
- Important limitations, caveats, or populations the report glossed over.
- Places where contradictory evidence was underweighted.
- Over-generalizations from narrow studies.

Be constructive and precise. Cite the report's own [E#] tags where relevant. If \
the report is genuinely sound, say so honestly rather than inventing faults."""

CRITIC_TEMPLATE = """Audit the following drug-intelligence report for {drug}. \
Write ONLY a "## Critical Appraisal" section (3-6 concise bullet points) that \
flags the report's weaknesses per the rules. Do NOT rewrite or summarize the \
report — only critique it. Reference [E#] tags where useful.

REPORT:
{report}

Output just the "## Critical Appraisal" section."""


def append_appraisal(drug: str, report_md: str) -> str:
    """Return report_md UNCHANGED with a Critical Appraisal section appended."""
    if not report_md.strip():
        return report_md

    client = _get_client()
    log.info(f"[critic] auditing report for {drug!r} (additive appraisal) …")
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": CRITIC_TEMPLATE.format(
                    drug=drug, report=report_md)},
            ],
            temperature=0.3,
        )
    except Exception as e:
        log.warning(f"[critic] failed: {e} — returning report unchanged")
        return report_md

    # cost tracking
    try:
        from core.metrics import record_llm
        u = getattr(resp, "usage", None)
        if u is not None:
            record_llm(getattr(u, "prompt_tokens", 0),
                       getattr(u, "completion_tokens", 0), config.LLM_MODEL)
    except Exception:
        pass

    appraisal = resp.choices[0].message.content.strip()
    if not appraisal:
        return report_md
    # Ensure the header is present exactly once.
    if not appraisal.lower().startswith("## critical appraisal"):
        appraisal = "## Critical Appraisal\n\n" + appraisal

    # PURELY ADDITIVE: original report, then a separator, then the appraisal.
    # The Sources list usually sits at the end of report_md; we append the
    # appraisal AFTER everything so nothing existing is disturbed.
    return report_md.rstrip() + "\n\n" + appraisal + "\n"
