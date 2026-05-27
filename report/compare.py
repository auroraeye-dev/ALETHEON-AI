"""
report/compare.py
=================
D1 — Comparator agent.

Generates a HEAD-TO-HEAD comparison of two drugs across shared dimensions
(overview, efficacy, safety, evidence strength), not two separate reports glued
together. Reuses the existing drug-filtered, section-targeted retrieval so each
drug's evidence stays on-drug (A1 precision applies per drug).

Each drug's evidence is tagged [A#] (drug 1) and [B#] (drug 2) so citations make
clear which drug a claim is about.
"""

import os
from datetime import datetime

from core.config import config
from core.logging_setup import log
from report.generate import _get_client, _tier_tally


def _flatten_tag(sections: dict, prefix: str) -> tuple[list[dict], str]:
    """Flatten a drug's section dict into a deduped, prefix-tagged evidence list.
    Returns (ordered_chunks_with_tag, evidence_block_text)."""
    seen, ordered = {}, []
    for sec_chunks in sections.values():
        for c in sec_chunks:
            key = (c["source"], c["source_id"], c["text"][:50])
            if key not in seen:
                c = dict(c)
                c["tag"] = f"{prefix}{len(ordered) + 1}"
                seen[key] = c
                ordered.append(c)
    lines = []
    for c in ordered:
        lines.append(f"[{c['tag']}] (tier: {c['tier']}) {c['text']}")
    return ordered, "\n".join(lines)


COMPARE_SYSTEM = """You are Aletheon, a drug-intelligence analyst. You write \
HEAD-TO-HEAD drug comparisons for healthcare and pharma researchers, grounded \
ONLY in the provided evidence, with honesty about evidence strength.

RULES:
1. Use ONLY the provided evidence. Drug 1 evidence is tagged [A#], Drug 2 [B#].
2. Cite every claim with its tag(s). Never attribute one drug's evidence to the other.
3. Compare on shared dimensions; where evidence exists for one drug but not the \
other, say so explicitly rather than inventing parity.
4. Flag preprint-based claims as not peer-reviewed.
5. Neutral and factual — do NOT declare an overall "winner"; present the \
evidence-based trade-offs and let the reader decide."""

COMPARE_TEMPLATE = """Compare these two drugs head-to-head.

DRUG 1 = {drug1}    (evidence tagged [A#])
DRUG 2 = {drug2}    (evidence tagged [B#])

Evidence mix — {drug1}: {tally1}
Evidence mix — {drug2}: {tally2}

=== {drug1} EVIDENCE ([A#]) ===
{block1}

=== {drug2} EVIDENCE ([B#]) ===
{block2}

Write a Markdown comparison with EXACTLY these sections:

## Overview
1-2 sentences on what each drug is and its primary use, cited.

## Efficacy — Head to Head
Compare effectiveness on shared indications. Note where one has stronger/▒more \
evidence. Cite [A#]/[B#].

## Safety — Head to Head
Compare adverse effects, warnings, contraindications. Cite both sides.

## Evidence Strength
Comment on how strong each drug's evidence base is (tiers, # sources). Be honest \
where one is thinner.

## Bottom Line
Neutral summary of the key trade-offs (NOT a winner declaration), cited.

Cite using [A#]/[B#] tags only."""


def generate_comparison(drug1: str, sections1: dict, drug2: str, sections2: dict) -> str:
    ordered1, block1 = _flatten_tag(sections1, "A")
    ordered2, block2 = _flatten_tag(sections2, "B")
    if not ordered1 or not ordered2:
        missing = drug1 if not ordered1 else drug2
        return f"# Aletheon Comparison: {drug1} vs {drug2}\n\n_No evidence retrieved for {missing}._\n"

    client = _get_client()
    log.info(f"[compare] generating {drug1} vs {drug2} "
             f"({len(ordered1)} + {len(ordered2)} chunks) …")
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": COMPARE_SYSTEM},
            {"role": "user", "content": COMPARE_TEMPLATE.format(
                drug1=drug1, drug2=drug2,
                tally1=_tier_tally(ordered1), tally2=_tier_tally(ordered2),
                block1=block1, block2=block2)},
        ],
        temperature=config.LLM_TEMPERATURE,
    )
    # cost tracking
    try:
        from core.metrics import record_llm
        u = getattr(resp, "usage", None)
        if u is not None:
            record_llm(getattr(u, "prompt_tokens", 0),
                       getattr(u, "completion_tokens", 0), config.LLM_MODEL)
    except Exception:
        pass
    body = resp.choices[0].message.content

    # Combined sources list, both drugs — deduped by source document (A3).
    def _doc_lines(ordered):
        by_doc, order = {}, []
        for c in ordered:
            key = (c["source"], c["source_id"])
            if key not in by_doc:
                by_doc[key] = {"tags": [], "title": c["title"], "tier": c["tier"],
                               "source": c["source"], "source_id": c["source_id"]}
                order.append(key)
            by_doc[key]["tags"].append(c["tag"])
        out = []
        for key in order:
            d = by_doc[key]
            tag_str = "".join(f"[{t}]" for t in d["tags"])
            out.append(f"- **{tag_str}** ({d['tier']}) {d['title']} — "
                       f"{d['source']}:{d['source_id']}")
        return out

    src = ["\n## Sources\n", f"\n**{drug1}:**"] + _doc_lines(ordered1)
    src += [f"\n**{drug2}:**"] + _doc_lines(ordered2)

    header = (f"# Aletheon Comparison: {drug1} vs {drug2}\n\n"
              f"_Generated {datetime.now():%Y-%m-%d %H:%M} · "
              f"{len(ordered1)} + {len(ordered2)} evidence chunks · "
              f"grounded in retrieved sources only_\n\n")
    return header + body + "\n" + "\n".join(src)


def save_comparison(drug1: str, drug2: str, md: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in f"{drug1}_vs_{drug2}".lower())
    folder = os.path.join(config.REPORTS_DIR, "comparisons")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w") as f:
        f.write(md)
    log.info(f"[compare] saved -> {path}")
    return path
