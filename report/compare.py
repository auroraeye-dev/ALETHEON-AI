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
    Returns (ordered_chunks_with_tag, evidence_block_text).
    For each chunk, prepends an EXTRACTED NUMBERS block when statistical values
    are present, so the comparator quotes actual numbers rather than adjectives.
    (Step 1 of the Elicit-gap closure — the comparison report is the place
    quantitative writing matters most.)"""
    from report.stats_extract import format_numbers_for_prompt
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
        numbers_block = format_numbers_for_prompt(c["text"])
        header = f"[{c['tag']}] (tier: {c['tier']})"
        if numbers_block:
            lines.append(f"{header}\n{numbers_block}\nFULL TEXT: {c['text']}")
        else:
            lines.append(f"{header} {c['text']}")
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
evidence-based trade-offs and let the reader decide.
6. QUANTIFY YOUR CLAIMS. When an evidence chunk includes an EXTRACTED NUMBERS \
block (effect sizes, CIs, p-values, percentages, sample sizes), QUOTE the actual \
numbers verbatim when you make a comparative claim from that source — never \
write in adjectives when a number is available. Good: "ibuprofen showed lower \
GI events (4.0% vs 7.1% for aspirin, p<0.001) [A7]" / "aspirin associated with \
OR 7.58 (95% CI 2.64–21.78) for upper GI ulceration [B12]". Weak (avoid): \
"ibuprofen has a lower incidence" / "aspirin showed more side effects". This is \
the single most important rule for a useful comparison."""

COMPARE_TEMPLATE = """Compare these two drugs head-to-head.

DRUG 1 = {drug1}    (evidence tagged [A#])
DRUG 2 = {drug2}    (evidence tagged [B#])

Evidence mix — {drug1}: {tally1}
Evidence mix — {drug2}: {tally2}

==== STRUCTURED PER-PAPER FINDINGS ({drug1}) ====
The PRIMARY input for your synthesis is the structured rows below. Each row
captures one paper's study type, population, intervention, numeric outcomes,
safety signal, and conclusion. When making a comparative claim, QUOTE the
OUTCOME_# field verbatim rather than describing it in adjectives.

{findings1}

==== STRUCTURED PER-PAPER FINDINGS ({drug2}) ====

{findings2}

==== SUPPORTING RAW EVIDENCE ({drug1}, [A#]) ====
Raw chunks — use ONLY as supporting context where the structured rows above
were partial. Prefer the structured findings.

{block1}

==== SUPPORTING RAW EVIDENCE ({drug2}, [B#]) ====

{block2}

Write a Markdown comparison with EXACTLY these sections, in this order:

## Bottom Line
A 3-5 sentence executive summary that quotes the most important comparative \
numbers verbatim from the structured findings above. This is the FIRST thing \
a busy reader sees — it must stand alone. Example shape: "In direct head-to-head \
trials, [drug1] showed X% vs Y% on [endpoint] (p=Z) [A#], while [drug2] showed \
A vs B on [other endpoint] [B#]. Safety profiles differ: [drug1] [signal+number], \
[drug2] [signal+number]. Evidence base: N peer-reviewed studies for drug1 vs M \
for drug2, with [comment on direct head-to-head coverage]."

## Key Trade-offs
A bulleted list (4-6 bullets) of the most important comparative differences. \
EVERY bullet must quote at least one verbatim number from the structured findings \
(effect size, percentage, CI, p-value, sample size) and cite its [A#]/[B#] tag. \
NO adjective-only bullets. If a bullet has no quantitative anchor, drop it. \
Example: "- **GI tolerability**: [drug2] associated with OR 7.58 (95% CI \
2.64–21.78) for upper GI ulceration vs [drug1] [B#]."

## Overview
1-2 sentences on what each drug is and its primary use, cited.

## Efficacy — Head to Head
Compare effectiveness on shared indications. Quote numbers verbatim. Note where \
one has stronger/more evidence. Cite [A#]/[B#].

## Safety — Head to Head
Compare adverse effects, warnings, contraindications. Quote numbers verbatim. \
Cite both sides.

## Evidence Strength
Comment on how strong each drug's evidence base is (tiers, # sources, # direct \
head-to-head studies). Be honest where one is thinner.

## Clinical Bottom Line
2-4 sentences of CONCRETE clinical guidance, grounded ONLY in the comparative \
evidence above. Format: "Prefer [drug] when [specific clinical scenario] because \
[cited evidence]. Prefer [other drug] when [other scenario] because [cited \
evidence]. Caution: [specific population/contraindication with number]." If the \
evidence does not support a clear directional recommendation in a given \
scenario, say so explicitly — do not invent guidance.

Cite using [A#]/[B#] tags only."""


def _evidence_to_chunk_dicts(evs):
    """Chunk a list of Evidence objects and return them in the dict shape that
    sections[...] uses (matching core/retrieve._hits_to_dicts)."""
    from core.chunk import chunk_evidence
    out = []
    for ev in evs:
        for ch in chunk_evidence(ev):
            out.append({
                "text": ch.text,
                "source": ch.source,
                "source_id": ch.source_id,
                "title": ch.title or "",
                "url": ch.url or "",
                "tier": ch.tier,
                "score": 1.0,  # head-to-head papers are boosted by definition
            })
    return out


MIN_HEAD_TO_HEAD = 3  # threshold for "rich" vs "thin" head-to-head literature


def generate_comparison(drug1: str, sections1: dict, drug2: str, sections2: dict) -> str:
    # ---- Step 2: comparison-aware retrieval ----
    # Before flattening either drug's sections, fetch direct head-to-head
    # literature mentioning BOTH drugs (e.g. "ibuprofen versus aspirin"). These
    # papers rarely surface in the top-40 single-drug results, but they're the
    # ones that actually carry comparative numbers (OR, p, CI) the report needs.
    from core.combine import fetch_head_to_head
    head_to_head_evs = fetch_head_to_head(drug1, drug2)
    h2h_chunks_all = _evidence_to_chunk_dicts(head_to_head_evs)
    # FIX from step-2 benchmark: a head-to-head paper can chunk into 5-10 pieces;
    # if we push all of them into section dicts, the per-section retrieval gets
    # flooded and surfaces fragments instead of focused single-drug evidence.
    # Limit to ONE representative chunk per paper for injection — the structured
    # extraction step below will see the full text anyway via this representative.
    h2h_chunks: list = []
    seen_h2h: set = set()
    for c in h2h_chunks_all:
        key = (c["source"], c["source_id"])
        if key in seen_h2h:
            continue
        seen_h2h.add(key)
        h2h_chunks.append(c)
    n_h2h = len(h2h_chunks)

    if h2h_chunks:
        for sections in (sections1, sections2):
            sections.setdefault("overview", []).extend(h2h_chunks)
            sections.setdefault("efficacy", []).extend(h2h_chunks)
        log.info(f"[compare] injected {n_h2h} head-to-head paper(s) "
                 f"(1 representative chunk each) into both drug streams")

    ordered1, block1 = _flatten_tag(sections1, "A")
    ordered2, block2 = _flatten_tag(sections2, "B")
    if not ordered1 or not ordered2:
        missing = drug1 if not ordered1 else drug2
        return f"# Aletheon Comparison: {drug1} vs {drug2}\n\n_No evidence retrieved for {missing}._\n"

    # ---- Step 3: per-paper structured extraction for both drugs ----
    # Extract findings for each side. Each finding inherits the [A#]/[B#] tag of
    # its source's first chunk. The synthesis prompt then sees structured rows
    # with verbatim numbers paired to outcomes, not raw chunks.
    from report.extract import (extract_findings, findings_to_synthesis_block,
                                findings_to_compact_table, findings_to_section_table)
    findings1 = extract_findings(ordered1)
    findings2 = extract_findings(ordered2)
    sid_to_tag_1: dict = {}
    sid_to_tag_2: dict = {}
    for c in ordered1:
        sid_to_tag_1.setdefault(c["source_id"], c["tag"])
    for c in ordered2:
        sid_to_tag_2.setdefault(c["source_id"], c["tag"])
    for f in findings1:
        f.tag = sid_to_tag_1.get(f.source_id, "")
    for f in findings2:
        f.tag = sid_to_tag_2.get(f.source_id, "")

    findings_block_1 = findings_to_synthesis_block(findings1)
    findings_block_2 = findings_to_synthesis_block(findings2)

    # Honest fallback disclosure when head-to-head literature is thin.
    h2h_note = ""
    if n_h2h == 0:
        h2h_note = (f"_Note: no direct head-to-head literature was retrieved for "
                    f"{drug1} vs {drug2}. The comparison below draws on single-drug "
                    f"evidence, so comparative statements should be interpreted as "
                    f"indirect inference, not direct comparison._\n\n")
    elif n_h2h < MIN_HEAD_TO_HEAD:
        h2h_note = (f"_Note: only {n_h2h} direct head-to-head paper(s) were "
                    f"retrieved for this drug pair. Some comparative statements "
                    f"below draw on single-drug evidence rather than direct "
                    f"comparison — interpret cautiously._\n\n")

    client = _get_client()
    log.info(f"[compare] generating {drug1} vs {drug2} "
             f"({len(ordered1)} + {len(ordered2)} chunks, "
             f"{len(findings1)} + {len(findings2)} extracted findings, "
             f"{n_h2h} head-to-head paper(s)) …")
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": COMPARE_SYSTEM},
            {"role": "user", "content": COMPARE_TEMPLATE.format(
                drug1=drug1, drug2=drug2,
                tally1=_tier_tally(ordered1), tally2=_tier_tally(ordered2),
                findings1=findings_block_1, findings2=findings_block_2,
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
              f"{n_h2h} head-to-head paper(s) · "
              f"{len(findings1)} + {len(findings2)} extracted findings · "
              f"grounded in retrieved sources only_\n\n")

    report_md = header + h2h_note + body + "\n" + "\n".join(src)

    # Step A: section restructure. The synthesis (Bottom Line, Key Trade-offs,
    # Efficacy, Safety) leads the report; the Evidence Tables move to an audit
    # appendix immediately before Sources. Readers get the answer in the first
    # 30 seconds; auditors get the structured rows when they want them.
    import re as _re
    compact_blocks = []
    if findings1:
        inner = findings_to_compact_table(findings1, heading_level=4)
        if inner:
            compact_blocks.append(f"### {drug1} — extracted findings\n\n" + inner)
    if findings2:
        inner = findings_to_compact_table(findings2, heading_level=4)
        if inner:
            compact_blocks.append(f"### {drug2} — extracted findings\n\n" + inner)
    if compact_blocks:
        appendix = (
            "\n## Evidence Tables (Audit Appendix)\n"
            "_Per-paper structured extraction — for verification. The synthesis "
            "above is drawn from these rows; this section lets you audit each "
            "claim back to its source._\n\n"
            + "\n\n".join(compact_blocks)
            + "\n\n"
        )
        # Insert immediately BEFORE the Sources heading. Sources section is
        # always last; if for some reason it isn't there, append the appendix
        # at the end.
        if "\n## Sources" in report_md:
            report_md = report_md.replace("\n## Sources", appendix + "\n## Sources", 1)
        else:
            report_md = report_md + "\n" + appendix

    # Per-section sub-tables under Efficacy and Safety. For comparison reports,
    # we show both drugs' findings together so the reader sees the parallel.
    def _combined_section_table(focus):
        parts = []
        t1 = findings_to_section_table(findings1, focus=focus)
        t2 = findings_to_section_table(findings2, focus=focus)
        if t1:
            parts.append(f"_{drug1}:_\n" + t1)
        if t2:
            parts.append(f"_{drug2}:_\n" + t2)
        return "\n\n".join(parts) if parts else ""

    eff_table = _combined_section_table("efficacy")
    if eff_table:
        report_md = _re.sub(
            r"(## Efficacy[^\n]*\n)",
            lambda m: m.group(1) + eff_table + "\n\n",
            report_md, count=1)
    safety_table = _combined_section_table("safety")
    if safety_table:
        report_md = _re.sub(
            r"(## Safety[^\n]*\n)",
            lambda m: m.group(1) + safety_table + "\n\n",
            report_md, count=1)

    return report_md


def save_comparison(drug1: str, drug2: str, md: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in f"{drug1}_vs_{drug2}".lower())
    folder = os.path.join(config.REPORTS_DIR, "comparisons")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w") as f:
        f.write(md)
    log.info(f"[compare] saved -> {path}")
    return path
