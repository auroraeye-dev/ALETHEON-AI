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
from report.generate import _get_client, _tier_tally, synthesize


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
1. Use ONLY the structured findings provided. Drug 1 evidence is tagged [A#], \
Drug 2 [B#]. These rows have already passed a strict head-to-head screening \
gate (direct comparison, monotherapy, clinical outcome, human participants).
2. Cite every comparative claim with its tag(s). Never attribute one drug's \
finding to the other. Never cite a tag that does not appear in the structured \
findings — if you have no source, write fewer bullets.
3. If a structured findings block is EMPTY or sparse, do NOT invent claims to \
fill space. Write fewer sentences honestly. A short Bottom Line that quotes \
real numbers is better than a long Bottom Line that paraphrases.
4. Flag preprint-based claims as not peer-reviewed.
5. Neutral and factual — do NOT declare an overall "winner"; present the \
evidence-based trade-offs and let the reader decide.
6. QUANTIFY YOUR CLAIMS. Each row contains numeric outcomes (effect sizes, CIs, \
p-values, percentages, sample sizes). QUOTE the actual numbers verbatim when \
making a comparative claim — never write in adjectives when a number is \
available. Good: "ibuprofen showed lower GI events (4.0% vs 7.1% for aspirin, \
p<0.001) [A7]" / "aspirin associated with OR 7.58 (95% CI 2.64–21.78) for \
upper GI ulceration [B12]". Weak (avoid): "ibuprofen has a lower incidence" / \
"aspirin showed more side effects". This is the single most important rule \
for a useful comparison."""

COMPARE_TEMPLATE = """Compare these two drugs head-to-head.

DRUG 1 = {drug1}    (evidence tagged [A#])
DRUG 2 = {drug2}    (evidence tagged [B#])

Evidence mix — {drug1}: {tally1}
Evidence mix — {drug2}: {tally2}

==== STRUCTURED PER-PAPER FINDINGS ({drug1}) — SCREENED HEAD-TO-HEAD ONLY ====
The ONLY input for your synthesis is the structured rows below. Each row
captures one paper that passed strict head-to-head screening: study type,
population, intervention, numeric outcomes, safety signal, conclusion. When
making a comparative claim, QUOTE the OUTCOME_# field verbatim rather than
describing it in adjectives. You have NO access to other evidence — do not
attempt to cite [A#]/[B#] tags that are not present in these blocks.

{findings1}

==== STRUCTURED PER-PAPER FINDINGS ({drug2}) — SCREENED HEAD-TO-HEAD ONLY ====

{findings2}

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


# When 0 papers pass the head-to-head screening gate, we use this restricted
# template instead. The Bottom Line / Key Trade-offs / Efficacy — Head to Head /
# Clinical Bottom Line sections are STRUCTURALLY REMOVED — the LLM is not asked
# to write them at all. The remaining sections (Overview, single-drug context,
# Safety with explicit single-drug framing, Evidence Strength) draw on raw
# single-drug evidence honestly framed as such.
#
# Why this is built differently from the normal template:
# In the normal template, "use raw evidence only as supporting context" is a
# RULE the LLM may or may not follow. When screening drops to 0, soft rules
# aren't enough — the LLM will quote spurious comparative numbers from the raw
# chunks (we observed this: a paracetamol+ibuprofen vs ibuprofen-monotherapy
# RCT got quoted as "direct head-to-head trial" with a confident citation).
# The fix is to REMOVE the sections that demand comparative claims. The LLM
# physically cannot quote head-to-head numbers if no head-to-head sections
# exist in the requested output.
COMPARE_TEMPLATE_NO_H2H = """Compare these two drugs head-to-head.

DRUG 1 = {drug1}    (evidence tagged [A#])
DRUG 2 = {drug2}    (evidence tagged [B#])

Evidence mix — {drug1}: {tally1}
Evidence mix — {drug2}: {tally2}

==== IMPORTANT: NO HEAD-TO-HEAD EVIDENCE PASSED SCREENING ====
The strict head-to-head screening gate rejected ALL candidate papers for this
drug pair. The papers we retrieved either compared one of these drugs against
a different drug, used combination therapy in one arm, lacked clinical outcomes,
were trial registrations without results, or were not in human participants.

You therefore CANNOT make any direct comparative numerical claim like
"drug 1 showed X% vs Y% for drug 2" — there is no evidence that supports such
a comparison. The evidence below is single-drug evidence for each drug
separately, suitable for describing each drug independently but NOT for making
head-to-head claims about relative efficacy or safety.

==== SINGLE-DRUG EVIDENCE ({drug1}, [A#]) ====
{block1}

==== SINGLE-DRUG EVIDENCE ({drug2}, [B#]) ====
{block2}

Write a Markdown report with EXACTLY these sections, in this order:

## Head-to-Head Status
A 2-3 sentence honest statement that no direct head-to-head evidence survived
screening for {drug1} vs {drug2}. State what this means for the reader:
specific comparative claims (relative efficacy, relative safety, dose
equivalence) cannot be made from the available evidence. Suggest what the
reader should do instead (consult primary head-to-head literature, clinical
guidelines, or specialist review).

## Overview
1-2 sentences on each drug independently — what it is, what it's used for.
Cite [A#] for drug 1 statements, [B#] for drug 2 statements. Do NOT compare
the two drugs in this section.

## {drug1} — Single-Drug Profile
A short summary (3-5 sentences or bullets) of what the single-drug evidence
shows about {drug1}: indications, key efficacy findings WITH NUMBERS where
available, key safety signals WITH NUMBERS. Cite [A#] only. Do NOT introduce
[B#] tags. Do NOT make comparative claims.

## {drug2} — Single-Drug Profile
Same shape, for {drug2}. Cite [B#] only.

## Evidence Strength
Honest assessment of each drug's evidence base separately (peer-reviewed
counts, regulatory sources, study types). Be explicit that head-to-head
evidence was NOT found in this retrieval. Do NOT speculate about what direct
comparisons would show.

## What This Report Cannot Tell You
2-4 bullets listing the specific comparative questions the reader might want
answered (relative efficacy on shared indications, relative safety profile,
when to choose one over the other) and explicitly state that the current
evidence base does not answer them. This is the most important section in
this restricted report — it is the antidote to false confidence.

Cite using [A#]/[B#] tags ONLY. Do NOT cite a tag from one drug to make a
statement about the other drug. Do NOT use comparative language ("better,"
"worse," "higher," "lower," "more effective") between the two drugs anywhere
in this report."""


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


_REFUSAL_SYSTEM = """You are Aletheon, a drug-intelligence analyst. You are \
writing the descriptive sections of a comparison report where the strict \
head-to-head screening gate found insufficient direct evidence to support \
comparative claims. Your job is ONLY to describe each drug factually based on \
its individual evidence — NEVER to compare them. Use only the structured \
per-paper findings provided. Cite [A#] for drug 1, [B#] for drug 2. Never \
attribute one drug's finding to the other. Never write phrases like "compared \
to", "vs", "better than", "lower than" between the two drugs."""

_REFUSAL_TEMPLATE = """The strict head-to-head screening gate found insufficient \
direct comparative evidence for {drug1} vs {drug2}, so this report cannot \
include Bottom Line, Key Trade-offs, Efficacy, Safety, or Clinical Bottom Line \
sections. You are writing ONLY two descriptive sections, each grounded in the \
respective drug's separate evidence.

==== STRUCTURED FINDINGS FOR {drug1} (descriptive only — do NOT compare) ====

{findings1}

==== STRUCTURED FINDINGS FOR {drug2} (descriptive only — do NOT compare) ====

{findings2}

Write a Markdown report with EXACTLY these two sections, in this order:

## Overview
A 2-3 sentence factual description of {drug1} (its drug class, primary indications) \
followed by a 2-3 sentence factual description of {drug2}. Cite [A#] for {drug1} \
claims and [B#] for {drug2} claims. Do NOT compare the two.

## Evidence Strength
Describe each drug's evidence base separately: number of structured findings, \
study types represented (RCTs, meta-analyses, observational, label/monograph). \
One paragraph per drug. Do NOT compare strengths.

That is the entire content. Do not add any other sections."""


def _build_refusal_report(drug1: str, drug2: str,
                          ordered1: list, ordered2: list,
                          findings1: list, findings2: list,
                          n_h2h_retrieved: int, n_h2h_screened: int,
                          sections1: dict, sections2: dict) -> str:
    """Build the report in 'refusal mode' — when screening produced too few
    direct head-to-head papers, we DO NOT ask the LLM to synthesize comparative
    sections at all. Instead, the comparative sections are replaced by a hard-
    coded honest refusal block. The LLM only writes descriptive Overview and
    Evidence Strength sections, each grounded in one drug's evidence only.

    This is a structural enforcement of the screening contract: if the LLM
    has no head-to-head structured findings in its prompt, it physically
    cannot quote them. The bug where the LLM cherry-picked combination-therapy
    numbers from raw chunks for the Bottom Line is impossible here, because
    the prompt never asks for a Bottom Line."""
    from report.extract import (findings_to_synthesis_block,
                                findings_to_compact_table)

    # Build the two descriptive sections via a small LLM call.
    findings_block_1 = findings_to_synthesis_block(
        [f for f in findings1 if f.extraction_quality != "failed"])
    findings_block_2 = findings_to_synthesis_block(
        [f for f in findings2 if f.extraction_quality != "failed"])

    client = _get_client()
    log.info(f"[compare] generating REFUSAL-mode report for {drug1} vs {drug2} "
             f"({n_h2h_screened}/{n_h2h_retrieved} h2h papers passed screening; "
             f"descriptive sections only) …")
    descriptive_body = synthesize(
        system_prompt=_REFUSAL_SYSTEM,
        user_prompt=_REFUSAL_TEMPLATE.format(
            drug1=drug1, drug2=drug2,
            findings1=findings_block_1, findings2=findings_block_2),
        temperature=0.0,
    )

    # The hard-coded refusal block replaces what would have been Bottom Line /
    # Key Trade-offs / Efficacy / Safety / Clinical Bottom Line. Honest, dated,
    # cites the specific screening counts so the reader knows EXACTLY why no
    # comparison is being made.
    refusal_block = (
        f"## Why this report does not compare {drug1} and {drug2}\n\n"
        f"This comparison query retrieved **{n_h2h_retrieved} papers** that "
        f"mentioned both drugs. After strict head-to-head screening (direct "
        f"comparison required, monotherapy required, clinical outcome required, "
        f"human participants required), **{n_h2h_screened} passed**.\n\n"
        f"Aletheon will not invent comparative claims when the underlying "
        f"evidence does not support them. The Bottom Line, Key Trade-offs, "
        f"Efficacy, Safety, and Clinical Bottom Line sections are intentionally "
        f"omitted from this report because the screened head-to-head literature "
        f"is too thin to ground them honestly.\n\n"
        f"What this report DOES provide:\n\n"
        f"- A factual Overview of each drug (below), drawn from its separate "
        f"evidence.\n"
        f"- An Evidence Strength summary for each drug (below).\n"
        f"- The full Evidence Tables (appendix) showing every paper retrieved "
        f"and what was extracted from it.\n"
        f"- The full Sources list.\n\n"
        f"If you need a direct comparison of these two drugs, the indexed "
        f"comparison literature in this retrieval was insufficient. Consider "
        f"querying each drug separately, or treating any apparent comparative "
        f"signal in the evidence tables below as exploratory rather than "
        f"conclusive.\n\n"
    )

    # Combined sources list, both drugs — deduped (reused from rich path).
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
    src = ["\n## Sources\n", f"\n**{drug1}:**\n"] + _doc_lines(ordered1)
    src += [f"\n**{drug2}:**\n"] + _doc_lines(ordered2)

    header = (f"# Aletheon Comparison: {drug1} vs {drug2}\n\n"
              f"_Generated {datetime.now():%Y-%m-%d %H:%M} · "
              f"{len(ordered1)} + {len(ordered2)} evidence chunks · "
              f"{len(findings1)} + {len(findings2)} extracted findings · "
              f"**{n_h2h_screened}/{n_h2h_retrieved} passed head-to-head "
              f"screening — comparison refused** · "
              f"grounded in retrieved sources only_\n\n")

    # Assemble: header → refusal block → descriptive sections from LLM →
    # evidence appendix → sources.
    report_md = header + refusal_block + descriptive_body + "\n" + "\n".join(src)

    # Add the evidence-tables appendix (same structure as the rich path).
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
            "_Per-paper structured extraction. The screening gate rejected all "
            "or most of these for direct comparison, but the structured rows "
            "are still useful for understanding each drug's separate evidence._\n\n"
            + "\n\n".join(compact_blocks)
            + "\n\n"
        )
        if "\n## Sources" in report_md:
            report_md = report_md.replace("\n## Sources",
                                          appendix + "\n## Sources", 1)
        else:
            report_md = report_md + "\n" + appendix

    return report_md


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

    # ---- Step 3b: h2h full-abstract re-extraction ----
    # Standard extraction ran on chunks — text fragments that often lack the key
    # comparative numbers (Drug A outcome vs Drug B outcome, p-value, n per arm).
    # For the head-to-head candidate papers specifically, re-run extraction on the
    # FULL ABSTRACT fetched directly from the source API, using a comparative-
    # specific prompt that explicitly structures both arms' outcomes.
    # Enriched findings override the chunk-extracted ones for h2h candidates only.
    from report.h2h_reextract import reextract_h2h_candidates
    if head_to_head_evs:
        enriched = reextract_h2h_candidates(head_to_head_evs, drug1, drug2)
        if enriched:
            log.info(f"[compare] h2h re-extraction produced {len(enriched)} enriched "
                     f"finding(s) — overriding chunk-based extractions for h2h candidates")
            def _override_findings(findings: list, enriched: dict) -> list:
                result = []
                for f in findings:
                    if f.source_id in enriched:
                        enriched_f = enriched[f.source_id]
                        enriched_f.tag = f.tag  # preserve the A#/B# tag
                        result.append(enriched_f)
                    else:
                        result.append(f)
                return result
            findings1 = _override_findings(findings1, enriched)
            findings2 = _override_findings(findings2, enriched)

    # ---- Step B: head-to-head screening gate ----
    # Before the synthesis sees the findings, run a strict per-paper LLM screen:
    # is THIS paper a direct head-to-head clinical comparison of (drug1, drug2)
    # as monotherapy, on a clinical outcome, in humans? Only YES findings flow
    # into the comparison synthesis. NO findings stay in the evidence base for
    # Overview / Evidence Strength but are explicitly marked "context only."
    # Fixes the audit's #2 / #3 gap (head-to-head strictness + monotherapy QC).
    from report.screen import screen_findings_for_head_to_head, partition_findings
    all_findings = findings1 + findings2
    decisions = screen_findings_for_head_to_head(all_findings, drug1, drug2)
    h2h_findings_1, ctx_findings_1 = partition_findings(findings1, decisions)
    h2h_findings_2, ctx_findings_2 = partition_findings(findings2, decisions)
    n_h2h_screened = len(h2h_findings_1) + len(h2h_findings_2)

    # ---- CONTRACT ENFORCEMENT (hard partition, not soft prompt rule) ----
    # The previous "prompt asks LLM to prefer structured findings" approach
    # didn't work: when the synthesis prompt also contained raw chunks from
    # combination-therapy or single-drug papers, the LLM would cherry-pick
    # comparative-looking numbers from them and quote them in the Bottom Line.
    # That produced false comparative claims with apparently-valid citations —
    # the worst possible failure mode for an evidence-honest tool.
    #
    # The fix is structural: when screening produces too few real head-to-head
    # papers, we DO NOT ask the LLM to write comparative sections at all. We
    # short-circuit to a hard refusal block. The LLM physically cannot quote
    # what isn't in its context.
    if n_h2h_screened < MIN_HEAD_TO_HEAD:
        log.info(f"[compare] only {n_h2h_screened} h2h paper(s) — using refusal "
                 f"mode (skipping LLM synthesis of comparative sections)")
        return _build_refusal_report(
            drug1, drug2, ordered1, ordered2,
            findings1, findings2,
            n_h2h_retrieved=n_h2h, n_h2h_screened=n_h2h_screened,
            sections1=sections1, sections2=sections2,
        )

    # Screening kept enough papers: rich path. Build the synthesis prompt with
    # ONLY the structured findings — no raw chunks (those have already done
    # their job by feeding the extraction step).
    findings_block_1 = findings_to_synthesis_block(h2h_findings_1)
    findings_block_2 = findings_to_synthesis_block(h2h_findings_2)

    client = _get_client()
    log.info(f"[compare] generating {drug1} vs {drug2} "
             f"(rich path — {n_h2h_screened} screened head-to-head paper(s), "
             f"structured findings only — raw chunks suppressed) …")

    # CONTRACT ENFORCEMENT: the synthesis prompt for the rich path receives
    # ONLY structured findings — raw chunks are deliberately omitted. The
    # extraction step has already pulled the relevant numbers out of those
    # chunks; passing the raw text again only invites the LLM to cherry-pick
    # quantitative claims from rejected papers (which is exactly the failure
    # we saw in the 2026-05-31 22:03 benchmark).
    user_prompt = COMPARE_TEMPLATE.format(
        drug1=drug1, drug2=drug2,
        tally1=_tier_tally(ordered1), tally2=_tier_tally(ordered2),
        findings1=findings_block_1, findings2=findings_block_2,
        # Raw evidence intentionally REMOVED from the rich-path prompt.
        # COMPARE_TEMPLATE no longer references {block1}/{block2}.
    )

    body = synthesize(
        system_prompt=COMPARE_SYSTEM,
        user_prompt=user_prompt,
        temperature=config.LLM_TEMPERATURE,
    )

    # ---- Post-generation lint: contract violation detector ----
    # When screening returned 0, the body should not contain sentences that
    # cite both [A#] AND [B#] tags — those are the dead-giveaway of a
    # comparative claim. (We tested broader phrase-scanning like "vs" /
    # "compared to" / "lower than" — those false-positive on legitimate
    # cases like "ibuprofen vs aspirin" appearing in the section title or
    # the comparative *query* being described, and on paper-internal designs
    # like "paracetamol+ibuprofen versus ibuprofen monotherapy". The
    # mixed-tag heuristic is structural: if a sentence pulls evidence from
    # BOTH drug streams to make ONE claim, that claim is comparative by
    # construction.) We log but don't auto-rewrite — surfacing the issue is
    # enough for now; auto-rewriting risks mangling correct content.
    if n_h2h_screened == 0:
        import re as _re_lint
        lint_violations = []
        for sent in _re_lint.split(r"(?<=[\.\!\?])\s+", body):
            has_a = bool(_re_lint.search(r"\[A\d+\]", sent))
            has_b = bool(_re_lint.search(r"\[B\d+\]", sent))
            if has_a and has_b:
                lint_violations.append(sent.strip()[:160])
        if lint_violations:
            log.warning(f"[compare:lint] {len(lint_violations)} sentence(s) "
                        f"cite both [A#] and [B#] despite 0 h2h evidence "
                        f"(comparative claim from non-h2h sources):")
            for v in lint_violations[:5]:
                log.warning(f"[compare:lint]   - {v}…")
        else:
            log.info("[compare:lint] no mixed-tag comparative claims in NO_H2H body")

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

    src = ["\n## Sources\n", f"\n**{drug1}:**\n"] + _doc_lines(ordered1)
    src += [f"\n**{drug2}:**\n"] + _doc_lines(ordered2)

    header = (f"# Aletheon Comparison: {drug1} vs {drug2}\n\n"
              f"_Generated {datetime.now():%Y-%m-%d %H:%M} · "
              f"{len(ordered1)} + {len(ordered2)} evidence chunks · "
              f"{len(findings1)} + {len(findings2)} extracted findings · "
              f"**{n_h2h_screened} passed head-to-head screening** · "
              f"grounded in retrieved sources only_\n\n")

    report_md = header + body + "\n" + "\n".join(src)

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