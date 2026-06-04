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
import re
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
    """Format the evidence the LLM sees, grouped by which section it's for.
    For each chunk, we ALSO prepend an EXTRACTED NUMBERS block when the chunk
    contains statistical values — this cues the LLM to quote actual numbers
    instead of writing in adjectives. (Step 1 of the Elicit-gap closure.)"""
    from report.stats_extract import format_numbers_for_prompt
    lines = []
    label = {
        "overview": "OVERVIEW evidence",
        "efficacy": "EFFICACY / FINDINGS evidence",
        "safety": "SAFETY evidence",
        "contradiction": "CONTRADICTION-CHECK evidence (both supportive AND opposing — compare these for conflicts)",
        "preprint": "PREPRINT evidence (NOT peer-reviewed)",
        "dosing": "DOSING & ADMINISTRATION evidence",
        "interactions": "DRUG INTERACTIONS evidence",
        "mechanism": "MECHANISM OF ACTION evidence",
        "populations": "USE IN SPECIFIC POPULATIONS evidence",
        # Med-affairs reviewer additions
        "blackbox": "BLACK BOX evidence (FDA-boxed warnings — these are the most serious regulatory signals)",
        "cv_risk": "CV RISK evidence (cardiovascular outcomes, thrombotic risk, hypertension, heart failure)",
        "pregnancy": "PREGNANCY evidence (pregnancy + lactation + reproductive safety — FDA SPL Section 8)",
        "pk_pd": "PK/PD evidence (pharmacokinetics — absorption, distribution, metabolism, excretion, half-life)",
    }
    for section in ["overview", "blackbox", "safety", "cv_risk", "efficacy",
                    "pk_pd", "dosing", "interactions", "mechanism",
                    "pregnancy", "populations", "contradiction", "preprint"]:
        tags = section_tags.get(section, [])
        if not tags:
            continue
        lines.append(f"\n=== {label[section]} ===")
        seen = set()
        for c in ordered:
            if c["tag"] in tags and c["tag"] not in seen:
                seen.add(c["tag"])
                numbers_block = format_numbers_for_prompt(c["text"])
                # Surface the chunk's source-section label so the LLM can
                # apply section-aware citation discipline. For Black Box
                # specifically, this lets it check "is this actually from a
                # 'boxed warning' chunk or from 'warnings'/'purpose'?" before
                # quoting.
                src_section = c.get("section") or ""
                header = (f"[{c['tag']}] (tier: {c['tier']}, "
                          f"{c['source']}:{c['source_id']}"
                          f"{', source-section: ' + src_section if src_section else ''})")
                if numbers_block:
                    lines.append(f"{header}\n{numbers_block}\nFULL TEXT: {c['text']}")
                else:
                    lines.append(f"{header} {c['text']}")
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
7. Be concise, factual, neutral. No marketing language.
8. QUANTIFY YOUR CLAIMS. When an evidence chunk includes an EXTRACTED NUMBERS \
block (effect sizes, CIs, p-values, percentages, sample sizes), QUOTE the actual \
numbers verbatim when you make a claim from that source — never write in \
adjectives when a number is available. Good: "reduced events with HR 0.60 (95% \
CI 0.31–1.17) [E14]" / "liver enzyme elevation in 62% vs 18% (p=0.009) [E7]". \
Weak (avoid): "showed improved outcomes [E14]" / "had a lower rate [E7]". \
For chunks without an EXTRACTED NUMBERS block (e.g. OTC labels, FAERS summaries), \
prose is fine — only the quantitative chunks demand quantitative writing."""

USER_TEMPLATE = """Drug / query: {drug}

Evidence mix retrieved: {tier_tally}
{tier_authority}

==== STRUCTURED PER-PAPER FINDINGS ====
The PRIMARY input for your synthesis is the structured rows below. Each row is
a per-paper extraction: study type, population, intervention, key numeric
outcomes (with verbatim values), safety signal, and conclusion. When you make
a claim, PREFER quoting an OUTCOME_# field directly (e.g. "GI ulceration: OR
7.58 (95% CI 2.64–21.78)") rather than describing it in adjectives. Each row's
[E#] tag is your citation handle.

{findings}

==== SUPPORTING RAW EVIDENCE ====
The evidence below is the raw, section-grouped chunk text — use it ONLY as
supporting context (for example, when an extracted row was partial or you need
section-specific phrasing like dosing). Always prefer the structured findings
above when both apply. Cite with [E#] tags.

{evidence}

Write a Markdown report with EXACTLY these sections:

## Summary
4-6 sentence executive summary that quotes the most important numbers verbatim \
from the structured findings above (effect sizes, percentages, p-values, sample \
sizes, dose limits) and cites their [E#] tags. This is the FIRST thing a busy \
reader sees — it must stand alone as the "what you need to know." Lead with \
the primary indication, then the most important efficacy finding with its \
number, then the most important safety signal with its number, then a one-line \
note on evidence strength. Example shape: "Drug X is indicated for [primary \
condition]. In [study type, n=...], it showed [outcome: X% vs Y%, p=Z] [E#]. \
The most important safety signal is [signal+number] [E#]. Evidence base is \
[N peer-reviewed studies, M regulatory sources]."

## Key Findings
Most important efficacy/clinical points (use EFFICACY evidence), bulleted, cited.
After EACH finding, append its confidence tag: \
**(Confidence: High|Medium|Low — reason)**.
IMPORTANT: Key Findings must draw ONLY from peer-reviewed and regulatory evidence. \
Do NOT place preprint findings here — preprints belong ONLY in the "Preprint / \
Emerging Evidence" section below. If a point is supported only by a preprint, leave \
it out of Key Findings.

## Black Box Warnings
A black box warning is a SPECIFIC FDA REGULATORY CATEGORY — the most serious \
type of warning in a prescription drug label. It appears in a black-bordered \
box at the very top of an Rx label, NOT in any other section.

STRICT RULES FOR THIS SECTION — every bullet must satisfy ALL of these:
1. The cited chunk MUST come from a section labeled "boxed warning" or \
"black box warning" — check the source-section tag in the chunk header.
2. Consumer warnings like "stomach bleeding warning" on an OTC Drug Facts \
label are NOT boxed warnings, even if they look serious. Do not upgrade them. \
Similarly, severe warnings in Warnings and Precautions (Section 5) of an Rx \
label are NOT boxed warnings — boxed warnings appear above all other sections.
3. If you cannot point to a chunk with source-section "boxed warning" / \
"black box warning", choose ONE of these refusal messages based on what was \
actually retrieved (read the source-section tags carefully):
   - If the retrieved labels are clearly OTC consumer products (titles like \
     "Equate", "care one", "HEB", "Goodys", "Rapidol", or chunks tagged \
     "purpose"/"uses"/"drug facts"): write "No FDA boxed warnings identified — \
     the retrieved labels are OTC consumer products, which do not carry FDA \
     boxed warnings. Prescription strength labels would be needed."
   - If the retrieved labels are prescription labels (titles include the \
     generic/brand drug name, source-section tags include "warnings and \
     precautions", "indications", "dosage and administration", etc.) but \
     none of them include a boxed warning section: write "No FDA boxed \
     warning is present in this drug's prescribing information."
   - If you genuinely can't tell from the retrieved chunks: write "No FDA \
     boxed warnings identified in the retrieved evidence."
4. Do not synthesize a boxed warning from general knowledge of the drug class \
even if you know one exists. If the retrieved chunks don't include it, refuse.
5. If BLACK BOX evidence IS provided AND contains chunks with source-section \
"boxed warning", list each warning as a separate bullet citing the source. \
Each bullet covers: the risk, the affected population, contraindications. \
Quote the warning text verbatim where possible.

## Safety / Warnings
Adverse effects, contraindications, risks (use SAFETY evidence), bulleted, cited.

## Cardiovascular Risk Profile
ONLY if CV RISK evidence is provided above. Summarize cardiovascular signals — \
risk of myocardial infarction, stroke, hypertension, heart failure, thrombotic \
events. Quote effect sizes verbatim where evidence reports them (HR, RR, OR \
with CIs and p-values). If the evidence is regulatory-only (label warnings \
without trial numbers), say so explicitly. If no CV-specific evidence was \
provided, OMIT this section entirely.

## Pharmacokinetics & Pharmacodynamics
ONLY if PK/PD evidence is provided above. Cover these in bullets where \
evidence supports them — DO NOT invent values:
- Absorption: bioavailability, Tmax, food effect
- Distribution: volume of distribution, protein binding
- Metabolism: CYP enzymes involved, primary metabolites
- Excretion: half-life, primary route (renal vs hepatic), fraction unchanged
- Pharmacodynamics: onset of action, duration of effect
Quote numbers verbatim. Omit any bullet for which evidence has no value. If \
no PK/PD evidence was provided at all, OMIT this section entirely.

## Dosing & Administration
ONLY if DOSING & ADMINISTRATION evidence is provided above. Summarize recommended \
doses, routes, frequency, and max limits, bulleted, cited. If no dosing evidence \
was provided, OMIT this section entirely (do not write the header). \
SANITY CHECK every dose you write: oral OTC analgesics are NOT dosed at dozens of \
tablets per day — if the evidence seems to imply an implausibly high count (e.g. \
more than ~12 tablets/day), you have likely misread a label table; re-read it and \
state the correct, plausible limit, and ensure the adult maximum is never wildly \
higher than the elderly maximum.

## Drug Interactions
ONLY if DRUG INTERACTIONS evidence is provided above. Summarize notable \
interactions and what to avoid, bulleted, cited. If no interaction evidence was \
provided, OMIT this section entirely.

## Mechanism of Action
ONLY if MECHANISM OF ACTION evidence is provided above. When it IS provided, you \
MUST create this section and explain how the drug works here (do not fold this \
into the Summary), cited. If no mechanism evidence was provided, OMIT this section entirely.

## Pregnancy, Lactation & Reproductive Safety
IMPORTANT: this is the section where a single sentence of real evidence beats \
silence. If PREGNANCY evidence chunks are present above, USE THEM, even if \
the content is just one warning line. Cover what you can from the evidence:
- Pregnancy contraindications or category (e.g. "contraindicated in pregnancy", \
  "Pregnancy Category X", "do not use in pregnancy", trimester-specific risks)
- Lactation guidance (presence in breast milk, advice for nursing mothers)
- Reproductive safety / fertility / contraception requirements
- Specific warnings for pregnant patients (e.g. "third trimester avoidance")
If the evidence has only partial coverage (e.g. pregnancy but not lactation), \
write only the bullets the evidence supports. Even a single "If pregnant, ask \
a health professional before use" warning from an OTC label is reportable \
content — quote it and cite. Do NOT invent guidance not present in the \
evidence. Only OMIT this section entirely if the PREGNANCY evidence block \
above is genuinely empty (no chunks were provided).

## Use in Specific Populations
Cover guidance for elderly, pediatric, renal impairment, hepatic impairment \
from the USE IN SPECIFIC POPULATIONS evidence. (Pregnancy/lactation has its \
own section above — do NOT duplicate.) Same instruction as Pregnancy: a single \
real warning is reportable. Only OMIT if the evidence block is genuinely empty.

## Contradictions & Disagreements
Examine the CONTRADICTION-CHECK evidence (and all other evidence) for points where \
findings conflict — e.g. one source reports benefit, another reports no effect or \
harm. Surface each conflict with both sides cited. If genuinely none conflict, \
write "No direct contradictions found in current sources."

## Preprint / Emerging Evidence (not yet peer-reviewed)
Use ONLY the PREPRINT evidence, flagged as not peer-reviewed. IMPORTANT: either \
list preprint findings OR write exactly "No preprint evidence in current sources." \
— NEVER both. If you write any preprint bullet, do NOT also write the \
"No preprint evidence" line. Only write that line when there is genuinely no \
preprint evidence at all.

## Clinical Bottom Line
2-4 sentences of CONCRETE practical guidance, grounded ONLY in the evidence \
above. Format: "Use [drug] for [specific indication+context] at [dose]. The \
primary risk to monitor is [signal+number]. Avoid or use cautiously in [specific \
population with cited reason]." Do NOT recommend uses the evidence does not \
support. If the evidence does not support a clear recommendation for a clinical \
context, say so explicitly.

Cite using [E#] tags only. For the optional monograph sections (Dosing, Drug \
Interactions, Mechanism of Action, Use in Specific Populations), include a section \
ONLY when its evidence block was provided — otherwise omit that header completely. \
Do not add any sections beyond those listed."""


# B1 — length guidance injected per depth. This shapes HOW MUCH the LLM writes
# from the (depth-scaled) evidence it's given. Pairs with retrieval depth:
# detailed retrieves more evidence AND asks for fuller prose.
DEPTH_GUIDANCE = {
    "short": (
        "LENGTH: Keep this BRIEF — about one page. Summary in 1-2 sentences. "
        "3-4 of the most important Key Findings only. Safety as a short list of "
        "the top risks. Be concise; omit minor details."
    ),
    "medium": (
        "LENGTH: A balanced report of moderate length (2-3 pages). Cover the key "
        "points in each section without exhaustive detail."
    ),
    "detailed": (
        "LENGTH: A THOROUGH, comprehensive report (4-5 pages). Cover findings in "
        "depth: include specifics (doses, effect sizes, populations, study types) "
        "wherever the evidence provides them. Key Findings should be a fuller list "
        "with explanatory detail per point. Safety should comprehensively cover "
        "adverse effects, contraindications, interactions, and at-risk groups. "
        "Use ALL relevant evidence provided — but NEVER invent detail not in the "
        "evidence; if evidence is thin, say so rather than padding."
    ),
}


def _fix_preprint_fallback(md: str) -> str:
    """If the Preprint section has real bullets, remove any stray
    'No preprint evidence...' fallback line (they're mutually exclusive)."""
    m = re.search(r"(## Preprint / Emerging Evidence[^\n]*\n)(.*?)(?=\n## |\Z)",
                  md, flags=re.S)
    if not m:
        return md
    header, sec = m.group(1), m.group(2)
    has_bullets = bool(re.search(r"^\s*[-*]\s+\S", sec, flags=re.M))
    has_fallback = "no preprint evidence" in sec.lower()
    if has_bullets and has_fallback:
        cleaned = "\n".join(
            ln for ln in sec.splitlines()
            if "no preprint evidence" not in ln.lower())
        md = md[:m.start()] + header + cleaned + md[m.end():]
    return md


def generate_report(drug: str, sections: dict[str, list[dict]], depth: str = "medium",
                    appendices: set | None = None) -> str:
    """Generate the cited report from section-targeted evidence, at the given depth.

    appendices: optional set of {"trace", "stats"} to append F2 traceability and/or
    F3 reported-statistics sections. None/empty keeps the standard clean report.
    """
    section_tags, ordered = _merge_and_tag(sections)
    if not ordered:
        return f"# Aletheon Report: {drug}\n\n_No evidence retrieved._\n"

    # ---- STEP 3: per-paper structured extraction ----
    # Before the synthesis call, turn each retrieved paper into a structured row.
    # The synthesis prompt receives these rows (verbatim numbers paired with
    # outcomes) instead of raw chunks alone — so the model quotes numbers
    # rather than writing in adjectives. Each finding inherits the [E#] tag of
    # its source's first chunk.
    from report.extract import (extract_findings, findings_to_synthesis_block,
                                findings_to_compact_table, findings_to_section_table)
    findings = extract_findings(ordered)
    # Map source_id -> [E#] tag (first chunk encountered for that source).
    sid_to_tag: dict = {}
    for c in ordered:
        sid_to_tag.setdefault(c["source_id"], c["tag"])
    for f in findings:
        f.tag = sid_to_tag.get(f.source_id, "")

    findings_block = findings_to_synthesis_block(findings)
    evidence_block = _evidence_block(section_tags, ordered)
    guidance = DEPTH_GUIDANCE.get(depth, DEPTH_GUIDANCE["medium"])
    client = _get_client()

    log.info(f"[report] generating for {drug!r} (depth={depth}) from {len(ordered)} "
             f"unique chunks ({len(findings)} extracted findings) "
             f"across {len([s for s in sections if sections[s]])} sections …")
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                drug=drug,
                evidence=evidence_block,
                findings=findings_block,
                tier_tally=_tier_tally(ordered),
                tier_authority=TIER_AUTHORITY) + "\n\n" + guidance},
        ],
        temperature=config.LLM_TEMPERATURE,
    )
    # cost/speed tracking (E2)
    try:
        from core.metrics import record_llm
        u = getattr(resp, "usage", None)
        if u is not None:
            record_llm(getattr(u, "prompt_tokens", 0),
                       getattr(u, "completion_tokens", 0), config.LLM_MODEL)
    except Exception:
        pass
    body = resp.choices[0].message.content

    # Safeguard: the model occasionally emits BOTH preprint bullets AND the
    # "No preprint evidence" fallback line in the Preprint section, which
    # contradicts itself. If the preprint section has any real content, strip
    # the stray fallback line so the report can never self-contradict.
    body = _fix_preprint_fallback(body)

    # Sources list — deduped by SOURCE DOCUMENT (A3). One evidence document can
    # be split into several chunks, each with its own [E#] tag for in-text
    # citation. We keep those tags valid, but collapse the visible Sources list
    # so each real document appears ONCE, showing all the tags that point to it.
    sources_lines = ["\n## Sources\n\n"]
    by_doc = {}          # (source, source_id) -> {tags, title, tier, url}
    doc_order = []
    for c in ordered:
        key = (c["source"], c["source_id"])
        if key not in by_doc:
            by_doc[key] = {"tags": [], "title": c["title"], "tier": c["tier"],
                           "url": c["url"], "source": c["source"],
                           "source_id": c["source_id"]}
            doc_order.append(key)
        by_doc[key]["tags"].append(c["tag"])
    for key in doc_order:
        d = by_doc[key]
        tag_str = "".join(f"[{t}]" for t in d["tags"])
        sources_lines.append(
            f"- **{tag_str}** ({d['tier']}) {d['title']} — "
            f"{d['source']}:{d['source_id']}  \n  {d['url']}"
        )
    sources = "\n".join(sources_lines)
    n_docs = len(doc_order)

    # If retrieval was degraded (some sources empty/errored), surface that in
    # the header so the reader knows the evidence base is smaller for an
    # external reason. The reader shouldn't have to scan the log to know.
    degraded_note = ""
    try:
        from core.combine import get_last_source_outcomes
        outcomes = get_last_source_outcomes()
        if outcomes:
            failed = [n for n, (s, _) in outcomes.items() if s in ("empty", "error")]
            if failed:
                degraded_note = (
                    f" · **degraded retrieval** ({len(outcomes) - len(failed)}/"
                    f"{len(outcomes)} sources responded; "
                    f"unavailable: {', '.join(failed)})"
                )
    except Exception:
        # Header degradation note is best-effort; never block the report.
        pass

    header = (f"# Aletheon Report: {drug}\n\n"
              f"_Generated {datetime.now():%Y-%m-%d %H:%M} · "
              f"{len(ordered)} evidence chunks from {n_docs} sources "
              f"(section-targeted, {depth}){degraded_note} · "
              f"grounded in retrieved sources only_\n\n")

    report_md = header + body + "\n" + sources

    # Step A: section restructure. Compact Evidence Table moves from "right
    # after Summary" (where it pushed the synthesis down the page) to the
    # audit-appendix position just before Sources. Per-section sub-tables
    # stay where they are — they're small and they support the prose nearby.
    if findings:
        compact = findings_to_compact_table(findings, heading_level=3)
        if compact:
            appendix = (
                "\n## Evidence Table (Audit Appendix)\n"
                "_Per-paper structured extraction — for verification. The "
                "synthesis above is drawn from these rows; this section lets "
                "you audit each claim back to its source._\n\n"
                + compact + "\n\n"
            )
            if "\n## Sources" in report_md:
                report_md = report_md.replace("\n## Sources",
                                              appendix + "\n## Sources", 1)
            else:
                report_md = report_md + "\n" + appendix

        # Per-section sub-tables stay inline (small, support nearby prose).
        eff_table = findings_to_section_table(findings, focus="efficacy")
        if eff_table:
            report_md = re.sub(
                r"(## Key Findings[^\n]*\n)",
                lambda m: m.group(1) + eff_table + "\n\n",
                report_md, count=1)
        safety_table = findings_to_section_table(findings, focus="safety")
        if safety_table:
            report_md = re.sub(
                r"(## Safety / Warnings[^\n]*\n)",
                lambda m: m.group(1) + safety_table + "\n\n",
                report_md, count=1)

    # Retraction-Watch / Crossref check on the cited evidence. If any cited
    # paper has been retracted/corrected/flagged, surface that prominently —
    # it's exactly the kind of issue our brand promise is built on catching.
    try:
        from report.retraction_check import check_evidence, format_retraction_block
        issues = check_evidence(ordered)
        if issues:
            block = format_retraction_block(issues)
            # insert BEFORE Sources so it's prominent, not buried in an appendix
            report_md = report_md.replace("## Sources",
                                          block + "\n\n## Sources", 1)
            log.warning(f"[report] {len(issues)} retraction/correction flag(s) "
                        f"surfaced in the report")
    except Exception as e:
        log.warning(f"[report] retraction check skipped: {e}")

    # Citation grounding check — verifies that numeric claims in the report
    # actually appear in their cited evidence chunks. Catches the "clinically
    # true content cited to the wrong source" failure mode that broke trust
    # in the OTC-boxed-warning and PharmGKB-pregnancy cases. Deterministic,
    # no extra LLM calls.
    try:
        from report.citation_check import check_grounding, format_grounding_block
        grounding_flags = check_grounding(report_md, ordered)
        if grounding_flags:
            block = format_grounding_block(grounding_flags)
            # Insert before Sources, after any retraction block (so the order is
            # Retraction → Grounding → Sources, both quality warnings adjacent)
            report_md = report_md.replace("## Sources",
                                          block + "\n\n## Sources", 1)
            unsupported_n = sum(1 for f in grounding_flags if f["issue"] == "unsupported")
            log.warning(f"[report] {unsupported_n} unsupported and "
                        f"{len(grounding_flags) - unsupported_n} partially-grounded "
                        f"claim(s) surfaced in the report")
    except Exception as e:
        log.warning(f"[report] grounding check skipped: {e}")

    # Numeric sanity guardrail (always on) — flag dangerous/implausible dosing
    # figures (e.g. an LLM misreading "8 tablets/day" as "48 tablets/day"). We
    # annotate rather than silently rewrite, since we don't invent the right dose.
    from report.sanity import check_report
    report_md, sanity_warnings = check_report(report_md)
    if sanity_warnings:
        for w in sanity_warnings:
            log.warning(f"[sanity] {w}")

    # F2/F3 optional appendices. Off by default so the standard report stays
    # clean; enabled via --trace (traceability) and --stats (reported stats).
    if appendices:
        # Build the [E#] -> source index from the same `ordered` chunks.
        evidence_index = {c["tag"]: {
            "source": c["source"], "source_id": c["source_id"],
            "tier": c["tier"], "title": c["title"], "url": c["url"],
        } for c in ordered}

        if "stats" in appendices:
            from report.stats_extract import extract_from_evidence, format_stats_markdown
            stats = extract_from_evidence(ordered)
            block = format_stats_markdown(stats)
            if block:
                report_md += "\n\n" + block

        if "trace" in appendices:
            from report.traceability import build_traceability, format_traceability_markdown
            trace = build_traceability(report_md, evidence_index)
            block = format_traceability_markdown(trace)
            if block:
                report_md += "\n\n" + block

    return report_md


def save_report(drug: str, report_md: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in drug.lower())
    folder = os.path.join(config.REPORTS_DIR, safe)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"report_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w") as f:
        f.write(report_md)
    log.info(f"[report] saved -> {path}")
    return path
