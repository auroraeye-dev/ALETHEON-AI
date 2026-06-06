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
        "blackbox": "BLACKBOX (FDA boxed warning) candidate evidence",
        "cv_risk": "CV_RISK (cardiovascular risk) evidence",
        "pk_pd": "PK_PD (pharmacokinetics & pharmacodynamics) evidence",
        "pregnancy": "PREGNANCY / LACTATION / REPRODUCTIVE SAFETY evidence",
    }
    for section in ["overview", "efficacy", "safety", "blackbox", "cv_risk",
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
                header = f"[{c['tag']}] (tier: {c['tier']}, {c['source']}:{c['source_id']})"
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
prose is fine — only the quantitative chunks demand quantitative writing.
9. NEVER USE PLACEHOLDERS for missing values. If a sample size, p-value, CI, \
or any other number ISN'T in the extracted finding, do NOT write "n=...", \
"(p=?)", "[CI not reported]", "in [year]", or any similar placeholder. Instead: \
either OMIT the parenthetical entirely, or REPHRASE the sentence so it doesn't \
imply a number you don't have. Bad: "In a trial (n=...; P=0.005), atorvastatin \
reduced events". Good: "In a trial of patients with acute coronary syndrome, \
atorvastatin reduced events (P=0.005) [E8]." Or omit P if you don't have it."""

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

## Safety / Warnings
Adverse effects, contraindications, risks (use SAFETY evidence), bulleted, cited.

## Black Box Warnings
ONLY if BLACKBOX evidence is provided above AND it describes an FDA boxed warning \
(the most serious type of warning in a prescription drug label — appears in a \
black-bordered box at the top of the official label). Quote / paraphrase the \
boxed warning faithfully, cited. CRITICAL DISTINCTION: OTC product warnings, \
stomach-bleed warnings, allergy warnings on OTC labels, and general "Warnings" \
sections are NOT boxed warnings. If the evidence does NOT contain a real FDA \
boxed warning, write EXACTLY: "No FDA boxed warning is present in this drug's \
prescribing information." Do NOT manufacture a boxed warning from OTC label text.

## Cardiovascular Risk Profile
ONLY if CV_RISK evidence is provided above. Summarize the drug's cardiovascular \
safety/efficacy signals from the evidence (MACE, stroke, MI, CV death rates with \
their effect sizes and CIs), bulleted, cited. If no CV-specific evidence was \
provided, write "Insufficient CV-specific evidence in retrieved sources."

## Pharmacokinetics & Pharmacodynamics
ONLY if PK_PD evidence is provided above. Summarize absorption, distribution, \
metabolism, elimination, half-life, protein binding, and any clinically relevant \
PK/PD parameters with their values verbatim from the evidence, cited. If no PK/PD \
evidence was provided, write "Insufficient PK/PD evidence in retrieved sources."

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
into the Summary), cited. If no mechanism evidence was provided, write \
"Insufficient mechanism-of-action evidence in retrieved sources."

## Pregnancy, Lactation & Reproductive Safety
ONLY if PREGNANCY evidence is provided above. Summarize guidance for pregnancy, \
lactation, and reproductive safety (teratogenicity, fetal risk category, \
contraindication status, breastfeeding recommendations) verbatim where possible, \
cited. If no pregnancy/lactation evidence was provided, write \
"Insufficient pregnancy/lactation evidence in retrieved sources."

## Use in Specific Populations
ONLY if USE IN SPECIFIC POPULATIONS evidence is provided above. Summarize guidance \
for elderly, pediatric, and renal/hepatic impairment (NOT pregnancy — that has its \
own section above), cited. If no such evidence was provided, write \
"Insufficient population-specific evidence in retrieved sources."

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
    # PRISMA stage 3 (Eligibility): how many reports went into extraction, how
    # many came back with complete/partial findings, how many failed. We track
    # the actual reasons so a reviewer can audit them.
    try:
        import core.combine as _combine
        complete = sum(1 for f in findings if getattr(f, "extraction_quality", "") == "complete")
        partial = sum(1 for f in findings if getattr(f, "extraction_quality", "") == "partial")
        failed = len(findings) - complete - partial
        _combine._PRISMA_COUNTS["reports_assessed"] = len(findings)
        excluded = dict(_combine._PRISMA_COUNTS.get("reports_excluded", {}) or {})
        if failed:
            excluded["Failed LLM extraction (no parseable findings)"] = failed
        _combine._PRISMA_COUNTS["reports_excluded"] = excluded
        # PRISMA stage 4 (Included): only papers with at least one extracted
        # finding (complete OR partial) count as "Included in synthesis". This
        # OVERRIDES the count from _retrieve_node, which was based on unique
        # source_ids in the retrieved chunks — that produced included > assessed
        # because raw chunks reach synthesis even when extraction failed. The
        # correct definition matches PRISMA: included = assessed - excluded.
        _combine._PRISMA_COUNTS["studies_included"] = complete + partial
        _combine._PRISMA_COUNTS["reports_included"] = complete + partial
    except Exception as e:
        log.warning(f"[report] PRISMA extraction counts skipped: {e}")
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

    # Safeguard: catch placeholder text like "n=...", "(p=?)", "[CI not reported]"
    # that the LLM sometimes emits despite Rule 9. These look like the model
    # invented a number it didn't have. Strip the offending parenthetical and
    # log a warning so we know the prompt rule isn't holding 100% of the time.
    import re as _re
    _placeholder_patterns = [
        # "(n=...; P=0.005)" → "(P=0.005)" — strip just the n=... part
        (_re.compile(r"\bn\s*=\s*\.{2,}\s*[;,]?\s*"), ""),
        # "(n=...)" entire parenthetical → drop the whole parenthetical
        (_re.compile(r"\s*\(\s*n\s*=\s*\.{2,}\s*\)"), ""),
        # "[year=...]" or similar bracketed placeholders
        (_re.compile(r"\s*\[\s*[a-z ]{2,20}=\s*\.{2,}\s*\]"), ""),
        # "p=?" / "P=?" or "p=...; "
        (_re.compile(r"\b[pP]\s*=\s*[?\.]{1,3}\s*[;,]?\s*"), ""),
        # "(CI not reported)" / "[not reported]" etc.
        (_re.compile(r"\s*[\(\[](?:CI |95% CI |HR |RR |p[- ])?not\s+reported[\)\]]",
                     _re.IGNORECASE), ""),
        # Empty parens left behind: "(; )" or "(  )"
        (_re.compile(r"\s*\(\s*[;,]?\s*\)"), ""),
    ]
    flagged = False
    for pat, repl in _placeholder_patterns:
        if pat.search(body):
            body = pat.sub(repl, body)
            flagged = True
    if flagged:
        log.warning("[report] stripped placeholder text from LLM output "
                    "(e.g. 'n=...', 'p=?', 'not reported') — prompt rule 9 "
                    "wasn't followed perfectly. Synthesis still cited the "
                    "available numbers; the missing-value placeholders are gone.")

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

    # Degraded-retrieval badge: when some sources failed or returned empty,
    # surface that in the header so a reviewer sees the data limit immediately
    # rather than discovering it in the log.
    degraded_note = ""
    try:
        import core.combine as _combine
        outcomes = dict(getattr(_combine, "_LAST_SOURCE_OUTCOMES", {}) or {})
        if outcomes:
            ok = [n for n, (k, _) in outcomes.items() if k == "ok"]
            bad = [n for n, (k, _) in outcomes.items() if k != "ok"]
            if bad:
                degraded_note = (f" · **degraded retrieval** ({len(ok)}/{len(outcomes)} "
                                 f"sources responded; unavailable: {', '.join(sorted(bad))})")
    except Exception:
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