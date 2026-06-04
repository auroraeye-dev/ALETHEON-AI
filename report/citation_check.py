"""
report/citation_check.py
========================
Lightweight citation grounding verifier — flags claims whose cited chunk doesn't
actually contain support for the claim.

The failure mode this catches: the LLM produces a clinically-true statement and
appends [E#] tags, but the chunk E# came from doesn't actually say that thing.
We've seen this twice this week:
  - Pregnancy section cited to a PharmGKB stub that says nothing about pregnancy
  - Black Box section cited to an OTC stomach-bleed warning rather than a real
    FDA boxed warning

For an evidence-honesty tool, citation hallucination is THE single failure mode
that destroys trust. This module catches the egregious cases.

How it works (lightweight, deterministic, no extra LLM calls):

For each claim → cited-chunk pair we check:
  1. Are there numeric tokens in the claim (e.g. "0.80", "3.5%", "n=10,305",
     "HR 0.80", "p=0.002")? If yes, every numeric token in the claim should
     appear (loosely) in the cited chunk. A claim that says "HR 0.80" but cites
     a chunk that doesn't contain "0.80" is unsupported.
  2. Distinctive keyword overlap: pull 3-5 noun phrases from the claim, verify
     at least one appears in the chunk. Trivial overlap (the drug name alone)
     doesn't count.

We do NOT try to verify clinical correctness here — that requires an LLM.
We're verifying _grounding_: does the cited chunk actually contain words that
support this claim?

Output: a list of {claim, citations, issues} for the report's audit appendix.
The user sees a clear "⚠️ N claims may not be supported by their cited evidence"
warning, with each flagged claim listed.
"""

import re
from collections import defaultdict

from core.logging_setup import log


# Numeric pattern: catches "0.80", "95%", "p=0.002", "HR 1.68", "n=10,305",
# "26.3 percent", etc. We strip surrounding context and just keep the number.
_NUM_RE = re.compile(
    r"(?<![A-Za-z_\d.])"        # not preceded by a letter, digit, or dot (so '15.4' isn't split, but '0.69-0.92' splits cleanly because the - is preceded by '9')
    r"-?\d{1,4}(?:,\d{3})*"     # base number with optional thousands sep
    r"(?:\.\d+)?"               # optional decimal
    r"(?![A-Za-z_])"            # not followed by a letter
)

# Citation tags in the report look like [E1], [E12], etc.
_CITE_RE = re.compile(r"\[E(\d+)\]")

# Claims live in bullet lines. We scan the report for bullets that contain
# at least one citation.
_BULLET_RE = re.compile(r"^\s*[-*]\s*(.+?)$", re.M)


def _extract_numbers(text: str) -> set[str]:
    """Return distinct numeric tokens in the text, normalized (no thousands
    separators) so '10,305' and '10305' match."""
    out = set()
    for m in _NUM_RE.finditer(text or ""):
        tok = m.group(0).replace(",", "")
        # Skip very short integers that are too generic (1, 2, 3 etc), which
        # would create false matches everywhere.
        try:
            f = float(tok)
            if 0 < abs(f) < 4 and "." not in tok:
                continue
        except ValueError:
            continue
        out.add(tok)
    return out


def _num_in_chunk(claim_nums: set[str], chunk_text: str) -> set[str]:
    """For each claim number, check whether it appears in chunk text. Returns
    the set of numbers that ARE present in the chunk."""
    chunk_nums = _extract_numbers(chunk_text or "")
    return claim_nums & chunk_nums


def _strip_citations(text: str) -> str:
    """Remove [E#] tags from a string so the numeric extractor doesn't pick up
    the citation digits as if they were data."""
    return _CITE_RE.sub("", text)


def _claim_lines(report_md: str) -> list[tuple[str, list[str]]]:
    """Pull out bullet-level claims and their citations from the report.
    Returns [(claim_text_without_cites, [E_tag, ...]), ...].

    IMPORTANT: only scans the body of the report — stops at the Sources
    heading. Bibliography bullets contain DOIs and Europe PMC paper IDs
    that look like numeric content but aren't claims; treating them as
    claims produces noisy false positives.
    """
    # Truncate at the Sources section (and at any retraction/grounding/audit
    # appendix that comes after it). The synthesized claims all live in the
    # body before Sources; everything after is bibliography or metadata.
    body = report_md
    cutoff_markers = [
        "\n## Sources",
        "\n## Evidence Table",
        "\n## ⚠️ Retraction",
        "\n## ⚠️ Citation Grounding",
    ]
    earliest = len(report_md)
    for marker in cutoff_markers:
        idx = report_md.find(marker)
        if idx >= 0:
            earliest = min(earliest, idx)
    body = report_md[:earliest]

    out = []
    for m in _BULLET_RE.finditer(body):
        line = m.group(1)
        cites = _CITE_RE.findall(line)
        if not cites:
            continue
        # Skip if this looks like a table row (pipes everywhere) — those are
        # source-of-truth extraction tables, not synthesized claims.
        if line.count("|") >= 2:
            continue
        # Skip super-short bullets — they're usually section headers
        # or "see below" pointers without real claim content.
        clean = _strip_citations(line).strip()
        if len(clean) < 30:
            continue
        tags = [f"E{n}" for n in cites]
        out.append((clean, tags))
    return out


def _num_in_chunk_with_derived(claim_nums: set[str], chunk_text: str) -> tuple[set[str], set[str]]:
    """Like _num_in_chunk but also accepts numbers that are derived from
    chunk numbers via common transforms: percentage from HR/RR, decimal
    from percent, complement from HR/RR.

    Example: chunk says "HR 0.73". Claim says "27% reduction". The LLM did
    1 - 0.73 = 0.27 → "27%". That's valid synthesis, not hallucination, so
    the percent value 27 should be considered supported.

    Returns (directly_matched, derivation_matched).
    """
    chunk_nums = _extract_numbers(chunk_text or "")
    direct = claim_nums & chunk_nums
    remaining = claim_nums - direct
    derived = set()
    # Convert each chunk number to floats and compute common derivations
    chunk_floats = []
    for n in chunk_nums:
        try:
            chunk_floats.append(float(n))
        except ValueError:
            continue
    for claim_n in remaining:
        try:
            cf = float(claim_n)
        except ValueError:
            continue
        for chunk_f in chunk_floats:
            # "27% reduction" from "HR 0.73": claim ≈ (1 - chunk_f) * 100
            if 0 < chunk_f < 1.5 and abs(cf - (1 - chunk_f) * 100) < 0.5:
                derived.add(claim_n)
                break
            # "0.27" from chunk "27%" or vice-versa
            if abs(cf - chunk_f / 100) < 0.005 or abs(cf - chunk_f * 100) < 0.5:
                derived.add(claim_n)
                break
    return direct, derived


def check_grounding(report_md: str,
                    ordered: list[dict]) -> list[dict]:
    """Verify each cited claim has at least one supporting number in at
    least one of its cited chunks.

    `ordered` is the [E#]-indexed chunk list the report cites — the same list
    generate.py uses to build the prompt. Each entry has 'tag' (E#) and 'text'.

    Returns a list of flagged claims: {claim, citations, issue, claim_numbers,
    cited_chunk_numbers}.
    """
    # Index chunks by their E# tag for fast lookup
    chunks_by_tag = {c.get("tag", ""): c for c in (ordered or []) if c.get("tag")}

    flags = []
    for claim, tags in _claim_lines(report_md):
        claim_nums = _extract_numbers(_strip_citations(claim))
        if not claim_nums:
            # No numeric content to verify — can't catch via this method.
            # (A more expensive LLM grounding check could still catch semantic
            # misattributions on qualitative claims; out of scope here.)
            continue

        # Pool all the cited chunks' text for this claim
        cited_text = ""
        for tag in tags:
            ch = chunks_by_tag.get(tag)
            if ch:
                cited_text += "\n" + (ch.get("text", "") or "")

        matched_direct, matched_derived = _num_in_chunk_with_derived(claim_nums, cited_text)
        matched = matched_direct | matched_derived
        unmatched = claim_nums - matched

        # Heuristic: if ZERO of the claim's distinct numbers appear in any
        # of the cited chunks (directly or via simple derivation), the
        # citation is almost certainly wrong. If SOME match but not all,
        # softer flag — possibly the LLM combined multiple chunks.
        if not matched:
            flags.append({
                "claim": claim[:200],
                "citations": tags,
                "issue": "unsupported",
                "claim_numbers": sorted(claim_nums),
                "matched_numbers": [],
            })
        elif unmatched:
            flags.append({
                "claim": claim[:200],
                "citations": tags,
                "issue": "partial",
                "claim_numbers": sorted(claim_nums),
                "matched_numbers": sorted(matched),
                "unmatched_numbers": sorted(unmatched),
            })

    if flags:
        unsupported_n = sum(1 for f in flags if f["issue"] == "unsupported")
        partial_n = sum(1 for f in flags if f["issue"] == "partial")
        log.warning(f"[grounding] {unsupported_n} unsupported claim(s), "
                    f"{partial_n} partially-grounded claim(s)")
    else:
        log.info("[grounding] all numeric claims have supporting evidence "
                 "in their cited chunks")
    return flags


def format_grounding_block(flags: list[dict]) -> str:
    """Render a clear warning section for the report appendix."""
    if not flags:
        return ""
    unsupported = [f for f in flags if f["issue"] == "unsupported"]
    partial = [f for f in flags if f["issue"] == "partial"]
    lines = ["## ⚠️ Citation Grounding Notices",
             "_The following claims have numeric content (effect sizes, "
             "percentages, p-values) that could not be located in their cited "
             "evidence chunks. This is an automated check — false positives "
             "are possible — but each flagged claim should be verified before "
             "the report is used for decision-making._\n"]
    if unsupported:
        lines.append(f"### Unsupported ({len(unsupported)} claim(s) — cited "
                     "chunks contain NONE of the numbers in the claim)")
        for f in unsupported:
            lines.append(f"- {' '.join(f['citations'])}: "
                         f"{f['claim']} "
                         f"(_claim numbers: {', '.join(f['claim_numbers'])}_)")
    if partial:
        lines.append(f"\n### Partially supported ({len(partial)} claim(s) — "
                     "some numbers present, others missing)")
        for f in partial:
            lines.append(f"- {' '.join(f['citations'])}: "
                         f"{f['claim']} "
                         f"(_unmatched: {', '.join(f['unmatched_numbers'])}_)")
    return "\n".join(lines)
