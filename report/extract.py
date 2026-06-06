"""
report/extract.py
=================
Step 3 — PER-PAPER STRUCTURED EXTRACTION.

The architectural fix that closes the Elicit-gap. Before synthesis, each retrieved
paper is turned into a structured ExtractedFinding row by a focused LLM call.
The synthesis prompt then sees STRUCTURED ROWS, not raw chunks — so it can no
longer write in adjectives, because each row already pins the numbers to their
outcomes ("GI ulceration: OR 7.58 (95% CI 2.64–21.78)").

Why this matters: with raw chunks, the LLM does extraction AND synthesis at the
same time and goes vague. With pre-extracted rows, the synthesis LLM only has
to weigh and combine structured data — the harder extraction job has already
happened, focused on one paper at a time, in a small contained call.

Schema (locked):
  tag, source_id, source, tier, study_type, population, n, intervention_dose,
  key_outcome_1, key_outcome_2, key_outcome_3, safety_signal, conclusion,
  extraction_quality

DESIGN STANCE:
  - Each KEY OUTCOME stays as one STRING (e.g. "GI ulceration: OR 7.58 (95% CI
    2.64-21.78), p<0.001") rather than separate fields, so the number stays
    bound to its outcome — the cross-pairing risk from F3 doesn't repeat here.
  - Max 3 key outcomes — abstracts rarely have more, and the table stays
    scannable.
  - extraction_quality is honest: "complete" / "partial" / "failed". Failed
    rows still appear in the table so the reader knows which papers were
    auditable and which weren't.
  - Caching is critical (re-running same paper across reports = wasted LLM
    calls). Cache key = (source_id, hash(title+text)).
"""

import json
import time
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field

from core.config import config
from core.logging_setup import log


# ---- Schema ----

@dataclass
class ExtractedFinding:
    tag: str = ""                # E# / A# / B# — filled by synthesis step
    source_id: str = ""
    source: str = ""             # europepmc / clinicaltrials / fda / ...
    tier: str = ""
    study_type: str = ""         # "RCT" / "systematic review" / etc.
    population: str = ""
    n: str = ""                  # "n=1,245" / "8,677 patients" / "4 trials"
    intervention_dose: str = ""  # "Aspirin 81mg daily" / "Aspirin 3g/d vs Ibuprofen 1.2g/d"
    key_outcome_1: str = ""      # "GI ulceration: OR 7.58 (95% CI 2.64-21.78)"
    key_outcome_2: str = ""
    key_outcome_3: str = ""
    safety_signal: str = ""      # One-line safety summary
    conclusion: str = ""         # Author's stated conclusion, one sentence
    extraction_quality: str = "complete"  # complete | partial | failed
    title: str = ""              # For the table display
    url: str = ""                # For the table display

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Caching ----
# Reuses the same cache directory as fetch caching but with its own filename
# prefix so they don't collide.

# Bump _EXTRACT_CACHE_VERSION whenever the prompt or post-processing logic
# changes in a way that would produce different output. Old cache entries are
# physically separated by the version stamp in the filename, so they don't
# need to be deleted — they just stop being read. Set in code (not env) so a
# fix author can't forget.
# v1 = original prompt
# v2 = loosened failed/partial/complete criteria + server-side rescue (the
#      "extraction rescue" fix from earlier this session)
_EXTRACT_CACHE_VERSION = "v2"

def _extract_cache_path(source_id: str, content_hash: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in source_id)[:40]
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(
        config.CACHE_DIR,
        f"extract_{_EXTRACT_CACHE_VERSION}_{safe}_{content_hash[:12]}.json",
    )


def _content_hash(title: str, text: str) -> str:
    h = hashlib.sha1((title + "||" + text).encode("utf-8")).hexdigest()
    return h


def _cache_get(source_id: str, content_hash: str) -> ExtractedFinding | None:
    path = _extract_cache_path(source_id, content_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        # cache entries don't expire — extractions are deterministic given
        # the same (paper, model). The cache key includes the content hash, so
        # if the paper text changes, a new entry is created automatically.
        return ExtractedFinding(**data["finding"])
    except Exception as e:
        log.warning(f"[extract] cache read failed for {source_id}: {e}")
        return None


def _cache_put(source_id: str, content_hash: str, finding: ExtractedFinding) -> None:
    path = _extract_cache_path(source_id, content_hash)
    try:
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "finding": finding.to_dict()}, f)
    except Exception as e:
        log.warning(f"[extract] cache write failed for {source_id}: {e}")


# ---- The extraction LLM call ----

_EXTRACT_SYSTEM = (
    "You are an evidence extractor for a drug-intelligence tool. You read ONE "
    "paper's title and abstract and return a STRICT JSON object capturing the "
    "structured findings. You NEVER invent numbers — if the paper doesn't "
    "report a value, leave the field empty (\"\"). You quote numbers VERBATIM "
    "from the paper, paired with their outcomes inside one string.\n\n"
    "RULES:\n"
    "1. study_type: pick the BEST match from {RCT, systematic review, "
    "meta-analysis, cohort study, case-control, observational, label/monograph, "
    "chemistry record, trial registration, other}. \"trial registration\" is "
    "for ClinicalTrials.gov entries without reported results.\n"
    "2. key_outcome_1/2/3: each is ONE string in the format "
    "\"<outcome label>: <verbatim numeric result>\" — e.g. "
    "\"GI ulceration: OR 7.58 (95% CI 2.64-21.78), p<0.001\". Keep numbers "
    "paired with the outcome they belong to. Use 1 outcome if only one is "
    "reported, up to 3. Leave 2/3 empty if not applicable. If the paper reports "
    "qualitative outcomes only (no numbers), describe them in plain text in "
    "key_outcome_1 — don't leave it empty just because numbers are missing.\n"
    "3. safety_signal: ONE sentence on adverse events / safety findings. "
    "Empty if no safety data.\n"
    "4. conclusion: ONE sentence summarizing the paper's stated conclusion, "
    "verbatim if possible. This field is almost always extractable from any "
    "real paper's abstract — only leave empty if the input has no abstract "
    "text at all.\n"
    "5. n: how the paper phrases sample size (\"n=1,245\", \"8,677 patients\", "
    "\"4 trials, 8000 participants\"). Empty if unknown.\n"
    "6. extraction_quality: be GENEROUS, not strict — this field decides "
    "whether the paper feeds downstream synthesis, so over-marking 'failed' "
    "silently throws away real evidence. Use these criteria:\n"
    "   - \"complete\": you filled conclusion + at least one of "
    "(key_outcome_1, safety_signal) + study_type. Most real abstracts hit this.\n"
    "   - \"partial\": you got SOMETHING extractable (at minimum conclusion OR "
    "one outcome) but fields are sparse. PREFER PARTIAL OVER FAILED when in "
    "doubt — a partial extraction is far more useful than a discard.\n"
    "   - \"failed\": ONLY use this if the input has NO extractable clinical "
    "content at all — i.e. a trial registration stub with no results, a pure "
    "chemistry/computational paper with no clinical claim, or no abstract "
    "provided. If you can write even a one-sentence conclusion from the "
    "abstract, the answer is partial, not failed.\n\n"
    "Return ONLY the JSON object — no prose, no markdown fence, no commentary."
)

_EXTRACT_TEMPLATE = """Paper to extract from:

TITLE: {title}
SOURCE: {source}:{source_id}
TIER: {tier}

ABSTRACT/CONTENT:
{text}

Return JSON with keys: study_type, population, n, intervention_dose,
key_outcome_1, key_outcome_2, key_outcome_3, safety_signal, conclusion,
extraction_quality.
"""


def _llm_extract_one(ev: dict, client) -> ExtractedFinding:
    """One paper -> one structured ExtractedFinding via LLM call."""
    title = ev.get("title", "")[:300]
    text = ev.get("text", "")[:4000]  # cap to keep prompts bounded
    source = ev.get("source", "")
    source_id = ev.get("source_id", "")
    tier = ev.get("tier", "")
    url = ev.get("url", "")

    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": _EXTRACT_TEMPLATE.format(
                    title=title, source=source, source_id=source_id,
                    tier=tier, text=text)},
            ],
            temperature=0.0,  # deterministic — extraction must be reproducible
            response_format={"type": "json_object"},
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

        data = json.loads(resp.choices[0].message.content)

        f = ExtractedFinding(
            source_id=source_id,
            source=source,
            tier=tier,
            title=title,
            url=url,
            study_type=str(data.get("study_type", "") or "")[:80],
            population=str(data.get("population", "") or "")[:200],
            n=str(data.get("n", "") or "")[:80],
            intervention_dose=str(data.get("intervention_dose", "") or "")[:200],
            key_outcome_1=str(data.get("key_outcome_1", "") or "")[:300],
            key_outcome_2=str(data.get("key_outcome_2", "") or "")[:300],
            key_outcome_3=str(data.get("key_outcome_3", "") or "")[:300],
            safety_signal=str(data.get("safety_signal", "") or "")[:300],
            conclusion=str(data.get("conclusion", "") or "")[:400],
            extraction_quality=str(data.get("extraction_quality", "complete") or "complete")[:20],
        )

        # SERVER-SIDE RESCUE: the LLM is biased toward "failed" on uncertainty,
        # which silently discards real evidence. If we have ANY salvageable
        # content (conclusion or a key outcome or a safety signal), override
        # to "partial" so screening/synthesis still see the row. The only thing
        # that truly deserves "failed" is a row with no extracted content at all.
        has_content = bool(
            f.conclusion.strip()
            or f.key_outcome_1.strip() or f.key_outcome_2.strip() or f.key_outcome_3.strip()
            or f.safety_signal.strip()
            or f.study_type.strip()
        )
        if f.extraction_quality == "failed" and has_content:
            log.info(f"[extract] rescued LLM-declared-failed paper {source_id}: "
                     f"has content (conclusion={bool(f.conclusion)}, "
                     f"outcomes={bool(f.key_outcome_1 or f.key_outcome_2 or f.key_outcome_3)}, "
                     f"safety={bool(f.safety_signal)}) — reclassifying as 'partial'")
            f.extraction_quality = "partial"
        elif f.extraction_quality == "complete" and not has_content:
            # Mirror-image safety: if the LLM said "complete" but emitted no
            # content, downgrade to failed so it doesn't poison synthesis.
            f.extraction_quality = "failed"
        return f
    except Exception as e:
        # Infrastructure failure (network, timeout, JSON parse) — distinguish
        # this from LLM-judged "failed" so logs are honest about whether
        # papers are lost to API errors vs to content judgment.
        log.warning(f"[extract] LLM extraction errored for {source_id}: {e}")
        return ExtractedFinding(
            source_id=source_id, source=source, tier=tier,
            title=title, url=url,
            extraction_quality="failed",  # treated like failed downstream, but the log marker is "errored"
            conclusion="(extraction call errored — see full evidence chunk in Sources)",
        )


def extract_findings(evidence_chunks: list[dict],
                     max_papers: int = 30,
                     max_parallel: int = 8) -> list[ExtractedFinding]:
    """Extract structured findings from a list of evidence chunks.

    Dedups by source_id first (one paper = one row, regardless of how many
    chunks of it were retrieved). Caches per (source_id, content_hash) so
    re-runs of the same paper are free. Runs up to max_parallel LLM calls
    concurrently. Caps at max_papers to bound cost on dense retrieval runs.
    """
    if not evidence_chunks:
        return []

    # Dedup by source_id, picking the chunk with the longest text as the
    # representative (more text = more to extract from).
    by_src: dict = {}
    for c in evidence_chunks:
        sid = c.get("source_id", "")
        if not sid:
            continue
        if sid not in by_src or len(c.get("text", "")) > len(by_src[sid].get("text", "")):
            by_src[sid] = c

    papers = list(by_src.values())[:max_papers]
    log.info(f"[extract] extracting findings from {len(papers)} unique paper(s) "
             f"(parallel={max_parallel}) …")

    # Check cache first; only LLM-extract the misses.
    findings_by_sid: dict = {}
    to_llm: list[dict] = []
    for p in papers:
        sid = p.get("source_id", "")
        ch = _content_hash(p.get("title", ""), p.get("text", ""))
        cached = _cache_get(sid, ch)
        if cached is not None:
            findings_by_sid[sid] = cached
        else:
            to_llm.append((p, ch))

    n_cached = len(findings_by_sid)
    if to_llm:
        log.info(f"[extract] {n_cached} cached, {len(to_llm)} need LLM extraction")
        from report.generate import _get_client
        client = _get_client()

        def _do_one(item):
            p, ch = item
            f = _llm_extract_one(p, client)
            _cache_put(p.get("source_id", ""), ch, f)
            return p.get("source_id", ""), f

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [pool.submit(_do_one, item) for item in to_llm]
            for fut in as_completed(futures):
                try:
                    sid, f = fut.result()
                    findings_by_sid[sid] = f
                except Exception as e:
                    log.warning(f"[extract] worker failed: {e}")
    else:
        log.info(f"[extract] all {n_cached} papers cached — no LLM calls needed")

    # Preserve original chunk order.
    ordered = []
    seen = set()
    for c in evidence_chunks:
        sid = c.get("source_id", "")
        if sid and sid not in seen and sid in findings_by_sid:
            seen.add(sid)
            ordered.append(findings_by_sid[sid])

    n_complete = sum(1 for f in ordered if f.extraction_quality == "complete")
    n_partial = sum(1 for f in ordered if f.extraction_quality == "partial")
    n_failed = sum(1 for f in ordered if f.extraction_quality == "failed")
    log.info(f"[extract] done: {n_complete} complete, {n_partial} partial, "
             f"{n_failed} failed")
    return ordered


# ---- Rendering: structured rows -> the synthesis-prompt input ----

def findings_to_synthesis_block(findings: list[ExtractedFinding]) -> str:
    """Format extracted findings as the input the synthesis LLM sees.
    Each finding is a labeled block with its [E#] tag and the structured fields.
    The synthesis prompt then writes prose THAT QUOTES THESE FIELDS, rather than
    summarizing raw chunks."""
    lines = []
    for f in findings:
        if f.extraction_quality == "failed":
            continue  # synthesis ignores failed rows (they're still in the table)
        block = [f"[{f.tag}] {f.title or f.source_id} ({f.source}:{f.source_id}, "
                 f"tier: {f.tier}, study_type: {f.study_type})"]
        if f.population:
            block.append(f"  POPULATION: {f.population}")
        if f.n:
            block.append(f"  N: {f.n}")
        if f.intervention_dose:
            block.append(f"  INTERVENTION: {f.intervention_dose}")
        for i, ko in enumerate([f.key_outcome_1, f.key_outcome_2, f.key_outcome_3], 1):
            if ko:
                block.append(f"  OUTCOME_{i}: {ko}")
        if f.safety_signal:
            block.append(f"  SAFETY: {f.safety_signal}")
        if f.conclusion:
            block.append(f"  CONCLUSION: {f.conclusion}")
        lines.append("\n".join(block))
    return "\n\n".join(lines)


# ---- Rendering: the visible table at the top of the report ----

def findings_to_compact_table(findings: list[ExtractedFinding],
                              max_rows: int = 30,
                              heading_level: int = 2) -> str:
    """Render the top-of-report compact 'Evidence Table' as markdown.
    Columns are chosen to be the most informative without making the table too
    wide for screens/PDF — full details remain available in the Sources list.

    heading_level: 2 -> "## Evidence Table" (top-level appendix heading).
                   3 -> "### Evidence Table" (nested under a parent appendix).
                   4 -> drops the heading entirely (caller provides its own).

    Failed-extraction rows (no findings extractable) are EXCLUDED from the
    visible table — they're noise visually. A count footer preserves the
    audit trail ("N papers excluded with no extractable findings")."""
    if not findings:
        return ""
    visible = [f for f in findings if f.extraction_quality in ("complete", "partial")]
    n_failed = len(findings) - len(visible)
    hash_prefix = "#" * heading_level if heading_level in (2, 3) else ""
    heading_line = f"{hash_prefix} Evidence Table" if hash_prefix else ""
    if not visible:
        # All papers failed — be honest, don't render an empty table.
        prefix = heading_line + "\n" if heading_line else ""
        return (f"{prefix}_All {len(findings)} retrieved papers "
                f"yielded no extractable findings — see Sources for raw chunks._")
    rows = visible[:max_rows]
    lines = []
    if heading_line:
        lines.append(heading_line)
    lines.extend([
        "_Per-paper structured extraction. Each row is the LLM's "
        "extraction of one source, with numbers quoted verbatim. "
        "Quality flags: ✓ complete · ⚠ partial._\n",
        "| Tag | Quality | Source | Study type | n | Key finding | Conclusion |",
        "|-----|---------|--------|------------|---|-------------|------------|"
    ])
    for f in rows:
        q = {"complete": "✓", "partial": "⚠"}.get(f.extraction_quality, "?")

        def clip(s: str, n: int) -> str:
            """Smart truncation that cuts at WORD boundaries (never mid-word)
            and tries to preserve statistical content (numbers, CIs, p-values,
            HRs, percentages) — the parts a reviewer actually needs.

            If a string contains a statistical marker AND is over budget, we
            keep everything from the start of the outcome up through the END
            of the first stats-bearing parenthetical, then ellipsis. The
            comparator (which appears right after the headline number) is
            usually the most important content to preserve.
            """
            s = (s or "").replace("|", "/").replace("\n", " ")
            s = " ".join(s.split())  # collapse whitespace
            if len(s) <= n:
                return s.strip()
            # Word-boundary cut: never split mid-word
            cut = s[:n].rsplit(" ", 1)[0].rstrip(",;:")
            return cut + "…"

        short_src = (f.title or f.source_id)[:35]
        if len(f.title or f.source_id) > 35:
            short_src += "…"
        short_src = short_src.replace("|", "/").replace("\n", " ")
        key = f.key_outcome_1 or f.safety_signal or "—"
        # Higher limits (200 / 200) than before (120 / 130) — gives a full
        # statistical claim including effect size, CI, comparator, AND p-value
        # room to fit without forcing truncation. The PFS comparison line we
        # care about ("28·8 vs 6·8 months ... HR 0·33 ... p<0·0001") is ~192
        # chars and now fits fully. PDF cell auto-wraps if a row is unusually
        # long; that's acceptable for the audit appendix.
        lines.append(f"| {f.tag} | {q} | {short_src} | "
                     f"{clip(f.study_type, 25)} | {clip(f.n, 20)} | "
                     f"{clip(key, 200)} | {clip(f.conclusion, 200)} |")
    if n_failed:
        lines.append(f"\n_{n_failed} additional paper(s) retrieved had no "
                     f"extractable findings (e.g. trial registrations with no "
                     f"reported results, or chemistry/preprint records). "
                     f"They remain available in Sources for traceability._")
    return "\n".join(lines)


def findings_to_section_table(findings: list[ExtractedFinding],
                              focus: str = "efficacy") -> str:
    """A narrower per-section sub-table for inline use under each section
    header. focus='efficacy' shows key_outcome_1; focus='safety' shows safety_signal.
    """
    if not findings:
        return ""

    def _clip_word(s: str, n: int) -> str:
        """Word-boundary truncation — no mid-word cuts."""
        s = (s or "").replace("|", "/").replace("\n", " ")
        s = " ".join(s.split())
        if len(s) <= n:
            return s.strip()
        cut = s[:n].rsplit(" ", 1)[0].rstrip(",;:")
        return cut + "…"

    lines = []
    if focus == "safety":
        lines.append("\n_Findings used in this section:_\n")
        lines.append("| Tag | Source | Safety signal |")
        lines.append("|-----|--------|---------------|")
        any_row = False
        for f in findings:
            if f.extraction_quality == "failed" or not f.safety_signal:
                continue
            short = _clip_word(f.title or f.source_id, 50)
            sig = _clip_word(f.safety_signal, 220)
            lines.append(f"| {f.tag} | {short} | {sig} |")
            any_row = True
        if not any_row:
            return ""
        return "\n".join(lines)

    # default: efficacy / key outcome
    lines.append("\n_Findings used in this section:_\n")
    lines.append("| Tag | Source | Study type | Key finding |")
    lines.append("|-----|--------|------------|-------------|")
    any_row = False
    for f in findings:
        if f.extraction_quality == "failed" or not f.key_outcome_1:
            continue
        short = _clip_word(f.title or f.source_id, 50)
        ko = _clip_word(f.key_outcome_1, 220)
        lines.append(f"| {f.tag} | {short} | {_clip_word(f.study_type, 20)} | {ko} |")
        any_row = True
    if not any_row:
        return ""
    return "\n".join(lines)