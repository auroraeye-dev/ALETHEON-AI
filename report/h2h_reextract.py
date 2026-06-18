"""
report/h2h_reextract.py
=======================
HEAD-TO-HEAD FULL-ABSTRACT RE-EXTRACTION NODE.

Problem this solves:
    The standard extraction pipeline runs on CHUNKS — text fragments produced
    by the chunker from a representative snippet of each paper. For most papers
    this is fine. For head-to-head comparison papers it fails badly: the key
    numbers (Drug A HbA1c -1.2%, Drug B HbA1c -0.8%, p=0.003) are often not
    in the representative chunk, so the extractor returns empty fields, the
    screener sees no clinical outcome, and the paper is correctly but wastefully
    rejected.

Fix:
    For every paper identified as a head-to-head candidate (tagged
    ev.extra['head_to_head'] = True by combine.fetch_head_to_head), this module:

    1. Fetches the FULL ABSTRACT from EuropePMC's article API (for PMID sources)
       or uses the full text already stored in ev.text (for ClinicalTrials and
       other sources that already return full text).
    2. Runs a COMPARATIVE-SPECIFIC extraction prompt that explicitly asks for
       Drug A arm outcome, Drug B arm outcome, between-arm statistic, p-value,
       n per arm.
    3. Caches results under a separate key prefix (extract_h2h_v1_*) so they
       don't collide with or overwrite the standard per-drug extraction cache.
    4. Returns enriched ExtractedFinding objects that replace the chunk-based
       ones for h2h candidates before the screening step.

Usage in compare.py:
    from report.h2h_reextract import reextract_h2h_candidates
    enriched = reextract_h2h_candidates(head_to_head_evs, drug1, drug2)
"""

import json
import os
import time
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import config
from core.logging_setup import log
from report.extract import ExtractedFinding

_H2H_CACHE_VERSION = "v1"
_EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(source_id: str, content_hash: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in source_id)[:40]
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(
        config.CACHE_DIR,
        f"extract_h2h_{_H2H_CACHE_VERSION}_{safe}_{content_hash[:12]}.json",
    )


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_get(source_id: str, content_hash: str):
    path = _cache_path(source_id, content_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return ExtractedFinding(**data["finding"])
    except Exception as e:
        log.warning(f"[h2h_reextract] cache read failed for {source_id}: {e}")
        return None


def _cache_put(source_id: str, content_hash: str, finding: ExtractedFinding) -> None:
    path = _cache_path(source_id, content_hash)
    try:
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "finding": finding.to_dict()}, f)
    except Exception as e:
        log.warning(f"[h2h_reextract] cache write failed for {source_id}: {e}")


# ---------------------------------------------------------------------------
# Full abstract fetcher
# ---------------------------------------------------------------------------

def _fetch_full_abstract_epmc(pmid: str) -> str:
    """Fetch the full abstract for a PMID from EuropePMC's search API."""
    try:
        params = {
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "format": "json",
            "pageSize": 1,
            "resultType": "core",
        }
        r = requests.get(_EPMC_SEARCH_URL, params=params,
                         headers=_HEADERS, timeout=20)
        r.raise_for_status()
        results = r.json().get("resultList", {}).get("result", [])
        if results:
            abstract = results[0].get("abstractText", "")
            title = results[0].get("title", "")
            if abstract:
                return f"TITLE: {title}\n\nABSTRACT: {abstract}"
    except Exception as e:
        log.warning(f"[h2h_reextract] EPMC abstract fetch failed for {pmid}: {e}")
    return ""


def _get_best_text(ev) -> str:
    """Get the best available text for a h2h evidence item.
    Priority: full abstract from EPMC API > ev.text already stored."""
    if ev.source == "europepmc" and ev.source_id.isdigit():
        full = _fetch_full_abstract_epmc(ev.source_id)
        if full and len(full) > len(ev.text or ""):
            return full
    return f"TITLE: {ev.title or ''}\n\nTEXT: {ev.text or ''}"


# ---------------------------------------------------------------------------
# Comparative extraction prompt
# ---------------------------------------------------------------------------

_H2H_EXTRACT_SYSTEM = (
    "You are a comparative evidence extractor for a drug-intelligence platform. "
    "You read ONE paper that directly compares two drugs and return a STRICT JSON "
    "object capturing the structured comparative findings. You NEVER invent "
    "numbers — if the paper doesn't report a value, leave the field empty (\"\"). "
    "You quote numbers VERBATIM from the abstract, paired with their outcomes.\n\n"
    "RULES:\n"
    "1. study_type: pick the BEST match from {RCT, systematic review, "
    "meta-analysis, cohort study, case-control, observational, label/monograph, "
    "trial registration, other}.\n"
    "2. This paper compares DRUG1 vs DRUG2. Structure your extraction around "
    "the COMPARISON:\n"
    "   - key_outcome_1: the PRIMARY comparative endpoint with BOTH arms' "
    "numbers. Format: \"<outcome>: DRUG1 <value> vs DRUG2 <value> (<statistic>)\""
    " e.g. \"HbA1c reduction: metformin -1.2% vs glipizide -0.8% (p=0.003)\"\n"
    "   - key_outcome_2: a SECONDARY comparative endpoint (same format), or "
    "a safety comparison, or \"\" if none reported.\n"
    "   - key_outcome_3: another secondary endpoint or \"\".\n"
    "3. intervention_dose: describe BOTH arms with doses if reported.\n"
    "4. n: total sample size AND per-arm if reported.\n"
    "5. safety_signal: ONE sentence comparing adverse events between arms "
    "with numbers if reported.\n"
    "6. conclusion: the paper's stated conclusion in ONE sentence.\n"
    "7. population: describe the study population concisely.\n"
    "8. extraction_quality:\n"
    "   - \"complete\": filled conclusion + at least ONE comparative outcome "
    "with numbers from BOTH arms.\n"
    "   - \"partial\": some content but comparative numbers incomplete.\n"
    "   - \"failed\": NO extractable clinical content.\n"
    "PREFER partial over failed when in doubt.\n\n"
    "Return ONLY the JSON object — no prose, no markdown fence."
)

_H2H_EXTRACT_TEMPLATE = """Drug pair being compared: {drug1} vs {drug2}

Paper:
{text}

Extract the comparative findings. Both arms must appear in key_outcome_1 if
the paper reports them. Quote numbers verbatim. Return JSON."""


# ---------------------------------------------------------------------------
# LLM extraction call
# ---------------------------------------------------------------------------

def _llm_extract_h2h(ev, drug1: str, drug2: str, text: str, client) -> ExtractedFinding:
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": _H2H_EXTRACT_SYSTEM},
                {"role": "user", "content": _H2H_EXTRACT_TEMPLATE.format(
                    drug1=drug1, drug2=drug2, text=text[:6000],
                )},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        try:
            from core.metrics import record_llm
            u = getattr(resp, "usage", None)
            if u is not None:
                record_llm(getattr(u, "prompt_tokens", 0),
                           getattr(u, "completion_tokens", 0), config.LLM_MODEL)
        except Exception:
            pass

        data = json.loads(resp.choices[0].message.content)

        quality = str(data.get("extraction_quality", "partial"))
        has_content = any([
            data.get("key_outcome_1"),
            data.get("key_outcome_2"),
            data.get("conclusion"),
            data.get("safety_signal"),
        ])
        if quality == "failed" and has_content:
            quality = "partial"

        return ExtractedFinding(
            tag="",
            source_id=ev.source_id,
            source=ev.source,
            tier=ev.tier,
            study_type=str(data.get("study_type", ""))[:50],
            population=str(data.get("population", ""))[:200],
            n=str(data.get("n", ""))[:60],
            intervention_dose=str(data.get("intervention_dose", ""))[:200],
            key_outcome_1=str(data.get("key_outcome_1", ""))[:400],
            key_outcome_2=str(data.get("key_outcome_2", ""))[:400],
            key_outcome_3=str(data.get("key_outcome_3", ""))[:400],
            safety_signal=str(data.get("safety_signal", ""))[:300],
            conclusion=str(data.get("conclusion", ""))[:400],
            extraction_quality=quality,
            title=ev.title or "",
            url=ev.url or "",
        )

    except Exception as e:
        log.warning(f"[h2h_reextract] LLM call failed for {ev.source_id}: {e}")
        return ExtractedFinding(
            source_id=ev.source_id,
            source=ev.source,
            tier=ev.tier,
            extraction_quality="failed",
            title=ev.title or "",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reextract_h2h_candidates(
    h2h_evidence: list,
    drug1: str,
    drug2: str,
    max_parallel: int = 8,
) -> dict:
    """Re-extract full abstracts for h2h candidate papers using a
    comparative-specific prompt.

    Returns dict mapping source_id -> ExtractedFinding.
    Only papers with extraction_quality != 'failed' are included.
    """
    if not h2h_evidence:
        return {}

    # Deduplicate by source_id
    seen = {}
    for ev in h2h_evidence:
        if ev.source_id not in seen:
            seen[ev.source_id] = ev
    candidates = list(seen.values())

    log.info(f"[h2h_reextract] re-extracting {len(candidates)} h2h candidate(s) "
             f"for {drug1!r} vs {drug2!r} (parallel={max_parallel}) …")

    results: dict = {}
    to_llm: list = []

    for ev in candidates:
        text = _get_best_text(ev)
        ch = _content_hash(text)
        cached = _cache_get(ev.source_id, ch)
        if cached is not None:
            results[ev.source_id] = (cached, ch)
        else:
            to_llm.append((ev, text, ch))

    n_cached = len(results)
    n_llm = len(to_llm)

    if to_llm:
        log.info(f"[h2h_reextract] {n_cached} cached, {n_llm} need LLM re-extraction")
        from report.generate import _get_client
        client = _get_client()

        def _do_one(ev, text, ch):
            finding = _llm_extract_h2h(ev, drug1, drug2, text, client)
            _cache_put(ev.source_id, ch, finding)
            return ev.source_id, finding, ch

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [pool.submit(_do_one, ev, text, ch)
                       for ev, text, ch in to_llm]
            for fut in as_completed(futures):
                try:
                    sid, finding, ch = fut.result()
                    results[sid] = (finding, ch)
                except Exception as e:
                    log.warning(f"[h2h_reextract] worker failed: {e}")
    else:
        log.info(f"[h2h_reextract] all {n_cached} h2h re-extractions cached")

    findings_only = {sid: f for sid, (f, _) in results.items()}
    n_complete = sum(1 for f in findings_only.values() if f.extraction_quality == "complete")
    n_partial  = sum(1 for f in findings_only.values() if f.extraction_quality == "partial")
    n_failed   = sum(1 for f in findings_only.values() if f.extraction_quality == "failed")
    log.info(f"[h2h_reextract] done: {n_complete} complete, "
             f"{n_partial} partial, {n_failed} failed")

    return {sid: f for sid, f in findings_only.items()
            if f.extraction_quality != "failed"}