"""
core/combine.py
===============
DAY 4: the COMBINE agent.

Calls every registered source's fetch(drug), merges results into one unified
list[Evidence], dedups, and—crucially—fails GRACEFULLY: if one source's API is
down or blocked (e.g. NCBI on your network), the others still succeed.

Adding a new source later = add one line to SOURCES. That's the whole point of
the Evidence contract: combine doesn't care what a source is, only that it
returns list[Evidence].
"""

from core.models import Evidence
from core.logging_setup import log
from core.relevance import filter_relevant

# Each entry: (name, fetch_function). Only sources that work on your network.
# PubMed is intentionally omitted for now (NCBI blocked on your network);
# re-add it here in one line once you're on a network that allows NCBI.
from sources import fda, clinicaltrials, europepmc, faers, dailymed, pubchem, chembl, pharmgkb

SOURCES = [
    ("fda", fda.fetch),
    ("clinicaltrials", clinicaltrials.fetch),
    ("europepmc", europepmc.fetch),
    ("faers", faers.fetch),
    ("dailymed", dailymed.fetch),
    ("pubchem", pubchem.fetch),
    ("chembl", chembl.fetch),
    ("pharmgkb", pharmgkb.fetch),
]


def get_all_evidence(drug: str) -> list[Evidence]:
    """Fetch from every source, merge, dedup. One dead source can't kill the run."""
    all_evidence: list[Evidence] = []

    from core.cache import cached_fetch
    for name, fetch_fn in SOURCES:
        try:
            evidence = cached_fetch(name, fetch_fn, drug)
            all_evidence.extend(evidence)
        except Exception as e:
            # graceful: log and continue, never crash the whole pipeline
            log.warning(f"[combine] source {name!r} failed: {e}")

    # A1 PRECISION GATE: drop evidence that's about a DIFFERENT drug (sibling
    # leak), keeping anything genuinely about the target or ambiguous.
    before = len(all_evidence)
    all_evidence, dropped = filter_relevant(all_evidence, drug)
    if dropped:
        log.info(f"[combine] precision filter dropped {dropped}/{before} "
                 f"off-drug evidence for {drug!r}")

    # Dedup on (source, source_id) — same doc fetched twice collapses to one.
    seen = set()
    deduped = []
    for ev in all_evidence:
        key = (ev.source, ev.source_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)

    # Quick tally by tier so you can see the evidence mix.
    by_tier = {}
    for ev in deduped:
        by_tier[ev.tier] = by_tier.get(ev.tier, 0) + 1
    tier_summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_tier.items()))
    log.info(f"[combine] {len(deduped)} total evidence for {drug!r} ({tier_summary})")

    return deduped


# ----------------------------------------------------------------------
# Comparison-aware retrieval (step 2 of the synthesis upgrade).
# When the gateway detects a "compare X vs Y" query, the single-drug pipelines
# return their best literature for each drug separately — but the best
# literature for a comparison QUERY is the HEAD-TO-HEAD studies that mention
# both drugs. Those rarely surface in the top results of a single-drug search,
# so we fetch them explicitly here and let the comparator boost them.
#
# Targets Europe PMC and ClinicalTrials only (the two sources where comparative
# clinical studies actually live; FDA labels and FAERS aren't comparative by
# nature, ChEMBL/PubChem/PharmGKB are single-compound).
# ----------------------------------------------------------------------

def fetch_head_to_head(drug1: str, drug2: str) -> list[Evidence]:
    """Search for head-to-head literature mentioning BOTH drugs.

    Returns Evidence items each marked with extra['head_to_head'] = True so the
    comparator can identify and boost them. Survives gracefully if a source
    fails: we log and continue, never crash. May return [] if literature is
    genuinely thin — caller handles fallback messaging."""
    from sources import europepmc, clinicaltrials
    log.info(f"[combine] head-to-head search for {drug1!r} vs {drug2!r} ...")

    # Build a few comparative query phrasings. Europe PMC and ClinicalTrials
    # both default to relevance-ranked text search, so a deliberate "X versus Y"
    # phrasing surfaces direct comparison studies that single-drug searches miss.
    queries = [
        f"{drug1} versus {drug2}",
        f"{drug1} vs {drug2}",
        f"{drug1} compared with {drug2}",
    ]

    out: list[Evidence] = []
    seen_ids = set()

    for q in queries:
        # Europe PMC — primary source of head-to-head literature.
        try:
            results = europepmc.fetch(q)
            for ev in results:
                key = (ev.source, ev.source_id)
                if key in seen_ids:
                    continue
                # Sanity guard: only keep if BOTH drug names appear in the
                # title/abstract. Europe PMC's text search returns relevance-
                # ranked hits but some will only contain one drug.
                text = (ev.title + " " + ev.text).lower()
                if drug1.lower() in text and drug2.lower() in text:
                    ev.extra = dict(ev.extra or {})
                    ev.extra["head_to_head"] = True
                    ev.extra["section"] = "efficacy"  # comparison papers belong in efficacy
                    seen_ids.add(key)
                    out.append(ev)
        except Exception as e:
            log.warning(f"[combine] head-to-head Europe PMC failed for {q!r}: {e}")

        # ClinicalTrials.gov — direct head-to-head trial registrations.
        try:
            trials = clinicaltrials.fetch(q)
            for ev in trials:
                key = (ev.source, ev.source_id)
                if key in seen_ids:
                    continue
                text = (ev.title + " " + ev.text).lower()
                if drug1.lower() in text and drug2.lower() in text:
                    ev.extra = dict(ev.extra or {})
                    ev.extra["head_to_head"] = True
                    ev.extra["section"] = "efficacy"
                    seen_ids.add(key)
                    out.append(ev)
        except Exception as e:
            log.warning(f"[combine] head-to-head ClinicalTrials failed for {q!r}: {e}")

    log.info(f"[combine] head-to-head: found {len(out)} paper(s) mentioning both "
             f"{drug1!r} and {drug2!r}")
    return out

