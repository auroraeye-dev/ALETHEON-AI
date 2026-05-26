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
from sources import fda, clinicaltrials, europepmc, faers

SOURCES = [
    ("fda", fda.fetch),
    ("clinicaltrials", clinicaltrials.fetch),
    ("europepmc", europepmc.fetch),
    ("faers", faers.fetch),
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
