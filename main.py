"""
main.py
=======
The single entry point for Aletheon.

Day 1:  python main.py            -> prints "alive" + a sample Evidence
Day 3+: python main.py "aspirin"  -> runs the full pipeline -> cited report
"""

import sys

from core.models import Evidence, TIER_PEER_REVIEWED
from core.config import config
from core.logging_setup import log


def healthcheck():
    """Day-1 proof that the skeleton stands and imports work."""
    log.info("Aletheon is alive ✅")

    missing = config.check()
    if missing:
        log.warning("Missing config (fine for Day 1): " + "; ".join(missing))

    # Prove the Evidence contract works by building one by hand.
    sample = Evidence(
        source="pubmed",
        source_id="12345",
        title="A sample paper about aspirin and cardiovascular outcomes",
        text="This is placeholder abstract text.",
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        tier=TIER_PEER_REVIEWED,
        doc_type="paper",
        date="2024-01-01",
    )
    log.info("Sample evidence created: " + sample.short())


def run_pipeline(drug: str):
    """Day 3+ : the real flow. Stubbed until then."""
    log.info(f"Running Aletheon pipeline for: {drug!r}")
    log.info("Pipeline not built yet — see the 14-day plan. (Day 3 closes the loop.)")
    # Future shape:
    #   evidence = combine.get_all_evidence(drug)
    #   vectorstore.index(chunk_and_embed(evidence))
    #   chunks = retrieve.search(drug)
    #   report = generate.report(drug, chunks)
    #   save(report)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pipeline(sys.argv[1])
    else:
        healthcheck()
