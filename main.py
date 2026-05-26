"""
main.py
=======
Entry point for Aletheon.

  python main.py                  -> healthcheck (Day 1)
  python main.py ingest aspirin   -> fetch PubMed -> chunk -> embed -> store (Day 2)
  python main.py search "does aspirin prevent stroke?"  -> retrieve test (Day 2)
  python main.py "aspirin"        -> full pipeline (Day 3, not built yet)
"""

import sys

from core.models import Evidence, TIER_PEER_REVIEWED
from core.config import config
from core.logging_setup import log


def healthcheck():
    log.info("Aletheon is alive ✅")
    missing = config.check()
    if missing:
        log.warning("Missing config: " + "; ".join(missing))
    sample = Evidence(
        source="pubmed", source_id="12345",
        title="A sample paper about aspirin and cardiovascular outcomes",
        text="This is placeholder abstract text.",
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        tier=TIER_PEER_REVIEWED, doc_type="paper", date="2024-01-01",
    )
    log.info("Sample evidence created: " + sample.short())


def ingest(drug: str, reset: bool = False):
    """Day 4: fetch from ALL working sources -> combine -> chunk -> embed -> store."""
    from core.combine import get_all_evidence
    from core.chunk import chunk_all
    from core.embed import embed_texts
    from storage import vectorstore

    log.info(f"=== INGEST: {drug!r} ===")
    if reset:
        vectorstore.reset()
        log.info("[ingest] store reset — starting clean")
    evidence = get_all_evidence(drug)
    if not evidence:
        log.warning("No evidence fetched — stopping.")
        return

    chunks = chunk_all(evidence, drug=drug.lower().strip())
    log.info(f"Chunked {len(evidence)} papers -> {len(chunks)} chunks")

    vectors = embed_texts([c.text for c in chunks])
    vectorstore.index_chunks(chunks, vectors)
    log.info(f"Done. Qdrant now holds {vectorstore.count()} chunks total.")


def search(query: str):
    """Day 2: embed a query and show what comes back from Qdrant."""
    from core.embed import embed_query
    from storage import vectorstore

    log.info(f"=== SEARCH: {query!r} ===")
    qvec = embed_query(query)
    hits = vectorstore.search(qvec)
    if not hits:
        log.info("No hits. Did you run `ingest` first?")
        return
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"\n#{i}  score={h.score:.3f}  [{p['tier']}] {p['source']}:{p['source_id']}")
        print(f"    {p['title'][:80]}")
        print(f"    {p['text'][:160]}…")


def run_pipeline(drug: str, depth: str = "medium"):
    """Full flow — SECTION-TARGETED retrieval -> generate cited report, at `depth`."""
    from core.retrieve import retrieve_for_report
    from report.generate import generate_report, save_report

    log.info(f"=== REPORT PIPELINE: {drug!r} (depth={depth}) ===")
    sections = retrieve_for_report(drug, depth=depth)
    if not any(sections.values()):
        log.warning("No evidence in store. Run `python main.py ingest <drug>` first.")
        return

    report_md = generate_report(drug, sections, depth=depth)
    path = save_report(drug, report_md)

    print("\n" + "=" * 70)
    print(report_md)
    print("=" * 70)
    print(f"\nSaved to: {path}\n")


if __name__ == "__main__":
    args = sys.argv[1:]

    def _parse_depth(arglist):
        if "--short" in arglist:
            return "short"
        if "--detailed" in arglist:
            return "detailed"
        return "medium"

    FLAGS = {"--reset", "--short", "--detailed", "--medium", "--critic"}

    if not args:
        healthcheck()
    elif args[0] == "cache-clear":
        from core.cache import clear
        n = clear()
        print(f"Cleared {n} cache entries.")
    elif args[0] == "ingest" and len(args) > 1:
        reset = "--reset" in args
        drug = next(a for a in args[1:] if a not in FLAGS)
        ingest(drug, reset=reset)
    elif args[0] == "search" and len(args) > 1:
        search(" ".join(a for a in args[1:] if a not in FLAGS))
    elif args[0] == "flow" and len(args) > 1:
        # Full orchestrated pipeline (LangGraph): parallel fetch -> combine ->
        # index -> retrieve -> report (-> evaluate/correct -> critic).
        from core.graph import run as run_flow
        reset = "--reset" in args
        depth = _parse_depth(args)
        critic = "--critic" in args
        drug = next(a for a in args[1:] if a not in FLAGS)
        result = run_flow(drug, reset=reset, depth=depth, critic=critic)
        print("\n" + "=" * 70)
        print(result["report"])
        print("=" * 70)
        print(f"\nSaved to: {result['report_path']}\n")
    elif args[0] == "compare" and len(args) > 2:
        # D1 — head-to-head comparison of two drugs.
        from core.graph import run as run_flow
        from core.retrieve import retrieve_for_report
        from report.compare import generate_comparison, save_comparison
        reset = "--reset" in args
        depth = _parse_depth(args)
        names = [a for a in args[1:] if a not in FLAGS]
        drug1, drug2 = names[0], names[1]
        log.info(f"=== COMPARE: {drug1} vs {drug2} ===")
        # Ingest both into the shared store (drug-tagged, so they coexist).
        run_flow(drug1, reset=reset, depth=depth)
        run_flow(drug2, reset=False, depth=depth)
        # Retrieve each drug's evidence separately, then compare.
        s1 = retrieve_for_report(drug1, depth=depth)
        s2 = retrieve_for_report(drug2, depth=depth)
        md = generate_comparison(drug1, s1, drug2, s2)
        path = save_comparison(drug1, drug2, md)
        print("\n" + "=" * 70)
        print(md)
        print("=" * 70)
        print(f"\nSaved to: {path}\n")
    else:
        depth = _parse_depth(args)
        drug = next(a for a in args if a not in FLAGS)
        run_pipeline(drug, depth=depth)
