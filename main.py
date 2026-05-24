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


def ingest(drug: str):
    """Day 2: fetch a source -> Evidence -> chunks -> embeddings -> Qdrant.

    Using FDA as the test-pilot source (works on your network).
    PubMed is wired up too but NCBI is currently blocked on your network.
    """
    from sources import fda
    from core.chunk import chunk_all
    from core.embed import embed_texts
    from storage import vectorstore

    log.info(f"=== INGEST: {drug!r} ===")
    evidence = fda.fetch(drug)
    if not evidence:
        log.warning("No evidence fetched — stopping.")
        return

    chunks = chunk_all(evidence)
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


def run_pipeline(drug: str):
    log.info(f"Full pipeline for {drug!r} — not built yet (Day 3 closes the loop).")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        healthcheck()
    elif args[0] == "ingest" and len(args) > 1:
        ingest(args[1])
    elif args[0] == "search" and len(args) > 1:
        search(" ".join(args[1:]))
    else:
        run_pipeline(args[0])
