"""
diag_pregnancy.py — dump the pregnancy-tagged chunks for a drug.

Usage:
    python diag_pregnancy.py atorvastatin

Tells us *exactly* what content the LLM is seeing in the PREGNANCY evidence
block, so we can stop guessing why the section comes back empty.
"""

import sys
from core.combine import get_all_evidence
from core.chunk import chunk_evidence
from core.embed import embed_chunks
from core.retrieve import retrieve_for_report
from storage.vectorstore import index, search
from core.config import config


def main(drug: str):
    print(f"\n=== fetching evidence for {drug!r} ===")
    evidence = get_all_evidence(drug)
    print(f"got {len(evidence)} evidence rows")

    print(f"\n=== chunking ===")
    chunks = chunk_evidence(evidence)
    print(f"produced {len(chunks)} chunks")

    # Count how many chunks have section="pregnancy" or related
    preg_labels = ["pregnancy", "lactation", "nursing mothers",
                   "females and males of reproductive potential"]
    preg_chunks = [c for c in chunks if (c.section or "").lower() in preg_labels]
    print(f"chunks tagged with pregnancy-related sections: {len(preg_chunks)}")
    section_counts = {}
    for c in preg_chunks:
        sec = (c.section or "").lower()
        section_counts[sec] = section_counts.get(sec, 0) + 1
    for sec, n in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  section={sec!r}: {n} chunks")

    # Now dump the actual text of each pregnancy-tagged chunk so we can see
    # what the LLM is being asked to use.
    print(f"\n=== DUMP: all pregnancy-tagged chunks (text only) ===")
    for i, c in enumerate(preg_chunks[:20]):  # cap at 20 to keep output sane
        print(f"\n--- chunk {i+1}/{len(preg_chunks)} ---")
        print(f"source: {c.source}:{c.source_id}")
        print(f"section: {c.section!r}")
        print(f"title: {c.title[:80]}")
        print(f"text ({len(c.text)} chars):")
        print(c.text[:1500])
        if len(c.text) > 1500:
            print(f"... [truncated, total {len(c.text)} chars]")

    # Also show what semantic retrieval pulls when we query for pregnancy
    print(f"\n=== embedding + qdrant index ===")
    embed_chunks(chunks)
    index(chunks)

    print(f"\n=== retrieval for 'pregnancy' section ===")
    sections = retrieve_for_report(drug, depth="detailed")
    preg_retrieved = sections.get("pregnancy", [])
    print(f"retrieved {len(preg_retrieved)} chunks for the pregnancy section\n")
    for i, c in enumerate(preg_retrieved):
        print(f"--- retrieved chunk {i+1} ---")
        print(f"source: {c['source']}:{c['source_id']}")
        print(f"section tag: {c.get('section', '')!r}")
        print(f"text ({len(c['text'])} chars): {c['text'][:500]}")
        if len(c['text']) > 500:
            print(f"... [truncated]")
        print()


if __name__ == "__main__":
    drug = sys.argv[1] if len(sys.argv) > 1 else "atorvastatin"
    main(drug)
