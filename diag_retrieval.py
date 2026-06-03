"""
diag_retrieval.py — dump exactly what the retriever gives the LLM for
pregnancy + pk_pd. This is the chunks the synthesis prompt sees.

Usage:
    python diag_retrieval.py atorvastatin
"""

import sys
from core.combine import get_all_evidence
from core.chunk import chunk_all
from core.embed import embed_texts
from storage.vectorstore import index_chunks, reset as reset_collection
from core.retrieve import retrieve_for_report


def main(drug: str):
    print(f"\n{'='*70}")
    print(f"  RETRIEVAL DIAGNOSTIC: {drug!r}")
    print(f"{'='*70}\n")

    print("[step 1] fetching all sources...")
    evidence = get_all_evidence(drug)
    print(f"   got {len(evidence)} unique evidence rows")

    print("\n[step 2] chunking...")
    chunks = chunk_all(evidence, drug=drug)
    print(f"   got {len(chunks)} chunks")

    tag_counts: dict[str, int] = {}
    for c in chunks:
        sec = (c.section or "(untagged)").lower()
        tag_counts[sec] = tag_counts.get(sec, 0) + 1
    print(f"   chunks by section tag (top 20):")
    for sec, n in sorted(tag_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"     {n:4d}  {sec!r}")

    print("\n[step 3] embedding + indexing...")
    reset_collection()
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    index_chunks(chunks, vectors)
    print(f"   indexed {len(chunks)} chunks")

    print("\n[step 4] running retrieve_for_report (depth=detailed)...")
    sections = retrieve_for_report(drug, depth="detailed")

    for target in ["pregnancy", "pk_pd"]:
        retrieved = sections.get(target, [])
        print(f"\n{'='*70}")
        print(f"  RETRIEVED CHUNKS for {target!r} ({len(retrieved)} chunks)")
        print(f"{'='*70}")
        for i, c in enumerate(retrieved, 1):
            print(f"\n--- chunk {i}/{len(retrieved)} ---")
            print(f"source: {c.get('source')}:{c.get('source_id')}")
            print(f"tier: {c.get('tier')}")
            print(f"section tag: {c.get('section', '')!r}")
            print(f"title: {(c.get('title') or '')[:80]}")
            text = c.get('text', '')
            print(f"text length: {len(text)} chars")
            print(f"text body:")
            print(text[:1200])
            if len(text) > 1200:
                print(f"... [truncated, total {len(text)} chars]")


if __name__ == "__main__":
    drug = sys.argv[1] if len(sys.argv) > 1 else "atorvastatin"
    main(drug)