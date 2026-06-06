"""
diag_sections.py — dump exactly what the retriever pulls for the sections
that keep coming back empty: pregnancy, pk_pd, populations, blackbox, cv_risk.

Shows the FULL text of every chunk plus its source, length, and section tag,
so you can see whether:
  (a) chunks are coming through as PharmGKB junk (filter problem)
  (b) chunks are coming through as real DailyMed content but the LLM is
      still rendering "insufficient" (synthesis prompt problem)
  (c) section-tagged chunks exist in Qdrant but the retriever isn't getting
      them (vectorstore filter problem)

Usage:
    python diag_sections.py atorvastatin
"""
import sys
from core.combine import get_all_evidence
from core.chunk import chunk_all
from core.embed import embed_texts
from storage.vectorstore import index_chunks, reset as reset_collection
from core.retrieve import retrieve_for_report


def main(drug: str):
    print(f"\n{'='*80}")
    print(f"  SECTION DIAGNOSTIC: {drug!r}")
    print(f"{'='*80}\n")

    print("[1/4] fetching all sources...")
    evidence = get_all_evidence(drug)
    print(f"   got {len(evidence)} unique evidence rows")

    print("\n[2/4] chunking...")
    chunks = chunk_all(evidence, drug=drug)
    print(f"   got {len(chunks)} chunks")

    # Show how chunks are distributed across the sections we care about
    interesting_tags = {
        "pregnancy", "lactation", "nursing mothers",
        "use in specific populations", "reproductive",
        "females and males of reproductive potential",
        "pharmacokinetics", "pharmacodynamics", "clinical pharmacology",
        "mechanism of action", "absorption", "metabolism", "elimination",
        "boxed warning", "warnings", "warnings and precautions",
    }
    print(f"\n   chunks tagged with interesting sections:")
    by_tag: dict[str, list] = {}
    for c in chunks:
        sec = (c.section or "").lower()
        if sec in interesting_tags:
            by_tag.setdefault(sec, []).append(c)
    if not by_tag:
        print("   (none — chunker isn't tagging any pregnancy/PK/populations chunks)")
    for sec in sorted(by_tag.keys()):
        chunks_in_tag = by_tag[sec]
        sources_in_tag = {}
        for c in chunks_in_tag:
            sources_in_tag[c.source] = sources_in_tag.get(c.source, 0) + 1
        src_str = ", ".join(f"{s}:{n}" for s, n in sorted(sources_in_tag.items()))
        print(f"     {sec!r}: {len(chunks_in_tag)} chunks ({src_str})")

    print("\n[3/4] embedding + indexing...")
    reset_collection()
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    index_chunks(chunks, vectors)

    print("\n[4/4] running retrieve_for_report (depth=detailed)...")
    sections = retrieve_for_report(drug, depth="detailed")

    # Dump the chunks for the sections that have been failing
    for target in ["pregnancy", "pk_pd", "populations", "blackbox", "cv_risk"]:
        retrieved = sections.get(target, [])
        print(f"\n{'='*80}")
        print(f"  {target!r}: {len(retrieved)} chunks reached the synthesis prompt")
        print(f"{'='*80}")
        if not retrieved:
            print("  (no chunks — section will render as 'Insufficient evidence')")
            continue
        # Source breakdown
        src_counts: dict[str, int] = {}
        for c in retrieved:
            src_counts[c.get('source', '?')] = src_counts.get(c.get('source', '?'), 0) + 1
        print(f"  source breakdown: {dict(src_counts)}")
        # Full text of each chunk
        for i, c in enumerate(retrieved, 1):
            src = c.get("source", "?")
            sid = c.get("source_id", "?")
            sec = c.get("section", "")
            text = c.get("text", "")
            print(f"\n  --- chunk {i}/{len(retrieved)} ---")
            print(f"  source: {src}:{sid}")
            print(f"  section tag: {sec!r}")
            print(f"  length: {len(text)} chars")
            print(f"  text:")
            # show first 500 chars, then "..."
            print("    " + text[:500].replace("\n", "\n    "))
            if len(text) > 500:
                print(f"    ... [truncated, total {len(text)} chars]")


if __name__ == "__main__":
    drug = sys.argv[1] if len(sys.argv) > 1 else "atorvastatin"
    main(drug)