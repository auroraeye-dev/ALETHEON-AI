"""
diag_evidence.py — what is each source actually returning for this drug?

For a given drug, fetches from every source individually, then:
  1. Prints a summary per source: paper count, total text length, sources of text
  2. Grep-counts mentions of key terms (pregnancy, lactation, PK, etc.) per source
  3. Dumps the first 3 chunks per term per source for spot-inspection
  4. Reports cross-source coverage matrix so you can see where each topic lives

This tells you whether:
  (a) The data exists in your retrieved evidence but the chunker/retriever isn't surfacing it
  (b) The data isn't in your retrieved evidence — you need a different source

Usage:
    python diag_evidence.py atorvastatin > /tmp/diag_atorva.txt
    less /tmp/diag_atorva.txt
"""

import sys
import re
from collections import defaultdict

# Topics we want to know about. Each topic = list of regex patterns that
# would indicate the source contains real content on it. Word-boundary
# matching avoids "preGRANULated" matching "preg".
TOPICS = {
    "pregnancy": [r"\bpregnan", r"\bgestation", r"\bfetal\b", r"\bteratogen",
                  r"\bcategory\s+[abcdx]\b", r"\bthird\s+trimester\b"],
    "lactation": [r"\blactation\b", r"\bbreast\s*milk\b", r"\bnursing\s+mother",
                  r"\bbreastfeed", r"\bbreast-feed"],
    "reproductive": [r"\breproductive\b", r"\bfertility\b", r"\bcontraception\b",
                     r"\bcontraceptive\b"],
    "pk": [r"\bpharmacokinetic", r"\bbioavailability\b", r"\bhalf-life\b",
           r"\bhalf\s+life\b", r"\bC\s*max\b", r"\bT\s*max\b",
           r"\bAUC\b", r"\bclearance\b"],
    "metabolism": [r"\bCYP[0-9][A-Z]?[0-9]?\b", r"\bmetabolism\b",
                   r"\bcytochrome\b", r"\bP450\b", r"\bP-?glycoprotein\b"],
    "elimination": [r"\belimination\b", r"\bexcretion\b", r"\brenal\s+clearance\b",
                    r"\burinary\s+excretion\b"],
}

def count_topic_hits(text: str, patterns: list[str]) -> int:
    """How many places in this text mention any of these patterns."""
    n = 0
    for p in patterns:
        n += len(re.findall(p, text, flags=re.IGNORECASE))
    return n


def get_snippet(text: str, patterns: list[str], context_chars: int = 150) -> str | None:
    """Return ~150 chars around the first pattern match, or None."""
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            start = max(0, m.start() - context_chars)
            end = min(len(text), m.end() + context_chars)
            snippet = text[start:end].replace("\n", " ")
            return f"...{snippet}..."
    return None


def main(drug: str):
    print(f"\n{'='*70}")
    print(f"  EVIDENCE DIAGNOSTIC: {drug!r}")
    print(f"{'='*70}\n")

    # Import per-source fetchers directly so we can isolate each
    from sources import (fda, clinicaltrials, europepmc, faers,
                         dailymed, pubchem, chembl, pharmgkb, semanticscholar)

    SOURCES = [
        ("fda", fda.fetch),
        ("dailymed", dailymed.fetch),
        ("clinicaltrials", clinicaltrials.fetch),
        ("europepmc", europepmc.fetch),
        ("semanticscholar", semanticscholar.fetch),
        ("pubchem", pubchem.fetch),
        ("chembl", chembl.fetch),
        ("pharmgkb", pharmgkb.fetch),
        ("faers", faers.fetch),
    ]

    # Phase 1: fetch from each source independently and tally
    per_source: dict[str, list] = {}
    for name, fetch_fn in SOURCES:
        try:
            print(f"[{name}] fetching...")
            evidence = fetch_fn(drug)
            per_source[name] = evidence or []
            print(f"[{name}] got {len(per_source[name])} evidence rows\n")
        except Exception as e:
            print(f"[{name}] FAILED: {e}\n")
            per_source[name] = []

    # Phase 2: per-source topic coverage matrix
    print(f"\n{'='*70}")
    print(f"  TOPIC COVERAGE MATRIX (mentions per source per topic)")
    print(f"{'='*70}\n")
    header = f"{'source':<20}" + "".join(f"{t:<14}" for t in TOPICS) + "total_chars"
    print(header)
    print("-" * len(header))

    coverage = defaultdict(dict)  # source -> topic -> count
    for source_name, evidence in per_source.items():
        total_text = ""
        for ev in evidence:
            total_text += "\n" + (ev.text or "")
        row = f"{source_name:<20}"
        for topic, patterns in TOPICS.items():
            n = count_topic_hits(total_text, patterns)
            coverage[source_name][topic] = n
            row += f"{n:<14}"
        row += f"{len(total_text):,}"
        print(row)

    # Phase 3: per-topic, find the strongest source for each
    print(f"\n{'='*70}")
    print(f"  WHERE EACH TOPIC LIVES (strongest source per topic)")
    print(f"{'='*70}\n")
    for topic in TOPICS:
        by_source = [(s, coverage[s].get(topic, 0)) for s in per_source]
        by_source.sort(key=lambda x: -x[1])
        top = [(s, n) for s, n in by_source if n > 0]
        if not top:
            print(f"  {topic:<14} → NOT FOUND in any source")
        else:
            print(f"  {topic:<14} → " + ", ".join(f"{s}({n})" for s, n in top[:4]))

    # Phase 4: for each topic, dump 2 representative snippets from the best source
    print(f"\n{'='*70}")
    print(f"  SAMPLE CONTENT (first match per topic from strongest source)")
    print(f"{'='*70}\n")
    for topic, patterns in TOPICS.items():
        # Find best source
        best = max(per_source.items(),
                   key=lambda kv: count_topic_hits("\n".join(e.text or "" for e in kv[1]), patterns))
        best_name, best_evidence = best
        if count_topic_hits("\n".join(e.text or "" for e in best_evidence), patterns) == 0:
            print(f"\n--- {topic.upper()} ---")
            print(f"  No content found in any source.\n")
            continue

        # Get 2 snippets from the best source
        print(f"\n--- {topic.upper()} (from {best_name}) ---")
        n_shown = 0
        for ev in best_evidence:
            if n_shown >= 2:
                break
            snip = get_snippet(ev.text or "", patterns)
            if snip:
                title = (ev.title or "")[:60]
                section = (ev.extra or {}).get("section", "") if hasattr(ev, "extra") else ""
                print(f"  Source: {best_name}:{ev.source_id}")
                print(f"  Title: {title}")
                print(f"  Section tag: {section!r}")
                print(f"  Snippet: {snip}")
                print()
                n_shown += 1


if __name__ == "__main__":
    drug = sys.argv[1] if len(sys.argv) > 1 else "atorvastatin"
    main(drug)