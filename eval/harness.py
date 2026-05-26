"""
eval/harness.py
===============
E1 + E2 — evaluation harness.

Runs a batch of drugs through the full pipeline and produces OBJECTIVE,
repeatable metrics so you can measure quality before/after any change — instead
of eyeballing one report.

Per drug it captures:
  - evidence count + tier distribution (regulatory/peer_reviewed/real_world/preprint)
  - precision-filter drops (off-drug evidence removed)
  - # Key Findings and their confidence breakdown (High/Medium/Low)
  - empty-section flags (did Safety/Findings/etc. come up empty?)
  - whether a contradiction surfaced
  - cost (USD) and speed (seconds) per report   <- E2

Writes a CSV + a printed summary table.

Usage:
  python -m eval.harness                  # default drug set, medium depth
  python -m eval.harness --detailed       # detailed depth
  python -m eval.harness aspirin ibuprofen metformin   # custom drugs
"""

import sys
import csv
import re
import os
from datetime import datetime

from core.logging_setup import log
from core import metrics


DEFAULT_DRUGS = [
    "aspirin", "ibuprofen", "metformin", "atorvastatin", "lisinopril",
    "omeprazole", "amoxicillin", "sertraline", "warfarin", "prednisone",
]


def _parse_report(md: str) -> dict:
    """Extract structural metrics from a generated report's markdown."""
    # Key findings: bullet lines in the Key Findings section
    findings_block = ""
    m = re.search(r"## Key Findings(.*?)(?:\n## )", md, flags=re.S)
    if m:
        findings_block = m.group(1)
    n_findings = len(re.findall(r"^\s*[-*]\s", findings_block, flags=re.M))

    # Confidence breakdown across the whole report
    confs = re.findall(r"Confidence:\s*(High|Medium|Low)", md, flags=re.I)
    conf_counts = {"High": 0, "Medium": 0, "Low": 0}
    for c in confs:
        conf_counts[c.capitalize()] = conf_counts.get(c.capitalize(), 0) + 1

    # Empty-section detection
    empties = []
    for sec in ["Key Findings", "Safety / Warnings", "Preprint"]:
        m = re.search(rf"## {re.escape(sec)}[^\n]*\n(.*?)(?:\n## |\Z)", md, flags=re.S)
        if m:
            body = m.group(1).strip().lower()
            if ("insufficient evidence" in body or "no preprint evidence" in body
                    or len(body) < 30):
                empties.append(sec)

    # Contradiction surfaced?
    contra_surfaced = False
    m = re.search(r"## Contradictions.*?\n(.*?)(?:\n## |\Z)", md, flags=re.S)
    if m:
        contra_surfaced = "no direct contradictions" not in m.group(1).lower()

    # Source/tier counts from the Sources list
    tiers = re.findall(r"\*\*\[E\d+\]\*\*\s*\((\w+)\)", md)
    tier_counts = {}
    for t in tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1

    return {
        "n_sources": len(tiers),
        "tier_counts": tier_counts,
        "n_findings": n_findings,
        "conf_high": conf_counts["High"],
        "conf_medium": conf_counts["Medium"],
        "conf_low": conf_counts["Low"],
        "empty_sections": ";".join(empties) if empties else "",
        "contradiction_surfaced": contra_surfaced,
    }


def run_eval(drugs: list[str], depth: str = "medium") -> list[dict]:
    from core.graph import run as run_flow

    rows = []
    for i, drug in enumerate(drugs, 1):
        log.info(f"=== EVAL {i}/{len(drugs)}: {drug} (depth={depth}) ===")
        metrics.reset()
        try:
            # reset store only on the first drug; the rest coexist (drug-tagged)
            result = run_flow(drug, reset=(i == 1), depth=depth)
            metrics.stop_timer()
            cost = metrics.summary()
            parsed = _parse_report(result.get("report", ""))
            row = {
                "drug": drug,
                "n_sources": parsed["n_sources"],
                "regulatory": parsed["tier_counts"].get("regulatory", 0),
                "peer_reviewed": parsed["tier_counts"].get("peer_reviewed", 0),
                "real_world": parsed["tier_counts"].get("real_world", 0),
                "preprint": parsed["tier_counts"].get("preprint", 0),
                "n_findings": parsed["n_findings"],
                "conf_high": parsed["conf_high"],
                "conf_medium": parsed["conf_medium"],
                "conf_low": parsed["conf_low"],
                "empty_sections": parsed["empty_sections"],
                "contradiction": "yes" if parsed["contradiction_surfaced"] else "no",
                "cost_usd": cost["total_cost_usd"],
                "elapsed_sec": cost["elapsed_sec"],
            }
        except Exception as e:
            log.warning(f"[eval] {drug} failed: {e}")
            row = {"drug": drug, "error": str(e)[:80]}
        rows.append(row)
    return rows


def _print_table(rows: list[dict]):
    cols = ["drug", "n_sources", "regulatory", "peer_reviewed", "real_world",
            "preprint", "n_findings", "conf_high", "conf_medium", "conf_low",
            "contradiction", "empty_sections", "cost_usd", "elapsed_sec"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['drug'].ljust(widths['drug'])}  ERROR: {r['error']}")
            continue
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

    # Aggregates
    ok = [r for r in rows if "error" not in r]
    if ok:
        n = len(ok)
        avg_cost = sum(r["cost_usd"] for r in ok) / n
        avg_time = sum(r["elapsed_sec"] for r in ok) / n
        avg_src = sum(r["n_sources"] for r in ok) / n
        contra = sum(1 for r in ok if r["contradiction"] == "yes")
        any_empty = sum(1 for r in ok if r["empty_sections"])
        print("-" * len(header))
        print(f"\nAGGREGATE over {n} drugs:")
        print(f"  avg sources/report : {avg_src:.1f}")
        print(f"  avg cost/report    : ${avg_cost:.4f}")
        print(f"  avg time/report    : {avg_time:.1f}s")
        print(f"  contradictions surfaced : {contra}/{n}")
        print(f"  reports w/ empty section: {any_empty}/{n}")


def main():
    args = sys.argv[1:]
    depth = "medium"
    if "--detailed" in args:
        depth = "detailed"; args.remove("--detailed")
    if "--short" in args:
        depth = "short"; args.remove("--short")
    drugs = args if args else DEFAULT_DRUGS

    rows = run_eval(drugs, depth=depth)
    _print_table(rows)

    # Write CSV
    os.makedirs("data/eval", exist_ok=True)
    path = f"data/eval/eval_{datetime.now():%Y%m%d_%H%M%S}.csv"
    ok = [r for r in rows if "error" not in r]
    if ok:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ok[0].keys()))
            w.writeheader()
            w.writerows(ok)
        print(f"\nSaved metrics -> {path}")


if __name__ == "__main__":
    main()
