"""
core/graph.py
=============
LangGraph orchestration for Aletheon.

The pipeline as an explicit state graph:

    [fetch_fda]  [fetch_clinicaltrials]  [fetch_europepmc]   <- run in PARALLEL (fan-out)
         \\              |                      /
          \\             |                     /
                     [combine]                                <- fan-in: dedup
                         |
                     [index]                                  <- chunk + embed + store
                         |
                     [retrieve]                               <- section-targeted
                         |
                     [report]                                 <- cited report
                         |
                       (END)

THE KEY LESSON (the bug that burned us before):
    `evidence` uses an `add` REDUCER (Annotated[list, operator.add]).
    That means when the parallel fetch agents each return their slice of
    evidence, LangGraph MERGES them (concatenates) instead of the last one
    OVERWRITING the others. Without this reducer, parallel writes clobber each
    other and you lose data — exactly the "state gets wiped" bug.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from core.logging_setup import log
from core.models import Evidence


# ---- STATE -------------------------------------------------------------
# Each field's reducer decides how node outputs are merged into state.
class DrugState(TypedDict):
    drug: str
    depth: str
    # ACCUMULATE: parallel fetch agents each append their evidence here.
    # The operator.add reducer concatenates lists instead of overwriting.
    evidence: Annotated[list, operator.add]
    # The deduped result of combine. Plain field (replace semantics) — combine
    # writes here instead of back into `evidence`, so the add-reducer doesn't
    # double-append the combined list.
    combined: list
    # single-value fields (default behavior = replace, which is correct here)
    sections: dict
    report: str
    report_path: str
    # feedback loop (Build 2): self-evaluation verdict + a guard against looping
    verdict: dict
    corrected: bool
    # critic agent (D2): opt-in additive appraisal
    critic: bool
    # F2/F3: optional appendices to append {"trace","stats"}; None = clean report
    appendices: set


# ---- NODES (each wraps an existing agent; no agent logic changes) ------

def _fetch_node(name, fetch_fn):
    """Build a graph node from a source's fetch() function (cached)."""
    def node(state: DrugState) -> dict:
        import core.combine as _combine
        try:
            from core.cache import cached_fetch
            ev = cached_fetch(name, fetch_fn, state["drug"])
            outcome = ("ok", len(ev)) if ev else ("empty", 0)
        except Exception as e:
            log.warning(f"[graph:{name}] failed: {e}")
            ev = []
            outcome = ("error", 0)
        # Degraded-mode tracker (existing)
        _combine._LAST_SOURCE_OUTCOMES[name] = outcome
        # PRISMA per-source count: number of UNIQUE PAPERS / LABELS, not number
        # of internal sub-section Evidence rows. DailyMed returns ~80 rows for
        # one SPL label (one row per <section>); a med-affairs reviewer reading
        # the PRISMA diagram expects "labels identified", not "sections". Using
        # source_id as the key collapses sub-sections of the same label.
        unique_count = len({e.source_id for e in ev}) if ev else 0
        _combine._PRISMA_COUNTS["per_source"][name] = unique_count
        # Return ONLY the evidence key — the add-reducer merges it with the
        # evidence other parallel nodes produce. We never overwrite.
        return {"evidence": ev}
    return node


def _combine_node(state: DrugState) -> dict:
    """Fan-in: precision-filter + dedup the merged evidence from all fetch nodes."""
    from core.relevance import filter_relevant
    import core.combine as _combine

    # A1 PRECISION GATE: drop off-drug (sibling) evidence first.
    raw = state["evidence"]
    filtered, dropped = filter_relevant(raw, state["drug"])
    if dropped:
        log.info(f"[graph:combine] precision filter dropped {dropped} off-drug")

    seen, deduped = set(), []
    for ev in filtered:
        key = (ev.source, ev.source_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
    by_tier = {}
    for ev in deduped:
        by_tier[ev.tier] = by_tier.get(ev.tier, 0) + 1
    log.info(f"[graph:combine] {len(deduped)} unique evidence "
             f"({', '.join(f'{k}: {v}' for k, v in sorted(by_tier.items()))})")

    # PRISMA stage 2 counts — work in UNIQUE PAPER space (source, source_id),
    # not Evidence-row space. The per-source counts upstream are already in
    # paper space; mixing the two would make the diagram fail to balance.
    raw_paper_ids = {(e.source, e.source_id) for e in raw}
    filtered_paper_ids = {(e.source, e.source_id) for e in filtered}
    off_drug_papers = raw_paper_ids - filtered_paper_ids
    # 'Duplicates removed' = unique-paper count in `filtered` minus unique-paper
    # count in `deduped`. Since dedup keys on (source, source_id), the
    # difference is zero in practice — but the slot stays in PRISMA for
    # transparency.
    duplicates_at_dedup = len(filtered_paper_ids) - len(deduped)
    _combine._PRISMA_COUNTS["off_drug_removed"] = len(off_drug_papers)
    _combine._PRISMA_COUNTS["duplicates_removed"] = max(0, duplicates_at_dedup)
    _combine._PRISMA_COUNTS["records_screened"] = len(deduped)

    # Write to `combined` (replace semantics) — NOT back into `evidence`, which
    # has an add-reducer that would re-append and double the list.
    return {"combined": deduped}


def _index_node(state: DrugState) -> dict:
    """Chunk + embed + store the combined evidence."""
    from core.chunk import chunk_all
    from core.embed import embed_texts
    from storage import vectorstore
    from core.errors import NoEvidenceFound

    drug_key = state["drug"].lower().strip()
    combined = state.get("combined", [])
    # E4: if every source came back empty, don't proceed to build an empty
    # report — fail clearly so the user knows (likely a misspelled/unknown drug).
    if not combined:
        raise NoEvidenceFound(
            f"No evidence found for {state['drug']!r} from any source. "
            f"Check the spelling, or try a generic name (e.g. 'acetaminophen' "
            f"instead of a brand).")
    chunks = chunk_all(combined, drug=drug_key)
    log.info(f"[graph:index] {len(combined)} evidence -> {len(chunks)} chunks")
    if not chunks:
        raise NoEvidenceFound(
            f"Evidence for {state['drug']!r} could not be processed into "
            f"searchable chunks.")
    vectors = embed_texts([c.text for c in chunks])
    vectorstore.index_chunks(chunks, vectors)
    return {}


def _retrieve_node(state: DrugState) -> dict:
    from core.retrieve import retrieve_for_report
    import core.combine as _combine
    sections = retrieve_for_report(state["drug"], depth=state.get("depth", "medium"))
    # PRISMA inclusion-stage counts: unique studies cited, total chunks cited,
    # sections that have at least one chunk. We work from `sections` because
    # that's exactly what gets handed to the synthesis prompt.
    seen = set()
    chunk_total = 0
    sections_with_content = 0
    for chunks in (sections or {}).values():
        chunk_total += len(chunks or [])
        if chunks:
            sections_with_content += 1
        for c in (chunks or []):
            seen.add((c.get("source"), c.get("source_id")))
    _combine._PRISMA_COUNTS["studies_included"] = len(seen)
    _combine._PRISMA_COUNTS["reports_included"] = len(seen)
    _combine._PRISMA_COUNTS["chunks_in_report"] = chunk_total
    _combine._PRISMA_COUNTS["sections_with_evidence"] = sections_with_content
    return {"sections": sections}


def _report_node(state: DrugState) -> dict:
    from report.generate import generate_report, save_report
    import core.combine as _combine
    report_md = generate_report(state["drug"], state["sections"],
                                depth=state.get("depth", "medium"),
                                appendices=state.get("appendices"))
    # Splice the PRISMA flow diagram into the report after the header but
    # before Bottom Line / Key Findings. The data was recorded by upstream
    # nodes into _PRISMA_COUNTS; here we just render and inject.
    try:
        from report.prisma import PrismaCounts, build_prisma_block
        counts = PrismaCounts(
            per_source=dict(_combine._PRISMA_COUNTS["per_source"]),
            duplicates_removed=_combine._PRISMA_COUNTS["duplicates_removed"],
            off_drug_removed=_combine._PRISMA_COUNTS["off_drug_removed"],
            records_screened=_combine._PRISMA_COUNTS["records_screened"],
            reports_assessed=_combine._PRISMA_COUNTS["reports_assessed"],
            reports_excluded=dict(_combine._PRISMA_COUNTS["reports_excluded"]),
            reports_sought=_combine._PRISMA_COUNTS["records_screened"],
            reports_retrieved=_combine._PRISMA_COUNTS["records_screened"],
            studies_included=_combine._PRISMA_COUNTS["studies_included"],
            reports_included=_combine._PRISMA_COUNTS["reports_included"],
            chunks_in_report=_combine._PRISMA_COUNTS["chunks_in_report"],
            sections_with_evidence=_combine._PRISMA_COUNTS["sections_with_evidence"],
        )
        prisma_block = build_prisma_block(counts)
        # Insert just before the first major synthesis section.
        # Try several insertion points in priority order; if none match, append
        # at the very end of the report so the diagram is at least present.
        insertion_markers = ["## Bottom Line", "## Summary", "## Key Findings"]
        inserted = False
        for marker in insertion_markers:
            if marker in report_md:
                report_md = report_md.replace(marker, prisma_block + "\n" + marker, 1)
                inserted = True
                break
        if not inserted:
            report_md = report_md + "\n\n" + prisma_block
    except Exception as e:
        log.warning(f"[graph:report] PRISMA injection failed: {e}")

    path = save_report(state["drug"], report_md)
    return {"report": report_md, "report_path": path}


def _evaluate_node(state: DrugState) -> dict:
    """Build 2 feedback loop: self-evaluate the report, decide if a corrective
    pass is warranted (cheap, structural — no extra LLM call)."""
    from report.evaluate import evaluate_report
    verdict = evaluate_report(state.get("report", ""))
    return {"verdict": verdict}


def _correct_node(state: DrugState) -> dict:
    """Corrective pass: re-retrieve with boosts for weak sections, regenerate
    the report ONCE. Runs only when evaluate flagged the report as weak."""
    from core.retrieve import retrieve_for_report
    from report.generate import generate_report, save_report

    verdict = state.get("verdict", {})
    boost = verdict.get("boost", {})
    log.info(f"[graph:correct] corrective pass — boosting {list(boost)} ...")
    sections = retrieve_for_report(state["drug"], depth=state.get("depth", "medium"),
                                   boost=boost)
    report_md = generate_report(state["drug"], sections, depth=state.get("depth", "medium"))
    path = save_report(state["drug"], report_md)
    return {"sections": sections, "report": report_md, "report_path": path,
            "corrected": True}


def _needs_correction(state: DrugState) -> str:
    """Conditional edge: route to corrective pass if weak AND not already corrected."""
    verdict = state.get("verdict", {})
    if verdict.get("needs_retry") and not state.get("corrected"):
        return "correct"
    return "done"


def _critic_node(state: DrugState) -> dict:
    """D2: opt-in. Append a Critical Appraisal to the FINAL report (additive)."""
    if not state.get("critic"):
        return {}
    from report.critic import append_appraisal
    from report.generate import save_report
    appraised = append_appraisal(state["drug"], state.get("report", ""))
    path = save_report(state["drug"], appraised)
    return {"report": appraised, "report_path": path}


# ---- GRAPH ASSEMBLY ----------------------------------------------------

def build_graph(reset: bool = False):
    """Construct the LangGraph pipeline. Returns a compiled graph."""
    from core.combine import SOURCES  # the (name, fetch_fn) list you already maintain

    if reset:
        from storage import vectorstore
        vectorstore.reset()
        log.info("[graph] store reset — starting clean")

    g = StateGraph(DrugState)

    # Add one parallel node per source.
    fetch_names = []
    for name, fetch_fn in SOURCES:
        node_name = f"fetch_{name}"
        g.add_node(node_name, _fetch_node(name, fetch_fn))
        fetch_names.append(node_name)

    g.add_node("combine", _combine_node)
    g.add_node("index", _index_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("report", _report_node)
    g.add_node("evaluate", _evaluate_node)
    g.add_node("correct", _correct_node)
    g.add_node("critic", _critic_node)

    # Fan-out: START -> all fetch nodes in parallel.
    for fn in fetch_names:
        g.set_entry_point(fn) if False else None  # entry handled below
    # LangGraph: set multiple entry edges so fetches run concurrently.
    for fn in fetch_names:
        g.add_edge("__start__", fn)
    # Fan-in: every fetch node -> combine. combine waits for ALL of them.
    for fn in fetch_names:
        g.add_edge(fn, "combine")

    # Linear tail: combine -> index -> retrieve -> report -> evaluate.
    g.add_edge("combine", "index")
    g.add_edge("index", "retrieve")
    g.add_edge("retrieve", "report")
    g.add_edge("report", "evaluate")

    # FEEDBACK LOOP (Build 2): evaluate -> (correct or straight to critic).
    # Either way we pass through the critic node (which no-ops unless enabled),
    # so the appraisal always sees the FINAL report. corrected guard prevents
    # looping more than once.
    g.add_conditional_edges("evaluate", _needs_correction,
                            {"correct": "correct", "done": "critic"})
    g.add_edge("correct", "critic")
    g.add_edge("critic", END)

    return g.compile()


def run(drug: str, reset: bool = False, depth: str = "medium", critic: bool = False,
        appendices: set | None = None) -> dict:
    """Run the full orchestrated pipeline for `drug` at the given depth.

    Raises AletheonError subclasses (InvalidDrugName, NoEvidenceFound,
    PipelineError) on expected failure modes so the CLI can show a clean
    message instead of a stack trace.
    """
    from core.errors import validate_drug_name, AletheonError, PipelineError
    drug = validate_drug_name(drug)  # may raise InvalidDrugName

    # Reset module-level PRISMA + degraded-mode state — avoids leakage across
    # multiple pipeline runs in a harness or test session.
    from core.combine import _reset_prisma_state
    _reset_prisma_state()

    log.info(f"=== LANGGRAPH PIPELINE: {drug!r} (depth={depth}"
             + (", +critic" if critic else "") + ") ===")
    try:
        graph = build_graph(reset=reset)
        final = graph.invoke({"drug": drug, "depth": depth, "evidence": [], "combined": [],
                              "sections": {}, "report": "", "report_path": "",
                              "verdict": {}, "corrected": False, "critic": critic,
                              "appendices": appendices})
    except AletheonError:
        raise  # already a clean, typed error — let the CLI format it
    except Exception as e:
        # Wrap any unexpected failure so the CLI shows a clean message.
        raise PipelineError(f"Pipeline failed while processing {drug!r}: {e}") from e
    return final