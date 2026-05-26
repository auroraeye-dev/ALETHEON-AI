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


# ---- NODES (each wraps an existing agent; no agent logic changes) ------

def _fetch_node(name, fetch_fn):
    """Build a graph node from a source's fetch() function."""
    def node(state: DrugState) -> dict:
        try:
            ev = fetch_fn(state["drug"])
        except Exception as e:
            log.warning(f"[graph:{name}] failed: {e}")
            ev = []
        # Return ONLY the evidence key — the add-reducer merges it with the
        # evidence other parallel nodes produce. We never overwrite.
        return {"evidence": ev}
    return node


def _combine_node(state: DrugState) -> dict:
    """Fan-in: precision-filter + dedup the merged evidence from all fetch nodes."""
    from core.relevance import filter_relevant

    # A1 PRECISION GATE: drop off-drug (sibling) evidence first.
    filtered, dropped = filter_relevant(state["evidence"], state["drug"])
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
    # Write to `combined` (replace semantics) — NOT back into `evidence`, which
    # has an add-reducer that would re-append and double the list.
    return {"combined": deduped}


def _index_node(state: DrugState) -> dict:
    """Chunk + embed + store the combined evidence."""
    from core.chunk import chunk_all
    from core.embed import embed_texts
    from storage import vectorstore

    drug_key = state["drug"].lower().strip()
    chunks = chunk_all(state["combined"], drug=drug_key)
    log.info(f"[graph:index] {len(state['combined'])} evidence -> {len(chunks)} chunks")
    vectors = embed_texts([c.text for c in chunks])
    vectorstore.index_chunks(chunks, vectors)
    return {}


def _retrieve_node(state: DrugState) -> dict:
    from core.retrieve import retrieve_for_report
    sections = retrieve_for_report(state["drug"], depth=state.get("depth", "medium"))
    return {"sections": sections}


def _report_node(state: DrugState) -> dict:
    from report.generate import generate_report, save_report
    report_md = generate_report(state["drug"], state["sections"],
                                depth=state.get("depth", "medium"))
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


def run(drug: str, reset: bool = False, depth: str = "medium", critic: bool = False) -> dict:
    """Run the full orchestrated pipeline for `drug` at the given depth."""
    log.info(f"=== LANGGRAPH PIPELINE: {drug!r} (depth={depth}"
             + (", +critic" if critic else "") + ") ===")
    graph = build_graph(reset=reset)
    final = graph.invoke({"drug": drug, "depth": depth, "evidence": [], "combined": [],
                          "sections": {}, "report": "", "report_path": "",
                          "verdict": {}, "corrected": False, "critic": critic})
    return final
