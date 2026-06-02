"""
report/screen.py
================
Step B — HEAD-TO-HEAD SCREENING GATE for comparison queries.

The architectural fix for the audit's #2 / #3 gap. Before the comparator
synthesizes, every candidate finding is passed through a focused screening LLM
call that asks: "Does THIS paper provide a direct head-to-head clinical
comparison between DRUG1 and DRUG2, as monotherapy, on a clinical outcome?"

Only YES findings are passed to the synthesis. NO findings are retained in the
evidence base (so [A#]/[B#] tags don't break) but they're excluded from the
synthesis prompt for the comparison sections, and the Evidence Tables clearly
mark them as "context only."

The screening LLM reads the STRUCTURED extracted row (study_type, population,
intervention_dose, key_outcomes, conclusion) — much cleaner signal than the raw
abstract. The extraction step has already disambiguated what the paper is about;
the screening step just decides whether that "about" is a head-to-head.

WHY THIS HONESTLY MATTERS:
The previous report's Key Trade-offs cited "63.9% vs 45.5%" — a real number
from a real paper — but the paper compared paracetamol+ibuprofen vs ibuprofen
monotherapy, NOT ibuprofen vs aspirin. The LLM had no way to know the paper
wasn't answering the question being asked. Screening fixes this at the source:
the paper never reaches the synthesis prompt because it fails screening on
"monotherapy" and "drug pair match."

Caching: by (paper_id, drug1, drug2). Same paper for a different drug pair gets
re-screened; same pair gets cached.
"""

import json
import time
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from core.config import config
from core.logging_setup import log


@dataclass
class ScreenDecision:
    source_id: str
    is_head_to_head: bool
    reason: str           # one-line LLM explanation
    confidence: str       # "high" / "medium" / "low"


# ---- Caching ----

def _screen_cache_path(source_id: str, drug1: str, drug2: str) -> str:
    pair = "_vs_".join(sorted([drug1.lower(), drug2.lower()]))
    pair_safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in pair)[:40]
    safe = "".join(ch if ch.isalnum() else "_" for ch in source_id)[:40]
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, f"screen_{pair_safe}_{safe}.json")


def _cache_get(source_id: str, drug1: str, drug2: str) -> ScreenDecision | None:
    path = _screen_cache_path(source_id, drug1, drug2)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        d = data["decision"]
        return ScreenDecision(
            source_id=d["source_id"],
            is_head_to_head=bool(d["is_head_to_head"]),
            reason=str(d["reason"]),
            confidence=str(d["confidence"]),
        )
    except Exception as e:
        log.warning(f"[screen] cache read failed for {source_id}: {e}")
        return None


def _cache_put(source_id: str, drug1: str, drug2: str, decision: ScreenDecision) -> None:
    path = _screen_cache_path(source_id, drug1, drug2)
    try:
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "decision": {
                "source_id": decision.source_id,
                "is_head_to_head": decision.is_head_to_head,
                "reason": decision.reason,
                "confidence": decision.confidence,
            }}, f)
    except Exception as e:
        log.warning(f"[screen] cache write failed for {source_id}: {e}")


# ---- The screening LLM call ----

_SCREEN_SYSTEM = (
    "You are a clinical evidence screener. Given the structured extraction of "
    "ONE paper and a target drug pair, you decide whether the paper provides a "
    "DIRECT HEAD-TO-HEAD CLINICAL COMPARISON between the two named drugs, "
    "as monotherapy (not in combination with other active drugs), on at least "
    "one clinical outcome (efficacy or safety) in human participants.\n\n"
    "Return ONLY a JSON object with keys: is_head_to_head (boolean), reason "
    "(one short sentence), confidence (\"high\" | \"medium\" | \"low\").\n\n"
    "Strict screening criteria — ALL of these must be true for YES:\n"
    "1. The paper compares the two NAMED drugs to each other directly.\n"
    "   - NO if it compares one of them to a third drug or placebo.\n"
    "   - NO if it studies a fixed-dose combination (e.g. paracetamol + "
    "ibuprofen) as one arm versus a single drug as the other arm.\n"
    "2. Both drugs are used as MONOTHERAPY in the comparison arms.\n"
    "   - NO if either drug is combined with another active drug in its arm.\n"
    "   - Adjuncts like saline, placebo, or vehicle are OK.\n"
    "3. The paper reports at least one CLINICAL OUTCOME for the comparison "
    "(efficacy endpoint, safety event, adverse-reaction rate, PK measurement "
    "tied to a clinical claim).\n"
    "   - NO if it's purely chemistry, in-vitro, or computational.\n"
    "   - NO if it's a trial registration with no reported results.\n"
    "4. Human participants (not animal-only).\n\n"
    "Confidence guidance: HIGH when the row clearly states both drugs in "
    "intervention_dose and a clinical outcome quantifies the comparison. "
    "MEDIUM when one criterion is implied but not explicit. "
    "LOW when the row is sparse and you're guessing.\n\n"
    "Be STRICT. False positives (letting in non-comparison papers) corrupt the "
    "synthesis. False negatives (dropping a real comparison) just mean we "
    "have fewer findings — that's recoverable. Prefer NO when uncertain."
)

_SCREEN_TEMPLATE = """Drug pair to compare: {drug1} vs {drug2}

Paper (structured extraction):
- source: {source}:{source_id}
- title: {title}
- study_type: {study_type}
- population: {population}
- intervention_dose: {intervention_dose}
- key_outcome_1: {key_outcome_1}
- key_outcome_2: {key_outcome_2}
- key_outcome_3: {key_outcome_3}
- safety_signal: {safety_signal}
- conclusion: {conclusion}

Is this paper a direct head-to-head clinical comparison of {drug1} vs {drug2},
as monotherapy, on a clinical outcome, in human participants? Apply ALL four
criteria strictly. Return JSON.
"""


def _llm_screen_one(finding, drug1: str, drug2: str, client) -> ScreenDecision:
    """One extracted finding -> one screening decision via LLM."""
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": _SCREEN_SYSTEM},
                {"role": "user", "content": _SCREEN_TEMPLATE.format(
                    drug1=drug1, drug2=drug2,
                    source=finding.source, source_id=finding.source_id,
                    title=finding.title or "",
                    study_type=finding.study_type or "",
                    population=finding.population or "",
                    intervention_dose=finding.intervention_dose or "",
                    key_outcome_1=finding.key_outcome_1 or "",
                    key_outcome_2=finding.key_outcome_2 or "",
                    key_outcome_3=finding.key_outcome_3 or "",
                    safety_signal=finding.safety_signal or "",
                    conclusion=finding.conclusion or "",
                )},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            from core.metrics import record_llm
            u = getattr(resp, "usage", None)
            if u is not None:
                record_llm(getattr(u, "prompt_tokens", 0),
                           getattr(u, "completion_tokens", 0), config.LLM_MODEL)
        except Exception:
            pass

        data = json.loads(resp.choices[0].message.content)
        return ScreenDecision(
            source_id=finding.source_id,
            is_head_to_head=bool(data.get("is_head_to_head", False)),
            reason=str(data.get("reason", ""))[:200],
            confidence=str(data.get("confidence", "low"))[:10],
        )
    except Exception as e:
        log.warning(f"[screen] LLM screen failed for {finding.source_id}: {e}")
        # On failure: default to NO. False negatives are recoverable; false
        # positives corrupt the synthesis.
        return ScreenDecision(
            source_id=finding.source_id,
            is_head_to_head=False,
            reason="(screening failed — defaulting to exclude)",
            confidence="low",
        )


def screen_findings_for_head_to_head(findings: list,
                                     drug1: str,
                                     drug2: str,
                                     max_parallel: int = 8) -> dict:
    """Screen each ExtractedFinding for head-to-head relevance to the drug pair.

    Returns a dict {source_id: ScreenDecision}. Failed-extraction findings are
    not screened (they're already excluded from synthesis anyway); they get a
    NO decision by default.

    Caches per (source_id, drug pair) so re-runs of the same pair are free."""
    if not findings:
        return {}

    # Skip failed-extraction findings — nothing to screen.
    candidates = [f for f in findings if f.extraction_quality != "failed"]
    if not candidates:
        log.info(f"[screen] no extractable findings to screen for "
                 f"{drug1!r} vs {drug2!r}")
        return {}

    log.info(f"[screen] screening {len(candidates)} paper(s) for "
             f"head-to-head {drug1!r} vs {drug2!r} (parallel={max_parallel}) …")

    decisions: dict = {}
    to_llm: list = []
    for f in candidates:
        cached = _cache_get(f.source_id, drug1, drug2)
        if cached is not None:
            decisions[f.source_id] = cached
        else:
            to_llm.append(f)

    n_cached = len(decisions)
    if to_llm:
        log.info(f"[screen] {n_cached} cached, {len(to_llm)} need LLM screening")
        from report.generate import _get_client
        client = _get_client()

        def _do_one(f):
            d = _llm_screen_one(f, drug1, drug2, client)
            _cache_put(f.source_id, drug1, drug2, d)
            return f.source_id, d

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [pool.submit(_do_one, f) for f in to_llm]
            for fut in as_completed(futures):
                try:
                    sid, d = fut.result()
                    decisions[sid] = d
                except Exception as e:
                    log.warning(f"[screen] worker failed: {e}")
    else:
        log.info(f"[screen] all {n_cached} screening decisions cached — no LLM calls")

    n_yes = sum(1 for d in decisions.values() if d.is_head_to_head)
    n_no = sum(1 for d in decisions.values() if not d.is_head_to_head)
    log.info(f"[screen] done: {n_yes} YES (real head-to-head), {n_no} NO (excluded from comparison)")
    # Audit transparency: log rejection reasons. If the screening seems too
    # strict in practice, this is how you'll spot it — read the reasons and see
    # if any reads as a false-negative ("rejected because confidence low" on a
    # paper that actually was a head-to-head).
    if n_no > 0:
        log.info(f"[screen] rejection reasons (audit trail):")
        for sid, d in decisions.items():
            if not d.is_head_to_head:
                log.info(f"[screen]   NO  {sid}: ({d.confidence}) {d.reason}")
        if n_yes > 0:
            log.info(f"[screen] kept papers:")
            for sid, d in decisions.items():
                if d.is_head_to_head:
                    log.info(f"[screen]   YES {sid}: ({d.confidence}) {d.reason}")
    return decisions


def partition_findings(findings: list,
                       decisions: dict) -> tuple[list, list]:
    """Split findings into (head_to_head, context_only) based on screening
    decisions. Failed-extraction findings always go to context_only.
    Findings without a decision (shouldn't happen but defensive) go to context_only."""
    h2h, context = [], []
    for f in findings:
        d = decisions.get(f.source_id)
        if d is not None and d.is_head_to_head and f.extraction_quality != "failed":
            h2h.append(f)
        else:
            context.append(f)
    return h2h, context
