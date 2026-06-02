"""
core/retrieve.py
================
DAY 7 (retrieval upgrade): section-targeted retrieval.

Instead of one vague query feeding the whole report, we retrieve SEPARATELY
for each section with a focused query — and we can filter by tier (e.g. pull
preprints specifically). This is what fixes the empty Safety/Preprint sections:
the report no longer hopes the right chunks float to the top of one generic
search; it asks for exactly what each section needs.
"""

from core.embed import embed_query
from storage import vectorstore
from core.config import config
from core.logging_setup import log


def _hits_to_dicts(hits) -> list[dict]:
    out = []
    for h in hits:
        p = h.payload
        out.append({
            "text": p.get("text", ""),
            "source": p.get("source", ""),
            "source_id": p.get("source_id", ""),
            "title": p.get("title", ""),
            "url": p.get("url", ""),
            "tier": p.get("tier", ""),
            "score": h.score,
        })
    return out


def retrieve(query: str, top_k: int = None, tier: str = None) -> list[dict]:
    """Single-query retrieval (still used by the `search` command)."""
    top_k = top_k or config.RETRIEVE_TOP_K
    qvec = embed_query(query)
    hits = vectorstore.search(qvec, top_k=top_k, tier=tier)
    results = _hits_to_dicts(hits)
    log.info(f"[retrieve] {len(results)} chunks for {query!r}"
             + (f" (tier={tier})" if tier else ""))
    return results


def _dedup(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        key = (c["source"], c["source_id"], c["text"][:60])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# B1 — depth profiles. Each controls how many chunks each section pulls, which
# is what actually drives report length: more retrieved evidence -> a longer
# report that's still fully grounded (not padded/hallucinated).
# B1 — depth profiles now live in config (E5) so they're tunable via .env
# without editing code. These module-level defaults are kept as a fallback and
# for reference, but retrieve_for_report reads config.depth_profiles() at call
# time so env overrides take effect.
DEPTH_PROFILES = {
    "short":    {"overview": 3, "safety": 4, "eff_pr": 2, "eff_reg": 2, "contra": 2, "preprint": 2,
                 "dosing": 2, "interactions": 2, "mechanism": 2, "populations": 2},
    "medium":   {"overview": 8, "safety": 8, "eff_pr": 5, "eff_reg": 4, "contra": 4, "preprint": 5,
                 "dosing": 4, "interactions": 4, "mechanism": 3, "populations": 4},
    "detailed": {"overview": 14, "safety": 16, "eff_pr": 10, "eff_reg": 8, "contra": 8, "preprint": 8,
                 "dosing": 6, "interactions": 6, "mechanism": 5, "populations": 6},
}


def retrieve_for_report(drug: str, depth: str = "medium", boost: dict = None) -> dict[str, list[dict]]:
    """Section-targeted retrieval with tier-aware biasing and variable depth.

    `depth` (short|medium|detailed) scales how many chunks each section pulls.
    `boost` (optional) adds EXTRA chunks to specific weak sections — used by the
    feedback loop's corrective pass to deepen sections that came up thin.

    Refinements carried over from the polish pass:
      - Efficacy pulls separately from peer-reviewed + regulatory (strong RCTs surface).
      - Contradiction pull retrieves BOTH supportive and opposing evidence.
    """
    drug_key = drug.lower().strip()
    profiles = config.depth_profiles()
    prof = dict(profiles.get(depth, profiles["medium"]))
    boost = boost or {}
    # Apply per-section boosts (corrective pass deepens weak sections).
    if boost.get("overview"):
        prof["overview"] += boost["overview"]
    if boost.get("safety"):
        prof["safety"] += boost["safety"]
    if boost.get("efficacy"):
        prof["eff_pr"] += boost["efficacy"]
        prof["eff_reg"] += max(1, boost["efficacy"] // 2)
    if boost.get("preprint"):
        prof["preprint"] += boost["preprint"]
    log.info(f"[retrieve] depth={depth}" + (f" boost={boost}" if boost else ""))

    section_queries = {
        "overview": f"{drug} overview indication mechanism of action what it treats",
        "safety":   f"{drug} adverse effects side effects warnings contraindications risks bleeding toxicity",
    }

    out: dict[str, list[dict]] = {}

    # Overview + Safety: standard drug-filtered retrieval, depth-scaled.
    for section, q in section_queries.items():
        qvec = embed_query(q)
        hits = vectorstore.search(qvec, top_k=prof[section], drug=drug_key)
        out[section] = _hits_to_dicts(hits)
        log.info(f"[retrieve:{section}] {len(out[section])} chunks")

    # Efficacy/Findings: pull SEPARATELY from the strongest tiers so RCTs and
    # regulatory evidence drive the findings, not whatever ranks highest overall.
    eff_q = f"{drug} efficacy clinical trial RCT outcomes effectiveness results benefit"
    eff_vec = embed_query(eff_q)
    eff = []
    eff += _hits_to_dicts(vectorstore.search(eff_vec, top_k=prof["eff_pr"], drug=drug_key, tier="peer_reviewed"))
    eff += _hits_to_dicts(vectorstore.search(eff_vec, top_k=prof["eff_reg"], drug=drug_key, tier="regulatory"))
    out["efficacy"] = _dedup(eff)
    log.info(f"[retrieve:efficacy] {len(out['efficacy'])} chunks (peer-reviewed + regulatory)")

    # Contradictions: deliberately retrieve BOTH sides so conflicts can surface.
    pro = embed_query(f"{drug} effective benefit reduces risk improves outcomes")
    con = embed_query(f"{drug} no benefit ineffective increased risk harm no significant difference")
    contra = []
    contra += _hits_to_dicts(vectorstore.search(pro, top_k=prof["contra"], drug=drug_key))
    contra += _hits_to_dicts(vectorstore.search(con, top_k=prof["contra"], drug=drug_key))
    out["contradiction"] = _dedup(contra)
    log.info(f"[retrieve:contradiction] {len(out['contradiction'])} chunks (both sides)")

    # Preprint section: dedicated pull, filtered to preprints AND this drug.
    qvec = embed_query(f"{drug} preprint emerging recent findings")
    pre_hits = vectorstore.search(qvec, top_k=prof["preprint"], tier="preprint", drug=drug_key)
    out["preprint"] = _hits_to_dicts(pre_hits)
    log.info(f"[retrieve:preprint] {len(out['preprint'])} preprint chunks")

    # B2 — richer structure. Dedicated retrieval for monograph-style sections.
    # KEY: we first pull chunks the semantic chunker already TAGGED with the
    # matching label section (high precision — a real "mechanism of action"
    # chunk), then top up with a plain semantic query if needed. This is what
    # makes Mechanism / Interactions / Populations reliably surface as their
    # own sections instead of getting folded into Safety/Overview.
    b2_specs = {
        "dosing": {
            "query": f"{drug} dosage administration recommended dose mg frequency maximum",
            "sections": ["dosage and administration", "dosage", "how supplied"],
        },
        "interactions": {
            "query": f"{drug} drug interactions concomitant use avoid combination",
            "sections": ["drug interactions", "drug interaction"],
        },
        "mechanism": {
            "query": f"{drug} mechanism of action how it works pharmacology",
            "sections": ["mechanism of action", "clinical pharmacology"],
        },
        "populations": {
            "query": f"{drug} use in specific populations pregnancy elderly pediatric renal hepatic",
            "sections": ["use in specific populations", "use in specific population",
                         "geriatric use", "pediatric use", "renal impairment",
                         "hepatic impairment"],
        },
        # ---- Med-affairs reviewer additions (Jun 2026) ----
        # These four sections were explicitly flagged as missing by a med-affairs
        # reviewer evaluating Aletheon's single-drug template. Each one targets
        # an FDA-label section that almost always exists (so the retrieval
        # consistently has material to work with) plus a focused semantic query.
        "blackbox": {
            # FDA Boxed Warnings (the "black box" — usually the FIRST thing
            # a med-affairs person checks). For NSAIDs this is CV thrombotic
            # risk + GI bleed/perforation. The SPL label is "boxed warning".
            "query": (f"{drug} boxed warning black box serious cardiovascular "
                      f"thrombotic gastrointestinal bleeding perforation"),
            "sections": ["boxed warning", "boxed warnings",
                         "warnings and precautions", "warnings",
                         "black box warning"],
        },
        "cv_risk": {
            # Cardiovascular risk profile — MI/stroke/CV death, the explicit
            # NSAID class warning. Look in W&P, CV-specific sub-sections, and
            # pulls from peer-reviewed CV outcome trials (CLASS, TARGET, etc.)
            "query": (f"{drug} cardiovascular risk myocardial infarction stroke "
                      f"thrombotic events hypertension heart failure "
                      f"CV outcomes risk"),
            "sections": ["warnings and precautions", "warnings",
                         "cardiovascular thrombotic events",
                         "cardiovascular events"],
        },
        "pregnancy": {
            # FDA SPL Section 8.1/8.2 — Pregnancy + Lactation. Plus the
            # reproductive safety bits (8.3). NSAIDs have a well-known
            # 3rd-trimester ductus closure warning so this should always
            # surface real label content.
            "query": (f"{drug} pregnancy lactation breastfeeding reproductive "
                      f"safety teratogenic fetal third trimester ductus "
                      f"arteriosus nursing mothers"),
            "sections": ["pregnancy", "lactation", "nursing mothers",
                         "use in specific populations", "reproductive",
                         "females and males of reproductive potential"],
        },
        "pk_pd": {
            # FDA SPL Section 12 — Clinical Pharmacology (incl. Pharmacokinetics).
            # Cmax, Tmax, half-life, AUC, metabolism (CYP), excretion route.
            # Med-affairs uses this to model dosing in special populations.
            "query": (f"{drug} pharmacokinetics pharmacodynamics absorption "
                      f"distribution metabolism excretion half-life Cmax Tmax "
                      f"AUC bioavailability CYP cytochrome plasma"),
            "sections": ["clinical pharmacology", "pharmacokinetics",
                         "pharmacodynamics", "mechanism of action",
                         "absorption", "metabolism"],
        },
    }
    for section, spec in b2_specs.items():
        k = prof[section]
        qvec = embed_query(spec["query"])
        # Pass 1: section-tagged chunks (precise).
        tagged = _hits_to_dicts(
            vectorstore.search(qvec, top_k=k, drug=drug_key, section=spec["sections"]))
        results = tagged
        # Pass 2: if the label-tagged pull was thin, top up with a plain query.
        if len(tagged) < k:
            extra = _hits_to_dicts(
                vectorstore.search(qvec, top_k=k, drug=drug_key))
            results = _dedup(tagged + extra)[:k]
        out[section] = results
        log.info(f"[retrieve:{section}] {len(out[section])} chunks "
                 f"({len(tagged)} section-tagged)")

    return out
