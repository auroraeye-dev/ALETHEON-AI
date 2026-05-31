"""
sources/chembl.py
=================
ChEMBL (EBI) — bioactivity data, mechanism of action, drug targets.

Endpoint: https://www.ebi.ac.uk/chembl/api/data  (open REST, no auth)
Flow:
  1. /molecule/search.json?q=<drug>     -> ChEMBL molecule(s)
  2. /mechanism.json?molecule_chembl_id=<id>  -> mechanism + target
  3. /drug.json?molecule_chembl_id=<id> -> approved-drug info if applicable

Returns ONE Evidence per resolved molecule describing its mechanism and targets.
Tier = peer_reviewed (it's curated experimental bioactivity data from the
literature, more like high-quality scientific reference than a regulatory claim).

NETWORK NOTE: ChEMBL is on ebi.ac.uk (same family as Europe PMC) — should be
accessible on the user's normal network.
"""

import requests
from core.models import Evidence, TIER_PEER_REVIEWED
from core.logging_setup import log

BASE = "https://www.ebi.ac.uk/chembl/api/data"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)", "Accept": "application/json"}


def _search_molecule(drug: str) -> dict | None:
    try:
        r = requests.get(f"{BASE}/molecule/search.json",
                         params={"q": drug, "limit": 5},
                         headers=HEADERS, timeout=25)
        r.raise_for_status()
        data = r.json()
        molecules = data.get("molecules", [])
        if not molecules:
            return None

        # Require that the queried drug name appears in the molecule's pref_name
        # or one of its synonyms — otherwise ChEMBL's fuzzy text search will
        # silently match any drug name embedded inside a sentence (e.g. the
        # query "comparison between ibuprofen and aspirin" matched on
        # "aspirin" and returned aspirin's record as if that were the answer).
        # We want clean misses for malformed/multi-drug queries, not silent
        # wrong-drug hits.
        drug_l = drug.strip().lower()

        def _is_close_name_match(m: dict) -> bool:
            # Acceptable matches: exact equality, OR queried name is a short
            # exact-token subset of the molecule's pref_name / synonyms (e.g.
            # query "asa" matches synonym "ASA"). We deliberately do NOT match
            # the other direction (molecule_name in drug_l), because that's what
            # caused the bug where the long query
            # "comparison between ibuprofen and aspirin" silently matched aspirin.
            pref = (m.get("pref_name") or "").lower()
            if drug_l == pref:
                return True
            # If the user typed a short query, allow it as a token of pref_name.
            if len(drug_l) >= 3 and drug_l in pref.split():
                return True
            for s in (m.get("molecule_synonyms") or []):
                name = (s.get("molecule_synonym") or s.get("synonyms") or "").lower()
                if drug_l == name:
                    return True
                if len(drug_l) >= 3 and drug_l in name.split():
                    return True
            return False

        # First pass: approved drug (max_phase=4) with a name match.
        for m in molecules:
            if m.get("max_phase") in (4, "4") and _is_close_name_match(m):
                return m
        # Second pass: any name match.
        for m in molecules:
            if _is_close_name_match(m):
                return m
        # No real name match — bail rather than returning a silently-wrong drug.
        log.info(f"[chembl] no close name match for {drug!r} in top results")
        return None
    except Exception as e:
        log.warning(f"[chembl] molecule search failed: {e}")
        return None


def _get_mechanisms(chembl_id: str) -> list[dict]:
    try:
        r = requests.get(f"{BASE}/mechanism.json",
                         params={"molecule_chembl_id": chembl_id, "limit": 10},
                         headers=HEADERS, timeout=25)
        r.raise_for_status()
        return r.json().get("mechanisms", [])
    except Exception as e:
        log.warning(f"[chembl] mechanism lookup failed: {e}")
        return []


def _get_target_name(target_chembl_id: str) -> str:
    if not target_chembl_id:
        return ""
    try:
        r = requests.get(f"{BASE}/target/{target_chembl_id}.json",
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get("pref_name", "") or ""
    except Exception:
        return ""


def fetch(drug: str) -> list[Evidence]:
    log.info(f"[chembl] looking up '{drug}' ...")
    mol = _search_molecule(drug)
    if not mol:
        log.info(f"[chembl] no molecule found for '{drug}'")
        return []
    chembl_id = mol.get("molecule_chembl_id", "")
    pref_name = mol.get("pref_name") or drug.title()
    max_phase = mol.get("max_phase")

    mechs = _get_mechanisms(chembl_id)

    # Compose evidence text
    parts = [f"ChEMBL ID: {chembl_id}", f"Preferred name: {pref_name}"]
    if max_phase is not None:
        phase_label = {4: "approved drug", 3: "Phase 3", 2: "Phase 2",
                       1: "Phase 1", 0: "preclinical"}.get(max_phase, f"phase {max_phase}")
        parts.append(f"Development status: {phase_label}")

    if mechs:
        parts.append("Reported mechanism(s) of action:")
        for m in mechs[:6]:
            action = m.get("action_type") or m.get("mechanism_of_action") or ""
            target_id = m.get("target_chembl_id", "")
            target = _get_target_name(target_id) if target_id else ""
            moa = m.get("mechanism_of_action", "")
            line = f"- {moa or action}"
            if target:
                line += f" (target: {target})"
            parts.append(line)
    else:
        parts.append("No specific mechanism records in ChEMBL.")

    text = "\n".join(parts)
    ev = Evidence(
        source="chembl",
        source_id=chembl_id,
        title=f"ChEMBL bioactivity: {pref_name}",
        text=text,
        url=f"https://www.ebi.ac.uk/chembl/explore/compound/{chembl_id}",
        tier=TIER_PEER_REVIEWED,
        doc_type="bioactivity",
        date=None,
        extra={"section": "mechanism of action", "chembl_id": chembl_id},
    )
    log.info(f"[chembl] returned {chembl_id} with {len(mechs)} mechanism(s) for '{drug}'")
    return [ev]
