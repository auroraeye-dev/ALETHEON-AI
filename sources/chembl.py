"""
sources/chembl.py
=================
ChEMBL (EBI) — bioactivity, mechanism of action, drug targets.
Endpoint: https://www.ebi.ac.uk/chembl/api/data  (open REST, no auth)
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
        molecules = r.json().get("molecules", [])
        if not molecules:
            return None
        for m in molecules:
            if m.get("max_phase") in (4, "4"):
                return m
        return molecules[0]
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
    log.info(f"[chembl] looking up {drug!r} ...")
    mol = _search_molecule(drug)
    if not mol:
        log.info(f"[chembl] no molecule found for {drug!r}")
        return []
    chembl_id = mol.get("molecule_chembl_id", "")
    pref_name = mol.get("pref_name") or drug.title()
    max_phase = mol.get("max_phase")

    mechs = _get_mechanisms(chembl_id)

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
    log.info(f"[chembl] returned {chembl_id} with {len(mechs)} mechanism(s) for {drug!r}")
    return [ev]