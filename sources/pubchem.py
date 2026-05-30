"""
sources/pubchem.py
==================
PubChem (NIH) — drug chemistry and structural data.

Endpoint: https://pubchem.ncbi.nlm.nih.gov/rest/pug  (PUG REST, no auth)
Flow:
  1. /compound/name/<drug>/cids/JSON       -> CID(s)
  2. /compound/cid/<CID>/property/<list>/JSON -> properties (formula, weight, etc.)
  3. /compound/cid/<CID>/synonyms/JSON     -> brand/generic synonyms

Returns ONE Evidence per resolved compound describing its chemistry. Tier =
regulatory (it's curated NIH reference data; not a clinical claim, but it's
authoritative reference information — closer to regulatory than peer-reviewed).

NETWORK NOTE: PubChem lives on ncbi.nlm.nih.gov, which has been intermittently
DNS-blocked on the user's network (same as DailyMed/PubMed). Fails gracefully
when blocked.
"""

import requests
from core.models import Evidence, TIER_REGULATORY
from core.logging_setup import log

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)", "Accept": "application/json"}
PROPS = ("MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName,"
         "XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount")


def _get_cid(drug: str) -> int | None:
    try:
        r = requests.get(f"{BASE}/compound/name/{drug}/cids/JSON",
                         headers=HEADERS, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None
    except Exception as e:
        log.warning(f"[pubchem] CID lookup failed: {e}")
        return None


def _get_props(cid: int) -> dict:
    try:
        r = requests.get(f"{BASE}/compound/cid/{cid}/property/{PROPS}/JSON",
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        rows = r.json().get("PropertyTable", {}).get("Properties", [])
        return rows[0] if rows else {}
    except Exception as e:
        log.warning(f"[pubchem] properties failed: {e}")
        return {}


def _get_synonyms(cid: int, max_syn: int = 8) -> list[str]:
    try:
        r = requests.get(f"{BASE}/compound/cid/{cid}/synonyms/JSON",
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        rows = r.json().get("InformationList", {}).get("Information", [])
        syns = rows[0].get("Synonym", []) if rows else []
        return syns[:max_syn]
    except Exception:
        return []


def fetch(drug: str) -> list[Evidence]:
    log.info(f"[pubchem] looking up '{drug}' ...")
    cid = _get_cid(drug)
    if cid is None:
        log.info(f"[pubchem] no compound found for '{drug}'")
        return []
    props = _get_props(cid)
    if not props:
        return []
    syns = _get_synonyms(cid)

    # Build a single, dense Evidence document describing the compound's chemistry.
    text_parts = [f"PubChem CID: {cid}"]
    if props.get("IUPACName"):
        text_parts.append(f"IUPAC name: {props['IUPACName']}")
    if props.get("MolecularFormula"):
        text_parts.append(f"Molecular formula: {props['MolecularFormula']}")
    if props.get("MolecularWeight"):
        text_parts.append(f"Molecular weight: {props['MolecularWeight']} g/mol")
    if props.get("CanonicalSMILES"):
        text_parts.append(f"SMILES: {props['CanonicalSMILES']}")
    # Drug-likeness descriptors (Lipinski-relevant)
    if props.get("XLogP") is not None:
        text_parts.append(f"XLogP (lipophilicity): {props['XLogP']}")
    if props.get("HBondDonorCount") is not None:
        text_parts.append(f"H-bond donors: {props['HBondDonorCount']}")
    if props.get("HBondAcceptorCount") is not None:
        text_parts.append(f"H-bond acceptors: {props['HBondAcceptorCount']}")
    if props.get("RotatableBondCount") is not None:
        text_parts.append(f"Rotatable bonds: {props['RotatableBondCount']}")
    if syns:
        text_parts.append(f"Synonyms: {', '.join(syns)}")

    text = ". ".join(text_parts) + "."
    ev = Evidence(
        source="pubchem",
        source_id=str(cid),
        title=f"PubChem chemistry: {drug.title()}",
        text=text,
        url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        tier=TIER_REGULATORY,
        doc_type="chemistry",
        date=None,
        extra={"section": "mechanism of action", "cid": cid},
    )
    log.info(f"[pubchem] returned chemistry for '{drug}' (CID {cid})")
    return [ev]
