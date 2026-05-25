"""sources/europepmc.py — Europe PMC agent (papers + preprints, tier-tagged)."""
import requests
from core.models import Evidence, TIER_PREPRINT, TIER_PEER_REVIEWED
from core.logging_setup import log

EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
HEADERS = {"User-Agent": "Aletheon/0.1 (research prototype)"}


def fetch(drug: str, page_size: int = 40) -> list[Evidence]:
    log.info(f"[europepmc] searching for {drug!r} ...")
    params = {"query": drug, "format": "json", "pageSize": page_size, "resultType": "core"}
    try:
        r = requests.get(EPMC_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"[europepmc] request failed: {e}")
        return []

    results = data.get("resultList", {}).get("result", [])
    log.info(f"[europepmc] hitCount={data.get('hitCount', 0)}, got {len(results)} in page")
    if not results:
        return []

    out, n_pre = [], 0
    for item in results:
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstractText") or "").strip()
        if not (title or abstract):
            continue

        is_preprint = (item.get("source") or "").upper() == "PPR"
        if is_preprint:
            n_pre += 1
        tier = TIER_PREPRINT if is_preprint else TIER_PEER_REVIEWED

        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        doi = item.get("doi")
        if pmid:
            sid, url = pmid, f"https://europepmc.org/article/MED/{pmid}"
        elif pmcid:
            sid, url = pmcid, f"https://europepmc.org/article/PMC/{pmcid}"
        elif doi:
            sid, url = doi, f"https://doi.org/{doi}"
        else:
            sid, url = item.get("id", "unknown"), "https://europepmc.org/"

        body = title + ("\n\n" + abstract if abstract else "")
        out.append(Evidence(
            source="europepmc",
            source_id=str(sid),
            title=title or "(no title)",
            text=body,
            url=url,
            tier=tier,
            doc_type="preprint" if is_preprint else "paper",
            date=str(item.get("firstPublicationDate") or item.get("pubYear") or "") or None,
        ))

    log.info(f"[europepmc] returned {len(out)} ({n_pre} preprints, {len(out)-n_pre} peer-reviewed)")
    return out