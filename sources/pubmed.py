"""
sources/pubmed.py
==================
DAY 2: port your existing PubMed script into this single function.

The contract every source must follow:
    fetch(drug: str) -> list[Evidence]

Rules:
  - return clean Evidence objects (NOT csv files, NOT raw json)
  - set tier = "peer_reviewed"
  - set source_id = PMID  (so the report can cite it)
  - set url = the pubmed link
  - on any error, return [] (don't crash the whole pipeline)
"""

from core.models import Evidence, TIER_PEER_REVIEWED
from core.logging_setup import log


def fetch(drug: str) -> list[Evidence]:
    log.info(f"[pubmed] fetching for {drug!r} … (STUB — fill in on Day 2)")
    # TODO Day 2: port Pubmed_complete_data_get.py logic here.
    # For each paper, build:
    #   Evidence(
    #       source="pubmed",
    #       source_id=pmid,
    #       title=title,
    #       text=abstract,
    #       url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    #       tier=TIER_PEER_REVIEWED,
    #       doc_type="paper",
    #       date=pub_date,
    #   )
    return []
