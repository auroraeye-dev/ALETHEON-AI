"""
core/models.py
================
THE most important file in Aletheon. This defines the single shape that every
data source must return. Once everything is an `Evidence` object, nothing
downstream needs to know whether it came from PubMed, FDA, or a preprint server.

This is what kills the entire class of "a field got silently dropped" bugs.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


# ---- Evidence tiers, in order of authority -------------------------------
# Used to (a) separate preprints in the report and (b) weight confidence later.
TIER_REGULATORY = "regulatory"        # FDA, EMA, DailyMed       (highest trust)
TIER_PEER_REVIEWED = "peer_reviewed"  # PubMed, Europe PMC, trials
TIER_REAL_WORLD = "real_world"        # CDC, WHO, FAERS adverse events
TIER_PREPRINT = "preprint"            # bioRxiv, medRxiv  (NOT peer-reviewed!)
TIER_PATENT = "patent"                # Lens

VALID_TIERS = {
    TIER_REGULATORY,
    TIER_PEER_REVIEWED,
    TIER_REAL_WORLD,
    TIER_PREPRINT,
    TIER_PATENT,
}


@dataclass
class Evidence:
    """One piece of evidence from any source, in a unified shape."""

    source: str                      # e.g. "pubmed", "clinicaltrials", "fda"
    source_id: str                   # PMID / NCT id / DOI — used for citations
    title: str
    text: str                        # abstract / summary / label text
    url: str                         # link back to the original
    tier: str                        # one of VALID_TIERS (see above)
    doc_type: str = "document"       # "paper", "trial", "label", "patent", ...
    date: Optional[str] = None       # publication / trial date if available
    extra: dict = field(default_factory=dict)  # anything source-specific

    def __post_init__(self):
        # Fail loudly if a source forgets to set a valid tier.
        if self.tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier {self.tier!r} for source {self.source!r}. "
                f"Must be one of: {sorted(VALID_TIERS)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        """One-line summary for logging."""
        t = self.title[:70] + ("…" if len(self.title) > 70 else "")
        return f"[{self.tier}] {self.source}:{self.source_id} — {t}"
