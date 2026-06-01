"""
core/config.py
================
Loads settings and secrets from a .env file (never hard-code keys).
Everything configurable lives here so you change it in one place.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root once, when this module is imported.
load_dotenv()


class Config:
    # ---- API keys (set these in your .env file) ----
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")          # optional, raises PubMed limits
    LENS_API_KEY = os.getenv("LENS_API_KEY", "")          # for patents (later)

    # ---- Models ----
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")     # cheap + good for reports

    # ---- Qdrant (local) ----
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "aletheon_evidence")

    # ---- Chunking / retrieval defaults ----
    CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "400"))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "8"))

    # ---- E5: tunable knobs ----
    # Per-source result limits (how many records each source fetches).
    FDA_MAX_LABELS = int(os.getenv("FDA_MAX_LABELS", "10"))
    CLINICALTRIALS_PAGE_SIZE = int(os.getenv("CLINICALTRIALS_PAGE_SIZE", "50"))
    EUROPEPMC_PAGE_SIZE = int(os.getenv("EUROPEPMC_PAGE_SIZE", "100"))
    SEMANTIC_SCHOLAR_PAGE_SIZE = int(os.getenv("SEMANTIC_SCHOLAR_PAGE_SIZE", "50"))
    FAERS_TOP_N = int(os.getenv("FAERS_TOP_N", "25"))
    DAILYMED_MAX_LABELS = int(os.getenv("DAILYMED_MAX_LABELS", "3"))

    # LLM sampling temperature for report / comparison / critic generation.
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    CRITIC_TEMPERATURE = float(os.getenv("CRITIC_TEMPERATURE", "0.3"))

    # Depth profiles (B1) — how many chunks each section pulls per depth level.
    # The single biggest quality/length/cost knob. Override any value via env,
    # e.g. DEPTH_DETAILED_SAFETY=20. Falls back to the defaults below.
    @classmethod
    def depth_profiles(cls) -> dict:
        def _n(name, default):
            return int(os.getenv(name, str(default)))
        return {
            "short": {
                "overview": _n("DEPTH_SHORT_OVERVIEW", 3),
                "safety": _n("DEPTH_SHORT_SAFETY", 4),
                "eff_pr": _n("DEPTH_SHORT_EFF_PR", 2),
                "eff_reg": _n("DEPTH_SHORT_EFF_REG", 2),
                "contra": _n("DEPTH_SHORT_CONTRA", 2),
                "preprint": _n("DEPTH_SHORT_PREPRINT", 2),
                "dosing": _n("DEPTH_SHORT_DOSING", 2),
                "interactions": _n("DEPTH_SHORT_INTERACTIONS", 2),
                "mechanism": _n("DEPTH_SHORT_MECHANISM", 2),
                "populations": _n("DEPTH_SHORT_POPULATIONS", 2),
            },
            "medium": {
                "overview": _n("DEPTH_MEDIUM_OVERVIEW", 8),
                "safety": _n("DEPTH_MEDIUM_SAFETY", 8),
                "eff_pr": _n("DEPTH_MEDIUM_EFF_PR", 5),
                "eff_reg": _n("DEPTH_MEDIUM_EFF_REG", 4),
                "contra": _n("DEPTH_MEDIUM_CONTRA", 4),
                "preprint": _n("DEPTH_MEDIUM_PREPRINT", 5),
                "dosing": _n("DEPTH_MEDIUM_DOSING", 4),
                "interactions": _n("DEPTH_MEDIUM_INTERACTIONS", 4),
                "mechanism": _n("DEPTH_MEDIUM_MECHANISM", 3),
                "populations": _n("DEPTH_MEDIUM_POPULATIONS", 4),
            },
            "detailed": {
                "overview": _n("DEPTH_DETAILED_OVERVIEW", 14),
                "safety": _n("DEPTH_DETAILED_SAFETY", 16),
                "eff_pr": _n("DEPTH_DETAILED_EFF_PR", 10),
                "eff_reg": _n("DEPTH_DETAILED_EFF_REG", 8),
                "contra": _n("DEPTH_DETAILED_CONTRA", 8),
                "preprint": _n("DEPTH_DETAILED_PREPRINT", 8),
                "dosing": _n("DEPTH_DETAILED_DOSING", 6),
                "interactions": _n("DEPTH_DETAILED_INTERACTIONS", 6),
                "mechanism": _n("DEPTH_DETAILED_MECHANISM", 5),
                "populations": _n("DEPTH_DETAILED_POPULATIONS", 6),
            },
        }

    # ---- Paths ----
    DATA_DIR = os.getenv("DATA_DIR", "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    REPORTS_DIR = os.path.join(DATA_DIR, "reports")
    CACHE_DIR = os.path.join(DATA_DIR, "cache")
    # How long a cached fetch stays fresh (hours). Drug evidence changes slowly,
    # so a day is reasonable for a prototype; tune as needed.
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

    @classmethod
    def check(cls):
        """Warn (don't crash) if important keys are missing."""
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY (needed for embeddings + report)")
        return missing


config = Config()
