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
