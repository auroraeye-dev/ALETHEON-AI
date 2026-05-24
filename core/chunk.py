"""
core/chunk.py
============
STUB — filled in later in the plan. See Aletheon_14Day_Sprint_Plan.md.
"""
"""
core/chunk.py
=============
DAY 2: simple paragraph-aware chunking (Day 7 upgrades this to be smarter).

Splits an Evidence's text into chunks that try to break at paragraph / sentence
boundaries rather than mid-word. Each chunk keeps a reference back to its parent
Evidence so citations survive all the way to the report.
"""

import re
from dataclasses import dataclass

from core.models import Evidence
from core.config import config


@dataclass
class Chunk:
    text: str
    # carried-through metadata so we can cite + filter later
    source: str
    source_id: str
    title: str
    url: str
    tier: str
    doc_type: str
    chunk_index: int


def _rough_token_count(text: str) -> int:
    # ~4 chars per token is a fine heuristic for chunk sizing (Day 8 uses tiktoken).
    return max(1, len(text) // 4)


def _split_paragraphs(text: str) -> list[str]:
    # Split on blank lines first; fall back to sentence-ish splits for long blobs.
    paras = re.split(r"\n\s*\n", text.strip())
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if _rough_token_count(p) <= config.CHUNK_TARGET_TOKENS:
            out.append(p)
        else:
            # too long: split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sentences:
                if _rough_token_count(buf + " " + s) > config.CHUNK_TARGET_TOKENS and buf:
                    out.append(buf.strip())
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                out.append(buf.strip())
    return out


def chunk_evidence(ev: Evidence) -> list[Chunk]:
    """Turn one Evidence into one or more Chunks (boundary-aware)."""
    # Prepend the title to the first chunk so the topic is always in-context.
    body = f"{ev.title}\n\n{ev.text}" if ev.title else ev.text
    pieces = _split_paragraphs(body)
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(Chunk(
            text=piece,
            source=ev.source,
            source_id=ev.source_id,
            title=ev.title,
            url=ev.url,
            tier=ev.tier,
            doc_type=ev.doc_type,
            chunk_index=i,
        ))
    return chunks


def chunk_all(evidence: list[Evidence]) -> list[Chunk]:
    chunks = []
    for ev in evidence:
        chunks.extend(chunk_evidence(ev))
    return chunks
