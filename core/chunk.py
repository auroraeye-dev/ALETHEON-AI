"""
core/chunk.py
=============
Lighter SEMANTIC chunking.

Instead of slicing purely by paragraph/size, we split on *meaning boundaries*
using cheap structural+semantic signals (no extra embedding calls):

  1. SECTION HEADERS — medical text (esp. FDA labels) is organized into named
     sections ("WARNINGS", "DOSAGE AND ADMINISTRATION", "ADVERSE REACTIONS",
     "CONTRAINDICATIONS", ...). Splitting on these keeps one topic per chunk,
     so retrieval pulls a clean "warnings" chunk instead of a blob mixing
     dosing + warnings + interactions.
  2. COHERENT SENTENCE GROUPING within a section, respecting the token target.
  3. SMALL OVERLAP between consecutive chunks so an idea split across a
     boundary still appears whole in at least one chunk.

Each chunk keeps a reference back to its parent Evidence so citations survive.
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
    drug: str = ""   # the drug this chunk was ingested under (for filtering)
    section: str = ""  # detected section label (e.g. "warnings"), for context


def _rough_token_count(text: str) -> int:
    # ~4 chars per token is a fine heuristic for chunk sizing.
    return max(1, len(text) // 4)


# Common medical/label section headers. Splitting on these is the core of the
# "semantic" behavior: each becomes its own topical chunk group.
_SECTION_PATTERNS = [
    r"indications?(?:\s+and\s+usage)?",
    r"dosage(?:\s+and\s+administration)?",
    r"contraindications?",
    r"warnings?(?:\s+and\s+precautions?)?",
    r"precautions?",
    r"adverse\s+reactions?",
    r"drug\s+interactions?",
    r"use\s+in\s+specific\s+populations?",
    r"overdosage?",
    r"clinical\s+pharmacology",
    r"mechanism\s+of\s+action",
    r"clinical\s+studies",
    r"how\s+supplied",
    r"description",
    r"boxed\s+warning",
]
# A header is one of the above terms on its own line / followed by a colon,
# case-insensitive, possibly in caps.
_HEADER_RE = re.compile(
    r"(?im)^[\s>#*-]*((?:" + "|".join(_SECTION_PATTERNS) + r"))\s*[:\-—]?\s*$"
)
# Also catch inline ALLCAPS headers like "WARNINGS" or "ADVERSE REACTIONS:"
_INLINE_HEADER_RE = re.compile(
    r"(?m)(?:^|\n)\s*((?:" + "|".join(_SECTION_PATTERNS) + r"))\s*[:\-—]",
    re.IGNORECASE,
)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_label, section_text) by detected headers.
    If no headers found, returns one ('', text) covering the whole thing."""
    # Find header positions (line-based first).
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        matches = list(_INLINE_HEADER_RE.finditer(text))
    if not matches:
        return [("", text.strip())]

    sections = []
    # Text before the first header (e.g. title/intro) becomes an untitled section.
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(("", pre))

    for i, m in enumerate(matches):
        label = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((label, body))
    return sections


def _group_sentences(text: str, target: int, overlap_sentences: int = 1) -> list[str]:
    """Group sentences into chunks near `target` tokens, with small overlap so
    ideas spanning a boundary aren't lost."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    chunks, buf = [], []
    for s in sentences:
        candidate = " ".join(buf + [s])
        if _rough_token_count(candidate) > target and buf:
            chunks.append(" ".join(buf).strip())
            # start next chunk with an overlap of the last N sentences
            buf = buf[-overlap_sentences:] if overlap_sentences else []
            buf.append(s)
        else:
            buf.append(s)
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks


def _semantic_split(text: str) -> list[tuple[str, str]]:
    """Return list of (section_label, chunk_text) using semantic boundaries."""
    target = config.CHUNK_TARGET_TOKENS
    out = []
    for label, sec_text in _split_into_sections(text):
        # Within a section, split on blank lines first (natural paragraphs),
        # then group sentences to hit the token target with overlap.
        paras = [p.strip() for p in re.split(r"\n\s*\n", sec_text) if p.strip()]
        for p in paras:
            if _rough_token_count(p) <= target:
                out.append((label, p))
            else:
                for piece in _group_sentences(p, target):
                    out.append((label, piece))
    return out


def chunk_evidence(ev: Evidence, drug: str = "") -> list[Chunk]:
    """Turn one Evidence into one or more semantically-coherent Chunks."""
    # Prepend the title so the topic is always in-context for the first chunk.
    body = f"{ev.title}\n\n{ev.text}" if ev.title else ev.text
    pieces = _semantic_split(body)
    chunks = []
    for i, (section, piece) in enumerate(pieces):
        # Prepend the section label to the chunk text so retrieval embeddings
        # "know" this is e.g. a warnings chunk — cheap context, big precision win.
        text = f"[{section}] {piece}" if section else piece
        chunks.append(Chunk(
            text=text,
            source=ev.source,
            source_id=ev.source_id,
            title=ev.title,
            url=ev.url,
            tier=ev.tier,
            doc_type=ev.doc_type,
            chunk_index=i,
            drug=drug,
            section=section,
        ))
    return chunks


def chunk_all(evidence: list[Evidence], drug: str = "") -> list[Chunk]:
    chunks = []
    for ev in evidence:
        chunks.extend(chunk_evidence(ev, drug=drug))
    return chunks
