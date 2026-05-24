"""
storage/vectorstore.py
======================
DAY 2: Qdrant wrapper — push chunks (with metadata incl. tier) and search.
STUB for now.
"""
"""
storage/vectorstore.py
======================
DAY 2: Qdrant wrapper in EMBEDDED mode (no Docker needed).

Qdrant stores vectors in a local folder (data/qdrant). We push chunks with
their embeddings + metadata (incl. tier + source for citations and filtering),
and we can search by a query vector.

Switching to a real Qdrant server later (Docker / AWS) is a ONE-LINE change
in _get_client() — nothing else in the codebase changes.
"""

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from core.config import config
from core.embed import EMBED_DIM
from core.chunk import Chunk
from core.logging_setup import log

_client = None
QDRANT_PATH = os.path.join(config.DATA_DIR, "qdrant")


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        # EMBEDDED mode: local on-disk storage, no server, no Docker.
        # To use a server later:  QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def ensure_collection():
    """Create the collection if it doesn't exist yet."""
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        log.info(f"[qdrant] created collection {config.QDRANT_COLLECTION!r}")


def index_chunks(chunks: list[Chunk], vectors: list[list[float]]):
    """Store chunks + their vectors with metadata payloads."""
    if not chunks:
        log.info("[qdrant] nothing to index")
        return
    ensure_collection()
    client = _get_client()
    points = []
    for ch, vec in zip(chunks, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": ch.text,
                "source": ch.source,
                "source_id": ch.source_id,
                "title": ch.title,
                "url": ch.url,
                "tier": ch.tier,
                "doc_type": ch.doc_type,
                "chunk_index": ch.chunk_index,
            },
        ))
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    log.info(f"[qdrant] indexed {len(points)} chunks")


def search(query_vector: list[float], top_k: int = None, tier: str = None):
    """Return the top_k most similar chunks. Optionally filter by tier."""
    client = _get_client()
    top_k = top_k or config.RETRIEVE_TOP_K
    qfilter = None
    if tier:
        qfilter = Filter(must=[FieldCondition(key="tier", match=MatchValue(value=tier))])
    hits = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    ).points
    return hits  # each hit has .score and .payload


def count() -> int:
    client = _get_client()
    try:
        return client.count(collection_name=config.QDRANT_COLLECTION).count
    except Exception:
        return 0


def reset():
    """Wipe the collection (handy during dev)."""
    client = _get_client()
    try:
        client.delete_collection(config.QDRANT_COLLECTION)
        log.info("[qdrant] collection reset")
    except Exception:
        pass
