import logging
import os
from typing import Optional

import chromadb
from chromadb.config import Settings

from app.services.embedding_service import generate_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"

# Persistent ChromaDB storage lives inside app/data/chroma_db/
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")

# Module-level singleton for the ChromaDB client
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_collection() -> chromadb.Collection:
    """
    Return the ChromaDB collection, initialising the persistent client on first use.

    Uses cosine similarity as the distance metric — appropriate for semantic search
    over OpenAI embedding vectors.
    """
    global _chroma_client

    if _chroma_client is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB persistent client initialised at: %s", CHROMA_DB_PATH)

    collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_chunks_in_vector_db(chunks: list[dict]) -> int:
    """
    Generate embeddings for all chunks and upsert them into ChromaDB.

    Using upsert (instead of add) ensures that re-uploading the same document
    overwrites stale vectors rather than creating duplicates.

    Args:
        chunks: List of chunk metadata dicts produced by build_chunk_metadata().
                Each dict must contain: chunk_id, chunk_index,
                source_file_name, chunk_text.

    Returns:
        The number of chunks successfully indexed.

    Raises:
        Exception: Propagates embedding or ChromaDB failures to the caller
                   so the route layer can return a meaningful HTTP error.
    """
    if not chunks:
        logger.warning("store_chunks_in_vector_db received an empty chunk list — nothing to index.")
        return 0

    total = len(chunks)
    logger.info("Starting vector indexing for %d chunk(s).", total)

    # ------------------------------------------------------------------
    # Generate embeddings in a single batched API call
    # ------------------------------------------------------------------
    texts = [chunk["chunk_text"] for chunk in chunks]
    try:
        embeddings = generate_embeddings(texts)
    except Exception as exc:
        logger.error("Embedding generation failed during indexing: %s", exc)
        raise

    logger.info("Embeddings ready — upserting into ChromaDB collection '%s'.", COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Prepare ChromaDB payload
    # ------------------------------------------------------------------
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "source_file_name": chunk["source_file_name"],
        }
        for chunk in chunks
    ]

    # ------------------------------------------------------------------
    # Upsert into collection
    # ------------------------------------------------------------------
    try:
        collection = _get_collection()
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(
            "Successfully indexed %d/%d chunk(s) into '%s'.",
            total,
            total,
            COLLECTION_NAME,
        )
        return total
    except Exception as exc:
        logger.error("ChromaDB upsert failed: %s", exc)
        raise