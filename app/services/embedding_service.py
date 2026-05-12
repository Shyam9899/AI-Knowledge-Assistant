import logging
import os
from typing import Optional

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"

# Module-level singleton — avoids re-creating the client on every call
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return the shared OpenAI client, initialising it on first use."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file before generating embeddings."
            )
        _client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialised (model: %s).", EMBEDDING_MODEL)
    return _client


def generate_embedding(text: str) -> list[float]:
    """
    Generate a single embedding vector for the given text string.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        OpenAIError: On API-level failures (rate limit, invalid key, etc.).
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate an embedding for empty text.")

    try:
        response = _get_client().embeddings.create(input=text, model=EMBEDDING_MODEL)
        embedding = response.data[0].embedding
        logger.debug("Generated embedding for text of length %d.", len(text))
        return embedding
    except OpenAIError as exc:
        logger.error("OpenAI embedding request failed: %s", exc)
        raise


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts in a single batched API call.

    Batching all texts into one request minimises latency and API overhead
    compared to calling generate_embedding() in a loop.

    Args:
        texts: A list of input strings to embed.

    Returns:
        A list of embedding vectors in the same order as the input list.

    Raises:
        ValueError:   If the texts list is empty.
        OpenAIError:  On API-level failures.
    """
    if not texts:
        raise ValueError("texts list must not be empty.")

    logger.info(
        "Requesting batch embeddings for %d text(s) via %s.", len(texts), EMBEDDING_MODEL
    )

    try:
        response = _get_client().embeddings.create(input=texts, model=EMBEDDING_MODEL)
        # response.data is guaranteed to be ordered in the same order as the input
        embeddings = [item.embedding for item in response.data]
        logger.info("Batch embedding complete: %d vector(s) returned.", len(embeddings))
        return embeddings
    except OpenAIError as exc:
        logger.error("OpenAI batch embedding request failed: %s", exc)
        raise