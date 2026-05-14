import os
import re

# Absolute path to the app/data/ directory where uploaded files are stored
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_data_dir() -> str:
    """Return the path to the data directory, creating it if it doesn't already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def get_chunks_dir() -> str:
    """Return the path to the chunks sub-directory, creating it if it doesn't exist."""
    chunks_dir = os.path.join(DATA_DIR, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    return chunks_dir


def clean_extracted_text(text: str) -> str:
    """
    Lightly normalise raw PDF-extracted text while preserving semantic meaning.

    Operations performed (in order):
    1. Replace tabs, form-feeds, non-breaking spaces, and runs of spaces/tabs
       with a single regular space.
    2. Strip trailing whitespace from every line.
    3. Collapse three or more consecutive blank lines down to two (a paragraph break).
    4. Strip leading/trailing whitespace from the entire string.

    Intentionally avoids aggressive cleaning (e.g. removing punctuation or
    lowercasing) so downstream chunking and embedding remain high-quality.
    """
    # 1. Normalise horizontal whitespace (tabs, nbsp, multiple spaces → single space)
    text = re.sub(r"[ \t\f\v\xa0]+", " ", text)

    # 2. Strip trailing spaces from individual lines
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # 3. Collapse 3+ consecutive newlines into two (preserve paragraph structure)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_context(retrieved_chunks: list[dict]) -> str:
    """
    Combine a list of retrieved chunks into a single prompt-ready context string.

    Each chunk is labelled [Chunk N] for clarity. Chunks are separated by a
    blank line to keep the context readable and avoid blending adjacent passages.

    Args:
        retrieved_chunks: List of dicts as returned by retrieve_relevant_chunks().
                          Each dict must contain a "chunk_text" key.

    Returns:
        A formatted multi-chunk context string, or an empty string if no
        chunks were provided.

    Example output:
        [Chunk 1]
        All purchases are eligible for a full refund within 30 days.

        [Chunk 2]
        Refunds are processed within 5 to 7 business days.
    """
    if not retrieved_chunks:
        return ""

    parts = [
        f"[Chunk {i}]\n{chunk['chunk_text'].strip()}"
        for i, chunk in enumerate(retrieved_chunks, start=1)
        if chunk.get("chunk_text", "").strip()
    ]

    if not parts:
        return ""

    return "\n\n".join(parts)