import logging
import uuid

from pypdf import PdfReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from all pages of a PDF file using PyPDF.

    - Skips pages that yield no extractable text (e.g. scanned image pages).
    - Raises ValueError for corrupted, empty, or unreadable PDFs so the caller
      can surface a meaningful HTTP error.
    """
    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF file: {exc}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("PDF contains no pages.")

    logger.info("Starting text extraction — %d page(s) found.", total_pages)

    page_texts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text and text.strip():
                page_texts.append(text)
            else:
                logger.warning("Page %d/%d yielded no text — skipping.", i + 1, total_pages)
        except Exception as exc:
            logger.warning("Error extracting page %d/%d: %s — skipping.", i + 1, total_pages, exc)

    if not page_texts:
        raise ValueError(
            "No extractable text found in the PDF. "
            "The file may contain only scanned images or be corrupted."
        )

    logger.info(
        "Text extraction complete: %d/%d pages had content.", len(page_texts), total_pages
    )
    return "\n".join(page_texts)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text:       The cleaned input text to split.
        chunk_size: Number of words per chunk.
        overlap:    Number of words shared between consecutive chunks
                    to preserve semantic continuity.

    Returns:
        An ordered list of non-empty chunk strings.

    Example (chunk_size=4, overlap=2):
        words = [A, B, C, D, E, F]
        chunk 0 → "A B C D"
        chunk 1 → "C D E F"
    """
    words = text.split()
    if not words:
        logger.warning("chunk_text received empty text — returning no chunks.")
        return []

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})."
        )

    chunks: list[str] = []
    step = chunk_size - overlap  # how many words to advance each iteration
    start = 0

    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        start += step

    logger.info(
        "Chunking complete: %d chunk(s) generated (chunk_size=%d, overlap=%d).",
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Chunk Metadata
# ---------------------------------------------------------------------------

def build_chunk_metadata(chunks: list[str], source_filename: str) -> list[dict]:
    """
    Wrap each chunk string in a structured metadata dictionary.

    Each entry contains:
        chunk_id        — a unique UUID string
        chunk_index     — zero-based position in the document
        source_file_name — the original PDF filename
        chunk_text      — the chunk content
    """
    metadata = [
        {
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": idx,
            "source_file_name": source_filename,
            "chunk_text": chunk,
        }
        for idx, chunk in enumerate(chunks)
    ]
    logger.info(
        "Built metadata for %d chunk(s) from '%s'.", len(metadata), source_filename
    )
    return metadata