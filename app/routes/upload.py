import json
import logging
import os
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import UploadResponse
from app.services.pdf_service import build_chunk_metadata, chunk_text, extract_text_from_pdf
from app.services.retrieval_service import store_chunks_in_vector_db
from app.utils.helpers import clean_extracted_text, get_chunks_dir, get_data_dir

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"application/pdf"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Full document ingestion and semantic indexing pipeline:

      1. Validate file type and filename.
      2. Save the uploaded PDF to app/data/.
      3. Extract raw text from all PDF pages via PyPDF.
      4. Clean extracted text (normalise whitespace / blank lines).
      5. Split cleaned text into overlapping word-based chunks.
      6. Build structured metadata for each chunk.
      7. Persist chunk metadata as JSON to app/data/chunks/.
      8. Generate OpenAI embeddings and upsert into ChromaDB.
      9. Return indexing statistics.

    LLM response generation and retrieval are deferred to Phase 4.
    """

    # ------------------------------------------------------------------
    # Step 1: Validate file
    # ------------------------------------------------------------------
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                "Only PDF files are accepted."
            ),
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no name.")

    logger.info("Upload received: '%s' (%s)", file.filename, file.content_type)

    # ------------------------------------------------------------------
    # Step 2: Save PDF to disk
    # ------------------------------------------------------------------
    pdf_path = os.path.join(get_data_dir(), file.filename)
    try:
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("PDF saved to: %s", pdf_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {exc}"
        ) from exc
    finally:
        await file.close()

    # ------------------------------------------------------------------
    # Step 3: Extract raw text
    # ------------------------------------------------------------------
    try:
        raw_text = extract_text_from_pdf(pdf_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected extraction error for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=500, detail="PDF text extraction failed unexpectedly.")

    # ------------------------------------------------------------------
    # Step 4: Clean text
    # ------------------------------------------------------------------
    cleaned_text = clean_extracted_text(raw_text)
    logger.info(
        "Text cleaned for '%s': %d characters retained.", file.filename, len(cleaned_text)
    )

    # ------------------------------------------------------------------
    # Step 5: Generate overlapping chunks
    # ------------------------------------------------------------------
    chunks = chunk_text(cleaned_text)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="PDF was processed but no text chunks could be generated.",
        )

    # ------------------------------------------------------------------
    # Step 6: Build chunk metadata
    # ------------------------------------------------------------------
    chunk_metadata = build_chunk_metadata(chunks, file.filename)

    # ------------------------------------------------------------------
    # Step 7: Persist chunk metadata JSON (for inspection / recovery)
    # ------------------------------------------------------------------
    base_name = os.path.splitext(file.filename)[0]
    chunks_json_path = os.path.join(get_chunks_dir(), f"{base_name}_chunks.json")
    try:
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
        logger.info("Chunk metadata saved to: %s", chunks_json_path)
    except Exception as exc:
        logger.error("Failed to persist chunk metadata: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save chunk metadata.")

    # ------------------------------------------------------------------
    # Step 8: Generate embeddings + store in ChromaDB
    # ------------------------------------------------------------------
    try:
        indexed_count = store_chunks_in_vector_db(chunk_metadata)
    except EnvironmentError as exc:
        # Missing OPENAI_API_KEY
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Vector indexing failed for '%s': %s", file.filename, exc
        )
        raise HTTPException(
            status_code=500,
            detail="Embedding generation or vector DB storage failed.",
        )

    # ------------------------------------------------------------------
    # Step 9: Return summary
    # ------------------------------------------------------------------
    logger.info(
        "Pipeline complete for '%s': %d chunk(s) generated, %d indexed.",
        file.filename,
        len(chunks),
        indexed_count,
    )
    return UploadResponse(
        message="PDF processed and indexed successfully",
        filename=file.filename,
        total_chunks=len(chunks),
        indexed_chunks=indexed_count,
    )