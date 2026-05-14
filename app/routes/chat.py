import logging

from fastapi import APIRouter, HTTPException
from openai import OpenAIError

from app.models.schemas import AskResponse, QuestionRequest, SourceChunk
from app.services.llm_service import generate_rag_response
from app.services.retrieval_service import retrieve_relevant_chunks
from app.utils.helpers import build_context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: QuestionRequest):
    """
    Single-turn RAG question answering pipeline:

      1. Receive user question.
      2. Embed the query and retrieve semantically relevant chunks from ChromaDB.
      3. Build a structured context string from retrieved chunks.
      4. Generate a grounded answer via OpenAI Chat Completions.
      5. Return the answer along with retrieval metadata.
    """
    logger.info("Query received: '%s'", request.question)

    # ------------------------------------------------------------------
    # Step 2: Retrieve relevant chunks
    # ------------------------------------------------------------------
    try:
        chunks = retrieve_relevant_chunks(request.question)
    except EnvironmentError as exc:
        # Missing OPENAI_API_KEY
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Retrieval failed for query '%s': %s", request.question, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve relevant context from the knowledge base.",
        )

    logger.info("Retrieved %d chunk(s) for query.", len(chunks))

    if not chunks:
        logger.warning("No chunks retrieved — knowledge base may be empty or query too broad.")
        return AskResponse(
            question=request.question,
            answer="I could not find relevant information in the provided documents.",
            retrieved_chunks=0,
            sources=[],
        )

    # ------------------------------------------------------------------
    # Step 3: Build context
    # ------------------------------------------------------------------
    context = build_context(chunks)
    logger.info("Context built from %d chunk(s) (%d chars).", len(chunks), len(context))

    # ------------------------------------------------------------------
    # Step 4: Generate grounded LLM response
    # ------------------------------------------------------------------
    try:
        answer = generate_rag_response(request.question, context)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except OpenAIError as exc:
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to generate a response from the language model.",
        )
    except Exception as exc:
        logger.error("Unexpected error during LLM generation: %s", exc)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    logger.info(
        "Response generated for query '%s' using %d chunk(s).",
        request.question[:60],
        len(chunks),
    )

    # ------------------------------------------------------------------
    # Step 5: Return answer + retrieval metadata
    # ------------------------------------------------------------------
    sources = [
        SourceChunk(
            source_file_name=chunk["metadata"]["source_file_name"],
            chunk_index=chunk["metadata"]["chunk_index"],
        )
        for chunk in chunks
    ]

    return AskResponse(
        question=request.question,
        answer=answer,
        retrieved_chunks=len(chunks),
        sources=sources,
    )