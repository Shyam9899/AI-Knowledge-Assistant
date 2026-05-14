import logging
import os
from typing import Optional

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

LLM_MODEL = "gpt-4.1-mini"

# Module-level singleton — mirrors the pattern used in embedding_service
_client: Optional[OpenAI] = None

# ---------------------------------------------------------------------------
# System prompt — grounds the model strictly in provided context
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions strictly using "
    "the provided context.\n\n"
    "Rules:\n"
    "- Answer only from the information present in the context below.\n"
    "- If the answer cannot be found in the context, respond with exactly: "
    "\"I could not find relevant information in the provided documents.\"\n"
    "- Do not fabricate information or use outside knowledge.\n"
    "- Be factual, concise, and grounded in the source material."
)


def _get_client() -> OpenAI:
    """Return the shared OpenAI client, initialising it on first use."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file before generating responses."
            )
        _client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialised for LLM service (model: %s).", LLM_MODEL)
    return _client


def generate_rag_response(question: str, context: str) -> str:
    """
    Generate a grounded response to the user's question using retrieved context.

    Constructs a two-message chat prompt:
      - System message: instructs the model to answer only from context.
      - User message:   provides the retrieved context and the question.

    Args:
        question: The user's natural language question.
        context:  Pre-built context string from build_context().

    Returns:
        A factual, context-grounded answer string.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is missing.
        OpenAIError:      On API-level failures (rate limit, auth, etc.).
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    # If retrieval returned nothing, short-circuit before calling the API
    if not context or not context.strip():
        logger.warning("generate_rag_response called with empty context — returning fallback.")
        return "I could not find relevant information in the provided documents."

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question.strip()}"

    logger.info("Sending prompt to %s (context length: %d chars).", LLM_MODEL, len(context))

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # Low temp for factual, deterministic answers
        )
        answer = response.choices[0].message.content
        logger.info("LLM response generated successfully.")
        return answer.strip() if answer else "No response was generated."
    except OpenAIError as exc:
        logger.error("OpenAI chat completion request failed: %s", exc)
        raise