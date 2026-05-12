from fastapi import APIRouter

from app.models.schemas import AskResponse, QuestionRequest

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: QuestionRequest):
    """
    Accept a natural language question and return an answer.
    Retrieval pipeline and LLM integration will be wired here in Phase 4.
    """
    # Placeholder — real retrieval + LLM response comes in Phase 4
    return AskResponse(answer="Dummy response for now")