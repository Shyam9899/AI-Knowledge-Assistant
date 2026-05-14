from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Request body for the POST /ask endpoint."""

    question: str


class UploadResponse(BaseModel):
    """Response body for the POST /upload endpoint."""

    message: str
    filename: str
    total_chunks: int
    indexed_chunks: int


class SourceChunk(BaseModel):
    """Metadata for a single retrieved source chunk."""

    source_file_name: str
    chunk_index: int


class AskResponse(BaseModel):
    """Response body for the POST /ask endpoint."""

    question: str
    answer: str
    retrieved_chunks: int
    sources: list[SourceChunk] = []


class ChunkMetadata(BaseModel):
    """Structured metadata for a single document chunk."""

    chunk_id: str
    chunk_index: int
    source_file_name: str
    chunk_text: str