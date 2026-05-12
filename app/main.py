import os

from dotenv import load_dotenv
from fastapi import FastAPI

from app.routes import chat, upload

load_dotenv()

app = FastAPI(
    title="AI Knowledge Assistant",
    description="A RAG-based AI assistant that answers questions from your uploaded documents.",
    version="1.0.0",
)

# --- Router registration ---
app.include_router(upload.router, tags=["Upload"])
app.include_router(chat.router, tags=["Chat"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Returns service health status."""
    return {"status": "ok"}