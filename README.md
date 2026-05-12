# AI Knowledge Assistant

A production-style AI Knowledge Assistant built with FastAPI, following a **Retrieval Augmented Generation (RAG)** architecture. Users can upload PDF documents and ask contextual questions answered by an LLM using retrieved document context.

> **Current phase:** Phase 1 — Backend skeleton and project foundation.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend Framework | FastAPI + Uvicorn |
| AI / LLM | OpenAI API |
| Vector Database | ChromaDB |
| PDF Parsing | PyPDF |
| Containerization | Docker |
| Config Management | python-dotenv |

---

## Project Structure

```
app/
├── main.py                    # FastAPI app entry point & health endpoint
├── routes/
│   ├── upload.py              # POST /upload — file upload handler
│   └── chat.py                # POST /ask   — question answering handler
├── services/
│   ├── pdf_service.py         # PDF parsing & chunking        (Phase 2)
│   ├── embedding_service.py   # Embedding generation          (Phase 3)
│   ├── retrieval_service.py   # Vector DB storage & retrieval (Phase 3/4)
│   └── llm_service.py         # LLM response generation       (Phase 4)
├── models/
│   └── schemas.py             # Pydantic request/response schemas
├── utils/
│   └── helpers.py             # Shared utility functions
└── data/                      # Uploaded PDF files (auto-created at runtime)
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- pip

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ai-knowledge-assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Run Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API base URL: `http://localhost:8000`
- Interactive Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Docker Setup

### Build the image

```bash
docker build -t ai-knowledge-assistant .
```

### Run the container

```bash
docker run -p 8000:8000 --env-file .env ai-knowledge-assistant
```

The API will be available at `http://localhost:8000`.

---

## API Endpoints

| Method | Endpoint  | Description                       |
|--------|-----------|-----------------------------------|
| GET    | /health   | Service health check              |
| POST   | /upload   | Upload a PDF document             |
| POST   | /ask      | Ask a question over uploaded docs |

### `GET /health`

Returns service health status.

**Response:**
```json
{
  "status": "ok"
}
```

---

### `POST /upload`

Upload a PDF file using `multipart/form-data`.

**Request:** Form field named `file` containing a `.pdf` file.

**Response:**
```json
{
  "message": "File uploaded successfully",
  "filename": "sample.pdf"
}
```

**Error (invalid file type):**
```json
{
  "detail": "Invalid file type 'image/png'. Only PDF files are accepted."
}
```

---

### `POST /ask`

Ask a natural language question over uploaded documents.

**Request body:**
```json
{
  "question": "What is the refund policy?"
}
```

**Response:**
```json
{
  "answer": "Dummy response for now"
}
```

> Real retrieval and LLM-generated answers will be available after Phase 4.

---

## Development Phases

| Phase | Description |
|---|---|
| **Phase 1** ✅ | FastAPI skeleton, health/upload/ask endpoints, Docker setup |
| **Phase 2** | PDF parsing + text chunking pipeline |
| **Phase 3** | Embedding generation + ChromaDB vector store integration |
| **Phase 4** | Semantic retrieval + OpenAI LLM response generation |
| **Phase 5** | Citations, conversation memory, streaming, multi-document support |