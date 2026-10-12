# app.py
import io
import os
import uuid
from collections import OrderedDict
from pathlib import Path

import anthropic
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prompts import build_document_block, build_messages, build_system

PROVIDER = os.getenv("PROVIDER", "anthropic")          # "anthropic" or "ollama"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB file limit
MAX_DOC_CHARS = 150_000               # text sent to the model per document
MAX_DOCS_IN_MEMORY = 20

claude = anthropic.Anthropic() if PROVIDER == "anthropic" else None  # reads ANTHROPIC_API_KEY
DOCS: "OrderedDict[str, dict]" = OrderedDict()   # doc_id -> {"name", "text"}, kept in RAM only

app = FastAPI(title="StudyBuddy")


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=30)
    technique: str = "zero_shot"          # zero_shot | few_shot | role
    role: str = "general"
    level: str = "teen"                   # kid | teen | college
    extras: list[str] = Field(default_factory=list)
    doc_id: str | None = None


def extract_text(ext: str, data: bytes) -> str:
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    # .txt, .md, .csv
    return data.decode("utf-8", errors="replace")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "document"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file. Use one of: {', '.join(sorted(ALLOWED_EXT))}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 10 MB.")

    try:
        text = extract_text(ext, data).strip()
    except Exception as e:
        raise HTTPException(400, f"Could not read this file: {e}")

    if not text:
        raise HTTPException(400, "No readable text found. Scanned PDFs need OCR first.")

    truncated = len(text) > MAX_DOC_CHARS
    text = text[:MAX_DOC_CHARS]

    doc_id = uuid.uuid4().hex
    DOCS[doc_id] = {"name": name, "text": text}
    if len(DOCS) > MAX_DOCS_IN_MEMORY:
        DOCS.popitem(last=False)   # drop the oldest document

    return {"doc_id": doc_id, "name": name, "chars": len(text), "truncated": truncated}


@app.post("/api/chat")
def chat(body: ChatIn):
    doc = None
    if body.doc_id:
        doc = DOCS.get(body.doc_id)
        if doc is None:
            raise HTTPException(404, "Document not found. Please upload it again.")

    system = build_system(body.technique, body.role, body.level, body.extras, has_doc=doc is not None)
    doc_block = build_document_block(doc["name"], doc["text"]) if doc else None
    messages = build_messages(body.technique, body.message, body.history)

    try:
        if PROVIDER == "ollama":
            full_system = system + (f"\n\n{doc_block}" if doc_block else "")
            r = httpx.post(OLLAMA_URL, timeout=180, json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [{"role": "system", "content": full_system}, *messages],
            })
            r.raise_for_status()
            reply = r.json()["message"]["content"]
        else:
            system_blocks = [{"type": "text", "text": system}]
            if doc_block:
                # Cache marker: follow-up questions about the same file reuse the cached document.
                system_blocks.append({
                    "type": "text",
                    "text": doc_block,
                    "cache_control": {"type": "ephemeral"},
                })
            resp = claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=system_blocks,
                messages=messages,
            )
            reply = "".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:
        raise HTTPException(502, f"Model error: {e}")

    return {"reply": reply}


@app.get("/api/config")
def config():
    model = CLAUDE_MODEL if PROVIDER == "anthropic" else OLLAMA_MODEL
    return {"provider": PROVIDER, "model": model}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")