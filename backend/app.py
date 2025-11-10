import io
import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader

from classifier import simple_classify, suggest_reply
from hf_client import classify_with_hf, generate_reply_with_hf


app = FastAPI(title="AutoU Email Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessResponse(BaseModel):
    category: str
    reply: str
    preview: str
    provider: str  


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts).strip()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "hf": bool(os.getenv("HF_API_TOKEN", "")),
        "hf_classify_model": os.getenv("HF_CLASSIFY_MODEL", "facebook/bart-large-mnli"),
        "hf_reply_model": os.getenv("HF_REPLY_MODEL", "tiiuae/falcon-7b-instruct"),
    }


@app.post("/process", response_model=ProcessResponse)
async def process_email(
    file: UploadFile | None = None,
    text: str | None = Form(default=None),
):
    content = ""
    if file and file.filename:
        data = await file.read()
        fname = (file.filename or "").lower()
        if fname.endswith(".pdf"):
            content = extract_text_from_pdf(data)
        else:
            content = data.decode("utf-8", errors="ignore")
    elif text:
        content = text

    content = (content or "").strip()
    if not content:
        return ProcessResponse(
            category="Improdutivo",
            reply="Não foi possível ler conteúdo. Envie um .txt/.pdf válido ou cole o texto.",
            preview="",
            provider="heuristic",
        )

    preview = content[:400] + ("..." if len(content) > 400 else "")

    category_hf = classify_with_hf(content)
    if category_hf is not None:
        reply_hf = generate_reply_with_hf(category_hf, content) or suggest_reply(category_hf, content)
        return ProcessResponse(
            category=category_hf,
            reply=reply_hf,
            preview=preview,
            provider="huggingface",
        )

    category = simple_classify(content)
    reply = suggest_reply(category, content)
    return ProcessResponse(
        category=category,
        reply=reply,
        preview=preview,
        provider="heuristic",
    )
