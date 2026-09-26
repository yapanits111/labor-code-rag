"""Web app: a JSON API plus a single-page UI.

    uvicorn app.main:app --reload      # http://localhost:8000
"""
import re
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from labor_rag.embed import MODEL_NAME, embed_query
from labor_rag.generate import answer, passage_header
from labor_rag.llm import DEFAULT_PROVIDER, LABELS, SUPPORTED_PROVIDERS, is_configured
from labor_rag.store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

SOURCE_PDFS = {  # public copies of the two documents, for "open at page" links
    "labor_code": "https://natlex.ilo.org/dyn/natlex2/natlex2/files/download/15242/PHL15242%202022.pdf",
    "handbook": "https://nwpc.dole.gov.ph/wp-content/uploads/2024/11/"
                "Workers-Statutory-Monetary-Benefits-Handbook-2024-Edition.pdf",
}
RATE_LIMIT = (8, 60)  # requests per window (seconds), per client: protects the free LLM quota
# Free LLM tiers hit rate limits; try these in order rather than show a visitor
# an error. The eval measures the first one.
FALLBACK_ORDER = ["groq", "groq-small", "gemini", "claude"]

@asynccontextmanager
async def lifespan(_app: FastAPI):
    embed_query("warm up")  # load the embedding model at startup, not on the first question
    yield


app = FastAPI(title="Labor Code PH Assistant", lifespan=lifespan,
              description="Answers about Philippine labor law, grounded in the Labor Code "
                          "and DOLE's Handbook, with citations.")
store = VectorStore.load(ROOT / "index", expected_model=MODEL_NAME)
_requests: dict[str, deque] = defaultdict(deque)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    provider: str = DEFAULT_PROVIDER


def _rate_limited(client: str) -> bool:
    limit, window = RATE_LIMIT
    now, hits = time.monotonic(), _requests[client]
    while hits and now - hits[0] > window:
        hits.popleft()
    if len(hits) >= limit:
        return True
    hits.append(now)
    return False


def _passage(n: int, chunk: dict, cited: list[int]) -> dict:
    if chunk["doc"] == "labor_code":
        source = "Labor Code"
        ref = f"Art. {chunk['article']}"
        if chunk.get("old_article"):
            ref += f" (formerly {chunk['old_article']})"
        heading = chunk["title"]
        part = re.search(r"\((part \d+ of \d+)\)", chunk["label"])
        if part:  # long articles are split; tell the parts apart
            heading += f" · {part.group(1)}"
    else:
        source = "DOLE Handbook"
        ref = chunk["section"]
        heading = chunk["label"].split(f"{chunk['section']}, ", 1)[1] if (
            f"{chunk['section']}, " in chunk["label"]) else ""
        if not heading:  # fall back to the first subheading inside the passage
            sub = re.search(r"^[A-Z]\. [A-Z].{3,110}$", chunk["text"], re.M)
            heading = sub.group(0) if sub else ""
    if chunk["kind"] == "notes":
        heading = f"Footnotes{': ' + heading if heading else ''}"
    return {
        "n": n,
        "label": passage_header(chunk),
        "source": source,
        "ref": ref,
        "heading": heading,
        "pages": chunk["pages"],
        "doc": chunk["doc"],
        "kind": chunk["kind"],
        "text": chunk["text"],
        "score": round(chunk["score"], 3),
        "cited": n in cited,
        "url": f"{SOURCE_PDFS[chunk['doc']]}#page={chunk['pdf_page']}",
    }


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    if req.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(400, f"Unknown provider. Use one of {SUPPORTED_PROVIDERS}.")
    client = request.headers.get("x-forwarded-for", request.client.host).split(",")[0]
    if _rate_limited(client):
        raise HTTPException(429, "Too many questions at once. Please wait a minute.")
    start = time.perf_counter()
    providers = [req.provider] + [p for p in FALLBACK_ORDER if p != req.provider]
    for provider in [p for p in providers if is_configured(p)]:
        try:
            result = answer(req.question.strip(), store, provider=provider)
            break
        except (ValueError, RuntimeError) as e:
            print(f"{provider} failed, trying the next provider: {e}")
    else:
        raise HTTPException(503, "The AI service is busy right now. Please try again in a minute.")
    return {
        "answer": result["answer"],
        "passages": [_passage(n, c, result["cited"])
                     for n, c in enumerate(result["passages"], start=1)],
        "provider": None if result.get("declined") else LABELS[provider],
        "declined": bool(result.get("declined")),
        "seconds": round(time.perf_counter() - start, 1),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "passages": len(store), "embedding_model": store.model_name}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
