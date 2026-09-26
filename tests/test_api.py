"""API tests. The LLM call is replaced with a stub, and the startup warm-up
isn't triggered (TestClient is not used as a context manager), so these run
offline."""
import pytest
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def fake_answer(monkeypatch):
    main._requests.clear()
    chunk = next(c for c in main.store.chunks if c["id"] == "art-94")

    def fake(question, store, provider):
        return {"answer": "Twice your regular rate [1].",
                "passages": [{**chunk, "score": 0.81}], "cited": [1]}

    monkeypatch.setattr(main, "answer", fake)
    monkeypatch.setattr(main, "is_configured", lambda provider: True)  # CI has no API keys


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["passages"] == len(main.store)


def test_index_page():
    r = client.get("/")
    assert r.status_code == 200 and "Know your" in r.text


def test_ask_returns_answer_with_linked_sources():
    r = client.post("/api/ask", json={"question": "What is holiday pay?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Twice your regular rate [1]."
    source = body["passages"][0]
    assert source["cited"] and source["label"].startswith("Labor Code Art. 94")
    assert source["ref"] == "Art. 94" and source["heading"] == "Right to Holiday Pay"
    art94 = next(c for c in main.store.chunks if c["id"] == "art-94")
    assert source["url"] == f"{main.SOURCE_PDFS['labor_code']}#page={art94['pdf_page']}"


def test_question_validation():
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/ask", json={"question": "a" * 501}).status_code == 422
    assert client.post("/api/ask", json={"question": "hello?", "provider": "nope"}).status_code == 400


def test_rate_limit():
    limit, _ = main.RATE_LIMIT
    codes = [client.post("/api/ask", json={"question": "What is holiday pay?"}).status_code
             for _ in range(limit + 1)]
    assert codes[:limit] == [200] * limit and codes[-1] == 429


def test_falls_back_to_next_provider_when_one_is_rate_limited(monkeypatch):
    chunk = next(c for c in main.store.chunks if c["id"] == "art-94")
    tried = []

    def flaky(question, store, provider):
        tried.append(provider)
        if provider == "groq":
            raise RuntimeError("Groq's free-tier limit is used up right now.")
        return {"answer": "Twice [1].", "passages": [{**chunk, "score": 0.8}], "cited": [1]}

    monkeypatch.setattr(main, "answer", flaky)
    r = client.post("/api/ask", json={"question": "What is holiday pay?"})
    assert r.status_code == 200
    assert tried[0] == "groq" and r.json()["provider"] != "Groq"


def test_all_providers_failing_gives_friendly_503(monkeypatch):
    def down(question, store, provider):
        raise RuntimeError("limit")

    monkeypatch.setattr(main, "answer", down)
    r = client.post("/api/ask", json={"question": "What is holiday pay?"})
    assert r.status_code == 503 and "try again" in r.json()["detail"]
