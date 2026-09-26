"""Calibrate the relevance threshold (MIN_RELEVANCE in labor_rag/generate.py).

Retrieval always returns its top k, even for "tell me a joke", so the app
needs a cut-off below which it declines instead of calling the LLM. This
compares the best-passage similarity of questions it must answer (the eval
set plus casual phrasings) with questions it should decline.

    python eval/scope_check.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labor_rag.embed import MODEL_NAME, embed_query  # noqa: E402
from labor_rag.generate import MIN_RELEVANCE  # noqa: E402
from labor_rag.store import VectorStore  # noqa: E402


def main():
    store = VectorStore.load(ROOT / "index", expected_model=MODEL_NAME)
    scope = json.loads((ROOT / "eval" / "scope_questions.json").read_text(encoding="utf-8"))
    evalset = [q["q"] for q in json.loads((ROOT / "eval" / "questions.json").read_text(
        encoding="utf-8")) if q["sources"]]

    def best(q: str) -> float:
        return store.search(embed_query(q), 1)[0]["score"]

    on_topic = [(best(q), q) for q in evalset + scope["answer"]]
    off_topic = [(best(q), q) for q in scope["decline"]]
    print(f"on-topic  ({len(on_topic)}): lowest {min(on_topic)[0]:.3f}  ({min(on_topic)[1]!r})")
    print(f"off-topic ({len(off_topic)}): highest {max(off_topic)[0]:.3f}  ({max(off_topic)[1]!r})\n")
    for t in sorted({0.58, 0.60, 0.62, 0.64, 0.66, MIN_RELEVANCE}):
        wrong = sum(s < t for s, _ in on_topic)
        stopped = sum(s < t for s, _ in off_topic)
        mark = "  <- MIN_RELEVANCE" if t == MIN_RELEVANCE else ""
        print(f"threshold {t:.2f}: on-topic declined {wrong}/{len(on_topic)}, "
              f"off-topic stopped before the LLM {stopped}/{len(off_topic)}{mark}")


if __name__ == "__main__":
    main()
