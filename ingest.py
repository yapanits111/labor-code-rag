"""Build the search index: data/corpus.json -> chunks -> vectors -> index/.

    python ingest.py

Run again whenever the corpus or the chunking changes.
"""
import json
import time

from labor_rag.chunk import chunk_records, embedding_input
from labor_rag.embed import MODEL_NAME, embed_documents
from labor_rag.store import VectorStore


def main():
    records = json.load(open("data/corpus.json", encoding="utf-8"))
    chunks = chunk_records(records)
    kinds = {k: sum(c["kind"] == k for c in chunks) for k in ("article", "notes", "guide")}
    print(f"{len(records)} records -> {len(chunks)} chunks {kinds}")

    start = time.perf_counter()
    vectors = embed_documents([embedding_input(c) for c in chunks])
    print(f"Embedded with {MODEL_NAME}: {vectors.shape[0]} x {vectors.shape[1]} "
          f"in {time.perf_counter() - start:.1f}s")

    store = VectorStore(MODEL_NAME)
    store.add(chunks, vectors)
    store.save("index")
    print("Saved index to index/")


if __name__ == "__main__":
    main()
