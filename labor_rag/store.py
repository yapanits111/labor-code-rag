"""Step 3: the simplest possible vector store.

Per chunk we keep three things: the vector, the original text, and metadata
(article number, section, page). Vectors live in one NumPy matrix (row i = chunk i);
text + metadata live in a parallel list. On disk that is two files:

    index/vectors.npy   the (n, d) matrix
    index/chunks.json   chunk records + which embedding model produced them

~700 chunks x 384 dimensions is one small matrix: brute-force search takes
about a millisecond, so no vector database is needed at this size.
"""
import json
from pathlib import Path

import numpy as np

from .similarity import cosine_similarity_matrix


class VectorStore:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.chunks: list[dict] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[dict], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("need exactly one vector per chunk")
        self.chunks.extend(chunks)
        self.vectors = (vectors if self.vectors.size == 0
                        else np.vstack([self.vectors, vectors]))

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[dict]:
        """Nearest-neighbour search: score every chunk, return the top k,
        each with its similarity `score` attached."""
        if len(self) == 0:
            return []
        scores = cosine_similarity_matrix(query_vector, self.vectors)
        top = np.argsort(scores)[::-1][:k]  # highest score first
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        meta = {"model": self.model_name, "chunks": self.chunks}
        (folder / "chunks.json").write_text(json.dumps(meta, indent=1),
                                            encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path, expected_model: str | None = None) -> "VectorStore":
        folder = Path(folder)
        if not (folder / "chunks.json").exists():
            raise FileNotFoundError(
                f"No index in '{folder}'. Run `python ingest.py` first.")
        meta = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        if expected_model and meta["model"] != expected_model:
            raise ValueError(
                f"Index was built with '{meta['model']}' but queries would use "
                f"'{expected_model}'. Re-run `python ingest.py`.")
        store = cls(meta["model"])
        store.chunks = meta["chunks"]
        store.vectors = np.load(folder / "vectors.npy")
        return store
