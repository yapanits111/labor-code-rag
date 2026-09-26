"""Step 2 (and 4): turn text into vectors.

We use a small local model (BAAI/bge-small-en-v1.5, 384 dimensions) through
fastembed, which runs on ONNX — no PyTorch, no API key, no cost. The model is
downloaded (~130 MB) on first use and cached.

Rule that matters: documents and queries MUST be embedded by the same model.
Vectors from different models live in different spaces, and comparing them is
meaningless. The store records the model name and refuses a mismatch.
"""
import os

import numpy as np

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

_model = None


def _get_model():
    """Load the model once, lazily, on first call."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        # MODEL_CACHE_DIR lets the Docker image bake the model in at build time
        _model = TextEmbedding(model_name=MODEL_NAME,
                               cache_dir=os.getenv("MODEL_CACHE_DIR") or None)
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed many passages at once. Returns an (n, d) float32 array."""
    vectors = _get_model().passage_embed(texts)
    return np.array(list(vectors), dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a user question. BGE models are trained with a short instruction
    prefix on queries; `query_embed` adds it for us. Returns a (d,) array."""
    return np.array(next(iter(_get_model().query_embed(text))), dtype=np.float32)


def get_embedding(text: str) -> list[float]:
    """Single-text convenience wrapper, returning a plain list of floats."""
    return embed_documents([text])[0].tolist()
