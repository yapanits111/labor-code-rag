"""Cosine similarity, written out by hand.

cos(a, b) = (a . b) / (|a| * |b|)

It is the cosine of the angle between two vectors. Dividing by the lengths
throws magnitude away, so only *direction* is compared — and in an embedding
space, direction is what encodes meaning.
    1.0  -> same direction (same meaning)
    0.0  -> perpendicular (unrelated)
   -1.0  -> opposite direction
"""
import math

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-Python version: no NumPy, so every step is visible."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0  # a zero vector has no direction
    return dot / (norm_a * norm_b)


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Same formula, vectorised: score one query vector against every row of
    `matrix` (shape n x d) in a single pass. Returns n scores.

    This is what retrieval actually runs — a brute-force scan over all chunks.
    Fine for thousands of vectors; at millions, vector databases switch to
    approximate nearest neighbour indexes (HNSW, IVF) instead of scanning all.
    """
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * query_norm
    denom[denom == 0] = 1.0  # avoid divide-by-zero; those rows score 0
    return (matrix @ query) / denom
