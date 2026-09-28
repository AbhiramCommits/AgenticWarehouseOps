"""Embedding providers.

The sentence-transformers model is loaded lazily so that importing this module
(and the rest of the agent layer) never triggers a model download. CI and
offline dev use the deterministic :class:`FakeEmbedder`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that turns texts into fixed-size float vectors."""

    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an ``(n, dimension)`` float array."""


class SentenceTransformerEmbedder:
    """Embeds text with ``all-MiniLM-L6-v2`` (384 dims); loads lazily."""

    dimension = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Configure the model; nothing is downloaded until :meth:`embed`.

        Args:
            model_name: Hugging Face sentence-transformers model id.
        """
        self._model_name = model_name
        self._model: Any = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts``, downloading the model on first use.

        Args:
            texts: Texts to embed.

        Returns:
            An ``(n, 384)`` L2-normalised float array.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        if not texts:
            return np.zeros((0, self.dimension), dtype=float)
        output = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(output, dtype=float)


class FakeEmbedder:
    """Deterministic, offline embedder for CI and dev.

    Vectors are character-trigram hash bags projected into ``dimension``
    buckets and L2-normalised: identical texts embed identically, and
    lexically similar texts (shared trigrams) stay close, which is enough for
    relevance tests and offline development.
    """

    dimension = 128

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into deterministic 128-dim trigram vectors.

        Args:
            texts: Texts to embed.

        Returns:
            An ``(n, 128)`` L2-normalised float array.
        """
        vectors = np.zeros((len(texts), self.dimension), dtype=float)
        for row, text in enumerate(texts):
            lowered = text.lower()
            grams = [lowered[i : i + 3] for i in range(max(1, len(lowered) - 2))]
            for gram in grams:
                bucket = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16) % self.dimension
                vectors[row, bucket] += 1.0
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors
