"""Embedding index: one :class:`VectorStore` interface plus batched, resumable
embedding jobs.

Vectors live in the warehouse (``agent.embeddings`` in DuckDB); search loads
the filtered matrix into numpy for cosine ranking, so nothing depends on
network access or extension downloads. Embedding jobs are keyed on content
hash: unchanged rows are never re-embedded.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from agentic_warehouse_ops.agent.embedder import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from agentic_warehouse_ops.common.warehouse import WarehouseEngine

EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS agent.embeddings (
    id VARCHAR PRIMARY KEY,
    content TEXT,
    content_hash VARCHAR,
    embedding FLOAT[],
    metadata JSON
)
"""


def content_hash(text: str) -> str:
    """Return the SHA-256 content hash used for embed-skip decisions."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_embedder() -> Embedder:
    """Return the configured embedder.

    Defaults to the real ``all-MiniLM-L6-v2`` model; set ``AGENT_EMBEDDER=fake``
    for deterministic offline embeddings (CI, Airflow without a model cache).
    """
    kind = os.environ.get("AGENT_EMBEDDER", "sentence-transformer").strip().lower()
    if kind == "fake":
        return FakeEmbedder()
    return SentenceTransformerEmbedder()


class VectorStore(ABC):
    """Interface every vector index implements."""

    @abstractmethod
    def upsert(self, ids: list[str], texts: list[str], metadata: list[dict[str, Any]]) -> None:
        """Insert or replace one record per ``(id, text, metadata)`` triple.

        Args:
            ids: Unique record ids; an existing id is overwritten.
            texts: Raw text content for each record.
            metadata: JSON-serialisable dict per record.
        """

    @abstractmethod
    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the ``k`` nearest records for ``query``.

        Args:
            query: Free-text query.
            k: Maximum number of results.
            filters: Optional ``metadata_key -> value`` equality filters.

        Returns:
            Records ``{"id", "content", "metadata", "score"}`` ordered by
            descending cosine similarity.
        """


class DuckDBVectorStore(VectorStore):
    """Vector store backed by ``agent.embeddings`` in the warehouse."""

    def __init__(self, engine: WarehouseEngine, embedder: Embedder) -> None:
        """Create the backing table if needed.

        Args:
            engine: Warehouse engine holding the vectors.
            embedder: Embedder used for both ingest and queries.
        """
        self._engine = engine
        self._embedder = embedder
        engine.execute("CREATE SCHEMA IF NOT EXISTS agent")
        engine.execute(EMBEDDINGS_DDL)

    def upsert(self, ids: list[str], texts: list[str], metadata: list[dict[str, Any]]) -> None:
        if not ids:
            return
        vectors = self._embedder.embed(texts)
        self._engine.begin()
        try:
            for record_id in ids:
                self._engine.execute("DELETE FROM agent.embeddings WHERE id = ?", [record_id])
            rows = [
                (
                    record_id,
                    text,
                    content_hash(text),
                    vectors[index].tolist(),
                    json.dumps(meta, default=str),
                )
                for index, (record_id, text, meta) in enumerate(
                    zip(ids, texts, metadata, strict=True)
                )
            ]
            self._engine.executemany(
                "INSERT INTO agent.embeddings (id, content, content_hash, embedding, metadata)"
                " VALUES (?, ?, ?, ?, ?::JSON)",
                rows,
            )
            self._engine.commit()
        except Exception:
            self._engine.rollback()
            raise

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        for key, value in (filters or {}).items():
            where.append(f"json_extract_string(metadata, '$.\"{key}\"') = ?")
            params.append(str(value))
        sql = "SELECT id, content, embedding, metadata FROM agent.embeddings"
        if where:
            sql += " WHERE " + " AND ".join(where)
        # Fixed row order + stable sort so equal scores (templated tickets often
        # embed identically) always rank the same way: ties go to the lowest id.
        sql += " ORDER BY id"
        frame = self._engine.execute(sql, params)
        if frame.empty:
            return []
        query_vector = self._embedder.embed([query])[0]
        matrix = np.vstack([np.asarray(values, dtype=float) for values in frame["embedding"]])
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
        scores = (matrix @ query_vector) / (norms + 1e-9)
        top = np.argsort(-scores, kind="stable")[:k]
        results: list[dict[str, Any]] = []
        for index in top:
            results.append(
                {
                    "id": str(frame.iloc[index]["id"]),
                    "content": str(frame.iloc[index]["content"]),
                    "metadata": json.loads(str(frame.iloc[index]["metadata"])),
                    "score": float(scores[index]),
                }
            )
        return results

    def content_hashes(self) -> dict[str, str]:
        """Return the stored ``id -> content_hash`` mapping for skip decisions."""
        frame = self._engine.execute("SELECT id, content_hash FROM agent.embeddings")
        return {
            str(record_id): str(hash_value)
            for record_id, hash_value in zip(frame["id"], frame["content_hash"], strict=True)
        }


@dataclass(frozen=True)
class EmbedJobResult:
    """Outcome of a resumable embedding job."""

    kind: str
    total: int
    embedded: int
    skipped: int


def embed_support_tickets(
    engine: WarehouseEngine,
    embedder: Embedder | None = None,
    batch_size: int = 64,
) -> EmbedJobResult:
    """Embed support-ticket body text into the vector index, resumably.

    Rows whose content hash already exists are skipped, so rerunning this job
    only embeds new or changed tickets.

    Args:
        engine: Warehouse engine with a populated ``raw.support_tickets``.
        embedder: Optional embedder; defaults to :func:`get_embedder`.
        batch_size: Embedding batch size.

    Returns:
        Counts of total, newly embedded, and skipped rows.
    """
    resolved = embedder or get_embedder()
    store = DuckDBVectorStore(engine, resolved)
    frame = engine.execute(
        "SELECT ticket_id, body_text, created_ts, channel, subject FROM raw.support_tickets"
    )
    existing = store.content_hashes()
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for record in frame.to_dict("records"):
        ticket_id = str(record["ticket_id"])
        text = str(record["body_text"])
        if existing.get(ticket_id) == content_hash(text):
            continue
        created = record["created_ts"]
        rows.append(
            (
                ticket_id,
                text,
                {
                    "kind": "ticket",
                    "ticket_id": ticket_id,
                    "channel": str(record["channel"]),
                    "created_ts": created.isoformat() if created is not None else None,
                    "subject": str(record["subject"]),
                },
            )
        )
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        store.upsert(
            [row[0] for row in batch],
            [row[1] for row in batch],
            [row[2] for row in batch],
        )
    return EmbedJobResult(
        kind="ticket",
        total=len(frame),
        embedded=len(rows),
        skipped=len(frame) - len(rows),
    )


def embed_catalog(
    engine: WarehouseEngine,
    manifest: dict[str, Any],
    embedder: Embedder | None = None,
    batch_size: int = 64,
) -> EmbedJobResult:
    """Embed model and column descriptions from a dbt manifest.

    Args:
        engine: Warehouse engine receiving the vectors.
        manifest: Parsed dbt ``manifest.json``.
        embedder: Optional embedder; defaults to :func:`get_embedder`.
        batch_size: Embedding batch size.

    Returns:
        Counts of total, newly embedded, and skipped rows.
    """
    resolved = embedder or get_embedder()
    store = DuckDBVectorStore(engine, resolved)
    existing = store.content_hashes()
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for node in manifest.get("nodes", {}).values():
        if node.get("resource_type") != "model" or node.get("package_name") != "warehouse":
            continue
        name = node["name"]
        model_text = node.get("description") or name
        if existing.get(name) != content_hash(model_text):
            rows.append(
                (
                    name,
                    model_text,
                    {"kind": "catalog", "model": name, "column": None},
                )
            )
        for col_name, column in (node.get("columns") or {}).items():
            meta = column.get("meta") or {}
            text = column.get("description") or f"{name}.{col_name}"
            record_id = f"{name}.{col_name}"
            if existing.get(record_id) == content_hash(text):
                continue
            rows.append(
                (
                    record_id,
                    text,
                    {
                        "kind": "catalog",
                        "model": name,
                        "column": col_name,
                        "classification": meta.get("classification", "public"),
                        "pii": bool(meta.get("pii", False)),
                    },
                )
            )
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        store.upsert(
            [row[0] for row in batch],
            [row[1] for row in batch],
            [row[2] for row in batch],
        )
    return EmbedJobResult(kind="catalog", total=len(rows), embedded=len(rows), skipped=0)


__all__ = [
    "DuckDBVectorStore",
    "EmbedJobResult",
    "VectorStore",
    "content_hash",
    "embed_catalog",
    "embed_support_tickets",
    "get_embedder",
]
