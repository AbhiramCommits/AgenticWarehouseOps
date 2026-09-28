"""Parse dbt artifacts and emit per-run lineage artifacts.

Consumes ``target/manifest.json`` (and optionally ``target/catalog.json``) and
emits ``artifacts/lineage/<run_id>.json`` describing every model's columns
with their governance classification, upstream refs/sources, materialization
type, and the manifest hash — plus a Mermaid lineage graph alongside it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_RESOURCE = "model"
SOURCE_RESOURCE = "source"


def compute_manifest_hash(manifest: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash of the dbt manifest contents.

    Args:
        manifest: Parsed ``manifest.json`` contents.

    Returns:
        Hex digest of the canonical JSON encoding.
    """
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()


def validate_meta_completeness(
    manifest: dict[str, Any], package_name: str = "warehouse"
) -> list[str]:
    """Return violations for any model column missing governance meta keys.

    Enforcement is the point: a non-empty result means the dbt project has a
    column without ``pii``, ``classification``, or ``owner`` metadata.

    Args:
        manifest: Parsed ``manifest.json`` contents.
        package_name: Only models in this package are checked.

    Returns:
        Violations as ``"<model>.<column>: missing [...]"`` strings; empty
        when every column is fully tagged.
    """
    violations: list[str] = []
    for node in manifest.get("nodes", {}).values():
        if node.get("resource_type") != MODEL_RESOURCE or node.get("package_name") != package_name:
            continue
        for col_name, col in (node.get("columns") or {}).items():
            meta = col.get("meta") or {}
            missing = [key for key in ("pii", "classification", "owner") if key not in meta]
            if missing:
                violations.append(f"{node['name']}.{col_name}: missing {sorted(missing)}")
    return violations


def _short_name(unique_id: str) -> str:
    """Reduce a dbt unique id to a readable node label for diagrams."""
    parts = unique_id.split(".")
    if unique_id.startswith("source."):
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


def _column_meta(col: dict[str, Any]) -> dict[str, Any]:
    return col.get("meta") or {}


def build_lineage(
    manifest_path: str | Path,
    catalog_path: str | Path | None = None,
    run_id: str | None = None,
    artifacts_dir: str | Path = "artifacts/lineage",
) -> dict[str, Any]:
    """Build and persist the lineage artifact for one pipeline run.

    Args:
        manifest_path: Path to dbt ``target/manifest.json``.
        catalog_path: Optional path to ``target/catalog.json``; when present,
            column data types are attached to the lineage.
        run_id: Pipeline run id used in the artifact filenames.
        artifacts_dir: Directory receiving ``<run_id>.json`` and ``<run_id>.mmd``.

    Returns:
        The lineage payload, including ``manifest_hash`` and the written
        artifact paths.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    catalog: dict[str, Any] | None = None
    if catalog_path and Path(catalog_path).exists():
        catalog = json.loads(Path(catalog_path).read_text())
    manifest_hash = compute_manifest_hash(manifest)

    models: list[dict[str, Any]] = []
    edges: list[tuple[str, str]] = []
    for unique_id, node in manifest.get("nodes", {}).items():
        if node.get("resource_type") != MODEL_RESOURCE or node.get("package_name") != "warehouse":
            continue
        catalog_columns = (
            (catalog or {}).get("nodes", {}).get(unique_id, {}).get("columns", {})
            if catalog
            else {}
        )
        columns: list[dict[str, Any]] = []
        for col_name, col in (node.get("columns") or {}).items():
            meta = _column_meta(col)
            columns.append(
                {
                    "name": col_name,
                    "classification": meta.get("classification", "public"),
                    "pii": bool(meta.get("pii", False)),
                    "data_type": (catalog_columns.get(col_name) or {}).get("type"),
                }
            )
        upstream_refs = [
            upstream
            for upstream in node.get("depends_on", {}).get("nodes", [])
            if upstream.startswith(f"{MODEL_RESOURCE}.")
        ]
        upstream_sources = [
            upstream
            for upstream in node.get("depends_on", {}).get("nodes", [])
            if upstream.startswith(f"{SOURCE_RESOURCE}.")
        ]
        models.append(
            {
                "name": node["name"],
                "unique_id": unique_id,
                "schema": node.get("schema"),
                "materialized": (node.get("config") or {}).get("materialized"),
                "columns": columns,
                "upstream_refs": upstream_refs,
                "upstream_sources": upstream_sources,
            }
        )
        for upstream in upstream_refs + upstream_sources:
            edges.append((_short_name(upstream), node["name"]))

    sources: list[dict[str, Any]] = []
    for unique_id, source in manifest.get("sources", {}).items():
        sources.append(
            {
                "name": source.get("name"),
                "source_name": source.get("source_name"),
                "schema": source.get("schema"),
                "identifier": source.get("identifier"),
                "unique_id": unique_id,
            }
        )
        edges.append((_short_name(unique_id), _short_name(unique_id)))

    payload: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest_hash": manifest_hash,
        "models": models,
        "sources": sources,
    }

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_id}.json"
    mmd_path = out_dir / f"{run_id}.mmd"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    lines = ["flowchart TD"]
    for upstream, downstream in edges:
        lines.append(
            f"    {upstream.replace('.', '_')}[{upstream!r}] --> {downstream.replace('.', '_')}[{downstream!r}]"
        )
    mmd_path.write_text("\n".join(lines) + "\n")

    payload["artifacts"] = {"lineage": str(json_path), "mermaid": str(mmd_path)}
    return payload
