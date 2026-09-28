"""Reproducibility harness: manifest snapshots and question replay.

Every transform run snapshots its dbt manifest to
``artifacts/manifests/<manifest_hash>.json`` and registers it in
``meta.manifest_registry``. A recorded agent question can then be replayed
against its pinned manifest version, producing an ``IDENTICAL`` /
``DRIFTED`` / ``STALE_SCHEMA`` verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_warehouse_ops.agent.audit import get_question, get_tool_calls
from agentic_warehouse_ops.agent.tools import FilterClause, OrderByClause, compile_query
from agentic_warehouse_ops.common.warehouse import WarehouseEngine
from agentic_warehouse_ops.governance.catalog import compute_manifest_hash, load_manifest

MANIFEST_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS meta.manifest_registry (
    manifest_hash VARCHAR PRIMARY KEY,
    git_sha VARCHAR,
    created_at TIMESTAMP,
    dbt_version VARCHAR,
    model_count BIGINT
)
"""


def _utcnow() -> datetime:
    """Return the current UTC time as a naive datetime."""
    return datetime.now(UTC).replace(tzinfo=None)


def _git_sha() -> str | None:
    """Resolve the current git revision from env or the local repo."""
    from_env = os.environ.get("GIT_SHA")
    if from_env:
        return from_env
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001 - git may be unavailable (containers)
        return None


def snapshot_manifest(
    manifest_path: str | Path,
    engine: WarehouseEngine,
    git_sha: str | None = None,
    artifacts_dir: str | Path = "artifacts/manifests",
) -> str:
    """Snapshot a dbt manifest and register it, returning its hash.

    Args:
        manifest_path: Path to ``target/manifest.json``.
        engine: Warehouse engine receiving the registry row.
        git_sha: Optional revision; defaults to ``GIT_SHA`` or ``git rev-parse``.
        artifacts_dir: Directory receiving ``<manifest_hash>.json``.

    Returns:
        The manifest hash (SHA-256).
    """
    manifest = load_manifest(manifest_path)
    digest = compute_manifest_hash(manifest)
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{digest}.json").write_text(json.dumps(manifest, indent=2, default=str))
    engine.execute("CREATE SCHEMA IF NOT EXISTS meta")
    engine.execute(MANIFEST_REGISTRY_DDL)
    model_count = sum(
        1
        for node in manifest.get("nodes", {}).values()
        if node.get("resource_type") == "model" and node.get("package_name") == "warehouse"
    )
    engine.execute("DELETE FROM meta.manifest_registry WHERE manifest_hash = ?", [digest])
    engine.execute(
        "INSERT INTO meta.manifest_registry"
        " (manifest_hash, git_sha, created_at, dbt_version, model_count)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            digest,
            git_sha or _git_sha(),
            _utcnow(),
            (manifest.get("metadata") or {}).get("dbt_version"),
            model_count,
        ],
    )
    return digest


def _mart_node(manifest: dict[str, Any], mart: str) -> dict[str, Any] | None:
    for node in manifest.get("nodes", {}).values():
        if node.get("resource_type") == "model" and node.get("name") == mart:
            return node
    return None


def _replay_query_call(
    engine: WarehouseEngine,
    pinned_manifest: dict[str, Any],
    current_manifest: dict[str, Any],
    call: dict[str, Any],
) -> dict[str, Any]:
    """Re-execute one recorded query_mart call against both manifest versions."""
    args = json.loads(call["tool_args"])
    mart = args["mart"]
    comparison: dict[str, Any] = {
        "tool": call["tool_name"],
        "mart": mart,
        "recorded_rows": int(call["rows_returned"]),
        "pinned_rows": None,
        "current_rows": None,
        "note": None,
    }
    pinned_node = _mart_node(pinned_manifest, mart)
    if pinned_node is None:
        comparison["note"] = f"pinned manifest lacks mart '{mart}'"
        return comparison
    current_node = _mart_node(current_manifest, mart)
    if current_node is None:
        comparison["note"] = f"mart '{mart}' no longer exists in the current schema"
        comparison["stale"] = True
        return comparison
    filters = [FilterClause.model_validate(item) for item in args.get("filters", [])]
    order_by = [OrderByClause.model_validate(item) for item in args.get("order_by", [])]
    try:
        pinned_sql, pinned_params = compile_query(
            pinned_node, args["select"], filters, args.get("group_by", []), order_by, args["limit"]
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a stale comparison
        comparison["note"] = f"pinned compile failed: {exc}"
        return comparison
    comparison["pinned_rows"] = int(len(engine.execute(pinned_sql, pinned_params)))
    try:
        current_sql, current_params = compile_query(
            current_node, args["select"], filters, args.get("group_by", []), order_by, args["limit"]
        )
        comparison["current_rows"] = int(len(engine.execute(current_sql, current_params)))
    except Exception as exc:  # noqa: BLE001 - stale schema manifests here
        comparison["note"] = f"current schema no longer supports the call: {exc}"
        comparison["stale"] = True
    return comparison


def replay_question(
    engine: WarehouseEngine,
    question_id: str,
    artifacts_dir: str | Path = "artifacts/manifests",
    current_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replay a recorded question against its pinned manifest version.

    Args:
        engine: Warehouse engine holding the audit rows.
        question_id: Question id recorded by the agent.
        artifacts_dir: Directory holding ``<manifest_hash>.json`` snapshots.
        current_manifest_path: Current manifest to compare against; defaults
            to ``dbt/target/manifest.json`` when present, else the newest
            snapshot in ``artifacts_dir``.

    Returns:
        ``{"question_id", "manifest_hash", "verdict", "comparisons"}`` where
        verdict is ``IDENTICAL``, ``DRIFTED``, or ``STALE_SCHEMA``.

    Raises:
        ValueError: If the question id is unknown.
    """
    question = get_question(engine, question_id)
    if question is None:
        raise ValueError(f"unknown question_id '{question_id}'")
    pinned_hash = question["manifest_hash"]
    artifacts = Path(artifacts_dir)
    pinned_path = artifacts / f"{pinned_hash}.json"
    if not pinned_path.exists():
        return {
            "question_id": question_id,
            "manifest_hash": pinned_hash,
            "verdict": "STALE_SCHEMA",
            "reason": f"pinned manifest snapshot {pinned_hash} not found",
            "comparisons": [],
        }
    pinned_manifest = load_manifest(pinned_path)
    if current_manifest_path and Path(current_manifest_path).exists():
        current_manifest = load_manifest(current_manifest_path)
    else:
        live = Path("dbt/target/manifest.json")
        if live.exists():
            current_manifest = load_manifest(live)
        else:
            snapshots = sorted(artifacts.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if not snapshots:
                raise ValueError(f"no current manifest available in {artifacts_dir}")
            current_manifest = load_manifest(snapshots[-1])

    calls = get_tool_calls(engine, question_id)
    comparisons: list[dict[str, Any]] = []
    stale = False
    for call in calls.to_dict("records"):
        call_row: dict[str, Any] = {str(key): value for key, value in call.items()}
        if call_row["tool_name"] != "query_mart":
            continue
        comparison = _replay_query_call(engine, pinned_manifest, current_manifest, call_row)
        comparisons.append(comparison)
        if comparison.get("stale"):
            stale = True
    if stale:
        verdict = "STALE_SCHEMA"
    else:
        drifted = any(
            item["recorded_rows"] != item["pinned_rows"]
            or item["pinned_rows"] != item["current_rows"]
            for item in comparisons
        )
        verdict = "DRIFTED" if drifted else "IDENTICAL"
    return {
        "question_id": question_id,
        "manifest_hash": pinned_hash,
        "verdict": verdict,
        "comparisons": comparisons,
    }
