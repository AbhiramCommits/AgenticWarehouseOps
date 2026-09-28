"""Tests for the CLI and API wrappers plus the offline eval warehouse builder."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_agent import _manifest

from agentic_warehouse_ops.agent import api as agent_api
from agentic_warehouse_ops.agent import cli as agent_cli
from agentic_warehouse_ops.agent.models import AgentAnswer
from agentic_warehouse_ops.evals.warehouse import build_eval_warehouse
from agentic_warehouse_ops.ingestion.generate_sources import generate_dataset


def test_cli_version_command() -> None:
    """`awo version` prints the package version."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(agent_cli.app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_ask_with_scripted_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """`awo ask` runs a scripted agent and prints the trace."""

    def fake_agent(question: str, run_id: str | None = None) -> AgentAnswer:
        return AgentAnswer(
            answer="Three customers.",
            tool_calls=[],
            manifest_hash="abc123",
            run_id=run_id or "cli-test",
            question_id="q-cli",
        )

    monkeypatch.setattr(agent_cli, "build_agent", lambda: fake_agent)
    from typer.testing import CliRunner

    result = CliRunner().invoke(agent_cli.app, ["ask", "How many customers?"])
    assert result.exit_code == 0
    assert "Three customers." in result.output
    assert "abc123" in result.output


def test_api_ask_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /ask returns a serialised AgentAnswer."""
    answer = AgentAnswer(
        answer="Three customers.",
        manifest_hash="abc123",
        run_id="api-test",
        question_id="q-api",
    )
    monkeypatch.setattr(agent_api, "_get_agent", lambda: lambda question: answer)
    client = TestClient(agent_api.app)
    response = client.post("/ask", json={"question": "How many customers?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Three customers."
    assert payload["manifest_hash"] == "abc123"
    assert payload["question_id"] == "q-api"
    assert response.status_code == 200


def test_build_eval_warehouse_from_seeds(tmp_path: Path) -> None:
    """The offline warehouse builder loads generated seeds without S3."""
    seeds_dir = tmp_path / "seeds"
    generate_dataset(
        out_dir=seeds_dir,
        seed=7,
        days=1,
        orders_per_day=30,
        defect_rate=0.0,
        start_date=date(2026, 3, 1),
    )
    engine = build_eval_warehouse(
        seeds_dir=seeds_dir,
        duckdb_path=str(tmp_path / "w.duckdb"),
        manifest=_manifest(),
    )
    customers = engine.execute("SELECT COUNT(*) AS n FROM raw.customers")
    assert int(customers.iloc[0]["n"]) > 0
    embedded = engine.execute("SELECT COUNT(*) AS n FROM agent.embeddings")
    assert int(embedded.iloc[0]["n"]) > 0
