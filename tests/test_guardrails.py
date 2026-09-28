"""Tests for the guardrails, audit log, and replay harness."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_agent import _manifest, seed_tables

from agentic_warehouse_ops.agent.audit import (
    get_question,
    get_tool_calls,
    list_audit,
    migrate_agent_audit,
)
from agentic_warehouse_ops.agent.embedder import FakeEmbedder
from agentic_warehouse_ops.agent.graph import FakeLLM, run_agent
from agentic_warehouse_ops.agent.guardrails import (
    GuardrailConfig,
    GuardrailSession,
    GuardrailViolation,
    assert_read_only_select,
    guard_tools,
)
from agentic_warehouse_ops.agent.index import DuckDBVectorStore
from agentic_warehouse_ops.agent.tools import ToolContext, build_tools
from agentic_warehouse_ops.common.reproducibility import replay_question, snapshot_manifest
from agentic_warehouse_ops.common.warehouse import DuckDBEngine, WarehouseEngine


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[WarehouseEngine]:
    yield DuckDBEngine(str(tmp_path / "guard.duckdb"))


@pytest.fixture
def manifest() -> dict[str, Any]:
    return _manifest()


@pytest.fixture
def seeded_engine(engine: WarehouseEngine) -> WarehouseEngine:
    return seed_tables(engine)


def _session(
    engine: WarehouseEngine | None,
    manifest: dict[str, Any],
    config: GuardrailConfig | None = None,
) -> GuardrailSession:
    session = GuardrailSession(
        manifest=manifest,
        config=config or GuardrailConfig(max_questions_per_minute=1000),
        engine=engine,
        manifest_hash="test-hash",
        question_id="q-test",
        run_id="run-test",
        llm_model="fake-scripted",
    )
    session.start_question()
    return session


def _guarded_tools(
    engine: WarehouseEngine, manifest: dict[str, Any], config: GuardrailConfig | None = None
) -> tuple[dict[str, Any], GuardrailSession]:
    session = _session(engine, manifest, config)
    tools = build_tools(
        ToolContext(
            engine=engine,
            manifest=manifest,
            vector_store=DuckDBVectorStore(engine, FakeEmbedder()),
            embedder=FakeEmbedder(),
        )
    )
    return guard_tools(tools, session), session


def _agent_tools(engine: WarehouseEngine, manifest: dict[str, Any]) -> dict[str, Any]:
    return build_tools(
        ToolContext(
            engine=engine,
            manifest=manifest,
            vector_store=DuckDBVectorStore(engine, FakeEmbedder()),
            embedder=FakeEmbedder(),
        )
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.write_text(json.dumps(manifest))
    return path


def test_pii_denied_at_args_layer(seeded_engine: WarehouseEngine, manifest: dict[str, Any]) -> None:
    """A PII column request is refused with a structured violation."""
    tools, session = _guarded_tools(seeded_engine, manifest)
    with pytest.raises(GuardrailViolation) as caught:
        tools["query_mart"].invoke({"mart": "dim_customer", "select": ["email"], "limit": 5})
    violation = caught.value
    assert violation.rule == "pii"
    assert violation.column == "email"
    assert violation.classification == "restricted"
    assert violation.fatal is False
    rows = list_audit(seeded_engine, question_id="q-test")
    assert len(rows) == 1
    assert rows.iloc[0]["guardrail_verdict"].startswith("denied: pii")
    assert session._tool_calls == 1


def test_pii_denied_at_result_layer(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """An unexpected PII column in results is caught before it reaches the LLM."""
    session = _session(seeded_engine, manifest)
    with pytest.raises(GuardrailViolation) as caught:
        session._post_check(
            "query_mart",
            {"mart": "dim_customer"},
            {"records": [{"country": "US", "email": "leak@example.com"}], "row_count": 1},
        )
    violation = caught.value
    assert violation.rule == "pii"
    assert violation.column == "email"


def test_row_cap_injected_and_byte_ceiling_truncates(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """The hard LIMIT is injected and oversized results are truncated."""
    tools, _ = _guarded_tools(
        seeded_engine, manifest, GuardrailConfig(max_rows=2, max_result_bytes=60)
    )
    result = tools["query_mart"].invoke(
        {"mart": "dim_customer", "select": ["country"], "limit": 500}
    )
    assert "LIMIT 2" in result["sql"]
    assert result["row_count"] == 2
    assert len(result["records"]) <= 2


def test_cost_cap_aborts_with_partial_answer(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """Exceeding the cost budget aborts the run with a partial answer + reason."""
    tools = _agent_tools(seeded_engine, manifest)
    llm = FakeLLM(
        script=[
            json.dumps({"action": "tool", "tool": "lookup_schema", "args": {}, "reason": "cheap"}),
            json.dumps(
                {
                    "action": "tool",
                    "tool": "query_mart",
                    "args": {"mart": "dim_customer", "select": ["country"], "limit": 50},
                    "reason": "expensive",
                }
            ),
            "Partial answer with the reason.",
        ]
    )
    config = GuardrailConfig(max_bytes_scanned=10_000, max_questions_per_minute=1000)
    answer = run_agent(
        "Describe customers.",
        tools=tools,
        llm=llm,
        manifest=manifest,
        engine=seeded_engine,
        config=config,
        run_id="cost-test",
    )
    assert answer.status == "aborted"
    assert answer.abort_reason is not None
    assert "cost budget" in answer.abort_reason
    assert answer.answer == "Partial answer with the reason."


def test_sqlglot_read_only_enforcement() -> None:
    """Statement separators, DDL, and DML are rejected; one SELECT passes."""
    assert_read_only_select("SELECT 1")
    assert_read_only_select("SELECT country FROM raw_marts.dim_customer WHERE x = ?")
    for bad in [
        "SELECT 1; DROP TABLE raw_marts.dim_customer;",
        "CREATE TABLE evil (x INT)",
        "UPDATE raw_marts.dim_customer SET country = 'XX'",
        "DELETE FROM raw_marts.dim_customer",
        "WITH cte AS (SELECT 1 AS x) SELECT * FROM cte; INSERT INTO raw_marts.dim_customer VALUES (9)",
        "SELECT 1 UNION SELECT 2; SELECT 3",
    ]:
        with pytest.raises(GuardrailViolation) as caught:
            assert_read_only_select(bad)
        assert caught.value.rule == "read_only"


def test_audit_rows_written_on_success_and_failure(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """Audit rows exist for a successful run and a failing run."""
    migrate_agent_audit(seeded_engine)
    tools = _agent_tools(seeded_engine, manifest)
    success_llm = FakeLLM(
        script=[
            json.dumps(
                {
                    "action": "tool",
                    "tool": "query_mart",
                    "args": {"mart": "dim_customer", "select": ["country"], "limit": 3},
                    "reason": "count customers",
                }
            ),
            json.dumps({"action": "finish", "reason": "done"}),
            "Three customers.",
        ]
    )
    answer = run_agent(
        "How many customers?",
        tools=tools,
        llm=success_llm,
        manifest=manifest,
        engine=seeded_engine,
        config=GuardrailConfig(max_questions_per_minute=1000),
    )
    assert answer.status == "success"
    question_row = get_question(seeded_engine, answer.question_id)
    assert question_row is not None
    assert question_row["status"] == "success"
    calls = get_tool_calls(seeded_engine, answer.question_id)
    assert len(calls) == 1
    assert calls.iloc[0]["guardrail_verdict"] == "allowed"
    assert calls.iloc[0]["rows_returned"] == 3

    # Failure path: the LLM explodes after the first tool call.
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ExplodingLLM(FakeLLM):
        def _generate(
            self, messages: list[Any], stop: list[str] | None = None, **kwargs: Any
        ) -> Any:
            self._index += 1
            if self._index >= 2:
                raise RuntimeError("graph exploded")
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=self.script[0]))]
            )

    with pytest.raises(RuntimeError, match="graph exploded"):
        run_agent(
            "Explode please.",
            tools=tools,
            llm=ExplodingLLM(
                script=[
                    json.dumps(
                        {
                            "action": "tool",
                            "tool": "query_mart",
                            "args": {"mart": "dim_customer", "select": ["country"], "limit": 3},
                            "reason": "first call",
                        }
                    )
                ]
            ),
            manifest=manifest,
            engine=seeded_engine,
            config=GuardrailConfig(max_questions_per_minute=1000),
        )
    all_rows = list_audit(seeded_engine, limit=50)
    failed_rows = all_rows[all_rows["question_id"] != answer.question_id]
    assert len(failed_rows) == 1
    assert failed_rows.iloc[0]["guardrail_verdict"] == "allowed"


def test_replay_identical_then_drifted(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any], tmp_path: Path
) -> None:
    """Replay reports IDENTICAL on unchanged data and DRIFTED after mutation."""
    migrate_agent_audit(seeded_engine)
    artifacts_dir = tmp_path / "manifests"
    manifest_path = _write_manifest(tmp_path / "manifest.json", manifest)
    snapshot_manifest(manifest_path, seeded_engine, artifacts_dir=artifacts_dir)

    tools = _agent_tools(seeded_engine, manifest)
    llm = FakeLLM(
        script=[
            json.dumps(
                {
                    "action": "tool",
                    "tool": "query_mart",
                    "args": {"mart": "dim_customer", "select": ["country"], "limit": 10},
                    "reason": "count customers",
                }
            ),
            json.dumps({"action": "finish", "reason": "done"}),
            "Three customers.",
        ]
    )
    answer = run_agent(
        "How many customers?",
        tools=tools,
        llm=llm,
        manifest=manifest,
        engine=seeded_engine,
        config=GuardrailConfig(max_questions_per_minute=1000),
    )

    identical = replay_question(
        seeded_engine,
        answer.question_id,
        artifacts_dir=artifacts_dir,
        current_manifest_path=manifest_path,
    )
    assert identical["verdict"] == "IDENTICAL"

    seeded_engine.execute(
        "INSERT INTO raw_marts.dim_customer VALUES (99, 'New Person', 'n@example.com', '+1',"
        " DATE '2026-09-20', 'US', 'new')"
    )
    drifted = replay_question(
        seeded_engine,
        answer.question_id,
        artifacts_dir=artifacts_dir,
        current_manifest_path=manifest_path,
    )
    assert drifted["verdict"] == "DRIFTED"
    assert drifted["comparisons"][0]["recorded_rows"] == 3
    assert drifted["comparisons"][0]["pinned_rows"] == 4


def test_replay_reports_stale_schema(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any], tmp_path: Path
) -> None:
    """A pinned model missing from the current schema reports STALE_SCHEMA."""
    migrate_agent_audit(seeded_engine)
    artifacts_dir = tmp_path / "manifests"
    manifest_path = _write_manifest(tmp_path / "manifest.json", manifest)
    snapshot_manifest(manifest_path, seeded_engine, artifacts_dir=artifacts_dir)

    tools = _agent_tools(seeded_engine, manifest)
    llm = FakeLLM(
        script=[
            json.dumps(
                {
                    "action": "tool",
                    "tool": "query_mart",
                    "args": {"mart": "dim_customer", "select": ["country"], "limit": 10},
                    "reason": "count",
                }
            ),
            json.dumps({"action": "finish", "reason": "done"}),
            "Done.",
        ]
    )
    answer = run_agent(
        "Count customers.",
        tools=tools,
        llm=llm,
        manifest=manifest,
        engine=seeded_engine,
        config=GuardrailConfig(max_questions_per_minute=1000),
    )

    stale_manifest = _manifest()
    del stale_manifest["nodes"]["model.warehouse.dim_customer"]
    stale_path = _write_manifest(tmp_path / "stale.json", stale_manifest)
    result = replay_question(
        seeded_engine,
        answer.question_id,
        artifacts_dir=artifacts_dir,
        current_manifest_path=stale_path,
    )
    assert result["verdict"] == "STALE_SCHEMA"
