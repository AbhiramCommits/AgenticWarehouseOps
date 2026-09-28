"""Eval runner: score the agent against the committed question set.

The runner uses the deterministic scripted FakeLLM (plans are derived from
each question's ``expected_tools``), so every scored signal comes from the
real stack — typed tool execution against the warehouse, guardrail
enforcement, audit rows — and needs no API keys or network.

Answer accuracy is measured on the agent's tool-derived evidence: extracted
values/counts from ``query_mart`` results, ticket-id sets from
``vector_search_tickets``, schema facts from ``lookup_schema``, and guardrail
denials for governance-refusal questions.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agentic_warehouse_ops.agent.audit import get_tool_calls
from agentic_warehouse_ops.agent.embedder import Embedder, FakeEmbedder
from agentic_warehouse_ops.agent.graph import FakeLLM, run_agent
from agentic_warehouse_ops.agent.guardrails import GuardrailConfig
from agentic_warehouse_ops.agent.index import (
    DuckDBVectorStore,
    embed_catalog,
    embed_support_tickets,
)
from agentic_warehouse_ops.agent.tools import ToolContext, build_tools
from agentic_warehouse_ops.common.warehouse import WarehouseEngine, get_engine
from agentic_warehouse_ops.governance.catalog import load_manifest

DEFAULT_QUESTIONS = "agentic_warehouse_ops/evals/questions.yaml"
REFUSAL_CATEGORY = "governance-refusal"

_FINAL_REFUSAL = "I cannot provide that information: the requested column is restricted PII."
_FINAL_GENERIC = "Answer derived from the tool evidence."


def _norm(value: Any) -> str:
    """Normalise an extracted value for comparison."""
    if value is None:
        return "None"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _last_query_result(answer: Any) -> dict[str, Any] | None:
    for entry in reversed(answer.tool_results):
        if entry.get("tool") == "query_mart" and isinstance(entry.get("result"), dict):
            result = entry["result"]
            if result.get("records"):
                return result
    return None


def _extract_answer(answer: Any, question: dict[str, Any]) -> Any:
    """Extract the candidate answer from the agent's tool results."""
    expected = question["expected_answer"]
    kind = expected["kind"]
    if kind == "refusal":
        return "refused" if _was_refused(answer) else "answered"
    if kind == "set":
        return set(answer.evidence_ids)
    if kind == "count":
        for call in reversed(answer.tool_calls):
            if call.name == "query_mart" and call.error is None:
                return call.row_count
        return None
    result = _last_query_result(answer)
    if result is None:
        return None
    column = expected.get("column") or list(result["records"][0].keys())[0]
    return result["records"][0].get(column)


def _was_refused(answer: Any) -> bool:
    denied = any(
        call.error and ("restricted" in call.error.lower() or "pii" in call.error.lower())
        for call in answer.tool_calls
    )
    succeeded = any(call.name == "query_mart" and call.error is None for call in answer.tool_calls)
    return denied and not succeeded


def _score_answer(answer: Any, question: dict[str, Any]) -> bool:
    """Score answer accuracy for one question."""
    expected = question["expected_answer"]
    kind = expected["kind"]
    candidate = _extract_answer(answer, question)
    if kind == "refusal":
        return candidate == "refused"
    if kind == "set":
        expected_set = set(str(value) for value in expected["values"])
        actual_set = set(str(value) for value in (candidate or set()))
        if not expected_set and not actual_set:
            return True
        if not expected_set or not actual_set:
            return False
        intersection = expected_set & actual_set
        precision = len(intersection) / len(actual_set)
        recall = len(intersection) / len(expected_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return f1 >= 0.999
    if kind == "numeric":
        try:
            return abs(float(candidate) - float(expected["value"])) <= float(
                expected.get("tolerance", 0.01)
            )
        except (TypeError, ValueError):
            return False
    if kind == "schema":
        return _score_schema(answer, expected)
    return _norm(candidate) == _norm(expected.get("value"))


def _score_schema(answer: Any, expected: dict[str, Any]) -> bool:
    for entry in reversed(answer.tool_results):
        result = entry.get("result")
        if not isinstance(result, dict) or "models" not in result:
            continue
        for model in result["models"]:
            if model.get("name") != expected.get("model"):
                continue
            by_name = {column["name"]: column for column in model.get("columns", [])}
            for name in expected.get("restricted", []):
                if by_name.get(name, {}).get("available") is not False:
                    return False
            for name in expected.get("available", []):
                if by_name.get(name, {}).get("available") is not True:
                    return False
            return True
    return False


def _tool_sequences(answer: Any, question: dict[str, Any]) -> tuple[bool, bool]:
    """Return (exact-match, set-match) for tool selection."""
    expected_names = [step["tool"] for step in question["expected_tools"]]
    actual_names = [call.name for call in answer.tool_calls]
    exact = actual_names == expected_names
    as_set = set(actual_names) == set(expected_names)
    return exact, as_set


def _build_script(question: dict[str, Any]) -> list[str]:
    """Build the FakeLLM script for one question from its expected tools."""
    script: list[str] = []
    for step in question["expected_tools"]:
        script.append(
            json.dumps(
                {
                    "action": "tool",
                    "tool": step["tool"],
                    "args": step.get("args", {}),
                    "reason": f"step for {question['id']}",
                }
            )
        )
    script.append(json.dumps({"action": "finish", "reason": "expected steps complete"}))
    if question["expected_answer"].get("kind") == "refusal":
        script.append(_FINAL_REFUSAL)
    else:
        script.append(_FINAL_GENERIC)
    return script


def run_eval(
    questions_path: str | Path = DEFAULT_QUESTIONS,
    out_dir: str | Path = "artifacts/evals",
    baseline: str | Path | None = None,
    max_regression: float = 0.05,
    engine: WarehouseEngine | None = None,
    embedder: Embedder | None = None,
    manifest_path: str | Path = "dbt/target/manifest.json",
) -> tuple[dict[str, Any], list[str]]:
    """Run the full eval and return ``(summary, regressions)``.

    Args:
        questions_path: Path to the YAML question set.
        out_dir: Root for the timestamped output directory.
        baseline: Optional path to a previous ``summary.json`` to diff against.
        max_regression: Absolute score drop that fails the run.
        engine: Optional warehouse engine; defaults to the DuckDB profile.
        embedder: Optional embedder; defaults to the deterministic
            :class:`~agentic_warehouse_ops.agent.embedder.FakeEmbedder`.
        manifest_path: dbt manifest used for schema validation.

    Returns:
        ``(summary, regressions)`` where ``regressions`` lists human-readable
        failures against the baseline (empty when none).
    """
    resolved_engine = engine or get_engine()
    resolved_embedder = embedder or FakeEmbedder()
    manifest = load_manifest(manifest_path)
    store = DuckDBVectorStore(resolved_engine, resolved_embedder)
    embed_support_tickets(resolved_engine, embedder=resolved_embedder)
    embed_catalog(resolved_engine, manifest, embedder=resolved_embedder)
    tools = build_tools(
        ToolContext(
            engine=resolved_engine,
            manifest=manifest,
            vector_store=store,
            embedder=resolved_embedder,
        )
    )
    config = GuardrailConfig(max_questions_per_minute=1000)
    questions = yaml.safe_load(Path(questions_path).read_text())["questions"]

    rows: list[dict[str, Any]] = []
    latencies: list[int] = []
    tool_counts: list[int] = []
    bytes_per_question: list[int] = []
    per_category: dict[str, dict[str, Any]] = {}
    refused: list[bool] = []
    for question in questions:
        llm = FakeLLM(script=_build_script(question))
        started = time.monotonic()
        answer = run_agent(
            question["question"],
            tools=tools,
            llm=llm,
            manifest=manifest,
            engine=resolved_engine,
            config=config,
            run_id=f"eval-{question['id']}",
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        correct = _score_answer(answer, question)
        exact, as_set = _tool_sequences(answer, question)
        audit_rows = get_tool_calls(resolved_engine, answer.question_id)
        bytes_scanned = int(audit_rows["bytes_scanned"].sum()) if not audit_rows.empty else 0
        latencies.append(latency_ms)
        tool_counts.append(len(answer.tool_calls))
        bytes_per_question.append(bytes_scanned)
        if question["category"] == REFUSAL_CATEGORY:
            refused.append(correct)
        category = per_category.setdefault(
            question["category"],
            {"n": 0, "correct": 0, "exact": 0, "set_match": 0},
        )
        category["n"] += 1
        category["correct"] += int(correct)
        category["exact"] += int(exact)
        category["set_match"] += int(as_set)
        rows.append(
            {
                "id": question["id"],
                "category": question["category"],
                "difficulty": question["difficulty"],
                "question": question["question"],
                "correct": bool(correct),
                "tool_exact": bool(exact),
                "tool_set_match": bool(as_set),
                "latency_ms": latency_ms,
                "tool_calls": len(answer.tool_calls),
                "bytes_scanned": bytes_scanned,
                "expected": question["expected_answer"],
                "actual": _extract_answer(answer, question),
                "answer_status": answer.status,
                "abort_reason": answer.abort_reason,
                "question_id": answer.question_id,
            }
        )

    total = len(questions)
    refusal_precision = sum(refused) / len(refused) if refused else 1.0
    refusal_recall = (
        sum(refused) / len([q for q in questions if q["category"] == REFUSAL_CATEGORY])
        if any(q["category"] == REFUSAL_CATEGORY for q in questions)
        else 1.0
    )
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_questions": total,
        "overall": {
            "accuracy": round(sum(row["correct"] for row in rows) / total, 4),
            "tool_exact_rate": round(sum(row["tool_exact"] for row in rows) / total, 4),
            "tool_set_rate": round(sum(row["tool_set_match"] for row in rows) / total, 4),
            "refusal_precision": round(refusal_precision, 4),
            "refusal_recall": round(refusal_recall, 4),
            "mean_latency_ms": round(sum(latencies) / total, 1),
            "mean_tool_calls": round(sum(tool_counts) / total, 2),
            "mean_bytes_scanned": round(sum(bytes_per_question) / total, 1),
        },
        "by_category": {
            name: {
                "n": stats["n"],
                "accuracy": round(stats["correct"] / stats["n"], 4),
                "tool_exact_rate": round(stats["exact"] / stats["n"], 4),
                "tool_set_rate": round(stats["set_match"] / stats["n"], 4),
            }
            for name, stats in sorted(per_category.items())
        },
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    output_dir = Path(out_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "questions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "summary.md").write_text(_summary_markdown(summary))

    regressions: list[str] = []
    if baseline:
        regressions = _compare_baseline(summary, Path(baseline), max_regression)
    return summary, regressions


def _compare_baseline(
    summary: dict[str, Any],
    baseline_path: Path,
    max_regression: float,
) -> list[str]:
    """Return regression lines when the new summary drops below the baseline."""
    baseline = json.loads(baseline_path.read_text())
    regressions: list[str] = []
    for metric, new_value in summary["overall"].items():
        old_value = baseline["overall"].get(metric)
        if old_value is None or metric in (
            "mean_latency_ms",
            "mean_tool_calls",
            "mean_bytes_scanned",
        ):
            continue
        if new_value < old_value - max_regression:
            regressions.append(
                f"overall.{metric}: {new_value} < baseline {old_value} - {max_regression}"
            )
    for category, stats in summary["by_category"].items():
        old = baseline.get("by_category", {}).get(category, {})
        for metric in ("accuracy", "tool_exact_rate", "tool_set_rate"):
            old_value = old.get(metric)
            if old_value is None:
                continue
            if stats[metric] < old_value - max_regression:
                regressions.append(
                    f"{category}.{metric}: {stats[metric]} < baseline {old_value} - {max_regression}"
                )
    return regressions


def _summary_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as a markdown report."""
    lines = [
        f"# Eval summary — {summary['generated_at']}",
        "",
        f"Total questions: **{summary['total_questions']}**",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for metric, value in summary["overall"].items():
        lines.append(f"| {metric} | {value} |")
    lines += [
        "",
        "## By category",
        "",
        "| category | n | accuracy | tool exact | tool set |",
        "| --- | --- | --- | --- | --- |",
    ]
    for category, stats in summary["by_category"].items():
        lines.append(
            f"| {category} | {stats['n']} | {stats['accuracy']} | "
            f"{stats['tool_exact_rate']} | {stats['tool_set_rate']} |"
        )
    return "\n".join(lines) + "\n"
