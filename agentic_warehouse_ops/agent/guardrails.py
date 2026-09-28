"""Guardrail middleware wrapping every tool invocation.

No tool may be called without going through :class:`GuardrailSession`; the
graph and any external caller must use the tools produced by
:func:`guard_tools`. The session enforces, in order: rate limits, the
per-question cost budget and wall-clock deadline, PII denial at argument
validation, a hard row LIMIT, read-only SQL, and a second PII re-scan of
results before anything reaches the LLM.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import sqlglot
import sqlglot.expressions as exp
from langchain_core.tools import StructuredTool

from agentic_warehouse_ops.agent.audit import record_tool_call
from agentic_warehouse_ops.common.warehouse import WarehouseEngine
from agentic_warehouse_ops.governance.pii import get_pii_columns


class GuardrailViolation(Exception):
    """Structured denial raised by the guardrails.

    Attributes:
        rule: Short rule id (``pii``, ``row_cap``, ``cost_cap``, ``timeout``,
            ``rate_limit``, ``read_only``).
        message: Human-readable reason.
        column: Column involved (PII denials only).
        classification: Column classification (PII denials only).
        fatal: True when the whole run must abort (cost/timeout/rate).
    """

    def __init__(
        self,
        rule: str,
        message: str,
        *,
        column: str | None = None,
        classification: str | None = None,
        fatal: bool = False,
    ) -> None:
        super().__init__(message)
        self.rule = rule
        self.message = message
        self.column = column
        self.classification = classification
        self.fatal = fatal


def assert_read_only_select(sql: str, dialect: str = "duckdb") -> None:
    """Raise unless ``sql`` parses to exactly one plain SELECT.

    Rejects statement separators (multiple statements), DDL, and DML anywhere
    in the tree — including inside CTEs.

    Args:
        sql: SQL to vet.
        dialect: sqlglot dialect to parse with.

    Raises:
        GuardrailViolation: With ``rule="read_only"`` on any violation.
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a denial
        raise GuardrailViolation("read_only", f"SQL failed to parse: {exc}", fatal=False) from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise GuardrailViolation(
            "read_only",
            "only exactly one SELECT statement is permitted (no separators, DDL, or DML)",
            fatal=False,
        )
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter)
    for node in statements[0].walk():
        if isinstance(node, forbidden):
            raise GuardrailViolation(
                "read_only",
                f"forbidden {node.__class__.__name__} node in SQL",
                fatal=False,
            )


@dataclass
class GuardrailConfig:
    """Tunable guardrail limits for one agent deployment."""

    max_rows: int = 1000
    max_result_bytes: int = 256 * 1024
    max_bytes_scanned: int = 64 * 1024 * 1024
    max_tool_calls: int = 25
    max_questions_per_minute: int = 60
    timeout_seconds: float = 120.0
    caller_role: str = "analyst"


def estimate_bytes(tool_name: str, args: dict[str, Any]) -> int:
    """Estimate scanned bytes for a tool call (credit accounting)."""
    if tool_name == "query_mart":
        return max(1, int(args.get("limit", 1000))) * 512
    if tool_name == "vector_search_tickets":
        return int(args.get("k", 5)) * 1024
    return 2048


def _json_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalise tool args for audit JSON (pydantic models -> dicts)."""
    import datetime as _datetime

    def convert(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, (_datetime.date, _datetime.datetime)):
            return value.isoformat()
        return value

    return {name: convert(value) for name, value in args.items()}


@dataclass
class GuardrailSession:
    """Per-question guardrail state; wraps and audits every tool call."""

    manifest: dict[str, Any]
    config: GuardrailConfig = field(default_factory=GuardrailConfig)
    engine: WarehouseEngine | None = None
    manifest_hash: str = "unknown"
    question_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    llm_model: str = "unknown"

    _pii_columns: set[str] = field(init=False, default_factory=set)
    _tool_calls: int = field(init=False, default=0)
    _bytes_used: int = field(init=False, default=0)
    _deadline: float = field(init=False, default=0.0)
    _started: bool = field(init=False, default=False)

    _question_times: deque = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pii_columns = get_pii_columns(self.manifest)
        self._deadline = time.monotonic() + self.config.timeout_seconds
        self._question_times = deque()
        self._lock = threading.Lock()
        if self.engine is not None:
            from agentic_warehouse_ops.agent.audit import migrate_agent_audit

            migrate_agent_audit(self.engine)

    def start_question(self) -> None:
        """Begin a question, enforcing the questions-per-minute rate limit.

        Raises:
            GuardrailViolation: Fatal, when the rate limit is breached.
        """
        now = time.monotonic()
        with self._lock:
            while self._question_times and now - self._question_times[0] > 60.0:
                self._question_times.popleft()
            if len(self._question_times) >= self.config.max_questions_per_minute:
                raise GuardrailViolation(
                    "rate_limit",
                    f"max {self.config.max_questions_per_minute} questions per minute exceeded",
                    fatal=True,
                )
            self._question_times.append(now)
        self._started = True

    def _classification(self, mart: str, column: str) -> str:
        node = next(
            (n for n in self.manifest.get("nodes", {}).values() if n.get("name") == mart),
            None,
        )
        meta = ((node or {}).get("columns") or {}).get(column, {}).get("meta") or {}
        return str(meta.get("classification", "restricted"))

    def _deny_pii_columns(self, mart: str, columns: list[str]) -> None:
        """Reject any requested column in the PII set (first enforcement layer)."""
        for column in columns:
            if f"{mart}.{column}" in self._pii_columns:
                raise GuardrailViolation(
                    "pii",
                    f"column '{mart}.{column}' is PII and cannot be queried",
                    column=column,
                    classification=self._classification(mart, column),
                    fatal=False,
                )

    def _pre_check(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalise arguments before execution."""
        checked = dict(args)
        if tool_name == "query_mart":
            mart = str(checked.get("mart", ""))
            self._deny_pii_columns(mart, list(checked.get("select", [])))
            checked["limit"] = min(
                int(checked.get("limit", self.config.max_rows)), self.config.max_rows
            )
        return checked

    def _post_check(
        self, tool_name: str, args: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Re-scan results: PII columns, row cap, and byte ceiling."""
        checked = dict(result)
        if tool_name == "query_mart" and "records" in checked:
            mart = str(args.get("mart", ""))
            for record in checked["records"]:
                for key in record:
                    if f"{mart}.{key}" in self._pii_columns:
                        raise GuardrailViolation(
                            "pii",
                            f"unexpected PII column '{mart}.{key}' appeared in results",
                            column=str(key),
                            classification=self._classification(mart, str(key)),
                            fatal=False,
                        )
            checked["records"] = list(checked["records"])[: self.config.max_rows]
            while checked["records"]:
                size = len(json.dumps(checked["records"], default=str).encode("utf-8"))
                if size <= self.config.max_result_bytes:
                    break
                checked["records"].pop()
        return checked

    def _write_audit(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        compiled_sql: str | None,
        rows_returned: int,
        bytes_scanned: int,
        duration_ms: int,
        guardrail_verdict: str,
    ) -> None:
        if self.engine is None:
            return
        record_tool_call(
            self.engine,
            audit_id=uuid.uuid4().hex,
            question_id=self.question_id,
            run_id=self.run_id,
            tool_name=tool_name,
            tool_args=_json_args(tool_args),
            compiled_sql=compiled_sql,
            rows_returned=rows_returned,
            bytes_scanned=bytes_scanned,
            duration_ms=duration_ms,
            guardrail_verdict=guardrail_verdict,
            caller_role=self.config.caller_role,
            llm_model=self.llm_model,
            manifest_hash=self.manifest_hash,
        )

    def invoke(self, tool_name: str, args: dict[str, Any], fn: Callable[..., Any]) -> Any:
        """Run one tool through every guardrail, audit included.

        Args:
            tool_name: Tool name (also the audit key).
            args: Tool arguments (schema-validated upstream).
            fn: The underlying tool function.

        Returns:
            The tool result, post-processed by the guardrails.

        Raises:
            GuardrailViolation: On any denial; ``fatal`` is True for
                cost/timeout/rate-limit aborts.
        """
        now = time.monotonic()
        if now > self._deadline:
            raise GuardrailViolation(
                "timeout",
                f"wall-clock budget of {self.config.timeout_seconds}s exceeded",
                fatal=True,
            )
        if self._tool_calls >= self.config.max_tool_calls:
            raise GuardrailViolation(
                "rate_limit",
                f"max {self.config.max_tool_calls} tool calls per question exceeded",
                fatal=True,
            )
        estimated = estimate_bytes(tool_name, args)
        if self._bytes_used + estimated > self.config.max_bytes_scanned:
            raise GuardrailViolation(
                "cost_cap",
                f"cost budget exceeded: {self._bytes_used + estimated} > "
                f"{self.config.max_bytes_scanned} estimated bytes",
                fatal=True,
            )
        try:
            checked_args = self._pre_check(tool_name, args)
        except GuardrailViolation as exc:
            self._tool_calls += 1
            self._write_audit(
                tool_name=tool_name,
                tool_args=args,
                compiled_sql=None,
                rows_returned=0,
                bytes_scanned=estimated,
                duration_ms=0,
                guardrail_verdict=f"denied: {exc.rule}: {exc.message}",
            )
            raise
        started = time.monotonic()
        try:
            result = fn(checked_args)
        except Exception as exc:  # noqa: BLE001 - every failure is audited
            self._tool_calls += 1
            self._write_audit(
                tool_name=tool_name,
                tool_args=checked_args,
                compiled_sql=None,
                rows_returned=0,
                bytes_scanned=estimated,
                duration_ms=int((time.monotonic() - started) * 1000),
                guardrail_verdict=f"denied: execution error: {str(exc)[:200]}",
            )
            raise
        duration_ms = int((time.monotonic() - started) * 1000)
        if not isinstance(result, dict):
            result = {"value": result, "row_count": 0}
        checked = self._post_check(tool_name, checked_args, result)
        if isinstance(checked.get("sql"), str):
            assert_read_only_select(str(checked["sql"]))
        self._tool_calls += 1
        self._bytes_used += estimated
        self._write_audit(
            tool_name=tool_name,
            tool_args=checked_args,
            compiled_sql=str(checked["sql"]) if checked.get("sql") else None,
            rows_returned=int(checked.get("row_count", 0)),
            bytes_scanned=estimated,
            duration_ms=duration_ms,
            guardrail_verdict="allowed",
        )
        return checked


def guard_tools(
    tools: dict[str, StructuredTool],
    session: GuardrailSession,
) -> dict[str, StructuredTool]:
    """Wrap every tool so no invocation can bypass the guardrails.

    Args:
        tools: Tools from :func:`agentic_warehouse_ops.agent.tools.build_tools`.
        session: The guardrail session enforcing limits and auditing.

    Returns:
        Equivalent tools whose ``invoke`` always passes through the session.
    """
    guarded: dict[str, StructuredTool] = {}
    for name, tool in tools.items():

        def make_guarded(tool_name: str, original: StructuredTool) -> Callable[..., Any]:
            def guarded_func(**kwargs: Any) -> Any:
                return session.invoke(tool_name, kwargs, lambda args: original.invoke(args))

            return guarded_func

        guarded[name] = StructuredTool.from_function(
            func=make_guarded(name, tool),
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
        )
    return guarded


__all__ = [
    "GuardrailConfig",
    "GuardrailSession",
    "GuardrailViolation",
    "assert_read_only_select",
    "estimate_bytes",
    "guard_tools",
]
