"""LangGraph tool-calling agent.

Graph: ``plan -> select_tool -> execute_tool -> (loop, max 5 iterations) ->
synthesize_answer``. The LLM is pluggable behind :func:`get_llm`; the bundled
:class:`FakeLLM` follows a deterministic script so the whole graph runs in CI
with no API key and no network.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import Field, PrivateAttr

from agentic_warehouse_ops.agent.guardrails import GuardrailViolation
from agentic_warehouse_ops.agent.models import AgentAnswer, AgentStep, ToolCallRecord
from agentic_warehouse_ops.agent.tools import ToolContext, build_tools
from agentic_warehouse_ops.governance.catalog import compute_manifest_hash, load_manifest

MAX_ITERATIONS = 5
MAX_EVIDENCE_CHARS = 6000

SYSTEM_PROMPT = """You answer questions about a retail-ops warehouse using typed tools only.
Contract:
- You may ONLY use the tools listed below; never write or suggest SQL yourself.
- Every value in your final answer must come from tool results; never guess or
  invent numbers, dates, names, or ids that a tool did not return.
- Restricted (PII) columns are unavailable by design; do not ask for them
  again if a tool reports them as restricted.
- Plan one step at a time and reply ONLY with a JSON object:
  {"action": "tool", "tool": "<name>", "args": {...}, "reason": "..."}
  or {"action": "finish", "reason": "..."}.

Available tools:
{tools}"""

SYNTHESIZE_PROMPT = """You answer questions about a retail-ops warehouse.
Answer concisely and truthfully using ONLY the evidence below; if the evidence
is insufficient, say so instead of guessing. Never invent numbers or ids.

Question: {question}

Evidence: {evidence}

Tool history: {history}"""


class AgentState(TypedDict, total=False):
    """State threaded through the graph."""

    question: str
    run_id: str
    history: list[ToolCallRecord]
    evidence: list[Any]
    sql_executed: list[str]
    evidence_ids: list[str]
    step: AgentStep
    iteration: int
    error: str
    answer: str
    abort_reason: str | None
    route: str


class FakeLLM(BaseChatModel):
    """Deterministic scripted chat model for CI and offline development.

    Each :meth:`_generate` call consumes the next entry of ``script``; once
    exhausted it repeats the last entry, or ``fallback`` when empty.
    """

    script: list[str] = Field(default_factory=list)
    fallback: str = "No LLM provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or AGENT_LLM_PROVIDER=fake with a script."
    model_name: str = "fake-scripted"

    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._index < len(self.script):
            content = self.script[self._index]
        elif self.script:
            content = self.script[-1]
        else:
            content = self.fallback
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return the configured LLM.

    Provider resolution: explicit ``provider`` argument, then
    ``AGENT_LLM_PROVIDER``, then API-key autodetection (``OPENAI_API_KEY`` or
    ``ANTHROPIC_API_KEY``). Falls back to a :class:`FakeLLM` that explains how
    to configure a real provider, keeping every code path runnable offline.

    Args:
        provider: Optional ``"fake"``, ``"openai"``, or ``"anthropic"``.
    """
    name = (provider or os.environ.get("AGENT_LLM_PROVIDER", "")).strip().lower()
    if name == "fake":
        return FakeLLM()
    model = os.environ.get("AGENT_LLM_MODEL", "")
    if name == "openai" or (not name and os.environ.get("OPENAI_API_KEY")):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o-mini")
    if name == "anthropic" or (not name and os.environ.get("ANTHROPIC_API_KEY")):
        from langchain_anthropic import ChatAnthropic

        return cast(Any, ChatAnthropic)(model=model or "claude-3-5-haiku-latest")
    return FakeLLM()


def _parse_step(content: str) -> AgentStep:
    """Parse an LLM response into an :class:`AgentStep`; unparseable output
    becomes a safe ``finish`` step so the graph never loops on garbage."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return AgentStep.model_validate_json(text)
    except Exception:
        return AgentStep(action="finish", reason=f"unparseable plan: {content[:200]!r}")


def _build_plan_node(
    llm: BaseChatModel, tools: dict[str, StructuredTool]
) -> Callable[[AgentState], AgentState]:
    tools_blob = json.dumps(
        {name: cast(Any, tool.args_schema).model_json_schema() for name, tool in tools.items()},
        default=str,
    )

    def plan(state: AgentState) -> AgentState:
        history_blob = json.dumps(
            [record.model_dump() for record in state.get("history", [])],
            default=str,
        )[-4000:]
        prompt = (
            f"Question: {state['question']}\nTool history: {history_blob}\n"
            "Plan the next step as JSON only."
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT.replace("{tools}", tools_blob)),
                HumanMessage(content=prompt),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"step": _parse_step(content)}

    return plan


def _build_select_node(tools: dict[str, StructuredTool]) -> Callable[[AgentState], AgentState]:
    def select_tool(state: AgentState) -> AgentState:
        step = state.get("step")
        if step is None or step.action != "tool" or step.tool not in tools:
            return {"route": "synthesize", "error": "no valid tool selected"}
        return {"route": "execute_tool"}

    return select_tool


def _build_execute_node(
    tools: dict[str, StructuredTool],
    max_iterations: int,
) -> Callable[[AgentState], AgentState]:
    def execute_tool(state: AgentState) -> AgentState:
        step = state.get("step")
        if step is None or step.tool is None:
            return {"route": "synthesize"}
        iteration = state.get("iteration", 0) + 1
        record = ToolCallRecord(name=step.tool, args=step.args)
        started = time.monotonic()
        history = [*state.get("history", []), record]
        sql_executed = list(state.get("sql_executed", []))
        evidence = list(state.get("evidence", []))
        evidence_ids = list(state.get("evidence_ids", []))
        try:
            result = tools[step.tool].invoke(step.args)
        except GuardrailViolation as exc:
            record.duration_ms = int((time.monotonic() - started) * 1000)
            record.error = str(exc)[:500]
            if exc.fatal:
                return {
                    "iteration": iteration,
                    "history": history,
                    "abort_reason": exc.message,
                    "route": "synthesize",
                }
            return {
                "iteration": iteration,
                "history": history,
                "error": str(exc)[:500],
                "route": "synthesize" if iteration >= max_iterations else "plan",
            }
        except Exception as exc:  # noqa: BLE001 - any tool failure is recorded
            record.duration_ms = int((time.monotonic() - started) * 1000)
            record.error = str(exc)[:500]
            return {
                "iteration": iteration,
                "history": history,
                "error": str(exc)[:500],
                "route": "synthesize" if iteration >= max_iterations else "plan",
            }
        record.duration_ms = int((time.monotonic() - started) * 1000)
        if isinstance(result, dict):
            record.row_count = int(result.get("row_count", 0))
            if "records" in result:
                evidence.extend(result["records"])
            elif "results" in result:
                evidence.extend(result["results"])
                evidence_ids.extend(
                    str(item.get("ticket_id"))
                    for item in result["results"]
                    if item.get("ticket_id")
                )
            elif "models" in result:
                evidence.append(result["models"])
            if result.get("sql"):
                sql_executed.append(str(result["sql"]))
        return {
            "iteration": iteration,
            "history": history,
            "evidence": evidence,
            "evidence_ids": evidence_ids,
            "sql_executed": sql_executed,
            "route": "synthesize" if iteration >= max_iterations else "plan",
        }

    return execute_tool


def _build_synthesize_node(llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    def synthesize(state: AgentState) -> AgentState:
        evidence_blob = json.dumps(state.get("evidence", []), default=str)[:MAX_EVIDENCE_CHARS]
        history_blob = json.dumps(
            [record.model_dump() for record in state.get("history", [])],
            default=str,
        )[-3000:]
        prompt = SYNTHESIZE_PROMPT.format(
            question=state["question"],
            evidence=evidence_blob,
            history=history_blob,
        )
        abort_reason = state.get("abort_reason")
        if abort_reason:
            prompt += (
                f"\nRUN ABORTED: {abort_reason}. Give a partial answer using the evidence "
                "above and state the abort reason."
            )
        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZE_PROMPT.split("Question:")[0].strip()),
                HumanMessage(content=prompt),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"answer": content}

    return synthesize


def build_graph(
    llm: BaseChatModel,
    tools: dict[str, StructuredTool],
    max_iterations: int = MAX_ITERATIONS,
) -> Any:
    """Compile the agent state graph.

    Args:
        llm: Chat model used for planning and synthesis.
        tools: Tool map from :func:`agentic_warehouse_ops.agent.tools.build_tools`.
        max_iterations: Cap on tool executions (default 5).

    Returns:
        A compiled LangGraph graph.
    """
    graph = StateGraph(AgentState)
    graph.add_node("plan", cast(Any, _build_plan_node(llm, tools)))
    graph.add_node("select_tool", cast(Any, _build_select_node(tools)))
    graph.add_node("execute_tool", cast(Any, _build_execute_node(tools, max_iterations)))
    graph.add_node("synthesize", cast(Any, _build_synthesize_node(llm)))
    graph.set_entry_point("plan")
    graph.add_edge("plan", "select_tool")
    graph.add_conditional_edges(
        "select_tool",
        lambda state: state.get("route", "synthesize"),
        {"execute_tool": "execute_tool", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "execute_tool",
        lambda state: state.get("route", "plan"),
        {"plan": "plan", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_agent(
    question: str,
    *,
    tools: dict[str, StructuredTool],
    llm: BaseChatModel,
    manifest: dict[str, Any],
    run_id: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
    engine: Any = None,
    config: Any = None,
    session: Any = None,
) -> AgentAnswer:
    """Run the full agent graph through the guardrails and return the answer.

    Every tool call is wrapped by a :class:`GuardrailSession`, every call and
    the question itself are written to ``meta.agent_audit`` /
    ``meta.agent_questions`` (even when the run fails), and the answer is
    stamped with the manifest hash it ran against.

    Args:
        question: Natural-language question.
        tools: Tool map bound to the runtime context.
        llm: Chat model for planning/synthesis.
        manifest: Parsed dbt manifest (hash embedded in the answer).
        run_id: Optional run id; defaults to a fresh UUID.
        max_iterations: Tool-execution cap (default 5).
        engine: Optional warehouse engine; when set, audit rows are written.
        config: Optional :class:`GuardrailConfig`; defaults are used otherwise.
        session: Optional pre-built guardrail session.

    Returns:
        A populated :class:`AgentAnswer`. Guardrail aborts return a partial
        answer with ``status="aborted"`` and the reason.

    Raises:
        Exception: Re-raised after recording the failed question.
    """
    import structlog

    from agentic_warehouse_ops.agent.audit import migrate_agent_audit, record_question
    from agentic_warehouse_ops.agent.guardrails import (
        GuardrailConfig,
        GuardrailSession,
        GuardrailViolation,
        guard_tools,
    )
    from agentic_warehouse_ops.common.logging import bind_run, unbind_run

    log = structlog.get_logger()
    run_id = run_id or uuid.uuid4().hex
    question_id = uuid.uuid4().hex
    manifest_hash = compute_manifest_hash(manifest)
    llm_model = str(getattr(llm, "model_name", None) or getattr(llm, "_llm_type", "unknown"))
    if engine is not None:
        migrate_agent_audit(engine)
    resolved_session: GuardrailSession = session or GuardrailSession(
        manifest=manifest,
        config=config or GuardrailConfig(),
        engine=engine,
        manifest_hash=manifest_hash,
        question_id=question_id,
        run_id=run_id,
        llm_model=llm_model,
    )
    guarded_tools = guard_tools(tools, resolved_session)
    bind_run(run_id=run_id, question_id=question_id)
    log.info("agent_start", llm_model=llm_model, manifest_hash=manifest_hash)

    def _finish(answer: AgentAnswer, status: str, tool_calls: int, total_cost: int) -> AgentAnswer:
        if engine is not None:
            record_question(
                engine,
                question_id=question_id,
                run_id=run_id,
                question=question,
                answer=answer.answer,
                status=status,
                tool_call_count=tool_calls,
                total_cost=total_cost,
                manifest_hash=manifest_hash,
                llm_model=llm_model,
                caller_role=resolved_session.config.caller_role,
            )
        unbind_run()
        return answer

    try:
        resolved_session.start_question()
    except GuardrailViolation as exc:
        log.warning("agent_rejected", reason=exc.message)
        return _finish(
            AgentAnswer(
                answer=f"Request rejected: {exc.message}",
                manifest_hash=manifest_hash,
                run_id=run_id,
                question_id=question_id,
                status="rejected",
                abort_reason=exc.message,
            ),
            "rejected",
            0,
            0,
        )
    graph = build_graph(llm, guarded_tools, max_iterations=max_iterations)
    try:
        final: dict[str, Any] = graph.invoke(
            {
                "question": question,
                "run_id": run_id,
                "history": [],
                "evidence": [],
                "sql_executed": [],
                "evidence_ids": [],
                "iteration": 0,
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        log.exception("agent_failed", reason=str(exc)[:300])
        record_question(
            engine,
            question_id=question_id,
            run_id=run_id,
            question=question,
            answer=f"run failed: {str(exc)[:500]}",
            status="failed",
            tool_call_count=resolved_session._tool_calls,
            total_cost=resolved_session._bytes_used,
            manifest_hash=manifest_hash,
            llm_model=llm_model,
            caller_role=resolved_session.config.caller_role,
        ) if engine is not None else None
        unbind_run()
        raise
    abort_reason = final.get("abort_reason")
    status = "aborted" if abort_reason else "success"
    answer = AgentAnswer(
        answer=str(final.get("answer", "")),
        tool_calls=list(final.get("history", [])),
        sql_executed=list(final.get("sql_executed", [])),
        evidence_ids=list(final.get("evidence_ids", [])),
        manifest_hash=manifest_hash,
        run_id=run_id,
        question_id=question_id,
        status=status,
        abort_reason=abort_reason,
    )
    log.info("agent_complete", status=status, tool_calls=len(answer.tool_calls))
    return _finish(answer, status, resolved_session._tool_calls, resolved_session._bytes_used)


def build_agent(
    *,
    manifest_path: str | Path = Path("dbt/target/manifest.json"),
    llm: BaseChatModel | None = None,
    embedder: Any = None,
    engine: Any = None,
    vector_store: Any = None,
) -> Callable[..., AgentAnswer]:
    """Build a ready-to-ask agent bound to local defaults.

    Args:
        manifest_path: Path to the dbt manifest (column validation source).
        llm: Optional chat model; defaults to :func:`get_llm`.
        embedder: Optional embedder; defaults to :func:`get_embedder`.
        engine: Optional warehouse engine; defaults to the DuckDB profile.
        vector_store: Optional vector store; defaults to
            :class:`~agentic_warehouse_ops.agent.index.DuckDBVectorStore`.

    Returns:
        A callable ``ask(question, run_id=None) -> AgentAnswer``.
    """
    from agentic_warehouse_ops.agent.index import DuckDBVectorStore, get_embedder
    from agentic_warehouse_ops.common.warehouse import get_engine

    manifest = load_manifest(manifest_path)
    resolved_engine = engine or get_engine()
    resolved_embedder = embedder or get_embedder()
    resolved_store = vector_store or DuckDBVectorStore(resolved_engine, resolved_embedder)
    tools = build_tools(
        ToolContext(
            engine=resolved_engine,
            manifest=manifest,
            vector_store=resolved_store,
            embedder=resolved_embedder,
        )
    )
    resolved_llm = llm or get_llm()

    def ask(question: str, run_id: str | None = None) -> AgentAnswer:
        return run_agent(
            question,
            tools=tools,
            llm=resolved_llm,
            manifest=manifest,
            run_id=run_id,
            engine=resolved_engine,
        )

    return ask


__all__ = ["FakeLLM", "build_agent", "build_graph", "get_llm", "run_agent"]
