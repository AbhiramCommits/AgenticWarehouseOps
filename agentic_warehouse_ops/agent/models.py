"""Pydantic models for agent responses and planning steps."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AgentStep(BaseModel):
    """One LLM-planned action: invoke a tool, or finish the task."""

    action: Literal["tool", "finish"]
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @model_validator(mode="after")
    def _require_tool_for_tool_action(self) -> AgentStep:
        if self.action == "tool" and not self.tool:
            raise ValueError("tool name is required when action is 'tool'")
        return self


class ToolCallRecord(BaseModel):
    """One executed tool call, as reported in the :class:`AgentAnswer` trace."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    row_count: int = 0
    duration_ms: int = 0
    error: str | None = None


class AgentAnswer(BaseModel):
    """The agent's final, auditable answer."""

    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    sql_executed: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    manifest_hash: str
    run_id: str
    question_id: str = ""
    status: str = "success"
    abort_reason: str | None = None
