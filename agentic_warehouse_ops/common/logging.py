"""Structured logging setup shared by CLIs, DAGs, and the agent."""

from __future__ import annotations

import logging
import os

import structlog


def configure_logging(level: int = logging.INFO, json_mode: bool | None = None) -> None:
    """Configure structlog with a console (or JSON) renderer.

    Set ``AWO_LOG_JSON=1`` to emit one JSON object per line so ingestion, dbt,
    and agent logs can be shipped and correlated by their bound ids.

    Args:
        level: Minimum log level to emit (defaults to ``logging.INFO``).
        json_mode: Override JSON mode; defaults to the ``AWO_LOG_JSON`` env var.
    """
    json_enabled = (
        json_mode if json_mode is not None else os.environ.get("AWO_LOG_JSON", "0") == "1"
    )
    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_enabled
        else structlog.dev.ConsoleRenderer()
    )
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_run(run_id: str | None = None, question_id: str | None = None) -> None:
    """Bind the pipeline/agent run ids onto all subsequent log records.

    Args:
        run_id: Pipeline or agent run id.
        question_id: Agent question id (when applicable).
    """
    structlog.contextvars.bind_contextvars(
        run_id=run_id or "unknown",
        question_id=question_id or "unknown",
    )


def unbind_run() -> None:
    """Clear any previously bound run context."""
    structlog.contextvars.clear_contextvars()
