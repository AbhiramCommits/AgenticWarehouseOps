"""CLI entrypoint: ``awo ask "<question>"``."""

from __future__ import annotations

import json

import typer

from agentic_warehouse_ops.agent.graph import build_agent

app = typer.Typer(help="Ask questions over the governed warehouse marts.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the package version."""
    from agentic_warehouse_ops import __version__

    typer.echo(__version__)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question about the marts."),
    trace: bool = typer.Option(
        True, "--trace/--no-trace", help="Print the tool trace after the answer."
    ),
) -> None:
    """Answer a question and print the tool trace."""
    agent = build_agent()
    answer = agent(question)
    typer.echo(f"\n{answer.answer}\n")
    if not trace:
        return
    for call in answer.tool_calls:
        suffix = f" ERROR: {call.error}" if call.error else ""
        typer.echo(
            f"[tool] {call.name}({json.dumps(call.args, default=str)})"
            f" -> {call.row_count} rows in {call.duration_ms}ms{suffix}"
        )
    for sql in answer.sql_executed:
        typer.echo(f"[sql] {sql}")
    if answer.evidence_ids:
        typer.echo(f"[evidence] {', '.join(answer.evidence_ids)}")
    typer.echo(f"[run] {answer.run_id} manifest_hash={answer.manifest_hash[:12]}")


if __name__ == "__main__":
    app()
