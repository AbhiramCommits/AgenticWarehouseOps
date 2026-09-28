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
def audit(
    last: int = typer.Option(20, "--last", help="Number of recent questions to show."),
    question_id: str | None = typer.Option(None, "--question-id", help="Filter to one question."),
) -> None:
    """Show the agent audit log (questions and their tool calls)."""
    from agentic_warehouse_ops.agent.audit import (
        list_audit,
        list_questions,
        migrate_agent_audit,
    )
    from agentic_warehouse_ops.common.warehouse import get_engine

    engine = get_engine()
    migrate_agent_audit(engine)
    questions = list_questions(engine, limit=last)
    for row in questions.to_dict("records"):
        typer.echo(
            f"[question] {row['question_id']} status={row['status']} "
            f"tools={row['tool_call_count']} cost={row['total_cost']} model={row['llm_model']}"
        )
        typer.echo(f"  q: {row['question'][:120]}")
        typer.echo(f"  a: {row['answer'][:160]}")
    calls = list_audit(engine, limit=last, question_id=question_id)
    for row in calls.to_dict("records"):
        typer.echo(
            f"[call] {row['tool_name']} rows={row['rows_returned']} "
            f"bytes={row['bytes_scanned']} verdict={row['guardrail_verdict']} "
            f"question={row['question_id']}"
        )


@app.command()
def replay(
    question_id: str = typer.Option(
        ..., "--question-id", help="Question id recorded in the audit log."
    ),
    current_manifest: str | None = typer.Option(
        None, "--current-manifest", help="Path to the current manifest to diff against."
    ),
) -> None:
    """Re-execute a recorded question's tool calls and print the drift verdict."""
    from agentic_warehouse_ops.common.reproducibility import replay_question
    from agentic_warehouse_ops.common.warehouse import get_engine

    result = replay_question(
        get_engine(),
        question_id,
        current_manifest_path=current_manifest,
    )
    typer.echo(f"question: {result['question_id']} manifest={result['manifest_hash'][:12]}")
    typer.echo(f"verdict: {result['verdict']}")
    if result.get("reason"):
        typer.echo(f"reason: {result['reason']}")
    for comparison in result.get("comparisons", []):
        typer.echo(
            f"[call] {comparison['tool']} {comparison['mart']} "
            f"recorded={comparison['recorded_rows']} pinned={comparison['pinned_rows']} "
            f"current={comparison['current_rows']} note={comparison['note']}"
        )


@app.command()
def eval(  # noqa: A002 - command name required by the eval spec
    questions: str = typer.Option(
        "agentic_warehouse_ops/evals/questions.yaml",
        "--questions",
        help="Path to the YAML question set.",
    ),
    out_dir: str = typer.Option(
        "artifacts/evals", "--out-dir", help="Root dir for the timestamped output."
    ),
    baseline: str | None = typer.Option(
        None, "--baseline", help="Path to a previous summary.json to diff against."
    ),
    max_regression: float = typer.Option(
        0.05, "--max-regression", help="Absolute score drop that fails the run."
    ),
    embedder: str = typer.Option(
        "fake", "--embedder", help="Embedder: fake (deterministic) or sentence-transformer."
    ),
) -> None:
    """Score the agent against the committed question set."""
    from agentic_warehouse_ops.agent.embedder import SentenceTransformerEmbedder
    from agentic_warehouse_ops.evals.run_eval import run_eval

    resolved_embedder = (
        SentenceTransformerEmbedder() if embedder == "sentence-transformer" else None
    )
    summary, regressions = run_eval(
        questions_path=questions,
        out_dir=out_dir,
        baseline=baseline,
        max_regression=max_regression,
        embedder=resolved_embedder,
    )
    typer.echo(f"overall: {json.dumps(summary['overall'])}")
    if baseline:
        if regressions:
            for regression in regressions:
                typer.echo(f"REGRESSION: {regression}", err=True)
            raise typer.Exit(code=1)
        typer.echo("baseline comparison: no regressions")


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
