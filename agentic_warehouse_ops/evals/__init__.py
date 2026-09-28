"""Evaluation harness for the agent.

``awo eval`` runs the committed question set (``questions.yaml``) through the
full agent stack — typed tools, guardrails, audit — using the deterministic
scripted FakeLLM, scores the answers, and writes per-question JSONL plus a
markdown summary to ``artifacts/evals/<timestamp>/``.
"""

from agentic_warehouse_ops.evals.run_eval import run_eval

__all__ = ["run_eval"]
