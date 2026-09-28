# Changelog

## 0.1.0 — platform foundation (all layers)

**Foundation**
- uv-managed Python 3.11 package; dual-backend warehouse engines (DuckDB
  default, Snowflake via env) behind one interface; deterministic synthetic
  retail-ops generator (30 days, ~5k orders/day, controlled defects);
  boto3-only S3 path with local MinIO.

**Ingestion**
- Airflow 2.9 stack; idempotent delete-then-insert loader with pydantic
  validation and `_rejects` tables; run registry (`meta.pipeline_runs`,
  `meta.load_audit`); data-quality gate (floors, reject ceiling, empty
  partitions); `dags/ingest_raw.py`.

**Transformation & governance**
- dbt project (staging views, 6 marts) with dev/prod targets; generic tests
  on every model (PKs, relationships, accepted_values) plus singular tests;
  every column tagged `pii`/`classification`/`owner` with a
  `governance_meta` test enforcing it; lineage artifacts (JSON + Mermaid);
  PII guardrail source of truth; role-based grant DDL reapplied on every
  build; `dags/transform_and_govern.py`.

**Agent**
- Typed tool-calling LangGraph agent (no free-text SQL): `query_mart`,
  `vector_search_tickets`, `lookup_schema`; resumable hash-keyed embedding
  jobs behind one `VectorStore` interface; pluggable LLM with scripted
  FakeLLM; `AgentAnswer` audit payload; `awo ask` CLI and FastAPI `/ask`.

**Hardening**
- Guardrail middleware wrapping every tool call (PII denied twice, row/byte
  caps, cost budget, timeout, rate limits, sqlglot read-only check);
  `meta.agent_audit` / `meta.agent_questions` audit log written even on
  failure; manifest snapshots + registry + `awo replay` with
  IDENTICAL/DRIFTED/STALE_SCHEMA verdicts; JSON structured logging bound to
  run/question ids.

**Evals & CI**
- 50-question scored eval set (aggregation/filter/retrieval/schema/
  governance-refusal/multi-tool) with committed baseline; `awo eval` with
  baseline regression gate; GitHub Actions CI (ruff, ruff-format, mypy,
  pytest with 80% coverage gate, dbt build on seeded data, meta-completeness
  check, eval vs baseline) needing no cloud credentials.
