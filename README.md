# agentic-warehouse-ops

A data platform that lands raw retail-ops files from S3 into a warehouse, transforms them with dbt, governs PII, and exposes a guardrailed tool-calling agent that answers questions over the governed marts. DuckDB is the default warehouse so everything runs locally for free; Snowflake and real AWS S3 are env-var switches. Every layer — ingestion, dbt, agent — writes to one auditable registry, and the agent can only ever execute typed, parameterised reads.

```
               ┌─────────────┐
               │  S3 / MinIO │
               └──────┬──────┘
                      │  boto3 (landing)
               ┌──────▼──────┐
               │   Airflow   │
               └──────┬──────┘
                      │  COPY
               ┌──────▼──────┐
               │  raw schema │
               └──────┬──────┘
                      │  dbt
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────────┐       ┌──────────────────┐
│  staging models   │       │      marts       │
└─────────┬─────────┘       └────────┬─────────┘
          └────────────┬─────────────┘
                       ▼
           ┌──────────────────────┐
           │  governance manifest │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │   LangGraph agent    │
           │  (tool-calling LLM)  │
           └──────────────────────┘
```

## Quickstart (under ten commands)

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
git clone <this-repo> && cd agentic-warehouse-ops
make install          # 1. uv sync — locks and installs everything
make seed             # 2. deterministic sources -> data/seeds (2026-08-28..09-26, clean)
make minio-up         # 3. local MinIO + raw-landing bucket
make minio-seed       # 4. upload data/seeds -> s3://raw-landing
make airflow-up       # 5. webserver + scheduler + postgres (login airflow/airflow)
# 6. run the pipeline for one day:
docker compose -f docker/docker-compose.yml --profile airflow exec airflow-scheduler \
  airflow dags trigger -e 2026-08-28T00:00:00 ingest_raw
# 7. once ingest finishes, transform + embed + snapshot:
docker compose -f docker/docker-compose.yml --profile airflow exec airflow-scheduler \
  airflow dags trigger -e 2026-08-28T00:00:00 transform_and_govern
# 8. ask the agent a question:
uv run awo ask "How many orders were cancelled on 2026-08-30?"
```

Steps 2-7 are also available offline as `make eval-warehouse` (loads the seeds
straight into DuckDB, no S3/Airflow) — that is what CI runs.

## Evaluation

`make eval` runs the committed 50-question set through the full agent stack
(typed tools, guardrails, audit) with the deterministic FakeLLM — no API key,
no network. Results land in `artifacts/evals/<timestamp>/` (JSONL + summary.md
+ summary.json), and `--baseline` fails CI on regressions.

Committed baseline (`artifacts/evals/baseline/summary.json`, run
`make eval` to verify):

| metric | value |
| --- | --- |
| answer accuracy | 1.0 |
| tool selection (exact / set) | 1.0 / 1.0 |
| refusal precision / recall | 1.0 / 1.0 |
| mean latency | 108.8 ms |
| mean tool calls | 1.16 |
| mean bytes scanned / question | 117,320 |

| category | n | accuracy |
| --- | --- | --- |
| aggregation | 10 | 1.0 |
| filter | 10 | 1.0 |
| retrieval | 10 | 1.0 |
| schema | 6 | 1.0 |
| governance-refusal | 6 | 1.0 |
| multi-tool | 8 | 1.0 |

## Governance and guardrails

Every column carries `meta: {pii, classification, owner}` (see
`docs/governance.md` for the scheme, role matrix, and guardrail list). The
agent has no free-text SQL parameter anywhere; every tool call passes through
`GuardrailSession`, which denies PII twice (at args and on results), injects a
hard `LIMIT 1000`, caps result bytes, enforces a per-question cost budget and
timeout, rate-limits, and sqlglot-verifies that every query is exactly one
`SELECT`. Denials are audited — here is a real one:

```bash
uv run awo ask "List customer emails for churned accounts."
```

```text
[tool] query_mart({"mart": "dim_customer", "select": ["email"], "filters": [], ...}) -> 0 rows ERROR: column 'dim_customer.email' is PII and cannot be queried
I cannot provide that information: the requested column is restricted PII.
```

The corresponding audit row (reproduce with `uv run awo audit --last 20`):

```text
tool_name: query_mart
guardrail_verdict: denied: pii: column 'dim_customer.email' is PII and cannot be queried
rows_returned: 0
caller_role: analyst
llm_model: fake-scripted
```

## Reproducibility

Every transform run snapshots its dbt manifest to
`artifacts/manifests/<hash>.json` and registers it in `meta.manifest_registry`;
every answer records the manifest hash it ran against. Replay a recorded
question against its pinned manifest:

```bash
uv run awo replay --question-id 871573a90d544238bb572cc19138a2b7
```

```text
question: 871573a90d544238bb572cc19138a2b7 manifest=bd925c067036
verdict: IDENTICAL
[call] query_mart dim_customer recorded=5 pinned=5 current=5 note=None
```

A `DRIFTED` verdict lists per-call row diffs; `STALE_SCHEMA` means a pinned
model no longer exists.

## Layout

```
agentic_warehouse_ops/        the Python package
├── ingestion/                generator, S3 loader, run registry, quality gate
├── governance/               catalog/lineage, PII guardrails, role grants
├── agent/                    typed tools, LangGraph graph, guardrails, audit
├── evals/                    questions.yaml + eval runner + offline warehouse
└── common/                   warehouse adapters, settings, S3, reproducibility
dags/                         ingest_raw + transform_and_govern
dbt/                          staging views + marts, schema.yml meta, tests
docker/                       MinIO + Airflow 2.9 stack
docs/                         governance.md, deploy.md
tests/                        pytest suite (36 tests, 80% coverage gate)
.github/workflows/ci.yml      lint, typecheck, tests, dbt build, eval baseline
```

## Deployment

Switching backends (DuckDB -> Snowflake, MinIO -> AWS S3, Airflow -> MWAA) and
secrets handling are covered in [`docs/deploy.md`](docs/deploy.md).

## Development

```bash
make test          # pytest + coverage
make lint          # ruff check
make typecheck     # mypy
make dbt-build     # dbt build (run + tests; the quality gate)
make eval          # awo eval (offline, deterministic)
```

CI runs all of the above plus `dbt build` on seeded data and the eval against
the committed baseline — with no cloud credentials.
