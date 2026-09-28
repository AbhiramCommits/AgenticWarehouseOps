# Deployment

## Warehouse: DuckDB -> Snowflake

Everything reads the same `Settings`/env vars and the same engine interface;
only configuration changes.

1. Copy `.env.example` to `.env` and fill the Snowflake block:

   ```bash
   SNOWFLAKE_ACCOUNT=xy12345.us-east-1
   SNOWFLAKE_USER=svc_airflow
   SNOWFLAKE_PASSWORD=...
   SNOWFLAKE_ROLE=loader
   SNOWFLAKE_WAREHOUSE=warehouse_ops_wh
   SNOWFLAKE_DATABASE=warehouse_ops
   ```

2. Create the warehouse and roles (run once, as ACCOUNTADMIN):

   ```sql
   CREATE WAREHOUSE warehouse_ops_wh WITH WAREHOUSE_SIZE = 'XSMALL';
   CREATE ROLE loader; CREATE ROLE analyst_ro; CREATE ROLE engineer_rw; CREATE ROLE pii_reader;
   GRANT ROLE loader TO ROLE SYSADMIN;
   GRANT USAGE ON WAREHOUSE warehouse_ops_wh TO ROLE loader;
   GRANT CREATE DATABASE, CREATE SCHEMA ON ACCOUNT TO ROLE loader; -- or pre-create DB/schemas
   ```

3. Switch the platform to Snowflake: `WAREHOUSE_PROFILE=snowflake` (engine)
   and `DBT_TARGET=prod` (dbt profile; the `prod` target reads the same env
   vars). The grant DDL is generated from the manifest meta by
   `governance/grants.py` and reapplied on every dbt build via the
   `on-run-end` hook: `analyst_ro` secure views (marts minus restricted),
   `engineer_rw` (all marts), `pii_reader` (marts incl. restricted).

## Object storage: MinIO -> AWS S3

All code talks to S3 through boto3 only; the endpoint is the single switch.

- Set `S3_ENDPOINT_URL=` (empty) and real `S3_ACCESS_KEY`/`S3_SECRET_KEY`,
  or leave all three unset to use the default AWS credential chain (env vars,
  `~/.aws/credentials`, or the instance IAM role).
- Create the bucket (`raw-landing`) and attach a least-privilege policy for
  the loader role:

  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["s3:ListBucket", "s3:GetObject", "s3:PutObject"],
        "Resource": [
          "arn:aws:s3:::raw-landing",
          "arn:aws:s3:::raw-landing/*"
        ]
      }
    ]
  }
  ```

- Encrypt at rest: enable SSE-KMS on the bucket (or default SSE-S3); no code
  change is needed — boto3 uses the bucket policy.
- Keep the bucket private; access goes through the loader role only.

## Orchestration: local Airflow -> MWAA / Astronomer

The DAGs import only `agentic_warehouse_ops` modules (heavy imports are lazy
inside task bodies), so they port unchanged.

- **MWAA**: deploy the repo via CI (DAGs -> the MWAA S3 bucket under `dags/`,
  package -> `plugins/` or a Python wheel in `requirements.txt`), set the same
  env vars in the MWAA environment, and add the same Airflow connections
  (Snowflake, S3) there.
- **Astronomer**: copy `dags/` and `agentic_warehouse_ops/` into the Astro
  project, add the Python deps to `requirements.txt`, and set env vars in
  `.env` / Deployment environment.
- Either way, `WAREHOUSE_PROFILE=snowflake` + `DBT_TARGET=prod` and the S3
  settings above are the only runtime config the DAGs need.

## Secrets handling

- **Never commit credentials.** Code reads config exclusively through
  `Settings` (env vars / local `.env`); `.env` is gitignored and
  `.env.example` is the documented template.
- Local dev: `.env` only.
- Production: store secrets in **AWS Secrets Manager** (or your vault) and
  inject them as env vars at task runtime, or use **Airflow connections**
  (Snowflake/S3) referenced by the DAGs instead of env vars.
- The airflow image contains no baked-in credentials; MinIO defaults
  (`minioadmin/minioadmin`) exist only in the local compose file.

## Cost control (agent)

The guardrails double as cost controls, tuned in `GuardrailConfig`:

| knob | default | effect |
| --- | --- | --- |
| `max_rows` | 1000 | hard LIMIT on every `query_mart` |
| `max_result_bytes` | 256 KiB | context never balloons |
| `max_bytes_scanned` | 64 MiB/question | per-question credit budget; abort with partial answer |
| `timeout_seconds` | 120 s | wall-clock cap per question |
| `max_tool_calls` | 25 | loop amplification is impossible |
| `max_questions_per_minute` | 60 | global throughput ceiling |

LLM spend is bounded by construction: at most 25 tool calls plus two LLM
round-trips per question, and the model only ever sees capped, redacted
evidence.
