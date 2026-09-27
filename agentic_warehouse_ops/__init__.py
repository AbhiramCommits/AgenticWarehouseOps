"""agentic_warehouse_ops: land, transform, govern, and query retail-ops data.

The platform flows data through four stages: raw files land in object storage,
Airflow copies them into a warehouse raw schema, dbt builds staging and mart
models, and a governance layer gates a tool-calling LLM agent that answers
questions over the governed marts.
"""

__version__ = "0.1.0"
