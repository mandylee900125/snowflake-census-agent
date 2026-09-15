# US Census Chat Agent

A chat agent that answers natural-language questions about US demographics,
grounded in SafeGraph's US Open Census Data (ACS 2019 5-year estimates, census
block group level) served from Snowflake.

**Live demo:** _(paste your Streamlit URL here)_
**Credentials:** _(none — the app is public)_

---

## The core problem

The dataset has **thousands of columns**, most named with opaque ACS codes
(`B19013e1`) whose meaning lives in a separate field-descriptions table. That
single fact drives the whole architecture:

- The schema **cannot** fit in a prompt, so hard-coding a column subset would
  answer demo questions and fail everything else.
- Column names alone are **not searchable** — nothing about `B19013e1` matches
  "median household income".

So the agent builds a searchable index of every column, enriched with its human
description from the dataset's own metadata table, and retrieves the ~20
relevant columns per question. The model sees only those.

## Architecture

```
  user turn
      │
  1.  guardrail ─── off-topic / unsafe ──▶ refuse (~1s, no DB, no SQL)
      │
  2.  rewrite with history ("what about Queens?" → full question)
      │
  3.  retrieve schema ── BM25 over all columns ──▶ top ~20
      │
  4.  generate SQL ── model sees only those columns; may return
      │                answerable:false with a reason or a clarifying question
  5.  validate ── deterministic: SELECT-only, single statement, forced LIMIT
      │
  6.  execute ── read-only session, row cap, statement timeout
      │           └─ on error: one repair attempt with the error text
  7.  answer ── streamed, grounded only in the returned rows
```

| Module | Responsibility |
|---|---|
| `src/census_agent/schema_index.py` | BM25 retrieval over enriched column metadata |
| `src/census_agent/guardrails.py` | SQL validator (deterministic) + topic gate (LLM) |
| `src/census_agent/pipeline.py` | Orchestration; emits progress/SQL/data/token events |
| `src/census_agent/llm.py` | Anthropic wrapper — models, streaming, error mapping |
| `src/census_agent/snowflake_client.py` | Connection reuse, read-only execution |
| `src/census_agent/prompts.py` | All prompts and output schemas, in one file |
| `app.py` | Streamlit chat UI |

**Two guardrails, deliberately independent.** The topic gate is an LLM call
and can in principle be talked around. The SQL validator is pure Python that
never asks the model anything — so even a fully prompt-injected model cannot
reach a write. `tests/test_sql_validator.py` is the densest file in the suite
for that reason.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in Anthropic + Snowflake credentials

python scripts/explore_schema.py   # see what the share actually contains
python scripts/build_index.py      # build schema_index.json (once)
streamlit run app.py
```

`schema_index.json` is built once and committed, so the deployed app never
introspects thousands of columns on boot.

```bash
pytest          # 64 tests, no network or credentials required
```

## Deployment

Streamlit Community Cloud, from this repo, `app.py` as the entrypoint.
Credentials go in **App settings → Secrets** using the key names in
`.streamlit/secrets.toml.example`.

## Interpretations of ambiguous requirements

The brief left several things open. These are the calls made, and why:

1. **"Conversation context" = the agent resolves follow-ups, not that it
   remembers across sessions.** Each turn rewrites the question to stand alone
   using the last few turns; history is per-browser-session and not persisted.
   Cross-session memory would need user identity, which the brief does not ask
   for.

2. **Answers are grounded strictly in returned rows.** The agent will not fill
   a gap from the model's own knowledge of US demographics, even when it
   plausibly could. A wrong-but-confident number is worse than "I can't
   answer that."

3. **"Guardrails" means both topic and safety.** Off-topic questions are
   refused, and generated SQL is independently validated as read-only. Refusal
   of on-topic-but-unanswerable questions is handled downstream with a specific
   reason rather than a flat refusal.

4. **Aggregation above block-group level is done in SQL.** Questions about a
   city or county sum the constituent block groups via the crosswalk tables
   rather than returning raw CBG rows.

5. **The SQL is shown to the user.** Not requested, but a data agent whose work
   can't be checked isn't one you'd hand to a customer.

6. **Latency budget.** The 60s cap is met by a cheap fast-fail gate, retrieval
   instead of a giant prompt, and a streamed final answer so the interface
   never looks hung.

## Known limits

Recorded honestly in [REFLECTION.md](REFLECTION.md). The short version: all
figures are 2019 ACS estimates; margin of error is not surfaced; retrieval is
lexical, so a question phrased entirely in synonyms of the column description
can miss.
