# US Census Chat Agent

A chat agent that answers natural-language questions about US demographics,
grounded in SafeGraph's US Open Census Data on the Snowflake Marketplace: ACS
2020 and 2019 5-year estimates plus the 2020 decennial count, at census block
group level (242,335 block groups, 17,052 columns across 73 tables).

**Live demo:** https://mandy-census-agent.streamlit.app
**Credentials:** none — the app is public. If it has been idle for a few days
Streamlit shows a "wake up" button; click it and allow about a minute.

---

## The core problem

The dataset has **17,052 columns**, named with opaque ACS codes (`B19013e1`)
whose meaning lives in a separate field-descriptions table. Two things follow:

- Sending the full schema on every request would be large, expensive, and
  mostly irrelevant to the question — and hard-coding a column subset would
  answer demo questions and fail everything else.
- Column names alone are **not searchable** — nothing about `B19013e1` matches
  "median household income".

So the agent builds a searchable index of every variable, enriched with its
human description from the dataset's own metadata table, and retrieves
roughly 20 relevant columns for each question. The model sees only those,
plus a short static block of rules retrieval cannot find (how block-group ids
encode state and county, how to join the FIPS name table, that medians can't
be summed).

## Architecture

```
  user turn
      │
  1.  guardrail ─── off-topic / unsafe ──▶ refuse (~1.5s, no DB, no SQL)
      │   the same cheap call rewrites follow-ups ("what about Queens?" →
      │   full question) and suggests Census vocabulary ("women" → "female")
      │
  2.  retrieve schema ── two-level BM25 (ACS table + cell), capped per
      │                  table ──▶ top ~20 of 4,378 variables
      │
  3.  generate SQL ── model sees only those columns + static schema rules;
      │                may return answerable:false with a reason or a
      │                clarifying question
  4.  validate ── deterministic: SELECT-only, single statement, forced LIMIT
      │
  5.  execute ── read-only session, row cap, statement timeout
      │           └─ on error: one repair attempt with the error text
  6.  answer ── streamed, grounded only in the returned rows
```

| Module | Responsibility |
|---|---|
| `src/census_agent/schema_index.py` | Two-level BM25 retrieval over enriched column metadata |
| `src/census_agent/census_layout.py` | Everything specific to this share's shape: vintages, estimate/MOE pairs, description hierarchy, table priorities |
| `src/census_agent/guardrails.py` | The two guards: the Python SQL validator and the AI topic gate (`classify_question`) |
| `src/census_agent/pipeline.py` | Orchestration; emits progress/SQL/data/token events |
| `src/census_agent/llm.py` | Anthropic wrapper — models, streaming, error mapping |
| `src/census_agent/snowflake_client.py` | Connection reuse, read-only execution |
| `src/census_agent/prompts.py` | All prompts and output schemas, in one file |
| `app.py` | Streamlit chat UI |
| `scripts/explore_schema.py` | Dump what the share contains — run first |
| `scripts/build_index.py` | Build `schema_index.json`; fails if any of 18 probe questions misses its column |
| `scripts/ask.py` | Run the agent from a terminal with per-stage timings |

**Defense in depth — four layers, each assuming the one above it failed.**

1. **Topic gate** (LLM, Haiku). Cheap and fast, and in principle it can be
   talked around — so nothing below trusts it.
2. **SQL validator** (pure Python, no model). Allows only a single read-only
   query, blocks write operations and administrative functions, requires a
   `LIMIT` on the outer query, and restricts queries to the Census database.
   Testing found that a Snowflake administrative function (`SYSTEM$...`)
   could bypass a simple keyword check, which led to strengthening the
   validator and adding the read-only role. `tests/test_sql_validator.py` is
   the densest file in the suite.
3. **Execution limits.** A per-statement timeout that shrinks as the turn
   ages, and a row cap.
4. **Least privilege.** The deployed app connects with a read-only Snowflake
   role (`CENSUS_READER`, created by `scripts/create_readonly_role.sql`). If
   every layer above failed, the session still cannot write.

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
pytest              # 113 unit tests, no network or credentials, ~2s
pytest -m live -v   # 17 live tests against the real model + database (~$1, ~5 min)
python scripts/ask.py "median rent in Austin?" "what about Dallas?"   # one-off, with timings and cost
```

The unit suite uses a scripted model and a stubbed database, so it checks
*our* orchestration, validation and error handling for the cases covered.
The live suite is the only thing that checks the model's behaviour on this
schema: it asserts on
structure (which stage answered, which tables the SQL touched, whether a
number is in a plausible range against ground truth read from the share)
rather than wording, so it survives non-determinism.

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

4. **Cities are approximated by their county, and the answer says so.** The
   share maps block groups to state and county only — there is no city, ZIP,
   or neighbourhood table despite the listing's wording. "Brooklyn" becomes
   Kings County, NY; the assumption is recorded and repeated in the answer.
   Neighbourhood questions get a clarifying question rather than a guess.

5. **2020 is the default year.** Both 2019 and 2020 ACS are indexed once per
   variable; the model uses 2020 unless asked for 2019, and every answer
   names the year it used.

6. **Latency budget.** The 60s cap is met by a cheap fast-fail gate, retrieval
   instead of a giant prompt, a time budget shared across the stages, and a
   streamed final answer so the interface never looks hung.

## Known limits

Recorded in [REFLECTION.md](REFLECTION.md). The short version:
county-level medians are approximations of block-group medians; margin of
error is not surfaced; city questions are answered at county level; retrieval
is lexical, so a phrasing with no word in common with the column label can
still miss despite the vocabulary expansion.
