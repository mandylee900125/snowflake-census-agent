# Decisions log

Running log kept while building. Each entry: what was chosen, why, and what
was rejected. This is the source material for the reflection and for the
follow-up review.

---

**Python + Streamlit, not TypeScript + Next.js.**
I write JavaScript daily, so this was a real tradeoff. Streamlit gives a
multi-turn chat UI, session state, and one-click deploy from a private repo in
roughly 40 lines. Next.js would have meant spending my scarcest hours on UI
plumbing instead of on the retrieval layer, which is what the brief actually
evaluates. Rejected: Next.js + Vercel, Gradio (weaker session-state story).

**Snowflake Marketplace share, not the SafeGraph CSV download.**
The share is queryable immediately and includes the metadata/crosswalk tables.
Loading CSVs would have cost hours of ETL before any agent code existed.

**Retrieval over the schema, not a hard-coded column subset.**
The central decision. ~7,500 columns cannot go in a prompt. Hard-coding 20
columns passes a demo and fails every question outside it. Retrieval is why
the agent can answer questions I never anticipated.

**BM25, not embeddings.**
Census column descriptions are short, keyword-dense, and full of exact terms
users type verbatim ("renter-occupied", "median household income"). Lexical
search handles those well, needs no embedding pipeline or vector store, adds
no per-query latency, and is trivially debuggable — I can see exactly why a
column was retrieved. The real cost: a question phrased entirely in synonyms
("how rich is this area?" → "median household income") can miss. A hybrid
BM25 + embedding retriever is the first thing I would add with more time.

**Column descriptions joined in from the metadata table before indexing.**
Without this, `B19013e1` is unsearchable — the code shares no tokens with any
question a human would ask. `test_description_enrichment_makes_coded_columns_findable`
pins this; it's the difference between retrieval working and not.

**Relevance gate is token overlap, not `score > 0`.**
My first cut dropped any candidate scoring ≤ 0. A test caught that this
silently discards correct matches: BM25 IDF goes negative for terms appearing
in most documents, so a right answer matched on a common word ("income")
scores at or below zero. Sign is an artefact of the weighting; overlap is the
actual signal. Now: require a content-word overlap, rank the survivors by
score.

**Two independent guardrails.**
The topic gate is an LLM call and is therefore defeasible. The SQL validator
is deterministic Python that never consults the model — single statement,
SELECT-only, keyword denylist, forced LIMIT. A fully prompt-injected model
still cannot reach a write. The validator masks string literals and quoted
identifiers before scanning, because Census column names are quoted free text
that legitimately contains keyword-like words
(`"Total: Renter-occupied housing units"`).

**Claude Opus 5 for SQL generation and answer synthesis.**
Text-to-SQL over an unfamiliar schema is the hard part; correctness matters
more than a few seconds. Effort set to `medium` to stay inside the latency
budget — raise to `high` if SQL quality needs it. One config constant.

**Claude Haiku 4.5 for the topic gate and question rewrite.**
Both are narrow classification tasks where the frontier model buys little, and
the brief explicitly asks for a *fast-fail* path for unanswerable questions.
This keeps a rejection at ~1s instead of ~8s. Configurable via
`GUARDRAIL_MODEL` if the accuracy tradeoff proves wrong.

**Structured outputs rather than JSON-in-prose + parsing.**
SQL generation returns a schema-validated object (`answerable`, `sql`,
`assumptions`, `clarification`, `explanation`). No regex extraction, no
retry-on-parse loop. It also gives the model a first-class way to decline:
`answerable: false` with a reason beats being forced to emit SQL for a
question the data cannot answer.

**One repair attempt, not a loop.**
A failed query is retried once with the error text fed back. Bounded because
an unbounded repair loop is how you blow the 60s budget and the token bill on
a question that was never answerable.

**Pipeline is a generator of events.**
`answer_question()` yields status/sql/data/token/message/error. The UI renders
progress as work happens (latency requirement), and tests assert on the event
stream with a scripted model and stubbed database — no network, no
credentials, ~0.1s for the full suite.

**Show the SQL in the UI.**
Ten minutes of work. A data agent whose reasoning can't be audited is not one
you hand to a customer.
