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

---

## After connecting to the real share

**Explore before building.** `scripts/explore_schema.py` ran before any
retrieval code was adapted. What it found changed the design: two ACS
vintages (2019 and 2020) with identical structure, ~30 tables per vintage
split by ACS table family, estimate/margin-of-error column pairs, a 10-level
description hierarchy, a separate 2020 decennial table, and a state/county
FIPS crosswalk — but no city, ZIP, or neighbourhood table. The scaffold had
assumed one vintage and a flat code→description map.

**One index document per variable, not per table copy.**
The 2019 and 2020 tables share every column code. Indexing both would hand
the model duplicate candidates and halve the useful context. Each document
records which vintages carry it and defaults to the newest; the prompt tells
the model the 2019 sibling table exists. Rejected: indexing only 2020 (loses
the 2019 data the listing includes) and indexing both (duplicates).

**Estimates only; margin-of-error columns reachable by rule.**
`B19013m1` shares its description with `B19013e1` and would pair with it in
every result, costing half the candidate slots. The SQL prompt states the
e→m naming rule instead. Comprehensive mapping is preserved — nothing is
unreachable — without the noise.

**Two-level retrieval: table score + cell score, with a per-table cap.**
ACS data is a set of ~365 tables, each a grid of cells that share most of
their words. A flat column index let a 23-column table ("Tenure By Units In
Structure") flood the top 20 for "renter occupied housing units" and push
the plain `Tenure → Renter occupied` column out entirely. Scoring the table
and the cell separately, adding them, and capping any one table at 5 slots
fixed it — 8/8 probes → the right column in the top 20. This is how a Census
analyst works: pick the table, then the cell.

**Probes as a build gate.** `build_index.py` runs 18 reviewer-style
questions and exits non-zero if any expected column misses the top 20. Every
retrieval change today was evaluated against that list rather than by eye.
Four of my first "expected" codes turned out not to exist in this share
(SafeGraph includes a subset of ACS tables) — a useful reminder that the
share, not my memory of the ACS, is the ground truth.

**Retrieval bugs the probes caught, in order:**
1. *No stemming.* "unemployment rate" returned nothing because the column
   says "Unemployed". Added a deliberately crude suffix stripper applied to
   both corpus and query. Snowball is the upgrade if it proves too blunt.
2. *Repeated words inflated scores.* "Total Fields Of Bachelor's Degrees
   [universe: TOTAL BACHELOR'S DEGREE MAJORS…]" counted "bachelor" twice.
   Switched the column index to one occurrence per term (binary TF): Census
   labels are short and repetition is structural, not informative.
3. *Table topics leaked into every cell.* The "Race" table's topic list names
   every race, so the tokens for `White alone` contained "asian" and it tied
   with `Asian alone` for "asian population". Topics now live only in the
   table-level index.
4. *Stopwords scored.* "are" is rare in Census labels and therefore high-IDF;
   "People who *are* White alone" beat "Total > Female" for "how many women
   are there". Stopwords are dropped from the query before scoring.
5. *Negative IDF inverted the priority penalty.* BM25 gives a negative IDF to
   any term in more than half the documents. A 0.3× priority on a negative
   score makes it *less* negative — the penalty became a boost. IDF is now
   floored at a small positive value, which also fixes the earlier
   observation that correct matches on common words scored below zero.

**Priorities for whole table classes.** B99xxx "Allocation Of…" tables are
imputation-quality flags that share every word with the real table; they
rank at 0.3×. Race-iteration tables (`B19013A`…`I`, "(White Alone)") rank at
0.8× so the base table wins ties unless the question names a race. Both are
still indexed — the foreign-born question was in fact answered correctly
from an allocation table's parent row, which is the argument for penalising
rather than excluding.

**Query expansion folded into the topic gate.** "women" matches no Census
label; "female" does. The gate call already sees the history and the
message, so it now also returns the standalone question and 3–8 words of
Census vocabulary, which are appended to the retrieval query. One fewer
model call per turn than the original separate rewrite step, and synonym
help on every question rather than only follow-ups. Rejected: embeddings
(a second retrieval system to build and explain, for a problem a one-line
prompt addition solved); running the rewrite separately on every turn
(+~1s per question for nothing).

**Static schema notes in the SQL prompt.** Retrieval cannot be trusted to
surface a *join rule* — nothing in "median income in Brooklyn" matches the
FIPS crosswalk table. So the CBG id structure, the FIPS join, the geography
lookup, the vintage rule, the median-of-medians caveat, and the four
universal denominator columns (total population, households, occupied and
total housing units) are always in the prompt. Everything else is
retrieved.

**City questions are answered at county level, and say so.** The share has
no city table. The model maps a city to its county from world knowledge
(Seattle → King County) and records it as an assumption that the final
answer repeats. My first wording of this in the shared dataset description
("cannot answer exact city boundaries") made the topic gate refuse a
landing-page example outright — geography is now explicitly the SQL stage's
decision, and the gate is told never to reject on geography.

**The e1 cell of any listed table is fair game.** Every ACS table's first
cell is its universe total. The model was reluctant to use `B15003e1`
(population 25+) as a denominator because it wasn't in the retrieved list,
and computed an attainment rate over all ages instead. Now stated as a rule.

**Commit `schema_index.json`.** 1.6 MB. The alternative — building it on
boot — puts a 13-second Snowflake introspection in front of the first
question on every cold start of the deployed app, and makes deployment
depend on the metadata tables being reachable at that moment.

**`scripts/ask.py`.** A terminal front-end to the same pipeline, with
per-stage timings. Every behaviour above was verified with it against the
real model and database before touching the UI.
