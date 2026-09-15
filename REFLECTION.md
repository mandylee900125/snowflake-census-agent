# Reflection

> Skeleton written at the start so it accumulates honestly instead of being
> reconstructed at hour 23. **Fill the `TODO` markers before submitting** —
> and cut anything that didn't actually happen.

## Development process

I read the brief twice before writing code and spent the first hour on the
Snowflake trial and `scripts/explore_schema.py` — looking at what the share
actually contains rather than what I assumed. That was the right call: seeing
that most columns are ACS codes (`B19013e1`) with meanings in a separate
metadata table changed the design. Retrieval over column *names* would have
failed almost every question.

I kept `DECISIONS.md` as I went rather than reconstructing rationale
afterwards. It's the honest record of what I chose and what I rejected.

_TODO: how you actually used AI coding tools — what you delegated, what you
had to correct, where you disagreed with the generated code. They ask about
this directly in the review._

## Key architectural decisions

Full log in [DECISIONS.md](DECISIONS.md). The three that mattered most:

1. **Retrieval over the schema instead of a hard-coded column subset.** ~7,500
   columns can't go in a prompt. Hard-coding a subset passes a demo and fails
   everything else. This is why the agent handles questions I never
   anticipated.

2. **Two independent guardrails.** The topic gate is an LLM call and is
   defeasible; the SQL validator is deterministic Python that never consults
   the model. A fully prompt-injected model still cannot reach a write.

3. **Time bought where the brief grades.** I spent it on retrieval, failure
   handling, and tests, and deliberately not on UI. Streamlit's default chat
   look is plain — that was the trade.

## Where I invested, and what I deliberately left out

**Invested:** the schema retrieval layer, the SQL validator, graceful
degradation paths, and the test suite.

**Left out on purpose:**
- **Authentication.** Not required, and it would have cost an hour that
  retrieval needed.
- **Visualisation of results.** A raw table is honest and took minutes; charts
  would have been demo polish over correctness.
- **Query result caching.** Real cost saving in production, no effect on what's
  being evaluated.
- **Embedding-based retrieval.** See the known limits below — this is the first
  thing I'd add, not something I think was unnecessary.
- **Multi-table join reasoning beyond the crosswalk tables.** The agent handles
  block-group → city/county aggregation; more exotic joins are untested.

## What I'd do differently with more time

1. **Hybrid retrieval (BM25 + embeddings).** The clearest weakness. Lexical
   search misses questions phrased entirely in synonyms — "how rich is this
   area?" shares no tokens with "median household income". Roughly a half-day:
   embed the column descriptions once, blend the two ranked lists.

2. **An evaluation set with expected answers, not just expected columns.** Right
   now I test that retrieval surfaces the right column and that the pipeline
   degrades correctly. I don't test that the final number is *right*. I'd
   hand-verify 25–30 questions against the ACS and assert on values.

3. **Surface margin of error.** ACS figures are estimates with MOE columns
   sitting right next to them. Reporting a point estimate without its
   uncertainty is a real correctness gap, not a cosmetic one.

4. **Cost and latency instrumentation per stage.** I know the pipeline fits in
   60s; I can't currently tell you the p95 or which stage dominates.

5. **Revisit `effort: medium` on SQL generation.** Chosen for latency without
   measuring the quality cost. I'd sweep it against an eval set.

## Edge cases and failure modes I identified but did not fully address

- **Ambiguous place names.** "Springfield" matches ~30 places. The agent can
  ask for clarification, but it has no ranked list of candidates to offer, so
  the question comes back vaguer than it should.
- **Synonym-only questions.** Covered above — the known consequence of lexical
  retrieval.
- **Silent aggregation errors.** If the model sums block groups with a wrong
  crosswalk filter, the query succeeds and returns a plausible wrong number.
  Nothing currently catches this; it's the failure mode I'd most want a
  verification pass for.
- **Margin of error.** The prompt forbids summing an estimate with its MOE, but
  nothing enforces it.
- **Estimate vs. count confusion.** ACS values are survey estimates, not
  counts. The answer prompt says so; a user skimming may still read them as
  exact.
- **Very large result sets.** Capped at 500 rows, and only the first 50 reach
  the answer prompt. A question whose answer needs the whole tail gets a
  confidently incomplete summary.
- **Rate limiting under concurrent reviewers.** Several people opening the demo
  at once share one Anthropic key and one Snowflake warehouse. Handled as a
  clean error, not queued.
- _TODO: add anything you hit while building — especially things that broke._

## Testing approach

**What I test:** the code I wrote. 64 tests, no network or credentials
required, full suite in ~0.1s.

- `test_sql_validator.py` — the densest file, because it's the security
  boundary. Stacked statements, writes hidden behind comments, writes inside
  CTEs, LIMIT clamping, and the inverse case: quoted Census identifiers that
  *contain* keyword-like words must not be falsely rejected.
- `test_schema_index.py` — golden questions that must each retrieve a named
  column. If retrieval misses, nothing downstream can recover.
- `test_pipeline.py` — orchestration against a scripted model and stubbed
  database: off-topic refusal, prompt injection, unanswerable questions,
  ambiguity, database outage, query repair, repair budget exhaustion, and
  follow-up rewriting.

**The tradeoff:** scripting the model makes tests deterministic, free, and
fast — and means I am testing my orchestration, not the model's judgment. A
regression in SQL quality from a prompt change would not fail this suite.

**What I'd add:**
1. An end-to-end eval set with verified expected values (see above) — the
   biggest gap.
2. Adversarial input as a regression corpus — every injection attempt I tried
   by hand, asserted to never produce executable write SQL.
3. Retrieval quality metrics (recall@20 over a labelled question set) so
   retrieval changes are measurable rather than vibes.
4. A single live smoke test against real Snowflake and a real model, run
   manually before deploy. Everything else stays hermetic.

## Honest self-assessment

_TODO: written last, after you know what actually shipped. Be specific — name
the weakest part of the submission and why you accepted it. The brief says
incomplete-but-self-aware scores better than complete-but-unreflective, and
they mean it._

One concrete example to keep: my first cut of the retrieval relevance gate
dropped any candidate scoring ≤ 0, which silently discarded correct matches
because BM25 IDF goes negative for terms common across documents. A test
caught it. That's the kind of bug that looks fine in a demo and is wrong in
production.
