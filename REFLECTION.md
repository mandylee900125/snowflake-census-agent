# Reflection

Written at the end of the 24 hours. It covers the four things the brief
asked for: how it was built and the decisions that mattered, what I'd do
with more time, edge cases found but not fully fixed, and testing.

---

## 1. How it was built, and the decisions that mattered

**The problem.** The Census dataset has more than 17,000 columns with coded
names such as `B19013e1`. Sending the entire schema to the AI for every
question would be noisy and inefficient, so the main problem was finding a
small set of relevant columns before asking Claude to write SQL.

**How I built it.** I first explored the real Snowflake data to understand
how the Census tables and metadata were structured. That showed me that my
initial assumptions about the data were not completely correct, so the
retrieval approach had to be adapted to the real schema.

The app then follows a simple pipeline:

```
user question → gate → schema search → Claude writes SQL → SQL safety check
             → Snowflake runs the query → Claude explains the result
```

Claude Code did most of the implementation. My role was setting up Snowflake
and the deployment, testing the system against the real data, reviewing the
proposed decisions, asking for problems to be audited, and deciding which
trade-offs to accept.

**The decisions that mattered.**

1. **Search the full schema instead of hard-coding columns.** The app builds
   a searchable index from the Census metadata. For each question, it
   retrieves roughly 20 relevant columns and gives those to Claude. This lets
   the app work across the full dataset instead of only supporting a few
   demo questions.

2. **Search table first, then column.** A flat search sometimes returned too
   many columns from one Census table. The search was changed to first
   identify relevant tables and then relevant columns within those tables,
   with a cap on how many results can come from one table.

3. **Use BM25 keyword search instead of embeddings for the first version.**
   Census metadata uses fairly specific terminology, so keyword search was
   fast, simple, and easy to inspect. The first AI step can also add
   vocabulary hints such as mapping "women" to "Female." I would consider
   adding embeddings later for questions that use very different wording.

4. **Use multiple safety layers.** An AI gate filters obviously unrelated or
   off-topic questions. Python then validates generated SQL before
   execution, execution has limits and timeouts, and the Snowflake
   connection uses a read-only role. The goal was not to rely on the AI
   alone for safety.

5. **Track an overall time budget.** The pipeline tracks the time spent on
   each request so it stays within the 60-second requirement.

---

## 2. What I'd improve with more time

1. **A bigger set of test questions with checked answers.** I would expand
   the current test set so more types of questions are compared against
   known-correct results.

2. **A consistent approximation of county medians.** Right now the AI is
   instructed how to approximate county medians, but different phrasings
   could produce slightly different calculations. I would move that logic
   into code so the same method is used every time.

3. **Embeddings alongside keyword search.** BM25 works well when the user's
   wording overlaps with the Census labels. Embeddings could help when the
   user uses very different wording.

---

## 3. Edge cases and failure modes found but not fully fixed

- **Wrong-but-believable numbers.** The SQL can run successfully but still
  use the wrong Census field or geography. Live tests catch this only for
  questions we already tested. With more time, I would add a second check
  comparing the generated SQL back to the user's question.

- **Ambiguous place names.** For place names like "Springfield," the app may
  need to ask the user which location they mean.

- **Places spanning counties.** For places like New York City, the app may
  need clarification or must clearly state which county or borough it used.

---

## 4. Testing

**Unit tests.** Claude Code generated most of the test implementation, while
I reviewed what was being tested and used the results to find and fix
problems. The unit tests use a fake AI and fake database to test the Python
logic without network calls, including SQL safety, schema search, pipeline
behavior, retries, and error handling.

**Search checks.** I used 18 test questions to verify that the correct Census
columns appeared in the top search results. This helped measure whether
retrieval changes actually improved the system.

**Live tests.** I also ran 17 questions against the real Claude API and
Snowflake database. These checked that the full system produced the expected
SQL structure and known-correct results. One test was inconsistent because
Claude sometimes omitted an ORDER BY for a "highest" question, so I chose to
tighten the prompt and make the ORDER BY requirement explicit rather than
weakening the test.

**Trade-off.** Unit tests are fast and free, but they cannot catch changes in
AI behavior. Live tests can, but they cost money and can be somewhat
non-deterministic.

With more time, I would expand the set of live questions with more
hand-checked answers.

---

## 5. Self-assessment

**Weakest part.** County-level medians are approximations rather than exact
Census medians, so this is an area I would improve with more time.

**Most confident in.** Failure handling. During testing, the app returned
readable error messages instead of crashing when something went wrong.

**What I'm still learning.** Claude Code wrote most of the Python
implementation, so I am still building deeper familiarity with some of the
lower-level details. I understand the overall architecture, the major
trade-offs, and how the components work together, but I would not claim I
could rewrite every module from scratch without assistance.
