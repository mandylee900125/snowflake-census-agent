"""Orchestration: question in, grounded answer out.

Implemented as a generator of events so the UI can show progress while the
work happens. Each stage can end the turn early with a plain-language message
-- there is no path where the user gets a blank screen or a stack trace.

    guardrail (+ rewrite, search terms) -> retrieve schema -> generate SQL
              -> validate -> execute -> (repair once) -> answer
"""
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from . import config, guardrails, prompts, snowflake_client
from .llm import LLMUnavailable
from .schema_index import SchemaIndex

log = logging.getLogger(__name__)

MAX_SQL_ATTEMPTS = 2


def _event(kind, **kwargs):
    # type: (str, Any) -> Dict[str, Any]
    payload = {"type": kind}
    payload.update(kwargs)
    return payload


def _generate_sql(llm, question, schema_context, previous_error=None):
    # type: (Any, str, str, Optional[str]) -> Dict[str, Any]
    user = "Relevant columns:\n%s\n\nQuestion: %s" % (schema_context, question)
    if previous_error:
        user += (
            "\n\nYour previous query failed with this error:\n%s\n"
            "Write a corrected query, or set answerable to false if it cannot "
            "be fixed with the columns above." % previous_error
        )
    return llm.structured(
        system=prompts.SQL_SYSTEM,
        user=user,
        schema=prompts.SQL_SCHEMA,
        model=config.SQL_MODEL,
        effort=config.SQL_EFFORT,
        max_tokens=8000,
    )


def _query_timeout(started):
    # type: (float) -> int
    """Seconds a Snowflake query may take, given what the turn has spent."""
    remaining = (config.TURN_BUDGET_SECONDS - (time.monotonic() - started)
                 - config.ANSWER_RESERVE_SECONDS)
    return int(max(config.MIN_QUERY_TIMEOUT_SECONDS,
                   min(config.QUERY_TIMEOUT_SECONDS, remaining)))


def _explain_unanswerable(plan):
    # type: (Dict[str, Any]) -> str
    parts = [plan.get("explanation") or "I can't answer that from this dataset."]
    if plan.get("clarification"):
        parts.append(plan["clarification"])
    return " ".join(p for p in parts if p)


def answer_question(llm, index, question, history=None):
    # type: (Any, SchemaIndex, str, Optional[List[dict]]) -> Iterator[Dict[str, Any]]
    """Run one turn. Yields status / sql / data / token / message / error events."""
    history = history or []
    started = time.monotonic()

    # 1. Fast-fail gate, before any expensive work.
    yield _event("status", text="Checking the question")
    try:
        verdict = guardrails.classify_question(llm, question, history)
    except LLMUnavailable as exc:
        yield _event("error", text=str(exc))
        return

    if not verdict.should_answer:
        yield _event("message", text=verdict.reason or
                     "I can only answer questions about US Census demographic data.")
        return

    # 2. The gate also resolved follow-ups ("what about Queens?") into a
    #    standalone question; retrieval and SQL generation work on that.
    standalone = verdict.standalone_question or question
    if standalone != question:
        yield _event("status", text="Interpreting as: %s" % standalone)

    # 3. Retrieve the handful of columns that matter. The gate's Census-
    #    vocabulary terms are appended so "women" can find "Female".
    yield _event("status", text="Searching %d columns" % len(index))
    retrieval_query = " ".join(part for part in (standalone, verdict.search_terms) if part)
    schema_context = index.render_context(retrieval_query)

    # 4-6. Generate, validate, execute -- with one repair attempt.
    last_error = None  # type: Optional[str]
    columns = None  # type: Optional[List[str]]
    rows = None  # type: Optional[List[tuple]]
    plan = {}  # type: Dict[str, Any]
    safe_sql = None  # type: Optional[str]
    out_of_time = False

    for attempt in range(MAX_SQL_ATTEMPTS):
        if attempt > 0 and time.monotonic() - started > config.REPAIR_CUTOFF_SECONDS:
            # A repair is another model call plus another query. Past this
            # point it would blow the 60s budget; explaining beats hanging.
            log.warning("Skipping SQL repair: %.0fs already elapsed", time.monotonic() - started)
            out_of_time = True
            break
        yield _event("status", text="Writing SQL" if attempt == 0 else "Fixing the query")
        try:
            plan = _generate_sql(llm, standalone, schema_context, last_error)
        except LLMUnavailable as exc:
            yield _event("error", text=str(exc))
            return

        if not plan.get("answerable"):
            yield _event("message", text=_explain_unanswerable(plan))
            return

        try:
            safe_sql = guardrails.validate_sql(plan.get("sql") or "")
        except guardrails.UnsafeSQL as exc:
            last_error = str(exc)
            log.warning("Rejected generated SQL: %s", exc)
            continue

        yield _event("sql", sql=safe_sql)
        yield _event("status", text="Querying Snowflake")
        try:
            columns, rows = snowflake_client.run_select(
                safe_sql, timeout_seconds=_query_timeout(started)
            )
            break
        except snowflake_client.SnowflakeUnavailable as exc:
            log.exception("Snowflake unavailable")
            yield _event("error", text=(
                "I'm having trouble connecting to the Census database, so I "
                "can't look this up right now. The question itself was fine -- "
                "please try again in a moment."
            ))
            return
        except snowflake_client.QueryFailed as exc:
            last_error = str(exc)
            log.warning("Query failed on attempt %d: %s", attempt + 1, exc)
            continue

    if rows is None:
        text = ("I understood the question but couldn't build a working query for "
                "it against this dataset. The last problem was: %s" % (last_error or "unknown"))
        if out_of_time:
            text += (" I stopped retrying to stay within the time limit -- please "
                     "try again, or rephrase the question.")
        yield _event("message", text=text)
        return

    yield _event("data", columns=columns, rows=rows)

    # 7. Explain the rows -- and only the rows.
    yield _event("status", text="Writing the answer")
    answer_prompt = (
        "Question: %s\n\nSQL that was run:\n%s\n\nResults:\n%s\n\nAssumptions made: %s"
        % (
            standalone,
            safe_sql,
            prompts.render_results(columns, rows),
            "; ".join(plan.get("assumptions") or []) or "none",
        )
    )
    messages = [{"role": "user", "content": answer_prompt}]
    try:
        for token in llm.stream_text(
            system=prompts.ANSWER_SYSTEM,
            messages=messages,
            model=config.SQL_MODEL,
            effort=config.SQL_EFFORT,
        ):
            yield _event("token", text=token)
    except LLMUnavailable as exc:
        yield _event("error", text=(
            "I got the data but couldn't write the summary: %s You can still "
            "read the query results above." % exc
        ))
        return

    yield _event("done")
