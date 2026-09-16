"""Two independent guardrails.

  validate_sql()      -- deterministic. No LLM, no network. This is the one
                         that has to be right, so it is pure and unit-tested.
  classify_question() -- an LLM topic gate, the cheap fast-fail path for
                         off-topic and adversarial input.

They are separate on purpose: a prompt-injected model must still not be able
to run a DELETE, because the validator never asks the model anything.
"""
import re
from typing import List, Optional, Tuple

from . import config

# Anything that writes, changes session state, or reaches outside the query.
FORBIDDEN_KEYWORDS = frozenset([
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT",
    "DROP", "CREATE", "ALTER", "TRUNCATE", "RENAME",
    "GRANT", "REVOKE", "USE", "SET", "UNSET",
    "COPY", "PUT", "GET", "REMOVE", "LIST",
    "CALL", "EXECUTE", "EXEC",
])

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SINGLE_QUOTED = re.compile(r"'(?:[^']|'')*'")
_DOUBLE_QUOTED = re.compile(r'"(?:[^"]|"")*"')
_LIMIT_CLAUSE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
# SYSTEM$ABORT_SESSION, SYSTEM$CANCEL_ALL_QUERIES, ... are administrative but
# callable from a plain SELECT, so a keyword denylist alone does not stop them.
_SYSTEM_FUNCTION = re.compile(r"\bSYSTEM\$", re.IGNORECASE)

# A read-only role is not a confined one. Verified on the real account: under
# CENSUS_READER the session could still SELECT from
# SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY (other people's query text) and the
# sample database, because every role inherits PUBLIC. So the validator pins
# every fully-qualified name to the configured database and refuses the
# account-metadata schemas outright. Unqualified names resolve to the session
# database, which the connection sets.
FORBIDDEN_SCHEMAS = frozenset([
    "INFORMATION_SCHEMA", "ACCOUNT_USAGE", "READER_ACCOUNT_USAGE",
    "ORGANIZATION_USAGE", "DATA_SHARING_USAGE", "SNOWFLAKE_SAMPLE_DATA",
])
# db.schema.object -- identifiers may be quoted; the view has literal
# contents blanked but keeps the quote characters, so quoted names survive.
_THREE_PART = re.compile(
    r'(?:"([^"]*)"|([A-Za-z_][A-Za-z0-9_$]*))\s*\.\s*'
    r'(?:"([^"]*)"|([A-Za-z_][A-Za-z0-9_$]*))\s*\.\s*'
    r'(?:"([^"]*)"|([A-Za-z_][A-Za-z0-9_$]*))'
)


class UnsafeSQL(ValueError):
    """The generated SQL is not a safe read-only query."""


def _blank(match):
    # type: (re.Match) -> str
    """Replace a match with spaces of the same length, keeping the outer
    quotes of a literal so the SQL still parses. Length-preserving so that
    positions found in the masked text apply to the original."""
    text = match.group(0)
    if text[0] in "'\"" and len(text) >= 2:
        return text[0] + " " * (len(text) - 2) + text[0]
    return " " * len(text)


def _mask_literals(sql):
    # type: (str) -> str
    """Blank out string literals and quoted identifiers before scanning.

    Census column names are quoted free text -- "Total: Renter-occupied
    housing units" -- and a naive keyword scan would flag legitimate
    identifiers containing words like GET or SET. Literals are masked
    BEFORE comments are stripped, so a literal containing -- survives.
    """
    sql = _SINGLE_QUOTED.sub(_blank, sql)
    sql = _DOUBLE_QUOTED.sub(_blank, sql)
    return sql


def _strip_comments(sql):
    # type: (str) -> str
    """Blank comments (length-preserving). Call on masked text only."""
    return _BLOCK_COMMENT.sub(_blank, _LINE_COMMENT.sub(_blank, sql))


def _analysis_view(sql):
    # type: (str) -> str
    """The text we scan: literals masked, then comments blanked, same length as `sql`."""
    return _strip_comments(_mask_literals(sql))


def validate_sql(sql, max_rows=None):
    # type: (str, Optional[int]) -> str
    """Return safe-to-execute SQL, or raise UnsafeSQL.

    Guarantees on the returned string:
      * exactly one statement
      * it is a SELECT (or a WITH ... SELECT)
      * it contains no data-modifying or session-modifying keyword, and no
        SYSTEM$ administrative function
      * every fully-qualified name is in the configured database, and no
        account-metadata schema (ACCOUNT_USAGE, INFORMATION_SCHEMA, ...) is
        referenced
      * its outermost query has a LIMIT no larger than max_rows

    The returned SQL is the model's text with only a trailing semicolon
    removed and the LIMIT adjusted -- comments and literals are executed
    as written; all scanning happens on a masked copy of the same length.
    """
    max_rows = max_rows or config.MAX_ROWS

    if not sql or not sql.strip():
        raise UnsafeSQL("The model returned an empty query.")

    body = sql.strip().rstrip(";").rstrip()
    view = _analysis_view(body)

    # One statement only. A second statement is a classic injection shape
    # (SELECT 1; DROP TABLE x). Semicolons inside literals are masked.
    if [part for part in view.split(";") if part.strip()][1:]:
        raise UnsafeSQL("Only one SQL statement may be executed at a time.")

    first = view.split(None, 1)
    first_word = first[0].upper() if first else ""
    if first_word not in ("SELECT", "WITH"):
        raise UnsafeSQL(
            "Only SELECT queries are allowed; this one started with '%s'." % (first_word or "nothing")
        )

    words = set(re.findall(r"[A-Za-z_]+", view.upper()))
    banned = sorted(words & FORBIDDEN_KEYWORDS)
    if banned:
        raise UnsafeSQL("Query contains disallowed keyword(s): %s." % ", ".join(banned))
    if _SYSTEM_FUNCTION.search(view):
        raise UnsafeSQL("Query calls a SYSTEM$ administrative function.")
    _check_scope(body, view)

    return _enforce_limit(body, view, max_rows)


def _check_scope(sql, view):
    # type: (str, str) -> None
    """Every db.schema.object reference must name the configured database,
    and no reference may touch an account-metadata schema."""
    if words_in(view) & FORBIDDEN_SCHEMAS:
        raise UnsafeSQL("Query references an account-metadata schema; only the Census data is allowed.")
    allowed = (config._get("SNOWFLAKE_DATABASE") or "").upper()
    for m in _THREE_PART.finditer(view):
        # Masked view has blanked quoted names; read them from the original.
        db = (sql[m.start(1):m.end(1)] if m.group(1) is not None else m.group(2)).upper()
        if allowed and db != allowed:
            raise UnsafeSQL("Query references database %s; only %s is allowed." % (db, allowed))


def words_in(view):
    # type: (str) -> set
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", view.upper()))


def _enforce_limit(sql, view, max_rows):
    # type: (str, str, int) -> str
    """Ensure the OUTERMOST query has a LIMIT no larger than max_rows.

    A LIMIT inside a subquery or CTE does not bound the result, so only a
    LIMIT at parenthesis depth zero counts. `view` is the masked analysis
    text, same length as `sql`, so match positions carry over.
    """
    top_level = [
        m for m in _LIMIT_CLAUSE.finditer(view)
        if view.count("(", 0, m.start()) == view.count(")", 0, m.start())
    ]
    if not top_level:
        return "%s\nLIMIT %d" % (sql.rstrip(), max_rows)
    last = top_level[-1]
    if int(last.group(1)) <= max_rows:
        return sql
    return sql[:last.start()] + ("LIMIT %d" % max_rows) + sql[last.end():]


# --- Topic gate -------------------------------------------------------------

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "on_topic": {
            "type": "boolean",
            "description": "True if answering requires US Census demographic data.",
        },
        "category": {
            "type": "string",
            "enum": ["census_question", "followup", "greeting", "off_topic", "unsafe"],
        },
        "reason": {
            "type": "string",
            "description": "One sentence, addressed to the user, explaining the call.",
        },
        "standalone_question": {
            "type": ["string", "null"],
            "description": "The message rewritten to stand alone; null if off-topic.",
        },
        "search_terms": {
            "type": ["string", "null"],
            "description": "Census-vocabulary keywords for schema search; null if off-topic.",
        },
    },
    "required": ["on_topic", "category", "reason", "standalone_question", "search_terms"],
    "additionalProperties": False,
}


class TopicVerdict(object):
    def __init__(self, on_topic, category, reason, standalone_question="", search_terms=""):
        # type: (bool, str, str, str, str) -> None
        self.on_topic = on_topic
        self.category = category
        self.reason = reason
        # Filled for on-topic messages: the question with follow-up references
        # resolved, and Census-vocabulary keywords for schema retrieval.
        self.standalone_question = standalone_question
        self.search_terms = search_terms

    @property
    def should_answer(self):
        # type: () -> bool
        """Greetings are on-topic enough to respond to without hitting SQL."""
        return self.on_topic and self.category in ("census_question", "followup")


def classify_question(llm, question, history=None, timeout=None):
    # type: (object, str, Optional[List[dict]], Optional[float]) -> TopicVerdict
    """Fast-fail gate. Runs before any schema retrieval or SQL generation.

    The same cheap call also resolves follow-ups into a standalone question
    and suggests Census-vocabulary search terms: it already has the history
    and the message in front of it, and a second round trip for that would
    cost another second on every turn.
    """
    from .prompts import TOPIC_GATE_SYSTEM, render_history

    context = render_history(history or [], limit=config.MAX_HISTORY_TURNS)
    user = "Recent conversation:\n%s\n\nClassify this message:\n%s" % (
        context or "(none)", question
    )
    data = llm.structured(
        system=TOPIC_GATE_SYSTEM,
        user=user,
        schema=TOPIC_SCHEMA,
        model=config.GUARDRAIL_MODEL,
        max_tokens=512,
        timeout=timeout,
    )
    return TopicVerdict(
        on_topic=bool(data.get("on_topic")),
        category=str(data.get("category", "off_topic")),
        reason=str(data.get("reason", "")),
        standalone_question=str(data.get("standalone_question") or ""),
        search_terms=str(data.get("search_terms") or ""),
    )
