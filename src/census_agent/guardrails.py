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


class UnsafeSQL(ValueError):
    """The generated SQL is not a safe read-only query."""


def _strip_comments(sql):
    # type: (str) -> str
    return _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))


def _mask_literals(sql):
    # type: (str) -> str
    """Blank out string literals and quoted identifiers before keyword scanning.

    Census column names are quoted free text -- e.g.
    "Total: Renter-occupied housing units" -- and a naive keyword scan over
    raw SQL would flag legitimate identifiers containing words like GET or SET.
    """
    sql = _SINGLE_QUOTED.sub("''", sql)
    sql = _DOUBLE_QUOTED.sub('""', sql)
    return sql


def validate_sql(sql, max_rows=None):
    # type: (str, Optional[int]) -> str
    """Return safe-to-execute SQL, or raise UnsafeSQL.

    Guarantees on the returned string:
      * exactly one statement
      * it is a SELECT (or a WITH ... SELECT)
      * it contains no data-modifying or session-modifying keyword
      * it has a LIMIT no larger than max_rows
    """
    max_rows = max_rows or config.MAX_ROWS

    if not sql or not sql.strip():
        raise UnsafeSQL("The model returned an empty query.")

    cleaned = _strip_comments(sql).strip()
    masked = _mask_literals(cleaned)

    # One statement only. A trailing semicolon is fine; a second statement
    # is a classic injection shape (SELECT 1; DROP TABLE x).
    statements = [s for s in masked.split(";") if s.strip()]
    if len(statements) > 1:
        raise UnsafeSQL("Only one SQL statement may be executed at a time.")

    body = cleaned.rstrip().rstrip(";").rstrip()
    masked_body = _mask_literals(_strip_comments(body))

    first = masked_body.lstrip().split(None, 1)
    first_word = first[0].upper() if first else ""
    if first_word not in ("SELECT", "WITH"):
        raise UnsafeSQL(
            "Only SELECT queries are allowed; this one started with '%s'." % (first_word or "nothing")
        )

    words = set(re.findall(r"[A-Za-z_]+", masked_body.upper()))
    banned = sorted(words & FORBIDDEN_KEYWORDS)
    if banned:
        raise UnsafeSQL("Query contains disallowed keyword(s): %s." % ", ".join(banned))

    return _enforce_limit(body, max_rows)


def _enforce_limit(sql, max_rows):
    # type: (str, int) -> str
    """Ensure a LIMIT exists and is no larger than max_rows."""
    matches = list(_LIMIT_CLAUSE.finditer(_mask_literals(sql)))
    if not matches:
        return "%s\nLIMIT %d" % (sql.rstrip(), max_rows)

    last = matches[-1]
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
    },
    "required": ["on_topic", "category", "reason"],
    "additionalProperties": False,
}


class TopicVerdict(object):
    def __init__(self, on_topic, category, reason):
        # type: (bool, str, str) -> None
        self.on_topic = on_topic
        self.category = category
        self.reason = reason

    @property
    def should_answer(self):
        # type: () -> bool
        """Greetings are on-topic enough to respond to without hitting SQL."""
        return self.on_topic and self.category in ("census_question", "followup")


def classify_question(llm, question, history=None):
    # type: (object, str, Optional[List[dict]]) -> TopicVerdict
    """Fast-fail gate. Runs before any schema retrieval or SQL generation."""
    from .prompts import TOPIC_GATE_SYSTEM, render_history

    context = render_history(history or [], limit=4)
    user = "Recent conversation:\n%s\n\nClassify this message:\n%s" % (
        context or "(none)", question
    )
    data = llm.structured(
        system=TOPIC_GATE_SYSTEM,
        user=user,
        schema=TOPIC_SCHEMA,
        model=config.GUARDRAIL_MODEL,
        max_tokens=512,
    )
    return TopicVerdict(
        on_topic=bool(data.get("on_topic")),
        category=str(data.get("category", "off_topic")),
        reason=str(data.get("reason", "")),
    )
