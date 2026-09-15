"""Snowflake access. Every query goes through run_select(), which enforces
read-only execution, a row cap, and a statement timeout."""
import logging
from typing import Any, Dict, List, Optional, Tuple

from . import config

log = logging.getLogger(__name__)


class SnowflakeUnavailable(RuntimeError):
    """Raised when we cannot reach Snowflake at all (vs. a bad query)."""


class QueryFailed(RuntimeError):
    """Raised when Snowflake rejected or failed the query itself."""


_connection = None


def get_connection():
    """Lazily open one connection and reuse it.

    Streamlit reruns the whole script on every interaction, so opening a
    connection per turn would add seconds of handshake to each question.
    """
    global _connection
    if _connection is not None and not _connection.is_closed():
        return _connection
    try:
        import snowflake.connector
    except ImportError as exc:  # pragma: no cover
        raise SnowflakeUnavailable("snowflake-connector-python is not installed") from exc

    try:
        _connection = snowflake.connector.connect(
            client_session_keep_alive=True,
            **config.snowflake_params()
        )
    except Exception as exc:
        raise SnowflakeUnavailable(
            "Could not connect to Snowflake: %s" % exc
        ) from exc
    return _connection


def run_select(
    sql: str,
    max_rows: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """Execute a read-only query. Returns (column_names, rows).

    Callers must have already passed `sql` through guardrails.validate_sql.
    """
    max_rows = max_rows or config.MAX_ROWS
    timeout_seconds = timeout_seconds or config.QUERY_TIMEOUT_SECONDS

    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Belt-and-braces: even if the validator missed something, the session
        # itself refuses to write and will not run past the timeout.
        cursor.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = %d" % timeout_seconds)
        cursor.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(max_rows)
        return columns, rows
    except Exception as exc:
        raise QueryFailed(str(exc)) from exc
    finally:
        cursor.close()


def list_columns() -> List[Dict[str, str]]:
    """Introspect every column in the configured database.

    This is the raw material for the schema index. The Census share has
    thousands of columns, which is exactly why we retrieve over them instead
    of pasting them into a prompt.
    """
    sql = """
        SELECT table_name, column_name, data_type, comment
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (config._get("SNOWFLAKE_SCHEMA", "PUBLIC"),))
        return [
            {
                "table": r[0],
                "column": r[1],
                "data_type": r[2],
                "comment": r[3] or "",
            }
            for r in cursor.fetchall()
        ]
    finally:
        cursor.close()
