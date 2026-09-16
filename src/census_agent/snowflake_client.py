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
    A cursor is created per call: the connector documents connections as
    shareable between threads and cursors as not.
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
            # Session-wide ceiling. The per-statement timeout in run_select
            # is tighter; this catches anything that bypasses it.
            session_parameters={"STATEMENT_TIMEOUT_IN_SECONDS": config.QUERY_TIMEOUT_SECONDS},
            **config.snowflake_params()
        )
    except Exception as exc:
        raise SnowflakeUnavailable(
            "Could not connect to Snowflake: %s" % exc
        ) from exc
    _check_role(_connection)
    return _connection


ADMIN_ROLES = ("ACCOUNTADMIN", "SECURITYADMIN", "SYSADMIN", "USERADMIN")


def _check_role(conn):
    """Verify the session got the configured role, and warn loudly if it is
    an administrative one. Configuration says what we asked for; this says
    what we actually got."""
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT CURRENT_ROLE()")
            role = (cur.fetchone() or [""])[0] or ""
        finally:
            cur.close()
    except Exception:  # pragma: no cover - diagnostics must never break connect
        log.warning("Could not verify the Snowflake role")
        return
    wanted = (config._get("SNOWFLAKE_ROLE") or "").upper()
    if wanted and role.upper() != wanted:
        log.warning("Snowflake session role is %s, not the configured %s", role, wanted)
    if role.upper() in ADMIN_ROLES:
        log.warning("Snowflake session is running as %s. Use the read-only role "
                    "from scripts/create_readonly_role.sql.", role)


def _is_query_error(exc):
    # type: (Exception) -> bool
    """True if Snowflake rejected the SQL itself.

    Compile errors, bad identifiers and statement timeouts all arrive as
    ProgrammingError -- the model can be asked to fix those. Session and
    authentication failures use the same class but a 390xxx error code, and
    everything else (OperationalError, InterfaceError, socket errors) is the
    connection or the service. None of those must be fed back to the model
    as if it had written bad SQL.
    """
    try:
        from snowflake.connector import errors
    except ImportError:  # pragma: no cover
        return True
    if not isinstance(exc, errors.ProgrammingError):
        return False
    errno = getattr(exc, "errno", None) or 0
    return not (390000 <= errno < 391000)


def run_select(
    sql: str,
    max_rows: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """Execute a read-only query. Returns (column_names, rows).

    Callers must have already passed `sql` through guardrails.validate_sql.
    Raises QueryFailed when the SQL is at fault, SnowflakeUnavailable when
    the connection or service is; the pipeline treats those differently.
    """
    global _connection
    max_rows = max_rows or config.MAX_ROWS
    timeout_seconds = timeout_seconds or config.QUERY_TIMEOUT_SECONDS

    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Per-statement, client-enforced: the connector cancels the query in
        # Snowflake when it expires, so one slow query cannot eat the turn.
        cursor.execute(sql, timeout=timeout_seconds)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(max_rows)
        return columns, rows
    except Exception as exc:
        if _is_query_error(exc):
            raise QueryFailed(str(exc)) from exc
        # Drop the connection so the next call reconnects instead of reusing
        # a dead socket, and tell the caller this was the service, not the SQL.
        log.exception("Snowflake connection or service error")
        _connection = None
        raise SnowflakeUnavailable("Lost the connection to Snowflake: %s" % exc) from exc
    finally:
        try:
            cursor.close()
        except Exception:  # pragma: no cover - already failing
            pass


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
