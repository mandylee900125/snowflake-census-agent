"""A Snowflake error is one of two different things: the model's SQL was bad
(feed the error back for one repair) or the connection/service failed (do not
ask the model to fix a network error; drop the connection so the next call
reconnects). Classifying them the same way -- the first version did -- sends
a socket error to the model as if it had written bad SQL.

Error classes were checked against the real account: compile errors, bad
identifiers and client-side statement timeouts all raise ProgrammingError.
"""
import pytest

from census_agent import snowflake_client as sc

errors = pytest.importorskip("snowflake.connector.errors")


class FakeCursor(object):
    def __init__(self, exc):
        self.exc = exc
        self.timeout = None
        self.description = [("N",)]

    def execute(self, sql, timeout=None):
        self.timeout = timeout
        if self.exc is not None:
            raise self.exc

    def fetchmany(self, n):
        return [(1,)]

    def close(self):
        pass


class FakeConn(object):
    def __init__(self, exc):
        self.cursor_obj = FakeCursor(exc)

    def cursor(self):
        return self.cursor_obj

    def is_closed(self):
        return False


def _install(monkeypatch, exc):
    conn = FakeConn(exc)
    monkeypatch.setattr(sc, "_connection", conn)
    monkeypatch.setattr(sc, "get_connection", lambda: conn)
    return conn


def _err(cls, msg, errno=None):
    return cls(msg=msg, errno=errno, send_telemetry=False)


def test_bad_sql_is_a_query_failure_the_model_can_repair(monkeypatch):
    _install(monkeypatch, _err(errors.ProgrammingError, "SQL compilation error: invalid identifier", 904))
    with pytest.raises(sc.QueryFailed, match="compilation"):
        sc.run_select("SELECT nope")
    assert sc._connection is not None  # the connection is fine; keep it


def test_statement_timeout_is_also_the_querys_fault(monkeypatch):
    _install(monkeypatch, _err(errors.ProgrammingError,
                               "SQL execution was cancelled by the client due to a timeout", 604))
    with pytest.raises(sc.QueryFailed, match="timeout"):
        sc.run_select("SELECT slow")


def test_connection_errors_are_not_fed_back_to_the_model(monkeypatch):
    _install(monkeypatch, _err(errors.OperationalError, "Could not connect to Snowflake backend"))
    with pytest.raises(sc.SnowflakeUnavailable, match="Lost the connection"):
        sc.run_select("SELECT 1")
    assert sc._connection is None  # dropped, so the next call reconnects


def test_expired_session_is_a_service_problem_despite_its_error_class(monkeypatch):
    _install(monkeypatch, _err(errors.ProgrammingError, "Authentication token has expired", 390114))
    with pytest.raises(sc.SnowflakeUnavailable):
        sc.run_select("SELECT 1")
    assert sc._connection is None


def test_per_statement_timeout_is_passed_to_the_connector(monkeypatch):
    conn = _install(monkeypatch, None)
    columns, rows = sc.run_select("SELECT 1", timeout_seconds=7)
    assert conn.cursor_obj.timeout == 7
    assert (columns, rows) == (["N"], [(1,)])
