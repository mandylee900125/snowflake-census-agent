"""The validator is the security boundary, so it gets the densest tests.

A prompt-injected model must not be able to reach a write through it.
"""
import pytest

from census_agent.guardrails import UnsafeSQL, validate_sql


class TestAcceptsReadOnly:
    def test_plain_select_gets_a_limit(self):
        out = validate_sql("SELECT 1 FROM t")
        assert out.upper().startswith("SELECT")
        assert "LIMIT" in out.upper()

    def test_cte_is_allowed(self):
        out = validate_sql("WITH x AS (SELECT 1 AS a FROM t) SELECT a FROM x")
        assert "LIMIT" in out.upper()

    def test_existing_small_limit_is_preserved(self):
        assert validate_sql("SELECT 1 FROM t LIMIT 5").endswith("LIMIT 5")

    def test_oversized_limit_is_clamped(self):
        out = validate_sql("SELECT 1 FROM t LIMIT 999999", max_rows=100)
        assert "LIMIT 100" in out
        assert "999999" not in out

    def test_trailing_semicolon_is_fine(self):
        assert "LIMIT" in validate_sql("SELECT 1 FROM t;").upper()

    def test_leading_whitespace_and_newlines(self):
        assert validate_sql("\n\n   SELECT 1 FROM t\n").upper().startswith("SELECT")


class TestRejectsWrites:
    @pytest.mark.parametrize("sql", [
        "DELETE FROM users",
        "DROP TABLE census",
        "UPDATE t SET x = 1",
        "INSERT INTO t VALUES (1)",
        "CREATE TABLE evil (id INT)",
        "ALTER TABLE t ADD COLUMN x INT",
        "TRUNCATE TABLE t",
        "GRANT ALL ON t TO PUBLIC",
        "MERGE INTO t USING s ON t.id = s.id",
        "CALL some_proc()",
        "USE DATABASE other",
    ])
    def test_non_select_statements_are_rejected(self, sql):
        with pytest.raises(UnsafeSQL):
            validate_sql(sql)

    def test_stacked_statement_is_rejected(self):
        with pytest.raises(UnsafeSQL, match="one SQL statement"):
            validate_sql("SELECT 1 FROM t; DROP TABLE t")

    def test_write_hidden_after_a_line_comment_is_rejected(self):
        # Comments are stripped before scanning, so this must not slip past
        # as "just a comment".
        with pytest.raises(UnsafeSQL):
            validate_sql("SELECT 1 FROM t -- ok\n; DELETE FROM t")

    def test_write_inside_a_cte_is_rejected(self):
        with pytest.raises(UnsafeSQL):
            validate_sql("WITH x AS (SELECT 1) SELECT 1 FROM x; DROP TABLE x")

    def test_block_comment_cannot_hide_a_second_statement(self):
        with pytest.raises(UnsafeSQL):
            validate_sql("SELECT 1 /* nice */ ; TRUNCATE TABLE t")


class TestLiteralsAreNotKeywords:
    """Census column names are quoted free text and routinely contain words
    that look like SQL keywords. Flagging those would break real questions."""

    def test_quoted_identifier_containing_a_keyword_is_allowed(self):
        out = validate_sql('SELECT "Total: Renter-occupied housing units" FROM B25003')
        assert "Renter-occupied" in out

    def test_string_literal_containing_a_keyword_is_allowed(self):
        out = validate_sql("SELECT 1 FROM t WHERE name = 'DROP CITY'")
        assert "DROP CITY" in out

    def test_column_named_like_a_keyword_is_allowed(self):
        out = validate_sql('SELECT "GET workers" FROM t')
        assert "LIMIT" in out.upper()


class TestDegenerateInput:
    @pytest.mark.parametrize("sql", ["", "   ", "\n", None])
    def test_empty_is_rejected(self, sql):
        with pytest.raises(UnsafeSQL):
            validate_sql(sql)

    def test_prose_is_rejected(self):
        with pytest.raises(UnsafeSQL):
            validate_sql("I cannot answer that question.")

    def test_comment_only_is_rejected(self):
        with pytest.raises(UnsafeSQL):
            validate_sql("-- just a comment")


class TestAdministrativeFunctions:
    """A SELECT can still be dangerous: SYSTEM$ functions are administrative.

    Found by probing the real account: SELECT SYSTEM$CANCEL_ALL_QUERIES(...)
    passed the keyword denylist and executed.
    """

    def test_system_functions_are_rejected(self):
        from census_agent.guardrails import UnsafeSQL, validate_sql
        for sql in ("SELECT SYSTEM$CANCEL_ALL_QUERIES(123)",
                    "select system$abort_session(1) from t",
                    "WITH x AS (SELECT SYSTEM$ABORT_SESSION(1) AS y) SELECT * FROM x"):
            with pytest.raises(UnsafeSQL, match="SYSTEM"):
                validate_sql(sql)

    def test_system_inside_a_quoted_identifier_is_fine(self):
        from census_agent.guardrails import validate_sql
        assert validate_sql('SELECT "SYSTEM$-like label" FROM t LIMIT 5')
