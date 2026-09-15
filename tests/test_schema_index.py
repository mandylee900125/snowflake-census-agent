"""Retrieval tests: the golden questions each name a column that MUST surface.

If retrieval misses, nothing downstream can recover -- the model simply never
sees the column it needed. These are the highest-value tests in the suite.
"""
import pytest

from census_agent.schema_index import ColumnDoc, SchemaIndex, build_docs, tokenize


class TestTokenize:
    def test_splits_snake_case(self):
        assert tokenize("MEDIAN_HOUSEHOLD_INCOME") == ["median", "household", "income"]

    def test_splits_camel_case(self):
        assert tokenize("medianHouseholdIncome") == ["median", "household", "income"]

    def test_keeps_digits_for_acs_codes(self):
        assert "19013" in tokenize("B19013e1")

    def test_handles_punctuation_in_census_column_names(self):
        assert tokenize("Total: Renter-occupied housing units") == [
            "total", "renter", "occupied", "housing", "units"
        ]

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize(None) == []


GOLDEN = [
    ("What is the median household income?", "B19013e1"),
    ("How many renter occupied housing units are there?", "B25003e3"),
    ("How many people bike to work?", "B08301e18"),
    ("How much land area does the block group cover?", "AMOUNT_LAND"),
]


class TestSearch:
    @pytest.mark.parametrize("question,expected_column", GOLDEN)
    def test_golden_question_retrieves_its_column(self, index, question, expected_column):
        hits = index.search(question, limit=5)
        assert expected_column in [h.column for h in hits], (
            "%r did not retrieve %s; got %s"
            % (question, expected_column, [h.column for h in hits])
        )

    def test_results_are_capped(self, index):
        assert len(index.search("housing income work", limit=2)) <= 2

    def test_irrelevant_question_returns_nothing(self, index):
        # Zero-score hits are dropped so the model is never handed unrelated
        # columns and tempted to use them anyway.
        assert index.search("chocolate chip cookie recipe") == []

    def test_empty_query_returns_nothing(self, index):
        assert index.search("") == []

    def test_render_context_names_the_table(self, index):
        ctx = index.render_context("median household income")
        assert "B19013" in ctx and "B19013e1" in ctx

    def test_render_context_is_honest_when_nothing_matches(self, index):
        assert "no columns" in index.render_context("weather forecast tomorrow").lower()


class TestBuildAndPersist:
    def test_description_enrichment_makes_coded_columns_findable(self):
        columns = [{"table": "B19013", "column": "B19013e1", "data_type": "NUMBER", "comment": ""}]
        # Without a description, the code alone is unsearchable.
        bare = SchemaIndex(build_docs(columns))
        assert bare.search("median household income") == []
        # With the metadata table joined in, it is retrievable.
        enriched = SchemaIndex(build_docs(columns, {"B19013e1": "Median household income"}))
        assert enriched.search("median household income")

    def test_comment_wins_over_metadata_lookup(self):
        docs = build_docs(
            [{"table": "T", "column": "C", "data_type": "NUMBER", "comment": "real comment"}],
            {"C": "fallback"},
        )
        assert docs[0].description == "real comment"

    def test_roundtrip_through_disk(self, index, tmp_path):
        path = str(tmp_path / "index.json")
        index.save(path)
        assert len(SchemaIndex.load(path)) == len(index)

    def test_missing_index_file_explains_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_index"):
            SchemaIndex.load(str(tmp_path / "nope.json"))

    def test_empty_index_is_rejected_loudly(self):
        with pytest.raises(ValueError):
            SchemaIndex([])
