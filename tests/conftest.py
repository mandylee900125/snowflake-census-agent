import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from census_agent.schema_index import ColumnDoc, SchemaIndex  # noqa: E402


class FakeLLM(object):
    """Scripted stand-in for LLMClient.

    Tests assert on our orchestration, validation, and error handling -- the
    parts we wrote. Scripting the model keeps them deterministic and free,
    at the cost of not testing real model behaviour (see REFLECTION.md).
    """

    def __init__(self, structured_responses=None, stream_text="Here is the answer."):
        self.structured_responses = list(structured_responses or [])
        self._stream_text = stream_text
        self.calls = []

    def structured(self, system, user, schema, model=None, max_tokens=None, effort=None, timeout=None):
        self.calls.append({"system": system, "user": user, "model": model})
        if not self.structured_responses:
            raise AssertionError("FakeLLM ran out of scripted structured responses")
        response = self.structured_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def stream_text(self, system, messages, model=None, max_tokens=None, effort=None, timeout=None):
        self.calls.append({"system": system, "messages": messages, "model": model})
        for word in self._stream_text.split():
            yield word + " "


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def index():
    return SchemaIndex([
        ColumnDoc("B19013", "B19013e1", "NUMBER",
                  "Median household income in the past 12 months (2019 inflation-adjusted dollars)"),
        ColumnDoc("B25003", "B25003e3", "NUMBER",
                  "Total: Renter-occupied housing units"),
        ColumnDoc("B25003", "B25003e2", "NUMBER",
                  "Total: Owner-occupied housing units"),
        ColumnDoc("B08301", "B08301e18", "NUMBER",
                  "Means of transportation to work: Bicycle"),
        ColumnDoc("B01001", "B01001e26", "NUMBER",
                  "Sex By Age [universe: Total population]: Total > Female"),
        ColumnDoc("CBG_GEO", "CENSUS_BLOCK_GROUP", "VARCHAR",
                  "Census block group identifier"),
        ColumnDoc("CBG_GEO", "AMOUNT_LAND", "NUMBER", "Land area in square meters"),
    ])


def on_topic(category="census_question", standalone=None, search_terms=None):
    return {"on_topic": True, "category": category, "reason": "ok",
            "standalone_question": standalone, "search_terms": search_terms}


def sql_plan(sql, answerable=True, clarification=None, assumptions=None, explanation="ok",
             decline_reason=None):
    return {
        "answerable": answerable,
        "sql": sql,
        "assumptions": assumptions or [],
        "clarification": clarification,
        "decline_reason": decline_reason or (None if answerable else "not_in_dataset"),
        "explanation": explanation,
    }
