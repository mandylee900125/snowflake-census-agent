"""Ask the agent from the terminal, without Streamlit.

Useful for debugging a single question end to end against the real model and
database, and for timing each stage. Pass several questions to simulate a
conversation with follow-ups.

    python scripts/ask.py "median household income in Brooklyn?" "what about Queens?"
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from census_agent import config  # noqa: E402
from census_agent.llm import LLMClient  # noqa: E402
from census_agent.pipeline import answer_question  # noqa: E402
from census_agent.schema_index import SchemaIndex  # noqa: E402

# USD per million tokens: (input, output, cache read, cache write).
# Source: Anthropic pricing page, checked 2026-09-15.
PRICES = {
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
}


def cost_usd(usage):
    total = 0.0
    for model, u in usage.items():
        p = next((v for k, v in PRICES.items() if model.startswith(k)), None)
        if not p:
            continue
        total += (u["input"] * p[0] + u["output"] * p[1]
                  + u["cache_read"] * p[2] + u["cache_write"] * p[3]) / 1e6
    return total


def main(questions):
    llm = LLMClient()
    index = SchemaIndex.load(config.SCHEMA_INDEX_PATH)
    history = []
    for question in questions:
        print("\n" + "=" * 78 + "\nQ: %s" % question)
        started = time.time()
        spent_before = cost_usd(llm.usage)
        answer = []
        for event in answer_question(llm, index, question, history):
            elapsed = time.time() - started
            kind = event["type"]
            if kind == "status":
                print("  [%5.1fs] %s" % (elapsed, event["text"]))
            elif kind == "sql":
                print("  [%5.1fs] SQL:\n%s" % (elapsed, _indent(event["sql"])))
            elif kind == "data":
                print("  [%5.1fs] %d row(s); columns %s" % (elapsed, len(event["rows"]), event["columns"]))
                for row in event["rows"][:5]:
                    print("      %s" % (row,))
            elif kind == "token":
                answer.append(event["text"])
            elif kind in ("message", "error"):
                answer.append(event["text"])
                print("  [%5.1fs] %s: %s" % (elapsed, kind.upper(), event["text"]))
        text = "".join(answer).strip()
        print("  [%5.1fs] A: %s" % (time.time() - started, text))
        spent = cost_usd(llm.usage) - spent_before
        print("  cost: $%.4f  (%s)" % (spent, "; ".join(
            "%s in=%d out=%d cached=%d" % (m.split("-")[1], u["input"], u["output"], u["cache_read"])
            for m, u in llm.usage.items())))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": text})


def _indent(text):
    return "\n".join("      " + line for line in text.splitlines())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
