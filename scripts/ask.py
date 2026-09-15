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


def main(questions):
    llm = LLMClient()
    index = SchemaIndex.load(config.SCHEMA_INDEX_PATH)
    history = []
    for question in questions:
        print("\n" + "=" * 78 + "\nQ: %s" % question)
        started = time.time()
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
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": text})


def _indent(text):
    return "\n".join("      " + line for line in text.splitlines())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
