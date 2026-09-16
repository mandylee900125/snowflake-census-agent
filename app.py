"""Streamlit chat UI for the US Census agent."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st  # noqa: E402

from census_agent import config  # noqa: E402
from census_agent.llm import LLMClient  # noqa: E402
from census_agent.schema_index import SchemaIndex  # noqa: E402
from census_agent.pipeline import answer_question  # noqa: E402

st.set_page_config(page_title="US Census Agent", page_icon="📊", layout="centered")

EXAMPLES = [
    "What's the median household income in Brooklyn?",
    "Which counties in Texas have the highest share of renters?",
    "How does educational attainment differ between Seattle and Portland?",
]


@st.cache_resource(show_spinner=False)
def get_llm():
    return LLMClient()


@st.cache_resource(show_spinner=False)
def get_index():
    return SchemaIndex.load(config.SCHEMA_INDEX_PATH)


def boot():
    """Load dependencies, or explain precisely what is missing and stop.

    A blank page is the worst possible failure for a reviewer, so every
    startup problem gets a named cause and a fix.
    """
    try:
        llm = get_llm()
    except Exception as exc:
        st.error("**The assistant is not configured.**\n\n%s" % exc)
        st.stop()
    try:
        index = get_index()
    except FileNotFoundError as exc:
        st.error(
            "**The schema index is missing.**\n\n%s\n\nIt is built once from "
            "Snowflake and committed to the repo." % exc
        )
        st.stop()
    except Exception as exc:
        st.error("**Could not load the schema index.**\n\n%s" % exc)
        st.stop()
    return llm, index


def md(text):
    """Render model text as markdown without Streamlit's LaTeX surprise.

    st.markdown treats "$74,600 ... $74,621" as an inline formula and typesets
    everything between the two dollar signs as math. Census answers are full
    of dollar amounts, so escape every $ before rendering.
    """
    return text.replace("$", "\\$")


def render_turn(turn):
    """Re-render a stored turn on Streamlit's rerun."""
    with st.chat_message(turn["role"]):
        st.markdown(md(turn["content"]))
        if turn.get("sql"):
            with st.expander("SQL"):
                st.code(turn["sql"], language="sql")
        if turn.get("rows"):
            with st.expander("Results (%d rows)" % len(turn["rows"])):
                st.dataframe(turn["rows"], use_container_width=True)


def run_turn(llm, index, question, history):
    """Consume pipeline events and render them as they arrive."""
    answer = []
    sql = None
    columns, rows = None, None

    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=False)
        sql_slot = st.empty()
        data_slot = st.empty()
        answer_slot = st.empty()

        for event in answer_question(llm, index, question, history):
            kind = event["type"]
            if kind == "status":
                status.update(label=event["text"])
            elif kind == "sql":
                sql = event["sql"]
                with sql_slot.expander("SQL"):
                    st.code(sql, language="sql")
            elif kind == "data":
                columns, rows = event["columns"], event["rows"]
                with data_slot.expander("Results (%d rows)" % len(rows)):
                    st.dataframe(
                        [dict(zip(columns, r)) for r in rows],
                        use_container_width=True,
                    )
            elif kind == "token":
                answer.append(event["text"])
                answer_slot.markdown(md("".join(answer)))
            elif kind == "message":
                answer.append(event["text"])
                answer_slot.markdown(md(event["text"]))
            elif kind == "error":
                answer.append(event["text"])
                answer_slot.warning(event["text"])
            elif kind == "done":
                pass

        status.update(label="Done", state="complete")

    return {
        "role": "assistant",
        "content": "".join(answer) or "(no response)",
        "sql": sql,
        "rows": [dict(zip(columns, r)) for r in rows] if rows else None,
    }


def render_sidebar(index):
    """Scope and limits, so a reviewer knows what to expect before typing."""
    with st.sidebar:
        st.markdown("### About this agent")
        st.markdown(
            "Each question is turned into one read-only SQL query against the "
            "Census data in Snowflake, and the answer is written from the rows "
            "that came back -- never from memory. The SQL and the rows are "
            "shown under every answer."
        )
        st.markdown("**Covers**")
        st.markdown(
            "- ACS 5-year estimates, **2020** (default) and 2019\n"
            "- 2020 decennial census counts\n"
            "- %d variables: age, sex, race, income, poverty, employment, "
            "education, housing, rent, commuting, language, households, "
            "veterans, insurance, internet\n"
            "- Any US state, county, or census block group" % len(index)
        )
        st.markdown("**Good to know**")
        st.markdown(
            "- Cities are approximated by their county (Brooklyn = Kings County) "
            "and the answer says so\n"
            "- County medians are weighted averages of block-group medians\n"
            "- No ZIP codes or neighbourhood boundaries; no years other than 2019/2020\n"
            "- Follow-ups work (\"what about Queens?\"); history lasts for this tab"
        )
        if st.button("New conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def main():
    st.title("📊 US Census Agent")
    st.caption(
        "Ask about US demographics. Grounded in SafeGraph's US Open Census Data "
        "(ACS 2020 and 2019 5-year estimates plus the 2020 decennial count, "
        "census block group level) via Snowflake."
    )

    llm, index = boot()
    render_sidebar(index)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        cols = st.columns(len(EXAMPLES))
        for col, example in zip(cols, EXAMPLES):
            if col.button(example, use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    for turn in st.session_state.messages:
        render_turn(turn)

    question = st.chat_input("Ask about US demographics…")
    if not question:
        question = st.session_state.pop("pending", None)
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(md(question))

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    try:
        turn = run_turn(llm, index, question, history)
    except Exception as exc:  # last-resort net: never show a traceback
        st.error(
            "Something went wrong handling that question. Please try "
            "rephrasing it.\n\n`%s`" % exc
        )
        turn = {"role": "assistant", "content": "Sorry — that question failed."}

    st.session_state.messages.append(turn)


if __name__ == "__main__":
    main()
