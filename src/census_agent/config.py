"""Configuration. Reads from Streamlit secrets first, then env vars, then .env."""
import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Streamlit secrets take priority so the deployed app needs no .env file."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def require(key: str) -> str:
    val = _get(key)
    if not val:
        raise RuntimeError(
            "Missing required setting '%s'. Set it in .env locally, or in "
            "Streamlit Cloud under App settings -> Secrets." % key
        )
    return val


# --- Models -----------------------------------------------------------------
# SQL generation and answer synthesis: quality matters most, this is the core
# of the product. Opus 5 has thinking on by default.
SQL_MODEL = _get("SQL_MODEL", "claude-opus-5")
# Guardrail classification: a fast-fail path the assignment explicitly asks for.
# Haiku keeps the reject path ~1s instead of ~8s. One-line change if you
# decide the accuracy tradeoff isn't worth it.
GUARDRAIL_MODEL = _get("GUARDRAIL_MODEL", "claude-haiku-4-5")

# Effort controls thinking depth / token spend on Opus 5. "medium" keeps us
# well inside the 60s budget; raise to "high" if SQL quality needs it.
SQL_EFFORT = _get("SQL_EFFORT", "medium")

# --- Safety limits ----------------------------------------------------------
MAX_ROWS = int(_get("MAX_ROWS", "500"))
QUERY_TIMEOUT_SECONDS = int(_get("QUERY_TIMEOUT_SECONDS", "30"))
MAX_SCHEMA_CANDIDATES = int(_get("MAX_SCHEMA_CANDIDATES", "20"))
# Diversity cap: no single ACS table may fill more than this many slots.
MAX_CANDIDATES_PER_GROUP = int(_get("MAX_CANDIDATES_PER_GROUP", "5"))
MAX_HISTORY_TURNS = int(_get("MAX_HISTORY_TURNS", "8"))

SCHEMA_INDEX_PATH = _get("SCHEMA_INDEX_PATH", "schema_index.json")


def anthropic_api_key() -> str:
    return require("ANTHROPIC_API_KEY")


def snowflake_params() -> dict:
    params = {
        "account": require("SNOWFLAKE_ACCOUNT"),
        "user": require("SNOWFLAKE_USER"),
        "password": require("SNOWFLAKE_PASSWORD"),
        "warehouse": require("SNOWFLAKE_WAREHOUSE"),
        "database": require("SNOWFLAKE_DATABASE"),
        "schema": _get("SNOWFLAKE_SCHEMA", "PUBLIC"),
    }
    role = _get("SNOWFLAKE_ROLE")
    if role:
        params["role"] = role
    return params
