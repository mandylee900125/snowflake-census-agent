"""Build the schema index and write it to disk.

Run once after connecting to Snowflake, then commit the JSON so the deployed
app never has to introspect thousands of columns on boot.

    python scripts/build_index.py

If field descriptions aren't picked up automatically, run
scripts/explore_schema.py to see how the metadata table is keyed and set
--desc-table / --code-col / --desc-col accordingly.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from census_agent import config, snowflake_client  # noqa: E402
from census_agent.schema_index import SchemaIndex, build_docs  # noqa: E402


def load_descriptions(table, code_col, desc_cols):
    """Read {column_code: human description} from the dataset's metadata table.

    This is what makes coded ACS columns (B19013e1) findable by lexical search.
    """
    select = ", ".join('"%s"' % c for c in [code_col] + desc_cols)
    sql = 'SELECT %s FROM "%s"' % (select, table)
    names, rows = snowflake_client.run_select(sql, max_rows=100000)
    out = {}
    for row in rows:
        code = row[0]
        if not code:
            continue
        parts = [str(v) for v in row[1:] if v not in (None, "", "null")]
        out[str(code)] = " ".join(parts)
    return out


def autodetect(columns):
    """Guess the metadata table and its code/description columns."""
    tables = {}
    for c in columns:
        tables.setdefault(c["table"], []).append(c["column"])
    for table, cols in tables.items():
        upper = table.upper()
        if "FIELD" in upper and ("DESCRIPTION" in upper or "METADATA" in upper):
            code = next((c for c in cols if "FIELD" in c.upper() or "ID" in c.upper()), cols[0])
            descs = [c for c in cols if c != code]
            return table, code, descs
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc-table")
    parser.add_argument("--code-col")
    parser.add_argument("--desc-col", action="append", default=[])
    parser.add_argument("--out", default=config.SCHEMA_INDEX_PATH)
    args = parser.parse_args()

    print("Introspecting columns…")
    columns = snowflake_client.list_columns()
    print("  found %d columns" % len(columns))

    table, code_col, desc_cols = args.desc_table, args.code_col, args.desc_col
    if not table:
        table, code_col, desc_cols = autodetect(columns)

    descriptions = {}
    if table:
        print("Loading field descriptions from %s (%s -> %s)…" % (table, code_col, desc_cols))
        try:
            descriptions = load_descriptions(table, code_col, desc_cols)
            print("  loaded %d descriptions" % len(descriptions))
        except Exception as exc:
            print("  WARNING: could not load descriptions: %s" % exc)
            print("  Coded columns will be hard to retrieve. Run explore_schema.py.")
    else:
        print("WARNING: no field-description table detected. Run explore_schema.py.")

    docs = build_docs(columns, descriptions)
    index = SchemaIndex(docs)
    path = index.save(args.out)
    described = sum(1 for d in docs if d.description)
    print("\nWrote %s: %d columns, %d with descriptions (%.0f%%)"
          % (path, len(docs), described, 100.0 * described / max(len(docs), 1)))

    for probe in ["median household income", "renter occupied housing", "commute by bicycle"]:
        hits = index.search(probe, limit=3)
        print("\n  %r ->" % probe)
        for h in hits:
            print("     %s -- %s" % (h.qualified_name, h.description[:60]))


if __name__ == "__main__":
    main()
