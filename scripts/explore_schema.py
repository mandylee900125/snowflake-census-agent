"""Dump what the Census share actually contains.

Run this FIRST, before writing any query logic. It prints the tables, the
column counts, and a sample of each metadata table so you can see how the
field descriptions are keyed.

    python scripts/explore_schema.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from census_agent import snowflake_client  # noqa: E402


def main():
    print("Connecting to Snowflake…")
    columns = snowflake_client.list_columns()
    counts = Counter(c["table"] for c in columns)

    print("\n%d columns across %d tables\n" % (len(columns), len(counts)))
    for table, n in counts.most_common():
        print("  %-55s %6d columns" % (table, n))

    metadata_tables = [t for t in counts if "METADATA" in t.upper() or "DESCRIPTION" in t.upper()]
    print("\nLikely metadata / description tables: %s" % (metadata_tables or "NONE FOUND"))

    for table in metadata_tables:
        print("\n--- %s ---" % table)
        cols = [c["column"] for c in columns if c["table"] == table]
        print("columns: %s" % ", ".join(cols))
        try:
            names, rows = snowflake_client.run_select(
                'SELECT * FROM "%s" LIMIT 5' % table, max_rows=5
            )
            for row in rows:
                print("  " + " | ".join(str(v)[:60] for v in row))
        except Exception as exc:
            print("  (could not sample: %s)" % exc)

    print("\nSample of non-metadata columns:")
    for c in columns[:15]:
        print("  %s.%s (%s) %s" % (c["table"], c["column"], c["data_type"], c["comment"][:40]))


if __name__ == "__main__":
    main()
