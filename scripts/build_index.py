"""Build the schema index and write it to disk.

Run once after connecting to Snowflake, then commit the JSON so the deployed
app never has to introspect thousands of columns on boot.

    python scripts/build_index.py

The share's layout (two ACS vintages, estimate/margin-of-error column pairs,
a 10-level description hierarchy, a separate decennial table) is handled in
census_layout.py. This script only fetches the raw material and reports on
what the index can find.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from census_agent import census_layout, config, snowflake_client  # noqa: E402
from census_agent.schema_index import SchemaIndex  # noqa: E402

# Newest vintage last so its description wins when both define a code (the
# 2020 text says "2020 inflation-adjusted dollars", which is what the default
# table actually contains).
ACS_META_TABLES = ["2019_METADATA_CBG_FIELD_DESCRIPTIONS", "2020_METADATA_CBG_FIELD_DESCRIPTIONS"]
REDISTRICTING_META_TABLE = "2020_REDISTRICTING_METADATA_CBG_FIELD_DESCRIPTIONS"

# Questions a reviewer would plausibly ask, with the column each must surface.
PROBES = [
    # (question as the retrieval layer sees it, column that must be in the top 20)
    # Raw phrasings, and phrasings with the gate's Census-vocabulary terms
    # appended -- that is what the pipeline actually searches with.
    ("median household income", "B19013e1"),
    ("renter occupied housing units", "B25003e3"),
    ("commute to work by bicycle", "B08301e18"),
    ("median gross rent", "B25064e1"),
    ("hispanic or latino population", "B03002e12"),
    ("median age", "B01002e1"),
    ("total population", "B01003e1"),
    ("total population 2020 census", "P0010001"),
    ("asian population", "B02001e5"),
    ("households with no vehicle", "B25044e3"),
    ("people who speak spanish at home", "C16002e3"),
    ("median home value", "B25077e1"),
    ("veterans", "B21001e2"),
    ("unemployment rate unemployed civilian labor force", "B23025e5"),
    ("how many women are there female sex by age", "B01001e26"),
    ("seniors 65 years and over sex by age", "B01001e20"),
    ("college degree educational attainment bachelor's degree population 25 years and over", "B15003e22"),
    ("married couple families household type family households", "B11001e3"),
]


def fetch_rows_as_dicts(table):
    """SELECT * from a metadata table, as a list of {column: value}."""
    names, rows = snowflake_client.run_select('SELECT * FROM "%s"' % table, max_rows=100000)
    return [dict(zip(names, row)) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=config.SCHEMA_INDEX_PATH)
    args = parser.parse_args()

    print("Introspecting columns…")
    columns = snowflake_client.list_columns()
    print("  found %d columns across %d tables"
          % (len(columns), len(set(c["table"] for c in columns))))

    acs_meta = {}
    for table in ACS_META_TABLES:
        print("Loading %s…" % table)
        try:
            rows = fetch_rows_as_dicts(table)
        except snowflake_client.QueryFailed as exc:
            print("  WARNING: skipped (%s)" % exc)
            continue
        for row in rows:
            code = row.get("TABLE_ID")
            if code:
                acs_meta[str(code)] = row
        print("  %d descriptions" % len(rows))

    redistricting_meta = {}
    print("Loading %s…" % REDISTRICTING_META_TABLE)
    try:
        for row in fetch_rows_as_dicts(REDISTRICTING_META_TABLE):
            code = row.get("COLUMN_ID")
            if code:
                redistricting_meta[str(code)] = row
        print("  %d descriptions" % len(redistricting_meta))
    except snowflake_client.QueryFailed as exc:
        print("  WARNING: skipped (%s)" % exc)

    docs = census_layout.build_census_docs(columns, acs_meta, redistricting_meta)
    index = SchemaIndex(docs)
    path = index.save(args.out)

    described = sum(1 for d in docs if d.description)
    both = sum(1 for d in docs if len(d.vintages) > 1)
    print("\nWrote %s: %d variables (%d with descriptions, %d present in both ACS vintages)"
          % (path, len(docs), described, both))
    undescribed = [d for d in docs if not d.description]
    if undescribed:
        print("  Undescribed: %s" % ", ".join(d.qualified_name for d in undescribed[:10]))

    print("\nRetrieval probes (top %d):" % config.MAX_SCHEMA_CANDIDATES)
    failures = 0
    for question, expected in PROBES:
        hits = index.search(question)
        names = [h.column for h in hits]
        rank = names.index(expected) + 1 if expected in names else None
        status = "ok  rank %2d" % rank if rank else "MISS      "
        failures += rank is None
        print("  %s  %-40r -> %s" % (status, question, expected))
        if rank is None:
            for h in hits[:3]:
                print("             got %s -- %s" % (h.column, h.description[:70]))
    if failures:
        print("\n%d probe(s) missed. Fix retrieval before deploying." % failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
