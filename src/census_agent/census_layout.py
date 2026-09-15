"""Knowledge of how the SafeGraph Census share is laid out.

Everything that is specific to *this* dataset's shape lives here, so the
retrieval and prompt code stay generic. What scripts/explore_schema.py found:

  * Two ACS 5-year vintages, 2019 and 2020, with identical structure. Each is
    ~30 tables named {vintage}_CBG_{family} (2020_CBG_B19 holds the income
    tables), ~8,100 columns per vintage.
  * Columns are ACS variable codes. B19013e1 is an estimate; B19013m1 is its
    margin of error. Every table has CENSUS_BLOCK_GROUP as its key.
  * {vintage}_METADATA_CBG_FIELD_DESCRIPTIONS describes each code with a
    title, topic list, universe, and a 10-level label hierarchy.
  * 2020_REDISTRICTING_CBG_DATA is the 2020 decennial count (P-codes), with
    its own flat description table.
  * {vintage}_METADATA_CBG_FIPS_CODES maps state/county FIPS to names; there
    is no city, ZIP, or neighbourhood crosswalk.

Index design decisions (see DECISIONS.md):
  * One document per variable, not per table copy. The 2019 and 2020 tables
    share codes, so indexing both would hand the model duplicate candidates
    and halve the useful context. The doc records which vintages carry it.
  * Estimates only. Margin-of-error columns share the estimate's description
    and would crowd out other candidates; the SQL prompt explains the e -> m
    naming rule so they remain reachable.
"""
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .schema_index import ColumnDoc

DEFAULT_VINTAGE = "2020"
ACS_VINTAGES = ("2020", "2019")

# Tables that are indexed but whose columns have no metadata row.
PATTERNS_TABLE = "2019_CBG_PATTERNS"
PATTERNS_DESCRIPTIONS = {
    "DATE_RANGE_START": "SafeGraph foot-traffic patterns 2019: start of the observation window",
    "DATE_RANGE_END": "SafeGraph foot-traffic patterns 2019: end of the observation window",
    "RAW_VISIT_COUNT": "SafeGraph foot-traffic patterns 2019: raw count of visits to places in this block group",
    "RAW_VISITOR_COUNT": "SafeGraph foot-traffic patterns 2019: raw count of unique visitors to places in this block group",
    "VISITOR_HOME_CBGS": "SafeGraph foot-traffic patterns 2019: JSON map of visitors' home block groups to counts",
    "VISITOR_WORK_CBGS": "SafeGraph foot-traffic patterns 2019: JSON map of visitors' work block groups to counts",
    "DISTANCE_FROM_HOME": "SafeGraph foot-traffic patterns 2019: median distance visitors travelled from home, metres",
    "RELATED_SAME_DAY_BRAND": "SafeGraph foot-traffic patterns 2019: JSON of brands visited the same day",
    "RELATED_SAME_MONTH_BRAND": "SafeGraph foot-traffic patterns 2019: JSON of brands visited the same month",
    "TOP_BRANDS": "SafeGraph foot-traffic patterns 2019: JSON list of most-visited brands",
    "POPULARITY_BY_HOUR": "SafeGraph foot-traffic patterns 2019: JSON array of visits by hour of day",
    "POPULARITY_BY_DAY": "SafeGraph foot-traffic patterns 2019: JSON map of visits by day of week",
}

REDISTRICTING_TABLE = "2020_REDISTRICTING_CBG_DATA"
REDISTRICTING_VINTAGE = "2020 decennial"

# Not indexed: geometry (spatial queries are out of scope), the two demo views
# from the Marketplace listing, and the metadata tables themselves, which are
# described statically in the prompt because every geographic question needs
# them and retrieval cannot be trusted to surface a join rule.
_SKIP_TABLE_RE = re.compile(
    r"GEOMETRY|METADATA|RENT_PERCENTAGE_HOUSEHOLD_INCOME|TOTAL_RENTAL_GEO", re.I
)
_ACS_TABLE_RE = re.compile(r"^(\d{4})_CBG_([A-Z]\d{2})$")

# Rank multipliers for whole ACS table classes. B99xxx "Allocation Of ..."
# tables count how many answers were imputed -- data-quality flags that share
# every word with the real table and are almost never what a question means.
# Race-iteration tables (B19013A..I, "... (White Alone)") are legitimate but
# should lose ties to their base table unless the question names the race.
ALLOCATION_PRIORITY = 0.3
RACE_ITERATION_PRIORITY = 0.8
_ACS_CODE_RE = re.compile(r"^([A-Z]\d{5}[A-Z]?)([em])(\d+)$")
_KEY_COLUMN = "CENSUS_BLOCK_GROUP"


def parse_acs_table(table):
    # type: (str) -> Optional[Tuple[str, str]]
    """'2020_CBG_B19' -> ('2020', 'B19'); None for non-ACS tables."""
    m = _ACS_TABLE_RE.match(table)
    return (m.group(1), m.group(2)) if m else None


def parse_acs_code(column):
    # type: (str) -> Optional[Tuple[str, str, str]]
    """'B19013e1' -> ('B19013', 'e', '1'); None if not an ACS variable code."""
    m = _ACS_CODE_RE.match(column)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def acs_priority(group):
    # type: (str) -> float
    """Rank multiplier for an ACS table id like 'B99051' or 'B19013A'."""
    if group.startswith("B99"):
        return ALLOCATION_PRIORITY
    if re.match(r"^[BC]\d{5}[A-Z]$", group):
        return RACE_ITERATION_PRIORITY
    return 1.0


def is_estimate(column):
    # type: (str) -> bool
    parsed = parse_acs_code(column)
    return bool(parsed) and parsed[1] == "e"


def margin_of_error_column(column):
    # type: (str) -> Optional[str]
    """The MOE column that pairs with an estimate: B19013e1 -> B19013m1."""
    parsed = parse_acs_code(column)
    if not parsed or parsed[1] != "e":
        return None
    return "%sm%s" % (parsed[0], parsed[2])


def describe_acs_field(row):
    # type: (Dict[str, Optional[str]]) -> str
    """Human description from a FIELD_DESCRIPTIONS row.

    The row's own shape is noisy: FIELD_LEVEL_1 is just 'Estimate' or
    'MarginOfError', FIELD_LEVEL_2 repeats the title in upper case, and
    FIELD_LEVEL_3 repeats the universe. What is left after dropping those is
    the actual label path, e.g. 'Total > Male > 22 to 24 years'.
    """
    title = (row.get("TABLE_TITLE") or "").strip()
    universe = (row.get("TABLE_UNIVERSE") or "").strip()
    levels = []
    for i in range(4, 11):
        # The share has a typo in one column name (FIELD_LEVELl_9); accept both.
        val = row.get("FIELD_LEVEL_%d" % i)
        if val is None and i == 9:
            val = row.get("FIELD_LEVELl_9")
        val = (val or "").strip()
        if val and val.lower() not in ("none", "null"):
            levels.append(val)
    path = " > ".join(levels)
    desc = title
    if universe:
        desc += " [universe: %s]" % universe
    if path and path.lower() != title.lower():
        desc += ": " + path
    return desc


def describe_acs_group(row):
    # type: (Dict[str, Optional[str]]) -> str
    """Searchable text for the ACS table a field belongs to: title, universe, topics."""
    return " ".join(
        (row.get(k) or "").strip() for k in ("TABLE_TITLE", "TABLE_UNIVERSE", "TABLE_TOPICS")
    ).strip()


def describe_redistricting_group(row):
    # type: (Dict[str, Optional[str]]) -> str
    return " ".join(
        ["2020 decennial census"]
        + [(row.get(k) or "").strip() for k in ("COLUMN_TOPIC", "COLUMN_UNIVERSE")]
    ).strip()


def describe_redistricting_field(row):
    # type: (Dict[str, Optional[str]]) -> str
    """'RACE [universe: Total population]: White alone' from the P-table metadata."""
    topic = (row.get("COLUMN_TOPIC") or "").strip()
    universe = (row.get("COLUMN_UNIVERSE") or "").strip()
    name = (row.get("FIELD_NAME") or "").strip()
    desc = "2020 decennial census %s" % topic.lower() if topic else "2020 decennial census"
    if universe:
        desc += " [universe: %s]" % universe
    if name:
        desc += ": " + name
    return desc


def build_census_docs(columns, acs_meta, redistricting_meta=None):
    # type: (Iterable[Dict[str, str]], Dict[str, Dict[str, str]], Optional[Dict[str, Dict[str, str]]]) -> List[ColumnDoc]
    """Turn INFORMATION_SCHEMA rows plus the share's metadata into index docs.

    `acs_meta` maps a variable code (from either vintage's metadata table) to
    its description row. `redistricting_meta` maps a P-code the same way.
    Returns one doc per variable, with `vintages` listing every ACS vintage
    whose table actually has that column.
    """
    redistricting_meta = redistricting_meta or {}
    by_code = {}  # type: Dict[str, ColumnDoc]
    order = []  # type: List[str]

    for row in columns:
        table = row["table"]
        column = row["column"]
        if _SKIP_TABLE_RE.search(table) or column == _KEY_COLUMN:
            continue

        acs = parse_acs_table(table)
        if acs:
            vintage, _family = acs
            if not is_estimate(column):
                continue  # MOE columns reachable via the e -> m rule
            meta = acs_meta.get(column, {})
            doc = by_code.get(column)
            if doc is None:
                doc = ColumnDoc(
                    table=table,
                    column=column,
                    data_type=row.get("data_type", ""),
                    description=describe_acs_field(meta) if meta else "",
                    topic=(meta.get("TABLE_TOPICS") or "").strip(),
                    vintages=[vintage],
                    group=parse_acs_code(column)[0],
                    group_label=describe_acs_group(meta) if meta else "",
                    priority=acs_priority(parse_acs_code(column)[0]),
                )
                by_code[column] = doc
                order.append(column)
            else:
                if vintage not in doc.vintages:
                    doc.vintages.append(vintage)
            # The newest vintage is the table the model should use by default.
            if _vintage_rank(vintage) < _vintage_rank(_table_vintage(doc.table)):
                doc.table = table
            continue

        if table == REDISTRICTING_TABLE:
            meta = redistricting_meta.get(column, {})
            doc = ColumnDoc(
                table=table,
                column=column,
                data_type=row.get("data_type", ""),
                description=describe_redistricting_field(meta) if meta else "",
                topic=(meta.get("COLUMN_TOPIC") or "").strip(),
                vintages=[REDISTRICTING_VINTAGE],
                group=column[:4],  # P0010001 -> P001, the decennial table id
                group_label=describe_redistricting_group(meta) if meta else "",
            )
            key = "%s.%s" % (table, column)
            by_code[key] = doc
            order.append(key)
            continue

        if table == PATTERNS_TABLE:
            doc = ColumnDoc(
                table=table,
                column=column,
                data_type=row.get("data_type", ""),
                description=PATTERNS_DESCRIPTIONS.get(column, row.get("comment", "")),
                vintages=["2019"],
            )
            key = "%s.%s" % (table, column)
            by_code[key] = doc
            order.append(key)
            continue

        # Anything else in the share: index it with whatever comment it has,
        # so a new table added to the listing is still reachable.
        doc = ColumnDoc(table=table, column=column,
                        data_type=row.get("data_type", ""),
                        description=row.get("comment", ""))
        key = "%s.%s" % (table, column)
        by_code[key] = doc
        order.append(key)

    for doc in by_code.values():
        doc.vintages.sort(key=_vintage_rank)
    return [by_code[k] for k in order]


def _table_vintage(table):
    # type: (str) -> str
    parsed = parse_acs_table(table)
    return parsed[0] if parsed else ""


def _vintage_rank(vintage):
    # type: (str) -> int
    """Lower is newer / preferred. Unknown vintages sort last."""
    try:
        return ACS_VINTAGES.index(vintage)
    except ValueError:
        return len(ACS_VINTAGES)


def sibling_table(table, vintage):
    # type: (str, str) -> Optional[str]
    """The same ACS family table in another vintage: 2020_CBG_B19 -> 2019_CBG_B19."""
    parsed = parse_acs_table(table)
    if not parsed:
        return None
    return "%s_CBG_%s" % (vintage, parsed[1])
