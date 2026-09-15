"""Tests for the share-specific index construction.

These pin the decisions that make retrieval work on the real dataset: one doc
per variable across vintages, estimates only, descriptions built from the
metadata hierarchy, and the ranking rules that stop big or irrelevant tables
from burying the right answer.
"""
from census_agent import census_layout as layout
from census_agent.schema_index import SchemaIndex, stem


def meta_row(code, title, universe, topics="", *levels):
    row = {"TABLE_ID": code, "TABLE_NUMBER": code.split("e")[0], "TABLE_TITLE": title,
           "TABLE_TOPICS": topics, "TABLE_UNIVERSE": universe,
           "FIELD_LEVEL_1": "Estimate", "FIELD_LEVEL_2": title.upper(), "FIELD_LEVEL_3": universe}
    for i, level in enumerate(levels, start=4):
        row["FIELD_LEVEL_%d" % i] = level
    return row


def col(table, column, data_type="NUMBER"):
    return {"table": table, "column": column, "data_type": data_type, "comment": ""}


class TestCodeParsing:
    def test_acs_table_and_code(self):
        assert layout.parse_acs_table("2020_CBG_B19") == ("2020", "B19")
        assert layout.parse_acs_table("2020_METADATA_CBG_FIPS_CODES") is None
        assert layout.parse_acs_code("B19013e1") == ("B19013", "e", "1")
        assert layout.parse_acs_code("B19013Ae1") == ("B19013A", "e", "1")
        assert layout.parse_acs_code("CENSUS_BLOCK_GROUP") is None

    def test_margin_of_error_pairing(self):
        assert layout.margin_of_error_column("B19013e1") == "B19013m1"
        assert layout.margin_of_error_column("B19013m1") is None

    def test_priorities(self):
        assert layout.acs_priority("B19013") == 1.0
        assert layout.acs_priority("B19013A") < 1.0   # race iteration
        assert layout.acs_priority("B99051") < layout.acs_priority("B19013A")  # allocation flag


class TestDescriptions:
    def test_drops_the_noise_levels_and_keeps_the_label_path(self):
        row = meta_row("B01001e10", "Sex By Age", "Total population", "Age and Sex",
                       "Total", "Male", "22 to 24 years")
        assert layout.describe_acs_field(row) == (
            "Sex By Age [universe: Total population]: Total > Male > 22 to 24 years"
        )

    def test_title_only_field_is_not_repeated(self):
        row = meta_row("B19013e1", "Median Household Income", "Households", "",
                       "Median household income")
        assert layout.describe_acs_field(row) == "Median Household Income [universe: Households]"

    def test_tolerates_the_shares_misspelt_level_column(self):
        row = meta_row("X", "T", "U", "", "a", "b", "c", "d", "e")
        row["FIELD_LEVELl_9"] = "nine"   # sic: the real table has this typo
        assert layout.describe_acs_field(row).endswith("> e > nine")

    def test_redistricting_description(self):
        row = {"FIELD_NAME": "White alone", "COLUMN_ID": "P0010003",
               "COLUMN_TOPIC": "RACE", "COLUMN_UNIVERSE": "Total population"}
        assert layout.describe_redistricting_field(row) == (
            "2020 decennial census race [universe: Total population]: White alone"
        )


class TestBuildCensusDocs:
    META = {
        "B19013e1": meta_row("B19013e1", "Median Household Income", "Households",
                             "Income Households Families Individuals"),
        "B02001e5": meta_row("B02001e5", "Race", "Total population",
                             "Asian, Black or African American, White", "Total", "Asian alone"),
        "B02001e2": meta_row("B02001e2", "Race", "Total population",
                             "Asian, Black or African American, White", "Total", "White alone"),
    }
    COLUMNS = [
        col("2019_CBG_B19", "CENSUS_BLOCK_GROUP", "TEXT"),
        col("2019_CBG_B19", "B19013e1"), col("2019_CBG_B19", "B19013m1"),
        col("2020_CBG_B19", "CENSUS_BLOCK_GROUP", "TEXT"),
        col("2020_CBG_B19", "B19013e1"), col("2020_CBG_B19", "B19013m1"),
        col("2020_CBG_B02", "B02001e5"), col("2020_CBG_B02", "B02001e2"),
        col("2020_CBG_GEOMETRY", "GEOMETRY", "GEOGRAPHY"),
        col("2020_METADATA_CBG_FIPS_CODES", "COUNTY", "TEXT"),
        col("2020_REDISTRICTING_CBG_DATA", "P0010003"),
    ]
    REDISTRICTING = {"P0010003": {"FIELD_NAME": "White alone", "COLUMN_ID": "P0010003",
                                  "COLUMN_TOPIC": "RACE", "COLUMN_UNIVERSE": "Total population"}}

    def docs(self):
        return layout.build_census_docs(self.COLUMNS, self.META, self.REDISTRICTING)

    def test_one_doc_per_variable_with_both_vintages_recorded(self):
        docs = {d.column: d for d in self.docs()}
        assert docs["B19013e1"].vintages == ["2020", "2019"]
        assert docs["B19013e1"].table == "2020_CBG_B19"   # newest is the default
        assert "2019_CBG_B19" in docs["B19013e1"].render()

    def test_margin_of_error_key_and_metadata_columns_are_not_indexed(self):
        names = [d.column for d in self.docs()]
        assert "B19013m1" not in names
        assert "CENSUS_BLOCK_GROUP" not in names
        assert "GEOMETRY" not in names
        assert "COUNTY" not in names

    def test_redistricting_columns_are_indexed_with_their_own_vintage(self):
        docs = {d.column: d for d in self.docs()}
        assert docs["P0010003"].vintages == ["2020 decennial"]
        assert docs["P0010003"].group == "P001"
        assert "White alone" in docs["P0010003"].description

    def test_group_and_priority_come_from_the_acs_code(self):
        docs = {d.column: d for d in self.docs()}
        assert docs["B19013e1"].group == "B19013"
        assert docs["B19013e1"].priority == 1.0

    def test_table_topics_do_not_leak_into_sibling_columns(self):
        # Regression: the Race table's topic list names every race, so with
        # topics indexed per column "White alone" matched "asian population".
        index = SchemaIndex(self.docs())
        assert index.search("asian population")[0].column == "B02001e5"


class TestRanking:
    def test_stemming_bridges_query_and_label_inflections(self):
        assert stem("unemployment") == stem("unemployed")
        assert stem("veterans") == stem("veteran")
        assert stem("families") == stem("family")
        assert stem("bus") == "bus"   # not everything ending in s is plural

    def test_a_large_table_cannot_fill_every_slot(self):
        big = [layout.ColumnDoc("T", "big%d" % i, "NUMBER", "Tenure by units in structure: renter occupied %d" % i,
                                group="B25032") for i in range(30)]
        small = layout.ColumnDoc("T", "B25003e3", "NUMBER", "Tenure: Total > Renter occupied", group="B25003")
        hits = SchemaIndex(big + [small]).search("renter occupied", limit=10, max_per_group=5)
        assert "B25003e3" in [h.column for h in hits]
        assert sum(1 for h in hits if h.group == "B25032") <= 5

    def test_low_priority_group_loses_a_tie(self):
        real = layout.ColumnDoc("T", "B05002e13", "NUMBER", "Nativity: Foreign born", group="B05002")
        flag = layout.ColumnDoc("T", "B99051e5", "NUMBER", "Allocation of citizenship: Foreign born",
                                group="B99051", priority=layout.ALLOCATION_PRIORITY)
        hits = SchemaIndex([flag, real]).search("foreign born")
        assert hits[0].column == "B05002e13"

    def test_stopwords_in_the_question_do_not_score(self):
        # "are" is rare in Census labels and therefore high-IDF; it must not
        # let "People who are White alone" beat the plain Female count.
        female = layout.ColumnDoc("T", "B01001e26", "NUMBER", "Sex By Age: Total > Female", group="B01001")
        decoy = layout.ColumnDoc("T", "B01002Ae3", "NUMBER",
                                 "Median Age By Sex (White Alone) [universe: People who are White alone]: Female",
                                 group="B01002A")
        hits = SchemaIndex([decoy, female]).search("how many women are there female")
        assert hits[0].column == "B01001e26"
