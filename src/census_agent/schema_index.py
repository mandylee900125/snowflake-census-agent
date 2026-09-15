"""Schema retrieval.

The Census share exposes thousands of columns, most named with opaque ACS
codes (B19013e1) whose meaning lives in a separate field-descriptions table.
Two consequences drive this module:

  1. The schema cannot go in the prompt. We retrieve the ~20 columns relevant
     to each question and show the model only those.
  2. Column names alone are not searchable. We enrich each column with its
     human-readable description before indexing, so "median household income"
     can actually match B19013e1.

Retrieval is BM25 at two levels. ACS data is organised as ~365 tables
(B25003 "Tenure"), each a grid of cells (B25003e3 "Renter occupied"). Cells
in one table share most of their words, so a flat column index lets a big
table flood the results and crowd out the right answer from a small one. So:
score the table and the cell separately, add them, and cap how many cells any
one table can contribute. See DECISIONS.md for why lexical search rather than
embeddings.
"""
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from . import config

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")

# BM25's IDF goes negative for a term present in more than half the documents
# ("total", "population"), which makes a *match* on that term lower a score.
# Flooring IDF keeps every match non-negative, so (a) a common-but-relevant
# term still helps a little, and (b) multiplying by a priority < 1 is a
# penalty rather than, for a negative score, a boost.
_IDF_FLOOR = 0.1
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Used only to decide whether a question and a column genuinely overlap --
# not removed from the BM25 corpus, which handles common terms via IDF.
_STOPWORDS = frozenset([
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with",
    "and", "or", "is", "are", "was", "were", "be", "been", "as", "that",
    "this", "it", "its", "how", "what", "which", "who", "where", "when",
    "many", "much", "do", "does", "did", "can", "could", "would", "should",
    "there", "their", "have", "has", "had", "me", "my", "i", "you", "your",
    "show", "tell", "give", "get", "find", "list", "s", "t",
])


def stem(token):
    # type: (str) -> str
    """Light suffix stripping so 'unemployment', 'unemployed' and 'unemploy'
    land on the same term. Deliberately crude: the corpus and the query go
    through the same function, so consistency matters more than linguistics.
    A real stemmer (Snowball) is a one-line upgrade if this proves too blunt.
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    for suffix, replacement, min_len in (
        ("ies", "y", 5), ("sses", "ss", 6), ("ment", "", 6), ("ing", "", 6),
        ("ed", "", 5), ("s", "", 4),
    ):
        if token.endswith(suffix) and len(token) >= min_len:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                return token
            return token[: -len(suffix)] + replacement
    return token


def terms(text):
    # type: (str) -> List[str]
    """Tokenize then stem: what the index and the query both use."""
    return [stem(t) for t in tokenize(text)]


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, with snake_case and camelCase split apart.

    'MEDIAN_HOUSEHOLD_INCOME' and 'medianHouseholdIncome' must both produce
    ['median', 'household', 'income'] or column names won't match questions.
    """
    if not text:
        return []
    text = _CAMEL_RE.sub(" ", text)
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class ColumnDoc(object):
    """One searchable column.

    `vintages` lists the dataset editions that carry this column (e.g.
    ["2020", "2019"]); `table` is the copy the model should use by default.
    `topic` is the metadata table's subject tags, indexed for search but not
    shown to the model, which sees the description instead.
    """

    def __init__(self, table, column, data_type="", description="", topic="",
                 vintages=None, group=None, group_label="", priority=1.0):
        # type: (str, str, str, str, str, Optional[List[str]], Optional[str], str, float) -> None
        self.table = table
        self.column = column
        self.data_type = data_type
        self.description = description
        self.topic = topic
        self.vintages = list(vintages or [])
        # The ACS table this cell belongs to (B25003), and its searchable
        # title. Columns with no finer grouping use their SQL table.
        self.group = group or table
        self.group_label = group_label
        # Rank multiplier. <1 for columns that are technically valid but
        # rarely what a question means (e.g. imputation-flag tables).
        self.priority = float(priority)

    @property
    def qualified_name(self):
        # type: () -> str
        return '%s."%s"' % (self.table, self.column)

    def search_text(self):
        # type: () -> str
        """What the column-level index sees: name, table, and description.

        `topic` is deliberately absent. It is table-wide ("Race" lists every
        race), so putting it on each cell would make "White alone" match a
        question about Asians. It belongs to the group index instead.
        """
        return " ".join([self.table, self.column, self.description])

    def to_dict(self):
        # type: () -> Dict[str, Any]
        d = {
            "table": self.table,
            "column": self.column,
            "data_type": self.data_type,
            "description": self.description,
        }  # type: Dict[str, Any]
        if self.topic:
            d["topic"] = self.topic
        if self.vintages:
            d["vintages"] = self.vintages
        if self.group != self.table:
            d["group"] = self.group
        if self.group_label:
            d["group_label"] = self.group_label
        if self.priority != 1.0:
            d["priority"] = self.priority
        return d

    @classmethod
    def from_dict(cls, d):
        # type: (Dict[str, Any]) -> ColumnDoc
        return cls(
            table=d["table"],
            column=d["column"],
            data_type=d.get("data_type", ""),
            description=d.get("description", ""),
            topic=d.get("topic", ""),
            vintages=d.get("vintages"),
            group=d.get("group"),
            group_label=d.get("group_label", ""),
            priority=d.get("priority", 1.0),
        )

    def render(self):
        # type: () -> str
        """One line of schema context for the SQL-generation prompt."""
        desc = self.description or "(no description available)"
        line = '- %s."%s" (%s) -- %s' % (self.table, self.column, self.data_type or "?", desc)
        if len(self.vintages) > 1:
            line += "  [also in: %s]" % ", ".join(
                "%s_%s" % (v, self.table.split("_", 1)[1]) for v in self.vintages[1:]
            )
        return line


def _floored(bm25):
    for term, idf in bm25.idf.items():
        if idf < _IDF_FLOOR:
            bm25.idf[term] = _IDF_FLOOR
    return bm25


class SchemaIndex(object):
    """BM25 index over column documents."""

    def __init__(self, docs):
        # type: (List[ColumnDoc]) -> None
        if not docs:
            raise ValueError("Cannot build a schema index with zero columns.")
        self.docs = docs
        from rank_bm25 import BM25Okapi
        # Each term once per document. Census labels repeat their title in
        # the universe and the path ("Bachelor's Degrees [universe: BACHELOR'S
        # DEGREE MAJORS]"), and raw term frequency would reward that noise.
        corpus = [sorted(set(terms(d.search_text()))) for d in docs]
        self._token_sets = [set(tokens) for tokens in corpus]
        self._bm25 = _floored(BM25Okapi(corpus))

        # Second index over the groups (ACS tables). One doc per group, built
        # from its label or, failing that, the words its columns share.
        group_ids = []  # type: List[str]
        group_text = {}  # type: Dict[str, str]
        for d in docs:
            if d.group not in group_text:
                group_ids.append(d.group)
                group_text[d.group] = " ".join([d.group_label or d.table, d.topic]).strip()
        self._group_ids = group_ids
        self._group_pos = dict((g, i) for i, g in enumerate(group_ids))
        group_corpus = [sorted(set(terms(group_text[g]))) for g in group_ids]
        self._group_token_sets = [set(t) for t in group_corpus]
        self._group_bm25 = _floored(BM25Okapi(group_corpus))
        self._doc_group_pos = [self._group_pos[d.group] for d in docs]

    def __len__(self):
        # type: () -> int
        return len(self.docs)

    def tables(self):
        # type: () -> List[str]
        seen = []
        for d in self.docs:
            if d.table not in seen:
                seen.append(d.table)
        return seen

    def search(self, query, limit=None, max_per_group=None):
        # type: (str, Optional[int], Optional[int]) -> List[ColumnDoc]
        """Return the columns most likely to answer `query`, best first.

        A column's score is its own BM25 score plus its group's, so a cell in
        the right table outranks a similar-looking cell in the wrong one. No
        group may fill more than `max_per_group` of the slots, which is what
        stops a 70-column table from burying a one-column answer.
        """
        limit = limit or config.MAX_SCHEMA_CANDIDATES
        max_per_group = max_per_group or config.MAX_CANDIDATES_PER_GROUP
        tokens = terms(query)
        if not tokens:
            return []
        # Score on content words only. Function words are rare in Census
        # labels, which makes them high-IDF: "are" would otherwise let
        # "People who are White alone" outrank "Total > Female" for "how
        # many women are there".
        content = [t for t in tokens if t not in _STOPWORDS] or tokens
        col_scores = self._bm25.get_scores(content)
        group_scores = self._group_bm25.get_scores(content)

        # Relevance gate: require a real term overlap, then rank the survivors
        # by score. Overlap, not a score threshold, is the signal: with the
        # IDF floor a match on a very common word scores barely above zero,
        # and a threshold would silently drop correct matches.
        content = set(content)
        candidates = [
            i for i in range(len(self.docs))
            if (self._token_sets[i] | self._group_token_sets[self._doc_group_pos[i]]) & content
        ]
        candidates.sort(
            key=lambda i: (col_scores[i] + group_scores[self._doc_group_pos[i]]) * self.docs[i].priority,
            reverse=True,
        )

        taken = {}  # type: Dict[str, int]
        out = []  # type: List[ColumnDoc]
        for i in candidates:
            g = self.docs[i].group
            if taken.get(g, 0) >= max_per_group:
                continue
            taken[g] = taken.get(g, 0) + 1
            out.append(self.docs[i])
            if len(out) >= limit:
                break
        return out

    def render_context(self, query, limit=None):
        # type: (str, Optional[int]) -> str
        """The schema snippet injected into the SQL-generation prompt."""
        hits = self.search(query, limit)
        if not hits:
            return "(no columns in the dataset matched this question)"
        by_table = {}  # type: Dict[str, List[ColumnDoc]]
        for doc in hits:
            by_table.setdefault(doc.table, []).append(doc)
        blocks = []
        for table, cols in by_table.items():
            lines = ["TABLE %s" % table] + [c.render() for c in cols]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # --- persistence --------------------------------------------------------

    def save(self, path=None):
        # type: (Optional[str]) -> str
        path = path or config.SCHEMA_INDEX_PATH
        with open(path, "w") as fh:
            json.dump([d.to_dict() for d in self.docs], fh)
        return path

    @classmethod
    def load(cls, path=None):
        # type: (Optional[str]) -> SchemaIndex
        path = path or config.SCHEMA_INDEX_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                "No schema index at %s. Run: python scripts/build_index.py" % path
            )
        with open(path) as fh:
            return cls([ColumnDoc.from_dict(d) for d in json.load(fh)])


def build_docs(columns, descriptions=None):
    # type: (Iterable[Dict[str, str]], Optional[Dict[str, str]]) -> List[ColumnDoc]
    """Turn INFORMATION_SCHEMA rows into searchable docs.

    `descriptions` maps a bare column name (e.g. 'B19013e1') to its human
    description, sourced from the dataset's own field-metadata table. Without
    it, coded columns are effectively invisible to lexical search.
    """
    descriptions = descriptions or {}
    docs = []
    for row in columns:
        name = row["column"]
        desc = row.get("comment") or descriptions.get(name) or descriptions.get(name.upper()) or ""
        docs.append(
            ColumnDoc(
                table=row["table"],
                column=name,
                data_type=row.get("data_type", ""),
                description=desc,
            )
        )
    return docs
