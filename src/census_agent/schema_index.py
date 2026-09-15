"""Schema retrieval.

The Census share exposes thousands of columns, most named with opaque ACS
codes (B19013e1) whose meaning lives in a separate field-descriptions table.
Two consequences drive this module:

  1. The schema cannot go in the prompt. We retrieve the ~20 columns relevant
     to each question and show the model only those.
  2. Column names alone are not searchable. We enrich each column with its
     human-readable description before indexing, so "median household income"
     can actually match B19013e1.

Retrieval is BM25 over those enriched descriptions. See DECISIONS.md for why
lexical search rather than embeddings.
"""
import json
import os
import re
from typing import Dict, Iterable, List, Optional

from . import config

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")
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
    """One searchable column."""

    def __init__(self, table, column, data_type="", description=""):
        # type: (str, str, str, str) -> None
        self.table = table
        self.column = column
        self.data_type = data_type
        self.description = description

    @property
    def qualified_name(self):
        # type: () -> str
        return '%s."%s"' % (self.table, self.column)

    def search_text(self):
        # type: () -> str
        """What BM25 actually indexes: name, table, and human description."""
        return " ".join([self.table, self.column, self.description])

    def to_dict(self):
        # type: () -> Dict[str, str]
        return {
            "table": self.table,
            "column": self.column,
            "data_type": self.data_type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d):
        # type: (Dict[str, str]) -> ColumnDoc
        return cls(
            table=d["table"],
            column=d["column"],
            data_type=d.get("data_type", ""),
            description=d.get("description", ""),
        )

    def render(self):
        # type: () -> str
        """One line of schema context for the SQL-generation prompt."""
        desc = self.description or "(no description available)"
        return '- %s."%s" (%s) -- %s' % (self.table, self.column, self.data_type or "?", desc)


class SchemaIndex(object):
    """BM25 index over column documents."""

    def __init__(self, docs):
        # type: (List[ColumnDoc]) -> None
        if not docs:
            raise ValueError("Cannot build a schema index with zero columns.")
        self.docs = docs
        from rank_bm25 import BM25Okapi
        corpus = [tokenize(d.search_text()) for d in docs]
        self._token_sets = [set(tokens) for tokens in corpus]
        self._bm25 = BM25Okapi(corpus)

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

    def search(self, query, limit=None):
        # type: (str, Optional[int]) -> List[ColumnDoc]
        """Return the columns most likely to answer `query`, best first."""
        limit = limit or config.MAX_SCHEMA_CANDIDATES
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)

        # Relevance gate: require a real content-word overlap, then rank the
        # survivors by score. Do NOT gate on `score > 0` -- BM25 IDF goes
        # negative for terms that appear in most documents, so a correct match
        # on a common word ("income", "housing") can score at or below zero.
        # Sign is an artefact of the weighting; overlap is the actual signal.
        content = set(t for t in tokens if t not in _STOPWORDS)
        if not content:
            content = set(tokens)
        candidates = [i for i in range(len(self.docs)) if self._token_sets[i] & content]
        candidates.sort(key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in candidates[:limit]]

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
