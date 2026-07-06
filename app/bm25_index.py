"""Local BM25 sparse index for legal contract chunks.

:class:`BM25Index` provides keyword-based retrieval over the same chunk
corpus that is stored in Pinecone.  The index is built offline from
parser output and persisted to disk as JSON.  At query time it is loaded
into memory and searched with :pypi:`rank_bm25`.

This module is intentionally narrow:

- No embedding logic.
- No Pinecone logic.
- No LLM logic.
- No parser logic (callers supply pre-built chunk records).

Typical usage::

    from app.bm25_index import BM25Index

    index = BM25Index.from_chunks(chunks)
    index.save(DEFAULT_INDEX_PATH)

    loaded = BM25Index.load(DEFAULT_INDEX_PATH)
    results = loaded.search("data breach notification", top_k=5)
"""

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.chunker import ChunkData
from app.vector_store import QueryResult

# Default on-disk location for the persisted BM25 corpus.
DEFAULT_INDEX_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "bm25_index.json"
)

_INDEX_VERSION = 1
_TOKEN_PATTERN = re.compile(r"\w+")


class BM25Index:
    """In-memory BM25 index over legal contract chunks.

    Stores the full chunk metadata alongside a :class:`rank_bm25.BM25Okapi`
    scorer built from tokenized ``chunk_text`` values.  The index is
    persisted as JSON so it can be rebuilt independently of Pinecone
    ingestion.

    Example:
        >>> index = BM25Index.from_chunks(chunks)
        >>> index.save(path)
        >>> loaded = BM25Index.load(path)
        >>> results = loaded.search("limitation of liability", top_k=3)
        >>> results[0]["section"]
        '8.1'
    """

    def __init__(self, chunks: list[ChunkData]) -> None:
        """Build an in-memory BM25 index from ``chunks``.

        Args:
            chunks: Non-empty list of chunk records.  Each record must
                include a unique ``chunk_id`` and non-empty ``chunk_text``.

        Raises:
            ValueError: If ``chunks`` is empty or any chunk is missing
                required fields.
        """
        if not chunks:
            raise ValueError("chunks must not be empty.")

        self._chunks = list(chunks)
        self._chunk_ids = [chunk["chunk_id"] for chunk in self._chunks]
        if len(set(self._chunk_ids)) != len(self._chunk_ids):
            raise ValueError("chunks must have globally unique chunk_id values.")

        self._tokenized_corpus = [
            self._tokenize(chunk["chunk_text"]) for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_chunks(cls, chunks: list[ChunkData]) -> "BM25Index":
        """Build a new index from parser-produced chunk records.

        Args:
            chunks: Non-empty list of :class:`~app.chunker.ChunkData` dicts.

        Returns:
            A ready-to-search :class:`BM25Index` instance.
        """
        return cls(chunks)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX_PATH) -> "BM25Index":
        """Load a persisted index from disk.

        Args:
            path: Path to the JSON index file created by :meth:`save`.
                Defaults to :data:`DEFAULT_INDEX_PATH`.

        Returns:
            A :class:`BM25Index` instance rebuilt from the saved corpus.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is malformed or has an unsupported
                version number.
        """
        index_path = Path(path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at '{index_path}'.  "
                "Run 'python scripts/build_bm25_index.py' to create it."
            )

        with index_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)

        if payload.get("version") != _INDEX_VERSION:
            raise ValueError(
                f"Unsupported BM25 index version: {payload.get('version')!r}.  "
                f"Expected {_INDEX_VERSION}.  Rebuild the index."
            )

        chunks: list[ChunkData] = payload["chunks"]
        return cls(chunks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, path: Path | str = DEFAULT_INDEX_PATH) -> None:
        """Persist the chunk corpus to disk.

        Only the chunk records are saved.  The BM25 scorer is rebuilt on
        load, which is fast for corpora of a few hundred chunks.

        Args:
            path: Destination JSON file.  Parent directories are created
                automatically.  Defaults to :data:`DEFAULT_INDEX_PATH`.
        """
        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": _INDEX_VERSION,
            "chunks": self._chunks,
        }

        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[QueryResult]:
        """Return the top BM25 matches for ``query``.

        Args:
            query: Natural-language question or keyword phrase.  Must not
                be empty or blank.
            top_k: Maximum number of results to return.  Must be ≥ 1.
            filter: Optional metadata filter using the same dict syntax as
                :meth:`~app.vector_store.VectorStore.query`.  Only chunks
                that pass the filter are eligible.  Supported operators:
                ``$eq``, ``$in``, ``$and``, ``$or``.

        Returns:
            A list of :class:`~app.vector_store.QueryResult` dicts ordered
            by descending BM25 score.

        Raises:
            ValueError: If ``query`` is empty or blank, or if ``top_k`` is
                less than 1.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty or blank.")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        tokenized_query = self._tokenize(query)

        # Score every document, then rank eligible chunks.
        raw_scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(raw_scores)),
            key=lambda i: raw_scores[i],
            reverse=True,
        )

        results: list[QueryResult] = []
        for index in ranked_indices:
            score = float(raw_scores[index])
            if score <= 0.0:
                break

            chunk = self._chunks[index]
            if filter is not None and not self._matches_filter(chunk, filter):
                continue

            results.append(self._chunk_to_result(chunk, score))
            if len(results) >= top_k:
                break

        return results

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 using lowercase word tokens."""
        return _TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _chunk_to_result(chunk: ChunkData, score: float) -> QueryResult:
        """Convert a chunk record to a :class:`~app.vector_store.QueryResult`."""
        return QueryResult(
            chunk_id=chunk["chunk_id"],
            document_name=chunk["document_name"],
            section=chunk["section"],
            section_title=chunk["section_title"],
            parent_section=chunk["parent_section"],
            pages=list(chunk["pages"]),
            chunk_text=chunk["chunk_text"],
            score=score,
        )

    @staticmethod
    def _matches_filter(chunk: ChunkData, filter_expr: dict[str, Any]) -> bool:
        """Evaluate a Pinecone-style metadata filter against one chunk."""
        if "$and" in filter_expr:
            return all(
                BM25Index._matches_filter(chunk, sub)
                for sub in filter_expr["$and"]
            )
        if "$or" in filter_expr:
            return any(
                BM25Index._matches_filter(chunk, sub)
                for sub in filter_expr["$or"]
            )

        for field, condition in filter_expr.items():
            if field.startswith("$"):
                continue

            value = chunk.get(field)  # type: ignore[arg-type]
            if isinstance(condition, dict):
                if "$eq" in condition:
                    if value != condition["$eq"]:
                        return False
                elif "$in" in condition:
                    allowed = condition["$in"]
                    if isinstance(value, list):
                        if not any(str(item) in allowed for item in value):
                            return False
                    elif str(value) not in allowed:
                        return False
                else:
                    raise ValueError(
                        f"Unsupported filter operator in field '{field}': "
                        f"{list(condition.keys())}"
                    )
            else:
                if value != condition:
                    return False

        return True
