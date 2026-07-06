"""Semantic retrieval layer for legal contract chunks.

:class:`LegalRetriever` is the sole entry point for query-time retrieval.  It
wires :class:`~app.embeddings.EmbeddingGenerator` (query text → dense vector)
and :class:`~app.vector_store.VectorStore` (dense vector → ranked results) into
a single :meth:`~LegalRetriever.retrieve` call, shielding every caller from
both OpenAI and Pinecone internals.

This module is intentionally narrow:

- No parser imports.
- No LLM imports.
- No retry logic (handled inside :class:`~app.embeddings.EmbeddingGenerator`).
- No prompt construction.

Typical usage::

    from app.retriever import LegalRetriever

    retriever = LegalRetriever()
    results = retriever.retrieve("data breach notification obligations", top_k=5)
    for r in results:
        print(r["section_title"], r["score"])
"""

from app.embeddings import EmbeddingGenerator
from app.vector_store import QueryResult, VectorStore


class LegalRetriever:
    """Performs semantic similarity retrieval over ingested legal contract chunks.

    Combines embedding generation and vector store querying into a single
    :meth:`retrieve` call.  Callers pass a plain query string and receive an
    ordered list of :class:`~app.vector_store.QueryResult` dicts — no OpenAI
    or Pinecone types leak through.

    Construction is **eager**: both the :class:`~app.embeddings.EmbeddingGenerator`
    and :class:`~app.vector_store.VectorStore` are initialised in
    :meth:`__init__`.  This validates credentials and confirms that the
    Pinecone index exists before the first query is attempted, following the
    fail-fast principle.

    Example:
        >>> retriever = LegalRetriever()
        >>> results = retriever.retrieve("limitation of liability clause", top_k=3)
        >>> results[0]["section_title"]
        'Limitation of Liability'
        >>> results[0]["score"]
        0.87
    """

    def __init__(self, namespace: str = "") -> None:
        """Initialise the retriever and validate all external dependencies.

        Constructs an :class:`~app.embeddings.EmbeddingGenerator` and a
        :class:`~app.vector_store.VectorStore`, both of which read credentials
        from :func:`~app.config.get_config`.  The VectorStore constructor
        verifies that the configured Pinecone index exists; if it does not,
        a :exc:`ValueError` is raised here rather than at query time.

        Args:
            namespace: Pinecone namespace to query.  Must match the namespace
                used during ingestion.  Defaults to ``""`` (the default
                namespace).  Pass a custom value when the index is partitioned
                by tenant, environment, or document set.

        Raises:
            ValueError: If any required environment variable is missing
                (propagated from :func:`~app.config.get_config`), or if the
                configured Pinecone index does not exist.
        """
        self._embedder = EmbeddingGenerator()
        self._store = VectorStore(namespace=namespace)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,  # noqa: A002
    ) -> list[QueryResult]:
        """Embed ``query`` and return the ``top_k`` most similar chunks.

        The method embeds the query string with the configured OpenAI model,
        submits the resulting vector to Pinecone, and returns the matches in
        descending order of cosine similarity.

        Args:
            query: Natural-language question or clause description to search
                for.  Must not be empty or consist entirely of whitespace.
            top_k: Maximum number of results to return.  Must be ≥ 1.
                Defaults to 5.  The actual number of results may be lower if
                the index contains fewer matching vectors.
            filter: Optional Pinecone metadata filter expressed as a dict.
                Supports all Pinecone filter operators (``$eq``, ``$in``,
                ``$and``, ``$or``, etc.).

                Restrict results to a single document::

                    filter={"document_name": {"$eq": "nda.pdf"}}

                Restrict results to specific sections::

                    filter={"section": {"$in": ["3.1", "3.2", "3.3"]}}

                Defaults to ``None`` (no filtering).

        Returns:
            A list of :class:`~app.vector_store.QueryResult` dicts ordered
            by descending similarity score.  Each dict contains:
            ``chunk_id``, ``document_name``, ``section``, ``section_title``,
            ``parent_section``, ``pages``, ``chunk_text``, and ``score``.

        Raises:
            ValueError: If ``query`` is empty or blank, or if ``top_k`` is
                less than 1.
            RuntimeError: If the OpenAI embeddings API call fails after all
                retry attempts (propagated from
                :class:`~app.embeddings.EmbeddingGenerator`).

        Example:
            >>> retriever = LegalRetriever()
            >>> results = retriever.retrieve(
            ...     "data retention and deletion obligations",
            ...     top_k=3,
            ...     filter={"document_name": {"$eq": "data_processing_agreement.pdf"}},
            ... )
            >>> len(results) <= 3
            True
            >>> results[0]["score"] >= results[-1]["score"]
            True
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty or blank.")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        query_vector = self._embedder.embed(query)
        return self._store.query(vector=query_vector, top_k=top_k, filter=filter)
