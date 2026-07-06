"""Hybrid dense + sparse retrieval for legal contract chunks.

:class:`HybridRetriever` combines semantic vector search (Pinecone via
:class:`~app.retriever.LegalRetriever`) with keyword search (BM25 via
:class:`~app.bm25_index.BM25Index`), merges the candidate lists, and
reranks them with Reciprocal Rank Fusion (RRF).

The public :meth:`retrieve` signature matches
:class:`~app.retriever.LegalRetriever`, so ``HybridRetriever`` can be
injected into :class:`~app.analyzer.RiskAnalyzer` without any changes to
the analyzer or FastAPI layers.

Typical usage::

    from app.hybrid_retriever import HybridRetriever
    from app.analyzer import RiskAnalyzer

    retriever = HybridRetriever()
    results = retriever.retrieve("Section 3.2 confidentiality obligations", top_k=5)

    analyzer = RiskAnalyzer(retriever=retriever)
    answer = analyzer.analyze("What are the confidentiality obligations?")
"""

from pathlib import Path
from typing import Any

from app.bm25_index import BM25Index, DEFAULT_INDEX_PATH
from app.retriever import LegalRetriever
from app.vector_store import QueryResult

# Standard RRF constant from the original fusion literature (Cormack et al.).
_DEFAULT_RRF_K = 60

# Fetch more candidates from each channel than the final ``top_k`` so that
# chunks found by only one retrieval path still have a chance after fusion.
_DEFAULT_FETCH_MULTIPLIER = 2


class HybridRetriever:
    """Combines dense vector retrieval and BM25 keyword retrieval.

    For each query the retriever:

    1. Fetches dense candidates from Pinecone via
       :class:`~app.retriever.LegalRetriever`.
    2. Fetches sparse candidates from the local BM25 index.
    3. Merges both lists by ``chunk_id``.
    4. Reranks the merged set with Reciprocal Rank Fusion (RRF).
    5. Returns the top ``top_k`` fused results.

    This class does **not**:

    - Modify :class:`~app.vector_store.VectorStore` or
      :class:`~app.embeddings.EmbeddingGenerator`.
    - Build prompts or call LLMs.
    - Parse PDFs.

    Example:
        >>> retriever = HybridRetriever()
        >>> results = retriever.retrieve("data breach notification", top_k=5)
        >>> results[0]["chunk_id"]
        'data_processing_agreement_pdf_4_2.1_a1b2c3d4'
    """

    def __init__(
        self,
        namespace: str = "",
        bm25_index_path: Path | str = DEFAULT_INDEX_PATH,
        dense_retriever: LegalRetriever | None = None,
        bm25_index: BM25Index | None = None,
        rrf_k: int = _DEFAULT_RRF_K,
        fetch_multiplier: int = _DEFAULT_FETCH_MULTIPLIER,
    ) -> None:
        """Initialise hybrid retrieval components.

        In normal use, omit ``dense_retriever`` and ``bm25_index`` — both
        are constructed internally.  Pass pre-built instances in tests to
        avoid real API calls and disk dependencies.

        Args:
            namespace: Pinecone namespace for dense retrieval.  Forwarded
                to :class:`~app.retriever.LegalRetriever`.
            bm25_index_path: Path to the persisted BM25 JSON index.
                Used only when ``bm25_index`` is ``None``.
            dense_retriever: Optional pre-built dense retriever.
            bm25_index: Optional pre-loaded BM25 index.
            rrf_k: RRF smoothing constant.  Higher values reduce the
                influence of rank position.  Defaults to ``60``.
            fetch_multiplier: Number of candidates to fetch from each
                channel is ``top_k * fetch_multiplier``.  Defaults to ``2``.

        Raises:
            ValueError: If ``rrf_k`` or ``fetch_multiplier`` is less than 1.
            FileNotFoundError: If the BM25 index file does not exist and
                ``bm25_index`` is not provided.
        """
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be at least 1, got {rrf_k}.")
        if fetch_multiplier < 1:
            raise ValueError(
                f"fetch_multiplier must be at least 1, got {fetch_multiplier}."
            )

        self._dense = dense_retriever or LegalRetriever(namespace=namespace)
        self._bm25 = bm25_index or BM25Index.load(bm25_index_path)
        self._rrf_k = rrf_k
        self._fetch_multiplier = fetch_multiplier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[QueryResult]:
        """Retrieve and rerank contract chunks using hybrid search.

        Dense and sparse channels each return up to
        ``top_k * fetch_multiplier`` candidates.  The lists are fused
        with RRF and the top ``top_k`` results are returned.

        Args:
            query: Natural-language question or keyword phrase.  Must not
                be empty or consist entirely of whitespace.
            top_k: Number of final results to return.  Must be ≥ 1.
            filter: Optional metadata filter applied to both retrieval
                channels.  Uses the same dict syntax as
                :meth:`~app.vector_store.VectorStore.query`.

        Returns:
            A list of :class:`~app.vector_store.QueryResult` dicts ordered
            by descending fused RRF score.

        Raises:
            ValueError: If ``query`` is empty or blank, or if ``top_k`` is
                less than 1.
            RuntimeError: If the dense embedding API call fails after all
                retry attempts.
            FileNotFoundError: If the BM25 index was not loaded at
                construction time and is missing on a subsequent reload
                attempt (not applicable when index is injected).
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty or blank.")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        candidate_k = top_k * self._fetch_multiplier

        dense_results = self._dense.retrieve(
            query,
            top_k=candidate_k,
            filter=filter,
        )
        sparse_results = self._bm25.search(
            query,
            top_k=candidate_k,
            filter=filter,
        )

        fused = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=self._rrf_k,
        )
        return fused[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        *,
        dense_results: list[QueryResult],
        sparse_results: list[QueryResult],
        rrf_k: int,
    ) -> list[QueryResult]:
        """Merge ranked lists with Reciprocal Rank Fusion.

        RRF score for document ``d``:

            score(d) = Σ  1 / (rrf_k + rank_i)

        where ``rank_i`` is the 1-based rank of ``d`` in each list it
        appears in.  This method is scale-free: cosine similarity scores
        and raw BM25 scores never need normalisation.

        Args:
            dense_results: Dense channel results, highest score first.
            sparse_results: Sparse channel results, highest score first.
            rrf_k: RRF smoothing constant.

        Returns:
            Deduplicated results ordered by descending fused score.
        """
        fused_scores: dict[str, float] = {}
        chunk_by_id: dict[str, QueryResult] = {}

        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            chunk_by_id[chunk_id] = result

        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            chunk_by_id.setdefault(chunk_id, result)

        sorted_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )

        return [
            QueryResult(
                chunk_id=chunk_id,
                document_name=chunk_by_id[chunk_id]["document_name"],
                section=chunk_by_id[chunk_id]["section"],
                section_title=chunk_by_id[chunk_id]["section_title"],
                parent_section=chunk_by_id[chunk_id]["parent_section"],
                pages=list(chunk_by_id[chunk_id]["pages"]),
                chunk_text=chunk_by_id[chunk_id]["chunk_text"],
                score=fused_scores[chunk_id],
            )
            for chunk_id in sorted_ids
        ]
