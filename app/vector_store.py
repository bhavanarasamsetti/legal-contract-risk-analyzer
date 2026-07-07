from typing import Any, TypedDict

from pinecone import (
    Pinecone,
    ServerlessSpec,
    Vector,
)

from app.chunker import ChunkData
from app.config import get_config


class QueryResult(TypedDict):
    """A single result returned by :meth:`VectorStore.query`.

    Contains all metadata fields from the original :class:`~app.chunker.ChunkData`
    plus the similarity score assigned by Pinecone.

    This is a project-native type.  No Pinecone SDK types appear in its
    fields, so callers above this layer do not depend on Pinecone internals.

    Attributes:
        chunk_id: Unique identifier of the matched chunk
            (e.g. ``"nda_pdf_1_3.2"``).
        document_name: Filename of the source PDF
            (e.g. ``"nda.pdf"``).
        section: Heading token that opened this chunk (e.g. ``"3.2"``).
        section_title: Descriptive title from the heading line, or empty
            string when the heading carries no title.
        parent_section: Heading token of the nearest ancestor section, or
            empty string for top-level sections.
        pages: Sorted list of 1-based page numbers that contributed text
            to this chunk.
        chunk_text: Full text of the clause or subsection.
        score: Cosine similarity score in ``[0.0, 1.0]``.  Higher is more
            similar to the query vector.
    """

    chunk_id: str
    document_name: str
    section: str
    section_title: str
    parent_section: str
    pages: list[int]
    chunk_text: str
    score: float


class VectorStore:
    """Manages persistent vector storage for legal contract chunks in Pinecone.

    This class is the sole interface between the application and the Pinecone
    vector database.  It owns index connection management, metadata serialisation,
    upsert batching, and result deserialisation.

    Responsibilities:

    - Connect to an existing Pinecone serverless index on construction.
    - Upsert ``(vector, metadata)`` pairs derived from
      :class:`~app.chunker.ChunkData` dicts.
    - Query by dense vector similarity with optional metadata filtering.
    - Delete vectors by ID, metadata filter, or namespace wipe.

    This class does **not**:

    - Generate embeddings — receives pre-computed ``list[float]`` vectors.
    - Parse documents — receives fully assembled
      :class:`~app.chunker.ChunkData` dicts.
    - Construct prompts or call LLMs.

    The Pinecone vector ID is set to ``chunk_id``, making upserts naturally
    idempotent: re-running the ingestion pipeline overwrites existing vectors
    rather than duplicating them.

    Metadata type mapping:

    Pinecone metadata values are limited to ``str``, ``int``, ``float``,
    ``bool``, and ``list[str]``.  The ``pages`` field (``list[int]`` in
    :class:`~app.chunker.ChunkData`) is stored as ``list[str]`` and converted
    back to ``list[int]`` when results are returned.

    Example:
        >>> store = VectorStore()
        >>> store.upsert_chunks(chunks, vectors)
        250
        >>> results = store.query(query_vector, top_k=5)
        >>> results[0]["section_title"]
        'Confidentiality Obligations'
    """

    def __init__(
        self,
        namespace: str = "",
        upsert_batch_size: int = 10,
    ) -> None:
        """Connect to an existing Pinecone index.

        Reads ``pinecone_api_key`` and ``pinecone_index_name`` from
        :func:`~app.config.get_config`.  The index must already exist;
        use :meth:`create_index` for first-time setup.

        Args:
            namespace: Pinecone namespace to use for all operations.
                Defaults to ``""`` (the default namespace).  Namespaces
                allow logical partitioning within a single index without
                additional cost.
            upsert_batch_size: Number of vectors per upsert batch.
                Pinecone recommends 100 for serverless indexes.  Must be
                between 1 and 1,000.

        Raises:
            ValueError: If ``upsert_batch_size`` is outside ``[1, 1000]``,
                or if the configured index name does not exist.
        """
        if not 1 <= upsert_batch_size <= 1000:
            raise ValueError(
                f"upsert_batch_size must be between 1 and 1000, got {upsert_batch_size}."
            )

        config = get_config()
        self._namespace = namespace
        self._upsert_batch_size = upsert_batch_size

        self._pc = Pinecone(api_key=config.pinecone_api_key)
        index_name = config.pinecone_index_name

        # Verify the index exists before acquiring the handle.  Pinecone's
        # Index() constructor does not validate existence at call time; a
        # missing index only surfaces on the first actual API operation.
        # Checking here produces a clear ValueError immediately rather than a
        # confusing error buried inside the first upsert or query.
        existing = {idx.name for idx in self._pc.list_indexes()}
        if index_name not in existing:
            raise ValueError(
                f"Pinecone index '{index_name}' does not exist.  "
                "Run VectorStore.create_index() or create it in the Pinecone console "
                "before connecting."
            )

        self._index = self._pc.Index(index_name)

    # ------------------------------------------------------------------
    # Index lifecycle (classmethod — infrastructure, not runtime)
    # ------------------------------------------------------------------

    @classmethod
    def create_index(
        cls,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
    ) -> None:
        """Create a new Pinecone serverless index for this project.

        This is a one-time infrastructure operation, typically called from
        ``scripts/ingest.py`` before the first upsert.  It reads
        ``pinecone_api_key`` and ``pinecone_index_name`` from
        :func:`~app.config.get_config`.

        For ``text-embedding-3-small`` pass ``dimension=1536``.
        For ``text-embedding-3-large`` pass ``dimension=3072``.

        Args:
            dimension: Vector dimensionality.  Must match the embedding
                model used to generate the vectors that will be upserted.
            cloud: Cloud provider for the serverless index.
                Supported values: ``"aws"``, ``"gcp"``, ``"azure"``.
                Defaults to ``"aws"``.
            region: Cloud region for the serverless index
                (e.g. ``"us-east-1"``).  Must be a region supported by
                the chosen ``cloud`` provider.
            metric: Distance metric used for similarity search.
                Defaults to ``"cosine"``.  Other options: ``"euclidean"``,
                ``"dotproduct"``.

        Raises:
            ValueError: If the index already exists, to prevent accidental
                overwrites of production data.
            PineconeApiException: For other Pinecone API errors.
        """
        config = get_config()
        pc = Pinecone(api_key=config.pinecone_api_key)
        index_name = config.pinecone_index_name

        existing = {idx.name for idx in pc.list_indexes()}
        if index_name in existing:
            raise ValueError(
                f"Pinecone index '{index_name}' already exists.  "
                "Delete it first or choose a different PINECONE_INDEX_NAME."
            )

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[ChunkData],
        vectors: list[list[float]],
        show_progress: bool = False,
    ) -> int:
        """Upsert a list of chunks and their corresponding vectors into Pinecone.

        Each chunk is stored as a Pinecone vector whose ID is the
        ``chunk_id`` field and whose metadata holds all remaining
        :class:`~app.chunker.ChunkData` fields.  Using ``chunk_id`` as the
        vector ID makes upserts idempotent: re-running the ingestion pipeline
        overwrites existing vectors rather than creating duplicates.

        The ``pages`` field is converted from ``list[int]`` to ``list[str]``
        before storage because Pinecone metadata only supports ``list[str]``.
        It is converted back to ``list[int]`` by :meth:`query`.

        Args:
            chunks: Ordered list of :class:`~app.chunker.ChunkData` dicts.
            vectors: Ordered list of dense float vectors, one per chunk.
                Must be the same length as ``chunks``.
            show_progress: If ``True``, display a tqdm progress bar during
                batched upsert.  Useful for long ingestion runs in a terminal.
                Defaults to ``False`` for clean script output.

        Returns:
            Total number of vectors successfully upserted.

        Raises:
            ValueError: If ``chunks`` and ``vectors`` have different lengths,
                or if either list is empty.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have the same length; "
                f"got {len(chunks)} chunks and {len(vectors)} vectors."
            )
        if not chunks:
            raise ValueError("chunks must not be empty.")

        pinecone_vectors = [
            Vector(
                id=chunk["chunk_id"],
                values=vector,
                metadata=self._chunk_to_metadata(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        total_upserted = 0

        for i in range(0, len(pinecone_vectors), self._upsert_batch_size):
            batch = pinecone_vectors[i:i + self._upsert_batch_size]

            response = self._index.upsert(
                vectors=batch,
                namespace=self._namespace,
            )
            total_upserted += response.upserted_count

        return total_upserted

            

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """Query the index for the most similar vectors to ``vector``.

        Args:
            vector: The dense query vector.  Must have the same
                dimensionality as the vectors stored in the index.
            top_k: Number of nearest neighbours to return.  Must be ≥ 1.
                Defaults to 5.
            filter: Optional Pinecone metadata filter expressed as a dict.
                Supports all Pinecone filter operators (``$eq``, ``$in``,
                ``$and``, etc.).  Example — restrict to a single document:

                .. code-block:: python

                    filter={"document_name": {"$eq": "nda.pdf"}}

                Example — restrict to specific pages:

                .. code-block:: python

                    filter={"pages": {"$in": ["3", "4"]}}

        Returns:
            A list of :class:`QueryResult` dicts ordered by descending
            similarity score (most relevant first).  May be shorter than
            ``top_k`` if the index contains fewer matching vectors.

        Raises:
            ValueError: If ``top_k`` is less than 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        response = self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=self._namespace,
            filter=filter,
            include_metadata=True,
        )

        return [
            self._scored_vector_to_result(match)
            for match in response.matches
        ]

    def delete(
        self,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
    ) -> None:
        """Delete vectors from the index.

        Exactly one of ``ids``, ``filter``, or ``delete_all`` should be
        provided.  If none are provided this method is a no-op.

        Args:
            ids: List of vector IDs (``chunk_id`` values) to delete.
            filter: Pinecone metadata filter; all matching vectors are
                deleted.  Example — delete all chunks from one document:

                .. code-block:: python

                    filter={"document_name": {"$eq": "old_contract.pdf"}}

            delete_all: If ``True``, delete every vector in the namespace.
                Use with caution — this cannot be undone.

        Raises:
            ValueError: If more than one of the mutually exclusive
                arguments are provided simultaneously.
        """
        provided = sum([
            ids is not None,
            filter is not None,
            delete_all,
        ])
        if provided > 1:
            raise ValueError(
                "Provide at most one of: ids, filter, or delete_all=True."
            )

        self._index.delete(
            ids=ids,
            filter=filter,
            delete_all=delete_all,
            namespace=self._namespace,
        )

    def stats(self) -> dict[str, Any]:
        """Return index statistics for the configured namespace.

        Returns:
            A dict with at least ``total_vector_count`` and
            ``dimension`` keys, as reported by Pinecone.

        Example:
            >>> store = VectorStore()
            >>> store.stats()
            {'dimension': 1536, 'total_vector_count': 250, ...}
        """
        response = self._index.describe_index_stats()
        return {
            "dimension": response.dimension,
            "total_vector_count": response.total_vector_count,
            "namespaces": {
                ns: info.vector_count
                for ns, info in (response.namespaces or {}).items()
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_metadata(chunk: ChunkData) -> dict[str, Any]:
        """Serialise a :class:`~app.chunker.ChunkData` into a Pinecone metadata dict.

        Pinecone metadata values are restricted to ``str``, ``int``,
        ``float``, ``bool``, and ``list[str]``.  The ``pages`` field
        (``list[int]``) is converted to ``list[str]`` to satisfy this
        constraint while preserving Pinecone's native ``$in`` filter support.

        Args:
            chunk: The chunk whose fields are to be stored as metadata.

        Returns:
            A dict mapping metadata key names to Pinecone-compatible values.
        """
        return {
            "chunk_id": chunk["chunk_id"],
            "document_name": chunk["document_name"],
            "section": chunk["section"],
            "section_title": chunk["section_title"],
            "parent_section": chunk["parent_section"],
            # Convert list[int] → list[str]: Pinecone only supports list[str]
            "pages": [str(p) for p in chunk["pages"]],
            "chunk_text": chunk["chunk_text"],
        }

    @staticmethod
    def _scored_vector_to_result(match: Any) -> QueryResult:
        """Convert a Pinecone ``ScoredVector`` to a :class:`QueryResult`.

        Pinecone-specific types are fully unwrapped here.  No Pinecone SDK
        types appear in the returned dict, keeping the upstream call stack
        independent of the vector database implementation.

        Args:
            match: A ``pinecone.ScoredVector`` object from a query response.

        Returns:
            A :class:`QueryResult` dict with all fields populated.
        """
        meta = match.metadata or {}
        return QueryResult(
            chunk_id=meta.get("chunk_id", match.id),
            document_name=meta.get("document_name", ""),
            section=meta.get("section", ""),
            section_title=meta.get("section_title", ""),
            parent_section=meta.get("parent_section", ""),
            # Convert list[str] back to list[int]
            pages=[int(p) for p in meta.get("pages", [])],
            chunk_text=meta.get("chunk_text", ""),
            score=match.score,
        )
