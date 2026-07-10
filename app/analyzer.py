"""Orchestration layer for legal contract question answering.

:class:`RiskAnalyzer` is the top-level entry point for answering questions
about ingested legal contracts.  It wires :class:`~app.retriever.LegalRetriever`
(question → relevant chunks) and :class:`~app.llm.LegalContractLLM`
(chunks + question → answer) into a single :meth:`~RiskAnalyzer.analyze` call
and returns a self-contained :class:`AnalysisResult` TypedDict.

This module is intentionally narrow:

- No configuration reads (``get_config`` is not imported).
- No direct OpenAI or Pinecone calls.
- No prompt construction.
- No retrieval logic.
- No embedding logic.
- No parser imports.

All business logic lives inside the collaborators.  ``RiskAnalyzer`` owns
only the orchestration: call retriever, call LLM, assemble result.

Typical usage::

    from app.analyzer import RiskAnalyzer

    analyzer = RiskAnalyzer(top_k=5)
    result = analyzer.analyze("What are the data breach notification obligations?")
    print(result["answer"])
    for source in result["sources"]:
        print(source["document_name"], source["section"])
"""

from typing import TypedDict

# NOTE: do not import 'trace' (e.g. from numpy) - we use 'trace' as the
# context variable from langfuse.start_as_current_observation().

from app.llm import LegalContractLLM
from app.retriever import LegalRetriever
from app.vector_store import QueryResult
from app.langfuse_client import langfuse

# ---------------------------------------------
# NEW TYPES
# ---------------------------------------------

class Finding(TypedDict):
    severity: str
    title: str
    description: str


class Recommendation(TypedDict):
    title: str
    description: str


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

class AnalysisResult(TypedDict):
    """The project-native return type of :meth:`RiskAnalyzer.analyze`.

    Contains everything produced by a single question-answering operation:
    the original question, the LLM's answer, the retrieved source chunks,
    the model identifier, and token usage counts.

    This is a self-contained record.  No OpenAI, Pinecone, or other SDK
    types appear in its fields, keeping every caller independent of the
    underlying infrastructure.

    Attributes:
        question: The original question passed to :meth:`~RiskAnalyzer.analyze`.
            Included so that the result is self-contained when stored,
            logged, or passed through async pipelines.
        answer: The LLM's text response grounded in the retrieved excerpts.
        sources: The contract chunks retrieved from Pinecone and passed to
            the LLM as context.  Ordered by descending similarity score.
            Each element is a :class:`~app.vector_store.QueryResult` dict
            containing ``chunk_id``, ``document_name``, ``section``,
            ``section_title``, ``parent_section``, ``pages``, ``chunk_text``,
            and ``score``.
        model: The model ID that generated the answer, as reported by the
            OpenAI API (e.g. ``"gpt-4o-mini-2024-07-18"``).  Useful for
            cost attribution and audit logs.
        prompt_tokens: Number of tokens consumed by the prompt sent to the
            LLM (system message + context + question).
        completion_tokens: Number of tokens in the LLM's answer.
        total_tokens: Sum of ``prompt_tokens`` and ``completion_tokens``.
    """

    question: str
    answer: str

    risk_score: float
    risk_level: str
    confidence: int
    high_risk: int
    medium_risk: int
    low_risk: int

    findings: list[Finding]

    recommendations: list[Recommendation]

    sources: list[QueryResult]

    model: str

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    contract_name: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RiskAnalyzer:
    """Orchestrates retrieval and LLM completion for legal contract analysis.

    ``RiskAnalyzer`` is the top-level entry point for answering questions
    about ingested legal contracts.  It holds a
    :class:`~app.retriever.LegalRetriever` and a
    :class:`~app.llm.LegalContractLLM`, calls them in order, and returns
    a single :class:`AnalysisResult` dict.

    This class does **not**:

    - Read configuration or environment variables.
    - Call OpenAI or Pinecone directly.
    - Build prompts or manage token budgets.
    - Parse documents or perform retrieval.
    - Calculate risk scores (future responsibility).

    Construction is **eager**: unless collaborators are injected, real
    :class:`~app.retriever.LegalRetriever` and
    :class:`~app.llm.LegalContractLLM` instances are created in
    :meth:`__init__`.  The retriever will verify that the configured
    Pinecone index exists, so misconfigured environments fail at startup
    rather than at the first query.

    For unit tests, pass pre-constructed mock objects via the ``retriever``
    and ``llm`` parameters so that no real API credentials are needed.

    Example:
        >>> analyzer = RiskAnalyzer(top_k=5)
        >>> result = analyzer.analyze(
        ...     "What are the termination-for-convenience rights?",
        ...     filter={"document_name": {"$eq": "employment_agreement.pdf"}},
        ... )
        >>> result["answer"]
        'According to [1], either party may terminate...'
        >>> len(result["sources"]) <= 5
        True
        >>> result["total_tokens"]
        743
    """

    def __init__(
        self,
        top_k: int = 5,
        namespace: str = "",
        retriever: LegalRetriever | None = None,
        llm: LegalContractLLM | None = None,
    ) -> None:
        """Initialise the analyzer and its collaborators.

        In normal use, omit ``retriever`` and ``llm`` — the analyzer
        constructs them internally.  Pass pre-built instances only in tests
        or when injecting alternative implementations (e.g. a LangChain
        retriever).

        Args:
            top_k: Number of contract chunks to retrieve per question.
                Passed to :meth:`~app.retriever.LegalRetriever.retrieve`
                on every :meth:`analyze` call.  Must be ≥ 1.  Defaults to
                ``5``.  Different analysis modes (quick vs. thorough) are
                represented by different ``RiskAnalyzer`` instances rather
                than per-call overrides.
            namespace: Pinecone namespace to query.  Must match the
                namespace used during ingestion.  Forwarded to the
                :class:`~app.retriever.LegalRetriever` constructor.
                Defaults to ``""`` (the default namespace).
            retriever: Optional pre-built :class:`~app.retriever.LegalRetriever`
                instance.  When ``None`` (the default), one is constructed
                internally using ``namespace``.  Pass a mock in tests to
                avoid real Pinecone calls.
            llm: Optional pre-built :class:`~app.llm.LegalContractLLM`
                instance.  When ``None`` (the default), one is constructed
                with default parameters.  Pass a mock in tests to avoid
                real OpenAI calls.

        Raises:
            ValueError: If ``top_k`` is less than 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        self._top_k = top_k
        self._retriever = retriever or LegalRetriever(namespace=namespace)
        self._llm = llm or LegalContractLLM()

    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        question: str,
        filter: dict | None = None,  # noqa: A002
    ) -> AnalysisResult:
        """Answer a question about the ingested contracts.

        Retrieves the most relevant contract chunks for ``question``, passes
        them to the LLM alongside the question, and returns an
        :class:`AnalysisResult` containing the answer, source citations,
        and token usage.

        Args:
            question: A natural-language question about one or more of the
                ingested contracts.  Must not be empty or consist entirely
                of whitespace.
            filter: Optional Pinecone metadata filter to scope retrieval to
                a subset of the indexed contracts.  Supports all Pinecone
                filter operators (``$eq``, ``$in``, ``$and``, ``$or``,
                etc.).

                Restrict results to one document::

                    filter={"document_name": {"$eq": "nda.pdf"}}

                Restrict to specific sections::

                    filter={"section": {"$in": ["3.1", "3.2"]}}

                Defaults to ``None`` (no filtering — search the full index).

        Returns:
            An :class:`AnalysisResult` dict with the following fields:

            - ``question``: the original question (echoed back).
            - ``answer``: the LLM's text response, grounded in the
              retrieved excerpts.
            - ``sources``: list of retrieved :class:`~app.vector_store.QueryResult`
              dicts, ordered by descending similarity score.
            - ``model``: the LLM model ID used.
            - ``prompt_tokens``, ``completion_tokens``, ``total_tokens``:
              token usage for the LLM call.

        Raises:
            ValueError: If ``question`` is empty or blank.
            RuntimeError: If the embedding API or the Chat Completions API
                fails after all retry attempts (propagated from the
                collaborators).

        Example:
            >>> analyzer = RiskAnalyzer()
            >>> result = analyzer.analyze("Who is liable for data breaches?")
            >>> result["answer"]
            'According to [2], the data processor is solely liable...'
            >>> result["sources"][0]["document_name"]
            'data_processing_agreement.pdf'
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty or blank.")
        with langfuse.start_as_current_observation(
            name="Contract Analysis",
            as_type="chain",
            input={
                "question": question,
                "filter": filter,
            },
        ) as trace:
            sources = self._retriever.retrieve(
                question,
                top_k=self._top_k,
                filter=filter,
            )

            llm_response = self._llm.complete(question, context=sources)
            try:
                risk_score = int(llm_response.get("risk_score", 50))
                risk_score = max(0, min(100, risk_score))
            except (ValueError, TypeError):
                risk_score = 50

            if risk_score < 40:
                risk_level = "Low"
            elif risk_score < 70:
                risk_level = "Medium"
            else:
                risk_level = "High"

            confidence = llm_response["confidence"]
            findings = llm_response["findings"]
            recommendations = llm_response["recommendations"]
            high = len([f for f in findings if f["severity"] == "High"])
            medium = len([f for f in findings if f["severity"] == "Medium"])
            low = len([f for f in findings if f["severity"] == "Low"])
            contract_name = (
                sources[0]["document_name"] if sources else "Unknown Contract"
            )
            unique_sources = []
            seen = set()

            for source in sources:
                key = (
                    source["document_name"],
                    source["section"],
                    tuple(source["pages"]),
                )

                if key not in seen:
                    seen.add(key)
                    unique_sources.append(source)

            sources = unique_sources
            trace.update(
                output={
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "contract": contract_name,
                }
            )

        return AnalysisResult(
            question=question,
            answer=llm_response["answer"],
            risk_score=float(risk_score),
            risk_level=risk_level,
            confidence=confidence,
            high_risk=high,
            medium_risk=medium,
            low_risk=low,
            findings=findings,
            recommendations=recommendations,
            sources=sources,
            model=llm_response["model"],
            prompt_tokens=llm_response["prompt_tokens"],
            completion_tokens=llm_response["completion_tokens"],
            total_tokens=llm_response["total_tokens"],
            contract_name=contract_name,
        )
