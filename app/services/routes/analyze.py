"""Contract analysis route.

``POST /analyze`` accepts a natural-language question about the ingested
legal contracts, delegates the full analysis to
:class:`~app.analyzer.RiskAnalyzer`, and returns a structured
:class:`AnalyzeResponse`.

This module owns:

- The :class:`AnalyzeRequest` Pydantic model (HTTP request schema).
- The :class:`SourceReference` Pydantic model (serialized citation).
- The :class:`AnalyzeResponse` Pydantic model (HTTP response schema).
- The :func:`get_analyzer` FastAPI dependency.
- The ``POST /analyze`` route handler.

This module does **not**:

- Call OpenAI or Pinecone directly.
- Perform retrieval or prompt construction.
- Contain business logic.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.analyzer import RiskAnalyzer

router = APIRouter(tags=["Analysis"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for ``POST /analyze``.

    Attributes:
        question: A natural-language question about one or more ingested
            legal contracts.  Must be at least 1 character long.  A
            whitespace-only string passes this constraint but will be
            rejected with ``422`` by :class:`~app.analyzer.RiskAnalyzer`.
        filter: Optional Pinecone metadata filter to scope retrieval to a
            subset of the indexed contracts.  Supports all Pinecone filter
            operators (``$eq``, ``$in``, ``$and``, ``$or``, etc.).

            Restrict results to one document:

            .. code-block:: json

                {"filter": {"document_name": {"$eq": "nda.pdf"}}}

            Restrict to specific sections:

            .. code-block:: json

                {"filter": {"section": {"$in": ["3.1", "3.2"]}}}

            Omit or pass ``null`` to search the full index (default).
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        description="Natural-language question about the ingested contracts.",
        examples=["What are the data breach notification obligations?"],
    )
    filter: dict | None = Field(  # noqa: A003
        default=None,
        description=(
            "Optional Pinecone metadata filter.  Omit or pass null to "
            "search the full index."
        ),
        examples=[{"document_name": {"$eq": "nda.pdf"}}],
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SourceReference(BaseModel):
    """A single retrieved contract excerpt cited in the analysis.

    This is a projected view of :class:`~app.vector_store.QueryResult`.
    The ``chunk_text`` field is intentionally excluded: returning the full
    clause text of every source would inflate response payloads and expose
    complete contract content in API logs and client caches.  Callers
    receive the citation metadata needed to locate the source; the full
    text remains server-side.

    Attributes:
        chunk_id: Unique identifier of the matched chunk.
        document_name: Filename of the source PDF.
        section: Heading token that opened this chunk (e.g. ``"3.2"``).
        section_title: Descriptive title of the section, or empty string.
        parent_section: Heading token of the nearest ancestor section,
            or empty string for top-level sections.
        pages: Sorted list of 1-based page numbers that contributed text
            to this chunk.
        score: Cosine similarity score in ``[0.0, 1.0]``.
    """

    chunk_id: str
    document_name: str
    section: str
    section_title: str
    parent_section: str
    pages: list[int]
    score: float
class Finding(BaseModel):
    severity: str
    title: str
    description: str


class Recommendation(BaseModel):
    severity: str
    title: str
    description: str

class AnalyzeResponse(BaseModel):
    """Response body for ``POST /analyze``.


    Attributes:
        question: The original question, echoed back for logging and
            traceability.
        answer: The LLM's text response, grounded in the retrieved contract
            excerpts.  Citations appear as ``[1]``, ``[2]``, etc.,
            corresponding to positions in ``sources``.
        sources: Retrieved contract excerpts that informed the answer,
            ordered by descending similarity score.  ``chunk_text`` is
            excluded; use the ``section`` and ``document_name`` fields to
            locate the full clause.
        model: The OpenAI model ID that generated the answer (e.g.
            ``"gpt-4o-mini-2024-07-18"``).  Useful for cost attribution.
        prompt_tokens: Tokens consumed by the prompt (system message +
            context + question).
        completion_tokens: Tokens in the model's answer.
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
    sources: list[SourceReference]
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    contract_name: str

# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_analyzer(request: Request) -> RiskAnalyzer:
    """FastAPI dependency: retrieve the shared :class:`~app.analyzer.RiskAnalyzer`.

    Reads the analyzer instance from ``app.state``, which is set during
    the application lifespan startup.  Raises ``503`` if the analyzer is
    not yet initialised (startup still in progress).

    Args:
        request: The incoming HTTP request, used to access ``app.state``.

    Returns:
        The shared :class:`~app.analyzer.RiskAnalyzer` instance.

    Raises:
        HTTPException: ``503`` if ``app.state.analyzer`` is ``None``.
    """
    analyzer = getattr(request.app.state, "analyzer", None)
    if analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is starting up. Please retry in a moment.",
        )
    return analyzer


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a legal contract question",
    description=(
        "Embeds the question, retrieves the most relevant contract excerpts, "
        "and returns an LLM-generated answer grounded in those excerpts."
    ),
)
async def analyze_contract(
    body: AnalyzeRequest,
    analyzer: RiskAnalyzer = Depends(get_analyzer),
) -> AnalyzeResponse:
    """Answer a natural-language question about the ingested contracts.

    Delegates entirely to :meth:`~app.analyzer.RiskAnalyzer.analyze`.
    All retrieval, embedding, prompt construction, and LLM interaction
    are handled by the underlying service layer — this handler only
    validates HTTP concerns and maps domain exceptions to HTTP status codes.

    Args:
        body: The validated request body containing ``question`` and
            optional ``filter``.
        analyzer: The shared :class:`~app.analyzer.RiskAnalyzer` instance,
            injected by :func:`get_analyzer`.

    Returns:
        An :class:`AnalyzeResponse` containing the answer, source
        citations, model identifier, and token usage.

    Raises:
        HTTPException:
            - ``422`` if the question is blank or semantically invalid.
            - ``503`` if OpenAI or Pinecone is temporarily unavailable.
            - ``500`` for unexpected internal errors.
    """
    try:
        result = analyzer.analyze(body.question, filter=body.filter)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return AnalyzeResponse(
        question=result["question"],
        answer=result["answer"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        high_risk=result["high_risk"],
        medium_risk=result["medium_risk"],
        low_risk=result["low_risk"],
        findings=[
            Finding.model_validate(item)
            for item in result["findings"]
        ],
        recommendations=[
            Recommendation.model_validate(item)
            for item in result["recommendations"]
        ],
        sources=[SourceReference.model_validate(s) for s in result["sources"]],
        model=result["model"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
        contract_name=result["contract_name"],
    )
