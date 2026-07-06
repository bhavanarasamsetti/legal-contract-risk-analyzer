"""Structured risk scoring for legal contract clauses.

:class:`StructuredRiskScorer` retrieves relevant contract excerpts with
:class:`~app.hybrid_retriever.HybridRetriever`, asks
:class:`~app.llm.LegalContractLLM` to assess risk in structured JSON,
validates the response, and enriches each clause with metadata from the
retrieved sources.

This module is intentionally narrow:

- No parser imports.
- No embedding or vector-store logic.
- No direct OpenAI or Pinecone calls.
- No FastAPI formatting.

Typical usage::

    from app.risk_scoring import StructuredRiskScorer

    scorer = StructuredRiskScorer(top_k=10)
    result = scorer.score("What liability risks exist in the data processing agreement?")
    for clause in result.clauses:
        print(clause.risk_level, clause.clause_title, clause.explanation)
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.hybrid_retriever import HybridRetriever
from app.llm import LegalContractLLM
from app.vector_store import QueryResult

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_RISK_SYSTEM_PROMPT = """\
You are a legal contract risk analyst. Your task is to identify and assess \
risky clauses in the contract excerpts provided to you.

Rules:
1. Base your analysis ONLY on the provided contract excerpts. Do not invent \
clauses or draw on external legal knowledge beyond what the excerpts contain.
2. Identify only clauses that present a material legal or commercial risk \
relative to the user's question.
3. For every identified clause you MUST explain WHY it is risky — cite the \
specific language or obligation that creates the risk.
4. Assign a risk_score integer from 1 to 10:
   - 1–3  → Low risk
   - 4–6  → Medium risk
   - 7–10 → High risk
5. Assign risk_level consistent with the score: "Low", "Medium", or "High".
6. Provide a concrete, actionable recommendation for each identified clause.
7. Use source_index to reference the excerpt number shown in the context \
(e.g. source_index 1 refers to excerpt [1]).
8. Return ONLY valid JSON. No markdown fences, no commentary outside the JSON.

Output schema (return exactly this structure):
{
  "clauses": [
    {
      "source_index": 1,
      "risk_score": 8,
      "risk_level": "High",
      "explanation": "Why this clause is risky, citing specific language.",
      "recommendation": "Concrete action to mitigate or address the risk."
    }
  ]
}

If no material risks are found in the excerpts, return: {"clauses": []}\
"""

_DEFAULT_TOP_K = 10
_DEFAULT_MAX_TOKENS = 4096

_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)

RiskLevel = Literal["Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# Public schema
# ---------------------------------------------------------------------------

class ClauseRiskAssessment(BaseModel):
    """Structured risk assessment for a single contract clause.

    Metadata fields (``clause_text``, ``clause_title``, ``document_name``,
    ``page_numbers``) are populated from the retrieved source chunk, not
    from LLM generation, to prevent hallucinated citations.

    Attributes:
        clause_text: Full text of the assessed clause or subsection.
        clause_title: Descriptive section title, or the section number when
            no title is available.
        document_name: Filename of the source PDF.
        page_numbers: Sorted 1-based page numbers for the clause.
        risk_score: Integer risk score from 1 (lowest) to 10 (highest).
        risk_level: Categorical risk band derived from ``risk_score``.
        explanation: Why this clause is risky, citing specific language.
        recommendation: Concrete action to mitigate or address the risk.
    """

    clause_text: str
    clause_title: str
    document_name: str
    page_numbers: list[int]
    risk_score: int = Field(ge=1, le=10)
    risk_level: RiskLevel
    explanation: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)

    @field_validator("explanation", "recommendation")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        """Reject blank strings after stripping whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty or blank.")
        return stripped


class RiskScoringResult(BaseModel):
    """The project-native return type of :meth:`StructuredRiskScorer.score`.

    Attributes:
        question: The original risk-focus question or topic.
        clauses: Identified risky clauses with structured assessments.
            Empty when no material risks are found or no excerpts were
            retrieved.
        model: OpenAI model ID used for scoring, or empty string when no
            LLM call was made.
        prompt_tokens: Tokens consumed by the scoring prompt.
        completion_tokens: Tokens in the model's JSON response.
        total_tokens: Sum of prompt and completion tokens.
    """

    question: str
    clauses: list[ClauseRiskAssessment]
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Internal LLM response schema (parsed before enrichment)
# ---------------------------------------------------------------------------

class _LLMClauseItem(BaseModel):
    """One clause entry in the raw LLM JSON response."""

    source_index: int = Field(ge=1)
    risk_score: int = Field(ge=1, le=10)
    risk_level: RiskLevel
    explanation: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)


class _LLMResponsePayload(BaseModel):
    """Top-level JSON object expected from the LLM."""

    clauses: list[_LLMClauseItem]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StructuredRiskScorer:
    """Orchestrates hybrid retrieval and structured LLM risk scoring.

    For each :meth:`score` call the scorer:

    1. Retrieves relevant excerpts via :class:`~app.hybrid_retriever.HybridRetriever`.
    2. Sends excerpts and the user's question to
       :class:`~app.llm.LegalContractLLM` with a risk-specific system prompt.
    3. Parses and validates the JSON response.
    4. Enriches each clause with metadata from the matching retrieved source.

    This class does **not**:

    - Call OpenAI or Pinecone directly.
    - Parse PDFs or build embeddings.
    - Format responses for HTTP.

    Example:
        >>> scorer = StructuredRiskScorer()
        >>> result = scorer.score("What are the data breach liability risks?")
        >>> result.clauses[0].risk_level
        'High'
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        namespace: str = "",
        retriever: HybridRetriever | None = None,
        llm: LegalContractLLM | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialise the scorer and its collaborators.

        Args:
            top_k: Number of excerpts to retrieve per scoring request.
                Defaults to ``10`` — higher than Q&A retrieval because
                risk analysis benefits from broader clause coverage.
            namespace: Pinecone namespace forwarded to
                :class:`~app.hybrid_retriever.HybridRetriever`.
            retriever: Optional pre-built hybrid retriever for tests.
            llm: Optional pre-built LLM instance for tests.  When
                ``None``, a :class:`~app.llm.LegalContractLLM` is created
                with ``temperature=0.0`` and an elevated ``max_tokens``
                budget for JSON output.
            system_prompt: Override the default risk-scoring system prompt.
                Defaults to ``None``.

        Raises:
            ValueError: If ``top_k`` is less than 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        self._top_k = top_k
        self._retriever = retriever or HybridRetriever(namespace=namespace)
        self._llm = llm or LegalContractLLM(
            temperature=0.0,
            max_tokens=_DEFAULT_MAX_TOKENS,
            system_prompt=system_prompt or _RISK_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        question: str,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> RiskScoringResult:
        """Identify and score risky clauses for ``question``.

        Retrieves contract excerpts, asks the LLM to produce structured
        JSON risk assessments, validates the output, and enriches each
        clause with metadata from the retrieved sources.

        Args:
            question: A natural-language question or risk focus area
                (e.g. ``"What liability risks exist in termination clauses?"``).
                Must not be empty or blank.
            filter: Optional metadata filter forwarded to hybrid retrieval.
                Same syntax as :meth:`~app.hybrid_retriever.HybridRetriever.retrieve`.

        Returns:
            A :class:`RiskScoringResult` with validated clause assessments
            and token usage metadata.

        Raises:
            ValueError: If ``question`` is empty or blank.
            RuntimeError: If the LLM response is not valid JSON, fails schema
                validation, or references an invalid ``source_index``.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty or blank.")

        sources = self._retriever.retrieve(
            question,
            top_k=self._top_k,
            filter=filter,
        )

        if not sources:
            return RiskScoringResult(
                question=question,
                clauses=[],
                model="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        llm_response = self._llm.complete(question, context=sources)
        clauses = self._parse_and_enrich(llm_response["content"], sources)

        return RiskScoringResult(
            question=question,
            clauses=clauses,
            model=llm_response["model"],
            prompt_tokens=llm_response["prompt_tokens"],
            completion_tokens=llm_response["completion_tokens"],
            total_tokens=llm_response["total_tokens"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_and_enrich(
        self,
        content: str,
        sources: list[QueryResult],
    ) -> list[ClauseRiskAssessment]:
        """Parse LLM JSON and enrich clauses with source metadata."""
        payload = _extract_json_payload(content)

        try:
            parsed = _LLMResponsePayload.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(
                f"LLM risk response failed schema validation: {exc}"
            ) from exc

        assessments: list[ClauseRiskAssessment] = []
        for item in parsed.clauses:
            if item.source_index > len(sources):
                raise RuntimeError(
                    f"LLM referenced source_index {item.source_index} but only "
                    f"{len(sources)} excerpt(s) were provided."
                )

            source = sources[item.source_index - 1]
            assessments.append(
                ClauseRiskAssessment(
                    clause_text=source["chunk_text"],
                    clause_title=_clause_title(source),
                    document_name=source["document_name"],
                    page_numbers=list(source["pages"]),
                    risk_score=item.risk_score,
                    risk_level=_risk_level_from_score(item.risk_score),
                    explanation=item.explanation.strip(),
                    recommendation=item.recommendation.strip(),
                )
            )

        return assessments


def _risk_level_from_score(score: int) -> RiskLevel:
    """Map a 1–10 risk score to a deterministic risk level."""
    if score <= 3:
        return "Low"
    if score <= 6:
        return "Medium"
    return "High"


def _clause_title(source: QueryResult) -> str:
    """Return the best available title for a source chunk."""
    if source["section_title"]:
        return source["section_title"]
    return source["section"]


def _extract_json_payload(content: str) -> dict[str, Any]:
    """Extract a JSON object from raw LLM text, stripping markdown fences."""
    text = content.strip()
    fence_match = _JSON_FENCE_PATTERN.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM risk response is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "LLM risk response must be a JSON object with a 'clauses' key."
        )

    return payload
