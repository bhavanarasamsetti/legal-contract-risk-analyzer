"""LLM interaction layer for legal contract analysis.

:class:`LegalContractLLM` is the sole interface between the application and
the OpenAI Chat Completions API.  It accepts a user question and a list of
retrieved :class:`~app.vector_store.QueryResult` dicts, builds a structured
prompt, calls the API with retry, and returns the model response as a
project-native :class:`LLMResponse` TypedDict.

This module is intentionally narrow:

- No retrieval logic.
- No parser logic.
- No risk-scoring or answer parsing (those belong to ``RiskAnalyzer``).
- No FastAPI formatting.

Typical usage::

    from app.llm import LegalContractLLM
    from app.retriever import LegalRetriever

    retriever = LegalRetriever()
    llm = LegalContractLLM()

    results = retriever.retrieve("data breach notification obligations", top_k=5)
    response = llm.complete("What are the data breach notification obligations?", results)
    print(response["content"])
"""

import time
from typing import TypedDict

import tiktoken
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.config import get_config
from app.vector_store import QueryResult

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default system prompt for legal contract analysis.  Establishes the model's
# role, grounding constraint, citation requirement, conflict-detection rule,
# and uncertainty-handling behaviour.  Can be overridden via the constructor.
_DEFAULT_SYSTEM_PROMPT = """\
You are a legal contract analysis assistant. Your task is to answer questions \
about legal contracts based exclusively on the contract excerpts provided to you.

Rules:
1. Base your answer ONLY on the provided contract excerpts. Do not draw on \
external legal knowledge beyond what the excerpts explicitly contain.
2. Cite the source number (e.g. [1], [2]) whenever you refer to a specific excerpt.
3. If the answer cannot be determined from the provided excerpts, state this \
clearly: "The provided excerpts do not contain sufficient information to answer \
this question."
4. If excerpts from different documents contain conflicting terms, identify the \
conflict explicitly and name both documents.
5. Be precise and factual. Do not speculate about intent or make legal judgements \
beyond what is stated in the text.
6. Keep your response focused and directly relevant to the question asked.\
"""

# Seconds to wait before the first retry.  Each subsequent attempt doubles
# this value (exponential backoff).
_INITIAL_BACKOFF_SECONDS = 2

# Number of times to attempt an API call before propagating the exception.
_MAX_ATTEMPTS = 3

# Estimated token overhead per context block for the "[N] Document: ..." header
# line.  Used in _trim_context to account for formatting tokens beyond chunk_text.
_CONTEXT_HEADER_TOKENS = 25

# Fallback tiktoken encoding when the configured model is not in the registry.
# cl100k_base covers GPT-4 and earlier; o200k_base covers GPT-4o and GPT-4o-mini.
# Using cl100k_base as a conservative fallback avoids under-counting tokens.
_FALLBACK_ENCODING = "cl100k_base"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class LLMResponse(TypedDict):
    """The project-native return type of :meth:`LegalContractLLM.complete`.

    Unwraps the OpenAI SDK response so that no SDK types appear outside
    ``app/llm.py``.  Callers receive plain Python primitives.

    Attributes:
        content: The model's text answer.
        model: The model ID that was actually used, as reported by the API
            (e.g. ``"gpt-4o-mini-2024-07-18"``).  Useful for logging and
            cost attribution.
        prompt_tokens: Number of tokens consumed by the prompt (system message
            + user message).
        completion_tokens: Number of tokens in the model's response.
        total_tokens: Sum of ``prompt_tokens`` and ``completion_tokens``.
    """

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LegalContractLLM:
    """Manages OpenAI Chat Completions calls for legal contract question answering.

    This class is the sole interface between the application and the OpenAI
    Chat Completions API.  It owns prompt construction, token-budget
    management, and retry logic.

    Responsibilities:

    - Build a system prompt and a structured user message from the question
      and retrieved context chunks.
    - Trim context to stay within the configured token budget before sending.
    - Call ``client.chat.completions.create`` with exponential-backoff retry
      on transient API errors.
    - Unwrap the SDK response into an :class:`LLMResponse` TypedDict.

    This class does **not**:

    - Retrieve context from Pinecone (that is :class:`~app.retriever.LegalRetriever`).
    - Calculate risk scores or parse the model's answer (that is ``RiskAnalyzer``).
    - Format responses for HTTP (that is the FastAPI layer).

    Example:
        >>> llm = LegalContractLLM()
        >>> response = llm.complete(
        ...     "What are the data retention obligations?",
        ...     context=retrieved_results,
        ... )
        >>> response["content"]
        'According to [1], the data processor must...'
        >>> response["total_tokens"]
        812
    """

    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_context_tokens: int = 6_000,
        system_prompt: str | None = None,
    ) -> None:
        """Initialise the LLM client and validate all parameters.

        Reads ``openai_api_key`` and ``openai_chat_model`` from
        :func:`~app.config.get_config` and constructs the OpenAI client.
        No network call is made at construction time.

        Args:
            temperature: Sampling temperature passed to the Chat Completions
                API.  Must be in ``[0.0, 2.0]``.  Defaults to ``0.0`` for
                deterministic, reproducible answers — appropriate for legal
                analysis where consistency is required.
            max_tokens: Maximum number of tokens the model may generate in
                its response (``max_completion_tokens`` in the API).  Must
                be ≥ 1.  Defaults to ``1024`` (~750 words), sufficient for
                focused clause analysis.
            max_context_tokens: Maximum number of tokens that may be
                consumed by injected context chunks.  Chunks that would
                exceed this budget (lowest-scored first) are silently
                dropped before the API call.  Defaults to ``6000``, which
                is safe for all common models including GPT-4 (8k window).
            system_prompt: Override the default legal analysis system
                prompt.  Pass a custom string when ``RiskAnalyzer`` needs
                a structured-output prompt (e.g. JSON risk scores).
                Defaults to ``None``, which uses the module-level
                ``_DEFAULT_SYSTEM_PROMPT``.

        Raises:
            ValueError: If ``temperature`` is outside ``[0.0, 2.0]``,
                ``max_tokens`` is less than 1, or ``max_context_tokens``
                is less than 1.
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0.0, 2.0], got {temperature}."
            )
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {max_tokens}.")
        if max_context_tokens < 1:
            raise ValueError(
                f"max_context_tokens must be at least 1, got {max_context_tokens}."
            )

        config = get_config()
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_chat_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_context_tokens = max_context_tokens
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

        try:
            self._tokenizer = tiktoken.encoding_for_model(self._model)
        except KeyError:
            # Model not yet registered with tiktoken (e.g. a newly released
            # variant).  Fall back to a conservative encoder so that token
            # counting never silently under-counts.
            self._tokenizer = tiktoken.get_encoding(_FALLBACK_ENCODING)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        question: str,
        context: list[QueryResult],
    ) -> LLMResponse:
        """Embed the question, inject context, and return the model's answer.

        Trims ``context`` to the configured token budget, builds a two-message
        prompt (system + user), and calls the OpenAI Chat Completions API with
        exponential-backoff retry.

        Args:
            question: The user's natural-language question about the contracts.
                Must not be empty or consist entirely of whitespace.
            context: Retrieved contract excerpts from
                :class:`~app.retriever.LegalRetriever`, ordered by descending
                similarity score.  May be an empty list — the model will
                respond that no excerpts were available.  Higher-scored chunks
                are always preserved when the token budget requires trimming.

        Returns:
            An :class:`LLMResponse` dict containing the model's text answer
            (``content``), the model ID used (``model``), and token usage
            counts (``prompt_tokens``, ``completion_tokens``, ``total_tokens``).

        Raises:
            ValueError: If ``question`` is empty or blank.
            RuntimeError: If the API call fails after all retry attempts, or
                if the model returns an empty or refused response.
            AuthenticationError: Re-raised immediately on invalid API key.
            BadRequestError: Re-raised immediately on malformed requests.

        Example:
            >>> llm = LegalContractLLM()
            >>> response = llm.complete("Who bears liability for data breaches?", ctx)
            >>> print(response["content"])
            'According to [2], the data processor bears sole liability for...'
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty or blank.")

        trimmed_context = self._trim_context(context)
        messages = self._build_messages(question, trimmed_context)
        return self._complete_with_retry(messages)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trim_context(self, context: list[QueryResult]) -> list[QueryResult]:
        """Drop lowest-scored context chunks that exceed the token budget.

        Iterates through ``context`` in order (highest score first, as
        returned by :class:`~app.retriever.LegalRetriever`) and accumulates
        chunks until the next chunk would push the token count past
        ``_max_context_tokens``.  The first chunk is always included even if
        it alone exceeds the budget, so that the model always has at least
        one excerpt to reference.

        Token counting uses the exact ``chunk_text`` length plus
        ``_CONTEXT_HEADER_TOKENS`` estimated overhead for the
        ``[N] Document: ...`` header line.

        Args:
            context: Retrieved results ordered by descending similarity score.

        Returns:
            A (possibly shorter) list of :class:`~app.vector_store.QueryResult`
            dicts that fit within the token budget.
        """
        trimmed: list[QueryResult] = []
        accumulated = 0

        for result in context:
            chunk_tokens = (
                len(self._tokenizer.encode(result["chunk_text"]))
                + _CONTEXT_HEADER_TOKENS
            )
            if trimmed and accumulated + chunk_tokens > self._max_context_tokens:
                break
            trimmed.append(result)
            accumulated += chunk_tokens

        return trimmed

    def _format_context_block(self, index: int, result: QueryResult) -> str:
        """Format one :class:`~app.vector_store.QueryResult` as a numbered block.

        Produces a compact, scannable block that gives the model document
        provenance (name, section, title, pages) alongside the clause text.
        The similarity score is intentionally omitted — the model should
        evaluate all excerpts on their content, not their retrieval rank.

        Args:
            index: 1-based position of this result in the context list.
                Used as the citation number ``[N]``.
            result: The retrieved chunk to format.

        Returns:
            A multi-line string beginning with a ``[N] Document: ...``
            header, followed by the ``chunk_text``.
        """
        pages_str = ", ".join(str(p) for p in result["pages"])
        title_part = f" | Title: {result['section_title']}" if result["section_title"] else ""
        header = (
            f"[{index}] Document: {result['document_name']}"
            f" | Section: {result['section']}"
            f"{title_part}"
            f" | Pages: {pages_str}"
        )
        return f"{header}\n{result['chunk_text']}"

    def _build_messages(
        self,
        question: str,
        context: list[QueryResult],
    ) -> list[dict[str, str]]:
        """Assemble the two-message prompt list for the Chat Completions API.

        Produces exactly two messages:

        1. ``system`` — the legal analysis role and rules.
        2. ``user`` — numbered context blocks followed by the question.

        Args:
            question: The validated, non-blank user question.
            context: Already-trimmed list of retrieved excerpts.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts ready to be
            passed directly to ``client.chat.completions.create``.
        """
        if context:
            blocks = "\n\n".join(
                self._format_context_block(i + 1, result)
                for i, result in enumerate(context)
            )
            context_section = f"CONTRACT EXCERPTS\n-----------------\n\n{blocks}"
        else:
            context_section = (
                "CONTRACT EXCERPTS\n-----------------\n\n"
                "No contract excerpts were retrieved for this question."
            )

        user_content = f"{context_section}\n\nQUESTION: {question}"

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _complete_with_retry(
        self,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        """Call the Chat Completions API with exponential-backoff retry.

        Retries on transient errors (:class:`~openai.RateLimitError`,
        :class:`~openai.APITimeoutError`, :class:`~openai.APIConnectionError`,
        :class:`~openai.InternalServerError`).  Raises immediately on
        permanent errors (:class:`~openai.AuthenticationError`,
        :class:`~openai.BadRequestError`).

        Args:
            messages: The fully assembled prompt message list.

        Returns:
            An :class:`LLMResponse` dict with the model's answer and token
            usage metadata.

        Raises:
            RuntimeError: If all retry attempts are exhausted, or if the
                model returns a null or refused response.
            AuthenticationError: Re-raised immediately on auth failure.
            BadRequestError: Re-raised immediately on bad input.
        """
        backoff = _INITIAL_BACKOFF_SECONDS
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self._temperature,
                    max_completion_tokens=self._max_tokens,
                )

                choice = response.choices[0]
                content = choice.message.content

                if not content:
                    refusal = choice.message.refusal
                    if refusal:
                        raise RuntimeError(
                            f"Model refused to answer: {refusal}"
                        )
                    raise RuntimeError("Model returned an empty response.")

                usage = response.usage
                return LLMResponse(
                    content=content,
                    model=response.model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                )

            except (AuthenticationError, BadRequestError):
                # Configuration or input errors — retrying cannot help.
                raise

            except (
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
            ) as exc:
                last_exc = exc
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(backoff)
                    backoff *= 2

        raise RuntimeError(
            f"OpenAI Chat Completions API call failed after {_MAX_ATTEMPTS} "
            f"attempts. Last error: {last_exc}"
        ) from last_exc
