import time

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

# Maximum tokens accepted by the text-embedding-3-small (and -large) models.
# Inputs that exceed this limit are rejected by the API with an unhelpful
# error; checking upfront produces a clear ValueError instead.
_EMBEDDING_TOKEN_LIMIT = 8_192

# Seconds to wait before the first retry.  Each subsequent attempt doubles
# this value (exponential backoff).
_INITIAL_BACKOFF_SECONDS = 2

# Number of times to attempt an API call before propagating the exception.
_MAX_ATTEMPTS = 3


class EmbeddingGenerator:
    """Generates dense vector embeddings for text using the OpenAI API.

    This class is the sole interface between the application and the OpenAI
    embeddings endpoint.  It owns batching, token validation, and retry logic
    so that callers can pass plain strings and receive plain float vectors
    without handling any API concerns.

    The class is deliberately narrow in scope:

    - It does **not** store embeddings or interact with any vector database.
    - It does **not** know about :class:`~app.chunker.ChunkData` or any other
      domain type.  Callers pass strings and receive vectors.
    - It is **stateless** after construction: every call to :meth:`embed` or
      :meth:`embed_batch` is independent.

    Example:
        >>> gen = EmbeddingGenerator()
        >>> vector = gen.embed("Confidentiality obligations of the parties.")
        >>> len(vector)
        1536
        >>> vectors = gen.embed_batch(["clause one", "clause two"])
        >>> len(vectors)
        2
    """

    def __init__(self, batch_size: int = 512) -> None:
        """Initialise the embedding generator.

        Reads ``openai_api_key`` and ``openai_embedding_model`` from
        :func:`~app.config.get_config` and constructs the OpenAI client.

        Args:
            batch_size: Maximum number of texts to send in a single API
                request.  OpenAI accepts up to 2,048 inputs per call; the
                default of 512 is a practical ceiling that balances throughput
                against retry cost.  Must be between 1 and 2,048.

        Raises:
            ValueError: If ``batch_size`` is outside the range ``[1, 2048]``.
        """
        if not 1 <= batch_size <= 2048:
            raise ValueError(
                f"batch_size must be between 1 and 2048, got {batch_size}."
            )

        config = get_config()
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_embedding_model
        self._batch_size = batch_size
        self._tokenizer = tiktoken.encoding_for_model(self._model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed a single string and return its vector representation.

        This is a convenience wrapper around :meth:`embed_batch`.

        Args:
            text: The input string to embed.  Must not exceed
                ``_EMBEDDING_TOKEN_LIMIT`` tokens.

        Returns:
            A float vector of length 1,536 (``text-embedding-3-small``) or
            3,072 (``text-embedding-3-large``).

        Raises:
            ValueError: If ``text`` is empty or exceeds the token limit.
            RuntimeError: If the API call fails after all retry attempts.

        Example:
            >>> gen = EmbeddingGenerator()
            >>> vec = gen.embed("The parties agree to keep all information confidential.")
            >>> isinstance(vec[0], float)
            True
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty.")

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings and return their vector representations.

        Splits ``texts`` into batches of at most ``batch_size`` items, calls
        the OpenAI embeddings API once per batch, and reassembles the results
        in the original input order.

        Args:
            texts: Non-empty list of strings to embed.  Each string must not
                exceed ``_EMBEDDING_TOKEN_LIMIT`` tokens.

        Returns:
            A list of float vectors in the same order as ``texts``.

        Raises:
            ValueError: If ``texts`` is empty, any element is blank, or any
                element exceeds the token limit.
            RuntimeError: If any batch API call fails after all retry attempts.

        Example:
            >>> gen = EmbeddingGenerator()
            >>> vecs = gen.embed_batch(["clause one text", "clause two text"])
            >>> len(vecs) == 2
            True
        """
        if not texts:
            raise ValueError("texts must not be empty.")

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"texts[{i}] is empty or blank.")
            self._check_token_limit(text, index=i)

        vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            batch_vectors = self._embed_with_retry(batch)
            vectors.extend(batch_vectors)

        return vectors

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_token_limit(self, text: str, index: int = 0) -> None:
        """Raise ValueError if ``text`` exceeds the model token limit.

        Args:
            text: The string to check.
            index: The position in the original input list, used in the
                error message so callers can identify the offending item.

        Raises:
            ValueError: If the token count exceeds ``_EMBEDDING_TOKEN_LIMIT``.
        """
        token_count = len(self._tokenizer.encode(text))
        if token_count > _EMBEDDING_TOKEN_LIMIT:
            raise ValueError(
                f"texts[{index}] contains {token_count} tokens, which exceeds "
                f"the model limit of {_EMBEDDING_TOKEN_LIMIT} tokens.  "
                "Split or truncate the text before embedding."
            )

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API with exponential backoff on transient errors.

        Retries on :class:`~openai.RateLimitError`,
        :class:`~openai.APITimeoutError`,
        :class:`~openai.APIConnectionError`, and
        :class:`~openai.InternalServerError` (HTTP 500).  Raises immediately on
        :class:`~openai.AuthenticationError` and
        :class:`~openai.BadRequestError` because retrying cannot fix them.

        Args:
            texts: A single batch of strings (already validated and sized).

        Returns:
            A list of float vectors in the same order as ``texts``.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
            AuthenticationError: Re-raised immediately on auth failure.
            BadRequestError: Re-raised immediately on bad input.
        """
        backoff = _INITIAL_BACKOFF_SECONDS
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self._model,
                )
                # Sort by index before extracting vectors.  The OpenAI spec
                # does not guarantee that response.data preserves input order;
                # the index field exists precisely to handle reordering.
                return [
                    item.embedding
                    for item in sorted(response.data, key=lambda x: x.index)
                ]

            except (AuthenticationError, BadRequestError):
                # Configuration or input errors — retrying cannot help.
                raise

            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as exc:
                last_exc = exc
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(backoff)
                    backoff *= 2

        raise RuntimeError(
            f"OpenAI embeddings API call failed after {_MAX_ATTEMPTS} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc
