"""Tests for app/embeddings.py.

Mocking strategy
----------------
Three external dependencies are intercepted:

1. ``app.embeddings.get_config`` — replaced by a lambda that returns a
   ``MagicMock`` with ``openai_api_key`` and ``openai_embedding_model`` set.
   This prevents the real credential validation from running during tests.

2. ``app.embeddings.OpenAI`` — replaced by a ``MagicMock`` whose ``return_value``
   is the shared ``mock_openai_client`` fixture.  Both the ``gen`` fixture and
   every test function that requests ``mock_openai_client`` receive the same
   object, so tests can configure ``mock_openai_client.embeddings.create``
   without inspecting ``gen._client`` directly.

3. ``app.embeddings.time.sleep`` — replaced by a no-op lambda so retry tests
   complete in milliseconds.

``tiktoken`` is NOT mocked.  It performs a pure disk read (no network calls)
and using the real tokeniser makes the token-limit tests meaningful.  For tests
that need to trigger the limit without building an 8 192-token string,
``app.embeddings._EMBEDDING_TOKEN_LIMIT`` is temporarily patched to 3 via
``monkeypatch.setattr``.

OpenAI API exceptions are constructed with a minimal ``MagicMock`` httpx
response so no real HTTP connections are required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import httpx
import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

# Make sure the project root is on sys.path when the test file is executed
# directly or via pytest from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings import EmbeddingGenerator, _EMBEDDING_TOKEN_LIMIT, _MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_api_response(status_code: int) -> MagicMock:
    """Return a minimal mock ``httpx.Response`` for OpenAI exception construction.

    The OpenAI SDK exception base classes require a ``response`` argument that
    satisfies ``isinstance(x, httpx.Response)``.  Using ``spec=httpx.Response``
    satisfies that check without making any real HTTP calls.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers()
    resp.request = MagicMock(spec=httpx.Request)
    return resp


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    """Build a mock ``CreateEmbeddingResponse`` from a list of float vectors.

    Items are intentionally stored in **reverse index order** inside
    ``response.data`` so that the sort-by-index logic in
    ``_embed_with_retry`` is exercised, not merely trusted.
    """
    response = MagicMock()
    response.data = [
        MagicMock(embedding=vec, index=i)
        for i, vec in reversed(list(enumerate(vectors)))
    ]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """A MagicMock that stands in for the ``openai.OpenAI`` client instance."""
    return MagicMock()


@pytest.fixture()
def gen(mock_openai_client: MagicMock, monkeypatch: pytest.MonkeyPatch) -> EmbeddingGenerator:
    """Return a fully wired ``EmbeddingGenerator`` backed by a mock OpenAI client.

    The fixture patches:

    - ``app.embeddings.get_config`` → fake Config (no credentials required).
    - ``app.embeddings.OpenAI``     → returns ``mock_openai_client``.
    - ``app.embeddings.time.sleep`` → no-op (retry tests run instantly).
    """
    fake_config = MagicMock()
    fake_config.openai_api_key = "test-key"
    fake_config.openai_embedding_model = "text-embedding-3-small"

    monkeypatch.setattr("app.embeddings.get_config", lambda: fake_config)
    monkeypatch.setattr(
        "app.embeddings.OpenAI",
        MagicMock(return_value=mock_openai_client),
    )
    monkeypatch.setattr("app.embeddings.time.sleep", lambda _: None)

    return EmbeddingGenerator(batch_size=512)


# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    """Validate batch_size bounds checking in __init__.

    These tests require NO mocking because the ValueError guard fires before
    get_config() or OpenAI() are ever reached.
    """

    def test_batch_size_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingGenerator(batch_size=0)

    def test_batch_size_above_maximum_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingGenerator(batch_size=2049)


# ---------------------------------------------------------------------------
# 2. embed()
# ---------------------------------------------------------------------------

class TestEmbed:
    """Tests for the single-string convenience wrapper."""

    def test_returns_exactly_one_vector(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        mock_openai_client.embeddings.create.return_value = _make_embedding_response(
            [[0.1, 0.2, 0.3]]
        )
        result = gen.embed("Confidentiality obligations of the parties.")
        assert isinstance(result, list)
        assert len(result) == 3  # vector dimension

    def test_returned_vector_matches_mocked_response(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        expected = [0.11, 0.22, 0.33, 0.44]
        mock_openai_client.embeddings.create.return_value = _make_embedding_response(
            [expected]
        )
        result = gen.embed("The parties agree to maintain confidentiality.")
        assert result == expected

    def test_blank_string_raises_value_error(self, gen: EmbeddingGenerator) -> None:
        with pytest.raises(ValueError, match="empty"):
            gen.embed("   ")

    def test_empty_string_raises_value_error(self, gen: EmbeddingGenerator) -> None:
        with pytest.raises(ValueError, match="empty"):
            gen.embed("")


# ---------------------------------------------------------------------------
# 3. embed_batch()
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    """Tests for multi-string embedding and batching behaviour."""

    def test_vectors_returned_in_input_order(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        """Verify that sort-by-index correctly reorders a reversed response."""
        vectors = [[float(i)] * 4 for i in range(3)]
        # _make_embedding_response stores data in reverse index order
        mock_openai_client.embeddings.create.return_value = _make_embedding_response(
            vectors
        )
        result = gen.embed_batch(["text one", "text two", "text three"])
        assert result == vectors

    def test_batching_splits_across_multiple_api_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_openai_client: MagicMock,
    ) -> None:
        """With batch_size=2 and 5 inputs, exactly 3 API calls must be made."""
        fake_config = MagicMock()
        fake_config.openai_api_key = "test-key"
        fake_config.openai_embedding_model = "text-embedding-3-small"
        monkeypatch.setattr("app.embeddings.get_config", lambda: fake_config)
        monkeypatch.setattr(
            "app.embeddings.OpenAI",
            MagicMock(return_value=mock_openai_client),
        )
        monkeypatch.setattr("app.embeddings.time.sleep", lambda _: None)

        small_gen = EmbeddingGenerator(batch_size=2)

        # Prepare one response per batch (2 + 2 + 1 items)
        all_vectors = [[float(i)] * 3 for i in range(5)]
        mock_openai_client.embeddings.create.side_effect = [
            _make_embedding_response(all_vectors[0:2]),
            _make_embedding_response(all_vectors[2:4]),
            _make_embedding_response(all_vectors[4:5]),
        ]

        texts = [f"clause {i}" for i in range(5)]
        result = small_gen.embed_batch(texts)

        assert result == all_vectors
        assert mock_openai_client.embeddings.create.call_count == 3

    def test_empty_list_raises_value_error(self, gen: EmbeddingGenerator) -> None:
        with pytest.raises(ValueError, match="empty"):
            gen.embed_batch([])

    def test_blank_element_raises_value_error(self, gen: EmbeddingGenerator) -> None:
        with pytest.raises(ValueError, match=r"texts\[1\]"):
            gen.embed_batch(["valid text", "   ", "also valid"])


# ---------------------------------------------------------------------------
# 4. Token limit
# ---------------------------------------------------------------------------

class TestTokenLimit:
    """Verify that _check_token_limit enforces the per-input token ceiling."""

    def test_short_text_does_not_raise(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        mock_openai_client.embeddings.create.return_value = _make_embedding_response(
            [[0.5, 0.5]]
        )
        # Should complete without raising — no assertion on the vector needed
        gen.embed("Short legal clause.")

    def test_oversized_text_raises_value_error(
        self, gen: EmbeddingGenerator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Patch the module-level limit to 3 tokens so a normal sentence trips it
        monkeypatch.setattr("app.embeddings._EMBEDDING_TOKEN_LIMIT", 3)
        # "four token text here" tokenises to more than 3 tokens
        with pytest.raises(ValueError, match="tokens"):
            gen.embed("four token text here")

    def test_error_message_includes_index(
        self, gen: EmbeddingGenerator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.embeddings._EMBEDDING_TOKEN_LIMIT", 3)
        with pytest.raises(ValueError, match=r"texts\[2\]"):
            gen.embed_batch(["ok", "also ok", "this one is definitely too long"])


# ---------------------------------------------------------------------------
# 5. Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Verify that _embed_with_retry handles transient and permanent errors."""

    def test_rate_limit_error_retries_and_succeeds(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        """A single RateLimitError on attempt 1 must be retried; attempt 2 succeeds."""
        expected = [[0.9, 0.8, 0.7]]
        mock_openai_client.embeddings.create.side_effect = [
            RateLimitError("rate limited", response=_make_api_response(429), body={}),
            _make_embedding_response(expected),
        ]
        result = gen.embed_batch(["penalty clause"])
        assert result == expected
        assert mock_openai_client.embeddings.create.call_count == 2

    def test_internal_server_error_retries_and_succeeds(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        """A single InternalServerError (HTTP 500) must be retried; attempt 2 succeeds."""
        expected = [[0.1, 0.2]]
        mock_openai_client.embeddings.create.side_effect = [
            InternalServerError(
                "server error", response=_make_api_response(500), body={}
            ),
            _make_embedding_response(expected),
        ]
        result = gen.embed_batch(["indemnification clause"])
        assert result == expected
        assert mock_openai_client.embeddings.create.call_count == 2

    def test_all_attempts_exhausted_raises_runtime_error(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        """Failing on every attempt must raise RuntimeError, not the original error."""
        mock_openai_client.embeddings.create.side_effect = [
            RateLimitError("rate limited", response=_make_api_response(429), body={})
        ] * _MAX_ATTEMPTS
        with pytest.raises(RuntimeError, match="failed after"):
            gen.embed_batch(["limitation of liability"])
        assert mock_openai_client.embeddings.create.call_count == _MAX_ATTEMPTS

    def test_authentication_error_raises_immediately(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        """AuthenticationError must propagate on the first attempt with no retry."""
        mock_openai_client.embeddings.create.side_effect = AuthenticationError(
            "invalid key", response=_make_api_response(401), body={}
        )
        with pytest.raises(AuthenticationError):
            gen.embed_batch(["governing law clause"])
        assert mock_openai_client.embeddings.create.call_count == 1

    def test_bad_request_error_raises_immediately(
        self, gen: EmbeddingGenerator, mock_openai_client: MagicMock
    ) -> None:
        """BadRequestError must propagate on the first attempt with no retry."""
        mock_openai_client.embeddings.create.side_effect = BadRequestError(
            "bad input", response=_make_api_response(400), body={}
        )
        with pytest.raises(BadRequestError):
            gen.embed_batch(["termination clause"])
        assert mock_openai_client.embeddings.create.call_count == 1

    def test_retry_sleep_is_called_between_attempts(
        self, monkeypatch: pytest.MonkeyPatch, mock_openai_client: MagicMock
    ) -> None:
        """sleep() must be called once between two attempts (not after the last)."""
        sleep_mock = MagicMock()
        fake_config = MagicMock()
        fake_config.openai_api_key = "test-key"
        fake_config.openai_embedding_model = "text-embedding-3-small"
        monkeypatch.setattr("app.embeddings.get_config", lambda: fake_config)
        monkeypatch.setattr(
            "app.embeddings.OpenAI",
            MagicMock(return_value=mock_openai_client),
        )
        monkeypatch.setattr("app.embeddings.time.sleep", sleep_mock)

        timed_gen = EmbeddingGenerator(batch_size=512)
        expected = [[0.3, 0.4]]
        mock_openai_client.embeddings.create.side_effect = [
            RateLimitError("rate limited", response=_make_api_response(429), body={}),
            _make_embedding_response(expected),
        ]
        timed_gen.embed_batch(["force majeure clause"])

        # sleep called exactly once (between attempt 1 and attempt 2)
        sleep_mock.assert_called_once()
        # first arg is the initial backoff duration (2 seconds)
        from app.embeddings import _INITIAL_BACKOFF_SECONDS
        assert sleep_mock.call_args == call(_INITIAL_BACKOFF_SECONDS)
