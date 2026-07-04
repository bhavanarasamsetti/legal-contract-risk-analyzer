import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv


@dataclass
class Config:
    """Central configuration for the Legal Contract Risk Analyzer application.

    All settings are sourced from environment variables. Required variables
    raise a ValueError on instantiation if missing; optional variables fall
    back to sensible defaults.
    """

    # --- OpenAI ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_chat_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Pinecone ---
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index_name: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "")
    )

    # --- Future Integrations ---
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )

    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )

    huggingface_token: str = field(
        default_factory=lambda: os.getenv("HF_TOKEN", "")
    )

    def __post_init__(self) -> None:
        """Validate that all required environment variables are present."""
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError for any missing required configuration values."""
        required: dict[str, str] = {
            "OPENAI_API_KEY": self.openai_api_key,
            "PINECONE_API_KEY": self.pinecone_api_key,
            "PINECONE_INDEX_NAME": self.pinecone_index_name,
        }

        missing = [name for name, value in required.items() if not value]

        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Please set them in your .env file or export them to the environment."
            )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the application configuration singleton.

    The ``Config`` object is constructed and validated on the first call and
    then cached for the lifetime of the process.  Subsequent calls return the
    same instance with no overhead.

    Using a cached factory instead of a module-level eager singleton means
    that importing ``app.config`` never triggers credential validation.
    Validation only fires when a module explicitly requests configuration,
    which keeps parser tests, CLI tools, and import-time side-effects clean.

    To reset the cache in tests (e.g. to inject a different environment):

        from app.config import get_config
        get_config.cache_clear()

    Returns:
        The validated :class:`Config` instance.

    Raises:
        ValueError: If any required environment variable is missing.
    """
    load_dotenv()
    return Config()
