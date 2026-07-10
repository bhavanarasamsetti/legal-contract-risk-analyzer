from langfuse import Langfuse

from app.config import get_config

config = get_config()

langfuse = Langfuse(
    public_key=config.langfuse_public_key,
    secret_key=config.langfuse_secret_key,
    host=config.langfuse_base_url,
)