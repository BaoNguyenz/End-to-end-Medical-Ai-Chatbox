"""Client for the local OpenAI-compatible inference server."""

from openai import OpenAI

from src.config import settings


def create_llm_client() -> OpenAI:
    """Never fall back to the paid OpenAI endpoint in the application."""
    if not settings.llm_base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("LLM_BASE_URL must point to a local server")
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=180.0,
        max_retries=1,
    )
