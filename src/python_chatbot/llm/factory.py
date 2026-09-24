"""Factory for the chat model used by the assistant."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from python_chatbot.core.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the OpenAI chat model described by ``settings``.

    The function returns the provider-neutral :class:`~langchain_core.language_models.BaseChatModel`
    interface, so callers (and tests) can substitute any other chat model without
    touching the rest of the pipeline. Transient failures are retried by the OpenAI
    client itself, with exponential backoff, up to ``settings.max_retries`` times.

    Args:
        settings: Resolved application settings.

    Returns:
        A configured chat model.

    Raises:
        ConfigurationError: If the OpenAI API key is not configured.
    """
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.require_openai_api_key(),
        base_url=settings.openai_base_url,
        temperature=settings.temperature,
        max_completion_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
