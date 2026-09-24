"""LangSmith observability wiring.

LangChain emits traces automatically when the LangSmith SDK finds the documented
environment variables. Because our settings may come from a ``.env`` file (which the
SDK does not read), :func:`configure_tracing` exports the resolved values before the
first model call. Nothing is exported unless tracing was explicitly enabled.
"""

import os

from python_chatbot.core.config import Settings
from python_chatbot.core.exceptions import ConfigurationError


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing according to ``settings``.

    Fails fast: asking for tracing without a LangSmith key is a configuration error,
    detected at startup rather than as a silent absence of traces.

    Args:
        settings: The resolved application settings.

    Returns:
        ``True`` when tracing was enabled, ``False`` when it is off.

    Raises:
        ConfigurationError: If tracing is on but ``LANGSMITH_API_KEY`` is missing.
    """
    if not settings.langsmith_tracing:
        return False
    if settings.langsmith_api_key is None or not settings.langsmith_api_key.get_secret_value():
        raise ConfigurationError(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set.",
            hint="Set LANGSMITH_API_KEY or disable tracing with LANGSMITH_TRACING=false.",
        )
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    return True
