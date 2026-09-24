"""Tests for LangSmith tracing configuration."""

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from python_chatbot.core.config import Settings
from python_chatbot.core.exceptions import ConfigurationError
from python_chatbot.core.tracing import configure_tracing


@pytest.fixture
def isolated_environ() -> Iterator[None]:
    """Restore ``os.environ`` after a test that lets the code under test export variables."""
    with patch.dict(os.environ):
        yield


@pytest.mark.usefixtures("isolated_environ")
def test_tracing_off_exports_nothing() -> None:
    """When disabled, no LangSmith variable is touched."""
    assert configure_tracing(Settings(_env_file=None)) is False
    assert "LANGSMITH_TRACING" not in os.environ


@pytest.mark.usefixtures("isolated_environ")
def test_tracing_on_exports_the_sdk_variables() -> None:
    """Enabled tracing publishes the values the LangSmith SDK reads."""
    settings = Settings(
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("lsv2-key"),
        langsmith_project="proj",
        langsmith_endpoint="https://eu.api.smith.langchain.com",
        _env_file=None,
    )

    assert configure_tracing(settings) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2-key"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"


@pytest.mark.usefixtures("isolated_environ")
def test_endpoint_is_only_exported_when_configured() -> None:
    """The default LangSmith endpoint is left alone."""
    settings = Settings(langsmith_tracing=True, langsmith_api_key=SecretStr("k"), _env_file=None)

    configure_tracing(settings)

    assert "LANGSMITH_ENDPOINT" not in os.environ


@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_tracing_without_a_key_is_a_startup_error(key: SecretStr | None) -> None:
    """Asking for traces without credentials fails fast instead of silently not tracing."""
    settings = Settings(langsmith_tracing=True, langsmith_api_key=key, _env_file=None)

    with pytest.raises(ConfigurationError, match="LANGSMITH_API_KEY"):
        configure_tracing(settings)
