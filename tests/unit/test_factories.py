"""Tests for the OpenAI model factory and the composition root (no network involved)."""

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from python_chatbot.chat.assistant import PythonAssistant
from python_chatbot.chat.factory import build_assistant
from python_chatbot.core.config import Settings
from python_chatbot.core.exceptions import ConfigurationError
from python_chatbot.llm.factory import build_chat_model


@pytest.fixture
def isolated_environ() -> Iterator[None]:
    """Restore ``os.environ`` after tests that trigger tracing configuration."""
    with patch.dict(os.environ):
        yield


def test_chat_model_reflects_the_settings() -> None:
    """The OpenAI client is configured from settings, key included."""
    settings = Settings(
        openai_api_key=SecretStr("sk-test"),
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=256,
        max_retries=5,
        timeout_seconds=12,
        _env_file=None,
    )

    chat_model = build_chat_model(settings)

    assert isinstance(chat_model, ChatOpenAI)
    assert chat_model.model_name == "gpt-4o-mini"
    assert chat_model.temperature == 0.0
    assert chat_model.max_retries == 5
    assert chat_model.max_tokens == 256
    assert chat_model.request_timeout == 12
    key = chat_model.openai_api_key
    assert isinstance(key, SecretStr)
    assert key.get_secret_value() == "sk-test"


def test_chat_model_requires_a_key() -> None:
    """Building the model without a credential fails before any request is made."""
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_chat_model(Settings(_env_file=None))


def test_build_assistant_composes_everything(settings: Settings) -> None:
    """The composition root returns a ready assistant."""
    assert isinstance(build_assistant(settings), PythonAssistant)


@pytest.mark.usefixtures("isolated_environ")
def test_build_assistant_enables_tracing_when_requested() -> None:
    """LangSmith variables are exported as part of assembling the assistant."""
    settings = Settings(
        openai_api_key=SecretStr("sk-test"),
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("lsv2"),
        _env_file=None,
    )

    build_assistant(settings)

    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_build_assistant_fails_on_inconsistent_tracing_config() -> None:
    """Tracing on without a LangSmith key stops startup with a clear error."""
    settings = Settings(openai_api_key=SecretStr("sk-test"), langsmith_tracing=True, _env_file=None)

    with pytest.raises(ConfigurationError, match="LANGSMITH_API_KEY"):
        build_assistant(settings)
