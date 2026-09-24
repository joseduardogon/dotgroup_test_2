"""Shared fixtures."""

import pytest
from pydantic import SecretStr

from python_chatbot.chat.assistant import PythonAssistant
from python_chatbot.chat.memory import ConversationMemory
from python_chatbot.core.config import Settings
from tests.fakes import ScriptedChatModel

ENV_VARS = (
    "OPENAI_API_KEY",
    "PYCHAT_OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "PYCHAT_OPENAI_BASE_URL",
    "PYCHAT_MODEL",
    "PYCHAT_TEMPERATURE",
    "PYCHAT_MAX_HISTORY_MESSAGES",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every variable the application reads so tests never see the host's."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    """Return settings with a dummy key and no dependency on a ``.env`` file."""
    return Settings(openai_api_key=SecretStr("sk-test"), _env_file=None)


@pytest.fixture
def model() -> ScriptedChatModel:
    """Return a scripted chat model answering ``"Use [] or list()."``."""
    return ScriptedChatModel(responses=["Use [] or list()."])


@pytest.fixture
def assistant(model: ScriptedChatModel) -> PythonAssistant:
    """Return an assistant wired to the scripted model with a fresh memory."""
    return PythonAssistant(model, model_name="scripted", memory=ConversationMemory(20))
