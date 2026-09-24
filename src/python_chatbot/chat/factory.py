"""Composition root: turns :class:`Settings` into a ready-to-use assistant."""

from python_chatbot.chat.assistant import PythonAssistant
from python_chatbot.chat.memory import ConversationMemory
from python_chatbot.core.config import Settings
from python_chatbot.core.tracing import configure_tracing
from python_chatbot.llm.factory import build_chat_model


def build_assistant(settings: Settings) -> PythonAssistant:
    """Wire tracing, the OpenAI chat model and memory into a :class:`PythonAssistant`.

    Args:
        settings: Resolved application settings.

    Returns:
        A fully configured assistant.

    Raises:
        ConfigurationError: If the OpenAI key is missing, or LangSmith tracing is
            enabled without a LangSmith key.
    """
    configure_tracing(settings)
    return PythonAssistant(
        build_chat_model(settings),
        model_name=settings.model,
        memory=ConversationMemory(settings.max_history_messages),
        max_question_chars=settings.max_question_chars,
    )
