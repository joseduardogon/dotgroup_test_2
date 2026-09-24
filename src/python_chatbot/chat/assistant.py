"""The assistant: an LCEL pipeline plus validation, memory and error translation."""

from collections.abc import Generator

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from python_chatbot.chat.memory import ConversationMemory
from python_chatbot.chat.prompts import HISTORY_KEY, PROMPT_VERSION, QUESTION_KEY, build_prompt
from python_chatbot.core.exceptions import (
    ChatbotError,
    InvalidQuestionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMRequestError,
    LLMUnavailableError,
)

DEFAULT_SESSION = "default"
RUN_NAME = "python_assistant"
TRUNCATION_NOTICE = "\n\n[Answer truncated by the token limit. Raise PYCHAT_MAX_TOKENS.]"
_LENGTH_FINISH_REASON = "length"


def _was_truncated(message: BaseMessage) -> bool:
    """Tell whether the provider stopped generating because it hit the token limit.

    Args:
        message: A complete reply or the last streamed chunk.

    Returns:
        ``True`` when the finish reason is ``length``.
    """
    return message.response_metadata.get("finish_reason") == _LENGTH_FINISH_REASON


def translate_provider_error(error: openai.OpenAIError) -> ChatbotError:
    """Map an OpenAI SDK exception to the application's error vocabulary.

    Args:
        error: Exception raised by the OpenAI client.

    Returns:
        The equivalent :class:`ChatbotError`; the caller chains the original with ``from``.
    """
    if isinstance(error, openai.AuthenticationError | openai.PermissionDeniedError):
        return LLMAuthenticationError(
            "OpenAI rejected the API key.",
            hint="Check OPENAI_API_KEY and that the key can use the configured model.",
        )
    if isinstance(error, openai.RateLimitError):
        return LLMRateLimitError(
            "OpenAI rate limit or quota exceeded.",
            hint="Wait a moment and retry, or check the billing/quota of your account.",
        )
    if isinstance(error, openai.APIConnectionError | openai.InternalServerError):
        return LLMUnavailableError(
            "Could not reach the OpenAI API.",
            hint="Check your connection and OPENAI_BASE_URL, then try again shortly.",
        )
    if isinstance(error, openai.NotFoundError):
        return LLMRequestError(
            "The configured model was not found for this account.",
            hint="Set PYCHAT_MODEL to a model you have access to.",
        )
    return LLMRequestError(f"OpenAI rejected the request: {error}")


class PythonAssistant:
    """Answers Python questions through ``prompt | chat model``.

    The pipeline is written in LangChain Expression Language (LCEL), which gives
    streaming, async, batching and LangSmith tracing without extra code. Each run is
    tagged with the prompt version, model and session so traces can be filtered and
    compared in LangSmith.
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        model_name: str = "unknown",
        memory: ConversationMemory | None = None,
        max_question_chars: int = 4000,
    ) -> None:
        """Assemble the assistant.

        Args:
            model: Any LangChain chat model (OpenAI in production, a fake in tests).
            model_name: Label recorded in traces; informational only.
            memory: Conversation store. A private one is created when omitted.
            max_question_chars: Longest accepted question.
        """
        self._chain = build_prompt() | model
        self._model_name = model_name
        self._memory = memory if memory is not None else ConversationMemory()
        self._max_question_chars = max_question_chars

    def ask(self, question: str, session_id: str = DEFAULT_SESSION) -> str:
        """Answer ``question`` in a single, non-streamed call.

        Args:
            question: The user's text.
            session_id: Conversation the question belongs to.

        Returns:
            The assistant's complete answer. When the provider cut it at the token limit,
            :data:`TRUNCATION_NOTICE` is appended (it is not stored in the history).

        Raises:
            InvalidQuestionError: If the question is empty or too long.
            LLMError: If the provider fails (see :func:`translate_provider_error`).
        """
        cleaned, payload, config = self._prepare(question, session_id)
        try:
            message = self._chain.invoke(payload, config)
        except openai.OpenAIError as exc:
            raise translate_provider_error(exc) from exc
        self._commit(session_id, cleaned, message.text)
        return message.text + TRUNCATION_NOTICE if _was_truncated(message) else message.text

    def stream(
        self, question: str, session_id: str = DEFAULT_SESSION
    ) -> Generator[str, None, None]:
        """Answer ``question`` token by token.

        The exchange is stored in memory only after the stream finishes successfully,
        so an interrupted or failed answer never pollutes the history.

        Args:
            question: The user's text.
            session_id: Conversation the question belongs to.

        Yields:
            Successive fragments of the answer, followed by :data:`TRUNCATION_NOTICE` when
            the provider cut the answer at the token limit.

        Raises:
            InvalidQuestionError: If the question is empty or too long.
            LLMError: If the provider fails (see :func:`translate_provider_error`).
        """
        cleaned, payload, config = self._prepare(question, session_id)
        fragments: list[str] = []
        truncated = False
        try:
            for chunk in self._chain.stream(payload, config):
                truncated = truncated or _was_truncated(chunk)
                if chunk.text:
                    fragments.append(chunk.text)
                    yield chunk.text
        except openai.OpenAIError as exc:
            raise translate_provider_error(exc) from exc
        self._commit(session_id, cleaned, "".join(fragments))
        if truncated:
            yield TRUNCATION_NOTICE

    def reset(self, session_id: str = DEFAULT_SESSION) -> None:
        """Forget the conversation history of ``session_id``.

        Args:
            session_id: Conversation to clear.
        """
        self._memory.reset(session_id)

    def _prepare(
        self, question: str, session_id: str
    ) -> tuple[str, dict[str, object], RunnableConfig]:
        """Validate the input and build the chain payload and trace configuration.

        Args:
            question: Raw user text.
            session_id: Conversation identifier.

        Returns:
            The cleaned question, the chain input and the run configuration.

        Raises:
            InvalidQuestionError: If the question is empty or exceeds the limit.
        """
        cleaned = question.strip()
        if not cleaned:
            raise InvalidQuestionError("The question is empty.", hint="Type a Python question.")
        if len(cleaned) > self._max_question_chars:
            raise InvalidQuestionError(
                f"The question has {len(cleaned)} characters; the limit is "
                f"{self._max_question_chars}.",
                hint="Shorten it or split it into smaller questions.",
            )
        history = self._memory.history(session_id)
        payload: dict[str, object] = {HISTORY_KEY: history, QUESTION_KEY: cleaned}
        config: RunnableConfig = {
            "run_name": RUN_NAME,
            "tags": ["python-assistant", f"prompt:{PROMPT_VERSION}", f"model:{self._model_name}"],
            "metadata": {
                "session_id": session_id,
                "history_messages": len(history),
                "prompt_version": PROMPT_VERSION,
            },
        }
        return cleaned, payload, config

    def _commit(self, session_id: str, question: str, answer: str) -> None:
        """Store a finished exchange, rejecting empty completions.

        Args:
            session_id: Conversation identifier.
            question: The cleaned user question.
            answer: The model's full answer.

        Raises:
            LLMRequestError: If the model produced no text (e.g. a content filter).
        """
        if not answer.strip():
            raise LLMRequestError(
                "The model returned an empty answer.", hint="Rephrase the question and retry."
            )
        self._memory.append_exchange(session_id, question, answer)
