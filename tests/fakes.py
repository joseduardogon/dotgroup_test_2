"""Test doubles: a scripted chat model and helpers to build OpenAI SDK errors offline."""

from collections.abc import Iterator
from typing import Any

import httpx2
import openai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Chat model that replays canned answers and records every prompt it receives.

    Attributes:
        responses: Answers returned in order; the last one repeats when exhausted.
        error: Raised on every call instead of answering, when set.
        error_after_chunks: When streaming, raise ``error`` after this many chunks.
        finish_reason: Value reported by the provider metadata (``length`` = truncated).
    """

    responses: list[str] = Field(default_factory=lambda: ["ok"])
    error: BaseException | None = None
    error_after_chunks: int | None = None
    finish_reason: str = "stop"
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """Return the message lists received so far, one entry per model call."""
        return self._calls

    @property
    def _llm_type(self) -> str:
        """Identify the fake in traces."""
        return "scripted-fake"

    def _next_response(self) -> str:
        """Pick the answer for the current call."""
        index = min(len(self._calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Return the next scripted answer (or raise the configured error)."""
        self._calls.append(list(messages))
        if self.error is not None and self.error_after_chunks is None:
            raise self.error
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=self._next_response(),
                        response_metadata={"finish_reason": self.finish_reason},
                    )
                )
            ]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the next scripted answer word by word."""
        self._calls.append(list(messages))
        if self.error is not None and self.error_after_chunks is None:
            raise self.error
        words = self._next_response().split(" ")
        for position, word in enumerate(words):
            if self.error is not None and position == self.error_after_chunks:
                raise self.error
            suffix = "" if position == len(words) - 1 else " "
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + suffix))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", response_metadata={"finish_reason": self.finish_reason}
            )
        )


def openai_error(kind: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    """Build an OpenAI SDK status error without any network access.

    Args:
        kind: The exception class to instantiate.
        status: HTTP status code carried by the fake response.

    Returns:
        A ready-to-raise exception.
    """
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(status, request=request)
    return kind("simulated failure", response=response, body=None)


def connection_error() -> openai.APIConnectionError:
    """Build an OpenAI connection error without any network access.

    Returns:
        A ready-to-raise exception.
    """
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APIConnectionError(request=request)
