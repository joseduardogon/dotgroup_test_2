"""Tests for the assistant: the LCEL pipeline, memory, validation and error translation."""

import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tracers.context import collect_runs

from python_chatbot.chat.assistant import PythonAssistant, translate_provider_error
from python_chatbot.chat.memory import ConversationMemory
from python_chatbot.chat.prompts import PROMPT_VERSION
from python_chatbot.core.exceptions import (
    InvalidQuestionError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMUnavailableError,
)
from tests.fakes import ScriptedChatModel, connection_error, openai_error

QUESTION = "Como criar uma lista em Python?"


def test_ask_returns_the_model_answer(assistant: PythonAssistant) -> None:
    """The user question goes in, the model's text comes out."""
    assert assistant.ask(QUESTION) == "Use [] or list()."


def test_model_receives_system_prompt_then_the_question(
    assistant: PythonAssistant, model: ScriptedChatModel
) -> None:
    """The first prompt is exactly ``[system, human]`` with the trimmed question."""
    assistant.ask(f"  {QUESTION}  ")

    sent = model.calls[0]
    assert [type(m) for m in sent] == [SystemMessage, HumanMessage]
    assert sent[1].content == QUESTION


def test_second_turn_replays_the_first_exchange(
    assistant: PythonAssistant, model: ScriptedChatModel
) -> None:
    """Follow-up questions carry the previous turn so the model can resolve references."""
    assistant.ask(QUESTION)
    assistant.ask("E como adiciono um item?")

    sent = model.calls[1]
    assert [type(m) for m in sent] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert sent[1].content == QUESTION
    assert sent[2].content == "Use [] or list()."
    assert sent[3].content == "E como adiciono um item?"


def test_sessions_do_not_share_history(
    assistant: PythonAssistant, model: ScriptedChatModel
) -> None:
    """A different session starts from a clean slate."""
    assistant.ask(QUESTION, session_id="alice")
    assistant.ask("outra pergunta", session_id="bob")

    assert len(model.calls[1]) == 2


def test_reset_forgets_the_conversation(
    assistant: PythonAssistant, model: ScriptedChatModel
) -> None:
    """After ``reset`` the next prompt has no history."""
    assistant.ask(QUESTION)
    assistant.reset()
    assistant.ask("nova conversa")

    assert len(model.calls[1]) == 2


def test_history_window_limits_the_context_sent_to_the_model() -> None:
    """Old turns fall out of the prompt once the configured window is full."""
    model = ScriptedChatModel(responses=["a"])
    assistant = PythonAssistant(model, memory=ConversationMemory(max_messages=2))

    for index in range(3):
        assistant.ask(f"q{index}")

    contents = [message.content for message in model.calls[2][1:]]
    assert contents == ["q1", "a", "q2"]


def test_stream_yields_fragments_that_join_to_the_full_answer(
    assistant: PythonAssistant,
) -> None:
    """Streaming produces the same text, in several pieces."""
    fragments = list(assistant.stream(QUESTION))

    assert len(fragments) > 1
    assert "".join(fragments) == "Use [] or list()."


def test_streamed_answer_is_remembered_after_completion(
    assistant: PythonAssistant, model: ScriptedChatModel
) -> None:
    """A fully consumed stream becomes context for the next turn."""
    list(assistant.stream(QUESTION))
    assistant.ask("continue")

    assert model.calls[1][2].content == "Use [] or list()."


def test_failed_stream_does_not_pollute_the_history() -> None:
    """If the provider dies mid-answer, the half exchange is discarded."""
    model = ScriptedChatModel(
        responses=["uma resposta longa demais"],
        error=connection_error(),
        error_after_chunks=2,
    )
    assistant = PythonAssistant(model)

    with pytest.raises(LLMUnavailableError):
        list(assistant.stream(QUESTION))

    model.error = None
    model.error_after_chunks = None
    assistant.ask("de novo")

    assert len(model.calls[1]) == 2


def test_abandoned_stream_is_not_remembered(
    assistant: PythonAssistant, model: ScriptedChatModel
) -> None:
    """Closing the stream early (Ctrl-C in the REPL) stores nothing."""
    stream = assistant.stream(QUESTION)
    next(stream)
    stream.close()

    assistant.ask("outra")

    assert len(model.calls[1]) == 2


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_questions_are_rejected_without_calling_the_model(
    assistant: PythonAssistant, model: ScriptedChatModel, question: str
) -> None:
    """Empty input never costs an API call."""
    with pytest.raises(InvalidQuestionError, match="empty"):
        assistant.ask(question)

    assert model.calls == []


def test_overlong_questions_are_rejected_without_calling_the_model(
    model: ScriptedChatModel,
) -> None:
    """The size limit protects cost and context length."""
    assistant = PythonAssistant(model, max_question_chars=10)

    with pytest.raises(InvalidQuestionError, match="limit is 10"):
        assistant.ask("x" * 11)

    assert model.calls == []


def test_empty_model_answer_is_an_error_and_is_not_stored() -> None:
    """A blank completion (e.g. content filter) is reported, not silently remembered."""
    model = ScriptedChatModel(responses=["   ", "depois"])
    assistant = PythonAssistant(model)

    with pytest.raises(LLMRequestError, match="empty answer"):
        assistant.ask("q1")
    assistant.ask("q2")

    assert len(model.calls[1]) == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (openai_error(openai.AuthenticationError, 401), LLMAuthenticationError),
        (openai_error(openai.PermissionDeniedError, 403), LLMAuthenticationError),
        (openai_error(openai.RateLimitError, 429), LLMRateLimitError),
        (openai_error(openai.InternalServerError, 500), LLMUnavailableError),
        (connection_error(), LLMUnavailableError),
        (openai_error(openai.NotFoundError, 404), LLMRequestError),
        (openai_error(openai.BadRequestError, 400), LLMRequestError),
    ],
)
def test_provider_errors_are_translated_for_ask_and_stream(
    error: openai.OpenAIError, expected: type[LLMError]
) -> None:
    """Both call styles surface the domain error and chain the original cause."""
    assistant = PythonAssistant(ScriptedChatModel(error=error))

    with pytest.raises(expected) as from_ask:
        assistant.ask(QUESTION)
    with pytest.raises(expected):
        list(assistant.stream(QUESTION))

    assert from_ask.value.__cause__ is error


def test_translated_errors_carry_actionable_hints() -> None:
    """Users are told what to do, not just what failed."""
    for error in (
        openai_error(openai.AuthenticationError, 401),
        openai_error(openai.RateLimitError, 429),
        openai_error(openai.NotFoundError, 404),
        connection_error(),
    ):
        assert translate_provider_error(error).hint


def test_failed_call_leaves_history_untouched() -> None:
    """A provider failure on the first turn does not create phantom context."""
    model = ScriptedChatModel(error=openai_error(openai.RateLimitError, 429))
    assistant = PythonAssistant(model)

    with pytest.raises(LLMRateLimitError):
        assistant.ask("q")
    model.error = None
    assistant.ask("q2")

    assert len(model.calls[1]) == 2


def test_runs_are_tagged_for_langsmith(assistant: PythonAssistant) -> None:
    """Traces carry the run name, prompt version, model and session for filtering."""
    with collect_runs() as collected:
        assistant.ask(QUESTION, session_id="demo")

    run = collected.traced_runs[0]
    assert run.name == "python_assistant"
    assert f"prompt:{PROMPT_VERSION}" in (run.tags or [])
    assert "model:scripted" in (run.tags or [])
    assert run.metadata["session_id"] == "demo"
    assert run.metadata["history_messages"] == 0
    assert run.metadata["prompt_version"] == PROMPT_VERSION
