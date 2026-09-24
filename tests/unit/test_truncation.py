"""Tests for the notice shown when the provider cuts an answer at the token limit."""

from python_chatbot.chat.assistant import TRUNCATION_NOTICE, PythonAssistant
from python_chatbot.chat.memory import ConversationMemory
from tests.fakes import ScriptedChatModel


def test_truncated_answer_gets_a_visible_notice() -> None:
    """``finish_reason == "length"`` must never look like a complete answer."""
    model = ScriptedChatModel(responses=["resposta cortada no me"], finish_reason="length")

    answer = PythonAssistant(model).ask("q")

    assert answer == "resposta cortada no me" + TRUNCATION_NOTICE


def test_streamed_truncated_answer_ends_with_the_notice() -> None:
    """The notice arrives as the last streamed fragment."""
    model = ScriptedChatModel(responses=["resposta cortada no me"], finish_reason="length")

    fragments = list(PythonAssistant(model).stream("q"))

    assert fragments[-1] == TRUNCATION_NOTICE
    assert "".join(fragments[:-1]) == "resposta cortada no me"


def test_complete_answers_have_no_notice() -> None:
    """A normal ``stop`` finish leaves the text untouched, streamed or not."""
    assistant = PythonAssistant(ScriptedChatModel(responses=["completa."]))

    assert assistant.ask("a") == "completa."
    assert "".join(assistant.stream("b")) == "completa."


def test_notice_is_not_stored_in_the_history() -> None:
    """The warning is for the reader; the model must not see it on the next turn."""
    model = ScriptedChatModel(responses=["parcial"], finish_reason="length")
    assistant = PythonAssistant(model, memory=ConversationMemory())

    assistant.ask("primeira")
    assistant.ask("segunda")

    assert model.calls[1][2].content == "parcial"
