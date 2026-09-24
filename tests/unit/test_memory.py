"""Tests for the bounded, thread-safe conversation memory."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from python_chatbot.chat.memory import ConversationMemory


def test_exchange_is_stored_as_human_then_ai() -> None:
    """A recorded exchange replays as a question followed by its answer."""
    memory = ConversationMemory()

    memory.append_exchange("s", "q1", "a1")

    assert memory.history("s") == [HumanMessage(content="q1"), AIMessage(content="a1")]


def test_history_returns_a_copy() -> None:
    """Callers cannot corrupt the store by mutating the returned list."""
    memory = ConversationMemory()
    memory.append_exchange("s", "q", "a")

    memory.history("s").clear()

    assert len(memory.history("s")) == 2


def test_unknown_session_has_empty_history() -> None:
    """Reading an unknown session neither fails nor creates it."""
    memory = ConversationMemory()

    assert memory.history("nobody") == []
    assert len(memory) == 0


def test_window_drops_the_oldest_exchanges() -> None:
    """Only the most recent messages survive, cut on an exchange boundary."""
    memory = ConversationMemory(max_messages=4)
    for index in range(5):
        memory.append_exchange("s", f"q{index}", f"a{index}")

    contents = [message.content for message in memory.history("s")]

    assert contents == ["q3", "a3", "q4", "a4"]


def test_odd_window_never_starts_with_an_orphan_answer() -> None:
    """With ``max_messages=5`` the window rounds down to two whole exchanges."""
    memory = ConversationMemory(max_messages=5)
    for index in range(4):
        memory.append_exchange("s", f"q{index}", f"a{index}")

    history = memory.history("s")

    assert isinstance(history[0], HumanMessage)
    assert len(history) == 4


def test_window_of_one_still_keeps_the_latest_exchange() -> None:
    """A degenerate window keeps one whole exchange rather than half of it."""
    memory = ConversationMemory(max_messages=1)
    memory.append_exchange("s", "q1", "a1")
    memory.append_exchange("s", "q2", "a2")

    assert [m.content for m in memory.history("s")] == ["q2", "a2"]


def test_zero_disables_memory() -> None:
    """``max_messages=0`` stores nothing."""
    memory = ConversationMemory(max_messages=0)

    memory.append_exchange("s", "q", "a")

    assert memory.history("s") == []


def test_negative_window_is_rejected() -> None:
    """A negative window is a programming error."""
    with pytest.raises(ValueError, match=">= 0"):
        ConversationMemory(max_messages=-1)


def test_sessions_are_isolated_and_resettable() -> None:
    """Resetting one conversation leaves the others alone."""
    memory = ConversationMemory()
    memory.append_exchange("a", "qa", "aa")
    memory.append_exchange("b", "qb", "ab")

    memory.reset("a")
    memory.reset("never-existed")

    assert memory.history("a") == []
    assert len(memory.history("b")) == 2
    assert len(memory) == 1


def test_concurrent_writers_do_not_lose_exchanges() -> None:
    """Many threads appending to one session keep every exchange."""
    memory = ConversationMemory(max_messages=200)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: memory.append_exchange("s", f"q{i}", f"a{i}"), range(100)))

    assert len(memory.history("s")) == 200
