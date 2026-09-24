"""Conversation memory: explicit, bounded and thread-safe."""

from collections import defaultdict
from threading import Lock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationMemory:
    """In-process store of chat history, one independent list per session.

    History is deliberately explicit state owned by the application instead of hidden
    inside a chain: the exact messages replayed to the model are always inspectable,
    testable and bounded. Only *complete* exchanges are stored, so a failed or
    interrupted turn never leaves a dangling question in the context.

    The window keeps the most recent ``max_messages`` messages, always cut on an
    exchange boundary so the history never starts with an orphan answer.
    """

    def __init__(self, max_messages: int = 20) -> None:
        """Create an empty memory.

        Args:
            max_messages: Maximum messages kept per session. ``0`` disables memory.

        Raises:
            ValueError: If ``max_messages`` is negative.
        """
        if max_messages < 0:
            raise ValueError("max_messages must be >= 0")
        self._max_messages = max_messages
        self._sessions: defaultdict[str, list[BaseMessage]] = defaultdict(list)
        self._lock = Lock()

    def history(self, session_id: str) -> list[BaseMessage]:
        """Return a copy of the session's messages, oldest first.

        Args:
            session_id: Conversation identifier.

        Returns:
            The messages that should be replayed to the model.
        """
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        """Record a completed question/answer pair and enforce the window.

        Args:
            session_id: Conversation identifier.
            question: The user's message.
            answer: The assistant's full reply.
        """
        if self._max_messages == 0:
            return
        with self._lock:
            messages = self._sessions[session_id]
            messages.extend([HumanMessage(content=question), AIMessage(content=answer)])
            keep = self._max_messages - (self._max_messages % 2)
            keep = max(keep, 2)
            del messages[:-keep]

    def reset(self, session_id: str) -> None:
        """Forget one conversation.

        Args:
            session_id: Conversation identifier; unknown ids are ignored.
        """
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        """Return the number of active sessions.

        Returns:
            How many conversations currently hold history.
        """
        with self._lock:
            return len(self._sessions)
