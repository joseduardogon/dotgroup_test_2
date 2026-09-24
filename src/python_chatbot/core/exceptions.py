"""Exception hierarchy of the chatbot.

Provider specific failures (``openai.RateLimitError`` and friends) are translated at
the boundary of the assistant into these types, so the CLI and any future front-end
depend on the domain vocabulary instead of on the OpenAI SDK.
"""


class ChatbotError(Exception):
    """Base class for every error deliberately raised by the application.

    Attributes:
        exit_code: Process exit status the CLI uses when the error is fatal.
        hint: Optional, actionable advice shown to the user next to the message.
    """

    exit_code: int = 1
    hint: str | None = None

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        """Create the error.

        Args:
            message: Human readable description of what went wrong.
            hint: What the user can do about it, when there is something to do.
        """
        super().__init__(message)
        if hint is not None:
            self.hint = hint


class ConfigurationError(ChatbotError):
    """Raised when required settings are missing or inconsistent."""

    exit_code = 2


class InvalidQuestionError(ChatbotError):
    """Raised when the user's input cannot be sent to the model (empty or too long)."""

    exit_code = 64


class LLMError(ChatbotError):
    """Base class for failures that originate in the language model provider."""


class LLMAuthenticationError(LLMError):
    """Raised when the provider rejects the API key."""

    exit_code = 2


class LLMRateLimitError(LLMError):
    """Raised when the provider throttles the request or the quota is exhausted."""


class LLMUnavailableError(LLMError):
    """Raised on network failures, timeouts and provider side (5xx) errors."""


class LLMRequestError(LLMError):
    """Raised when the provider refuses the request itself (for example, context too long)."""
