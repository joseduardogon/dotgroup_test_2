"""Typed configuration following the twelve-factor "config in the environment" rule."""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from python_chatbot.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Validated runtime configuration.

    Chatbot options use the ``PYCHAT_`` prefix. Credentials and LangSmith options keep
    the names the OpenAI and LangSmith SDKs already document (``OPENAI_API_KEY``,
    ``LANGSMITH_*``), so an existing shell setup works unchanged. Values may also come
    from a local ``.env`` file. Secrets are :class:`~pydantic.SecretStr` and therefore
    never appear in ``repr`` or logs.

    Attributes:
        openai_api_key: OpenAI credential. Optional at load time so that commands such as
            ``pychat check`` can report a missing key instead of crashing on import.
        openai_base_url: Alternative API endpoint (Azure gateway, corporate proxy, local
            OpenAI-compatible server). ``None`` uses the official endpoint.
        model: OpenAI chat model name (any chat-completions model, e.g. ``gpt-4o``).
        temperature: Sampling temperature; low values favour deterministic code answers.
        max_tokens: Upper bound on generated tokens per answer.
        timeout_seconds: Per-request network timeout.
        max_retries: Automatic retries (exponential backoff) for transient failures.
        max_history_messages: Messages of context replayed to the model on each turn.
        max_question_chars: Longest accepted question, protecting cost and context size.
        langsmith_tracing: Send traces of every run to LangSmith.
        langsmith_api_key: LangSmith credential, required when tracing is on.
        langsmith_project: LangSmith project that groups the traces.
        langsmith_endpoint: Custom LangSmith endpoint (self-hosted or EU region).
    """

    model_config = SettingsConfigDict(
        env_prefix="PYCHAT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "PYCHAT_OPENAI_API_KEY")
    )
    openai_base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_BASE_URL", "PYCHAT_OPENAI_BASE_URL")
    )
    model: str = Field(default="gpt-4o", min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=16384)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_history_messages: int = Field(default=20, ge=0, le=200)
    max_question_chars: int = Field(default=4000, ge=1)

    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="dotgroup-test-2", validation_alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str | None = Field(default=None, validation_alias="LANGSMITH_ENDPOINT")

    def require_openai_api_key(self) -> SecretStr:
        """Return the OpenAI key or fail with an actionable message.

        Returns:
            The configured credential.

        Raises:
            ConfigurationError: If no key is configured.
        """
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
            raise ConfigurationError(
                "OPENAI_API_KEY is not set.",
                hint="Export it or put it in a .env file (see .env.example).",
            )
        return self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance, parsed once.

    Returns:
        The cached settings object.
    """
    return Settings()
