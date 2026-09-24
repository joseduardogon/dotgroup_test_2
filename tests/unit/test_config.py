"""Tests for settings loading and validation."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from python_chatbot.core.config import Settings, get_settings
from python_chatbot.core.exceptions import ConfigurationError


def test_defaults_are_sensible_and_key_is_absent() -> None:
    """Without any environment the settings load and no credential is assumed."""
    settings = Settings(_env_file=None)

    assert settings.model == "gpt-4o"
    assert settings.temperature == 0.2
    assert settings.max_history_messages == 20
    assert settings.langsmith_tracing is False
    assert settings.openai_api_key is None


def test_standard_sdk_variable_names_are_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPENAI_API_KEY`` and ``LANGSMITH_*`` work unprefixed, ``PYCHAT_*`` for the rest."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "demo")
    monkeypatch.setenv("PYCHAT_MODEL", "gpt-4o-mini")

    settings = Settings(_env_file=None)

    assert settings.require_openai_api_key().get_secret_value() == "sk-from-env"
    assert settings.langsmith_tracing is True
    assert settings.langsmith_project == "demo"
    assert settings.model == "gpt-4o-mini"


def test_values_are_read_from_a_dotenv_file(tmp_path: Path) -> None:
    """A ``.env`` file supplies both prefixed and SDK-style variables."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-file\nPYCHAT_TEMPERATURE=0.7\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.require_openai_api_key().get_secret_value() == "sk-file"
    assert settings.temperature == 0.7


def test_secrets_never_leak_into_repr() -> None:
    """Printing the settings does not reveal credentials."""
    settings = Settings(
        openai_api_key=SecretStr("sk-super-secret"),
        langsmith_api_key=SecretStr("lsv2-secret"),
        _env_file=None,
    )

    rendered = repr(settings) + str(settings)

    assert "sk-super-secret" not in rendered
    assert "lsv2-secret" not in rendered


@pytest.mark.parametrize("key", [None, SecretStr(""), SecretStr("   ")])
def test_require_key_fails_with_actionable_hint(key: SecretStr | None) -> None:
    """A missing or blank key raises a configuration error that says how to fix it."""
    settings = Settings(openai_api_key=key, _env_file=None)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY") as caught:
        settings.require_openai_api_key()

    assert caught.value.hint is not None
    assert caught.value.exit_code == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 3.0},
        {"temperature": -0.1},
        {"max_tokens": 0},
        {"timeout_seconds": 0},
        {"max_retries": -1},
        {"max_history_messages": 201},
        {"model": ""},
    ],
)
def test_out_of_range_values_are_rejected(overrides: dict[str, object]) -> None:
    """Invalid numeric settings fail fast at load time."""
    with pytest.raises(ValidationError):
        Settings.model_validate(overrides)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process-wide accessor parses the environment once."""
    get_settings.cache_clear()
    monkeypatch.setenv("PYCHAT_MODEL", "first")
    first = get_settings()
    monkeypatch.setenv("PYCHAT_MODEL", "second")

    assert get_settings() is first
    assert get_settings().model == "first"
    get_settings.cache_clear()
