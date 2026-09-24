"""End-to-end tests of the command line interface with an injected fake model."""

import io
from collections.abc import Iterator
from typing import cast

import openai
import pytest
import typer
from pydantic import SecretStr
from typer.testing import CliRunner

from python_chatbot import __version__
from python_chatbot.chat.assistant import PythonAssistant
from python_chatbot.chat.memory import ConversationMemory
from python_chatbot.cli.app import create_cli, use_utf8_output
from python_chatbot.core.config import Settings
from tests.fakes import ScriptedChatModel, openai_error

runner = CliRunner()
NNBSP = chr(0x202F)


def _cli(model: ScriptedChatModel, settings: Settings | None = None) -> typer.Typer:
    """Build a CLI whose assistant uses ``model``."""
    return create_cli(
        factory=lambda _settings: PythonAssistant(model, memory=ConversationMemory()),
        settings_loader=lambda: (
            settings or Settings(openai_api_key=SecretStr("sk"), _env_file=None)
        ),
    )


def test_ask_prints_the_answer() -> None:
    """A one-shot question prints the model's answer."""
    cli = _cli(ScriptedChatModel(responses=["Use colchetes."]))

    result = runner.invoke(cli, ["ask", "lista?"])

    assert result.exit_code == 0
    assert "Use colchetes." in result.output


def test_ask_raw_prints_plain_text() -> None:
    """``--raw`` emits the answer without Markdown rendering."""
    cli = _cli(ScriptedChatModel(responses=["**negrito** e `code`"]))

    result = runner.invoke(cli, ["ask", "--raw", "q"])

    assert result.exit_code == 0
    assert "**negrito** e `code`" in result.output


def test_ask_rejects_blank_questions_with_a_usage_exit_code() -> None:
    """An empty question is reported and exits with the invalid-input code."""
    result = runner.invoke(_cli(ScriptedChatModel()), ["ask", "   "])

    assert result.exit_code == 64
    assert "empty" in result.output


def test_ask_reports_provider_failures_with_hint() -> None:
    """A rate limit becomes a friendly message and a non-zero exit code."""
    model = ScriptedChatModel(error=openai_error(openai.RateLimitError, 429))

    result = runner.invoke(_cli(model), ["ask", "q"])

    assert result.exit_code == 1
    assert "rate limit" in result.output.lower()
    assert "Hint" in result.output


def test_missing_key_stops_before_any_model_call() -> None:
    """The real factory refuses to start without a key and exits with the config code."""
    cli = create_cli(settings_loader=lambda: Settings(_env_file=None))

    result = runner.invoke(cli, ["ask", "q"])

    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output


def test_invalid_environment_values_are_reported_as_configuration_errors() -> None:
    """A malformed setting (validation failure) does not produce a traceback."""

    def bad_loader() -> Settings:
        return Settings.model_validate({"temperature": 9})

    result = runner.invoke(create_cli(settings_loader=bad_loader), ["ask", "q"])

    assert result.exit_code == 2
    assert "Invalid configuration" in result.output
    assert "temperature" in result.output


def test_chat_session_remembers_and_supports_commands() -> None:
    """The REPL streams answers, shows help, resets memory and exits cleanly."""
    model = ScriptedChatModel(responses=["primeira", "segunda", "terceira"])
    script = "\n".join(["oi", "/help", "mais", "", "/reset", "de novo", "/exit"]) + "\n"

    result = runner.invoke(_cli(model), ["chat"], input=script)

    assert result.exit_code == 0
    assert "primeira" in result.output
    assert "Commands:" in result.output
    assert "Conversation cleared." in result.output
    assert len(model.calls) == 3
    assert len(model.calls[1]) == 4
    assert len(model.calls[2]) == 2


def test_chat_survives_a_failed_turn() -> None:
    """One provider error does not end the session; the next question works."""
    model = ScriptedChatModel(responses=["ok"], error=openai_error(openai.RateLimitError, 429))
    cli = _cli(model)

    result = runner.invoke(cli, ["chat"], input="q1\n/exit\n")
    model.error = None
    second = runner.invoke(cli, ["chat"], input="q2\n/exit\n")

    assert result.exit_code == 0
    assert "rate limit" in result.output.lower()
    assert "ok" in second.output


def test_chat_ends_on_end_of_input() -> None:
    """Ctrl-D (EOF) leaves the loop without an error."""
    result = runner.invoke(_cli(ScriptedChatModel()), ["chat"], input="")

    assert result.exit_code == 0


def test_chat_fails_fast_without_configuration() -> None:
    """Starting the REPL without a key is an immediate configuration error."""
    result = runner.invoke(create_cli(settings_loader=lambda: Settings(_env_file=None)), ["chat"])

    assert result.exit_code == 2


def test_check_reports_configuration_without_leaking_secrets() -> None:
    """``check`` summarises the setup and never prints the key."""
    settings = Settings(openai_api_key=SecretStr("sk-very-secret"), _env_file=None)

    result = runner.invoke(_cli(ScriptedChatModel(), settings), ["check"])

    assert result.exit_code == 0
    assert "gpt-4o" in result.output
    assert "sk-very-secret" not in result.output
    assert "Configuration looks good" in result.output


def test_check_fails_when_the_key_is_missing() -> None:
    """A missing key is the first thing ``check`` reports."""
    result = runner.invoke(_cli(ScriptedChatModel(), Settings(_env_file=None)), ["check"])

    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output


def test_check_fails_on_inconsistent_tracing() -> None:
    """Tracing without a LangSmith key is caught by ``check`` too."""
    settings = Settings(openai_api_key=SecretStr("sk"), langsmith_tracing=True, _env_file=None)

    result = runner.invoke(_cli(ScriptedChatModel(), settings), ["check"])

    assert result.exit_code == 2
    assert "LANGSMITH_API_KEY" in result.output


def test_version_flag() -> None:
    """``--version`` prints the package version."""
    result = runner.invoke(_cli(ScriptedChatModel()), ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_arguments_shows_help() -> None:
    """Running the tool bare lists the available commands."""
    result = runner.invoke(_cli(ScriptedChatModel()), [])

    assert "ask" in result.output
    assert "chat" in result.output
    assert "check" in result.output


@pytest.mark.parametrize("command", ["ask", "chat", "check"])
def test_every_command_has_help(command: str) -> None:
    """Each command documents itself."""
    result = runner.invoke(_cli(ScriptedChatModel()), [command, "--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output


class _InterruptedOnce:
    """Assistant double whose first streamed answer is cancelled with Ctrl-C."""

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.questions: list[str] = []
        self.resets: list[str] = []

    def stream(self, question: str, session_id: str) -> Iterator[str]:
        """Interrupt the first answer, then answer normally."""
        self.questions.append(question)
        self.resets.append(f"stream:{session_id}")
        if len(self.questions) == 1:
            yield "parcial"
            raise KeyboardInterrupt
        yield "resposta completa"

    def reset(self, session_id: str) -> None:
        """Record a reset request."""
        self.resets.append(session_id)


def test_ctrl_c_cancels_only_the_current_answer() -> None:
    """Interrupting a streaming answer returns to the prompt instead of killing the chat."""
    double = _InterruptedOnce()
    cli = create_cli(
        factory=lambda _settings: cast("PythonAssistant", double),
        settings_loader=lambda: Settings(openai_api_key=SecretStr("sk"), _env_file=None),
    )

    result = runner.invoke(cli, ["chat"], input="primeira\nsegunda\n/exit\n")

    assert result.exit_code == 0
    assert "Answer cancelled." in result.output
    assert "resposta completa" in result.output
    assert double.questions == ["primeira", "segunda"]


def test_utf8_output_survives_characters_outside_the_ansi_code_page() -> None:
    """A cp1252 stream crashes on a narrow no-break space; after the fix it writes UTF-8."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")

    with pytest.raises(UnicodeEncodeError):
        stream.write("a" + NNBSP + "b")
    use_utf8_output([stream])
    stream.write("a" + NNBSP + "b ção")
    stream.flush()

    assert ("a" + NNBSP + "b ção").encode() in raw.getvalue()


def test_utf8_output_ignores_streams_that_cannot_be_reconfigured() -> None:
    """Objects without ``reconfigure`` (test capture buffers, custom streams) are skipped."""
    use_utf8_output([io.StringIO()])
