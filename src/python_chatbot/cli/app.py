"""Command line interface (Typer).

The CLI is built by :func:`create_cli`, which receives the assistant factory and the
settings loader as parameters. Production uses the real ones; tests inject fakes, so the
commands are exercised end to end without a network or an API key.
"""

from collections.abc import Callable
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from python_chatbot import __version__
from python_chatbot.chat.assistant import DEFAULT_SESSION, PythonAssistant
from python_chatbot.chat.factory import build_assistant
from python_chatbot.cli.render import render_error, render_stream
from python_chatbot.cli.repl import run_repl
from python_chatbot.core.config import Settings, get_settings
from python_chatbot.core.exceptions import ChatbotError, ConfigurationError
from python_chatbot.core.tracing import configure_tracing

AssistantFactory = Callable[[Settings], PythonAssistant]
SettingsLoader = Callable[[], Settings]


def _load_settings(loader: SettingsLoader) -> Settings:
    """Load settings, turning validation failures into a :class:`ConfigurationError`.

    Args:
        loader: Callable that builds the :class:`Settings`.

    Returns:
        The validated settings.

    Raises:
        ConfigurationError: If an environment variable has an invalid value.
    """
    try:
        return loader()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigurationError(
            f"Invalid configuration: {problems}",
            hint="Fix the variable in the environment or .env.",
        ) from exc


def create_cli(
    factory: AssistantFactory = build_assistant,
    settings_loader: SettingsLoader = get_settings,
) -> typer.Typer:
    """Build the Typer application.

    Args:
        factory: Creates the assistant from settings (defaults to the OpenAI-backed one).
        settings_loader: Returns the settings (defaults to the cached environment loader).

    Returns:
        The configured :class:`typer.Typer` instance.
    """
    app = typer.Typer(
        name="pychat",
        help="PyMentor: ask a GPT model questions about Python (LangChain + LangSmith).",
        no_args_is_help=True,
        add_completion=False,
    )
    console = Console()
    error_console = Console(stderr=True)

    def fail(error: ChatbotError) -> typer.Exit:
        """Print ``error`` and build the matching process exit.

        Args:
            error: The error that stops the command.

        Returns:
            An exit exception carrying the error's exit code.
        """
        render_error(error_console, error)
        return typer.Exit(error.exit_code)

    def show_version(requested: bool) -> None:
        """Print the version and exit when ``--version`` was passed.

        Args:
            requested: Whether the flag was present on the command line.

        Raises:
            typer.Exit: Always after printing, so no command is required.
        """
        if requested:
            console.print(f"pychat {__version__}")
            raise typer.Exit

    @app.callback()
    def root(
        version: Annotated[
            bool,
            typer.Option(
                "--version", help="Show the version and exit.", is_eager=True, callback=show_version
            ),
        ] = False,
    ) -> None:
        """PyMentor command line interface."""

    @app.command()
    def ask(
        question: Annotated[str, typer.Argument(help="Your Python question, in quotes.")],
        raw: Annotated[
            bool, typer.Option("--raw", help="Plain text output (no Markdown), for pipes.")
        ] = False,
    ) -> None:
        """Ask a single question and print the answer."""
        try:
            assistant = factory(_load_settings(settings_loader))
            render_stream(console, assistant.stream(question), raw=raw)
        except ChatbotError as error:
            raise fail(error) from error

    @app.command()
    def chat(
        session: Annotated[
            str, typer.Option("--session", help="Conversation id (keeps separate histories).")
        ] = DEFAULT_SESSION,
        raw: Annotated[
            bool, typer.Option("--raw", help="Plain text output (no Markdown).")
        ] = False,
    ) -> None:
        """Start an interactive conversation that remembers the previous turns."""
        try:
            assistant = factory(_load_settings(settings_loader))
        except ChatbotError as error:
            raise fail(error) from error
        run_repl(assistant, console, error_console, session_id=session, raw=raw)

    @app.command()
    def check() -> None:
        """Validate the configuration without calling any external service."""
        try:
            settings = _load_settings(settings_loader)
            settings.require_openai_api_key()
            tracing = configure_tracing(settings)
        except ChatbotError as error:
            raise fail(error) from error
        table = Table(show_header=False, box=None)
        table.add_row("model", settings.model)
        table.add_row("temperature", str(settings.temperature))
        table.add_row("history window", f"{settings.max_history_messages} messages")
        table.add_row("OpenAI key", "configured (hidden)")
        table.add_row(
            "LangSmith tracing", f"on, project '{settings.langsmith_project}'" if tracing else "off"
        )
        console.print(table)
        console.print("[green]Configuration looks good.[/]")

    return app


def main() -> None:
    """Console-script entry point (``pychat``)."""
    create_cli()()
