"""Interactive read-eval-print loop."""

from rich.console import Console
from rich.markup import escape

from python_chatbot.chat.assistant import PythonAssistant
from python_chatbot.cli.render import render_error, render_stream
from python_chatbot.core.exceptions import ChatbotError

HELP_TEXT = """\
Commands:
  /help    show this help
  /reset   forget the conversation so far
  /exit    leave (also /quit, Ctrl-D or Ctrl-C)
Anything else is sent to the assistant as a Python question."""

EXIT_COMMANDS = frozenset({"/exit", "/quit"})


def run_repl(
    assistant: PythonAssistant,
    console: Console,
    error_console: Console,
    *,
    session_id: str,
    raw: bool = False,
) -> None:
    """Run the chat loop until the user leaves.

    Recoverable problems (empty input, rate limits, network errors) are reported and the
    loop continues, so one bad turn never ends the session. Ctrl-C while an answer is
    streaming cancels only that answer (nothing is remembered); at the prompt it leaves.

    Args:
        assistant: The assistant answering the questions.
        console: Console for the conversation.
        error_console: Console for error messages.
        session_id: Conversation identifier passed to the assistant.
        raw: Print answers as plain text instead of rendered Markdown.
    """
    console.print("[bold]PyMentor[/] - ask me anything about Python. Type /help for commands.")
    while True:
        try:
            line = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue
        command = line.lower()
        if command in EXIT_COMMANDS:
            return
        if command == "/help":
            console.print(escape(HELP_TEXT))
            continue
        if command == "/reset":
            assistant.reset(session_id)
            console.print("[dim]Conversation cleared.[/]")
            continue
        console.print("[bold green]pymentor>[/]")
        try:
            render_stream(console, assistant.stream(line, session_id), raw=raw)
        except ChatbotError as error:
            render_error(error_console, error)
        except KeyboardInterrupt:
            console.print("\n[dim]Answer cancelled.[/]")
