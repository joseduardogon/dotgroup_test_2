"""Console rendering helpers, kept separate so they can be tested with an in-memory console."""

from collections.abc import Iterable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape

from python_chatbot.core.exceptions import ChatbotError


def render_stream(console: Console, fragments: Iterable[str], *, raw: bool = False) -> str:
    """Print an answer as it is generated and return the complete text.

    In rendered mode the growing text is displayed as Markdown (code blocks are syntax
    highlighted) and refreshed in place. In raw mode the fragments are written verbatim,
    which is what scripts and pipes want.

    Args:
        console: Destination console.
        fragments: Successive pieces of the answer, e.g. from ``PythonAssistant.stream``.
        raw: Write plain text instead of rendering Markdown.

    Returns:
        The full answer.
    """
    parts: list[str] = []
    if raw:
        for fragment in fragments:
            parts.append(fragment)
            console.out(fragment, end="", highlight=False)
        console.out("")
        return "".join(parts)

    with Live(
        Markdown(""), console=console, refresh_per_second=12, vertical_overflow="visible"
    ) as live:
        for fragment in fragments:
            parts.append(fragment)
            live.update(Markdown("".join(parts)))
    return "".join(parts)


def render_error(console: Console, error: ChatbotError) -> None:
    """Print an error and, when available, the advice to fix it.

    Args:
        console: Destination console (normally stderr).
        error: The application error to display.
    """
    console.print(f"[bold red]Error:[/] {escape(str(error))}")
    if error.hint:
        console.print(f"[dim]Hint: {escape(error.hint)}[/]")
