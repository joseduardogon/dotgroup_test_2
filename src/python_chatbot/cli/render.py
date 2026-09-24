"""Console rendering helpers, kept separate so they can be tested with an in-memory console."""

from collections.abc import Iterable, Iterator

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.segment import Segment

from python_chatbot.core.exceptions import ChatbotError

_RESERVED_ROWS = 2


class _TailMarkdown:
    """Markdown view that only ever shows the last screenful of the text.

    ``rich.live.Live`` redraws its frame in place, which only works while the frame fits
    on the screen: once an answer is taller than the terminal the top rows scroll into
    the history where they can no longer be erased, and every refresh leaves another copy
    of them behind. Rendering just the tail keeps each frame shorter than the terminal, so
    nothing ever escapes into the scrollback while the answer is streaming.
    """

    def __init__(self, text: str) -> None:
        """Wrap the answer received so far.

        Args:
            text: The Markdown source accumulated up to now.
        """
        self._text = text

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the last rows of the Markdown that fit in the terminal.

        Args:
            console: The console being rendered to.
            options: Rendering options (the frame height limit is ignored on purpose).

        Yields:
            The segments of the visible rows, separated by newlines.
        """
        limit = max(console.size.height - _RESERVED_ROWS, 1)
        rows = console.render_lines(Markdown(self._text), options.update(height=None), pad=False)
        for index, row in enumerate(rows[-limit:]):
            if index:
                yield Segment.line()
            yield from row


def _stream_raw(console: Console, fragments: Iterable[str]) -> Iterator[str]:
    """Write each fragment verbatim as it arrives and pass it through.

    Args:
        console: Destination console.
        fragments: Successive pieces of the answer.

    Yields:
        The same fragments, after they were written.
    """
    for fragment in fragments:
        console.out(fragment, end="", highlight=False)
        yield fragment
    console.out("")


def render_stream(console: Console, fragments: Iterable[str], *, raw: bool = False) -> str:
    """Print an answer as it is generated and return the complete text.

    Raw mode writes the fragments verbatim, which is what scripts and pipes want.
    Rendered mode shows a live, transient view of the *end* of the answer while it
    streams (see :class:`_TailMarkdown`) and then prints the complete Markdown exactly
    once, with syntax-highlighted code, so the scrollback never contains duplicates. If
    the stream fails or is interrupted, whatever arrived so far is still printed.

    Args:
        console: Destination console.
        fragments: Successive pieces of the answer, e.g. from ``PythonAssistant.stream``.
        raw: Write plain text instead of rendering Markdown.

    Returns:
        The full answer.
    """
    if raw:
        return "".join(_stream_raw(console, fragments))

    text = ""
    try:
        with Live(
            _TailMarkdown(text), console=console, refresh_per_second=8, transient=True
        ) as live:
            for fragment in fragments:
                text += fragment
                live.update(_TailMarkdown(text))
    finally:
        if text:
            console.print(Markdown(text))
    return text


def render_error(console: Console, error: ChatbotError) -> None:
    """Print an error and, when available, the advice to fix it.

    Args:
        console: Destination console (normally stderr).
        error: The application error to display.
    """
    console.print(f"[bold red]Error:[/] {escape(str(error))}")
    if error.hint:
        console.print(f"[dim]Hint: {escape(error.hint)}[/]")
