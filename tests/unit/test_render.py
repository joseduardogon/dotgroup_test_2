"""Tests for console rendering helpers."""

import io

from rich.console import Console

from python_chatbot.cli.render import render_error, render_stream
from python_chatbot.core.exceptions import ChatbotError, ConfigurationError


def _console() -> tuple[Console, io.StringIO]:
    """Create a plain, wide, colourless console writing to memory."""
    buffer = io.StringIO()
    return Console(file=buffer, width=100, color_system=None, force_terminal=False), buffer


def test_raw_stream_writes_fragments_verbatim_and_returns_the_text() -> None:
    """Raw mode preserves Markdown syntax exactly, ready for pipes."""
    console, buffer = _console()

    text = render_stream(console, iter(["Use ", "`[]`", " ou list()"]), raw=True)

    assert text == "Use `[]` ou list()"
    assert buffer.getvalue() == "Use `[]` ou list()\n"


def test_rendered_stream_returns_the_text_and_shows_the_content() -> None:
    """Rendered mode converts Markdown but still returns the original source text."""
    console, buffer = _console()

    text = render_stream(console, iter(["# Listas\n\n", "Use `[]`."]))

    assert text == "# Listas\n\nUse `[]`."
    assert "Listas" in buffer.getvalue()
    assert "Use [] ." not in buffer.getvalue()


def test_rendered_stream_shows_code_blocks() -> None:
    """Fenced code reaches the terminal."""
    console, buffer = _console()

    render_stream(console, iter(["```python\n", "nums = [1, 2]\n", "```\n"]))

    assert "nums = [1, 2]" in buffer.getvalue()


def test_error_is_printed_with_its_hint() -> None:
    """The user sees what failed and what to do about it."""
    console, buffer = _console()

    render_error(console, ConfigurationError("no key", hint="export it"))

    output = buffer.getvalue()
    assert "Error: no key" in output
    assert "Hint: export it" in output


def test_error_without_hint_prints_a_single_line() -> None:
    """No empty hint line is printed."""
    console, buffer = _console()

    render_error(console, ChatbotError("boom"))

    assert buffer.getvalue().strip() == "Error: boom"


def test_rich_markup_in_error_text_is_escaped() -> None:
    """Text such as ``[red]`` from a provider message is shown literally, not interpreted."""
    console, buffer = _console()

    render_error(console, ChatbotError("bad [red]tag[/red]"))

    assert "bad [red]tag[/red]" in buffer.getvalue()
