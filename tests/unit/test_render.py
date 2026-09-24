"""Tests for console rendering helpers."""

import io
from collections.abc import Iterator

import pytest
from rich.console import Console
from rich.markdown import Markdown

from python_chatbot.cli.render import _TailMarkdown, render_error, render_stream
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


def _terminal(height: int) -> tuple[Console, io.StringIO]:
    """Create a fake interactive terminal of the given height writing to memory."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=60, height=height, force_terminal=True, color_system=None)
    return console, buffer


def _long_answer(lines: int) -> list[str]:
    """Build a Markdown answer made of many distinct paragraphs, one fragment each."""
    return [f"Paragraph number {index:03d}.\n\n" for index in range(lines)]


@pytest.mark.parametrize("paragraphs", [1, 3, 8, 40, 200])
def test_live_frame_is_always_shorter_than_the_terminal(paragraphs: int) -> None:
    """Regression: an in-place redraw can only erase a frame that fits on the screen.

    The previous renderer drew the whole answer on every refresh. Once it was taller than the
    terminal, the rows that scrolled off the top could not be erased and each refresh left
    another copy in the scrollback (a real ``pychat ask`` printed the first paragraph dozens of
    times). The tail view must therefore never exceed the terminal height, at any answer size.
    """
    console, _ = _terminal(height=10)
    text = "".join(_long_answer(paragraphs))

    frame_rows = len(console.render_lines(_TailMarkdown(text), pad=False))

    assert frame_rows < console.size.height


def test_the_premise_a_full_markdown_frame_would_overflow_the_terminal() -> None:
    """A full Markdown frame really is taller than the terminal, which is why the tail exists."""
    console, _ = _terminal(height=10)

    naive_rows = len(console.render_lines(Markdown("".join(_long_answer(40))), pad=False))

    assert naive_rows > console.size.height


def test_full_answer_is_printed_once_in_order_after_streaming() -> None:
    """The final output contains every paragraph, in order."""
    console, buffer = _terminal(height=10)

    render_stream(console, iter(_long_answer(30)))

    output = buffer.getvalue()
    positions = [output.rindex(f"Paragraph number {index:03d}.") for index in range(30)]
    assert positions == sorted(positions)


def test_partial_answer_is_printed_when_the_stream_fails() -> None:
    """Text that already arrived is not lost when the provider fails mid-answer."""
    console, buffer = _console()

    def failing() -> Iterator[str]:
        yield "recebido antes da falha"
        raise ChatbotError("boom")

    with pytest.raises(ChatbotError):
        render_stream(console, failing())

    assert "recebido antes da falha" in buffer.getvalue()


def test_tail_view_shows_only_the_last_rows_that_fit() -> None:
    """The live frame is shorter than the terminal and ends with the newest text."""
    console, buffer = _terminal(height=8)
    text = "".join(_long_answer(40))

    console.print(_TailMarkdown(text))

    rows = [row for row in buffer.getvalue().splitlines() if row.strip()]
    assert len(rows) <= 6
    assert "Paragraph number 039." in rows[-1]
    assert "Paragraph number 000." not in buffer.getvalue()
