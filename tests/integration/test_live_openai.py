"""Opt-in smoke test against the real OpenAI API.

Skipped by default (``-m "not live"`` is part of the pytest options). Run it explicitly
with an API key in the environment; it makes one small, paid request::

    OPENAI_API_KEY=sk-... poetry run pytest -m live --no-cov
"""

import os

import pytest

from python_chatbot.chat.factory import build_assistant
from python_chatbot.core.config import Settings

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
def test_real_model_explains_how_to_create_a_list() -> None:
    """The exact example from the assignment produces a detailed, code-bearing answer."""
    assistant = build_assistant(Settings(max_tokens=600))

    answer = assistant.ask("Como criar uma lista em Python?")

    assert "```" in answer
    assert "[" in answer
    assert len(answer) > 200
