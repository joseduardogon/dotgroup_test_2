"""Opt-in smoke test against the real OpenAI API.

Skipped by default (``-m "not live"`` is part of the pytest options). Put your key in the
environment or in a local ``.env`` file (never in the repository) and run it explicitly;
it makes one small, paid request::

    poetry run pytest -m live --no-cov -rs

Without a key the test is reported as *skipped* with the reason, not as a failure.
"""

import pytest

from python_chatbot.chat.factory import build_assistant
from python_chatbot.core.config import Settings

pytestmark = pytest.mark.live


def test_real_model_explains_how_to_create_a_list() -> None:
    """The exact example from the assignment produces a detailed, code-bearing answer."""
    settings = Settings(max_tokens=600)
    if settings.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY is not set (environment or .env)")

    answer = build_assistant(settings).ask("Como criar uma lista em Python?")

    assert "```" in answer
    assert "[" in answer
    assert len(answer) > 200
