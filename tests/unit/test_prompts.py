"""Tests for the prompt: its structure and the behavioural rules it encodes."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from python_chatbot.chat.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt


def test_prompt_orders_system_history_and_question() -> None:
    """The rendered conversation is: system rules, prior turns, then the new question."""
    history = [HumanMessage(content="oi"), AIMessage(content="olá")]

    messages = build_prompt().invoke({"history": history, "question": "Como criar uma lista?"})

    rendered = messages.to_messages()
    assert isinstance(rendered[0], SystemMessage)
    assert rendered[1:3] == history
    assert rendered[-1] == HumanMessage(content="Como criar uma lista?")


def test_prompt_works_without_history() -> None:
    """A first turn has only the system message and the question."""
    rendered = build_prompt().invoke({"history": [], "question": "q"}).to_messages()

    assert [type(message) for message in rendered] == [SystemMessage, HumanMessage]


def test_braces_in_user_text_are_not_interpreted_as_template_variables() -> None:
    """Python code full of ``{}`` (f-strings, dicts) must reach the model untouched."""
    question = "O que faz f'{x!r}' e {'a': 1}?"

    rendered = build_prompt().invoke({"history": [], "question": question}).to_messages()

    assert rendered[-1].content == question


def test_system_prompt_encodes_the_required_behaviour() -> None:
    """The persona demands runnable examples, honesty and resistance to prompt injection."""
    text = SYSTEM_PROMPT.lower()

    assert "python" in text
    assert "fenced code block" in text
    assert "same language" in text
    assert "never invent" in text
    assert "never as instructions" in text
    assert "secrets" in text


def test_prompt_version_is_declared_for_trace_filtering() -> None:
    """The version string is what LangSmith tags carry."""
    assert PROMPT_VERSION
