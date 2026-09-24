"""Python programming assistant built on LangChain, OpenAI chat models and LangSmith.

The package is organised in small, single-purpose layers:

* ``core`` - settings, exceptions and observability wiring.
* ``llm`` - construction of the chat model (the only place that knows OpenAI).
* ``chat`` - prompt, conversation memory and the ``PythonAssistant`` that composes them
  into a LangChain Expression Language (LCEL) pipeline.
* ``cli`` - the terminal front-end (interactive REPL and one-shot questions).

Dependencies point inward: ``cli -> chat -> llm/core``. Nothing below ``cli`` prints or
reads from the terminal, which keeps the domain logic testable without a TTY or a network.
"""

from importlib.metadata import version

__version__ = version("dotgroup-test-2")

__all__ = ["__version__"]
