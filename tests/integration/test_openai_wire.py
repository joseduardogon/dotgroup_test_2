"""Wire-level integration tests against a local, OpenAI-compatible HTTP server.

These tests run the *real* ``langchain-openai`` client and the real OpenAI SDK over
HTTP, but against a tiny server on ``127.0.0.1`` that imitates ``/v1/chat/completions``.
They prove what a mocked chat model cannot: the request we send is well formed, the
streaming (SSE) response is parsed, and genuine HTTP failures (401, 429, 500) are turned
into the application's errors. No API key, no internet and no cost.
"""

import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from pydantic import SecretStr

from python_chatbot.chat.assistant import TRUNCATION_NOTICE
from python_chatbot.chat.factory import build_assistant
from python_chatbot.core.config import Settings
from python_chatbot.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMUnavailableError,
)

pytestmark = pytest.mark.integration

ANSWER = ["Uma ", "lista ", "usa ", "colchetes."]


@dataclass
class FakeOpenAI:
    """Controls and records the behaviour of the local server.

    Attributes:
        status: HTTP status returned to the next request(s).
        finish_reason: Finish reason reported for successful completions.
        requests: Decoded JSON bodies of chat completion requests, in order.
        headers: Request headers of chat completion requests, in order.
        paths: ``(method, path)`` of every request the server saw, in order.
    """

    status: int = 200
    finish_reason: str = "stop"
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    paths: list[tuple[str, str]] = field(default_factory=list)
    base_url: str = ""


def _handler_for(state: FakeOpenAI) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to ``state``."""

    class Handler(BaseHTTPRequestHandler):
        """Answers POST /v1/chat/completions like the OpenAI API."""

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the default stderr access log."""

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            """Write a JSON response."""
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            """Answer any GET (e.g. LangSmith's ``/info`` probe) with an empty object."""
            state.paths.append(("GET", self.path))
            self._send_json(200, {})

        def do_POST(self) -> None:
            """Handle a chat completion request; acknowledge any other POST (LangSmith runs)."""
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            state.paths.append(("POST", self.path))
            if not self.path.endswith("/chat/completions"):
                self._send_json(200, {})
                return
            request = json.loads(raw)
            state.requests.append(request)
            state.headers.append({k.lower(): v for k, v in self.headers.items()})
            if state.status != 200:
                error = {"message": f"simulated {state.status}", "type": "simulated", "code": None}
                self._send_json(state.status, {"error": error})
                return
            if request.get("stream"):
                self._stream()
            else:
                message = {"role": "assistant", "content": "".join(ANSWER)}
                self._send_json(
                    200,
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": request["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": state.finish_reason,
                            }
                        ],
                    },
                )

        def _stream(self) -> None:
            """Write the answer as server-sent events, one word per chunk."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta in [*({"content": word} for word in ANSWER), {}]:
                finished = not delta
                chunk = {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": state.finish_reason if finished else None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    return Handler


@pytest.fixture
def server() -> Iterator[FakeOpenAI]:
    """Run the fake OpenAI server for one test."""
    state = FakeOpenAI()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    state.base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield state
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _settings(server: FakeOpenAI, **overrides: Any) -> Settings:
    """Build settings that point the real client at the fake server without retries."""
    return Settings(
        openai_api_key=SecretStr("sk-wire-test"),
        openai_base_url=server.base_url,
        max_retries=0,
        _env_file=None,
        **overrides,
    )


def test_streamed_answer_is_assembled_from_server_sent_events(server: FakeOpenAI) -> None:
    """The real client streams tokens and the assistant returns them in order."""
    assistant = build_assistant(_settings(server))

    fragments = list(assistant.stream("Como criar uma lista em Python?"))

    assert "".join(fragments) == "".join(ANSWER)
    assert len(fragments) >= len(ANSWER)


def test_request_carries_model_temperature_prompt_and_credentials(server: FakeOpenAI) -> None:
    """The HTTP request is exactly what the OpenAI API expects."""
    assistant = build_assistant(_settings(server, model="gpt-4o-mini", temperature=0.1))

    assistant.ask("Como criar uma lista em Python?")

    body = server.requests[0]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.1
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "PyMentor" in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "Como criar uma lista em Python?"
    assert server.headers[0]["authorization"] == "Bearer sk-wire-test"


def test_second_turn_sends_the_first_exchange_over_the_wire(server: FakeOpenAI) -> None:
    """Conversation memory is visible in the actual HTTP payload."""
    assistant = build_assistant(_settings(server))

    assistant.ask("primeira")
    assistant.ask("segunda")

    roles = [m["role"] for m in server.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (500, LLMUnavailableError),
    ],
)
def test_real_http_failures_become_application_errors(
    server: FakeOpenAI, status: int, expected: type[Exception]
) -> None:
    """HTTP error statuses travel through the OpenAI SDK and are translated."""
    server.status = status
    assistant = build_assistant(_settings(server))

    with pytest.raises(expected):
        assistant.ask("qualquer coisa")


TRACING_SCRIPT = """
import sys
from langchain_core.tracers.langchain import wait_for_all_tracers
from pydantic import SecretStr
from python_chatbot.chat.assistant import TRUNCATION_NOTICE
from python_chatbot.chat.factory import build_assistant
from python_chatbot.core.config import Settings

openai_url, langsmith_url, tracing = sys.argv[1:4]
settings = Settings(
    openai_api_key=SecretStr("sk-wire-test"),
    openai_base_url=openai_url,
    max_retries=0,
    langsmith_tracing=tracing == "on",
    langsmith_api_key=SecretStr("lsv2-wire-test"),
    langsmith_endpoint=langsmith_url,
    _env_file=None,
)
build_assistant(settings).ask("Como criar uma lista em Python?")
wait_for_all_tracers()
"""


@pytest.mark.parametrize(("tracing", "expect_runs"), [("on", True), ("off", False)])
def test_traces_are_sent_only_when_tracing_is_enabled(
    server: FakeOpenAI, tracing: str, expect_runs: bool
) -> None:
    """A trace leaves the process if and only if tracing is switched on in our settings.

    Runs in a fresh interpreter, exactly like the CLI: the LangSmith SDK caches
    environment lookups, so this proves that exporting the variables at start-up (after
    the modules were imported) is early enough for traces to be emitted, and the ``off``
    case is the negative control that keeps the test honest. The local server plays both
    OpenAI and LangSmith.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            TRACING_SCRIPT,
            server.base_url,
            server.base_url.removesuffix("/v1"),
            tracing,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    runs = [path for method, path in server.paths if method == "POST" and "/runs" in path]
    assert bool(runs) is expect_runs, server.paths


def test_length_finish_reason_from_the_real_client_produces_the_notice(server: FakeOpenAI) -> None:
    """The real client surfaces ``finish_reason`` so a cut answer is flagged, not silent."""
    server.finish_reason = "length"
    assistant = build_assistant(_settings(server))

    answer = assistant.ask("Como criar uma lista em Python?")
    streamed = "".join(assistant.stream("de novo"))

    assert answer.endswith(TRUNCATION_NOTICE)
    assert streamed.endswith(TRUNCATION_NOTICE)
