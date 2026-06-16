from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from orion.llm.exceptions import (
    OllamaConfigurationError,
    OllamaConnectionError,
    OllamaInvalidResponseError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from orion.llm.intent_parser import interpret_intent
from orion.llm.models import (
    IntentInterpretation,
    IntentParseResult,
    IntentType,
)
from orion.llm import ollama_client
from orion.llm.ollama_client import (
    OllamaClient,
    validate_local_ollama_url,
)


def test_valid_local_ollama_url_is_accepted() -> None:
    assert (
        validate_local_ollama_url(
            "http://127.0.0.1:11434/"
        )
        == "http://127.0.0.1:11434"
    )
    assert (
        validate_local_ollama_url(
            "http://localhost:11434"
        )
        == "http://localhost:11434"
    )
    assert (
        validate_local_ollama_url(
            "http://[::1]:11434"
        )
        == "http://[::1]:11434"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0:11434",
        "http://192.168.1.10:11434",
        "https://ollama.example.com",
    ],
)
def test_remote_ollama_url_is_rejected(
    url: str,
) -> None:
    with pytest.raises(OllamaConfigurationError):
        validate_local_ollama_url(url)


def test_ollama_client_parses_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            _response(
                _chat_payload(
                    {
                        "original_text": "Abre la calculadora",
                        "normalized_text": "abre la calculadora",
                        "intent": "open_application",
                        "application_name": "calculadora",
                    }
                )
            )
        ),
    )

    result = OllamaClient().interpret_messages(
        [{"role": "user", "content": "Abre la calculadora"}]
    )

    assert result.interpretation.intent is IntentType.OPEN_APPLICATION
    assert result.interpretation.application_name == "calculadora"
    assert result.duration_ms >= 0


def test_ollama_client_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            _response(
                {
                    "message": {
                        "content": "not-json"
                    }
                }
            )
        ),
    )

    with pytest.raises(OllamaInvalidResponseError):
        OllamaClient().interpret_messages([])


def test_ollama_client_rejects_invalid_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            _response(
                _chat_payload(
                    {
                        "original_text": "Abre la calculadora",
                        "normalized_text": "abre la calculadora",
                        "intent": "open_calculator",
                        "application_name": "calculadora",
                    }
                )
            )
        ),
    )

    with pytest.raises(OllamaInvalidResponseError):
        OllamaClient().interpret_messages([])


def test_ollama_client_rejects_open_application_without_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            _response(
                _chat_payload(
                    {
                        "original_text": "Abre",
                        "normalized_text": "abre",
                        "intent": "open_application",
                    }
                )
            )
        ),
    )

    with pytest.raises(OllamaInvalidResponseError):
        OllamaClient().interpret_messages([])


def test_ollama_client_rejects_clarification_without_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            _response(
                _chat_payload(
                    {
                        "original_text": "Prondo",
                        "normalized_text": "prondo",
                        "intent": "unknown",
                        "needs_clarification": True,
                    }
                )
            )
        ),
    )

    with pytest.raises(OllamaInvalidResponseError):
        OllamaClient().interpret_messages([])


def test_ollama_client_handles_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            httpx.TimeoutException("timeout")
        ),
    )

    with pytest.raises(OllamaTimeoutError):
        OllamaClient().interpret_messages([])


def test_ollama_client_handles_connection_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            httpx.ConnectError("refused")
        ),
    )

    with pytest.raises(OllamaConnectionError):
        OllamaClient().interpret_messages([])


def test_ollama_client_handles_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            _response(
                {
                    "error": "model qwen3:8b not found"
                },
                status_code=404,
            )
        ),
    )

    with pytest.raises(OllamaModelNotFoundError):
        OllamaClient().interpret_messages([])


def test_ollama_client_preserves_original_text() -> None:
    class FakeClient:
        def interpret_messages(
            self,
            messages: list[dict[str, str]],
        ) -> IntentParseResult:
            return IntentParseResult(
                interpretation=IntentInterpretation(
                    original_text="Abre la calculadora",
                    normalized_text="abre la calculadora",
                    intent=IntentType.OPEN_APPLICATION,
                    application_name="calculadora",
                ),
                duration_ms=12.0,
            )

    result = interpret_intent(
        "Abre la calculadora",
        client=FakeClient(),
    )

    assert result.interpretation.original_text == "Abre la calculadora"


def test_ollama_client_rejects_modified_original_text() -> None:
    class FakeClient:
        def interpret_messages(
            self,
            messages: list[dict[str, str]],
        ) -> IntentParseResult:
            return IntentParseResult(
                interpretation=IntentInterpretation(
                    original_text="abre la calculadora",
                    normalized_text="abre la calculadora",
                    intent=IntentType.OPEN_APPLICATION,
                    application_name="calculadora",
                ),
                duration_ms=12.0,
            )

    with pytest.raises(OllamaInvalidResponseError):
        interpret_intent(
            "Abre la calculadora",
            client=FakeClient(),
        )


def test_ollama_client_measures_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([10.0, 10.125])

    monkeypatch.setattr(
        ollama_client,
        "perf_counter",
        lambda: next(times),
    )
    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            _response(
                _chat_payload(
                    {
                        "original_text": "Hola Orion",
                        "normalized_text": "hola orion",
                        "intent": "conversation",
                        "assistant_reply": "Hola.",
                    }
                )
            )
        ),
    )

    result = OllamaClient().interpret_messages([])

    assert result.duration_ms == 125.0


def test_ollama_client_sends_required_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[dict[str, Any]] = []

    monkeypatch.setattr(
        ollama_client.httpx,
        "Client",
        _client_factory(
            _response(
                _chat_payload(
                    {
                        "original_text": "Hola Orion",
                        "normalized_text": "hola orion",
                        "intent": "conversation",
                        "assistant_reply": "Hola.",
                    }
                )
            ),
            captured_requests=captured_requests,
        ),
    )

    OllamaClient().interpret_messages(
        [{"role": "user", "content": "Hola Orion"}]
    )

    request = captured_requests[0]
    payload = request["json"]

    assert request["url"] == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "qwen3:8b"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["keep_alive"] == "10m"
    assert payload["options"] == {"temperature": 0}
    assert isinstance(payload["format"], dict)
    assert payload["format"]["title"] == "IntentInterpretation"


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(
        self,
        result: _FakeResponse | Exception,
        captured_requests: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.result = result
        self.captured_requests = captured_requests
        self.kwargs = kwargs

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None

    def post(
        self,
        url: str,
        json: dict[str, Any],
    ) -> _FakeResponse:
        if self.captured_requests is not None:
            self.captured_requests.append(
                {
                    "url": url,
                    "json": json,
                    "client_kwargs": self.kwargs,
                }
            )

        if isinstance(self.result, Exception):
            raise self.result

        return self.result


def _client_factory(
    result: _FakeResponse | Exception,
    captured_requests: list[dict[str, Any]] | None = None,
) -> type[_FakeClient]:
    class BoundFakeClient(_FakeClient):
        def __init__(
            self,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                result=result,
                captured_requests=captured_requests,
                **kwargs,
            )

    return BoundFakeClient


def _response(
    payload: dict[str, Any],
    status_code: int = 200,
) -> _FakeResponse:
    return _FakeResponse(
        payload=payload,
        status_code=status_code,
    )


def _chat_payload(
    content: dict[str, Any],
) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps(content),
        }
    }
