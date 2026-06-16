from __future__ import annotations

import json
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from orion.config import settings
from orion.llm.exceptions import (
    OllamaConfigurationError,
    OllamaConnectionError,
    OllamaError,
    OllamaInvalidResponseError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from orion.llm.models import (
    IntentInterpretation,
    IntentParseResult,
)


LOCAL_OLLAMA_HOSTS = {
    "127.0.0.1",
    "localhost",
    "::1",
}


def validate_local_ollama_url(
    base_url: str,
) -> str:
    parsed_url = urlparse(
        base_url.strip()
    )

    if parsed_url.scheme not in {"http", "https"}:
        raise OllamaConfigurationError(
            "OLLAMA_BASE_URL debe usar http o https."
        )

    if not parsed_url.netloc:
        raise OllamaConfigurationError(
            "OLLAMA_BASE_URL debe incluir host y puerto."
        )

    host = parsed_url.hostname

    if host not in LOCAL_OLLAMA_HOSTS:
        raise OllamaConfigurationError(
            "OLLAMA_BASE_URL debe apuntar a un host local "
            "(127.0.0.1, localhost o ::1)."
        )

    return base_url.rstrip("/")


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
        timeout_seconds: float = settings.OLLAMA_TIMEOUT_SECONDS,
        keep_alive: str = settings.OLLAMA_KEEP_ALIVE,
    ) -> None:
        self.base_url = validate_local_ollama_url(
            base_url
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    def interpret_messages(
        self,
        messages: list[dict[str, str]],
    ) -> IntentParseResult:
        payload = self._build_payload(
            messages=messages,
        )

        started_at = perf_counter()

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
            ) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
        except httpx.TimeoutException as error:
            raise OllamaTimeoutError(
                "Ollama no respondio dentro del timeout "
                f"de {self.timeout_seconds:.1f} segundos."
            ) from error
        except httpx.ConnectError as error:
            raise OllamaConnectionError(
                "No se pudo conectar con Ollama local. "
                "Verifica que el servidor este encendido."
            ) from error
        except httpx.RequestError as error:
            raise OllamaConnectionError(
                f"No se pudo consultar Ollama local: {error}"
            ) from error

        duration_ms = round(
            (perf_counter() - started_at) * 1000,
            2,
        )

        if response.status_code >= 400:
            self._raise_for_error_response(
                response=response,
            )

        interpretation = self._parse_response(
            response=response,
        )

        return IntentParseResult(
            interpretation=interpretation,
            duration_ms=duration_ms,
        )

    def _build_payload(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "format": IntentInterpretation.model_json_schema(),
            "messages": messages,
            "options": {
                "temperature": 0,
            },
        }

    def _parse_response(
        self,
        response: httpx.Response,
    ) -> IntentInterpretation:
        try:
            response_payload = response.json()
        except ValueError as error:
            raise OllamaInvalidResponseError(
                "Ollama devolvio JSON invalido."
            ) from error

        content = _extract_message_content(
            response_payload
        )

        try:
            if isinstance(content, str):
                return IntentInterpretation.model_validate_json(
                    content
                )

            return IntentInterpretation.model_validate(
                content
            )
        except (ValidationError, ValueError, TypeError) as error:
            raise OllamaInvalidResponseError(
                "La interpretacion de Ollama no cumple el schema."
            ) from error

    def _raise_for_error_response(
        self,
        response: httpx.Response,
    ) -> None:
        message = _extract_error_message(
            response
        )

        if (
            response.status_code == 404
            or (
                "model" in message.casefold()
                and "not found" in message.casefold()
            )
            or "not found" in message.casefold()
        ):
            raise OllamaModelNotFoundError(
                "El modelo de Ollama no existe o no esta "
                f"disponible: {self.model}."
            )

        raise OllamaError(
            f"Ollama respondio con error HTTP "
            f"{response.status_code}: {message}"
        )


def _extract_message_content(
    payload: dict[str, Any],
) -> Any:
    message = payload.get("message")

    if isinstance(message, dict) and "content" in message:
        return message["content"]

    if "response" in payload:
        return payload["response"]

    raise OllamaInvalidResponseError(
        "La respuesta de Ollama no contiene message.content."
    )


def _extract_error_message(
    response: httpx.Response,
) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text

    error = payload.get("error")

    if isinstance(error, str):
        return error

    return json.dumps(
        payload,
        ensure_ascii=False,
    )
