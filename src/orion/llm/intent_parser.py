from __future__ import annotations

from orion.llm.exceptions import OllamaInvalidResponseError
from orion.llm.models import IntentParseResult
from orion.llm.ollama_client import OllamaClient
from orion.llm.prompts import build_messages


def interpret_intent(
    text: str,
    client: OllamaClient | None = None,
) -> IntentParseResult:
    ollama_client = client or OllamaClient()

    result = ollama_client.interpret_messages(
        messages=build_messages(
            text=text,
        ),
    )

    if result.interpretation.original_text != text:
        raise OllamaInvalidResponseError(
            "Ollama no preservo original_text exactamente."
        )

    return result
