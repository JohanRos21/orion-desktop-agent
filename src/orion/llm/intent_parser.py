from __future__ import annotations

from orion.llm.models import (
    IntentInterpretation,
    IntentParseResult,
)
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

    return IntentParseResult(
        interpretation=IntentInterpretation(
            original_text=text,
            **result.payload.model_dump(),
        ),
        duration_ms=result.duration_ms,
    )
