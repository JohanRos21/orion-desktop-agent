from __future__ import annotations

from orion.llm.models import (
    IntentInterpretation,
    IntentType,
)
from orion.tools.contracts import ActionRequest


def route_intent_to_action(
    interpretation: IntentInterpretation,
    source: str = "local_text",
) -> ActionRequest | None:
    if interpretation.intent in {
        IntentType.CONVERSATION,
        IntentType.UNKNOWN,
    }:
        return None

    if interpretation.intent is IntentType.OPEN_APPLICATION:
        application_name = interpretation.application_name

        if application_name is None or not application_name.strip():
            return None

        return ActionRequest(
            tool_name="open_application",
            arguments={
                "application_name": application_name,
            },
            original_text=interpretation.original_text,
            source=source,
        )

    return None
