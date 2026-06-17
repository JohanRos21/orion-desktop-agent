from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class IntentType(str, Enum):
    CONVERSATION = "conversation"
    OPEN_APPLICATION = "open_application"
    UNKNOWN = "unknown"
    END_SESSION = "end_session"


class OllamaIntentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_text: str
    intent: IntentType
    application_name: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    assistant_reply: str | None = None

    @model_validator(mode="after")
    def validate_intent_contract(self) -> Self:
        if (
            self.intent is IntentType.OPEN_APPLICATION
            and not _has_text(self.application_name)
        ):
            raise ValueError(
                "open_application exige application_name."
            )

        if (
            self.intent is IntentType.UNKNOWN
            and not self.needs_clarification
        ):
            raise ValueError(
                "unknown debe solicitar aclaracion."
            )

        if (
            self.intent is IntentType.END_SESSION
            and self.application_name is not None
        ):
            raise ValueError(
                "end_session no acepta application_name."
            )

        if (
            self.needs_clarification
            and not _has_text(self.clarification_question)
        ):
            raise ValueError(
                "needs_clarification exige clarification_question."
            )

        return self


class IntentInterpretation(OllamaIntentPayload):
    original_text: str


@dataclass(frozen=True, slots=True)
class OllamaIntentPayloadResult:
    payload: OllamaIntentPayload
    duration_ms: float


@dataclass(frozen=True, slots=True)
class IntentParseResult:
    interpretation: IntentInterpretation
    duration_ms: float


def _has_text(value: str | None) -> bool:
    return value is not None and bool(value.strip())
