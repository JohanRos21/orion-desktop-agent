from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class IntentType(str, Enum):
    CONVERSATION = "conversation"
    OPEN_APPLICATION = "open_application"
    UNKNOWN = "unknown"


class IntentInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_text: str
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
            self.needs_clarification
            and not _has_text(self.clarification_question)
        ):
            raise ValueError(
                "needs_clarification exige clarification_question."
            )

        return self


@dataclass(frozen=True, slots=True)
class IntentParseResult:
    interpretation: IntentInterpretation
    duration_ms: float


def _has_text(value: str | None) -> bool:
    return value is not None and bool(value.strip())
