import pytest
from pydantic import ValidationError

from orion.llm.models import (
    IntentInterpretation,
    IntentType,
)


def test_intent_interpretation_accepts_valid_response() -> None:
    interpretation = IntentInterpretation(
        original_text="Abre la calculadora",
        normalized_text="abre la calculadora",
        intent=IntentType.OPEN_APPLICATION,
        application_name="calculadora",
    )

    assert interpretation.original_text == "Abre la calculadora"
    assert interpretation.intent is IntentType.OPEN_APPLICATION


def test_intent_interpretation_rejects_invalid_enum() -> None:
    with pytest.raises(ValidationError):
        IntentInterpretation(
            original_text="Abre la calculadora",
            normalized_text="abre la calculadora",
            intent="open_calculator",
            application_name="calculadora",
        )


def test_open_application_requires_application_name() -> None:
    with pytest.raises(ValidationError):
        IntentInterpretation(
            original_text="Abre",
            normalized_text="abre",
            intent=IntentType.OPEN_APPLICATION,
        )


def test_clarification_requires_question() -> None:
    with pytest.raises(ValidationError):
        IntentInterpretation(
            original_text="Prondo",
            normalized_text="prondo",
            intent=IntentType.UNKNOWN,
            needs_clarification=True,
        )


def test_unknown_requires_clarification() -> None:
    with pytest.raises(ValidationError):
        IntentInterpretation(
            original_text="Prondo",
            normalized_text="prondo",
            intent=IntentType.UNKNOWN,
        )


def test_intent_interpretation_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        IntentInterpretation(
            original_text="Hola",
            normalized_text="hola",
            intent=IntentType.CONVERSATION,
            assistant_reply="Hola.",
            extra_field="not allowed",
        )
