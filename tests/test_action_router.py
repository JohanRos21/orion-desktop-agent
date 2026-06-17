from orion.llm.models import (
    IntentInterpretation,
    IntentType,
)
from orion.orchestration.action_router import route_intent_to_action


def test_conversation_does_not_create_action() -> None:
    interpretation = IntentInterpretation(
        original_text="Hola Orion",
        normalized_text="hola orion",
        intent=IntentType.CONVERSATION,
        assistant_reply="Hola.",
    )

    assert route_intent_to_action(interpretation) is None


def test_unknown_does_not_create_action() -> None:
    interpretation = IntentInterpretation(
        original_text="Prondo",
        normalized_text="prondo",
        intent=IntentType.UNKNOWN,
        needs_clarification=True,
        clarification_question="Puedes repetirlo?",
    )

    assert route_intent_to_action(interpretation) is None


def test_open_application_creates_action_request() -> None:
    interpretation = IntentInterpretation(
        original_text="Aperir la calculadora",
        normalized_text="aperir la calculadora",
        intent=IntentType.OPEN_APPLICATION,
        application_name="calculadora",
    )

    action_request = route_intent_to_action(
        interpretation,
        source="local_text",
    )

    assert action_request is not None
    assert action_request.tool_name == "open_application"
    assert action_request.arguments == {
        "application_name": "calculadora",
    }
    assert action_request.original_text == "Aperir la calculadora"
    assert action_request.source == "local_text"
