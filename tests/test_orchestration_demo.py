from orion.llm.models import (
    IntentInterpretation,
    IntentType,
)
from orion.orchestration.action_router import route_intent_to_action
from orion.orchestration.demo import build_audit_record
from orion.policy.engine import PolicyEngine
from orion.policy.models import PolicyDecisionType
from orion.tools.default_registry import build_default_registry


def test_audit_record_is_built_for_allowed_action() -> None:
    interpretation = IntentInterpretation(
        original_text="Abre la calculadora",
        normalized_text="abre la calculadora",
        intent=IntentType.OPEN_APPLICATION,
        application_name="calculadora",
    )
    action_request = route_intent_to_action(
        interpretation
    )
    assert action_request is not None

    decision = PolicyEngine(
        registry=build_default_registry(),
    ).evaluate(
        action_request
    )

    audit_record = build_audit_record(
        interpretation=interpretation,
        action_request=action_request,
        policy_decision=decision,
        llm_ms=42.0,
    )

    assert audit_record.original_text == "Abre la calculadora"
    assert audit_record.intent == "open_application"
    assert audit_record.tool_name == "open_application"
    assert audit_record.arguments == {
        "application_name": "calculadora",
    }
    assert audit_record.source == "local_text"
    assert audit_record.risk == "low"
    assert audit_record.decision == PolicyDecisionType.ALLOW.value
    assert audit_record.llm_ms == 42.0
    assert audit_record.policy_ms >= 0


def test_audit_record_is_built_without_action() -> None:
    interpretation = IntentInterpretation(
        original_text="Hola Orion",
        normalized_text="hola orion",
        intent=IntentType.CONVERSATION,
        assistant_reply="Hola.",
    )

    audit_record = build_audit_record(
        interpretation=interpretation,
        action_request=None,
        policy_decision=None,
        llm_ms=12.0,
    )

    assert audit_record.tool_name is None
    assert audit_record.arguments == {}
    assert audit_record.decision is None
    assert audit_record.reason == "no se requiere herramienta"
    assert audit_record.policy_ms == 0.0
